#!/usr/bin/env python3
"""
Script 78 — HydroST: Spatio-Temporal Hydrological Transfer-Learning Model.

Architecture (custom, designed for Chancay-Huaral):
  1. Static encoder  : MLP(n_static -> hid) — per-entity physical attributes
  2. Entity-aware    : concat(dyn, static_emb) projected to hid per time step
  3. Shared LSTM     : temporal dynamics, weights shared across 9 sub-basins
  4. Spatial MHA     : cross-entity attention at final hidden step
  5. Pretrain head   : Linear(hid->1)  — MSE on GR4J outlet Q (Phase 1)
  6. Quantile head   : Linear->SiLU->Linear(3) with softplus no-crossing  (Phase 2)

Transfer learning:
  Phase 1  1981-2020 : pretrain on GR4J-corrected outlet Q (MSE, all 9 entity inputs)
  Phase 2  2021-2022 : finetune on observed Q at outlet (PinballLoss, w=5 for Q>Q90)

Sealed test: 2024-2025 is NEVER used for any model decision.

Run in .venv313:
    python scripts/05_models/78_hydrost_Q.py [--skip_pretrain]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
LOG_DIR = ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "78_hydrost_Q.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("hydrost")

# ── Constants ────────────────────────────────────────────────────────────────

ENTITIES   = ["sub_634","sub_640","sub_641","sub_646","sub_649","sub_650","sub_653","sub_655","sub_656"]
OUTLET     = "sub_634"
OUTLET_IDX = 0

Q_CONV    = 86.4 / 3062.62   # mm/day -> m3/s (total basin 3062.62 km2)
Q90_MM    = 40.89 * Q_CONV   # ~1.154 mm/day — alert threshold
Q99_MM    = 77.72 * Q_CONV

QUANTILES = [0.1, 0.5, 0.9]

# DATA SPLITS  ─  SEALED TEST: never touch 2024+ for model decisions
PRETRAIN_END   = pd.Timestamp("2020-12-31")
FT_START       = pd.Timestamp("2021-01-01")
FT_END         = pd.Timestamp("2022-12-31")
VAL_START      = pd.Timestamp("2023-01-01")
VAL_END        = pd.Timestamp("2023-12-31")
TEST_START     = pd.Timestamp("2024-01-01")   # SEALED

# Sequence parameters
ENC    = 60     # lookback days

# Architecture
HID    = 64
HEADS  = 4
LAYERS = 2
DROP   = 0.25

# Training
BATCH1  = 256
BATCH2  = 64
EP1     = 50
EP2     = 150
LR1     = 1e-3
LR2     = 1e-4
WD      = 1e-4
PAT     = 25    # early stopping patience
ALERT_W = 5.0   # weight for Q>Q90 samples in Phase 2
ALPHA   = 0.7   # PinballLoss weight; (1-ALPHA) = MSE_P50 weight

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamic features per entity (no future info, per calendar day)
# q_mm = same for all entities (outlet autorregressive Q); pr_mm etc. spatially distributed
DYN_COLS = [
    "pr_mm", "pet_mm", "tmean_c",
    "api", "spi_30d", "water_deficit_30d",
    "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
    "oni_index",
    "q_mm",          # outlet autoregressive Q (GR4J 1981-2020, obs 2021+)
    "ndvi_fill",     # MODIS NDVI forward-filled 16d, then DOY clim
    "snow_fill",     # MODIS snow forward-filled 16d, then 0
    "lsat_ndvi_fill",  # Landsat NDVI forward-filled 32d, then DOY clim
    "lsat_mndwi_fill", # Landsat MNDWI forward-filled 32d, then DOY clim
]
N_DYN = len(DYN_COLS)   # 16

STATIC_COLS = [
    "lat", "lon", "elevation_m", "basin_area_km2", "slope_deg",
    "hyps_integral", "soil_sand_pct", "soil_silt_pct", "soil_clay_pct",
    "soil_bdod", "soil_soc", "soil_cec", "soil_ph",
]
N_STATIC = len(STATIC_COLS)   # 13


# ── Satellite forward-fill ────────────────────────────────────────────────────

def _doy_clim(series: pd.Series, train_mask: pd.Series) -> dict:
    """Compute DOY climatological median on training period only."""
    train_vals = series[train_mask]
    return train_vals.groupby(train_vals.index.dayofyear).median().to_dict()

def fill_satellite(df: pd.DataFrame, train_mask_col: str = "is_train") -> pd.DataFrame:
    """
    For each entity, forward-fill satellite columns within sensor revisit window,
    then fall back to DOY climatological median (train only).  Never impute with 0
    for vegetative indices.

    Policies (per sensor):
      MODIS NDVI/LSWI/EVI: forward-fill up to 16 days (MOD13A1 revisit)
      MODIS snow:          forward-fill up to 8 days; remaining -> 0 (no-snow default)
      Landsat NDVI/MNDWI:  forward-fill up to 32 days (L5+L8/9 combined revisit)
    """
    log.info("Applying satellite forward-fill per entity...")
    df = df.copy()
    train_mask_global = df["date"] <= PRETRAIN_END

    for eid in ENTITIES:
        mask = df["entity_id"] == eid
        idx  = df.index[mask]
        dates = df.loc[idx, "date"]

        for col, max_gap, zero_fallback in [
            ("ndvi_mean",   16, False),
            ("snow_cover_pct", 8,  True),
            ("lsat_ndvi",   32, False),
            ("lsat_mndwi",  32, False),
        ]:
            s = df.loc[idx, col].copy()
            s.index = dates
            s = s.sort_index()

            # Forward-fill within sensor revisit window
            s_ff = s.ffill(limit=max_gap)

            if zero_fallback:
                s_ff = s_ff.fillna(0.0)
            else:
                # DOY climatological median from train period
                train_mask = dates[dates <= PRETRAIN_END].index
                clim = _doy_clim(s, s.index.isin(train_mask))
                remaining = s_ff[s_ff.isna()].index
                for d in remaining:
                    doy = d.dayofyear
                    s_ff.loc[d] = clim.get(doy, s.median())

            # Write back using new column names
            new_col = col.replace("ndvi_mean", "ndvi_fill").replace("snow_cover_pct", "snow_fill") \
                         .replace("lsat_ndvi", "lsat_ndvi_fill").replace("lsat_mndwi", "lsat_mndwi_fill")
            df.loc[idx, new_col] = s_ff.values

    log.info("Satellite fill complete.")
    return df


# ── Data loading and preprocessing ───────────────────────────────────────────

def load_data():
    log.info("Loading D7...")
    df = pd.read_csv(ROOT / "data/model_ready/D7_multientity.csv", parse_dates=["date"])
    df = df.sort_values(["entity_id", "date"]).reset_index(drop=True)
    log.info(f"  {len(df):,} rows x {df.shape[1]} cols")

    # Add satellite filled columns
    for col in ["ndvi_fill", "snow_fill", "lsat_ndvi_fill", "lsat_mndwi_fill"]:
        df[col] = np.nan
    df = fill_satellite(df)

    return df


def build_arrays(df: pd.DataFrame):
    """
    Build numpy arrays:
      dyn   (n_days, n_entities, N_DYN)   — dynamic features
      static(n_entities, N_STATIC)         — static attributes
      target(n_days,)                      — q_next_1d at outlet (mm/day)
      dates (n_days,)                      — datetime index
      scaler_dyn (mean, std)               — fit on train only
    """
    # Sort by date for consistent indexing
    entities_order = ENTITIES
    dates = df[df["entity_id"] == OUTLET]["date"].sort_values().values
    n_days = len(dates)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    # Static: one row per entity
    static_arr = np.zeros((len(entities_order), N_STATIC), dtype=np.float32)
    for i, eid in enumerate(entities_order):
        row = df[df["entity_id"] == eid][STATIC_COLS].iloc[0].values
        static_arr[i] = row.astype(np.float32)

    # Dynamic: pivot each feature to (n_days, n_entities)
    log.info("Pivoting dynamic features...")
    dyn_arr = np.zeros((n_days, len(entities_order), N_DYN), dtype=np.float32)
    df_indexed = df.set_index(["date", "entity_id"])

    for fi, col in enumerate(DYN_COLS):
        pivoted = df.pivot(index="date", columns="entity_id", values=col)
        pivoted = pivoted.loc[[pd.Timestamp(d) for d in dates], entities_order]
        dyn_arr[:, :, fi] = pivoted.values.astype(np.float32)

    # Target: q_next_1d at outlet (mm/day)
    outlet_df = df[df["entity_id"] == OUTLET].set_index("date").sort_index()
    target = outlet_df.loc[[pd.Timestamp(d) for d in dates], "q_next_1d"].values.astype(np.float32)

    # Scaler: fit on pretrain (1981-2020) only
    train_mask = np.array([pd.Timestamp(d) <= PRETRAIN_END for d in dates])
    dyn_mean = np.nanmean(dyn_arr[train_mask], axis=(0, 1), keepdims=False)  # (N_DYN,)
    dyn_std  = np.nanstd(dyn_arr[train_mask], axis=(0, 1), keepdims=False)
    dyn_std[dyn_std == 0] = 1.0

    # Also scale static
    static_mean = static_arr.mean(axis=0)
    static_std  = static_arr.std(axis=0)
    static_std[static_std == 0] = 1.0

    # Apply scaling
    dyn_arr = (dyn_arr - dyn_mean[None, None, :]) / dyn_std[None, None, :]
    static_arr = (static_arr - static_mean[None, :]) / static_std[None, :]

    # Fill any remaining NaN with 0 after scaling
    dyn_arr = np.nan_to_num(dyn_arr, nan=0.0)
    static_arr = np.nan_to_num(static_arr, nan=0.0)

    scalers = {
        "dyn_mean": dyn_mean, "dyn_std": dyn_std,
        "static_mean": static_mean, "static_std": static_std,
        "target_std": np.nanstd(target[train_mask]),
    }
    log.info(f"Arrays: dyn={dyn_arr.shape}, static={static_arr.shape}, target={target.shape}")
    log.info(f"NaN in dyn after scale: {np.isnan(dyn_arr).sum()}")
    return dyn_arr, static_arr, target, np.array(dates), scalers


# ── Dataset ──────────────────────────────────────────────────────────────────

class HydroDataset(Dataset):
    """Sequence dataset for HydroST.

    Each item: window [t-ENC:t] of all 9 entities -> q_next_1d target at t.
    Phase 1 target: scalar q_next_1d (mm/day, GR4J 1981-2020).
    Phase 2 target: same but obs Q + sample weight for Q>Q90.
    """

    def __init__(self, dyn, static, target, dates, target_dates):
        self.dyn    = dyn      # (n_days, n_ent, N_DYN)
        self.static = static   # (n_ent, N_STATIC)
        self.target = target   # (n_days,)
        self.date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

        self.indices = []
        for d in target_dates:
            t = pd.Timestamp(d)
            idx = self.date_to_idx.get(t)
            if idx is None or idx < ENC:
                continue
            tgt = target[idx]
            if np.isnan(tgt):
                continue
            self.indices.append((idx, float(tgt)))

        log.info(f"  Dataset: {len(self.indices)} sequences in target period")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx, tgt = self.indices[i]
        # x_dyn: (n_ent, ENC, N_DYN)
        x_dyn    = torch.tensor(self.dyn[idx - ENC:idx].transpose(1, 0, 2), dtype=torch.float32)
        x_static = torch.tensor(self.static, dtype=torch.float32)
        y        = torch.tensor(tgt, dtype=torch.float32)
        return x_dyn, x_static, y


# ── Model ─────────────────────────────────────────────────────────────────────

class StaticEncoder(nn.Module):
    def __init__(self, n_in: int, hid: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in, hid), nn.SiLU(),
            nn.Linear(hid, hid),
        )

    def forward(self, x):   # (B, N, n_static)
        return self.mlp(x)  # (B, N, hid)


class HydroST(nn.Module):
    """
    Spatio-Temporal model for outlet streamflow prediction.

    Processes 9 sub-basins jointly:
      - Static encoder produces entity-aware embeddings
      - Shared LSTM (entity-aware via static concat) captures temporal dynamics
      - Spatial multi-head attention aggregates cross-entity information
      - Quantile head outputs P10/P50/P90 for outlet
    """

    def __init__(
        self,
        n_dyn:    int = N_DYN,
        n_static: int = N_STATIC,
        n_ent:    int = 9,
        hid:      int = HID,
        heads:    int = HEADS,
        layers:   int = LAYERS,
        dropout:  float = DROP,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.n_ent       = n_ent
        self.outlet_idx  = OUTLET_IDX
        self.use_spatial = use_spatial

        # 1. Static encoder
        self.static_enc = StaticEncoder(n_static, hid)

        # 2. Input projection: dyn + static_emb -> hid
        self.input_proj = nn.Sequential(
            nn.Linear(n_dyn + hid, hid),
            nn.SiLU(),
        )

        # 3. Shared LSTM across entities
        self.lstm = nn.LSTM(hid, hid, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)

        # 4. Spatial multi-head attention
        self.spatial_attn = nn.MultiheadAttention(hid, heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(hid)

        self.drop = nn.Dropout(dropout)

        # 5a. Pretrain head (Phase 1): scalar per entity
        self.pretrain_head = nn.Linear(hid, 1)

        # 5b. Quantile head (Phase 2): P10/P50/P90 for outlet
        self.quantile_head = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid // 2, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _encode(self, x_dyn, x_static):
        """
        x_dyn   : (B, N, L, n_dyn)
        x_static: (B, N, n_static)
        returns  : (B, N, hid) — entity representations
        """
        B, N, L, _ = x_dyn.shape

        # Static embedding: (B, N, hid)
        s_emb = self.static_enc(x_static)

        # Expand static to each time step and concat with dyn: (B, N, L, n_dyn+hid)
        s_exp = s_emb.unsqueeze(2).expand(-1, -1, L, -1)
        x = torch.cat([x_dyn, s_exp], dim=-1)   # (B, N, L, n_dyn+hid)

        # Project: (B, N, L, hid)
        x = self.input_proj(x)

        # LSTM per entity: treat N as batch dim -> (B*N, L, hid)
        x_flat  = x.reshape(B * N, L, -1)
        out, _  = self.lstm(x_flat)        # (B*N, L, hid)
        h_last  = out[:, -1, :]            # (B*N, hid) — last step
        h_last  = h_last.reshape(B, N, -1) # (B, N, hid)

        # Spatial attention across 9 entities (ablation: skip if use_spatial=False)
        if self.use_spatial:
            h_attn, _ = self.spatial_attn(h_last, h_last, h_last)  # (B, N, hid)
            h = self.spatial_norm(h_last + h_attn)                  # residual
        else:
            h = h_last

        return h   # (B, N, hid)

    def forward_pretrain(self, x_dyn, x_static):
        """Phase 1: predict scalar q for each entity."""
        h = self._encode(x_dyn, x_static)   # (B, N, hid)
        h = self.drop(h)
        pred = self.pretrain_head(h).squeeze(-1)   # (B, N)
        return pred

    def forward(self, x_dyn, x_static):
        """Phase 2: predict P10/P50/P90 at outlet."""
        h = self._encode(x_dyn, x_static)            # (B, N, hid)
        h_out = self.drop(h[:, self.outlet_idx, :])  # (B, hid)
        raw = self.quantile_head(h_out)              # (B, 3)

        # Softplus no-crossing guarantee
        p50 = raw[:, 0]
        p10 = p50 - F.softplus(raw[:, 1])
        p90 = p50 + F.softplus(raw[:, 2])
        return torch.stack([p10, p50, p90], dim=1)  # (B, 3)


# ── Loss functions ────────────────────────────────────────────────────────────

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    total = torch.zeros(1, device=pred.device)
    for i, q in enumerate(QUANTILES):
        err = target - pred[:, i]
        loss_i = torch.where(err >= 0, q * err, (q - 1) * err)
        if weights is not None:
            loss_i = loss_i * weights
        total = total + loss_i.mean()
    return total / len(QUANTILES)


def compute_weights(target_mm: torch.Tensor) -> torch.Tensor:
    """Sample weights: ALERT_W for Q>Q90, 1.0 otherwise."""
    w = torch.ones_like(target_mm)
    w[target_mm > Q90_MM] = ALERT_W
    return w


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(obs_m3: np.ndarray, pred_m3: np.ndarray,
                    p10_m3: np.ndarray | None = None,
                    p90_m3: np.ndarray | None = None) -> dict:
    obs  = np.array(obs_m3).astype(float)
    pred = np.array(pred_m3).astype(float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]

    # NSE
    nse_denom = np.sum((obs - obs.mean()) ** 2)
    nse = 1 - np.sum((obs - pred) ** 2) / nse_denom if nse_denom > 0 else np.nan

    # NSE_sqrt (Pushpalatha 2012)
    s_obs, s_pred = np.sqrt(np.maximum(obs, 0)), np.sqrt(np.maximum(pred, 0))
    s_denom = np.sum((s_obs - s_obs.mean()) ** 2)
    nse_sqrt = 1 - np.sum((s_obs - s_pred) ** 2) / s_denom if s_denom > 0 else np.nan

    # Alert metrics (Q90 threshold)
    Q90_M3 = 40.89
    obs_alert  = obs  >= Q90_M3
    pred_alert = pred >= Q90_M3
    tp = np.sum(obs_alert & pred_alert)
    fp = np.sum(~obs_alert & pred_alert)
    fn = np.sum(obs_alert & ~pred_alert)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    pod = tp / (tp + fn)      if (tp + fn)      > 0 else 0.0
    far = fp / (tp + fp)      if (tp + fp)      > 0 else 0.0

    j_alert = 0.25 * nse_sqrt + 0.25 * nse + 0.30 * csi + 0.10 * pod - 0.10 * far

    out = {"NSE": nse, "NSE_sqrt": nse_sqrt, "J_alert": j_alert,
           "CSI": csi, "POD": pod, "FAR": far}

    if p10_m3 is not None and p90_m3 is not None:
        p10, p90 = np.array(p10_m3)[mask], np.array(p90_m3)[mask]
        picp = np.mean((obs >= p10) & (obs <= p90))
        pinaw = np.mean(p90 - p10) / (obs.max() - obs.min() + 1e-8)
        out["PICP"] = picp
        out["PINAW"] = pinaw

    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train_phase1(model: HydroST, loader: DataLoader, epochs: int) -> HydroST:
    """Phase 1: pretrain on GR4J outlet Q (MSE loss on all entity predictions)."""
    opt   = torch.optim.AdamW(model.parameters(), lr=LR1, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR1, epochs=epochs,
                                                  steps_per_epoch=len(loader))
    model.train()
    log.info(f"Phase 1: {epochs} epochs, {len(loader.dataset)} sequences")

    for ep in range(epochs):
        total_loss = 0.0
        for x_dyn, x_static, y in loader:
            x_dyn    = x_dyn.to(DEVICE)
            x_static = x_static.to(DEVICE)
            y        = y.to(DEVICE)

            opt.zero_grad()
            pred = model.forward_pretrain(x_dyn, x_static)  # (B, N_ent)
            # Target: all entities predict the same outlet Q (y replicated)
            y_exp = y.unsqueeze(1).expand(-1, len(ENTITIES))
            loss  = F.mse_loss(pred, y_exp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()

        if (ep + 1) % 10 == 0 or ep == 0:
            log.info(f"  EP {ep+1:3d}/{epochs}  loss={total_loss/len(loader):.5f}")

    return model


def train_phase2(model: HydroST, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int) -> HydroST:
    """Phase 2: finetune on observed Q with PinballLoss + sample weighting."""
    opt   = torch.optim.AdamW(model.parameters(), lr=LR2, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss  = float("inf")
    best_state = None
    patience   = 0

    log.info(f"Phase 2: {epochs} epochs, {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x_dyn, x_static, y in train_loader:
            x_dyn    = x_dyn.to(DEVICE)
            x_static = x_static.to(DEVICE)
            y        = y.to(DEVICE)
            w        = compute_weights(y)

            opt.zero_grad()
            pred = model(x_dyn, x_static)   # (B, 3)
            # Mixed loss: pinball (alert-weighted) + MSE on P50 (preserves NSE)
            loss = ALPHA * pinball_loss(pred, y, w) + (1 - ALPHA) * F.mse_loss(pred[:, 1], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        # Validation (no weights, unbiased estimate)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_dyn, x_static, y in val_loader:
                pred = model(x_dyn.to(DEVICE), x_static.to(DEVICE))
                val_loss += pinball_loss(pred, y.to(DEVICE)).item()

        sched.step()
        val_loss /= len(val_loader)

        if (ep + 1) % 10 == 0 or ep == 0:
            log.info(f"  EP {ep+1:3d}/{epochs}  train={train_loss/len(train_loader):.5f}  "
                     f"val={val_loss:.5f}  {'[BEST]' if val_loss < best_loss else ''}")

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= PAT:
                log.info(f"  Early stop at epoch {ep+1}")
                break

    log.info(f"  Best val loss: {best_loss:.5f}")
    model.load_state_dict(best_state)
    return model


def evaluate(model: HydroST, loader: DataLoader, label: str) -> dict:
    """Evaluate model on a DataLoader, return metrics dict."""
    model.eval()
    preds_p50, preds_p10, preds_p90, targets = [], [], [], []

    with torch.no_grad():
        for x_dyn, x_static, y in loader:
            pred = model(x_dyn.to(DEVICE), x_static.to(DEVICE)).cpu().numpy()
            preds_p10.extend(pred[:, 0])
            preds_p50.extend(pred[:, 1])
            preds_p90.extend(pred[:, 2])
            targets.extend(y.numpy())

    # Convert mm/day -> m3/s
    obs_m3   = np.array(targets)  / Q_CONV
    p50_m3   = np.array(preds_p50) / Q_CONV
    p10_m3   = np.array(preds_p10) / Q_CONV
    p90_m3   = np.array(preds_p90) / Q_CONV

    metrics = compute_metrics(obs_m3, p50_m3, p10_m3, p90_m3)

    log.info(f"\n{'='*55}")
    log.info(f"  Evaluation: {label}")
    log.info(f"{'='*55}")
    log.info(f"  NSE        = {metrics['NSE']:.4f}")
    log.info(f"  NSE_sqrt   = {metrics['NSE_sqrt']:.4f}")
    log.info(f"  J_alert    = {metrics['J_alert']:.4f}")
    log.info(f"  CSI_Q90    = {metrics['CSI']:.4f}")
    log.info(f"  POD        = {metrics['POD']:.4f}")
    log.info(f"  FAR        = {metrics['FAR']:.4f}")
    if "PICP" in metrics:
        log.info(f"  PICP       = {metrics['PICP']:.4f}")
        log.info(f"  PINAW      = {metrics['PINAW']:.4f}")
    log.info(f"{'='*55}")

    return {
        "obs_m3": obs_m3, "p50_m3": p50_m3,
        "p10_m3": p10_m3, "p90_m3": p90_m3,
        **metrics,
    }


# ── Save predictions ──────────────────────────────────────────────────────────

def save_predictions(results_dict: dict, label: str):
    out = {}
    for key in ["obs_m3", "p50_m3", "p10_m3", "p90_m3"]:
        if key in results_dict:
            out[key.replace("_m3", "")] = results_dict[key]
    df = pd.DataFrame(out)
    path = OUT_DIR / f"78_hydrost_{label}.csv"
    df.to_csv(path, index=False)
    log.info(f"Predictions saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Load pretrained weights if available, skip Phase 1")
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    log.info(f"HydroST  hid={HID}  heads={HEADS}  layers={LAYERS}  enc={ENC}")

    # ── Load and preprocess data ─────────────────────────────────────────────
    df = load_data()
    dyn_arr, static_arr, target, dates, scalers = build_arrays(df)

    # ── Build date ranges ────────────────────────────────────────────────────
    dates_pd = pd.DatetimeIndex(dates)
    pretrain_dates = dates_pd[dates_pd <= PRETRAIN_END]
    ft_dates       = dates_pd[(dates_pd >= FT_START) & (dates_pd <= FT_END)]
    val_dates      = dates_pd[(dates_pd >= VAL_START) & (dates_pd <= VAL_END)]
    # TEST DATES: defined but NEVER passed to any training/selection procedure
    # test_dates = dates_pd[dates_pd >= TEST_START]

    log.info(f"Pretrain sequences (1981-2020) target days: {len(pretrain_dates)}")
    log.info(f"Finetune sequences (2021-2022): {len(ft_dates)}")
    log.info(f"Val sequences     (2023)       : {len(val_dates)}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    ds_pretrain = HydroDataset(dyn_arr, static_arr, target, dates, pretrain_dates)
    ds_ft       = HydroDataset(dyn_arr, static_arr, target, dates, ft_dates)
    ds_val      = HydroDataset(dyn_arr, static_arr, target, dates, val_dates)

    loader_pre = DataLoader(ds_pretrain, batch_size=BATCH1, shuffle=True,  num_workers=0, pin_memory=True)
    loader_ft  = DataLoader(ds_ft,       batch_size=BATCH2, shuffle=True,  num_workers=0, pin_memory=True)
    loader_val = DataLoader(ds_val,       batch_size=BATCH2, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HydroST().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"HydroST parameters: {n_params:,}")

    # ── Phase 1: Pretrain on GR4J ────────────────────────────────────────────
    ckpt_path = OUT_DIR / "78_hydrost_pretrained.pt"
    if args.skip_pretrain and ckpt_path.exists():
        log.info(f"Skipping Phase 1 — loading checkpoint: {ckpt_path.name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    else:
        log.info("\n=== PHASE 1: Pretrain on GR4J (1981-2020) ===")
        model = train_phase1(model, loader_pre, EP1)
        torch.save(model.state_dict(), ckpt_path)
        log.info(f"Pretrained weights saved: {ckpt_path.name}")

    # Phase 1 val evaluation (sanity check on val period, deterministic)
    log.info("\n--- Phase 1 val check (P50 from pretrain head) ---")
    model.eval()
    p1_preds, p1_targets = [], []
    with torch.no_grad():
        for x_dyn, x_static, y in loader_val:
            pred_all = model.forward_pretrain(x_dyn.to(DEVICE), x_static.to(DEVICE))
            p1_preds.extend(pred_all[:, OUTLET_IDX].cpu().numpy())
            p1_targets.extend(y.numpy())
    p1_m3  = np.array(p1_preds)  / Q_CONV
    obs_m3 = np.array(p1_targets) / Q_CONV
    p1_metrics = compute_metrics(obs_m3, p1_m3)
    log.info(f"  Phase1 val NSE={p1_metrics['NSE']:.4f}  NSE_sqrt={p1_metrics['NSE_sqrt']:.4f}")

    # ── Phase 2: Finetune on Q obs ───────────────────────────────────────────
    log.info("\n=== PHASE 2: Finetune on Q obs (2021-2022) ===")
    model = train_phase2(model, loader_ft, loader_val, EP2)

    # ── Validation evaluation ─────────────────────────────────────────────────
    log.info("\n=== VALIDATION EVALUATION (2023) ===")
    val_results = evaluate(model, loader_val, "val_2023")
    save_predictions(val_results, "val_2023")

    # ── Save final model ──────────────────────────────────────────────────────
    final_ckpt = OUT_DIR / "78_hydrost_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    log.info(f"\nFinal model saved: {final_ckpt.name}")

    # ── Model summary ─────────────────────────────────────────────────────────
    log.info("\n=== HYDROST RESULT SUMMARY ===")
    log.info(f"  Val 2023  NSE={val_results['NSE']:.4f}  NSE_sqrt={val_results['NSE_sqrt']:.4f}"
             f"  J_alert={val_results['J_alert']:.4f}")
    log.info(f"  Val 2023  CSI={val_results['CSI']:.4f}  POD={val_results['POD']:.4f}"
             f"  FAR={val_results['FAR']:.4f}")
    if "PICP" in val_results:
        log.info(f"  Val 2023  PICP={val_results['PICP']:.4f}  PINAW={val_results['PINAW']:.4f}")

    log.info("\n[NOTE] Test period 2024-2025 is SEALED — not evaluated here.")
    log.info("  Run 78b_hydrost_test_eval.py after all model decisions are final.")


if __name__ == "__main__":
    main()
