#!/usr/bin/env python3
"""
Script 74: TFT v2 — Pinball Loss + Sample Weighting + Mayor Capacidad

Mejoras sobre Script 72 (TFT-transfer, NSE=0.88 1d, POD=29%):
  1. PinballLoss P10/P50/P90 — salidas probabilísticas.
     La cabeza P90 penaliza subestimar → ataca POD directamente.
  2. Sample weighting w=5 para Q > Q90 en fase finetune.
     Los 138 eventos de alerta (2021-23) pesan 5× más en el gradiente.
  3. Mayor capacidad: HID 32→64, HEADS 2→4, ENC 45→60.
  4. No-cruce garantizado: P10 < P50 < P90 via softplus deltas.

Arquitectura transfer (= Script 72):
  Fase 1 pretrain : GR4J corregido 1981-2017  (val GR4J 2018-2020)
  Fase 2 finetune : Q obs real 2021-23H1       (val Q obs 2023H2, w_alert=5)
  Test            : Q obs real 2024-2025        (N≈423, nunca visto)

Métricas nuevas vs Script 72:
  POD_det   P50 > Q90   (determinista, comparable a Scr72)
  POD_prob  P90 > Q90   (probabilista, más sensible)
  Coverage  % Q obs en [P10, P90]  (target ≥ 80 %)
  Width     ancho medio banda (m³/s)
  IS        Interval Score  (proper scoring rule, α = 0.8)

Entorno: .venv313  (Python 3.13, PyTorch 2.6.0+cu124, RTX 4070 SUPER)

Salidas:
  outputs/ml_Q/tft_v2_results.csv
  outputs/ml_Q/tft_v2_predictions.csv
  outputs/figures/ml_Q/TFT01_quantile_bands.png
  outputs/figures/ml_Q/TFT02_v1v2_comparison.png
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Fix crítico: torch.load en Lightning 2.6 (Regla 15) ───────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

torch.set_float32_matmul_precision("high")   # Tensor Cores RTX 4070 SUPER

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("tft_v2")

D6_CSV  = ROOT / "data/model_ready/D6_multientity.csv"
QOBS    = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
ONI_CSV = ROOT / "data/silver/enso/S3_oni_1950_2026.csv"
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

Q_CONV = 86.4 / 3062.62   # m³/s → mm/d
Q90    = 40.89             # m³/s
SEED   = 42

# ── Hiperparámetros v2 (vs Scr72: HID=32, HEADS=2, ENC=45) ───────────────────
ENC      = 60       # días de contexto (ventana encoder)
HID      = 64       # hidden size  (2× v1)
HEADS    = 4        # attention heads (2× v1)
DROP     = 0.3
LR       = 7e-4
LR_FT    = 1e-4
WD       = 3e-5
BATCH    = 64
MAX_EP1  = 200
MAX_EP2  = 80
PAT1     = 30
PAT2     = 15
ALERT_W  = 5.0      # peso para Q > Q90 en finetune
QUANTILES = [0.1, 0.5, 0.9]
ALPHA    = 0.8      # cobertura objetivo para Interval Score

SPATIAL  = ["pr_mm", "tmax_c", "tmin_c", "pet_mm", "api",
             "spi_30d", "spi_90d", "water_deficit_30d"]
SHARED   = ["oni_index", "sin_doy_1", "cos_doy_1", "hydro_month", "is_wet_season"]
KEY_SUBS = ["sub_649", "sub_655", "sub_656"]   # cuencas altas (79 % del caudal)


# ── Feature engineering (idéntico a Script 72) ────────────────────────────────

def build_wide(D6):
    cols = {}
    for c in SPATIAL:
        piv = D6.pivot_table(index="date", columns="entity_id", values=c)
        cols[f"{c}_basin"] = piv.mean(axis=1)
        for e in KEY_SUBS:
            if e in piv.columns:
                cols[f"{c}_{e}"] = piv[e]
    df = pd.DataFrame(cols)
    ref = D6[D6["entity_id"] == "sub_634"].set_index("date")
    df = df.join(ref[SHARED + ["q_mm", "q_next_1d", "q_sum_next_7d"]]).sort_index()
    q = df["q_mm"]
    df["q_lag7"]  = q.shift(7)
    df["q_roll7"] = q.rolling(7,  min_periods=3).mean()
    df["q_roll30"]= q.rolling(30, min_periods=10).mean()
    return df


# ── Métricas ──────────────────────────────────────────────────────────────────

def _nse(o, p):
    return 1 - np.sum((o-p)**2) / (np.sum((o-o.mean())**2) + 1e-12)

def _nse_sqrt(o, p):
    w = np.maximum(o, 0)**0.5
    return 1 - np.sum(w*(o-p)**2) / (np.sum(w*(o-o.mean())**2) + 1e-12)

def _lognse(o, p):
    lo = np.log(o + 1); lp = np.log(np.clip(p, 0, None) + 1)
    return 1 - np.sum((lo-lp)**2) / (np.sum((lo-lo.mean())**2) + 1e-12)

def _kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    a = p.std() / (o.std() + 1e-12)
    b = p.mean() / (o.mean() + 1e-12)
    return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)

def det_metrics(o, p, thr):
    eo, ep = o > thr, p > thr
    TP = int(np.sum(eo & ep)); FP = int(np.sum(~eo & ep))
    FN = int(np.sum(eo & ~ep)); TN = int(np.sum(~eo & ~ep))
    POD = TP / (TP + FN + 1e-9)
    FAR = FP / (FP + TN + 1e-9)
    CSI = TP / (TP + FP + FN + 1e-9)
    dh  = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
    HSS = 2*(TP*TN - FP*FN) / (dh + 1e-9) if dh > 0 else 0
    pb  = (p.sum() - o.sum()) / (o.sum() + 1e-12) * 100
    q90l = np.percentile(o, 90); pk = o > q90l
    pb90 = (p[pk].sum()-o[pk].sum())/(o[pk].sum()+1e-12)*100 if pk.sum()>3 else np.nan
    j = 0.25*_nse_sqrt(o,p) + 0.25*_nse(o,p) + 0.30*CSI + 0.10*POD - 0.10*FAR
    return {
        "NSE":      round(_nse(o,p),   3),
        "NSE_sqrt": round(_nse_sqrt(o,p), 3),
        "logNSE":   round(_lognse(o,p),3),
        "KGE":      round(_kge(o,p),   3),
        "PBIAS":    round(float(pb),   1),
        "PBIAS_Q90":round(float(pb90), 1) if not np.isnan(pb90) else np.nan,
        "POD_det":  round(POD, 3),
        "FAR":      round(FAR, 3),
        "CSI":      round(CSI, 3),
        "HSS":      round(HSS, 3),
        "J_alert":  round(j,   4),
        "N":        len(o),
    }

def prob_metrics(o, p10, p50, p90, thr):
    """Métricas para salida probabilística P10/P50/P90."""
    inside   = (o >= p10) & (o <= p90)
    coverage = float(inside.mean())
    width    = float((p90 - p10).mean())
    # Interval Score (Gneiting & Raftery 2007): penaliza ancho excesivo y misses
    IS = (width
          + (2/ALPHA) * np.maximum(p10 - o, 0)
          + (2/ALPHA) * np.maximum(o - p90, 0))
    # POD probabilista: P90 > umbral (más sensible que P50 > umbral)
    eo = o > thr; ep90 = p90 > thr
    TP = int(np.sum(eo & ep90)); FP = int(np.sum(~eo & ep90))
    FN = int(np.sum(eo & ~ep90)); TN = int(np.sum(~eo & ~ep90))
    POD_prob = TP / (TP + FN + 1e-9)
    FAR_prob = FP / (FP + TN + 1e-9)
    return {
        "Coverage80": round(coverage, 3),
        "Width_m3s":  round(width / Q_CONV, 2),
        "IS":         round(float(IS.mean()), 4),
        "POD_prob":   round(POD_prob, 3),
        "FAR_prob":   round(FAR_prob, 3),
    }


# ── Modelo TFTLite v2 ─────────────────────────────────────────────────────────

class VSN(nn.Module):
    def __init__(self, nf, h):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, nf))

    def forward(self, x):
        return x * torch.softmax(self.w(x), dim=-1) * x.shape[-1]


class TFTLitev2(nn.Module):
    """VSN → LSTM → Multi-head Attention → 3 quantile heads.

    Salida garantizada P10 < P50 < P90 via softplus deltas:
      p50 = head_raw[:, 0]
      p10 = p50 - softplus(head_raw[:, 1])
      p90 = p50 + softplus(head_raw[:, 2])
    """
    def __init__(self, nf):
        super().__init__()
        self.vsn  = VSN(nf, HID)
        self.proj = nn.Linear(nf, HID)
        self.pos  = nn.Parameter(torch.randn(1, ENC, HID) * 0.02)
        self.lstm = nn.LSTM(HID, HID, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(HID, HEADS, dropout=DROP, batch_first=True)
        self.norm = nn.LayerNorm(HID)
        self.drop = nn.Dropout(DROP)
        # 3 outputs: [p50_center, delta_down, delta_up]
        self.head = nn.Sequential(
            nn.Linear(HID, HID // 2), nn.ReLU(),
            nn.Dropout(DROP),
            nn.Linear(HID // 2, 3),
        )

    def forward(self, x):                          # x: (B, ENC, nf)
        h = self.proj(self.vsn(x)) + self.pos
        o, _ = self.lstm(h)
        a, _ = self.attn(o, o, o)
        h = self.norm(o + self.drop(a))
        raw = self.head(h[:, -1, :])               # (B, 3)
        p50 = raw[:, 0]
        p10 = p50 - F.softplus(raw[:, 1])          # siempre < p50
        p90 = p50 + F.softplus(raw[:, 2])          # siempre > p50
        return torch.stack([p10, p50, p90], dim=1) # (B, 3)


# ── Pinball loss ──────────────────────────────────────────────────────────────

def pinball_loss(pred, target, weights=None):
    """pred (B,3), target (B,), weights (B,) opcionales."""
    total = 0.0
    for i, q in enumerate(QUANTILES):
        err = target - pred[:, i]
        l   = torch.where(err >= 0, q * err, (q - 1) * err)
        total += (l * weights).mean() if weights is not None else l.mean()
    return total / len(QUANTILES)


# ── Sequence builder ──────────────────────────────────────────────────────────

def make_seqs(Xs, ysc, mask, y_raw=None, alert_thr=None, alert_w=1.0):
    """Construye secuencias de longitud ENC para las fechas en `mask`.

    Returns tensores (X_seq, y_seq, w_seq) + array de índices globales.
    Devuelve None si no hay secuencias válidas.
    """
    mask_set = set(np.where(mask)[0])
    Xs_out, ys_out, ws_out, idx_out = [], [], [], []
    for i in range(ENC, len(Xs)):
        if i not in mask_set or not np.isfinite(ysc[i]):
            continue
        Xs_out.append(Xs[i - ENC: i])
        ys_out.append(ysc[i])
        w = alert_w if (y_raw is not None and alert_thr is not None
                         and y_raw[i] > alert_thr) else 1.0
        ws_out.append(w)
        idx_out.append(i)

    if not Xs_out:
        return None, None, None, None
    return (
        torch.tensor(np.array(Xs_out, np.float32)),
        torch.tensor(np.array(ys_out, np.float32)),
        torch.tensor(np.array(ws_out, np.float32)),
        np.array(idx_out),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_phase(model, Xtr, ytr, wtr, Xv, yv, lr, max_ep, pat, dev):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=8)
    dl    = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=BATCH, shuffle=True)
    Xv_d  = Xv.to(dev)
    yv_np = yv.numpy()
    best, best_state, no_improve = np.inf, None, 0

    for ep in range(max_ep):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            p50_v = model(Xv_d)[:, 1].cpu().numpy()   # val loss sobre P50
        vl = float(np.mean((p50_v - yv_np)**2))
        sched.step(vl)

        if vl < best:
            best = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= pat:
                break

    model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Script 74 — TFT v2 (Quantile + Weighted) | device={dev}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # ── Datos ─────────────────────────────────────────────────────────────────
    D6    = pd.read_csv(D6_CSV, parse_dates=["date"])
    qobs  = pd.read_csv(QOBS, index_col=0, parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm = qobs * Q_CONV
    oni   = pd.read_csv(ONI_CSV, parse_dates=["date"]).set_index("date")["oni"]

    df    = build_wide(D6)
    dates = df.index
    feat  = [c for c in df.columns if c not in ["q_next_1d", "q_sum_next_7d"]]
    nf    = len(feat)
    log.info(f"Wide features: {nf} | {dates.min().date()} → {dates.max().date()}")

    # Splits (= Script 72 para comparación directa)
    m_p  = dates <= "2017-12-31"                                   # pretrain
    m_pv = (dates >= "2018-01-01") & (dates <= "2020-12-31")      # val pretrain
    m_ft = (dates >= "2021-01-01") & (dates <= "2023-06-30")      # finetune
    m_fv = (dates >= "2023-07-01") & (dates <= "2023-12-31")      # val finetune
    m_te = dates >= "2024-01-01"                                   # test
    log.info(f"Pretrain: {m_p.sum()}d | FT: {m_ft.sum()}d | Val FT: {m_fv.sum()}d | Test: {m_te.sum()}d")

    # Normalizar features (solo pretrain, anti-leakage)
    X  = df[feat].fillna(0).values.astype(np.float32)
    mu = X[m_p].mean(0); sd = X[m_p].std(0) + 1e-8
    Xs = (X - mu) / sd

    targets = {
        "q_next_1d":     (qobs_mm.shift(-1),                            1),
        "q_sum_next_7d": (qobs_mm.shift(-1).rolling(7).sum().shift(-6), 7),
    }

    all_rows  = []
    all_preds = []

    for target, (obs_real, hmult) in targets.items():
        log.info(f"\n{'='*55}\n[{target}]\n{'='*55}")
        y    = df[target].values.astype(np.float32)
        thr  = Q90 * Q_CONV * hmult

        # log1p + estandarizar target (fit solo en pretrain)
        ylog = np.log1p(np.clip(y, 0, None))
        ymu  = np.nanmean(ylog[m_p])
        ysd  = np.nanstd(ylog[m_p]) + 1e-8
        ysc  = ((ylog - ymu) / ysd).astype(np.float32)
        inv  = lambda v: np.expm1(np.asarray(v) * ysd + ymu)

        # Construir secuencias por split
        Xp_t,  yp_t,  wp_t,  idxp  = make_seqs(Xs, ysc, m_p)
        Xpv_t, ypv_t, wpv_t, idxpv = make_seqs(Xs, ysc, m_pv)
        Xft_t, yft_t, wft_t, idxft = make_seqs(Xs, ysc, m_ft,
                                                 y_raw=y, alert_thr=thr, alert_w=ALERT_W)
        Xfv_t, yfv_t, wfv_t, idxfv = make_seqs(Xs, ysc, m_fv)
        Xte_t, yte_t, wte_t, idxte = make_seqs(Xs, ysc, m_te)

        if Xte_t is None:
            log.warning(f"[{target}] Sin secuencias de test — omitiendo.")
            continue

        n_alert_ft = int((y[idxft] > thr).sum()) if idxft is not None else 0
        log.info(f"  Seqs → pretrain={len(idxp)} | finetune={len(idxft)} "
                 f"(alertas={n_alert_ft}) | test={len(idxte)}")

        # ── Fase 1: pretrain GR4J ─────────────────────────────────────────
        model = TFTLitev2(nf).to(dev)
        t0 = time.time()
        log.info(f"  Fase 1 pretrain (GR4J ≤2017, max={MAX_EP1} ep, pat={PAT1})...")
        model = train_phase(model, Xp_t, yp_t, wp_t,
                             Xpv_t, ypv_t, LR, MAX_EP1, PAT1, dev)
        log.info(f"  Fase 1 OK ({time.time()-t0:.0f}s)")

        # ── Fase 2: finetune Q obs con sample weighting ───────────────────
        t0 = time.time()
        log.info(f"  Fase 2 finetune (Q obs 2021-23H1, w_alert={ALERT_W}, max={MAX_EP2} ep)...")
        model = train_phase(model, Xft_t, yft_t, wft_t,
                             Xfv_t, yfv_t, LR_FT, MAX_EP2, PAT2, dev)
        log.info(f"  Fase 2 OK ({time.time()-t0:.0f}s)")

        # ── Inferencia test ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            raw_scaled = model(Xte_t.to(dev)).cpu().numpy()   # (N_te, 3)

        p10_mm = inv(raw_scaled[:, 0])
        p50_mm = inv(raw_scaled[:, 1])
        p90_mm = inv(raw_scaled[:, 2])

        dte   = dates[idxte]
        o_ser = obs_real.reindex(dte)
        valid = o_ser.notna()

        o_v   = o_ser[valid].values                   # en mm/d
        p10_v = p10_mm[valid.values]
        p50_v = p50_mm[valid.values]
        p90_v = p90_mm[valid.values]

        # Métricas deterministas (P50) y probabilistas
        m_det  = det_metrics(o_v, p50_v, thr)
        m_prob = prob_metrics(o_v, p10_v, p50_v, p90_v, thr)

        row = {**m_det, **m_prob, "target": target, "model": "TFT-v2"}
        all_rows.append(row)

        log.info(f"  P50  → NSE={m_det['NSE']} KGE={m_det['KGE']} "
                 f"J_alert={m_det['J_alert']} POD_det={m_det['POD_det']:.0%}")
        log.info(f"  P90  → POD_prob={m_prob['POD_prob']:.0%} "
                 f"FAR_prob={m_prob['FAR_prob']:.0%}")
        log.info(f"  Banda → Coverage={m_prob['Coverage80']:.1%} "
                 f"Width={m_prob['Width_m3s']:.1f}m³/s IS={m_prob['IS']:.4f}")

        # ENSO breakdown (q_next_1d)
        if target == "q_next_1d":
            sub = pd.DataFrame({"obs": o_v, "p50": p50_v},
                                index=dte[valid.values])
            oni_d = oni.reindex(sub.index, method="ffill")
            wet   = sub.index.month.isin([11, 12, 1, 2, 3, 4])
            log.info(f"  Por temporada — húmeda NSE={_nse(sub[wet]['obs'].values, sub[wet]['p50'].values):.3f}"
                     f" | seca NSE={_nse(sub[~wet]['obs'].values, sub[~wet]['p50'].values):.3f}")
            for lbl, mk in [("Niño", oni_d > 0.5), ("Niña", oni_d < -0.5),
                             ("Neutral", (oni_d >= -0.5) & (oni_d <= 0.5))]:
                mv = mk.values
                if mv.sum() > 10:
                    log.info(f"  ENSO {lbl}: NSE={_nse(sub[mv]['obs'].values, sub[mv]['p50'].values):.3f} (n={mv.sum()})")

        # Guardar predicciones
        for d, oo, p1, p5, p9 in zip(dte[valid.values],
                                      o_v / (Q_CONV * hmult),
                                      p10_v / (Q_CONV * hmult),
                                      p50_v / (Q_CONV * hmult),
                                      p90_v / (Q_CONV * hmult)):
            all_preds.append({"date": d, "q_obs": oo, "q_p10": p1,
                               "q_p50": p5, "q_p90": p9, "target": target})

        # ── Figura TFT01: banda probabilística (q_next_1d) ────────────────
        if target == "q_next_1d":
            p = pd.DataFrame({
                "date": dte[valid.values],
                "obs":  o_v / Q_CONV,
                "p10":  p10_v / Q_CONV,
                "p50":  p50_v / Q_CONV,
                "p90":  p90_v / Q_CONV,
            }).sort_values("date")

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.fill_between(p["date"], p["p10"], p["p90"],
                            alpha=0.25, color="#3498db",
                            label=f"Banda P10-P90  (cobertura = {m_prob['Coverage80']:.0%})")
            ax.plot(p["date"], p["p50"], color="#2980b9", lw=1.2, label="P50 (mediana)")
            ax.plot(p["date"], p["obs"], color="#c0392b", lw=1.4, label="Q obs real")
            ax.axhline(Q90, ls=":", color="orange", alpha=0.8,
                       label=f"Q90 = {Q90:.1f} m³/s")

            # Marcar días donde P90 detecta alerta
            alert_days = p[p["p90"] > Q90]
            if len(alert_days):
                ax.scatter(alert_days["date"], alert_days["p90"],
                           marker="v", s=18, color="#f39c12", zorder=5,
                           label=f"Alerta P90 ({len(alert_days)} días)")

            ax.set_ylabel("Q (m³/s)"); ax.set_ylim(0)
            ax.set_title(
                f"TFT v2 — Pronóstico probabilístico Q +1 día  |  test 2024-2025\n"
                f"P50: NSE={m_det['NSE']:.3f}  KGE={m_det['KGE']:.3f}  "
                f"POD_det={m_det['POD_det']:.0%}  |  "
                f"POD_prob={m_prob['POD_prob']:.0%}  "
                f"Coverage={m_prob['Coverage80']:.0%}  "
                f"Width={m_prob['Width_m3s']:.1f} m³/s",
                fontweight="bold", fontsize=10,
            )
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig.tight_layout()
            fig.savefig(FIG_DIR / "TFT01_quantile_bands.png", bbox_inches="tight")
            plt.close(fig)
            log.info("  → TFT01_quantile_bands.png")

    # ── Guardar resultados ────────────────────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "tft_v2_results.csv", index=False)
    pd.DataFrame(all_preds).to_csv(OUT_DIR / "tft_v2_predictions.csv", index=False)
    log.info(f"Resultados → tft_v2_results.csv | predicciones → tft_v2_predictions.csv")

    # ── Figura TFT02: comparativa v1 vs v2 ────────────────────────────────────
    v1_csv = OUT_DIR / "transfer_metrics_full.csv"
    if v1_csv.exists() and all_rows:
        v1 = pd.read_csv(v1_csv)
        v1_tft = v1[v1["model"] == "TFT-transfer"].copy()
        # Unificar nombre de columna POD → POD_det para comparación
        if "POD" in v1_tft.columns and "POD_det" not in v1_tft.columns:
            v1_tft = v1_tft.rename(columns={"POD": "POD_det"})
        v1_tft = v1_tft.set_index("target")

        v2 = pd.DataFrame(all_rows).set_index("target")
        cmp_cols  = ["NSE", "KGE", "J_alert", "POD_det", "CSI"]
        cmp_names = ["NSE", "KGE", "J_alert", "POD", "CSI"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, tgt in zip(axes, ["q_next_1d", "q_sum_next_7d"]):
            if tgt not in v1_tft.index or tgt not in v2.index:
                ax.set_visible(False); continue

            bv = np.array([v1_tft.loc[tgt, c] if c in v1_tft.columns else np.nan
                           for c in cmp_cols], dtype=float)
            nv = np.array([v2.loc[tgt, c]     if c in v2.columns     else np.nan
                           for c in cmp_cols], dtype=float)

            x, w = np.arange(len(cmp_cols)), 0.35
            ax.bar(x - w/2, bv, w, label="TFT v1 (Scr72, Huber)",   color="#e74c3c", alpha=0.8)
            ax.bar(x + w/2, nv, w, label="TFT v2 (Scr74, Pinball)", color="#27ae60", alpha=0.8)

            for xi, (b, n) in enumerate(zip(bv, nv)):
                if not (np.isnan(b) or np.isnan(n)):
                    d = n - b
                    ax.annotate(f"{d:+.3f}",
                                (xi + w/2, max(b, n) + 0.02),
                                ha="center", va="bottom",
                                fontsize=7, color="#27ae60" if d >= 0 else "#c0392b",
                                fontweight="bold")

            ax.set_xticks(x); ax.set_xticklabels(cmp_names, rotation=30, ha="right")
            ax.set_ylim(-0.15, 1.2)
            ax.set_title("Q +1 día" if tgt == "q_next_1d" else "Q suma 7 días",
                         fontweight="bold")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylabel("Métrica")

        fig.suptitle(
            "TFT v1 (HuberLoss, HID=32)  vs  TFT v2 (PinballLoss+Weights, HID=64)\n"
            "Test = Q obs real 2024-2025  |  Δ sobre barra derecha",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "TFT02_v1v2_comparison.png", bbox_inches="tight")
        plt.close(fig)
        log.info("  → TFT02_v1v2_comparison.png")

        log.info("\n=== DELTA TFT v1 → v2 ===")
        for tgt in ["q_next_1d", "q_sum_next_7d"]:
            if tgt not in v1_tft.index or tgt not in v2.index:
                continue
            log.info(f"  [{tgt}]")
            for c in cmp_cols:
                b = v1_tft.loc[tgt, c] if c in v1_tft.columns else np.nan
                n = v2.loc[tgt, c]     if c in v2.columns     else np.nan
                if not (np.isnan(b) or np.isnan(n)):
                    arrow = "▲" if n >= b else "▼"
                    log.info(f"    {c:12s}: {b:.3f} → {n:.3f}  {arrow} {n-b:+.3f}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    log.info("\n=== RESUMEN FINAL TFT v2 ===")
    df_res = pd.DataFrame(all_rows)
    show = [c for c in ["target", "NSE", "NSE_sqrt", "KGE", "J_alert",
                         "POD_det", "POD_prob", "CSI", "Coverage80", "Width_m3s", "N"]
            if c in df_res.columns]
    log.info(df_res[show].to_string(index=False))
    log.info("SCRIPT 74 COMPLETADO")


if __name__ == "__main__":
    main()
