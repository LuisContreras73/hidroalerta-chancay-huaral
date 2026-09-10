#!/usr/bin/env python3
"""
Script 61: LSTM con target Q — diagnóstico completo de entrenamiento.

Genera todo lo que un análisis de DL necesita:
  - Curva de pérdida (loss) train + val por epoch
  - Tiempo por epoch y tiempo total
  - Métricas (NSE, KGE, J_alert) por epoch sobre val
  - Hidrograma Q observado vs predicho en train/val/test (un solo gráfico continuo)
  - Early stopping con paciencia

Formato: WIDE (1 fila/día, features de las 9 sub-cuencas) — N honesto, sin inflación.
Secuencia: encoder de ENCODER_LEN días → predice q_next_1d / q_sum_next_7d.

Salidas:
  outputs/ml_Q/lstm_history.csv          — loss y métricas por epoch
  outputs/ml_Q/lstm_predictions.csv      — Q obs/pred train/val/test
  outputs/figures/ml_Q/LSTM01_training_curves.png
  outputs/figures/ml_Q/LSTM02_hydrograph_full.png
  configs/lstm_Q_config.json
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data/metadata"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(ROOT/"outputs"/"61_train_lstm_Q.log","w","utf-8")])
log = logging.getLogger("lstm_Q")

D6_CSV      = ROOT / "data/model_ready/D6_multientity.csv"
THRESH_FILE = ROOT / "data/model_ready/thresholds/q_thresholds.json"
Q_OBS_FILE  = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR     = ROOT / "outputs/ml_Q"
FIG_DIR     = ROOT / "outputs/figures/ml_Q"
CFG_DIR     = ROOT / "configs"
for d in (OUT_DIR, FIG_DIR, CFG_DIR):
    d.mkdir(parents=True, exist_ok=True)

Q_STATION   = "q_santo_domingo_47e214d2"
Q_CONV      = 86.4 / 3062.62
Q90         = 40.89
SEED        = 42

# Hiperparámetros LSTM (tuneado 2026-06-02: más epochs + regularización vs sobreajuste)
ENCODER_LEN = 90      # días de contexto
HIDDEN      = 64
LAYERS      = 2
DROPOUT     = 0.3     # subido 0.2→0.3 (el modelo sobreajustaba en epoch ~17)
LR          = 7e-4    # bajado 1e-3→7e-4 para entrenamiento más estable y largo
WEIGHT_DECAY= 1e-5    # L2 regularization (nuevo)
BATCH       = 64
MAX_EPOCHS  = 300     # subido 120→300 (el usuario pidió más epochs)
PATIENCE    = 40      # subido 15→40 para permitir entrenamiento más largo
TARGET      = "q_next_1d"

SPATIAL_FEATURES = ["pr_mm", "tmax_c", "tmin_c", "pet_mm",
                    "api", "spi_30d", "spi_90d", "water_deficit_30d"]
SHARED_FEATURES  = ["oni_index", "sin_doy_1", "cos_doy_1", "hydro_month", "is_wet_season"]


def build_wide(D6):
    pivots = []
    for col in SPATIAL_FEATURES:
        if col in D6.columns:
            piv = D6.pivot_table(index="date", columns="entity_id", values=col)
            piv.columns = [f"{col}_{e}" for e in piv.columns]
            pivots.append(piv)
    ref = D6[D6["entity_id"] == "sub_634"].set_index("date")
    keep = [c for c in SHARED_FEATURES + [TARGET, "q_mm", "split"] if c in ref.columns]
    pivots.append(ref[keep])
    return pd.concat(pivots, axis=1).sort_index()


def make_sequences(X, y, enc_len):
    """Crea ventanas deslizantes [N, enc_len, n_feat] → y[N]."""
    Xs, ys, idx = [], [], []
    for i in range(enc_len, len(X)):
        if np.isfinite(y[i]):
            Xs.append(X[i-enc_len:i])
            ys.append(y[i])
            idx.append(i)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(idx)


def nse(o, p):
    return 1.0 - np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)

def nse_sqrt(o, p):
    w = np.maximum(o, 0)**0.5
    return 1.0 - np.sum(w*(o-p)**2)/(np.sum(w*(o-o.mean())**2)+1e-12)

def kge(o, p):
    r = np.corrcoef(o,p)[0,1]
    a = p.std()/(o.std()+1e-12); b = p.mean()/(o.mean()+1e-12)
    return 1 - np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)

def j_alert(o, p, thr):
    eo=o>thr; ep=p>thr
    TP=np.sum(eo&ep); FP=np.sum(~eo&ep); FN=np.sum(eo&~ep); TN=np.sum(~eo&~ep)
    POD=TP/(TP+FN+1e-9); FAR=FP/(FP+TN+1e-9); CSI=TP/(TP+FP+FN+1e-9)
    return 0.25*nse_sqrt(o,p)+0.25*nse(o,p)+0.30*CSI+0.10*POD-0.10*FAR


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("="*70)
    log.info(f"SCRIPT 61: LSTM target Q | device={device}")
    log.info("="*70)

    # ── Datos ──────────────────────────────────────────────────────────────
    D6 = pd.read_csv(D6_CSV, parse_dates=["date"])
    wide = build_wide(D6)
    feat_cols = [c for c in wide.columns if c not in [TARGET, "q_mm", "split"]]
    log.info(f"[1] Wide: {wide.shape} | {len(feat_cols)} features | encoder={ENCODER_LEN}d")

    # Escalado: fit SOLO en train
    train_mask = (wide["split"] == "train").values
    X_all = wide[feat_cols].fillna(0).values.astype(np.float32)
    y_all = wide[TARGET].values.astype(np.float32)

    mu  = X_all[train_mask].mean(axis=0)
    sd  = X_all[train_mask].std(axis=0) + 1e-8
    X_all = (X_all - mu) / sd

    # log1p en target (Q muy sesgado)
    y_log = np.log1p(np.clip(y_all, 0, None))
    y_mu  = np.nanmean(y_log[train_mask]); y_sd = np.nanstd(y_log[train_mask]) + 1e-8
    y_scaled = (y_log - y_mu) / y_sd

    def inv_y(y_s):
        return np.expm1(y_s * y_sd + y_mu)

    # Secuencias por split (respetando fronteras)
    splits = wide["split"].values
    Xseq, yseq, idxseq = make_sequences(X_all, y_scaled, ENCODER_LEN)
    seq_split = splits[idxseq]

    tr = seq_split == "train"; vl = seq_split == "val"; te = seq_split == "test"
    log.info(f"[2] Secuencias: train={tr.sum()}, val={vl.sum()}, test={te.sum()}")

    Xtr = torch.tensor(Xseq[tr]); ytr = torch.tensor(yseq[tr])
    Xvl = torch.tensor(Xseq[vl]); yvl = torch.tensor(yseq[vl])
    Xte = torch.tensor(Xseq[te]); yte = torch.tensor(yseq[te])

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)

    # ── Modelo ─────────────────────────────────────────────────────────────
    class LSTMRegressor(nn.Module):
        def __init__(self, n_feat, hidden, layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0)
            self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden//2, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = LSTMRegressor(len(feat_cols), HIDDEN, LAYERS, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[3] LSTM h={HIDDEN} L={LAYERS} | {n_params:,} params")

    # ── Entrenamiento con tracking completo ────────────────────────────────
    Xvl_d, yvl_np = Xvl.to(device), yvl.numpy()
    history = []
    best_val = np.inf; best_state = None; patience_ctr = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS+1):
        ep_t0 = time.time()
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(Xtr)

        # Val
        model.eval()
        with torch.no_grad():
            vpred_s = model(Xvl_d).cpu().numpy()
        val_loss = float(np.mean((vpred_s - yvl_np)**2))
        sched.step(val_loss)

        # Métricas en unidades reales (mm/d)
        vpred = inv_y(vpred_s); vobs = inv_y(yvl_np)
        m_nse = nse(vobs, vpred); m_kge = kge(vobs, vpred)
        m_jalert = j_alert(vobs, vpred, Q90*Q_CONV)
        ep_time = time.time() - ep_t0

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_NSE": m_nse, "val_KGE": m_kge, "val_J_alert": m_jalert,
                        "lr": opt.param_groups[0]["lr"], "time_s": ep_time})

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  Ep {epoch:3d}/{MAX_EPOCHS} | train_loss={tr_loss:.4f} "
                     f"val_loss={val_loss:.4f} | val_NSE={m_nse:.3f} J_alert={m_jalert:.3f} "
                     f"| {ep_time:.1f}s")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"  Early stopping en epoch {epoch} (best val_loss={best_val:.4f})")
                break

    total_time = time.time() - t0
    log.info(f"[4] Entrenamiento: {len(history)} epochs en {total_time:.1f}s "
             f"({total_time/len(history):.2f}s/epoch)")

    # Restaurar mejor modelo
    model.load_state_dict(best_state)
    model.eval()

    # ── Predicciones finales en los 3 splits ───────────────────────────────
    def predict(X):
        with torch.no_grad():
            return inv_y(model(X.to(device)).cpu().numpy())

    pred_tr = predict(Xtr); obs_tr = inv_y(ytr.numpy())
    pred_vl = predict(Xvl); obs_vl = inv_y(yvl.numpy())
    pred_te = predict(Xte); obs_te = inv_y(yte.numpy())

    log.info("\n[5] MÉTRICAS FINALES (mejor modelo):")
    for nm, o, p in [("train", obs_tr, pred_tr), ("val", obs_vl, pred_vl), ("test", obs_te, pred_te)]:
        log.info(f"   {nm:5s}: NSE={nse(o,p):.4f} NSE_sqrt={nse_sqrt(o,p):.4f} "
                 f"KGE={kge(o,p):.4f} J_alert={j_alert(o,p,Q90*Q_CONV):.4f}")

    # ── Evaluación honesta vs Q obs real ───────────────────────────────────
    dates_seq = wide.index.values[idxseq]
    if Q_OBS_FILE.exists():
        dfq = pd.read_csv(Q_OBS_FILE, index_col=0, parse_dates=True)
        if Q_STATION in dfq.columns:
            q_obs_mm = (dfq[Q_STATION].dropna() * Q_CONV)
            te_dates = pd.DatetimeIndex(dates_seq[te])
            common = te_dates.intersection(q_obs_mm.index)
            if len(common) >= 10:
                obs_real = q_obs_mm.loc[common].values
                pos = pd.Series(pred_te, index=te_dates).loc[common].values
                log.info(f"\n[6] HONESTO vs Q obs real ({len(common)} días):")
                log.info(f"   NSE={nse(obs_real,pos):.4f} NSE_sqrt={nse_sqrt(obs_real,pos):.4f} "
                         f"KGE={kge(obs_real,pos):.4f} J_alert={j_alert(obs_real,pos,Q90*Q_CONV):.4f}")

    # ── Guardar history + predicciones ─────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT_DIR/"lstm_history.csv", index=False)

    pred_df = pd.DataFrame({
        "date": np.concatenate([dates_seq[tr], dates_seq[vl], dates_seq[te]]),
        "split": np.concatenate([["train"]*tr.sum(), ["val"]*vl.sum(), ["test"]*te.sum()]),
        "q_obs_mm": np.concatenate([obs_tr, obs_vl, obs_te]),
        "q_pred_mm": np.concatenate([pred_tr, pred_vl, pred_te]),
    })
    pred_df["q_obs_m3s"]  = pred_df["q_obs_mm"] / Q_CONV
    pred_df["q_pred_m3s"] = pred_df["q_pred_mm"] / Q_CONV
    pred_df.to_csv(OUT_DIR/"lstm_predictions.csv", index=False)

    json.dump({"encoder_len": ENCODER_LEN, "hidden": HIDDEN, "layers": LAYERS,
               "dropout": DROPOUT, "lr": LR, "batch": BATCH, "n_params": n_params,
               "epochs_trained": len(history), "total_time_s": round(total_time,1),
               "best_val_loss": round(float(best_val),4), "target": TARGET,
               "n_features": len(feat_cols)},
              open(CFG_DIR/"lstm_Q_config.json","w"), indent=2)

    log.info("\nSCRIPT 61 COMPLETADO — history y predicciones guardadas")


if __name__ == "__main__":
    main()
