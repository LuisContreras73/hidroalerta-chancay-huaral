#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 187 — DISTRIBUCIÓN de la atención por paso de lookback (análogo de la Fig. 8 del paper
original: "Distribution of relative attention scores by TFT for previous time steps").

El paper es MENSUAL (lookback 12 meses) y muestra atención ~plana con violines anchos en los extremos.
Nosotros somos DIARIOS (enc=90 días). El 178 solo mostró la atención MEDIA (mapa de calor); aquí
sacamos la DISTRIBUCIÓN sobre las emisiones del test (caja+bigote por bin de lookback, eje log) para
comparar en el mismo formato. Modelo: canónico+GRU (ganador), seed0, test 2024-25.

Atención relativa = att_día / media_sobre_días(att)  → 1 = atención uniforme; >1 sobre-atendido.

Salida: outputs/ml_Q/187_attention_distribution.csv · reports/figures/DI_attention_distribution.png
Run (.venv313): python scripts/05_models/187_attention_distribution.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("attn187"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]

df = R.build(H); dts = df.index; q = df["q"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]

m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                     drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
m.load_state_dict(torch.load(OUT / "179_canonico_GRU_seed0.pt", map_location=DEV)); m.eval()

X = T.seqs_for_enc(df, te, enc, H, (mu, sd))
store = []
h = m.attn.register_forward_hook(lambda mm, i, o: store.append(o[1].detach().cpu()))
with torch.no_grad():
    for s in range(0, len(te), 128):
        sl = slice(s, s + 128); m(X[0][sl].to(DEV), X[1][sl].to(DEV), X[2][sl].to(DEV))
h.remove()
att = torch.cat(store).numpy()                       # (N, L, L), L = enc + H
N, L, _ = att.shape
dec2enc = att[:, enc:, :enc].mean(1)                 # (N, enc): atención media de los H horizontes a cada día pasado
# atención relativa por muestra (uniforme = 1)
rel = dec2enc / (dec2enc.mean(1, keepdims=True) + 1e-12)
lookback_day = np.arange(enc, 0, -1)                 # día 90 (más antiguo) → 1 (emisión)

# bins de lookback (para legibilidad; ~14 cajas como los 12 meses del paper)
edges = [1, 2, 3, 4, 6, 8, 11, 15, 22, 31, 46, 61, enc + 1]
labels_bin, data_bin, stats = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    cols = [k for k, d in enumerate(lookback_day) if a <= d < b]
    if not cols: continue
    vals = rel[:, cols].ravel()
    labels_bin.append(f"{a}" if b - a == 1 else f"{a}-{b-1}")
    data_bin.append(vals)
    stats.append(dict(bin=labels_bin[-1], dia_desde=a, dia_hasta=b - 1, n=len(vals),
                      mediana=round(float(np.median(vals)), 3), p25=round(float(np.percentile(vals, 25)), 3),
                      p75=round(float(np.percentile(vals, 75)), 3), p05=round(float(np.percentile(vals, 5)), 3),
                      p95=round(float(np.percentile(vals, 95)), 3)))
pd.DataFrame(stats).to_csv(OUT / "187_attention_distribution.csv", index=False)
log.info("\n" + pd.DataFrame(stats).to_string())
rec = np.median(rel[:, lookback_day <= 7]) / np.median(rel[:, lookback_day > 60])
log.info(f"ratio recencia: mediana(atención últimos 7d) / mediana(>60d) = {rec:.2f}×")

# ── figura estilo Fig. 8 (caja+bigote, eje log, lookback más antiguo → emisión) ──
fig, ax = plt.subplots(figsize=(11, 5))
pos = np.arange(len(data_bin))
bp_ = ax.boxplot(data_bin, positions=pos, widths=0.6, showfliers=False, patch_artist=True, whis=[5, 95],
                 medianprops=dict(color="#08404a", lw=1.6), whiskerprops=dict(color="#2f6f9e"),
                 capprops=dict(color="#2f6f9e"), boxprops=dict(facecolor="#bcdfe8", edgecolor="#2f6f9e"))
ax.axhline(1.0, color="#c0392b", lw=1.0, ls="--", alpha=.7, label="atención uniforme (=1)")
ax.set_yscale("log"); ax.set_xticks(pos); ax.set_xticklabels(labels_bin, fontsize=9)
ax.set_xlabel("Lookback (días antes de la emisión) — más antiguo ← → reciente", fontsize=10)
ax.set_ylabel("Atención relativa (uniforme = 1; bigotes = p5–p95)", fontsize=10)
p95_rec = np.percentile(rel[:, lookback_day <= 3], 95); p95_old = np.percentile(rel[:, lookback_day > 60], 95)
ax.set_title("Distribución de atención del TFT por paso de lookback — canónico+GRU (test 2024-25)\n"
             f"MEDIANA casi plana ({rec:.2f}×, como el paper mensual); la recencia vive en la COLA: "
             f"p95 día 1-3 ≈ {p95_rec:.1f} vs >60 d ≈ {p95_old:.1f}", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", which="both", lw=.3, alpha=.4)
fig.tight_layout(); fig.savefig(FIG / "DI_attention_distribution.png", dpi=150)
log.info(f"figura: {FIG/'DI_attention_distribution.png'}")
log.info("ATTENTION_DIST_DONE")
