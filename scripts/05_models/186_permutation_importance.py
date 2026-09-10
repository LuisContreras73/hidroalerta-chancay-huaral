#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 186 — Importancia de variables por PERMUTACIÓN (agnóstica al modelo), estilo del paper
original (barras horizontales ordenadas, eje log). Complementa la VSN (nativa del TFT) con la
importancia estándar que esperan los revisores Q1.

Método: sobre el ganador canónico+GRU (ensemble 3 seeds, sin re-entrenar), para cada feature del
encoder se PERMUTA esa columna entre muestras del test (rompe su relación con el target preservando
su distribución marginal y autocorrelación temporal) y se mide la CAÍDA de skill:
    ΔNSE = NSE_base − NSE_permutado   (grande = feature importante)
    ΔCRPS = CRPS_permutado − CRPS_base
Se promedia sobre n_rep permutaciones. Se compara contra el peso VSN → si coinciden, la
interpretabilidad es robusta.

Panel (a) = importancia por permutación (ΔNSE, h14). Panel (b) = peso VSN. NO hay panel de atributos
ESTÁTICOS: somos single-basin (los estáticos no varían en una cuenca) — eso llega en el 2º paper
multi-cuenca (CAMELS-PE). Se declara en el título.

Salida: outputs/ml_Q/186_permutation_importance.csv · reports/figures/DI_permutation_importance.png
Run (.venv313): python scripts/05_models/186_permutation_importance.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("perm186"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV, C = M.R, M.T, M.DEV, M.C
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; NREP = 8; rng = np.random.default_rng(0)

df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]

models = []
for s in range(3):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                         drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT / f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)
labels = [str(c) for c in R.PAST] + ["q_pasado"]          # 9 PAST + q_pasado
NP = len(R.PAST)

X = T.seqs_for_enc(df, te, enc, H, (mu, sd))               # (x_past, q_past, x_fut, y)
xp0, qp0, xf0 = X[0].clone(), X[1].clone(), X[2].clone()
o14 = np.array([obs[i + 13] for i in te]); o1 = np.array([obs[i] for i in te])


def pred_ens(xp, qp, xf):
    ps = []
    for m in models:
        with torch.no_grad():
            ps.append(m(xp.to(DEV), qp.to(DEV), xf.to(DEV)).cpu().numpy())
    return np.mean(ps, 0)


def skill(PR, o, h):
    p = PR[:, h - 1, :]; mm = np.isfinite(o) & np.isfinite(p[:, 1])
    return C.met(o[mm], p[mm])["NSE"], C.met(o[mm], p[mm])["CRPS"]


PRb = pred_ens(xp0, qp0, xf0)
base14 = skill(PRb, o14, 14); base1 = skill(PRb, o1, 1)
log.info(f"baseline: h1 NSE={base1[0]} CRPS={base1[1]} | h14 NSE={base14[0]} CRPS={base14[1]}")

rows = []
B = xp0.shape[0]
for j, lab in enumerate(labels):
    d14n, d14c, d1n = [], [], []
    for _ in range(NREP):
        perm = rng.permutation(B)
        xp, qp = xp0.clone(), qp0.clone()
        if j < NP:
            xp[:, :, j] = xp0[perm, :, j]           # permuta la columna-feature entre muestras
        else:
            qp[:] = qp0[perm]                        # permuta el canal autorregresivo q_pasado
        PR = pred_ens(xp, qp, xf0)
        n14, c14 = skill(PR, o14, 14); n1, _ = skill(PR, o1, 1)
        d14n.append(base14[0] - n14); d14c.append(c14 - base14[1]); d1n.append(base1[0] - n1)
    rows.append(dict(feature=lab, dNSE_h14=round(float(np.mean(d14n)), 4), dNSE_h14_std=round(float(np.std(d14n)), 4),
                     dCRPS_h14=round(float(np.mean(d14c)), 4), dNSE_h1=round(float(np.mean(d1n)), 4)))
    log.info(f"  {lab:12s} ΔNSE_h14={rows[-1]['dNSE_h14']:+.4f}  ΔNSE_h1={rows[-1]['dNSE_h1']:+.4f}  ΔCRPS_h14={rows[-1]['dCRPS_h14']:+.4f}")

# VSN (peso nativo) para el panel b — hook sobre el encoder, media test
store = []
h = models[0].vsn_past.register_forward_hook(lambda mm, i, o: store.append(o[1].detach().cpu()))
with torch.no_grad():
    for s0 in range(0, B, 256):
        sl = slice(s0, s0 + 256); models[0](xp0[sl].to(DEV), qp0[sl].to(DEV), xf0[sl].to(DEV))
h.remove()
vsn = torch.cat(store).mean((0, 1)).numpy()
for r in rows:
    r["VSN"] = round(float(vsn[labels.index(r["feature"])]), 4)

dd = pd.DataFrame(rows); dd.to_csv(OUT / "186_permutation_importance.csv", index=False)
log.info("\n" + dd.to_string())

# ── figura estilo paper: 2 paneles horizontales, ordenados, eje log ──
FLOOR = 1e-4
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
d_a = dd.sort_values("dNSE_h14")
va = np.clip(d_a["dNSE_h14"].values, FLOOR, None)
ax[0].barh(np.arange(len(d_a)), va, color="#2f6f9e")
ax[0].set_yticks(np.arange(len(d_a))); ax[0].set_yticklabels(d_a["feature"], fontsize=9)
ax[0].set_xscale("log"); ax[0].set_xlabel("Importancia por permutación  (ΔNSE, h14)", fontsize=10)
ax[0].set_title("(a) Forzantes dinámicas — modelo-agnóstico", fontsize=11, fontweight="bold")
ax[0].grid(axis="x", which="both", lw=.3, alpha=.4)
d_b = dd.sort_values("VSN")
vb = np.clip(d_b["VSN"].values, FLOOR, None)
ax[1].barh(np.arange(len(d_b)), vb, color="#c85a2e")
ax[1].set_yticks(np.arange(len(d_b))); ax[1].set_yticklabels(d_b["feature"], fontsize=9)
ax[1].set_xscale("log"); ax[1].set_xlabel("Peso de selección VSN (nativo del TFT)", fontsize=10)
ax[1].set_title("(b) Misma forzante, vista VSN", fontsize=11, fontweight="bold")
ax[1].grid(axis="x", which="both", lw=.3, alpha=.4)
fig.suptitle("Importancia de variables — canónico+GRU (test 2024-25). Single-basin → sin panel de "
             "atributos estáticos (llega en el 2º paper multi-cuenca)", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "DI_permutation_importance.png", dpi=150)
log.info(f"figura: {FIG/'DI_permutation_importance.png'}")
log.info("PERM_IMPORTANCE_DONE")
