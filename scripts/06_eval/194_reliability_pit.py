#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 194 — Verificación probabilística de ALERTA que faltaba: DIAGRAMA DE CONFIABILIDAD (reliability)
de P(Q≥Q90) e histograma PIT, + CRPS de cola (obs≥Q80), para el ganador canónico+GRU y Chronos-2.
Cierra parte del hueco del red-team (147 ya tenía AUC/BSS; faltaban reliability + PIT).

De los 3 cuantiles (p10,p50,p90) se reconstruye una CDF predictiva por interpolación lineal (con colas
extrapoladas por la pendiente del segmento extremo). Con ella:
  · PIT_i = F(obs_i)  → histograma; uniforme ⇒ calibrado.
  · π_i = 1 − F(Q90)  → prob. de excedencia; binned vs frecuencia observada = reliability.
  · CRPS_cola = CRPS medio SOLO en días obs≥Q80 (métrica de cola robusta con 3 cuantiles).
NOTA honesta: un twCRPS pleno requiere cuantiles densos (Fase A densq99); aquí se da la versión de cola
computable con 3 cuantiles.

Salida: outputs/ml_Q/194_reliability_pit.csv · reports/figures/DI_reliability_pit.png
Run (.venv313): python scripts/06_eval/194_reliability_pit.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rel194"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90; QLV = np.array([0.1, 0.5, 0.9]); LEADS = [1, 7, 14]


def cdf_at(qp, y):
    """F_pred(y) desde 3 cuantiles (n,3) a valores y (n,). Interp lineal + colas por pendiente extrema."""
    p10, p50, p90 = qp[:, 0], qp[:, 1], qp[:, 2]
    F = np.empty_like(y, dtype=float)
    s_lo = 0.4 / np.maximum(p50 - p10, 1e-6); s_hi = 0.4 / np.maximum(p90 - p50, 1e-6)
    below = y <= p10; mid1 = (y > p10) & (y <= p50); mid2 = (y > p50) & (y <= p90); above = y > p90
    F[below] = 0.1 + s_lo[below] * (y[below] - p10[below])
    F[mid1] = 0.1 + (0.4 / np.maximum(p50 - p10, 1e-6))[mid1] * (y[mid1] - p10[mid1])
    F[mid2] = 0.5 + (0.4 / np.maximum(p90 - p50, 1e-6))[mid2] * (y[mid2] - p50[mid2])
    F[above] = 0.9 + s_hi[above] * (y[above] - p90[above])
    return np.clip(F, 1e-3, 1 - 1e-3)


def crps_c(o, qp):
    t = np.zeros(len(o))
    for i, qv in enumerate(QLV):
        e = o - qp[:, i]; t += np.where(e >= 0, qv * e, (qv - 1) * e)
    return t / 3


# ── datos + ganador ──
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
Q80 = float(np.nanpercentile(q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]], 80))

models = []
for s in range(3):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT / f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)
X = T.seqs_for_enc(df, te, enc, H, (mu, sd)); ps = []
for m in models:
    with torch.no_grad(): ps.append(m(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
PRg = np.mean(ps, 0)
ch = pd.read_csv(OUT / "137_chronos2_preds.csv", parse_dates=["date"])

def preds(model, Lh):
    o = np.array([obs[i + Lh - 1] for i in te])
    if model == "canónico+GRU":
        qp = PRg[:, Lh - 1, :]; dd = np.array([dts[i + Lh - 1] for i in te])
    else:
        sub = ch[ch.lead == Lh].set_index("date"); dd = np.array([dts[i + Lh - 1] for i in te])
        qp = np.stack([[sub["p10"].get(t, np.nan), sub["p50"].get(t, np.nan), sub["p90"].get(t, np.nan)] for t in dd], 0)
    m = np.isfinite(o) & np.isfinite(qp).all(1)
    return o[m], qp[m]


rows = []; pit_store = {}; rel_store = {}
for model in ["canónico+GRU", "Chronos-2"]:
    for Lh in LEADS:
        o, qp = preds(model, Lh)
        pit = cdf_at(qp, o)
        pi = 1 - cdf_at(qp, np.full(len(o), Q90))          # prob de exceder Q90
        exc = (o >= Q90).astype(float)
        # reliability por bins de prob
        bins = np.linspace(0, 1, 6); idx = np.clip(np.digitize(pi, bins) - 1, 0, len(bins) - 2)
        rel = []
        for b in range(len(bins) - 1):
            mb = idx == b
            if mb.sum() >= 3: rel.append((float(pi[mb].mean()), float(exc[mb].mean()), int(mb.sum())))
        rel_store[(model, Lh)] = rel; pit_store[(model, Lh)] = pit
        # métricas de calibración
        pit_ks = float(np.max(np.abs(np.sort(pit) - (np.arange(1, len(pit) + 1) / len(pit)))))  # KS vs uniforme
        cov = float(((o >= qp[:, 0]) & (o <= qp[:, 2])).mean())   # cobertura banda p10-p90 (nominal 0.8)
        crps_all = float(crps_c(o, qp).mean())
        tail = o >= Q80; crps_tail = float(crps_c(o[tail], qp[tail]).mean()) if tail.sum() >= 3 else np.nan
        rows.append(dict(model=model, h=Lh, N=len(o), cobertura_p10p90=round(cov, 3), PIT_KS=round(pit_ks, 3),
                         CRPS=round(crps_all, 3), CRPS_cola_Q80=round(crps_tail, 3), n_cola=int(tail.sum())))
        log.info(f"{model:13s} h{Lh:2d}: cobertura p10-p90={cov:.2f} (nominal 0.80) · PIT_KS={pit_ks:.3f} · CRPS={crps_all:.3f} · CRPS_cola={crps_tail:.3f}")

res = pd.DataFrame(rows); res.to_csv(OUT / "194_reliability_pit.csv", index=False)

# ── figura: reliability (h7,h14) + PIT (h14) ──
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
COL = {"canónico+GRU": "#2f9e6f", "Chronos-2": "#7b3fb0"}
for k, Lh in enumerate([7, 14]):
    ax[k].plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfecto")
    for model in COL:
        rel = rel_store.get((model, Lh), [])
        if rel:
            xs = [r[0] for r in rel]; ys = [r[1] for r in rel]
            ax[k].plot(xs, ys, "-o", color=COL[model], lw=1.8, ms=5, label=model)
    ax[k].set_xlabel("prob. pronosticada P(Q≥Q90)"); ax[k].set_ylabel("frecuencia observada")
    ax[k].set_title(f"(a{k+1}) Confiabilidad — h{Lh}", fontsize=10.5, fontweight="bold")
    ax[k].legend(fontsize=8); ax[k].grid(lw=.3, alpha=.4); ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1)
# PIT hist h14 ganador
p = pit_store.get(("canónico+GRU", 14), np.array([]))
ax[2].hist(p, bins=10, range=(0, 1), color="#2f9e6f", alpha=0.8, edgecolor="white")
ax[2].axhline(len(p) / 10, color="#c0392b", ls="--", lw=1, label="uniforme (calibrado)")
ax[2].set_xlabel("PIT = F_pred(obs)"); ax[2].set_ylabel("frecuencia")
ax[2].set_title("(b) Histograma PIT — canónico+GRU h14", fontsize=10.5, fontweight="bold"); ax[2].legend(fontsize=8)
fig.suptitle("Verificación probabilística de alerta: confiabilidad + PIT (test 2024-25)", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "DI_reliability_pit.png", dpi=150)
log.info(f"figura: {FIG/'DI_reliability_pit.png'}"); log.info("RELIABILITY_PIT_DONE")
