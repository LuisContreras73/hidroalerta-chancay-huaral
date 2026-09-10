#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 191 — RIGOR del ensemble híbrido, condicionado a EVENTOS + SIGNIFICANCIA + ALERTA.
Responde a: "¿testeaste bien el híbrido? ¿en eventos de lluvia? ¿es significativa la ganancia?"

Sobre las preds alineadas (Chronos-2 de 137 × canónico+GRU de checkpoints 179), test 2024-25:
  (A) Desempeño por RÉGIMEN: evento (obs objetivo ≥ Q90), húmeda (ene-abr) y calma/seca — CRPS y MAE
      (robustos en submuestras chicas; NSE es inestable con poca varianza).
  (B) SIGNIFICANCIA de la ganancia: ΔCRPS por-muestra (híbrido − mejor individual) con IC 95 %
      block-bootstrap (bloque = decorrelación) → estilo Diebold-Mariano. ¿El IC queda < 0?
  (C) ALERTA: CSI/POD/FAR al umbral Q90 por lead, para Chronos / local / híbrido.

Híbrido = media de cuantiles 0.5·Chronos + 0.5·local (SIN ajuste → sin sobreajuste para la significancia).

Salida: outputs/ml_Q/191_hibrido_rigor.csv · reports/figures/DI_hibrido_rigor_eventos.png
Run (.venv313): python scripts/06_eval/191_hibrido_rigor_eventos.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rig191"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90; QS = [0.1, 0.5, 0.9]; LEADS = [1, 3, 7, 14]; rng = np.random.default_rng(0)


def crps_contrib(o, qp):          # (n,) CRPS por muestra (pinball medio sobre 3 cuantiles)
    t = np.zeros(len(o))
    for i, qv in enumerate(QS):
        e = o - qp[:, i]; t += np.where(e >= 0, qv * e, (qv - 1) * e)
    return t / 3


def nse(o, p): return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def lag1(x):
    x = x[np.isfinite(x)]; x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if len(x) > 2 else 0.0


def block_ci_mean(x, blk, reps=2000):     # IC95 de la media por moving-block bootstrap
    n = len(x); nb = int(np.ceil(n / blk)); smax = max(1, n - blk + 1); out = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, smax, nb)])[:n] % n
        out.append(x[idx].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ── preds locales canónico+GRU (ensemble 3 seeds) ──
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2023-12-01")]
models = []
for s in range(3):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                         drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT / f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)
X = T.seqs_for_enc(df, te, enc, H, (mu, sd)); ps = []
for m in models:
    with torch.no_grad(): ps.append(m(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
PR = np.mean(ps, 0)
rl = [dict(date=dts[i + L - 1], lead=L, lp10=PR[k, L - 1, 0], lp50=PR[k, L - 1, 1], lp90=PR[k, L - 1, 2])
      for k, i in enumerate(te) for L in LEADS]
loc = pd.DataFrame(rl)
ch = pd.read_csv(OUT / "137_chronos2_preds.csv", parse_dates=["date"]).rename(
    columns={"p10": "cp10", "p50": "cp50", "p90": "cp90"})[["date", "lead", "obs", "cp10", "cp50", "cp90"]]
d = ch.merge(loc, on=["date", "lead"], how="inner").dropna().sort_values(["lead", "date"]).reset_index(drop=True)
mth = pd.to_datetime(d.date).dt.month

rows, sig = [], []
for L in LEADS:
    s = (d.lead == L).values
    o = d.obs.values[s]; C = np.stack([d.cp10, d.cp50, d.cp90], 1)[s]; Lc = np.stack([d.lp10, d.lp50, d.lp90], 1)[s]
    Hy = 0.5 * C + 0.5 * Lc; wet = np.isin(mth.values[s], [1, 2, 3, 4]); ev = o >= Q90
    cc, cl, chy = crps_contrib(o, C), crps_contrib(o, Lc), crps_contrib(o, Hy)
    MODS = {"Chronos-2": (C, cc), "canónico+GRU": (Lc, cl), "híbrido-mean": (Hy, chy)}
    for reg, mk in [("global", np.ones_like(o, bool)), ("húmeda(ene-abr)", wet),
                    ("calma/seca", ~wet), (f"evento(≥Q90,n={int(ev.sum())})", ev)]:
        if mk.sum() < 3: continue
        for nm, (qp, cci) in MODS.items():
            oa, pa = o[mk] >= Q90, qp[mk, 1] >= Q90
            tp, fp, fn = int(np.sum(oa & pa)), int(np.sum(~oa & pa)), int(np.sum(oa & ~pa))
            rows.append(dict(lead=L, regimen=reg, modelo=nm, n=int(mk.sum()),
                             CRPS=round(float(cci[mk].mean()), 3), MAE=round(float(np.mean(np.abs(o[mk] - qp[mk, 1]))), 2),
                             sesgo=round(float(np.mean(qp[mk, 1] - o[mk])), 2),
                             NSE=round(nse(o[mk], qp[mk, 1]), 3) if mk.sum() > 5 and o[mk].std() > 1e-6 else np.nan,
                             CSI=round(tp / (tp + fp + fn), 3) if (tp + fp + fn) else 0.0,
                             POD=round(tp / (tp + fn), 3) if (tp + fn) else np.nan))
    # significancia: mejor individual por CRPS
    best_c = cc if cc.mean() <= cl.mean() else cl; best_nm = "Chronos" if cc.mean() <= cl.mean() else "local"
    diff = chy - best_c                    # <0 = híbrido mejor
    blk = max(1, round(-1 / np.log(min(max(lag1(diff), 1e-3), 0.999))))
    lo, hi = block_ci_mean(diff, blk)
    sig.append(dict(lead=L, mejor_individual=best_nm, dCRPS=round(float(diff.mean()), 4),
                    IC_lo=round(lo, 4), IC_hi=round(hi, 4), signif=("SÍ" if hi < 0 else "no")))
    log.info(f"h{L:2d}: ΔCRPS(híbrido−{best_nm})={diff.mean():+.4f} IC95=[{lo:+.4f},{hi:+.4f}] "
             f"{'SIGNIF' if hi<0 else 'no signif'} | evento n={int(ev.sum())}")
res = pd.DataFrame(rows); res.to_csv(OUT / "191_hibrido_rigor.csv", index=False)
sg = pd.DataFrame(sig); sg.to_csv(OUT / "191_hibrido_significancia.csv", index=False)
log.info("\n=== CRPS por régimen (h1 y h14) ===\n" +
         res[res.lead.isin([1, 14])].pivot_table(index=["lead", "regimen"], columns="modelo", values="CRPS").to_string())
log.info("\n=== significancia ΔCRPS ===\n" + sg.to_string())

# ── figura ──
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
regs = ["calma/seca", "húmeda(ene-abr)", f"evento(≥Q90,n={int((d.obs.values[(d.lead==1).values]>=Q90).sum())})"]
COL = {"Chronos-2": "#7b3fb0", "canónico+GRU": "#2f9e6f", "híbrido-mean": "#0e7d90"}
sub = res[(res.lead == 7) & (res.regimen.isin(regs))]
x = np.arange(len(regs)); w = 0.26
for j, (nm, c) in enumerate(COL.items()):
    vals = [sub[(sub.regimen == r) & (sub.modelo == nm)]["CRPS"].values[0] if not sub[(sub.regimen == r) & (sub.modelo == nm)].empty else 0 for r in regs]
    ax[0].bar(x + (j - 1) * w, vals, w, label=nm, color=c)
ax[0].set_xticks(x); ax[0].set_xticklabels(["calma/seca", "húmeda", "evento ≥Q90"], fontsize=9)
ax[0].set_ylabel("CRPS (↓ mejor)"); ax[0].set_title("(a) CRPS por régimen — h7", fontsize=11, fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(axis="y", lw=.3, alpha=.4)
ax[1].axhline(0, color="#c0392b", lw=1, ls="--", alpha=.7)
xx = np.arange(len(LEADS)); dv = [s["dCRPS"] for s in sig]
er = [[s["dCRPS"] - s["IC_lo"] for s in sig], [s["IC_hi"] - s["dCRPS"] for s in sig]]
cols = ["#0e7d90" if s["signif"] == "SÍ" else "#b8860b" for s in sig]
ax[1].errorbar(xx, dv, yerr=er, fmt="o", capsize=5, color="#333", ecolor="#888", zorder=1)
ax[1].scatter(xx, dv, c=cols, s=70, zorder=2)
ax[1].set_xticks(xx); ax[1].set_xticklabels([f"h{L}" for L in LEADS]); ax[1].set_ylabel("ΔCRPS  híbrido − mejor individual")
ax[1].set_title("(b) ¿Es SIGNIFICATIVA la ganancia? (IC95 block-boot)\nverde=IC<0 (sí) · <0 = híbrido mejor", fontsize=10.5, fontweight="bold")
ax[1].grid(axis="y", lw=.3, alpha=.4)
fig.suptitle("Rigor del híbrido: desempeño por evento + significancia de la ganancia (test 2024-25)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "DI_hibrido_rigor_eventos.png", dpi=150)
log.info(f"figura: {FIG/'DI_hibrido_rigor_eventos.png'}"); log.info("HIBRIDO_RIGOR_DONE")
