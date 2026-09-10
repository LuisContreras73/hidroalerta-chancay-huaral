#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 190 — ENSEMBLE HÍBRIDO medible: Chronos-2 (fundacional zero-shot) × canónico+GRU (local).
¿Un blend supera a cada modelo por separado? Cierra el argumento operativo con número.

Chronos gana/empata a h1-3 (nowcast), el local gana a h3-14 (subestacional). Un ensemble debería
juntar lo mejor de ambos. Se prueban 4 formas de combinar, con la MISMA vara (test 2024-25):
  · switch   — Chronos si h≤3, local si h≥4 (regla, sin ajuste).
  · mean     — 0.5·Chronos + 0.5·local (cuantiles, sin ajuste).
  · w_opt    — peso por lead que MAXIMIZA NSE en TODO el test (TECHO in-sample, no reclamable).
  · w_cv     — peso por lead ajustado FUERA DE MUESTRA (2-fold temporal: ajusta en mitad, evalúa en
               la otra) → el resultado HONESTO y reclamable.
Métricas: NSE (p50) y CRPS (pinball sobre p10/p50/p90, idéntico a C.crps3). Preds de Chronos ya
guardadas en 137; las del local se generan de los checkpoints 179 (sin re-entrenar).

Salida: outputs/ml_Q/190_ensemble_hibrido.csv · reports/figures/DI_ensemble_hibrido.png
Run (.venv313): python scripts/06_eval/190_ensemble_hibrido.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("hyb190"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; QS = [0.1, 0.5, 0.9]; LEADS = [1, 3, 7, 14]


def crps3(o, qp):
    t = 0.0
    for i, qv in enumerate(QS):
        e = o - qp[:, i]; t += np.mean(np.where(e >= 0, qv * e, (qv - 1) * e))
    return t / 3


def nse(o, p): return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)

# ── preds locales canónico+GRU (ensemble 3 seeds) por (fecha objetivo, lead) ──
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
PR = np.mean(ps, 0)   # (n,H,3)
rl = []
for k, i in enumerate(te):
    for L in LEADS:
        rl.append(dict(date=dts[i + L - 1], lead=L, lp10=PR[k, L - 1, 0], lp50=PR[k, L - 1, 1], lp90=PR[k, L - 1, 2]))
loc = pd.DataFrame(rl)

# ── Chronos-2 (137) ──
ch = pd.read_csv(OUT / "137_chronos2_preds.csv", parse_dates=["date"]).rename(
    columns={"p10": "cp10", "p50": "cp50", "p90": "cp90"})[["date", "lead", "obs", "cp10", "cp50", "cp90"]]
d = ch.merge(loc, on=["date", "lead"], how="inner").dropna()
log.info(f"merge: {len(d)} filas; por lead: {dict(d.groupby('lead').size())}")

C = np.stack([d.cp10, d.cp50, d.cp90], 1); Lc = np.stack([d.lp10, d.lp50, d.lp90], 1)  # (N,3)
o = d.obs.values; lead = d.lead.values; date = d.date.values


def blend(w, mask):  # w escalar, mask booleano de filas
    qp = w * C[mask] + (1 - w) * Lc[mask]; return qp


rows = []
for L in LEADS:
    mk = lead == L; oL = o[mk]; Cs, Ls = C[mk], Lc[mk]
    dts_L = date[mk]
    # modelos base
    base = {"Chronos-2": Cs, "canónico+GRU": Ls}
    # switch: chronos si L<=3 else local
    sw = Cs if L <= 3 else Ls
    # mean
    mn = 0.5 * Cs + 0.5 * Ls
    # w_opt (techo in-sample): maximiza NSE
    ws = np.linspace(0, 1, 21)
    wopt = ws[np.argmax([nse(oL, (w * Cs + (1 - w) * Ls)[:, 1]) for w in ws])]
    op = wopt * Cs + (1 - wopt) * Ls
    # w_cv (fuera de muestra, 2-fold temporal por fecha): ajusta w (min CRPS) en un fold, aplica al otro
    order = np.argsort(dts_L); half = len(order) // 2
    f1 = np.zeros(len(dts_L), bool); f1[order[:half]] = True; f2 = ~f1   # 1ª mitad / 2ª mitad por fecha
    def tune(fit, ap):
        if fit.sum() < 5 or ap.sum() < 1: return 0.5 * Cs[ap] + 0.5 * Ls[ap], np.nan
        wc = ws[np.argmin([crps3(oL[fit], (w * Cs[fit] + (1 - w) * Ls[fit])) for w in ws])]
        return wc * Cs[ap] + (1 - wc) * Ls[ap], wc
    p_f2, w_f1 = tune(f1, f2); p_f1, w_f2 = tune(f2, f1)
    cv = np.zeros_like(Cs); cv[f2] = p_f2; cv[f1] = p_f1
    variants = {**base, "híbrido-switch": sw, "híbrido-mean": mn,
                f"híbrido-w_opt(techo)": op, "híbrido-w_cv(fuera-muestra)": cv}
    for name, qp in variants.items():
        rows.append(dict(modelo=name, h=L, NSE=round(nse(oL, qp[:, 1]), 3), CRPS=round(float(crps3(oL, qp)), 3)))
    log.info(f"h{L:2d}: wopt={wopt:.2f} wcv≈({w_f1:.2f},{w_f2:.2f}) | " +
             " ".join(f"{r['modelo'][:14]}={r['NSE']}" for r in rows if r['h'] == L))
res = pd.DataFrame(rows); res.to_csv(OUT / "190_ensemble_hibrido.csv", index=False)
log.info("\n=== NSE ===\n" + res.pivot_table(index="modelo", columns="h", values="NSE").to_string())
log.info("\n=== CRPS ===\n" + res.pivot_table(index="modelo", columns="h", values="CRPS").to_string())

# ── figura ──
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
STY = {"Chronos-2": ("#7b3fb0", "-s"), "canónico+GRU": ("#2f9e6f", "-o"),
       "híbrido-mean": ("#0e7d90", "--D"), "híbrido-w_cv(fuera-muestra)": ("#d24e39", "-^")}
for mdl, (c, ls) in STY.items():
    s = res[res.modelo == mdl].sort_values("h")
    ax[0].plot(s["h"], s["NSE"], ls, color=c, lw=2.2, ms=6, label=mdl)
    ax[1].plot(s["h"], s["CRPS"], ls, color=c, lw=2.2, ms=6, label=mdl)
for a, ttl, yl in ((ax[0], "(a) NSE (↑ mejor)", "NSE"), (ax[1], "(b) CRPS (↓ mejor)", "CRPS")):
    a.set_xlabel("horizonte (días)"); a.set_ylabel(yl); a.set_xticks(LEADS); a.set_title(ttl, fontsize=11, fontweight="bold")
    a.grid(lw=.3, alpha=.4); a.legend(fontsize=8)
fig.suptitle("Ensemble híbrido Chronos-2 × canónico+GRU — ¿el blend supera a cada uno? (test 2024-25)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "DI_ensemble_hibrido.png", dpi=150)
log.info(f"figura: {FIG/'DI_ensemble_hibrido.png'}"); log.info("ENSEMBLE_HIBRIDO_DONE")
