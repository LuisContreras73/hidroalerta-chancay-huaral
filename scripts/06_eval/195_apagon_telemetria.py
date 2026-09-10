#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 195 — EXPERIMENTO DE APAGÓN DE TELEMETRÍA (mide la 'compuerta de disponibilidad' que hasta hoy
solo se afirmaba). El insumo dominante es q_past (telemetría con ~77% de vacíos en 2025, que se cae
JUSTO en crecida). Aquí se simula el apagón congelando (carry-forward) los últimos k días del canal
q_past del encoder y se MIDE la degradación de NSE/CRPS/POD del ganador canónico+GRU — global y en
ventanas de EVENTO (horizonte con Q≥Q90) — más el modo degradado (persistencia/climatología).

Salida: outputs/ml_Q/195_apagon_telemetria.csv · reports/figures/DI_apagon_telemetria.png
Run (.venv313): python scripts/06_eval/195_apagon_telemetria.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("apg195"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90; QS = [0.1, 0.5, 0.9]; LEADS = [1, 7, 14]
OUTAGES = [0, 1, 3, 7, 14, 30]   # días recientes de q_past congelados (0 = telemetría completa)

def crps_c(o, qp):
    t = np.zeros(len(o))
    for i, qv in enumerate(QS):
        e = o - qp[:, i]; t += np.where(e >= 0, qv * e, (qv - 1) * e)
    return t / 3
def nse(o, p): return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)

df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
doy = dts.dayofyear.values
clim = np.array([np.nanmean(obs[(doy == d) & (dts.values <= np.datetime64("2022-12-31")) & np.isfinite(obs)]) if
                 np.isfinite(obs[(doy == d) & (dts.values <= np.datetime64("2022-12-31"))]).any() else np.nan for d in range(367)])
ev_mask = np.array([np.nanmax(q[i:i + H]) >= Q90 for i in te])   # emisión cuyo horizonte contiene evento

models = []
for s in range(3):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT / f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)
X0 = T.seqs_for_enc(df, te, enc, H, (mu, sd))
xp0, qp0, xf0 = X0[0], X0[1].clone(), X0[2]

def predict(qp_tensor):
    ps = []
    for m in models:
        with torch.no_grad(): ps.append(m(xp0.to(DEV), qp_tensor.to(DEV), xf0.to(DEV)).cpu().numpy())
    return np.mean(ps, 0)

rows = []
for k in OUTAGES:
    qpk = qp0.clone()
    if k > 0:
        last = qpk[:, enc - k - 1:enc - k]          # último valor conocido antes del apagón
        qpk[:, enc - k:] = last                       # congela los últimos k días (carry-forward)
    PR = predict(qpk)
    for Lh in LEADS:
        o = np.array([obs[i + Lh - 1] for i in te]); p = PR[:, Lh - 1, 1]; qp = PR[:, Lh - 1, :]
        for reg, mk in [("global", np.ones(len(te), bool)), ("evento(≥Q90)", ev_mask)]:
            m = mk & np.isfinite(o) & np.isfinite(p)
            if m.sum() < 3: continue
            oa, pa = o[m] >= Q90, p[m] >= Q90; tp, fp, fn = int((oa & pa).sum()), int((~oa & pa).sum()), int((oa & ~pa).sum())
            rows.append(dict(apagon_dias=k, h=Lh, regimen=reg, N=int(m.sum()),
                             NSE=round(nse(o[m], p[m]), 3), CRPS=round(float(crps_c(o[m], qp[m]).mean()), 3),
                             POD=round(tp / (tp + fn), 3) if (tp + fn) else np.nan,
                             CSI=round(tp / (tp + fp + fn), 3) if (tp + fp + fn) else 0.0,
                             sesgo_pico=round(float(np.mean(p[m] - o[m])), 2)))
    log.info(f"apagón {k:2d}d: h1 NSE={[r for r in rows if r['apagon_dias']==k and r['h']==1 and r['regimen']=='global'][0]['NSE']} · "
             f"h14 NSE={[r for r in rows if r['apagon_dias']==k and r['h']==14 and r['regimen']=='global'][0]['NSE']} · "
             f"h1 NSE(evento)={[r for r in rows if r['apagon_dias']==k and r['h']==1 and r['regimen']=='evento(≥Q90)'][0]['NSE']}")

# modo degradado sin telemetría: persistencia y climatología
for Lh in LEADS:
    o = np.array([obs[i + Lh - 1] for i in te]); per = np.array([q[i - 1] for i in te]); cl = np.array([clim[dts[i + Lh - 1].dayofyear] for i in te])
    for nm, p in [("persistencia(fallback)", per), ("climatología(fallback)", cl)]:
        m = np.isfinite(o) & np.isfinite(p)
        rows.append(dict(apagon_dias=-1, h=Lh, regimen=f"fallback:{nm}", N=int(m.sum()), NSE=round(nse(o[m], p[m]), 3),
                         CRPS=round(float(np.mean(np.abs(o[m] - p[m]))), 3), POD=np.nan, CSI=np.nan, sesgo_pico=np.nan))
res = pd.DataFrame(rows); res.to_csv(OUT / "195_apagon_telemetria.csv", index=False)
log.info("\n=== NSE global vs días de apagón ===\n" +
         res[(res.regimen == "global")].pivot_table(index="apagon_dias", columns="h", values="NSE").to_string())

# figura: NSE vs apagón (global y evento) por horizonte
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
xs = [k for k in OUTAGES]
for reg, a, ttl in [("global", ax[0], "(a) NSE vs apagón — global"), ("evento(≥Q90)", ax[1], "(b) NSE vs apagón — EVENTO (≥Q90)")]:
    for Lh, c in zip(LEADS, ["#0e7d90", "#e08a2a", "#d24e39"]):
        ys = [res[(res.apagon_dias == k) & (res.h == Lh) & (res.regimen == reg)]["NSE"].values for k in OUTAGES]
        ys = [v[0] if len(v) else np.nan for v in ys]
        a.plot(xs, ys, "-o", color=c, lw=1.9, ms=5, label=f"h{Lh}")
    a.set_xlabel("días recientes de q_past congelados (apagón)"); a.set_ylabel("NSE"); a.set_title(ttl, fontsize=10.5, fontweight="bold")
    a.grid(lw=.3, alpha=.4); a.legend(fontsize=8); a.axhline(0, color="#999", lw=.7, ls=":")
fig.suptitle("Apagón de telemetría: degradación del ganador al congelar q_past (test 2024-25)", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "DI_apagon_telemetria.png", dpi=150)
log.info(f"figura: {FIG/'DI_apagon_telemetria.png'}"); log.info("APAGON_DONE")
