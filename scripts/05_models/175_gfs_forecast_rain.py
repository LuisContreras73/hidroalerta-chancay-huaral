#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 175 — GFS como forzante FUTURA del decoder (lluvia pronosticada, sin leakage).

Reactiva la señal anticipatoria que el gen4 honesto desactivó (FUT=solo calendario), pero
de forma LEGÍTIMA: usa el pronóstico de lluvia GFS archivado (B11, script 54) = lo que habría
estado disponible operacionalmente cada día. NO es observación futura (eso sería fuga).

Pipeline:
  1. QM (quantile mapping) por lead: corrige el sesgo/distribución de GFS vs PISCO (ajuste
     SOLO en train 2016-2022, aplicado a todo — sin leakage).
  2. FUT = [pr_gfs_QM, gfs_avail, sin, cos]; el decoder ingiere la lluvia pronosticada 1..H.
  3. Diseño ROBUSTO (validado en la investigación upstream): channel-dropout sobre el canal
     GFS + indicador de disponibilidad. GFS enmascarado pre-2016 (no existe) y en fallos.
     -> una red, degrada con gracia si GFS falta ('GFS caído' ≈ base calendario).
  4. Split gen4: train ≤2022 (GR4J, GFS real 2016+), val 2023, test 2024-25 (aforo real).

Eval: 'GFS sano' vs 'GFS caído' vs baseline calendario, por lead h=1..14, en NSE/CRPS y
ALERTA (CSI/POD sobre Q90=40.89) — la hipótesis es que la lluvia pronosticada rompe el techo
de POD=0 a h≥3.

Salida: outputs/ml_Q/175_gfs_forecast.csv (+ 175_gfs_qm_skill.csv)
Run (.venv313, GPU libre): python scripts/05_models/175_gfs_forecast_rain.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gfs175")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")   # TFTCanonMatrix + R + T + honesto
R, T, DEV, Cmet = M.R, M.T, M.DEV, M.C.met
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
H = 14; ENC = bp["enc"]; PAST = list(R.PAST); NP = len(PAST); Q90 = R.Q90
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values; arr = df[PAST].values
doy = dts.dayofyear.values
SIN = np.sin(2 * np.pi * doy / 365.25).astype(np.float32); COS = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
date_pos = {d: k for k, d in enumerate(dts)}   # fecha -> índice fila

# ── GFS: pivot init_date × lead ──
g = pd.read_csv(ROOT / "data/bronze/B11_gfs_daily_leads.csv", parse_dates=["init_date"])
piv = g.pivot_table(index="init_date", columns="lead", values="pr_gfs_mm")   # filas=init, cols=lead 1..14
gfs_dates = set(piv.index)

# ── QM por lead: ajuste en train (init 2016-2022), aplica a todo ──
TR_END = pd.Timestamp("2022-12-31")
QLEV = np.linspace(0.01, 0.99, 99)
qm_maps = {}
for L in range(1, H + 1):
    ser = piv[L].dropna()
    tr_init = ser.index[ser.index <= TR_END]
    gf = ser.loc[tr_init].values
    ob = np.array([df["pr"].values[date_pos[d + pd.Timedelta(days=L)]]
                   if (d + pd.Timedelta(days=L)) in date_pos else np.nan for d in tr_init])
    m = np.isfinite(gf) & np.isfinite(ob)
    if m.sum() < 100:
        qm_maps[L] = None; continue
    gq = np.quantile(gf[m], QLEV); oq = np.quantile(ob[m], QLEV)
    qm_maps[L] = (gq, oq)


def gfs_qm(val, L):
    mp = qm_maps.get(L)
    if mp is None or not np.isfinite(val):
        return np.nan
    gq, oq = mp
    return float(np.interp(val, gq, oq))


# skill QM (antes/después) para reporte
skill_rows = []
for L in (1, 3, 7, 14):
    ser = piv[L].dropna()
    d0 = [d for d in ser.index if (d + pd.Timedelta(days=L)) in date_pos]
    raw = ser.loc[d0].values
    cor = np.array([gfs_qm(v, L) for v in raw])
    ob = np.array([df["pr"].values[date_pos[d + pd.Timedelta(days=L)]] for d in d0])
    mm = np.isfinite(raw) & np.isfinite(ob) & np.isfinite(cor)
    skill_rows.append(dict(lead=L, corr=round(float(np.corrcoef(raw[mm], ob[mm])[0, 1]), 3),
                           bias_raw=round(float(raw[mm].mean() - ob[mm].mean()), 2),
                           bias_qm=round(float(cor[mm].mean() - ob[mm].mean()), 2)))
pd.DataFrame(skill_rows).to_csv(OUT / "175_gfs_qm_skill.csv", index=False)
log.info("QM skill:\n" + pd.DataFrame(skill_rows).to_string(index=False))

# ── construir FUT con GFS: para emisión i (fecha d), horizonte h -> gfs_qm(piv[d][h]) ──
GFS = np.zeros((len(df), H), np.float32); GAV = np.zeros((len(df), H), np.float32)
qmu = float(np.mean(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]))
qsd = float(np.std(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]) + 1e-6)
# normalizador de la lluvia QM (mm) con stats de train
allqm = []
for d in piv.index:
    if d <= TR_END:
        for L in range(1, H + 1):
            v = piv.at[d, L] if L in piv.columns else np.nan
            if np.isfinite(v): allqm.append(gfs_qm(v, L))
gmu = float(np.nanmean(allqm)); gsd = float(np.nanstd(allqm) + 1e-6)
for d in piv.index:
    k = date_pos.get(d)
    if k is None:
        continue
    for L in range(1, H + 1):
        v = piv.at[d, L] if L in piv.columns else np.nan
        if np.isfinite(v):
            GFS[k, L - 1] = (gfs_qm(v, L) - gmu) / gsd; GAV[k, L - 1] = 1.0

mu = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].mean(0)
sd = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].std(0) + 1e-6
tr = [i for i in range(ENC, len(df) - H) if dts[i] <= TR_END and np.isfinite(q[i - ENC:i + H]).all()]
va = [i for i in range(ENC, len(df) - H) if dts[i].year == 2023 and np.isfinite(q[i - ENC:i + H]).all()]
te = [i for i in range(ENC, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]


def seqs(idxs, gfs_on=True):
    Xp, Qp, Xf, Y = [], [], [], []
    for i in idxs:
        Xp.append(np.nan_to_num((arr[i - ENC:i] - mu) / sd)); Qp.append(np.nan_to_num(q[i - ENC:i]))
        hh = np.arange(1, H + 1)
        g_ = GFS[i, :] if gfs_on else np.zeros(H, np.float32)
        av = GAV[i, :] if gfs_on else np.zeros(H, np.float32)
        fut = np.stack([g_, av, SIN[i + hh - 1], COS[i + hh - 1]], axis=1)  # (H,4)
        Xf.append(fut); Y.append(np.nan_to_num(q[i:i + H]))
    t = lambda a: torch.tensor(np.array(a, np.float32))
    return t(Xp), t(Qp), t(Xf), t(Y)


Xtr = seqs(tr); Xva = seqs(va); Xte_on = seqs(te, True); Xte_off = seqs(te, False)
log.info(f"train {len(tr)} (GFS real {int((GAV[tr,0]>0).sum())}) · val {len(va)} · test {len(te)}")


def train_cd(model, X, Xv, lr, wd, ep=140, pat=25, cdrop=0.3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180)
    dl = DataLoader(TensorDataset(*X), batch_size=bp["bs"], shuffle=True)
    best = np.inf; bs = None; bad = 0
    for e in range(ep):
        model.train()
        for xb, qb, fb, yb in dl:
            if cdrop > 0:                        # channel-dropout del canal GFS (0) + avail (1) del FUT
                msk = torch.rand(fb.shape[0]) < cdrop; fb = fb.clone(); fb[msk, :, 0:2] = 0.0
            opt.zero_grad(); R.pinball(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = R.pinball(model(Xv[0].to(DEV), Xv[1].to(DEV), Xv[2].to(DEV)), Xv[3].to(DEV)).item()
        if vl < best - 1e-4:
            best, bs, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= pat:
                break
    if bs:
        model.load_state_dict(bs)
    return model


def evalm(model, X):
    model.eval()
    with torch.no_grad():
        PR = model(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy()
    return PR


rows = []
for s in range(3):
    torch.manual_seed(s); np.random.seed(s)
    mo = M.TFTCanonMatrix(NP, 4, hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd).to(DEV)
    mo = train_cd(mo, Xtr, Xva, bp["lr"], bp["wd"], cdrop=0.2)
    torch.save(mo.state_dict(), OUT / f"175_gfs_seed{s}.pt")
    for tag, X in [("GFS-sano", Xte_on), ("GFS-caido", Xte_off)]:
        PR = evalm(mo, X)
        for h in (1, 3, 7, 14):
            o = np.array([obs[te[k] + h - 1] for k in range(len(te))])
            rows.append(dict(seed=s, modo=tag, h=h, **Cmet(o, PR[:, h - 1, :])))
    log.info(f"seed {s} OK")

res = pd.DataFrame(rows)
agg = res.groupby(["modo", "h"]).agg(NSE=("NSE", "mean"), CRPS=("CRPS", "mean"),
                                     CSI=("CSI", "mean"), POD=("POD", "mean")).round(3).reset_index()
agg.to_csv(OUT / "175_gfs_forecast.csv", index=False)
log.info("\n" + agg.to_string(index=False))
log.info("Guardado: 175_gfs_forecast.csv + 175_gfs_qm_skill.csv")
