#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 176 — GFS DISTRIBUIDO por sub-cuenca (routing espacial) vs lumped (175).

El lumped (175) promedia toda la cuenca, DILUYENDO la señal con la salida árida (sub_634 ~0.03
mm) mientras las cabeceras andinas pronostican ~2.5 mm. Aquí el decoder recibe la lluvia
pronosticada de las 9 sub-cuencas por separado (B12, script 55) → el VSN aprende a pesar las
cabeceras (donde nace la crecida). Sigue siendo forecast (honesto, operativo) y robusto
(channel-dropout + fallback calendario). Mismo split/harness que 175.

Hipótesis: distribuir mejora el multi-día MÁS que el lumped, y podría tocar el techo de alerta
(saber que lloverá fuerte en cabeceras = anticipar crecida).

Salida: outputs/ml_Q/176_gfs_distribuido.csv
Run (.venv313): python scripts/05_models/176_gfs_distribuido.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gfs176"); optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")
R, T, DEV, Cmet = M.R, M.T, M.DEV, M.C.met
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
H = 14; ENC = bp["enc"]; PAST = list(R.PAST); NP = len(PAST); Q90 = R.Q90; TR_END = pd.Timestamp("2022-12-31")
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values; arr = df[PAST].values
doy = dts.dayofyear.values; SIN = np.sin(2*np.pi*doy/365.25).astype(np.float32); COS = np.cos(2*np.pi*doy/365.25).astype(np.float32)
date_pos = {d: k for k, d in enumerate(dts)}

# ── B12: lluvia pronosticada por sub-cuenca ──
b = pd.read_csv(ROOT / "data/bronze/B12_gfs_daily_leads_subcuencas.csv", parse_dates=["init_date"])
SUBS = sorted(b.entity_id.unique()); NS = len(SUBS)   # 9
# GFS[k, sub_idx, lead-1] para cada fila-fecha k
GFS = np.zeros((len(df), NS, H), np.float32); GAV = np.zeros((len(df), H), np.float32)
for (init, ent), g in b.groupby(["init_date", "entity_id"]):
    k = date_pos.get(init)
    if k is None or ent not in SUBS:
        continue
    si = SUBS.index(ent)
    for _, row in g.iterrows():
        L = int(row["lead"]); GFS[k, si, L - 1] = row["pr_gfs_mm"]; GAV[k, L - 1] = 1.0
# z-norm por sub-cuenca con stats de train (inits <=2022)
trmask = np.array([dts[k] <= TR_END and GAV[k].any() for k in range(len(df))])
gmu = np.zeros(NS); gsd = np.ones(NS)
for si in range(NS):
    vals = GFS[trmask, si, :][GAV[trmask] > 0]
    if vals.size > 50:
        gmu[si] = vals.mean(); gsd[si] = vals.std() + 1e-6
GFSz = GFS.copy()
for si in range(NS):
    GFSz[:, si, :] = (GFS[:, si, :] - gmu[si]) / gsd[si]
GFSz = GFSz * GAV[:, None, :]   # 0 donde no hay dato

qmu = float(np.mean(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]))
qsd = float(np.std(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]) + 1e-6)
mu = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].mean(0)
sd = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].std(0) + 1e-6
tr = [i for i in range(ENC, len(df) - H) if dts[i] <= TR_END and np.isfinite(q[i - ENC:i + H]).all()]
va = [i for i in range(ENC, len(df) - H) if dts[i].year == 2023 and np.isfinite(q[i - ENC:i + H]).all()]
te = [i for i in range(ENC, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
NF = NS + 3   # 9 rains + avail + sin + cos


def seqs(idxs, gfs_on=True):
    Xp, Qp, Xf, Y = [], [], [], []
    for i in idxs:
        Xp.append(np.nan_to_num((arr[i - ENC:i] - mu) / sd)); Qp.append(np.nan_to_num(q[i - ENC:i]))
        hh = np.arange(1, H + 1)
        rains = GFSz[i].T if gfs_on else np.zeros((H, NS), np.float32)   # (H, NS)
        av = GAV[i] if gfs_on else np.zeros(H, np.float32)
        fut = np.concatenate([rains, av[:, None], SIN[i + hh - 1][:, None], COS[i + hh - 1][:, None]], axis=1)  # (H, NF)
        Xf.append(fut); Y.append(np.nan_to_num(q[i:i + H]))
    t = lambda a: torch.tensor(np.array(a, np.float32))
    return t(Xp), t(Qp), t(Xf), t(Y)


Xtr = seqs(tr); Xva = seqs(va); Xte_on = seqs(te, True); Xte_off = seqs(te, False)
log.info(f"subs={SUBS} NF={NF} · train {len(tr)} (GFS real {int(GAV[tr,0].sum())}) · test {len(te)}")


def train_cd(model, X, Xv, lr, wd, ep=140, pat=25, cdrop=0.2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180)
    dl = DataLoader(TensorDataset(*X), batch_size=bp["bs"], shuffle=True); best = np.inf; bsd = None; bad = 0
    for e in range(ep):
        model.train()
        for xb, qb, fb, yb in dl:
            if cdrop > 0:
                msk = torch.rand(fb.shape[0]) < cdrop; fb = fb.clone(); fb[msk, :, 0:NS + 1] = 0.0   # apaga las 9 lluvias + avail
            opt.zero_grad(); R.pinball(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = R.pinball(model(Xv[0].to(DEV), Xv[1].to(DEV), Xv[2].to(DEV)), Xv[3].to(DEV)).item()
        if vl < best - 1e-4:
            best, bsd, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= pat:
                break
    if bsd:
        model.load_state_dict(bsd)
    return model


rows = []
for s in range(3):
    torch.manual_seed(s); np.random.seed(s)
    mo = M.TFTCanonMatrix(NP, NF, hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd).to(DEV)
    mo = train_cd(mo, Xtr, Xva, bp["lr"], bp["wd"])
    torch.save(mo.state_dict(), OUT / f"176_gfsdist_seed{s}.pt")
    for tag, X in [("GFSdist-sano", Xte_on), ("GFSdist-caido", Xte_off)]:
        with torch.no_grad():
            PR = mo(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy()
        for h in (1, 3, 7, 14):
            o = np.array([obs[te[k] + h - 1] for k in range(len(te))])
            rows.append(dict(seed=s, modo=tag, h=h, **Cmet(o, PR[:, h - 1, :])))
    log.info(f"seed {s} OK")
res = pd.DataFrame(rows)
agg = res.groupby(["modo", "h"]).agg(NSE=("NSE", "mean"), CRPS=("CRPS", "mean"), CSI=("CSI", "mean"), POD=("POD", "mean")).round(3).reset_index()
agg.to_csv(OUT / "176_gfs_distribuido.csv", index=False)
log.info("\n" + agg.to_string(index=False)); log.info("Guardado: 176_gfs_distribuido.csv")
