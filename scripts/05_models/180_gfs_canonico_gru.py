#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 180 — GFS (lluvia pronosticada) sobre el modelo GANADOR: canónico+GRU.

El head-to-head (179) mostró que canónico+GRU es la mejor arquitectura a régimen largo
(h14 NSE 0.755 > canónico 0.706 > +RevIN 0.660). El GFS (175) se había probado solo sobre el
canónico-LSTM. Aquí montamos la palanca GFS-lumped (QM por lead + channel-dropout, honesta) sobre
canónico+GRU → probamos la mejor forzante sobre la mejor arquitectura. Mismo split/harness que 175.

TRAZABILIDAD: curvas de loss por época (180_loss_curves.csv) + métricas sano/caído con dispersión
de seeds (180_gfs_gru.csv) + checkpoints. Referencia sin-GFS: 179 canónico+GRU (h14=0.755).

Run (.venv313, GPU): python scripts/05_models/180_gfs_canonico_gru.py  [--smoke]
"""
import importlib.util, logging, sys
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gfsgru180"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOKE = "--smoke" in sys.argv


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")
R, T, DEV, Cmet = M.R, M.T, M.DEV, M.C.met
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; ENC = bp["enc"]; PAST = list(R.PAST); NP = len(PAST); Q90 = R.Q90; TR_END = pd.Timestamp("2022-12-31")
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values; arr = df[PAST].values
doy = dts.dayofyear.values
SIN = np.sin(2*np.pi*doy/365.25).astype(np.float32); COS = np.cos(2*np.pi*doy/365.25).astype(np.float32)
date_pos = {d: k for k, d in enumerate(dts)}

# ── GFS lumped + QM por lead (idéntico a 175) ──
g = pd.read_csv(ROOT/"data/bronze/B11_gfs_daily_leads.csv", parse_dates=["init_date"])
piv = g.pivot_table(index="init_date", columns="lead", values="pr_gfs_mm"); QLEV = np.linspace(0.01, 0.99, 99); qm = {}
for L in range(1, H+1):
    ser = piv[L].dropna(); tri = ser.index[ser.index <= TR_END]; gf = ser.loc[tri].values
    ob = np.array([df["pr"].values[date_pos[d+pd.Timedelta(days=L)]] if (d+pd.Timedelta(days=L)) in date_pos else np.nan for d in tri])
    m = np.isfinite(gf) & np.isfinite(ob); qm[L] = (np.quantile(gf[m], QLEV), np.quantile(ob[m], QLEV)) if m.sum() >= 100 else None
def gq(v, L):
    mp = qm.get(L); return float(np.interp(v, mp[0], mp[1])) if (mp and np.isfinite(v)) else np.nan
GFS = np.zeros((len(df), H), np.float32); GAV = np.zeros((len(df), H), np.float32)
allqm = [gq(piv.at[d, L], L) for d in piv.index if d <= TR_END for L in range(1, H+1) if L in piv.columns and np.isfinite(piv.at[d, L])]
gmu = float(np.nanmean(allqm)); gsd = float(np.nanstd(allqm)+1e-6)
for d in piv.index:
    k = date_pos.get(d)
    if k is None: continue
    for L in range(1, H+1):
        v = piv.at[d, L] if L in piv.columns else np.nan
        if np.isfinite(v): GFS[k, L-1] = (gq(v, L)-gmu)/gsd; GAV[k, L-1] = 1.0
qmu = float(np.mean(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]))
qsd = float(np.std(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]])+1e-6)
mu = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].mean(0)
sd = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].std(0)+1e-6
tr = [i for i in range(ENC, len(df)-H) if dts[i] <= TR_END and np.isfinite(q[i-ENC:i+H]).all()]
va = [i for i in range(ENC, len(df)-H) if dts[i].year == 2023 and np.isfinite(q[i-ENC:i+H]).all()]
te = [i for i in range(ENC, len(df)-H) if dts[i] >= pd.Timestamp("2024-01-01")]


def seqs(idxs, gfs_on=True):
    Xp, Qp, Xf, Y = [], [], [], []
    for i in idxs:
        Xp.append(np.nan_to_num((arr[i-ENC:i]-mu)/sd)); Qp.append(np.nan_to_num(q[i-ENC:i]))
        hh = np.arange(1, H+1); g_ = GFS[i] if gfs_on else np.zeros(H, np.float32); av = GAV[i] if gfs_on else np.zeros(H, np.float32)
        Xf.append(np.stack([g_, av, SIN[i+hh-1], COS[i+hh-1]], axis=1)); Y.append(np.nan_to_num(q[i:i+H]))
    t = lambda a: torch.tensor(np.array(a, np.float32)); return t(Xp), t(Qp), t(Xf), t(Y)


Xtr = seqs(tr); Xva = seqs(va); Xte_on = seqs(te, True); Xte_off = seqs(te, False)
# --cdrop=X : tasa de channel-dropout del GFS (0.2 robusto por defecto; 0.0 = compromiso total/techo)
CD = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--cdrop=")), 0.2))
SUF = "" if abs(CD - 0.2) < 1e-9 else f"_cd{CD:g}"
log.info(f"canónico+GRU+GFS · channel-dropout={CD} · train {len(tr)} (GFS real {int((GAV[tr,0]>0).sum())}) · test {len(te)}")
loss_rows = []; NSEED = 1 if SMOKE else 3; EP = 6 if SMOKE else 140; PAT = 3 if SMOKE else 25


def train_cd_logged(model, seed, cdrop=0.2):
    opt = torch.optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True); best = np.inf; bsd = None; bad = 0
    for e in range(EP):
        model.train(); tl = 0.0
        for xb, qb, fb, yb in dl:
            if cdrop > 0:
                msk = torch.rand(fb.shape[0]) < cdrop; fb = fb.clone(); fb[msk, :, 0:2] = 0.0
            opt.zero_grad(); loss = R.pinball(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = R.pinball(model(Xva[0].to(DEV), Xva[1].to(DEV), Xva[2].to(DEV)), Xva[3].to(DEV)).item()
        loss_rows.append(dict(seed=seed, epoch=e, train_loss=tl/len(dl), val_loss=vl))
        if vl < best-1e-4:
            best, bsd, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT:
                break
    if bsd:
        model.load_state_dict(bsd)
    return model


rows = []; seed_pr = {"GFS-sano": [], "GFS-caido": []}; per_seed = {("GFS-sano", L): [] for L in (1, 3, 7, 14)}
per_seed.update({("GFS-caido", L): [] for L in (1, 3, 7, 14)})
for s in range(NSEED):
    torch.manual_seed(s); np.random.seed(s)
    mo = M.TFTCanonMatrix(NP, 4, hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"],
                          q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)   # ← GANADOR: GRU
    mo = train_cd_logged(mo, s, cdrop=CD)
    torch.save(mo.state_dict(), OUT/f"180_gfsgru{SUF}_seed{s}.pt")
    for tag, X in [("GFS-sano", Xte_on), ("GFS-caido", Xte_off)]:
        with torch.no_grad():
            PR = mo(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy()
        seed_pr[tag].append(PR)
        for L in (1, 3, 7, 14):
            o = np.array([obs[te[k]+L-1] for k in range(len(te))]); p = PR[:, L-1, 1]; mm = np.isfinite(o) & np.isfinite(p)
            per_seed[(tag, L)].append(1-np.sum((o[mm]-p[mm])**2)/np.sum((o[mm]-o[mm].mean())**2))
    log.info(f"seed {s} OK")
for tag in ("GFS-caido", "GFS-sano"):
    ens = np.mean(seed_pr[tag], 0)
    for L in (1, 3, 7, 14):
        o = np.array([obs[te[k]+L-1] for k in range(len(te))])
        rows.append(dict(modo=tag, h=L, **Cmet(o, ens[:, L-1, :]), NSE_seed_std=round(float(np.std(per_seed[(tag, L)])), 3)))
agg = pd.DataFrame(rows); agg.to_csv(OUT/f"180_gfs_gru{SUF}.csv", index=False)
pd.DataFrame(loss_rows).to_csv(OUT/f"180_loss_curves{SUF}.csv", index=False)
piv2 = agg.pivot_table(index="modo", columns="h", values="NSE")
log.info("\n=== NSE canónico+GRU+GFS (sano vs caído) ===\n"+piv2.to_string())
d = piv2.loc["GFS-sano"]-piv2.loc["GFS-caido"]
log.info("ΔNSE GFS (sano-caído): "+" ".join(f"h{L}={d[L]:+.3f}" for L in (1, 3, 7, 14)))
log.info("Ref sin-GFS (179 canónico+GRU): h1=0.935 h3=0.876 h7=0.788 h14=0.755")
log.info("GFSGRU_DONE; 180_gfs_gru.csv + 180_loss_curves.csv")
