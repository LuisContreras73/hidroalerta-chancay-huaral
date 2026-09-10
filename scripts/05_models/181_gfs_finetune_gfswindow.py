#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 181 — Test JUSTO del GFS: pretrain(1981-2022) + fine-tune(ventana rica en GFS 2016-2022).

Motivación: en el esquema estándar (175/180) el GFS está diluido al ~15% del train (solo 2016+),
así que el modelo lo infra-usa. El test de techo cdrop=0 NO quitaba esa dilución. Aquí:
  · Fase 1 (pretrain): canónico+GRU+GFS sobre TODO 1981-2022 (GFS gated ~15%, channel-dropout 0.2).
  · Fase 2 (fine-tune): sobre SOLO 2016-2022 (GFS presente ~100%, cdrop=0, lr reducido) → la vía-GFS
    se refina donde el GFS existe, sin perder la base de 41 años.
Compara GFS-sano vs el modelo LIMPIO canónico+GRU (179, h14=0.755). Si supera 0.755, la dilución
era el problema; si no, el veredicto negativo del GFS-régimen es robusto.

Run (.venv313, GPU): python scripts/05_models/181_gfs_finetune_gfswindow.py  [--smoke]
"""
import importlib.util, logging, sys
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gfsft181"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOKE = "--smoke" in sys.argv


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")
R, DEV, Cmet = M.R, M.DEV, M.C.met
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; ENC = bp["enc"]; PAST = list(R.PAST); NP = len(PAST); TR_END = pd.Timestamp("2022-12-31")
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values; arr = df[PAST].values
doy = dts.dayofyear.values
SIN = np.sin(2*np.pi*doy/365.25).astype(np.float32); COS = np.cos(2*np.pi*doy/365.25).astype(np.float32)
date_pos = {d: k for k, d in enumerate(dts)}

# GFS lumped + QM (idéntico a 175/180)
g = pd.read_csv(ROOT/"data/bronze/B11_gfs_daily_leads.csv", parse_dates=["init_date"])
piv = g.pivot_table(index="init_date", columns="lead", values="pr_gfs_mm"); QLEV = np.linspace(0.01, 0.99, 99); qm = {}
for L in range(1, H+1):
    ser = piv[L].dropna(); tri = ser.index[ser.index <= TR_END]; gf = ser.loc[tri].values
    ob = np.array([df["pr"].values[date_pos[d+pd.Timedelta(days=L)]] if (d+pd.Timedelta(days=L)) in date_pos else np.nan for d in tri])
    mk = np.isfinite(gf) & np.isfinite(ob); qm[L] = (np.quantile(gf[mk], QLEV), np.quantile(ob[mk], QLEV)) if mk.sum() >= 100 else None
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
tr_ft = [i for i in tr if GAV[i].any()]                      # ventana rica en GFS (2016-2022)
va = [i for i in range(ENC, len(df)-H) if dts[i].year == 2023 and np.isfinite(q[i-ENC:i+H]).all()]
te = [i for i in range(ENC, len(df)-H) if dts[i] >= pd.Timestamp("2024-01-01")]
log.info(f"pretrain {len(tr)} (GFS {int(sum(GAV[i].any() for i in tr))}) · fine-tune(GFS-rico) {len(tr_ft)} · test {len(te)}")


def seqs(idxs, gfs_on=True):
    Xp, Qp, Xf, Y = [], [], [], []
    for i in idxs:
        Xp.append(np.nan_to_num((arr[i-ENC:i]-mu)/sd)); Qp.append(np.nan_to_num(q[i-ENC:i]))
        hh = np.arange(1, H+1); g_ = GFS[i] if gfs_on else np.zeros(H, np.float32); av = GAV[i] if gfs_on else np.zeros(H, np.float32)
        Xf.append(np.stack([g_, av, SIN[i+hh-1], COS[i+hh-1]], axis=1)); Y.append(np.nan_to_num(q[i:i+H]))
    t = lambda a: torch.tensor(np.array(a, np.float32)); return t(Xp), t(Qp), t(Xf), t(Y)


Xpre = seqs(tr); Xft = seqs(tr_ft); Xva = seqs(va); Xte_on = seqs(te, True); Xte_off = seqs(te, False)
NSEED = 1 if SMOKE else 3; EP1 = 6 if SMOKE else 120; EP2 = 4 if SMOKE else 60; PAT = 3 if SMOKE else 20; loss_rows = []


def train_phase(model, X, lr, cdrop, ep, tag, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=bp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    dl = DataLoader(TensorDataset(*X), batch_size=bp["bs"], shuffle=True); best = np.inf; bsd = None; bad = 0
    for e in range(ep):
        model.train(); tl = 0.0
        for xb, qb, fb, yb in dl:
            if cdrop > 0:
                msk = torch.rand(fb.shape[0]) < cdrop; fb = fb.clone(); fb[msk, :, 0:2] = 0.0
            opt.zero_grad(); loss = R.pinball(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = R.pinball(model(Xva[0].to(DEV), Xva[1].to(DEV), Xva[2].to(DEV)), Xva[3].to(DEV)).item()
        loss_rows.append(dict(seed=seed, fase=tag, epoch=e, train_loss=tl/len(dl), val_loss=vl))
        if vl < best-1e-4:
            best, bsd, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT: break
    if bsd: model.load_state_dict(bsd)
    return model


rows = []; seed_pr = {"GFS-sano": [], "GFS-caido": []}
for s in range(NSEED):
    torch.manual_seed(s); np.random.seed(s)
    mo = M.TFTCanonMatrix(NP, 4, hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    mo = train_phase(mo, Xpre, bp["lr"], 0.2, EP1, "pretrain", s)          # base full 1981-2022
    mo = train_phase(mo, Xft, bp["lr"]*0.3, 0.0, EP2, "finetune-GFS", s)   # foco GFS 2016-2022
    torch.save(mo.state_dict(), OUT/f"181_gfsft_seed{s}.pt")
    for tag, X in [("GFS-sano", Xte_on), ("GFS-caido", Xte_off)]:
        with torch.no_grad():
            seed_pr[tag].append(mo(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
    log.info(f"seed {s} OK")
for tag in ("GFS-caido", "GFS-sano"):
    ens = np.mean(seed_pr[tag], 0)
    for L in (1, 3, 7, 14):
        o = np.array([obs[te[k]+L-1] for k in range(len(te))]); rows.append(dict(modo=tag, h=L, **Cmet(o, ens[:, L-1, :])))
agg = pd.DataFrame(rows); agg.to_csv(OUT/"181_gfs_finetune.csv", index=False); pd.DataFrame(loss_rows).to_csv(OUT/"181_loss_curves.csv", index=False)
p = agg.pivot_table(index="modo", columns="h", values="NSE"); log.info("\n=== NSE (pretrain+finetune-GFS) ===\n"+p.to_string())
limpio = {1: 0.935, 3: 0.876, 7: 0.788, 14: 0.755}
log.info("VEREDICTO vs canónico+GRU LIMPIO (179): " + " ".join(
    f"h{L}: sano {p.loc['GFS-sano', L]:.3f} vs {limpio[L]:.3f} → {'GANA' if p.loc['GFS-sano', L] > limpio[L] else 'no'}" for L in (1, 3, 7, 14)))
log.info("GFSFT_DONE; 181_gfs_finetune.csv + 181_loss_curves.csv")
