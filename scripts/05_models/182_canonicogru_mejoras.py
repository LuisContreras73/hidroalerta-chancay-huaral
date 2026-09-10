#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 182 — Buscar mejoras sobre el GANADOR (canónico+GRU, sin GFS): encoder + ensemble.

Dos levers para el régimen a horizonte largo (donde el seed-std es alto, 0.076 a h14):
  · ENCODER más largo (más contexto de estado): enc ∈ {90(base), 120, 150}.
  · ENSEMBLE mayor: 5 seeds (vs 3 del 179) → reduce varianza, sube NSE del ensemble.
Comparación JUSTA: mismo conjunto de train (valid filter i≥150 para todos los enc) y misma vara
(C.met sobre aforo, test 2024-25). FUT=[sin,cos] (calendario; NADA de GFS — ya descartado).
Referencia a batir: canónico+GRU 179 (h14=0.755, con enc=90/3 seeds).

TRAZABILIDAD: curvas de loss (182_loss_curves.csv), NSE + seed-std por enc (182_mejoras.csv), ckpts.
Run (.venv313, GPU): python scripts/05_models/182_canonicogru_mejoras.py  [--smoke]
"""
import importlib.util, logging, sys
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mej182"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOKE = "--smoke" in sys.argv


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")
R, DEV, C = M.R, M.DEV, M.C
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; PAST = list(R.PAST); NP = len(PAST); TR_END = pd.Timestamp("2022-12-31")
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values; arr = df[PAST].values
doy = dts.dayofyear.values
SIN = np.sin(2*np.pi*doy/365.25).astype(np.float32); COS = np.cos(2*np.pi*doy/365.25).astype(np.float32)
EM = 150   # valid filter común a todos los enc (comparación justa: mismas muestras)
qtr = q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr)+1e-6)
mu = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].mean(0)
sd = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].std(0)+1e-6
tr = [i for i in range(EM, len(df)-H) if dts[i] <= TR_END and np.isfinite(q[i-EM:i+H]).all()]
va = [i for i in range(EM, len(df)-H) if dts[i].year == 2023 and np.isfinite(q[i-EM:i+H]).all()]
te = [i for i in range(EM, len(df)-H) if dts[i] >= pd.Timestamp("2024-01-01")]


def seqs(idxs, enc):
    Xp, Qp, Xf, Y = [], [], [], []
    for i in idxs:
        Xp.append(np.nan_to_num((arr[i-enc:i]-mu)/sd)); Qp.append(np.nan_to_num(q[i-enc:i]))
        hh = np.arange(1, H+1); Xf.append(np.stack([SIN[i+hh-1], COS[i+hh-1]], axis=1)); Y.append(np.nan_to_num(q[i:i+H]))
    t = lambda a: torch.tensor(np.array(a, np.float32)); return t(Xp), t(Qp), t(Xf), t(Y)


ENCS = [90] if SMOKE else [90, 120, 150]; NSEED = 1 if SMOKE else 5; EP = 6 if SMOKE else 140; PAT = 3 if SMOKE else 25
loss_rows = []; log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · EM={EM} · encs={ENCS} · seeds={NSEED}")


def train_logged(model, X, Xv, enc, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    dl = DataLoader(TensorDataset(*X), batch_size=bp["bs"], shuffle=True); best = np.inf; bsd = None; bad = 0
    for e in range(EP):
        model.train(); tl = 0.0
        for xb, qb, fb, yb in dl:
            opt.zero_grad(); loss = R.pinball(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = R.pinball(model(Xv[0].to(DEV), Xv[1].to(DEV), Xv[2].to(DEV)), Xv[3].to(DEV)).item()
        loss_rows.append(dict(enc=enc, seed=seed, epoch=e, train_loss=tl/len(dl), val_loss=vl))
        if vl < best-1e-4:
            best, bsd, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT: break
    if bsd: model.load_state_dict(bsd)
    return model


rows = []
for enc in ENCS:
    Xtr = seqs(tr, enc); Xva = seqs(va, enc); Xte = seqs(te, enc); preds = []; per_seed = {L: [] for L in (1, 3, 7, 14)}
    for s in range(NSEED):
        torch.manual_seed(s); np.random.seed(s)
        mo = M.TFTCanonMatrix(NP, 2, hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
        mo = train_logged(mo, Xtr, Xva, enc, s); torch.save(mo.state_dict(), OUT/f"182_gru_enc{enc}_seed{s}.pt")
        with torch.no_grad():
            PR = mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy()
        preds.append(PR)
        for L in (1, 3, 7, 14):
            o = np.array([obs[i+L-1] for i in te]); p = PR[:, L-1, 1]; m = np.isfinite(o) & np.isfinite(p)
            per_seed[L].append(1-np.sum((o[m]-p[m])**2)/np.sum((o[m]-o[m].mean())**2))
    ens = np.mean(preds, 0)
    for L in (1, 3, 7, 14):
        j = [i+L-1 for i in te]; o = np.array([obs[k] for k in j])
        rows.append(dict(enc=enc, h=L, NSE=round(C.met(o, ens[:, L-1, :])["NSE"], 3), NSE_seed_std=round(float(np.std(per_seed[L])), 3), nseeds=NSEED))
    pd.DataFrame(rows).to_csv(OUT/"182_mejoras.csv", index=False)
    log.info(f"enc={enc} ✔ h14 NSE={[r for r in rows if r['enc']==enc and r['h']==14][0]['NSE']}")
pd.DataFrame(loss_rows).to_csv(OUT/"182_loss_curves.csv", index=False)
p = pd.DataFrame(rows).pivot_table(index="enc", columns="h", values="NSE")
log.info("\n=== NSE por encoder (ensemble de %d seeds) ===\n" % NSEED + p.to_string())
log.info("Referencia canónico+GRU 179 (enc=90, 3 seeds): h1=0.935 h3=0.876 h7=0.788 h14=0.755")
log.info("MEJORAS_DONE; 182_mejoras.csv + 182_loss_curves.csv")
