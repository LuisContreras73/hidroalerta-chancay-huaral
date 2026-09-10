#!/usr/bin/env python3
"""
Script 90 — Why did the global ONI help the monthly model more than the coastal
index? Statistical diagnosis + clean feature ablation.

Three questions:
  Q1. How related are the two indices?  -> r and R^2 between ONI (Nino 3.4) and
      coastal (Nino 1+2), overall and by season. High R^2 => redundancy.
  Q2. How does each relate to the TARGET (monthly Q anomaly)?  -> Pearson/Spearman
      and partial correlation (each controlling for the other).
  Q3. Which actually helps the model?  -> clean ablation of the monthly anomaly
      model with feature sets {no ENSO, +ONI, +coastal, +both}, multi-seed, on the
      honest real-obs test. Plus permutation importance of ONI vs coastal.

Run in .venv:
    python scripts/06_eval/90_enso_index_diagnosis.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT   = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "generacion_paper/figures"
OUTML  = ROOT / "outputs/ml_Q"
FIGDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("enso_diag")

# Reuse the monthly frame builder from Script 86
spec = importlib.util.spec_from_file_location("m86", ROOT/"scripts/05_models/86_monthly_water_availability.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

TRAIN_END = "2014-12-31"; VAL_END = "2020-12-31"; WET = [12,1,2,3,4]


def nse(o,p):
    o,p=np.asarray(o,float),np.asarray(p,float); return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
def kge(o,p):
    o,p=np.asarray(o,float),np.asarray(p,float); r=np.corrcoef(o,p)[0,1]
    return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
def partial_corr(x, y, z):
    """corr(x,y | z): correlation of residuals after regressing out z."""
    X=np.c_[np.ones_like(z), z]
    rx = x - X@np.linalg.lstsq(X,x,rcond=None)[0]
    ry = y - X@np.linalg.lstsq(X,y,rcond=None)[0]
    return np.corrcoef(rx,ry)[0,1]


def main():
    m = M.build_monthly()
    m = m.dropna(subset=["oni_l1","coastal_l1","y"])

    # ── Q1. Relationship between the two indices ──────────────────────────────
    oni, coa = m["oni_l1"].values, m["coastal_l1"].values
    r_idx = np.corrcoef(oni, coa)[0,1]
    log.info(f"Q1. ONI vs coastal: r={r_idx:.3f}  R2={r_idx**2:.3f}  "
             f"(std ONI={oni.std():.2f}, coastal={coa.std():.2f})")

    # ── Q2. Relationship of each index with the TARGET anomaly ────────────────
    tr = m[m.index <= TRAIN_END]
    clim = tr.groupby("y_month")["y"].mean()
    anom = (m["y"] - m["y_month"].map(clim)).values
    rr = {}
    rr["ONI"]     = np.corrcoef(oni, anom)[0,1]
    rr["coastal"] = np.corrcoef(coa, anom)[0,1]
    pc_oni = partial_corr(oni, anom, coa)   # ONI controlling for coastal
    pc_coa = partial_corr(coa, anom, oni)   # coastal controlling for ONI
    log.info(f"Q2. corr with Q-anomaly: ONI={rr['ONI']:+.3f}  coastal={rr['coastal']:+.3f}")
    log.info(f"    partial corr (controlling the other): ONI|coa={pc_oni:+.3f}  coa|ONI={pc_coa:+.3f}")

    # ── Q3. Clean feature ablation (anomaly model) ────────────────────────────
    te = m[m.index > VAL_END]
    clim_issue = tr.groupby(tr.index.month)["q"].mean()
    def anomalize(fr):
        a=fr.copy()
        a["q_l1"]=a["q_l1"]-a.index.month.map(clim_issue)
        a["q_l2"]=a["q_l2"]-((a.index.month-2)%12+1).map(clim_issue)
        a["q_l3"]=a["q_l3"]-((a.index.month-3)%12+1).map(clim_issue)
        return a
    tr_a, te_a = anomalize(tr), anomalize(te)
    y_anom = (tr["y"]-tr["y_month"].map(clim)).values
    te_clim = te["y_month"].map(clim).values
    mask = te["y_obs"].notna().values
    obs = te["y_obs"].values[mask]

    BASE = ["q_l1","q_l2","q_l3","pr_l1","pr_2m","swvl4_l1","swvl1_l1","spi90_l1"]
    SETS = {"No ENSO": BASE, "+ ONI (global)": BASE+["oni_l1"],
            "+ Coastal (Nino 1+2)": BASE+["coastal_l1"], "+ Both": BASE+["oni_l1","coastal_l1"]}

    def fit_p50(feats, seed):
        mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=300,
                                learning_rate=0.03, num_leaves=12, min_child_samples=20,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                random_state=seed, verbose=-1)
        mdl.fit(tr_a[feats], y_anom)
        return te_clim + mdl.predict(te_a[feats]), mdl

    abl = {}
    for name, feats in SETS.items():
        nses, kges = [], []
        for s in range(10):
            p,_ = fit_p50(feats, s); p=p[mask]
            nses.append(nse(obs,p)); kges.append(kge(obs,p))
        abl[name] = (np.mean(nses), np.std(nses), np.mean(kges), np.std(kges))
        log.info(f"Q3. {name:22s} NSE={np.mean(nses):.3f}±{np.std(nses):.3f}  "
                 f"KGE={np.mean(kges):.3f}±{np.std(kges):.3f}")

    # Permutation importance of ONI vs coastal in the +Both model
    _, mdl = fit_p50(SETS["+ Both"], 0)
    base_pred = te_clim + mdl.predict(te_a[SETS["+ Both"]])
    base_err = np.mean((obs - base_pred[mask])**2)
    perm = {}
    rng = np.random.default_rng(0)
    for f in ["oni_l1","coastal_l1"]:
        errs=[]
        for _ in range(20):
            Xp = te_a[SETS["+ Both"]].copy()
            Xp[f] = rng.permutation(Xp[f].values)
            pp = te_clim + mdl.predict(Xp)
            errs.append(np.mean((obs-pp[mask])**2))
        perm[f] = np.mean(errs)/base_err - 1   # relative MSE increase
    log.info(f"Permutation importance (rel. MSE increase): ONI={perm['oni_l1']:+.3f}  "
             f"coastal={perm['coastal_l1']:+.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) scatter ONI vs coastal with R2
    ax=axes[0]
    sc=ax.scatter(oni, coa, c=m.index.month.isin(WET), cmap="coolwarm", s=18, alpha=0.6)
    lo,hi=min(oni.min(),coa.min()),max(oni.max(),coa.max())
    ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
    ax.set_xlabel("ONI — global (Nino 3.4) [degC]"); ax.set_ylabel("Coastal (Nino 1+2) [degC]")
    ax.set_title(f"(a) The two indices share variance\nr={r_idx:.2f}, R²={r_idx**2:.2f} "
                 f"(coastal = global + episodic noise)", fontsize=10)
    ax.grid(alpha=0.3)

    # (b) correlation with target anomaly (simple + partial)
    ax=axes[1]
    labels=["ONI\n(simple)","Coastal\n(simple)","ONI\n(partial)","Coastal\n(partial)"]
    vals=[rr["ONI"], rr["coastal"], pc_oni, pc_coa]
    cols=["#1f77b4","#cc2222","#1f77b4","#cc2222"]
    ax.bar(range(4), vals, color=cols, alpha=0.85)
    ax.axhline(0,color="black",lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=8)
    ax.set_ylabel("Correlation with monthly Q anomaly")
    ax.set_title("(b) Each index vs the target anomaly\n(partial = controlling for the other)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # (c) ablation skill
    ax=axes[2]
    names=list(abl.keys()); nse_m=[abl[n][0] for n in names]; nse_s=[abl[n][1] for n in names]
    cols2=["#999","#1f77b4","#cc2222","#7e57c2"]
    ax.bar(range(len(names)), nse_m, yerr=nse_s, capsize=4, color=cols2, alpha=0.85)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("NSE on honest test (10 seeds)")
    ax.set_title("(c) Does each index help the model?\nclean ablation (mean ± s.d.)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(min(nse_m)-0.05, max(nse_m)+0.05)

    fig.suptitle("Why global ONI helped more than the coastal index in the monthly model",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig13_enso_diagnosis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {FIGDIR/'fig13_enso_diagnosis.png'}")


if __name__ == "__main__":
    main()
