#!/usr/bin/env python3
"""
Script 100 — Multi-horizon skill vs lead time (how we sustain "we are better").

Persistence wins at 1 day but decays fast; a forcing-driven ML model should hold
skill longer. This produces the key defensible figure: NSE and J_alert vs forecast
horizon (h = 1,2,3,5,7,10,14 days) for ML (LightGBM) vs persistence vs climatology,
evaluated on the honest real-observation test.

Run in .venv:
    python scripts/06_eval/100_multi_horizon_skill.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("multih")
Q_CONV = 86.4/3062.62; Q90 = 40.89
HORIZONS = [1, 2, 3, 5, 7, 10, 14]
TRAIN_END = "2023-12-31"   # test = 2024+ (real obs)


def build_daily():
    d7 = pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv", parse_dates=["date"])
    o = d7[d7.entity_id=="sub_634"].set_index("date").sort_index()
    pr = d7.groupby("date")["pr_mm"].mean()
    swvl = pd.read_parquet(OUTML/"era5_swvl_per_entity.parquet")
    if "date" not in swvl.columns: swvl = swvl.reset_index()
    sw = swvl.groupby("date")["swvl4"].mean()
    enso = pd.read_csv(ROOT/"data/silver/enso/S3b_enso_coastal_global.csv", parse_dates=["date"]).set_index("date")

    df = pd.DataFrame(index=o.index)
    df["q"]   = o["q_mm"]/Q_CONV                 # D7 continuous (obs+GR4J) — features/train target
    df["pr"]  = pr.reindex(o.index)
    df["api"] = o["api"]; df["spi30"]=o["spi_30d"]; df["spi90"]=o["spi_90d"]
    df["wdef"]= o["water_deficit_30d"]; df["oni"]=o["oni_index"]
    df["swvl4"]= sw.reindex(o.index)
    df["coastal"]= enso["coastal"].reindex(o.index).ffill()
    # lags / rolls (known at t)
    df["q_l1"]=df["q"].shift(1); df["q_l7"]=df["q"].shift(7)
    df["q_r7"]=df["q"].rolling(7,min_periods=3).mean(); df["q_r30"]=df["q"].rolling(30,min_periods=10).mean()
    df["pr_s7"]=df["pr"].rolling(7,min_periods=3).sum()
    doy=df.index.dayofyear
    df["sin"]=np.sin(2*np.pi*doy/365.25); df["cos"]=np.cos(2*np.pi*doy/365.25)

    obs = pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv", parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    df["obs"] = obs.reindex(o.index)
    return df

FEATS = ["q","q_l1","q_l7","q_r7","q_r30","pr","pr_s7","api","spi30","spi90","wdef","swvl4","oni","coastal","sin","cos"]


def metrics(o, p):
    o,p=np.asarray(o,float),np.asarray(p,float)
    nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None))
    nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
    oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if (tp+fp+fn) else 0; pod=tp/(tp+fn) if (tp+fn) else 0; far=fp/(tp+fp) if (tp+fp) else 0
    return dict(NSE=nse, J=0.25*nsq+0.25*nse+0.30*csi+0.10*pod-0.10*far, CSI=csi, POD=pod, FAR=far)


def main():
    df = build_daily()
    clim = df[df.index<=TRAIN_END].groupby(df[df.index<=TRAIN_END].index.dayofyear)["q"].mean()
    rows=[]
    for h in HORIZONS:
        d = df.copy()
        d["y_train"] = d["q"].shift(-h)                 # continuous target (train)
        d["y_obs"]   = d["obs"].shift(-h)               # REAL obs at t+h (eval)
        tr = d[(d.index<=TRAIN_END)].dropna(subset=FEATS+["y_train"])
        te = d[(d.index>TRAIN_END)].dropna(subset=FEATS+["y_obs"])
        mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03,
                                num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                reg_lambda=1.0, random_state=0, verbose=-1)
        mdl.fit(tr[FEATS], np.log1p(tr["y_train"].clip(lower=0)))
        pred_ml = np.expm1(mdl.predict(te[FEATS]))
        obs = te["y_obs"].values
        pred_pers = te["q"].values                       # persistence = today's flow
        pred_clim = te.index.dayofyear.map(clim).values
        for name,p in [("ML (LightGBM)",pred_ml),("Persistence",pred_pers),("Climatology",pred_clim)]:
            m=metrics(obs,p); rows.append(dict(h=h, model=name, n=len(obs), **m))
        mlm=metrics(obs,pred_ml); pm=metrics(obs,pred_pers)
        log.info(f"h={h:2d}d (n={len(obs)}): ML NSE={mlm['NSE']:.3f} J={mlm['J']:.3f} | "
                 f"Persist NSE={pm['NSE']:.3f} J={pm['J']:.3f}")
    res=pd.DataFrame(rows); res.to_csv(OUTML/"100_multi_horizon.csv", index=False)

    # ── Figure: NSE and J_alert vs horizon ────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    col={"ML (LightGBM)":"#2ca02c","Persistence":"#d62728","Climatology":"#7f7f7f"}
    for key,axi,lab in [("NSE",ax[0],"Nash-Sutcliffe Efficiency (NSE)"),("J",ax[1],"Composite alert score J_alert")]:
        for mdl in ["ML (LightGBM)","Persistence","Climatology"]:
            s=res[res.model==mdl]
            axi.plot(s["h"], s[key], "-o", color=col[mdl], lw=2, ms=5, label=mdl)
        axi.set_xlabel("Forecast horizon (days)"); axi.set_ylabel(lab)
        axi.grid(alpha=0.3); axi.set_xticks(HORIZONS)
    ax[0].legend(fontsize=9); ax[0].set_title("(a) Continuous skill vs lead time")
    ax[1].set_title("(b) Alert skill vs lead time")
    # mark crossover where ML overtakes persistence in NSE
    mln=res[res.model=="ML (LightGBM)"].set_index("h")["NSE"]; pn=res[res.model=="Persistence"].set_index("h")["NSE"]
    cross=[h for h in HORIZONS if mln[h]>pn[h]]
    if cross: ax[0].axvline(min(cross), color="black", ls=":", lw=1); ax[0].annotate(
        f"ML > persistence from ~{min(cross)} d", (min(cross), ax[0].get_ylim()[0]+0.05), fontsize=8)
    fig.suptitle("Forecast skill vs lead time: persistence dominates at 1 day but decays; "
                 "the ML model sustains skill at longer horizons", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGDIR/"fig20_multi_horizon.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    log.info(f"ML>persistence (NSE) from horizon: {min(cross) if cross else 'never'} d")
    log.info(f"Saved: {FIGDIR/'fig20_multi_horizon.png'} and 100_multi_horizon.csv")


if __name__ == "__main__":
    main()
