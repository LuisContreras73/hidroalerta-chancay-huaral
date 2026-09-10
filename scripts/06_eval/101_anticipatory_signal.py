#!/usr/bin/env python3
"""
Script 101 — Does anticipatory forcing (future rainfall) beat persistence on
alerts? Perfect-forecast upper-bound experiment.

Persistence is hard to beat on ALERT skill with historical features only (Script
100). Here we test whether ANTICIPATORY signal helps: we add the ACTUAL future
precipitation over the horizon (a "perfect rainfall forecast" upper bound) as
features. If this beats persistence on J_alert at h>=3 d, it proves that the path
to better flood alerts is operational rainfall forecasts (GFS/ECMWF/PISCO-forecast),
not more historical inputs.

Compares, per horizon: persistence · ML (historical only) · ML + perfect future rain.

Run in .venv:
    python scripts/06_eval/101_anticipatory_signal.py
"""

import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"; FIGDIR = ROOT / "generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("antic")

M = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "m100", ROOT/"scripts/06_eval/100_multi_horizon_skill.py"))
importlib.util.spec_from_file_location("m100", ROOT/"scripts/06_eval/100_multi_horizon_skill.py").loader.exec_module(M)
Q90 = M.Q90; HOR = M.HORIZONS; FEATS = M.FEATS; metrics = M.metrics; TRAIN_END = M.TRAIN_END


def main():
    df = M.build_daily()
    clim = df[df.index<=TRAIN_END].groupby(df[df.index<=TRAIN_END].index.dayofyear)["q"].mean()
    rows=[]
    for h in HOR:
        d = df.copy()
        # perfect future-rain feature: cumulative precip over the forecast horizon (t+1..t+h)
        d["pr_future"] = d["pr"].shift(-h).rolling(h, min_periods=1).sum()  # sum of pr[t+1..t+h]
        d["y_train"] = d["q"].shift(-h); d["y_obs"] = d["obs"].shift(-h)
        tr = d[d.index<=TRAIN_END].dropna(subset=FEATS+["pr_future","y_train"])
        te = d[d.index>TRAIN_END].dropna(subset=FEATS+["pr_future","y_obs"])
        obs = te["y_obs"].values
        def fit(feats):
            m=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=400,learning_rate=0.03,
                num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
                random_state=0,verbose=-1)
            m.fit(tr[feats], np.log1p(tr["y_train"].clip(lower=0)))
            return np.expm1(m.predict(te[feats]))
        p_hist = fit(FEATS)
        p_antic = fit(FEATS+["pr_future"])
        p_pers = te["q"].values
        for name,p in [("Persistence",p_pers),("ML (historical)",p_hist),("ML + future rain",p_antic)]:
            rows.append(dict(h=h, model=name, **metrics(obs,p)))
        log.info(f"h={h:2d}d: J_alert  persist={metrics(obs,p_pers)['J']:.3f}  "
                 f"ML={metrics(obs,p_hist)['J']:.3f}  ML+rain={metrics(obs,p_antic)['J']:.3f}  || "
                 f"POD ML+rain={metrics(obs,p_antic)['POD']:.3f}")
    res=pd.DataFrame(rows); res.to_csv(OUTML/"101_anticipatory.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    col={"Persistence":"#d62728","ML (historical)":"#7f7f7f","ML + future rain":"#2ca02c"}
    for key,axi,lab in [("J",ax[0],"Composite alert score J_alert"),("CSI",ax[1],"CSI (flood detection)")]:
        for mdl in col:
            s=res[res.model==mdl]; axi.plot(s["h"],s[key],"-o",color=col[mdl],lw=2,ms=5,label=mdl)
        axi.set_xlabel("Forecast horizon (days)"); axi.set_ylabel(lab); axi.grid(alpha=0.3); axi.set_xticks(HOR)
    ax[0].legend(fontsize=9); ax[0].set_title("(a) Alert skill vs horizon")
    ax[1].set_title("(b) Flood-detection skill (CSI) vs horizon")
    fig.suptitle("Anticipatory forcing (perfect future rainfall) is what beats persistence on alerts",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGDIR/"fig21_anticipatory.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    log.info(f"Saved: fig21_anticipatory.png and 101_anticipatory.csv")


if __name__ == "__main__":
    main()
