#!/usr/bin/env python3
"""
Script 93 — SHAP interpretability for the monthly anomaly model (LightGBM).

Complements the daily DL interpretability (integrated gradients + VSN, Scripts
82-84) with the standard tree-model attribution tool for the monthly management
product, which had no interpretability figure. SHAP (Lundberg & Lee 2017) gives
consistent, locally-accurate feature attributions.

Panels:
  (a) SHAP summary (beeswarm) -> global importance + direction of each feature.
  (b) SHAP dependence for the ENSO indices -> shows the opposite-sign effect of
      global ONI vs coastal on the streamflow anomaly (ties to Script 90).

Run in .venv:
    python scripts/06_eval/93_shap_monthly.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

ROOT   = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("shap_m")

TRAIN_END = "2014-12-31"

NICE = {
    "q_l1":"Streamflow (this month)","q_l2":"Streamflow (1-mo lag)","q_l3":"Streamflow (2-mo lag)",
    "pr_l1":"Precipitation (this month)","pr_2m":"Precipitation (2-mo sum)",
    "swvl4_l1":"Deep soil moisture (storage)","swvl1_l1":"Surface soil moisture",
    "oni_l1":"ONI (global El Nino)","spi90_l1":"SPI-90",
    "coastal_l1":"Coastal El Nino (Nino 1+2)",
}


def main():
    spec = importlib.util.spec_from_file_location("m86", ROOT/"scripts/05_models/86_monthly_water_availability.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    m = M.build_monthly().dropna(subset=["oni_l1","coastal_l1","y"])

    tr = m[m.index <= TRAIN_END]
    clim = tr.groupby("y_month")["y"].mean()
    clim_issue = tr.groupby(tr.index.month)["q"].mean()
    def anomalize(fr):
        a=fr.copy()
        a["q_l1"]=a["q_l1"]-a.index.month.map(clim_issue)
        a["q_l2"]=a["q_l2"]-((a.index.month-2)%12+1).map(clim_issue)
        a["q_l3"]=a["q_l3"]-((a.index.month-3)%12+1).map(clim_issue)
        return a
    tr_a = anomalize(tr)
    feats = ["q_l1","q_l2","q_l3","pr_l1","pr_2m","swvl4_l1","swvl1_l1","oni_l1","coastal_l1","spi90_l1"]
    y_anom = (tr["y"]-tr["y_month"].map(clim)).values

    mdl = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.03,
                            num_leaves=12, min_child_samples=20, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=2.0, random_state=0, verbose=-1)
    mdl.fit(tr_a[feats], y_anom)

    X = tr_a[feats].rename(columns=NICE)
    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(X)
    log.info(f"SHAP values: {sv.shape}")

    # Global importance ranking
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    log.info("Mean |SHAP| (global importance):")
    for k,v in imp.items(): log.info(f"  {k:32s} {v:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1,2,1)
    plt.sca(ax1)
    shap.summary_plot(sv, X, show=False, plot_size=None, max_display=10)
    ax1.set_title("(a) SHAP summary — monthly anomaly model\n(feature effect on predicted Q anomaly)", fontsize=10)

    # (b) dependence: ONI vs coastal on the same axes (opposite-sign check)
    ax2 = fig.add_subplot(1,2,2)
    oni = tr_a["oni_l1"].values; coa = tr_a["coastal_l1"].values
    sv_oni = sv[:, feats.index("oni_l1")]; sv_coa = sv[:, feats.index("coastal_l1")]
    ax2.scatter(oni, sv_oni, s=12, c="#1f77b4", alpha=0.6, label="ONI (global)")
    ax2.scatter(coa, sv_coa, s=12, c="#cc2222", alpha=0.6, label="Coastal (Nino 1+2)")
    # linear trend lines
    for x,sv_,c in [(oni,sv_oni,"#1f77b4"),(coa,sv_coa,"#cc2222")]:
        b=np.polyfit(x,sv_,1); xs=np.linspace(x.min(),x.max(),50); ax2.plot(xs,np.polyval(b,xs),color=c,lw=2)
    ax2.axhline(0,color="black",lw=0.6); ax2.axvline(0,color="black",lw=0.6)
    ax2.set_xlabel("ENSO index anomaly [degC]"); ax2.set_ylabel("SHAP value (effect on Q anomaly)")
    ax2.set_title("(b) SHAP dependence — ENSO indices\nopposite-sign effect: global (−) vs coastal (+)", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("SHAP interpretability of the monthly water-availability model",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig16_shap_monthly.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    b_oni=np.polyfit(oni,sv_oni,1)[0]; b_coa=np.polyfit(coa,sv_coa,1)[0]
    log.info(f"SHAP slope: ONI={b_oni:+.3f} (expect <0), coastal={b_coa:+.3f} (expect >0)")
    log.info(f"Saved: {FIGDIR/'fig16_shap_monthly.png'}")


if __name__ == "__main__":
    main()
