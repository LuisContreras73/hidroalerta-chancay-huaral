#!/usr/bin/env python3
"""
Script 107 — FAIR multi-horizon comparison (equal information).

Addresses the fairness concern: if RA-TFT gets future rainfall, a simple baseline
must get the SAME future rainfall. Combines:
  - Persistence (no forcing) .......... floor / value-of-anticipation reference
  - LightGBM + future rain ............ anticipatory baseline (Script 101)
  - RA-TFT + realistic rain forecast .. our architecture, anticipatory (Script 105)
all at horizons 1..14, honest test 2024.

Verdict this produces: whether the RA-TFT architecture adds value BEYOND simply
having the anticipatory rainfall (spoiler from the numbers: LightGBM+rain wins).

Run in .venv (matplotlib/pandas):
    python scripts/06_eval/107_fair_comparison.py
"""
import logging
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"; FIG=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("fair")

def main():
    lg=pd.read_csv(OUTML/"101_anticipatory.csv")     # h,model,NSE,J,CSI,POD,FAR
    ra=pd.read_csv(OUTML/"105_ratft_multihorizon.csv")  # model,h,NSE,J,CSI
    series={
        "Persistence (no forcing)":      lg[lg.model=="Persistence"].set_index("h"),
        "LightGBM + future rain":        lg[lg.model=="ML + future rain"].set_index("h"),
        "RA-TFT + realistic rain":       ra[ra.model=="RA-TFT (+realistic rain forecast)"].set_index("h"),
    }
    col={"Persistence (no forcing)":"#d62728","LightGBM + future rain":"#2ca02c","RA-TFT + realistic rain":"#1f77b4"}

    fig,ax=plt.subplots(1,2,figsize=(15,5.5))
    for key,axi,lab in [("NSE",ax[0],"NSE"),("J",ax[1],"J_alert")]:
        for name,dfm in series.items():
            s=dfm.sort_index(); axi.plot(s.index, s[key], "-o", color=col[name], lw=2, ms=4, label=name)
        axi.set_xlabel("Forecast horizon (days)"); axi.set_ylabel(lab); axi.grid(alpha=0.3); axi.set_xticks([1,2,3,5,7,10,14])
    ax[0].legend(fontsize=8); ax[0].set_title("(a) Continuous skill — equal information")
    ax[1].set_title("(b) Alert skill — equal information")
    fig.suptitle("Fair comparison (all anticipatory models get the SAME future rain):\n"
                 "the anticipatory FORCING beats persistence; the RA-TFT architecture does NOT beat simple LightGBM+rain",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG/"fig23_fair_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # table
    log.info("\n=== FAIR COMPARISON (J_alert / NSE by horizon) ===")
    for h in [1,3,7,14]:
        row=" | ".join(f"{n.split(' (')[0].split(' +')[0][:9]}: J={series[n].loc[h,'J']:.3f} NSE={series[n].loc[h,'NSE']:.3f}"
                       for n in series if h in series[n].index)
        log.info(f"h={h:2d}: {row}")
    log.info(f"Saved: {FIG/'fig23_fair_comparison.png'}")

if __name__=="__main__":
    main()
