#!/usr/bin/env python3
"""
Script 95 — Dedicated figure for the evaluation-integrity contribution (C1).

Shows WHY and HOW MUCH evaluating against gap-filled streamflow inflates alert
skill. Two panels:
  (a) Timeline of the 2024-2025 test period: which days are REAL observations
      vs GR4J-reconstructed (gap-filled). Makes the 423-of-730 problem visual.
  (b) Metrics computed on the contaminated (730 d) vs honest (423 real d) set,
      with the inflation annotated. Numbers from Script 80 (HydroST, same model).

Run in .venv (data only, no GPU):
    python scripts/06_eval/95_contamination_figure.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("contam")
Q_CONV = 86.4/3062.62

# Metrics from Script 80 (HydroST evaluated on contaminated 730 d vs honest 423 real d)
METRICS = {  # metric : (contaminated, honest)
    "J_alert": (0.693, 0.475), "CSI": (0.667, 0.258),
    "POD": (0.841, 0.471), "FAR": (0.237, 0.636),
}


def main():
    # ── Reconstruct real-vs-filled mask for the test period ──────────────────
    d7 = pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv", parse_dates=["date"])
    q_d7 = (d7[d7.entity_id=="sub_634"].set_index("date")["q_next_1d"]/Q_CONV)  # target (filled)
    obs = pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",
                      parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    obs_next = obs.shift(-1)  # q_next_1d convention

    test = pd.DataFrame({"target": q_d7}).loc["2024-01-01":"2025-12-31"]
    test["real"] = obs_next.reindex(test.index)
    is_real = test["real"].notna().values
    n_real = int(is_real.sum()); n_tot = len(test)
    log.info(f"Test: {n_tot} days, {n_real} real ({n_real/n_tot*100:.0f}%), {n_tot-n_real} GR4J-filled")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios":[1.6,1]})

    # ── (a) Timeline ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(test.index, test["target"], "-", color="#e8a000", lw=1.0,
            label=f"GR4J-reconstructed target ({n_tot-n_real} days)", zorder=1)
    real_pts = test[is_real]
    ax.plot(real_pts.index, real_pts["real"], ".", color="black", ms=3,
            label=f"Real observation ({n_real} days)", zorder=3)
    # shade filled stretches
    filled = ~is_real; ingap=False
    for i,d in enumerate(test.index):
        if filled[i] and not ingap: g0=d; ingap=True
        if (not filled[i] or i==len(test.index)-1) and ingap:
            ax.axvspan(g0, test.index[i], color="#ffe4e4", alpha=0.7, zorder=0); ingap=False
    ax.axvspan(np.nan,np.nan,color="#ffe4e4",label="gauge gap (GR4J-filled)")
    ax.axhline(40.89, color="red", ls=":", lw=1, label="Q90 alert threshold")
    ax.set_ylabel("Streamflow [m³/s]")
    ax.set_title(f"(a) Test period 2024-2025: only {n_real}/{n_tot} days are real observations\n"
                 "2025 gauge largely down → target reconstructed with GR4J", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    # ── (b) Metric inflation ──────────────────────────────────────────────────
    ax = axes[1]
    mets = list(METRICS.keys())
    contam = [METRICS[m][0] for m in mets]; honest = [METRICS[m][1] for m in mets]
    x = np.arange(len(mets))
    ax.bar(x-0.2, contam, 0.4, color="#cc2222", alpha=0.85, label="Contaminated (730 d)")
    ax.bar(x+0.2, honest, 0.4, color="#2ca02c", alpha=0.85, label="Honest (423 real d)")
    for i,m in enumerate(mets):
        d = METRICS[m][1]-METRICS[m][0]
        ax.annotate(f"{d:+.2f}", (i, max(contam[i],honest[i])+0.03), ha="center", fontsize=8,
                    color="black")
    ax.set_xticks(x); ax.set_xticklabels(mets)
    ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.0)
    ax.set_title("(b) Alert metrics: contaminated vs honest\nJ_alert inflated by +0.22, CSI ×2.6", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Evaluating against gap-filled streamflow inflates extreme-event skill "
                 "(evaluation-integrity contribution)", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig02_contamination.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {FIGDIR/'fig02_contamination.png'}")


if __name__ == "__main__":
    main()
