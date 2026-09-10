#!/usr/bin/env python3
"""
Script 87 — Coastal vs global El Niño: which index predicts THIS basin?

Peru experiences TWO distinct El Niño phenomena (Takahashi et al. 2014; ENFEN):
  - "Global"/canonical El Niño  -> ONI / Niño 3.4 (central Pacific). Already used.
  - "Coastal" El Niño (El Niño Costero) -> ICEN / Niño 1+2 (0-10°S, 80-90°W),
    off the Peru-Ecuador coast. Drove the catastrophic 2017 and 2023 coastal
    floods WITH a near-neutral ONI — the two indices decouple.

For a western-Andes coastal basin (Chancay-Huaral, ~11°S) the coastal index is
physically expected to be the more relevant rainfall/flood driver. This script:
  1. Downloads all Niño-region SST anomalies from NOAA (ERSSTv5).
  2. Builds a coastal index (ICEN proxy = 3-month running mean of Niño 1+2 anom).
  3. Compares coastal vs global as predictors of basin monthly precip and Q,
     overall and in the wet season, across lags — quantifying which matters here.
  4. Saves the coastal index for use as a model covariate.

Run in .venv:
    python scripts/06_eval/87_enso_coastal_vs_global.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT    = Path(__file__).resolve().parent.parent.parent
ENSO_DIR = ROOT / "data/silver/enso"
FIG_DIR = ROOT / "outputs/qa_satellite"
ENSO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("enso_coastal")

NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
Q_CONV = 86.4 / 3062.62
WET = [12, 1, 2, 3, 4]

# ICEN (IGP/ENFEN) coastal-event categories on the 3-mo Niño 1+2 anomaly
ICEN_CATS = [(-np.inf,-1.4,"La Niña Costera fuerte"), (-1.4,-1.0,"Fría moderada"),
             (-1.0,-0.4,"Fría débil"), (-0.4,0.4,"Neutral"), (0.4,1.0,"Cálida débil"),
             (1.0,1.7,"El Niño Costero moderado"), (1.7,3.0,"Fuerte"), (3.0,np.inf,"Extraordinario")]


def download_nino():
    log.info("Downloading NOAA ERSSTv5 Niño regions...")
    r = requests.get(NINO_URL, timeout=60); r.raise_for_status()
    rows = []
    for line in r.text.strip().split("\n")[1:]:
        p = line.split()
        if len(p) < 10:
            continue
        rows.append(dict(year=int(p[0]), month=int(p[1]),
                         nino12=float(p[2]),  nino12_anom=float(p[3]),
                         nino3=float(p[4]),   nino3_anom=float(p[5]),
                         nino4=float(p[6]),   nino4_anom=float(p[7]),
                         nino34=float(p[8]),  nino34_anom=float(p[9])))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    return df.set_index("date").sort_index()


def categorize_icen(v):
    for lo, hi, name in ICEN_CATS:
        if lo <= v < hi:
            return name
    return "Neutral"


def main():
    nino = download_nino()
    # Coastal index (ICEN proxy): 3-month running mean of Niño 1+2 anomaly
    nino["coastal"] = nino["nino12_anom"].rolling(3, center=True, min_periods=2).mean()
    # Global index = ONI (3-mo running mean of Niño 3.4 anomaly)
    nino["global"] = nino["nino34_anom"].rolling(3, center=True, min_periods=2).mean()

    # Save coastal index
    out = nino[["nino12_anom", "coastal", "nino34_anom", "global"]].copy()
    out["icen_category"] = out["coastal"].map(categorize_icen)
    out.to_csv(ENSO_DIR / "S3b_enso_coastal_global.csv")
    log.info(f"Saved coastal/global index: S3b_enso_coastal_global.csv ({len(out)} months)")

    # Recent decoupling highlight
    recent = out.loc["2023-01":].head(4)
    log.info(f"\n2023 (coastal flood year) — coastal vs global:")
    for d, row in recent.iterrows():
        log.info(f"  {d.date()}  coastal={row['coastal']:+.2f}  global={row['global']:+.2f}  [{row['icen_category']}]")

    # ── Basin monthly precip + Q ──────────────────────────────────────────────
    d7 = pd.read_csv(ROOT / "data/model_ready/D7_multientity.csv", parse_dates=["date"])
    pr_basin = d7.groupby("date")["pr_mm"].mean().resample("MS").sum()
    q_basin  = (d7[d7.entity_id=="sub_634"].set_index("date")["q_mm"]/Q_CONV).resample("MS").mean()

    df = pd.DataFrame({"pr": pr_basin, "q": q_basin}).join(out[["coastal","global"]])
    df = df.loc["1981":"2025"].dropna()

    # ── Correlation comparison (overall + wet season) across lags ─────────────
    log.info("\n" + "="*64)
    log.info("PREDICTOR COMPARISON — Spearman corr with basin precip / Q")
    log.info("(index leads target by k months; wet=Dec-Apr)")
    log.info("="*64)
    log.info(f"{'lag':>4} | {'pr~coastal':>11} {'pr~global':>10} | {'q~coastal':>10} {'q~global':>9}")
    results = []
    for k in range(0, 4):
        c_pr = df["coastal"].shift(k).corr(df["pr"], method="spearman")
        g_pr = df["global"].shift(k).corr(df["pr"], method="spearman")
        c_q  = df["coastal"].shift(k).corr(df["q"],  method="spearman")
        g_q  = df["global"].shift(k).corr(df["q"],  method="spearman")
        results.append((k, c_pr, g_pr, c_q, g_q))
        log.info(f"{k:>4} | {c_pr:>11.3f} {g_pr:>10.3f} | {c_q:>10.3f} {g_q:>9.3f}")

    # Wet-season only (target month in wet season)
    wet_mask = df.index.month.isin(WET)
    dfw = df[wet_mask]
    log.info("\nWet season only (target Dec-Apr):")
    log.info(f"{'lag':>4} | {'pr~coastal':>11} {'pr~global':>10} | {'q~coastal':>10} {'q~global':>9}")
    wet_results = []
    for k in range(0, 4):
        c_pr = df["coastal"].shift(k)[wet_mask].corr(dfw["pr"], method="spearman")
        g_pr = df["global"].shift(k)[wet_mask].corr(dfw["pr"], method="spearman")
        c_q  = df["coastal"].shift(k)[wet_mask].corr(dfw["q"],  method="spearman")
        g_q  = df["global"].shift(k)[wet_mask].corr(dfw["q"],  method="spearman")
        wet_results.append((k, c_pr, g_pr, c_q, g_q))
        log.info(f"{k:>4} | {c_pr:>11.3f} {g_pr:>10.3f} | {c_q:>10.3f} {g_q:>9.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: time series coastal vs global, mark 2017 & 2023 coastal events
    ax = axes[0]
    ts = out.loc["2010":]
    ax.plot(ts.index, ts["global"],  color="#1f77b4", lw=1.3, label="Global (ONI / Niño 3.4)")
    ax.plot(ts.index, ts["coastal"], color="#cc2222", lw=1.5, label="Coastal (ICEN proxy / Niño 1+2)")
    ax.axhline(0.4, color="gray", ls=":", lw=1); ax.axhline(-0.4, color="gray", ls=":", lw=1)
    for yr in [2017, 2023]:
        ax.axvspan(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-05-01"),
                   color="#ffd6d6", alpha=0.5)
        ax.annotate(f"{yr}\ncostero", (pd.Timestamp(f"{yr}-03-01"), ax.get_ylim()[1]*0.8),
                    ha="center", fontsize=8, color="#cc2222")
    ax.set_ylabel("SST anomaly (°C)"); ax.set_title("Coastal vs global El Niño (2010-2026)\nshaded = 2017/2023 coastal floods")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: correlation bars (wet season, lag 0) precip & Q
    ax = axes[1]
    k0 = wet_results[0]
    labels = ["Precip\n(wet)", "Q\n(wet)"]
    coastal_vals = [k0[1], k0[3]]
    global_vals  = [k0[2], k0[4]]
    x = np.arange(2)
    ax.bar(x-0.2, coastal_vals, 0.4, color="#cc2222", alpha=0.85, label="Coastal (Niño 1+2)")
    ax.bar(x+0.2, global_vals,  0.4, color="#1f77b4", alpha=0.85, label="Global (Niño 3.4)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman correlation (wet season, lag 0)")
    ax.set_title("Which El Niño index predicts the basin?\nwet-season precip & streamflow")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    for i,(c,g) in enumerate(zip(coastal_vals, global_vals)):
        ax.text(i-0.2, c, f"{c:.2f}", ha="center", va="bottom" if c>=0 else "top", fontsize=8)
        ax.text(i+0.2, g, f"{g:.2f}", ha="center", va="bottom" if g>=0 else "top", fontsize=8)

    fig.tight_layout(); fig.savefig(FIG_DIR / "17_enso_coastal_vs_global.png", dpi=150); plt.close(fig)
    log.info("\nSaved: 17_enso_coastal_vs_global.png")
    log.info("Done.")


if __name__ == "__main__":
    main()
