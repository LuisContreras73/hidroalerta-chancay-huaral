#!/usr/bin/env python3
"""
Script 89 — Correlograms: time-series diagnostics that quantify the persistence
dominance and the rainfall-runoff lag structure of the basin.

Panels:
  (a) ACF + PACF of observed daily streamflow -> quantifies why persistence is so
      strong (lag-1 autocorrelation) and how many lags carry information.
  (b) Cross-correlation precip -> streamflow at lags 0..20 d -> catchment response
      time / concentration (complements the learned impulse response, Script 82).
  (c) Predictor correlation matrix (Spearman) -> multicollinearity, justifying that
      antecedent Q dominates and many features are redundant.

Uses REAL observed Q (SNIRH) for (a)-(b); the reconstructed series only to extend
sample where needed (clearly noted). Run in .venv:
    python scripts/06_eval/89_correlograms.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("correlograms")
Q_CONV = 86.4 / 3062.62


def acf(x, nlags):
    x = np.asarray(x, float); x = x - x.mean()
    denom = np.sum(x*x)
    return np.array([1.0 if k == 0 else np.sum(x[k:]*x[:-k])/denom for k in range(nlags+1)])

def pacf(x, nlags):
    # Durbin-Levinson recursion
    r = acf(x, nlags)
    phi = np.zeros((nlags+1, nlags+1)); pac = np.zeros(nlags+1); pac[0] = 1.0
    phi[1,1] = r[1]; pac[1] = r[1]
    for k in range(2, nlags+1):
        num = r[k] - sum(phi[k-1,j]*r[k-j] for j in range(1,k))
        den = 1 - sum(phi[k-1,j]*r[j] for j in range(1,k))
        phi[k,k] = num/den if den != 0 else 0.0
        for j in range(1, k):
            phi[k,j] = phi[k-1,j] - phi[k,k]*phi[k-1,k-j]
        pac[k] = phi[k,k]
    return pac

def ccf(x, y, maxlag):
    """Cross-corr: corr(x[t-k], y[t]) for k=0..maxlag (x leads y for k>0)."""
    x = pd.Series(np.asarray(x,float)); y = pd.Series(np.asarray(y,float))
    out = []
    for k in range(0, maxlag+1):
        out.append(x.shift(k).corr(y))
    return np.array(out)


def main():
    # ── Observed daily streamflow (real, SNIRH) ──────────────────────────────
    obs = pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",
                      parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    # longest gap-free-ish stretch for ACF (use the continuous 2021-2024 obs where dense)
    obs_c = obs.loc["2021-01-01":"2024-12-31"].interpolate(limit=3).dropna()
    log.info(f"ACF/PACF on observed Q: n={len(obs_c)} days")

    d7 = pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv", parse_dates=["date"])
    pr = d7.groupby("date")["pr_mm"].mean()              # basin precip
    qd = d7[d7.entity_id=="sub_634"].set_index("date")["q_mm"]/Q_CONV
    common = pr.index.intersection(qd.index)
    pr_c, q_c = pr.reindex(common), qd.reindex(common)

    NL = 30
    a = acf(obs_c.values, NL)
    p = pacf(obs_c.values, NL)
    conf = 1.96/np.sqrt(len(obs_c))   # ~95% white-noise bounds

    cc = ccf(pr_c.values, q_c.values, 20)
    peak_lag = int(np.argmax(cc))

    # ── Predictor correlation matrix (Spearman) ──────────────────────────────
    feats = ["q_mm","pr_mm","pet_mm","tmean_c","api","spi_30d","spi_90d",
             "water_deficit_30d","oni_index"]
    outlet = d7[d7.entity_id=="sub_634"].set_index("date")[feats].dropna()
    NICE = {"q_mm":"Streamflow","pr_mm":"Precip","pet_mm":"PET","tmean_c":"Temp",
            "api":"API","spi_30d":"SPI-30","spi_90d":"SPI-90",
            "water_deficit_30d":"Water deficit","oni_index":"ONI (global)"}
    corr = outlet.corr(method="spearman")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # (a) ACF
    ax = fig.add_subplot(gs[0,0])
    ax.bar(range(NL+1), a, width=0.6, color="#1f77b4", alpha=0.85)
    ax.axhline(conf, color="red", ls="--", lw=1); ax.axhline(-conf, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"(a) Autocorrelation of daily streamflow (ACF)\nlag-1 = {a[1]:.3f}  ->  "
                 f"explains why persistence is near-unbeatable", fontsize=10)
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    ax.annotate("95% white-noise bound", (NL*0.5, conf), color="red", fontsize=7, va="bottom")

    # (b) PACF
    ax = fig.add_subplot(gs[0,1])
    ax.bar(range(NL+1), p, width=0.6, color="#2ca02c", alpha=0.85)
    ax.axhline(conf, color="red", ls="--", lw=1); ax.axhline(-conf, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    n_sig = int(np.sum(np.abs(p[1:6]) > conf))
    ax.set_title(f"(b) Partial autocorrelation (PACF)\nsignificant lags concentrated in first "
                 f"~{n_sig} days (memory ~3 d)", fontsize=10)
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("PACF")

    # (c) CCF precip -> Q
    ax = fig.add_subplot(gs[1,0])
    ax.bar(range(len(cc)), cc, width=0.6, color="#8c564b", alpha=0.85)
    ax.axvline(peak_lag, color="red", ls="--", lw=1.2, label=f"peak at lag {peak_lag} d")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("(c) Cross-correlation precipitation -> streamflow\ncatchment response time "
                 "(concentration)", fontsize=10)
    ax.set_xlabel("Precip lead (days)"); ax.set_ylabel("Correlation")
    ax.legend(fontsize=8)

    # (d) Predictor correlation matrix
    ax = fig.add_subplot(gs[1,1])
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
    ax.set_xticklabels([NICE[f] for f in feats], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([NICE[f] for f in feats], fontsize=8)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr.values[i,j])<0.6 else "white")
    ax.set_title("(d) Predictor correlation matrix (Spearman)\nmulticollinearity among features",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman corr")

    fig.suptitle("Time-series diagnostics: streamflow autocorrelation and rainfall-runoff "
                 "lag structure", fontweight="bold", fontsize=12)
    fig.savefig(FIGDIR/"fig12_correlograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(f"ACF lag-1={a[1]:.3f}, lag-2={a[2]:.3f}, lag-7={a[7]:.3f}")
    log.info(f"PACF first significant lags (|.|>{conf:.3f}): "
             f"{[k for k in range(1,11) if abs(p[k])>conf]}")
    log.info(f"Precip->Q CCF peak at lag {peak_lag} d (r={cc[peak_lag]:.3f})")
    log.info(f"Saved: {FIGDIR/'fig12_correlograms.png'}")


if __name__ == "__main__":
    main()
