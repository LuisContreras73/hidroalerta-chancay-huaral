#!/usr/bin/env python3
"""
Script 84 — TFT event-centered interpretability (Lim et al. 2021 style).

The original TFT paper showed that around significant events (e.g. the 2008
financial crisis in the S&P 500 volatility data) the model's behaviour changed:
the temporal attention shifted to specific dates and the Variable Selection
Network (VSN) re-weighted which inputs mattered. This script reproduces that
analysis for our streamflow floods:

  1. VARIABLE SELECTION (VSN) — extract the softmax selection weights the TFT
     assigns to each of the 41 inputs. Rank them, and compare FLOOD vs BASEFLOW
     days to see which variables the model emphasises during peak events.

  2. REGIME-CHANGE DETECTION — for each test day compute the temporal attention
     vector; measure how far it deviates from the average pattern. Spikes in this
     deviation, overlaid on the Q hydrograph, reveal whether the model "notices"
     flood onsets (analogous to the paper's 2008-crisis attention spike).

  3. EVENT ZOOM — for the single largest flood, show that day's attention-over-
     lags and top variables vs a calm reference day.

    python scripts/06_eval/84_tft_event_interpretability.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/qa_satellite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tft_event")

spec81 = importlib.util.spec_from_file_location("tune81", ROOT / "scripts/05_models/81_tune_tft_optuna.py")
S = importlib.util.module_from_spec(spec81)
spec81.loader.exec_module(S)
T = S.T
DEVICE = S.DEVICE
Q_CONV = S.Q_CONV
Q90    = S.Q90

# Nombres legibles con sub-cuencas REALES (Alto Chancay=649, Carac=655, Baños=656).
def _f(var, loc):
    names = {"basin": "basin", "649": "Alto Chancay", "655": "Carac", "656": "Baños"}
    return f"{var}, {names[loc]}"

FEAT_NAMES = [
    _f("Precip","basin"),_f("Precip","649"),_f("Precip","655"),_f("Precip","656"),
    _f("Tmax","basin"),_f("Tmax","649"),_f("Tmax","655"),_f("Tmax","656"),
    _f("Tmin","basin"),_f("Tmin","649"),_f("Tmin","655"),_f("Tmin","656"),
    _f("PET","basin"),_f("PET","649"),_f("PET","655"),_f("PET","656"),
    _f("API","basin"),_f("API","649"),_f("API","655"),_f("API","656"),
    _f("SPI-30","basin"),_f("SPI-30","649"),_f("SPI-30","655"),_f("SPI-30","656"),
    _f("SPI-90","basin"),_f("SPI-90","649"),_f("SPI-90","655"),_f("SPI-90","656"),
    _f("Water deficit","basin"),_f("Water deficit","649"),_f("Water deficit","655"),_f("Water deficit","656"),
    "ONI (global)","Seasonality (sin)","Seasonality (cos)","Hydrological month","Wet-season flag",
    "Streamflow (today)","Streamflow (7-day lag)","Streamflow (7-day mean)","Streamflow (30-day mean)",
]


def vsn_weights(model, x):
    """Selection weights = softmax(VSN.w(x)); shape (B, ENC, nf)."""
    return torch.softmax(model.vsn.w(x), dim=-1)


def temporal_attn(model, x):
    h = model.proj(model.vsn(x)) + model.pos
    o, _ = model.lstm(h)
    _, aw = model.attn(o, o, o, need_weights=True, average_attn_weights=True)
    return aw[:, -1, :]   # (B, ENC)


def main():
    data = S.prepare_data()
    bp = pd.read_csv(OUT_DIR / "81_best_params.csv", index_col=0)["0"].to_dict()
    cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]),
               DROP=float(bp["DROP"]), LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]),
               WD=float(bp["WD"]), ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
    log.info(f"Training Optuna-best TFT (ENC={cfg['ENC']})...")
    _, model = S.run_config(cfg, data, seed=42)
    model.eval()
    ENC = T.ENC

    # Test sequences aligned to dates + real obs
    Xseq, yseq, wseq, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
    seq_dates = pd.DatetimeIndex(data["dates"][idx])
    obs_mm = data["obs_real_mm"][idx]
    real = np.isfinite(obs_mm)
    Xseq, seq_dates, obs_m3 = Xseq[real], seq_dates[real], obs_mm[real] / Q_CONV

    # ── Collect VSN weights + attention per day ──────────────────────────────
    vsn_all, attn_all = [], []
    with torch.no_grad():
        for i in range(len(Xseq)):
            xi = Xseq[i:i+1].to(DEVICE)
            vsn_all.append(vsn_weights(model, xi)[0].mean(0).cpu().numpy())   # mean over lags -> (nf,)
            attn_all.append(temporal_attn(model, xi)[0].cpu().numpy())        # (ENC,)
    vsn_all  = np.array(vsn_all)    # (T, nf)
    attn_all = np.array(attn_all)   # (T, ENC)

    flood = obs_m3 > Q90
    log.info(f"Test days: {len(obs_m3)}  flood(Q>Q90): {flood.sum()}")

    # ── 1. VSN variable selection: overall + flood vs base ───────────────────
    imp_all  = vsn_all.mean(0)
    imp_flood = vsn_all[flood].mean(0) if flood.sum() else imp_all
    imp_base  = vsn_all[~flood].mean(0)

    order = np.argsort(-imp_all)[:15]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(order))
    ax.barh(y - 0.2, imp_base[order],  0.4, color="#4a90d9", label="Baseflow", alpha=0.85)
    ax.barh(y + 0.2, imp_flood[order], 0.4, color="#cc2222", label="Flood (Q>Q90)", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels([FEAT_NAMES[i] for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1/len(FEAT_NAMES), color="gray", ls="--", lw=1, label="uniform (1/41)")
    ax.set_xlabel("VSN selection weight")
    ax.set_title("TFT Variable Selection Network — top 15 inputs\n"
                 "how variable importance shifts on flood vs baseflow days")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / "10_tft_vsn_importance.png", dpi=150); plt.close(fig)
    log.info("Saved: 10_tft_vsn_importance.png")

    # Which variables increase most during floods
    delta = imp_flood - imp_base
    up = np.argsort(-delta)[:8]
    log.info("\nVariables the TFT emphasises MORE during floods (Δ selection weight):")
    for i in up:
        log.info(f"  {FEAT_NAMES[i]:14s}: base={imp_base[i]:.4f} -> flood={imp_flood[i]:.4f}  (Δ{delta[i]:+.4f})")

    # ── 2. Regime-change detection: attention deviation over time ─────────────
    mean_attn = attn_all.mean(0)
    # deviation metric: L1 distance of each day's attention from the mean pattern
    dev = np.abs(attn_all - mean_attn).sum(1)
    dev_series = pd.Series(dev, index=seq_dates).sort_index()
    q_series   = pd.Series(obs_m3, index=seq_dates).sort_index()

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.fill_between(q_series.index, 0, q_series.values, color="#4a90d9", alpha=0.35, label="Q obs (m³/s)")
    ax1.axhline(Q90, color="#1f4e79", ls=":", lw=1, label="Q90 alert")
    ax1.set_ylabel("Q (m³/s)", color="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(dev_series.index, dev_series.values, color="#cc2222", lw=1.2, label="attention deviation")
    ax2.set_ylabel("Attention pattern deviation (L1 from mean)", color="#cc2222")
    ax1.set_title("Regime-change detection: does the TFT's attention shift around flood peaks?\n"
                  "(red = how unusual the attention pattern is each day)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88), fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "11_tft_regime_change.png", dpi=150); plt.close(fig)
    log.info("Saved: 11_tft_regime_change.png")

    # Correlation between attention deviation and flow level / flow rise
    q_aligned = q_series.reindex(dev_series.index)
    rise = q_aligned.diff().abs()
    corr_level = np.corrcoef(dev_series.values, q_aligned.values)[0, 1]
    valid = rise.notna()
    corr_rise  = np.corrcoef(dev_series.values[valid.values], rise.values[valid.values])[0, 1]
    log.info(f"\nAttention-deviation correlation: with Q level r={corr_level:.3f}, "
             f"with |dQ/dt| r={corr_rise:.3f}")

    # ── 3. Event zoom: largest flood vs calm day ─────────────────────────────
    peak_i = int(np.argmax(obs_m3))
    calm_i = int(np.argmin(np.abs(obs_m3 - np.median(obs_m3))))
    log.info(f"\nPeak event: {seq_dates[peak_i].date()} Q={obs_m3[peak_i]:.1f} | "
             f"calm ref: {seq_dates[calm_i].date()} Q={obs_m3[calm_i]:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    lags = np.arange(ENC)
    axes[0].plot(lags, attn_all[peak_i][::-1], "-o", ms=3, color="#cc2222", label=f"peak {seq_dates[peak_i].date()}")
    axes[0].plot(lags, attn_all[calm_i][::-1], "-o", ms=3, color="#4a90d9", label=f"calm {seq_dates[calm_i].date()}")
    axes[0].set_xlabel("Lag before forecast (days)"); axes[0].set_ylabel("Attention weight")
    axes[0].set_title("Attention over lags: peak vs calm"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, min(30, ENC))

    o2 = np.argsort(-(vsn_all[peak_i]))[:12]
    yy = np.arange(len(o2))
    axes[1].barh(yy - 0.2, vsn_all[calm_i][o2], 0.4, color="#4a90d9", label="calm", alpha=0.85)
    axes[1].barh(yy + 0.2, vsn_all[peak_i][o2], 0.4, color="#cc2222", label="peak", alpha=0.85)
    axes[1].set_yticks(yy); axes[1].set_yticklabels([FEAT_NAMES[i] for i in o2], fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_xlabel("VSN selection weight")
    axes[1].set_title("Variable selection: peak vs calm"); axes[1].legend(); axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / "12_tft_event_zoom.png", dpi=150); plt.close(fig)
    log.info("Saved: 12_tft_event_zoom.png")

    log.info("Done.")


if __name__ == "__main__":
    main()
