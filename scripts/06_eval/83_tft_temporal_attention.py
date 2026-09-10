#!/usr/bin/env python3
"""
Script 83 — TFT temporal attention: how the model distributes focus over time.

HydroST uses an LSTM (no explicit temporal attention), so Script 82 used
integrated-gradient saliency. The TFT has an explicit self-attention layer over
the ENC-day encoder window — this script extracts those weights.

For the final query position (the day from which we forecast), we read how much
attention it places on each of the past ENC days. We compare:
  - Flood days (Q>Q90) vs baseflow days
  - The effective lookback (where cumulative attention reaches 50% / 90%)

Trains the Optuna-best TFT config once (from 81_best_params.csv), then analyses.

    python scripts/06_eval/83_tft_temporal_attention.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/qa_satellite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tft_attn")

# Import tuner (gives prepare_data + run_config) and TFT module
spec81 = importlib.util.spec_from_file_location("tune81", ROOT / "scripts/05_models/81_tune_tft_optuna.py")
S = importlib.util.module_from_spec(spec81)
spec81.loader.exec_module(S)
T = S.T   # the TFT-v2 module
DEVICE = S.DEVICE
Q_CONV = S.Q_CONV
Q90    = S.Q90


def temporal_attention(model, x):
    """Replay TFTLitev2.forward and capture self-attention weights.
    Returns attn for the LAST query position: (B, ENC)."""
    h = model.proj(model.vsn(x)) + model.pos
    o, _ = model.lstm(h)
    _, attn_w = model.attn(o, o, o, need_weights=True, average_attn_weights=True)  # (B,ENC,ENC)
    return attn_w[:, -1, :]   # last day attends to each past day


def main():
    data = S.prepare_data()
    bp = pd.read_csv(OUT_DIR / "81_best_params.csv", index_col=0)["0"].to_dict()
    cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]),
               DROP=float(bp["DROP"]), LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]),
               WD=float(bp["WD"]), ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
    log.info(f"Training Optuna-best TFT config: ENC={cfg['ENC']} HID={cfg['HID']} HEADS={cfg['HEADS']}")

    _, model = S.run_config(cfg, data, seed=42)
    model.eval()
    ENC = T.ENC   # set by run_config

    # Build test sequences + classify by regime
    Xseq, yseq, wseq, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
    obs_mm = data["obs_real_mm"][idx]
    real = np.isfinite(obs_mm)
    obs_m3 = obs_mm[real] / Q_CONV
    Xseq = Xseq[real]
    thr = Q90

    attn_flood, attn_base = [], []
    with torch.no_grad():
        for i in range(len(Xseq)):
            aw = temporal_attention(model, Xseq[i:i+1].to(DEVICE))[0].cpu().numpy()
            if obs_m3[i] > thr:
                attn_flood.append(aw)
            else:
                attn_base.append(aw)
    attn_flood = np.array(attn_flood)
    attn_base  = np.array(attn_base)
    log.info(f"Flood seqs: {len(attn_flood)}  Base seqs: {len(attn_base)}")

    mf = attn_flood.mean(0) if len(attn_flood) else np.zeros(ENC)
    mb = attn_base.mean(0)
    # orient by lag-before-forecast: index 0 = most recent day
    mf_r, mb_r = mf[::-1], mb[::-1]
    lags = np.arange(ENC)

    # Effective lookback
    def lookback(w):
        c = np.cumsum(w) / w.sum()
        return int(np.argmax(c >= 0.5)), int(np.argmax(c >= 0.9))
    f50, f90 = lookback(mf_r)
    b50, b90 = lookback(mb_r)
    log.info(f"Flood attention lookback: 50% in last {f50}d, 90% in last {f90}d")
    log.info(f"Base  attention lookback: 50% in last {b50}d, 90% in last {b90}d")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(lags, mb_r, "-o", ms=3, color="#4a90d9", label=f"Baseflow (n={len(attn_base)})")
    ax.plot(lags, mf_r, "-o", ms=3, color="#cc2222", label=f"Flood Q>Q90 (n={len(attn_flood)})")
    ax.axhline(1/ENC, color="gray", ls="--", lw=1, label="uniform")
    ax.set_xlabel("Lag before forecast (days; 0 = most recent)")
    ax.set_ylabel("Temporal attention weight (final query)")
    ax.set_title("TFT temporal attention by flow regime")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, min(30, ENC))

    ax = axes[1]
    cf = np.cumsum(mf_r) / mf_r.sum()
    cb = np.cumsum(mb_r) / mb_r.sum()
    ax.plot(lags, cb, color="#4a90d9", label="Baseflow")
    ax.plot(lags, cf, color="#cc2222", label="Flood")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("Lag before forecast (days)")
    ax.set_ylabel("Cumulative attention")
    ax.set_title("Effective lookback (cumulative attention)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, ENC)

    fig.suptitle("How the TFT distributes attention over the input window", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_tft_temporal_attention.png", dpi=150)
    plt.close(fig)
    log.info("Saved: 07_tft_temporal_attention.png")

    # Also a heatmap: attention matrix (query day x key day) averaged over flood
    if len(attn_flood):
        # rebuild full matrices for a flood subset
        mats = []
        with torch.no_grad():
            cnt = 0
            for i in range(len(Xseq)):
                if obs_m3[i] > thr:
                    h = model.proj(model.vsn(Xseq[i:i+1].to(DEVICE))) + model.pos
                    o, _ = model.lstm(h)
                    _, aw = model.attn(o, o, o, need_weights=True, average_attn_weights=True)
                    mats.append(aw[0].cpu().numpy())
                    cnt += 1
                    if cnt >= 60:
                        break
        M = np.mean(mats, 0)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M, cmap="magma", aspect="auto", origin="lower")
        ax.set_xlabel("Key day (attended-to)")
        ax.set_ylabel("Query day")
        ax.set_title("TFT attention matrix (flood days, avg)\nrow = query, col = attended day")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "08_tft_attention_matrix.png", dpi=150)
        plt.close(fig)
        log.info("Saved: 08_tft_attention_matrix.png")

    # ── Integrated-gradient temporal saliency (faithful attribution) ──────────
    # Attention weights ≠ feature importance (Jain & Wallace 2019). Compute IG
    # on P50 w.r.t. the input window to see what ACTUALLY drives the forecast.
    log.info("\nComputing IG temporal saliency for TFT (apples-to-apples vs HydroST)...")
    flood_seqs = [Xseq[i:i+1] for i in range(len(Xseq)) if obs_m3[i] > thr][:60]
    sal = np.zeros(ENC)
    steps = 16
    # cuDNN RNN backward requires training mode; disable cuDNN to keep eval-mode determinism
    with torch.backends.cudnn.flags(enabled=False):
        for xs in flood_seqs:
            x = xs.to(DEVICE)
            baseline = torch.zeros_like(x)
            ig = torch.zeros_like(x)
            for a in np.linspace(0, 1, steps):
                xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
                out = model(xi)[0, 1]   # P50
                g = torch.autograd.grad(out, xi)[0]
                ig += g
            ig = (ig / steps) * (x - baseline)
            sal += ig[0].abs().sum(-1).detach().cpu().numpy()  # sum over features -> (ENC,)
    sal /= len(flood_seqs)
    sal_r = sal[::-1]
    c = np.cumsum(sal_r) / sal_r.sum()
    s50 = int(np.argmax(c >= 0.5)); s90 = int(np.argmax(c >= 0.9))
    log.info(f"TFT IG saliency lookback: 50% in last {s50}d, 90% in last {s90}d "
             f"(attention said 50%@{f50}d) -> shows attention vs attribution gap")

    fig, ax = plt.subplots(figsize=(11, 6))
    lags_full = np.arange(ENC)
    ax.bar(lags_full, sal_r / sal_r.sum(), color="#2ca02c", alpha=0.7, label="IG attribution (what drives output)")
    ax.plot(lags_full, mf_r / mf_r.sum(), "-o", ms=3, color="#cc2222", label="Attention weight (where it looks)")
    ax.set_xlabel("Lag before forecast (days; 0 = most recent)")
    ax.set_ylabel("Normalised importance")
    ax.set_title("TFT: attention weights vs integrated-gradient attribution (flood days)\n"
                 "near-uniform attention but recency-concentrated attribution → 'attention is not explanation'")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(-0.5, min(30, ENC))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_tft_attention_vs_attribution.png", dpi=150)
    plt.close(fig)
    log.info("Saved: 09_tft_attention_vs_attribution.png")

    log.info("Done.")


if __name__ == "__main__":
    main()
