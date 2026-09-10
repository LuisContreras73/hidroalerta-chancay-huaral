#!/usr/bin/env python3
"""
Script 97 — Intuitive temporal-influence figure (replaces the hard-to-read
raw-attention fig18).

Why: the TFT's attention weights are near-uniform (that IS the finding), so a raw
attention plot looks empty/confusing. The faithful, intuitive way to show the
temporal mechanism is integrated-gradient ATTRIBUTION (what actually drives the
forecast), presented as:
  (a) "How far back does the model look?" — attribution per lag day + cumulative
      curve with 50%/90% effective-memory markers.
  (b) Flood case study — the observed streamflow in the input window, with each
      day shaded by its attribution to that specific forecast (spotlight on the
      recent rising limb).

Run in .venv313:
    python scripts/06_eval/97_temporal_influence_figure.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("temporal")
Q90 = 40.89


def ig_per_lag(model, x, steps=24):
    """Integrated-gradient attribution of P50 to each input, summed over features
    -> vector of length ENC (one value per lag day)."""
    x = x.clone()
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    with torch.backends.cudnn.flags(enabled=False):
        for a in np.linspace(0, 1, steps):
            xi = (baseline + a*(x-baseline)).clone().requires_grad_(True)
            out = model(xi)[0, 1]  # P50
            g = torch.autograd.grad(out, xi)[0]
            total += g
    ig = ((total/steps)*(x-baseline))[0].abs().sum(-1).detach().cpu().numpy()  # (ENC,)
    return ig


def main():
    spec = importlib.util.spec_from_file_location("tune81", ROOT/"scripts/05_models/81_tune_tft_optuna.py")
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    T = S.T
    data = S.prepare_data()
    bp = pd.read_csv(OUTML/"81_best_params.csv", index_col=0)["0"].to_dict()
    cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]),
               DROP=float(bp["DROP"]), LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]),
               WD=float(bp["WD"]), ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
    log.info("Training Optuna-best TFT...")
    _, model = S.run_config(cfg, data, seed=42); model.eval()
    ENC = T.ENC; DEV = S.DEVICE

    Xseq, y, w, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
    obs = data["obs_real_mm"][idx]; real = np.isfinite(obs)
    Xr, idxr, obsr = Xseq[real], np.array(idx)[real], obs[real]/S.Q_CONV
    flood = obsr > Q90
    log.info(f"Test real seqs={len(Xr)}, floods={int(flood.sum())}")

    # ── (a) aggregate attribution per lag over flood sequences ────────────────
    flood_pos = np.where(flood)[0][:60]
    accum = np.zeros(ENC)
    for j in flood_pos:
        accum += ig_per_lag(model, Xr[j:j+1].to(DEV))
    accum /= len(flood_pos)
    recent = accum[::-1]                      # index 0 = most recent day
    cum = np.cumsum(recent)/recent.sum()
    d50 = int(np.argmax(cum>=0.5)); d90 = int(np.argmax(cum>=0.9))
    log.info(f"Effective memory: 50% within last {d50}d, 90% within last {d90}d")

    # ── (b) largest flood case study ──────────────────────────────────────────
    jstar = flood_pos[np.argmax(obsr[flood_pos])]
    g = idxr[jstar]
    ig_case = ig_per_lag(model, Xr[jstar:jstar+1].to(DEV))[::-1]   # idx0=most recent
    win_dates = pd.DatetimeIndex(data["dates"][g-ENC:g])
    q_win = (data["df"]["q_mm"].reindex(win_dates).values)/S.Q_CONV # m3/s, oldest->newest
    q_win_recent = q_win[::-1]                                      # idx0 = most recent
    tgt_date = pd.Timestamp(data["dates"][g]); tgt_obs = obsr[jstar]
    log.info(f"Case study: forecast for {tgt_date.date()} (obs {tgt_obs:.1f} m3/s)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # (a) attribution per lag + cumulative
    lags = np.arange(ENC)
    a0 = ax[0]
    a0.bar(lags, recent/recent.sum(), color="#08519c", alpha=0.8, label="attribution per lag")
    a0b = a0.twinx()
    a0b.plot(lags, cum, color="#cc2222", lw=2, label="cumulative")
    a0b.axhline(0.9, color="gray", ls=":", lw=1); a0b.axhline(0.5, color="gray", ls=":", lw=1)
    a0b.set_ylim(0,1.02); a0b.set_ylabel("cumulative share", color="#cc2222")
    a0.axvline(d90, color="red", ls="--", lw=1.2)
    a0.annotate(f"90% within last {d90} days", (d90+0.5, a0.get_ylim()[1]*0.8), color="red", fontsize=9)
    a0.set_xlim(-0.5, min(30,ENC)); a0.set_xlabel("Lag before forecast (days; 0 = most recent)")
    a0.set_ylabel("Attribution share (integrated gradients)")
    a0.set_title(f"(a) How far back does the model look?\n50% within last {d50} d, 90% within {d90} d "
                 "(≈ catchment memory)", fontsize=10)

    # (b) case study: streamflow window colored by attribution
    a1 = ax[1]
    x_ax = -lags  # most recent at 0, older to the left (negative)
    a1.plot(x_ax, q_win_recent, "-", color="gray", lw=1.2, zorder=1)
    sc = a1.scatter(x_ax, q_win_recent, c=ig_case/ig_case.max(), cmap="viridis",
                    s=30+180*(ig_case/ig_case.max()), zorder=3)
    a1.axhline(Q90, color="red", ls=":", lw=1, label="Q90 alert")
    a1.scatter([1],[tgt_obs], marker="*", s=300, color="red", edgecolor="white", zorder=4,
               label=f"forecast target ({tgt_date.date()})")
    a1.set_xlim(-min(30,ENC), 3); a1.set_xlabel("Lag before forecast (days)")
    a1.set_ylabel("Streamflow [m³/s]")
    a1.set_title(f"(b) Flood case study ({tgt_date.date()}, obs {tgt_obs:.0f} m³/s)\n"
                 "point size/color = influence on the forecast → recent rising limb dominates", fontsize=10)
    a1.legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=a1, fraction=0.046, label="relative influence")

    fig.suptitle("Temporal influence of the TFT (integrated-gradient attribution) — "
                 "the model forecasts from the recent flow trajectory", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig18_tft_attention.png", dpi=300, bbox_inches="tight")  # replaces old fig18
    plt.close(fig)
    log.info(f"Saved (replaces old fig18): {FIGDIR/'fig18_tft_attention.png'}")


if __name__ == "__main__":
    main()
