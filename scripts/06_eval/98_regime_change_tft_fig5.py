#!/usr/bin/env python3
"""
Script 98 — TFT "Figure-5"-style regime-change visualisation, adapted.

Inspired by Fig. 5 of Lim et al. (2021) but adapted to our architecture/data:
detect WHEN the model's internal behaviour changes and compare its attention
between a typical and an atypical regime.

Adaptation (honest): our TFT's attention is near-uniform, so raw attention
distance is a flat/misleading regime metric. We instead drive the bottom panel
with the change of the model's INTERNAL STATE (L2 distance of the daily latent
embedding from the mean embedding), which genuinely tracks regime; the attention
deviation is overlaid (faint) to show it stays flat. The top panels still show
the ATTENTION distribution (as requested), revealing the key message: the state
changes at floods while attention does not.

Layout:
  bottom  : streamflow + internal-state-change curve + shaded regime regions +
            two auto-selected instants (normal / atypical)
  top-left: window around the NORMAL instant — streamflow + attention (dual Y)
  top-right: window around the ATYPICAL instant — streamflow + attention (dual Y)

Run in .venv313:
    python scripts/06_eval/98_regime_change_tft_fig5.py
"""

import numpy as _np
import torch as _torch

def ig_per_lag(model, x, steps=24):
    """Integrated-gradient attribution of P50 to each context position (len ENC)."""
    baseline = _torch.zeros_like(x); total = _torch.zeros_like(x)
    with _torch.backends.cudnn.flags(enabled=False):
        for a in _np.linspace(0, 1, steps):
            xi = (baseline + a*(x-baseline)).clone().requires_grad_(True)
            out = model(xi)[0, 1]
            total += _torch.autograd.grad(out, xi)[0]
    return ((total/steps)*(x-baseline))[0].abs().sum(-1).detach().cpu().numpy()

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fig5")
Q90 = 40.89


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

    # All test sequences (continuity for the global panel)
    Xseq, y, w, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
    gidx = np.array(idx)
    tgt_dates = pd.DatetimeIndex([data["dates"][g] for g in gidx])
    qser = (data["df"]["q_mm"] / S.Q_CONV)     # m3/s series (context)
    q_tgt = qser.reindex(tgt_dates).values

    # ── Extract per-day attention (last query) + latent embedding ─────────────
    attn, emb = [], []
    with torch.no_grad():
        for i in range(len(Xseq)):
            x = Xseq[i:i+1].to(DEV)
            h = model.proj(model.vsn(x)) + model.pos
            o,_ = model.lstm(h)
            a, aw = model.attn(o,o,o, need_weights=True, average_attn_weights=True)
            h2 = model.norm(o + model.drop(a))
            emb.append(h2[0,-1,:].cpu().numpy())
            attn.append(aw[0,-1,:].cpu().numpy())
    attn = np.array(attn); emb = np.array(emb)

    # ── Regime metric: internal-state change (z-scored L2 dist from mean emb) ──
    emb_mean = emb.mean(0)
    dev_state = np.linalg.norm(emb - emb_mean, axis=1)
    dev_state = (dev_state - dev_state.mean())/dev_state.std()
    # attention deviation (faint overlay, to show it's flat)
    attn_mean = attn.mean(0)
    dev_attn_raw = np.abs(attn - attn_mean).sum(1)
    if dev_attn_raw.std() < 1e-8:
        dev_attn = np.zeros_like(dev_attn_raw)   # attention identical every day
        log.info("Attention deviation ~ 0 for all days (input-independent attention).")
    else:
        dev_attn = (dev_attn_raw - dev_attn_raw.mean())/dev_attn_raw.std()

    thr = 1.5   # z-score threshold for "atypical regime"
    atypical = dev_state > thr
    log.info(f"Atypical days (state z>{thr}): {int(atypical.sum())} of {len(dev_state)}")
    # correlation of each metric with |dQ/dt| (regime change of the series)
    dq = np.abs(np.gradient(np.nan_to_num(q_tgt)))
    ca = np.corrcoef(dev_attn,dq)[0,1] if dev_attn.std()>1e-8 else 0.0
    log.info(f"corr(state-change, |dQ/dt|)={np.corrcoef(dev_state,dq)[0,1]:.3f}  "
             f"corr(attn-change,|dQ/dt|)={ca:.3f}")

    # ── Auto-select instants: normal (low dev, stable) & atypical (max dev) ────
    i_aty = int(np.argmax(dev_state))
    # normal: lowest dev among days with valid streamflow, away from atypical ones
    order = np.argsort(dev_state)
    i_nor = next(k for k in order if np.isfinite(q_tgt[k]))
    log.info(f"Normal instant: {tgt_dates[i_nor].date()} (Q={q_tgt[i_nor]:.1f}, z={dev_state[i_nor]:.2f})")
    log.info(f"Atypical instant: {tgt_dates[i_aty].date()} (Q={q_tgt[i_aty]:.1f}, z={dev_state[i_aty]:.2f})")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.1], hspace=0.30, wspace=0.24)

    # Per-instant temporal ATTRIBUTION (integrated gradients) — the faithful focus
    # measure, since raw attention is exactly uniform (ratio max/min = 1.0).
    ig_nor = ig_per_lag(model, Xseq[i_nor:i_nor+1].to(DEV))
    ig_aty = ig_per_lag(model, Xseq[i_aty:i_aty+1].to(DEV))

    def window_panel(ax, i0, ig_vec, title):
        g = gidx[i0]
        wdates = pd.DatetimeIndex(data["dates"][g-ENC:g])
        qwin = qser.reindex(wdates).values
        pos = np.arange(ENC)                            # 0 = oldest ... ENC-1 = most recent
        # left axis: streamflow of the window (blue)
        ax.plot(pos, qwin, "-", color="#1f77b4", lw=1.9, label="streamflow")
        ax.set_ylabel("Streamflow [m³/s]", color="#1f77b4"); ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_xlabel("Position in context window (0 = oldest → most recent)")
        # right axis (normalised focus): attribution (green, structured) + attention (gray, flat)
        ax2 = ax.twinx()
        ig_n = ig_vec/ig_vec.max()
        attn_n = attn[i0]/attn[i0].max()                # exactly flat (=1 everywhere)
        ax2.fill_between(pos, 0, ig_n, color="#2ca02c", alpha=0.25)
        ax2.plot(pos, ig_n, "-", color="#2ca02c", lw=1.8, label="attribution (integrated grad.)")
        ax2.plot(pos, attn_n, "--", color="gray", lw=1.2, label="attention (uniform)")
        ax2.set_ylabel("Normalised focus", color="#2ca02c"); ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=10)
        return ax, ax2

    axL = fig.add_subplot(gs[0,0])
    _, axL2 = window_panel(axL, i_nor, ig_nor, f"(a) NORMAL regime — {tgt_dates[i_nor].date()} (Q={q_tgt[i_nor]:.0f} m³/s)")
    axR = fig.add_subplot(gs[0,1])
    _, axR2 = window_panel(axR, i_aty, ig_aty, f"(b) ATYPICAL regime — {tgt_dates[i_aty].date()} (Q={q_tgt[i_aty]:.0f} m³/s)")
    axL2.legend(fontsize=7, loc="upper left")

    # bottom global — paper colour scheme: blue series, red change metric, purple regimes
    axB = fig.add_subplot(gs[1,:])
    axB.plot(tgt_dates, q_tgt, color="#1f77b4", lw=1.1, label="Streamflow (variable)")
    axB.axhline(Q90, color="#1f77b4", ls=":", lw=0.8, alpha=0.6)
    axB.set_ylabel("Streamflow [m³/s]", color="#1f77b4"); axB.tick_params(axis="y", labelcolor="#1f77b4")
    axc = axB.twinx()
    axc.plot(tgt_dates, dev_state, color="#d62728", lw=1.5, label="internal-behaviour change")
    axc.axhline(thr, color="#d62728", ls="--", lw=1)
    axc.set_ylabel("Model internal change (z-score)", color="#d62728"); axc.tick_params(axis="y", labelcolor="#d62728")
    # purple shaded regimes
    inr=False
    for k,d in enumerate(tgt_dates):
        if atypical[k] and not inr: s0=d; inr=True
        if (not atypical[k] or k==len(tgt_dates)-1) and inr:
            axc.axvspan(s0, tgt_dates[k], color="#9467bd", alpha=0.22, zorder=0); inr=False
    # vertical lines for the two selected instants
    for xi, lab, col in [(i_nor,"normal","#1f77b4"), (i_aty,"atypical","#d62728")]:
        axB.axvline(tgt_dates[xi], color=col, lw=2, ls="-")
        axB.annotate(lab, (tgt_dates[xi], axB.get_ylim()[1]*0.92), color=col, fontsize=9,
                     ha="center", fontweight="bold")
    axB.set_title("(c) Internal-behaviour-change timeline — purple = detected atypical regimes; "
                  "vertical lines = selected instants", fontsize=10)
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    from matplotlib.patches import Patch
    h1,l1 = axB.get_legend_handles_labels(); h2,l2 = axc.get_legend_handles_labels()
    axB.legend(h1+h2+[Patch(color="#9467bd", alpha=0.3)], l1+l2+["atypical regime"],
               fontsize=8, loc="upper right", ncol=2)

    fig.suptitle("Internal-behaviour change of the TFT across regimes (adapted from Lim et al. 2021, Fig. 5)\n"
                 "top: attention distribution in a normal vs an atypical window · bottom: state-change timeline",
                 fontweight="bold", fontsize=12)
    fig.savefig(FIGDIR/"fig19_regime_change.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {FIGDIR/'fig19_regime_change.png'}")


if __name__ == "__main__":
    main()
