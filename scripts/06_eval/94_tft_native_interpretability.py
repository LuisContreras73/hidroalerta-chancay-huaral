#!/usr/bin/env python3
"""
Script 94 — TFT NATIVE interpretability (clean paper figures).

Rationale: SHAP/LIME are ill-suited to a TFT (they assume tabular independent
features; perturbing a multivariate sequence breaks temporal dependence and
produces unrealistic inputs). The correct tools for a TFT are its BUILT-IN
mechanisms — the Variable Selection Network (VSN) and the temporal attention —
plus gradient attribution (Script 83). This script produces two clean, dedicated
figures from a single trained (Optuna-best) TFT:

  fig17 · VSN interpretability:  (a) global variable selection importance,
                                 (b) regime shift (flood vs baseflow).
  fig18 · Temporal attention:    (a) mean attention over lags by regime,
                                 (b) attention matrix (query x key day).

Run in .venv313:
    python scripts/06_eval/94_tft_native_interpretability.py
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
log = logging.getLogger("tft_native")
Q90 = 40.89

# Readable feature names (real sub-basin names): Alto Chancay=649, Carac=655, Baños=656
def _f(var, loc):
    names = {"basin":"basin", "649":"Alto Chancay", "655":"Carac", "656":"Baños"}
    return f"{var} ({names[loc]})"
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


def main():
    spec = importlib.util.spec_from_file_location("tune81", ROOT/"scripts/05_models/81_tune_tft_optuna.py")
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    T = S.T
    data = S.prepare_data()
    bp = pd.read_csv(OUTML/"81_best_params.csv", index_col=0)["0"].to_dict()
    cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]),
               DROP=float(bp["DROP"]), LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]),
               WD=float(bp["WD"]), ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
    log.info("Training Optuna-best TFT (single run for interpretability)...")
    _, model = S.run_config(cfg, data, seed=42); model.eval()
    ENC = T.ENC

    # Test sequences + regime
    Xseq, y, w, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
    obs = data["obs_real_mm"][idx]; real = np.isfinite(obs)
    Xseq = Xseq[real]; obs_m3 = obs[real]/S.Q_CONV
    flood = obs_m3 > Q90

    # ── Extract VSN weights + temporal attention ──────────────────────────────
    vsn_all, attn_last, attn_full = [], [], []
    with torch.no_grad():
        for i in range(len(Xseq)):
            x = Xseq[i:i+1].to(S.DEVICE)
            vsn = torch.softmax(model.vsn.w(x), dim=-1)[0].mean(0).cpu().numpy()  # mean over lags
            vsn_all.append(vsn)
            h = model.proj(model.vsn(x)) + model.pos
            o,_ = model.lstm(h)
            _, aw = model.attn(o,o,o, need_weights=True, average_attn_weights=True)
            attn_last.append(aw[0,-1,:].cpu().numpy())
            attn_full.append(aw[0].cpu().numpy())
    vsn_all = np.array(vsn_all); attn_last = np.array(attn_last)
    imp = pd.Series(vsn_all.mean(0), index=FEAT_NAMES).sort_values(ascending=False)
    imp_f = pd.Series(vsn_all[flood].mean(0), index=FEAT_NAMES)
    imp_b = pd.Series(vsn_all[~flood].mean(0), index=FEAT_NAMES)
    delta = (imp_f - imp_b).sort_values(ascending=False)

    # ══ FIG 17 · VSN interpretability ═════════════════════════════════════════
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    top = imp.head(12)[::-1]
    ax[0].barh(range(len(top)), top.values, color="#08519c", alpha=0.85)
    ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.index, fontsize=9)
    ax[0].axvline(1/len(FEAT_NAMES), color="gray", ls="--", lw=1, label="uniform (1/41)")
    ax[0].set_xlabel("Variable selection weight (VSN)")
    ax[0].set_title("(a) Global variable importance (VSN)\nwhich inputs the TFT selects overall", fontsize=10)
    ax[0].legend(fontsize=8)

    mv = pd.concat([delta.head(6), delta.tail(3)])
    cols = ["#cc2222" if v>0 else "#4a90d9" for v in mv.values]
    ax[1].barh(range(len(mv)), mv.values, color=cols, alpha=0.85)
    ax[1].set_yticks(range(len(mv))); ax[1].set_yticklabels(mv.index, fontsize=9)
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_xlabel("Δ VSN weight (flood − baseflow)")
    ax[1].set_title("(b) Regime shift of variable selection\nred = emphasised MORE during floods", fontsize=10)
    fig.suptitle("TFT Variable Selection Network — native interpretability",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGDIR/"fig17_tft_vsn.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    log.info(f"Saved fig17_tft_vsn.png. Top VSN-flood movers: {list(delta.head(4).index)}")

    # ══ FIG 18 · Temporal attention ═══════════════════════════════════════════
    mf = attn_last[flood].mean(0)[::-1]; mb = attn_last[~flood].mean(0)[::-1]  # idx0 = most recent
    lags = np.arange(ENC)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(lags, mb, "-o", ms=3, color="#4a90d9", label=f"baseflow (n={int((~flood).sum())})")
    ax[0].plot(lags, mf, "-o", ms=3, color="#cc2222", label=f"flood (n={int(flood.sum())})")
    ax[0].axhline(1/ENC, color="gray", ls="--", lw=1, label="uniform")
    ax[0].set_xlabel("Lag before forecast (days; 0 = most recent)")
    ax[0].set_ylabel("Attention weight (final query)")
    ax[0].set_title("(a) Temporal attention by regime\n(near-uniform; see IG attribution in Fig. 7)", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_xlim(0, min(30,ENC))

    M = np.mean([attn_full[i] for i in range(len(attn_full)) if real[i] and flood[np.sum(real[:i+1])-1]][:60], axis=0) \
        if flood.any() else np.mean(attn_full[:60], axis=0)
    # simpler robust mean over all test matrices
    M = np.mean(attn_full[:min(len(attn_full),200)], axis=0)
    im = ax[1].imshow(M, cmap="magma", aspect="auto", origin="lower")
    ax[1].set_xlabel("Key day (attended-to)"); ax[1].set_ylabel("Query day")
    ax[1].set_title("(b) Attention matrix (mean)\nrow = query day, column = attended day", fontsize=10)
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="attention weight")
    fig.suptitle("TFT temporal attention mechanism — native interpretability",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGDIR/"fig18_tft_attention.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    log.info(f"Saved fig18_tft_attention.png")


if __name__ == "__main__":
    main()
