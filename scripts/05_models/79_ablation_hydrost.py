#!/usr/bin/env python3
"""
Script 79 — HydroST ablation study.

Quantifies the contribution of each design component by removing one at a time
and measuring the impact on val (2023) and test (2024-2025) performance.

Variants:
  full          : complete HydroST (reference)
  no_pretrain   : random init, skip GR4J Phase-1 pretraining
  no_spatial    : remove cross-basin spatial multi-head attention
  no_satellite  : drop MODIS/Landsat satellite features (NDVI/snow/MNDWI)
  no_weight     : ALERT_W = 1.0 (no upweighting of Q>Q90 events)
  no_mixed_loss : ALPHA = 1.0 (pure pinball, no MSE_P50 term)

Each variant runs 3 seeds; we report mean +/- std to separate true component
effects from training noise (paper-grade reporting).

Run in .venv313:
    python scripts/05_models/79_ablation_hydrost.py [--seeds 3] [--quick]
"""

import argparse
import importlib.util
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import building blocks from Script 78 ──────────────────────────────────────
spec = importlib.util.spec_from_file_location("hydrost", ROOT / "scripts/05_models/78_hydrost_Q.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

log = logging.getLogger("ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEVICE = H.DEVICE

# Satellite columns to drop in the no_satellite variant
SAT_COLS = ["ndvi_fill", "snow_fill", "lsat_ndvi_fill", "lsat_mndwi_fill"]

# Default config snapshot (restore between variants)
DEFAULTS = dict(
    DYN_COLS=list(H.DYN_COLS),
    N_DYN=H.N_DYN,
    ALERT_W=H.ALERT_W,
    ALPHA=H.ALPHA,
)

VARIANTS = ["full", "no_pretrain", "no_spatial", "no_satellite", "no_weight", "no_mixed_loss"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def restore_defaults():
    H.DYN_COLS = list(DEFAULTS["DYN_COLS"])
    H.N_DYN    = DEFAULTS["N_DYN"]
    H.ALERT_W  = DEFAULTS["ALERT_W"]
    H.ALPHA    = DEFAULTS["ALPHA"]


def eval_on(model, loader):
    """Return metrics dict for a model on a loader (m3/s space)."""
    model.eval()
    p10, p50, p90, y_true = [], [], [], []
    with torch.no_grad():
        for x_dyn, x_static, y in loader:
            pred = model(x_dyn.to(DEVICE), x_static.to(DEVICE)).cpu().numpy()
            p10.extend(pred[:, 0]); p50.extend(pred[:, 1]); p90.extend(pred[:, 2])
            y_true.extend(y.numpy())
    obs = np.array(y_true) / H.Q_CONV
    return H.compute_metrics(obs, np.array(p50)/H.Q_CONV,
                             np.array(p10)/H.Q_CONV, np.array(p90)/H.Q_CONV)


def run_variant(variant: str, seed: int, df, quick: bool):
    """Train one variant with one seed; return (val_metrics, test_metrics)."""
    restore_defaults()

    # ── Apply ablation config ────────────────────────────────────────────────
    use_spatial   = True
    skip_pretrain = False

    if variant == "no_satellite":
        H.DYN_COLS = [c for c in DEFAULTS["DYN_COLS"] if c not in SAT_COLS]
        H.N_DYN    = len(H.DYN_COLS)
    elif variant == "no_spatial":
        use_spatial = False
    elif variant == "no_pretrain":
        skip_pretrain = True
    elif variant == "no_weight":
        H.ALERT_W = 1.0
    elif variant == "no_mixed_loss":
        H.ALPHA = 1.0

    seed_everything(seed)

    # ── Build arrays with (possibly reduced) dyn cols ────────────────────────
    dyn, static, target, dates, _ = H.build_arrays(df)

    dates_pd = pd.DatetimeIndex(dates)
    pre_dates  = dates_pd[dates_pd <= H.PRETRAIN_END]
    ft_dates   = dates_pd[(dates_pd >= H.FT_START) & (dates_pd <= H.FT_END)]
    val_dates  = dates_pd[(dates_pd >= H.VAL_START) & (dates_pd <= H.VAL_END)]
    test_dates = dates_pd[dates_pd >= H.TEST_START]

    ds_pre  = H.HydroDataset(dyn, static, target, dates, pre_dates)
    ds_ft   = H.HydroDataset(dyn, static, target, dates, ft_dates)
    ds_val  = H.HydroDataset(dyn, static, target, dates, val_dates)
    ds_test = H.HydroDataset(dyn, static, target, dates, test_dates)

    ld_pre  = DataLoader(ds_pre,  batch_size=H.BATCH1, shuffle=True,  num_workers=0)
    ld_ft   = DataLoader(ds_ft,   batch_size=H.BATCH2, shuffle=True,  num_workers=0)
    ld_val  = DataLoader(ds_val,  batch_size=H.BATCH2, shuffle=False, num_workers=0)
    ld_test = DataLoader(ds_test, batch_size=H.BATCH2, shuffle=False, num_workers=0)

    # ── Build model ──────────────────────────────────────────────────────────
    model = H.HydroST(n_dyn=H.N_DYN, use_spatial=use_spatial).to(DEVICE)

    ep1 = 20 if quick else H.EP1
    ep2 = 60 if quick else H.EP2

    # ── Phase 1 (optional) ───────────────────────────────────────────────────
    if not skip_pretrain:
        model = H.train_phase1(model, ld_pre, ep1)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    model = H.train_phase2(model, ld_ft, ld_val, ep2)

    val_m  = eval_on(model, ld_val)
    test_m = eval_on(model, ld_test)
    return val_m, test_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="fewer epochs for a fast smoke test")
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}  |  seeds={args.seeds}  quick={args.quick}")

    # Load + satellite-fill ONCE (shared across variants)
    df = H.load_data()

    rows = []
    for variant in VARIANTS:
        log.info(f"\n{'#'*60}\n# VARIANT: {variant}\n{'#'*60}")
        for seed in range(args.seeds):
            log.info(f"\n--- {variant} | seed {seed} ---")
            val_m, test_m = run_variant(variant, seed, df, args.quick)
            rows.append(dict(variant=variant, seed=seed, split="val",
                             **{k: v for k, v in val_m.items() if not isinstance(v, np.ndarray)}))
            rows.append(dict(variant=variant, seed=seed, split="test",
                             **{k: v for k, v in test_m.items() if not isinstance(v, np.ndarray)}))
            log.info(f"  val  J_alert={val_m['J_alert']:.4f} NSE={val_m['NSE']:.4f} "
                     f"CSI={val_m['CSI']:.4f} POD={val_m['POD']:.4f}")
            log.info(f"  test J_alert={test_m['J_alert']:.4f} NSE={test_m['NSE']:.4f} "
                     f"CSI={test_m['CSI']:.4f} POD={test_m['POD']:.4f}")

    restore_defaults()

    # ── Save raw results ──────────────────────────────────────────────────────
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "79_ablation_raw.csv", index=False)

    # ── Aggregate: mean +/- std per variant per split ────────────────────────
    metrics_cols = ["NSE", "NSE_sqrt", "J_alert", "CSI", "POD", "FAR", "PICP", "PINAW"]
    agg = raw.groupby(["variant", "split"])[metrics_cols].agg(["mean", "std"])
    agg.to_csv(OUT_DIR / "79_ablation_summary.csv")

    # ── Delta vs full (test split) ────────────────────────────────────────────
    test_means = raw[raw["split"] == "test"].groupby("variant")[metrics_cols].mean()
    full_row = test_means.loc["full"]
    delta = test_means - full_row
    delta = delta.drop("full")

    log.info("\n" + "="*70)
    log.info("ABLATION SUMMARY — TEST 2024-2025 (mean over seeds)")
    log.info("="*70)
    log.info(f"\n{'variant':<16} {'J_alert':>8} {'dJ':>7} {'NSE':>7} {'CSI':>7} {'POD':>7}")
    log.info(f"{'full (ref)':<16} {full_row['J_alert']:>8.4f} {'--':>7} "
             f"{full_row['NSE']:>7.4f} {full_row['CSI']:>7.4f} {full_row['POD']:>7.4f}")
    for v in delta.index:
        r = test_means.loc[v]
        log.info(f"{v:<16} {r['J_alert']:>8.4f} {delta.loc[v,'J_alert']:>+7.4f} "
                 f"{r['NSE']:>7.4f} {r['CSI']:>7.4f} {r['POD']:>7.4f}")
    log.info("="*70)
    log.info("dJ = change in J_alert when component is removed (negative = component helps)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        variants_sorted = delta["J_alert"].sort_values().index.tolist()
        vals = delta.loc[variants_sorted, "J_alert"].values
        colors = ["#cc2222" if v < 0 else "#2ca02c" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([v.replace("no_", "− ") for v in variants_sorted])
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Δ J_alert vs full model (test 2024-2025)")
        ax.set_title("HydroST ablation: component contribution to J_alert\n"
                     "(negative bar = removing it hurts → component is important)")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(OUT_DIR.parent / "qa_satellite" / "02_ablation_J_alert.png", dpi=150)
        # also drop a copy near the model outputs
        fig.savefig(OUT_DIR / "79_ablation_J_alert.png", dpi=150)
        plt.close(fig)
        log.info(f"Plot saved: 79_ablation_J_alert.png")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

    log.info(f"\nResults saved: 79_ablation_raw.csv, 79_ablation_summary.csv")


if __name__ == "__main__":
    main()
