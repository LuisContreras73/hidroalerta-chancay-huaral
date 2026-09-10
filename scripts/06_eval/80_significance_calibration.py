#!/usr/bin/env python3
"""
Script 80 — Statistical significance + calibration (paper-grade evaluation).

CRITICAL FIX: D7's q_next_1d in 2024-2025 contains 307 GR4J-filled days
(only 423 of 730 are REAL observations). This script re-evaluates HydroST on
REAL-OBS-ONLY days (Regla 5) and provides:

  1. Honest test metrics on real-obs days (423) for HydroST vs TFT-v2
  2. Diebold-Mariano test for forecast-accuracy significance (HydroST vs TFT-v2)
  3. Reliability / calibration diagrams for the P10/P50/P90 quantile bands
  4. Coverage by hydrological season (wet vs dry) — exposes seasonal mis-calibration

Run in .venv313:
    python scripts/06_eval/80_significance_calibration.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/qa_satellite"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sig_cal")

# Import HydroST building blocks
spec = importlib.util.spec_from_file_location("hydrost", ROOT / "scripts/05_models/78_hydrost_Q.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
DEVICE = H.DEVICE
Q_CONV = H.Q_CONV


# ── Real observation mask ─────────────────────────────────────────────────────

def load_real_obs() -> pd.Series:
    """Real observed Q at outlet 47E214D2 (m3/s), indexed by date. NaN = no obs."""
    obs = pd.read_csv(ROOT / "data/silver/snirh/S1_snirh_daily_q.csv", parse_dates=["date"])
    return obs.set_index("date")["q_santo_domingo_47e214d2"]


def predict_hydrost(model, dyn, static, target, dates, target_dates):
    """Return DataFrame indexed by target date with obs/p10/p50/p90 (m3/s)."""
    ds = H.HydroDataset(dyn, static, target, dates, target_dates)
    # Recover the date for each retained sequence
    idx_to_date = {i: pd.Timestamp(d) for i, d in enumerate(dates)}
    seq_dates = [idx_to_date[idx] for (idx, _) in ds.indices]

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    p10, p50, p90, y = [], [], [], []
    with torch.no_grad():
        for x_dyn, x_static, yy in loader:
            pred = model(x_dyn.to(DEVICE), x_static.to(DEVICE)).cpu().numpy()
            p10.extend(pred[:, 0]); p50.extend(pred[:, 1]); p90.extend(pred[:, 2])
            y.extend(yy.numpy())

    df = pd.DataFrame({
        "date": seq_dates,
        "q_obs_d7": np.array(y) / Q_CONV,
        "p10": np.array(p10) / Q_CONV,
        "p50": np.array(p50) / Q_CONV,
        "p90": np.array(p90) / Q_CONV,
    }).set_index("date")
    return df


def dm_test(obs, pred1, pred2, h=1, power=2):
    """
    Diebold-Mariano test. H0: equal predictive accuracy.
    pred1 = candidate (HydroST), pred2 = benchmark (TFT-v2).
    Returns (DM stat, p-value). Negative DM => pred1 more accurate.
    """
    e1 = np.abs(obs - pred1) ** power
    e2 = np.abs(obs - pred2) ** power
    d  = e1 - e2
    n  = len(d)
    d_mean = d.mean()

    # Newey-West long-run variance (lag h-1)
    gamma0 = np.sum((d - d_mean) ** 2) / n
    var = gamma0
    for lag in range(1, h):
        cov = np.sum((d[lag:] - d_mean) * (d[:-lag] - d_mean)) / n
        var += 2 * (1 - lag / h) * cov
    dm = d_mean / np.sqrt(var / n)

    # Harvey-Leybourne-Newbold small-sample correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corr = dm * hln
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_corr), df=n - 1))
    return dm_corr, p_value


def main():
    # ── Load data + model ─────────────────────────────────────────────────────
    df = H.load_data()
    dyn, static, target, dates, _ = H.build_arrays(df)
    dates_pd = pd.DatetimeIndex(dates)

    model = H.HydroST().to(DEVICE)
    ckpt = OUT_DIR / "78_hydrost_final.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    log.info(f"Loaded {ckpt.name}")

    # ── Predict on test 2024-2025 (all 730 calendar days) ────────────────────
    test_dates = dates_pd[dates_pd >= H.TEST_START]
    pred_df = predict_hydrost(model, dyn, static, target, dates, test_dates)

    # ── Real-obs mask: q_next_1d at date t = obs at t+1 ───────────────────────
    obs_real = load_real_obs()
    obs_next = obs_real.shift(-1)  # align to q_next_1d convention
    pred_df["q_obs_real"] = obs_next.reindex(pred_df.index)

    n_total = len(pred_df)
    real_mask = pred_df["q_obs_real"].notna()
    n_real = int(real_mask.sum())
    log.info(f"Test days: {n_total} calendar, {n_real} with REAL obs ({n_real/n_total*100:.0f}%)")

    clean = pred_df[real_mask].copy()

    # Sanity: D7 target should equal real obs on real-obs days
    diff = (clean["q_obs_d7"] - clean["q_obs_real"]).abs().max()
    log.info(f"Max |D7 target - real obs| on real days: {diff:.2e} (should be ~0)")

    # ── Honest HydroST metrics (real obs only) ───────────────────────────────
    m_clean = H.compute_metrics(clean["q_obs_real"].values, clean["p50"].values,
                                clean["p10"].values, clean["p90"].values)
    m_contaminated = H.compute_metrics(pred_df["q_obs_d7"].values, pred_df["p50"].values,
                                       pred_df["p10"].values, pred_df["p90"].values)

    log.info("\n" + "="*60)
    log.info("HydroST TEST metrics: contaminated (730d, GR4J-filled) vs honest (423d real)")
    log.info("="*60)
    for k in ["NSE", "NSE_sqrt", "J_alert", "CSI", "POD", "FAR", "PICP", "PINAW"]:
        log.info(f"  {k:10s}  contaminated={m_contaminated[k]:.4f}   honest={m_clean[k]:.4f}")

    clean.to_csv(OUT_DIR / "80_hydrost_test_realobs.csv")

    # ── Align with TFT-v2 for DM test ─────────────────────────────────────────
    tft = pd.read_csv(OUT_DIR / "tft_v2_predictions.csv", parse_dates=["date"])
    tft = tft[tft["target"] == "q_next_1d"].set_index("date")
    common = clean.index.intersection(tft.index)
    log.info(f"\nCommon real-obs days HydroST ∩ TFT-v2: {len(common)}")

    obs_c   = clean.loc[common, "q_obs_real"].values
    hydro_c = clean.loc[common, "p50"].values
    tft_c   = tft.loc[common, "q_p50"].values

    m_hydro = H.compute_metrics(obs_c, hydro_c)
    m_tft   = H.compute_metrics(obs_c, tft_c)

    log.info("\n" + "="*60)
    log.info(f"HEAD-TO-HEAD on {len(common)} common real-obs days")
    log.info("="*60)
    log.info(f"  {'metric':10s}  {'HydroST':>10s}  {'TFT-v2':>10s}")
    for k in ["NSE", "NSE_sqrt", "J_alert", "CSI", "POD", "FAR"]:
        log.info(f"  {k:10s}  {m_hydro[k]:>10.4f}  {m_tft[k]:>10.4f}")

    # DM test (MAE-based, h=1)
    dm_mae, p_mae = dm_test(obs_c, hydro_c, tft_c, h=1, power=1)
    dm_mse, p_mse = dm_test(obs_c, hydro_c, tft_c, h=1, power=2)
    log.info("\nDiebold-Mariano (HydroST vs TFT-v2, H0: equal accuracy):")
    log.info(f"  MAE-based: DM={dm_mae:+.3f}  p={p_mae:.4f}  "
             f"{'HydroST better' if dm_mae<0 and p_mae<0.05 else 'TFT better' if dm_mae>0 and p_mae<0.05 else 'no sig. diff'}")
    log.info(f"  MSE-based: DM={dm_mse:+.3f}  p={p_mse:.4f}  "
             f"{'HydroST better' if dm_mse<0 and p_mse<0.05 else 'TFT better' if dm_mse>0 and p_mse<0.05 else 'no sig. diff'}")

    # ── Calibration / reliability diagram ─────────────────────────────────────
    plot_calibration(clean)

    # ── Seasonal coverage ─────────────────────────────────────────────────────
    seasonal_coverage(clean)

    log.info("\nDone. Outputs: 80_hydrost_test_realobs.csv, calibration + coverage figures")


def plot_calibration(clean: pd.DataFrame):
    """Reliability diagram: nominal vs empirical coverage across quantile levels.
    We only have P10/P50/P90 trained; approximate intermediate coverage via the
    central interval [P10,P90]=80% nominal, and P50 median exceedance."""
    obs = clean["q_obs_real"].values
    p10, p50, p90 = clean["p10"].values, clean["p50"].values, clean["p90"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: PIT-style coverage at known nominal levels
    nominal = {"P10 (10%)": (obs <= p10).mean(),
               "P50 (50%)": (obs <= p50).mean(),
               "P90 (90%)": (obs <= p90).mean()}
    ax = axes[0]
    levels = [0.10, 0.50, 0.90]
    emp    = list(nominal.values())
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(levels, emp, "o-", color="#1f77b4", ms=10, lw=2, label="HydroST")
    for lv, e, name in zip(levels, emp, nominal):
        ax.annotate(f"{e:.2f}", (lv, e), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel("Nominal cumulative probability")
    ax.set_ylabel("Empirical P(obs ≤ quantile)")
    ax.set_title("Reliability diagram (quantile calibration)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 2: interval coverage 80% nominal — by Q magnitude bin
    ax = axes[1]
    inside = (obs >= p10) & (obs <= p90)
    bins = pd.qcut(obs, q=5, duplicates="drop")
    cov_by_bin = pd.Series(inside, index=clean.index).groupby(bins.codes).mean()
    bin_centers = pd.Series(obs).groupby(bins.codes).median()
    ax.bar(range(len(cov_by_bin)), cov_by_bin.values, color="#2ca02c", alpha=0.8)
    ax.axhline(0.80, color="red", ls="--", lw=1.5, label="nominal 80%")
    ax.set_xticks(range(len(cov_by_bin)))
    ax.set_xticklabels([f"{c:.0f}" for c in bin_centers.values], fontsize=9)
    ax.set_xlabel("Q magnitude bin (median m³/s)")
    ax.set_ylabel("Empirical coverage of [P10,P90]")
    ax.set_title(f"80% interval coverage by flow magnitude\n(overall PICP={inside.mean():.2f})")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_calibration.png", dpi=150)
    plt.close(fig)
    log.info(f"Calibration figure saved: 03_calibration.png")


def seasonal_coverage(clean: pd.DataFrame):
    """Coverage and error by wet (Dec-Apr) vs dry (May-Nov) season."""
    obs = clean["q_obs_real"]
    p10, p50, p90 = clean["p10"], clean["p50"], clean["p90"]
    month = clean.index.month
    wet = np.isin(month, [12, 1, 2, 3, 4])

    log.info("\n" + "="*60)
    log.info("SEASONAL DIAGNOSTICS (wet=Dec-Apr, dry=May-Nov)")
    log.info("="*60)
    for name, mask in [("WET", wet), ("DRY", ~wet)]:
        if mask.sum() == 0:
            continue
        o, m, lo, hi = obs[mask], p50[mask], p10[mask], p90[mask]
        picp = ((o >= lo) & (o <= hi)).mean()
        nse_denom = ((o - o.mean())**2).sum()
        nse = 1 - ((o - m)**2).sum() / nse_denom if nse_denom > 0 else np.nan
        mae = (o - m).abs().mean()
        log.info(f"  {name}  n={mask.sum():3d}  PICP={picp:.3f}  NSE={nse:.3f}  "
                 f"MAE={mae:.2f} m³/s  meanQ={o.mean():.1f}")


if __name__ == "__main__":
    main()
