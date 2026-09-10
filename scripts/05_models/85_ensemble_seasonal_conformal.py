#!/usr/bin/env python3
"""
Script 85 — Final ensemble + seasonal conformal calibration.

Combines the two best models (TFT-tuned + HydroST) and fixes the documented
dry-season miscalibration (§9 of HALLAZGOS_PAPER) with SEASON-SPECIFIC conformal
prediction intervals.

Pipeline (all honest — real obs only, test sealed until final eval):
  1. Date-aligned P10/P50/P90 from TFT-tuned and HydroST on VAL 2023 + TEST 2024-25.
  2. Ensemble P50 = NSE-weighted average (weights fit on VAL).
  3. Seasonal split-conformal: nonconformity scores s = max(P10-obs, obs-P90)
     computed PER SEASON (wet Dec-Apr / dry May-Nov) on VAL → per-season radius
     q at level (1-alpha)(1+1/n). Apply to TEST bands by season.
  4. Evaluate coverage before/after, overall and per season.

Run in .venv313:
    python scripts/05_models/85_ensemble_seasonal_conformal.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/qa_satellite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ensemble")

# Import HydroST (H) and the TFT tuner (S, which holds TFT module T)
specH = importlib.util.spec_from_file_location("hydrost", ROOT / "scripts/05_models/78_hydrost_Q.py")
H = importlib.util.module_from_spec(specH); specH.loader.exec_module(H)
specS = importlib.util.spec_from_file_location("tune81", ROOT / "scripts/05_models/81_tune_tft_optuna.py")
S = importlib.util.module_from_spec(specS); specS.loader.exec_module(S)
T = S.T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_CONV = H.Q_CONV
Q90    = 40.89   # m3/s

ALPHA = 0.20     # 1 - nominal coverage of the [P10,P90] band (80%)
WET_MONTHS = [12, 1, 2, 3, 4]


def season_of(idx) -> np.ndarray:
    return np.where(np.isin(pd.DatetimeIndex(idx).month, WET_MONTHS), "wet", "dry")


# ── HydroST date-indexed predictions ──────────────────────────────────────────

def hydrost_predictions():
    H.DEVICE = DEVICE
    df = H.load_data()
    dyn, static, target, dates, _ = H.build_arrays(df)
    model = H.HydroST().to(DEVICE)
    model.load_state_dict(torch.load(OUT_DIR / "78_hydrost_final.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    dates_pd = pd.DatetimeIndex(dates)

    out = {}
    for name, mask in [("val", (dates_pd >= H.VAL_START) & (dates_pd <= H.VAL_END)),
                       ("test", dates_pd >= H.TEST_START)]:
        ds = H.HydroDataset(dyn, static, target, dates, dates_pd[mask])
        seq_dates = [pd.Timestamp(dates[idx]) for (idx, _) in ds.indices]
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        p10, p50, p90 = [], [], []
        with torch.no_grad():
            for x_dyn, x_static, _ in loader:
                pr = model(x_dyn.to(DEVICE), x_static.to(DEVICE)).cpu().numpy()
                p10.extend(pr[:, 0]); p50.extend(pr[:, 1]); p90.extend(pr[:, 2])
        out[name] = pd.DataFrame({"date": seq_dates,
                                  "h_p10": np.array(p10)/Q_CONV,
                                  "h_p50": np.array(p50)/Q_CONV,
                                  "h_p90": np.array(p90)/Q_CONV}).set_index("date")
    return out


# ── TFT-tuned date-indexed predictions ────────────────────────────────────────

def tft_predictions():
    data = S.prepare_data()
    bp = pd.read_csv(OUT_DIR / "81_best_params.csv", index_col=0)["0"].to_dict()
    cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]),
               DROP=float(bp["DROP"]), LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]),
               WD=float(bp["WD"]), ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
    log.info("Training Optuna-best TFT for ensemble...")
    _, model = S.run_config(cfg, data, seed=42)
    model.eval()

    out = {}
    for name, mask in [("val", data["m_val"]), ("test", data["m_te"])]:
        Xseq, yseq, wseq, idx = T.make_seqs(data["Xs"], data["ysc"], mask)
        seq_dates = pd.DatetimeIndex(data["dates"][idx])
        obs_mm = data["obs_real_mm"][idx]
        with torch.no_grad():
            pr = model(Xseq.to(DEVICE)).cpu().numpy()
        df = pd.DataFrame({"date": seq_dates,
                           "t_p10": data["inv"](pr[:, 0])/Q_CONV,
                           "t_p50": data["inv"](pr[:, 1])/Q_CONV,
                           "t_p90": data["inv"](pr[:, 2])/Q_CONV,
                           "obs":   obs_mm/Q_CONV}).set_index("date")
        out[name] = df[np.isfinite(df["obs"])]
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(obs, p50):
    obs, p50 = np.asarray(obs), np.asarray(p50)
    nse = 1 - np.sum((obs-p50)**2)/np.sum((obs-obs.mean())**2)
    s_o, s_p = np.sqrt(np.maximum(obs,0)), np.sqrt(np.maximum(p50,0))
    nse_sqrt = 1 - np.sum((s_o-s_p)**2)/np.sum((s_o-s_o.mean())**2)
    oa, pa = obs >= Q90, p50 >= Q90
    tp, fp, fn = np.sum(oa&pa), np.sum(~oa&pa), np.sum(oa&~pa)
    csi = tp/(tp+fp+fn) if (tp+fp+fn) else 0
    pod = tp/(tp+fn) if (tp+fn) else 0
    far = fp/(tp+fp) if (tp+fp) else 0
    j = 0.25*nse_sqrt + 0.25*nse + 0.30*csi + 0.10*pod - 0.10*far
    return dict(NSE=nse, NSE_sqrt=nse_sqrt, J_alert=j, CSI=csi, POD=pod, FAR=far)


def picp(obs, lo, hi):
    return float(((obs >= lo) & (obs <= hi)).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hy = hydrost_predictions()
    tf = tft_predictions()

    # Merge on common real-obs dates per split
    merged = {}
    for split in ["val", "test"]:
        m = tf[split].join(hy[split], how="inner")
        merged[split] = m.dropna(subset=["obs", "t_p50", "h_p50"])
        log.info(f"{split}: {len(merged[split])} common real-obs days")

    val, test = merged["val"], merged["test"]

    # ── Ensemble weight from VAL NSE ─────────────────────────────────────────
    nse_t = metrics(val["obs"], val["t_p50"])["NSE"]
    nse_h = metrics(val["obs"], val["h_p50"])["NSE"]
    wt = max(nse_t, 0) / (max(nse_t, 0) + max(nse_h, 0))
    log.info(f"VAL NSE: TFT={nse_t:.4f} HydroST={nse_h:.4f} -> w_tft={wt:.3f}")

    for d in (val, test):
        d["e_p50"] = wt*d["t_p50"] + (1-wt)*d["h_p50"]
        d["e_p10"] = wt*d["t_p10"] + (1-wt)*d["h_p10"]
        d["e_p90"] = wt*d["t_p90"] + (1-wt)*d["h_p90"]

    # ── Model comparison on TEST (honest) ────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("TEST 2024-2025 (honest) — P50 skill")
    log.info("="*60)
    for name, col in [("TFT-tuned","t_p50"), ("HydroST","h_p50"), ("Ensemble","e_p50")]:
        m = metrics(test["obs"], test[col])
        log.info(f"  {name:10s} NSE={m['NSE']:.4f} J_alert={m['J_alert']:.4f} "
                 f"CSI={m['CSI']:.4f} POD={m['POD']:.4f} FAR={m['FAR']:.4f}")

    # ── Seasonal conformal calibration (fit on VAL) ──────────────────────────
    val_season = season_of(val.index)
    q_season = {}
    for s in ["wet", "dry"]:
        sub = val[val_season == s]
        if len(sub) < 5:
            q_season[s] = 0.0; continue
        scores = np.maximum(sub["e_p10"] - sub["obs"], sub["obs"] - sub["e_p90"]).values
        n = len(scores)
        level = min(1.0, (1-ALPHA)*(1 + 1/n))
        q_season[s] = float(np.quantile(scores, level))
    log.info(f"\nSeasonal conformal radius (m3/s): wet={q_season['wet']:.2f}  dry={q_season['dry']:.2f}")

    # Also a single global radius for comparison
    scores_all = np.maximum(val["e_p10"] - val["obs"], val["obs"] - val["e_p90"]).values
    n = len(scores_all)
    q_global = float(np.quantile(scores_all, min(1.0,(1-ALPHA)*(1+1/n))))
    log.info(f"Global conformal radius (m3/s): {q_global:.2f}")

    # ── Apply to TEST ────────────────────────────────────────────────────────
    test_season = season_of(test.index)
    test["lo_raw"], test["hi_raw"] = test["e_p10"], test["e_p90"]
    test["lo_glob"] = test["e_p10"] - q_global
    test["hi_glob"] = test["e_p90"] + q_global
    test["lo_seas"] = test["e_p10"] - np.array([q_season[s] for s in test_season])
    test["hi_seas"] = test["e_p90"] + np.array([q_season[s] for s in test_season])

    # ── Coverage comparison ──────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info(f"PICP (target {1-ALPHA:.0%}) — raw vs global-conformal vs seasonal-conformal")
    log.info("="*60)
    def width(lo, hi): return float((hi-lo).mean())
    for label, mask in [("ALL", np.ones(len(test), bool)),
                        ("WET", test_season=="wet"), ("DRY", test_season=="dry")]:
        t = test[mask]
        log.info(f"  {label:4s} (n={mask.sum():3d})  "
                 f"raw PICP={picp(t['obs'],t['lo_raw'],t['hi_raw']):.3f} (w={width(t['lo_raw'],t['hi_raw']):.1f})  "
                 f"global={picp(t['obs'],t['lo_glob'],t['hi_glob']):.3f} (w={width(t['lo_glob'],t['hi_glob']):.1f})  "
                 f"seasonal={picp(t['obs'],t['lo_seas'],t['hi_seas']):.3f} (w={width(t['lo_seas'],t['hi_seas']):.1f})")

    # ── Save predictions ─────────────────────────────────────────────────────
    test.to_csv(OUT_DIR / "85_ensemble_test_calibrated.csv")
    log.info("\nSaved: 85_ensemble_test_calibrated.csv")

    # ── Figure: hydrograph with seasonal-calibrated bands ────────────────────
    t = test.sort_index()
    ts = season_of(t.index)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(t.index, t["lo_seas"], t["hi_seas"], color="#9ecae1", alpha=0.5,
                    label="80% seasonal-conformal band")
    ax.plot(t.index, t["e_p50"], color="#08519c", lw=1.2, label="Ensemble P50")
    ax.plot(t.index, t["obs"], ".", color="black", ms=3, label="Obs (real)")
    ax.axhline(Q90, color="red", ls=":", lw=1, label="Q90 alert")
    # shade wet seasons
    for yr in [2024, 2025]:
        ax.axvspan(pd.Timestamp(f"{yr-1}-12-01"), pd.Timestamp(f"{yr}-04-30"),
                   color="#fff3cd", alpha=0.3)
    ax.set_ylabel("Q (m³/s)")
    ax.set_title("Final ensemble forecast with seasonal-conformal intervals (test 2024-2025)\n"
                 "yellow = wet season (Dec-Apr)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "13_ensemble_seasonal_bands.png", dpi=150); plt.close(fig)
    log.info("Saved: 13_ensemble_seasonal_bands.png")

    # ── Figure: coverage bars ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = ["ALL", "WET", "DRY"]
    masks  = [np.ones(len(test),bool), test_season=="wet", test_season=="dry"]
    raw  = [picp(test[m]["obs"], test[m]["lo_raw"], test[m]["hi_raw"]) for m in masks]
    glob = [picp(test[m]["obs"], test[m]["lo_glob"], test[m]["hi_glob"]) for m in masks]
    seas = [picp(test[m]["obs"], test[m]["lo_seas"], test[m]["hi_seas"]) for m in masks]
    x = np.arange(3)
    ax.bar(x-0.25, raw, 0.25, label="raw bands", color="#cc2222", alpha=0.8)
    ax.bar(x,      glob,0.25, label="global conformal", color="#e8a000", alpha=0.8)
    ax.bar(x+0.25, seas,0.25, label="seasonal conformal", color="#2ca02c", alpha=0.8)
    ax.axhline(1-ALPHA, color="black", ls="--", lw=1.5, label=f"nominal {1-ALPHA:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("PICP (interval coverage)"); ax.set_ylim(0, 1.05)
    ax.set_title("Interval coverage by season: raw vs conformal calibration")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG_DIR / "14_seasonal_coverage.png", dpi=150); plt.close(fig)
    log.info("Saved: 14_seasonal_coverage.png")

    log.info("Done.")


if __name__ == "__main__":
    main()
