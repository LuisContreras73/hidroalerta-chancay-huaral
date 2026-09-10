#!/usr/bin/env python3
"""
Script 18: Modelos ML baseline para pronóstico de precipitación.

Modelos: Persistencia, Climatología DOY, Random Forest, XGBoost, LightGBM.
Dataset: D5_tft_ready.csv (Script 16).
Targets: pr_next_1d, pr_sum_next_3d, pr_sum_next_7d.
Split temporal: 70% train / 15% val / 15% test (ya definido en D5).

Salidas:
  outputs/ml_baselines/leaderboard.csv
  outputs/ml_baselines/feature_importance.csv
  outputs/figures/basin/M01_ml_baselines.png
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "18_ml_baselines.log", "w", "utf-8"),
    ],
)
log = logging.getLogger("ml_baselines")

SEED   = 42
TARGET_MAP = {
    "pr_next_1d":     1,
    "pr_sum_next_3d": 3,
    "pr_sum_next_7d": 7,
}
PAST_FEATURES = [
    "pr_mm", "tmax_c", "tmin_c", "tmean_c", "pet_mm",
    "oni_index", "api", "spi_30d", "spi_90d", "water_deficit_30d",
]
CALENDAR_FEATURES = [
    "doy", "month", "hydro_month", "is_wet_season",
    "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
    "clim_pr_p50", "clim_pr_p90",
]
STATIC_FEATURES = [
    "elevation_m", "basin_area_km2", "slope_deg", "hyps_integral",
    "soil_sand_pct", "soil_clay_pct", "soil_bdod", "soil_ph",
]


# ─── Métricas ────────────────────────────────────────────────────────────────

def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    if len(o) < 3:
        return dict(mae=np.nan, rmse=np.nan, nse=np.nan, kge=np.nan, r=np.nan, n=0)
    mae  = float(np.mean(np.abs(o - p)))
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    nse  = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    r    = float(np.corrcoef(o, p)[0, 1])
    alpha = p.std() / (o.std() + 1e-12)
    beta  = p.mean() / (o.mean() + 1e-12)
    kge   = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
    return dict(mae=mae, rmse=rmse, nse=nse, kge=kge, r=r, n=int(mask.sum()))


# ─── Baselines clásicos ───────────────────────────────────────────────────────

def baseline_persistence(train, val, test, target):
    """Persistencia: ŷ(t) = y(t-1) (pr_mm del día anterior)."""
    preds = {}
    for name, df in [("val", val), ("test", test)]:
        preds[name] = df["pr_mm"].values  # pr_mm del día = proxy lag-1 hacia el siguiente
    return preds


def baseline_climatology(train, test, target):
    """Climatología DOY: usa clim_pr_p50 del dataset (calculado solo en train)."""
    preds = {}
    for name, df in [("test", test)]:
        preds[name] = df["clim_pr_p50"].values
    return preds


# ─── Feature matrix ──────────────────────────────────────────────────────────

def build_X(df: pd.DataFrame, extra_lags: bool = True) -> pd.DataFrame:
    """Construye matrix de features a partir de D5."""
    cols = PAST_FEATURES + CALENDAR_FEATURES + STATIC_FEATURES
    available = [c for c in cols if c in df.columns]
    X = df[available].copy()

    if extra_lags and "pr_mm" in df.columns:
        for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
            X[f"pr_lag_{lag}"] = df["pr_mm"].shift(lag)
        for lag in [3, 7, 14, 30]:
            X[f"pr_sum_{lag}d"] = df["pr_mm"].rolling(lag).sum()
        for lag in [3, 7]:
            if "tmax_c" in df.columns:
                X[f"tmax_lag_{lag}"] = df["tmax_c"].shift(lag)
            if "tmin_c" in df.columns:
                X[f"tmin_lag_{lag}"] = df["tmin_c"].shift(lag)

    return X.fillna(0)


# ─── ML trainers ─────────────────────────────────────────────────────────────

def train_rf(X_tr, y_tr):
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(SEED)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.6,
        random_state=SEED,
        n_jobs=-1,
    )
    mask = np.isfinite(y_tr)
    model.fit(X_tr[mask], y_tr[mask])
    return model


def train_xgb(X_tr, y_tr, X_val=None, y_val=None):
    import xgboost as xgb
    mask_tr = np.isfinite(y_tr)
    params = dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        random_state=SEED,
        verbosity=0,
        early_stopping_rounds=40,
    )
    eval_set = None
    if X_val is not None and y_val is not None:
        mask_val = np.isfinite(y_val)
        eval_set = [(X_val[mask_val], y_val[mask_val])]
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr[mask_tr], y_tr[mask_tr],
              eval_set=eval_set, verbose=False)
    return model


def train_lgb(X_tr, y_tr, X_val=None, y_val=None):
    import lightgbm as lgb
    mask_tr = np.isfinite(y_tr)
    params = dict(
        n_estimators=600,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=SEED,
        verbosity=-1,
    )
    callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=-1)]
    eval_set = None
    if X_val is not None and y_val is not None:
        mask_val = np.isfinite(y_val)
        eval_set = [(X_val[mask_val], y_val[mask_val])]
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {"callbacks": callbacks} if eval_set else {}
    if eval_set:
        fit_kwargs["eval_set"] = eval_set
    model.fit(X_tr[mask_tr], y_tr[mask_tr], **fit_kwargs)
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = PROJECT_ROOT / "outputs" / "ml_baselines"
    fig_dir = PROJECT_ROOT / "outputs" / "figures" / "basin"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("SCRIPT 18: ML Baselines — Cuenca Chancay-Huaral")
    log.info("=" * 70)

    # ── Cargar D5 ─────────────────────────────────────────────────────────────
    d5_path = PROJECT_ROOT / "data" / "model_ready" / "D5_tft_ready.csv"
    if not d5_path.exists():
        log.error("D5_tft_ready.csv no encontrado. Ejecutar script 16 primero.")
        sys.exit(1)

    df = pd.read_csv(d5_path, index_col=0, parse_dates=True)
    log.info(f"D5 cargado: {len(df)} filas × {len(df.columns)} cols")
    log.info(f"Período: {df.index.min().date()} a {df.index.max().date()}")

    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "val"].copy()
    test  = df[df["split"] == "test"].copy()
    log.info(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    X_tr  = build_X(train)
    X_val = build_X(val)
    X_test= build_X(test)

    results = []
    fi_records = []

    for target, horizon in TARGET_MAP.items():
        log.info(f"\n── Target: {target} (H={horizon}d) ──")
        y_tr   = train[target].values
        y_val_ = val[target].values
        y_test = test[target].values

        t0 = time.perf_counter()

        # ── Persistencia ──────────────────────────────────────────────────────
        pers_pred = val["pr_mm"].values
        m = metrics(y_val_, pers_pred)
        log.info(f"  Persistencia  val: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "Persistencia", "target": target, "horizon": horizon,
                        "split": "val", **m})
        pers_pred_te = test["pr_mm"].values
        m = metrics(y_test, pers_pred_te)
        log.info(f"  Persistencia  test: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "Persistencia", "target": target, "horizon": horizon,
                        "split": "test", **m})

        # ── Climatología ──────────────────────────────────────────────────────
        clim_pred_val  = val["clim_pr_p50"].values
        clim_pred_test = test["clim_pr_p50"].values
        m = metrics(y_val_, clim_pred_val)
        log.info(f"  Climatología  val: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "Climatologia_DOY", "target": target, "horizon": horizon,
                        "split": "val", **m})
        m = metrics(y_test, clim_pred_test)
        log.info(f"  Climatología  test: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "Climatologia_DOY", "target": target, "horizon": horizon,
                        "split": "test", **m})

        # ── Random Forest ─────────────────────────────────────────────────────
        log.info("  Entrenando Random Forest ...")
        rf = train_rf(X_tr.values, y_tr)
        pv = np.clip(rf.predict(X_val.values), 0, None)
        pt = np.clip(rf.predict(X_test.values), 0, None)
        m = metrics(y_val_, pv); log.info(f"  RF val:  MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "RandomForest", "target": target, "horizon": horizon,
                        "split": "val", **m})
        m = metrics(y_test, pt); log.info(f"  RF test: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "RandomForest", "target": target, "horizon": horizon,
                        "split": "test", **m})
        for feat, imp in zip(X_tr.columns, rf.feature_importances_):
            fi_records.append({"model": "RandomForest", "target": target, "feature": feat, "importance": imp})

        # ── XGBoost ───────────────────────────────────────────────────────────
        log.info("  Entrenando XGBoost ...")
        xgb_m = train_xgb(X_tr.values, y_tr, X_val.values, y_val_)
        pv = np.clip(xgb_m.predict(X_val.values), 0, None)
        pt = np.clip(xgb_m.predict(X_test.values), 0, None)
        m = metrics(y_val_, pv); log.info(f"  XGB val:  MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "XGBoost", "target": target, "horizon": horizon,
                        "split": "val", **m})
        m = metrics(y_test, pt); log.info(f"  XGB test: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "XGBoost", "target": target, "horizon": horizon,
                        "split": "test", **m})
        fi_xgb = xgb_m.feature_importances_
        for feat, imp in zip(X_tr.columns, fi_xgb):
            fi_records.append({"model": "XGBoost", "target": target, "feature": feat, "importance": imp})

        # ── LightGBM ──────────────────────────────────────────────────────────
        log.info("  Entrenando LightGBM ...")
        lgb_m = train_lgb(X_tr.values, y_tr, X_val.values, y_val_)
        pv = np.clip(lgb_m.predict(X_val), 0, None)
        pt = np.clip(lgb_m.predict(X_test), 0, None)
        m = metrics(y_val_, pv); log.info(f"  LGB val:  MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "LightGBM", "target": target, "horizon": horizon,
                        "split": "val", **m})
        m = metrics(y_test, pt); log.info(f"  LGB test: MAE={m['mae']:.3f} NSE={m['nse']:.3f}")
        results.append({"model": "LightGBM", "target": target, "horizon": horizon,
                        "split": "test", **m})
        fi_lgb = lgb_m.feature_importances_
        for feat, imp in zip(X_tr.columns, fi_lgb / (fi_lgb.sum() + 1e-12)):
            fi_records.append({"model": "LightGBM", "target": target, "feature": feat, "importance": imp})

        elapsed = time.perf_counter() - t0
        log.info(f"  Target {target} completado en {elapsed:.1f}s")

    # ── Guardar leaderboard ────────────────────────────────────────────────────
    lb = pd.DataFrame(results)
    lb_path = out_dir / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    log.info(f"\nLeaderboard guardado: {lb_path}")

    fi_df = pd.DataFrame(fi_records)
    fi_path = out_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    log.info(f"Feature importance guardado: {fi_path}")

    # ── Figura M01 ────────────────────────────────────────────────────────────
    _plot_baselines(lb, fig_dir / "M01_ml_baselines.png")

    # ── Resumen ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("RESUMEN — Test set (pr_next_1d)")
    sub = lb[(lb["target"] == "pr_next_1d") & (lb["split"] == "test")]
    log.info(sub[["model", "mae", "rmse", "nse", "kge"]].to_string(index=False))
    log.info("=" * 70)


def _plot_baselines(lb: pd.DataFrame, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models   = ["Persistencia", "Climatologia_DOY", "RandomForest", "XGBoost", "LightGBM"]
    targets  = list(TARGET_MAP.keys())
    horizons = list(TARGET_MAP.values())
    metrics_show = ["mae", "rmse", "nse", "kge"]
    metric_labels = ["MAE (mm/d)", "RMSE (mm/d)", "NSE", "KGE"]

    fig, axes = plt.subplots(len(metrics_show), len(targets),
                             figsize=(13, 11), constrained_layout=True)

    colors = {
        "Persistencia":    "#7f8c8d",
        "Climatologia_DOY":"#95a5a6",
        "RandomForest":    "#2980b9",
        "XGBoost":         "#e74c3c",
        "LightGBM":        "#27ae60",
    }
    x     = np.arange(len(models))
    width = 0.6

    for ci, (target, hor) in enumerate(zip(targets, horizons)):
        for ri, (met, mlabel) in enumerate(zip(metrics_show, metric_labels)):
            ax = axes[ri, ci]
            test_sub = lb[(lb["target"] == target) & (lb["split"] == "test")]
            vals = []
            for model in models:
                row = test_sub[test_sub["model"] == model]
                vals.append(float(row[met].values[0]) if len(row) else np.nan)

            bars = ax.bar(x, vals, width=width,
                          color=[colors[m] for m in models],
                          edgecolor="white", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(
                ["Pers.", "Clim.", "RF", "XGB", "LGB"],
                fontsize=7, rotation=30, ha="right"
            )
            if ri == 0:
                ax.set_title(f"H={hor}d\n({target})", fontsize=8, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)

            # Referencia: línea en 0 para NSE/KGE
            if met in ("nse", "kge"):
                ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

            # Etiquetas encima de barras
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*abs(bar.get_height() + 1e-6),
                            f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)

    # Leyenda
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[m], label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "ML Baselines — Pronóstico de precipitación (test set)\n"
        "Cuenca Chancay-Huaral · PISCOp 1981-2020",
        fontsize=10, fontweight="bold", y=1.01
    )

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Figura guardada: {fig_path.name}")


if __name__ == "__main__":
    main()
