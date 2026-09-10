#!/usr/bin/env python3
"""Script 09: Entrenar modelos baseline (climatología, persistencia, seasonal naive, Ridge)."""
import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_baselines")

TARGETS = ["pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d"]
FEATURE_COLS_BASIC = [
    "pr_lag_1", "pr_lag_2", "pr_lag_3", "pr_lag_7", "pr_lag_14",
    "pr_sum_3d", "pr_sum_7d",
    "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
    "month", "hydro_month",
    "clim_pr_p50_doy",
]


def main():
    import pandas as pd
    import numpy as np
    from hidroalerta.evaluation.metrics import compute_all_regression_metrics
    from hidroalerta.tracking.run_manager import RunManager
    from hidroalerta.models.baselines.climatology import ClimatologyDOY
    from hidroalerta.models.baselines.persistence import Persistence
    from hidroalerta.models.baselines.seasonal_naive import SeasonalNaive

    mr_path = PROJECT_ROOT / "data" / "model_ready" / "D4_pisco_model_ready_precip.csv"
    if not mr_path.exists():
        logger.error("Model-ready no encontrado. Correr script 08 primero.")
        sys.exit(1)

    df = pd.read_csv(mr_path, parse_dates=["date"])
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    rm = RunManager(project_root=PROJECT_ROOT)

    models_to_run = [
        ("M0_climatology_doy", "baseline", ClimatologyDOY()),
        ("M1_persistence", "baseline", None),
        ("M2_seasonal_naive", "baseline", SeasonalNaive()),
    ]

    for target in TARGETS:
        horizon = 1 if "1d" in target else (3 if "3d" in target else 7)
        logger.info(f"\n{'='*60}\nTarget: {target} | Horizon: {horizon}\n{'='*60}")

        for model_id, family, model in models_to_run:
            logger.info(f"  Entrenando {model_id}...")
            try:
                y_train = train[target].values
                y_val = val[target].values
                y_test = test[target].values

                if model_id == "M0_climatology_doy":
                    model_inst = ClimatologyDOY()
                    model_inst.fit(train["date"], pd.Series(y_train))
                    pred_val = model_inst.predict(val["date"])
                    pred_test = model_inst.predict(test["date"])

                elif model_id == "M1_persistence":
                    model_inst = Persistence(horizon=horizon)
                    model_inst.fit(None, None)
                    feature_col = "pr_lag_1" if "pr_lag_1" in val.columns else None
                    if feature_col:
                        pred_val = val[feature_col].values
                        pred_test = test[feature_col].values
                    else:
                        pred_val = np.zeros(len(val))
                        pred_test = np.zeros(len(test))

                elif model_id == "M2_seasonal_naive":
                    model_inst = SeasonalNaive()
                    model_inst.fit(train["date"], pd.Series(y_train))
                    pred_val = model_inst.predict(val["date"])
                    pred_test = model_inst.predict(test["date"])

                # Calcular métricas
                mask_val = ~np.isnan(y_val)
                mask_test = ~np.isnan(y_test)
                metrics_val = compute_all_regression_metrics(y_val[mask_val], pred_val[mask_val])
                metrics_test = compute_all_regression_metrics(y_test[mask_test], pred_test[mask_test])

                run_id = rm.create_run_id(
                    model_name=model_id,
                    target=target,
                    horizon=horizon,
                    dataset_version="D4",
                    seed=0
                )
                run_dir = rm.create_run_dir(family, model_id, run_id)
                rm.save_metrics(run_dir, metrics_test, split="test")
                rm.save_predictions(run_dir, test["date"], y_test, pred_test, split="test")
                rm.update_leaderboard(
                    run_id=run_id, model_family=family, model_name=model_id,
                    target=target, horizon=horizon, metrics=metrics_test,
                    train_start=str(train["date"].min()), train_end=str(train["date"].max()),
                    val_start=str(val["date"].min()), val_end=str(val["date"].max()),
                    test_start=str(test["date"].min()), test_end=str(test["date"].max()),
                    figures_path=str(run_dir / "figures"), model_path="none",
                    status="success"
                )
                logger.info(f"    {model_id} | MAE={metrics_test.get('mae', 'N/A'):.3f} | NSE={metrics_test.get('nse', 'N/A'):.3f}")

            except Exception as e:
                logger.error(f"    ERROR en {model_id}: {e}")
                rm.log_failed_run(model_id, target, str(e))

    logger.info("\nBaselines entrenados. Ver outputs/leaderboards/all_runs.csv")


if __name__ == "__main__":
    main()
