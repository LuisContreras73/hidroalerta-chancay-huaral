#!/usr/bin/env python3
"""Script 11: Entrenar modelos ML (Random Forest, XGBoost, LightGBM)."""
import logging
import sys
import random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_ml")

TARGETS = ["pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d"]
N_TRIALS = 5
SEED = 42


def get_feature_cols(df):
    exclude = {"date", "entity_id", "split", "pr_mm", "tmax_c", "tmin_c", "tmean_c",
               "pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d", "rain_alert_category_next_7d",
               "quality_flag"}
    return [c for c in df.columns if c not in exclude and not c.startswith("rain_alert")]


def random_params_rf():
    return {
        "n_estimators": random.choice([100, 200, 300, 500]),
        "max_depth": random.choice([3, 4, 5, 6, None]),
        "min_samples_leaf": random.choice([5, 10, 20]),
        "max_features": random.choice([0.5, 0.7, "sqrt"]),
        "random_state": SEED,
    }


def random_params_xgb():
    return {
        "n_estimators": random.choice([200, 400, 600]),
        "max_depth": random.choice([2, 3, 4, 5]),
        "learning_rate": random.choice([0.01, 0.03, 0.05, 0.1]),
        "subsample": random.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": random.choice([0.7, 0.8, 0.9, 1.0]),
        "random_state": SEED,
    }


def random_params_lgb():
    return {
        "n_estimators": random.choice([200, 400, 600]),
        "num_leaves": random.choice([15, 31, 63]),
        "learning_rate": random.choice([0.01, 0.03, 0.05, 0.1]),
        "subsample": random.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": random.choice([0.7, 0.8, 0.9, 1.0]),
        "random_state": SEED,
        "verbosity": -1,
    }


def main():
    import pandas as pd
    from hidroalerta.evaluation.metrics import compute_all_regression_metrics
    from hidroalerta.tracking.run_manager import RunManager
    from hidroalerta.models.ml.random_forest import RandomForestModel
    from hidroalerta.models.ml.xgboost_model import XGBoostModel
    from hidroalerta.models.ml.lightgbm_model import LightGBMModel

    random.seed(SEED)
    np.random.seed(SEED)

    mr_path = PROJECT_ROOT / "data" / "model_ready" / "D4_pisco_model_ready_precip.csv"
    if not mr_path.exists():
        logger.error("Model-ready no encontrado. Correr script 08 primero.")
        sys.exit(1)

    df = pd.read_csv(mr_path, parse_dates=["date"])
    feature_cols = get_feature_cols(df)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train = train[feature_cols].fillna(0)
    X_val = val[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)

    rm = RunManager(project_root=PROJECT_ROOT)

    model_configs = [
        ("M5_random_forest", "ml", RandomForestModel, random_params_rf),
        ("M6_xgboost", "ml", XGBoostModel, random_params_xgb),
        ("M7_lightgbm", "ml", LightGBMModel, random_params_lgb),
    ]

    for target in TARGETS:
        horizon = 1 if "1d" in target else (3 if "3d" in target else 7)
        y_train = train[target].values
        y_val = val[target].values
        y_test = test[target].values

        for model_id, family, ModelClass, param_fn in model_configs:
            logger.info(f"\n{'='*50}\n{model_id} | {target}\n{'='*50}")
            for trial in range(N_TRIALS):
                params = param_fn()
                logger.info(f"  Trial {trial+1}/{N_TRIALS}: {params}")
                try:
                    model = ModelClass(params=params, seed=SEED)
                    model.fit(X_train, y_train, X_val, y_val)
                    pred_test = model.predict(X_test)
                    pred_test = np.clip(pred_test, 0, None)

                    mask = ~np.isnan(y_test)
                    metrics = compute_all_regression_metrics(y_test[mask], pred_test[mask])

                    run_id = rm.create_run_id(model_id, target, horizon, "D4", SEED)
                    run_dir = rm.create_run_dir(family, model_id, run_id)
                    rm.save_metrics(run_dir, metrics)
                    rm.save_predictions(run_dir, test["date"], y_test, pred_test)

                    # Feature importance
                    fi = model.get_feature_importance()
                    if fi:
                        import pandas as _pd
                        fi_df = _pd.DataFrame(list(fi.items()), columns=["feature", "importance"])
                        fi_df.to_csv(run_dir / "feature_importance.csv", index=False)

                    rm.update_leaderboard(
                        run_id=run_id, model_family=family, model_name=model_id,
                        target=target, horizon=horizon, metrics=metrics,
                        train_start=str(train["date"].min()), train_end=str(train["date"].max()),
                        val_start=str(val["date"].min()), val_end=str(val["date"].max()),
                        test_start=str(test["date"].min()), test_end=str(test["date"].max()),
                        figures_path=str(run_dir / "figures"), model_path=str(run_dir / "model_artifact"),
                        status="success", trial_id=trial
                    )
                    logger.info(f"    MAE={metrics.get('mae', 'N/A'):.3f} NSE={metrics.get('nse', 'N/A'):.3f}")

                except Exception as e:
                    logger.error(f"    Trial {trial+1} FALLÓ: {e}")
                    rm.log_failed_run(model_id, target, str(e))

    logger.info("\nML entrenado. Ver outputs/leaderboards/all_runs.csv")


if __name__ == "__main__":
    main()
