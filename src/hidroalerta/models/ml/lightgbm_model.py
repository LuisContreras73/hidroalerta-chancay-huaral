"""Modelo M7: LightGBM para pronóstico pluviométrico."""
import logging
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False
    logger.warning("lightgbm no instalado. LightGBMModel usará RandomForest como fallback.")

try:
    from sklearn.ensemble import RandomForestRegressor
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


class LightGBMModel:
    """Wrapper de LGBMRegressor. Si lightgbm no está disponible, usa RandomForest."""

    MODEL_FAMILY = "ml"
    MODEL_NAME = "lightgbm"

    def __init__(self, params: dict = None, seed: int = 42):
        self.params = params or {}
        self.seed = seed
        self.model = None
        self.feature_names = None
        self.best_params = {}
        self._using_fallback = not _LGB

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        mask_tr = ~np.isnan(np.array(y_train, dtype=float))
        X_tr = X_train[mask_tr] if hasattr(X_train, '__getitem__') else X_train
        y_tr = np.array(y_train)[mask_tr]

        if hasattr(X_tr, 'columns'):
            self.feature_names = list(X_tr.columns)

        if _LGB:
            default_params = {
                "n_estimators": 400,
                "max_depth": 5,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.seed,
                "verbosity": -1,
                "n_jobs": -1,
            }
            params = {**default_params, **self.params}
            self.best_params = params

            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
            eval_set = None
            if X_val is not None and y_val is not None:
                mask_val = ~np.isnan(np.array(y_val, dtype=float))
                eval_set = [(X_val[mask_val], np.array(y_val)[mask_val])]

            self.model = lgb.LGBMRegressor(**params)
            fit_kwargs = {}
            if eval_set:
                fit_kwargs = {"eval_set": eval_set, "callbacks": callbacks}
            self.model.fit(X_tr, y_tr, **fit_kwargs)
            logger.info(f"LightGBM entrenado. n_estimators={params['n_estimators']}, num_leaves={params['num_leaves']}")

        elif _SKLEARN:
            logger.warning("Usando RandomForest como fallback de LightGBM")
            self._using_fallback = True
            self.model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=self.seed, n_jobs=-1)
            self.model.fit(X_tr, y_tr)
            self.best_params = {"fallback": "RandomForest"}
        else:
            raise ImportError("Ni lightgbm ni scikit-learn están instalados")

        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Llamar fit() antes de predict()")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        if self.model is None or self.feature_names is None:
            return {}
        scores = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
        return dict(sorted(zip(self.feature_names, scores), key=lambda x: x[1], reverse=True))

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Modelo guardado en {path}/model.pkl")

    def load(self, path: Path):
        with open(Path(path) / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        return self
