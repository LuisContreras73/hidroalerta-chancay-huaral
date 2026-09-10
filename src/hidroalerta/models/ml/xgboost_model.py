"""Modelo M6: XGBoost para pronóstico pluviométrico."""
import logging
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    logger.warning("xgboost no instalado. XGBoostModel usará RandomForest como fallback.")

try:
    from sklearn.ensemble import RandomForestRegressor
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


class XGBoostModel:
    """Wrapper de XGBRegressor. Si xgboost no está, usa RandomForest como fallback."""

    MODEL_FAMILY = "ml"
    MODEL_NAME = "xgboost"

    def __init__(self, params: dict = None, seed: int = 42):
        self.params = params or {}
        self.seed = seed
        self.model = None
        self.feature_names = None
        self.best_params = {}
        self._using_fallback = not _XGB

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrenar XGBoost (o RandomForest fallback) con early stopping opcional."""
        mask_tr = ~np.isnan(np.array(y_train, dtype=float))
        X_tr = X_train[mask_tr] if hasattr(X_train, '__getitem__') else X_train
        y_tr = np.array(y_train)[mask_tr]

        if hasattr(X_tr, 'columns'):
            self.feature_names = list(X_tr.columns)

        if _XGB:
            default_params = {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.seed,
                "tree_method": "hist",
                "verbosity": 0,
            }
            params = {**default_params, **self.params}
            self.best_params = params

            callbacks = []
            eval_set = None
            if X_val is not None and y_val is not None:
                mask_val = ~np.isnan(np.array(y_val, dtype=float))
                eval_set = [(X_val[mask_val], np.array(y_val)[mask_val])]
                params["early_stopping_rounds"] = params.pop("early_stopping_rounds", 30)

            self.model = xgb.XGBRegressor(**params)
            fit_kwargs = {}
            if eval_set:
                fit_kwargs = {"eval_set": eval_set, "verbose": False}
            self.model.fit(X_tr, y_tr, **fit_kwargs)
            logger.info(f"XGBoost entrenado. Params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, lr={params['learning_rate']}")

        elif _SKLEARN:
            logger.warning("Usando RandomForest como fallback de XGBoost")
            self._using_fallback = True
            self.model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=self.seed, n_jobs=-1)
            self.model.fit(X_tr, y_tr)
            self.best_params = {"fallback": "RandomForest"}
        else:
            raise ImportError("Ni xgboost ni scikit-learn están instalados")

        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Llamar fit() antes de predict()")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        if self.model is None or self.feature_names is None:
            return {}
        if _XGB and not self._using_fallback:
            scores = self.model.feature_importances_
        else:
            scores = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
        return dict(sorted(zip(self.feature_names, scores), key=lambda x: x[1], reverse=True))

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        fpath = path / "model.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Modelo guardado en {fpath}")

    def load(self, path: Path):
        with open(Path(path) / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        return self
