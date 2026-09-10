"""Modelo M5: Random Forest para pronóstico pluviométrico."""
import logging
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    logger.warning("scikit-learn no instalado. RandomForestModel no disponible.")


class RandomForestModel:
    """Wrapper de sklearn RandomForestRegressor con búsqueda de hiperparámetros."""

    MODEL_FAMILY = "ml"
    MODEL_NAME = "random_forest"

    def __init__(self, params: dict = None, n_trials: int = 5, seed: int = 42):
        if not _SKLEARN:
            raise ImportError("scikit-learn es necesario para RandomForestModel")
        self.params = params or {}
        self.n_trials = n_trials
        self.seed = seed
        self.model = None
        self.feature_names = None
        self.best_params = {}

    def fit(self, X_train, y_train, params_grid: dict = None):
        """Entrenar con RandomizedSearch si hay params_grid, sino con params fijos."""
        mask = ~np.isnan(y_train) if hasattr(y_train, '__len__') else slice(None)
        X_tr = X_train[mask] if hasattr(mask, '__len__') else X_train
        y_tr = y_train[mask] if hasattr(mask, '__len__') else y_train

        if hasattr(X_tr, 'columns'):
            self.feature_names = list(X_tr.columns)

        if params_grid and self.n_trials > 1:
            base_rf = RandomForestRegressor(random_state=self.seed, n_jobs=-1)
            search = RandomizedSearchCV(
                base_rf, params_grid,
                n_iter=self.n_trials,
                cv=3, scoring="neg_mean_absolute_error",
                random_state=self.seed, n_jobs=-1
            )
            search.fit(X_tr, y_tr)
            self.model = search.best_estimator_
            self.best_params = search.best_params_
            logger.info(f"RandomForest best params: {self.best_params}")
        else:
            params = {**{"n_estimators": 200, "max_depth": 5, "random_state": self.seed}, **self.params}
            self.model = RandomForestRegressor(**params, n_jobs=-1)
            self.model.fit(X_tr, y_tr)
            self.best_params = params

        logger.info(f"RandomForest entrenado con {X_tr.shape[0]} muestras, {X_tr.shape[1]} features")
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Llamar fit() antes de predict()")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        """Retorna importancia de features como dict {nombre: importancia}."""
        if self.model is None or self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))

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
