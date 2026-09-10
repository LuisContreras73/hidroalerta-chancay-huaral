"""Modelos M3/M4: Regresión lineal y Ridge para pronóstico pluviométrico."""
import logging
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    logger.warning("scikit-learn no instalado.")


class LinearLagsModel:
    """M3: Regresión lineal con lags básicos."""

    MODEL_FAMILY = "ml"
    MODEL_NAME = "linear_lags"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.feature_names = None
        self.best_params = {"fit_intercept": True}

    def fit(self, X_train, y_train):
        if not _SKLEARN:
            raise ImportError("scikit-learn requerido")
        mask = ~np.isnan(np.array(y_train, dtype=float))
        X_tr = X_train[mask]
        y_tr = np.array(y_train)[mask]
        if hasattr(X_tr, 'columns'):
            self.feature_names = list(X_tr.columns)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression(fit_intercept=True))
        ])
        self.model.fit(X_tr, y_tr)
        logger.info(f"LinearLags entrenado con {X_tr.shape[0]} muestras")
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        if self.model is None or self.feature_names is None:
            return {}
        coefs = np.abs(self.model.named_steps["reg"].coef_)
        return dict(sorted(zip(self.feature_names, coefs), key=lambda x: x[1], reverse=True))

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)


class RidgeModel:
    """M4: Ridge regression con regularización L2."""

    MODEL_FAMILY = "ml"
    MODEL_NAME = "ridge"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.model = None
        self.feature_names = None
        self.best_params = {"alpha": alpha}

    def fit(self, X_train, y_train):
        if not _SKLEARN:
            raise ImportError("scikit-learn requerido")
        mask = ~np.isnan(np.array(y_train, dtype=float))
        X_tr = X_train[mask]
        y_tr = np.array(y_train)[mask]
        if hasattr(X_tr, 'columns'):
            self.feature_names = list(X_tr.columns)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.alpha))
        ])
        self.model.fit(X_tr, y_tr)
        logger.info(f"Ridge(alpha={self.alpha}) entrenado con {X_tr.shape[0]} muestras")
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        if self.model is None or self.feature_names is None:
            return {}
        coefs = np.abs(self.model.named_steps["reg"].coef_)
        return dict(sorted(zip(self.feature_names, coefs), key=lambda x: x[1], reverse=True))

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
