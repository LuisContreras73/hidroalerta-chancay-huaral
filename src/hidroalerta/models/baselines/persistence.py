"""Baseline M1: Persistencia - usar el último valor observado como predicción."""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Persistence:
    """Baseline de persistencia.

    Para horizonte h, predice el valor actual (o acumulado reciente)
    como predicción del futuro. Es el benchmark mínimo razonable.
    """

    def __init__(self, horizon: int = 1, use_rolling: bool = True, window: int = 7):
        """
        Args:
            horizon: días de pronóstico.
            use_rolling: si True, para h>1 usa suma del rolling window actual.
            window: ventana de suma para pronósticos multi-día.
        """
        self.horizon = horizon
        self.use_rolling = use_rolling
        self.window = window
        self._fitted = False

    def fit(self, X_train, y_train):
        """Fit trivial: no hay parámetros a aprender."""
        self._fitted = True
        logger.info(f"Persistence fitted (horizon={self.horizon}). Baseline trivial.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predecir usando la columna de valor actual relevante.

        Busca en X: pr_lag_1 (último pr observado) o pr_sum_{window}d para multi-día.
        """
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de predict()")

        if self.horizon == 1:
            # Para 1 día: usar último pr observado
            col = "pr_lag_1"
            if col not in X.columns:
                col = next((c for c in X.columns if "lag_1" in c), None)
            if col is None:
                logger.warning("No se encontró pr_lag_1. Predicción = 0.")
                return np.zeros(len(X))
            return X[col].fillna(0).values

        else:
            # Para multi-día: suma actual de los últimos `window` días
            col = f"pr_sum_{self.window}d"
            if col in X.columns:
                return X[col].fillna(0).values
            # Fallback: pr_lag_1 * horizon
            col_lag = "pr_lag_1"
            if col_lag in X.columns:
                logger.warning(f"Columna {col} no encontrada. Usando pr_lag_1 * {self.horizon}.")
                return (X[col_lag].fillna(0) * self.horizon).values
            return np.zeros(len(X))

    def predict_from_series(self, pr_series: pd.Series) -> np.ndarray:
        """Predecir directamente desde serie de precipitación."""
        if self.horizon == 1:
            return pr_series.shift(1).fillna(0).values
        return pr_series.rolling(self.window).sum().shift(1).fillna(0).values
