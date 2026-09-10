"""Baseline M0: Climatología por día del año (mediana histórica)."""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClimatologyDOY:
    """Predictor baseline: mediana histórica por día del año.

    Calcula percentiles por doy solo en train. Predice la mediana (p50)
    del doy correspondiente para cualquier fecha futura.
    """

    def __init__(self, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)):
        self.quantiles = quantiles
        self._stats = {}  # doy -> {q: valor}
        self._fitted = False

    def fit(self, dates_train: pd.Series, y_train: pd.Series):
        """Ajustar climatología sobre datos de train.

        Args:
            dates_train: Serie de fechas (datetime) de train.
            y_train: Serie de valores target de train.
        """
        df = pd.DataFrame({"doy": pd.to_datetime(dates_train).dt.dayofyear, "y": y_train})
        df = df.dropna(subset=["y"])

        for doy, group in df.groupby("doy"):
            self._stats[int(doy)] = {
                q: float(group["y"].quantile(q)) for q in self.quantiles
            }

        # Rellenar doys faltantes por interpolación del más cercano
        all_doys = set(range(1, 367))
        missing = all_doys - set(self._stats.keys())
        for doy in missing:
            neighbor = min(self._stats.keys(), key=lambda d: abs(d - doy))
            self._stats[int(doy)] = self._stats[neighbor]

        self._fitted = True
        logger.info(f"ClimatologyDOY ajustada con {len(df)} muestras, {len(self._stats)} doys únicos")
        return self

    def predict(self, dates: pd.Series, quantile: float = 0.50) -> np.ndarray:
        """Predecir usando la climatología del doy.

        Args:
            dates: Fechas para predecir.
            quantile: Cuantil a usar como predicción puntual (default 0.50 = mediana).
        Returns:
            Array de predicciones.
        """
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de predict()")
        doys = pd.to_datetime(dates).dt.dayofyear
        preds = np.array([
            self._stats.get(int(d), {}).get(quantile, np.nan) for d in doys
        ])
        return preds

    def predict_quantile(self, dates: pd.Series, q: float) -> np.ndarray:
        """Predicción para un cuantil específico."""
        return self.predict(dates, quantile=q)

    def predict_all_quantiles(self, dates: pd.Series) -> pd.DataFrame:
        """Devuelve DataFrame con una columna por cuantil."""
        doys = pd.to_datetime(dates).dt.dayofyear
        result = {}
        for q in self.quantiles:
            col = f"clim_p{int(q*100):02d}"
            result[col] = [self._stats.get(int(d), {}).get(q, np.nan) for d in doys]
        return pd.DataFrame(result, index=dates.index)

    def get_doy_stats(self) -> pd.DataFrame:
        """Retorna tabla de estadísticos por día del año."""
        rows = []
        for doy in sorted(self._stats.keys()):
            row = {"doy": doy}
            row.update(self._stats[doy])
            rows.append(row)
        return pd.DataFrame(rows)
