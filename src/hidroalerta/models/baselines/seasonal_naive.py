"""Baseline M2: Seasonal Naive - valor del mismo día del año anterior."""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SeasonalNaive:
    """Predictor baseline: valor del mismo día del año anterior.

    Para cada fecha t, predice el valor observado en t - 365 días.
    Captura estacionalidad mejor que persistencia simple.
    """

    def __init__(self, lag_years: int = 1):
        self.lag_years = lag_years
        self._lookup: dict = {}  # (year, doy) -> valor
        self._fitted = False

    def fit(self, dates_train: pd.Series, y_train: pd.Series):
        """Guardar todos los valores de train indexados por (year, doy).

        Args:
            dates_train: Serie de fechas de train.
            y_train: Serie de valores target de train.
        """
        dates_dt = pd.to_datetime(dates_train)
        df = pd.DataFrame({
            "year": dates_dt.dt.year,
            "doy": dates_dt.dt.dayofyear,
            "y": y_train.values,
        })
        df = df.dropna(subset=["y"])
        for _, row in df.iterrows():
            self._lookup[(int(row["year"]), int(row["doy"]))] = row["y"]
        self._fitted = True
        logger.info(f"SeasonalNaive ajustado con {len(df)} muestras de train")
        return self

    def predict(self, dates: pd.Series) -> np.ndarray:
        """Predecir usando el valor de hace `lag_years` años en el mismo doy.

        Si no existe, usa el promedio del doy en todos los años disponibles.
        """
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de predict()")
        dates_dt = pd.to_datetime(dates)
        preds = []
        for dt in dates_dt:
            year = dt.year
            doy = dt.dayofyear
            # Intentar año anterior
            val = self._lookup.get((year - self.lag_years, doy))
            if val is None:
                # Fallback: cualquier año disponible para ese doy
                candidates = [v for (y, d), v in self._lookup.items() if d == doy]
                val = float(np.median(candidates)) if candidates else 0.0
            preds.append(val)
        return np.array(preds)
