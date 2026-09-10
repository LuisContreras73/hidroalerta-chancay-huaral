"""Feature engineering para datos PISCO (precipitación y temperatura).

Referencia metodológica:
- Senamhi PISCO v2.1 (Aybar et al. 2020)
- FAO-56: Allen et al. (1998), Hargreaves-Samani PET
- Todas las funciones respetan el split train/test para evitar data leakage.
"""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hidroalerta.utils.datetime_utils import (
    compute_ra_hargreaves,
    doy_fourier,
    hydro_month,
    hydro_phase,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columnas estándar internas
# ---------------------------------------------------------------------------
_COL_DATE = "date"
_COL_PR = "pr_mm"
_COL_TMAX = "tmax_c"
_COL_TMIN = "tmin_c"
_COL_TMEAN = "tmean_c"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_pisco_data(path: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV o NetCDF con datos PISCO diarios.

    Columnas esperadas (acepta variantes): date, pr_mm (o pr),
    tmax_c (o tmax), tmin_c (o tmin).

    Returns
    -------
    pd.DataFrame con columna 'date' como datetime.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    logger.info("Cargando datos PISCO desde %s", path)

    if suffix in {".csv", ".txt"}:
        # Detectar automáticamente si la primera columna lleva el nombre 'date'
        _header = pd.read_csv(path, nrows=0)
        if _COL_DATE in _header.columns:
            df = pd.read_csv(path, parse_dates=[_COL_DATE])
        else:
            # La primera columna es la fecha pero con nombre distinto
            df = pd.read_csv(path)
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col])
            df = df.rename(columns={first_col: _COL_DATE})
    elif suffix in {".nc", ".nc4", ".netcdf"}:
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "xarray es necesario para leer NetCDF. Instalar con: pip install xarray"
            ) from exc
        ds = xr.open_dataset(path)
        df = ds.to_dataframe().reset_index()
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Use CSV o NetCDF.")

    df = normalize_columns(df)
    if _COL_DATE in df.columns:
        df[_COL_DATE] = pd.to_datetime(df[_COL_DATE])
        df = df.sort_values(_COL_DATE).reset_index(drop=True)

    logger.info(
        "Datos cargados: %d filas, %d columnas. Rango: %s → %s",
        len(df),
        len(df.columns),
        df[_COL_DATE].min() if _COL_DATE in df.columns else "?",
        df[_COL_DATE].max() if _COL_DATE in df.columns else "?",
    )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas a nombres estándar internos del paquete.

    Mapeos aceptados:
    - pr / precipitation / precip / rainfall / rain / pp → pr_mm
    - tmax / tmax_degc / t_max / temp_max → tmax_c
    - tmin / tmin_degc / t_min / temp_min → tmin_c
    - tmean / tavg / t_mean / temp_mean → tmean_c
    """
    rename_map: dict[str, str] = {}
    col_lower = {c.lower(): c for c in df.columns}

    _pr_aliases = {"pr", "precipitation", "precip", "rainfall", "rain", "pp"}
    _tmax_aliases = {"tmax", "tmax_degc", "t_max", "temp_max", "tempmax"}
    _tmin_aliases = {"tmin", "tmin_degc", "t_min", "temp_min", "tempmin"}
    _tmean_aliases = {"tmean", "tavg", "t_mean", "temp_mean", "tempmean", "temp"}

    for alias in _pr_aliases:
        if alias in col_lower and _COL_PR not in df.columns:
            rename_map[col_lower[alias]] = _COL_PR
            break
    for alias in _tmax_aliases:
        if alias in col_lower and _COL_TMAX not in df.columns:
            rename_map[col_lower[alias]] = _COL_TMAX
            break
    for alias in _tmin_aliases:
        if alias in col_lower and _COL_TMIN not in df.columns:
            rename_map[col_lower[alias]] = _COL_TMIN
            break
    for alias in _tmean_aliases:
        if alias in col_lower and _COL_TMEAN not in df.columns:
            rename_map[col_lower[alias]] = _COL_TMEAN
            break

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug("Columnas renombradas: %s", rename_map)
    return df


# ---------------------------------------------------------------------------
# Temperatura derivada
# ---------------------------------------------------------------------------

def add_temperature_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega tmean_c = (tmax_c + tmin_c) / 2 si ambas columnas existen."""
    df = df.copy()
    if _COL_TMAX in df.columns and _COL_TMIN in df.columns:
        df[_COL_TMEAN] = (df[_COL_TMAX] + df[_COL_TMIN]) / 2.0
        logger.debug("tmean_c calculada como promedio de tmax_c y tmin_c.")
    else:
        logger.warning("No se puede calcular tmean_c: faltan tmax_c o tmin_c.")
    return df


# ---------------------------------------------------------------------------
# PET Hargreaves-Samani (FAO-56 simplificado)
# ---------------------------------------------------------------------------

def compute_pet_hargreaves(df: pd.DataFrame, lat: float) -> pd.DataFrame:
    """Calcula PET diaria (mm/día) mediante el método Hargreaves-Samani (FAO-56).

    Requiere columnas: tmax_c, tmin_c, date.

    Formula (Hargreaves & Samani 1985):
        PET = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)

    Ra se calcula teóricamente por latitud y día del año (FAO-56 eq. 21).

    Limitación conocida: Ra teórica ignora nubosidad real. La PET puede
    estar sobreestimada en meses húmedos. Usar como aproximación a escala
    diaria. Para mayor precisión usar Penman-Monteith con datos de viento
    y humedad relativa.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas tmax_c, tmin_c, date.
    lat : float
        Latitud en grados decimales (negativo = hemisferio sur).

    Returns
    -------
    pd.DataFrame con nueva columna 'pet_mm'.
    """
    df = df.copy()
    required = {_COL_TMAX, _COL_TMIN, _COL_DATE}
    missing = required - set(df.columns)
    if missing:
        logger.warning("PET no calculada; columnas faltantes: %s", missing)
        return df

    if _COL_TMEAN not in df.columns:
        df = add_temperature_derived(df)

    doy: np.ndarray = np.asarray(pd.to_datetime(df[_COL_DATE]).dt.dayofyear, dtype=np.int32)
    ra = compute_ra_hargreaves(doy, lat)
    td: np.ndarray = np.asarray((df[_COL_TMAX] - df[_COL_TMIN]).clip(lower=0.0), dtype=np.float64)
    tmean: np.ndarray = np.asarray(df[_COL_TMEAN], dtype=np.float64)

    pet = 0.0023 * ra * (tmean + 17.8) * np.sqrt(td)
    pet = np.where(pet < 0, 0.0, pet)
    df["pet_mm"] = pet
    logger.info(
        "PET Hargreaves-Samani calculada (lat=%.4f°); Ra teórica FAO-56. "
        "Nota: puede sobreestimarse en meses húmedos.",
        lat,
    )
    return df


# ---------------------------------------------------------------------------
# Lags de precipitación
# ---------------------------------------------------------------------------

def add_precipitation_lags(
    df: pd.DataFrame,
    lags: Sequence[int] = (1, 2, 3, 5, 7, 10, 14, 21, 30),
) -> pd.DataFrame:
    """Agrega variables de precipitación rezagada pr_lag_{n} (mm)."""
    df = df.copy()
    if _COL_PR not in df.columns:
        logger.warning("No se pueden calcular lags: columna '%s' ausente.", _COL_PR)
        return df
    for lag in lags:
        df[f"pr_lag_{lag}"] = df[_COL_PR].shift(lag)
    logger.debug("Lags de precipitación agregados: %s", list(lags))
    return df


# ---------------------------------------------------------------------------
# Ventanas móviles (rolling sums)
# ---------------------------------------------------------------------------

def add_rolling_windows(
    df: pd.DataFrame,
    windows: Sequence[int] = (1, 3, 5, 7, 15, 30),
) -> pd.DataFrame:
    """Agrega sumas acumuladas de precipitación pr_sum_{n}d.

    La suma incluye el día actual y los (n-1) días anteriores.
    No genera leakage respecto al target futuro.
    """
    df = df.copy()
    if _COL_PR not in df.columns:
        logger.warning("No se pueden calcular rolling windows: '%s' ausente.", _COL_PR)
        return df
    for w in windows:
        df[f"pr_sum_{w}d"] = df[_COL_PR].rolling(window=w, min_periods=1).sum()
    logger.debug("Rolling windows de precipitación agregados: %s", list(windows))
    return df


# ---------------------------------------------------------------------------
# Antecedent Precipitation Index (API)
# ---------------------------------------------------------------------------

def add_antecedent_precipitation_index(
    df: pd.DataFrame,
    k: float = 0.85,
    windows: Sequence[int] = (7, 15, 30),
) -> pd.DataFrame:
    """Calcula el Antecedent Precipitation Index (API) acumulado.

    API_t = P_t + k * P_{t-1} + k² * P_{t-2} + ... (hasta ventana de n días)

    La ventana limita cuántos días hacia atrás se consideran.

    Parameters
    ----------
    k : float
        Factor de recesión (0 < k < 1). Default 0.85.
    windows : sequence of int
        Horizontes de acumulación (días hacia atrás).
    """
    df = df.copy()
    if _COL_PR not in df.columns:
        logger.warning("API no calculado: columna '%s' ausente.", _COL_PR)
        return df

    pr: np.ndarray = np.asarray(df[_COL_PR].fillna(0.0), dtype=np.float64)
    n = len(pr)

    for w in windows:
        api = np.zeros(n, dtype=np.float64)
        weights: np.ndarray = np.asarray(k ** np.arange(w), dtype=np.float64)
        for i in range(n):
            start = max(0, i - w + 1)
            seg: np.ndarray = np.asarray(pr[start: i + 1][::-1], dtype=np.float64)
            w_seg: np.ndarray = weights[: len(seg)]
            api[i] = float(np.dot(seg, w_seg))
        df[f"api_{w}d"] = api
        logger.debug("API_%dd calculado (k=%.2f).", w, k)

    return df


# ---------------------------------------------------------------------------
# Estacionalidad Fourier
# ---------------------------------------------------------------------------

def add_fourier_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega componentes de Fourier de orden 1 y 2 para capturar estacionalidad anual."""
    df = df.copy()
    if _COL_DATE not in df.columns:
        logger.warning("Fourier no calculado: columna 'date' ausente.")
        return df
    doy: np.ndarray = np.asarray(pd.to_datetime(df[_COL_DATE]).dt.dayofyear, dtype=np.int32)
    sin1, cos1 = doy_fourier(doy, order=1)
    sin2, cos2 = doy_fourier(doy, order=2)
    df["sin_doy_1"] = sin1
    df["cos_doy_1"] = cos1
    df["sin_doy_2"] = sin2
    df["cos_doy_2"] = cos2
    logger.debug("Componentes Fourier de estacionalidad (órdenes 1 y 2) agregados.")
    return df


# ---------------------------------------------------------------------------
# Variables de calendario
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario: DOY, mes, semana, mes hidrológico y fase."""
    df = df.copy()
    if _COL_DATE not in df.columns:
        logger.warning("Calendar features no calculadas: columna 'date' ausente.")
        return df
    dt = pd.to_datetime(df[_COL_DATE])
    df["day_of_year"] = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["hydro_month"] = dt.dt.month.map(hydro_month)
    df["hydro_phase"] = dt.dt.month.map(hydro_phase)
    logger.debug("Calendar features agregadas (DOY, mes, semana, mes hidro, fase hidro).")
    return df


# ---------------------------------------------------------------------------
# Climatología por DOY
# ---------------------------------------------------------------------------

def add_climatology_doy(
    df: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Calcula percentiles climatológicos por DOY usando SOLO el conjunto de entrenamiento.

    Nuevas columnas: clim_pr_p10_doy, clim_pr_p25_doy, clim_pr_p50_doy,
    clim_pr_p75_doy, clim_pr_p90_doy.

    Parameters
    ----------
    train_mask : pd.Series[bool]
        Máscara booleana alineada con el índice de df (filas de entrenamiento).
    """
    df = df.copy()
    if _COL_PR not in df.columns or _COL_DATE not in df.columns:
        logger.warning("Climatología DOY no calculada: faltan columnas necesarias.")
        return df

    df_train = df.loc[train_mask].copy()
    df_train["_doy"] = pd.to_datetime(df_train[_COL_DATE]).dt.dayofyear

    percentiles = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}
    clim: dict[str, pd.Series] = {}
    for pname, q in percentiles.items():
        clim[pname] = df_train.groupby("_doy")[_COL_PR].quantile(q)

    doy_col = pd.to_datetime(df[_COL_DATE]).dt.dayofyear
    for pname in percentiles:
        df[f"clim_pr_{pname}_doy"] = doy_col.map(clim[pname])

    logger.info(
        "Climatología DOY calculada con %d filas de entrenamiento.", int(train_mask.sum())
    )
    return df


# ---------------------------------------------------------------------------
# Variables objetivo (targets)
# ---------------------------------------------------------------------------

def add_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 7),
) -> pd.DataFrame:
    """Crea variables objetivo usando shift negativo (sin leakage).

    - horizonte 1 → pr_next_1d  : precipitación del día siguiente
    - horizonte 3 → pr_sum_next_3d : suma de los próximos 3 días
    - horizonte 7 → pr_sum_next_7d : suma de los próximos 7 días

    Los valores fuera del rango del dataset se convierten en NaN.
    """
    df = df.copy()
    if _COL_PR not in df.columns:
        logger.warning("Targets no calculados: columna '%s' ausente.", _COL_PR)
        return df

    for h in horizons:
        if h == 1:
            df["pr_next_1d"] = df[_COL_PR].shift(-1)
        else:
            # Suma acumulada hacia adelante: sum(pr[t+1], ..., pr[t+h])
            # Calculado como rolling sum sobre la serie invertida
            pr_reversed = df[_COL_PR].iloc[::-1]
            rolling_sum = pr_reversed.rolling(window=h, min_periods=h).sum().iloc[::-1]
            # Shift -h para alinear con el índice actual (el período t toma la suma t+1..t+h)
            df[f"pr_sum_next_{h}d"] = rolling_sum.shift(-h)

    logger.debug("Targets agregados para horizontes: %s", list(horizons))
    return df


# ---------------------------------------------------------------------------
# Categoría de alerta de lluvia
# ---------------------------------------------------------------------------

def add_alert_category(
    df: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Calcula rain_alert_category_next_7d basado en percentiles del train set.

    Categorías:
    - 'seco_bajo'  : precipitación <= p25 del train
    - 'normal'     : (p25, p75]
    - 'humedo'     : (p75, p90]
    - 'extremo'    : > p90

    Requiere que add_targets() haya sido ejecutado (columna pr_sum_next_7d).
    """
    df = df.copy()
    target_col = "pr_sum_next_7d"
    if target_col not in df.columns:
        logger.warning(
            "Columna '%s' no encontrada; ejecutar add_targets() antes.", target_col
        )
        return df

    train_vals = df.loc[train_mask, target_col].dropna()
    p25 = float(train_vals.quantile(0.25))
    p75 = float(train_vals.quantile(0.75))
    p90 = float(train_vals.quantile(0.90))

    logger.info(
        "Percentiles de alerta calculados (solo train): p25=%.2f mm, p75=%.2f mm, p90=%.2f mm",
        p25, p75, p90,
    )

    def _categorize(v: float) -> str:
        if pd.isna(v):
            return float("nan")  # type: ignore[return-value]
        if v <= p25:
            return "seco_bajo"
        if v <= p75:
            return "normal"
        if v <= p90:
            return "humedo"
        return "extremo"

    df["rain_alert_category_next_7d"] = df[target_col].apply(_categorize)
    return df


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def build_full_feature_set(
    df: pd.DataFrame,
    lat: float,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Construye el conjunto completo de features en el orden correcto.

    Orden de ejecución:
    1. normalize_columns
    2. add_temperature_derived
    3. compute_pet_hargreaves
    4. add_calendar_features
    5. add_fourier_seasonality
    6. add_precipitation_lags
    7. add_rolling_windows
    8. add_antecedent_precipitation_index
    9. add_climatology_doy  (solo con train_mask)
    10. add_targets
    11. add_alert_category  (solo con train_mask)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos crudos PISCO.
    lat : float
        Latitud de la cuenca (grados decimales, negativo = sur).
    train_mask : pd.Series[bool]
        Máscara booleana de filas de entrenamiento.

    Returns
    -------
    pd.DataFrame con todas las features y targets.
    """
    logger.info(
        "Iniciando build_full_feature_set: %d filas, lat=%.4f°, train=%d filas",
        len(df), lat, int(train_mask.sum()),
    )

    df = normalize_columns(df)
    df = add_temperature_derived(df)
    df = compute_pet_hargreaves(df, lat=lat)
    df = add_calendar_features(df)
    df = add_fourier_seasonality(df)
    df = add_precipitation_lags(df)
    df = add_rolling_windows(df)
    df = add_antecedent_precipitation_index(df)
    df = add_climatology_doy(df, train_mask=train_mask)
    df = add_targets(df)
    df = add_alert_category(df, train_mask=train_mask)

    logger.info("Feature set construido: %d columnas totales.", len(df.columns))
    return df
