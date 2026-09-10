"""Funciones de limpieza y control de calidad de series temporales hidrometeorológicas."""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detección de outliers
# ---------------------------------------------------------------------------

def detect_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Detecta outliers usando el rango intercuartílico (IQR).

    Un valor es outlier si cae fuera de [Q1 - k*IQR, Q3 + k*IQR].

    Parameters
    ----------
    series : pd.Series
        Serie numérica.
    k : float
        Multiplicador del IQR (default 3.0 para series de precipitación).

    Returns
    -------
    pd.Series[bool] : True donde el valor es outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    n_out = int(mask.sum())
    logger.debug(
        "detect_outliers_iqr('%s'): %d outliers detectados (k=%.1f, IQR=%.3f).",
        series.name, n_out, k, iqr,
    )
    return mask


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Detecta outliers usando el Z-score modificado (mediana ± threshold * MAD).

    Más robusto que el Z-score clásico para distribuciones asimétricas
    como la precipitación.

    Parameters
    ----------
    series : pd.Series
        Serie numérica.
    threshold : float
        Umbral del Z-score modificado (default 3.5).

    Returns
    -------
    pd.Series[bool] : True donde el valor es outlier.
    """
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0.0:
        # Fallback a std si MAD es cero (serie constante)
        std = series.std()
        if std == 0.0:
            logger.warning(
                "detect_outliers_zscore('%s'): MAD y std = 0; sin outliers.", series.name
            )
            return pd.Series(False, index=series.index)
        z = (series - series.mean()).abs() / std
    else:
        # Z-score modificado de Iglewicz & Hoaglin (1993)
        z = 0.6745 * (series - med).abs() / mad

    mask = z > threshold
    logger.debug(
        "detect_outliers_zscore('%s'): %d outliers (umbral=%.1f).",
        series.name, int(mask.sum()), threshold,
    )
    return mask


# ---------------------------------------------------------------------------
# Flags de calidad
# ---------------------------------------------------------------------------

def flag_missing(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Agrega columna 'quality_score' (0–1) basada en la fracción de valores presentes.

    quality_score = fracción de columnas especificadas que NO son NaN en esa fila.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    columns : sequence of str
        Columnas a incluir en el cálculo del quality_score.

    Returns
    -------
    pd.DataFrame con columna 'quality_score' añadida.
    """
    df = df.copy()
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        logger.warning("flag_missing: ninguna de las columnas %s está en el DataFrame.", columns)
        df["quality_score"] = float("nan")
        return df

    df["quality_score"] = df[valid_cols].notna().mean(axis=1)
    logger.debug(
        "flag_missing: quality_score calculado sobre %d columnas. Media=%.3f.",
        len(valid_cols), df["quality_score"].mean(),
    )
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega flags de calidad estándar al DataFrame.

    Flags añadidos (bool):
    - flag_missing_pr   : precipitación (pr_mm) es NaN
    - flag_missing_tmax : temperatura máxima (tmax_c) es NaN
    - flag_missing_tmin : temperatura mínima (tmin_c) es NaN
    - flag_outlier_pr   : precipitación es outlier IQR (k=5)
    - flag_negative_pr  : precipitación < 0
    - flag_tmax_lt_tmin : tmax_c < tmin_c (inconsistencia física)

    Returns
    -------
    pd.DataFrame con columnas flag_* añadidas.
    """
    df = df.copy()

    if "pr_mm" in df.columns:
        df["flag_missing_pr"] = df["pr_mm"].isna()
        df["flag_negative_pr"] = df["pr_mm"].fillna(0.0) < 0.0
        df["flag_outlier_pr"] = detect_outliers_iqr(df["pr_mm"].fillna(0.0), k=5.0)
    if "tmax_c" in df.columns:
        df["flag_missing_tmax"] = df["tmax_c"].isna()
    if "tmin_c" in df.columns:
        df["flag_missing_tmin"] = df["tmin_c"].isna()
    if "tmax_c" in df.columns and "tmin_c" in df.columns:
        df["flag_tmax_lt_tmin"] = df["tmax_c"] < df["tmin_c"]

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    logger.info("add_quality_flags: %d columnas de flag añadidas.", len(flag_cols))
    return df


# ---------------------------------------------------------------------------
# Rango físico
# ---------------------------------------------------------------------------

def clip_physical_range(
    df: pd.DataFrame,
    variable: str,
    min_val: float,
    max_val: float,
) -> pd.DataFrame:
    """Recorta los valores de *variable* al rango físico [min_val, max_val].

    Los valores fuera de rango se reemplazan por NaN (en lugar de clipear)
    para que queden marcados como faltantes y puedan ser interpolados.

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
        Nombre de la columna a limpiar.
    min_val, max_val : float
        Límites físicos válidos.

    Returns
    -------
    pd.DataFrame con valores fuera de rango → NaN.
    """
    df = df.copy()
    if variable not in df.columns:
        logger.warning("clip_physical_range: columna '%s' no encontrada.", variable)
        return df

    out_of_range = (df[variable] < min_val) | (df[variable] > max_val)
    n_out = int(out_of_range.sum())
    if n_out > 0:
        df.loc[out_of_range, variable] = np.nan
        logger.info(
            "clip_physical_range('%s'): %d valores fuera de [%.2f, %.2f] → NaN.",
            variable, n_out, min_val, max_val,
        )
    return df


# ---------------------------------------------------------------------------
# Interpolación de brechas cortas
# ---------------------------------------------------------------------------

def interpolate_short_gaps(
    df: pd.DataFrame,
    column: str,
    max_gap_days: int = 3,
) -> pd.DataFrame:
    """Interpola linealmente brechas de NaN de hasta *max_gap_days* días consecutivos.

    Brechas más largas se dejan como NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con índice DatetimeIndex o columna 'date' ordenada.
    column : str
        Columna a interpolar.
    max_gap_days : int
        Número máximo de días consecutivos a interpolar (default 3).

    Returns
    -------
    pd.DataFrame con NaN cortos interpolados.
    """
    df = df.copy()
    if column not in df.columns:
        logger.warning("interpolate_short_gaps: columna '%s' no encontrada.", column)
        return df

    n_before = int(df[column].isna().sum())
    df[column] = df[column].interpolate(
        method="linear",
        limit=max_gap_days,
        limit_direction="forward",
    )
    n_after = int(df[column].isna().sum())
    logger.info(
        "interpolate_short_gaps('%s'): %d NaN → %d NaN (max_gap=%d días).",
        column, n_before, n_after, max_gap_days,
    )
    return df
