"""Utilidades temporales: meses hidrológicos, splits y fourier estacional."""

import logging
import math
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calendario hidrológico andino (año hidrológico: oct–sep)
# ---------------------------------------------------------------------------

_HYDRO_MONTH_MAP: dict[int, int] = {
    10: 1, 11: 2, 12: 3,
    1: 4, 2: 5, 3: 6,
    4: 7, 5: 8, 6: 9,
    7: 10, 8: 11, 9: 12,
}

# Fases para la cuenca Chancay-Huaral (costa/sierra central del Perú)
# húmedo: dic-mar  |  seco: jun-sep  |  transición inicio: oct-nov  |  transición fin: abr-may
_HYDRO_PHASE_MAP: dict[int, str] = {
    1: "dry",
    2: "dry",
    3: "transition_end",
    4: "transition_end",
    5: "dry",
    6: "dry",
    7: "dry",
    8: "dry",
    9: "dry",
    10: "transition_onset",
    11: "transition_onset",
    12: "wet",
}


def hydro_month(month: int) -> int:
    """Convierte mes calendario (1–12) a mes hidrológico andino (oct=1, sep=12)."""
    if month not in _HYDRO_MONTH_MAP:
        raise ValueError(f"Mes inválido: {month}. Debe ser 1–12.")
    return _HYDRO_MONTH_MAP[month]


def hydro_phase(month: int) -> str:
    """Devuelve la fase hidrológica andina para un mes calendario.

    Fases: 'wet', 'dry', 'transition_onset', 'transition_end'.
    """
    if month not in _HYDRO_PHASE_MAP:
        raise ValueError(f"Mes inválido: {month}. Debe ser 1–12.")
    return _HYDRO_PHASE_MAP[month]


# ---------------------------------------------------------------------------
# Split temporal cronológico
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Divide el DataFrame en masks booleanas (train, val, test) de forma cronológica.

    El test ocupa la fracción restante: 1 - train_ratio - val_ratio.

    Parameters
    ----------
    df:
        DataFrame con índice DatetimeIndex o columna 'date'.
    train_ratio:
        Fracción de datos para entrenamiento (default 0.70).
    val_ratio:
        Fracción de datos para validación (default 0.15).

    Returns
    -------
    train_mask, val_mask, test_mask : pd.Series de bool
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio debe ser < 1.0")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    idx = np.arange(n)
    train_mask = pd.Series(idx < n_train, index=df.index, name="train_mask")
    val_mask = pd.Series((idx >= n_train) & (idx < n_train + n_val), index=df.index, name="val_mask")
    test_mask = pd.Series(idx >= n_train + n_val, index=df.index, name="test_mask")

    logger.info(
        "Split temporal — train: %d filas, val: %d filas, test: %d filas",
        train_mask.sum(), val_mask.sum(), test_mask.sum(),
    )
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Fourier estacional
# ---------------------------------------------------------------------------

def doy_fourier(doy: np.ndarray | int | Sequence, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula componentes de Fourier para el día del año (doy).

    Parameters
    ----------
    doy:
        Día del año (1–366).
    order:
        Armónico (1 = anual, 2 = semianual, ...).

    Returns
    -------
    sin_val, cos_val : np.ndarray
    """
    doy = np.asarray(doy, dtype=float)
    angle = 2.0 * math.pi * order * doy / 365.25
    return np.sin(angle), np.cos(angle)


# ---------------------------------------------------------------------------
# Radiación extraterrestre (FAO-56) para PET Hargreaves
# ---------------------------------------------------------------------------

def compute_ra_hargreaves(doy: np.ndarray | int | Sequence, lat_deg: float) -> np.ndarray:
    """Calcula la radiación extraterrestre Ra (mm/día equivalentes) según FAO-56.

    Ra se expresa en MJ m⁻² d⁻¹ y luego se convierte a mm/día
    usando el factor 0.408 (= 1/λ donde λ ≈ 2.45 MJ kg⁻¹).

    Parameters
    ----------
    doy:
        Día del año (1–366).
    lat_deg:
        Latitud en grados decimales (negativo = hemisferio sur).

    Returns
    -------
    Ra_mm_day : np.ndarray
        Radiación extraterrestre en mm/día equivalentes.

    Notes
    -----
    Fórmulas de Allen et al. (1998), FAO Irrigation and Drainage Paper 56.
    La aproximación asume días "medios" por DOY y es suficiente para
    el método Hargreaves-Samani a escala diaria.
    """
    doy = np.asarray(doy, dtype=float)
    lat_rad = math.radians(lat_deg)

    # Distancia relativa Tierra-Sol (ecuación FAO-56 eq. 23)
    dr = 1.0 + 0.033 * np.cos(2.0 * math.pi * doy / 365.0)

    # Declinación solar (eq. 24)
    delta = 0.409 * np.sin(2.0 * math.pi * doy / 365.0 - 1.39)

    # Ángulo horario de puesta de sol (eq. 25)
    ws = np.arccos(-math.tan(lat_rad) * np.tan(delta))

    # Constante solar Gsc = 0.0820 MJ m⁻² min⁻¹
    Gsc = 0.0820

    # Ra en MJ m⁻² d⁻¹ (eq. 21)
    Ra_mj = (
        (24.0 * 60.0 / math.pi)
        * Gsc
        * dr
        * (ws * math.sin(lat_rad) * np.sin(delta) + math.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )

    # Convertir a mm/día (factor FAO: Ra_mm = 0.408 * Ra_MJ)
    Ra_mm = 0.408 * Ra_mj
    return Ra_mm
