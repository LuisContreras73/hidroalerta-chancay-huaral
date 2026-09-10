"""Métricas de evaluación para modelos de precipitación e hidrología.

Todas las funciones ignoran NaN de forma segura usando sólo observaciones
válidas (pares donde ambos obs y pred son finitos).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
)

logger = logging.getLogger(__name__)


def _clean_pair(
    obs: Any,
    pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve sólo los pares donde ambos valores son finitos (no NaN, no inf)."""
    obs_arr = np.asarray(obs, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        logger.debug("_clean_pair: %d pares con NaN/inf descartados.", n_dropped)
    return obs_arr[mask], pred_arr[mask]


def _require_min(obs: np.ndarray, n: int = 2) -> None:
    if len(obs) < n:
        raise ValueError(
            f"Se necesitan al menos {n} observaciones válidas; sólo hay {len(obs)}."
        )


# ---------------------------------------------------------------------------
# Métricas de regresión individuales
# ---------------------------------------------------------------------------

def compute_mae(obs: Any, pred: Any) -> float:
    """Mean Absolute Error (MAE)."""
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs)
    return float(np.mean(np.abs(obs - pred)))


def compute_rmse(obs: Any, pred: Any) -> float:
    """Root Mean Squared Error (RMSE)."""
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def compute_nse(obs: Any, pred: Any) -> float:
    """Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 - SS_res / SS_tot
    Rango (-inf, 1]. 1 = perfecto; 0 = igual a la media; <0 = peor que la media.
    """
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    if ss_tot == 0.0:
        logger.warning("NSE: varianza observada = 0; devuelve NaN.")
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_kge(obs: Any, pred: Any) -> float:
    """Kling-Gupta Efficiency (KGE) — Gupta et al. (2009).

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    donde r = correlacion Pearson, alpha = std_pred/std_obs, beta = mean_pred/mean_obs.
    Rango (-inf, 1]. 1 = perfecto.
    """
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs, n=3)
    mean_obs = obs.mean()
    mean_pred = pred.mean()
    std_obs = float(obs.std(ddof=1))
    std_pred = float(pred.std(ddof=1))
    if std_obs == 0.0 or mean_obs == 0.0:
        logger.warning("KGE: std_obs o mean_obs = 0; devuelve NaN.")
        return float("nan")
    r, _ = stats.pearsonr(obs, pred)
    alpha = std_pred / std_obs
    beta = mean_pred / mean_obs
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def compute_pbias(obs: Any, pred: Any) -> float:
    """Porcentaje de sesgo (PBIAS).

    PBIAS = 100 * sum(obs - pred) / sum(obs)
    Positivo = subestimacion; negativo = sobreestimacion. Ideal = 0.
    """
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs)
    sum_obs = float(obs.sum())
    if sum_obs == 0.0:
        logger.warning("PBIAS: sum(obs) = 0; devuelve NaN.")
        return float("nan")
    return 100.0 * float((obs - pred).sum()) / sum_obs


def compute_r2(obs: Any, pred: Any) -> float:
    """Coeficiente de determinacion R^2 (sklearn)."""
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs)
    return float(r2_score(obs, pred))


def compute_pearson(obs: Any, pred: Any) -> float:
    """Correlacion de Pearson."""
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs, n=3)
    r, _ = stats.pearsonr(obs, pred)
    return float(r)


def compute_spearman(obs: Any, pred: Any) -> float:
    """Correlacion de Spearman (rango)."""
    obs, pred = _clean_pair(obs, pred)
    _require_min(obs, n=3)
    rho, _ = stats.spearmanr(obs, pred)
    return float(rho)


# ---------------------------------------------------------------------------
# Metricas probabilisticas
# ---------------------------------------------------------------------------

def compute_pinball_loss(obs: Any, pred_quantile: Any, q: float) -> float:
    """Pinball loss (quantile loss) para el cuantil q en (0, 1)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q debe estar en (0, 1); recibido: {q}")
    obs, pred_quantile = _clean_pair(obs, pred_quantile)
    _require_min(obs)
    err = obs - pred_quantile
    loss = np.where(err >= 0, q * err, (q - 1.0) * err)
    return float(loss.mean())


def compute_coverage(obs: Any, pred_low: Any, pred_high: Any) -> float:
    """Fraccion de observaciones dentro del intervalo [pred_low, pred_high]."""
    obs_arr = np.asarray(obs, dtype=np.float64)
    low_arr = np.asarray(pred_low, dtype=np.float64)
    high_arr = np.asarray(pred_high, dtype=np.float64)
    mask = np.isfinite(obs_arr) & np.isfinite(low_arr) & np.isfinite(high_arr)
    obs_arr, low_arr, high_arr = obs_arr[mask], low_arr[mask], high_arr[mask]
    if len(obs_arr) == 0:
        return float("nan")
    inside = (obs_arr >= low_arr) & (obs_arr <= high_arr)
    return float(inside.mean())


def compute_interval_width(pred_low: Any, pred_high: Any) -> float:
    """Ancho medio del intervalo de prediccion [pred_low, pred_high]."""
    low_arr = np.asarray(pred_low, dtype=np.float64)
    high_arr = np.asarray(pred_high, dtype=np.float64)
    mask = np.isfinite(low_arr) & np.isfinite(high_arr)
    widths = (high_arr - low_arr)[mask]
    if len(widths) == 0:
        return float("nan")
    return float(widths.mean())


# ---------------------------------------------------------------------------
# Metricas de clasificacion
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    obs_cat: Any,
    pred_cat: Any,
) -> dict[str, Any]:
    """Metricas para prediccion de categorias de alerta.

    Returns
    -------
    dict con: accuracy, balanced_accuracy, macro_f1, confusion_matrix.
    """
    obs_arr = np.asarray(obs_cat)
    pred_arr = np.asarray(pred_cat)
    mask = pd.notna(obs_arr) & pd.notna(pred_arr)
    obs_arr = obs_arr[mask]
    pred_arr = pred_arr[mask]
    if len(obs_arr) < 2:
        logger.warning("Clasificacion: menos de 2 muestras validas.")
        return {}
    return {
        "accuracy": accuracy_score(obs_arr, pred_arr),
        "balanced_accuracy": balanced_accuracy_score(obs_arr, pred_arr),
        "macro_f1": float(f1_score(obs_arr, pred_arr, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(obs_arr, pred_arr).tolist(),
    }


# ---------------------------------------------------------------------------
# Suite completa
# ---------------------------------------------------------------------------

def compute_all_regression_metrics(
    obs: Any,
    pred: Any,
    pred_p10: Any | None = None,
    pred_p50: Any | None = None,
    pred_p90: Any | None = None,
) -> dict[str, float]:
    """Calcula todas las metricas de regresion y opcionalmente probabilisticas.

    Parameters
    ----------
    obs, pred :
        Observaciones y predicciones puntuales.
    pred_p10, pred_p50, pred_p90 :
        Predicciones de cuantiles p10, p50, p90 (opcionales).

    Returns
    -------
    dict con todas las metricas disponibles.
    """
    result: dict[str, float] = {}
    scalar_metrics: dict[str, Any] = {
        "mae": compute_mae,
        "rmse": compute_rmse,
        "nse": compute_nse,
        "kge": compute_kge,
        "pbias": compute_pbias,
        "r2": compute_r2,
        "pearson": compute_pearson,
        "spearman": compute_spearman,
    }
    for name, fn in scalar_metrics.items():
        try:
            result[name] = fn(obs, pred)
        except Exception as exc:
            logger.warning("Metrica '%s' fallo: %s", name, exc)
            result[name] = float("nan")

    if pred_p10 is not None:
        try:
            result["pinball_p10"] = compute_pinball_loss(obs, pred_p10, q=0.10)
        except Exception as exc:
            logger.warning("pinball_p10 fallo: %s", exc)
            result["pinball_p10"] = float("nan")

    if pred_p50 is not None:
        try:
            result["pinball_p50"] = compute_pinball_loss(obs, pred_p50, q=0.50)
        except Exception as exc:
            logger.warning("pinball_p50 fallo: %s", exc)
            result["pinball_p50"] = float("nan")

    if pred_p90 is not None:
        try:
            result["pinball_p90"] = compute_pinball_loss(obs, pred_p90, q=0.90)
        except Exception as exc:
            logger.warning("pinball_p90 fallo: %s", exc)
            result["pinball_p90"] = float("nan")

    if pred_p10 is not None and pred_p90 is not None:
        try:
            result["coverage_p10_p90"] = compute_coverage(obs, pred_p10, pred_p90)
            result["interval_width"] = compute_interval_width(pred_p10, pred_p90)
        except Exception as exc:
            logger.warning("Metricas de intervalo fallaron: %s", exc)

    logger.debug("Metricas calculadas: %s", list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# Mejora relativa vs baseline
# ---------------------------------------------------------------------------

def improvement_vs_baseline(
    metric_model: float,
    metric_baseline: float,
    lower_is_better: bool = True,
) -> float:
    """Porcentaje de mejora del modelo respecto a un baseline.

    Para metricas donde menor es mejor (MAE, RMSE):
        improvement = (baseline - model) / |baseline| * 100

    Para metricas donde mayor es mejor (NSE, KGE, R2):
        improvement = (model - baseline) / |1 - baseline| * 100

    Returns
    -------
    float : porcentaje de mejora (positivo = modelo es mejor).
    """
    if not np.isfinite(metric_model) or not np.isfinite(metric_baseline):
        return float("nan")
    if lower_is_better:
        denom = abs(metric_baseline) if metric_baseline != 0.0 else 1.0
        return float((metric_baseline - metric_model) / denom * 100.0)
    else:
        denom = abs(1.0 - metric_baseline) if metric_baseline != 1.0 else 1.0
        return float((metric_model - metric_baseline) / denom * 100.0)
