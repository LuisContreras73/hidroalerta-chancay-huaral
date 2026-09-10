"""
Tracking completo para corridas de Deep Learning (TFT, LSTM).

Genera y mantiene config.json con trazabilidad completa:
  run / model / data / hyperparams / evaluation / results / artifacts

Uso típico en Script 22/32:
    from hidroalerta.tracking.dl_run import DLRun

    run = DLRun.create(
        runs_dir   = RUNS_DIR,
        exp_id     = "E01",
        exp_notes  = "MAELoss + patience=20",
        architecture = "TFT",
        target     = "pr_sum_next_7d",
        hyperparams = {...},
        data_config = {...},
    )
    # ... entrenamiento ...
    run.save_results(pred_df, best_val_loss=0.72, best_epoch=45)
    run.register_plot("VTFT01_attention.png")
    run.register_checkpoint("checkpoints/best-epoch=45.ckpt")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Thresholds usados para CSI (mm/7d para target semanal, mm/d para diario)
CSI_THRESHOLDS = [1, 5, 10, 20]

# Temporadas hidrológicas cuenca Chancay-Huaral
WET_MONTHS  = {11, 12, 1, 2, 3, 4}   # noviembre-abril
DRY_MONTHS  = {5, 6, 7, 8, 9, 10}    # mayo-octubre


# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def _nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom < 1e-10:
        return float("nan")
    return float(1 - np.sum((obs - pred) ** 2) / denom)


def _kge(obs: np.ndarray, pred: np.ndarray) -> float:
    if obs.std() < 1e-10 or pred.std() < 1e-10 or obs.mean() < 1e-10:
        return float("nan")
    r     = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(pred.std() / obs.std())
    beta  = float(pred.mean() / obs.mean())
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def _pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    total_obs = np.sum(obs)
    if abs(total_obs) < 1e-10:
        return float("nan")
    return float(100 * np.sum(pred - obs) / total_obs)


def _csi(obs: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    hits   = int(np.sum((obs >= threshold) & (pred >= threshold)))
    misses = int(np.sum((obs >= threshold) & (pred < threshold)))
    fa     = int(np.sum((obs < threshold) & (pred >= threshold)))
    denom  = hits + misses + fa
    return round(hits / denom, 4) if denom > 0 else float("nan")


def _metrics_block(obs: np.ndarray, pred: np.ndarray, thresholds: list[int] = CSI_THRESHOLDS) -> dict:
    obs, pred = np.array(obs, dtype=float), np.array(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    block: dict[str, Any] = {
        "n":     int(len(obs)),
        "nse":   round(_nse(obs, pred),   4),
        "kge":   round(_kge(obs, pred),   4),
        "mae":   round(float(np.mean(np.abs(obs - pred))), 4),
        "rmse":  round(_rmse(obs, pred),  4),
        "pbias": round(_pbias(obs, pred), 2),
    }
    for t in thresholds:
        block[f"csi_{t}mm"] = _csi(obs, pred, t)
    return block


def compute_full_metrics(
    pred_df: pd.DataFrame,
    obs_col: str = "obs",
    pred_col: str = "q50",
    entity_col: str = "entity_id",
    date_col: str = "date",
    thresholds: list[int] = CSI_THRESHOLDS,
) -> dict:
    """
    Computa métricas en 4 niveles desde un DataFrame de predicciones.

    Parameters
    ----------
    pred_df : DataFrame con columnas date, entity_id, obs, q50 (al menos)
    obs_col / pred_col : nombres de columnas de observado y predicho
    thresholds : umbrales en mm para CSI

    Returns
    -------
    dict con claves: global, per_entity, per_season, rainfall_thresholds_mm
    """
    df = pred_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_month"] = df[date_col].dt.month
    df["_season"] = df["_month"].apply(
        lambda m: "wet" if m in WET_MONTHS else "dry"
    )

    obs  = df[obs_col].values
    pred = df[pred_col].values

    results: dict[str, Any] = {
        "rainfall_thresholds_mm": thresholds,
        "global":     _metrics_block(obs, pred, thresholds),
        "per_entity": {},
        "per_season": {},
    }

    # Por entidad
    for eid, grp in df.groupby(entity_col):
        results["per_entity"][str(eid)] = _metrics_block(
            grp[obs_col].values, grp[pred_col].values, thresholds
        )

    # Por temporada
    for season in ("wet", "dry"):
        grp = df[df["_season"] == season]
        results["per_season"][season] = _metrics_block(
            grp[obs_col].values, grp[pred_col].values, thresholds
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# DLRun — objeto principal de trazabilidad
# ──────────────────────────────────────────────────────────────────────────────

class DLRun:
    """Gestiona config.json de trazabilidad completa para un run DL.

    Crea y mantiene la siguiente estructura en ``run_dir/config.json``:

    .. code-block:: json

        {
          "run":         { run_id, exp_id, exp_notes, timestamp, script },
          "model":       { architecture, family, n_params, library },
          "data":        { dataset_version, entities, target, normalizer,
                           train/val/test periods, past/future/static vars },
          "hyperparams": { encoder_length, hidden_size, loss_function, ... },
          "evaluation":  { protocol, rainfall_thresholds_mm, ... },
          "results":     { global, per_entity, per_season, per_horizon },
          "artifacts":   { config, training_log, predictions, checkpoint, plots, ... }
        }
    """

    CONFIG_FILE = "config.json"

    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = Path(run_dir)
        self._cfg   = config

    # ------------------------------------------------------------------
    # Constructor principal
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        runs_dir: Path,
        exp_id: str,
        exp_notes: str,
        architecture: str,            # "TFT" | "LSTM"
        target: str,
        hyperparams: dict,
        data_config: dict,
        script: str = "",
        n_params: int | None = None,
    ) -> "DLRun":
        """Crea directorio de run, persiste config.json inicial y devuelve DLRun."""
        ts       = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id   = f"{ts}_{exp_id}_{target}"
        run_dir  = Path(runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "attention").mkdir(exist_ok=True)

        cfg: dict[str, Any] = {
            "run": {
                "run_id":    run_id,
                "exp_id":    exp_id,
                "exp_notes": exp_notes,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "script":    script,
            },
            "model": {
                "architecture": architecture,
                "family":       "deep_learning",
                "full_name":    _ARCH_NAMES.get(architecture, architecture),
                "n_params":     n_params,
                "library":      "pytorch_forecasting",
            },
            "data": {
                **data_config,
                "target": target,
            },
            "hyperparams": hyperparams,
            "evaluation": {
                "protocol": "step0_fair",
                "protocol_description": (
                    "Evaluación en paso 0 del decoder con contexto val+test "
                    "para warmup del encoder (sin data leakage)"
                ),
                "rainfall_thresholds_mm": CSI_THRESHOLDS,
                "metric_primary":   "nse",
                "metric_secondary": "kge",
                "obs_column_in_predictions": "obs",
                "pred_column_in_predictions": "q50",
            },
            "results": {
                "global":     {},
                "per_entity": {},
                "per_season": {"wet": {}, "dry": {}},
                "per_horizon": {},
            },
            "artifacts": {
                "config":           cls.CONFIG_FILE,
                "training_log":     "training.log",
                "metrics_per_epoch": "metrics_per_epoch.csv",
                "predictions":      f"predictions/test_predictions_{target}.csv",
                "checkpoint":       None,
                "plots":            [],
                "attention_vsn":    f"attention/vsn_weights_{target}.csv",
                "attention_encoder": f"attention/encoder_attention_{target}.csv",
            },
        }

        obj = cls(run_dir, cfg)
        obj._persist()
        log.info("DLRun creado: %s", run_id)
        return obj

    # ------------------------------------------------------------------
    # Actualización de resultados
    # ------------------------------------------------------------------

    def save_results(
        self,
        pred_df: pd.DataFrame,
        best_val_loss: float | None = None,
        best_epoch: int | None = None,
        obs_col: str = "obs",
        pred_col: str = "q50",
    ) -> dict:
        """
        Computa métricas completas desde pred_df y persiste en config.json.

        pred_df debe tener: date, entity_id, obs, q50 (al menos).
        """
        metrics = compute_full_metrics(pred_df, obs_col=obs_col, pred_col=pred_col)

        if best_val_loss is not None:
            metrics["global"]["best_val_loss"] = round(float(best_val_loss), 6)
        if best_epoch is not None:
            metrics["global"]["best_epoch"] = int(best_epoch)

        self._cfg["results"] = metrics
        self._persist()
        log.info(
            "Resultados guardados — NSE=%.3f  KGE=%.3f  MAE=%.3f  n=%d",
            metrics["global"].get("nse", float("nan")),
            metrics["global"].get("kge", float("nan")),
            metrics["global"].get("mae", float("nan")),
            metrics["global"].get("n", 0),
        )
        return metrics

    def register_plot(self, filename: str) -> None:
        """Agrega un plot al manifest de artefactos."""
        plots = self._cfg["artifacts"].setdefault("plots", [])
        if filename not in plots:
            plots.append(filename)
        self._persist()

    def register_checkpoint(self, ckpt_path: str) -> None:
        """Registra la ruta del checkpoint en artifacts."""
        self._cfg["artifacts"]["checkpoint"] = ckpt_path
        self._persist()

    def set_n_params(self, n: int) -> None:
        self._cfg["model"]["n_params"] = int(n)
        self._persist()

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._cfg["run"]["run_id"]

    @property
    def dir(self) -> Path:
        return self.run_dir

    @property
    def config(self) -> dict:
        return self._cfg

    # ------------------------------------------------------------------
    # Carga desde directorio existente
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Path) -> "DLRun":
        cfg_path = Path(run_dir) / cls.CONFIG_FILE
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(run_dir, cfg)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        cfg_path = self.run_dir / self.CONFIG_FILE
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self._cfg, f, indent=2, ensure_ascii=False, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ARCH_NAMES = {
    "TFT":         "TemporalFusionTransformer",
    "LSTM":        "RecurrentNetwork_LSTM",
    "GRU":         "RecurrentNetwork_GRU",
    "Transformer": "VanillaTransformer_EncoderDecoder",
}


def build_data_config(
    dataset_version: str,
    dataset_file: str,
    entities: list[str],
    train_period: tuple[str, str],
    val_period:   tuple[str, str],
    test_period:  tuple[str, str],
    target_normalizer: str,
    past_vars:    list[str],
    future_vars:  list[str],
    static_vars:  list[str],
    target_description: str = "",
) -> dict:
    """Constructor de la sección 'data' del config."""
    def _period(start: str, end: str) -> dict:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        return {"start": start, "end": end, "n_days": int((e - s).days + 1)}

    return {
        "dataset_version":    dataset_version,
        "dataset_file":       dataset_file,
        "entities":           entities,
        "n_entities":         len(entities),
        "target_description": target_description,
        "target_normalizer":  target_normalizer,
        "train_period":       _period(*train_period),
        "val_period":         _period(*val_period),
        "test_period":        _period(*test_period),
        "past_vars":          past_vars,
        "future_vars":        future_vars,
        "static_vars":        static_vars,
    }


def build_tft_hyperparams(
    encoder_length: int,
    prediction_length: int,
    hidden_size: int,
    lstm_layers: int,
    num_attention_heads: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    loss_function: str,
    quantiles: list[float],
    add_relative_time_idx: bool = False,
    gradient_clip_val: float | None = None,
    lr_scheduler: str | None = None,
) -> dict:
    """Constructor de la sección 'hyperparams' para TFT."""
    return {
        "encoder_length":          encoder_length,
        "prediction_length":       prediction_length,
        "hidden_size":             hidden_size,
        "lstm_layers":             lstm_layers,
        "num_attention_heads":     num_attention_heads,
        "dropout":                 dropout,
        "learning_rate":           learning_rate,
        "lr_scheduler":            lr_scheduler,
        "gradient_clip_val":       gradient_clip_val,
        "batch_size":              batch_size,
        "max_epochs":              max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "loss_function":           loss_function,
        "quantiles":               quantiles,
        "add_relative_time_idx":   add_relative_time_idx,
    }


def build_transformer_hyperparams(
    encoder_length: int,
    prediction_length: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    loss_function: str,
    gradient_clip_val: float | None = 0.1,
    lr_scheduler: str | None = "ReduceLROnPlateau",
) -> dict:
    """Constructor de la sección 'hyperparams' para VanillaTransformer."""
    return {
        "encoder_length":          encoder_length,
        "prediction_length":       prediction_length,
        "d_model":                 d_model,
        "nhead":                   nhead,
        "num_encoder_layers":      num_encoder_layers,
        "num_decoder_layers":      num_decoder_layers,
        "dim_feedforward":         dim_feedforward,
        "dropout":                 dropout,
        "learning_rate":           learning_rate,
        "lr_scheduler":            lr_scheduler,
        "gradient_clip_val":       gradient_clip_val,
        "batch_size":              batch_size,
        "max_epochs":              max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "loss_function":           loss_function,
        "norm_first":              True,   # Pre-LayerNorm
        "static_injection":        "additive_bias",
    }


def build_lstm_hyperparams(
    encoder_length: int,
    prediction_length: int,
    hidden_size: int,
    rnn_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    loss_function: str,
    quantiles: list[float],
    gradient_clip_val: float | None = None,
    lr_scheduler: str | None = None,
) -> dict:
    """Constructor de la sección 'hyperparams' para LSTM."""
    return {
        "encoder_length":          encoder_length,
        "prediction_length":       prediction_length,
        "hidden_size":             hidden_size,
        "rnn_layers":              rnn_layers,
        "dropout":                 dropout,
        "learning_rate":           learning_rate,
        "lr_scheduler":            lr_scheduler,
        "gradient_clip_val":       gradient_clip_val,
        "batch_size":              batch_size,
        "max_epochs":              max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "loss_function":           loss_function,
        "quantiles":               quantiles,
    }
