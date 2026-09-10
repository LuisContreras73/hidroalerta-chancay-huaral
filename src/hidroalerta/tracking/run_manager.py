"""Sistema de tracking de corridas experimentales para HidroAlerta."""

import csv
import json
import logging
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import yaml

from hidroalerta.utils.io import ensure_dir, get_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columnas canónicas del leaderboard
# ---------------------------------------------------------------------------
ALL_RUNS_COLUMNS: list[str] = [
    "run_id", "date_time", "model_family", "model_name", "target", "horizon",
    "dataset_version", "feature_set", "train_start", "train_end",
    "val_start", "val_end", "test_start", "test_end", "seed", "n_trials",
    "trial_id", "status",
    "mae", "rmse", "nse", "kge", "pbias", "r2", "pearson", "spearman",
    "pinball_p10", "pinball_p50", "pinball_p90", "coverage_p10_p90",
    "interval_width", "main_metric", "main_metric_value",
    "improvement_vs_persistence_pct", "improvement_vs_climatology_pct",
    "config_path", "figures_path", "model_path", "notes",
]

# Template de model card
_MODEL_CARD_TEMPLATE = """\
# Model Card: {model_name}

## Run ID
{run_id}

## Descripción del modelo
<!-- Breve descripción del modelo y su propósito -->

## Datos
- Dataset version: {dataset_version}
- Target: {target}
- Horizonte: {horizon}
- Train: {train_start} → {train_end}
- Val:   {val_start} → {val_end}
- Test:  {test_start} → {test_end}

## Métricas (test set)
| Métrica | Valor |
|---------|-------|
| MAE     | {mae} |
| RMSE    | {rmse} |
| NSE     | {nse} |
| KGE     | {kge} |
| R²      | {r2} |

## Limitaciones
<!-- Describir limitaciones conocidas del modelo -->

## Uso
<!-- Instrucciones de uso -->
"""


# ---------------------------------------------------------------------------
# Generación de run_id
# ---------------------------------------------------------------------------

def generate_run_id(
    model_name: str,
    target: str,
    horizon: str | int,
    dataset_version: str,
    seed: int,
    dt: datetime | None = None,
) -> str:
    """Genera un run_id determinístico con formato estándar.

    Formato: YYYYMMDD_HHMMSS__<model_name>__<target>__<horizon>__<dataset_version>__seed<seed>
    """
    if dt is None:
        dt = datetime.now(tz=timezone.utc)
    ts = dt.strftime("%Y%m%d_%H%M%S")

    def _slug(s: str) -> str:
        return str(s).lower().replace(" ", "_").replace("-", "_").replace(".", "_")

    run_id = f"{ts}__{_slug(model_name)}__{_slug(target)}__{_slug(str(horizon))}__{_slug(dataset_version)}__seed{seed}"
    logger.debug("run_id generado: %s", run_id)
    return run_id


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------

class RunManager:
    """Gestiona el ciclo de vida de una corrida experimental.

    Parameters
    ----------
    outputs_dir:
        Directorio base de salidas. Si es None se resuelve como
        <project_root>/outputs.
    """

    def __init__(self, outputs_dir: str | Path | None = None) -> None:
        if outputs_dir is None:
            outputs_dir = get_project_root() / "outputs"
        self.outputs_dir = Path(outputs_dir)
        self._run_dir: Path | None = None
        self._run_id: str | None = None
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        run_id: str,
        model_family: str,
        model_name: str,
        config: dict[str, Any],
        params: dict[str, Any] | None = None,
        notes: str = "",
    ) -> Path:
        """Crea el directorio de corrida y persiste config / params iniciales.

        Returns
        -------
        run_dir : Path
            Directorio donde se guardarán todos los artefactos.
        """
        self._run_id = run_id
        self._meta = {
            "run_id": run_id,
            "model_family": model_family,
            "model_name": model_name,
            "date_time": datetime.now(tz=timezone.utc).isoformat(),
            "status": "running",
        }

        run_dir = (
            self.outputs_dir / "runs" / model_family / model_name / run_id
        )
        ensure_dir(run_dir)
        self._run_dir = run_dir
        logger.info("Directorio de corrida creado: %s", run_dir)

        # Guardar config.yaml
        self._save_yaml(config, run_dir / "config.yaml")

        # Guardar params.json
        if params is not None:
            self._save_json(params, run_dir / "params.json")

        # Guardar notes.md
        notes_path = run_dir / "notes.md"
        notes_path.write_text(notes or "# Notas\n\n<!-- Agregar notas aquí -->\n", encoding="utf-8")

        return run_dir

    # ------------------------------------------------------------------
    # Persistencia de artefactos
    # ------------------------------------------------------------------

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Guarda métricas en metrics.json dentro del run_dir."""
        self._assert_setup()
        path = self._run_dir / "metrics.json"
        self._save_json(metrics, path)
        logger.info("Métricas guardadas en %s", path)

    def save_predictions(
        self,
        dates: Any,
        entity_ids: Any,
        observed: Any,
        predicted: Any,
        split: Any,
    ) -> None:
        """Guarda predictions.csv con columnas: date, entity_id, observed, predicted, split."""
        self._assert_setup()
        df = pd.DataFrame(
            {
                "date": dates,
                "entity_id": entity_ids,
                "observed": observed,
                "predicted": predicted,
                "split": split,
            }
        )
        path = self._run_dir / "predictions.csv"
        df.to_csv(path, index=False)
        logger.info("Predictions guardadas: %d filas → %s", len(df), path)

        # Guardar residuals.csv
        df_res = df.copy()
        df_res["residual"] = df_res["observed"] - df_res["predicted"]
        df_res[["date", "entity_id", "split", "residual"]].to_csv(
            self._run_dir / "residuals.csv", index=False
        )
        logger.info("Residuals guardados en %s", self._run_dir / "residuals.csv")

    def save_data_snapshot(self, info: dict[str, Any]) -> None:
        """Guarda data_snapshot_info.json con metadatos del dataset usado."""
        self._assert_setup()
        path = self._run_dir / "data_snapshot_info.json"
        self._save_json(info, path)
        logger.info("Data snapshot guardado en %s", path)

    def save_model_card(self, fields: dict[str, Any]) -> None:
        """Renderiza y guarda model_card.md con los campos proporcionados."""
        self._assert_setup()
        defaults = {k: "N/A" for k in [
            "model_name", "run_id", "dataset_version", "target", "horizon",
            "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
            "mae", "rmse", "nse", "kge", "r2",
        ]}
        defaults.update({k: str(v) for k, v in fields.items()})
        card = _MODEL_CARD_TEMPLATE.format(**defaults)
        path = self._run_dir / "model_card.md"
        path.write_text(card, encoding="utf-8")
        logger.info("Model card guardado en %s", path)

    # ------------------------------------------------------------------
    # Finalización
    # ------------------------------------------------------------------

    def finish(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
        leaderboard_row: dict[str, Any] | None = None,
    ) -> None:
        """Marca la corrida como *status* y actualiza el leaderboard global."""
        self._assert_setup()
        self._meta["status"] = status

        if metrics:
            self.save_metrics(metrics)

        # Actualizar leaderboard
        row = self._build_leaderboard_row(status, metrics or {}, leaderboard_row or {})
        self._append_to_leaderboard(row)
        logger.info("Corrida %s finalizada con estado: %s", self._run_id, status)

    def _build_leaderboard_row(
        self,
        status: str,
        metrics: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {col: "" for col in ALL_RUNS_COLUMNS}
        row.update(self._meta)
        row.update(metrics)
        row.update(extra)
        row["status"] = status
        if self._run_dir:
            row["config_path"] = str(self._run_dir / "config.yaml")
            row["figures_path"] = str(self.outputs_dir / "figures")
        return row

    def _append_to_leaderboard(self, row: dict[str, Any]) -> None:
        lb_dir = self.outputs_dir / "leaderboards"
        ensure_dir(lb_dir)
        lb_path = lb_dir / "all_runs.csv"
        file_exists = lb_path.exists()
        with lb_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ALL_RUNS_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
                logger.info("Leaderboard creado en %s", lb_path)
            writer.writerow(row)
        logger.info("Fila añadida al leaderboard: run_id=%s", row.get("run_id"))

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _assert_setup(self) -> None:
        if self._run_dir is None:
            raise RuntimeError("RunManager.setup() debe llamarse antes de usar este método.")

    @staticmethod
    def _save_yaml(data: dict, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _save_json(data: dict, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Context manager de alto nivel
# ---------------------------------------------------------------------------

@contextmanager
def run_context(
    run_manager: RunManager,
    metrics_fn: Any | None = None,
) -> Generator[RunManager, None, None]:
    """Context manager que captura excepciones y marca la corrida como 'failed'.

    Usage
    -----
    ::

        rm = RunManager()
        rm.setup(run_id, ...)
        with run_context(rm):
            # código del experimento
            ...
            rm.finish("success", metrics=metrics)
    """
    try:
        yield run_manager
    except Exception:
        tb = traceback.format_exc()
        logger.error("Corrida fallida:\n%s", tb)
        if run_manager._run_dir is not None:
            error_path = run_manager._run_dir / "error.txt"
            error_path.write_text(tb, encoding="utf-8")
        run_manager.finish(status="failed")
        raise
