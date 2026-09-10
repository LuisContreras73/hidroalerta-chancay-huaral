"""Utilidades de entrada/salida: YAML, JSON, directorios y rutas del proyecto."""

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """Carga un archivo YAML y devuelve un diccionario."""
    path = Path(path)
    logger.debug("Cargando YAML desde %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    logger.debug("YAML cargado correctamente desde %s", path)
    return data or {}


def save_yaml(data: dict, path: str | Path) -> None:
    """Serializa *data* a un archivo YAML."""
    path = Path(path)
    ensure_dir(path.parent)
    logger.debug("Guardando YAML en %s", path)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.debug("YAML guardado en %s", path)


def load_json(path: str | Path) -> dict:
    """Carga un archivo JSON y devuelve un diccionario."""
    path = Path(path)
    logger.debug("Cargando JSON desde %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.debug("JSON cargado correctamente desde %s", path)
    return data


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """Serializa *data* a un archivo JSON con indentación."""
    path = Path(path)
    ensure_dir(path.parent)
    logger.debug("Guardando JSON en %s", path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
    logger.debug("JSON guardado en %s", path)


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio *path* (y padres) si no existe. Devuelve el Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto buscando pyproject.toml / setup.cfg hacia arriba."""
    current = Path(__file__).resolve()
    markers = {"pyproject.toml", "setup.cfg", "setup.py", ".git"}
    for parent in current.parents:
        if any((parent / m).exists() for m in markers):
            logger.debug("Raíz del proyecto detectada en %s", parent)
            return parent
    # Fallback: tres niveles arriba de este archivo (src/hidroalerta/utils/)
    fallback = current.parents[3]
    logger.warning(
        "No se encontró marcador de proyecto; usando fallback %s", fallback
    )
    return fallback


def get_run_dir(
    base: str | Path,
    model_family: str,
    model_name: str,
    run_id: str,
) -> Path:
    """Construye y crea la carpeta de salida para una corrida.

    Estructura: <base>/runs/<model_family>/<model_name>/<run_id>/
    """
    run_dir = Path(base) / "runs" / model_family / model_name / run_id
    ensure_dir(run_dir)
    logger.info("Directorio de corrida: %s", run_dir)
    return run_dir
