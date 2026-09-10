#!/usr/bin/env python3
"""Script 01: Inventariar y documentar datos crudos disponibles."""
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ingest_raw")

KNOWN_EXTENSIONS = {".nc", ".nc4", ".netcdf", ".csv", ".xlsx", ".xls", ".h5", ".tif"}


def scan_directory(base: Path) -> dict:
    inventory = {}
    for source_dir in sorted(base.iterdir()):
        if not source_dir.is_dir():
            continue
        files = []
        for f in sorted(source_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in KNOWN_EXTENSIONS:
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(base)),
                    "size_mb": round(f.stat().st_size / 1e6, 3),
                    "extension": f.suffix.lower(),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        inventory[source_dir.name] = {
            "n_files": len(files),
            "files": files,
            "status": "has_data" if files else "empty"
        }
        if files:
            logger.info(f"  {source_dir.name}: {len(files)} archivos")
        else:
            logger.warning(f"  {source_dir.name}: VACÍO")
    return inventory


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    logger.info(f"Escaneando {raw_dir}")

    inventory = scan_directory(raw_dir)

    n_total = sum(s["n_files"] for s in inventory.values())
    n_empty = sum(1 for s in inventory.values() if s["status"] == "empty")
    logger.info(f"Total: {n_total} archivos en {len(inventory)} fuentes. Vacías: {n_empty}")

    out = {
        "scan_date": datetime.now().isoformat(),
        "raw_dir": str(raw_dir),
        "total_files": n_total,
        "sources": inventory
    }
    out_path = PROJECT_ROOT / "data" / "metadata" / "raw_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Inventario guardado: {out_path}")


if __name__ == "__main__":
    main()
