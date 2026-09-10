#!/usr/bin/env python3
"""Script 08: Preparar dataset model-ready desde Gold."""
import logging
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_model_ready")

TARGET_COLS = ["pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d",
               "rain_alert_category_next_7d"]
ID_COLS = ["date", "entity_id", "split"]
EXCLUDE_FROM_FEATURES = set(TARGET_COLS + ID_COLS + ["pr_mm", "tmax_c", "tmin_c", "tmean_c"])


def main():
    import pandas as pd

    gold_path = PROJECT_ROOT / "data" / "gold" / "G1_pisco_features_hydro.csv"
    if not gold_path.exists():
        logger.error("Gold no encontrado. Correr script 07 primero.")
        sys.exit(1)

    df = pd.read_csv(gold_path, parse_dates=["date"])
    logger.info(f"Gold cargado: {len(df)} filas, {len(df.columns)} columnas")

    # Quitar filas sin ningún target
    targets_present = [t for t in TARGET_COLS if t in df.columns]
    df_valid = df.dropna(subset=targets_present[:1])  # al menos el primer target
    logger.info(f"Filas con target válido: {len(df_valid)} (de {len(df)})")

    # Guardar model_ready completo
    mr_dir = PROJECT_ROOT / "data" / "model_ready"
    mr_dir.mkdir(parents=True, exist_ok=True)
    out_path = mr_dir / "D4_pisco_model_ready_precip.csv"
    df_valid.to_csv(out_path, index=False)
    logger.info(f"Model-ready guardado: {out_path}")

    # Guardar schema del dataset
    feature_cols = [c for c in df_valid.columns if c not in EXCLUDE_FROM_FEATURES]
    schema = {
        "dataset_name": "D4_pisco_model_ready_precip",
        "dataset_version": "D4_pisco_model_ready_precip_v001",
        "created": pd.Timestamp.now().isoformat(),
        "n_rows": len(df_valid),
        "n_feature_cols": len(feature_cols),
        "n_target_cols": len(targets_present),
        "feature_cols": feature_cols,
        "target_cols": targets_present,
        "id_cols": [c for c in ID_COLS if c in df_valid.columns],
        "splits": df_valid["split"].value_counts().to_dict() if "split" in df_valid.columns else {},
        "date_range": {
            "start": str(df_valid["date"].min()),
            "end": str(df_valid["date"].max())
        },
        "missing_rates": df_valid[targets_present].isnull().mean().to_dict(),
        "source": "PISCO (pr, tmax, tmin) + feature engineering",
        "note": "Dataset preliminar para pronóstico pluviométrico. NO contiene caudal observado."
    }
    schema_path = PROJECT_ROOT / "data" / "metadata" / "D4_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    logger.info(f"Schema guardado: {schema_path}")
    logger.info(f"Features: {len(feature_cols)}, Targets: {len(targets_present)}")


if __name__ == "__main__":
    main()
