#!/usr/bin/env python3
"""Script 07: Construir features hidrometeorológicas desde Silver a Gold."""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_features")


def main():
    import pandas as pd
    from hidroalerta.features.pisco_features import build_full_feature_set
    from hidroalerta.utils.datetime_utils import temporal_split

    # Cargar silver
    silver_path = PROJECT_ROOT / "data" / "silver" / "S1_pisco_daily_silver.csv"
    if not silver_path.exists():
        bronze_path = PROJECT_ROOT / "data" / "bronze" / "B1_pisco_daily_bronze.csv"
        if bronze_path.exists():
            logger.warning("Silver no encontrado, usando bronze como base")
            df = pd.read_csv(bronze_path, parse_dates=["date"])
        else:
            logger.error("No se encontró silver ni bronze. Correr script 02 primero.")
            sys.exit(1)
    else:
        df = pd.read_csv(silver_path, parse_dates=["date"])

    logger.info(f"Datos cargados: {len(df)} filas, {df['date'].min()} a {df['date'].max()}")

    # Split temporal para calcular climatología solo con train
    train_mask, val_mask, test_mask = temporal_split(df)
    logger.info(f"Split: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # Latitud aproximada de la cuenca Chancay-Huaral
    lat = -11.35

    # Construir features
    df_features = build_full_feature_set(df, lat=lat, train_mask=train_mask)
    df_features["split"] = "train"
    df_features.loc[val_mask, "split"] = "val"
    df_features.loc[test_mask, "split"] = "test"

    # Guardar gold
    gold_dir = PROJECT_ROOT / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "G1_pisco_features_hydro.csv"
    df_features.to_csv(out_path, index=False)
    logger.info(f"Gold guardado: {out_path} ({len(df_features)} filas, {len(df_features.columns)} columnas)")

    # Guardar lista de features
    feature_cols = [c for c in df_features.columns
                    if c not in ["date", "entity_id", "split", "pr_mm", "tmax_c", "tmin_c",
                                  "tmean_c", "pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d",
                                  "rain_alert_category_next_7d"]]
    meta = PROJECT_ROOT / "data" / "metadata" / "gold_features_list.txt"
    meta.write_text("\n".join(feature_cols), encoding="utf-8")
    logger.info(f"Lista de {len(feature_cols)} features guardada en {meta}")


if __name__ == "__main__":
    main()
