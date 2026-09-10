#!/usr/bin/env python3
"""
Script 04: Ingestar estaciones SENAMHI (.txt) → capa Silver.

Formato txt: YEAR MONTH DAY PR TMAX TMIN (sep=espacio/tab, -99.9=missing)
Coordenadas cargadas desde configs/station_catalog.yaml.
Confidence level: high=SNIRH exacto, medium=ANA literatura, low=estimado (verificar).

Salidas:
  data/silver/senamhi/S2_{station_id}.csv       -- serie completa por estación
  data/silver/senamhi/S2_senamhi_catalog.csv    -- inventario
  data/silver/senamhi/S2_senamhi_precip_all.csv -- precipitación consolidada
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "04_ingest_senamhi.log", "w", "utf-8"),
    ],
)
logger = logging.getLogger("ingest_senamhi")

RAW_DIR    = PROJECT_ROOT / "data" / "raw" / "senamhi"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "senamhi"
CATALOG_PATH = PROJECT_ROOT / "configs" / "station_catalog.yaml"

MISSING_VALUE = -99.9
MISSING_TOL   = 0.05   # tolerancia para detectar -99.9

# Rangos físicos razonables (QA/QC básico)
QA_BOUNDS = {
    "pr_mm":   (0.0,  200.0),
    "tmax_c":  (-5.0,  35.0),
    "tmin_c":  (-15.0, 25.0),
}


def load_catalog() -> list[dict]:
    """Carga el catálogo de estaciones SENAMHI desde YAML."""
    if not CATALOG_PATH.exists():
        logger.warning(f"Catálogo no encontrado: {CATALOG_PATH}")
        return []
    with open(CATALOG_PATH, encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    return cat.get("meteorologic_senamhi", [])


def read_senamhi_txt(path: Path) -> pd.DataFrame | None:
    """
    Lee un txt SENAMHI con columnas YEAR MONTH DAY PR [TMAX TMIN].
    Retorna DataFrame con índice de fecha y columnas pr_mm, tmax_c, tmin_c.
    """
    try:
        # Intentar con diferentes separadores
        for sep in [r"\s+", "\t", " "]:
            try:
                df = pd.read_csv(path, sep=sep, header=None, engine="python",
                                 na_values=[str(MISSING_VALUE), "-99.9", "99.9"])
                if len(df.columns) >= 4:
                    break
            except Exception:
                continue
        else:
            logger.error(f"  No se pudo leer: {path.name}")
            return None

        df.columns = range(len(df.columns))

        # Columnas: 0=año, 1=mes, 2=día, 3=PR, [4=TMAX, 5=TMIN]
        df = df.rename(columns={0: "year", 1: "month", 2: "day", 3: "pr_mm"})
        if len(df.columns) > 4:
            df = df.rename(columns={4: "tmax_c"})
        if len(df.columns) > 5:
            df = df.rename(columns={5: "tmin_c"})

        # Crear índice de fecha
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        df["day"]   = pd.to_numeric(df["day"],   errors="coerce").astype("Int64")

        # Filtrar filas con fecha inválida
        valid = df[["year", "month", "day"]].notna().all(axis=1)
        df = df[valid].copy()

        dates = pd.to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]},
            errors="coerce"
        )
        df = df[dates.notna()].copy()
        df.index = dates[dates.notna()]
        df.index.name = "date"
        df = df.drop(columns=["year", "month", "day"], errors="ignore")

        # Convertir a numérico y enmascarar MISSING
        for col in ["pr_mm", "tmax_c", "tmin_c"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # -99.9 ya manejado por na_values, pero por seguridad:
                mask = (df[col] < (MISSING_VALUE + MISSING_TOL)) & (df[col] > (MISSING_VALUE - MISSING_TOL))
                df.loc[mask, col] = np.nan

        return df

    except Exception as e:
        logger.error(f"  Error leyendo {path.name}: {e}")
        return None


def apply_qa(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """
    QA/QC básico: clip a rangos físicos y marcar outliers.
    No imputa — solo marca como NaN los valores imposibles.
    """
    df = df.copy()
    for col, (lo, hi) in QA_BOUNDS.items():
        if col in df.columns:
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                logger.warning(f"    QA {station_name} | {col}: {n_out} valores fuera de [{lo},{hi}] → NaN")
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def main():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SCRIPT 04: Ingesta SENAMHI .txt → Silver Layer")
    logger.info(f"Fuente: {RAW_DIR}")
    logger.info(f"Salida: {SILVER_DIR}")
    logger.info("=" * 70)

    catalog = load_catalog()
    cat_by_file = {entry["file"]: entry for entry in catalog}
    logger.info(f"Estaciones en catálogo: {len(catalog)}")

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    logger.info(f"Archivos .txt encontrados: {len(txt_files)}")

    catalog_rows = []
    precip_frames = []

    for txt_path in txt_files:
        logger.info(f"\n  Procesando: {txt_path.name}")

        # Buscar metadata en catálogo
        meta = cat_by_file.get(txt_path.name)
        if meta is None:
            # Intentar coincidencia parcial (case insensitive)
            for fname, m in cat_by_file.items():
                if fname.lower().replace(".txt", "") in txt_path.stem.lower() or \
                   txt_path.stem.lower() in fname.lower():
                    meta = m
                    break

        if meta is None:
            logger.warning(f"    Sin metadata en catálogo para {txt_path.name}. Usando defaults.")
            meta = {
                "id": txt_path.stem.lower().replace(" ", "_"),
                "name": txt_path.stem,
                "lat": None, "lon": None, "alt_m": None,
                "coord_confidence": "unknown",
                "variables": ["pr", "tmax", "tmin"],
            }

        station_id = meta.get("id", txt_path.stem.lower())
        station_name = meta.get("name", txt_path.stem)
        lat  = meta.get("lat")
        lon  = meta.get("lon")
        alt  = meta.get("alt_m")
        conf = meta.get("coord_confidence", "unknown")

        logger.info(f"    Estación: {station_name}")
        logger.info(f"    Coords:   lat={lat} lon={lon} alt={alt}m  [confianza: {conf}]")

        df = read_senamhi_txt(txt_path)
        if df is None or len(df) == 0:
            logger.warning(f"    Sin datos en {txt_path.name}")
            continue

        df = apply_qa(df, station_name)

        # Estadísticas
        cols_present = [c for c in ["pr_mm", "tmax_c", "tmin_c"] if c in df.columns]
        logger.info(f"    Período:  {df.index.min().date()} → {df.index.max().date()} ({len(df)} filas)")
        logger.info(f"    Variables: {cols_present}")
        for col in cols_present:
            n_nan = df[col].isna().sum()
            pct = 100 * n_nan / len(df)
            logger.info(f"      {col}: mean={df[col].mean():.2f}  NaN={n_nan} ({pct:.1f}%)")

        # Agregar columnas de metadata como columnas de referencia
        df["station_id"]   = station_id
        df["station_name"] = station_name
        df["lat"]          = lat
        df["lon"]          = lon
        df["alt_m"]        = alt
        df["coord_conf"]   = conf

        # Guardar CSV individual
        out_name = f"S2_{station_id}.csv"
        out_path = SILVER_DIR / out_name
        df.to_csv(out_path)
        logger.info(f"    -> {out_name}")

        # Acumular precipitación para consolidado
        if "pr_mm" in df.columns:
            pr_s = df["pr_mm"].rename(f"pr_{station_id}")
            precip_frames.append(pr_s.to_frame())

        catalog_rows.append({
            "file":           txt_path.name,
            "station_id":     station_id,
            "station_name":   station_name,
            "lat":            lat,
            "lon":            lon,
            "alt_m":          alt,
            "coord_confidence": conf,
            "variables":      ",".join(cols_present),
            "n_rows":         len(df),
            "date_min":       str(df.index.min().date()),
            "date_max":       str(df.index.max().date()),
            "pct_nan_pr":     round(100 * df["pr_mm"].isna().mean(), 1) if "pr_mm" in df.columns else None,
            "silver_file":    out_name,
        })

    # ── Guardar catálogo ──────────────────────────────────────────────────
    cat_df = pd.DataFrame(catalog_rows)
    cat_df.to_csv(SILVER_DIR / "S2_senamhi_catalog.csv", index=False)
    logger.info(f"\nCatálogo guardado: S2_senamhi_catalog.csv ({len(cat_df)} estaciones)")

    # ── Consolidar precipitación ──────────────────────────────────────────
    if precip_frames:
        pr_all = pd.concat(precip_frames, axis=1).sort_index()
        pr_all.index.name = "date"
        pr_path = SILVER_DIR / "S2_senamhi_precip_all.csv"
        pr_all.to_csv(pr_path)
        logger.info(f"Precipitación consolidada: {pr_path.name}")
        logger.info(f"  Columnas: {list(pr_all.columns)}")
        logger.info(f"  Período:  {pr_all.index.min().date()} → {pr_all.index.max().date()}")

    # ── Tabla resumen ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("ESTACIONES SENAMHI PROCESADAS")
    logger.info(f"{'Estación':<18} {'Lat':>8} {'Lon':>9} {'Alt':>6} {'Conf':<8} {'Variables':<20} {'Período'}")
    logger.info("-" * 90)
    for _, r in cat_df.sort_values("alt_m", ascending=False).iterrows():
        logger.info(f"{r['station_name']:<18} {r['lat']:>8.4f} {r['lon']:>9.4f} "
                    f"{str(r['alt_m']):>6} {r['coord_confidence']:<8} "
                    f"{r['variables']:<20} {r['date_min']} → {r['date_max']}")
    logger.info("=" * 70)
    logger.info("\nIngesta SENAMHI completada.")


if __name__ == "__main__":
    main()
