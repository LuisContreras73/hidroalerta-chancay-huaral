#!/usr/bin/env python3
"""
Script 03: Ingestar reportes SNIRH/ANA → capa Silver.

Lee todos los xlsx de data/raw/ana_snirh/, extrae metadata de estación
y datos de la tabla, y guarda una serie por variable/estación en
data/silver/snirh/.

Salidas:
  data/silver/snirh/S1_{station_id}_{variable}.csv
  data/silver/snirh/S1_snirh_catalog.csv   -- inventario de todos los archivos
  data/silver/snirh/S1_snirh_daily_q.csv   -- caudal diario consolidado (todas estaciones)
"""
import logging
import re
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
        logging.FileHandler(PROJECT_ROOT / "outputs" / "03_ingest_snirh.log", "w", "utf-8"),
    ],
)
logger = logging.getLogger("ingest_snirh")

RAW_DIR    = PROJECT_ROOT / "data" / "raw" / "ana_snirh"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "snirh"
CATALOG_PATH = PROJECT_ROOT / "configs" / "station_catalog.yaml"

# Variables que nos importan para el proyecto
TARGET_VARS = {
    "Caudal Promedio Diario":    ("caudal_diario",   "m3/s", "1D"),
    "Nivel Promedio 1 Día":      ("nivel_diario",    "m",    "1D"),
    "Precipitación Acumulada 1 Día": ("precip_diaria", "mm", "1D"),
    # Horarias (guardamos pero no son el foco principal)
    "Caudal Promedio Horario":   ("caudal_horario",  "m3/s", "1H"),
    "Caudal Instantáneo 1 Hora": ("caudal_instante", "m3/s", "1H"),
    "Nivel Promedio 1 Hora":     ("nivel_horario",   "m",    "1H"),
}

MISSING_VALUES = [-9999, -999, -99.9, -9.9, 9999, 999]


def parse_snirh_excel(path: Path) -> dict | None:
    """
    Parsea un xlsx SNIRH con cabecera de metadata (filas 1-12)
    y tabla de datos a partir de la fila 14.
    Retorna dict con metadata + DataFrame de datos, o None si error.
    """
    try:
        raw = pd.read_excel(path, header=None, dtype=str)
    except Exception as e:
        logger.error(f"  No se pudo leer {path.name}: {e}")
        return None

    meta = {}

    # ── Extraer metadata de filas 1-9 ──────────────────────────────────────
    for i in range(min(12, len(raw))):
        row_vals = [str(v).strip() for v in raw.iloc[i] if str(v).strip() not in ("", "nan")]
        if len(row_vals) >= 2:
            key = row_vals[0].rstrip(":")
            val = " ".join(row_vals[1:])
            if "Estación" in key or "Estacion" in key:
                # "Santa Cruz (Código: 155202)"
                m = re.match(r"(.+?)\s*\(C[oó]digo:\s*([^\)]+)\)", val)
                if m:
                    meta["station_name"] = m.group(1).strip()
                    meta["station_code"] = m.group(2).strip()
                else:
                    meta["station_name"] = val
                    meta["station_code"] = None
            elif "Variable" in key:
                meta["variable_raw"] = val
            elif "WGS" in key or "Geogr" in key:
                # "Latitud: -11.3836 / Longitud: -77.0503 / Altitud(msnm): 614"
                lat_m = re.search(r"Latitud:\s*([-\d.]+)", val)
                lon_m = re.search(r"Longitud:\s*([-\d.]+)", val)
                alt_m = re.search(r"Altitud.*?:\s*([\d.]+)", val)
                meta["lat"]   = float(lat_m.group(1)) if lat_m else None
                meta["lon"]   = float(lon_m.group(1)) if lon_m else None
                meta["alt_m"] = float(alt_m.group(1)) if (alt_m and alt_m.group(1)) else None
            elif "Tipo" in key:
                meta["station_type"] = val
            elif "Unidad Hidro" in key:
                meta["cuenca"] = val
            elif "Nombre de la Fuente" in key:
                meta["fuente"] = val.replace(":", "").strip()

    # ── Identificar fila de encabezado de datos (FECHA, HORA, VALOR) ────────
    header_row = None
    for i in range(len(raw)):
        row_str = " ".join(str(v) for v in raw.iloc[i])
        if "FECHA" in row_str.upper() and "VALOR" in row_str.upper():
            header_row = i
            break

    if header_row is None:
        logger.warning(f"  {path.name}: no se encontró fila de datos")
        return None

    # ── Extraer datos directamente del DataFrame ya leído ───────────────────
    # Usar la fila header_row como nombres de columna y las siguientes como datos
    # (evitamos re-leer para no tener problemas con header+skiprows en pandas)
    col_names = [str(v).strip() for v in raw.iloc[header_row]]
    df_data = raw.iloc[header_row + 1:].copy()
    df_data.columns = col_names
    df_data = df_data.reset_index(drop=True)

    # Buscar columnas FECHA y VALOR
    fecha_col = next((c for c in df_data.columns if "FECHA" in c.upper()), None)
    valor_col = next((c for c in df_data.columns if "VALOR" in c.upper()), None)
    hora_col  = next((c for c in df_data.columns if "HORA" in c.upper()), None)

    if fecha_col is None or valor_col is None:
        logger.warning(f"  {path.name}: columnas FECHA/VALOR no encontradas")
        return None

    df_data = df_data[[c for c in [fecha_col, hora_col, valor_col] if c]].copy()
    df_data.columns = ["fecha", "hora", "valor"][:len(df_data.columns)]

    # Parsear fechas
    df_data["fecha"] = pd.to_datetime(df_data["fecha"], dayfirst=True, errors="coerce")
    df_data = df_data.dropna(subset=["fecha"])

    # Parsear valores numéricos
    df_data["valor"] = pd.to_numeric(df_data["valor"], errors="coerce")
    # Enmascarar valores faltantes
    for mv in MISSING_VALUES:
        df_data.loc[df_data["valor"] == mv, "valor"] = np.nan

    # Resolver variable
    var_raw = meta.get("variable_raw", "")
    var_info = None
    for key, info in TARGET_VARS.items():
        if key.lower() in var_raw.lower():
            var_info = info
            break
    if var_info is None:
        # Fallback: parsear del texto
        var_info = ("variable_desconocida", "?", "?")
        logger.warning(f"  Variable no reconocida: '{var_raw}' en {path.name}")

    meta["var_name"] = var_info[0]
    meta["var_units"] = var_info[1]
    meta["var_freq"] = var_info[2]
    meta["n_rows"] = len(df_data)
    meta["date_min"] = str(df_data["fecha"].min().date()) if len(df_data) > 0 else None
    meta["date_max"] = str(df_data["fecha"].max().date()) if len(df_data) > 0 else None
    meta["source_file"] = path.name

    # Si es horaria, crear columna datetime completo
    if "hora" in df_data.columns:
        try:
            df_data["datetime"] = pd.to_datetime(
                df_data["fecha"].astype(str) + " " + df_data["hora"].astype(str),
                errors="coerce"
            )
        except Exception:
            df_data["datetime"] = df_data["fecha"]

    return {"meta": meta, "data": df_data}


def make_station_id(station_name: str, code: str | None) -> str:
    """Crea un id limpio para la estación."""
    name = station_name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if code:
        code_clean = re.sub(r"[^a-z0-9]+", "", code.lower())
        return f"{name}_{code_clean}"
    return name


def main():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SCRIPT 03: Ingesta SNIRH/ANA → Silver Layer")
    logger.info(f"Fuente: {RAW_DIR}")
    logger.info(f"Salida: {SILVER_DIR}")
    logger.info("=" * 70)

    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    logger.info(f"Archivos xlsx encontrados: {len(xlsx_files)}")

    catalog_rows = []
    # Colectores por (station_id, var_name, freq)
    series_daily: dict[str, list] = {}

    for xlsx_path in xlsx_files:
        logger.info(f"\n  Procesando: {xlsx_path.name}")
        result = parse_snirh_excel(xlsx_path)
        if result is None:
            continue

        meta = result["meta"]
        df   = result["data"]

        station_id = make_station_id(
            meta.get("station_name", "unknown"),
            meta.get("station_code")
        )
        var_name = meta["var_name"]
        freq     = meta["var_freq"]
        key      = f"{station_id}__{var_name}"

        logger.info(f"    Estación: {meta.get('station_name')} ({meta.get('station_code')})")
        logger.info(f"    Variable: {var_name} [{meta['var_units']}] freq={freq}")
        logger.info(f"    Filas: {meta['n_rows']} | {meta['date_min']} → {meta['date_max']}")
        logger.info(f"    Coords: lat={meta.get('lat')} lon={meta.get('lon')} alt={meta.get('alt_m')}m")

        # ── Guardar CSV individual ────────────────────────────────────────
        out_name = f"S1_{station_id}__{var_name}.csv"
        out_path = SILVER_DIR / out_name

        if freq == "1D":
            df_save = df[["fecha", "valor"]].rename(columns={"fecha": "date", "valor": var_name})
            df_save = df_save.sort_values("date").drop_duplicates("date")
            df_save = df_save.set_index("date")
        else:
            col_dt = "datetime" if "datetime" in df.columns else "fecha"
            df_save = df[[col_dt, "valor"]].rename(columns={col_dt: "datetime", "valor": var_name})
            df_save = df_save.sort_values("datetime").drop_duplicates("datetime")
            df_save = df_save.set_index("datetime")

        df_save.to_csv(out_path)
        logger.info(f"    -> Guardado: {out_name}")

        # Acumular caudal diario para consolidado
        if var_name == "caudal_diario":
            col_q = f"q_{station_id}"
            if key not in series_daily:
                series_daily[key] = []
            series_daily[key].append(df_save.rename(columns={var_name: col_q}))

        # ── Registrar en catálogo ─────────────────────────────────────────
        catalog_rows.append({
            "file":         xlsx_path.name,
            "station_id":   station_id,
            "station_name": meta.get("station_name"),
            "station_code": meta.get("station_code"),
            "station_type": meta.get("station_type"),
            "lat":          meta.get("lat"),
            "lon":          meta.get("lon"),
            "alt_m":        meta.get("alt_m"),
            "cuenca":       meta.get("cuenca"),
            "var_name":     var_name,
            "var_units":    meta["var_units"],
            "var_freq":     freq,
            "n_rows":       meta["n_rows"],
            "date_min":     meta["date_min"],
            "date_max":     meta["date_max"],
            "silver_file":  out_name,
        })

    # ── Guardar catálogo ──────────────────────────────────────────────────
    cat_df = pd.DataFrame(catalog_rows)
    cat_path = SILVER_DIR / "S1_snirh_catalog.csv"
    cat_df.to_csv(cat_path, index=False)
    logger.info(f"\nCatálogo guardado: {cat_path.name} ({len(cat_df)} entradas)")

    # ── Consolidar caudal diario ──────────────────────────────────────────
    if series_daily:
        daily_frames = []
        for key, frames in series_daily.items():
            combined = pd.concat(frames).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            daily_frames.append(combined)

        q_all = pd.concat(daily_frames, axis=1).sort_index()
        q_all.index.name = "date"
        q_path = SILVER_DIR / "S1_snirh_daily_q.csv"
        q_all.to_csv(q_path)
        logger.info(f"Caudal diario consolidado: {q_path.name}")
        logger.info(f"  Columnas: {list(q_all.columns)}")
        logger.info(f"  Período:  {q_all.index.min()} a {q_all.index.max()}")
        logger.info(f"  Filas:    {len(q_all)}")
        logger.info("\nEstadísticas de caudal (m³/s):")
        logger.info(q_all.describe().to_string())
    else:
        logger.warning("No se encontraron series de caudal diario.")

    # ── Resumen por estación ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN DE ESTACIONES")
    logger.info("=" * 70)
    if len(cat_df) > 0:
        for stn in cat_df["station_name"].unique():
            sub = cat_df[cat_df["station_name"] == stn]
            coords = f"lat={sub['lat'].iloc[0]:.4f} lon={sub['lon'].iloc[0]:.4f} alt={sub['alt_m'].iloc[0]}m"
            logger.info(f"\n  {stn}  [{coords}]")
            for _, row in sub.iterrows():
                logger.info(f"    {row['var_name']:25s} {row['var_freq']:3s}  "
                            f"{row['date_min']} → {row['date_max']}  ({row['n_rows']} filas)")

    logger.info("\nIngesta SNIRH completada.")


if __name__ == "__main__":
    main()
