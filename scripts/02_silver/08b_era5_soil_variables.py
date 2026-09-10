#!/usr/bin/env python3
"""
Script 08b: Descarga ERA5-Land — variables de suelo y escorrentía.

Contexto
--------
El TFT en D6 usa precipitación, temperatura y PET como forzantes pasados.
Los papers de TFT aplicado a caudal (Gauch 2021, Kratzert 2022) muestran
que agregar variables de estado del suelo mejora significativamente la
detección de eventos extremos. Para Chancay-Huaral esto es crítico porque
actualmente CSI=0.00 para eventos ≥20 mm/día.

Variables descargadas (ERA5-Land)
----------------------------------
swvl1     — Soil water layer 1  (0-7 cm)    [m³/m³]
swvl3     — Soil water layer 3  (28-100 cm) [m³/m³]  — saturación profunda
sro       — Surface runoff      (m)          — señal directa de Q generado
ssro      — Sub-surface runoff  (m)          — flujo base
sd        — Snow depth water equivalent (m)  — deshielo diferido en cabecera
e         — Evaporation total   (m)          — pérdida real ET

Período
-------
1981-01-01 a 2020-12-31 (para alinear con D5/D6 train-test)
2021-01-01 a 2026-05 si está disponible (para GR4J / real-time)

Bounding box
------------
lat: [-12.0, -10.5]   lon: [-77.5, -76.3]   (cuenca + buffer 0.25°)

Requisitos
----------
pip install cdsapi
Clave CDS gratuita en: https://cds.climate.copernicus.eu/user/register
Guardar en: ~/.cdsapirc
  url: https://cds.climate.copernicus.eu/api/v2
  key: UID:KEY

Nota: ERA5-Land tiene ~6 GB/década → solo descargar si hay red estable.
Tiempo estimado: 30-60 min por año (depende de la cola CDS).

Salidas
-------
data/bronze/B8_era5land_soil_1981_2020.nc   — variables de suelo históricas
data/bronze/B8_era5land_soil_basin_mean.csv — series de cuenca (área media)
data/bronze/B8_era5land_2021_2026.nc        — extensión reciente (si disponible)

Integración con D6
------------------
Después de ejecutar, agregar al script de construcción D6:
  past_observed += ["swvl1", "swvl3", "sro", "sd"]
Esto debería mejorar CSI para eventos extremos ≥20mm.

Referencias
-----------
Gauch et al. (2021) Rainfall-runoff prediction at multiple timescales.
  doi:10.5194/hess-25-2045-2021
Kratzert et al. (2022) Towards learning universal, regional, and local
  hydrological behaviors. doi:10.5194/hess-26-5483-2022
Muñoz-Sabater et al. (2021) ERA5-Land. doi:10.5194/essd-13-4349-2021
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
OUT_NC  = ROOT / "data/bronze/B8_era5land_soil_1981_2020.nc"
OUT_CSV = ROOT / "data/bronze/B8_era5land_soil_basin_mean.csv"
SHP     = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
           / "Cuenca_Chancay___Huaral.shp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("s08b")

# ── Configuración CDS ──────────────────────────────────────────────────────────

BBOX_CDS = {
    "north": -10.5, "west": -77.5, "south": -12.0, "east": -76.3
}

VARIABLES_ERA5 = [
    "volumetric_soil_water_layer_1",   # swvl1: 0-7 cm
    "volumetric_soil_water_layer_3",   # swvl3: 28-100 cm
    "surface_runoff",                  # sro: m/día
    "sub_surface_runoff",              # ssro: flujo base
    "snow_depth_water_equivalent",     # sd: m
    "total_evaporation",               # e: m/día
]

# Nombres cortos para el CSV de salida
VAR_SHORT = {
    "volumetric_soil_water_layer_1": "swvl1_m3m3",
    "volumetric_soil_water_layer_3": "swvl3_m3m3",
    "surface_runoff":                "sro_mm",
    "sub_surface_runoff":            "ssro_mm",
    "snow_depth_water_equivalent":   "sd_m",
    "total_evaporation":             "e_mm",
}

YEAR_START = 1981
YEAR_END   = 2020


# ── Descarga CDS por año ───────────────────────────────────────────────────────

def download_year(year: int, out_path: Path) -> bool:
    """Descarga un año de ERA5-Land para las variables de suelo."""
    try:
        import cdsapi
    except ImportError:
        log.error("cdsapi no instalado. Ejecutar: pip install cdsapi")
        log.error("Luego registrar clave gratuita en: https://cds.climate.copernicus.eu")
        return False

    if out_path.exists():
        log.info(f"  Ya existe: {out_path.name} — saltando.")
        return True

    log.info(f"  Descargando ERA5-Land {year} ...")
    try:
        c = cdsapi.Client(quiet=True)
        c.retrieve(
            "reanalysis-era5-land",
            {
                "product_type": "reanalysis",
                "variable": VARIABLES_ERA5,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day":   [f"{d:02d}" for d in range(1, 32)],
                "time":  ["00:00", "06:00", "12:00", "18:00"],
                "area":  [BBOX_CDS["north"], BBOX_CDS["west"],
                          BBOX_CDS["south"], BBOX_CDS["east"]],
                "format": "netcdf",
            },
            str(out_path),
        )
        log.info(f"  Descargado: {out_path.name} ({out_path.stat().st_size/1e6:.0f} MB)")
        return True
    except Exception as e:
        log.error(f"  Error en descarga {year}: {e}")
        return False


# ── Extracción media de cuenca ─────────────────────────────────────────────────

def extract_basin_mean(nc_path: Path) -> pd.DataFrame:
    """
    Lee un NetCDF ERA5-Land, aplica máscara cuenca Chancay-Huaral,
    calcula media ponderada (coseno-latitud) y retorna DataFrame diario.
    """
    import netCDF4 as nc4
    import geopandas as gpd
    from shapely.geometry import Point

    log.info(f"  Extrayendo media cuenca de {nc_path.name} ...")
    ds = nc4.Dataset(nc_path)

    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]

    # Máscara cuenca
    basin = gpd.read_file(SHP).to_crs("EPSG:4326")
    poly  = basin.geometry.union_all()
    mask  = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            mask[i, j] = poly.contains(Point(float(lo), float(la)))
    cos_w   = np.cos(np.radians(lats.astype(float)))
    weights = np.where(mask, cos_w[:, None], 0.0)
    w_sum   = weights.sum()

    if w_sum == 0:
        log.error("Ningún pixel dentro de la cuenca — verificar bbox o shapefile")
        ds.close()
        return pd.DataFrame()

    # Tiempo
    time_var  = ds.variables["time"]
    try:
        import cftime
        times = nc4.num2date(time_var[:], time_var.units, calendar="gregorian")
    except Exception:
        times = nc4.num2date(time_var[:], time_var.units)

    # ERA5-Land es hourly → agregar a diario (sum para flujos, mean para estado)
    dates_all = np.array([pd.Timestamp(str(t)[:10]) for t in times])
    dates_uniq = pd.date_range(dates_all[0], dates_all[-1], freq="D")

    records = []
    for date in dates_uniq:
        idx = np.where(dates_all == date)[0]
        if len(idx) == 0:
            records.append({"date": date, **{v: np.nan for v in VAR_SHORT.values()}})
            continue

        row = {"date": date}
        for nc_var, short in VAR_SHORT.items():
            vnames = [k for k in ds.variables if nc_var.replace("_", "")[:6].lower()
                      in k.lower() or nc_var[:4].lower() in k.lower()]
            # Mapeo simplificado a nombres ERA5-Land
            nc_map = {
                "swvl1_m3m3": "swvl1", "swvl3_m3m3": "swvl3",
                "sro_mm": "sro", "ssro_mm": "ssro",
                "sd_m": "sd", "e_mm": "e",
            }
            nc_key = nc_map.get(short, short[:4])
            if nc_key not in ds.variables:
                row[short] = np.nan
                continue
            data_t = ds.variables[nc_key][idx]   # (n_steps, nlat, nlon)
            # Convertir acumulaciones: sumar los 4 pasos (ERA5: m/step → m/día)
            if short in ("sro_mm", "ssro_mm", "e_mm"):
                data_d = np.nansum(data_t, axis=0) * 1000  # m → mm
            else:
                data_d = np.nanmean(data_t, axis=0)        # estado: media diaria
            basin_val = float(np.nansum(data_d * weights) / w_sum)
            row[short] = round(basin_val, 6)
        records.append(row)

    ds.close()
    return pd.DataFrame(records).set_index("date")


# ── Pipeline completo ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("SCRIPT 08b: ERA5-Land soil variables — HidroAlerta Chancay-Huaral")
    log.info("=" * 70)

    # Verificar cdsapi
    try:
        import cdsapi
        log.info("  cdsapi disponible.")
    except ImportError:
        log.error("cdsapi no instalado.")
        log.error("  pip install cdsapi")
        log.error("  Registrar clave gratuita: https://cds.climate.copernicus.eu")
        log.error("  Guardar ~/.cdsapirc con url y key.")
        sys.exit(1)

    tmp_dir = ROOT / "data/raw/_tmp_era5_soil"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Descarga por año
    nc_files = []
    for year in range(YEAR_START, YEAR_END + 1):
        nc_path = tmp_dir / f"era5land_soil_{year}.nc"
        ok = download_year(year, nc_path)
        if ok and nc_path.exists():
            nc_files.append(nc_path)

    if not nc_files:
        log.error("No se descargó ningún archivo.")
        sys.exit(1)

    # Extraer media de cuenca para cada año
    all_dfs = []
    for nc_path in nc_files:
        df_y = extract_basin_mean(nc_path)
        if not df_y.empty:
            all_dfs.append(df_y)

    if not all_dfs:
        log.error("No se pudo extraer ninguna serie.")
        sys.exit(1)

    df_all = pd.concat(all_dfs).sort_index()

    # Convertir unidades finales
    # swvl: ya en m³/m³ (adimensional, no cambiar)
    # sro, ssro: ya convertido a mm/día arriba
    # sd: m (dejar como m — magnitudes pequeñas, más interpretable)
    # e: mm/día (convertido arriba, pero ERA5 da negativo para evaporación)
    if "e_mm" in df_all.columns:
        df_all["e_mm"] = -df_all["e_mm"]  # ERA5 convención: negativo hacia abajo → positivo

    df_all.index.name = "date"
    df_all = df_all.reset_index()
    df_all.to_csv(OUT_CSV, index=False)
    log.info(f"\nSerie de cuenca guardada: {OUT_CSV}")
    log.info(f"  Período: {df_all['date'].min()} a {df_all['date'].max()}")
    log.info(f"  Filas: {len(df_all)}")
    log.info(f"  Columnas: {list(df_all.columns)}")

    # Preview de estadísticas
    print("\nEstadísticas descriptivas (media de cuenca):")
    print(df_all.drop(columns="date").describe().round(4).to_string())

    log.info("\n" + "="*70)
    log.info("PRÓXIMOS PASOS:")
    log.info("  1. Incorporar estas variables a D6_multientity.csv (Script 20)")
    log.info("  2. Agregar swvl1/swvl3/sro/sd a schema past_observed")
    log.info("  3. Re-entrenar TFT con D6_v2 (Script 22 + sweep Script 27)")
    log.info("  4. Comparar NSE/CSI antes y después en Script 25")
    log.info("="*70)


if __name__ == "__main__":
    main()
