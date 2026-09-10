#!/usr/bin/env python3
"""
Script 29: Índices espectrales MODIS por sub-cuenca — HidroAlerta Chancay-Huaral.

Contexto hidrológico
---------------------
La cuenca Chancay-Huaral abarca un gradiente altitudinal extremo (521-4507 m).
Los índices espectrales capturan procesos que los forzantes atmosféricos no ven:
  • Cobertura de nieve (NDSI)   → input diferido de deshielo en cabecera (>3800m)
  • Estado vegetación (NDVI/EVI)→ intercepción y ET real vs PET estimada
  • Humedad suelo/vegetación    → condición antecedente para runoff
  • LST                         → balance energético, confirma presencia de nieve

Índices producidos (fuente: MODIS Terra/Aqua 2000-2020)
-------------------------------------------------------
ndsi            — Normalized Difference Snow Index               MOD10A1 500m diario
snow_cover_pct  — % píxeles con nieve por sub-cuenca            MOD10A1
ndvi            — Normalized Difference Vegetation Index         MOD13Q1 250m 16d
evi             — Enhanced Vegetation Index (menos saturación)   MOD13Q1 250m 16d
ndwi_veg        — NDW Vegetation (NIR-SWIR1) agua en dosel      MOD09GA 500m diario
lswi            — Land Surface Water Index = NDWI_veg            MOD09GA (alias)
lst_day_c       — LST diurna (°C)                               MOD11A1 1km diario
lst_night_c     — LST nocturna (°C)                             MOD11A1 1km diario

Período efectivo: 2000-02-24 → 2020-12-31 (MODIS).
Para 1981-1999: completar con climatología mensual (lag 0 = sin dato).

Relevancia por zona altitudinal
--------------------------------
Zona costera   sub_634  (521 m):  NDVI, EVI (vegetación estacional costera)
Zona transición sub_640-sub_653 (1480-1664 m): NDVI, NDWI, LST
Zona andina    sub_650-sub_641 (2841-3470 m): NDVI, NDWI, LSWI, LST
Zona puna      sub_646-sub_655 (3812-4058 m): NDSI, snow_cover, LST
Cabecera       sub_656-sub_649 (4466-4507 m): NDSI, snow_cover (CRÍTICO), LST

Método de descarga
------------------
MODO 1 — Google Earth Engine (recomendado, gratis):
  pip install earthengine-api
  earthengine authenticate
  Luego: python scripts/29_spectral_indices.py --mode gee

MODO 2 — NASA AppEEARS REST API (alternativa, requiere cuenta EARTHDATA gratuita):
  Registrar en: https://urs.earthdata.nasa.gov
  Guardar usuario/contraseña en configs/earthdata_credentials.yaml
  Luego: python scripts/29_spectral_indices.py --mode appeears

MODO 3 — Simulación con climatología (sin red, para pruebas):
  python scripts/29_spectral_indices.py --mode simulate

Salidas
-------
data/silver/S4_spectral_indices.csv  — serie 2000-2020, una fila por (date, entity_id)
data/silver/S4_spectral_clim.csv     — climatología mensual por (entity_id, month)
outputs/figures/spectral/VS01_spectral_overview.png
outputs/figures/spectral/VS02_snow_cover_timeseries.png

Integración con D6
------------------
Después de ejecutar, agregar a past_observed en D6 schema:
  ["ndsi", "snow_cover_pct", "ndvi", "evi", "lswi", "lst_day_c"]

Para 1981-1999 (antes de MODIS): usar S4_spectral_clim.csv para imputar
  por (entity_id, month) → permite usar TFT con imputation en past_observed.

Impacto esperado en métricas
-----------------------------
  NDSI en sub_649-sub_656 → mejora NSE 7d en período post-deshielo (AGO-SEP)
  NDWI/LSWI en sub_640-sub_650 → mejora detección eventos extremos ≥10mm
  LST_day → mejora PET implícita + detección de ondas de calor

Referencias
-----------
Hall et al.   (2002) MODIS Snow Cover. doi:10.1016/S0034-4257(02)00091-0
Didan (2021)  MOD13Q1 NDVI/EVI v061. doi:10.5067/MODIS/MOD13Q1.061
Wan et al.    (2021) MOD11A1 LST v061. doi:10.5067/MODIS/MOD11A1.061
Gao (1996)    NDWI. doi:10.1016/S0034-4257(96)00067-3
Xiao et al.   (2004) LSWI. doi:10.1016/j.rse.2003.11.008
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
import sys as _sys; _sys.path.insert(0, str(ROOT / "data/metadata"))
from entity_labels import ENTITY_ELEVATIONS, ENTITIES_ASC, entity_label
SHP_SUB = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas"
           / "Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_geogpsperu_UH_MENORES.shp")
OUT_DIR = ROOT / "data/silver"
OUT_CSV = OUT_DIR / "S4_spectral_indices.csv"
OUT_CLIM= OUT_DIR / "S4_spectral_clim.csv"
FIG_DIR = ROOT / "outputs/figures/spectral"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("s29")

SPECTRAL_COLS = [
    "ndsi", "snow_cover_pct", "ndvi", "evi", "lswi", "lst_day_c", "lst_night_c"
]

DATE_START = "2000-02-24"
DATE_END   = "2025-12-31"


# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — Google Earth Engine
# ══════════════════════════════════════════════════════════════════════════════

GEE_SCRIPT_TEMPLATE = '''
// Script GEE equivalente (ejecutar en https://code.earthengine.google.com)
// Para verificar resultados o exportar manualmente

var subcuencas = ee.FeatureCollection('users/YOUR_USER/chancay_subcuencas');
var startDate = '{start}';
var endDate   = '{end}';

// 1. NDSI y snow_cover desde MOD10A1
var snow = ee.ImageCollection('MODIS/061/MOD10A1')
  .filterDate(startDate, endDate)
  .select(['NDSI_Snow_Cover', 'NDSI']);

var snowStats = snow.map(function(img) {{
  var ndsi = img.select('NDSI').multiply(0.0001);
  var snow_pct = img.select('NDSI_Snow_Cover').gt(40).rename('snow_cover_pct');
  return img.date().format('YYYY-MM-dd')
    .map(function(date) {{
      return subcuencas.map(function(feat) {{
        var ndsi_mean = ndsi.reduceRegion({{reducer: ee.Reducer.mean(), geometry: feat.geometry(), scale: 500}});
        var snow_mean = snow_pct.reduceRegion({{reducer: ee.Reducer.mean(), geometry: feat.geometry(), scale: 500}});
        return feat.set({{date: date, ndsi: ndsi_mean.get('NDSI'), snow_cover_pct: snow_mean.get('NDSI_Snow_Cover')}});
      }});
    }});
}});

// 2. NDVI, EVI desde MOD13Q1 (16-day)
var veg = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(startDate, endDate)
  .select(['NDVI', 'EVI']);

// 3. LST desde MOD11A1
var lst = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate(startDate, endDate)
  .select(['LST_Day_1km', 'LST_Night_1km'])
  .map(function(img) {{
    return img.multiply(0.02).subtract(273.15);  // Kelvin * 0.02 - 273.15 = Celsius
  }});

// Export
Export.table.toDrive({{collection: snowStats.flatten(), description: 'ndsi_chancay', fileFormat: 'CSV'}});
'''


def run_gee(start: str, end: str) -> pd.DataFrame:
    """Descarga índices espectrales a NetCDF (rasters) usando xee."""
    try:
        import ee
        import geemap
    except ImportError:
        log.error("Librerías faltantes: pip install earthengine-api geemap")
        return pd.DataFrame()

    try:
        # xee optimiza mejor con highvolume endpoint
        ee.Initialize(project='ana-chancay-huaral', opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        log.error(f"GEE no autenticado: {e}")
        return pd.DataFrame()

    import geopandas as gpd
    from shapely.geometry import mapping
    import sys

    out_dir = ROOT / "data/silver/rasters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / "S4_spectral_monthly.nc"

    log.info("Cargando shapefile de límite de cuenca entera ...")
    shp_path = ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite/Cuenca_Chancay___Huaral.shp"
    gdf_cue = gpd.read_file(shp_path).to_crs("EPSG:4326")
    geom = ee.Geometry(mapping(gdf_cue.geometry.iloc[0]))

    months = pd.date_range(start, end, freq='MS')
    
    def get_monthly(date):
        s = ee.Date(date)
        e = s.advance(1, 'month')

        # ── Índices MODIS ───────────────────────────────────────────────
        ndsi = (ee.ImageCollection("MODIS/061/MOD10A1")
                .filterDate(s, e).select("NDSI_Snow_Cover")  # banda correcta del producto
                .median().multiply(0.01).rename("ndsi"))      # escala: 0-100 → 0-1
        ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(s, e).select("NDVI")
                .median().multiply(0.0001).rename("ndvi"))
        lst = (ee.ImageCollection("MODIS/061/MOD11A1")
               .filterDate(s, e).select("LST_Day_1km")
               .median().multiply(0.02).subtract(273.15).rename("lst_day_c"))  # Kelvin → °C

        # MOD09GA: reflectancias de superficie (escala: multiplicar por 0.0001)
        # Bandas: B01=Red(620-670), B02=NIR(841-876), B03=Blue(459-479), B04=Green(545-565)
        #         B06=SWIR1(1628-1652), B07=SWIR2(2105-2155)
        sr = (ee.ImageCollection("MODIS/061/MOD09GA")
              .filterDate(s, e)
              .select(["sur_refl_b01","sur_refl_b02","sur_refl_b03",
                       "sur_refl_b04","sur_refl_b06","sur_refl_b07"])
              .median().multiply(0.0001))

        # Bandas crudas renombradas (para imagen visual y cálculos propios)
        red  = sr.select("sur_refl_b01").rename("red")    # 620-670nm
        nir  = sr.select("sur_refl_b02").rename("nir")    # 841-876nm
        blue = sr.select("sur_refl_b03").rename("blue")   # 459-479nm
        green= sr.select("sur_refl_b04").rename("green")  # 545-565nm
        swir1= sr.select("sur_refl_b06").rename("swir1")  # 1628-1652nm
        swir2= sr.select("sur_refl_b07").rename("swir2")  # 2105-2155nm

        # LSWI calculado con bandas correctas
        lswi = nir.subtract(swir1).divide(nir.add(swir1).add(1e-9)).rename("lswi")

        # Combinar: índices + bandas crudas para imagen natural
        # Composición color natural MODIS: red=B01, green=B04, blue=B03
        # Composición falsa-color (vegetación): NIR=B02, red=B01, green=B04
        img = ee.Image([ndsi, ndvi, lst, lswi, red, green, blue, nir, swir1, swir2])\
                .set('system:time_start', s.millis())\
                .set('system:index', ee.String('spectral_').cat(s.format('YYYY_MM')))
        return img.clip(geom)

    log.info("Construyendo ImageCollection mensual en GEE ...")
    ee_dates = ee.List([d.strftime('%Y-%m-%d') for d in months])
    col = ee.ImageCollection.fromImages(ee_dates.map(get_monthly))

    log.info("Lanzando tareas de exportación a Google Drive ...")
    log.info("  Carpeta Drive: HidroAlerta_Rasters/S29_spectral/")
    log.info("  Bandas por imagen: NDSI, NDVI, LST, LSWI + RED, GREEN, BLUE, NIR, SWIR1, SWIR2 (10 total)")

    # Iterate and export
    n_total = len(months)
    for i, date_str in enumerate(ee_dates.getInfo()):
        s = ee.Date(date_str)
        e = s.advance(1, 'month')
        year_mo = f"{date_str[:4]}_{date_str[5:7]}"
        img = col.filterDate(s, e).first()

        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"S29_spectral_{year_mo}",
            folder="HidroAlerta_Rasters/S29_spectral",
            fileNamePrefix=f"S29_spectral_{year_mo}",
            scale=500,
            region=geom,
            crs='EPSG:4326',
            maxPixels=1e10,
            fileFormat='GeoTIFF',
        )
        task.start()
        if (i + 1) % 30 == 0 or i == n_total - 1:
            log.info(f"  [{i+1}/{n_total}] Enviadas hasta {date_str[:7]}")

    log.info(f"¡{n_total} tareas de Exportación a Drive enviadas!")

    # Retornamos un DataFrame dummy con la marca para saltar el procesamiento de CSV
    return pd.DataFrame([{"mode": "raster_netcdf"}])



# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — NASA AppEEARS REST API
# ══════════════════════════════════════════════════════════════════════════════

APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"

APPEEARS_LAYERS = [
    # (product, layer, short_name)
    ("MOD10A1.061",  "NDSI_Snow_Cover",  "snow_cover_pct"),
    ("MOD10A1.061",  "NDSI",             "ndsi_raw"),
    ("MOD13Q1.061",  "250m_16_days_NDVI","ndvi"),
    ("MOD13Q1.061",  "250m_16_days_EVI", "evi"),
    ("MOD11A1.061",  "LST_Day_1km",      "lst_day_k"),
    ("MOD11A1.061",  "LST_Night_1km",    "lst_night_k"),
]


def run_appeears(start: str, end: str, username: str, password: str) -> pd.DataFrame:
    """
    Submete un task AppEEARS para las sub-cuencas y descarga el resultado.
    AppEEARS procesa en la nube — puede tardar 10-60 min según la cola.
    """
    import requests as rq
    import json, time

    log.info("Autenticando en NASA AppEEARS ...")
    token_r = rq.post(
        f"{APPEEARS_URL}/login",
        auth=(username, password),
        timeout=30,
    )
    if token_r.status_code != 200:
        log.error(f"AppEEARS login fallido: {token_r.status_code}")
        return pd.DataFrame()
    token = token_r.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Cargar shapefile como GeoJSON para AppEEARS
    import geopandas as gpd
    gdf = gpd.read_file(SHP_SUB).to_crs("EPSG:4326")
    geojson = json.loads(gdf.to_json())

    layers = [{"layer": lyr, "product": prod} for prod, lyr, _ in APPEEARS_LAYERS]

    task_payload = {
        "task_type": "area",
        "task_name": "chancay_huaral_spectral",
        "params": {
            "dates": [{"startDate": start, "endDate": end}],
            "layers": layers,
            "output": {"format": {"type": "netcdf4"}, "projection": "geographic"},
            "geo": geojson,
        }
    }

    log.info("Enviando task a AppEEARS ...")
    task_r = rq.post(f"{APPEEARS_URL}/task", json=task_payload, headers=headers, timeout=60)
    if task_r.status_code != 202:
        log.error(f"Task submission fallida: {task_r.status_code} — {task_r.text[:200]}")
        return pd.DataFrame()
    task_id = task_r.json()["task_id"]
    log.info(f"Task enviado: {task_id}")
    log.info("Esperando procesamiento AppEEARS (puede tardar 10-60 min) ...")

    while True:
        status_r = rq.get(f"{APPEEARS_URL}/status/{task_id}", headers=headers, timeout=30)
        status = status_r.json().get("status", "unknown")
        log.info(f"  Estado: {status}")
        if status == "done":
            break
        if status in ("error", "deleted"):
            log.error(f"Task fallido con estado: {status}")
            return pd.DataFrame()
        time.sleep(60)

    # Descargar archivos
    files_r = rq.get(f"{APPEEARS_URL}/bundle/{task_id}", headers=headers, timeout=30)
    files   = files_r.json()["files"]
    tmp_dir = ROOT / "data/raw/_tmp_appeears"
    tmp_dir.mkdir(exist_ok=True)

    downloaded = []
    for f in files:
        fname = f["file_name"]
        fid   = f["file_id"]
        dl_r  = rq.get(f"{APPEEARS_URL}/bundle/{task_id}/{fid}",
                       headers=headers, stream=True, timeout=120)
        out_path = tmp_dir / fname
        with open(out_path, "wb") as fh:
            for chunk in dl_r.iter_content(65536):
                fh.write(chunk)
        log.info(f"  Descargado: {fname} ({out_path.stat().st_size/1e6:.1f} MB)")
        downloaded.append(out_path)

    # Parsear NetCDF
    dfs = []
    for nc_path in downloaded:
        if nc_path.suffix != ".nc":
            continue
        try:
            df_nc = _parse_appeears_nc(nc_path)
            dfs.append(df_nc)
        except Exception as e:
            log.warning(f"  Error parseando {nc_path.name}: {e}")

    return _merge_spectral_dfs(dfs) if dfs else pd.DataFrame()


def _parse_appeears_nc(nc_path: Path) -> pd.DataFrame:
    """Extrae series de cuenca de un NetCDF AppEEARS."""
    import netCDF4 as nc4
    ds = nc4.Dataset(nc_path)
    # AppEEARS NetCDF: dims = (time, lat, lon)
    # Por ahora retorna vacío — implementar según estructura real del archivo
    ds.close()
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MODO 3 — Simulación/climatología (sin red, para pruebas)
# ══════════════════════════════════════════════════════════════════════════════

def run_simulate() -> pd.DataFrame:
    """
    Genera índices espectrales sintéticos con patrones realistas
    para pruebas de integración con D6 sin necesidad de red.

    Basado en:
    - NDSI: función de temperatura (tmin_c < 0) y altitud (>3500m activa)
    - NDVI: estacionalidad andina (pico MAM, mínimo AGO)
    - LST: correlación con tmax_c con corrección altitudinal
    """
    log.info("Modo simulación — generando índices sintéticos realistas ...")
    dates  = pd.date_range(DATE_START, DATE_END, freq="D")
    n      = len(dates)
    doy    = dates.dayofyear.values
    month  = dates.month.values

    records = []
    for entity_id, elev in ENTITY_ELEVATIONS.items():
        rng = np.random.default_rng(hash(entity_id) % (2**32))

        # NDSI: > 0 solo en alta montaña, temporada fría (JJA-SON)
        ndsi_base = np.zeros(n)
        if elev > 3800:
            cold_months = np.isin(month, [5, 6, 7, 8, 9, 10])
            intensity   = (elev - 3800) / 700  # 0→1 para 3800→4500m
            ndsi_base   = np.where(cold_months, intensity * 0.65, 0.05) * (
                1 + 0.15 * rng.standard_normal(n))
            ndsi_base   = np.clip(ndsi_base, 0, 0.95)
        elif elev > 3000:
            cold_months  = np.isin(month, [6, 7, 8])
            ndsi_base    = np.where(cold_months, 0.15, 0.02) * (1 + 0.1*rng.standard_normal(n))
            ndsi_base    = np.clip(ndsi_base, 0, 0.5)

        snow_cover_pct = np.clip(ndsi_base * 100, 0, 100)

        # NDVI: pico en verano austral (DJF-MAM) dependiendo de altitud
        peak_month = 3 if elev > 2000 else 1   # MAM en puna, ENE en costa
        ndvi_amp   = 0.55 - elev / 15000        # amplitud decae con altitud
        ndvi_base  = 0.25 + ndvi_amp * np.cos(2*np.pi*(doy - peak_month*30) / 365)
        ndvi_base  = ndvi_base + 0.04*rng.standard_normal(n)
        ndvi_base  = np.clip(ndvi_base, -0.1, 0.9)

        # EVI: similar a NDVI pero menos saturado en dosel denso
        evi_base = ndvi_base * 0.85 - 0.05 + 0.02*rng.standard_normal(n)
        evi_base = np.clip(evi_base, -0.2, 0.9)

        # LSWI: correlaciona con NDVI pero con lag de humedad del suelo
        lswi_base = ndvi_base * 0.7 - 0.1 + 0.03*rng.standard_normal(n)
        lswi_base = np.clip(lswi_base, -0.5, 0.8)

        # LST día: temperatura aproximada de superficie (Kelvin→Celsius)
        lapse_rate  = -6.5  # °C/1000m
        lst_day_ref = 28 + lapse_rate * (elev / 1000)  # ref en costa 28°C
        lst_day_amp = 8 * np.cos(2*np.pi*(doy - 15) / 365)   # estacionalidad
        lst_day_c   = lst_day_ref + lst_day_amp + 2*rng.standard_normal(n)
        lst_day_c   = np.where(ndsi_base > 0.4, -2 + rng.uniform(0,4,n), lst_day_c)

        lst_night_c = lst_day_c - 12 + 2*rng.standard_normal(n)  # DTR ~12°C

        for i, date in enumerate(dates):
            records.append({
                "date":          date,
                "entity_id":     entity_id,
                "ndsi":          round(float(ndsi_base[i]), 4),
                "snow_cover_pct":round(float(snow_cover_pct[i]), 2),
                "ndvi":          round(float(ndvi_base[i]), 4),
                "evi":           round(float(evi_base[i]), 4),
                "lswi":          round(float(lswi_base[i]), 4),
                "lst_day_c":     round(float(lst_day_c[i]), 2),
                "lst_night_c":   round(float(lst_night_c[i]), 2),
                "source":        "simulated",
            })

    df = pd.DataFrame(records)
    log.info(f"  Simulación: {len(df)} filas, {df['entity_id'].nunique()} entidades")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Post-procesamiento común
# ══════════════════════════════════════════════════════════════════════════════

def _merge_spectral_dfs(dfs: list) -> pd.DataFrame:
    """Fusiona DataFrames de diferentes productos (por date + entity_id)."""
    if not dfs:
        return pd.DataFrame()
    base = dfs[0]
    for df in dfs[1:]:
        if df.empty:
            continue
        on = [c for c in ["date", "entity_id"] if c in df.columns and c in base.columns]
        base = base.merge(df, on=on, how="outer")
    return base


def interpolate_16day_to_daily(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Interpola columnas 16-day (NDVI/EVI) a resolución diaria por entidad."""
    out = []
    for entity in df["entity_id"].unique():
        sub = df[df["entity_id"] == entity].set_index("date").sort_index()
        sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq="D"))
        for col in cols:
            if col in sub.columns:
                sub[col] = sub[col].interpolate(method="time", limit=20, limit_direction="both")
        sub["entity_id"] = entity
        out.append(sub.reset_index().rename(columns={"index": "date"}))
    return pd.concat(out)


def compute_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula climatología mensual por (entity_id, month) para imputar 1981-1999."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    cols = [c for c in SPECTRAL_COLS if c in df.columns]
    clim = (df.groupby(["entity_id", "month"])[cols]
              .mean().round(4).reset_index())
    return clim


def extend_to_historical(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """
    Extiende la serie espectral 2000-2020 a 1981-1999 usando la climatología mensual.
    Para el TFT, los NaN en past_observed se manejan con allow_missing_timesteps=True,
    pero es mejor rellenar con climatología para mantener el gradiente altitudinal.
    """
    hist_dates = pd.date_range("1981-01-01", "2000-02-23", freq="D")
    entities   = df["entity_id"].unique()
    records    = []
    for date in hist_dates:
        m = date.month
        for entity in entities:
            row = {"date": date, "entity_id": entity, "source": "climatology"}
            clim_row = clim[(clim["entity_id"] == entity) & (clim["month"] == m)]
            for col in SPECTRAL_COLS:
                if col in clim.columns and not clim_row.empty:
                    row[col] = float(clim_row[col].values[0])
                else:
                    row[col] = np.nan
            records.append(row)
    df_hist = pd.DataFrame(records)
    df["source"] = df.get("source", "modis")
    return pd.concat([df_hist, df], ignore_index=True).sort_values(["entity_id", "date"])


# ══════════════════════════════════════════════════════════════════════════════
# Figuras
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(df: pd.DataFrame):
    """VS01: Panel resumen de todos los índices por zona altitudinal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entities_sorted = ENTITIES_ASC
    colors = plt.cm.plasma_r(np.linspace(0.1, 0.9, len(entities_sorted)))

    plot_vars = [c for c in ["ndsi", "ndvi", "lswi", "lst_day_c"] if c in df.columns]
    if not plot_vars:
        return

    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(16, 4*len(plot_vars)), sharex=True)
    if len(plot_vars) == 1:
        axes = [axes]

    df["date"] = pd.to_datetime(df["date"])

    for ax, var in zip(axes, plot_vars):
        for ent, col in zip(entities_sorted, colors):
            sub = df[df["entity_id"] == ent].sort_values("date")
            if sub[var].notna().sum() < 10:
                continue
            # Resample a mensual para claridad
            monthly = sub.set_index("date")[var].resample("ME").mean()
            ax.plot(monthly.index, monthly.values, color=col, lw=1.2, alpha=0.8,
                    label=entity_label(ent, "label"))
        ax.set_ylabel(var)
        ax.grid(alpha=0.3)
        if var == "ndsi":
            ax.set_title("NDSI (solo sub-cuencas >3800m activas)", fontsize=10, fontweight="bold")
        elif var == "ndvi":
            ax.set_title("NDVI (gradiente: mayor en puna, menor en costa)", fontsize=10, fontweight="bold")

    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Índices espectrales MODIS — Cuenca Chancay-Huaral 2000-2020",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fname = FIG_DIR / "VS01_spectral_overview.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VS01 guardada: {fname.name}")


def plot_snow_cover(df: pd.DataFrame):
    """VS02: Serie de cobertura de nieve por sub-cuenca de alta montaña."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "snow_cover_pct" not in df.columns:
        return

    high_entities = {e: v for e, v in ENTITY_ELEVATIONS.items() if v > 3000}
    entities_sorted = sorted(high_entities, key=lambda e: high_entities[e], reverse=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(entities_sorted)))

    df["date"] = pd.to_datetime(df["date"])
    src_col    = df["source"] if "source" in df.columns else pd.Series(["modis"]*len(df))
    modis_only = df[src_col.isin(["modis", "simulated"])]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    ax = axes[0]
    for ent, col in zip(entities_sorted, colors):
        sub = (modis_only[modis_only["entity_id"] == ent]
               .sort_values("date")
               .set_index("date")["snow_cover_pct"]
               .resample("W").mean())
        if sub.notna().sum() < 10:
            continue
        ax.plot(sub.index, sub.values, color=col, lw=1.0, alpha=0.85,
                label=entity_label(ent, "label"))
    ax.set_ylabel("Cobertura de nieve (%)")
    ax.set_title("(a) Cobertura nieve semanal por sub-cuenca (>3000m)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Climatología mensual de snow_cover en cabecera
    ax2 = axes[1]
    for ent, col in zip(entities_sorted, colors):
        sub = modis_only[modis_only["entity_id"] == ent].copy()
        if sub.empty or sub["snow_cover_pct"].notna().sum() < 10:
            continue
        sub["month"] = pd.to_datetime(sub["date"]).dt.month
        clim = sub.groupby("month")["snow_cover_pct"].agg(["mean","std"])
        ax2.plot(clim.index, clim["mean"].values, "-o", color=col, markersize=5, lw=1.5,
                 label=entity_label(ent, "label"))
        ax2.fill_between(clim.index,
                         clim["mean"] - clim["std"],
                         clim["mean"] + clim["std"], color=col, alpha=0.15)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"])
    ax2.set_ylabel("Cobertura nieve (%) media mensual ±1std")
    ax2.set_title("(b) Climatología mensual de nieve — pico en JJA (invierno austral)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    fig.suptitle("Nieve y hielo — Sub-cuencas cabecera Chancay-Huaral | MODIS MOD10A1",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fname = FIG_DIR / "VS02_snow_cover_timeseries.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VS02 guardada: {fname.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Script 29: Índices espectrales MODIS")
    parser.add_argument("--mode", choices=["gee", "appeears", "simulate"],
                        default="gee",
                        help="Modo de descarga (default: gee — usa TIFs en raw/gee/; simulate SOLO para pruebas sin datos)")
    parser.add_argument("--start", default=DATE_START)
    parser.add_argument("--end",   default=DATE_END)
    parser.add_argument("--user",  default="", help="Usuario EARTHDATA (modo appeears)")
    parser.add_argument("--password", default="", help="Password EARTHDATA (modo appeears)")
    parser.add_argument("--no-extend", action="store_true",
                        help="No extender a 1981-1999 con climatología")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SCRIPT 29: Índices espectrales MODIS — HidroAlerta Chancay-Huaral")
    log.info("=" * 70)
    log.info(f"  Modo: {args.mode} | Período: {args.start} → {args.end}")

    # 1. Descarga / simulación
    if args.mode == "gee":
        df = run_gee(args.start, args.end)
    elif args.mode == "appeears":
        if not args.user or not args.password:
            log.error("--user y --password requeridos para modo appeears")
            log.error("  Registrar cuenta gratuita: https://urs.earthdata.nasa.gov")
            sys.exit(1)
        df = run_appeears(args.start, args.end, args.user, args.password)
    else:
        df = run_simulate()

    if df.empty:
        log.error("No se obtuvieron datos. Verificar modo y credenciales.")
        sys.exit(1)

    if "mode" in df.columns and df["mode"].iloc[0] == "raster_netcdf":
        log.info("Modo NetCDF finalizado. Se omite la generación del CSV tabular.")
        sys.exit(0)

    # 2. Interpolar NDVI/EVI 16-day → daily
    cols_16d = [c for c in ["ndvi", "evi"] if c in df.columns]
    if cols_16d and args.mode != "simulate":
        log.info(f"Interpolando {cols_16d} (16-day → daily) ...")
        df = interpolate_16day_to_daily(df, cols_16d)

    # 3. Calcular climatología
    log.info("Calculando climatología mensual ...")
    clim = compute_climatology(df)
    clim.to_csv(OUT_CLIM, index=False)
    log.info(f"  Climatología guardada: {OUT_CLIM.name}")

    # 4. Extender a 1981-1999 con climatología
    if not args.no_extend:
        log.info("Extendiendo a 1981-1999 con climatología mensual ...")
        df = extend_to_historical(df, clim)

    # 5. Guardar
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["entity_id", "date"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    log.info(f"\nSerie completa guardada: {OUT_CSV}")
    log.info(f"  Filas: {len(df)} | Entidades: {df['entity_id'].nunique()}")
    log.info(f"  Período: {df['date'].min()} → {df['date'].max()}")
    log.info(f"  Columnas: {list(df.columns)}")

    # 6. Figuras
    log.info("\nGenerando figuras ...")
    plot_overview(df)
    plot_snow_cover(df)

    # 7. Instrucciones de integración con D6
    log.info("\n" + "="*70)
    log.info("INTEGRACIÓN CON D6 — pasos para agregar índices espectrales:")
    log.info("  1. Verificar S4_spectral_indices.csv — columnas y rango de valores")
    log.info("  2. En Script 20 (build_d6), agregar merge con S4 por (date, entity_id)")
    log.info("  3. En D6_schema.json, agregar a past_observed:")
    log.info('       ["ndsi", "snow_cover_pct", "ndvi", "evi", "lswi", "lst_day_c"]')
    log.info("  4. Re-ejecutar Script 22 — comparar NSE con y sin índices")
    log.info("  5. En Script 25 (parallel coordinates), agregar columna n_spectral_features")
    log.info("")
    log.info("IMPACTO ESPERADO:")
    log.info("  NDSI/snow_cover → NSE mejora en agosto-octubre (deshielo)")
    log.info("  LSWI → CSI mejora para eventos >=10mm (humedad antecedente)")
    log.info("  LST_day → detección de eventos extremos en El Nino")
    log.info("="*70)


if __name__ == "__main__":
    main()
