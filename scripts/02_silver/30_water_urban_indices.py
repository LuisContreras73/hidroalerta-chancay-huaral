#!/usr/bin/env python3
"""
Script 30: Índices espectrales — Cuerpos de agua y zonas urbanas.

Contexto hidrológico
---------------------
La cuenca Chancay-Huaral presenta dos elementos de cobertura con gran
impacto en la generación de caudal que los índices atmosféricos no capturan:

(A) CUERPOS DE AGUA EN CABECERA (>3500m)
    La estación Vichaycocha (47E257D8, 3503m) mide el caudal de salida de
    la laguna principal: Q_mean=3.4 m³/s, aporta ~16% del Q total en Santo
    Domingo. Pico en FEB-MAR (7.3 m³/s), mínimo en AGO (0.8 m³/s).

    El área del espejo de agua es un proxy directo de almacenamiento lacustre.
    Cuando NDWI sube → laguna llena → mayor caudal base futuro.
    Detección DEM (pendiente<5°, elev>3500m): 64 zonas candidatas en la cuenca.

    Lagunas identificadas (ver CATALOGO_LAGUNAS abajo):
      - Vichaycocha (3503m, ~0.5 km²)  — MEDIDA (estación 47E257D8) ✅
      - Chuchón / Cochaquillo (~4100-4400m) — glaciales
      - Múltiples lagunas >4400m (sub_649, sub_656)

(B) ZONAS URBANAS EN CUENCA BAJA (sub_634, 521m)
    La ciudad de Huaral y zona costera de Chancay están en sub_634.
    Urbanización → mayor coeficiente de escorrentía → respuesta más rápida.
    NDBI positivo en estas zonas; cambio temporal indica expansión urbana.

Índices producidos
------------------

--- CUERPOS DE AGUA (dinámicos → past_observed en D6) ---
ndwi_water    McFeeters (1996): (Green-NIR)/(Green+NIR)         Landsat/S2/MODIS
mndwi         Xu (2006): (Green-SWIR1)/(Green+SWIR1)           Landsat/S2
awei_nsh      Feyisa (2014) no-shadow: 4(G-SWIR1)-0.25*NIR-2.75*SWIR2  Landsat
awei_sh       Feyisa (2014) shadow: B+2.5G-1.5(NIR+SWIR1)-0.25*SWIR2   Landsat
wri           (Green+Red)/(NIR+SWIR1)                           genérico
water_pct     % píxeles con MNDWI>0.3 por sub-cuenca           derivado

--- URBANO/IMPERMEABLE (semi-estático → puede ser static o slow-dynamic) ---
ndbi          Zha (2003): (SWIR1-NIR)/(SWIR1+NIR)              Landsat/S2
ui            Urban Index: (SWIR2-NIR)/(SWIR2+NIR)             Landsat
bu            Built-up: NDBI-NDVI                               derivado
ibi           Index-based Built-up (Xu 2008)                    Landsat
urban_pct     % píxeles con NDBI>0.1 por sub-cuenca            derivado

--- VARIANTES POR FUENTE ---
Landsat 5/7/8/9: resolución 30m, 1984-presente (mejor para lagunas pequeñas)
Sentinel-2:      resolución 10m, 2015-presente (más reciente, mayor detalle)
MODIS:           resolución 500m, 2000-presente (cobertura histórica, diario)
JRC Surface Water (GEE): mensual 1984-presente, ya procesado por Google

Estrategia de integración con D6
----------------------------------
Dinámicos (time series):
  water_pct, ndwi_water, mndwi → past_observed
  Período: 2000-2020 MODIS (500m), rellenar 1984-1999 con Landsat mensual
  Para 1981-1983: climatología mensual (monthly mean 2000-2020)

Estáticos (sin cambio significativo en 40 años):
  urban_pct (% área urbana media 2000-2020) → static_covariates
  Varía poco: Huaral creció, pero en D6 entrenamos 1981-2008

Uso del script
--------------
  python scripts/30_water_urban_indices.py --mode simulate   # sin red (prueba)
  python scripts/30_water_urban_indices.py --mode gee         # GEE (recomendado)
  python scripts/30_water_urban_indices.py --mode jrc         # JRC Surface Water (GEE)

Salidas
-------
  data/silver/S5_water_indices.csv      — time series (date, entity_id, ndwi, mndwi, ...)
  data/silver/S5_urban_static.csv       — fracción urbana por entidad (estático)
  data/bronze/B9_lagoons_catalog.csv    — catálogo de lagunas identificadas
  outputs/figures/spectral/VS03_water_bodies.png
  outputs/figures/spectral/VS04_urban_indices.png
  outputs/figures/spectral/VS05_lagoons_map.png

Referencias
-----------
McFeeters (1996) doi:10.1080/01431169608948714    NDWI
Xu (2006) doi:10.1080/01431160600589179           MNDWI
Feyisa et al. (2014) doi:10.1016/j.rse.2014.01.011  AWEI
Zha et al. (2003) doi:10.1080/01431160304987        NDBI
Xu (2008) doi:10.1080/01431160802039957             IBI
Pekel et al. (2016) doi:10.1038/nature20584         JRC Global Surface Water
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndimage_label

ROOT    = Path(__file__).parent.parent
import sys as _sys; _sys.path.insert(0, str(ROOT / "data/metadata"))
from entity_labels import ENTITY_ELEVATIONS, ENTITIES_ASC, entity_label
SHP_CUE = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
           / "Cuenca_Chancay___Huaral.shp")
SHP_SUB = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas"
           / "Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_geogpsperu_UH_MENORES.shp")
DEM_NC  = ROOT / "data/bronze/B4_dem_basin_30m.nc"
OUT_DIR = ROOT / "data/silver"
FIG_DIR = ROOT / "outputs/figures/spectral"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("s30")

MONTH_NAMES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

DATE_START = "2000-02-24"
DATE_END   = "2025-12-31"


# ══════════════════════════════════════════════════════════════════════════════
# CATÁLOGO DE LAGUNAS — basado en DEM, SNIRH y literatura
# ══════════════════════════════════════════════════════════════════════════════

LAGOONS_CATALOG = [
    # Identificadas por DEM (pendiente<5°, elev>3500m) + conocimiento geográfico
    {
        "nombre": "Laguna Vichaycocha",
        "lat": -11.139, "lon": -76.625, "elevation_m": 3503,
        "area_km2_approx": 0.8,
        "entity_id": "sub_649",
        "tipo": "natural_regulada",
        "medida": True,
        "station_id": "47E257D8",
        "notas": ("Laguna principal de cabecera. Estacion SNIRH mide Q salida. "
                  "Q_mean=3.4 m3/s, 16% del Q total cuenca. Pico FEB-MAR."),
    },
    {
        "nombre": "Laguna Cochaquillo",
        "lat": -11.091, "lon": -76.529, "elevation_m": 4625,
        "area_km2_approx": 0.8,
        "entity_id": "sub_649",
        "tipo": "glacial",
        "medida": False,
        "station_id": None,
        "notas": "Mayor cluster DEM plano >3500m. Probable laguna glacial en cabecera NE.",
    },
    {
        "nombre": "Laguna Chuchon area",
        "lat": -11.217, "lon": -76.511, "elevation_m": 4497,
        "area_km2_approx": 0.31,
        "entity_id": "sub_656",
        "tipo": "glacial",
        "medida": False,
        "station_id": None,
        "notas": "Zona plana ~4500m en sub_656. Posible laguna glacial o bofedal.",
    },
    {
        "nombre": "Laguna Pucacocha (probable)",
        "lat": -11.284, "lon": -76.496, "elevation_m": 4471,
        "area_km2_approx": 0.29,
        "entity_id": "sub_656",
        "tipo": "glacial",
        "medida": False,
        "station_id": None,
        "notas": "Cluster DEM plano en zona alta. Necesita verificacion con imagenes.",
    },
    {
        "nombre": "Bofedal/laguna sub_649 NE",
        "lat": -11.164, "lon": -76.556, "elevation_m": 4615,
        "area_km2_approx": 0.08,
        "entity_id": "sub_649",
        "tipo": "bofedal",
        "medida": False,
        "station_id": None,
        "notas": "Zona plana >4600m. Posible bofedal andino con almacenamiento.",
    },
    {
        "nombre": "Laguna zona Banos",
        "lat": -11.246, "lon": -76.546, "elevation_m": 4550,
        "area_km2_approx": 0.18,
        "entity_id": "sub_646",
        "tipo": "glacial",
        "medida": False,
        "station_id": None,
        "notas": "Sub-cuenca Banos. Zona plana >4500m en sub_646.",
    },
]


def build_lagoons_catalog() -> pd.DataFrame:
    df = pd.DataFrame(LAGOONS_CATALOG)
    out_path = ROOT / "data/bronze/B9_lagoons_catalog.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Catálogo de lagunas guardado: {out_path.name}")
    log.info(f"  {len(df)} lagunas ({df['medida'].sum()} con estacion, "
             f"{(~df['medida']).sum()} inferidas del DEM)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FÓRMULAS DE ÍNDICES ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_ratio(a, b, eps=1e-9):
    return np.where(np.abs(a + b) > eps, (a - b) / (a + b + eps), 0.0)


def compute_water_indices(blue, green, red, nir, swir1, swir2):
    """
    Calcula todos los índices de cuerpos de agua.
    Entradas: arrays normalizados (reflectancia 0-1).

    NDWI (McFeeters 1996): detecta agua abierta, suprime vegetación
    MNDWI (Xu 2006): mejor que NDWI para agua turbida y urbana
    AWEI_nsh (Feyisa 2014): mínimo falsos positivos en zonas sin sombra
    AWEI_sh (Feyisa 2014): incluye corrección de sombra topográfica
    WRI: Water Ratio Index — complemento de NDWI
    """
    ndwi_water = _safe_ratio(green, nir)
    mndwi      = _safe_ratio(green, swir1)
    awei_nsh   = 4*(green - swir1) - (0.25*nir + 2.75*swir2)
    awei_sh    = blue + 2.5*green - 1.5*(nir + swir1) - 0.25*swir2
    wri        = (green + red) / (nir + swir1 + 1e-9)
    water_pct  = (mndwi > 0.3).astype(float)  # threshold Xu 2006

    return {
        "ndwi_water": ndwi_water,
        "mndwi":      mndwi,
        "awei_nsh":   awei_nsh,
        "awei_sh":    awei_sh,
        "wri":        wri,
        "water_pct":  water_pct,
    }


def compute_urban_indices(blue, green, red, nir, swir1, swir2):
    """
    Calcula índices de zonas urbanas/impermeables.

    NDBI (Zha 2003): positivo para suelo desnudo y urbano
    UI: Urban Index, énfasis en SWIR2
    BU: Built-Up = NDBI - NDVI (elimina confusión con vegetación)
    IBI (Xu 2008): combina NDBI, SAVI y MNDWI para separar urbano
    """
    ndvi   = _safe_ratio(nir, red)
    ndbi   = _safe_ratio(swir1, nir)
    ui     = _safe_ratio(swir2, nir)
    bu     = ndbi - ndvi
    # IBI: combina tres señales (suelo desnudo, SWIR, agua)
    savi      = 1.5 * (nir - red) / (nir + red + 0.5 + 1e-9)
    mndwi_tmp = _safe_ratio(green, swir1)
    ibi_num   = ndbi - 0.5 * (savi + mndwi_tmp)
    ibi_den   = ndbi + 0.5 * (savi + mndwi_tmp)
    ibi     = np.where(np.abs(ibi_den) > 1e-9, ibi_num / ibi_den, 0.0)
    urban_pct = (ndbi > 0.1).astype(float)

    return {
        "ndbi":       ndbi,
        "ui":         ui,
        "bu":         bu,
        "ibi":        ibi,
        "urban_pct":  urban_pct,
        "ndvi":       ndvi,  # también útil aquí
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODO GEE — descarga via Google Earth Engine
# ══════════════════════════════════════════════════════════════════════════════

GEE_CODE = '''
// ── GEE Code para índices de agua y urbano — Chancay-Huaral ──
// Ejecutar en: https://code.earthengine.google.com

var subcuencas = ee.FeatureCollection('users/YOUR_USER/chancay_subcuencas');
var basin = ee.FeatureCollection('users/YOUR_USER/chancay_cuenca');
var start = '2000-01-01', end = '2020-12-31';

// ── 1. JRC Global Surface Water (1984-2023) ────────────────────────────────
// Ya procesado, mensual, 30m — el más fácil y confiable
var jrc = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
  .filterDate(start, end);

var waterFreq = jrc.map(function(img) {
  var water = img.select('water').eq(2);  // 2 = water, 1 = land
  return subcuencas.map(function(f) {
    var pct = water.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: f.geometry(), scale: 30, maxPixels: 1e7
    });
    return f.set({date: img.date().format('YYYY-MM'), water_pct_jrc: pct.get('water')});
  });
}).flatten();

// ── 2. Landsat 8 OLI — índices de agua y urbano (2013-2020) ───────────────
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(basin.geometry())
  .filterDate('2013-01-01', end)
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .map(function(img) {
    // Factor de escala Landsat Collection 2
    var B = img.select('SR_B2').multiply(0.0000275).add(-0.2);  // Blue
    var G = img.select('SR_B3').multiply(0.0000275).add(-0.2);  // Green
    var R = img.select('SR_B4').multiply(0.0000275).add(-0.2);  // Red
    var N = img.select('SR_B5').multiply(0.0000275).add(-0.2);  // NIR
    var S1= img.select('SR_B6').multiply(0.0000275).add(-0.2);  // SWIR1
    var S2= img.select('SR_B7').multiply(0.0000275).add(-0.2);  // SWIR2

    var ndwi   = G.subtract(N).divide(G.add(N)).rename('ndwi_water');
    var mndwi  = G.subtract(S1).divide(G.add(S1)).rename('mndwi');
    var awei   = G.multiply(4).subtract(S1.multiply(4)).subtract(N.multiply(0.25)).subtract(S2.multiply(2.75)).rename('awei_nsh');
    var ndbi   = S1.subtract(N).divide(S1.add(N)).rename('ndbi');
    var ndvi   = N.subtract(R).divide(N.add(R)).rename('ndvi');
    var bu     = ndbi.subtract(ndvi).rename('bu');

    return img.addBands([ndwi, mndwi, awei, ndbi, bu])
              .set({system_time_start: img.get('system:time_start')});
  });

// Reducir mensualmente por subcuenca
var months = ee.List.sequence(0, 95);  // 96 meses 2013-2020
var l8_monthly = months.map(function(m) {
  var date = ee.Date('2013-01-01').advance(m, 'month');
  var endDate = date.advance(1, 'month');
  var composite = l8.filterDate(date, endDate).median();
  return subcuencas.map(function(f) {
    var stats = composite.select(['ndwi_water','mndwi','awei_nsh','ndbi','bu'])
      .reduceRegion({reducer: ee.Reducer.mean(), geometry: f.geometry(), scale: 30, maxPixels: 1e7});
    return f.set(stats).set({date: date.format('YYYY-MM')});
  });
}).flatten();

// ── 3. Landsat 5 TM — índices históricos 1984-2012 ────────────────────────
// Mismo proceso pero con Landsat 5 TM (B1=Blue,B2=Green,B3=Red,B4=NIR,B5=SWIR1,B7=SWIR2)

// ── Export ────────────────────────────────────────────────────────────────
Export.table.toDrive({
  collection: waterFreq.flatten(),
  description: 'jrc_water_chancay',
  fileFormat: 'CSV'
});
Export.table.toDrive({
  collection: l8_monthly.flatten(),
  description: 'l8_indices_chancay',
  fileFormat: 'CSV'
});
'''


def run_gee_water(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga índices de agua y urbano a NetCDF (Rasters) vía xee."""
    try:
        import ee
        import geemap
        ee.Initialize(project='ana-chancay-huaral', opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        log.error(f"GEE no disponible: {e}")
        return pd.DataFrame(), pd.DataFrame()

    import geopandas as gpd
    from shapely.geometry import mapping

    out_dir = ROOT / "data/silver/rasters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc_jrc = out_dir / "S5_water_jrc_monthly.nc"
    out_nc_l8 = out_dir / "S5_urban_l8_monthly.nc"

    gdf_cue = gpd.read_file(SHP_CUE).to_crs("EPSG:4326")
    geom = ee.Geometry(mapping(gdf_cue.geometry.iloc[0]))

    # ── JRC Surface Water (mensual) ─────────────────────────────────────────
    log.info("Descargando JRC Global Surface Water a NetCDF ...")
    jrc = (ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
           .filterDate(start, end).select("water"))
           
    jrc = jrc.map(lambda img: img.clip(geom))

    log.info("Enviando tareas de exportación de JRC Water a Google Drive ...")
    jrc_dates = pd.date_range(start, end, freq='MS')
    for i, date in enumerate(jrc_dates):
        s_date = date.strftime('%Y-%m-%d')
        e_date = (date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        year_mo = date.strftime('%Y_%m')
        img = jrc.filterDate(s_date, e_date).first()
        if not img: continue

        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"S30_jrc_{year_mo}",
            folder="HidroAlerta_Rasters/S30_water",
            fileNamePrefix=f"S30_jrc_{year_mo}",
            scale=30,
            region=geom,
            crs='EPSG:4326',
            maxPixels=1e10,
            fileFormat='GeoTIFF',
        )
        task.start()
    log.info(f"¡{len(jrc_dates)} tareas JRC enviadas a Drive!")

    # ── Landsat 8 (NDWI, MNDWI, NDBI) ────────────────────────────────────
    log.info("Descargando Landsat 8 índices a NetCDF (2013-2025) ...")
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(geom)
          .filterDate("2013-01-01", end)
          .filter(ee.Filter.lt("CLOUD_COVER", 20)))

    def _scale_l8(img):
        # Escala oficial C2 L2: multiplicar por 0.0000275 y sumar -0.2 → reflectancia 0-1
        scaled = img.select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]
                            ).multiply(0.0000275).add(-0.2).rename(
                             ["blue","green","red","nir","swir1","swir2"])
        # Banda alias
        B = scaled.select("blue")
        G = scaled.select("green")
        R = scaled.select("red")
        N = scaled.select("nir")
        S1= scaled.select("swir1")
        eps = 1e-9
        # Índices
        ndwi  = G.subtract(N).divide(G.add(N).add(eps)).rename("ndwi")
        mndwi = G.subtract(S1).divide(G.add(S1).add(eps)).rename("mndwi")
        ndbi  = S1.subtract(N).divide(S1.add(N).add(eps)).rename("ndbi")
        ndvi  = N.subtract(R).divide(N.add(R).add(eps)).rename("ndvi")
        ndsi  = G.subtract(S1).divide(G.add(S1).add(eps)).rename("ndsi")
        # Mantener bandas crudas + índices
        return (scaled.addBands([ndwi, mndwi, ndbi, ndvi, ndsi])
                .copyProperties(img, ["system:time_start"]))

    l8 = l8.map(_scale_l8)

    months = pd.date_range("2013-01-01", end, freq='MS')
    def get_l8_monthly(date):
        s = ee.Date(date)
        e = s.advance(1, 'month')
        comp = l8.filterDate(s, e).median().clip(geom)
        return comp.set('system:time_start', s.millis())

    export_bands_l8 = ["blue","green","red","nir","swir1","swir2",
                       "ndwi","mndwi","ndbi","ndvi","ndsi"]
        
    ee_dates = ee.List([d.strftime('%Y-%m-%d') for d in months])
    l8_monthly = ee.ImageCollection.fromImages(ee_dates.map(get_l8_monthly))

    log.info("Enviando tareas de exportación Landsat 8 a Google Drive ...")
    log.info("  Carpeta Drive: HidroAlerta_Rasters/S30_l8/")
    log.info(f"  Bandas por imagen: {export_bands_l8} ({len(export_bands_l8)} total)")
    ee_dates_list = ee_dates.getInfo()
    n_l8 = len(ee_dates_list)
    for i, date_str in enumerate(ee_dates_list):
        s = ee.Date(date_str)
        e = s.advance(1, 'month')
        year_mo = f"{date_str[:4]}_{date_str[5:7]}"
        img = l8_monthly.filterDate(s, e).first()

        task = ee.batch.Export.image.toDrive(
            image=img.select(export_bands_l8),
            description=f"S30_l8_{year_mo}",
            folder="HidroAlerta_Rasters/S30_l8",
            fileNamePrefix=f"S30_l8_{year_mo}",
            scale=30,
            region=geom,
            crs='EPSG:4326',
            maxPixels=1e10,
            fileFormat='GeoTIFF',
        )
        task.start()
        if (i + 1) % 30 == 0 or i == n_l8 - 1:
            log.info(f"  [{i+1}/{n_l8}] Enviadas hasta {date_str[:7]}")
    log.info(f"¡{n_l8} tareas Landsat 8 enviadas a Drive!")

    return pd.DataFrame([{"mode": "raster_netcdf"}]), pd.DataFrame([{"mode": "raster_netcdf"}])
    
    # Merge both into df_water
    df_water = pd.merge(jrc_df, l8_df, on=["date", "entity_id"], how="outer")
    df_water["source"] = "gee"
    if "water_pct" in df_water.columns:
        df_water["water_pct"] = df_water["water_pct"] * 100  # convert to percentage
        
    # Create static df_urban
    df_urban_records = []
    for ent in df_water["entity_id"].unique():
        sub = df_water[df_water["entity_id"] == ent]
        urban_pct_daily = (sub["ndbi"] > 0.1).astype(float) * 100 if "ndbi" in sub.columns else pd.Series([0.0])
        df_urban_records.append({
            "entity_id": ent,
            "elevation_m": ENTITY_ELEVATIONS.get(ent, 0),
            "urban_pct": round(float(urban_pct_daily.mean()), 2),
            "urban_pct_std": round(float(urban_pct_daily.std()), 3),
        })
    df_urban = pd.DataFrame(df_urban_records)
    
    return df_water, df_urban


# ══════════════════════════════════════════════════════════════════════════════
# MODO SIMULACIÓN — patrones sintéticos realistas
# ══════════════════════════════════════════════════════════════════════════════

def run_simulate_water() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera series sintéticas con patrones hidrológicamente realistas basados en:
    - Patrón Vichaycocha (Q medido 2023-2026): estacional FEB=máx, AGO=mín
    - Gradiente altitudinal: más agua en zonas altas
    - Urbanismo en sub_634 (Huaral): NDBI positivo, constante (urbanización lenta)
    """
    log.info("Modo simulación — generando índices de agua y urbano ...")
    dates  = pd.date_range(DATE_START, DATE_END, freq="D")
    n      = len(dates)
    doy    = dates.dayofyear.values
    month  = dates.month.values
    year   = dates.year.values

    # Ciclo estacional agua: pico FEB (día 45), mínimo AGO (día 225)
    # Basado en Q climatológico de Vichaycocha
    Q_CLIM_MONTHLY = {1:5.1, 2:6.9, 3:7.3, 4:4.9, 5:2.3, 6:1.3,
                      7:1.0, 8:0.8, 9:0.8, 10:1.2, 11:1.8, 12:3.9}
    Q_MAX = max(Q_CLIM_MONTHLY.values())  # 7.3

    water_records = []
    urban_records = []

    for entity_id, elev in ENTITY_ELEVATIONS.items():
        rng = np.random.default_rng(sum(ord(c) for c in entity_id))

        # Factor de presencia de agua por altitud
        # Sub-cuencas altas tienen lagunas permanentes
        if elev > 4000:
            water_base_pct = 0.05      # 5% del área = lagunas
            water_seasonal = 0.03      # varía ±3% con lluvia
        elif elev > 3000:
            water_base_pct = 0.02
            water_seasonal = 0.01
        else:
            water_base_pct = 0.005     # solo ríos/canales
            water_seasonal = 0.002

        # Señal estacional basada en Q Vichaycocha (proxy de almacenamiento)
        q_seasonal = np.array([Q_CLIM_MONTHLY[m] / Q_MAX for m in month])

        # NDWI: proporcional a almacenamiento de agua
        ndwi_base = -0.3 + (elev / 15000)  # base negativo en zonas bajas
        ndwi_amp  = 0.4 if elev > 3500 else 0.15
        ndwi_w    = ndwi_base + ndwi_amp * q_seasonal + 0.05 * rng.standard_normal(n)
        ndwi_w    = np.clip(ndwi_w, -0.8, 0.9)

        # MNDWI: similar pero más positivo en agua abierta
        mndwi     = ndwi_w * 1.2 + 0.1 + 0.03 * rng.standard_normal(n)
        mndwi     = np.clip(mndwi, -0.8, 0.9)

        # water_pct: fracción de píxeles con agua
        wpc = water_base_pct + water_seasonal * q_seasonal
        wpc = np.clip(wpc + 0.005 * rng.standard_normal(n), 0, 1)
        water_pct = wpc * 100  # a porcentaje

        # AWEI_nsh: ~ 3 × MNDWI para agua abierta en alta montaña
        awei_nsh = np.where(mndwi > 0, mndwi * 2.5, mndwi * 0.5)
        awei_nsh += 0.05 * rng.standard_normal(n)

        # NDBI (urbano): negativo en zonas altas/vegetadas, positivo en sub_634
        if entity_id == "sub_634":
            ndbi_base = 0.08  # Huaral: zona urbano-suburbana
            # Ligero crecimiento urbano 2000-2020
            ndbi = ndbi_base + 0.0003 * (year - 2000) + 0.02 * rng.standard_normal(n)
        else:
            ndbi_base = -0.15 + elev / 50000
            ndbi = ndbi_base + 0.02 * rng.standard_normal(n)
        ndbi = np.clip(ndbi, -0.5, 0.5)

        # BU = NDBI - NDVI
        ndvi_sim = 0.35 - elev/20000 + 0.25*q_seasonal
        bu   = ndbi - np.clip(ndvi_sim + 0.03*rng.standard_normal(n), -0.2, 0.9)

        # urban_pct: casi estático
        if entity_id == "sub_634":
            urban_base = 12.0  # 12% del área es urbano en sub_634
        elif entity_id == "sub_640":
            urban_base = 2.5
        else:
            urban_base = 0.3
        urban_pct_daily = urban_base + 0.1 * rng.standard_normal(n)
        urban_pct_daily = np.clip(urban_pct_daily, 0, 100)

        for i, date in enumerate(dates):
            water_records.append({
                "date":       date,
                "entity_id":  entity_id,
                "ndwi_water": round(float(ndwi_w[i]), 4),
                "mndwi":      round(float(mndwi[i]), 4),
                "awei_nsh":   round(float(awei_nsh[i]), 4),
                "water_pct":  round(float(water_pct[i]), 3),
                "ndbi":       round(float(ndbi[i]), 4),
                "bu":         round(float(bu[i]), 4),
                "source":     "simulated",
            })

        # Estático urbano: un valor por entidad (media del período)
        urban_records.append({
            "entity_id":    entity_id,
            "elevation_m":  elev,
            "urban_pct":    round(float(urban_pct_daily.mean()), 2),
            "urban_pct_std": round(float(urban_pct_daily.std()), 3),
        })

    df_water = pd.DataFrame(water_records)
    df_urban = pd.DataFrame(urban_records)
    log.info(f"  Series agua: {len(df_water)} filas, {df_water['entity_id'].nunique()} entidades")
    log.info(f"  Urbano estático: {len(df_urban)} entidades")
    return df_water, df_urban


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def plot_water_bodies(df: pd.DataFrame, lagunas: pd.DataFrame):
    """VS03: Dinámica de cuerpos de agua por sub-cuenca."""
    df["date"] = pd.to_datetime(df["date"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors = plt.cm.Blues_r(np.linspace(0.1, 0.8, len(ENTITIES_ASC)))

    # Panel 1: MNDWI mensual por entidad
    ax = axes[0]
    for ent, col in zip(ENTITIES_ASC, colors):
        sub = df[df["entity_id"] == ent].sort_values("date").set_index("date")
        if "mndwi" not in sub.columns or sub["mndwi"].notna().sum() < 30:
            continue
        monthly = sub["mndwi"].resample("ME").mean()
        ax.plot(monthly.index, monthly.values, color=col, lw=1.2, alpha=0.85,
                label=entity_label(ent, "label"))
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.3, color="#2980b9", lw=0.8, ls=":", alpha=0.6, label="umbral agua (0.3)")
    ax.set_ylabel("MNDWI mensual")
    ax.set_title("(a) MNDWI — detección cuerpos de agua por sub-cuenca\n"
                 "Valores > 0.3 indican agua libre (lagunas en cabecera)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: water_pct en sub-cuencas altas (>3500m)
    ax2 = axes[1]
    high_ents = [e for e in ENTITIES_ASC if ENTITY_ELEVATIONS[e] > 3500]
    colors_h  = plt.cm.Reds_r(np.linspace(0.1, 0.8, len(high_ents)))
    for ent, col in zip(high_ents, colors_h):
        sub = df[df["entity_id"] == ent].sort_values("date").set_index("date")
        if "water_pct" not in sub.columns:
            continue
        monthly = sub["water_pct"].resample("ME").mean()
        ax2.plot(monthly.index, monthly.values, color=col, lw=1.5,
                 label=entity_label(ent, "label"))
    ax2.set_ylabel("% área con agua libre")
    ax2.set_title("(b) Fracción de agua libre (MNDWI>0.3) — sub-cuencas >3500m\n"
                  "Proxy de almacenamiento lacustre (Vichaycocha y otras lagunas)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    # Panel 3: Climatología mensual MNDWI vs Q_Vichaycocha
    ax3 = axes[2]
    Q_CLIM = {1:5.1, 2:6.9, 3:7.3, 4:4.9, 5:2.3, 6:1.3,
              7:1.0, 8:0.8, 9:0.8, 10:1.2, 11:1.8, 12:3.9}
    df["month"] = df["date"].dt.month
    for ent, col in zip(["sub_649","sub_656","sub_646"], plt.cm.Blues_r([0.2, 0.5, 0.7])):
        sub = df[df["entity_id"] == ent]
        if sub.empty:
            continue
        clim = sub.groupby("month")["mndwi"].mean()
        ax3.plot(clim.index, clim.values, "-o", color=col, markersize=5, lw=1.5,
                 label=f"MNDWI {entity_label(ent, 'short')}")
    ax3_twin = ax3.twinx()
    ax3_twin.bar(list(Q_CLIM.keys()), list(Q_CLIM.values()),
                 color="#e74c3c", alpha=0.4, width=0.4, label="Q Vichaycocha (m3/s)")
    ax3_twin.set_ylabel("Q Vichaycocha (m³/s)", color="#e74c3c")
    ax3.set_xticks(range(1, 13)); ax3.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax3.set_ylabel("MNDWI climatología")
    ax3.set_title("(c) Climatología mensual MNDWI vs Q Vichaycocha\n"
                  "MNDWI sigue el ciclo de almacenamiento lacustre", fontsize=10, fontweight="bold")
    lines1, lab1 = ax3.get_legend_handles_labels()
    lines2, lab2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)

    fig.suptitle("Índices de cuerpos de agua — Cuenca Chancay-Huaral\n"
                 "Lagunas cabecera: Vichaycocha + glaciales >4400m", fontsize=12, y=1.01)
    fig.tight_layout()
    fname = FIG_DIR / "VS03_water_bodies.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VS03 guardada: {fname.name}")


def plot_urban_indices(df: pd.DataFrame, df_urban: pd.DataFrame):
    """VS04: Índices urbanos y cobertura impermeable."""
    df["date"] = pd.to_datetime(df["date"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Barras de urban_pct por entidad
    ax = axes[0]
    df_urban_s = df_urban.sort_values("elevation_m")
    colors_u = ["#e74c3c" if r["urban_pct"] > 5 else
                "#e67e22" if r["urban_pct"] > 1 else "#2ecc71"
                for _, r in df_urban_s.iterrows()]
    bars = ax.barh(range(len(df_urban_s)), df_urban_s["urban_pct"], color=colors_u, height=0.7)
    ax.set_yticks(range(len(df_urban_s)))
    ax.set_yticklabels([entity_label(r["entity_id"], "tick")
                        for _, r in df_urban_s.iterrows()], fontsize=9)
    ax.set_xlabel("Fracción urbana / impermeable (%)")
    ax.set_title("(a) Cobertura urbana por sub-cuenca (NDBI>0.1)\n"
                 "Sub_634 (Huaral): mayor impermeabilidad", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, df_urban_s["urban_pct"]):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    ax.axvline(5, color="red", ls="--", lw=0.8, alpha=0.6, label="umbral urbano 5%")
    ax.legend(fontsize=8)

    # Serie temporal NDBI en sub_634 (urbanización)
    ax2 = axes[1]
    sub634 = df[df["entity_id"] == "sub_634"].sort_values("date").set_index("date")
    if "ndbi" in sub634.columns and sub634["ndbi"].notna().sum() > 30:
        annual = sub634["ndbi"].resample("YE").mean()
        ax2.plot(annual.index, annual.values, "-o", color="#e74c3c", lw=2, markersize=6)
        # Tendencia
        y = annual.values
        x = np.arange(len(y))
        mask = np.isfinite(y)
        if mask.sum() > 3:
            from scipy import stats as st
            slope, intercept, r, p, _ = st.linregress(x[mask], y[mask])
            trend = slope * x + intercept
            ax2.plot(annual.index, trend, "--", color="#c0392b", lw=1.5, alpha=0.7,
                     label=f"tendencia NDBI: {slope*10:.4f}/dec (p={p:.3f})")
    ax2.set_ylabel("NDBI anual (sub_634 Huaral)")
    ax2.set_xlabel("Año")
    ax2.set_title("(b) Evolución temporal NDBI en sub_634 (Huaral)\n"
                  "NDBI creciente = expansión urbana/impermeabilización", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axhline(0.1, color="red", lw=0.8, ls=":", alpha=0.5, label="umbral NDBI urbano")

    fig.suptitle("Índices urbanos/impermeables — Cuenca Chancay-Huaral",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fname = FIG_DIR / "VS04_urban_indices.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VS04 guardada: {fname.name}")


def plot_lagoons_map(lagunas: pd.DataFrame):
    """VS05: Mapa de lagunas identificadas sobre DEM."""
    try:
        import netCDF4 as nc4

        ds = nc4.Dataset(DEM_NC)
        lat = np.array(ds.variables["lat"][:])
        lon = np.array(ds.variables["lon"][:])
        dem = np.array(ds.variables["dem"][:])
        ds.close()

        fig, ax = plt.subplots(figsize=(10, 9))

        # DEM de fondo con hillshade simulado
        im = ax.imshow(dem, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       cmap="terrain", origin="upper", vmin=0, vmax=5000, alpha=0.8)
        plt.colorbar(im, ax=ax, label="Elevación (m)", fraction=0.03, pad=0.02)

        # Contornos
        LON, LAT = np.meshgrid(lon, lat)
        ax.contour(LON, LAT, dem, levels=[1000, 2000, 3000, 4000, 4500],
                   colors="white", linewidths=0.5, alpha=0.6)

        # Lagunas
        for _, row in lagunas.iterrows():
            color  = "#e74c3c" if row["medida"] else "#3498db"
            marker = "D" if row["medida"] else "o"
            size   = 180 if row["medida"] else 100
            ax.scatter(row["lon"], row["lat"], c=color, s=size, marker=marker,
                       edgecolors="white", linewidths=1.5, zorder=5)
            ax.annotate(
                row["nombre"].replace("Laguna ","L."),
                (row["lon"], row["lat"]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
            )

        # Estación Vichaycocha (triángulo)
        vich = lagunas[lagunas["medida"]].iloc[0]
        ax.scatter(vich["lon"], vich["lat"], c="#f39c12", s=300, marker="^",
                   edgecolors="white", linewidths=2, zorder=6,
                   label=f"Laguna medida (estacion SNIRH)")
        ax.scatter([], [], c="#3498db", s=80, marker="o", label="Laguna inferida (DEM)")

        # Centroides sub-cuencas
        for ent, elev in ENTITY_ELEVATIONS.items():
            pass  # (omitido para brevedad — añadir si se tiene GeoDataFrame)

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title("Cuerpos de agua identificados — Cuenca Chancay-Huaral\n"
                     "Fuente: SNIRH (medido) + análisis DEM 30m (inferido)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3, color="white", lw=0.5)

        fname = FIG_DIR / "VS05_lagoons_map.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  VS05 guardada: {fname.name}")
    except Exception as e:
        log.warning(f"  VS05 (mapa lagunas): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSIÓN 1981-1999 CON CLIMATOLOGÍA
# ══════════════════════════════════════════════════════════════════════════════

def extend_to_historical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena 1981-01-01 → DATE_START-1d usando climatología mensual del período MODIS.
    Vectorizado: no itera día a día sino que usa merge sobre (entity_id, month).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    cols = ["ndwi_water", "mndwi", "awei_nsh", "water_pct", "ndbi", "bu"]
    cols = [c for c in cols if c in df.columns]
    clim = df.groupby(["entity_id", "month"])[cols].mean()

    hist_dates  = pd.date_range("1981-01-01", DATE_START, freq="D")[:-1]
    entities    = df["entity_id"].unique()
    hist_idx    = pd.MultiIndex.from_product([entities, hist_dates],
                                             names=["entity_id", "date"])
    df_hist = pd.DataFrame(index=hist_idx).reset_index()
    df_hist["month"]  = df_hist["date"].dt.month
    df_hist["source"] = "climatology"
    df_hist = df_hist.join(clim, on=["entity_id", "month"])

    df = df.drop(columns=["month"], errors="ignore")
    return pd.concat([df_hist.drop(columns="month"), df],
                     ignore_index=True).sort_values(["entity_id", "date"])


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Script 30: Water & Urban Indices")
    parser.add_argument("--mode", choices=["gee", "jrc", "simulate"], default="jrc",
                        help="Modo (default: jrc — usa TIFs JRC/Landsat reales; simulate SOLO para pruebas)")
    parser.add_argument("--start", default=DATE_START)
    parser.add_argument("--end",   default=DATE_END)
    parser.add_argument("--no-extend", action="store_true")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SCRIPT 30: Índices agua y urbano — HidroAlerta Chancay-Huaral")
    log.info("=" * 70)

    # Catálogo de lagunas
    lagunas = build_lagoons_catalog()
    log.info(f"\nLagunas catalogadas:")
    for _, r in lagunas.iterrows():
        medida = "MEDIDA" if r["medida"] else "inferida"
        log.info(f"  [{medida}] {r['nombre']} — {r['elevation_m']}m, "
                 f"~{r['area_km2_approx']} km2, entity={r['entity_id']}")

    # Descarga/simulación
    if args.mode in ("gee", "jrc"):
        df_water, df_urban = run_gee_water(args.start, args.end)
    else:
        df_water, df_urban = run_simulate_water()

    if df_water.empty and df_urban.empty:
        log.error("No se obtuvieron datos (JRC/L8).")
        sys.exit(1)

    if "mode" in df_water.columns and df_water["mode"].iloc[0] == "raster_netcdf":
        log.info("Modo NetCDF finalizado. Se omite la generación del CSV tabular.")
        sys.exit(0)

    # Extender a 1981
    if not args.no_extend:
        log.info("Extendiendo 1981-1999 con climatología ...")
        df_water = extend_to_historical(df_water)

    # Guardar
    df_water["date"] = pd.to_datetime(df_water["date"]).dt.strftime("%Y-%m-%d")
    df_water = df_water.sort_values(["entity_id", "date"]).reset_index(drop=True)
    out_water = OUT_DIR / "S5_water_indices.csv"
    df_water.to_csv(out_water, index=False)
    log.info(f"\nSerie agua guardada: {out_water.name} ({len(df_water)} filas)")

    out_urban = OUT_DIR / "S5_urban_static.csv"
    df_urban.to_csv(out_urban, index=False)
    log.info(f"Urbano estático guardado: {out_urban.name}")

    # Figuras
    log.info("\nGenerando figuras ...")
    plot_water_bodies(df_water, lagunas)
    plot_urban_indices(df_water, df_urban)
    plot_lagoons_map(lagunas)

    log.info("\n" + "="*70)
    log.info("INTEGRACIÓN CON D6:")
    log.info("  past_observed (dinámico): ndwi_water, mndwi, water_pct, ndbi, bu")
    log.info("  static_covariates: urban_pct (de S5_urban_static.csv)")
    log.info("")
    log.info("CUERPOS DE AGUA IDENTIFICADOS:")
    log.info("  1 laguna medida: Vichaycocha (3503m, 16% Q cuenca)")
    log.info("  5 lagunas inferidas: >4400m, area 0.07-0.80 km2")
    log.info("  Total agua libre estimada: ~2 km2 en >3500m")
    log.info("")
    log.info("URBANISMO:")
    log.info("  sub_634 (Huaral): ~12% urbano/impermeable — mayor runoff")
    log.info("  sub_640: ~2.5% — crecimiento periurbano")
    log.info("  Resto: <0.5% — cuenca natural")
    log.info("="*70)


if __name__ == "__main__":
    main()
