#!/usr/bin/env python3
"""
SCRIPT 32: Sentinel-2 MSI — HidroAlerta Chancay-Huaral
=========================================================
Satélite : Copernicus Sentinel-2A (Jun 2015) / Sentinel-2B (Mar 2017)  (ESA)
Sensor   : MSI (Multi-Spectral Instrument, push-broom)
Cobertura: Jun 2015 – presente  (S2A solo: Jun 2015 – Mar 2017)
Revisita : ~5 días (combinando S2A + S2B)

Resoluciones nativas por banda:
  10m : B2(Blue 490nm), B3(Green 560nm), B4(Red 665nm), B8(NIR 842nm)
  20m : B5(RE1 705nm), B6(RE2 740nm), B7(RE3 783nm), B8A(nNIR 865nm),
        B11(SWIR1 1610nm), B12(SWIR2 2190nm)
  60m : B1(Coastal 443nm), B9(WV 945nm)  — no se exportan, baja resolución

Resolución export : 20m  (compromiso entre detalle y espacio en disco)
Colección GEE    : COPERNICUS/S2_SR_HARMONIZED (reflectancia de superficie,
                    con corrección atmosférica Sen2Cor, armonizada entre S2A/S2B)

FACTOR DE ESCALA: Las bandas en GEE están en rango 0–10000.
  Reflectancia real = valor / 10000  (aplicado al calcular índices)

DIFERENCIAS CLAVE VS MODIS/LANDSAT EN LOS MISMOS ÍNDICES:
───────────────────────────────────────────────────────────────────────────────
Índice  │ MODIS (MOD13Q1)        │ Landsat 8 OLI         │ Sentinel-2 MSI
────────┼────────────────────────┼───────────────────────┼───────────────────
NDVI    │ (B2_NIR - B1_RED) /    │ (B5_NIR - B4_RED) /   │ (B8_NIR - B4_RED) /
        │  (B2 + B1)             │  (B5 + B4)            │  (B8 + B4)
        │  NIR=841-876nm         │  NIR=851-878nm        │  NIR=785-900nm (más
        │                        │                       │  ancho → valores ~5%
        │                        │                       │  más altos que L8)
        │ Alternativa comparab.: │                       │  B8A=855-875nm  ←
        │  usa B8A en S2         │                       │  más comparable a L8
────────┼────────────────────────┼───────────────────────┼───────────────────
NDWI    │ (B4_Green - B2_NIR) /  │ (B3_Green - B5_NIR) / │ (B3_Green - B8_NIR)/
(McFee) │  (B4 + B2)             │  (B3 + B5)            │  (B3 + B8)
        │  Green=545-565nm       │  Green=533-590nm      │  Green=543-578nm
        │                        │                       │  → comparable
────────┼────────────────────────┼───────────────────────┼───────────────────
MNDWI   │ (B4_Green - B6_SWIR1)/ │ (B3_Green - B6_SWIR1)/│ (B3_Green - B11) /
(Xu)    │  (B4 + B6)             │  (B3 + B6)            │  (B3 + B11)
        │  SWIR1=1628-1652nm     │  SWIR1=1570-1650nm    │  SWIR1=1565-1655nm
        │                        │                       │  → muy comparable
────────┼────────────────────────┼───────────────────────┼───────────────────
NDSI    │ (B4_Green - B6_SWIR) / │ (B3_Green - B6_SWIR) /│ (B3_Green - B11) /
(Nieve) │  (B4 + B6)             │  (B3 + B6)            │  (B3 + B11)
        │                        │                       │  → comparable
────────┼────────────────────────┼───────────────────────┼───────────────────
NDBI    │ (B6_SWIR - B2_NIR) /   │ (B6_SWIR1 - B5_NIR) / │ (B11_SWIR - B8_NIR)/
(Urban) │  (B6 + B2)             │  (B6 + B5)            │  (B11 + B8)
        │                        │                       │  → comparable
────────┼────────────────────────┼───────────────────────┼───────────────────
EVI     │ 2.5*(B2-B1)/           │ 2.5*(B5-B4)/          │ 2.5*(B8-B4)/
        │  (B2+6*B1-7.5*B3+1)   │  (B5+6*B4-7.5*B2+1)  │  (B8+6*B4-7.5*B2+1)
        │  Coef. orig. MODIS     │  Coef. iguales        │  Coef. iguales
        │                        │                       │  → similar a L8
───────────────────────────────────────────────────────────────────────────────

ÍNDICES EXCLUSIVOS DE SENTINEL-2 (requieren bandas Red-Edge):
  NDRE   (Normalized Difference Red-Edge) = (B8A - B5) / (B8A + B5)
         → Estrés hídrico en cultivos, salud vegetal avanzada
         → NO tiene equivalente en MODIS ni Landsat 8 (sin bandas Red-Edge)
  CIre   (Red-Edge Chlorophyll Index)     = (B7 / B5) - 1
         → Contenido de clorofila hoja, correlaciona con nitrógeno
  BSI    (Bare Soil Index)               = ((B11+B4)-(B8+B2)) / ((B11+B4)+(B8+B2))
         → Suelo desnudo, erosión, áreas agrícolas post-cosecha

BANDAS EXPORTADAS (compuesto mensual = mediana a 20m):
───────────────────────────────────────────────────────────────────────
  # Bandas "crudas" para imagen natural y cálculo propio de índices:
  Band  Nombre     λ central  Resol.  Descripción
  B2    Blue       490nm      10m→20m Azul (visible)
  B3    Green      560nm      10m→20m Verde (visible)
  B4    Red        665nm      10m→20m Rojo (visible)
  B5    RedEdge1   705nm      20m     Red-Edge 1
  B6    RedEdge2   740nm      20m     Red-Edge 2
  B7    RedEdge3   783nm      20m     Red-Edge 3
  B8    NIR        842nm      10m→20m NIR ancho
  B8A   NarrowNIR  865nm      20m     NIR estrecho (comparable a L8 B5)
  B11   SWIR1      1610nm     20m     SWIR corto (nubes, suelo, agua)
  B12   SWIR2      2190nm     20m     SWIR largo (minerales, agua profunda)

  # Índices pre-calculados (para conveniencia/reproducibilidad):
  NDVI   Vegetación (con B8)
  NDVI_8A Vegetación con B8A (comparable a MODIS/L8)
  EVI    Enhanced Vegetation Index
  NDWI   Agua superficial (McFeeters 1996)
  MNDWI  Agua modificada (Xu 2006)
  NDSI   Nieve (Dozier 1989)
  NDBI   Urbano/construido
  NDRE   Red-Edge Vegetation (exclusivo S2)
  CIre   Clorofila Red-Edge (exclusivo S2)
  BSI    Suelo desnudo
  NBR    Normalized Burn Ratio (incendios/erosión)
───────────────────────────────────────────────────────────────────────

ESTIMACIÓN DE ESPACIO (cuenca ~3,400 km²):
  Pixels a 20m: 3,400km² / (0.02km)² ≈ 8,500,000 px/banda
  Tamaño float32 sin comprimir: 8.5M × 4 bytes ≈ 34MB/banda/imagen
  Con compresión GeoTIFF DEFLATE: ~6-8MB/banda/imagen
  21 bandas × 120 meses (2015–2025) × 7MB ≈ ~17.6 GB (Drive)
  → Para reducir, se puede exportar a 30m: 21 × 120 × 3MB ≈ 7.6 GB ✓
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
SHP_CUE = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
           / "Cuenca_Chancay___Huaral.shp")

GEE_PROJECT      = "ana-chancay-huaral"
GEE_DRIVE_FOLDER = "HidroAlerta_Rasters"
SCALE_M          = 30          # 30m: ajustado para caber en ~14 GB Drive total
START_DEFAULT    = "2015-06-23" # S2A primer dato disponible
END_DEFAULT      = "2025-12-31"
CLOUD_PCT_MAX    = 20          # % de nubosidad máx. por imagen

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("S32_sentinel2")


# ─── Función principal GEE ────────────────────────────────────────────────────
def run_gee(start: str, end: str) -> None:
    """
    Envía tareas de exportación de Sentinel-2 MSI a Google Drive.
    Cada tarea genera un GeoTIFF multibanda mensual con 10 bandas crudas
    + 11 índices pre-calculados (21 bandas totales).
    """
    try:
        import ee
        import geopandas as gpd
        from shapely.geometry import mapping
        ee.Initialize(project=GEE_PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as e:
        log.error(f"GEE no disponible: {e}")
        return

    log.info("Cargando límite de cuenca Chancay-Huaral ...")
    gdf = gpd.read_file(SHP_CUE).to_crs("EPSG:4326")
    geom = ee.Geometry(mapping(gdf.geometry.iloc[0]))

    # Colección S2 SR Harmonizada (con corrección atmosférica + harmonización A/B)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_MAX)))

    # Función de preprocesamiento + índices
    def process_s2(img):
        # Factor de escala: 0-10000 → 0-1 reflectancia
        scaled = img.select(["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]).divide(10000)

        # Bandas individuales
        B2  = scaled.select("B2")   # Blue  490nm
        B3  = scaled.select("B3")   # Green 560nm
        B4  = scaled.select("B4")   # Red   665nm
        B5  = scaled.select("B5")   # RE1   705nm
        B6  = scaled.select("B6")   # RE2   740nm
        B7  = scaled.select("B7")   # RE3   783nm
        B8  = scaled.select("B8")   # NIR   842nm (ancho)
        B8A = scaled.select("B8A")  # nNIR  865nm (estrecho, comparable L8 B5)
        B11 = scaled.select("B11")  # SWIR1 1610nm
        B12 = scaled.select("B12")  # SWIR2 2190nm

        eps = 1e-9   # evitar división por cero

        # ── Índices de Vegetación ───────────────────────────────────────────
        ndvi    = B8.subtract(B4).divide(B8.add(B4).add(eps)).rename("NDVI")
        ndvi_8a = B8A.subtract(B4).divide(B8A.add(B4).add(eps)).rename("NDVI_8A")
        evi     = B8.subtract(B4).multiply(2.5).divide(
                    B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1).add(eps)
                  ).rename("EVI")
        ndre    = B8A.subtract(B5).divide(B8A.add(B5).add(eps)).rename("NDRE")
        cire    = B7.divide(B5.add(eps)).subtract(1).rename("CIre")

        # ── Índices de Agua ─────────────────────────────────────────────────
        ndwi    = B3.subtract(B8).divide(B3.add(B8).add(eps)).rename("NDWI")
        mndwi   = B3.subtract(B11).divide(B3.add(B11).add(eps)).rename("MNDWI")

        # ── Índices de Nieve y Suelo ────────────────────────────────────────
        ndsi    = B3.subtract(B11).divide(B3.add(B11).add(eps)).rename("NDSI")
        bsi     = (B11.add(B4).subtract(B8.add(B2))).divide(
                    B11.add(B4).add(B8).add(B2).add(eps)
                  ).rename("BSI")

        # ── Índices Urbanos y Quemado ───────────────────────────────────────
        ndbi    = B11.subtract(B8).divide(B11.add(B8).add(eps)).rename("NDBI")
        nbr     = B8.subtract(B12).divide(B8.add(B12).add(eps)).rename("NBR")

        return (scaled
                .addBands([ndvi, ndvi_8a, evi, ndre, cire,
                           ndwi, mndwi, ndsi, bsi, ndbi, nbr])
                .copyProperties(img, ["system:time_start"]))

    s2 = s2.map(process_s2)

    # Bandas a exportar (10 crudas + 11 índices = 21 total)
    export_bands = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
                    "NDVI","NDVI_8A","EVI","NDRE","CIre",
                    "NDWI","MNDWI","NDSI","BSI","NDBI","NBR"]

    # Iterar por mes
    months = pd.date_range(start, end, freq="MS")
    n_total = len(months)
    log.info(f"Enviando {n_total} tareas de exportación S2 a Google Drive ...")
    log.info(f"  Bandas por imagen: {len(export_bands)} | Resolución: {SCALE_M}m")
    log.info(f"  Carpeta Drive: {GEE_DRIVE_FOLDER}/S32_s2/")

    tasks_sent = 0
    for i, date in enumerate(months):
        s_date = date.strftime("%Y-%m-%d")
        e_date = (date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        year_mo = date.strftime("%Y_%m")

        composite = (s2.filterDate(s_date, e_date)
                       .median()
                       .select(export_bands)
                       .clip(geom))

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f"S32_s2_{year_mo}",
            folder=f"{GEE_DRIVE_FOLDER}/S32_s2",
            fileNamePrefix=f"S32_s2_{year_mo}",
            scale=SCALE_M,
            region=geom,
            crs="EPSG:4326",
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks_sent += 1
        if (i + 1) % 20 == 0 or i == n_total - 1:
            log.info(f"  [{i+1}/{n_total}] Tareas enviadas hasta {year_mo}")

    log.info(f"✅ {tasks_sent} tareas Sentinel-2 enviadas.")
    log.info("   Progresa en: https://code.earthengine.google.com/  (pestaña Tasks)")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Script 32 — Exportar Sentinel-2 MSI a Google Drive"
    )
    parser.add_argument("--start", default=START_DEFAULT)
    parser.add_argument("--end",   default=END_DEFAULT)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SCRIPT 32: Sentinel-2 MSI — HidroAlerta Chancay-Huaral")
    log.info(f"  Satélite: Sentinel-2A/B | Sensor: MSI | Resolución export: {SCALE_M}m")
    log.info(f"  Colección: S2_SR_HARMONIZED | Nubosidad máx: {CLOUD_PCT_MAX}%")
    log.info(f"  Período: {args.start} → {args.end}")
    log.info(f"  Drive: {GEE_DRIVE_FOLDER}/S32_s2/")
    log.info("=" * 70)

    run_gee(args.start, args.end)


if __name__ == "__main__":
    main()
