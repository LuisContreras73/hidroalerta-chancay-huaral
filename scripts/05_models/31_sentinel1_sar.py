#!/usr/bin/env python3
"""
SCRIPT 31: Sentinel-1 SAR — HidroAlerta Chancay-Huaral
=========================================================
Satélite : Copernicus Sentinel-1A / 1B (ESA)
Sensor   : C-SAR (C-band Synthetic Aperture Radar, 5.6 GHz)
Modo     : IW (Interferometric Wide Swath, principal modo global)
Polariz. : VV + VH
Cobertura: Abril 2014 – presente
Revisita : ~6 días (combinando S1A y S1B)

Resolución nativa : 10m (GRD procesado a 10m en GEE)
Resolución export : 30m  (resampleado para consistencia con L8/S2 y menor peso)

FÍSICA DEL SAR:
- Mide retrodispersión del radar (señal enviada y recibida de vuelta).
- Unidades: dB (decibelios, ya calibradas en GEE con colección S1_GRD).
- VV  : verticalmente emitido, verticalmente recibido → sensible a estructuras,
        superficie del agua (muy baja retrodispersión → agua aparece oscura).
- VH  : verticalmente emitido, horizontalmente recibido → sensible a vegetación
        y volumen, útil para humedad del suelo.
- VENTAJA CLAVE: el SAR penetra nubes y lluvia → siempre tiene datos,
  incluso en época húmeda (Nov–Mar) cuando ópticos (MODIS, S2, L8) fallan.

DIFERENCIA CON ÓPTICOS:
- No existe "color natural" porque no mide luz visible.
- Composición falsa-color estándar: VV→R, VH→G, VV/VH→B
- Los índices son puramente matemáticos (ratio, diferencia en dB), no tienen
  equivalente directo con NDVI/NDWI ópticos.

BANDAS EXPORTADAS (compuesto mensual = mediana):
─────────────────────────────────────────────────────────────────────
  Band  Descripción                    Fórmula      Rango típico (dB)
  VV    Retrodispersión VV             raw          -25 a -5 dB
  VH    Retrodispersión VH             raw          -30 a -15 dB
  VV_VH Ratio de polarizaciones        VV − VH       8 a 15 dB
  RVI   Radar Vegetation Index         4*VH/(VV+VH) 0 a 1 (linear)
  dpol  Diferencia de polarizaciones   VH − VV      negativo (en dB)
─────────────────────────────────────────────────────────────────────

ÍNDICES — NOTA SOBRE DIFERENCIAS ENTRE SENSORES:
  Los índices SAR (RVI, dpol) NO tienen equivalente en sensores ópticos.
  No existe forma de comparar directamente VV/VH con NDVI óptico.
  Sí existe correlación estadística, pero son física completamente diferente.

ESTIMACIÓN DE ESPACIO (cuenca ~3,400 km²):
  Pixels a 30m: 3,400km² / (0.03km)² ≈ 3,778,000 px/banda
  Tamaño por banda por imagen: ~4MB (float32 sin comprimir), ~1.5MB comprimido
  5 bandas × 130 meses (2014–2025) × 1.5MB ≈ 975 MB en Drive
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

GEE_PROJECT   = "ana-chancay-huaral"
GEE_DRIVE_FOLDER = "HidroAlerta_Rasters"
SCALE_M       = 30          # metros de resolución de exportación
START_DEFAULT = "2014-04-01"  # S1 comienza abril 2014
END_DEFAULT   = "2025-12-31"

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("S31_sentinel1")


# ─── Función principal GEE ────────────────────────────────────────────────────
def run_gee(start: str, end: str) -> None:
    """
    Envía tareas de exportación de Sentinel-1 SAR a Google Drive.
    Cada tarea genera un GeoTIFF multibanda mensual con:
      Band 1: VV   (retrodispersión vertical, dB)
      Band 2: VH   (retrodispersión horizontal, dB)
      Band 3: VV_VH (ratio VV - VH en dB, indicador de superficie/volumen)
      Band 4: RVI  (Radar Vegetation Index = 4*VH_lin / (VV_lin + VH_lin))
      Band 5: dpol (VH - VV en dB, negativo → relacionado con humedad suelo)
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

    # Cargar geometría de la cuenca
    log.info("Cargando límite de cuenca Chancay-Huaral ...")
    gdf = gpd.read_file(SHP_CUE).to_crs("EPSG:4326")
    geom = ee.Geometry(mapping(gdf.geometry.iloc[0]))

    # Colección Sentinel-1 GRD (Ground Range Detected)
    # Filtro: modo IW (más común en continentes), VV+VH
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))

    # Función de cálculo de índices SAR
    def add_sar_indices(img):
        vv = img.select("VV")   # en dB
        vh = img.select("VH")   # en dB

        # Convertir dB → lineal para RVI
        vv_lin = ee.Image(10).pow(vv.divide(10))
        vh_lin = ee.Image(10).pow(vh.divide(10))

        vv_vh = vv.subtract(vh).rename("VV_VH")          # ratio dB
        rvi   = vh_lin.multiply(4).divide(
                    vv_lin.add(vh_lin).add(1e-9)
                ).rename("RVI")                           # 0-1
        dpol  = vh.subtract(vv).rename("dpol")           # siempre negativo

        return img.addBands([vv_vh, rvi, dpol])

    s1 = s1.map(add_sar_indices)

    # Iterar por mes y enviar tarea a Drive
    months = pd.date_range(start, end, freq="MS")
    n_total = len(months)
    log.info(f"Enviando {n_total} tareas de exportación SAR a Google Drive ...")
    log.info(f"  Carpeta Drive: {GEE_DRIVE_FOLDER}/S31_sar/")

    tasks_sent = 0
    for i, date in enumerate(months):
        s_date = date.strftime("%Y-%m-%d")
        e_date = (date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        year_mo = date.strftime("%Y_%m")

        # Mediana mensual → robusto a ruido de imagen única
        composite = (s1.filterDate(s_date, e_date)
                       .median()
                       .select(["VV", "VH", "VV_VH", "RVI", "dpol"])
                       .clip(geom))

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f"S31_sar_{year_mo}",
            folder=f"{GEE_DRIVE_FOLDER}/S31_sar",
            fileNamePrefix=f"S31_sar_{year_mo}",
            scale=SCALE_M,
            region=geom,
            crs="EPSG:4326",
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks_sent += 1
        if (i + 1) % 20 == 0 or i == n_total - 1:
            log.info(f"  [{i+1}/{n_total}] Últimas tareas enviadas hasta {year_mo}")

    log.info(f"✅ {tasks_sent} tareas SAR enviadas exitosamente.")
    log.info("   Progresa en: https://code.earthengine.google.com/  (pestaña Tasks)")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Script 31 — Exportar Sentinel-1 SAR a Google Drive"
    )
    parser.add_argument("--start", default=START_DEFAULT)
    parser.add_argument("--end",   default=END_DEFAULT)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SCRIPT 31: Sentinel-1 SAR — HidroAlerta Chancay-Huaral")
    log.info(f"  Satélite: Sentinel-1A/B | Sensor: C-SAR IW | Resolución export: {SCALE_M}m")
    log.info(f"  Período: {args.start} → {args.end}")
    log.info(f"  Drive: {GEE_DRIVE_FOLDER}/S31_sar/")
    log.info("=" * 70)

    run_gee(args.start, args.end)


if __name__ == "__main__":
    main()
