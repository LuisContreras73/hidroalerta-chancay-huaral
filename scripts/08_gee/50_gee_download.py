#!/usr/bin/env python3
"""
Script 50: Descarga directa GEE → local (sin Google Drive)

Descarga todos los rasters GEE del proyecto HidroAlerta directamente al disco
usando la Python Earth Engine API. No requiere Drive como intermediario.

Hitos de descarga (orden de prioridad):
  HITO 1 🔴 — CHIRPS, MOD10A1 snow                    (~1.4 GB)
  HITO 2 🟡 — MODIS NDVI/EVI/LST/LSWI/ET, SMAP, S1   (~7 GB)
  HITO 3 🟢 — Landsat, ESA WorldCover, JRC water      (~5 GB)
  S2      ⭐ — Sentinel-2 30m (año a año, más pesado)  (~4-5 GB)

Uso:
    python scripts/50_gee_download.py --hito 1
    python scripts/50_gee_download.py --hito 2
    python scripts/50_gee_download.py --hito 3
    python scripts/50_gee_download.py --source chirps   # solo una fuente
    python scripts/50_gee_download.py --source s2 --year 2017
"""

import argparse
import json
import logging
import time
from pathlib import Path

import ee
import geemap
import requests
from tqdm import tqdm

# ─── Configuración ────────────────────────────────────────────────────────────
GEE_PROJECT  = "ana-chancay-huaral"
DRIVE_FOLDER = "GEE_HidroAlerta"

# Bbox cuenca con buffer 0.1° para píxeles de borde (modo legacy)
BBOX = [-77.45, -12.05, -76.25, -10.65]
REGION     = None  # set in init_gee()
BASIN_GEOM = None  # ee.Geometry del polígono cuenca + buffer — set in init_gee() --native

ROOT    = Path(__file__).parent.parent
GEE_DIR = ROOT / "data/raw/gee"

BASIN_SHP = ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite/Cuenca_Chancay___Huaral.shp"
BUFFER_M  = 500   # metros de buffer alrededor del polígono de cuenca

USE_NATIVE = False  # activado con --native; cambia scale + clip a cuenca

# Resoluciones legacy (modo original — bbox completo)
SCALE = {
    "chirps":    5566,
    "snow":      500,
    "vegetation":250,
    "thermal":   1000,
    "lswi":      500,
    "et":        500,
    "smap":      10000,
    "s1":        500,
    "s2":        500,
    "landsat":   250,
    "landcover": 100,
    "jrc":       100,
}

# Resoluciones nativas — con clip al polígono de cuenca (--native)
# S1/S2 nativos 10m → 30m de compromiso (10m = 91 GB incluso con clip)
SCALE_NATIVE = {
    "chirps":    5566,   # nativo 0.05° — sin cambio
    "snow":      500,    # MOD10A1 nativo 500m — sin cambio
    "vegetation":250,    # MOD13Q1 nativo 250m — sin cambio
    "thermal":   1000,   # MOD11A1 nativo 1km — sin cambio
    "lswi":      500,    # MOD09GA nativo 500m — sin cambio
    "et":        500,    # MOD16A2 nativo 500m — sin cambio
    "smap":      10000,  # SMAP nativo ~9km — sin cambio
    "s1":        500,    # Sentinel-1: 30m con clip = 995 MB/año >> límite (D044). Clip a 500m: 3.6 MB/año OK.
    "s2":        500,    # Sentinel-2: igual (D044). Clip 500m_clip = 83% menos que legacy bbox.
    "landsat":   30,     # Landsat nativo 30m ✓
    "landcover": 30,     # ESA WorldCover 10m nativo; 10m=130MB>límite → 30m clip (~2MB OK, D044)
    "jrc":       30,     # JRC nativo 30m ✓
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("gee_download")


# ─── Inicialización GEE ───────────────────────────────────────────────────────

def _load_basin_geom() -> "ee.Geometry":
    """Lee el shapefile de la cuenca, aplica buffer y lo convierte a ee.Geometry."""
    import geopandas as gpd
    from shapely.geometry import mapping

    basin = gpd.read_file(str(BASIN_SHP)).to_crs(epsg=4326)
    basin_utm = basin.to_crs(epsg=32718)
    basin_buf = basin_utm.buffer(BUFFER_M).to_crs(epsg=4326)
    geom = basin_buf.geometry.union_all().simplify(0.001)  # ~100m tolerance
    return ee.Geometry(mapping(geom))


def init_gee(native: bool = False):
    global REGION, BASIN_GEOM, USE_NATIVE
    try:
        ee.Initialize(project=GEE_PROJECT)
        REGION = ee.Geometry.Rectangle(BBOX)
        USE_NATIVE = native
        if native:
            BASIN_GEOM = _load_basin_geom()
            log.info(f"GEE inicializado — project: {GEE_PROJECT} | modo NATIVE (clip cuenca)")
        else:
            log.info(f"GEE inicializado — project: {GEE_PROJECT} | modo legacy bbox")
    except Exception as e:
        log.error(f"Error GEE: {e}")
        log.error("Ejecuta: python -c \"import ee; ee.Authenticate()\" y reintenta.")
        raise


def _sc(key: str) -> int:
    """Devuelve la resolución correcta según USE_NATIVE."""
    return SCALE_NATIVE[key] if USE_NATIVE else SCALE[key]


def _suffix(key: str) -> str:
    """Sufijo de nombre de archivo con la resolución activa.
    En modo --native con misma resolución que legacy (MODIS, CHIRPS, SMAP),
    añade '_clip' para distinguir del archivo legacy sin recorte de cuenca.
    """
    sc = _sc(key)
    base = f"{sc}m"
    if USE_NATIVE and sc == SCALE[key]:
        base += "_clip"
    return base


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _out(subfolder: str, filename: str) -> Path:
    p = GEE_DIR / subfolder
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def _already_done(path: Path) -> bool:
    if path.exists() and path.stat().st_size > 1000:
        log.info(f"  Ya existe: {path.name} — saltando")
        return True
    return False


def _write_basic_sidecar(out_path: Path, scale: int, extra: dict | None = None):
    """Escribe sidecar JSON mínimo para archivos sin sidecar dedicado (CHIRPS, MODIS, SMAP)."""
    json_path = out_path.with_suffix(".json")
    if json_path.exists():
        return
    meta = {
        "filename":    out_path.name,
        "scale_m":     scale,
        "native":      USE_NATIVE,
        "clip_region": "basin_polygon_500m_buffer (EPSG:4326)" if USE_NATIVE else "bbox_1.1x0.9deg",
        "generated_by": "50_gee_download.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    json_path.write_text(json.dumps(meta, indent=2))


def _export_image(image: ee.Image, out_path: Path, scale: int, desc: str = "",
                  sidecar_extra: dict | None = None):
    """Descarga un ee.Image al disco usando geemap.
    En modo --native: usa region=BASIN_GEOM (restringe bbox de descarga al polígono).
    NO aplica image.clip() — GEE falla al transformar bordes de tiles MODIS sinusoidales (D038).
    El region= ya recorta el GeoTIFF al bbox del polígono de cuenca (~70% menos píxeles).
    Escribe sidecar JSON básico al finalizar si no existe.
    """
    if _already_done(out_path):
        return
    log.info(f"  Descargando {desc} → {out_path.name}")
    t0 = time.time()

    region = BASIN_GEOM if (USE_NATIVE and BASIN_GEOM is not None) else REGION

    try:
        geemap.ee_export_image(
            image,
            filename=str(out_path),
            scale=scale,
            region=region,
            crs="EPSG:4326",
            file_per_band=False,
        )
        elapsed = time.time() - t0
        if not out_path.exists() or out_path.stat().st_size < 500:
            size = out_path.stat().st_size if out_path.exists() else 0
            raise RuntimeError(
                f"Descarga falló silenciosamente ({size} bytes). "
                "Probable causa: request > 50 MB. Ajustar scale o chunking."
            )
        size_mb = out_path.stat().st_size / 1e6
        log.info(f"  OK {out_path.name} ({size_mb:.1f} MB, {elapsed:.0f}s)")
        _write_basic_sidecar(out_path, scale, sidecar_extra)
    except Exception as ex:
        log.error(f"  ERROR descargando {out_path.name}: {ex}")
        raise


def _col_to_stack(col: ee.ImageCollection, var_name: str) -> ee.Image:
    """Convierte una ImageCollection a imagen multi-banda. Banda = VARNAME__YYYY_MM_dd."""
    def rename_band(img):
        date = img.date().format("YYYY_MM_dd")
        return img.rename(ee.String(var_name).cat("__").cat(date))
    return col.map(rename_band).toBands()


def _multi_var_stack(col: ee.ImageCollection, bands: list[str]) -> ee.Image:
    """Multi-variable stack: VARNAME__YYYY_MM_dd por cada banda."""
    def rename_bands(img):
        date = img.date().format("YYYY_MM_dd")
        renamed = [ee.String(b).cat("__").cat(date) for b in bands]
        return img.select(bands).rename(renamed)
    return col.map(rename_bands).toBands()


def _submit_drive_task(image: ee.Image, desc: str, scale: int):
    """Para datasets muy grandes: envía tarea a Drive (S2 full res, JRC full res)."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=desc,
        folder=DRIVE_FOLDER,
        scale=scale,
        region=REGION,
        crs="EPSG:4326",
        maxPixels=1e12,
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info(f"  ▶ Tarea Drive enviada: {desc}  (revisar en GEE Tasks panel)")
    return task


# ═══════════════════════════════════════════════════════════════════════════════
# HITO 1 — Descargas críticas y pequeñas
# ═══════════════════════════════════════════════════════════════════════════════

def download_chirps(years_range=(1981, 2025)):
    """GEE0: CHIRPS Daily precipitation 0.05° — 1 archivo por año (365 bandas).
    Convención: banda i → día i del año (desde Jan 1). Fechas reconstruibles
    con pd.date_range(f'{yr}-01-01', periods=n_bands, freq='D').
    """
    scale  = _sc("chirps")
    suffix = _suffix("chirps")  # "5566m" legacy | "5566m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── CHIRPS (GEE0) {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")

    for yr in range(years_range[0], years_range[1] + 1):
        fname = f"chirps_precip_{yr}{fsufx}.tif"
        out = _out("chirps", fname)
        if _already_done(out):
            continue
        chunk = col.filterDate(f"{yr}-01-01", f"{yr}-12-31")
        stack = chunk.toBands()  # banda i = día i del año
        _export_image(stack, out, scale, f"CHIRPS {yr}", sidecar_extra={
            "collection": "UCSB-CHG/CHIRPS/DAILY",
            "yr": yr, "units": "mm/día",
            "band_convention": "banda i (0-based) = día i del año; "
                               "pd.date_range(f'{yr}-01-01', periods=n_bands, freq='D')",
        })


def download_snow_modis(years_range=(2000, 2025)):
    """GEE1: MOD10A1 NDSI snow cover 500m — chunks bimestrales.

    Por qué 2 meses y no anual:
      bbox 500m = ~81K px × 2 bandas × 365 dias × 4B ≈ 118 MB/año.
      Límite API getDownloadURL = 50 MB → chunk bimestral ≈ 39 MB, OK.
      Genera 6 archivos/año: m0102, m0304, m0506, m0708, m0910, m1112.
    """
    scale  = _sc("snow")
    suffix = _suffix("snow")  # "500m" legacy | "500m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── MOD10A1 Snow (GEE1) — bimestral {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    col = ee.ImageCollection("MODIS/061/MOD10A1").select(["NDSI_Snow_Cover", "NDSI"])

    for yr in range(years_range[0], years_range[1] + 1):
        for m0 in [1, 3, 5, 7, 9, 11]:
            m1 = m0 + 1
            next_m = m1 + 1 if m1 < 12 else 1
            next_y = yr if m1 < 12 else yr + 1
            date_start = f"{yr}-{m0:02d}-01"
            date_end   = f"{next_y}-{next_m:02d}-01"
            fname = f"mod10a1_snow_{yr}_m{m0:02d}{m1:02d}{fsufx}.tif"
            out = _out("snow", fname)
            if _already_done(out):
                continue
            chunk = col.filterDate(date_start, date_end)
            stack = chunk.toBands()
            _export_image(stack, out, scale, f"MOD10A1 {yr}-m{m0:02d}/{m1:02d}",
                          sidecar_extra={
                              "collection": "MODIS/061/MOD10A1",
                              "yr": yr, "months": f"{m0:02d}-{m1:02d}",
                              "band_convention": "alternadas: par=NDSI_Snow_Cover [0-100], impar=NDSI [-1,1]",
                              "qa_note": "snow_cover >100 son códigos especiales (200=lago congelado). Enmascarar → NaN.",
                          })


# ═══════════════════════════════════════════════════════════════════════════════
# HITO 2 — Datos medianos
# ═══════════════════════════════════════════════════════════════════════════════

def download_ndvi_modis(years_range=(2000, 2025)):
    """GEE2a: MOD13Q1 NDVI+EVI 250m 16-day — chunks semestrales.

    Por qué semestral:
      bbox 250m = ~331K px × 2 bandas × 23 composites × 4B ≈ 61 MB/año > 50 MB.
      H1 (ene-jun) ≈ 11 composites × 2 × 331K × 4B ≈ 29 MB — OK.
      Genera 2 archivos/año: h1, h2.
    """
    scale  = _sc("vegetation")
    suffix = _suffix("vegetation")  # "250m" legacy | "250m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── MOD13Q1 NDVI/EVI (GEE2a) — semestral {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .select(["NDVI", "EVI"]))

    for yr in range(years_range[0], years_range[1] + 1):
        for half, (d0, d1) in enumerate([
            (f"{yr}-01-01", f"{yr}-07-01"),
            (f"{yr}-07-01", f"{yr+1}-01-01"),
        ], start=1):
            fname = f"mod13q1_ndvi_evi_{yr}_h{half}{fsufx}.tif"
            out = _out("vegetation", fname)
            if _already_done(out):
                continue
            chunk = col.filterDate(d0, d1)
            stack = _multi_var_stack(chunk, ["NDVI", "EVI"])
            _export_image(stack, out, scale, f"MOD13Q1 {yr}-H{half}",
                          sidecar_extra={
                              "collection": "MODIS/061/MOD13Q1",
                              "yr": yr, "half": half,
                              "band_convention": "alternadas: par=NDVI, impar=EVI por composite 16-días",
                              "scale_factor": 0.0001,
                              "qa_note": "valores raw ×0.0001 → índice [-1,1]; fill fuera de [-10000,10000] → NaN",
                          })


def download_lst_modis(years_range=(2000, 2025)):
    """GEE2b: MOD11A1 LST día+noche 1km — chunks semestrales.

    Por qué semestral:
      bbox 1km = ~20.7K px × 2 bandas × 365 días × 4B ≈ 60 MB/año > 50 MB.
      Semestre ≈ 182 días × 2 × 20.7K × 4B ≈ 30 MB — OK.
      Genera 2 archivos/año: h1, h2.
    """
    scale  = _sc("thermal")
    suffix = _suffix("thermal")  # "1000m" legacy | "1000m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── MOD11A1 LST día+noche (GEE2b) — semestral {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    col = (ee.ImageCollection("MODIS/061/MOD11A1")
           .select(["LST_Day_1km", "LST_Night_1km"]))

    for yr in range(years_range[0], years_range[1] + 1):
        for half, (d0, d1) in enumerate([
            (f"{yr}-01-01", f"{yr}-07-01"),
            (f"{yr}-07-01", f"{yr+1}-01-01"),
        ], start=1):
            fname = f"mod11a1_lst_{yr}_h{half}{fsufx}.tif"
            out = _out("thermal", fname)
            if _already_done(out):
                continue
            chunk = col.filterDate(d0, d1)
            stack = _multi_var_stack(chunk, ["LST_Day_1km", "LST_Night_1km"])
            _export_image(stack, out, scale, f"MOD11A1 LST {yr}-H{half}",
                          sidecar_extra={
                              "collection": "MODIS/061/MOD11A1",
                              "yr": yr, "half": half,
                              "band_convention": "alternadas: par=LST_Day_1km, impar=LST_Night_1km por día",
                              "scale_factor": 0.02,
                              "units": "K×0.02 → K real; -273.15 → °C",
                              "qa_note": "fill=0 → NaN; rango físico 7000-17000 raw (140-340 K)",
                          })


def download_lswi_modis(years_range=(2000, 2025)):
    """GEE2c: MOD09GA LSWI 500m — chunks bimestrales.

    Por qué bimestral:
      bbox 500m = ~83K px × 1 banda × 365 días × 4B ≈ 121 MB/año > 50 MB.
      Bimestre ≈ 61 días × 1 × 83K × 4B ≈ 20 MB — OK.
      Genera 6 archivos/año: m0102, m0304, m0506, m0708, m0910, m1112.
    """
    scale  = _sc("lswi")
    suffix = _suffix("lswi")  # "500m" legacy | "500m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── MOD09GA LSWI (GEE2c) — bimestral {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    raw = ee.ImageCollection("MODIS/061/MOD09GA")

    def compute_lswi(img):
        nir  = img.select("sur_refl_b02")
        swir = img.select("sur_refl_b06")
        lswi = nir.subtract(swir).divide(nir.add(swir)).rename("LSWI")
        return lswi.copyProperties(img, ["system:time_start"])

    col = raw.map(compute_lswi)

    for yr in range(years_range[0], years_range[1] + 1):
        for m0 in [1, 3, 5, 7, 9, 11]:
            m1 = m0 + 1
            next_m = m1 + 1 if m1 < 12 else 1
            next_y = yr if m1 < 12 else yr + 1
            date_start = f"{yr}-{m0:02d}-01"
            date_end   = f"{next_y}-{next_m:02d}-01"
            fname = f"mod09ga_lswi_{yr}_m{m0:02d}{m1:02d}{fsufx}.tif"
            out = _out("vegetation", fname)
            if _already_done(out):
                continue
            chunk = col.filterDate(date_start, date_end)
            stack = _col_to_stack(chunk, "LSWI")
            _export_image(stack, out, scale, f"MOD09GA LSWI {yr}-m{m0:02d}/{m1:02d}",
                          sidecar_extra={
                              "collection": "MODIS/061/MOD09GA",
                              "yr": yr, "months": f"{m0:02d}-{m1:02d}",
                              "band_convention": "1 banda por día: LSWI = (NIR-SWIR)/(NIR+SWIR)",
                              "units": "float [-1, 1]",
                          })


def download_et_modis(years_range=(2001, 2025)):
    """GEE6: MOD16A2GF ET+PET 500m 8-day — por año.

    Por qué anual y no bienal:
      bbox 500m = ~83K px × 2 bandas × 46 composites × 4B ≈ 30.5 MB/año — OK.
      2 años = 61 MB → excede 50 MB. Usar 1 año por archivo.
    """
    scale  = _sc("et")
    suffix = _suffix("et")  # "500m" legacy | "500m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── MOD16A2GF ET+PET (GEE6) — anual {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    col = (ee.ImageCollection("MODIS/061/MOD16A2GF")
           .select(["ET", "PET"]))

    for yr in range(years_range[0], years_range[1] + 1):
        fname = f"mod16a2_et_pet_{yr}{fsufx}.tif"
        out = _out("et", fname)
        if _already_done(out):
            continue
        chunk = col.filterDate(f"{yr}-01-01", f"{yr+1}-01-01")
        stack = _multi_var_stack(chunk, ["ET", "PET"])
        _export_image(stack, out, scale, f"MOD16A2 ET {yr}", sidecar_extra={
            "collection": "MODIS/061/MOD16A2GF",
            "yr": yr,
            "band_convention": "alternadas: par=ET, impar=PET por composite 8-días",
            "scale_factor": 0.1,
            "units": "kg/m²/8días × 0.1 → mm/8días; ÷8 → mm/día",
            "qa_note": "fill=32767 → NaN; rango físico ET [0,300] mm/8d",
        })


def download_smap(years_range=(2015, 2025)):
    """GEE8: SMAP SPL4SMGP/008 ~9km — datos 3-horarios raw por mes.

    Por qué raw 3-horarios (no diario pre-agregado):
      ee.ImageCollection creada dentro de ee.List.map() no filtra
      correctamente server-side (filterDate retorna imagen vacía, 0 bandas).
      Fix: loop Python client-side por mes, sin server-side map.
      Descarga datos 3-horarios raw (8 imgs/día) → agregar a diario en
      post-proceso promediando bandas en grupos de 8.
      Mes típico: 31d × 8imgs × 3 bandas = 744 bandas < límite 1024.
      Tamaño: 744 × ~210px × 4B = ~600 KB/mes — trivial.
      Genera 12 archivos/año: smap_soil_{yr}_{mo:02d}_3h.tif
    """
    scale  = _sc("smap")
    suffix = _suffix("smap")  # "10000m" legacy | "10000m_clip" native
    fsufx  = f"_{suffix}" if USE_NATIVE else ""
    log.info(f"── SMAP SPL4SMGP/008 (GEE8) — mensual 3h {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")
    SMAP_BANDS = ["sm_surface", "sm_rootzone", "sm_profile"]

    for yr in range(years_range[0], years_range[1] + 1):
        for mo in range(1, 13):
            fname = f"smap_soil_{yr}_{mo:02d}_3h{fsufx}.tif"
            out = _out("smap", fname)
            if _already_done(out):
                continue
            start_mo = f"{yr}-{mo:02d}-01"
            end_mo = f"{yr}-{mo+1:02d}-01" if mo < 12 else f"{yr+1}-01-01"
            col = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
                   .filterDate(start_mo, end_mo)
                   .select(SMAP_BANDS))
            n_imgs = col.size().getInfo()
            if n_imgs == 0:
                log.warning(f"  Sin datos SMAP {yr}-{mo:02d} — saltando")
                continue
            if n_imgs > 1020:
                log.warning(f"  SMAP {yr}-{mo:02d}: {n_imgs} imgs > 1020 — "
                            "descargando primera mitad")
                col = col.limit(1020)
            stack = _multi_var_stack(col, SMAP_BANDS)
            _export_image(stack, out, scale, f"SMAP {yr}-{mo:02d}", sidecar_extra={
                "collection": "NASA/SMAP/SPL4SMGP/008",
                "yr": yr, "month": mo,
                "band_vars": SMAP_BANDS,
                "band_convention": "n_imgs × 3 bandas: cada grupo de 3 = [sm_surface, sm_rootzone, sm_profile] "
                                   "a la misma hora (3h). Agregar diario: nanmean(bandas[i::8]) para var i.",
                "units": "m³/m³ [0.02-0.50]",
                "n_obs_per_day": 8,
            })


def download_sentinel1(years_range=(2014, 2025)):
    """GEE10: Sentinel-1 SAR VV+VH compositos mensuales — 1 archivo/año.

    Modo legacy: scale=500m (bbox completo).
    Modo native: scale=30m con clip al polígono de cuenca (~10 GB total).
      S1 nativo 10m con clip → 91 GB; 30m con clip → 10 GB — compromiso práctico.
    Genera 1 archivo/año: s1_vv_vh_ratio_{yr}_{scale}m.tif
    """
    scale  = _sc("s1")
    suffix = _suffix("s1")
    log.info(f"── Sentinel-1 SAR (GEE10) — {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")

    for yr in range(years_range[0], years_range[1] + 1):
        fname    = f"s1_vv_vh_ratio_{yr}_{suffix}.tif"
        out      = _out("sentinel1", fname)
        json_out = out.with_suffix(".json")
        tif_done  = out.exists() and out.stat().st_size > 1000
        json_done = json_out.exists() and json_out.stat().st_size > 10
        if tif_done and json_done:
            log.info(f"  Ya existe: {fname} + JSON — saltando")
            continue

        filter_region = BASIN_GEOM if (USE_NATIVE and BASIN_GEOM) else REGION
        valid_imgs, valid_months = [], []
        for mo in range(1, 13):
            start  = f"{yr}-{mo:02d}-01"
            end_mo = mo + 1 if mo < 12 else 1
            end_yr = yr if mo < 12 else yr + 1
            end    = f"{end_yr}-{end_mo:02d}-01"
            col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterDate(start, end)
                   .filterBounds(filter_region)
                   .filter(ee.Filter.eq("instrumentMode", "IW"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                   .select(["VV", "VH"]))
            n = col.size().getInfo()
            if n == 0:
                continue
            mean_img = col.mean()
            ratio = mean_img.select("VH").subtract(mean_img.select("VV")).rename("RATIO")
            valid_imgs.append(
                mean_img.addBands(ratio).set("system:time_start", ee.Date(start).millis())
            )
            valid_months.append(mo)

        if not valid_imgs:
            log.warning(f"  Sin datos S1 IW VV+VH para {yr} — saltando")
            continue
        log.info(f"  {yr}: {len(valid_months)}/12 meses con datos S1")

        if not tif_done:
            col_yr = ee.ImageCollection(valid_imgs)
            stack  = _multi_var_stack(col_yr, ["VV", "VH", "RATIO"])
            _export_image(stack, out, scale, f"Sentinel-1 {yr}")

        if not json_done:
            meta = {
                "yr": yr, "valid_months": valid_months,
                "n_bands": len(valid_months) * 3,
                "band_vars": ["VV", "VH", "RATIO"],
                "scale_m": scale, "native": USE_NATIVE,
                "convention": "band 3*i → VV, 3*i+1 → VH, 3*i+2 → RATIO for month valid_months[i]",
            }
            json_out.write_text(json.dumps(meta, indent=2))
            log.info(f"  Sidecar JSON escrito: {json_out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sentinel-2 — directo por mes (30m, más pesado — progreso lento)
# ═══════════════════════════════════════════════════════════════════════════════

def download_sentinel2(years_range=(2017, 2025)):
    """GEE9: Sentinel-2 MSI NDVI/NDWI/NDSI — 1 archivo/año.

    Modo legacy: scale=500m (bbox completo).
    Modo native: scale=30m con clip al polígono de cuenca (~3.7 GB total).
    Estructura: meses válidos × 3 índices (NDVI_S2/NDWI_S2/NDSI_S2).
    """
    scale  = _sc("s2")
    suffix = _suffix("s2")
    log.info(f"── Sentinel-2 MSI (GEE9) — {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")

    def add_indices(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI_S2")
        ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI_S2")
        ndsi = img.normalizedDifference(["B3", "B11"]).rename("NDSI_S2")
        return ee.Image.cat([ndvi, ndwi, ndsi]).copyProperties(img, ["system:time_start"])

    filter_region = BASIN_GEOM if (USE_NATIVE and BASIN_GEOM) else REGION
    for yr in range(years_range[0], years_range[1] + 1):
        fname = f"s2_ndvi_ndwi_ndsi_{yr}_{suffix}.tif"
        out   = _out("sentinel2", fname)
        if _already_done(out):
            continue

        valid_imgs, valid_months = [], []
        for mo in range(1, 13):
            start_mo = f"{yr}-{mo:02d}-01"
            end_mo_n = mo + 1 if mo < 12 else 1
            end_yr_n = yr if mo < 12 else yr + 1
            end_str  = f"{end_yr_n}-{end_mo_n:02d}-01"
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterDate(start_mo, end_str)
                   .filterBounds(filter_region)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
                   .select(["B3", "B4", "B8", "B11"]))
            n = col.size().getInfo()
            if n == 0:
                continue
            composite = col.map(add_indices).median()
            valid_imgs.append(composite.set("system:time_start", ee.Date(start_mo).millis()))
            valid_months.append(mo)

        if not valid_imgs:
            log.warning(f"  Sin datos S2 (<60% nube) para {yr} — saltando")
            continue
        log.info(f"  {yr}: {len(valid_imgs)}/12 meses con datos S2")
        col_yr = ee.ImageCollection(valid_imgs)
        stack  = _multi_var_stack(col_yr, ["NDVI_S2", "NDWI_S2", "NDSI_S2"])
        _export_image(stack, out, scale, f"Sentinel-2 {yr}")
        meta = {
            "yr": yr, "valid_months": valid_months,
            "n_bands": len(valid_imgs) * 3,
            "band_vars": ["NDVI_S2", "NDWI_S2", "NDSI_S2"],
            "scale_m": scale, "native": USE_NATIVE,
            "convention": "band 3*i → NDVI_S2, 3*i+1 → NDWI_S2, 3*i+2 → NDSI_S2 for month valid_months[i]",
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# HITO 3 — Datos grandes y estáticos
# ═══════════════════════════════════════════════════════════════════════════════

def download_landsat(years_range=(1984, 2025)):
    """GEE3: Landsat 5/7/8/9 NDVI+NDWI+MNDWI mensual — chunks semestrales.

    Modo legacy: scale=250m (bbox completo, ~1.7 GB total).
    Modo native: scale=30m con clip al polígono de cuenca (~20 GB total).
      Landsat nativo 30m; semestral sigue siendo necesario para mantenerse
      bajo el límite de 50 MB/request incluso con el clip.
    Genera 2 archivos/año: {yr}_h{1|2}_{scale}m.tif
    """
    scale  = _sc("landsat")
    suffix = _suffix("landsat")
    log.info(f"── Landsat (GEE3) — semestral {suffix} {'(clip cuenca)' if USE_NATIVE else ''} ──")

    def get_collection(yr):
        if yr >= 2014:
            col_id = "LANDSAT/LC08/C02/T1_L2"
            bands  = ["SR_B3", "SR_B4", "SR_B5", "SR_B6"]  # Green, Red, NIR, SWIR1
        elif yr >= 2000:
            col_id = "LANDSAT/LE07/C02/T1_L2"
            bands  = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
        else:
            col_id = "LANDSAT/LT05/C02/T1_L2"
            bands  = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
        return col_id, bands

    def add_indices(img):
        ndvi  = img.normalizedDifference(["NIR", "Red"]).rename("NDVI_L")
        ndwi  = img.normalizedDifference(["Green", "NIR"]).rename("NDWI_L")
        mndwi = img.normalizedDifference(["Green", "SWIR1"]).rename("MNDWI_L")
        return ee.Image.cat([ndvi, ndwi, mndwi]).copyProperties(img, ["system:time_start"])

    HALVES = [
        (1,  (1,  7)),   # H1: ene-jun
        (2,  (7, 13)),   # H2: jul-dic (end_mo=13 → next year jan)
    ]

    col_id, src_bands = None, None  # set per year
    filter_region = BASIN_GEOM if (USE_NATIVE and BASIN_GEOM) else REGION

    for yr in range(years_range[0], years_range[1] + 1):
        col_id, src_bands = get_collection(yr)
        for half, (mo_start, mo_end) in HALVES:
            fname    = f"landsat_ndvi_ndwi_{yr}_h{half}_{suffix}.tif"
            out      = _out("water_land", fname)
            json_out = out.with_suffix(".json")
            tif_done  = out.exists() and out.stat().st_size > 1000
            json_done = json_out.exists() and json_out.stat().st_size > 10
            if tif_done and json_done:
                log.info(f"  Ya existe: {fname} + JSON — saltando")
                continue

            valid_imgs = []
            valid_months = []
            for mo in range(mo_start, mo_end):
                end_mo = mo + 1 if mo < 12 else 1
                end_yr = yr if mo < 12 else yr + 1
                start = f"{yr}-{mo:02d}-01"
                end = f"{end_yr}-{end_mo:02d}-01"
                col = (ee.ImageCollection(col_id)
                       .filterDate(start, end)
                       .filterBounds(filter_region)
                       .select(src_bands, ["Green", "Red", "NIR", "SWIR1"]))
                n = col.size().getInfo()
                if n == 0:
                    continue
                composite = col.map(add_indices).median()
                valid_imgs.append(composite.set("system:time_start", ee.Date(start).millis()))
                valid_months.append(mo)

            if not valid_imgs:
                log.warning(f"  Sin datos Landsat {yr}-H{half} — saltando")
                continue
            log.info(f"  {yr}-H{half}: {len(valid_months)}/6 meses con datos Landsat")

            if not tif_done:
                col_yr = ee.ImageCollection(valid_imgs)
                stack  = _multi_var_stack(col_yr, ["NDVI_L", "NDWI_L", "MNDWI_L"])
                _export_image(stack, out, scale, f"Landsat {yr}-H{half}")

            if not json_done:
                meta = {
                    "yr": yr, "half": half, "valid_months": valid_months,
                    "n_bands": len(valid_months) * 3,
                    "band_vars": ["NDVI_L", "NDWI_L", "MNDWI_L"],
                    "scale_m": scale, "native": USE_NATIVE,
                    "convention": "band 3*i → NDVI_L, 3*i+1 → NDWI_L, 3*i+2 → MNDWI_L for month valid_months[i]",
                    "sensor": col_id.split("/")[1],
                }
                json_out.write_text(json.dumps(meta, indent=2))
                log.info(f"  Sidecar JSON escrito: {json_out.name}")


def download_landcover():
    """GEE5: ESA WorldCover 2020 + MCD12Q1 IGBP — estático.

    Modo legacy: scale=100m (bbox completo).
    Modo native: ESA a 10m (nativo), MCD12Q1 se mantiene en 100m (MODIS ~500m nativo).
    """
    scale_esa = _sc("landcover")          # 10m native, 100m legacy
    scale_mcd = SCALE["landcover"]        # MCD12Q1 siempre en 100m
    suffix_esa = _suffix("landcover")
    log.info(f"── Land Cover (GEE5) — ESA {suffix_esa}, MCD12Q1 100m {'(clip cuenca)' if USE_NATIVE else ''} ──")

    # ESA WorldCover 2020
    out_wc = _out("land_cover", f"esa_worldcover_2020_{suffix_esa}.tif")
    if not _already_done(out_wc):
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        _export_image(wc, out_wc, scale_esa, "ESA WorldCover 2020")

    # MODIS MCD12Q1 IGBP anual — mantener 100m (ya superresuelto respecto a nativo 500m)
    col = ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1")
    for yr in range(2001, 2024):
        out_lc = _out("land_cover", f"mcd12q1_igbp_{yr}.tif")
        if _already_done(out_lc):
            continue
        img = col.filterDate(f"{yr}-01-01", f"{yr+1}-01-01").first()
        _export_image(img.rename(f"LC_{yr}"), out_lc, scale_mcd, f"MCD12Q1 IGBP {yr}")


def download_jrc_water(years_range=(1984, 2021)):
    """GEE7: JRC Global Surface Water — ocurrencia anual por quinquenio.

    Modo legacy: scale=500m (bbox completo, trivial).
    Modo native: scale=30m con clip al polígono de cuenca (~0.1 GB total).
      JRC nativo 30m; con clip la bajada es muy pequeña.
    Genera 1 archivo/quinquenio: jrc_water_{yr}_{y1}_{scale}m.tif
    """
    scale  = _sc("jrc")
    suffix = _suffix("jrc")
    log.info(f"── JRC Water (GEE7) — {suffix} {'(clip cuenca, 1 año/archivo)' if USE_NATIVE else '(quinquenio)'} ──")
    col = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").select("water")

    if USE_NATIVE:
        # 30m nativo: 1 año × 1 banda Float32 = ~65 MB > 50 MB (2× GEE overhead D037).
        # Fix: .multiply(100).toInt16() → ~32 MB OK.
        # Escala: divide por 100 al leer para obtener ocurrencia en [0, 2].
        for yr in range(years_range[0], years_range[1] + 1):
            fname = f"jrc_water_{yr}_{suffix}.tif"
            out   = _out("water_land", fname)
            json_out = out.with_suffix(".json")
            if _already_done(out):
                continue
            annual = col.filterDate(f"{yr}-01-01", f"{yr+1}-01-01")
            img = annual.mean().rename(f"water_{yr}").multiply(100).toInt16()
            _export_image(img, out, scale, f"JRC Water {yr}")
            if out.exists() and out.stat().st_size > 1000:
                meta = {"yr": yr, "scale_m": scale, "native": True,
                        "int16_scale_factor": 100,
                        "note": "divide by 100 to recover mean water classification [0-2]"}
                json_out.write_text(json.dumps(meta, indent=2))
    else:
        for yr in range(years_range[0], years_range[1] + 1, 5):
            y1    = min(yr + 4, years_range[1])
            fname = f"jrc_water_{yr}_{y1}_{suffix}.tif"
            out   = _out("water_land", fname)
            if _already_done(out):
                continue

            def annual_occ(y):
                annual = col.filterDate(f"{y}-01-01", f"{y+1}-01-01")
                return annual.mean().rename(f"water_{y}")
            annual_imgs = [annual_occ(y) for y in range(yr, y1 + 1)]
            stack = ee.Image.cat(annual_imgs)
            _export_image(stack, out, scale, f"JRC Water {yr}-{y1}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

SOURCES = {
    "chirps":     (download_chirps,     1),
    "snow":       (download_snow_modis, 1),
    "ndvi":       (download_ndvi_modis, 2),
    "lst":        (download_lst_modis,  2),
    "lswi":       (download_lswi_modis, 2),
    "et":         (download_et_modis,   2),
    "smap":       (download_smap,       2),
    "s1":         (download_sentinel1,  2),
    "s2":         (download_sentinel2,  2),
    "landsat":    (download_landsat,    3),
    "landcover":  (download_landcover,  3),
    "jrc":        (download_jrc_water,  3),
}


def main():
    parser = argparse.ArgumentParser(
        description="Descarga GEE → local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Modo legacy (bbox completo, resoluciones originales):
  python scripts/50_gee_download.py --hito 1
  python scripts/50_gee_download.py --source jrc

  # Modo nativo (clip al polígono de cuenca, resoluciones nativas):
  python scripts/50_gee_download.py --source jrc --native
  python scripts/50_gee_download.py --source jrc --native --verify   # + mapa visual
  python scripts/50_gee_download.py --source landsat --native --year 2020
  python scripts/50_gee_download.py --source s1 --native              # S1 30m clip cuenca

  # Verificar un TIF ya descargado:
  python scripts/52_verify_gee_tif.py --tif data/raw/gee/water_land/jrc_water_1984_1988_30m.tif
        """
    )
    parser.add_argument("--hito",   type=int, choices=[1, 2, 3],
                        help="Ejecutar un hito completo")
    parser.add_argument("--source", type=str, choices=list(SOURCES.keys()),
                        help="Descargar solo una fuente")
    parser.add_argument("--year",   type=int,
                        help="Año concreto a descargar (aplica a --source)")
    parser.add_argument("--native", action="store_true",
                        help="Resolución nativa + clip al polígono de cuenca")
    parser.add_argument("--verify", action="store_true",
                        help="Generar mapa de verificación tras cada descarga")
    args = parser.parse_args()

    if not args.hito and not args.source:
        parser.print_help()
        return

    init_gee(native=args.native)

    if args.verify:
        import subprocess, sys
        _orig_export = _export_image.__wrapped__ if hasattr(_export_image, "__wrapped__") else None

        def _export_and_verify(image, out_path, scale, desc=""):
            _export_image.__wrapped__(image, out_path, scale, desc) if _orig_export else None
            subprocess.run([
                sys.executable, str(ROOT / "scripts/52_verify_gee_tif.py"),
                "--tif", str(out_path),
            ], check=False)

    def _run_source(src, year=None):
        fn, _ = SOURCES[src]
        no_year = ("landcover",)
        year_ok = src not in no_year
        if year and year_ok:
            fn(years_range=(year, year))
        else:
            fn()

    if args.source:
        _run_source(args.source, args.year)
        if args.verify:
            # Verificar el último archivo de la fuente
            import subprocess, sys
            src_folder = {
                "s1": "sentinel1", "s2": "sentinel2", "landsat": "water_land",
                "jrc": "water_land", "landcover": "land_cover",
            }.get(args.source)
            if src_folder:
                tifs = sorted((GEE_DIR / src_folder).glob("*.tif"))
                if tifs:
                    subprocess.run([
                        sys.executable, str(ROOT / "scripts/52_verify_gee_tif.py"),
                        "--tif", str(tifs[-1]),
                    ], check=False)
        return

    # Ejecutar por hito
    hito_sources = [k for k, (_, h) in SOURCES.items() if h == args.hito]
    log.info(f"=== HITO {args.hito} — {len(hito_sources)} fuentes ===")
    for src in hito_sources:
        _run_source(src)

    log.info(f"=== HITO {args.hito} COMPLETADO ===")
    log.info(f"Archivos en: {GEE_DIR}")


if __name__ == "__main__":
    main()
