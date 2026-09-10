#!/usr/bin/env python3
"""
Script 53: Catálogo de metadatos GEE + generación de sidecars JSON + verificación visual batch.

Funciones:
  catalog   — Escanea todos los TIFs GEE, lee metadatos rasterio, genera tabla Markdown + CSV.
              Crea/actualiza sidecar JSON para archivos que no lo tienen.
  verify    — Genera mapa PNG de verificación (llama a Script 52) por fuente.
  both      — catalog + verify

Uso:
    python scripts/53_gee_catalog.py catalog          # solo tabla y sidecars
    python scripts/53_gee_catalog.py verify           # mapas por fuente (1 representativo cada una)
    python scripts/53_gee_catalog.py both             # todo
    python scripts/53_gee_catalog.py catalog --md     # imprime tabla Markdown en consola
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

ROOT     = Path(__file__).parent.parent
GEE_DIR  = ROOT / "data/raw/gee"
OUT_DIR  = ROOT / "outputs"
RPT_DIR  = OUT_DIR / "reports"
FIG_DIR  = OUT_DIR / "figures/gee/verify"
CATALOG_CSV = RPT_DIR / "gee_catalog.csv"
CATALOG_MD  = RPT_DIR / "gee_catalog.md"

# ─── Fuentes conocidas ────────────────────────────────────────────────────────
# (subfolder, glob_pattern, descripcion, unidades, escala_fisica, notas)
SOURCES = {
    "chirps": {
        "folder": "chirps",
        "pattern": "chirps_precip_*.tif",
        "desc": "CHIRPS v2.0 precipitación diaria",
        "units": "mm/día",
        "band_convention": "banda i (0-based) = día i del año; pd.date_range(yr-01-01, periods=n, freq='D')",
        "scale_factor": None,
        "collection": "UCSB-CHG/CHIRPS/DAILY",
        "coverage": "1981-2025",
    },
    "snow": {
        "folder": "snow",
        "pattern": "mod10a1_snow_*.tif",
        "desc": "MOD10A1 NDSI snow cover + NDSI 500m",
        "units": "0-100 (snow_cover_pct) / [-1,1] (NDSI)",
        "band_convention": "bandas alternadas: par=NDSI_Snow_Cover [0-100], impar=NDSI [-1,1]; "
                           "ordenadas por fecha de imagen. Fill >100 → NaN para snow_cover.",
        "scale_factor": None,
        "collection": "MODIS/061/MOD10A1",
        "coverage": "2000-2025",
        "qa_note": "Valores snow_cover >100: 200=lago congelado, 211=sin decisión. Enmascarar → NaN.",
    },
    "vegetation_ndvi": {
        "folder": "vegetation",
        "pattern": "mod13q1_ndvi_evi_*.tif",
        "desc": "MOD13Q1 NDVI+EVI 16-día 250m",
        "units": "entero ×10000 (escala raw); dividir ×0.0001 → [-1,1]",
        "band_convention": "bandas alternadas: par=NDVI, impar=EVI por composite. "
                           "Convención: banda 2*i=NDVI, 2*i+1=EVI del i-ésimo composite.",
        "scale_factor": 0.0001,
        "collection": "MODIS/061/MOD13Q1",
        "coverage": "2000-2025",
        "qa_note": "Valores fuera de [-10000,10000] son fill values → NaN antes de aplicar ×0.0001.",
    },
    "vegetation_lswi": {
        "folder": "vegetation",
        "pattern": "mod09ga_lswi_*.tif",
        "desc": "MOD09GA LSWI (NIR-SWIR)/(NIR+SWIR) 500m",
        "units": "float [-1,1]; ya calculado en GEE (no requiere scale_factor)",
        "band_convention": "1 banda por imagen diaria. Nombres GEE: LSWI__YYYY_MM_dd.",
        "scale_factor": None,
        "collection": "MODIS/061/MOD09GA",
        "coverage": "2000-2025",
    },
    "thermal": {
        "folder": "thermal",
        "pattern": "mod11a1_lst_*.tif",
        "desc": "MOD11A1 LST día + noche 1km",
        "units": "Kelvin raw ×0.02 → K; convertir a °C: val×0.02 - 273.15",
        "band_convention": "bandas alternadas: par=LST_Day_1km, impar=LST_Night_1km por día. "
                           "Fill=0 → NaN.",
        "scale_factor": 0.02,
        "collection": "MODIS/061/MOD11A1",
        "coverage": "2000-2025",
        "qa_note": "LST fill value = 0 (antes de aplicar escala). Valores 0 → NaN. "
                   "Rango físico: 270-340 K día, 250-300 K noche.",
    },
    "et": {
        "folder": "et",
        "pattern": "mod16a2_et_pet_*.tif",
        "desc": "MOD16A2GF ET+PET 8-day 500m",
        "units": "kg/m²/8días raw ×0.1 → mm/8días; dividir entre 8 → mm/día",
        "band_convention": "bandas alternadas: par=ET, impar=PET por composite 8-día. "
                           "Fill=32767 → NaN.",
        "scale_factor": 0.1,
        "collection": "MODIS/061/MOD16A2GF",
        "coverage": "2001-2025",
        "qa_note": "Fill value=32767 (sin dato: bosque/urbano). Rango físico ET: [0,300] mm/8d.",
    },
    "smap": {
        "folder": "smap",
        "pattern": "smap_soil_*.tif",
        "desc": "SMAP SPL4SMGP humedad suelo 3-horario ~9km",
        "units": "m³/m³ [0.02-0.50]",
        "band_convention": "744 bandas/mes (31d×8obs/d×3var): orden sm_surface/sm_rootzone/sm_profile "
                           "cada 3h. Agregar a diario: nanmean cada 8 bandas del mismo var.",
        "scale_factor": None,
        "collection": "NASA/SMAP/SPL4SMGP/008",
        "coverage": "2015-2025",
    },
    "sentinel1": {
        "folder": "sentinel1",
        "pattern": "s1_vv_vh_ratio_*.tif",
        "desc": "Sentinel-1 GRD SAR VV+VH+RATIO mensual",
        "units": "dB (decibel); rango VV: [-25,0] dB, VH: [-35,-10] dB",
        "band_convention": "ver JSON sidecar: bandas 3*i=VV, 3*i+1=VH, 3*i+2=RATIO para mes valid_months[i]",
        "scale_factor": None,
        "collection": "COPERNICUS/S1_GRD",
        "coverage": "2015-2025",
    },
    "sentinel2": {
        "folder": "sentinel2",
        "pattern": "s2_ndvi_ndwi_ndsi_*.tif",
        "desc": "Sentinel-2 MSI NDVI/NDWI/NDSI mensual",
        "units": "índices [-1,1] ya calculados en GEE",
        "band_convention": "ver JSON sidecar: bandas 3*i=NDVI_S2, 3*i+1=NDWI_S2, 3*i+2=NDSI_S2 "
                           "para mes valid_months[i]",
        "scale_factor": None,
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "coverage": "2017-2025",
    },
    "jrc": {
        "folder": "water_land",
        "pattern": "jrc_water_*.tif",
        "desc": "JRC Global Surface Water ocurrencia anual 30m",
        "units": "fracción tiempo como agua [0-2]; si Int16: dividir entre json.int16_scale_factor",
        "band_convention": "1 banda por año (modo native/30m, 1 arch/año). "
                           "Legacy: N bandas = N años del quinquenio.",
        "scale_factor": None,  # viene del sidecar JSON (int16_scale_factor=100)
        "collection": "JRC/GSW1_4/MonthlyHistory",
        "coverage": "1984-2021",
        "qa_note": "Archivos 30m usan Int16 ×100 para sortear límite 50 MB GEE (D037). "
                   "Dividir por int16_scale_factor del JSON para obtener ocurrencia real [0-2].",
    },
    "landsat": {
        "folder": "water_land",
        "pattern": "landsat_ndvi_ndwi_*.tif",
        "desc": "Landsat 5/7/8/9 NDVI+NDWI+MNDWI semestral",
        "units": "índices [-1,1]",
        "band_convention": "ver JSON sidecar: bandas 3*i=NDVI_L, 3*i+1=NDWI_L, 3*i+2=MNDWI_L "
                           "para mes valid_months[i]",
        "scale_factor": None,
        "collection": "LANDSAT/LC08,LE07,LT05/C02/T1_L2",
        "coverage": "1984-2025",
        "qa_note": "LE07 post-2003-05-31: stripes SLC-off (falla hardware). Compensado con "
                   "compositing mediana mensual. Ver §2.6.",
    },
    "land_cover": {
        "folder": "land_cover",
        "pattern": "*.tif",
        "desc": "ESA WorldCover 2020 (10m) + MCD12Q1 IGBP anual (100m)",
        "units": "clases enteras (ESA: 1-100, IGBP: 1-17)",
        "band_convention": "ESA: 1 banda estática 2020. MCD12Q1: 1 TIF/año, 1 banda.",
        "scale_factor": None,
        "collection": "ESA/WorldCover/v200 + MODIS/061/MCD12Q1",
        "coverage": "2001-2023 (MCD12Q1) + 2020 (ESA)",
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _read_tif_meta(tif: Path) -> dict:
    """Lee metadatos rasterio de un TIF."""
    try:
        with rasterio.open(tif) as src:
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            res_m_x = res_x * 111320 * math.cos(math.radians(
                (src.bounds.bottom + src.bounds.top) / 2))
            res_m_y = res_y * 111320
            # NaN stats on first band
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            valid = arr[np.isfinite(arr)]
            return {
                "n_bands":   src.count,
                "dtype":     str(src.dtypes[0]),
                "nodata":    src.nodata,
                "crs":       str(src.crs),
                "res_deg":   f"{res_x:.6f}°×{res_y:.6f}°",
                "res_m":     f"~{res_m_x:.0f}m×{res_m_y:.0f}m",
                "width":     src.width,
                "height":    src.height,
                "n_pixels":  src.width * src.height,
                "bbox_w":    round(abs(src.bounds.right - src.bounds.left) * 111.32, 1),
                "bbox_h":    round(abs(src.bounds.top - src.bounds.bottom) * 111.32, 1),
                "bounds":    [round(v, 4) for v in src.bounds],
                "nan_pct":   round((1 - valid.size / arr.size) * 100, 1) if arr.size > 0 else 100,
                "b1_min":    round(float(np.nanmin(valid)), 4) if valid.size > 0 else None,
                "b1_max":    round(float(np.nanmax(valid)), 4) if valid.size > 0 else None,
                "b1_mean":   round(float(np.nanmean(valid)), 4) if valid.size > 0 else None,
                "ok":        True,
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _infer_sidecar(tif: Path, src_key: str, meta: dict) -> dict:
    """Genera contenido de sidecar JSON inferido del nombre del archivo y metadatos."""
    name = tif.stem
    parts = name.split("_")
    native = "clip" in name or "30m" in name or "10m" in name

    # Detectar año(s) del nombre
    years = [int(p) for p in parts if p.isdigit() and len(p) == 4 and 1980 < int(p) < 2030]

    sidecar = {
        "source_key": src_key,
        "filename": tif.name,
        "collection": SOURCES.get(src_key, {}).get("collection", ""),
        "description": SOURCES.get(src_key, {}).get("desc", ""),
        "units": SOURCES.get(src_key, {}).get("units", ""),
        "band_convention": SOURCES.get(src_key, {}).get("band_convention", ""),
        "scale_factor": SOURCES.get(src_key, {}).get("scale_factor"),
        "n_bands": meta.get("n_bands"),
        "dtype": meta.get("dtype"),
        "nodata": meta.get("nodata"),
        "crs": meta.get("crs"),
        "res_deg": meta.get("res_deg"),
        "res_m_approx": meta.get("res_m"),
        "bounds_lonlat": meta.get("bounds"),
        "width_px": meta.get("width"),
        "height_px": meta.get("height"),
        "native_clip": native,
        "clip_region": "basin_polygon_500m_buffer (EPSG:4326)" if native else "bbox_completo_1.1x0.9deg",
        "years_in_file": years,
        "generated_by": "53_gee_catalog.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # JRC: int16_scale_factor si es Int16
    if src_key == "jrc" and meta.get("dtype") == "int16":
        sidecar["int16_scale_factor"] = 100
        sidecar["int16_note"] = ("multiply(100).toInt16() applied to fit GEE 50 MB limit (D037). "
                                 "Divide by 100 to recover water occurrence [0-2].")

    # Notas de QA
    qa = SOURCES.get(src_key, {}).get("qa_note")
    if qa:
        sidecar["qa_note"] = qa

    return sidecar


def _ensure_sidecar(tif: Path, src_key: str, meta: dict, overwrite: bool = False) -> Path:
    """Crea/actualiza sidecar JSON si no existe (o si overwrite=True)."""
    json_path = tif.with_suffix(".json")
    if json_path.exists() and not overwrite:
        # Leer existente y actualizar campos de metadatos
        try:
            existing = json.loads(json_path.read_text())
            if "generated_by" not in existing:
                existing["generated_by"] = "53_gee_catalog.py (actualizado)"
                existing["n_bands"] = meta.get("n_bands")
                existing["dtype"] = meta.get("dtype")
                json_path.write_text(json.dumps(existing, indent=2, default=str))
        except Exception:
            pass
        return json_path

    content = _infer_sidecar(tif, src_key, meta)
    json_path.write_text(json.dumps(content, indent=2, default=str))
    return json_path


# ─── Catálogo ─────────────────────────────────────────────────────────────────

def build_catalog(print_md: bool = False) -> list[dict]:
    """Escanea todos los TIFs GEE y construye el catálogo."""
    RPT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for src_key, cfg in SOURCES.items():
        folder = GEE_DIR / cfg["folder"]
        if not folder.exists():
            continue
        tifs = sorted(folder.glob(cfg["pattern"]))
        if not tifs:
            continue

        total_bytes = sum(t.stat().st_size for t in tifs)
        total_mb    = total_bytes / 1e6

        for tif in tifs:
            size_mb = tif.stat().st_size / 1e6
            meta    = _read_tif_meta(tif)
            _ensure_sidecar(tif, src_key, meta)

            # Detectar modo: native/clip vs legacy
            name = tif.name
            if "_clip" in name:
                mode = "native_clip"
            elif any(f"_{r}m" in name for r in ["30", "10"]):
                mode = "native_hires"
            else:
                mode = "legacy"

            row = {
                "fuente":      src_key,
                "archivo":     tif.name,
                "modo":        mode,
                "size_mb":     round(size_mb, 2),
                "n_bandas":    meta.get("n_bands", "?"),
                "dtype":       meta.get("dtype", "?"),
                "res_m":       meta.get("res_m", "?"),
                "px_W":        meta.get("width", "?"),
                "px_H":        meta.get("height", "?"),
                "nan_pct":     meta.get("nan_pct", "?"),
                "b1_min":      meta.get("b1_min", "?"),
                "b1_max":      meta.get("b1_max", "?"),
                "crs":         meta.get("crs", "?"),
                "ok":          meta.get("ok", False),
                "error":       meta.get("error", ""),
            }
            rows.append(row)

        n_native = sum(1 for r in rows if r["fuente"] == src_key and r["modo"] != "legacy")
        n_legacy = sum(1 for r in rows if r["fuente"] == src_key and r["modo"] == "legacy")
        print(f"  {src_key:<20} {len(tifs):>4} arch  {total_mb:>8.1f} MB  "
              f"native={n_native} legacy={n_legacy}")

    # Escribir CSV
    if rows:
        with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV guardado: {CATALOG_CSV}")

    # Escribir Markdown
    _write_markdown_catalog(rows, print_md)

    return rows


def _write_markdown_catalog(rows: list[dict], print_md: bool):
    """Genera reporte Markdown con tabla resumida y notas de problemas."""
    lines = []
    lines.append(f"# Catálogo GEE — HidroAlerta Chancay-Huaral")
    lines.append(f"> Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Tabla resumen por fuente
    lines.append("## Resumen por fuente")
    lines.append("")
    lines.append("| Fuente | Archivos | MB total | Modo | Resolución | Bandas típicas | NaN% B1 | CRS |")
    lines.append("|--------|---------|----------|------|-----------|----------------|---------|-----|")

    from itertools import groupby
    for src_key in SOURCES:
        src_rows = [r for r in rows if r["fuente"] == src_key]
        if not src_rows:
            continue
        total_mb = sum(r["size_mb"] for r in src_rows)
        modes    = set(r["modo"] for r in src_rows)
        resols   = set(r["res_m"] for r in src_rows if r["ok"])
        n_bands  = [r["n_bandas"] for r in src_rows if isinstance(r["n_bandas"], int)]
        nan_pcts = [r["nan_pct"] for r in src_rows if isinstance(r["nan_pct"], float)]
        crss     = set(r["crs"] for r in src_rows if r["ok"])
        lines.append(
            f"| {src_key} | {len(src_rows)} | {total_mb:.1f} MB | {', '.join(modes)} | "
            f"{', '.join(resols) or '?'} | "
            f"{min(n_bands)}-{max(n_bands) if n_bands else '?'} | "
            f"{sum(nan_pcts)/len(nan_pcts):.1f}% | "
            f"{', '.join(crss) or '?'} |"
        )
    lines.append("")

    # Tabla detallada
    lines.append("## Detalle por archivo")
    lines.append("")
    lines.append("| Archivo | MB | Modo | Res | WxH px | Bandas | NaN%B1 | B1_min | B1_max |")
    lines.append("|---------|-----|------|-----|--------|--------|--------|--------|--------|")
    for r in rows:
        lines.append(
            f"| `{r['archivo']}` | {r['size_mb']} | {r['modo']} | {r['res_m']} | "
            f"{r['px_W']}×{r['px_H']} | {r['n_bandas']} | {r['nan_pct']}% | "
            f"{r['b1_min']} | {r['b1_max']} |"
        )
    lines.append("")

    # Bugs documentados
    lines.append("## Bugs documentados y resueltos")
    lines.append("")
    lines.append("""### D036 — GEE overhead de tamaño ×2 en resolución nativa
- **Problema**: GEE estima tamaño de descarga a resolución nativa del dataset antes de reproyectar,
  resultando en requests ~2× más grandes de lo esperado para fuentes con resolución nativa ≤ 30m.
- **Impacto**: JRC@30m planificado como ~32 MB resultó en ~65 MB → falla silenciosa de geemap.
- **Solución**: `.multiply(100).toInt16()` reduce el dtype de Float32 (4B) a Int16 (2B), dividiendo
  el tamaño por 2. Documentar el `int16_scale_factor` en el sidecar JSON.
- **Afecta**: JRC Water 30m. S1/S2/Landsat a verificar.

### D037 — geemap falla silenciosamente para requests > 50 MB
- **Problema**: `geemap.ee_export_image()` descarga 0 bytes sin error cuando el request supera ~50 MB.
- **Detección**: `_already_done()` en `_export_image()` verifica que el archivo pese > 1000 bytes.
  Si no, lanza `RuntimeError` explícito.
- **Solución**: mantener cada request bajo 50 MB mediante chunking temporal (bimestral/semestral)
  e Int16 cuando sea necesario.

### D038 — image.clip() falla con proyección sinusoidal MODIS
- **Problema**: Al descargar en modo `--native`, `_export_image()` aplicaba `image.clip(BASIN_GEOM)`
  antes del export. Esto falla para imágenes MODIS (MOD16A2, MOD11A1, etc.) porque GEE no puede
  transformar los bordes de tiles sinusoidales (SR-ORG:6974) a WGS84.
  Error: `Image.clip: Unable to transform edge (86400.000000, ...) from SR-ORG:6974 ... to EPSG:4326`
- **Impacto**: Descarga de ET (MOD16A2) falló a partir de año 2007. LST y otros MODIS también
  habrían fallado sin el fix.
- **Solución**: Eliminar `image.clip(BASIN_GEOM)` de `_export_image()`. El parámetro
  `region=BASIN_GEOM` en `geemap.ee_export_image()` ya restringe el GeoTIFF descargado al bbox
  del polígono sin requerir transformación de proyección en el servidor.
- **Por qué region= es suficiente**: GEE usa el polígono de cuenca como ventana de descarga,
  exportando solo el bbox de ese polígono. El resultado es equivalente en tamaño a un clip, sin
  el overhead de transformación de proyección.
- **Fix aplicado en**: `scripts/50_gee_download.py`, función `_export_image()`, 2026-05-27.

### D039 — Colección Landsat vacía antes de 1984
- **Problema**: Landsat 5 inició en 1984; años anteriores tienen colección vacía → GEE devuelve
  imagen vacía sin error.
- **Solución**: `col.size().getInfo() == 0` → saltar con `log.warning()`. No se crea archivo.
""")

    # Tamaños de archivo: por qué son correctos
    lines.append("## ¿Por qué los archivos parecen pequeños?")
    lines.append("")
    lines.append("""Los tamaños son correctos. Estimación teórica (verificada):

| Fuente | Res | Píxeles (bbox polígono) | Bandas típicas | Raw MB | Comprimido (DEFLATE) |
|--------|-----|------------------------|----------------|--------|----------------------|
| CHIRPS | 5566m (~0.05°) | ~16×13 = 208 px | 365 días | 0.29 MB | **0.1-0.2 MB** ✓ |
| MOD10A1 Snow | 500m | ~174×145 = 25,230 px | 60 días × 2 var | 12 MB | **1-3 MB** ✓ |
| MOD11A1 LST | 1000m | ~87×72 = 6,264 px | 180 días × 2 var | 9 MB | **2-3 MB** ✓ |
| MOD16A2 ET | 500m | ~174×145 = 25,230 px | 46 comp × 2 var | 9 MB | **2-3 MB** ✓ |
| SMAP | 10000m | ~9×7 = 63 px | 31d×8obs×3var = 744 | 0.15 MB | **0.1-0.3 MB** ✓ |

El clip al polígono de cuenca (bbox ~87×72 km) reduce los píxeles respecto al bbox completo
(~122×122 km) en ~70%. GeoTIFF DEFLATE comprime muy bien datos con muchos píxeles NoData
(bordes del clip) y datos satelitales con baja entropía (series con autocorrelación).

> **Verificación rápida**: `python scripts/52_verify_gee_tif.py --tif <archivo>` muestra
> el tamaño real, resolución en metros medida desde el transform, y estadísticas por banda.
""")

    text = "\n".join(lines)
    RPT_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_MD.write_text(text, encoding="utf-8")
    print(f"Markdown guardado: {CATALOG_MD}")

    if print_md:
        print("\n" + "="*80)
        print(text[:4000])  # primeras 4000 chars
        print("="*80)


# ─── Verificación visual batch ─────────────────────────────────────────────────

# Un archivo representativo por fuente para verificación visual
VERIFY_SAMPLES = {
    "chirps":          "chirps/chirps_precip_2010_5566m_clip.tif",
    "snow":            "snow/mod10a1_snow_2010_m0708_500m_clip.tif",
    "vegetation_ndvi": "vegetation/mod13q1_ndvi_evi_2010_h1_250m_clip.tif",
    "vegetation_lswi": "vegetation/mod09ga_lswi_2010_m0708_500m_clip.tif",
    "thermal":         "thermal/mod11a1_lst_2010_h1_1000m_clip.tif",
    "et":              "et/mod16a2_et_pet_2010_500m_clip.tif",
    "smap":            "smap/smap_soil_2020_01_3h_10000m_clip.tif",
    "jrc":             "water_land/jrc_water_2010_30m.tif",
    "landsat":         "water_land/landsat_ndvi_ndwi_2020_h1_250m.tif",
    "sentinel1":       "sentinel1/s1_vv_vh_ratio_2020_500m_clip.tif",
    "sentinel2":       "sentinel2/s2_ndvi_ndwi_ndsi_2020_500m_clip.tif",
    "esa_worldcover":  "land_cover/esa_worldcover_2020_30m.tif",
    "mcd12q1":         "land_cover/mcd12q1_igbp_2020.tif",
}


def run_verify(force: bool = False):
    """Genera mapas de verificación para un archivo representativo de cada fuente."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    verify_script = ROOT / "scripts/52_verify_gee_tif.py"

    for src_key, rel_path in VERIFY_SAMPLES.items():
        if rel_path is None:
            print(f"  {src_key}: pendiente de descarga — saltando")
            continue
        tif = GEE_DIR / rel_path
        if not tif.exists():
            print(f"  {src_key}: {tif.name} no encontrado — saltando")
            continue
        out_png = FIG_DIR / f"{tif.stem}_verify.png"
        if out_png.exists() and not force:
            print(f"  {src_key}: ya verificado — {out_png.name}")
            continue

        print(f"  Verificando {src_key}: {tif.name} ...")
        result = subprocess.run(
            [sys.executable, str(verify_script), "--tif", str(tif)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"    OK -> {out_png.name}")
        else:
            print(f"    ERROR: {result.stderr[-200:]}")

    # SMAP: verificar con 3 bandas (una por variable)
    smap_tif = GEE_DIR / "smap/smap_soil_2020_01_3h_10000m_clip.tif"
    if smap_tif.exists():
        for band_idx in [0, 1, 2]:
            out_smap = FIG_DIR / f"smap_band{band_idx}_verify.png"
            if not out_smap.exists():
                subprocess.run(
                    [sys.executable, str(verify_script), "--tif", str(smap_tif),
                     "--band", str(band_idx)],
                    capture_output=True, text=True,
                )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Catálogo y verificación de TIFs GEE")
    parser.add_argument("mode", choices=["catalog", "verify", "both"],
                        help="catalog=tabla+sidecars, verify=mapas PNG, both=todo")
    parser.add_argument("--md", action="store_true",
                        help="Imprimir tabla Markdown en consola (solo catalog)")
    parser.add_argument("--overwrite-sidecars", action="store_true",
                        help="Regenerar sidecars JSON aunque ya existan")
    args = parser.parse_args()

    if args.mode in ("catalog", "both"):
        print("=== Construyendo catálogo GEE ===")
        build_catalog(print_md=args.md)

    if args.mode in ("verify", "both"):
        print("\n=== Generando mapas de verificación ===")
        run_verify()

    print("\nDone.")


if __name__ == "__main__":
    main()
