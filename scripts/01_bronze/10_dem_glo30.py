#!/usr/bin/env python3
"""
Script 10b: DEM Copernicus GLO-30 (30m) — atributos topográficos cuenca Chancay-Huaral.

Fuente
------
Copernicus DEM GLO-30 (1 arc-segundo ≈ 30 m) disponible en AWS S3 público:
  s3://copernicus-dem-30m/  (sin autenticación, Cloud-Optimized GeoTIFF)

Reemplaza Script 10 (SRTM 90m) con resolución 3× mejor y mayor precisión
en zonas andinas.  Las mismas salidas se sobrescriben en B4_* para que el
pipeline (Script 16, etc.) no requiera cambios.

Productos derivados
-------------------
  1. Pendiente (slope) grados — Horn (1981)
  2. Aspecto (aspect) 0-360° desde Norte
  3. Curva hipsométrica + integral hipsométrica (Strahler, 1952)
  4. Estadísticos topográficos por subcuenca

Salidas
-------
  data/raw/dem/glo30/               — tiles crudos cacheados
  data/bronze/B4_dem_basin_30m.nc   — DEM + slope + aspect (30m, cuenca)
  data/bronze/B4_dem_subcuencas_stats.csv
  outputs/figures/basin/T01_dem_analisis.png

Referencias
-----------
Copernicus DEM: https://doi.org/10.5270/ESA-c5d3d65
Horn (1981) doi:10.1109/PROC.1981.11918
Strahler (1952) doi:10.1130/0016-7606(1952)63
"""
import datetime
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
import requests
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(Path(__file__).parent.parent / "outputs" / "10b_glo30.log",
                                  "w", "utf-8")],
)
log = logging.getLogger("dem_glo30")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
SHP_CUE = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
            / "Cuenca_Chancay___Huaral.shp")
SHP_SUB = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas"
            / "Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_"
              "geogpsperu_UH_MENORES.shp")
DEM_DIR = ROOT / "data/raw/dem/glo30"
OUT_NC  = ROOT / "data/bronze/B4_dem_basin_30m.nc"
OUT_CSV = ROOT / "data/bronze/B4_dem_subcuencas_stats.csv"
FIG_DIR = ROOT / "outputs/figures/basin"
OUT_FIG = FIG_DIR / "T01_dem_analisis.png"

DEM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# AWS base URL (bucket público, sin auth)
AWS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DETECCIÓN Y DESCARGA DE TILES GLO-30
# ══════════════════════════════════════════════════════════════════════════════

def _tile_name(lat_sw: int, lon_sw: int) -> str:
    """Nombre estándar del tile GLO-30 dado el SW corner en grados enteros."""
    ns  = "S" if lat_sw < 0 else "N"
    ew  = "W" if lon_sw < 0 else "E"
    lat_abs = abs(lat_sw)
    lon_abs = abs(lon_sw)
    return f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM"


def _tile_url(tile: str) -> str:
    return f"{AWS_BASE}/{tile}/{tile}.tif"


def tiles_for_bbox(lon_min: float, lon_max: float,
                   lat_min: float, lat_max: float) -> list[tuple[int,int]]:
    """
    Devuelve lista de (lat_sw, lon_sw) de todos los tiles GLO-30 que
    intersectan el bounding-box dado.  Cada tile cubre 1°×1°; el nombre
    del tile es el SW corner (floor hacia -∞).
    """
    lat0 = math.floor(lat_min)
    lat1 = math.floor(lat_max)   # si lat_max es entero exacto, ese tile no es necesario
    lon0 = math.floor(lon_min)
    lon1 = math.floor(lon_max)
    tiles = []
    for la in range(lat0, lat1 + 1):
        for lo in range(lon0, lon1 + 1):
            tiles.append((la, lo))
    return tiles


def download_tiles(basin_wgs84) -> list[Path]:
    """
    Determina qué tiles cubre la cuenca y los descarga si no están cacheados.
    Retorna lista de paths a los .tif locales.
    """
    bounds = basin_wgs84.total_bounds   # (minx, miny, maxx, maxy)
    lon_min, lat_min, lon_max, lat_max = bounds
    log.info(f"Bbox cuenca WGS84: lon=[{lon_min:.3f}, {lon_max:.3f}]  "
             f"lat=[{lat_min:.3f}, {lat_max:.3f}]")

    sw_corners = tiles_for_bbox(lon_min, lon_max, lat_min, lat_max)
    log.info(f"Tiles GLO-30 necesarios: {len(sw_corners)}")

    paths = []
    for lat_sw, lon_sw in sw_corners:
        tile = _tile_name(lat_sw, lon_sw)
        url  = _tile_url(tile)
        dest = DEM_DIR / f"{tile}.tif"

        if dest.exists():
            log.info(f"  Cache OK: {tile}.tif ({dest.stat().st_size/1e6:.0f} MB)")
        else:
            log.info(f"  Descargando {tile}.tif ...")
            try:
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1024 * 1024):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total and downloaded % (20 * 1024 * 1024) < 1024 * 1024:
                            log.info(f"    {downloaded/1e6:.0f}/{total/1e6:.0f} MB  "
                                     f"({downloaded/total*100:.0f}%)")
                log.info(f"  OK: {dest.name}  ({dest.stat().st_size/1e6:.0f} MB)")
            except Exception as e:
                log.error(f"  ERROR descargando {tile}: {e}")
                if dest.exists():
                    dest.unlink()
                continue
        paths.append(dest)

    if not paths:
        raise RuntimeError("No se pudo descargar ningún tile GLO-30.")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 2. MERGE + CLIP A LA CUENCA
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_clip(tile_paths: list[Path], basin_wgs84) -> tuple:
    """
    Fusiona los tiles (si hay más de uno) y recorta al shapefile de cuenca.
    Retorna (dem_array, lat_1d, lon_1d, res_deg).
    """
    log.info("Fusionando tiles y recortando a cuenca ...")
    shapes = list(basin_wgs84.geometry)

    # Abrir todos los tiles
    srcs = [rasterio.open(p) for p in tile_paths]

    if len(srcs) > 1:
        merged, merged_transform = rio_merge(srcs)
        merged = merged[0].astype("f4")
        # Construir meta para el merged
        meta = srcs[0].meta.copy()
        meta.update({"height": merged.shape[0], "width": merged.shape[1],
                     "transform": merged_transform})
        # Guardar merged temporal en MemoryFile para hacer clip
        with MemoryFile() as memf:
            with memf.open(**meta) as tmp:
                tmp.write(merged, 1)
                dem_clip, clip_transform = rio_mask(tmp, shapes, crop=True,
                                                    nodata=-9999.0, filled=True)
        res = (abs(merged_transform.e), abs(merged_transform.a))
    else:
        with srcs[0] as src:
            dem_clip, clip_transform = rio_mask(src, shapes, crop=True,
                                               nodata=-9999.0, filled=True)
            res = src.res   # (row_res, col_res) en grados

    for s in srcs:
        s.close()

    dem = dem_clip[0].astype("f4")
    dem[dem == -9999.0] = np.nan
    dem[dem < 0]        = np.nan   # void / océano

    nrows, ncols = dem.shape
    col_coords   = clip_transform.c + (np.arange(ncols) + 0.5) * clip_transform.a
    row_coords   = clip_transform.f + (np.arange(nrows) + 0.5) * clip_transform.e
    lon_1d, lat_1d = col_coords, row_coords

    res_m = abs(res[1]) * 111320 * np.cos(np.radians(abs(np.nanmean(lat_1d))))
    log.info(f"DEM cuenca: {nrows}×{ncols} px  "
             f"(resolución ≈ {res_m:.0f}m)  "
             f"elev [{np.nanmin(dem):.0f}, {np.nanmax(dem):.0f}] m")
    return dem, lat_1d, lon_1d, res


# ══════════════════════════════════════════════════════════════════════════════
# 3. PENDIENTE Y ASPECTO (Horn, 1981)
# ══════════════════════════════════════════════════════════════════════════════

def compute_slope_aspect(dem: np.ndarray, lat_1d: np.ndarray,
                         res_deg: tuple) -> tuple[np.ndarray, np.ndarray]:
    lat_mean_rad = np.radians(float(np.nanmean(lat_1d)))
    dx_m = abs(res_deg[1]) * 111320 * np.cos(lat_mean_rad)
    dy_m = abs(res_deg[0]) * 111320

    dem_filled = dem.copy()
    nan_mask   = np.isnan(dem_filled)
    dem_filled[nan_mask] = float(np.nanmean(dem_filled))

    dz_dy, dz_dx = np.gradient(dem_filled, dy_m, dx_m)

    slope_deg  = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope_deg[nan_mask] = np.nan

    aspect_deg = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360) % 360
    aspect_deg[nan_mask] = np.nan

    log.info(f"Slope: media={np.nanmean(slope_deg):.1f}°  "
             f"max={np.nanmax(slope_deg):.1f}°")
    return slope_deg.astype("f4"), aspect_deg.astype("f4")


# ══════════════════════════════════════════════════════════════════════════════
# 4. GUARDAR NetCDF
# ══════════════════════════════════════════════════════════════════════════════

def save_nc(dem, slope, aspect, lat_1d, lon_1d):
    if OUT_NC.exists():
        OUT_NC.unlink()
    ds = nc.Dataset(OUT_NC, "w", format="NETCDF4")
    ds.createDimension("lat", len(lat_1d))
    ds.createDimension("lon", len(lon_1d))

    v = ds.createVariable("lat", "f4", ("lat",))
    v.units = "degrees_north"; v[:] = lat_1d.astype("f4")

    v = ds.createVariable("lon", "f4", ("lon",))
    v.units = "degrees_east";  v[:] = lon_1d.astype("f4")

    for varname, data, units, longname in [
        ("dem",    dem,    "m",       "Elevation Copernicus GLO-30 30m"),
        ("slope",  slope,  "degrees", "Slope Horn (1981)"),
        ("aspect", aspect, "degrees", "Aspect from North clockwise Horn (1981)"),
    ]:
        vv = ds.createVariable(varname, "f4", ("lat", "lon"),
                               zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        vv.units = units; vv.long_name = longname
        vv[:] = data

    ds.title      = "DEM Copernicus GLO-30 — Cuenca Chancay-Huaral (30m)"
    ds.source     = "Copernicus DEM GLO-30 via AWS s3://copernicus-dem-30m"
    ds.references = "https://doi.org/10.5270/ESA-c5d3d65"
    ds.history    = f"Creado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ds.Conventions = "CF-1.8"
    ds.close()
    log.info(f"  -> {OUT_NC.name} ({OUT_NC.stat().st_size/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ESTADÍSTICOS POR SUBCUENCA
# ══════════════════════════════════════════════════════════════════════════════

def stats_by_subcuenca(dem, slope, lat_1d, lon_1d) -> pd.DataFrame:
    log.info("Calculando estadísticos por subcuenca ...")
    subcuencas     = gpd.read_file(SHP_SUB).to_crs("EPSG:4326")
    subcuencas_utm = subcuencas.to_crs("EPSG:32718")
    subcuencas["area_km2"] = subcuencas_utm.area / 1e6

    # Transform para el DEM recortado
    half_lon = (lon_1d[1] - lon_1d[0]) / 2 if len(lon_1d) > 1 else 0.0001
    half_lat = abs(lat_1d[1] - lat_1d[0]) / 2 if len(lat_1d) > 1 else 0.0001
    clip_transform = from_bounds(
        lon_1d[0]  - half_lon, lat_1d[-1] - half_lat,
        lon_1d[-1] + half_lon, lat_1d[0]  + half_lat,
        len(lon_1d), len(lat_1d)
    )
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": len(lon_1d), "height": len(lat_1d),
        "count": 1, "crs": "EPSG:4326",
        "transform": clip_transform, "nodata": np.nan,
    }

    records = []
    with MemoryFile() as mf_dem:
        with mf_dem.open(**profile) as ds_dem:
            ds_dem.write(dem, 1)
            with MemoryFile() as mf_sl:
                with mf_sl.open(**profile) as ds_sl:
                    ds_sl.write(slope, 1)

                    for _, row in subcuencas.iterrows():
                        try:
                            d_clip, _ = rio_mask(ds_dem, [row.geometry],
                                                 crop=False, nodata=np.nan, filled=True)
                            d_px = d_clip[0][np.isfinite(d_clip[0])]
                            if len(d_px) < 10:
                                continue
                            s_clip, _ = rio_mask(ds_sl, [row.geometry],
                                                 crop=False, nodata=np.nan, filled=True)
                            s_px = s_clip[0][np.isfinite(s_clip[0])]
                            records.append({
                                "subcuenca_id"  : int(row["ID"]),
                                "area_km2"      : round(float(row["area_km2"]), 2),
                                "elev_min_m"    : round(float(d_px.min()), 0),
                                "elev_max_m"    : round(float(d_px.max()), 0),
                                "elev_mean_m"   : round(float(d_px.mean()), 1),
                                "elev_median_m" : round(float(np.median(d_px)), 1),
                                "elev_std_m"    : round(float(d_px.std()), 1),
                                "relief_m"      : round(float(d_px.max() - d_px.min()), 0),
                                "slope_mean_deg": round(float(s_px.mean()), 2),
                                "slope_max_deg" : round(float(s_px.max()), 2),
                                "n_pixels"      : len(d_px),
                            })
                        except Exception as e:
                            log.warning(f"  Subcuenca {row['ID']}: {e}")

    df = pd.DataFrame(records).sort_values("elev_mean_m", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    log.info(f"  -> {OUT_CSV.name}  ({len(df)} subcuencas)")
    log.info(df[["subcuenca_id", "area_km2", "elev_mean_m",
                 "relief_m", "slope_mean_deg"]].to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. CURVA HIPSOMÉTRICA (Strahler, 1952)
# ══════════════════════════════════════════════════════════════════════════════

def hypsometric_curve(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    elev = dem[np.isfinite(dem)].ravel()
    elev_sorted = np.sort(elev)[::-1]
    emin, emax  = elev_sorted.min(), elev_sorted.max()
    area_pct    = np.linspace(0, 100, len(elev_sorted))
    elev_norm   = (elev_sorted - emin) / (emax - emin + 1e-6) * 100
    hi          = float(np.trapezoid(elev_norm / 100, area_pct / 100))
    return area_pct, elev_sorted, hi


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURA T01
# ══════════════════════════════════════════════════════════════════════════════

def plot_dem(dem, slope, lat_1d, lon_1d, df_stats, hi, n_tiles):
    log.info("Generando figura T01 ...")
    basin  = gpd.read_file(SHP_CUE).to_crs("EPSG:4326")
    subcue = gpd.read_file(SHP_SUB).to_crs("EPSG:4326")
    # Filtrar subcuencas al interior de la cuenca (el shapefile puede incluir
    # subcuencas de otras cuencas, lo que deformaría el extent del mapa)
    subcue = subcue[subcue.intersects(basin.union_all())].copy()
    log.info(f"  Subcuencas dentro de la cuenca: {len(subcue)}")

    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    LON, LAT = np.meshgrid(lon_1d, lat_1d)

    # ── (a) DEM hillshade ────────────────────────────────────────────────────
    ax_dem = fig.add_subplot(gs[0, 0])
    ls      = LightSource(azdeg=315, altdeg=35)
    dem_f   = np.where(np.isfinite(dem), dem, 0.0)
    rgb     = ls.shade(dem_f, cmap=cm.terrain, vert_exag=2.0,
                       blend_mode="overlay", vmin=0, vmax=float(np.nanmax(dem_f)))
    ax_dem.imshow(rgb, extent=[lon_1d[0], lon_1d[-1], lat_1d[-1], lat_1d[0]],
                  origin="upper", aspect="equal")
    sm_d = ScalarMappable(cmap=cm.terrain,
                          norm=Normalize(0, float(np.nanmax(dem_f))))
    sm_d.set_array([])
    cb = fig.colorbar(sm_d, ax=ax_dem, fraction=0.032, pad=0.02)
    cb.set_label("Elevación (m s.n.m.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    basin.boundary.plot(ax=ax_dem, color="white", lw=1.8, zorder=5)
    basin.boundary.plot(ax=ax_dem, color="#111", lw=0.8, ls="--", zorder=6)
    subcue.boundary.plot(ax=ax_dem, color="cyan", lw=0.5, alpha=0.5, zorder=4)
    # Fijar límites al extent del DEM recortado (geopandas puede expandirlos)
    ax_dem.set_xlim(lon_1d[0], lon_1d[-1])
    ax_dem.set_ylim(lat_1d[-1], lat_1d[0])
    ax_dem.set_title(f"(a) DEM Copernicus GLO-30  ≈ 30 m\n"
                     f"Cuenca Chancay-Huaral  |  "
                     f"[{int(np.nanmin(dem))}–{int(np.nanmax(dem))} m s.n.m.]"
                     f"  ({n_tiles} tile{'s' if n_tiles>1 else ''})",
                     fontweight="bold", fontsize=9)
    ax_dem.set_xlabel("Lon (°)", fontsize=8)
    ax_dem.set_ylabel("Lat (°)", fontsize=8)
    ax_dem.tick_params(labelsize=7)

    # ── (b) Pendiente ────────────────────────────────────────────────────────
    ax_sl = fig.add_subplot(gs[0, 1])
    vmax_s = float(np.nanpercentile(slope, 98))
    im_s   = ax_sl.pcolormesh(LON, LAT, np.ma.masked_invalid(slope),
                               cmap="YlOrRd", vmin=0, vmax=vmax_s, shading="auto")
    cb_s   = fig.colorbar(im_s, ax=ax_sl, fraction=0.032, pad=0.02)
    cb_s.set_label("Pendiente (°)", fontsize=8)
    cb_s.ax.tick_params(labelsize=7)
    basin.boundary.plot(ax=ax_sl, color="black", lw=1.2, zorder=5)
    subcue.boundary.plot(ax=ax_sl, color="gray", lw=0.4, alpha=0.6, zorder=4)
    # Fijar límites al extent del DEM recortado
    ax_sl.set_xlim(lon_1d[0], lon_1d[-1])
    ax_sl.set_ylim(lat_1d[-1], lat_1d[0])
    ax_sl.set_title(f"(b) Pendiente — Horn (1981)\n"
                    f"Media = {np.nanmean(slope):.1f}°  |  "
                    f"Máxima = {np.nanmax(slope):.1f}°",
                    fontweight="bold", fontsize=9)
    ax_sl.set_xlabel("Lon (°)", fontsize=8)
    ax_sl.set_ylabel("Lat (°)", fontsize=8)
    ax_sl.tick_params(labelsize=7)
    ax_sl.set_aspect("equal")

    # ── (c) Curva hipsométrica ───────────────────────────────────────────────
    ax_hy = fig.add_subplot(gs[1, 0])
    area_pct, elev_abs, _ = hypsometric_curve(dem)
    ax_hy.plot(area_pct, elev_abs, color="#2980b9", lw=2.0,
               label="Curva hipsométrica")
    ax_hy.fill_between(area_pct, elev_abs, elev_abs.min(),
                        alpha=0.18, color="#2980b9")
    ax_hy.set_xlabel("Área acumulada desde la cima (%)", fontsize=9)
    ax_hy.set_ylabel("Elevación (m s.n.m.)", fontsize=9)
    if hi >= 0.60:
        etapa = "Joven (HI ≥ 0.60)"
    elif hi >= 0.35:
        etapa = "Maduro (0.35 ≤ HI < 0.60)"
    else:
        etapa = "Viejo / penillanura (HI < 0.35)"
    ax_hy.set_title(f"(c) Curva hipsométrica — Strahler (1952)\n"
                    f"HI = {hi:.3f}  →  {etapa}",
                    fontweight="bold", fontsize=9)
    ax_hy.text(0.98, 0.02,
               f"Elev. mínima: {int(elev_abs.min())} m\n"
               f"Elev. media:  {int(np.nanmean(dem))} m\n"
               f"Elev. máxima: {int(elev_abs.max())} m\n"
               f"Rango total:  {int(elev_abs.max()-elev_abs.min())} m",
               transform=ax_hy.transAxes, ha="right", va="bottom",
               fontsize=8, family="monospace",
               bbox=dict(fc="white", ec="gray", pad=3, lw=0.7))
    ax_hy.grid(alpha=0.3, lw=0.7)
    ax_hy.legend(fontsize=8)

    # ── (d) Elevación por subcuenca ──────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 1])
    df_s  = df_stats.sort_values("elev_mean_m", ascending=True).reset_index(drop=True)
    y     = np.arange(len(df_s))
    ax_sc.barh(y, df_s["elev_max_m"] - df_s["elev_min_m"],
               left=df_s["elev_min_m"], height=0.55,
               color="#aed6f1", edgecolor="gray", lw=0.5, label="Rango (min-max)")
    ax_sc.scatter(df_s["elev_mean_m"], y, color="#2980b9", s=40,
                  zorder=5, label="Elevación media")
    for i, row in df_s.iterrows():
        ax_sc.text(df_s["elev_max_m"].max() + 30, i,
                   f"S̄={row['slope_mean_deg']:.1f}°",
                   va="center", fontsize=6.5, color="#7f8c8d")
    ax_sc.set_yticks(y)
    ax_sc.set_yticklabels([f"SC-{int(sid)}" for sid in df_s["subcuenca_id"]],
                          fontsize=7.5)
    ax_sc.set_xlabel("Elevación (m s.n.m.)", fontsize=9)
    ax_sc.set_title("(d) Perfil de elevación por subcuenca\n"
                    "Rango min-max + media  |  S̄ = pendiente media",
                    fontweight="bold", fontsize=9)
    ax_sc.legend(fontsize=8, loc="lower right")
    ax_sc.grid(axis="x", alpha=0.3, lw=0.7)
    ax_sc.set_xlim(left=0)

    fig.suptitle("Análisis topográfico — Cuenca Chancay-Huaral\n"
                 f"Copernicus GLO-30 ≈ 30 m  |  {len(df_stats)} subcuencas"
                 f"  |  Área = 3 062.62 km²",
                 fontsize=12, fontweight="bold")
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {OUT_FIG.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("Script 10b: DEM Copernicus GLO-30 (30m) — Chancay-Huaral")
    log.info("=" * 70)

    basin_wgs84 = gpd.read_file(SHP_CUE).to_crs("EPSG:4326")

    tile_paths = download_tiles(basin_wgs84)
    dem, lat_1d, lon_1d, res_deg = merge_and_clip(tile_paths, basin_wgs84)
    slope, aspect = compute_slope_aspect(dem, lat_1d, res_deg)
    save_nc(dem, slope, aspect, lat_1d, lon_1d)
    df_stats = stats_by_subcuenca(dem, slope, lat_1d, lon_1d)
    _, _, hi = hypsometric_curve(dem)
    plot_dem(dem, slope, lat_1d, lon_1d, df_stats, hi, n_tiles=len(tile_paths))

    log.info("\n" + "=" * 70)
    log.info("=== DONE: DEM GLO-30 completado ===")
    log.info(f"  Tiles descargados : {len(tile_paths)}")
    log.info(f"  NC salida         : {OUT_NC.name}  ({OUT_NC.stat().st_size/1e6:.1f} MB)")
    log.info(f"  Stats CSV         : {OUT_CSV.name}  ({len(df_stats)} subcuencas)")
    log.info(f"  Figura            : {OUT_FIG.name}")
    log.info(f"  HI cuenca         : {hi:.3f}  (0=viejo · 0.35-0.60=maduro · >0.60=joven)")
    log.info(f"  Elev media        : {np.nanmean(dem):.0f} m s.n.m.")
    log.info(f"  Pendiente media   : {np.nanmean(slope):.1f}°")
    log.info(f"  Resolución        : ≈{abs(res_deg[1])*111320*np.cos(np.radians(abs(np.nanmean(lat_1d)))):.0f} m/pixel")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
