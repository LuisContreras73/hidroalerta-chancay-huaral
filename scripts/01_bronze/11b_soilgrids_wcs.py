#!/usr/bin/env python3
"""
Script 11b: SoilGrids v2.0 — Rasters reales por subcuenca (WCS)
================================================================
Descarga GeoTIFFs de alta resolución (250 m) desde el Web Coverage
Service (WCS) de ISRIC para el bbox de la cuenca Chancay-Huaral,
calcula estadísticas zonales por subcuenca y genera la figura VM04.

Fuente : SoilGrids v2.0 (ISRIC, Poggio et al. 2021, doi:10.5194/soil-7-217-2021)
WCS    : https://maps.isric.org/mapserv?map=/map/{property}.map
CRS    : descarga en EPSG:4326, clip con shapefiles de subcuencas
Res.   : 250 m (≈ 0.00225°)

Variables descargadas (profundidad 0-5 cm, media)
--------------------------------------------------
  clay   — arcilla      [g/kg]
  sand   — arena        [g/kg]
  silt   — limo         [g/kg]
  soc    — C.O.S.       [dg/kg → g/kg ×0.1]
  bdod   — den.aparente [cg/cm³ → g/cm³ ×0.01]
  phh2o  — pH agua      [pHx10 → pH ×0.1]
  cec    — CIC          [mmol(c)/kg]

Salidas
-------
  data/bronze/B5_soilgrids_rasters/   — GeoTIFF por variable × profundidad
  data/bronze/B5_soilgrids_subcuencas.csv — media/std/p25/p75 por subcuenca
  outputs/figures/VM04_suelos_subcuencas.png — figura de presentación
"""

import io
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS as RioCRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("soilgrids_wcs")

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 9,
})

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
GEOJSON     = ROOT / "outputs" / "subcuencas_wgs84_simplified.geojson"
SHP_BASIN   = (ROOT / "data" / "raw" / "shapefiles" /
               "cuenca_chancay_huaral" / "limite" /
               "Cuenca_Chancay___Huaral.shp")
OUT_RASTERS = ROOT / "data" / "bronze" / "B5_soilgrids_rasters"
OUT_CSV     = ROOT / "data" / "bronze" / "B5_soilgrids_subcuencas.csv"
OUT_FIG     = ROOT / "outputs" / "figures" / "static_maps" / "VM04_suelos.png"

OUT_RASTERS.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "figures" / "static_maps").mkdir(parents=True, exist_ok=True)

# ── Bbox de la cuenca (WGS84) ─────────────────────────────────────────────────
# Basado en los shapefiles: cuenca Chancay-Huaral completa
BBOX = dict(west=-77.50, east=-76.30, south=-11.90, north=-10.80)

# ── Definición de variables SoilGrids ─────────────────────────────────────────
PROPS = {
    "clay":  {"label": "Arcilla",       "units": "g/kg",       "factor": 1.0,   "cmap": "YlOrBr_r", "vmax": 400},
    "sand":  {"label": "Arena",         "units": "g/kg",       "factor": 1.0,   "cmap": "YlOrBr",   "vmax": 700},
    "silt":  {"label": "Limo",          "units": "g/kg",       "factor": 1.0,   "cmap": "RdPu",     "vmax": 400},
    "soc":   {"label": "C.O.S.",        "units": "g/kg",       "factor": 0.1,   "cmap": "YlGn",     "vmax": 80},
    "bdod":  {"label": "Den.Aparente",  "units": "g/cm³",      "factor": 0.01,  "cmap": "Purples",  "vmax": 1.8},
    "phh2o": {"label": "pH agua",       "units": "—",          "factor": 0.1,   "cmap": "RdYlGn",   "vmax": 8.5},
    "cec":   {"label": "CIC",           "units": "mmol(c)/kg", "factor": 1.0,   "cmap": "Blues",    "vmax": 500},
}
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]
DEPTH_MAIN = "0-5cm"   # profundidad principal para los mapas

# ── Metadatos de subcuencas ────────────────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from styles.variable_styles import SUBCUENCA_META, ENTITY_ORDER
    ENTITY_COLORS = {e: SUBCUENCA_META[e]["color"] for e in SUBCUENCA_META}
    ENTITY_NAMES  = {e: SUBCUENCA_META[e]["nombre"] for e in SUBCUENCA_META}
    ENTITY_ELEVS  = {e: SUBCUENCA_META[e]["elev_m"] for e in SUBCUENCA_META}
except Exception:
    ENTITY_COLORS = {}
    ENTITY_NAMES  = {}
    ENTITY_ELEVS  = {}
    ENTITY_ORDER  = []


# ══════════════════════════════════════════════════════════════════════════════
# 1. Descarga WCS
# ══════════════════════════════════════════════════════════════════════════════

def _wcs_url(prop: str, depth: str) -> str:
    """URL WCS 2.0.1 para una variable y profundidad dadas."""
    coverage_id = f"{prop}_{depth}_mean"
    crs_epsg = "http://www.opengis.net/def/crs/EPSG/0/4326"
    params = (
        f"SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={coverage_id}"
        f"&FORMAT=image/tiff"
        f"&SUBSETTINGCRS={crs_epsg}"
        f"&OUTPUTCRS={crs_epsg}"
        f"&SUBSET=long({BBOX['west']},{BBOX['east']})"
        f"&SUBSET=lat({BBOX['south']},{BBOX['north']})"
    )
    base = f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
    return f"{base}&{params}"


def download_raster(prop: str, depth: str, force: bool = False) -> Path:
    """
    Descarga un GeoTIFF de SoilGrids WCS.
    Devuelve la ruta al archivo .tif descargado.
    """
    out = OUT_RASTERS / f"soilgrids_{prop}_{depth.replace('-','_')}_mean.tif"
    if out.exists() and not force:
        log.info(f"  Ya existe: {out.name}")
        return out

    url = _wcs_url(prop, depth)
    log.info(f"  Descargando {prop} {depth} ...")
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and resp.content[:4] in (b"II*\x00", b"MM\x00*", b"II\x2b\x00"):
                out.write_bytes(resp.content)
                sz = out.stat().st_size / 1024
                log.info(f"    -> {out.name}  ({sz:.0f} KB)")
                return out
            elif resp.status_code == 200:
                # Puede ser XML de error
                log.warning(f"    Respuesta inesperada: {resp.text[:200]}")
            else:
                log.warning(f"    HTTP {resp.status_code} (intento {attempt+1})")
        except Exception as ex:
            log.warning(f"    Error intento {attempt+1}: {ex}")
        time.sleep(3 * (attempt + 1))

    log.error(f"  FALLO al descargar {prop} {depth}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Estadísticas zonales por subcuenca
# ══════════════════════════════════════════════════════════════════════════════

def zonal_stats_rasterio(tif_path: Path, gdf: gpd.GeoDataFrame,
                          factor: float = 1.0) -> pd.DataFrame:
    """
    Calcula estadísticas zonales (mean, std, p25, p75, min, max) para
    cada geometría en gdf usando rasterio.mask.
    """
    records = []
    with rasterio.open(tif_path) as src:
        nodata = src.nodata if src.nodata is not None else -32768
        for eid, row in gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = rio_mask(src, geom, crop=True, nodata=nodata,
                                        all_touched=True)
                data = out_image[0].astype(float)
                # Enmascarar nodata y valores inválidos
                data[data == nodata] = np.nan
                data[data < -9000] = np.nan
                data[data > 1e6]   = np.nan
                vals = data[~np.isnan(data)] * factor
                if len(vals) >= 3:
                    records.append({
                        "entity_id": eid,
                        "n_pixels":  len(vals),
                        "mean":  float(np.mean(vals)),
                        "std":   float(np.std(vals)),
                        "p25":   float(np.percentile(vals, 25)),
                        "p75":   float(np.percentile(vals, 75)),
                        "min":   float(np.min(vals)),
                        "max":   float(np.max(vals)),
                    })
                else:
                    log.warning(f"    Pocos píxeles ({len(vals)}) para {eid}")
            except Exception as ex:
                log.warning(f"    Error zonal {eid}: {ex}")
    return pd.DataFrame(records).set_index("entity_id") if records else pd.DataFrame()


def compute_all_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Descarga todos los TIFs y calcula estadísticas por subcuenca."""
    all_records = []

    for prop, meta in PROPS.items():
        for depth in DEPTHS:
            tif = download_raster(prop, depth)
            if tif is None or not tif.exists():
                log.warning(f"  Saltando {prop} {depth}: archivo no disponible")
                continue

            if not HAS_RASTERIO:
                log.error("rasterio no instalado — no se pueden calcular estadísticas")
                continue

            log.info(f"  Calculando estadísticas zonales: {prop} {depth} ...")
            stats = zonal_stats_rasterio(tif, gdf, factor=meta["factor"])
            if stats.empty:
                continue

            for eid, row in stats.iterrows():
                all_records.append({
                    "entity_id": eid,
                    "property":  prop,
                    "depth":     depth,
                    "label":     meta["label"],
                    "units":     meta["units"],
                    **{k: row[k] for k in ["n_pixels","mean","std","p25","p75","min","max"]},
                })
            time.sleep(0.5)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df.to_csv(OUT_CSV, index=False)
        log.info(f"  -> {OUT_CSV.name}  ({len(df)} filas)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Figura VM04
# ══════════════════════════════════════════════════════════════════════════════

def _read_raster_array(prop: str, depth: str):
    """Lee un TIF y devuelve (array_2d, transform, crs, nodata)."""
    tif = OUT_RASTERS / f"soilgrids_{prop}_{depth.replace('-','_')}_mean.tif"
    if not tif.exists() or not HAS_RASTERIO:
        return None, None, None, None
    with rasterio.open(tif) as src:
        data  = src.read(1).astype(float)
        nd    = src.nodata if src.nodata is not None else -32768
        trans = src.transform
        crs   = src.crs
        data[data == nd]   = np.nan
        data[data < -9000] = np.nan
        data[data > 1e6]   = np.nan
    return data, trans, crs, nd


def _extent_from_transform(transform, shape):
    """Calcula [west, east, south, north] desde transform y shape."""
    rows, cols = shape
    west  = transform.c
    north = transform.f
    east  = west  + cols * transform.a
    south = north + rows * transform.e   # transform.e es negativo
    return [west, east, south, north]


def plot_VM04(df_stats: pd.DataFrame, gdf: gpd.GeoDataFrame,
              basin: gpd.GeoDataFrame = None):
    """
    Genera la figura VM04: mapa espacial de textura + estadísticas por subcuenca.

    Layout (16:9, 4 paneles superiores + 2 paneles inferiores):
      Fila 1: mapa arcilla | mapa arena | mapa SOC | mapa pH
      Fila 2: barras textura por subcuenca | perfil profundidad CIC y Den.Aparente
    """
    log.info("Generando figura VM04 ...")

    # ── Colores por entidad ──────────────────────────────────────────────────
    entities_ord = [e for e in ENTITY_ORDER if e in gdf.index]
    if not entities_ord:
        entities_ord = list(gdf.index)

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    fig.patch.set_facecolor("white")
    gs  = GridSpec(2, 4, figure=fig,
                   hspace=0.45, wspace=0.30,
                   left=0.04, right=0.97,
                   top=0.89, bottom=0.08)

    # ── Colores de subcuencas para anotaciones ───────────────────────────────
    def entity_color(eid):
        return ENTITY_COLORS.get(eid, "#8899AA")

    # ── Helper: dibujar mapa raster + choropleth de subcuencas ──────────────
    def draw_soil_map(ax, prop, depth=DEPTH_MAIN, overlay_poly=True):
        meta   = PROPS[prop]
        data, trans, crs, _ = _read_raster_array(prop, depth)

        ax.set_facecolor("#DDE8F0")
        if basin is not None:
            basin.boundary.plot(ax=ax, color="#7A9AB0", linewidth=1.0, zorder=2)

        if data is not None:
            factor = meta["factor"]
            arr    = data * factor
            ext    = _extent_from_transform(trans, data.shape)
            cmap   = matplotlib.colormaps[meta["cmap"]]
            norm   = mcolors.Normalize(vmin=0, vmax=meta["vmax"])
            masked = np.ma.masked_invalid(arr)
            ax.imshow(masked, extent=ext, cmap=cmap, norm=norm,
                      origin="upper", aspect="auto", zorder=1, interpolation="bilinear")

            # Colorbar pequeño
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.038, pad=0.02, shrink=0.85)
            cb.set_label(meta["units"], fontsize=7)
            cb.ax.tick_params(labelsize=6.5)

            # Overlay: contornos de subcuencas
            gdf.boundary.plot(ax=ax, color="white", linewidth=1.2, zorder=4)
            gdf.boundary.plot(ax=ax, color="#333355", linewidth=0.5, zorder=5)

            # Etiquetas con valor medio por subcuenca
            if not df_stats.empty and prop in df_stats["property"].values:
                sub = df_stats[(df_stats.property==prop) &
                               (df_stats.depth==depth)].set_index("entity_id")
                gdf_utm = gdf.to_crs(epsg=32718)
                gdf["cx"] = gdf_utm.geometry.centroid.to_crs(epsg=4326).x
                gdf["cy"] = gdf_utm.geometry.centroid.to_crs(epsg=4326).y
                for eid, row_ in gdf.iterrows():
                    if eid in sub.index:
                        val = sub.loc[eid, "mean"]
                        ax.annotate(
                            f"{val:.0f}" if factor >= 1 else f"{val:.2f}",
                            (row_["cx"], row_["cy"]),
                            fontsize=7, ha="center", va="center",
                            fontweight="bold", color="#FFFFFF", zorder=8,
                            path_effects=[pe.withStroke(linewidth=2.2,
                                                        foreground="#111122")],
                        )
        else:
            # Fallback: choropleth con CSV basin_mean si no hay raster
            if not df_stats.empty and prop in df_stats["property"].values:
                sub = df_stats[(df_stats.property==prop) &
                               (df_stats.depth==depth)].set_index("entity_id")
                cmap_  = matplotlib.colormaps[meta["cmap"]]
                norm_  = mcolors.Normalize(vmin=0, vmax=meta["vmax"])
                colors_ = [cmap_(norm_(sub.loc[e,"mean"])) if e in sub.index
                           else (0.8,0.8,0.85,1) for e in gdf.index]
                gdf.plot(ax=ax, color=colors_, edgecolor="white",
                         linewidth=0.8, alpha=0.92, zorder=2)
            else:
                gdf.plot(ax=ax, color="#CCCCCC", edgecolor="white",
                         linewidth=0.8, zorder=2)

        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5, colors="#666677", pad=1)
        ax.set_xlabel("Lon", fontsize=6.5, color="#666677", labelpad=1)
        ax.set_ylabel("Lat", fontsize=6.5, color="#666677", labelpad=1)
        for sp in ax.spines.values():
            sp.set_edgecolor("#BBBBCC")
        lbl = f"{meta['label']} ({depth})"
        ax.set_title(lbl, fontsize=9.5, fontweight="bold", pad=5, color="#1A2A4A")

    # ── Fila 1: 4 mapas ──────────────────────────────────────────────────────
    maps_to_draw = ["clay", "sand", "soc", "phh2o"]
    for col_i, prop in enumerate(maps_to_draw):
        ax = fig.add_subplot(gs[0, col_i])
        draw_soil_map(ax, prop)
        # Flecha N
        ax.annotate("N", xy=(0.93, 0.93), xytext=(0.93, 0.84),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=8, fontweight="bold", ha="center", color="#222233",
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#222233"))

    # ── Fila 2a: Barras de textura por subcuenca ─────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :2])
    ax_bar.set_facecolor("#F4F7FA")

    names_ord  = [ENTITY_NAMES.get(e, e) for e in entities_ord]
    elevs_ord  = [ENTITY_ELEVS.get(e, 0)  for e in entities_ord]
    n_ent = len(entities_ord)

    if not df_stats.empty:
        x = np.arange(n_ent)
        w = 0.27
        tex_props = [("clay","Arcilla","#8B6914"),
                     ("sand","Arena","#DEB887"),
                     ("silt","Limo","#7799AA")]
        for offset, (prop, lbl, col) in enumerate(tex_props):
            sub = df_stats[(df_stats.property==prop) &
                           (df_stats.depth==DEPTH_MAIN)].set_index("entity_id")
            vals = [sub.loc[e,"mean"]/10 if e in sub.index else 0
                    for e in entities_ord]   # g/kg → %
            errs = [sub.loc[e,"std"]/10  if e in sub.index else 0
                    for e in entities_ord]
            bars = ax_bar.bar(x + (offset - 1)*w, vals, w,
                              label=lbl, color=col,
                              edgecolor="white", linewidth=0.4,
                              alpha=0.88, yerr=errs,
                              error_kw=dict(capsize=2.5, capthick=0.8,
                                            elinewidth=0.8, ecolor="#555"))

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(
            [f"{n}\n({e}m)" for n, e in zip(names_ord, elevs_ord)],
            fontsize=7.5, rotation=20, ha="right")
        ax_bar.set_ylabel("Contenido (%)", fontsize=9, labelpad=3)
        ax_bar.set_ylim(0, 80)
        ax_bar.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.85)
        ax_bar.set_title(
            "Textura del suelo por sub-cuenca (0-5 cm) · SoilGrids 250 m · Barras = ±1σ",
            fontsize=9.5, fontweight="bold", pad=6, color="#1A2A4A")
        for sp in ax_bar.spines.values():
            sp.set_edgecolor("#CCCCDD")
        ax_bar.tick_params(colors="#444455")
        ax_bar.grid(axis="y", color="#DDDDEE", linewidth=0.5, alpha=0.7)
    else:
        ax_bar.text(0.5, 0.5, "Sin datos de estadísticas por subcuenca\n"
                    "(ejecutar descarga WCS primero)",
                    ha="center", va="center", transform=ax_bar.transAxes,
                    fontsize=11, color="#AA5522")
        ax_bar.set_title("Textura por sub-cuenca", fontsize=9.5, fontweight="bold")

    # ── Fila 2b: Perfil de profundidad — SOC y CIC ───────────────────────────
    ax_prof = fig.add_subplot(gs[1, 2:])
    ax_prof.set_facecolor("#F4F7FA")
    ax2_prof = ax_prof.twiny()

    depth_labels = ["0-5", "5-15", "15-30", "30-60"]
    depth_mids   = [2.5, 10.0, 22.5, 45.0]   # cm

    if not df_stats.empty:
        # SOC promedio de cuenca
        soc_sub = df_stats[df_stats.property=="soc"]
        if not soc_sub.empty:
            soc_by_d = soc_sub.groupby("depth")["mean"].mean()
            soc_vals  = [soc_by_d.get(f"{d}cm", np.nan) for d in DEPTHS]
            ax_prof.plot(soc_vals, depth_mids, "o-",
                         color="#27AE60", linewidth=2, markersize=7,
                         label="SOC (g/kg)", zorder=4)
            ax_prof.fill_betweenx(depth_mids,
                                  soc_sub.groupby("depth")["p25"].mean().values,
                                  soc_sub.groupby("depth")["p75"].mean().values,
                                  color="#27AE60", alpha=0.15)

        # CIC promedio de cuenca
        cec_sub = df_stats[df_stats.property=="cec"]
        if not cec_sub.empty:
            cec_by_d = cec_sub.groupby("depth")["mean"].mean()
            cec_vals  = [cec_by_d.get(f"{d}cm", np.nan) for d in DEPTHS]
            ax2_prof.plot(cec_vals, depth_mids, "s--",
                          color="#1565C0", linewidth=2, markersize=7,
                          label="CIC [mmol(c)/kg]", zorder=4)

    ax_prof.set_ylabel("Profundidad (cm)", fontsize=9, labelpad=3)
    ax_prof.set_xlabel("SOC (g/kg)", fontsize=9, color="#27AE60", labelpad=3)
    ax2_prof.set_xlabel("CIC [mmol(c)/kg]", fontsize=9, color="#1565C0", labelpad=3)
    ax_prof.invert_yaxis()
    ax_prof.tick_params(axis="x", colors="#27AE60", labelsize=8)
    ax2_prof.tick_params(axis="x", colors="#1565C0", labelsize=8)
    ax_prof.tick_params(axis="y", labelsize=8, colors="#444455")
    ax_prof.set_title(
        "Perfil de profundidad\n(promedio cuenca: SOC + CIC)",
        fontsize=9.5, fontweight="bold", pad=8, color="#1A2A4A")
    ax_prof.grid(color="#DDDDEE", linewidth=0.5, alpha=0.7)
    # Leyenda combinada
    lines1, labs1 = ax_prof.get_legend_handles_labels()
    lines2, labs2 = ax2_prof.get_legend_handles_labels()
    ax_prof.legend(lines1 + lines2, labs1 + labs2,
                   fontsize=8, loc="lower right", framealpha=0.85)
    for sp in ax_prof.spines.values():
        sp.set_edgecolor("#CCCCDD")

    # ── Suptítulo ─────────────────────────────────────────────────────────────
    n_pts = (
        f" | {len(df_stats['entity_id'].unique())} sub-cuencas"
        if not df_stats.empty and "entity_id" in df_stats.columns
        else ""
    )
    fig.suptitle(
        "Propiedades del Suelo — Cuenca Chancay-Huaral\n"
        f"SoilGrids v2.0 (ISRIC, 250 m)  ·  Poggio et al. (2021)  ·  WCS bbox {BBOX['west']}°/{BBOX['east']}°{n_pts}",
        fontsize=12, fontweight="bold", color="#1A2A4A", y=0.97,
    )

    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  -> {OUT_FIG.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("Script 11b: SoilGrids WCS — Rasters por subcuenca")
    log.info("=" * 70)

    if not HAS_RASTERIO:
        log.error("rasterio no instalado. Instalar con: pip install rasterio")
        log.error("Continuando solo con descarga de rasters...")

    # ── Cargar geometrías ─────────────────────────────────────────────────────
    if not GEOJSON.exists():
        log.error(f"GeoJSON no encontrado: {GEOJSON}")
        return
    gdf = gpd.read_file(GEOJSON).set_index("entity_id")
    log.info(f"  Subcuencas: {len(gdf)} entidades")

    basin = None
    if SHP_BASIN.exists():
        basin = gpd.read_file(SHP_BASIN).to_crs(epsg=4326)

    # ── Cargar CSV previo si existe ───────────────────────────────────────────
    df_stats = pd.DataFrame()
    if OUT_CSV.exists():
        log.info(f"  Cargando estadísticas previas: {OUT_CSV.name}")
        df_stats = pd.read_csv(OUT_CSV)
        log.info(f"  -> {len(df_stats)} filas × {len(df_stats.columns)} cols")

    # ── Descargar rasters que falten ──────────────────────────────────────────
    missing_rasters = []
    for prop in PROPS:
        for depth in DEPTHS:
            tif = OUT_RASTERS / f"soilgrids_{prop}_{depth.replace('-','_')}_mean.tif"
            if not tif.exists():
                missing_rasters.append((prop, depth))

    if missing_rasters:
        log.info(f"\nDescargando {len(missing_rasters)} rasters faltantes ...")
        for prop, depth in missing_rasters:
            download_raster(prop, depth)
            time.sleep(0.5)
    else:
        log.info("  Todos los rasters ya descargados.")

    # ── Calcular estadísticas si no están ─────────────────────────────────────
    if df_stats.empty and HAS_RASTERIO:
        log.info("\nCalculando estadísticas zonales por subcuenca ...")
        df_stats = compute_all_stats(gdf)
    elif not df_stats.empty:
        log.info("  Usando estadísticas previas del CSV.")
    else:
        log.info("  Sin rasterio — se generará figura con rasters solo (sin stats por subcuenca)")

    # ── Generar figura VM04 ───────────────────────────────────────────────────
    log.info("\nGenerando figura VM04 ...")
    plot_VM04(df_stats, gdf, basin)

    # ── Resumen ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("=== DONE: Script 11b completado ===")
    if not df_stats.empty:
        # Textura promedio de cuenca (0-5cm)
        for prop in ["clay", "sand", "silt"]:
            sub = df_stats[(df_stats.property==prop) & (df_stats.depth==DEPTH_MAIN)]
            if not sub.empty:
                val = sub["mean"].mean() / 10
                log.info(f"  {PROPS[prop]['label']:12s} (0-5cm): {val:.1f}%")
    log.info(f"  Figura: {OUT_FIG}")
    log.info(f"  CSV   : {OUT_CSV}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
