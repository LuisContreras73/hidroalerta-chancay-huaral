#!/usr/bin/env python3
"""
Script 11: Atributos de suelo SoilGrids 250m — Cuenca Chancay-Huaral.

Fuente de datos
---------------
SoilGrids v2.0 (ISRIC — World Soil Information, 2020).
API REST pública sin autenticación: https://rest.isric.org/soilgrids/v2.0/

Variables descargadas (media de cuenca a 4 profundidades)
----------------------------------------------------------
  sand    — contenido de arena                    [g/kg]
  silt    — contenido de limo                     [g/kg]
  clay    — contenido de arcilla                  [g/kg]
  soc     — carbono orgánico del suelo            [dg/kg]
  bdod    — densidad aparente suelo seco          [cg/cm³]
  phh2o   — pH en agua                            [pHx10]
  cec     — capacidad de intercambio catiónico    [mmol(c)/kg]
  cfvo    — volumen fragmentos gruesos            [cm³/dm³]

Profundidades: 0-5cm, 5-15cm, 15-30cm, 30-60cm (relevantes para la modelación)

Metodología
-----------
Se muestrea una grilla regular de puntos dentro de la cuenca (paso ~0.05°,
aproximadamente igual a la resolución SoilGrids) y se calcula la media
ponderada por área (coseno-latitud).  Cada punto se consulta con la API REST.
El resultado es representativo de las condiciones medias de la cuenca.

Nota: para rasters completos de SoilGrids usar WCS (Web Coverage Service).
Esta implementación usa la API puntual que es más simple y robusta.

Salidas
-------
  data/bronze/B5_soilgrids_basin_mean.csv    — medias de cuenca por profundidad
  data/bronze/B5_soilgrids_points.csv        — datos crudos por punto muestreado
  outputs/figures/basin/T02_suelos_analisis.png

Referencias
-----------
Poggio et al. (2021) doi:10.5194/soil-7-217-2021   SoilGrids v2.0
"""
import logging, time, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("soilgrids")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
SHP_CUE = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
           / "Cuenca_Chancay___Huaral.shp")
OUT_MEAN   = ROOT / "data/bronze/B5_soilgrids_basin_mean.csv"
OUT_POINTS = ROOT / "data/bronze/B5_soilgrids_points.csv"
FIG_DIR    = ROOT / "outputs/figures/basin"
OUT_FIG    = FIG_DIR / "T02_suelos_analisis.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── API SoilGrids ─────────────────────────────────────────────────────────────
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES    = ["sand", "silt", "clay", "soc", "bdod", "phh2o", "cec", "cfvo"]
DEPTHS        = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]

PROP_META = {
    "sand"  : {"label": "Arena",       "units": "g/kg",       "factor": 1.0},
    "silt"  : {"label": "Limo",        "units": "g/kg",       "factor": 1.0},
    "clay"  : {"label": "Arcilla",     "units": "g/kg",       "factor": 1.0},
    "soc"   : {"label": "C.O.S.",      "units": "g/kg",       "factor": 0.1},  # dg/kg→g/kg
    "bdod"  : {"label": "Den.Aparen.", "units": "g/cm³",      "factor": 0.01}, # cg/cm³→g/cm³
    "phh2o" : {"label": "pH agua",     "units": "—",          "factor": 0.1},  # pHx10→pH
    "cec"   : {"label": "CIC",         "units": "mmol(c)/kg", "factor": 1.0},
    "cfvo"  : {"label": "Frag.Grus.", "units": "cm³/dm³",    "factor": 1.0},
}

# ── Grilla de muestreo ────────────────────────────────────────────────────────
SAMPLE_STEP = 0.05   # grados (~5.5 km en lat, ~4.5 km en lon a lat -11°)


def _build_sample_points(shp_path: Path, step: float) -> list[tuple[float, float]]:
    """Genera grilla de puntos dentro de la cuenca."""
    basin   = gpd.read_file(shp_path).to_crs("EPSG:4326")
    polygon = basin.geometry.union_all()
    bounds  = basin.total_bounds   # minx, miny, maxx, maxy

    lons = np.arange(bounds[0] + step/2, bounds[2], step)
    lats = np.arange(bounds[1] + step/2, bounds[3], step)

    points = []
    for lat in lats:
        for lon in lons:
            if polygon.contains(Point(lon, lat)):
                points.append((round(float(lat), 4), round(float(lon), 4)))
    log.info(f"  Puntos de muestreo dentro de cuenca: {len(points)}")
    return points


def _query_point(lat: float, lon: float,
                 props: list[str], depths: list[str],
                 retries: int = 3) -> dict | None:
    """Consulta SoilGrids REST API para un punto lat/lon."""
    params = {
        "lon"     : lon,
        "lat"     : lat,
        "property": props,
        "depth"   : depths,
        "value"   : "mean",
    }
    for attempt in range(retries):
        try:
            r = requests.get(SOILGRIDS_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)   # back-off
            else:
                log.warning(f"    HTTP {r.status_code} en ({lat},{lon})")
                return None
        except Exception as e:
            log.warning(f"    Error ({lat},{lon}): {e}")
            time.sleep(1)
    return None


def _parse_response(data: dict) -> dict:
    """Extrae valores medios de la respuesta JSON de SoilGrids."""
    out = {}
    if not data or "properties" not in data:
        return out
    for layer in data["properties"].get("layers", []):
        prop = layer["name"]
        factor = PROP_META.get(prop, {}).get("factor", 1.0)
        for depth_info in layer.get("depths", []):
            depth = depth_info["label"]
            val   = depth_info["values"].get("mean")
            if val is not None and val != -32768:
                out[f"{prop}_{depth}"] = round(float(val) * factor, 4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_soilgrids():
    if OUT_POINTS.exists():
        log.info(f"  Datos ya descargados: {OUT_POINTS.name}")
        return pd.read_csv(OUT_POINTS)

    log.info("Construyendo grilla de muestreo ...")
    points = _build_sample_points(SHP_CUE, SAMPLE_STEP)

    records = []
    for i, (lat, lon) in enumerate(points):
        log.info(f"  Punto {i+1}/{len(points)}: lat={lat}, lon={lon}")
        data = _query_point(lat, lon, PROPERTIES, DEPTHS)
        if data:
            row = {"lat": lat, "lon": lon}
            row.update(_parse_response(data))
            records.append(row)
        time.sleep(0.4)   # respetar rate limit de la API

    df = pd.DataFrame(records)
    df.to_csv(OUT_POINTS, index=False)
    log.info(f"  -> {OUT_POINTS.name}  ({len(df)} puntos, {df.shape[1]} cols)")
    return df


def compute_basin_means(df: pd.DataFrame) -> pd.DataFrame:
    """Media ponderada por área (coseno-latitud) sobre los puntos de cuenca."""
    cos_w = np.cos(np.radians(df["lat"].values))
    w_sum = cos_w.sum()

    prop_cols = [c for c in df.columns if c not in ["lat", "lon"]]
    records   = []
    for col in prop_cols:
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            continue
        wmean = np.sum(vals[mask] * cos_w[mask]) / cos_w[mask].sum()
        # Separar propiedad y profundidad
        parts = col.rsplit("_", 1)
        prop  = parts[0] if len(parts) == 2 else col
        depth = parts[1] if len(parts) == 2 else "—"
        meta  = PROP_META.get(prop, {})
        records.append({
            "property"   : prop,
            "depth"      : depth,
            "label"      : meta.get("label", prop),
            "units"      : meta.get("units", "—"),
            "basin_mean" : round(float(wmean), 4),
            "n_points"   : int(mask.sum()),
        })

    df_mean = pd.DataFrame(records)
    df_mean.to_csv(OUT_MEAN, index=False)
    log.info(f"  -> {OUT_MEAN.name}  ({len(df_mean)} filas)")
    return df_mean


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA T02
# ══════════════════════════════════════════════════════════════════════════════
def plot_soils(df_mean: pd.DataFrame, df_pts: pd.DataFrame):
    log.info("Generando figura T02 ...")

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)

    depths_ordered = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]
    colors_d = ["#1a5276", "#2980b9", "#7fb3d3", "#aed6f1"]

    # ── (a) Textura del suelo (arena/limo/arcilla por profundidad) ────────────
    ax_tex = fig.add_subplot(gs[0, 0])
    texture_props = ["sand", "silt", "clay"]
    texture_labels = ["Arena", "Limo", "Arcilla"]
    x = np.arange(len(depths_ordered))
    width = 0.25
    colors_t = ["#f4d03f", "#a9cce3", "#884ea0"]
    for j, (prop, lbl, col) in enumerate(zip(texture_props, texture_labels, colors_t)):
        vals = []
        for d in depths_ordered:
            row = df_mean[(df_mean["property"] == prop) & (df_mean["depth"] == d)]
            vals.append(float(row["basin_mean"].values[0]) / 10 if len(row) > 0 else np.nan)
        ax_tex.bar(x + j*width, vals, width, label=lbl, color=col,
                   edgecolor="gray", lw=0.5, alpha=0.9)
    ax_tex.set_xticks(x + width)
    ax_tex.set_xticklabels(depths_ordered, fontsize=9)
    ax_tex.set_ylabel("Contenido (%)", fontsize=9)
    ax_tex.set_title("(a) Textura del suelo\nArena / Limo / Arcilla por profundidad",
                     fontweight="bold", fontsize=9)
    ax_tex.legend(fontsize=8); ax_tex.grid(axis="y", alpha=0.3, lw=0.7)
    ax_tex.set_ylim(0, 100)

    # ── (b) SOC y pH por profundidad ─────────────────────────────────────────
    ax_soc = fig.add_subplot(gs[0, 1])
    ax_ph  = ax_soc.twinx()
    for prop, ax_use, col, lbl, ls in [
        ("soc",   ax_soc, "#27ae60", "SOC (g/kg)", "-"),
        ("phh2o", ax_ph,  "#e74c3c", "pH agua",    "--"),
    ]:
        vals = []
        for d in depths_ordered:
            row = df_mean[(df_mean["property"] == prop) & (df_mean["depth"] == d)]
            vals.append(float(row["basin_mean"].values[0]) if len(row) > 0 else np.nan)
        ax_use.plot(depths_ordered, vals, f"o{ls}", color=col,
                    lw=1.8, ms=7, label=lbl)
    ax_soc.set_ylabel("Carbono orgánico (g/kg)", fontsize=9, color="#27ae60")
    ax_ph.set_ylabel("pH agua", fontsize=9, color="#e74c3c")
    ax_soc.tick_params(axis="y", colors="#27ae60")
    ax_ph.tick_params(axis="y", colors="#e74c3c")
    lines1, labs1 = ax_soc.get_legend_handles_labels()
    lines2, labs2 = ax_ph.get_legend_handles_labels()
    ax_soc.legend(lines1+lines2, labs1+labs2, fontsize=8)
    ax_soc.set_title("(b) Carbono orgánico y pH\npor profundidad",
                     fontweight="bold", fontsize=9)
    ax_soc.grid(alpha=0.3, lw=0.7)

    # ── (c) Densidad aparente y CIC ──────────────────────────────────────────
    ax_bd = fig.add_subplot(gs[1, 0])
    ax_cic = ax_bd.twinx()
    for prop, ax_use, col, lbl, ls in [
        ("bdod", ax_bd,  "#8e44ad", "Den. Aparente (g/cm³)", "-"),
        ("cec",  ax_cic, "#e67e22", "CIC (mmol(c)/kg)",      "--"),
    ]:
        vals = []
        for d in depths_ordered:
            row = df_mean[(df_mean["property"] == prop) & (df_mean["depth"] == d)]
            vals.append(float(row["basin_mean"].values[0]) if len(row) > 0 else np.nan)
        ax_use.plot(depths_ordered, vals, f"s{ls}", color=col,
                    lw=1.8, ms=7, label=lbl)
    ax_bd.set_ylabel("Densidad aparente (g/cm³)", fontsize=9, color="#8e44ad")
    ax_cic.set_ylabel("CIC (mmol(c)/kg)", fontsize=9, color="#e67e22")
    ax_bd.tick_params(axis="y", colors="#8e44ad")
    ax_cic.tick_params(axis="y", colors="#e67e22")
    l1, lb1 = ax_bd.get_legend_handles_labels()
    l2, lb2 = ax_cic.get_legend_handles_labels()
    ax_bd.legend(l1+l2, lb1+lb2, fontsize=8)
    ax_bd.set_title("(c) Densidad aparente y CIC\npor profundidad",
                    fontweight="bold", fontsize=9)
    ax_bd.grid(alpha=0.3, lw=0.7)

    # ── (d) Mapa espacial de textura (arena 0-5cm) por puntos ────────────────
    ax_map = fig.add_subplot(gs[1, 1])
    basin  = gpd.read_file(SHP_CUE).to_crs("EPSG:4326")
    basin.boundary.plot(ax=ax_map, color="#2c3e50", lw=1.5, zorder=5)
    col_sand = "sand_0-5cm"
    if col_sand in df_pts.columns:
        vals_s = df_pts[col_sand].values / 10  # g/kg → %
        sc = ax_map.scatter(df_pts["lon"], df_pts["lat"], c=vals_s,
                            cmap="YlOrBr", s=60, alpha=0.85,
                            vmin=0, vmax=100, zorder=4, edgecolors="none")
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm_s = ScalarMappable(cmap="YlOrBr", norm=Normalize(0, 100))
        sm_s.set_array([])
        cb_s = fig.colorbar(sm_s, ax=ax_map, fraction=0.032, pad=0.02)
        cb_s.set_label("Arena 0-5 cm (%)", fontsize=8)
        cb_s.ax.tick_params(labelsize=7)
    ax_map.set_xlabel("Lon (°)", fontsize=8); ax_map.set_ylabel("Lat (°)", fontsize=8)
    ax_map.set_title("(d) Distribución espacial de arena (0-5 cm)\n"
                     "Puntos de muestreo dentro de cuenca",
                     fontweight="bold", fontsize=9)
    ax_map.grid(alpha=0.3, lw=0.6)
    ax_map.set_aspect("equal")
    ax_map.tick_params(labelsize=7)

    fig.suptitle("Atributos de suelo — Cuenca Chancay-Huaral\n"
                 "SoilGrids v2.0 (ISRIC, 250m)  |  Poggio et al. (2021)  |  "
                 f"API REST muestreada en {len(df_pts)} puntos (paso 0.05°)",
                 fontsize=11, fontweight="bold")
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {OUT_FIG.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("Script 11: SoilGrids v2.0 — Cuenca Chancay-Huaral")
    log.info("=" * 70)

    df_pts  = download_soilgrids()
    df_mean = compute_basin_means(df_pts)
    plot_soils(df_mean, df_pts)

    log.info("\n" + "=" * 70)
    log.info("=== DONE: SoilGrids completado ===")
    # Resumen textura
    for prop, lbl in [("sand","Arena"),("silt","Limo"),("clay","Arcilla")]:
        row = df_mean[(df_mean["property"]==prop)&(df_mean["depth"]=="0-5cm")]
        if len(row) > 0:
            v = float(row["basin_mean"].values[0]) / 10
            log.info(f"  {lbl:8s} (0-5cm): {v:.1f}%")
    log.info("Siguiente: Script 12 — ONI ENSO")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
