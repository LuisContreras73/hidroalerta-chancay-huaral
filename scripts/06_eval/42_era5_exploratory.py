#!/usr/bin/env python3
"""
Script 42: Exploración visual ERA5-Land (años disponibles en data/raw/era5/daily/parts/)
Genera 4 figuras para verificar y entender las variables descargadas.

  ERA01_series_temporales.png   — Series diarias y ciclo anual de vars clave
  ERA02_ciclo_estacional.png    — Climatología mensual por variable y grupo
  ERA03_mapas_espaciales.png    — Mapas medios (bbox cuenca, con máscara shapefile)
  ERA04_correlaciones.png       — Matriz de correlación entre variables cuenca-mean

Ejecutar: python scripts/42_era5_exploratory.py
  — Detecta automáticamente todos los archivos disponibles.
  — Cuando ERA5 1996-2020 esté descargado, re-ejecutar sin cambios.

NOTAS TÉCNICAS (bugs CDS corregidos aquí, aplicar a otros scripts que lean era5):
  1. accum2/inst2 tienen lons duplicados (-180/180 y 0/360 simultáneamente).
     La mitad con datos reales varía según la variable → usar _merge_dup_lons()
     que aplica nanmean sobre todas las ocurrencias de cada lon único.
  2. LON_MIN=-77.50 (no -77.30): los archivos ERA5 cubren hasta -77.5°; el valor
     anterior excluía ~3 píxeles del Bajo Chancay en los mapas espaciales.
  3. _bbox_idx usa eps=1e-6 para evitar fallos por precisión float en los límites.
  4. _basin_mask_spatial aplica buffer(0.05°) al polígono para capturar píxeles
     cuyos bordes tocan la cuenca aunque su centro quede afuera (cachitos).
  5. snowc (cubierta de nieve): usar stat="max" en mapas espaciales para visualizar
     el gradiente altitudinal (media anual ~3% — demasiado bajo para ver patrón).
"""

import re
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("era5_explore")

ROOT     = Path(__file__).parent.parent
PARTS    = ROOT / "data/raw/era5/daily/parts"
OUT_DIR  = ROOT / "reports/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cuenca bbox (WGS84) — expandido hasta el borde real de los archivos ERA5
# Los archivos tienen lons desde -77.5, por eso LON_MIN=-77.50 (antes -77.30 cortaba 3 px del Bajo Chancay)
LAT_MIN, LAT_MAX = -11.90, -10.80
LON_MIN, LON_MAX = -77.50, -76.40

SHP_CUENCA    = ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite/Cuenca_Chancay___Huaral.shp"
SHP_SUBCUENCAS = ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas/Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_geogpsperu_UH_MENORES.shp"

# ── Definición de variables por grupo ────────────────────────────────────────
VARS = {
    "accum1": {
        "tp":   {"label": "Precip. total (tp)",   "unit": "mm/día",       "scale": 1000,  "color": "#2196F3"},
        "e":    {"label": "Evaporación (e)",       "unit": "mm/día",       "scale": -1000, "color": "#00BCD4"},
        "pev":  {"label": "Evapot. pot. (pev)",    "unit": "mm/día",       "scale": -1000, "color": "#4CAF50"},
        "ro":   {"label": "Escorrentía (ro)",      "unit": "mm/día",       "scale": 1000,  "color": "#795548"},
        "sro":  {"label": "Esc. superficial (sro)","unit": "mm/día",       "scale": 1000,  "color": "#A1887F"},
    },
    "accum2": {
        "ssrd": {"label": "Rad. solar ↓ (ssrd)",   "unit": "MJ/m²/día",   "scale": 1e-6,  "color": "#FF9800"},
        "strd": {"label": "Rad. térmica ↓ (strd)", "unit": "MJ/m²/día",   "scale": 1e-6,  "color": "#FF5722"},
        "ssr":  {"label": "Rad. solar neta (ssr)",  "unit": "MJ/m²/día",   "scale": 1e-6,  "color": "#FFC107"},
    },
    "inst1": {
        "t2m":   {"label": "Temp. 2m (t2m)",       "unit": "°C",          "scale": 1, "offset": -273.15, "color": "#F44336"},
        "d2m":   {"label": "Temp. rocío (d2m)",     "unit": "°C",          "scale": 1, "offset": -273.15, "color": "#E91E63"},
        "u10":   {"label": "Viento U 10m",          "unit": "m/s",         "scale": 1,  "color": "#9C27B0"},
        "v10":   {"label": "Viento V 10m",          "unit": "m/s",         "scale": 1,  "color": "#673AB7"},
        "swvl1": {"label": "Humedad suelo L1 (swvl1)","unit": "m³/m³",    "scale": 1,  "color": "#8BC34A"},
        "swvl2": {"label": "Humedad suelo L2 (swvl2)","unit": "m³/m³",    "scale": 1,  "color": "#558B2F"},
        "sp":    {"label": "Presión superf. (sp)",  "unit": "hPa",         "scale": 0.01, "color": "#607D8B"},
    },
    "inst2": {
        "stl1":  {"label": "Temp. suelo L1 (stl1)", "unit": "°C",         "scale": 1, "offset": -273.15, "color": "#795548"},
        "snowc": {"label": "Cubierta nieve (%)",     "unit": "%",          "scale": 1,  "color": "#90CAF9"},
        "sd":    {"label": "Equivalente agua nieve", "unit": "m",          "scale": 1,  "color": "#1565C0"},
        "lai_lv":{"label": "LAI veg. baja",          "unit": "m²/m²",     "scale": 1,  "color": "#388E3C"},
        "lai_hv":{"label": "LAI veg. alta",          "unit": "m²/m²",     "scale": 1,  "color": "#1B5E20"},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Lectura de datos
# ═══════════════════════════════════════════════════════════════════════════════

def _lon_fix(lons: np.ndarray) -> np.ndarray:
    """Convierte longitudes 0-360 a -180 a 180."""
    lons = np.array(lons, dtype=float)
    lons[lons > 180] -= 360
    return lons


def _bbox_idx(lats, lons):
    eps = 1e-6  # tolerancia floating-point (< resolución ERA5 de 0.1°)
    li = np.where((lats >= LAT_MIN - eps) & (lats <= LAT_MAX + eps))[0]
    lj = np.where((lons >= LON_MIN - eps) & (lons <= LON_MAX + eps))[0]
    return li, lj


def _read_group_years(group: str, var_names: list[str]) -> pd.DataFrame:
    """Lee todos los años disponibles para un grupo y devuelve DataFrame diario (basin-mean)."""
    pattern = re.compile(rf"era5_{group}_(\d{{4}})\.nc")
    files   = sorted(
        (f for f in PARTS.iterdir() if pattern.match(f.name)),
        key=lambda f: int(pattern.match(f.name).group(1)),
    )
    if not files:
        log.warning(f"No files for group {group}")
        return pd.DataFrame()

    # Acumular una lista de dicts {var: array} por año
    yearly_frames = []
    for f in files:
        year = int(pattern.match(f.name).group(1))
        log.info(f"  Leyendo {f.name} ...")
        try:
            ds = nc.Dataset(f)
            lats     = np.array(ds["latitude"][:])
            lons_raw = np.array(ds["longitude"][:])
            unique_lons = _unique_lons(lons_raw)
            li, lj = _bbox_idx(lats, unique_lons)

            # Fechas
            time_var = ds["time"]
            try:
                times = nc.num2date(time_var[:], time_var.units,
                                    only_use_cftime_datetimes=False,
                                    only_use_python_datetimes=True)
                dates = pd.to_datetime([t.strftime("%Y-%m-%d") for t in times])
            except Exception:
                n_days = len(time_var)
                dates = pd.date_range(f"{year}-01-01", periods=n_days)

            row = {"date": dates}
            for vn in var_names:
                if vn not in ds.variables:
                    continue
                meta   = VARS[group].get(vn, {})
                scale  = meta.get("scale",  1.0)
                offset = meta.get("offset", 0.0)
                raw_full = np.array(ds[vn][:], dtype=float)
                fill = getattr(ds[vn], "_FillValue", None)
                if fill is not None:
                    raw_full[np.abs(raw_full - fill) < 1e-3 * abs(fill + 1e-10)] = np.nan
                # Colapsar duplicados con nanmean, luego recortar bbox
                _, raw_merged = _merge_dup_lons(lons_raw, raw_full)
                raw = raw_merged[:, li[0]:li[-1]+1, lj[0]:lj[-1]+1]
                basin_mean = np.nanmean(raw.reshape(len(dates), -1), axis=1)
                row[vn]    = basin_mean * scale + offset

            ds.close()
            if len(row) > 1:   # tiene al menos una variable
                yearly_frames.append(pd.DataFrame(row).set_index("date"))
        except Exception as ex:
            log.warning(f"  Error en {f.name}: {ex}")

    if not yearly_frames:
        return pd.DataFrame()

    return pd.concat(yearly_frames, axis=0).sort_index()


def _period_str(data: dict) -> str:
    """Devuelve '1981-2007' calculado desde los DataFrames realmente cargados."""
    years = []
    for df in data.values():
        if not df.empty:
            years += [df.index.min().year, df.index.max().year]
    return f"{min(years)}-{max(years)}" if years else "?"


def load_all_data() -> dict[str, pd.DataFrame]:
    """Carga todos los grupos disponibles y retorna dict de DataFrames."""
    data = {}
    for group, var_dict in VARS.items():
        log.info(f"Cargando grupo {group} ...")
        vars_in_group = list(var_dict.keys())
        df = _read_group_years(group, vars_in_group)
        if not df.empty:
            # Calcular velocidad del viento si disponible
            if group == "inst1" and "u10" in df.columns and "v10" in df.columns:
                df["wspd"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        data[group] = df
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Figura 1: Series de tiempo + ciclo anual de variables clave
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ERA01_series(data: dict):
    """Panel de 8 series temporales con media móvil 30d y ciclo anual inset."""

    KEY_VARS = [
        ("accum1", "tp",    "Precip. ERA5 (tp)",    "mm/día"),
        ("accum1", "pev",   "ETP ERA5 (pev)",        "mm/día"),
        ("accum1", "ro",    "Escorrentía (ro)",       "mm/día"),
        ("inst1",  "t2m",   "Temperatura 2m",         "°C"),
        ("inst1",  "wspd",  "Velocidad viento 10m",   "m/s"),
        ("inst1",  "swvl1", "Humedad suelo L1",        "m³/m³"),
        ("accum2", "ssrd",  "Rad. solar ↓ (ssrd)",    "MJ/m²/día"),
        ("inst2",  "snowc", "Cubierta de nieve",       "%"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
    fig.patch.set_facecolor("#F8F9FA")
    period = _period_str(data)
    fig.suptitle(
        f"ERA5-Land — Series de tiempo diarias ({period})\n"
        "Media cuenca Chancay-Huaral (bbox 0.1°)",
        fontsize=13, fontweight="bold",
    )

    axes_flat = axes.flatten()
    for idx, (grp, vn, label, unit) in enumerate(KEY_VARS):
        ax = axes_flat[idx]
        ax.set_facecolor("#FFFFFF")
        df = data.get(grp, pd.DataFrame())
        col = vn if vn != "wspd" else "wspd"
        if df.empty or col not in df.columns:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=9)
            continue

        s = df[col].dropna()
        if len(s) == 0:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=9)
            continue
        color = VARS[grp].get(vn, {}).get("color", "#555") if vn != "wspd" else "#9C27B0"

        # Datos diarios (fondo ligero)
        ax.fill_between(s.index, s.values, alpha=0.18, color=color)
        ax.plot(s.index, s.values, lw=0.4, color=color, alpha=0.4)

        # Media móvil 30 días
        smooth = s.rolling(30, center=True, min_periods=15).mean()
        ax.plot(smooth.index, smooth.values, lw=1.6, color=color, label="MM 30d")

        # Estadísticas
        ax.set_title(f"{label} [{unit}]", fontsize=9, fontweight="bold", pad=4)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(axis="both", labelsize=7.5)

        # Línea de media
        mean_val = s.mean()
        ax.axhline(mean_val, color=color, ls="--", lw=0.9, alpha=0.7)
        ax.text(s.index[-1], mean_val, f" μ={mean_val:.2f}", va="center",
                fontsize=7, color=color)

        # Inset: ciclo mensual (climatología)
        ax_in = ax.inset_axes([0.70, 0.55, 0.28, 0.40])
        monthly = s.groupby(s.index.month).mean()
        ax_in.bar(monthly.index, monthly.values, color=color, alpha=0.8, width=0.8)
        ax_in.set_xticks(range(1, 13, 2))
        ax_in.set_xticklabels(["E","M","M","J","S","N"], fontsize=6)
        ax_in.tick_params(axis="y", labelsize=6)
        ax_in.set_facecolor("#F5F5F5")
        ax_in.spines["top"].set_visible(False)
        ax_in.spines["right"].set_visible(False)
        ax_in.set_title("Ciclo anual", fontsize=6, pad=2)

        # Años El Niño moderado-fuerte (adaptativo al rango real de datos)
        nino_years = [1983, 1987, 1992, 1994, 1998, 2003, 2007]
        for yr in nino_years:
            t0 = pd.Timestamp(f"{yr}-01-01")
            t1 = pd.Timestamp(f"{yr}-12-31")
            if t0 >= s.index.min() and t1 <= s.index.max():
                ax.axvspan(t0, t1, alpha=0.07, color="#F44336")
        if idx == 0 and pd.Timestamp("1983-01-01") >= s.index.min():
            ax.text(pd.Timestamp("1983-06-01"), s.max() * 0.85,
                    "Niño", color="#C62828", fontsize=6.5, style="italic")

    # Etiqueta de años en eje X del primer subplot
    for idx in range(len(KEY_VARS)):
        axes_flat[idx].xaxis.set_major_locator(
            matplotlib.dates.YearLocator(2)
        )
        axes_flat[idx].xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%Y")
        )
        plt.setp(axes_flat[idx].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "ERA01_series_temporales.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"ERA01 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Figura 2: Climatología mensual por grupo de variables
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ERA02_climatologia(data: dict):
    """Ciclos estacionales por grupo — 4 paneles, uno por grupo."""
    MONTH_LABELS = ["Ene","Feb","Mar","Abr","May","Jun",
                    "Jul","Ago","Sep","Oct","Nov","Dic"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#F8F9FA")
    period = _period_str(data)
    fig.suptitle(
        f"ERA5-Land — Climatología Mensual {period}\n"
        "Cuenca Chancay-Huaral — cada línea es una variable del grupo",
        fontsize=12, fontweight="bold",
    )

    GROUP_ORDER = ["accum1", "accum2", "inst1", "inst2"]
    GROUP_LABELS = {
        "accum1": "Fluxes de agua (mm/día)",
        "accum2": "Radiación (MJ/m²/día)",
        "inst1":  "Variables instantáneas: T, viento, humedad suelo",
        "inst2":  "Suelo, nieve y vegetación",
    }

    for idx, grp in enumerate(GROUP_ORDER):
        ax = axes.flatten()[idx]
        ax.set_facecolor("#FFFFFF")
        df = data.get(grp, pd.DataFrame())
        ax.set_title(GROUP_LABELS[grp], fontsize=10, fontweight="bold")

        if df.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue

        ax2 = None  # eje derecho para temperatura si es inst1
        lines_left, lines_right = [], []

        for vn, meta in VARS[grp].items():
            col = vn
            if col not in df.columns:
                continue
            s = df[col].dropna()
            monthly = s.groupby(s.index.month).agg(["mean", "std"])

            color = meta["color"]
            lbl   = meta["label"].split("(")[0].strip()
            unit  = meta["unit"]

            # Temperatura en eje derecho para inst1
            if grp == "inst1" and "°C" in unit:
                if ax2 is None:
                    ax2 = ax.twinx()
                    ax2.set_ylabel("°C", fontsize=8, color="#F44336")
                    ax2.tick_params(axis="y", labelcolor="#F44336", labelsize=7.5)
                    ax2.spines["right"].set_color("#F44336")
                l, = ax2.plot(monthly.index, monthly["mean"], lw=2.0,
                               color=color, label=lbl, marker="o", markersize=3)
                ax2.fill_between(monthly.index,
                                  monthly["mean"] - monthly["std"],
                                  monthly["mean"] + monthly["std"],
                                  color=color, alpha=0.10)
                lines_right.append(l)
            else:
                l, = ax.plot(monthly.index, monthly["mean"], lw=2.0,
                              color=color, label=lbl, marker="o", markersize=3)
                ax.fill_between(monthly.index,
                                 monthly["mean"] - monthly["std"],
                                 monthly["mean"] + monthly["std"],
                                 color=color, alpha=0.12)
                lines_left.append(l)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.axvspan(12, 12.5, alpha=0.05, color="blue")
        ax.axvspan(0.5, 4.5, alpha=0.05, color="blue", label="_nolegend_")

        # Anotar estaciones
        ax.text(2.0, ax.get_ylim()[1] * 0.97, "Húmedo", color="#1565C0",
                fontsize=7, style="italic", ha="center")
        ax.text(7.5, ax.get_ylim()[1] * 0.97, "Seco", color="#E65100",
                fontsize=7, style="italic", ha="center")

        all_lines = lines_left + lines_right
        if all_lines:
            ax.legend(handles=all_lines, fontsize=7, loc="upper right",
                      framealpha=0.85, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "ERA02_ciclo_estacional.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"ERA02 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Figura 3: Mapas espaciales medios 1981-1995
# ═══════════════════════════════════════════════════════════════════════════════

def _unique_lons(lons_raw: np.ndarray) -> np.ndarray:
    """Retorna array ordenado de longitudes únicas tras convertir 0/360 → -180/180."""
    return np.sort(np.unique(np.round(_lon_fix(lons_raw), 4)))


def _merge_dup_lons(lons_raw: np.ndarray,
                    raw_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Colapsa longitudes duplicadas de raw_full usando nanmean por ocurrencias.

    Bug CDS confirmado en accum2/inst2: cada lon aparece 2 veces (formato -180/180
    y 0/360). Para distintas variables, los datos reales pueden estar en la primera
    o segunda ocurrencia; la otra suele ser todo-NaN.
    nanmean([NaN, valor]) = valor; nanmean([valor, valor]) = valor ← robusto.
    Retorna (lons_unicas_sorted, raw_merged con shape [..., n_unique]).
    """
    lons_fix = _lon_fix(lons_raw)
    lons_rounded = np.round(lons_fix, 4)
    unique_vals = np.sort(np.unique(lons_rounded))

    if len(unique_vals) == len(lons_raw):
        return unique_vals, raw_full  # sin duplicados — accum1/inst1

    n_t, n_lat = raw_full.shape[:2]
    raw_merged = np.full((n_t, n_lat, len(unique_vals)), np.nan)
    for j, ulon in enumerate(unique_vals):
        occ = np.where(lons_rounded == ulon)[0]
        raw_merged[:, :, j] = np.nanmean(raw_full[:, :, occ], axis=2)
    return unique_vals, raw_merged


def _basin_mask_spatial(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Construye máscara bool (n_lat × n_lon): True si la celda de 0.1°
    intersecta el polígono de la cuenca, incluyendo protuberancias delgadas.
    Se aplica un buffer de 0.05° al polígono para capturar pixels en bordes.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
        shp     = gpd.read_file(SHP_CUENCA).to_crs("EPSG:4326")
        polygon = shp.geometry.union_all()
        # Buffer 0.05° (~5 km) para incluir pixels que tocan el borde
        polygon_ext = polygon.buffer(0.05)
        mask    = np.zeros((len(lats), len(lons)), dtype=bool)
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                cell = box(lo - 0.05, la - 0.05, lo + 0.05, la + 0.05)
                mask[i, j] = polygon_ext.intersects(cell)
        return mask
    except Exception as ex:
        log.warning(f"No se pudo construir máscara de cuenca: {ex}")
        return np.ones((len(lats), len(lons)), dtype=bool)


def _read_spatial_mean(group: str, var_name: str,
                       stat: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (lats, lons, campo_espacial_1981-1990) para una variable.
    - stat="mean"  → media temporal por pixel (default)
    - stat="max"   → máximo temporal por pixel (útil para nieve)
    - stat="jja"   → media de meses JJA (pico de nieve en Andes)
    - Deduplica longitudes (bug CDS en accum2/inst2).
    - Aplica máscara del shapefile: píxeles fuera de la cuenca → NaN.
    """
    pattern = re.compile(rf"era5_{group}_(\d{{4}})\.nc")
    files   = sorted(
        (f for f in PARTS.iterdir() if pattern.match(f.name)),
        key=lambda f: int(pattern.match(f.name).group(1)),
    )
    accum  = None
    count  = 0
    lats_o = lons_o = None
    mask   = None
    meta   = VARS[group].get(var_name, {})
    scale  = meta.get("scale",  1.0)
    offset = meta.get("offset", 0.0)

    for f in files[:10]:
        try:
            ds       = nc.Dataset(f)
            lats     = np.array(ds["latitude"][:])
            lons_raw = np.array(ds["longitude"][:])
            unique_lons = _unique_lons(lons_raw)
            li, lj = _bbox_idx(lats, unique_lons)

            if var_name not in ds.variables:
                ds.close()
                continue

            # Leer, colapsar duplicados con nanmean, recortar bbox
            raw_full = np.array(ds[var_name][:], dtype=float)
            fill = getattr(ds[var_name], "_FillValue", None)
            if fill is not None:
                raw_full[np.abs(raw_full - fill) < 1e-3 * abs(fill + 1e-10)] = np.nan
            _, raw_merged = _merge_dup_lons(lons_raw, raw_full)
            raw = raw_merged[:, li[0]:li[-1]+1, lj[0]:lj[-1]+1]

            raw = raw * scale + offset
            if stat == "max":
                yr_mean = np.nanmax(raw, axis=0)
            elif stat == "jja":
                n_days = raw.shape[0]
                jja_idx = [k for k in range(n_days) if 152 <= (k % 365) <= 243]
                raw_jja = raw[jja_idx] if jja_idx else raw
                yr_mean = np.nanmean(raw_jja, axis=0)
            else:
                yr_mean = np.nanmean(raw, axis=0)

            if accum is None:
                lats_o = lats[li]
                lons_o = unique_lons[lj]
                # Máscara shapefile: se construye una sola vez
                mask   = _basin_mask_spatial(lats_o, lons_o)
                accum  = yr_mean.copy()
            else:
                accum += yr_mean
            count += 1
            ds.close()
        except Exception as ex:
            log.warning(f"  Error en {f.name}: {ex}")

    if count == 0 or accum is None:
        return None, None, None

    result = accum / count

    # NOTA: 3 píxeles del Bajo Chancay (lon~-77.3°, lat~-11.6/-11.7°) son NaN porque
    # ERA5-Land los clasifica como océano (land-sea mask a 0.1°). Para cálculos de media
    # de cuenca nanmean los omite automáticamente sin impacto significativo.
    # Para mapas de presentación formal, rellenar con:
    #   from scipy.ndimage import distance_transform_edt
    #   _, idx = distance_transform_edt(np.isnan(result), return_indices=True)
    #   result[np.isnan(result) & mask] = result[idx[0], idx[1]][np.isnan(result) & mask]

    # Enmascarar píxeles fuera de la cuenca
    result[~mask] = np.nan
    return lats_o, lons_o, result


def _load_shapes():
    """Carga cuenca y sub-cuencas en WGS84. Retorna (cuenca_gdf, subcuencas_gdf)."""
    cuenca     = gpd.read_file(SHP_CUENCA).to_crs("EPSG:4326")     if SHP_CUENCA.exists()     else None
    subcuencas = gpd.read_file(SHP_SUBCUENCAS).to_crs("EPSG:4326") if SHP_SUBCUENCAS.exists() else None
    return cuenca, subcuencas


def _draw_shapes(ax, cuenca, subcuencas, lw_cuenca=1.6, lw_sub=0.6):
    """Dibuja los polígonos de cuenca y sub-cuencas sobre un eje matplotlib."""
    if subcuencas is not None:
        for geom in subcuencas.geometry:
            if geom is None:
                continue
            # puede ser Polygon o MultiPolygon
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for g in geoms:
                xs, ys = g.exterior.xy
                ax.plot(xs, ys, color="white", lw=lw_sub, alpha=0.55, zorder=4)

    if cuenca is not None:
        for geom in cuenca.geometry:
            if geom is None:
                continue
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for g in geoms:
                xs, ys = g.exterior.xy
                ax.plot(xs, ys, color="black", lw=lw_cuenca, alpha=0.92, zorder=5)


def plot_ERA03_mapas(data: dict):
    """Mapas espaciales medios para 6 variables clave con límite de cuenca superpuesto."""
    # (grupo, variable, título, colormap, flip_sign, stat)
    MAP_VARS = [
        ("accum1", "tp",    "Precip. media (tp)\nmm/día",        "Blues",    False, "mean"),
        ("accum1", "pev",   "ETP media (pev)\nmm/día",           "Greens",   False, "mean"),
        ("inst1",  "t2m",   "Temperatura 2m\n°C",                "RdYlBu_r", False, "mean"),
        ("inst1",  "swvl1", "Humedad suelo L1\nm³/m³",           "YlOrBr_r", False, "mean"),
        ("accum2", "ssrd",  "Rad. solar ↓\nMJ/m²/día",          "YlOrRd",   False, "mean"),
        ("inst2",  "snowc", "Cubierta nieve\n% (máx. anual)",    "Blues",    False, "max"),
    ]

    cuenca, subcuencas = _load_shapes()
    log.info(f"Shapefile cuenca: {'OK' if cuenca is not None else 'no encontrado'}")
    log.info(f"Shapefile sub-cuencas: {'OK' if subcuencas is not None else 'no encontrado'}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#E8EAF0")
    period = _period_str(data)
    fig.suptitle(
        f"ERA5-Land — Mapas espaciales medios {period}\n"
        "Cuenca Chancay-Huaral (borde negro) · Sub-cuencas (líneas blancas) · resolución 0.1°",
        fontsize=12, fontweight="bold",
    )

    for idx, (grp, vn, title, cmap, flip, stat) in enumerate(MAP_VARS):
        ax = axes.flatten()[idx]
        ax.set_facecolor("#D0D8E8")
        lats, lons, grid = _read_spatial_mean(grp, vn, stat=stat)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)

        if grid is None:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue

        if flip:
            grid = -grid

        # Ordenar lat de mayor a menor para imshow (origen upper-left)
        if lats[0] < lats[-1]:
            grid = grid[::-1, :]
            lats = lats[::-1]

        # Fondo: imagen rasterizada del bbox completo (incluye zona fuera de cuenca)
        im = ax.imshow(
            grid, cmap=cmap, aspect="auto",
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            origin="upper", alpha=0.88, zorder=2,
        )

        # Colorbar
        cb = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cb.ax.tick_params(labelsize=7)

        # Contornos del campo sobre el bbox
        try:
            lons_sorted = np.sort(lons)
            lats_for_contour = lats[::-1] if lats[0] > lats[-1] else lats
            grid_for_contour = grid[::-1] if lats[0] > lats[-1] else grid
            LON_G, LAT_G = np.meshgrid(lons_sorted, lats_for_contour)
            ax.contour(LON_G, LAT_G, grid_for_contour,
                       levels=5, colors="white", linewidths=0.5, alpha=0.5, zorder=3)
        except Exception:
            pass

        # ── Shapefiles superpuestos ────────────────────────────────────────────
        _draw_shapes(ax, cuenca, subcuencas)

        # Límites del mapa ajustados al bbox con pequeño margen
        margin = 0.05
        ax.set_xlim(lons.min() - margin, lons.max() + margin)
        ax.set_ylim(lats.min() - margin, lats.max() + margin)

        # Ticks y etiquetas
        ax.set_xticks(np.round(np.linspace(lons.min(), lons.max(), 4), 1))
        ax.set_yticks(np.round(np.linspace(lats.min(), lats.max(), 4), 1))
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Longitud", fontsize=8)
        ax.set_ylabel("Latitud", fontsize=8)
        ax.grid(True, color="#AAAAAA", lw=0.3, alpha=0.4, zorder=1)

        # Puntos de la grilla ERA5 (para visualizar la resolución)
        LON_PTS, LAT_PTS = np.meshgrid(lons, lats)
        ax.scatter(LON_PTS.flatten(), LAT_PTS.flatten(),
                   s=4, c="white", alpha=0.35, zorder=6, linewidths=0)

    # Leyenda global
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.6, label="Límite cuenca"),
        Line2D([0], [0], color="white", lw=0.8, alpha=0.7, label="Sub-cuencas"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markersize=4, alpha=0.5, label="Grilla ERA5 (0.1°)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = OUT_DIR / "ERA03_mapas_espaciales.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"ERA03 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Figura 4: Correlaciones entre variables + scatter precipitación
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ERA04_correlaciones(data: dict):
    """Matriz de correlación + scatter tp vs variables clave."""
    # Construir DataFrame combinado (mensual — evita ruido diario)
    dfs = {}
    for grp, df in data.items():
        if df.empty:
            continue
        for col in df.columns:
            dfs[col] = df[col]

    combined = pd.DataFrame(dfs)
    # Añadir velocidad viento si no existe
    if "u10" in combined.columns and "v10" in combined.columns:
        combined["wspd"] = np.sqrt(combined["u10"]**2 + combined["v10"]**2)

    # Resample mensual para reducir ruido
    monthly = combined.resample("ME").mean()

    # Seleccionar variables para la matriz
    CORR_VARS = [v for v in ["tp","pev","ro","t2m","d2m","swvl1","swvl2",
                               "ssrd","strd","ssr","snowc","stl1","wspd","sp"]
                 if v in monthly.columns]

    corr = monthly[CORR_VARS].corr()

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── Panel A: Matriz de correlación ────────────────────────────────────────
    ax_corr = fig.add_subplot(gs[:, :2])
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.copy()
    corr_masked[mask_upper] = np.nan

    im = ax_corr.imshow(corr_masked, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax_corr.set_xticks(range(len(CORR_VARS)))
    ax_corr.set_xticklabels(CORR_VARS, rotation=45, ha="right", fontsize=9)
    ax_corr.set_yticks(range(len(CORR_VARS)))
    ax_corr.set_yticklabels(CORR_VARS, fontsize=9)

    for i in range(len(CORR_VARS)):
        for j in range(len(CORR_VARS)):
            if i >= j:
                val = corr_masked.iloc[i, j]
                if not np.isnan(val):
                    ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center",
                                 fontsize=7.5, color="white" if abs(val) > 0.6 else "black")

    plt.colorbar(im, ax=ax_corr, fraction=0.025, pad=0.02).ax.tick_params(labelsize=8)
    period = _period_str(data)
    ax_corr.set_title(f"Correlación mensual entre variables ERA5-Land ({period})",
                       fontsize=10, fontweight="bold", pad=8)

    # ── Panel B: tp vs t2m estacional ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    if "tp" in monthly.columns and "t2m" in monthly.columns:
        sc = ax_b.scatter(monthly["t2m"], monthly["tp"],
                          c=monthly.index.month, cmap="RdYlBu_r",
                          s=30, alpha=0.75, edgecolors="none")
        plt.colorbar(sc, ax=ax_b, label="Mes").ax.tick_params(labelsize=7)
        ax_b.set_xlabel("Temperatura 2m (°C)", fontsize=8)
        ax_b.set_ylabel("Precipitación (mm/día)", fontsize=8)
        ax_b.set_title("Precip. vs Temperatura\n(color = mes)", fontsize=9, fontweight="bold")
        r = monthly[["t2m","tp"]].dropna().corr().iloc[0,1]
        ax_b.text(0.05, 0.92, f"r={r:.3f}", transform=ax_b.transAxes, fontsize=9)

    # ── Panel C: tp vs swvl1 ──────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 2])
    if "tp" in monthly.columns and "swvl1" in monthly.columns:
        lag1 = monthly["tp"].shift(1)
        ax_c.scatter(lag1, monthly["swvl1"],
                     c=monthly.index.month, cmap="viridis",
                     s=30, alpha=0.75, edgecolors="none")
        ax_c.set_xlabel("Precip. mes anterior (mm/día)", fontsize=8)
        ax_c.set_ylabel("Humedad suelo L1 (m³/m³)", fontsize=8)
        ax_c.set_title("Humedad suelo vs Precip. (lag-1)\nrespuesta del suelo", fontsize=9, fontweight="bold")
        r = monthly[["swvl1"]].assign(tp_lag=lag1).dropna().corr().iloc[0,1]
        ax_c.text(0.05, 0.92, f"r={r:.3f}", transform=ax_c.transAxes, fontsize=9)

    fig.suptitle(
        f"ERA5-Land — Correlaciones y relaciones entre variables ({period}, resolución mensual)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "ERA04_correlaciones.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"ERA04 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=== Script 42: Exploración ERA5-Land ===")
    log.info(f"Leyendo archivos desde: {PARTS}")

    data = load_all_data()

    # Resumen rápido de lo cargado
    log.info("Datos cargados:")
    for grp, df in data.items():
        if not df.empty:
            log.info(f"  {grp}: {len(df)} días ({df.index.min().date()} - {df.index.max().date()})  cols={list(df.columns)}")

    log.info("Generando ERA01 — Series temporales ...")
    plot_ERA01_series(data)

    log.info("Generando ERA02 — Climatología mensual ...")
    plot_ERA02_climatologia(data)

    log.info("Generando ERA03 — Mapas espaciales ...")
    plot_ERA03_mapas(data)

    log.info("Generando ERA04 — Correlaciones ...")
    plot_ERA04_correlaciones(data)

    log.info("=== DONE ===")
    log.info(f"Figuras en: {OUT_DIR}/ERA01-ERA04.png")

    # Imprimir estadísticas básicas de las variables más importantes
    log.info("")
    period = _period_str(data)
    log.info(f"Estadísticas cuenca-mean ({period}):")
    KEY = [("accum1","tp"), ("accum1","pev"), ("inst1","t2m"),
           ("inst1","swvl1"), ("accum2","ssrd"), ("inst2","snowc")]
    for grp, vn in KEY:
        df = data.get(grp, pd.DataFrame())
        if not df.empty and vn in df.columns:
            s = df[vn].dropna()
            log.info(f"  {vn:8s}: mean={s.mean():.3f}  std={s.std():.3f}  "
                     f"min={s.min():.3f}  max={s.max():.3f}  "
                     f"[{VARS[grp][vn]['unit']}]")


if __name__ == "__main__":
    main()
