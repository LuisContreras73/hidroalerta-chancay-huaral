"""
Script 43 — GEE Data Overview Figures
Genera 8 figuras (GEE01-GEE08) mostrando el inventario y cobertura
de todos los datos GEE descargados en data/raw/gee/.

No requiere rasterio ni procesamiento espacial — solo lee
nombres de archivo y JSON sidecars para mostrar disponibilidad.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
GEE_DIR = ROOT / "data" / "raw" / "gee"
OUT_DIR = ROOT / "outputs" / "figures" / "gee"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "accent1":  "#58a6ff",
    "accent2":  "#3fb950",
    "accent3":  "#f78166",
    "accent4":  "#d2a8ff",
    "accent5":  "#ffa657",
    "text":     "#c9d1d9",
    "subtext":  "#8b949e",
    "grid":     "#21262d",
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["panel"],
    "axes.edgecolor":    STYLE["grid"],
    "text.color":        STYLE["text"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["subtext"],
    "ytick.color":       STYLE["subtext"],
    "grid.color":        STYLE["grid"],
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
})

# ── Fuentes GEE y su info ────────────────────────────────────────────────────
SOURCES = {
    "chirps":     {"label": "CHIRPS (lluvia)",          "color": STYLE["accent1"],  "start": 1981, "end": 2025},
    "snow":       {"label": "MOD10A1 (nieve/NDSI)",     "color": "#79c0ff",          "start": 2000, "end": 2025},
    "vegetation": {"label": "MOD13Q1 (NDVI/EVI)",       "color": STYLE["accent2"],   "start": 2000, "end": 2025},
    "lswi_veg":   {"label": "MOD09GA (LSWI)",           "color": "#56d364",          "start": 2000, "end": 2025},
    "thermal":    {"label": "MOD11A1 (LST día/noche)",  "color": STYLE["accent3"],   "start": 2000, "end": 2025},
    "et":         {"label": "MOD16A2 (ET/PET)",         "color": "#ff7b72",          "start": 2001, "end": 2025},
    "smap":       {"label": "SMAP (humedad suelo)",     "color": STYLE["accent4"],   "start": 2015, "end": 2025},
    "sentinel1":  {"label": "Sentinel-1 (SAR VV/VH)",  "color": "#bc8cff",          "start": 2015, "end": 2025},
    "sentinel2":  {"label": "Sentinel-2 (NDVI/NDWI)",  "color": STYLE["accent5"],   "start": 2017, "end": 2025},
    "land_cover": {"label": "Land Cover (ESA/MCD12Q1)","color": "#e3b341",          "start": 2001, "end": 2023},
    "landsat":    {"label": "Landsat 5/7/8 (NDVI/NDWI)","color": "#f0883e",         "start": 1984, "end": 2025},
    "jrc":        {"label": "JRC Water History",        "color": "#58a6ff",          "start": 1984, "end": 2021},
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _save(fig, name, dpi=150):
    out = OUT_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Guardado: {out.name}")


def _count_tifs(folder_name, pattern="*.tif"):
    folder = GEE_DIR / folder_name
    if not folder.exists():
        return {}
    counts = defaultdict(int)
    for f in folder.glob(pattern):
        # extraer año del nombre
        parts = f.stem.split("_")
        for p in parts:
            if p.isdigit() and len(p) == 4 and 1980 <= int(p) <= 2030:
                counts[int(p)] += 1
                break
    return dict(counts)


def _read_json_sidecars(folder_name, pattern="*.json"):
    folder = GEE_DIR / folder_name
    if not folder.exists():
        return []
    data = []
    for f in sorted(folder.glob(pattern)):
        try:
            d = json.loads(f.read_text())
            data.append(d)
        except Exception:
            pass
    return data


def _folder_mb(folder_name):
    folder = GEE_DIR / folder_name
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6

# ── GEE01 — Heatmap de cobertura por fuente × año ────────────────────────────
def plot_GEE01_coverage_heatmap():
    log.info("GEE01 — Heatmap cobertura GEE...")

    years = list(range(1981, 2026))

    # Construir matriz de disponibilidad (0=sin datos, 1=parcial, 2=completo)
    rows = []
    labels = []
    colors_row = []

    folder_map = {
        "chirps":     ("chirps",     "chirps"),
        "snow":       ("snow",       "snow"),
        "ndvi_evi":   ("vegetation", "mod13*"),
        "lswi":       ("vegetation", "mod09*"),
        "lst":        ("thermal",    "thermal"),
        "et_pet":     ("et",         "et"),
        "smap":       ("smap",       "smap"),
        "sentinel1":  ("sentinel1",  "sentinel1"),
        "sentinel2":  ("sentinel2",  "sentinel2"),
        "landcover":  ("land_cover", "land_cover"),
        "landsat":    ("water_land", "landsat*"),
        "jrc":        ("water_land", "jrc*"),
    }

    source_labels = {
        "chirps":    "CHIRPS (lluvia)",
        "snow":      "MOD10A1 (nieve)",
        "ndvi_evi":  "MOD13Q1 (NDVI/EVI)",
        "lswi":      "MOD09GA (LSWI)",
        "lst":       "MOD11A1 (LST)",
        "et_pet":    "MOD16A2 (ET/PET)",
        "smap":      "SMAP (sm)",
        "sentinel1": "Sentinel-1 (SAR)",
        "sentinel2": "Sentinel-2 (MSI)",
        "landcover": "Land Cover",
        "landsat":   "Landsat 5/7/8",
        "jrc":       "JRC Water",
    }

    source_colors = {
        "chirps":    STYLE["accent1"],
        "snow":      "#79c0ff",
        "ndvi_evi":  STYLE["accent2"],
        "lswi":      "#56d364",
        "lst":       STYLE["accent3"],
        "et_pet":    "#ff7b72",
        "smap":      STYLE["accent4"],
        "sentinel1": "#bc8cff",
        "sentinel2": STYLE["accent5"],
        "landcover": "#e3b341",
        "landsat":   "#f0883e",
        "jrc":       "#58a6ff",
    }

    for key, (folder, pat) in folder_map.items():
        start_yr = SOURCES.get(
            folder if folder != "water_land" else key.replace("_", ""),
            SOURCES.get(key, {"start": 1981})
        )["start"] if key not in ("jrc", "landsat") else (1984 if key == "jrc" else 1984)

        end_yr = SOURCES.get(key, {"end": 2025}).get("end", 2025)
        if key == "jrc": end_yr = 2021
        if key == "smap": start_yr = 2015
        if key == "sentinel1": start_yr = 2015
        if key == "sentinel2": start_yr = 2017
        if key == "landcover": start_yr = 2001; end_yr = 2023

        # contar archivos por año en la carpeta
        folder_path = GEE_DIR / folder
        counts_yr = defaultdict(int)
        if folder_path.exists():
            glob_pat = f"*{pat.replace('*','').strip()}*.tif" if "*" in pat else f"{pat}*.tif"
            for f in folder_path.glob("*.tif"):
                nm = f.name
                # check si coincide con el patrón simplificado
                pat_core = pat.replace("*", "").strip()
                if pat_core and pat_core not in nm:
                    continue
                for part in nm.split("_"):
                    if part.isdigit() and len(part) == 4 and 1980 <= int(part) <= 2030:
                        counts_yr[int(part)] += 1
                        break

        row = []
        for yr in years:
            if yr < start_yr or yr > end_yr:
                row.append(0)   # fuera de rango del sensor
            elif counts_yr.get(yr, 0) > 0:
                row.append(2)   # datos disponibles
            else:
                row.append(1)   # período esperado pero sin datos (raro)

        rows.append(row)
        labels.append(source_labels[key])
        colors_row.append(source_colors[key])

    data = np.array(rows)
    n_sources, n_years = data.shape

    fig, ax = plt.subplots(figsize=(18, 6), facecolor=STYLE["bg"])
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    # Dibujar bloques por celda
    for i in range(n_sources):
        for j, yr in enumerate(years):
            val = data[i, j]
            if val == 0:
                color = STYLE["grid"]
                alpha = 0.3
            elif val == 1:
                color = STYLE["accent3"]
                alpha = 0.7
            else:
                color = colors_row[i]
                alpha = 0.85
            rect = mpatches.FancyBboxPatch(
                (j + 0.05, i + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=alpha, edgecolor="none",
            )
            ax.add_patch(rect)

    ax.set_xlim(0, n_years)
    ax.set_ylim(0, n_sources)
    ax.set_yticks(np.arange(n_sources) + 0.5)
    ax.set_yticklabels(labels[::-1] if False else labels, fontsize=9)

    # Eje X: décadas + años clave
    tick_positions = [i for i, yr in enumerate(years) if yr % 5 == 0]
    tick_labels    = [str(years[i]) for i in tick_positions]
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_title("Cobertura Temporal — Datos GEE Descargados (1981-2025)",
                 fontsize=14, fontweight="bold", color=STYLE["text"], pad=12)

    # Leyenda
    legend_handles = [
        mpatches.Patch(color=STYLE["accent2"], label="Datos disponibles", alpha=0.85),
        mpatches.Patch(color=STYLE["grid"],    label="Fuera de rango del sensor", alpha=0.5),
        mpatches.Patch(color=STYLE["accent3"], label="Período esperado sin datos", alpha=0.7),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"])

    # Líneas verticales de hitos satelitales
    milestones = {2000: "MODIS\n2000", 2015: "S-1\n2015", 2017: "S-2\n2017"}
    for yr, lbl in milestones.items():
        x = years.index(yr)
        ax.axvline(x + 0.5, color=STYLE["subtext"], lw=0.8, ls="--", alpha=0.5)
        ax.text(x + 0.55, n_sources - 0.2, lbl, color=STYLE["subtext"],
                fontsize=7, va="top", ha="left")

    ax.grid(axis="x", which="minor", color=STYLE["grid"], lw=0.3, alpha=0.3)
    plt.tight_layout()
    _save(fig, "GEE01_coverage_heatmap.png", dpi=150)


# ── GEE02 — Tamaños por fuente GEE ───────────────────────────────────────────
def plot_GEE02_folder_sizes():
    log.info("GEE02 — Tamaños por carpeta GEE...")

    folders = {
        "chirps":     "CHIRPS\n(lluvia)",
        "snow":       "MOD10A1\n(nieve)",
        "vegetation": "MODIS Veg\n(NDVI/EVI/LSWI)",
        "thermal":    "MOD11A1\n(LST)",
        "et":         "MOD16A2\n(ET/PET)",
        "smap":       "SMAP\n(humedad)",
        "sentinel1":  "Sentinel-1\n(SAR)",
        "sentinel2":  "Sentinel-2\n(MSI)",
        "land_cover": "Land\nCover",
        "water_land": "Landsat +\nJRC Water",
    }

    sizes = {k: _folder_mb(k) for k in folders}
    labels = list(folders.values())
    values = [sizes[k] for k in folders]
    total_gb = sum(values) / 1024

    colors = [
        STYLE["accent1"], "#79c0ff", STYLE["accent2"], STYLE["accent3"],
        "#ff7b72", STYLE["accent4"], "#bc8cff", STYLE["accent5"],
        "#e3b341", "#f0883e"],

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=STYLE["bg"],
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Barplot
    bar_colors = [
        STYLE["accent1"], "#79c0ff", STYLE["accent2"], STYLE["accent3"],
        "#ff7b72", STYLE["accent4"], "#bc8cff", STYLE["accent5"],
        "#e3b341", "#f0883e"]
    bars = ax1.barh(range(len(labels)), values, color=bar_colors, height=0.65, alpha=0.85)

    for bar, val in zip(bars, values):
        if val > 20:
            ax1.text(val + 10, bar.get_y() + bar.get_height()/2,
                     f"{val/1024:.2f} GB" if val > 500 else f"{val:.0f} MB",
                     va="center", ha="left", color=STYLE["text"], fontsize=8)

    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Tamaño (MB)", color=STYLE["subtext"], fontsize=9)
    ax1.set_title(f"Tamaño por Fuente GEE — Total: {total_gb:.2f} GB",
                  fontsize=12, fontweight="bold", color=STYLE["text"])
    ax1.set_facecolor(STYLE["panel"])
    ax1.grid(axis="x", color=STYLE["grid"], lw=0.5)

    # Pie chart
    pie_vals  = [v for v in values if v > 5]
    pie_lbls  = [labels[i] for i, v in enumerate(values) if v > 5]
    pie_clrs  = [bar_colors[i] for i, v in enumerate(values) if v > 5]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=None, colors=pie_clrs,
        autopct="%1.0f%%", startangle=140,
        pctdistance=0.75, wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(STYLE["bg"])
    ax2.legend(wedges, [l.replace("\n", " ") for l in pie_lbls],
               loc="lower center", bbox_to_anchor=(0.5, -0.25),
               fontsize=7, ncol=2, facecolor=STYLE["panel"],
               edgecolor=STYLE["grid"], labelcolor=STYLE["text"])
    ax2.set_title("Distribución de almacenamiento", fontsize=10,
                  color=STYLE["text"], fontweight="bold")
    ax2.set_facecolor(STYLE["bg"])

    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    _save(fig, "GEE02_folder_sizes.png", dpi=150)


# ── GEE03 — Conteo de archivos por fuente ────────────────────────────────────
def plot_GEE03_file_counts():
    log.info("GEE03 — Conteo de archivos GEE...")

    data = {
        "CHIRPS\n(lluvia anual)":        {"tif": 45,  "json": 0,   "color": STYLE["accent1"],  "period": "1981-2025"},
        "MOD10A1\n(nieve bimestral)":    {"tif": 155, "json": 0,   "color": "#79c0ff",          "period": "2000-2025"},
        "MOD13Q1\n(NDVI/EVI semest.)":   {"tif": 52,  "json": 0,   "color": STYLE["accent2"],   "period": "2000-2025"},
        "MOD09GA\n(LSWI bimestral)":     {"tif": 156, "json": 0,   "color": "#56d364",          "period": "2000-2025"},
        "MOD11A1\n(LST semestral)":      {"tif": 52,  "json": 0,   "color": STYLE["accent3"],   "period": "2000-2025"},
        "MOD16A2\n(ET/PET anual)":       {"tif": 25,  "json": 0,   "color": "#ff7b72",          "period": "2001-2025"},
        "SMAP\n(sm mensual)":            {"tif": 130, "json": 0,   "color": STYLE["accent4"],   "period": "2015-2025"},
        "Sentinel-1\n(SAR anual)":       {"tif": 11,  "json": 11,  "color": "#bc8cff",          "period": "2015-2025"},
        "Sentinel-2\n(MSI anual)":       {"tif": 9,   "json": 9,   "color": STYLE["accent5"],   "period": "2017-2025"},
        "Land Cover\n(anual)":           {"tif": 24,  "json": 0,   "color": "#e3b341",          "period": "2001-2023"},
        "Landsat\n(NDVI/NDWI H1/H2)":   {"tif": 87,  "json": 87,  "color": "#f0883e",          "period": "1984-2025"},
        "JRC Water\n(quinquenal)":       {"tif": 8,   "json": 0,   "color": "#58a6ff",          "period": "1984-2021"},
    }

    labels = list(data.keys())
    tif_counts  = [data[k]["tif"]  for k in labels]
    json_counts = [data[k]["json"] for k in labels]
    colors      = [data[k]["color"] for k in labels]
    periods     = [data[k]["period"] for k in labels]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, tif_counts,  w, label="TIF (raster)", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, json_counts, w, label="JSON (sidecar)", color=colors, alpha=0.45,
                   edgecolor=colors, linewidth=1.2)

    for bar, val in zip(bars1, tif_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=8, color=STYLE["text"])

    for bar, val in zip(bars2, json_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=7.5,
                    color=STYLE["subtext"])

    # Período debajo de cada label
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n({periods[i]})" for i, lbl in enumerate(labels)],
        fontsize=7.5
    )
    ax.set_ylabel("Número de archivos", color=STYLE["subtext"], fontsize=10)
    ax.set_title(f"Archivos GEE Descargados por Fuente — Total: 754 TIF + 107 JSON",
                 fontsize=13, fontweight="bold", color=STYLE["text"], pad=10)

    total_tif  = sum(tif_counts)
    total_json = sum(json_counts)
    ax.legend(
        labels=["TIF (raster)", "JSON (sidecar — meses válidos)"],
        fontsize=9, facecolor=STYLE["panel"],
        edgecolor=STYLE["grid"], labelcolor=STYLE["text"],
        loc="upper right"
    )
    ax.grid(axis="y", color=STYLE["grid"], lw=0.5)
    ax.set_ylim(0, max(tif_counts) * 1.15)

    plt.tight_layout()
    _save(fig, "GEE03_file_counts.png", dpi=150)


# ── GEE04 — Landsat: cobertura de meses válidos por año ──────────────────────
def plot_GEE04_landsat_coverage():
    log.info("GEE04 — Cobertura Landsat por año/semestre...")

    sidecars = _read_json_sidecars("water_land", "landsat_ndvi_ndwi_*.json")
    if not sidecars:
        log.warning("No se encontraron JSON sidecars de Landsat — saltando GEE04")
        return

    # Construir tabla: año × semestre → n_meses_validos, sensor
    records = []
    for d in sidecars:
        yr   = d.get("yr", None)
        half = d.get("half", None)
        meses = len(d.get("valid_months", []))
        sensor = d.get("sensor", "?")
        if yr and half:
            records.append({"yr": yr, "half": half, "n_meses": meses, "sensor": sensor})

    df = pd.DataFrame(records).sort_values(["yr", "half"])

    years   = sorted(df.yr.unique())
    sensor_color = {"LT05": "#56d364", "LE07": "#ffa657", "LC08": "#79c0ff", "?": STYLE["subtext"]}

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), facecolor=STYLE["bg"],
                              gridspec_kw={"hspace": 0.4})

    for hi, (half_name, half_idx) in enumerate([("H1 — Enero a Junio", 1), ("H2 — Julio a Diciembre", 2)]):
        ax = axes[hi]
        ax.set_facecolor(STYLE["panel"])

        sub = df[df.half == half_idx].set_index("yr")
        x_vals, y_vals, colors_bar, sensors_bar = [], [], [], []

        for yr in range(1984, 2026):
            n = sub.loc[yr, "n_meses"] if yr in sub.index else 0
            sensor = sub.loc[yr, "sensor"] if yr in sub.index else "?"
            x_vals.append(yr)
            y_vals.append(n)
            colors_bar.append(sensor_color.get(sensor, STYLE["subtext"]))
            sensors_bar.append(sensor)

        bars = ax.bar(x_vals, y_vals, color=colors_bar, alpha=0.85, width=0.8, edgecolor="none")
        ax.axhline(6, color=STYLE["subtext"], lw=0.8, ls="--", alpha=0.5, label="Máximo 6 meses")
        ax.set_xlim(1983.5, 2025.5)
        ax.set_ylim(0, 7.5)
        ax.set_ylabel("Meses con imágenes", color=STYLE["subtext"], fontsize=9)
        ax.set_title(f"Landsat {half_name}", fontsize=11, color=STYLE["text"], fontweight="bold")
        ax.grid(axis="y", color=STYLE["grid"], lw=0.5)

        # Eje X limpio
        tick_yrs = [yr for yr in x_vals if yr % 5 == 0]
        ax.set_xticks(tick_yrs)
        ax.set_xticklabels(tick_yrs, fontsize=8)

        # Línea SLC-off LE07
        ax.axvline(2003.5, color=STYLE["accent3"], lw=0.8, ls=":", alpha=0.7)
        ax.text(2003.7, 7.0, "LE07 SLC-off\n(2003)", color=STYLE["accent3"],
                fontsize=7, va="top")

    # Leyenda de sensores
    legend_h = [mpatches.Patch(color=c, label=s) for s, c in sensor_color.items() if s != "?"]
    axes[0].legend(handles=legend_h, loc="upper left", fontsize=8,
                   facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"])

    fig.suptitle("Cobertura Landsat 5/7/8 — Meses válidos por semestre (1984-2025)",
                 fontsize=13, fontweight="bold", color=STYLE["text"], y=1.01)
    _save(fig, "GEE04_landsat_coverage.png", dpi=150)


# ── GEE05 — Sentinel-1: meses válidos por año ────────────────────────────────
def plot_GEE05_sentinel1_coverage():
    log.info("GEE05 — Cobertura Sentinel-1 por año...")

    sidecars = _read_json_sidecars("sentinel1", "s1_vv_vh_ratio_*.json")
    if not sidecars:
        log.warning("No JSON sidecars S1 — saltando GEE05")
        return

    records = [(d.get("yr"), len(d.get("valid_months", []))) for d in sidecars if d.get("yr")]
    if not records:
        return
    years_s1, n_meses = zip(*sorted(records))

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    bars = ax.bar(years_s1, n_meses, color=STYLE["accent4"], alpha=0.85, width=0.7)
    ax.axhline(12, color=STYLE["subtext"], lw=0.8, ls="--", alpha=0.5, label="12 meses completos")

    for bar, val in zip(bars, n_meses):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                str(val), ha="center", va="bottom", fontsize=9, color=STYLE["text"])

    ax.set_xlim(2014.5, 2025.5)
    ax.set_ylim(0, 14)
    ax.set_xticks(list(years_s1))
    ax.set_xticklabels(list(years_s1), fontsize=9)
    ax.set_ylabel("Meses con observaciones IW", color=STYLE["subtext"], fontsize=9)
    ax.set_title("Sentinel-1 GRD — Meses válidos por año (2015-2025, 500m)",
                 fontsize=12, fontweight="bold", color=STYLE["text"])
    ax.legend(fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
              labelcolor=STYLE["text"])
    ax.grid(axis="y", color=STYLE["grid"], lw=0.5)

    plt.tight_layout()
    _save(fig, "GEE05_sentinel1_coverage.png", dpi=150)


# ── GEE06 — Disponibilidad SMAP mensual ──────────────────────────────────────
def plot_GEE06_smap_coverage():
    log.info("GEE06 — Disponibilidad SMAP por año/mes...")

    smap_dir = GEE_DIR / "smap"
    if not smap_dir.exists():
        log.warning("Carpeta smap no existe — saltando")
        return

    # Nombre: smap_soil_{yr}_{mm}_3h.tif
    records = []
    for f in smap_dir.glob("smap_soil_*.tif"):
        parts = f.stem.split("_")
        if len(parts) >= 4:
            try:
                yr = int(parts[2])
                mm = int(parts[3])
                records.append((yr, mm))
            except ValueError:
                pass

    if not records:
        return

    years = sorted(set(r[0] for r in records))
    months = list(range(1, 13))
    grid = np.zeros((len(years), 12), dtype=int)

    for yr, mm in records:
        if yr in years and 1 <= mm <= 12:
            grid[years.index(yr), mm - 1] = 1

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    cmap = ListedColormap([STYLE["grid"], STYLE["accent4"]])
    im = ax.imshow(grid.T, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="upper")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=9, rotation=45)
    month_labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_labels, fontsize=9)

    ax.set_title("SMAP SPL4SMGP — Disponibilidad por Año/Mes (2015-2025, 10 km)",
                 fontsize=12, fontweight="bold", color=STYLE["text"])

    legend_h = [
        mpatches.Patch(color=STYLE["accent4"], label="Disponible"),
        mpatches.Patch(color=STYLE["grid"],    label="Sin datos"),
    ]
    ax.legend(handles=legend_h, loc="upper right", fontsize=8,
              facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"])

    for i in range(len(years)):
        for j in range(12):
            if grid[i, j]:
                ax.text(i, j, "✓", ha="center", va="center",
                        fontsize=7, color=STYLE["bg"], fontweight="bold")

    plt.tight_layout()
    _save(fig, "GEE06_smap_coverage.png", dpi=150)


# ── GEE07 — Resumen visual del volumen total GEE ─────────────────────────────
def plot_GEE07_total_summary():
    log.info("GEE07 — Resumen total GEE...")

    hitos = {
        "HITO 1\nCHIRPS + MOD10A1": {
            "archivos": 200, "gb": 1.12, "color": STYLE["accent1"],
            "fuentes": ["CHIRPS 1981-2025\n(45 TIF)", "MOD10A1 Snow\n(155 TIF)"],
            "icon": "🌧❄️"
        },
        "HITO 2\nMODIS + SMAP + S1 + S2": {
            "archivos": 545, "gb": 2.95, "color": STYLE["accent2"],
            "fuentes": ["MOD13Q1+MOD09GA (208)", "MOD11A1 (52)", "MOD16A2 (25)",
                        "SMAP (130)", "S1 (22)", "S2 (18)"],
            "icon": "🛰️🌿"
        },
        "HITO 3\nLandsat + JRC + LC": {
            "archivos": 171, "gb": 1.60, "color": STYLE["accent5"],
            "fuentes": ["Landsat 5/7/8 (174)", "JRC Water (8)", "Land Cover (24)"],
            "icon": "🏔️💧"
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=STYLE["bg"])
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle(f"GEE Pipeline — Inventario Completo: ~754 TIF + 107 JSON = ~6.0 GB",
                 fontsize=14, fontweight="bold", color=STYLE["text"], y=1.02)

    for ax, (hito, info) in zip(axes, hitos.items()):
        ax.set_facecolor(STYLE["panel"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Caja del hito
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02",
            facecolor=STYLE["bg"],
            edgecolor=info["color"], linewidth=2.5
        )
        ax.add_patch(rect)

        # Título hito
        ax.text(0.5, 0.88, hito, ha="center", va="center",
                fontsize=10, fontweight="bold", color=info["color"],
                transform=ax.transAxes)

        # Números grandes
        ax.text(0.5, 0.70, f"{info['archivos']}", ha="center", va="center",
                fontsize=32, fontweight="bold", color=STYLE["text"],
                transform=ax.transAxes)
        ax.text(0.5, 0.59, "archivos", ha="center", va="center",
                fontsize=9, color=STYLE["subtext"], transform=ax.transAxes)

        ax.text(0.5, 0.48, f"{info['gb']:.2f} GB", ha="center", va="center",
                fontsize=18, fontweight="bold", color=info["color"],
                transform=ax.transAxes)

        # Detalle de fuentes
        y_start = 0.34
        for fuente in info["fuentes"]:
            ax.text(0.5, y_start, f"• {fuente}", ha="center", va="top",
                    fontsize=7.5, color=STYLE["subtext"], transform=ax.transAxes)
            y_start -= 0.08

    plt.tight_layout()
    _save(fig, "GEE07_total_summary.png", dpi=150)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Script 43 — GEE Data Overview ===")
    log.info(f"Output: {OUT_DIR}")

    plot_GEE01_coverage_heatmap()
    plot_GEE02_folder_sizes()
    plot_GEE03_file_counts()
    plot_GEE04_landsat_coverage()
    plot_GEE05_sentinel1_coverage()
    plot_GEE06_smap_coverage()
    plot_GEE07_total_summary()

    log.info("=== DONE — Figuras en outputs/figures/gee/ ===")
