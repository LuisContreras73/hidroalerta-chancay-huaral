#!/usr/bin/env python3
"""
Script 41: Inventario y coherencia de datos — HidroAlerta Chancay-Huaral

Genera 8 figuras PNG + resumen Markdown para presentación:

  DI01_timeline.png           — Línea de tiempo de todas las fuentes (incl. satélites)
  DI02_d6_coherence.png       — Coherencia del dataset D6 (NaN%, rangos, fuentes)
  DI03_era5_status.png        — Estado de descarga ERA5-Land por año y grupo
  DI04_pipeline.png           — Catálogo de archivos bronze y flujo de scripts
  DI05_variable_catalogue.png — Inventario completo de variables por fuente
  DI06_satellite_timeline.png — Heatmap de disponibilidad de sensores satélite 1981-2025
  DI07_two_level_arch.png     — Diagrama arquitectura dos niveles (TFT + LightGBM)
  DI08_explainability.png     — Mapa "quién explica qué" (TFT VSN/Atención + SHAP)
  S8_data_inventory.md        — Resumen en Markdown para presentación/informe

Ejecutar desde la raíz del proyecto:
    python scripts/41_data_inventory.py

No requiere argumentos. Lee directamente los archivos de datos para obtener
rangos reales (D6, SNIRH), con valores conocidos hardcodeados para PISCO/ERA5.
"""

import re
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("data_inventory")

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "reports/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D6_PATH    = ROOT / "data/model_ready/D6_multientity.csv"
ERA5_PARTS = ROOT / "data/raw/era5/daily/parts"
SILVER_Q   = ROOT / "data/silver/snirh"

# ── Paleta de colores por categoría ──────────────────────────────────────────
C = {
    "model":  "#5C4B9F",   # morado — D6 / modelo
    "pr":     "#2196F3",   # azul   — precipitación
    "pr_gap": "#EF5350",   # rojo   — gap precipitación
    "temp":   "#FF9800",   # naranja — temperatura
    "pet":    "#4CAF50",   # verde  — ETP
    "pet_hs": "#A5D6A7",   # verde claro — ETP Hargreaves calibrado
    "q":      "#00BCD4",   # cyan   — caudal observado
    "q_miss": "#B0BEC5",   # gris claro — período sin datos de caudal
    "era5":   "#607D8B",   # gris azulado — ERA5 disponible
    "era5m":  "#FFCDD2",   # rosa   — ERA5 faltante
    "enso":   "#8BC34A",   # lima   — índices climáticos
    "geo":    "#795548",   # café   — estático
    "train":  "#1565C0",
    "val":    "#F9A825",
    "test":   "#C62828",
    # Satélites
    "landsat": "#8D6E63",  # marrón — Landsat 5/7/8
    "modis":   "#26A69A",  # teal   — MODIS Terra/Aqua
    "s1":      "#5C6BC0",  # índigo — Sentinel-1 SAR
    "s2":      "#26C6DA",  # cyan   — Sentinel-2 MSI
    "smap":    "#EF9A9A",  # rosa   — SMAP
}

TODAY = pd.Timestamp("2026-05-21")
Y_CUT = 2025   # corte del análisis para mostrar en figuras
Y0    = 1950   # año mínimo para la línea de tiempo


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Leer datos reales
# ═══════════════════════════════════════════════════════════════════════════════

def _read_d6_stats() -> dict:
    """Estadísticas reales de D6_multientity.csv."""
    if not D6_PATH.exists():
        log.warning("D6 no encontrado — usando valores conocidos")
        return {
            "shape": (131490, None),
            "period": ("1981-01-01", "2020-12-31"),
            "nan_pct": {"pr_mm": 2.51, "tmax_c": 0.0, "tmin_c": 0.0,
                        "pet_mm": 0.0, "q_mm": 99.2, "oni_index": 0.0},
            "n_entities": 9,
        }

    log.info("Leyendo D6_multientity.csv ...")
    df = pd.read_csv(D6_PATH, parse_dates=["date"], low_memory=False)

    stats = {
        "shape": df.shape,
        "period": (str(df["date"].min().date()), str(df["date"].max().date())),
        "n_entities": df["entity_id"].nunique() if "entity_id" in df.columns else 9,
        "nan_pct": {},
    }
    numeric_vars = ["pr_mm", "tmax_c", "tmin_c", "pet_mm", "q_mm", "oni_index"]
    for v in numeric_vars:
        if v in df.columns:
            stats["nan_pct"][v] = round(df[v].isna().mean() * 100, 2)
        else:
            stats["nan_pct"][v] = None
    return stats


def _read_snirh_periods() -> list[dict]:
    """Rangos reales de los CSV de caudal SNIRH."""
    stations = []
    mapping = {
        "S1_santo_domingo_47e214d2__caudal_diario.csv": {
            "label": "Q Santo Domingo (47E214D2, auto)\n614 m s.n.m.",
            "color": C["q"],
        },
        "S1_puente_callantama_47e22148__caudal_diario.csv": {
            "label": "Q Callantama (47E22148)\n780 m s.n.m.",
            "color": "#00ACC1",
        },
        "S1_vichaycocha_47e257d8__caudal_diario.csv": {
            "label": "Q Vichaycocha (47E257D8)\n3880 m s.n.m.",
            "color": "#006064",
        },
    }
    for fname, meta in mapping.items():
        fpath = SILVER_Q / fname
        if fpath.exists():
            try:
                tmp = pd.read_csv(fpath, parse_dates=[0], index_col=0)
                if not tmp.empty:
                    start = tmp.index.min()
                    end   = tmp.index.max()
                    stations.append({**meta, "start": start, "end": end})
                    continue
            except Exception:
                pass
        # fallback conocido
        stations.append({**meta, "start": pd.Timestamp("2020-09-01"), "end": TODAY})
    return stations


def _read_era5_years() -> dict[str, list[int]]:
    """Años disponibles por grupo de archivos ERA5 (sin _tmp_ prefijos)."""
    groups = {"accum1": [], "accum2": [], "inst1": [], "inst2": []}
    pattern = re.compile(r"era5_(accum1|accum2|inst1|inst2)_(\d{4})\.nc")
    if ERA5_PARTS.exists():
        for f in ERA5_PARTS.iterdir():
            m = pattern.match(f.name)
            if m:
                grp, yr = m.group(1), int(m.group(2))
                groups[grp].append(yr)
    for g in groups:
        groups[g] = sorted(set(groups[g]))
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DI01 — Línea de tiempo de fuentes de datos
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(ax, y, x0, x1, color, alpha=1.0, lw=0, hatch=None, zorder=3):
    ax.barh(
        y, x1 - x0, left=x0, height=0.55,
        color=color, alpha=alpha, linewidth=lw,
        hatch=hatch, edgecolor="white" if hatch else "none",
        zorder=zorder,
    )


def _label(ax, y, text, x0, x1, fontsize=7.5, color="black"):
    xmid = (x0 + x1) / 2
    ax.text(xmid, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold", zorder=5)


def _era5_bar(ax, y, years_available):
    # Mostrar 1981-2025: verde=disponible, rojo=faltante (incluye gap 1996-2025)
    all_years = list(range(1981, Y_CUT + 1))
    for yr in all_years:
        x0 = yr
        x1 = yr + 1
        if yr in years_available:
            _bar(ax, y, x0, x1, C["era5"], alpha=0.9)
        else:
            _bar(ax, y, x0, x1, C["era5m"], alpha=0.9)


def plot_DI01_timeline(d6_stats, snirh_periods, era5_groups):
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Filas (de arriba a abajo, y positivo = arriba en imshow; usamos y inverso)
    rows = [
        # (y, label_izq, barras_fn)
        # ── Sección: Modelo ────────────────────────────────────────────────
        (21.0, "Sección: MODELO", None),
        (20.0, "D6 — Train (1981-2008)", "train"),
        (19.0, "D6 — Val   (2009-2014)", "val"),
        (18.0, "D6 — Test  (2015-2020)", "test"),
        # ── Sección: Precipitación ────────────────────────────────────────
        (16.5, "Sección: PRECIPITACIÓN", None),
        (15.5, "PISCOp v2.1 update\n(0.05°, ~5 km)", "piscop"),
        (14.0, "PISCOp v2.1 update — GAP 2020\n(pr_mm = NaN en D6)", "piscop_gap"),
        (12.5, "CHIRPS 2020 (pendiente)\nScript 05 — requiere internet", "chirps"),
        (11.0, "SENAMHI — estaciones cuenca\n(6 pluviómetros, 1981-2016)", "senamhi_pr"),
        # ── Sección: Temperatura ──────────────────────────────────────────
        (9.5,  "Sección: TEMPERATURA", None),
        (8.5,  "PISCOt v1.2 Tmax\n(0.01°, ~1 km)", "tmax"),
        (7.0,  "PISCOt v1.2 Tmin\n(0.01°, ~1 km)", "tmin"),
        # ── Sección: ETP ──────────────────────────────────────────────────
        (5.5,  "Sección: EVAPOTRANSPIRACIÓN", None),
        (4.5,  "PET Penman-Monteith\n(ETP.nc, 1981-2016)", "pet_pm"),
        (3.0,  "PET Hargreaves-Samani cal.\n(2017-2020, calibrado con PM)", "pet_hs"),
        # ── Sección: Caudal ───────────────────────────────────────────────
        (1.5,  "Sección: CAUDAL OBSERVADO", None),
    ]
    # Añadir filas SNIRH dinámicamente
    q_y_positions = []
    y_cur = 0.5
    for s in snirh_periods:
        rows.append((y_cur, s["label"], {"type": "snirh", "data": s}))
        q_y_positions.append(y_cur)
        y_cur -= 1.3

    y_era5_header = y_cur - 0.2
    rows.append((y_era5_header, "Sección: ERA5-LAND", None))
    y_cur -= 1.3
    era5_labels = {
        "accum1": "ERA5 Acumuladas #1\n(tp, ssrd, ...)",
        "accum2": "ERA5 Acumuladas #2\n(e, ro, ...)",
        "inst1":  "ERA5 Instantáneas #1\n(t2m, d2m, u10, v10)",
        "inst2":  "ERA5 Instantáneas #2\n(sp, strd, ...)",
    }
    era5_y_map = {}
    for grp, lbl in era5_labels.items():
        rows.append((y_cur, lbl, {"type": "era5", "group": grp}))
        era5_y_map[grp] = y_cur
        y_cur -= 1.3

    y_enso = y_cur - 0.2
    rows.append((y_enso, "Sección: ÍNDICES CLIMÁTICOS", None))
    y_cur -= 1.2
    y_oni = y_cur
    rows.append((y_oni, "ONI / ENSO (NOAA)\n1950-2026", "oni"))
    y_cur -= 1.3
    y_geo = y_cur
    rows.append((y_geo, "Geología GEOCATMIN\n(estático, 1:100k)", "geo"))
    y_cur -= 1.5

    # ── Sección: Satélites ────────────────────────────────────────────────
    y_sat_header = y_cur
    rows.append((y_sat_header, "Sección: SATÉLITES (GEE)", None))
    y_cur -= 1.2
    SAT_ROWS = [
        (y_cur,       "Landsat 5/7/8\n(30m, mensual, GEE3/GEE7)", "landsat", 1984, 2025),
        (y_cur - 1.3, "MODIS Terra/Aqua\n(250-500m, 8-16d, GEE1/2a/2b/2c/6)", "modis", 2000.14, 2025),
        (y_cur - 2.6, "SMAP\n(10km, diario, GEE8)", "smap", 2015.25, 2025),
        (y_cur - 3.9, "Sentinel-1 SAR\n(10m, 12d, GEE10)", "s1", 2014.25, 2025),
        (y_cur - 5.2, "Sentinel-2 MSI\n(10m, 5d, GEE9)", "s2", 2017.24, 2025),
    ]
    for sy, slabel, skind, sstart, send in SAT_ROWS:
        rows.append((sy, slabel, {"type": "sat", "color": C[skind], "start": sstart, "end": send, "key": skind}))
    y_cur -= 6.5

    y_min = y_cur - 0.5
    y_max = 22.0

    # ── Dibujar barras ─────────────────────────────────────────────────────
    for row in rows:
        y, label, kind = row[0], row[1], row[2]

        if kind is None:
            # Encabezado de sección
            ax.text(Y0, y, label.replace("Sección: ", ""),
                    va="center", ha="left", fontsize=9, color="gray",
                    fontweight="bold", style="italic")
            ax.axhline(y - 0.3, color="#DDDDDD", lw=0.8, xmin=0.04)
            continue

        # Etiqueta izquierda
        ax.text(Y0 - 1, y, label, va="center", ha="right",
                fontsize=8, color="#333333", wrap=True)

        # ── Barras según tipo ──────────────────────────────────────────────
        if kind == "train":
            _bar(ax, y, 1981, 2009, C["train"], alpha=0.85)
            _label(ax, y, "Train  1981-2008  (28 años)", 1981, 2009)
        elif kind == "val":
            _bar(ax, y, 2009, 2015, C["val"], alpha=0.85)
            _label(ax, y, "Val  2009-2014  (6 años)", 2009, 2015, color="#333")
        elif kind == "test":
            _bar(ax, y, 2015, 2021, C["test"], alpha=0.85)
            _label(ax, y, "Test  2015-2020  (6 años)", 2015, 2021)

        elif kind == "piscop":
            _bar(ax, y, 1981, 2020, C["pr"])
            _label(ax, y, "PISCOp  1981-2019  (14 244 días)", 1981, 2020)

        elif kind == "piscop_gap":
            _bar(ax, y, 1981, 2020, C["pr"], alpha=0.2)
            _bar(ax, y, 2020, 2021, C["pr_gap"], alpha=0.95)
            _label(ax, y, "pr_mm = NaN en todo 2020 (3 294 filas en D6)", 1981, 2021,
                   color="#333")

        elif kind == "chirps":
            _bar(ax, y, 2020, 2021, "#BDBDBD", hatch="///", alpha=0.7)
            ax.text(2020.5, y, "Pendiente\nScript 05", ha="center", va="center",
                    fontsize=7, color="#555")

        elif kind == "senamhi_pr":
            _bar(ax, y, 1981, 2017, C["pr"], alpha=0.45)
            _label(ax, y, "SENAMHI  1981-2016  (6 estaciones)", 1981, 2017, color="#333")

        elif kind == "tmax":
            _bar(ax, y, 1981, 2021, C["temp"])
            _label(ax, y, "PISCOt Tmax  1981-2020  (14 610 días)", 1981, 2021)

        elif kind == "tmin":
            _bar(ax, y, 1981, 2021, C["temp"], alpha=0.7)
            _label(ax, y, "PISCOt Tmin  1981-2020  (14 610 días)", 1981, 2021)

        elif kind == "pet_pm":
            _bar(ax, y, 1981, 2017, C["pet"])
            _label(ax, y, "PM  1981-2016  (13 149 días)", 1981, 2017)

        elif kind == "pet_hs":
            _bar(ax, y, 2017, 2021, C["pet_hs"])
            _label(ax, y, "HS-cal  2017-2020", 2017, 2021, color="#333")
            _bar(ax, y, 1981, 2017, C["pet"], alpha=0.2)   # sombra

        elif isinstance(kind, dict) and kind.get("type") == "snirh":
            s     = kind["data"]
            start = s["start"].year + s["start"].dayofyear / 365
            end   = s["end"].year   + s["end"].dayofyear   / 365
            # Sin datos antes del inicio (fondo gris)
            _bar(ax, y, 1981, start, C["q_miss"], alpha=0.25)
            end_display = min(end, Y_CUT + 1)
            _bar(ax, y, start, end_display, s["color"])
            n_days = (s["end"] - s["start"]).days
            end_label = f"{min(s['end'], pd.Timestamp(f'{Y_CUT}-12-31')).strftime('%Y-%m')}"
            _label(ax, y,
                   f"{s['start'].strftime('%Y-%m')} -> {end_label}  ({n_days} dias)",
                   max(start, 1985), end_display, color="white")

        elif isinstance(kind, dict) and kind.get("type") == "era5":
            grp  = kind["group"]
            yrs  = era5_groups.get(grp, [])
            _era5_bar(ax, y, yrs)
            n = len(yrs)
            total_era5 = Y_CUT - 1981 + 1  # 1981-2025
            ax.text(Y_CUT + 0.3, y, f"{n} / {total_era5} años", va="center",
                    fontsize=7.5, color="#555")

        elif kind == "oni":
            _bar(ax, y, 1950, TODAY.year + 0.5, C["enso"])
            _label(ax, y, "ONI  1950 → 2026  (76 años)", 1950, TODAY.year + 0.5)

        elif kind == "geo":
            ax.scatter([1990], [y], s=200, marker="D", color=C["geo"], zorder=5)
            ax.text(1991, y, "Estático (GEOCATMIN 1:100k)", va="center", fontsize=8)

        elif isinstance(kind, dict) and kind.get("type") == "sat":
            sstart = kind["start"]
            send   = kind["end"]
            col    = kind["color"]
            # Período pre-lanzamiento (gris muy claro)
            _bar(ax, y, 1950, sstart, "#E0E0E0", alpha=0.4)
            # Período activo
            _bar(ax, y, sstart, send + 0.5, col, alpha=0.85)
            yr_str = f"{int(sstart)} → {int(send)}"
            _label(ax, y, yr_str, max(sstart, 1990), send + 0.5, color="white", fontsize=7)
            # Marcador de lanzamiento
            ax.axvline(sstart, color=col, lw=1.2, ls="-.", alpha=0.7, zorder=3)
            ax.scatter([sstart], [y + 0.45], s=60, marker="^", color=col, zorder=6)

    # ── Sombreado 4 eras satélite ─────────────────────────────────────────
    # Solo en la zona de satélites (y_min ... y_sat_header)
    era_bands = [
        (1981, 2000,    "#FAFAFA",  "pre-MODIS"),
        (2000, 2014.25, "#E8F5E9",  "MODIS"),
        (2014.25, 2017.24, "#E3F2FD", "MODIS+S1"),
        (2017.24, 2026, "#EDE7F6",  "Constelación completa"),
    ]
    for xe0, xe1, ebg, elbl in era_bands:
        ax.axvspan(xe0, xe1, ymin=0.0,
                   ymax=(y_sat_header - y_min) / (y_max - y_min),
                   color=ebg, alpha=0.55, zorder=0)

    # ── Ejes y decoración ─────────────────────────────────────────────────
    ax.set_xlim(Y0 - 5, Y_CUT + 2)
    ax.set_ylim(y_min - 0.5, y_max)
    ax.set_xlabel("Año", fontsize=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#AAAAAA")

    # Grilla vertical ligera
    for yr in range(1950, Y_CUT + 2, 5):
        ax.axvline(yr, color="#E0E0E0", lw=0.6, zorder=1)
    for yr in range(1950, Y_CUT + 2, 10):
        ax.axvline(yr, color="#CCCCCC", lw=1.0, zorder=1)

    # Línea de corte de análisis (2025)
    ax.axvline(Y_CUT, color="#E53935", lw=1.5, ls="--", zorder=4, alpha=0.7)
    ax.text(Y_CUT + 0.1, y_max - 0.5, "Corte\n2025", color="#E53935", fontsize=7.5)

    # Línea de split D6
    for yr, lbl in [(1981, ""), (2009, "↑ Val"), (2015, "↑ Test"), (2021, "")]:
        if yr in (2009, 2015):
            ax.axvline(yr, color="#888888", lw=0.8, ls=":", zorder=2)

    ax.set_title(
        "Inventario de Fuentes de Datos — HidroAlerta Chancay-Huaral\n"
        "Cuenca Chancay-Huaral (3 062 km²) · Período 1981-2025 · "
        "D6 (1981-2020) + D7 (1981-2025 incl. satélites GEE) · ERA5 pendiente 1996-2025",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Leyenda
    handles = [
        mpatches.Patch(color=C["train"],   label="D6 Train (1981-2008)"),
        mpatches.Patch(color=C["val"],     label="D6 Val (2009-2014)"),
        mpatches.Patch(color=C["test"],    label="D6 Test (2015-2020)"),
        mpatches.Patch(color=C["pr"],      label="Precipitación PISCO"),
        mpatches.Patch(color=C["pr_gap"],  label="Gap pr_mm (NaN 2020)"),
        mpatches.Patch(color=C["temp"],    label="Temperatura PISCOt"),
        mpatches.Patch(color=C["pet"],     label="ETP Penman-Monteith"),
        mpatches.Patch(color=C["pet_hs"], label="ETP Hargreaves-Samani"),
        mpatches.Patch(color=C["q"],       label="Caudal observado SNIRH"),
        mpatches.Patch(color=C["era5"],    label="ERA5-Land descargado"),
        mpatches.Patch(color=C["era5m"],   label="ERA5-Land FALTANTE"),
        mpatches.Patch(color=C["enso"],    label="ONI / ENSO"),
        mpatches.Patch(color=C["landsat"], label="Landsat 5/7/8 (GEE)"),
        mpatches.Patch(color=C["modis"],   label="MODIS Terra/Aqua (GEE)"),
        mpatches.Patch(color=C["smap"],    label="SMAP (GEE)"),
        mpatches.Patch(color=C["s1"],      label="Sentinel-1 SAR (GEE)"),
        mpatches.Patch(color=C["s2"],      label="Sentinel-2 MSI (GEE)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              ncol=4, framealpha=0.9, bbox_to_anchor=(0.0, -0.07))

    plt.tight_layout()
    out = OUT_DIR / "DI01_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI01 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DI02 — Coherencia del dataset D6
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI02_d6_coherence(d6_stats):
    nan_pct = d6_stats["nan_pct"]
    n_rows, _ = d6_stats["shape"]
    period_start, period_end = d6_stats["period"]
    n_ent = d6_stats["n_entities"]

    # Tabla de variables
    VAR_INFO = [
        ("pr_mm",      "Precipitación",        "PISCOp v2.1 update",          "1981-2019",  "mm/día",   "#2196F3"),
        ("tmax_c",     "Temperatura máx.",      "PISCOt v1.2 (QM correg.)",    "1981-2020",  "°C",       "#FF9800"),
        ("tmin_c",     "Temperatura mín.",      "PISCOt v1.2 (delta-T correg.)","1981-2020", "°C",       "#FFB74D"),
        ("pet_mm",     "Evapotranspiración",    "PM (1981-2016) + HS-cal (2017-2020)", "1981-2020", "mm/día", "#4CAF50"),
        ("q_mm",       "Caudal observado",      "SNIRH 47E214D2",              "2020-09→2020-12", "mm/día", "#00BCD4"),
        ("oni_index",  "Índice ONI / ENSO",     "NOAA ONI",                    "1950-2026",  "°C anomalía", "#8BC34A"),
    ]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Barras de NaN% ────────────────────────────────────────────
    ax_nan = fig.add_subplot(gs[0, 0])
    var_labels = [v[0] for v in VAR_INFO]
    nan_vals   = [nan_pct.get(v[0], 0) or 0 for v in VAR_INFO]
    colors_nan = [v[5] for v in VAR_INFO]

    bars = ax_nan.barh(var_labels, nan_vals, color=colors_nan, edgecolor="white", height=0.55)
    ax_nan.set_xlabel("% valores NaN en D6", fontsize=9)
    ax_nan.set_title("Completitud de variables en D6", fontsize=10, fontweight="bold")
    ax_nan.axvline(5, color="#BBBBBB", ls="--", lw=0.8)
    ax_nan.set_xlim(0, 105)

    for bar, val in zip(bars, nan_vals):
        xpos = val + 0.5
        label_str = f"{val:.1f}%"
        ax_nan.text(xpos, bar.get_y() + bar.get_height() / 2,
                    label_str, va="center", fontsize=8.5,
                    color="#C62828" if val > 5 else "#2E7D32")

    # Anotación q_mm
    ax_nan.annotate(
        "Solo 120 días válidos\n(2020-09 → 2020-12)",
        xy=(99.2, 4), xytext=(60, 3.5),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=7.5, color="#555",
    )

    # ── Panel B: Línea de tiempo por variable ─────────────────────────────
    ax_tl = fig.add_subplot(gs[0, 1])
    ranges = [
        ("pr_mm",    1981, 2020, 1981, 2019, "#2196F3"),
        ("tmax_c",   1981, 2020, 1981, 2020, "#FF9800"),
        ("tmin_c",   1981, 2020, 1981, 2020, "#FFB74D"),
        ("pet_mm",   1981, 2020, 1981, 2020, "#4CAF50"),
        ("q_mm",     1981, 2020, 2020, 2021, "#00BCD4"),
        ("oni_index",1981, 2020, 1981, 2021, "#8BC34A"),
    ]
    for i, (var, d6s, d6e, vs, ve, col) in enumerate(ranges):
        y = len(ranges) - 1 - i
        ax_tl.barh(y, d6e - d6s, left=d6s, height=0.4, color=col, alpha=0.2)
        ax_tl.barh(y, ve - vs,   left=vs,  height=0.4, color=col, alpha=0.9)
        ax_tl.text(d6s - 0.3, y, var, va="center", ha="right", fontsize=8)

    ax_tl.set_xlim(1979, 2022)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("Año", fontsize=9)
    ax_tl.set_title("Período válido por variable (vs. D6 1981-2020)", fontsize=10, fontweight="bold")
    ax_tl.axvline(2020, color="#888", ls=":", lw=0.8)
    ax_tl.axvline(1981, color="#888", ls=":", lw=0.8)
    for yr in range(1980, 2022, 5):
        ax_tl.axvline(yr, color="#EEEEEE", lw=0.6)

    # ── Panel C: Tabla de fuentes ──────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")

    col_labels = ["Variable", "Descripción", "Fuente", "Período válido",
                  "Unidad", "NaN%", "Estado"]
    table_data = []
    for v, desc, src, period, unit, _ in VAR_INFO:
        pct = nan_pct.get(v, 0) or 0
        if pct == 0:
            estado = "✓ Completo"
        elif pct < 5:
            estado = "⚠ Gap menor"
        elif pct > 90:
            estado = "✗ Solo 2020-Q4"
        else:
            estado = "✗ Gap 2020"
        table_data.append([v, desc, src, period, unit, f"{pct:.1f}%", estado])

    colors_table = []
    for row in table_data:
        pct_str = row[5]
        pct_val = float(pct_str.replace("%", ""))
        if pct_val == 0:
            row_c = ["#E8F5E9"] * 7
        elif pct_val < 5:
            row_c = ["#FFF9C4"] * 7
        else:
            row_c = ["#FFEBEE"] * 7
        colors_table.append(row_c)

    tab = ax_tab.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=colors_table,
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.6)

    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#CCCCCC")

    ax_tab.set_title(
        f"Tabla de fuentes — D6_multientity.csv  |  "
        f"{n_rows:,} filas × {n_ent} sub-cuencas  |  {period_start} → {period_end}",
        fontsize=10, fontweight="bold", pad=8,
    )

    fig.suptitle(
        "Coherencia del Dataset D6 — HidroAlerta Chancay-Huaral",
        fontsize=13, fontweight="bold", y=0.99,
    )

    out = OUT_DIR / "DI02_d6_coherence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI02 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DI03 — Estado de descarga ERA5-Land
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI03_era5_status(era5_groups):
    all_years = list(range(1981, Y_CUT + 1))
    grp_order = ["accum1", "accum2", "inst1", "inst2"]
    grp_labels = {
        "accum1": "Acumuladas #1\n(tp, ssrd, fal, slhf, sshf, str)",
        "accum2": "Acumuladas #2\n(e, ro, sro, snro, smlt, ...)",
        "inst1":  "Instantáneas #1\n(t2m, d2m, u10, v10, sp)",
        "inst2":  "Instantáneas #2\n(strd, stl1, swvl1, ..., lai)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "Estado de Descarga ERA5-Land por Ano y Grupo de Variables (1981-2025)\n"
        "Verde = disponible  |  Rojo = FALTANTE (requiere CDS API)",
        fontsize=12, fontweight="bold",
    )

    axes_flat = axes.flatten()
    for idx, grp in enumerate(grp_order):
        ax = axes_flat[idx]
        ax.set_facecolor("#F5F5F5")
        avail = set(era5_groups.get(grp, []))

        x_ok  = [yr for yr in all_years if yr in avail]
        x_mis = [yr for yr in all_years if yr not in avail]

        ax.bar(x_ok,  [1] * len(x_ok),  color="#43A047", width=0.8, label="Disponible")
        ax.bar(x_mis, [1] * len(x_mis), color="#E53935", width=0.8,
               label="FALTANTE", alpha=0.85)

        ax.set_yticks([])
        ax.set_title(grp_labels[grp], fontsize=9, fontweight="bold")
        ax.set_xticks(range(1981, Y_CUT + 1, 2))
        ax.set_xticklabels(range(1981, Y_CUT + 1, 2), rotation=45, fontsize=7)

        n_ok = len(x_ok)
        pct  = n_ok / len(all_years) * 100
        ax.text(0.02, 0.92,
                f"{n_ok} / {len(all_years)} anos  ({pct:.0f}%)",
                transform=ax.transAxes, fontsize=9,
                color="#2E7D32" if pct > 50 else "#C62828", fontweight="bold")

        # Anotar gap
        if x_mis:
            miss_min, miss_max = min(x_mis), max(x_mis)
            ax.annotate(
                f"Faltan\n{miss_min}-{miss_max}",
                xy=((miss_min + miss_max) / 2, 0.5),
                ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold",
            )

        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # Nota global
    fig.text(
        0.5, 0.01,
        "✅ ESTRATEGIA: GEE4_era5_raster.js exporta ERA5-Land 1981-2025 recortado a la cuenca (90 píxeles @ 0.1°) → ~160 MB  "
        "·  CDS API descarga región completa (~200 GB) — NO necesario si se usa GEE4",
        ha="center", fontsize=8, color="#2E7D32", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = OUT_DIR / "DI03_era5_status.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI03 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DI04 — Catálogo de archivos Bronze y flujo de scripts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI04_pipeline():
    BRONZE_CATALOG = [
        # (archivo, estado, script_gen, periodo, dims, uso_en_D6, nota)
        ("B2_pr_basin_grid.nc",          "DEPRECADO",  "02",  "1981-2016 (T=13149)", "lat×lon×T",    "No", "ETP.nc original"),
        ("B2_tmax_basin_grid.nc",         "DEPRECADO",  "03b", "1981-2016 (T=13149)", "lat×lon×T",    "No", "Sustituido por v12"),
        ("B2_tmin_basin_grid.nc",         "DEPRECADO",  "06",  "1981-2016 (T=13149)", "lat×lon×T",    "No", "Sustituido por v12"),
        ("B2_pet_basin_grid.nc",          "ACTIVO",     "02",  "1981-2016 (T=13149)", "lat×lon×T",    "Sí","ETP Penman-Monteith"),
        ("B2_pr_basin_grid_v21.nc",       "ACTIVO",     "04",  "1981-2019 (T=14244)", "z×lat×lon",    "Sí","PISCOp v2.1 update"),
        ("B2_tmax_basin_grid_v12.nc",     "ACTIVO",     "03b", "1981-2020 (T=14610)", "time×lat×lon", "Sí","PISCOt v1.2"),
        ("B2_tmin_basin_grid_v12.nc",     "ACTIVO",     "06",  "1981-2020 (T=14610)", "time×lat×lon", "Sí","PISCOt v1.2"),
        ("B2_pet_hargreaves_cal.nc",      "ACTIVO",     "16",  "1981-2020 (T=14610)", "time×lat×lon", "Sí","HS calibrado 2017-2020"),
        ("B2_tmax_corrected_spatial.nc",  "ACTIVO",     "19",  "1981-2020",           "time×lat×lon", "Sí","QM + delta-T corregido"),
        ("B2_tmin_corrected_spatial.nc",  "ACTIVO",     "19",  "1981-2020",           "time×lat×lon", "Sí","QM + delta-T corregido"),
        ("B4_dem_basin_90m.nc",           "ACTIVO",     "07",  "estático",            "lat×lon",      "Sí","SRTM 90m"),
        ("B4_dem_basin_30m.nc",           "ACTIVO",     "07",  "estático",            "lat×lon",      "No","SRTM 30m (referencia)"),
    ]

    col_labels = ["Archivo Bronze", "Estado", "Script\ngenerador",
                  "Período / T", "Dims", "En D6", "Descripción"]
    table_data = [[r[0], r[1], f"#{r[2]}", r[3], r[4], r[5], r[6]]
                  for r in BRONZE_CATALOG]

    def row_color(estado):
        if estado == "DEPRECADO":
            return ["#FFEBEE"] * 7
        return ["#E8F5E9"] * 7

    cell_colors = [row_color(r[1]) for r in BRONZE_CATALOG]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")

    tab = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.65)

    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold")
        if col == 1 and row > 0:
            val = table_data[row - 1][1]
            cell.set_facecolor("#E53935" if val == "DEPRECADO" else "#43A047")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#CCCCCC")

    ax.set_title(
        "Catálogo de Archivos Bronze — Estado y Trazabilidad\n"
        "Rojo = deprecado (no usar en modelos)  |  Verde = activo",
        fontsize=12, fontweight="bold", pad=16,
    )

    fig.text(
        0.5, 0.01,
        "Todos los archivos activos incluyen metadatos CF-1.8: "
        "ds.source, ds.history, ds.processing, ds.generated_by, ds.period_start, ds.period_end",
        ha="center", fontsize=8, color="#555", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = OUT_DIR / "DI04_bronze_catalog.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI04 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DI05 — Inventario completo de variables disponibles por fuente
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI05_variable_catalogue():
    """
    Inventario completo de ~60 variables brutas agrupadas por fuente.
    Estado = disponibilidad técnica (disponible / pendiente descarga / pendiente GEE).
    La selección de features para el modelo es análisis posterior (importancia, correlación).

    Panel izquierdo : conteo de variables por fuente y estado de disponibilidad
    Panel derecho   : tabla completa agrupada por fuente (sin juicio de selección)
    """

    # ── Inventario por fuente ─────────────────────────────────────────────────
    # Cada entrada: (nombre_variable, tipo, periodo, resolucion, estado, nota_breve)
    # estado: "D" = Disponible  |  "PD" = Pendiente descarga  |  "PG" = Pendiente GEE
    #         "DC" = Disponible con condición (pocos años / poca cobertura)

    SOURCES = [
        {
            "nombre": "PISCOp v2.1 update",
            "color": "#1E88E5",
            "resolucion": "0.05° (~5 km)",
            "vars": [
                ("pr  (precipitación diaria)",       "Precipitación",  "1981-2019", "D",
                 "QM mensual 100 cuantiles vs SENAMHI 1984-2013"),
            ],
        },
        {
            "nombre": "PISCOt v1.2",
            "color": "#FB8C00",
            "resolucion": "0.01° (~1 km)",
            "vars": [
                ("tmax  (temperatura máxima)",        "Temperatura",    "1981-2020", "D",
                 "Corrección delta-T mensual vs SENAMHI"),
                ("tmin  (temperatura mínima)",        "Temperatura",    "1981-2020", "D",
                 "Corrección delta-T mensual vs SENAMHI"),
            ],
        },
        {
            "nombre": "SENAMHI (estaciones)",
            "color": "#43A047",
            "resolucion": "puntual",
            "vars": [
                ("pr  (8 pluviómetros)",              "Precipitación",  "1981-2016", "D",
                 "Referencia para calibración QM"),
                ("tmax / tmin  (2-4 estaciones)",     "Temperatura",    "1981-2016", "D",
                 "Referencia para corrección delta-T"),
            ],
        },
        {
            "nombre": "ERA5-Land (CDS, 0.1°)",
            "color": "#607D8B",
            "resolucion": "0.1° (~11 km)",
            "vars": [
                # Precipitación / hidrología
                ("tp  (precipitación total)",         "Precipitación",  "1981-1995*", "PD",
                 "Sobreestima ~2x vs PISCOp (4.05 vs 1.98 mm/d)"),
                ("ro  (escorrentía total)",           "Hidrología",     "1981-1995*", "PD",
                 "Escorrentía superficial + subsuperficial"),
                ("sro  (escorrentía superficial)",    "Hidrología",     "1981-1995*", "PD", ""),
                ("ssro  (escorrentía subsup.)",       "Hidrología",     "1981-1995*", "PD", ""),
                ("smlt  (fusión de nieve)",           "Nieve",          "1981-1995*", "PD", ""),
                ("sf  (caída de nieve)",              "Nieve",          "1981-1995*", "PD", ""),
                # Evaporación
                ("e  (evaporación total)",            "Evapotransp.",   "1981-1995*", "PD", ""),
                ("pev  (evap. potencial)",            "Evapotransp.",   "1981-1995*", "PD",
                 "Alternativa a PET Hargreaves"),
                ("evabs  (evap. suelo desnudo)",      "Evapotransp.",   "1981-1995*", "PD", ""),
                ("evaow  (evap. aguas abiertas)",     "Evapotransp.",   "1981-1995*", "PD", ""),
                ("evavt  (evap. vegetación)",         "Evapotransp.",   "1981-1995*", "PD", ""),
                # Radiación
                ("ssrd  (rad. solar descendente)",    "Radiación",      "1981-1995*", "PD",
                 "Input alternativo para cálculo PET"),
                ("strd  (rad. térmica descendente)",  "Radiación",      "1981-1995*", "PD", ""),
                ("ssr  (rad. solar neta)",            "Radiación",      "1981-1995*", "PD", ""),
                ("str  (rad. térmica neta)",          "Radiación",      "1981-1995*", "PD", ""),
                # Temperatura / viento / presión
                ("t2m  (temperatura 2 m)",            "Temperatura",    "1981-1995*", "PD",
                 "Complementa PISCOt; útil en zonas sin datos"),
                ("d2m  (punto de rocío 2 m)",         "Humedad atm.",   "1981-1995*", "PD",
                 "Proxy humedad relativa para PET"),
                ("skt  (temperatura superficial)",    "Temperatura",    "1981-1995*", "PD", ""),
                ("u10  (viento U 10 m)",              "Viento",         "1981-1995*", "PD",
                 "Input velocidad de viento para PM"),
                ("v10  (viento V 10 m)",              "Viento",         "1981-1995*", "PD", ""),
                ("sp  (presión superficial)",         "Atmósfera",      "1981-1995*", "PD", ""),
                # Humedad de suelo
                ("swvl1  (humedad suelo capa 1, 0-7cm)",  "Suelo",      "1981-1995*", "PD",
                 "Estado antecedente de saturación"),
                ("swvl2  (humedad suelo capa 2, 7-28cm)", "Suelo",      "1981-1995*", "PD", ""),
                ("swvl3  (humedad suelo capa 3, 28-100cm)","Suelo",     "1981-1995*", "PD", ""),
                ("swvl4  (humedad suelo capa 4, 100-289cm)","Suelo",    "1981-1995*", "PD", ""),
                # Temperatura de suelo
                ("stl1  (temp. suelo capa 1)",        "Suelo",          "1981-1995*", "PD", ""),
                ("stl2  (temp. suelo capa 2)",        "Suelo",          "1981-1995*", "PD", ""),
                ("stl3  (temp. suelo capa 3)",        "Suelo",          "1981-1995*", "PD", ""),
                ("stl4  (temp. suelo capa 4)",        "Suelo",          "1981-1995*", "PD", ""),
                # Nieve / vegetación / albedo
                ("snowc  (cobertura de nieve %)",     "Nieve",          "1981-1995*", "PD",
                 "Disponible en Script 42; max. anual 0-80%"),
                ("sd  (profundidad nieve)",           "Nieve",          "1981-1995*", "PD", ""),
                ("sde  (equiv. agua nieve)",          "Nieve",          "1981-1995*", "PD", ""),
                ("asn  (albedo nieve)",               "Nieve",          "1981-1995*", "PD", ""),
                ("lai_lv  (LAI veg. baja)",           "Vegetación",     "1981-1995*", "PD", ""),
                ("lai_hv  (LAI veg. alta)",           "Vegetación",     "1981-1995*", "PD", ""),
                ("fal  (albedo pronóstico)",          "Superficie",     "1981-1995*", "PD", ""),
            ],
        },
        {
            "nombre": "SNIRH / ANA (caudal)",
            "color": "#00ACC1",
            "resolucion": "puntual",
            "vars": [
                ("Q  47E214D2 Santo Domingo (614 m)",  "Caudal obs.",   "2020-09→hoy", "D",
                 "Única estación con datos desde 2020; 120 días en D6"),
                ("Q  47E22148 Callantama (780 m)",     "Caudal obs.",   "2023-04→hoy", "DC",
                 "Histórico insuficiente para calibración (< 3 años)"),
                ("Q  47E257D8 Vichaycocha (3880 m)",   "Caudal obs.",   "2023-01→hoy", "DC",
                 "Histórico insuficiente para calibración (< 3 años)"),
            ],
        },
        {
            "nombre": "NOAA (ONI/ENSO)",
            "color": "#8BC34A",
            "resolucion": "índice global",
            "vars": [
                ("oni_index  (Oceanic Niño Index)",    "Clima global",  "1950-2026", "D",
                 "#2 importancia TFT (VSN weights); ENSO modula lluvia andina"),
            ],
        },
        {
            "nombre": "DEM SRTM",
            "color": "#795548",
            "resolucion": "90 m / 30 m",
            "vars": [
                ("elevación media por sub-cuenca",     "Topografía",    "estático", "D",
                 "521 m (Bajo Chancay) → 4507 m (Alto Chancay)"),
                ("pendiente media",                    "Topografía",    "estático", "D", ""),
                ("área acumulada",                     "Topografía",    "estático", "D", ""),
            ],
        },
        {
            "nombre": "GEOCATMIN 1:100k",
            "color": "#A1887F",
            "resolucion": "vectorial 1:100k",
            "vars": [
                ("litología / unidad geológica",       "Geología",      "estático", "D",
                 "Permeabilidad indirecta; rasterizado e intersectado"),
            ],
        },
        {
            "nombre": "MODIS / GEE",
            "color": "#26A69A",
            "resolucion": "250-500 m",
            "vars": [
                ("NDVI  (MOD13Q1, 250m, 16-day)",      "Vegetación",    "2000-2025", "PG",
                 "GEE2a — estado vegetación cuenca"),
                ("EVI  (MOD13Q1, 250m, 16-day)",       "Vegetación",    "2000-2025", "PG",
                 "GEE2a — índice vegetación mejorado"),
                ("LSWI  (MOD09GA, 500m, diario)",      "Agua/Veg.",     "2000-2025", "PG",
                 "GEE2c — Land Surface Water Index"),
                ("LST_día  (MOD11A1, 1km, diario)",    "Temperatura",   "2000-2025", "PG",
                 "GEE2b — temp. superficial terrestre día"),
                ("LST_noche  (MOD11A1, 1km, diario)",  "Temperatura",   "2000-2025", "PG",
                 "GEE2b — temp. superficial terrestre noche"),
                ("NDSI/snow  (MOD10A1, 500m, diario)", "Nieve",         "2000-2025", "PG",
                 "GEE1 — cobertura de nieve andina"),
                ("ET  (MOD16A2GF, 500m, 8-day)",       "Evapotransp.",  "2001-2025", "PG",
                 "GEE6 — evapotranspiración real MODIS"),
                ("PET  (MOD16A2GF, 500m, 8-day)",      "Evapotransp.",  "2001-2025", "PG",
                 "GEE6 — evapotranspiración potencial MODIS"),
            ],
        },
        {
            "nombre": "Sentinel-1 SAR / GEE",
            "color": "#5C6BC0",
            "resolucion": "10 m",
            "vars": [
                ("VV  (retrodispersión VV)",  "SAR",  "2014-2025", "PG",
                 "GEE10 — sensible a humedad suelo y agua; ~2.7 GB total"),
                ("VH  (retrodispersión VH)",  "SAR",  "2014-2025", "PG",
                 "GEE10 — sensible a vegetación"),
                ("VV/VH  (ratio SAR)",        "SAR",  "2014-2025", "PG",
                 "GEE10 — discrimina agua/vegetación/suelo"),
            ],
        },
        {
            "nombre": "Sentinel-2 MSI / GEE",
            "color": "#26C6DA",
            "resolucion": "10-30 m",
            "vars": [
                ("NDVI_S2  (B8-B4)/(B8+B4)",  "Vegetación", "2017-2025", "PG",
                 "GEE9 — 30m mensual, año a año (~2.3 GB/año)"),
                ("NDWI_S2  (B3-B8)/(B3+B8)",  "Agua",       "2017-2025", "PG",
                 "GEE9 — índice de agua superficial"),
                ("NDSI_S2  (B3-B11)/(B3+B11)", "Nieve",     "2017-2025", "PG",
                 "GEE9 — índice nieve a 30m"),
            ],
        },
        {
            "nombre": "SMAP / GEE",
            "color": "#EF9A9A",
            "resolucion": "10 km",
            "vars": [
                ("ssm  (humedad sup. 0-5cm)",        "Suelo", "2015-2025", "PG",
                 "GEE8 — humedad suelo superficial; proxy estado antecedente"),
                ("susm  (humedad sub-sup. 0-100cm)", "Suelo", "2015-2025", "PG",
                 "GEE8 — humedad perfil suelo"),
                ("smp  (profile 0-100cm)",           "Suelo", "2015-2025", "PG",
                 "GEE8 — humedad perfil completo"),
            ],
        },
        {
            "nombre": "Derivadas (D6)",
            "color": "#5C4B9F",
            "resolucion": "calculado",
            "vars": [
                ("tmean_c  = (tmax+tmin)/2",           "Temperatura",   "1981-2020", "D",
                 "Input PET y feature térmica directa"),
                ("pet_mm  (Penman-M. + HS-cal)",       "Evapotransp.",  "1981-2020", "D",
                 "PM 1981-2016; HS×k_mensual 2017-2020"),
                ("api  (precip. antecedente índice)",  "Hidrología",    "1981-2020", "D",
                 "#1 importancia TFT (VSN weights); decaimiento exp."),
                ("spi_30d  (SPI 30 días)",             "Sequía",        "1981-2020", "D",
                 "Estandarizado solo con media/sigma de train"),
                ("spi_90d  (SPI 90 días)",             "Sequía",        "1981-2020", "D",
                 "Escala estacional"),
                ("water_deficit_30d  (PET-PR acum.)",  "Hidrología",    "1981-2020", "D",
                 "Balance hídrico a corto plazo"),
                ("pr_sum_next_7d  (TARGET)",           "Target",        "1981-2019", "D",
                 "Suma precipitación 7 días siguientes — variable objetivo"),
            ],
        },
    ]

    # Colores de fondo por estado
    STATE_BG = {
        "D":  "#E8F5E9",   # disponible → verde claro
        "DC": "#FFF9C4",   # disponible con condición → amarillo muy claro
        "PD": "#E3F2FD",   # pendiente descarga → azul claro
        "PG": "#F3E5F5",   # pendiente GEE → lila claro
    }
    STATE_LABEL = {
        "D":  "Disponible",
        "DC": "Disponible\n(cond.)",
        "PD": "Pend. descarga\nERA5 1996-2025",
        "PG": "Pend. GEE",
    }
    STATE_COLOR = {
        "D":  "#2E7D32",
        "DC": "#F9A825",
        "PD": "#1565C0",
        "PG": "#7B1FA2",
    }

    # Conteo por fuente para el panel de barras
    src_names  = [s["nombre"] for s in SOURCES]
    src_colors = [s["color"]  for s in SOURCES]
    counts_D   = [sum(1 for v in s["vars"] if v[3] == "D")  for s in SOURCES]
    counts_DC  = [sum(1 for v in s["vars"] if v[3] == "DC") for s in SOURCES]
    counts_PD  = [sum(1 for v in s["vars"] if v[3] == "PD") for s in SOURCES]
    counts_PG  = [sum(1 for v in s["vars"] if v[3] == "PG") for s in SOURCES]

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(1, 10, figure=fig, wspace=0.5)

    ax_bar = fig.add_subplot(gs[0, :3])
    ax_tab = fig.add_subplot(gs[0, 3:])

    # ── Panel A: Barras apiladas por fuente ──────────────────────────────────
    y_pos = np.arange(len(src_names))
    left_D  = np.zeros(len(src_names))
    left_DC = np.array(counts_D, dtype=float)
    left_PD = left_DC + np.array(counts_DC, dtype=float)
    left_PG = left_PD + np.array(counts_PD, dtype=float)

    ax_bar.barh(y_pos, counts_D,  left=left_D,  height=0.6,
                color="#2E7D32", label="Disponible", alpha=0.85)
    ax_bar.barh(y_pos, counts_DC, left=left_DC, height=0.6,
                color="#F9A825", label="Disponible (condición)", alpha=0.85)
    ax_bar.barh(y_pos, counts_PD, left=left_PD, height=0.6,
                color="#1565C0", label="Pendiente descarga ERA5", alpha=0.75)
    ax_bar.barh(y_pos, counts_PG, left=left_PG, height=0.6,
                color="#7B1FA2", label="Pendiente GEE", alpha=0.75)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(src_names, fontsize=8)
    ax_bar.set_xlabel("N.° de variables", fontsize=9)
    ax_bar.set_title("Variables disponibles\npor fuente de datos",
                     fontsize=10, fontweight="bold")
    ax_bar.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_bar.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax_bar.set_facecolor("#FAFAFA")

    # Total al final de cada barra
    totals = [d + dc + pd + pg
              for d, dc, pd, pg in zip(counts_D, counts_DC, counts_PD, counts_PG)]
    for i, tot in enumerate(totals):
        if tot > 0:
            ax_bar.text(tot + 0.1, i, str(tot), va="center", fontsize=8,
                        fontweight="bold", color="#333")

    # ── Panel B: Tabla agrupada por fuente ────────────────────────────────────
    ax_tab.axis("off")

    col_labels = ["Fuente", "Variable", "Tipo", "Período", "Estado", "Nota"]

    table_data   = []
    cell_colors  = []

    for src in SOURCES:
        for v_name, v_tipo, v_periodo, v_estado, v_nota in src["vars"]:
            table_data.append([
                src["nombre"],
                v_name,
                v_tipo,
                v_periodo,
                STATE_LABEL[v_estado],
                v_nota,
            ])
            bg = STATE_BG[v_estado]
            cell_colors.append([bg] * 6)

    tab = ax_tab.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        cellColours=cell_colors,
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.8)
    tab.scale(1.0, 1.30)

    # Formato encabezado + columna estado
    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and col == 4:  # columna Estado
            estado_val = table_data[row - 1][4].split("\n")[0]
            rkey = {v: k for k, v in {
                "D": "Disponible", "DC": "Disponible", "PD": "Pendiente descarga",
                "PG": "Pendiente GEE",
            }.items()}.get(estado_val, "D")
            for k, lbl in STATE_LABEL.items():
                if lbl.split("\n")[0] == estado_val:
                    rkey = k
                    break
            cell.set_facecolor(STATE_COLOR.get(rkey, "#888"))
            cell.set_text_props(color="white", fontweight="bold", fontsize=6.0)
        elif row > 0 and col == 0:  # columna Fuente — negrita si es primera fila del grupo
            cell.set_text_props(fontsize=6.5)
        cell.set_edgecolor("#CCCCCC")
        cell.PAD = 0.04

    # Nota al pie
    total_vars = len(table_data)
    n_disp = sum(1 for r in table_data if "Disponible" in r[4])
    n_pend = total_vars - n_disp

    ax_tab.set_title(
        f"Inventario completo — {total_vars} variables  "
        f"({n_disp} disponibles  |  {n_pend} pendientes de integrar)",
        fontsize=10, fontweight="bold", pad=8,
    )

    # Leyenda de colores de filas
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=STATE_BG["D"],  label="Disponible y procesada"),
        Patch(color=STATE_BG["DC"], label="Disponible con condición (cobertura limitada)"),
        Patch(color=STATE_BG["PD"], label="Pendiente: descarga ERA5 1996-2025 (~200 GB)"),
        Patch(color=STATE_BG["PG"], label="Pendiente: scripts Google Earth Engine"),
    ]
    ax_tab.legend(handles=legend_patches, loc="upper right",
                  fontsize=7, framealpha=0.9, bbox_to_anchor=(1.0, 1.06))

    fig.suptitle(
        "HidroAlerta Chancay-Huaral — Inventario de Variables Disponibles por Fuente\n"
        "Estado = disponibilidad técnica  ·  "
        "La selección de features para el modelo se realizará mediante análisis de importancia y correlación",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out = OUT_DIR / "DI05_variable_catalogue.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI05 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DI06 — Heatmap de disponibilidad de sensores satélite 1981-2025
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI06_satellite_timeline():
    """
    Heatmap donde filas = sensores, columnas = años.
    Color por estado de máscara: -1=gris (pre-lanzamiento), 0=amarillo (gap), +1=color.
    """
    SENSORS = [
        # (nombre, color, año_lanzamiento, año_fin_datos, nota_breve)
        ("ERA5-Land\n(CDS/GEE4)",   C["era5"],    1950, 1995,  "CDS: 1981-1995 ✓  |  1996-2025 pendiente"),
        ("CHIRPS\n(0.05°)",         C["pr"],      1981, 2025,  "1981-2019 ✓  |  2020 pendiente Script05"),
        ("PISCOp v2.1\n(0.05°)",    "#1565C0",    1981, 2019,  "1981-2019 ✓  |  termina 2019"),
        ("PISCOt v1.2\n(0.01°)",    C["temp"],    1981, 2020,  "1981-2020 ✓"),
        ("Landsat 5/7/8\n(30m)",    C["landsat"], 1984, 2025,  "GEE3/GEE7 — pendiente descarga"),
        ("MODIS Terra\n(250-500m)", C["modis"],   2000, 2025,  "GEE1/2a/2b/2c/6 — pendiente descarga"),
        ("SMAP\n(10km)",            C["smap"],    2015, 2025,  "GEE8 — pendiente descarga"),
        ("Sentinel-1\n(SAR 10m)",   C["s1"],      2014, 2025,  "GEE10 — pendiente descarga"),
        ("Sentinel-2\n(MSI 10m)",   C["s2"],      2017, 2025,  "GEE9 — pendiente, año a año"),
    ]

    years = list(range(1981, 2026))
    n_sensors = len(SENSORS)
    n_years = len(years)

    # Construir matriz: -1=pre-lanzamiento, 0=gap/pendiente, 1=disponible
    matrix = np.full((n_sensors, n_years), -1.0)
    avail_overrides = {
        # (sensor_idx, year): state
        # ERA5-Land 1981-1995
        **{(0, yr - 1981): 1 for yr in range(1981, 1996)},
        **{(0, yr - 1981): 0 for yr in range(1996, 2026)},
        # CHIRPS 1981-2019
        **{(1, yr - 1981): 1 for yr in range(1981, 2020)},
        **{(1, yr - 1981): 0 for yr in range(2020, 2026)},
        # PISCOp 1981-2019
        **{(2, yr - 1981): 1 for yr in range(1981, 2020)},
        **{(2, yr - 1981): 0 for yr in range(2020, 2026)},
        # PISCOt 1981-2020
        **{(3, yr - 1981): 1 for yr in range(1981, 2021)},
        **{(3, yr - 1981): 0 for yr in range(2021, 2026)},
        # Landsat 1984+
        **{(4, yr - 1981): 0 for yr in range(1984, 2026)},
        # MODIS 2000+
        **{(5, yr - 1981): 0 for yr in range(2000, 2026)},
        # SMAP 2015+
        **{(6, yr - 1981): 0 for yr in range(2015, 2026)},
        # Sentinel-1 2014+
        **{(7, yr - 1981): 0 for yr in range(2014, 2026)},
        # Sentinel-2 2017+
        **{(8, yr - 1981): 0 for yr in range(2017, 2026)},
    }
    for (si, yi), state in avail_overrides.items():
        matrix[si, yi] = state

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#FAFAFA")

    for si, (sname, scol, slaunch, send, snote) in enumerate(SENSORS):
        for yi, yr in enumerate(years):
            state = matrix[si, yi]
            if state == -1:
                bg = "#E8E8E8"
                ec = "#CCCCCC"
            elif state == 0:
                bg = "#FFF9C4"
                ec = "#F9A825"
            else:
                bg = scol
                ec = scol
            rect = mpatches.FancyBboxPatch(
                (yi - 0.45, si - 0.4), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=bg, edgecolor=ec, linewidth=0.4, zorder=3,
            )
            ax.add_patch(rect)

    # Etiquetas eje Y
    ax.set_yticks(range(n_sensors))
    ax.set_yticklabels([s[0] for s in SENSORS], fontsize=8)

    # Etiquetas eje X (cada 5 años)
    xtick_pos = [i for i, yr in enumerate(years) if yr % 5 == 0]
    xtick_lbl = [years[i] for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=8, rotation=45)
    ax.set_xlim(-0.5, n_years - 0.5)
    ax.set_ylim(-0.7, n_sensors - 0.3)
    ax.invert_yaxis()

    # Líneas de lanzamiento
    for si, (sname, scol, slaunch, send, snote) in enumerate(SENSORS):
        if slaunch > 1981:
            launch_xi = slaunch - 1981
            ax.axvline(launch_xi, color=scol, lw=1.2, ls="--", alpha=0.6, zorder=4)

    # Notas laterales
    for si, (sname, scol, slaunch, send, snote) in enumerate(SENSORS):
        ax.text(n_years + 0.3, si, snote, va="center", fontsize=6.5,
                color="#555", style="italic")

    # 4-era shading (full height)
    for xe0, xe1, ebg in [
        (0,           2000 - 1981, "#F5F5F5"),
        (2000 - 1981, 2014.25 - 1981, "#E8F5E9"),
        (2014.25 - 1981, 2017.24 - 1981, "#E3F2FD"),
        (2017.24 - 1981, n_years, "#EDE7F6"),
    ]:
        ax.axvspan(xe0 - 0.5, xe1 - 0.5, color=ebg, alpha=0.3, zorder=0)

    # Etiquetas de eras
    era_labels = [
        (0, 2000 - 1981, "Pre-MODIS\n(1981-1999)"),
        (2000 - 1981, 2014.25 - 1981, "MODIS\n(2000-2013)"),
        (2014.25 - 1981, 2017.24 - 1981, "MODIS+S1\n(2014-2016)"),
        (2017.24 - 1981, n_years, "Constelación\ncompleta (2017+)"),
    ]
    for xe0, xe1, elbl in era_labels:
        ax.text((xe0 + xe1) / 2 - 0.5, -0.55, elbl, ha="center", va="top",
                fontsize=7.5, color="#444", fontweight="bold")

    # Leyenda
    legend_patches = [
        mpatches.Patch(color="#E8E8E8", label="Pre-lanzamiento (máscara = -1)"),
        mpatches.Patch(color="#FFF9C4", label="Lanzado, sin datos descargados (máscara = 0)"),
        mpatches.Patch(color=C["modis"], label="Datos disponibles (máscara = +1)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7.5,
              framealpha=0.9, bbox_to_anchor=(1.0, 1.12))

    ax.set_title(
        "Disponibilidad de Sensores Satélite por Año — HidroAlerta Chancay-Huaral\n"
        "Estado actual: solo ERA5 1981-1995, CHIRPS/PISCO 1981-2019 y PISCOt 1981-2020 descargados. "
        "Satélites GEE pendientes.",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "DI06_satellite_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI06 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DI07 — Arquitectura dos niveles (TFT diario + LightGBM mensual)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI07_two_level_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, facecolor, edgecolor, text, fontsize=9, text_color="white"):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.15",
                              facecolor=facecolor, edgecolor=edgecolor,
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                multialignment="center", zorder=4)

    def arrow(x0, y0, x1, y1, label="", col="#555"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                    zorder=5)
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx + 0.1, my, label, fontsize=7.5, color=col, style="italic")

    # ── Entradas ──────────────────────────────────────────────────────────────
    box(0.2, 7.0, 3.0, 1.5, "#1565C0", "#0D47A1",
        "ENTRADAS DIARIAS\n(D6/D7, 1981-2025)\nPISCO · ERA5 · GEE · ONI")
    box(0.2, 5.0, 3.0, 1.5, "#4A148C", "#311B92",
        "ENTRADAS MENSUALES\n(D6_monthly)\nAgregados + ENSO forecast")

    # ── Nivel 2: LightGBM mensual ─────────────────────────────────────────────
    box(4.5, 5.0, 3.5, 2.5, "#7B1FA2", "#6A1B9A",
        "NIVEL 2\nLightGBM Mensual\n─────────────\nHorizonte: 1-6 meses\nEncoder: 24 meses\n"
        "SHAP explainability")
    arrow(3.2, 5.75, 4.5, 6.25, "features mensuales")

    # ── Nivel 1: TFT diario ───────────────────────────────────────────────────
    box(4.5, 1.8, 3.5, 2.8, "#1565C0", "#0D47A1",
        "NIVEL 1\nTFT Diario\n─────────────\nHorizonte: 1-14 días\nEncoder: 90 días\n"
        "QuantileLoss P10/P50/P90\nVSN + Temporal Attention")
    arrow(3.2, 7.75, 4.5, 3.2, "features diarios")

    # ── Flecha inter-nivel ────────────────────────────────────────────────────
    arrow(8.0, 6.25, 9.2, 3.6, "seasonal_forecast\n(known_future)", "#7B1FA2")

    # ── Salidas Nivel 2 ───────────────────────────────────────────────────────
    box(9.2, 5.3, 3.2, 2.2, "#880E4F", "#6A1B9A",
        "SALIDAS NIVEL 2\n─────────────\nQ_mensual (m³/s)\npr_mensual (mm)\nAnomalia ENSO\nRiesgo sequía")
    arrow(8.0, 6.8, 9.2, 6.4)

    # ── Salidas Nivel 1 ───────────────────────────────────────────────────────
    box(9.2, 1.5, 3.2, 3.5, "#B71C1C", "#0D47A1",
        "SALIDAS NIVEL 1\n─────────────\nQ_7d  [P10,P50,P90]\npr_next_7d  (mm)\nflood_risk_index\nsnow_melt_risk\net_deficit_7d")
    arrow(8.0, 3.2, 9.2, 3.25)

    # ── Explicabilidad ────────────────────────────────────────────────────────
    box(12.6, 5.3, 3.0, 2.2, "#004D40", "#00695C",
        "SHAP (Nivel 2)\n─────────────\nWaterfall plots\nFeature importance\nby sub-cuenca")
    arrow(12.4, 6.4, 12.6, 6.4)

    box(12.6, 1.5, 3.0, 3.5, "#004D40", "#00695C",
        "TFT Explicabilidad\n─────────────\nVSN weights\n(qué features usa)\nTemporal Attention\n(qué días importan)\nSARIMAX baseline")
    arrow(12.4, 3.25, 12.6, 3.25)

    # ── Dashboard ─────────────────────────────────────────────────────────────
    box(6.5, 0.1, 3.0, 1.3, "#37474F", "#263238",
        "Streamlit Dashboard + APScheduler\n(inferencia semanal automática)")
    arrow(8.0, 1.8, 8.0, 1.4)
    arrow(10.8, 1.5, 9.5, 1.4)

    # ── Leyenda / nota ────────────────────────────────────────────────────────
    ax.text(0.2, 0.3,
            "D6_monthly: agregación mensual de D6 (script 60)  ·  "
            "seasonal_forecast = output Nivel 2 → known_future Nivel 1 (sin leakage)\n"
            "ONI operacional: oni_lag2 = oni.shift(2) — publicación NOAA ~60 días después del mes de referencia",
            fontsize=7, color="#555", style="italic", va="bottom")

    ax.set_title(
        "Arquitectura Dos Niveles — HidroAlerta Chancay-Huaral\n"
        "Nivel 1: TFT diario (1-14 días, probabilístico)  ·  Nivel 2: LightGBM mensual (1-6 meses, SHAP)",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    out = OUT_DIR / "DI07_two_level_arch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI07 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DI08 — Mapa de explicabilidad "quién explica qué"
# ═══════════════════════════════════════════════════════════════════════════════

def plot_DI08_explainability():
    ROWS = [
        # (pregunta, herramienta, modelo, tipo_salida, color_bg)
        ("¿Qué features usa el modelo\n(pesos globales)?",
         "TFT VSN weights\n(ã_j softmax)",
         "Nivel 1 — TFT",
         "Ranking de variables\n(barra por feature)",
         "#E3F2FD"),
        ("¿En qué días pasados se\nenfocó para esta predicción?",
         "TFT Temporal Attention\n(heatmap Q·K^T/√d_k)",
         "Nivel 1 — TFT",
         "Heatmap tiempo\n(encoder 90 días)",
         "#E3F2FD"),
        ("¿Por qué este caudal mensual\npredominó en el pronóstico?",
         "SHAP Shapley values\n(Σφ_j = f(x) - E[f(x)])",
         "Nivel 2 — LightGBM",
         "Waterfall plot\n(contribución por feature)",
         "#F3E5F5"),
        ("¿El modelo captura más que\nuna línea base estadística?",
         "SARIMAX comparativo\n(φ(B)Φ(B^s) baseline)",
         "Baseline vs Nivel 1/2",
         "NSE / KGE / QS_τ\nvs SARIMAX",
         "#FFF9C4"),
        ("¿El intervalo P10-P90 está\nbien calibrado?",
         "Coverage 80% + QS_τ\n(Quantile Score)",
         "Nivel 1 — TFT",
         "Coverage real ~80%\nCalibración probabilística",
         "#E8F5E9"),
        ("¿Cuál sub-cuenca aporta más\nal caudal total?",
         "SHAP por entidad\n(entity_id como feature)",
         "Nivel 2 — LightGBM",
         "SHAP por sub-cuenca\n(mapa espacial)",
         "#F3E5F5"),
        ("¿Qué señal ENSO amplifica\nel riesgo de inundación?",
         "TFT Attention + SHAP ONI\n(oni_lag2 feature)",
         "Nivel 1 + Nivel 2",
         "Contribución oni_lag2\nen eventos extremos",
         "#FBE9E7"),
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")

    col_labels = ["Pregunta de usuario", "Herramienta técnica", "Modelo responsable",
                  "Tipo de salida"]
    table_data  = [[r[0], r[1], r[2], r[3]] for r in ROWS]
    cell_colors = [[r[4]] * 4 for r in ROWS]

    tab = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.8)

    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_edgecolor("#CCCCCC")
        cell.PAD = 0.06

    # Colorear la columna "Modelo responsable"
    model_colors = {
        "Nivel 1 — TFT":      "#1565C0",
        "Nivel 2 — LightGBM": "#7B1FA2",
        "Baseline vs Nivel 1/2": "#E65100",
        "Nivel 1 + Nivel 2":  "#00695C",
    }
    for i, r in enumerate(ROWS):
        cell = tab[i + 1, 2]
        mc = model_colors.get(r[2], "#555")
        cell.set_facecolor(mc)
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "HidroAlerta — Mapa de Explicabilidad: ¿Quién Explica Qué?\n"
        "Cada pregunta de usuario se responde con una herramienta técnica específica del modelo.",
        fontsize=12, fontweight="bold", pad=14,
    )

    # Leyenda lateral
    legend_patches = [
        mpatches.Patch(color="#E3F2FD", label="TFT Nivel 1 (diario 1-14d)"),
        mpatches.Patch(color="#F3E5F5", label="LightGBM Nivel 2 (mensual 1-6m)"),
        mpatches.Patch(color="#FFF9C4", label="Comparación baseline SARIMAX"),
        mpatches.Patch(color="#E8F5E9", label="Evaluación probabilística"),
        mpatches.Patch(color="#FBE9E7", label="Análisis de eventos extremos ENSO"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.9, bbox_to_anchor=(1.0, 1.05))

    fig.text(
        0.5, 0.01,
        "VSN = Variable Selection Network  ·  SHAP = SHapley Additive exPlanations  ·  "
        "QS_τ = Quantile Score  ·  KGE = Kling-Gupta Efficiency  ·  oni_lag2 = ONI con retraso 2 meses",
        ha="center", fontsize=7.5, color="#555", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = OUT_DIR / "DI08_explainability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"DI08 guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Resumen Markdown
# ═══════════════════════════════════════════════════════════════════════════════

def write_markdown(d6_stats, snirh_periods, era5_groups):
    nan_pct = d6_stats["nan_pct"]
    n_rows, _ = d6_stats["shape"]
    p_start, p_end = d6_stats["period"]
    n_ent = d6_stats["n_entities"]

    era5_avail = {g: len(v) for g, v in era5_groups.items()}
    era5_total = 45  # 1981-2025

    lines = [
        "# Inventario y Coherencia de Datos — HidroAlerta Chancay-Huaral",
        "",
        f"> Generado automáticamente por `scripts/41_data_inventory.py`  "
        f"| {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Resumen ejecutivo",
        "",
        "| Ítem | Valor |",
        "|------|-------|",
        f"| Cuenca | Chancay-Huaral (3 062.62 km²) |",
        f"| Dataset principal | D6_multientity.csv |",
        f"| Filas D6 | {n_rows:,} ({n_ent} sub-cuencas × 14 610 días) |",
        f"| Período D6 | {p_start} → {p_end} |",
        f"| Split | Train 1981-2008 / Val 2009-2014 / Test 2015-2020 |",
        f"| Período test efectivo | **2015-2019** (pr_mm NaN en 2020) |",
        "",
        "## 2. Variables en D6 — Estado de completitud",
        "",
        "| Variable | Fuente | Período válido | NaN% | Estado |",
        "|----------|--------|---------------|------|--------|",
    ]

    VAR_ROWS = [
        ("pr_mm",     "PISCOp v2.1 update",    "1981-2019", "pr_mm"),
        ("tmax_c",    "PISCOt v1.2 + delta-T",  "1981-2020", "tmax_c"),
        ("tmin_c",    "PISCOt v1.2 + delta-T",  "1981-2020", "tmin_c"),
        ("pet_mm",    "PM (1981-2016) + HS-cal (2017-2020)", "1981-2020", "pet_mm"),
        ("q_mm",      "SNIRH 47E214D2 (auto)",  "2020-09 → 2020-12", "q_mm"),
        ("oni_index", "NOAA ONI",               "1950-2026", "oni_index"),
    ]
    for var, src, period, key in VAR_ROWS:
        pct = nan_pct.get(key, 0) or 0
        if pct == 0:
            estado = "✅ Completo"
        elif pct < 5:
            estado = "⚠️  Gap menor (2020)"
        elif pct > 90:
            estado = "❌ Solo 120 días"
        else:
            estado = "❌ Gap 2020"
        lines.append(f"| `{var}` | {src} | {period} | {pct:.1f}% | {estado} |")

    lines += [
        "",
        "### Notas críticas",
        "",
        "- **pr_mm en 2020 = NaN**: PISCOp v2.1 update termina el 2019-12-31. "
          "Afecta 3 294 filas (2.5% de D6). "
          "Solución pendiente: Script 05 (CHIRPS 2020, requiere internet).",
        "- **q_mm 99.2% NaN**: La estación hidrométrica automática 47E214D2 "
          "comenzó a registrar en septiembre 2020. "
          "No usar q_mm como target de entrenamiento (solo validación 2020-Q4).",
        "- **Período test efectivo**: usar **2015-2019** para evaluar métricas. "
          "No reportar resultados de 2020 hasta completar CHIRPS.",
        "",
        "## 3. Fuentes de datos por variable",
        "",
        "| Variable | Fuente | Resolución | Script | Corrección aplicada |",
        "|----------|--------|------------|--------|---------------------|",
        "| pr_mm | PISCOp v2.1 update (SENAMHI/IGP) | 0.05° (~5 km) | 04, 19 | QM mensual (100 cuantiles, 1984-2013) |",
        "| tmax_c | PISCOt v1.2 (SENAMHI/IGP) | 0.01° (~1 km) | 03b, 19 | Delta-T mensual vs. SENAMHI |",
        "| tmin_c | PISCOt v1.2 (SENAMHI/IGP) | 0.01° (~1 km) | 06, 19 | Delta-T mensual vs. SENAMHI |",
        "| pet_mm | ETP.nc (PM) + HS calibrado | 0.05° | 02, 16 | Calibración PM 1981-2016 → extensión HS |",
        "| q_mm | SNIRH 47E214D2 | estación puntual | 10, 20 | Conversión m³/s → mm/día |",
        "| oni_index | NOAA ONI (ERSSTv5) | índice puntual | 09 | Ninguna |",
        "| geología | GEOCATMIN 1:100k | vectorial | 14 | Rasterización e intersección sub-cuencas |",
        "",
        "## 4. Archivos Bronze — Estado",
        "",
        "| Archivo | Estado | Período | Uso en D6 |",
        "|---------|--------|---------|-----------|",
        "| B2_pr_basin_grid.nc | ❌ DEPRECADO (1981-2016) | — | No |",
        "| B2_tmax_basin_grid.nc | ❌ DEPRECADO (1981-2016) | — | No |",
        "| B2_tmin_basin_grid.nc | ❌ DEPRECADO (1981-2016) | — | No |",
        "| **B2_pr_basin_grid_v21.nc** | ✅ Activo | 1981-2019 | Sí |",
        "| **B2_tmax_basin_grid_v12.nc** | ✅ Activo | 1981-2020 | Sí |",
        "| **B2_tmin_basin_grid_v12.nc** | ✅ Activo | 1981-2020 | Sí |",
        "| **B2_pet_hargreaves_cal.nc** | ✅ Activo | 1981-2020 | Sí |",
        "| **B2_tmax_corrected_spatial.nc** | ✅ Activo | 1981-2020 | Sí |",
        "| **B2_tmin_corrected_spatial.nc** | ✅ Activo | 1981-2020 | Sí |",
        "",
        "## 5. ERA5-Land — Estado de descarga",
        "",
        "| Grupo | Variables | Años disponibles | Años faltantes |",
        "|-------|-----------|-----------------|----------------|",
    ]

    ERA5_VARS = {
        "accum1": "tp, ssrd, fal, slhf, sshf, str",
        "accum2": "e, ro, sro, snro, smlt, sde, sf, ssr",
        "inst1":  "t2m, d2m, u10, v10, sp",
        "inst2":  "strd, stl1, swvl1, swvl2, swvl3, lai_lv, lai_hv",
    }
    for grp, vars_str in ERA5_VARS.items():
        n_ok = era5_avail.get(grp, 0)
        n_miss = era5_total - n_ok
        yrs_ok = sorted(era5_groups.get(grp, []))
        if yrs_ok:
            ok_range = f"1981-{max(yrs_ok)}"
        else:
            ok_range = "—"
        if n_miss > 0:
            miss_yrs = [yr for yr in range(1981, 1981 + era5_total)
                        if yr not in era5_groups.get(grp, [])]
            if miss_yrs:
                if len(miss_yrs) == 1:
                    miss_range = f"{miss_yrs[0]} (1 año)"
                else:
                    miss_range = f"{min(miss_yrs)}-{max(miss_yrs)} ({n_miss} años)"
            else:
                miss_range = f"({n_miss} años)"
        else:
            miss_range = "—"
        lines.append(
            f"| {grp} | {vars_str} | {ok_range} ({n_ok} años) | {miss_range} |"
        )

    lines += [
        "",
        "> **Estrategia ERA5**: usar `GEE4_era5_raster.js` (GEE Code Editor) — exporta ERA5-Land 1981-2025 "
          "recortado a la cuenca (~160 MB, ~90 píxeles @ 0.1°). "
          "El script CDS API `08e_era5land_daily_cds.py` descarga una región mayor (~200 GB) y NO es necesario "
          "mientras GEE4 esté disponible.",
        "> API key CDS: disponible **solo** en `~/.cdsapirc` — NUNCA embeber en scripts. Solo usar si GEE4 falla.",
        "",
        "## 6. Estaciones de caudal observado (SNIRH)",
        "",
        "| Código | Nombre | Altitud | Período | Días válidos |",
        "|--------|--------|---------|---------|-------------|",
    ]
    for s in snirh_periods:
        n_days = (s["end"] - s["start"]).days
        lbl    = s["label"].split("\n")[0]
        alt    = s["label"].split("\n")[1] if "\n" in s["label"] else "—"
        lines.append(
            f"| — | {lbl} | {alt} | "
            f"{s['start'].strftime('%Y-%m-%d')} → {s['end'].strftime('%Y-%m-%d')} "
            f"| {n_days} |"
        )

    lines += [
        "",
        "## 7. Estrategia multi-modal satélite (D7)",
        "",
        "| Sensor | Script GEE | Período | Resolución | Máscaras |",
        "|--------|------------|---------|------------|---------|",
        "| Landsat 5/7/8 | GEE3/GEE7 | 1984-2025 | 30m mensual | -1/0/+1 |",
        "| MODIS (NDVI/EVI/LST/ET/snow) | GEE1/2a/2b/2c/6 | 2000-2025 | 250-1000m | -1/0/+1 |",
        "| SMAP soil moisture | GEE8 | 2015-2025 | 10km diario | -1/0/+1 |",
        "| Sentinel-1 SAR | GEE10 | 2014-2025 | 10m | -1/0/+1 |",
        "| Sentinel-2 MSI | GEE9 | 2017-2025 | 10-30m | -1/0/+1 |",
        "",
        "**Convención de máscara 3 estados**: `-1` = sensor no lanzado (pre-launch), "
        "`0` = lanzado pero sin observación disponible, `+1` = observación válida.",
        "",
        "**Modality Dropout** (entrenamiento): MODIS=0.15, S1=0.20, S2=0.25, ERA5=0.05.",
        "",
        "## 8. Arquitectura dos niveles",
        "",
        "```",
        "Nivel 2 — LightGBM mensual (1-6 meses)",
        "  · Encoder: 24 meses de historia agregada",
        "  · Inputs: 14 variables (pr_sum, tmax, pet, oni_lag2, snow, ...)",
        "  · Outputs: Q_mensual, pr_mensual, anomalía ENSO, riesgo sequía",
        "  · Explicabilidad: SHAP waterfall por sub-cuenca",
        "        ↓ seasonal_forecast → known_future (sin leakage)",
        "Nivel 1 — TFT diario (1-14 días, probabilístico)",
        "  · Encoder: 90 días de historia diaria",
        "  · Inputs: ~30 variables (D6/D7 + known_future = seasonal_forecast + clim_pet)",
        "  · Outputs: Q_7d [P10,P50,P90], pr_next_7d, flood_risk_index, snow_melt_risk",
        "  · Explicabilidad: VSN weights + Temporal Attention heatmap",
        "```",
        "",
        "**Nota anti-leakage**: `pet_mm` en el decoder se reemplaza por `clim_pet` "
        "(climatología diaria calculada solo con datos de entrenamiento). "
        "`oni_index` siempre se usa como `oni_lag2 = oni.shift(2)` para respetar "
        "el retraso operacional de publicación NOAA (~60 días).",
        "",
        "## 9. Tareas pendientes para completar la cadena de datos",
        "",
        "| Prioridad | Tarea | Script | Bloquea |",
        "|-----------|-------|--------|---------|",
        "| 🔴 Alta | Ejecutar GEE0 (CHIRPS) + GEE4 (ERA5) + GEE1 (snow) | GEE | D7 HITO1 |",
        "| 🔴 Alta | Descargar ERA5 1996-2025 via CDS API (~200 GB) | 08e | Modelos ERA5 completo |",
        "| 🔴 Alta | Obtener CHIRPS 2020 para pr_mm | 05 | Test completo 2020 |",
        "| 🟡 Media | Construir D6_monthly (agregación mensual) | 60 | LightGBM Nivel 2 |",
        "| 🟡 Media | Construir D7 (D6v2 + máscaras satélite) | 51 | TFT multi-modal |",
        "| 🟡 Media | Entrenar LightGBM Nivel 2 | 61 | Pronóstico estacional |",
        "| 🟡 Media | Regenerar D6 después de CHIRPS + oni_lag2 fix | 20 | Métricas de test |",
        "| 🟢 Baja | Ejecutar GEE HITO2 (MODIS, S1, S2, SMAP, Landsat) | GEE | D7 completo |",
        "| 🟢 Baja | Pipeline semanal automatizado | 70 | Producción |",
        "| 🟢 Baja | Dashboard Streamlit | 71 | Presentación concurso |",
        "",
        "---",
        f"*Generado por `scripts/41_data_inventory.py` — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    md_path = ROOT / "reports/S8_data_inventory.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Markdown guardado: {md_path}")
    return md_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=== Script 41: Inventario de datos ===")

    log.info("Leyendo estadísticas D6 ...")
    d6_stats = _read_d6_stats()
    log.info(f"D6: {d6_stats['shape'][0]:,} filas, {d6_stats['n_entities']} entidades, "
             f"{d6_stats['period'][0]} → {d6_stats['period'][1]}")

    log.info("Leyendo períodos SNIRH ...")
    snirh_periods = _read_snirh_periods()
    for s in snirh_periods:
        log.info(f"  {s['label'].split(chr(10))[0]}: "
                 f"{s['start'].date()} → {s['end'].date()}")

    log.info("Auditando archivos ERA5 ...")
    era5_groups = _read_era5_years()
    for g, yrs in era5_groups.items():
        log.info(f"  {g}: {len(yrs)} años — {yrs[:3]}...{yrs[-1] if yrs else '?'}")

    log.info("Generando DI01 — Línea de tiempo ...")
    plot_DI01_timeline(d6_stats, snirh_periods, era5_groups)

    log.info("Generando DI02 — Coherencia D6 ...")
    plot_DI02_d6_coherence(d6_stats)

    log.info("Generando DI03 — Estado ERA5 ...")
    plot_DI03_era5_status(era5_groups)

    log.info("Generando DI04 — Catálogo Bronze ...")
    plot_DI04_pipeline()

    log.info("Generando DI05 — Catálogo de variables ...")
    plot_DI05_variable_catalogue()

    log.info("Generando DI06 — Heatmap satélites ...")
    plot_DI06_satellite_timeline()

    log.info("Generando DI07 — Arquitectura dos niveles ...")
    plot_DI07_two_level_architecture()

    log.info("Generando DI08 — Mapa de explicabilidad ...")
    plot_DI08_explainability()

    log.info("Escribiendo resumen Markdown ...")
    md_path = write_markdown(d6_stats, snirh_periods, era5_groups)

    log.info("=== DONE ===")
    log.info(f"  Figuras: {OUT_DIR}/DI01-DI08")
    log.info(f"  Markdown: {md_path}")
    log.info("")
    log.info("NaN% por variable en D6:")
    for v, pct in d6_stats["nan_pct"].items():
        flag = "⚠" if (pct or 0) > 0 else "✓"
        log.info(f"  {flag} {v:15s}: {pct:.1f}%" if pct is not None else f"  — {v}: no disponible")


if __name__ == "__main__":
    main()
