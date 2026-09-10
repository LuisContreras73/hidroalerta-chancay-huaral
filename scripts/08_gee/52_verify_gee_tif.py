#!/usr/bin/env python3
"""
Script 52: Verificación visual de TIFs descargados de GEE.

Genera un mapa PNG con:
  - Banda seleccionada renderizada con colormap apropiado
  - Contorno del polígono de cuenca superpuesto (rojo)
  - Estadísticas: shape, res, NaN%, min/max/mean/std
  - Tabla lateral con resumen por banda

Uso:
    python scripts/52_verify_gee_tif.py --tif data/raw/gee/water_land/jrc_water_1984_1988_30m.tif
    python scripts/52_verify_gee_tif.py --tif data/raw/gee/sentinel1/s1_vv_vh_ratio_2020_30m.tif --band 1
    python scripts/52_verify_gee_tif.py --tif data/raw/gee/sentinel2/s2_ndvi_ndwi_ndsi_2022_30m.tif --rgb 0 3 6
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from matplotlib.gridspec import GridSpec

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "outputs/figures/gee/verify"
BASIN_SHP = ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite/Cuenca_Chancay___Huaral.shp"

STYLE = dict(
    bg      = "#0d1117",
    panel   = "#161b22",
    text    = "#c9d1d9",
    accent  = "#58a6ff",
    red     = "#ff7b72",
    green   = "#3fb950",
    yellow  = "#d29922",
)


def _cmap_for_tif(tif_path: Path) -> str:
    name = tif_path.name.lower()
    if any(x in name for x in ("ndvi", "evi", "lswi")):
        return "RdYlGn"
    if any(x in name for x in ("ndwi", "mndwi", "water", "jrc")):
        return "Blues"
    if "snow" in name or "ndsi" in name:
        return "PuBu"
    if "lst" in name or "thermal" in name:
        return "RdYlBu_r"
    if any(x in name for x in ("vv", "vh", "s1", "sar")):
        return "gray"
    if "landcover" in name or "igbp" in name or "worldcover" in name:
        return "tab20"
    if "smap" in name or "soil" in name:
        return "YlOrBr"
    if "chirps" in name or "precip" in name or "pr_" in name:
        return "Blues"
    return "viridis"


def _band_label(tif_path: Path, band_idx: int, total_bands: int) -> str:
    """Intenta construir un label descriptivo para la banda."""
    name = tif_path.stem
    if total_bands == 1:
        return name
    return f"{name} | banda {band_idx + 1}/{total_bands}"


def _read_stats(data: np.ndarray):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return dict(n=0, nan_pct=100, min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    total = data.size
    nan_pct = (total - valid.size) / total * 100
    return dict(
        n=valid.size,
        nan_pct=nan_pct,
        min=float(np.nanmin(valid)),
        max=float(np.nanmax(valid)),
        mean=float(np.nanmean(valid)),
        std=float(np.nanstd(valid)),
    )


def verify_tif(tif_path: Path, band_idx: int = 0, rgb_bands=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        n_bands    = src.count
        res_x      = abs(src.transform.a) * 111320
        res_y      = abs(src.transform.e) * 111320
        crs        = src.crs
        bounds     = src.bounds
        band_names = src.descriptions or [f"Banda {i+1}" for i in range(n_bands)]

        # Leer banda principal o RGB
        if rgb_bands and len(rgb_bands) == 3 and all(b < n_bands for b in rgb_bands):
            data_r = src.read(rgb_bands[0] + 1).astype(float)
            data_g = src.read(rgb_bands[1] + 1).astype(float)
            data_b = src.read(rgb_bands[2] + 1).astype(float)
            data_r[data_r == src.nodata] = np.nan if src.nodata else data_r[data_r == src.nodata]
            is_rgb = True
            band_idx_display = rgb_bands[0]
        else:
            band_idx = min(band_idx, n_bands - 1)
            data = src.read(band_idx + 1).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            is_rgb = False
            band_idx_display = band_idx

        # Leer todas las bandas para la tabla de estadísticas (max 20)
        stats_all = []
        for b in range(min(n_bands, 20)):
            arr = src.read(b + 1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            st = _read_stats(arr)
            st["name"] = band_names[b] or f"B{b+1}"
            stats_all.append(st)

    # Cargar shapefile de cuenca
    try:
        basin = gpd.read_file(str(BASIN_SHP)).to_crs(crs.to_epsg() or 4326)
    except Exception:
        basin = None

    # ─── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor=STYLE["bg"])
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.03)
    ax_map  = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    for ax in [ax_map, ax_info]:
        ax.set_facecolor(STYLE["panel"])

    # ─── Mapa principal ───────────────────────────────────────────────────────
    cmap = _cmap_for_tif(tif_path)

    if is_rgb:
        def _norm_band(d):
            p2, p98 = np.nanpercentile(d[np.isfinite(d)], [2, 98]) if np.any(np.isfinite(d)) else (0, 1)
            return np.clip((d - p2) / max(p98 - p2, 1e-9), 0, 1)
        rgb_img = np.stack([_norm_band(data_r), _norm_band(data_g), _norm_band(data_b)], axis=2)
        # Mask NaN
        mask = ~(np.isfinite(data_r) & np.isfinite(data_g) & np.isfinite(data_b))
        rgb_img[mask] = 0.0
        ax_map.imshow(rgb_img,
                      extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                      origin="upper", aspect="equal")
        main_stats = _read_stats(data_r)
    else:
        arr_plot = data.copy()
        p2, p98 = (np.nanpercentile(arr_plot[np.isfinite(arr_plot)], [2, 98])
                   if np.any(np.isfinite(arr_plot)) else (0, 1))
        im = ax_map.imshow(
            arr_plot,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin="upper", aspect="equal",
            cmap=cmap, vmin=p2, vmax=p98,
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax_map, fraction=0.03, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=STYLE["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=STYLE["text"], fontsize=8)
        cbar.outline.set_edgecolor(STYLE["text"])
        main_stats = _read_stats(arr_plot)

    # Contorno cuenca
    if basin is not None:
        try:
            basin.boundary.plot(ax=ax_map, color=STYLE["red"], linewidth=1.5, zorder=10)
        except Exception:
            pass

    ax_map.set_xlabel("Longitud", color=STYLE["text"], fontsize=9)
    ax_map.set_ylabel("Latitud", color=STYLE["text"], fontsize=9)
    ax_map.tick_params(colors=STYLE["text"], labelsize=8)
    for spine in ax_map.spines.values():
        spine.set_edgecolor(STYLE["panel"])

    # ─── Panel de información ─────────────────────────────────────────────────
    ax_info.axis("off")
    lines = [
        ("Archivo",    tif_path.name),
        ("Shape",      f"{n_bands} bandas"),
        ("Resolución", f"~{res_x:.0f}m × {res_y:.0f}m"),
        ("CRS",        str(crs)),
        ("Extent",     f"[{bounds.left:.3f}, {bounds.bottom:.3f}]"),
        ("",           f"[{bounds.right:.3f}, {bounds.top:.3f}]"),
        ("", ""),
        ("── Banda mostrada ──", ""),
        ("Índice",     f"{band_idx_display + 1}/{n_bands}"),
        ("NaN %",      f"{main_stats['nan_pct']:.1f}%"),
        ("Min",        f"{main_stats['min']:.4g}"),
        ("Max",        f"{main_stats['max']:.4g}"),
        ("Mean",       f"{main_stats['mean']:.4g}"),
        ("Std",        f"{main_stats['std']:.4g}"),
        ("", ""),
        (f"── Estadísticas por banda (primeras {len(stats_all)}) ──", ""),
    ]

    y0 = 0.98
    dy = 0.042
    for label, val in lines:
        color = STYLE["accent"] if label.startswith("──") else STYLE["text"]
        ax_info.text(0.02, y0, label, transform=ax_info.transAxes,
                     color=color, fontsize=8, va="top", fontfamily="monospace")
        if val:
            ax_info.text(0.52, y0, val, transform=ax_info.transAxes,
                         color=STYLE["text"], fontsize=8, va="top", fontfamily="monospace")
        y0 -= dy

    # Tabla de bandas
    col_w = [0.02, 0.30, 0.54, 0.72, 0.90]
    headers = ["#", "NaN%", "Min", "Max", "Mean"]
    for i, h in enumerate(headers):
        ax_info.text(col_w[i], y0, h, transform=ax_info.transAxes,
                     color=STYLE["accent"], fontsize=7, va="top", fontfamily="monospace", fontweight="bold")
    y0 -= dy * 0.8

    for st in stats_all:
        color = STYLE["red"] if st["nan_pct"] > 80 else STYLE["text"]
        vals = [
            str(stats_all.index(st) + 1),
            f"{st['nan_pct']:.0f}%",
            f"{st['min']:.3g}" if np.isfinite(st["min"]) else "NaN",
            f"{st['max']:.3g}" if np.isfinite(st["max"]) else "NaN",
            f"{st['mean']:.3g}" if np.isfinite(st["mean"]) else "NaN",
        ]
        for i, v in enumerate(vals):
            ax_info.text(col_w[i], y0, v, transform=ax_info.transAxes,
                         color=color, fontsize=6.5, va="top", fontfamily="monospace")
        y0 -= dy * 0.75
        if y0 < 0.02:
            break

    if n_bands > 20:
        ax_info.text(0.02, y0, f"... (+{n_bands - 20} bandas)", transform=ax_info.transAxes,
                     color=STYLE["yellow"], fontsize=7, va="top")

    # Leyenda cuenca
    if basin is not None:
        patch = mpatches.Patch(edgecolor=STYLE["red"], facecolor="none",
                               linewidth=1.5, label="Límite cuenca")
        ax_map.legend(handles=[patch], loc="lower right",
                      facecolor=STYLE["panel"], edgecolor=STYLE["text"],
                      labelcolor=STYLE["text"], fontsize=8)

    # Título
    nan_color = STYLE["red"] if main_stats["nan_pct"] > 50 else \
                STYLE["yellow"] if main_stats["nan_pct"] > 10 else STYLE["green"]
    fig.suptitle(
        f"{tif_path.name}   |   {n_bands} bandas   |   ~{res_x:.0f}m   |   "
        f"NaN {main_stats['nan_pct']:.1f}%",
        color=STYLE["text"], fontsize=11, y=0.99,
    )

    out_png = OUT_DIR / f"{tif_path.stem}_verify.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    plt.close()
    print(f"Verificacion guardada: {out_png}")
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Verificación visual de TIF GEE")
    parser.add_argument("--tif",  required=True, help="Ruta al TIF a verificar")
    parser.add_argument("--band", type=int, default=0,
                        help="Índice de banda a mostrar (0-based, default=0)")
    parser.add_argument("--rgb",  type=int, nargs=3, metavar=("R", "G", "B"),
                        help="Índices 0-based de bandas para composición RGB")
    args = parser.parse_args()

    tif_path = Path(args.tif)
    if not tif_path.exists():
        # Intentar ruta relativa desde ROOT
        tif_path = ROOT / args.tif
    if not tif_path.exists():
        print(f"ERROR: TIF no encontrado: {args.tif}")
        sys.exit(1)

    out = verify_tif(tif_path, band_idx=args.band, rgb_bands=args.rgb)
    print(f"Mapa: {out}")


if __name__ == "__main__":
    main()
