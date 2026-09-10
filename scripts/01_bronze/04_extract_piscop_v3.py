#!/usr/bin/env python3
"""
Script 01_bronze/04_extract_piscop_v3.py: Recorta PISCOp v3.0 (1981-2025, 0.1°) a la cuenca Chancay-Huaral.

Diferencias clave respecto a 04_extract_piscop_v21.py:
  - Archivo fuente: PISCOp_d.nc (no PISCOp_daily.nc)
  - Coordenadas: "latitude"/"longitude" (no "lat"/"lon")
  - Variable pr: "precipitation" (no "pc")
  - Dimensión tiempo: "Z1" con units=unknown → reconstruir por posición
  - Fill value: -1.1754940241844054e+38 (no -3.4e38)
  - N_DAYS: 16436 (1981-01-01 → 2025-12-31)
  - Output: B2_pr_basin_grid_v3.nc

Outputs:
  1. data/bronze/B2_pr_basin_grid_v3.nc   — grilla cuenca 0.1°, CF-1.8, zlib-4
  2. data/bronze/B2_pisco_v3_basin_mean.csv — serie diaria media cuenca (cos×frac)
  3. data/bronze/B2_pisco_v3_monthly.csv   — serie mensual PISCOp_m.nc (cuenca)
  4. outputs/figures/basin/                — mapas F05v3, F06v3, F07v3

Ref: Gutierrez, L. y Lavado-Casimiro, W. (2025). PISCOp (v3.0).
     SENAMHI. https://hdl.handle.net/20.500.12542/4183
     doi:10.6084/m9.figshare.32411886
"""
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import box as shapely_box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("piscop_v3")

ROOT     = Path(__file__).parent.parent.parent
SRC_D    = ROOT / "data/raw/pisco/PISCOp_d.nc"
SRC_M    = ROOT / "data/raw/pisco/PISCOp_m.nc"
SRC_CLIM = ROOT / "data/raw/pisco/PISCOp_clim2.nc"
SHP_PATH = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
            / "Cuenca_Chancay___Huaral.shp")
OUT_GRID = ROOT / "data/bronze/B2_pr_basin_grid_v3.nc"
OUT_CSV  = ROOT / "data/bronze/B2_pisco_v3_basin_mean.csv"
OUT_MON  = ROOT / "data/bronze/B2_pisco_v3_monthly.csv"
FIG_DIR  = ROOT / "outputs/figures/basin"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BBOX   = {"lat_min": -11.90, "lat_max": -10.80,
          "lon_min": -77.30, "lon_max": -76.40}
T0     = pd.Timestamp("1981-01-01")
N_DAYS = 16436   # 1981-01-01 → 2025-12-31 (verificado con dim Z1)
N_MON  = 540     # 1981-01 → 2025-12

# Fill value de PISCOp v3.0 (Float32, declarado en atributos del NC)
FILL_V3 = np.float32(-1.1754940241844054e+38)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de grilla
# ══════════════════════════════════════════════════════════════════════════════
def _get_grid_v3(src_nc):
    """Lee lat/lon del NC v3.0. Coordenadas: 'latitude', 'longitude'."""
    ds = nc.Dataset(src_nc)
    lat = np.array(ds["latitude"][:])
    lon = np.array(ds["longitude"][:])
    ds.close()
    return lat, lon


def _bbox_indices(lat_all, lon_all):
    """Índices de la subgrilla dentro del BBOX de la cuenca."""
    li = np.where((lat_all >= BBOX["lat_min"]) & (lat_all <= BBOX["lat_max"]))[0]
    lj = np.where((lon_all >= BBOX["lon_min"]) & (lon_all <= BBOX["lon_max"]))[0]
    return li, lj


# ══════════════════════════════════════════════════════════════════════════════
# Máscara coseno-fracción (igual que v2.1, reutilizable)
# ══════════════════════════════════════════════════════════════════════════════
def build_mask_weights(lat_sub, lon_sub, shp_path):
    HALF   = 0.05
    basin  = gpd.read_file(shp_path).to_crs("EPSG:4326")
    poly   = basin.geometry.union_all()

    frac_map = np.zeros((len(lat_sub), len(lon_sub)), dtype=float)
    for i, la in enumerate(lat_sub):
        for j, lo in enumerate(lon_sub):
            cell  = shapely_box(float(lo)-HALF, float(la)-HALF,
                                float(lo)+HALF, float(la)+HALF)
            inter = poly.intersection(cell)
            frac_map[i, j] = inter.area / cell.area

    mask   = frac_map > 0.0
    cos_w  = np.cos(np.radians(lat_sub.astype(float)))
    weights = cos_w[:, None] * frac_map
    weights = np.where(mask, weights, 0.0)
    w_sum   = weights.sum()
    log.info(f"Mascara: {mask.sum()} pixeles  |  shape {mask.shape}  |  Sw={w_sum:.4f}")
    log.info(f"  100%% dentro: {(frac_map >= 0.999).sum()}  |  "
             f"borde (<50%%): {((frac_map > 0) & (frac_map < 0.5)).sum()}")
    return mask, weights, w_sum, frac_map


# ══════════════════════════════════════════════════════════════════════════════
# Extracción diaria → B2_pr_basin_grid_v3.nc
# ══════════════════════════════════════════════════════════════════════════════
def extract_daily(mask, weights, w_sum, lat_sub, lon_sub, frac_map):
    dates  = pd.date_range(T0, periods=N_DAYS, freq="D")
    t_cf   = np.arange(N_DAYS, dtype="f8")   # "days since 1981-01-01" CF

    log.info(f"Creando {OUT_GRID.name} ({N_DAYS} dias, {len(lat_sub)} lat x {len(lon_sub)} lon)...")
    ds_out = nc.Dataset(OUT_GRID, "w", format="NETCDF4")
    ds_out.createDimension("time", N_DAYS)
    ds_out.createDimension("lat",  len(lat_sub))
    ds_out.createDimension("lon",  len(lon_sub))

    v_t           = ds_out.createVariable("time", "f8", ("time",))
    v_t.units     = "days since 1981-01-01"
    v_t.calendar  = "standard"
    v_t[:]        = t_cf

    v_lat         = ds_out.createVariable("lat", "f4", ("lat",))
    v_lat.units   = "degrees_north"
    v_lat[:]      = lat_sub.astype("f4")

    v_lon         = ds_out.createVariable("lon", "f4", ("lon",))
    v_lon.units   = "degrees_east"
    v_lon[:]      = lon_sub.astype("f4")

    v_mask              = ds_out.createVariable("basin_mask", "i1", ("lat","lon"),
                                                zlib=True, complevel=4)
    v_mask.long_name    = "Basin pixel mask: 1=any intersection, 0=outside"
    v_mask[:]           = mask.astype("i1")

    v_frac              = ds_out.createVariable("basin_frac", "f4", ("lat","lon"),
                                                zlib=True, complevel=4)
    v_frac.long_name    = "Fraction of cell area inside basin polygon [0-1]"
    v_frac[:]           = frac_map.astype("f4")

    CHUNK_T = 30
    v_pr = ds_out.createVariable(
        "pr", "f4", ("time","lat","lon"),
        zlib=True, complevel=4, shuffle=True,
        chunksizes=(CHUNK_T, len(lat_sub), len(lon_sub)),
        fill_value=np.float32(np.nan),
    )
    v_pr.units      = "mm/day"
    v_pr.long_name  = "Daily precipitation — PISCOp v3.0 (Gutierrez & Lavado-Casimiro 2025)"
    v_pr.source     = "SENAMHI — PISCOp v3.0"
    v_pr.resolution = "0.1 degree (~10 km)"

    _now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _proc = (
        "1_bbox_clip(lon=[-77.30,-76.40],lat=[-11.90,-10.80]); "
        "2_fill_to_NaN(fill_value=-1.18e38); "
        "3_clip_to_zero(pr>=0); "
        "4_polygon_intersection_mask(Cuenca_Chancay___Huaral.shp); "
        "5_basin_mean=cos_lat_x_frac_weighted(Trenberth1984)"
    )
    ds_out.title       = "PISCOp v3.0 — Cuenca Chancay-Huaral (0.1 deg)"
    ds_out.institution = "HidroAlerta Chancay-Huaral — Concurso ANA 2026"
    ds_out.source      = (
        "PISCOp v3.0 (SENAMHI)  |  "
        "Archivo: PISCOp_d.nc  |  "
        "doi:10.6084/m9.figshare.32411886"
    )
    ds_out.history     = (
        f"{_now}: scripts/01_bronze/04_extract_piscop_v3.py | "
        f"Fuente: PISCOp_d.nc (PISCOp v3.0, 1981-2025) | "
        f"Operaciones: {_proc}"
    )
    ds_out.processing   = _proc
    ds_out.period_start = "1981-01-01"
    ds_out.period_end   = "2025-12-31"
    ds_out.n_timesteps  = str(N_DAYS)
    ds_out.generated_by = "scripts/01_bronze/04_extract_piscop_v3.py"
    ds_out.decision_ref = "D050 — PISCOp v3.0 supersedes v2.1 + CHIRPS QM (D012)"
    ds_out.references   = (
        "Gutierrez & Lavado-Casimiro (2025) hdl:20.500.12542/4183; "
        "Trenberth (1984) doi:10.1175/1520-0493(1984)112"
    )
    ds_out.Conventions  = "CF-1.8"

    CHUNK_R     = 365
    wmean_list  = []
    mask_flat   = mask.flatten()
    w_basin     = weights.flatten()[mask_flat]

    src = nc.Dataset(SRC_D)
    lat_all = np.array(src["latitude"][:])
    lon_all = np.array(src["longitude"][:])
    li, lj  = _bbox_indices(lat_all, lon_all)

    log.info(f"Leyendo PISCOp_d.nc en chunks de {CHUNK_R} dias...")
    for i in range(0, N_DAYS, CHUNK_R):
        sl  = slice(i, i + CHUNK_R)
        # Dimensión temporal en v3.0 se llama "Z1" (units=unknown).
        # Indexar por posición es equivalente y correcto (D030 pattern).
        raw = np.array(
            src["precipitation"][sl, li[0]:li[-1]+1, lj[0]:lj[-1]+1],
            dtype="f4"
        )
        # Enmascarar fill (−1.18e+38) y píxeles fuera de cuenca
        valid = (raw > FILL_V3 / 2.0) & mask[None, :, :]
        data  = np.where(valid, raw, np.nan).astype("f4")
        data  = np.where(data < 0, 0.0, data)   # pr ≥ 0 por definición física
        ds_out["pr"][sl, :, :] = data

        basin_px = data.reshape(data.shape[0], -1)[:, mask_flat]
        wmean    = (basin_px * w_basin[None, :]).sum(axis=1) / w_sum
        wmean_list.append(pd.Series(wmean, index=dates[sl]))

        pct = min((i + CHUNK_R) / N_DAYS * 100, 100)
        if (i // CHUNK_R) % 5 == 0:
            log.info(f"  {pct:.0f}%%  ({dates[i].date()} ...)")
        ds_out.sync()

    src.close()
    ds_out.close()
    log.info(f"NC guardado: {OUT_GRID}  ({OUT_GRID.stat().st_size/1e6:.1f} MB)")

    pr_mean = pd.concat(wmean_list)
    pr_mean.name       = "pr_mm"
    pr_mean.index.name = "date"
    pr_mean.to_csv(OUT_CSV, header=True)
    log.info(f"CSV guardado: {OUT_CSV}")
    log.info(f"  Media cuenca: {pr_mean.mean():.4f} mm/d = {pr_mean.mean()*365:.1f} mm/año")
    log.info(f"  NaN: {pr_mean.isna().sum()} / {len(pr_mean)}  "
             f"({pr_mean.isna().mean()*100:.2f}%%)")
    return pr_mean


# ══════════════════════════════════════════════════════════════════════════════
# Extracción mensual → B2_pisco_v3_monthly.csv
# ══════════════════════════════════════════════════════════════════════════════
def extract_monthly(mask, weights, w_sum, lat_sub, lon_sub):
    log.info("Extrayendo PISCOp_m.nc (mensual)...")
    dates_m = pd.date_range("1981-01", periods=N_MON, freq="MS")

    src = nc.Dataset(SRC_M)
    lat_all = np.array(src["latitude"][:])
    lon_all = np.array(src["longitude"][:])
    li, lj  = _bbox_indices(lat_all, lon_all)

    mask_flat = mask.flatten()
    w_basin   = weights.flatten()[mask_flat]
    wmean_m   = []

    CHUNK_M = 12
    for i in range(0, N_MON, CHUNK_M):
        sl  = slice(i, i + CHUNK_M)
        raw = np.array(
            src["precipitation"][sl, li[0]:li[-1]+1, lj[0]:lj[-1]+1],
            dtype="f4"
        )
        valid = (raw > FILL_V3 / 2.0) & mask[None, :, :]
        data  = np.where(valid, raw, np.nan).astype("f4")
        data  = np.where(data < 0, 0.0, data)
        bpx   = data.reshape(data.shape[0], -1)[:, mask_flat]
        wm    = (bpx * w_basin[None, :]).sum(axis=1) / w_sum
        wmean_m.append(pd.Series(wm, index=dates_m[sl]))

    src.close()
    df_m          = pd.concat(wmean_m).rename("pr_mm_month")
    df_m.index.name = "month"
    df_m.to_csv(OUT_MON, header=True)
    log.info(f"CSV mensual guardado: {OUT_MON}")
    log.info(f"  Años: 1981-2025  |  Media anual: {df_m.mean()*12:.1f} mm/año")
    return df_m


# ══════════════════════════════════════════════════════════════════════════════
# Extracción climatología → dict {mes: valor_mm}
# ══════════════════════════════════════════════════════════════════════════════
def extract_clim(mask, weights, w_sum, lat_sub, lon_sub):
    log.info("Extrayendo PISCOp_clim2.nc (climatologia mensual 1991-2015)...")
    src = nc.Dataset(SRC_CLIM)
    lat_all = np.array(src["latitude"][:])
    lon_all = np.array(src["longitude"][:])
    li, lj  = _bbox_indices(lat_all, lon_all)

    mask_flat = mask.flatten()
    w_basin   = weights.flatten()[mask_flat]
    clim      = []

    raw = np.array(src["PISCOp_clim2"][:, li[0]:li[-1]+1, lj[0]:lj[-1]+1], dtype="f4")
    src.close()

    for m in range(12):
        row   = raw[m]
        valid = (row > FILL_V3 / 2.0) & mask
        data  = np.where(valid, row, np.nan)
        bpx   = data.flatten()[mask_flat]
        wm    = (bpx * w_basin).sum() / w_sum
        clim.append(wm)

    months = range(1, 13)
    df_clim = pd.Series(clim, index=list(months), name="pr_clim_mm_month")
    df_clim.index.name = "month"
    out_clim = ROOT / "data/bronze/B2_pisco_v3_clim_mensual.csv"
    df_clim.to_csv(out_clim, header=True)
    log.info(f"Climatologia guardada: {out_clim}")
    log.info(f"  Total anual (suma 12 meses): {df_clim.sum():.1f} mm/año")
    for m, v in zip(months, clim):
        log.info(f"    Mes {m:02d}: {v:.1f} mm")
    return df_clim


# ══════════════════════════════════════════════════════════════════════════════
# Mapas exploratorios
# ══════════════════════════════════════════════════════════════════════════════
def _cell_edges(centers):
    h = abs(centers[1] - centers[0]) / 2
    return np.sort(np.concatenate([centers - h, [centers[-1] + h]]))


def _pcm(ax, lon_sub, lat_sub, field, cmap, vmin, vmax):
    import numpy.ma as ma
    LON_E, LAT_E = np.meshgrid(
        np.sort(_cell_edges(lon_sub)), np.sort(_cell_edges(lat_sub))
    )
    fma = ma.masked_invalid(np.where(field == 0, np.nan, field)
                            if vmin >= 0 else ma.masked_invalid(field))
    return ax.pcolormesh(LON_E, LAT_E, fma, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="flat")


def _add_basin(ax):
    try:
        b = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
        b.boundary.plot(ax=ax, color="white",   linewidth=2.0, zorder=5)
        b.boundary.plot(ax=ax, color="#111111", linewidth=1.0,
                        linestyle="--", zorder=6)
    except Exception:
        pass


def make_maps(pr_mean, lat_sub, lon_sub, mask, frac_map):
    log.info("Generando mapas v3...")
    ds    = nc.Dataset(OUT_GRID)
    pr    = np.array(ds["pr"][:])
    ds.close()
    dates = pd.date_range(T0, periods=N_DAYS, freq="D")
    pr_m  = np.where(mask[None, :, :], pr, np.nan)

    # F05v3: Climatología anual
    clim_ann = np.nanmean(pr_m, axis=0) * 365
    vmax_ann = float(np.nanpercentile(clim_ann[mask], 98)) if mask.any() else 1000.0
    fig, ax  = plt.subplots(figsize=(6, 7))
    im = _pcm(ax, lon_sub, lat_sub, clim_ann, "jet", 0, vmax_ann)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("mm / año", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    _add_basin(ax)
    ax.set_title("PISCOp v3.0 — Precipitación media anual (1981–2025)\nCuenca Chancay-Huaral",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitud (°)", fontsize=8); ax.set_ylabel("Latitud (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, alpha=0.4, color="gray", linestyle=":")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "F05v3_pr_climatologia_anual.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  F05v3_pr_climatologia_anual.png")

    # F07v3: Serie anual (media cuenca)
    ann = pr_mean.resample("YE").sum()
    years_arr = ann.index.year.values
    vals_arr  = ann.values.astype(float)
    cmap_j    = plt.colormaps["jet"]
    norm_v    = (vals_arr - vals_arr.min()) / (vals_arr.max() - vals_arr.min() + 1e-9)
    colors    = [cmap_j(v) for v in norm_v]

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.bar(years_arr, vals_arr, color=colors, alpha=0.88,
           edgecolor="white", linewidth=0.4)
    ax.axhline(float(np.mean(vals_arr)), color="k", lw=1.8, ls="--",
               label=f"Media {np.mean(vals_arr):.0f} mm/año")
    nino_yrs = [1982,1983,1987,1991,1992,1997,1998,2002,2004,2009,2015,2016,2017]
    for yr in nino_yrs:
        if yr in years_arr:
            ax.axvspan(yr-0.4, yr+0.4, color="red", alpha=0.18, zorder=0)
    ax.set_xlabel("Año"); ax.set_ylabel("Precipitación (mm/año)")
    ax.set_title("PISCOp v3.0 — Precipitación anual media cuenca Chancay-Huaral (1981–2025)\n"
                 "cos×frac ponderada  |  Rojo = años El Niño", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sort_idx = np.argsort(vals_arr)
    for idx in list(sort_idx[-3:]) + list(sort_idx[:3]):
        ax.text(years_arr[idx], vals_arr[idx]+5, f"{vals_arr[idx]:.0f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "F07v3_pr_serie_anual.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  F07v3_pr_serie_anual.png")

    log.info("Mapas v3 completados.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=== Script 01_bronze/04 — PISCOp v3.0 (D050) ===")

    # Verificar archivos fuente
    for f in (SRC_D, SRC_M, SRC_CLIM):
        if not f.exists():
            raise FileNotFoundError(f"Archivo fuente no encontrado: {f}")

    # Grilla de referencia (diario)
    lat_all, lon_all = _get_grid_v3(SRC_D)
    li, lj           = _bbox_indices(lat_all, lon_all)
    lat_sub          = lat_all[li]
    lon_sub          = lon_all[lj]
    log.info(f"Subgrilla cuenca: {len(lat_sub)} lat x {len(lon_sub)} lon  (0.1 deg)")
    log.info(f"  lat [{lat_sub.min():.2f}, {lat_sub.max():.2f}]  "
             f"lon [{lon_sub.min():.2f}, {lon_sub.max():.2f}]")

    mask, weights, w_sum, frac_map = build_mask_weights(lat_sub, lon_sub, SHP_PATH)

    # 1) Diario → NC + CSV
    pr_mean = extract_daily(mask, weights, w_sum, lat_sub, lon_sub, frac_map)

    # 2) Mensual → CSV (para GR2M y LightGBM Nivel 2)
    extract_monthly(mask, weights, w_sum, lat_sub, lon_sub)

    # 3) Climatología 1991-2015 → CSV (referencia climatológica; ver D056)
    extract_clim(mask, weights, w_sum, lat_sub, lon_sub)

    # 4) Mapas
    make_maps(pr_mean, lat_sub, lon_sub, mask, frac_map)

    log.info("=== DONE ===")
    log.info(f"Outputs principales:")
    log.info(f"  {OUT_GRID}")
    log.info(f"  {OUT_CSV}")
    log.info(f"  {OUT_MON}")


if __name__ == "__main__":
    main()
