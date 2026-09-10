#!/usr/bin/env python3
"""
Script 06: Extrae Tmin (1981-2020) desde PISCOt v1.2 (0.1°).

Metodología
-----------
PISCOt v1.2 (Aybar et al., 2020; Huerta et al., 2022) proporciona temperatura
mínima diaria a 0.01° (~1 km) para el dominio peruano 1981-2020.  Se recorta
al bounding-box de la cuenca Chancay-Huaral y se aplica la máscara de cuenca
para obtener:

  (1) Grilla espacial enmascarada → B2_tmin_basin_grid_v12.nc
  (2) Media ponderada por área:
        T̄ = Σ(wᵢⱼ · Tᵢⱼ) / Σ(wᵢⱼ)
        wᵢⱼ = cos(latᵢ · π/180)   — ponderación coseno-latitud

      La ponderación coseno corrige el hecho de que, en coordenadas esféricas,
      la superficie real de cada celda crece hacia el ecuador:
        ΔA ∝ cos(φ)  (Trenberth, 1984; CLIVAR 2012).

  (3) Columna tmin actualizada en B2_pisco_basin_mean_all.csv

Nota de versiones: PISCOt v1.1 y v1.2 presentan un sesgo documentado de
~-1.37 °C (menor en v1.2 por mayor red de estaciones).  Se extrae la serie
completa 1981-2020 de v1.2 para garantizar homogeneidad temporal.

Referencias
-----------
Aybar et al. (2020) doi:10.1038/s41597-020-00829-y
Huerta et al. (2022) doi:10.1175/JHM-D-21-0151.1
Trenberth (1984) doi:10.1175/1520-0493(1984)112<0326:SAFSWA>2.0.CO;2
"""
import zipfile, logging, shutil, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("piscot12_tmin")

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
ZIP_PATH = ROOT / "data/raw/pisco/Minimum temperature (PISCOt v1.2).zip"
SHP_PATH = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
            / "Cuenca_Chancay___Huaral.shp")
OUT_GRID = ROOT / "data/bronze/B2_tmin_basin_grid_v12.nc"
OUT_MEAN = ROOT / "data/bronze/B2_tmin_basin_mean.csv"
OUT_ALL  = ROOT / "data/bronze/B2_pisco_basin_mean_all.csv"
TMP_DIR  = ROOT / "data/raw/pisco/_tmp_piscot12_tmin"

BBOX  = {"lat_min": -11.90, "lat_max": -10.80,
         "lon_min": -77.30, "lon_max": -76.40}
YEARS = list(range(1981, 2021))


# ── 1. Máscara de cuenca + pesos coseno ───────────────────────────────────────
def build_mask_and_weights(lat_sub: np.ndarray, lon_sub: np.ndarray,
                           shp_path: Path):
    """
    Construye máscara booleana y pesos coseno-latitud para la cuenca.

    La máscara se construye con punto-en-polígono (Point.within) sobre cada
    centroide de celda.  Para grillas de 0.01° el error de borde es <0.01°
    (~1 km) y se considera despreciable frente al área total de la cuenca.

    Parámetros
    ----------
    lat_sub, lon_sub : arrays 1-D con los centros de celda del sub-dominio.

    Retorna
    -------
    mask    : bool   (n_lat, n_lon)  — True = dentro de cuenca
    weights : float  (n_lat, n_lon)  — cos(latᵢ) dentro, 0 fuera
    w_sum   : float  — Σwᵢⱼ (para normalizar la media ponderada)
    """
    basin   = gpd.read_file(shp_path).to_crs("EPSG:4326")
    polygon = basin.geometry.union_all()

    mask = np.zeros((len(lat_sub), len(lon_sub)), dtype=bool)
    for i, la in enumerate(lat_sub):
        for j, lo in enumerate(lon_sub):
            mask[i, j] = polygon.contains(Point(float(lo), float(la)))

    cos_w   = np.cos(np.radians(lat_sub.astype(float)))
    weights = np.where(mask, cos_w[:, None], 0.0)
    w_sum   = weights.sum()

    n_px = int(mask.sum())
    log.info(f"Máscara: {n_px} px en cuenca | bbox {mask.shape[0]}×{mask.shape[1]}")
    log.info(f"Pesos coseno: min={cos_w[mask.any(axis=1)].min():.6f}  "
             f"max={cos_w[mask.any(axis=1)].max():.6f}  Σw={w_sum:.4f}")
    return mask, weights, w_sum


# ── 2. Inicializar NetCDF de salida ───────────────────────────────────────────
def init_nc(out_path: Path, lat_sub: np.ndarray, lon_sub: np.ndarray,
            mask: np.ndarray, n_days_total: int):
    ds = nc.Dataset(out_path, "w", format="NETCDF4")
    ds.createDimension("time", n_days_total)
    ds.createDimension("lat",  len(lat_sub))
    ds.createDimension("lon",  len(lon_sub))

    v_time = ds.createVariable("time", "f8", ("time",))
    v_time.units    = "days since 1981-01-01"
    v_time.calendar = "standard"
    v_time.long_name = "time"

    v_lat = ds.createVariable("lat", "f4", ("lat",))
    v_lat.units    = "degrees_north"; v_lat.long_name = "latitude"
    v_lat[:]       = lat_sub.astype("f4")

    v_lon = ds.createVariable("lon", "f4", ("lon",))
    v_lon.units    = "degrees_east"; v_lon.long_name = "longitude"
    v_lon[:]       = lon_sub.astype("f4")

    v_mask = ds.createVariable("basin_mask", "i1", ("lat", "lon"),
                               zlib=True, complevel=4)
    v_mask.long_name = "Basin pixel mask (1=inside Chancay-Huaral, 0=outside)"
    v_mask[:]        = mask.astype("i1")

    chunk_t = min(30, n_days_total)
    v_tmin  = ds.createVariable(
        "tmin", "f4", ("time", "lat", "lon"),
        zlib=True, complevel=4, shuffle=True,
        chunksizes=(chunk_t, len(lat_sub), len(lon_sub)),
        fill_value=np.float32(np.nan),
    )
    v_tmin.units      = "degC"
    v_tmin.long_name  = "Daily minimum temperature (PISCOt v1.2)"
    v_tmin.source     = "SENAMHI/IGP – PISCOt v1.2"
    v_tmin.resolution = "0.01 degree (~1 km)"
    v_tmin.note       = ("Pixels outside basin boundary set to NaN. "
                         "Use basin_mask variable to identify valid pixels.")

    _proc = (
        "1_bbox_clip(lon=[-77.30,-76.40],lat=[-11.90,-10.80]); "
        "2_polygon_mask(shapefile=Cuenca_Chancay___Huaral.shp,Point.contains); "
        "3_fill_outside_basin_to_NaN; "
        "4_basin_mean=cos_lat_weighted(Trenberth1984,only_basin_px)"
    )
    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ds.title        = "PISCOt v1.2 Tmin — Cuenca Chancay-Huaral (bbox 0.01°)"
    ds.institution  = "HidroAlerta Chancay-Huaral – Concurso ANA 2026"
    ds.source       = (
        "PISCOt_v1.2 (SENAMHI/IGP)  |  "
        "ZIP: Minimum temperature (PISCOt v1.2).zip  |  "
        "Archivos anuales: tmin_daily_{year}.nc  |  "
        "Periodo: 1981-2020"
    )
    ds.history      = (
        f"{_now}: Script 06 (06_extract_tmin_piscot12.py) | "
        f"Fuente: Minimum temperature (PISCOt v1.2).zip | "
        f"Operaciones: {_proc}"
    )
    ds.processing   = _proc
    ds.period_start = "1981-01-01"
    ds.period_end   = "2020-12-31"
    ds.generated_by = "scripts/06_extract_tmin_piscot12.py"
    ds.references   = ("Aybar et al. (2020) doi:10.1038/s41597-020-00829-y; "
                       "Huerta et al. (2022) doi:10.1175/JHM-D-21-0151.1; "
                       "Trenberth (1984) doi:10.1175/1520-0493(1984)112")
    ds.Conventions  = "CF-1.8"
    return ds


# ── 3. Procesar un año ────────────────────────────────────────────────────────
def process_year(year: int, mask: np.ndarray, weights: np.ndarray,
                 w_sum: float, t0: pd.Timestamp) -> tuple:
    fname  = f"tmin_daily_{year}.nc"
    tmp_nc = TMP_DIR / fname
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"  {year}: extrayendo del ZIP...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extract(fname, path=TMP_DIR)

    src     = nc.Dataset(tmp_nc)
    lat_all = np.array(src["latitude"][:])
    lon_all = np.array(src["longitude"][:])
    li = np.where((lat_all >= BBOX["lat_min"]) & (lat_all <= BBOX["lat_max"]))[0]
    lj = np.where((lon_all >= BBOX["lon_min"]) & (lon_all <= BBOX["lon_max"]))[0]

    raw        = np.array(src["tmin"][:, li[0]:li[-1]+1, lj[0]:lj[-1]+1], dtype="f4")
    time_num   = np.array(src["time"][:])
    time_units = src["time"].units
    src.close()
    tmp_nc.unlink()

    times_py  = nc.num2date(time_num, time_units,
                            only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
    times_list = list(times_py)
    dates      = pd.to_datetime([t.strftime("%Y-%m-%d") for t in times_list])
    t_days     = (dates - t0).days.astype("f8")

    # Píxeles fuera de cuenca → NaN
    data = np.where(mask[None, :, :], raw, np.nan).astype("f4")

    # Media ponderada coseno (solo píxeles de cuenca)
    # Se evita el bug IEEE 754: 0 × NaN = NaN usando indexación directa
    mask_flat = mask.flatten()
    basin_px  = data.reshape(len(dates), -1)[:, mask_flat]
    w_basin   = weights.flatten()[mask_flat]
    wmean     = (basin_px * w_basin[None, :]).sum(axis=1) / w_sum

    log.info(f"  {year}: {len(dates)} días, Tmean(cos-w)={wmean.mean():.2f}°C — OK")
    return t_days, data, wmean, dates


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = pd.Timestamp("1981-01-01")

    # Grilla de referencia: extraer un año para obtener lat/lon
    log.info("Leyendo grilla de referencia del ZIP...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extract("tmin_daily_2017.nc", path=TMP_DIR)
    ref = TMP_DIR / "tmin_daily_2017.nc"
    src_ref = nc.Dataset(ref)
    lat_all  = np.array(src_ref["latitude"][:])
    lon_all  = np.array(src_ref["longitude"][:])
    src_ref.close()
    ref.unlink()

    li = np.where((lat_all >= BBOX["lat_min"]) & (lat_all <= BBOX["lat_max"]))[0]
    lj = np.where((lon_all >= BBOX["lon_min"]) & (lon_all <= BBOX["lon_max"]))[0]
    lat_sub = lat_all[li]
    lon_sub = lon_all[lj]
    log.info(f"Sub-grilla: {len(lat_sub)} lat × {len(lon_sub)} lon  (0.01°)")

    mask, weights, w_sum = build_mask_and_weights(lat_sub, lon_sub, SHP_PATH)

    n_days_total = sum(
        366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        for y in YEARS
    )
    log.info(f"Total días a escribir: {n_days_total}")

    log.info(f"Creando {OUT_GRID.name}...")
    ds_out = init_nc(OUT_GRID, lat_sub, lon_sub, mask, n_days_total)

    mean_list = []
    t_cursor  = 0

    for year in YEARS:
        t_days, data, wmean, dates = process_year(year, mask, weights, w_sum, t0)
        n = len(dates)
        ds_out["time"][t_cursor:t_cursor + n] = t_days
        ds_out["tmin"][t_cursor:t_cursor + n, :, :] = data
        ds_out.sync()
        mean_list.append(pd.Series(wmean, index=dates, name="tmin"))
        t_cursor += n

    ds_out.close()
    log.info(f"NC guardado: {OUT_GRID}  "
             f"({OUT_GRID.stat().st_size / 1e6:.1f} MB)")

    full_mean = pd.concat(mean_list)
    full_mean.name = "tmin"
    full_mean.index.name = "date"
    full_mean.to_frame().to_csv(OUT_MEAN)
    log.info(f"CSV guardado: {OUT_MEAN}  ({len(full_mean)} días, "
             f"{full_mean.isna().sum()} NaN)")

    # Actualizar basin_mean_all
    log.info("Actualizando tmin en B2_pisco_basin_mean_all.csv...")
    all_df = pd.read_csv(OUT_ALL, index_col=0, parse_dates=True)
    all_df.index = pd.DatetimeIndex(all_df.index).normalize()
    full_mean.index = pd.DatetimeIndex(full_mean.index).normalize()

    new_idx = full_mean.index[full_mean.index > all_df.index.max()]
    if len(new_idx) > 0:
        ext = pd.DataFrame({c: np.nan for c in all_df.columns}, index=new_idx)
        all_df = pd.concat([all_df, ext])

    all_df.loc[full_mean.index, "tmin"] = full_mean.values
    all_df.to_csv(OUT_ALL)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    log.info("=== DONE ===")
    log.info(f"  Grid NC : {OUT_GRID.name}  "
             f"{lat_sub.shape[0]}lat × {lon_sub.shape[0]}lon × {n_days_total}t")
    log.info(f"  CSV mean: 1981-01-01 → 2020-12-31  ponderación coseno aplicada")
    log.info(f"  tmin media cuenca: {full_mean.mean():.2f}°C")


if __name__ == "__main__":
    main()
