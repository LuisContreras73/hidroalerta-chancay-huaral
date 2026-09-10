#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 08e: Descarga ERA5-Land DIARIO via CDS API — Cuenca Chancay-Huaral.

ESTRATEGIA
----------
El CDS API rechaza peticiones de mas de ~5,000 campos (vars x dias).
Estrategia validada empiricamente: <= 11 variables por anno → 4 grupos.

  INST_1 (11 vars, time=12:00 UTC): vars de temperatura y humedad
  INST_2 (10 vars, time=12:00 UTC): vars de suelo y nieve
  ACCUM_1 ( 8 vars, time=00:00 UTC): precipitacion y escorrentia
  ACCUM_2 ( 7 vars, time=00:00 UTC): radiacion y evaporacion secundaria

  4 grupos x 40 anos = 160 requests x ~6 min = ~16 horas en total.
  Los archivos parciales se guardan en data/raw/era5/daily/parts/ para
  poder reanudar la descarga en cualquier punto.

OUTPUT FINAL
------------
  data/raw/era5/daily/
    era5land_daily_1981_1990.nc   # 36 vars x 3650 dias x 12lat x 14lon
    era5land_daily_1991_2000.nc
    era5land_daily_2001_2010.nc
    era5land_daily_2011_2020.nc

  Total estimado: ~70 MB comprimido (zlib-5)

USO
---
  python scripts/08e_era5land_daily_cds.py                  # todo (160 requests)
  python scripts/08e_era5land_daily_cds.py --year 1981      # solo anno 1981
  python scripts/08e_era5land_daily_cds.py --decade 1981    # solo 1981-1990
  python scripts/08e_era5land_daily_cds.py --group inst1    # solo grupo inst1
  python scripts/08e_era5land_daily_cds.py --diagnose       # estado sin descargar
  python scripts/08e_era5land_daily_cds.py --merge          # solo compilar decadas

FLUJO COMPLETO
--------------
  1. Este script: descarga ~70 MB en 4 archivos decada
  2. Script 08f: extraccion area-ponderada por shapefile → B10_era5land_daily.csv
  3. Script 20b: integrar en D6v2 (swvl1-4, ro, sd, ssr, sp como past_observed)
"""

import argparse
import io
import sys
import warnings
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent

try:
    import cdsapi
except ImportError:
    print("ERROR: pip install cdsapi")
    sys.exit(1)

try:
    import xarray as xr
except ImportError:
    print("ERROR: pip install xarray netCDF4")
    sys.exit(1)

# ── Configuracion ─────────────────────────────────────────────────────────────

AREA = [-10.8, -77.5, -11.9, -76.2]   # [north, west, south, east]

ALL_YEARS  = list(range(1981, 2026))
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS   = [f"{d:02d}" for d in range(1, 32)]

DECADES = {
    1981: list(range(1981, 1991)),
    1991: list(range(1991, 2001)),
    2001: list(range(2001, 2011)),
    2011: list(range(2011, 2021)),
    2021: list(range(2021, 2026)),   # período parcial 2021-2025
}

# Variables instantaneas: snap a las 12:00 UTC (representativo del dia)
# u10/v10 son instantaneas aunque en el docstring original estaban con accum.
VAR_GROUPS = {
    "inst1": {
        "time": "12:00",
        "vars": [
            "2m_temperature",               # t2m
            "2m_dewpoint_temperature",       # d2m
            "skin_temperature",              # skt
            "10m_u_component_of_wind",       # u10
            "10m_v_component_of_wind",       # v10
            "surface_pressure",              # sp
            "volumetric_soil_water_layer_1", # swvl1  0-7cm
            "volumetric_soil_water_layer_2", # swvl2  7-28cm
            "volumetric_soil_water_layer_3", # swvl3  28-100cm
            "volumetric_soil_water_layer_4", # swvl4  100-289cm
            "forecast_albedo",               # fal
        ],
    },
    "inst2": {
        "time": "12:00",
        "vars": [
            "soil_temperature_level_1",          # stl1  0-7cm
            "soil_temperature_level_2",          # stl2  7-28cm
            "soil_temperature_level_3",          # stl3  28-100cm
            "soil_temperature_level_4",          # stl4  100-289cm
            "snow_depth_water_equivalent",       # sd (SWE)
            "snow_cover",                        # snowc
            "snow_albedo",                       # snalb
            "snow_depth",                        # sde
            "leaf_area_index_high_vegetation",   # lai_hv
            "leaf_area_index_low_vegetation",    # lai_lv
        ],
    },
    "accum1": {
        "time": "00:00",
        "vars": [
            "total_precipitation",          # tp
            "total_evaporation",            # e
            "potential_evaporation",        # pev
            "runoff",                       # ro
            "surface_runoff",               # sro
            "sub_surface_runoff",           # ssro
            "snowmelt",                     # smlt
            "snowfall",                     # sf
        ],
    },
    "accum2": {
        "time": "00:00",
        "vars": [
            "surface_net_solar_radiation",                              # ssr
            "surface_net_thermal_radiation",                            # str
            "surface_solar_radiation_downwards",                        # ssrd
            "surface_thermal_radiation_downwards",                      # strd
            "evaporation_from_vegetation_transpiration",                # evavt
            "evaporation_from_bare_soil",                               # evabs
            "evaporation_from_open_water_surfaces_excluding_oceans",    # evaow
        ],
    },
}

ALL_GROUPS  = list(VAR_GROUPS.keys())
TOTAL_VARS  = sum(len(g["vars"]) for g in VAR_GROUPS.values())

RAW_DAILY = ROOT / "data" / "raw" / "era5" / "daily"
PARTS_DIR = RAW_DAILY / "parts"
RAW_DAILY.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _part_path(group: str, year: int) -> Path:
    return PARTS_DIR / f"era5_{group}_{year}.nc"


def _open_download(raw_path: Path) -> "xr.Dataset":
    """
    Abre un archivo descargado del CDS.
    Si es un ZIP (ocurre cuando CDS detecta diferencias estructurales),
    extrae todos los NetCDF del ZIP y los fusiona.
    Si es un NC directo, lo abre normalmente.
    """
    # Detectar si el archivo es realmente un ZIP
    is_zip = False
    try:
        with open(str(raw_path), "rb") as f:
            header = f.read(4)
        if header == b"PK\x03\x04":
            is_zip = True
    except Exception:
        pass

    if is_zip:
        extract_dir = raw_path.parent / f"_unzip_{raw_path.stem}"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(str(raw_path), "r") as zf:
            zf.extractall(str(extract_dir))
        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"No se encontraron .nc en el ZIP: {raw_path}")
        datasets = [xr.open_dataset(str(f)) for f in nc_files]
        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = xr.merge(datasets, join="outer")
            for d in datasets:
                d.close()
        # Limpiar temporales del unzip
        import shutil
        shutil.rmtree(str(extract_dir), ignore_errors=True)
    else:
        ds = xr.open_dataset(str(raw_path))

    return ds


def _normalize_time(ds: "xr.Dataset") -> "xr.Dataset":
    """Normaliza la coordenada de tiempo al dia (sin hora)."""
    tc = "valid_time" if "valid_time" in ds.coords else "time"
    dates = pd.DatetimeIndex(ds[tc].values).normalize()
    ds = ds.assign_coords({tc: dates})
    if tc == "valid_time":
        ds = ds.rename({"valid_time": "time"})
    for c in ["valid_time", "expver", "number"]:
        if c in ds.coords:
            ds = ds.drop_vars(c)
    return ds


# ── Descarga individual ───────────────────────────────────────────────────────

def download_part(group: str, year: int, client: "cdsapi.Client") -> Path:
    """
    Descarga un grupo de variables para un anno especifico.
    Guarda en parts/era5_{group}_{year}.nc.
    Retorna el path del archivo guardado.
    """
    out_path = _part_path(group, year)
    if out_path.exists():
        mb = out_path.stat().st_size / 1024**2
        print(f"    [{group} {year}] ya existe ({mb:.1f} MB), omitiendo.")
        return out_path

    cfg = VAR_GROUPS[group]
    n_vars = len(cfg["vars"])
    n_fields = n_vars * 365  # estimacion (sin bisiestos)
    print(f"    [{group} {year}] {n_vars} vars x ~365 dias = ~{n_fields} campos ...",
          flush=True)
    t0 = datetime.now()

    raw = PARTS_DIR / f"_tmp_{group}_{year}.nc"
    client.retrieve(
        "reanalysis-era5-land",
        {
            "product_type":    "reanalysis",
            "variable":        cfg["vars"],
            "year":            str(year),
            "month":           ALL_MONTHS,
            "day":             ALL_DAYS,
            "time":            cfg["time"],
            "area":            AREA,
            "format":          "netcdf",
            "download_format": "unarchived",
        },
        str(raw),
    )

    # Abrir (con soporte ZIP), cargar en RAM y normalizar tiempo.
    # .load() es critico en Windows: libera el file handle del raw antes de borrarlo.
    ds = _open_download(raw)
    ds = _normalize_time(ds)
    ds.load()
    ds.to_netcdf(str(out_path),
                 encoding={v: {"zlib": True, "complevel": 5}
                           for v in ds.data_vars})
    ds.close()
    del ds
    import gc; gc.collect()
    raw.unlink(missing_ok=True)

    mb   = out_path.stat().st_size / 1024**2
    secs = (datetime.now() - t0).seconds
    print(f"    [{group} {year}] OK — {mb:.1f} MB ({secs}s)", flush=True)
    return out_path


# ── Compilacion por decada ────────────────────────────────────────────────────

def merge_decade(start_year: int, force: bool = False) -> Path:
    """
    Combina todos los grupos y anos de una decada en un solo NetCDF.
    Solo ejecuta si todas las partes estan disponibles.
    """
    end_year = max(DECADES[start_year])
    out_path = RAW_DAILY / f"era5land_daily_{start_year}_{end_year}.nc"
    if out_path.exists() and not force:
        mb = out_path.stat().st_size / 1024**2
        print(f"  [{start_year}-{start_year+9}] ya existe ({mb:.1f} MB)", flush=True)
        return out_path

    years = DECADES[start_year]
    # Verificar que todas las partes existen
    missing = []
    for yr in years:
        for grp in ALL_GROUPS:
            p = _part_path(grp, yr)
            if not p.exists():
                missing.append(f"{grp}_{yr}")
    if missing:
        print(f"  [{start_year}-{start_year+9}] faltan partes: {missing[:5]}{'...' if len(missing)>5 else ''}")
        return None

    print(f"\n  Compilando decada {start_year}-{start_year+9} ...", flush=True)

    # Construir anno a anno, luego concatenar
    yearly = []
    for yr in years:
        parts = []
        for grp in ALL_GROUPS:
            ds = _normalize_time(xr.open_dataset(str(_part_path(grp, yr))))
            parts.append(ds)
        ds_yr = xr.merge(parts, join="outer")
        for p in parts:
            p.close()
        yearly.append(ds_yr)
        print(f"    Anno {yr} — {len(ds_yr.data_vars)} vars, {len(ds_yr.time)} dias",
              flush=True)

    ds_all = xr.concat(yearly, dim="time")
    for ds_yr in yearly:
        ds_yr.close()

    # Ordenar por tiempo
    ds_all = ds_all.sortby("time")

    print(f"  Guardando {out_path.name} ...", flush=True)
    ds_all.to_netcdf(
        str(out_path),
        encoding={v: {"zlib": True, "complevel": 5} for v in ds_all.data_vars},
    )
    ds_all.close()

    mb = out_path.stat().st_size / 1024**2
    print(f"  Guardado: {out_path.name} — {mb:.1f} MB", flush=True)
    return out_path


# ── Diagnostico ──────────────────────────────────────────────────────────────

def diagnostico():
    """Imprime estado de descarga de partes y decadas."""
    print("\n=== ESTADO ERA5-Land Daily ===")

    # Decadas finales
    total_dec_mb = 0
    for start in DECADES:
        end_year = max(DECADES[start])
        p = RAW_DAILY / f"era5land_daily_{start}_{end_year}.nc"
        if p.exists():
            mb = p.stat().st_size / 1024**2
            total_dec_mb += mb
            with xr.open_dataset(str(p)) as ds:
                n_days = len(ds.time)
                n_vars = len(ds.data_vars)
            print(f"  DECADA {p.name:40s} {mb:6.1f} MB  {n_days:4d} dias  {n_vars} vars")
        else:
            print(f"  DECADA {p.name:40s} --- PENDIENTE ---")

    # Partes descargadas
    parts_ok = {grp: 0 for grp in ALL_GROUPS}
    for grp in ALL_GROUPS:
        for yr in ALL_YEARS:
            if _part_path(grp, yr).exists():
                parts_ok[grp] += 1
    total_parts = sum(parts_ok.values())
    total_needed = len(ALL_GROUPS) * len(ALL_YEARS)

    print(f"\n  Partes ({total_parts}/{total_needed}):")
    for grp, cnt in parts_ok.items():
        bar = "█" * cnt + "░" * (len(ALL_YEARS) - cnt)
        print(f"    {grp:8s} [{bar}] {cnt}/{len(ALL_YEARS)} anos")

    print(f"\n  Total decadas: {total_dec_mb:.1f} MB")
    print(f"  Estimado completo: ~70 MB")
    print(f"  Pendiente: {total_needed - total_parts} partes")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_groups = list(VAR_GROUPS.keys())

    parser = argparse.ArgumentParser(description="ERA5-Land daily — 36 vars, 1981-2020")
    parser.add_argument("--year",    type=int, choices=ALL_YEARS, default=None,
                        help="Descargar solo este anno (ej: 1981)")
    parser.add_argument("--decade",  type=int, choices=list(DECADES.keys()), default=None,
                        help="Descargar todos los annos de esta decada (ej: 1981)")
    parser.add_argument("--group",   type=str, choices=all_groups, default=None,
                        help="Solo descargar este grupo de variables")
    parser.add_argument("--merge",   action="store_true",
                        help="Solo compilar decadas (sin descargar)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Solo mostrar estado")
    args = parser.parse_args()

    print("=" * 70)
    print("SCRIPT 08e: ERA5-Land diario — Chancay-Huaral 1981-2025")
    print(f"  {TOTAL_VARS} variables | {len(ALL_GROUPS)} grupos | {len(ALL_YEARS)} annos | grilla 12x14")
    print(f"  Partes en: {PARTS_DIR.relative_to(ROOT)}")
    print("=" * 70)

    # Limpiar archivos _tmp_ cuyo archivo final ya existe (artefactos de runs interrumpidos)
    stale = sorted(PARTS_DIR.glob("_tmp_*.nc"))
    cleaned = 0
    for tmp in stale:
        final_name = tmp.name.replace("_tmp_", "era5_")
        if (PARTS_DIR / final_name).exists():
            tmp.unlink()
            cleaned += 1
    if cleaned:
        print(f"  Limpiados {cleaned} archivos _tmp_ obsoletos.")

    if args.diagnose:
        diagnostico()
        return

    if args.merge:
        print("\nCompilar decadas (--merge):")
        for start in DECADES:
            merge_decade(start)
        diagnostico()
        return

    # Determinar que descargar
    if args.year:
        years_todo = [args.year]
    elif args.decade:
        years_todo = DECADES[args.decade]
    else:
        years_todo = ALL_YEARS

    groups_todo = [args.group] if args.group else all_groups

    n_total = len(years_todo) * len(groups_todo)
    print(f"\nDescargando: {len(years_todo)} annos x {len(groups_todo)} grupos = "
          f"{n_total} requests\n")

    c = cdsapi.Client()
    n_done = 0
    for yr in years_todo:
        for grp in groups_todo:
            n_done += 1
            print(f"\n  [{n_done}/{n_total}] Anno {yr}, grupo {grp}:", flush=True)
            try:
                download_part(grp, yr, c)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                print(f"    Continuando con siguiente...", flush=True)
                continue

        # Intentar compilar decada si todos los annos del grupo estan listos
        for start, yr_list in DECADES.items():
            if yr in yr_list:
                dec_path = RAW_DAILY / f"era5land_daily_{start}_{start+9}.nc"
                if not dec_path.exists():
                    all_ready = all(
                        _part_path(g, y).exists()
                        for g in ALL_GROUPS
                        for y in yr_list
                    )
                    if all_ready:
                        print(f"\n  Todos los annos de {start}-{start+9} listos. Compilando...",
                              flush=True)
                        merge_decade(start)

    print("\n" + "=" * 70)
    diagnostico()
    print("\n  SIGUIENTE: python scripts/08f_era5land_extraction.py")
    print("  (extraccion area-ponderada por sub-cuenca con shapefile)")
    print("=" * 70)


if __name__ == "__main__":
    main()
