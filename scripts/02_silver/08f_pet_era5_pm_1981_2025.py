#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 08f: ERA5-Land basin-mean extraction + ETo PISCOeo_pm-like to 2025.

Methodological background
-------------------------
PISCOeo_pm (SENAMHI official) covers 1981-2016. The hscal product
(B2_pet_basin_mean_hscal.csv) bridges to 2020 using Hargreaves calibrated
against the PM reference. For 2021-2025 only ERA5-Land is available.

Strategy
--------
Phase 1 (1981-2020): Use hscal as-is (already the best estimate; embeds
  PM reference for 1981-2016 and calibrated HS for 2017-2020).
Phase 2 (2021-2025): Apply monthly bias-correction to ERA5 pev
  (ECMWF's internal FAO-56 PM), calibrated against hscal over 1981-2016.

Why pev directly and not recomputed PM?
  ERA5 computes pev with hourly Tmax/Tmin internally; the daily 12-UTC
  snapshot of t2m is not a true Tmean. Using pev + monthly calibration
  is more accurate than reconstructing PM from the daily snapshots.

Calibration window: 1981-2016 (full PISCOeo_pm coverage; same window
  used by script 07b to calibrate the hscal product).

Validation: ERA5 pev vs hscal for 1981-2016 and cross-check 2017-2020.

Outputs
-------
  data/bronze/B10_era5land_daily.csv      — basin-mean ERA5 variables
  data/bronze/B2_pet_era5pm_1981_2025.csv — three-phase ETo [mm/day]
  outputs/08f_era5_pet_validation.png     — validation figure
  outputs/08f_era5_pet_extraction.log
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "outputs" / "08f_era5_pet_extraction.log", "w", "utf-8"),
    ],
)
log = logging.getLogger("era5_08f")

# ── Paths ──────────────────────────────────────────────────────────────────────
ERA5_DAILY_DIR = ROOT / "data" / "raw" / "era5" / "daily"
BASIN_SHP      = (ROOT / "data" / "raw" / "shapefiles" /
                  "cuenca_chancay_huaral" / "limite" /
                  "Cuenca_Chancay___Huaral.shp")
HSCAL_CSV      = ROOT / "data" / "bronze" / "B2_pet_basin_mean_hscal.csv"
OUT_ERA5_CSV   = ROOT / "data" / "bronze" / "B10_era5land_daily.csv"
OUT_PET_CSV    = ROOT / "data" / "bronze" / "B2_pet_era5pm_1981_2025.csv"
OUT_FIG        = ROOT / "outputs" / "08f_era5_pet_validation.png"

CAL_END_YEAR  = 2016   # last year with PISCOeo_pm (same as script 07b)

DECADES = [
    (1981, 1990),
    (1991, 2000),
    (2001, 2010),
    (2011, 2020),
    (2021, 2025),
]

# Variables to keep in B10 (subset of ERA5 36-var set)
B10_VARS = [
    "t2m",    # 2-m temperature [K] → °C in output
    "d2m",    # 2-m dewpoint [K] → °C
    "u10",    # 10-m U-wind [m/s]
    "v10",    # 10-m V-wind [m/s]
    "sp",     # surface pressure [Pa]
    "pev",    # potential evaporation [m] → mm/day (sign-flip)
    "ssr",    # net solar radiation [J/m²] → MJ/m²/day
    "str",    # net thermal radiation [J/m²] → MJ/m²/day
    "ssrd",   # solar radiation downward [J/m²] → MJ/m²/day
    "tp",     # total precipitation [m] → mm
    "e",      # total evaporation [m] → mm
    "ro",     # runoff [m] → mm
    "swvl1",  # soil water layer 1 [m³/m³]
    "swvl2",  # soil water layer 2 [m³/m³]
    "swvl3",  # soil water layer 3 [m³/m³]
    "swvl4",  # soil water layer 4 [m³/m³]
    "sd",     # snow water equivalent [m]
]


# ── Basin mask ────────────────────────────────────────────────────────────────

def build_basin_mask(lats, lons, shp_path):
    """
    Returns a 2-D boolean array (lat x lon) with True for ERA5 pixels whose
    centre falls within the basin shapefile polygon.
    Uses geopandas Point-in-polygon on the shapefile CRS (expects WGS84).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    gdf  = gpd.read_file(str(shp_path))
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    basin = gdf.union_all()   # merge all features if multi-part

    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            mask[i, j] = basin.contains(Point(lo, la))
    n_pix = mask.sum()
    log.info(f"   Basin mask: {n_pix} / {mask.size} pixels inside polygon")
    return mask


# ── Unit-convert a basin-mean dict (raw ERA5 → physical units) ────────────────

def _convert_units(raw: dict) -> dict:
    """Convert raw ERA5 daily basin-mean values to physical units."""
    out = {}
    # Temperature K→°C
    for v in ("t2m", "d2m"):
        if v in raw:
            out[v] = raw[v] - 273.15

    # Wind speed (10 m) m/s — keep as-is; FAO-56 height correction in PET step
    for v in ("u10", "v10"):
        if v in raw:
            out[v] = raw[v]

    # Derived: wind speed at 2m from 10-m components (FAO-56 eq. 47)
    if "u10" in raw and "v10" in raw:
        ws10 = np.sqrt(raw["u10"] ** 2 + raw["v10"] ** 2)
        out["ws2m"] = ws10 * (4.87 / np.log(67.8 * 10 - 5.42))

    # Surface pressure Pa — keep as-is
    if "sp" in raw:
        out["sp"] = raw["sp"]

    # Radiation J/m² → MJ/m²/day
    for v in ("ssr", "str", "ssrd"):
        if v in raw:
            out[v] = raw[v] / 1e6

    # Net radiation MJ/m²/day = ssr + str
    if "ssr" in raw and "str" in raw:
        out["Rn"] = (raw["ssr"] + raw["str"]) / 1e6

    # pev [m, negative] → [mm/day, positive]
    if "pev" in raw:
        out["pev"] = -raw["pev"] * 1000.0

    # Precipitation + evaporation + runoff m→mm
    for v in ("tp", "e", "ro"):
        if v in raw:
            out[v] = raw[v] * 1000.0

    # Soil water layer volumetric [m³/m³] — keep as-is
    for v in ("swvl1", "swvl2", "swvl3", "swvl4"):
        if v in raw:
            out[v] = raw[v]

    # Snow water equivalent m→mm
    if "sd" in raw:
        out["sd"] = raw["sd"] * 1000.0

    return out


# ── Extract basin-mean time series from ERA5 decade files ─────────────────────

def extract_era5_basin_mean(mask, lat_weights) -> pd.DataFrame:
    """
    Open each decade NC file sequentially, apply mask, area-weight mean.
    Returns a DataFrame with one row per day, columns = converted ERA5 vars.
    """
    try:
        import xarray as xr
    except ImportError:
        log.error("xarray not installed. pip install xarray netCDF4")
        sys.exit(1)

    frames = []
    for start, end in DECADES:
        nc_path = ERA5_DAILY_DIR / f"era5land_daily_{start}_{end}.nc"
        if not nc_path.exists():
            log.warning(f"   Missing decade file: {nc_path.name} — skipping")
            continue

        log.info(f"   Opening {nc_path.name} ...")
        ds = xr.open_dataset(str(nc_path))

        # Fix longitude wraparound: keep only negative lons (≤ 0)
        # The download bounding box produces duplicate lons as both negative
        # and 360-based equivalents. Select the negative set.
        lons_all = ds.longitude.values
        lon_sel  = lons_all[lons_all <= 0]
        ds = ds.sel(longitude=lon_sel)

        # Subset to ERA5 vars we want
        keep = [v for v in B10_VARS if v in ds.data_vars]
        ds = ds[keep]

        lats = ds.latitude.values    # already filtered by sel above
        lons = ds.longitude.values
        T    = len(ds.time)

        # weights: cos(lat) broadcast to (lat, lon) where mask=True
        w2d = np.outer(lat_weights, np.ones(len(lons)))  # lat x lon
        w2d = np.where(mask, w2d, 0.0)
        w_sum = w2d.sum()
        if w_sum == 0:
            log.error("   Basin mask sum is 0 — check shapefile CRS")
            sys.exit(1)

        # Compute basin-mean for each variable and time step.
        # NaN × 0 = NaN (IEEE 754) so we must use nansum and normalize by
        # the sum of weights for *valid* pixels only (handles ocean fill values).
        rows = []
        time_values = pd.DatetimeIndex(ds.time.values).normalize()
        for v in keep:
            da = ds[v].values  # (time, lat, lon), may contain NaN fill values
            # Weighted numerator (nansum treats NaN contribution as 0)
            basin_sum = np.nansum(da * w2d, axis=(1, 2))
            # Denominator: sum of weights only where data is valid
            valid_w   = np.where(np.isfinite(da), w2d, 0.0)  # (time, lat, lon)
            valid_w_sum = valid_w.sum(axis=(1, 2))            # (time,)
            basin_mean = np.where(valid_w_sum > 0,
                                  basin_sum / valid_w_sum, np.nan)
            rows.append((v, basin_mean))

        raw_dict = {v: arr for v, arr in rows}
        converted = _convert_units(raw_dict)

        df = pd.DataFrame(converted, index=time_values)
        df.index.name = "date"
        log.info(f"   {nc_path.name}: {T} days, {len(keep)} vars extracted")
        frames.append(df)
        ds.close()

    if not frames:
        log.error("No ERA5 decade files found!")
        sys.exit(1)

    df_all = pd.concat(frames, axis=0).sort_index()
    # Remove any duplicate dates (unlikely but safe)
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    log.info(f"   Total: {len(df_all)} days  "
             f"{df_all.index[0].date()} → {df_all.index[-1].date()}")
    return df_all


# ── Monthly bias-correction: ERA5 pev vs hscal ───────────────────────────────

def monthly_bias_correction(era5_pev: pd.Series, hscal: pd.Series,
                             cal_end: int) -> dict:
    """
    Compute monthly correction factors k_m = mean(hscal_m) / mean(pev_m)
    on the calibration window 1981-cal_end.
    Returns dict {1..12: k}.
    """
    common = hscal.index.intersection(era5_pev.index)
    cal    = common[common.year <= cal_end]

    k = {}
    for m in range(1, 13):
        idx_m    = cal[cal.month == m]
        sum_hscal = hscal.loc[idx_m].sum()
        sum_era5  = era5_pev.loc[idx_m].sum()
        k[m] = float(sum_hscal / sum_era5) if sum_era5 > 1e-6 else 1.0

    log.info("   k_m (hscal/ERA5pev calibration):")
    log.info("   " + "  ".join(f"M{m}={k[m]:.3f}" for m in range(1, 13)))
    return k


def apply_monthly_correction(series: pd.Series, k: dict) -> pd.Series:
    out = series.copy()
    for m in range(1, 13):
        out[out.index.month == m] *= k[m]
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def _kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return np.nan
    r     = float(np.corrcoef(o, s)[0, 1])
    alpha = s.std() / (o.std() + 1e-9)
    beta  = s.mean() / (o.mean() + 1e-9)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def _pbias(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    return float((s.sum() - o.sum()) / (o.sum() + 1e-9) * 100)


def _nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    return float(1 - np.sum((o - s)**2) / (np.sum((o - o.mean())**2) + 1e-9))


def report_metrics(label, obs: pd.Series, sim: pd.Series):
    common = obs.index.intersection(sim.index)
    o = obs.loc[common].values.astype(float)
    s = sim.loc[common].values.astype(float)
    log.info(f"   {label}: n={len(common)}"
             f"  NSE={_nse(o,s):.3f}"
             f"  KGE={_kge(o,s):.3f}"
             f"  PBIAS={_pbias(o,s):+.1f}%"
             f"  obs_mean={np.nanmean(o):.3f}"
             f"  sim_mean={np.nanmean(s):.3f} mm/d")


# ── Validation figure ─────────────────────────────────────────────────────────

def make_figure(hscal: pd.Series, pev_raw: pd.Series, pev_cal: pd.Series,
                pet_final: pd.Series):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("matplotlib not available — skipping figure")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # Panel 1: Monthly mean climatology
    ax = axes[0]
    months = range(1, 13)
    clim_hscal = [hscal[hscal.index.month == m].mean() for m in months]
    clim_raw   = [pev_raw[pev_raw.index.month == m].mean() for m in months]
    clim_cal   = [pev_cal[pev_cal.index.month == m].mean() for m in months]
    ax.plot(months, clim_hscal, "b-o", label="hscal (reference 1981-2020)", lw=2)
    ax.plot(months, clim_raw,   "r--s", label="ERA5 pev (raw)", lw=1.5)
    ax.plot(months, clim_cal,   "g-^", label="ERA5 pev (bias-corrected)", lw=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("ETo [mm/day]")
    ax.set_title("Monthly climatology (1981-2016 calibration window)")
    ax.legend(fontsize=8)
    ax.set_xticks(list(months))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.grid(True, alpha=0.3)

    # Panel 2: Annual mean time series
    ax = axes[1]
    ann_hscal = hscal.resample("YE").mean()
    ann_raw   = pev_raw.resample("YE").mean()
    ann_cal   = pev_cal.resample("YE").mean()
    ann_final = pet_final.resample("YE").mean()
    ax.plot(ann_hscal.index.year, ann_hscal.values, "b-o", label="hscal", lw=2)
    ax.plot(ann_raw.index.year,   ann_raw.values,   "r--", label="ERA5 pev raw", lw=1.2)
    ax.plot(ann_cal.index.year,   ann_cal.values,   "g--", label="ERA5 pev cal", lw=1.2)
    ax.plot(ann_final.index.year, ann_final.values, "k-",  label="Final product", lw=2)
    ax.axvline(2020.5, color="gray", ls=":", lw=1, label="hscal end / ERA5 ext start")
    ax.axvline(2016.5, color="orange", ls=":", lw=1, label="Calibration end")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean ETo [mm/day]")
    ax.set_title("Annual mean ETo — three-phase product")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Scatter ERA5 pev cal vs hscal (monthly aggregated, 1981-2016)
    ax = axes[2]
    common = hscal.index.intersection(pev_cal.index)
    cal_idx = common[common.year <= CAL_END_YEAR]
    hscal_m = hscal.loc[cal_idx].resample("ME").mean()
    pcal_m  = pev_cal.loc[cal_idx].resample("ME").mean()
    common_m = hscal_m.index.intersection(pcal_m.index)
    x = hscal_m.loc[common_m].values
    y = pcal_m.loc[common_m].values
    ax.scatter(x, y, s=15, alpha=0.5, c="green")
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_xlabel("hscal ETo [mm/day] (monthly mean)")
    ax.set_ylabel("ERA5 pev calibrated [mm/day]")
    ax.set_title(f"ERA5 pev (cal) vs hscal — monthly (1981-2016)  r={r:.3f}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_FIG), dpi=120)
    plt.close(fig)
    log.info(f"   Figura guardada: {OUT_FIG.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("SCRIPT 08f: ERA5-Land basin-mean extraction + ETo 1981-2025")
    log.info("=" * 70)

    # ── 1. Build basin mask ────────────────────────────────────────────────────
    log.info("[1] Building ERA5 basin mask from shapefile ...")
    if not BASIN_SHP.exists():
        log.error(f"Shapefile not found: {BASIN_SHP}")
        sys.exit(1)

    # Use the first decade file to get the grid coordinates
    nc0 = ERA5_DAILY_DIR / "era5land_daily_1981_1990.nc"
    if not nc0.exists():
        log.error(f"First decade file missing: {nc0}")
        sys.exit(1)

    try:
        import xarray as xr
    except ImportError:
        log.error("xarray not installed. pip install xarray netCDF4")
        sys.exit(1)

    with xr.open_dataset(str(nc0)) as ds0:
        lons_all = ds0.longitude.values
        lon_sel  = lons_all[lons_all <= 0]     # keep negative lons only
        lats     = ds0.latitude.values
        lons     = lon_sel

    log.info(f"   ERA5 grid: {len(lats)} lat x {len(lons)} lon")
    log.info(f"   Lat: {lats[0]:.1f} → {lats[-1]:.1f}")
    log.info(f"   Lon: {lons[0]:.1f} → {lons[-1]:.1f}")

    mask        = build_basin_mask(lats, lons, BASIN_SHP)
    lat_weights = np.cos(np.radians(lats))

    # ── 2. Extract basin-mean ERA5 time series ─────────────────────────────────
    log.info("[2] Extracting basin-mean ERA5 variables (1981-2025) ...")
    df_era5 = extract_era5_basin_mean(mask, lat_weights)

    # Save B10
    df_era5.to_csv(str(OUT_ERA5_CSV), float_format="%.6f")
    log.info(f"   Saved: {OUT_ERA5_CSV.name}  shape={df_era5.shape}")

    # Quick diagnostics
    log.info(f"   t2m  mean: {df_era5['t2m'].mean():.2f} °C")
    log.info(f"   pev  mean: {df_era5['pev'].mean():.3f} mm/day")
    log.info(f"   Rn   mean: {df_era5['Rn'].mean():.3f} MJ/m²/day")
    log.info(f"   ws2m mean: {df_era5['ws2m'].mean():.3f} m/s")

    # ── 3. Load hscal reference ────────────────────────────────────────────────
    log.info("[3] Loading hscal reference (1981-2020) ...")
    if not HSCAL_CSV.exists():
        log.error(f"hscal not found: {HSCAL_CSV}")
        sys.exit(1)

    hscal_df = pd.read_csv(str(HSCAL_CSV), index_col=0, parse_dates=True)
    hscal_df.index = pd.DatetimeIndex(hscal_df.index).normalize()
    hscal = hscal_df["pet"].rename("hscal")
    log.info(f"   hscal: {hscal.index[0].date()} → {hscal.index[-1].date()}"
             f"  mean={hscal.mean():.3f} mm/day")

    # ── 4. ERA5 pev raw series ─────────────────────────────────────────────────
    pev_raw = df_era5["pev"].rename("pev_raw")
    log.info(f"   ERA5 pev raw: mean={pev_raw.mean():.3f} mm/day")

    # ── 5. Validate ERA5 pev raw vs hscal (1981-2016) ─────────────────────────
    log.info("[4] Validating ERA5 pev raw vs hscal (1981-2016) ...")
    hscal_cal = hscal[hscal.index.year <= CAL_END_YEAR]
    report_metrics("ERA5 pev raw vs hscal (1981-2016)", hscal_cal, pev_raw)

    # ── 6. Monthly bias correction ─────────────────────────────────────────────
    log.info("[5] Calibrating monthly bias-correction factors ...")
    k_monthly = monthly_bias_correction(pev_raw, hscal, CAL_END_YEAR)

    pev_cal = apply_monthly_correction(pev_raw, k_monthly).rename("pev_cal")

    log.info("[6] Validating ERA5 pev calibrated vs hscal ...")
    report_metrics("ERA5 pev cal vs hscal (1981-2016)", hscal_cal, pev_cal)
    report_metrics("ERA5 pev cal vs hscal (2017-2020)",
                   hscal[hscal.index.year > CAL_END_YEAR], pev_cal)

    # ── 7. Three-phase final ETo ───────────────────────────────────────────────
    log.info("[7] Assembling three-phase ETo (hscal 1981-2020 + ERA5 2021-2025) ...")

    full_idx  = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    pet_final = hscal.reindex(full_idx)            # 1981-2020: hscal as-is

    ext_mask  = pet_final.isna()                   # 2021-2025 gap
    pet_final.loc[ext_mask] = pev_cal.reindex(full_idx).loc[ext_mask]

    n_hscal = (~ext_mask).sum()
    n_era5  = ext_mask.sum()
    log.info(f"   hscal portion  : {n_hscal} days  (1981-2020)")
    log.info(f"   ERA5 cal portion: {n_era5} days  (2021-2025)")
    log.info(f"   Final mean: {pet_final.mean():.3f} mm/day")
    log.info(f"   NaN check: {pet_final.isna().sum()} missing values")

    # ── 8. Save final PET ──────────────────────────────────────────────────────
    log.info("[8] Saving outputs ...")
    pet_df = pd.DataFrame({
        "pet":     pet_final,
        "pev_era5_raw": pev_raw.reindex(full_idx),
        "pev_era5_cal": pev_cal.reindex(full_idx),
        "hscal":        hscal.reindex(full_idx),
        "source":       np.where(full_idx.year <= 2020, "hscal", "era5_cal"),
    }, index=full_idx)
    pet_df.index.name = "date"
    pet_df.to_csv(str(OUT_PET_CSV), float_format="%.6f")
    log.info(f"   Saved: {OUT_PET_CSV.name}  shape={pet_df.shape}")

    # ── 9. Monthly summary ─────────────────────────────────────────────────────
    log.info("[9] Monthly summary — calibration factors and validation:")
    ext_years = [2021, 2022, 2023, 2024, 2025]
    for m in range(1, 13):
        era5_m   = pev_cal[pev_cal.index.month == m]
        era5_ext = era5_m[era5_m.index.year.isin(ext_years)]
        log.info(f"   M{m:02d}: k={k_monthly[m]:.3f}  "
                 f"ERA5cal_mean={era5_m.mean():.3f}  "
                 f"ERA5ext_mean={era5_ext.mean() if len(era5_ext) else float('nan'):.3f} mm/d")

    # ── 10. Figure ────────────────────────────────────────────────────────────
    log.info("[10] Generating validation figure ...")
    make_figure(hscal, pev_raw, pev_cal, pet_final.rename("pet_final"))

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("RESUMEN 08f")
    log.info(f"  B10_era5land_daily.csv    : {len(df_era5)} dias, "
             f"{len(df_era5.columns)} vars")
    log.info(f"  B2_pet_era5pm_1981_2025.csv: "
             f"1981-01-01 → 2025-12-31  NaN={pet_final.isna().sum()}")
    log.info(f"  Calibracion 1981-{CAL_END_YEAR}: "
             + " ".join(f"M{m}={k_monthly[m]:.3f}" for m in range(1, 13)))
    log.info(f"  Estrategia: hscal (1981-2020) + ERA5pev_cal (2021-2025)")
    log.info(f"  SIGUIENTE: actualizar 09_calibracion_gr4j.py con nuevo PET")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
