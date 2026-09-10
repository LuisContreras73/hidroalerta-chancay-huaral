#!/usr/bin/env python3
"""
Script 51: Build D7 — sub-basin satellite features from GEE TIFs.

Reads all GEE rasters (CHIRPS, MODIS, SMAP, JRC, Landsat, S1, S2, ESA),
applies fill-value masks (D040-D043), computes zonal statistics per sub-basin,
and merges with D6 to produce D7_multientity.csv (1981-2025, 9 entities).

QA fill masks (DECISIONS.md D040-D046):
  CHIRPS     : fill = -9999  → NaN (D040)
  MOD10A1    : snow_cover > 100 → NaN (fill/flags), < 0 → NaN (D041, D046)
  MOD13Q1    : × 0.0001  (raw int16 scale)
  MOD09GA    : NO scale (ratio already [-1,1]), 99.9% NaN dry season → forward-fill (D042,D045)
  MOD11A1    : fill = 0 → NaN, × 0.02  (Kelvin), cloud gaps forward-filled (D043)
  MOD16A2    : fill >= 32700 → NaN (32761=ocean, 32767=fill), × 0.1 (mm/8días) (D043,D045)
  JRC        : ÷ int16_scale_factor (= 100 from sidecar JSON)
  Landsat/S2/S1: normalizedDifference/dB already in correct units, NO scale (D045)

Performance: reads each TIF once (all bands), applies sub-basin masks vectorised.

Usage:
    python scripts/51_build_satellite_features.py --source chirps --year_start 2015 --year_end 2025
    python scripts/51_build_satellite_features.py --merge_d6   # all sources → D7
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message=".*TIFFReadDirectory.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice.*")

ROOT      = Path(__file__).parent.parent.parent   # scripts/08_gee → scripts → project root
GEE_DIR   = ROOT / "data/raw/gee"
OUT_DIR   = ROOT / "data/model_ready"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

SHP_SUB = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas"
           / "Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_geogpsperu_UH_MENORES.shp")
D6_PATH = ROOT / "data/model_ready/D6_multientity.csv"
D7_PATH = ROOT / "data/model_ready/D7_multientity.csv"

ENTITIES = ["sub_634", "sub_640", "sub_641", "sub_646",
            "sub_649", "sub_650", "sub_653", "sub_655", "sub_656"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "51_build_satellite_features.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("sat_features")
logging.getLogger("rasterio").setLevel(logging.ERROR)  # suppress GDAL CPLE_AppDefined noise


# ─── Sub-basin geometries ──────────────────────────────────────────────────────

def load_subbasins() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(SHP_SUB)).to_crs(epsg=4326)
    gdf["entity_id"] = "sub_" + gdf["ID"].astype(str)
    gdf = gdf[gdf["entity_id"].isin(ENTITIES)].set_index("entity_id")
    return gdf[["geometry"]]


# ─── Fill-value masks (D040-D043) ─────────────────────────────────────────────

def _mask_chirps(arr):
    return np.where(arr == -9999, np.nan, arr.astype(float))

def _mask_snow(arr):
    a = arr.astype(float)
    a[a > 100] = np.nan  # MODIS fill/flag codes (200-255)
    a[a < 0] = np.nan    # -9999 fill not set in TIF nodata metadata
    return a

def _scale_01(arr):
    return arr.astype(float) * 0.0001

def _mask_lst(arr):
    a = arr.astype(float); a[a == 0] = np.nan; return a * 0.02

def _mask_et(arr):
    a = arr.astype(float)
    a[a >= 32700] = np.nan  # MOD16A2 fills: 32700-32767 (ocean=32761, fill=32767)
    return a * 0.1

def _mask_jrc(arr, sf=100.0):
    return arr.astype(float) / sf


# ─── Vectorised zonal statistics ──────────────────────────────────────────────

class TifZonalExtractor:
    """
    Opens a TIF once, builds per-entity pixel masks, then rapidly
    computes nanmean (and optionally nanstd) per band for each sub-basin.

    bands_arr shape: (n_bands, height, width)
    Result: dict  entity_id → np.ndarray shape (n_bands,)
    """

    def __init__(self, tif: Path, gdf: gpd.GeoDataFrame,
                 preprocess=None, bands: list[int] | None = None):
        """
        tif       : path to GeoTIFF
        gdf       : GeoDataFrame with entity geometries (CRS may differ from TIF)
        preprocess: callable applied to full (n_bands, H, W) float array
        bands     : 1-indexed list of bands to read (None = all)
        """
        with rasterio.open(tif) as src:
            self.transform = src.transform
            self.crs       = src.crs
            self.shape     = (src.height, src.width)
            if bands is None:
                raw = src.read().astype(float)
            else:
                raw = src.read(bands).astype(float)
            nd = src.nodata
            if nd is not None:
                raw[raw == nd] = np.nan

        if preprocess is not None:
            self.arr = preprocess(raw)
        else:
            self.arr = raw
        # arr shape: (n_bands, H, W)

        # Build pixel masks per entity
        gdf_proj = gdf.to_crs(self.crs) if gdf.crs != self.crs else gdf
        self.masks = {}
        for eid, row in gdf_proj.iterrows():
            geom = row.geometry
            try:
                msk = geometry_mask(
                    [geom], out_shape=self.shape,
                    transform=self.transform, invert=True,
                )
                self.masks[eid] = msk
            except Exception:
                self.masks[eid] = np.zeros(self.shape, dtype=bool)

    def zonal_mean(self) -> dict[str, np.ndarray]:
        """Returns {entity_id: ndarray(n_bands)} of nanmean per band."""
        result = {}
        for eid, msk in self.masks.items():
            if not msk.any():
                result[eid] = np.full(self.arr.shape[0], np.nan)
                continue
            pixels = self.arr[:, msk]  # (n_bands, n_pixels)
            result[eid] = np.nanmean(pixels, axis=1)
        return result

    def zonal_nodata_frac(self) -> dict[str, np.ndarray]:
        """Returns {entity_id: ndarray(n_bands)} of NaN fraction per band."""
        result = {}
        for eid, msk in self.masks.items():
            if not msk.any():
                result[eid] = np.ones(self.arr.shape[0])
                continue
            pixels = self.arr[:, msk]
            result[eid] = np.isnan(pixels).mean(axis=1)
        return result


def _tif_rows_from_extractor(
    tif: Path, gdf: gpd.GeoDataFrame, dates: list[pd.Timestamp],
    col_names: list[str], preprocess=None,
    band_groups: list[list[int]] | None = None,
) -> list[dict]:
    """
    Core helper: read `tif`, compute zonal mean per entity, return list of row dicts.

    dates       : list of pd.Timestamp, length = n_composites
    col_names   : list of column names, length = bands_per_composite
    band_groups : if None, each composite = consecutive single bands matching col_names.
                  Otherwise list of [list of 1-indexed band indices] per composite.
    """
    if not (tif.exists() and tif.stat().st_size > 500):
        return []

    ext = TifZonalExtractor(tif, gdf, preprocess=preprocess)
    means = ext.zonal_mean()
    n_bands = ext.arr.shape[0]
    n_cols  = len(col_names)

    rows = []
    for i, dt in enumerate(dates):
        if band_groups is not None:
            # Custom band grouping
            bg = band_groups[i]
            for j, (col, b1) in enumerate(zip(col_names, bg)):
                b0 = b1 - 1  # to 0-indexed
                for eid in ENTITIES:
                    val = means[eid][b0] if b0 < n_bands else np.nan
                    row = {"date": dt.date(), "entity_id": eid, col: val}
                    rows.append(row)
        else:
            # Sequential: composite i uses bands i*n_cols .. i*n_cols + n_cols-1
            for j, col in enumerate(col_names):
                b0 = i * n_cols + j  # 0-indexed
                if b0 >= n_bands:
                    break
                for eid in ENTITIES:
                    val = means[eid][b0]
                    row = rows[-len(ENTITIES) + (ENTITIES.index(eid) if j > 0 else -len(ENTITIES))] \
                          if j > 0 else None
                    # Build rows: one row per (date, entity) with all col_names filled
                    pass

            # Cleaner: build (date, entity) rows with all bands for this composite
            for eid in ENTITIES:
                row = {"date": dt.date(), "entity_id": eid}
                for j, col in enumerate(col_names):
                    b0 = i * n_cols + j
                    row[col] = means[eid][b0] if b0 < n_bands else np.nan
                rows.append(row)

    return rows


# ─── Vectorised helper ────────────────────────────────────────────────────────

def _extract_tif(
    tif: Path, gdf: gpd.GeoDataFrame,
    dates: list[pd.Timestamp], col_names: list[str],
    preprocess=None,
) -> list[dict]:
    """
    Vectorised: read tif once, compute sub-basin means for all bands, map to rows.

    Assumes bands are ordered: for each composite i and variable j,
      band index = i * len(col_names) + j   (0-indexed)
    """
    if not (tif.exists() and tif.stat().st_size > 500):
        return []

    ext   = TifZonalExtractor(tif, gdf, preprocess=preprocess)
    means = ext.zonal_mean()
    n_bands = ext.arr.shape[0]
    n_cols  = len(col_names)
    rows = []

    for i, dt in enumerate(dates):
        for eid in ENTITIES:
            row = {"date": dt.date(), "entity_id": eid}
            for j, col in enumerate(col_names):
                b0 = i * n_cols + j
                row[col] = means[eid][b0] if b0 < n_bands else np.nan
            rows.append(row)

    return rows


# ─── Source extractors ────────────────────────────────────────────────────────

def _pick(folder: Path, yr: int | None, patterns: list[str]) -> Path | None:
    for pat in patterns:
        p = folder / (pat.format(yr=yr) if yr else pat)
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def extract_chirps(gdf, y0, y1):
    log.info("Extracting CHIRPS...")
    d = GEE_DIR / "chirps"; rows = []
    for yr in range(y0, y1 + 1):
        tif = _pick(d, yr, ["chirps_precip_{yr}_5566m_clip.tif",
                             "chirps_precip_{yr}_5566m.tif"])
        if tif is None: continue
        with rasterio.open(tif) as src: n = src.count
        dates = pd.date_range(f"{yr}-01-01", periods=n, freq="D").tolist()
        rows += _extract_tif(tif, gdf, dates, ["chirps_pr_mm"], _mask_chirps)
        log.info(f"  CHIRPS {yr}: {n} bands")
    return _to_df(rows, ["chirps_pr_mm"])


def extract_snow(gdf, y0, y1):
    """MOD10A1 snow cover — bimestral 6 chunks/year (same structure as LSWI), daily."""
    log.info("Extracting MOD10A1 snow...")
    d = GEE_DIR / "snow"; rows = []
    BIMESTRES = [("m0102", 1), ("m0304", 3), ("m0506", 5),
                 ("m0708", 7), ("m0910", 9), ("m1112", 11)]
    for yr in range(max(y0, 2000), y1 + 1):
        for btag, bmo in BIMESTRES:
            tif = _pick(d, yr, [f"mod10a1_snow_{{yr}}_{btag}_500m_clip.tif",
                                  f"mod10a1_snow_{{yr}}_{btag}_500m.tif"])
            if tif is None: continue
            with rasterio.open(tif) as src: n = src.count
            start = pd.Timestamp(f"{yr}-{bmo:02d}-01")
            dates = pd.date_range(start, periods=n, freq="D").tolist()
            rows += _extract_tif(tif, gdf, dates, ["snow_cover_pct"], _mask_snow)
    log.info(f"  Snow: {len(rows)} rows extracted (before dedup)")
    df = _to_df(rows, ["snow_cover_pct"])
    if df.empty: return df
    # Bimestral TIFs may have overlapping date ranges at chunk boundaries
    df = df[~df.index.duplicated(keep="first")]
    log.info(f"  Snow: {len(df)} rows after dedup")
    return df


def extract_ndvi_evi(gdf, y0, y1):
    log.info("Extracting MOD13Q1 NDVI/EVI...")
    d = GEE_DIR / "vegetation"; rows = []
    for yr in range(max(y0, 2000), y1 + 1):
        for half, tag, mo_start in [(1, "h1", 1), (2, "h2", 7)]:
            tif = _pick(d, yr, [f"mod13q1_ndvi_evi_{{yr}}_{tag}_250m_clip.tif",
                                  f"mod13q1_ndvi_evi_{{yr}}_{tag}_250m.tif"])
            if tif is None: continue
            with rasterio.open(tif) as src: n = src.count
            n_comp = n // 2
            start = pd.Timestamp(f"{yr}-{mo_start:02d}-01")
            dates = pd.date_range(start, periods=n_comp, freq="16D").tolist()
            rows += _extract_tif(tif, gdf, dates, ["ndvi_mean", "evi_mean"], _scale_01)
            log.info(f"  NDVI {yr}-H{half}: {n_comp} composites")
    df = _to_df(rows, ["ndvi_mean", "evi_mean"])
    if df.empty: return df
    # 16-day composites → forward-fill to daily within available period
    return _monthly_to_daily(df, max(y0, 2000), y1, ENTITIES)


def extract_lswi(gdf, y0, y1):
    log.info("Extracting MOD09GA LSWI...")
    d = GEE_DIR / "vegetation"; rows = []
    BIMESTRES = [("m0102", 1), ("m0304", 3), ("m0506", 5),
                 ("m0708", 7), ("m0910", 9), ("m1112", 11)]
    for yr in range(max(y0, 2000), y1 + 1):
        for btag, bmo in BIMESTRES:
            tif = _pick(d, yr, [f"mod09ga_lswi_{{yr}}_{btag}_500m_clip.tif",
                                  f"mod09ga_lswi_{{yr}}_{btag}_500m.tif"])
            if tif is None: continue
            with rasterio.open(tif) as src: n = src.count
            start = pd.Timestamp(f"{yr}-{bmo:02d}-01")
            dates = pd.date_range(start, periods=n, freq="D").tolist()
            # LSWI = (NIR-SWIR)/(NIR+SWIR) computed server-side; already in [-1,1], no scale
            rows += _extract_tif(tif, gdf, dates, ["lswi_mean"], None)
    log.info(f"  LSWI: {len(rows)} rows extracted")
    df = _to_df(rows, ["lswi_mean"])
    if df.empty: return df
    # D042: forward-fill per entity
    for eid in ENTITIES:
        mask = df.index.get_level_values("entity_id") == eid
        df.loc[mask, "lswi_mean"] = df.loc[mask, "lswi_mean"].ffill()
    return df


def extract_lst(gdf, y0, y1):
    log.info("Extracting MOD11A1 LST...")
    d = GEE_DIR / "thermal"; rows = []
    for yr in range(max(y0, 2000), y1 + 1):
        for half, tag, mo_start in [(1, "h1", 1), (2, "h2", 7)]:
            tif = _pick(d, yr, [f"mod11a1_lst_{{yr}}_{tag}_1000m_clip.tif",
                                  f"mod11a1_lst_{{yr}}_{tag}_1000m.tif"])
            if tif is None: continue
            with rasterio.open(tif) as src: n = src.count
            n_days = n // 2
            start = pd.Timestamp(f"{yr}-{mo_start:02d}-01")
            dates = pd.date_range(start, periods=n_days, freq="D").tolist()
            rows += _extract_tif(tif, gdf, dates, ["lst_day_K", "lst_night_K"], _mask_lst)
            log.info(f"  LST {yr}-H{half}: {n_days} days")
    df = _to_df(rows, ["lst_day_K", "lst_night_K"])
    if df.empty: return df
    # Forward-fill cloud-cover gaps (28-33% NaN from cloud masking) within each entity
    df = df.sort_index()
    for col in ["lst_day_K", "lst_night_K"]:
        df[col] = df.groupby(level="entity_id")[col].transform(lambda s: s.ffill())
    return df


def extract_et(gdf, y0, y1):
    log.info("Extracting MOD16A2 ET/PET...")
    d = GEE_DIR / "et"; rows = []
    for yr in range(max(y0, 2001), y1 + 1):
        tif = _pick(d, yr, ["mod16a2_et_pet_{yr}_500m_clip.tif",
                             "mod16a2_et_pet_{yr}_500m.tif"])
        if tif is None: continue
        with rasterio.open(tif) as src: n = src.count
        n_comp = n // 2
        dates = pd.date_range(f"{yr}-01-01", periods=n_comp, freq="8D").tolist()
        rows += _extract_tif(tif, gdf, dates, ["et_mm8d", "pet_mm8d"], _mask_et)
        log.info(f"  ET {yr}: {n_comp} composites")
    df = _to_df(rows, ["et_mm8d", "pet_mm8d"])
    if df.empty: return df
    # 8-day composites → forward-fill to daily
    return _monthly_to_daily(df, max(y0, 2001), y1, ENTITIES)


def extract_smap(gdf, y0, y1):
    """SMAP SPL4SMGP/008 — 3-hourly stored → aggregate to daily mean."""
    log.info("Extracting SMAP soil moisture...")
    d = GEE_DIR / "smap"; rows = []

    for yr in range(max(y0, 2015), y1 + 1):
        for mo in range(1, 13):
            tif = _pick(d, yr, [f"smap_soil_{{yr}}_{mo:02d}_3h_10000m_clip.tif",
                                  f"smap_soil_{{yr}}_{mo:02d}_3h_10000m.tif"])
            if tif is None: continue

            with rasterio.open(tif) as src: n_bands = src.count
            # Structure: n_obs × 3 vars (sm_surface, sm_rootzone, sm_profile)
            # 8 observations/day × 3 vars = 24 bands/day
            n_obs  = n_bands // 3
            n_days = n_obs // 8
            if n_days == 0: continue

            # Read all bands once
            ext = TifZonalExtractor(tif, gdf, preprocess=None)  # raw m³/m³ values
            means = ext.zonal_mean()  # {eid: (n_bands,)}

            start = pd.Timestamp(f"{yr}-{mo:02d}-01")
            days = pd.date_range(start, periods=n_days, freq="D")

            for d_idx, dt in enumerate(days):
                # For each day, 8 time steps; each step has 3 vars consecutively
                # Bands (1-indexed): day_start + t*3 + [1,2,3] for t in range(8)
                day_b0 = d_idx * 24  # 0-indexed start band for this day
                for eid in ENTITIES:
                    em = means[eid]
                    # Average 8 time steps for sm_surface (var 0) and sm_rootzone (var 1)
                    sm_s = np.nanmean([em[day_b0 + t*3    ] for t in range(8)
                                        if day_b0 + t*3 < len(em)])
                    sm_r = np.nanmean([em[day_b0 + t*3 + 1] for t in range(8)
                                        if day_b0 + t*3 + 1 < len(em)])
                    rows.append({"date": dt.date(), "entity_id": eid,
                                 "sm_surface": sm_s if not np.isnan(sm_s) else None,
                                 "sm_rootzone": sm_r if not np.isnan(sm_r) else None})

        log.info(f"  SMAP {yr}: done")
    return _to_df(rows, ["sm_surface", "sm_rootzone"])


def extract_jrc(gdf, y0, y1):
    """JRC annual occurrence [0,2], replicated daily."""
    log.info("Extracting JRC surface water...")
    d = GEE_DIR / "water_land"; rows = []
    for yr in range(max(y0, 1984), min(y1, 2021) + 1):
        tif = _pick(d, yr, ["jrc_water_{yr}_30m.tif"])
        if tif is None:
            # Fall back to quinquennial legacy
            for y_q in range(1984, 2022, 5):
                y_qe = min(y_q + 4, 2021)
                if y_q <= yr <= y_qe:
                    cand = d / f"jrc_water_{y_q}_{y_qe}_500m.tif"
                    if cand.exists() and cand.stat().st_size > 500:
                        tif = cand; break
        if tif is None: continue

        sf = 100.0
        sc_path = tif.with_suffix(".json")
        if sc_path.exists():
            sf = json.loads(sc_path.read_text()).get("int16_scale_factor", 100.0)

        def _jrc(arr, _sf=sf): return _mask_jrc(arr, _sf)

        ext   = TifZonalExtractor(tif, gdf, preprocess=_jrc)
        means = ext.zonal_mean()

        # Band 0 = annual mean for this year (or for legacy, band index = yr - y_q)
        # For 30m annual files: 1 band. For legacy 500m quinquennial: multiple bands.
        with rasterio.open(tif) as src: n = src.count
        if n == 1:
            band_idx = 0
        else:
            # Quinquennial: determine which band corresponds to yr
            fname = tif.stem  # e.g. jrc_water_1984_1988_500m
            parts = tif.stem.split("_")
            try: y_q_start = int(parts[2]); band_idx = yr - y_q_start
            except: band_idx = 0

        for day in pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="D"):
            for eid in ENTITIES:
                rows.append({"date": day.date(), "entity_id": eid,
                             "jrc_water_occ": means[eid][band_idx] if band_idx < n else np.nan})

    return _to_df(rows, ["jrc_water_occ"])


def extract_landsat(gdf, y0, y1):
    """Landsat NDVI/NDWI/MNDWI — semestral, forward-filled daily."""
    log.info("Extracting Landsat...")
    d = GEE_DIR / "water_land"; rows = []
    for yr in range(max(y0, 1984), y1 + 1):
        for half, tag, mo_start in [(1, "h1", 1), (2, "h2", 7)]:
            tif = _pick(d, yr, [f"landsat_ndvi_ndwi_{{yr}}_{tag}_30m_clip.tif",
                                  f"landsat_ndvi_ndwi_{{yr}}_{tag}_250m.tif"])
            if tif is None: continue
            sc_path = tif.with_suffix(".json")
            valid_months = json.loads(sc_path.read_text()).get("valid_months", []) \
                           if sc_path.exists() else list(range(mo_start, mo_start + 6))
            dates = [pd.Timestamp(f"{yr}-{m:02d}-01") for m in valid_months]
            # NDVI/NDWI/MNDWI computed via normalizedDifference server-side; already [-1,1]
            rows += _extract_tif(tif, gdf, dates,
                                 ["lsat_ndvi", "lsat_ndwi", "lsat_mndwi"], None)
            log.info(f"  Landsat {yr}-H{half}: {len(valid_months)} months")
    df = _to_df(rows, ["lsat_ndvi", "lsat_ndwi", "lsat_mndwi"])
    if df.empty: return df
    # Reindex to daily, forward-fill
    return _monthly_to_daily(df, y0, y1, ENTITIES)


def extract_s1(gdf, y0, y1):
    log.info("Extracting Sentinel-1...")
    d = GEE_DIR / "sentinel1"; rows = []
    for yr in range(max(y0, 2015), y1 + 1):
        tif = _pick(d, yr, ["s1_vv_vh_ratio_{yr}_500m_clip.tif",
                             "s1_vv_vh_ratio_{yr}_500m.tif"])
        if tif is None: continue
        sc_path = tif.with_suffix(".json")
        valid_months = json.loads(sc_path.read_text()).get("valid_months", list(range(1, 13))) \
                       if sc_path.exists() else list(range(1, 13))
        dates = [pd.Timestamp(f"{yr}-{m:02d}-01") for m in valid_months]
        rows += _extract_tif(tif, gdf, dates, ["s1_vv", "s1_vh", "s1_ratio"])
        log.info(f"  S1 {yr}: {len(valid_months)} months")
    df = _to_df(rows, ["s1_vv", "s1_vh", "s1_ratio"])
    return _monthly_to_daily(df, y0, y1, ENTITIES) if not df.empty else df


def extract_s2(gdf, y0, y1):
    log.info("Extracting Sentinel-2...")
    d = GEE_DIR / "sentinel2"; rows = []
    for yr in range(max(y0, 2017), y1 + 1):
        tif = _pick(d, yr, ["s2_ndvi_ndwi_ndsi_{yr}_500m_clip.tif",
                             "s2_ndvi_ndwi_ndsi_{yr}_500m.tif"])
        if tif is None: continue
        sc_path = tif.with_suffix(".json")
        valid_months = json.loads(sc_path.read_text()).get("valid_months", list(range(1, 13))) \
                       if sc_path.exists() else list(range(1, 13))
        dates = [pd.Timestamp(f"{yr}-{m:02d}-01") for m in valid_months]
        # S2 NDVI/NDWI/NDSI via normalizedDifference server-side; already in [-1,1]
        rows += _extract_tif(tif, gdf, dates,
                             ["s2_ndvi", "s2_ndwi", "s2_ndsi"], None)
        log.info(f"  S2 {yr}: {len(valid_months)} months")
    df = _to_df(rows, ["s2_ndvi", "s2_ndwi", "s2_ndsi"])
    return _monthly_to_daily(df, y0, y1, ENTITIES) if not df.empty else df


def extract_landcover_static(gdf):
    """ESA WorldCover + MCD12Q1 majority class per sub-basin (static)."""
    log.info("Extracting land cover (static)...")
    d = GEE_DIR / "land_cover"
    rows = []

    esa_tif = _pick(d, None, ["esa_worldcover_2020_30m.tif",
                               "esa_worldcover_2020_10m.tif",
                               "esa_worldcover_2020_100m.tif"])
    if esa_tif:
        from rasterstats import zonal_stats as _zs
        with rasterio.open(esa_tif) as src:
            arr = src.read(1); tf = src.transform; crs = src.crs
        gdf_p = gdf.to_crs(crs) if gdf.crs != crs else gdf
        for eid, row in gdf_p.iterrows():
            z = _zs([row.geometry.__geo_interface__], arr, affine=tf,
                    stats=["majority"], nodata=0)[0]
            rows.append({"entity_id": eid, "esa_lc_majority": z.get("majority")})
    else:
        rows = [{"entity_id": eid, "esa_lc_majority": np.nan} for eid in ENTITIES]

    lc_df = pd.DataFrame(rows).set_index("entity_id")

    # MCD12Q1 2023 (most recent)
    mcd_2023 = d / "mcd12q1_igbp_2023.tif"
    if mcd_2023.exists():
        from rasterstats import zonal_stats as _zs
        with rasterio.open(mcd_2023) as src:
            arr = src.read(1); tf = src.transform; crs = src.crs
        gdf_p = gdf.to_crs(crs) if gdf.crs != crs else gdf
        for eid, row in gdf_p.iterrows():
            z = _zs([row.geometry.__geo_interface__], arr, affine=tf,
                    stats=["majority"], nodata=255)[0]
            lc_df.loc[eid, "mcd_lc_2023"] = z.get("majority")

    return lc_df


# ─── DataFrame helpers ────────────────────────────────────────────────────────

def _to_df(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "entity_id"]).sort_index()


def _monthly_to_daily(df: pd.DataFrame, y0: int, y1: int,
                      entities: list[str]) -> pd.DataFrame:
    """Forward-fill monthly (or semestral) values to daily."""
    date_range = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    midx = pd.MultiIndex.from_product([date_range, pd.CategoricalIndex(entities)],
                                       names=["date", "entity_id"])
    return df.reindex(midx).groupby(level="entity_id").ffill()


# ─── Main ─────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "chirps":  extract_chirps,
    "snow":    extract_snow,
    "ndvi":    extract_ndvi_evi,
    "lswi":    extract_lswi,
    "lst":     extract_lst,
    "et":      extract_et,
    "smap":    extract_smap,
    "jrc":     extract_jrc,
    "landsat": extract_landsat,
    "s1":      extract_s1,
    "s2":      extract_s2,
}


def merge_with_d6(sat_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Loading D6...")
    d6 = pd.read_csv(D6_PATH, parse_dates=["date"]).set_index(["date", "entity_id"]).sort_index()
    gdf = load_subbasins()
    lc_df = extract_landcover_static(gdf)
    d7 = d6.join(sat_df, how="left").join(lc_df, on="entity_id", how="left")
    if "chirps_pr_mm" in d7.columns and "pr_mm" in d7.columns:
        mask_nan = d7["pr_mm"].isna() & d7["chirps_pr_mm"].notna()
        d7.loc[mask_nan, "pr_mm"] = d7.loc[mask_nan, "chirps_pr_mm"]
        log.info(f"Filled {mask_nan.sum()} NaN pr_mm from CHIRPS")
    log.info(f"D7 shape: {d7.shape}")
    return d7


def main():
    parser = argparse.ArgumentParser(description="Build D7 satellite features from GEE TIFs")
    parser.add_argument("--source",     choices=list(EXTRACTORS.keys()))
    parser.add_argument("--year_start", type=int, default=1981)
    parser.add_argument("--year_end",   type=int, default=2025)
    parser.add_argument("--merge_d6",   action="store_true")
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = load_subbasins()
    log.info(f"Sub-basins: {list(gdf.index)}")

    def _run(src):
        cache = OUT_DIR / f"sat_{src}.parquet"
        if cache.exists() and not args.force:
            log.info(f"{src}: loading from cache")
            return pd.read_parquet(cache)
        df = EXTRACTORS[src](gdf, args.year_start, args.year_end)
        if not df.empty:
            df.to_parquet(cache)
            log.info(f"  → {cache.name} ({cache.stat().st_size/1e6:.2f} MB)")
        return df

    if args.source:
        df = _run(args.source)
        log.info(f"{args.source}: {df.shape if not df.empty else 'empty'}")
        return

    if args.merge_d6:
        frames = {}
        for src in EXTRACTORS:
            try:
                frames[src] = _run(src)
            except Exception as ex:
                log.error(f"{src}: FAILED — {ex}", exc_info=True)

        date_range = pd.date_range(f"{args.year_start}-01-01",
                                   f"{args.year_end}-12-31", freq="D")
        midx = pd.MultiIndex.from_product(
            [date_range, pd.CategoricalIndex(ENTITIES)],
            names=["date", "entity_id"],
        )
        sat_df = pd.DataFrame(index=midx)
        for df in frames.values():
            if not df.empty:
                sat_df = sat_df.join(df, how="left")

        d7 = merge_with_d6(sat_df)
        d7.to_csv(D7_PATH)
        log.info(f"D7 saved → {D7_PATH} ({D7_PATH.stat().st_size/1e6:.1f} MB)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
