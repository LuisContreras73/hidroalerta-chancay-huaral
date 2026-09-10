#!/usr/bin/env python3
"""
Script 20: Extrae forzantes PISCO por sub-cuenca → D6_multientity.csv

Motivación
----------
El TFT (Temporal Fusion Transformer) está diseñado para múltiples entidades
con covariables estáticas diferenciadas. Con una sola entidad (basin-mean),
los embeddings estáticos no aportan valor y la atención espacial no existe.

Este script:
  1. Carga el shapefile de 9 sub-cuencas (Chancay-Huaral)
  2. Para cada pixel PISCO dentro de la cuenca, identifica a qué sub-cuenca pertenece
     (spatial join punto-en-polígono usando geopandas)
  3. Extrae media de pr, tmax, tmin, pet calibrados (B2_pisco_basin_mean_corrected.csv)
     agrupada por sub-cuenca (ponderado por fracción del pixel)
  4. Calcula features dinámicas: API, SPI_30d, SPI_90d, water_deficit_30d por entidad
  5. Agrega covariables estáticas del DEM y SoilGrids por sub-cuenca
  6. Añade features calendáricas y ONI (known_future)
  7. Apila en formato long → D6_multientity.csv + D6_schema.json

Nota sobre resolución
---------------------
PISCO: 0.1° (~11 km). 41 pixels en la cuenca. 9 sub-cuencas (~340 km² promedio).
Cada sub-cuenca cubre 2-5 pixels PISCO. Las diferencias espaciales reflejan
el gradiente altitudinal real (cuenca costa-sierra: 0-5291 m).

Para TFT multi-entidad, esto proporciona:
  - Atención espacial: qué sub-cuenca activa el modelo antes de un evento
  - Variable importance por sub-cuenca: qué variables son clave en cada zona
  - Representación física del gradient orográfico de precipitación

Salidas:
  data/model_ready/D6_multientity.csv   -- 9 entidades × 14610 días = 131490 filas
  data/model_ready/D6_schema.json       -- schema TFT multi-entidad
"""
import datetime
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.stats import gamma as gamma_dist

ROOT = Path(__file__).parent.parent.parent   # scripts/03_gold → scripts → project root
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "outputs" / "20_multientidad.log", "w", "utf-8"),
    ],
)
log = logging.getLogger("multientidad")

# ── Rutas ──────────────────────────────────────────────────────────────────────
SHP_SUBCUENCAS = ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas" / \
    "Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_geogpsperu_UH_MENORES.shp"
PR_NC      = ROOT / "data/bronze/B2_pr_basin_grid_v3.nc"    # PISCOp v3.0 1981-2025 (D050)
TMAX_NC    = ROOT / "data/bronze/B2_tmax_basin_grid_v12.nc"
TMIN_NC    = ROOT / "data/bronze/B2_tmin_basin_grid_v12.nc"
CORRECTED  = ROOT / "data/bronze/B2_pisco_basin_mean_corrected.csv"
DEM_STATS  = ROOT / "data/bronze/B4_dem_subcuencas_stats.csv"
SOIL_CSV   = ROOT / "data/bronze/B5_soilgrids_basin_mean.csv"
ONI_CSV    = ROOT / "data/silver/enso/S3_oni_1950_2026.csv"
Q_CSV      = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
G1_Q_SIM   = ROOT / "data/gold/G1_q_sim_gr4j.csv"           # GR4J original 1981-2025
G1_Q_CORR  = ROOT / "data/gold/G1_q_sim_gr4j_corrected.csv" # GR4J corregido por QM (preferido)
OUT_CSV    = ROOT / "data/model_ready/D6_multientity.csv"
OUT_SCHEMA = ROOT / "data/model_ready/D6_schema.json"

T0     = pd.Timestamp("1981-01-01")
T1     = pd.Timestamp("2025-12-31")   # EXTENDIDO a 2025 (antes 2020) — usar Q obs real 2021-2025
DATES  = pd.date_range(T0, T1, freq="D")
N_DAYS = len(DATES)   # 16436

B10_ERA5 = None  # cargado en main(), basin-mean ERA5 daily (t2m hasta 2025)

API_K    = 0.85   # decay antecedent precipitation index (basin-mean como default)
N_Q      = 100    # cuantiles SPI

BASIN_LAT = -11.35
BASIN_LON = -77.05
Q_STATION = "q_santo_domingo_47e214d2"
Q_CONV    = 86.4 / 3062.62   # m³/s → mm/día


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_api(pr: np.ndarray, k: float = API_K) -> np.ndarray:
    """API exponencial: API_t = k × API_{t-1} + pr_t"""
    api = np.zeros(len(pr))
    for i in range(1, len(pr)):
        api[i] = k * api[i - 1] + (pr[i - 1] if np.isfinite(pr[i - 1]) else 0.0)
    return api


def compute_spi(pr: pd.Series, window: int) -> pd.Series:
    roll = pr.rolling(window, min_periods=int(window * 0.5))
    mu   = roll.mean()
    sig  = roll.std()
    spi  = (pr - mu) / (sig + 1e-6)
    return spi.clip(-3.5, 3.5)


def ra_mm(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Rₐ [mm/día] FAO-56 en una latitud."""
    phi = np.radians(lat_deg)
    dr  = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    d   = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws  = np.arccos(np.clip(-np.tan(phi) * np.tan(d), -1, 1))
    Ra  = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(ws))
    return Ra * 0.408


def build_known_future(dates: pd.DatetimeIndex, clim_pr: np.ndarray) -> pd.DataFrame:
    doy   = dates.dayofyear.values
    month = dates.month.values
    week  = dates.isocalendar().week.values.astype(int)
    hm    = np.where(month >= 9, month - 8, month + 4).astype(int)
    wet   = ((month >= 11) | (month <= 4)).astype(int)
    s1    = np.sin(2 * np.pi * doy / 365.25)
    c1    = np.cos(2 * np.pi * doy / 365.25)
    s2    = np.sin(4 * np.pi * doy / 365.25)
    c2    = np.cos(4 * np.pi * doy / 365.25)
    # Climatología mensual de precipitación (p50 / p90) usando clim_pr [12]
    p50   = clim_pr[month - 1, 0]
    p90   = clim_pr[month - 1, 1]
    return pd.DataFrame({
        "doy": doy, "month": month, "week": week, "hydro_month": hm,
        "is_wet_season": wet,
        "sin_doy_1": s1, "cos_doy_1": c1, "sin_doy_2": s2, "cos_doy_2": c2,
        "clim_pr_p50": p50, "clim_pr_p90": p90,
    }, index=dates)


def assign_split(dates: pd.DatetimeIndex) -> np.ndarray:
    # Split extendido a 2025 (D6 ahora 1981-2025):
    #   train: 1981-2015 (GR4J reconstruct)
    #   val:   2016-2020 (GR4J reconstruct + Q obs sep-dic 2020)
    #   test:  2021-2025 (Q OBSERVADO REAL — ~1485 días)
    split = np.where(dates.year <= 2015, "train",
            np.where(dates.year <= 2020, "val", "test"))
    return split


# ── 1. Sub-cuencas shapefile ───────────────────────────────────────────────────
def load_subcuencas() -> gpd.GeoDataFrame:
    shp = gpd.read_file(SHP_SUBCUENCAS).to_crs("EPSG:4326")
    # Filtrar outlier fuera de la cuenca
    cents = shp.to_crs("EPSG:32718").centroid.to_crs("EPSG:4326")
    valid = (cents.y > -12.0) & (cents.y < -10.5)
    shp   = shp[valid].copy().reset_index(drop=True)
    # Centroides en WGS84
    cents_utm = shp.to_crs("EPSG:32718").centroid.to_crs("EPSG:4326")
    shp["lat_c"] = cents_utm.y.values
    shp["lon_c"] = cents_utm.x.values
    shp["area_km2_wgs"] = shp.to_crs("EPSG:32718").geometry.area.values / 1e6
    shp["sub_id"] = shp["ID"].astype(str)
    shp["entity_id"] = "sub_" + shp["sub_id"]
    log.info(f"   {len(shp)} sub-cuencas cargadas:")
    for _, r in shp.iterrows():
        log.info(f"     {r['entity_id']}  lat={r['lat_c']:.3f}  lon={r['lon_c']:.3f}  "
                 f"area={r['area_km2_wgs']:.1f} km²")
    return shp


# ── 2. Extraer PISCO por sub-cuenca ───────────────────────────────────────────
def extract_pisco_per_subcuenca(subcuencas: gpd.GeoDataFrame) -> dict:
    """
    Devuelve dict {entity_id: {'pr': Series, 'tmax': Series, 'tmin': Series, 'pet': Series}}
    usando spatial join de centroides PISCO sobre polígonos de sub-cuencas.
    """
    # Cargar grilla pr (0.1°, 41 pixels) para asignación espacial
    ds_pr  = nc.Dataset(PR_NC)
    lats_p = np.array(ds_pr["lat"][:])       # (11,) — sur a norte
    lons_p = np.array(ds_pr["lon"][:])       # (9,)
    mask_p = np.array(ds_pr["basin_mask"][:]).astype(bool)  # (11, 9)
    pr_raw = np.array(ds_pr["pr"][:])        # (14244, 11, 9)
    ds_pr.close()

    # Fechas del PR — longitud desde el archivo (v3: 1981-2025, 16436 días)
    n_pr = pr_raw.shape[0]
    dates_pr = pd.date_range("1981-01-01", periods=n_pr, freq="D")

    # Cargar tmax / tmin (0.01°, 2533 pixels activos)
    ds_tx = nc.Dataset(TMAX_NC)
    lats_tx = np.array(ds_tx["lat"][:])      # (111,)
    lons_tx = np.array(ds_tx["lon"][:])      # (91,)
    mask_tx = np.array(ds_tx["basin_mask"][:]).astype(bool)
    tmax_raw = np.array(ds_tx["tmax"][:])    # (14610, 111, 91)
    ds_tx.close()

    ds_tn  = nc.Dataset(TMIN_NC)
    tmin_raw = np.array(ds_tn["tmin"][:])
    ds_tn.close()

    # Fechas tmax/tmin (PISCOt 1981-2020)
    dates_t = pd.date_range("1981-01-01", periods=14610, freq="D")

    # Cargar PET corregida (basin-mean) como fallback; por sub-cuenca no tenemos grilla
    df_corr = pd.read_csv(CORRECTED, index_col=0, parse_dates=True)
    df_corr = df_corr[[c for c in df_corr.columns if not c.endswith("_orig")]]

    # ── Construir GeoDataFrame de centroides PISCO (pr, 0.1°) ────────────────
    points = []
    for i, lat in enumerate(lats_p):
        for j, lon in enumerate(lons_p):
            if mask_p[i, j]:
                points.append({"pi": i, "pj": j, "lat": lat, "lon": lon})
    gdf_pts = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy([p["lon"] for p in points],
                                    [p["lat"] for p in points]),
        crs="EPSG:4326",
    )

    # Spatial join: cada pixel PISCO → sub-cuenca
    joined = gpd.sjoin(gdf_pts, subcuencas[["entity_id", "geometry"]],
                       how="left", predicate="within")
    # Pixels sin asignación (borde): asignar al centroide más cercano
    no_match = joined["entity_id"].isna()
    if no_match.any():
        for idx in joined[no_match].index:
            pt = gdf_pts.loc[idx, "geometry"]
            dists = subcuencas.geometry.centroid.distance(pt)
            joined.loc[idx, "entity_id"] = subcuencas.loc[dists.idxmin(), "entity_id"]

    log.info("   Asignación PISCO pixels (0.1°) → sub-cuencas:")
    for eid, grp in joined.groupby("entity_id"):
        log.info(f"     {eid}: {len(grp)} pixels")

    # ── Extraer series por sub-cuenca ─────────────────────────────────────────
    result = {}
    for _, row in subcuencas.iterrows():
        eid  = row["entity_id"]
        pxls = joined[joined["entity_id"] == eid][["pi", "pj"]].values

        # --- pr (0.1°) ---
        if len(pxls) > 0:
            pr_stack = np.array([pr_raw[:, p[0], p[1]] for p in pxls])
            pr_mean  = np.nanmean(pr_stack, axis=0)   # (n_pr,)
        else:
            pr_mean = np.full(n_pr, np.nan)
        pr_ser = pd.Series(pr_mean, index=dates_pr, name="pr_mm")
        # Aplicar QM: usar corrección de basin-mean como proxy (no tenemos QM por pixel)
        # La escala relativa entre sub-cuencas se preserva; la corrección absoluta
        # se aplica multiplicativamente con la relación corrected/original
        pr_basin_orig = df_corr["pr_orig"].reindex(dates_pr) if "pr_orig" in df_corr.columns \
            else df_corr["pr"].reindex(dates_pr)
        pr_basin_corr = df_corr["pr"].reindex(dates_pr)
        # ratio = corr/orig; NaN for dates beyond correction file (2020+) → monthly mean fallback
        ratio = (pr_basin_corr / (pr_basin_orig + 1e-9)).clip(0, 10)
        if ratio.isna().any():
            monthly_r = ratio.groupby(ratio.index.month).mean()
            gap = ratio.isna()
            ratio.loc[gap] = ratio.index[gap].month.map(monthly_r)
        pr_ser_corr = (pr_ser * ratio.values).clip(0, None)
        pr_ser_corr = pr_ser_corr.reindex(DATES)   # extend to 2020 (NaN for 2020)

        # --- tmax / tmin (0.01°): usar pixel más cercano al centroide sub-cuenca ---
        lat_c, lon_c = float(row["lat_c"]), float(row["lon_c"])
        ii = int(np.argmin(np.abs(lats_tx - lat_c)))
        jj = int(np.argmin(np.abs(lons_tx - lon_c)))
        # Si pixel fuera de cuenca, buscar el más cercano dentro
        if not mask_tx[ii, jj]:
            ii_flat, jj_flat = np.where(mask_tx)
            dists = (ii_flat - ii)**2 + (jj_flat - jj)**2
            best  = np.argmin(dists)
            ii, jj = ii_flat[best], jj_flat[best]

        tmax_ser = pd.Series(tmax_raw[:, ii, jj], index=dates_t, name="tmax_c")
        tmin_ser = pd.Series(tmin_raw[:, ii, jj], index=dates_t, name="tmin_c")

        # Aplicar corrección delta-T (mensual, desde B6_T_correction.csv)
        try:
            tc = pd.read_csv(ROOT / "data/bronze/B6_T_correction.csv")
            tc.index = tc.iloc[:, 0].astype(int)  # mes 1-12
            for m in range(1, 13):
                idx_m = dates_t.month == m
                tmax_ser.iloc[idx_m] += float(tc.loc[m, "delta_tmax_c"])
                tmin_ser.iloc[idx_m] += float(tc.loc[m, "delta_tmin_c"])
        except Exception as e:
            log.warning(f"   No se pudo aplicar delta-T: {e}")

        tmax_ser = tmax_ser.reindex(DATES)
        tmin_ser = tmin_ser.reindex(DATES)

        # --- EXTENSIÓN 2021-2025: derivar T de ERA5 t2m calibrado por sub-cuenca ---
        # PISCOt solo llega a 2020. Para 2021-2025 usamos ERA5 t2m (B10) calibrado
        # contra PISCOt por offset mensual (reconstruye gradiente altitudinal).
        # Mismo principio que la corrección PISCOt vs SENAMHI: usar la serie corregida
        # de alta calidad (PISCOt) como referencia para calibrar ERA5.
        tmax_ser = tmax_ser.astype("float64")
        tmin_ser = tmin_ser.astype("float64")
        gap_mask = tmax_ser.isna() | tmin_ser.isna()
        if gap_mask.any() and B10_ERA5 is not None and "t2m" in B10_ERA5.columns:
            t2m = B10_ERA5["t2m"].reindex(DATES)
            # offset y DTR mensual calibrados en período común PISCOt (pre-2021)
            common = ~(tmax_ser.isna()) & t2m.notna()
            tmean_pisco_c = ((tmax_ser + tmin_ser) / 2.0)[common]
            t2m_c = t2m[common]
            dtr_c = (tmax_ser - tmin_ser)[common]
            offset_m = (tmean_pisco_c - t2m_c).groupby(tmean_pisco_c.index.month).mean()
            dtr_m    = dtr_c.groupby(dtr_c.index.month).mean()
            # Reconstruir T en el gap (2021-2025)
            gm = DATES[gap_mask].month
            tmean_recon = t2m[gap_mask].values + gm.map(offset_m).values
            dtr_recon   = gm.map(dtr_m).values
            tmax_ser.loc[gap_mask] = tmean_recon + dtr_recon / 2.0
            tmin_ser.loc[gap_mask] = tmean_recon - dtr_recon / 2.0

        tmean    = (tmax_ser + tmin_ser) / 2.0

        # --- PET HS calibrada por sub-cuenca ---
        doy_arr = DATES.dayofyear.values.astype(float)
        Ra      = ra_mm(doy_arr, lat_c)
        dtr     = np.maximum((tmax_ser - tmin_ser).values, 0.0)
        et0_hs  = 0.0023 * Ra * np.sqrt(dtr) * (tmean.values + 17.8)
        # Factor k mensual: calibrar SOLO en días con pet_ref válido (1981-2020),
        # aplicar a todo el período (incluido 2021-2025 extendido).
        pet_ref = df_corr["pet"].reindex(DATES)
        for m in range(1, 13):
            idx_m  = np.asarray(DATES.month == m)
            valid  = idx_m & np.isfinite(pet_ref.values) & np.isfinite(et0_hs)
            km = float(np.nansum(pet_ref.values[valid]) / (np.nansum(et0_hs[valid]) + 1e-9))
            et0_hs[idx_m] *= km
        pet_ser = pd.Series(et0_hs.clip(0, None), index=DATES, name="pet_mm")

        result[eid] = {
            "pr_mm":  pr_ser_corr,
            "tmax_c": tmax_ser,
            "tmin_c": tmin_ser,
            "tmean_c": tmean,
            "pet_mm": pet_ser,
        }
        log.info(f"   [{eid}] pr_mean={pr_ser_corr.mean():.3f} mm/d  "
                 f"tmax={tmax_ser.mean():.1f}°C  tmin={tmin_ser.mean():.1f}°C  "
                 f"pet={pet_ser.mean():.3f} mm/d")

    return result


# ── 3. Covariables estáticas por sub-cuenca ────────────────────────────────────
def load_static_covariates(subcuencas: gpd.GeoDataFrame) -> pd.DataFrame:
    dem = pd.read_csv(DEM_STATS).set_index("subcuenca_id")

    soil = pd.read_csv(SOIL_CSV)
    top  = soil[soil["depth"] == "0-5cm"].set_index("property")["basin_mean"]
    soil_static = {
        "soil_sand_pct": float(top.get("sand",  np.nan)) * 0.1,
        "soil_silt_pct": float(top.get("silt",  np.nan)) * 0.1,
        "soil_clay_pct": float(top.get("clay",  np.nan)) * 0.1,
        "soil_bdod":     float(top.get("bdod",  np.nan)),
        "soil_soc":      float(top.get("soc",   np.nan)),
        "soil_cec":      float(top.get("cec",   np.nan)) * 0.1,
        "soil_ph":       float(top.get("phh2o", np.nan)),
    }

    rows = []
    for _, row in subcuencas.iterrows():
        sub_id = int(row["sub_id"])
        d = dem.loc[sub_id] if sub_id in dem.index else {}
        r = {
            "entity_id":      row["entity_id"],
            "lat":            float(row["lat_c"]),
            "lon":            float(row["lon_c"]),
            "basin_area_km2": float(row["area_km2_wgs"]),
            "elevation_m":    float(d.get("elev_mean_m", 2678.0)),
            "slope_deg":      float(d.get("slope_mean_deg", 25.0)),
            "hyps_integral":  0.506,   # HI global cuenca (Strahler)
        }
        r.update(soil_static)
        rows.append(r)
    return pd.DataFrame(rows).set_index("entity_id")


# ── 4. ONI diario ─────────────────────────────────────────────────────────────
def load_oni() -> pd.Series:
    df = pd.read_csv(ONI_CSV)
    df["date"] = pd.to_datetime(df["date"])
    oni = df.set_index("date")["oni"].sort_index()
    # 2-month lag: ENSO signal takes ~2 months to affect Andean hydrology
    oni_lagged = oni.shift(2, freq="MS")
    oni_daily = oni_lagged.reindex(DATES, method="ffill")
    return oni_daily.rename("oni_index")


# ── 5. Caudal observado + reconstrucción GR4J como fallback ────────────────────
def load_q() -> pd.Series:
    # Observed (only ~2020-09 onward, everything else is NaN)
    df_obs = pd.read_csv(Q_CSV, index_col=0, parse_dates=True)
    if Q_STATION in df_obs.columns:
        q_obs = (df_obs[Q_STATION] * Q_CONV).rename("q_mm").reindex(DATES)
    else:
        q_obs = pd.Series(np.nan, index=DATES, name="q_mm")

    # GR4J reconstruction 1981-2025 — preferir el CORREGIDO por QM (más realista,
    # sin el aplanamiento del original). Fallback al original si no existe.
    try:
        if G1_Q_CORR.exists():
            df_gr4j = pd.read_csv(G1_Q_CORR, index_col=0, parse_dates=True)
            col = "q_sim_m3s_corrected"
            log.info("   Q fallback: GR4J CORREGIDO por QM (G1_q_sim_gr4j_corrected.csv)")
        else:
            df_gr4j = pd.read_csv(G1_Q_SIM, index_col=0, parse_dates=True)
            col = "q_sim_m3s"
            log.info("   Q fallback: GR4J original (corregido no encontrado)")
        q_gr4j = (df_gr4j[col] * Q_CONV).rename("q_mm").reindex(DATES)
    except Exception as e:
        log.warning(f"   No se pudo cargar GR4J Q: {e} — usando solo observado")
        q_gr4j = pd.Series(np.nan, index=DATES, name="q_mm")

    # Observed takes priority; GR4J corregido rellena gaps y período histórico
    return q_obs.combine_first(q_gr4j)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("SCRIPT 20: Forzantes por sub-cuenca → D6 multi-entidad")
    log.info("=" * 70)

    # ── Cargar ERA5 t2m (para extender temperatura 2021-2025) ─────────────────
    global B10_ERA5
    b10_path = ROOT / "data/bronze/B10_era5land_daily.csv"
    if b10_path.exists():
        B10_ERA5 = pd.read_csv(b10_path, index_col=0, parse_dates=True)
        B10_ERA5.index = pd.DatetimeIndex(B10_ERA5.index).normalize()
        log.info(f"[0] ERA5 B10 cargado para extensión T 2021-2025 "
                 f"({B10_ERA5.index.min().date()} → {B10_ERA5.index.max().date()})")
    else:
        log.warning("[0] B10_era5land_daily.csv no encontrado — T quedará NaN en 2021-2025")

    # ── Cargar sub-cuencas ────────────────────────────────────────────────────
    log.info("\n[1] Cargando sub-cuencas ...")
    subcuencas = load_subcuencas()

    # ── Extraer PISCO por sub-cuenca ──────────────────────────────────────────
    log.info("\n[2] Extrayendo forzantes PISCO por sub-cuenca ...")
    pisco_by_sub = extract_pisco_per_subcuenca(subcuencas)

    # ── Covariables estáticas ─────────────────────────────────────────────────
    log.info("\n[3] Cargando covariables estáticas (DEM + SoilGrids) ...")
    static_df = load_static_covariates(subcuencas)
    log.info(f"   Estáticas: {static_df.columns.tolist()}")

    # ── ONI y caudal ──────────────────────────────────────────────────────────
    log.info("\n[4] Cargando ONI y caudal ...")
    oni_series = load_oni()
    q_series   = load_q()
    log.info(f"   ONI: {oni_series.notna().sum()} días | Q: {q_series.notna().sum()} días")

    # ── Construir dataset largo ────────────────────────────────────────────────
    log.info("\n[5] Construyendo D6 multi-entidad ...")
    frames = []

    for _, row in subcuencas.iterrows():
        eid = row["entity_id"]
        frc = pisco_by_sub[eid]
        st  = static_df.loc[eid]

        pr   = frc["pr_mm"]
        tmax = frc["tmax_c"]
        tmin = frc["tmin_c"]
        tmn  = frc["tmean_c"]
        pet  = frc["pet_mm"]

        # Climatología mensual pr (p50, p90) para known_future
        pr_vals = pr.values
        clim_pr = np.zeros((12, 2))
        for m in range(1, 13):
            idx_m = DATES.month == m
            vals  = pr_vals[idx_m]
            vals  = vals[np.isfinite(vals)]
            if len(vals) > 0:
                clim_pr[m - 1, 0] = np.nanpercentile(vals, 50)
                clim_pr[m - 1, 1] = np.nanpercentile(vals, 90)

        # Features dinámicas
        api   = compute_api(pr.values)
        spi30 = compute_spi(pr.fillna(0), 30)
        spi90 = compute_spi(pr.fillna(0), 90)
        wdef  = (pet - pr.reindex(DATES)).rolling(30, min_periods=15).mean()

        # Water deficit rolling 30d
        water_def = pd.Series(
            (pet.values - pr.fillna(0).values),
            index=DATES).rolling(30, min_periods=15).mean()

        # Targets: pr propio de sub-cuenca + Q en Santo Domingo (solo salida)
        pr_next_1d     = pr.shift(-1)
        pr_sum_next_3d = pr.shift(-1).rolling(3).sum().shift(-2)
        pr_sum_next_7d = pr.shift(-1).rolling(7).sum().shift(-6)
        # Q target: mismo valor para todas las entidades (salida única cuenca)
        q_next_1d    = q_series.shift(-1)
        q_sum_next_7d = q_series.shift(-1).rolling(7).sum().shift(-6)

        # Known future
        kf = build_known_future(DATES, clim_pr)

        # Ensamblar fila por día
        df_eid = pd.DataFrame({
            # Identificadores
            "date":      DATES,
            "entity_id": eid,
            "split":     assign_split(DATES),
            # Past observed
            "pr_mm":     pr.values,
            "tmax_c":    tmax.values,
            "tmin_c":    tmin.values,
            "tmean_c":   tmn.values,
            "pet_mm":    pet.values,
            "q_mm":      q_series.values,
            "oni_index": oni_series.values,
            "api":       api,
            "spi_30d":   spi30.values,
            "spi_90d":   spi90.values,
            "water_deficit_30d": water_def.values,
            # Known future
            **{c: kf[c].values for c in kf.columns},
            # Static (repetidas por día)
            "lat":            float(st["lat"]),
            "lon":            float(st["lon"]),
            "elevation_m":    float(st["elevation_m"]),
            "basin_area_km2": float(st["basin_area_km2"]),
            "slope_deg":      float(st["slope_deg"]),
            "hyps_integral":  float(st["hyps_integral"]),
            "soil_sand_pct":  float(st["soil_sand_pct"]),
            "soil_silt_pct":  float(st["soil_silt_pct"]),
            "soil_clay_pct":  float(st["soil_clay_pct"]),
            "soil_bdod":      float(st["soil_bdod"]),
            "soil_soc":       float(st["soil_soc"]),
            "soil_cec":       float(st["soil_cec"]),
            "soil_ph":        float(st["soil_ph"]),
            # Targets
            "pr_next_1d":      pr_next_1d.values,
            "pr_sum_next_3d":  pr_sum_next_3d.values,
            "pr_sum_next_7d":  pr_sum_next_7d.values,
            "q_next_1d":       q_next_1d.values,
            "q_sum_next_7d":   q_sum_next_7d.values,
        })
        frames.append(df_eid)

    D6 = pd.concat(frames, ignore_index=True)
    D6 = D6.sort_values(["entity_id", "date"]).reset_index(drop=True)

    log.info(f"   D6: {len(D6)} filas | {D6['entity_id'].nunique()} entidades | "
             f"{D6.columns.tolist().__len__()} columnas")

    # ── Guardar CSV ────────────────────────────────────────────────────────────
    log.info("\n[6] Guardando D6_multientity.csv ...")
    D6.to_csv(OUT_CSV, index=False)
    log.info(f"   Guardado: {OUT_CSV.name}  ({OUT_CSV.stat().st_size / 1e6:.1f} MB)")

    # ── Schema JSON ───────────────────────────────────────────────────────────
    schema = {
        "past_observed": [
            "pr_mm", "tmax_c", "tmin_c", "tmean_c", "pet_mm",
            "q_mm", "oni_index", "api", "spi_30d", "spi_90d", "water_deficit_30d"
        ],
        "known_future": [
            "doy", "month", "week", "hydro_month", "is_wet_season",
            "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
            "clim_pr_p50", "clim_pr_p90"
        ],
        "static_covariates": [
            "entity_id", "lat", "lon", "elevation_m", "basin_area_km2",
            "slope_deg", "hyps_integral",
            "soil_sand_pct", "soil_silt_pct", "soil_clay_pct",
            "soil_bdod", "soil_soc", "soil_cec", "soil_ph"
        ],
        "targets": [
            "pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d",
            "q_next_1d", "q_sum_next_7d"
        ],
        "split_column": "split",
        "time_column":  "date",
        "entity_column": "entity_id",
        "n_entities":    int(D6["entity_id"].nunique()),
        "entity_ids":    sorted(D6["entity_id"].unique().tolist()),
        "n_rows_total":  len(D6),
        "encoder_length": 90,
        "prediction_length": 7,
        "tft_params": {
            "hidden_size": 64,
            "lstm_layers": 2,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "max_epochs": 150,
        },
        "note": (
            "Multi-entity TFT: 9 sub-cuencas Chancay-Huaral. "
            "pr por sub-cuenca desde PISCOp 0.1° con QM proporcional. "
            "tmax/tmin desde PISCOt 0.01° pixel más cercano al centroide, con delta-T. "
            "pet: HS calibrada con k mensual basin-mean. "
            "q_mm: SNIRH Santo Domingo (salida cuenca), mismo valor todas las entidades. "
            "Entidades ordenadas de baja (sub_634, costa) a alta (sub_649/656, cabecera)."
        ),
        "generated_by": "scripts/20_subcuencas_multientidad.py",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lineage": {
            "pr_mm": {
                "source": "PISCOp v3.0 (1981-2025, 0.1°) con QM proporcional basin-mean",
                "file": "data/bronze/B2_pr_basin_grid_v3.nc",
                "generated_by_step": "Script 20 — spatial join 0.1° pixel → sub-cuenca; QM ratio de B2_pisco_basin_mean_corrected.csv",
                "method": "Quantile Mapping mensual proporcional; ratio calibrado 1981-2019; media mensual para 2020+",
                "known_issue": "2020: QM ratio extrapolado por media mensual (v3 no tiene QM independiente aun)",
            },
            "tmax_c": {
                "source": "PISCOt v1.2 (1981-2020, 0.01°)",
                "file": "data/bronze/B2_tmax_corrected_spatial.nc",
                "generated_by_step": "Script 06c — correccion espacial + delta-T mensual (Script 19)",
                "method": "pixel mas cercano al centroide de sub-cuenca; delta-T = mean(Tobs-m) - mean(Tpisco-m)",
            },
            "tmin_c": {
                "source": "PISCOt v1.2 (1981-2020, 0.01°)",
                "file": "data/bronze/B2_tmin_corrected_spatial.nc",
                "generated_by_step": "Script 06c — correccion espacial + delta-T mensual (Script 19)",
                "method": "pixel mas cercano al centroide de sub-cuenca; delta-T = mean(Tobs-m) - mean(Tpisco-m)",
            },
            "pet_mm": {
                "source": "PISCO ETP 1981-2016 + Hargreaves-Samani calibrado 2017-2020",
                "file": "data/bronze/B2_pet_hargreaves_cal.nc",
                "generated_by_step": "Script 07b — HS con k mensual calibrado vs Penman-Monteith",
                "method": "HS: PET = 0.0023*(Tmean+17.8)*TD^0.5*Ra; k calibrado por mes en 1981-2016",
            },
            "q_mm": {
                "source": "SNIRH ANA — Estacion 47E214D2 (Santo Domingo Automatica, 614 m)",
                "file": "data/silver/snirh/S1_snirh_daily_q.csv",
                "generated_by_step": "Script 03 — ingestión y conversion m3/s -> mm/dia por area de cuenca",
                "method": "q_mm = Q_m3s * 86400 / (area_km2 * 1e6) * 1000",
                "known_issue": "Solo disponible 2020-09 en adelante; NaN para 1981-2020-08",
            },
            "oni_index": {
                "source": "NOAA Oceanic Nino Index (ONI) 1950-2026",
                "file": "data/silver/enso/S3_oni_1950_2026.csv",
                "method": "3-month running mean of ERSST.v5 SST anomalies in Nino 3.4 region",
            },
            "static_covariates": {
                "elevation_m": "Copernicus GLO-30 DEM (30m) → estadisticas por sub-cuenca shapefile",
                "soil_*": "SoilGrids 250m — media areal por sub-cuenca",
                "source_files": "data/bronze/B4_dem_subcuencas_stats.csv, data/bronze/B5_soilgrids_basin_mean.csv",
            },
        },
    }
    with open(OUT_SCHEMA, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    log.info(f"   Schema: {OUT_SCHEMA.name}")

    # ── Resumen por entidad ────────────────────────────────────────────────────
    log.info("\n======================================================================")
    log.info("RESUMEN D6 POR ENTIDAD")
    log.info("  entity_id       elev_m   pr_mean  tmax_mean  tmin_mean  pet_mean")
    for eid, grp in D6.groupby("entity_id"):
        st = static_df.loc[eid]
        log.info(f"  {eid:12s}  {st['elevation_m']:6.0f}   "
                 f"{grp['pr_mm'].mean():.3f}    "
                 f"{grp['tmax_c'].mean():.1f}     "
                 f"{grp['tmin_c'].mean():.1f}     "
                 f"{grp['pet_mm'].mean():.3f}")

    log.info("")
    log.info("SIGUIENTE PASOS:")
    log.info("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    log.info("  pip install pytorch-forecasting lightning")
    log.info("  python scripts/22_tft_training.py")
    log.info("======================================================================")


if __name__ == "__main__":
    main()
