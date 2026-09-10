#!/usr/bin/env python3
"""
Script 07: Extiende la evapotranspiración potencial (PET) a 1981-2020
usando el método Hargreaves-Samani (HS) calibrado contra PISCOp PET.

Motivación
----------
PISCOp v1.1 proporciona PET (Penman-Monteith de referencia) solo hasta 2016.
Se requiere PET para 2017-2020 para completar los forzantes del modelo
hidrológico GR4J.  El método HS usa únicamente temperatura máxima y mínima
(disponibles en PISCOt v1.2, 1981-2020), lo que garantiza consistencia en
la fuente de datos.

Fórmula Hargreaves-Samani (1985)
---------------------------------
  ET₀ = 0.0023 × Rₐ × √(ΔT) × (T̄ + 17.8)   [mm/día]

  donde:
    T̄  = (Tmax + Tmin) / 2            [°C]
    ΔT = max(Tmax − Tmin, 0)          [°C] — rango diurno; ≥0 por consistencia
    Rₐ = radiación extraterrestre      [mm/día equivalente hídrico]

Radiación Extraterrestre Rₐ (FAO-56, Allen et al. 1998, Ec. 21-28)
---------------------------------------------------------------------
  dr  = 1 + 0.033 · cos(2π·J/365)          — distancia relativa Tierra-Sol (Ec.23)
  δ   = 0.409 · sin(2π·J/365 − 1.39)       — declinación solar [rad]          (Ec.24)
  φ   = latitud [rad]
  ωs  = arccos(−tan(φ)·tan(δ))             — ángulo horario del atardecer [rad](Ec.25)

  Rₐ = (24·60/π) · 0.0820 · dr
       · [ωs·sin(φ)·sin(δ) + cos(φ)·cos(δ)·sin(ωs)]   [MJ/m²/día]  (Ec.21)

  Rₐ [mm/día] = Rₐ [MJ/m²/día] × 0.408                              (÷ λ=2.45 MJ/kg)

Calibración por píxel (período de traslape 1981-2016)
-------------------------------------------------------
PISCOp PET usa Penman-Monteith completo (Allen et al., 1998), método más
preciso pero que requiere humedad, viento y radiación.  Para corregir el
sesgo sistemático de HS respecto a PM, se computa un factor de escala por
píxel y mes:

  k_m[i,j] = Σ_{y=1981}^{2016} PET_PM[m,y,i,j]
            / Σ_{y=1981}^{2016} ET₀_HS[m,y,i,j]

  PET_cal[d,i,j] = k_{month(d)}[i,j] × ET₀_HS[d,i,j]

Este enfoque (HS calibrado) es equivalente a la "corrección de sesgo
mensual" y es metodológicamente aceptado (Droogers & Allen, 2002).

Validación en período de traslape
----------------------------------
Se reporta KGE, Pearson-r, RMSE y PBIAS entre PET_cal y PET_PM (1981-2016)
tanto a nivel de media de cuenca diaria como a nivel mensual acumulado.

Referencias
-----------
Hargreaves & Samani (1985). doi:10.13031/2013.26773
Allen et al. (1998). FAO Irrigation and Drainage Paper 56.
Droogers & Allen (2002). doi:10.1023/A:1015508305735
Aybar et al. (2020). doi:10.1038/s41597-020-00829-y

Productos
---------
  B2_pet_hargreaves_cal.nc    — grilla PET calibrada, 1981-2020, 0.01°
  B2_pet_basin_mean_hscal.csv — media ponderada coseno de cuenca
  B2_pisco_basin_mean_all.csv — columna 'pet' actualizada 2017-2020
"""
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("pet_hs")

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
TMAX_NC     = ROOT / "data/bronze/B2_tmax_basin_grid_v12.nc"
TMIN_NC     = ROOT / "data/bronze/B2_tmin_basin_grid_v12.nc"
PET_REF_NC  = ROOT / "data/bronze/B2_pet_basin_grid.nc"      # PISCOp v1.1 (1981-2016)
OUT_NC      = ROOT / "data/bronze/B2_pet_hargreaves_cal.nc"
OUT_MEAN    = ROOT / "data/bronze/B2_pet_basin_mean_hscal.csv"
OUT_ALL     = ROOT / "data/bronze/B2_pisco_basin_mean_all.csv"

T0     = pd.Timestamp("1981-01-01")
N_DAYS = 14610  # 1981-01-01 → 2020-12-31 (40 años, con bisiestos)
CAL_END_YEAR = 2016  # último año de PISCOp PET disponible


# ── 1. Radiación extraterrestre Rₐ ───────────────────────────────────────────
def _ra_mm(doy: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
    """
    Rₐ [mm/día] para cada (día, latitud).
    doy     : (T,)   — día del año (1-366)
    lat_rad : (n_lat,) — latitud en radianes
    Retorna : (T, n_lat) — Rₐ en mm/día equivalente hídrico
    """
    J  = doy[:, None]                              # (T, 1) para broadcast
    φ  = lat_rad[None, :]                          # (1, n_lat)

    dr  = 1 + 0.033 * np.cos(2 * np.pi * J / 365)            # Ec.23 FAO-56
    δ   = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)         # Ec.24 FAO-56
    ωs  = np.arccos(np.clip(-np.tan(φ) * np.tan(δ), -1, 1))  # Ec.25 FAO-56

    # Ec.21 FAO-56 [MJ/m²/día]
    Ra_mj = ((24 * 60 / np.pi) * 0.0820 * dr
             * (ωs * np.sin(φ) * np.sin(δ)
                + np.cos(φ) * np.cos(δ) * np.sin(ωs)))

    return Ra_mj * 0.408   # MJ/m²/día → mm/día  (λ = 2.45 MJ/kg)


# ── 2. ET₀ Hargreaves-Samani ──────────────────────────────────────────────────
def hargreaves_samani(tmax: np.ndarray, tmin: np.ndarray,
                      Ra_mm: np.ndarray) -> np.ndarray:
    """
    ET₀_HS = 0.0023 × Rₐ × √(ΔT) × (T̄ + 17.8)   [mm/día]
    tmax, tmin, Ra_mm : (T, n_lat, n_lon)
    """
    Tmean = (tmax + tmin) / 2.0
    TD    = np.maximum(tmax - tmin, 0.0)    # rango diurno ≥ 0
    return 0.0023 * Ra_mm * np.sqrt(TD) * (Tmean + 17.8)


# ── 3. Factor de calibración k mensual por píxel ─────────────────────────────
def compute_calibration_factors(et0_hs: np.ndarray, pet_ref: np.ndarray,
                                dates: pd.DatetimeIndex,
                                cal_end_year: int) -> np.ndarray:
    """
    k_m[i,j] = Σ PET_PM[m,y,i,j] / Σ ET₀_HS[m,y,i,j]
    para y en 1981-cal_end_year, por mes m.

    Retorna: k (12, n_lat, n_lon)
    """
    n_lat, n_lon = et0_hs.shape[1], et0_hs.shape[2]
    k = np.ones((12, n_lat, n_lon), dtype="f8")   # default = 1 (sin corrección)

    cal_idx = dates.year <= cal_end_year
    dates_c = dates[cal_idx]
    hs_c    = et0_hs[cal_idx]
    ref_c   = pet_ref[cal_idx]

    for m in range(1, 13):
        idx = dates_c.month == m
        sum_ref = np.nansum(ref_c[idx], axis=0)   # (n_lat, n_lon)
        sum_hs  = np.nansum(hs_c[idx],  axis=0)
        # Evitar división por cero: si HS ≈ 0 mantener k=1
        with np.errstate(divide="ignore", invalid="ignore"):
            k_m = np.where(sum_hs > 1e-6, sum_ref / sum_hs, 1.0)
        k[m - 1] = k_m

    log.info(f"Factor k mensual (media cuenca): "
             + "  ".join(f"M{m+1}={k[m].mean():.3f}" for m in range(12)))
    return k


# ── 4. Métricas de validación ─────────────────────────────────────────────────
def validation_metrics(obs: np.ndarray, sim: np.ndarray, label: str):
    """KGE, Pearson-r, RMSE, PBIAS sobre series 1D."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) == 0:
        log.warning(f"[{label}] Sin datos para validación")
        return

    # Kling-Gupta Efficiency (Gupta et al., 2009)
    r    = float(np.corrcoef(o, s)[0, 1])
    alpha = s.std() / (o.std() + 1e-9)           # razón de variabilidades
    beta  = s.mean() / (o.mean() + 1e-9)          # sesgo relativo
    kge   = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    rmse  = float(np.sqrt(np.mean((o - s)**2)))
    pbias = float((s.sum() - o.sum()) / o.sum() * 100)

    log.info(f"[{label}]  KGE={kge:.3f}  r={r:.3f}  "
             f"RMSE={rmse:.3f} mm/d  PBIAS={pbias:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Cargar Tmax y Tmin ────────────────────────────────────────────────────
    log.info("Cargando Tmax y Tmin (PISCOt v1.2)...")
    ds_tx = nc.Dataset(TMAX_NC)
    lat_sub = np.array(ds_tx["lat"][:])
    lon_sub = np.array(ds_tx["lon"][:])
    mask    = np.array(ds_tx["basin_mask"][:]).astype(bool)
    tmax    = np.array(ds_tx["tmax"][:])   # (T, n_lat, n_lon)
    ds_tx.close()

    ds_tn  = nc.Dataset(TMIN_NC)
    tmin   = np.array(ds_tn["tmin"][:])   # (T, n_lat, n_lon)
    ds_tn.close()

    assert tmax.shape == tmin.shape, "Tmax y Tmin deben tener la misma forma"
    T, n_lat, n_lon = tmax.shape
    log.info(f"Grilla: {n_lat}lat × {n_lon}lon × {T}t")

    dates = pd.date_range(T0, periods=T, freq="D")
    assert T == N_DAYS, f"Se esperaban {N_DAYS} días, se encontraron {T}"

    # ── Rₐ: radiación extraterrestre por día y latitud ────────────────────────
    log.info("Calculando Rₐ (FAO-56)...")
    doy     = np.array(dates.dayofyear, dtype=float)      # (T,)
    lat_rad = np.radians(lat_sub.astype(float))            # (n_lat,)
    Ra_lat  = _ra_mm(doy, lat_rad)                         # (T, n_lat)
    Ra      = Ra_lat[:, :, None] * np.ones((1, 1, n_lon))  # (T, n_lat, n_lon)

    # ── ET₀ Hargreaves-Samani (sin calibrar) ──────────────────────────────────
    log.info("Calculando ET₀_HS (sin calibrar)...")
    et0_hs = hargreaves_samani(tmax, tmin, Ra)             # (T, n_lat, n_lon)

    # ── Cargar PISCOp PET de referencia (1981-2016) ───────────────────────────
    log.info("Cargando PISCOp PET de referencia...")
    ds_ref   = nc.Dataset(PET_REF_NC)
    pet_vars = [v for v in ds_ref.variables if v.lower() not in
                ("time", "lat", "latitude", "lon", "longitude",
                 "basin_mask", "basin_frac")]
    pet_var  = pet_vars[0]
    log.info(f"  Variable PET encontrada: '{pet_var}'")

    pet_time_raw = np.array(ds_ref["time"][:])
    pet_tunits   = ds_ref["time"].units
    pet_cal      = getattr(ds_ref["time"], "calendar", "standard")
    pet_times_py = nc.num2date(pet_time_raw, pet_tunits, calendar=pet_cal,
                               only_use_cftime_datetimes=False,
                               only_use_python_datetimes=True)
    pet_dates = pd.DatetimeIndex(
        [t.strftime("%Y-%m-%d") for t in pet_times_py])

    pet_ref_raw = np.array(ds_ref[pet_var][:])    # (T_ref, n_lat, n_lon) — puede ser mayor bbox
    lat_ref     = np.array(ds_ref["lat"][:] if "lat" in ds_ref.variables
                           else ds_ref["latitude"][:])
    lon_ref     = np.array(ds_ref["lon"][:] if "lon" in ds_ref.variables
                           else ds_ref["longitude"][:])
    ds_ref.close()

    # Alinear grilla de referencia con la grilla de Tmax/Tmin
    li_ref = np.argmin(np.abs(lat_ref[:, None] - lat_sub[None, :]), axis=0)
    lj_ref = np.argmin(np.abs(lon_ref[:, None] - lon_sub[None, :]), axis=0)
    # pet_ref con misma grilla que tmax/tmin
    pet_ref_full = pet_ref_raw[:, li_ref, :][:, :, lj_ref]  # (T_ref, n_lat, n_lon)

    # Alinear temporalmente: solo fechas presentes en ambas series
    common_idx_hs  = pd.Index(dates).isin(pet_dates)
    common_idx_ref = pd.Index(pet_dates).isin(dates)
    et0_hs_cal_period = et0_hs[common_idx_hs]
    pet_ref_aligned   = pet_ref_full[common_idx_ref]
    dates_cal         = dates[common_idx_hs]
    log.info(f"  Período de traslape para calibración: "
             f"{dates_cal.min().date()} → {dates_cal.max().date()} "
             f"({len(dates_cal)} días)")

    # ── Factor de calibración k mensual por píxel ─────────────────────────────
    log.info("Calculando factores de calibración k mensual (1981-2016)...")
    k = compute_calibration_factors(et0_hs_cal_period, pet_ref_aligned,
                                    dates_cal, CAL_END_YEAR)

    # ── PET calibrado ─────────────────────────────────────────────────────────
    log.info("Aplicando calibración k a ET₀_HS (1981-2020)...")
    pet_cal_arr = np.empty_like(et0_hs)
    for m in range(1, 13):
        idx = dates.month == m
        pet_cal_arr[idx] = et0_hs[idx] * k[m - 1][None, :, :]

    # Píxeles fuera de cuenca → NaN
    pet_cal_arr = np.where(mask[None, :, :], pet_cal_arr, np.nan).astype("f4")

    # ── Validación: PET_cal vs PET_PM en 1981-2016 ───────────────────────────
    log.info("=== Validación PET_HS_cal vs PET_PM (1981-2016) ===")
    # Pesos coseno para media de cuenca
    cos_w  = np.cos(np.radians(lat_sub.astype(float)))
    w      = np.where(mask, cos_w[:, None], 0.0)
    w_flat = w.flatten()
    wb     = w_flat[mask.flatten()]
    ws     = wb.sum()

    def _basin_mean(arr3d):
        return (arr3d.reshape(arr3d.shape[0], -1)[:, mask.flatten()]
                * wb[None, :]).sum(axis=1) / ws

    pet_pm_mean  = _basin_mean(pet_ref_aligned)
    pet_hs_mean  = _basin_mean(et0_hs_cal_period)
    pet_cal_mean = _basin_mean(pet_cal_arr[common_idx_hs])

    validation_metrics(pet_pm_mean, pet_hs_mean,  "HS sin calibrar (diario)")
    validation_metrics(pet_pm_mean, pet_cal_mean, "HS calibrado    (diario)")

    # Validación mensual
    pet_pm_mon  = pd.Series(pet_pm_mean,  index=dates_cal).resample("ME").mean()
    pet_cal_mon = pd.Series(pet_cal_mean, index=dates_cal).resample("ME").mean()
    validation_metrics(pet_pm_mon.values, pet_cal_mon.values,
                       "HS calibrado (mensual)")

    # ── Guardar NC ────────────────────────────────────────────────────────────
    log.info(f"Guardando {OUT_NC.name}...")
    ds_out = nc.Dataset(OUT_NC, "w", format="NETCDF4")
    ds_out.createDimension("time", T)
    ds_out.createDimension("lat",  n_lat)
    ds_out.createDimension("lon",  n_lon)

    v_t = ds_out.createVariable("time", "f8", ("time",))
    v_t.units    = "days since 1981-01-01"
    v_t.calendar = "standard"
    v_t[:]       = np.arange(T, dtype="f8")

    v_lat = ds_out.createVariable("lat", "f4", ("lat",))
    v_lat.units = "degrees_north"; v_lat[:] = lat_sub.astype("f4")

    v_lon = ds_out.createVariable("lon", "f4", ("lon",))
    v_lon.units = "degrees_east"; v_lon[:] = lon_sub.astype("f4")

    v_mask = ds_out.createVariable("basin_mask", "i1", ("lat","lon"),
                                   zlib=True, complevel=4)
    v_mask.long_name = "Basin pixel mask (1=inside Chancay-Huaral)"
    v_mask[:]        = mask.astype("i1")

    v_pet = ds_out.createVariable(
        "pet", "f4", ("time","lat","lon"),
        zlib=True, complevel=4, shuffle=True,
        chunksizes=(30, n_lat, n_lon),
        fill_value=np.float32(np.nan),
    )
    v_pet.units      = "mm/day"
    v_pet.long_name  = "Reference ET₀ Hargreaves-Samani calibrated against PISCOp PET"
    v_pet.method     = ("Hargreaves-Samani (1985) calibrated with monthly pixel-level "
                        "scaling factor k computed from PISCOp PET (1981-2016).")
    v_pet.references = ("Hargreaves & Samani (1985) doi:10.13031/2013.26773; "
                        "Allen et al. (1998) FAO-56; "
                        "Droogers & Allen (2002) doi:10.1023/A:1015508305735")
    v_pet[:]         = pet_cal_arr

    # Guardar k como variable diagnóstica (12, n_lat, n_lon)
    v_k = ds_out.createVariable("k_calibration", "f4", zlib=True,
                                dimensions=("lat","lon"),
                                complevel=4)
    # Guardar k promedio mensual (media sobre los 12 meses)
    v_k.long_name = "Mean monthly calibration factor k (mean over 12 months)"
    v_k[:]        = k.mean(axis=0).astype("f4")

    ds_out.title      = ("PET Hargreaves-Samani calibrado — "
                         "Cuenca Chancay-Huaral (0.01°, 1981-2020)")
    ds_out.institution = "HidroAlerta Chancay-Huaral – Concurso ANA 2026"
    ds_out.history     = (f"Calculado con script 07. "
                          f"Creado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ds_out.references  = ("Hargreaves & Samani (1985); Allen et al. (1998) FAO-56; "
                          "Droogers & Allen (2002); Aybar et al. (2020)")
    ds_out.Conventions = "CF-1.8"
    ds_out.close()
    log.info(f"  NC guardado: {OUT_NC.stat().st_size / 1e6:.1f} MB")

    # ── Guardar CSV de media de cuenca ────────────────────────────────────────
    full_mean = pd.Series(_basin_mean(pet_cal_arr), index=dates, name="pet")
    full_mean.index.name = "date"
    full_mean.to_frame().to_csv(OUT_MEAN)
    log.info(f"  CSV guardado: {OUT_MEAN.name}  "
             f"(media={full_mean.mean():.3f} mm/día, "
             f"NaN={full_mean.isna().sum()})")

    # ── Actualizar basin_mean_all (solo 2017-2020) ────────────────────────────
    log.info("Actualizando pet en B2_pisco_basin_mean_all.csv (2017-2020)...")
    all_df = pd.read_csv(OUT_ALL, index_col=0, parse_dates=True)
    all_df.index = pd.DatetimeIndex(all_df.index).normalize()
    full_mean.index = pd.DatetimeIndex(full_mean.index).normalize()

    # Solo actualizar el período que faltaba (2017-2020)
    ext_idx = full_mean.index[full_mean.index.year >= 2017]
    all_df.loc[ext_idx, "pet"] = full_mean.loc[ext_idx].values
    all_df.to_csv(OUT_ALL)

    n_ext = int((all_df.index.year >= 2017).sum())
    log.info(f"  pet actualizada: {ext_idx.min().date()} → {ext_idx.max().date()} "
             f"({len(ext_idx)} días)")

    log.info("=== DONE ===")
    log.info(f"  PET media cuenca 1981-2020: {full_mean.mean():.3f} mm/día")
    log.info(f"  PET media cuenca 1981-2016: "
             f"{full_mean[full_mean.index.year <= 2016].mean():.3f} mm/día")
    log.info(f"  PET media cuenca 2017-2020: "
             f"{full_mean[full_mean.index.year >= 2017].mean():.3f} mm/día")


if __name__ == "__main__":
    main()
