#!/usr/bin/env python3
"""
Script 07b: Recompute basin-mean PET usando temperaturas corregidas (Script 19 delta-T).

Motivación
----------
Script 07 usó PISCOt Tmax/Tmin sin corregir. Script 19 identificó que PISCO
subestima Tmin en 2-3°C y tiene sesgo estacional en Tmax. Dado que:

  PET_HS ∝ DTR^0.5 × (Tmean + 17.8)   donde DTR = Tmax - Tmin

Corregir Tmin (sube 2-3°C) reduce DTR → reduce PET_HS. Físicamente correcto:
un clima más cálido en la noche → menor evapotranspiración potencial diurna.

Enfoque
-------
Opera sobre series de media de cuenca (no la grilla completa) porque la corrección
delta-T de Script 19 es espacialmente uniforme (una delta por mes). Por linealidad
del promedio: E[HS(T+δ)] ≈ HS(E[T+δ]) para δ pequeño.

Parámetros:
  Ra calculada en el centroide de la cuenca (lat=-11.35°)
  k mensual recalibrado contra PISCOp PM PET (1981-2016)

Salidas:
  B2_pisco_basin_mean_corrected.csv — columna 'pet' reemplazada con PET corregida
  B6_pet_comparison.csv             — original vs corregida por mes (métricas y valores)
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
        logging.FileHandler(ROOT / "outputs" / "07b_pet_corregida.log", "w", "utf-8"),
    ],
)
log = logging.getLogger("pet_07b")

# ── Rutas ──────────────────────────────────────────────────────────────────────
# NOTA: Este script ya produjo B2_pet_basin_mean_hscal.csv (output vigente).
# PET_PM_CSV está en _superseded/ — archivo de referencia PM 1981-2016 usado
# para calibrar los coeficientes Hargreaves mensuales. No borrar.
CORRECTED_CSV = ROOT / "data/bronze/B2_pisco_basin_mean_corrected.csv"
PET_PM_CSV    = ROOT / "data/bronze/_superseded/B2_pet_basin_mean.csv"   # ref PM 1981-2016
OUT_COMP_CSV  = ROOT / "data/bronze/B6_pet_comparison.csv"

BASIN_LAT_DEG = -11.35   # centroide cuenca (para Ra)
CAL_END_YEAR  = 2016     # último año con PISCOp PM PET


# ── Radiación extraterrestre Rₐ (FAO-56) ──────────────────────────────────────
def ra_mm(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Rₐ [mm/día] en el centroide de la cuenca. Ec. 21-25 Allen et al. 1998."""
    phi = np.radians(lat_deg)
    dr  = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    d   = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws  = np.arccos(np.clip(-np.tan(phi) * np.tan(d), -1, 1))
    Ra_mj = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(ws)
    )
    return Ra_mj * 0.408   # MJ/m²/d → mm/d


def hargreaves_samani(tmax: np.ndarray, tmin: np.ndarray,
                      Ra: np.ndarray) -> np.ndarray:
    tmean = (tmax + tmin) / 2.0
    dtr   = np.maximum(tmax - tmin, 0.0)
    return 0.0023 * Ra * np.sqrt(dtr) * (tmean + 17.8)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return np.nan
    r     = float(np.corrcoef(o, s)[0, 1])
    alpha = s.std() / (o.std() + 1e-9)
    beta  = s.mean() / (o.mean() + 1e-9)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    return float((s.sum() - o.sum()) / (o.sum() + 1e-9) * 100)


def main():
    log.info("=" * 70)
    log.info("SCRIPT 07b: PET con temperatura corregida")
    log.info("=" * 70)

    # ── 1. Cargar forzantes corregidos ─────────────────────────────────────────
    log.info("[1] Cargando B2_pisco_basin_mean_corrected.csv ...")
    df = pd.read_csv(CORRECTED_CSV, index_col=0, parse_dates=True)
    df = df[[c for c in df.columns if not c.endswith("_orig")]]
    tmax_c = df["tmax"].values          # corregido Script 19
    tmin_c = df["tmin"].values
    pet_orig = df["pet"].values         # PET original Script 07 (T sin corregir)
    dates    = df.index
    T        = len(dates)
    log.info(f"   Período: {dates[0].date()} → {dates[-1].date()}  ({T} días)")
    log.info(f"   Tmax corr media: {np.nanmean(tmax_c):.2f}°C  "
             f"Tmin corr media: {np.nanmean(tmin_c):.2f}°C")

    # ── 2. Rₐ y ET₀ HS con T corregida ────────────────────────────────────────
    log.info("[2] Calculando ET₀ Hargreaves-Samani con T corregida ...")
    doy   = dates.dayofyear.values.astype(float)
    Ra    = ra_mm(doy, BASIN_LAT_DEG)
    et0_c = hargreaves_samani(tmax_c, tmin_c, Ra)

    # Referencia: ET₀ con T original (para comparar)
    tmax_orig_col = df.get("tmax_orig", None)
    tmin_orig_col = df.get("tmin_orig", None)
    # reconstruir T original = T corregida - delta (no disponible directamente)
    # Usar pet_orig como proxy de ET₀ original: se cargó arriba

    log.info(f"   ET₀_HS corr media: {np.nanmean(et0_c):.3f} mm/día")

    # ── 3. Cargar PISCOp PM PET de referencia (1981-2016) ─────────────────────
    log.info("[3] Cargando PISCOp PM PET (referencia calibración) ...")
    pet_pm_df = pd.read_csv(PET_PM_CSV, index_col=0, parse_dates=True)
    # Detectar columna de PET
    pet_pm_col = [c for c in pet_pm_df.columns if "pet" in c.lower()][0]
    pet_pm = pet_pm_df[pet_pm_col].rename("pet_pm")
    # Normalizar índice (puede venir con timestamp 12:00:00 del NetCDF)
    pet_pm.index = pd.DatetimeIndex(pet_pm.index).normalize()
    log.info(f"   PM PET: {pet_pm.index[0].date()} → {pet_pm.index[-1].date()}  "
             f"media={pet_pm.mean():.3f} mm/día")

    # ── 4. Factor de calibración k mensual (1981-2016) ─────────────────────────
    log.info("[4] Ajustando factor k mensual (ET₀_HS_corr vs PM_PET 1981-2016) ...")
    et0_series  = pd.Series(et0_c, index=dates, name="et0_hs_corr")
    common_idx  = pet_pm.index.intersection(et0_series.index)
    cal_idx     = common_idx[common_idx.year <= CAL_END_YEAR]
    log.info(f"   Período calibración: {cal_idx.min().date()} → {cal_idx.max().date()} "
             f"({len(cal_idx)} días)")

    k_monthly = {}
    for m in range(1, 13):
        pm_m  = pet_pm.loc[cal_idx[cal_idx.month == m]]
        et0_m = et0_series.loc[cal_idx[cal_idx.month == m]]
        sum_pm  = pm_m.sum()
        sum_et0 = et0_m.sum()
        k = float(sum_pm / sum_et0) if sum_et0 > 1e-6 else 1.0
        k_monthly[m] = k

    log.info("   k mensual: " +
             "  ".join(f"M{m}={k:.3f}" for m, k in k_monthly.items()))

    # ── 5. PET calibrada con T corregida ──────────────────────────────────────
    log.info("[5] Aplicando k mensual ...")
    pet_corr = np.empty(T, dtype="f8")
    for m in range(1, 13):
        idx = dates.month == m
        pet_corr[idx] = et0_c[idx] * k_monthly[m]

    pet_corr_series = pd.Series(pet_corr, index=dates, name="pet_corr")

    log.info(f"   PET original (Script 07) media: {np.nanmean(pet_orig):.3f} mm/día")
    log.info(f"   PET corregida (07b) media:       {np.nanmean(pet_corr):.3f} mm/día")
    log.info(f"   Diferencia media: {np.nanmean(pet_corr - pet_orig):+.3f} mm/día")

    # ── 6. Validación contra PM PET (período de traslape) ─────────────────────
    log.info("[6] Validación vs PISCOp PM PET (1981-2016) ...")
    pet_corr_al = pet_corr_series.reindex(cal_idx)
    pet_pm_al   = pet_pm.reindex(cal_idx)
    pet_orig_al = pd.Series(pet_orig, index=dates, name="pet_orig").reindex(cal_idx)

    log.info(f"   KGE PET_orig  vs PM: {kge(pet_pm_al.values, pet_orig_al.values):.3f}  "
             f"PBIAS={pbias(pet_pm_al.values, pet_orig_al.values):+.1f}%")
    log.info(f"   KGE PET_corr  vs PM: {kge(pet_pm_al.values, pet_corr_al.values):.3f}  "
             f"PBIAS={pbias(pet_pm_al.values, pet_corr_al.values):+.1f}%")

    # ── 7. Comparación mensual ─────────────────────────────────────────────────
    log.info("[7] Generando tabla comparativa mensual ...")
    df_comp_list = []
    for m in range(1, 13):
        idx_m = dates.month == m
        df_comp_list.append({
            "month": m,
            "pet_orig_mean":  float(np.nanmean(pet_orig[idx_m])),
            "pet_corr_mean":  float(np.nanmean(pet_corr[idx_m])),
            "delta_pet_mean": float(np.nanmean(pet_corr[idx_m] - pet_orig[idx_m])),
            "k_factor":       k_monthly[m],
        })
    df_comp = pd.DataFrame(df_comp_list)
    df_comp.to_csv(OUT_COMP_CSV, index=False)
    log.info(f"   Guardado: {OUT_COMP_CSV.name}")
    log.info("\n" + df_comp.to_string(index=False, float_format="{:.3f}".format))

    # ── 8. Estrategia híbrida: PM PET (1981-2016) + HS(corrT) (2017-2020) ──────
    # PM PET es Penman-Monteith completo → no depende de T directamente → mantener
    # HS(corrT) solo para el período de extensión donde PM PET no está disponible
    log.info("[8] Estrategia híbrida: PM PET 1981-2016 + HS(corrT,cal) 2017-2020 ...")

    pet_hybrid = pet_corr.copy()
    # Restaurar PM PET para 1981-2016 (mejor estimación disponible)
    pm_pet_norm = pet_pm.reindex(dates)   # NaN para 2017-2020 (sin datos PM)
    mask_pm_avail = pm_pet_norm.notna().values
    pet_hybrid[mask_pm_avail] = pm_pet_norm.values[mask_pm_avail]

    n_pm   = mask_pm_avail.sum()
    n_hs   = (~mask_pm_avail).sum()
    log.info(f"   PM PET (1981-2016): {n_pm} días | HS(corrT,cal) (2017-2020): {n_hs} días")
    log.info(f"   PET hybrid media:  {np.nanmean(pet_hybrid):.3f} mm/día")

    # Comparar HS(corrT) vs HS(origT) para período de extensión (2017-2020)
    ext_mask = ~mask_pm_avail
    pet_orig_ext = pet_orig[ext_mask]
    pet_corr_ext = pet_hybrid[ext_mask]
    log.info(f"   Extensión 2017-2020 — HS(origT): {np.nanmean(pet_orig_ext):.3f} mm/día  "
             f"→ HS(corrT): {np.nanmean(pet_corr_ext):.3f} mm/día  "
             f"Δ={np.nanmean(pet_corr_ext - pet_orig_ext):+.3f} mm/día")

    df["pet"] = pet_hybrid
    df.to_csv(CORRECTED_CSV)
    log.info(f"   CSV guardado: {CORRECTED_CSV.name}")

    # ── Resumen ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("RESUMEN PET CORREGIDA")
    log.info(f"  Estrategia: PM PET para 1981-2016 | HS(corrT,cal) para 2017-2020")
    log.info(f"  PET hybrid media 1981-2020: {np.nanmean(pet_hybrid):.3f} mm/día")
    log.info(f"  Efecto en 2017-2020: HS(corrT) vs HS(origT): "
             f"{np.nanmean(pet_corr_ext - pet_orig_ext):+.3f} mm/día")
    log.info(f"  k mensual (HS→PM calibración): "
             + " ".join(f"M{m}={k_monthly[m]:.3f}" for m in range(1, 13)))
    log.info("  Salidas: B6_pet_comparison.csv, B2_pisco_basin_mean_corrected.csv")
    log.info("  SIGUIENTE: python scripts/20_subcuencas_multientidad.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
