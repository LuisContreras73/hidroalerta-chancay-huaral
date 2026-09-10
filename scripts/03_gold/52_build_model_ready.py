#!/usr/bin/env python3
"""
Script 52: Build model-ready splits — D6_Q y D7_Q

Genera carpetas data/model_ready/splits/{D6_Q,D7_Q}/ con:
  train.parquet / val.parquet / test.parquet
  scaler.joblib          — StandardScaler fit SOLO en train (PROTOCOLO §3)
  scaler_meta.json       — {feature: {mean, std, log1p: bool}}
  sensor_masks.json      — {satellite_feature: launch_date} (D7 únicamente)

Genera data/model_ready/thresholds/q_thresholds.json con umbrales de alerta
calibrados con Q observado real (47E214D2), no con GR4J.

Anti-leakage garantizado:
  - StandardScaler.fit() SOLO con train split
  - log1p aplicado antes del fit (sin look-ahead)
  - NaN → 0 para fit del scaler (no imputa, solo evita error)
  - Targets NO se escalan (LSTM/TFT usan GroupNormalizer internamente)
  - ffill de satelitales aplicado POR SPLIT para no cruzar fronteras temporales
  - Q90/Q99 calibrados con Q observado real, independientemente del scaler

Uso:
    python scripts/03_gold/52_build_model_ready.py            # D6_Q + D7_Q
    python scripts/03_gold/52_build_model_ready.py --only d6  # solo D6_Q
    python scripts/03_gold/52_build_model_ready.py --only d7  # solo D7_Q
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("build_model_ready")

# ── Rutas ──────────────────────────────────────────────────────────────────────
D6_CSV     = ROOT / "data/model_ready/D6_multientity.csv"
D7_CSV     = ROOT / "data/model_ready/D7_multientity.csv"
Q_CSV      = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
SPLITS_DIR = ROOT / "data/model_ready/splits"
THRESH_DIR = ROOT / "data/model_ready/thresholds"
Q_STATION  = "q_santo_domingo_47e214d2"
Q_CONV     = 86.4 / 3062.62  # mm/d → m³/s inverse: m3s = q_mm / Q_CONV

# ── Configuración de features ──────────────────────────────────────────────────
# Columnas que NO se escalan (IDs, metadata, booleanas, targets, enteros calendario)
SKIP_SCALE = {
    "date", "entity_id", "split",
    "doy", "month", "week", "hydro_month", "is_wet_season",
    "esa_lc_majority", "mcd_lc_2023",
    # Targets — no escalar, LSTM/TFT normalizan internamente con GroupNormalizer
    "pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d",
    "q_next_1d", "q_sum_next_7d",
}

# Features con distribución muy sesgada (skew > 2) → log1p antes del StandardScaler
LOG1P_COLS = {
    "pr_mm",        # skew 3.34
    "chirps_pr_mm", # skew 3.68 (D7)
    "api",          # skew 2.28
    "q_mm",         # skew 8.86 (GR4J en train, obs en test)
    "et_mm8d",      # skew positivo, pre-2001 NaN → log1p cuando > 0
}

# Fecha de lanzamiento de sensores satelitales (para sensor_masks.json en D7)
SENSOR_LAUNCH = {
    "snow_cover_pct":  "2000-02-24",
    "ndvi_mean":       "2000-02-24",
    "evi_mean":        "2000-02-24",
    "lswi_mean":       "2000-02-24",
    "lst_day_K":       "2000-02-24",
    "lst_night_K":     "2000-02-24",
    "et_mm8d":         "2001-01-01",
    "pet_mm8d":        "2001-01-01",
    "sm_surface":      "2015-03-31",
    "sm_rootzone":     "2015-03-31",
    "jrc_water_occ":   "1984-03-01",
    "lsat_ndvi":       "1984-03-01",
    "lsat_ndwi":       "1984-03-01",
    "lsat_mndwi":      "1984-03-01",
    "s1_vv":           "2015-04-03",
    "s1_vh":           "2015-04-03",
    "s1_ratio":        "2015-04-03",
    "s2_ndvi":         "2017-03-28",
    "s2_ndwi":         "2017-03-28",
    "s2_ndsi":         "2017-03-28",
    "chirps_pr_mm":    "1981-01-01",
    "esa_lc_majority": "2020-01-01",
    "mcd_lc_2023":     "2001-01-01",
}


# ── Umbrales de alerta Q (calibrados con Q obs real) ──────────────────────────
def build_q_thresholds() -> dict:
    """Calcula umbrales Q90/Q99 con Q observado real 47E214D2."""
    try:
        df_q = pd.read_csv(Q_CSV, index_col=0, parse_dates=True)
        if Q_STATION not in df_q.columns:
            raise KeyError(Q_STATION)
        q_obs = df_q[Q_STATION].dropna()   # m³/s
        thresholds = {
            "station":   Q_STATION,
            "n_obs_days": int(len(q_obs)),
            "period":    f"{q_obs.index.min().date()} / {q_obs.index.max().date()}",
            "source":    "Q observado SNIRH 47E214D2 — NO GR4J",
            "Q50":  float(round(q_obs.quantile(0.50), 2)),
            "Q75":  float(round(q_obs.quantile(0.75), 2)),
            "Q90":  float(round(q_obs.quantile(0.90), 2)),   # alerta naranja
            "Q95":  float(round(q_obs.quantile(0.95), 2)),
            "Q99":  float(round(q_obs.quantile(0.99), 2)),   # alerta roja
            "Qmax": float(round(q_obs.max(), 2)),
            "Q_mean": float(round(q_obs.mean(), 2)),
            "Q_std":  float(round(q_obs.std(), 2)),
            "Q75_ecologico_m3s": float(round(q_obs.quantile(0.75), 2)),
            "alert_orange_label": "Q > Q90 (exceedance frecuencia 10%)",
            "alert_red_label":    "Q > Q99 (exceedance frecuencia 1%)",
        }
        log.info(f"   Q obs: n={len(q_obs)}, Q90={thresholds['Q90']} m³/s, Q99={thresholds['Q99']} m³/s")
    except Exception as e:
        log.warning(f"   No se pudo cargar Q obs: {e} — usando valores hardcoded de G1_q_sim_gr4j")
        thresholds = {
            "station": Q_STATION, "source": "HARDCODED — revisar",
            "Q90": 40.9, "Q99": 77.7, "Q_mean": 17.1,
        }
    return thresholds


# ── Aplicar ffill por split (anti-leakage) ────────────────────────────────────
def apply_ffill_per_split(df: pd.DataFrame, sat_cols: list[str]) -> pd.DataFrame:
    """Forward-fill satelitales DENTRO de cada (entity, split) — nunca cruza fronteras."""
    df = df.copy()
    for col in sat_cols:
        if col in df.columns:
            df[col] = (
                df.groupby(["entity_id", "split"])[col]
                .transform(lambda s: s.ffill())
            )
    return df


# ── Constructor principal ──────────────────────────────────────────────────────
def build_splits(df: pd.DataFrame, name: str, sat_cols: list[str] | None = None) -> dict:
    """
    Genera splits parquet + scaler + metadata para un dataset.
    Retorna dict con estadísticas del proceso.
    """
    out_dir = SPLITS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corregir ffill por split (Riesgo D de TRAINING_PREP.md)
    if sat_cols:
        log.info(f"   [{name}] Aplicando ffill per-split en {len(sat_cols)} features satelitales ...")
        df = apply_ffill_per_split(df, sat_cols)

    # 2. Separar splits
    train = df[df["split"] == "train"].copy().reset_index(drop=True)
    val   = df[df["split"] == "val"].copy().reset_index(drop=True)
    test  = df[df["split"] == "test"].copy().reset_index(drop=True)
    log.info(f"   [{name}] splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # 3. Identificar columnas a escalar
    num_cols = [
        c for c in df.columns
        if c not in SKIP_SCALE
        and df[c].dtype in ("float64", "float32", "int64", "int32")
        and c not in {"doy", "month", "week", "hydro_month"}
    ]

    # 4. Aplicar log1p a features sesgadas (ANTES del scaler, usando solo la mask de cols disponibles)
    log1p_applied = []
    for col in LOG1P_COLS:
        if col in df.columns and col in num_cols:
            for split_df in [train, val, test]:
                split_df[col] = np.log1p(split_df[col].clip(lower=0))
            log1p_applied.append(col)
    if log1p_applied:
        log.info(f"   [{name}] log1p aplicado en: {log1p_applied}")

    # 5. Fit del scaler SOLO en train (NaN → 0 únicamente para fit, no imputa datos)
    scaler = StandardScaler()
    scaler.fit(train[num_cols].fillna(0.0))
    log.info(f"   [{name}] StandardScaler fit en train ({len(num_cols)} features)")

    # 6. Transform todos los splits (NaN permanece NaN — no se imputa)
    def scale_split(split_df: pd.DataFrame) -> pd.DataFrame:
        out = split_df.copy()
        scaled = scaler.transform(split_df[num_cols].fillna(0.0))
        # Restaurar NaN donde estaban (scaler no debería imputar)
        nan_mask = split_df[num_cols].isna()
        scaled_df = pd.DataFrame(scaled, columns=num_cols, index=split_df.index)
        scaled_df[nan_mask] = np.nan
        out[num_cols] = scaled_df
        return out

    train_s = scale_split(train)
    val_s   = scale_split(val)
    test_s  = scale_split(test)

    # 7. Guardar parquet
    train_s.to_parquet(out_dir / "train.parquet", index=False)
    val_s.to_parquet(out_dir / "val.parquet", index=False)
    test_s.to_parquet(out_dir / "test.parquet", index=False)
    log.info(f"   [{name}] Parquets guardados en {out_dir.name}/")

    # 8. Guardar scaler
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # 9. Metadata del scaler (para reproducibilidad e interpretabilidad)
    scaler_meta = {
        col: {
            "mean":    float(scaler.mean_[i]),
            "std":     float(scaler.scale_[i]),
            "log1p":   col in log1p_applied,
            "train_nan_pct": float(train[col].isna().mean() * 100),
        }
        for i, col in enumerate(num_cols)
    }
    with open(out_dir / "scaler_meta.json", "w") as f:
        json.dump(scaler_meta, f, indent=2)

    # 10. Sensor masks para D7
    if sat_cols:
        masks = {col: SENSOR_LAUNCH.get(col, "1981-01-01") for col in sat_cols}
        with open(out_dir / "sensor_masks.json", "w") as f:
            json.dump(masks, f, indent=2)

    # 11. Feature manifest (para modelos)
    skip_now = SKIP_SCALE | {"split"}
    past_obs = [c for c in df.columns if c not in skip_now and "next" not in c
                and c not in {"sin_doy_1","cos_doy_1","sin_doy_2","cos_doy_2",
                               "clim_pr_p50","clim_pr_p90","oni_index",
                               "doy","month","week","hydro_month","is_wet_season",
                               "lat","lon","elevation_m","basin_area_km2",
                               "slope_deg","hyps_integral"} | {f"soil_{x}" for x in ["sand_pct","silt_pct","clay_pct","bdod","soc","cec","ph"]}]
    known_fut = ["sin_doy_1","cos_doy_1","sin_doy_2","cos_doy_2",
                 "clim_pr_p50","clim_pr_p90","oni_index",
                 "doy","month","week","hydro_month","is_wet_season"]
    static_cv = ["lat","lon","elevation_m","basin_area_km2","slope_deg","hyps_integral",
                 "soil_sand_pct","soil_silt_pct","soil_clay_pct","soil_bdod","soil_soc","soil_cec","soil_ph"]
    targets_q = [c for c in df.columns if "q_next" in c or "q_sum" in c]
    targets_pr = [c for c in df.columns if "pr_next" in c or "pr_sum" in c]
    if sat_cols:
        static_cv += [c for c in sat_cols if "lc" in c or "majority" in c]

    manifest = {
        "dataset": name,
        "n_rows":  {"train": len(train), "val": len(val), "test": len(test)},
        "past_observed": [c for c in past_obs if c in df.columns],
        "known_future":  [c for c in known_fut if c in df.columns],
        "static_covariates": [c for c in static_cv if c in df.columns],
        "targets_Q":  targets_q,
        "targets_pr": targets_pr,
        "log1p_cols": log1p_applied,
        "scaled_cols": num_cols,
        "encoder_length_recommended": 90,
        "prediction_length_recommended": 7,
        "note": (
            "q_mm en train = GR4J reconstruction (fisica, no sintetico). "
            "Evaluacion final SIEMPRE sobre Q observado real (2020-09+). "
            "NaN pre-lanzamiento en satelitales = ausencia real, no imputar con cero."
        ),
    }
    with open(out_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"   [{name}] feature_manifest.json guardado")

    return {
        "name": name,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "scaled_cols": len(num_cols),
        "log1p_cols": log1p_applied,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Script 52: Build model-ready splits")
    parser.add_argument("--only", choices=["d6", "d7"], default=None,
                        help="Procesar solo D6 o D7 (default: ambos)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SCRIPT 52: Build model-ready splits (anti-leakage)")
    log.info("=" * 70)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    THRESH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Umbrales Q (antes de los splits, independiente)
    log.info("\n[1] Calculando umbrales Q con datos observados reales ...")
    thresholds = build_q_thresholds()
    with open(THRESH_DIR / "q_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"   Guardado: {THRESH_DIR / 'q_thresholds.json'}")

    results = []

    # 2. D6_Q
    if args.only in (None, "d6"):
        log.info("\n[2] Procesando D6_Q ...")
        D6 = pd.read_csv(D6_CSV, parse_dates=["date"])
        log.info(f"   Cargado: {len(D6)} filas, {D6.shape[1]} cols")
        res = build_splits(D6, name="D6_Q", sat_cols=None)
        results.append(res)

    # 3. D7_Q
    if args.only in (None, "d7"):
        log.info("\n[3] Procesando D7_Q ...")
        D7 = pd.read_csv(D7_CSV, parse_dates=["date"])
        log.info(f"   Cargado: {len(D7)} filas, {D7.shape[1]} cols")
        # Features satelitales de D7 (delta vs D6)
        D6_cols = set(pd.read_csv(D6_CSV, nrows=0).columns)
        sat_cols = [c for c in D7.columns if c not in D6_cols]
        log.info(f"   Satelitales D7: {len(sat_cols)} features")
        res = build_splits(D7, name="D7_Q", sat_cols=sat_cols)
        results.append(res)

    # 4. Resumen
    log.info("\n" + "=" * 70)
    log.info("RESUMEN")
    log.info(f"  Thresholds: Q90={thresholds['Q90']} m3/s, Q99={thresholds['Q99']} m3/s")
    for r in results:
        log.info(f"  {r['name']}: train={r['rows']['train']}, val={r['rows']['val']}, "
                 f"test={r['rows']['test']} | {r['scaled_cols']} cols escaladas | "
                 f"log1p={r['log1p_cols']}")
    log.info(f"  Salida: {SPLITS_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
