#!/usr/bin/env python3
"""
Script 16: Preparar dataset de entrada para TFT (Temporal Fusion Transformer).

Estructura del dataset TFT:
  - past_observed:    pr, tmax, tmin, pet, q_m3s, api, spi_30d, spi_90d
  - known_future:     doy, month, sin/cos harmonics, is_wet_season, hydro_month,
                      clim_pr_p50, clim_pr_p90
  - static_covariates: lat, lon, elevation_m, basin_area_km2, entity_id

Salida:
  data/model_ready/D5_tft_ready.csv       -- dataset completo
  data/model_ready/D5_tft_schema.json     -- schema de columnas por rol
  outputs/figures/tft/                    -- figuras de verificacion
"""
import logging
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "16_tft_inputs.log", "w", "utf-8"),
    ],
)
logger = logging.getLogger("tft_inputs")

# ── Constantes de cuenca (valores DEM Script 10 + shapefile Script 02) ────────
BASIN_LAT       = -11.35    # centroide cuenca
BASIN_LON       = -77.05
BASIN_ELEVATION = 2678.0    # m (DEM Copernicus GLO-30 30m B4_dem_basin_30m.nc)
BASIN_AREA_KM2  = 3062.62   # km² (shapefile UTM 18S EPSG:32718)
BASIN_SLOPE_DEG = 25.0      # grados (media DEM GLO-30; SRTM 90m suavizaba → 22.2°)
BASIN_HI        = 0.506     # Integral Hipsométrica Strahler (madura)
ENTITY_ID       = "chancay_huaral"

# Estación de caudal principal
Q_STATION_COL   = "q_santo_domingo_47e214d2"
Q_CONV          = 86.4 / BASIN_AREA_KM2   # m³/s → mm/día

# ── Parámetros TFT ───────────────────────────────────────────────────────────
ENCODER_LENGTH    = 90     # días de contexto (pasado)
PREDICTION_LENGTH = 7      # días de horizonte (futuro)
API_K             = 0.85   # decay del Antecedent Precipitation Index
FOURIER_ORDERS    = [1, 2] # armónicos anuales


def compute_spi(pr_series: pd.Series, window: int) -> pd.Series:
    """SPI simplificado: z-score sobre ventana rolling con distribución gamma aproximada."""
    from scipy.stats import gamma as gamma_dist
    roll = pr_series.rolling(window, min_periods=int(window * 0.5))
    mu  = roll.mean()
    sig = roll.std()
    # Normalización simple (gamma-SPI completo requeriría ajuste por DOY)
    spi = (pr_series - mu) / (sig + 1e-6)
    return spi.clip(-3.5, 3.5)


def build_known_future(dates: pd.DatetimeIndex, clim_pr: dict) -> pd.DataFrame:
    """
    Construye las variables known_future del TFT:
    variables que conocemos con certeza para cualquier fecha futura
    (solo dependen del calendario, no de observaciones).
    """
    doy_arr   = dates.dayofyear.tolist()
    month_arr = dates.month.tolist()
    week_arr  = dates.isocalendar().week.tolist()
    year_arr  = dates.year.tolist()

    df = pd.DataFrame(index=dates)
    df["doy"]           = doy_arr
    df["month"]         = month_arr
    df["week"]          = week_arr
    df["year"]          = year_arr
    df["hydro_month"]   = [((m - 10) % 12) + 1 for m in month_arr]
    df["is_wet_season"] = [int(m in {10, 11, 12, 1, 2, 3, 4}) for m in month_arr]

    # Fourier harmonics anuales
    doy_np = np.asarray(doy_arr, dtype=np.float64)
    for k in FOURIER_ORDERS:
        angle = 2 * np.pi * k * doy_np / 365.25
        df[f"sin_doy_{k}"] = np.sin(angle).tolist()
        df[f"cos_doy_{k}"] = np.cos(angle).tolist()

    # Climatología DOY (interpolada desde dict {doy: valor})
    p50 = clim_pr.get("p50", {})
    p90 = clim_pr.get("p90", {})
    df["clim_pr_p50"] = pd.Series(
        [p50.get(d, np.nan) for d in doy_arr], index=dates, dtype=float
    ).ffill().tolist()
    df["clim_pr_p90"] = pd.Series(
        [p90.get(d, np.nan) for d in doy_arr], index=dates, dtype=float
    ).ffill().tolist()

    return df


def compute_climatology_doy(pr_train: pd.Series) -> dict:
    """Climatología por DOY (solo en train para evitar leakage)."""
    df = pr_train.to_frame("pr")
    df["doy"] = df.index.dayofyear
    clim = df.groupby("doy")["pr"].quantile([0.25, 0.50, 0.75, 0.90]).unstack()
    clim.columns = [f"p{int(q*100)}" for q in [0.25, 0.50, 0.75, 0.90]]
    result = {}
    for col in clim.columns:
        result[col] = clim[col].to_dict()
    return result


def temporal_split(df: pd.DataFrame, train=0.70, val=0.15):
    """Split 70/15/15 temporal sin data leakage."""
    n = len(df)
    n_train = int(n * train)
    n_val   = int(n * val)
    df = df.copy()
    df["split"] = "test"
    df.iloc[:n_train, df.columns.get_loc("split")] = "train"
    df.iloc[n_train:n_train+n_val, df.columns.get_loc("split")] = "val"
    return df


def main():
    out_dir     = PROJECT_ROOT / "data" / "model_ready"
    fig_dir     = PROJECT_ROOT / "outputs" / "figures" / "tft"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SCRIPT 16: Preparacion de inputs TFT")
    logger.info("=" * 70)

    # ── 1. Cargar datos PISCO basin-mean ─────────────────────────────────────
    # Preferir versión calibrada QM/delta (Script 19); fallback a original
    pisco_csv_corr = PROJECT_ROOT / "data" / "bronze" / "B2_pisco_basin_mean_corrected.csv"
    pisco_csv_orig = PROJECT_ROOT / "data" / "bronze" / "B2_pisco_basin_mean_all.csv"
    pisco_csv = pisco_csv_corr if pisco_csv_corr.exists() else pisco_csv_orig
    if pisco_csv.exists():
        pisco = pd.read_csv(pisco_csv, index_col=0, parse_dates=True)
        # Drop *_orig columns if present (QM output includes originals for reference)
        pisco = pisco[[c for c in pisco.columns if not c.endswith("_orig")]]
        logger.info(f"PISCO basin-mean cargado: {pisco_csv.name}")
        logger.info(f"  Columnas: {list(pisco.columns)}")
        logger.info(f"  Periodo:  {pisco.index.min().date()} a {pisco.index.max().date()}")
    else:
        # Fallback: usar B1 (solo precipitacion)
        b1_path = PROJECT_ROOT / "data" / "bronze" / "B1_pisco_daily_bronze.csv"
        if b1_path.exists():
            raw = pd.read_csv(b1_path, index_col=0, parse_dates=True)
            pisco = pd.DataFrame({
                "pr":   raw.get("pr_mm", raw.get("Prec", pd.Series(dtype=float))),
                "tmax": raw.get("tmax_c", np.nan),
                "tmin": raw.get("tmin_c", np.nan),
                "pet":  raw.get("pet_mm", np.nan),
            }, index=raw.index)
            logger.warning(f"B2 no encontrado. Usando B1 ({b1_path.name})")
        else:
            logger.error("Sin datos PISCO. Correr script 02 primero.")
            sys.exit(1)

    # Estandarizar nombres de columnas
    pisco = pisco.rename(columns={"Prec": "pr", "PET": "pet"})

    # ── 2. Cargar caudal SNIRH (Santo Domingo 47e214d2) ──────────────────────
    q_snirh = PROJECT_ROOT / "data" / "silver" / "snirh" / "S1_snirh_daily_q.csv"
    q_series = None
    if q_snirh.exists():
        q_df = pd.read_csv(q_snirh, index_col=0, parse_dates=True)
        if Q_STATION_COL in q_df.columns:
            q_series = q_df[Q_STATION_COL].rename("q_m3s")
            logger.info(f"Caudal cargado: {q_snirh.name} → col={Q_STATION_COL} ({q_series.notna().sum()} obs)")
        else:
            logger.warning(f"Columna {Q_STATION_COL} no encontrada. Columnas: {q_df.columns.tolist()}")
    else:
        logger.warning(f"SNIRH no encontrado: {q_snirh}")

    # ── 2b. Cargar ONI (covariable mensual → diaria) ──────────────────────────
    oni_csv = PROJECT_ROOT / "data" / "silver" / "enso" / "S3_oni_1950_2026.csv"
    oni_series = None
    if oni_csv.exists():
        oni_df = pd.read_csv(oni_csv)
        oni_df["date"] = pd.to_datetime(oni_df["date"])
        oni_df = oni_df.set_index("date").sort_index()
        oni_series = oni_df["oni"].rename("oni_index")
        logger.info(f"ONI cargado: {oni_csv.name} ({len(oni_series)} meses)")
    else:
        logger.warning(f"ONI no encontrado: {oni_csv}")

    # ── 2c. Cargar stats suelo SoilGrids (covariables estáticas) ─────────────
    soil_csv = PROJECT_ROOT / "data" / "bronze" / "B5_soilgrids_basin_mean.csv"
    soil_static = {}
    if soil_csv.exists():
        soil_df = pd.read_csv(soil_csv)
        top = soil_df[soil_df["depth"] == "0-5cm"].set_index("property")["basin_mean"]
        soil_static = {
            "soil_sand_pct":  float(top.get("sand",   np.nan)) * 0.1,  # g/kg → %
            "soil_silt_pct":  float(top.get("silt",   np.nan)) * 0.1,
            "soil_clay_pct":  float(top.get("clay",   np.nan)) * 0.1,
            "soil_bdod":      float(top.get("bdod",   np.nan)),         # ya en g/cm³
            "soil_soc":       float(top.get("soc",    np.nan)),         # ya en dg/kg
            "soil_cec":       float(top.get("cec",    np.nan)) * 0.1,  # mmol→cmol
            "soil_ph":        float(top.get("phh2o",  np.nan)),         # ya en pH
        }
        logger.info(f"Suelos cargados: sand={soil_static['soil_sand_pct']:.1f}% clay={soil_static['soil_clay_pct']:.1f}%")
    else:
        logger.warning(f"SoilGrids no encontrado: {soil_csv}")

    # ── 3. Construir índice temporal completo ─────────────────────────────────
    date_min = pisco.index.min()
    date_max = pisco.index.max()
    full_idx = pd.date_range(date_min, date_max, freq="D")
    pisco    = pisco.reindex(full_idx)
    logger.info(f"Rango temporal: {date_min.date()} a {date_max.date()} ({len(full_idx)} dias)")

    # ── 4. Features pasados observados ───────────────────────────────────────
    df = pd.DataFrame(index=full_idx)
    df.index.name = "date"

    # Precipitacion
    pr_col = "pr" if "pr" in pisco.columns else pisco.columns[0]
    df["pr_mm"] = pisco[pr_col].clip(lower=0)

    # Temperatura
    df["tmax_c"]  = pisco.get("tmax", np.nan)
    df["tmin_c"]  = pisco.get("tmin", np.nan)
    df["tmean_c"] = (df["tmax_c"] + df["tmin_c"]) / 2

    # PET (usar PISCO ETP o Hargreaves como fallback)
    if "pet" in pisco.columns:
        df["pet_mm"] = pisco["pet"].clip(lower=0)
    elif not df["tmean_c"].isna().all():
        # Hargreaves-Samani (FAO-56 simplificado)
        lat_rad = np.deg2rad(BASIN_LAT)
        doy = df.index.dayofyear
        dr  = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws   = np.arccos(-np.tan(lat_rad) * np.tan(decl))
        Ra   = (24 * 60 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )
        df["pet_mm"] = 0.0023 * (df["tmean_c"] + 17.8) * np.sqrt(df["tmax_c"] - df["tmin_c"]) * Ra
        df["pet_mm"] = df["pet_mm"].clip(lower=0)
        logger.info("PET calculada con Hargreaves-Samani")
    else:
        df["pet_mm"] = np.nan

    # Caudal observado (solo disponible 2020+)
    if q_series is not None:
        df["q_m3s"] = q_series
        df["q_mm"]  = df["q_m3s"] * Q_CONV   # m³/s → mm/día (para target homogéneo con pr/pet)
    else:
        df["q_m3s"] = np.nan
        df["q_mm"]  = np.nan

    # ONI (mensual → diaria vía forward-fill)
    if oni_series is not None:
        oni_daily = oni_series.resample("D").ffill().reindex(full_idx, method="ffill")
        df["oni_index"] = oni_daily.values
    else:
        df["oni_index"] = np.nan

    # API (Antecedent Precipitation Index)
    df["api"] = df["pr_mm"].ewm(alpha=1 - API_K, adjust=False).mean()

    # SPI 30 y 90 días
    df["spi_30d"] = compute_spi(df["pr_mm"].fillna(0), 30)
    df["spi_90d"] = compute_spi(df["pr_mm"].fillna(0), 90)

    # Water year deficit (PET - PR, rolling 30d)
    df["water_deficit_30d"] = (df["pet_mm"] - df["pr_mm"]).rolling(30, min_periods=15).mean()

    # ── 5. Split temporal (antes de calcular climatología) ───────────────────
    df = temporal_split(df)
    train_mask = df["split"] == "train"
    logger.info(f"Split: train={train_mask.sum()} val={(df['split']=='val').sum()} test={(df['split']=='test').sum()}")

    # ── 6. Climatología DOY (solo en train) ──────────────────────────────────
    clim = compute_climatology_doy(df.loc[train_mask, "pr_mm"])

    # ── 7. Known future features ─────────────────────────────────────────────
    fut = build_known_future(df.index, clim)
    for col in fut.columns:
        df[col] = fut[col].values

    # ── 8. Static covariates ─────────────────────────────────────────────────
    df["entity_id"]       = ENTITY_ID
    df["lat"]             = BASIN_LAT
    df["lon"]             = BASIN_LON
    df["elevation_m"]     = BASIN_ELEVATION
    df["basin_area_km2"]  = BASIN_AREA_KM2
    df["slope_deg"]       = BASIN_SLOPE_DEG
    df["hyps_integral"]   = BASIN_HI
    # Suelos (0-5 cm) — estáticas
    for k, v in soil_static.items():
        df[k] = v

    # ── 9. Targets ────────────────────────────────────────────────────────────
    df["pr_next_1d"]      = df["pr_mm"].shift(-1)
    df["pr_sum_next_3d"]  = df["pr_mm"].shift(-1).rolling(3).sum().shift(-(3-1))
    df["pr_sum_next_7d"]  = df["pr_mm"].shift(-1).rolling(7).sum().shift(-(7-1))
    # Target caudal (si disponible)
    if not df["q_m3s"].isna().all():
        df["q_next_1d"]     = df["q_m3s"].shift(-1)
        df["q_sum_next_7d"] = df["q_m3s"].shift(-1).rolling(7).sum().shift(-(7-1))

    # ── 10. Guardar ───────────────────────────────────────────────────────────
    out_csv = out_dir / "D5_tft_ready.csv"
    df.to_csv(out_csv)
    logger.info(f"\nDataset TFT guardado: {out_csv.name}")
    logger.info(f"  Filas: {len(df)}  Columnas: {len(df.columns)}")
    logger.info(f"  Periodo: {df.index.min().date()} a {df.index.max().date()}")

    # ── 11. Schema JSON ───────────────────────────────────────────────────────
    past_cols = ["pr_mm", "tmax_c", "tmin_c", "tmean_c", "pet_mm",
                 "q_m3s", "q_mm", "oni_index",
                 "api", "spi_30d", "spi_90d", "water_deficit_30d"]
    future_cols = [c for c in df.columns if c.startswith(("doy", "month", "week", "hydro",
                                                            "sin_", "cos_", "is_wet",
                                                            "clim_pr"))]
    soil_cols = list(soil_static.keys())
    static_cols = (["entity_id", "lat", "lon", "elevation_m", "basin_area_km2",
                    "slope_deg", "hyps_integral"] + soil_cols)
    target_cols = [c for c in df.columns if c.startswith(("pr_next", "pr_sum", "q_next", "q_sum"))]

    schema = {
        "past_observed":      [c for c in past_cols if c in df.columns],
        "known_future":       future_cols,
        "static_covariates":  static_cols,
        "targets":            target_cols,
        "split_column":       "split",
        "time_column":        "date",
        "entity_column":      "entity_id",
        "encoder_length":     ENCODER_LENGTH,
        "prediction_length":  PREDICTION_LENGTH,
        "tft_params": {
            "hidden_size":          64,
            "lstm_layers":          2,
            "num_attention_heads":  4,
            "dropout":              0.1,
            "learning_rate":        3e-4,
            "batch_size":           64,
            "max_epochs":           100,
        },
        "note": (
            "known_future contiene solo variables calendario/Fourier/climatologia. "
            "NO incluir observaciones futuras reales (leakage). "
            "q_m3s disponible solo desde 2020-09-01 (SNIRH Santo Domingo). "
            "oni_index: ONI mensual NOAA interpolado a diario (past_observed). "
            "static: DEM SRTM-90m (Script10) + SoilGrids 0-5cm (Script11)."
        ),
    }

    schema_path = out_dir / "D5_tft_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    logger.info(f"Schema guardado: {schema_path.name}")

    # ── 12. Figura de verificacion ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        df["pr_mm"].plot(ax=axes[0], color="steelblue", lw=0.5, alpha=0.8)
        axes[0].set_ylabel("Pr (mm/d)"); axes[0].set_title("Precipitacion PISCO - Cuenca Chancay-Huaral")

        if not df["tmax_c"].isna().all():
            df[["tmax_c", "tmin_c"]].plot(ax=axes[1], lw=0.5)
            axes[1].set_ylabel("T (degC)")
        else:
            axes[1].text(0.5, 0.5, "Temperatura no disponible", transform=axes[1].transAxes, ha="center")

        df["pet_mm"].plot(ax=axes[2], color="orange", lw=0.5)
        axes[2].set_ylabel("PET (mm/d)")

        if not df["q_m3s"].isna().all():
            df["q_m3s"].plot(ax=axes[3], color="darkblue", lw=0.8)
            axes[3].set_ylabel("Q (m3/s)")
        else:
            axes[3].text(0.5, 0.5, "Caudal solo disponible 2020+", transform=axes[3].transAxes, ha="center")

        # Marcar splits
        for ax in axes:
            val_start = df[df["split"] == "val"].index.min()
            test_start = df[df["split"] == "test"].index.min()
            ax.axvline(val_start, color="orange", lw=1.5, ls="--", alpha=0.7)
            ax.axvline(test_start, color="red", lw=1.5, ls="--", alpha=0.7)
            ax.grid(True, alpha=0.3)

        axes[0].legend(["Pr", "Val start", "Test start"], loc="upper right")
        plt.tight_layout()
        fig_path = fig_dir / "D5_tft_overview.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figura guardada: {fig_path.name}")
    except Exception as e:
        logger.warning(f"Error generando figura: {e}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN TFT INPUT")
    logger.info(f"  past_observed  ({len(schema['past_observed'])}): {schema['past_observed']}")
    logger.info(f"  known_future   ({len(schema['known_future'])}): {schema['known_future']}")
    logger.info(f"  static         ({len(schema['static_covariates'])}): {schema['static_covariates']}")
    logger.info(f"  targets        ({len(schema['targets'])}): {schema['targets']}")
    logger.info(f"  encoder={ENCODER_LENGTH}d  prediction={PREDICTION_LENGTH}d")
    logger.info(f"  NaN en q_m3s: {df['q_m3s'].isna().sum()} / {len(df)} ({df['q_m3s'].isna().mean()*100:.1f}%)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
