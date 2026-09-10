#!/usr/bin/env python3
"""
Script 73: LightGBM AR + Dinámicas Satelitales + ERA5 swvl — target Q.

Extiende al campeón (Script 65, NSE≈0.97) con features satelitales D7 y
humedad de suelo ERA5 swvl1-4, todo por sub-cuenca. Flujo integrado:
  1. Verifica/re-construye D7 extendido a 2025 via Script 51 (por defecto)
  2. Extrae ERA5 swvl1-4 por entidad de era5land_daily*.nc (inline, vectorizado)
  3. Wide pivot: SPATIAL + SAT + swvl por entidad (≈184 features)
  4. LightGBM AR + sat (mismos hiperparámetros que Script 65 para comparación justa)
  5. SHAP por categoría + métricas completas + figura comparativa SAT03

Features nuevas vs Script 65 (per entity × 9 sub-cuencas):
  SAT (D7): snow_cover_pct, ndvi_mean, lswi_mean, lst_day_K, et_mm8d,
            sm_surface, sm_rootzone
  ERA5:     swvl1 (0-7cm), swvl2 (7-28cm), swvl3 (28-100cm), swvl4 (100-289cm)

NaN estructural (pre-lanzamiento sensor; LightGBM maneja via fillna(0)):
  MODIS vars  : NaN antes 2000-02-24 (≈48-63% de filas de train)
  SMAP sm_*   : NaN antes 2015-03-31 (≈85% de filas de train)
  ERA5 swvl   : 0% NaN (completo 1981-2025)

Splits idénticos a Script 65: train ≤ 2015-12-31, test ≥ 2021-01-01.

Uso:
    python scripts/05_models/73_lgbm_sat_Q.py            # verifica D7, extrae swvl, entrena
    python scripts/05_models/73_lgbm_sat_Q.py --build_d7  # fuerza re-build D7 antes de entrenar
    python scripts/05_models/73_lgbm_sat_Q.py --no_build_d7  # no toca D7

Salidas:
    outputs/ml_Q/lgbm_sat_results.csv
    outputs/ml_Q/era5_swvl_per_entity.parquet  (caché)
    outputs/figures/ml_Q/SAT01_shap_{1d,7d}.png
    outputs/figures/ml_Q/SAT02_hydrograph.png
    outputs/figures/ml_Q/SAT03_comparison.png
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("lgbm_sat")

D7_CSV    = ROOT / "data/model_ready/D7_multientity.csv"
QOBS      = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
ERA5_DIR  = ROOT / "data/raw/era5/daily"
SHP_SUB   = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/subcuencas"
             / "Cuenca_Chancay___Huaral_SUBCUENCAS_juansuyo_931381206_geogpsperu_UH_MENORES.shp")
SCRIPT_51  = ROOT / "scripts/08_gee/51_build_satellite_features.py"
# Ambos scripts usan .venv (Python 3.14): lightgbm, shap, rasterstats, geopandas
# .venv313 (Python 3.13) = solo PyTorch/Lightning (TFT, LSTM)
OUT_DIR   = ROOT / "outputs/ml_Q"
FIG_DIR   = ROOT / "outputs/figures/ml_Q"
SWVL_CACHE = OUT_DIR / "era5_swvl_per_entity.parquet"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

Q_CONV = 86.4 / 3062.62   # m³/s → mm/d
Q90    = 40.89             # m³/s (Q obs real 47E214D2)
SEED   = 42

ENTITIES = ["sub_634", "sub_640", "sub_641", "sub_646",
            "sub_649", "sub_650", "sub_653", "sub_655", "sub_656"]

SPATIAL     = ["pr_mm", "tmax_c", "tmin_c", "pet_mm", "api",
               "spi_30d", "spi_90d", "water_deficit_30d"]
SAT_SPATIAL = ["snow_cover_pct", "ndvi_mean", "lswi_mean", "lst_day_K",
               "et_mm8d", "sm_surface", "sm_rootzone"]
SWVL_VARS   = ["swvl1", "swvl2", "swvl3", "swvl4"]
SHARED      = ["oni_index", "sin_doy_1", "cos_doy_1", "hydro_month", "is_wet_season"]

try:
    from entity_labels import ENTITY_META
    SUB_NAMES = {e: m["nombre"] for e, m in ENTITY_META.items()}
except Exception:
    SUB_NAMES = {}

# Nombres legibles para SHAP
VAR_ES = {
    "pr_mm": "Precip", "tmax_c": "Tmax", "tmin_c": "Tmin", "pet_mm": "ETP",
    "api": "API", "spi_30d": "SPI30", "spi_90d": "SPI90",
    "water_deficit_30d": "DéficitH", "oni_index": "ONI",
    "q_mm": "Q hoy", "q_lag1": "Q ayer", "q_lag7": "Q -7d",
    "q_roll7": "Q media7d", "q_roll30": "Q media30d",
    "pr_basin": "Precip cuenca", "pr_roll7": "Precip acum7d",
    "pr_roll30": "Precip acum30d",
    "sin_doy_1": "Estacional", "cos_doy_1": "Estacional2",
    "hydro_month": "Mes hidro", "is_wet_season": "T.húmeda",
    "snow_cover_pct": "Nieve", "ndvi_mean": "NDVI", "lswi_mean": "LSWI",
    "lst_day_K": "LST día", "et_mm8d": "ET-MODIS", "sm_surface": "SMAP-sup",
    "sm_rootzone": "SMAP-raíz",
    "swvl1": "swvl1(0-7cm)", "swvl2": "swvl2(7-28cm)",
    "swvl3": "swvl3(28-100cm)", "swvl4": "swvl4(100-289cm)",
}


def pretty(f):
    for e, n in SUB_NAMES.items():
        if f.endswith("_" + e):
            base = f[:-(len(e) + 1)]
            sname = n[:12] if n else e
            return f"{VAR_ES.get(base, base)}·{sname}"
    return VAR_ES.get(f, f)


# ── ERA5 swvl extraction ───────────────────────────────────────────────────────

def _entity_masks(lats, lons, gdf):
    """Índices ERA5 (i,j) por entidad. Usa centroide si ningún pixel cae dentro."""
    masks = {}
    for eid, row in gdf.iterrows():
        pts = [
            (i, j)
            for i, la in enumerate(lats)
            for j, lo in enumerate(lons)
            if row.geometry.contains(Point(float(lo), float(la)))
        ]
        if not pts:
            cx, cy = row.geometry.centroid.coords[0]
            i = int(np.argmin(np.abs(lats - cy)))
            j = int(np.argmin(np.abs(lons - cx)))
            pts = [(i, j)]
            log.debug(f"  {eid}: centroide ERA5 ({float(lats[i]):.2f}, {float(lons[j]):.2f})")
        masks[eid] = np.array(pts, dtype=int)   # (K, 2)
        log.info(f"  {eid}: {len(pts)} píxeles ERA5 swvl")
    return masks


def extract_era5_swvl(era5_dir, gdf):
    """Extrae swvl1-4 (m³/m³) por entidad. Retorna DataFrame (date, entity_id) × SWVL_VARS.

    Las longitudes ERA5 pueden venir en formato 0-360 → se convierten a -180..180.
    _FillValue = NaN → ya está manejado por numpy masked array. GRIB_missingValue ≈ 3.4e38 → enmascarado.
    """
    nc_files = sorted(era5_dir.glob("era5land_daily_????_????.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No era5land_daily_*.nc en {era5_dir}")

    # Leer malla (constante para todos los archivos)
    with nc.Dataset(str(nc_files[0])) as ds:
        lats = np.array(ds["latitude"][:])
        # ERA5 almacena longitudes en 0-360 para hemis. oeste → convertir a -180..180
        lons_raw = np.array(ds["longitude"][:])
        lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)

    log.info(f"ERA5 grid: {len(lats)}×{len(lons)} | "
             f"lat [{lats.min():.2f}, {lats.max():.2f}] "
             f"lon [{lons.min():.2f}, {lons.max():.2f}]")
    log.info("Pre-computando máscaras por entidad...")
    entity_masks = _entity_masks(lats, lons, gdf)

    records = []
    for nc_file in nc_files:
        log.info(f"  ERA5 swvl: {nc_file.name}")
        with nc.Dataset(str(nc_file)) as ds:
            cal  = getattr(ds["time"], "calendar", "standard")
            t_nc = nc.num2date(np.array(ds["time"][:]), ds["time"].units, calendar=cal)
            dates = pd.to_datetime([t.isoformat()[:10] for t in t_nc])

            # Cargar 4 vars (T, H, W) — fill values → NaN
            arrs = {}
            for v in SWVL_VARS:
                raw = np.array(ds[v][:], dtype=np.float32)
                raw[raw > 1e10] = np.nan   # GRIB_missingValue ≈ 3.4e38
                arrs[v] = raw              # (T, H, W); _FillValue=NaN ya queda NaN

        for eid, idxs in entity_masks.items():
            row_dict = {"date": dates, "entity_id": eid}
            for v, arr in arrs.items():
                # arr[:, idxs[:,0], idxs[:,1]] → (T, K) → nanmean → (T,)
                row_dict[v] = np.nanmean(arr[:, idxs[:, 0], idxs[:, 1]], axis=1)
            records.append(pd.DataFrame(row_dict))

    swvl_df = pd.concat(records, ignore_index=True)
    swvl_df["date"] = pd.to_datetime(swvl_df["date"])
    swvl_df = (swvl_df
               .sort_values(["date", "entity_id"])
               .set_index(["date", "entity_id"]))
    nan_total = swvl_df.isna().sum().sum()
    log.info(f"ERA5 swvl: {swvl_df.shape} | NaN total: {nan_total}")
    return swvl_df


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(D7_ext):
    """Wide pivot 1 fila/día con SPATIAL + SAT + swvl per entity + AR + PR lags.

    D7_ext columnas: date, entity_id, SPATIAL, SAT_SPATIAL, SWVL_VARS, SHARED, targets, q_mm.
    """
    all_pivot_vars = SPATIAL + [c for c in SAT_SPATIAL + SWVL_VARS
                                 if c in D7_ext.columns]
    pivots = []
    for c in all_pivot_vars:
        p = D7_ext.pivot_table(index="date", columns="entity_id",
                               values=c, aggfunc="mean")
        p.columns = [f"{c}_{e}" for e in p.columns]
        pivots.append(p)

    ref = D7_ext[D7_ext["entity_id"] == "sub_634"].set_index("date")
    target_cols = [c for c in ["q_mm", "q_next_1d", "q_sum_next_7d"]
                   if c in ref.columns]
    df = pd.concat(pivots + [ref[SHARED + target_cols]], axis=1).sort_index()

    # Q autoregresivo (r(1)=0.85 — dominante para 1d)
    q = df["q_mm"]
    df["q_lag1"]   = q.shift(1)
    df["q_lag7"]   = q.shift(7)
    df["q_roll7"]  = q.rolling(7,  min_periods=3).mean()
    df["q_roll30"] = q.rolling(30, min_periods=10).mean()

    # Precipitación media cuenca + acumulados
    pr_cols = [c for c in df.columns if c.startswith("pr_mm_")]
    df["pr_basin"]  = df[pr_cols].mean(axis=1)
    df["pr_roll7"]  = df["pr_basin"].rolling(7,  min_periods=3).sum()
    df["pr_roll30"] = df["pr_basin"].rolling(30, min_periods=10).sum()

    return df


# ── Métricas (idénticas a Script 65 / 72) ─────────────────────────────────────

def _nse(o, p):
    return 1 - np.sum((o - p)**2) / (np.sum((o - o.mean())**2) + 1e-12)

def _nse_sqrt(o, p):
    w = np.maximum(o, 0)**0.5
    return 1 - np.sum(w*(o-p)**2) / (np.sum(w*(o-o.mean())**2) + 1e-12)

def _kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    a = p.std() / (o.std() + 1e-12)
    b = p.mean() / (o.mean() + 1e-12)
    return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)

def compute_metrics(o, p, thr):
    eo = o > thr; ep = p > thr
    TP = int(np.sum(eo & ep)); FP = int(np.sum(~eo & ep))
    FN = int(np.sum(eo & ~ep)); TN = int(np.sum(~eo & ~ep))
    POD = TP / (TP + FN + 1e-9)
    FAR = FP / (FP + TN + 1e-9)
    CSI = TP / (TP + FP + FN + 1e-9)
    j   = 0.25*_nse_sqrt(o,p) + 0.25*_nse(o,p) + 0.30*CSI + 0.10*POD - 0.10*FAR
    return {
        "NSE":      round(_nse(o, p),      4),
        "NSE_sqrt": round(_nse_sqrt(o, p), 4),
        "KGE":      round(_kge(o, p),      4),
        "J_alert":  round(j,               4),
        "POD":      round(POD, 3),
        "FAR":      round(FAR, 3),
        "CSI":      round(CSI, 3),
        "N":        len(o),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LightGBM AR + sat + ERA5 swvl")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build_d7",    action="store_true",
                       help="Forzar re-build D7→2025 via Script 51 (aunque D7 ya exista)")
    group.add_argument("--no_build_d7", action="store_true",
                       help="No tocar D7 aunque no llegue a 2025")
    args = parser.parse_args()

    # ── 1. Verificar/re-construir D7 ────────────────────────────────────────
    d7_end = pd.Timestamp("1900-01-01")
    if D7_CSV.exists():
        tmp    = pd.read_csv(D7_CSV, usecols=["date"], parse_dates=["date"])
        d7_end = tmp["date"].max()
        log.info(f"D7 existente: {tmp.shape[0]:,} filas | hasta {d7_end.date()}")

    needs_build = d7_end < pd.Timestamp("2024-12-31")

    if args.build_d7 or (needs_build and not args.no_build_d7):
        reason = "forzado por --build_d7" if args.build_d7 else f"D7 termina en {d7_end.date()}"
        log.warning(f"Re-construyendo D7 extendido a 2025 ({reason}). Esto tarda ~30-60 min...")
        subprocess.run(
            [sys.executable, str(SCRIPT_51), "--merge_d6", "--force"],
            check=True, cwd=str(ROOT),
        )
        log.info("D7 re-construido.")
    elif needs_build and args.no_build_d7:
        log.warning(f"D7 termina en {d7_end.date()} y --no_build_d7 activo. "
                    "Las features satelitales serán NaN en test 2021-2025.")

    # ── 2. Cargar D7 ────────────────────────────────────────────────────────
    log.info("Cargando D7...")
    D7 = pd.read_csv(D7_CSV, parse_dates=["date"])
    log.info(f"D7: {D7.shape} | {D7.date.min().date()} → {D7.date.max().date()}")

    # ── 3. ERA5 swvl1-4 por entidad ─────────────────────────────────────────
    import geopandas as gpd
    gdf = gpd.read_file(str(SHP_SUB)).to_crs(epsg=4326)
    gdf["entity_id"] = "sub_" + gdf["ID"].astype(str)
    gdf = gdf[gdf["entity_id"].isin(ENTITIES)].set_index("entity_id")[["geometry"]]

    if SWVL_CACHE.exists() and not args.build_d7:
        log.info("Cargando ERA5 swvl desde caché...")
        swvl_df = pd.read_parquet(SWVL_CACHE)
    else:
        log.info("Extrayendo ERA5 swvl1-4 por sub-cuenca...")
        swvl_df = extract_era5_swvl(ERA5_DIR, gdf)
        swvl_df.to_parquet(SWVL_CACHE)
        log.info(f"ERA5 swvl caché → {SWVL_CACHE.name} "
                 f"({SWVL_CACHE.stat().st_size/1e6:.1f} MB)")

    # ── 4. Merge D7 + swvl ──────────────────────────────────────────────────
    D7 = D7.set_index(["date", "entity_id"]).join(swvl_df, how="left").reset_index()
    log.info(f"D7 + swvl: {D7.shape} | swvl NaN: "
             f"{D7['swvl1'].isna().sum():,} / {len(D7):,} filas")

    # ── 5. Build features ───────────────────────────────────────────────────
    log.info("Construyendo features wide...")
    df    = build_features(D7)
    dates = df.index
    log.info(f"Wide: {df.shape}")

    m_tr = dates <= "2015-12-31"
    m_te = dates >= "2021-01-01"
    log.info(f"Train: {m_tr.sum():,}d | Test: {m_te.sum():,}d")

    feat = [c for c in df.columns if c not in ["q_next_1d", "q_sum_next_7d"]]

    # Clasificar features para SHAP coloring
    ar_set   = {c for c in feat if c.startswith("q_")}
    sat_set  = {c for c in feat if any(c.startswith(s + "_") for s in SAT_SPATIAL)}
    swvl_set = {c for c in feat if any(c.startswith(s + "_") for s in SWVL_VARS)}

    log.info(f"Features: {len(feat)} total "
             f"(AR={len(ar_set)}, Sat={len(sat_set)}, swvl={len(swvl_set)}, "
             f"meteo/otros={len(feat)-len(ar_set)-len(sat_set)-len(swvl_set)})")

    # ── 6. Cargar Q obs real ─────────────────────────────────────────────────
    from lightgbm import LGBMRegressor
    import shap
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    qobs      = pd.read_csv(QOBS, index_col=0, parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm   = qobs * Q_CONV
    qobs_next1 = qobs_mm.shift(-1)
    qobs_sum7  = qobs_mm.shift(-1).rolling(7).sum().shift(-6)

    results = []; preds_out = {}; shap_data = {}

    # ── 7. Entrenar y evaluar ────────────────────────────────────────────────
    for target, obs_real in [("q_next_1d", qobs_next1), ("q_sum_next_7d", qobs_sum7)]:
        y  = df[target].values
        tr = m_tr & np.isfinite(y)

        # fillna(0) = mismo tratamiento que Script 65 (baseline consistente)
        # NaN satelitales pre-lanzamiento → 0 (sentinel; modelo aprende "0 = sin dato")
        Xtr = df.loc[tr, feat].fillna(0)
        ytr = y[tr]

        log.info(f"[{target}] Entrenando — Xtr: {Xtr.shape}...")
        model = LGBMRegressor(
            n_estimators=500, num_leaves=31, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
        model.fit(Xtr, ytr)

        # Test honesto vs Q obs real
        Xte  = df.loc[m_te, feat].fillna(0)
        pred = pd.Series(model.predict(Xte), index=dates[m_te])
        o    = obs_real.reindex(dates[m_te])
        valid = o.notna()
        thr  = Q90 * Q_CONV * (7 if "7d" in target else 1)
        m    = compute_metrics(o[valid].values, pred[valid].values, thr)
        m["target"] = target
        results.append(m)

        log.info(f"[{target}] NSE={m['NSE']} NSE_sqrt={m['NSE_sqrt']} "
                 f"KGE={m['KGE']} J_alert={m['J_alert']} "
                 f"POD={m['POD']:.0%} FAR={m['FAR']:.0%} CSI={m['CSI']:.3f} N={m['N']}")

        preds_out[target] = pd.DataFrame({
            "date":   dates[m_te][valid.values],
            "q_obs":  o[valid].values / (Q_CONV * (7 if "7d" in target else 1)),
            "q_pred": pred[valid].values / (Q_CONV * (7 if "7d" in target else 1)),
        })

        # SHAP (muestra representativa de test)
        expl = shap.TreeExplainer(model)
        Xsh  = Xte.sample(min(500, len(Xte)), random_state=SEED)
        shap_data[target] = (expl.shap_values(Xsh), Xsh)

    pd.DataFrame(results).to_csv(OUT_DIR / "lgbm_sat_results.csv", index=False)
    log.info(f"Resultados → lgbm_sat_results.csv")

    # ── 8. SHAP por categoría ────────────────────────────────────────────────
    def _color(f):
        if f in ar_set or f.startswith("q_"):       return "#e74c3c"  # rojo — AR caudal
        if f in swvl_set or f.startswith("swvl"):   return "#27ae60"  # verde — ERA5 swvl
        if f in sat_set:                              return "#f39c12"  # naranja — satelital
        return "#3498db"                                                # azul — meteo

    legend_patches = [
        Patch(fc="#e74c3c", label="Q autoregresivo"),
        Patch(fc="#f39c12", label="Satelital (MODIS/SMAP)"),
        Patch(fc="#27ae60", label="ERA5 swvl (humedad suelo)"),
        Patch(fc="#3498db", label="Meteorología / calendario"),
    ]

    for target in ["q_next_1d", "q_sum_next_7d"]:
        sv, Xsh = shap_data[target]
        imp = (pd.Series(np.abs(sv).mean(0), index=Xsh.columns)
               .sort_values(ascending=True).tail(20))
        colors = [_color(f) for f in imp.index]

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh([pretty(f) for f in imp.index], imp.values, color=colors, alpha=0.85)
        ax.set_xlabel("Importancia SHAP media |valor|")
        tname = "Q +1 día" if target == "q_next_1d" else "Q suma 7 días"
        ax.set_title(f"SHAP — LightGBM AR + Satelital · {tname}\n"
                     f"Script 73: features por sub-cuenca", fontweight="bold")
        ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        tag = "1d" if target == "q_next_1d" else "7d"
        fig.savefig(FIG_DIR / f"SAT01_shap_{tag}.png", bbox_inches="tight")
        plt.close(fig)
        log.info(f"  → SAT01_shap_{tag}.png")

    # ── 9. Hidrograma test ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for ax, target, r in zip(axes, ["q_next_1d", "q_sum_next_7d"], results):
        p = preds_out[target].sort_values("date")
        ax.fill_between(p["date"], 0, p["q_obs"],  alpha=0.12, color="#c0392b")
        ax.plot(p["date"], p["q_obs"],  color="#c0392b", lw=1.4, label="Q obs real")
        ax.plot(p["date"], p["q_pred"], color="#2ecc71", lw=1.2, ls="--",
                label="LightGBM AR+Sat")
        ax.axhline(Q90, ls=":", color="orange", alpha=0.7,
                   label=f"Q90 = {Q90:.1f} m³/s")
        tname = "Q +1 día (m³/s)" if target == "q_next_1d" else "Q suma 7d (m³/s)"
        ax.set_ylabel(tname); ax.set_ylim(0)
        ax.set_title(
            f"{target}: NSE={r['NSE']:.3f}  KGE={r['KGE']:.3f}  "
            f"POD={r['POD']:.0%}  J_alert={r['J_alert']:.3f}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle(
        "LightGBM AR + Satelital + ERA5 swvl — test Q obs real 2021-2025",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "SAT02_hydrograph.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  → SAT02_hydrograph.png")

    # ── 10. Comparativa vs Script 65 ─────────────────────────────────────────
    baseline_csv = OUT_DIR / "lgbm_ar_results.csv"
    if baseline_csv.exists():
        base = pd.read_csv(baseline_csv).set_index("target")
        new  = pd.DataFrame(results).set_index("target")
        cmp_metrics = ["NSE", "NSE_sqrt", "KGE", "J_alert", "POD", "FAR", "CSI"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, target in zip(axes, ["q_next_1d", "q_sum_next_7d"]):
            if target not in base.index or target not in new.index:
                ax.set_visible(False); continue
            bv = base.loc[target, cmp_metrics].values.astype(float)
            nv = new.loc[target,  cmp_metrics].values.astype(float)
            x  = np.arange(len(cmp_metrics)); w = 0.35
            ax.bar(x - w/2, bv, w, label="Scr 65 — D6 (sin sat)",  color="#3498db", alpha=0.8)
            ax.bar(x + w/2, nv, w, label="Scr 73 — D7+swvl",       color="#27ae60", alpha=0.8)
            for xi, (b, n) in enumerate(zip(bv, nv)):
                delta = n - b
                col = "#27ae60" if delta >= 0 else "#e74c3c"
                ax.annotate(f"{delta:+.3f}",
                            (xi + w/2, max(b, n) + 0.02),
                            ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(cmp_metrics, rotation=30, ha="right")
            ax.set_ylim(-0.15, 1.15)
            tname = "Q +1 día" if target == "q_next_1d" else "Q suma 7 días"
            ax.set_title(tname, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylabel("Métrica")
        fig.suptitle(
            "Comparativa: Script 65 (D6) vs Script 73 (D7 + ERA5 swvl)\n"
            "Test = Q obs real 2021-2025 | Δ sobre barra derecha",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "SAT03_comparison.png", bbox_inches="tight")
        plt.close(fig)
        log.info("  → SAT03_comparison.png")

        log.info("\n=== DELTA vs Script 65 (D6 sin sat) ===")
        for target in ["q_next_1d", "q_sum_next_7d"]:
            if target not in base.index:
                continue
            log.info(f"  [{target}]")
            for mn in ["NSE", "NSE_sqrt", "J_alert", "POD", "CSI"]:
                if mn not in base.columns or mn not in new.columns:
                    continue
                b_val = base.loc[target, mn]
                n_val = new.loc[target,  mn]
                arrow = "▲" if n_val >= b_val else "▼"
                log.info(f"    {mn:12s}: {b_val:.3f} → {n_val:.3f}  {arrow} {n_val-b_val:+.3f}")
    else:
        log.warning("lgbm_ar_results.csv no encontrado — omitiendo SAT03.")

    log.info("\n=== RESUMEN FINAL — LightGBM AR + Satelital + ERA5 swvl ===")
    log.info(
        pd.DataFrame(results)[["target", "NSE", "NSE_sqrt", "KGE",
                                "J_alert", "POD", "FAR", "CSI", "N"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
