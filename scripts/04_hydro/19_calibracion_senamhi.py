#!/usr/bin/env python3
"""
Script 19: Validación y corrección de sesgo PISCO vs estaciones SENAMHI
          Calibración punto-a-píxel + Quantile Mapping mensual

Metodología
-----------
1. Para cada estación SENAMHI: extraer el píxel PISCO más cercano y comparar
   con el valor observado día a día (validación punto-a-píxel).

2. Métricas de validación por estación: KGE, NSE, r, RMSE, PBIAS,
   frecuencia días lluviosos (drizzle effect), gradiente altitudinal.

3. Referencia observada de cuenca: media ponderada por IDW (1/dist²) de
   las estaciones representativas de la cuenca interior (excl. Donoso
   y Huayan por razones documentadas en B6_calibration_stats.csv).

4. Corrección Quantile Mapping (QM) mensual — precipitación:
   - Período calibración: 1984-2013 (overlap mayoría de estaciones)
   - CDF empírica con 100 cuantiles por mes
   - Extrapolación: lineal más allá del rango de calibración
   - Aplicación: serie completa 1981-2020

5. Corrección delta mensual — temperatura (Tmax, Tmin):
   - delta[m] = mean(T_obs_m) - mean(T_pisco_m)
   - T_corr = T_pisco + delta[m]

Consideración metodológica
--------------------------
PISCO asimila estaciones SENAMHI en su interpolación nacional (Aybar et al.
2020), por lo que la validación no es totalmente independiente. Sin embargo:
  - La calibración local corrige sesgos sistemáticos que la interpolación
    nacional no elimina para esta cuenca específica.
  - Los extremos (p99 PISCO=7 mm vs estaciones 11-18 mm) requieren
    corrección explícita antes de usar en modelos hidrológicos.
  - El drizzle effect (PISCO: 99% días lluviosos vs estaciones 9-35%)
    distorsiona features de ML como API, SPI, frecuencia seca/húmeda.

Salidas
-------
  data/bronze/B2_pisco_basin_mean_corrected.csv  -- serie corregida 1981-2020
  data/bronze/B6_calibration_stats.csv           -- métricas por estación
  data/bronze/B6_qm_params.csv                   -- cuantiles QM por mes
  data/bronze/B6_T_correction.csv                -- delta T mensual
  outputs/figures/basin/V09_validacion_pr.png
  outputs/figures/basin/V10_gradiente_altitudinal.png
  outputs/figures/basin/V11_qm_precipitacion.png
  outputs/figures/basin/V12_correccion_temperatura.png
"""
import datetime
import json
import logging
import sys
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
        logging.FileHandler(
            PROJECT_ROOT / "outputs" / "19_calibracion_senamhi.log", "w", "utf-8"
        ),
    ],
)
log = logging.getLogger("calibracion_senamhi")

# ── Parámetros ────────────────────────────────────────────────────────────────
CAL_START       = "1984-01-01"   # inicio período calibración (Donoso disponible)
CAL_END         = "2013-12-31"   # fin conservador (antes del cierre de estaciones)
N_QUANTILES     = 100
PRECIP_THRESH   = 0.1            # mm/día — umbral "día lluvioso"
BASIN_LAT       = -11.35         # centroide cuenca
BASIN_LON       = -77.05

# Estaciones incluidas en la referencia observada de cuenca
# Excluidas: donoso (350m, zona costera, climatología diferente al interior)
#            huayan (2800m, solo 9% días lluviosos — posible problema de datos)
STATIONS_BASIN_REF = [
    "paccho", "pachamachay", "pallac", "parquin",
    "picoy", "pirca", "santa_cruz",
]


# ─── Utilidades ──────────────────────────────────────────────────────────────

def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10:
        return np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = s.std() / (o.std() + 1e-9)
    beta  = s.mean() / (o.mean() + 1e-9)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10:
        return np.nan
    ss_res = np.sum((o - s) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    return float((s.sum() - o.sum()) / (o.sum() + 1e-9) * 100)


def pearson_r(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 5:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    return float(np.sqrt(np.mean((o - s) ** 2)))


def wet_day_freq(series: np.ndarray, thresh: float = PRECIP_THRESH) -> float:
    valid = series[np.isfinite(series)]
    return float((valid >= thresh).mean()) if len(valid) > 0 else np.nan


def idw_weights(
    station_lats: list, station_lons: list,
    target_lat: float, target_lon: float,
    power: float = 2.0,
) -> np.ndarray:
    """Pesos IDW desde cada estación hacia el punto objetivo."""
    dlat = np.array(station_lats) - target_lat
    dlon = np.array(station_lons) - target_lon
    dist2 = dlat**2 + dlon**2 + 1e-12
    w = 1.0 / dist2**power
    return w / w.sum()


# ─── 1. Cargar datos SENAMHI ─────────────────────────────────────────────────

def load_senamhi(silver_dir: Path, catalog_file: Path) -> dict:
    cat = pd.read_csv(catalog_file)
    stations = {}
    for _, row in cat.iterrows():
        sid = row["station_id"]
        f   = silver_dir / row["silver_file"]
        if not f.exists():
            log.warning(f"  [{sid}] No encontrado: {f.name}")
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df = df.sort_index()
        stations[sid] = {
            "df":   df,
            "name": row["station_name"],
            "lat":  float(row["lat"]),
            "lon":  float(row["lon"]),
            "alt":  float(row["alt_m"]),
        }
        n = df["pr_mm"].notna().sum() if "pr_mm" in df.columns else 0
        log.info(f"  [{sid}] {row['station_name']} alt={row['alt_m']:.0f}m  "
                 f"pr_obs={n} días")
    return stations


# ─── 2. Extraer píxel PISCO en ubicación de cada estación ────────────────────

def extract_pisco_at_stations(nc_path: Path, var_name: str,
                               stations: dict) -> dict:
    """Para cada estación extrae la serie temporal del píxel más cercano."""
    import netCDF4 as nc_lib

    ds = nc_lib.Dataset(nc_path, "r")
    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    time_var = ds.variables["time"]
    dates = pd.to_datetime(
        nc_lib.num2date(time_var[:], time_var.units,
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)
    )

    result = {}
    for sid, info in stations.items():
        lat_idx = int(np.argmin(np.abs(lats - info["lat"])))
        lon_idx = int(np.argmin(np.abs(lons - info["lon"])))
        pixel   = ds.variables[var_name][:, lat_idx, lon_idx]
        arr     = np.asarray(pixel, dtype=np.float32)
        arr[arr < -999] = np.nan  # fill values

        series = pd.Series(arr, index=dates, name=f"pisco_{var_name}")
        result[sid] = series
        log.info(f"  [{sid}] {var_name} pixel=({lats[lat_idx]:.2f},{lons[lon_idx]:.2f})  "
                 f"mean={np.nanmean(arr):.3f}")

    ds.close()
    return result


# ─── 3. Métricas validación punto-a-píxel ────────────────────────────────────

def compute_validation_metrics(stations: dict,
                                pisco_pr: dict,
                                pisco_tmax: dict,
                                pisco_tmin: dict) -> pd.DataFrame:
    records = []
    for sid, info in stations.items():
        df_obs = info["df"]
        # Overlap
        common_idx = df_obs.index

        for var, pisco_dict in [("pr", pisco_pr),
                                 ("tmax", pisco_tmax),
                                 ("tmin", pisco_tmin)]:
            obs_col = f"{var}_mm" if var == "pr" else f"{var}_c"
            if pisco_dict is None or sid not in pisco_dict:
                continue
            if obs_col not in df_obs.columns:
                continue

            obs_s  = df_obs[obs_col].reindex(common_idx)
            pisco_s = pisco_dict[sid].reindex(common_idx)

            o = obs_s.values
            p = pisco_s.values

            rec = {
                "station_id":   sid,
                "station_name": info["name"],
                "alt_m":        info["alt"],
                "lat":          info["lat"],
                "lon":          info["lon"],
                "variable":     var,
                "n_pairs":      int((np.isfinite(o) & np.isfinite(p)).sum()),
                "obs_mean":     float(np.nanmean(o)),
                "pisco_mean":   float(np.nanmean(p)),
                "pbias_pct":    pbias(o, p),
                "rmse":         rmse(o, p),
                "kge":          kge(o, p),
                "nse":          nse(o, p),
                "pearson_r":    pearson_r(o, p),
            }
            if var == "pr":
                rec["wet_freq_obs"]   = wet_day_freq(o)
                rec["wet_freq_pisco"] = wet_day_freq(p)
                rec["drizzle_ratio"]  = (
                    rec["wet_freq_pisco"] / (rec["wet_freq_obs"] + 1e-9)
                )
            records.append(rec)

    return pd.DataFrame(records)


# ─── 4. Referencia observada de cuenca (IDW) ─────────────────────────────────

def build_basin_obs_reference(stations: dict,
                               station_ids: list,
                               variable: str,
                               full_index: pd.DatetimeIndex) -> pd.Series:
    """
    Promedio ponderado por IDW (1/dist²) de estaciones seleccionadas.
    Resultado: serie diaria de 'observado de cuenca'.
    """
    obs_col = "pr_mm" if variable == "pr" else f"{variable}_c"

    frames = []
    lats_sel, lons_sel = [], []

    for sid in station_ids:
        if sid not in stations:
            continue
        df = stations[sid]["df"]
        if obs_col not in df.columns:
            continue
        s = df[obs_col].reindex(full_index)
        frames.append(s.rename(sid))
        lats_sel.append(stations[sid]["lat"])
        lons_sel.append(stations[sid]["lon"])

    if not frames:
        return pd.Series(np.nan, index=full_index, name=f"obs_{variable}")

    mat = pd.concat(frames, axis=1)  # (n_days, n_stations)
    weights = idw_weights(lats_sel, lons_sel, BASIN_LAT, BASIN_LON)

    # Weighted mean ignorando NaN (renormalizar pesos donde hay datos)
    values = mat.values  # (n_days, n_stations)
    result = np.full(len(full_index), np.nan)

    for i in range(len(full_index)):
        row  = values[i]
        mask = np.isfinite(row)
        if mask.sum() == 0:
            continue
        w_valid = weights[mask]
        w_valid = w_valid / w_valid.sum()
        result[i] = np.dot(row[mask], w_valid)

    return pd.Series(result, index=full_index, name=f"obs_{variable}")


# ─── 5. Quantile Mapping mensual ─────────────────────────────────────────────

def fit_qm_monthly(pisco_cal: pd.Series,
                   obs_cal: pd.Series,
                   n_q: int = N_QUANTILES) -> dict:
    """
    Ajusta funciones de QM por mes (12 meses).
    Retorna dict: mes → (pisco_quantiles, obs_quantiles)
    """
    from scipy.interpolate import interp1d

    qm_params = {}
    q_vals = np.linspace(0, 1, n_q + 1)

    for month in range(1, 13):
        p_m = pisco_cal[pisco_cal.index.month == month].dropna()
        o_m = obs_cal[obs_cal.index.month == month].dropna()

        if len(p_m) < 30 or len(o_m) < 30:
            log.warning(f"  QM mes {month}: datos insuficientes ({len(p_m)}/{len(o_m)})")
            qm_params[month] = None
            continue

        p_q = np.quantile(p_m.values, q_vals)
        o_q = np.quantile(o_m.values, q_vals)
        qm_params[month] = (p_q, o_q)
        log.info(f"  QM mes {month:02d}: p_mean={p_m.mean():.2f}→o_mean={o_m.mean():.2f}  "
                 f"p_p99={np.quantile(p_m,0.99):.1f}→o_p99={np.quantile(o_m,0.99):.1f}")

    return qm_params


def apply_qm(pisco_full: pd.Series, qm_params: dict) -> pd.Series:
    """Aplica QM mensual a la serie completa."""
    from scipy.interpolate import interp1d

    corrected = pisco_full.copy()

    for month in range(1, 13):
        params = qm_params.get(month)
        if params is None:
            continue

        p_q, o_q = params
        # Función de mapeo con extrapolación lineal
        f_map = interp1d(p_q, o_q, kind="linear",
                         bounds_error=False,
                         fill_value=(o_q[0], o_q[-1]))

        mask = pisco_full.index.month == month
        vals = pisco_full[mask].fillna(0).values
        corr_vals = np.maximum(f_map(vals), 0.0)  # no negativos

        # Restaurar NaN donde PISCO era NaN
        nan_mask = pisco_full[mask].isna().values
        corr_vals[nan_mask] = np.nan

        corrected[mask] = corr_vals

    return corrected


def save_qm_params(qm_params: dict, out_path: Path) -> None:
    records = []
    for month, params in qm_params.items():
        if params is None:
            continue
        p_q, o_q = params
        for i, (pq, oq) in enumerate(zip(p_q, o_q)):
            records.append({"month": month, "quantile_pct": i,
                            "pisco_value": pq, "obs_value": oq})
    pd.DataFrame(records).to_csv(out_path, index=False)


# ─── 6. Corrección delta temperatura ─────────────────────────────────────────

def fit_delta_T_monthly(pisco_cal: pd.Series,
                        obs_cal: pd.Series) -> pd.Series:
    """
    Delta mensual aditivo: delta[m] = mean(obs_m) - mean(pisco_m)
    Retorna Serie indexed 1..12.
    """
    deltas = {}
    for month in range(1, 13):
        p_m = pisco_cal[pisco_cal.index.month == month].dropna()
        o_m = obs_cal[obs_cal.index.month == month].dropna()
        if len(p_m) < 10 or len(o_m) < 10:
            deltas[month] = 0.0
        else:
            deltas[month] = float(o_m.mean() - p_m.mean())
    return pd.Series(deltas, name="delta_c")


def apply_delta_T(pisco_full: pd.Series, deltas: pd.Series) -> pd.Series:
    corrected = pisco_full.copy()
    for month in range(1, 13):
        mask = pisco_full.index.month == month
        corrected[mask] = pisco_full[mask] + deltas.get(month, 0.0)
    return corrected


# ─── 7. Figuras ──────────────────────────────────────────────────────────────

def plot_V09_validacion_pr(stats: pd.DataFrame, stations: dict,
                           pisco_pr: dict, fig_dir: Path) -> None:
    """V09: Validación precipitación: scatter, climatología, drizzle, sesgo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    stats_pr = stats[stats["variable"] == "pr"].set_index("station_id")
    ids_plot  = [s for s in stats_pr.index if s in stations]

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Scatter PISCO vs obs por estación (coloreado por altitud)
    ax_sc = fig.add_subplot(gs[0, :2])
    alts  = [stations[s]["alt"] for s in ids_plot if s in pisco_pr]
    norm  = plt.Normalize(min(alts), max(alts))
    cmap  = plt.cm.RdYlGn_r

    for sid in ids_plot:
        if sid not in pisco_pr:
            continue
        obs_s   = stations[sid]["df"]["pr_mm"].dropna()
        pisco_s = pisco_pr[sid].reindex(obs_s.index).dropna()
        common  = obs_s.reindex(pisco_s.index).dropna()
        pisco_c = pisco_s.reindex(common.index).dropna()
        alt     = stations[sid]["alt"]
        ax_sc.scatter(common.values, pisco_c.values,
                      s=2, alpha=0.25, color=cmap(norm(alt)),
                      label=stations[sid]["name"])

    xlim = ax_sc.get_xlim()
    ax_sc.plot([0, xlim[1]], [0, xlim[1]], "k--", lw=0.8, label="1:1")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax_sc, label="Altitud (m)", shrink=0.8)
    ax_sc.set_xlabel("Pr observada (mm/día)"); ax_sc.set_ylabel("PISCO pixel (mm/día)")
    ax_sc.set_title("(a) Scatter punto-a-píxel — todas las estaciones")
    ax_sc.set_xlim(left=0); ax_sc.set_ylim(bottom=0)

    # (b) Tabla KGE/PBIAS por estación
    ax_tb = fig.add_subplot(gs[0, 2])
    ax_tb.axis("off")
    table_data = [["Estación", "Alt(m)", "KGE", "PBIAS%", "r"]]
    for sid in ids_plot:
        if sid not in stats_pr.index:
            continue
        r = stats_pr.loc[sid]
        table_data.append([
            r["station_name"][:12],
            f"{r['alt_m']:.0f}",
            f"{r['kge']:.2f}",
            f"{r['pbias_pct']:+.0f}",
            f"{r['pearson_r']:.2f}",
        ])
    tbl = ax_tb.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    ax_tb.set_title("(b) Métricas validación pr", fontsize=9)

    # (c) Climatología mensual: PISCO vs obs por estación
    ax_cl = fig.add_subplot(gs[1, :2])
    months = range(1, 13)
    for sid in ids_plot:
        if sid not in pisco_pr or "pr_mm" not in stations[sid]["df"].columns:
            continue
        obs_m   = stations[sid]["df"]["pr_mm"].groupby(
                      stations[sid]["df"].index.month).mean()
        pisco_m = pisco_pr[sid].groupby(pisco_pr[sid].index.month).mean()
        c = cmap(norm(stations[sid]["alt"]))
        ax_cl.plot(months, [obs_m.get(m, np.nan) for m in months],
                   "-o", ms=3, color=c, alpha=0.7,
                   label=f"Obs {stations[sid]['name'][:8]}")
        ax_cl.plot(months, [pisco_m.get(m, np.nan) for m in months],
                   "--", color=c, alpha=0.5, lw=0.8)
    ax_cl.set_xticks(range(1, 13))
    ax_cl.set_xticklabels(["E","F","M","A","M","J","J","A","S","O","N","D"])
    ax_cl.set_ylabel("Pr media (mm/día)")
    ax_cl.set_title("(c) Climatología mensual — sólido=obs, guion=PISCO píxel")

    # (d) Drizzle effect: freq días lluviosos
    ax_dr = fig.add_subplot(gs[1, 2])
    pr_stats = stats_pr.reset_index()
    names = [r[:8] for r in pr_stats["station_name"]]
    x = np.arange(len(names))
    ax_dr.barh(x, pr_stats["wet_freq_pisco"] * 100, color="#e74c3c",
               alpha=0.7, label="PISCO píxel")
    ax_dr.barh(x, pr_stats["wet_freq_obs"] * 100, color="#2980b9",
               alpha=0.7, label="Observado")
    ax_dr.set_yticks(x); ax_dr.set_yticklabels(names, fontsize=7)
    ax_dr.set_xlabel("% días lluviosos (≥0.1 mm)"); ax_dr.legend(fontsize=7)
    ax_dr.set_title("(d) Drizzle effect", fontsize=9)

    # (e) PBIAS por estación (barras)
    ax_pb = fig.add_subplot(gs[2, :2])
    colors_pb = ["#e74c3c" if v > 0 else "#2980b9"
                 for v in pr_stats["pbias_pct"]]
    bars = ax_pb.bar(names, pr_stats["pbias_pct"], color=colors_pb, alpha=0.8)
    ax_pb.axhline(0, color="k", lw=0.8)
    ax_pb.bar_label(bars, fmt="%.0f%%", fontsize=7)
    ax_pb.set_ylabel("PBIAS (%)")
    ax_pb.set_title("(e) Sesgo volumétrico PISCO vs observado (+= PISCO sobreestima)")
    ax_pb.tick_params(axis="x", rotation=30, labelsize=7)

    # (f) Medias observadas vs PISCO por estación
    ax_mn = fig.add_subplot(gs[2, 2])
    ax_mn.scatter(pr_stats["obs_mean"], pr_stats["pisco_mean"],
                  s=60, c=pr_stats["alt_m"], cmap="RdYlGn_r", zorder=3)
    for _, row in pr_stats.iterrows():
        ax_mn.annotate(row["station_name"][:7],
                       (row["obs_mean"], row["pisco_mean"]),
                       fontsize=6, ha="left", va="bottom")
    lim = max(pr_stats["obs_mean"].max(), pr_stats["pisco_mean"].max()) * 1.15
    ax_mn.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax_mn.set_xlabel("Media obs (mm/día)"); ax_mn.set_ylabel("Media PISCO (mm/día)")
    ax_mn.set_title("(f) Media anual: PISCO vs observado", fontsize=9)

    fig.suptitle("V09 — Validación PISCO precipitación vs estaciones SENAMHI\n"
                 "Cuenca Chancay-Huaral | Período calibración 1984-2013",
                 fontsize=11, fontweight="bold")
    out = fig_dir / "V09_validacion_pr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura guardada: {out.name}")


def plot_V10_gradiente_altitudinal(stats: pd.DataFrame, stations: dict,
                                   fig_dir: Path) -> None:
    """V10: Gradiente altitudinal precipitación y temperatura."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("V10 — Gradiente altitudinal: PISCO vs observado\n"
                 "Cuenca Chancay-Huaral", fontsize=11, fontweight="bold")

    for ax, (var, lbl, units) in zip(
        axes.flat,
        [("pr", "Precipitación media anual", "mm/día"),
         ("pr", "PBIAS precipitación", "%"),
         ("tmax", "Temperatura máxima media", "°C"),
         ("tmin", "Temperatura mínima media", "°C")],
    ):
        sv = stats[stats["variable"] == var].set_index("station_id")
        if sv.empty:
            continue
        alts = sv["alt_m"].values
        if lbl == "PBIAS precipitación":
            y_obs   = sv["pbias_pct"].values
            ax.scatter(alts, y_obs, s=70, c="#e74c3c", zorder=3)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Altitud (m)"); ax.set_ylabel("PBIAS (%)")
        else:
            y_obs   = sv["obs_mean"].values
            y_pisco = sv["pisco_mean"].values
            ax.scatter(alts, y_obs,   s=70, c="#2980b9", label="Observado",  zorder=3)
            ax.scatter(alts, y_pisco, s=70, c="#e74c3c", label="PISCO píxel", marker="^", zorder=3)
            # Línea de tendencia observado
            if len(alts) > 2:
                z = np.polyfit(alts, y_obs, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(alts.min(), alts.max(), 50)
                ax.plot(x_fit, p(x_fit), "--", color="#2980b9", alpha=0.6,
                        label=f"Tendencia obs ({z[0]:.4f}/m)")
            ax.set_xlabel("Altitud (m)"); ax.set_ylabel(f"{lbl} ({units})")
            ax.legend(fontsize=8)

        for sid in sv.index:
            ax.annotate(stations[sid]["name"][:8],
                        (sv.loc[sid, "alt_m"],
                         sv.loc[sid, "pbias_pct"] if "PBIAS" in lbl
                         else sv.loc[sid, "obs_mean"]),
                        fontsize=6, ha="left", va="bottom")
        ax.set_title(f"({chr(97 + list(axes.flat).index(ax))}) {lbl}", fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "V10_gradiente_altitudinal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura guardada: {out.name}")


def plot_V11_qm_precipitacion(pisco_basin: pd.Series,
                               obs_ref: pd.Series,
                               pr_corr: pd.Series,
                               cal_start: str, cal_end: str,
                               fig_dir: Path) -> None:
    """V11: QM precipitación — CDF, climatología, Q-Q, serie mensual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal_mask = (pisco_basin.index >= cal_start) & (pisco_basin.index <= cal_end)
    p_cal    = pisco_basin[cal_mask].dropna()
    o_cal    = obs_ref[cal_mask].dropna()
    c_cal    = pr_corr[cal_mask].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("V11 — Corrección Quantile Mapping precipitación\n"
                 f"Cuenca Chancay-Huaral | Calibración {cal_start[:4]}-{cal_end[:4]}",
                 fontsize=11, fontweight="bold")

    # (a) CDF comparativa
    ax = axes[0, 0]
    for series, label, color, lw in [
        (p_cal[p_cal >= PRECIP_THRESH], "PISCO original", "#e74c3c", 1.5),
        (c_cal[c_cal >= PRECIP_THRESH], "PISCO corregido (QM)", "#27ae60", 2.0),
        (o_cal[o_cal >= PRECIP_THRESH], "Observado red estaciones", "#2980b9", 1.5),
    ]:
        s_sorted = np.sort(series.dropna().values)
        cdf = np.linspace(0, 1, len(s_sorted))
        ax.plot(s_sorted, cdf, color=color, lw=lw, label=label)
    ax.set_xlabel("Precipitación (mm/día)"); ax.set_ylabel("Probabilidad acumulada")
    ax.set_title("(a) CDF — días lluviosos (≥0.1 mm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Climatología mensual
    ax = axes[0, 1]
    months = range(1, 13)
    labels_m = ["E","F","M","A","M","J","J","A","S","O","N","D"]
    p_mon = pisco_basin.groupby(pisco_basin.index.month).mean()
    c_mon = pr_corr.groupby(pr_corr.index.month).mean()
    o_mon = obs_ref.groupby(obs_ref.index.month).mean()
    ax.bar(months, [o_mon.get(m, np.nan) for m in months],
           color="#2980b9", alpha=0.6, label="Observado red")
    ax.plot(months, [p_mon.get(m, np.nan) for m in months],
            "-o", color="#e74c3c", ms=6, lw=1.5, label="PISCO original")
    ax.plot(months, [c_mon.get(m, np.nan) for m in months],
            "-s", color="#27ae60", ms=6, lw=1.5, label="PISCO corregido")
    ax.set_xticks(months); ax.set_xticklabels(labels_m)
    ax.set_ylabel("Pr media (mm/día)")
    ax.set_title("(b) Climatología mensual"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Q-Q plot antes y después de corrección
    ax = axes[1, 0]
    q_vals = np.linspace(0.01, 0.99, 99)
    q_obs  = np.quantile(o_cal.dropna(), q_vals)
    q_pis  = np.quantile(p_cal.dropna(), q_vals)
    q_cor  = np.quantile(c_cal.dropna(), q_vals)
    ax.plot(q_obs, q_pis, "o", ms=4, color="#e74c3c", alpha=0.7, label="PISCO original")
    ax.plot(q_obs, q_cor, "s", ms=4, color="#27ae60", alpha=0.7, label="PISCO corregido")
    lim = max(q_obs.max(), q_pis.max(), q_cor.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")
    ax.set_xlabel("Cuantiles observados (mm/día)")
    ax.set_ylabel("Cuantiles PISCO (mm/día)")
    ax.set_title("(c) Q-Q plot"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Serie mensual 1981-2020
    ax = axes[1, 1]
    p_annual = pisco_basin.resample("ME").mean()
    c_annual = pr_corr.resample("ME").mean()
    o_annual = obs_ref.resample("ME").mean()
    ax.fill_between(o_annual.index, o_annual.values, alpha=0.3,
                    color="#2980b9", label="Observado red")
    ax.plot(p_annual.index, p_annual.values, color="#e74c3c",
            lw=0.8, alpha=0.8, label="PISCO original")
    ax.plot(c_annual.index, c_annual.values, color="#27ae60",
            lw=1.2, alpha=0.9, label="PISCO corregido")
    ax.axvline(pd.Timestamp(cal_end), color="gray", ls=":", lw=1,
               label="Fin calibración")
    ax.set_ylabel("Pr media mensual (mm/día)")
    ax.set_title("(d) Serie mensual 1981-2020"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "V11_qm_precipitacion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura guardada: {out.name}")


def plot_V12_correccion_temperatura(pisco_tmax: pd.Series,
                                    pisco_tmin: pd.Series,
                                    tmax_corr: pd.Series,
                                    tmin_corr: pd.Series,
                                    obs_tmax: pd.Series,
                                    obs_tmin: pd.Series,
                                    fig_dir: Path) -> None:
    """V12: Corrección temperatura — bias mensual, climatología, scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("V12 — Corrección delta temperatura Tmax/Tmin\n"
                 "Cuenca Chancay-Huaral", fontsize=11, fontweight="bold")
    months = range(1, 13)
    labels_m = ["E","F","M","A","M","J","J","A","S","O","N","D"]

    for row_idx, (var_pisco, var_corr, var_obs, label) in enumerate([
        (pisco_tmax, tmax_corr, obs_tmax, "Tmax"),
        (pisco_tmin, tmin_corr, obs_tmin, "Tmin"),
    ]):
        # Climatología mensual
        ax = axes[row_idx, 0]
        p_m = var_pisco.groupby(var_pisco.index.month).mean()
        c_m = var_corr.groupby(var_corr.index.month).mean()
        o_m = var_obs.groupby(var_obs.index.month).mean()
        ax.plot(months, [o_m.get(m, np.nan) for m in months],
                "-o", color="#2980b9", ms=5, lw=1.5, label="Observado red")
        ax.plot(months, [p_m.get(m, np.nan) for m in months],
                "-s", color="#e74c3c", ms=5, lw=1.2, label="PISCO original")
        ax.plot(months, [c_m.get(m, np.nan) for m in months],
                "-^", color="#27ae60", ms=5, lw=1.5, label="PISCO corregido")
        ax.set_xticks(months); ax.set_xticklabels(labels_m)
        ax.set_ylabel(f"{label} (°C)")
        ax.set_title(f"({'ab'[row_idx]}) Climatología mensual {label}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Delta mensual
        ax = axes[row_idx, 1]
        delta_m = {m: float(o_m.get(m, np.nan)) - float(p_m.get(m, np.nan))
                   for m in months}
        colors_d = ["#e74c3c" if v > 0 else "#2980b9"
                    for v in delta_m.values()]
        bars = ax.bar(months, list(delta_m.values()), color=colors_d, alpha=0.8)
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(months); ax.set_xticklabels(labels_m)
        ax.set_ylabel("Delta (°C)  [obs - PISCO]")
        ax.set_title(f"({'cd'[row_idx]}) Delta mensual {label} (positivo = PISCO frío)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "V12_correccion_temperatura.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura guardada: {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    silver_dir  = PROJECT_ROOT / "data" / "silver" / "senamhi"
    bronze_dir  = PROJECT_ROOT / "data" / "bronze"
    fig_dir     = PROJECT_ROOT / "outputs" / "figures" / "basin"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("SCRIPT 19: Calibración PISCO vs SENAMHI — Chancay-Huaral")
    log.info("=" * 70)

    # ── 1. Cargar estaciones ──────────────────────────────────────────────────
    log.info("\n[1] Cargando estaciones SENAMHI ...")
    catalog  = silver_dir / "S2_senamhi_catalog.csv"
    stations = load_senamhi(silver_dir, catalog)
    if not stations:
        log.error("No se encontraron estaciones SENAMHI. Verificar data/silver/senamhi/")
        sys.exit(1)
    log.info(f"  Total estaciones cargadas: {len(stations)}")

    # ── 2. Extraer píxeles PISCO en ubicación de cada estación ───────────────
    log.info("\n[2] Extrayendo píxeles PISCO en ubicación de estaciones ...")

    pr_nc   = bronze_dir / "B2_pr_basin_grid_v21.nc"
    tmax_nc = bronze_dir / "B2_tmax_basin_grid_v12.nc"
    tmin_nc = bronze_dir / "B2_tmin_basin_grid_v12.nc"

    pisco_pr_at_stn   = None
    pisco_tmax_at_stn = None
    pisco_tmin_at_stn = None

    if pr_nc.exists():
        log.info("  Extrayendo pr ...")
        pisco_pr_at_stn = extract_pisco_at_stations(pr_nc, "pr", stations)
    else:
        log.warning(f"  {pr_nc.name} no encontrado")

    if tmax_nc.exists():
        log.info("  Extrayendo tmax ...")
        pisco_tmax_at_stn = extract_pisco_at_stations(tmax_nc, "tmax", stations)
    else:
        log.warning(f"  {tmax_nc.name} no encontrado")

    if tmin_nc.exists():
        log.info("  Extrayendo tmin ...")
        pisco_tmin_at_stn = extract_pisco_at_stations(tmin_nc, "tmin", stations)
    else:
        log.warning(f"  {tmin_nc.name} no encontrado")

    # ── 3. Métricas validación punto-a-píxel ─────────────────────────────────
    log.info("\n[3] Calculando métricas de validación ...")
    stats = compute_validation_metrics(
        stations, pisco_pr_at_stn, pisco_tmax_at_stn, pisco_tmin_at_stn
    )
    out_stats = bronze_dir / "B6_calibration_stats.csv"
    stats.to_csv(out_stats, index=False)
    log.info(f"  Métricas guardadas: {out_stats.name}")

    log.info("\n  Resumen KGE por variable:")
    for var in ["pr", "tmax", "tmin"]:
        sv = stats[stats["variable"] == var]
        if sv.empty:
            continue
        log.info(f"  {var}: KGE={sv['kge'].mean():.2f}±{sv['kge'].std():.2f}  "
                 f"PBIAS={sv['pbias_pct'].mean():+.1f}%  "
                 f"r={sv['pearson_r'].mean():.2f}")
    if "wet_freq_obs" in stats.columns:
        sv_pr = stats[stats["variable"] == "pr"]
        log.info(f"  Drizzle — freq obs: {sv_pr['wet_freq_obs'].mean():.0%}  "
                 f"PISCO: {sv_pr['wet_freq_pisco'].mean():.0%}")

    # ── 4. Serie PISCO basin mean ─────────────────────────────────────────────
    log.info("\n[4] Cargando PISCO basin mean ...")
    pisco_bm = pd.read_csv(
        bronze_dir / "B2_pisco_basin_mean_all.csv",
        index_col=0, parse_dates=True,
    )
    full_idx = pisco_bm.index
    pisco_pr_basin   = pisco_bm["pr"].copy()
    pisco_tmax_basin = pisco_bm["tmax"].copy()
    pisco_tmin_basin = pisco_bm["tmin"].copy()

    # ── 5. Referencia observada de cuenca (IDW) ───────────────────────────────
    log.info("\n[5] Construyendo referencia observada de cuenca (IDW) ...")
    log.info(f"  Estaciones incluidas: {STATIONS_BASIN_REF}")

    obs_pr_ref   = build_basin_obs_reference(
        stations, STATIONS_BASIN_REF, "pr",   full_idx)
    obs_tmax_ref = build_basin_obs_reference(
        stations, STATIONS_BASIN_REF, "tmax", full_idx)
    obs_tmin_ref = build_basin_obs_reference(
        stations, STATIONS_BASIN_REF, "tmin", full_idx)

    # Overlap común para calibración
    cal_mask = (full_idx >= CAL_START) & (full_idx <= CAL_END)
    log.info(f"  Período calibración: {CAL_START} a {CAL_END}  "
             f"({cal_mask.sum()} días)")
    log.info(f"  Obs pr ref — media cal: {obs_pr_ref[cal_mask].mean():.3f} mm/día  "
             f"vs PISCO: {pisco_pr_basin[cal_mask].mean():.3f} mm/día")
    log.info(f"  Obs tmax ref — media: {obs_tmax_ref[cal_mask].mean():.1f}°C  "
             f"vs PISCO: {pisco_tmax_basin[cal_mask].mean():.1f}°C")

    # ── 6. Quantile Mapping precipitación ────────────────────────────────────
    log.info("\n[6] Ajustando Quantile Mapping mensual para precipitación ...")
    qm_params = fit_qm_monthly(
        pisco_pr_basin[cal_mask], obs_pr_ref[cal_mask], N_QUANTILES
    )
    save_qm_params(qm_params, bronze_dir / "B6_qm_params.csv")

    log.info("  Aplicando QM a serie completa 1981-2020 ...")
    pr_corrected = apply_qm(pisco_pr_basin, qm_params)

    log.info(f"  Antes QM — mean={pisco_pr_basin.mean():.3f}  "
             f"wet%={wet_day_freq(pisco_pr_basin.values):.0%}  "
             f"p99={np.nanquantile(pisco_pr_basin,0.99):.1f}")
    log.info(f"  Después QM — mean={pr_corrected.mean():.3f}  "
             f"wet%={wet_day_freq(pr_corrected.values):.0%}  "
             f"p99={np.nanquantile(pr_corrected,0.99):.1f}")
    log.info(f"  Referencia obs — mean={obs_pr_ref[cal_mask].mean():.3f}  "
             f"wet%={wet_day_freq(obs_pr_ref[cal_mask].values):.0%}")

    # ── 7. Corrección delta temperatura ──────────────────────────────────────
    log.info("\n[7] Ajustando corrección delta mensual temperatura ...")
    delta_tmax = fit_delta_T_monthly(
        pisco_tmax_basin[cal_mask], obs_tmax_ref[cal_mask])
    delta_tmin = fit_delta_T_monthly(
        pisco_tmin_basin[cal_mask], obs_tmin_ref[cal_mask])

    tmax_corrected = apply_delta_T(pisco_tmax_basin, delta_tmax)
    tmin_corrected = apply_delta_T(pisco_tmin_basin, delta_tmin)

    T_corr_df = pd.DataFrame({
        "delta_tmax_c": delta_tmax,
        "delta_tmin_c": delta_tmin,
    })
    T_corr_df.to_csv(bronze_dir / "B6_T_correction.csv")
    log.info(f"  Delta Tmax mensual: {delta_tmax.values}")
    log.info(f"  Delta Tmin mensual: {delta_tmin.values}")

    # ── 8. Guardar serie corregida ────────────────────────────────────────────
    log.info("\n[8] Guardando serie corregida ...")

    corrected_df = pd.DataFrame({
        "pr":    pr_corrected,
        "pr_orig": pisco_pr_basin,
        "tmax":  tmax_corrected,
        "tmax_orig": pisco_tmax_basin,
        "tmin":  tmin_corrected,
        "tmin_orig": pisco_tmin_basin,
        "pet":   pisco_bm.get("pet", np.nan),  # PET sin cambio (se recalcula en Script 07)
    }, index=full_idx)
    corrected_df.index.name = "date"

    out_corr = bronze_dir / "B2_pisco_basin_mean_corrected.csv"
    corrected_df.to_csv(out_corr)
    log.info(f"  Serie corregida: {out_corr.name}  ({len(corrected_df)} filas)")

    # ── Trazabilidad: sidecar JSON con lineage completo ───────────────────────
    lineage = {
        "file": out_corr.name,
        "generated_by": "scripts/19_calibracion_senamhi.py",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_data": {
            "precipitation": (
                "data/bronze/B2_pisco_basin_mean_all.csv  col='pr'  "
                "PISCOp v2.1 update (1981-2019, QM input)"
            ),
            "temperature": (
                "data/bronze/B2_pisco_basin_mean_all.csv  cols='tmax','tmin'  "
                "PISCOt v1.2 (1981-2020, delta-T input)"
            ),
            "observations": "data/raw/senamhi/*.txt  (DONOSO, HUAYAN, PACCHO, PACHAMACHAY, PALLAC, PARQUIN, PICOY, PIRCA, SANTA_CRUZ)",
        },
        "methods": {
            "precipitation_bias_correction": {
                "method": "Quantile Mapping (QM) mensual empirico",
                "formula": "pr_corr = F_obs_m^{-1}( F_pisco_m(pr_raw) )",
                "description": (
                    "Para cada mes m: se estiman CDFs empiricas con N_QUANTILES={} cuantiles "
                    "para PISCO (F_pisco_m) y observaciones IDW (F_obs_m). "
                    "Extrapolacion lineal mas alla del rango de calibracion."
                ).format(N_QUANTILES),
                "calibration_period": f"{CAL_START} a {CAL_END}",
                "n_quantiles": N_QUANTILES,
                "application": "serie completa 1981-2019 (2020 = NaN en pr por limite de PISCOp)",
                "params_file": "data/bronze/B6_qm_params.csv  (cuantiles por mes)",
                "reference_obs": "IDW (1/dist^2) de estaciones representativas",
            },
            "temperature_bias_correction": {
                "method": "Correccion delta mensual",
                "formula": "T_corr[t] = T_pisco[t] + delta_m   donde   delta_m = mean(T_obs_m) - mean(T_pisco_m)",
                "description": "Additive bias correction mensual entre PISCO y media altitudinal de estaciones.",
                "params_file": "data/bronze/B6_T_correction.csv  (delta_tmax, delta_tmin por mes)",
            },
        },
        "columns": {
            "pr":        "precipitacion QM-corregida (mm/dia)  — fuente: PISCOp v2.1 update",
            "pr_orig":   "precipitacion PISCO original sin correccion (mm/dia)",
            "tmax":      "temperatura maxima delta-corregida (degC)  — fuente: PISCOt v1.2",
            "tmax_orig": "temperatura maxima PISCO original (degC)",
            "tmin":      "temperatura minima delta-corregida (degC)  — fuente: PISCOt v1.2",
            "tmin_orig": "temperatura minima PISCO original (degC)",
            "pet":       "evapotranspiracion potencial PISCO original (mm/dia)  — extension en Script 07",
        },
        "period_expected": "1981-01-01 a 2020-12-31",
        "known_issues": {
            "pr_2020": "NaN — PISCOp v2.1 update solo llega a 2019-12-31",
            "tmax_tmin_valid": "1981-01-01 a 2020-12-31 desde PISCOt v1.2",
        },
    }
    lineage_path = bronze_dir / "B2_pisco_basin_mean_corrected_lineage.json"
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, ensure_ascii=False)
    log.info(f"  Trazabilidad: {lineage_path.name}")

    # ── 9. Figuras ────────────────────────────────────────────────────────────
    log.info("\n[9] Generando figuras V09-V12 ...")
    try:
        if pisco_pr_at_stn is not None:
            plot_V09_validacion_pr(stats, stations, pisco_pr_at_stn, fig_dir)
        plot_V10_gradiente_altitudinal(stats, stations, fig_dir)
        plot_V11_qm_precipitacion(
            pisco_pr_basin, obs_pr_ref, pr_corrected,
            CAL_START, CAL_END, fig_dir,
        )
        if not obs_tmax_ref.isna().all():
            plot_V12_correccion_temperatura(
                pisco_tmax_basin, pisco_tmin_basin,
                tmax_corrected, tmin_corrected,
                obs_tmax_ref, obs_tmin_ref,
                fig_dir,
            )
    except Exception as exc:
        log.error(f"Error en figuras: {exc}", exc_info=True)

    # ── Resumen final ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("RESUMEN CALIBRACIÓN")
    log.info(f"  Estaciones validadas: {len(stations)}")
    log.info(f"  Período calibración:  {CAL_START} a {CAL_END}")
    log.info(f"  Pr PISCO original: mean={pisco_pr_basin.mean():.3f}  "
             f"p99={np.nanquantile(pisco_pr_basin,0.99):.1f} mm/d")
    log.info(f"  Pr CORREGIDA (QM): mean={pr_corrected.mean():.3f}  "
             f"p99={np.nanquantile(pr_corrected,0.99):.1f} mm/d")
    log.info(f"  Pr observada ref:  mean={obs_pr_ref[cal_mask].mean():.3f} mm/d")
    log.info(f"  Salidas: {out_corr.name}, B6_calibration_stats.csv, "
             f"B6_qm_params.csv, B6_T_correction.csv")
    log.info(f"  Figuras: V09-V12 en outputs/figures/basin/")
    log.info("=" * 70)
    log.info("\nSIGUIENTE PASO: Regenerar D5_tft_ready.csv usando "
             "B2_pisco_basin_mean_corrected.csv (modificar Script 16).")


if __name__ == "__main__":
    main()
