"""Módulo de visualización: figuras estándar por corrida y análisis exploratorio."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("matplotlib no instalado; visualizaciones deshabilitadas.")

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

STYLE = "seaborn-v0_8-whitegrid"
DPI = 150
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREEN = "#2ca02c"


def _save(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def plot_observed_vs_predicted(dates, obs, pred, title: str, save_path: Path,
                                units: str = "mm"):
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pd.to_datetime(dates), obs, label="Observado", color=BLUE, lw=1.2, alpha=0.9)
    ax.plot(pd.to_datetime(dates), pred, label="Predicho", color=ORANGE, lw=1.2,
            linestyle="--", alpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"Precipitación ({units})")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _save(fig, save_path)


def plot_scatter_obs_pred(obs, pred, title: str, save_path: Path,
                           units: str = "mm"):
    if not _MPL:
        return
    obs, pred = np.array(obs), np.array(pred)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs, pred, alpha=0.4, s=15, color=BLUE)
    lim = max(obs.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"Observado ({units})")
    ax.set_ylabel(f"Predicho ({units})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend()
    _save(fig, save_path)


def plot_residuals_timeseries(dates, residuals, title: str, save_path: Path):
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pd.to_datetime(dates), residuals, color=RED, lw=0.8, alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Residuo (mm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _save(fig, save_path)


def plot_residuals_distribution(residuals, title: str, save_path: Path):
    if not _MPL:
        return
    res = np.array(residuals)
    res = res[~np.isnan(res)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(res, bins=50, color=BLUE, alpha=0.7, edgecolor="white")
    ax.axvline(0, color=RED, lw=1.5, linestyle="--", label="Sesgo=0")
    ax.axvline(np.mean(res), color=ORANGE, lw=1.5, linestyle=":", label=f"Media={np.mean(res):.2f}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Residuo (mm)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    _save(fig, save_path)


def plot_errors_by_month(dates, residuals, title: str, save_path: Path):
    if not _MPL:
        return
    df = pd.DataFrame({"date": pd.to_datetime(dates), "res": residuals}).dropna()
    df["month"] = df["date"].dt.month
    monthly_mae = df.groupby("month")["res"].apply(lambda x: np.mean(np.abs(x)))

    months_labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                     "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly_mae.index, monthly_mae.values, color=BLUE, alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months_labels)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Mes")
    ax.set_ylabel("MAE (mm)")
    _save(fig, save_path)


def plot_feature_importance(feature_names, importances, title: str, save_path: Path,
                             top_n: int = 20):
    if not _MPL:
        return
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(range(len(names)), vals[::-1], color=BLUE, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names)[::-1], fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Importancia")
    _save(fig, save_path)


def plot_prediction_interval(dates, obs, pred_low, pred_mid, pred_high,
                               title: str, save_path: Path, units: str = "mm"):
    if not _MPL:
        return
    dates_dt = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates_dt, pred_low, pred_high, alpha=0.25, color=ORANGE,
                    label="Intervalo P10-P90")
    ax.plot(dates_dt, pred_mid, color=ORANGE, lw=1.2, label="Mediana P50", linestyle="--")
    ax.plot(dates_dt, obs, color=BLUE, lw=1.2, alpha=0.9, label="Observado")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"Precipitación ({units})")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _save(fig, save_path)


def plot_pisco_timeseries(df: pd.DataFrame, save_dir: Path):
    """Serie temporal completa de precipitación PISCO."""
    if not _MPL:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    dates = pd.to_datetime(df["date"])

    axes[0].bar(dates, df["pr_mm"], width=1, color=BLUE, alpha=0.7, label="Precipitación (mm)")
    axes[0].set_ylabel("pr (mm/día)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Serie temporal PISCO - Cuenca Chancay-Huaral", fontsize=13)

    if "tmax_c" in df.columns:
        axes[1].plot(dates, df["tmax_c"], color=RED, lw=0.8, alpha=0.8, label="Tmax (°C)")
        axes[1].plot(dates, df["tmin_c"], color=BLUE, lw=0.8, alpha=0.8, label="Tmin (°C)")
        if "tmean_c" in df.columns:
            axes[1].plot(dates, df["tmean_c"], color=GREEN, lw=1, alpha=0.9, label="Tmean (°C)")
        axes[1].set_ylabel("Temperatura (°C)")
        axes[1].legend(loc="upper right")

    axes[2].plot(dates, df["pr_mm"].rolling(30).sum(), color=BLUE, lw=1.2)
    axes[2].set_ylabel("pr acumulado 30d (mm)")
    axes[2].set_xlabel("Fecha")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    _save(fig, save_dir / "pisco_pr_timeseries.png")


def plot_pisco_monthly_climatology(df: pd.DataFrame, save_dir: Path):
    """Climatología mensual de precipitación PISCO."""
    if not _MPL:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_m = df.copy()
    df_m["month"] = pd.to_datetime(df_m["date"]).dt.month
    monthly = df_m.groupby("month")["pr_mm"].agg(["mean", "median",
                                                    lambda x: x.quantile(0.25),
                                                    lambda x: x.quantile(0.75)])
    monthly.columns = ["mean", "median", "q25", "q75"]
    monthly["pr_sum_month"] = df_m.groupby("month")["pr_mm"].mean() * 30

    months_labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                     "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(monthly.index, monthly["mean"], color=BLUE, alpha=0.8, label="Media diaria")
    ax1.errorbar(monthly.index, monthly["median"],
                 yerr=[monthly["median"] - monthly["q25"], monthly["q75"] - monthly["median"]],
                 fmt="o", color=RED, capsize=4, label="Mediana ± IQR")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(months_labels, rotation=45)
    ax1.set_title("Precipitación media diaria por mes (PISCO)", fontsize=12)
    ax1.set_ylabel("mm/día")
    ax1.legend()

    ax2.bar(monthly.index, monthly["pr_sum_month"], color=GREEN, alpha=0.8)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(months_labels, rotation=45)
    ax2.set_title("Precipitación mensual estimada (PISCO)", fontsize=12)
    ax2.set_ylabel("mm/mes")

    _save(fig, save_dir / "pisco_pr_monthly_climatology.png")
    logger.info("Figura climatología mensual PISCO guardada")


def plot_pisco_distribution(df: pd.DataFrame, save_dir: Path):
    """Distribución de precipitación PISCO."""
    if not _MPL:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pr = df["pr_mm"].dropna()
    pr_wet = pr[pr > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histograma completo
    axes[0].hist(pr, bins=60, color=BLUE, alpha=0.8, edgecolor="white")
    axes[0].set_title("Distribución de precipitación diaria\n(incluyendo días secos)", fontsize=11)
    axes[0].set_xlabel("mm/día")
    axes[0].set_ylabel("Frecuencia")
    p_dry = (pr == 0).mean() * 100
    axes[0].text(0.98, 0.95, f"Días secos: {p_dry:.1f}%",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=10)

    # Solo días lluviosos
    axes[1].hist(pr_wet, bins=50, color=ORANGE, alpha=0.8, edgecolor="white")
    axes[1].set_title("Distribución de precipitación\n(solo días lluviosos > 0 mm)", fontsize=11)
    axes[1].set_xlabel("mm/día")
    axes[1].set_ylabel("Frecuencia")

    # Boxplot por estación hidrológica
    if "hydro_phase" in df.columns:
        df_plot = df[["hydro_phase", "pr_mm"]].dropna()
        phases = ["wet", "transition_onset", "transition_end", "dry"]
        data = [df_plot[df_plot["hydro_phase"] == p]["pr_mm"].values for p in phases]
        axes[2].boxplot(data, labels=["Húmedo", "Inicio\nlluvias", "Fin\nlluvias", "Seco"],
                        patch_artist=True,
                        boxprops=dict(facecolor=BLUE, alpha=0.6))
        axes[2].set_title("Precipitación por fase hidrológica", fontsize=11)
        axes[2].set_ylabel("mm/día")
    else:
        axes[2].text(0.5, 0.5, "hydro_phase\nno disponible", ha="center", va="center",
                     transform=axes[2].transAxes)

    _save(fig, save_dir / "pisco_pr_distribution.png")
    logger.info("Figura distribución PISCO guardada")


def plot_leaderboard_barplot(df_leaderboard: pd.DataFrame, metric: str,
                              save_path: Path, lower_is_better: bool = True):
    """Barplot comparativo de modelos por métrica."""
    if not _MPL:
        return
    df = df_leaderboard.dropna(subset=[metric]).copy()
    if df.empty:
        logger.warning(f"No hay datos para {metric} en el leaderboard")
        return

    df_sorted = df.sort_values(metric, ascending=lower_is_better)
    colors = [GREEN if i == 0 else BLUE for i in range(len(df_sorted))]

    fig, ax = plt.subplots(figsize=(max(8, len(df_sorted) * 1.2), 6))
    bars = ax.bar(range(len(df_sorted)), df_sorted[metric], color=colors, alpha=0.85)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["model_name"].values, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Comparación de modelos - {metric.upper()}\nCuenca Chancay-Huaral (módulo pluviométrico)", fontsize=12)
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    _save(fig, save_path)
    logger.info(f"Leaderboard barplot guardado: {save_path}")


def plot_metric_vs_trial(df_runs: pd.DataFrame, model_name: str,
                          metric: str, save_path: Path):
    """Evolución de métrica a través de trials."""
    if not _MPL:
        return
    df = df_runs[df_runs["model_name"] == model_name].dropna(subset=[metric]).copy()
    if df.empty:
        return
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[metric], "o-", color=BLUE, markersize=6)
    best_idx = df[metric].idxmin()
    ax.plot(best_idx, df.loc[best_idx, metric], "*", color=RED, markersize=14,
            label=f"Mejor: {df.loc[best_idx, metric]:.4f}")
    ax.set_title(f"{model_name} - {metric} por trial", fontsize=12)
    ax.set_xlabel("Trial")
    ax.set_ylabel(metric.upper())
    ax.legend()
    _save(fig, save_path)
