"""Sistema de leaderboard: rankings, comparación de modelos y figuras."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from hidroalerta.utils.io import ensure_dir, get_project_root

logger = logging.getLogger(__name__)

# Métrica principal por defecto (menor es mejor para MAE)
_DEFAULT_METRIC = "mae"
_HIGHER_IS_BETTER = {"nse", "kge", "r2", "pearson", "spearman", "coverage_p10_p90"}


def _is_higher_better(metric: str) -> bool:
    return metric.lower() in _HIGHER_IS_BETTER


# ---------------------------------------------------------------------------
# Carga del leaderboard
# ---------------------------------------------------------------------------

def load_all_runs(outputs_dir: str | Path | None = None) -> pd.DataFrame:
    """Carga outputs/leaderboards/all_runs.csv y devuelve un DataFrame.

    Devuelve DataFrame vacío si el archivo no existe.
    """
    if outputs_dir is None:
        outputs_dir = get_project_root() / "outputs"
    lb_path = Path(outputs_dir) / "leaderboards" / "all_runs.csv"
    if not lb_path.exists():
        logger.warning("Leaderboard no encontrado en %s", lb_path)
        return pd.DataFrame()
    df = pd.read_csv(lb_path, low_memory=False)
    logger.info("Leaderboard cargado: %d corridas desde %s", len(df), lb_path)
    return df


# ---------------------------------------------------------------------------
# Generación de sub-leaderboards
# ---------------------------------------------------------------------------

def generate_best_by_model(
    df: pd.DataFrame,
    metric: str = _DEFAULT_METRIC,
    outputs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Genera best_by_model.csv: mejor corrida por model_name según *metric*."""
    if df.empty:
        logger.warning("DataFrame vacío; no se generará best_by_model.csv")
        return df

    df_success = df[df["status"] == "success"].copy()
    if df_success.empty:
        logger.warning("No hay corridas exitosas en el leaderboard.")
        return df_success

    df_success[metric] = pd.to_numeric(df_success[metric], errors="coerce")

    if _is_higher_better(metric):
        idx = df_success.groupby("model_name")[metric].idxmax()
    else:
        idx = df_success.groupby("model_name")[metric].idxmin()

    best = df_success.loc[idx.dropna()].reset_index(drop=True)

    if outputs_dir is not None:
        out_path = Path(outputs_dir) / "leaderboards" / "best_by_model.csv"
        ensure_dir(out_path.parent)
        best.to_csv(out_path, index=False)
        logger.info("best_by_model.csv guardado en %s (%d filas)", out_path, len(best))

    return best


def generate_best_by_target_horizon(
    df: pd.DataFrame,
    metric: str = _DEFAULT_METRIC,
    outputs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Genera best_by_target_horizon.csv: mejor corrida por (target, horizon)."""
    if df.empty:
        return df

    df_success = df[df["status"] == "success"].copy()
    df_success[metric] = pd.to_numeric(df_success[metric], errors="coerce")

    if _is_higher_better(metric):
        idx = df_success.groupby(["target", "horizon"])[metric].idxmax()
    else:
        idx = df_success.groupby(["target", "horizon"])[metric].idxmin()

    best = df_success.loc[idx.dropna()].reset_index(drop=True)

    if outputs_dir is not None:
        out_path = Path(outputs_dir) / "leaderboards" / "best_by_target_horizon.csv"
        ensure_dir(out_path.parent)
        best.to_csv(out_path, index=False)
        logger.info("best_by_target_horizon.csv guardado en %s", out_path)

    return best


def generate_failed_runs(
    df: pd.DataFrame,
    outputs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Genera failed_runs.csv con todas las corridas fallidas."""
    failed = df[df["status"] == "failed"].copy() if not df.empty else df

    if outputs_dir is not None:
        out_path = Path(outputs_dir) / "leaderboards" / "failed_runs.csv"
        ensure_dir(out_path.parent)
        failed.to_csv(out_path, index=False)
        logger.info("failed_runs.csv guardado en %s (%d filas)", out_path, len(failed))

    return failed


def generate_model_comparison(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    outputs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Genera model_comparison.csv en outputs/metrics/ con estadísticas por modelo."""
    if df.empty:
        return df

    if metrics is None:
        metrics = ["mae", "rmse", "nse", "kge", "r2", "pbias", "pearson", "spearman"]

    df_success = df[df["status"] == "success"].copy()
    for m in metrics:
        if m in df_success.columns:
            df_success[m] = pd.to_numeric(df_success[m], errors="coerce")

    existing_metrics = [m for m in metrics if m in df_success.columns]
    agg_fns: dict[str, Any] = {}
    for m in existing_metrics:
        agg_fns[f"{m}_mean"] = (m, "mean")
        agg_fns[f"{m}_std"] = (m, "std")
        agg_fns[f"{m}_min"] = (m, "min")
        agg_fns[f"{m}_max"] = (m, "max")
    agg_fns["n_runs"] = ("run_id", "count")

    comparison = df_success.groupby("model_name").agg(**agg_fns).reset_index()

    if outputs_dir is not None:
        out_path = Path(outputs_dir) / "metrics" / "model_comparison.csv"
        ensure_dir(out_path.parent)
        comparison.to_csv(out_path, index=False)
        logger.info("model_comparison.csv guardado en %s", out_path)

    return comparison


# ---------------------------------------------------------------------------
# Figuras del leaderboard
# ---------------------------------------------------------------------------

def generate_leaderboard_figures(
    df: pd.DataFrame | None = None,
    outputs_dir: str | Path | None = None,
    metric: str = "nse",
) -> None:
    """Genera todas las figuras del leaderboard en outputs/figures/model_comparison/.

    Figuras:
    - leaderboard_barplot.png
    - rmse_by_model.png
    - nse_by_model.png
    - metric_vs_trial.png
    - top_10_runs.png
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib/seaborn no instalados; no se generarán figuras.")
        return

    if outputs_dir is None:
        outputs_dir = get_project_root() / "outputs"
    outputs_dir = Path(outputs_dir)

    if df is None:
        df = load_all_runs(outputs_dir)

    if df.empty:
        logger.warning("Sin datos para generar figuras del leaderboard.")
        return

    fig_dir = outputs_dir / "figures" / "model_comparison"
    ensure_dir(fig_dir)

    df_success = df[df["status"] == "success"].copy()
    for col in ["mae", "rmse", "nse", "kge", "r2", "pbias"]:
        if col in df_success.columns:
            df_success[col] = pd.to_numeric(df_success[col], errors="coerce")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. leaderboard_barplot.png (métrica principal por modelo)
    _barplot_by_model(
        df_success, metric=metric, fig_dir=fig_dir,
        fname="leaderboard_barplot.png",
        title=f"{metric.upper()} por modelo (corridas exitosas)",
    )

    # 2. rmse_by_model.png
    _barplot_by_model(
        df_success, metric="rmse", fig_dir=fig_dir,
        fname="rmse_by_model.png",
        title="RMSE por modelo",
    )

    # 3. nse_by_model.png
    _barplot_by_model(
        df_success, metric="nse", fig_dir=fig_dir,
        fname="nse_by_model.png",
        title="NSE por modelo (Nash-Sutcliffe Efficiency)",
    )

    # 4. metric_vs_trial.png
    _metric_vs_trial(df_success, metric=metric, fig_dir=fig_dir)

    # 5. top_10_runs.png
    _top_10_runs(df_success, metric=metric, fig_dir=fig_dir)

    logger.info("Figuras del leaderboard guardadas en %s", fig_dir)


# ---------------------------------------------------------------------------
# Helpers de figuras
# ---------------------------------------------------------------------------

def _barplot_by_model(
    df: pd.DataFrame,
    metric: str,
    fig_dir: Path,
    fname: str,
    title: str,
) -> None:
    """Barplot de *metric* agrupado por model_name."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    if metric not in df.columns or df[metric].isna().all():
        logger.warning("Métrica '%s' no disponible para barplot.", metric)
        return

    agg = (
        df.groupby("model_name")[metric]
        .mean()
        .dropna()
        .sort_values(ascending=not _is_higher_better(metric))
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    agg.plot(kind="bar", ax=ax, color=sns.color_palette("Blues_d", len(agg)))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Modelo")
    ax.set_ylabel(metric.upper())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_path = fig_dir / fname
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Figura guardada: %s", save_path)


def _metric_vs_trial(df: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    """Evolución de la métrica con el número de trial."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    if metric not in df.columns or "trial_id" not in df.columns:
        return

    df_plot = df[["model_name", "trial_id", metric]].dropna()
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, grp in df_plot.groupby("model_name"):
        grp_sorted = grp.sort_values("trial_id")
        ax.plot(grp_sorted["trial_id"].astype(str), grp_sorted[metric], marker="o", label=model)
    ax.set_title(f"{metric.upper()} vs Trial por modelo", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel(metric.upper())
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_path = fig_dir / "metric_vs_trial.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Figura guardada: %s", save_path)


def _top_10_runs(df: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    """Barplot horizontal de las 10 mejores corridas."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    if metric not in df.columns:
        return

    ascending = not _is_higher_better(metric)
    top = df.nsmallest(10, metric) if ascending else df.nlargest(10, metric)
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(top))
    ax.barh(top["run_id"].astype(str), top[metric], color=colors)
    ax.set_title(f"Top 10 corridas — {metric.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Run ID")
    ax.invert_yaxis()
    fig.tight_layout()
    save_path = fig_dir / "top_10_runs.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Figura guardada: %s", save_path)


# ---------------------------------------------------------------------------
# Función de conveniencia
# ---------------------------------------------------------------------------

def rebuild_leaderboard(outputs_dir: str | Path | None = None) -> None:
    """Regenera todos los sub-leaderboards y figuras a partir de all_runs.csv."""
    if outputs_dir is None:
        outputs_dir = get_project_root() / "outputs"
    outputs_dir = Path(outputs_dir)

    df = load_all_runs(outputs_dir)
    if df.empty:
        logger.warning("No hay datos en el leaderboard; nada que reconstruir.")
        return

    generate_best_by_model(df, outputs_dir=outputs_dir)
    generate_best_by_target_horizon(df, outputs_dir=outputs_dir)
    generate_failed_runs(df, outputs_dir=outputs_dir)
    generate_model_comparison(df, outputs_dir=outputs_dir)
    generate_leaderboard_figures(df, outputs_dir=outputs_dir)
    logger.info("Leaderboard reconstruido completamente.")
