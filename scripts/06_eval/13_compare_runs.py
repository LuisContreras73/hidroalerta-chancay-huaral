#!/usr/bin/env python3
"""Script 13: Comparar todas las corridas y generar reporte de comparación."""
import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("compare_runs")


def main():
    import pandas as pd
    import numpy as np

    leaderboard_path = PROJECT_ROOT / "outputs" / "leaderboards" / "all_runs.csv"
    if not leaderboard_path.exists() or leaderboard_path.stat().st_size < 100:
        logger.error("all_runs.csv vacío o no existe. Correr scripts 09 y 11 primero.")
        sys.exit(1)

    df = pd.read_csv(leaderboard_path)
    df = df[df["status"] == "success"].copy()
    logger.info(f"Corridas exitosas: {len(df)}")

    if df.empty:
        logger.warning("No hay corridas exitosas para comparar.")
        sys.exit(0)

    # Best by model
    best_by_model = df.sort_values("mae").groupby("model_name").first().reset_index()
    best_by_model.to_csv(PROJECT_ROOT / "outputs" / "leaderboards" / "best_by_model.csv", index=False)

    # Best by target/horizon
    best_by_th = df.sort_values("mae").groupby(["target", "horizon"]).first().reset_index()
    best_by_th.to_csv(PROJECT_ROOT / "outputs" / "leaderboards" / "best_by_target_horizon.csv", index=False)

    # Top 10 global
    top10 = df.nsmallest(10, "mae")[["run_id", "model_name", "target", "horizon", "mae", "rmse", "nse", "kge"]]
    logger.info(f"\n{'='*60}\nTOP 10 CORRIDAS (por MAE)\n{'='*60}")
    logger.info(f"\n{top10.to_string(index=False)}")

    # Figuras comparativas
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = PROJECT_ROOT / "outputs" / "figures" / "model_comparison"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Leaderboard barplot por target
        for target in df["target"].unique():
            df_t = df[df["target"] == target].dropna(subset=["mae"])
            if df_t.empty:
                continue
            df_agg = df_t.groupby("model_name")["mae"].min().sort_values()
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(df_agg.index, df_agg.values, color="#1f77b4", alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", fontsize=9)
            ax.set_xticklabels(df_agg.index, rotation=35, ha="right")
            ax.set_title(f"MAE por modelo — Target: {target}", fontsize=12)
            ax.set_ylabel("MAE (mm)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"leaderboard_barplot_{target}.png", dpi=150)
            plt.close(fig)
            logger.info(f"Figura guardada: leaderboard_barplot_{target}.png")

        # Barplot global (mejor corrida por modelo)
        df_best = df.sort_values("mae").groupby("model_name")["mae"].min().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ca02c"] + ["#1f77b4"] * (len(df_best) - 1)
        bars = ax.bar(df_best.index, df_best.values, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set_xticklabels(df_best.index, rotation=35, ha="right")
        ax.set_title("MAE mínimo por modelo — HidroAlerta Chancay-Huaral\n(módulo pluviométrico)", fontsize=12)
        ax.set_ylabel("MAE (mm)")
        fig.tight_layout()
        fig.savefig(fig_dir / "leaderboard_barplot.png", dpi=150)
        plt.close(fig)
        logger.info("Figura global guardada: leaderboard_barplot.png")

    except Exception as e:
        logger.warning(f"No se pudieron generar figuras: {e}")

    # Generar reporte markdown
    report_lines = [
        "# Reporte de Comparación de Modelos",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Top 10 corridas (MAE más bajo)",
        "",
        top10.to_string(index=False),
        "",
        "## Mejor modelo por target/horizon",
        "",
        best_by_th[["target", "horizon", "model_name", "mae", "rmse", "nse"]].to_string(index=False),
        "",
        "## Conclusiones",
        "- Ver figuras en `outputs/figures/model_comparison/`",
        "- Leaderboard completo en `outputs/leaderboards/all_runs.csv`",
    ]
    report_path = PROJECT_ROOT / "outputs" / "reports" / "model_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Reporte guardado: {report_path}")


if __name__ == "__main__":
    main()
