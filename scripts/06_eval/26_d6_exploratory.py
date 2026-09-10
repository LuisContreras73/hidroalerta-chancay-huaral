#!/usr/bin/env python3
"""
Script 26: Análisis exploratorio D6 multi-entidad.

Genera 5 figuras (VD601–VD605) que caracterizan el dataset D6 desde la
perspectiva del gradiente altitudinal y las relaciones espacio-temporales
entre las 9 sub-cuencas. No requiere el modelo TFT — solo D6_multientity.csv.

Figuras
-------
VD601 — Gradiente altitudinal: climatología de pr_mm, wet_pct, std/mean (CV)
VD602 — Cross-correlación lag: pr_mm entre sub_649 (cabecera) → sub_634 (costa)
VD603 — Estacionalidad mensual por entidad: climatología pr_mm heatmap
VD604 — Correlación ONI × pr_mm por altitud y mes
VD605 — Correlación Pearson entre variables de D6 (heatmap)

Salidas
-------
outputs/figures/d6/VD601_altitudinal_gradient.png
outputs/figures/d6/VD602_lag_crosscorr.png
outputs/figures/d6/VD603_seasonal_heatmap.png
outputs/figures/d6/VD604_oni_correlation.png
outputs/figures/d6/VD605_feature_correlation.png
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ROOT   = Path(__file__).parent.parent
D6_CSV = ROOT / "data/model_ready/D6_multientity.csv"
OUT    = ROOT / "outputs/figures/d6"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "data/metadata"))
from entity_labels import ENTITY_META, ENTITY_ELEVATIONS, ENTITIES_ASC, entity_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("s26")

ALTITUDES = [ENTITY_ELEVATIONS[e] for e in ENTITIES_ASC]

MONTH_NAMES  = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]


# ── Carga ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    log.info(f"Cargando {D6_CSV.name} ...")
    df = pd.read_csv(D6_CSV, parse_dates=["date"])
    df = df.sort_values(["entity_id", "date"]).reset_index(drop=True)
    df["elevation_m"] = df["entity_id"].map(ENTITY_ELEVATIONS)
    log.info(f"  {len(df)} filas, {df['entity_id'].nunique()} entidades, "
             f"{df['date'].min().year}-{df['date'].max().year}")
    return df


# ── VD601: Gradiente altitudinal ───────────────────────────────────────────────

def plot_altitudinal_gradient(df: pd.DataFrame):
    tr = df[df["split"] == "train"].copy()

    stats_ent = tr.groupby("entity_id")["pr_mm"].agg(
        mean="mean", std="std",
        p90=lambda x: x.quantile(0.90),
        p99=lambda x: x.quantile(0.99),
        wet_pct=lambda x: (x > 1.0).mean(),
    ).copy()
    stats_ent["cv"] = stats_ent["std"] / (stats_ent["mean"] + 1e-9)
    stats_ent["elevation_m"] = stats_ent.index.map(ENTITY_ELEVATIONS)
    stats_ent = stats_ent.sort_values("elevation_m")
    elev = stats_ent["elevation_m"].values

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def _scatter_trend(ax, y, label, color, ylabel):
        ax.scatter(elev, y, color=color, s=80, zorder=5)
        for ent, yi, ei in zip(stats_ent.index, y, elev):
            ax.annotate(entity_label(ent, "annot"), (ei, yi),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
        slope, intercept, r, p, _ = stats.linregress(elev, y)
        xfit = np.linspace(elev.min(), elev.max(), 100)
        ax.plot(xfit, slope*xfit + intercept, "--", color=color, lw=1.2, alpha=0.7)
        ax.set_xlabel("Elevación (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(0.05, 0.92, f"r={r:.2f}  p={'<0.001' if p<0.001 else f'{p:.3f}'}",
                transform=ax.transAxes, fontsize=8, color=color)
        ax.grid(alpha=0.3)

    _scatter_trend(fig.add_subplot(gs[0,0]), stats_ent["mean"].values,
                   "(a) Precipitación media (mm/día)", "#2980b9", "pr_mm media")
    _scatter_trend(fig.add_subplot(gs[0,1]), stats_ent["wet_pct"].values * 100,
                   "(b) Fracción días húmedos (pr>1mm, %)", "#27ae60", "Wet days (%)")
    _scatter_trend(fig.add_subplot(gs[0,2]), stats_ent["p99"].values,
                   "(c) Percentil 99 precipitación (mm/día)", "#e74c3c", "pr_mm p99")
    _scatter_trend(fig.add_subplot(gs[1,0]), stats_ent["std"].values,
                   "(d) Desviación estándar pr_mm", "#8e44ad", "pr_mm std")
    _scatter_trend(fig.add_subplot(gs[1,1]), stats_ent["cv"].values,
                   "(e) Coeficiente de variación (std/mean)", "#e67e22", "CV")
    _scatter_trend(fig.add_subplot(gs[1,2]), stats_ent["p90"].values,
                   "(f) Percentil 90 precipitación (mm/día)", "#c0392b", "pr_mm p90")

    fig.suptitle("Gradiente altitudinal — Cuenca Chancay-Huaral (9 sub-cuencas)\n"
                 "Entrenamiento 1981-2008, D6_multientity", fontsize=12, y=0.98)
    fname = OUT / "VD601_altitudinal_gradient.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VD601 guardada: {fname.name}")


# ── VD602: Cross-correlación lag ───────────────────────────────────────────────

def plot_lag_crosscorr(df: pd.DataFrame):
    tr = df[df["split"] == "train"].copy()
    # Pivot: filas=fecha, columnas=entity, valores=pr_mm
    piv = tr.pivot_table(index="date", columns="entity_id", values="pr_mm")
    piv = piv[ENTITIES_ASC]  # orden altitudinal

    # Para cada par (entidad alta, entidad baja): ccf a lag 0-15 días
    pairs = [
        ("sub_649", "sub_640", "4507m → 1480m"),
        ("sub_649", "sub_634", "4507m → 521m"),
        ("sub_646", "sub_634", "3812m → 521m"),
        ("sub_650", "sub_634", "2841m → 521m"),
    ]
    max_lag = 20
    colors  = ["#e74c3c", "#2980b9", "#27ae60", "#e67e22"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for (src, tgt, lbl), col in zip(pairs, colors):
        lags, ccf_vals = [], []
        x = piv[src].dropna()
        y = piv[tgt].dropna()
        common = x.index.intersection(y.index)
        xc = (x[common] - x[common].mean()) / (x[common].std() + 1e-9)
        yc = (y[common] - y[common].mean()) / (y[common].std() + 1e-9)
        n  = len(common)
        for lag in range(0, max_lag + 1):
            if lag == 0:
                r = np.corrcoef(xc.values, yc.values)[0, 1]
            else:
                r = np.corrcoef(xc.values[:-lag], yc.values[lag:])[0, 1]
            lags.append(lag)
            ccf_vals.append(r)
        best_lag = lags[np.argmax(ccf_vals)]
        ax.plot(lags, ccf_vals, "-o", color=col, markersize=4, lw=1.5,
                label=f"{lbl} (lag óptimo={best_lag}d, r={max(ccf_vals):.2f})")

    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(1.96 / np.sqrt(len(common)), color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-1.96 / np.sqrt(len(common)), color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Lag (días)")
    ax.set_ylabel("Correlación cruzada")
    ax.set_title("(a) CCF: sub-cuencas alta → baja (cabecera anticipa costa)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, max_lag + 0.5)

    # Mapa de correlación entre todos los pares a lag=0 y lag=3
    ax2 = axes[1]
    n_ent = len(ENTITIES_ASC)
    corr_mat = np.zeros((n_ent, n_ent))
    for i, e1 in enumerate(ENTITIES_ASC):
        for j, e2 in enumerate(ENTITIES_ASC):
            if i == j:
                corr_mat[i, j] = 1.0
                continue
            x1 = piv[e1].dropna(); x2 = piv[e2].dropna()
            comm = x1.index.intersection(x2.index)
            if len(comm) > 30:
                corr_mat[i, j] = np.corrcoef(x1[comm].values, x2[comm].values)[0, 1]

    labels_short = [entity_label(e, "tick") for e in ENTITIES_ASC]
    im = ax2.imshow(corr_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(n_ent)); ax2.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(n_ent)); ax2.set_yticklabels(labels_short, fontsize=8)
    for i in range(n_ent):
        for j in range(n_ent):
            ax2.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if corr_mat[i,j] < 0.8 else "white")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="r (Pearson, lag=0)")
    ax2.set_title("(b) Matriz de correlación pr_mm entre entidades (lag=0)",
                  fontsize=10, fontweight="bold")

    fig.suptitle("Cross-correlación espacial pr_mm — D6 multi-entidad (train 1981-2008)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fname = OUT / "VD602_lag_crosscorr.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VD602 guardada: {fname.name}")


# ── VD603: Estacionalidad mensual por entidad ──────────────────────────────────

def plot_seasonal_heatmap(df: pd.DataFrame):
    tr = df[df["split"] == "train"].copy()
    tr["month"] = tr["date"].dt.month

    # pr_mm climatología mensual por entidad
    clim = tr.groupby(["entity_id", "month"])["pr_mm"].mean().unstack(level=1)
    clim = clim.loc[ENTITIES_ASC]  # orden altitudinal
    clim.columns = [MONTH_NAMES[m-1] for m in clim.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap normalizado por fila (contraste intra-entidad)
    ax = axes[0]
    norm_clim = clim.div(clim.max(axis=1), axis=0)
    ylabels   = [entity_label(e, "tick") for e in ENTITIES_ASC]
    im = ax.imshow(norm_clim.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_yticks(range(len(ENTITIES_ASC))); ax.set_yticklabels(ylabels, fontsize=9)
    plt.colorbar(im, ax=ax, label="pr_mm / max_mensual (normalizado por fila)")
    for i in range(len(ENTITIES_ASC)):
        for j in range(12):
            ax.text(j, i, f"{clim.values[i,j]:.1f}", ha="center", va="center",
                    fontsize=6.5, color="black" if norm_clim.values[i,j] < 0.7 else "white")
    ax.set_title("(a) Climatología pr_mm (mm/día) — normalizada por fila\n"
                 "Evidencia: régimen húmedo DJF más claro en cabecera",
                 fontsize=10, fontweight="bold")

    # Líneas de climatología mensual por entidad
    ax2 = axes[1]
    cmap_e = plt.cm.plasma_r(np.linspace(0.1, 0.9, len(ENTITIES_ASC)))
    for i, (ent, col) in enumerate(zip(ENTITIES_ASC, cmap_e)):
        vals = clim.loc[ent].values
        elev = ENTITY_ELEVATIONS[ent]
        ax2.plot(range(12), vals, "-o", color=col, markersize=4, lw=1.5,
                 label=entity_label(ent, "label"))
    ax2.set_xticks(range(12)); ax2.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax2.set_ylabel("pr_mm media (mm/día)")
    ax2.set_title("(b) Climatología mensual por sub-cuenca\n"
                  "Gradiente: cabecera wet season DJF/MAM pronunciada", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2, loc="upper right")
    ax2.grid(alpha=0.3)
    # Banda wet season
    ax2.axvspan(11, 12, alpha=0.08, color="blue", label="DJF")
    ax2.axvspan(0, 3, alpha=0.08, color="blue")
    ax2.axvspan(2, 5, alpha=0.06, color="green")

    fig.suptitle("Estacionalidad pr_mm — D6 multi-entidad (train 1981-2008)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fname = OUT / "VD603_seasonal_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VD603 guardada: {fname.name}")


# ── VD604: Correlación ONI × pr_mm ────────────────────────────────────────────

def plot_oni_correlation(df: pd.DataFrame):
    tr = df[df["split"] == "train"].copy()
    tr["month"] = tr["date"].dt.month

    # Correlación ONI vs pr_mm por entidad (todo el período)
    oni_pr_corr = tr.groupby("entity_id").apply(
        lambda g: np.corrcoef(g["oni_index"].fillna(0), g["pr_mm"])[0, 1]
    ).rename("r_oni_pr")

    # Correlación por mes × entidad
    r_monthly = {}
    for ent in ENTITIES_ASC:
        sub = tr[tr["entity_id"] == ent]
        r_monthly[ent] = [
            np.corrcoef(sub[sub["month"] == m]["oni_index"].fillna(0),
                        sub[sub["month"] == m]["pr_mm"])[0, 1]
            for m in range(1, 13)
        ]
    r_df = pd.DataFrame(r_monthly, index=MONTH_NAMES).T
    r_df = r_df.loc[ENTITIES_ASC]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter: elevación vs correlación global ONI-pr
    ax = axes[0]
    elev_vals = [ENTITY_ELEVATIONS[e] for e in ENTITIES_ASC]
    r_vals    = [float(oni_pr_corr.get(e, np.nan)) for e in ENTITIES_ASC]
    colors    = ["#e74c3c" if r < 0 else "#2980b9" for r in r_vals]
    ax.bar(range(len(ENTITIES_ASC)), r_vals, color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(ENTITIES_ASC)))
    ax.set_xticklabels([entity_label(e, "tick") for e in ENTITIES_ASC], fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.2,  color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-0.2, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel("r (ONI, pr_mm) — período train 1981-2008")
    ax.set_title("(a) Correlación ONI × pr_mm por sub-cuenca\n"
                 "Rojo=negativo (La Niña más lluvia), Azul=positivo", fontsize=10, fontweight="bold")
    for i, (r, e) in enumerate(zip(r_vals, ENTITIES_ASC)):
        ax.text(i, r + (0.005 if r >= 0 else -0.005), f"{r:.2f}",
                ha="center", va="bottom" if r >= 0 else "top", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Heatmap: correlación ONI-pr por mes × entidad
    ax2 = axes[1]
    ylabels = [entity_label(e, "tick") for e in ENTITIES_ASC]
    im = ax2.imshow(r_df.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax2.set_xticks(range(12)); ax2.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax2.set_yticks(range(len(ENTITIES_ASC))); ax2.set_yticklabels(ylabels, fontsize=9)
    plt.colorbar(im, ax=ax2, label="r (ONI, pr_mm)", fraction=0.046, pad=0.04)
    for i in range(len(ENTITIES_ASC)):
        for j in range(12):
            v = r_df.values[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if abs(v) > 0.3 else "black")
    ax2.set_title("(b) Correlación ONI × pr_mm por mes y sub-cuenca\n"
                  "Azul=El Niño aporta lluvia, Rojo=El Niño suprime lluvia",
                  fontsize=10, fontweight="bold")

    fig.suptitle("Influencia ENSO/ONI en precipitación — D6 multi-entidad (train 1981-2008)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fname = OUT / "VD604_oni_correlation.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VD604 guardada: {fname.name}")


# ── VD605: Heatmap correlación entre features ──────────────────────────────────

def plot_feature_correlation(df: pd.DataFrame):
    tr = df[(df["split"] == "train") & (df["entity_id"] == "sub_649")].copy()

    feat_cols = [
        "pr_mm", "tmax_c", "tmin_c", "pet_mm", "oni_index",
        "api", "spi_30d", "spi_90d", "water_deficit_30d",
        "clim_pr_p50", "clim_pr_p90",
        "pr_next_1d", "pr_sum_next_7d",
    ]
    feat_cols = [c for c in feat_cols if c in tr.columns]
    corr = tr[feat_cols].dropna(how="all").corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    n = len(feat_cols)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(feat_cols, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.7 else "black")
    plt.colorbar(im, ax=ax, label="r (Pearson)", fraction=0.03, pad=0.02)
    ax.set_title(f"Correlación entre features D6 — {entity_label('sub_649')} | train 1981-2008\n"
                 "Correlaciones claves para el TFT y diseño de features adicionales",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fname = OUT / "VD605_feature_correlation.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VD605 guardada: {fname.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("SCRIPT 26: Análisis exploratorio D6 — HidroAlerta Chancay-Huaral")
    log.info("=" * 70)

    df = load_data()

    log.info("\n[1/5] Gradiente altitudinal ...")
    plot_altitudinal_gradient(df)

    log.info("[2/5] Cross-correlación lag entre sub-cuencas ...")
    plot_lag_crosscorr(df)

    log.info("[3/5] Estacionalidad mensual por entidad ...")
    plot_seasonal_heatmap(df)

    log.info("[4/5] Correlación ONI × pr_mm ...")
    plot_oni_correlation(df)

    log.info("[5/5] Correlación entre features ...")
    plot_feature_correlation(df)

    log.info(f"\nScript 26 completado. Figuras en: {OUT}")
    log.info("  VD601 — gradiente altitudinal")
    log.info("  VD602 — cross-correlación lag")
    log.info("  VD603 — estacionalidad mensual")
    log.info("  VD604 — correlación ONI")
    log.info("  VD605 — correlación features")


if __name__ == "__main__":
    main()
