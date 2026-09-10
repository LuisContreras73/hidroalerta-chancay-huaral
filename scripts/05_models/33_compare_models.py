#!/usr/bin/env python3
"""
Script 33: Comparacion unificada de todos los modelos — TFT vs LSTM vs ML.

Objetivo
--------
Tabla y figura de comparacion final para el informe del concurso ANA.
Integra resultados de:
  Script 22 — TFT multi-entidad (main model)
  Script 31 — ML baselines D6 (RF, XGB, LGB)
  Script 32 — LSTM baseline D6

Contexto cientifico
-------------------
La jerarquia esperada de rendimiento (mejor a peor):
  TFT > LSTM > LGB >= XGB >= RF > Climatologia > Persistencia

Si TFT supera a LSTM: confirma que VSN + multi-head attention contribuyen.
Si LSTM supera a ML:  confirma que las dependencias temporales largas (90d)
                      no son capturadas por lags estaticos de los arboles.

La brecha TFT-ML en CSI para umbrales altos (10/20mm) es el resultado
mas importante para justificar el uso de QuantileLoss y atencion temporal.

Skill Score
-----------
  SS_clim = (metric_model - metric_clim) / (metric_perfect - metric_clim)
  Para NSE: metric_perfect = 1.0, metric_clim = NSE(Climatologia)
  SS > 0: supera climatologia. SS = 1: perfecto.

Salidas
-------
  outputs/comparison/C01_leaderboard.csv          — tabla completa de metricas
  outputs/comparison/C01_skill_scores.csv         — skill scores relativos
  outputs/figures/comparison/VC01_leaderboard.png — figura principal del informe
  outputs/figures/comparison/VC02_quantile_coverage.png
  outputs/figures/comparison/VC03_csi_by_threshold.png

Uso
---
  python scripts/33_compare_models.py
  python scripts/33_compare_models.py --target pr_sum_next_7d
  python scripts/33_compare_models.py --list-runs    # muestra runs disponibles
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT             = Path(__file__).parent.parent
TFT_DIR          = ROOT / "models/tft/runs"
LSTM_DIR         = ROOT / "models/lstm/runs"
TRANSFORMER_DIR  = ROOT / "models/transformer/runs"
ML_DIR           = ROOT / "models/ml/runs/D6"
D6_FILE          = ROOT / "data/model_ready/D6_multientity.csv"

sys.path.insert(0, str(ROOT / "data/metadata"))
from entity_labels import ENTITY_ELEVATIONS, ENTITIES_ASC, entity_label

# Cada ejecucion de Script 33 genera su propia carpeta con fecha → no sobreescribe
_today   = datetime.now().strftime("%Y-%m-%d")
OUT_DIR  = ROOT / "outputs/comparisons" / _today
FIG_DIR  = OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("s33")

TARGET_HORIZON = {"pr_next_1d": 1, "pr_sum_next_3d": 3, "pr_sum_next_7d": 7}
THRESHOLDS     = [1, 5, 10, 20]
QUANTILES_COLS = ["q10", "q25", "q50", "q75", "q90"]

MODEL_META = {
    "Persistencia":  {"color": "#95a5a6", "marker": "o",  "ls": "--", "type": "baseline"},
    "Climatologia":  {"color": "#7f8c8d", "marker": "s",  "ls": "--", "type": "baseline"},
    "RF_pooled":     {"color": "#3498db", "marker": "o",  "ls": "-",  "type": "ml"},
    "XGB_pooled":    {"color": "#e74c3c", "marker": "s",  "ls": "-",  "type": "ml"},
    "LGB_pooled":    {"color": "#27ae60", "marker": "^",  "ls": "-",  "type": "ml"},
    "LSTM":        {"color": "#f39c12", "marker": "D",  "ls": "-",  "type": "dl"},
    "Transformer": {"color": "#e67e22", "marker": "P",  "ls": "-",  "type": "dl"},
    "TFT":         {"color": "#8e44ad", "marker": "*",  "ls": "-",  "type": "dl"},
}


# ══════════════════════════════════════════════════════════════════════════════
# METRICAS
# ══════════════════════════════════════════════════════════════════════════════

def _nse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 10:
        return np.nan
    return float(1.0 - np.sum((o-s)**2) / (np.sum((o - np.mean(o))**2) + 1e-12))


def _kge(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 10:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    a = s.std() / (o.std() + 1e-12)
    b = s.mean() / (o.mean() + 1e-12)
    return float(1.0 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2))


def _threshold_csi(obs, sim, thr):
    m = np.isfinite(obs) & np.isfinite(sim)
    ob, sb = obs[m] >= thr, sim[m] >= thr
    h = np.sum(ob & sb)
    return float(h / (np.sum(ob | sb) + 1e-9))


def _quantile_coverage(obs, q_lo, q_hi):
    """Fraccion de observaciones dentro de [q_lo, q_hi]."""
    m = np.isfinite(obs) & np.isfinite(q_lo) & np.isfinite(q_hi)
    return float(np.mean((obs[m] >= q_lo[m]) & (obs[m] <= q_hi[m])))


def _pinball(obs, pred, q):
    """Pinball loss para cuantil q."""
    m = np.isfinite(obs) & np.isfinite(pred)
    e = obs[m] - pred[m]
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def load_ml_metrics(target: str) -> pd.DataFrame | None:
    """Carga metricas de Script 31."""
    path = ML_DIR / f"M01_{target}_metrics.csv"
    if not path.exists():
        log.warning(f"ML metrics no encontradas: {path.name}. Correr Script 31 primero.")
        return None
    return pd.read_csv(path)


def load_ml_predictions(target: str) -> pd.DataFrame | None:
    path = ML_DIR / f"M01_{target}_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def find_best_tft_run(target: str) -> Path | None:
    """Encuentra el run TFT con mejor val_loss para el target dado."""
    candidates = sorted(TFT_DIR.glob(f"*_{target}"))
    if not candidates:
        return None

    best_dir, best_loss = None, float("inf")
    for run_dir in candidates:
        metrics_csv = run_dir / "metrics_per_epoch.csv"
        if not metrics_csv.exists():
            continue
        df_m = pd.read_csv(metrics_csv).dropna(subset=["val_loss"])
        if df_m.empty:
            continue
        min_loss = df_m["val_loss"].min()
        model_type = json.loads((run_dir / "config.json").read_text()).get("model", "TFT")
        if "LSTM" not in model_type and min_loss < best_loss:
            best_loss = min_loss
            best_dir  = run_dir
    return best_dir


def find_best_lstm_run(target: str) -> Path | None:
    candidates = sorted(LSTM_DIR.glob(f"*{target}*"))
    if not candidates:
        return None
    # Prefer runs that already have predictions (e.g. inference-only runs), latest first
    pred_runs = [r for r in reversed(candidates)
                 if (r / "predictions" / f"test_predictions_{target}.csv").exists()]
    if pred_runs:
        return pred_runs[0]
    # Fallback: pick by lowest val_loss among trained runs
    best_dir, best_loss = None, float("inf")
    for run_dir in candidates:
        metrics_csv = run_dir / "metrics_per_epoch.csv"
        if not metrics_csv.exists():
            continue
        df_m = pd.read_csv(metrics_csv).dropna(subset=["val_loss"])
        if df_m.empty:
            continue
        min_loss = df_m["val_loss"].min()
        if min_loss < best_loss:
            best_loss = min_loss
            best_dir  = run_dir
    return best_dir


def find_best_transformer_run(target: str) -> Path | None:
    """Encuentra el run VanillaTransformer con predicciones para el target dado."""
    candidates = sorted(TRANSFORMER_DIR.glob(f"*{target}*")) if TRANSFORMER_DIR.exists() else []
    if not candidates:
        return None
    pred_runs = [r for r in reversed(candidates)
                 if (r / "predictions" / f"test_predictions_{target}.csv").exists()]
    if pred_runs:
        return pred_runs[0]
    best_dir, best_loss = None, float("inf")
    for run_dir in candidates:
        metrics_csv = run_dir / "metrics_per_epoch.csv"
        if not metrics_csv.exists():
            continue
        df_m = pd.read_csv(metrics_csv).dropna(subset=["val_loss"])
        if df_m.empty:
            continue
        min_loss = df_m["val_loss"].min()
        if min_loss < best_loss:
            best_loss = min_loss
            best_dir  = run_dir
    return best_dir


def load_dl_predictions(run_dir: Path, target: str, model_label: str) -> pd.DataFrame | None:
    """Carga predicciones de un run TFT o LSTM."""
    pred_file = run_dir / "predictions" / f"test_predictions_{target}.csv"
    if not pred_file.exists():
        log.warning(f"  {model_label}: predicciones no encontradas en {run_dir.name}")
        return None
    df = pd.read_csv(pred_file, parse_dates=["date"])
    log.info(f"  {model_label}: cargadas {len(df)} predicciones de {run_dir.name}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CALCULO METRICAS UNIFICADO
# ══════════════════════════════════════════════════════════════════════════════

def compute_dl_metrics(name: str, pred_df: pd.DataFrame, target: str,
                       obs_df: pd.DataFrame) -> list[dict]:
    """
    Calcula metricas para un modelo DL dado sus predicciones.
    obs_df: DataFrame con columnas [date, entity_id, target] del test set.
    """
    if pred_df is None:
        return []

    # Unir con observaciones reales
    merged = pred_df.merge(
        obs_df[["date", "entity_id", target]].rename(columns={target: "obs_true"}),
        on=["date", "entity_id"], how="inner",
    )

    # Usar q50 como prediccion puntual (columna puede llamarse q50 o q05 segun version)
    q50_col = "q50" if "q50" in merged.columns else None
    if q50_col is None:
        # Buscar columna con cuantil 50
        for c in merged.columns:
            if "50" in c:
                q50_col = c
                break
    if q50_col is None:
        log.warning(f"  {name}: no se encontro columna q50 en predicciones")
        return []

    rows = []
    for scope in ["GLOBAL"] + ENTITIES_ASC:
        if scope == "GLOBAL":
            sub = merged
        else:
            sub = merged[merged["entity_id"] == scope]
        if sub.empty:
            continue

        obs_a  = sub["obs_true"].values
        q50_a  = sub[q50_col].values
        row = {
            "model":     name,
            "entity_id": scope,
            "NSE":  _nse(obs_a, q50_a),
            "KGE":  _kge(obs_a, q50_a),
            "MAE":  float(np.nanmean(np.abs(obs_a - q50_a))),
            "RMSE": float(np.sqrt(np.nanmean((obs_a - q50_a)**2))),
        }

        # CSI por umbral
        for thr in THRESHOLDS:
            row[f"CSI_{thr}mm"] = _threshold_csi(obs_a, q50_a, thr)

        # Cobertura de cuantiles (si hay q10 y q90)
        q10_col = "q10" if "q10" in merged.columns else None
        q90_col = "q90" if "q90" in merged.columns else None
        if q10_col and q90_col:
            row["coverage_80"] = _quantile_coverage(
                obs_a, sub[q10_col].values, sub[q90_col].values)
        else:
            row["coverage_80"] = np.nan

        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SKILL SCORES
# ══════════════════════════════════════════════════════════════════════════════

def compute_skill_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Skill Score relativo a Climatologia y a Persistencia.
    SS_clim(NSE) = (NSE_model - NSE_clim) / (1 - NSE_clim)
    Valores: SS=0 igual que climatologia, SS=1 perfecto, SS<0 peor que climatologia.
    """
    global_df  = metrics_df[metrics_df["entity_id"] == "GLOBAL"].set_index("model")
    clim_nse   = global_df.loc["Climatologia", "NSE"]   if "Climatologia" in global_df.index else np.nan
    clim_kge   = global_df.loc["Climatologia", "KGE"]   if "Climatologia" in global_df.index else np.nan
    pers_nse   = global_df.loc["Persistencia", "NSE"]   if "Persistencia" in global_df.index else np.nan

    rows = []
    for model in global_df.index:
        r = global_df.loc[model]
        nse = r.get("NSE", np.nan)
        kge = r.get("KGE", np.nan)
        rows.append({
            "model": model,
            "NSE":   nse,
            "KGE":   kge,
            "MAE":   r.get("MAE", np.nan),
            "SS_clim_NSE": (nse - clim_nse) / (1.0 - clim_nse + 1e-9) if np.isfinite(clim_nse) and np.isfinite(nse) else np.nan,
            "SS_pers_NSE": (nse - pers_nse) / (1.0 - pers_nse + 1e-9) if np.isfinite(pers_nse) and np.isfinite(nse) else np.nan,
            **{f"CSI_{t}mm": r.get(f"CSI_{t}mm", np.nan) for t in THRESHOLDS},
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def plot_leaderboard(metrics_df: pd.DataFrame, ss_df: pd.DataFrame, target: str):
    """
    VC01: Figura principal para el informe.
    6 paneles: NSE | KGE | MAE | CSI por umbral | SS_clim | cobertura cuantiles
    """
    global_df = metrics_df[metrics_df["entity_id"] == "GLOBAL"].set_index("model")
    models    = list(global_df.index)

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.35)

    def _color(m):
        return MODEL_META.get(m, {}).get("color", "#aaa")

    def _bar(ax, col, title, ylabel="", higher_is_better=True, ref_line=None):
        vals   = [float(global_df.loc[m, col]) if m in global_df.index else np.nan for m in models]
        colors = [_color(m) for m in models]
        bars   = ax.bar(range(len(models)), vals, color=colors, alpha=0.85, width=0.65,
                        edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel or col, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        if ref_line is not None:
            ax.axhline(ref_line, color="black", lw=1, ls="--", alpha=0.5)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ypos = max(float(val), 0) + ax.get_ylim()[1] * 0.01
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
        best_idx = int(np.nanargmax(vals) if higher_is_better else np.nanargmin(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

    # (a) NSE
    ax_nse = fig.add_subplot(gs[0, 0])
    _bar(ax_nse, "NSE", "(a) NSE — Nash-Sutcliffe\nmas alto = mejor", ref_line=0)

    # (b) KGE
    ax_kge = fig.add_subplot(gs[0, 1])
    _bar(ax_kge, "KGE", "(b) KGE — Kling-Gupta\nmas alto = mejor", ref_line=0)

    # (c) MAE
    ax_mae = fig.add_subplot(gs[0, 2])
    _bar(ax_mae, "MAE", "(c) MAE (mm)\nmas bajo = mejor", higher_is_better=False)

    # (d) CSI por umbral
    ax_csi = fig.add_subplot(gs[1, 0])
    for m in models:
        if m not in global_df.index:
            continue
        csi_vals = [float(global_df.loc[m, f"CSI_{t}mm"])
                    if f"CSI_{t}mm" in global_df.columns else np.nan
                    for t in THRESHOLDS]
        meta = MODEL_META.get(m, {})
        ax_csi.plot(THRESHOLDS, csi_vals, ls=meta.get("ls", "-"),
                    marker=meta.get("marker", "o"), ms=6, lw=1.8,
                    color=_color(m), label=m, alpha=0.9)
    ax_csi.set_xscale("log")
    ax_csi.set_xticks(THRESHOLDS)
    ax_csi.set_xticklabels([f"{t}mm" for t in THRESHOLDS])
    ax_csi.set_xlabel("Umbral (mm)"); ax_csi.set_ylabel("CSI")
    ax_csi.set_title("(d) CSI por umbral\nCSI=0 en >=20mm = falla en extremos",
                     fontsize=9, fontweight="bold")
    ax_csi.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax_csi.legend(fontsize=7); ax_csi.grid(alpha=0.3)

    # (e) Skill Score relativo a Climatologia
    ax_ss = fig.add_subplot(gs[1, 1])
    ss_global = ss_df.set_index("model")
    if "SS_clim_NSE" in ss_global.columns:
        vals = [float(ss_global.loc[m, "SS_clim_NSE"])
                if m in ss_global.index else np.nan for m in models]
        cols = [_color(m) for m in models]
        bars = ax_ss.bar(range(len(models)), vals, color=cols, alpha=0.85, width=0.65)
        ax_ss.axhline(0, color="black", lw=1.2, ls="--")
        ax_ss.axhline(1, color="#27ae60", lw=0.8, ls=":", alpha=0.6, label="perfecto")
        ax_ss.set_xticks(range(len(models)))
        ax_ss.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
        ax_ss.set_title("(e) Skill Score vs Climatologia\nSS>0 supera climatologia",
                        fontsize=9, fontweight="bold")
        ax_ss.set_ylabel("SS (NSE)")
        ax_ss.legend(fontsize=8); ax_ss.grid(alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ypos = max(float(val), 0) + 0.01
                ax_ss.text(bar.get_x() + bar.get_width()/2, ypos,
                           f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    # (f) Cobertura de cuantiles (IC 80% = [q10, q90])
    ax_cov = fig.add_subplot(gs[1, 2])
    cov_col = "coverage_80"
    if cov_col in global_df.columns:
        vals = [float(global_df.loc[m, cov_col]) if m in global_df.index else np.nan
                for m in models]
        ax_cov.bar(range(len(models)), vals,
                   color=[_color(m) for m in models], alpha=0.85, width=0.65)
        ax_cov.axhline(0.80, color="gold", lw=1.5, ls="--",
                       label="ideal 80% coverage")
        ax_cov.set_xticks(range(len(models)))
        ax_cov.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
        ax_cov.set_title("(f) Cobertura IC 80% [q10-q90]\nIdeal ~80% (calibracion perfecta)",
                         fontsize=9, fontweight="bold")
        ax_cov.set_ylabel("Fraccion de obs cubiertas")
        ax_cov.set_ylim(0, 1); ax_cov.legend(fontsize=8); ax_cov.grid(alpha=0.3, axis="y")
    else:
        ax_cov.text(0.5, 0.5, "Cobertura no disponible\n(requiere predicciones cuantil\nde TFT y LSTM)",
                    ha="center", va="center", transform=ax_cov.transAxes, fontsize=9, color="gray")
        ax_cov.set_title("(f) Cobertura IC 80% [q10-q90]", fontsize=9, fontweight="bold")

    horizon = TARGET_HORIZON.get(target, "?")
    fig.suptitle(f"Comparacion de modelos — {target}  (H={horizon}d)\n"
                 f"Dataset D6: 9 sub-cuencas Chancay-Huaral | Test 2015-2020\n"
                 f"Borde dorado = mejor modelo en cada metrica",
                 fontsize=11, fontweight="bold", y=1.01)

    out = FIG_DIR / f"VC01_{target}_leaderboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VC01: {out.name}")


def plot_entity_nse(metrics_df: pd.DataFrame, target: str):
    """VC02: NSE por sub-cuenca para cada modelo."""
    ent_df = metrics_df[metrics_df["entity_id"] != "GLOBAL"]
    models = [m for m in metrics_df["model"].unique()
              if m not in ("Persistencia",)]
    n = len(models)
    if n == 0:
        return

    x = np.arange(len(ENTITIES_ASC))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, m in enumerate(models):
        df_m = ent_df[ent_df["model"] == m].set_index("entity_id")
        vals = [df_m.loc[e, "NSE"] if e in df_m.index else np.nan
                for e in ENTITIES_ASC]
        ax.bar(x + i*w - (n-1)*w/2, vals, w,
               label=m, color=MODEL_META.get(m, {}).get("color", "#aaa"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [entity_label(e, "tick") for e in ENTITIES_ASC],
        fontsize=9)
    ax.set_ylabel("NSE")
    ax.set_title(f"NSE por sub-cuenca — {target}\n"
                 f"Izq=costera (521m) → Der=cabecera (4507m)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=min(n, 4))
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    out = FIG_DIR / f"VC02_{target}_nse_by_entity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VC02: {out.name}")


def plot_csi_detail(metrics_df: pd.DataFrame, target: str):
    """VC03: CSI por umbral en detalle con barras por modelo."""
    global_df = metrics_df[metrics_df["entity_id"] == "GLOBAL"].set_index("model")
    models = list(global_df.index)

    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(16, 5), sharey=False)
    for ax, thr in zip(axes, THRESHOLDS):
        col   = f"CSI_{thr}mm"
        if col not in global_df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        vals  = [float(global_df.loc[m, col]) if m in global_df.index else np.nan
                 for m in models]
        cols  = [MODEL_META.get(m, {}).get("color", "#aaa") for m in models]
        bars  = ax.bar(range(len(models)), vals, color=cols, alpha=0.85, width=0.65)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(f"Umbral >={thr} mm", fontsize=9, fontweight="bold")
        ax.set_ylabel("CSI")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
        ax.axhline(0, color="black", lw=0.8)
        for bar, val in zip(bars, vals):
            if np.isfinite(val) and val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, float(val) + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle(f"CSI por umbral — {target}\nFalla total (CSI=0) en extremos es la "
                 "motivacion principal del TFT con QuantileLoss",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / f"VC03_{target}_csi_detail.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  VC03: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def list_runs():
    for label, runs_dir in [("TFT", TFT_DIR), ("LSTM", LSTM_DIR),
                             ("Transformer", TRANSFORMER_DIR)]:
        runs = sorted(runs_dir.glob("*"))
        log.info(f"Runs disponibles en {runs_dir.relative_to(ROOT)}:")
        for r in runs:
            metrics = r / "metrics_per_epoch.csv"
            if metrics.exists():
                df_m = pd.read_csv(metrics).dropna(subset=["val_loss"])
                best = df_m["val_loss"].min() if not df_m.empty else float("nan")
                n_ep = len(df_m)
                has_pred = (r / "predictions").exists()
                log.info(f"  [{label}] {r.name}  best_val={best:.3f}  epocas={n_ep}  preds={'si' if has_pred else 'no'}")
            else:
                has_pred = (r / "predictions").exists()
                log.info(f"  [{label}] {r.name}  (inference-only)  preds={'si' if has_pred else 'no'}")


def main():
    parser = argparse.ArgumentParser(description="Script 33: Comparacion de modelos")
    parser.add_argument("--target", default="pr_sum_next_7d",
                        choices=["pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d"])
    parser.add_argument("--list-runs", action="store_true")
    args = parser.parse_args()

    if args.list_runs:
        list_runs()
        return

    log.info("=" * 70)
    log.info(f"SCRIPT 33: Comparacion de modelos — {args.target}")
    log.info("=" * 70)

    # ── 1. Cargar metricas ML (Script 31) ────────────────────────────────────
    ml_metrics = load_ml_metrics(args.target)
    if ml_metrics is not None:
        log.info(f"ML baselines: {ml_metrics['model'].nunique()} modelos cargados")
        all_metrics = ml_metrics.copy()
    else:
        all_metrics = pd.DataFrame()

    # ── 2. Cargar y calcular metricas TFT ────────────────────────────────────
    obs_df = pd.read_csv(D6_FILE, usecols=["date", "entity_id", "split", args.target],
                         parse_dates=["date"])
    obs_test = obs_df[obs_df["split"] == "test"]

    tft_run = find_best_tft_run(args.target)
    if tft_run:
        log.info(f"\nTFT run encontrado: {tft_run.name}")
        tft_preds = load_dl_predictions(tft_run, args.target, "TFT")
        if tft_preds is not None:
            tft_metrics = compute_dl_metrics("TFT", tft_preds, args.target, obs_test)
            if tft_metrics:
                all_metrics = pd.concat([all_metrics,
                                         pd.DataFrame(tft_metrics)], ignore_index=True)
    else:
        log.warning("TFT: no se encontraron predicciones. Entrenamiento aun en progreso.")

    # ── 3. Cargar y calcular metricas LSTM ───────────────────────────────────
    lstm_run = find_best_lstm_run(args.target)
    if lstm_run:
        log.info(f"LSTM run encontrado: {lstm_run.name}")
        lstm_preds = load_dl_predictions(lstm_run, args.target, "LSTM")
        if lstm_preds is not None:
            lstm_metrics = compute_dl_metrics("LSTM", lstm_preds, args.target, obs_test)
            if lstm_metrics:
                all_metrics = pd.concat([all_metrics,
                                         pd.DataFrame(lstm_metrics)], ignore_index=True)
    else:
        log.info("LSTM: no se encontraron predicciones (Script 32 pendiente).")

    # ── 3b. Cargar y calcular métricas VanillaTransformer ────────────────────
    transformer_run = find_best_transformer_run(args.target)
    if transformer_run:
        log.info(f"Transformer run encontrado: {transformer_run.name}")
        tr_preds = load_dl_predictions(transformer_run, args.target, "Transformer")
        if tr_preds is not None:
            tr_metrics = compute_dl_metrics("Transformer", tr_preds, args.target, obs_test)
            if tr_metrics:
                all_metrics = pd.concat([all_metrics,
                                         pd.DataFrame(tr_metrics)], ignore_index=True)
    else:
        log.info("Transformer: no se encontraron predicciones (Script 34 pendiente).")

    if all_metrics.empty:
        log.error("Sin metricas disponibles. Correr Scripts 31 y/o 22 primero.")
        sys.exit(1)

    # ── 4. Skill Scores ───────────────────────────────────────────────────────
    ss_df = compute_skill_scores(all_metrics)

    # ── 5. Guardar tablas ────────────────────────────────────────────────────
    global_metrics = all_metrics[all_metrics["entity_id"] == "GLOBAL"]
    out_lb = OUT_DIR / f"C01_{args.target}_leaderboard.csv"
    global_metrics.to_csv(out_lb, index=False)
    log.info(f"\nLeaderboard: {out_lb.name}")

    out_ss = OUT_DIR / f"C01_{args.target}_skill_scores.csv"
    ss_df.to_csv(out_ss, index=False)

    # ── 6. Imprimir tabla en consola ──────────────────────────────────────────
    log.info("\nLEADERBOARD GLOBAL (test 2015-2020):")
    log.info(f"{'Modelo':<16} {'NSE':>7} {'KGE':>7} {'MAE':>7}  "
             f"{'CSI_1':>6} {'CSI_5':>6} {'CSI_10':>7} {'CSI_20':>7}  "
             f"{'SS_clim':>8}")
    log.info("-" * 88)
    ss_idx = ss_df.set_index("model")
    for m in global_metrics["model"].unique():
        r = global_metrics[global_metrics["model"] == m].iloc[0]
        ss = ss_idx.loc[m, "SS_clim_NSE"] if m in ss_idx.index else float("nan")
        log.info(
            f"{m:<16} {r.get('NSE', float('nan')):>7.3f} {r.get('KGE', float('nan')):>7.3f} "
            f"{r.get('MAE', float('nan')):>7.3f}  "
            f"{r.get('CSI_1mm', float('nan')):>6.3f} {r.get('CSI_5mm', float('nan')):>6.3f} "
            f"{r.get('CSI_10mm', float('nan')):>7.3f} {r.get('CSI_20mm', float('nan')):>7.3f}  "
            f"{ss:>8.3f}"
        )

    # ── 7. Figuras ────────────────────────────────────────────────────────────
    log.info("\nGenerando figuras ...")
    plot_leaderboard(all_metrics, ss_df, args.target)
    plot_entity_nse(all_metrics, args.target)
    plot_csi_detail(all_metrics, args.target)

    log.info("\n" + "=" * 70)
    log.info("Script 33 completado.")
    n_models = global_metrics["model"].nunique()
    log.info(f"  {n_models} modelos comparados para {args.target}")
    if tft_run is None:
        log.info("  NOTA: Correr de nuevo cuando TFT (Script 22) termine de entrenar")
    if lstm_run is None:
        log.info("  NOTA: Correr Script 32 para agregar LSTM al comparativo")
    if transformer_run is None:
        log.info("  NOTA: Correr Script 34 para agregar VanillaTransformer al comparativo")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
