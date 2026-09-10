#!/usr/bin/env python3
"""
Script 18b: Métricas de umbral para evaluación de alertas (POD/FAR/CSI/ETS).

Motivación
----------
Script 18 reportó NSE/MAE/KGE, métricas continuas adecuadas para evaluar el
ajuste de la distribución. Pero para un sistema de alertas hidrológicas, lo que
importa es la DETECCIÓN BINARIA: ¿el modelo identifica correctamente los días
de lluvia intensa (pr > umbral) o caudal alto (q > percentil)?

Un modelo con NSE=0.6 puede tener POD=0.3 (pierde 70% de los eventos críticos),
lo que lo haría inaceptable para operación de alertas. Esta es la crítica del
hidrólogo senior que Script 18 no respondió.

Umbrales evaluados
-------------------
  Precipitación (D5v2 pr_pisco_mm):
    - 1 mm/d   → días húmedos (básico)
    - 5 mm/d   → precipitación moderada
    - 10 mm/d  → evento significativo
    - 20 mm/d  → evento intenso (p85-p90 de días húmedos)
    - p90, p95 → percentiles de distribución total

  Caudal (q_snirh_m3s):
    - p50 → mediana (flujo base normal)
    - p75 → caudal moderado
    - p90 → caudal alto
    - p95 → evento extremo (~2yr retorno)

Modelos evaluados
-----------------
  Los mismos que Script 18: LightGBM, XGBoost, RF, Ridge, climatología diaria.
  Se carga el modelo persistido si existe; si no, se re-entrena brevemente.

Salidas
-------
  outputs/metrics/D7_threshold_metrics.csv  — tabla completa por modelo/umbral
  outputs/figures/V14_pod_far_csi.png       — diagram skill score
  outputs/figures/V15_performance_diagram.png — diagrama performance (BIAS vs CSI)
  outputs/18b_threshold_metrics.log
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "outputs" / "18b_threshold_metrics.log", "w", "utf-8"),
    ],
)
log = logging.getLogger("thresh18b")

# ── Rutas ──────────────────────────────────────────────────────────────────────
D5_CSV   = ROOT / "data/model_ready/D5_tft_ready.csv"
OUT_DIR  = ROOT / "outputs/metrics"
FIG_DIR  = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PR_THRESHOLDS   = [1.0, 5.0, 10.0, 20.0]   # mm/día
Q_PERCENTILES   = [50, 75, 90, 95]


# ── Métricas de contingencia ───────────────────────────────────────────────────
def contingency_metrics(obs, pred, threshold):
    """
    POD  = hits / (hits + misses)   → probabilidad de detección
    FAR  = false_alarms / (hits + false_alarms) → tasa de falsa alarma
    CSI  = hits / (hits + misses + false_alarms) → índice de éxito crítico
    ETS  = (hits - hits_random) / (hits + misses + fa - hits_random) → Equitable Threat Score
    BIAS = (hits + false_alarms) / (hits + misses) → sesgo de frecuencia
    """
    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    obs_pos = o >= threshold
    prd_pos = p >= threshold
    hits    = np.sum(obs_pos & prd_pos)
    misses  = np.sum(obs_pos & ~prd_pos)
    fa      = np.sum(~obs_pos & prd_pos)
    tn      = np.sum(~obs_pos & ~prd_pos)
    total   = hits + misses + fa + tn
    hits_r  = (hits + misses) * (hits + fa) / (total + 1e-9)  # hits aleatorios
    pod     = hits / (hits + misses + 1e-9)
    far     = fa   / (hits + fa + 1e-9)
    csi     = hits / (hits + misses + fa + 1e-9)
    ets     = (hits - hits_r) / (hits + misses + fa - hits_r + 1e-9)
    bias    = (hits + fa) / (hits + misses + 1e-9)
    return {
        "hits": int(hits), "misses": int(misses),
        "false_alarms": int(fa), "true_negatives": int(tn),
        "POD": float(pod), "FAR": float(far),
        "CSI": float(csi), "ETS": float(ets), "BIAS": float(bias),
        "n_events_obs": int(hits + misses),
        "n_obs": int(total),
    }


def load_data():
    """Carga D5v2 y separa splits."""
    log.info(f"   Cargando {D5_CSV.name} ...")
    df = pd.read_csv(D5_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"   Filas: {len(df)}  Split: {df['split'].value_counts().to_dict()}")
    return df


def train_and_predict(df, target_col):
    """Entrena modelos ML básicos y devuelve predicciones test."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    try:
        from lightgbm import LGBMRegressor
        has_lgb = True
    except ImportError:
        has_lgb = False
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    # Features: todas las columnas excepto targets, split, date, entity
    exclude = ["date", "split", "entity_id", "time_idx",
               "pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d",
               "q_next_1d", "q_sum_next_7d"]
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "float32", "int64", "int32"]]

    train_df = df[df["split"] == "train"].dropna(subset=[target_col])
    test_df  = df[df["split"] == "test"].dropna(subset=[target_col])

    if len(train_df) < 100 or len(test_df) < 30:
        log.warning(f"   Datos insuficientes para {target_col}. Saltando.")
        return None, None, None

    X_train = train_df[feat_cols].fillna(0.0).values
    y_train = train_df[target_col].values
    X_test  = test_df[feat_cols].fillna(0.0).values
    y_test  = test_df[target_col].values

    models = {}
    models["Ridge"]    = Ridge(alpha=1.0)
    models["RF"]       = RandomForestRegressor(n_estimators=100, max_depth=8,
                                               n_jobs=-1, random_state=42)
    if has_lgb:
        models["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=6, random_state=42, verbose=-1)
    if has_xgb:
        models["XGBoost"]  = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=6, random_state=42,
                                          eval_metric="rmse", verbosity=0)

    preds = {"y_true": y_test}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds[name] = mdl.predict(X_test if name != "LightGBM"
                                  else test_df[feat_cols].fillna(0.0))
        log.info(f"   {name:<12} {target_col}: entrenado ({len(y_train)} muestras train)")

    # Climatología: media del mismo mes en training set
    clim_map = train_df.groupby(train_df["date"].dt.month if "date" in train_df.columns
                                 else train_df.index.month)[target_col].mean().to_dict()
    if "date" in test_df.columns:
        preds["Climatologia"] = test_df["date"].dt.month.map(clim_map).values
    else:
        preds["Climatologia"] = np.full(len(y_test), y_train.mean())

    return preds, X_test, y_test


def main():
    log.info("=" * 70)
    log.info("SCRIPT 18b: Métricas de umbral — alertas hidrológicas")
    log.info("=" * 70)

    df = load_data()

    all_rows = []

    # ── Precipitación ─────────────────────────────────────────────────────────
    for target_col, thresholds, threshold_type in [
        ("pr_next_1d",    PR_THRESHOLDS, "fixed_mm"),
        ("pr_sum_next_7d", [t * 7 for t in PR_THRESHOLDS], "fixed_mm"),  # escala 7d
    ]:
        log.info(f"\n[PR] Target: {target_col}")
        preds, _, y_test = train_and_predict(df, target_col)
        if preds is None:
            continue

        # Añadir umbrales de percentil sobre la serie observada
        obs_full = df[df["split"] == "test"].dropna(subset=[target_col])[target_col].values
        pct_thresholds = [(np.percentile(obs_full, p), f"p{p}") for p in [90, 95]]
        named_thresholds = [(t, f"{t:.1f}mm") for t in thresholds] + pct_thresholds

        for threshold, thr_label in named_thresholds:
            n_events = int(np.sum(y_test >= threshold))
            freq     = n_events / len(y_test)
            for model_name, y_pred in preds.items():
                if model_name == "y_true":
                    continue
                m = contingency_metrics(y_test, y_pred if hasattr(y_pred, "__len__")
                                        else np.array(y_pred), threshold)
                row = {
                    "target": target_col,
                    "model": model_name,
                    "threshold": threshold,
                    "threshold_label": thr_label,
                    "n_events_obs": n_events,
                    "event_freq": freq,
                    **m,
                }
                all_rows.append(row)
                log.info(f"   {model_name:<12} {thr_label:>8}: "
                         f"POD={m['POD']:.2f}  FAR={m['FAR']:.2f}  "
                         f"CSI={m['CSI']:.2f}  ETS={m['ETS']:.2f}  "
                         f"BIAS={m['BIAS']:.2f}  n_events={n_events}")

    # ── Caudal (si disponible) ─────────────────────────────────────────────────
    q_col_test = "q_next_1d"
    if q_col_test in df.columns:
        test_q = df[(df["split"] == "test")][q_col_test].dropna()
        if len(test_q) > 100:
            log.info(f"\n[Q] Target: {q_col_test}")
            preds_q, _, y_q = train_and_predict(df, q_col_test)
            if preds_q is not None:
                q_percentile_thresholds = [(np.percentile(y_q, p), f"p{p}") for p in Q_PERCENTILES]
                for threshold, thr_label in q_percentile_thresholds:
                    n_events = int(np.sum(y_q >= threshold))
                    for model_name, y_pred_q in preds_q.items():
                        if model_name == "y_true":
                            continue
                        m = contingency_metrics(y_q, np.asarray(y_pred_q), threshold)
                        row = {
                            "target": q_col_test,
                            "model": model_name,
                            "threshold": threshold,
                            "threshold_label": thr_label,
                            "n_events_obs": n_events,
                            "event_freq": n_events / len(y_q),
                            **m,
                        }
                        all_rows.append(row)

    # ── Guardar resultados ────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "D7_threshold_metrics.csv"
    df_out.to_csv(out_csv, index=False)
    log.info(f"\n   Guardado: {out_csv}")

    # ── Figura V14: POD/FAR/CSI por modelo ───────────────────────────────────
    log.info("\n[FIG] Generando figuras V14, V15 ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # V14: CSI por umbral y modelo (precipitación 1d)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        target_pr = "pr_next_1d"
        df_pr = df_out[df_out["target"] == target_pr]
        models_pr = [m for m in df_pr["model"].unique()]
        thresholds_pr = sorted(df_pr["threshold_label"].unique())
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(models_pr)))

        ax = axes[0]
        x = np.arange(len(thresholds_pr))
        w = 0.8 / max(1, len(models_pr))
        for i, (mdl, col) in enumerate(zip(models_pr, colors)):
            vals = [df_pr[(df_pr["model"] == mdl) & (df_pr["threshold_label"] == t)]["CSI"].mean()
                    for t in thresholds_pr]
            ax.bar(x + i * w - 0.4, vals, width=w, label=mdl, color=col, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds_pr, rotation=30, ha="right")
        ax.set_ylabel("CSI (Critical Success Index)")
        ax.set_title(f"CSI por umbral — {target_pr}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(0.3, color="orange", ls="--", lw=0.8, alpha=0.7, label="umbral mínimo")

        ax2 = axes[1]
        for i, (mdl, col) in enumerate(zip(models_pr, colors)):
            pods = [df_pr[(df_pr["model"] == mdl) & (df_pr["threshold_label"] == t)]["POD"].mean()
                    for t in thresholds_pr]
            fars = [df_pr[(df_pr["model"] == mdl) & (df_pr["threshold_label"] == t)]["FAR"].mean()
                    for t in thresholds_pr]
            ax2.plot(fars, pods, "o-", color=col, label=mdl, lw=2, ms=6)
        ax2.set_xlabel("FAR (False Alarm Rate)")
        ax2.set_ylabel("POD (Probability of Detection)")
        ax2.set_title(f"POD vs FAR — {target_pr}")
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)  # diagonal
        ax2.fill_between([0, 0.3], [0.7, 0.7], [1, 1], alpha=0.07,
                         color="green", label="zona ideal")

        fig.suptitle(
            "Script 18b: Métricas de umbral para alertas — HidroAlerta Chancay-Huaral\n"
            "D5v2 (PISCO QM-corregido) — test 2015-2019",
            fontsize=11
        )
        fig.tight_layout()
        fname14 = FIG_DIR / "V14_pod_far_csi.png"
        fig.savefig(fname14, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"   V14 guardada: {fname14.name}")

        # V15: Performance Diagram (BIAS vs CSI) — Mason & Graham style
        fig2, ax3 = plt.subplots(figsize=(8, 8))

        # Líneas CSI constante
        sr_vals = np.linspace(0.01, 1.0, 200)   # SR = 1 - FAR
        for csi_line in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pod_line = csi_line / (sr_vals - csi_line * sr_vals + csi_line)
            pod_line = np.clip(pod_line, 0, 1)
            ax3.plot(sr_vals, pod_line, ":", color="gray", lw=0.6, alpha=0.5)
            idx = np.argmin(np.abs(sr_vals - 0.9))
            if pod_line[idx] <= 1.0:
                ax3.text(sr_vals[idx], pod_line[idx], f"CSI={csi_line:.1f}",
                         fontsize=7, color="gray", alpha=0.8)

        # Líneas BIAS constante (POD / SR = BIAS)
        for bias_line in [0.5, 1.0, 2.0, 5.0]:
            sr_b = np.linspace(0.01, 1.0, 200)
            pod_b = bias_line * sr_b
            pod_b = np.clip(pod_b, 0, 1)
            ax3.plot(sr_b, pod_b, "-", color="lightblue", lw=0.8, alpha=0.7)

        # Puntos de los modelos (promedio sobre umbrales 5mm y 10mm)
        df_key = df_out[(df_out["target"] == "pr_next_1d") &
                        (df_out["threshold_label"].isin(["5.0mm", "10.0mm"]))]
        for i, (mdl, col) in enumerate(zip(models_pr, colors)):
            sub = df_key[df_key["model"] == mdl]
            if len(sub) == 0:
                continue
            sr  = 1 - sub["FAR"].mean()
            pod = sub["POD"].mean()
            ax3.scatter(sr, pod, s=120, color=col, zorder=5, label=mdl, edgecolors="k", lw=0.5)
            ax3.annotate(mdl, (sr, pod), textcoords="offset points",
                         xytext=(5, 5), fontsize=8)

        ax3.set_xlabel("Success Ratio (SR = 1 - FAR)")
        ax3.set_ylabel("POD (Probability of Detection)")
        ax3.set_title("Performance Diagram — Umbrales 5mm y 10mm/día\n"
                      "HidroAlerta Chancay-Huaral — test 2015-2019")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=9, loc="lower left")
        fig2.tight_layout()
        fname15 = FIG_DIR / "V15_performance_diagram.png"
        fig2.savefig(fname15, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        log.info(f"   V15 guardada: {fname15.name}")

    except Exception as e:
        log.warning(f"   Error generando figuras: {e}")
        import traceback
        log.warning(traceback.format_exc())

    # ── Resumen ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("RESUMEN MÉTRICAS DE UMBRAL — pr_next_1d, umbral 5mm")
    log.info(f"{'Modelo':<14} {'POD':>5} {'FAR':>5} {'CSI':>5} {'ETS':>5} {'BIAS':>5}")
    sub = df_out[(df_out["target"] == "pr_next_1d") & (df_out["threshold_label"] == "5.0mm")]
    for _, row in sub.iterrows():
        log.info(f"  {row['model']:<12} {row['POD']:>5.2f} {row['FAR']:>5.2f} "
                 f"{row['CSI']:>5.2f} {row['ETS']:>5.2f} {row['BIAS']:>5.2f}")
    log.info("")
    log.info(f"  Salida: {out_csv.name}")
    log.info("  Figuras: V14_pod_far_csi.png, V15_performance_diagram.png")
    log.info("  SIGUIENTE: python scripts/22_tft_training.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
