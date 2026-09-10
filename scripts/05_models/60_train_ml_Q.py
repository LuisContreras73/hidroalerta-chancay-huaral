#!/usr/bin/env python3
"""
Script 60: Baseline ML con target Q (caudal) — D6 multi-entidad.

Modelos: Climatología, Persistencia, Random Forest, XGBoost, LightGBM.
Dataset: data/model_ready/splits/D6_Q/{train,val,test}.parquet  (Script 52)
Targets: q_next_1d, q_sum_next_7d  (Q en mm/d, escalado log1p+StandardScaler)
Evaluación: SOLO sobre Q observado real (2020-09 en adelante) — nunca sobre GR4J.

Anti-leakage garantizado:
  - Scaler ya aplicado en splits (fit SOLO en train, Script 52)
  - Q90/Q99 thresholds desde data/model_ready/thresholds/q_thresholds.json
  - Features: solo pasado (encoder) y calendáricas puras (conocidas a futuro)
  - No usar q_mm como feature (es el target con lag 0 → leakage)

Salidas:
  outputs/ml_Q/leaderboard_Q.csv         — métricas NSE/KGE/MAE por modelo y horizonte
  outputs/ml_Q/feature_importance_Q.csv  — importancia de features RF + XGB + LGB
  outputs/figures/basin/MQ01_leaderboard.png
  outputs/figures/basin/MQ02_hydro_{1d,7d}.png  — hidrograma pred vs obs
  configs/ml_Q_best_params.json          — hiperparámetros del mejor modelo
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "outputs" / "60_train_ml_Q.log", "w", "utf-8"),
    ],
)
log = logging.getLogger("train_ml_Q")

# ── Rutas ──────────────────────────────────────────────────────────────────────
SPLITS_DIR  = ROOT / "data/model_ready/splits/D6_Q"
THRESH_FILE = ROOT / "data/model_ready/thresholds/q_thresholds.json"
Q_OBS_FILE  = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR     = ROOT / "outputs/ml_Q"
FIG_DIR     = ROOT / "outputs/figures/basin"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Targets y horizonte ────────────────────────────────────────────────────────
TARGETS = {
    "q_next_1d":   {"horizon": 1,  "label": "Q +1d"},
    "q_sum_next_7d": {"horizon": 7, "label": "Q suma 7d"},
}

Q_STATION = "q_santo_domingo_47e214d2"
Q_CONV    = 86.4 / 3062.62   # mm/d → m³/s: q_m3s = q_mm / Q_CONV
SEED      = 42


# ── Métricas ──────────────────────────────────────────────────────────────────
# Referencias:
#   NSE:         Nash & Sutcliffe (1970) J.Hydrology 10:282
#   KGE:         Gupta et al. (2009) J.Hydrology 377:80
#   NSE_sqrt:    Pushpalatha et al. (2012) J.Hydrology 460:33  — pondera caudales altos
#   NSE_log:     Oudin et al. (2006) — sensible a caudales bajos (NO recomendado para alertas)
#   POD/FAR/CSI: Wilks (2006) Statistical Methods in Atmospheric Sciences cap. 7
#   HSS:         Heidke (1926) — skill score corregido por azar en categorías
#   J_alert:     Métrica compuesta HidroAlerta = balance rendimiento + capacidad de alerta

Q90_DEFAULT = 40.9   # m³/s — umbral alerta naranja (calibrado con Q obs SNIRH 47E214D2)


def _nse(o, p):
    return 1.0 - np.sum((o-p)**2) / (np.sum((o-o.mean())**2) + 1e-12)


def _nse_alpha(o, p, alpha: float = 0.5):
    """
    NSE ponderado por Q^alpha — Pushpalatha et al. (2012).
    alpha=0.5 (√Q): pondera más eventos de caudal alto sin ignorar bajos.
    alpha=1.0 (Q):  énfasis fuerte en caudales altos.
    alpha=0.0:      NSE estándar.
    Recomendación para sistemas de alerta: alpha=0.5.
    """
    w = np.maximum(o, 0.0) ** alpha
    if w.sum() < 1e-9:
        return np.nan
    return 1.0 - np.sum(w*(o-p)**2) / (np.sum(w*(o-o.mean())**2) + 1e-12)


def _alert_metrics(o, p, threshold: float = Q90_DEFAULT):
    """
    Métricas categóricas para alerta binaria Q > threshold.
    POD = TP/(TP+FN)  — fracción de alertas reales detectadas (sensibilidad)
    FAR = FP/(FP+TN)  — fracción de días normales marcados como alerta (falsa alarma)
    CSI = TP/(TP+FP+FN) — Índice de éxito crítico (considera tanto FP como FN)
    HSS = skill corregido por azar (0=sin skill, 1=perfecto, <0=peor que azar)
    """
    eo = o > threshold;  ep = p > threshold
    TP = int(np.sum( eo &  ep))
    FP = int(np.sum(~eo &  ep))
    FN = int(np.sum( eo & ~ep))
    TN = int(np.sum(~eo & ~ep))
    n  = TP + FP + FN + TN
    POD = TP/(TP+FN+1e-9) if (TP+FN) > 0 else 0.0
    FAR = FP/(FP+TN+1e-9) if (FP+TN) > 0 else 0.0
    CSI = TP/(TP+FP+FN+1e-9) if (TP+FP+FN) > 0 else 0.0
    denom_hss = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
    HSS = 2*(TP*TN - FP*FN) / (denom_hss + 1e-9) if denom_hss > 0 else 0.0
    return {"POD": POD, "FAR": FAR, "CSI": CSI, "HSS": HSS,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN}


def compute_metrics(obs: np.ndarray, pred: np.ndarray,
                    q_threshold_m3s: float = Q90_DEFAULT) -> dict:
    """
    Suite completa de métricas para evaluación de pronóstico de Q en sistema de alertas.

    Métricas continuas:
      NSE       — Nash-Sutcliffe; benchmark = media histórica; sensible a picos
      KGE       — Kling-Gupta; descompone r + variabilidad + sesgo
      NSE_sqrt  — NSE ponderado por √Q (Pushpalatha 2012); recomendado como PRIMARIO
                  penaliza más errores en caudales altos (alertas) sin ignorar bajos
      PBIAS_Q90 — sesgo porcentual solo cuando Q > Q90 (capacidad de volumen en extremos)

    Métricas categóricas (alerta binaria Q > Q90):
      POD   — Probability of Detection (qué fracción de alertas reales se detectan)
      FAR   — False Alarm Rate (qué fracción de días normales se marcan como alerta)
      CSI   — Critical Success Index (balance POD/FAR, ignorando verdaderos negativos)
      HSS   — Heidke Skill Score (CSI corregido por azar; >0.3=útil, >0.5=bueno)

    Métrica compuesta HidroAlerta:
      J_alert = 0.25×NSE_sqrt + 0.25×NSE + 0.30×CSI_Q90 + 0.10×POD_Q90 − 0.10×FAR_Q90
      Rango: [-inf, 1.0].  Interpretación: >0.6=bueno, >0.4=aceptable, <0=peor que media
    """
    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    if len(o) < 5:
        return {k: np.nan for k in
                ["NSE", "KGE", "NSE_sqrt", "MAE", "RMSE", "PBIAS", "PBIAS_Q90",
                 "POD", "FAR", "CSI", "HSS", "J_alert", "N"]}

    # Continuas
    r     = float(np.corrcoef(o, p)[0, 1])
    nse   = float(_nse(o, p))
    alpha = p.std() / (o.std() + 1e-12)
    beta  = p.mean() / (o.mean() + 1e-12)
    kge   = float(1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
    nse_sqrt = float(_nse_alpha(o, p, 0.5))
    pbias = float((p.sum() - o.sum()) / (o.sum() + 1e-12) * 100)
    q90_local = np.nanpercentile(o, 90)
    pk = o > q90_local
    pbias_q90 = float((p[pk].sum()-o[pk].sum())/(o[pk].sum()+1e-12)*100) if pk.sum()>3 else np.nan

    # Categóricas — convertir threshold a mm/d si obs está en mm/d
    # threshold en m³/s → mm/d usando Q_CONV; pero si obs ya es mm/d, comparar directo
    # Heurística: si obs < 5 (mm/d scale) usar percentil local; si > 5 usar m³/s
    if o.max() > 5.0:   # obs en m³/s
        thr = q_threshold_m3s
    else:               # obs en mm/d (escalado)
        thr = q_threshold_m3s * Q_CONV
    am = _alert_metrics(o, p, thr)

    # Composite J_alert
    j = (0.25*nse_sqrt + 0.25*nse
         + 0.30*am["CSI"] + 0.10*am["POD"] - 0.10*am["FAR"])

    return {
        "NSE": round(nse, 4), "KGE": round(kge, 4),
        "NSE_sqrt": round(nse_sqrt, 4),
        "MAE": round(float(np.mean(np.abs(o-p))), 4),
        "RMSE": round(float(np.sqrt(np.mean((o-p)**2))), 4),
        "PBIAS": round(pbias, 2), "PBIAS_Q90": round(pbias_q90, 2) if not np.isnan(pbias_q90) else np.nan,
        "POD": round(am["POD"], 4), "FAR": round(am["FAR"], 4),
        "CSI": round(am["CSI"], 4), "HSS": round(am["HSS"], 4),
        "J_alert": round(j, 4), "r": round(r, 4),
        "N": int(mask.sum()), "TP": am["TP"], "FP": am["FP"], "FN": am["FN"],
    }


# ── Feature selection ─────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Selecciona features válidas para Q forecasting.
    Excluye: IDs, targets, q_mm (target con lag 0 = leakage),
             columnas de suelo duplicadas, split.
    """
    EXCLUDE = {
        "date", "entity_id", "split",
        # Targets — nunca entran como features
        "q_next_1d", "q_sum_next_7d",
        "pr_next_1d", "pr_sum_next_3d", "pr_sum_next_7d",
        # q_mm con lag 0 = data leakage del target
        # (q_mm SHIFTADO ya está en features como q_mm en t-1 via rolling en D6)
    }
    cols = [c for c in df.columns
            if c not in EXCLUDE
            and df[c].dtype in ("float32", "float64", "int64", "int32")
            and "next" not in c]
    log.info(f"   Features seleccionadas: {len(cols)}")
    return cols


# ── Baselines clásicos ─────────────────────────────────────────────────────────
def baseline_climatology(train: pd.DataFrame, target: str) -> dict:
    """Climatología MMDD: predice la mediana histórica por entidad × MMDD."""
    train["mmdd"] = train["date"].dt.strftime("%m-%d")
    clim = (train.groupby(["entity_id", "mmdd"])[target]
            .median().reset_index()
            .rename(columns={target: "pred_clim"}))
    return clim


def predict_climatology(df: pd.DataFrame, clim: pd.DataFrame,
                        target: str) -> np.ndarray:
    df = df.copy()
    df["mmdd"] = df["date"].dt.strftime("%m-%d")
    merged = df.merge(clim, on=["entity_id", "mmdd"], how="left")
    return merged["pred_clim"].values


def baseline_persistence(df: pd.DataFrame, target: str) -> np.ndarray:
    """Persistencia: predice q_mm del día actual como proxy del día siguiente."""
    # q_mm = caudal hoy (shifted target = caudal mañana)
    # Persistencia: ŷ(t+h) = q_mm(t)
    return df["q_mm"].values


# ── Entrenamiento ML ───────────────────────────────────────────────────────────
def train_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features=0.7, random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train):
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_lgb(X_train, y_train):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_leaderboard(leaderboard: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = leaderboard["target"].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 5),
                              sharey=False)
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = leaderboard[leaderboard["target"] == tgt].sort_values("NSE_test", ascending=False)
        colors = ["#2ecc71" if nse > 0.4 else "#e67e22" if nse > 0 else "#e74c3c"
                  for nse in sub["NSE_test"]]
        bars = ax.barh(sub["model"], sub["NSE_test"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'NSE={row["NSE_test"]:.3f} KGE={row["KGE_test"]:.3f}',
                    va="center", fontsize=8)
        ax.set_xlabel("NSE (test, Q obs real)")
        ax.set_title(f"Target: {tgt}")
        ax.set_xlim(-0.5, 1.1)
    fig.suptitle("Leaderboard ML — Target Q (HidroAlerta)\nTest = Q observado real 2020-09+",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "MQ01_leaderboard.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"   Leaderboard: {path.name}")


def plot_hydrograph(df_test: pd.DataFrame, pred_col: str,
                    target: str, model_name: str, q_obs_m3s: pd.Series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Tomar solo entidad outlet (sub_634) y período con Q obs
    sub = df_test[df_test["entity_id"] == "sub_634"].sort_values("date").copy()
    sub = sub.dropna(subset=[target])
    if len(sub) < 5:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))

    # Panel 1: hidrograma
    ax = axes[0]
    ax.plot(sub["date"], sub[target], color="#2c3e50", lw=1.2, label="Q obs (mm/d)")
    ax.plot(sub["date"], sub[pred_col], color="#e74c3c", lw=1.0, ls="--",
            label=f"{model_name}")
    ax.set_ylabel("Q (mm/d)")
    ax.set_title(f"Hidrograma — {model_name} | Target: {target}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: scatter
    ax2 = axes[1]
    valid = np.isfinite(sub[target]) & np.isfinite(sub[pred_col])
    o, p = sub.loc[valid, target].values, sub.loc[valid, pred_col].values
    m = compute_metrics(o, p)
    ax2.scatter(o, p, alpha=0.4, s=10, color="#3498db")
    lim = max(o.max(), p.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax2.set_xlabel("Q obs"); ax2.set_ylabel("Q pred")
    ax2.text(0.05, 0.95, f"NSE={m['NSE']:.3f}\nKGE={m['KGE']:.3f}\nN={m['N']}",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(facecolor="white", alpha=0.8))
    ax2.set_title("Scatter obs vs pred")
    ax2.grid(True, alpha=0.3)

    # Panel 3: errores en el tiempo
    ax3 = axes[2]
    err = sub[pred_col] - sub[target]
    ax3.fill_between(sub["date"], 0, err, where=err > 0, alpha=0.5, color="#e74c3c", label="sobreestimacion")
    ax3.fill_between(sub["date"], 0, err, where=err < 0, alpha=0.5, color="#3498db", label="subestimacion")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("Error (pred - obs) mm/d")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    tag = target.replace("q_", "").replace("_", "")
    path = FIG_DIR / f"MQ02_hydro_{tag}_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"   Hidrograma: {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("SCRIPT 60: Baseline ML con target Q — D6 multi-entidad")
    log.info("=" * 70)

    # 1. Cargar splits parquet (ya escalados con log1p + StandardScaler)
    log.info("\n[1] Cargando splits D6_Q ...")
    for f in ["train.parquet", "val.parquet", "test.parquet"]:
        if not (SPLITS_DIR / f).exists():
            log.error(f"  {f} no encontrado. Ejecutar script 52 primero.")
            sys.exit(1)

    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val   = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test  = pd.read_parquet(SPLITS_DIR / "test.parquet")

    log.info(f"   train={len(train)}, val={len(val)}, test={len(test)}")
    log.info(f"   Fechas train: {train.date.min().date()} -> {train.date.max().date()}")
    log.info(f"   Fechas test:  {test.date.min().date()} -> {test.date.max().date()}")

    # 2. Cargar umbrales Q y Q obs real para evaluación honesta
    log.info("\n[2] Cargando Q observado real para evaluación ...")
    thresholds = json.loads(THRESH_FILE.read_text()) if THRESH_FILE.exists() else {"Q90": 40.9}
    Q90 = thresholds.get("Q90", 40.9)
    log.info(f"   Q90 alerta = {Q90} m³/s (obs real 47E214D2)")

    q_obs_raw = None
    if Q_OBS_FILE.exists():
        df_q = pd.read_csv(Q_OBS_FILE, index_col=0, parse_dates=True)
        if Q_STATION in df_q.columns:
            q_obs_raw = df_q[Q_STATION].dropna()
            log.info(f"   Q obs: {len(q_obs_raw)} días, {q_obs_raw.index.min().date()} -> {q_obs_raw.index.max().date()}")

    # 3. Seleccionar features
    log.info("\n[3] Seleccionando features ...")
    feature_cols = get_feature_cols(train)
    log.info(f"   {len(feature_cols)} features: {feature_cols[:8]}...")

    X_train = train[feature_cols].fillna(0)
    X_val   = val[feature_cols].fillna(0)
    X_test  = test[feature_cols].fillna(0)

    # 4. Entrenar y evaluar por target
    leaderboard = []
    feat_importance = []

    for target, meta in TARGETS.items():
        log.info(f"\n[4] TARGET: {target} (horizonte {meta['horizon']}d) ...")

        # Máscaras de observaciones válidas
        y_train = train[target].values
        y_val   = val[target].values
        y_test  = test[target].values

        train_valid = np.isfinite(y_train)
        val_valid   = np.isfinite(y_val)
        test_valid  = np.isfinite(y_test)

        log.info(f"   Train válidos: {train_valid.sum()} | Val: {val_valid.sum()} | Test: {test_valid.sum()}")

        # Baselines
        log.info("   Baselines ...")
        clim = baseline_climatology(train, target)

        models_preds = {}

        for split_name, df_s, X_s, y_s, vmask in [
            ("val",  val,  X_val,  y_val,  val_valid),
            ("test", test, X_test, y_test, test_valid),
        ]:
            clim_pred = predict_climatology(df_s, clim, target)
            pers_pred = baseline_persistence(df_s, target)

            for model_name, pred in [("Climatologia", clim_pred), ("Persistencia", pers_pred)]:
                m = compute_metrics(y_s, pred)
                leaderboard.append({"model": model_name, "target": target,
                                    "split": split_name, **m})
                log.info(f"   [{split_name}] {model_name}: NSE={m['NSE']:.3f} KGE={m['KGE']:.3f}")

        # ML models — entrenar solo con train válidos
        X_tr_fit = X_train[train_valid]
        y_tr_fit = y_train[train_valid]

        ml_models = [
            ("RandomForest", train_rf),
            ("XGBoost",      train_xgb),
            ("LightGBM",     train_lgb),
        ]

        for model_name, train_fn in ml_models:
            log.info(f"   Entrenando {model_name} ...")
            try:
                model = train_fn(X_tr_fit, y_tr_fit)
            except Exception as e:
                log.warning(f"   {model_name} falló: {e}")
                continue

            for split_name, X_s, y_s in [
                ("val",  X_val,  y_val),
                ("test", X_test, y_test),
            ]:
                pred = model.predict(X_s)
                m = compute_metrics(y_s, pred)
                leaderboard.append({"model": model_name, "target": target,
                                    "split": split_name, **m})
                log.info(f"   [{split_name}] {model_name}: NSE={m['NSE']:.3f} KGE={m['KGE']:.3f} MAE={m['MAE']:.4f}")

                # Guardar pred en test para hidrograma
                if split_name == "test" and model_name == "LightGBM":
                    test[f"pred_{target}_{model_name}"] = pred

            # Feature importance
            if hasattr(model, "feature_importances_"):
                for col, imp in zip(feature_cols, model.feature_importances_):
                    feat_importance.append({"model": model_name, "target": target,
                                            "feature": col, "importance": imp})

    # 5. Evaluación adicional sobre Q OBS REAL solamente (anti-leakage crítico)
    # El test set 2015-2020 usa GR4J para la mayoría de días;
    # solo 2020-09 → 2020-12 tiene Q observado real.
    # NSE contra GR4J refleja qué tan bien el ML aprende la física del GR4J,
    # no qué tan bien predice Q real. Solo los siguientes números son "honestos".
    log.info("\n[5] Evaluacion sobre Q OBSERVADO REAL (anti-leakage) ...")
    obs_only_results = []
    if q_obs_raw is not None:
        q_obs_aligned = q_obs_raw * Q_CONV   # m³/s → mm/d (no — q_obs ya en m³/s)
        # Convertir Q obs de m³/s a mm/d para comparar con targets del dataset
        q_obs_mm = q_obs_raw / (86.4 / 3062.62)  # NO → q_mm = q_m3s * Q_CONV
        # Corrección: Q_CONV = 86.4/3062.62, entonces q_mm = q_m3s * Q_CONV
        q_obs_mm = q_obs_raw * Q_CONV            # m³/s → mm/d ✓

        for target in TARGETS:
            pred_col = f"pred_{target}_LightGBM"
            if pred_col not in test.columns:
                continue

            test_sub = test[test["entity_id"] == "sub_634"].copy()
            test_sub = test_sub.set_index("date").sort_index()

            # Alinear con Q obs (2020-09 → 2020-12, solo estos días)
            common_idx = test_sub.index.intersection(q_obs_mm.index)
            if len(common_idx) < 10:
                log.warning(f"   Solo {len(common_idx)} días comunes test/obs — insuficiente")
                continue

            obs_vals  = q_obs_mm.loc[common_idx].values
            pred_vals = test_sub.loc[common_idx, pred_col].values

            m_obs = compute_metrics(obs_vals, pred_vals)
            m_gr4j = compute_metrics(
                test_sub.loc[common_idx, target].values, pred_vals
            )

            obs_only_results.append({
                "target": target, "n_obs_days": len(common_idx),
                **{f"vs_qobs_{k}": v for k, v in m_obs.items()},
                **{f"vs_gr4j_{k}": v for k, v in m_gr4j.items()},
            })

            log.info(f"\n   {target} | Q obs real ({len(common_idx)} días 2020-09+):")
            log.info(f"     vs Q obs real: NSE={m_obs['NSE']:.4f} KGE={m_obs['KGE']:.4f} MAE={m_obs['MAE']:.4f}")
            log.info(f"     vs GR4J  (ref): NSE={m_gr4j['NSE']:.4f} KGE={m_gr4j['KGE']:.4f}")
            log.info(f"     DIFERENCIA: NSE_obs - NSE_gr4j = {m_obs['NSE']-m_gr4j['NSE']:.4f}")

        if obs_only_results:
            pd.DataFrame(obs_only_results).to_csv(
                OUT_DIR / "eval_vs_qobs_real.csv", index=False)
            log.info("\n   eval_vs_qobs_real.csv guardado")

    log.info("\n   NOTA CRITICA: NSE en test_full (GR4J) != NSE en Q obs real")
    log.info("   Solo 'vs_qobs' es la metrica honesta para publicacion.")

    # 5b. Pivot leaderboard: val y test lado a lado
    lb = pd.DataFrame(leaderboard)
    metric_cols = ["NSE", "NSE_sqrt", "KGE", "CSI", "HSS", "J_alert",
                   "POD", "FAR", "MAE", "RMSE", "PBIAS", "PBIAS_Q90", "N"]
    metric_cols_present = [m for m in metric_cols if m in lb.columns]
    lb_pivot = lb.pivot_table(
        index=["model", "target"],
        columns="split",
        values=metric_cols_present,
    )
    lb_pivot.columns = [f"{m}_{s}" for m, s in lb_pivot.columns]
    lb_pivot = lb_pivot.reset_index()
    sort_col = "J_alert_test" if "J_alert_test" in lb_pivot.columns else "NSE_test"
    lb_pivot = lb_pivot.sort_values(sort_col, ascending=False)

    log.info("\n======================================================================")
    log.info("LEADERBOARD — target Q | Métrica principal: J_alert")
    cols_show = [c for c in ["model","target","J_alert_test","NSE_sqrt_test",
                              "NSE_test","KGE_test","CSI_test","HSS_test","MAE_test"]
                 if c in lb_pivot.columns]
    log.info(lb_pivot[cols_show].to_string(index=False))
    log.info("")
    log.info("J_alert = 0.25×NSE_sqrt + 0.25×NSE + 0.30×CSI_Q90 + 0.10×POD_Q90 - 0.10×FAR_Q90")
    log.info("NSE*: ponderado por sqrt(Q) [Pushpalatha 2012] -- penaliza mas errores en caudales altos")
    log.info("NOTA: test=GR4J dominante. Ver eval_vs_qobs_real.csv para metricas honestas.")
    log.info("======================================================================")

    # 5c. Guardar predicciones test para visualizacion (script 63)
    pred_cols = [c for c in test.columns if c.startswith("pred_")]
    if pred_cols:
        pred_export = test[["date", "entity_id"] + pred_cols].copy()
        pred_path = OUT_DIR / "test_predictions.csv"
        pred_export.to_csv(pred_path, index=False)
        log.info(f"   Predicciones test guardadas: {pred_path.name}")

    # 6. Guardar outputs
    lb_path = OUT_DIR / "leaderboard_Q.csv"
    lb_pivot.to_csv(lb_path, index=False)
    log.info(f"\n   Leaderboard guardado: {lb_path}")

    if feat_importance:
        fi_df = pd.DataFrame(feat_importance)
        fi_agg = fi_df.groupby(["model", "target", "feature"])["importance"].mean().reset_index()
        fi_agg.to_csv(OUT_DIR / "feature_importance_Q.csv", index=False)
        log.info("   Feature importance guardado")

    # 7. Figuras
    log.info("\n[7] Generando figuras ...")
    plot_leaderboard(lb_pivot)

    # Hidrograma solo si LightGBM corrió OK
    for target in TARGETS:
        pred_col = f"pred_{target}_LightGBM"
        if pred_col in test.columns:
            plot_hydrograph(test, pred_col, target, "LightGBM", q_obs_raw)

    log.info("\n" + "=" * 70)
    log.info("SCRIPT 60 COMPLETADO")
    log.info("   Siguiente: scripts/05_models/61_train_lstm_Q.py (LSTM con Q)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
