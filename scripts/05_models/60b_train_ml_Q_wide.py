#!/usr/bin/env python3
"""
Script 60b: Baseline ML target Q — FORMATO WIDE (metodológicamente correcto).

Diferencia clave con Script 60 (pooled long):
  - Script 60:  131,490 filas (9 entidades apiladas) → NSE inflado 9× (N falso)
  - Script 60b: 14,610 filas (1 por día) con features de las 9 sub-cuencas como columnas
                → NSE honesto (N real), captura gradiente altitudinal

Justificación: el target Q es ÚNICO por día (Q outlet). Apilarlo 9× replica el target
y crea N artificial. El wide pivot usa cada sub-cuenca como feature distinta.
Ver docs/SUBCUENCAS_DESIGN.md §2.

Salidas:
  outputs/ml_Q/leaderboard_Q_wide.csv
  outputs/ml_Q/feature_importance_Q_wide.csv
  outputs/figures/ml_Q/MQ07_wide_vs_pooled.png
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(ROOT/"outputs"/"60b_train_ml_Q_wide.log","w","utf-8")])
log = logging.getLogger("train_ml_Q_wide")

D6_CSV      = ROOT / "data/model_ready/D6_multientity.csv"
THRESH_FILE = ROOT / "data/model_ready/thresholds/q_thresholds.json"
Q_OBS_FILE  = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR     = ROOT / "outputs/ml_Q"
FIG_DIR     = ROOT / "outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

Q_STATION = "q_santo_domingo_47e214d2"
Q_CONV    = 86.4 / 3062.62
Q90       = 40.89
SEED      = 42

TARGETS = ["q_next_1d", "q_sum_next_7d"]

# Features que VARÍAN por sub-cuenca → pivotear (1 columna por entidad)
SPATIAL_FEATURES = ["pr_mm", "tmax_c", "tmin_c", "tmean_c", "pet_mm",
                    "api", "spi_30d", "spi_90d", "water_deficit_30d", "elevation_m"]
# Features IGUALES para todas las entidades en cada fecha → tomar 1 vez
SHARED_FEATURES  = ["oni_index", "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
                    "hydro_month", "is_wet_season", "clim_pr_p50", "clim_pr_p90"]


# ── Métricas (importadas conceptualmente de Script 60) ─────────────────────────
def _nse(o, p):
    return 1.0 - np.sum((o-p)**2) / (np.sum((o-o.mean())**2) + 1e-12)

def _nse_sqrt(o, p):
    w = np.maximum(o, 0.0) ** 0.5
    return 1.0 - np.sum(w*(o-p)**2) / (np.sum(w*(o-o.mean())**2) + 1e-12)

def compute_metrics(obs, pred, thr_mm):
    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    if len(o) < 5:
        return {k: np.nan for k in ["NSE","NSE_sqrt","KGE","MAE","POD","FAR","CSI","HSS","J_alert","N"]}
    r = float(np.corrcoef(o, p)[0,1])
    nse = float(_nse(o, p))
    nse_sq = float(_nse_sqrt(o, p))
    alpha = p.std()/(o.std()+1e-12); beta = p.mean()/(o.mean()+1e-12)
    kge = float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
    eo = o > thr_mm; ep = p > thr_mm
    TP=int(np.sum(eo&ep)); FP=int(np.sum(~eo&ep)); FN=int(np.sum(eo&~ep)); TN=int(np.sum(~eo&~ep))
    POD = TP/(TP+FN+1e-9); FAR = FP/(FP+TN+1e-9); CSI = TP/(TP+FP+FN+1e-9)
    dh = (TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)
    HSS = 2*(TP*TN-FP*FN)/(dh+1e-9) if dh>0 else 0.0
    j = 0.25*nse_sq + 0.25*nse + 0.30*CSI + 0.10*POD - 0.10*FAR
    return {"NSE":round(nse,4),"NSE_sqrt":round(nse_sq,4),"KGE":round(kge,4),
            "MAE":round(float(np.mean(np.abs(o-p))),4),
            "POD":round(POD,4),"FAR":round(FAR,4),"CSI":round(CSI,4),
            "HSS":round(HSS,4),"J_alert":round(j,4),"N":int(mask.sum()),
            "TP":TP,"FP":FP,"FN":FN}


# ── Construir formato WIDE ──────────────────────────────────────────────────────
def build_wide(D6: pd.DataFrame) -> pd.DataFrame:
    """1 fila por día. Features espaciales pivoteadas por sub-cuenca."""
    pivots = []
    for col in SPATIAL_FEATURES:
        if col not in D6.columns:
            continue
        piv = D6.pivot_table(index="date", columns="entity_id", values=col)
        piv.columns = [f"{col}_{e}" for e in piv.columns]
        pivots.append(piv)
    # Compartidas + target + split (tomar de sub_634, son iguales)
    ref = D6[D6["entity_id"] == "sub_634"].set_index("date")
    keep = [c for c in SHARED_FEATURES + TARGETS + ["split"] if c in ref.columns]
    pivots.append(ref[keep])
    wide = pd.concat(pivots, axis=1)
    return wide


def main():
    log.info("="*70)
    log.info("SCRIPT 60b: Baseline ML target Q — FORMATO WIDE (correcto)")
    log.info("="*70)

    D6 = pd.read_csv(D6_CSV, parse_dates=["date"])
    log.info(f"\n[1] D6 cargado: {D6.shape} (pooled long)")

    wide = build_wide(D6)
    log.info(f"[2] Wide construido: {wide.shape} (1 fila/día, N real)")

    thresholds = json.loads(THRESH_FILE.read_text()) if THRESH_FILE.exists() else {"Q90": Q90}
    q90_mm = thresholds.get("Q90", Q90) * Q_CONV   # m³/s → mm/d para comparar con targets

    q_obs = None
    if Q_OBS_FILE.exists():
        dfq = pd.read_csv(Q_OBS_FILE, index_col=0, parse_dates=True)
        if Q_STATION in dfq.columns:
            q_obs = dfq[Q_STATION].dropna()

    from sklearn.ensemble import RandomForestRegressor
    try:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        HAS_BOOST = True
    except ImportError:
        HAS_BOOST = False

    feature_cols = [c for c in wide.columns if c not in TARGETS + ["split"]]
    log.info(f"[3] Features wide: {len(feature_cols)} (espaciales×9 + compartidas)")

    train = wide[wide["split"]=="train"]
    val   = wide[wide["split"]=="val"]
    test  = wide[wide["split"]=="test"]
    log.info(f"    train={len(train)}, val={len(val)}, test={len(test)}")

    X_tr = train[feature_cols].fillna(0)
    X_vl = val[feature_cols].fillna(0)
    X_te = test[feature_cols].fillna(0)

    leaderboard = []
    feat_imp = []

    for target in TARGETS:
        log.info(f"\n[4] TARGET: {target} (WIDE)")
        y_tr = train[target].values
        y_vl = val[target].values
        y_te = test[target].values
        m_tr = np.isfinite(y_tr)

        models = [("RandomForest", RandomForestRegressor(
                    n_estimators=300, max_depth=6, min_samples_leaf=10,
                    max_features=0.7, random_state=SEED, n_jobs=-1))]
        if HAS_BOOST:
            models += [
                ("XGBoost",  XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=SEED, n_jobs=-1, verbosity=0)),
                ("LightGBM", LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=SEED, n_jobs=-1, verbosity=-1)),
            ]

        for name, model in models:
            model.fit(X_tr[m_tr], y_tr[m_tr])
            for split_name, X_s, y_s in [("val", X_vl, y_vl), ("test", X_te, y_te)]:
                pred = model.predict(X_s)
                met = compute_metrics(y_s, pred, q90_mm)
                leaderboard.append({"model": name, "target": target,
                                    "split": split_name, **met})
                log.info(f"    [{split_name}] {name}: NSE={met['NSE']:.4f} "
                         f"NSE_sqrt={met['NSE_sqrt']:.4f} J_alert={met['J_alert']:.4f}")
            if hasattr(model, "feature_importances_"):
                for c, imp in zip(feature_cols, model.feature_importances_):
                    feat_imp.append({"model": name, "target": target,
                                     "feature": c, "importance": float(imp)})

    # Guardar
    lb = pd.DataFrame(leaderboard)
    lb_pivot = lb.pivot_table(index=["model","target"], columns="split",
                              values=["NSE","NSE_sqrt","KGE","J_alert","CSI","MAE","N"])
    lb_pivot.columns = [f"{m}_{s}" for m,s in lb_pivot.columns]
    lb_pivot = lb_pivot.reset_index().sort_values("J_alert_test", ascending=False)
    lb_pivot.to_csv(OUT_DIR/"leaderboard_Q_wide.csv", index=False)

    pd.DataFrame(feat_imp).to_csv(OUT_DIR/"feature_importance_Q_wide.csv", index=False)

    log.info("\n" + "="*70)
    log.info("LEADERBOARD WIDE — NSE honesto (N real, sin inflación 9×)")
    cols = [c for c in ["model","target","J_alert_test","NSE_sqrt_test","NSE_test","KGE_test","N_test"]
            if c in lb_pivot.columns]
    log.info(lb_pivot[cols].to_string(index=False))

    # Comparación pooled vs wide
    pooled_path = OUT_DIR / "leaderboard_Q.csv"
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        log.info("\n=== POOLED (Script 60) vs WIDE (Script 60b) — q_next_1d RF ===")
        for fmt, lbf in [("POOLED", pooled), ("WIDE", lb_pivot)]:
            row = lbf[(lbf["model"]=="RandomForest") & (lbf["target"]=="q_next_1d")]
            if len(row) > 0:
                r = row.iloc[0]
                log.info(f"  {fmt:7s}: NSE_test={r.get('NSE_test',np.nan):.4f} "
                         f"N_test={int(r.get('N_test',0))}")
        log.info("  → WIDE tiene N real (~3940 test); POOLED infla 9× (~19728)")

    log.info("\nSCRIPT 60b COMPLETADO")


if __name__ == "__main__":
    main()
