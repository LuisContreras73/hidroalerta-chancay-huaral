#!/usr/bin/env python3
"""
Script 75: Ensemble Final — LightGBM AR + TFT v2 con calibración conformal

Combina el mejor determinístico (LightGBM AR, NSE=0.966 en 2024-2025)
con el mejor probabilístico (TFT v2, POD_prob=94%, Scr74).

Pipeline:
  1. Retrain LightGBM AR (train ≤2023-12-31, mismo split de Scr72)
     → predicciones para todo 2024-2025
  2. Cargar TFT v2 predictions (tft_v2_predictions.csv)
  3. Ensemble P50:  w = NSE_i / sum(NSE_i)  (NSE-weighted average)
  4. Calibración conformal post-hoc de las bandas TFT:
       Cal set  : 2024-01-09 → 2024-06-30  (N≈174, primer semestre test)
       Eval set : 2024-07-01 → fin test     (N≈249, incluye temporada húmeda)
       Score  s = max(P10 - obs, obs - P90)
       q_conf = cuantil (1-α)(1+1/N_cal) de {s_i} → garantiza ≥80 % cobertura
       P10_adj = P10 - q_conf   |   P90_adj = P90 + q_conf
  5. Métricas finales: Ensemble vs LGB vs TFT v2 (P50 + banda calibrada)
  6. Producto operativo: ensemble_predictions.csv + TFT03_ensemble_final.png

Entorno: .venv  (Python 3.14, lightgbm, pandas, matplotlib)

Salidas:
  outputs/ml_Q/ensemble_predictions.csv
  outputs/figures/ml_Q/TFT03_ensemble_final.png
  outputs/figures/ml_Q/TFT04_model_leaderboard.png
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("ensemble")

D6_CSV   = ROOT / "data/model_ready/D6_multientity.csv"
QOBS     = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
TFT_PRED = ROOT / "outputs/ml_Q/tft_v2_predictions.csv"
OUT_DIR  = ROOT / "outputs/ml_Q"
FIG_DIR  = ROOT / "outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

Q_CONV = 86.4 / 3062.62
Q90    = 40.89
SEED   = 42
ALPHA  = 0.80   # cobertura objetivo para calibración conformal

SPATIAL  = ["pr_mm", "tmax_c", "tmin_c", "pet_mm", "api",
             "spi_30d", "spi_90d", "water_deficit_30d"]
SHARED   = ["oni_index", "sin_doy_1", "cos_doy_1", "hydro_month", "is_wet_season"]
KEY_SUBS = ["sub_649", "sub_655", "sub_656"]


# ── Feature engineering (= Scripts 72/74) ─────────────────────────────────────

def build_wide(D6):
    cols = {}
    for c in SPATIAL:
        piv = D6.pivot_table(index="date", columns="entity_id", values=c)
        cols[f"{c}_basin"] = piv.mean(axis=1)
        for e in KEY_SUBS:
            if e in piv.columns:
                cols[f"{c}_{e}"] = piv[e]
    df = pd.DataFrame(cols)
    ref = D6[D6["entity_id"] == "sub_634"].set_index("date")
    df = df.join(ref[SHARED + ["q_mm", "q_next_1d", "q_sum_next_7d"]]).sort_index()
    q = df["q_mm"]
    df["q_lag7"]  = q.shift(7)
    df["q_roll7"] = q.rolling(7,  min_periods=3).mean()
    df["q_roll30"]= q.rolling(30, min_periods=10).mean()
    return df


# ── Métricas ──────────────────────────────────────────────────────────────────

def _nse(o, p):
    return 1 - np.sum((o-p)**2) / (np.sum((o-o.mean())**2) + 1e-12)

def _nse_sqrt(o, p):
    w = np.maximum(o, 0)**0.5
    return 1 - np.sum(w*(o-p)**2) / (np.sum(w*(o-o.mean())**2) + 1e-12)

def _kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    a = p.std() / (o.std() + 1e-12)
    b = p.mean() / (o.mean() + 1e-12)
    return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)

def _lognse(o, p):
    lo = np.log(o+1); lp = np.log(np.clip(p, 0, None)+1)
    return 1 - np.sum((lo-lp)**2) / (np.sum((lo-lo.mean())**2) + 1e-12)

def det_metrics(o, p, thr, label=""):
    eo, ep = o > thr, p > thr
    TP = int(np.sum(eo & ep)); FP = int(np.sum(~eo & ep))
    FN = int(np.sum(eo & ~ep)); TN = int(np.sum(~eo & ~ep))
    POD = TP / (TP + FN + 1e-9)
    FAR = FP / (FP + TN + 1e-9)
    CSI = TP / (TP + FP + FN + 1e-9)
    j   = 0.25*_nse_sqrt(o,p) + 0.25*_nse(o,p) + 0.30*CSI + 0.10*POD - 0.10*FAR
    dh  = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
    HSS = 2*(TP*TN - FP*FN) / (dh + 1e-9) if dh > 0 else 0
    return {
        "model": label,
        "NSE":      round(_nse(o,p),    3),
        "NSE_sqrt": round(_nse_sqrt(o,p),3),
        "logNSE":   round(_lognse(o,p), 3),
        "KGE":      round(_kge(o,p),    3),
        "J_alert":  round(j,            4),
        "POD":      round(POD,          3),
        "FAR":      round(FAR,          3),
        "CSI":      round(CSI,          3),
        "HSS":      round(HSS,          3),
        "N":        len(o),
    }

def band_metrics(o, p10, p90, label=""):
    inside   = (o >= p10) & (o <= p90)
    coverage = float(inside.mean())
    width    = float((p90 - p10).mean()) / Q_CONV
    IS = ((p90-p10) + (2/ALPHA)*np.maximum(p10-o, 0) + (2/ALPHA)*np.maximum(o-p90, 0))
    eo = o > Q90*Q_CONV; ep90 = p90 > Q90*Q_CONV
    TP90 = int(np.sum(eo & ep90)); FN90 = int(np.sum(eo & ~ep90)); FP90 = int(np.sum(~eo & ep90))
    TN90 = int(np.sum(~eo & ~ep90))
    POD_prob = TP90 / (TP90 + FN90 + 1e-9)
    FAR_prob = FP90 / (FP90 + TN90 + 1e-9)
    return {
        "model":       label,
        "Coverage80":  round(coverage,  3),
        "Width_m3s":   round(width,     2),
        "IS":          round(float(IS.mean()), 4),
        "POD_prob":    round(POD_prob,  3),
        "FAR_prob":    round(FAR_prob,  3),
    }


# ── Calibración conformal ──────────────────────────────────────────────────────

def conformal_calibrate(obs_cal, p10_cal, p90_cal, alpha=ALPHA):
    """Calcula el margen q_conf tal que Coverage ≥ alpha en el cal set.

    Nonconformity score: s = max(P10 - obs, obs - P90, 0)  (distancia al borde)
    q_conf = cuantil ceil((1-alpha)(1+1/n)) de los scores.
    """
    scores = np.maximum(np.maximum(p10_cal - obs_cal, obs_cal - p90_cal), 0)
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n   # quantile level ajustado
    level = min(level, 1.0)
    q_conf = float(np.quantile(scores, level))
    log.info(f"  Conformal cal: n={n}, α={alpha}, level={level:.3f}, q_conf={q_conf:.4f} mm/d "
             f"({q_conf/Q_CONV:.2f} m³/s)")
    return q_conf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from lightgbm import LGBMRegressor

    log.info("Script 75 — Ensemble Final LightGBM + TFT v2")

    # ── 1. Cargar datos y features ────────────────────────────────────────────
    D6    = pd.read_csv(D6_CSV, parse_dates=["date"])
    qobs  = pd.read_csv(QOBS, index_col=0, parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm = qobs * Q_CONV

    df    = build_wide(D6)
    dates = df.index
    feat  = [c for c in df.columns if c not in ["q_next_1d", "q_sum_next_7d"]]
    log.info(f"Features: {len(feat)} | {dates.min().date()} → {dates.max().date()}")

    m_tr = dates <= "2023-12-31"   # mismo split que Script 72
    m_te = dates >= "2024-01-01"

    # ── 2. Cargar TFT v2 predictions ──────────────────────────────────────────
    tft_df = pd.read_csv(TFT_PRED, parse_dates=["date"])
    log.info(f"TFT v2 predictions: {tft_df.shape} | {tft_df.date.min().date()} → {tft_df.date.max().date()}")

    all_rows_det  = []
    all_rows_band = []
    all_preds_out = []

    targets_cfg = {
        "q_next_1d":     (qobs_mm.shift(-1),                            1),
        "q_sum_next_7d": (qobs_mm.shift(-1).rolling(7).sum().shift(-6), 7),
    }

    for target, (obs_real, hmult) in targets_cfg.items():
        log.info(f"\n{'='*55}\n[{target}]\n{'='*55}")
        thr = Q90 * Q_CONV * hmult

        # ── 2a. LightGBM AR (retrain ≤2023) ──────────────────────────────
        y   = df[target].values
        trm = m_tr & np.isfinite(y)
        lgb = LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=-1, verbosity=-1)
        lgb.fit(df.loc[trm, feat].fillna(0), y[trm])
        lgb_pred = pd.Series(lgb.predict(df.loc[m_te, feat].fillna(0)), index=dates[m_te])
        log.info(f"  LightGBM retrenado: train={trm.sum()} días")

        # ── 2b. TFT v2 predictions (ya filtradas a 2024-2025) ────────────
        tft_t = tft_df[tft_df["target"] == target].set_index("date").sort_index()

        # ── 2c. Q obs real en período test ────────────────────────────────
        o_full = obs_real.reindex(tft_t.index)   # alíneado al índice TFT (fechas con preds)
        valid  = o_full.notna()

        o_v     = o_full[valid].values
        dte_v   = tft_t.index[valid]

        # LightGBM en las mismas fechas
        lgb_v   = lgb_pred.reindex(tft_t.index[valid]).values
        p10_v   = tft_t["q_p10"][valid].values * Q_CONV * hmult
        p50_v   = tft_t["q_p50"][valid].values * Q_CONV * hmult
        p90_v   = tft_t["q_p90"][valid].values * Q_CONV * hmult

        # ── 3. Ensemble P50: pesos proporcionales a NSE ────────────────
        nse_lgb = _nse(o_v, lgb_v)
        nse_tft = _nse(o_v, p50_v)
        # Asegurar pesos positivos (clip NSE mínimo 0.01)
        w_lgb   = max(nse_lgb, 0.01)
        w_tft   = max(nse_tft, 0.01)
        w_total = w_lgb + w_tft
        w_lgb  /= w_total; w_tft /= w_total
        ens_v   = w_lgb * lgb_v + w_tft * p50_v
        log.info(f"  Pesos ensemble: LGB={w_lgb:.2f} (NSE={nse_lgb:.3f}) | "
                 f"TFT={w_tft:.2f} (NSE={nse_tft:.3f})")

        # ── 4. Calibración conformal ──────────────────────────────────────
        # Cal set: primer semestre 2024 | Eval set: resto
        m_cal  = dte_v < pd.Timestamp("2024-07-01")
        m_eval = ~m_cal

        q_conf = conformal_calibrate(o_v[m_cal], p10_v[m_cal], p90_v[m_cal])
        p10_cal = p10_v - q_conf
        p90_cal = p90_v + q_conf

        # Cobertura antes y después en eval set
        cov_raw = float(((o_v[m_eval] >= p10_v[m_eval]) & (o_v[m_eval] <= p90_v[m_eval])).mean())
        cov_adj = float(((o_v[m_eval] >= p10_cal[m_eval]) & (o_v[m_eval] <= p90_cal[m_eval])).mean())
        log.info(f"  Cobertura eval → antes={cov_raw:.1%} | después={cov_adj:.1%} (target={ALPHA:.0%})")

        # ── 5. Métricas ───────────────────────────────────────────────────
        r_lgb = det_metrics(o_v, lgb_v,   thr, label="LightGBM-AR")
        r_tft = det_metrics(o_v, p50_v,   thr, label="TFT-v2-P50")
        r_ens = det_metrics(o_v, ens_v,   thr, label="Ensemble")
        for r in [r_lgb, r_tft, r_ens]:
            r["target"] = target
            all_rows_det.append(r)

        b_raw = band_metrics(o_v, p10_v,   p90_v,   label="TFT-v2-raw")
        b_cal = band_metrics(o_v, p10_cal, p90_cal, label="TFT-v2-cal")
        for b in [b_raw, b_cal]:
            b["target"] = target
            all_rows_band.append(b)

        log.info(f"  LGB  → NSE={r_lgb['NSE']:.3f} KGE={r_lgb['KGE']:.3f} POD={r_lgb['POD']:.0%}")
        log.info(f"  TFT  → NSE={r_tft['NSE']:.3f} KGE={r_tft['KGE']:.3f} POD={r_tft['POD']:.0%}")
        log.info(f"  ENS  → NSE={r_ens['NSE']:.3f} KGE={r_ens['KGE']:.3f} POD={r_ens['POD']:.0%} J_alert={r_ens['J_alert']:.4f}")
        log.info(f"  Banda RAW → Coverage={b_raw['Coverage80']:.1%} POD_prob={b_raw['POD_prob']:.0%}")
        log.info(f"  Banda CAL → Coverage={b_cal['Coverage80']:.1%} POD_prob={b_cal['POD_prob']:.0%}")

        # Guardar predicciones
        for d, oo, lg, tf, en, b10, b90, b10c, b90c in zip(
                dte_v, o_v/Q_CONV/hmult, lgb_v/Q_CONV/hmult,
                p50_v/Q_CONV/hmult, ens_v/Q_CONV/hmult,
                p10_v/Q_CONV/hmult, p90_v/Q_CONV/hmult,
                p10_cal/Q_CONV/hmult, p90_cal/Q_CONV/hmult):
            all_preds_out.append({
                "date": d, "target": target, "q_obs": oo,
                "q_lgb": lg, "q_tft_p50": tf, "q_ens": en,
                "q_p10_raw": b10, "q_p90_raw": b90,
                "q_p10_cal": b10c, "q_p90_cal": b90c,
            })

        # ── 6. Figura TFT03: ensemble + banda (q_next_1d) ────────────────
        if target == "q_next_1d":
            p = pd.DataFrame({
                "date": dte_v,
                "obs":   o_v   / Q_CONV,
                "lgb":   lgb_v / Q_CONV,
                "tft":   p50_v / Q_CONV,
                "ens":   ens_v / Q_CONV,
                "p10r":  p10_v / Q_CONV,
                "p90r":  p90_v / Q_CONV,
                "p10c":  p10_cal / Q_CONV,
                "p90c":  p90_cal / Q_CONV,
            }).sort_values("date")

            fig, axes = plt.subplots(2, 1, figsize=(15, 10))

            # --- Panel 1: Ensemble + banda calibrada ---
            ax = axes[0]
            ax.fill_between(p["date"], p["p10c"], p["p90c"],
                            alpha=0.25, color="#27ae60",
                            label=f"Banda P10-P90 calibrada (cob.={cov_adj:.0%})")
            ax.fill_between(p["date"], p["p10r"], p["p90r"],
                            alpha=0.12, color="#3498db",
                            label=f"Banda P10-P90 raw (cob.={b_raw['Coverage80']:.0%})")
            ax.plot(p["date"], p["obs"],  color="#c0392b", lw=1.4, label="Q obs real")
            ax.plot(p["date"], p["ens"],  color="#8e44ad", lw=1.2, ls="-",
                    label=f"Ensemble P50  (NSE={r_ens['NSE']:.3f}  KGE={r_ens['KGE']:.3f})")
            ax.plot(p["date"], p["lgb"],  color="#2980b9", lw=0.9, ls="--", alpha=0.7,
                    label=f"LightGBM  (NSE={r_lgb['NSE']:.3f}  POD={r_lgb['POD']:.0%})")
            ax.axhline(Q90, ls=":", color="orange", alpha=0.8, label=f"Q90={Q90:.1f} m³/s")

            # Marcar alertas correctas del P90
            alert_p90 = p[p["p90c"] > Q90]
            alert_obs  = p[p["obs"]  > Q90]
            if len(alert_p90):
                ax.scatter(alert_p90["date"], [Q90+2]*len(alert_p90),
                           marker="v", s=20, color="#f39c12", zorder=5,
                           label=f"Alerta P90 ({len(alert_p90)} días, POD_prob={b_cal['POD_prob']:.0%})")

            ax.set_ylabel("Q (m³/s)"); ax.set_ylim(0)
            ax.set_title("Producto final — Ensemble LightGBM + TFT v2  |  Q +1 día  |  test 2024-2025",
                         fontweight="bold", fontsize=11)
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

            # --- Panel 2: comparativa LGB / TFT P50 / Ensemble ---
            ax2 = axes[1]
            ax2.plot(p["date"], p["obs"], color="#c0392b", lw=1.4, label="Q obs real")
            ax2.plot(p["date"], p["ens"], color="#8e44ad", lw=1.3,
                     label=f"Ensemble  NSE={r_ens['NSE']:.3f}  POD={r_ens['POD']:.0%}")
            ax2.plot(p["date"], p["lgb"], color="#2980b9", lw=1.0, ls="--",
                     label=f"LightGBM  NSE={r_lgb['NSE']:.3f}  POD={r_lgb['POD']:.0%}")
            ax2.plot(p["date"], p["tft"], color="#27ae60", lw=1.0, ls="-.",
                     label=f"TFT v2 P50  NSE={r_tft['NSE']:.3f}  POD={r_tft['POD']:.0%}")
            ax2.axhline(Q90, ls=":", color="orange", alpha=0.8)
            ax2.set_ylabel("Q (m³/s)"); ax2.set_ylim(0)
            ax2.set_title("Comparativa de pronósticos puntuales  (P50 de cada modelo)",
                          fontsize=10, fontweight="bold")
            ax2.legend(fontsize=8, loc="upper right")
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

            fig.tight_layout()
            fig.savefig(FIG_DIR / "TFT03_ensemble_final.png", bbox_inches="tight")
            plt.close(fig)
            log.info("  → TFT03_ensemble_final.png")

    # ── 7. Guardar productos ──────────────────────────────────────────────────
    pd.DataFrame(all_preds_out).to_csv(OUT_DIR / "ensemble_predictions.csv", index=False)
    log.info(f"Predicciones → ensemble_predictions.csv")

    # ── 8. Figura TFT04: leaderboard final de todos los modelos ──────────────
    # Cargar métricas históricas de los modelos anteriores
    leader_data = []
    # LGB (Scr65, train ≤2015, test 2021-2025)
    f65 = OUT_DIR / "lgbm_ar_results.csv"
    if f65.exists():
        df65 = pd.read_csv(f65)
        for _, row in df65.iterrows():
            leader_data.append({"model": "LGB-AR Scr65\n(test 2021-25)",
                                 "target": row["target"],
                                 "NSE": row["NSE"], "POD": row["POD"],
                                 "J_alert": row["J_alert"], "KGE": row["KGE"]})
    # TFT v1 (Scr72, test 2024-2025)
    f72 = OUT_DIR / "transfer_metrics_full.csv"
    if f72.exists():
        df72 = pd.read_csv(f72)
        tft1 = df72[df72["model"] == "TFT-transfer"]
        for _, row in tft1.iterrows():
            leader_data.append({"model": "TFT-v1 Scr72\n(test 2024-25)",
                                 "target": row["target"],
                                 "NSE": row["NSE"], "POD": row["POD"],
                                 "J_alert": row.get("J_alert", np.nan), "KGE": row["KGE"]})
    # TFT v2 + Ensemble (Scr74/75, test 2024-2025)
    for r in all_rows_det:
        label_map = {"LightGBM-AR": "LGB Scr75\n(test 2024-25)",
                     "TFT-v2-P50":  "TFT-v2 Scr74\n(test 2024-25)",
                     "Ensemble":    "Ensemble Scr75\n(test 2024-25)"}
        leader_data.append({
            "model":   label_map.get(r["model"], r["model"]),
            "target":  r["target"],
            "NSE":     r["NSE"], "POD": r["POD"],
            "J_alert": r["J_alert"], "KGE": r["KGE"],
        })

    ldf = pd.DataFrame(leader_data)
    metrics_plot = ["NSE", "KGE", "POD", "J_alert"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    for ax, tgt in zip(axes, ["q_next_1d", "q_sum_next_7d"]):
        sub = ldf[ldf["target"] == tgt].copy()
        models = sub["model"].tolist()
        x = np.arange(len(metrics_plot))
        w = 0.8 / len(models)

        for i, (_, row) in enumerate(sub.iterrows()):
            vals = [row.get(m, np.nan) for m in metrics_plot]
            bars = ax.bar(x + i*w - 0.4 + w/2, vals, w,
                          label=row["model"], color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels(metrics_plot, fontsize=10)
        ax.set_ylim(-0.1, 1.15)
        ax.axhline(0, color="gray", lw=0.5)
        tname = "Q +1 día" if tgt == "q_next_1d" else "Q suma 7 días"
        ax.set_title(tname, fontweight="bold", fontsize=11)
        ax.legend(fontsize=7, loc="upper right", ncol=1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylabel("Valor de la métrica")

    fig.suptitle("Leaderboard final — HidroAlerta Chancay-Huaral\n"
                 "Todos los modelos comparados en métricas clave",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "TFT04_model_leaderboard.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  → TFT04_model_leaderboard.png")

    # ── Resumen final ─────────────────────────────────────────────────────────
    log.info("\n=== RESUMEN DETERMINÍSTICO (test 2024-2025) ===")
    df_det = pd.DataFrame(all_rows_det)
    show_d = [c for c in ["model","target","NSE","NSE_sqrt","KGE","J_alert","POD","FAR","CSI","N"]
              if c in df_det.columns]
    log.info(df_det[show_d].to_string(index=False))

    log.info("\n=== RESUMEN PROBABILÍSTICO (banda P10-P90, test 2024-2025) ===")
    df_band = pd.DataFrame(all_rows_band)
    show_b = [c for c in ["model","target","Coverage80","Width_m3s","IS","POD_prob","FAR_prob"]
              if c in df_band.columns]
    log.info(df_band[show_b].to_string(index=False))

    log.info("\nSCRIPT 75 COMPLETADO — productos operativos generados")


if __name__ == "__main__":
    main()
