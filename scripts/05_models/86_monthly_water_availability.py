#!/usr/bin/env python3
"""
Script 86 — Monthly water-availability forecasting (ANA management product).

Complements the daily flood-alert system with a MONTHLY streamflow forecast,
the timescale ANA needs for water-resource management (irrigation scheduling,
reservoir operation, drought planning).

Scientific framing (completes the multi-horizon narrative):
  At DAILY scale the forecast is persistence-dominated (Scripts 82-83). At
  MONTHLY scale the predictable signal shifts to catchment MEMORY (storage,
  recession via deep soil moisture swvl4 and antecedent flow) and SEASONALITY,
  with ENSO as a weak negative modulator (El Niño → central-Andes drought).
  This repurposes the very signals found redundant at 1-day lead.

Targets:
  - next-month mean flow Q[m+1]  (m³/s)
Products for management:
  - P10/P50/P90 monthly forecast bands
  - tercile category (below / normal / above normal) vs climatology
Baselines: climatology (month-of-year mean) and persistence — skill is reported
RELATIVE to climatology (the relevant seasonal-forecast benchmark).

Honest eval: test months scored only where >=20 days of REAL obs exist.
Run in .venv:
    python scripts/05_models/86_monthly_water_availability.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/qa_satellite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("monthly")

Q_CONV = 86.4 / 3062.62
ENTITIES = ["sub_634","sub_640","sub_641","sub_646","sub_649","sub_650","sub_653","sub_655","sub_656"]

TRAIN_END = "2014-12-31"
VAL_END   = "2020-12-31"   # test = 2021-01 onward (where real obs exist)
WET = [12, 1, 2, 3, 4]     # wet-season target months (Dec-Apr)


def build_monthly():
    d7 = pd.read_csv(ROOT / "data/model_ready/D7_multientity.csv", parse_dates=["date"])
    outlet = d7[d7["entity_id"] == "sub_634"].set_index("date").sort_index()

    # Basin-mean precip across entities (spatially distributed signal)
    pr_basin = d7.groupby("date")["pr_mm"].mean()

    # Soil moisture (storage) — basin mean
    swvl = pd.read_parquet(OUT_DIR / "era5_swvl_per_entity.parquet")
    if "date" not in swvl.columns:
        swvl = swvl.reset_index()
    swvl_basin = swvl.groupby("date")[["swvl1", "swvl4"]].mean()

    # Daily series -> monthly
    qd = outlet["q_mm"] / Q_CONV            # GR4J 1981-2020 + obs 2021+
    m = pd.DataFrame({
        "q":      qd.resample("MS").mean(),
        "pr":     pr_basin.resample("MS").sum(),
        "oni":    outlet["oni_index"].resample("MS").mean(),
        "spi90":  outlet["spi_90d"].resample("MS").last(),
        "swvl1":  swvl_basin["swvl1"].resample("MS").mean(),
        "swvl4":  swvl_basin["swvl4"].resample("MS").mean(),
    })

    # Real observed monthly Q (>=20 valid days required)
    obs = pd.read_csv(ROOT / "data/silver/snirh/S1_snirh_daily_q.csv",
                      parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    obs_m   = obs.resample("MS").mean()
    obs_cnt = obs.resample("MS").count()
    m["q_obs"] = obs_m.where(obs_cnt >= 20)

    # Coastal El Niño index (Niño 1+2 / ICEN proxy) — distinct from global ONI (Script 87)
    enso = pd.read_csv(ROOT / "data/silver/enso/S3b_enso_coastal_global.csv",
                       parse_dates=["date"]).set_index("date")
    m["coastal"] = enso["coastal"].reindex(m.index)
    m["cg_contrast"] = enso["coastal"].reindex(m.index) - enso["global"].reindex(m.index)

    # ── Features (all known at issue time t, predict t+1) ─────────────────────
    m["q_l1"], m["q_l2"], m["q_l3"] = m["q"].shift(0), m["q"].shift(1), m["q"].shift(2)
    m["pr_l1"]   = m["pr"].shift(0)
    m["pr_2m"]   = m["pr"].rolling(2).sum()
    m["swvl4_l1"]= m["swvl4"].shift(0)
    m["swvl1_l1"]= m["swvl1"].shift(0)
    m["oni_l1"]  = m["oni"].shift(0)
    m["spi90_l1"]= m["spi90"].shift(0)
    m["coastal_l1"]  = m["coastal"].shift(0)
    m["cg_contrast_l1"] = m["cg_contrast"].shift(0)
    # target month seasonality (month of t+1)
    tgt_month = (m.index.month % 12) + 1
    m["sin_m"] = np.sin(2*np.pi*tgt_month/12)
    m["cos_m"] = np.cos(2*np.pi*tgt_month/12)

    # Target: next-month mean flow (from the GR4J+obs continuous series for training,
    # but we ALSO carry next-month real obs for honest testing)
    m["y"]      = m["q"].shift(-1)
    m["y_obs"]  = m["q_obs"].shift(-1)
    m["y_month"]= tgt_month
    return m.dropna(subset=["q_l1", "q_l2", "q_l3", "pr_2m"])


FEATURES = ["q_l1","q_l2","q_l3","pr_l1","pr_2m","swvl4_l1","swvl1_l1","oni_l1","spi90_l1","sin_m","cos_m"]


def nse(o, p):
    o, p = np.asarray(o), np.asarray(p)
    return 1 - np.sum((o-p)**2)/np.sum((o-o.mean())**2)

def kge(o, p):
    o, p = np.asarray(o), np.asarray(p)
    r = np.corrcoef(o, p)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (p.std()/o.std()-1)**2 + (p.mean()/o.mean()-1)**2)


def main():
    m = build_monthly()
    log.info(f"Monthly frame: {len(m)} months {m.index.min().date()}..{m.index.max().date()}")

    tr = m[m.index <= TRAIN_END]
    va = m[(m.index > TRAIN_END) & (m.index <= VAL_END)]
    te = m[m.index > VAL_END]
    log.info(f"Train={len(tr)} Val={len(va)} Test={len(te)} months")

    # ── Climatology baseline (month-of-year mean from TRAIN) ─────────────────
    clim = tr.groupby("y_month")["y"].mean()
    te_clim = te["y_month"].map(clim).values

    # ── Persistence baseline (next month = this month) ───────────────────────
    te_pers = te["q_l1"].values

    # ── ANOMALY modelling (correct for seasonal forecasting) ──────────────────
    # Climatology beats raw ML because monthly Q is seasonality-dominated. The
    # skill question is whether we can predict the DEPARTURE from normal. So we
    # model anomalies: y_anom = y - clim[month]; final = clim + predicted anomaly.
    clim_issue = tr.groupby(tr.index.month)["q"].mean()   # clim of the issue-month flow
    def anomalize(frame):
        a = frame.copy()
        a["q_l1"] = a["q_l1"] - a.index.month.map(clim_issue)
        a["q_l2"] = a["q_l2"] - ((a.index.month-1-1) % 12 + 1).map(clim_issue)
        a["q_l3"] = a["q_l3"] - ((a.index.month-2-1) % 12 + 1).map(clim_issue)
        return a
    tr_a, te_a = anomalize(tr), anomalize(te)
    BASE_FEATURES    = ["q_l1","q_l2","q_l3","pr_l1","pr_2m","swvl4_l1","swvl1_l1","oni_l1","spi90_l1"]
    COASTAL_FEATURES = BASE_FEATURES + ["coastal_l1","cg_contrast_l1"]

    y_anom = (tr["y"] - tr["y_month"].map(clim)).values

    # Improvement #2: up-weight wet-season training months to fix peak underprediction
    W_WET = 3.0
    sw = np.where(tr["y_month"].isin(WET).values, W_WET, 1.0)

    def fit_quantiles(feats, sample_weight=None):
        out = {}
        imp = None
        for q, name in [(0.1,"p10"),(0.5,"p50"),(0.9,"p90")]:
            params = dict(objective="quantile", alpha=q, n_estimators=300, learning_rate=0.03,
                          num_leaves=12, min_child_samples=20, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=2.0, verbose=-1)
            mdl = lgb.LGBMRegressor(**params)
            mdl.fit(tr_a[feats], y_anom, sample_weight=sample_weight)
            out[name] = te_clim + mdl.predict(te_a[feats])
            if name == "p50":
                imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
        out["p10"] = np.minimum(out["p10"], out["p50"])
        out["p90"] = np.maximum(out["p90"], out["p50"])
        return out, imp

    # A/B: base (global ONI only) vs +coastal (Niño 1+2).
    # On this short test the coastal index does not improve aggregate skill
    # (its value is episodic/extreme-event flagging, see Script 87 §9d), so the
    # BASE model is the operational monthly product.
    # Wet-season weighting was tested but DID NOT help (worse NSE & wet-MAE on this
    # short test — storm timing is not monthly-predictable, so up-weighting overfits).
    # Headline = unweighted base model. Kept the wet variant for the honest A/B.
    preds_coastal, _   = fit_quantiles(COASTAL_FEATURES)
    preds_wet, _       = fit_quantiles(BASE_FEATURES, sample_weight=sw)     # rejected variant
    preds, imp         = fit_quantiles(BASE_FEATURES)                       # operational (unweighted)
    preds_base         = preds

    # ── Honest eval on real-obs test months ──────────────────────────────────
    mask = te["y_obs"].notna().values
    obs = te["y_obs"].values[mask]
    log.info(f"\nTest months with real next-month obs: {mask.sum()}")

    rows = []
    for name, pred in [("Climatology", te_clim), ("Persistence", te_pers),
                       ("LGBM operational", preds["p50"]),
                       ("LGBM +coastal", preds_coastal["p50"]),
                       ("LGBM +wet-weight", preds_wet["p50"])]:
        p = np.asarray(pred)[mask]
        mse = np.mean((obs-p)**2)
        # wet-season MAE (the management-relevant high-flow months)
        wet_te = te["y_month"].isin(WET).values[mask]
        wet_mae = np.mean(np.abs(obs[wet_te]-p[wet_te])) if wet_te.any() else np.nan
        rows.append(dict(model=name, NSE=nse(obs,p), KGE=kge(obs,p),
                         MAE=np.mean(np.abs(obs-p)), RMSE=np.sqrt(mse), wet_MAE=wet_mae))
    res = pd.DataFrame(rows)
    # Skill vs climatology (MSESS)
    mse_clim = np.mean((obs - np.asarray(te_clim)[mask])**2)
    res["skill_vs_clim"] = res.apply(
        lambda r: 1 - (r["RMSE"]**2)/mse_clim, axis=1)
    log.info("\n" + "="*60)
    log.info("MONTHLY FORECAST SKILL (test, real obs)")
    log.info("="*60)
    log.info("\n" + res.round(3).to_string(index=False))

    # ── Tercile category skill (below/normal/above normal) ───────────────────
    # Terciles per target month from train climatology distribution
    terc = {}
    for mo in range(1, 13):
        vals = tr[tr["y_month"] == mo]["y"]
        terc[mo] = (vals.quantile(1/3), vals.quantile(2/3))
    def categorize(val, mo):
        lo, hi = terc[mo]
        return 0 if val < lo else (2 if val > hi else 1)
    te_months = te["y_month"].values[mask]
    obs_cat  = np.array([categorize(o, mo) for o, mo in zip(obs, te_months)])
    pred_cat = np.array([categorize(p, mo) for p, mo in zip(np.asarray(preds["p50"])[mask], te_months)])
    clim_cat_acc = 1/3  # random/climatology baseline
    hit = np.mean(obs_cat == pred_cat)
    log.info(f"\nTercile category hit rate: {hit:.3f} (climatology/random = {clim_cat_acc:.3f})")
    log.info(f"Feature importance (P50): {imp.head(6).to_dict()}")

    # ── Improvement #1: seasonal conformal band calibration (leave-one-year-out)
    # Raw quantile bands under-cover (PICP~0.47). We inflate them per season using
    # split-conformal, validated leave-one-year-out within the real-obs test set so
    # each year's radius comes only from OTHER years (no peeking).
    ALPHA = 0.20
    p10m, p50m, p90m = (np.asarray(preds[k])[mask] for k in ("p10","p50","p90"))
    te_idx = te.index[mask]
    yrs    = (te_idx + pd.offsets.MonthBegin(1)).year
    seas   = np.where(pd.Index(te_idx + pd.offsets.MonthBegin(1)).month.isin(WET), "wet", "dry")
    lo_cal, hi_cal = p10m.copy(), p90m.copy()
    for i in range(len(obs)):
        # calibration scores from OTHER years, same season
        sel = (yrs != yrs[i]) & (seas == seas[i])
        if sel.sum() < 3:
            sel = (yrs != yrs[i])
        scores = np.maximum(p10m[sel] - obs[sel], obs[sel] - p90m[sel])
        n = sel.sum()
        q = np.quantile(scores, min(1.0, (1-ALPHA)*(1 + 1/n)))
        lo_cal[i] = p10m[i] - q
        hi_cal[i] = p90m[i] + q
    picp_raw = float(((obs>=p10m)&(obs<=p90m)).mean())
    picp_cal = float(((obs>=lo_cal)&(obs<=hi_cal)).mean())
    w_raw = float((p90m-p10m).mean()); w_cal = float((hi_cal-lo_cal).mean())
    log.info(f"\nBand calibration (target {1-ALPHA:.0%}): "
             f"raw PICP={picp_raw:.2f} (w={w_raw:.1f}) -> conformal PICP={picp_cal:.2f} (w={w_cal:.1f})")

    # ── Continuous seasonal-conformal bands (for ALL test months, incl. gaps) ──
    # Visualisation: the model issues a forecast every month even when the gauge
    # is down; we apply the seasonal conformal radius to every month so the band
    # is continuous. Metrics above remain on real-obs months only (honest).
    q_seas = {}
    for s in ("wet","dry"):
        sc = np.maximum(p10m[seas==s]-obs[seas==s], obs[seas==s]-p90m[seas==s])
        q_seas[s] = (np.quantile(sc, min(1.0,(1-ALPHA)*(1+1/len(sc)))) if len(sc) >= 3 else np.nan)
    q_def = np.nanmean([v for v in q_seas.values() if np.isfinite(v)])
    all_seas = np.where(pd.Index(te.index + pd.offsets.MonthBegin(1)).month.isin(WET), "wet", "dry")
    radius = np.array([q_seas.get(s) if np.isfinite(q_seas.get(s, np.nan)) else q_def for s in all_seas])

    # ── Save ─────────────────────────────────────────────────────────────────
    out = te.copy()
    out["pred_p10"], out["pred_p50"], out["pred_p90"] = preds["p10"], preds["p50"], preds["p90"]
    out["pred_p10_cal"] = preds["p10"] - radius   # continuous (all months)
    out["pred_p90_cal"] = preds["p90"] + radius
    out["clim"], out["persist"] = te_clim, te_pers
    out["has_obs"] = out["y_obs"].notna()
    out[["q_obs","y_obs","has_obs","y_month","pred_p10","pred_p50","pred_p90",
         "pred_p10_cal","pred_p90_cal","clim","persist"]].to_csv(
        OUT_DIR / "86_monthly_forecast.csv")
    res.to_csv(OUT_DIR / "86_monthly_skill.csv", index=False)
    log.info("\nSaved: 86_monthly_forecast.csv, 86_monthly_skill.csv")

    # ── Figure: CONTINUOUS monthly forecast; obs markers only where real ──────
    o = out.sort_index()
    tgt = o.index + pd.offsets.MonthBegin(1)              # target month
    obs_pts = o[o["has_obs"]]; obs_tgt = obs_pts.index + pd.offsets.MonthBegin(1)
    fig, ax = plt.subplots(figsize=(15, 6))

    # shade gap stretches (no gauge observation) so the reader sees them
    gap = ~o["has_obs"].values
    in_gap = False
    for i, d in enumerate(tgt):
        if gap[i] and not in_gap:
            g0 = d; in_gap = True
        if (not gap[i] or i == len(tgt)-1) and in_gap:
            ax.axvspan(g0, tgt[i], color="#f2f2f2", alpha=0.9, zorder=0)
            in_gap = False
    ax.axvspan(np.nan, np.nan, color="#f2f2f2", label="gauge gap (no observation)")

    ax.fill_between(tgt, o["pred_p10_cal"], o["pred_p90_cal"], color="#9ecae1", alpha=0.5,
                    label="P10–P90 (seasonal-conformal)", zorder=1)
    ax.plot(tgt, o["pred_p50"], "-", color="#08519c", lw=1.6, label="Forecast P50 (anomaly model)", zorder=3)
    ax.plot(tgt, o["clim"], "--", color="gray", lw=1.1, label="Climatology", zorder=2)
    ax.plot(obs_tgt, obs_pts["y_obs"], "s", color="black", ms=5, label="Observation (real)", zorder=4)
    ax.set_ylabel("Monthly mean streamflow [m³/s]")
    ax.set_title("Monthly water-availability forecast (1-month lead), test 2021-2025\n"
                 "continuous forecast incl. ungauged 2025 months; observations shown only where measured")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "15_monthly_forecast.png", dpi=300); plt.close(fig)
    log.info(f"Saved: 15_monthly_forecast.png (continuous; {int(gap.sum())} ungauged months shown)")

    # ── Figure: skill bars + feature importance ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    bar_colors = ["#999","#bbb","#7fb3d5","#f0a868","#2ca02c"][:len(res)]
    axes[0].bar(res["model"], res["NSE"], color=bar_colors, alpha=0.85)
    axes[0].set_ylabel("NSE (vs real obs)"); axes[0].set_title("Monthly forecast skill by model")
    axes[0].grid(alpha=0.3, axis="y"); axes[0].tick_params(axis="x", labelrotation=30, labelsize=8)
    for i,v in enumerate(res["NSE"]): axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    imp.sort_values().plot.barh(ax=axes[1], color="#08519c", alpha=0.85)
    axes[1].set_xlabel("LightGBM importance"); axes[1].set_title("Monthly model — feature importance")
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / "16_monthly_skill.png", dpi=150); plt.close(fig)
    log.info("Saved: 16_monthly_skill.png")
    log.info("Done.")


if __name__ == "__main__":
    main()
