#!/usr/bin/env python3
"""
Script 88 — Leaderboard honesto a 1 día con IC bootstrap (fig03 del paper/PPT).

v2 (2026-07-06): consume las predicciones CURADAS del dashboard
(hidroalerta-dashboard/data/forecast_multimodelo.csv, generadas por 135 desde las
salidas corregidas del 125) para que la figura y el dashboard cuenten exactamente
los mismos números. Se retiran LSTM y GR4J del lineup (petición PPT): el lineup
es el publicado — Climatología · Persistencia · LightGBM · HydroST · RA-TFT.
Etiquetas en español (destino: PPT/paper).

Alineación: el CSV curado ya está indexado por fecha OBJETIVO con su obs.
IC: bootstrap 95 % (B=1000) sobre los días comunes con aforo real.

Run en .venv313:
    python scripts/06_eval/88_leaderboard_full.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("leaderboard")

Q90 = 40.89
DASH_CSV = Path("D:/ANA Concurso/hidroalerta-dashboard/data/forecast_multimodelo.csv")
TEST_START = pd.Timestamp("2024-01-01")
DAY = pd.Timedelta(days=1)


def obs_series():
    obs = pd.read_csv(ROOT / "data/silver/snirh/S1_snirh_daily_q.csv",
                      parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    return obs


def aligned_pred(path, date_col, pred_col, obs_col, obs_ref, target_filter=None):
    """Load a model CSV and return P50 indexed by TARGET date.

    Robust alignment: the file stores an obs column for each row; we pick the
    date shift s in {0,+1,-1} that makes the file's obs match SNIRH obs exactly
    (median abs error minimal). This removes all date-convention ambiguity.
    """
    df = pd.read_csv(OUTML / path, parse_dates=[date_col])
    if target_filter is not None and "target" in df.columns:
        df = df[df["target"] == target_filter]
    df = df.dropna(subset=[pred_col, obs_col])
    best = None
    for s in (0, 1, -1):
        tgt = df[date_col] + s * DAY
        ref = obs_ref.reindex(tgt).values
        fobs = df[obs_col].values
        m = np.isfinite(ref) & np.isfinite(fobs)
        if m.sum() < 20:
            continue
        err = np.median(np.abs(ref[m] - fobs[m]))
        if best is None or err < best[0]:
            best = (err, s)
    s = best[1] if best else 0
    log.info(f"  {path}: aligned with shift {s:+d}d (obs match err={best[0]:.3f})")
    tgt = df[date_col] + s * DAY
    return pd.Series(df[pred_col].values, index=tgt)


def regen_tft_tuned():
    """Regenerate TFT-tuned per-day P50 on the test set (target-date indexed)."""
    try:
        import torch
        spec = importlib.util.spec_from_file_location("tune81", ROOT / "scripts/05_models/81_tune_tft_optuna.py")
        S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
        T = S.T
        data = S.prepare_data()
        bp = pd.read_csv(OUTML / "81_best_params.csv", index_col=0)["0"].to_dict()
        cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]),
                   DROP=float(bp["DROP"]), LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]),
                   WD=float(bp["WD"]), ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
        _, model = S.run_config(cfg, data, seed=42); model.eval()
        Xseq, y, w, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
        with torch.no_grad():
            p50 = data["inv"](model(Xseq.to(S.DEVICE)).cpu().numpy()[:, 1]) / S.Q_CONV
        tgt = pd.DatetimeIndex(data["dates"][idx])  # target date convention of TFT pipeline
        return pd.Series(p50, index=tgt)
    except Exception as e:
        log.warning(f"TFT-tuned regeneration skipped: {e}")
        return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    nse = 1 - np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    so, sp = np.sqrt(np.clip(o,0,None)), np.sqrt(np.clip(p,0,None))
    nse_sq = 1 - np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
    oa, pa = o>=Q90, p>=Q90
    tp,fp,fn = np.sum(oa&pa), np.sum(~oa&pa), np.sum(oa&~pa)
    csi = tp/(tp+fp+fn) if (tp+fp+fn) else 0.0
    pod = tp/(tp+fn) if (tp+fn) else 0.0
    far = fp/(tp+fp) if (tp+fp) else 0.0
    j = 0.25*nse_sq + 0.25*nse + 0.30*csi + 0.10*pod - 0.10*far
    return dict(NSE=nse, NSE_sqrt=nse_sq, J_alert=j, CSI=csi, POD=pod, FAR=far)


def bootstrap_ci(o, p, fn_key, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    o, p = np.asarray(o), np.asarray(p)
    n = len(o); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(metrics(o[idx], p[idx])[fn_key])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    obs = obs_series()

    # Climatología estacional (media por día-del-año, serie pre-2024 de D7)
    d7 = pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv", parse_dates=["date"])
    qd = (d7[d7.entity_id=="sub_634"].set_index("date")["q_mm"]/(86.4/3062.62))
    clim_pre = qd[qd.index < TEST_START]
    doy_clim = clim_pre.groupby(clim_pre.index.dayofyear).mean()

    # ── P50 por modelo a lead=1 desde el CSV curado del dashboard (125→135) ──
    cur = pd.read_csv(DASH_CSV, parse_dates=["date"])
    cur = cur[(cur["lead"] == 1)]
    preds = {}
    for name in ["HydroST", "LightGBM", "RA-TFT"]:
        d = cur[cur["model"] == name].dropna(subset=["p50"])
        preds[name] = d.set_index("date")["p50"]

    test_dates = pd.date_range(TEST_START, obs.index.max(), freq="D")
    preds["Climatología"] = pd.Series(
        [doy_clim.get(d.dayofyear, np.nan) for d in test_dates], index=test_dates)
    preds["Persistencia"] = obs.shift(1)   # pronóstico para d = obs en d-1

    # ── Días comunes con aforo real ──────────────────────────────────────────
    obs = obs[~obs.index.duplicated(keep="last")]
    df = pd.DataFrame({"obs": obs})
    for k, sr in preds.items():
        sr = sr[~sr.index.duplicated(keep="last")]
        df[k] = sr.reindex(df.index)
    df = df[(df.index >= TEST_START)].dropna()
    log.info(f"Días comunes de prueba honesta: {len(df)}")
    log.info(f"Eventos de alerta (obs>Q90): {(df['obs']>Q90).sum()}")

    # ── Métricas + IC bootstrap ──────────────────────────────────────────────
    rows = []
    order = ["Climatología", "Persistencia", "LightGBM", "HydroST", "RA-TFT"]
    for name in [m for m in order if m in df.columns]:
        m = metrics(df["obs"], df[name])
        jlo, jhi = bootstrap_ci(df["obs"].values, df[name].values, "J_alert")
        nlo, nhi = bootstrap_ci(df["obs"].values, df[name].values, "NSE")
        rows.append(dict(model=name, **{k: round(v,3) for k,v in m.items()},
                         J_lo=round(jlo,3), J_hi=round(jhi,3),
                         NSE_lo=round(nlo,3), NSE_hi=round(nhi,3)))
    res = pd.DataFrame(rows)
    log.info("\n" + res[["model","NSE","NSE_sqrt","J_alert","J_lo","J_hi","CSI","POD","FAR"]].to_string(index=False))
    res.to_csv(OUTML/"88_leaderboard_full.csv", index=False)

    # ── Figura: J_alert y NSE con IC bootstrap 95 % (paleta del dashboard) ───
    COL = {"Climatología": "#9AA7B1", "Persistencia": "#B7C2CC",
           "LightGBM": "#8FA6B4", "HydroST": "#0A3D54", "RA-TFT": "#0B6E8C"}
    ES_BASE = {"Climatología", "Persistencia"}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for ax, key, lab in [(axes[0], "J_alert", "Puntaje compuesto de alerta  J_alert  (mayor = mejor)"),
                         (axes[1], "NSE", "Eficiencia de Nash–Sutcliffe  NSE  (mayor = mejor)")]:
        rs = res.sort_values(key)
        y = np.arange(len(rs))
        lo = rs[key] - rs[f"{'J' if key=='J_alert' else 'NSE'}_lo"]
        hi = rs[f"{'J' if key=='J_alert' else 'NSE'}_hi"] - rs[key]
        ax.barh(y, rs[key], color=[COL[m] for m in rs["model"]], alpha=0.92,
                xerr=[lo, hi], capsize=3,
                error_kw=dict(ecolor="#33414C", lw=1.1))
        ax.set_yticks(y)
        ax.set_yticklabels([m + (" (línea base)" if m in ES_BASE else "") for m in rs["model"]])
        for yi, (v, hv, mname) in enumerate(zip(rs[key], hi, rs["model"])):
            if mname == "RA-TFT":
                ax.get_yticklabels()[yi].set_fontweight("bold")
            ax.text(v + hv + 0.012, yi, f"{v:.2f}", va="center", fontsize=9,
                    color="#33414C")
        ax.set_xlabel(lab); ax.grid(alpha=0.3, axis="x")
    axes[0].set_title("(a) Desempeño de alerta (IC bootstrap 95 %)")
    axes[1].set_title("(b) Habilidad continua (IC bootstrap 95 %)")

    from matplotlib.patches import Patch
    handles = [Patch(color="#B7C2CC", label="Líneas base ingenuas"),
               Patch(color="#8FA6B4", label="ML de referencia (LightGBM)"),
               Patch(color="#0A3D54", label="HydroST (alerta a 1–2 días)"),
               Patch(color="#0B6E8C", label="RA-TFT (propuesto, multi-día)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    n_ev = int((df["obs"] > Q90).sum())
    fig.suptitle(f"Comparativa honesta a 1 día — prueba 2024–2025 (N={len(df)} días con aforo real, "
                 f"{n_ev} eventos de alerta)\n"
                 f"A 1 día la persistencia fija un techo duro (autocorrelación 0,98); "
                 f"la ventaja del modelo propuesto aparece a multi-día (ver figura de horizontes)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig03_leaderboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Guardado: {FIGDIR/'fig03_leaderboard.png'} + 88_leaderboard_full.csv")


if __name__ == "__main__":
    main()
