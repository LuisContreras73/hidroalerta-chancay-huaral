#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 147 — Verificación PROBABILÍSTICA de alerta (roster Q1 #1). CERO entrenamiento.

Hipótesis a poner a prueba: "no hay alerta multi-día (POD=0 a h≥3)" se afirmó
BINARIZANDO con el p50. Pero un pronóstico probabilístico puede tener habilidad de
EXCEDENCIA que el POD binario esconde. Aquí se evalúa P(Q>umbral) desde los cuantiles
con métricas estándar de verificación (Jolliffe & Stephenson; Hersbach 2000; WMO):
  - Brier Score (BS) + Brier Skill Score (BSS) vs climatología (tasa base).
  - Diagrama de confiabilidad (calibración) + resolución.
  - Curva ROC + AUC (discriminación) — independiente de calibración.

P(Q>T) desde (p10,p50,p90): normal en dos piezas sobre log(Q) (mismo método que la
matriz de excedencia del dashboard): σ_inf=(logp50−logp10)/1.2816, σ_sup análoga;
z=(logT−logp50)/σ; P=1−Φ(z). Reusa forecast_multimodelo.csv (gen4+baselines) +
137_*_preds (fundacionales). Umbral principal Q90=40.89 (también Q80).

Salidas: outputs/ml_Q/147_prob_alert.csv (BS/BSS/AUC por modelo×lead×umbral)
         reports/diagnostico/FIG15_prob_alert_verification.png

Run en .venv313:
    python scripts/06_eval/147_prob_alert_verification.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "reports/diagnostico"
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prob147")

Q90 = 40.89
Q80 = 28.0                       # ~p80 histórico (memoria: media 17,1, p90 40,9)
Z90 = 1.2815515594              # Phi^-1(0.9)
LEADS = [1, 3, 7, 14]
COL = {"RA-TFT": "#0B6E8C", "HydroST": "#0A3D54", "LightGBM": "#8FA6B4",
       "Persistencia": "#B7C2CC", "Chronos-2": "#7B2D8E", "TimesFM-2.5": "#B8860B"}


def p_exceed(p10, p50, p90, thr):
    """P(Q>thr) con normal de dos piezas sobre log(Q). Robusta a banda degenerada."""
    p10 = np.maximum(p10, 1e-3); p50 = np.maximum(p50, 1e-3); p90 = np.maximum(p90, 1e-3)
    lo = np.log(p50) - np.log(p10); hi = np.log(p90) - np.log(p50)
    s_lo = np.maximum(lo, 1e-3) / Z90; s_hi = np.maximum(hi, 1e-3) / Z90
    lt = np.log(np.maximum(thr, 1e-3))
    s = np.where(lt >= np.log(p50), s_hi, s_lo)
    z = (lt - np.log(p50)) / np.maximum(s, 1e-6)
    return 1 - norm.cdf(z)


def cargar():
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    fc = fc[fc.model.isin(["RA-TFT", "HydroST", "LightGBM", "Persistencia"])]
    ch = pd.read_csv(OUT / "137_chronos2_preds.csv", parse_dates=["date"]).assign(model="Chronos-2")
    tf = pd.read_csv(OUT / "137_timesfm25_preds.csv", parse_dates=["date"]).assign(model="TimesFM-2.5")
    d = pd.concat([fc[["date", "model", "lead", "obs", "p10", "p50", "p90"]], ch, tf],
                  ignore_index=True).dropna(subset=["obs", "p10", "p50", "p90"])
    return d


def brier_bss(pf, o, base):
    bs = np.mean((pf - o) ** 2)
    bs_clim = np.mean((base - o) ** 2)
    bss = 1 - bs / bs_clim if bs_clim > 0 else np.nan
    return bs, bss


def auc(pf, o):
    """AUC por rangos (Mann-Whitney), robusto a clases desbalanceadas."""
    pos, neg = pf[o == 1], pf[o == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def reliability(pf, o, bins=6):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pf >= a) & (pf < b) if b < 1 else (pf >= a) & (pf <= b)
        if m.sum() >= 5:
            xs.append(pf[m].mean()); ys.append(o[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    d = cargar()
    modelos = ["Persistencia", "LightGBM", "HydroST", "RA-TFT", "Chronos-2", "TimesFM-2.5"]
    filas = []
    for thr, tname in [(Q90, "Q90"), (Q80, "Q80")]:
        for L in LEADS:
            for m in modelos:
                s = d[(d.model == m) & (d.lead == L)]
                if len(s) < 30:
                    continue
                o = (s.obs.values >= thr).astype(float)
                base = o.mean()
                pf = p_exceed(s.p10.values, s.p50.values, s.p90.values, thr)
                bs, bss = brier_bss(pf, o, base)
                filas.append(dict(umbral=tname, lead=L, model=m, N=len(s),
                                  eventos=int(o.sum()), tasa_base=round(base, 3),
                                  BS=round(bs, 4), BSS=round(bss, 3),
                                  AUC=round(auc(pf, o), 3)))
    res = pd.DataFrame(filas)
    res.to_csv(OUT / "147_prob_alert.csv", index=False)
    log.info("\n" + res[res.umbral == "Q90"].to_string(index=False))

    # ── FIG15 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (a) BSS vs lead (Q90) — ¿hay habilidad probabilística?
    ax = axes[0]
    r90 = res[res.umbral == "Q90"]
    for m in modelos:
        sm = r90[r90.model == m].sort_values("lead")
        ax.plot(sm.lead, sm.BSS, "-o", color=COL[m], lw=2.2 if m == "RA-TFT" else 1.5,
                ms=6 if m == "RA-TFT" else 4, label=m)
    ax.axhline(0, color="#C0392B", ls=":", lw=1.3)
    ax.text(13.8, 0.02, "sin habilidad (=climatología)", ha="right", fontsize=8.5, color="#C0392B")
    ax.set_xticks(LEADS); ax.set_xlabel("horizonte (días)")
    ax.set_ylabel("Brier Skill Score (Q90)")
    ax.set_title("(a) ¿Hay habilidad probabilística de alerta?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) confiabilidad a h7
    ax = axes[1]
    ax.plot([0, 1], [0, 1], ":", color="#33414C", lw=1.2)
    for m in ["RA-TFT", "Chronos-2", "LightGBM"]:
        s = d[(d.model == m) & (d.lead == 7)]
        o = (s.obs.values >= Q90).astype(float)
        pf = p_exceed(s.p10.values, s.p50.values, s.p90.values, Q90)
        xs, ys, _ = reliability(pf, o)
        if len(xs):
            ax.plot(xs, ys, "-o", color=COL[m], lw=2, ms=5, label=m)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("prob. pronosticada P(Q>Q90)"); ax.set_ylabel("frecuencia observada")
    ax.set_title("(b) Confiabilidad a 7 días (diagonal = perfecta)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    # (c) AUC vs lead (discriminación)
    ax = axes[2]
    for m in modelos:
        sm = r90[r90.model == m].sort_values("lead")
        ax.plot(sm.lead, sm.AUC, "-o", color=COL[m], lw=2.2 if m == "RA-TFT" else 1.5,
                ms=6 if m == "RA-TFT" else 4, label=m)
    ax.axhline(0.5, color="#C0392B", ls=":", lw=1.3)
    ax.text(13.8, 0.51, "azar", ha="right", fontsize=9, color="#C0392B")
    ax.set_xticks(LEADS); ax.set_xlabel("horizonte (días)")
    ax.set_ylabel("AUC (discriminación, Q90)")
    ax.set_title("(c) ¿Distingue días de crecida de los demás?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("FIG15 · Verificación PROBABILÍSTICA de alerta — la pregunta que el POD binario "
                 "esconde\n(P(Q>Q90) desde cuantiles; BSS>0 = hay habilidad; AUC>0.5 = discrimina)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG15_prob_alert_verification.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("FIG15 + 147_prob_alert.csv listos")


if __name__ == "__main__":
    main()
