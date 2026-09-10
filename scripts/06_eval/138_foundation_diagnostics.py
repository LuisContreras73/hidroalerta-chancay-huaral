#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 138 — Diagnóstico de los modelos fundacionales (gen6): ¿dónde se rompen?

Consume las predicciones por día del 137 (Chronos-2, TimesFM 2.5) + RA-TFT (gen4,
forecast_multimodelo) + aforo real, y produce 4 figuras de diagnóstico en
reports/diagnostico (FIG6–FIG9) + estadísticas en consola:

  FIG6  Hidrogramas de la temporada húmeda 2024 (leads 1 y 7): ¿siguen los eventos?
  FIG7  Ruptura por magnitud: MAE por clase de caudal (bajo/medio/alto) y análisis
        de los 17 días de alerta (obs>Q90) — subpredicción en picos.
  FIG8  Banda q10–q90: cobertura empírica y ancho medio por lead vs RA-TFT.
  FIG9  Ruptura por dinámica: error en SUBIDAS (dQ>0, lluvia) vs RECESIONES, y
        2024 (contexto obs) vs 2025 (contexto reconstruido + gaps).

Run en .venv313:
    python scripts/06_eval/138_foundation_diagnostics.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUTML = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "reports/diagnostico"
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fm138")

Q90 = 40.89
COL = {"Chronos-2": "#7B2D8E", "TimesFM-2.5": "#B8860B",
       "RA-TFT": "#0B6E8C", "obs": "#0C1E2A"}


def cargar():
    ch = pd.read_csv(OUTML / "137_chronos2_preds.csv", parse_dates=["date"])
    tf = pd.read_csv(OUTML / "137_timesfm25_preds.csv", parse_dates=["date"])
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    ra = fc[fc.model == "RA-TFT"].copy()
    ch["model"], tf["model"] = "Chronos-2", "TimesFM-2.5"
    ra = ra[["date", "model", "lead", "obs", "p10", "p50", "p90"]]
    d = pd.concat([ch, tf, ra], ignore_index=True).dropna(subset=["obs", "p50"])
    return d


def fig6_hidrogramas(d):
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8), sharex=True)
    win = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31"))
    for ax, L in zip(axes, [1, 7]):
        sub = d[(d.lead == L) & d.date.between(*win)]
        o = sub[sub.model == "Chronos-2"].set_index("date")["obs"]
        ax.plot(o.index, o.values, color=COL["obs"], lw=2.2, label="Aforo observado")
        for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT"]:
            s = sub[sub.model == m].set_index("date")
            if m == "Chronos-2":
                ax.fill_between(s.index, s["p10"], s["p90"], color=COL[m], alpha=0.15,
                                label="Chronos-2 banda q10–q90")
            ax.plot(s.index, s["p50"], color=COL[m], lw=1.6, label=f"{m} (p50)")
        ax.axhline(Q90, color="#C0392B", ls=":", lw=1.2)
        ax.text(win[1], Q90 + 1.5, "umbral de alerta Q90", ha="right", fontsize=9,
                color="#C0392B")
        ax.set_title(f"a {L} día{'s' if L > 1 else ''} de anticipación")
        ax.set_ylabel("caudal (m³/s)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, ncol=2, framealpha=0.9)
    fig.suptitle("FIG6 · Fundacionales en la temporada húmeda 2024 — siguen el régimen, "
                 "llegan tarde a los picos\n(zero-shot univariado: sin lluvia, solo caudal pasado)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG6_fundacionales_hidrogramas_2024.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig); log.info("FIG6 OK")


def fig7_magnitud(d):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    # (a) MAE por clase de caudal a lead 7
    ax = axes[0]
    clases = [("bajo (<p50: 11 m³/s)", 0, 11.3), ("medio", 11.3, Q90),
              (f"alto (>Q90: {Q90:.0f})", Q90, 9e9)]
    w = 0.26
    for i, m in enumerate(["Chronos-2", "TimesFM-2.5", "RA-TFT"]):
        s = d[(d.model == m) & (d.lead == 7)]
        maes = [np.abs(s[(s.obs >= a) & (s.obs < b)].obs
                       - s[(s.obs >= a) & (s.obs < b)].p50).mean() for _, a, b in clases]
        ax.bar(np.arange(3) + (i - 1) * w, maes, w, color=COL[m], label=m, alpha=0.92)
        for x, v in zip(np.arange(3) + (i - 1) * w, maes):
            ax.text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in clases], fontsize=9)
    ax.set_ylabel("MAE (m³/s)"); ax.set_title("(a) Error por clase de caudal — 7 días")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    # (b) los 17 días de alerta: predicho vs observado por lead (Chronos-2)
    ax = axes[1]
    ev = d[(d.obs >= Q90)]
    for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT"]:
        ratio = [100 * (ev[(ev.model == m) & (ev.lead == L)].p50
                        / ev[(ev.model == m) & (ev.lead == L)].obs).mean()
                 for L in [1, 3, 7, 14]]
        ax.plot([1, 3, 7, 14], ratio, "-o", color=COL[m], lw=2, label=m)
    ax.axhline(100, color="#33414C", ls=":", lw=1.2)
    ax.text(13.8, 101.5, "predicción perfecta", ha="right", fontsize=9, color="#33414C")
    ax.set_xticks([1, 3, 7, 14]); ax.set_ylim(0, 115)
    ax.set_xlabel("anticipación (días)")
    ax.set_ylabel("p50 / observado en días de alerta (%)")
    ax.set_title("(b) Cuánto del pico ven — días con obs > Q90 (N=17)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("FIG7 · La ruptura es en los EXTREMOS: a más anticipación, el pico "
                 "predicho se desvanece hacia el régimen medio",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG7_fundacionales_ruptura_extremos.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig); log.info("FIG7 OK")


def fig8_banda(d):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    leads = [1, 3, 7, 14]
    for ax, (fn, tit, ylab, ref) in zip(axes, [
            (lambda s: 100 * ((s.obs >= s.p10) & (s.obs <= s.p90)).mean(),
             "(a) Cobertura empírica de la banda q10–q90", "% de días dentro", 80),
            (lambda s: (s.p90 - s.p10).mean(),
             "(b) Ancho medio de la banda", "ancho (m³/s)", None)]):
        for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT"]:
            vals = [fn(d[(d.model == m) & (d.lead == L)]) for L in leads]
            ax.plot(leads, vals, "-o", color=COL[m], lw=2, label=m)
        if ref:
            ax.axhline(ref, color="#33414C", ls=":", lw=1.2)
            ax.text(13.8, ref + 0.8, "nominal 80 %", ha="right", fontsize=9)
        ax.set_xticks(leads); ax.set_xlabel("anticipación (días)")
        ax.set_ylabel(ylab); ax.set_title(tit); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("FIG8 · Honestidad de la incertidumbre: cobertura y ancho de banda",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG8_fundacionales_banda.png", dpi=200, bbox_inches="tight")
    plt.close(fig); log.info("FIG8 OK")


def fig9_dinamica(d):
    # dQ del día objetivo (subida = evento de lluvia en curso)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    obs1 = d[(d.model == "Chronos-2") & (d.lead == 1)].set_index("date")["obs"]
    dq = obs1.diff()
    ax = axes[0]
    leads = [1, 3, 7, 14]
    for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT"]:
        up, dn = [], []
        for L in leads:
            s = d[(d.model == m) & (d.lead == L)].set_index("date")
            rising = dq.reindex(s.index) > 1.0            # subida franca (>1 m³/s/d)
            err = np.abs(s.obs - s.p50)
            up.append(err[rising].mean()); dn.append(err[~rising].mean())
        ax.plot(leads, up, "-o", color=COL[m], lw=2, label=f"{m} · subidas")
        ax.plot(leads, dn, "--s", color=COL[m], lw=1.3, alpha=0.6, label=f"{m} · resto")
    ax.set_xticks(leads); ax.set_xlabel("anticipación (días)"); ax.set_ylabel("MAE (m³/s)")
    ax.set_title("(a) Subidas francas (dQ>1 m³/s·d) vs resto")
    ax.legend(fontsize=8, ncol=1); ax.grid(alpha=0.3)
    ax = axes[1]
    w = 0.26
    for i, m in enumerate(["Chronos-2", "TimesFM-2.5", "RA-TFT"]):
        nses = []
        for yr in [2024, 2025]:
            s = d[(d.model == m) & (d.lead == 7) & (d.date.dt.year == yr)]
            nses.append(1 - np.sum((s.obs - s.p50) ** 2) / np.sum((s.obs - s.obs.mean()) ** 2))
        ax.bar(np.arange(2) + (i - 1) * w, nses, w, color=COL[m], label=m, alpha=0.92)
        for x, v in zip(np.arange(2) + (i - 1) * w, nses):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["2024 (contexto = aforo real)", "2025 (contexto con reconstrucción)"])
    ax.set_ylabel("NSE a 7 días"); ax.set_title("(b) Sensibilidad a la calidad del contexto")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("FIG9 · Dónde duele: subidas por lluvia (información que no tienen) y "
                 "contexto degradado (2025)", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG9_fundacionales_dinamica_contexto.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig); log.info("FIG9 OK")


def stats(d):
    log.info("── Estadísticas de ruptura ──")
    for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT"]:
        s14 = d[(d.model == m) & (d.lead == 14)]
        s1 = d[(d.model == m) & (d.lead == 1)]
        log.info(f"{m:12s} · máx p50 a 1d: {s1.p50.max():6.1f} | a 14d: {s14.p50.max():6.1f} "
                 f"(obs máx del test: {s1.obs.max():.1f}) | p50 medio 14d: {s14.p50.mean():.1f}")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = cargar()
    fig6_hidrogramas(d); fig7_magnitud(d); fig8_banda(d); fig9_dinamica(d)
    stats(d)
    log.info(f"FIG6–FIG9 en {FIGDIR}")


if __name__ == "__main__":
    main()
