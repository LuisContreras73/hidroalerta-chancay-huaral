#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 136 — Regenera fig20–fig23 del paper/PPT con las métricas VIGENTES.

Motivación (2026-07-06): las cuatro figuras databan de generaciones anteriores de
modelos (scripts 100/101/105/107) y contradecían los resultados publicados — la
fig23 vieja afirmaba que RA-TFT no superaba a LightGBM, hallazgo revertido por el
tuning honesto (114) y las salidas corregidas (125). Aquí todo se reconstruye
desde los datos CURADOS del dashboard para que PPT y dashboard cuenten lo mismo:
  - metricas_modelos.csv       (leads 1,2,3,5,7,14 × 4 modelos)
  - forecast_multimodelo.csv   (p10/p50/p90 por día; leads 1,3,7,14)
  - D7_multientity             (climatología DOY pre-2024)
  - 101_anticipatory.csv       (fig21: ablación de señal, generación LightGBM —
                                se conserva como estudio de señal, reetiquetada)

Roles (sin solaparse):
  fig20 — ¿quién gana según el horizonte? (NSE + J_alert)
  fig21 — ¿de dónde sale la habilidad multi-día? (ablación lluvia futura)
  fig22 — calidad probabilística (CRPS + cobertura empírica de banda 80 %)
  fig23 — comparación justa en días comunes: a h≥7 RA-TFT SÍ supera a LightGBM

Run en .venv313:
    python scripts/06_eval/136_figs20_23_refresh.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")
FIGDIR = ROOT / "generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("figs20-23")

Q90 = 40.89
TEST_START = pd.Timestamp("2024-01-01")

COL = {"Climatología": "#9AA7B1", "Persistencia": "#B7C2CC",
       "LightGBM": "#8FA6B4", "HydroST": "#0A3D54", "RA-TFT": "#0B6E8C"}
LW = {"RA-TFT": 2.8}
ORDEN = ["Persistencia", "LightGBM", "RA-TFT"]


def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    so, sp = np.sqrt(np.clip(o, 0, None)), np.sqrt(np.clip(p, 0, None))
    nse_sq = 1 - np.sum((so - sp) ** 2) / np.sum((so - so.mean()) ** 2)
    oa, pa = o >= Q90, p >= Q90
    tp, fp, fn = np.sum(oa & pa), np.sum(~oa & pa), np.sum(oa & ~pa)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    j = 0.25 * nse_sq + 0.25 * nse + 0.30 * csi + 0.10 * pod - 0.10 * far
    return dict(NSE=nse, J=j, CSI=csi, POD=pod, FAR=far)


def cargar():
    met = pd.read_csv(DASH / "metricas_modelos.csv")
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    d7 = pd.read_csv(ROOT / "data/model_ready/D7_multientity.csv", parse_dates=["date"])
    qd = (d7[d7.entity_id == "sub_634"].set_index("date")["q_mm"] / (86.4 / 3062.62))
    doy = qd[qd.index < TEST_START].groupby(qd[qd.index < TEST_START].index.dayofyear).mean()
    return met, fc, doy


def por_dia_comun(fc, doy, lead):
    """P50 de los 3 modelos + climatología + obs en los días COMUNES del lead."""
    sub = fc[fc["lead"] == lead].pivot_table(index="date", columns="model",
                                             values="p50", aggfunc="last")
    obs = fc[(fc["lead"] == lead) & (fc["model"] == "RA-TFT")].set_index("date")["obs"]
    sub["obs"] = obs
    sub = sub.dropna(subset=["obs"] + ORDEN)
    sub["Climatología"] = [doy.get(d.dayofyear, np.nan) for d in sub.index]
    return sub


# ── fig20: ¿quién gana según el horizonte? ────────────────────────────────────
def fig20(met, fc, doy):
    leads_met = sorted(met["lead"].unique())            # 1,2,3,5,7,14
    leads_fc = [1, 3, 7, 14]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # (a) NSE (tabla curada = mismos números que el dashboard)
    ax = axes[0]
    for m in ["Persistencia", "LightGBM", "HydroST", "RA-TFT"]:
        d = met[met["model"] == m].sort_values("lead")
        ax.plot(d["lead"], d["NSE"], "-o", color=COL[m], lw=LW.get(m, 1.7),
                ms=6 if m == "RA-TFT" else 4.5, label=m, zorder=5 if m == "RA-TFT" else 3)
    clim_nse = [metrics(s["obs"], s["Climatología"])["NSE"]
                for s in (por_dia_comun(fc, doy, l) for l in leads_fc)]
    ax.plot(leads_fc, clim_nse, "--o", color=COL["Climatología"], lw=1.4, ms=4,
            label="Climatología")
    ax.set_title("(a) Habilidad continua (NSE) según el horizonte")
    ax.set_ylabel("Eficiencia de Nash–Sutcliffe (NSE)")
    ax.legend(fontsize=9, framealpha=0.9)

    # (b) J_alert por día (días comunes por lead)
    ax = axes[1]
    for m in ORDEN + ["Climatología"]:
        js = [metrics(s["obs"], s[m])["J"] for s in (por_dia_comun(fc, doy, l) for l in leads_fc)]
        est = "--o" if m == "Climatología" else "-o"
        ax.plot(leads_fc, js, est, color=COL[m], lw=LW.get(m, 1.7),
                ms=6 if m == "RA-TFT" else 4.5, label=m, zorder=5 if m == "RA-TFT" else 3)
    s1 = por_dia_comun(fc, doy, 1)
    hs = fc[(fc["lead"] == 1) & (fc["model"] == "HydroST")].dropna(subset=["obs", "p50"])
    ax.plot([1], [metrics(hs["obs"], hs["p50"])["J"]], "D", color=COL["HydroST"], ms=8,
            label="HydroST (1 día)", zorder=6)
    ax.set_title("(b) Puntaje compuesto de alerta (J_alert)")
    ax.set_ylabel("J_alert (mayor = mejor)")
    ax.legend(fontsize=9, framealpha=0.9)

    for ax in axes:
        ax.set_xlabel("Horizonte de pronóstico (días)")
        ax.set_xticks(leads_met); ax.grid(alpha=0.3)
    fig.suptitle("Habilidad según el horizonte — prueba 2024–2025\n"
                 "A 1 día la persistencia y HydroST fijan el techo; desde 5–7 días "
                 "solo el RA-TFT sostiene la habilidad (NSE 0,68 a 7 d; 0,49 a 14 d)",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig20_multi_horizon.png", dpi=300, bbox_inches="tight")
    plt.close(fig); log.info("fig20 OK")


# ── fig21: ablación de la señal anticipatoria (reetiquetada, gen. LightGBM) ──
def fig21():
    a = pd.read_csv(ROOT / "outputs/ml_Q/101_anticipatory.csv")
    NOM = {"Persistence": "Persistencia", "ML (historical)": "ML (solo pasado)",
           "ML + future rain": "ML + lluvia futura (perfecta)"}
    C = {"Persistencia": COL["Persistencia"], "ML (solo pasado)": COL["LightGBM"],
         "ML + lluvia futura (perfecta)": COL["RA-TFT"]}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    for ax, key, tit, ylab in [
            (axes[0], "J", "(a) Puntaje de alerta según el horizonte", "J_alert (mayor = mejor)"),
            (axes[1], "CSI", "(b) Detección de crecidas (CSI)", "CSI (mayor = mejor)")]:
        for m0, m in NOM.items():
            d = a[a["model"] == m0].sort_values("h")
            ax.plot(d["h"], d[key], "-o", color=C[m], lw=2.4 if "lluvia" in m else 1.7,
                    ms=6 if "lluvia" in m else 4.5, label=m,
                    zorder=5 if "lluvia" in m else 3)
        ax.set_title(tit); ax.set_ylabel(ylab)
        ax.set_xlabel("Horizonte de pronóstico (días)")
        ax.set_xticks(sorted(a["h"].unique())); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle("¿De dónde sale la habilidad multi-día? La lluvia sobre el horizonte\n"
                 "Ablación de señal (generación LightGBM): sin lluvia futura, el ML no "
                 "supera a la persistencia; el RA-TFT operacionaliza esta señal (figs. 20, 22–23)",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig21_anticipatory.png", dpi=300, bbox_inches="tight")
    plt.close(fig); log.info("fig21 OK")


# ── fig22: calidad probabilística por horizonte ───────────────────────────────
def fig22(met, fc):
    leads_met = sorted(met["lead"].unique())
    leads_fc = [1, 3, 7, 14]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    ax = axes[0]
    for m in ORDEN:
        d = met[met["model"] == m].sort_values("lead").dropna(subset=["CRPS"])
        ax.plot(d["lead"], d["CRPS"], "-o", color=COL[m], lw=LW.get(m, 1.7),
                ms=6 if m == "RA-TFT" else 4.5, label=m, zorder=5 if m == "RA-TFT" else 3)
    ax.set_title("(a) Calidad probabilística (CRPS; menor = mejor)")
    ax.set_ylabel("CRPS (m³/s)"); ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xticks(leads_met)

    ax = axes[1]
    for m in ["LightGBM", "RA-TFT"]:
        cov = []
        for l in leads_fc:
            d = fc[(fc["lead"] == l) & (fc["model"] == m)].dropna(subset=["obs", "p10", "p90"])
            cov.append(100 * ((d["obs"] >= d["p10"]) & (d["obs"] <= d["p90"])).mean())
        ax.plot(leads_fc, cov, "-o", color=COL[m], lw=LW.get(m, 1.7),
                ms=6 if m == "RA-TFT" else 4.5, label=m, zorder=5 if m == "RA-TFT" else 3)
    ax.axhline(80, color="#33414C", ls=":", lw=1.2)
    ax.text(13.9, 80.8, "nominal 80 %", ha="right", fontsize=9, color="#33414C")
    ax.set_ylim(40, 100)
    ax.set_title("(b) Cobertura empírica de la banda P10–P90")
    ax.set_ylabel("% de días con el aforo dentro de la banda")
    ax.legend(fontsize=9, framealpha=0.9); ax.set_xticks(leads_fc)

    for ax in axes:
        ax.set_xlabel("Horizonte de pronóstico (días)"); ax.grid(alpha=0.3)
    fig.suptitle("Calidad probabilística por horizonte — prueba 2024–2025\n"
                 "RA-TFT con el mejor CRPS a 1–3 y 14 días (empate técnico a 7); su banda "
                 "cubre 85 % a 1 día y se estrecha a multi-día (57–62 %): sobreconfianza "
                 "declarada, con recalibración prevista en el plan de monitoreo",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig22_ratft_multihorizon.png", dpi=300, bbox_inches="tight")
    plt.close(fig); log.info("fig22 OK")


# ── fig23: comparación justa en días comunes ──────────────────────────────────
def fig23(fc, doy):
    leads = [1, 3, 7, 14]
    res = {m: [] for m in ORDEN}
    ns = []
    for l in leads:
        s = por_dia_comun(fc, doy, l)
        ns.append(len(s))
        for m in ORDEN:
            res[m].append(metrics(s["obs"], s[m])["NSE"])
    crps = {m: pd.read_csv(DASH / "metricas_modelos.csv").query("model==@m")
            .set_index("lead")["CRPS"].reindex(leads).tolist() for m in ORDEN}

    x = np.arange(len(leads)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    for ax, dat, tit, ylab, better in [
            (axes[0], res, "(a) Habilidad continua (NSE) — mismos días para todos",
             "NSE (mayor = mejor)", "up"),
            (axes[1], crps, "(b) Calidad probabilística (CRPS)",
             "CRPS (m³/s; menor = mejor)", "down")]:
        for i, m in enumerate(ORDEN):
            ax.bar(x + (i - 1) * w, dat[m], w, color=COL[m], alpha=0.92, label=m)
            for xi, v in zip(x + (i - 1) * w, dat[m]):
                if np.isfinite(v):
                    ax.text(xi, v + (0.012 if better == "up" else 0.03),
                            f"{v:.2f}", ha="center", fontsize=8, color="#33414C")
        ax.set_xticks(x); ax.set_xticklabels([f"{l} d" for l in leads])
        ax.set_xlabel("Horizonte de pronóstico"); ax.set_ylabel(ylab)
        ax.set_title(tit); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle("Comparación justa — mismos días de prueba, misma información conocida a futuro\n"
                 f"A 7–14 días el RA-TFT supera a LightGBM y a la persistencia en NSE; en CRPS "
                 f"gana a 1–3 y 14 días (a 7, empate técnico con LightGBM) — "
                 f"N por horizonte: {', '.join(map(str, ns))}",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig23_fair_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig); log.info("fig23 OK")


def main():
    met, fc, doy = cargar()
    fig20(met, fc, doy)
    fig21()
    fig22(met, fc)
    fig23(fc, doy)
    log.info(f"4 figuras regeneradas en {FIGDIR}")


if __name__ == "__main__":
    main()
