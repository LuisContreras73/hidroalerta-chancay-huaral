#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 139 — Comparaciones profundas fundacionales (gen6) vs locales (gen4).

Más allá de la tabla de métricas (137) y el diagnóstico de ruptura (138):
  FIG10  Curvas de habilidad COMPLETAS por horizonte (NSE + CRPS) con todos los
         modelos — la figura panorámica del paper.
  FIG11  Tasa de victoria por pares (|error| menor, día a día) con IC bootstrap
         95 % + desglose temporada húmeda/seca — ¿la ventaja es significativa y
         dónde vive?
  FIG12  Mecanismo: (a) desfase efectivo (correlación cruzada pred↔obs: ¿el
         pronóstico anticipa o repite el pasado?); (b) revisión del pronóstico
         del día del PICO máximo del test — lectura operativa.
  Tabla  Operativa: parámetros, latencia de inferencia medida, VRAM, requisitos
         (outputs/ml_Q/139_tabla_operativa.csv).

Run en .venv313:
    python scripts/06_eval/139_foundation_vs_local.py
"""
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUTML = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "reports/diagnostico"
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fm139")

COL = {"Chronos-2": "#7B2D8E", "TimesFM-2.5": "#B8860B", "RA-TFT": "#0B6E8C",
       "HydroST": "#0A3D54", "LightGBM": "#8FA6B4", "Persistencia": "#B7C2CC"}
LEADS = [1, 3, 7, 14]
MESES_HUMEDOS = {12, 1, 2, 3, 4}


def cargar_panel():
    """Panel largo (date, lead, model, obs, p10, p50, p90) con gen4 + gen6."""
    ch = pd.read_csv(OUTML / "137_chronos2_preds.csv", parse_dates=["date"]).assign(model="Chronos-2")
    tf = pd.read_csv(OUTML / "137_timesfm25_preds.csv", parse_dates=["date"]).assign(model="TimesFM-2.5")
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    fc = fc[fc.model.isin(["RA-TFT", "Persistencia", "LightGBM"])]
    d = pd.concat([ch, tf, fc[["date", "model", "lead", "obs", "p10", "p50", "p90"]]],
                  ignore_index=True).dropna(subset=["obs", "p50"])
    return d


def fig10_curvas(d):
    met = pd.read_csv(DASH / "metricas_modelos.csv")
    fm = pd.read_csv(OUTML / "137_foundation_zeroshot.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    ax = axes[0]
    for m in ["Persistencia", "LightGBM", "HydroST", "RA-TFT"]:
        s = met[met.model == m].sort_values("lead")
        ax.plot(s.lead, s.NSE, "-o", color=COL[m], lw=2.4 if m == "RA-TFT" else 1.5,
                ms=5, label=m)
    for m in ["Chronos-2", "TimesFM-2.5"]:
        s = fm[fm.model == m].sort_values("lead")
        ax.plot(s.lead, s.NSE, "--D", color=COL[m], lw=2.2, ms=6,
                label=f"{m} (zero-shot)")
    ax.set_ylabel("NSE"); ax.set_title("(a) Habilidad continua (NSE)")
    ax.legend(fontsize=8.5, ncol=2)
    ax = axes[1]
    for m in ["Persistencia", "LightGBM", "RA-TFT"]:
        s = met[met.model == m].sort_values("lead").dropna(subset=["CRPS"])
        ax.plot(s.lead, s.CRPS, "-o", color=COL[m], lw=2.4 if m == "RA-TFT" else 1.5,
                ms=5, label=m)
    for m in ["Chronos-2", "TimesFM-2.5"]:
        s = fm[fm.model == m].sort_values("lead")
        ax.plot(s.lead, s.CRPS, "--D", color=COL[m], lw=2.2, ms=6,
                label=f"{m} (zero-shot)")
    ax.set_ylabel("CRPS (m³/s; menor = mejor)")
    ax.set_title("(b) Calidad probabilística (CRPS)")
    for ax in axes:
        ax.set_xticks(sorted(met.lead.unique())); ax.grid(alpha=0.3)
        ax.set_xlabel("horizonte de pronóstico (días)")
    fig.suptitle("FIG10 · Panorámica con fundacionales — Chronos-2 zero-shot redefine el techo\n"
                 "del RÉGIMEN (NSE/CRPS), pero no el de la ALERTA (POD=0 a h≥3; ver FIG7)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG10_panoramica_con_fundacionales.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig); log.info("FIG10 OK")


def fig11_winrate(d, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    pares = [("Chronos-2", "RA-TFT"), ("Chronos-2", "Persistencia"),
             ("TimesFM-2.5", "RA-TFT")]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    ax = axes[0]
    for k, (A, Bm) in enumerate(pares):
        wr, lo, hi = [], [], []
        for L in LEADS:
            p = d[d.lead == L].pivot_table(index="date", columns="model",
                                           values=["obs", "p50"], aggfunc="last")
            ea = (p[("p50", A)] - p[("obs", A)]).abs()
            eb = (p[("p50", Bm)] - p[("obs", Bm)]).abs()
            m = ea.notna() & eb.notna()
            w = (ea[m] < eb[m]).values.astype(float)
            wr.append(100 * w.mean())
            bs = [w[rng.integers(0, len(w), len(w))].mean() for _ in range(B)]
            lo.append(100 * (w.mean() - np.percentile(bs, 2.5)))
            hi.append(100 * (np.percentile(bs, 97.5) - w.mean()))
        x = np.array(LEADS) * (1 + 0.03 * (k - 1))
        ax.errorbar(x, wr, yerr=[lo, hi], fmt="-o", color=COL[A],
                    alpha=1 if Bm == "RA-TFT" else 0.55,
                    lw=2 if Bm == "RA-TFT" else 1.3, capsize=3,
                    label=f"{A} vs {Bm}")
    ax.axhline(50, color="#33414C", ls=":", lw=1.2)
    ax.text(13.9, 50.8, "empate (50 %)", ha="right", fontsize=9)
    ax.set_xticks(LEADS); ax.set_ylabel("% de días con |error| menor")
    ax.set_xlabel("horizonte (días)")
    ax.set_title("(a) Tasa de victoria día a día (IC bootstrap 95 %)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    w = 0.26
    for i, m in enumerate(["Chronos-2", "TimesFM-2.5", "RA-TFT"]):
        maes = []
        for hum in [True, False]:
            s = d[(d.model == m) & (d.lead == 7)]
            s = s[s.date.dt.month.isin(MESES_HUMEDOS) == hum]
            maes.append((s.obs - s.p50).abs().mean())
        ax.bar(np.arange(2) + (i - 1) * w, maes, w, color=COL[m], label=m, alpha=0.92)
        for x, v in zip(np.arange(2) + (i - 1) * w, maes):
            ax.text(x, v + 0.08, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(2)); ax.set_xticklabels(["húmeda (dic–abr)", "seca (may–nov)"])
    ax.set_ylabel("MAE a 7 días (m³/s)")
    ax.set_title("(b) ¿Dónde vive la ventaja? — por temporada")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("FIG11 · Significancia y territorio de la ventaja",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG11_winrate_temporadas.png", dpi=200, bbox_inches="tight")
    plt.close(fig); log.info("FIG11 OK")


def fig12_mecanismo(d):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    # (a) correlación pred↔obs desplazado, a 7 días: TODOS pican en s≈7 (= eco del
    # estado inicial; hallazgo: sin lluvia futura NADIE anticipa, ni el RA-TFT) —
    # el matiz está en la altura y el hombro de cada curva en s=0.
    ax = axes[0]
    L = 7
    for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT", "Persistencia"]:
        s = d[(d.model == m) & (d.lead == L)].set_index("date").sort_index()
        obs_full = s["obs"]
        shifts = list(range(0, 11))
        rs = []
        for sh in shifts:
            o = obs_full.shift(sh).values
            mm = np.isfinite(o) & np.isfinite(s["p50"].values)
            rs.append(np.corrcoef(s["p50"].values[mm], o[mm])[0, 1])
        ax.plot(shifts, rs, "-o", color=COL[m], lw=2, ms=4.5, label=m)
        k = int(np.argmax(rs))
        ax.plot(shifts[k], rs[k], "o", color=COL[m], ms=10, mfc="none", mew=2)
    ax.axvline(0, color="#2E8B6F", ls=":", lw=1.4)
    ax.text(0.15, ax.get_ylim()[0] + 0.02, "anticipación real (s=0)",
            color="#2E8B6F", fontsize=9, rotation=90, va="bottom")
    ax.axvline(L, color="#C0392B", ls=":", lw=1.4)
    ax.text(L + 0.15, ax.get_ylim()[0] + 0.02, "eco del estado inicial (s=7)",
            color="#C0392B", fontsize=9, rotation=90, va="bottom")
    ax.set_xticks(shifts)
    ax.set_xlabel("desfase s (días): corr( pred(t), obs(t−s) )")
    ax.set_ylabel("correlación")
    ax.set_title("(a) A 7 días, TODOS pican en s≈7: sin lluvia futura nadie anticipa")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    # (b) revisión del pronóstico para el día del pico máximo del test
    ax = axes[1]
    obs1 = d[(d.model == "Chronos-2") & (d.lead == 1)]
    pico = obs1.loc[obs1.obs.idxmax()]
    T, qmax = pico["date"], pico["obs"]
    for m in ["Chronos-2", "TimesFM-2.5", "RA-TFT"]:
        xs, ys = [], []
        for L in LEADS:
            s = d[(d.model == m) & (d.lead == L) & (d.date == T)]
            if len(s):
                xs.append(L); ys.append(float(s.p50.iloc[0]))
        ax.plot(xs, ys, "-o", color=COL[m], lw=2, ms=7, label=m)
    ax.axhline(qmax, color="#0C1E2A", lw=2)
    ax.text(13.8, qmax + 0.8, f"observado: {qmax:.1f} m³/s", ha="right", fontsize=9.5)
    ax.axhline(40.89, color="#C0392B", ls=":", lw=1.2)
    ax.text(13.8, 41.7, "umbral Q90", ha="right", fontsize=9, color="#C0392B")
    ax.invert_xaxis()
    ax.set_xticks(LEADS); ax.set_xlabel("días de anticipación (→ se acerca el evento)")
    ax.set_ylabel("p50 pronosticado para el día del pico (m³/s)")
    ax.set_title(f"(b) Revisión del pronóstico — pico del test ({T.date()})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("FIG12 · El mecanismo: los fundacionales repiten el pasado con elegancia; "
                 "la anticipación del evento exige meteorología",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG12_mecanismo_desfase_revision.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig); log.info("FIG12 OK")


def tabla_operativa():
    """Latencia de inferencia medida en ESTA máquina + ficha comparativa."""
    import torch
    n, reps = 128, 3
    ctx = np.random.rand(n, 512).astype("float32") * 20 + 10
    filas = []

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(str(ROOT / "models/foundation/chronos-2"),
                                               device_map="cuda", torch_dtype=torch.bfloat16)
    x = torch.tensor(ctx).unsqueeze(1)
    pipe.predict_quantiles(inputs=x[:8], prediction_length=14, quantile_levels=[0.1, 0.5, 0.9])
    t0 = time.perf_counter()
    for _ in range(reps):
        pipe.predict_quantiles(inputs=x, prediction_length=14, quantile_levels=[0.1, 0.5, 0.9])
    ms_ch = (time.perf_counter() - t0) / (reps * n) * 1000
    del pipe; torch.cuda.empty_cache()

    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        str(ROOT / "models/foundation/timesfm-2.5-200m"))
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64,
                                     normalize_inputs=True,
                                     use_continuous_quantile_head=True,
                                     fix_quantile_crossing=True))
    m.forecast(horizon=14, inputs=list(ctx[:8]))
    t0 = time.perf_counter()
    for _ in range(reps):
        m.forecast(horizon=14, inputs=list(ctx))
    ms_tf = (time.perf_counter() - t0) / (reps * n) * 1000

    filas = [
        dict(modelo="Chronos-2", params_M=119.5, entrenamiento_local="no (zero-shot)",
             datos_locales="512 d de contexto", ms_por_emision=round(ms_ch, 1),
             covariables="sí (pasadas y futuras, in-context)",
             alerta_util="1–2 d", banda="calibrada (85–88 %)"),
        dict(modelo="TimesFM-2.5", params_M=231.3, entrenamiento_local="no (zero-shot)",
             datos_locales="512 d de contexto", ms_por_emision=round(ms_tf, 1),
             covariables="no nativas", alerta_util="1 d", banda="calibrada (81–87 %)"),
        dict(modelo="RA-TFT (gen4)", params_M=0.4, entrenamiento_local="sí (~1 h GPU + tuning)",
             datos_locales="45 años forzantes + aforo", ms_por_emision=np.nan,
             covariables="sí (arquitectura propia; GFS en curso)",
             alerta_util="1–2 d (prob. a 3–14)", banda="sobreconfiada multi-día (57–62 %)"),
    ]
    t = pd.DataFrame(filas)
    t.to_csv(OUTML / "139_tabla_operativa.csv", index=False)
    log.info("\n" + t.to_string(index=False))


def main():
    d = cargar_panel()
    fig10_curvas(d)
    fig11_winrate(d)
    fig12_mecanismo(d)
    tabla_operativa()
    log.info(f"FIG10–FIG12 en {FIGDIR} · tabla en outputs/ml_Q/139_tabla_operativa.csv")


if __name__ == "__main__":
    main()
