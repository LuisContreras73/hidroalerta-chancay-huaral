#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 151 — Frontera de Pareto: desempeño vs nº de parámetros (estilo paper de eficiencia).

Figura estándar en papers reconocidos (EfficientNet Tan&Le 2019; Chronos/TimesFM/Moirai;
DLinear Zeng 2023): eje x = params (log), eje y = desempeño, cada modelo un punto,
frontera de Pareto resaltada. Aquí: NSE por horizonte vs params.

HONESTIDAD (clave): los fundacionales (Chronos-2 ~120M, TimesFM ~231M) están
PREENTRENADOS y se usan ZERO-SHOT (no se entrenaron en nuestros datos) → se marcan
distinto y NO entran en la frontera de Pareto de los modelos entrenados-desde-cero
(comparar params de un pretrained con uno from-scratch es apples-to-oranges). Persistencia
y climatología = líneas base sin params (líneas horizontales de referencia).

Lee: 145_canonical_tft.csv, 150_canonical_hidsweep.csv (si existe), 137_foundation_zeroshot.csv.
Salida: reports/diagnostico/FIG16_pareto_perf_params.png

Run en .venv313 (cero entrenamiento; solo plotea CSVs existentes):
    python scripts/06_eval/151_pareto_perf_params.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")
FIGDIR = ROOT / "reports/diagnostico"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pareto151")

# params (K) conocidos (T3_model_summaries + smoke tests)
PARAMS_K = {"DLinear": 8, "RA-TFT (nuestro)": 357,
            "TFT canónico hid16": 24, "TFT canónico hid32": 90,
            "TFT canónico hid64": 343, "TFT canónico hid128": 1342,
            "Chronos-2 (zero-shot)": 119500, "TimesFM-2.5 (zero-shot)": 231300}
COL = {"entrenado": "#0B6E8C", "canónico": "#0A3D54", "dlinear": "#8FA6B4",
       "fundacional": "#7B2D8E", "base": "#B7C2CC"}


def nse_at(csv, model, h):
    try:
        t = pd.read_csv(OUT / csv)
        hcol = "h" if "h" in t.columns else "lead"
        s = t[(t.model == model) & (t[hcol] == h)]
        return float(s.NSE.values[0]) if len(s) else None
    except Exception:
        return None


def recolectar(h):
    pts = []  # (nombre, params_K, NSE, categoría)
    # entrenados desde cero
    pts.append(("DLinear", 8, nse_at("145_canonical_tft.csv", "DLinear", h), "dlinear"))
    pts.append(("RA-TFT (nuestro)", 357, nse_at("145_canonical_tft.csv", "RA-TFT (gen4, nuestro)", h), "entrenado"))
    pts.append(("TFT canónico hid128", 1342, nse_at("145_canonical_tft.csv", "TFT-canónico", h), "canónico"))
    # barrido de tamaño (si ya corrió el 150)
    f150 = OUT / "150_canonical_hidsweep.csv"
    if f150.exists():
        t = pd.read_csv(f150)
        for hid in [16, 32, 64]:
            s = t[(t.hid == hid) & (t.h == h)]
            if len(s):
                pts.append((f"TFT canónico hid{hid}", PARAMS_K[f"TFT canónico hid{hid}"],
                            float(s.NSE.values[0]), "canónico"))
    # fundacionales zero-shot (categoría aparte)
    pts.append(("Chronos-2 (zero-shot)", 119500, nse_at("137_foundation_zeroshot.csv", "Chronos-2", h), "fundacional"))
    pts.append(("TimesFM-2.5 (zero-shot)", 231300, nse_at("137_foundation_zeroshot.csv", "TimesFM-2.5", h), "fundacional"))
    return [(n, p, v, c) for n, p, v, c in pts if v is not None]


def pareto_front(pts):
    """Pareto entre entrenados-desde-cero (max NSE, min params). Excluye fundacionales."""
    ent = sorted([(p, v, n) for n, p, v, c in pts if c != "fundacional"])
    front, best = [], -9
    for p, v, n in ent:            # params ascendente
        if v > best:
            front.append((p, v)); best = v
    return front


def base_lines(h):
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    out = {}
    for m in ["Persistencia"]:
        s = fc[(fc.model == m) & (fc.lead == h)].dropna(subset=["obs", "p50"])
        if len(s):
            o, p = s.obs.values, s.p50.values
            out[m] = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, h in zip(axes, [7, 14]):
        pts = recolectar(h)
        front = pareto_front(pts)
        if front:
            fx, fy = zip(*front)
            ax.plot(fx, fy, "--", color="#C0392B", lw=1.4, zorder=1,
                    label="frontera de Pareto (entrenados)")
        for n, p, v, c in pts:
            mk = "D" if c == "fundacional" else "o"
            ax.scatter(p, v, s=130 if "nuestro" in n else 90, marker=mk,
                       color=COL[c], edgecolor="white", lw=1.2, zorder=3)
            dy = 0.012 if "hid16" not in n else -0.02
            ax.annotate(n.replace(" (zero-shot)", "\n(zero-shot)").replace("TFT canónico ", ""),
                        (p, v), fontsize=7.5, ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")
        for m, v in base_lines(h).items():
            ax.axhline(v, color=COL["base"], ls=":", lw=1.2)
            ax.text(ax.get_xlim()[1], v, f" {m} ({v:.2f})", fontsize=8,
                    color="#5B6B78", va="center", ha="right")
        ax.set_xscale("log"); ax.set_xlabel("parámetros (escala log)")
        ax.set_ylabel(f"NSE a {h} días"); ax.grid(alpha=0.3, which="both")
        ax.set_title(f"({'a' if h==7 else 'b'}) Horizonte {h} días")
        if h == 7:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("FIG16 · Desempeño vs parámetros — frontera de Pareto\n"
                 "◆ = fundacional preentrenado (zero-shot, no entrenado en nuestros datos: "
                 "params no comparables) · ● = entrenado desde cero",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "FIG16_pareto_perf_params.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    hid_sweep = (OUT / "150_canonical_hidsweep.csv").exists()
    log.info(f"FIG16 lista (barrido hid {'incluido' if hid_sweep else 'PENDIENTE — re-correr tras 150'})")


if __name__ == "__main__":
    main()
