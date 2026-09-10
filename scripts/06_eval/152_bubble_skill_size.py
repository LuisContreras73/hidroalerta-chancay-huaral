#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 152 — Bubble chart: habilidad vs horizonte con TAMAÑO = nº de parámetros.

Estilo de los papers de scaling / fundacionales (el tamaño del marcador codifica el
tamaño del modelo). Cuenta la historia central (NSE decae con el horizonte) AÑADIENDO
la dimensión de costo de un vistazo: burbuja grande = modelo grande.

Diseño (buenas prácticas dataviz):
  - eje x = horizonte, eje y = NSE (habilidad); una línea por modelo.
  - TAMAÑO de burbuja ∝ log10(params) — el rango es de 4 órdenes (8K→231M), por eso
    log (área lineal sería absurda: una burbuja 30 000× otra). Leyenda de tamaños explícita.
  - color por familia (paleta del proyecto); fundacionales ◆ (zero-shot, preentrenados).
  - persistencia/climatología como líneas finas de referencia (sin burbuja).

Salida: reports/diagnostico/FIG17_bubble_skill_size.png
Run en .venv313 (cero entrenamiento):
    python scripts/06_eval/152_bubble_skill_size.py
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
log = logging.getLogger("bubble152")

LEADS = [1, 3, 7, 14]
# (nombre, params_K, csv, model_en_csv, color, familia)
MODELOS = [
    ("DLinear", 8, "145_canonical_tft.csv", "DLinear", "#8FA6B4", "ent"),
    ("RA-TFT (nuestro)", 357, "145_canonical_tft.csv", "RA-TFT (gen4, nuestro)", "#0B6E8C", "ent"),
    ("TFT canónico", 1342, "145_canonical_tft.csv", "TFT-canónico", "#0A3D54", "ent"),
    ("Chronos-2", 119500, "137_foundation_zeroshot.csv", "Chronos-2", "#7B2D8E", "fm"),
    ("TimesFM-2.5", 231300, "137_foundation_zeroshot.csv", "TimesFM-2.5", "#B8860B", "fm"),
]


def nse_series(csv, model):
    t = pd.read_csv(OUT / csv)
    hcol = "h" if "h" in t.columns else "lead"
    out = {}
    for L in LEADS:
        s = t[(t.model == model) & (t[hcol] == L)]
        if len(s):
            out[L] = float(s.NSE.values[0])
    return out


def bsize(params_k):
    return 40 + 95 * np.log10(params_k)      # log → rango visible ~130-570


def main():
    fig, ax = plt.subplots(figsize=(11, 6.8))
    # referencias (persistencia, climatología) como líneas finas
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    for m, col, ls in [("Persistencia", "#B7C2CC", ":")]:
        ys = []
        for L in LEADS:
            s = fc[(fc.model == m) & (fc.lead == L)].dropna(subset=["obs", "p50"])
            o, p = s.obs.values, s.p50.values
            ys.append(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))
        ax.plot(LEADS, ys, ls, color=col, lw=1.5, zorder=1, label=f"{m} (base)")

    for nombre, pk, csv, mcsv, col, fam in MODELOS:
        ser = nse_series(csv, mcsv)
        xs = [L for L in LEADS if L in ser]; ys = [ser[L] for L in xs]
        if not xs:
            continue
        mk = "D" if fam == "fm" else "o"
        ax.plot(xs, ys, "-", color=col, lw=1.8, alpha=0.55, zorder=2)
        ax.scatter(xs, ys, s=bsize(pk), marker=mk, color=col, edgecolor="white",
                   lw=1.4, zorder=4, alpha=0.9)
        ax.annotate(nombre, (xs[-1], ys[-1]), fontsize=9, fontweight="bold", color=col,
                    xytext=(10, 0), textcoords="offset points", va="center")

    # leyenda de TAMAÑOS (qué significa cada diámetro)
    for pk, lab in [(10, "10 K"), (350, "350 K"), (1342, "1,3 M"), (120000, "120 M")]:
        ax.scatter([], [], s=bsize(pk), color="#5B6B78", edgecolor="white",
                   lw=1.2, label=f"{lab} parámetros")
    ax.set_xticks(LEADS); ax.set_xlabel("horizonte de pronóstico (días)")
    ax.set_ylabel("NSE (habilidad continua)")
    ax.set_xlim(0.3, 17); ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.92)
    ax.set_title("FIG17 · Habilidad vs horizonte — el TAMAÑO de la burbuja = nº de parámetros\n"
                 "◆ fundacional preentrenado (zero-shot) · ● entrenado desde cero · "
                 "tamaño en escala log", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "FIG17_bubble_skill_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("FIG17 lista")


if __name__ == "__main__":
    main()
