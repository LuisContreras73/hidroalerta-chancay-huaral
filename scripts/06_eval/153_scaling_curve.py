#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 153 — Curva de ESCALADO: una misma arquitectura a tamaños crecientes.

El plot de "familia de modelos" de los papers de scaling (EfficientNet-B0..B7;
Chronos tiny→large): la MISMA arquitectura a params crecientes, burbujas que crecen
a lo largo de la curva, para leer si escalar ayuda o satura/sobreajusta.

Familia principal: TFT canónico a hid ∈ {16,32,64,128} → {24K,90K,343K,1342K}
(barrido 150 + hid128 del 145). Referencias (un punto): RA-TFT nuestro (357K), DLinear (8K).

Salida: reports/diagnostico/FIG18_scaling_curve.png
Prerrequisito: outputs/ml_Q/150_canonical_hidsweep.csv (barrido). Sin él, avisa y sale.

Run en .venv313 (cero entrenamiento):
    python scripts/06_eval/153_scaling_curve.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "reports/diagnostico"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("scaling153")

PARAMS_K = {16: 24, 32: 90, 64: 343, 128: 1342}
CANON = "#0A3D54"


def bsize(pk):
    return 40 + 430 * np.sqrt(pk / 1342)     # radio ∝ params^0.25 → crecimiento visible


def nse_at(csv, model, h, hcol="h"):
    t = pd.read_csv(OUT / csv)
    hc = hcol if hcol in t.columns else "lead"
    s = t[(t.model == model) & (t[hc] == h)]
    return float(s.NSE.values[0]) if len(s) else None


def main():
    f150 = OUT / "150_canonical_hidsweep.csv"
    if not f150.exists():
        log.warning("Falta 150_canonical_hidsweep.csv — corre primero el barrido 150. Salgo.")
        return
    sweep = pd.read_csv(f150)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, h in zip(axes, [7, 14]):
        # familia canónico: hid 16/32/64 (150) + 128 (145)
        fam = []
        for hid in [16, 32, 64]:
            s = sweep[(sweep.hid == hid) & (sweep.h == h)]
            if len(s):
                fam.append((PARAMS_K[hid], float(s.NSE.values[0]), hid))
        v128 = nse_at("145_canonical_tft.csv", "TFT-canónico", h)
        if v128 is not None:
            fam.append((PARAMS_K[128], v128, 128))
        fam.sort()
        if fam:
            fx, fy, fh = zip(*fam)
            ax.plot(fx, fy, "-", color=CANON, lw=2, zorder=2, alpha=0.7)
            ax.scatter(fx, fy, s=[bsize(p) for p in fx], color=CANON, edgecolor="white",
                       lw=1.5, zorder=4, alpha=0.9)
            for p, v, hd in fam:
                ax.annotate(f"hid{hd}\n{p}K", (p, v), fontsize=8, ha="center",
                            va="bottom", xytext=(0, 10), textcoords="offset points",
                            color=CANON, fontweight="bold")
        # referencias (1 punto cada uno)
        ra = nse_at("145_canonical_tft.csv", "RA-TFT (gen4, nuestro)", h)
        dl = nse_at("145_canonical_tft.csv", "DLinear", h)
        if ra is not None:
            ax.scatter([357], [ra], s=bsize(357), color="#0B6E8C", edgecolor="white",
                       lw=1.5, marker="s", zorder=4)
            ax.annotate("RA-TFT\n(nuestro)", (357, ra), fontsize=8, ha="center", va="top",
                        xytext=(0, -12), textcoords="offset points", color="#0B6E8C")
        if dl is not None:
            ax.scatter([8], [dl], s=bsize(8), color="#8FA6B4", edgecolor="white",
                       lw=1.5, marker="^", zorder=4)
            ax.annotate("DLinear", (8, dl), fontsize=8, ha="left", va="center",
                        xytext=(8, 0), textcoords="offset points", color="#5B6B78")
        ax.set_xscale("log"); ax.set_xlabel("parámetros (escala log)")
        ax.set_ylabel(f"NSE a {h} días"); ax.grid(alpha=0.3, which="both")
        ax.set_title(f"({'a' if h==7 else 'b'}) Horizonte {h} días")
    fig.suptitle("FIG18 · Curva de escalado del TFT canónico (misma arquitectura, params crecientes)\n"
                 "burbuja crece con el tamaño · ¿escalar ayuda o satura? · ■ RA-TFT nuestro · ▲ DLinear",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "FIG18_scaling_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("FIG18 (curva de escalado) lista")


if __name__ == "__main__":
    main()
