#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 189 — Figura PANORÁMICA para presentar: local (canónico+GRU) vs FUNDACIONALES zero-shot
(Chronos-2, TimesFM-2.5) vs baselines, NSE y CRPS por horizonte, test 2024-25.

Une (no re-corre) los resultados ya calculados:
  · outputs/ml_Q/188_benchmark_test.csv   (local + persistencia + climatología, mismo harness)
  · outputs/ml_Q/137_foundation_zeroshot.csv (Chronos-2, TimesFM-2.5, zero-shot)
Mismo test, mismos leads, N comparable → tabla y figura únicas para la exposición.

Mensaje: los fundacionales igualan al local a h1 (nowcast) SIN entrenar, pero se estancan a horizonte
largo porque son UNIVARIADOS (solo caudal); el local gana en subestacional usando lluvia+ENSO.

Salida: outputs/ml_Q/189_panoramica.csv · reports/figures/DI_panoramica_local_vs_fundacional.png
Run (.venv313): python scripts/06_eval/189_panoramica_local_fundacional.py
"""
import logging
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log = logging.getLogger("pano189")

loc = pd.read_csv(OUT / "188_benchmark_test.csv")
fnd = pd.read_csv(OUT / "137_foundation_zeroshot.csv").rename(columns={"lead": "h"})
loc = loc[loc.modelo.isin(["canónico+GRU", "persistencia", "climatología"])][["modelo", "h", "NSE", "CRPS"]]
fnd = fnd[["model", "h", "NSE", "CRPS"]].rename(columns={"model": "modelo"})
pan = pd.concat([loc, fnd], ignore_index=True)
pan.to_csv(OUT / "189_panoramica.csv", index=False)
log.info("\n=== NSE por horizonte ===\n" + pan.pivot_table(index="modelo", columns="h", values="NSE").to_string())
log.info("\n=== CRPS por horizonte ===\n" + pan.pivot_table(index="modelo", columns="h", values="CRPS").to_string())

STYLE = {
    "canónico+GRU": ("#2f9e6f", "-o", 2.6, "canónico+GRU (local, entrenado)"),
    "Chronos-2": ("#7b3fb0", "-s", 2.2, "Chronos-2 (fundacional, zero-shot)"),
    "TimesFM-2.5": ("#c05a2e", "-^", 2.2, "TimesFM-2.5 (fundacional, zero-shot)"),
    "persistencia": ("#888", "--", 1.4, "persistencia"),
    "climatología": ("#b8860b", ":", 1.4, "climatología"),
}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for mdl, (c, ls, lw, lab) in STYLE.items():
    s = pan[pan.modelo == mdl].sort_values("h")
    if s.empty: continue
    ax[0].plot(s["h"], s["NSE"], ls, color=c, lw=lw, ms=5, label=lab)
    ax[1].plot(s["h"], s["CRPS"], ls, color=c, lw=lw, ms=5, label=lab)
ax[0].axvspan(0.5, 3.5, color="#7b3fb0", alpha=0.06)
ax[0].text(2, 0.30, "nowcast:\nfundacional ≈ local", ha="center", fontsize=8, color="#5a2d82")
ax[0].axvspan(3.5, 14.5, color="#2f9e6f", alpha=0.06)
ax[0].text(10.5, 0.30, "subestacional:\nel local GANA (lluvia+ENSO)", ha="center", fontsize=8, color="#1f6f4f")
ax[0].set_xlabel("horizonte (días)"); ax[0].set_ylabel("NSE (↑ mejor)"); ax[0].set_xticks([1, 3, 7, 14])
ax[0].set_title("(a) Habilidad determinista — NSE", fontsize=11, fontweight="bold"); ax[0].grid(lw=.3, alpha=.4); ax[0].legend(fontsize=8)
ax[1].set_xlabel("horizonte (días)"); ax[1].set_ylabel("CRPS (↓ mejor)"); ax[1].set_xticks([1, 3, 7, 14])
ax[1].set_title("(b) Habilidad probabilística — CRPS", fontsize=11, fontweight="bold"); ax[1].grid(lw=.3, alpha=.4); ax[1].legend(fontsize=8)
fig.suptitle("Local entrenado vs. Fundacionales zero-shot vs. baselines — test 2024-25 (Santo Domingo, Chancay-Huaral)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "DI_panoramica_local_vs_fundacional.png", dpi=150)
log.info(f"figura: {FIG/'DI_panoramica_local_vs_fundacional.png'}"); log.info("PANORAMICA_DONE")
