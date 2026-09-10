#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 161 — Reconstrucción del atractor A NIVEL MENSUAL (espejo de Saéz et al. 2026)
y comparación con las reconstrucciones diarias.

MOTIVACIÓN. El paper (Saéz et al. 2026, HP 40:e70582, cuenca B23 = Chancay-Huaral)
trabaja MENSUAL. Aquí replicamos su construcción para nuestra propia serie y comparamos
tres embeddings, para justificar con evidencia por qué el embedding diferencial funciona
mensual pero no diario.

MÉTODO (fiel al paper donde importa; unidades y fuentes citadas):
  · Agregación a mensual: caudal → lámina de escorrentía ρ (mm/mes) = Σ_días q·86400/A·1000
    (A = 3062.62 km²); precipitación ϖ (mm/mes) = Σ_días pr. Transformación raíz:
    R = √ρ, P = √ϖ  (Saéz et al. 2026, §3.3; probaron log/none/√ y solo √ dio modelos
    estables).
  · Re-muestreo a 12 puntos/mes para estimar derivadas finas. El paper usa interpolación
    de Stineman (Stineman 1980) que minimiza inflexiones y evita el overshoot de splines.
    Aquí usamos PCHIP (Fritsch & Carlson 1980, "Monotone Piecewise Cubic Interpolation",
    SIAM J. Numer. Anal. 17:238) — interpolante cúbico monótono/shape-preserving, mismo
    espíritu (sin overshoot), disponible en scipy. DECLARADO como sustituto de Stineman.
  · Retrato diferencial (R, dR/dt): reconstrucción del espacio de fases por derivadas
    (Packard et al. 1980, PRL 45:712; Takens 1981). dR/dt = derivada analítica del
    interpolante PCHIP (unidad: √mm por mes).
  · Retrato (P, R): plano precipitación–escorrentía (Fig. 5 del paper).

COMPARACIÓN (3 embeddings del MISMO caudal observado):
  (1) mensual (R, dR/dt)  → esperado LIMPIO, como su Fig. 6.
  (2) diario  (R, dR/dt)  → esperado RUIDOSO (derivada diaria domina el ruido) — por eso
      el 159 no se parecía al atractor.
  (3) diario ventanas-60d→PCA (fig25 / script 111) → LIMPIO, criterio correcto a diario.

Salida: dinamica_no_lineal/figuras/figura3_reconstruccion_mensual_diaria.png
Run en .venv313: python scripts/06_eval/161_monthly_reconstruction.py

Referencias: Takens F. (1981) LNM 898; Packard et al. (1980) PRL 45:712;
Stineman R.W. (1980) Creative Computing 6(7):54; Fritsch & Carlson (1980) SIAM JNA 17:238;
Savitzky & Golay (1964) Anal. Chem. 36:1627; Saéz et al. (2026) HP 40:e70582.
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("recon161")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R
AREA_KM2 = 3062.62
L = 60          # ventana diaria para el embedding PCA (= fig25/script 111)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14)
    q = pd.Series(df["q"].values, index=df.index)         # m³/s diario (completo)
    pr = pd.Series(df["pr"].values, index=df.index)        # mm/día
    api = pd.Series(df["api"].values, index=df.index)

    # ── (1) MENSUAL estilo paper ────────────────────────────────────────────────
    # escorrentía mm/mes = Σ_días q[m³/s]·86400[s]/A[m²]·1000[mm/m]
    A_m2 = AREA_KM2 * 1e6
    rho = (q * 86400.0 / A_m2 * 1000.0).resample("MS").sum()     # mm/mes
    varpi = pr.resample("MS").sum()                              # mm/mes
    Rm = np.sqrt(np.clip(rho.values, 0, None))                  # R = √ρ
    Pm = np.sqrt(np.clip(varpi.values, 0, None))                # P = √ϖ
    tm = np.arange(len(Rm), dtype=float)                        # índice de mes
    # re-muestreo 12/mes (PCHIP) para derivar fino
    tf = np.linspace(0, len(Rm) - 1, (len(Rm) - 1) * 12 + 1)
    pch = PchipInterpolator(tm, Rm)
    Rf = pch(tf); dRf = pch.derivative()(tf) * 12.0             # dR/dmes (12 sub-pasos/mes)
    Pf = PchipInterpolator(tm, Pm)(tf)
    log.info(f"mensual: {len(Rm)} meses ({df.index.min().year}-{df.index.max().year}); "
             f"ρ mediana={np.median(rho.values):.1f} mm/mes, máx={rho.values.max():.0f}")

    # ── (2) DIARIO diferencial (R, dR/dt) — el criterio ruidoso del 159 ─────────
    Rd = np.sqrt(np.clip(q.values, 0, None))
    dRd = savgol_filter(Rd, 15, 3, deriv=1)                     # d/día

    # ── (3) DIARIO ventanas-60d → PCA (fig25) — criterio correcto a diario ──────
    prv = pr.values; apiv = api.values; qv = q.values
    F = np.stack([qv, prv, apiv]); mu = F.mean(1, keepdims=True); sd = F.std(1, keepdims=True) + 1e-6
    Fz = (F - mu) / sd
    ends = [i for i in range(L, len(qv)) if df.index[i] >= pd.Timestamp("2015-01-01")]
    W = np.stack([Fz[:, e - L:e].ravel() for e in ends]).astype(np.float32)
    Epca = PCA(2).fit_transform(W)
    Q90 = R.Q90; flood = qv[ends] > Q90

    # ── figura comparativa ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.7), dpi=140)
    # (P,R) mensual — Fig. 5 del paper
    sc = ax[0].scatter(Pf, Rf, s=6, c=tf, cmap="viridis", alpha=0.6, linewidths=0)
    ax[0].plot(Pf, Rf, color="#9fb3c0", lw=0.3, alpha=0.5, zorder=0)
    ax[0].set_title("1 · MENSUAL plano (P, R)  [≈ Fig. 5 del paper]", fontsize=10)
    ax[0].set_xlabel("P = √precip  (√mm)"); ax[0].set_ylabel("R = √escorrentía  (√mm)")
    # (R,dR/dt) mensual — Fig. 6 del paper (LIMPIO)
    ax[1].plot(Rf, dRf, color="#0B6E8C", lw=0.6, alpha=0.8)
    ax[1].axhline(0, color="#e2e8ee", lw=0.8)
    ax[1].set_title("2 · MENSUAL diferencial (R, dR/dt)  [≈ Fig. 6]\nLIMPIO — como el paper", fontsize=10)
    ax[1].set_xlabel("R = √escorrentía  (√mm)"); ax[1].set_ylabel("dR/dt  (√mm / mes)")
    # (R,dR/dt) diario — RUIDOSO
    ax[2].scatter(Rd, dRd, s=4, c="#C0392B", alpha=0.35, linewidths=0)
    ax[2].axhline(0, color="#e2e8ee", lw=0.8)
    ax[2].set_title("3 · DIARIO diferencial (R, dR/dt)\nRUIDOSO — por esto el 159 fallaba", fontsize=10)
    ax[2].set_xlabel("R = √Q  (√(m³/s))"); ax[2].set_ylabel("dR/dt  (√(m³/s) / día)")
    # ventanas-PCA diario — LIMPIO
    ax[3].scatter(Epca[~flood, 0], Epca[~flood, 1], s=5, c="#4a90d9", alpha=0.3, linewidths=0)
    ax[3].scatter(Epca[flood, 0], Epca[flood, 1], s=12, c="#cc2222", alpha=0.7, linewidths=0)
    ax[3].set_title("4 · DIARIO ventanas-60d → PCA (fig25)\nLIMPIO — criterio correcto a diario", fontsize=10)
    ax[3].set_xlabel("PC1"); ax[3].set_ylabel("PC2"); ax[3].set_xticks([]); ax[3].set_yticks([])
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.04); cb.set_label("mes (2015→2025 barrido temporal)", fontsize=7)
    fig.suptitle("Reconstrucción del atractor: mensual vs diario — Chancay-Huaral (B23)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura3_reconstruccion_mensual_diaria.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura3_reconstruccion_mensual_diaria.png'}")


if __name__ == "__main__":
    main()
