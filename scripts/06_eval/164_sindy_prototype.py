#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 164 — Prototipo SINDy (Sparse Identification of Nonlinear Dynamics) sobre el
caudal, como versión ML-moderna del global modelling de Saéz et al. (prueba de concepto
para el 2º paper).

IDEA. Saéz et al. ajustan una ODE polinómica por mínimos cuadrados + búsqueda de estructura
(GPoM). SINDy (Brunton, Proctor & Kutz 2016, PNAS 113:3932) hace lo MISMO con el estándar
ML: dado el estado y su derivada, ajusta d/dt = Θ(estado)·Ξ con Ξ DISPERSA vía STLSQ
(Sequentially Thresholded Least Squares). Con forzamiento (lluvia) = SINDy-c (Brunton,
Proctor & Kutz 2016, IFAC 49:710).

MODELO (1er orden forzado, tipo reservorio no lineal):
    dR/dt = Θ(R, P) · Ξ,   R=√Q, P=√precip (mismas variables que el paper)
  Biblioteca Θ (polinomios hasta grado 3): {1, R, P, R², RP, P², R³, R²P, RP², P³}.
  → comparamos qué términos sobreviven vs los del paper (RP², R²P, R², R³).

DATOS: tramo de AFORO REAL contiguo más largo (script 163: 2022-07→2023-12, honesto, sin
relleno). Derivada dR/dt por Savitzky-Golay (ventana 15, orden 3). STLSQ con umbral λ.

MÉTRICA: R² del ajuste de la derivada (in-sample) + la ecuación dispersa recuperada +
coincidencia de términos con Saéz et al. NO se integra la ODE aquí (validación por
integración = trabajo del paper; se declara).

VARIANTES MODERNAS a citar en la propuesta (no implementadas aquí): E-SINDy (ensemble,
Fasel et al. 2022) para robustez/UQ; Weak/Integral SINDy (Messenger & Bortz 2021) para
ruido; SINDy-PI (Kaheman et al. 2020) para dinámica implícita/racional.

Salidas: dinamica_no_lineal/figuras/figura6_sindy_aforo_real.png + .../resultados/164_sindy.csv
Run en .venv313: python scripts/06_eval/164_sindy_prototype.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sindy164")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R

TERMS = [("1", lambda r, p: np.ones_like(r)), ("R", lambda r, p: r), ("P", lambda r, p: p),
         ("R^2", lambda r, p: r**2), ("R*P", lambda r, p: r*p), ("P^2", lambda r, p: p**2),
         ("R^3", lambda r, p: r**3), ("R^2*P", lambda r, p: r**2*p), ("R*P^2", lambda r, p: r*p**2),
         ("P^3", lambda r, p: p**3)]
SAEZ_TERMS = {"R*P^2", "R^2*P", "R^2", "R^3"}      # términos que reporta el paper


def stlsq(Theta, y, lam=0.05, iters=15):
    """Sequentially Thresholded Least Squares (Brunton et al. 2016).
    Columnas normalizadas para umbralar de forma comparable; se re-escala al final."""
    cs = Theta.std(0) + 1e-12
    Tn = Theta / cs
    Xi = np.linalg.lstsq(Tn, y, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        big = ~small
        if big.any():
            Xi[big] = np.linalg.lstsq(Tn[:, big], y, rcond=None)[0]
    return Xi / cs                                  # coeficientes en escala original


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14); dts = df.index
    obs = df["obs"].values.astype(float); pr = df["pr"].values.astype(float)
    finite = np.isfinite(obs)
    # tramo real contiguo más largo (= script 163)
    bs = be = s = 0
    for i in range(1, len(finite) + 1):
        if i < len(finite) and finite[i] and finite[i - 1]:
            continue
        if i - s > be - bs:
            bs, be = s, i
        s = i
    log.info(f"tramo real: {dts[bs].date()}→{dts[be-1].date()} ({be-bs} d)")

    Rr = np.sqrt(np.clip(obs[bs:be], 0, None))
    Pr = np.sqrt(np.clip(pr[bs:be], 0, None))
    dR = savgol_filter(Rr, 15, 3, deriv=1)          # dR/dt (por día)
    Theta = np.column_stack([f(Rr, Pr) for _, f in TERMS])

    # barrido de umbral λ: elegir el más disperso con R²≥0.9·R²(full)
    r2_full = None; best = None
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        Xi = stlsq(Theta, dR, lam=lam)
        pred = Theta @ Xi
        ss = 1 - np.sum((dR - pred)**2) / np.sum((dR - dR.mean())**2)
        nnz = int((Xi != 0).sum())
        if lam == 0.0:
            r2_full = ss
        log.info(f"  λ={lam}: R²={ss:.3f}, términos={nnz}")
        if ss >= 0.9 * r2_full and (best is None or nnz < best[2]):
            best = (lam, Xi, nnz, ss)
    lam, Xi, nnz, ss = best
    eq = " + ".join(f"{Xi[i]:+.3g}·{TERMS[i][0]}" for i in range(len(TERMS)) if Xi[i] != 0)
    activos = {TERMS[i][0] for i in range(len(TERMS)) if Xi[i] != 0}
    coincide = activos & SAEZ_TERMS
    log.info(f"ELEGIDO λ={lam}: {nnz} términos, R²={ss:.3f}")
    log.info(f"  dR/dt = {eq}")
    log.info(f"  términos activos: {sorted(activos)}")
    log.info(f"  coinciden con Saéz {sorted(SAEZ_TERMS)} → {sorted(coincide)}")

    pd.DataFrame([dict(tramo=f"{dts[bs].date()}..{dts[be-1].date()}", n=int(be-bs),
                       lambda_=lam, n_terminos=nnz, R2=round(ss, 3),
                       ecuacion=eq, terminos_activos=";".join(sorted(activos)),
                       coinciden_saez=";".join(sorted(coincide)))]).to_csv(RESDIR / "164_sindy.csv", index=False)

    # figura: dR/dt real vs SINDy + ecuación
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    pred = Theta @ Xi
    ax[0].plot(dts[bs:be], dR, color="#0A3D54", lw=1.1, label="dR/dt observado (Savitzky-Golay)")
    ax[0].plot(dts[bs:be], pred, color="#C0392B", lw=1.0, alpha=0.8, label=f"SINDy ({nnz} términos)")
    ax[0].set_title(f"A · Ajuste de la derivada (aforo real)\nR²={ss:.3f}, λ={lam}", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].set_ylabel("dR/dt (√(m³/s)/día)"); ax[0].tick_params(labelsize=8)
    ax[1].scatter(dR, pred, s=8, c="#0B6E8C", alpha=0.5, linewidths=0)
    lims = [min(dR.min(), pred.min()), max(dR.max(), pred.max())]
    ax[1].plot(lims, lims, "--", color="#888", lw=0.8)
    ax[1].set_title("B · SINDy vs observado (1:1)", fontsize=10)
    ax[1].set_xlabel("dR/dt observado"); ax[1].set_ylabel("dR/dt SINDy"); ax[1].tick_params(labelsize=8)
    fig.suptitle(f"SINDy sobre aforo real — dR/dt = {eq}", fontsize=10.5, y=1.06)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura6_sindy_aforo_real.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura6_sindy_aforo_real.png'}")


if __name__ == "__main__":
    main()
