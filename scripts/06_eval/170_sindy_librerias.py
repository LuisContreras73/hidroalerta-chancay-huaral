#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 170 — (1) matriz sparsa de WSINDy (weak-form) y (2) exploración de LIBRERÍAS más
grandes (Fourier/senos + forzante estacional): ¿ayudan o sobreajustan?

La librería Θ de SINDy es PERSONALIZABLE. Aquí probamos:
  · base   = PolynomialLibrary(degree=3) sobre [R,P]  (10 candidatas)
  · +Fourier = base ⊕ FourierLibrary (sin/cos de R y P)  — los "senos" pedidos
  · +estacional = base + control estacional [sin(2πt/365), cos(2πt/365)] (el ciclo anual real)
Para cada una: nº de términos activos, R² de la derivada, y —clave— la PROBABILIDAD DE
INCLUSIÓN E-SINDy de los términos NUEVOS: si son robustos → ayudan; si no → sobreajuste.

Referencia hidrología (SINDy/descubrimiento de ecuaciones YA usado en el campo):
  Song et al. (2022) WRR 58:e2022WR031926 (flujo de humedad de suelo, sparse regression);
  descubrimiento de ecuaciones en geociencias, Comm. Earth & Env. (2024). Nuestro nicho
  (ODE de escorrentía de cuenca vs global modelling de Saéz) es relativamente novedoso.

Salidas: dinamica_no_lineal/figuras/figura12_sindy_librerias.png
         dinamica_no_lineal/resultados/170_sindy_librerias.csv
Run en .venv313: python scripts/06_eval/170_sindy_librerias.py
"""
import importlib.util
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("libs170")
SAEZ = {"R^3", "R^2 P", "R P^2", "R^2"}

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R


def esindy_incl(lib, X, U, feats_names, thr=0.02):
    ens = ps.EnsembleOptimizer(ps.STLSQ(threshold=thr), bagging=True, n_models=150)
    m = ps.SINDy(feature_library=lib, optimizer=ens, differentiation_method=ps.SmoothedFiniteDifference())
    m.fit(X, t=1.0, u=U, feature_names=feats_names)
    feats = m.get_feature_names()
    incl = np.mean(np.abs(np.array(ens.coef_list)[:, 0, :]) > 1e-10, axis=0)
    return feats, incl


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14); obs = df["obs"].values.astype(float); pr = df["pr"].values.astype(float)
    fin = np.isfinite(obs); bs = be = s = 0
    for i in range(1, len(fin) + 1):
        if i < len(fin) and fin[i] and fin[i - 1]:
            continue
        if i - s > be - bs:
            bs, be = s, i
        s = i
    Rr = np.sqrt(np.clip(obs[bs:be], 0, None)); Pr = np.sqrt(np.clip(pr[bs:be], 0, None))
    n = len(Rr); t = np.arange(n, dtype=float); X = Rr.reshape(-1, 1); U = Pr.reshape(-1, 1)
    log.info(f"aforo real {n} d")

    # ═══ (1) MATRIZ SPARSA DE WSINDy (weak-form) ═══
    wlib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=3, include_bias=True),
                             spatiotemporal_grid=t, K=200, is_uniform=True)
    mw = ps.SINDy(feature_library=wlib, optimizer=ps.STLSQ(threshold=0.02))
    mw.fit(X, t=1.0, u=U, feature_names=["R", "P"])
    fw = [f.replace("x0", "R").replace("u0", "P") for f in mw.get_feature_names()]
    cw = mw.coefficients()[0]
    print("\n═══ MATRIZ SPARSA Ξ — WSINDy (weak-form) ═══")
    print(f"  {'término':<8} {'Ξ (coef WSINDy)':>16}")
    print("  " + "-" * 26)
    for f, c in zip(fw, cw):
        act = " ◄ ACTIVO" + ("  (Saéz)" if f in SAEZ else "") if abs(c) > 1e-10 else ""
        print(f"  {f:<8} {c:>16.5f}{act}")
    print(f"  Ecuación WSINDy: dR/dt = {mw.equations(precision=5)[0]}")
    print(f"  Coinciden con Saéz: {sorted({f for f, c in zip(fw, cw) if abs(c) > 1e-10} & SAEZ)}\n")

    # ═══ (2) EXPLORACIÓN DE LIBRERÍAS ═══
    rows = []
    # base poly3
    base = ps.PolynomialLibrary(degree=3)
    fB, iB = esindy_incl(base, X, U, ["R", "P"])
    mB = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3), optimizer=ps.STLSQ(threshold=0.02),
                  differentiation_method=ps.SmoothedFiniteDifference()); mB.fit(X, t=1.0, u=U, feature_names=["R", "P"])
    r2B = mB.score(X, t=1.0, u=U)
    rows.append(("base poly3", len(fB), int(np.sum(np.abs(mB.coefficients()[0]) > 1e-10)), r2B, [], ""))

    # +Fourier (senos/cos de R,P)
    lF = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=3), ps.FourierLibrary(n_frequencies=1)])
    fF, iF = esindy_incl(lF, X, U, ["R", "P"])
    mF = ps.SINDy(feature_library=ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=3), ps.FourierLibrary(n_frequencies=1)]),
                  optimizer=ps.STLSQ(threshold=0.02), differentiation_method=ps.SmoothedFiniteDifference())
    mF.fit(X, t=1.0, u=U, feature_names=["R", "P"]); r2F = mF.score(X, t=1.0, u=U)
    new_F = [(fF[k], iF[k]) for k in range(len(fF)) if ("sin" in fF[k] or "cos" in fF[k])]
    rows.append(("+Fourier (sin/cos)", len(fF), int(np.sum(np.abs(mF.coefficients()[0]) > 1e-10)), r2F,
                 new_F, "senos de R,P"))

    # +estacional (control sin/cos anual)
    sinA = np.sin(2 * np.pi * t / 365.25); cosA = np.cos(2 * np.pi * t / 365.25)
    Ue = np.column_stack([Pr, sinA, cosA])
    lE = ps.PolynomialLibrary(degree=2)
    fE, iE = esindy_incl(lE, X, Ue, ["R", "P", "sA", "cA"])
    mE = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), optimizer=ps.STLSQ(threshold=0.02),
                  differentiation_method=ps.SmoothedFiniteDifference()); mE.fit(X, t=1.0, u=Ue, feature_names=["R", "P", "sA", "cA"])
    r2E = mE.score(X, t=1.0, u=Ue)
    new_E = [(fE[k], iE[k]) for k in range(len(fE)) if ("sA" in fE[k] or "cA" in fE[k])]
    rows.append(("+estacional (sin/cos anual)", len(fE), int(np.sum(np.abs(mE.coefficients()[0]) > 1e-10)), r2E,
                 new_E, "ciclo anual"))

    print("═══ EXPLORACIÓN DE LIBRERÍAS (¿ayudan o sobreajustan?) ═══")
    print(f"  {'librería':<28} {'#cand':>6} {'#activos':>9} {'R²deriv':>9}  términos NUEVOS robustos (E-SINDy)")
    for name, ncand, nact, r2, new, _ in rows:
        rob = ", ".join(f"{f}={p:.2f}" for f, p in new if p > 0.3) or "(ninguno robusto)" if new else "—"
        print(f"  {name:<28} {ncand:>6} {nact:>9} {r2:>9.3f}  {rob}")
    pd.DataFrame([dict(libreria=r[0], n_candidatas=r[1], n_activos=r[2], R2_deriv=round(r[3], 3),
                       nuevos_robustos=";".join(f"{f}={p:.2f}" for f, p in r[4] if p > 0.3))
                  for r in rows]).to_csv(RESDIR / "170_sindy_librerias.csv", index=False)

    # figura
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), dpi=135)
    names = [r[0] for r in rows]; r2s = [r[3] for r in rows]
    ax[0].bar(range(len(names)), r2s, color=["#0B6E8C", "#D68910", "#2E8B6F"])
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, fontsize=8, rotation=12)
    ax[0].set_ylabel("R² de la derivada"); ax[0].set_title("A · ¿Mejora el ajuste al agrandar la librería?", fontsize=10)
    allnew = [(nm, p) for r in rows for (nm, p) in r[4]]
    if allnew:
        ax[1].barh(range(len(allnew)), [p for _, p in allnew],
                   color=["#2E8B6F" if p > 0.3 else "#C0392B" for _, p in allnew])
        ax[1].set_yticks(range(len(allnew))); ax[1].set_yticklabels([nm for nm, _ in allnew], fontsize=8)
        ax[1].axvline(0.5, color="#888", lw=0.7, ls=":"); ax[1].invert_yaxis()
    ax[1].set_xlabel("prob. inclusión (E-SINDy)"); ax[1].set_title("B · ¿Los términos NUEVOS son robustos?\n(verde>0.3=algo robusto; rojo=sobreajuste)", fontsize=10)
    fig.suptitle("SINDy: librerías más grandes (Fourier/senos + estacional) — ¿ayudan o sobreajustan? — aforo real (B23)",
                 fontsize=11, y=1.03)
    fig.tight_layout(); fig.savefig(FIGDIR / "figura12_sindy_librerias.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura12_sindy_librerias.png'}")


if __name__ == "__main__":
    main()
