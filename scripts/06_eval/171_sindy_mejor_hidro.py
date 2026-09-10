#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 171 — El "punto dulce" de SINDy para hidrología: WEAK-FORM + CONTROL ESTACIONAL.
Combina lo mejor que hallamos (167-170): weak-form (robusto al ruido, recupera la estructura
no lineal de Saéz) + forzante estacional sin/cos anual (el ciclo anual, físicamente real).

Config: WeakPDELibrary(PolynomialLibrary(degree=3)) con control u=[P, sin(2πt/365), cos(2πt/365)].
STLSQ + E-SINDy (UQ). Reporta la matriz sparsa, la ecuación, qué términos de Saéz {R³,R²P,RP²,R²}
recupera, qué términos estacionales, y la robustez E-SINDy de cada uno.

Cierra la exploración SINDy de la línea de dinámica no lineal (2º paper). El salto de calidad
siguiente NO es más SINDy: es SINDy con restricción de estabilidad (TrappingSR3, requiere cvxpy)
o el Neural ODE. Ver PROPUESTA_PAPER2_NEURAL_ODE §próximos pasos.

Salidas: dinamica_no_lineal/figuras/figura13_sindy_mejor_hidro.png
         dinamica_no_lineal/resultados/171_sindy_mejor.csv
Run en .venv313: python scripts/06_eval/171_sindy_mejor_hidro.py
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
log = logging.getLogger("mejor171")
SAEZ = {"R^3", "R^2 P", "R P^2", "R^2"}

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R


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
    n = len(Rr); t = np.arange(n, dtype=float); X = Rr.reshape(-1, 1)
    sA = np.sin(2 * np.pi * t / 365.25); cA = np.cos(2 * np.pi * t / 365.25)
    Uc = np.column_stack([Pr, sA, cA])
    names = ["R", "P", "sA", "cA"]
    log.info(f"aforo real {n} d · config: weak-form + control estacional")

    def build():
        return ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=3, include_bias=True),
                                 spatiotemporal_grid=t, K=200, is_uniform=True)

    m = ps.SINDy(feature_library=build(), optimizer=ps.STLSQ(threshold=0.03))
    m.fit(X, t=1.0, u=Uc, feature_names=names)
    feats = [f.replace("x0", "R").replace("u0", "P").replace("u1", "sA").replace("u2", "cA")
             for f in m.get_feature_names()]
    coef = m.coefficients()[0]

    ens = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.03), bagging=True, n_models=150)
    me = ps.SINDy(feature_library=build(), optimizer=ens)
    me.fit(X, t=1.0, u=Uc, feature_names=names)
    incl = np.mean(np.abs(np.array(ens.coef_list)[:, 0, :]) > 1e-10, axis=0)

    active = [(feats[k], coef[k], incl[k]) for k in range(len(feats)) if abs(coef[k]) > 1e-10]
    saez_hit = sorted({f for f, c, p in active} & SAEZ)
    seas_hit = sorted({f for f, c, p in active if ("sA" in f or "cA" in f)})
    print("\n═══ MEJOR SINDy hidrología: weak-form + control estacional ═══")
    print(f"  {'término':<10} {'Ξ':>12} {'incl(E-SINDy)':>14}")
    print("  " + "-" * 38)
    for f, c, p in sorted(active, key=lambda z: -z[2]):
        tag = "  (Saéz)" if f in SAEZ else ("  (estac.)" if ("sA" in f or "cA" in f) else "")
        print(f"  {f:<10} {c:>12.4f} {p:>13.2f}{tag}")
    print(f"\n  dR/dt = {m.equations(precision=4)[0].replace('x0','R').replace('u0','P').replace('u1','sA').replace('u2','cA')}")
    print(f"  Términos de Saéz recuperados: {saez_hit}")
    print(f"  Términos estacionales: {seas_hit}\n")
    log.info(f"activos={len(active)} · Saéz={saez_hit} · estacionales={seas_hit}")

    pd.DataFrame([dict(config="weak-form + control estacional", n=n, n_activos=len(active),
                       saez=";".join(saez_hit), estacionales=";".join(seas_hit),
                       ecuacion=m.equations(precision=4)[0])]).to_csv(RESDIR / "171_sindy_mejor.csv", index=False)

    # figura: barras de robustez de los términos activos, coloreadas
    act_sorted = sorted(active, key=lambda z: z[2])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=135)
    cols = ["#C0392B" if f in SAEZ else ("#D68910" if ("sA" in f or "cA" in f) else "#0B6E8C")
            for f, c, p in act_sorted]
    ax.barh(range(len(act_sorted)), [p for f, c, p in act_sorted], color=cols)
    ax.set_yticks(range(len(act_sorted))); ax.set_yticklabels([f for f, c, p in act_sorted], fontsize=9)
    ax.set_xlabel("probabilidad de inclusión (E-SINDy)")
    ax.axvline(0.5, color="#888", lw=0.7, ls=":")
    ax.set_title("Weak-form + control estacional SOBREAJUSTA: 33 términos activos (colinealidad estacional)\n"
                 "todos con inclusión ~1 = no parsimonioso → el usable es WSINDy en [R,P] (fig. 12) — B23", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIGDIR / "figura13_sindy_mejor_hidro.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura13_sindy_mejor_hidro.png'}")


if __name__ == "__main__":
    main()
