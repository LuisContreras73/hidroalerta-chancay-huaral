#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 168 — Cerrar la brecha con Saéz EN IGUALDAD DE CONDICIONES: (A) SINDy MENSUAL
(misma escala que el paper) y (B) WSINDy (weak-form, robusto al ruido) a diario.

CONTEXTO. El 167 mostró que a DIARIO ruidoso la estructura exacta no es identificable
(E-SINDy: solo el "esqueleto forzado" de bajo orden es robusto; R³/R²P no). Dos formas
de cerrar la brecha con Saéz, en igualdad de condiciones:
  (A) MENSUAL, como ellos: caudal → ρ (mm/mes), R=√ρ, P=√ϖ. Igualdad EXTRA de condiciones:
      su caudal es GR2M-derivado (no aforo puro); nuestro caudal mensual también es la serie
      completa (rellenada) → ambos son mensuales y model-derived. ¿Emergen R³/R²P al ser suave?
  (B) WSINDy (weak/integral form, Messenger & Bortz 2021; PySINDy WeakPDELibrary): integra
      contra funciones test en vez de derivar → robusto al ruido. ¿Recupera estructura limpia
      a DIARIO sin irse a mensual?

Ambos con E-SINDy/UQ donde aplica + validación por integración. Comparación con los términos
de Saéz {R³, R²P, RP², R²}.

Salidas: dinamica_no_lineal/figuras/figura10_sindy_mensual_wsindy.png
         dinamica_no_lineal/resultados/168_sindy_mensual_wsindy.csv
Run en .venv313: python scripts/06_eval/168_sindy_mensual_wsindy.py
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
log = logging.getLogger("mw168")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R
AREA_M2 = 3062.62 * 1e6
SAEZ = {"R^3", "R^2 P", "R P^2", "R^2"}


def esindy_incl(X, U, thr, feats_expected):
    ens = ps.EnsembleOptimizer(ps.STLSQ(threshold=thr), bagging=True, n_models=200)
    m = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
                 optimizer=ens, differentiation_method=ps.SmoothedFiniteDifference())
    m.fit(X, t=1.0, u=U, feature_names=["R", "P"])
    feats = m.get_feature_names()
    coefs = np.array(ens.coef_list)[:, 0, :]
    incl = np.mean(np.abs(coefs) > 1e-10, axis=0)
    return m, feats, incl


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14); dts = df.index
    q = pd.Series(df["q"].values, index=dts); pr = pd.Series(df["pr"].values, index=dts)
    obs = df["obs"].values.astype(float); finite = np.isfinite(obs)
    rows = []

    # ─────────── (A) MENSUAL (como Saéz) ───────────
    rho = (q * 86400.0 / AREA_M2 * 1000.0).resample("MS").sum().values     # mm/mes
    varpi = pr.resample("MS").sum().values
    Rm = np.sqrt(np.clip(rho, 0, None)); Pm = np.sqrt(np.clip(varpi, 0, None))
    tm = np.arange(len(Rm), dtype=float)
    Xm = Rm.reshape(-1, 1); Um = Pm.reshape(-1, 1)
    log.info(f"(A) MENSUAL: {len(Rm)} meses (1981-2025)")
    m_A = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
                   optimizer=ps.STLSQ(threshold=0.02), differentiation_method=ps.SmoothedFiniteDifference())
    m_A.fit(Xm, t=1.0, u=Um, feature_names=["R", "P"])
    eqA = m_A.equations(precision=4)[0]; r2A = m_A.score(Xm, t=1.0, u=Um)
    _, featsA, inclA = esindy_incl(Xm, Um, 0.02, None)
    robustA = sorted([featsA[i] for i in range(len(featsA)) if inclA[i] > 0.5])
    saezA = sorted({featsA[i] for i in range(len(featsA)) if inclA[i] > 0.5} & SAEZ)
    trajA = np.nan
    try:
        Rs = m_A.simulate(Xm[0], tm, u=Um).ravel(); L = min(len(Rs), len(Rm))
        if np.all(np.isfinite(Rs[:L])):
            trajA = 1 - np.sum((Rm[:L] - Rs[:L]) ** 2) / np.sum((Rm[:L] - Rm[:L].mean()) ** 2)
    except Exception:
        Rs = None
    log.info(f"(A) STLSQ dR/dt = {eqA}")
    log.info(f"(A) R²deriv={r2A:.3f} · integración R²={trajA:.3f} · robustos(incl>0.5)={robustA} · coinciden Saéz={saezA}")
    rows.append(dict(caso="mensual (Saéz-cond)", n=len(Rm), R2_deriv=round(r2A, 3),
                     R2_integracion=round(trajA, 3) if np.isfinite(trajA) else "diverge",
                     robustos=";".join(robustA), coinciden_saez=";".join(saezA), ecuacion=eqA))

    # ─────────── (B) WSINDy weak-form a DIARIO (aforo real) ───────────
    bs = be = s = 0
    for i in range(1, len(finite) + 1):
        if i < len(finite) and finite[i] and finite[i - 1]:
            continue
        if i - s > be - bs:
            bs, be = s, i
        s = i
    Rd = np.sqrt(np.clip(obs[bs:be], 0, None)); Pd = np.sqrt(np.clip(pr.values[bs:be], 0, None))
    td = np.arange(len(Rd), dtype=float); Xd = Rd.reshape(-1, 1); Ud = Pd.reshape(-1, 1)
    log.info(f"(B) WSINDy diario real: {len(Rd)} d")
    wlib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=3, include_bias=True),
                             spatiotemporal_grid=td, K=200, is_uniform=True)
    m_B = ps.SINDy(feature_library=wlib, optimizer=ps.STLSQ(threshold=0.02))
    m_B.fit(Xd, t=1.0, u=Ud, feature_names=["R", "P"])
    eqB = m_B.equations(precision=4)[0]
    featsB = m_B.get_feature_names()
    coefB = m_B.coefficients()[0]
    activosB = {featsB[i].replace("x0", "R").replace("u0", "P") for i in range(len(featsB)) if abs(coefB[i]) > 1e-10}
    saezB = sorted(activosB & SAEZ)
    log.info(f"(B) WSINDy dR/dt = {eqB}")
    log.info(f"(B) términos activos={sorted(activosB)} · coinciden Saéz={saezB}")
    rows.append(dict(caso="WSINDy diario real", n=len(Rd), R2_deriv="(weak, n/a)",
                     R2_integracion="", robustos=";".join(sorted(activosB)),
                     coinciden_saez=";".join(saezB), ecuacion=eqB))

    pd.DataFrame(rows).to_csv(RESDIR / "168_sindy_mensual_wsindy.csv", index=False)

    # ─────────── figura ───────────
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), dpi=135)
    order = np.argsort(-inclA)
    fn = [featsA[i] for i in order]; pv = inclA[order]
    cols = ["#C0392B" if featsA[i] in SAEZ else "#0B6E8C" for i in order]
    ax[0].barh(range(len(fn)), pv, color=cols); ax[0].set_yticks(range(len(fn))); ax[0].set_yticklabels(fn, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel("prob. inclusión (E-SINDy)")
    ax[0].set_title("A · MENSUAL (como Saéz): términos robustos\nrojo = términos de Saéz", fontsize=10)
    ax[1].plot(tm, Rm, color="#0A3D54", lw=1.0, label="observado (mensual)")
    if Rs is not None and np.all(np.isfinite(Rs)):
        ax[1].plot(tm[:len(Rs)], Rs, color="#C0392B", lw=1.0, alpha=0.8, label="ODE integrada")
    ax[1].set_title(f"B · MENSUAL: validación por integración (R²={trajA:.2f})", fontsize=10)
    ax[1].set_xlabel("mes"); ax[1].set_ylabel("R=√ρ"); ax[1].legend(fontsize=8)
    yb = [1 if f in activosB else 0 for f in ["R^3", "R^2 P", "R P^2", "R^2", "R P", "R", "P", "1"]]
    lb = ["R^3", "R^2 P", "R P^2", "R^2", "R P", "R", "P", "1"]
    cB = ["#C0392B" if l in SAEZ else "#0B6E8C" for l in lb]
    ax[2].barh(range(len(lb)), yb, color=cB); ax[2].set_yticks(range(len(lb))); ax[2].set_yticklabels(lb, fontsize=8)
    ax[2].invert_yaxis(); ax[2].set_xlim(0, 1.2); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["ausente", "presente"], fontsize=8)
    ax[2].set_title("C · WSINDy diario: términos activos\nrojo = términos de Saéz", fontsize=10)
    fig.suptitle("SINDy en igualdad de condiciones: mensual (como Saéz) y weak-form (WSINDy) — Chancay-Huaral (B23)",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura10_sindy_mensual_wsindy.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura10_sindy_mensual_wsindy.png'}")


if __name__ == "__main__":
    main()
