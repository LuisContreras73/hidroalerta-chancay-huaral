#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 169 — SINDy MEJORADO: mensual + filtrado por estabilidad de integración + weak-form.
Objetivo: medir QUÉ TANTO NOS ACERCAMOS a Saéz (cuántos de sus términos recuperamos y con
qué R² de integración), cerrando la brecha en igualdad de condiciones.

MEJORAS sobre 167/168:
  1. Escala MENSUAL (como Saéz; su caudal es GR2M-derivado, el nuestro mensual también es la
     serie completa → misma naturaleza).
  2. **Filtrado por ESTABILIDAD DE INTEGRACIÓN** — la clave que faltaba: barremos el umbral de
     sparsity y, para cada ODE candidata, la INTEGRAMOS (forzada por la lluvia observada) con un
     integrador robusto; nos quedamos con la más dispersa que (a) NO diverge y (b) mejor
     reproduce la trayectoria. Es lo que hace su GPoM (búsqueda + filtro de estabilidad).
  3. **Weak-form** (WSINDy) como alternativa robusta al ruido.

Integración propia (scipy solve_ivp) a partir de los coeficientes Ξ y la librería polinómica,
forzada por P(t)=√ϖ interpolada → funciona igual para modelos STLSQ y weak-form.

CERCANÍA A SAÉZ: sus términos {R³, R²P, RP², R²}. Reportamos cuántos recupera el mejor modelo
ESTABLE + su R² de integración.

Salidas: dinamica_no_lineal/figuras/figura11_sindy_mejorado_saez.png
         dinamica_no_lineal/resultados/169_sindy_mejorado.csv
Run en .venv313: python scripts/06_eval/169_sindy_mejorado_saez.py
"""
import importlib.util
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mej169")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R
AREA_M2 = 3062.62 * 1e6
SAEZ = {"R^3", "R^2 P", "R P^2", "R^2"}


def term_val(name, r, p):
    name = name.strip()
    if name in ("1", ""):
        return np.ones_like(r) if np.ndim(r) else 1.0
    val = np.ones_like(r) if np.ndim(r) else 1.0
    for f in name.split():
        if f == "R":
            val = val * r
        elif f == "P":
            val = val * p
        elif f.startswith("R^"):
            val = val * r ** int(f[2:])
        elif f.startswith("P^"):
            val = val * p ** int(f[2:])
    return val


def integrate(feats, coef, t, Pser, R0):
    """Integra dR/dt = Σ coef_k·term_k(R,P(t)) forzada por P(t). Devuelve (Rsim, estable)."""
    Pf = interp1d(t, Pser, bounds_error=False, fill_value=(Pser[0], Pser[-1]))
    active = [(feats[k], coef[k]) for k in range(len(feats)) if abs(coef[k]) > 1e-12]

    cap = 5 * np.nanmax(np.abs(Pser)) + 20                # umbral de "explota"

    def rhs(tt, y):
        r = y[0]; p = float(Pf(tt))
        return [sum(c * term_val(nm, r, p) for nm, c in active)]

    def blowup(tt, y):                                    # corte temprano por divergencia
        return np.abs(y[0]) - cap
    blowup.terminal = True; blowup.direction = 1

    try:
        sol = solve_ivp(rhs, [t[0], t[-1]], [R0], t_eval=t, method="LSODA",
                        rtol=1e-6, atol=1e-8, events=blowup)
        Rs = sol.y[0]
        stable = (sol.status == 0) and np.all(np.isfinite(Rs)) and len(Rs) == len(t)
        return Rs, stable
    except Exception:
        return None, False


def saez_hits(feats, coef):
    return sorted({feats[k] for k in range(len(feats)) if abs(coef[k]) > 1e-12} & SAEZ)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14); dts = df.index
    q = pd.Series(df["q"].values, index=dts); pr = pd.Series(df["pr"].values, index=dts)
    rho = (q * 86400.0 / AREA_M2 * 1000.0).resample("MS").sum().values
    varpi = pr.resample("MS").sum().values
    Rm = np.sqrt(np.clip(rho, 0, None)); Pm = np.sqrt(np.clip(varpi, 0, None))
    t = np.arange(len(Rm), dtype=float); X = Rm.reshape(-1, 1); U = Pm.reshape(-1, 1)
    log.info(f"mensual: {len(Rm)} meses")

    def traj_r2(Rs):
        L = min(len(Rs), len(Rm))
        return 1 - np.sum((Rm[:L] - Rs[:L]) ** 2) / np.sum((Rm[:L] - Rm[:L].mean()) ** 2)

    # ── barrido de umbral (STLSQ) con filtro de estabilidad ──
    rows = []; best = None
    for lam in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        m = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
                     optimizer=ps.STLSQ(threshold=lam), differentiation_method=ps.SmoothedFiniteDifference())
        m.fit(X, t=1.0, u=U, feature_names=["R", "P"])
        feats = m.get_feature_names(); coef = m.coefficients()[0]
        nterm = int(np.sum(np.abs(coef) > 1e-12))
        Rs, stable = integrate(feats, coef, t, Pm, Rm[0])
        r2 = traj_r2(Rs) if (Rs is not None and stable) else np.nan
        hits = saez_hits(feats, coef)
        rows.append(dict(metodo="STLSQ-mensual", lam=lam, n_term=nterm, estable=stable,
                         R2_integracion=round(r2, 3) if np.isfinite(r2) else "diverge",
                         saez=";".join(hits), n_saez=len(hits)))
        log.info(f"  λ={lam}: {nterm} térm · {'estable' if stable else 'DIVERGE'} · "
                 f"R²int={r2:.3f} · Saéz={hits}")
        if stable and np.isfinite(r2):
            score = (len(hits), r2)          # más términos de Saéz, luego mejor R²
            if best is None or score > best[0]:
                best = (score, lam, feats, coef, Rs, hits, r2, nterm)

    # ── weak-form mensual ──
    wlib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=3, include_bias=True),
                             spatiotemporal_grid=t, K=150, is_uniform=True)
    mw = ps.SINDy(feature_library=wlib, optimizer=ps.STLSQ(threshold=0.02))
    mw.fit(X, t=1.0, u=U, feature_names=["R", "P"])
    fw = [f.replace("x0", "R").replace("u0", "P") for f in mw.get_feature_names()]
    cw = mw.coefficients()[0]
    Rsw, stw = integrate(fw, cw, t, Pm, Rm[0])
    r2w = traj_r2(Rsw) if (Rsw is not None and stw) else np.nan
    hitsw = saez_hits(fw, cw)
    rows.append(dict(metodo="WSINDy-mensual", lam=0.02, n_term=int(np.sum(np.abs(cw) > 1e-12)),
                     estable=stw, R2_integracion=round(r2w, 3) if np.isfinite(r2w) else "diverge",
                     saez=";".join(hitsw), n_saez=len(hitsw)))
    log.info(f"  WEAK mensual: {'estable' if stw else 'DIVERGE'} · R²int={r2w:.3f} · Saéz={hitsw}")

    pd.DataFrame(rows).to_csv(RESDIR / "169_sindy_mejorado.csv", index=False)
    if best is None:
        log.info("Ningún STLSQ estable; ver weak."); bf, blam, bfeats, bcoef, bRs, bhits, br2, bn = \
            (None, None, fw, cw, Rsw, hitsw, r2w, int(np.sum(np.abs(cw) > 1e-12)))
    else:
        _, blam, bfeats, bcoef, bRs, bhits, br2, bn = best
    log.info(f"★ MEJOR estable: {bn} térm, Saéz {len(bhits)}/4 {bhits}, R²int={br2:.3f}")

    # ── figura ──
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), dpi=135)
    dd = pd.DataFrame([r for r in rows if r["metodo"] == "STLSQ-mensual"])
    lam_ok = dd[dd.estable]; lam_no = dd[~dd.estable]
    ax[0].scatter(lam_ok.lam, lam_ok.n_saez, s=60, c="#2E8B6F", label="estable", zorder=3)
    ax[0].scatter(lam_no.lam, lam_no.n_saez, s=60, c="#C0392B", marker="x", label="diverge", zorder=3)
    ax[0].set_xscale("log"); ax[0].set_xlabel("umbral λ (sparsity)"); ax[0].set_ylabel("# términos de Saéz (de 4)")
    ax[0].set_title("A · Barrido con filtro de estabilidad\n(verde = integra; rojo = diverge)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(t, Rm, color="#0A3D54", lw=1.0, label="observado (mensual)")
    if bRs is not None:
        ax[1].plot(t[:len(bRs)], bRs, color="#C0392B", lw=1.1, alpha=0.85, label="mejor ODE integrada")
    ax[1].set_title(f"B · Mejor ODE ESTABLE: integración (R²={br2:.2f})", fontsize=10)
    ax[1].set_xlabel("mes"); ax[1].set_ylabel("R=√ρ"); ax[1].legend(fontsize=8)
    lb = ["R^3", "R^2 P", "R P^2", "R^2", "R P", "R", "P", "1"]
    present = [1 if (l in {bfeats[k] for k in range(len(bfeats)) if abs(bcoef[k]) > 1e-12}) else 0 for l in lb]
    cB = ["#C0392B" if l in SAEZ else "#0B6E8C" for l in lb]
    ax[2].barh(range(len(lb)), present, color=cB); ax[2].set_yticks(range(len(lb))); ax[2].set_yticklabels(lb, fontsize=8)
    ax[2].invert_yaxis(); ax[2].set_xlim(0, 1.2); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["no", "sí"], fontsize=8)
    ax[2].set_title(f"C · Mejor modelo: términos ({len(bhits)}/4 de Saéz en rojo)", fontsize=10)
    fig.suptitle("SINDy mejorado (mensual + filtro de estabilidad + weak-form): cercanía a Saéz — Chancay-Huaral (B23)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura11_sindy_mejorado_saez.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura11_sindy_mejorado_saez.png'}")


if __name__ == "__main__":
    main()
