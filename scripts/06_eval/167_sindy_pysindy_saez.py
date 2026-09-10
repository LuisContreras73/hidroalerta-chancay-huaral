#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 167 — SINDy con PySINDy (E-SINDy) + validación por integración + comparación con
el global modelling de Saéz et al. Mejora del prototipo artesanal (164).

QUÉ CAMBIA vs 164:
  · Usa el paquete estándar **PySINDy 2.1.0** (no artesanal).
  · **E-SINDy** (Ensemble-SINDy, Fasel et al. 2022, Proc. R. Soc. A 478:20210904): bagging
    sobre subconjuntos → robustez en poco-dato/mucho-ruido + **UQ** (probabilidad de inclusión
    de cada término).
  · Forzante lluvia como **control** (SINDy-c) — sistema no autónomo, como el paper.
  · **VALIDACIÓN POR INTEGRACIÓN**: se integra la ODE descubierta (forzada por la lluvia) y se
    compara la trayectoria y el retrato de fase con lo observado — el test honesto (lo que hace
    Saéz), no solo el R² de la derivada in-sample.

MODELO: dR/dt = Θ(R, P)·Ξ, R=√Q, P=√precip, biblioteca polinómica grado 3 (misma que Saéz:
RP², R²P, R², R³, …). Datos: tramo de AFORO REAL contiguo más largo (script 163).

COMPARACIÓN CON SAÉZ (Saéz et al. 2026, HP 40:e70582, cuenca B23):
  método (GPoM polinómico ↔ E-SINDy), escala (mensual ↔ diaria real), términos recuperados,
  validación por integración, y UQ (ellos no reportan; E-SINDy sí).

Salidas: dinamica_no_lineal/figuras/figura9_sindy_pysindy.png + .../resultados/167_sindy_pysindy.csv
Run en .venv313: python scripts/06_eval/167_sindy_pysindy_saez.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pysindy167")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R
SAEZ = {"R^3", "R^2 P", "R P^2", "R^2"}      # términos del paper (notación PySINDy)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14); dts = df.index
    obs = df["obs"].values.astype(float); pr = df["pr"].values.astype(float)
    finite = np.isfinite(obs)
    # tramo real contiguo más largo (= 163/164)
    bs = be = s = 0
    for i in range(1, len(finite) + 1):
        if i < len(finite) and finite[i] and finite[i - 1]:
            continue
        if i - s > be - bs:
            bs, be = s, i
        s = i
    Rr = np.sqrt(np.clip(obs[bs:be], 0, None))
    Pr = np.sqrt(np.clip(pr[bs:be], 0, None))
    n = len(Rr); t = np.arange(n, dtype=float)
    X = Rr.reshape(-1, 1); U = Pr.reshape(-1, 1)
    log.info(f"tramo real: {dts[bs].date()}→{dts[be-1].date()} ({n} d)")

    lib = ps.PolynomialLibrary(degree=3)
    diff = ps.SmoothedFiniteDifference()          # derivada suavizada (menos ruido)

    # ── (1) STLSQ base ──
    m1 = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=0.02),
                  differentiation_method=diff)
    m1.fit(X, t=1.0, u=U, feature_names=["R", "P"])
    feats = m1.get_feature_names()
    eq1 = m1.equations(precision=4)[0]
    r2_deriv = m1.score(X, t=1.0, u=U)
    log.info(f"[STLSQ] dR/dt = {eq1}")
    log.info(f"[STLSQ] R² derivada = {r2_deriv:.3f}")

    # ── (2) E-SINDy (ensemble) → probabilidad de inclusión + UQ ──
    ens = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.02), bagging=True, n_models=200)
    m2 = ps.SINDy(feature_library=lib, optimizer=ens, differentiation_method=diff)
    m2.fit(X, t=1.0, u=U, feature_names=["R", "P"])
    coefs = np.array(ens.coef_list)[:, 0, :]      # (n_models, n_feats)
    incl = np.mean(np.abs(coefs) > 1e-10, axis=0)
    med = np.median(coefs, axis=0)
    iqr = np.percentile(coefs, 75, axis=0) - np.percentile(coefs, 25, axis=0)
    top = sorted([(feats[i], incl[i], med[i]) for i in range(len(feats)) if incl[i] > 0.2],
                 key=lambda z: -z[1])
    log.info("[E-SINDy] términos robustos (incl>0.2): " +
             ", ".join(f"{f} p={p:.2f} c={c:+.4g}" for f, p, c in top))

    # ── (3) VALIDACIÓN POR INTEGRACIÓN ──
    traj_r2 = np.nan; Rsim = None
    try:
        Rsim = m1.simulate(X[0], t, u=U).ravel()
        L = min(len(Rsim), n)
        if np.all(np.isfinite(Rsim[:L])):
            rr, rs = Rr[:L], Rsim[:L]
            traj_r2 = 1 - np.sum((rr - rs) ** 2) / np.sum((rr - rr.mean()) ** 2)
        log.info(f"[integración] trayectoria R² = {traj_r2:.3f} "
                 f"({'estable' if np.all(np.isfinite(Rsim)) else 'DIVERGE'})")
    except Exception as e:
        log.info(f"[integración] falló/diverge: {e}")

    terms_found = {f.replace(" ", "").replace("^", "^") for f, p, _ in top}
    saez_norm = {s.replace(" ", "") for s in SAEZ}
    coincide = sorted({f for f, p, _ in top} & {"R^3", "R^2 P", "R P^2", "R^2"})
    pd.DataFrame([dict(tramo=f"{dts[bs].date()}..{dts[be-1].date()}", n=n,
                       ecuacion_stlsq=eq1, R2_derivada=round(r2_deriv, 3),
                       R2_trayectoria_integrada=round(traj_r2, 3) if np.isfinite(traj_r2) else "diverge",
                       terminos_robustos=";".join(f"{f}({p:.2f})" for f, p, _ in top),
                       coinciden_saez=";".join(coincide))]).to_csv(RESDIR / "167_sindy_pysindy.csv", index=False)

    # ── figura ──
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), dpi=135)
    order = np.argsort(-incl)
    fn = [feats[i] for i in order]; pv = incl[order]
    cols = ["#C0392B" if feats[i] in ("R^3", "R^2 P", "R P^2", "R^2") else "#0B6E8C" for i in order]
    ax[0].barh(range(len(fn)), pv, color=cols); ax[0].set_yticks(range(len(fn))); ax[0].set_yticklabels(fn, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel("prob. de inclusión (E-SINDy)")
    ax[0].set_title("A · Términos robustos (rojo = términos de Saéz)", fontsize=10)
    ax[1].plot(t, Rr, color="#0A3D54", lw=1.1, label="observado")
    if Rsim is not None and np.all(np.isfinite(Rsim)):
        ax[1].plot(t[:len(Rsim)], Rsim, color="#C0392B", lw=1.0, alpha=0.8, label="ODE integrada")
    ax[1].set_title(f"B · Validación por integración (R² traj = {traj_r2:.2f})", fontsize=10)
    ax[1].set_xlabel("día"); ax[1].set_ylabel("R = √Q"); ax[1].legend(fontsize=8)
    dRo = savgol_filter(Rr, 15, 3, deriv=1)
    ax[2].scatter(Rr, dRo, s=7, c="#0A3D54", alpha=0.5, linewidths=0, label="observado")
    if Rsim is not None and np.all(np.isfinite(Rsim)):
        dRs = savgol_filter(Rsim, 15, 3, deriv=1)
        ax[2].scatter(Rsim, dRs, s=7, c="#C0392B", alpha=0.5, linewidths=0, label="ODE integrada")
    ax[2].axhline(0, color="#e2e8ee", lw=0.8)
    ax[2].set_title("C · Retrato de fase (R, dR/dt): ¿reproduce la órbita?", fontsize=10)
    ax[2].set_xlabel("R = √Q"); ax[2].set_ylabel("dR/dt"); ax[2].legend(fontsize=8)
    fig.suptitle("SINDy con PySINDy (E-SINDy) vs global modelling de Saéz — aforo real, Chancay-Huaral (B23)",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura9_sindy_pysindy.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura9_sindy_pysindy.png'}")


if __name__ == "__main__":
    main()
