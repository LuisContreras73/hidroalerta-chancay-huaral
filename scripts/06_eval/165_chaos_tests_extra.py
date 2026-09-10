#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 165 — Batería de tests dinámicos COMPLEMENTARIA, orientada a hidrología.
Cierra el capítulo "¿caos o no?" con discriminadores robustos y un clásico hidrológico.

TESTS Y FUENTES (implementados desde cero para verificación):
  [A] Plano COMPLEJIDAD-ENTROPÍA (discriminador caos vs ruido).
      Entropía de permutación H_S (Bandt & Pompe 2002, PRL 88:174102): distribución de
      patrones ordinales de orden d; H_S = S(P)/ln(d!) ∈ [0,1].
      Complejidad estadística C (Rosso et al. 2007, PRL 99:154102; López-Ruiz, Mancini,
      Calbet 1995): C = Q_J[P,P_e]·H_S, con Q_J = disequilibrio por divergencia de
      Jensen-Shannon al uniforme, normalizado. Caos → C alta a H intermedia; ruido → C≈0 a H≈1.
  [B] DFA / exponente de Hurst (Peng et al. 1994, Phys. Rev. E 49:1685; contexto hidrológico
      Hurst 1951; Mandelbrot & Wallis 1968 "Joseph effect"). Fluctuación F(n)~n^α:
      α=0,5 sin memoria; 0,5<α<1 persistencia de largo alcance; α≈1 ruido 1/f; α≈1,5 browniano.
  [C] Surrogados PSEUDO-PERIÓDICOS (Small, Yu & Harrison 2001, PRL 87:188101): preservan la
      pseudo-periodicidad (ciclo estacional) pero destruyen el determinismo fino → testean
      no-linealidad MÁS ALLÁ del ciclo anual (cierra el hueco del IAAFT). Discriminante:
      ERROR DE PREDICCIÓN NO LINEAL (predictor local en el espacio de fases). Si el dato es
      MÁS predecible que sus surrogados → determinismo no capturado por el ciclo.

Serie: R=√Q diaria (pipeline del modelo). Caveats de longitud/forzamiento como en 162.

Salidas: dinamica_no_lineal/figuras/figura7_tests_complementarios.png + .../resultados/165_chaos_extra.csv
Run en .venv313: python scripts/06_eval/165_chaos_tests_extra.py
"""
import importlib.util
import logging
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extra165")
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R


# ── [A] entropía de permutación + complejidad estadística ──────────────────────
def perm_distribution(x, d=5, tau=1):
    n = len(x) - (d - 1) * tau
    allp = {p: 0 for p in permutations(range(d))}
    for i in range(n):
        vec = x[i:i + d * tau:tau]
        allp[tuple(np.argsort(vec, kind="mergesort"))] += 1
    counts = np.array(list(allp.values()), float)
    return counts / counts.sum()


def shannon(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def complexity_entropy(x, d=5, tau=1):
    P = perm_distribution(x, d, tau)
    N = math.factorial(d)
    Pe = np.full(N, 1.0 / N)
    Hs = shannon(P) / math.log(N)                       # entropía normalizada
    M = 0.5 * (P + Pe)
    JS = shannon(M) - 0.5 * shannon(P) - 0.5 * shannon(Pe)
    Qmax = -0.5 * (((N + 1) / N) * math.log(N + 1) - 2 * math.log(2 * N) + math.log(N))
    Q = JS / Qmax
    return Hs, Q * Hs                                    # (H_S, C)


# ── [B] DFA / Hurst ─────────────────────────────────────────────────────────────
def dfa(x, scales):
    y = np.cumsum(x - x.mean())
    F = []
    for n in scales:
        segs = len(y) // n
        if segs < 1:
            F.append(np.nan); continue
        rms = []
        t = np.arange(n)
        for v in range(segs):
            seg = y[v * n:(v + 1) * n]
            fit = np.polyval(np.polyfit(t, seg, 1), t)
            rms.append(np.mean((seg - fit) ** 2))
        F.append(np.sqrt(np.mean(rms)))
    F = np.array(F)
    ok = np.isfinite(F) & (F > 0)
    alpha = np.polyfit(np.log(np.array(scales)[ok]), np.log(F[ok]), 1)[0]
    return alpha, np.array(scales)[ok], F[ok]


# ── [C] error de predicción no lineal + surrogados pseudo-periódicos ───────────
def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.stack([x[i * tau:i * tau + n] for i in range(m)], axis=1)


def nlpe(x, m=4, tau=8, k=8, theiler=20):
    """RMSE de un predictor local-constante en el espacio de fases (1 paso)."""
    Z = embed(x, m, tau); N = len(Z) - 1
    tree = cKDTree(Z[:N])
    nxt = x[(m - 1) * tau + 1: (m - 1) * tau + 1 + N]     # valor siguiente alineado
    cur = x[(m - 1) * tau: (m - 1) * tau + N]
    err = []
    for i in range(0, N, 2):                              # submuestreo x2 por costo
        _, idx = tree.query(Z[i], k=k + theiler + 1)
        idx = idx[(np.abs(idx - i) > theiler) & (idx < N)][:k]
        if len(idx) == 0:
            continue
        err.append((nxt[i] - nxt[idx].mean()) ** 2)
    return math.sqrt(np.mean(err))


def pps_surrogate(x, m=4, tau=8, rho=None):
    """Surrogado pseudo-periódico (Small et al. 2001)."""
    Z = embed(x, m, tau); N = len(Z)
    if rho is None:
        rho = 0.7 * np.median(np.linalg.norm(Z - Z.mean(0), axis=1))
    out = np.empty(len(x)); out[:] = np.nan
    idx = int(RNG.integers(N - 1))
    obs0 = (m - 1) * tau
    out[obs0] = x[obs0 + 0]
    j = idx
    for t in range(1, len(x) - obs0):
        d = np.linalg.norm(Z[:N - 1] - Z[j], axis=1)
        w = np.exp(-d / rho); w /= w.sum()
        k = int(RNG.choice(N - 1, p=w))
        j = k + 1
        out[obs0 + t] = x[obs0 + j] if obs0 + j < len(x) else x[-1]
    return out[obs0:obs0 + (len(x) - obs0)]


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14)
    x = np.sqrt(np.clip(df["q"].values.astype(float), 0, None))
    log.info(f"serie R=√Q: N={len(x)}")

    # [A] plano complejidad-entropía: dato + referencias
    d = 5
    Hd, Cd = complexity_entropy(x, d=d)
    # referencias: ruido blanco, senoidal anual, AR(1) fuerte
    wn = RNG.standard_normal(len(x))
    sine = np.sin(2 * np.pi * np.arange(len(x)) / 365.25) + 0.05 * RNG.standard_normal(len(x))
    ar = np.zeros(len(x))
    for i in range(1, len(x)):
        ar[i] = 0.95 * ar[i - 1] + RNG.standard_normal()
    refs = {"dato R=√Q": (Hd, Cd), "ruido blanco": complexity_entropy(wn, d=d),
            "senoidal anual": complexity_entropy(sine, d=d), "AR(1) φ=0.95": complexity_entropy(ar, d=d)}
    log.info("[A] plano H-C (d=5): " + ", ".join(f"{k}=({v[0]:.3f},{v[1]:.3f})" for k, v in refs.items()))

    # [B] DFA
    scales = np.unique(np.logspace(np.log10(10), np.log10(len(x) // 4), 20).astype(int))
    alpha, sc, F = dfa(x, scales)
    log.info(f"[B] DFA α (Hurst) = {alpha:.3f}  (0.5 sin memoria; >0.5 persistencia largo alcance)")

    # [C] PPS: error de predicción no lineal dato vs surrogados
    step = max(1, len(x) // 3000); xs = x[::step]
    err_data = nlpe(xs)
    n_sur = 19
    err_sur = np.array([nlpe(pps_surrogate(xs)) for _ in range(n_sur)])
    rank = int((err_sur < err_data).sum())
    reject = rank == 0                                    # dato MÁS predecible que todos → determinismo
    log.info(f"[C] PPS: NLPE(dato)={err_data:.4f} vs surrogados μ={err_sur.mean():.4f} "
             f"[{err_sur.min():.4f},{err_sur.max():.4f}] rank {rank}/{n_sur} → "
             f"{'determinismo MÁS ALLÁ del ciclo (rechaza)' if reject else 'no concluyente'}")

    pd.DataFrame([dict(H_perm=round(Hd, 3), C_stat=round(Cd, 3), dfa_alpha=round(alpha, 3),
                       nlpe_data=round(err_data, 4), nlpe_sur_mean=round(float(err_sur.mean()), 4),
                       pps_rank=f"{rank}/{n_sur}", pps_determinismo=bool(reject))]
                 ).to_csv(RESDIR / "165_chaos_extra.csv", index=False)

    # figura
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), dpi=135)
    cols = {"dato R=√Q": "#C0392B", "ruido blanco": "#8FA0AC", "senoidal anual": "#D68910", "AR(1) φ=0.95": "#0B6E8C"}
    for k, (h, c) in refs.items():
        ax[0].scatter(h, c, s=90 if "dato" in k else 55, c=cols[k], label=k, zorder=3 if "dato" in k else 2,
                      edgecolors="k" if "dato" in k else "none", linewidths=1.2)
    ax[0].set_title("[A] Plano complejidad-entropía (d=5)\nRosso et al. 2007", fontsize=10)
    ax[0].set_xlabel("H_S (entropía de permutación)"); ax[0].set_ylabel("C (complejidad estadística)")
    ax[0].legend(fontsize=8)
    ax[1].loglog(sc, F, "o-", color="#0B6E8C", ms=4)
    ax[1].set_title(f"[B] DFA / Hurst: α={alpha:.3f}\nPeng et al. 1994", fontsize=10)
    ax[1].set_xlabel("escala n (días)"); ax[1].set_ylabel("F(n)")
    ax[2].hist(err_sur, bins=12, color="#8FA0AC", alpha=0.8, label="surrogados PPS")
    ax[2].axvline(err_data, color="#C0392B", lw=2, label="dato")
    ax[2].set_title(f"[C] Surrogados pseudo-periódicos\nNLPE: rank {rank}/{n_sur} → "
                    f"{'determinista' if reject else '—'}", fontsize=10)
    ax[2].set_xlabel("error de predicción no lineal"); ax[2].legend(fontsize=8)
    fig.suptitle("Tests complementarios: complejidad-entropía, DFA/Hurst, surrogados PPS", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura7_tests_complementarios.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura7_tests_complementarios.png'}")


if __name__ == "__main__":
    main()
