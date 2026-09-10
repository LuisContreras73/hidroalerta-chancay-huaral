#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 162 — BATERÍA DE TESTS DE SISTEMAS DINÁMICOS sobre el caudal observado.
¿Es un atractor de baja dimensión? ¿Hay determinismo no lineal (más allá de un ciclo
estacional + ruido lineal)? ¿Hay evidencia de caos (Lyapunov>0)?

Implementado DESDE CERO (sin librería de caja negra) para que cada fórmula sea
verificable contra su fuente. Serie: R = √Q diario (m³/s), 1981-2025 (pipeline del
modelo, q completo). Se subsamplea a paso 'step' donde el nº de pares lo exige (declarado).

MÉTODOS Y FUENTES (para verificación):
  [1] Average Mutual Information → retardo τ. Fraser & Swinney (1986), Phys. Rev. A 33:1134.
      I(τ) = Σ_ij p_ij(τ) log[ p_ij(τ) / (p_i p_j) ];  τ* = primer mínimo local de I(τ).
  [2] False Nearest Neighbors → dimensión de embedding m. Kennel, Brown & Abarbanel (1992),
      Phys. Rev. A 45:3403. Vecino falso si |x_{i+mτ}-x_{j+mτ}|/‖y_i-y_j‖_m > R_tol (=15)
      o ‖·‖_{m+1}/σ > A_tol (=2). m* = donde FNN→~0.
  [3] Dimensión de correlación D2. Grassberger & Procaccia (1983), Phys. Rev. Lett. 50:346.
      C(r) = (2/[N(N-1)]) Σ_{i<j, |i-j|>w} Θ(r-‖y_i-y_j‖);  D2 = d ln C(r)/d ln r (zona lineal).
      Ventana de Theiler w (Theiler 1986, Phys. Rev. A 34:2427) excluye pares correlacionados.
      Se comprueba CONVERGENCIA de D2 al subir m (satura → dim finita; crece → estocástico).
  [4] Máximo exponente de Lyapunov λ1. Rosenstein, Collins & De Luca (1993), Physica D 65:117.
      ⟨ln d_j(i)⟩ ≈ ln d_j(0) + λ1·(i·Δt);  λ1 = pendiente de la región lineal.
  [5] Test de no-linealidad por SURROGADOS IAAFT. Schreiber & Schmitz (1996), Phys. Rev.
      Lett. 77:635 (surrogados que preservan espectro de potencia + histograma de amplitud →
      destruyen no-linealidad). Estadístico discriminante: asimetría de reversión temporal
      T_rev = ⟨(x_{t+1}-x_t)^3⟩ / ⟨(x_{t+1}-x_t)^2⟩^{3/2}  (Schreiber & Schmitz 1997,
      Physica D 142:346). Si T_rev(dato) cae FUERA de la nube de surrogados → se rechaza el
      nulo "proceso lineal gaussiano con el mismo espectro" → hay no-linealidad/determinismo.
  Contexto Takens (1981) LNM 898; Packard et al. (1980) PRL 45:712.

CAVEAT honesto: el caudal está FUERTEMENTE FORZADO por el ciclo estacional (pico anual);
D2 y λ1 en sistemas forzados/ruidosos son ESTIMADOS con incertidumbre. El paper (Saéz et
al. 2026) NO estima D2/λ1 — ajusta ODEs (global modelling), justamente porque a mensual
(≈540 puntos) faltan datos para invariantes robustos. Por eso los invariantes se corren a
DIARIO (16436 pts). No se afirma "caos" salvo que λ1>0 sea robusto Y los surrogados se rechacen.

Salidas: dinamica_no_lineal/figuras/figura4_tests_dinamicos.png + dinamica_no_lineal/resultados/162_chaos_tests.csv
Run en .venv313: python scripts/06_eval/162_chaos_tests.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("chaos162")
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R


# [1] Average Mutual Information (Fraser & Swinney 1986)
def ami(x, max_tau=200, bins=24):
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    out = []
    for tau in range(1, max_tau + 1):
        a, b = xn[:-tau], xn[tau:]
        c, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
        pab = c / c.sum(); pa = pab.sum(1); pb = pab.sum(0)
        nz = pab > 0
        I = np.sum(pab[nz] * np.log(pab[nz] / (np.outer(pa, pb)[nz] + 1e-15)))
        out.append(I)
    return np.array(out)


def first_local_min(v):
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] <= v[i + 1]:
            return i + 1
    return int(np.argmin(v)) + 1


def embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    return np.stack([x[i * tau:i * tau + N] for i in range(m)], axis=1)


# [2] False Nearest Neighbors (Kennel et al. 1992)
def fnn(x, tau, ms, Rtol=15.0, Atol=2.0):
    sigma = x.std(); fr = []
    for m in ms:
        Ym = embed(x, m, tau); Ym1 = embed(x, m + 1, tau)
        n = len(Ym1); Ym = Ym[:n]
        tree = cKDTree(Ym); d, idx = tree.query(Ym, k=2)
        nn = idx[:, 1]; dnn = d[:, 1] + 1e-12
        extra = np.abs(Ym1[:, -1] - Ym1[nn, -1])
        f1 = extra / dnn > Rtol
        f2 = np.sqrt(dnn ** 2 + extra ** 2) / sigma > Atol
        fr.append(float(np.mean(f1 | f2)))
    return np.array(fr)


# [3] Correlation dimension D2 (Grassberger-Procaccia 1983) con ventana de Theiler
def corr_dim(Y, w, n_ref=2000, n_r=22):
    N = len(Y)
    ref = RNG.choice(N, size=min(n_ref, N), replace=False)
    tree = cKDTree(Y)
    dmax = np.percentile(np.linalg.norm(Y - Y.mean(0), axis=1), 95) * 2
    rs = np.logspace(np.log10(dmax / 300), np.log10(dmax), n_r)
    Csum = np.zeros(n_r)
    npairs = 0
    for i in ref:
        dist, jidx = tree.query(Y[i], k=min(N, 400))  # vecinos cercanos
        jidx = jidx[np.abs(jidx - i) > w]              # Theiler
        dd = np.linalg.norm(Y[jidx] - Y[i], axis=1)
        Csum += np.array([(dd < r).sum() for r in rs])
        npairs += len(jidx)
    C = Csum / max(npairs, 1)
    ok = C > 0
    return rs, C, ok


def d2_slope(rs, C, ok):
    lr, lc = np.log(rs[ok]), np.log(C[ok])
    if len(lr) < 5:
        return np.nan
    # zona de escala: percentiles medios de C
    m = (C[ok] > 0.02) & (C[ok] < 0.5)
    if m.sum() < 4:
        m = np.ones(len(lr), bool)
    return float(np.polyfit(lr[m], lc[m], 1)[0])


# [4] Lyapunov máximo (Rosenstein et al. 1993)
def lyap_rosenstein(Y, w, k_max=40, n_ref=1500):
    N = len(Y); tree = cKDTree(Y)
    ref = RNG.choice(N - k_max, size=min(n_ref, N - k_max), replace=False)
    div = np.zeros(k_max); cnt = np.zeros(k_max)
    for i in ref:
        dist, jidx = tree.query(Y[i], k=min(N, 200))
        good = jidx[(np.abs(jidx - i) > w) & (jidx < N - k_max)]
        if len(good) == 0:
            continue
        j = good[0]
        for k in range(k_max):
            d = np.linalg.norm(Y[i + k] - Y[j + k])
            if d > 0:
                div[k] += np.log(d); cnt[k] += 1
    ok = cnt > 0
    curve = np.full(k_max, np.nan); curve[ok] = div[ok] / cnt[ok]
    return curve


# [5] Surrogados IAAFT (Schreiber & Schmitz 1996)
def iaaft(x, iters=200):
    n = len(x); amp = np.abs(np.fft.rfft(x)); xs = np.sort(x)
    s = RNG.permutation(x)
    for _ in range(iters):
        S = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=n)
        s = xs[s.argsort().argsort()]
    return s


def t_rev(x):   # asimetría de reversión temporal (Schreiber & Schmitz 1997)
    d = np.diff(x)
    return float(np.mean(d ** 3) / (np.mean(d ** 2) ** 1.5 + 1e-15))


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14)
    x_full = np.sqrt(np.clip(df["q"].values.astype(float), 0, None))   # R = √Q diario
    log.info(f"serie R=√Q diaria: N={len(x_full)}")

    # [1] AMI → τ
    A = ami(x_full, max_tau=200, bins=24); tau = first_local_min(A)
    log.info(f"[1] AMI: τ* = {tau} días (primer mínimo)")

    # subsample para invariantes (coste de pares); paso = max(1, τ//? ) declarado
    step = 2
    x = x_full[::step]; tau_s = max(1, tau // step)
    log.info(f"    subsample step={step} → N={len(x)}, τ(sub)={tau_s}")

    # [2] FNN → m
    ms = list(range(1, 9)); Fr = fnn(x, tau_s, ms)
    m_star = next((ms[i] for i, f in enumerate(Fr) if f < 0.05), ms[int(np.argmin(Fr))])
    log.info(f"[2] FNN por m: {dict(zip(ms, Fr.round(3)))} → m* = {m_star}")

    # [3] D2 con convergencia sobre m
    w = tau_s * 2
    d2_by_m = {}
    curves = {}
    for m in [2, 3, 4, 5, 6]:
        Y = embed(x, m, tau_s)
        rs, Cc, ok = corr_dim(Y, w)
        d2_by_m[m] = round(d2_slope(rs, Cc, ok), 3)
        curves[m] = (rs, Cc, ok)
    log.info(f"[3] D2(m): {d2_by_m}  (satura → dim finita)")

    # [4] Lyapunov (en m*)
    Yly = embed(x, max(m_star, 3), tau_s)
    lycurve = lyap_rosenstein(Yly, w)
    kk = np.arange(len(lycurve))
    lin = slice(1, 12)  # región lineal inicial
    finite = np.isfinite(lycurve[lin])
    lam = float(np.polyfit(kk[lin][finite], lycurve[lin][finite], 1)[0]) * (1.0 / step)  # por día
    log.info(f"[4] Lyapunov λ1 ≈ {lam:.4f} /día (pendiente región lineal; >0 sugiere caos)")

    # [5] Surrogados IAAFT + T_rev
    n_sur = 39                       # 39 surrogados → test bilateral α≈0.05 (rank)
    trev_data = t_rev(x_full)
    trev_sur = np.array([t_rev(iaaft(x_full)) for _ in range(n_sur)])
    rank = int((trev_sur < trev_data).sum())
    reject = (rank == 0) or (rank == n_sur)     # dato en la cola extrema
    log.info(f"[5] IAAFT: T_rev(dato)={trev_data:.4f}; surrogados μ={trev_sur.mean():.4f} "
             f"σ={trev_sur.std():.4f}; rango [{trev_sur.min():.4f},{trev_sur.max():.4f}]; "
             f"rank={rank}/{n_sur} → {'RECHAZA nulo lineal (no-linealidad)' if reject else 'NO rechaza'}")

    # guardar métricas
    pd.DataFrame([dict(N=len(x_full), tau=tau, m_star=m_star,
                       **{f"D2_m{m}": v for m, v in d2_by_m.items()},
                       lyap_per_day=round(lam, 4), trev_data=round(trev_data, 4),
                       trev_sur_mean=round(float(trev_sur.mean()), 4),
                       trev_sur_std=round(float(trev_sur.std()), 4),
                       surrogate_rank=f"{rank}/{n_sur}",
                       reject_linear_null=bool(reject))]).to_csv(RESDIR / "162_chaos_tests.csv", index=False)

    # ── figura ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2), dpi=135)
    ax[0].plot(range(1, len(A) + 1), A, color="#0B6E8C"); ax[0].axvline(tau, color="#C0392B", ls="--")
    ax[0].set_title(f"[1] AMI → τ*={tau} d\n(Fraser-Swinney 1986)", fontsize=9.5)
    ax[0].set_xlabel("retardo τ (días)"); ax[0].set_ylabel("I(τ) [nats]")
    ax[1].plot(ms, Fr, "o-", color="#0B6E8C"); ax[1].axvline(m_star, color="#C0392B", ls="--"); ax[1].axhline(0.05, color="#aaa", lw=0.7)
    ax[1].set_title(f"[2] FNN → m*={m_star}\n(Kennel et al. 1992)", fontsize=9.5)
    ax[1].set_xlabel("dim. embedding m"); ax[1].set_ylabel("fracción vecinos falsos")
    for m, (rs, Cc, ok) in curves.items():
        ax[2].plot(np.log(rs[ok]), np.log(Cc[ok]), "-", lw=1, label=f"m={m}")
    ax[2].set_title("[3] Grassberger-Procaccia\nln C(r) vs ln r", fontsize=9.5)
    ax[2].set_xlabel("ln r"); ax[2].set_ylabel("ln C(r)"); ax[2].legend(fontsize=7)
    ax[3].plot(kk, lycurve, "o-", color="#0B6E8C", ms=3)
    ax[3].plot(kk[lin], np.polyval(np.polyfit(kk[lin][finite], lycurve[lin][finite], 1), kk[lin]), "--", color="#C0392B")
    ax[3].set_title(f"[4] Lyapunov (Rosenstein 1993)\nλ1≈{lam:.4f}/día", fontsize=9.5)
    ax[3].set_xlabel("pasos k"); ax[3].set_ylabel("⟨ln d(k)⟩")
    ax[4].hist(trev_sur, bins=15, color="#8FA0AC", alpha=0.8, label="surrogados IAAFT")
    ax[4].axvline(trev_data, color="#C0392B", lw=2, label="dato")
    ax[4].set_title(f"[5] Surrogados: T_rev\nrank {rank}/{n_sur} → "
                    f"{'no-lineal' if reject else 'no concluyente'}", fontsize=9.5)
    ax[4].set_xlabel("T_rev"); ax[4].legend(fontsize=7)
    fig.suptitle("Tests dinámicos: AMI, FNN, D2, Lyapunov, surrogados — R=√Q (B23)",
                 fontsize=11.5, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura4_tests_dinamicos.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura4_tests_dinamicos.png'}")


if __name__ == "__main__":
    main()
