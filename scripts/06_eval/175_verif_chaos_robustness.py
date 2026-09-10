#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 175 (VERIFICACIÓN #3) — ¿es ROBUSTO el "NO caos de baja dimensión"? (D2, λ1)

El 162 fija a mano ventana de Theiler w=2τ y región de ajuste slice(1,12). Wang & Gan (1998):
una w demasiado pequeña hace que D2 SATURE espuriamente BAJO (falso caos). Aquí barremos:
  · D2 vs dimensión de embedding m (¿satura=caos, o crece=estocástico/alta dim?) para VARIAS
    ventanas de Theiler w (τ, 2τ, 5τ, ~estacional 180). Si D2 crece con m para toda w razonable
    → NO es caos de baja dimensión (robusto).
  · λ1 (Rosenstein) del dato vs distribución de λ1 de surrogados IAAFT. Si λ1_dato NO supera a los
    surrogados → NO hay evidencia de exponente de Lyapunov positivo genuino (no caos determinista).

Salida: dinamica_no_lineal/resultados/175_verif_chaos.csv
Run (.venv313): python scripts/06_eval/175_verif_chaos_robustness.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("verif175")
spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C); R = C.R
rng = np.random.default_rng(0)


def ami_tau(x, max_tau=200, bins=24):
    xn = (x-x.min())/(x.max()-x.min()+1e-12); A = []
    for tau in range(1, max_tau+1):
        a, b = xn[:-tau], xn[tau:]; H2, _, _ = np.histogram2d(a, b, bins)
        pab = H2/H2.sum(); pa = pab.sum(1); pb = pab.sum(0); nz = pab > 0
        A.append(np.sum(pab[nz]*np.log(pab[nz]/(np.outer(pa, pb)[nz]))))
    A = np.array(A)
    for i in range(1, len(A)-1):
        if A[i] < A[i-1] and A[i] < A[i+1]:
            return i+1
    return int(np.argmin(A))+1


def embed(x, m, tau):
    N = len(x)-(m-1)*tau; return np.stack([x[i*tau:i*tau+N] for i in range(m)], axis=1)


def corr_dim_slope(Y, w, n_ref=1500, n_r=24):
    N = len(Y); refs = rng.choice(N, min(n_ref, N), replace=False)
    d = []
    for i in refs:
        j = np.arange(N); mask = np.abs(j-i) > w
        dd = np.sqrt(((Y[mask]-Y[i])**2).sum(1)); d.append(dd)
    d = np.concatenate(d); d = d[d > 0]
    rs = np.logspace(np.log10(np.percentile(d, 1)), np.log10(np.percentile(d, 60)), n_r)
    C = np.array([(d < r).mean() for r in rs]); ok = C > 0
    lr, lc = np.log(rs[ok]), np.log(C[ok])
    # pendiente en la zona central (evita saturación y ruido de colas)
    if len(lr) < 6:
        return np.nan
    a, b = int(0.2*len(lr)), int(0.8*len(lr))
    return float(np.polyfit(lr[a:b], lc[a:b], 1)[0])


def lyap(Y, w, k_max=30, n_ref=1200):
    N = len(Y); refs = rng.choice(N-k_max, min(n_ref, N-k_max), replace=False)
    div = np.zeros(k_max)
    for i in refs:
        j = np.arange(N-k_max); mask = np.abs(j-i) > w
        cand = j[mask]; dd = ((Y[cand]-Y[i])**2).sum(1); nn = cand[np.argmin(dd)]
        for k in range(k_max):
            div[k] += np.log(np.sqrt(((Y[i+k]-Y[nn+k])**2).sum())+1e-12)
    div /= len(refs)
    return float(np.polyfit(np.arange(1, 11), div[1:11], 1)[0])   # pendiente región inicial


def iaaft(x, iters=150):
    amp = np.sort(x); S = np.abs(np.fft.rfft(x)); y = rng.permutation(x)
    for _ in range(iters):
        Y = np.fft.rfft(y); Y = S*np.exp(1j*np.angle(Y)); y = np.fft.irfft(Y, n=len(x)); y = amp[np.argsort(np.argsort(y))]
    return y


def main():
    df = R.build(14); x_full = np.sqrt(np.clip(df["q"].values.astype(float), 0, None))
    tau = ami_tau(x_full); log.info(f"AMI τ*={tau} d")
    step = 2; x = x_full[::step]; ts = max(1, tau//step)
    log.info(f"subsample step={step} → N={len(x)}, τ_sub={ts}")

    # ── D2 vs m para varias ventanas de Theiler ──
    ws = {"τ": ts, "2τ(162)": 2*ts, "5τ": 5*ts, "estacional~180": max(180//step, 5*ts)}
    log.info("=== D2(m) por ventana de Theiler (satura→caos; crece→estocástico/alta dim) ===")
    rows_d2 = {}
    for wn, w in ws.items():
        d2 = {m: round(corr_dim_slope(embed(x, m, ts), w), 2) for m in [3, 5, 7, 10]}
        rows_d2[wn] = d2
        crece = d2[10] > d2[3] + 0.5
        log.info(f"  w={wn:14s} D2(m=3,5,7,10)={list(d2.values())} → {'CRECE (no satura → no caos baja dim)' if crece else 'satura'}")

    # ── λ1 dato vs surrogados IAAFT ──
    Y = embed(x, 5, ts); wD = 2*ts
    lam_data = lyap(Y, wD)
    lam_sur = np.array([lyap(embed(iaaft(x_full)[::step], 5, ts), wD) for _ in range(12)])
    z = (lam_data - lam_sur.mean())/(lam_sur.std()+1e-9)
    log.info("=== Lyapunov λ1: dato vs surrogados ===")
    log.info(f"  λ1(dato)={lam_data:.4f}/paso; surrogados μ={lam_sur.mean():.4f}±{lam_sur.std():.4f}; z={z:.1f}σ "
             f"→ {'λ1>surrogados (posible caos)' if z > 2 else 'NO distinguible de surrogado lineal → sin evidencia de caos'}")

    pd.DataFrame([dict(tau=tau, **{f"D2_{wn}_m10": rows_d2[wn][10] for wn in ws},
                       D2_2tau_m3=rows_d2["2τ(162)"][3], D2_2tau_m10=rows_d2["2τ(162)"][10],
                       lam_data=round(lam_data,4), lam_sur_mean=round(float(lam_sur.mean()),4),
                       lam_sur_std=round(float(lam_sur.std()),4), lam_z=round(z,2),
                       chaos_evidence=bool(z > 2))]).to_csv(RESDIR/"175_verif_chaos.csv", index=False)
    log.info("=== VEREDICTO #3 ===")
    log.info("  Si D2 crece con m para toda w Y λ1 no supera surrogados → 'NO caos de baja dim' es ROBUSTO.")
    log.info(f"guardado: {RESDIR/'175_verif_chaos.csv'}")


if __name__ == "__main__":
    main()
