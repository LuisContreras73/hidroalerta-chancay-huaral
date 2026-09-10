#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 174 (VERIFICACIÓN #2) — ¿sobreviven H, Δh y la no-linealidad en el AFORO REAL?

Todo (162/166/173) se calculó sobre df["q"] = serie mayormente RELLENADA con GR4J (~1.629/16.436
días reales). Aquí recomputamos sobre el aforo REAL observado y comparamos de forma JUSTA (misma
longitud), para saber cuánto de la estructura es del dato medido vs del relleno.

Estrategia: (1) tramo real continuo más largo; (2) H (climacograma), Δh (MFDFA), T_rev + surrogados
sobre ESE tramo; (3) control shuffled; (4) comparar contra el MISMO tramo rellenado y contra
ventanas aleatorias de la serie rellenada de igual longitud (valor esperado ± dispersión).
Honestidad: series cortas → MFDFA con rango de escalas limitado; se reporta como tal.

Salida: dinamica_no_lineal/resultados/174_verif_realdata.csv
Run (.venv313): python scripts/06_eval/174_verif_realdata.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("verif174")
spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C); R = C.R
rng = np.random.default_rng(0)


def mfdfa_dh(x, scales, qs, order=2):
    Y = np.cumsum(x - x.mean()); N = len(Y); Fq = np.zeros((len(qs), len(scales)))
    for si, s in enumerate(scales):
        ns = N // s
        if ns < 1:
            Fq[:, si] = np.nan; continue
        F2 = []
        for v in range(ns):
            seg = Y[v*s:(v+1)*s]; t = np.arange(s); F2.append(np.mean((seg-np.polyval(np.polyfit(t, seg, order), t))**2))
        for v in range(ns):
            seg = Y[N-(v+1)*s:N-v*s]; t = np.arange(s); F2.append(np.mean((seg-np.polyval(np.polyfit(t, seg, order), t))**2))
        F2 = np.array(F2); F2 = F2[F2 > 1e-12]
        for qi, q in enumerate(qs):
            Fq[qi, si] = np.exp(0.5*np.mean(np.log(F2))) if abs(q) < 1e-6 else np.mean(F2**(q/2.0))**(1.0/q)
    ok = np.isfinite(Fq[0])
    hq = np.array([np.polyfit(np.log(scales[ok]), np.log(Fq[qi][ok]), 1)[0] for qi in range(len(qs))])
    return hq.max()-hq.min()


def climaco_H(x, scales):
    scales = [k for k in scales if len(x)//k >= 3]
    g = [x[:len(x)//k*k].reshape(len(x)//k, k).mean(1).var(ddof=1) for k in scales]
    return 1 + np.polyfit(np.log(scales), np.log(g), 1)[0]/2


def t_rev(x):
    d = np.diff(x); return np.mean(d**3)/(np.mean(d**2)**1.5 + 1e-12)


def iaaft(x, iters=200):
    amp = np.sort(x); S = np.abs(np.fft.rfft(x)); y = rng.permutation(x)
    for _ in range(iters):
        Y = np.fft.rfft(y); Y = S*np.exp(1j*np.angle(Y)); y = np.fft.irfft(Y, n=len(x))
        y = amp[np.argsort(np.argsort(y))]
    return y


def metrics(x, L):
    scales = np.unique(np.logspace(np.log10(8), np.log10(L//4), 12).astype(int)); scales = scales[scales >= 4]
    ks = np.unique(np.logspace(0, np.log10(L//8), 14).astype(int)); ks = ks[ks >= 2]
    qs = np.linspace(-5, 5, 21)
    return climaco_H(x, ks), mfdfa_dh(x, scales, qs), scales


def main():
    df = R.build(14); q = df["q"].values.astype(float); obs = df["obs"].values.astype(float)
    real = np.isfinite(obs)
    # tramos reales continuos
    idx = np.where(real)[0]; runs = np.split(idx, np.where(np.diff(idx) != 1)[0]+1)
    runs = [r for r in runs if len(r) >= 60]; runs.sort(key=len, reverse=True)
    seg = runs[0]; L = len(seg)
    log.info(f"aforo real: {real.sum()}/{len(real)} días · {len(runs)} tramos≥60d · más largo={L}d "
             f"({df.index[seg[0]].date()}→{df.index[seg[-1]].date()})")

    xr = np.sqrt(np.clip(obs[seg], 0, None))          # REAL
    xf = np.sqrt(np.clip(q[seg], 0, None))            # RELLENO en misma ventana
    Hr, dhr, sc = metrics(xr, L); Hf, dhf, _ = metrics(xf, L)
    log.info(f"escalas MFDFA usadas: {list(sc)} (rango limitado por L={L})")
    log.info(f"REAL (mismo tramo):    H={Hr:.3f}  Δh={dhr:.3f}")
    log.info(f"RELLENO (mismo tramo): H={Hf:.3f}  Δh={dhf:.3f}")

    # control shuffled sobre el real
    NC = 30
    Hsh = np.array([climaco_H(rng.permutation(xr), np.unique(np.logspace(0, np.log10(L//8), 14).astype(int))) for _ in range(NC)])
    dhsh = np.array([mfdfa_dh(rng.permutation(xr), sc, np.linspace(-5, 5, 21)) for _ in range(NC)])
    log.info(f"SHUFFLED del real:     H={Hsh.mean():.3f}±{Hsh.std():.3f} (≈0.5)  Δh={dhsh.mean():.3f}±{dhsh.std():.3f}")

    # ventanas aleatorias de igual longitud en la serie completa (rellenada) → esperado a esa L
    K = 60; Hw, dhw = [], []
    for _ in range(K):
        s0 = rng.integers(0, len(q)-L); w = np.sqrt(np.clip(q[s0:s0+L], 0, None))
        Hw.append(metrics(w, L)[0]); dhw.append(metrics(w, L)[1])
    Hw, dhw = np.array(Hw), np.array(dhw)
    log.info(f"ventanas L={L} rellenas: H={Hw.mean():.3f}±{Hw.std():.3f}  Δh={dhw.mean():.3f}±{dhw.std():.3f} (esperado a esta longitud)")

    # no-linealidad sobre el real
    ns = 39; td = t_rev(xr); ts = np.array([t_rev(iaaft(xr)) for _ in range(ns)])
    rank = int((ts < td).sum())
    log.info(f"NO-LINEALIDAD real: T_rev={td:.4f}; surrogados μ={ts.mean():.4f}±{ts.std():.4f}; rank={rank}/{ns} "
             f"→ {'RECHAZA nulo lineal' if rank in (0, ns) else 'NO rechaza'}")

    log.info("=== VEREDICTO #2 ===")
    log.info(f"  Memoria larga REAL: H={Hr:.3f} (shuffled {Hsh.mean():.2f}) → {'SOBREVIVE' if Hr-Hsh.mean() > 2*Hsh.std() else 'no distinguible'}")
    log.info(f"  Multifractalidad REAL: Δh={dhr:.3f} vs shuffled {dhsh.mean():.3f} → {'SOBREVIVE (pero L corto, rango limitado)' if dhr-dhsh.mean() > 2*dhsh.std() else 'NO robusta en real (underpowered)'}")
    log.info(f"  real vs relleno misma ventana: ΔH={Hr-Hf:+.3f}, ΔΔh={dhr-dhf:+.3f}")

    pd.DataFrame([dict(real_days=int(real.sum()), n_runs=len(runs), longest=L,
                       H_real=round(Hr,3), H_fill_samewin=round(Hf,3), H_shuffled=round(float(Hsh.mean()),3),
                       H_fillwindows=round(float(Hw.mean()),3),
                       dh_real=round(dhr,3), dh_fill_samewin=round(dhf,3), dh_shuffled=round(float(dhsh.mean()),3),
                       dh_shuffled_std=round(float(dhsh.std()),3), dh_fillwindows=round(float(dhw.mean()),3),
                       trev_real=round(td,4), surrogate_rank=f"{rank}/{ns}")]).to_csv(RESDIR/"174_verif_realdata.csv", index=False)
    log.info(f"guardado: {RESDIR/'174_verif_realdata.csv'}")


if __name__ == "__main__":
    main()
