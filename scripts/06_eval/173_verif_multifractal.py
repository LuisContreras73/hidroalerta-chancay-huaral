#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 173 (VERIFICACIÓN) — ¿es GENUINA la multifractalidad (Δh) y la memoria larga (H)?

El script 166 declara "multifractal" solo por Δh>0,1, SIN controles. La literatura (Kantelhardt
2002; Koutsoyiannis 2006) exige distinguir multifractalidad genuina (de correlaciones no lineales)
de la ESPURIA (por distribución de cola pesada, finitud, estacionalidad, o el relleno GR4J).

Controles obligatorios sobre R=√Q:
  · SHUFFLED (barajado): destruye TODA correlación temporal, conserva la distribución.
    → si Δh_orig ≈ Δh_shuffled, la multifractalidad NO viene de correlaciones (espuria).
  · SURROGADO IAAFT: conserva espectro (correlación lineal) + distribución, destruye no-linealidad.
    → Δh_orig > Δh_surr indica multifractalidad NO LINEAL genuina.
  · DESESTACIONALIZADO (resta climatología día-del-año): controla si el ciclo anual infla H/Δh.
Descomposición estándar: Δh_correlaciones = Δh_orig − Δh_shuffled.

Salida: dinamica_no_lineal/resultados/173_verif_multifractal.csv
Run (.venv313): python scripts/06_eval/173_verif_multifractal.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("verif173")
spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C); R = C.R
rng = np.random.default_rng(0)


def mfdfa_dh(x, scales, qs, order=2):
    Y = np.cumsum(x - x.mean()); N = len(Y); Fq = np.zeros((len(qs), len(scales)))
    for si, s in enumerate(scales):
        ns = N // s; F2 = []
        for v in range(ns):
            seg = Y[v*s:(v+1)*s]; t = np.arange(s); fit = np.polyval(np.polyfit(t, seg, order), t); F2.append(np.mean((seg-fit)**2))
        for v in range(ns):
            seg = Y[N-(v+1)*s:N-v*s]; t = np.arange(s); fit = np.polyval(np.polyfit(t, seg, order), t); F2.append(np.mean((seg-fit)**2))
        F2 = np.array(F2); F2 = F2[F2 > 1e-12]
        for qi, q in enumerate(qs):
            Fq[qi, si] = np.exp(0.5*np.mean(np.log(F2))) if abs(q) < 1e-6 else np.mean(F2**(q/2.0))**(1.0/q)
    hq = np.array([np.polyfit(np.log(scales), np.log(Fq[qi]), 1)[0] for qi in range(len(qs))])
    return hq.max()-hq.min(), float(hq[np.argmin(np.abs(qs-2))])


def climaco_H(x, scales):
    g = [x[:len(x)//k*k].reshape(len(x)//k, k).mean(1).var(ddof=1) for k in scales]
    return 1 + np.polyfit(np.log(scales), np.log(g), 1)[0]/2


def iaaft(x, iters=200):
    amp = np.sort(x); S = np.abs(np.fft.rfft(x)); y = rng.permutation(x)
    for _ in range(iters):
        Y = np.fft.rfft(y); Y = S*np.exp(1j*np.angle(Y)); y = np.fft.irfft(Y, n=len(x))
        y = amp[np.argsort(np.argsort(y))]
    return y


def main():
    df = R.build(14); dts = df.index
    x = np.sqrt(np.clip(df["q"].values.astype(float), 0, None))
    doy = dts.dayofyear.values
    clim = pd.Series(x).groupby(doy).transform("mean").values
    x_des = x - clim + x.mean()          # desestacionalizado (resta climatología DOY)
    scales = np.unique(np.logspace(np.log10(16), np.log10(len(x)//6), 18).astype(int))
    ks = np.unique(np.logspace(0, np.log10(len(x)//10), 24).astype(int)); ks = ks[ks >= 2]
    qs = np.linspace(-5, 5, 21); NC = 20

    dh_o, h2_o = mfdfa_dh(x, scales, qs); H_o = climaco_H(x, ks)
    dh_d, _ = mfdfa_dh(x_des, scales, qs); H_d = climaco_H(x_des, ks)
    log.info(f"ORIGINAL:        Δh={dh_o:.3f}  h(2)={h2_o:.3f}  H_clim={H_o:.3f}")
    log.info(f"DESESTACIONAL.:  Δh={dh_d:.3f}              H_clim={H_d:.3f}  (Δ vs orig: {dh_d-dh_o:+.3f} / {H_d-H_o:+.3f})")

    dh_sh, H_sh = [], []
    for _ in range(NC):
        xs = rng.permutation(x); dh_sh.append(mfdfa_dh(xs, scales, qs)[0]); H_sh.append(climaco_H(xs, ks))
    dh_su = [mfdfa_dh(iaaft(x), scales, qs)[0] for _ in range(NC)]
    dh_sh, H_sh, dh_su = map(np.array, (dh_sh, H_sh, dh_su))
    log.info(f"SHUFFLED (n={NC}): Δh={dh_sh.mean():.3f}±{dh_sh.std():.3f}   H_clim={H_sh.mean():.3f}±{H_sh.std():.3f}  (esperado H≈0.5)")
    log.info(f"SURROGADO IAAFT:  Δh={dh_su.mean():.3f}±{dh_su.std():.3f}")

    dh_corr = dh_o - dh_sh.mean()        # multifractalidad atribuible a correlaciones
    z_sh = (dh_o - dh_sh.mean())/(dh_sh.std()+1e-9)
    z_su = (dh_o - dh_su.mean())/(dh_su.std()+1e-9)
    log.info("=== VEREDICTO ===")
    log.info(f"  Δh por correlaciones (orig−shuffled) = {dh_corr:.3f}  (z={z_sh:.1f}σ sobre shuffled)")
    log.info(f"  Δh vs surrogado IAAFT: z={z_su:.1f}σ  → {'multifractalidad NO LINEAL genuina' if z_su>2 else 'NO distinguible de lineal+distribución'}")
    log.info(f"  Fracción del Δh=1,26 que es genuina (correlaciones): {100*dh_corr/dh_o:.0f}%  (el resto = distribución/finitud)")
    log.info(f"  Hurst: orig {H_o:.3f} vs desestacional {H_d:.3f} vs shuffled {H_sh.mean():.3f} (control≈0.5) → memoria larga {'ROBUSTA' if H_d>0.6 else 'DUDOSA (estacional)'}")

    pd.DataFrame([dict(dh_orig=round(dh_o,3), dh_deseason=round(dh_d,3), dh_shuffled=round(float(dh_sh.mean()),3),
                       dh_shuffled_std=round(float(dh_sh.std()),3), dh_surrogate=round(float(dh_su.mean()),3),
                       dh_correlaciones=round(dh_corr,3), pct_genuina=round(100*dh_corr/dh_o,1),
                       z_vs_shuffled=round(z_sh,2), z_vs_surrogate=round(z_su,2),
                       H_orig=round(H_o,3), H_deseason=round(H_d,3), H_shuffled=round(float(H_sh.mean()),3))]
                 ).to_csv(RESDIR/"173_verif_multifractal.csv", index=False)
    log.info(f"guardado: {RESDIR/'173_verif_multifractal.csv'}")


if __name__ == "__main__":
    main()
