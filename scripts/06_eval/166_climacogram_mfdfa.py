#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 166 — Climacograma (Hurst-Kolmogorov, Koutsoyiannis) + MFDFA (multifractal).
Herramientas que la hidrología prefiere para "memoria larga" — hablan el idioma del
campo de Koutsoyiannis (el marco correcto según su crítica al "falso caos").

CONTEXTO (debate del campo): Koutsoyiannis (2006, HSJ 51:1065) argumenta que los procesos
hidrológicos NO son caos de baja dimensión y que sus peculiaridades (distribución en J,
autocorrelación alta) generan conclusiones falaces de caos; el marco correcto es
Hurst-Kolmogorov (persistencia de largo alcance). Nuestros tests (162/165) apuntan ahí.

TESTS Y FUENTES:
  [A] CLIMACOGRAMA (Koutsoyiannis 2010 HESS; Dimitriadis & Koutsoyiannis 2015, SERRA):
      γ(k) = varianza del proceso PROMEDIADO a escala k. Para un proceso Hurst-Kolmogorov
      γ(k) ∝ k^(2H-2) → pendiente β de log γ vs log k da H = 1 + β/2. (H=0,5 sin memoria;
      H→1 persistencia fuerte.) Preferido a DFA por menor sesgo (Koutsoyiannis).
  [B] MFDFA (Kantelhardt et al. 2002, Physica A 316:87): DFA multifractal. Función de
      fluctuación de orden q, Fq(s) ∝ s^h(q). Si h(q) VARÍA con q → MULTIFRACTAL; si es
      constante → monofractal. h(2)=Hurst estándar. Ancho Δh = h(q_min)-h(q_max) = fuerza
      de la multifractalidad. Espectro f(α) por transformada de Legendre.

Serie: R=√Q diaria. Caveat: el ciclo estacional domina escalas < 1 año (se declara).

Salidas: dinamica_no_lineal/figuras/figura8_memoria_larga_multifractal.png + .../resultados/166_climaco_mfdfa.csv
Run en .venv313: python scripts/06_eval/166_climacogram_mfdfa.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("clim166")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R


def climacogram(x, scales):
    g = []
    for k in scales:
        m = len(x) // k
        avg = x[:m * k].reshape(m, k).mean(1)
        g.append(avg.var(ddof=1))
    return np.array(g)


def mfdfa(x, scales, qs, order=2):
    Y = np.cumsum(x - x.mean())
    N = len(Y)
    Fq = np.zeros((len(qs), len(scales)))
    for si, s in enumerate(scales):
        ns = N // s
        F2 = []
        for v in range(ns):                              # desde el inicio
            seg = Y[v * s:(v + 1) * s]; t = np.arange(s)
            fit = np.polyval(np.polyfit(t, seg, order), t)
            F2.append(np.mean((seg - fit) ** 2))
        for v in range(ns):                              # desde el final (2*ns segmentos)
            seg = Y[N - (v + 1) * s:N - v * s]; t = np.arange(s)
            fit = np.polyval(np.polyfit(t, seg, order), t)
            F2.append(np.mean((seg - fit) ** 2))
        F2 = np.array(F2)
        F2 = F2[F2 > 1e-12]                               # excluye segmentos de var cero (q<0)
        for qi, q in enumerate(qs):
            if abs(q) < 1e-6:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                Fq[qi, si] = np.mean(F2 ** (q / 2.0)) ** (1.0 / q)
    hq = np.array([np.polyfit(np.log(scales), np.log(Fq[qi]), 1)[0] for qi in range(len(qs))])
    return hq, Fq


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14)
    x = np.sqrt(np.clip(df["q"].values.astype(float), 0, None))
    log.info(f"serie R=√Q: N={len(x)}")

    # [A] climacograma
    ks = np.unique(np.logspace(0, np.log10(len(x) // 10), 24).astype(int)); ks = ks[ks >= 1]
    g = climacogram(x, ks)
    beta = np.polyfit(np.log(ks), np.log(g), 1)[0]
    H_clim = 1 + beta / 2
    log.info(f"[A] climacograma: β={beta:.3f} → H = {H_clim:.3f} (0,5 sin memoria; →1 persistencia)")

    # [B] MFDFA
    scales = np.unique(np.logspace(np.log10(16), np.log10(len(x) // 6), 18).astype(int))
    qs = np.linspace(-5, 5, 21)
    hq, Fq = mfdfa(x, scales, qs)
    h2 = float(hq[np.argmin(np.abs(qs - 2))])
    dh = float(hq.max() - hq.min())
    # espectro multifractal f(α) por Legendre: τ(q)=q h(q)-1; α=dτ/dq; f=qα-τ
    tau = qs * hq - 1
    alpha = np.gradient(tau, qs)
    falpha = qs * alpha - tau
    log.info(f"[B] MFDFA: h(2)={h2:.3f} (≈Hurst), ancho Δh={dh:.3f} "
             f"({'MULTIFRACTAL' if dh > 0.1 else 'casi monofractal'})")

    pd.DataFrame([dict(H_climacograma=round(H_clim, 3), beta_clim=round(beta, 3),
                       h2_mfdfa=round(h2, 3), ancho_multifractal=round(dh, 3),
                       multifractal=bool(dh > 0.1))]).to_csv(RESDIR / "166_climaco_mfdfa.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), dpi=135)
    ax[0].loglog(ks, g, "o-", color="#0B6E8C", ms=4)
    ax[0].loglog(ks, np.exp(np.polyval([beta, np.log(g[0]) - beta * np.log(ks[0])], np.log(ks))),
                 "--", color="#C0392B", lw=1)
    ax[0].set_title(f"[A] Climacograma (Koutsoyiannis)\nH={H_clim:.3f}", fontsize=10)
    ax[0].set_xlabel("escala k (días)"); ax[0].set_ylabel("γ(k) varianza a escala k")
    ax[1].plot(qs, hq, "o-", color="#0B6E8C", ms=4); ax[1].axhline(0.5, color="#aaa", lw=0.7, ls=":")
    ax[1].set_title(f"[B] MFDFA: h(q)\nh(2)={h2:.3f}, Δh={dh:.3f}", fontsize=10)
    ax[1].set_xlabel("q"); ax[1].set_ylabel("h(q)")
    ax[2].plot(alpha, falpha, "o-", color="#6C4675", ms=4)
    ax[2].set_title("[C] Espectro multifractal f(α)", fontsize=10)
    ax[2].set_xlabel("α (exponente de Hölder)"); ax[2].set_ylabel("f(α)")
    fig.suptitle("Memoria larga (climacograma) y multifractalidad (MFDFA) — R=√Q (B23)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura8_memoria_larga_multifractal.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura8_memoria_larga_multifractal.png'}")


if __name__ == "__main__":
    main()
