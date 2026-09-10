#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 163 — El atractor con Q REAL (aforo) vs Q RELLENADO. Integridad ante hidrólogos senior.

MOTIVACIÓN (advertencia del usuario, correcta). El caudal que usa el embedding de fig25/160
está COMPLETO porque en los huecos se rellenó (producto/modelo conceptual). El aforo REAL
solo cubre ~2020-2024; 2015-2019 tienen CERO días reales. Entonces buena parte del atractor
"limpio" 2015-2025 es dinámica del MODELO DE RELLENO, no del río. Aquí lo separamos y probamos
si la estructura (y la no-linealidad) SOBREVIVE con dato real.

QUÉ HACE:
  1. Marca cada ventana como REAL (todos sus 60 días con aforo real, df['obs'] finito) o
     RELLENADA. Colorea el atractor 2015-2025 por esa marca → se ve cuánto es relleno.
  2. Atractor SOLO con ventanas reales (≈2021-2024) proyectado en la misma base PCA.
  3. Test de no-linealidad (surrogados IAAFT + T_rev; ver 162) sobre el tramo REAL contiguo
     más largo, para comprobar que la no-linealidad NO es artefacto del rellenador.

SUSTENTO PARA HIDRÓLOGOS SENIOR (lo que este script permite decir con honestidad):
  · Separamos aforo real de relleno; la caracterización dinámica se verifica en dato real.
  · Si la no-linealidad se mantiene en el tramo real → es del río, no del filler.
  · Declaramos que 2015-2019 es 100% sintético y no se sobre-interpreta.
  · Limitaciones: registro real corto (~4 años), huecos, sin afirmar caos.

Salidas: dinamica_no_lineal/figuras/figura5_integridad_aforo_real.png
         dinamica_no_lineal/resultados/163_realdata.csv
Run en .venv313: python scripts/06_eval/163_attractor_realdata.py
Fuentes de los tests: ver script 162 (Fraser-Swinney, Schreiber-Schmitz, etc.).
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("real163")
RNG = np.random.default_rng(0)
L = 60

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
R = C.R


def iaaft(x, iters=200):
    n = len(x); amp = np.abs(np.fft.rfft(x)); xs = np.sort(x)
    s = RNG.permutation(x)
    for _ in range(iters):
        S = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=n)
        s = xs[s.argsort().argsort()]
    return s


def t_rev(x):
    d = np.diff(x)
    return float(np.mean(d ** 3) / (np.mean(d ** 2) ** 1.5 + 1e-15))


def longest_run(finite):
    best_s = best_e = s = 0
    for i in range(1, len(finite) + 1):
        if i < len(finite) and finite[i] and finite[i - 1]:
            continue
        if i - s > best_e - best_s:
            best_s, best_e = s, i
        s = i
    return best_s, best_e


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    df = R.build(14); dts = df.index
    q = df["q"].values.astype(float)          # rellenado (completo)
    obs = df["obs"].values.astype(float)       # aforo REAL (con huecos)
    pr = df["pr"].values.astype(float); api = df["api"].values.astype(float)
    finite_obs = np.isfinite(obs)
    log.info(f"aforo real: {int(finite_obs.sum())}/{len(obs)} días")

    # embedding fig25 (base ajustada en filled 2015+, como 160)
    F = np.stack([q, pr, api]); mu = F.mean(1, keepdims=True); sd = F.std(1, keepdims=True) + 1e-6
    Fz = (F - mu) / sd
    ends = [i for i in range(L, len(dts)) if dts[i] >= pd.Timestamp("2015-01-01")]
    W = np.stack([Fz[:, e - L:e].ravel() for e in ends]).astype(np.float32)
    pca = PCA(2).fit(W); E = pca.transform(W)
    # marca real: ventana con TODOS sus 60 días de aforo real
    real_win = np.array([finite_obs[e - L:e].all() for e in ends])
    log.info(f"ventanas 2015+: {len(ends)} | reales (60/60 aforo): {int(real_win.sum())}")

    # atractor SOLO real: canal q = obs real (finito) en esas ventanas
    Fr = np.stack([np.where(finite_obs, obs, np.nan), pr, api])
    Frz = (Fr - mu) / sd
    ends_real = [e for e in ends if finite_obs[e - L:e].all()]
    Er = pca.transform(np.stack([Frz[:, e - L:e].ravel() for e in ends_real]).astype(np.float32)) \
        if ends_real else np.empty((0, 2))

    # test de no-linealidad en el tramo REAL contiguo más largo
    s0, s1 = longest_run(finite_obs)
    xr = np.sqrt(np.clip(obs[s0:s1], 0, None))
    trev_r = t_rev(xr); sur = np.array([t_rev(iaaft(xr)) for _ in range(39)])
    rank = int((sur < trev_r).sum()); reject = rank in (0, 39)
    log.info(f"tramo real contiguo: {dts[s0].date()}→{dts[s1-1].date()} ({s1-s0} d). "
             f"T_rev={trev_r:.3f} vs surrogados [{sur.min():.3f},{sur.max():.3f}] rank {rank}/39 → "
             f"{'NO-LINEAL (rechaza)' if reject else 'no concluyente'}")

    pd.DataFrame([dict(dias_reales=int(finite_obs.sum()), n_ventanas=len(ends),
                       n_ventanas_reales=int(real_win.sum()),
                       tramo_real=f"{dts[s0].date()}..{dts[s1-1].date()}", tramo_dias=int(s1 - s0),
                       trev_real=round(trev_r, 3), trev_sur_min=round(float(sur.min()), 3),
                       trev_sur_max=round(float(sur.max()), 3), rank=f"{rank}/39",
                       nolineal_en_real=bool(reject))]).to_csv(RESDIR / "163_realdata.csv", index=False)

    # figura
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=140)
    ax[0].scatter(E[~real_win, 0], E[~real_win, 1], s=6, c="#D68910", alpha=0.35, linewidths=0, label="ventana con relleno")
    ax[0].scatter(E[real_win, 0], E[real_win, 1], s=8, c="#0B6E8C", alpha=0.7, linewidths=0, label="ventana 100% aforo real")
    ax[0].set_title(f"A · Atractor 2015-2025: real vs relleno\nsolo {int(real_win.sum())}/{len(ends)} ventanas son 100% reales", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper right"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].scatter(E[:, 0], E[:, 1], s=3, c="#dfe6ec", alpha=0.5, linewidths=0)
    if len(Er):
        ax[1].scatter(Er[:, 0], Er[:, 1], s=8, c="#0B6E8C", alpha=0.7, linewidths=0)
    ax[1].set_title(f"B · Atractor SOLO aforo real (≈2021-2024)\n{len(ends_real)} ventanas", fontsize=10)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[2].hist(sur, bins=15, color="#8FA0AC", alpha=0.8, label="surrogados IAAFT")
    ax[2].axvline(trev_r, color="#C0392B", lw=2, label="dato REAL")
    ax[2].set_title(f"C · No-linealidad en aforo REAL\nT_rev={trev_r:.2f}, rank {rank}/39 → "
                    f"{'no-lineal' if reject else '—'}", fontsize=10)
    ax[2].set_xlabel("T_rev"); ax[2].legend(fontsize=8)
    fig.suptitle("Integridad del atractor: aforo real vs relleno — Chancay-Huaral (B23)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figura5_integridad_aforo_real.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura5_integridad_aforo_real.png'}")


if __name__ == "__main__":
    main()
