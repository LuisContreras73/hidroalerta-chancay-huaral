#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 160 — Fidelidad de atractor en el ESPACIO PCA CORRECTO (el de fig25/script 111).

CORRIGE el criterio del 159. El 159 usó el embedding diferencial (Q, dQ/dt), fiel a la
Fig. 6 del paper pero RUIDOSO en resolución diaria → no se parecía al atractor limpio.
El criterio correcto para nuestros datos diarios es el de fig25 (script 111): ventanas
de L=60 días de 3 canales (q, pr, api), estandarizadas globalmente → PCA(2). El solape
de ventanas produce la órbita suave (el atractor de los papers).

PREGUNTA (igual que 159, pero en el espacio bueno): ¿el TFT reproduce la geometría del
atractor observado, y cómo se degrada con el horizonte?

MÉTODO:
  1. Embedding fig25: F=[q,pr,api] (pipeline del modelo; q idéntico a D7, corr=1.0),
     z-score global, ventanas L=60, PCA(2) AJUSTADA sobre ventanas 2015+ (= atractor ref).
  2. Predicción del canónico (145, 3 seeds, mediana) sobre test (2024+) → serie diaria
     q̂_h para h=1,7,14. Persistencia h14 = último obs del origen.
  3. Se PROYECTA sobre la MISMA PCA: se reemplaza SOLO el canal q por q̂_h (pr, api
     observados), se estandariza con el MISMO (mu, sd), y se transforma. Ventanas cuyo
     contenido [e-L, e) cae completo en el periodo test-predicho (apples-to-apples).
  4. Métricas de geometría vs observado-test (mismas fechas de fin de ventana):
     área del casco convexo (razón), alcance de los "brazos" de crecida (p99 del radio),
     divergencia Jensen-Shannon de densidad 2D. + persistencia como referencia.

Nota honesta: el embedding usa la serie q COMPLETA (rellenada, = fig25); en 2025 el obs
real tiene 77% de huecos, por eso la referencia allí es la serie rellenada (consistente
con fig25). El canal q es el de mayor varianza (crecidas) → si el modelo suaviza, los
BRAZOS del atractor se retraen: visible y medible.

Salidas: dinamica_no_lineal/figuras/figura2_fidelidad_atractor_tft.png + dinamica_no_lineal/resultados/160_attractor_pca_metrics.csv
Run en .venv313: python scripts/06_eval/160_attractor_pca_fidelity.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("attr160")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
T, R, DEV, H = C.T, C.R, C.DEV, C.H
Q90 = R.Q90
L = 60          # ventana (idéntica a fig25/script 111)


def windows(Fz, ends):
    """(3,N) z-scored → (len(ends), 3*L) ventanas [e-L, e)."""
    return np.stack([Fz[:, e - L:e].ravel() for e in ends]).astype(np.float32)


def hull_area(P):
    P = P[np.isfinite(P).all(1)]
    if len(P) < 3:
        return np.nan
    try:
        return float(ConvexHull(P).volume)
    except Exception:
        return np.nan


def reach99(P, c):
    """p99 del radio desde el centroide observado — mide alcance de los brazos."""
    r = np.sqrt(((P - c) ** 2).sum(1))
    return float(np.percentile(r, 99))


def js_div(A, B, ref_range, bins=26):
    (x0, x1), (y0, y1) = ref_range
    ha, _, _ = np.histogram2d(A[:, 0], A[:, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    hb, _, _ = np.histogram2d(B[:, 0], B[:, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    P = ha.ravel() + 1e-9; Q = hb.ravel() + 1e-9
    P /= P.sum(); Q /= Q.sum(); M = 0.5 * (P + Q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return float(0.5 * kl(P, M) + 0.5 * kl(Q, M))


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    df = R.build(H); dts = df.index
    q = df["q"].values.astype(float); pr = df["pr"].values.astype(float); api = df["api"].values.astype(float)

    # ── embedding fig25: z-score global + PCA(2) ajustada en ventanas 2015+ ──
    F = np.stack([q, pr, api]); mu = F.mean(1, keepdims=True); sd = F.std(1, keepdims=True) + 1e-6
    Fz = (F - mu) / sd
    ends_all = [i for i in range(L, len(dts)) if dts[i] >= pd.Timestamp("2015-01-01")]
    X_all = windows(Fz, ends_all)
    pca = PCA(2).fit(X_all); E_all = pca.transform(X_all)
    flood_all = q[ends_all] > Q90
    log.info(f"embedding fig25: {len(ends_all)} ventanas, var explicada {pca.explained_variance_ratio_.round(3)}")

    # ── predicción del canónico (3 seeds, mediana) sobre test ──
    EM = bp["enc"]
    tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    q_tr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31")]]
    qmu, qsd = float(np.mean(q_tr)), float(np.std(q_tr) + 1e-6)
    muS = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sdS = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xte = T.seqs_for_enc(df, te, EM, H, (muS, sdS))
    P3 = []
    for s in range(3):
        mo = C.TFTCanonical(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                            H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd).to(DEV)
        mo.load_state_dict(torch.load(OUT / f"145_TFTcanonico_seed{s}.pt", map_location=DEV))
        mo.eval()
        with torch.no_grad():
            P3.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
    PR = np.mean(P3, 0)      # (len(te), H, 3)

    def qhat_series(h):
        s = q.copy()
        for k, i in enumerate(te):
            j = i + h - 1
            if j < len(s):
                s[j] = PR[k, h - 1, 1]
        return s

    def qpers_series(h):
        s = q.copy()
        for k, i in enumerate(te):
            j = i + h - 1
            if j < len(s) and np.isfinite(q[i - 1]):
                s[j] = q[i - 1]
        return s

    # ventanas de fin en el periodo test-predicho: [e-L, e) ⊂ dates>=2024-01-01
    t0 = next(i for i in range(len(dts)) if dts[i] >= pd.Timestamp("2024-01-01"))
    ends_test = [e for e in range(L, len(dts)) if e - L >= t0]
    E_obs_test = pca.transform(windows(Fz, ends_test))
    c_obs = E_obs_test.mean(0)
    rng = [[E_all[:, 0].min(), E_all[:, 0].max()], [E_all[:, 1].min(), E_all[:, 1].max()]]

    def proj(qseries):
        Fp = np.stack([qseries, pr, api]); Fpz = (Fp - mu) / sd
        return pca.transform(windows(Fpz, ends_test))

    casos = [("Observado (test)", E_obs_test, "#0A3D54"),
             ("TFT h=1", proj(qhat_series(1)), "#1BA8C4"),
             ("TFT h=7", proj(qhat_series(7)), "#0B6E8C"),
             ("TFT h=14", proj(qhat_series(14)), "#6C4675"),
             ("Persistencia h=14", proj(qpers_series(14)), "#C0392B")]

    A_obs = hull_area(E_obs_test); reach_obs = reach99(E_obs_test, c_obs)
    rows = []
    for nombre, E, _c in casos:
        A = hull_area(E)
        rows.append(dict(caso=nombre, n=len(E),
                         area=round(A, 1), area_ratio=round(A / A_obs, 3),
                         reach_ratio=round(reach99(E, c_obs) / reach_obs, 3),
                         JS_dens=round(js_div(E_obs_test, E, rng), 4)))
    met = pd.DataFrame(rows)
    met.to_csv(RESDIR / "160_attractor_pca_metrics.csv", index=False)
    log.info("\n" + met.to_string(index=False))

    # ── figura estilo fig25 (atractor limpio); fondo = atractor completo tenue ──
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), dpi=140)
    for ax, (nombre, E, col) in zip(axes, casos):
        ax.scatter(E_all[:, 0], E_all[:, 1], s=3, c="#dfe6ec", alpha=0.5, linewidths=0, zorder=1)
        ax.scatter(E[:, 0], E[:, 1], s=7, c=col, alpha=0.65, linewidths=0, zorder=2)
        r = met[met.caso == nombre].iloc[0]
        sub = "referencia" if nombre.startswith("Observado") else \
              f"área={r.area_ratio}× · alcance={r.reach_ratio}× · JS={r.JS_dens}"
        ax.set_title(f"{nombre}\n{sub}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(rng[0]); ax.set_ylim(rng[1])
    fig.suptitle("Fidelidad del atractor del TFT en el espacio de fases (PCA) — test 2024+",
                 fontsize=11.5, y=1.06)
    fig.savefig(FIGDIR / "figura2_fidelidad_atractor_tft.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura2_fidelidad_atractor_tft.png'}")


if __name__ == "__main__":
    main()
