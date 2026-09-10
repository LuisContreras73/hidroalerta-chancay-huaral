#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 159 — EVALUACIÓN EN EL ESPACIO DE FASES: ¿reproduce el modelo la geometría
del atractor observado? (rigor nivel-revisor; los autores de Saéz et al. 2026 —
Rau/UTEC, Lavado/SENAMHI, Mangiarotti — pueden revisar esto).

MOTIVACIÓN. Saéz et al. 2026 (HP 40:e70582, cuenca B23 = Chancay-Huaral) modelan la
dinámica lluvia-caudal con la global modelling technique y validan reproduciendo la
GEOMETRÍA de las órbitas en el espacio de fases (su Fig. 6: retrato diferencial
(R, dR/dt) con R=√ρ). Un pronosticador puede tener buen NSE y aun así COLAPSAR el
atractor hacia la media (fallo conocido del DL). Aquí evaluamos si el TFT reproduce
el atractor y cómo se degrada con el horizonte — una verificación dinámica que va
más allá de NSE/CRPS.

MÉTODO (espejo del paper donde importa, honesto donde diferimos):
  - Transformación √Q (como su R=√ρ), unidades m³/s.
  - Derivada dR/dt por SUAVIZADO POLINÓMICO LOCAL (Savitzky-Golay, ventana 15 d,
    orden 3) — análogo a la derivada por ventana de GPoM (9 pasos). CORTE EN HUECOS:
    se estima solo dentro de tramos contiguos de obs real (Regla 5); como ellos
    insertan un hueco entre cuencas para no generar derivadas espurias.
  - Diferencia declarada: nosotros DIARIO (ellos MENSUAL) → más dinámica rápida y más
    ruido; por eso el suavizado. NO afirmamos caos (Lyapunov+); es un atractor de
    baja dimensión FORZADO ESTACIONALMENTE con variabilidad interanual (ENSO).

COMPARACIÓN (mismas fechas, apples-to-apples):
  Observado vs TFT-canónico (ensemble 3 seeds, mediana) a h=1/7/14, + Persistencia h14
  como referencia (preserva la varianza trivialmente → separa fidelidad-de-atractor de
  habilidad-de-pronóstico). Ventana de test (2024+, solo obs finita).

MÉTRICAS de geometría vs observado (robustas primero, D2 con caveat):
  1. Área de la órbita = área del casco convexo de (R, dR/dt) → razón pred/obs
     (<1 = atractor encogido/colapsado).
  2. Amplitud dinámica: std(R), std(dR/dt) → razones (¿preserva el rango de tasas?).
  3. Divergencia de densidad en fase: Jensen-Shannon entre histogramas 2D (grid común).
  4. Dimensión de correlación D2 (Grassberger-Procaccia sobre la nube 2D) — indicativa
     en series cortas; se reporta con caveat.

Salidas: dinamica_no_lineal/figuras/figura1_retrato_fase_diario.png + dinamica_no_lineal/resultados/159_attractor_metrics.csv
Run en .venv313: python scripts/06_eval/159_attractor_geometry.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "dinamica_no_lineal/figuras"
RESDIR = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("attractor159")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
T, R, DEV, H, LEADS = C.T, C.R, C.DEV, C.H, C.LEADS
SG_WIN, SG_ORD = 15, 3          # Savitzky-Golay: ventana ~medio mes, orden 3


# ── derivada con corte en huecos (no derivar a través de NaN) ──────────────────
def deriv_segmented(dates, vals):
    """R=√Q y dR/dt por Savitzky-Golay dentro de tramos contiguos (paso diario)
    de obs finita; devuelve arrays alineados (NaN donde no se puede estimar)."""
    vals = np.asarray(vals, float)
    R_ = np.sqrt(np.clip(vals, 0, None))
    dR = np.full_like(R_, np.nan)
    finite = np.isfinite(R_)
    # tramos contiguos en fecha (paso 1 día) y con dato finito
    d = pd.DatetimeIndex(dates)
    brk = np.where((np.diff(d.values).astype("timedelta64[D]").astype(int) != 1) |
                   (~finite[1:]) | (~finite[:-1]))[0] + 1
    for seg in np.split(np.arange(len(R_)), brk):
        if len(seg) >= SG_WIN and np.isfinite(R_[seg]).all():
            dR[seg] = savgol_filter(R_[seg], SG_WIN, SG_ORD, deriv=1)  # d/dt (paso=1 día)
    return R_, dR


# ── métricas de geometría ──────────────────────────────────────────────────────
def hull_area(x, y):
    P = np.column_stack([x, y]); P = P[np.isfinite(P).all(1)]
    if len(P) < 3:
        return np.nan
    try:
        return float(ConvexHull(P).volume)   # 'volume' en 2D = área
    except Exception:
        return np.nan


def js_div(xo, yo, xp, yp, bins=24):
    m = np.isfinite(xo) & np.isfinite(yo)
    rng = [[np.min(xo[m]), np.max(xo[m])], [np.min(yo[m]), np.max(yo[m])]]
    Ho, _, _ = np.histogram2d(xo[m], yo[m], bins=bins, range=rng)
    mp = np.isfinite(xp) & np.isfinite(yp)
    Hp, _, _ = np.histogram2d(xp[mp], yp[mp], bins=bins, range=rng)
    P = Ho.ravel() + 1e-9; Q = Hp.ravel() + 1e-9
    P /= P.sum(); Q /= Q.sum(); Mx = 0.5 * (P + Q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return float(0.5 * kl(P, Mx) + 0.5 * kl(Q, Mx))


def corr_dim(x, y, nr=18):
    """Dimensión de correlación (Grassberger-Procaccia) de la nube 2D (R,dR).
    Indicativa en series cortas — se reporta con caveat."""
    P = np.column_stack([x, y]); P = P[np.isfinite(P).all(1)]
    n = len(P)
    if n < 50:
        return np.nan
    from scipy.spatial.distance import pdist
    dd = pdist(P)
    dd = dd[dd > 0]
    rs = np.logspace(np.log10(np.percentile(dd, 2)), np.log10(np.percentile(dd, 60)), nr)
    C_ = np.array([(dd < r).mean() for r in rs])
    ok = C_ > 0
    if ok.sum() < 5:
        return np.nan
    sl = np.polyfit(np.log(rs[ok]), np.log(C_[ok]), 1)[0]
    return float(sl)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True); RESDIR.mkdir(parents=True, exist_ok=True)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    df = R.build(H); dts = df.index; obs = df["obs"].values; q = df["q"].values
    EM = bp["enc"]
    tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    va = [i for i in range(EM, len(df) - H) if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    q_tr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
    qmu, qsd = float(np.mean(q_tr)), float(np.std(q_tr) + 1e-6)
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))

    # predicción ensemble del canónico (3 seeds, mediana)
    preds = []
    for s in range(3):
        mo = C.TFTCanonical(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                            H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd).to(DEV)
        mo.load_state_dict(torch.load(OUT / f"145_TFTcanonico_seed{s}.pt", map_location=DEV))
        mo.eval()
        with torch.no_grad():
            preds.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
    PR = np.mean(preds, 0)      # (N, H, 3) — [p10, p50, p90]

    # Par (predicho, observado) sobre las MISMAS fechas de validez (te[k]+h-1);
    # se compara solo donde AMBOS y sus derivadas son finitos (Regla 5, apples-to-apples).
    def series_pair(h, kind):
        fechas, vpred, vobs = [], [], []
        for k, i in enumerate(te):
            j = i + h - 1
            if j >= len(dts):
                continue
            fechas.append(dts[j]); vobs.append(obs[j])
            if kind == "obs":
                vpred.append(obs[j])
            elif kind == "tft":
                vpred.append(PR[k, h - 1, 1])                 # mediana
            elif kind == "persist":
                vpred.append(q[i - 1] if np.isfinite(q[i - 1]) else np.nan)
        f = pd.DatetimeIndex(fechas)
        Rp, dRp = deriv_segmented(f, np.array(vpred, float))  # derivada sobre serie completa
        Ro, dRo = deriv_segmented(f, np.array(vobs, float))   #   (corte en huecos)
        m = np.isfinite(Rp) & np.isfinite(dRp) & np.isfinite(Ro) & np.isfinite(dRo)
        return Rp[m], dRp[m], Ro[m], dRo[m]                   # nubes emparejadas (mismas fechas)

    casos = [("Observado", "obs", 1, "#0A3D54"),
             ("TFT h=1", "tft", 1, "#1BA8C4"),
             ("TFT h=7", "tft", 7, "#0B6E8C"),
             ("TFT h=14", "tft", 14, "#6C4675"),
             ("Persistencia h=14", "persist", 14, "#C0392B")]

    rows, port = [], {}
    for nombre, kind, h, _c in casos:
        Rp, dRp, Ro, dRo = series_pair(h, kind)
        port[nombre] = (Rp, dRp)                              # nube predicha (emparejada)
        A, Ao = hull_area(Rp, dRp), hull_area(Ro, dRo)
        rows.append(dict(
            caso=nombre, h=h, n=int(len(Rp)),
            area=round(A, 2) if np.isfinite(A) else np.nan,
            area_ratio=round(A / Ao, 3) if (np.isfinite(A) and Ao) else np.nan,
            std_R_ratio=round(np.std(Rp) / np.std(Ro), 3) if np.std(Ro) else np.nan,
            std_dR_ratio=round(np.std(dRp) / np.std(dRo), 3) if np.std(dRo) else np.nan,
            JS_dens=round(js_div(Ro, dRo, Rp, dRp), 4),
            D2=round(corr_dim(Rp, dRp), 3),
            D2_obs=round(corr_dim(Ro, dRo), 3)))
    met = pd.DataFrame(rows)
    met.to_csv(RESDIR / "159_attractor_metrics.csv", index=False)
    log.info("\n" + met.to_string(index=False))
    log.info("Lectura: area_ratio<1 = atractor encogido; std_dR_ratio<1 = dinámica (tasas) "
             "amortiguada; JS↑ = densidad de fase más distinta; D2 vs D2_obs = dimensión del atractor.")

    # ── figura ──────────────────────────────────────────────────────────────────
    xall = np.concatenate([port[n][0][np.isfinite(port[n][0])] for n, *_ in casos])
    yall = np.concatenate([port[n][1][np.isfinite(port[n][1])] for n, *_ in casos])
    xlim = (0, np.nanpercentile(xall, 99.5)); ylim = (np.nanpercentile(yall, 0.5), np.nanpercentile(yall, 99.5))
    fig = plt.figure(figsize=(16, 7.2), dpi=130)
    gs = fig.add_gridspec(2, 5, height_ratios=[2.2, 1.0], hspace=0.42, wspace=0.28)
    for idx, (nombre, kind, h, col) in enumerate(casos):
        ax = fig.add_subplot(gs[0, idx])
        Rr, dRr = port[nombre]
        ax.plot(Rr, dRr, color="#c8d2da", lw=0.4, alpha=0.6, zorder=1)
        ax.scatter(Rr, dRr, s=6, c=col, alpha=0.6, linewidths=0, zorder=2)
        ax.axhline(0, color="#e2e8ee", lw=0.8, zorder=0)
        r = met[met.caso == nombre].iloc[0]
        sub = "atractor observado" if nombre == "Observado" else \
              f"área={r.area_ratio}× · JS={r.JS_dens}"
        ax.set_title(f"{nombre}\n{sub}", fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("R = √Q  (√(m³/s))", fontsize=9)
        if idx == 0:
            ax.set_ylabel("dR/dt  (√(m³/s)/día)", fontsize=9)
        ax.tick_params(labelsize=8)
    # barras de métricas
    order = ["TFT h=1", "TFT h=7", "TFT h=14", "Persistencia h=14"]
    sub = met.set_index("caso").loc[order]
    axb = fig.add_subplot(gs[1, 0:2]); axj = fig.add_subplot(gs[1, 3:5])
    xpos = np.arange(len(order)); cols = ["#1BA8C4", "#0B6E8C", "#6C4675", "#C0392B"]
    axb.bar(xpos, sub.area_ratio.values, color=cols); axb.axhline(1, color="#0A3D54", lw=1, ls="--")
    axb.set_xticks(xpos); axb.set_xticklabels(order, fontsize=7.5, rotation=12)
    axb.set_title("Área de órbita / observado  (1 = fiel; <1 = colapsa)", fontsize=9)
    axb.set_ylabel("razón", fontsize=8); axb.tick_params(labelsize=8)
    axj.bar(xpos, sub.JS_dens.values, color=cols)
    axj.set_xticks(xpos); axj.set_xticklabels(order, fontsize=7.5, rotation=12)
    axj.set_title("Divergencia de densidad en fase (JS, ↓ mejor)", fontsize=9)
    axj.set_ylabel("JS (bits)", fontsize=8); axj.tick_params(labelsize=8)
    fig.suptitle("Retrato de fase diferencial (R, dR/dt) a paso diario — Chancay-Huaral (B23)",
                 fontsize=12, y=1.0)
    fig.savefig(FIGDIR / "figura1_retrato_fase_diario.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'figura1_retrato_fase_diario.png'}")


if __name__ == "__main__":
    main()
