#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 172 — Evaluación OPERATIVA de alerta temprana de los modelos TFT.
¿Cómo funcionan como sistema de alerta de crecidas (Q>Q90), operativamente y por
tiempo de anticipación? Añade el TFT-canónico (145) al banco de alerta y mide lo que
faltaba: POD/FAR/CSI por horizonte, discriminación (AUC) vs calibración (BSS), el
TIEMPO DE ANTICIPACIÓN por evento, y la línea temporal operativa.

Umbral de alerta: Q90 = 40.89 m³/s. Test 2024 (aforo real): 17 días de excedencia en
5 eventos → potencia estadística LIMITADA (los grandes eventos fueron 2023/val); se declara.

P(Q>Q90) de los 3 cuantiles (p10,p50,p90): lognormal ajustada por 2 cuantiles
  μ=log(p50), σ=(log p90−log p10)/(2·1.2816);  P=1−Φ((log Q90−μ)/σ).
Deterministic alert: p50>Q90 → POD/FAR/CSI. Probabilística: AUC (ROC), BSS.
Lead-time: para cada evento, máximo horizonte h con P(exced)>0.3 anticipando el día.

Contexto: 147_prob_alert.csv (RA-TFT, persistencia, Chronos-2, etc.) — aquí se suma el canónico.

Salidas: reports/diagnostico/alerta_operativa_temprana.png + outputs/ml_Q/172_alerta_operativa.csv
Run en .venv313: python scripts/06_eval/172_alerta_operativa.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "reports/diagnostico"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("alerta172")

spec = importlib.util.spec_from_file_location("c145", ROOT / "scripts/05_models/145_canonical_tft.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
T, R, DEV, H = C.T, C.R, C.DEV, C.H
Q90 = R.Q90
LEADS = [1, 3, 7, 14]
P_TRIG = 0.30          # probabilidad de disparo operativo


def p_exceed(q10, q50, q90, thr):
    q10 = np.clip(q10, 1e-3, None); q50 = np.clip(q50, 1e-3, None)
    q90 = np.maximum(q90, q50 + 1e-3)
    mu = np.log(q50); sig = np.maximum((np.log(q90) - np.log(q10)) / (2 * 1.2816), 1e-3)
    return 1 - norm.cdf((np.log(thr) - mu) / sig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    df = R.build(H); dts = df.index; obs = df["obs"].values.astype(float); q = df["q"].values.astype(float)
    EM = bp["enc"]
    tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    q_tr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31")]]
    qmu, qsd = float(np.mean(q_tr)), float(np.std(q_tr) + 1e-6)
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xte = T.seqs_for_enc(df, te, EM, H, (mu, sd))
    P3 = []
    for s in range(3):
        mo = C.TFTCanonical(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                            H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd).to(DEV)
        mo.load_state_dict(torch.load(OUT / f"145_TFTcanonico_seed{s}.pt", map_location=DEV)); mo.eval()
        with torch.no_grad():
            P3.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
    PR = np.mean(P3, 0)      # (N, H, 3)

    # métricas por horizonte
    rows = []
    for h in LEADS:
        j = [i + h - 1 for i in te]
        o = np.array([obs[k] for k in j]); m = np.isfinite(o)
        oe = (o[m] > Q90).astype(int)
        p10, p50, p90 = PR[m, h - 1, 0], PR[m, h - 1, 1], PR[m, h - 1, 2]
        pe = p_exceed(p10, p50, p90, Q90)
        # determinista (mediana > Q90)
        pa = (p50 > Q90).astype(int)
        tp = int(((pa == 1) & (oe == 1)).sum()); fp = int(((pa == 1) & (oe == 0)).sum()); fn = int(((pa == 0) & (oe == 1)).sum())
        pod = tp / (tp + fn) if (tp + fn) else np.nan
        far = fp / (tp + fp) if (tp + fp) else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        auc = roc_auc_score(oe, pe) if len(np.unique(oe)) > 1 else np.nan
        base = oe.mean(); bs = np.mean((pe - oe) ** 2); bss = 1 - bs / (base * (1 - base) + 1e-9)
        rows.append(dict(lead=h, n=int(m.sum()), eventos=int(oe.sum()),
                         POD=round(pod, 3), FAR=round(far, 3), CSI=round(csi, 3),
                         AUC=round(auc, 3), BSS=round(bss, 3)))
        log.info(f"h={h:2d}: POD={pod:.2f} FAR={far:.2f} CSI={csi:.2f} | AUC={auc:.2f} BSS={bss:+.2f}")
    met = pd.DataFrame(rows); met.to_csv(OUT / "172_alerta_operativa.csv", index=False)

    # P(exced) por (día objetivo, horizonte) para lead-time + timeline
    Pmat = {h: {} for h in LEADS}
    for k, i in enumerate(te):
        for h in LEADS:
            tgt = i + h - 1
            Pmat[h][tgt] = float(p_exceed(PR[k, h - 1, 0], PR[k, h - 1, 1], PR[k, h - 1, 2], Q90))
    # eventos (rachas de excedencia en test)
    tdays = sorted({i for i in range(len(df)) if dts[i] >= pd.Timestamp("2024-01-01") and np.isfinite(obs[i])})
    oe_full = {i: (obs[i] > Q90) for i in tdays}
    ev_starts = [i for i in tdays if oe_full[i] and not oe_full.get(i - 1, False)]
    leadtimes = []
    for D in ev_starts:
        lt = 0
        for h in LEADS:
            if Pmat[h].get(D, 0) > P_TRIG:
                lt = h
        leadtimes.append((dts[D].date(), obs[D], lt))
    log.info("Lead-time por evento (máx h con P>0.3 anticipando el inicio):")
    for d, o, lt in leadtimes:
        log.info(f"  evento {d} (Q={o:.1f}): {lt} d de anticipación")

    # ── figura ──
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), dpi=130)
    hs = [r["lead"] for r in rows]
    ax[0, 0].plot(hs, [r["POD"] for r in rows], "o-", label="POD (aciertos)", color="#2E8B6F")
    ax[0, 0].plot(hs, [r["CSI"] for r in rows], "s-", label="CSI", color="#0B6E8C")
    ax[0, 0].plot(hs, [r["FAR"] for r in rows], "^--", label="FAR (falsas)", color="#C0392B")
    ax[0, 0].set_title("Alerta determinista (mediana > Q90) por horizonte", fontsize=11)
    ax[0, 0].set_xlabel("horizonte (días)"); ax[0, 0].set_ylabel("métrica"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_ylim(-0.05, 1.05)
    ax[0, 1].plot(hs, [r["AUC"] for r in rows], "o-", label="AUC (discriminación)", color="#0A3D54")
    ax[0, 1].plot(hs, [r["BSS"] for r in rows], "s-", label="BSS (calibración)", color="#D68910")
    ax[0, 1].axhline(0.5, color="#aaa", lw=0.7, ls=":"); ax[0, 1].axhline(0, color="#aaa", lw=0.7)
    ax[0, 1].set_title("Discrimina bien (AUC↑) pero calibra mal a horizonte largo (BSS→<0)", fontsize=11)
    ax[0, 1].set_xlabel("horizonte (días)"); ax[0, 1].legend(fontsize=8)
    # timeline operativo h=3 y h=7
    xs = [dts[i] for i in tdays]
    for h, col in [(3, "#0B6E8C"), (7, "#D68910")]:
        ax[1, 0].plot(xs, [Pmat[h].get(i, np.nan) for i in tdays], color=col, lw=1.1, label=f"P(exced) h={h}")
    exc_days = [dts[i] for i in tdays if oe_full[i]]
    for xd in exc_days:
        ax[1, 0].axvline(xd, color="#C0392B", lw=0.8, alpha=0.5)
    ax[1, 0].axhline(P_TRIG, color="#555", lw=0.7, ls="--")
    ax[1, 0].set_title("Línea temporal operativa (rojo = excedencia real; línea = disparo 0.3)", fontsize=11)
    ax[1, 0].set_ylabel("P(Q>Q90)"); ax[1, 0].legend(fontsize=8); ax[1, 0].tick_params(axis="x", labelsize=7)
    # lead-time por evento
    labs = [str(d) for d, o, lt in leadtimes]; lts = [lt for d, o, lt in leadtimes]
    ax[1, 1].bar(range(len(labs)), lts, color="#0B6E8C")
    ax[1, 1].set_xticks(range(len(labs))); ax[1, 1].set_xticklabels(labs, fontsize=7, rotation=20)
    ax[1, 1].set_title(f"Anticipación por evento (máx h con P>{P_TRIG})", fontsize=11)
    ax[1, 1].set_ylabel("días de anticipación"); ax[1, 1].set_yticks(LEADS)
    fig.suptitle("Evaluación operativa de alerta temprana — TFT canónico, Chancay-Huaral (test 2024, 17 excedencias/5 eventos)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(); fig.savefig(FIGDIR / "alerta_operativa_temprana.png", bbox_inches="tight")
    log.info(f"figura: {FIGDIR / 'alerta_operativa_temprana.png'}")


if __name__ == "__main__":
    main()
