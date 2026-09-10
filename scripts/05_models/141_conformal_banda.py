#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 141 — Recalibración CONFORMAL de la banda del RA-TFT (ablación ABL-gen5 b).

Problema medido (FIG8): la banda q10–q90 del RA-TFT publicado cubre 85 % a 1 día
pero colapsa a 57–62 % a multi-día (nominal: 80 %). Los fundacionales, en cambio,
cubren 81–88 % — su única ventaja "estructural" además del régimen.

Método: CQR (Conformalized Quantile Regression, Romano et al. 2019, split-conformal):
  1. Predicciones de VALIDACIÓN 2023 con los checkpoints CONGELADOS de la release
     gen4_2026-07-02 (los 3 seeds del 125; no se entrena nada).
  2. Por lead h: puntaje E_i = max(p10−y, y−p90); corrección q̂_h = cuantil
     ⌈(n+1)·0,8⌉/n de {E_i} (cobertura objetivo 80 %).
  3. Banda recalibrada en TEST (2024–25): [p10−q̂_h, p90+q̂_h]. El p50 NO se toca.
  Regla 3 respetada: la calibración usa SOLO val-2023; el test queda intacto.

Salidas (append-only, regla 6):
  outputs/ml_Q/141_conformal_banda.csv       (cobertura/ancho/CRPS antes vs después)
  outputs/ml_Q/141_qhat_por_lead.csv         (correcciones q̂_h — para operación)
  reports/diagnostico/FIG14_conformal_banda.png

Run en .venv313:
    python scripts/05_models/141_conformal_banda.py
"""
import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
REL = ROOT / "models/releases/gen4_2026-07-02"
FIGDIR = ROOT / "reports/diagnostico"
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("conformal141")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# mismos módulos que el 125: T=114 (fija R.FUT=[sin,cos]), R=105
spec = importlib.util.spec_from_file_location(
    "t114", ROOT / "scripts/05_models/114_tft_intensive.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R = T.R
DEV = R.DEV
H = 14
LEADS = [1, 3, 7, 14]
ALPHA = 0.2                      # banda q10–q90 → cobertura nominal 80 %


def main():
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
    EM = 90
    tr = [i for i in range(EM, len(df) - H)
          if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    va = [i for i in range(EM, len(df) - H)
          if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31")
          and np.isfinite(q[i - EM:i + H]).all()]
    log.info(f"val-2023: {len(va)} emisiones · checkpoints: {REL.name}")
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6,
          "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    P = []
    for s in range(3):
        mo = R.RATFT(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                     H=H, drop=bp["drop"], use_future=True).to(DEV)
        mo.load_state_dict(torch.load(REL / f"125_tft_seed{s}.pt", map_location=DEV))
        mo.eval()
        with torch.no_grad():
            P.append(mo(Xva[0].to(DEV), Xva[1].to(DEV), Xva[2].to(DEV)).cpu().numpy())
        log.info(f"  seed {s}: inferencia val OK")
    PRva = np.mean(P, 0)                              # (n_va, H, 3)

    # ── q̂ por lead (CQR) sobre val-2023 con obs real ─────────────────────────
    qhat = {}
    for h in LEADS:
        j = [i + h - 1 for i in va]
        y = np.array([obs[k] for k in j])
        m = np.isfinite(y)
        p10, p90 = PRva[m, h - 1, 0], PRva[m, h - 1, 2]
        E = np.maximum(p10 - y[m], y[m] - p90)
        n = len(E)
        qhat[h] = float(np.quantile(E, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n),
                                    method="higher"))
    pd.DataFrame([{"lead": h, "qhat_m3s": round(v, 3)} for h, v in qhat.items()]
                 ).to_csv(OUT / "141_qhat_por_lead.csv", index=False)
    log.info("q̂ por lead (m³/s): " + ", ".join(f"h{h}={v:+.2f}" for h, v in qhat.items()))

    # ── aplicar al TEST publicado (banda del dashboard) ──────────────────────
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    ra = fc[(fc.model == "RA-TFT")].dropna(subset=["obs", "p10", "p90"])
    filas = []
    for h in LEADS:
        s = ra[ra.lead == h]
        y, p10, p50, p90 = s.obs.values, s.p10.values, s.p50.values, s.p90.values
        p10c, p90c = p10 - qhat[h], p90 + qhat[h]
        p10c = np.minimum(p10c, p50); p90c = np.maximum(p90c, p50)   # sin cruce
        def cov(a, b): return 100 * np.mean((y >= a) & (y <= b))
        def crps3(lo, md, hi):
            t = 0.0
            for qv, arr in zip([0.1, 0.5, 0.9], [lo, md, hi]):
                e = y - arr
                t += np.mean(np.where(e >= 0, qv * e, (qv - 1) * e))
            return t / 3
        filas.append(dict(lead=h, N=len(s),
                          cobertura_antes=round(cov(p10, p90), 1),
                          cobertura_despues=round(cov(p10c, p90c), 1),
                          ancho_antes=round(float(np.mean(p90 - p10)), 2),
                          ancho_despues=round(float(np.mean(p90c - p10c)), 2),
                          CRPS_antes=round(float(crps3(p10, p50, p90)), 3),
                          CRPS_despues=round(float(crps3(p10c, p50, p90c)), 3)))
    res = pd.DataFrame(filas)
    res.to_csv(OUT / "141_conformal_banda.csv", index=False)
    log.info("\n" + res.to_string(index=False))

    # ── FIG14 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax = axes[0]
    ax.plot(res.lead, res.cobertura_antes, "-o", color="#0B6E8C", lw=2,
            label="RA-TFT publicado (gen4)")
    ax.plot(res.lead, res.cobertura_despues, "-o", color="#2E8B6F", lw=2.4,
            label="RA-TFT + conformal (val-2023)")
    ax.axhline(80, color="#33414C", ls=":", lw=1.2)
    ax.text(13.8, 80.8, "nominal 80 %", ha="right", fontsize=9)
    ax.set_ylim(40, 100); ax.set_ylabel("cobertura empírica q10–q90 (%)")
    ax.set_title("(a) Cobertura de banda en test 2024–2025")
    ax = axes[1]
    ax.plot(res.lead, res.CRPS_antes, "-o", color="#0B6E8C", lw=2, label="antes")
    ax.plot(res.lead, res.CRPS_despues, "-o", color="#2E8B6F", lw=2.4,
            label="después (conformal)")
    ax.set_ylabel("CRPS (m³/s; menor = mejor)")
    ax.set_title("(b) CRPS con la banda recalibrada (p50 intacto)")
    for ax in axes:
        ax.set_xticks(LEADS); ax.set_xlabel("horizonte (días)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("FIG14 · Ablación ABL-gen5(b): recalibración conformal CQR de la banda "
                 "del RA-TFT\n(calibrada SOLO con val-2023, checkpoints congelados de la "
                 "release gen4 — cero reentrenamiento)", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG14_conformal_banda.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("FIG14 + 141_conformal_banda.csv listos")


if __name__ == "__main__":
    main()
