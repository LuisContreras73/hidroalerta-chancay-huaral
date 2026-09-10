#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 149 — Ablación de las variantes pendientes de arch_variants (tabla Δ).

Continúa el formato de tabla que ya usamos (142/143/144): quitar/cambiar UNA cosa →
cuánto sube/baja NSE y CRPS vs el RA-TFT gen4 congelado. Entrena las 4 variantes
que quedaron SOLO codificadas en arch_variants.py:
  RA-TFT[scalenorm]  — norma = un escalar aprendido (Nguyen & Salazar 2019)
  RA-TFT[rezero]     — sin norma, residual escalar init-0 (Bachlechner 2020)
  RA-TFT[gatedrms]   — RMSNorm + compuerta
  RA-TFT[maskedself] — cross-attention → MASKED self-attention (eje cross vs masked-self)

Protocolo A/B idéntico: mismos datos, split, hparams (intv_h14), seeds 0/1/2,
T.train_es. Baseline = RA-TFT gen4 (release congelada). Artefactos append-only: 149_*.

Salida: outputs/ml_Q/149_ablacion_variantes.csv

Run en .venv313 (lanzar SÓLO cuando la GPU esté libre — no competir con otros trains):
    python scripts/05_models/149_ablacion_variantes.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
REL = ROOT / "models/releases/gen4_2026-07-02"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("abl149")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


T = _load("scripts/05_models/114_tft_intensive.py", "t114")
V = _load("scripts/05_models/arch_variants.py", "av")
R = T.R
DEV = R.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]


def crps3(o, qp):
    t = 0.0
    for i, qv in enumerate([0.1, 0.5, 0.9]):
        e = o - qp[:, i]
        t += np.mean(np.where(e >= 0, qv * e, (qv - 1) * e))
    return t / 3


def met(o, qp):
    o = np.asarray(o, float); p = qp[:, 1]
    m = np.isfinite(o) & np.isfinite(p); o2, p2, qp2 = o[m], p[m], qp[m]
    nse = 1 - np.sum((o2 - p2) ** 2) / np.sum((o2 - o2.mean()) ** 2)
    so, sp = np.sqrt(np.clip(o2, 0, None)), np.sqrt(np.clip(p2, 0, None))
    nsq = 1 - np.sum((so - sp) ** 2) / np.sum((so - so.mean()) ** 2)
    r = np.corrcoef(o2, p2)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (p2.std() / o2.std() - 1) ** 2 + (p2.mean() / o2.mean() - 1) ** 2)
    oa, pa = o2 >= Q90, p2 >= Q90
    tp, fp, fn = np.sum(oa & pa), np.sum(~oa & pa), np.sum(oa & ~pa)
    return dict(N=int(m.sum()), NSE=round(nse, 3), NSE_sqrt=round(nsq, 3),
                KGE=round(kge, 3), MAE=round(float(np.mean(np.abs(o2 - p2))), 2),
                CRPS=round(float(crps3(o2, qp2)), 3),
                CSI=round(tp / (tp + fp + fn), 3) if (tp + fp + fn) else 0.0,
                POD=round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
                FAR=round(fp / (tp + fp), 3) if (tp + fp) else 0.0)


def main():
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    df = R.build(H); dts = df.index; obs = df["obs"].values; q = df["q"].values
    EM = 90
    tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    va = [i for i in range(EM, len(df) - H) if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · enc={bp['enc']}")

    rows = []
    for nombre, (Cls, kw) in V.VARIANTES.items():
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = Cls(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                     H=H, drop=bp["drop"], use_future=True, **kw).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            tag = nombre.replace("RA-TFT[", "").replace("]", "")
            torch.save(mo.state_dict(), OUT / f"149_{tag}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(model=nombre, h=h, **met(o, PR[:, h - 1, :])))
        log.info(f"✔ {nombre} completo")

    nuevo = pd.DataFrame(rows)
    base = pd.read_csv(REL / "125_full_metrics_corrected.csv")
    base = base[(base.model == "TFT-honesto") & base.h.isin(LEADS)].assign(
        model="RA-TFT (gen4)")[
        ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]]
    tab = pd.concat([base, nuevo], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "149_ablacion_variantes.csv", index=False)
    # tabla Δ vs baseline (formato que pidió el usuario)
    b14 = base[base.h == 14].iloc[0]
    log.info("\n=== Δ vs RA-TFT gen4 a h14 (NSE / CRPS) ===")
    log.info(f"  baseline: NSE {b14.NSE} · CRPS {b14.CRPS}")
    for nombre in [n for n in V.VARIANTES]:
        r = nuevo[(nuevo.model == nombre) & (nuevo.h == 14)].iloc[0]
        log.info(f"  {nombre:20s} NSE {r.NSE:+.3f} (Δ{r.NSE-b14.NSE:+.3f}) · CRPS {r.CRPS:.3f} (Δ{r.CRPS-b14.CRPS:+.3f})")
    log.info("Guardado: 149_ablacion_variantes.csv")


if __name__ == "__main__":
    main()
