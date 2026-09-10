#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 150 — Barrido de tamaño (hidden_size) del TFT canónico → ¿con cuál gana?

Motivación (pregunta del usuario): el canónico a hid=128 (heredado del estudio del
RA-TFT) da 1,34M params — ~40× el TFT de referencia (hid≈16, ~24K). Los params
escalan ~hid². Con ~15k muestras y 1 estación, un modelo grande puede SOBREAJUSTAR;
uno pequeño puede rendir igual o mejor. Este barrido lo mide, comparando a TAMAÑOS
comparables con el TFT original — sin Optuna (solo se cambia hid, todo lo demás fijo).

Variantes: TFT canónico a hid ∈ {16, 32, 64}. hid=128 ya está en 145 (se añade como ref).
Protocolo A/B idéntico: mismos datos, split, resto de hparams (intv_h14), seeds 0/1/2.
Artefactos append-only: 150_*.

Salida: outputs/ml_Q/150_canonical_hidsweep.csv (hid × horizonte: NSE/CRPS + params)

Run en .venv313 (SÓLO con GPU libre):
    python scripts/05_models/150_canonical_hidsweep.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("hidsweep150")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


T = _load("scripts/05_models/114_tft_intensive.py", "t114")
C = _load("scripts/05_models/145_canonical_tft.py", "c145")
R = T.R
DEV = R.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]
HIDS = [16, 32, 64]      # 128 ya está medido en 145_canonical_tft.csv


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
    oa, pa = o2 >= Q90, p2 >= Q90
    tp, fp, fn = np.sum(oa & pa), np.sum(~oa & pa), np.sum(oa & ~pa)
    return dict(N=int(m.sum()), NSE=round(nse, 3), NSE_sqrt=round(nsq, 3),
                MAE=round(float(np.mean(np.abs(o2 - p2))), 2),
                CRPS=round(float(crps3(o2, qp2)), 3),
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
    qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
    qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))

    rows = []
    for hid in HIDS:
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = C.TFTCanonical(len(R.PAST), len(R.FUT), hid=hid, heads=2, H=H,
                                drop=bp["drop"], enc=bp["enc"], q_mu=qmu, q_sd=qsd).to(DEV)
            npar = sum(p.numel() for p in mo.parameters())
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"150_canonico_hid{hid}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(hid=hid, params_K=round(npar / 1e3), h=h, **met(o, PR[:, h - 1, :])))
        log.info(f"✔ hid={hid} ({npar/1e3:.0f}K) completo")

    nuevo = pd.DataFrame(rows)
    # ref hid=128 desde 145
    c145 = pd.read_csv(OUT / "145_canonical_tft.csv")
    c145 = c145[(c145.model == "TFT-canónico") & c145.h.isin(LEADS)].copy()
    c145["hid"] = 128; c145["params_K"] = 1342
    cols = ["hid", "params_K", "h", "N", "NSE", "NSE_sqrt", "MAE", "CRPS", "POD", "FAR"]
    keep = [c for c in cols if c in c145.columns]
    tab = pd.concat([nuevo[cols], c145[keep]], ignore_index=True).sort_values(["h", "hid"])
    tab.to_csv(OUT / "150_canonical_hidsweep.csv", index=False)
    log.info("\n=== NSE por hidden_size (h7 y h14) ===")
    for h in [7, 14]:
        s = tab[tab.h == h].sort_values("hid")
        log.info(f"h={h}:\n" + s[["hid", "params_K", "NSE", "CRPS"]].to_string(index=False))
    log.info("Guardado: 150_canonical_hidsweep.csv")


if __name__ == "__main__":
    main()
