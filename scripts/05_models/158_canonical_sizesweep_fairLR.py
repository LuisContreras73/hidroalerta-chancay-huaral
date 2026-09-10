#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 158 (A7) — Barrido de TAMAÑO del TFT canónico con LR JUSTO POR TAMAÑO.

Corrige el confound del 150: allí los canónicos chicos (hid 16/32/64) se entrenaron
con el LR tuneado para hid128 → sub-entrenados, curva de escalado NO válida (no
monótona: hid64 peor que hid16). Aquí cada tamaño elige su PROPIO LR en validación
(pérdida pinball), luego se entrena con 3 seeds. Protocolo uniforme para todos →
curva de escalado / Pareto publicable (performance vs #parámetros).

PROTOCOLO (idéntico salvo hid y lr):
  datos/split/enc/heads/drop/wd/bs = estudio intv_h14 (mismos que 145).
  Para cada hid ∈ {16,32,64,128}:
    1) SELECCIÓN: entrena seed 0 con cada lr ∈ LRS; mide pinball de validación;
       guarda también la predicción de test de ese seed 0. Elige lr* = argmin val.
    2) FINAL: reusa seed 0 (lr*) + entrena seeds 1,2 → media de 3 → métricas de test.
  hid128 se corre también por la grilla para consistencia del protocolo; el modelo
  de PRODUCCIÓN (hid128 tuneado por Optuna en 145) da NSE h1 0,943 y se anota aparte.

Salida: outputs/ml_Q/158_canonical_sizesweep_fairLR.csv (escrita incrementalmente
por tamaño) + checkpoints 158_hid{hid}_seed{s}.pt. Append-only (nº nuevo).
Run en .venv313: python scripts/05_models/158_canonical_sizesweep_fairLR.py

Nota: convive con otro entrenamiento en GPU (proyecto Olas de Calor) → ambos van
más lentos por contención de cómputo, pero caben en memoria (canónico ≤ hid128 ~1,5 GB).
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
log = logging.getLogger("sizesweep158")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")
T = C.T; R = C.R; DEV = C.DEV; H = C.H; LEADS = C.LEADS

SIZES = [16, 32, 64, 128]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def val_pinball(mo, Xv):
    mo.eval()
    with torch.no_grad():
        return float(R.pinball(mo(Xv[0].to(DEV), Xv[1].to(DEV), Xv[2].to(DEV)),
                               Xv[3].to(DEV)).item())


def test_pred(mo, Xte):
    mo.eval()
    with torch.no_grad():
        return mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy()


def train_one(hid, heads, drop, wd, bs, lr, seed, Xtr, Xva):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(TensorDataset(*Xtr), batch_size=bs, shuffle=True)
    mo = C.TFTCanonical(len(R.PAST), len(R.FUT), hid=hid, heads=heads,
                        H=H, drop=drop, q_mu=QMU, q_sd=QSD).to(DEV)
    return T.train_es(mo, dl, Xva, lr, wd)


def main():
    global QMU, QSD
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    LRS = sorted({round(bp["lr"], 8), 3e-4, 1e-3, 3e-3})
    log.info(f"grilla LR por tamaño: {LRS}  (bp.lr={bp['lr']:.2e}, heads={bp['heads']}, "
             f"drop={bp['drop']}, wd={bp['wd']:.1e}, bs={bp['bs']}, enc={bp['enc']})")

    df = R.build(H); dts = df.index; obs = df["obs"].values; q = df["q"].values
    EM = bp["enc"]
    tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    va = [i for i in range(EM, len(df) - H) if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    q_tr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
    QMU, QSD = float(np.mean(q_tr)), float(np.std(q_tr) + 1e-6)
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)}")

    rows = []
    for hid in SIZES:
        heads = bp["heads"] if hid % bp["heads"] == 0 else 2
        # 1) selección de LR (seed 0)
        sel = {}
        for lr in LRS:
            mo = train_one(hid, heads, bp["drop"], bp["wd"], bp["bs"], lr, 0, Xtr, Xva)
            vp = val_pinball(mo, Xva)
            sel[lr] = dict(vp=vp, tp0=test_pred(mo, Xte),
                           state={k: v.cpu().clone() for k, v in mo.state_dict().items()},
                           P=n_params(mo))
            log.info(f"  hid{hid} lr={lr:.2e} → val pinball {vp:.4f}")
        lr_star = min(sel, key=lambda k: sel[k]["vp"])
        P = sel[lr_star]["P"]
        log.info(f"  ✔ hid{hid}: lr*={lr_star:.2e} ({P/1000:.0f}K params)")
        # 2) final: reusa seed0(lr*) + seeds 1,2
        torch.save(sel[lr_star]["state"], OUT / f"158_hid{hid}_seed0.pt")
        preds = [sel[lr_star]["tp0"]]
        for s in (1, 2):
            mo = train_one(hid, heads, bp["drop"], bp["wd"], bp["bs"], lr_star, s, Xtr, Xva)
            torch.save(mo.state_dict(), OUT / f"158_hid{hid}_seed{s}.pt")
            preds.append(test_pred(mo, Xte))
        PR = np.mean(preds, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(hid=hid, params_K=round(P / 1000), lr=lr_star, h=h, **C.met(o, PR[:, h - 1, :])))
        # escritura incremental (no perder progreso si se interrumpe)
        pd.DataFrame(rows).to_csv(OUT / "158_canonical_sizesweep_fairLR.csv", index=False)
        log.info(f"  guardado parcial tras hid{hid}")

    tab = pd.DataFrame(rows)
    log.info("\n" + tab.to_string(index=False))
    log.info("Comparar contra 150 (LR fijo) para ver cuánto sube cada chico con LR justo.")
    log.info("Producción hid128 (Optuna, 145): NSE h1 0,943 / h7 0,743 / h14 0,720.")


if __name__ == "__main__":
    main()
