#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 142 — Ablación ABL-gen5(g): pre-RMSNorm (receta de los fundacionales) en RA-TFT.

Verificado en los pesos locales (2026-07-09):
  - TimesFM 2.5 bloque: pre_attn_ln/post_attn_ln + query_ln/key_ln (QK-norm) +
    pre_ff_ln/post_ff_ln — todo RMSNorm (sandwich + QK-norm).
  - Chronos-2 (T5): layer_norm RMS al INICIO de cada subcapa (pre-norm).
  - RA-TFT gen4: UNA nn.LayerNorm POST-residual tras la cross-attention.

Hipótesis (Zhang & Sennrich 2019 — RMSNorm; Xiong et al. 2020 — Pre-LN estabiliza
gradientes): mover a pre-norm RMS antes del Q/K/V de la cross-attention + norma
final pre-cabeza mejora la estabilidad del entrenamiento y las colas.

Diseño A/B honesto: MISMOS hiperparámetros (estudio intv_h14), MISMOS seeds (0,1,2),
MISMO split y datos; solo cambia el esquema de normalización. Baseline = release
congelada gen4_2026-07-02 (no se recomputa). Artefactos append-only: 142_*.

Salidas: outputs/ml_Q/142_ablacion_rmsnorm.csv (+ checkpoints 142_preRMS_seed*.pt)

Run en .venv313:
    python scripts/05_models/142_ablacion_rmsnorm.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
REL = ROOT / "models/releases/gen4_2026-07-02"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rms142")
optuna.logging.set_verbosity(optuna.logging.WARNING)

spec = importlib.util.spec_from_file_location(
    "t114", ROOT / "scripts/05_models/114_tft_intensive.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R = T.R
DEV = R.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]


class RATFT_preRMS(R.RATFT):
    """RA-TFT con la receta de normalización de los fundacionales:
    RMSNorm PRE-atención en la vía query (decoder) y key/value (encoder) —
    'antes del Q K V' — residual limpio sin post-LN, y RMSNorm final pre-cabeza."""

    def __init__(self, n_past, n_future, hid=64, heads=4, H=7, drop=0.2,
                 use_future=True):
        super().__init__(n_past, n_future, hid=hid, heads=heads, H=H, drop=drop,
                         use_future=use_future)
        self.rms_q = nn.RMSNorm(hid)
        self.rms_kv = nn.RMSNorm(hid)
        self.rms_out = nn.RMSNorm(hid)

    def forward(self, x_past, q_past, x_future):
        qn = self.revin.norm(q_past)
        xp = torch.cat([x_past, qn.unsqueeze(-1)], dim=-1) \
            if x_past.shape[-1] != 0 else qn.unsqueeze(-1)
        e = self.enc_proj(xp); e, (hN, cN) = self.enc_lstm(e)
        if not self.use_future:
            x_future = torch.zeros_like(x_future)
        d = self.dec_proj(x_future); d, _ = self.dec_lstm(d, (hN, cN))
        a, _ = self.cross(self.rms_q(d), self.rms_kv(e), self.rms_kv(e))  # pre-norm QKV
        h = d + self.drop(a) * self.gate(a)                               # residual limpio
        h = self.rms_out(h)                                               # norma final pre-cabeza
        raw = self.head(h)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([self.revin.denorm(p10), self.revin.denorm(p50),
                            self.revin.denorm(p90)], -1)


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
    kge = 1 - np.sqrt((r - 1) ** 2 + (p2.std() / o2.std() - 1) ** 2
                      + (p2.mean() / o2.mean() - 1) ** 2)
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
    tr = [i for i in range(EM, len(df) - H)
          if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    va = [i for i in range(EM, len(df) - H)
          if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31")
          and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6,
          "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · enc={bp['enc']}")

    P = []
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
        mo = RATFT_preRMS(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                          H=H, drop=bp["drop"], use_future=True).to(DEV)
        mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
        torch.save(mo.state_dict(), OUT / f"142_preRMS_seed{s}.pt")
        with torch.no_grad():
            P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        log.info(f"seed {s}: entrenado y evaluado")
    PR = np.mean(P, 0)

    rows = []
    for h in LEADS:
        j = [i + h - 1 for i in te]
        o = np.array([obs[k] for k in j])
        rows.append(dict(model="RA-TFT+preRMS", h=h, **met(o, PR[:, h - 1, :])))
    nuevo = pd.DataFrame(rows)

    base = pd.read_csv(REL / "125_full_metrics_corrected.csv")
    base = base[(base.model == "TFT-honesto") & base.h.isin(LEADS)]
    base = base.assign(model="RA-TFT (gen4, congelado)")[
        ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]]
    tab = pd.concat([base, nuevo], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "142_ablacion_rmsnorm.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 142_ablacion_rmsnorm.csv + 142_preRMS_seed{0,1,2}.pt")


if __name__ == "__main__":
    main()
