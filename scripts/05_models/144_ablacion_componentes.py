#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 144 — Ablación LEAVE-ONE-OUT de los componentes NATIVOS del RA-TFT.

Responde la exigencia Q1 "justifiquen que cada componente de su arquitectura gana
su lugar" (a diferencia del banco 142/143, que probó ideas AÑADIDAS de los
fundacionales; aquí se QUITA una pieza propia a la vez). Cada letra del acrónimo
y cada bloque de diseño tiene su fila:

  full       = arquitectura completa (reentrenada — chequeo de reproducción de gen4
               + piso de varianza por semilla)
  revin_off  = QUITA la R: normalización por-ventana (RevIN) → estándar GLOBAL de
               train. Aísla el aporte de la normalización por régimen.
  cross_off  = QUITA la A: el decoder NO atiende al encoder (solo su LSTM, inicializado
               con el estado final del encoder). Aísla la cross-attention anticipatoria.
  gate_off   = QUITA el gating estilo GRN: residual pleno (d + drop(a)) sin compuerta.
  nofuture   = QUITA las covariables futuras (calendario): decoder recibe ceros.

Protocolo A/B idéntico a gen4: MISMO T.train_es (pérdida pinball, AdamW, cosine,
early stopping), MISMOS hiperparámetros (estudio intv_h14), MISMOS seeds (0,1,2),
MISMO split (train ≤2022 · val 2023 · test 2024–25). Solo cambia el componente.
Baseline de referencia = release congelada gen4_2026-07-02. Artefactos: 144_*.

Salida: outputs/ml_Q/144_ablacion_componentes.csv (+ checkpoints 144_<abl>_seed*.pt)

Run en .venv313:
    python scripts/05_models/144_ablacion_componentes.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
REL = ROOT / "models/releases/gen4_2026-07-02"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("comp144")
optuna.logging.set_verbosity(optuna.logging.WARNING)

spec = importlib.util.spec_from_file_location(
    "t114", ROOT / "scripts/05_models/114_tft_intensive.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R = T.R
DEV = R.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]
ABLACIONES = ["full", "revin_off", "cross_off", "gate_off", "nofuture"]


class RATFT_abl(R.RATFT):
    """RA-TFT con un componente nativo desactivado (leave-one-out)."""

    def __init__(self, *a, ablate="full", q_mu=0.0, q_sd=1.0, **k):
        super().__init__(*a, **k)
        self.ablate = ablate
        self.register_buffer("q_mu", torch.tensor(float(q_mu)))
        self.register_buffer("q_sd", torch.tensor(float(q_sd)))

    def _norm(self, x):
        if self.ablate == "revin_off":
            return (x - self.q_mu) / self.q_sd          # global (no per-ventana)
        return self.revin.norm(x)

    def _denorm(self, y):
        if self.ablate == "revin_off":
            return y * self.q_sd + self.q_mu
        return self.revin.denorm(y)

    def forward(self, x_past, q_past, x_future):
        qn = self._norm(q_past)
        xp = torch.cat([x_past, qn.unsqueeze(-1)], dim=-1) \
            if x_past.shape[-1] != 0 else qn.unsqueeze(-1)
        e = self.enc_proj(xp); e, (hN, cN) = self.enc_lstm(e)
        xf = torch.zeros_like(x_future) if (not self.use_future or self.ablate == "nofuture") \
            else x_future
        d = self.dec_proj(xf); d, _ = self.dec_lstm(d, (hN, cN))
        if self.ablate == "cross_off":
            h = self.norm(d)                            # sin cross-attention
        else:
            a, _ = self.cross(d, e, e)
            resid = d + self.drop(a) if self.ablate == "gate_off" \
                else d + self.drop(a) * self.gate(a)
            h = self.norm(resid)
        raw = self.head(h)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([self._denorm(p10), self._denorm(p50), self._denorm(p90)], -1)


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
    q_tr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31")
              and np.isfinite(q[i])]]
    q_mu, q_sd = float(np.mean(q_tr)), float(np.std(q_tr) + 1e-6)
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6,
          "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · q global μ={q_mu:.1f} σ={q_sd:.1f}")

    rows = []
    for abl in ABLACIONES:
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = RATFT_abl(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                           H=H, drop=bp["drop"], use_future=True,
                           ablate=abl, q_mu=q_mu, q_sd=q_sd).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"144_{abl}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]
            o = np.array([obs[k] for k in j])
            rows.append(dict(model=f"RA-TFT[{abl}]", h=h, **met(o, PR[:, h - 1, :])))
        log.info(f"✔ {abl} completo")

    nuevo = pd.DataFrame(rows)
    base = pd.read_csv(REL / "125_full_metrics_corrected.csv")
    base = base[(base.model == "TFT-honesto") & base.h.isin(LEADS)].assign(
        model="RA-TFT (gen4, congelado)")[
        ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]]
    tab = pd.concat([base, nuevo], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "144_ablacion_componentes.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 144_ablacion_componentes.csv")


if __name__ == "__main__":
    main()
