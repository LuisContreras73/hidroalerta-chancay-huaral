#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 174 — Baselines RECURRENTES PUROS: LSTM y GRU encoder-decoder, solos.

Referencia mínima para el paper: ¿cuánto aporta la maquinaria del TFT (VSN, atención
interpretable, GRN, gating) por encima de una red recurrente plana? Aquí el modelo es
SOLO enc-dec recurrente (proyección lineal → celda → cabeza cuantílica); sin selección
de variables, sin atención, sin normas especiales. Aísla el efecto de la celda.

Honesto e idéntico al resto: hereda R.FUT=['sin','cos'] (calendario, patcheado por 114),
mismos datos/split (train ≤2022 GR4J · val 2023 · test 2024-25 aforo real), normalización
solo-train, harness T.train_es (pinball), seeds 0/1/2, HP base (estudio intv_h14).

Compara LSTM vs GRU y contra persistencia + climatología (y el canónico si su CSV existe).
Uso (.venv313, GPU): python scripts/05_models/174_recurrent_baselines.py --H 30
Salida: outputs/ml_Q/174_recurrent_baselines_H{H}.csv (+ checkpoints 174_*.pt)
"""
import argparse
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rec174")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")   # datos/met/R/T/DEV honestos
R, T, C, DEV = M.R, M.T, M.C, M.DEV
Q90 = R.Q90


class PureRec(nn.Module):
    """Enc-dec recurrente plano. kind='lstm'|'gru'. Sin VSN/atención/GRN — la referencia."""

    def __init__(self, n_past, n_future, hid=128, H=14, drop=0.2, q_mu=0.0, q_sd=1.0, kind="lstm", **_):
        super().__init__()
        self.H = H; self.is_lstm = (kind == "lstm")
        self.register_buffer("q_mu", torch.tensor(float(q_mu)))
        self.register_buffer("q_sd", torch.tensor(float(q_sd)))
        cell = nn.LSTM if self.is_lstm else nn.GRU
        self.enc_in = nn.Linear(n_past + 1, hid)      # +1 = canal Q normalizado
        self.dec_in = nn.Linear(n_future, hid)
        self.enc = cell(hid, hid, batch_first=True)
        self.dec = cell(hid, hid, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hid, 3)

    def forward(self, x_past, q_past, x_future):
        qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        e_in = torch.tanh(self.enc_in(past))
        _, st = self.enc(e_in)                         # LSTM: (hN,cN) · GRU: hN — ambos sirven al dec
        d, _ = self.dec(torch.tanh(self.dec_in(x_future)), st)
        raw = self.head(self.drop(d))
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], -1) * self.q_sd + self.q_mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=30)
    a = ap.parse_args()
    H = a.H
    M.H = H; M.LEADS = M.leads_for(H)          # alinear el módulo de datos al horizonte
    LEADS = M.LEADS
    sfx = "" if H == 14 else f"_H{H}"
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    (Xtr, Xva, Xte, te, obs), qmu, qsd = M.load_data(bp)
    log.info(f"H={H} leads={LEADS} · R.FUT={R.FUT} · train {len(Xtr[0])} · test {len(te)}")

    rows = []
    for kind in ("lstm", "gru"):
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = PureRec(len(R.PAST), len(R.FUT), hid=bp["hid"], H=H, drop=bp["drop"],
                         q_mu=qmu, q_sd=qsd, kind=kind).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"174_{kind}{sfx}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        PR = np.mean(P, 0)
        npar = sum(p.numel() for p in mo.parameters())
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(model=kind.upper(), h=h, params=npar, **C.met(o, PR[:, h - 1, :])))
        log.info(f"✔ {kind.upper()} (params={npar/1e3:.0f}K)")

    # referencias: persistencia + climatología (día-del-año, train ≤2022)
    df = R.build(H); dts = df.index; q = df["q"].values
    clim_src = pd.Series(q, index=dts)
    clim = clim_src[dts <= pd.Timestamp("2022-12-31")].groupby(
        clim_src[dts <= pd.Timestamp("2022-12-31")].index.dayofyear).mean()
    for h in LEADS:
        j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
        pp = np.array([q[i - 1] for i in te])                       # persistencia
        rows.append(dict(model="Persistencia", h=h, params=0, **C.met(o, np.stack([pp] * 3, 1))))
        pc = np.array([clim.get(dts[i + h - 1].dayofyear, np.nan) for i in te])  # climatología
        rows.append(dict(model="Climatología", h=h, params=0, **C.met(o, np.stack([pc] * 3, 1))))

    tab = pd.DataFrame(rows).sort_values(["h", "model"])
    tab.to_csv(OUT / f"174_recurrent_baselines{sfx}.csv", index=False)
    log.info("\n" + tab[tab.h.isin([1, LEADS[-1]])].to_string(index=False))
    log.info(f"Guardado: 174_recurrent_baselines{sfx}.csv")


if __name__ == "__main__":
    main()
