#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 155 — Componentes posicionales/modernos sobre el TFT canónico (RoPE, etc.).

Amplía el banco 154 con componentes que faltaban, verificados en los pesos de los
fundacionales (TimesFM usa rotary_position_embedding + PerDimScale):
  rope     — Rotary Position Embeddings (Su et al. 2021) en Q,K de la atención.
  swiglu   — FFN con SwiGLU en vez de la GRN estándar (activación tipo LLaMA/PaLM).

MATIZ HONESTO a declarar en el paper: nuestros modelos tienen LSTM enc/dec, que ya
codifica el ORDEN de la secuencia; RoPE es esencial en modelos PURO-atención (TimesFM
no tiene recurrencia). Aquí RoPE puede aportar poco — esta ablación lo CONFIRMA o
DESMIENTE, no lo asume.

Protocolo A/B idéntico: backbone canónico intacto, mismos datos/split/hparams/seeds.
Baseline = canónico puro (145). Artefactos append-only: 155_*.

Salida: outputs/ml_Q/155_canonical_rope.csv
Run en .venv313 (SÓLO con GPU libre; tras 154):
    python scripts/05_models/155_canonical_rope.py
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rope155")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")
T = C.T; R = C.R; DEV = C.DEV; Q90 = R.Q90
H = 14; LEADS = [1, 3, 7, 14]


class RoPEIMHA(C.IMHA):
    """IMHA del canónico con RoPE aplicado a Q y K por cabeza."""
    def _rope(self, x, L):                      # x: (B, h, L, dh)
        dh = x.shape[-1]; dev = x.device
        inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=dev).float() / dh))
        ang = torch.outer(torch.arange(L, device=dev).float(), inv)   # (L, dh/2)
        cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        qh = self._rope(self.q(q).view(B, Lq, self.h, self.dh).transpose(1, 2), Lq)
        kh = self._rope(self.k(k).view(B, Lk, self.h, self.dh).transpose(1, 2), Lk)
        vs = self.v(v).unsqueeze(1)
        sc = qh @ kh.transpose(-1, -2) / self.dh ** 0.5
        if mask is not None:
            sc = sc.masked_fill(mask, -1e9)
        att = torch.softmax(sc, dim=-1)
        return self.o((self.drop(att) @ vs).mean(1)), att.mean(1)


class SwiGLU(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.w = nn.Linear(d, 2 * d); self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop); self.norm = nn.LayerNorm(d)

    def forward(self, x, c=None):
        h = self.w(x); a, b = h.chunk(2, -1)
        return self.norm(x + self.drop(self.o(F.silu(a) * b)))


class TFTCanonModern(C.TFTCanonical):
    def __init__(self, *a, mod="rope", **k):
        super().__init__(*a, **k)
        self.mod = mod
        hid = self.norm_lstm.normalized_shape[0]
        if mod == "rope":
            self.attn = RoPEIMHA(hid, self.attn.h, 0.1)
        if mod == "swiglu":
            self.ff = SwiGLU(hid, 0.1)


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
                MAE=round(float(np.mean(np.abs(o2 - p2))), 2), CRPS=round(float(crps3(o2, qp2)), 3),
                POD=round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
                FAR=round(fp / (tp + fp), 3) if (tp + fp) else 0.0)


VARIANTES = {"canónico+RoPE": "rope", "canónico+SwiGLU": "swiglu"}


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
    for nombre, mod in VARIANTES.items():
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = TFTCanonModern(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                                H=H, drop=bp["drop"], enc=bp["enc"], q_mu=qmu, q_sd=qsd, mod=mod).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"155_{mod}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(model=nombre, h=h, **met(o, PR[:, h - 1, :])))
        log.info(f"✔ {nombre}")
    tab = pd.DataFrame(rows).sort_values(["h", "model"])
    tab.to_csv(OUT / "155_canonical_rope.csv", index=False)
    log.info("\n" + tab.to_string(index=False))


if __name__ == "__main__":
    main()
