#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 157 — Sandwich RMS VERDADERO (pre + post-atención + final) sobre el canónico.

Corrige el bug del 154: allí "preRMS" y "sandwich" eran byte-idénticos (misma rama:
solo pre-RMS + swap de add&norm a RMS; faltaba el post-atención distinto). El sandwich
real (como en 143 para el RA-TFT) aplica RMSNorm a la SALIDA de la atención ANTES del
residual con compuerta — ese es el "post" que faltaba.

Variante: canónico+sandwichReal = pre_attn RMS + rms_post(a) + norm_att RMS (final).
A/B idéntico: mismos datos/split/hparams (intv_h14)/seeds. Baseline = canónico puro (145)
y el pre-solo del 154 (referencia). Artefactos: 157_*.

Salida: outputs/ml_Q/157_canonical_sandwich_real.csv
Run en .venv313 (GPU): python scripts/05_models/157_canonical_sandwich_real.py
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
log = logging.getLogger("swreal157")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")
T = C.T; R = C.R; DEV = C.DEV; H = 14; LEADS = [1, 3, 7, 14]


class TFTSandwichReal(C.TFTCanonical):
    """Canónico + sandwich RMS genuino: pre_attn + post-atención + final."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        hid = self.norm_lstm.normalized_shape[0]
        self.norm_lstm = nn.RMSNorm(hid)
        self.norm_att = nn.RMSNorm(hid)           # final (en el add&norm)
        self.pre_attn = nn.RMSNorm(hid)           # pre-QKV
        self.rms_post = nn.RMSNorm(hid)           # POST-atención (el que faltaba en 154)

    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past); sf, _ = self.vsn_fut(x_future)
        ctx = self.static.expand(B, -1)
        e, (hN, cN) = self.enc_lstm(sp); d, _ = self.dec_lstm(sf, (hN, cN))
        seq = self._glu_add_norm(torch.cat([e, d], 1), torch.cat([sp, sf], 1),
                                 self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1).view(1, L, L)
        a, _ = self.attn(self.pre_attn(seq), self.pre_attn(seq), self.pre_attn(seq), mask=mask)
        a = self.rms_post(a)                      # ← POST-atención (la diferencia real)
        seq = self._glu_add_norm(a, seq, self.gate_att, self.norm_att)
        seq = self.ff(seq)
        raw = self.head(seq[:, -self.H:, :])
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], -1) * self.q_sd + self.q_mu


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
    P = []
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
        mo = TFTSandwichReal(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                             H=H, drop=bp["drop"], enc=bp["enc"], q_mu=qmu, q_sd=qsd).to(DEV)
        mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
        torch.save(mo.state_dict(), OUT / f"157_sandwichreal_seed{s}.pt")
        with torch.no_grad():
            P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        log.info(f"  seed {s} OK")
    PR = np.mean(P, 0)
    rows = []
    for h in LEADS:
        j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
        rows.append(dict(model="canónico+sandwichReal", h=h, **C.met(o, PR[:, h - 1, :])))
    nuevo = pd.DataFrame(rows)
    c145 = pd.read_csv(OUT / "145_canonical_tft.csv")
    base = c145[(c145.model == "TFT-canónico") & c145.h.isin(LEADS)]
    cols = ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]
    tab = pd.concat([base[[c for c in cols if c in base.columns]], nuevo], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "157_canonical_sandwich_real.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    base14 = base[base.h == 14].NSE.values[0]; nu14 = nuevo[nuevo.h == 14].NSE.values[0]
    log.info(f"h14: canónico puro {base14:.3f} vs sandwich-REAL {nu14:.3f} (Δ{nu14-base14:+.3f}) · pre-solo(154)=0.582")


if __name__ == "__main__":
    main()
