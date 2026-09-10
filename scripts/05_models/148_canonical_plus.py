#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 148 — RA-TFT/c: apila nuestras mejoras sobre el backbone CANÓNICO.

Ablación escalonada (cada fila añade UNA mejora al canónico) — los experimentos
decisivos #1 y #3 del diseño (docs/MODELO_FINAL_DISENO.md), que NO requieren GFS:
  canónico           = 145 (VSN + GRN + Masked Interp. MHSA + norma global, SIN RevIN)
  canónico +RevIN    = norma por régimen del canal Q (nuestra «R»; en el canónico
                       reemplaza la estándar global — ablación 144 la probó esencial
                       en nuestra arq., y aquí debe estabilizar la alta varianza por
                       seed del canónico y subir skill)
  canónico +RevIN +sandwich = además normalización sandwich RMS (mejor add del banco
                       143, +0,145 h14 en nuestra arq.) en los add&norm.

Protocolo A/B idéntico (mismos datos, split, hparams intv_h14, seeds 0/1/2, T.train_es).
Baseline de referencia = canónico (145) + RA-TFT gen4 congelado. Artefactos: 148_*.
GFS (mejora #2, la «A») queda para después del paso de features B11.

Salida: outputs/ml_Q/148_canonical_plus.csv (+ checkpoints 148_<variante>_seed*.pt)

Run en .venv313:
    python scripts/05_models/148_canonical_plus.py
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
log = logging.getLogger("cplus148")
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


class TFTCanonicalPlus(C.TFTCanonical):
    """Canónico + RevIN (régimen) y opcionalmente + sandwich RMS."""

    def __init__(self, *a, revin=True, sandwich=False, **k):
        super().__init__(*a, **k)
        self.use_revin, self.use_sandwich = revin, sandwich
        hid = self.norm_lstm.normalized_shape[0]
        if revin:
            self.revin = R.RevIN()
        if sandwich:
            self.norm_lstm = nn.RMSNorm(hid)
            self.norm_att = nn.RMSNorm(hid)
            self.pre_attn = nn.RMSNorm(hid)

    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        if self.use_revin:
            qn = self.revin.norm(q_past).unsqueeze(-1)
        else:
            qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past)
        sf, _ = self.vsn_fut(x_future)
        ctx = self.static.expand(B, -1)
        e, (hN, cN) = self.enc_lstm(sp)
        d, _ = self.dec_lstm(sf, (hN, cN))
        seq_in = torch.cat([sp, sf], dim=1)
        seq = torch.cat([e, d], dim=1)
        seq = self._glu_add_norm(seq, seq_in, self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        qkv = self.pre_attn(seq) if self.use_sandwich else seq
        a, _ = self.attn(qkv, qkv, qkv, mask=mask.view(1, L, L))
        seq = self._glu_add_norm(a, seq, self.gate_att, self.norm_att)
        seq = self.ff(seq)
        dec = seq[:, -self.H:, :]
        raw = self.head(dec)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        qs = torch.stack([p10, p50, p90], -1)
        if self.use_revin:
            return torch.stack([self.revin.denorm(qs[..., i]) for i in range(3)], -1)
        return qs * self.q_sd + self.q_mu


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


VARIANTES = {
    "canónico+RevIN": dict(revin=True, sandwich=False),
    "canónico+RevIN+sandwich": dict(revin=True, sandwich=True),
}


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
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · enc={bp['enc']}")

    rows = []
    for nombre, kw in VARIANTES.items():
        P, seed_nse14 = [], []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = TFTCanonicalPlus(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                                  H=H, drop=bp["drop"], enc=bp["enc"], q_mu=qmu, q_sd=qsd, **kw).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"148_{nombre.replace('+', '_').replace('ó','o')}_seed{s}.pt")
            with torch.no_grad():
                pr = mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy()
            P.append(pr)
            j14 = [i + 13 for i in te]; o14 = np.array([obs[k] for k in j14])
            seed_nse14.append(met(o14, pr[:, 13, :])["NSE"])
            log.info(f"  {nombre} seed {s} OK (NSE14={seed_nse14[-1]:.3f})")
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(model=nombre, h=h, **met(o, PR[:, h - 1, :])))
        log.info(f"✔ {nombre}: NSE14 por seed {seed_nse14} (ensemble {rows[-1]['NSE']})")

    nuevo = pd.DataFrame(rows)
    # referencias: canónico (145) + RA-TFT gen4 (release)
    refs = []
    c145 = pd.read_csv(OUT / "145_canonical_tft.csv")
    refs.append(c145[(c145.model == "TFT-canónico") & c145.h.isin(LEADS)])
    g4 = pd.read_csv(REL / "125_full_metrics_corrected.csv")
    g4 = g4[(g4.model == "TFT-honesto") & g4.h.isin(LEADS)].assign(model="RA-TFT (gen4)")
    cols = ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]
    tab = pd.concat([g4[cols], refs[0][cols], nuevo], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "148_canonical_plus.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 148_canonical_plus.csv")


if __name__ == "__main__":
    main()
