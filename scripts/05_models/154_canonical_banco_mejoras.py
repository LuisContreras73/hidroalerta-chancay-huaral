#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 154 — BANCO completo de mejoras aplicadas INDIVIDUALMENTE al TFT canónico.

Espeja el banco que corrimos sobre nuestro RA-TFT (142/143), ahora sobre el backbone
CANÓNICO, una mejora a la vez, para medir cuáles TRANSFIEREN al mejor backbone:
  preRMS    — RMSNorm pre-atención + final (142)
  sandwich  — pre + post-atención + final RMS (143, ganó en RA-TFT)
  qknorm    — RMSNorm en Q,K por cabeza dentro de la atención (143, dañó al RA-TFT)
  arcsinh   — transformación arcsinh del canal Q en la entrada (143)
  densq7    — cabeza de 7 cuantiles monótonos (143)
  alertw    — pérdida pinball ponderada a caudales altos (143)
(RevIN ya se probó en 148: dañó al canónico. Se referencia en la tabla.)

Protocolo A/B idéntico: backbone canónico intacto, mismos datos/split/hparams
(intv_h14)/seeds 0/1/2. Baseline = canónico puro (145). Artefactos: 154_*.

Salida: outputs/ml_Q/154_canonical_banco_mejoras.csv
Run en .venv313 (SÓLO con GPU libre; es largo: 6 variantes × 3 seeds):
    python scripts/05_models/154_canonical_banco_mejoras.py
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
log = logging.getLogger("cbanco154")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")
T = C.T
R = C.R
DEV = C.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]
Q7 = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


class QKNormIMHA(C.IMHA):
    """IMHA del canónico + RMSNorm en Q y K por cabeza (variante qknorm)."""
    def __init__(self, d, heads, drop=0.1):
        super().__init__(d, heads, drop)
        self.rq = nn.RMSNorm(self.dh); self.rk = nn.RMSNorm(self.dh)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        qh = self.rq(self.q(q).view(B, Lq, self.h, self.dh)).transpose(1, 2)
        kh = self.rk(self.k(k).view(B, Lk, self.h, self.dh)).transpose(1, 2)
        vs = self.v(v).unsqueeze(1)
        sc = qh @ kh.transpose(-1, -2) / self.dh ** 0.5
        if mask is not None:
            sc = sc.masked_fill(mask, -1e9)
        att = torch.softmax(sc, dim=-1)
        return self.o((self.drop(att) @ vs).mean(1)), att.mean(1)


class TFTCanonMod(C.TFTCanonical):
    """Canónico con UNA mejora aplicada (mod)."""
    def __init__(self, *a, mod="none", **k):
        super().__init__(*a, **k)
        self.mod = mod
        hid = self.norm_lstm.normalized_shape[0]
        if mod in ("prerms", "sandwich"):
            self.norm_lstm = nn.RMSNorm(hid); self.norm_att = nn.RMSNorm(hid)
            self.pre_attn = nn.RMSNorm(hid)
        if mod == "qknorm":
            heads = self.attn.h
            self.attn = QKNormIMHA(hid, heads, 0.1)
        if mod == "densq7":
            self.head = nn.Linear(hid, 7)

    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        if self.mod == "arcsinh":
            a_ = torch.asinh(q_past)
            qn = ((a_ - a_.mean(1, keepdim=True)) / (a_.std(1, keepdim=True) + 1e-5)).unsqueeze(-1)
        else:
            qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past); sf, _ = self.vsn_fut(x_future)
        ctx = self.static.expand(B, -1)
        e, (hN, cN) = self.enc_lstm(sp); d, _ = self.dec_lstm(sf, (hN, cN))
        seq = self._glu_add_norm(torch.cat([e, d], 1), torch.cat([sp, sf], 1), self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1).view(1, L, L)
        qkv = self.pre_attn(seq) if self.mod in ("prerms", "sandwich") else seq
        a, _ = self.attn(qkv, qkv, qkv, mask=mask)
        seq = self._glu_add_norm(a, seq, self.gate_att, self.norm_att)
        seq = self.ff(seq)
        dec = seq[:, -self.H:, :]
        raw = self.head(dec)
        if self.mod == "densq7":
            p50 = raw[..., 3]
            p25 = p50 - F.softplus(raw[..., 2]); p10 = p25 - F.softplus(raw[..., 1])
            p05 = p10 - F.softplus(raw[..., 0])
            p75 = p50 + F.softplus(raw[..., 4]); p90 = p75 + F.softplus(raw[..., 5])
            p95 = p90 + F.softplus(raw[..., 6])
            qs = torch.stack([p05, p10, p25, p50, p75, p90, p95], -1)
        else:
            p50 = raw[..., 0]
            p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
            qs = torch.stack([p10, p50, p90], -1)
        if self.mod == "arcsinh":
            a_ = torch.asinh(q_past); mu_, sd_ = a_.mean(1, keepdim=True), a_.std(1, keepdim=True) + 1e-5
            return torch.stack([torch.sinh(torch.clamp(qs[..., i] * sd_ + mu_, -15, 15))
                                for i in range(qs.shape[-1])], -1)
        return qs * self.q_sd + self.q_mu


def pinball_n(pred, y, levels):
    t = 0.0
    for i, qv in enumerate(levels):
        e = y - pred[..., i]
        t = t + torch.mean(torch.clamp(e, min=0) * qv + torch.clamp(-e, min=0) * (1 - qv))
    return t / len(levels)


def pinball_alertw(pred, y):
    w = 1.0 + 2.0 * (y >= 0.8 * Q90).float()
    t = 0.0
    for i, qv in enumerate([0.1, 0.5, 0.9]):
        e = y - pred[..., i]
        t = t + torch.mean(w * (torch.clamp(e, min=0) * qv + torch.clamp(-e, min=0) * (1 - qv)))
    return t / 3


def train_loss(model, dl, Xv, lr, wd, loss_fn, max_ep=180, pat=25):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    best, bs, bad = np.inf, None, 0
    for ep in range(max_ep):
        model.train()
        for xb, qb, fb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv[0].to(DEV), Xv[1].to(DEV), Xv[2].to(DEV)), Xv[3].to(DEV)).item()
        if vl < best - 1e-4:
            best, bad = vl, 0; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    if bs:
        model.load_state_dict(bs)
    return model


# nombre → (mod, loss, idx_para_extraer_10/50/90)
VARIANTES = {
    "canónico+preRMS":   ("prerms",  "p3", [0, 1, 2]),
    "canónico+sandwich": ("sandwich", "p3", [0, 1, 2]),
    "canónico+qknorm":   ("qknorm",  "p3", [0, 1, 2]),
    "canónico+arcsinh":  ("arcsinh", "p3", [0, 1, 2]),
    "canónico+densq7":   ("densq7",  "p7", [1, 3, 5]),
    "canónico+alertw":   ("none",    "aw", [0, 1, 2]),
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
    LOSS = {"p3": lambda p, y: pinball_n(p, y, [0.1, 0.5, 0.9]),
            "p7": lambda p, y: pinball_n(p, y, Q7), "aw": pinball_alertw}
    log.info(f"train {len(tr)} · test {len(te)} · variantes: {list(VARIANTES)}")

    rows = []
    for nombre, (mod, lk, idx) in VARIANTES.items():
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = TFTCanonMod(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                             H=H, drop=bp["drop"], enc=bp["enc"], q_mu=qmu, q_sd=qsd, mod=mod).to(DEV)
            mo = train_loss(mo, dl, Xva, bp["lr"], bp["wd"], LOSS[lk])
            tag = nombre.replace("canónico+", "").replace("ó", "o")
            torch.save(mo.state_dict(), OUT / f"154_can_{tag}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy()[:, :, idx])
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(model=nombre, h=h, **C.met(o, PR[:, h - 1, :])))
        log.info(f"✔ {nombre}")

    nuevo = pd.DataFrame(rows)
    cols = ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]
    piezas = [nuevo]
    c145 = pd.read_csv(OUT / "145_canonical_tft.csv")
    piezas.append(c145[(c145.model == "TFT-canónico") & c145.h.isin(LEADS)])
    if (OUT / "148_canonical_plus.csv").exists():
        c148 = pd.read_csv(OUT / "148_canonical_plus.csv")
        piezas.append(c148[(c148.model == "canónico+RevIN") & c148.h.isin(LEADS)])
    tab = pd.concat([p[[c for c in cols if c in p.columns]] for p in piezas], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "154_canonical_banco_mejoras.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 154_canonical_banco_mejoras.csv")


if __name__ == "__main__":
    main()
