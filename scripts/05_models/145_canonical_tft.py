#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 145 — TFT CANÓNICO (Lim et al. 2021) fiel + DLinear, como baselines JUSTOS.

Por qué (exigencia Q1): nuestro RA-TFT es una versión SIMPLIFICADA del TFT (cross-
attention estándar + compuerta sigmoide + LayerNorm). Un revisor pide comparar
contra el TFT REAL. Aquí se implementa el TFT canónico con sus componentes propios:
  - GRN (Gated Residual Network): ELU→dense→GLU→add&LayerNorm, contexto opcional.
  - VSN (Variable Selection Network): pesos softmax por-instante sobre variables,
    cada variable por su GRN. Da importancia de variable interpretable.
  - "Masked Interpretable Multi-Head Attention" (término EXACTO del paper, Fig. 2):
    self-attention temporal con máscara causal de decoder (get_decoder_mask: la
    posición t solo atiende a ≤ t) usando la Interpretable Multi-Head Attention del
    §4.5 (VALOR COMPARTIDO entre cabezas → pesos aditivos/interpretables). Verificado
    en arXiv 1912.09363 §4.5.3 + código google-research/tft (InterpretableMultiHead-
    Attention + get_decoder_mask). Aquí: torch.triu + valor compartido.
  - Enriquecimiento estático (aquí: contexto aprendido; single-station no tiene
    covariables estáticas reales — simplificación honesta, declarada).
  - LSTM enc/dec + salida cuantílica multi-horizonte.
Además DLinear (Zeng et al. 2023, "Are Transformers Effective for TS?"): el baseline
lineal que batió a muchos transformers — obligatorio para un paper riguroso.

Comparación JUSTA: MISMO harness (T.train_es, pinball), datos, split (train ≤2022 ·
val 2023 · test 2024-25), hiperparámetros base (estudio intv_h14), seeds (0,1,2).
El TFT canónico NO usa RevIN (normaliza Q con estándar global de train) — así la
comparación aísla arquitectura; RevIN es nuestra mejora (ver ablación 144).

Roadmap de mejoras encima del canónico (siguientes scripts, tras GFS):
  +RevIN → +decoder anticipatorio (GFS) → +sandwich RMS → +IMHA interpretable.
Baselines recientes pendientes (stubs documentados): PatchTST, iTransformer, N-HiTS.

Salida: outputs/ml_Q/145_canonical_tft.csv (+ checkpoints 145_<arch>_seed*.pt)
Módulo reutilizable: las clases GRN/VSN/IMHA/TFTCanonical/DLinear se importan.

Run en .venv313:
    python scripts/05_models/145_canonical_tft.py
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
log = logging.getLogger("tft145")
optuna.logging.set_verbosity(optuna.logging.WARNING)

spec = importlib.util.spec_from_file_location(
    "t114", ROOT / "scripts/05_models/114_tft_intensive.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R = T.R
DEV = R.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]
QS = [0.1, 0.5, 0.9]


# ── Componentes canónicos TFT (Lim et al. 2021) ───────────────────────────────
class GRN(nn.Module):
    """Gated Residual Network: ELU→dense→GLU→dropout→add&LayerNorm."""

    def __init__(self, d, drop=0.1, d_ctx=None):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.ctx = nn.Linear(d_ctx, d, bias=False) if d_ctx else None
        self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, 2 * d)          # GLU
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(d)

    def forward(self, a, c=None):
        eta = self.fc1(a)
        if self.ctx is not None and c is not None:
            eta = eta + self.ctx(c)
        eta = self.fc2(F.elu(eta))
        x = self.gate(self.drop(eta))
        glu = x[..., :x.shape[-1] // 2] * torch.sigmoid(x[..., x.shape[-1] // 2:])
        return self.norm(a + glu)


class VSN(nn.Module):
    """Variable Selection Network: cada variable→su GRN; pesos softmax por-instante."""

    def __init__(self, n_vars, d, drop=0.1):
        super().__init__()
        self.n = n_vars
        self.proj = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
        self.var_grn = nn.ModuleList([GRN(d, drop) for _ in range(n_vars)])
        self.sel = nn.Sequential(nn.Linear(n_vars * d, n_vars), )
        self.d = d

    def forward(self, x):                        # x: (B, L, n_vars)
        feats = [self.var_grn[i](self.proj[i](x[..., i:i + 1])) for i in range(self.n)]
        stacked = torch.stack(feats, dim=-2)     # (B, L, n_vars, d)
        flat = stacked.flatten(-2)               # (B, L, n_vars*d)
        w = torch.softmax(self.sel(flat), dim=-1).unsqueeze(-1)   # (B,L,n_vars,1)
        return (stacked * w).sum(-2), w.squeeze(-1)               # (B,L,d), (B,L,n_vars)


class IMHA(nn.Module):
    """Interpretable Multi-Head Attention: VALOR compartido entre cabezas."""

    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, self.dh)           # V COMPARTIDO (una cabeza de valor)
        self.o = nn.Linear(self.dh, d)
        self.drop = nn.Dropout(drop)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        qh = self.q(q).view(B, Lq, self.h, self.dh).transpose(1, 2)
        kh = self.k(k).view(B, Lk, self.h, self.dh).transpose(1, 2)
        vs = self.v(v).unsqueeze(1)              # (B,1,Lk,dh) compartido
        sc = qh @ kh.transpose(-1, -2) / self.dh ** 0.5           # (B,h,Lq,Lk)
        if mask is not None:
            sc = sc.masked_fill(mask, -1e9)
        att = torch.softmax(sc, dim=-1)
        out = (self.drop(att) @ vs).mean(1)      # promedio sobre cabezas → interpretable
        return self.o(out), att.mean(1)          # (B,Lq,d), pesos (B,Lq,Lk)


class TFTCanonical(nn.Module):
    """TFT canónico single-station: VSN + LSTM enc/dec + enriquecimiento estático
    (contexto aprendido) + IMHA + GRN posicional + salida cuantílica. SIN RevIN."""

    def __init__(self, n_past, n_future, hid=64, heads=4, H=7, drop=0.2,
                 q_mu=0.0, q_sd=1.0, **_):
        super().__init__()
        self.H = H
        self.register_buffer("q_mu", torch.tensor(float(q_mu)))
        self.register_buffer("q_sd", torch.tensor(float(q_sd)))
        self.vsn_past = VSN(n_past + 1, hid, drop)      # +1 = canal Q
        self.vsn_fut = VSN(n_future, hid, drop)
        self.static = nn.Parameter(torch.zeros(1, hid)) # contexto estático aprendido
        self.enc_lstm = nn.LSTM(hid, hid, batch_first=True)
        self.dec_lstm = nn.LSTM(hid, hid, batch_first=True)
        self.gate_lstm = nn.Linear(hid, 2 * hid)
        self.norm_lstm = nn.LayerNorm(hid)
        self.enrich = GRN(hid, drop, d_ctx=hid)
        self.attn = IMHA(hid, heads, drop)
        self.gate_att = nn.Linear(hid, 2 * hid)
        self.norm_att = nn.LayerNorm(hid)
        self.ff = GRN(hid, drop)
        self.head = nn.Linear(hid, 3)

    def _glu_add_norm(self, x, resid, gate, norm):
        g = gate(x)
        glu = g[..., :g.shape[-1] // 2] * torch.sigmoid(g[..., g.shape[-1] // 2:])
        return norm(resid + glu)

    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past)                      # (B,ENC,hid)
        sf, _ = self.vsn_fut(x_future)                   # (B,H,hid)
        ctx = self.static.expand(B, -1)                  # (B,hid)
        e, (hN, cN) = self.enc_lstm(sp)
        d, _ = self.dec_lstm(sf, (hN, cN))
        seq_in = torch.cat([sp, sf], dim=1)              # skip para add&norm
        seq = torch.cat([e, d], dim=1)
        seq = self._glu_add_norm(seq, seq_in, self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        a, _ = self.attn(seq, seq, seq, mask=mask.view(1, L, L))
        seq = self._glu_add_norm(a, seq, self.gate_att, self.norm_att)
        seq = self.ff(seq)
        dec = seq[:, -self.H:, :]                         # posiciones del horizonte
        raw = self.head(dec)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        qs = torch.stack([p10, p50, p90], -1)
        return qs * self.q_sd + self.q_mu                # denorm global


class DLinear(nn.Module):
    """DLinear (Zeng et al. 2023): descomposición tendencia/estacional + lineal.
    Baseline "¿hace falta un transformer?". Usa SOLO el canal Q pasado (univariado,
    como los fundacionales) → lineal directo a H×3 cuantiles."""

    def __init__(self, n_past, n_future, hid=64, heads=4, H=7, drop=0.2,
                 enc=90, q_mu=0.0, q_sd=1.0, **_):
        super().__init__()
        self.H = H; self.enc = enc; self.k = 25
        self.register_buffer("q_mu", torch.tensor(float(q_mu)))
        self.register_buffer("q_sd", torch.tensor(float(q_sd)))
        self.lin_trend = nn.Linear(enc, H * 3)
        self.lin_seas = nn.Linear(enc, H * 3)

    def forward(self, x_past, q_past, x_future):
        qn = (q_past - self.q_mu) / self.q_sd             # (B,enc)
        trend = F.avg_pool1d(qn.unsqueeze(1), self.k, 1,
                             padding=self.k // 2)[:, 0, :qn.shape[1]]
        seas = qn - trend
        out = self.lin_trend(trend) + self.lin_seas(seas)  # (B, H*3)
        qs = out.view(-1, self.H, 3)
        p50 = qs[..., 1]
        p10 = p50 - F.softplus(qs[..., 0]); p90 = p50 + F.softplus(qs[..., 2])
        return torch.stack([p10, p50, p90], -1) * self.q_sd + self.q_mu


# ── métricas ──────────────────────────────────────────────────────────────────
def crps3(o, qp):
    t = 0.0
    for i, qv in enumerate(QS):
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


ARCHS = {"TFT-canónico": TFTCanonical, "DLinear": DLinear}


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
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · enc={bp['enc']}")

    rows = []
    for nombre, Cls in ARCHS.items():
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = Cls(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                     H=H, drop=bp["drop"], enc=bp["enc"], q_mu=q_mu, q_sd=q_sd).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"145_{nombre.replace('-', '').replace('ó', 'o')}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
            log.info(f"  {nombre} seed {s} OK")
        PR = np.mean(P, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]
            o = np.array([obs[k] for k in j])
            rows.append(dict(model=nombre, h=h, **met(o, PR[:, h - 1, :])))
        log.info(f"✔ {nombre} completo")

    nuevo = pd.DataFrame(rows)
    base = pd.read_csv(REL / "125_full_metrics_corrected.csv")
    base = base[(base.model == "TFT-honesto") & base.h.isin(LEADS)].assign(
        model="RA-TFT (gen4, nuestro)")[
        ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]]
    tab = pd.concat([base, nuevo], ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "145_canonical_tft.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 145_canonical_tft.csv + checkpoints")


if __name__ == "__main__":
    main()
