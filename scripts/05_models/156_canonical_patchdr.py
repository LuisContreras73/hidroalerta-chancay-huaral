#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 156 — PARCHEO DE DOBLE RESOLUCIÓN sobre el TFT canónico (Ola 3, componente propio).

LA IDEA (candidata a componente innovador del paper):
  La cuenca es flashy → el detalle DIARIO reciente importa (CCF lluvia→caudal pica a
  ~4 d). Pero la memoria ESTACIONAL también (hallazgo 114: ENC debe escalar con el
  horizonte). El parcheo de los fundacionales (Chronos-2 P=16, TimesFM P=32 —
  verificado en pesos) resuelve el conflicto:
    [últimos K=14 días a resolución DIARIA] ⊕ [contexto lejano en PARCHES de P=15 días]
  → permite CONTEXTO DE ~1 AÑO (374 días) en solo 14+24=38 tokens (hoy 90 días = 90
  tokens). Memoria estacional completa sin explotar la secuencia.

DISEÑO (ablación limpia — el canónico NO se toca):
  El parcheo es un FRONT-END compresor por variable: contexto lejano (B, 360, C) →
  (B, 24, 15, C) → proyección lineal aprendida por variable sobre el eje de 15 días →
  (B, 24, C); se concatena con los 14 días recientes → (B, 38, C) → entra al canónico
  (VSN/LSTM/atención) SIN CAMBIOS. El canal Q recibe el mismo tratamiento.

CELDAS (3 seeds c/u):
  patchDR-374 — la estrella: contexto 374 d (14 diarios + 24 parches de 15)
  patchDR-89  — control de misma-información: contexto 89 d (14 diarios + 5 parches
                de 15) ≈ el enc=90 actual, para separar "parcheo" de "más contexto".
Baseline = canónico puro (145). Mismos hparams (intv_h14), seeds 0/1/2, T.train_es.
Nota declarada: con enc=374 el train pierde ~284 días iniciales de 1981 (ventana);
el conjunto de TEST es IDÉNTICO al de 145 (verificado por construcción).

Salida: outputs/ml_Q/156_canonical_patchdr.csv (+ checkpoints 156_*.pt)
Run en .venv313 (SÓLO con GPU libre; encadenar tras 155):
    python scripts/05_models/156_canonical_patchdr.py
"""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("patchdr156")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")
T = C.T; R = C.R; DEV = C.DEV; Q90 = R.Q90
H = 14; LEADS = [1, 3, 7, 14]
K_REC = 14      # días recientes a resolución diaria
P = 15          # tamaño de parche (días)


class PatchDualRes(nn.Module):
    """Front-end: (B, ENC, C) → (B, K + n_parches, C). Proyección lineal aprendida
    por variable sobre el eje temporal del parche (peso (C, P), init=media)."""

    def __init__(self, n_ch, k_rec=K_REC, p=P):
        super().__init__()
        self.k, self.p = k_rec, p
        self.w = nn.Parameter(torch.full((n_ch, p), 1.0 / p))   # init = promedio del parche

    def forward(self, x):                       # x: (B, ENC, C)
        far, rec = x[:, :-self.k, :], x[:, -self.k:, :]
        Bt, L, Cn = far.shape
        n = L // self.p
        far = far[:, L - n * self.p:, :].reshape(Bt, n, self.p, Cn)
        far = torch.einsum("bnpc,cp->bnc", far, self.w)          # (B, n, C)
        return torch.cat([far, rec], dim=1)                      # (B, n+k, C)


class TFTCanonPatchDR(C.TFTCanonical):
    """Canónico intacto + front-end de parcheo doble-resolución en el encoder."""

    def __init__(self, *a, enc_ctx=374, **k):
        super().__init__(*a, **k)
        n_past = self.vsn_past.n - 1                              # sin el canal Q
        self.patch_x = PatchDualRes(n_past)
        self.patch_q = PatchDualRes(1)
        self.enc_ctx = enc_ctx

    def forward(self, x_past, q_past, x_future):
        qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)     # (B, ENC, 1)
        xp = self.patch_x(x_past)                                 # (B, T', C)
        qp = self.patch_q(qn)                                     # (B, T', 1)
        past = torch.cat([xp, qp], dim=-1)
        B = past.shape[0]
        sp, _ = self.vsn_past(past)
        sf, _ = self.vsn_fut(x_future)
        ctx = self.static.expand(B, -1)
        e, (hN, cN) = self.enc_lstm(sp)
        d, _ = self.dec_lstm(sf, (hN, cN))
        seq = self._glu_add_norm(torch.cat([e, d], 1), torch.cat([sp, sf], 1),
                                 self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1).view(1, L, L)
        a, _ = self.attn(seq, seq, seq, mask=mask)
        seq = self._glu_add_norm(a, seq, self.gate_att, self.norm_att)
        seq = self.ff(seq)
        raw = self.head(seq[:, -self.H:, :])
        p50 = raw[..., 0]
        p10 = p50 - torch.nn.functional.softplus(raw[..., 1])
        p90 = p50 + torch.nn.functional.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], -1) * self.q_sd + self.q_mu


CELDAS = {"canónico+patchDR374": 374, "canónico+patchDR89": 89}


def main():
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    df = R.build(H); dts = df.index; obs = df["obs"].values; q = df["q"].values
    qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
    qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)

    rows = []
    for nombre, ENCC in CELDAS.items():
        tr = [i for i in range(ENCC, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")
              and np.isfinite(q[i - ENCC:i + H]).all()]
        va = [i for i in range(ENCC, len(df) - H)
              if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31")
              and np.isfinite(q[i - ENCC:i + H]).all()]
        te = [i for i in range(ENCC, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
        mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
        sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6,
              "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
        Xtr = T.seqs_for_enc(df, tr, ENCC, H, (mu, sd))
        Xva = T.seqs_for_enc(df, va, ENCC, H, (mu, sd))
        Xte = T.seqs_for_enc(df, te, ENCC, H, (mu, sd))
        n_tok = (ENCC - K_REC) // P + K_REC
        log.info(f"{nombre}: ctx={ENCC} d → {n_tok} tokens · train {len(tr)} · test {len(te)}")
        P3 = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = TFTCanonPatchDR(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                                 H=H, drop=bp["drop"], enc=ENCC, q_mu=qmu, q_sd=qsd,
                                 enc_ctx=ENCC).to(DEV)
            mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
            torch.save(mo.state_dict(), OUT / f"156_patchdr{ENCC}_seed{s}.pt")
            with torch.no_grad():
                P3.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
            log.info(f"  seed {s} OK")
        PR = np.mean(P3, 0)
        for h in LEADS:
            j = [i + h - 1 for i in te]; o = np.array([obs[k] for k in j])
            rows.append(dict(model=nombre, h=h, **C.met(o, PR[:, h - 1, :])))
        log.info(f"✔ {nombre}")

    nuevo = pd.DataFrame(rows)
    cols = ["model", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]
    c145 = pd.read_csv(OUT / "145_canonical_tft.csv")
    base = c145[(c145.model == "TFT-canónico") & c145.h.isin(LEADS)]
    tab = pd.concat([base[[c for c in cols if c in base.columns]], nuevo],
                    ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "156_canonical_patchdr.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 156_canonical_patchdr.csv")


if __name__ == "__main__":
    main()
