#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo arch_variants — variantes de NORMALIZACIÓN y ATENCIÓN del RA-TFT (gen5).

Clases reutilizables (importables por los runners de entrenamiento y por el
generador de tablas 146). NO entrena nada al importarse. Consolida las variantes
pendientes documentadas en docs/ZOO_ARQUITECTURAS.md (Familia I: normalización;
eje cross vs masked-self en atención).

Normalizaciones nuevas (además de las ya probadas en 142/143: LayerNorm, preRMS,
sandwich, QK-norm):
  - ScaleNorm  (Nguyen & Salazar 2019): un solo escalar aprendido, g·x/‖x‖.
  - ReZero     (Bachlechner 2020): sin norma; residual con escalar α init-0.
  - GatedRMSNorm: RMSNorm seguida de compuerta sigmoide.

Atención:
  - RATFTMaskedSelf: reemplaza la cross-attention por MASKED SELF-attention
    (máscara causal, como el TFT canónico) → ablaciona el eje "cross vs masked-self".

Todas derivan de R.RATFT (scripts/05_models/105_ratft.py) para heredar RevIN, VSN
implícito, LSTM enc/dec, cabeza cuantílica y denorm. Se entrenan con el MISMO
harness (T.train_es) y se comparan contra la release congelada gen4_2026-07-02.
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("r105", ROOT / "scripts/05_models/105_ratft.py")
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)


# ── Normalizaciones ───────────────────────────────────────────────────────────
class ScaleNorm(nn.Module):
    """ScaleNorm (Nguyen & Salazar 2019): g · x / ‖x‖₂ (un escalar aprendido)."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(d ** 0.5))
        self.eps = eps

    def forward(self, x):
        return self.g * x / (x.norm(dim=-1, keepdim=True) + self.eps)


class GatedRMSNorm(nn.Module):
    """RMSNorm + compuerta sigmoide por-canal."""

    def __init__(self, d):
        super().__init__()
        self.rms = nn.RMSNorm(d)
        self.gate = nn.Linear(d, d)

    def forward(self, x):
        n = self.rms(x)
        return n * torch.sigmoid(self.gate(x))


# ── Variantes de RA-TFT por NORMALIZACIÓN ─────────────────────────────────────
class RATFTNorm(R.RATFT):
    """RA-TFT con la norma post-residual intercambiable.
    norm_kind ∈ {layernorm(=base), scalenorm, gatedrms, rezero}."""

    def __init__(self, *a, norm_kind="scalenorm", **k):
        super().__init__(*a, **k)
        hid = self.norm.normalized_shape[0]
        self.norm_kind = norm_kind
        if norm_kind == "scalenorm":
            self.norm = ScaleNorm(hid)
        elif norm_kind == "gatedrms":
            self.norm = GatedRMSNorm(hid)
        elif norm_kind == "rezero":
            self.norm = nn.Identity()
            self.alpha = nn.Parameter(torch.zeros(1))     # residual escalar init-0

    def forward(self, x_past, q_past, x_future):
        qn = self.revin.norm(q_past)
        xp = torch.cat([x_past, qn.unsqueeze(-1)], dim=-1) if x_past.shape[-1] != 0 else qn.unsqueeze(-1)
        e = self.enc_proj(xp); e, (hN, cN) = self.enc_lstm(e)
        if not self.use_future:
            x_future = torch.zeros_like(x_future)
        d = self.dec_proj(x_future); d, _ = self.dec_lstm(d, (hN, cN))
        a, _ = self.cross(d, e, e)
        gated = self.drop(a) * self.gate(a)
        if self.norm_kind == "rezero":
            h = d + self.alpha * gated                    # sin norma, escalar aprendido
        else:
            h = self.norm(d + gated)
        raw = self.head(h)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([self.revin.denorm(p10), self.revin.denorm(p50),
                            self.revin.denorm(p90)], -1)


# ── Variante de RA-TFT por ATENCIÓN (cross → masked self) ─────────────────────
class RATFTMaskedSelf(R.RATFT):
    """RA-TFT con MASKED SELF-attention en vez de cross-attention (ablaciona el eje
    'cross vs masked-self' que distingue nuestro modelo del TFT canónico). El decoder
    se auto-atiende con máscara causal sobre [encoder ⊕ decoder]."""

    def forward(self, x_past, q_past, x_future):
        qn = self.revin.norm(q_past)
        xp = torch.cat([x_past, qn.unsqueeze(-1)], dim=-1) if x_past.shape[-1] != 0 else qn.unsqueeze(-1)
        e = self.enc_proj(xp); e, (hN, cN) = self.enc_lstm(e)
        if not self.use_future:
            x_future = torch.zeros_like(x_future)
        d = self.dec_proj(x_future); d, _ = self.dec_lstm(d, (hN, cN))
        seq = torch.cat([e, d], dim=1)                    # (B, ENC+H, hid)
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        a, _ = self.cross(seq, seq, seq, attn_mask=mask)  # masked self-attention
        a = a[:, -self.H:, :]                             # posiciones del horizonte
        h = self.norm(d + self.drop(a) * self.gate(a))
        raw = self.head(h)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([self.revin.denorm(p10), self.revin.denorm(p50),
                            self.revin.denorm(p90)], -1)


# Registro para runners/summaries (nombre → (Clase, kwargs extra))
VARIANTES = {
    "RA-TFT[scalenorm]": (RATFTNorm, dict(norm_kind="scalenorm")),
    "RA-TFT[rezero]":    (RATFTNorm, dict(norm_kind="rezero")),
    "RA-TFT[gatedrms]":  (RATFTNorm, dict(norm_kind="gatedrms")),
    "RA-TFT[maskedself]": (RATFTMaskedSelf, dict()),
}
