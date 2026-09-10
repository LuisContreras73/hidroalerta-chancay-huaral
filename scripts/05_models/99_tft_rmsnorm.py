#!/usr/bin/env python3
"""
Script 99 — TFT + RMSNorm-before-QKV (architectural innovation, clean A/B).

Does NOT modify/overwrite the existing TFT (Scripts 74/81). It defines a new
model variant that applies RMSNorm to the LSTM output BEFORE the Q/K/V projection
of the self-attention (pre-attention RMS normalization, à la LLaMA; RMSNorm =
Zhang & Sennrich 2019). Rationale: normalizing the representation right before
Q,K,V stabilises attention-score scales and has been shown to help.

Runs a clean A/B with the SAME data pipeline, tuned config and seeds:
  baseline  = current TFT (single post-attention LayerNorm)
  +RMSNorm  = same + RMSNorm before Q/K/V
plus the persistence baseline, all on the honest real-obs test.

Run in .venv313:
    python scripts/05_models/99_tft_rmsnorm.py --seeds 3
"""

import argparse, importlib.util, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tft_rms")

spec = importlib.util.spec_from_file_location("tune81", ROOT/"scripts/05_models/81_tune_tft_optuna.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
T = S.T
DEV = S.DEVICE; Q_CONV = S.Q_CONV; Q90 = S.Q90


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class TFTLiteRMS(nn.Module):
    """TFTLite v2 with an optional RMSNorm applied to the attention input (pre-QKV)."""
    def __init__(self, nf, cfg, use_rms=True):
        super().__init__()
        HID, ENC, HEADS, DROP = cfg["HID"], cfg["ENC"], cfg["HEADS"], cfg["DROP"]
        self.vsn  = T.VSN(nf, HID)
        self.proj = nn.Linear(nf, HID)
        self.pos  = nn.Parameter(torch.randn(1, ENC, HID) * 0.02)
        self.lstm = nn.LSTM(HID, HID, num_layers=1, batch_first=True)
        self.use_rms = use_rms
        self.qkv_norm = RMSNorm(HID) if use_rms else None      # <-- innovation
        self.attn = nn.MultiheadAttention(HID, HEADS, dropout=DROP, batch_first=True)
        self.norm = nn.LayerNorm(HID)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Sequential(nn.Linear(HID, HID//2), nn.ReLU(), nn.Dropout(DROP),
                                  nn.Linear(HID//2, 3))
    def forward(self, x):
        h = self.proj(self.vsn(x)) + self.pos
        o, _ = self.lstm(h)
        qkv = self.qkv_norm(o) if self.use_rms else o          # RMSNorm before Q/K/V
        a, _ = self.attn(qkv, qkv, qkv)
        h = self.norm(o + self.drop(a))
        raw = self.head(h[:, -1, :])
        p50 = raw[:, 0]; p10 = p50 - F.softplus(raw[:, 1]); p90 = p50 + F.softplus(raw[:, 2])
        return torch.stack([p10, p50, p90], dim=1)


def build_and_train(cfg, data, seed, use_rms):
    """Mirror S.run_config but with the RMS-capable model (no overwrite of T)."""
    T.ENC, T.HID, T.HEADS, T.DROP, T.WD = cfg["ENC"], cfg["HID"], cfg["HEADS"], cfg["DROP"], cfg["WD"]
    torch.manual_seed(seed); np.random.seed(seed)
    Xp, yp, wp, _ = T.make_seqs(data["Xs"], data["ysc"], data["m_p"])
    y_mm = data["df"]["q_next_1d"].values.astype(np.float32); thr = Q90*Q_CONV
    Xft, yft, wft, _ = T.make_seqs(data["Xs"], data["ysc"], data["m_ft"],
                                   y_raw=y_mm, alert_thr=thr, alert_w=cfg["ALERT_W"])
    k = int(len(Xft)*0.8)
    model = TFTLiteRMS(data["nf"], cfg, use_rms=use_rms).to(DEV)
    model = S.train(model, Xp, yp, wp, Xp[-200:], yp[-200:], cfg["LR"], 80, 18, cfg["mix_alpha"])
    model = S.train(model, Xft[:k], yft[:k], wft[:k], Xft[k:], yft[k:], cfg["LR_FT"], 70, 15, cfg["mix_alpha"])
    return model


def eval_honest(model, data):
    Xseq, y, w, idx = T.make_seqs(data["Xs"], data["ysc"], data["m_te"])
    obs_mm = data["obs_real_mm"][idx]; real = np.isfinite(obs_mm)
    with torch.no_grad():
        p50 = data["inv"](model(Xseq.to(DEV)).cpu().numpy()[:, 1]) / Q_CONV
    obs = obs_mm[real]/Q_CONV
    return T.det_metrics(obs, p50[real], Q90)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    data = S.prepare_data()
    bp = pd.read_csv(OUTML/"81_best_params.csv", index_col=0)["0"].to_dict()
    cfg = dict(ENC=int(bp["ENC"]), HID=int(bp["HID"]), HEADS=int(bp["HEADS"]), DROP=float(bp["DROP"]),
               LR=float(bp["LR"]), LR_FT=float(bp["LR_FT"]), WD=float(bp["WD"]),
               ALERT_W=float(bp["ALERT_W"]), mix_alpha=float(bp["mix_alpha"]))
    log.info(f"Config: {cfg}")

    rows = []
    for use_rms, name in [(False, "baseline (LayerNorm only)"), (True, "+ RMSNorm before QKV")]:
        for s in range(a.seeds):
            model = build_and_train(cfg, data, seed=100+s, use_rms=use_rms)
            m = eval_honest(model, data)
            rows.append(dict(variant=name, seed=s, **m))
            log.info(f"  {name:26s} seed{s}  NSE={m['NSE']:.3f} J={m['J_alert']:.3f} "
                     f"CSI={m['CSI']:.3f} POD={m['POD_det']:.3f} FAR={m['FAR']:.3f}")

    df = pd.DataFrame(rows)
    agg = df.groupby("variant").agg(NSE=("NSE","mean"), NSE_sd=("NSE","std"),
                                    J=("J_alert","mean"), J_sd=("J_alert","std"),
                                    CSI=("CSI","mean"), POD=("POD_det","mean"), FAR=("FAR","mean"))
    log.info("\n=== A/B RESULT (honest test, mean over seeds) ===")
    log.info("\n"+agg.round(4).to_string())
    df.to_csv(OUTML/"99_tft_rmsnorm_ab.csv", index=False)
    log.info("\nReference baseline (persistence, Script 88): NSE=0.964, J_alert=0.690")
    log.info("Saved: 99_tft_rmsnorm_ab.csv")


if __name__ == "__main__":
    main()
