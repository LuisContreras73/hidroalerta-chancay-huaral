#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 143 — BANCO de ablaciones gen5: ideas de fundacionales + clásicas, clasificadas.

Taxonomía (tabla T1 del paper):
  I.  Normalización interna (de los fundacionales, VERIFICADA en sus pesos):
      qknorm   = pre-RMS + QK-norm por cabeza (TimesFM: query_ln/key_ln)
      sandwich = pre-RMS + RMS post-atención + RMS final (TimesFM: *_ln pre/post)
      (preRMS solo → script 142, corre aparte)
  II. Transformación de entrada (Chronos-2: use_arcsinh + instance norm):
      asinh    = RevIN sobre arcsinh(Q); salida sinh(denorm) — cuantiles se
                 preservan (transformación monótona); pérdida en espacio físico
  III. Cabeza / objetivo:
      densq7   = cabeza de 7 cuantiles monótonos (05,10,25,50,75,90,95) — los FMs
                 usan 21/deciles; se evalúa igual en q10/50/90
      alertw   = pinball ponderado a caudales altos (w=1+2·[y≥0,8·Q90]) — espíritu
                 de la Regla 20 (J_alert), versión en la pérdida
  IV. Calibración post-hoc: conformal → script 141 (✅ hecha)
  V.  Información (GFS) → 2×2 aparte (requiere features B11 con QM solo-train)

Protocolo A/B: mismos datos/split/hiperparámetros (estudio intv_h14)/seeds (0,1,2)
que la release congelada gen4_2026-07-02; solo cambia el componente. Artefactos
append-only: 143_*. Consolida también el resultado del 142 si ya existe.

Salida: outputs/ml_Q/143_ablaciones_banco.csv (+ checkpoints 143_<var>_seed*.pt)

Run en .venv313:
    python scripts/05_models/143_ablaciones_banco.py
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
log = logging.getLogger("banco143")
optuna.logging.set_verbosity(optuna.logging.WARNING)

spec = importlib.util.spec_from_file_location(
    "t114", ROOT / "scripts/05_models/114_tft_intensive.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R = T.R
DEV = R.DEV
Q90 = R.Q90
H = 14
LEADS = [1, 3, 7, 14]
Q7 = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


# ── módulos ───────────────────────────────────────────────────────────────────
class CrossAttnQKNorm(nn.Module):
    """Cross-attention con QK-norm por cabeza (receta TimesFM: query_ln/key_ln)."""

    def __init__(self, hid, heads, drop):
        super().__init__()
        assert hid % heads == 0
        self.h, self.dh = heads, hid // heads
        self.q = nn.Linear(hid, hid, bias=False)
        self.k = nn.Linear(hid, hid, bias=False)
        self.v = nn.Linear(hid, hid, bias=False)
        self.o = nn.Linear(hid, hid, bias=False)
        self.rms_q = nn.RMSNorm(self.dh)
        self.rms_k = nn.RMSNorm(self.dh)
        self.drop = nn.Dropout(drop)

    def forward(self, q, k, v):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        qh = self.rms_q(self.q(q).view(B, Lq, self.h, self.dh)).transpose(1, 2)
        kh = self.rms_k(self.k(k).view(B, Lk, self.h, self.dh)).transpose(1, 2)
        vh = self.v(v).view(B, Lk, self.h, self.dh).transpose(1, 2)
        w = torch.softmax(qh @ kh.transpose(-1, -2) / self.dh ** 0.5, dim=-1)
        out = (self.drop(w) @ vh).transpose(1, 2).reshape(B, Lq, -1)
        return self.o(out), None


class RATFT_X(R.RATFT):
    """RA-TFT flexible: norm ∈ {qknorm, sandwich}, asinh de entrada, cabeza nq∈{3,7}."""

    def __init__(self, n_past, n_future, hid=64, heads=4, H=7, drop=0.2,
                 use_future=True, norm="qknorm", asinh=False, nq=3):
        super().__init__(n_past, n_future, hid=hid, heads=heads, H=H, drop=drop,
                         use_future=use_future)
        self.variant_norm, self.use_asinh, self.nq = norm, asinh, nq
        self.rms_q = nn.RMSNorm(hid); self.rms_kv = nn.RMSNorm(hid)
        self.rms_out = nn.RMSNorm(hid)
        if norm == "qknorm":
            self.cross = CrossAttnQKNorm(hid, heads, drop)
        if norm == "sandwich":
            self.rms_post = nn.RMSNorm(hid)
        if nq == 7:
            self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.SiLU(),
                                      nn.Dropout(drop), nn.Linear(hid // 2, 7))

    def forward(self, x_past, q_past, x_future):
        qp = torch.asinh(q_past) if self.use_asinh else q_past
        qn = self.revin.norm(qp)
        xp = torch.cat([x_past, qn.unsqueeze(-1)], dim=-1) \
            if x_past.shape[-1] != 0 else qn.unsqueeze(-1)
        e = self.enc_proj(xp); e, (hN, cN) = self.enc_lstm(e)
        if not self.use_future:
            x_future = torch.zeros_like(x_future)
        d = self.dec_proj(x_future); d, _ = self.dec_lstm(d, (hN, cN))
        a, _ = self.cross(self.rms_q(d), self.rms_kv(e), self.rms_kv(e))
        if self.variant_norm == "sandwich":
            a = self.rms_post(a)
        h = d + self.drop(a) * self.gate(a)
        h = self.rms_out(h)
        raw = self.head(h)
        if self.nq == 3:
            p50 = raw[..., 0]
            lo = [p50 - F.softplus(raw[..., 1])]
            hi = [p50 + F.softplus(raw[..., 2])]
            qs = torch.stack(lo + [p50] + hi, -1)          # (B,H,3): 10,50,90
        else:                                              # 7 cuantiles monótonos
            p50 = raw[..., 0]
            p25 = p50 - F.softplus(raw[..., 1]); p10 = p25 - F.softplus(raw[..., 2])
            p05 = p10 - F.softplus(raw[..., 3])
            p75 = p50 + F.softplus(raw[..., 4]); p90 = p75 + F.softplus(raw[..., 5])
            p95 = p90 + F.softplus(raw[..., 6])
            qs = torch.stack([p05, p10, p25, p50, p75, p90, p95], -1)
        out = self.revin.denorm(qs.permute(0, 2, 1)).permute(0, 2, 1) \
            if False else torch.stack([self.revin.denorm(qs[..., i])
                                       for i in range(qs.shape[-1])], -1)
        return torch.sinh(out) if self.use_asinh else out


# ── pérdidas ──────────────────────────────────────────────────────────────────
def pinball_n(pred, y, levels):
    t = 0.0
    for i, qv in enumerate(levels):
        e = y - pred[..., i]
        t = t + torch.mean(torch.clamp(e, min=0) * qv + torch.clamp(-e, min=0) * (1 - qv))
    return t / len(levels)


def pinball_alertw(pred, y, levels=(0.1, 0.5, 0.9)):
    w = 1.0 + 2.0 * (y >= 0.8 * Q90).float()
    t = 0.0
    for i, qv in enumerate(levels):
        e = y - pred[..., i]
        li = torch.clamp(e, min=0) * qv + torch.clamp(-e, min=0) * (1 - qv)
        t = t + torch.mean(w * li)
    return t / len(levels)


def train_es_x(model, dl, Xv, lr, wd, loss_fn, max_ep=180, pat=25):
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
            vl = loss_fn(model(Xv[0].to(DEV), Xv[1].to(DEV), Xv[2].to(DEV)),
                         Xv[3].to(DEV)).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    if bs:
        model.load_state_dict(bs)
    return model


# ── métricas (idénticas al 142) ───────────────────────────────────────────────
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
    tp, fp, fn = np.sum(oa & pa), np.sum(~oa & pa), np.sum(oa & ~fn if False else oa & ~pa)
    fn = np.sum(oa & ~pa)
    return dict(N=int(m.sum()), NSE=round(nse, 3), NSE_sqrt=round(nsq, 3),
                KGE=round(kge, 3), MAE=round(float(np.mean(np.abs(o2 - p2))), 2),
                CRPS=round(float(crps3(o2, qp2)), 3),
                CSI=round(tp / (tp + fp + fn), 3) if (tp + fp + fn) else 0.0,
                POD=round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
                FAR=round(fp / (tp + fp), 3) if (tp + fp) else 0.0)


VARIANTES = {
    # nombre: (taxonomía, kwargs del modelo, pérdida, niveles para extraer 10/50/90)
    "qknorm":   ("I-normalización", dict(norm="qknorm"),   "pinball3", [0, 1, 2]),
    "sandwich": ("I-normalización", dict(norm="sandwich"), "pinball3", [0, 1, 2]),
    "asinh":    ("II-entrada",      dict(norm="qknorm_off", asinh=True), "pinball3", [0, 1, 2]),
    "densq7":   ("III-cabeza",      dict(norm="qknorm_off", nq=7), "pinball7", [1, 3, 5]),
    "alertw":   ("III-objetivo",    dict(norm="qknorm_off"), "alertw", [0, 1, 2]),
}


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
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6,
          "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))
    log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · enc={bp['enc']}")

    LOSS = {"pinball3": lambda p, y: pinball_n(p, y, [0.1, 0.5, 0.9]),
            "pinball7": lambda p, y: pinball_n(p, y, Q7),
            "alertw": pinball_alertw}

    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--solo", default=None)
    solo = ap.parse_args().solo

    rows = []
    for nombre, (taxo, kw, lname, idx) in VARIANTES.items():
        if solo and nombre != solo:
            continue
        kw = dict(kw)
        # BUG corregido (2026-07-09): pop sin restaurar dejaba a 'sandwich' con el
        # default 'qknorm' — sandwich del primer run era qknorm duplicado.
        esquema = kw.pop("norm", "prerms_only")
        kw["norm"] = "prerms_only" if esquema == "qknorm_off" else esquema
        P = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
            mo = RATFT_X(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                         H=H, drop=bp["drop"], use_future=True, **kw).to(DEV)
            mo = train_es_x(mo, dl, Xva, bp["lr"], bp["wd"], LOSS[lname])
            torch.save(mo.state_dict(), OUT / f"143_{nombre}_seed{s}.pt")
            with torch.no_grad():
                P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
        PR = np.mean(P, 0)[:, :, idx]           # extrae q10/q50/q90
        for h in LEADS:
            j = [i + h - 1 for i in te]
            o = np.array([obs[k] for k in j])
            rows.append(dict(model=f"RA-TFT+{nombre}", taxonomia=taxo, h=h,
                             **met(o, PR[:, h - 1, :])))
        log.info(f"✔ variante {nombre} ({taxo}) completa")

    nuevo = pd.DataFrame(rows)
    base = pd.read_csv(REL / "125_full_metrics_corrected.csv")
    base = base[(base.model == "TFT-honesto") & base.h.isin(LEADS)].assign(
        model="RA-TFT (gen4, congelado)", taxonomia="baseline")[
        ["model", "taxonomia", "h", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS",
         "CSI", "POD", "FAR"]]
    piezas = [base, nuevo]
    f142 = OUT / "142_ablacion_rmsnorm.csv"
    if f142.exists():
        d142 = pd.read_csv(f142)
        d142 = d142[d142.model == "RA-TFT+preRMS"].assign(taxonomia="I-normalización")
        piezas.append(d142)
    tab = pd.concat(piezas, ignore_index=True).sort_values(["h", "model"])
    tab.to_csv(OUT / "143_ablaciones_banco.csv", index=False)
    log.info("\n" + tab.to_string(index=False))
    log.info("Guardado: 143_ablaciones_banco.csv")


if __name__ == "__main__":
    main()
