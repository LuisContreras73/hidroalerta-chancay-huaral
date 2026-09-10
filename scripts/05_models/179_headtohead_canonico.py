#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 179 — Head-to-head LIMPIO: canónico vs +GRU vs +RevIN(regime-norm) vs +GRU+RevIN.

Mismo harness (173/M.load_data), mismo H=14, mismos 3 seeds, misma vara (C.met sobre aforo).
Aísla dos preguntas del registro maestro:
  · ¿GRU > LSTM como núcleo? (rec=gru)
  · ¿La REGIME-NORMALIZATION del RA-TFT publicado (RevIN: normaliza cada muestra por la media/std
    de su propia ventana de caudal) AYUDA o PERJUDICA el horizonte largo? — explicaría por qué el
    canónico-145 (global-norm) supera al RA-TFT-125 (regime-norm) a h>=7.

TRAZABILIDAD: guarda la curva de loss por época (train+val) [179_loss_curves.csv] + una figura
[reports/figures/DI_headtohead_loss.png], métricas con dispersión de seeds [179_headtohead.csv],
y checkpoints. Para responder luego "¿cómo iba bajando el loss?".

Run (.venv313, GPU): python scripts/05_models/179_headtohead_canonico.py  [--smoke]
"""
import importlib.util, logging, sys
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
FIG = ROOT / "reports/figures"; FIG.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("h2h179"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOKE = "--smoke" in sys.argv


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")
R, T, DEV, C = M.R, M.T, M.DEV, M.C
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; LEADS = [1, 3, 7, 14]


class TFTCanonRevIN(M.TFTCanonMatrix):
    """Canónico con REGIME-NORMALIZATION (RevIN): normaliza q_past por la media/std de CADA muestra
    (su régimen reciente) en vez de por la media/std global de train, y des-normaliza la salida
    igual. Réplica del rasgo distintivo del RA-TFT publicado."""
    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        mu = q_past.mean(1, keepdim=True); sd = q_past.std(1, keepdim=True) + 1e-3   # (B,1) régimen
        qn = ((q_past - mu) / sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past); sf, _ = self.vsn_fut(x_future)
        ctx = self.static.expand(B, -1)
        e, d = self.rec(sp, sf)
        seq = self._gan(torch.cat([e, d], 1), torch.cat([sp, sf], 1), self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        Ln = seq.shape[1]
        mask = torch.triu(torch.ones(Ln, Ln, device=seq.device, dtype=torch.bool), 1).view(1, Ln, Ln)
        qkv = self.pre_attn(seq) if self.pre_attn is not None else seq
        a, _ = self.attn(qkv, qkv, qkv, mask=mask)
        seq = self._gan(a, seq, self.gate_att, self.norm_att); seq = self.ff(seq)
        raw = self.head(seq[:, -self.H:, :])
        p50 = raw[..., 0]; p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], -1) * sd.unsqueeze(-1) + mu.unsqueeze(-1)


CONFIGS = {
    "canónico": (M.TFTCanonMatrix, dict(M.BASE)),
    "canónico+GRU": (M.TFTCanonMatrix, {**M.BASE, "rec": "gru"}),
    "canónico+RevIN": (TFTCanonRevIN, dict(M.BASE)),
    "canónico+GRU+RevIN": (TFTCanonRevIN, {**M.BASE, "rec": "gru"}),
}
if SMOKE:
    CONFIGS = {"canónico": CONFIGS["canónico"], "canónico+RevIN": CONFIGS["canónico+RevIN"]}
NSEED = 1 if SMOKE else 3; EP = 6 if SMOKE else 140; PAT = 3 if SMOKE else 25

(Xtr, Xva, Xte, te, obs), qmu, qsd = M.load_data(bp)
loss_rows = []


def train_logged(model, cfg_name, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
    best = np.inf; bsd = None; bad = 0
    for e in range(EP):
        model.train(); tl = 0.0
        for xb, qb, fb, yb in dl:
            opt.zero_grad(); loss = R.pinball(model(xb.to(DEV), qb.to(DEV), fb.to(DEV)), yb.to(DEV))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = R.pinball(model(Xva[0].to(DEV), Xva[1].to(DEV), Xva[2].to(DEV)), Xva[3].to(DEV)).item()
        loss_rows.append(dict(config=cfg_name, seed=seed, epoch=e, train_loss=tl / len(dl), val_loss=vl))
        if vl < best - 1e-4:
            best, bsd, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT:
                break
    if bsd:
        model.load_state_dict(bsd)
    return model


rows = []
for name, (cls, cfg) in CONFIGS.items():
    seed_preds, per_seed_nse = [], {L: [] for L in LEADS}
    for s in range(NSEED):
        torch.manual_seed(s); np.random.seed(s)
        mo = cls(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                 drop=bp["drop"], q_mu=qmu, q_sd=qsd, **cfg).to(DEV)
        mo = train_logged(mo, name, s)
        torch.save(mo.state_dict(), OUT / f"179_{name.replace('+','_').replace('ó','o')}_seed{s}.pt")
        with torch.no_grad():
            PR = mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy()
        seed_preds.append(PR)
        for L in LEADS:
            o = np.array([obs[i + L - 1] for i in te]); p = PR[:, L - 1, 1]
            m = np.isfinite(o) & np.isfinite(p)
            per_seed_nse[L].append(1 - np.sum((o[m] - p[m]) ** 2) / np.sum((o[m] - o[m].mean()) ** 2))
    ens = np.mean(seed_preds, 0)   # ensemble (media de seeds) = como el registro maestro
    for L in LEADS:
        j = [i + L - 1 for i in te]; o = np.array([obs[k] for k in j])
        met = C.met(o, ens[:, L - 1, :])
        rows.append(dict(config=name, h=L, NSE=round(met["NSE"], 3),
                         NSE_seed_std=round(float(np.std(per_seed_nse[L])), 3),
                         CRPS=round(met.get("CRPS", np.nan), 3), CSI=round(met["CSI"], 3),
                         POD=round(met["POD"], 3), params=sum(p.numel() for p in mo.parameters())))
    log.info(f"✔ {name}: " + " ".join(f"h{L}={[r for r in rows if r['config']==name and r['h']==L][0]['NSE']}" for L in LEADS))
    pd.DataFrame(rows).to_csv(OUT / "179_headtohead.csv", index=False)

pd.DataFrame(loss_rows).to_csv(OUT / "179_loss_curves.csv", index=False)

# ── figura de trazabilidad: curvas de loss (media sobre seeds) ──
lc = pd.DataFrame(loss_rows); fig, ax = plt.subplots(figsize=(8, 4.6))
cols = {"canónico": "#0e7d90", "canónico+GRU": "#2f9e6f", "canónico+RevIN": "#e08a2a", "canónico+GRU+RevIN": "#d24e39"}
for name in CONFIGS:
    g = lc[lc.config == name].groupby("epoch").agg(tr=("train_loss", "mean"), vl=("val_loss", "mean"))
    c = cols.get(name, "#666")
    ax.plot(g.index, g.tr, color=c, lw=1.2, alpha=0.5, ls="--")
    ax.plot(g.index, g.vl, color=c, lw=2.0, label=name)
ax.set_xlabel("época"); ax.set_ylabel("pinball loss"); ax.set_title("Head-to-head — curvas de loss (— val, -- train, media de seeds)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(lw=0.3, alpha=0.4)
fig.savefig(FIG / "DI_headtohead_loss.png", dpi=150, bbox_inches="tight")

r = pd.DataFrame(rows); log.info("\n=== NSE (ensemble) ===\n" + r.pivot_table(index="config", columns="h", values="NSE").to_string())
log.info("HEADTOHEAD_DONE; 179_headtohead.csv + 179_loss_curves.csv + DI_headtohead_loss.png")
