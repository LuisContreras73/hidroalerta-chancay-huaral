#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 188 — BENCHMARK de desempeño en TEST 2024-25 (leaderboard definitivo, mismo harness).
"Probemos de verdad los modelos": familia 179 (canónico, +GRU, +RevIN, +GRU+RevIN) + persistencia +
climatología, con métricas RIGUROSAS: NSE, NSE*clim (referente único), KGE, CRPS, CSI, y IC 95 % por
block-bootstrap (bloque = decorrelación) para NSE — sin re-entrenar (checkpoints 179_*).

Salida: outputs/ml_Q/188_benchmark_test.csv · reports/figures/DI_benchmark_test.png
Run (.venv313): python scripts/05_models/188_benchmark_test.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bench188"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV, C = M.R, M.T, M.DEV, M.C
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90; rng = np.random.default_rng(0)


class TFTCanonRevIN(M.TFTCanonMatrix):
    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]; mu = q_past.mean(1, keepdim=True); sd = q_past.std(1, keepdim=True) + 1e-3
        qn = ((q_past - mu) / sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past); sf, _ = self.vsn_fut(x_future); ctx = self.static.expand(B, -1)
        e, d = self.rec(sp, sf); seq = self._gan(torch.cat([e, d], 1), torch.cat([sp, sf], 1), self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1)); Ln = seq.shape[1]
        mask = torch.triu(torch.ones(Ln, Ln, device=seq.device, dtype=torch.bool), 1).view(1, Ln, Ln)
        qkv = self.pre_attn(seq) if self.pre_attn is not None else seq
        a, _ = self.attn(qkv, qkv, qkv, mask=mask); seq = self._gan(a, seq, self.gate_att, self.norm_att); seq = self.ff(seq)
        raw = self.head(seq[:, -self.H:, :]); p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], -1) * sd.unsqueeze(-1) + mu.unsqueeze(-1)


df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]

# climatología estacional de train (referente único, suavizada 15 d)
doy = dts.dayofyear.values; clim = np.full(367, np.nan)
for d in range(1, 367):
    v = obs[(doy == d) & (dts.values <= np.datetime64("2022-12-31")) & np.isfinite(obs)]
    if len(v): clim[d] = v.mean()
sm = clim.copy()
for d in range(1, 367):
    w = [(d + k - 1) % 366 + 1 for k in range(-7, 8)]; vv = clim[w]; sm[d] = np.nanmean(vv) if np.isfinite(vv).any() else clim[d]
clim = sm; clim_of = lambda idxs, Lh: np.array([clim[dts[i + Lh - 1].dayofyear] for i in idxs])

FAM = {"canónico": ("canonico", M.TFTCanonMatrix, dict(M.BASE)),
       "canónico+GRU": ("canonico_GRU", M.TFTCanonMatrix, {**M.BASE, "rec": "gru"}),
       "canónico+RevIN": ("canonico_RevIN", TFTCanonRevIN, dict(M.BASE)),
       "canónico+GRU+RevIN": ("canonico_GRU_RevIN", TFTCanonRevIN, {**M.BASE, "rec": "gru"})}
models = {}
for disp, (fn, cls, cfg) in FAM.items():
    ms = []
    for s in range(3):
        mo = cls(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, **cfg).to(DEV)
        mo.load_state_dict(torch.load(OUT / f"179_{fn}_seed{s}.pt", map_location=DEV)); mo.eval(); ms.append(mo)
    models[disp] = ms


def pred_ens(ms, idxs):
    X = T.seqs_for_enc(df, idxs, enc, H, (mu, sd)); ps = []
    for mo in ms:
        with torch.no_grad(): ps.append(mo(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
    return np.mean(ps, 0)


def nse(o, p): return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def lag1(x):
    x = x[np.isfinite(x)]; x = x - x.mean(); return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if len(x) > 2 else 0.0


def boot_ci(o, p, blk, reps=1000):
    Ns = len(o); nb = int(np.ceil(Ns / blk)); out = []; smax = max(1, Ns - blk + 1)
    for _ in range(reps):
        idx = np.concatenate([np.arange(st, st + blk) for st in rng.integers(0, smax, nb)])[:Ns] % Ns
        if o[idx].std() > 1e-9: out.append(nse(o[idx], p[idx]))
    return (round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3))


rows = []
for Lh in (1, 3, 7, 14):
    o = np.array([obs[i + Lh - 1] for i in te]); oref = clim_of(te, Lh)
    cands = {d: (pred_ens(ms, te)[:, Lh - 1, :]) for d, ms in models.items()}   # (n,3)
    det = {"persistencia": np.array([q[i - 1] for i in te]), "climatología": oref.copy()}
    for disp in list(cands.keys()) + list(det.keys()):
        if disp in cands:
            qp = cands[disp]; p = qp[:, 1]
        else:
            p = det[disp]; qp = np.stack([p, p, p], 1)
        mk = np.isfinite(o) & np.isfinite(p) & np.isfinite(oref); oo, pp, orr, qpp = o[mk], p[mk], oref[mk], qp[mk]
        r = np.corrcoef(oo, pp)[0, 1]; kge = 1 - np.sqrt((r - 1) ** 2 + (pp.std() / oo.std() - 1) ** 2 + (pp.mean() / oo.mean() - 1) ** 2)
        nse_clim = 1 - np.sum((oo - pp) ** 2) / np.sum((oo - orr) ** 2)
        oa, pa = oo >= Q90, pp >= Q90; tp, fp, fn = np.sum(oa & pa), np.sum(~oa & pa), np.sum(oa & ~pa)
        crps = C.met(oo, qpp)["CRPS"] if disp in cands else round(float(np.mean(np.abs(oo - pp))), 3)
        blk = max(1, round(-1 / np.log(min(max(lag1(oo - pp), 1e-3), 0.999))))
        ci = boot_ci(oo, pp, blk)
        rows.append(dict(modelo=disp, h=Lh, N=len(oo), NSE=round(nse(oo, pp), 3), NSE_clim=round(float(nse_clim), 3),
                         KGE=round(float(kge), 3), CRPS=crps, CSI=round(tp / (tp + fp + fn), 3) if (tp + fp + fn) else 0.0,
                         NSE_lo=ci[0], NSE_hi=ci[1]))
    best = max([r for r in rows if r["h"] == Lh], key=lambda r: r["NSE_clim"])
    log.info(f"h{Lh:2d} mejor(NSE*clim)={best['modelo']} {best['NSE_clim']} | " +
             " ".join(f"{r['modelo'][:10]}={r['NSE']}" for r in rows if r["h"] == Lh))
dd = pd.DataFrame(rows); dd.to_csv(OUT / "188_benchmark_test.csv", index=False)

# ── figura: NSE vs horizonte con IC95 (líneas) + leaderboard h14 (barras) ──
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
COL = {"canónico": "#0e7d90", "canónico+GRU": "#2f9e6f", "canónico+RevIN": "#e08a2a",
       "canónico+GRU+RevIN": "#d24e39", "persistencia": "#888", "climatología": "#b8860b"}
for disp in COL:
    s = dd[dd.modelo == disp].sort_values("h")
    ax[0].plot(s["h"], s["NSE"], "-o", color=COL[disp], lw=1.8, ms=4, label=disp)
    ax[0].fill_between(s["h"], s["NSE_lo"], s["NSE_hi"], color=COL[disp], alpha=0.10)
ax[0].set_xlabel("horizonte (días)"); ax[0].set_ylabel("NSE (banda = IC95 block-boot)")
ax[0].set_title("(a) NSE vs horizonte — test 2024-25", fontsize=11, fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(lw=.3, alpha=.4); ax[0].set_xticks([1, 3, 7, 14])
d14 = dd[dd.h == 14].sort_values("NSE_clim")
ax[1].barh(np.arange(len(d14)), d14["NSE_clim"], color=[COL[m] for m in d14["modelo"]])
ax[1].errorbar(d14["NSE"], np.arange(len(d14)), xerr=[d14["NSE"] - d14["NSE_lo"], d14["NSE_hi"] - d14["NSE"]],
               fmt="D", color="#333", ms=4, capsize=3, label="NSE ± IC95")
ax[1].set_yticks(np.arange(len(d14))); ax[1].set_yticklabels(d14["modelo"], fontsize=8)
ax[1].axvline(0, color="#999", lw=.8, ls=":"); ax[1].set_xlabel("skill (h14): barra=NSE*clim, ♦=NSE±IC95")
ax[1].set_title("(b) Leaderboard h14", fontsize=11, fontweight="bold"); ax[1].legend(fontsize=8); ax[1].grid(axis="x", lw=.3, alpha=.4)
fig.tight_layout(); fig.savefig(FIG / "DI_benchmark_test.png", dpi=150)
log.info(f"figura: {FIG/'DI_benchmark_test.png'}"); log.info("BENCHMARK_TEST_DONE")
