#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 193 — SIGNIFICANCIA PAREADA (Diebold-Mariano por block-bootstrap + Wilcoxon + corrección Holm).
Responde al golpe letal del red-team: "con N_eff~5 y 3 seeds, los IC a h14 solapan → el ganador NO
está separado". Aquí se prueba, PARA CADA horizonte, si el ganador canónico+GRU bate a cada rival con
un test PAREADO día-a-día (no IC marginales) — reclamando solo lo que sobrevive a Holm.

Sobre test 2024-25, aforo real, ensemble 3 seeds (checkpoints 179, SIN reentrenar):
  · pérdida determinista por día  e_i = (obs_i − p50_i)^2   (para NSE/MSE)
  · pérdida probabilística por día c_i = CRPS3 (pinball sobre p10/p50/p90)
  · diferencial d_i = loss(rival) − loss(ganador); d>0 ⇒ el ganador es mejor
  · DM por moving-block-bootstrap del promedio de d (bloque = decorrelación) → IC95 + p bilateral
  · Wilcoxon signed-rank (referencia no paramétrica; anticonservador con autocorrelación → se declara)
  · Holm sobre las 5 comparaciones por (métrica, horizonte)

Salida: outputs/ml_Q/193_significancia_pareada.csv · reports/figures/DI_significancia_pareada.png
Run (.venv313): python scripts/06_eval/193_significancia_pareada.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sig193"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90; QS = [0.1, 0.5, 0.9]; LEADS = [1, 3, 7, 14]; rng = np.random.default_rng(0)


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


def crps_c(o, qp):
    t = np.zeros(len(o))
    for i, qv in enumerate(QS):
        e = o - qp[:, i]; t += np.where(e >= 0, qv * e, (qv - 1) * e)
    return t / 3


def lag1(x):
    x = x[np.isfinite(x)]; x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if len(x) > 2 and np.sum(x * x) > 0 else 0.0


def block_boot(d, blk, reps=3000):
    n = len(d); nb = int(np.ceil(n / blk)); smax = max(1, n - blk + 1); out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, smax, nb)])[:n] % n
        out[r] = d[idx].mean()
    lo, hi = np.percentile(out, [2.5, 97.5]); p = 2 * min((out <= 0).mean(), (out >= 0).mean())  # bilateral
    return float(lo), float(hi), float(min(p, 1.0))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * pvals[idx]; run = max(run, val); adj[idx] = min(run, 1.0)
    return adj


# datos + normalización train
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
doy = dts.dayofyear.values
clim = np.array([np.nanmean(obs[(doy == d) & (dts.values <= np.datetime64("2022-12-31")) & np.isfinite(obs)]) if
                 np.isfinite(obs[(doy == d) & (dts.values <= np.datetime64("2022-12-31"))]).any() else np.nan
                 for d in range(367)])

FAM = {"canónico+GRU": ("canonico_GRU", M.TFTCanonMatrix, {**M.BASE, "rec": "gru"}),
       "canónico": ("canonico", M.TFTCanonMatrix, dict(M.BASE)),
       "canónico+RevIN": ("canonico_RevIN", TFTCanonRevIN, dict(M.BASE)),
       "canónico+GRU+RevIN": ("canonico_GRU_RevIN", TFTCanonRevIN, {**M.BASE, "rec": "gru"})}
PR = {}
for disp, (fn, cls, cfg) in FAM.items():
    ps = []
    for s in range(3):
        mo = cls(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, **cfg).to(DEV)
        mo.load_state_dict(torch.load(OUT / f"179_{fn}_seed{s}.pt", map_location=DEV)); mo.eval()
        X = T.seqs_for_enc(df, te, enc, H, (mu, sd))
        with torch.no_grad(): ps.append(mo(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
    PR[disp] = np.mean(ps, 0)

REF = "canónico+GRU"; RIVALS = ["canónico", "canónico+RevIN", "canónico+GRU+RevIN", "persistencia", "climatología"]
rows = []
for Lh in LEADS:
    o = np.array([obs[i + Lh - 1] for i in te]); m = np.isfinite(o)
    # predicciones por modelo (p50 y qp) sobre días finitos
    def get(disp):
        if disp == "persistencia":
            p = np.array([q[i - 1] for i in te]); return p, np.stack([p, p, p], 1)
        if disp == "climatología":
            p = np.array([clim[dts[i + Lh - 1].dayofyear] for i in te]); return p, np.stack([p, p, p], 1)
        qp = PR[disp][:, Lh - 1, :]; return qp[:, 1], qp
    p_ref, qp_ref = get(REF)
    e_ref = (o - p_ref) ** 2; c_ref = crps_c(o, qp_ref)
    tmp = []
    for rv in RIVALS:
        p_rv, qp_rv = get(rv); e_rv = (o - p_rv) ** 2; c_rv = crps_c(o, qp_rv)
        for metric, dref, driv in [("MSE", e_ref, e_rv), ("CRPS", c_ref, c_rv)]:
            d = (driv - dref)[m]                      # >0 ⇒ ganador mejor
            blk = max(1, round(-1 / np.log(min(max(lag1(d), 1e-3), 0.999))))
            lo, hi, pboot = block_boot(d, blk)
            try: pw = float(wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
            except Exception: pw = np.nan
            tmp.append(dict(h=Lh, rival=rv, metric=metric, N=int(m.sum()), decorr=blk,
                            delta=round(float(d.mean()), 4), IC_lo=round(lo, 4), IC_hi=round(hi, 4),
                            p_boot=round(pboot, 4), p_wilcoxon=round(pw, 4)))
    # Holm por (métrica, horizonte) sobre las 5 comparaciones
    for metric in ("MSE", "CRPS"):
        grp = [r for r in tmp if r["metric"] == metric]
        adj = holm(np.array([r["p_boot"] for r in grp]))
        for r, a in zip(grp, adj):
            r["p_boot_holm"] = round(float(a), 4); r["signif_holm"] = "sí" if a < 0.05 else "no"
    rows += tmp
    sig_win = [r["rival"] for r in tmp if r["metric"] == "MSE" and r["signif_holm"] == "sí" and r["delta"] > 0]
    sig_lose = [r["rival"] for r in tmp if r["metric"] == "MSE" and r["signif_holm"] == "sí" and r["delta"] < 0]
    log.info(f"h{Lh:2d} (MSE): ganador SUPERA signif(Holm) a → {sig_win or '—'} | ES SUPERADO signif por → {sig_lose or '—'}")

res = pd.DataFrame(rows); res.to_csv(OUT / "193_significancia_pareada.csv", index=False)
log.info("\n=== ΔMSE (rival−ganador) e IC95 block-boot + Holm, por horizonte ===\n" +
         res[res.metric == "MSE"].pivot_table(index="rival", columns="h", values="delta").to_string())

# figura: matriz de significancia (rival × horizonte) para MSE y CRPS
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for k, metric in enumerate(("MSE", "CRPS")):
    sub = res[res.metric == metric]
    piv = sub.pivot_table(index="rival", columns="h", values="delta").reindex(RIVALS)
    sg = sub.pivot_table(index="rival", columns="h", values="signif_holm", aggfunc="first").reindex(RIVALS)
    Z = piv.values
    im = ax[k].imshow(np.sign(Z) * np.log1p(np.abs(Z)), cmap="RdYlGn", aspect="auto")
    ax[k].set_xticks(range(len(LEADS))); ax[k].set_xticklabels([f"h{h}" for h in LEADS])
    ax[k].set_yticks(range(len(RIVALS))); ax[k].set_yticklabels(RIVALS, fontsize=8)
    for i in range(len(RIVALS)):
        for j in range(len(LEADS)):
            star = "★" if sg.values[i, j] == "sí" else ""
            ax[k].text(j, i, f"{Z[i,j]:+.2f}\n{star}", ha="center", va="center", fontsize=7)
    ax[k].set_title(f"({'a' if k==0 else 'b'}) Δ{metric} rival−ganador  (★ = signif. Holm p<0.05)", fontsize=10.5, fontweight="bold")
fig.suptitle("Significancia PAREADA: ¿el ganador canónico+GRU supera a cada rival? (test 2024-25, DM block-boot + Holm)", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(FIG / "DI_significancia_pareada.png", dpi=150)
log.info(f"figura: {FIG/'DI_significancia_pareada.png'}"); log.info("SIGNIFICANCIA_PAREADA_DONE")
