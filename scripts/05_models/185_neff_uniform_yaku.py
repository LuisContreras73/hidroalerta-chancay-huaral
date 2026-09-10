#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 185 — Tres preguntas de rigor sobre las métricas por split (para revisores):

  (A) N_eff (tamaño de muestra EFECTIVO): el caudal diario está muy autocorrelacionado, así que
      "360 días de val" NO son 360 datos independientes. N_eff = N·(1−r1)/(1+r1) con r1 = autocorr
      lag-1 (aproximación AR(1)). Se reporta para la serie de aforo y para los ERRORES del ganador.

  (B) ESTRATEGIA DE UNIFORMIZACIÓN de métricas entre train/val/test. El NSE clásico no es comparable
      entre periodos porque su denominador es la varianza DE CADA periodo (Yaku tiene varianza enorme
      → el NSE "engaña"). Se uniformiza con:
        · NSE*_clim = skill contra UN ÚNICO referente fijo = la climatología estacional de TRAIN
          aplicada a todos los splits (mismo "modelo nulo" en los tres → comparable). [WMO]
        · descomposición KGE (r, α=ratio de variabilidad, β=sesgo) → dice POR QUÉ difiere.
        · IC 95% por BLOCK-BOOTSTRAP con bloque = tiempo de decorrelación (respeta N_eff) → dice si
          val-2023 es SIGNIFICATIVAMENTE peor o solo más incierto por pocos datos efectivos.

  (C) ¿Otro modelo fue MEJOR EN YAKU? Compara la familia 179 (canónico, +GRU, +RevIN, +GRU+RevIN) +
      persistencia + climatología, en la ventana val-2023 y en la crecida tight Yaku (feb-abr 2023).
      RevIN (regime-norm) fue DISEÑADO para saltos de régimen → hipótesis: ayuda en Yaku aunque en
      global perjudique. Sin re-entrenar (usa checkpoints 179_*).

Salida: outputs/ml_Q/185_neff_uniform.csv · outputs/ml_Q/185_yaku_models.csv
        reports/figures/DI_neff_uniform_yaku.png
Run (.venv313): python scripts/05_models/185_neff_uniform_yaku.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("neff185"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV, C = M.R, M.T, M.DEV, M.C
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90
rng = np.random.default_rng(0)


# ── RevIN (regime-norm) — réplica del rasgo del RA-TFT publicado (idéntico al de 179) ──
class TFTCanonRevIN(M.TFTCanonMatrix):
    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        mu = q_past.mean(1, keepdim=True); sd = q_past.std(1, keepdim=True) + 1e-3
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


# ── datos + normalización de train (idéntico a 183) ──
df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trmask = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trmask].values.mean(0), "f": df[R.FUT].iloc[trmask].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trmask].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trmask].values.std(0) + 1e-6}

fin = lambda i: np.isfinite(q[i - EM:i + H]).all()
tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and fin(i)]
va = [i for i in range(EM, len(df) - H) if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31") and fin(i)]
te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
yaku = [i for i in range(EM, len(df) - H) if pd.Timestamp("2023-02-01") <= dts[i] <= pd.Timestamp("2023-04-30") and fin(i)]
SPLITS = {"train": tr, "val_2023": va, "test_2024_25": te, "general": tr + va + te}

# ── climatología estacional de TRAIN (referente ÚNICO para el NSE* uniforme) ──
doy_all = dts.dayofyear.values
clim = np.full(367, np.nan)
for d in range(1, 367):
    vals = obs[(doy_all == d) & (dts.values <= np.datetime64("2022-12-31")) & np.isfinite(obs)]
    if len(vals): clim[d] = vals.mean()
# suavizado circular (ventana 15 días) para quitar ruido de muestreo
sm = clim.copy()
for d in range(1, 367):
    w = [(d + k - 1) % 366 + 1 for k in range(-7, 8)]
    v = clim[w]; sm[d] = np.nanmean(v) if np.isfinite(v).any() else clim[d]
clim = sm
clim_of = lambda idxs, L: np.array([clim[dts[i + L - 1].dayofyear] for i in idxs])


# ── modelos DL ──
def load_family(name, cls, cfg):
    ms = []
    for s in range(3):
        m = cls(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                drop=bp["drop"], q_mu=qmu, q_sd=qsd, **cfg).to(DEV)
        m.load_state_dict(torch.load(OUT / f"179_{name}_seed{s}.pt", map_location=DEV)); m.eval(); ms.append(m)
    return ms


FAMILIES = {
    "canónico": ("canonico", M.TFTCanonMatrix, dict(M.BASE)),
    "canónico+GRU": ("canonico_GRU", M.TFTCanonMatrix, {**M.BASE, "rec": "gru"}),
    "canónico+RevIN": ("canonico_RevIN", TFTCanonRevIN, dict(M.BASE)),
    "canónico+GRU+RevIN": ("canonico_GRU_RevIN", TFTCanonRevIN, {**M.BASE, "rec": "gru"}),
}
models = {disp: load_family(fn, cls, cfg) for disp, (fn, cls, cfg) in FAMILIES.items()}


def predict_ens(ms, idxs):
    X = T.seqs_for_enc(df, idxs, enc, H, (mu, sd))
    ps = []
    for m in ms:
        with torch.no_grad():
            ps.append(m(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
    return np.mean(ps, 0)   # (n,H,3)


# ── métricas uniformes ──
def nse(o, p):
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def uniform_metrics(o, p, oref):
    """o=aforo, p=pred, oref=climatología-de-train en las mismas fechas (referente fijo)."""
    r = np.corrcoef(o, p)[0, 1]
    alpha = p.std() / o.std(); beta = p.mean() / o.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse_clim = 1 - np.sum((o - p) ** 2) / np.sum((o - oref) ** 2)   # skill vs referente ÚNICO
    return dict(NSE=round(nse(o, p), 3), NSE_clim=round(float(nse_clim), 3), KGE=round(float(kge), 3),
                r=round(float(r), 3), alpha=round(float(alpha), 3), beta=round(float(beta), 3),
                MAE=round(float(np.mean(np.abs(o - p))), 2))


def lag1(x):
    x = x[np.isfinite(x)]; x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if len(x) > 2 else np.nan


def neff_ar1(x):
    r1 = lag1(x); N = int(np.isfinite(x).sum())
    r1c = min(max(r1, 0.0), 0.999)
    return N, r1, N * (1 - r1c) / (1 + r1c), (-1 / np.log(r1c) if r1c > 1e-6 else 1.0)


def block_boot_nse(o, p, block, reps=1000):
    N = len(o); nb = int(np.ceil(N / block)); out = []
    starts_max = max(1, N - block + 1)
    for _ in range(reps):
        s = rng.integers(0, starts_max, nb)
        idx = np.concatenate([np.arange(st, st + block) for st in s])[:N] % N
        ob, pb = o[idx], p[idx]
        if ob.std() > 1e-9: out.append(nse(ob, pb))
    return (round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)) if out else (np.nan, np.nan)


# ═══ (A)+(B): N_eff + métricas uniformes por split (modelo ganador canónico+GRU, h14) ═══
win = models["canónico+GRU"]
rowsA = []
for name, idxs in SPLITS.items():
    obs_ser = np.array([obs[i] for i in idxs])                      # serie de aforo (emisión)
    for Lh in (1, 14):
        PR = predict_ens(win, idxs); p = PR[:, Lh - 1, 1]
        o = np.array([obs[i + Lh - 1] for i in idxs]); oref = clim_of(idxs, Lh)
        m = np.isfinite(o) & np.isfinite(p) & np.isfinite(oref); o, p, oref = o[m], p[m], oref[m]
        err = o - p
        No, r1o, neffo, tauo = neff_ar1(o)
        Ne, r1e, neffe, taue = neff_ar1(err)
        um = uniform_metrics(o, p, oref)
        ci = block_boot_nse(o, p, max(1, round(taue)))
        rowsA.append(dict(split=name, h=Lh, N=len(o),
                          r1_obs=round(r1o, 3), Neff_obs=round(neffo, 1),
                          r1_err=round(r1e, 3), Neff_err=round(neffe, 1), decorr_dias=round(taue, 1),
                          **um, NSE_CI95_lo=ci[0], NSE_CI95_hi=ci[1]))
    log.info(f"{name:14s} N={len(idxs)} Neff_obs≈{rowsA[-2]['Neff_obs']} | h14 NSE={rowsA[-1]['NSE']} "
             f"NSE*clim={rowsA[-1]['NSE_clim']} CI95=[{rowsA[-1]['NSE_CI95_lo']},{rowsA[-1]['NSE_CI95_hi']}]")
dfA = pd.DataFrame(rowsA); dfA.to_csv(OUT / "185_neff_uniform.csv", index=False)


# ═══ (C): ¿otro modelo mejor en Yaku? ═══
def persist_pred(idxs, L):   # persistencia: último caudal conocido en emisión, plano
    return np.array([q[i - 1] for i in idxs])


WINDOWS = {"val_2023 (año Yaku)": va, "crecida Yaku (feb-abr 23)": yaku}
rowsC = []
for wname, idxs in WINDOWS.items():
    for Lh in (1, 3, 7, 14):
        o = np.array([obs[i + Lh - 1] for i in idxs]); oref = clim_of(idxs, Lh)
        peak = float(np.nanmax(o))
        cands = {disp: predict_ens(ms, idxs)[:, Lh - 1, 1] for disp, ms in models.items()}
        cands["persistencia"] = persist_pred(idxs, Lh)
        cands["climatología"] = oref.copy()
        for disp, p in cands.items():
            m = np.isfinite(o) & np.isfinite(p) & np.isfinite(oref); oo, pp, orr = o[m], p[m], oref[m]
            um = uniform_metrics(oo, pp, orr)
            i_pk = int(np.nanargmax(oo)); pk_err = round(float(pp[i_pk] - oo[i_pk]), 1)
            rowsC.append(dict(ventana=wname, modelo=disp, h=Lh, **um, err_en_pico=pk_err))
    # log del ganador por ventana a h14
    sub = [r for r in rowsC if r["ventana"] == wname and r["h"] == 14]
    best = max(sub, key=lambda r: r["NSE_clim"])
    log.info(f"[{wname}] h14 mejor por NSE*clim: {best['modelo']} ({best['NSE_clim']})  | "
             + " ".join(f"{r['modelo'].split('+')[-1][:6]}={r['NSE']}" for r in sub))
dfC = pd.DataFrame(rowsC); dfC.to_csv(OUT / "185_yaku_models.csv", index=False)


# ═══ figura ═══
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
# panel 1: N vs N_eff por split (h14, obs)
d1 = dfA[dfA.h == 14]; x = np.arange(len(d1)); w = 0.38
ax[0].bar(x - w / 2, d1["N"], w, label="N (nominal)", color="#b8c4cc")
ax[0].bar(x + w / 2, d1["Neff_obs"], w, label="N_eff (efectivo)", color="#0e7d90")
ax[0].set_xticks(x); ax[0].set_xticklabels(d1["split"], rotation=20, ha="right")
ax[0].set_ylabel("nº de observaciones"); ax[0].set_title("(A) Muestra efectiva N_eff = N·(1−r₁)/(1+r₁)", fontsize=11, fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(axis="y", lw=.3, alpha=.4)
for i, (nn, ne) in enumerate(zip(d1["N"], d1["Neff_obs"])):
    ax[0].text(i + w / 2, ne, f"{ne:.0f}", ha="center", va="bottom", fontsize=7)
# panel 2: NSE con IC95 block-bootstrap + NSE*clim (uniforme) por split, h14
d2 = dfA[dfA.h == 14].reset_index(drop=True); x = np.arange(len(d2))
lo = d2["NSE"] - d2["NSE_CI95_lo"]; hi = d2["NSE_CI95_hi"] - d2["NSE"]
ax[1].errorbar(x - .12, d2["NSE"], yerr=[lo, hi], fmt="o", color="#2f9e6f", capsize=4, label="NSE (IC95 block-boot)")
ax[1].plot(x + .12, d2["NSE_clim"], "s", color="#d24e39", label="NSE*clim (uniforme)")
ax[1].axhline(0, color="#999", lw=.8, ls=":"); ax[1].set_xticks(x); ax[1].set_xticklabels(d2["split"], rotation=20, ha="right")
ax[1].set_ylabel("skill (h14)"); ax[1].set_title("(B) Métricas uniformizadas + incertidumbre", fontsize=11, fontweight="bold")
ax[1].legend(fontsize=8); ax[1].grid(axis="y", lw=.3, alpha=.4)
# panel 3: modelos en la crecida Yaku (h14, NSE*clim)
d3 = dfC[(dfC.ventana == "crecida Yaku (feb-abr 23)") & (dfC.h == 14)].sort_values("NSE_clim")
cols3 = ["#d24e39" if "RevIN" in mm else ("#2f9e6f" if mm.startswith("can") else "#b8860b") for mm in d3["modelo"]]
ax[2].barh(np.arange(len(d3)), d3["NSE_clim"], color=cols3)
ax[2].set_yticks(np.arange(len(d3))); ax[2].set_yticklabels(d3["modelo"], fontsize=8)
ax[2].axvline(0, color="#999", lw=.8, ls=":"); ax[2].set_xlabel("NSE*clim (h14)")
ax[2].set_title("(C) ¿Qué modelo gana en la crecida Yaku?", fontsize=11, fontweight="bold")
ax[2].grid(axis="x", lw=.3, alpha=.4)
fig.tight_layout(); fig.savefig(FIG / "DI_neff_uniform_yaku.png", dpi=150)
log.info(f"figura: {FIG/'DI_neff_uniform_yaku.png'}")
log.info("NEFF_UNIFORM_YAKU_DONE")
