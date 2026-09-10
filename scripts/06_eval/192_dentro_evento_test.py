#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 192 — "¿Qué pasó DENTRO del modelo en el evento?" — deep-dive del mayor evento de test
(crecida ene-feb 2024, pico obs 56.7 el 2024-02-02), conectando MECANISMO → predicción → decisión.

Tres paneles:
  (a) Hidrograma del evento a h7: aforo real vs canónico+GRU (banda p10-p90) vs Chronos-2 vs híbrido.
  (b) VSN del encoder en las emisiones que apuntan al EVENTO vs CALMA (¿qué variables mira el modelo
      cuando pronostica la crecida?) — hook sobre canónico+GRU seed0.
  (c) Atención a los días de lookback en la emisión que pronostica el PICO (¿a qué días mira?).

Salida: outputs/ml_Q/192_dentro_evento.csv · reports/figures/DI_dentro_evento_test.png
Run (.venv313): python scripts/06_eval/192_dentro_evento_test.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"; FIG = ROOT / "reports/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("dent192"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90; LH = 7

df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
trm = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31")]
mu = {"p": df[R.PAST].iloc[trm].values.mean(0), "f": df[R.FUT].iloc[trm].values.mean(0)}
sd = {"p": df[R.PAST].iloc[trm].values.std(0) + 1e-6, "f": df[R.FUT].iloc[trm].values.std(0) + 1e-6}

m0 = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                      drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
m0.load_state_dict(torch.load(OUT / "179_canonico_GRU_seed0.pt", map_location=DEV)); m0.eval()
models = [m0]
for s in (1, 2):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                         drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT / f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)

# ── (a) hidrograma a h7 del evento ene-mar 2024 ──
win = [i for i in range(EM, len(df) - H) if pd.Timestamp("2024-01-01") <= dts[i + LH - 1] <= pd.Timestamp("2024-03-20")]
X = T.seqs_for_enc(df, win, enc, H, (mu, sd)); ps = []
for m in models:
    with torch.no_grad(): ps.append(m(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
PR = np.mean(ps, 0)
tgt = [dts[i + LH - 1] for i in win]; o_ev = np.array([obs[i + LH - 1] for i in win])
loc50 = PR[:, LH - 1, 1]; loc10 = PR[:, LH - 1, 0]; loc90 = PR[:, LH - 1, 2]
ch = pd.read_csv(OUT / "137_chronos2_preds.csv", parse_dates=["date"])
chm = ch[ch.lead == LH].set_index("date")["p50"]
ch50 = np.array([chm.get(t, np.nan) for t in tgt])
hyb50 = 0.5 * ch50 + 0.5 * loc50
pk = int(np.nanargmax(o_ev))
log.info(f"evento h{LH}: pico obs {o_ev[pk]:.1f} el {tgt[pk].date()} | local {loc50[pk]:.1f} chronos {ch50[pk]:.1f} híbrido {hyb50[pk]:.1f}")
pd.DataFrame(dict(date=tgt, obs=o_ev, local_p50=loc50, local_p10=loc10, local_p90=loc90,
                  chronos_p50=ch50, hibrido_p50=hyb50)).to_csv(OUT / "192_dentro_evento.csv", index=False)

# ── (b) VSN evento vs calma (test) ──
labels = [str(c) for c in R.PAST] + ["q_pasado"]
def vsn(idxs):
    Xi = T.seqs_for_enc(df, idxs, enc, H, (mu, sd)); store = []
    h = m0.vsn_past.register_forward_hook(lambda mm, i, o: store.append(o[1].detach().cpu()))
    with torch.no_grad():
        for s0 in range(0, len(idxs), 128):
            sl = slice(s0, s0 + 128); m0(Xi[0][sl].to(DEV), Xi[1][sl].to(DEV), Xi[2][sl].to(DEV))
    h.remove(); return torch.cat(store).mean((0, 1)).numpy()
ev_em = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01") and np.nanmax(q[i:i + H]) >= Q90 and np.isfinite(q[i - EM:i + H]).all()]
ca_em = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01") and np.nanmax(q[i:i + H]) < 0.4 * Q90 and np.isfinite(q[i - EM:i + H]).all()]
vsn_ev, vsn_ca = vsn(ev_em), vsn(ca_em)
log.info(f"VSN evento (n={len(ev_em)}): " + ", ".join(f"{labels[i]}={vsn_ev[i]:.2f}" for i in np.argsort(vsn_ev)[::-1][:4]))
log.info(f"VSN calma  (n={len(ca_em)}): " + ", ".join(f"{labels[i]}={vsn_ca[i]:.2f}" for i in np.argsort(vsn_ca)[::-1][:4]))

# ── (c) atención a lookback en la emisión que pronostica el pico ──
i_pk = win[pk]                       # emisión que apunta al pico a h7
Xp = T.seqs_for_enc(df, [i_pk], enc, H, (mu, sd)); att_store = []
hh = m0.attn.register_forward_hook(lambda mm, i, o: att_store.append(o[1].detach().cpu()))
with torch.no_grad(): m0(Xp[0].to(DEV), Xp[1].to(DEV), Xp[2].to(DEV))
hh.remove()
A = att_store[0][0].numpy(); dec2enc = A[enc:, :enc].mean(0)      # (enc,) atención media de los H a cada día
lookback = np.arange(enc, 0, -1)

# ── figura ──
fig = plt.subplots(1, 3, figsize=(17, 4.8))[0]; ax = fig.axes
ax[0].fill_between(tgt, loc10, loc90, color="#2f9e6f", alpha=0.18, label="local p10–p90")
ax[0].plot(tgt, o_ev, "-", color="#111", lw=2.2, label="aforo real")
ax[0].plot(tgt, loc50, "-o", color="#2f9e6f", ms=3, lw=1.6, label="canónico+GRU")
ax[0].plot(tgt, ch50, "-s", color="#7b3fb0", ms=3, lw=1.4, label="Chronos-2")
ax[0].plot(tgt, hyb50, "--", color="#0e7d90", lw=1.6, label="híbrido")
ax[0].axhline(Q90, color="#c0392b", lw=.9, ls=":", label="Q90 (alerta)")
ax[0].set_title(f"(a) Crecida ene-feb 2024 a h{LH} — pico {o_ev[pk]:.0f} m³/s", fontsize=10.5, fontweight="bold")
ax[0].set_ylabel("caudal (m³/s)"); ax[0].tick_params(axis="x", rotation=35, labelsize=7); ax[0].legend(fontsize=7); ax[0].grid(lw=.3, alpha=.4)
xo = np.argsort(vsn_ev); yy = np.arange(len(labels)); wid = 0.4
ax[1].barh(yy + wid/2, vsn_ev[xo], wid, color="#d24e39", label=f"evento (n={len(ev_em)})")
ax[1].barh(yy - wid/2, vsn_ca[xo], wid, color="#0e7d90", label=f"calma (n={len(ca_em)})")
ax[1].set_yticks(yy); ax[1].set_yticklabels([labels[i] for i in xo], fontsize=8)
ax[1].set_title("(b) VSN encoder: evento vs calma\n(¿qué mira al pronosticar la crecida?)", fontsize=10.5, fontweight="bold")
ax[1].set_xlabel("peso de selección"); ax[1].legend(fontsize=8); ax[1].grid(axis="x", lw=.3, alpha=.4)
ax[2].plot(-lookback, dec2enc, color="#0e7d90", lw=1.8)
ax[2].fill_between(-lookback, 0, dec2enc, color="#0e7d90", alpha=0.15)
ax[2].set_title(f"(c) Atención a lookback en la emisión del pico\n({dts[i_pk].date()} → +{LH}d)", fontsize=10.5, fontweight="bold")
ax[2].set_xlabel("días antes de la emisión"); ax[2].set_ylabel("atención media"); ax[2].grid(lw=.3, alpha=.4)
fig.suptitle("Dentro del modelo durante la crecida de test (ene-feb 2024) — mecanismo → predicción", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(FIG / "DI_dentro_evento_test.png", dpi=150)
log.info(f"figura: {FIG/'DI_dentro_evento_test.png'}"); log.info("DENTRO_EVENTO_DONE")
