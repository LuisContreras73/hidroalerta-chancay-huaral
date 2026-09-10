#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 184 — ¿Cómo cambia la SELECCIÓN DE VARIABLES (VSN) del TFT durante eventos de lluvia
extrema (Ciclón Yaku 2023, El Niño costero 2017) vs periodos normales?

Captura por forward-hook los pesos de la VSN del encoder (que dicen cuánto pesa cada feature) y los
compara entre grupos de emisión: Yaku (feb-abr 2023), Niño costero (feb-abr 2017), crecida genérica
(el horizonte contiene Q>Q90), y NORMAL seco (jun-sep, caudal bajo). Hipótesis: en evento, el modelo
desplaza peso hacia lluvia/humedad/Niño-costero y menos al calendario. Ensemble de 3 seeds.

Salida: reports/figures/DI_vsn_eventos.png + dinamica_no_lineal/resultados/184_vsn_eventos.csv
Run (.venv313): python scripts/05_models/184_vsn_eventos.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
FIG = ROOT / "reports/figures"; RES = ROOT / "dinamica_no_lineal/resultados"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("vsn184"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV = M.R, M.T, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]; Q90 = R.Q90

df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr)+1e-6)
P = [i for i in range(EM, len(df)-H)]
mu = {"p": df[R.PAST].iloc[P].values.mean(0), "f": df[R.FUT].iloc[P].values.mean(0)}
sd = {"p": df[R.PAST].iloc[P].values.std(0)+1e-6, "f": df[R.FUT].iloc[P].values.std(0)+1e-6}
mu = {"p": df[R.PAST].iloc[[i for i in P if dts[i] <= pd.Timestamp("2022-12-31")]].values.mean(0),
      "f": df[R.FUT].iloc[[i for i in P if dts[i] <= pd.Timestamp("2022-12-31")]].values.mean(0)}
sd = {"p": df[R.PAST].iloc[[i for i in P if dts[i] <= pd.Timestamp("2022-12-31")]].values.std(0)+1e-6,
      "f": df[R.FUT].iloc[[i for i in P if dts[i] <= pd.Timestamp("2022-12-31")]].values.std(0)+1e-6}

# ── grupos de emisión ──
fin = lambda i: np.isfinite(q[i-EM:i+H]).all()
def win(a, b): return [i for i in range(EM, len(df)-H) if pd.Timestamp(a) <= dts[i] <= pd.Timestamp(b) and fin(i)]
crecida = [i for i in range(EM, len(df)-H) if fin(i) and np.nanmax(q[i:i+H]) > Q90]
normal = [i for i in range(EM, len(df)-H) if fin(i) and dts[i].month in (6, 7, 8, 9) and np.nanmax(q[i:i+H]) < 0.4*Q90]
rng = np.random.default_rng(0)
grupos = {
    "Yaku 2023 (feb-abr)": win("2023-02-01", "2023-04-30"),
    "Niño costero 2017 (feb-abr)": win("2017-02-01", "2017-04-30"),
    "Crecida (Q>Q90 en el horizonte)": list(rng.choice(crecida, min(400, len(crecida)), replace=False)),
    "Normal seco (jun-sep, Q bajo)": list(rng.choice(normal, min(400, len(normal)), replace=False)),
}
log.info({k: len(v) for k, v in grupos.items()})

# ── modelos + hook VSN ──
models = []
for s in range(3):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                         drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT/f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)
labels = [str(c) for c in R.PAST] + ["q_pasado"]

def vsn_importance(idxs):
    X = T.seqs_for_enc(df, idxs, enc, H, (mu, sd)); accum = []
    for m in models:
        store = []
        h = m.vsn_past.register_forward_hook(lambda mm, i, o: store.append(o[1].detach().cpu()))
        with torch.no_grad():
            for s0 in range(0, len(idxs), 256):
                sl = slice(s0, s0+256); m(X[0][sl].to(DEV), X[1][sl].to(DEV), X[2][sl].to(DEV))
        h.remove()
        w = torch.cat(store).mean((0, 1)).numpy()   # (NP+1,) media sobre emisiones y ENC
        accum.append(w)
    return np.mean(accum, 0)

imp = {g: vsn_importance(idx) for g, idx in grupos.items()}
tab = pd.DataFrame(imp, index=labels).round(3)
tab.to_csv(RES/"184_vsn_eventos.csv")
log.info("\n=== Peso VSN por variable y grupo ===\n" + tab.to_string())
# desplazamiento evento vs normal
base = imp["Normal seco (jun-sep, Q bajo)"]
log.info("\n=== Desplazamiento (Yaku − Normal) por variable ===")
for i, l in enumerate(labels):
    d = imp["Yaku 2023 (feb-abr)"][i] - base[i]
    if abs(d) > 0.01: log.info(f"  {l:10s}: {d:+.3f}")

# ── figura ──
fig, ax = plt.subplots(figsize=(11, 5.2)); x = np.arange(len(labels)); wd = 0.2
cols = {"Yaku 2023 (feb-abr)": "#d24e39", "Niño costero 2017 (feb-abr)": "#e08a2a",
        "Crecida (Q>Q90 en el horizonte)": "#6c4675", "Normal seco (jun-sep, Q bajo)": "#0e7d90"}
for k, (g, w) in enumerate(imp.items()):
    ax.bar(x + (k-1.5)*wd, w, wd, label=g, color=cols.get(g, "#888"))
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_ylabel("peso de selección VSN (medio)")
ax.set_title("Selección de variables (VSN encoder) por tipo de evento — canónico+GRU", fontsize=12, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", lw=0.3, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG/"DI_vsn_eventos.png", dpi=150)
log.info(f"figura: {FIG/'DI_vsn_eventos.png'}")
log.info("VSN_EVENTOS_DONE")
