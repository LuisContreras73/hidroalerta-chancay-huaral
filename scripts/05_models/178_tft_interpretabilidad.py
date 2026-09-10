#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 178 — Interpretabilidad del TFT: "cómo piensa el modelo" (diagnóstico interno).

Captura con FORWARD HOOKS (sin tocar el modelo) los mecanismos internos que el TFT ya calcula
pero descarta en su forward (`_`):
  1. VSN pasado — pesos de selección de variables del encoder (qué features del pasado usa).
  2. VSN futuro — pesos del decoder (¿cuánto usa la lluvia pronosticada GFS vs calendario?).
  3. Atención IMHA — mapa horizonte × día-pasado (a qué días atiende cada horizonte de pronóstico).

Usa el checkpoint del TFT+GFS (175, seed 0) sobre el test 2024-25. Figura de 3 paneles.
Salida: reports/figures/DI_interpretabilidad_tft.png
Run (.venv313): python scripts/05_models/178_tft_interpretabilidad.py
"""
import importlib.util
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
FIG = ROOT / "reports/figures"; FIG.mkdir(parents=True, exist_ok=True)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load("scripts/05_models/173_canonical_arch_matrix.py", "m173")
R, DEV = M.R, M.DEV
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; ENC = bp["enc"]; PAST = list(R.PAST); NP = len(PAST); TR_END = pd.Timestamp("2022-12-31")
df = R.build(H); dts = df.index; q = df["q"].values; arr = df[PAST].values
doy = dts.dayofyear.values
SIN = np.sin(2*np.pi*doy/365.25).astype(np.float32); COS = np.cos(2*np.pi*doy/365.25).astype(np.float32)
date_pos = {d: k for k, d in enumerate(dts)}

# ── GFS (idéntico a 175) ──
g = pd.read_csv(ROOT/"data/bronze/B11_gfs_daily_leads.csv", parse_dates=["init_date"])
piv = g.pivot_table(index="init_date", columns="lead", values="pr_gfs_mm")
QLEV = np.linspace(0.01, 0.99, 99); qm = {}
for L in range(1, H+1):
    ser = piv[L].dropna(); tri = ser.index[ser.index <= TR_END]; gf = ser.loc[tri].values
    ob = np.array([df["pr"].values[date_pos[d+pd.Timedelta(days=L)]] if (d+pd.Timedelta(days=L)) in date_pos else np.nan for d in tri])
    mm = np.isfinite(gf) & np.isfinite(ob)
    qm[L] = (np.quantile(gf[mm], QLEV), np.quantile(ob[mm], QLEV)) if mm.sum() >= 100 else None
def gq(v, L):
    mp = qm.get(L); return float(np.interp(v, mp[0], mp[1])) if (mp and np.isfinite(v)) else np.nan
GFS = np.zeros((len(df), H), np.float32); GAV = np.zeros((len(df), H), np.float32)
allqm = [gq(piv.at[d, L], L) for d in piv.index if d <= TR_END for L in range(1, H+1) if L in piv.columns and np.isfinite(piv.at[d, L])]
gmu = float(np.nanmean(allqm)); gsd = float(np.nanstd(allqm)+1e-6)
for d in piv.index:
    k = date_pos.get(d)
    if k is None: continue
    for L in range(1, H+1):
        v = piv.at[d, L] if L in piv.columns else np.nan
        if np.isfinite(v): GFS[k, L-1] = (gq(v, L)-gmu)/gsd; GAV[k, L-1] = 1.0
qmu = float(np.mean(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]]))
qsd = float(np.std(q[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]])+1e-6)
mu = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].mean(0)
sd = arr[[i for i in range(len(df)) if dts[i] <= TR_END and np.isfinite(q[i])]].std(0)+1e-6
te = [i for i in range(ENC, len(df)-H) if dts[i] >= pd.Timestamp("2024-01-01")]

def seqs(idxs):
    Xp, Qp, Xf = [], [], []
    for i in idxs:
        Xp.append(np.nan_to_num((arr[i-ENC:i]-mu)/sd)); Qp.append(np.nan_to_num(q[i-ENC:i]))
        hh = np.arange(1, H+1)
        Xf.append(np.stack([GFS[i], GAV[i], SIN[i+hh-1], COS[i+hh-1]], axis=1))
    t = lambda a: torch.tensor(np.array(a, np.float32))
    return t(Xp), t(Qp), t(Xf)

Xp, Qp, Xf = seqs(te)

# ── modelo + checkpoint ──
model = M.TFTCanonMatrix(NP, 4, hid=bp["hid"], heads=bp["heads"], H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd).to(DEV)
model.load_state_dict(torch.load(OUT/"175_gfs_seed0.pt", map_location=DEV)); model.eval()

store = {"vp": [], "vf": [], "att": []}
model.vsn_past.register_forward_hook(lambda m, i, o: store["vp"].append(o[1].detach().cpu()))
model.vsn_fut.register_forward_hook(lambda m, i, o: store["vf"].append(o[1].detach().cpu()))
model.attn.register_forward_hook(lambda m, i, o: store["att"].append(o[1].detach().cpu()))
with torch.no_grad():
    for s in range(0, len(te), 128):
        model(Xp[s:s+128].to(DEV), Qp[s:s+128].to(DEV), Xf[s:s+128].to(DEV))

vp = torch.cat(store["vp"]).mean((0, 1)).numpy()       # (NP+1,)
vf = torch.cat(store["vf"]).mean((0, 1)).numpy()        # (4,)
att = torch.cat(store["att"]).mean(0).numpy()           # (L, L), L=ENC+H
dec2enc = att[ENC:, :ENC]                                # (H, ENC): horizonte × día pasado
lab_p = [str(c) for c in PAST] + ["q_pasado"]
lab_f = ["GFS (lluvia pron.)", "GFS disp.", "sin(doy)", "cos(doy)"]
print("VSN pasado:", {lab_p[i]: round(float(vp[i]), 3) for i in np.argsort(vp)[::-1]})
print("VSN futuro:", {lab_f[i]: round(float(vf[i]), 3) for i in np.argsort(vf)[::-1]})

# ── figura ──
teal = LinearSegmentedColormap.from_list("teal", ["#f7fbfc", "#7fc9d6", "#0e7d90", "#083f49"])
fig = plt.figure(figsize=(13, 4.6)); plt.subplots_adjust(left=0.16, right=0.97, top=0.87, bottom=0.14, wspace=0.5)
ax1 = fig.add_subplot(1, 3, 1); o = np.argsort(vp)
ax1.barh(range(len(vp)), vp[o], color="#0e7d90"); ax1.set_yticks(range(len(vp))); ax1.set_yticklabels([lab_p[i] for i in o], fontsize=8)
ax1.set_title("a) VSN encoder — qué mira del pasado", fontsize=9.5, fontweight="bold"); ax1.set_xlabel("peso de selección medio", fontsize=8)
for i, v in enumerate(vp[o]): ax1.text(v+.003, i, f"{v:.2f}", va="center", fontsize=7)

ax2 = fig.add_subplot(1, 3, 2); o2 = np.argsort(vf)
cols = ["#e08a2a" if "GFS" in lab_f[i] else "#9bb" for i in o2]
ax2.barh(range(4), vf[o2], color=cols); ax2.set_yticks(range(4)); ax2.set_yticklabels([lab_f[i] for i in o2], fontsize=8)
ax2.set_title("b) VSN decoder — ¿usa el GFS?", fontsize=9.5, fontweight="bold"); ax2.set_xlabel("peso de selección medio", fontsize=8)
for i, v in enumerate(vf[o2]): ax2.text(v+.005, i, f"{v:.2f}", va="center", fontsize=7)

ax3 = fig.add_subplot(1, 3, 3)
im = ax3.imshow(dec2enc, aspect="auto", cmap=teal, origin="lower", extent=[-ENC, 0, 1, H])
ax3.set_title("c) Atención — a qué días atiende cada horizonte", fontsize=9.5, fontweight="bold")
ax3.set_xlabel("días antes de la emisión", fontsize=8); ax3.set_ylabel("horizonte de pronóstico (días)", fontsize=8)
cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04); cb.set_label("atención media", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.suptitle("Diagnóstico interno del TFT+GFS — mecanismos de selección y atención (test 2024-25, seed 0)", fontsize=11, fontweight="bold")
p = FIG/"DI_interpretabilidad_tft.png"; fig.savefig(p, dpi=150); print(f"Guardado: {p}")
