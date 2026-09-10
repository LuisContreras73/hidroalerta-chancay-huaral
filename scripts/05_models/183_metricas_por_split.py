#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 183 — Métricas por SPLIT (train/val/test) + periodo general, del modelo ganador
canónico+GRU. Enfoque "tránsito hidrológico": reportar el skill como se reporta un modelo
lluvia-escorrentía (calibración/validación/test), para revisores.

Carga los checkpoints 179_canonico_GRU_seed{0,1,2} (NO re-entrena) y evalúa el ensemble de 3 seeds:
  · vs AFORO REAL (obs) donde exista — el reporte honesto por split.
  · en TRAIN también vs el TARGET de entrenamiento (q = GR4J-derivado), para mostrar el "ajuste".
Split idéntico al del harness: EM=90 · train ≤2022 · val 2023 · test 2024+. FUT=[sin,cos] (calendario).

Salida: outputs/ml_Q/183_metricas_por_split.csv
Run (.venv313): python scripts/05_models/183_metricas_por_split.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, optuna, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("split183"); optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("m173", ROOT / "scripts/05_models/173_canonical_arch_matrix.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R, T, DEV, C = M.R, M.T, M.DEV, M.C
bp = optuna.load_study(study_name="intv_h14", storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
H = 14; EM = 90; enc = bp["enc"]

df = R.build(H); dts = df.index; q = df["q"].values; obs = df["obs"].values
qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
mu = {"p": df[R.PAST].iloc[[i for i in range(EM, len(df)-H) if dts[i] <= pd.Timestamp("2022-12-31")]].values.mean(0),
      "f": df[R.FUT].iloc[[i for i in range(EM, len(df)-H) if dts[i] <= pd.Timestamp("2022-12-31")]].values.mean(0)}
sd = {"p": df[R.PAST].iloc[[i for i in range(EM, len(df)-H) if dts[i] <= pd.Timestamp("2022-12-31")]].values.std(0)+1e-6,
      "f": df[R.FUT].iloc[[i for i in range(EM, len(df)-H) if dts[i] <= pd.Timestamp("2022-12-31")]].values.std(0)+1e-6}

fin = lambda i: np.isfinite(q[i-EM:i+H]).all()
tr = [i for i in range(EM, len(df)-H) if dts[i] <= pd.Timestamp("2022-12-31") and fin(i)]
va = [i for i in range(EM, len(df)-H) if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31") and fin(i)]
te = [i for i in range(EM, len(df)-H) if dts[i] >= pd.Timestamp("2024-01-01")]
gen = tr + va + te
log.info(f"train {len(tr)} · val {len(va)} · test {len(te)} · general {len(gen)}")

# modelos (ensemble 3 seeds)
models = []
for s in range(3):
    m = M.TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"], H=H,
                         drop=bp["drop"], q_mu=qmu, q_sd=qsd, rec="gru").to(DEV)
    m.load_state_dict(torch.load(OUT/f"179_canonico_GRU_seed{s}.pt", map_location=DEV)); m.eval(); models.append(m)


def predict(idxs):
    X = T.seqs_for_enc(df, idxs, enc, H, (mu, sd))
    ps = []
    for m in models:
        with torch.no_grad():
            ps.append(m(X[0].to(DEV), X[1].to(DEV), X[2].to(DEV)).cpu().numpy())
    return np.mean(ps, 0)


rows = []
for nombre, idxs in [("train", tr), ("val", va), ("test", te), ("general", gen)]:
    PR = predict(idxs)
    for h in (1, 3, 7, 14):
        o_obs = np.array([obs[i+h-1] for i in idxs])          # aforo REAL
        o_q = np.array([q[i+h-1] for i in idxs])              # target de entrenamiento (GR4J-derivado)
        m_obs = C.met(o_obs, PR[:, h-1, :])
        rows.append(dict(split=nombre, h=h, ref="aforo_real", **m_obs))
        if nombre in ("train", "general"):                    # el "ajuste" al target reconstruido
            rows.append(dict(split=nombre, h=h, ref="target_GR4J", **C.met(o_q, PR[:, h-1, :])))
    n_obs = int(np.isfinite([obs[i+0] for i in idxs]).sum())
    log.info(f"{nombre:8s} h1 NSE(aforo)={[r for r in rows if r['split']==nombre and r['h']==1 and r['ref']=='aforo_real'][0]['NSE']} (N_obs≈{n_obs})")

agg = pd.DataFrame(rows); agg.to_csv(OUT/"183_metricas_por_split.csv", index=False)
log.info("\n=== NSE vs AFORO REAL por split y horizonte ===\n" +
         agg[agg.ref == "aforo_real"].pivot_table(index="split", columns="h", values="NSE").to_string())
log.info("\n=== KGE vs aforo real ===\n" +
         agg[agg.ref == "aforo_real"].pivot_table(index="split", columns="h", values="KGE").to_string())
log.info("SPLIT_DONE; 183_metricas_por_split.csv")
