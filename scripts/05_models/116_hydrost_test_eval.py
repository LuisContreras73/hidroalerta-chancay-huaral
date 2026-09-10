#!/usr/bin/env python3
"""
Script 116 — Eval SELLADO de HydroST TUNEADO en test 2024+ (solo obs REALES).

Carga los 3 checkpoints del mejor config (Script 115) y evalúa a h=1 contra el caudal
OBSERVADO real (no el q_next_1d rellenado con GR4J → evita contaminación, Regla 5).
Cierra el leaderboard justo: HydroST tuneado vs TFT/LightGBM/persistencia a h=1.

Run in .venv313:
    python scripts/05_models/116_hydrost_test_eval.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import optuna

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hstest")
spec=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
H=importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
DEV=H.DEVICE; QC=H.Q_CONV

def main():
    st=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUTML/'115_hydrost.db'}")
    bp=st.best_params; log.info(f"Best config: valPin={st.best_value:.4f} {bp}")

    # Obs REALES (m3/s) por fecha
    obs=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"])
    real={d:v for d,v in zip(obs["date"],obs["q_santo_domingo_47e214d2"]) if np.isfinite(v)}

    df=H.load_data(); dyn,stat,tgt,dates,scl=H.build_arrays(df); H.ENC=bp["enc"]
    tpd=pd.DatetimeIndex(dates); test_dates=tpd[tpd>=H.TEST_START]
    ds=H.HydroDataset(dyn,stat,tgt,dates,test_dates)
    ld=DataLoader(ds,batch_size=256,shuffle=False)
    # fecha del target = dates[idx]+1d (q_next_1d); comparar contra obs real ahí
    pred_dates=[pd.Timestamp(dates[i])+pd.Timedelta(days=1) for i,_ in ds.indices]
    yreal=np.array([real.get(d,np.nan) for d in pred_dates])
    mk=np.isfinite(yreal); log.info(f"Test seq={len(ds.indices)}, con obs REAL={mk.sum()}")

    preds=[]
    for s in range(3):
        m=H.HydroST(hid=bp["hid"],heads=bp["heads"],layers=bp["layers"],dropout=bp["drop"]).to(DEV)
        m.load_state_dict(torch.load(OUTML/f"115_hydrost_best_seed{s}.pt",map_location=DEV,weights_only=True)); m.eval()
        p10,p50,p90=[],[],[]
        with torch.no_grad():
            for xd,xs,y in ld:
                pr=m(xd.to(DEV),xs.to(DEV)).cpu().numpy(); p10+=list(pr[:,0]); p50+=list(pr[:,1]); p90+=list(pr[:,2])
        preds.append(np.stack([p10,p50,p90],1))
    P=np.mean(preds,0)/QC   # mm/day -> m3/s
    o=yreal[mk]; p50=P[mk,1]; p10=P[mk,0]; p90=P[mk,2]
    met=H.compute_metrics(o,p50,p10,p90)
    log.info(f"HydroST TUNEADO — TEST 2024+ (obs reales, N={mk.sum()}, h=1):")
    log.info(f"  NSE={met['NSE']:.3f} NSE_sqrt={met['NSE_sqrt']:.3f} J_alert={met['J_alert']:.3f} CSI={met['CSI']:.3f} POD={met['POD']:.3f} FAR={met['FAR']:.3f} PICP={met.get('PICP',float('nan')):.3f}")
    pd.DataFrame([dict(model="HydroST-tuned",split="test2024_realobs",N=int(mk.sum()),
                       **{k:round(float(met[k]),3) for k in ["NSE","NSE_sqrt","J_alert","CSI","POD","FAR"]})]).to_csv(OUTML/"116_hydrost_test.csv",index=False)
    log.info("Saved: 116_hydrost_test.csv")

if __name__=="__main__":
    main()
