#!/usr/bin/env python3
"""
Script 121 — Hidrograma del test para HydroST TUNEADO (estilo fig10, h=1).

Carga los 3 checkpoints del mejor config (Script 115) y grafica el test 2024+ (obs reales):
obs vs P50 + banda P10-P90 + umbral Q90. Complementa fig27 (TFT) para el leaderboard visual.

Run in .venv313:
    python scripts/05_models/121_hydrost_test_hydrograph.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import optuna
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hs-fig")
spec=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
Hm=importlib.util.module_from_spec(spec); spec.loader.exec_module(Hm)
DEV=Hm.DEVICE; QC=Hm.Q_CONV; Q90=40.89

def main():
    bp=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    obs=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"])
    real={d:v for d,v in zip(obs["date"],obs["q_santo_domingo_47e214d2"]) if np.isfinite(v)}
    df=Hm.load_data(); dyn,stat,tgt,dates,scl=Hm.build_arrays(df); Hm.ENC=bp["enc"]
    tpd=pd.DatetimeIndex(dates); ds=Hm.HydroDataset(dyn,stat,tgt,dates,tpd[tpd>=Hm.TEST_START])
    ld=DataLoader(ds,batch_size=256,shuffle=False)
    pdt=[pd.Timestamp(dates[i])+pd.Timedelta(days=1) for i,_ in ds.indices]
    yreal=np.array([real.get(d,np.nan) for d in pdt])
    preds=[]
    for s in range(3):
        m=Hm.HydroST(hid=bp["hid"],heads=bp["heads"],layers=bp["layers"],dropout=bp["drop"]).to(DEV)
        m.load_state_dict(torch.load(OUT/f"115_hydrost_best_seed{s}.pt",map_location=DEV,weights_only=True)); m.eval()
        pp=[]
        with torch.no_grad():
            for xd,xs,y in ld: pp.append(m(xd.to(DEV),xs.to(DEV)).cpu().numpy())
        preds.append(np.concatenate(pp,0))
    PR=np.mean(preds,0)/QC
    x=np.array(pdt); p10,p50,p90=PR[:,0],PR[:,1],PR[:,2]; mk=np.isfinite(yreal)
    nse=1-np.sum((yreal[mk]-p50[mk])**2)/np.sum((yreal[mk]-yreal[mk].mean())**2)
    fig,ax=plt.subplots(figsize=(15,5))
    ax.fill_between(x,p10,p90,color="#fdae6b",alpha=0.5,label="P10–P90 (HydroST)",zorder=1)
    ax.plot(x,p50,"-",color="#e6550d",lw=1.5,label="Pronóstico P50",zorder=3)
    ax.plot(x[mk],yreal[mk],"o",color="black",ms=3.2,label="Obs real",zorder=4)
    ax.axhline(Q90,ls="--",color="#d62728",lw=1.0,alpha=0.8,label=f"Umbral alerta Q90={Q90:.1f}")
    ax.set_title(f"HydroST tuneado — hidrograma del test (h=1, obs reales N={mk.sum()}, NSE={nse:.3f})",fontsize=12,loc="left")
    ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.legend(fontsize=8,ncol=4,loc="upper right")
    fig.tight_layout()
    for p in [OUT/"121_hydrost_test_hydrograph.png", FIGP/"fig30_hydrost_test_hydrograph.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); log.info("Saved: 121_hydrost_test_hydrograph.png + fig30")

if __name__=="__main__":
    main()
