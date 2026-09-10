#!/usr/bin/env python3
"""
Script 129 — Verificación JUSTA de HydroST: ¿realmente se comporta tan bien?

Hallazgo: q_next_1d[t]=q_mm[t+1]; ventana HydroST termina en idx-1 y predice Q(idx+1)
=> el "h=1" de HydroST es en realidad LEAD 2 días. Aquí se compara HydroST contra el TFT
en las MISMAS fechas objetivo, al lead 1 y lead 2, con métricas completas. Run in .venv313.
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import optuna
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(message)s"); log=logging.getLogger("hf")
def imp(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
T=imp("t114",ROOT/"scripts/05_models/114_tft_intensive.py"); R=T.R; DEV=R.DEV; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]
Hm=imp("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")

def crps(o,qp):
    t=0
    for i,q in enumerate(QS): e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
def M(o,qp):
    o=np.asarray(o,float); p=qp[:,1]; m=np.isfinite(o)&np.isfinite(p); o,p,qp=o[m],p[m],qp[m]
    nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2); oa,pa=o>=Q90,p>=Q90
    tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    return dict(N=int(m.sum()),NSE=round(nse,3),KGE=round(kge(o,p),3),MAE=round(np.mean(np.abs(o-p)),2),
                CRPS=round(crps(o,qp),3),CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3))

def main():
    # obs reales por fecha
    ob=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]); real={d:v for d,v in zip(ob["date"],ob["q_santo_domingo_47e214d2"]) if np.isfinite(v)}
    # ── HydroST: pred (m3/s) por fecha objetivo (=dates[idx]+1) ──
    bph=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    d7=Hm.load_data(); dyn,stat,tgt,hd,_=Hm.build_arrays(d7); Hm.ENC=bph["enc"]; hpd=pd.DatetimeIndex(hd)
    dsh=Hm.HydroDataset(dyn,stat,tgt,hd,hpd[hpd>=Hm.TEST_START]); ldh=DataLoader(dsh,batch_size=256,shuffle=False)
    tdate=[pd.Timestamp(hd[i])+pd.Timedelta(days=1) for i,_ in dsh.indices]
    prs=[]
    for s in range(3):
        mm=Hm.HydroST(hid=bph["hid"],heads=bph["heads"],layers=bph["layers"],dropout=bph["drop"]).to(DEV)
        mm.load_state_dict(torch.load(OUT/f"115_hydrost_best_seed{s}.pt",map_location=DEV,weights_only=True)); mm.eval()
        pp=[]
        with torch.no_grad():
            for xd,xs,y in ldh: pp.append(mm(xd.to(DEV),xs.to(DEV)).cpu().numpy())
        prs.append(np.concatenate(pp,0))
    PH=np.mean(prs,0)/Hm.Q_CONV  # (N,3) m3/s
    hyd={d:PH[k] for k,d in enumerate(tdate)}
    # ── TFT: pred por fecha objetivo, lead 1 y lead 2 ──
    mt=np.load(OUT/"125_meta.npz"); te=list(mt["te"]); PR=np.load(OUT/"125_PR_test.npy")
    df=R.build(H); dts=df.index
    tft={1:{},2:{}}
    for pos,i in enumerate(te):
        for L in (1,2):
            D=dts[i+L-1]; tft[L][D]=PR[pos,L-1,:]
    # ── comparación en fechas objetivo comunes con obs real ──
    log.info("HydroST 'h=1' = LEAD 2 real. Comparación en las MISMAS fechas objetivo (obs reales):\n")
    # HydroST vs su propio lead real (2)
    D_h=[d for d in tdate if d in real]
    o=np.array([real[d] for d in D_h]); qp=np.stack([hyd[d] for d in D_h]); log.info(f"HydroST (lead 2 real)     : {M(o,qp)}")
    for L in (1,2):
        Dc=[d for d in D_h if d in tft[L]]; o2=np.array([real[d] for d in Dc]); qp2=np.stack([tft[L][d] for d in Dc])
        log.info(f"TFT lead {L} (mismas fechas): {M(o2,qp2)}")
    log.info("\n(comparar HydroST 'lead2' con TFT lead 2 = manzana-con-manzana; TFT lead 1 usa 1 día más de info)")

if __name__=="__main__":
    main()
