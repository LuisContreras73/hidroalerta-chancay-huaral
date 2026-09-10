#!/usr/bin/env python3
"""
Script 139 — HydroST entrenado a leads 3, 5, 7, 14 (completa la tabla por horizonte).
HydroST es de salida única: una red por horizonte. target_leadL[idx] = Q(dates[idx+L-1])
(ventana hasta idx-1 → lead L). Mismo mejor config (Script 115), 3 seeds, eval obs reales.
Run in .venv313 (background). Salida: 139_hydrost_multilead.csv
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import optuna
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hml")
spec=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
H=importlib.util.module_from_spec(spec); spec.loader.exec_module(H); DEV=H.DEVICE; QC=H.Q_CONV; Q90=40.89

def kge(o,p):
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)

def main():
    bp=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    log.info(f"Config: {bp}")
    df=H.load_data(); dyn,stat,tgt0,dates,scl=H.build_arrays(df); dpd=pd.DatetimeIndex(dates)
    qmm=np.concatenate([[np.nan],tgt0[:-1]]).astype(np.float32)   # q_mm[idx]=q_next_1d[idx-1]
    ob=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]); real={d:v for d,v in zip(ob["date"],ob["q_santo_domingo_47e214d2"]) if np.isfinite(v)}
    H.ENC=bp["enc"]; H.LR1=bp["lr1"]; H.LR2=bp["lr2"]; H.WD=bp["wd"]; H.PAT=25; H.ALPHA=bp["alpha"]; H.ALERT_W=bp["alert_w"]
    pre=dpd[dpd<=H.PRETRAIN_END]; ftd=dpd[(dpd>=H.FT_START)&(dpd<=H.FT_END)]; vad=dpd[(dpd>=H.VAL_START)&(dpd<=H.VAL_END)]
    rows=[]
    for L in [3,5,7,14]:
        tgtL=np.concatenate([qmm[L-1:],[np.nan]*(L-1)]).astype(np.float32)  # target_leadL[idx]=q_mm[idx+L-1]
        prs=[]
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            dsp=H.HydroDataset(dyn,stat,tgtL,dates,pre); dsf=H.HydroDataset(dyn,stat,tgtL,dates,ftd); dsv=H.HydroDataset(dyn,stat,tgtL,dates,vad)
            lp=DataLoader(dsp,batch_size=H.BATCH1,shuffle=True); lf=DataLoader(dsf,batch_size=bp["batch2"],shuffle=True); lv=DataLoader(dsv,batch_size=bp["batch2"],shuffle=False)
            m=H.HydroST(hid=bp["hid"],heads=bp["heads"],layers=bp["layers"],dropout=bp["drop"]).to(DEV)
            m=H.train_phase1(m,lp,50); m=H.train_phase2(m,lf,lv,150); prs.append(m)
        # eval test: pred_date = dates[idx+L-1]
        dst=H.HydroDataset(dyn,stat,tgtL,dates,dpd[dpd>=H.TEST_START]); ldt=DataLoader(dst,batch_size=256,shuffle=False)
        pdt=[pd.Timestamp(dates[i+L-1]) for i,_ in dst.indices]; yreal=np.array([real.get(d,np.nan) for d in pdt])
        allp=[]
        for m in prs:
            m.eval(); pp=[]
            with torch.no_grad():
                for xd,xs,y in ldt: pp.append(m(xd.to(DEV),xs.to(DEV)).cpu().numpy())
            allp.append(np.concatenate(pp,0))
        PH=np.mean(allp,0)/QC; mk=np.isfinite(yreal); o=yreal[mk]; p=PH[mk,1]
        nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2); oa,pa=o>=Q90,p>=Q90
        tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
        csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
        rows.append(dict(lead=L,N=int(mk.sum()),model="HydroST",NSE=round(nse,3),KGE=round(kge(o,p),3),
                         MAE=round(float(np.mean(np.abs(o-p))),2),CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3)))
        log.info(f"HydroST lead {L}: NSE={nse:.3f} POD={pod:.3f} FAR={far:.3f} N={mk.sum()}")
        pd.DataFrame(rows).to_csv(OUT/"139_hydrost_multilead.csv",index=False)  # guardado incremental
    log.info("HYDROST_MULTILEAD_DONE; guardado 139_hydrost_multilead.csv")

if __name__=="__main__":
    main()
