#!/usr/bin/env python3
"""
Script 131 — HydroST re-entrenado como LEAD-1 real (target = q_mm[idx] = Q en dates[idx]).

El HydroST original (78) predecía Q(idx+1) desde ventana hasta idx-1 = lead 2 (la ventana
excluía "hoy"). Aquí se corrige: target_lead1[idx] = q_mm[idx] = q_next_1d[idx-1] → predice
Q(dates[idx]) desde datos hasta dates[idx-1] = lead 1. Mismo mejor config (Script 115), 3 seeds,
presupuesto completo. Eval lead-1 en obs reales. Es un MODELO NUEVO (HydroST es de una salida).

Run in .venv313:
    python scripts/05_models/131_hydrost_lead1.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import optuna
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hl1")
spec=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
H=importlib.util.module_from_spec(spec); spec.loader.exec_module(H); DEV=H.DEVICE; QC=H.Q_CONV; Q90=40.89; QS=[0.1,0.5,0.9]

def crps(o,qp):
    t=0
    for i,q in enumerate(QS): e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)

def main():
    bp=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    log.info(f"Config (mismo mejor del 115): {bp}")
    df=H.load_data(); dyn,stat,tgt0,dates,scl=H.build_arrays(df)
    # target lead-1: q_mm[idx] = q_next_1d[idx-1] = tgt0[idx-1]
    tgt1=np.concatenate([[np.nan],tgt0[:-1]]).astype(np.float32)
    dpd=pd.DatetimeIndex(dates)
    pre=dpd[dpd<=H.PRETRAIN_END]; ftd=dpd[(dpd>=H.FT_START)&(dpd<=H.FT_END)]; vad=dpd[(dpd>=H.VAL_START)&(dpd<=H.VAL_END)]
    H.ENC=bp["enc"]; H.LR1=bp["lr1"]; H.LR2=bp["lr2"]; H.WD=bp["wd"]; H.PAT=25; H.ALPHA=bp["alpha"]; H.ALERT_W=bp["alert_w"]
    # obs reales
    ob=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]); real={d:v for d,v in zip(ob["date"],ob["q_santo_domingo_47e214d2"]) if np.isfinite(v)}
    res=[]
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        dsp=H.HydroDataset(dyn,stat,tgt1,dates,pre); dsf=H.HydroDataset(dyn,stat,tgt1,dates,ftd); dsv=H.HydroDataset(dyn,stat,tgt1,dates,vad)
        lp=DataLoader(dsp,batch_size=H.BATCH1,shuffle=True); lf=DataLoader(dsf,batch_size=bp["batch2"],shuffle=True); lv=DataLoader(dsv,batch_size=bp["batch2"],shuffle=False)
        m=H.HydroST(hid=bp["hid"],heads=bp["heads"],layers=bp["layers"],dropout=bp["drop"]).to(DEV)
        m=H.train_phase1(m,lp,50); m=H.train_phase2(m,lf,lv,150); torch.save(m.state_dict(),OUT/f"131_hydrost_lead1_seed{s}.pt")
        res.append(m); log.info(f"seed {s} entrenado")
    # eval lead-1: target = Q(dates[idx]); pred_date = dates[idx]
    dst=H.HydroDataset(dyn,stat,tgt1,dates,dpd[dpd>=H.TEST_START]); ldt=DataLoader(dst,batch_size=256,shuffle=False)
    pdt=[pd.Timestamp(dates[i]) for i,_ in dst.indices]; yreal=np.array([real.get(d,np.nan) for d in pdt])
    prs=[]
    for m in res:
        m.eval(); pp=[]
        with torch.no_grad():
            for xd,xs,y in ldt: pp.append(m(xd.to(DEV),xs.to(DEV)).cpu().numpy())
        prs.append(np.concatenate(pp,0))
    PH=np.mean(prs,0)/QC; mk=np.isfinite(yreal); o=yreal[mk]; qp=PH[mk]; p=qp[:,1]
    nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2); oa,pa=o>=Q90,p>=Q90
    tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    log.info(f"\n===== HydroST LEAD-1 (test 2024, obs reales N={mk.sum()}) =====")
    log.info(f"  NSE={nse:.3f} KGE={kge(o,p):.3f} MAE={np.mean(np.abs(o-p)):.2f} CRPS={crps(o,qp):.3f} CSI={csi:.3f} POD={pod:.3f} FAR={far:.3f}")
    pd.DataFrame({"date":np.array(pdt)[mk],"obs":o,"p10":qp[:,0],"p50":p,"p90":qp[:,2]}).to_csv(OUT/"131_hydrost_lead1_preds.csv",index=False)
    pd.DataFrame([dict(model="HydroST-lead1",N=int(mk.sum()),NSE=round(nse,3),KGE=round(kge(o,p),3),MAE=round(float(np.mean(np.abs(o-p))),2),CRPS=round(crps(o,qp),3),CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3))]).to_csv(OUT/"131_hydrost_lead1_metrics.csv",index=False)
    log.info("Saved: 131_hydrost_lead1_metrics.csv + preds + 3 checkpoints")

if __name__=="__main__":
    main()
