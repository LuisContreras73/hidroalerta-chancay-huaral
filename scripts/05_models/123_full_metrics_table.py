#!/usr/bin/env python3
"""
Script 123 — Tabla COMPLETA de métricas: todos los modelos × todos los horizontes.

Modelos: Persistencia, LightGBM (sin futuro), TFT honesto (mejor config H=14), HydroST
tuneado (solo h=1 por diseño). Horizontes: 1, 3, 7, 14. Test 2024+ SELLADO, solo obs REALES.
Métricas: NSE, NSE_sqrt, KGE, RMSE, MAE, PBIAS%, CRPS, CSI, POD, FAR (alerta @Q90) + lag(días).
'lag' = desfase objetivo (correlación cruzada): cuántos días hay que adelantar la predicción
para que calce mejor con la obs (0=sin desfase; ~h = persistencia pura).

Run in .venv313:
    python scripts/05_models/123_full_metrics_table.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset
import optuna

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("fullm")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]; HS=[1,3,7,14]

def crps(o,qp):
    t=0
    for i,q in enumerate(QS):
        e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)

def kge(o,p):
    if o.std()==0 or p.std()==0: return np.nan
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)

def delay_days(tdates,p50,o,maxlag=14):
    so=pd.Series(o,index=pd.DatetimeIndex(tdates)); sp=pd.Series(p50,index=pd.DatetimeIndex(tdates))
    best_s,best_r=0,-2
    for s in range(0,maxlag+1):
        d=pd.concat([so,sp.shift(-s)],axis=1).dropna()
        if len(d)>10:
            r=d.iloc[:,0].corr(d.iloc[:,1])
            if r>best_r: best_r,best_s=r,s
    return best_s

def met(o,qp,tdates=None):
    o=np.asarray(o,float); p=qp[:,1]; mk=np.isfinite(o)&np.isfinite(p); o,p=o[mk],p[mk]; qpm=qp[mk]
    nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None)); nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
    rmse=np.sqrt(np.mean((o-p)**2)); mae=np.mean(np.abs(o-p)); pbias=100*np.sum(p-o)/np.sum(o)
    oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    lag=delay_days(np.asarray(tdates)[mk],p,o) if tdates is not None else np.nan
    return dict(N=int(mk.sum()),NSE=round(nse,3),NSE_sqrt=round(nsq,3),KGE=round(kge(o,p),3),RMSE=round(rmse,2),
                MAE=round(mae,2),PBIAS=round(pbias,1),CRPS=round(crps(o,qpm),3),CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3),lag=lag)

def main():
    bp=optuna.load_study(study_name="intv_h14",storage=f"sqlite:///{OUT/'114_intensive_h14.db'}").best_params
    df=R.build(H); dts=df.index; obs=df["obs"].values; q=df["q"].values; EM=90
    tr=[i for i in range(EM,len(df)-H) if dts[i]<=pd.Timestamp("2022-12-31") and np.isfinite(q[i-EM:i+H]).all()]
    va=[i for i in range(EM,len(df)-H) if pd.Timestamp("2023-01-01")<=dts[i]<=pd.Timestamp("2023-12-31") and np.isfinite(q[i-EM:i+H]).all()]
    te=[i for i in range(EM,len(df)-H) if dts[i]>=pd.Timestamp("2024-01-01")]
    mu={"p":df[R.PAST].iloc[tr].values.mean(0),"f":df[R.FUT].iloc[tr].values.mean(0)}
    sd={"p":df[R.PAST].iloc[tr].values.std(0)+1e-6,"f":df[R.FUT].iloc[tr].values.std(0)+1e-6}
    Xtr=T.seqs_for_enc(df,tr,bp["enc"],H,(mu,sd)); Xva=T.seqs_for_enc(df,va,bp["enc"],H,(mu,sd)); Xte=T.seqs_for_enc(df,te,bp["enc"],H,(mu,sd))
    P=[]
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        dl=DataLoader(TensorDataset(*Xtr),batch_size=bp["bs"],shuffle=True)
        mo=R.RATFT(len(R.PAST),len(R.FUT),hid=bp["hid"],heads=bp["heads"],H=H,drop=bp["drop"],use_future=True).to(DEV)
        mo=T.train_es(mo,dl,Xva,bp["lr"],bp["wd"])
        with torch.no_grad(): P.append(mo(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy())
        log.info(f"TFT seed {s} ok")
    PR=np.mean(P,0)
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]; rows=[]
    for h in HS:
        td=[dts[i+h] for i in te]; o=np.array([obs[i+h] for i in te])
        # TFT
        rows.append(dict(model="TFT-honesto",h=h,**met(o,PR[:,h-1,:],td)))
        # LightGBM quantiles
        d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2024-01-01")].dropna(subset=FE)
        Pq=[]
        for qq in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=qq,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); Pq.append(pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h)))
        qp=np.sort(np.stack([s.reindex(pd.DatetimeIndex(td)).values for s in Pq],1),1)
        rows.append(dict(model="LightGBM",h=h,**met(o,qp,td)))
        # Persistencia naive
        pp=np.array([q[i] for i in te]); rows.append(dict(model="Persistencia",h=h,**met(o,np.stack([pp]*3,1),td)))
    tab=pd.DataFrame(rows)

    # HydroST tuneado (solo h=1)
    hs=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py"); Hm=importlib.util.module_from_spec(hs); hs.loader.exec_module(Hm)
    bph=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    ob=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]); real={d:v for d,v in zip(ob["date"],ob["q_santo_domingo_47e214d2"]) if np.isfinite(v)}
    d7=Hm.load_data(); dyn,stat,tgt,hd,_=Hm.build_arrays(d7); Hm.ENC=bph["enc"]; hpd=pd.DatetimeIndex(hd)
    dsh=Hm.HydroDataset(dyn,stat,tgt,hd,hpd[hpd>=Hm.TEST_START]); ldh=DataLoader(dsh,batch_size=256,shuffle=False)
    pdt=[pd.Timestamp(hd[i])+pd.Timedelta(days=1) for i,_ in dsh.indices]; yreal=np.array([real.get(x,np.nan) for x in pdt])
    prs=[]
    for s in range(3):
        mm=Hm.HydroST(hid=bph["hid"],heads=bph["heads"],layers=bph["layers"],dropout=bph["drop"]).to(DEV)
        mm.load_state_dict(torch.load(OUT/f"115_hydrost_best_seed{s}.pt",map_location=DEV,weights_only=True)); mm.eval()
        pp=[]
        with torch.no_grad():
            for xd,xs,y in ldh: pp.append(mm(xd.to(DEV),xs.to(DEV)).cpu().numpy())
        prs.append(np.concatenate(pp,0))
    PH=np.mean(prs,0)/Hm.Q_CONV
    tab=pd.concat([tab,pd.DataFrame([dict(model="HydroST-tuned",h=1,**met(yreal,PH,pdt))])],ignore_index=True)

    tab=tab.sort_values(["h","model"]).reset_index(drop=True)
    tab.to_csv(OUT/"123_full_metrics.csv",index=False)
    for h in HS:
        log.info(f"\n===== HORIZONTE h={h} =====\n"+tab[tab.h==h].drop(columns=["h"]).to_string(index=False))
    log.info("\nSaved: 123_full_metrics.csv")

if __name__=="__main__":
    main()
