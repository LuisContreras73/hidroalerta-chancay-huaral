#!/usr/bin/env python3
"""
Script 130 — Comparación MAESTRA justa: todos los modelos al MISMO lead y MISMAS fechas.

Lead L = días entre el último dato observado y la fecha objetivo. Todos predicen la fecha D
usando datos hasta D-L. Métricas en fechas objetivo comunes con obs REAL.
Modelos: Persistencia, LightGBM, TFT honesto (L=1..14), HydroST tuneado (solo L=2 por diseño).
Run in .venv313.
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb
from torch.utils.data import DataLoader
import optuna
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(message)s"); log=logging.getLogger("master")
def imp(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
T=imp("t114",ROOT/"scripts/05_models/114_tft_intensive.py"); R=T.R; DEV=R.DEV; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]
Hm=imp("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
LEADS=[1,2,3,5,7,14]

def crps(o,qp):
    t=0
    for i,q in enumerate(QS): e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
def metrics(o,qp):
    o=np.asarray(o,float); p=qp[:,1]; nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    return dict(NSE=round(nse,3),KGE=round(kge(o,p),3),MAE=round(np.mean(np.abs(o-p)),2),
                CRPS=round(crps(o,qp),3),CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3))

def qframe(dates,q10,q50,q90):
    return pd.DataFrame({"p10":q10,"p50":q50,"p90":q90},index=pd.DatetimeIndex(dates))

def main():
    df=R.build(H); dts=df.index; obs=pd.Series(df["obs"].values,index=dts)
    mt=np.load(OUT/"125_meta.npz"); te=list(mt["te"]); PR=np.load(OUT/"125_PR_test.npy"); q=df["q"].values
    # ---- TFT (por lead) y Persistencia (por lead) ----
    TFT={}; PERS={}
    for L in LEADS:
        D=[dts[i+L-1] for i in te]
        TFT[L]=qframe(D,PR[:,L-1,0],PR[:,L-1,1],PR[:,L-1,2])
        pp=np.array([q[i-1] for i in te]); PERS[L]=qframe(D,pp,pp,pp)
    # ---- LightGBM (por lead) ----
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]; LGB={}
    for L in LEADS:
        d=df.copy(); d["yt"]=d["q"].shift(-L); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
        Pq=[]
        for qq in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=qq,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); Pq.append(np.expm1(g.predict(ted[FE])))
        idx=ted.index+pd.Timedelta(days=L); Pq=np.sort(np.stack(Pq,1),1); LGB[L]=qframe(idx,Pq[:,0],Pq[:,1],Pq[:,2])
    # ---- HydroST (solo L=2) ----
    bph=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    d7=Hm.load_data(); dyn,stat,tgt,hd,_=Hm.build_arrays(d7); Hm.ENC=bph["enc"]; hpd=pd.DatetimeIndex(hd)
    dsh=Hm.HydroDataset(dyn,stat,tgt,hd,hpd[hpd>=Hm.TEST_START]); ldh=DataLoader(dsh,batch_size=256,shuffle=False)
    tdate=[pd.Timestamp(hd[i])+pd.Timedelta(days=1) for i,_ in dsh.indices]; prs=[]
    for s in range(3):
        mm=Hm.HydroST(hid=bph["hid"],heads=bph["heads"],layers=bph["layers"],dropout=bph["drop"]).to(DEV)
        mm.load_state_dict(torch.load(OUT/f"115_hydrost_best_seed{s}.pt",map_location=DEV,weights_only=True)); mm.eval()
        pp=[]
        with torch.no_grad():
            for xd,xs,y in ldh: pp.append(mm(xd.to(DEV),xs.to(DEV)).cpu().numpy())
        prs.append(np.concatenate(pp,0))
    PH=np.mean(prs,0)/Hm.Q_CONV; HYD={2:qframe(tdate,PH[:,0],PH[:,1],PH[:,2])}
    # ---- tabla por lead en fechas comunes con obs real ----
    rows=[]
    for L in LEADS:
        mods={"Persistencia":PERS[L],"LightGBM":LGB[L],"TFT-honesto":TFT[L]}
        if L in HYD: mods["HydroST"]=HYD[L]
        common=None
        for fr in mods.values(): common=fr.index if common is None else common.intersection(fr.index)
        common=common.intersection(obs.dropna().index); common=common.sort_values()
        o=obs.reindex(common).values
        for name,fr in mods.items():
            qp=fr.reindex(common)[["p10","p50","p90"]].values
            rows.append(dict(lead=L,N=len(common),model=name,**metrics(o,qp)))
    tab=pd.DataFrame(rows); tab.to_csv(OUT/"130_master_comparison.csv",index=False)
    for L in LEADS:
        log.info(f"\n===== LEAD {L} día(s)  (N={tab[tab.lead==L]['N'].iloc[0]} fechas comunes con obs real) =====")
        log.info(tab[tab.lead==L].drop(columns=["lead","N"]).to_string(index=False))
    log.info("\nSaved: 130_master_comparison.csv")

if __name__=="__main__":
    main()
