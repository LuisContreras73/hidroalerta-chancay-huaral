#!/usr/bin/env python3
"""
Script 135 — Curación de datos para el dashboard ampliado (selector de modelos + 4 secciones).

Extrae SOLO resultados agregados (no metodología) hacia hidroalerta-dashboard/data/:
  forecast_multimodelo.csv  (date, model, lead, obs, p10, p50, p90) — RA-TFT/LightGBM/Persistencia
                             a leads 1/3/7/14 + HydroST a lead 1, continuo en 2024-2025.
  mensual.csv               (date, obs, p10, p50, p90, clim) — disponibilidad mensual.
  enso.csv                  (date, costero, oni) — Niño costero vs global.
  eda_acf.csv               (lag, acf) — autocorrelación del caudal.
  eda_ccf.csv               (lag, ccf) — correlación cruzada lluvia->caudal.
Run in .venv313.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
PUB=Path("d:/ANA Concurso/hidroalerta-dashboard/data")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T); R=T.R; H=14; QS=[0.1,0.5,0.9]
FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]

def main():
    df=R.build(H); dts=df.index; obs=df["obs"].values; q=df["q"].values
    mt=np.load(OUT/"125_meta.npz"); te=list(mt["te"]); PR=np.load(OUT/"125_PR_test.npy")
    rows=[]
    # LightGBM por lead (continuo)
    lgbm_pred={}
    for L in [1,3,7,14]:
        d=df.copy(); d["yt"]=d["q"].shift(-L); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
        Pq=[]
        for qq in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=qq,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); Pq.append(pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=L)))
        lgbm_pred[L]=[np.sort(np.stack([s.reindex(pd.DatetimeIndex([dts[i+L-1] for i in te])).values for s in Pq],1),1)]
    for L in [1,3,7,14]:
        tg=[dts[i+L-1] for i in te]; o=np.array([obs[i+L-1] for i in te])
        # RA-TFT
        for k,i in enumerate(te):
            rows.append(dict(date=tg[k],model="RA-TFT",lead=L,obs=o[k],p10=PR[k,L-1,0],p50=PR[k,L-1,1],p90=PR[k,L-1,2]))
        # LightGBM
        qp=lgbm_pred[L][0]
        for k in range(len(te)):
            rows.append(dict(date=tg[k],model="LightGBM",lead=L,obs=o[k],p10=qp[k,0],p50=qp[k,1],p90=qp[k,2]))
        # Persistencia (puntual)
        for k,i in enumerate(te):
            pv=q[i-1]; rows.append(dict(date=tg[k],model="Persistencia",lead=L,obs=o[k],p10=pv,p50=pv,p90=pv))
    # HydroST lead-1 continuo (desde 134)
    hs=pd.read_csv(OUT/"134_hydrost_continuous.csv",parse_dates=["date"])
    for _,r in hs.iterrows():
        rows.append(dict(date=r["date"],model="HydroST",lead=1,obs=r["obs"],p10=r["p10"],p50=r["p50"],p90=r["p90"]))
    fm=pd.DataFrame(rows)
    for c in ["obs","p10","p50","p90"]: fm[c]=fm[c].round(3)
    fm["p10"]=fm["p10"].clip(lower=0)
    fm.to_csv(PUB/"forecast_multimodelo.csv",index=False)
    print("forecast_multimodelo.csv:", fm.shape, "modelos:", sorted(fm.model.unique()))

    # Mensual
    m=pd.read_csv(OUT/"86_monthly_forecast.csv",parse_dates=[0]); c0=m.columns[0]; m=m.rename(columns={c0:"issue"})
    m["date"]=pd.to_datetime(m["issue"])+pd.offsets.MonthBegin(1)
    men=m[["date"]].copy(); men["obs"]=m["y_obs"]; men["p50"]=m["pred_p50"]; men["p10"]=m["pred_p10_cal"]; men["p90"]=m["pred_p90_cal"]; men["clim"]=m["clim"]
    men=men.round(3); men.to_csv(PUB/"mensual.csv",index=False); print("mensual.csv:", men.shape)

    # ENSO
    en=pd.read_csv(ROOT/"data/silver/enso/S3b_enso_coastal_global.csv",parse_dates=["date"])
    en=en[en["date"]>=pd.Timestamp("2010-01-01")][["date","coastal","global"]].rename(columns={"global":"oni"})
    en=en.round(3); en.to_csv(PUB/"enso.csv",index=False); print("enso.csv:", en.shape)

    # EDA: ACF del caudal y CCF lluvia->caudal (numpy, sin dependencias)
    s=pd.Series(q,index=dts).dropna(); x=(s-s.mean()).values; n=len(x); den=np.sum(x*x)
    acf=[(L,round(float(np.sum(x[L:]*x[:n-L])/den),3)) for L in range(0,31)]
    pd.DataFrame(acf,columns=["lag","acf"]).to_csv(PUB/"eda_acf.csv",index=False)
    pr=df["pr"].reindex(dts).values; a=pd.Series(pr,index=dts); b=pd.Series(q,index=dts); j=pd.concat([a,b],axis=1).dropna()
    aa=(j.iloc[:,0]-j.iloc[:,0].mean()).values; bb=(j.iloc[:,1]-j.iloc[:,1].mean()).values; dd=np.sqrt(np.sum(aa*aa)*np.sum(bb*bb))
    ccf=[(L,round(float(np.sum(aa[:len(aa)-L]*bb[L:])/dd),3)) for L in range(0,16)]  # lluvia adelanta al caudal
    pd.DataFrame(ccf,columns=["lag","ccf"]).to_csv(PUB/"eda_ccf.csv",index=False)
    print("eda_acf/eda_ccf.csv escritos. ACF lag1=",acf[1][1]," CCF pico lag=",max(ccf,key=lambda t:t[1]))

if __name__=="__main__":
    main()
