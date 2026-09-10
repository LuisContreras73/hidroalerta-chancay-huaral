#!/usr/bin/env python3
"""
Script 117 — Mensual MULTI-HORIZONTE + estratificación estacional.

Extiende el Script 86 (1 mes de lead, agregado) a:
  - LEADS L = 1, 2, 3 meses (análogo al multi-horizonte diario).
  - Estratificación por TEMPORADA del mes objetivo: húmeda (Dic-Abr) vs seca (May-Nov).
Modelo = LightGBM de ANOMALÍAS (pred = climatología[mes objetivo] + anomalía predicha),
mismos features que 86. Baselines: climatología (mes objetivo) y persistencia (mes actual).
Eval HONESTO: solo meses con obs REAL del mes objetivo (>=20 días válidos).

Run in .venv:
    python scripts/05_models/117_monthly_multihorizon.py
"""
import logging
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("m-mh")
Q_CONV=86.4/3062.62; TRAIN_END="2014-12-31"; VAL_END="2020-12-31"; WET=[12,1,2,3,4]
BASE=["q_l1","q_l2","q_l3","pr_l1","pr_2m","swvl4_l1","swvl1_l1","oni_l1","spi90_l1"]

def nse(o,p): o,p=np.asarray(o,float),np.asarray(p,float); return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
def kge(o,p):
    o,p=np.asarray(o,float),np.asarray(p,float)
    if len(o)<3 or o.std()==0 or p.std()==0: return np.nan
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)

def build_base():
    d7=pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv",parse_dates=["date"])
    outlet=d7[d7["entity_id"]=="sub_634"].set_index("date").sort_index()
    pr_basin=d7.groupby("date")["pr_mm"].mean()
    swvl=pd.read_parquet(OUT/"era5_swvl_per_entity.parquet")
    if "date" not in swvl.columns: swvl=swvl.reset_index()
    swvl_basin=swvl.groupby("date")[["swvl1","swvl4"]].mean()
    qd=outlet["q_mm"]/Q_CONV
    m=pd.DataFrame({"q":qd.resample("MS").mean(),"pr":pr_basin.resample("MS").sum(),
                   "oni":outlet["oni_index"].resample("MS").mean(),"spi90":outlet["spi_90d"].resample("MS").last(),
                   "swvl1":swvl_basin["swvl1"].resample("MS").mean(),"swvl4":swvl_basin["swvl4"].resample("MS").mean()})
    obs=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    m["q_obs"]=obs.resample("MS").mean().where(obs.resample("MS").count()>=20)
    m["q_l1"],m["q_l2"],m["q_l3"]=m["q"].shift(0),m["q"].shift(1),m["q"].shift(2)
    m["pr_l1"]=m["pr"].shift(0); m["pr_2m"]=m["pr"].rolling(2).sum()
    m["swvl4_l1"]=m["swvl4"].shift(0); m["swvl1_l1"]=m["swvl1"].shift(0)
    m["oni_l1"]=m["oni"].shift(0); m["spi90_l1"]=m["spi90"].shift(0)
    return m

def main():
    m=build_base()
    clim_issue=m[m.index<=TRAIN_END].groupby(m[m.index<=TRAIN_END].index.month)["q"].mean()
    def anomalize(fr):
        a=fr.copy()
        a["q_l1"]=a["q_l1"]-a.index.month.map(clim_issue)
        a["q_l2"]=a["q_l2"]-((a.index.month-2)%12+1).map(clim_issue)
        a["q_l3"]=a["q_l3"]-((a.index.month-3)%12+1).map(clim_issue)
        return a
    rows=[]
    for L in [1,2,3]:
        mm=m.copy()
        mm["y"]=mm["q"].shift(-L); mm["y_obs"]=mm["q_obs"].shift(-L)
        mm["y_month"]=((mm.index.month-1+L)%12)+1
        mm=mm.dropna(subset=["q_l1","q_l2","q_l3","pr_2m"])
        tr=mm[mm.index<=TRAIN_END]; te=mm[mm.index>VAL_END]
        clim=tr.groupby("y_month")["y"].mean()
        te_clim=te["y_month"].map(clim).values; te_pers=te["q_l1"].values
        y_anom=(tr["y"]-tr["y_month"].map(clim)).values
        tr_a,te_a=anomalize(tr),anomalize(te)
        p50=np.full(len(te),np.nan)
        g=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=300,learning_rate=0.03,num_leaves=12,
                            min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,verbose=-1)
        g.fit(tr_a[BASE],y_anom); p50=te_clim+g.predict(te_a[BASE])
        mask=te["y_obs"].notna().values; obs=te["y_obs"].values; tmon=te["y_month"].values
        wet=np.isin(tmon,WET)
        for season,sel in [("all",mask),("wet",mask&wet),("dry",mask&(~wet))]:
            if sel.sum()<2: continue
            o=obs[sel]
            for name,pred in [("LGBM-anom",p50),("Climatology",te_clim),("Persistence",te_pers)]:
                p=np.asarray(pred)[sel]
                rows.append(dict(lead_m=L,season=season,N=int(sel.sum()),model=name,
                                 NSE=round(nse(o,p),3),KGE=round(kge(o,p),3),MAE=round(float(np.mean(np.abs(o-p))),2)))
    res=pd.DataFrame(rows); res.to_csv(OUT/"117_monthly_multihorizon.csv",index=False)
    log.info("\n"+res.to_string(index=False))
    log.info("\nSaved: 117_monthly_multihorizon.csv")
    # pivote NSE legible
    for season in ["all","wet","dry"]:
        sub=res[res.season==season].pivot(index="model",columns="lead_m",values="NSE")
        log.info(f"\n=== NSE por lead — temporada '{season}' ===\n"+sub.to_string())

if __name__=="__main__":
    main()
