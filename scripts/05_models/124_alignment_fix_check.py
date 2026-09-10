#!/usr/bin/env python3
"""
Script 124 — Verifica y corrige la alineación pred-obs del TFT multi-horizonte.

make_seqs: ventana = arr[i-ENC:i] (último input = i-1); target = tgt[i:i+H] -> posición j
predice el índice i+j. Por tanto, para horizonte h (h días tras el último input i-1) la
predicción es pr[:,h-1] y la OBS correcta es obs[i+h-1] (NO obs[i+h], que estaba en 114/123).
Persistencia coherente: Q en el día de emisión = q[i-1].

Este script: (1) prueba empíricamente qué alineación es la correcta (posición 0 vs obs[i] u obs[i+1]),
(2) recalcula NSE + lag con la alineación CORREGIDA para TFT/LightGBM/persistencia.

Run in .venv313:
    python scripts/05_models/124_alignment_fix_check.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset
import optuna

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("align")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]; HS=[1,3,7,14]

def nse(o,p):
    m=np.isfinite(o)&np.isfinite(p); o,p=o[m],p[m]; return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
def delay(td,p50,o,maxlag=14):
    m=np.isfinite(o)&np.isfinite(p50); so=pd.Series(o[m],index=pd.DatetimeIndex(np.asarray(td)[m])); sp=pd.Series(p50[m],index=so.index)
    b,br=0,-2
    for s in range(0,maxlag+1):
        d=pd.concat([so,sp.shift(-s)],axis=1).dropna()
        if len(d)>10:
            r=d.iloc[:,0].corr(d.iloc[:,1])
            if r>br: br,b=r,s
    return b

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
        log.info(f"seed {s} ok")
    PR=np.mean(P,0)

    # ── (1) PRUEBA de alineación: posición 0 (1er paso) contra obs[i] vs obs[i+1] ──
    log.info("\n===== PRUEBA DE ALINEACIÓN (posición 0 del modelo) =====")
    for pos in [0,6,13]:
        p0=PR[:,pos,1]
        o_same=np.array([obs[i+pos] for i in te])      # CORRECTO por make_seqs (tgt[i+pos])
        o_plus1=np.array([obs[i+pos+1] for i in te])    # el usado antes (obs[i+(pos+1)])
        log.info(f"  posición {pos} (=lead {pos+1}): NSE vs obs[i+{pos}]={nse(o_same,p0):.3f}  |  NSE vs obs[i+{pos+1}]={nse(o_plus1,p0):.3f}  <- el mayor es la alineación real")

    # ── (2) Métricas CORREGIDAS: target index j=i+h-1, persistencia q[i-1] ──
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]; rows=[]
    for h in HS:
        j=[i+h-1 for i in te]; td=[dts[k] for k in j]; o=np.array([obs[k] for k in j])
        # TFT
        rows.append(dict(h=h,model="TFT-honesto",NSE=round(nse(o,PR[:,h-1,1]),3),lag=delay(td,PR[:,h-1,1],o)))
        # LightGBM (emite en i-1, predice j=i-1+h)
        d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
        g=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
        g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); lg=pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h)).reindex(pd.DatetimeIndex(td)).values
        rows.append(dict(h=h,model="LightGBM",NSE=round(nse(o,lg),3),lag=delay(td,lg,o)))
        # Persistencia: Q emisión = q[i-1]
        pp=np.array([q[i-1] for i in te]); rows.append(dict(h=h,model="Persistencia",NSE=round(nse(o,pp),3),lag=delay(td,pp,o)))
    tab=pd.DataFrame(rows).sort_values(["h","model"]); tab.to_csv(OUT/"124_aligned_metrics.csv",index=False)
    log.info("\n===== MÉTRICAS CORREGIDAS (NSE + lag) =====")
    for h in HS: log.info(f"\n-- h={h} --\n"+tab[tab.h==h].drop(columns="h").to_string(index=False))
    log.info("\nSaved: 124_aligned_metrics.csv")

if __name__=="__main__":
    main()
