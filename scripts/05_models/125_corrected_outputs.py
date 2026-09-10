#!/usr/bin/env python3
"""
Script 125 — Regenera salidas con la ALINEACIÓN CORREGIDA (target = i+h-1).

Entrena el TFT honesto (mejor config H=14) UNA vez, GUARDA los 3 modelos (para el recursivo),
y produce: tabla completa de métricas corregida + fig27 (hidrogramas TFT) + fig29 (overlay+zoom),
todo con la alineación correcta (pr[:,h-1] ↔ obs[i+h-1]; persistencia = q[i-1]).

Run in .venv313:
    python scripts/05_models/125_corrected_outputs.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("corr")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]; HS=[1,3,7,14]

def crps(o,qp):
    t=0
    for i,q in enumerate(QS): e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)
def kge(o,p):
    if o.std()==0 or p.std()==0: return np.nan
    r=np.corrcoef(o,p)[0,1]; return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
def delay(td,p,o,maxlag=14):
    m=np.isfinite(o)&np.isfinite(p); so=pd.Series(o[m],index=pd.DatetimeIndex(np.asarray(td)[m])); sp=pd.Series(p[m],index=so.index)
    b,br=0,-2
    for s in range(0,maxlag+1):
        d=pd.concat([so,sp.shift(-s)],axis=1).dropna()
        if len(d)>10:
            r=d.iloc[:,0].corr(d.iloc[:,1])
            if r>br: br,b=r,s
    return b
def met(o,qp,td):
    o=np.asarray(o,float); p=qp[:,1]; m=np.isfinite(o)&np.isfinite(p); o2,p2,qp2=o[m],p[m],qp[m]
    nse=1-np.sum((o2-p2)**2)/np.sum((o2-o2.mean())**2)
    so,sp=np.sqrt(np.clip(o2,0,None)),np.sqrt(np.clip(p2,0,None)); nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
    oa,pa=o2>=Q90,p2>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    return dict(N=int(m.sum()),NSE=round(nse,3),NSE_sqrt=round(nsq,3),KGE=round(kge(o2,p2),3),RMSE=round(np.sqrt(np.mean((o2-p2)**2)),2),
                MAE=round(np.mean(np.abs(o2-p2)),2),PBIAS=round(100*np.sum(p2-o2)/np.sum(o2),1),CRPS=round(crps(o2,qp2),3),
                CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3),lag=delay(td,p,o))

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
        mo=T.train_es(mo,dl,Xva,bp["lr"],bp["wd"]); torch.save(mo.state_dict(),OUT/f"125_tft_seed{s}.pt")
        with torch.no_grad(): P.append(mo(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy())
        log.info(f"seed {s} ok (model saved)")
    PR=np.mean(P,0)
    np.save(OUT/"125_PR_test.npy",PR)   # cache para el recursivo/otros
    # guarda config+scaler para reuso
    np.savez(OUT/"125_meta.npz",enc=bp["enc"],hid=bp["hid"],heads=bp["heads"],drop=bp["drop"],
             mu_p=mu["p"],sd_p=sd["p"],mu_f=mu["f"],sd_f=sd["f"],te=np.array(te))

    # ── Tabla completa CORREGIDA (target j=i+h-1) ─────────────────────────────
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]; rows=[]
    for h in HS:
        j=[i+h-1 for i in te]; td=[dts[k] for k in j]; o=np.array([obs[k] for k in j])
        rows.append(dict(model="TFT-honesto",h=h,**met(o,PR[:,h-1,:],td)))
        d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
        Pq=[]
        for qq in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=qq,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); Pq.append(pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h)))
        qp=np.sort(np.stack([sq.reindex(pd.DatetimeIndex(td)).values for sq in Pq],1),1)
        rows.append(dict(model="LightGBM",h=h,**met(o,qp,td)))
        pp=np.array([q[i-1] for i in te]); rows.append(dict(model="Persistencia",h=h,**met(o,np.stack([pp]*3,1),td)))
    tab=pd.DataFrame(rows).sort_values(["h","model"]); tab.to_csv(OUT/"125_full_metrics_corrected.csv",index=False)
    for h in HS: log.info(f"\n== h={h} ==\n"+tab[tab.h==h].drop(columns="h").to_string(index=False))

    # ── fig27 corregida (hidrogramas TFT) ─────────────────────────────────────
    fig,axes=plt.subplots(3,1,figsize=(15,12),sharex=True)
    for ax,h in zip(axes,[1,7,14]):
        j=[i+h-1 for i in te]; tg=np.array([dts[k] for k in j]); o=np.array([obs[k] for k in j])
        p10,p50,p90=PR[:,h-1,0],PR[:,h-1,1],PR[:,h-1,2]; m=np.isfinite(o)
        ax.fill_between(tg,p10,p90,color="#9ecae1",alpha=0.55,label="P10–P90 (TFT)")
        ax.plot(tg,p50,"-",color="#08519c",lw=1.5,label="Pronóstico P50")
        ax.plot(tg[m],o[m],"o",color="black",ms=3.2,label="Obs real"); ax.axhline(Q90,ls="--",color="#d62728",lw=1,alpha=0.8,label=f"Q90={Q90:.1f}")
        ns=1-np.sum((o[m]-p50[m])**2)/np.sum((o[m]-o[m].mean())**2)
        ax.set_title(f"Horizonte de {h} día(s)   ·   NSE = {ns:.3f}",loc="left",fontsize=11); ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
        if h==1: ax.legend(fontsize=8,ncol=4,loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); fig.suptitle("Pronóstico de caudal por horizonte — periodo de prueba 2024 (RA-TFT)",fontsize=13,y=0.995)
    fig.tight_layout()
    for p in [OUT/"125_fig27_corrected.png",FIGP/"fig27_tft_test_hydrographs.png"]: fig.savefig(p,dpi=300)
    plt.close(fig)

    # ── fig29 corregida (overlay + zoom) ──────────────────────────────────────
    FEm={}
    for h in (1,7):
        d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
        g=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
        g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); FEm[h]=pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h))
    def series(h):
        j=[i+h-1 for i in te]; tg=np.array([dts[k] for k in j]); o=np.array([obs[k] for k in j])
        return tg,o,PR[:,h-1,1],np.array([q[i-1] for i in te]),FEm[h].reindex(pd.DatetimeIndex(tg)).values
    tg1,o1,_,_,_=series(1); m1=np.isfinite(o1); pk=tg1[m1][np.argmax(o1[m1])]; z0,z1=pk-pd.Timedelta(days=40),pk+pd.Timedelta(days=45)
    def draw(ax,h,x0=None,x1=None):
        tg,o,tft,pers,lg=series(h); m=np.isfinite(o)
        ax.plot(tg[m],o[m],"o-",color="black",ms=3,lw=0.8,label="Obs real"); ax.plot(tg,tft,"-",color="#08519c",lw=1.6,label="TFT")
        ax.plot(tg,lg,"-",color="#e6550d",lw=1.4,label="LightGBM"); ax.plot(tg,pers,"-",color="#31a354",lw=1.1,alpha=0.8,label="Persistencia")
        ax.axhline(Q90,ls="--",color="#d62728",lw=1,alpha=0.8,label=f"Q90={Q90:.1f}")
        if x0: ax.set_xlim(x0,x1)
        ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    fig,axes=plt.subplots(3,1,figsize=(15,12))
    draw(axes[0],7); axes[0].set_title("Serie completa — horizonte de 7 días",loc="left",fontsize=11); axes[0].legend(fontsize=8,ncol=5,loc="upper right")
    draw(axes[1],1,z0,z1); axes[1].set_title(f"Detalle del evento de crecida ({pd.Timestamp(pk).date()}) — horizonte de 1 día",loc="left",fontsize=11)
    draw(axes[2],7,z0,z1); axes[2].set_title(f"Detalle del evento de crecida ({pd.Timestamp(pk).date()}) — horizonte de 7 días",loc="left",fontsize=11)
    for ax in axes: ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle("Comparación de modelos en el periodo de prueba",fontsize=13,y=0.997); fig.tight_layout()
    for p in [OUT/"125_fig29_corrected.png",FIGP/"fig29_test_overlay_zoom.png"]: fig.savefig(p,dpi=300)
    plt.close(fig)
    log.info("\nSaved: 125_full_metrics_corrected.csv + fig27/fig29 corregidas + modelos 125_tft_seed{0,1,2}.pt + PR cache")

if __name__=="__main__":
    main()
