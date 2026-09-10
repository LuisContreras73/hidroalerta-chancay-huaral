#!/usr/bin/env python3
"""
Script 122 — Figura mensual con ESTRATIFICACIÓN ESTACIONAL (lead 1).

Serie del test con temporadas sombreadas (húmeda Dic-Abr azul, seca May-Nov naranja),
obs coloreadas por temporada, P50 + banda P10-P90 + climatología. Panel derecho:
dispersión obs-vs-pred por temporada (diagonal 1:1). Reutiliza 117.

Run in .venv:
    python scripts/05_models/122_monthly_seasonal_fig.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("m-seas")
spec=importlib.util.spec_from_file_location("m117",ROOT/"scripts/05_models/117_monthly_multihorizon.py")
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
TRAIN_END=M.TRAIN_END; VAL_END=M.VAL_END; BASE=M.BASE; WET=M.WET

def main():
    m=M.build_base(); ci=m[m.index<=TRAIN_END].groupby(m[m.index<=TRAIN_END].index.month)["q"].mean()
    def anom(fr):
        a=fr.copy(); a["q_l1"]=a["q_l1"]-a.index.month.map(ci)
        a["q_l2"]=a["q_l2"]-((a.index.month-2)%12+1).map(ci); a["q_l3"]=a["q_l3"]-((a.index.month-3)%12+1).map(ci); return a
    L=1; mm=m.copy(); mm["y"]=mm["q"].shift(-L); mm["y_obs"]=mm["q_obs"].shift(-L)
    mm["y_month"]=((mm.index.month-1+L)%12)+1; mm=mm.dropna(subset=["q_l1","q_l2","q_l3","pr_2m"])
    tr=mm[mm.index<=TRAIN_END]; te=mm[mm.index>VAL_END]; clim=tr.groupby("y_month")["y"].mean()
    tc=te["y_month"].map(clim).values; ya=(tr["y"]-tr["y_month"].map(clim)).values; tra,tea=anom(tr),anom(te)
    Q={}
    for q,nm in [(0.1,"p10"),(0.5,"p50"),(0.9,"p90")]:
        g=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=300,learning_rate=0.03,num_leaves=12,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,verbose=-1)
        g.fit(tra[BASE],ya); Q[nm]=tc+g.predict(tea[BASE])
    Q["p10"]=np.minimum(Q["p10"],Q["p50"]); Q["p90"]=np.maximum(Q["p90"],Q["p50"])
    tgt=te.index+pd.offsets.MonthBegin(L); yo=te["y_obs"].values; wet=np.isin(te["y_month"].values,WET); mk=np.isfinite(yo)

    fig,(ax,ax2)=plt.subplots(1,2,figsize=(16,5.5),gridspec_kw={"width_ratios":[2.4,1]})
    # sombreado temporada húmeda
    for i in range(len(tgt)):
        if wet[i]: ax.axvspan(tgt[i]-pd.offsets.MonthBegin(1),tgt[i],color="#deebf7",alpha=0.7,zorder=0)
    ax.fill_between(tgt,Q["p10"],Q["p90"],color="#c7e9c0",alpha=0.6,label="P10–P90",zorder=1)
    ax.plot(tgt,Q["p50"],"-",color="#238b45",lw=1.6,label="P50 (anomalías)",zorder=3)
    ax.plot(tgt,tc,"--",color="gray",lw=1.1,label="Climatología",zorder=2)
    ax.plot(tgt[mk&wet],yo[mk&wet],"s",color="#08519c",ms=5,label="Obs húmeda",zorder=4)
    ax.plot(tgt[mk&~wet],yo[mk&~wet],"o",color="#e6550d",ms=5,label="Obs seca",zorder=4)
    ax.axvspan(np.nan,np.nan,color="#deebf7",label="Temporada húmeda")
    ax.set_ylabel("Caudal mensual [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.legend(fontsize=8,ncol=3,loc="upper right")
    ax.set_title("Pronóstico mensual (lead 1) — temporada húmeda sombreada",loc="left",fontsize=11)
    # scatter obs-vs-pred por temporada
    def mets(sel):
        o,p=yo[sel],Q["p50"][sel]; n=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2); return n,float(np.mean(np.abs(o-p)))
    nw,mw=mets(mk&wet); nd,md=mets(mk&~wet)
    ax2.scatter(yo[mk&wet],Q["p50"][mk&wet],color="#08519c",s=30,label=f"Húmeda (NSE={nw:.2f}, MAE={mw:.1f})")
    ax2.scatter(yo[mk&~wet],Q["p50"][mk&~wet],color="#e6550d",s=30,label=f"Seca (NSE={nd:.2f}, MAE={md:.1f})")
    lim=max(yo[mk].max(),Q["p50"][mk].max())*1.05; ax2.plot([0,lim],[0,lim],"k--",lw=0.8,alpha=0.6)
    ax2.set_xlabel("Obs [m³/s]"); ax2.set_ylabel("Pronóstico P50 [m³/s]"); ax2.grid(alpha=0.3)
    ax2.set_title("Obs vs pronóstico por temporada",loc="left",fontsize=11); ax2.legend(fontsize=8,loc="upper left")
    fig.tight_layout()
    for p in [OUT/"122_monthly_seasonal.png", FIGP/"fig31_monthly_seasonal.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); log.info(f"Saved: 122_monthly_seasonal.png + fig31 (húmeda NSE={nw:.2f} MAE={mw:.1f}; seca NSE={nd:.2f} MAE={md:.1f})")

if __name__=="__main__":
    main()
