#!/usr/bin/env python3
"""
Script 119 — Figuras mensuales estilo fig10, MULTI-LEAD (1, 2, 3 meses).

Extiende la fig10 (1 mes) a un panel por lead: serie del test 2021+ con obs real,
banda P10-P90, P50 (modelo anomalías) y climatología. Reutiliza build_base del Script 117.

Run in .venv:
    python scripts/05_models/119_monthly_multilead_figs.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("m-fig")
spec=importlib.util.spec_from_file_location("m117",ROOT/"scripts/05_models/117_monthly_multihorizon.py")
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
TRAIN_END=M.TRAIN_END; VAL_END=M.VAL_END; BASE=M.BASE; WET=M.WET

def main():
    m=M.build_base()
    clim_issue=m[m.index<=TRAIN_END].groupby(m[m.index<=TRAIN_END].index.month)["q"].mean()
    def anomalize(fr):
        a=fr.copy()
        a["q_l1"]=a["q_l1"]-a.index.month.map(clim_issue)
        a["q_l2"]=a["q_l2"]-((a.index.month-2)%12+1).map(clim_issue)
        a["q_l3"]=a["q_l3"]-((a.index.month-3)%12+1).map(clim_issue)
        return a
    LEADS=[1,2,3]; fig,axes=plt.subplots(len(LEADS),1,figsize=(15,3.8*len(LEADS)),sharex=True)
    for ax,L in zip(axes,LEADS):
        mm=m.copy(); mm["y"]=mm["q"].shift(-L); mm["y_obs"]=mm["q_obs"].shift(-L)
        mm["y_month"]=((mm.index.month-1+L)%12)+1; mm=mm.dropna(subset=["q_l1","q_l2","q_l3","pr_2m"])
        tr=mm[mm.index<=TRAIN_END]; te=mm[mm.index>VAL_END]
        clim=tr.groupby("y_month")["y"].mean(); te_clim=te["y_month"].map(clim).values
        y_anom=(tr["y"]-tr["y_month"].map(clim)).values; tr_a,te_a=anomalize(tr),anomalize(te)
        Q={}
        for q,nm in [(0.1,"p10"),(0.5,"p50"),(0.9,"p90")]:
            g=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=300,learning_rate=0.03,num_leaves=12,
                                min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,verbose=-1)
            g.fit(tr_a[BASE],y_anom); Q[nm]=te_clim+g.predict(te_a[BASE])
        Q["p10"]=np.minimum(Q["p10"],Q["p50"]); Q["p90"]=np.maximum(Q["p90"],Q["p50"])
        tgt=te.index+pd.offsets.MonthBegin(L); yo=te["y_obs"].values; mk=np.isfinite(yo)
        ax.fill_between(tgt,Q["p10"],Q["p90"],color="#c7e9c0",alpha=0.6,label="P10–P90",zorder=1)
        ax.plot(tgt,Q["p50"],"-",color="#238b45",lw=1.6,label="Pronóstico P50 (anomalías)",zorder=3)
        ax.plot(tgt,te_clim,"--",color="gray",lw=1.1,label="Climatología",zorder=2)
        ax.plot(tgt[mk],yo[mk],"s",color="black",ms=4.5,label="Obs real",zorder=4)
        nse=1-np.sum((yo[mk]-Q["p50"][mk])**2)/np.sum((yo[mk]-yo[mk].mean())**2)
        kge=M.kge(yo[mk],Q["p50"][mk])
        ax.set_title(f"Lead {L} mes(es) — NSE={nse:.3f}  KGE={kge:.3f}  (test, obs reales N={mk.sum()})",fontsize=11,loc="left")
        ax.set_ylabel("Caudal mensual [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
        if L==LEADS[0]: ax.legend(fontsize=8,ncol=4,loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Pronóstico mensual de disponibilidad hídrica — test por horizonte (lead)",fontsize=13,y=0.997)
    fig.tight_layout()
    for p in [OUT/"119_monthly_multilead.png", FIGP/"fig28_monthly_multilead.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); log.info("Saved: 119_monthly_multilead.png + fig28")

if __name__=="__main__":
    main()
