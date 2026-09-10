#!/usr/bin/env python3
"""
Script 133 — Generación final de figuras para presentación (títulos formales, neutrales).

Regenera fig27/29/30/32/33/34/35 desde artefactos cacheados (sin re-entrenar). Títulos y
etiquetas en lenguaje científico formal, sin anotaciones editoriales. Run in .venv313.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T); R=T.R; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]
MODEL="RA-TFT"
def crps(o,qp):
    t=0
    for i,q in enumerate(QS): e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)
def nse(o,p):
    m=np.isfinite(o)&np.isfinite(p); o,p=o[m],p[m]; return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

def load():
    mt=np.load(OUT/"125_meta.npz"); te=list(mt["te"]); PR=np.load(OUT/"125_PR_test.npy")
    df=R.build(H); return df, df.index, df["obs"].values, df["q"].values, te, PR
FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]
def lgbm_series(df,h):
    d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
    Pq=[]
    for qq in QS:
        g=lgb.LGBMRegressor(objective="quantile",alpha=qq,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
        g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); Pq.append(pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h)))
    return Pq

def fig27(df,dts,obs,q,te,PR):
    fig,axes=plt.subplots(3,1,figsize=(15,12),sharex=True)
    for ax,h in zip(axes,[1,7,14]):
        j=[i+h-1 for i in te]; tg=np.array([dts[k] for k in j]); o=np.array([obs[k] for k in j]); m=np.isfinite(o)
        p10,p50,p90=PR[:,h-1,0],PR[:,h-1,1],PR[:,h-1,2]
        ax.fill_between(tg,p10,p90,color="#9ecae1",alpha=0.55,label="Intervalo P10–P90")
        ax.plot(tg,p50,"-",color="#08519c",lw=1.5,label="Pronóstico (mediana P50)")
        ax.plot(tg[m],o[m],"o",color="black",ms=3.2,label="Caudal observado")
        ax.axhline(Q90,ls="--",color="#d62728",lw=1,alpha=0.8,label=f"Umbral de alerta Q90 = {Q90:.1f} m³/s")
        ns=nse(o,p50)
        ax.set_title(f"Horizonte de {h} día(s)   ·   NSE = {ns:.3f}",loc="left",fontsize=11)
        ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
        if h==1: ax.legend(fontsize=8,ncol=4,loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"Pronóstico de caudal por horizonte — periodo de prueba 2024 ({MODEL})",fontsize=13,y=0.995)
    fig.tight_layout(); fig.savefig(FIGP/"fig27_tft_test_hydrographs.png",dpi=300); plt.close(fig)

def fig29(df,dts,obs,q,te,PR):
    lg={h:lgbm_series(df,h)[1] for h in (1,7)}
    def series(h):
        j=[i+h-1 for i in te]; tg=np.array([dts[k] for k in j]); o=np.array([obs[k] for k in j])
        return tg,o,PR[:,h-1,1],np.array([q[i-1] for i in te]),lg[h].reindex(pd.DatetimeIndex(tg)).values
    tg1,o1,_,_,_=series(1); m1=np.isfinite(o1); pk=tg1[m1][np.argmax(o1[m1])]; z0,z1=pk-pd.Timedelta(days=40),pk+pd.Timedelta(days=45)
    def draw(ax,h,x0=None,x1=None):
        tg,o,tft,pers,l=series(h); m=np.isfinite(o)
        ax.plot(tg[m],o[m],"o-",color="black",ms=3,lw=0.8,label="Caudal observado"); ax.plot(tg,tft,"-",color="#08519c",lw=1.6,label=MODEL)
        ax.plot(tg,l,"-",color="#e6550d",lw=1.4,label="LightGBM"); ax.plot(tg,pers,"-",color="#31a354",lw=1.1,alpha=0.8,label="Persistencia")
        ax.axhline(Q90,ls="--",color="#d62728",lw=1,alpha=0.8,label=f"Q90 = {Q90:.1f} m³/s")
        if x0: ax.set_xlim(x0,x1)
        ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    fig,axes=plt.subplots(3,1,figsize=(15,12))
    draw(axes[0],7); axes[0].set_title("Serie completa — horizonte de 7 días",loc="left",fontsize=11); axes[0].legend(fontsize=8,ncol=5,loc="upper right")
    draw(axes[1],1,z0,z1); axes[1].set_title(f"Detalle del evento de crecida ({pd.Timestamp(pk).date()}) — horizonte de 1 día",loc="left",fontsize=11)
    draw(axes[2],7,z0,z1); axes[2].set_title(f"Detalle del evento de crecida ({pd.Timestamp(pk).date()}) — horizonte de 7 días",loc="left",fontsize=11)
    for ax in axes: ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle("Comparación de modelos en el periodo de prueba",fontsize=13,y=0.997); fig.tight_layout()
    fig.savefig(FIGP/"fig29_test_overlay_zoom.png",dpi=300); plt.close(fig)

def fig30():
    d=pd.read_csv(OUT/"131_hydrost_lead1_preds.csv",parse_dates=["date"]).sort_values("date")
    ns=1-np.sum((d.obs-d.p50)**2)/np.sum((d.obs-d.obs.mean())**2)
    fig,ax=plt.subplots(figsize=(15,5))
    ax.fill_between(d.date,d.p10,d.p90,color="#fdae6b",alpha=0.5,label="Intervalo P10–P90")
    ax.plot(d.date,d.p50,"-",color="#e6550d",lw=1.5,label="Pronóstico (mediana P50)")
    ax.plot(d.date,d.obs,"o",color="black",ms=3.2,label="Caudal observado")
    ax.axhline(Q90,ls="--",color="#d62728",lw=1,alpha=0.8,label=f"Umbral de alerta Q90 = {Q90:.1f} m³/s")
    ax.set_title(f"HydroST — pronóstico de caudal a 1 día · periodo de prueba 2024   ·   NSE = {ns:.3f}",fontsize=12,loc="left")
    ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.legend(fontsize=8,ncol=4,loc="upper right")
    fig.tight_layout(); fig.savefig(FIGP/"fig30_hydrost_test_hydrograph.png",dpi=300); plt.close(fig)

def fig32():
    t=pd.read_csv(OUT/"126_recursive_vs_direct.csv")
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(t["h"],t["NSE_directo"],"-o",color="#08519c",lw=2,label="Estrategia directa (multi-horizonte)")
    ax.plot(t["h"],t["NSE_recursivo_clim"],"-s",color="#e6550d",lw=2,label="Recursiva (forzante = climatología)")
    ax.plot(t["h"],t["NSE_recursivo_seco"],"-^",color="#31a354",lw=2,label="Recursiva (precipitación futura = 0)")
    ax.axvline(4,ls="--",color="gray",alpha=0.7); ax.text(4.1,ax.get_ylim()[0]+0.02,"tiempo de concentración ≈ 4 d",fontsize=8,color="gray")
    ax.set_xlabel("Horizonte (días)"); ax.set_ylabel("NSE (periodo de prueba 2024)"); ax.grid(alpha=0.3)
    ax.set_title("Estrategias de pronóstico: directa frente a recursiva",fontsize=12); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGP/"fig32_recursive_vs_direct.png",dpi=300); plt.close(fig)

def fig33(df,dts,obs,q,te,PR):
    hs=list(range(1,15)); data={m:{"NSE":[],"CRPS":[]} for m in [MODEL,"LightGBM","Persistencia"]}
    for h in hs:
        j=[i+h-1 for i in te]; td=pd.DatetimeIndex([dts[k] for k in j]); o=np.array([obs[k] for k in j]); mk=np.isfinite(o)
        data[MODEL]["NSE"].append(nse(o,PR[:,h-1,1])); data[MODEL]["CRPS"].append(crps(o[mk],PR[mk,h-1,:]))
        pp=np.array([q[i-1] for i in te]); data["Persistencia"]["NSE"].append(nse(o,pp)); data["Persistencia"]["CRPS"].append(crps(o[mk],np.stack([pp]*3,1)[mk]))
        Pq=lgbm_series(df,h); qp=np.sort(np.stack([s.reindex(td).values for s in Pq],1),1)
        data["LightGBM"]["NSE"].append(nse(o,qp[:,1])); data["LightGBM"]["CRPS"].append(crps(o[mk],qp[mk]))
    col={MODEL:"#08519c","LightGBM":"#e6550d","Persistencia":"#31a354"}
    fig,ax=plt.subplots(1,2,figsize=(15,5.5))
    for m in data:
        ax[0].plot(hs,data[m]["NSE"],"-o",color=col[m],lw=2,ms=4,label=m); ax[1].plot(hs,data[m]["CRPS"],"-o",color=col[m],lw=2,ms=4,label=m)
    ax[0].set_ylabel("NSE"); ax[1].set_ylabel("CRPS [m³/s]")
    for a in ax: a.set_xlabel("Horizonte (días)"); a.grid(alpha=0.3)
    ax[0].set_title("(a) Ajuste puntual (NSE)"); ax[1].set_title("(b) Habilidad probabilística (CRPS)"); ax[0].legend(fontsize=9)
    fig.suptitle("Habilidad predictiva en función del horizonte — periodo de prueba 2024",fontsize=13,fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGP/"fig33_skill_vs_horizon.png",dpi=300); plt.close(fig)

def fig34():
    tab=pd.read_csv(OUT/"125_full_metrics_corrected.csv")
    mets=["NSE","NSE_sqrt","KGE","CRPS","CSI","POD","FAR"]; models=[MODEL if m=="TFT-honesto" else m for m in ["TFT-honesto","LightGBM","Persistencia"]]
    tab["model"]=tab["model"].replace({"TFT-honesto":MODEL}); HS=[1,3,7,14]
    fig,axes=plt.subplots(1,4,figsize=(17,3.6))
    for ax,h in zip(axes,HS):
        sub=tab[tab.h==h].set_index("model").reindex(models)[mets]
        ax.imshow(np.zeros_like(sub.values),cmap="Greys",vmin=0,vmax=1,aspect="auto")
        for yi,mo in enumerate(models):
            for xi,me in enumerate(mets):
                v=sub.loc[mo,me]; ax.text(xi,yi,("" if pd.isna(v) else f"{v:.2f}"),ha="center",va="center",fontsize=8)
        ax.set_xticks(range(len(mets))); ax.set_xticklabels(mets,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models if h==1 else [""]*len(models),fontsize=8)
        ax.set_title(f"h = {h} d",fontsize=10)
    fig.suptitle("Métricas de desempeño por horizonte — periodo de prueba 2024 (observaciones reales)",fontsize=12,fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGP/"fig34_metrics_table.png",dpi=300); plt.close(fig)

def fig35():
    fig,ax=plt.subplots(figsize=(15,7)); ax.set_xlim(0,15); ax.set_ylim(0,10); ax.axis("off")
    def box(x,y,w,h,txt,fc,fs=9):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",fc=fc,ec="#333",lw=1.2)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs)
    def arrow(x0,y0,x1,y1,c="#333"): ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=18,lw=2,color=c))
    ax.text(2.5,9.5,"Entradas (información disponible hasta el día t)",fontsize=11,fontweight="bold",ha="center")
    box(0.3,5.2,4.6,3.6,"Ventana pasada (90 días)\nprecipitación · PET · API · SPI-90\nhumedad de suelo · ONI · Niño costero\ncalendario (sin/cos)\n+ caudal observado (RevIN)","#deebf7",8)
    box(0.3,3.2,4.6,1.4,"Calendario futuro (sin/cos, t+1…t+14)\nÚnica covariable futura conocida","#fff2cc",8)
    ax.text(2.6,2.75,"La meteorología futura no se emplea como entrada",fontsize=8,color="#333",ha="center",style="italic")
    box(6.2,4.2,3.2,3.4,f"{MODEL}\n\nRevIN\n+ Codificador–Decodificador\n+ Atención cruzada\n+ Salida cuantílica","#c6dbef",10)
    arrow(4.9,7.0,6.2,6.3); arrow(4.9,3.9,6.2,5.2)
    box(10.8,4.2,4.0,3.4,"Salida: caudal del outlet\n\nHorizontes h = 1 … 14 días\nCuantiles P10 / P50 / P90\n(banda de incertidumbre)","#c7e9c0",9)
    arrow(9.4,5.9,10.8,5.9)
    ax.annotate("",xy=(2.6,5.0),xytext=(12.8,4.0),arrowprops=dict(arrowstyle="-|>",color="#e6550d",lw=1.5,ls="--",connectionstyle="arc3,rad=0.35"))
    ax.text(7.7,2.2,"Realimentación autorregresiva (modo recursivo): el caudal predicho puede reingresar como entrada",fontsize=8,color="#e6550d",ha="center",style="italic")
    ax.text(7.7,0.9,"En inferencia se emplean únicamente datos hasta el día t. La salida es exclusivamente el caudal, en múltiples horizontes, con incertidumbre.",
            fontsize=9,ha="center",bbox=dict(boxstyle="round",fc="#f7f7f7",ec="#999"))
    fig.suptitle("Proceso de inferencia del pronóstico de caudal",fontsize=14,fontweight="bold")
    fig.savefig(FIGP/"fig35_inference_process.png",dpi=300,bbox_inches="tight"); plt.close(fig)

def main():
    df,dts,obs,q,te,PR=load()
    fig27(df,dts,obs,q,te,PR); print("fig27 ok")
    fig29(df,dts,obs,q,te,PR); print("fig29 ok")
    fig30(); print("fig30 ok")
    fig32(); print("fig32 ok")
    fig33(df,dts,obs,q,te,PR); print("fig33 ok")
    fig34(); print("fig34 ok")
    fig35(); print("fig35 ok")
    print("Figuras finales regeneradas con títulos formales.")

if __name__=="__main__":
    main()
