#!/usr/bin/env python3
"""
Script 127 — Figuras resumen para PPT (reusa PR cacheado del Script 125, sin re-entrenar).

  fig33 = curvas de skill vs horizonte (NSE y CRPS, todos los modelos, h=1..14).
  fig34 = heatmap de métricas (modelos x horizontes).
  fig35 = diagrama del PROCESO DE INFERENCIA (entradas -> modelo -> salida target).
Alineación CORREGIDA (target = i+h-1). Run in .venv313.
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("ppt")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]
def crps(o,qp):
    t=0
    for i,q in enumerate(QS): e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)
def nse(o,p):
    m=np.isfinite(o)&np.isfinite(p); o,p=o[m],p[m]; return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

def skill_curves():
    PR=np.load(OUT/"125_PR_test.npy"); mt=np.load(OUT/"125_meta.npz"); te=list(mt["te"])
    df=R.build(H); dts=df.index; obs=df["obs"].values; q=df["q"].values
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]
    hs=list(range(1,15)); data={m:{"NSE":[],"CRPS":[]} for m in ["RA-TFT","LightGBM","Persistencia"]}
    for h in hs:
        j=[i+h-1 for i in te]; td=pd.DatetimeIndex([dts[k] for k in j]); o=np.array([obs[k] for k in j]); mk=np.isfinite(o)
        data["RA-TFT"]["NSE"].append(nse(o,PR[:,h-1,1])); data["RA-TFT"]["CRPS"].append(crps(o[mk],PR[mk,h-1,:]))
        pp=np.array([q[i-1] for i in te]); data["Persistencia"]["NSE"].append(nse(o,pp)); data["Persistencia"]["CRPS"].append(crps(o[mk],np.stack([pp]*3,1)[mk]))
        d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2023-12-01")].dropna(subset=FE)
        Pq=[]
        for qq in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=qq,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); Pq.append(pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h)).reindex(td).values)
        qp=np.sort(np.stack(Pq,1),1); data["LightGBM"]["NSE"].append(nse(o,qp[:,1])); data["LightGBM"]["CRPS"].append(crps(o[mk],qp[mk]))
    col={"RA-TFT":"#08519c","LightGBM":"#e6550d","Persistencia":"#31a354"}
    fig,ax=plt.subplots(1,2,figsize=(15,5.5))
    for m in data:
        ax[0].plot(hs,data[m]["NSE"],"-o",color=col[m],lw=2,ms=4,label=m); ax[1].plot(hs,data[m]["CRPS"],"-o",color=col[m],lw=2,ms=4,label=m)
    ax[0].set_ylabel("NSE"); ax[1].set_ylabel("CRPS (menor=mejor)")
    for a in ax: a.set_xlabel("Horizonte (días)"); a.grid(alpha=0.3); a.axvspan(0.5,3.5,color="#f0f0f0",zorder=0)
    ax[0].set_title("(a) Ajuste puntual (NSE)"); ax[1].set_title("(b) Habilidad probabilística (CRPS)")
    ax[0].legend(fontsize=9)
    fig.suptitle("Habilidad predictiva en función del horizonte — periodo de prueba 2024",fontsize=13,fontweight="bold")
    fig.tight_layout()
    for p in [OUT/"127_fig33_skill_curves.png",FIGP/"fig33_skill_vs_horizon.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); return data,hs

def heatmap():
    tab=pd.read_csv(OUT/"125_full_metrics_corrected.csv")
    tab["model"]=tab["model"].replace({"TFT-honesto":"RA-TFT"})
    mets=["NSE","NSE_sqrt","KGE","CRPS","CSI","POD","FAR"]; models=["RA-TFT","LightGBM","Persistencia"]; HS=[1,3,7,14]
    fig,axes=plt.subplots(1,4,figsize=(17,3.6))
    for ax,h in zip(axes,HS):
        sub=tab[tab.h==h].set_index("model").reindex(models)[mets]
        # normaliza por métrica (FAR invertido: menor mejor)
        disp=sub.copy()
        im=ax.imshow(np.zeros_like(sub.values),cmap="Greys",vmin=0,vmax=1,aspect="auto")
        for yi,mo in enumerate(models):
            for xi,me in enumerate(mets):
                v=sub.loc[mo,me]
                ax.text(xi,yi,("" if pd.isna(v) else f"{v:.2f}"),ha="center",va="center",fontsize=8)
        ax.set_xticks(range(len(mets))); ax.set_xticklabels(mets,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models if h==1 else [""]*len(models),fontsize=8)
        ax.set_title(f"h={h} d",fontsize=10)
    fig.suptitle("Métricas de desempeño por horizonte — periodo de prueba 2024 (observaciones reales)",fontsize=12,fontweight="bold")
    fig.tight_layout()
    for p in [OUT/"127_fig34_heatmap.png",FIGP/"fig34_metrics_table.png"]: fig.savefig(p,dpi=300)
    plt.close(fig)

def inference_diagram():
    fig,ax=plt.subplots(figsize=(15,7)); ax.set_xlim(0,15); ax.set_ylim(0,10); ax.axis("off")
    def box(x,y,w,h,txt,fc,fs=9,tc="black"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",fc=fc,ec="#333",lw=1.2)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,wrap=True)
    def arrow(x0,y0,x1,y1,c="#333"): ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=18,lw=2,color=c))
    # ENTRADAS
    ax.text(2.4,9.5,"Entradas (información disponible hasta el día t)",fontsize=11,fontweight="bold",ha="center")
    pastvars=["Precipitación","PET","API","SPI-90","Hum. suelo (swvl4)","ONI","Niño costero","Calendario (sin/cos)"]
    box(0.3,5.2,4.4,3.6,"VENTANA PASADA (90 días)\n"+" · ".join(pastvars)+"\n+ CAUDAL Q pasado (RevIN)","#deebf7",8)
    box(0.3,3.2,4.4,1.4,"CALENDARIO FUTURO\n(sin/cos de t+1…t+14)\n— único 'futuro' conocido","#fff2cc",8)
    ax.text(2.5,2.7,"La meteorología futura no se emplea como entrada",fontsize=8,color="#333",ha="center",style="italic")
    # MODELO
    box(6.0,4.2,3.2,3.4,"RA-TFT\n\nRevIN (régimen)\n+ Encoder-Decoder\n+ Cross-attention\n+ salida cuantílica","#c6dbef",10)
    arrow(4.7,7.0,6.0,6.3); arrow(4.7,3.9,6.0,5.2)
    # SALIDA
    box(10.6,4.2,4.1,3.4,"Salida: caudal del outlet\n\nHorizontes h = 1 … 14 días\nCuantiles P10 / P50 / P90\n(banda de incertidumbre)","#c7e9c0",9)
    arrow(9.2,5.9,10.6,5.9)
    # nota autoregresiva
    ax.annotate("",xy=(2.5,5.0),xytext=(12.6,4.0),arrowprops=dict(arrowstyle="-|>",color="#e6550d",lw=1.5,ls="--",connectionstyle="arc3,rad=0.35"))
    ax.text(7.5,2.2,"Realimentación autorregresiva (modo recursivo): el caudal predicho puede reingresar como entrada",fontsize=8,color="#e6550d",ha="center",style="italic")
    ax.text(7.5,0.9,"En inferencia se emplean únicamente datos hasta el día t. La salida es exclusivamente el caudal, en múltiples horizontes, con incertidumbre.",
            fontsize=9,ha="center",bbox=dict(boxstyle="round",fc="#f7f7f7",ec="#999"))
    fig.suptitle("Proceso de inferencia del pronóstico de caudal",fontsize=14,fontweight="bold")
    for p in [OUT/"127_fig35_inference.png",FIGP/"fig35_inference_process.png"]: fig.savefig(p,dpi=300,bbox_inches="tight")
    plt.close(fig)

def main():
    skill_curves(); log.info("fig33 skill curves ok")
    heatmap(); log.info("fig34 heatmap ok")
    inference_diagram(); log.info("fig35 inference diagram ok")
    log.info("Saved: fig33/fig34/fig35 (+ copias en outputs/ml_Q)")

if __name__=="__main__":
    main()
