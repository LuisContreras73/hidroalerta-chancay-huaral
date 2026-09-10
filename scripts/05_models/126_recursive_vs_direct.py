#!/usr/bin/env python3
"""
Script 126 — Pronóstico RECURSIVO (iterado) vs DIRECTO (idea del usuario).

En vez de predecir h directamente, encadena el paso a 1 día: realimenta el Q predicho y
avanza. Como los forzantes futuros son desconocidos, se ASUMEN (2 variantes honestas):
  (A) climatología estacional de forzantes  (B) lluvia futura = 0 (supuesto seco).
El calendario futuro (sin/cos) sí es conocido. Reutiliza los 3 modelos guardados por 125.
Alineación CORREGIDA: horizonte h -> índice i+h-1 (obs[i+h-1]).

Hipótesis: recursivo ≈ directo dentro del tiempo de concentración (~4d, lluvia pasada aún en
tránsito); a horizontes mayores empata/pierde (acumulación de error + falta de info futura).

Run in .venv313:
    python scripts/05_models/126_recursive_vs_direct.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
import optuna
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("recur")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; Q90=R.Q90; H=14

def nse(o,p):
    m=np.isfinite(o)&np.isfinite(p); o,p=o[m],p[m]; return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

def main():
    mt=np.load(OUT/"125_meta.npz"); enc=int(mt["enc"]); hid=int(mt["hid"]); heads=int(mt["heads"]); drop=float(mt["drop"])
    mu_p,sd_p,mu_f,sd_f=mt["mu_p"],mt["sd_p"],mt["mu_f"],mt["sd_f"]; te=list(mt["te"])
    PR=np.load(OUT/"125_PR_test.npy")   # directo (ensemble) P50 = PR[:,h-1,1]
    df=R.build(H); dts=df.index; obs=df["obs"].values; q=df["q"].values.astype(np.float32)
    PASTv=df[R.PAST].values.astype(np.float32); SIN=df["sin"].values.astype(np.float32); COS=df["cos"].values.astype(np.float32)
    doy=df.index.dayofyear.values
    # climatología estacional de forzantes PAST (train ≤2022) por día del año
    tr=[i for i in range(enc,len(df)-H) if dts[i]<=pd.Timestamp("2022-12-31")]
    climP=pd.DataFrame(PASTv[tr],columns=R.PAST); climP["doy"]=doy[tr]; climP=climP.groupby("doy").mean()
    clim_arr=np.zeros((367,len(R.PAST)),np.float32)
    for dd in range(1,367): clim_arr[dd]= climP.loc[dd].values if dd in climP.index else climP.mean().values
    iPR={"pr":R.PAST.index("pr")}
    models=[]
    for s in range(3):
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=hid,heads=heads,H=H,drop=drop,use_future=True).to(DEV)
        m.load_state_dict(torch.load(OUT/f"125_tft_seed{s}.pt",map_location=DEV,weights_only=True)); m.eval(); models.append(m)
    N=len(te); L=len(df)
    def ensemble_step(bufc,bufq,fidx):
        xp=((bufc-mu_p)/sd_p).astype(np.float32)
        fi=np.clip(fidx,0,L-1); xf=np.stack([SIN[fi],COS[fi]],-1); xf=((xf-mu_f)/sd_f).astype(np.float32)
        Xp=torch.tensor(np.nan_to_num(xp)).to(DEV); Qp=torch.tensor(np.nan_to_num(bufq)).to(DEV); Xf=torch.tensor(np.nan_to_num(xf)).to(DEV)
        with torch.no_grad():
            ps=[mo(Xp,Qp,Xf).cpu().numpy()[:,0,1] for mo in models]   # posición 0, P50
        return np.mean(ps,0)

    MAXH=7
    def run_recursive(zero_rain):
        bufc=np.stack([PASTv[i-enc:i].copy() for i in te]); bufq=np.stack([q[i-enc:i].copy() for i in te])
        cur=np.array([i-1 for i in te])   # último índice observado
        rec={}
        for step in range(1,MAXH+1):
            fidx=(cur[:,None]+1+np.arange(H)[None,:])   # calendario futuro para la ventana decoder
            pred=ensemble_step(bufc,bufq,fidx)          # pred del índice cur+1
            rec[step]=pred.copy()
            # construir día nuevo (índice cur+1): forzantes ASUMIDOS
            nd=cur+1; nv=clim_arr[np.clip(doy[np.clip(nd,0,L-1)],1,366)].copy()
            nv[:,R.PAST.index("sin")]=SIN[np.clip(nd,0,L-1)]; nv[:,R.PAST.index("cos")]=COS[np.clip(nd,0,L-1)]
            if zero_rain: nv[:,iPR["pr"]]=0.0
            bufc=np.concatenate([bufc[:,1:,:],nv[:,None,:]],1); bufq=np.concatenate([bufq[:,1:],pred[:,None]],1)
            cur=cur+1
        return rec

    rec_clim=run_recursive(False); rec_zero=run_recursive(True)
    HS=[1,3,5,7]; rows=[]
    for h in HS:
        o=np.array([obs[i+h-1] for i in te])
        d=nse(o,PR[:,h-1,1]); rc=nse(o,rec_clim[h]); rz=nse(o,rec_zero[h])
        rows.append(dict(h=h,NSE_directo=round(d,3),NSE_recursivo_clim=round(rc,3),NSE_recursivo_seco=round(rz,3)))
        log.info(f"h={h}: directo={d:.3f} | recursivo(clim)={rc:.3f} | recursivo(seco)={rz:.3f}")
    tab=pd.DataFrame(rows); tab.to_csv(OUT/"126_recursive_vs_direct.csv",index=False)

    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(tab["h"],tab["NSE_directo"],"-o",color="#08519c",lw=2,label="Estrategia directa (multi-horizonte)")
    ax.plot(tab["h"],tab["NSE_recursivo_clim"],"-s",color="#e6550d",lw=2,label="Recursiva (forzante = climatología)")
    ax.plot(tab["h"],tab["NSE_recursivo_seco"],"-^",color="#31a354",lw=2,label="Recursiva (precipitación futura = 0)")
    ax.axvline(4,ls="--",color="gray",alpha=0.7); ax.text(4.1,ax.get_ylim()[0]+0.02,"tiempo de concentración ≈ 4 d",fontsize=8,color="gray")
    ax.set_xlabel("Horizonte (días)"); ax.set_ylabel("NSE (periodo de prueba 2024)"); ax.grid(alpha=0.3)
    ax.set_title("Estrategias de pronóstico: directa frente a recursiva",fontsize=12); ax.legend(fontsize=9)
    fig.tight_layout()
    for p in [OUT/"126_recursive_vs_direct.png",FIGP/"fig32_recursive_vs_direct.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); log.info("\nSaved: 126_recursive_vs_direct.csv + fig32")

if __name__=="__main__":
    main()
