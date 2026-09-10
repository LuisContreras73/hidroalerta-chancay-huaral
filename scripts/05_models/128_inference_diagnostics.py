#!/usr/bin/env python3
"""
Script 128 — Diagnóstico de inferencia: (1) ¿solo usa el caudal Q? (ablación de inputs),
(2) ¿por qué los horizontes de fig27 se parecen? (similitud entre horizontes, ancho de banda,
predicción en el pico de la crecida). Reusa modelos guardados por 125. Run in .venv313.
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
logging.getLogger().setLevel(logging.INFO)
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"
log=logging.getLogger("diag"); logging.basicConfig(level=logging.INFO, format="%(message)s")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; H=14
def nse(o,p):
    m=np.isfinite(o)&np.isfinite(p); o,p=o[m],p[m]; return 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)

def main():
    mt=np.load(OUT/"125_meta.npz"); enc=int(mt["enc"]); hid=int(mt["hid"]); heads=int(mt["heads"]); drop=float(mt["drop"]); te=list(mt["te"])
    mu={"p":mt["mu_p"],"f":mt["mu_f"]}; sd={"p":mt["sd_p"],"f":mt["sd_f"]}
    df=R.build(H); dts=df.index; obs=df["obs"].values
    Xte=T.seqs_for_enc(df,te,enc,H,(mu,sd))
    models=[]
    for s in range(3):
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=hid,heads=heads,H=H,drop=drop,use_future=True).to(DEV)
        m.load_state_dict(torch.load(OUT/f"125_tft_seed{s}.pt",map_location=DEV,weights_only=True)); m.eval(); models.append(m)
    def predict(Xp,Qp,Xf):
        with torch.no_grad(): return np.mean([mo(Xp.to(DEV),Qp.to(DEV),Xf.to(DEV)).cpu().numpy() for mo in models],0)
    Xp,Qp,Xf,_=Xte
    PR=predict(Xp,Qp,Xf)
    # PAST channels: [pr,pet,api,spi90,swvl4,oni,coastal,sin,cos]; meteo=0..6, cal=7,8
    Xp_nometeo=Xp.clone(); Xp_nometeo[:,:,0:7]=0.0          # quita forzantes meteo pasados (deja Q + calendario)
    PR_nm=predict(Xp_nometeo,Qp,Xf)
    Qp_flat=Qp.clone()
    for k in range(Qp_flat.shape[0]): Qp_flat[k,:]=Qp_flat[k,:].mean()   # Q pasado plano (sin dinámica) -> ablación Q
    PR_noq=predict(Xp,Qp_flat,Xf)

    log.info("\n===== (1) ABLACIÓN DE INPUTS (NSE, test corregido i+h-1) =====")
    log.info(f"{'h':>3} | {'FULL':>7} | {'sin meteo (Q+cal)':>18} | {'Q plano (sin Q)':>16}")
    for h in [1,3,7,14]:
        o=np.array([obs[i+h-1] for i in te])
        log.info(f"{h:>3} | {nse(o,PR[:,h-1,1]):>7.3f} | {nse(o,PR_nm[:,h-1,1]):>18.3f} | {nse(o,PR_noq[:,h-1,1]):>16.3f}")

    log.info("\n===== (2) ¿POR QUÉ fig27 se parece entre horizontes? =====")
    log.info("Similitud (corr de Pearson) entre la predicción P50 de h=1 y la de cada h:")
    p1=PR[:,0,1]
    for h in [1,3,7,14]:
        r=np.corrcoef(p1,PR[:,h-1,1])[0,1]; bw=np.mean(PR[:,h-1,2]-PR[:,h-1,0])
        log.info(f"  h={h:>2}: corr(P50_h1,P50_h{h})={r:.3f} | ancho banda P90-P10 medio={bw:5.2f} m3/s")
    # pico de la crecida
    j1=[i+0 for i in te]; o1=np.array([obs[k] for k in j1]); m=np.isfinite(o1); pk_pos=np.argmax(np.where(m,o1,-1))
    pk_i=te[pk_pos]; log.info(f"\nPico de la crecida ~ {dts[pk_i].date()} (obs={o1[pk_pos]:.1f} m3/s). Predicción por horizonte para ese MISMO día objetivo:")
    # para target date D=dts[pk_i], la pred de horizonte h viene del issue i' con i'+h-1 = pk_i -> i'=pk_i-h+1
    for h in [1,3,7,14]:
        ip=pk_i-h+1
        if ip in te:
            pos=te.index(ip); log.info(f"  h={h:>2}: P50={PR[pos,h-1,1]:6.1f}  [P10={PR[pos,h-1,0]:5.1f}, P90={PR[pos,h-1,2]:5.1f}]  (obs={obs[pk_i]:.1f})")
    log.info("\nSaved: (diagnóstico en consola)")

if __name__=="__main__":
    main()
