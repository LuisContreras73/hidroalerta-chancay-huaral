#!/usr/bin/env python3
"""
Script 118 — Hidrogramas del TEST (estilo fig10) para el TFT HONESTO (sin leakage).

Visualiza el periodo de test 2024+ del modelo estrella: obs REAL vs P50 + banda P10-P90,
a horizontes h=1, 7, 14 (multi-panel). Umbral de alerta Q90=40.89. Reutiliza el mejor config
del Script 114 (estudio 114_intensive_h14.db). 3 seeds promediados. Sin meteorología futura.

Run in .venv313:
    python scripts/05_models/118_tft_test_hydrographs.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hydro")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; Q90=R.Q90; H=14

def main():
    st=optuna.load_study(study_name="intv_h14",storage=f"sqlite:///{OUT/'114_intensive_h14.db'}"); bp=st.best_params
    log.info(f"Best H=14 config: {bp}")
    df=R.build(H); dts=df.index; obs=df["obs"].values; EM=90
    tr=[i for i in range(EM,len(df)-H) if dts[i]<=pd.Timestamp("2022-12-31") and np.isfinite(df["q"].values[i-EM:i+H]).all()]
    va=[i for i in range(EM,len(df)-H) if pd.Timestamp("2023-01-01")<=dts[i]<=pd.Timestamp("2023-12-31") and np.isfinite(df["q"].values[i-EM:i+H]).all()]
    te=[i for i in range(EM,len(df)-H) if dts[i]>=pd.Timestamp("2024-01-01")]
    mu={"p":df[R.PAST].iloc[tr].values.mean(0),"f":df[R.FUT].iloc[tr].values.mean(0)}
    sd={"p":df[R.PAST].iloc[tr].values.std(0)+1e-6,"f":df[R.FUT].iloc[tr].values.std(0)+1e-6}
    Xtr=T.seqs_for_enc(df,tr,bp["enc"],H,(mu,sd)); Xva=T.seqs_for_enc(df,va,bp["enc"],H,(mu,sd)); Xte=T.seqs_for_enc(df,te,bp["enc"],H,(mu,sd))
    preds=[]
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        dl=DataLoader(TensorDataset(*Xtr),batch_size=bp["bs"],shuffle=True)
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=bp["hid"],heads=bp["heads"],H=H,drop=bp["drop"],use_future=True).to(DEV)
        m=T.train_es(m,dl,Xva,bp["lr"],bp["wd"])
        with torch.no_grad(): preds.append(m(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy())
        log.info(f"  seed {s} done")
    PR=np.mean(preds,0)  # (n_te, H, 3)

    HS=[1,7,14]; fig,axes=plt.subplots(len(HS),1,figsize=(15,4.0*len(HS)),sharex=True)
    for ax,h in zip(axes,HS):
        tgt=np.array([dts[i+h] for i in te]); o=np.array([obs[i+h] for i in te])
        p10,p50,p90=PR[:,h-1,0],PR[:,h-1,1],PR[:,h-1,2]
        ax.fill_between(tgt,p10,p90,color="#9ecae1",alpha=0.55,label="P10–P90 (TFT)",zorder=1)
        ax.plot(tgt,p50,"-",color="#08519c",lw=1.5,label="Pronóstico P50",zorder=3)
        mk=np.isfinite(o); ax.plot(tgt[mk],o[mk],"o",color="black",ms=3.2,label="Obs real",zorder=4)
        ax.axhline(Q90,ls="--",color="#d62728",lw=1.0,alpha=0.8,label=f"Umbral alerta Q90={Q90:.1f}")
        nse=1-np.sum((o[mk]-p50[mk])**2)/np.sum((o[mk]-o[mk].mean())**2)
        ax.set_title(f"Horizonte h={h} día(s) — NSE={nse:.3f}  (test 2024+, obs reales)",fontsize=11,loc="left")
        ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
        if h==HS[0]: ax.legend(fontsize=8,ncol=4,loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("TFT honesto (sin meteorología futura) — hidrograma del test por horizonte",fontsize=13,y=0.995)
    fig.tight_layout();
    for p in [OUT/"118_tft_test_hydrographs.png", FIGP/"fig27_tft_test_hydrographs.png"]:
        fig.savefig(p,dpi=300)
    plt.close(fig); log.info(f"Saved: 118_tft_test_hydrographs.png + fig27")

if __name__=="__main__":
    main()
