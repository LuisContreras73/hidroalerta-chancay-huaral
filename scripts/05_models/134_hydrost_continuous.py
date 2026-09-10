#!/usr/bin/env python3
"""
Script 134 — HydroST lead-1: pronóstico CONTINUO en todo el periodo de prueba (2024-2025).

fig30 previa se cortaba en 2025 porque el CSV se guardó filtrado a días con obs real. Aquí se
infiere sobre TODO el periodo (la inferencia no requiere obs; el modelo pronostica donde no hay
medición — su valor operacional) y se grafica P50 + banda continua, con obs solo donde existe.
Reutiliza los checkpoints del Script 131. Run in .venv313.
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import optuna
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hcont")
spec=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
H=importlib.util.module_from_spec(spec); spec.loader.exec_module(H); DEV=H.DEVICE; QC=H.Q_CONV; Q90=40.89

def main():
    bp=optuna.load_study(study_name="hydrost_arch",storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    df=H.load_data(); dyn,stat,tgt0,dates,scl=H.build_arrays(df); H.ENC=bp["enc"]
    tgt1=np.concatenate([[np.nan],tgt0[:-1]]).astype(np.float32)   # lead-1 target = q_mm[idx]
    dpd=pd.DatetimeIndex(dates); ENC=bp["enc"]
    # Inferencia sobre TODO el test: requiere ventana de entrada válida; NO requiere obs/target
    test_idx=[i for i in range(ENC,len(dates)) if dpd[i]>=H.TEST_START and np.isfinite(dyn[i-ENC:i]).all()]
    pdt=[dpd[i] for i in test_idx]
    log.info(f"Inferencia continua: {len(test_idx)} días, {pdt[0].date()} .. {pdt[-1].date()}")
    # obs reales por fecha (para marcadores)
    ob=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]); real={d:v for d,v in zip(ob["date"],ob["q_santo_domingo_47e214d2"]) if np.isfinite(v)}
    yreal=np.array([real.get(d,np.nan) for d in pdt])
    # construir tensores de entrada directamente (sin filtro de target)
    xs=torch.tensor(np.stack([stat]*len(test_idx)),dtype=torch.float32)
    xd=torch.tensor(np.stack([dyn[i-ENC:i].transpose(1,0,2) for i in test_idx]),dtype=torch.float32)
    prs=[]
    for s in range(3):
        m=H.HydroST(hid=bp["hid"],heads=bp["heads"],layers=bp["layers"],dropout=bp["drop"]).to(DEV)
        m.load_state_dict(torch.load(OUT/f"131_hydrost_lead1_seed{s}.pt",map_location=DEV,weights_only=True)); m.eval()
        with torch.no_grad():
            out=[]
            for k in range(0,len(test_idx),256):
                out.append(m(xd[k:k+256].to(DEV),xs[k:k+256].to(DEV)).cpu().numpy())
            prs.append(np.concatenate(out,0))
    PH=np.mean(prs,0)/QC
    x=np.array(pdt); p10,p50,p90=PH[:,0],PH[:,1],PH[:,2]; mk=np.isfinite(yreal)
    ns=1-np.sum((yreal[mk]-p50[mk])**2)/np.sum((yreal[mk]-yreal[mk].mean())**2)
    pd.DataFrame({"date":x,"p10":p10,"p50":p50,"p90":p90,"obs":yreal}).to_csv(OUT/"134_hydrost_continuous.csv",index=False)

    fig,ax=plt.subplots(figsize=(15,5))
    # sombrear tramos sin observación
    gap=~mk; ing=False
    for i,d in enumerate(x):
        if gap[i] and not ing: g0=d; ing=True
        if (not gap[i] or i==len(x)-1) and ing: ax.axvspan(g0,x[i],color="#f2f2f2",alpha=0.9,zorder=0); ing=False
    ax.axvspan(np.nan,np.nan,color="#f2f2f2",label="Periodo sin aforo (sin observación)")
    ax.fill_between(x,p10,p90,color="#fdae6b",alpha=0.5,label="Intervalo P10–P90",zorder=1)
    ax.plot(x,p50,"-",color="#e6550d",lw=1.4,label="Pronóstico (mediana P50)",zorder=3)
    ax.plot(x[mk],yreal[mk],"o",color="black",ms=3.2,label="Caudal observado",zorder=4)
    ax.axhline(Q90,ls="--",color="#d62728",lw=1,alpha=0.8,label=f"Umbral de alerta Q90 = {Q90:.1f} m³/s",zorder=2)
    ax.set_title(f"HydroST — pronóstico continuo de caudal a 1 día · periodo de prueba 2024–2025 (NSE = {ns:.3f} sobre {int(mk.sum())} días aforados)",fontsize=11.5,loc="left")
    ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.legend(fontsize=8,ncol=3,loc="upper right")
    fig.tight_layout()
    for p in [OUT/"134_hydrost_continuous.png",FIGP/"fig30_hydrost_test_hydrograph.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); log.info(f"Saved fig30 continua: {int(mk.sum())} días aforados de {len(x)} totales")

if __name__=="__main__":
    main()
