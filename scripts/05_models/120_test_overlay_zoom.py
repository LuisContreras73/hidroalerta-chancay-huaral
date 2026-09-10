#!/usr/bin/env python3
"""
Script 120 — Overlay comparativo + zoom a crecida (test diario).

Compara en el MISMO panel el TFT honesto vs LightGBM vs persistencia sobre el test 2024:
  Panel A: horizonte h=7 completo.
  Panel B: zoom a la crecida principal de 2024, h=1.
  Panel C: zoom a la crecida principal de 2024, h=7.
Sin meteorología futura (honesto). Reutiliza 114 (mejor config H=14) + LightGBM(sin futuro).

Run in .venv313:
    python scripts/05_models/120_test_overlay_zoom.py
"""
import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIGP=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("ovl")
spec=importlib.util.spec_from_file_location("t114",ROOT/"scripts/05_models/114_tft_intensive.py")
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
R=T.R; DEV=R.DEV; Q90=R.Q90; H=14; QS=[0.1,0.5,0.9]

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
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=bp["hid"],heads=bp["heads"],H=H,drop=bp["drop"],use_future=True).to(DEV)
        m=T.train_es(m,dl,Xva,bp["lr"],bp["wd"]);
        with torch.no_grad(): P.append(m(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy())
        log.info(f"seed {s} ok")
    PR=np.mean(P,0)
    tgt={h:np.array([dts[i+h] for i in te]) for h in (1,7)}
    o={h:np.array([obs[i+h] for i in te]) for h in (1,7)}
    tft={h:PR[:,h-1,1] for h in (1,7)}
    pers={h:np.array([q[i] for i in te]) for h in (1,7)}   # persistencia naive Q_{t+h}=Q_t
    # LightGBM (sin futuro) P50 por horizonte
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]; lgbm={}
    for h in (1,7):
        d=df.copy(); d["yt"]=d["q"].shift(-h); trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2024-01-01")].dropna(subset=FE)
        g=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
        g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0)))
        lgbm[h]=pd.Series(np.expm1(g.predict(ted[FE])),index=ted.index+pd.Timedelta(days=h))

    # crecida principal 2024 (pico de obs en test)
    mk1=np.isfinite(o[1]); pk=tgt[1][mk1][np.argmax(o[1][mk1])]; z0,z1=pk-pd.Timedelta(days=40),pk+pd.Timedelta(days=45)
    log.info(f"Crecida pico test: {pd.Timestamp(pk).date()}")

    def draw(ax,h,x0=None,x1=None):
        m=np.isfinite(o[h]); ax.plot(tgt[h][m],o[h][m],"o-",color="black",ms=3,lw=0.8,label="Obs real",zorder=5)
        ax.plot(tgt[h],tft[h],"-",color="#08519c",lw=1.6,label="TFT honesto",zorder=4)
        lg=lgbm[h].reindex(pd.DatetimeIndex(tgt[h])); ax.plot(tgt[h],lg.values,"-",color="#e6550d",lw=1.4,label="LightGBM",zorder=3)
        ax.plot(tgt[h],pers[h],"-",color="#31a354",lw=1.1,alpha=0.8,label="Persistencia",zorder=2)
        ax.axhline(Q90,ls="--",color="#d62728",lw=1.0,alpha=0.8,label=f"Q90={Q90:.1f}")
        if x0: ax.set_xlim(x0,x1)
        ax.set_ylabel("Caudal [m³/s]"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    fig,axes=plt.subplots(3,1,figsize=(15,12))
    draw(axes[0],7); axes[0].set_title("A) Test completo — h=7 días (TFT vs LightGBM vs persistencia)",loc="left",fontsize=11); axes[0].legend(fontsize=8,ncol=5,loc="upper right")
    draw(axes[1],1,z0,z1); axes[1].set_title(f"B) Zoom crecida {pd.Timestamp(pk).date()} — h=1 día",loc="left",fontsize=11)
    draw(axes[2],7,z0,z1); axes[2].set_title(f"C) Zoom crecida {pd.Timestamp(pk).date()} — h=7 días",loc="left",fontsize=11)
    for ax in axes: ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle("Comparación de modelos en el periodo de test — overlay y zoom a crecida",fontsize=13,y=0.997)
    fig.tight_layout()
    for p in [OUT/"120_test_overlay_zoom.png", FIGP/"fig29_test_overlay_zoom.png"]: fig.savefig(p,dpi=300)
    plt.close(fig); log.info("Saved: 120_test_overlay_zoom.png + fig29")

if __name__=="__main__":
    main()
