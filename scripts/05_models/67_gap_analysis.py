#!/usr/bin/env python3
"""
Script 67: Análisis de gaps en Q observado + predicción LSTM vs GR4J en los gaps.

Problema: el Q observado (47E214D2) tiene 371 días sin dato (18%), el mayor de
dic-2024 a sep-2025 (277 días). Durante esos gaps el modelo es la ÚNICA estimación.

Como el LSTM y el GR4J usan SOLO meteorología (no caudal pasado), ambos pueden
predecir Q en cualquier día. Este script:
  1. Reentrena el LSTM fine-tuned (fase1 GR4J + fase2 Q real)
  2. Predice sobre todo 2020-2025
  3. Compara LSTM vs GR4J:
     - En días CON obs: cuál acierta más
     - En los gaps: acuerdo/desacuerdo entre modelos (sin verdad)

Salida:
  outputs/ml_Q/gap_predictions.csv
  outputs/figures/ml_Q/GAP01_gaps_lstm_vs_gr4j.png
"""
import json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("gap_analysis")

D6_CSV  = ROOT / "data/model_ready/D6_multientity.csv"
GR4J_CSV= ROOT / "data/gold/G1_q_sim_gr4j.csv"
QOBS_CSV= ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR = ROOT / "outputs/ml_Q"
Q_CONV  = 86.4/3062.62; Q90=40.89; Q99=77.72; SEED=42
ENC=90; HID=64; LAY=2; DROP=0.2

SPATIAL=["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED =["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]
TARGET="q_next_1d"

def build_wide(D6):
    pivots=[]
    for c in SPATIAL:
        if c in D6.columns:
            p=D6.pivot_table(index="date",columns="entity_id",values=c)
            p.columns=[f"{c}_{e}" for e in p.columns]; pivots.append(p)
    ref=D6[D6["entity_id"]=="sub_634"].set_index("date")
    pivots.append(ref[[c for c in SHARED+[TARGET] if c in ref.columns]])
    return pd.concat(pivots,axis=1).sort_index()

def make_seq_all(X, enc):
    """Secuencias para TODOS los días (no filtra por target válido) → predecir en gaps."""
    Xs, idx = [], []
    for i in range(enc, len(X)):
        Xs.append(X[i-enc:i]); idx.append(i)
    return np.array(Xs, np.float32), np.array(idx)

def nse(o,p): return 1-np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1];a=p.std()/(o.std()+1e-12);b=p.mean()/(o.mean()+1e-12)
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)

def main():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Script 67: análisis de gaps | device={dev}")

    D6=pd.read_csv(D6_CSV,parse_dates=["date"])
    wide=build_wide(D6); feat=[c for c in wide.columns if c!=TARGET]; dates=wide.index
    X=wide[feat].fillna(0).values.astype(np.float32)
    y=wide[TARGET].values.astype(np.float32)

    # Períodos (igual que fine-tuning script 62)
    m_pre=dates<="2017-12-31"; m_prev=(dates>="2018-01-01")&(dates<="2020-12-31")
    m_ft=(dates>="2021-01-01")&(dates<="2022-12-31"); m_ftv=(dates>="2023-01-01")&(dates<="2023-12-31")

    mu,sd=X[m_pre].mean(0),X[m_pre].std(0)+1e-8; X=(X-mu)/sd
    ylog=np.log1p(np.clip(y,0,None)); ymu,ysd=np.nanmean(ylog[m_pre]),np.nanstd(ylog[m_pre])+1e-8
    ys=(ylog-ymu)/ysd; inv=lambda v:np.expm1(v*ysd+ymu)

    # Secuencias para entrenar (solo target válido) y para predecir (todas)
    def seq_train(mask):
        Xs,yy,idx=[],[],[]
        for i in range(ENC,len(X)):
            if mask[i] and np.isfinite(ys[i]):
                Xs.append(X[i-ENC:i]); yy.append(ys[i]); idx.append(i)
        return torch.tensor(np.array(Xs,np.float32)), torch.tensor(np.array(yy,np.float32))
    Xp,yp=seq_train(m_pre); Xpv,ypv=seq_train(m_prev)
    Xf,yf=seq_train(m_ft);  Xfv,yfv=seq_train(m_ftv)

    class L(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm=nn.LSTM(len(feat),HID,LAY,batch_first=True,dropout=DROP)
            s.head=nn.Sequential(nn.Linear(HID,HID//2),nn.ReLU(),nn.Dropout(DROP),nn.Linear(HID//2,1))
        def forward(s,x): o,_=s.lstm(x); return s.head(o[:,-1,:]).squeeze(-1)

    lf=nn.MSELoss()
    def train(model,Xt,yt,Xv,yv,lr,mx,pat):
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        dl=DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True)
        Xvd,yvn=Xv.to(dev),yv.numpy(); best=np.inf;bs=None;c=0
        for ep in range(mx):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); lf(model(xb),yb).backward(); opt.step()
            model.eval()
            with torch.no_grad(): vl=float(np.mean((model(Xvd).cpu().numpy()-yvn)**2))
            if vl<best: best=vl;bs={k:v.cpu().clone() for k,v in model.state_dict().items()};c=0
            else:
                c+=1
                if c>=pat: break
        model.load_state_dict(bs); return model

    log.info("Fase 1 (GR4J) ..."); m=L().to(dev); m=train(m,Xp,yp,Xpv,ypv,1e-3,120,15)
    log.info("Fase 2 (Q real) ..."); m=train(m,Xf,yf,Xfv,yfv,1e-4,60,10)

    # Predecir sobre TODO el período (incluidos gaps)
    Xall,idxall=make_seq_all(X,ENC)
    m.eval()
    with torch.no_grad():
        pred_all=inv(m(torch.tensor(Xall).to(dev)).cpu().numpy())
    dates_all=dates[idxall]
    lstm_pred=pd.Series(pred_all/Q_CONV, index=dates_all, name="lstm_m3s")  # m³/s

    # GR4J
    gr4j=pd.read_csv(GR4J_CSV,index_col=0,parse_dates=True)["q_sim_m3s"]
    # Q obs con gaps
    qobs=pd.read_csv(QOBS_CSV,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"]

    # Alinear todo en 2020-2025
    rng=pd.date_range("2020-09-01","2025-12-31",freq="D")
    df=pd.DataFrame(index=rng)
    df["q_obs"]=qobs.reindex(rng)
    df["lstm"]=lstm_pred.reindex(rng)
    df["gr4j"]=gr4j.reindex(rng)
    df["has_obs"]=df["q_obs"].notna()
    df.to_csv(OUT_DIR/"gap_predictions.csv")

    # Métricas en días CON obs (LSTM vs GR4J vs verdad)
    with_obs=df[df["has_obs"]].dropna(subset=["lstm","gr4j"])
    o=with_obs["q_obs"].values
    log.info("\n=== Días CON obs (LSTM vs GR4J vs verdad) ===")
    log.info(f"  LSTM: NSE={nse(o,with_obs['lstm'].values):.3f} KGE={kge(o,with_obs['lstm'].values):.3f}")
    log.info(f"  GR4J: NSE={nse(o,with_obs['gr4j'].values):.3f} KGE={kge(o,with_obs['gr4j'].values):.3f}")

    # En gaps: acuerdo LSTM vs GR4J
    gaps=df[~df["has_obs"]].dropna(subset=["lstm","gr4j"])
    if len(gaps)>0:
        diff=gaps["lstm"]-gaps["gr4j"]
        log.info(f"\n=== En GAPS ({len(gaps)} días sin obs) — acuerdo entre modelos ===")
        log.info(f"  LSTM media={gaps['lstm'].mean():.2f} | GR4J media={gaps['gr4j'].mean():.2f} m³/s")
        log.info(f"  Diferencia LSTM-GR4J: media={diff.mean():.2f}, |dif| media={diff.abs().mean():.2f} m³/s")
        log.info(f"  Correlación LSTM-GR4J en gaps: r={np.corrcoef(gaps['lstm'],gaps['gr4j'])[0,1]:.3f}")

    log.info("\nGuardado gap_predictions.csv")

if __name__=="__main__":
    main()
