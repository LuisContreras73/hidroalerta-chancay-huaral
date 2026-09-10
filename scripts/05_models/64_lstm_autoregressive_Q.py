#!/usr/bin/env python3
"""
Script 64: LSTM Autoregresivo — usar Q reciente como feature (idea A).

Naturaleza del dato: Q muy persistente (r(1)=0.85, r(7)=0.55). Conocer el caudal
de hoy da mucha información sobre el de mañana. El LSTM actual NO usaba Q como
input (para poder predecir en gaps). Aquí lo añadimos, rellenando gaps con GR4J
corregido — exactamente la situación operativa: si el sensor funcionó, uso ese
dato; si falló, uso la reconstrucción física.

Features autoregresivas añadidas (compartidas, del outlet):
  q_mm     = Q del día actual t   (válido para predecir q_next_1d = Q[t+1])
  q_lag7   = Q hace 7 días
  q_roll7  = media móvil 7 días (hasta hoy)
Todas de q_mm en D6 = Q_obs.combine_first(GR4J_corregido) → sin gaps.

Compara: LSTM-AR (con Q) vs LSTM base (sin Q) en el MISMO test (Q obs real).

ANTI-LEAKAGE: q_mm[t] y sus lags solo usan información ≤ t. El target Q[t+1] nunca
entra como feature.

Salida:
  outputs/ml_Q/autoregressive_comparison.csv
  outputs/ml_Q/autoregressive_predictions.csv
  outputs/figures/ml_Q/AR01_autoregressive_vs_base.png
"""
import json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("lstm_AR")

D6_CSV = ROOT/"data/model_ready/D6_multientity.csv"
QOBS   = ROOT/"data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR= ROOT/"outputs/ml_Q"; FIG_DIR=ROOT/"outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)

Q_CONV=86.4/3062.62; Q90=40.89; SEED=42
ENC=90; HID=64; LAY=2; DROP=0.3; LR=7e-4; WD=1e-5; BATCH=64; MAX_EP=300; PAT=40
TARGET="q_next_1d"

SPATIAL=["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED =["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]
# Features autoregresivas (compartidas, derivadas de q_mm)
AR_FEATURES=["q_mm","q_lag7","q_roll7"]

def build_wide(D6, autoregressive=False):
    pivots=[]
    for c in SPATIAL:
        if c in D6.columns:
            p=D6.pivot_table(index="date",columns="entity_id",values=c)
            p.columns=[f"{c}_{e}" for e in p.columns]; pivots.append(p)
    ref=D6[D6["entity_id"]=="sub_634"].set_index("date")
    keep=[c for c in SHARED+[TARGET] if c in ref.columns]
    base=pd.concat(pivots+[ref[keep]],axis=1).sort_index()
    if autoregressive:
        q=ref["q_mm"].copy()   # ya es obs + GR4J corregido
        base["q_mm"]=q
        base["q_lag7"]=q.shift(7)
        base["q_roll7"]=q.rolling(7,min_periods=3).mean()
    return base

def make_seq(X,y,enc):
    Xs,ys,idx=[],[],[]
    for i in range(enc,len(X)):
        if np.isfinite(y[i]):
            Xs.append(X[i-enc:i]);ys.append(y[i]);idx.append(i)
    return np.array(Xs,np.float32),np.array(ys,np.float32),np.array(idx)

def nse(o,p): return 1-np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)
def nse_sqrt(o,p):
    w=np.maximum(o,0)**0.5;return 1-np.sum(w*(o-p)**2)/(np.sum(w*(o-o.mean())**2)+1e-12)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1];a=p.std()/(o.std()+1e-12);b=p.mean()/(o.mean()+1e-12)
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def metr(o,p,thr):
    eo=o>thr;ep=p>thr
    TP=int(np.sum(eo&ep));FP=int(np.sum(~eo&ep));FN=int(np.sum(eo&~ep));TN=int(np.sum(~eo&~ep))
    POD=TP/(TP+FN+1e-9);FAR=FP/(FP+TN+1e-9);CSI=TP/(TP+FP+FN+1e-9)
    j=0.25*nse_sqrt(o,p)+0.25*nse(o,p)+0.30*CSI+0.10*POD-0.10*FAR
    return {"NSE":round(nse(o,p),4),"NSE_sqrt":round(nse_sqrt(o,p),4),"KGE":round(kge(o,p),4),
            "J_alert":round(j,4),"POD":round(POD,3),"FAR":round(FAR,3),"CSI":round(CSI,3),"N":len(o)}

def main():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(SEED);np.random.seed(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Script 64: LSTM autoregresivo vs base | device={dev}")

    D6=pd.read_csv(D6_CSV,parse_dates=["date"])
    qobs=pd.read_csv(QOBS,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"].dropna()

    class L(nn.Module):
        def __init__(s,nf):
            super().__init__()
            s.lstm=nn.LSTM(nf,HID,LAY,batch_first=True,dropout=DROP)
            s.head=nn.Sequential(nn.Linear(HID,HID//2),nn.ReLU(),nn.Dropout(DROP),nn.Linear(HID//2,1))
        def forward(s,x): o,_=s.lstm(x);return s.head(o[:,-1,:]).squeeze(-1)
    lf=nn.MSELoss()

    def run(autoregressive, tag):
        wide=build_wide(D6, autoregressive)
        feat=[c for c in wide.columns if c!=TARGET]
        dates=wide.index
        X=wide[feat].fillna(0).values.astype(np.float32); y=wide[TARGET].values.astype(np.float32)
        m_tr=dates<="2015-12-31"; m_vl=(dates>="2016-01-01")&(dates<="2020-12-31"); m_te=dates>="2021-01-01"
        mu,sd=X[m_tr].mean(0),X[m_tr].std(0)+1e-8; X=(X-mu)/sd
        ylog=np.log1p(np.clip(y,0,None));ymu,ysd=np.nanmean(ylog[m_tr]),np.nanstd(ylog[m_tr])+1e-8
        ysc=(ylog-ymu)/ysd; inv=lambda v:np.expm1(v*ysd+ymu)
        Xseq,yseq,idx=make_seq(X,ysc,ENC); sp=np.empty(len(idx),object)
        for k,i in enumerate(idx): sp[k]="train" if m_tr[i] else ("val" if m_vl[i] else "test")
        tr,vl,te=sp=="train",sp=="val",sp=="test"
        Xt=torch.tensor(Xseq[tr]);yt=torch.tensor(yseq[tr])
        Xv=torch.tensor(Xseq[vl]);yv=torch.tensor(yseq[vl])
        Xte=torch.tensor(Xseq[te])
        model=L(len(feat)).to(dev)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
        dl=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH,shuffle=True)
        Xvd,yvn=Xv.to(dev),yv.numpy();best=np.inf;bs=None;c=0;t0=time.time()
        for ep in range(MAX_EP):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad();lf(model(xb),yb).backward();opt.step()
            model.eval()
            with torch.no_grad(): vl_=float(np.mean((model(Xvd).cpu().numpy()-yvn)**2))
            if vl_<best:best=vl_;bs={k:v.cpu().clone() for k,v in model.state_dict().items()};c=0
            else:
                c+=1
                if c>=PAT: break
        model.load_state_dict(bs);model.eval()
        with torch.no_grad(): pr_te=inv(model(Xte.to(dev)).cpu().numpy())
        dts=dates[idx][te]
        # Evaluar SOLO vs Q obs real, con unidades y alineación correctas:
        # pred[t] = q_next_1d = Q[t+1]. Comparar contra Q_obs[t+1] en mm/d.
        qobs_mm=qobs*Q_CONV                              # m³/s → mm/d
        next_dates=pd.DatetimeIndex(dts)+pd.Timedelta(days=1)
        o=pd.Series(qobs_mm.reindex(next_dates).values, index=dts)  # Q_obs[t+1]
        s=pd.Series(pr_te,index=dts)
        valid=o.notna()
        m=metr(o[valid].values, s[valid].values, Q90*Q_CONV)   # umbral en mm/d
        log.info(f"  [{tag}] {len(feat)} feats | {ep+1} epochs {time.time()-t0:.0f}s | "
                 f"test honesto: NSE={m['NSE']} KGE={m['KGE']} J_alert={m['J_alert']} POD={m['POD']} N={m['N']}")
        # Guardar en m³/s para visualización
        return m, pd.DataFrame({"date":dts[valid.values],
                                "q_obs":o[valid].values/Q_CONV,
                                f"q_pred_{tag}":s[valid].values/Q_CONV})

    log.info("\n[1] LSTM BASE (sin Q autoregresivo) ...")
    m_base, pred_base = run(False, "base")
    log.info("\n[2] LSTM AUTOREGRESIVO (con Q reciente) ...")
    m_ar, pred_ar = run(True, "AR")

    # Comparación
    log.info("\n"+"="*60)
    log.info("COMPARACIÓN (test = Q obs real 2021-2025)")
    comp=pd.DataFrame([{"modelo":"LSTM base (sin Q)",**m_base},
                       {"modelo":"LSTM autoregresivo (+Q)",**m_ar}])
    log.info(comp[["modelo","NSE","NSE_sqrt","KGE","J_alert","POD","FAR","CSI"]].to_string(index=False))
    log.info(f"\nMEJORA AR: ΔNSE={m_ar['NSE']-m_base['NSE']:+.4f} ΔJ_alert={m_ar['J_alert']-m_base['J_alert']:+.4f}")
    comp.to_csv(OUT_DIR/"autoregressive_comparison.csv",index=False)
    pred=pred_base.merge(pred_ar,on=["date","q_obs"],how="outer")
    pred.to_csv(OUT_DIR/"autoregressive_predictions.csv",index=False)
    log.info("SCRIPT 64 COMPLETADO")

if __name__=="__main__":
    main()
