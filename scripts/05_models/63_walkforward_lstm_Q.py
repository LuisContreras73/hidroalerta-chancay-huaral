#!/usr/bin/env python3
"""
Script 63: LSTM Walk-Forward Cross-Validation con target Q.

Motivación (naturaleza de los datos):
  Los eventos de alerta (Q>Q90) son raros y se concentran en 2020-2023 (138 eventos).
  Un split cronológico 60/20/20 deja el test (2024-2025) SIN eventos (período seco+gaps).
  → Un solo split no representa todos los regímenes. Solución: walk-forward CV.

Cada fold:
  - FASE 1: pre-train con GR4J 1981-2020 (física) — se entrena UNA vez, se reutiliza
  - FASE 2: fine-tune con Q obs real hasta el año N
  - TEST:   Q obs real del año N+1 (solo días con dato)

Folds (expanding window sobre Q real):
  A: finetune ≤2022 → test 2023
  B: finetune ≤2023 → test 2024
  C: finetune ≤2024 → test 2025

Métricas promediadas sobre folds = evaluación robusta a través de regímenes.

Salida:
  outputs/ml_Q/walkforward_results.csv
  outputs/ml_Q/walkforward_predictions.csv
"""
import json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("walkforward")

D6_CSV  = ROOT/"data/model_ready/D6_multientity.csv"
QOBS    = ROOT/"data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR = ROOT/"outputs/ml_Q"
Q_CONV=86.4/3062.62; Q90=40.89; SEED=42; ENC=90; HID=64; LAY=2; DROP=0.2
SPATIAL=["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED=["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]
TARGET="q_next_1d"

FOLDS=[("2022-12-31","2023-01-01","2023-12-31"),
       ("2023-12-31","2024-01-01","2024-12-31"),
       ("2024-12-31","2025-01-01","2025-12-31")]

def build_wide(D6):
    pv=[]
    for c in SPATIAL:
        if c in D6.columns:
            p=D6.pivot_table(index="date",columns="entity_id",values=c)
            p.columns=[f"{c}_{e}" for e in p.columns]; pv.append(p)
    ref=D6[D6["entity_id"]=="sub_634"].set_index("date")
    pv.append(ref[[c for c in SHARED+[TARGET] if c in ref.columns]])
    return pd.concat(pv,axis=1).sort_index()

def nse(o,p): return 1-np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)
def nse_sqrt(o,p):
    w=np.maximum(o,0)**0.5; return 1-np.sum(w*(o-p)**2)/(np.sum(w*(o-o.mean())**2)+1e-12)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1];a=p.std()/(o.std()+1e-12);b=p.mean()/(o.mean()+1e-12)
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def alert(o,p,thr):
    eo=o>thr;ep=p>thr
    TP=int(np.sum(eo&ep));FP=int(np.sum(~eo&ep));FN=int(np.sum(eo&~ep));TN=int(np.sum(~eo&~ep))
    POD=TP/(TP+FN+1e-9);FAR=FP/(FP+TN+1e-9);CSI=TP/(TP+FP+FN+1e-9)
    return POD,FAR,CSI,TP,FP,FN

def main():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Script 63: Walk-Forward CV | device={dev}")

    D6=pd.read_csv(D6_CSV,parse_dates=["date"])
    wide=build_wide(D6); feat=[c for c in wide.columns if c!=TARGET]; dates=wide.index
    X=wide[feat].fillna(0).values.astype(np.float32); y=wide[TARGET].values.astype(np.float32)

    # Q OBSERVADO REAL (para evaluar test solo contra verdad, no GR4J de los gaps)
    qobs_raw=pd.read_csv(QOBS,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    # q_next_1d obs real = Q obs del día siguiente (mm/d)
    qobs_next=(qobs_raw.shift(-1)*Q_CONV)   # m³/s → mm/d, alineado a q_next_1d
    obs_real_by_date={d:v for d,v in qobs_next.items() if np.isfinite(v)}

    # Escalado con período pre-2020 (GR4J pretrain)
    m_pre=dates<="2017-12-31"
    mu,sd=X[m_pre].mean(0),X[m_pre].std(0)+1e-8; Xs=(X-mu)/sd
    ylog=np.log1p(np.clip(y,0,None)); ymu,ysd=np.nanmean(ylog[m_pre]),np.nanstd(ylog[m_pre])+1e-8
    ysc=(ylog-ymu)/ysd; inv=lambda v:np.expm1(v*ysd+ymu)

    def seq(mask):
        Xs2,yy,idx=[],[],[]
        for i in range(ENC,len(Xs)):
            if mask[i] and np.isfinite(ysc[i]):
                Xs2.append(Xs[i-ENC:i]);yy.append(ysc[i]);idx.append(i)
        return (torch.tensor(np.array(Xs2,np.float32)),torch.tensor(np.array(yy,np.float32)),
                np.array(idx))

    class L(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm=nn.LSTM(len(feat),HID,LAY,batch_first=True,dropout=DROP)
            s.head=nn.Sequential(nn.Linear(HID,HID//2),nn.ReLU(),nn.Dropout(DROP),nn.Linear(HID//2,1))
        def forward(s,x): o,_=s.lstm(x);return s.head(o[:,-1,:]).squeeze(-1)
    lf=nn.MSELoss()

    def train(model,Xt,yt,Xv,yv,lr,mx,pat):
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        dl=DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True)
        Xvd,yvn=Xv.to(dev),yv.numpy();best=np.inf;bs=None;c=0
        for ep in range(mx):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad();lf(model(xb),yb).backward();opt.step()
            model.eval()
            with torch.no_grad(): vl=float(np.mean((model(Xvd).cpu().numpy()-yvn)**2))
            if vl<best:best=vl;bs={k:v.cpu().clone() for k,v in model.state_dict().items()};c=0
            else:
                c+=1
                if c>=pat: break
        model.load_state_dict(bs);return model

    # FASE 1: pretrain GR4J 1981-2020 (una vez)
    log.info("FASE 1: pretrain GR4J 1981-2017 (val 2018-2020) ...")
    Xp,yp,_=seq(dates<="2017-12-31")
    Xpv,ypv,_=seq((dates>="2018-01-01")&(dates<="2020-12-31"))
    base=L().to(dev); base=train(base,Xp,yp,Xpv,ypv,1e-3,120,15)
    base_state={k:v.cpu().clone() for k,v in base.state_dict().items()}
    log.info("  Pretrain completo.")

    # WALK-FORWARD folds
    rows=[]; all_preds=[]
    for fi,(ft_end,te0,te1) in enumerate(FOLDS, 1):
        log.info(f"\n=== FOLD {fi}: finetune Q real ≤{ft_end} → test {te0[:4]} ===")
        m=L().to(dev); m.load_state_dict(base_state)  # reiniciar desde pretrain
        # finetune con Q real desde 2020-09 hasta ft_end (val = último 15%)
        ft_mask=(dates>="2020-09-01")&(dates<=ft_end)
        Xf,yf,idxf=seq(ft_mask)
        if len(Xf)<100:
            log.warning(f"  Fold {fi}: pocos datos finetune ({len(Xf)}), saltando"); continue
        ncut=int(len(Xf)*0.85)
        m=train(m, Xf[:ncut],yf[:ncut], Xf[ncut:],yf[ncut:], 1e-4,60,10)
        # test — predecir sobre TODOS los días del año (no filtrar por target D6)
        te_mask=(dates>=te0)&(dates<=te1)
        Xt_list,idxt=[],[]
        for i in range(ENC,len(Xs)):
            if te_mask[i]:
                Xt_list.append(Xs[i-ENC:i]); idxt.append(i)
        if len(Xt_list)<10:
            log.warning(f"  Fold {fi}: test vacío"); continue
        Xt=torch.tensor(np.array(Xt_list,np.float32)); idxt=np.array(idxt)
        m.eval()
        with torch.no_grad(): pr_all=inv(m(Xt.to(dev)).cpu().numpy())
        # Filtrar SOLO días con Q OBSERVADO REAL (no GR4J de gaps)
        te_dates=dates[idxt]
        keep=[k for k,d in enumerate(te_dates) if d in obs_real_by_date]
        if len(keep)<5:
            log.warning(f"  Fold {fi}: <5 días Q obs real, no evaluable"); continue
        pr=pr_all[keep]
        ob=np.array([obs_real_by_date[te_dates[k]] for k in keep])
        te_dates=te_dates[keep]
        POD,FAR,CSI,TP,FP,FN=alert(ob,pr,Q90*Q_CONV)
        row={"fold":fi,"test_year":te0[:4],"n_test":len(ob),
             "NSE":round(nse(ob,pr),4),"NSE_sqrt":round(nse_sqrt(ob,pr),4),
             "KGE":round(kge(ob,pr),4),"POD":round(POD,3),"FAR":round(FAR,3),
             "CSI":round(CSI,3),"alerts_obs":TP+FN}
        rows.append(row)
        log.info(f"  test {te0[:4]}: NSE={row['NSE']} KGE={row['KGE']} "
                 f"alertas_obs={TP+FN} POD={row['POD']} n_obs_real={len(ob)}")
        for d,o,p in zip(te_dates, ob, pr):
            all_preds.append({"fold":fi,"date":d,"q_obs_m3s":o/Q_CONV,"q_pred_m3s":p/Q_CONV})

    res=pd.DataFrame(rows)
    res.to_csv(OUT_DIR/"walkforward_results.csv",index=False)
    pd.DataFrame(all_preds).to_csv(OUT_DIR/"walkforward_predictions.csv",index=False)

    log.info("\n"+"="*60)
    log.info("WALK-FORWARD CV — resultados por fold")
    log.info(res.to_string(index=False))
    log.info("")
    log.info(f"PROMEDIO: NSE={res['NSE'].mean():.3f}±{res['NSE'].std():.3f} | "
             f"KGE={res['KGE'].mean():.3f} | CSI={res['CSI'].mean():.3f}")
    log.info(f"  (vs single-split fine-tune NSE=0.396 en solo test 2024-2025)")
    log.info("SCRIPT 63 COMPLETADO")

if __name__=="__main__":
    main()
