#!/usr/bin/env python3
"""
Script 72: TFT-lite con TRANSFER LEARNING + métricas completas, vs LightGBM AR.

Transfer (aprovecha 14600 días de GR4J que el TFT no usaba):
  FASE 1 pre-train: GR4J corregido 1981-2017 (val 2018-2020)
  FASE 2 fine-tune: Q real 2021-2023 (val 2023-H2), LR bajo
  TEST común:       Q real 2024-2025

Comparación justa: LightGBM AR re-entrenado (GR4J 1981-2020 + Q real 2021-2023),
mismo test 2024-2025.

Métricas completas: NSE, NSE_sqrt, KGE, logNSE, PBIAS, PBIAS_Q90, POD, FAR, CSI, HSS
+ desglose por temporada (húmeda nov-abr / seca may-oct) y fase ENSO (Niño/Niña/neutral).

Salida:
  outputs/ml_Q/transfer_metrics_full.csv
  outputs/ml_Q/transfer_predictions.csv
"""
import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log=logging.getLogger("tft_transfer")

D6_CSV=ROOT/"data/model_ready/D6_multientity.csv"
QOBS=ROOT/"data/silver/snirh/S1_snirh_daily_q.csv"
ONI_CSV=ROOT/"data/silver/enso/S3_oni_1950_2026.csv"
OUT_DIR=ROOT/"outputs/ml_Q"; OUT_DIR.mkdir(parents=True,exist_ok=True)
Q_CONV=86.4/3062.62; Q90=40.89; SEED=42
ENC=45; HID=32; HEADS=2; DROP=0.3; LR=7e-4; LR_FT=1e-4; WD=3e-5; BATCH=64
MAX_EP1=200; MAX_EP2=80; PAT1=30; PAT2=15

SPATIAL=["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED=["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]
KEY_SUBS=["sub_649","sub_655","sub_656"]

def build_wide(D6):
    cols={}
    for c in SPATIAL:
        piv=D6.pivot_table(index="date",columns="entity_id",values=c)
        cols[f"{c}_basin"]=piv.mean(axis=1)
        for e in KEY_SUBS:
            if e in piv.columns: cols[f"{c}_{e}"]=piv[e]
    df=pd.DataFrame(cols)
    ref=D6[D6["entity_id"]=="sub_634"].set_index("date")
    df=df.join(ref[SHARED+["q_mm","q_next_1d","q_sum_next_7d"]]).sort_index()
    q=df["q_mm"]
    df["q_lag7"]=q.shift(7); df["q_roll7"]=q.rolling(7,min_periods=3).mean()
    df["q_roll30"]=q.rolling(30,min_periods=10).mean()
    return df

# ── Métricas completas ──────────────────────────────────────────────────────
def _nse(o,p): return 1-np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)
def _nse_sqrt(o,p):
    w=np.maximum(o,0)**0.5;return 1-np.sum(w*(o-p)**2)/(np.sum(w*(o-o.mean())**2)+1e-12)
def _lognse(o,p):
    lo,lp=np.log(o+1),np.log(np.clip(p,0,None)+1)
    return 1-np.sum((lo-lp)**2)/(np.sum((lo-lo.mean())**2)+1e-12)
def _kge(o,p):
    r=np.corrcoef(o,p)[0,1];a=p.std()/(o.std()+1e-12);b=p.mean()/(o.mean()+1e-12)
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def full_metrics(o,p,thr):
    o=np.asarray(o);p=np.asarray(p)
    eo=o>thr;ep=p>thr
    TP=int(np.sum(eo&ep));FP=int(np.sum(~eo&ep));FN=int(np.sum(eo&~ep));TN=int(np.sum(~eo&~ep))
    POD=TP/(TP+FN+1e-9);FAR=FP/(FP+TN+1e-9);CSI=TP/(TP+FP+FN+1e-9)
    dh=(TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)
    HSS=2*(TP*TN-FP*FN)/(dh+1e-9) if dh>0 else 0
    pbias=(p.sum()-o.sum())/(o.sum()+1e-12)*100
    q90l=np.percentile(o,90); pk=o>q90l
    pb90=(p[pk].sum()-o[pk].sum())/(o[pk].sum()+1e-12)*100 if pk.sum()>3 else np.nan
    return {"NSE":round(_nse(o,p),3),"NSE_sqrt":round(_nse_sqrt(o,p),3),
            "logNSE":round(_lognse(o,p),3),"KGE":round(_kge(o,p),3),
            "PBIAS":round(pbias,1),"PBIAS_Q90":round(pb90,1) if not np.isnan(pb90) else np.nan,
            "POD":round(POD,3),"FAR":round(FAR,3),"CSI":round(CSI,3),"HSS":round(HSS,3),"N":len(o)}

def main():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset,DataLoader
    from lightgbm import LGBMRegressor
    torch.manual_seed(SEED);np.random.seed(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Script 72: TFT transfer + métricas completas | device={dev}")

    D6=pd.read_csv(D6_CSV,parse_dates=["date"])
    qobs=pd.read_csv(QOBS,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm=qobs*Q_CONV
    oni=pd.read_csv(ONI_CSV,parse_dates=["date"]).set_index("date")["oni"]
    df=build_wide(D6); dates=df.index
    feat=[c for c in df.columns if c not in ["q_next_1d","q_sum_next_7d"]]

    # Máscaras transfer
    m_p =dates<="2017-12-31"
    m_pv=(dates>="2018-01-01")&(dates<="2020-12-31")
    m_ft=(dates>="2021-01-01")&(dates<="2023-06-30")
    m_fv=(dates>="2023-07-01")&(dates<="2023-12-31")
    m_te=dates>="2024-01-01"
    log.info(f"Features={len(feat)} | pretrain≤2017, finetune 2021-23H1, test 2024-2025")

    targets={"q_next_1d":(qobs_mm.shift(-1),1),
             "q_sum_next_7d":(qobs_mm.shift(-1).rolling(7).sum().shift(-6),7)}

    class VSN(nn.Module):
        def __init__(s,nf,h):
            super().__init__();s.w=nn.Sequential(nn.Linear(nf,h),nn.ReLU(),nn.Linear(h,nf))
        def forward(s,x): return x*torch.softmax(s.w(x),-1)*x.shape[-1]
    class TFTLite(nn.Module):
        def __init__(s,nf):
            super().__init__()
            s.vsn=VSN(nf,HID);s.proj=nn.Linear(nf,HID)
            s.pos=nn.Parameter(torch.randn(1,ENC,HID)*0.02)
            s.lstm=nn.LSTM(HID,HID,1,batch_first=True)
            s.attn=nn.MultiheadAttention(HID,HEADS,dropout=DROP,batch_first=True)
            s.norm=nn.LayerNorm(HID);s.drop=nn.Dropout(DROP)
            s.head=nn.Sequential(nn.Linear(HID,HID//2),nn.ReLU(),nn.Dropout(DROP),nn.Linear(HID//2,1))
        def forward(s,x):
            h=s.proj(s.vsn(x))+s.pos;o,_=s.lstm(h)
            a,_=s.attn(o,o,o);h=s.norm(o+s.drop(a))
            return s.head(h[:,-1,:]).squeeze(-1)
    huber=nn.HuberLoss(delta=1.0)

    def seqs(X,ysc):
        Xs,ys,idx=[],[],[]
        for i in range(ENC,len(X)):
            if np.isfinite(ysc[i]): Xs.append(X[i-ENC:i]);ys.append(ysc[i]);idx.append(i)
        return np.array(Xs,np.float32),np.array(ys,np.float32),np.array(idx)

    def train_phase(model,Xtr,ytr,Xv,yv,lr,mx,pat):
        opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=WD)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",factor=0.5,patience=8)
        dl=DataLoader(TensorDataset(Xtr,ytr),batch_size=BATCH,shuffle=True)
        Xvd,yvn=Xv.to(dev),yv.numpy();best=np.inf;bs=None;c=0
        for ep in range(mx):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad();huber(model(xb),yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            model.eval()
            with torch.no_grad(): vl=float(np.mean((model(Xvd).cpu().numpy()-yvn)**2))
            sched.step(vl)
            if vl<best:best=vl;bs={k:v.cpu().clone() for k,v in model.state_dict().items()};c=0
            else:
                c+=1
                if c>=pat:break
        model.load_state_dict(bs);return model

    rows=[]; preds_all=[]
    for target,(obs_real,hmult) in targets.items():
        y=df[target].values.astype(np.float32)
        X=df[feat].fillna(0).values.astype(np.float32)
        mu,sd=X[m_p].mean(0),X[m_p].std(0)+1e-8;Xs=(X-mu)/sd
        ylog=np.log1p(np.clip(y,0,None));ymu,ysd=np.nanmean(ylog[m_p]),np.nanstd(ylog[m_p])+1e-8
        ysc=(ylog-ymu)/ysd;inv=lambda v:np.expm1(v*ysd+ymu)
        thr=Q90*Q_CONV*hmult

        # ── TFT transfer ──
        def subset(mask):
            Xq,yq,_=seqs(Xs*np.where(mask[:,None,None] if False else 1,1,1),ysc)  # placeholder
            return None
        # construir secuencias globales una vez, filtrar por máscara de la fecha t
        Xseq,yseq,idx=seqs(Xs,ysc)
        dts=dates[idx]
        def msk(m): return np.isin(idx,np.where(m)[0])
        t=time.time()
        model=TFTLite(len(feat)).to(dev)
        model=train_phase(model,torch.tensor(Xseq[msk(m_p)]),torch.tensor(yseq[msk(m_p)]),
                          torch.tensor(Xseq[msk(m_pv)]),torch.tensor(yseq[msk(m_pv)]),LR,MAX_EP1,PAT1)
        model=train_phase(model,torch.tensor(Xseq[msk(m_ft)]),torch.tensor(yseq[msk(m_ft)]),
                          torch.tensor(Xseq[msk(m_fv)]),torch.tensor(yseq[msk(m_fv)]),LR_FT,MAX_EP2,PAT2)
        te=msk(m_te)
        model.eval()
        with torch.no_grad(): pr_tft=inv(model(torch.tensor(Xseq[te]).to(dev)).cpu().numpy())
        dte=dts[te]
        o=obs_real.reindex(dte);valid=o.notna()
        m_tft=full_metrics(o[valid].values,pr_tft[valid.values],thr)
        m_tft.update({"model":"TFT-transfer","target":target})
        rows.append(m_tft)
        log.info(f"[TFT-transfer {target}] {time.time()-t:.0f}s NSE={m_tft['NSE']} "
                 f"logNSE={m_tft['logNSE']} PBIAS={m_tft['PBIAS']} POD={m_tft['POD']} N={m_tft['N']}")

        # ── LightGBM AR (mismo test 2024-2025; train GR4J+Q real ≤2023) ──
        mtr_lgb=(dates<="2023-12-31")
        ytr=df[target].values
        trm=mtr_lgb & np.isfinite(ytr)
        lgb=LGBMRegressor(n_estimators=500,num_leaves=31,learning_rate=0.03,subsample=0.8,
                          colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbosity=-1)
        lgb.fit(df.loc[trm,feat].fillna(0),ytr[trm])
        prl=pd.Series(lgb.predict(df.loc[m_te,feat].fillna(0)),index=dates[m_te])
        o2=obs_real.reindex(dates[m_te]);v2=o2.notna()
        m_lgb=full_metrics(o2[v2].values,prl[v2].values,thr)
        m_lgb.update({"model":"LightGBM-AR","target":target})
        rows.append(m_lgb)
        log.info(f"[LightGBM-AR {target}]  NSE={m_lgb['NSE']} "
                 f"logNSE={m_lgb['logNSE']} PBIAS={m_lgb['PBIAS']} POD={m_lgb['POD']} N={m_lgb['N']}")

        # desglose por temporada y ENSO (solo q_next_1d, modelo TFT)
        if target=="q_next_1d":
            sub=pd.DataFrame({"obs":o[valid].values,"pred":pr_tft[valid.values]},index=dte[valid.values])
            wet=sub.index.month.isin([11,12,1,2,3,4])
            log.info(f"  TFT por temporada: húmeda NSE={_nse(sub[wet]['obs'].values,sub[wet]['pred'].values):.3f} "
                     f"| seca NSE={_nse(sub[~wet]['obs'].values,sub[~wet]['pred'].values):.3f}")
            oni_d=oni.reindex(sub.index,method="ffill")
            for lbl,mk in [("Niño",oni_d>0.5),("Niña",oni_d<-0.5),("Neutral",(oni_d>=-0.5)&(oni_d<=0.5))]:
                mk=mk.values
                if mk.sum()>10:
                    log.info(f"  TFT ENSO {lbl}: NSE={_nse(sub[mk]['obs'].values,sub[mk]['pred'].values):.3f} (n={mk.sum()})")
            for d,oo,pp in zip(dte[valid.values],o[valid].values,pr_tft[valid.values]):
                preds_all.append({"date":d,"q_obs":oo/Q_CONV,"q_tft":pp/Q_CONV})

    pd.DataFrame(rows).to_csv(OUT_DIR/"transfer_metrics_full.csv",index=False)
    pd.DataFrame(preds_all).to_csv(OUT_DIR/"transfer_predictions.csv",index=False)
    log.info("\n=== MÉTRICAS COMPLETAS (test 2024-2025) ===")
    cols=["model","target","NSE","NSE_sqrt","logNSE","KGE","PBIAS","PBIAS_Q90","POD","FAR","CSI","HSS","N"]
    log.info(pd.DataFrame(rows)[cols].to_string(index=False))
    log.info("SCRIPT 72 COMPLETADO")

if __name__=="__main__":
    main()
