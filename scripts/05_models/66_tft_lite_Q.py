#!/usr/bin/env python3
"""
Script 66: TFT-lite (LSTM encoder + multi-head self-attention) — target Q, 1d y 7d.

Transformer-lite en PyTorch puro (sin neuralforecast/lightning, que tienen conflicto
de versiones). Captura lo esencial del TFT: atención temporal interpretable sobre la
ventana del encoder. Con Q autoregresivo (gap-fill GR4J corregido).

Compara contra LightGBM AR (1d NSE=0.95, 7d NSE=0.83).

Salida:
  outputs/ml_Q/tft_lite_results.csv
  outputs/ml_Q/tft_lite_predictions.csv
  outputs/figures/ml_Q/TFTL01_attention.png
"""
import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log=logging.getLogger("tft_lite")

D6_CSV=ROOT/"data/model_ready/D6_multientity.csv"
QOBS=ROOT/"data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR=ROOT/"outputs/ml_Q"; FIG_DIR=ROOT/"outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
Q_CONV=86.4/3062.62; Q90=40.89; SEED=42
# Tuneado 2026-06-03 vs sobreajuste: encoder corto, menos capacidad, +VSN +pos-encoding
ENC=45; HID=32; HEADS=2; DROP=0.3; LR=7e-4; WD=3e-5; BATCH=64; MAX_EP=300; PAT=40

SPATIAL=["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED=["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]

# Reducción de features espaciales: basin-mean + 3 sub-cuencas cabecera (79% del caudal)
KEY_SUBS=["sub_649","sub_655","sub_656"]   # cabecera alta (mayor aporte)

def build_wide(D6):
    cols={}
    for c in SPATIAL:
        piv=D6.pivot_table(index="date",columns="entity_id",values=c)
        cols[f"{c}_basin"]=piv.mean(axis=1)          # media de la cuenca
        for e in KEY_SUBS:                            # solo cabecera (no las 9)
            if e in piv.columns: cols[f"{c}_{e}"]=piv[e]
    df=pd.DataFrame(cols)
    ref=D6[D6["entity_id"]=="sub_634"].set_index("date")
    df=df.join(ref[SHARED+["q_mm","q_next_1d","q_sum_next_7d"]]).sort_index()
    q=df["q_mm"]
    df["q_lag7"]=q.shift(7); df["q_roll7"]=q.rolling(7,min_periods=3).mean()
    df["q_roll30"]=q.rolling(30,min_periods=10).mean()
    return df

def make_seq(X,y,enc):
    Xs,ys,idx=[],[],[]
    for i in range(enc,len(X)):
        if np.isfinite(y[i]): Xs.append(X[i-enc:i]);ys.append(y[i]);idx.append(i)
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
    log.info(f"Script 66: TFT-lite (LSTM+attention) | device={dev}")

    D6=pd.read_csv(D6_CSV,parse_dates=["date"])
    qobs=pd.read_csv(QOBS,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm=qobs*Q_CONV
    df=build_wide(D6); dates=df.index
    feat=[c for c in df.columns if c not in ["q_next_1d","q_sum_next_7d"]]
    m_tr=dates<="2015-12-31"; m_te=dates>="2021-01-01"
    log.info(f"Features: {len(feat)} (con Q autoregresivo)")

    class VSN(nn.Module):
        """Variable Selection Network: aprende pesos por feature (softmax) y los aplica.
        Clave del TFT: filtra las 72 features espaciales redundantes."""
        def __init__(s,nf,hid):
            super().__init__()
            s.w=nn.Sequential(nn.Linear(nf,hid),nn.ReLU(),nn.Linear(hid,nf))
            s.last_w=None
        def forward(s,x):                       # x: [B,T,nf]
            wt=torch.softmax(s.w(x),dim=-1)      # pesos por feature, por timestep
            s.last_w=wt.detach()
            return x*wt*x.shape[-1]              # reescala (mantiene magnitud)

    class TFTLite(nn.Module):
        def __init__(s,nf):
            super().__init__()
            s.vsn=VSN(nf,HID)                    # selección de variables
            s.proj=nn.Linear(nf,HID)
            s.pos=nn.Parameter(torch.randn(1,ENC,HID)*0.02)  # positional encoding aprendido
            s.lstm=nn.LSTM(HID,HID,1,batch_first=True)
            s.attn=nn.MultiheadAttention(HID,HEADS,dropout=DROP,batch_first=True)
            s.norm=nn.LayerNorm(HID)
            s.drop=nn.Dropout(DROP)
            s.head=nn.Sequential(nn.Linear(HID,HID//2),nn.ReLU(),nn.Dropout(DROP),nn.Linear(HID//2,1))
            s.last_attn=None
        def forward(s,x):
            x=s.vsn(x)                           # filtrar features
            h=s.proj(x)+s.pos                    # proyección + posición
            o,_=s.lstm(h)
            a,w=s.attn(o,o,o,need_weights=True,average_attn_weights=True)
            s.last_attn=w.detach()
            h=s.norm(o+s.drop(a))                # residual + norm (estilo transformer)
            return s.head(h[:,-1,:]).squeeze(-1)

    lf=nn.HuberLoss(delta=1.0)   # robusto a picos (vs MSE)
    targets={"q_next_1d":qobs_mm.shift(-1),
             "q_sum_next_7d":qobs_mm.shift(-1).rolling(7).sum().shift(-6)}
    results=[]; preds={}
    for target, obs_real in targets.items():
        y=df[target].values.astype(np.float32)
        X=df[feat].fillna(0).values.astype(np.float32)
        mu,sd=X[m_tr].mean(0),X[m_tr].std(0)+1e-8; X=(X-mu)/sd
        ylog=np.log1p(np.clip(y,0,None));ymu,ysd=np.nanmean(ylog[m_tr]),np.nanstd(ylog[m_tr])+1e-8
        ysc=(ylog-ymu)/ysd; inv=lambda v:np.expm1(v*ysd+ymu)
        Xseq,yseq,idx=make_seq(X,ysc,ENC)
        sp=np.array(["train" if m_tr[i] else ("test" if m_te[i] else "val") for i in idx])
        tr,vl,te=sp=="train",sp=="val",sp=="test"
        Xt=torch.tensor(Xseq[tr]);yt=torch.tensor(yseq[tr])
        Xv=torch.tensor(Xseq[vl]);yv=torch.tensor(yseq[vl]);Xte=torch.tensor(Xseq[te])
        model=TFTLite(len(feat)).to(dev)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",factor=0.5,patience=8)
        dl=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH,shuffle=True)
        Xvd,yvn=Xv.to(dev),yv.numpy();best=np.inf;bs=None;c=0;t0=time.time()
        for ep in range(MAX_EP):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad();lf(model(xb),yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)   # gradient clipping
                opt.step()
            model.eval()
            with torch.no_grad(): vl_=float(np.mean((model(Xvd).cpu().numpy()-yvn)**2))
            sched.step(vl_)
            if vl_<best:best=vl_;bs={k:v.cpu().clone() for k,v in model.state_dict().items()};c=0
            else:
                c+=1
                if c>=PAT: break
        model.load_state_dict(bs);model.eval()
        with torch.no_grad(): pr=inv(model(Xte.to(dev)).cpu().numpy())
        dts=dates[idx][te]
        o=obs_real.reindex(dts); s=pd.Series(pr,index=dts); valid=o.notna()
        thr=Q90*Q_CONV*(7 if "7d" in target else 1)
        m=metr(o[valid].values,s[valid].values,thr); m["target"]=target; results.append(m)
        log.info(f"[{target}] {ep+1} epochs {time.time()-t0:.0f}s | "
                 f"NSE={m['NSE']} KGE={m['KGE']} J_alert={m['J_alert']} POD={m['POD']} N={m['N']}")
        div=Q_CONV*(7 if "7d" in target else 1)
        preds[target]=pd.DataFrame({"date":dts[valid.values],"q_obs":o[valid].values/div,
                                    "q_pred":s[valid].values/div})
        # guardar atención del último batch de test (promedio sobre muestras)
        if target=="q_next_1d":
            with torch.no_grad(): model(Xte[:256].to(dev))
            attn=model.last_attn.cpu().numpy().mean(0)  # [enc, enc]
            np.save(OUT_DIR/"tft_lite_attention.npy", attn)

    pd.DataFrame(results).to_csv(OUT_DIR/"tft_lite_results.csv",index=False)
    log.info("\n=== TFT-lite vs LightGBM AR ===")
    lgb=pd.read_csv(OUT_DIR/"lgbm_ar_results.csv") if (OUT_DIR/"lgbm_ar_results.csv").exists() else None
    for r in results:
        lgb_nse = lgb[lgb["target"]==r["target"]]["NSE"].values[0] if lgb is not None else np.nan
        log.info(f"  {r['target']}: TFT-lite NSE={r['NSE']} | LightGBM AR NSE={lgb_nse}")
    log.info("SCRIPT 66 COMPLETADO")

if __name__=="__main__":
    main()
