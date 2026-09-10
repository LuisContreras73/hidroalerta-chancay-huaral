#!/usr/bin/env python3
"""
Script 114 — Búsqueda INTENSIVA de hiperparámetros del TFT honesto (sin leakage).

Amplía el espacio: ventana ENC {30,45,60,90}, capacidad hid {48..256}, heads, dropout,
lr, weight-decay, batch {64,128,256}. Modelos más grandes -> más uso de GPU por trial.
FUT = SOLO calendario futuro (sin meteorología futura -> sin data leakage).
Estudio Optuna RESUMIBLE por horizonte (SQLite), paralelizable (varios workers).
Eval final: h=1,3,7,14 vs LightGBM(sin futuro) y persistencia. CSV por horizonte.

Run in .venv313 (lanzar varios en paralelo con --search-only y --seed distinto):
    python scripts/05_models/114_tft_intensive.py --trials 80 --H 7
    python scripts/05_models/114_tft_intensive.py --trials 60 --H 7 --search-only --seed 1
"""
import argparse, importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("intv")
optuna.logging.set_verbosity(optuna.logging.WARNING)
R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py"))
importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py").loader.exec_module(R)
DEV=R.DEV; Q90=R.Q90; QS=[0.1,0.5,0.9]
R.FUT=["sin","cos"]   # honesto: solo calendario futuro

def crps(o,qp):
    t=0
    for i,q in enumerate(QS):
        e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)

def seqs_for_enc(df, idxs, enc, H, scaler):
    R.ENC=enc; return R.make_seqs(df, idxs, H, scaler)

def train_es(model, dl, Xv, lr, wd, max_ep=180, pat=25):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max_ep)
    best=np.inf; bs=None; bad=0
    for ep in range(max_ep):
        model.train()
        for xb,qb,fb,yb in dl:
            opt.zero_grad(); R.pinball(model(xb.to(DEV),qb.to(DEV),fb.to(DEV)),yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad(): vl=R.pinball(model(Xv[0].to(DEV),Xv[1].to(DEV),Xv[2].to(DEV)),Xv[3].to(DEV)).item()
        if vl<best-1e-4: best,bs,bad=vl,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else:
            bad+=1
            if bad>=pat: break
    if bs: model.load_state_dict(bs)
    return model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",type=int,default=80); ap.add_argument("--H",type=int,default=7)
    ap.add_argument("--search-only",action="store_true"); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args(); H=a.H
    df=R.build(H); dts=df.index; obs=df["obs"].values
    ENCMAX=90
    tr=[i for i in range(ENCMAX,len(df)-H) if dts[i]<=pd.Timestamp("2022-12-31") and np.isfinite(df["q"].values[i-ENCMAX:i+H]).all()]
    va=[i for i in range(ENCMAX,len(df)-H) if pd.Timestamp("2023-01-01")<=dts[i]<=pd.Timestamp("2023-12-31") and np.isfinite(df["q"].values[i-ENCMAX:i+H]).all()]
    te=[i for i in range(ENCMAX,len(df)-H) if dts[i]>=pd.Timestamp("2024-01-01")]
    scaler=({"p":df[R.PAST].iloc[tr].values.mean(0),"f":df[R.FUT].iloc[tr].values.mean(0)},
            {"p":df[R.PAST].iloc[tr].values.std(0)+1e-6,"f":df[R.FUT].iloc[tr].values.std(0)+1e-6})
    mu,sd=scaler; scl=(mu,sd)
    cache={}
    def get(enc):
        if enc not in cache:
            cache[enc]=(seqs_for_enc(df,tr,enc,H,scl), seqs_for_enc(df,va,enc,H,scl), seqs_for_enc(df,te,enc,H,scl))
        return cache[enc]

    def valcrps(m,Xva):
        m.eval()
        with torch.no_grad(): pr=m(Xva[0].to(DEV),Xva[1].to(DEV),Xva[2].to(DEV)).cpu().numpy()
        vals=[]
        for h in range(H):
            o=np.array([obs[i+h] for i in va]); mk=np.isfinite(o)&np.isfinite(pr[:,h,1])
            if mk.sum()>0: vals.append(crps(o[mk],pr[mk,h,:]))
        return float(np.mean(vals)) if vals and np.isfinite(np.mean(vals)) else float("inf")

    def objective(t):
        c=dict(enc=t.suggest_categorical("enc",[30,45,60,90]),
               hid=t.suggest_categorical("hid",[48,64,96,128,192,256]),
               heads=t.suggest_categorical("heads",[2,4,8]),
               drop=t.suggest_float("drop",0.05,0.5),
               lr=t.suggest_float("lr",5e-5,5e-3,log=True),
               wd=t.suggest_float("wd",1e-7,1e-2,log=True),
               bs=t.suggest_categorical("bs",[64,128,256]))
        if c["hid"]%c["heads"]!=0: raise optuna.TrialPruned()
        Xtr,Xva,_=get(c["enc"]); torch.manual_seed(0); np.random.seed(0)
        dl=DataLoader(TensorDataset(*Xtr),batch_size=c["bs"],shuffle=True)
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=c["hid"],heads=c["heads"],H=H,drop=c["drop"],use_future=True).to(DEV)
        m=train_es(m,dl,Xva,c["lr"],c["wd"]); return valcrps(m,Xva)

    dbname=f"114_intensive_h{H}.db"; storage=f"sqlite:///{OUTML/dbname}"
    st=optuna.create_study(direction="minimize",study_name=f"intv_h{H}",storage=storage,load_if_exists=True,sampler=TPESampler(seed=a.seed,n_startup_trials=12))
    log.info(f"H={H} seed={a.seed} search_only={a.search_only}. Trials: {len([t for t in st.trials if t.state.is_finished()])}; +{a.trials}")
    st.optimize(objective,n_trials=a.trials,callbacks=[lambda s,t: log.info(f"  #{t.number} valCRPS={t.value:.3f} best={s.best_value:.3f} {t.params.get('enc')},{t.params.get('hid')}") if t.value else None])
    if a.search_only: log.info(f"[search-only] best={st.best_value:.3f} {st.best_params}"); return
    bp=st.best_params; log.info(f"BEST valCRPS={st.best_value:.3f} {bp}")

    # ── Eval final h=1,3,7,14 vs LightGBM + persistencia ──────────────────────
    import lightgbm as lgb
    def metr(o,qp):
        o=np.asarray(o,float); p=qp[:,1]; nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
        so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None)); nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
        oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
        csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
        return dict(CRPS=round(crps(o,qp),3),NSE=round(nse,3),J=round(0.25*nsq+0.25*nse+0.30*csi+0.10*pod-0.10*far,3))
    HS=[h for h in [1,3,7,14] if h<=H]
    Xtr,Xva,Xte=get(bp["enc"]); ra={h:[] for h in HS}
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        dl=DataLoader(TensorDataset(*Xtr),batch_size=bp["bs"],shuffle=True)
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=bp["hid"],heads=bp["heads"],H=H,drop=bp["drop"],use_future=True).to(DEV)
        m=train_es(m,dl,Xva,bp["lr"],bp["wd"])
        with torch.no_grad(): pr=m(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy()
        for h in HS: ra[h].append(pr[:,h-1,:])
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]; rows=[]
    for h in HS:
        o=np.array([obs[i+h] for i in te]); mk=np.isfinite(o); mra=metr(o[mk],np.mean(ra[h],0)[mk])
        d=df.copy(); d["yt"]=d["q"].shift(-h); d["yo"]=d["obs"].shift(-h)
        trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2024-01-01")].dropna(subset=FE+["yo"])
        P=[]
        for q in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); P.append(np.expm1(g.predict(ted[FE])))
        mlg=metr(ted["yo"].values,np.sort(np.stack(P,1),1))
        op=np.array([obs[i+h-1] for i in te]); pp=np.array([df["q"].values[i-1] for i in te]); m2=np.isfinite(op); mpe=metr(op[m2],np.stack([pp[m2]]*3,1))
        rows.append(dict(h=h,model="TFT-intensivo",**mra)); rows.append(dict(h=h,model="LightGBM",**mlg)); rows.append(dict(h=h,model="Persistence",**mpe))
        log.info(f"h={h}d: TFT NSE={mra['NSE']} CRPS={mra['CRPS']} | LGBM NSE={mlg['NSE']} CRPS={mlg['CRPS']} | Persist NSE={mpe['NSE']}")
    pd.DataFrame(rows).to_csv(OUTML/f"114_intensive_h{H}_vs.csv",index=False); log.info(f"Saved: 114_intensive_h{H}_vs.csv")

if __name__=="__main__":
    main()
