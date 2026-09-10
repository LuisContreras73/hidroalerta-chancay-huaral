#!/usr/bin/env python3
"""
Script 113 — TFT HONESTO (sin forzante meteorológico futuro) + grid search Optuna.

Decisión: se ELIMINA la lluvia futura observada (era una cota superior optimista, no
operacional). El TFT honesto solo usa info disponible al pronosticar:
  - Encoder: ventana PASADA de forzantes (precip, PET, API, SPI, humedad suelo, ONI,
    costero) + caudal (RevIN).
  - Decoder: SOLO calendario futuro (sin/cos del día del año -> deterministas, conocidos).
  - Sin ninguna variable meteorológica futura -> sin leakage, operacional.

Grid search Optuna RESUMIBLE (SQLite), tuneado en VAL 2023, TEST 2024 sellado. Multi-horizonte
cuantílico. Compara vs LightGBM (mismos features PASADOS, sin futuro) y persistencia.

Run in .venv313:
    python scripts/05_models/113_tft_honest_tune.py --trials 60 --H 7
"""
import argparse, importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("tfth")
optuna.logging.set_verbosity(optuna.logging.WARNING)

R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/113_src.py") if False else importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py"))
importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py").loader.exec_module(R)
DEV=R.DEV; ENC=R.ENC; Q90=R.Q90; QS=[0.1,0.5,0.9]
R.FUT=["sin","cos"]   # <-- SOLO calendario futuro (conocido). SIN lluvia/PET futura.

def crps(o,qp):
    t=0
    for i,q in enumerate(QS):
        e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)

def train_es(model, dl, Xv,Qv,Fv,Yv, lr, wd, max_ep=150, pat=20):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max_ep)
    best=np.inf; bs=None; bad=0
    for ep in range(max_ep):
        model.train()
        for xb,qb,fb,yb in dl:
            opt.zero_grad(); R.pinball(model(xb.to(DEV),qb.to(DEV),fb.to(DEV)),yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad(): vl=R.pinball(model(Xv.to(DEV),Qv.to(DEV),Fv.to(DEV)),Yv.to(DEV)).item()
        if vl<best-1e-4: best,bs,bad=vl,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else:
            bad+=1
            if bad>=pat: break
    if bs: model.load_state_dict(bs)
    return model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",type=int,default=60); ap.add_argument("--H",type=int,default=7)
    ap.add_argument("--search-only",action="store_true",help="solo añade trials al estudio (worker paralelo); sin eval final")
    ap.add_argument("--seed",type=int,default=42,help="seed del sampler (distinto por worker)"); a=ap.parse_args(); H=a.H
    df=R.build(H); dts=df.index; obs=df["obs"].values
    tr=[i for i in range(ENC,len(df)-H) if dts[i]<=pd.Timestamp("2022-12-31") and np.isfinite(df["q"].values[i-ENC:i+H]).all()]
    va=[i for i in range(ENC,len(df)-H) if pd.Timestamp("2023-01-01")<=dts[i]<=pd.Timestamp("2023-12-31") and np.isfinite(df["q"].values[i-ENC:i+H]).all()]
    te=[i for i in range(ENC,len(df)-H) if dts[i]>=pd.Timestamp("2024-01-01")]
    mu={"p":df[R.PAST].iloc[tr].values.mean(0),"f":df[R.FUT].iloc[tr].values.mean(0)}
    sd={"p":df[R.PAST].iloc[tr].values.std(0)+1e-6,"f":df[R.FUT].iloc[tr].values.std(0)+1e-6}
    Xtr=R.make_seqs(df,tr,H,(mu,sd)); Xva=R.make_seqs(df,va,H,(mu,sd)); Xte=R.make_seqs(df,te,H,(mu,sd))
    dl=DataLoader(TensorDataset(*Xtr),batch_size=128,shuffle=True)

    def valcrps(m):
        m.eval()
        with torch.no_grad(): pr=m(Xva[0].to(DEV),Xva[1].to(DEV),Xva[2].to(DEV)).cpu().numpy()
        vals=[]
        for h in range(H):
            o=np.array([obs[i+h] for i in va]); mk=np.isfinite(o)&np.isfinite(pr[:,h,1])
            if mk.sum()>0: vals.append(crps(o[mk],pr[mk,h,:]))
        return float(np.mean(vals)) if vals and np.isfinite(np.mean(vals)) else float("inf")

    def objective(t):
        c=dict(hid=t.suggest_categorical("hid",[32,48,64,96,128]),heads=t.suggest_categorical("heads",[2,4,8]),
               drop=t.suggest_float("drop",0.05,0.4),lr=t.suggest_float("lr",1e-4,3e-3,log=True),wd=t.suggest_float("wd",1e-6,1e-3,log=True))
        if c["hid"]%c["heads"]!=0: raise optuna.TrialPruned()
        torch.manual_seed(0); np.random.seed(0)
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=c["hid"],heads=c["heads"],H=H,drop=c["drop"],use_future=True).to(DEV)
        m=train_es(m,dl,*Xva,c["lr"],c["wd"]); return valcrps(m)

    # estudio separado por horizonte (evita mezclar trials H=7 vs H=14 y contención SQLite)
    dbname = "113_tft_honest_study.db" if H==7 else f"113_tft_honest_h{H}_study.db"
    stname = "tft_honest" if H==7 else f"tft_honest_h{H}"
    storage=f"sqlite:///{OUTML/dbname}"
    st=optuna.create_study(direction="minimize",study_name=stname,storage=storage,load_if_exists=True,sampler=TPESampler(seed=a.seed,n_startup_trials=8))
    log.info(f"Worker seed={a.seed} search_only={a.search_only}. Trials en estudio: {len([t for t in st.trials if t.state.is_finished()])}; añadiendo {a.trials}.")
    st.optimize(objective,n_trials=a.trials,callbacks=[lambda s,t: log.info(f"  trial {t.number} valCRPS={t.value:.3f} best={s.best_value:.3f}") if t.value else None])
    if a.search_only:
        log.info(f"[search-only] listo. Best actual valCRPS={st.best_value:.3f} {st.best_params}"); return
    bp=st.best_params; log.info(f"BEST valCRPS={st.best_value:.3f} {bp}")

    # ── Final: 3 seeds test sellado vs LightGBM(sin futuro)+persistencia ───────
    import lightgbm as lgb
    def metr(o,qp):
        o=np.asarray(o,float); p=qp[:,1]; nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
        so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None)); nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
        oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
        csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
        return dict(CRPS=crps(o,qp),NSE=nse,J=0.25*nsq+0.25*nse+0.30*csi+0.10*pod-0.10*far)
    ra={h:[] for h in [1,3,7]}
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        Xtr_s=R.make_seqs(df,tr,H,(mu,sd)); dls=DataLoader(TensorDataset(*Xtr_s),batch_size=128,shuffle=True)
        m=R.RATFT(len(R.PAST),len(R.FUT),hid=bp["hid"],heads=bp["heads"],H=H,drop=bp["drop"],use_future=True).to(DEV)
        m=train_es(m,dls,*Xva,bp["lr"],bp["wd"])
        with torch.no_grad(): pr=m(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy()
        for h in [1,3,7]: ra[h].append(pr[:,h-1,:])
    rows=[]
    FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos"]   # LightGBM: SIN lluvia futura
    for h in [1,3,7]:
        o=np.array([obs[i+h] for i in te]); msk=np.isfinite(o); qp=np.mean(ra[h],0)[msk]; oo=o[msk]; mra=metr(oo,qp)
        d=df.copy(); d["yt"]=d["q"].shift(-h); d["yo"]=d["obs"].shift(-h)
        trd=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); ted=d[d.index>=pd.Timestamp("2024-01-01")].dropna(subset=FE+["yo"])
        P=[]
        for q in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(trd[FE],np.log1p(trd["yt"].clip(lower=0))); P.append(np.expm1(g.predict(ted[FE])))
        mlg=metr(ted["yo"].values,np.sort(np.stack(P,1),1))
        op=np.array([obs[i+h-1] for i in te]); pp=np.array([df["q"].values[i-1] for i in te]); mk2=np.isfinite(op)
        mpe=metr(op[mk2],np.stack([pp[mk2]]*3,1))
        rows.append(dict(h=h,model="TFT-honest",**mra)); rows.append(dict(h=h,model="LightGBM",**mlg)); rows.append(dict(h=h,model="Persistence",**mpe))
        log.info(f"h={h}d: TFT NSE={mra['NSE']:.3f} CRPS={mra['CRPS']:.3f} | LGBM NSE={mlg['NSE']:.3f} CRPS={mlg['CRPS']:.3f} | Persist NSE={mpe['NSE']:.3f}")
    pd.DataFrame(rows).to_csv(OUTML/"113_tft_honest_vs.csv",index=False); log.info("Saved: 113_tft_honest_vs.csv")

if __name__=="__main__":
    main()
