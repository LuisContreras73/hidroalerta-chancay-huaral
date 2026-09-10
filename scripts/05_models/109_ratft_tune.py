#!/usr/bin/env python3
"""
Script 109 — Optuna hyperparameter tuning of RA-TFT (fair, thorough).

Gives the architecture its best legitimate shot before any verdict: more epochs
(early stopping to 150), LR + cosine scheduler, and Optuna search over hid, heads,
dropout, LR, weight-decay, decoder depth. Tuned on VALIDATION 2023; TEST 2024 stays
SEALED. Final best config re-evaluated (3 seeds) vs LightGBM under the SAME realistic
rain (fair), reporting CRPS / NSE / J_alert per horizon.

Run in .venv313:
    python scripts/05_models/109_ratft_tune.py --trials 25 --H 7
"""
import argparse, importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("ratft_tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)

R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py"))
importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py").loader.exec_module(R)
DEV=R.DEV; ENC=R.ENC; PAST=R.PAST; FUT=R.FUT; Q90=R.Q90
QS=[0.1,0.5,0.9]; NOISE=0.7


def crps(o,qp):
    t=0
    for i,q in enumerate(QS):
        e=o-qp[:,i]; t+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return t/len(QS)

def train_es(model, dl, Xv,Qv,Fv,Yv, lr, max_ep=150, pat=20):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=model._wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max_ep)
    best=np.inf; best_state=None; bad=0
    for ep in range(max_ep):
        model.train()
        for xb,qb,fb,yb in dl:
            opt.zero_grad(); R.pinball(model(xb.to(DEV),qb.to(DEV),fb.to(DEV)),yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl=R.pinball(model(Xv.to(DEV),Qv.to(DEV),Fv.to(DEV)),Yv.to(DEV)).item()
        if vl<best-1e-4: best=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=pat: break
    if best_state: model.load_state_dict(best_state)
    return model

def build_model(cfg, nf_p, nf_f, H):
    m=R.RATFT(nf_p, nf_f, hid=cfg["hid"], heads=cfg["heads"], H=H, drop=cfg["drop"], use_future=True).to(DEV)
    m._wd=cfg["wd"]; return m


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",type=int,default=25); ap.add_argument("--H",type=int,default=7)
    a=ap.parse_args(); H=a.H
    df=R.build(H); dts=df.index; obs=df["obs"].values
    tr_idx=[i for i in range(ENC,len(df)-H) if dts[i]<=pd.Timestamp("2022-12-31") and np.isfinite(df["q"].values[i-ENC:i+H]).all()]
    va_idx=[i for i in range(ENC,len(df)-H) if pd.Timestamp("2023-01-01")<=dts[i]<=pd.Timestamp("2023-12-31") and np.isfinite(df["q"].values[i-ENC:i+H]).all()]
    te_idx=[i for i in range(ENC,len(df)-H) if dts[i]>=pd.Timestamp("2024-01-01")]
    mu={"p":df[PAST].iloc[tr_idx].values.mean(0),"f":df[FUT].iloc[tr_idx].values.mean(0)}
    sd={"p":df[PAST].iloc[tr_idx].values.std(0)+1e-6,"f":df[FUT].iloc[tr_idx].values.std(0)+1e-6}
    Xtr=R.make_seqs(df,tr_idx,H,(mu,sd),rain_noise=NOISE,rng=np.random.default_rng(0))
    Xva=R.make_seqs(df,va_idx,H,(mu,sd),rain_noise=NOISE,rng=np.random.default_rng(1))
    dl=DataLoader(TensorDataset(*Xtr),batch_size=128,shuffle=True)

    def val_crps(model):
        model.eval()
        with torch.no_grad(): pr=model(Xva[0].to(DEV),Xva[1].to(DEV),Xva[2].to(DEV)).cpu().numpy()  # (N,H,3)
        c=[]
        for hh in range(H):
            o=np.array([obs[i+hh] for i in va_idx]); m=np.isfinite(o)
            c.append(crps(o[m],pr[m,hh,:]))
        return float(np.mean(c))

    def objective(trial):
        cfg=dict(hid=trial.suggest_categorical("hid",[32,48,64,96,128]),
                 heads=trial.suggest_categorical("heads",[2,4,8]),
                 drop=trial.suggest_float("drop",0.05,0.4),
                 lr=trial.suggest_float("lr",1e-4,3e-3,log=True),
                 wd=trial.suggest_float("wd",1e-6,1e-3,log=True))
        if cfg["hid"]%cfg["heads"]!=0: raise optuna.TrialPruned()
        torch.manual_seed(0); np.random.seed(0)
        m=build_model(cfg,len(PAST),len(FUT),H)
        m=train_es(m,dl,*Xva,cfg["lr"])
        return val_crps(m)

    # Resumable study: persisted to SQLite -> if stopped, re-running continues where it left off.
    storage=f"sqlite:///{OUTML/'109_ratft_study.db'}"
    study=optuna.create_study(direction="minimize", study_name="ratft_crps",
                              storage=storage, load_if_exists=True,
                              sampler=TPESampler(seed=42,n_startup_trials=8))
    done=len([t for t in study.trials if t.state.is_finished()])
    log.info(f"Resuming study: {done} trials already done; adding up to {a.trials} more.")
    study.optimize(objective,n_trials=a.trials,
                   callbacks=[lambda st,tr: log.info(f"  trial {tr.number:2d} valCRPS={tr.value:.3f} best={st.best_value:.3f}") if tr.value else None])
    bp=study.best_params; log.info(f"\nBEST valCRPS={study.best_value:.3f}  params={bp}")
    cfg=dict(hid=bp["hid"],heads=bp["heads"],drop=bp["drop"],lr=bp["lr"],wd=bp["wd"])

    # ── Final: 3 seeds on SEALED test vs LightGBM (same realistic rain) ────────
    import lightgbm as lgb
    def metrics_full(o,qp):
        o=np.asarray(o,float); p=qp[:,1]
        nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
        so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None)); nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
        oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
        csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
        return dict(CRPS=crps(o,qp),NSE=nse,J=0.25*nsq+0.25*nse+0.30*csi+0.10*pod-0.10*far)
    Xte=R.make_seqs(df,te_idx,H,(mu,sd),rain_noise=NOISE,rng=np.random.default_rng(999))
    HS=[1,3,7]
    ra={h:[] for h in HS}
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        Xtr_s=R.make_seqs(df,tr_idx,H,(mu,sd),rain_noise=NOISE,rng=np.random.default_rng(s))
        dl_s=DataLoader(TensorDataset(*Xtr_s),batch_size=128,shuffle=True)
        m=build_model(cfg,len(PAST),len(FUT),H); m=train_es(m,dl_s,*Xva,cfg["lr"])
        with torch.no_grad(): pr=m(Xte[0].to(DEV),Xte[1].to(DEV),Xte[2].to(DEV)).cpu().numpy()
        for h in HS: ra[h].append(pr[:,h-1,:])
    rng=np.random.default_rng(7); rows=[]
    for h in HS:
        o=np.array([obs[i+h] for i in te_idx]); msk=np.isfinite(o)
        qp=np.mean(ra[h],0)[msk]; oo=o[msk]; m_ra=metrics_full(oo,qp)
        # LightGBM quantile same realistic rain
        d=df.copy(); fr=d["pr"].shift(-h).rolling(h,min_periods=1).sum().values*np.exp(rng.normal(0,NOISE,len(d)))
        d["fr"]=fr; d["yt"]=d["q"].shift(-h); d["yo"]=d["obs"].shift(-h)
        FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos","fr"]
        tr=d[d.index<=pd.Timestamp("2022-12-31")].dropna(subset=FE+["yt"]); te=d[d.index>=pd.Timestamp("2024-01-01")].dropna(subset=FE+["yo"])
        P=[]
        for q in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(tr[FE],np.log1p(tr["yt"].clip(lower=0))); P.append(np.expm1(g.predict(te[FE])))
        P=np.sort(np.stack(P,1),1); m_lg=metrics_full(te["yo"].values,P)
        rows.append(dict(h=h,model="RA-TFT (tuned)",**m_ra)); rows.append(dict(h=h,model="LightGBM",**m_lg))
        log.info(f"h={h}d: RA-TFT CRPS={m_ra['CRPS']:.3f} NSE={m_ra['NSE']:.3f} J={m_ra['J']:.3f} | "
                 f"LightGBM CRPS={m_lg['CRPS']:.3f} NSE={m_lg['NSE']:.3f} J={m_lg['J']:.3f} -> "
                 f"{'RA-TFT' if m_ra['CRPS']<m_lg['CRPS'] else 'LightGBM'} wins CRPS")
    pd.DataFrame(rows).to_csv(OUTML/"109_ratft_tuned_vs_lgbm.csv",index=False)
    log.info("Saved: 109_ratft_tuned_vs_lgbm.csv")


if __name__=="__main__":
    main()
