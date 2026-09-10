#!/usr/bin/env python3
"""
Script 108 — FAIR probabilistic comparison: RA-TFT vs LightGBM (equal info).

Gives the architecture its fair shot on the axis where a TFT has a STRUCTURAL
advantage: coherent multi-horizon PROBABILISTIC forecasts. Both models get the
SAME realistic (noisy) future rainfall. We compare, per horizon:
  - CRPS (approx via mean pinball over quantiles 0.1/0.5/0.9) — lower is better
  - PICP (coverage of the 80% [P10,P90] band, target 0.80)
  - PINAW (band width, informativeness)

If RA-TFT wins on CRPS/calibration, that is a legitimate architectural contribution
(coherent probabilistic multi-horizon), independent of point-skill.

Run in .venv313:
    python scripts/05_models/../06_eval/108_probabilistic_fair.py --seeds 2
"""
import argparse, importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("prob")

R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py"))
importlib.util.spec_from_file_location("r105",ROOT/"scripts/05_models/105_ratft.py").loader.exec_module(R)
DEV=R.DEV; ENC=R.ENC; PAST=R.PAST; FUT=R.FUT; Q90=R.Q90

QS=[0.1,0.5,0.9]
def crps_approx(o, qp):   # qp: (N,3) quantile preds for QS ; mean pinball = quantile score
    tot=0
    for i,q in enumerate(QS):
        e=o-qp[:,i]; tot+=np.mean(np.where(e>=0,q*e,(q-1)*e))
    return tot/len(QS)
def picp(o,lo,hi): return float(np.mean((o>=lo)&(o<=hi)))
def pinaw(o,lo,hi): return float(np.mean(hi-lo)/(o.max()-o.min()+1e-9))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=2); a=ap.parse_args()
    H=7; NOISE=0.7
    df=R.build(H); dts=df.index; obs=df["obs"].values
    tr_idx=[i for i in range(ENC,len(df)-H) if dts[i]<=pd.Timestamp("2023-12-31") and np.isfinite(df["q"].values[i-ENC:i+H]).all()]
    te_idx=[i for i in range(ENC,len(df)-H) if dts[i]>=pd.Timestamp("2024-01-01")]
    mu={"p":df[PAST].iloc[tr_idx].values.mean(0),"f":df[FUT].iloc[tr_idx].values.mean(0)}
    sd={"p":df[PAST].iloc[tr_idx].values.std(0)+1e-6,"f":df[FUT].iloc[tr_idx].values.std(0)+1e-6}

    HS=[1,3,7]
    # ── RA-TFT (multi-horizon, quantile) ──────────────────────────────────────
    ratft_pred={h:[] for h in HS}
    for s in range(a.seeds):
        torch.manual_seed(s); np.random.seed(s)
        Xp,Qp,Xf,Y=R.make_seqs(df,tr_idx,H,(mu,sd),rain_noise=NOISE,rng=np.random.default_rng(s))
        dl=DataLoader(TensorDataset(Xp,Qp,Xf,Y),batch_size=128,shuffle=True)
        m=R.RATFT(len(PAST),len(FUT),H=H,use_future=True).to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); m.train()
        for ep in range(50):
            for xb,qb,fb,yb in dl:
                opt.zero_grad(); R.pinball(m(xb.to(DEV),qb.to(DEV),fb.to(DEV)),yb.to(DEV)).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        Xpt,Qpt,Xft,_=R.make_seqs(df,te_idx,H,(mu,sd),rain_noise=NOISE,rng=np.random.default_rng(999))
        with torch.no_grad(): pr=m(Xpt.to(DEV),Qpt.to(DEV),Xft.to(DEV)).cpu().numpy()  # (N,H,3)
        for h in HS: ratft_pred[h].append(pr[:,h-1,:])   # (N,3)
    ratft_pred={h:np.mean(v,0) for h,v in ratft_pred.items()}

    # ── LightGBM quantile (per horizon, per quantile) with same realistic rain ─
    # future-rain feature per horizon = cumulative future rain (noisy)
    rng=np.random.default_rng(7)
    lgb_pred={}
    for h in HS:
        d=df.copy()
        fut_rain=d["pr"].shift(-h).rolling(h,min_periods=1).sum().values
        fut_rain=fut_rain*np.exp(rng.normal(0,NOISE,size=fut_rain.shape))   # realistic noise
        d["fr"]=fut_rain; d["yt"]=d["q"].shift(-h); d["yo"]=d["obs"].shift(-h)
        FE=["q","pr","api","spi90","swvl4","oni","coastal","sin","cos","fr"]
        tr=d[d.index<=pd.Timestamp("2023-12-31")].dropna(subset=FE+["yt"])
        te=d[(d.index>=pd.Timestamp("2024-01-01"))].dropna(subset=FE+["yo"])
        preds=[]
        for q in QS:
            g=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=400,learning_rate=0.03,num_leaves=31,
                min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
            g.fit(tr[FE],np.log1p(tr["yt"].clip(lower=0))); preds.append(np.expm1(g.predict(te[FE])))
        P=np.stack(preds,1); P=np.sort(P,1)  # enforce monotone
        lgb_pred[h]=(P, te["yo"].values)

    # ── Compare ───────────────────────────────────────────────────────────────
    log.info("\n=== FAIR PROBABILISTIC (realistic rain, both) — CRPS↓, PICP→0.80 ===")
    rows=[]
    for h in HS:
        o_ra=np.array([obs[i+h] for i in te_idx]); qp_ra=ratft_pred[h]
        mask=np.isfinite(o_ra); o_ra,qp_ra=o_ra[mask],qp_ra[mask]
        P_lg,o_lg=lgb_pred[h]; mlg=np.isfinite(o_lg); o_lg,P_lg=o_lg[mlg],P_lg[mlg]
        ra=dict(CRPS=crps_approx(o_ra,qp_ra),PICP=picp(o_ra,qp_ra[:,0],qp_ra[:,2]),PINAW=pinaw(o_ra,qp_ra[:,0],qp_ra[:,2]))
        lg_=dict(CRPS=crps_approx(o_lg,P_lg),PICP=picp(o_lg,P_lg[:,0],P_lg[:,2]),PINAW=pinaw(o_lg,P_lg[:,0],P_lg[:,2]))
        rows.append(dict(h=h,model="RA-TFT",**ra)); rows.append(dict(h=h,model="LightGBM",**lg_))
        log.info(f"h={h}d: RA-TFT CRPS={ra['CRPS']:.3f} PICP={ra['PICP']:.2f} | "
                 f"LightGBM CRPS={lg_['CRPS']:.3f} PICP={lg_['PICP']:.2f}  -> "
                 f"{'RA-TFT better' if ra['CRPS']<lg_['CRPS'] else 'LightGBM better'} (CRPS)")
    pd.DataFrame(rows).to_csv(OUTML/"108_probabilistic_fair.csv",index=False)
    log.info("Saved: 108_probabilistic_fair.csv")


if __name__=="__main__":
    main()
