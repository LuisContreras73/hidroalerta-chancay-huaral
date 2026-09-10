#!/usr/bin/env python3
"""
Script 102 — TFT (flagship) at multiple horizons vs persistence.

Complements the LightGBM multi-horizon curve (Script 100) with OUR flagship TFT,
retrained per horizon (target = outlet Q at t+h). Does not overwrite 74/81.

Run in .venv313:
    python scripts/05_models/102_tft_multi_horizon.py --seeds 2 --horizons 1,3,7
"""

import argparse, importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUTML = ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tft_mh")

spec = importlib.util.spec_from_file_location("tune81", ROOT/"scripts/05_models/81_tune_tft_optuna.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); T = S.T
DEV=S.DEVICE; Q_CONV=S.Q_CONV; Q90=S.Q90


def prepare_h(h):
    """Like S.prepare_data but target = outlet Q at t+h (mm)."""
    D6 = pd.read_csv(T.D6_CSV, parse_dates=["date"]); df = T.build_wide(D6)
    dates = df.index
    feat = [c for c in df.columns if c not in ["q_next_1d","q_sum_next_7d"]]
    m_p  = dates<="2017-12-31"; m_ft=(dates>="2021-01-01")&(dates<="2022-12-31")
    m_val=(dates>="2023-01-01")&(dates<="2023-12-31"); m_te=dates>="2024-01-01"
    X = df[feat].fillna(0).values.astype(np.float32)
    mu,sd = X[m_p].mean(0), X[m_p].std(0)+1e-8; Xs=(X-mu)/sd
    y = df["q_mm"].shift(-h).values.astype(np.float32)     # outlet Q at t+h (mm)
    ylog=np.log1p(np.clip(y,0,None)); ymu=np.nanmean(ylog[m_p]); ysd=np.nanstd(ylog[m_p])+1e-8
    ysc=((ylog-ymu)/ysd).astype(np.float32); inv=lambda v: np.expm1(np.asarray(v)*ysd+ymu)
    qobs = pd.read_csv(T.QOBS,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"].dropna()*Q_CONV
    obs_real = qobs.shift(-h).reindex(dates).values
    return dict(df=df, dates=dates, Xs=Xs, ysc=ysc, inv=inv, nf=len(feat),
                m_p=m_p, m_ft=m_ft, m_val=m_val, m_te=m_te, obs_real_mm=obs_real, q_today=df["q_mm"].values)


def metrics(o,p):
    o,p=np.asarray(o,float),np.asarray(p,float)
    nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None))
    nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
    oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    return dict(NSE=nse,J=0.25*nsq+0.25*nse+0.30*csi+0.10*pod-0.10*far,CSI=csi,POD=pod,FAR=far)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=2)
    ap.add_argument("--horizons",default="1,3,7"); a=ap.parse_args()
    HS=[int(x) for x in a.horizons.split(",")]
    bp=pd.read_csv(OUTML/"81_best_params.csv",index_col=0)["0"].to_dict()
    cfg=dict(ENC=int(bp["ENC"]),HID=int(bp["HID"]),HEADS=int(bp["HEADS"]),DROP=float(bp["DROP"]),
             LR=float(bp["LR"]),LR_FT=float(bp["LR_FT"]),WD=float(bp["WD"]),
             ALERT_W=float(bp["ALERT_W"]),mix_alpha=float(bp["mix_alpha"]))
    rows=[]
    for h in HS:
        data=prepare_h(h)
        T.ENC,T.HID,T.HEADS,T.DROP,T.WD=cfg["ENC"],cfg["HID"],cfg["HEADS"],cfg["DROP"],cfg["WD"]
        # persistence for horizon h on test real obs
        Xseq,y,w,idx=T.make_seqs(data["Xs"],data["ysc"],data["m_te"])
        obs_mm=data["obs_real_mm"][idx]; real=np.isfinite(obs_mm)
        q_today=data["q_today"][idx]/1.0                       # mm today (feature space) -> to m3/s below
        obs=obs_mm[real]/Q_CONV; pers=(data["q_today"][idx][real])/Q_CONV
        mp=metrics(obs,pers); rows.append(dict(h=h,model="Persistence",seed=-1,**mp))
        for s in range(a.seeds):
            torch.manual_seed(42+s); np.random.seed(42+s)
            Xp,yp,wp,_=T.make_seqs(data["Xs"],data["ysc"],data["m_p"])
            y_mm=data["df"]["q_mm"].shift(-h).values.astype(np.float32); thr=Q90*Q_CONV
            Xft,yft,wft,_=T.make_seqs(data["Xs"],data["ysc"],data["m_ft"],y_raw=y_mm,alert_thr=thr,alert_w=cfg["ALERT_W"])
            k=int(len(Xft)*0.8)
            model=T.TFTLitev2(data["nf"]).to(DEV)
            model=S.train(model,Xp,yp,wp,Xp[-200:],yp[-200:],cfg["LR"],80,18,cfg["mix_alpha"])
            model=S.train(model,Xft[:k],yft[:k],wft[:k],Xft[k:],yft[k:],cfg["LR_FT"],70,15,cfg["mix_alpha"])
            with torch.no_grad():
                p50=data["inv"](model(Xseq.to(DEV)).cpu().numpy()[:,1])/Q_CONV
            mm=metrics(obs,p50[real]); rows.append(dict(h=h,model="TFT",seed=s,**mm))
        tft=[r for r in rows if r["h"]==h and r["model"]=="TFT"]
        log.info(f"h={h}d: TFT NSE={np.mean([r['NSE'] for r in tft]):.3f} J={np.mean([r['J'] for r in tft]):.3f} "
                 f"| Persist NSE={mp['NSE']:.3f} J={mp['J']:.3f}")
    pd.DataFrame(rows).to_csv(OUTML/"102_tft_multi_horizon.csv",index=False)
    log.info("Saved: 102_tft_multi_horizon.csv")


if __name__=="__main__":
    main()
