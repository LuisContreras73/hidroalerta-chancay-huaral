#!/usr/bin/env python3
"""
Script 103 — Leaderboard on the VALIDATION period (2023), 1-day-ahead.

Complements the sealed-test leaderboard (Script 88) with performance on the
validation year (2023, ~99% real obs, includes wet+dry). IMPORTANT honesty note:
the TFT-tuned hyperparameters were SELECTED on this validation set, so its val
score is optimistically biased (in-sample for model selection); persistence,
climatology, GR4J, LightGBM and HydroST are NOT tuned on val → fair held-out.

Run in .venv313 (retrains TFT; reads others):
    python scripts/06_eval/103_validation_leaderboard.py
"""

import importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
FIGDIR=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log=logging.getLogger("val_lb")
Q90=40.89; Q_CONV=86.4/3062.62; DAY=pd.Timedelta(days=1)
VAL0,VAL1=pd.Timestamp("2023-01-01"),pd.Timestamp("2023-12-31")

spec=importlib.util.spec_from_file_location("tune81",ROOT/"scripts/05_models/81_tune_tft_optuna.py")
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S); T=S.T; DEV=S.DEVICE


def metrics(o,p):
    o,p=np.asarray(o,float),np.asarray(p,float)
    nse=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    so,sp=np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(p,0,None))
    nsq=1-np.sum((so-sp)**2)/np.sum((so-so.mean())**2)
    oa,pa=o>=Q90,p>=Q90; tp,fp,fn=np.sum(oa&pa),np.sum(~oa&pa),np.sum(oa&~pa)
    csi=tp/(tp+fp+fn) if(tp+fp+fn)else 0; pod=tp/(tp+fn) if(tp+fn)else 0; far=fp/(tp+fp) if(tp+fp)else 0
    return dict(NSE=round(nse,3),NSE_sqrt=round(nsq,3),J_alert=round(0.25*nsq+0.25*nse+0.30*csi+0.10*pod-0.10*far,3),
                CSI=round(csi,3),POD=round(pod,3),FAR=round(far,3))


def main():
    obs=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    obs=obs[~obs.index.duplicated()]
    # target date convention: forecast for day d, obs on d
    val_dates=pd.date_range(VAL0,VAL1,freq="D")

    preds={}
    # persistence: forecast for d = obs(d-1)
    preds["Persistence"]=obs.shift(1)
    # climatology: DOY mean from D7 q pre-2023
    d7=pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv",parse_dates=["date"])
    qd=(d7[d7.entity_id=="sub_634"].set_index("date")["q_mm"]/Q_CONV)
    clim=qd[qd.index<VAL0].groupby(qd[qd.index<VAL0].index.dayofyear).mean()
    preds["Climatology"]=pd.Series([clim.get(d.dayofyear,np.nan) for d in val_dates],index=val_dates)
    # GR4J simulation (same-day)
    gr=pd.read_csv(ROOT/"data/gold/G1_q_sim_gr4j_corrected.csv",parse_dates=["date"]).set_index("date")["q_sim_m3s_corrected"]
    preds["GR4J (sim)"]=gr[~gr.index.duplicated()]
    # HydroST from saved val predictions
    try:
        h=pd.read_csv(OUTML/"78_hydrost_val_2023.csv")
        # file has p10/p50/p90 (m3/s) + obs; align by matching obs to real
        if "date" in h.columns:
            preds["HydroST"]=pd.Series(h["p50"].values, index=pd.to_datetime(h["date"]))
        else:
            log.info("HydroST val CSV has no date col; skipping HydroST")
    except Exception as e:
        log.info(f"HydroST val skip: {e}")

    # LightGBM: retrain on pre-2023, predict val (fair held-out)
    import lightgbm as lgb
    m100=importlib.util.module_from_spec(importlib.util.spec_from_file_location("m100",ROOT/"scripts/06_eval/100_multi_horizon_skill.py"))
    importlib.util.spec_from_file_location("m100",ROOT/"scripts/06_eval/100_multi_horizon_skill.py").loader.exec_module(m100)
    df=m100.build_daily(); FEATS=m100.FEATS
    df["y_train"]=df["q"].shift(-1); df["y_obs"]=df["obs"].shift(-1)
    tr=df[df.index<VAL0].dropna(subset=FEATS+["y_train"])
    vv=df[(df.index>=VAL0)&(df.index<=VAL1)].dropna(subset=FEATS+["y_obs"])
    mdl=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=400,learning_rate=0.03,num_leaves=31,
        min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,verbose=-1)
    mdl.fit(tr[FEATS],np.log1p(tr["y_train"].clip(lower=0)))
    # LightGBM predicts for target date d+1 issued at d -> index by target date
    preds["LightGBM"]=pd.Series(np.expm1(mdl.predict(vv[FEATS])), index=vv.index+DAY)

    # TFT-tuned: retrain, predict val (BIASED: val = selection set)
    data=S.prepare_data(); bp=pd.read_csv(OUTML/"81_best_params.csv",index_col=0)["0"].to_dict()
    cfg=dict(ENC=int(bp["ENC"]),HID=int(bp["HID"]),HEADS=int(bp["HEADS"]),DROP=float(bp["DROP"]),
             LR=float(bp["LR"]),LR_FT=float(bp["LR_FT"]),WD=float(bp["WD"]),ALERT_W=float(bp["ALERT_W"]),mix_alpha=float(bp["mix_alpha"]))
    _,model=S.run_config(cfg,data,seed=42); model.eval()
    Xv,yv,wv,idxv=T.make_seqs(data["Xs"],data["ysc"],data["m_val"])
    with torch.no_grad():
        p=data["inv"](model(Xv.to(DEV)).cpu().numpy()[:,1])/Q_CONV
    preds["TFT-tuned*"]=pd.Series(p, index=pd.DatetimeIndex(data["dates"][idxv])+DAY)

    # ── Align all on val real-obs target dates & score ────────────────────────
    tgt=obs.reindex(val_dates).dropna()   # real obs on val target dates
    dfm=pd.DataFrame({"obs":tgt})
    for k,s in preds.items():
        s=s[~s.index.duplicated()]; dfm[k]=s.reindex(dfm.index)
    dfm=dfm.dropna()
    log.info(f"Validation 2023 common days: {len(dfm)}  (alert events obs>Q90: {(dfm['obs']>Q90).sum()})")
    rows=[]
    for k in [c for c in dfm.columns if c!="obs"]:
        rows.append(dict(model=k, **metrics(dfm["obs"], dfm[k])))
    res=pd.DataFrame(rows).sort_values("J_alert",ascending=False)
    log.info("\n=== VALIDATION 2023 LEADERBOARD (1-day) ===")
    log.info("\n"+res.to_string(index=False))
    log.info("* TFT-tuned: val was the hyperparameter-SELECTION set -> optimistically biased.")
    res.to_csv(OUTML/"103_validation_leaderboard.csv",index=False)
    log.info("Saved: 103_validation_leaderboard.csv")


if __name__=="__main__":
    main()
