#!/usr/bin/env python3
"""
Script 115 — Tuning MÁXIMO de HydroST (arquitectura completa, Optuna resumible).

Justicia del paper: si tuneamos el TFT al máximo, tuneamos HydroST igual antes de
comparar. Se busca arquitectura + entrenamiento: hid, heads, layers, dropout, ENC,
LR1(pretrain), LR2(finetune), WD, ALPHA (mezcla pinball/MSE), ALERT_W, batch2.

Estrategia de costo (pretrain GR4J 1981-2020 es caro y cambia con la arquitectura):
  - DURANTE la búsqueda: pretrain ACORTADO (pocas épocas) + SUBSAMPLEADO (día por medio).
    Suficiente para RANKEAR configuraciones.
  - El MEJOR config se RE-ENTRENA a presupuesto COMPLETO (EP1=50, EP2=150) con 3 seeds.
Objetivo: minimizar la pinball de validación 2023 (mismo criterio de selección que 78).
Reutiliza TODO el pipeline del Script 78 por import. TEST 2024-2025 SELLADO (eval aparte).

Run in .venv313 (tras liberar GPU del TFT; paralelizable con --search-only + --seed):
    python scripts/05_models/115_hydrost_tune.py --trials 40
    python scripts/05_models/115_hydrost_tune.py --trials 40 --search-only --seed 1
"""
import argparse, importlib.util, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("hstune")
optuna.logging.set_verbosity(optuna.logging.WARNING)
spec=importlib.util.spec_from_file_location("h78",ROOT/"scripts/05_models/78_hydrost_Q.py")
H=importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
DEV=H.DEVICE

def val_pinball(model, loader):
    model.eval(); tot=0.0; n=0
    with torch.no_grad():
        for xd,xs,y in loader:
            pr=model(xd.to(DEV),xs.to(DEV)); tot+=H.pinball_loss(pr,y.to(DEV)).item(); n+=1
    return tot/max(n,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",type=int,default=40)
    ap.add_argument("--search-only",action="store_true"); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--ep1-search",type=int,default=10); ap.add_argument("--tag",default=""); a=ap.parse_args()

    # ── Datos: cargar UNA vez (caro: fill satélite + pivot); independiente de ENC/arch ──
    df=H.load_data(); dyn,stat,tgt,dates,scl=H.build_arrays(df)
    dpd=pd.DatetimeIndex(dates)
    pre=dpd[dpd<=H.PRETRAIN_END]; pre_sub=pre[::2]   # subsample para búsqueda
    ftd=dpd[(dpd>=H.FT_START)&(dpd<=H.FT_END)]; vad=dpd[(dpd>=H.VAL_START)&(dpd<=H.VAL_END)]

    def datasets(enc):
        H.ENC=enc
        return (H.HydroDataset(dyn,stat,tgt,dates,pre_sub),
                H.HydroDataset(dyn,stat,tgt,dates,ftd),
                H.HydroDataset(dyn,stat,tgt,dates,vad))

    def build_train(c, full=False):
        H.ENC=c["enc"]; H.LR1=c["lr1"]; H.LR2=c["lr2"]; H.WD=c["wd"]
        H.PAT=c.get("pat",25); H.ALPHA=c["alpha"]; H.ALERT_W=c["alert_w"]
        pre_dates = pre if full else pre_sub
        ds_pre=H.HydroDataset(dyn,stat,tgt,dates,pre_dates); ds_ft=H.HydroDataset(dyn,stat,tgt,dates,ftd); ds_va=H.HydroDataset(dyn,stat,tgt,dates,vad)
        lp=DataLoader(ds_pre,batch_size=H.BATCH1,shuffle=True); lf=DataLoader(ds_ft,batch_size=c["batch2"],shuffle=True); lv=DataLoader(ds_va,batch_size=c["batch2"],shuffle=False)
        m=H.HydroST(hid=c["hid"],heads=c["heads"],layers=c["layers"],dropout=c["drop"]).to(DEV)
        ep1 = 50 if full else a.ep1_search
        m=H.train_phase1(m,lp,ep1); m=H.train_phase2(m,lf,lv, 150 if full else 80)
        return m,lv

    def objective(t):
        c=dict(hid=t.suggest_categorical("hid",[32,48,64,96,128]),
               heads=t.suggest_categorical("heads",[2,4,8]),
               layers=t.suggest_int("layers",1,3),
               drop=t.suggest_float("drop",0.1,0.45),
               enc=t.suggest_categorical("enc",[30,45,60,90]),
               lr1=t.suggest_float("lr1",3e-4,3e-3,log=True),
               lr2=t.suggest_float("lr2",3e-5,5e-4,log=True),
               wd=t.suggest_float("wd",1e-6,1e-3,log=True),
               alpha=t.suggest_float("alpha",0.5,0.9),
               alert_w=t.suggest_float("alert_w",2.0,8.0),
               batch2=t.suggest_categorical("batch2",[32,64,128]))
        if c["hid"]%c["heads"]!=0: raise optuna.TrialPruned()
        torch.manual_seed(0); np.random.seed(0)
        m,lv=build_train(c,full=False); return val_pinball(m,lv)

    storage=f"sqlite:///{OUTML/('115_hydrost'+(a.tag and '_'+a.tag)+'.db')}"
    st=optuna.create_study(direction="minimize",study_name="hydrost_arch"+(a.tag and '_'+a.tag),storage=storage,load_if_exists=True,sampler=TPESampler(seed=a.seed,n_startup_trials=12))
    log.info(f"seed={a.seed} search_only={a.search_only}. Trials: {len([t for t in st.trials if t.state.is_finished()])}; +{a.trials}")
    st.optimize(objective,n_trials=a.trials,callbacks=[lambda s,t: log.info(f"  #{t.number} valPin={t.value:.4f} best={s.best_value:.4f} hid={t.params.get('hid')} enc={t.params.get('enc')} L={t.params.get('layers')}") if t.value else None])
    if a.search_only: log.info(f"[search-only] best={st.best_value:.4f} {st.best_params}"); return
    bp=st.best_params; log.info(f"BEST valPin={st.best_value:.4f} {bp}")

    # ── Re-entreno FULL del mejor config, 3 seeds; eval val 2023 (test 2024-25 SELLADO) ──
    Q90=40.89; res=[]
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        m,lv=build_train(bp,full=True); r=H.evaluate(m,lv,f"val_seed{s}")
        res.append(r); torch.save(m.state_dict(), OUTML/f"115_hydrost_best_seed{s}.pt")
    NSE=np.mean([r["NSE"] for r in res]); JA=np.mean([r["J_alert"] for r in res]); CSI=np.mean([r["CSI"] for r in res])
    log.info(f"HydroST TUNEADO (val 2023, 3 seeds): NSE={NSE:.3f} J_alert={JA:.3f} CSI={CSI:.3f}")
    pd.DataFrame([dict(model="HydroST-tuned",split="val2023",NSE=round(NSE,3),J_alert=round(JA,3),CSI=round(CSI,3),**{k:bp[k] for k in bp})]).to_csv(OUTML/"115_hydrost_tuned.csv",index=False)
    log.info("Saved: 115_hydrost_tuned.csv  (+ 3 checkpoints). Test 2024-25 sellado: correr eval aparte con el best config.")

if __name__=="__main__":
    main()
