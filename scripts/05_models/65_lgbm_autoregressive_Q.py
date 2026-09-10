#!/usr/bin/env python3
"""
Script 65: LightGBM Autoregresivo + SHAP — target Q, horizontes 1d y 7d.

Baseline interpretable fuerte. Usa Q reciente + lags meteorológicos como features
tabulares (LightGBM no es secuencial). SHAP explica qué variable manda en cada predicción.

Features:
  - Espaciales (9 sub-cuencas): pr, tmax, tmin, pet, api, spi30, spi90, water_deficit
  - Autoregresivas Q (outlet): q_mm, q_lag1, q_lag7, q_roll7, q_roll30  [gap-fill GR4J corr]
  - Lags precipitación cuenca: pr_basin, pr_roll7, pr_roll30
  - Compartidas: oni, calendario

Targets: q_next_1d, q_sum_next_7d
Evaluación honesta: vs Q OBSERVADO REAL (1d: Q[t+1]; 7d: suma Q[t+1..t+7] sin gaps).

Salida:
  outputs/ml_Q/lgbm_ar_results.csv
  outputs/ml_Q/lgbm_ar_predictions.csv
  outputs/figures/ml_Q/LGB01_shap_{1d,7d}.png
  outputs/figures/ml_Q/LGB02_hydrograph.png
"""
import logging, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).parent.parent.parent
sys.path.insert(0,str(ROOT/"data/metadata"))
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log=logging.getLogger("lgbm_ar")

D6_CSV=ROOT/"data/model_ready/D6_multientity.csv"
QOBS=ROOT/"data/silver/snirh/S1_snirh_daily_q.csv"
OUT_DIR=ROOT/"outputs/ml_Q"; FIG_DIR=ROOT/"outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
Q_CONV=86.4/3062.62; Q90=40.89; SEED=42

SPATIAL=["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED=["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]

try:
    from entity_labels import ENTITY_META
    SUB_NAMES={e:m["nombre"] for e,m in ENTITY_META.items()}
except Exception:
    SUB_NAMES={}
VAR_ES={"pr_mm":"Precip","tmax_c":"Tmax","tmin_c":"Tmin","pet_mm":"ETP","api":"API",
        "spi_30d":"SPI30","spi_90d":"SPI90","water_deficit_30d":"DéficitH","oni_index":"ONI",
        "q_mm":"Q hoy","q_lag1":"Q ayer","q_lag7":"Q -7d","q_roll7":"Q media7","q_roll30":"Q media30",
        "pr_basin":"Precip cuenca","pr_roll7":"Precip 7d","pr_roll30":"Precip 30d",
        "sin_doy_1":"Estacional","cos_doy_1":"Estacional2","hydro_month":"Mes hidro","is_wet_season":"T.húmeda"}

def pretty(f):
    for e,n in SUB_NAMES.items():
        if f.endswith("_"+e): return f"{VAR_ES.get(f[:-(len(e)+1)],f[:-(len(e)+1)])}·{n}"
    return VAR_ES.get(f,f)

def build_features(D6):
    """Wide + autoregresivo + lags. 1 fila/día."""
    pivots=[]
    for c in SPATIAL:
        p=D6.pivot_table(index="date",columns="entity_id",values=c)
        p.columns=[f"{c}_{e}" for e in p.columns]; pivots.append(p)
    ref=D6[D6["entity_id"]=="sub_634"].set_index("date")
    df=pd.concat(pivots+[ref[SHARED+["q_mm","q_next_1d","q_sum_next_7d"]]],axis=1).sort_index()
    # Autoregresivo Q (de q_mm = obs + GR4J corregido)
    q=df["q_mm"]
    df["q_lag1"]=q.shift(1); df["q_lag7"]=q.shift(7)
    df["q_roll7"]=q.rolling(7,min_periods=3).mean()
    df["q_roll30"]=q.rolling(30,min_periods=10).mean()
    # Precipitación media cuenca + lags
    pr_cols=[c for c in df.columns if c.startswith("pr_mm_")]
    df["pr_basin"]=df[pr_cols].mean(axis=1)
    df["pr_roll7"]=df["pr_basin"].rolling(7,min_periods=3).sum()
    df["pr_roll30"]=df["pr_basin"].rolling(30,min_periods=10).sum()
    return df

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
    from lightgbm import LGBMRegressor
    import shap
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    D6=pd.read_csv(D6_CSV,parse_dates=["date"])
    qobs=pd.read_csv(QOBS,index_col=0,parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm=qobs*Q_CONV
    df=build_features(D6)
    dates=df.index
    m_tr=dates<="2015-12-31"; m_te=dates>="2021-01-01"

    feat=[c for c in df.columns if c not in ["q_next_1d","q_sum_next_7d"]]
    log.info(f"Features: {len(feat)} (incluye Q autoregresivo + lags)")

    # Targets reales para evaluación honesta
    qobs_next1=qobs_mm.shift(-1)                       # Q[t+1]
    # suma 7d real: requiere 7 días con obs
    qobs_sum7=qobs_mm.shift(-1).rolling(7).sum().shift(-6)

    results=[]; preds_out={}
    shap_data={}
    for target, obs_real in [("q_next_1d",qobs_next1),("q_sum_next_7d",qobs_sum7)]:
        y=df[target].values
        tr=m_tr & np.isfinite(y)
        Xtr=df.loc[tr,feat].fillna(0); ytr=y[tr]
        model=LGBMRegressor(n_estimators=500,num_leaves=31,learning_rate=0.03,
                            subsample=0.8,colsample_bytree=0.8,random_state=SEED,
                            n_jobs=-1,verbosity=-1)
        model.fit(Xtr,ytr)
        # Test: predecir y evaluar vs Q obs real
        Xte=df.loc[m_te,feat].fillna(0)
        pred=pd.Series(model.predict(Xte),index=dates[m_te])
        o=obs_real.reindex(dates[m_te])
        valid=o.notna()
        thr=Q90*Q_CONV if target=="q_next_1d" else Q90*Q_CONV*7
        m=metr(o[valid].values,pred[valid].values,thr)
        m["target"]=target; results.append(m)
        log.info(f"[{target}] test honesto: NSE={m['NSE']} KGE={m['KGE']} "
                 f"J_alert={m['J_alert']} POD={m['POD']} N={m['N']}")
        preds_out[target]=pd.DataFrame({"date":dates[m_te][valid.values],
                                        "q_obs":o[valid].values/(Q_CONV*(7 if "7d" in target else 1)),
                                        "q_pred":pred[valid].values/(Q_CONV*(7 if "7d" in target else 1))})
        # SHAP sobre muestra de test
        expl=shap.TreeExplainer(model)
        Xsh=Xte.sample(min(500,len(Xte)),random_state=SEED)
        shap_data[target]=(expl.shap_values(Xsh),Xsh)

    pd.DataFrame(results).to_csv(OUT_DIR/"lgbm_ar_results.csv",index=False)

    # ── SHAP figura (top features por target) ───────────────────────────────
    for target in ["q_next_1d","q_sum_next_7d"]:
        sv,Xsh=shap_data[target]
        imp=pd.Series(np.abs(sv).mean(0),index=Xsh.columns).sort_values(ascending=True).tail(15)
        fig,ax=plt.subplots(figsize=(9,7))
        colors=["#e74c3c" if "q_" in f and "lag" not in f.replace("q_lag","") or f.startswith("q_") else "#3498db" for f in imp.index]
        colors=["#e74c3c" if f.startswith("q_") else "#3498db" for f in imp.index]
        ax.barh([pretty(f) for f in imp.index],imp.values,color=colors,alpha=0.85)
        ax.set_xlabel("Importancia SHAP media |valor|")
        tname="Q +1 día" if target=="q_next_1d" else "Q suma 7 días"
        ax.set_title(f"SHAP — LightGBM autoregresivo · {tname}\n"
                     f"rojo = features de caudal (Q), azul = meteorología/otros",fontweight="bold")
        ax.grid(True,alpha=0.3,axis="x")
        fig.tight_layout()
        tag="1d" if target=="q_next_1d" else "7d"
        fig.savefig(FIG_DIR/f"LGB01_shap_{tag}.png",bbox_inches="tight");plt.close(fig)
        log.info(f"  -> LGB01_shap_{tag}.png")

    # ── Hidrograma test ──────────────────────────────────────────────────────
    fig,axes=plt.subplots(2,1,figsize=(15,8))
    for ax,target,r in zip(axes,["q_next_1d","q_sum_next_7d"],results):
        p=preds_out[target].sort_values("date")
        ax.fill_between(p["date"],0,p["q_obs"],alpha=0.15,color="#c0392b")
        ax.plot(p["date"],p["q_obs"],color="#c0392b",lw=1.4,label="Q observado real")
        ax.plot(p["date"],p["q_pred"],color="#9b59b6",lw=1.2,ls="--",label="LightGBM AR")
        tname="Q +1 día (m³/s)" if target=="q_next_1d" else "Q suma 7d (m³/s acum)"
        ax.set_ylabel(tname);ax.set_ylim(0)
        ax.set_title(f"{target}: NSE={r['NSE']:.3f} KGE={r['KGE']:.3f} POD={r['POD']:.0%} (test Q obs real)",
                     fontsize=10,fontweight="bold")
        ax.legend(fontsize=9,loc="upper right");ax.grid(True,alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("LightGBM Autoregresivo — pronóstico Q (test = Q observado real 2021-2025)",
                 fontsize=12,fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/"LGB02_hydrograph.png",bbox_inches="tight");plt.close(fig)
    log.info("  -> LGB02_hydrograph.png")

    log.info("\n=== RESUMEN LightGBM AR ===")
    log.info(pd.DataFrame(results)[["target","NSE","KGE","J_alert","POD","CSI","N"]].to_string(index=False))
    log.info("SCRIPT 65 COMPLETADO")

if __name__=="__main__":
    main()
