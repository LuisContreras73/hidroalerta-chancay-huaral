#!/usr/bin/env python3
"""
Script 62: LSTM con Fine-Tuning en 2 fases — cerrar domain gap GR4J → Q real.

Estrategia (elegida por el usuario 2026-06-02):
  FASE 1 (pretrain): LSTM con GR4J reconstruct 1981-2020 (aprende física hidrológica)
  FASE 2 (finetune): ajuste con Q OBSERVADO REAL 2021-2023 (adapta a distribución real)
  TEST:              Q OBSERVADO REAL 2024-2025 (~500 días) — nunca visto

Comparación contra BASELINE (mismo test, para medir mejora real):
  BASELINE = solo FASE 1 (sin fine-tune), evaluado en test 2024-2025
  FINE-TUNED = FASE 1 + FASE 2, evaluado en test 2024-2025

Anti-sobreajuste en fase 2: LR bajo (1e-4), early stopping con paciencia corta,
mismo scaler que fase 1 (no re-ajustar).

Salidas:
  outputs/ml_Q/finetune_comparison.csv   — métricas baseline vs fine-tuned en test
  outputs/ml_Q/finetune_predictions.csv  — predicciones de ambos en test 2024-2025
  outputs/ml_Q/finetune_history.csv      — loss por epoch (ambas fases)
  configs/lstm_finetune_config.json
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(ROOT/"outputs"/"62_finetune_lstm_Q.log","w","utf-8")])
log = logging.getLogger("finetune_lstm")

D6_CSV     = ROOT / "data/model_ready/D6_multientity.csv"
THRESH     = ROOT / "data/model_ready/thresholds/q_thresholds.json"
OUT_DIR    = ROOT / "outputs/ml_Q"
CFG_DIR    = ROOT / "configs"
OUT_DIR.mkdir(parents=True, exist_ok=True); CFG_DIR.mkdir(parents=True, exist_ok=True)

Q_CONV = 86.4 / 3062.62
Q90    = 40.89
SEED   = 42

# Períodos (cronológicos, sin solape)
PRETRAIN_END   = "2017-12-31"   # fase 1 train: 1981-2017 (GR4J)
PREVAL_START   = "2018-01-01"   # fase 1 val:   2018-2020 (GR4J, early stop fase 1)
PREVAL_END     = "2020-12-31"
FINETUNE_START = "2021-01-01"   # fase 2 train: 2021-2022 (Q obs real)
FINETUNE_END   = "2022-12-31"
FTVAL_START    = "2023-01-01"   # fase 2 val:   2023 (Q obs real, early stop fase 2)
FTVAL_END      = "2023-12-31"
TEST_START     = "2024-01-01"   # test: 2024-2025 (Q obs real, nunca visto)

ENCODER_LEN = 90
HIDDEN, LAYERS, DROPOUT = 64, 2, 0.2
LR_PHASE1, LR_PHASE2 = 1e-3, 1e-4
BATCH = 64
MAX_EP1, MAX_EP2 = 120, 60
PAT1, PAT2 = 15, 10
TARGET = "q_next_1d"

SPATIAL = ["pr_mm","tmax_c","tmin_c","pet_mm","api","spi_30d","spi_90d","water_deficit_30d"]
SHARED  = ["oni_index","sin_doy_1","cos_doy_1","hydro_month","is_wet_season"]


def build_wide(D6):
    pivots = []
    for col in SPATIAL:
        if col in D6.columns:
            piv = D6.pivot_table(index="date", columns="entity_id", values=col)
            piv.columns = [f"{col}_{e}" for e in piv.columns]
            pivots.append(piv)
    ref = D6[D6["entity_id"]=="sub_634"].set_index("date")
    pivots.append(ref[[c for c in SHARED+[TARGET] if c in ref.columns]])
    return pd.concat(pivots, axis=1).sort_index()


def make_seq(X, y, enc):
    Xs, ys, idx = [], [], []
    for i in range(enc, len(X)):
        if np.isfinite(y[i]):
            Xs.append(X[i-enc:i]); ys.append(y[i]); idx.append(i)
    return np.array(Xs, np.float32), np.array(ys, np.float32), np.array(idx)


def nse(o,p): return 1-np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)
def nse_sqrt(o,p):
    w=np.maximum(o,0)**0.5
    return 1-np.sum(w*(o-p)**2)/(np.sum(w*(o-o.mean())**2)+1e-12)
def kge(o,p):
    r=np.corrcoef(o,p)[0,1]; a=p.std()/(o.std()+1e-12); b=p.mean()/(o.mean()+1e-12)
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def j_alert(o,p,thr):
    eo=o>thr; ep=p>thr
    TP=np.sum(eo&ep);FP=np.sum(~eo&ep);FN=np.sum(eo&~ep);TN=np.sum(~eo&~ep)
    POD=TP/(TP+FN+1e-9);FAR=FP/(FP+TN+1e-9);CSI=TP/(TP+FP+FN+1e-9)
    return 0.25*nse_sqrt(o,p)+0.25*nse(o,p)+0.30*CSI+0.10*POD-0.10*FAR
def metrics_dict(o,p,thr):
    eo=o>thr; ep=p>thr
    TP=int(np.sum(eo&ep));FP=int(np.sum(~eo&ep));FN=int(np.sum(eo&~ep));TN=int(np.sum(~eo&~ep))
    return {"NSE":round(nse(o,p),4),"NSE_sqrt":round(nse_sqrt(o,p),4),
            "KGE":round(kge(o,p),4),"J_alert":round(j_alert(o,p,thr),4),
            "POD":round(TP/(TP+FN+1e-9),3),"FAR":round(FP/(FP+TN+1e-9),3),
            "CSI":round(TP/(TP+FP+FN+1e-9),3),"N":len(o),"TP":TP,"FP":FP,"FN":FN}


def main():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("="*70); log.info(f"SCRIPT 62: LSTM Fine-Tuning 2 fases | device={dev}"); log.info("="*70)

    D6 = pd.read_csv(D6_CSV, parse_dates=["date"])
    wide = build_wide(D6)
    feat = [c for c in wide.columns if c != TARGET]
    dates = wide.index

    # Máscaras temporales
    m_pre_tr = dates <= PRETRAIN_END
    m_pre_vl = (dates >= PREVAL_START) & (dates <= PREVAL_END)
    m_ft_tr  = (dates >= FINETUNE_START) & (dates <= FINETUNE_END)
    m_ft_vl  = (dates >= FTVAL_START) & (dates <= FTVAL_END)
    m_test   = dates >= TEST_START

    log.info(f"[1] Wide {wide.shape} | {len(feat)} features")
    log.info(f"    Fase1 train(GR4J): ...{PRETRAIN_END} | val: {PREVAL_START}..{PREVAL_END}")
    log.info(f"    Fase2 finetune(Q real): {FINETUNE_START}..{FINETUNE_END} | val: {FTVAL_START}..{FTVAL_END}")
    log.info(f"    Test (Q real): {TEST_START}.. ")

    # Escalado: fit SOLO en fase 1 train
    X = wide[feat].fillna(0).values.astype(np.float32)
    y = wide[TARGET].values.astype(np.float32)
    mu, sd = X[m_pre_tr].mean(0), X[m_pre_tr].std(0)+1e-8
    X = (X-mu)/sd
    ylog = np.log1p(np.clip(y,0,None))
    ymu, ysd = np.nanmean(ylog[m_pre_tr]), np.nanstd(ylog[m_pre_tr])+1e-8
    ys = (ylog-ymu)/ysd
    inv = lambda v: np.expm1(v*ysd+ymu)

    # Secuencias
    Xseq, yseq, idx = make_seq(X, ys, ENCODER_LEN)
    seqdate = dates[idx]
    def sub(mask_dates):
        m = np.isin(idx, np.where(mask_dates)[0])
        return torch.tensor(Xseq[m]), torch.tensor(yseq[m]), seqdate[m]

    Xp, yp, _    = sub(m_pre_tr)
    Xpv, ypv, _  = sub(m_pre_vl)
    Xf, yf, _    = sub(m_ft_tr)
    Xfv, yfv, _  = sub(m_ft_vl)
    Xt, yt, dt_t = sub(m_test)
    log.info(f"[2] Secuencias: pretrain={len(Xp)}, preval={len(Xpv)}, "
             f"finetune={len(Xf)}, ftval={len(Xfv)}, test={len(Xt)}")

    class LSTMReg(nn.Module):
        def __init__(s, nf, h, l, d):
            super().__init__()
            s.lstm = nn.LSTM(nf, h, l, batch_first=True, dropout=d if l>1 else 0)
            s.head = nn.Sequential(nn.Linear(h,h//2), nn.ReLU(), nn.Dropout(d), nn.Linear(h//2,1))
        def forward(s,x):
            o,_ = s.lstm(x); return s.head(o[:,-1,:]).squeeze(-1)

    loss_fn = nn.MSELoss()
    history = []

    def train_phase(model, Xtr, ytr, Xvl, yvl, lr, max_ep, pat, phase):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
        Xvl_d, yvl_np = Xvl.to(dev), yvl.numpy()
        best=np.inf; best_state=None; ctr=0
        for ep in range(1, max_ep+1):
            model.train(); tl=0
            for xb,yb in dl:
                xb,yb=xb.to(dev),yb.to(dev)
                opt.zero_grad(); l=loss_fn(model(xb),yb); l.backward(); opt.step()
                tl+=l.item()*len(xb)
            tl/=len(Xtr)
            model.eval()
            with torch.no_grad(): vp=model(Xvl_d).cpu().numpy()
            vl=float(np.mean((vp-yvl_np)**2))
            vo, vpp = inv(yvl_np), inv(vp)
            history.append({"phase":phase,"epoch":ep,"train_loss":tl,"val_loss":vl,
                            "val_NSE":nse(vo,vpp)})
            if vl<best: best=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; ctr=0
            else:
                ctr+=1
                if ctr>=pat:
                    log.info(f"    [{phase}] early stop ep{ep} (best val_loss={best:.4f})"); break
        model.load_state_dict(best_state)
        return model

    # ── FASE 1: pretrain con GR4J ──────────────────────────────────────────
    log.info("\n[3] FASE 1 — pretrain con GR4J 1981-2020 ...")
    t0=time.time()
    model = LSTMReg(len(feat), HIDDEN, LAYERS, DROPOUT).to(dev)
    model = train_phase(model, Xp, yp, Xpv, ypv, LR_PHASE1, MAX_EP1, PAT1, "fase1")
    t_phase1 = time.time()-t0

    # Guardar estado del baseline (solo fase 1)
    baseline_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    # Evaluar BASELINE en test 2024-2025
    model.eval()
    with torch.no_grad(): pred_base = inv(model(Xt.to(dev)).cpu().numpy())
    obs_test = inv(yt.numpy())
    m_base = metrics_dict(obs_test, pred_base, Q90*Q_CONV)
    log.info(f"[4] BASELINE (solo fase1) en test 2024-2025: "
             f"NSE={m_base['NSE']} J_alert={m_base['J_alert']} POD={m_base['POD']}")

    # ── FASE 2: fine-tune con Q obs real 2021-2023 ─────────────────────────
    log.info("\n[5] FASE 2 — fine-tune con Q obs real 2021-2022 (val 2023) ...")
    t0=time.time()
    model = train_phase(model, Xf, yf, Xfv, yfv, LR_PHASE2, MAX_EP2, PAT2, "fase2")
    t_phase2 = time.time()-t0

    with torch.no_grad(): pred_ft = inv(model(Xt.to(dev)).cpu().numpy())
    m_ft = metrics_dict(obs_test, pred_ft, Q90*Q_CONV)
    log.info(f"[6] FINE-TUNED en test 2024-2025: "
             f"NSE={m_ft['NSE']} J_alert={m_ft['J_alert']} POD={m_ft['POD']}")

    # ── Comparación ────────────────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("COMPARACIÓN BASELINE vs FINE-TUNED (test = Q obs real 2024-2025)")
    log.info("="*70)
    comp = pd.DataFrame([{"modelo":"Baseline (GR4J only)", **m_base},
                         {"modelo":"Fine-tuned (+Q real)", **m_ft}])
    log.info(comp[["modelo","NSE","NSE_sqrt","KGE","J_alert","POD","FAR","CSI","N"]].to_string(index=False))
    log.info("")
    d_nse = m_ft["NSE"]-m_base["NSE"]; d_j = m_ft["J_alert"]-m_base["J_alert"]
    log.info(f"MEJORA fine-tuning: ΔNSE={d_nse:+.4f}  ΔJ_alert={d_j:+.4f}")
    log.info(f"{'✓ Fine-tuning MEJORA' if d_nse>0 else '✗ Fine-tuning NO mejora'}")

    # ── Guardar ────────────────────────────────────────────────────────────
    comp.to_csv(OUT_DIR/"finetune_comparison.csv", index=False)
    pd.DataFrame({"date":dt_t, "q_obs_m3s":obs_test/Q_CONV,
                  "q_pred_baseline_m3s":pred_base/Q_CONV,
                  "q_pred_finetuned_m3s":pred_ft/Q_CONV}).to_csv(
        OUT_DIR/"finetune_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR/"finetune_history.csv", index=False)
    json.dump({"strategy":"2-phase finetune","encoder_len":ENCODER_LEN,
               "lr_phase1":LR_PHASE1,"lr_phase2":LR_PHASE2,
               "pretrain":"1981-2020 GR4J","finetune":"2021-2022 Q obs real",
               "test":"2024-2025 Q obs real","time_phase1_s":round(t_phase1,1),
               "time_phase2_s":round(t_phase2,1),
               "baseline_NSE":float(m_base["NSE"]),"finetuned_NSE":float(m_ft["NSE"]),
               "delta_NSE":float(round(d_nse,4))}, open(CFG_DIR/"lstm_finetune_config.json","w"), indent=2)

    log.info("\nSCRIPT 62 COMPLETADO")


if __name__ == "__main__":
    main()
