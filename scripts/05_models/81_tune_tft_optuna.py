#!/usr/bin/env python3
"""
Script 81 — Bayesian hyperparameter tuning of TFT-v2 (Optuna TPE).

Methodology (paper-grade, anti-contamination):
  - SELECTION metric: J_alert on VAL 2023 (full year: wet+dry) vs REAL obs.
    Fixes Script 74's flaw of early-stopping on dry-season 2023H2 (no alerts).
  - TEST 2024-2025 stays SEALED: evaluated ONCE for the final best config only.
  - Each trial averages N_SEEDS to separate true effects from training noise.
  - Adds a mixed loss (pinball + MSE_P50), the trick that lifted HydroST NSE.

Search space: ENC, HID, HEADS, DROP, LR, LR_FT, WD, ALERT_W, mix_alpha.

Reuses architecture/data helpers from Script 74 by importing the module and
overriding its module-level config globals per trial.

Run in .venv313:
    python scripts/05_models/81_tune_tft_optuna.py --trials 40 --seeds 2
"""

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tune_tft")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Import TFT-v2 building blocks ──────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("tftv2", ROOT / "scripts/05_models/74_tft_v2_quantile_Q.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_CONV = T.Q_CONV
Q90    = T.Q90


# ── Data prep (once) — q_next_1d only ─────────────────────────────────────────

def prepare_data():
    D6   = pd.read_csv(T.D6_CSV, parse_dates=["date"])
    qobs = pd.read_csv(T.QOBS, index_col=0, parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm = qobs * Q_CONV

    df    = T.build_wide(D6)
    dates = df.index
    feat  = [c for c in df.columns if c not in ["q_next_1d", "q_sum_next_7d"]]
    nf    = len(feat)

    # Splits — SEALED test 2024+
    m_p   = dates <= "2017-12-31"                              # pretrain (GR4J era)
    m_ft  = (dates >= "2021-01-01") & (dates <= "2022-12-31")  # finetune on obs
    m_val = (dates >= "2023-01-01") & (dates <= "2023-12-31")  # VAL: full 2023 (wet+dry)
    m_te  = dates >= "2024-01-01"                              # SEALED test

    # Normalise features on pretrain only (anti-leakage)
    X  = df[feat].fillna(0).values.astype(np.float32)
    mu = X[m_p].mean(0); sd = X[m_p].std(0) + 1e-8
    Xs = (X - mu) / sd

    # Target: q_next_1d. Train target from D6 (may be GR4J-filled in gaps);
    # evaluation always against REAL obs (qobs_mm.shift(-1)).
    y    = df["q_next_1d"].values.astype(np.float32)
    ylog = np.log1p(np.clip(y, 0, None))
    ymu  = np.nanmean(ylog[m_p]); ysd = np.nanstd(ylog[m_p]) + 1e-8
    ysc  = ((ylog - ymu) / ysd).astype(np.float32)
    inv  = lambda v: np.expm1(np.asarray(v) * ysd + ymu)

    obs_real = qobs_mm.shift(-1).reindex(dates).values  # m³/s? no — mm; convert later

    return dict(df=df, dates=dates, Xs=Xs, ysc=ysc, inv=inv, nf=nf,
                m_p=m_p, m_ft=m_ft, m_val=m_val, m_te=m_te,
                obs_real_mm=obs_real)


# ── Mixed-loss training ───────────────────────────────────────────────────────

def train(model, Xtr, ytr, wtr, Xv, yv, lr, max_ep, pat, mix_alpha):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=T.WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=8)
    dl    = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=T.BATCH, shuffle=True)
    Xv_d, yv_np = Xv.to(DEVICE), yv.numpy()
    best, best_state, no_improve = np.inf, None, 0

    for ep in range(max_ep):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = mix_alpha * T.pinball_loss(pred, yb, wb) + (1 - mix_alpha) * F.mse_loss(pred[:, 1], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            p50_v = model(Xv_d)[:, 1].cpu().numpy()
        vl = float(np.mean((p50_v - yv_np) ** 2))
        sched.step(vl)
        if vl < best:
            best, best_state, no_improve = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= pat:
                break
    model.load_state_dict(best_state)
    return model


def eval_split(model, data, mask, inv):
    """Return (obs_m3, p10_m3, p50_m3, p90_m3) on real-obs days within mask."""
    Xseq, yseq, wseq, idx = T.make_seqs(data["Xs"], data["ysc"], mask)
    if Xseq is None:
        return None
    model.eval()
    with torch.no_grad():
        pred = model(Xseq.to(DEVICE)).cpu().numpy()
    p10 = inv(pred[:, 0]) / Q_CONV
    p50 = inv(pred[:, 1]) / Q_CONV
    p90 = inv(pred[:, 2]) / Q_CONV
    obs_mm = data["obs_real_mm"][idx]
    real = np.isfinite(obs_mm)
    obs_m3 = obs_mm[real] / Q_CONV
    return obs_m3, p10[real], p50[real], p90[real]


def run_config(cfg, data, seed):
    # Override TFT-v2 module globals for this config
    T.ENC, T.HID, T.HEADS, T.DROP = cfg["ENC"], cfg["HID"], cfg["HEADS"], cfg["DROP"]
    T.WD = cfg["WD"]
    torch.manual_seed(seed); np.random.seed(seed)

    # Rebuild sequences (ENC-dependent)
    Xp, yp, wp, _    = T.make_seqs(data["Xs"], data["ysc"], data["m_p"])
    Xft, yft, wft, _ = T.make_seqs(data["Xs"], data["ysc"], data["m_ft"],
                                   y_raw=np.expm1(data["ysc"]), alert_thr=None, alert_w=1.0)
    # Alert weighting on finetune: weight Q>Q90 (in scaled space need raw mm)
    # recompute weights properly using real target mm
    y_mm = data["df"]["q_next_1d"].values.astype(np.float32)
    thr_mm = Q90 * Q_CONV
    Xft, yft, wft, idxft = T.make_seqs(data["Xs"], data["ysc"], data["m_ft"],
                                       y_raw=y_mm, alert_thr=thr_mm, alert_w=cfg["ALERT_W"])
    # Use a slice of finetune as inner val for early stopping (last 20%)
    n = len(Xft); k = int(n * 0.8)
    model = T.TFTLitev2(data["nf"]).to(DEVICE)

    # Phase 1: pretrain on GR4J era (epoch caps reduced for HP search speed; early-stop intact)
    model = train(model, Xp, yp, wp, Xp[-200:], yp[-200:], cfg["LR"], 80, 18, cfg["mix_alpha"])
    # Phase 2: finetune on obs
    model = train(model, Xft[:k], yft[:k], wft[:k], Xft[k:], yft[k:], cfg["LR_FT"], 70, 15, cfg["mix_alpha"])

    # Evaluate on VAL 2023 (selection)
    res = eval_split(model, data, data["m_val"], data["inv"])
    if res is None:
        return None, None
    obs, p10, p50, p90 = res
    m = T.det_metrics(obs, p50, Q90)
    return m, model


def objective(trial, data, n_seeds):
    cfg = dict(
        ENC      = trial.suggest_int("ENC", 30, 90, step=15),
        HID      = trial.suggest_categorical("HID", [32, 48, 64, 96, 128]),
        HEADS    = trial.suggest_categorical("HEADS", [2, 4, 8]),
        DROP     = trial.suggest_float("DROP", 0.1, 0.5),
        LR       = trial.suggest_float("LR", 1e-4, 3e-3, log=True),
        LR_FT    = trial.suggest_float("LR_FT", 1e-5, 5e-4, log=True),
        WD       = trial.suggest_float("WD", 1e-6, 1e-3, log=True),
        ALERT_W  = trial.suggest_float("ALERT_W", 1.0, 10.0),
        mix_alpha= trial.suggest_float("mix_alpha", 0.5, 1.0),
    )
    # HID must be divisible by HEADS
    if cfg["HID"] % cfg["HEADS"] != 0:
        raise optuna.TrialPruned()

    j_scores = []
    for s in range(n_seeds):
        m, _ = run_config(cfg, data, seed=42 + s)
        if m is None:
            raise optuna.TrialPruned()
        j_scores.append(m["J_alert"])
        trial.report(np.mean(j_scores), step=s)
        if trial.should_prune():
            raise optuna.TrialPruned()

    j_mean = float(np.mean(j_scores))
    trial.set_user_attr("j_std", float(np.std(j_scores)))
    return j_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    log.info(f"Device: {DEVICE}  trials={args.trials}  seeds={args.seeds}")
    data = prepare_data()
    log.info(f"Features={data['nf']}  val2023 days={data['m_val'].sum()}  test days={data['m_te'].sum()}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(lambda t: objective(t, data, args.seeds), n_trials=args.trials,
                   show_progress_bar=False,
                   callbacks=[lambda st, tr: log.info(
                       f"  trial {tr.number:3d}  J_val={tr.value:.4f} "
                       f"(±{tr.user_attrs.get('j_std', 0):.3f})  best={st.best_value:.4f}"
                       if tr.value is not None else f"  trial {tr.number:3d} pruned")])

    log.info("\n" + "="*60)
    log.info(f"BEST val J_alert = {study.best_value:.4f}")
    log.info(f"BEST params: {study.best_params}")
    log.info("="*60)

    # ── Retrain best config, evaluate ONCE on sealed test ─────────────────────
    best = study.best_params
    cfg = dict(best)
    cfg.setdefault("mix_alpha", best["mix_alpha"])
    log.info("\nRetraining best config + SEALED TEST evaluation (single shot)...")

    test_rows = []
    for s in range(max(args.seeds, 3)):
        m_val, model = run_config(cfg, data, seed=100 + s)
        res = eval_split(model, data, data["m_te"], data["inv"])
        obs, p10, p50, p90 = res
        m_test = T.det_metrics(obs, p50, Q90)
        test_rows.append(m_test)
        log.info(f"  seed {s}: val J={m_val['J_alert']:.4f}  TEST J={m_test['J_alert']:.4f} "
                 f"NSE={m_test['NSE']:.4f} CSI={m_test['CSI']:.4f} POD={m_test['POD_det']:.4f}")

    td = pd.DataFrame(test_rows)
    log.info("\n" + "="*60)
    log.info("TUNED TFT — SEALED TEST 2024-2025 (mean ± std over seeds)")
    log.info("="*60)
    for col in ["NSE", "NSE_sqrt", "J_alert", "CSI", "POD_det", "FAR"]:
        log.info(f"  {col:10s} = {td[col].mean():.4f} ± {td[col].std():.4f}")

    # Save study + results
    study.trials_dataframe().to_csv(OUT_DIR / "81_optuna_trials.csv", index=False)
    pd.Series(study.best_params).to_csv(OUT_DIR / "81_best_params.csv")
    td.to_csv(OUT_DIR / "81_tuned_test_metrics.csv", index=False)
    log.info("\nSaved: 81_optuna_trials.csv, 81_best_params.csv, 81_tuned_test_metrics.csv")


if __name__ == "__main__":
    main()
