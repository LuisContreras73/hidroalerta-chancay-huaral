#!/usr/bin/env python3
"""
Script 76: N-HiTS — Neural Hierarchical Interpolation for Time Series

Implementación desde cero en PyTorch (sin neuralforecast) para aprovechar
la GPU RTX 4070 SUPER y el entorno .venv313 ya configurado.

Arquitectura N-HiTS (Challu et al., 2022):
  - 3 stacks con resoluciones temporales distintas (pool_sizes = [8, 4, 1])
  - Cada stack: AdaptiveMaxPool1d → 3-block MLP → backcast + forecast
  - Backcast subtraction: cada stack procesa residuos del anterior
    → Stack 1 (pool=8): captura tendencias de ~2 semanas
    → Stack 2 (pool=4): captura ciclos semanales
    → Stack 3 (pool=1): captura variaciones diarias (alta frecuencia)
  - Forecast total: suma de contribuciones de todos los stacks
  - Sin atención (más rápido que TFT, diseñado para multi-step)

Ventaja sobre TFT/LGB para 7d:
  - Forecast directo multi-step (no recursivo) → no acumula error
  - MaxPool extrae multi-escala temporal de forma natural
  - Arquitectura jerárquica → descompone señal en componentes interpretables

Salidas cuantílicas: P10/P50/P90 con no-cruce garantizado via softplus.
  P10 < P50 < P90 → misma garantía que TFT v2 (Script 74).

Transfer (= Scripts 72/74):
  Fase 1: GR4J 1981-2017 pretrain  (val GR4J 2018-2020)
  Fase 2: Q obs 2021-23H1 finetune (val 2023H2, w_alert=5 para Q>Q90)
  Test:   Q obs 2024-2025

Entorno: .venv313 (Python 3.13, PyTorch 2.6.0+cu124, RTX 4070 SUPER)

Salidas:
  outputs/ml_Q/nhits_results.csv
  outputs/ml_Q/nhits_predictions.csv
  outputs/figures/ml_Q/NHITS01_quantile_bands.png
  outputs/figures/ml_Q/NHITS02_comparison.png
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Fix crítico torch.load (Lightning 2.6, Regla 15) ─────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load
torch.set_float32_matmul_precision("high")   # Tensor Cores RTX 4070 SUPER

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("nhits")

D6_CSV  = ROOT / "data/model_ready/D6_multientity.csv"
QOBS    = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
ONI_CSV = ROOT / "data/silver/enso/S3_oni_1950_2026.csv"
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/figures/ml_Q"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

Q_CONV    = 86.4 / 3062.62
Q90       = 40.89
SEED      = 42

# ── Hiperparámetros N-HiTS ────────────────────────────────────────────────────
ENC        = 60               # días de contexto encoder
POOL_SIZES = [8, 4, 1]        # resoluciones temporales (3 stacks)
N_BLOCKS   = 3                # bloques por stack
HIDDEN     = 128              # ancho MLP — reducido para evitar overfitting (~2.5M params)
N_LAYERS   = 3                # capas MLP por bloque
DROP       = 0.1
LR         = 5e-4
LR_FT      = 5e-5
WD         = 1e-5
BATCH      = 128
MAX_EP1    = 250
MAX_EP2    = 100
PAT1       = 35
PAT2       = 20
ALERT_W    = 5.0
QUANTILES  = [0.1, 0.5, 0.9]
ALPHA      = 0.80

SPATIAL  = ["pr_mm", "tmax_c", "tmin_c", "pet_mm", "api",
             "spi_30d", "spi_90d", "water_deficit_30d"]
SHARED   = ["oni_index", "sin_doy_1", "cos_doy_1", "hydro_month", "is_wet_season"]
KEY_SUBS = ["sub_649", "sub_655", "sub_656"]


# ── Features (= Scripts 72/74/75) ─────────────────────────────────────────────

def build_wide(D6):
    cols = {}
    for c in SPATIAL:
        piv = D6.pivot_table(index="date", columns="entity_id", values=c)
        cols[f"{c}_basin"] = piv.mean(axis=1)
        for e in KEY_SUBS:
            if e in piv.columns:
                cols[f"{c}_{e}"] = piv[e]
    df = pd.DataFrame(cols)
    ref = D6[D6["entity_id"] == "sub_634"].set_index("date")
    df = df.join(ref[SHARED + ["q_mm", "q_next_1d", "q_sum_next_7d"]]).sort_index()
    q = df["q_mm"]
    df["q_lag7"]  = q.shift(7)
    df["q_roll7"] = q.rolling(7,  min_periods=3).mean()
    df["q_roll30"]= q.rolling(30, min_periods=10).mean()
    return df


# ── Métricas (= Scripts 74/75) ────────────────────────────────────────────────

def _nse(o, p):     return 1 - np.sum((o-p)**2) / (np.sum((o-o.mean())**2) + 1e-12)
def _nse_sqrt(o,p):
    w=np.maximum(o,0)**0.5; return 1-np.sum(w*(o-p)**2)/(np.sum(w*(o-o.mean())**2)+1e-12)
def _lognse(o, p):
    lo=np.log(o+1); lp=np.log(np.clip(p,0,None)+1)
    return 1-np.sum((lo-lp)**2)/(np.sum((lo-lo.mean())**2)+1e-12)
def _kge(o, p):
    r=np.corrcoef(o,p)[0,1]; a=p.std()/(o.std()+1e-12); b=p.mean()/(o.mean()+1e-12)
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)

def det_metrics(o, p, thr):
    eo,ep = o>thr, p>thr
    TP=int(np.sum(eo&ep)); FP=int(np.sum(~eo&ep))
    FN=int(np.sum(eo&~ep)); TN=int(np.sum(~eo&~ep))
    POD=TP/(TP+FN+1e-9); FAR=FP/(FP+TN+1e-9); CSI=TP/(TP+FP+FN+1e-9)
    dh=(TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)
    HSS=2*(TP*TN-FP*FN)/(dh+1e-9) if dh>0 else 0
    j=0.25*_nse_sqrt(o,p)+0.25*_nse(o,p)+0.30*CSI+0.10*POD-0.10*FAR
    pb=(p.sum()-o.sum())/(o.sum()+1e-12)*100
    q90l=np.percentile(o,90); pk=o>q90l
    pb90=(p[pk].sum()-o[pk].sum())/(o[pk].sum()+1e-12)*100 if pk.sum()>3 else np.nan
    return {"NSE":round(_nse(o,p),3),"NSE_sqrt":round(_nse_sqrt(o,p),3),
            "logNSE":round(_lognse(o,p),3),"KGE":round(_kge(o,p),3),
            "PBIAS":round(float(pb),1),
            "PBIAS_Q90":round(float(pb90),1) if not np.isnan(pb90) else np.nan,
            "POD_det":round(POD,3),"FAR":round(FAR,3),"CSI":round(CSI,3),
            "HSS":round(HSS,3),"J_alert":round(j,4),"N":len(o)}

def prob_metrics(o, p10, p90, thr):
    inside = (o>=p10)&(o<=p90)
    width  = float((p90-p10).mean())
    IS = (p90-p10)+(2/ALPHA)*np.maximum(p10-o,0)+(2/ALPHA)*np.maximum(o-p90,0)
    eo=o>thr; ep90=p90>thr
    TP=int(np.sum(eo&ep90)); FN=int(np.sum(eo&~ep90)); FP=int(np.sum(~eo&ep90)); TN=int(np.sum(~eo&~ep90))
    return {"Coverage80":round(float(inside.mean()),3),
            "Width_m3s":round(width/Q_CONV,2),
            "IS":round(float(IS.mean()),4),
            "POD_prob":round(TP/(TP+FN+1e-9),3),
            "FAR_prob":round(FP/(FP+TN+1e-9),3)}


# ── N-HiTS Architecture ───────────────────────────────────────────────────────

class NHiTSBlock(nn.Module):
    """Un bloque N-HiTS: Pool → MLP → backcast + forecast (3 cuantiles)."""

    def __init__(self, nf, pool_size, enc_len, hidden, n_layers, dropout):
        super().__init__()
        pooled_len   = max(1, enc_len // pool_size)
        self.pool    = nn.AdaptiveMaxPool1d(pooled_len)
        flat_in      = nf * pooled_len

        mlp_layers = [nn.Linear(flat_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            mlp_layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        mlp_layers += [nn.Linear(hidden, hidden)]
        self.mlp = nn.Sequential(*mlp_layers)

        # Backcast: reconstruye input (ENC × nf) para restar residuo
        self.backcast_head = nn.Linear(hidden, nf * enc_len)

        # Forecast: 3 salidas — [p50_contrib, delta_down_raw, delta_up_raw]
        self.forecast_head = nn.Linear(hidden, 3)

        self.enc_len = enc_len
        self.nf      = nf

    def forward(self, x):
        # x: (B, ENC, nf)
        B = x.shape[0]
        pooled = self.pool(x.permute(0, 2, 1))    # (B, nf, pooled_len)
        flat   = pooled.reshape(B, -1)             # (B, flat_in)
        h      = self.mlp(flat)                    # (B, hidden)

        backcast = self.backcast_head(h).reshape(B, self.enc_len, self.nf)
        forecast = self.forecast_head(h)           # (B, 3)
        return backcast, forecast


class NHiTSStack(nn.Module):
    """Stack N-HiTS: n_blocks bloques que procesan residuos sucesivos."""

    def __init__(self, nf, pool_size, enc_len, hidden, n_layers, dropout, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            NHiTSBlock(nf, pool_size, enc_len, hidden, n_layers, dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        total_forecast = None
        for block in self.blocks:
            backcast, forecast = block(x)
            x = x - backcast
            total_forecast = forecast if total_forecast is None else total_forecast + forecast
        return x, total_forecast


class NHiTS(nn.Module):
    """N-HiTS completo: 3 stacks con resoluciones temporales distintas.

    Cuantiles garantizados P10 < P50 < P90 via softplus deltas sobre P50:
      raw[:,0] = contribución P50 acumulada de todos los stacks
      raw[:,1] = delta_down_raw  →  P10 = P50 - softplus(delta_down)
      raw[:,2] = delta_up_raw    →  P90 = P50 + softplus(delta_up)
    """

    def __init__(self, nf, enc_len=ENC, pool_sizes=POOL_SIZES,
                 n_blocks=N_BLOCKS, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROP):
        super().__init__()
        self.stacks = nn.ModuleList([
            NHiTSStack(nf, ps, enc_len, hidden, n_layers, dropout, n_blocks)
            for ps in pool_sizes
        ])

    def forward(self, x):                          # x: (B, ENC, nf)
        raw_total = None
        for stack in self.stacks:
            x, forecast = stack(x)
            raw_total = forecast if raw_total is None else raw_total + forecast

        p50 = raw_total[:, 0]
        p10 = p50 - F.softplus(raw_total[:, 1])   # garantiza P10 < P50
        p90 = p50 + F.softplus(raw_total[:, 2])   # garantiza P90 > P50
        return torch.stack([p10, p50, p90], dim=1) # (B, 3)


# ── Pinball Loss (= Script 74) ────────────────────────────────────────────────

def pinball_loss(pred, target, weights=None):
    total = 0.0
    for i, q in enumerate(QUANTILES):
        err = target - pred[:, i]
        l   = torch.where(err >= 0, q * err, (q-1) * err)
        total += (l * weights).mean() if weights is not None else l.mean()
    return total / len(QUANTILES)


# ── Sequence builder (= Script 74) ───────────────────────────────────────────

def make_seqs(Xs, ysc, mask, y_raw=None, alert_thr=None, alert_w=1.0):
    mask_set = set(np.where(mask)[0])
    Xs_out, ys_out, ws_out, idx_out = [], [], [], []
    for i in range(ENC, len(Xs)):
        if i not in mask_set or not np.isfinite(ysc[i]):
            continue
        Xs_out.append(Xs[i-ENC:i])
        ys_out.append(ysc[i])
        w = alert_w if (y_raw is not None and alert_thr is not None
                         and y_raw[i] > alert_thr) else 1.0
        ws_out.append(w)
        idx_out.append(i)
    if not Xs_out:
        return None, None, None, None
    return (torch.tensor(np.array(Xs_out, np.float32)),
            torch.tensor(np.array(ys_out, np.float32)),
            torch.tensor(np.array(ws_out, np.float32)),
            np.array(idx_out))


# ── Training loop (= Script 74) ───────────────────────────────────────────────

def train_phase(model, Xtr, ytr, wtr, Xv, yv, lr, max_ep, pat, dev):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10)
    dl    = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=BATCH, shuffle=True)
    Xv_d  = Xv.to(dev); yv_np = yv.numpy()
    best, best_state, no_imp = np.inf, None, 0

    for ep in range(max_ep):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            opt.zero_grad()
            pinball_loss(model(xb), yb, wb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            p50_v = model(Xv_d)[:, 1].cpu().numpy()
        vl = float(np.mean((p50_v - yv_np)**2))
        sched.step(vl)

        if vl < best:
            best = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= pat:
                break

    model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Script 76 — N-HiTS | device={dev}")
    log.info(f"Stacks pool_sizes={POOL_SIZES} | blocks/stack={N_BLOCKS} | hidden={HIDDEN}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # ── Datos ─────────────────────────────────────────────────────────────────
    D6    = pd.read_csv(D6_CSV, parse_dates=["date"])
    qobs  = pd.read_csv(QOBS, index_col=0, parse_dates=True)["q_santo_domingo_47e214d2"].dropna()
    qobs_mm = qobs * Q_CONV
    oni   = pd.read_csv(ONI_CSV, parse_dates=["date"]).set_index("date")["oni"]

    df    = build_wide(D6)
    dates = df.index
    feat  = [c for c in df.columns if c not in ["q_next_1d", "q_sum_next_7d"]]
    nf    = len(feat)

    # Contar parámetros del modelo
    dummy = NHiTS(nf).to(dev)
    n_params = sum(p.numel() for p in dummy.parameters() if p.requires_grad)
    del dummy
    log.info(f"Modelo N-HiTS: {n_params:,} parámetros | features={nf}")

    # Splits (= Scripts 72/74/75)
    m_p  = dates <= "2017-12-31"
    m_pv = (dates >= "2018-01-01") & (dates <= "2020-12-31")
    m_ft = (dates >= "2021-01-01") & (dates <= "2023-06-30")
    m_fv = (dates >= "2023-07-01") & (dates <= "2023-12-31")
    m_te = dates >= "2024-01-01"
    log.info(f"Pretrain: {m_p.sum()}d | FT: {m_ft.sum()}d | Test: {m_te.sum()}d")

    # Normalización solo en pretrain
    X  = df[feat].fillna(0).values.astype(np.float32)
    mu = X[m_p].mean(0); sd = X[m_p].std(0) + 1e-8
    Xs = (X - mu) / sd

    targets = {
        "q_next_1d":     (qobs_mm.shift(-1),                            1),
        "q_sum_next_7d": (qobs_mm.shift(-1).rolling(7).sum().shift(-6), 7),
    }

    all_rows  = []
    all_preds = []

    for target, (obs_real, hmult) in targets.items():
        log.info(f"\n{'='*55}\n[{target}]\n{'='*55}")
        y   = df[target].values.astype(np.float32)
        thr = Q90 * Q_CONV * hmult

        ylog = np.log1p(np.clip(y, 0, None))
        ymu  = np.nanmean(ylog[m_p]); ysd = np.nanstd(ylog[m_p]) + 1e-8
        ysc  = ((ylog - ymu) / ysd).astype(np.float32)
        inv  = lambda v: np.expm1(np.asarray(v) * ysd + ymu)

        Xp_t,  yp_t,  wp_t,  idxp  = make_seqs(Xs, ysc, m_p)
        Xpv_t, ypv_t, wpv_t, idxpv = make_seqs(Xs, ysc, m_pv)
        Xft_t, yft_t, wft_t, idxft = make_seqs(Xs, ysc, m_ft,
                                                 y_raw=y, alert_thr=thr, alert_w=ALERT_W)
        Xfv_t, yfv_t, wfv_t, idxfv = make_seqs(Xs, ysc, m_fv)
        Xte_t, yte_t, wte_t, idxte = make_seqs(Xs, ysc, m_te)
        if Xte_t is None:
            log.warning(f"[{target}] Sin secuencias test — omitiendo"); continue

        n_alert = int((y[idxft] > thr).sum()) if idxft is not None else 0
        log.info(f"  Seqs → pretrain={len(idxp)} | finetune={len(idxft)} "
                 f"(alertas={n_alert}) | test={len(idxte)}")

        # ── Fase 1: pretrain GR4J ──────────────────────────────────────────
        model = NHiTS(nf).to(dev)
        t0 = time.time()
        log.info(f"  Fase 1 pretrain (GR4J ≤2017, max={MAX_EP1} ep, pat={PAT1})...")
        model = train_phase(model, Xp_t, yp_t, wp_t,
                             Xpv_t, ypv_t, LR, MAX_EP1, PAT1, dev)
        log.info(f"  Fase 1 OK ({time.time()-t0:.0f}s)")

        # ── Fase 2: finetune Q obs con sample weighting ────────────────────
        t0 = time.time()
        log.info(f"  Fase 2 finetune (Q real 2021-23H1, w_alert={ALERT_W}, max={MAX_EP2} ep)...")
        model = train_phase(model, Xft_t, yft_t, wft_t,
                             Xfv_t, yfv_t, LR_FT, MAX_EP2, PAT2, dev)
        log.info(f"  Fase 2 OK ({time.time()-t0:.0f}s)")

        # ── Inferencia test ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            raw = model(Xte_t.to(dev)).cpu().numpy()   # (N_te, 3)

        p10_mm = inv(raw[:, 0])
        p50_mm = inv(raw[:, 1])
        p90_mm = inv(raw[:, 2])

        dte   = dates[idxte]
        o_ser = obs_real.reindex(dte)
        valid = o_ser.notna()
        o_v   = o_ser[valid].values
        p10_v = p10_mm[valid.values]
        p50_v = p50_mm[valid.values]
        p90_v = p90_mm[valid.values]

        m_det  = det_metrics(o_v, p50_v, thr)
        m_prob = prob_metrics(o_v, p10_v, p90_v, thr)
        row    = {**m_det, **m_prob, "target": target, "model": "N-HiTS"}
        all_rows.append(row)

        log.info(f"  P50  → NSE={m_det['NSE']} KGE={m_det['KGE']} "
                 f"J_alert={m_det['J_alert']} POD_det={m_det['POD_det']:.0%}")
        log.info(f"  P90  → POD_prob={m_prob['POD_prob']:.0%} FAR_prob={m_prob['FAR_prob']:.0%}")
        log.info(f"  Banda → Coverage={m_prob['Coverage80']:.1%} "
                 f"Width={m_prob['Width_m3s']:.1f}m³/s IS={m_prob['IS']:.4f}")

        # ENSO breakdown (1d)
        if target == "q_next_1d":
            sub  = pd.DataFrame({"obs": o_v, "p50": p50_v}, index=dte[valid.values])
            oni_d = oni.reindex(sub.index, method="ffill")
            wet   = sub.index.month.isin([11, 12, 1, 2, 3, 4])
            log.info(f"  Por temporada — húmeda NSE={_nse(sub[wet]['obs'].values, sub[wet]['p50'].values):.3f}"
                     f" | seca NSE={_nse(sub[~wet]['obs'].values, sub[~wet]['p50'].values):.3f}")
            for lbl, mk in [("Niño", oni_d > 0.5), ("Niña", oni_d < -0.5),
                             ("Neutral", (oni_d >= -0.5) & (oni_d <= 0.5))]:
                mv = mk.values
                if mv.sum() > 10:
                    log.info(f"  ENSO {lbl}: NSE={_nse(sub[mv]['obs'].values, sub[mv]['p50'].values):.3f} (n={mv.sum()})")

        for d, oo, p1, p5, p9 in zip(dte[valid.values],
                                      o_v/(Q_CONV*hmult), p10_v/(Q_CONV*hmult),
                                      p50_v/(Q_CONV*hmult), p90_v/(Q_CONV*hmult)):
            all_preds.append({"date": d, "q_obs": oo, "q_p10": p1,
                               "q_p50": p5, "q_p90": p9, "target": target})

        # ── Figura NHITS01: banda probabilística (1d) ──────────────────────
        if target == "q_next_1d":
            p = pd.DataFrame({
                "date": dte[valid.values], "obs":  o_v/Q_CONV,
                "p10":  p10_v/Q_CONV, "p50": p50_v/Q_CONV, "p90": p90_v/Q_CONV,
            }).sort_values("date")

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.fill_between(p["date"], p["p10"], p["p90"], alpha=0.25, color="#9b59b6",
                            label=f"Banda P10-P90 N-HiTS (cob.={m_prob['Coverage80']:.0%})")
            ax.plot(p["date"], p["p50"], color="#8e44ad", lw=1.3, label="N-HiTS P50")
            ax.plot(p["date"], p["obs"], color="#c0392b", lw=1.4, label="Q obs real")
            ax.axhline(Q90, ls=":", color="orange", alpha=0.8, label=f"Q90={Q90:.1f} m³/s")
            alert_days = p[p["p90"] > Q90]
            if len(alert_days):
                ax.scatter(alert_days["date"], [Q90+2]*len(alert_days),
                           marker="v", s=20, color="#f39c12", zorder=5,
                           label=f"Alerta P90 ({len(alert_days)} días, POD_prob={m_prob['POD_prob']:.0%})")
            ax.set_ylabel("Q (m³/s)"); ax.set_ylim(0)
            ax.set_title(
                f"N-HiTS — Pronóstico probabilístico Q +1 día  |  test 2024-2025\n"
                f"P50: NSE={m_det['NSE']:.3f}  KGE={m_det['KGE']:.3f}  "
                f"POD_det={m_det['POD_det']:.0%}  |  "
                f"POD_prob={m_prob['POD_prob']:.0%}  "
                f"Width={m_prob['Width_m3s']:.1f} m³/s",
                fontweight="bold", fontsize=10,
            )
            ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig.tight_layout()
            fig.savefig(FIG_DIR / "NHITS01_quantile_bands.png", bbox_inches="tight")
            plt.close(fig)
            log.info("  → NHITS01_quantile_bands.png")

    # ── Guardar ───────────────────────────────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "nhits_results.csv", index=False)
    pd.DataFrame(all_preds).to_csv(OUT_DIR / "nhits_predictions.csv", index=False)

    # ── Figura NHITS02: comparativa todos los modelos DL ──────────────────────
    rows_cmp = []
    # TFT v1 (Script 72)
    f72 = OUT_DIR / "transfer_metrics_full.csv"
    if f72.exists():
        df72 = pd.read_csv(f72)
        for _, r in df72[df72["model"] == "TFT-transfer"].iterrows():
            rows_cmp.append({"model": "TFT-v1", "target": r["target"],
                              "NSE": r["NSE"], "KGE": r["KGE"],
                              "POD": r.get("POD", np.nan), "J_alert": np.nan})
    # TFT v2 (Script 74)
    f74 = OUT_DIR / "tft_v2_results.csv"
    if f74.exists():
        df74 = pd.read_csv(f74)
        for _, r in df74.iterrows():
            rows_cmp.append({"model": "TFT-v2", "target": r["target"],
                              "NSE": r["NSE"], "KGE": r["KGE"],
                              "POD": r.get("POD_det", np.nan), "J_alert": r.get("J_alert", np.nan)})
    # N-HiTS (este script)
    for r in all_rows:
        rows_cmp.append({"model": "N-HiTS", "target": r["target"],
                          "NSE": r["NSE"], "KGE": r["KGE"],
                          "POD": r["POD_det"], "J_alert": r["J_alert"]})

    if rows_cmp:
        cdf = pd.DataFrame(rows_cmp)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cmp_metrics = ["NSE", "KGE", "POD", "J_alert"]
        model_colors = {"TFT-v1": "#e74c3c", "TFT-v2": "#27ae60", "N-HiTS": "#9b59b6"}

        for ax, tgt in zip(axes, ["q_next_1d", "q_sum_next_7d"]):
            sub = cdf[cdf["target"] == tgt]
            models = sub["model"].tolist()
            x = np.arange(len(cmp_metrics)); w = 0.8 / len(models)

            for i, (_, row) in enumerate(sub.iterrows()):
                vals = [float(row.get(m, np.nan)) for m in cmp_metrics]
                ax.bar(x + i*w - 0.4 + w/2, vals, w,
                       label=row["model"],
                       color=model_colors.get(row["model"], "#95a5a6"), alpha=0.85)

            ax.set_xticks(x); ax.set_xticklabels(cmp_metrics, fontsize=10)
            ax.set_ylim(-0.1, 1.15)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title("Q +1 día" if tgt == "q_next_1d" else "Q suma 7 días",
                         fontweight="bold")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylabel("Métrica")

        fig.suptitle("Comparativa modelos DL — TFT v1 / TFT v2 / N-HiTS\nTest Q obs real 2024-2025",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "NHITS02_comparison.png", bbox_inches="tight")
        plt.close(fig)
        log.info("  → NHITS02_comparison.png")

    log.info("\n=== RESUMEN FINAL N-HiTS ===")
    df_res = pd.DataFrame(all_rows)
    show = [c for c in ["target","NSE","NSE_sqrt","KGE","J_alert",
                         "POD_det","POD_prob","CSI","Coverage80","Width_m3s","N"]
            if c in df_res.columns]
    log.info(df_res[show].to_string(index=False))
    log.info("SCRIPT 76 COMPLETADO")


if __name__ == "__main__":
    main()
