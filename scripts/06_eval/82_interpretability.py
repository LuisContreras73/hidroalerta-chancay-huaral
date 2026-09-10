#!/usr/bin/env python3
"""
Script 82 — HydroST interpretability: how the model "thinks".

Three physically-interpretable analyses on the trained HydroST model:

  1. SPATIAL ATTENTION  — which of the 9 sub-basins the outlet representation
     attends to, split by flow regime (flood vs baseflow days).
     Hypothesis: high-altitude basins (sub_649/655/656, 79% of flow) dominate.

  2. TEMPORAL SALIENCY  — integrated gradients of P50 w.r.t. the input window,
     producing a (lag-day × feature) importance map. Reveals which variables
     and how many days back drive the forecast.

  3. RAINFALL IMPULSE RESPONSE — inject a unit precipitation pulse at each lag
     and measure the change in predicted Q. This recovers the model's *learned
     unit hydrograph*: response time, peak lag, and recession behaviour, which
     we compare against the catchment's physical concentration time.

Runs on the available device (CPU ok). Loads 78_hydrost_final.pt.

    python scripts/06_eval/82_interpretability.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs/ml_Q"
FIG_DIR = ROOT / "outputs/qa_satellite"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("interpret")

spec = importlib.util.spec_from_file_location("hydrost", ROOT / "scripts/05_models/78_hydrost_Q.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

DEVICE = torch.device("cpu")   # keep GPU free; model is tiny
H.DEVICE = DEVICE

# Entity metadata (elevation, sorted low->high) for labelling
ENT_ELEV = {
    "sub_634": 521, "sub_640": 1480, "sub_653": 1664, "sub_650": 2841,
    "sub_641": 3470, "sub_646": 3812, "sub_655": 4058, "sub_656": 4466, "sub_649": 4507,
}
ENT_LABEL = {e: f"{e}\n{ENT_ELEV[e]}m" for e in H.ENTITIES}


# ── Spatial attention extraction ──────────────────────────────────────────────

def encode_with_attention(model, x_dyn, x_static):
    """Replay HydroST._encode but capture spatial attention weights."""
    B, N, L, _ = x_dyn.shape
    s_emb = model.static_enc(x_static)
    s_exp = s_emb.unsqueeze(2).expand(-1, -1, L, -1)
    x = torch.cat([x_dyn, s_exp], dim=-1)
    x = model.input_proj(x)
    x_flat = x.reshape(B * N, L, -1)
    out, _ = model.lstm(x_flat)
    h_last = out[:, -1, :].reshape(B, N, -1)
    # attention with weights (averaged over heads)
    _, attn_w = model.spatial_attn(h_last, h_last, h_last,
                                   need_weights=True, average_attn_weights=True)
    return attn_w   # (B, N, N): attn_w[b, i, j] = how much query i attends to key j


def spatial_attention_analysis(model, dyn, static, target, dates):
    dates_pd = pd.DatetimeIndex(dates)
    test_mask = dates_pd >= H.TEST_START
    ds = H.HydroDataset(dyn, static, target, dates, dates_pd[test_mask])

    # Classify each sequence by flow regime (target Q)
    Q90_mm = H.Q90_MM
    attn_flood, attn_base = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x_dyn, x_static, y = ds[i]
            aw = encode_with_attention(model, x_dyn.unsqueeze(0), x_static.unsqueeze(0))
            outlet_attn = aw[0, H.OUTLET_IDX, :].numpy()   # how outlet attends to each basin
            if float(y) > Q90_mm:
                attn_flood.append(outlet_attn)
            else:
                attn_base.append(outlet_attn)

    attn_flood = np.array(attn_flood).mean(0) if attn_flood else np.zeros(9)
    attn_base  = np.array(attn_base).mean(0)

    # Sort entities by elevation for display
    order = sorted(range(9), key=lambda i: ENT_ELEV[H.ENTITIES[i]])
    labels = [ENT_LABEL[H.ENTITIES[i]] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    xpos = np.arange(9)
    ax.bar(xpos - 0.2, attn_base[order],  0.4, label=f"Baseflow days (n={len(attn_base) if attn_base.ndim else 0})",
           color="#4a90d9", alpha=0.85)
    ax.bar(xpos + 0.2, attn_flood[order], 0.4, label="Flood days (Q>Q90)", color="#cc2222", alpha=0.85)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Attention weight from outlet (sub_634)")
    ax.set_title("HydroST spatial attention: which sub-basins the outlet attends to\n"
                 "(basins ordered low→high elevation)")
    ax.axhline(1/9, color="gray", ls="--", lw=1, label="uniform (1/9)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_spatial_attention.png", dpi=150)
    plt.close(fig)
    log.info("Saved: 04_spatial_attention.png")

    # Log the ranking
    log.info("\nSpatial attention (flood days), high→low:")
    rank = np.argsort(-attn_flood)
    for r in rank:
        log.info(f"  {H.ENTITIES[r]:10s} ({ENT_ELEV[H.ENTITIES[r]]:4d}m): {attn_flood[r]:.3f}")
    return attn_flood, attn_base


# ── Temporal saliency (integrated gradients) ──────────────────────────────────

def temporal_saliency(model, dyn, static, target, dates, n_samples=150):
    dates_pd = pd.DatetimeIndex(dates)
    test_mask = dates_pd >= H.TEST_START
    ds = H.HydroDataset(dyn, static, target, dates, dates_pd[test_mask])

    # Focus on flood sequences (most decision-relevant)
    flood_idx = [i for i in range(len(ds)) if float(ds[i][2]) > H.Q90_MM]
    sel = flood_idx[:n_samples] if flood_idx else list(range(min(n_samples, len(ds))))
    log.info(f"Temporal saliency on {len(sel)} flood sequences")

    sal_accum = np.zeros((H.ENC, H.N_DYN))
    steps = 16
    for i in sel:
        x_dyn, x_static, _ = ds[i]
        x_dyn = x_dyn.unsqueeze(0); x_static = x_static.unsqueeze(0)
        baseline = torch.zeros_like(x_dyn)
        # Integrated gradients of P50 (output index 1) w.r.t. x_dyn
        ig = torch.zeros_like(x_dyn)
        for a in np.linspace(0, 1, steps):
            xi = (baseline + a * (x_dyn - baseline)).clone().requires_grad_(True)
            out = model(xi, x_static)[0, 1]   # P50
            grad = torch.autograd.grad(out, xi)[0]
            ig += grad
        ig = (ig / steps) * (x_dyn - baseline)
        # average over entities → (L, n_dyn)
        sal_accum += ig[0].abs().mean(0).detach().numpy()

    sal = sal_accum / len(sel)
    # Normalise per feature for visibility
    sal_norm = sal / (sal.sum(axis=0, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(sal_norm.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(H.N_DYN)); ax.set_yticklabels(H.DYN_COLS, fontsize=9)
    # x axis: lag days (rightmost = most recent)
    lags = np.arange(H.ENC)
    ax.set_xticks(np.arange(0, H.ENC, 5))
    ax.set_xticklabels([f"-{H.ENC-1-t}" for t in np.arange(0, H.ENC, 5)], fontsize=8)
    ax.set_xlabel("Lag (days before forecast; 0 = most recent day)")
    ax.set_title("HydroST temporal saliency (integrated gradients on P50, flood days)\n"
                 "brighter = larger influence on tomorrow's Q forecast")
    fig.colorbar(im, ax=ax, fraction=0.025, label="normalised |attribution|")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_temporal_saliency.png", dpi=150)
    plt.close(fig)
    log.info("Saved: 05_temporal_saliency.png")

    # Feature ranking (total attribution)
    feat_imp = sal.sum(0)
    rank = np.argsort(-feat_imp)
    log.info("\nFeature importance (total saliency), high→low:")
    for r in rank:
        log.info(f"  {H.DYN_COLS[r]:18s}: {feat_imp[r]:.4f}")

    # Effective memory: cumulative saliency over lags (summed over features)
    lag_imp = sal.sum(1)  # (L,)
    lag_imp_recent = lag_imp[::-1]  # index 0 = most recent
    cum = np.cumsum(lag_imp_recent) / lag_imp_recent.sum()
    d50 = int(np.argmax(cum >= 0.5))
    d90 = int(np.argmax(cum >= 0.9))
    log.info(f"\nEffective temporal memory: 50% of attribution in last {d50} days, 90% in last {d90} days")
    return sal, feat_imp


# ── Rainfall impulse response (learned unit hydrograph) ───────────────────────

def impulse_response(model, dyn, static, target, dates, n_base=100, pulse=3.0):
    """Add a precip pulse (in standardized units) at each lag and measure ΔP50."""
    dates_pd = pd.DatetimeIndex(dates)
    test_mask = dates_pd >= H.TEST_START
    ds = H.HydroDataset(dyn, static, target, dates, dates_pd[test_mask])
    pr_idx = H.DYN_COLS.index("pr_mm")

    sel = list(range(min(n_base, len(ds))))
    model.eval()

    delta_by_lag = np.zeros(H.ENC)
    with torch.no_grad():
        for i in sel:
            x_dyn, x_static, _ = ds[i]
            x_dyn = x_dyn.unsqueeze(0); x_static = x_static.unsqueeze(0)
            base_p50 = model(x_dyn, x_static)[0, 1].item()
            for t in range(H.ENC):
                xp = x_dyn.clone()
                xp[0, :, t, pr_idx] += pulse   # pulse at lag-position t, all entities
                p50 = model(xp, x_static)[0, 1].item()
                delta_by_lag[t] += (p50 - base_p50)
    delta_by_lag /= len(sel)
    # Convert to m3/s response and orient by lag-before-forecast
    delta_m3 = delta_by_lag / H.Q_CONV
    lag_before = np.arange(H.ENC)[::-1]  # day at position t is (ENC-1-t) days before
    resp = delta_m3[::-1]  # index 0 = most recent day

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(H.ENC), resp, "-o", ms=3, color="#1f77b4")
    ax.fill_between(np.arange(H.ENC), 0, resp, alpha=0.2, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.8)
    peak_lag = int(np.argmax(resp))
    ax.axvline(peak_lag, color="red", ls="--", lw=1.2,
               label=f"peak response at lag −{peak_lag}d")
    ax.set_xlabel("Lag before forecast (days; 0 = most recent)")
    ax.set_ylabel("Δ predicted Q  (m³/s per +%.0fσ rain pulse)" % pulse)
    ax.set_title("HydroST learned rainfall→runoff impulse response\n"
                 "(the model's internalised unit hydrograph)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, min(30, H.ENC))   # zoom to first 30 days
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_impulse_response.png", dpi=150)
    plt.close(fig)
    log.info("Saved: 06_impulse_response.png")

    log.info(f"\nImpulse response: peak at lag −{peak_lag}d, "
             f"response(0d)={resp[0]:.2f}, response(1d)={resp[1]:.2f}, "
             f"response(7d)={resp[7]:.2f} m³/s")
    return resp


def main():
    df = H.load_data()
    dyn, static, target, dates, _ = H.build_arrays(df)

    model = H.HydroST().to(DEVICE)
    model.load_state_dict(torch.load(OUT_DIR / "78_hydrost_final.pt", map_location=DEVICE, weights_only=True))
    log.info("Loaded 78_hydrost_final.pt on CPU")

    log.info("\n=== 1. SPATIAL ATTENTION ===")
    spatial_attention_analysis(model, dyn, static, target, dates)

    log.info("\n=== 2. TEMPORAL SALIENCY ===")
    temporal_saliency(model, dyn, static, target, dates)

    log.info("\n=== 3. RAINFALL IMPULSE RESPONSE ===")
    impulse_response(model, dyn, static, target, dates)

    log.info("\nDone. Figures: 04_spatial_attention, 05_temporal_saliency, 06_impulse_response")


if __name__ == "__main__":
    main()
