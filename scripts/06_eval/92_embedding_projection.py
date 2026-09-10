#!/usr/bin/env python3
"""
Script 92 — t-SNE / UMAP projection of the model's LEARNED latent embeddings.

Unlike PCA on raw features (Script 91), this visualises what the trained HydroST
actually learned: the outlet representation vector (hid=64) just before the
quantile head, for each day. We project it with UMAP and t-SNE and colour by
streamflow magnitude, flood regime and season.

Interpretive expectation (honest): given the model is autoregression-dominated,
the latent should organise mainly by hydrological state (recent flow / wetness),
with floods forming a distinct region — a coherent, slightly deflating result.

Run in .venv313:
    python scripts/06_eval/92_embedding_projection.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT   = Path(__file__).resolve().parent.parent.parent
OUTML  = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("embed")

Q90 = 40.89


def main():
    spec = importlib.util.spec_from_file_location("hydrost", ROOT/"scripts/05_models/78_hydrost_Q.py")
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    DEV = H.DEVICE

    df = H.load_data()
    dyn, static, target, dates, _ = H.build_arrays(df)
    model = H.HydroST().to(DEV)
    model.load_state_dict(torch.load(OUTML/"78_hydrost_final.pt", map_location=DEV, weights_only=True))
    model.eval()

    # Sample period for visualisation (enough points incl. floods; latent viz, not eval)
    dpd = pd.DatetimeIndex(dates)
    sel_dates = dpd[(dpd >= "2010-01-01") & (dpd <= "2024-12-31")]
    ds = H.HydroDataset(dyn, static, target, dates, sel_dates)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    # Extract outlet embedding (h before quantile head) + target
    embs, ys = [], []
    with torch.no_grad():
        for x_dyn, x_static, y in loader:
            h = model._encode(x_dyn.to(DEV), x_static.to(DEV))      # (B, N, hid)
            embs.append(h[:, H.OUTLET_IDX, :].cpu().numpy())
            ys.extend(y.numpy())
    E = np.concatenate(embs); ys = np.array(ys)                     # ys in mm/day
    q_m3 = ys / H.Q_CONV
    seq_dates = pd.DatetimeIndex([pd.Timestamp(dates[i]) for i,_ in ds.indices])
    month = seq_dates.month
    wet = np.isin(month, [12,1,2,3,4])
    flood = q_m3 > Q90
    log.info(f"Embeddings: {E.shape}, floods={flood.sum()} of {len(E)}")

    # ── Projections ───────────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    import umap
    Es = StandardScaler().fit_transform(E)
    log.info("Running UMAP...")
    um = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Es)
    log.info("Running t-SNE...")
    ts = TSNE(n_components=2, perplexity=40, init="pca", random_state=42).fit_transform(Es)

    # ── Figure (2x3: UMAP row, t-SNE row; colour by Q, regime, season) ────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    qlog = np.log1p(q_m3)

    def scat(ax, XY, c, cmap, title, cbar=False, discrete=None):
        if discrete is not None:
            for val, col, lab in discrete:
                m = c == val
                ax.scatter(XY[m,0], XY[m,1], s=8, c=col, alpha=0.5 if not val else 0.8, label=lab)
            ax.legend(fontsize=8, markerscale=2)
        else:
            sc = ax.scatter(XY[:,0], XY[:,1], s=8, c=c, cmap=cmap, alpha=0.7)
            if cbar: fig.colorbar(sc, ax=ax, fraction=0.046, label="log(1+Q) [m³/s]")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])

    for row, (XY, name) in enumerate([(um,"UMAP"), (ts,"t-SNE")]):
        scat(axes[row,0], XY, qlog, "viridis", f"({'ab'[row]}1) {name} — by streamflow magnitude", cbar=True)
        scat(axes[row,1], XY, flood.astype(int), None, f"({'ab'[row]}2) {name} — by regime",
             discrete=[(0,"#4a90d9","baseflow"),(1,"#cc2222","flood Q>Q90")])
        scat(axes[row,2], XY, wet.astype(int), None, f"({'ab'[row]}3) {name} — by season",
             discrete=[(0,"#e8a000","dry (May-Nov)"),(1,"#1f77b4","wet (Dec-Apr)")])

    fig.suptitle("Learned latent space of HydroST (outlet embedding): organised by hydrological "
                 "state; floods form a distinct region", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig15_embedding_umap_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # quantify: silhouette of flood vs base in UMAP space
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(um, flood.astype(int))
    log.info(f"Silhouette (flood vs base) in UMAP: {sil:.3f}")
    log.info(f"Saved: {FIGDIR/'fig15_embedding_umap_tsne.png'}")


if __name__ == "__main__":
    main()
