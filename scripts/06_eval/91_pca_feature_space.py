#!/usr/bin/env python3
"""
Script 91 — PCA of the feature space (supporting result, NOT a contribution).

Quantifies the redundancy/effective dimensionality of the 41 predictor features
used by the TFT, reinforcing the multicollinearity + parsimony narrative (why
simple models and persistence are hard to beat). Descriptive analysis.

Panels:
  (a) Cumulative explained variance (scree) -> effective dimensionality.
  (b) PC1 vs PC2 projection colored by flow regime (flood Q>Q90 vs base).
  (c) PC1 & PC2 loadings -> physical interpretation of the dominant modes.

Run in .venv:
    python scripts/06_eval/91_pca_feature_space.py
"""

import importlib.util
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT   = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "generacion_paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pca")

Q90 = 40.89
Q_CONV = 86.4/3062.62


def nice(code):
    """Short readable feature label (var + sub-basin real name)."""
    REAL = {"basin":"basin","sub_649":"Alto Chancay","sub_655":"Carac","sub_656":"Baños"}
    VAR = {"pr_mm":"Precip","tmax_c":"Tmax","tmin_c":"Tmin","pet_mm":"PET","api":"API",
           "spi_30d":"SPI-30","spi_90d":"SPI-90","water_deficit_30d":"Wdef"}
    if code.endswith("_basin"):
        base, loc = code[:-6], "basin"
    else:
        for sb in ("sub_649","sub_655","sub_656"):
            if code.endswith("_"+sb):
                base, loc = code[:-len(sb)-1], sb; break
        else:
            return code
    return f"{VAR.get(base, base)} ({REAL[loc]})"


def main():
    spec = importlib.util.spec_from_file_location("t", ROOT/"scripts/05_models/74_tft_v2_quantile_Q.py")
    T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    D6 = pd.read_csv(T.D6_CSV, parse_dates=["date"]); df = T.build_wide(D6)

    feat = [c for c in df.columns if c not in ["q_next_1d","q_sum_next_7d"]]
    X = df[feat].fillna(0).values
    mu, sd = X.mean(0), X.std(0)+1e-9
    Xs = (X - mu)/sd
    pca = PCA().fit(Xs)
    Z = pca.transform(Xs)
    evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
    k90 = int(np.argmax(cum>=0.9))+1
    log.info(f"n_features={len(feat)}  PC1={evr[0]:.3f}  PC1-3={cum[2]:.3f}  90% at {k90} PCs")

    # flow regime for coloring
    q = (df["q_next_1d"].values)             # mm/day (target)
    flood = q > (Q90*Q_CONV)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) scree / cumulative
    ax = axes[0]
    ax.bar(range(1,16), evr[:15], color="#9ecae1", alpha=0.9, label="individual")
    ax.plot(range(1,16), cum[:15], "o-", color="#08519c", label="cumulative")
    ax.axhline(0.9, color="red", ls="--", lw=1); ax.axvline(k90, color="red", ls=":", lw=1)
    ax.annotate(f"{k90} PCs → 90%", (k90+0.3, 0.6), color="red", fontsize=9)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"(a) Effective dimensionality\n41 features → ~{k90} PCs (PC1={evr[0]:.0%})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) PC1-PC2 projection colored by regime
    ax = axes[1]
    ax.scatter(Z[~flood,0], Z[~flood,1], s=6, c="#4a90d9", alpha=0.25, label="baseflow")
    ax.scatter(Z[flood,0],  Z[flood,1],  s=14, c="#cc2222", alpha=0.7, label="flood (Q>Q90)")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%}) — wetness/storage")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.set_title("(b) Feature-space projection by flow regime\nfloods occupy the high-wetness tail of PC1", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) loadings PC1 & PC2
    ax = axes[2]
    l1 = pd.Series(pca.components_[0], index=feat)
    top = l1.abs().sort_values(ascending=False).head(10).index
    ax.barh(range(len(top)), [pca.components_[0][feat.index(f)] for f in top],
            color="#08519c", alpha=0.85)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([nice(f) for f in top], fontsize=8)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("PC1 loading"); ax.set_title("(c) PC1 loadings\n(dominant mode = catchment wetness)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("PCA of the predictor space: strong redundancy supports the parsimony / "
                 "persistence-dominance findings", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig14_pca_feature_space.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {FIGDIR/'fig14_pca_feature_space.png'}")
    # quick separation stat: how much higher is PC1 on flood days
    log.info(f"PC1 mean — flood={Z[flood,0].mean():.2f} vs base={Z[~flood,0].mean():.2f}")


if __name__ == "__main__":
    main()
