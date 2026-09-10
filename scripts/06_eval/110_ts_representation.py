#!/usr/bin/env python3
"""
Script 110 — Time-series-native representation learning (complements UMAP/t-SNE
on model latents from Script 92).

Why: vanilla UMAP/t-SNE on raw features use Euclidean distance, which ignores
temporal warping/dynamics. TS-native methods respect it:
  (A) DTW clustering of flood hydrographs (elastic distance) -> event typology.
  (B) TS2Vec-inspired self-supervised contrastive encoder (dilated 1D-CNN + NT-Xent
      on augmented views) -> learned window embeddings -> UMAP, colored by regime.
      (Model-agnostic representation, GPU; complements the supervised HydroST latent.)

Refs: Berndt & Clifford 1994 (DTW); Cuturi & Blondel 2017 (soft-DTW);
Yue et al. 2022 (TS2Vec); Chen et al. 2020 (SimCLR / NT-Xent).

Run in .venv313 (GPU):
    python scripts/06_eval/110_ts_representation.py
"""
import logging
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"; FIG=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("tsrep")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_CONV=86.4/3062.62; Q90=40.89; L=60


def dtw(a,b):
    n,m=len(a),len(b); D=np.full((n+1,m+1),np.inf); D[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            c=abs(a[i-1]-b[j-1]); D[i,j]=c+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[n,m]


class TSEncoder(nn.Module):
    """Dilated 1D-CNN encoder (TS2Vec-style backbone)."""
    def __init__(self, c_in, hid=64, out=64):
        super().__init__()
        self.inp=nn.Conv1d(c_in,hid,1)
        self.blocks=nn.ModuleList([nn.Conv1d(hid,hid,3,padding=d,dilation=d) for d in [1,2,4,8]])
        self.head=nn.Linear(hid,out)
    def forward(self,x):            # x (B,C,L)
        h=self.inp(x)
        for b in self.blocks: h=h+F.gelu(b(h))
        h=h.max(dim=-1).values      # global max pool over time
        return F.normalize(self.head(h),dim=-1)

def augment(x):
    # jitter + scaling + random time masking (two-view SSL augmentations)
    x=x+0.1*torch.randn_like(x)
    x=x*(1+0.1*torch.randn(x.size(0),x.size(1),1,device=x.device))
    B,C,T=x.shape; m=(torch.rand(B,1,T,device=x.device)>0.15).float()
    return x*m

def nt_xent(z1,z2,tau=0.2):
    B=z1.size(0); z=torch.cat([z1,z2],0)
    sim=z@z.t()/tau; sim.fill_diagonal_(-1e9)
    tgt=torch.arange(B,device=z.device); tgt=torch.cat([tgt+B,tgt])
    return F.cross_entropy(sim,tgt)


def main():
    d7=pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv",parse_dates=["date"])
    o=d7[d7.entity_id=="sub_634"].set_index("date").sort_index()
    pr=d7.groupby("date")["pr_mm"].mean()
    q=(o["q_mm"]/Q_CONV); api=o["api"]
    obs=pd.read_csv(ROOT/"data/silver/snirh/S1_snirh_daily_q.csv",parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    dts=q.index

    # multivariate windows [q,pr,api], standardized
    feats=np.stack([q.values, pr.reindex(dts).values, api.values],0)  # (3, T)
    mu=np.nanmean(feats,1,keepdims=True); sd=np.nanstd(feats,1,keepdims=True)+1e-6
    feats=np.nan_to_num((feats-mu)/sd)
    idxs=[i for i in range(L,len(dts))]
    X=np.stack([feats[:,i-L:i] for i in idxs]).astype(np.float32)     # (N,3,L)
    qtgt=q.values[idxs]; flood=qtgt>Q90; wet=np.isin(pd.DatetimeIndex(dts[idxs]).month,[12,1,2,3,4])
    log.info(f"Windows: {X.shape}  floods={int(flood.sum())}")

    # ── (A) DTW clustering of flood hydrographs ───────────────────────────────
    fl=np.where(flood)[0]; fl=fl[:40]
    shapes=[q.values[idxs[p]-15:idxs[p]+1] for p in fl]  # global idx = idxs[p]; 16-day rising-to-peak
    shapes=[(s-s.min())/(s.max()-s.min()+1e-6) for s in shapes]  # normalized shape
    n=len(shapes); Dm=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): Dm[i,j]=Dm[j,i]=dtw(shapes[i],shapes[j])
    from scipy.cluster.hierarchy import linkage,fcluster
    from scipy.spatial.distance import squareform
    Z=linkage(squareform(Dm),method="average"); cl=fcluster(Z,t=3,criterion="maxclust")
    log.info(f"DTW flood-event clusters (k=3): sizes={np.bincount(cl)[1:].tolist()}")

    # ── (B) TS2Vec-style contrastive encoder ──────────────────────────────────
    torch.manual_seed(0)
    enc=TSEncoder(3).to(DEV); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt=torch.tensor(X,device=DEV)
    enc.train()
    for ep in range(80):
        perm=torch.randperm(len(Xt))
        for k in range(0,len(Xt),256):
            b=Xt[perm[k:k+256]]
            z1,z2=enc(augment(b)),enc(augment(b))
            opt.zero_grad(); nt_xent(z1,z2).backward(); opt.step()
    enc.eval()
    with torch.no_grad(): emb=enc(Xt).cpu().numpy()
    import umap
    from sklearn.metrics import silhouette_score
    U=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(emb)
    sil=silhouette_score(U,flood.astype(int))
    log.info(f"TS2Vec-style embedding: dim={emb.shape[1]}  silhouette(flood/base)={sil:.3f}")

    # ── Figure (lean) ─────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(15,5.5))
    # (a) DTW cluster mean shapes
    for c in np.unique(cl):
        members=[shapes[i] for i in range(n) if cl[i]==c]
        ml=min(len(s) for s in members); arr=np.stack([s[-ml:] for s in members])
        ax[0].plot(np.arange(-ml+1,1),arr.mean(0),lw=2.5,label=f"tipo {c} (n={len(members)})")
        for s in members: ax[0].plot(np.arange(-len(s)+1,1),s,alpha=0.15,color="gray")
    ax[0].set_title("(A) Tipología de crecidas por DTW (forma normalizada del hidrograma)")
    ax[0].set_xlabel("días antes del pico"); ax[0].set_ylabel("caudal normalizado"); ax[0].legend(fontsize=8)
    # (b) TS2Vec-style UMAP by regime
    ax[1].scatter(U[~flood,0],U[~flood,1],s=6,c="#4a90d9",alpha=0.25,label="base")
    ax[1].scatter(U[flood,0],U[flood,1],s=16,c="#cc2222",alpha=0.7,label="crecida Q>Q90")
    ax[1].set_title(f"(B) Embedding TS2Vec-style (self-sup.) → UMAP\nsilhouette crecida/base={sil:.2f}")
    ax[1].set_xticks([]);ax[1].set_yticks([]);ax[1].legend(fontsize=8)
    fig.suptitle("Representación NATIVA de series temporales (DTW + contrastivo TS2Vec) — complementa el latente del modelo",
                 fontweight="bold",fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"fig24_ts_representation.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    log.info(f"Saved: fig24_ts_representation.png")


if __name__=="__main__":
    main()
