#!/usr/bin/env python3
"""
Script 112 — Grand tour COMPLETO sobre ventanas MULTIVARIADAS (todas las forzantes).

Cada muestra = ventana de 60 días × 12 variables (lluvia, temp, PET, caudal, API,
SPI-30/90, déficit, ONI, humedad de suelo, Niño costero). Model-agnostic. Incluye los
8 métodos (como fig25): PCA, KPCA-RBF, t-SNE, UMAP, Isomap, LLE, DTW->MDS (multivariado),
TS2Vec-contrastivo. DTW/TS2Vec requieren secuencias -> por eso usamos ventanas, no vectores
estáticos. Color = régimen; silhouette en cada título.

Run in .venv313:
    python scripts/06_eval/112_grandtour_fullfeatures.py
"""
import logging, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

ROOT=Path(__file__).resolve().parent.parent.parent; OUTML=ROOT/"outputs/ml_Q"; FIG=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("gt2")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_CONV=86.4/3062.62; Q90=40.89; L=60
R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r110",ROOT/"scripts/06_eval/110_ts_representation.py"))
importlib.util.spec_from_file_location("r110",ROOT/"scripts/06_eval/110_ts_representation.py").loader.exec_module(R)

FEATS=["pr_mm","tmax_c","tmin_c","pet_mm","q_mm","api","spi_30d","spi_90d","water_deficit_30d","oni_index"]

def dtw_mv(A,B):  # A,B: (C,L) multivariate DTW (Euclidean over channels)
    L1,L2=A.shape[1],B.shape[1]; D=np.full((L1+1,L2+1),np.inf); D[0,0]=0
    for i in range(1,L1+1):
        for j in range(1,L2+1):
            c=np.linalg.norm(A[:,i-1]-B[:,j-1]); D[i,j]=c+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[L1,L2]

def main():
    d7=pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv",parse_dates=["date"])
    o=d7[d7.entity_id=="sub_634"].set_index("date").sort_index()
    sw=pd.read_parquet(OUTML/"era5_swvl_per_entity.parquet")
    if "date" not in sw.columns: sw=sw.reset_index()
    enso=pd.read_csv(ROOT/"data/silver/enso/S3b_enso_coastal_global.csv",parse_dates=["date"]).set_index("date")
    X=o[FEATS].copy(); X["swvl4"]=sw.groupby("date")["swvl4"].mean().reindex(o.index); X["coastal"]=enso["coastal"].reindex(o.index).ffill()
    cols=FEATS+["swvl4","coastal"]; dts=o.index
    M=X[cols].values.T.astype(float)   # (C, T)
    mu=np.nanmean(M,1,keepdims=True); sd=np.nanstd(M,1,keepdims=True)+1e-6; Mz=np.nan_to_num((M-mu)/sd)
    q=(o["q_mm"]/Q_CONV).values
    idx=np.array([i for i in range(L,len(dts)) if dts[i]>=pd.Timestamp("2015-01-01")])
    Wv=np.stack([Mz[:,i-L:i] for i in idx]).astype(np.float32)   # (N, C, L)
    flood=q[idx]>Q90; Xflat=Wv.reshape(len(Wv),-1)              # (N, C*L)
    log.info(f"Ventanas multivariadas: {Wv.shape} (C={len(cols)}) crecidas={int(flood.sum())}")
    def sil(e):
        try: return silhouette_score(e,flood.astype(int))
        except Exception: return np.nan

    emb={}
    emb["PCA"]=(PCA(2).fit_transform(Xflat),"PC1","PC2")
    emb["KPCA-RBF"]=(KernelPCA(2,kernel="rbf",gamma=1.0/Xflat.shape[1]).fit_transform(Xflat),"KPC1","KPC2")
    emb["t-SNE"]=(TSNE(2,perplexity=40,init="pca",random_state=42).fit_transform(Xflat),"dim1","dim2")
    emb["UMAP"]=(umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(Xflat),"dim1","dim2")
    emb["Isomap"]=(Isomap(n_components=2,n_neighbors=15).fit_transform(Xflat),"dim1","dim2")
    emb["LLE"]=(LocallyLinearEmbedding(n_components=2,n_neighbors=15).fit_transform(Xflat),"dim1","dim2")
    # DTW multivariado -> MDS (subset balanceado)
    rng=np.random.default_rng(0); fl=np.where(flood)[0]; bs=np.where(~flood)[0]
    subi=np.concatenate([fl[:80],rng.permutation(bs)[:80]]); n=len(subi); Dm=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): Dm[i,j]=Dm[j,i]=dtw_mv(Wv[subi[i]],Wv[subi[j]])
    emb["DTW->MDS"]=(MDS(2,dissimilarity="precomputed",random_state=0,normalized_stress="auto").fit_transform(Dm),"dim1","dim2",flood[subi])
    # TS2Vec contrastivo (12 canales)
    torch.manual_seed(0); enc=R.TSEncoder(len(cols)).to(DEV); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt=torch.tensor(Wv,device=DEV); enc.train()
    for ep in range(60):
        pm=torch.randperm(len(Xt))
        for k in range(0,len(Xt),256):
            b=Xt[pm[k:k+256]]; z1,z2=enc(R.augment(b)),enc(R.augment(b))
            opt.zero_grad(); R.nt_xent(z1,z2).backward(); opt.step()
    enc.eval()
    with torch.no_grad(): z=enc(Xt).cpu().numpy()
    emb["TS2Vec"]=(umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(z),"dim1","dim2")

    order=["PCA","KPCA-RBF","t-SNE","UMAP","Isomap","LLE","DTW->MDS","TS2Vec"]
    fig,axes=plt.subplots(2,4,figsize=(18,9)); axes=axes.ravel(); sils={}
    for ax,name in zip(axes,order):
        E=emb[name]
        if name=="DTW->MDS":
            P,xl,yl,fm=E; ax.scatter(P[~fm,0],P[~fm,1],s=8,c="#4a90d9",alpha=0.4); ax.scatter(P[fm,0],P[fm,1],s=18,c="#cc2222",alpha=0.8)
            s=silhouette_score(P,fm.astype(int)) if len(np.unique(fm))>1 else np.nan
        else:
            P,xl,yl=E; ax.scatter(P[~flood,0],P[~flood,1],s=6,c="#4a90d9",alpha=0.3,label="base"); ax.scatter(P[flood,0],P[flood,1],s=14,c="#cc2222",alpha=0.7,label="crecida")
            s=sil(P)
        sils[name]=s; ax.set_title(f"{name}  (sil={s:.2f})",fontsize=10); ax.set_xlabel(xl,fontsize=7); ax.set_ylabel(yl,fontsize=7); ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=8)
    fig.suptitle("Grand tour COMPLETO — ventanas multivariadas (12 forzantes × 60 d, 2015-2025). 8 métodos, "
                 "color=régimen. PCA/KPCA ejes=varianza; resto sin unidades físicas",fontweight="bold",fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"fig26_grandtour_fullfeatures.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    log.info("Silhouettes: "+", ".join(f"{n}={v:.2f}" for n,v in sils.items()))
    log.info("Saved: fig26_grandtour_fullfeatures.png (8 métodos, multivariado)")

if __name__=="__main__":
    main()
