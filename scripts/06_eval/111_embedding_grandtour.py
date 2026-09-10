#!/usr/bin/env python3
"""
Script 111 — Grand tour MODEL-AGNOSTIC de embeddings de series.

Corrección: NO usar el latente de HydroST (uno de los peores modelos). Estas figuras
caracterizan la ESTRUCTURA INTRÍNSECA de la serie de caudal (ventanas/eventos), no el
interior de un modelo. Mensaje defendible e independiente del modelo: qué familia de
embedding revela mejor la estructura de régimen (crecida vs base) en los DATOS.

Mismas ventanas de caudal [q, pr, api] (2015-2025, todas), varios métodos:
PCA, features->PCA, t-SNE, UMAP, Isomap, LLE, DTW->MDS, TS2Vec-contrastivo.
Color = régimen; silhouette (crecida/base) en cada título.

Run in .venv313 (GPU):
    python scripts/06_eval/111_embedding_grandtour.py
"""
import logging, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

ROOT=Path(__file__).resolve().parent.parent.parent; FIG=ROOT/"generacion_paper/figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"); log=logging.getLogger("gt")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_CONV=86.4/3062.62; Q90=40.89; L=60

R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r110",ROOT/"scripts/06_eval/110_ts_representation.py"))
importlib.util.spec_from_file_location("r110",ROOT/"scripts/06_eval/110_ts_representation.py").loader.exec_module(R)

def tsfeat(w):
    m=w.mean(); s=w.std()+1e-6; sl=np.polyfit(np.arange(len(w)),w,1)[0]
    ac=np.corrcoef(w[:-1],w[1:])[0,1] if len(w)>2 else 0
    p=np.abs(np.fft.rfft(w-m))**2; p=p/(p.sum()+1e-9); se=-np.sum(p*np.log(p+1e-12))
    return [m,s,sl,ac,s/(m+1e-6),np.argmax(w)/len(w),se]

def main():
    d7=pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv",parse_dates=["date"])
    o=d7[d7.entity_id=="sub_634"].set_index("date").sort_index()
    pr=d7.groupby("date")["pr_mm"].mean(); q=(o["q_mm"]/Q_CONV); api=o["api"]; dts=q.index
    F=np.stack([q.values,pr.reindex(dts).values,api.values],0)
    mu=np.nanmean(F,1,keepdims=True); sd=np.nanstd(F,1,keepdims=True)+1e-6; Fz=np.nan_to_num((F-mu)/sd)
    # todas las ventanas 2015-2025 (model-agnostic; serie continua)
    idx=np.array([i for i in range(L,len(dts)) if dts[i]>=pd.Timestamp("2015-01-01")])
    W=np.stack([Fz[:,i-L:i] for i in idx]).astype(np.float32)   # (N,3,L)
    qend=q.values[idx]; flood=qend>Q90; Xflat=W.reshape(len(W),-1)
    log.info(f"Ventanas 2015-2025: {W.shape}  crecidas={int(flood.sum())}")
    def sil(e):
        try: return silhouette_score(e,flood.astype(int))
        except Exception: return np.nan

    emb={}
    emb["PCA"]=PCA(2).fit_transform(Xflat)
    Feat=StandardScaler().fit_transform(np.array([tsfeat(W[i,0]) for i in range(len(W))]))
    emb["Features->PCA"]=PCA(2).fit_transform(Feat)
    emb["t-SNE"]=TSNE(2,perplexity=40,init="pca",random_state=42).fit_transform(Xflat)
    emb["UMAP"]=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(Xflat)
    emb["Isomap"]=Isomap(n_components=2,n_neighbors=15).fit_transform(Xflat)
    emb["LLE"]=LocallyLinearEmbedding(n_components=2,n_neighbors=15).fit_transform(Xflat)
    # DTW->MDS (subset balanceado por costo)
    rng=np.random.default_rng(0); fl=np.where(flood)[0]; bs=np.where(~flood)[0]
    subi=np.concatenate([fl[:90],rng.permutation(bs)[:90]]); qs=[W[i,0] for i in subi]
    n=len(qs); Dm=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): Dm[i,j]=Dm[j,i]=R.dtw(qs[i],qs[j])
    emb["DTW->MDS"]=(MDS(2,dissimilarity="precomputed",random_state=0,normalized_stress="auto").fit_transform(Dm), flood[subi])
    # TS2Vec contrastivo (model-agnostic)
    torch.manual_seed(0); enc=R.TSEncoder(3).to(DEV); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt=torch.tensor(W,device=DEV); enc.train()
    for ep in range(60):
        pm=torch.randperm(len(Xt))
        for k in range(0,len(Xt),256):
            b=Xt[pm[k:k+256]]; z1,z2=enc(R.augment(b)),enc(R.augment(b))
            opt.zero_grad(); R.nt_xent(z1,z2).backward(); opt.step()
    enc.eval()
    with torch.no_grad(): z=enc(Xt).cpu().numpy()
    emb["TS2Vec (contrastivo)"]=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(z)

    order=["PCA","Features->PCA","t-SNE","UMAP","Isomap","LLE","DTW->MDS","TS2Vec (contrastivo)"]
    fig,axes=plt.subplots(2,4,figsize=(18,9)); axes=axes.ravel()
    sils={}
    for ax,name in zip(axes,order):
        E=emb[name]
        if name=="DTW->MDS":
            E2,fm=E; ax.scatter(E2[~fm,0],E2[~fm,1],s=8,c="#4a90d9",alpha=0.4); ax.scatter(E2[fm,0],E2[fm,1],s=18,c="#cc2222",alpha=0.8)
            s=silhouette_score(E2,fm.astype(int)) if len(np.unique(fm))>1 else np.nan
        else:
            ax.scatter(E[~flood,0],E[~flood,1],s=6,c="#4a90d9",alpha=0.3,label="base"); ax.scatter(E[flood,0],E[flood,1],s=14,c="#cc2222",alpha=0.7,label="crecida")
            s=sil(E)
        sils[name]=s; ax.set_title(f"{name}  (silhouette={s:.2f})",fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Estructura de régimen en las ventanas de caudal (2015-2025) — comparación de métodos de embedding "
                 "(model-agnostic). Silhouette↑ = mejor separa crecida/base",fontweight="bold",fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"fig25_embedding_grandtour.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    log.info("Silhouettes: "+", ".join(f"{n}={v:.2f}" for n,v in sils.items()))
    log.info("Saved: fig25_embedding_grandtour.png (model-agnostic, sin HydroST)")

if __name__=="__main__":
    main()
