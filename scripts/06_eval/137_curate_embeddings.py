#!/usr/bin/env python3
"""
Script 137 — Cura coordenadas 2D de embeddings (8 métodos) para graficar interactivo
en el dashboard, con 3 coloraciones (magnitud q / crecida-base Q90 / temporada).
Métodos: PCA, Features->PCA, t-SNE, UMAP, Isomap, LLE, DTW->MDS, TS2Vec (contrastivo).
Reusa ventanas de caudal [q,pr,api] 2015-2025 (model-agnostic) del enfoque del Script 111.
Salida: hidroalerta-dashboard/data/embeddings_coords.csv + embeddings_sil.json + estaciones.csv
Run in .venv313 (GPU para TS2Vec).
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
ROOT=Path(__file__).resolve().parent.parent.parent
PUB=Path("d:/ANA Concurso/hidroalerta-dashboard"); DATA=PUB/"data"
Q_CONV=86.4/3062.62; Q90=40.89; L=60; DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
R=importlib.util.module_from_spec(importlib.util.spec_from_file_location("r110",ROOT/"scripts/06_eval/110_ts_representation.py"))
importlib.util.spec_from_file_location("r110",ROOT/"scripts/06_eval/110_ts_representation.py").loader.exec_module(R)

def tsfeat(w):
    m=w.mean(); s=w.std()+1e-6; sl=np.polyfit(np.arange(len(w)),w,1)[0]
    ac=np.corrcoef(w[:-1],w[1:])[0,1] if len(w)>2 else 0
    p=np.abs(np.fft.rfft(w-m))**2; p=p/(p.sum()+1e-9); se=-np.sum(p*np.log(p+1e-12))
    return [m,s,sl,ac,s/(m+1e-6),np.argmax(w)/len(w),se]

def main():
    # estaciones (6)
    est=[("Santo Domingo","47E214D2",-11.3701,-77.0282,"aforo","Estación de aforo automática (outlet de la cuenca)"),
         ("Santo Domingo","PHISIS0137",-11.3836,-77.0503,"aforo","Estación de aforo (sensor complementario)"),
         ("Puente Callantama","47E22148",-11.2989,-76.8518,"aforo","Estación de aforo (cuenca media)"),
         ("Vichaycocha","47E257D8",-11.1392,-76.6246,"aforo","Estación de aforo (cabecera alta)"),
         ("Santa Cruz","155202",-11.2000,-76.6333,"meteorologica","Estación meteorológica SENAMHI"),
         ("Pirca","155214",-11.2333,-76.6500,"meteorologica","Estación meteorológica SENAMHI")]
    pd.DataFrame(est,columns=["nombre","codigo","lat","lon","tipo","desc"]).to_csv(DATA/"estaciones.csv",index=False)

    d7=pd.read_csv(ROOT/"data/model_ready/D7_multientity.csv",parse_dates=["date"])
    o=d7[d7.entity_id=="sub_634"].set_index("date").sort_index()
    pr=d7.groupby("date")["pr_mm"].mean(); q=(o["q_mm"]/Q_CONV); api=o["api"]; dts=q.index
    F=np.stack([q.values,pr.reindex(dts).values,api.values],0)
    mu=np.nanmean(F,1,keepdims=True); sd=np.nanstd(F,1,keepdims=True)+1e-6; Fz=np.nan_to_num((F-mu)/sd)
    idx=np.array([i for i in range(L,len(dts)) if dts[i]>=pd.Timestamp("2015-01-01")])
    W=np.stack([Fz[:,i-L:i] for i in idx]).astype(np.float32)
    qend=q.values[idx]; flood=qend>Q90
    fecha=np.array([dts[i].strftime("%Y-%m-%d") for i in idx]); mes=np.array([dts[i].month for i in idx])
    temp=np.where(np.isin(mes,[12,1,2,3,4]),"humeda","seca")
    # TODAS las ventanas (proporción natural de crecidas ~7%), como en fig25
    rng=np.random.default_rng(0); fl=np.where(flood)[0]; bs=np.where(~flood)[0]
    Wk=W; Xk=W.reshape(len(W),-1); fk=flood; qk=qend; fek=fecha; tk=temp
    print(f"Ventanas {W.shape}; graficar TODAS {Wk.shape}; crecidas={int(fk.sum())} ({100*fk.mean():.1f}%)")

    emb={}
    emb["PCA"]=PCA(2).fit_transform(Xk)
    Feat=StandardScaler().fit_transform(np.array([tsfeat(Wk[i,0]) for i in range(len(Wk))]))
    emb["Features→PCA"]=PCA(2).fit_transform(Feat)
    emb["t-SNE"]=TSNE(2,perplexity=40,init="pca",random_state=42).fit_transform(Xk)
    emb["UMAP"]=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(Xk)
    emb["Isomap"]=Isomap(n_components=2,n_neighbors=15).fit_transform(Xk)
    emb["LLE"]=LocallyLinearEmbedding(n_components=2,n_neighbors=15).fit_transform(Xk)
    # TS2Vec contrastivo (model-agnostic)
    torch.manual_seed(0); enc=R.TSEncoder(3).to(DEV); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt=torch.tensor(Wk,device=DEV); enc.train()
    for ep in range(60):
        pm=torch.randperm(len(Xt))
        for k in range(0,len(Xt),256):
            b=Xt[pm[k:k+256]]; z1,z2=enc(R.augment(b)),enc(R.augment(b))
            opt.zero_grad(); R.nt_xent(z1,z2).backward(); opt.step()
    enc.eval()
    with torch.no_grad(): z=enc(Xt).cpu().numpy()
    emb["TS2Vec"]=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(z)

    rows=[]; sil={}
    def add(method,E,mask,fe,qv,te):
        try: sil[method]=round(float(silhouette_score(E,mask.astype(int))),3)
        except Exception: sil[method]=None
        for k in range(len(E)):
            rows.append(dict(metodo=method,x=round(float(E[k,0]),4),y=round(float(E[k,1]),4),
                             regimen=("crecida" if mask[k] else "base"),temporada=te[k],
                             q=round(float(qv[k]),1),fecha=fe[k]))
    for name in ["PCA","Features→PCA","t-SNE","UMAP","Isomap","LLE","TS2Vec"]:
        add(name,emb[name],fk,fek,qk,tk); print(f"{name}: sil={sil[name]}")
    # DTW->MDS (subconjunto balanceado por costo O(N^2))
    sub=np.concatenate([fl[:90],rng.permutation(bs)[:90]]); qs=[W[i,0] for i in sub]; n=len(qs); Dm=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): Dm[i,j]=Dm[j,i]=R.dtw(qs[i],qs[j])
    Ed=MDS(2,dissimilarity="precomputed",random_state=0,normalized_stress="auto").fit_transform(Dm)
    add("DTW→MDS",Ed,flood[sub],fecha[sub],qend[sub],temp[sub]); print(f"DTW→MDS: sil={sil['DTW→MDS']}")

    pd.DataFrame(rows).to_csv(DATA/"embeddings_coords.csv",index=False)
    (DATA/"embeddings_sil.json").write_text(json.dumps(sil,ensure_ascii=False,indent=2),encoding="utf-8")
    print("coords:",len(rows),"| metodos:",list(sil.keys()),"| sil:",sil)

if __name__=="__main__":
    main()
