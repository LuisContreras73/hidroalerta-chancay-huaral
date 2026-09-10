#!/usr/bin/env python3
"""
Script 136 — Cura datos EXTRA para el dashboard (estaciones, ablación ENSO, figuras embeddings).
Escribe SOLO resultados/coordenadas públicas hacia hidroalerta-dashboard/data|assets. Run in .venv313.
"""
from pathlib import Path
import pandas as pd, shutil
ROOT=Path(__file__).resolve().parent.parent.parent
PUB=Path("d:/ANA Concurso/hidroalerta-dashboard"); DATA=PUB/"data"; ASSETS=PUB/"assets"; FIG=ROOT/"generacion_paper/figures"

# 1) Estaciones (aforo + meteorológicas SENAMHI) con coords reales
est=[
 ("Santo Domingo","47E214D2",-11.3701,-77.0282,"aforo","Estación de aforo (outlet, salida de cuenca)"),
 ("Puente Callantama","47E22148",-11.2989,-76.8518,"aforo","Estación de aforo (cuenca media)"),
 ("Vichaycocha","47E257D8",-11.1392,-76.6246,"aforo","Estación de aforo (cabecera)"),
 ("Santa Cruz","155202",-11.2000,-76.6333,"meteorologica","Estación meteorológica SENAMHI"),
 ("Pirca","155214",-11.2333,-76.6500,"meteorologica","Estación meteorológica SENAMHI"),
]
pd.DataFrame(est,columns=["nombre","codigo","lat","lon","tipo","desc"]).to_csv(DATA/"estaciones.csv",index=False)
print("estaciones.csv:",len(est),"estaciones (3 aforo + 2 meteo)")

# 2) Ablación ENSO (Script 90): NSE por configuración + R² entre índices
abl=[("Sin ENSO",0.670),("+ ONI global",0.683),("+ Niño costero",0.676),("+ Ambos índices",0.691)]
pd.DataFrame(abl,columns=["config","NSE"]).to_csv(DATA/"enso_ablacion.csv",index=False)
extra={"r_indices":0.625,"r2_indices":0.39,
       "parcial_oni_dado_costero":-0.283,"parcial_costero_dado_oni":0.273,
       "simple_oni":-0.144,"simple_costero":0.121,
       "nota":"Signo opuesto -> supresion mutua; la correlacion parcial duplica la senal de cada indice. Validacion 2023: costero +2.11 (Fuerte ICEN) mientras ONI +0.16 (neutro)."}
import json; (DATA/"enso_extra.json").write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding="utf-8")
print("enso_ablacion.csv + enso_extra.json (R2=0.39)")

# 3) Figuras de embeddings (model-agnósticas) reescaladas para el dashboard
def reescala(src,dst,maxw=1000):
    try:
        from PIL import Image
        im=Image.open(src)
        if im.width>maxw: im=im.resize((maxw,int(im.height*maxw/im.width)),Image.LANCZOS)
        im.save(dst,optimize=True)
        return dst.stat().st_size//1024
    except Exception as e:
        shutil.copy(src,dst); return -1
for s,d in [("fig25_embedding_grandtour.png","embed_caudal.png"),("fig26_grandtour_fullfeatures.png","embed_multivariado.png")]:
    src=FIG/s
    if src.exists(): kb=reescala(src,ASSETS/d); print(f"{d}: {kb} KB")
    else: print(f"FALTA {s}")
print("Silhouettes (para captions): fig25 caudal -> Isomap 0.63, PCA 0.51, DTW->MDS 0.47 | fig26 multivariado -> DTW->MDS 0.43, Isomap 0.29, PCA 0.27")
