#!/usr/bin/env python3
"""
Script 138 — Gráfica de LEADERBOARD de modelos para PPT: NSE vs horizonte (test 2024, obs reales).
Todos los modelos al mismo lead; HydroST (solo 1-2 días) como marcadores. Estilo editorial.
Run in .venv313. Salida: fig36_leaderboard.png (+ copia en outputs/ml_Q).
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"outputs/ml_Q"; FIG=ROOT/"generacion_paper/figures"

# paleta (coherente con el dashboard)
INK="#0C1E2A"; PAPER="#FFFFFF"; WATER="#0B6E8C"; DEEP="#0A3D54"; CYAN="#1BA8C4"
GREY="#8FA0AC"; EARTH="#8B6F47"; GREEN="#2E8B6F"; CRIT="#C0392B"; HAIR="#E2E8EE"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.edgecolor":"#9AA8B2",
    "axes.linewidth":0.8,"figure.facecolor":PAPER,"axes.facecolor":PAPER})

df=pd.read_csv(OUT/"130_master_comparison.csv"); df["model"]=df["model"].replace({"TFT-honesto":"RA-TFT"})
hy1=pd.read_csv(OUT/"131_hydrost_lead1_metrics.csv")["NSE"].iloc[0]  # 0.966
# HydroST completo por horizonte: lead1 (131), lead2 (130), leads 3/5/7/14 (139)
hy_ml=pd.read_csv(OUT/"139_hydrost_multilead.csv")
hy2=float(df[(df.model=="HydroST")&(df.lead==2)]["NSE"].iloc[0])
HY={1:hy1,2:hy2}; HY.update({int(r.lead):float(r.NSE) for r in hy_ml.itertuples()})
def serie(m): s=df[df.model==m].sort_values("lead"); return s["lead"].values,s["NSE"].values

fig,ax=plt.subplots(figsize=(12,6.6))
# regiones
ax.axvspan(0.6,2.5,color="#EEF3F6",zorder=0); ax.axvspan(2.5,14.6,color="#F4FAFC",zorder=0)
ax.text(1.55,0.30,"Persistencia = techo\n(≤ 2 días)",ha="center",va="center",fontsize=10.5,color="#5B6B78",style="italic")
ax.text(8.6,0.30,"RA-TFT lidera a multi-día (≥ 3 días)",ha="center",va="center",fontsize=11,color=DEEP,style="italic")

for m,c,lw,z in [("Persistencia",GREY,2.0,3),("LightGBM",EARTH,2.0,3),("RA-TFT",WATER,3.2,5)]:
    L,N=serie(m); ax.plot(L,N,"-o",color=c,lw=lw,ms=6,label=m,zorder=z,
                          markerfacecolor=c,markeredgecolor="white",markeredgewidth=1.1)
# HydroST por horizonte (una red por lead): línea discontinua + rombos
hl=sorted(HY); ax.plot(hl,[HY[l] for l in hl],"--D",color=GREEN,lw=1.8,ms=8,
                       markerfacecolor=GREEN,markeredgecolor="white",markeredgewidth=1.2,zorder=6,
                       label="HydroST (1 red por horizonte)")
ax.annotate(f"HydroST: mejor a 1 día\nNSE = {hy1:.3f}",xy=(1,hy1),xytext=(2.5,0.985),
            fontsize=10.5,color=GREEN,fontweight="bold",
            arrowprops=dict(arrowstyle="-",color=GREEN,lw=1))
# marca del ganador RA-TFT a h=7 y 14
for L in (7,14):
    v=df[(df.model=="RA-TFT")&(df.lead==L)]["NSE"].iloc[0]
    ax.annotate(f"{v:.3f}",xy=(L,v),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9.5,color=WATER,fontweight="bold")

ax.set_xticks([1,2,3,5,7,14]); ax.set_xlim(0.6,14.6); ax.set_ylim(0.28,1.0)
ax.set_xlabel("Horizonte de pronóstico (días)",fontsize=12.5); ax.set_ylabel("NSE  (skill de ajuste; 1 = perfecto)",fontsize=12.5)
ax.grid(axis="y",color=HAIR,lw=0.9,zorder=0); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
leg=ax.legend(loc="upper right",frameon=True,framealpha=0.95,edgecolor=HAIR,fontsize=11)
ax.set_title("Leaderboard de modelos — habilidad predictiva por horizonte",fontsize=16.5,fontweight="bold",color=INK,loc="left",pad=46)
ax.text(0.0,1.055,"Caudal del río Chancay-Huaral · periodo de prueba 2024 · solo observaciones reales · sin datos futuros (sin leakage)",
        transform=ax.transAxes,fontsize=10.5,color="#5B6B78")
fig.subplots_adjust(top=0.84)
for p in [FIG/"fig36_leaderboard.png", OUT/"138_leaderboard.png"]: fig.savefig(p,dpi=300,bbox_inches="tight",facecolor=PAPER)
plt.close(fig); print("Saved fig36_leaderboard.png | RA-TFT L7=0.678 L14=0.494 | HydroST L1=%.3f"%hy1)

if __name__=="__main__": pass
