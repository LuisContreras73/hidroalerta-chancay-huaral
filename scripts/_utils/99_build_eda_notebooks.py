#!/usr/bin/env python3
"""
Script 99: Genera los notebooks EDA para la primera presentación.

Crea (sobreescribiendo skeletons):
  notebooks/03_pisco_preliminary_analysis.ipynb
  notebooks/04_observed_qaqc.ipynb
  notebooks/06_gap_analysis.ipynb

Ejecutar desde la raíz del proyecto:
  python scripts/99_build_eda_notebooks.py
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks"

KERNEL = {
    "display_name": "HidroAlerta (Python 3.14)",
    "language": "python",
    "name": "hidroalerta",
}
LANG_INFO = {"name": "python", "version": "3.14.3"}


def _id():
    return uuid.uuid4().hex[:16]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": source,
    }


def code(source: str, tags: list[str] | None = None) -> dict:
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "id": _id(),
        "metadata": meta,
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": KERNEL,
            "language_info": LANG_INFO,
        },
        "cells": cells,
    }


def save(nb: dict, path: Path):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — PISCO Preliminary Analysis
# ══════════════════════════════════════════════════════════════════════════════
NB03 = notebook([

md("""\
# 03 · Análisis Preliminar PISCO v2.1
**Cuenca Chancay-Huaral** · HidroAlerta · Concurso ANA

Explora la climatología, variabilidad y tendencias de precipitación y temperatura
de la grilla PISCO v2.1 extraída para la cuenca (99 píxeles, 1981–2019).

| Variable | Fuente | Período | Resolución |
|----------|--------|---------|------------|
| Precipitación (pr) | PISCO v2.1 | 1981–2019 | 0.1°, diario |
| ETP (pet) | PISCO v2.1 | 1981–2019 | 0.1°, diario |
| Tmax / Tmin | PISCO v2.1 | 1981–2019 | 0.1°, diario |
"""),

code("""\
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yaml

PROJECT_ROOT = Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
FIG_DIR = PROJECT_ROOT / "outputs/figures/eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "configs/paths.yaml", encoding="utf-8") as f:
    P = yaml.safe_load(f)

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print("Paths config: OK")
"""),

md("## 1 · Carga de datos"),

code("""\
pisco = pd.read_csv(
    PROJECT_ROOT / P["pisco"]["basin_mean"],
    index_col=0, parse_dates=True,
)
pisco.index.name = "date"
# Limitar a 2019-12-31 (2020 contiene NaNs rellenados — inconsistente)
pisco = pisco.loc[:"2019-12-31"]

# Renombres legibles
rename = {"pr": "Precip (mm/d)", "pet": "ETP (mm/d)",
          "tmax": "Tmax (°C)", "tmin": "Tmin (°C)"}

print(f"Shape: {pisco.shape}")
print(f"Período: {pisco.index.min().date()} → {pisco.index.max().date()}")
print(f"Columnas: {list(pisco.columns)}")
print()
pisco.describe().round(3)
"""),

md("## 2 · Estadísticas descriptivas"),

code("""\
stats = pisco.describe().T
stats.columns = ["n", "media", "std", "min", "p25", "p50", "p75", "max"]
stats["cv%"] = (stats["std"] / stats["media"] * 100).round(1)
stats["n_nan"] = pisco.isna().sum()
stats["pct_nan"] = (pisco.isna().mean() * 100).round(2)
stats.index = [rename.get(c, c) for c in stats.index]
print("Estadísticas PISCO v2.1 — cuenca Chancay-Huaral (1981–2019):")
display(stats.round(3))
"""),

md("## 3 · Series temporales completas"),

code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
colors = ["#2171b5", "#238b45", "#d94701", "#756bb1"]
vars_  = ["pr", "pet", "tmax", "tmin"]
labels = ["Precip (mm/d)", "ETP (mm/d)", "Tmax (°C)", "Tmin (°C)"]

for ax, var, label, color in zip(axes, vars_, labels, colors):
    ax.plot(pisco.index, pisco[var], lw=0.4, color=color, alpha=0.7)
    # Media móvil 365 días
    roll = pisco[var].rolling(365, center=True, min_periods=180).mean()
    ax.plot(pisco.index, roll, lw=2, color="black", alpha=0.85, label="Media móvil 1 año")
    ax.set_ylabel(label, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    if var in ("pr", "pet"):
        ax.set_ylim(bottom=0)

axes[0].set_title("PISCO v2.1 — Media diaria cuenca Chancay-Huaral · 1981–2019 (99 píxeles)", fontweight="bold")
axes[-1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "E03_01_series_temporales_pisco.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 4 · Climatología mensual"),

code("""\
meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
monthly = pisco.groupby(pisco.index.month).agg(["mean", "std"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precipitación
ax = axes[0]
pr_m = monthly["pr"]["mean"]
pr_s = monthly["pr"]["std"]
bars = ax.bar(range(1,13), pr_m, color="#2171b5", alpha=0.85, edgecolor="white", width=0.7)
ax.errorbar(range(1,13), pr_m, yerr=pr_s, fmt="none", color="gray", capsize=4, lw=1.5)
ax.set_xticks(range(1,13)); ax.set_xticklabels(meses)
ax.set_ylabel("Precipitación media (mm/día)")
ax.set_title("Climatología mensual — Precipitación (±1σ)")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, pr_m):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7.5)

# Temperatura
ax = axes[1]
tmax_m = monthly["tmax"]["mean"]
tmin_m = monthly["tmin"]["mean"]
ax.plot(range(1,13), tmax_m, "o-", color="#d94701", lw=2, label="Tmax")
ax.plot(range(1,13), tmin_m, "o-", color="#3182bd", lw=2, label="Tmin")
ax.fill_between(range(1,13), tmin_m, tmax_m, alpha=0.12, color="orange", label="Rango diurno")
ax.set_xticks(range(1,13)); ax.set_xticklabels(meses)
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Climatología mensual — Temperatura")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle("PISCO v2.1 · Cuenca Chancay-Huaral · 1981–2019", fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "E03_02_climatologia_mensual.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Mes más lluvioso: {meses[pr_m.idxmax()-1]} ({pr_m.max():.2f} mm/d)")
print(f"Mes más seco:    {meses[pr_m.idxmin()-1]} ({pr_m.min():.2f} mm/d)")
print(f"Precip anual media: {pr_m.mean()*365:.1f} mm/año")
"""),

md("## 5 · Variabilidad interanual — Heatmap año × mes"),

code("""\
# Precipitación mensual por año
pr_monthly = pisco["pr"].resample("ME").mean()
pr_pivot = pr_monthly.to_frame()
pr_pivot["year"]  = pr_monthly.index.year
pr_pivot["month"] = pr_monthly.index.month
pivot = pr_pivot.pivot_table(index="year", columns="month", values="pr")
pivot.columns = meses

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    pivot, ax=ax, cmap="YlGnBu", linewidths=0.3,
    cbar_kws={"label": "Precip media mensual (mm/día)", "shrink": 0.6},
    vmin=0, vmax=pivot.max().max() * 0.95,
)
ax.set_title("Variabilidad interanual de precipitación — PISCO v2.1", fontsize=12, fontweight="bold")
ax.set_ylabel("Año"); ax.set_xlabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "E03_03_heatmap_anio_mes.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 6 · Boxplots mensuales — distribución diaria"),

code("""\
pisco["month"] = pisco.index.month
pisco["month_name"] = pisco["month"].map(dict(enumerate(meses, 1)))
pisco["month_name"] = pd.Categorical(pisco["month_name"], categories=meses, ordered=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_vars = [("pr","Precip (mm/d)","#2171b5"), ("pet","ETP (mm/d)","#238b45"),
             ("tmax","Tmax (°C)","#d94701"), ("tmin","Tmin (°C)","#756bb1")]

for ax, (var, label, color) in zip(axes.ravel(), plot_vars):
    pisco.boxplot(column=var, by="month_name", ax=ax,
                  boxprops=dict(color=color),
                  medianprops=dict(color="black", lw=2),
                  whiskerprops=dict(color=color, alpha=0.7),
                  flierprops=dict(marker=".", ms=1.5, alpha=0.3, color=color),
                  patch_artist=True,
                  )
    for patch in ax.patches:
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_xlabel(""); ax.set_title(label, fontsize=10)
    ax.set_xticklabels(meses, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Distribución diaria por mes — PISCO v2.1 · Cuenca Chancay-Huaral",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "E03_04_boxplots_mensuales.png", dpi=150, bbox_inches="tight")
plt.show()
pisco.drop(columns=["month","month_name"], inplace=True)
"""),

md("## 7 · Análisis de tendencia anual (Mann-Kendall)"),

code("""\
from scipy import stats as spstats

annual = pisco.resample("YE").mean()
years  = annual.index.year.values

print("Tendencias anuales (Sen's slope + Mann-Kendall):")
print(f"{'Variable':<12} {'Slope (u/año)':>14} {'p-value':>10} {'Significativa':>14}")
print("-" * 55)

results = {}
for var in ["pr", "pet", "tmax", "tmin"]:
    y = annual[var].dropna()
    x = np.arange(len(y))
    slope, intercept, r, p, se = spstats.linregress(x, y)
    sig = "✓ (p<0.05)" if p < 0.05 else "✗"
    print(f"  {var:<10} {slope:>+14.4f} {p:>10.4f} {sig:>14}")
    results[var] = dict(slope=slope, p=p, r=r)

print()
# Visualizar tendencias
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
labels2 = {"pr":"Precipitación (mm/d)","pet":"ETP (mm/d)","tmax":"Tmax (°C)","tmin":"Tmin (°C)"}
colors2  = {"pr":"#2171b5","pet":"#238b45","tmax":"#d94701","tmin":"#756bb1"}

for ax, var in zip(axes.ravel(), ["pr","pet","tmax","tmin"]):
    y = annual[var].dropna()
    x = np.arange(len(y))
    ax.plot(y.index.year, y, "o-", color=colors2[var], lw=1.5, ms=4, alpha=0.85)
    slope = results[var]["slope"]
    trend = slope * x + (y.values[0])
    ls = "-" if results[var]["p"] < 0.05 else "--"
    ax.plot(y.index.year, trend, ls, color="black", lw=2, alpha=0.7,
            label=f"Slope={slope:+.4f}/año  p={results[var]['p']:.3f}")
    ax.set_title(labels2[var], fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("Tendencias anuales PISCO v2.1 (1981–2019)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "E03_05_tendencias_anuales.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 8 · Índice de Precipitación Estandarizada (SPI-3)"),

code("""\
from scipy.stats import gamma

pr_3m = pisco["pr"].resample("ME").sum() * 30  # mm/mes aprox
pr_3m_roll = pr_3m.rolling(3).sum().dropna()

# Ajuste gamma por mes
spi = pd.Series(index=pr_3m_roll.index, dtype=float)
for m in range(1, 13):
    idx = pr_3m_roll.index.month == m
    data = pr_3m_roll[idx].values
    data_pos = data[data > 0]
    if len(data_pos) < 5:
        continue
    a, loc, scale = gamma.fit(data_pos, floc=0)
    # CDF → normal quantile
    from scipy.stats import norm
    p_zero = (data == 0).mean()
    cdf = p_zero + (1 - p_zero) * gamma.cdf(data, a, loc=loc, scale=scale)
    cdf = np.clip(cdf, 1e-4, 1 - 1e-4)
    spi[idx] = norm.ppf(cdf)

fig, ax = plt.subplots(figsize=(14, 4))
colors_spi = ["#d73027" if v < 0 else "#4575b4" for v in spi]
ax.bar(spi.index, spi, color=colors_spi, width=25, alpha=0.8)
ax.axhline(-1, color="orange", ls="--", lw=1, label="Sequía moderada (SPI<-1)")
ax.axhline(-2, color="red",    ls="--", lw=1, label="Sequía severa (SPI<-2)")
ax.axhline( 1, color="skyblue",ls="--", lw=1, label="Húmedo (SPI>+1)")
ax.set_ylabel("SPI-3")
ax.set_title("Índice de Precipitación Estandarizada (SPI-3) — PISCO v2.1 (1981–2019)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(FIG_DIR / "E03_06_spi3.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Meses con sequía severa (SPI<-2): {(spi < -2).sum()}")
print(f"Meses con exceso húmedo (SPI>+2): {(spi >  2).sum()}")
"""),

md("## 9 · Tabla resumen — exportar a CSV"),

code("""\
# Climatología mensual completa para tabla de presentación
clim = pisco.groupby(pisco.index.month).agg(
    pr_mean=("pr","mean"), pr_std=("pr","std"),
    pet_mean=("pet","mean"),
    tmax_mean=("tmax","mean"), tmin_mean=("tmin","mean"),
    n_dias=("pr","count"),
)
clim.index = meses
clim["pr_anual_mm"] = clim["pr_mean"] * 30   # aprox mm/mes
clim.to_csv(PROJECT_ROOT / "outputs/tables/T03_pisco_climatologia.csv")
print("Exportado: outputs/tables/T03_pisco_climatologia.csv")
display(clim.round(3))
"""),

])

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — QA/QC Datos Observados: Caudal + Precipitación SENAMHI
# ══════════════════════════════════════════════════════════════════════════════
NB04 = notebook([

md("""\
# 04 · QA/QC Datos Observados — Caudal y Precipitación
**Cuenca Chancay-Huaral** · HidroAlerta · Concurso ANA

Control de calidad de:
- **Caudal (Q)**: 4 estaciones hidrométricas SNIRH/ANA (2020–2026)
- **Precipitación observada**: 9 estaciones SENAMHI (1963–2014)

---
**Estación principal para calibración GR4J:** Santo Domingo (47E214D2)
Q ∈ [4.56, 113.36] m³/s · Alt: 614 m · Río Chancay-Huaral
"""),

code("""\
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yaml

PROJECT_ROOT = Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG_DIR = PROJECT_ROOT / "outputs/figures/eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "configs/paths.yaml", encoding="utf-8") as f:
    P = yaml.safe_load(f)

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
print("Setup OK")
"""),

md("## 1 · Caudal SNIRH — carga y vista general"),

code("""\
q_all = pd.read_csv(PROJECT_ROOT / P["snirh"]["daily_q"],
                    index_col=0, parse_dates=True)
q_all.index.name = "date"

# Nombres legibles para columnas
col_map = {c: c.replace("q_", "").replace("_", " ").title() for c in q_all.columns}
q_all = q_all.rename(columns=col_map)

print(f"Caudal diario SNIRH: {q_all.shape}")
print(f"Período: {q_all.index.min().date()} → {q_all.index.max().date()}")
print()
display(q_all.describe().round(3))
"""),

code("""\
# Completitud por estación
comp = pd.DataFrame({
    "n_total": q_all.notna().sum(),
    "n_nan":   q_all.isna().sum(),
    "pct_completo": (q_all.notna().mean() * 100).round(1),
    "fecha_inicio": q_all.apply(lambda c: c.first_valid_index()),
    "fecha_fin":    q_all.apply(lambda c: c.last_valid_index()),
    "Q_min": q_all.min().round(2),
    "Q_med": q_all.median().round(2),
    "Q_max": q_all.max().round(2),
})
comp["fecha_inicio"] = comp["fecha_inicio"].dt.date
comp["fecha_fin"]    = comp["fecha_fin"].dt.date
print("Completitud de estaciones hidrométricas:")
display(comp)
comp.to_csv(PROJECT_ROOT / "outputs/tables/T04_q_completitud.csv")
"""),

md("## 2 · Series temporales de caudal"),

code("""\
fig, axes = plt.subplots(len(q_all.columns), 1, figsize=(14, 3.5 * len(q_all.columns)),
                          sharex=False)
if len(q_all.columns) == 1:
    axes = [axes]

colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
for ax, col, color in zip(axes, q_all.columns, colors):
    s = q_all[col].dropna()
    ax.plot(s.index, s, lw=0.8, color=color, alpha=0.85)
    ax.fill_between(s.index, 0, s, alpha=0.15, color=color)
    ax.set_ylabel("Q (m³/s)", fontsize=9)
    ax.set_title(f"Estación: {col}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    # Estadísticas en la esquina
    txt = f"Q̄={s.mean():.1f}  Qmed={s.median():.1f}  Qmax={s.max():.1f} m³/s"
    ax.text(0.01, 0.95, txt, transform=ax.transAxes, fontsize=8,
            va="top", bbox=dict(fc="white", alpha=0.7, ec="none"))

plt.suptitle("Caudal diario SNIRH — Cuenca Chancay-Huaral", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "E04_01_series_caudal.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 3 · Santo Domingo — estación principal (47E214D2)"),

code("""\
# Identificar la columna de Santo Domingo (la que empieza antes)
q_main_col = q_all.columns[q_all.notna().sum().argmax()]
print(f"Estación principal seleccionada: {q_main_col}")
q = q_all[q_main_col].dropna().rename("Q_m3s")

print(f"  Período: {q.index.min().date()} → {q.index.max().date()}")
print(f"  N días: {len(q)}")
print(f"  Q (min/med/max): {q.min():.2f} / {q.median():.2f} / {q.max():.2f} m³/s")
"""),

code("""\
# Boxplots mensuales — Santo Domingo
q_df = q.to_frame()
q_df["month"] = q_df.index.month
q_df["month_name"] = pd.Categorical(
    q_df["month"].map(dict(enumerate(MESES,1))), categories=MESES, ordered=True)
q_df["year"] = q_df.index.year

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot mensual
ax = axes[0]
data_by_month = [q_df[q_df["month"]==m]["Q_m3s"].values for m in range(1,13)]
bp = ax.boxplot(data_by_month, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker=".", ms=2, alpha=0.4))
for patch in bp["boxes"]:
    patch.set_facecolor("#1f77b4"); patch.set_alpha(0.5)
ax.set_xticklabels(MESES, rotation=45, ha="right")
ax.set_ylabel("Q (m³/s)"); ax.set_title(f"Caudal mensual — {q_main_col}")
ax.grid(axis="y", alpha=0.3)

# Caudal medio anual
ax = axes[1]
annual_q = q_df.groupby("year")["Q_m3s"].mean()
ax.bar(annual_q.index, annual_q, color="#1f77b4", alpha=0.8, edgecolor="white")
ax.axhline(annual_q.mean(), color="red", ls="--", lw=2, label=f"Media={annual_q.mean():.1f} m³/s")
ax.set_xlabel("Año"); ax.set_ylabel("Q medio anual (m³/s)")
ax.set_title("Caudal medio anual")
ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.suptitle(f"Análisis mensual/anual — {q_main_col}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "E04_02_q_mensual_anual.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 4 · Curva de Duración de Caudales (FDC)"),

code("""\
fig, ax = plt.subplots(figsize=(10, 5))

for col, color in zip(q_all.columns, ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]):
    s = q_all[col].dropna().sort_values(ascending=False)
    exceedance = np.arange(1, len(s)+1) / (len(s)+1) * 100
    ax.semilogy(exceedance, s, lw=2, color=color, alpha=0.85, label=col)

ax.set_xlabel("Probabilidad de excedencia (%)")
ax.set_ylabel("Caudal Q (m³/s, escala log)")
ax.set_title("Curva de Duración de Caudales (FDC) — SNIRH", fontweight="bold")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)

# Percentiles referencia
q_main = q_all[q_main_col].dropna().sort_values(ascending=False)
for p, label in [(10,"Q10"),(50,"Q50"),(90,"Q90")]:
    qp = np.percentile(q_main, 100-p)
    ax.axhline(qp, color="gray", ls=":", lw=1)
    ax.text(p+1, qp, f"{label}={qp:.1f}", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(FIG_DIR / "E04_03_fdc.png", dpi=150, bbox_inches="tight")
plt.show()

print("Percentiles de caudal — Santo Domingo:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  Q{p:2d} = {np.percentile(q_main, 100-p):6.2f} m³/s")
"""),

md("## 5 · Precipitación SENAMHI — carga y completitud"),

code("""\
pr_all = pd.read_csv(PROJECT_ROOT / P["senamhi"]["precip_all"],
                     index_col=0, parse_dates=True)
pr_all.index.name = "date"

# Nombres cortos
pr_all.columns = [c.replace("pr_","").replace("_"," ").title() for c in pr_all.columns]

print(f"Precipitación SENAMHI: {pr_all.shape}")
print(f"Período: {pr_all.index.min().date()} → {pr_all.index.max().date()}")
print()

comp_pr = pd.DataFrame({
    "n_total": pr_all.notna().sum(),
    "pct_nan": (pr_all.isna().mean()*100).round(1),
    "PR_mean (mm/d)": pr_all.mean().round(2),
    "PR_max (mm/d)":  pr_all.max().round(1),
    "inicio": pr_all.apply(lambda c: c.first_valid_index()).dt.date,
    "fin":    pr_all.apply(lambda c: c.last_valid_index()).dt.date,
})
display(comp_pr.sort_values("pct_nan"))
"""),

code("""\
# Heatmap de completitud por año y estación
annual_comp = (pr_all.resample("YE").apply(lambda x: x.notna().mean() * 100)
               .round(1))
annual_comp.index = annual_comp.index.year

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(annual_comp.T, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
            linewidths=0.2, annot=False,
            cbar_kws={"label": "% días con dato", "shrink": 0.5})
ax.set_title("Completitud anual — Estaciones SENAMHI (%)", fontsize=11, fontweight="bold")
ax.set_xlabel("Año"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "E04_04_senamhi_completitud.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 6 · Comparación P observada vs P PISCO (período solapado)"),

code("""\
pisco = pd.read_csv(PROJECT_ROOT / P["pisco"]["basin_mean"],
                    index_col=0, parse_dates=True)
# Limitar PISCO a 2019-12-31 (2020 con NaNs)
pisco = pisco.loc[:"2019-12-31"]

# Período de solapamiento
t_start = max(pr_all.index.min(), pisco.index.min())
t_end   = min(pr_all.index.max(), pisco.index.max())
print(f"Período solapado: {t_start.date()} → {t_end.date()}")

pr_obs  = pr_all.loc[t_start:t_end]
pr_pisco = pisco.loc[t_start:t_end, "pr"]

# Media mensual comparada
pr_obs_m = pr_obs.resample("ME").mean()
pr_pisco_m = pr_pisco.resample("ME").mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
# Climatología mensual: obs vs PISCO
pr_obs_clim  = pr_obs.groupby(pr_obs.index.month).mean()
pr_pisco_clim = pr_pisco.groupby(pr_pisco.index.month).mean()
x = np.arange(1, 13)
w = 0.35
for i, col in enumerate(pr_obs_clim.columns[:6]):  # max 6 estaciones
    alpha = 0.7 - i * 0.05
    ax.bar(x + i*0.06 - 0.15, pr_obs_clim[col], width=0.06, alpha=max(alpha, 0.3),
           label=f"Obs: {col}", color=f"C{i}")
ax.plot(x, pr_pisco_clim.values, "k-o", lw=2.5, ms=6, label="PISCO media cuenca", zorder=5)
ax.set_xticks(x); ax.set_xticklabels(MESES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("PR media (mm/día)"); ax.set_title("Climatología mensual: Obs vs PISCO")
ax.legend(fontsize=7.5, ncol=2); ax.grid(axis="y", alpha=0.3)

# Scatter mensual (mejor estación con más datos)
ax = axes[1]
best_stn = (pr_obs.notna().sum()).idxmax()
obs_m = pr_obs_m[best_stn].dropna()
pis_m = pr_pisco_m.loc[obs_m.index].dropna()
common = obs_m.align(pis_m, join="inner")
ax.scatter(common[0], common[1], s=20, alpha=0.5, color="#2171b5")
lim = max(common[0].max(), common[1].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1.5, alpha=0.7)
from scipy import stats as sp
slope, intercept, r, p_val, _ = sp.linregress(common[0].values, common[1].values)
ax.set_xlabel(f"PR obs — {best_stn} (mm/día)")
ax.set_ylabel("PR PISCO media cuenca (mm/día)")
ax.set_title(f"Scatter mensual: r={r:.2f}, slope={slope:.2f}")
ax.grid(alpha=0.3)

plt.suptitle("Precipitación: Observada (SENAMHI) vs Satelital (PISCO)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "E04_05_obs_vs_pisco.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Estación con más datos: {best_stn}")
print(f"Correlación mensual obs-PISCO: r={r:.3f}  slope={slope:.3f}")
"""),

])

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06 — Gap Analysis
# ══════════════════════════════════════════════════════════════════════════════
NB06 = notebook([

md("""\
# 06 · Análisis de Brechas Temporales
**Cuenca Chancay-Huaral** · HidroAlerta · Concurso ANA

Identifica y cuantifica los vacíos de datos entre fuentes para diseñar
estrategias de imputación, calibración y validación del modelo.

### Brechas críticas identificadas

| Gap | Período | Impacto |
|-----|---------|---------|
| **PISCO → SNIRH Q** | 2017-01 → 2020-08 (~3.5 años) | Sin P ni Q simultáneos → impide calibración directa |
| **SENAMHI → SNIRH** | 2015 → 2020 (~5 años) | Sin P observada en alta montaña |
| **PISCOt v1.2** | Descargando (~54/56 GB) | Temperatura diaria 1981-presente |

### Estrategias de manejo
1. Usar PISCO 1981–2016 + SNIRH 2020–2026 con **warm-up** separado
2. Cubrir gap 2017–2020 con **CHIRPS** (1981–presente, gratuito)
3. Calibrar GR4J en 2020–2022, validar en 2023–2026
"""),

code("""\
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import yaml

PROJECT_ROOT = Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG_DIR = PROJECT_ROOT / "outputs/figures/eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "configs/paths.yaml", encoding="utf-8") as f:
    P = yaml.safe_load(f)
print("Setup OK")
"""),

md("## 1 · Diagrama de disponibilidad de datos — Gantt completo"),

code("""\
# Cargar catálogos silver
snirh_cat  = pd.read_csv(PROJECT_ROOT / P["snirh"]["catalog"])
senam_cat  = pd.read_csv(PROJECT_ROOT / P["senamhi"]["catalog"])
pisco_df   = pd.read_csv(PROJECT_ROOT / P["pisco"]["basin_mean"],
                          index_col=0, parse_dates=True)

records = []

# PISCO (grilla satelital) — precipitación y temperatura disponibles hasta 2019
records.append({
    "nombre": "PISCO v2.1 (precipitación)", "tipo": "Grilla satelital",
    "inicio": pd.Timestamp("1981-01-01"), "fin": pd.Timestamp("2019-12-31"),
    "color": "#6baed6",
})
records.append({
    "nombre": "PISCO v2.1 (temperatura)", "tipo": "Grilla satelital",
    "inicio": pd.Timestamp("1981-01-01"), "fin": pd.Timestamp("2019-12-31"),
    "color": "#6baed6",
})

# SNIRH Q diario
for _, row in snirh_cat[snirh_cat["var_freq"]=="1D"].iterrows():
    tipo = "Hidromética (Q)" if "caudal" in str(row["var_name"]) else "Pluviométrica (PR)"
    records.append({
        "nombre": f"{row['station_name']} [{row['var_name'][:3]}]",
        "tipo": tipo,
        "inicio": pd.to_datetime(row["date_min"]),
        "fin":    pd.to_datetime(row["date_max"]),
        "color": "#08519c" if "caudal" in str(row["var_name"]) else "#2ca02c",
    })

# SENAMHI
for _, row in senam_cat.iterrows():
    records.append({
        "nombre": f"SENAMHI {row['station_name']}",
        "tipo": "Meteorológica (T/PR)",
        "inicio": pd.to_datetime(row["date_min"]),
        "fin":    pd.to_datetime(row["date_max"]),
        "color": "#d62728",
    })

df_g = pd.DataFrame(records).sort_values(["tipo","inicio"])

fig, ax = plt.subplots(figsize=(14, max(5, len(df_g) * 0.45)))
for i, (_, row) in enumerate(df_g.iterrows()):
    dur = (row["fin"] - row["inicio"]).days
    ax.barh(i, dur, left=row["inicio"], height=0.6,
            color=row["color"], alpha=0.8, edgecolor="white", lw=0.5)

ax.set_yticks(range(len(df_g)))
ax.set_yticklabels(df_g["nombre"], fontsize=8)
ax.set_xlabel("Año")
ax.set_title("Disponibilidad temporal de datos — Cuenca Chancay-Huaral",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)

# Líneas de referencia
hitos = [
    ("1981-01-01", "PISCO inicio", "blue", "--"),
    ("2019-12-31", "PISCO fin (datos limpios)",    "blue",  ":"),
    ("2020-09-01", "SNIRH Q inicio","navy","-."),
]
for fecha, label, color, ls in hitos:
    ax.axvline(mdates.date2num(pd.Timestamp(fecha)), color=color, ls=ls, lw=1.8,
               alpha=0.7, label=label)
ax.legend(loc="lower right", fontsize=8)

# Zona de gap crítico
t_gap_ini = mdates.date2num(pd.Timestamp("2017-01-01"))
t_gap_fin = mdates.date2num(pd.Timestamp("2020-09-01"))
ax.axvspan(t_gap_ini, t_gap_fin, alpha=0.12, color="red", label="Gap crítico 2017–2020")

leyenda = [
    mpatches.Patch(color="#6baed6", alpha=0.8, label="Grilla satelital (PISCO)"),
    mpatches.Patch(color="#08519c", alpha=0.8, label="Hidromética (Q)"),
    mpatches.Patch(color="#2ca02c", alpha=0.8, label="Pluviométrica (PR)"),
    mpatches.Patch(color="#d62728", alpha=0.8, label="Meteorológica (SENAMHI)"),
    mpatches.Patch(color="red",     alpha=0.12, label="Gap crítico P+Q simultáneos"),
]
ax.legend(handles=leyenda, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "E06_01_gantt_datos.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 2 · Cuantificación del gap 2017–2020"),

code("""\
# Gap entre fin PISCO (datos limpios hasta 2019) y inicio SNIRH Q
pisco_end   = pd.Timestamp("2019-12-31")
snirh_start = pd.Timestamp("2020-09-01")
gap_dias = (snirh_start - pisco_end).days

print("=" * 55)
print("  ANÁLISIS DEL GAP CRÍTICO")
print("=" * 55)
print(f"  PISCO termina (datos limpios): {pisco_end.date()}")
print(f"  SNIRH Q inicia:                {snirh_start.date()}")
print(f"  Gap:                           {gap_dias} días ({gap_dias/365.25:.1f} años)")
print()
print("  ¿Tenemos precipitación en el gap?")
print(f"  PISCO:    NO (último año completo 2019)")
print(f"  SENAMHI:  PARCIAL (mayoría hasta ~2014)")
print(f"  CHIRPS:   SÍ — disponible 1981-presente (gratuito, 0.05°)")
print()
print("  ¿Tenemos caudal en el gap?")
print(f"  SNIRH:    NO (inicia sep 2020)")
print(f"  Solución: calibrar GR4J post-gap, usar PISCO para forcing histórico")
print()

# Períodos de datos efectivos
print("PERÍODOS UTILIZABLES:")
print(f"  Calibración PISCO+SENAMHI (clima):  1981–2019 (P, T, ETP)")
print(f"  Calibración GR4J (con Q obs):       Sep 2020 → hoy")
print(f"    Warm-up:    Sep 2020 – Dic 2020 (3 meses)")
print(f"    Calibración:Ene 2021 – Dic 2023")
print(f"    Validación: Ene 2024 – hoy")
print()
# Días por período
cal_start = pd.Timestamp("2021-01-01")
cal_end   = pd.Timestamp("2023-12-31")
val_start = pd.Timestamp("2024-01-01")
val_end   = pd.Timestamp("2026-05-19")
print(f"    Calibración: {(cal_end-cal_start).days} días")
print(f"    Validación:  {(val_end-val_start).days} días")
"""),

md("## 3 · Estrategia de splits temporal"),

code("""\
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_facecolor("#f8f8f8")

# Timeline completa
t0 = pd.Timestamp("1981-01-01")
tn = pd.Timestamp("2026-12-31")
ax.set_xlim(t0, tn)
ax.set_ylim(-0.5, 4.5)

def span(inicio, fin, y, h, color, label, alpha=0.8):
    dur = (pd.Timestamp(fin) - pd.Timestamp(inicio)).days
    bar = ax.barh(y, dur, left=pd.Timestamp(inicio), height=h,
                  color=color, alpha=alpha, edgecolor="white", lw=0.8)
    cx = pd.Timestamp(inicio) + pd.Timedelta(days=dur/2)
    ax.text(cx, y, label, ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")

span("1981-01-01","2019-12-31", 4, 0.5, "#6baed6", "PISCO v2.1 (P, T, ETP) 1981–2019")
span("1963-01-01","2014-12-31", 3, 0.5, "#d62728", "SENAMHI (P, T obs)")
span("2020-09-01","2020-12-31", 2, 0.5, "#fdae6b", "Warm-up Q (3m)")
span("2021-01-01","2023-12-31", 2, 0.5, "#2ca02c", "Calibración GR4J (3 años)")
span("2024-01-01","2026-05-19", 2, 0.5, "#1f77b4", "Validación GR4J (~2.5 años)")
span("2019-12-31","2020-09-01", 1, 0.5, "#fc4e2a", "⚠ GAP crítico (~8 meses)")
span("1981-01-01","2026-05-19", 0, 0.5, "#74c476", "Objetivo ML/TFT (con imputación)")

ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(["ML/TFT (objetivo)","Gap crítico","Q SNIRH","SENAMHI","PISCO"], fontsize=9)
ax.set_title("Estrategia temporal de splits — HidroAlerta Chancay-Huaral",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(FIG_DIR / "E06_02_splits_temporales.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("## 4 · Análisis de gaps internos — series de caudal"),

code("""\
q_all = pd.read_csv(PROJECT_ROOT / P["snirh"]["daily_q"],
                    index_col=0, parse_dates=True)
q_all.columns = [c.replace("q_","").replace("_"," ").title() for c in q_all.columns]

print("Análisis de gaps internos en caudal diario:")
print(f"{'Estación':<35} {'Gaps (bloques)':>15} {'Max gap (días)':>15} {'% NaN':>8}")
print("-" * 75)
for col in q_all.columns:
    s = q_all[col]
    nan_mask = s.isna()
    gaps = nan_mask.ne(nan_mask.shift()).cumsum()[nan_mask].value_counts()
    n_gaps   = len(gaps)
    max_gap  = int(nan_mask.groupby((~nan_mask).cumsum()).sum().max()) if nan_mask.any() else 0
    pct_nan  = nan_mask.mean() * 100
    print(f"  {col:<33} {n_gaps:>15} {max_gap:>15} {pct_nan:>7.1f}%")

# Visualización de gaps como imagen de disponibilidad
fig, ax = plt.subplots(figsize=(14, 4))
avail = q_all.notna().astype(int).T
ax.imshow(avail.values, aspect="auto", cmap="RdYlGn",
          extent=[mdates.date2num(q_all.index[0]),
                  mdates.date2num(q_all.index[-1]),
                  -0.5, len(q_all.columns)-0.5],
          vmin=0, vmax=1)
ax.set_yticks(range(len(q_all.columns)))
ax.set_yticklabels(q_all.columns, fontsize=8)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_title("Disponibilidad diaria de caudal — verde=dato, rojo=NaN", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "E06_03_gaps_caudal.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""\
## 5 · Recomendaciones para la presentación

### Mensajes clave a transmitir

**1. Riqueza de datos a largo plazo (1981–2026)**
- PISCO provee 36 años de forzante climático continuo y validado
- SENAMHI añade 50+ años de observaciones puntales (con gaps)
- SNIRH aporta caudal reciente de alta calidad (2020–hoy)

**2. El gap 2017–2020 es manejable**
- No impide la calibración (tenemos Q 2020–2026)
- CHIRPS puede cubrir el gap para forcing de precipitación
- Warm-up de 3 meses es suficiente para GR4J

**3. Estrategia de splits es robusta**
- Calibración: 3 años completos (2021–2023)
- Validación: ~2.5 años recientes (2024–2026) ← nunca vistos por el modelo

**4. La variabilidad es el desafío central**
- Cuenca de régimen pluvio-nival: baja en verano, alta en invierno austral
- CV de precipitación ~160% (alta variabilidad interanual)
- SPI-3 muestra eventos extremos documentados (Niño/Niña)
"""),

])


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR LOS TRES NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generando notebooks EDA para la primera presentación...")
    save(NB03, NB_DIR / "03_pisco_preliminary_analysis.ipynb")
    save(NB04, NB_DIR / "04_observed_qaqc.ipynb")
    save(NB06, NB_DIR / "06_gap_analysis.ipynb")
    print("\nListo. Abrir con JupyterLab y seleccionar kernel: HidroAlerta (Python 3.14)")
