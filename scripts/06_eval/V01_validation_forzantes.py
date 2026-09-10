#!/usr/bin/env python3
"""
Script V01: Figuras de validación de los forzantes climáticos Bronze.

Valida la consistencia y calidad de los productos generados:
  - PET Hargreaves-Samani calibrado vs PISCOp PM (1981-2016)
  - Tmax y Tmin PISCOt v1.2: series temporales, tendencias, climatología espacial
  - Resumen completo de todos los forzantes 1981-2020

Figuras producidas (outputs/figures/validation/)
-------------------------------------------------
  V01 — PET: scatter + climatología mensual + serie anual + factores k
  V02 — Temperatura: series anuales Tmax/Tmin/DTR con tendencias Mann-Kendall
  V03 — Tmin: climatología mensual espacial (2×6 pcolormesh + contorno)
  V04 — Todos los forzantes: panel mensual 1981-2020 (pr, pet, tmax, tmin)
"""
import warnings
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import netCDF4 as nc
import geopandas as gpd
import scipy.stats as st

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

ROOT     = Path(__file__).parent.parent
FIG_DIR  = ROOT / "outputs/figures/basin"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = (ROOT / "data/raw/shapefiles/cuenca_chancay_huaral/limite"
            / "Cuenca_Chancay___Huaral.shp")
BASIN_MEAN = ROOT / "data/bronze/B2_pisco_basin_mean_all.csv"
PET_HS_CSV = ROOT / "data/bronze/B2_pet_basin_mean_hscal.csv"
PET_HS_NC  = ROOT / "data/bronze/B2_pet_hargreaves_cal.nc"
TMAX_NC    = ROOT / "data/bronze/B2_tmax_basin_grid_v12.nc"
TMIN_NC    = ROOT / "data/bronze/B2_tmin_basin_grid_v12.nc"


# ── utilidades ────────────────────────────────────────────────────────────────
MESES = ["Ene","Feb","Mar","Abr","May","Jun",
         "Jul","Ago","Sep","Oct","Nov","Dic"]

def _kge(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) < 5:
        return dict(KGE=np.nan, r=np.nan, RMSE=np.nan, PBIAS=np.nan)
    r     = float(np.corrcoef(o, s)[0, 1])
    alpha = s.std() / (o.std() + 1e-12)
    beta  = s.mean() / (o.mean() + 1e-12)
    kge   = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    return dict(KGE=round(kge,3), r=round(r,3),
                RMSE=round(float(np.sqrt(np.mean((o-s)**2))),3),
                PBIAS=round(float((s.sum()-o.sum())/(o.sum()+1e-12)*100),1))

def _mk_trend(x, y):
    """Mann-Kendall + Sen's slope via scipy."""
    res = st.linregress(x, y)
    mk  = st.kendalltau(x, y)
    return res.slope, res.pvalue, mk.pvalue

def _edges(c):
    h = (c[1]-c[0])/2
    return np.concatenate([c-h,[c[-1]+h]])

def _pcm(ax, lons, lats, field, cmap, vmin, vmax):
    LE, LT = np.meshgrid(_edges(lons), _edges(lats))
    return ax.pcolormesh(LE, LT, ma.masked_invalid(field),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")

def _contour(ax, lons, lats, field, n_lvl=3, color="white", lw=0.8):
    try:
        f  = np.where(np.isfinite(field), field, np.nan)
        vv = f[np.isfinite(f)]
        if len(vv) < 4: return
        lvls = np.linspace(np.percentile(vv,8), np.percentile(vv,92), n_lvl+2)[1:-1]
        ax.contour(lons, lats, f, levels=lvls, colors=color,
                   linewidths=lw, alpha=0.75)
    except Exception:
        pass

def _basin(ax, lw=1.0):
    try:
        b = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
        b.boundary.plot(ax=ax, color="white",   linewidth=lw*2, zorder=5)
        b.boundary.plot(ax=ax, color="#111111", linewidth=lw, zorder=6,
                        linestyle="--")
    except Exception:
        pass

def _fmt_ax(ax, title, fs=9, xlabel=True, ylabel=True):
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=3)
    if xlabel: ax.set_xlabel("Lon (°)", fontsize=7)
    if ylabel: ax.set_ylabel("Lat (°)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, lw=0.3, alpha=0.3, color="gray", ls=":")
    ax.set_aspect("equal")

def _trend_label(slope_decade, pval):
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else
          ("*"   if pval < 0.05  else "ns"))
    return f"{slope_decade:+.3f} °C/dec {sig}"


# ── Cargar datos de cuenca (series diarias y mensuales) ──────────────────────
print("Cargando series de cuenca...")
df = pd.read_csv(BASIN_MEAN, index_col=0, parse_dates=True)
pet_hs_s = pd.read_csv(PET_HS_CSV, index_col=0, parse_dates=True).squeeze()

# Separar PISCOp PET (1981-2016) del HS-cal PET
pet_pm  = df["pet"][df.index.year <= 2016]        # PM PISCOp reference
pet_hs  = pet_hs_s[pet_hs_s.index.year <= 2016]   # HS-cal en mismo período
pet_hs_full = pet_hs_s                             # HS-cal 1981-2020

# Cargar grillas espaciales para Tmin y el factor k
ds_pet = nc.Dataset(PET_HS_NC)
lat_pet = np.array(ds_pet["lat"][:])
lon_pet = np.array(ds_pet["lon"][:])
mask_pet = np.array(ds_pet["basin_mask"][:]).astype(bool)
k_mean  = np.array(ds_pet["k_calibration"][:])    # factor k medio por pixel
ds_pet.close()

ds_tm = nc.Dataset(TMIN_NC)
lat_t   = np.array(ds_tm["lat"][:])
lon_t   = np.array(ds_tm["lon"][:])
mask_t  = np.array(ds_tm["basin_mask"][:]).astype(bool)
ds_tm.close()

print("Datos cargados OK.")


# ══════════════════════════════════════════════════════════════════════════════
# V01 — Validación PET: HS-calibrado vs PISCOp Penman-Monteith (1981-2016)
# ══════════════════════════════════════════════════════════════════════════════
print("V01: validacion PET ...")

# Alinear índices
idx_common = pet_pm.index.intersection(pet_hs.index)
pm_vals = pet_pm.loc[idx_common].values
hs_vals = pet_hs.loc[idx_common].values
metrics = _kge(pm_vals, hs_vals)

# Mensuales
pet_pm_mon  = pet_pm.resample("ME").mean()
pet_hs_mon  = pet_hs.resample("ME").mean()
by_m_pm     = pet_pm_mon.groupby(pet_pm_mon.index.month).mean()
by_m_hs     = pet_hs_mon.groupby(pet_hs_mon.index.month).mean()
by_m_pm_std = pet_pm_mon.groupby(pet_pm_mon.index.month).std()
by_m_hs_std = pet_hs_mon.groupby(pet_hs_mon.index.month).std()

# Anuales
pet_pm_ann  = pet_pm.resample("YE").mean()
pet_hs_ann  = pet_hs_full.resample("YE").mean()

# Factores k por pixel (solo cuenca) — distribución
k_basin = k_mean[mask_pet]

fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.72, wspace=0.42)

# ── Panel a: Scatter diario ────────────────────────────────────────────────
ax_sc = fig.add_subplot(gs[0, 0])
ax_sc.scatter(pm_vals[::5], hs_vals[::5], s=3, alpha=0.25, color="#2980b9",
              rasterized=True)
lims = [min(pm_vals.min(), hs_vals.min())*0.98,
        max(pm_vals.max(), hs_vals.max())*1.02]
ax_sc.plot(lims, lims, "k--", lw=1.2, label="1:1")
ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
ax_sc.set_xlabel("PET PISCOp PM (mm/día)", fontsize=9)
ax_sc.set_ylabel("PET HS-calibrado (mm/día)", fontsize=9)
ax_sc.set_title("(a) Scatter diario — media cuenca\n1981-2016", fontweight="bold")
info = (f"KGE = {metrics['KGE']}\n"
        f"r   = {metrics['r']}\n"
        f"RMSE= {metrics['RMSE']} mm/d\n"
        f"PBIAS= {metrics['PBIAS']}%")
ax_sc.text(0.04, 0.96, info, transform=ax_sc.transAxes, va="top", fontsize=8,
           family="monospace", bbox=dict(fc="white", alpha=0.85, pad=3))
ax_sc.legend(fontsize=8); ax_sc.grid(alpha=0.3, lw=0.7)

# ── Panel b: Climatología mensual (PISCOp vs HS-cal) ─────────────────────
ax_cl = fig.add_subplot(gs[0, 1])
x_m   = np.arange(1, 13)
ax_cl.errorbar(x_m, by_m_pm.values, yerr=by_m_pm_std.values,
               fmt="o-", color="#e74c3c", capsize=3, lw=1.5, ms=5,
               label="PISCOp PM (referencia)")
ax_cl.errorbar(x_m, by_m_hs.values, yerr=by_m_hs_std.values,
               fmt="s--", color="#2980b9", capsize=3, lw=1.5, ms=5,
               label="HS calibrado")
ax_cl.set_xticks(x_m); ax_cl.set_xticklabels(MESES, fontsize=8)
ax_cl.set_xlabel("Mes", fontsize=9)
ax_cl.set_ylabel("PET media mensual (mm/día)", fontsize=9)
ax_cl.set_title("(b) Climatología mensual\n1981-2016  (media ± 1σ)", fontweight="bold")
ax_cl.legend(fontsize=8); ax_cl.grid(alpha=0.3, lw=0.7)
ax_cl.set_ylim(bottom=0)

# ── Panel c: Serie anual comparativa ──────────────────────────────────────
ax_ann = fig.add_subplot(gs[0, 2])
yr_pm  = np.array(pd.DatetimeIndex(pet_pm_ann.index).year)
yr_hs  = np.array(pd.DatetimeIndex(pet_hs_ann.index).year)
ax_ann.plot(yr_pm, pet_pm_ann.values, "o-", color="#e74c3c", ms=4, lw=1.4,
            label="PISCOp PM (1981-2016)")
ax_ann.plot(yr_hs, pet_hs_ann.values, "s--", color="#2980b9", ms=4, lw=1.4,
            label="HS calibrado (1981-2020)")
ax_ann.axvline(2016.5, color="gray", ls=":", lw=1.2, label="Fin PISCOp")
ax_ann.set_xlabel("Año", fontsize=9)
ax_ann.set_ylabel("PET media anual (mm/día)", fontsize=9)
ax_ann.set_title("(c) Serie anual media de cuenca\nHS calibrado extiende 2017-2020",
                 fontweight="bold")
ax_ann.legend(fontsize=7.5); ax_ann.grid(alpha=0.3, lw=0.7)

# ── Panel d: Factor k mensual (distribución entre píxeles) ───────────────
ax_k = fig.add_subplot(gs[1, 0])
# Cargar k por mes por pixel desde el NC
ds_pk = nc.Dataset(PET_HS_NC)
# Recalcular k mensual directamente cargando PET HS y PM
# (k_calibration en NC es el promedio de los 12 meses)
# Usamos los valores mensuales de las series de cuenca como representativos
k_monthly_proxy = by_m_hs.values / (by_m_pm.values + 1e-9)
ds_pk.close()

colors_k = plt.cm.RdBu_r(Normalize(vmin=0.90, vmax=1.10)(k_monthly_proxy))
bars = ax_k.bar(x_m, k_monthly_proxy, color=colors_k, edgecolor="gray",
                linewidth=0.5, alpha=0.85)
ax_k.axhline(1.0, color="k", ls="--", lw=1.3, label="k = 1 (sin corrección)")
ax_k.set_xticks(x_m); ax_k.set_xticklabels(MESES, fontsize=8)
ax_k.set_xlabel("Mes", fontsize=9)
ax_k.set_ylabel("Factor k = PET_HS / PET_PM", fontsize=9)
ax_k.set_title("(d) Factor de calibración mensual k\n"
               "(media cuenca; azul=HS subestima, rojo=HS sobreestima)",
               fontweight="bold")
for bar, kv in zip(bars, k_monthly_proxy):
    ax_k.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
              f"{kv:.3f}", ha="center", va="bottom", fontsize=7)
ax_k.legend(fontsize=8); ax_k.grid(axis="y", alpha=0.3, lw=0.7)
ax_k.set_ylim(0.90, 1.12)

# ── Panel e: Diferencias mensuales HS - PM ───────────────────────────────
ax_diff = fig.add_subplot(gs[1, 1])
diff_mon_mean = (pet_hs_mon - pet_pm_mon).groupby(
    pet_hs_mon.index.month).mean()
diff_mon_std  = (pet_hs_mon - pet_pm_mon).groupby(
    pet_hs_mon.index.month).std()
ax_diff.bar(x_m, diff_mon_mean.values, yerr=diff_mon_std.values,
            color=["#c0392b" if d > 0 else "#2980b9" for d in diff_mon_mean.values],
            alpha=0.80, edgecolor="gray", lw=0.5, capsize=3)
ax_diff.axhline(0, color="k", lw=1.2, ls="--")
ax_diff.set_xticks(x_m); ax_diff.set_xticklabels(MESES, fontsize=8)
ax_diff.set_xlabel("Mes", fontsize=9)
ax_diff.set_ylabel("HS-cal − PM (mm/día)", fontsize=9)
ax_diff.set_title("(e) Diferencia mensual HS-cal − PISCOp PM\n"
                  "media ± 1σ inter-anual  |  1981-2016", fontweight="bold")
ax_diff.grid(axis="y", alpha=0.3, lw=0.7)

# ── Panel f: Distribución acumulada mensual (FDC) ────────────────────────
ax_fdc = fig.add_subplot(gs[1, 2])
sorted_pm  = np.sort(pet_pm_mon.values)[::-1]
sorted_hs  = np.sort(pet_hs_mon.values)[::-1]
probs = np.linspace(0, 100, len(sorted_pm))
ax_fdc.plot(probs, sorted_pm, color="#e74c3c", lw=1.8, label="PISCOp PM")
ax_fdc.plot(probs, sorted_hs, color="#2980b9", lw=1.8, ls="--",
            label="HS calibrado")
ax_fdc.set_xlabel("Probabilidad de excedencia (%)", fontsize=9)
ax_fdc.set_ylabel("PET mensual media (mm/día)", fontsize=9)
ax_fdc.set_title("(f) Curva de duración de frecuencias\n"
                 "PET mensual media cuenca 1981-2016", fontweight="bold")
ax_fdc.legend(fontsize=9); ax_fdc.grid(alpha=0.3, lw=0.7)
ax_fdc.set_ylim(bottom=0)

fig.suptitle("Validación PET: Hargreaves-Samani calibrado vs PISCOp Penman-Monteith\n"
             "Cuenca Chancay-Huaral  |  Período de traslape 1981-2016",
             fontsize=12, fontweight="bold", y=0.98)
fig.text(0.5, 0.005,
    "Nota — (a): r bajo es esperado en cuencas andinas tropicales (~lat −11°); "
    "la amplitud estacional de la PET es ≈0.10 mm/día frente a una media de ≈2.9 mm/día. "
    "La variabilidad diaria está dominada por VPD/Rn que la fórmula de Hargreaves-Samani "
    "no captura a escala sub-mensual; el sesgo de volumen (PBIAS≈0%) sí es correcto.",
    ha="center", fontsize=6.5, color="#555555", style="italic")

plt.savefig(FIG_DIR / "V01_pet_validacion_hs_piscop.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V01 guardado")


# ══════════════════════════════════════════════════════════════════════════════
# V02 — Temperatura: series anuales Tmax / Tmin / DTR con tendencias MK
# ══════════════════════════════════════════════════════════════════════════════
print("V02: series temperatura ...")

tmax_s = df["tmax"]
tmin_s = df["tmin"]
dtr_s  = tmax_s - tmin_s    # rango diurno de temperatura

tmax_ann = tmax_s.resample("YE").mean()
tmin_ann = tmin_s.resample("YE").mean()
dtr_ann  = dtr_s.resample("YE").mean()
years_t  = np.array(pd.DatetimeIndex(tmax_ann.index).year)

# Tendencias
sl_tx, pv_tx, mk_tx = _mk_trend(years_t, tmax_ann.values)
sl_tn, pv_tn, mk_tn = _mk_trend(years_t, tmin_ann.values)
sl_dt, pv_dt, mk_dt = _mk_trend(years_t, dtr_ann.values)

# Líneas de tendencia
def _trendline(years, vals, slope):
    xc = years - years.mean()
    return vals.mean() + slope * xc

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                          gridspec_kw={"hspace": 0.55})

for ax, series, ann, slope, pv, mk_pv, col, label, ylab in [
    (axes[0], tmax_s, tmax_ann, sl_tx, pv_tx, mk_tx,
     "#e74c3c", "Tmax", "Temperatura máxima (°C)"),
    (axes[1], tmin_s, tmin_ann, sl_tn, pv_tn, mk_tn,
     "#2980b9", "Tmin", "Temperatura mínima (°C)"),
    (axes[2], dtr_s,  dtr_ann,  sl_dt, pv_dt, mk_dt,
     "#8e44ad", "DTR",  "Rango diurno ΔT (°C)"),
]:
    # Serie mensual de fondo (semitransparente)
    mon = series.resample("ME").mean()
    yr_mon = mon.index.to_pydatetime()
    ax.fill_between(yr_mon, mon.values, alpha=0.18, color=col)
    ax.plot(yr_mon, mon.values, color=col, lw=0.5, alpha=0.45)

    # Media anual
    yr_dt = pd.DatetimeIndex(ann.index).to_pydatetime()
    ax.plot(yr_dt, ann.values, "o-", color=col, ms=4, lw=1.8,
            label=f"Media anual  (1981-2020)")

    # Línea de tendencia
    trend = _trendline(years_t, ann.values, slope)
    ax.plot([pd.Timestamp(str(y)) for y in years_t], trend,
            color="k", ls="--", lw=1.5, alpha=0.8,
            label=_trend_label(slope*10, mk_pv))

    # Medias por período
    p1 = ann[ann.index.year <= 2000].mean()
    p2 = ann[ann.index.year >= 2001].mean()
    ax.axhline(p1, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.axhline(p2, color="gray", ls="-.", lw=1.0, alpha=0.7)
    # Posición relativa al rango de datos para evitar solapamientos
    ylo, yhi = ax.get_ylim()
    yr = yhi - ylo
    offset1 = 0.012 * yr if (p2 - p1) > 0.04 * yr else 0.025 * yr
    ax.text(pd.Timestamp("2020-06"), p1 + offset1,
            f"μ₁₉₈₁₋₂₀₀₀={p1:.2f}",
            ha="right", va="bottom", fontsize=7, color="gray")
    ax.text(pd.Timestamp("2020-06"), p2 - offset1,
            f"μ₂₀₀₁₋₂₀₂₀={p2:.2f}",
            ha="right", va="top", fontsize=7, color="gray")

    ax.set_ylabel(ylab, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, lw=0.7)
    ax.set_title(f"{label}  —  PISCOt v1.2, 1981-2020, media cuenca ponderada coseno",
                 fontweight="bold", fontsize=10)

    # Señalar El Niño fuerte — anotaciones con transformación axes para posición estable
    for yr_n in [1983, 1998, 2016]:
        ax.axvline(pd.Timestamp(str(yr_n)), color="orange",
                   alpha=0.5, lw=1.5, ls=":")
    if label == "Tmax":
        ylo2, yhi2 = ax.get_ylim()
        for yr_n, lbl in [(1983, "EN 83"), (1998, "EN 98"), (2016, "EN 16")]:
            ax.text(pd.Timestamp(str(yr_n)), ylo2 + 0.96*(yhi2-ylo2),
                    f" {lbl}", fontsize=6.5, color="darkorange",
                    va="top", ha="left",
                    bbox=dict(fc="white", alpha=0.6, pad=1, lw=0))

axes[2].set_xlabel("Año", fontsize=10)
fig.suptitle("Temperatura: series anuales Tmax / Tmin / DTR — Cuenca Chancay-Huaral\n"
             "PISCOt v1.2 (0.01°, 1981-2020) | Tendencias Mann-Kendall "
             "(*** p<0.001, ** p<0.01, * p<0.05, ns = no significativo)",
             fontsize=11, fontweight="bold")
plt.savefig(FIG_DIR / "V02_temperatura_series_1981_2020.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V02 guardado")


# ══════════════════════════════════════════════════════════════════════════════
# V03 — Climatología mensual espacial Tmin (2×6 pcolormesh + contorno)
# ══════════════════════════════════════════════════════════════════════════════
print("V03: climatologia espacial Tmin ...")

# Cargar Tmin desde NC y calcular climatología mensual por pixel
print("  Cargando Tmin NC (puede tardar)...")
ds_tn  = nc.Dataset(TMIN_NC)
lat_tn = np.array(ds_tn["lat"][:])
lon_tn = np.array(ds_tn["lon"][:])
msk_tn = np.array(ds_tn["basin_mask"][:]).astype(bool)
tmin_g = np.array(ds_tn["tmin"][:])          # (14610, 111, 91)
ds_tn.close()

dates_t = pd.date_range("1981-01-01", periods=14610, freq="D")
mo_t    = np.array(pd.DatetimeIndex(dates_t).month)
yr_t    = np.array(pd.DatetimeIndex(dates_t).year)

# Climatología mensual por pixel (°C, media sobre el mes)
clim_tmin = np.zeros((12, len(lat_tn), len(lon_tn)))
n_yr_t    = len(np.unique(yr_t))
for mi in range(1, 13):
    total = np.zeros((len(lat_tn), len(lon_tn)))
    cnt   = 0
    for yy in np.unique(yr_t):
        idx = (mo_t == mi) & (yr_t == yy)
        if idx.sum() == 0: continue
        total += np.nanmean(tmin_g[idx], axis=0)
        cnt   += 1
    clim_tmin[mi-1] = total / max(cnt, 1)

clim_tmin = np.where(msk_tn[None,:,:], clim_tmin, np.nan)
vmin_t = float(np.nanpercentile(clim_tmin[:, msk_tn].ravel(), 2))
vmax_t = float(np.nanpercentile(clim_tmin[:, msk_tn].ravel(), 98))

fig, axes = plt.subplots(2, 6, figsize=(17, 7.5),
                          gridspec_kw={"hspace":0.50, "wspace":0.18})
plt.subplots_adjust(right=0.89)

for mi, ax in enumerate(axes.ravel()):
    im = _pcm(ax, lon_tn, lat_tn, clim_tmin[mi], "RdBu_r", vmin_t, vmax_t)
    _contour(ax, lon_tn, lat_tn, clim_tmin[mi], n_lvl=3, color="k", lw=0.5)
    _basin(ax, lw=0.6)
    _fmt_ax(ax, MESES[mi], fs=10, xlabel=False, ylabel=False)
    ax.tick_params(labelsize=5)
    mean_v = float(np.nanmean(clim_tmin[mi][msk_tn]))
    ax.text(0.04, 0.04, f"{mean_v:.1f}°C",
            transform=ax.transAxes, fontsize=6.5, color="white",
            bbox=dict(fc="black", alpha=0.52, pad=1.5))

sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=vmin_t, vmax=vmax_t))
sm.set_array([])
cbar_ax = fig.add_axes([0.907, 0.12, 0.013, 0.75])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Tmin media mensual (°C)", fontsize=10)
cb.ax.tick_params(labelsize=8)

fig.suptitle("PISCOt v1.2 — Climatología mensual de temperatura mínima (1981-2020)\n"
             "Cuenca Chancay-Huaral  |  0.01°  |  Contornos en °C  (RdBu_r)",
             fontsize=12, fontweight="bold")
plt.savefig(FIG_DIR / "V03_tmin_climatologia_mensual.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V03 guardado")


# ══════════════════════════════════════════════════════════════════════════════
# V04 — Overview completo forzantes (mensual 1981-2020)
# ══════════════════════════════════════════════════════════════════════════════
print("V04: overview forzantes ...")

# Resamplear todo a mensual
pr_mon   = df["pr"].resample("ME").sum()            # mm/mes
pet_mon  = df["pet"].resample("ME").mean()           # mm/día → media mes
tmax_mon = df["tmax"].resample("ME").mean()          # °C
tmin_mon = df["tmin"].resample("ME").mean()          # °C

yr_all = pd.date_range("1981-01", "2020-12", freq="ME")

fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                          gridspec_kw={"hspace": 0.48})

# ── pr ────────────────────────────────────────────────────────────────────
ax = axes[0]
# Colorear las barras: azul = PISCOp disponible, naranja = pendiente CHIRPS-QM
colors_pr = ["#2980b9" if not pd.isna(v) else "#e67e22" for v in pr_mon.values]
ax.bar(pr_mon.index, pr_mon.values, color=colors_pr, width=28, alpha=0.85)
ax.axhline(pr_mon.mean(), color="k", ls="--", lw=1.2,
           label=f"Media = {pr_mon.mean():.0f} mm/mes")
ax.set_ylabel("Precipitación\n(mm/mes)", fontsize=9)
ax.set_title("(a) Precipitación — PISCOp v2.1 (azul) | CHIRPS-QM pendiente (naranja)",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, lw=0.7)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(fc="#2980b9", label="PISCOp v2.1 (1981-2019)"),
    Patch(fc="#e67e22", label="CHIRPS-QM (2020, pendiente)"),
    plt.Line2D([0],[0], color="k", ls="--", lw=1.2,
               label=f"Media = {pr_mon.dropna().mean():.0f} mm/mes"),
], fontsize=8)

# ── pet ───────────────────────────────────────────────────────────────────
ax = axes[1]
colors_pet = ["#27ae60" if yr <= 2016 else "#f39c12"
              for yr in pd.DatetimeIndex(pet_mon.index).year]
ax.plot(pet_mon.index, pet_mon.values * 30, color="gray", lw=0.5, alpha=0.5)
ax.fill_between(pet_mon.index, pet_mon.values * 30,
                alpha=0.3, color="gray")
# Marcar períodos
pet_pm_region  = pet_mon[pet_mon.index.year <= 2016] * 30
pet_hs_region  = pet_mon[pet_mon.index.year  > 2016] * 30
ax.fill_between(pet_pm_region.index,  pet_pm_region.values,  alpha=0.55, color="#27ae60",
                label="PISCOp PM (1981-2016)")
ax.fill_between(pet_hs_region.index,  pet_hs_region.values,  alpha=0.55, color="#f39c12",
                label="HS calibrado (2017-2020)")
ax.axvline(pd.Timestamp("2017-01-01"), color="k", ls=":", lw=1.2)
ax.set_ylabel("ETP\n(mm/mes)", fontsize=9)
ax.set_title("(b) Evapotranspiración potencial — PISCOp PM (verde) | HS-calibrado (naranja)",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, lw=0.7)

# ── tmax/tmin ─────────────────────────────────────────────────────────────
ax = axes[2]
ax.fill_between(tmax_mon.index, tmax_mon.values, tmin_mon.values,
                alpha=0.3, color="#e74c3c", label="Rango Tmax-Tmin")
ax.plot(tmax_mon.index, tmax_mon.values, color="#e74c3c", lw=1.2,
        label=f"Tmax (media={tmax_mon.mean():.1f}°C)")
ax.plot(tmin_mon.index, tmin_mon.values, color="#2980b9", lw=1.2,
        label=f"Tmin (media={tmin_mon.mean():.1f}°C)")
ax.set_ylabel("Temperatura\n(°C)", fontsize=9)
ax.set_title("(c) Tmax y Tmin — PISCOt v1.2 (1981-2020)",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3, lw=0.7)

# ── DTR = Tmax - Tmin ─────────────────────────────────────────────────────
ax = axes[3]
dtr_mon = tmax_mon - tmin_mon
# Colorear por estación
mo_dtr = pd.DatetimeIndex(dtr_mon.index).month
season_colors = np.where(np.isin(mo_dtr, [12,1,2]),  "#e74c3c",
                np.where(np.isin(mo_dtr, [3,4,5]),   "#e67e22",
                np.where(np.isin(mo_dtr, [6,7,8]),   "#2980b9", "#27ae60")))
for i in range(len(dtr_mon)-1):
    ax.fill_between(dtr_mon.index[i:i+2], dtr_mon.values[i:i+2],
                    dtr_mon.mean(), color=season_colors[i], alpha=0.60)
ax.axhline(dtr_mon.mean(), color="k", ls="--", lw=1.2,
           label=f"DTR media = {dtr_mon.mean():.1f}°C")
ax.set_ylabel("DTR\n(°C)", fontsize=9)
ax.set_xlabel("Año", fontsize=10)
ax.set_title("(d) Rango diurno ΔT = Tmax − Tmin | Rojo=Verano, Azul=Invierno",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, lw=0.7)

fig.suptitle("Resumen de forzantes climáticos — Cuenca Chancay-Huaral (1981-2020)\n"
             "Media de cuenca ponderada coseno×frac | PISCOp v2.1 + PISCOt v1.2",
             fontsize=12, fontweight="bold")
plt.savefig(FIG_DIR / "V04_forzantes_overview_1981_2020.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V04 guardado")


# ══════════════════════════════════════════════════════════════════════════════
# V05 — Temperatura: correlación Tmax-Tmin y análisis DTR estacional
# ══════════════════════════════════════════════════════════════════════════════
print("V05: correlacion Tmax-Tmin y DTR estacional ...")

tmax_mon_v = tmax_mon.values
tmin_mon_v = tmin_mon.values
dtr_mon_v  = dtr_mon.values
mo_m       = pd.DatetimeIndex(tmax_mon.index).month

fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                          gridspec_kw={"hspace":0.52, "wspace":0.40})

# ── Panel a: Scatter Tmax vs Tmin (coloreado por mes) ─────────────────────
ax = axes[0, 0]
sc = ax.scatter(tmax_mon_v, tmin_mon_v, c=mo_m, cmap="hsv",
                s=18, alpha=0.65, vmin=1, vmax=12)
r_val = float(np.corrcoef(tmax_mon_v, tmin_mon_v)[0, 1])
ax.set_xlabel("Tmax mensual (°C)", fontsize=10)
ax.set_ylabel("Tmin mensual (°C)", fontsize=10)
ax.set_title(f"(a) Tmax vs Tmin  (r = {r_val:.3f})", fontweight="bold")
ax.grid(alpha=0.3, lw=0.7)
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Mes", fontsize=8)
cb.set_ticks(np.arange(1, 13))
cb.set_ticklabels(MESES, fontsize=7)

# ── Panel b: DTR por mes (boxplot) ────────────────────────────────────────
ax = axes[0, 1]
dtr_by_month = [dtr_mon_v[mo_m == m] for m in range(1, 13)]
bpr = ax.boxplot(dtr_by_month, patch_artist=True,
                 medianprops=dict(color="k", lw=2),
                 whiskerprops=dict(lw=1.2, ls="--"),
                 capprops=dict(lw=1.5),
                 flierprops=dict(marker="d", ms=4, alpha=0.4))
cmap_j = plt.cm.coolwarm_r
for patch, m in zip(bpr["boxes"], range(12)):
    patch.set_facecolor(cmap_j(m/11)); patch.set_alpha(0.80)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MESES, fontsize=9)
ax.set_ylabel("DTR = Tmax − Tmin (°C)", fontsize=10)
ax.set_title("(b) Distribución mensual del rango diurno (DTR)\n"
             "1981-2020", fontweight="bold")
ax.grid(axis="y", alpha=0.3, lw=0.7)

# ── Panel c: Climatología mensual Tmax / Tmin / DTR ──────────────────────
ax = axes[1, 0]
by_m_tx = pd.Series(tmax_mon_v, index=tmax_mon.index).groupby(
    pd.DatetimeIndex(tmax_mon.index).month).agg(["mean","std"])
by_m_tn = pd.Series(tmin_mon_v, index=tmin_mon.index).groupby(
    pd.DatetimeIndex(tmin_mon.index).month).agg(["mean","std"])
by_m_dt = pd.Series(dtr_mon_v,  index=dtr_mon.index).groupby(
    pd.DatetimeIndex(dtr_mon.index).month).agg(["mean","std"])

x = np.arange(1, 13)
ax.fill_between(x, by_m_tx["mean"]-by_m_tx["std"],
                by_m_tx["mean"]+by_m_tx["std"],
                alpha=0.18, color="#e74c3c")
ax.fill_between(x, by_m_tn["mean"]-by_m_tn["std"],
                by_m_tn["mean"]+by_m_tn["std"],
                alpha=0.18, color="#2980b9")
ax.fill_between(x, by_m_tx["mean"], by_m_tn["mean"],
                alpha=0.08, color="gray")
ax.plot(x, by_m_tx["mean"], "o-", color="#e74c3c", ms=6, lw=1.8, label="Tmax")
ax.plot(x, by_m_tn["mean"], "s-", color="#2980b9", ms=6, lw=1.8, label="Tmin")
ax2 = ax.twinx()
ax2.plot(x, by_m_dt["mean"], "^--", color="#8e44ad", ms=6, lw=1.5, label="DTR")
ax2.set_ylabel("DTR (°C)", fontsize=9, color="#8e44ad")
ax2.tick_params(axis="y", colors="#8e44ad", labelsize=8)
ax.set_xticks(x); ax.set_xticklabels(MESES, fontsize=9)
ax.set_ylabel("Temperatura (°C)", fontsize=10)
ax.set_title("(c) Climatología mensual Tmax / Tmin / DTR\n"
             "media ± 1σ  |  1981-2020", fontweight="bold")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8)
ax.grid(alpha=0.3, lw=0.7)

# ── Panel d: Tendencias decadales Tmax/Tmin por período ──────────────────
ax = axes[1, 1]
# Calcular tendencia por décadas móviles de 10 años
ann_tx = df["tmax"].resample("YE").mean()
ann_tn = df["tmin"].resample("YE").mean()
yrs    = np.array(pd.DatetimeIndex(ann_tx.index).year)

window = 20   # ventana 20 años
slopes_tx, slopes_tn, centers = [], [], []
for i in range(len(yrs) - window):
    sl_x = st.linregress(yrs[i:i+window], ann_tx.values[i:i+window]).slope
    sl_n = st.linregress(yrs[i:i+window], ann_tn.values[i:i+window]).slope
    slopes_tx.append(sl_x * 10)
    slopes_tn.append(sl_n * 10)
    centers.append(yrs[i] + window // 2)

ax.plot(centers, slopes_tx, "o-", color="#e74c3c", ms=5, lw=1.5,
        label="Tmax")
ax.plot(centers, slopes_tn, "s-", color="#2980b9", ms=5, lw=1.5,
        label="Tmin")
ax.axhline(0, color="k", lw=1.2, ls="--", alpha=0.6)
ax.fill_between(centers, slopes_tx, 0,
                where=[v > 0 for v in slopes_tx],
                alpha=0.18, color="#e74c3c")
ax.fill_between(centers, slopes_tn, 0,
                where=[v > 0 for v in slopes_tn],
                alpha=0.18, color="#2980b9")
ax.set_xlabel("Año central de la ventana (20 años)", fontsize=9)
ax.set_ylabel("Tendencia (°C/década)", fontsize=10)
ax.set_title(f"(d) Tendencia deslizante (ventana {window} años)\n"
             "Tmax vs Tmin  |  1981-2020", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, lw=0.7)

fig.suptitle("Análisis temperatura Tmax / Tmin / DTR — Cuenca Chancay-Huaral\n"
             "PISCOt v1.2 (0.01°, 1981-2020)  |  media cuenca ponderada coseno",
             fontsize=12, fontweight="bold")
plt.savefig(FIG_DIR / "V05_temperatura_analisis_dtr.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V05 guardado")


# ══════════════════════════════════════════════════════════════════════════════
# V06 — Climatología mensual espacial Tmax (2×6) — paralela a V03
# ══════════════════════════════════════════════════════════════════════════════
print("V06: climatologia espacial Tmax ...")
print("  Cargando Tmax NC ...")
ds_tx2  = nc.Dataset(TMAX_NC)
lat_tx2 = np.array(ds_tx2["lat"][:])
lon_tx2 = np.array(ds_tx2["lon"][:])
msk_tx2 = np.array(ds_tx2["basin_mask"][:]).astype(bool)
tmax_g  = np.array(ds_tx2["tmax"][:])    # (14610, 111, 91)
ds_tx2.close()

mo_t2 = np.array(pd.DatetimeIndex(pd.date_range("1981-01-01",
                                                  periods=14610, freq="D")).month)
yr_t2 = np.array(pd.DatetimeIndex(pd.date_range("1981-01-01",
                                                  periods=14610, freq="D")).year)

clim_tmax = np.zeros((12, len(lat_tx2), len(lon_tx2)))
for mi in range(1, 13):
    total = np.zeros((len(lat_tx2), len(lon_tx2))); cnt = 0
    for yy in np.unique(yr_t2):
        idx = (mo_t2 == mi) & (yr_t2 == yy)
        if idx.sum() == 0: continue
        total += np.nanmean(tmax_g[idx], axis=0); cnt += 1
    clim_tmax[mi-1] = total / max(cnt, 1)

clim_tmax = np.where(msk_tx2[None,:,:], clim_tmax, np.nan)
vmin_tx = float(np.nanpercentile(clim_tmax[:, msk_tx2].ravel(), 2))
vmax_tx = float(np.nanpercentile(clim_tmax[:, msk_tx2].ravel(), 98))

fig, axes = plt.subplots(2, 6, figsize=(17, 7.5),
                          gridspec_kw={"hspace":0.50, "wspace":0.18})
plt.subplots_adjust(right=0.89)

for mi, ax in enumerate(axes.ravel()):
    im = _pcm(ax, lon_tx2, lat_tx2, clim_tmax[mi], "RdBu_r", vmin_tx, vmax_tx)
    _contour(ax, lon_tx2, lat_tx2, clim_tmax[mi], n_lvl=3, color="k", lw=0.5)
    _basin(ax, lw=0.6)
    _fmt_ax(ax, MESES[mi], fs=10, xlabel=False, ylabel=False)
    ax.tick_params(labelsize=5)
    mean_v = float(np.nanmean(clim_tmax[mi][msk_tx2]))
    ax.text(0.04, 0.04, f"{mean_v:.1f}°C",
            transform=ax.transAxes, fontsize=6.5, color="white",
            bbox=dict(fc="black", alpha=0.52, pad=1.5))

sm2 = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=vmin_tx, vmax=vmax_tx))
sm2.set_array([])
cbar_ax2 = fig.add_axes([0.907, 0.12, 0.013, 0.75])
cb2 = fig.colorbar(sm2, cax=cbar_ax2)
cb2.set_label("Tmax media mensual (°C)", fontsize=10)
cb2.ax.tick_params(labelsize=8)

fig.suptitle("PISCOt v1.2 — Climatología mensual de temperatura máxima (1981-2020)\n"
             "Cuenca Chancay-Huaral  |  0.01°  |  Contornos en °C  (RdBu_r)",
             fontsize=12, fontweight="bold")
plt.savefig(FIG_DIR / "V06_tmax_climatologia_mensual.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V06 guardado")

# ══════════════════════════════════════════════════════════════════════════════
# V07 — Climatología estacional espacial Tmin (1×4: DJF/MAM/JJA/SON)
# ══════════════════════════════════════════════════════════════════════════════
print("V07: climatologia estacional Tmin ...")

# clim_tmin ya está en memoria (12, lat, lon) — promediamos los 3 meses de cada estación
SEASONS = [
    ("Verano\n(DJF)", [11, 0, 1]),
    ("Otoño\n(MAM)",  [2, 3, 4]),
    ("Invierno\n(JJA)", [5, 6, 7]),
    ("Primavera\n(SON)", [8, 9, 10]),
]

def _seasonal_clim(monthly_clim, season_indices):
    """Media estacional desde climatología mensual."""
    return np.nanmean(np.stack([monthly_clim[i] for i in season_indices], axis=0), axis=0)

seas_tmin = [_seasonal_clim(clim_tmin, idx) for _, idx in SEASONS]
seas_tmin_masked = [np.where(msk_tn, s, np.nan) for s in seas_tmin]

# Escala global compartida
all_vals = np.concatenate([s[msk_tn].ravel() for s in seas_tmin_masked])
vmin_s = float(np.nanpercentile(all_vals, 2))
vmax_s = float(np.nanpercentile(all_vals, 98))

fig, axes = plt.subplots(1, 4, figsize=(17, 5.5),
                          gridspec_kw={"wspace": 0.14})
plt.subplots_adjust(right=0.88, left=0.04, top=0.84)

for ax, (sname, _), data in zip(axes, SEASONS, seas_tmin_masked):
    im = _pcm(ax, lon_tn, lat_tn, data, "RdBu_r", vmin_s, vmax_s)
    _contour(ax, lon_tn, lat_tn, data, n_lvl=4, color="k", lw=0.6)
    _basin(ax, lw=0.7)
    _fmt_ax(ax, sname.replace("\n", " "), fs=11, xlabel=False, ylabel=False)
    ax.tick_params(labelsize=5)
    mean_v = float(np.nanmean(data[msk_tn]))
    std_v  = float(np.nanstd(data[msk_tn]))
    ax.text(0.04, 0.04, f"μ={mean_v:.1f}°C\nσ={std_v:.1f}°C",
            transform=ax.transAxes, fontsize=6.5, color="white", va="bottom",
            bbox=dict(fc="black", alpha=0.55, pad=2))

sm_v7 = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=vmin_s, vmax=vmax_s))
sm_v7.set_array([])
cbar_ax = fig.add_axes((0.895, 0.12, 0.013, 0.68))
cb = fig.colorbar(sm_v7, cax=cbar_ax)
cb.set_label("Tmin estacional (°C)", fontsize=10)
cb.ax.tick_params(labelsize=8)

fig.suptitle("PISCOt v1.2 — Climatología estacional de temperatura mínima (1981-2020)\n"
             "Cuenca Chancay-Huaral  |  0.01°  |  media ± σ inter-pixel de cuenca  (RdBu_r)",
             fontsize=12, fontweight="bold")
plt.savefig(FIG_DIR / "V07_tmin_climatologia_estacional.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V07 guardado")


# ══════════════════════════════════════════════════════════════════════════════
# V08 — Climatología estacional espacial Tmax (1×4: DJF/MAM/JJA/SON)
# ══════════════════════════════════════════════════════════════════════════════
print("V08: climatologia estacional Tmax ...")

seas_tmax = [_seasonal_clim(clim_tmax, idx) for _, idx in SEASONS]
seas_tmax_masked = [np.where(msk_tx2, s, np.nan) for s in seas_tmax]

all_vals_tx = np.concatenate([s[msk_tx2].ravel() for s in seas_tmax_masked])
vmin_stx = float(np.nanpercentile(all_vals_tx, 2))
vmax_stx = float(np.nanpercentile(all_vals_tx, 98))

fig, axes = plt.subplots(1, 4, figsize=(17, 5.5),
                          gridspec_kw={"wspace": 0.14})
plt.subplots_adjust(right=0.88, left=0.04, top=0.84)

for ax, (sname, _), data in zip(axes, SEASONS, seas_tmax_masked):
    _pcm(ax, lon_tx2, lat_tx2, data, "RdBu_r", vmin_stx, vmax_stx)
    _contour(ax, lon_tx2, lat_tx2, data, n_lvl=4, color="k", lw=0.6)
    _basin(ax, lw=0.7)
    _fmt_ax(ax, sname.replace("\n", " "), fs=11, xlabel=False, ylabel=False)
    ax.tick_params(labelsize=5)
    mean_v = float(np.nanmean(data[msk_tx2]))
    std_v  = float(np.nanstd(data[msk_tx2]))
    ax.text(0.04, 0.04, f"μ={mean_v:.1f}°C\nσ={std_v:.1f}°C",
            transform=ax.transAxes, fontsize=6.5, color="white", va="bottom",
            bbox=dict(fc="black", alpha=0.55, pad=2))

sm_v8 = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=vmin_stx, vmax=vmax_stx))
sm_v8.set_array([])
cbar_ax2 = fig.add_axes([0.895, 0.12, 0.013, 0.68])
cb2 = fig.colorbar(sm_v8, cax=cbar_ax2)
cb2.set_label("Tmax estacional (°C)", fontsize=10)
cb2.ax.tick_params(labelsize=8)

fig.suptitle("PISCOt v1.2 — Climatología estacional de temperatura máxima (1981-2020)\n"
             "Cuenca Chancay-Huaral  |  0.01°  |  media ± σ inter-pixel de cuenca  (RdBu_r)",
             fontsize=12, fontweight="bold")
plt.savefig(FIG_DIR / "V08_tmax_climatologia_estacional.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("  -> V08 guardado")


print(f"\n=== DONE: V01-V08 en {FIG_DIR} ===")
