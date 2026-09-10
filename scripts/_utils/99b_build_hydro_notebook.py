#!/usr/bin/env python3
"""Script 99b: Genera notebook 05 de Hidroestadística completa."""
import json, uuid
from pathlib import Path

ROOT   = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks"
KERNEL    = {"display_name":"HidroAlerta (Python 3.14)","language":"python","name":"hidroalerta"}
LANG_INFO = {"name":"python","version":"3.14.3"}

def _id(): return uuid.uuid4().hex[:16]
def md(s):  return {"cell_type":"markdown","id":_id(),"metadata":{},"source":s}
def code(s):
    return {"cell_type":"code","id":_id(),"metadata":{},"source":s,
            "outputs":[],"execution_count":None}
def nb(cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":KERNEL,"language_info":LANG_INFO},
            "cells":cells}
def save(notebook, path):
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {path.name}")

# Todas las celdas de código se escriben con comillas simples para docstrings
# para evitar conflicto con las triple-comillas dobles del script generador.

SETUP = '''
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import yaml
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
import pymannkendall as mk
import hydroeval as he

PROJECT_ROOT = Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
plt.rcParams.update({"figure.dpi":130,"font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
FIG  = PROJECT_ROOT / "outputs/figures/eda"
TABS = PROJECT_ROOT / "outputs/tables"
FIG.mkdir(parents=True,exist_ok=True); TABS.mkdir(parents=True,exist_ok=True)

with open(PROJECT_ROOT/"configs/paths.yaml", encoding="utf-8") as f:
    P = yaml.safe_load(f)
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
print("Setup OK — statsmodels, pymannkendall, hydroeval cargados")
'''

LOAD_DATA = '''
# PISCO cuenca (diario) — limitado a 2019-12-31 (2020 tiene NaNs)
pisco = pd.read_csv(PROJECT_ROOT/P["pisco"]["basin_mean"],
                    index_col=0, parse_dates=True)
pisco.index.name = "date"
pisco = pisco.loc[:"2019-12-31"]  # excluir 2020 (NaNs / relleno inconsistente)

# Caudal SNIRH
q_all = pd.read_csv(PROJECT_ROOT/P["snirh"]["daily_q"],
                    index_col=0, parse_dates=True)
q_all.columns = [c.replace("q_","").replace("_"," ").title() for c in q_all.columns]
q_col = q_all.columns[q_all.notna().sum().argmax()]
q_obs = q_all[q_col].dropna().rename("Q_m3s")

# SENAMHI precipitación
pr_senamhi = pd.read_csv(PROJECT_ROOT/P["senamhi"]["precip_all"],
                          index_col=0, parse_dates=True)
pr_senamhi.columns = [c.replace("pr_","").replace("_"," ").title()
                       for c in pr_senamhi.columns]

print(f"PISCO  : {pisco.shape}  {pisco.index.min().date()} -> {pisco.index.max().date()}")
print(f"Q obs  : {len(q_obs)} dias  {q_obs.index.min().date()} -> {q_obs.index.max().date()}")
print(f"SENAMHI: {pr_senamhi.shape}  {pr_senamhi.index.min().date()} -> {pr_senamhi.index.max().date()}")
'''

STL_CODE = '''
def plot_stl(series, title, period, color, fig_name, log=False):
    s = series.resample("ME").mean().dropna()
    if log:
        s = np.log1p(s)
        sfx = " [log1p]"
    else:
        sfx = ""

    stl = STL(s, period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    parts = [("Observado", s, color, 0.9),
             ("Tendencia", res.trend, "black", 1.0),
             ("Estacional", res.seasonal, "#2ca02c", 0.8),
             ("Residual", res.resid, "#d62728", 0.7)]
    for ax, (label, data, c, alpha) in zip(axes, parts):
        ax.plot(data.index, data, lw=1.2, color=c, alpha=alpha)
        if label == "Residual":
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.fill_between(data.index, data, alpha=0.25, color=c)
        ax.set_ylabel(label + sfx, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Descomposicion STL — {title}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG/fig_name, dpi=150, bbox_inches="tight")
    plt.show()

    vt = res.trend.var(); vs = res.seasonal.var(); vr = res.resid.var()
    vT = vt + vs + vr
    print(f"  Tendencia  : {100*vt/vT:.1f}%")
    print(f"  Estacional : {100*vs/vT:.1f}%")
    print(f"  Residual   : {100*vr/vT:.1f}%")
    return res

print("-- STL Precipitacion PISCO --")
stl_pr = plot_stl(pisco["pr"], "Precipitacion PISCO (media cuenca)",
                  period=12, color="#2171b5",
                  fig_name="E05_01_stl_precipitacion.png")
print("\\n-- STL Caudal Santo Domingo (log) --")
stl_q  = plot_stl(q_obs, "Caudal Santo Domingo (log1p)",
                  period=12, color="#08519c",
                  fig_name="E05_02_stl_caudal.png", log=True)
'''

TESTS_CODE = '''
def pettitt_test(x):
    n = len(x)
    U = np.array([np.sum(np.sign(x[t] - x[:t])) for t in range(1, n)])
    K = np.abs(U).max()
    p = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    tau = int(np.argmax(np.abs(U))) + 1
    return K, p, tau

def run_tests(series, name, freq="ME"):
    s = series.resample(freq).mean().dropna()
    x = s.values
    # ADF
    adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
    # KPSS
    try:
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    # Mann-Kendall
    mk_res = mk.original_test(x)
    # Pettitt
    K, pett_p, tau = pettitt_test(x)
    cd = s.index[tau] if tau < len(s) else None

    print(f"\\n{'='*55}\\n  {name}\\n{'='*55}")
    print(f"  ADF   : stat={adf_stat:.3f}  p={adf_p:.4f}  "
          f"{'ESTACIONARIA' if adf_p<0.05 else 'no estacionaria'}")
    print(f"  KPSS  : stat={kpss_stat:.3f}  p={kpss_p:.4f}  "
          f"{'NO estacionaria' if kpss_p<0.05 else 'estacionaria'}")
    print(f"  MK    : {mk_res.trend}  slope={mk_res.slope:.5f}/mes  p={mk_res.p:.4f}")
    print(f"  Pettitt: p={pett_p:.4f}  "
          f"{'cambio en '+str(cd.date()) if (pett_p<0.05 and cd) else 'sin cambio'}")
    return {"serie":name,"adf_p":adf_p,"kpss_p":kpss_p,
            "mk_trend":mk_res.trend,"mk_slope":mk_res.slope,"mk_p":mk_res.p,
            "pettitt_p":pett_p}

results = []
results.append(run_tests(pisco["pr"],   "Precipitacion PISCO"))
results.append(run_tests(pisco["tmax"], "Temperatura maxima PISCO"))
results.append(run_tests(q_obs,         "Caudal Santo Domingo"))
best_stn = pr_senamhi.notna().sum().idxmax()
results.append(run_tests(pr_senamhi[best_stn], f"PR SENAMHI {best_stn}"))

df_tests = pd.DataFrame(results)
df_tests.to_csv(TABS/"T05_tests_estadisticos.csv", index=False)
display(df_tests.round(4))
'''

DIST_CODE = '''
def fit_dists(data, name, xlabel, fig_name):
    data = data[data > 0].dropna()
    x_r  = np.linspace(data.min()*0.5, data.max()*1.1, 500)
    dists = {"Gamma":st.gamma, "Log-Normal":st.lognorm,
              "GEV":st.genextreme, "Log-Pearson III":st.pearson3}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(data, bins=40, density=True, color="#6baed6",
                 alpha=0.6, edgecolor="white", label="Datos")
    fit_results = []
    colors_ = ["#e41a1c","#377eb8","#4daf4a","#984ea3"]
    for (dname, dist), col in zip(dists.items(), colors_):
        try:
            if dname == "Gamma":
                params = dist.fit(data, floc=0)
            elif dname == "Log-Pearson III":
                params = dist.fit(np.log(data))
            else:
                params = dist.fit(data)
            if dname == "Log-Pearson III":
                pdf_v = dist.pdf(np.log(x_r), *params) / x_r
                D, p_ks = st.kstest(np.log(data), lambda x: dist.cdf(x, *params))
                ll = np.sum(dist.logpdf(np.log(data), *params) - np.log(data))
            else:
                pdf_v = dist.pdf(x_r, *params)
                D, p_ks = st.kstest(data, lambda x: dist.cdf(x, *params))
                ll = np.sum(dist.logpdf(data, *params))
            aic = 2*len(params) - 2*ll
            axes[0].plot(x_r, pdf_v, lw=2.2, color=col,
                         label=f"{dname} (KS p={p_ks:.3f})")
            fit_results.append({"Dist":dname,"KS_D":round(D,4),
                                  "KS_p":round(p_ks,4),"AIC":round(aic,1),
                                  "params":params})
        except Exception as e:
            print(f"  Fallo {dname}: {e}")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel("Densidad")
    axes[0].set_title(f"Ajuste — {name}"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    best = min(fit_results, key=lambda r: r["AIC"])
    bd, bp = best["Dist"], best["params"]
    bd_dist = dists[bd]
    if bd == "Log-Pearson III":
        osm, osr = st.probplot(np.log(data), dist=bd_dist, sparams=bp, fit=False)
    else:
        osm, osr = st.probplot(data, dist=bd_dist, sparams=bp, fit=False)
    axes[1].scatter(osm, osr, s=15, color="#08519c", alpha=0.7)
    lims = [min(osm.min(),osr.min()), max(osm.max(),osr.max())]
    axes[1].plot(lims, lims, "r--", lw=1.5)
    axes[1].set_xlabel("Cuantiles teoricos"); axes[1].set_ylabel("Cuantiles obs")
    axes[1].set_title(f"Q-Q plot — {bd} (AIC={best['AIC']:.0f})")
    axes[1].grid(alpha=0.3)
    plt.suptitle(f"Distribucion de probabilidad — {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG/fig_name, dpi=150, bbox_inches="tight"); plt.show()
    df = pd.DataFrame(fit_results).drop(columns="params").sort_values("AIC")
    display(df)
    return df

pr_anual  = (pisco["pr"].resample("YE").mean() * 365).rename("pr_anual_mm")
df_pr = fit_dists(pr_anual, "Precipitacion anual PISCO (media cuenca)",
                  "Precipitacion anual (mm)", "E05_03_dist_precipitacion.png")

q_max = q_obs.resample("YE").max()
df_q  = fit_dists(q_max, "Caudal maximo anual — Santo Domingo",
                  "Q maximo anual (m3/s)", "E05_04_dist_caudal_max.png")
'''

RETORNO_CODE = '''
TRs = [2, 5, 10, 25, 50, 100, 200, 500]
exc = [1 - 1/tr for tr in TRs]

q_max = q_obs.resample("YE").max().dropna()
log_q = np.log(q_max.values)
params_lp3 = st.pearson3.fit(log_q)
q_tr = np.exp(st.pearson3.ppf(exc, *params_lp3))

tr_df = pd.DataFrame({"TR_anios": TRs, "Q_LP3_m3s": q_tr.round(1)})
tr_df.to_csv(TABS/"T05_periodos_retorno.csv", index=False)
print("Caudales de diseno — Log-Pearson III (Santo Domingo):")
display(tr_df)

# Grafico papel probabilidad
fig, ax = plt.subplots(figsize=(10, 5))
n = len(q_max)
prob_exc = np.arange(1, n+1) / (n+1)
tr_emp   = 1 / prob_exc
ax.semilogx(tr_emp[::-1], q_max.sort_values().values, "o",
            color="#08519c", ms=7, label="Datos empiricos")
TRs_f = np.logspace(np.log10(1.01), np.log10(500), 200)
q_f   = np.exp(st.pearson3.ppf(1 - 1/TRs_f, *params_lp3))
ax.semilogx(TRs_f, q_f, "-", color="#d94701", lw=2.5, label="Log-Pearson III")

# IC Bootstrap
bs_q = []
for _ in range(300):
    bs = np.random.choice(log_q, size=len(log_q), replace=True)
    p  = st.pearson3.fit(bs)
    bs_q.append(np.exp(st.pearson3.ppf(1 - 1/TRs_f, *p)))
bs_arr = np.array(bs_q)
ax.fill_between(TRs_f, np.percentile(bs_arr,5,axis=0),
                np.percentile(bs_arr,95,axis=0), alpha=0.2, color="#d94701",
                label="IC 90% bootstrap")
for tr in [10, 50, 100]:
    q_ref = np.exp(st.pearson3.ppf(1-1/tr, *params_lp3))
    ax.axhline(q_ref, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(tr, q_ref+1, f"TR{tr}={q_ref:.0f} m3/s", fontsize=8, color="gray")
ax.set_xlabel("Periodo de retorno (anios)"); ax.set_ylabel("Q maximo anual (m3/s)")
ax.set_title("Analisis de frecuencia — Caudal maximo anual · Santo Domingo",
             fontweight="bold")
ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(FIG/"E05_05_periodos_retorno.png", dpi=150, bbox_inches="tight")
plt.show()
'''

BUDYKO_CODE = '''
# NOTA: PISCO cubre 1981-2019 (cortado en 2019-12-31 para consistencia);
# caudal SNIRH desde sep-2020 -> sin solapamiento directo.
# Estrategia: usar PISCO P y PET para calcular indice de aridez anual;
# estimar ET por ecuacion de Budyko (Turc-Pike); Q_est = P - ET.
# Cuando CHIRPS 2020-2026 este disponible se completara con Q observado.

AREA_KM2 = 3062.6

def budyko_turcpike(P, PET):
    phi = PET / np.maximum(P, 1e-6)
    return np.sqrt(phi * np.tanh(1/phi) * (1 - np.exp(-phi))) * P

P_m   = (pisco["pr"]  * 30).resample("ME").mean()
PET_m = (pisco["pet"] * 30).resample("ME").mean()
bud   = pd.DataFrame({"P": P_m, "PET": PET_m}).dropna()
bud["ET_bud"] = budyko_turcpike(bud["P"], bud["PET"]).clip(upper=bud["P"])
bud["Q_bud"]  = np.maximum(0, bud["P"] - bud["ET_bud"])

annual = bud.resample("YE").sum()
annual["phi"]      = annual["PET"] / annual["P"]
annual["ET_ratio"] = annual["ET_bud"] / annual["P"]
annual["Q_ratio"]  = annual["Q_bud"]  / annual["P"]

phi_r = np.linspace(0.1, 3.5, 300)
et_b  = np.sqrt(phi_r * np.tanh(1/phi_r) * (1 - np.exp(-phi_r)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(phi_r, et_b, "k-", lw=2.5, label="Curva Budyko (Turc-Pike)")
ax.plot(phi_r, np.minimum(phi_r, 1), "gray", ls="--", lw=1.5,
        alpha=0.7, label="Lim. energetico/hidrico")
sc = ax.scatter(annual["phi"], annual["ET_ratio"],
                c=annual.index.year, cmap="viridis", s=80,
                edgecolors="white", lw=0.8, zorder=5, label="PISCO 1981-2019")
plt.colorbar(sc, ax=ax, label="Anio", shrink=0.7)
ax.set_xlabel("Indice de aridez PET/P")
ax.set_ylabel("Fraccion ET/P (Budyko)")
ax.set_title("Diagrama de Budyko — Chancay-Huaral (ET estimada, PISCO)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
phi_max = float(np.nanmax(annual["phi"].values)) if annual["phi"].notna().any() else 2.0
ax.set_xlim(0, max(phi_max * 1.2, 1.5)); ax.set_ylim(0, 1.05)

ax = axes[1]
cb = bud.groupby(bud.index.month).mean()
x  = np.arange(1, 13)
ax.bar(x-0.3, cb["P"],      0.25, label="P PISCO (mm/mes)",    color="#2171b5", alpha=0.8)
ax.bar(x,     cb["ET_bud"], 0.25, label="ET Budyko (estimada)", color="#d62728", alpha=0.8)
ax.bar(x+0.3, cb["Q_bud"],  0.25, label="Q estimado (mm/mes)", color="#08519c", alpha=0.8)
ax.plot(x, cb["PET"], "o--", color="orange", lw=2, ms=5, label="PET")
ax.set_xticks(x); ax.set_xticklabels(MESES, fontsize=8)
ax.set_ylabel("mm/mes"); ax.set_title("Balance hidrico mensual (PISCO 1981-2019)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plt.suptitle("Balance Hidrico — Marco de Budyko (ET estimada Turc-Pike)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG/"E05_06_budyko_balance.png", dpi=150, bbox_inches="tight")
plt.show()
phi_mean = annual["phi"].mean()
print(f"Coef. escorrentia Budyko  : {annual['Q_ratio'].mean():.3f}")
print(f"Indice de aridez medio    : {phi_mean:.2f}")
print(f"Zona climatica            : {'Semiarido' if phi_mean>1 else 'Subhumedo'}")
print("NOTA: Q observado (SNIRH 2020+) no solapa con PISCO (1981-2019).")
print("      Balance completo requiere CHIRPS 2020-2026 (pendiente).")
'''

DOBLE_MASA_CODE = '''
t0 = max(pr_senamhi.index.min(), pisco.index.min())
t1 = min(pr_senamhi.index.max(), pisco.index.max())
pr_obs_m  = pr_senamhi.loc[t0:t1].resample("ME").sum()
pr_pisco_m= (pisco.loc[t0:t1,"pr"] * 30).resample("ME").mean()

good = pr_obs_m.columns[pr_obs_m.notna().mean() > 0.7].tolist()[:6]
nf   = len(good)
ncols = 3; nrows = max(1, -(-nf//ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5*nrows), squeeze=False)
axf = axes.ravel()

for j, stn in enumerate(good):
    idx = pr_obs_m[stn].dropna().index.intersection(pr_pisco_m.dropna().index)
    if len(idx) < 12:
        continue
    obs_c = pr_obs_m.loc[idx, stn].cumsum()
    pis_c = pr_pisco_m.loc[idx].cumsum()
    ax = axf[j]
    ax.scatter(pis_c, obs_c, c=range(len(idx)), cmap="RdYlGn", s=12, alpha=0.8)
    sl, ic, r, p_v, _ = st.linregress(pis_c, obs_c)
    xf = np.array([pis_c.min(), pis_c.max()])
    ax.plot(xf, sl*xf+ic, "r--", lw=2, label=f"r={r:.2f}  slope={sl:.2f}")
    ax.set_xlabel("Acum PISCO (mm)"); ax.set_ylabel("Acum SENAMHI (mm)")
    ax.set_title(stn, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    if abs(sl-1) > 0.15:
        ax.text(0.05, 0.92, f"sesgo {(sl-1)*100:+.0f}%", transform=ax.transAxes,
                fontsize=8, color="red", va="top",
                bbox=dict(fc="white", alpha=0.7, ec="red"))

for ax in axf[nf:]: ax.set_visible(False)
plt.suptitle("Curva de Doble Masa — SENAMHI vs PISCO", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG/"E05_07_doble_masa.png", dpi=150, bbox_inches="tight")
plt.show()
'''

QM_CODE = '''
def quantile_mapping(obs, model, model_full):
    p_cal   = np.linspace(0, 1, 100)
    obs_q   = np.quantile(obs.dropna(), p_cal)
    mod_q   = np.quantile(model.dropna(), p_cal)
    corrected = np.interp(model_full.values, mod_q, obs_q,
                          left=obs_q[0], right=obs_q[-1])
    return pd.Series(corrected, index=model_full.index, name="pr_qm")

t0 = max(pr_senamhi.index.min(), pisco.index.min())
t1 = min(pr_senamhi.index.max(), pisco.index.max())
pr_obs_m   = pr_senamhi.loc[t0:t1].resample("ME").sum()
pr_pisco_m = (pisco.loc[t0:t1,"pr"]*30).resample("ME").mean()
pr_pisco_full = (pisco["pr"]*30).resample("ME").mean()

good = pr_obs_m.columns[pr_obs_m.notna().mean() > 0.7].tolist()[:3]
nf   = len(good)
fig, axes = plt.subplots(2, max(nf,1), figsize=(14, 7), squeeze=False)
qm_results = {}

for j, stn in enumerate(good):
    idx = pr_obs_m[stn].dropna().index.intersection(pr_pisco_m.dropna().index)
    if len(idx) < 24: continue
    obs_cal = pr_obs_m.loc[idx, stn]
    mod_cal = pr_pisco_m.loc[idx]
    qm_full = quantile_mapping(obs_cal, mod_cal, pr_pisco_full)
    qm_cal  = qm_full.loc[idx]

    rmse_r = np.sqrt(np.mean((mod_cal - obs_cal)**2))
    rmse_q = np.sqrt(np.mean((qm_cal   - obs_cal)**2))
    bias_r = (mod_cal.mean() - obs_cal.mean()) / obs_cal.mean() * 100
    bias_q = (qm_cal.mean()  - obs_cal.mean()) / obs_cal.mean() * 100
    qm_results[stn] = {"RMSE_raw":round(rmse_r,2),"RMSE_QM":round(rmse_q,2),
                        "Bias_raw%":round(bias_r,1),"Bias_QM%":round(bias_q,1)}

    pv = np.linspace(0.01,0.99,100)
    ax = axes[0,j]
    ax.plot(np.quantile(obs_cal,pv), np.quantile(mod_cal.dropna(),pv),
            "o", ms=4, color="#d62728", alpha=0.6, label=f"Raw RMSE={rmse_r:.1f}")
    ax.plot(np.quantile(obs_cal,pv), np.quantile(qm_cal.dropna(),pv),
            "o", ms=4, color="#2ca02c", alpha=0.6, label=f"QM  RMSE={rmse_q:.1f}")
    lims = [min(obs_cal.min(),mod_cal.min()), max(obs_cal.max(),mod_cal.max())]
    ax.plot(lims,lims,"k--",lw=1.5); ax.set_title(stn,fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xlabel("Q obs"); ax.set_ylabel("Q modelo")

    ax2 = axes[1,j]
    ax2.plot(obs_cal.index, obs_cal, lw=1.5, color="#08519c", label="Obs")
    ax2.plot(mod_cal.index, mod_cal, lw=1,   color="#d62728", alpha=0.7, label="Raw")
    ax2.plot(qm_cal.index,  qm_cal,  lw=1.5, color="#2ca02c", ls="--", label="QM")
    ax2.set_ylabel("mm/mes"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

plt.suptitle("Quantile Mapping — PISCO corregido con SENAMHI",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG/"E05_08_quantile_mapping.png", dpi=150, bbox_inches="tight")
plt.show()

df_qm = pd.DataFrame(qm_results).T
display(df_qm)
df_qm.to_csv(TABS/"T05_qm_resultados.csv")
print("\\nInterpretacion slope doble masa:")
print("  slope > 1: PISCO subestima respecto a SENAMHI")
print("  slope < 1: PISCO sobreestima")
'''

SUMMARY_CODE = '''
pr_anual_mm  = pisco["pr"].mean() * 365
pet_anual_mm = pisco["pet"].mean() * 365
phi_mean     = annual["phi"].mean()
rc_budyko    = annual["Q_ratio"].mean()
q_clean      = q_obs.dropna()

summary = [
    ("Periodo registro Q obs (SNIRH)", f"{q_obs.index.min().date()} -> {q_obs.index.max().date()}"),
    ("Q medio obs (m3/s)", f"{q_obs.mean():.2f}"),
    ("Q minimo / maximo obs (m3/s)", f"{q_obs.min():.2f} / {q_obs.max():.2f}"),
    ("Q50 mediana (m3/s)", f"{q_obs.median():.2f}"),
    ("Q90 / Q10 (m3/s)", f"{np.percentile(q_clean,90):.2f} / {np.percentile(q_clean,10):.2f}"),
    ("CV caudal (%)", f"{q_obs.std()/q_obs.mean()*100:.1f}"),
    ("Precip media PISCO 1981-2019 (mm/anio)", f"{pr_anual_mm:.0f}"),
    ("ETP media PISCO 1981-2019 (mm/anio)", f"{pet_anual_mm:.0f}"),
    ("Indice de aridez medio PET/P (PISCO)", f"{phi_mean:.2f}"),
    ("Coef. escorrentia Budyko (estimado)", f"{rc_budyko:.3f}"),
    ("Zona climatica (Thornthwaite)", "Semiarido a Subhumedo (gradiente altitudinal)"),
    ("Distribucion Q max (mejor ajuste)", "Log-Pearson III"),
    ("DEM recomendado", "Copernicus GLO-30 (30m, ESA 2022)"),
    ("Refs DEM Andes", "Uuemaa et al. 2020 Rem.Sens.; Hawker et al. 2022 ERL"),
]
df_sum = pd.DataFrame(summary, columns=["Indicador", "Valor"])
df_sum.to_csv(TABS/"T05_resumen_hidrologico.csv", index=False)
print("Resumen Hidrologico — Cuenca Chancay-Huaral")
display(df_sum)
'''

# ══════════════════════════════════════════════════════════════════════════════
NB05 = nb([

md("""\
# 05 · Hidroestadística y Análisis de Series
**Cuenca Chancay-Huaral** · HidroAlerta · Concurso ANA

Flujo de trabajo hidrológico estándar (WMO-168 §5):
1. Descomposición STL (tendencia · estacionalidad · residual)
2. Tests estadísticos (ADF, KPSS, Mann-Kendall, Pettitt)
3. Distribuciones de probabilidad + períodos de retorno
4. Balance hídrico — Marco de Budyko
5. Curva de doble masa SENAMHI vs PISCO
6. Corrección de sesgo por Quantile Mapping

**Referencias:** WMO-No. 168 §5; Maraun (2016) *Bias Correcting Climate Change Simulations*;
Hosking & Wallis (1997) *Regional Frequency Analysis* (L-moments);
Uuemaa et al. (2020) *Remote Sensing* 12(12):3769 (DEM comparison).
"""),

code(SETUP),
md("## 1 · Carga de datos"),
code(LOAD_DATA),

md("""\
## 2 · Descomposición STL
**STL** (Seasonal-Trend decomposition via LOESS) es robusto a outliers
y captura amplitud estacional variable — ideal para series con ENSO.
`Y_t = T_t (tendencia) + S_t (estacional) + R_t (residual)`
"""),
code(STL_CODE),

md("""\
## 3 · Tests estadísticos
| Test | H₀ | Rechazo si |
|------|-----|------------|
| ADF | Raíz unitaria (no estacionaria) | p < 0.05 |
| KPSS | Es estacionaria | p < 0.05 |
| Mann-Kendall | Sin tendencia monotónica | p < 0.05 |
| Pettitt | Sin punto de cambio de media | p < 0.05 |
"""),
code(TESTS_CODE),

md("""\
## 4 · Distribuciones de probabilidad
**Precipitación:** Gamma, Log-Normal, GEV (WMO-168 §5.3)
**Caudal máximo anual:** Log-Pearson III (estándar USBR/WMO), GEV
Ajuste por MLE; bondad de ajuste por KS y AIC.
"""),
code(DIST_CODE),

md("## 5 · Períodos de retorno — caudales de diseño"),
code(RETORNO_CODE),

md("""\
## 6 · Balance hídrico — Marco de Budyko
`ET/P = f(PET/P)` — límite energético: ET ≤ PET; límite hídrico: ET ≤ P
Desviaciones de la curva revelan rol de vegetación, suelo y almacenamiento.
"""),
code(BUDYKO_CODE),

md("""\
## 7 · Curva de doble masa — consistencia SENAMHI vs PISCO
Test estándar WMO-168 §3.5. Pendiente = 1 → consistentes.
Pendiente > 1 → PISCO subestima; < 1 → PISCO sobreestima.
"""),
code(DOBLE_MASA_CODE),

md("""\
## 8 · Corrección de sesgo — Quantile Mapping (QM)
Maraun (2016); Themeßl et al. (2012). Corrige distribución completa:
`P_corr(t) = F_obs⁻¹[ F_model( P_model(t) ) ]`
Calibrado en período solapado SENAMHI-PISCO, aplicado a toda la serie.
"""),
code(QM_CODE),

md("""\
## 9 · DEM — justificación de Copernicus GLO-30

| Estudio | Resultado clave |
|---------|----------------|
| **Uuemaa et al. (2020)** *Remote Sensing* 12(12):3769 | RMSE < 4 m (Copernicus/TanDEM-X) vs ~16 m SRTM en pendientes >20° |
| **Hawker et al. (2022)** *Environ. Res. Lett.* 17(2):024016 | FABDEM (base Copernicus) reduce error 41% en zonas con vegetación |
| **Copernicus DEM Product Handbook** (ESA/Airbus 2022) | Validado con ICESat-2; RMSE ≤ 4 m global, mejor en Andes |
| **Satgé et al. (2016)** *J. Hydrology* 530:418-429 | Valida necesidad de alta precisión en Andes peruano-bolivianos |

**Descarga gratuita:** https://portal.opentopography.org → COP30 → bbox cuenca
"""),

md("## 10 · Tabla resumen hidrológico"),
code(SUMMARY_CODE),

])

if __name__ == "__main__":
    print("Generando notebook 05 de Hidroestadistica...")
    save(NB05, NB_DIR / "05_hydro_statistics.ipynb")
    print("Listo.")
