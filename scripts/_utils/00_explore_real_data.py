#!/usr/bin/env python3
"""
Script 00: Exploración y análisis de todos los datos reales disponibles.
Genera inventario completo, figuras de diagnóstico y reporte de datos.

Datos analizados:
  - SENAMHI: 5 estaciones meteorológicas (txt, 1963-2014)
  - SNIRH/ANA: 6 estaciones hidrométrica + pluviométricas (xlsx, 2021-2026)
  - PISCO: PrecipDiar.nc, TemMaxDiar.nc, TemMinDiar.nc, ETP.nc (full grid)
  - PISCO GR2M: PISCO_GR2M_v2.0.nc (modelo mensual)
  - PISCO pre-extraídos: CLIMA_TEM_PRECIP/Prec_EO1, Tmm_EO1, etc.
"""
import sys
import logging
import warnings
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
FIG_DIR  = PROJECT_ROOT / "outputs" / "figures" / "data_inventory"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
PISCO_DIR = Path("d:/ANA Concurso/PISCO/Date")
PISCO_GR2M = Path("d:/ANA Concurso/PISCO_GR2M_v2.0.nc")

for d in [FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                               logging.FileHandler(PROJECT_ROOT / "outputs" / "explore_real_data.log",
                                                   encoding="utf-8")])
log = logging.getLogger("explore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ─── Estilos ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
COLORS = {"pr": "#1f77b4", "tmax": "#d62728", "tmin": "#2ca02c",
          "q": "#9467bd", "level": "#8c564b", "etp": "#e377c2"}

MONTHS_ES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 1: ESTACIONES SENAMHI (txt)
# ═══════════════════════════════════════════════════════════════════════════
SENAMHI_META = {
    "qc00000539.txt": {"code":"539",   "name":"?",           "lat":None,    "lon":None,    "alt":None,  "vars":["pr","tmax","tmin"]},
    "qc00000546.txt": {"code":"546",   "name":"?",           "lat":None,    "lon":None,    "alt":None,  "vars":["pr","tmax","tmin"]},
    "qc00155202.txt": {"code":"155202","name":"Santa Cruz",  "lat":-11.200, "lon":-76.633, "alt":3700,  "vars":["pr"]},
    "qc00155205.txt": {"code":"155205","name":"?",           "lat":None,    "lon":None,    "alt":None,  "vars":["pr"]},
    "qc00155214.txt": {"code":"155214","name":"Pirca",       "lat":-11.233, "lon":-76.650, "alt":3255,  "vars":["pr"]},
}

def read_senamhi(fpath):
    df = pd.read_csv(fpath, sep=r"\s+", header=None,
                     names=["year","month","day","pr_mm","tmax_c","tmin_c"],
                     na_values=[-99.9,"-99.9"])
    df["date"] = pd.to_datetime(df[["year","month","day"]], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Tmax/tmin solo si hay datos
    for col in ["tmax_c","tmin_c"]:
        if df[col].notna().sum() == 0:
            df[col] = np.nan
    df["tmean_c"] = (df["tmax_c"] + df["tmin_c"]) / 2
    return df

log.info("=== PARTE 1: Leyendo estaciones SENAMHI ===")
senamhi_dfs = {}
senamhi_stats = []
senamhi_dir = DATA_RAW / "senamhi"
for fname, meta in SENAMHI_META.items():
    fp = senamhi_dir / fname
    df = read_senamhi(fp)
    senamhi_dfs[meta["code"]] = (df, meta)
    n = len(df)
    pr_ok = df["pr_mm"].notna().sum()
    tmax_ok = df["tmax_c"].notna().sum()
    stat = {
        "code": meta["code"], "name": meta["name"],
        "lat": meta["lat"], "lon": meta["lon"], "alt": meta["alt"],
        "date_start": str(df["date"].min().date()),
        "date_end":   str(df["date"].max().date()),
        "n_days": n,
        "pr_pct_valid": round(100*pr_ok/n, 1),
        "tmax_pct_valid": round(100*tmax_ok/n, 1),
        "pr_max": round(df["pr_mm"].max(), 1),
        "tmax_range": f"[{df['tmax_c'].min():.1f}, {df['tmax_c'].max():.1f}]" if tmax_ok > 0 else "N/A",
    }
    senamhi_stats.append(stat)
    log.info(f"  {meta['code']}: {stat['date_start']} → {stat['date_end']} | "
             f"pr={stat['pr_pct_valid']}% | n={n}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 2: ESTACIONES SNIRH/ANA (xlsx)
# ═══════════════════════════════════════════════════════════════════════════
def read_snirh(fpath):
    xl = pd.ExcelFile(fpath)
    raw = pd.read_excel(fpath, sheet_name=0, header=None)
    meta = {
        "station":  str(raw.iloc[1, 1]).strip() if not pd.isna(raw.iloc[1, 1]) else "",
        "variable": str(raw.iloc[2, 1]).strip() if not pd.isna(raw.iloc[2, 1]) else "",
        "coords":   str(raw.iloc[4, 1]).strip() if not pd.isna(raw.iloc[4, 1]) else "",
        "operator": str(raw.iloc[3, 1]).strip() if not pd.isna(raw.iloc[3, 1]) else "",
        "source":   str(raw.iloc[9, 1]).strip() if not pd.isna(raw.iloc[9, 1]) else "",
    }
    # Extraer lat/lon de coords string
    coords_str = meta["coords"]
    lat = lon = alt = None
    try:
        import re
        lat_m = re.search(r"Latitud:\s*([-\d.]+)", coords_str)
        lon_m = re.search(r"Longitud:\s*([-\d.]+)", coords_str)
        alt_m = re.search(r"Altitud\(msnm\):\s*([\d.]+)", coords_str)
        if lat_m: lat = float(lat_m.group(1))
        if lon_m: lon = float(lon_m.group(1))
        if alt_m: alt = float(alt_m.group(1))
    except: pass
    meta["lat"] = lat; meta["lon"] = lon; meta["alt"] = alt

    # Datos desde fila 14 (0-indexed: 14)
    data = raw.iloc[14:].copy()
    data.columns = ["fecha", "hora", "valor"]
    data = data.dropna(subset=["valor"]).copy()
    data["fecha_dt"] = pd.to_datetime(data["fecha"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["fecha_dt"]).sort_values("fecha_dt").reset_index(drop=True)
    data["valor"] = pd.to_numeric(data["valor"], errors="coerce")
    return meta, data

log.info("=== PARTE 2: Leyendo estaciones SNIRH/ANA ===")
snirh_records = []
snirh_dir = DATA_RAW / "ana_snirh"
for fp in sorted(snirh_dir.glob("*.xlsx")):
    try:
        meta, data = read_snirh(fp)
        n = len(data)
        stat = {
            "file": fp.name,
            "station": meta["station"],
            "variable": meta["variable"],
            "lat": meta["lat"], "lon": meta["lon"], "alt": meta["alt"],
            "date_start": str(data["fecha_dt"].min().date()) if n > 0 else "N/A",
            "date_end":   str(data["fecha_dt"].max().date()) if n > 0 else "N/A",
            "n_records": n,
            "val_min": round(data["valor"].min(), 3) if n > 0 else None,
            "val_max": round(data["valor"].max(), 3) if n > 0 else None,
            "val_mean": round(data["valor"].mean(), 3) if n > 0 else None,
            "meta_full": meta,
            "data": data,
        }
        snirh_records.append(stat)
        log.info(f"  {meta['station'][:35]} | {meta['variable'][:35]} | "
                 f"n={n} | {stat['date_start']}→{stat['date_end']} | "
                 f"[{stat['val_min']},{stat['val_max']}]")
    except Exception as e:
        log.warning(f"  ERROR {fp.name}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 3: PISCO pre-extraídos (txt CLIMA_TEM_PRECIP)
# ═══════════════════════════════════════════════════════════════════════════
def read_pisco_txt(fpath, var_type="prec"):
    """Lee archivos Prec_EO*.txt o Tmm_EO*.txt de PISCO."""
    with open(fpath, encoding="latin1") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Primera línea es la fecha de inicio YYYYMMDD
    start_date = pd.to_datetime(lines[0], format="%Y%m%d")
    dates = pd.date_range(start=start_date, periods=len(lines)-1, freq="D")
    vals = []
    for line in lines[1:]:
        try:
            if "," in line:
                parts = [float(x) for x in line.split(",")]
                vals.append(parts)
            else:
                vals.append([float(line)])
        except:
            vals.append([np.nan])
    if var_type == "prec":
        df = pd.DataFrame({"date": dates, "pr_mm": [v[0] for v in vals]})
    else:  # temperatura: dos columnas → tmean, tdiff (o tmax,tmin)
        df = pd.DataFrame({
            "date": dates,
            "val1": [v[0] if len(v) > 0 else np.nan for v in vals],
            "val2": [v[1] if len(v) > 1 else np.nan for v in vals],
        })
    return df

log.info("=== PARTE 3: Leyendo PISCO pre-extraídos ===")
pisco_txt = {}
clima_dir = PISCO_DIR / "CLIMA_TEM_PRECIP"
for key, vtype in [("Prec_EO1","prec"),("Prec_EO2","prec"),
                   ("Tmm_EO1","temp"),("Tmm_EO2","temp")]:
    fp = clima_dir / f"{key}.txt"
    if fp.exists():
        df = read_pisco_txt(fp, vtype)
        pisco_txt[key] = df
        log.info(f"  {key}: {df['date'].min().date()} → {df['date'].max().date()} ({len(df)} días)")

# long_lat
ll_fp = PISCO_DIR / "long_lat.csv"
pisco_points = pd.read_csv(ll_fp) if ll_fp.exists() else pd.DataFrame()
log.info(f"  Puntos PISCO: {pisco_points.to_dict('records')}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 4: PISCO NetCDF (metadata solamente, luego extracción)
# ═══════════════════════════════════════════════════════════════════════════
log.info("=== PARTE 4: Inspeccionando PISCO NetCDF ===")
pisco_nc_info = {}
try:
    import netCDF4 as nc4
    for nc_name in ["PrecipDiar.nc", "TemMaxDiar.nc", "TemMinDiar.nc", "ETP.nc"]:
        fp = PISCO_DIR / nc_name
        if not fp.exists():
            continue
        with nc4.Dataset(fp) as ds:
            info = {
                "file": nc_name,
                "dimensions": {k: len(v) for k, v in ds.dimensions.items()},
                "variables": list(ds.variables.keys()),
                "time_units": None, "time_start": None, "time_end": None,
                "lat_range": None, "lon_range": None,
            }
            # Time
            if "time" in ds.variables:
                t = ds.variables["time"]
                info["time_units"] = getattr(t, "units", "?")
                try:
                    times = nc4.num2date(t[:], t.units, calendar=getattr(t,"calendar","standard"))
                    info["time_start"] = str(times[0])
                    info["time_end"] = str(times[-1])
                    info["n_times"] = len(times)
                except: pass
            # Lat/lon
            for latname in ["lat","latitude","y"]:
                if latname in ds.variables:
                    lats = ds.variables[latname][:]
                    info["lat_range"] = [float(lats.min()), float(lats.max())]
                    break
            for lonname in ["lon","longitude","x"]:
                if lonname in ds.variables:
                    lons = ds.variables[lonname][:]
                    info["lon_range"] = [float(lons.min()), float(lons.max())]
                    break
            # Main variable
            main_vars = [v for v in ds.variables if v not in
                         ["time","lat","lon","latitude","longitude","x","y","crs"]]
            info["main_variables"] = main_vars
            for mv in main_vars[:2]:
                var = ds.variables[mv]
                info[f"var_{mv}_units"] = getattr(var, "units", "?")
                info[f"var_{mv}_shape"] = list(var.shape)
            pisco_nc_info[nc_name] = info
            log.info(f"  {nc_name}: dims={info['dimensions']} | "
                     f"time={info['time_start']} → {info['time_end']} | "
                     f"lat={info['lat_range']} | lon={info['lon_range']}")
except Exception as e:
    log.warning(f"  Error leyendo NetCDF: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 5: PISCO GR2M (NetCDF mensual)
# ═══════════════════════════════════════════════════════════════════════════
log.info("=== PARTE 5: Inspeccionando PISCO GR2M ===")
gr2m_info = {}
try:
    import netCDF4 as nc4
    if PISCO_GR2M.exists():
        with nc4.Dataset(PISCO_GR2M) as ds:
            gr2m_info = {
                "dimensions": {k: len(v) for k, v in ds.dimensions.items()},
                "variables": list(ds.variables.keys()),
                "global_attrs": {k: str(getattr(ds, k, ""))[:100]
                                 for k in ds.ncattrs()},
            }
            if "time" in ds.variables:
                t = ds.variables["time"]
                try:
                    times = nc4.num2date(t[:], t.units,
                                         calendar=getattr(t,"calendar","standard"))
                    gr2m_info["time_start"] = str(times[0])
                    gr2m_info["time_end"] = str(times[-1])
                    gr2m_info["n_times"] = len(times)
                except: pass
            for mv in [v for v in ds.variables if v not in
                       ["time","lat","lon","latitude","longitude","x","y"]]:
                v = ds.variables[mv]
                gr2m_info[f"var_{mv}"] = {
                    "units": getattr(v,"units","?"),
                    "shape": list(v.shape),
                    "long_name": getattr(v,"long_name","?")[:60]
                }
            log.info(f"  GR2M dims={gr2m_info['dimensions']}")
            log.info(f"  GR2M time={gr2m_info.get('time_start')} → {gr2m_info.get('time_end')}")
            log.info(f"  GR2M vars={[k for k in gr2m_info if k.startswith('var_')]}")
except Exception as e:
    log.warning(f"  Error GR2M: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 6: EXTRAER PISCO en puntos de estaciones para la cuenca
# ═══════════════════════════════════════════════════════════════════════════
log.info("=== PARTE 6: Extrayendo PISCO en puntos de estaciones ===")

# Puntos de interés en la cuenca Chancay-Huaral
BASIN_POINTS = {
    "Santo_Domingo":     {"lat": -11.3836, "lon": -77.0503},
    "Puente_Callantama": {"lat": -11.2989, "lon": -76.8518},
    "Vichaycocha":       {"lat": -11.1392, "lon": -76.6246},
    "Santa_Cruz":        {"lat": -11.200,  "lon": -76.633},
    "Pirca":             {"lat": -11.233,  "lon": -76.650},
    "Basin_Centroid":    {"lat": -11.30,   "lon": -76.80},
}

pisco_extracted = {}
try:
    import netCDF4 as nc4
    fp_pr = PISCO_DIR / "PrecipDiar.nc"
    fp_tmax = PISCO_DIR / "TemMaxDiar.nc"
    fp_tmin = PISCO_DIR / "TemMinDiar.nc"
    fp_etp  = PISCO_DIR / "ETP.nc"

    with nc4.Dataset(fp_pr) as ds_pr, \
         nc4.Dataset(fp_tmax) as ds_tmax, \
         nc4.Dataset(fp_tmin) as ds_tmin:

        # Coordenadas
        lat_nc = ds_pr.variables.get("lat", ds_pr.variables.get("latitude", ds_pr.variables.get("y")))[:]
        lon_nc = ds_pr.variables.get("lon", ds_pr.variables.get("longitude", ds_pr.variables.get("x")))[:]
        t_var  = ds_pr.variables["time"]
        times  = nc4.num2date(t_var[:], t_var.units,
                              calendar=getattr(t_var,"calendar","standard"))
        dates_pr = pd.to_datetime([str(d) for d in times])

        # Variable principal
        pr_var_name  = [v for v in ds_pr.variables if v not in
                        ["time","lat","lon","latitude","longitude","x","y","crs"]][0]
        tx_var_name  = [v for v in ds_tmax.variables if v not in
                        ["time","lat","lon","latitude","longitude","x","y","crs"]][0]
        tn_var_name  = [v for v in ds_tmin.variables if v not in
                        ["time","lat","lon","latitude","longitude","x","y","crs"]][0]

        log.info(f"  Grid PISCO: lat [{lat_nc.min():.2f},{lat_nc.max():.2f}] "
                 f"lon [{lon_nc.min():.2f},{lon_nc.max():.2f}] "
                 f"tiempo [{dates_pr[0].date()},{dates_pr[-1].date()}] n={len(dates_pr)}")

        for pt_name, coords in BASIN_POINTS.items():
            target_lat = coords["lat"]
            target_lon = coords["lon"]
            # Índice más cercano
            ilat = np.argmin(np.abs(lat_nc - target_lat))
            ilon = np.argmin(np.abs(lon_nc - target_lon))
            dist = np.sqrt((lat_nc[ilat]-target_lat)**2 + (lon_nc[ilon]-target_lon)**2)
            log.info(f"  {pt_name}: target({target_lat},{target_lon}) → "
                     f"grid({float(lat_nc[ilat]):.4f},{float(lon_nc[ilon]):.4f}) dist={dist:.4f}°")

            # Extraer series temporales
            pr_ts   = ds_pr.variables[pr_var_name][:, ilat, ilon]
            tmax_ts = ds_tmax.variables[tx_var_name][:, ilat, ilon]
            tmin_ts = ds_tmin.variables[tn_var_name][:, ilat, ilon]

            # Manejar masked arrays y fill values
            pr_arr   = np.ma.filled(pr_ts,   np.nan).astype(float)
            tmax_arr = np.ma.filled(tmax_ts, np.nan).astype(float)
            tmin_arr = np.ma.filled(tmin_ts, np.nan).astype(float)

            # Reemplazar fill values grandes
            pr_arr[pr_arr > 1000] = np.nan
            tmax_arr[np.abs(tmax_arr) > 100] = np.nan
            tmin_arr[np.abs(tmin_arr) > 100] = np.nan

            df_pt = pd.DataFrame({
                "date":    dates_pr,
                "pr_mm":   pr_arr,
                "tmax_c":  tmax_arr,
                "tmin_c":  tmin_arr,
            })
            df_pt["tmean_c"] = (df_pt["tmax_c"] + df_pt["tmin_c"]) / 2
            pisco_extracted[pt_name] = df_pt
            log.info(f"    pr: [{pr_arr[~np.isnan(pr_arr)].min():.2f}, "
                     f"{pr_arr[~np.isnan(pr_arr)].max():.2f}] mm | "
                     f"tmax: [{tmax_arr[~np.isnan(tmax_arr)].min():.1f}, "
                     f"{tmax_arr[~np.isnan(tmax_arr)].max():.1f}]°C")

    # Guardar extracción de Basin_Centroid como PISCO oficial del proyecto
    if "Basin_Centroid" in pisco_extracted:
        df_basin = pisco_extracted["Basin_Centroid"].copy()
        df_basin["entity_id"] = "basin_centroid"
        out_pisco = DATA_RAW / "pisco" / "pisco_basin_centroid_daily.csv"
        out_pisco.parent.mkdir(parents=True, exist_ok=True)
        df_basin.to_csv(out_pisco, index=False)
        log.info(f"  PISCO cuenca guardado: {out_pisco}")

except Exception as e:
    log.warning(f"  Error extrayendo PISCO: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 7: FIGURAS
# ═══════════════════════════════════════════════════════════════════════════
log.info("=== PARTE 7: Generando figuras ===")

def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figura: {path.name}")

# ── FIG 1: Mapa de estaciones ──────────────────────────────────────────────
log.info("  Fig 1: Mapa de estaciones")
fig, ax = plt.subplots(figsize=(9, 8))

# Estaciones SENAMHI con coordenadas conocidas
senamhi_known = [s for s in senamhi_stats if s["lat"] is not None]
if senamhi_known:
    lons_s = [s["lon"] for s in senamhi_known]
    lats_s = [s["lat"] for s in senamhi_known]
    ax.scatter(lons_s, lats_s, c="#1f77b4", s=120, marker="^",
               zorder=5, label="SENAMHI (meteorológica)", edgecolors="k", lw=0.5)
    for s in senamhi_known:
        ax.annotate(f"{s['name']}\n({s['alt']}m)", (s["lon"], s["lat"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, color="#1f77b4")

# Estaciones SNIRH únicas (por nombre+lat/lon)
snirh_plotted = set()
for r in snirh_records:
    if r["lat"] and r["lon"]:
        key = (round(r["lat"],3), round(r["lon"],3))
        if key not in snirh_plotted:
            snirh_plotted.add(key)
            is_hydro = "Caudal" in r["variable"] or "Nivel" in r["variable"]
            marker = "o" if is_hydro else "s"
            color  = COLORS["q"] if is_hydro else "#ff7f0e"
            label_type = "SNIRH hidrom." if is_hydro else "SNIRH pluviom."
            sname = r["station"].split("(")[0].strip()
            ax.scatter(r["lon"], r["lat"], c=color, s=150, marker=marker,
                       zorder=5, edgecolors="k", lw=0.5)
            ax.annotate(f"{sname}\n({r['alt']}m)" if r["alt"] else sname,
                        (r["lon"], r["lat"]),
                        xytext=(5, -8), textcoords="offset points",
                        fontsize=8, color=color)

# Puntos PISCO extraídos
if pisco_extracted:
    px = [BASIN_POINTS[k]["lon"] for k in pisco_extracted]
    py = [BASIN_POINTS[k]["lat"] for k in pisco_extracted]
    ax.scatter(px, py, c="none", s=200, marker="D",
               edgecolors="orange", lw=1.5, zorder=4, label="Puntos PISCO extraídos")

# Decorar
legend_elements = [
    Patch(facecolor="#1f77b4", label="SENAMHI (meteo)", edgecolor="k"),
    Patch(facecolor=COLORS["q"],  label="SNIRH (hidrom.)", edgecolor="k"),
    Patch(facecolor="#ff7f0e",  label="SNIRH (pluviom.)", edgecolor="k"),
    Patch(facecolor="none", edgecolor="orange", label="PISCO extraído"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
ax.set_xlabel("Longitud (°W)")
ax.set_ylabel("Latitud (°S)")
ax.set_title("Red de estaciones — Cuenca Chancay-Huaral\nSENAMHI (1963-2014) + SNIRH/ANA (2021-2026) + PISCO",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
# Referencia de la cuenca aproximada
ax.set_xlim(-77.3, -76.4)
ax.set_ylim(-11.8, -10.9)
savefig(fig, "01_mapa_estaciones.png")

# ── FIG 2: Series temporales SENAMHI ──────────────────────────────────────
log.info("  Fig 2: Series temporales SENAMHI")
n_stat = len(senamhi_dfs)
fig, axes = plt.subplots(n_stat, 1, figsize=(16, 3*n_stat), sharex=False)
if n_stat == 1: axes = [axes]

for ax, (code, (df, meta)) in zip(axes, senamhi_dfs.items()):
    ax2 = ax.twinx()
    ax.bar(df["date"], df["pr_mm"], width=1, color=COLORS["pr"], alpha=0.6, label="pr (mm)")
    if df["tmax_c"].notna().any():
        ax2.plot(df["date"], df["tmax_c"], color=COLORS["tmax"], lw=0.6, alpha=0.7, label="Tmax")
        ax2.plot(df["date"], df["tmin_c"], color=COLORS["tmin"], lw=0.6, alpha=0.7, label="Tmin")
    name = meta["name"] if meta["name"] != "?" else f"cod. {code}"
    alt_str = f" | {meta['alt']} msnm" if meta["alt"] else ""
    ax.set_title(f"SENAMHI: {name}{alt_str} | {df['date'].min().year}–{df['date'].max().year}",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("pr (mm)", fontsize=8)
    ax2.set_ylabel("T (°C)", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

fig.suptitle("Series temporales — Estaciones SENAMHI (Cuenca Chancay-Huaral)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97])
savefig(fig, "02_senamhi_series_temporales.png")

# ── FIG 3: Completitud SENAMHI ─────────────────────────────────────────────
log.info("  Fig 3: Heatmap completitud SENAMHI")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
var_labels = {"pr_mm": "Precipitación", "tmax_c": "Tmax", "tmin_c": "Tmin"}
for ax, (var, vlabel) in zip(axes, var_labels.items()):
    # Pivot: filas=año, columnas=mes, valor=% disponible
    rows = []
    for code, (df, meta) in senamhi_dfs.items():
        if df[var].notna().sum() == 0:
            continue
        df_yr = df.copy()
        df_yr["year"] = df_yr["date"].dt.year
        df_yr["month"] = df_yr["date"].dt.month
        pct = df_yr.groupby(["year","month"])[var].apply(lambda x: 100*x.notna().sum()/max(len(x),1))
        name = meta["name"] if meta["name"] != "?" else f"cod.{code}"
        piv = pct.unstack(level="month")
        piv.columns = [MONTHS_ES[m-1] for m in piv.columns]
        piv.index.name = "Año"
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                       extent=[0, 12, piv.index.max(), piv.index.min()])
        ax.set_title(f"{vlabel}\n({name})", fontsize=9)
        ax.set_xticks(np.arange(0.5, 12.5))
        ax.set_xticklabels(MONTHS_ES, fontsize=7, rotation=45)
        break  # solo la primera estación con datos de esa variable

fig.suptitle("Completitud de datos SENAMHI (%)", fontsize=11, fontweight="bold")
fig.colorbar(im, ax=axes[-1], label="% disponible")
fig.tight_layout()
savefig(fig, "03_senamhi_completitud.png")

# ── FIG 4: Series SNIRH / Caudal ──────────────────────────────────────────
log.info("  Fig 4: Series caudal SNIRH")
# Seleccionar solo diarios de caudal y nivel
daily_q = [r for r in snirh_records
           if "Diario" in r["variable"] and
           ("Caudal" in r["variable"] or "Nivel" in r["variable"])]
daily_q = sorted(daily_q, key=lambda r: r["station"])

if daily_q:
    n_q = len(daily_q)
    fig, axes = plt.subplots(n_q, 1, figsize=(16, 3*n_q), sharex=False)
    if n_q == 1: axes = [axes]
    for ax, r in zip(axes, daily_q):
        data = r["data"]
        is_q = "Caudal" in r["variable"]
        color = COLORS["q"] if is_q else COLORS["level"]
        unit  = "m³/s" if is_q else "m"
        ax.plot(data["fecha_dt"], data["valor"], color=color, lw=0.8, alpha=0.9)
        ax.fill_between(data["fecha_dt"], data["valor"], alpha=0.2, color=color)
        sname = r["station"].split("(")[0].strip()
        alt_str = f" | {r['alt']}m" if r["alt"] else ""
        ax.set_title(f"{sname}{alt_str} | {r['variable']} | "
                     f"{r['date_start']} → {r['date_end']} | "
                     f"n={r['n_records']} | max={r['val_max']:.2f} {unit}",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
    fig.suptitle("Series temporales — Estaciones SNIRH/ANA (Cuenca Chancay-Huaral)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.97])
    savefig(fig, "04_snirh_caudales_series.png")

# ── FIG 5: Climatología mensual caudal ────────────────────────────────────
log.info("  Fig 5: Climatología mensual caudal")
fig, axes = plt.subplots(1, len(daily_q), figsize=(5*len(daily_q), 5))
if len(daily_q) == 1: axes = [axes]
for ax, r in zip(axes, daily_q):
    data = r["data"].copy()
    data["month"] = pd.to_datetime(data["fecha_dt"]).dt.month
    monthly = data.groupby("month")["valor"].agg(["mean","median",
        lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)])
    monthly.columns = ["mean","median","p10","p90"]
    unit = "m³/s" if "Caudal" in r["variable"] else "m"
    ax.fill_between(monthly.index, monthly["p10"], monthly["p90"],
                    alpha=0.25, color=COLORS["q"], label="P10-P90")
    ax.plot(monthly.index, monthly["mean"], "o-", color=COLORS["q"],
            lw=2, ms=6, label="Media")
    ax.plot(monthly.index, monthly["median"], "s--", color="#9467bd",
            lw=1.5, ms=4, alpha=0.7, label="Mediana")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS_ES, rotation=45, fontsize=8)
    ax.set_title(r["station"].split("(")[0].strip(), fontsize=9, fontweight="bold")
    ax.set_ylabel(f"Caudal ({unit})", fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle("Régimen hidrológico mensual — SNIRH/ANA", fontsize=11, fontweight="bold")
fig.tight_layout()
savefig(fig, "05_snirh_climatologia_mensual.png")

# ── FIG 6: PISCO extraído — serie temporal cuenca ─────────────────────────
if pisco_extracted:
    log.info("  Fig 6: PISCO series temporales por punto")
    key_pts = [k for k in ["Basin_Centroid","Santo_Domingo"] if k in pisco_extracted]
    fig, axes = plt.subplots(3, len(key_pts), figsize=(14, 10))
    if len(key_pts) == 1:
        axes = axes.reshape(-1, 1)

    for j, pt in enumerate(key_pts):
        df_pt = pisco_extracted[pt]
        ax0, ax1, ax2 = axes[0, j], axes[1, j], axes[2, j]

        ax0.bar(df_pt["date"], df_pt["pr_mm"], width=1,
                color=COLORS["pr"], alpha=0.7)
        ax0.set_title(f"PISCO — {pt.replace('_',' ')}", fontsize=10, fontweight="bold")
        ax0.set_ylabel("pr (mm/día)")
        ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax1.plot(df_pt["date"], df_pt["tmax_c"], color=COLORS["tmax"],
                 lw=0.6, alpha=0.8, label="Tmax")
        ax1.plot(df_pt["date"], df_pt["tmin_c"], color=COLORS["tmin"],
                 lw=0.6, alpha=0.8, label="Tmin")
        ax1.set_ylabel("T (°C)")
        ax1.legend(fontsize=7)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        df_pt["pr_sum30"] = df_pt["pr_mm"].rolling(30).sum()
        ax2.plot(df_pt["date"], df_pt["pr_sum30"], color=COLORS["pr"], lw=1.2)
        ax2.set_ylabel("pr acum. 30d (mm)")
        ax2.set_xlabel("Fecha")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Series temporales PISCO — Puntos cuenca Chancay-Huaral", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "06_pisco_series_cuenca.png")

# ── FIG 7: Climatología mensual PISCO ────────────────────────────────────
if "Basin_Centroid" in pisco_extracted:
    log.info("  Fig 7: Climatología mensual PISCO")
    df_bc = pisco_extracted["Basin_Centroid"].copy()
    df_bc["month"] = df_bc["date"].dt.month
    df_bc["year"] = df_bc["date"].dt.year

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Precipitación mensual
    pr_mo = df_bc.groupby("month")["pr_mm"].agg(["mean","median",
        lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)])
    pr_mo.columns = ["mean","median","p10","p90"]
    axes[0].bar(pr_mo.index, pr_mo["mean"]*30, color=COLORS["pr"], alpha=0.8)
    axes[0].errorbar(pr_mo.index, pr_mo["median"]*30,
                     yerr=[(pr_mo["median"]-pr_mo["p10"])*30,
                            (pr_mo["p90"]-pr_mo["median"])*30],
                     fmt="ko", capsize=4, ms=4, label="Mediana ± P10-P90")
    axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(MONTHS_ES, rotation=45)
    axes[0].set_title("Precipitación mensual PISCO\n(Centroide cuenca)", fontsize=10)
    axes[0].set_ylabel("mm/mes"); axes[0].legend(fontsize=7)

    # Temperatura mensual
    for col, color, label in [("tmax_c", COLORS["tmax"], "Tmax"),
                               ("tmin_c", COLORS["tmin"], "Tmin"),
                               ("tmean_c","#17becf","Tmean")]:
        if df_bc[col].notna().any():
            t_mo = df_bc.groupby("month")[col].mean()
            axes[1].plot(t_mo.index, t_mo.values, "o-", color=color,
                         lw=2, ms=5, label=label)
    axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(MONTHS_ES, rotation=45)
    axes[1].set_title("Temperatura mensual PISCO\n(Centroide cuenca)", fontsize=10)
    axes[1].set_ylabel("°C"); axes[1].legend(fontsize=7)

    # Distribución precipitación
    pr_vals = df_bc["pr_mm"].dropna()
    pr_wet = pr_vals[pr_vals > 0]
    axes[2].hist(pr_wet, bins=60, color=COLORS["pr"], alpha=0.8, edgecolor="white")
    axes[2].axvline(pr_wet.quantile(0.90), color="red", ls="--", lw=1.5,
                    label=f"P90={pr_wet.quantile(0.90):.1f} mm")
    axes[2].axvline(pr_wet.quantile(0.75), color="orange", ls="--", lw=1.5,
                    label=f"P75={pr_wet.quantile(0.75):.1f} mm")
    axes[2].set_title(f"Distribución precipitación diaria PISCO\n(solo días lluviosos >0mm, n={len(pr_wet):,})", fontsize=10)
    axes[2].set_xlabel("mm/día"); axes[2].set_ylabel("Frecuencia")
    axes[2].legend(fontsize=7)

    fig.suptitle("Climatología PISCO — Cuenca Chancay-Huaral", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "07_pisco_climatologia_mensual.png")

# ── FIG 8: PISCO pre-extraídos (EO1/EO2) ────────────────────────────────
if pisco_txt:
    log.info("  Fig 8: PISCO pre-extraídos EO1/EO2")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    pairs = [("Prec_EO1","Prec_EO2"), ("Tmm_EO1","Tmm_EO2")]
    for row, (k1, k2) in enumerate(pairs):
        for col, key in enumerate([k1, k2]):
            ax = axes[row, col]
            if key not in pisco_txt:
                ax.set_visible(False); continue
            df_t = pisco_txt[key]
            if "pr_mm" in df_t.columns:
                ax.bar(df_t["date"], df_t["pr_mm"], width=1,
                       color=COLORS["pr"], alpha=0.7)
                ax.set_ylabel("pr (mm/día)")
            else:
                ax.plot(df_t["date"], df_t["val1"], color=COLORS["tmax"], lw=0.5,
                        label="val1 (Tmax?)")
                ax.plot(df_t["date"], df_t["val2"], color=COLORS["tmin"], lw=0.5,
                        label="val2 (Tmin?)")
                ax.legend(fontsize=7); ax.set_ylabel("°C o mm")
            ax.set_title(f"PISCO pre-extraído: {key}", fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.suptitle("PISCO pre-extraídos (CLIMA_TEM_PRECIP)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "08_pisco_preextraidos_eo.png")

# ── FIG 9: Comparación PISCO vs SENAMHI ──────────────────────────────────
log.info("  Fig 9: Comparación PISCO vs SENAMHI (Santa Cruz)")
if "Basin_Centroid" in pisco_extracted and "155202" in senamhi_dfs:
    df_pisco = pisco_extracted["Santa_Cruz"].copy() if "Santa_Cruz" in pisco_extracted \
               else pisco_extracted["Basin_Centroid"].copy()
    df_sena, meta_sena = senamhi_dfs["155202"]

    # Período en común
    t_start = max(df_pisco["date"].min(), df_sena["date"].min())
    t_end   = min(df_pisco["date"].max(), df_sena["date"].max())
    df_p = df_pisco[(df_pisco["date"] >= t_start) & (df_pisco["date"] <= t_end)]
    df_s = df_sena[(df_sena["date"] >= t_start) & (df_sena["date"] <= t_end)]

    # Mensual
    df_p_mo = df_p.set_index("date")["pr_mm"].resample("ME").sum()
    df_s_mo = df_s.set_index("date")["pr_mm"].resample("ME").sum()
    common = df_p_mo.index.intersection(df_s_mo.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(common, df_p_mo.loc[common], label="PISCO (grilla)", color=COLORS["pr"], lw=1, alpha=0.8)
    axes[0].plot(common, df_s_mo.loc[common], label="SENAMHI Santa Cruz", color="k", lw=1, alpha=0.7, ls="--")
    axes[0].set_title("Precipitación mensual: PISCO vs SENAMHI Santa Cruz", fontsize=10)
    axes[0].set_ylabel("mm/mes"); axes[0].legend(); axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if len(common) > 10:
        axes[1].scatter(df_s_mo.loc[common], df_p_mo.loc[common], alpha=0.5, s=20, color=COLORS["pr"])
        lim = max(df_s_mo.loc[common].max(), df_p_mo.loc[common].max())*1.05
        axes[1].plot([0,lim],[0,lim],"k--",lw=1,label="1:1")
        corr = df_s_mo.loc[common].corr(df_p_mo.loc[common])
        bias = (df_p_mo.loc[common].mean() / df_s_mo.loc[common].mean() - 1)*100
        axes[1].set_title(f"PISCO vs SENAMHI (mensual)\nr={corr:.2f} | Bias PISCO={bias:+.1f}%", fontsize=10)
        axes[1].set_xlabel("SENAMHI (mm/mes)"); axes[1].set_ylabel("PISCO (mm/mes)")
        axes[1].legend()

    fig.tight_layout()
    savefig(fig, "09_pisco_vs_senamhi.png")

# ── FIG 10: Coherencia SENAMHI + SNIRH (Santa Cruz + Santo Domingo) ───────
log.info("  Fig 10: SENAMHI precipitación + SNIRH caudal")
q_daily = next((r for r in daily_q if "Santo Domingo" in r["station"]
                and "Caudal" in r["variable"]), None)
if q_daily and "155202" in senamhi_dfs:
    df_sena, _ = senamhi_dfs["155202"]
    df_q = q_daily["data"].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    ax1 = axes[0]
    ax1.bar(df_sena["date"], df_sena["pr_mm"], width=1,
            color=COLORS["pr"], alpha=0.7, label="Precip. Santa Cruz (SENAMHI)")
    ax1.set_ylabel("Precipitación (mm/día)"); ax1.legend(fontsize=8)
    ax1.set_title("Precipitación SENAMHI (Santa Cruz, 3700m)", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = axes[1]
    ax2.plot(df_q["fecha_dt"], df_q["valor"], color=COLORS["q"], lw=1, alpha=0.9)
    ax2.fill_between(df_q["fecha_dt"], df_q["valor"], alpha=0.2, color=COLORS["q"])
    ax2.set_ylabel("Caudal (m³/s)"); ax2.legend(fontsize=8)
    sname = q_daily["station"].split("(")[0].strip()
    ax2.set_title(f"Caudal SNIRH ({sname})", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.suptitle("Coherencia lluvia-caudal — Cuenca Chancay-Huaral", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "10_lluvia_caudal_coherencia.png")

# ── FIG 11: Análisis caudal horario (Santo Domingo) ─────────────────────
log.info("  Fig 11: Caudal horario Santo Domingo")
hourly_q = [r for r in snirh_records
            if "Santo Domingo" in r["station"] and "Hora" in r["variable"]
            and "Caudal" in r["variable"]]
if hourly_q:
    r_h = max(hourly_q, key=lambda r: r["n_records"])
    data_h = r_h["data"].copy()
    # Último año
    t_end = data_h["fecha_dt"].max()
    t_start = t_end - pd.Timedelta(days=365)
    data_recent = data_h[data_h["fecha_dt"] >= t_start]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].plot(data_recent["fecha_dt"], data_recent["valor"],
                 color=COLORS["q"], lw=0.8, alpha=0.9)
    axes[0].set_title(f"Caudal horario — Santo Domingo (último año, hasta {t_end.date()})", fontsize=10)
    axes[0].set_ylabel("m³/s")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Serie completa diaria
    data_daily_agg = data_h.set_index("fecha_dt")["valor"].resample("D").mean()
    axes[1].plot(data_daily_agg.index, data_daily_agg.values, color=COLORS["q"], lw=0.8)
    axes[1].fill_between(data_daily_agg.index, data_daily_agg.values, alpha=0.2, color=COLORS["q"])
    axes[1].set_title("Caudal diario medio — Santo Domingo (serie completa)", fontsize=10)
    axes[1].set_ylabel("m³/s")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.suptitle("Estación hidrométrica Santo Domingo — SNIRH/ANA", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "11_caudal_horario_santo_domingo.png")

# ── FIG 12: Comparación estaciones caudal ────────────────────────────────
log.info("  Fig 12: Comparación caudal entre estaciones")
q_records_daily = [r for r in daily_q if "Caudal" in r["variable"]]
if len(q_records_daily) >= 2:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    for r in q_records_daily:
        data = r["data"].copy()
        data_daily = data.set_index("fecha_dt")["valor"].resample("D").mean()
        sname = r["station"].split("(")[0].strip()
        axes[0].plot(data_daily.index, data_daily.values, lw=1.2, alpha=0.8,
                     label=f"{sname} ({r['alt']}m)" if r["alt"] else sname)

    axes[0].set_title("Caudales diarios — Estaciones SNIRH/ANA", fontsize=10)
    axes[0].set_ylabel("m³/s"); axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Curvas de duración de caudal
    for r in q_records_daily:
        data = r["data"].copy()
        vals = data["valor"].dropna().sort_values(ascending=False)
        exceedance = np.arange(1, len(vals)+1) / len(vals) * 100
        sname = r["station"].split("(")[0].strip()
        axes[1].semilogy(exceedance, vals.values, lw=2, alpha=0.8, label=sname)

    axes[1].set_title("Curva de duración de caudales (FDC)", fontsize=10)
    axes[1].set_xlabel("Probabilidad de excedencia (%)"); axes[1].set_ylabel("Caudal m³/s (escala log)")
    axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Análisis caudales — Cuenca Chancay-Huaral", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "12_comparacion_caudales.png")

# ── FIG 13: Análisis precip estaciones SNIRH (Santa Cruz + Pirca) ─────────
log.info("  Fig 13: Estaciones pluviométricas SNIRH")
pr_snirh = [r for r in snirh_records if "Precipitaci" in r["variable"] and "Diario" not in r["variable"]
            or ("Precipitaci" in r.get("variable","") and "1 D" in r.get("variable",""))]
pr_snirh = [r for r in snirh_records if "Precipitaci" in r.get("variable","")]

if pr_snirh:
    fig, axes = plt.subplots(len(pr_snirh), 1, figsize=(16, 4*len(pr_snirh)))
    if len(pr_snirh) == 1: axes = [axes]
    for ax, r in zip(axes, pr_snirh):
        data = r["data"].copy()
        ax.bar(data["fecha_dt"], data["valor"], width=1, color=COLORS["pr"], alpha=0.7)
        sname = r["station"].split("(")[0].strip()
        ax.set_title(f"SNIRH Precipitación: {sname} | {r['date_start']}→{r['date_end']} | "
                     f"max={r['val_max']:.1f} mm", fontsize=9)
        ax.set_ylabel("mm/día")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.suptitle("Precipitación SNIRH — Cuenca Chancay-Huaral", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "13_snirh_precipitacion.png")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 8: GUARDAR INVENTARIO JSON Y REPORTE MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════
log.info("=== PARTE 8: Guardando inventario y reporte ===")

inventory = {
    "generated": datetime.now().isoformat(),
    "senamhi_stations": senamhi_stats,
    "snirh_stations": [
        {k: v for k, v in r.items() if k != "data"} for r in snirh_records
    ],
    "pisco_netcdf": pisco_nc_info,
    "pisco_gr2m": {k: v for k, v in gr2m_info.items() if not isinstance(v, dict) or k.startswith("var_")},
    "pisco_extracted_points": list(pisco_extracted.keys()),
    "pisco_txt_series": {k: {"n_rows": len(df), "start": str(df["date"].min().date()), "end": str(df["date"].max().date())}
                         for k, df in pisco_txt.items()},
}

inv_path = PROJECT_ROOT / "data" / "metadata" / "data_inventory_real.json"
with open(inv_path, "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False, default=str)
log.info(f"  Inventario JSON: {inv_path}")

# Reporte markdown
def pct_str(n, total): return f"{100*n/total:.0f}%" if total > 0 else "N/A"

report = [
    "# Inventario de Datos Reales — HidroAlerta Chancay-Huaral",
    f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "## 1. Estaciones SENAMHI (datos históricos 1963–2014)",
    "",
    "| Código | Nombre | Lat | Lon | Alt (m) | Período | n días | PR valid | Variables |",
    "|--------|--------|-----|-----|---------|---------|--------|----------|-----------|",
]
for s in senamhi_stats:
    report.append(f"| {s['code']} | {s['name']} | {s['lat']} | {s['lon']} | {s['alt']} | "
                  f"{s['date_start']}→{s['date_end']} | {s['n_days']:,} | "
                  f"{s['pr_pct_valid']}% | {', '.join(['PR' if s['pr_pct_valid']>0 else '','TMAX' if s['tmax_pct_valid']>0 else ''])} |")

report += [
    "",
    "### Notas SENAMHI",
    "- Códigos 155202 (Santa Cruz, 3700m) y 155214 (Pirca, 3255m) tienen solo precipitación",
    "- Códigos 539 y 546 tienen pr + tmax + tmin (localización pendiente)",
    "- Código 155205 tiene solo pr con máximo histórico de 94.1 mm/día (zona alta)",
    "- Todos los datos terminan en 2014. Gap 2014–2021 a cubrir con PISCO o gestión adicional",
    "",
    "## 2. Estaciones SNIRH/ANA (datos recientes 2021–2026)",
    "",
    "| Estación | Variable | Lat | Lon | Alt (m) | Período | n registros | Min | Max |",
    "|----------|----------|-----|-----|---------|---------|-------------|-----|-----|",
]
for r in snirh_records:
    sname = r["station"].split("(")[0].strip()[:30]
    var   = r["variable"][:40]
    report.append(f"| {sname} | {var} | {r['lat']} | {r['lon']} | {r['alt']} | "
                  f"{r['date_start']}→{r['date_end']} | {r['n_records']:,} | "
                  f"{r['val_min']} | {r['val_max']} |")

report += [
    "",
    "### Estaciones hidrométrias clave",
    "- **Santo Domingo** (conv. PHISIS0137): lat=-11.3836, lon=-77.0503 — caudal y nivel diario/horario",
    "- **Santo Domingo** (auto 47E214D2): lat=-11.3701, lon=-77.0282, alt=614m — caudal horario/diario",
    "- **Puente Callantama** (auto 47E22148): lat=-11.2989, lon=-76.8518, alt=1378m — max 94 m³/s",
    "- **Vichaycocha** (auto 47E257D8): lat=-11.1392, lon=-76.6246, alt=3503m — zona alta",
    "",
    "## 3. PISCO NetCDF (grilla completa)",
    "",
]
for nc_name, info in pisco_nc_info.items():
    report.append(f"### {nc_name}")
    report.append(f"- Dimensiones: {info['dimensions']}")
    report.append(f"- Período: {info.get('time_start','?')} → {info.get('time_end','?')} ({info.get('n_times','?')} pasos)")
    report.append(f"- Dominio lat: {info.get('lat_range')} | lon: {info.get('lon_range')}")
    report.append(f"- Variables: {info.get('main_variables')}")
    report.append("")

report += [
    "## 4. PISCO GR2M (modelo mensual)",
    f"- Dimensiones: {gr2m_info.get('dimensions',{})}",
    f"- Período: {gr2m_info.get('time_start','?')} → {gr2m_info.get('time_end','?')}",
    f"- Variables: {[k.replace('var_','') for k in gr2m_info if k.startswith('var_')]}",
    "",
    "## 5. Figuras generadas",
    "",
    "| # | Archivo | Descripción |",
    "|---|---------|-------------|",
    "| 01 | 01_mapa_estaciones.png | Mapa de todas las estaciones |",
    "| 02 | 02_senamhi_series_temporales.png | Series temporales SENAMHI |",
    "| 03 | 03_senamhi_completitud.png | Heatmap completitud datos |",
    "| 04 | 04_snirh_caudales_series.png | Series caudal/nivel SNIRH |",
    "| 05 | 05_snirh_climatologia_mensual.png | Régimen hidrológico mensual |",
    "| 06 | 06_pisco_series_cuenca.png | PISCO series cuenca |",
    "| 07 | 07_pisco_climatologia_mensual.png | Climatología mensual PISCO |",
    "| 08 | 08_pisco_preextraidos_eo.png | PISCO pre-extraídos EO1/EO2 |",
    "| 09 | 09_pisco_vs_senamhi.png | Comparación PISCO vs SENAMHI |",
    "| 10 | 10_lluvia_caudal_coherencia.png | Lluvia-caudal coherencia |",
    "| 11 | 11_caudal_horario_santo_domingo.png | Caudal horario Santo Domingo |",
    "| 12 | 12_comparacion_caudales.png | FDC y comparación estaciones |",
    "| 13 | 13_snirh_precipitacion.png | Precipitación SNIRH |",
    "",
    "## 6. Conclusiones y próximos pasos",
    "",
    "### Datos disponibles para modelamiento",
    "- **Precipitación histórica** (1963-2014): SENAMHI 5 estaciones — suficiente para climatología",
    "- **Caudal observado** (2021-2026): SNIRH 3 estaciones — listo para GR4J y TFT",
    "- **PISCO grillado** (1981-presente): full Peru 5km diario — cobertura espacial completa",
    "- **Gap crítico 2014-2021**: no hay caudal histórico, solo PISCO puede cubrir precipitación",
    "",
    "### Prioridades inmediatas",
    "1. Parsear y limpiar caudal diario Santo Domingo → `data/silver/caudal_santo_domingo_daily.csv`",
    "2. Extraer PISCO en puntos de cuenca → `data/raw/pisco/pisco_basin_centroid_daily.csv` ✅",
    "3. Calcular área de cuenca para conversión Q mm/día",
    "4. Calibrar GR4J con precipitación PISCO + caudal Santo Domingo",
    "5. Construir dataset TFT con encoder 60-90 días",
]

report_path = REPORT_DIR / "data_inventory_report.md"
report_path.write_text("\n".join(report), encoding="utf-8")
log.info(f"  Reporte: {report_path}")

log.info("=== ANÁLISIS COMPLETADO ===")
log.info(f"  Figuras: {FIG_DIR}")
log.info(f"  Reporte: {report_path}")
log.info(f"  PISCO cuenca: {DATA_RAW}/pisco/pisco_basin_centroid_daily.csv")
