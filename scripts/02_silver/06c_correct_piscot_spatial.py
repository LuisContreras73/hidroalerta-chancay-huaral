"""
Script 06c: Corrección espacial PISCOt — delta dependiente de elevación
HidroAlerta Chancay-Huaral

Problema con Script 19 (delta uniforme)
----------------------------------------
Un solo delta por mes aplica la misma corrección a Donoso (350m) y a la
cabecera (4500m). El gradiente lapse rate real es ~5–6°C/1000m, por lo que
el sesgo de PISCOt también varía con altitud.

Metodología (elevación-dependiente)
--------------------------------------
Estaciones con datos reales de temperatura:
  donoso      350m   medium confidence  (1984-2014)
  huayan     2800m   low confidence     (1963-2014)
  picoy      2990m   medium confidence  (1967-2013)

Para cada mes m y variable (tmax, tmin):
  1. delta_s[m] = mean(T_obs_s | mes==m) − mean(T_pisco_pixel_s | mes==m)
     (usando período de solapamiento de cada estación con PISCOt 1981-2020)
  2. Regresión ponderada:  delta[m] = a[m] + b[m] × elev_m
     Pesos: donoso=0.70, huayan=0.40 (low conf), picoy=0.70
  3. Para cada píxel PISCOt (i,j):
     T_corr(i,j,t) = T_pisco(i,j,t) + a[month_t] + b[month_t] × DEM(i,j)

Limitación documentada:
  Solo 3 estaciones → regresión bien constrained para 350-3000m.
  La extrapolación a 3000-5000m asume que la relación delta~elev es lineal.
  Incertidumbre mayor en sub-cuencas altas (sub_649, sub_656 > 4000m).

Salidas
-------
  data/bronze/B2_tmax_corrected_spatial.nc      — Tmax corregida, grilla v1.2
  data/bronze/B2_tmin_corrected_spatial.nc      — Tmin corregida
  data/silver/B6b_T_spatial_stats.csv           — bias por estación/mes antes/después
  data/silver/B6c_piscot_corrected_subcuencas.csv — climatología por sub-cuenca
  outputs/figures/basin/V13_spatial_T_correction.png
"""

import sys
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows cp1252 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
TMAX_NC    = ROOT / "data/bronze/B2_tmax_basin_grid_v12.nc"
TMIN_NC    = ROOT / "data/bronze/B2_tmin_basin_grid_v12.nc"
DEM_NC     = ROOT / "data/bronze/B4_dem_basin_90m.nc"
CAT_CSV    = ROOT / "data/silver/senamhi/S2_senamhi_catalog.csv"
STA_DIR    = ROOT / "data/silver/senamhi"
BRONZE_DIR = ROOT / "data/bronze"
SILVER_DIR = ROOT / "data/silver"
FIG_DIR    = ROOT / "outputs/figures/basin"

TMAX_OUT   = BRONZE_DIR / "B2_tmax_corrected_spatial.nc"
TMIN_OUT   = BRONZE_DIR / "B2_tmin_corrected_spatial.nc"
STATS_OUT  = SILVER_DIR / "B6b_T_spatial_stats.csv"
CLIM_OUT   = SILVER_DIR / "B6c_piscot_corrected_subcuencas.csv"
FIG_OUT    = FIG_DIR / "V13_spatial_T_correction.png"

# ── Estaciones con datos de temperatura y sus pesos ───────────────────────────
CONF_WEIGHT = {'high': 1.0, 'medium': 0.70, 'low': 0.40}

# ── Sub-cuencas (centroides para extracción final) ────────────────────────────
SUBCUENCA_META = {
    'sub_634': {'lat': -11.59, 'lon': -77.13, 'elev_m': 521,  'nombre': 'Bajo Chancay'},
    'sub_640': {'lat': -11.49, 'lon': -76.96, 'elev_m': 1480, 'nombre': 'Aucallama'},
    'sub_641': {'lat': -11.37, 'lon': -76.76, 'elev_m': 3470, 'nombre': 'Medio Bajo'},
    'sub_646': {'lat': -11.27, 'lon': -76.69, 'elev_m': 3812, 'nombre': 'Pallac-Parquin'},
    'sub_649': {'lat': -11.25, 'lon': -76.55, 'elev_m': 4507, 'nombre': 'Alto Chancay'},
    'sub_650': {'lat': -11.26, 'lon': -76.84, 'elev_m': 2841, 'nombre': 'Anasmayo'},
    'sub_653': {'lat': -11.32, 'lon': -77.00, 'elev_m': 1664, 'nombre': 'Medio Alto'},
    'sub_655': {'lat': -11.13, 'lon': -76.75, 'elev_m': 4058, 'nombre': 'Carac'},
    'sub_656': {'lat': -11.11, 'lon': -76.60, 'elev_m': 4466, 'nombre': 'Baños'},
}


# ── 1. Cargar y filtrar estaciones con temperatura ────────────────────────────
def load_temp_stations() -> dict:
    cat = pd.read_csv(CAT_CSV)
    stations = {}
    for _, row in cat.iterrows():
        sid = row['station_id']
        fpath = STA_DIR / row['silver_file']
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        tmax_n = df['tmax_c'].notna().sum() if 'tmax_c' in df.columns else 0
        tmin_n = df['tmin_c'].notna().sum() if 'tmin_c' in df.columns else 0
        if tmax_n < 365:
            continue  # menos de 1 año de datos → omitir
        stations[sid] = {
            'df':   df,
            'lat':  float(row['lat']),
            'lon':  float(row['lon']),
            'elev': float(row['alt_m']),
            'name': row['station_name'],
            'conf': str(row.get('coord_confidence', 'low')),
            'weight': CONF_WEIGHT.get(str(row.get('coord_confidence', 'low')), 0.4),
        }
        print(f"  [{sid:15s}] {row['station_name']:20s}  "
              f"elev={row['alt_m']:.0f}m  tmax={tmax_n}d  conf={row.get('coord_confidence','?')}")
    return stations


# ── 2. Extraer series PISCOt en ubicación de cada estación ───────────────────
def extract_pisco_at_stations(ds_pisco: xr.Dataset, var: str,
                               stations: dict) -> dict:
    result = {}
    # Normalizar timestamps PISCOt a medianoche (sin componente horaria)
    time_idx = pd.DatetimeIndex(ds_pisco.time.values).normalize()
    for sid, info in stations.items():
        pt = ds_pisco[var].sel(lat=info['lat'], lon=info['lon'], method='nearest')
        result[sid] = pd.Series(pt.values, index=time_idx)
    return result


# ── 3. Calcular bias mensual por estación ────────────────────────────────────
def monthly_bias(stations: dict, pisco_at_sta: dict, var: str) -> pd.DataFrame:
    """delta[station, month] = mean(obs) - mean(pisco_pixel)"""
    rows = []
    for sid, info in stations.items():
        obs = info['df'][var].copy()
        # Normalizar a medianoche para alinear con PISCOt
        obs.index = pd.DatetimeIndex(obs.index).normalize()
        pis = pisco_at_sta[sid]
        # Alinear al período de solapamiento
        common = obs.index.intersection(pis.index)
        obs_c = obs.loc[common]
        pis_c = pis.loc[common]
        # Solo días con ambos válidos
        valid = obs_c.notna() & pis_c.notna()
        obs_c = obs_c[valid]
        pis_c = pis_c[valid]
        if len(obs_c) < 30:
            print(f"  AVISO: {sid} tiene <30 días válidos para {var}, omitido")
            continue
        for m in range(1, 13):
            mask = obs_c.index.month == m
            if mask.sum() < 10:
                continue
            delta = float(obs_c[mask].mean() - pis_c[mask].mean())
            rows.append({
                'station_id': sid,
                'elev_m':     info['elev'],
                'weight':     info['weight'],
                'month':      m,
                'delta':      delta,
                'n_days':     int(mask.sum()),
            })
    return pd.DataFrame(rows)


# ── 4. Regresión delta ~ elevación por mes ────────────────────────────────────
def fit_lapse_correction(bias_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta: delta[m] = a[m] + b[m] × elevación
    Devuelve DataFrame con columnas: month, a, b, r2, n_stations
    """
    rows = []
    for m in range(1, 13):
        df_m = bias_df[bias_df['month'] == m].dropna(subset=['delta'])
        if len(df_m) < 2:
            print(f"  Mes {m:02d}: solo {len(df_m)} estación(es), sin regresión — delta=0")
            rows.append({'month': m, 'a': 0.0, 'b': 0.0, 'r2': np.nan, 'n_sta': len(df_m)})
            continue
        x = df_m['elev_m'].values
        y = df_m['delta'].values
        w = df_m['weight'].values
        # Regresión ponderada: minimizar sum(w*(y - a - b*x)^2)
        # Equivalente a scipy linregress con pesos
        W  = np.diag(w)
        X  = np.column_stack([np.ones(len(x)), x])
        # Solución por mínimos cuadrados ponderados
        try:
            XtW  = X.T @ W
            coef = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
            a, b = float(coef[0]), float(coef[1])
        except Exception:
            a, b = float(np.average(y, weights=w)), 0.0
        y_hat = a + b * x
        ss_res = np.sum(w * (y - y_hat) ** 2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan
        print(f"  Mes {m:02d}: delta = {a:+.3f} + {b*1000:+.4f}°C/km × elev  "
              f"R²={r2:.3f}  n={len(df_m)}")
        rows.append({'month': m, 'a': a, 'b': b, 'r2': r2, 'n_sta': len(df_m)})
    return pd.DataFrame(rows).set_index('month')


# ── 5. Regrillado DEM a resolución PISCOt ────────────────────────────────────
def regrid_dem(dem_nc: Path, pisco_lats: np.ndarray,
               pisco_lons: np.ndarray) -> np.ndarray:
    """Interpola el DEM 90m a la grilla PISCOt v1.2 (0.01°)."""
    ds_dem = xr.open_dataset(str(dem_nc))
    dem_lats = ds_dem['lat'].values  # descendente o ascendente
    dem_lons = ds_dem['lon'].values
    dem_vals = ds_dem['dem'].values.astype(float)
    ds_dem.close()

    # Asegurar orden ascendente para el interpolador
    if dem_lats[0] > dem_lats[-1]:
        dem_lats = dem_lats[::-1]
        dem_vals = dem_vals[::-1, :]

    interp = RegularGridInterpolator(
        (dem_lats, dem_lons), dem_vals,
        method='linear', bounds_error=False, fill_value=None
    )
    # Grilla PISCOt como puntos
    lo_grid, la_grid = np.meshgrid(pisco_lons, pisco_lats)  # (nlat, nlon)
    pts = np.column_stack([la_grid.ravel(), lo_grid.ravel()])
    dem_pisco = interp(pts).reshape(len(pisco_lats), len(pisco_lons))
    # Fill NaN con media de cuenca (para píxeles fuera del DEM)
    basin_mean_elev = float(np.nanmean(dem_pisco))
    dem_pisco = np.where(np.isnan(dem_pisco), basin_mean_elev, dem_pisco)
    print(f"  DEM regrillado: shape={dem_pisco.shape}  "
          f"elev=[{dem_pisco.min():.0f}, {dem_pisco.max():.0f}]m  "
          f"media={basin_mean_elev:.0f}m")
    return dem_pisco


# ── 6. Aplicar corrección espacial al NetCDF completo ────────────────────────
def apply_spatial_correction(nc_in: Path, nc_out: Path, var: str,
                              lapse_params: pd.DataFrame,
                              dem_pisco: np.ndarray) -> xr.Dataset:
    """
    Aplica T_corr(i,j,t) = T(i,j,t) + a[mes_t] + b[mes_t] × DEM(i,j)
    Carga el NetCDF en chunks para no saturar memoria (~90 MB).
    """
    print(f"  Aplicando corrección a {nc_in.name} → {nc_out.name}")
    ds = xr.open_dataset(str(nc_in))
    t_arr = ds[var].values.astype(np.float32)   # (time, lat, lon)
    times = pd.DatetimeIndex(ds.time.values)
    months = times.month

    # Campo de corrección por mes: (12, nlat, nlon)
    # delta(i,j,m) = a[m] + b[m] * DEM(i,j)
    nlat, nlon = dem_pisco.shape
    delta_field = np.zeros((12, nlat, nlon), dtype=np.float32)
    for m in range(1, 13):
        if m not in lapse_params.index:
            continue
        a = float(lapse_params.loc[m, 'a'])
        b = float(lapse_params.loc[m, 'b'])
        delta_field[m - 1] = (a + b * dem_pisco).astype(np.float32)

    # Aplicar: vectorizado por timestep (evitar loop explícito)
    month_idx = (months - 1).values  # 0-based
    t_corr = t_arr + delta_field[month_idx]  # broadcasting: (T, lat, lon)

    # Construir Dataset de salida
    ds_corr = ds.copy(deep=False)
    ds_corr[var] = xr.DataArray(
        t_corr,
        coords=ds[var].coords,
        dims=ds[var].dims,
        attrs={**ds[var].attrs,
               'correction': 'elevation-dependent delta (Script 06c)',
               'correction_stations': 'donoso(350m), huayan(2800m), picoy(2990m)'}
    )
    ds_corr.to_netcdf(str(nc_out))
    ds.close()
    print(f"    Guardado: {nc_out.name}  ({nc_out.stat().st_size/1e6:.1f} MB)")
    return ds_corr


# ── 7. Extraer climatología por sub-cuenca del NetCDF corregido ───────────────
def extract_subcuenca_clim(ds_tmax: xr.Dataset, ds_tmin: xr.Dataset) -> pd.DataFrame:
    rows = []
    for eid, meta in SUBCUENCA_META.items():
        tmax_pt = ds_tmax['tmax'].sel(lat=meta['lat'], lon=meta['lon'], method='nearest')
        tmin_pt = ds_tmin['tmin'].sel(lat=meta['lat'], lon=meta['lon'], method='nearest')
        ser_tmax = pd.Series(tmax_pt.values, index=pd.DatetimeIndex(ds_tmax.time.values))
        ser_tmin = pd.Series(tmin_pt.values, index=pd.DatetimeIndex(ds_tmin.time.values))
        for m in range(1, 13):
            mask = ser_tmax.index.month == m
            rows.append({
                'entity_id': eid,
                'elev_m':    meta['elev_m'],
                'nombre':    meta['nombre'],
                'month':     m,
                'tmax_c':    round(float(ser_tmax[mask].mean()), 2),
                'tmin_c':    round(float(ser_tmin[mask].mean()), 2),
                'tmean_c':   round(float((ser_tmax[mask].mean() + ser_tmin[mask].mean()) / 2), 2),
            })
    return pd.DataFrame(rows)


# ── 8. Figura de validación ───────────────────────────────────────────────────
def plot_validation(bias_tmax: pd.DataFrame, lapse_tmax: pd.DataFrame,
                    bias_tmin: pd.DataFrame, lapse_tmin: pd.DataFrame,
                    clim_corr: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    MONTH_NAMES = ['Ene','Feb','Mar','Abr','May','Jun',
                   'Jul','Ago','Sep','Oct','Nov','Dic']
    ENTITY_ELEV = {e: m['elev_m'] for e, m in SUBCUENCA_META.items()}

    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.42,
                           left=0.07, right=0.97, top=0.91, bottom=0.08)

    # Panel 1: Scatter bias_tmax vs elevación para todos los meses
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.RdYlBu_r
    for m in range(1, 13):
        df_m = bias_tmax[bias_tmax['month'] == m]
        if df_m.empty:
            continue
        color = cmap((m - 1) / 11)
        ax1.scatter(df_m['elev_m'], df_m['delta'], color=color,
                    s=60, zorder=3, alpha=0.85)
        if m in lapse_tmax.index:
            a = lapse_tmax.loc[m, 'a']
            b = lapse_tmax.loc[m, 'b']
            xe = np.array([0, 5000])
            ax1.plot(xe, a + b * xe, color=color, lw=0.9, alpha=0.6)
    ax1.axhline(0, color='#555', lw=0.8, ls='--')
    ax1.set_xlabel('Elevación (m)', fontsize=9)
    ax1.set_ylabel('Bias Tmax: obs − PISCO (°C)', fontsize=9)
    ax1.set_title('Regresión delta_Tmax ~ elev (por mes)', fontsize=9.5, fontweight='bold')
    ax1.set_xlim(0, 5000)
    ax1.text(0.97, 0.05, 'Curvas = regresión mensual', transform=ax1.transAxes,
             ha='right', fontsize=7.5, color='#666')

    # Panel 2: Scatter bias_tmin vs elevación
    ax2 = fig.add_subplot(gs[0, 1])
    for m in range(1, 13):
        df_m = bias_tmin[bias_tmin['month'] == m]
        if df_m.empty:
            continue
        color = cmap((m - 1) / 11)
        ax2.scatter(df_m['elev_m'], df_m['delta'], color=color,
                    s=60, zorder=3, alpha=0.85)
        if m in lapse_tmin.index:
            a = lapse_tmin.loc[m, 'a']
            b = lapse_tmin.loc[m, 'b']
            xe = np.array([0, 5000])
            ax2.plot(xe, a + b * xe, color=color, lw=0.9, alpha=0.6)
    ax2.axhline(0, color='#555', lw=0.8, ls='--')
    ax2.set_xlabel('Elevación (m)', fontsize=9)
    ax2.set_ylabel('Bias Tmin: obs − PISCO (°C)', fontsize=9)
    ax2.set_title('Regresión delta_Tmin ~ elev (por mes)', fontsize=9.5, fontweight='bold')
    ax2.set_xlim(0, 5000)

    # Panel 3: Lapse rate b[m] (°C/1000m) para tmax y tmin
    ax3 = fig.add_subplot(gs[0, 2])
    months_plot = lapse_tmax.index.tolist()
    b_tmax = [lapse_tmax.loc[m, 'b'] * 1000 for m in months_plot]
    b_tmin = [lapse_tmin.loc[m, 'b'] * 1000 for m in months_plot]
    x_pos = np.arange(len(months_plot))
    ax3.bar(x_pos - 0.2, b_tmax, width=0.38, color='#CC3322', alpha=0.8, label='Tmax')
    ax3.bar(x_pos + 0.2, b_tmin, width=0.38, color='#1565C0', alpha=0.8, label='Tmin')
    ax3.axhline(0, color='#555', lw=0.8, ls='--')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([MONTH_NAMES[m-1] for m in months_plot], fontsize=8, rotation=45)
    ax3.set_ylabel('b (°C/1000m)', fontsize=9)
    ax3.set_title('Slope delta~elev por mes (°C/km)', fontsize=9.5, fontweight='bold')
    ax3.legend(fontsize=8)

    # Panel 4: Climatología Tmax corregida por sub-cuenca (todos los meses)
    ax4 = fig.add_subplot(gs[1, 0])
    entity_colors = plt.cm.tab10(np.linspace(0, 0.9, len(SUBCUENCA_META)))
    for i, eid in enumerate(sorted(SUBCUENCA_META, key=lambda e: ENTITY_ELEV[e])):
        df_e = clim_corr[clim_corr['entity_id'] == eid]
        lbl = f"{SUBCUENCA_META[eid]['nombre']} ({ENTITY_ELEV[eid]:.0f}m)"
        ax4.plot(df_e['month'], df_e['tmax_c'], color=entity_colors[i],
                 lw=1.6, marker='o', ms=4, label=lbl)
    ax4.set_xlabel('Mes', fontsize=9)
    ax4.set_ylabel('Tmax corregida (°C)', fontsize=9)
    ax4.set_title('Tmax corregida — climatología 1981-2020', fontsize=9.5, fontweight='bold')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(MONTH_NAMES, fontsize=7.5, rotation=30)
    ax4.legend(fontsize=6.5, loc='upper right', framealpha=0.85)

    # Panel 5: Lapse rate observado mes a mes (Tmax corrected)
    ax5 = fig.add_subplot(gs[1, 1])
    lapse_obs = []
    for m in range(1, 13):
        df_m = clim_corr[clim_corr['month'] == m].sort_values('elev_m')
        if len(df_m) >= 3:
            slope, _, r, _, _ = scipy.stats.linregress(df_m['elev_m'], df_m['tmax_c'])
            lapse_obs.append(slope * 1000)  # °C/1000m
        else:
            lapse_obs.append(np.nan)
    ax5.bar(range(1, 13), lapse_obs, color='#CC5500', alpha=0.8, edgecolor='white', lw=0.5)
    ax5.axhline(-6.5, color='#1A2A4A', lw=1.0, ls='--', label='Lapse rate env. −6.5°C/km')
    ax5.axhline(-5.0, color='#2255AA', lw=0.8, ls=':', label='DALR −5°C/km')
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(MONTH_NAMES, fontsize=8, rotation=30)
    ax5.set_ylabel('Lapse rate Tmax (°C/1000m)', fontsize=9)
    ax5.set_title('Lapse rate espacial observado en sub-cuencas', fontsize=9.5, fontweight='bold')
    ax5.legend(fontsize=8)

    # Panel 6: Delta aplicado por sub-cuenca y mes (heatmap)
    ax6 = fig.add_subplot(gs[1, 2])
    entities_by_elev = sorted(SUBCUENCA_META, key=lambda e: ENTITY_ELEV[e])
    # Calcular delta neto por entidad y mes (corrected - original)
    # Aquí usamos solo el componente de altitud: b[m] * elev_entity
    delta_matrix = np.zeros((len(entities_by_elev), 12))
    for i, eid in enumerate(entities_by_elev):
        elev = ENTITY_ELEV[eid]
        for m in range(1, 13):
            if m in lapse_tmax.index:
                delta_matrix[i, m-1] = (lapse_tmax.loc[m,'a'] +
                                         lapse_tmax.loc[m,'b'] * elev)
    vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    im = ax6.imshow(delta_matrix, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax)
    ax6.set_xticks(range(12))
    ax6.set_xticklabels(MONTH_NAMES, fontsize=7.5, rotation=45)
    ax6.set_yticks(range(len(entities_by_elev)))
    ax6.set_yticklabels([f"{SUBCUENCA_META[e]['nombre']}\n{ENTITY_ELEV[e]:.0f}m"
                          for e in entities_by_elev], fontsize=7)
    ax6.set_title('Δ Tmax aplicado por sub-cuenca (°C)', fontsize=9.5, fontweight='bold')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    fig.suptitle(
        'V13 — Corrección espacial PISCOt: delta dependiente de elevación\n'
        'Estaciones: Donoso (350m) · Huayan (2800m) · Picoy (2990m)',
        fontsize=11, fontweight='bold', color='#1A2A4A', y=0.97
    )
    fig.savefig(str(FIG_OUT), dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt_mod
    plt_mod.close(fig)
    print(f"  Figura: {FIG_OUT.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('Script 06c: Corrección espacial PISCOt (delta~elevación)')
    print('=' * 65)

    # 1. Cargar estaciones con temperatura
    print('\n[1] Estaciones SENAMHI con datos de temperatura:')
    stations = load_temp_stations()
    if len(stations) < 2:
        print('ERROR: Se necesitan al menos 2 estaciones con datos de temperatura.')
        sys.exit(1)

    # 2. Cargar PISCOt v1.2
    print('\n[2] Cargando PISCOt v1.2...')
    ds_tmax = xr.open_dataset(str(TMAX_NC))
    ds_tmin = xr.open_dataset(str(TMIN_NC))
    pisco_lats = ds_tmax['lat'].values
    pisco_lons = ds_tmax['lon'].values
    print(f'  Tmax: {ds_tmax.sizes}')
    print(f'  Tmin: {ds_tmin.sizes}')

    # 3. Extraer series PISCOt en estaciones
    print('\n[3] Extrayendo PISCOt en ubicación de estaciones...')
    pisco_tmax_at_sta = extract_pisco_at_stations(ds_tmax, 'tmax', stations)
    pisco_tmin_at_sta = extract_pisco_at_stations(ds_tmin, 'tmin', stations)

    # 4. Bias mensual por estación
    print('\n[4] Calculando bias mensual (obs - PISCOt pixel)...')
    print('  --- Tmax ---')
    bias_tmax = monthly_bias(stations, pisco_tmax_at_sta, 'tmax_c')
    print('  --- Tmin ---')
    bias_tmin = monthly_bias(stations, pisco_tmin_at_sta, 'tmin_c')

    if bias_tmax.empty:
        print('ERROR: No se pudo calcular bias de Tmax. Verificar datos SENAMHI.')
        sys.exit(1)

    # 5. Regresión delta ~ elevación por mes
    print('\n[5] Regresión ponderada delta ~ elevación por mes:')
    print('  --- Tmax ---')
    lapse_tmax = fit_lapse_correction(bias_tmax)
    print('  --- Tmin ---')
    lapse_tmin = fit_lapse_correction(bias_tmin)

    # 6. Regrillado DEM
    print('\n[6] Regrillando DEM 90m → grilla PISCOt v1.2...')
    dem_pisco = regrid_dem(DEM_NC, pisco_lats, pisco_lons)

    # 7. Aplicar corrección y guardar NetCDF
    print('\n[7] Aplicando corrección espacial...')
    if TMAX_OUT.exists():
        print(f'  {TMAX_OUT.name} ya existe — sobreescribiendo')
    ds_tmax_corr = apply_spatial_correction(TMAX_NC, TMAX_OUT, 'tmax',
                                             lapse_tmax, dem_pisco)
    if TMIN_OUT.exists():
        print(f'  {TMIN_OUT.name} ya existe — sobreescribiendo')
    ds_tmin_corr = apply_spatial_correction(TMIN_NC, TMIN_OUT, 'tmin',
                                             lapse_tmin, dem_pisco)
    ds_tmax.close()
    ds_tmin.close()

    # 8. Climatología por sub-cuenca
    print('\n[8] Extrayendo climatología por sub-cuenca del NetCDF corregido...')
    ds_tmax_c2 = xr.open_dataset(str(TMAX_OUT))
    ds_tmin_c2 = xr.open_dataset(str(TMIN_OUT))
    clim_corr = extract_subcuenca_clim(ds_tmax_c2, ds_tmin_c2)
    ds_tmax_c2.close()
    ds_tmin_c2.close()
    clim_corr.to_csv(str(CLIM_OUT), index=False)
    print(f'  Climatología: {CLIM_OUT.name}  ({len(clim_corr)} filas)')

    # 9. Estadísticas
    stats_rows = []
    for _, row in bias_tmax.iterrows():
        stats_rows.append({**row.to_dict(), 'variable': 'tmax'})
    for _, row in bias_tmin.iterrows():
        stats_rows.append({**row.to_dict(), 'variable': 'tmin'})
    pd.DataFrame(stats_rows).to_csv(str(STATS_OUT), index=False)
    print(f'  Stats: {STATS_OUT.name}')

    # 10. Figura
    print('\n[9] Generando figura de validación V13...')
    plot_validation(bias_tmax, lapse_tmax, bias_tmin, lapse_tmin, clim_corr)

    print()
    print('Script 06c completado.')
    print()
    print('Próximos pasos:')
    print('  1. Revisar V13_spatial_T_correction.png — validar pendientes lapse rate')
    print('  2. Actualizar Script 20 para usar B2_t{max,min}_corrected_spatial.nc')
    print('     en lugar de la corrección delta uniforme de B6_T_correction.csv')
    print('  3. ANIM07 en Script 39 ya puede usar B6c_piscot_corrected_subcuencas.csv')
    print('     (actualizar ruta de datos en anim07())')


if __name__ == '__main__':
    main()
