#!/usr/bin/env python3
"""
Script 40: Disponibilidad Hídrica — Cuenca Chancay-Huaral

Metodología
-----------
1. Caracterización estadística de caudales observados (SNIRH 2020-2026)
   - Curva de duración de caudales (FDC) con percentiles hídricos clave
   - Hidrograma mensual promedio y estacionalidad
   - Valores de diseño: Q50, Q75, Q90, Q95 (ANA: mínimos ecológicos)

2. Balance hídrico cuenca — enfoque Budyko (1974), Fu (1981)
   - Estimación independiente de Q histórico 1981-2019 desde P y PET
   - Validación cruzada: Budyko vs Q observado 2020-2026
   - Cierre del balance anual: P ≈ Q + ET (escala plurianual)

3. Validación mensual y anual
   - Nivel mensual: P vs Q con análisis de desfase (lag)
   - Nivel anual: P_anual vs Q_anual en período de solapamiento 2021-2025
   - Coeficiente de escorrentía Cc = Q/P mensual y anual

4. Análisis por estación (4 puntos de aforo)
   - Santo Domingo Auto.  47E214D2  (614 m)  cuenca completa, 2020-09→
   - Sto. Domingo Conv.   PHISIS0137 (614 m)  paralela,        2023-01→
   - Puente Callantama    47E22148   (punto intermedio)         2023-04→
   - Vichaycocha          47E257D8   (laguna tributaria)        2023-01→

Fuentes
-------
  Q observado : data/silver/snirh/S1_snirh_daily_q.csv
  P corregida : data/bronze/B2_pisco_basin_mean_corrected.csv  col='pr'
  PET         : data/bronze/B2_pisco_basin_mean_corrected.csv  col='pet'

Salidas
-------
  outputs/figures/hidro/VH01_fdc_caudales.png       Curva de duración
  outputs/figures/hidro/VH02_hidrograma_mensual.png  Hidrograma promedio + P
  outputs/figures/hidro/VH03_balance_mensual.png     Balance hídrico mensual
  outputs/figures/hidro/VH04_balance_anual.png       P vs Q anual + Budyko
  outputs/figures/hidro/VH05_escorrentia_anual.png   Cc y disponibilidad anual
  outputs/figures/hidro/VH06_estaciones_q.png        Comparación 4 estaciones
  data/silver/S7_disponibilidad_hidrica.csv          Tabla resumen mensual/anual

Trazabilidad (lineage)
----------------------
  Script: 40_disponibilidad_hidrica.py
  Fuente pr  : PISCOp v2.1 update → QM mensual (Script 19) → B2_pisco_basin_mean_corrected.csv
  Fuente pet : PM 1981-2016 + HS calibrado 2017-2020 (Scripts 07/07b)
  Fuente Q   : SNIRH ANA — S1_snirh_daily_q.csv (Scripts 03)
"""
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data/metadata"))
from entity_labels import entity_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("disp_hidrica")

# ── Parámetros cuenca ─────────────────────────────────────────────────────────
AREA_KM2   = 3062.62          # área cuenca total (UTM 18S, Script 10b)
AREA_M2    = AREA_KM2 * 1e6
MM_PER_M3S = 86400 / AREA_M2 * 1000   # factor: m³/s → mm/día

NOMBRES_EST = {
    "q_santo_domingo_47e214d2":     "Sto. Domingo Auto. 47E214D2 (614 m)",
    "q_santo_domingo_phisis0137":   "Sto. Domingo Conv. PHISIS0137 (614 m)",
    "q_puente_callantama_47e22148": "Puente Callantama 47E22148",
    "q_vichaycocha_47e257d8":       "Vichaycocha 47E257D8 (laguna)",
}
COLORES_EST = {
    "q_santo_domingo_47e214d2":     "#1f77b4",
    "q_santo_domingo_phisis0137":   "#aec7e8",
    "q_puente_callantama_47e22148": "#ff7f0e",
    "q_vichaycocha_47e257d8":       "#2ca02c",
}

# ── Rutas ─────────────────────────────────────────────────────────────────────
Q_CSV    = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
BM_CSV   = ROOT / "data/bronze/B2_pisco_basin_mean_corrected.csv"
FIG_DIR  = ROOT / "outputs/figures/hidro"
OUT_CSV  = ROOT / "data/silver/S7_disponibilidad_hidrica.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Carga de datos
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    log.info("Cargando datos...")

    # Q observado (m³/s) → también en mm/día
    q_raw = pd.read_csv(Q_CSV, index_col=0, parse_dates=True)
    q_raw.index = pd.DatetimeIndex(q_raw.index).normalize()

    # Estación principal: Santo Domingo 47E214D2
    q_sd = q_raw["q_santo_domingo_47e214d2"].dropna()
    q_sd_mm = q_sd * MM_PER_M3S   # mm/día

    # Forzantes PISCO
    bm = pd.read_csv(BM_CSV, index_col=0, parse_dates=True)
    bm.index = pd.DatetimeIndex(bm.index).normalize()
    pr  = bm["pr"].dropna()    # mm/día, 1981-2019
    pet = bm["pet"].dropna()   # mm/día, 1981-2020

    log.info(f"  Q Santo Domingo: {q_sd.index.min().date()} → {q_sd.index.max().date()}  N={len(q_sd)}")
    log.info(f"  P corregida:     {pr.index.min().date()} → {pr.index.max().date()}")
    log.info(f"  PET:             {pet.index.min().date()} → {pet.index.max().date()}")
    log.info(f"  Q media: {q_sd.mean():.2f} m³/s = {q_sd_mm.mean():.4f} mm/día = {q_sd_mm.mean()*365:.1f} mm/año")
    log.info(f"  P media (1981-2019): {pr.mean():.4f} mm/día = {pr.mean()*365:.1f} mm/año")
    log.info(f"  PET media (1981-2020): {pet.mean():.4f} mm/día = {pet.mean()*365:.1f} mm/año")

    return q_raw, q_sd, q_sd_mm, pr, pet


# ══════════════════════════════════════════════════════════════════════════════
# 2. Curva de Duración de Caudales (FDC)
# ══════════════════════════════════════════════════════════════════════════════
def compute_fdc(series: pd.Series) -> pd.DataFrame:
    """
    Curva de duración de caudales (FDC).
    Probabilidad de excedencia = rango / (n+1)  [Weibull plotting position]
    """
    sorted_q = series.sort_values(ascending=False).reset_index(drop=True)
    n = len(sorted_q)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100   # %
    return pd.DataFrame({"exceedance_pct": exceedance, "q_m3s": sorted_q.values})


def plot_VH01_fdc(q_sd: pd.Series, q_raw: pd.DataFrame, fig_dir: Path):
    """VH01: Curva de Duración de Caudales — Santo Domingo + otras estaciones."""
    log.info("  VH01: FDC caudales...")
    fdc_sd = compute_fdc(q_sd)

    # Percentiles clave (ANA: Q75 = caudal ecológico mínimo en Perú)
    q_pct = {p: float(np.percentile(q_sd.values, 100 - p)) for p in [10, 25, 50, 75, 90, 95]}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Panel izquierdo: FDC en escala log
    ax = axes[0]
    ax.semilogy(fdc_sd["exceedance_pct"], fdc_sd["q_m3s"],
                color="#1f77b4", lw=2, label="Sto. Domingo 47E214D2\n2020-09 → 2026-05")

    # Líneas verticales en percentiles clave
    colores_p = {"Q50": "#ff7f0e", "Q75": "#d62728", "Q90": "#9467bd", "Q95": "#8c564b"}
    for p, (label, col) in zip([50, 75, 90, 95], colores_p.items()):
        qv = q_pct[p]
        ax.axvline(p, color=col, lw=1.4, ls="--", alpha=0.8)
        ax.axhline(qv, color=col, lw=1.0, ls=":", alpha=0.6)
        ax.text(p + 1, qv * 1.05, f"{label}={qv:.1f} m³/s", fontsize=8, color=col)

    # Agregar otras estaciones disponibles (si tienen suficientes datos)
    for col, nombre in NOMBRES_EST.items():
        if col == "q_santo_domingo_47e214d2":
            continue
        s = q_raw[col].dropna()
        if len(s) > 180:
            fdc_i = compute_fdc(s)
            ax.semilogy(fdc_i["exceedance_pct"], fdc_i["q_m3s"],
                        color=COLORES_EST[col], lw=1.2, ls="--",
                        alpha=0.7, label=nombre.split("(")[0].strip())

    ax.set_xlabel("Probabilidad de excedencia (%)", fontsize=11)
    ax.set_ylabel("Caudal (m³/s)", fontsize=11)
    ax.set_title("Curva de Duración de Caudales — Cuenca Chancay-Huaral", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # Clasificar régimen hídrico
    cv = q_sd.std() / q_sd.mean()
    q10 = q_pct[10]; q90 = q_pct[90]
    ax.text(0.02, 0.05,
            f"CV = {cv:.2f}   Q10/Q90 = {q10/q90:.1f}\n"
            f"Régimen: {'alto' if cv > 1 else 'moderado' if cv > 0.5 else 'estable'}",
            transform=ax.transAxes, fontsize=9, color="#333333",
            bbox=dict(fc="lightyellow", alpha=0.8, pad=4))

    # Panel derecho: tabla de percentiles
    ax2 = axes[1]
    ax2.axis("off")
    data_tabla = [
        ["Excedencia", "Q (m³/s)", "Q (mm/día)"],
        ["Q10 (10%)", f"{q_pct[10]:.2f}", f"{q_pct[10]*MM_PER_M3S:.4f}"],
        ["Q25 (25%)", f"{q_pct[25]:.2f}", f"{q_pct[25]*MM_PER_M3S:.4f}"],
        ["Q50 (50%)", f"{q_pct[50]:.2f}", f"{q_pct[50]*MM_PER_M3S:.4f}"],
        ["Q75 (75%)", f"{q_pct[75]:.2f}", f"{q_pct[75]*MM_PER_M3S:.4f}"],
        ["Q90 (90%)", f"{q_pct[90]:.2f}", f"{q_pct[90]*MM_PER_M3S:.4f}"],
        ["Q95 (95%)", f"{q_pct[95]:.2f}", f"{q_pct[95]*MM_PER_M3S:.4f}"],
        ["Media",     f"{q_sd.mean():.2f}", f"{q_sd.mean()*MM_PER_M3S:.4f}"],
        ["Máximo",    f"{q_sd.max():.2f}",  f"{q_sd.max()*MM_PER_M3S:.4f}"],
        ["Mínimo",    f"{q_sd.min():.2f}",  f"{q_sd.min()*MM_PER_M3S:.4f}"],
    ]
    tabla = ax2.table(cellText=data_tabla[1:], colLabels=data_tabla[0],
                      loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.1, 1.6)
    # Colorear fila Q75 (caudal ecológico ANA)
    for j in range(3):
        tabla[4, j].set_facecolor("#ffcccc")   # Q75 = caudal mínimo ecológico
    ax2.set_title("Caudales de Diseño\n(Sto. Domingo 47E214D2)", fontsize=10, fontweight="bold")
    ax2.text(0.5, 0.02, "Q75 (rojo): caudal ecológico mínimo\nreferencial según criterio ANA",
             ha="center", transform=ax2.transAxes, fontsize=7.5, style="italic", color="#555")

    plt.suptitle(f"FDC — N={len(q_sd)} días obs. ({q_sd.index.min().date()} → {q_sd.index.max().date()})",
                 fontsize=10, color="#444")
    plt.tight_layout()
    plt.savefig(fig_dir / "VH01_fdc_caudales.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("    VH01_fdc_caudales.png guardado")
    return q_pct


# ══════════════════════════════════════════════════════════════════════════════
# 3. Hidrograma mensual promedio
# ══════════════════════════════════════════════════════════════════════════════
def plot_VH02_hidrograma_mensual(q_sd: pd.Series, q_sd_mm: pd.Series,
                                  pr: pd.Series, fig_dir: Path) -> pd.DataFrame:
    """
    VH02: Hidrograma mensual promedio (Q obs) + precipitación media mensual.
    Dos períodos: P = climatología PISCO 1981-2019 / Q = obs 2020-2026.
    """
    log.info("  VH02: hidrograma mensual...")

    MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

    # Climatología mensual Q observado
    q_mon = q_sd.groupby(q_sd.index.month).agg(
        media="mean", p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
        p10=lambda x: np.percentile(x, 10),
        p90=lambda x: np.percentile(x, 90),
    )
    # Climatología mensual P (1981-2019)
    pr_mon = pr.groupby(pr.index.month).mean() * 30.44   # mm/día → mm/mes

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    x = np.arange(1, 13)
    # Precipitación como barras (eje secundario, invertido arriba)
    ax2.bar(x, pr_mon.values, color="#4daedb", alpha=0.4, width=0.6, label="P media mensual (PISCOp 1981-2019)")
    ax2.set_ylabel("Precipitación media mensual (mm/mes)", fontsize=10, color="#4daedb")
    ax2.tick_params(axis="y", labelcolor="#4daedb")
    ax2.set_ylim(0, pr_mon.max() * 3)   # deja espacio arriba para el Q

    # Caudal promedio con banda de incertidumbre
    ax1.fill_between(x, q_mon["p10"].values, q_mon["p90"].values,
                     color="#1f77b4", alpha=0.15, label="P10-P90 (Q obs)")
    ax1.fill_between(x, q_mon["p25"].values, q_mon["p75"].values,
                     color="#1f77b4", alpha=0.30, label="P25-P75 (Q obs)")
    ax1.plot(x, q_mon["media"].values, color="#1f77b4", lw=2.5, marker="o",
             markersize=6, label="Q media obs (2020-2026)")

    # Anotar valores medios
    for xi, yi in zip(x, q_mon["media"].values):
        ax1.text(xi, yi + 0.3, f"{yi:.1f}", ha="center", va="bottom", fontsize=8.5, color="#1f77b4")

    ax1.set_xlabel("Mes", fontsize=11)
    ax1.set_ylabel("Caudal (m³/s)", fontsize=11, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MESES)
    ax1.set_xlim(0.5, 12.5)
    ax1.grid(axis="y", alpha=0.3)

    # Estaciones hídricas
    ax1.axvspan(11.5, 12.5, color="#ffeeaa", alpha=0.4, zorder=0)   # Dic
    ax1.axvspan(0.5, 3.5,  color="#ffeeaa", alpha=0.4, zorder=0)    # Ene-Mar (estación húmeda)
    ax1.text(2, ax1.get_ylim()[1] * 0.95, "Estación húmeda\n(DJF)", ha="center",
             fontsize=8, color="#aa7700", style="italic")
    ax1.text(7.5, ax1.get_ylim()[1] * 0.95, "Estiaje\n(JJA)", ha="center",
             fontsize=8, color="#555555", style="italic")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    plt.title("Hidrograma Mensual Promedio — Cuenca Chancay-Huaral\n"
              "Q: Sto. Domingo 47E214D2 (2020-2026) | P: PISCOp v2.1 update (1981-2019)",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "VH02_hidrograma_mensual.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("    VH02_hidrograma_mensual.png guardado")
    return q_mon


# ══════════════════════════════════════════════════════════════════════════════
# 4. Balance hídrico mensual (P vs Q+ET estimado)
# ══════════════════════════════════════════════════════════════════════════════
def plot_VH03_balance_mensual(pr: pd.Series, pet: pd.Series,
                               q_sd_mm: pd.Series, fig_dir: Path):
    """
    VH03: Balance hídrico mensual.
    P_mes = Q_mes + AET_mes + ΔS_mes
    AET ≈ min(PET, P)  (estimación Budyko mensual)
    ΔS residual: diferencia
    Desfase P→Q: correlación cruzada para detectar lag.
    """
    log.info("  VH03: balance hídrico mensual...")

    MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    x = np.arange(1, 13)

    # Climatologías mensuales (mm/mes)
    pr_mon  = pr.groupby(pr.index.month).mean() * 30.44
    pet_mon = pet.groupby(pet.index.month).mean() * 30.44
    q_mon   = q_sd_mm.groupby(q_sd_mm.index.month).mean() * 30.44

    # AET estimada mensualmente (Budyko simplificado): AET = min(P, PET)
    aet_mon = np.minimum(pr_mon.values, pet_mon.values)
    residual = pr_mon.values - q_mon.values - aet_mon   # ΔS = P - Q - AET

    # ── Correlación cruzada P → Q usando climatología mensual (sin solapamiento diario) ──
    # P (1981-2019) y Q obs (2020-2026) no se solapan en el tiempo.
    # Usamos las climatologías mensuales de ambas series para estimar el desfase estacional.
    pr_clim_mon = pr.groupby(pr.index.month).mean().values          # 12 valores
    q_clim_mon  = q_sd_mm.groupby(q_sd_mm.index.month).mean().values  # 12 valores
    # Correlación cruzada circular (lag 0-11 meses)
    lags  = np.arange(0, 12)
    corrs = np.array([
        np.corrcoef(pr_clim_mon, np.roll(q_clim_mon, lag))[0, 1] for lag in lags
    ])
    best_lag = int(lags[np.argmax(corrs)])   # en meses

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # Panel superior izquierdo: componentes del balance mensual
    ax1 = fig.add_subplot(gs[0, :])
    w = 0.27
    ax1.bar(x - w, pr_mon.values, width=w, color="#4daedb", label="P (PISCOp 1981-2019)")
    ax1.bar(x,     q_mon.values,  width=w, color="#1f77b4", label="Q obs (2020-2026) [mm/mes]")
    ax1.bar(x + w, aet_mon,       width=w, color="#e3752c", alpha=0.8, label="AET estimada = min(P,PET)")
    ax1.plot(x, pr_mon.values, color="#0077aa", lw=1.2, ls="--", alpha=0.7)

    # Residual ΔS como puntos
    for xi, dS in zip(x, residual):
        col = "#cc0000" if dS < -5 else "#009900" if dS > 5 else "#888888"
        ax1.scatter(xi, max(dS, 0) if dS > 0 else 0, marker="^" if dS > 0 else "v",
                    c=col, s=50, zorder=5)

    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(MESES)
    ax1.set_ylabel("mm/mes", fontsize=10)
    ax1.set_title("Componentes del Balance Hídrico Mensual (climatología)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.text(0.01, 0.97,
             f"Lag P→Q (climatología mensual): {best_lag} mes(es) (r={corrs[best_lag]:.2f})",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(fc="lightyellow", alpha=0.85, pad=3))

    # Panel inferior izquierdo: Cc mensual
    ax2 = fig.add_subplot(gs[1, 0])
    Cc_mon = np.where(pr_mon.values > 0.1, q_mon.values / pr_mon.values, np.nan)
    bars = ax2.bar(x, Cc_mon, color=["#d62728" if v > 0.5 else "#1f77b4" if v > 0.2 else "#aec7e8"
                                      for v in np.nan_to_num(Cc_mon)], alpha=0.85)
    ax2.axhline(np.nanmean(Cc_mon), color="#333", lw=1.5, ls="--",
                label=f"Media anual Cc = {np.nanmean(Cc_mon):.2f}")
    ax2.set_xticks(x); ax2.set_xticklabels(MESES)
    ax2.set_ylabel("Coeficiente de escorrentía Cc = Q/P", fontsize=9)
    ax2.set_title("Cc mensual (Q obs / P PISCOp)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Panel inferior derecho: correlación cruzada P→Q
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(lags, corrs, color="#1f77b4", lw=2)
    ax3.axvline(best_lag, color="#d62728", lw=1.5, ls="--",
                label=f"Lag óptimo = {best_lag} días")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.fill_between(lags, 0, corrs, where=corrs > 0, alpha=0.2, color="#1f77b4")
    ax3.set_xlabel("Desfase en meses (P lidera Q — climatología mensual)", fontsize=10)
    ax3.set_ylabel("Correlación de Pearson (r)", fontsize=10)
    ax3.set_title("Correlación cruzada P → Q diario\n(período solapamiento)", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.text(best_lag + 1, corrs[best_lag] - 0.04, f"r={corrs[best_lag]:.2f}", fontsize=9, color="#d62728")

    plt.suptitle("Balance Hídrico Mensual — Cuenca Chancay-Huaral", fontsize=12, fontweight="bold")
    plt.savefig(fig_dir / "VH03_balance_mensual.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("    VH03_balance_mensual.png guardado")

    return {"Cc_mensual": Cc_mon, "lag_optimo_dias": int(best_lag), "r_lag": float(corrs[best_lag]),
            "nota_lag": "lag en meses (correlacion cruzada circular sobre climatologia mensual)"}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Balance anual: P vs Q + Curva de Budyko
# ══════════════════════════════════════════════════════════════════════════════
def _budyko_fu(phi: np.ndarray, omega: float) -> np.ndarray:
    """
    Fu (1981) — generalización de Budyko:
      AET/P = 1 + φ - (1 + φ^ω)^(1/ω)
    φ = PET/P (índice de aridez)
    ω = parámetro de paisaje (2.0-3.0 típico cuencas andinas)
    Q/P = 1 - AET/P
    """
    return 1.0 - (1.0 + phi - (1.0 + phi ** omega) ** (1.0 / omega))


def plot_VH04_balance_anual(pr: pd.Series, pet: pd.Series,
                             q_sd_mm: pd.Series, fig_dir: Path) -> dict:
    """
    VH04: Balance hídrico anual.
    Izq: Diagrama de Budyko (φ = PET/P, Q/P) con estimación histórica 1981-2019.
    Der: P anual vs Q anual en período de solapamiento (2021-2024).
    """
    log.info("  VH04: balance anual + Budyko...")

    # Resamplear a anual (año hidrológico: Sep-Ago)
    pr_ann  = pr.resample("YE").sum()           # mm/año
    pet_ann = pet.resample("YE").sum()          # mm/año
    q_ann   = q_sd_mm.resample("YE").sum()      # mm/año (sin rellenar NaN)

    # Años con >= 330 días de Q
    q_count = q_sd_mm.resample("YE").count()
    q_ann_complete = q_ann[q_count >= 330]

    # Estadísticas medias (toda la serie disponible)
    P_mean   = float(pr_ann.mean())
    PET_mean = float(pet_ann.mean())
    Q_mean   = float(q_ann_complete.mean()) if len(q_ann_complete) > 0 else float(q_sd_mm.mean() * 365)
    phi_mean = PET_mean / P_mean
    Cc_mean  = Q_mean / P_mean

    log.info(f"  Balance anual (1981-2019):")
    log.info(f"    P media   = {P_mean:.1f} mm/año")
    log.info(f"    PET media = {PET_mean:.1f} mm/año")
    log.info(f"    Q obs     = {Q_mean:.1f} mm/año (período {q_ann_complete.index.min().year if len(q_ann_complete)>0 else 'NA'}-{q_ann_complete.index.max().year if len(q_ann_complete)>0 else 'NA'})")
    log.info(f"    φ (PET/P) = {phi_mean:.3f}")
    log.info(f"    Cc (Q/P)  = {Cc_mean:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel izquierdo: Diagrama de Budyko ──────────────────────────────────
    ax1 = axes[0]
    phi_range = np.linspace(0.1, 3.0, 300)

    omegas = [(1.5, "#aec7e8", ":"), (2.0, "#7fcdbb", "--"),
              (2.5, "#1f77b4", "-"), (3.0, "#08519c", "-.")]
    for om, col, ls in omegas:
        Cc_bu = _budyko_fu(phi_range, om)
        Cc_bu = np.clip(Cc_bu, 0, 1)
        ax1.plot(phi_range, Cc_bu, color=col, lw=1.5, ls=ls, label=f"Budyko-Fu (ω={om})")

    # Límites físicos
    ax1.plot([0, 1], [1, 0], color="gray", lw=1, ls="--", alpha=0.6, label="Límite árido (ET=P)")
    ax1.axhline(1, color="gray", lw=1, ls="--", alpha=0.6, label="Límite húmedo (ET=PET=P)")

    # Punto de la cuenca (observado)
    Q_P_obs = Q_mean / P_mean
    ax1.scatter([phi_mean], [Q_P_obs], s=150, color="#d62728", zorder=10,
                marker="*", label=f"Chancay-Huaral (φ={phi_mean:.2f}, Q/P={Q_P_obs:.2f})")

    # Estimaciones Budyko para distintos ω
    for om, col, _ in omegas:
        Cc_est = _budyko_fu(np.array([phi_mean]), om)[0]
        ax1.scatter([phi_mean], [Cc_est], s=50, color=col, zorder=8, marker="D", alpha=0.7)

    ax1.set_xlabel("Índice de aridez φ = PET/P", fontsize=11)
    ax1.set_ylabel("Coeficiente de escorrentía Q/P", fontsize=11)
    ax1.set_title("Diagrama de Budyko\nEstimación vs observado", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 3); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.text(0.02, 0.08,
             f"P={P_mean:.0f} mm/a | PET={PET_mean:.0f} mm/a\n"
             f"Q_obs={Q_mean:.0f} mm/a | Cc={Cc_mean:.2f}\n"
             f"AET_obs≈{P_mean - Q_mean:.0f} mm/a",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(fc="lightyellow", alpha=0.85, pad=4))

    # ── Panel derecho: P anual vs Q anual (período solapamiento) ─────────────
    ax2 = axes[1]

    # Años donde hay datos de ambos
    pr_ann_idx = pr_ann.to_frame("P")
    q_ann_idx  = q_ann_complete.to_frame("Q") if len(q_ann_complete) > 0 else pd.DataFrame()

    if len(q_ann_complete) >= 2:
        overlap = pr_ann_idx.join(q_ann_idx, how="inner").dropna()
        if len(overlap) >= 2:
            sc = ax2.scatter(overlap["P"], overlap["Q"], s=100, c=overlap.index.year,
                             cmap="RdYlGn", zorder=5, edgecolors="k", linewidths=0.5)
            plt.colorbar(sc, ax=ax2, label="Año", shrink=0.8)
            for yr, row in overlap.iterrows():
                ax2.annotate(str(yr.year), (row["P"], row["Q"]),
                             textcoords="offset points", xytext=(4, 4), fontsize=8)
            # Línea de tendencia
            slope, intercept, r, *_ = stats.linregress(overlap["P"], overlap["Q"])
            x_fit = np.linspace(overlap["P"].min(), overlap["P"].max(), 50)
            ax2.plot(x_fit, slope * x_fit + intercept, color="#d62728", lw=1.5, ls="--",
                     label=f"Tendencia: r={r:.2f}")

    # Línea Cc promedio
    x_cc = np.linspace(400, 1200, 50)
    ax2.plot(x_cc, Cc_mean * x_cc, color="#1f77b4", lw=1.5, ls=":",
             label=f"Cc media = {Cc_mean:.2f}")

    ax2.set_xlabel("Precipitación anual P (mm/año) — PISCOp 1981-2019", fontsize=10)
    ax2.set_ylabel("Caudal anual Q (mm/año) — Q obs.", fontsize=10)
    ax2.set_title("P anual vs Q anual\n(período de solapamiento)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    if len(q_ann_complete) == 0:
        ax2.text(0.5, 0.5,
                 "No hay años completos de Q\n(mínimo 330 días)\nN disponible: {} días".format(len(q_sd_mm)),
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10,
                 bbox=dict(fc="lightyellow", alpha=0.85, pad=8))

    plt.suptitle("Balance Hídrico Anual — Cuenca Chancay-Huaral", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "VH04_balance_anual.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("    VH04_balance_anual.png guardado")

    return {
        "P_mean_mm_a": P_mean, "PET_mean_mm_a": PET_mean, "Q_mean_mm_a": Q_mean,
        "phi_aridez": phi_mean, "Cc_anual": Cc_mean,
        "AET_obs_mm_a": P_mean - Q_mean,
        "Budyko_Q_w2": float(_budyko_fu(np.array([phi_mean]), 2.0)[0] * P_mean),
        "Budyko_Q_w25": float(_budyko_fu(np.array([phi_mean]), 2.5)[0] * P_mean),
        "Budyko_Q_w3": float(_budyko_fu(np.array([phi_mean]), 3.0)[0] * P_mean),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Disponibilidad hídrica anual y volúmenes
# ══════════════════════════════════════════════════════════════════════════════
def plot_VH05_disponibilidad(pr: pd.Series, pet: pd.Series,
                              q_sd: pd.Series, q_sd_mm: pd.Series,
                              fig_dir: Path, balance: dict):
    """
    VH05: Disponibilidad hídrica histórica estimada 1981-2019.
    Q estimado = Budyko-Fu(ω=2.5) × P_anual (serie histórica).
    Q observado sobrepuesto donde existe (2020-2026).
    Volúmenes en Hm³/año.
    """
    log.info("  VH05: disponibilidad histórica...")

    AREA_HM2 = AREA_KM2 / 100   # km² → Hm² (1 km² = 100 Hm²)

    pr_ann   = pr.resample("YE").sum()         # mm/año (1981-2019, 39 años)
    pet_ann  = pet.resample("YE").sum()        # mm/año (1981-2020, 40 años)
    # Alinear al período común
    common   = pr_ann.index.intersection(pet_ann.index)
    pr_ann   = pr_ann.loc[common]
    pet_ann  = pet_ann.loc[common]
    q_est_mm = _budyko_fu((pet_ann / pr_ann).values, omega=2.5) * pr_ann.values
    q_est_hm3 = q_est_mm * AREA_HM2 / 10      # mm → Hm³

    pr_ann_hm3 = pr_ann.values * AREA_HM2 / 10

    # Q observado anual en Hm³
    q_obs_ann_mm  = q_sd_mm.resample("YE").sum()
    q_obs_ann_hm3 = q_obs_ann_mm * AREA_HM2 / 10

    years_est = pd.DatetimeIndex(common).year
    years_obs = pd.DatetimeIndex(q_obs_ann_mm.index).year

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # Panel superior: volúmenes anuales
    ax1 = axes[0]
    cols_est = ["#aec7e8" if q < np.percentile(q_est_hm3, 25) else
                "#d62728" if q > np.percentile(q_est_hm3, 75) else "#1f77b4"
                for q in q_est_hm3]
    ax1.bar(years_est, q_est_hm3, color=cols_est, alpha=0.75, label="Q estimado Budyko-Fu (ω=2.5)")
    ax1.bar(years_obs, q_obs_ann_hm3.values, color="#ff7f0e", alpha=0.85,
            label="Q observado SNIRH (2020-2026)")

    q_p50 = float(np.percentile(q_est_hm3, 50))
    q_p75 = float(np.percentile(q_est_hm3, 75))
    q_p90 = float(np.percentile(q_est_hm3, 90))
    ax1.axhline(q_p50, color="#1f77b4", lw=1.5, ls="--", label=f"Q50 = {q_p50:.0f} Hm³/a")
    ax1.axhline(q_p75, color="#ff7f0e", lw=1.5, ls="--", label=f"Q75 = {q_p75:.0f} Hm³/a")

    # Años ENSO (El Niño intensos)
    nino_yrs = [1982, 1983, 1997, 1998, 2015, 2016, 2017]
    for yr in nino_yrs:
        if yr in years_est:
            ax1.axvspan(yr - 0.4, yr + 0.4, color="red", alpha=0.12, zorder=0)

    ax1.set_ylabel("Disponibilidad hídrica (Hm³/año)", fontsize=11)
    ax1.set_title("Disponibilidad Hídrica Anual — Cuenca Chancay-Huaral\n"
                  "Rojo translúcido = años El Niño intenso  |  Azul claro = años secos (Q < Q25)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.text(0.01, 0.97,
             f"Período 1981-2019  |  Media = {float(np.mean(q_est_hm3)):.0f} Hm³/a  "
             f"|  Q25 = {float(np.percentile(q_est_hm3,25)):.0f}  "
             f"|  Q75 = {q_p75:.0f}  |  Q90 = {q_p90:.0f} Hm³/a",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(fc="lightyellow", alpha=0.85, pad=3))

    # Panel inferior: Cc interanual (1981-2019)
    ax2 = axes[1]
    phi_ann = pet_ann / pr_ann    # ya alineados en la misma variable
    Cc_est  = _budyko_fu(phi_ann.values, omega=2.5)
    ax2.plot(years_est, Cc_est, color="#1f77b4", lw=1.8, marker="o", ms=4, label="Cc estimado Budyko")
    ax2.axhline(float(np.mean(Cc_est)), color="#333", lw=1.2, ls="--",
                label=f"Media = {float(np.mean(Cc_est)):.3f}")
    ax2.fill_between(years_est, Cc_est, float(np.mean(Cc_est)),
                     where=Cc_est > float(np.mean(Cc_est)),
                     alpha=0.15, color="#1f77b4", label="Sobre la media")
    ax2.fill_between(years_est, Cc_est, float(np.mean(Cc_est)),
                     where=Cc_est < float(np.mean(Cc_est)),
                     alpha=0.15, color="#d62728", label="Bajo la media")
    ax2.set_ylabel("Coeficiente de escorrentía Cc = Q/P", fontsize=10)
    ax2.set_xlabel("Año", fontsize=10)
    ax2.set_title("Variabilidad interanual del coeficiente de escorrentía (Cc)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "VH05_escorrentia_anual.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("    VH05_escorrentia_anual.png guardado")

    # Retornar tabla de disponibilidad anual
    disp_df = pd.DataFrame({
        "year": years_est,
        "P_mm_a": pr_ann.values.astype(float),
        "Q_est_mm_a": q_est_mm.astype(float),
        "Q_est_hm3_a": q_est_hm3.astype(float),
        "Cc_est": Cc_est.astype(float),
    })
    return disp_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. Comparación 4 estaciones SNIRH
# ══════════════════════════════════════════════════════════════════════════════
def plot_VH06_estaciones(q_raw: pd.DataFrame, fig_dir: Path):
    """
    VH06: Serie temporal de las 4 estaciones + climatología mensual comparada.
    """
    log.info("  VH06: comparación 4 estaciones...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Panel superior: series temporales superpuestas
    ax1 = axes[0]
    for col, nombre in NOMBRES_EST.items():
        s = q_raw[col].dropna()
        if len(s) > 0:
            ax1.plot(s.index, s.values, color=COLORES_EST[col], lw=1.2,
                     alpha=0.85, label=f"{nombre.split('(')[0].strip()} (N={len(s)})")

    ax1.set_ylabel("Caudal (m³/s)", fontsize=10)
    ax1.set_title("Series temporales — 4 estaciones SNIRH Chancay-Huaral", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(alpha=0.3)

    # Anotar contribución relativa de Vichaycocha
    q_sd   = q_raw["q_santo_domingo_47e214d2"].dropna()
    q_vich = q_raw["q_vichaycocha_47e257d8"].dropna()
    overlap_idx = q_sd.index.intersection(q_vich.index)
    if len(overlap_idx) > 30:
        contrib = q_vich.loc[overlap_idx].mean() / q_sd.loc[overlap_idx].mean() * 100
        ax1.text(0.01, 0.97, f"Contribución Vichaycocha ≈ {contrib:.1f}% del Q en Sto. Domingo",
                 transform=ax1.transAxes, fontsize=9, va="top",
                 bbox=dict(fc="#e8f4f8", alpha=0.9, pad=3))

    # Panel inferior: climatología mensual por estación
    ax2 = axes[1]
    MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    x = np.arange(1, 13)
    w = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for (col, nombre), offset in zip(NOMBRES_EST.items(), offsets):
        s = q_raw[col].dropna()
        if len(s) < 180:
            continue
        mon_mean = s.groupby(s.index.month).mean()
        # Rellenar meses faltantes con NaN
        full = pd.Series(np.nan, index=range(1, 13))
        full.update(mon_mean)
        ax2.bar(x + offset * w, full.values, width=w, color=COLORES_EST[col], alpha=0.8,
                label=nombre.split("(")[0].strip())

    ax2.set_xticks(x); ax2.set_xticklabels(MESES)
    ax2.set_ylabel("Caudal medio mensual (m³/s)", fontsize=10)
    ax2.set_title("Climatología mensual por estación", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "VH06_estaciones_q.png", dpi=180, bbox_inches="tight")
    plt.close()
    log.info("    VH06_estaciones_q.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Tabla resumen y CSV de salida
# ══════════════════════════════════════════════════════════════════════════════
def save_summary(q_sd: pd.Series, q_pct: dict, balance: dict,
                 q_mon: pd.DataFrame, disp_df: pd.DataFrame):
    """
    Guarda S7_disponibilidad_hidrica.csv y log del resumen completo.
    """
    MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

    # ── Tabla mensual ─────────────────────────────────────────────────────────
    mon_table = pd.DataFrame({
        "mes": MESES,
        "Q_media_m3s": q_mon["media"].values,
        "Q_p25_m3s":   q_mon["p25"].values,
        "Q_p75_m3s":   q_mon["p75"].values,
        "Q_media_mm_mes": q_mon["media"].values * MM_PER_M3S * 30.44,
    })

    # ── Tabla anual histórica (Budyko) ────────────────────────────────────────
    anual_table = disp_df.copy()

    # ── Resumen general ───────────────────────────────────────────────────────
    resumen = {
        "cuenca": "Chancay-Huaral",
        "area_km2": AREA_KM2,
        "periodo_observado": f"2020-09-01 → {q_sd.index.max().date()}",
        "n_dias_obs": len(q_sd),
        "Q_media_m3s":   round(q_sd.mean(), 3),
        "Q_media_mm_a":  round(q_sd.mean() * MM_PER_M3S * 365, 1),
        "Q_media_Hm3_a": round(q_sd.mean() * MM_PER_M3S * 365 * AREA_KM2 / 100 / 10, 1),
        "Q50_m3s":  round(q_pct[50], 3),
        "Q75_m3s":  round(q_pct[75], 3),
        "Q90_m3s":  round(q_pct[90], 3),
        "Q95_m3s":  round(q_pct[95], 3),
        "P_media_mm_a_1981_2019": round(balance["P_mean_mm_a"], 1),
        "PET_media_mm_a": round(balance["PET_mean_mm_a"], 1),
        "phi_aridez": round(balance["phi_aridez"], 3),
        "Cc_anual_obs": round(balance["Cc_anual"], 3),
        "AET_obs_mm_a": round(balance["AET_obs_mm_a"], 1),
        "Q_Budyko_w20_mm_a": round(balance["Budyko_Q_w2"], 1),
        "Q_Budyko_w25_mm_a": round(balance["Budyko_Q_w25"], 1),
        "Q_Budyko_w30_mm_a": round(balance["Budyko_Q_w3"], 1),
        "fuente_pr": "PISCOp v2.1 update (QM corregido, Script 19)",
        "fuente_pet": "PISCO PM 1981-2016 + HS calibrado 2017-2020 (Scripts 07/07b)",
        "fuente_q": "SNIRH ANA — 47E214D2 Santo Domingo Automática",
        "generado_por": "scripts/40_disponibilidad_hidrica.py",
        "generado_el": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Guardar CSV (sección mensual)
    mon_table.to_csv(OUT_CSV, index=False)
    # Guardar JSON lineage + resumen
    resumen_path = OUT_CSV.with_suffix(".json")
    with open(resumen_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # Guardar tabla anual (aparte)
    anual_path = OUT_CSV.parent / "S7_disponibilidad_anual_1981_2019.csv"
    anual_table.to_csv(anual_path, index=False)

    # ── Imprimir resumen en log ───────────────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("DISPONIBILIDAD HÍDRICA — CUENCA CHANCAY-HUARAL")
    log.info("=" * 68)
    log.info(f"  Área: {AREA_KM2} km²")
    log.info(f"  Q observado (47E214D2): {resumen['Q_media_m3s']} m³/s = "
             f"{resumen['Q_media_mm_a']} mm/a = {resumen['Q_media_Hm3_a']} Hm³/a")
    log.info("")
    log.info("  Caudales de diseño (FDC observada):")
    log.info(f"    Q50 = {resumen['Q50_m3s']:>6.2f} m³/s  ({resumen['Q50_m3s']*MM_PER_M3S*365:.0f} mm/a)")
    log.info(f"    Q75 = {resumen['Q75_m3s']:>6.2f} m³/s  ({resumen['Q75_m3s']*MM_PER_M3S*365:.0f} mm/a)  ← caudal ecológico ref. ANA")
    log.info(f"    Q90 = {resumen['Q90_m3s']:>6.2f} m³/s  ({resumen['Q90_m3s']*MM_PER_M3S*365:.0f} mm/a)")
    log.info(f"    Q95 = {resumen['Q95_m3s']:>6.2f} m³/s  ({resumen['Q95_m3s']*MM_PER_M3S*365:.0f} mm/a)")
    log.info("")
    log.info("  Balance hídrico (Budyko-Fu, ω=2.5):")
    log.info(f"    P   media 1981-2019 = {resumen['P_media_mm_a_1981_2019']} mm/a")
    log.info(f"    PET media 1981-2020 = {resumen['PET_media_mm_a']} mm/a")
    log.info(f"    φ   (PET/P)         = {resumen['phi_aridez']}")
    log.info(f"    Q   estimado ω=2.0  = {resumen['Q_Budyko_w20_mm_a']} mm/a")
    log.info(f"    Q   estimado ω=2.5  = {resumen['Q_Budyko_w25_mm_a']} mm/a")
    log.info(f"    Q   estimado ω=3.0  = {resumen['Q_Budyko_w30_mm_a']} mm/a")
    log.info(f"    Q   observado       = {resumen['Q_media_mm_a']} mm/a  (Cc={resumen['Cc_anual_obs']})")
    log.info("")
    log.info("  Archivos generados:")
    log.info(f"    {OUT_CSV}")
    log.info(f"    {resumen_path}")
    log.info(f"    {anual_path}")
    for img in ["VH01","VH02","VH03","VH04","VH05","VH06"]:
        log.info(f"    outputs/figures/hidro/{img}_*.png")
    log.info("=" * 68)

    return resumen


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 68)
    log.info("SCRIPT 40: Disponibilidad Hídrica — Cuenca Chancay-Huaral")
    log.info("=" * 68)

    # 1. Datos
    q_raw, q_sd, q_sd_mm, pr, pet = load_data()

    # 2. FDC
    q_pct = plot_VH01_fdc(q_sd, q_raw, FIG_DIR)

    # 3. Hidrograma mensual
    q_mon = plot_VH02_hidrograma_mensual(q_sd, q_sd_mm, pr, FIG_DIR)

    # 4. Balance mensual + lag P→Q
    bal_mensual = plot_VH03_balance_mensual(pr, pet, q_sd_mm, FIG_DIR)
    log.info(f"  Lag óptimo P→Q (climatología mensual): {bal_mensual['lag_optimo_dias']} mes(es)  (r={bal_mensual['r_lag']:.3f})")

    # 5. Balance anual + Budyko
    balance = plot_VH04_balance_anual(pr, pet, q_sd_mm, FIG_DIR)

    # 6. Disponibilidad histórica 1981-2019
    disp_df = plot_VH05_disponibilidad(pr, pet, q_sd, q_sd_mm, FIG_DIR, balance)

    # 7. Comparación estaciones
    plot_VH06_estaciones(q_raw, FIG_DIR)

    # 8. Resumen y CSV
    save_summary(q_sd, q_pct, balance, q_mon, disp_df)

    log.info("=== SCRIPT 40 COMPLETADO ===")


if __name__ == "__main__":
    main()
