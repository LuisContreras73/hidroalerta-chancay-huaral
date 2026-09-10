#!/usr/bin/env python3
"""
Script 12: Índice ONI (ENSO) desde NOAA — Cuenca Chancay-Huaral.

Fuente de datos
---------------
NOAA Climate Prediction Center (CPC) — Oceanic Niño Index (ONI).
Descarga pública sin autenticación desde el servidor de NOAA.
URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

El ONI es la media móvil de 3 meses de las anomalías de TSM en la región
Niño 3.4 (5°N-5°S, 120°W-170°W), calculada con la base de referencia
centrada en 30 años actualizada cada 5 años (ERSST v5, 1991-2020).

Clasificación de eventos (NOAA CPC)
-------------------------------------
  El Niño  : ONI ≥ +0.5°C durante al menos 5 trimestres consecutivos
  La Niña  : ONI ≤ −0.5°C durante al menos 5 trimestres consecutivos
  Neutro   : −0.5 < ONI < +0.5

Relevancia para la cuenca Chancay-Huaral
-----------------------------------------
La cuenca se encuentra en la vertiente occidental de los Andes centrales
peruanos (lat ≈ −11°), donde ENSO modula la variabilidad interanual de la
precipitación:
  • El Niño costero (años como 1983, 1998, 2017) → lluvias extremas, crecidas
  • La Niña        → déficit hídrico, sequías agrícolas
La serie ONI se utiliza como covariable exógena en el modelo TFT.

Salidas
-------
  data/silver/enso/S3_oni_1950_2026.csv    — serie ONI mensual clasificada
  outputs/figures/basin/T03_enso_oni.png

Referencias
-----------
Trenberth (1997) doi:10.1175/1520-0477(1997)078<2771:TESO>2.0.CO;2  ONI definition
Huang et al. (2017) doi:10.1175/JCLI-D-15-0743.1  ERSST v5
"""
import logging, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("enso_oni")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "data/silver/enso"
OUT_CSV = OUT_DIR / "S3_oni_1950_2026.csv"
FIG_DIR = ROOT / "outputs/figures/basin"
OUT_FIG = FIG_DIR / "T03_enso_oni.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── URLs NOAA ─────────────────────────────────────────────────────────────────
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Mapa de trimestres a mes central (para indexar por mes)
SEAS_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

# Thresholds ONI (NOAA)
THR_NINO = 0.5
THR_NINA = -0.5
CONSEC_MIN = 5   # trimestres consecutivos mínimos


# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCARGA Y PARSEO
# ══════════════════════════════════════════════════════════════════════════════
def download_oni() -> pd.DataFrame:
    log.info(f"Descargando ONI desde NOAA CPC ...")
    r = requests.get(ONI_URL, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().split("\n")

    records = []
    for line in lines[1:]:   # saltar cabecera
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            seas  = parts[0].strip()
            yr    = int(parts[1])
            total = float(parts[2])
            anom  = float(parts[3]) if len(parts) > 3 else np.nan
        except (ValueError, IndexError):
            continue
        month = SEAS_TO_MONTH.get(seas, np.nan)
        if not np.isnan(month):
            records.append({
                "year"     : yr,
                "month"    : int(month),
                "season"   : seas,
                "sst_total": total,
                "oni"      : anom,
            })

    df = pd.DataFrame(records)
    # Crear columna de fecha (primer día del mes central del trimestre)
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}
    )
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"  Descargado: {len(df)} registros  "
             f"({df['year'].min()}–{df['year'].max()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASIFICAR EVENTOS
# ══════════════════════════════════════════════════════════════════════════════
def classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica cada mes como El Niño / La Niña / Neutro usando el criterio
    NOAA: 5 trimestres consecutivos por encima/debajo del umbral.
    """
    oni = df["oni"].values
    n   = len(oni)

    # Estado preliminar (sin restricción de consecutivos)
    state_raw = np.where(oni >=  THR_NINO, 1,
                np.where(oni <= THR_NINA, -1, 0))

    # Verificar 5 consecutivos
    state_final = np.zeros(n, dtype=int)
    for event_val, event_str in [(1, "El Niño"), (-1, "La Niña")]:
        in_event = False
        start    = None
        for i in range(n):
            if state_raw[i] == event_val:
                if not in_event:
                    in_event, start = True, i
            else:
                if in_event:
                    length = i - start
                    if length >= CONSEC_MIN:
                        state_final[start:i] = event_val
                    in_event = False
        if in_event:
            length = n - start
            if length >= CONSEC_MIN:
                state_final[start:n] = event_val

    df["event"] = pd.Categorical(
        np.where(state_final ==  1, "El Niño",
        np.where(state_final == -1, "La Niña", "Neutro")),
        categories=["El Niño", "Neutro", "La Niña"],
        ordered=True,
    )

    # Intensidad dentro de cada evento
    df["intensity"] = "—"
    df.loc[(df["event"] == "El Niño") & (df["oni"] >= 2.0), "intensity"] = "Fuerte"
    df.loc[(df["event"] == "El Niño") & (df["oni"] >= 1.5) & (df["oni"] < 2.0), "intensity"] = "Moderado"
    df.loc[(df["event"] == "El Niño") & (df["oni"] < 1.5), "intensity"] = "Débil"
    df.loc[(df["event"] == "La Niña") & (df["oni"] <= -1.5), "intensity"] = "Fuerte"
    df.loc[(df["event"] == "La Niña") & (df["oni"] > -1.5), "intensity"] = "Débil/Mod."

    nino_n = (df["event"] == "El Niño").sum()
    nina_n = (df["event"] == "La Niña").sum()
    log.info(f"  El Niño: {nino_n} meses  |  La Niña: {nina_n} meses  |  "
             f"Neutro: {n-nino_n-nina_n} meses")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURA T03
# ══════════════════════════════════════════════════════════════════════════════
def plot_oni(df: pd.DataFrame):
    log.info("Generando figura T03 ...")

    # Sub-período relevante para la cuenca (forzantes disponibles + Q obs)
    df_hist = df[df["year"].between(1981, 2026)].copy()

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.58, wspace=0.35)

    # ── (a) Serie ONI completa 1981-2026 ─────────────────────────────────────
    ax_full = fig.add_subplot(gs[0, :])
    oni = df_hist["oni"].values
    dates = df_hist["date"].values
    events = df_hist["event"].values

    ax_full.fill_between(dates, oni, 0,
                          where=(events == "El Niño"),
                          color="#e74c3c", alpha=0.70, label="El Niño")
    ax_full.fill_between(dates, oni, 0,
                          where=(events == "La Niña"),
                          color="#2980b9", alpha=0.70, label="La Niña")
    ax_full.fill_between(dates, oni, 0,
                          where=(events == "Neutro"),
                          color="#bdc3c7", alpha=0.40)
    ax_full.plot(dates, oni, color="#2c3e50", lw=0.8, alpha=0.7)
    ax_full.axhline(THR_NINO,  color="#e74c3c", ls="--", lw=1.0, alpha=0.8)
    ax_full.axhline(THR_NINA,  color="#2980b9", ls="--", lw=1.0, alpha=0.8)
    ax_full.axhline(0, color="gray", lw=0.6)
    # Etiquetar eventos fuertes
    strong = df_hist[(df_hist["intensity"] == "Fuerte")]
    for _, row in strong.groupby((strong.index.to_series().diff() > 6).cumsum()):
        peak_idx = row["oni"].abs().idxmax()
        ax_full.annotate(f"{row.loc[peak_idx,'year']}",
                         xy=(row.loc[peak_idx,"date"], row.loc[peak_idx,"oni"]),
                         xytext=(0, 8 * np.sign(row.loc[peak_idx,"oni"])),
                         textcoords="offset points", ha="center", fontsize=6.5,
                         color="#7f8c8d")
    ax_full.set_ylabel("ONI (°C anomalía)", fontsize=9)
    ax_full.set_title("(a) Índice Oceánico Niño (ONI) 1981-2026\n"
                      "Media móvil 3 meses de anomalías TSM Niño 3.4  |  "
                      "Líneas punteadas: ±0.5°C (umbrales NOAA)",
                      fontweight="bold", fontsize=9)
    ax_full.legend(fontsize=8, loc="upper left")
    ax_full.grid(alpha=0.3, lw=0.7)
    ax_full.set_xlim(dates[0], dates[-1])

    # ── (b) ONI reciente (2020-2026) — período con Q observado ───────────────
    ax_rec = fig.add_subplot(gs[1, :])
    df_rec = df_hist[df_hist["year"] >= 2020]
    oni_r  = df_rec["oni"].values
    dates_r = df_rec["date"].values
    ev_r    = df_rec["event"].values
    ax_rec.fill_between(dates_r, oni_r, 0,
                         where=(ev_r=="El Niño"), color="#e74c3c", alpha=0.70)
    ax_rec.fill_between(dates_r, oni_r, 0,
                         where=(ev_r=="La Niña"), color="#2980b9", alpha=0.70)
    ax_rec.fill_between(dates_r, oni_r, 0,
                         where=(ev_r=="Neutro"),  color="#bdc3c7", alpha=0.40)
    ax_rec.plot(dates_r, oni_r, "o-", color="#2c3e50", lw=1.4, ms=4)
    ax_rec.axhline(THR_NINO,  color="#e74c3c", ls="--", lw=1.0)
    ax_rec.axhline(THR_NINA,  color="#2980b9", ls="--", lw=1.0)
    ax_rec.axhline(0, color="gray", lw=0.6)
    # Marcar inicio de Q observado
    ax_rec.axvline(pd.Timestamp("2020-09-01"), color="#27ae60",
                   ls=":", lw=2.0, alpha=0.8, label="Inicio Q obs (Sep 2020)")
    ax_rec.set_ylabel("ONI (°C)", fontsize=9)
    ax_rec.set_title("(b) ONI período con caudal observado (2020-2026)\n"
                     "Verde punteado = inicio Q obs SNIRH Santo Domingo",
                     fontweight="bold", fontsize=9)
    ax_rec.legend(fontsize=8); ax_rec.grid(alpha=0.3, lw=0.7)

    # ── (c) Distribución mensual ONI (boxplots por mes) ───────────────────────
    ax_mon = fig.add_subplot(gs[2, 0])
    by_m   = df_hist.groupby("month")["oni"]
    bdata  = [by_m.get_group(m).values if m in by_m.groups else []
              for m in range(1, 13)]
    MESES  = ["Ene","Feb","Mar","Abr","May","Jun",
              "Jul","Ago","Sep","Oct","Nov","Dic"]
    bpr = ax_mon.boxplot(bdata, patch_artist=True,
                         medianprops=dict(color="k",lw=2),
                         flierprops=dict(marker=".", ms=4, alpha=0.5))
    import matplotlib.cm as mcm
    cmap_m = matplotlib.colormaps["RdBu_r"]
    clim_m = [float(np.nanmean(d)) for d in bdata]
    vabs   = max(abs(v) for v in clim_m if not np.isnan(v))
    for patch, cv in zip(bpr["boxes"], clim_m):
        norm_v = (cv + vabs) / (2 * vabs + 1e-6)
        patch.set_facecolor(cmap_m(norm_v)); patch.set_alpha(0.80)
    ax_mon.axhline(THR_NINO, color="#e74c3c", ls="--", lw=1.0)
    ax_mon.axhline(THR_NINA, color="#2980b9", ls="--", lw=1.0)
    ax_mon.axhline(0, color="gray", lw=0.7)
    ax_mon.set_xticks(range(1, 13)); ax_mon.set_xticklabels(MESES, fontsize=8)
    ax_mon.set_ylabel("ONI (°C)", fontsize=9)
    ax_mon.set_title("(c) Distribución mensual del ONI\n1981-2026",
                     fontweight="bold", fontsize=9)
    ax_mon.grid(axis="y", alpha=0.3, lw=0.7)

    # ── (d) Frecuencia de años El Niño / La Niña por década ──────────────────
    ax_dec = fig.add_subplot(gs[2, 1])
    decades = [(1981,1990),(1991,2000),(2001,2010),(2011,2020),(2021,2026)]
    nino_f, nina_f, dec_labels = [], [], []
    for y0, y1 in decades:
        sub = df_hist[df_hist["year"].between(y0, y1)]
        n_tot  = len(sub)
        nino_f.append(round((sub["event"]=="El Niño").sum() / n_tot * 100, 1))
        nina_f.append(round((sub["event"]=="La Niña").sum() / n_tot * 100, 1))
        dec_labels.append(f"{y0}-\n{y1}")
    x  = np.arange(len(decades))
    w  = 0.35
    ax_dec.bar(x - w/2, nino_f, w, color="#e74c3c", alpha=0.85, label="El Niño")
    ax_dec.bar(x + w/2, nina_f, w, color="#2980b9", alpha=0.85, label="La Niña")
    for i, (vn, va) in enumerate(zip(nino_f, nina_f)):
        ax_dec.text(i-w/2, vn+0.5, f"{vn:.0f}%", ha="center", fontsize=7)
        ax_dec.text(i+w/2, va+0.5, f"{va:.0f}%", ha="center", fontsize=7)
    ax_dec.set_xticks(x); ax_dec.set_xticklabels(dec_labels, fontsize=8)
    ax_dec.set_ylabel("Frecuencia (%)", fontsize=9)
    ax_dec.set_title("(d) Frecuencia de eventos por década\n"
                     "% meses con evento activo (criterio 5 trim. consecutivos)",
                     fontweight="bold", fontsize=9)
    ax_dec.legend(fontsize=8); ax_dec.grid(axis="y", alpha=0.3, lw=0.7)
    ax_dec.set_ylim(0, 60)

    fig.suptitle("Índice Oceánico Niño (ONI) — Cuenca Chancay-Huaral\n"
                 "NOAA CPC  |  ERSST v5  |  Base 1991-2020  |  "
                 "Trenberth (1997): El Niño ≥ +0.5°C / La Niña ≤ −0.5°C por ≥5 trimestres",
                 fontsize=11, fontweight="bold")
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {OUT_FIG.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("Script 12: ONI ENSO — NOAA CPC")
    log.info("=" * 70)

    df = download_oni()
    df = classify_events(df)

    # Guardar CSV
    df.to_csv(OUT_CSV, index=False)
    log.info(f"  -> {OUT_CSV.name}  ({len(df)} meses, {df['year'].min()}-{df['year'].max()})")

    plot_oni(df)

    # Resumen eventos históricos relevantes
    log.info("\nEventos El Niño fuertes en período de interés (1981-2026):")
    fuertes = df[(df["event"]=="El Niño") & (df["intensity"]=="Fuerte") &
                 df["year"].between(1981,2026)]
    for yr in sorted(fuertes["year"].unique()):
        pk = fuertes[fuertes["year"]==yr]["oni"].max()
        log.info(f"  {yr}: ONI pico = {pk:.2f}°C")

    log.info("\n" + "=" * 70)
    log.info("=== DONE: ONI completado ===")
    log.info(f"  CSV : {OUT_CSV.name}  ({len(df)} filas)")
    log.info(f"  Fig : {OUT_FIG.name}")
    log.info("Siguiente: Scripts 08-09 (forzantes 2021-2026 + GR4J) cuando haya red")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
