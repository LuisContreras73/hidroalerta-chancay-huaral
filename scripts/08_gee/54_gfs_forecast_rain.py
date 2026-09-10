#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 54 — Pronósticos GFS de precipitación (archivo operativo) para la cuenca.

Objetivo (D0xx-GFS): darle al decoder del RA-TFT la lluvia PRONOSTICADA sobre el
horizonte (1–14 d) en vez de nada (el publicado usa solo calendario futuro) y sin
leakage: GFS archivado = lo que habría estado disponible operacionalmente cada día.

Fuente: GEE `NOAA/GFS0P25` (0.25°, archivo desde 2015-07-01). Corrida 00Z.
Banda: total_precipitation_surface (kg/m² = mm).

GOTCHA CRÍTICO del producto (documentado en el catálogo GEE):
  - Inits ANTES de 2019-11-07: la banda es ACUMULADA desde el inicio del pronóstico
    (monótona) → lluvia del día-lead L = TP(24·L) − TP(24·(L−1)).
  - Inits DESDE 2019-11-07: la banda se REINICIA cada 6 h (cada valor = acumulado
    de su ventana de 6 h) → día-lead L = Σ TP en {24(L−1)+6, +12, +18, +24}.
  Por eso se extraen SIEMPRE los múltiplos de 6 h (6..336) y el modo se decide
  por fecha del init al construir el bronze.

Capas (Regla 4: raw inmutable):
  raw    data/raw/gee/gfs/gfs_tp_YYYY_MM.csv    (init, forecast_hours, tp_mm)
  bronze data/bronze/B11_gfs_daily_leads.csv    (init_date, lead 1..14, pr_gfs_mm)

Uso (.venv313):
    python scripts/08_gee/54_gfs_forecast_rain.py --start 2024-01 --end 2025-12
    python scripts/08_gee/54_gfs_forecast_rain.py --build
    python scripts/08_gee/54_gfs_forecast_rain.py --validate
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAWDIR = ROOT / "data/raw/gee/gfs"
BRONZE = ROOT / "data/bronze/B11_gfs_daily_leads.csv"
CUENCA_GJ = Path("D:/ANA Concurso/hidroalerta-dashboard/data/cuenca_limite.geojson")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gfs")

H_MAX = 14
HOURS = list(range(6, 24 * H_MAX + 1, 6))          # 6..336 cada 6 h (56 pasos)
REGIME_CHANGE = pd.Timestamp("2019-11-07")          # cambio acumulada→ventanas 6 h

# Regla 16: nunca objetos ee.* a nivel módulo.
BASIN = None


def init_gee():
    global BASIN
    import ee
    ee.Initialize(project="ana-chancay-huaral")
    gj = json.loads(CUENCA_GJ.read_text(encoding="utf-8"))
    geom = gj["features"][0]["geometry"] if gj.get("type") == "FeatureCollection" else gj
    BASIN = ee.Geometry(geom)


def extraer_mes(anio: int, mes: int, out_csv: Path) -> None:
    """Media de cuenca de total_precipitation_surface para todos los inits 00Z del
    mes × forecast_hours múltiplos de 6 (≤336). Una sola colección mapeada
    server-side → DataFrame (sin descargas de imagen; solo reduceRegion)."""
    import ee
    t0 = pd.Timestamp(f"{anio}-{mes:02d}-01")
    t1 = t0 + pd.offsets.MonthBegin(1)
    # inits 00Z del mes como millis exactos (no dependemos de si system:time_start
    # apunta a creation_time o a forecast_time; filterDate solo poda, con holgura
    # de +15 d por si indexa por hora de pronóstico)
    inits_ms = [int(d.value // 10**6) for d in pd.date_range(t0, t1, freq="D", inclusive="left")]
    col = (ee.ImageCollection("NOAA/GFS0P25")
           .filterDate(t0.strftime("%Y-%m-%d"),
                       (t1 + pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
           .filter(ee.Filter.inList("creation_time", inits_ms))
           .filter(ee.Filter.inList("forecast_hours", HOURS))
           .select("total_precipitation_surface"))

    def to_feat(img):
        tp = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=BASIN,
                              scale=27830, bestEffort=True)
        return ee.Feature(None, {
            "init_ms": img.get("creation_time"),
            "fhour": img.get("forecast_hours"),
            "tp_mm": tp.get("total_precipitation_surface"),
        })

    fc = ee.FeatureCollection(col.map(to_feat))
    n = fc.size().getInfo()
    if n == 0:
        raise ValueError(f"GFS {anio}-{mes:02d}: colección vacía (¿fecha < 2015-07?)")
    rows = fc.getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in rows])
    df["init"] = pd.to_datetime(df["init_ms"], unit="ms")
    df = df[["init", "fhour", "tp_mm"]].sort_values(["init", "fhour"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"  {out_csv.name}: {df['init'].nunique()} inits × "
             f"{df['fhour'].nunique()} pasos ({len(df)} filas)")


def construir_bronze() -> None:
    """raw mensual → B11: lluvia diaria pronosticada por (init_date, lead 1..14),
    aplicando el régimen de acumulación correcto según la fecha del init."""
    files = sorted(RAWDIR.glob("gfs_tp_*.csv"))
    if not files:
        raise FileNotFoundError("No hay raw en data/raw/gee/gfs/ — corre la extracción primero")
    df = pd.concat([pd.read_csv(f, parse_dates=["init"]) for f in files], ignore_index=True)
    df = df.drop_duplicates(["init", "fhour"]).dropna(subset=["tp_mm"])
    out = []
    h24 = list(range(24, 24 * H_MAX + 1, 24))
    for init, g in df.groupby("init"):
        tp = g.set_index("fhour")["tp_mm"].reindex(HOURS)
        if init < REGIME_CHANGE:                   # acumulada desde el inicio:
            t24 = tp.loc[h24]                      # solo hacen falta los pasos de 24 h
            if t24.isna().any():
                continue
            dia = np.diff(np.concatenate([[0.0], t24.values]))
        else:                                      # ventanas de 6 h → suma de 4
            if tp.isna().any():                    # aquí sí se necesitan los 56 pasos
                continue
            v = tp.values.reshape(H_MAX, 4)        # (lead, 4 ventanas de 6 h)
            dia = v.sum(axis=1)
        dia = np.clip(dia, 0.0, None)              # tolerancia numérica del diff
        for L in range(1, H_MAX + 1):
            out.append((init.normalize(), L, round(float(dia[L - 1]), 3)))
    b = pd.DataFrame(out, columns=["init_date", "lead", "pr_gfs_mm"])
    b.to_csv(BRONZE, index=False)
    log.info(f"B11: {b['init_date'].nunique()} inits · leads 1–{H_MAX} → {BRONZE.name} "
             f"({BRONZE.stat().st_size/1024:.0f} KB)")


def validar() -> None:
    """GFS lead-L vs PISCOp v3 (media de cuenca, día objetivo): correlación, sesgo
    y acierto de día húmedo (>1 mm). El día objetivo de lead L es init_date+L
    (convención UTC del init 00Z; PISCO es día local — se acepta ±: se reporta
    también lead alineado −1 para inspección)."""
    b = pd.read_csv(BRONZE, parse_dates=["init_date"])
    pisco = pd.read_csv(ROOT / "data/bronze/B2_pisco_v3_basin_mean.csv",
                        parse_dates=["date"]).set_index("date").iloc[:, 0]
    log.info("lead |    r   r(-1d) | bias(mm/d) | acierto húmedo>1mm")
    for L in [1, 2, 3, 5, 7, 10, 14]:
        d = b[b["lead"] == L].copy()
        d["target"] = d["init_date"] + pd.Timedelta(days=L)
        obs = pisco.reindex(d["target"]).values
        obs_m1 = pisco.reindex(d["target"] - pd.Timedelta(days=1)).values
        m = np.isfinite(obs) & np.isfinite(d["pr_gfs_mm"].values)
        if m.sum() < 30:
            log.info(f"  {L:2d} | insuficiente ({m.sum()} días)")
            continue
        g, o = d["pr_gfs_mm"].values[m], obs[m]
        m1 = np.isfinite(obs_m1) & np.isfinite(d["pr_gfs_mm"].values)
        r = np.corrcoef(g, o)[0, 1]
        r1 = np.corrcoef(d["pr_gfs_mm"].values[m1], obs_m1[m1])[0, 1]
        bias = float(np.mean(g - o))
        hit = float(np.mean((g > 1) == (o > 1)))
        log.info(f"  {L:2d} | {r:5.2f}  {r1:5.2f} | {bias:+7.2f}    | {hit:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="YYYY-MM primer mes a extraer")
    ap.add_argument("--end", default=None, help="YYYY-MM último mes (incluido)")
    ap.add_argument("--build", action="store_true", help="raw → B11 bronze")
    ap.add_argument("--validate", action="store_true", help="B11 vs PISCOp v3")
    a = ap.parse_args()

    if a.start:
        init_gee()
        fin = a.end or a.start
        meses = pd.period_range(a.start, fin, freq="M")
        log.info(f"Extracción GFS 00Z: {len(meses)} meses ({meses[0]}→{meses[-1]})")
        for p in meses:
            out = RAWDIR / f"gfs_tp_{p.year}_{p.month:02d}.csv"
            if out.exists() and out.stat().st_size > 500:
                log.info(f"  {out.name}: ya existe, salto (raw inmutable)")
                continue
            extraer_mes(p.year, p.month, out)
    if a.build:
        construir_bronze()
    if a.validate:
        validar()
    if not (a.start or a.build or a.validate):
        ap.print_help()


if __name__ == "__main__":
    main()
