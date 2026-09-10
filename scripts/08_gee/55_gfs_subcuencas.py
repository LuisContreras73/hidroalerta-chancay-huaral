#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 55 — GFS de precipitación pronosticada GRILLADO por SUB-CUENCA (distribuido).

Extiende el 54 (media de cuenca, lumped) a las 9 sub-cuencas del D7 vía reduceRegions.
Objetivo: captar DÓNDE se pronostica la lluvia (cabeceras andinas vs cuenca baja) →
forzante futura DISTRIBUIDA para el decoder = routing espacial (la versión legítima y con
historia del "upstream" que resultó redundante). Sigue siendo forecast (init×lead) y honesto
(GFS archivado = lo disponible operacionalmente en cada init).

Mismo gotcha de régimen que el 54 (acumulada→ventanas 6h el 2019-11-07).

Capas (Regla 4):
  raw    data/raw/gee/gfs_grid/gfs_tp_grid_YYYY_MM.csv  (init, fhour, entity_id, tp_mm)
  bronze data/bronze/B12_gfs_daily_leads_subcuencas.csv (init_date, lead, entity_id, pr_gfs_mm)

Uso (.venv — earthengine-api):
    python scripts/08_gee/55_gfs_subcuencas.py --start 2016-07 --end 2025-12
    python scripts/08_gee/55_gfs_subcuencas.py --build
"""
import argparse, json, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAWDIR = ROOT / "data/raw/gee/gfs_grid"
BRONZE = ROOT / "data/bronze/B12_gfs_daily_leads_subcuencas.csv"
SUBS_GJ = ROOT / "outputs/subcuencas_wgs84_simplified.geojson"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gfs_grid")

H_MAX = 14
HOURS = list(range(6, 24 * H_MAX + 1, 6))
REGIME_CHANGE = pd.Timestamp("2019-11-07")
SUBS = None


def init_gee():
    global SUBS
    import ee
    ee.Initialize(project="ana-chancay-huaral")
    gj = json.loads(SUBS_GJ.read_text(encoding="utf-8"))
    feats = [ee.Feature(ee.Geometry(f["geometry"]), {"entity_id": f["properties"]["entity_id"]})
             for f in gj["features"]]
    SUBS = ee.FeatureCollection(feats)


def extraer_mes(anio, mes, out_csv):
    """reduceRegions sobre las 9 sub-cuencas, TROCEADO por bloques de inits (límite GEE
    de 5000 elementos por getInfo): 9 subs × 56 fhours × ≤8 inits ≈ 4000 < 5000."""
    import ee
    t0 = pd.Timestamp(f"{anio}-{mes:02d}-01"); t1 = t0 + pd.offsets.MonthBegin(1)
    days = list(pd.date_range(t0, t1, freq="D", inclusive="left"))
    parts = []
    for c in range(0, len(days), 8):
        chunk = days[c:c + 8]
        inits_ms = [int(d.value // 10**6) for d in chunk]
        col = (ee.ImageCollection("NOAA/GFS0P25")
               .filterDate(chunk[0].strftime("%Y-%m-%d"), (chunk[-1] + pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
               .filter(ee.Filter.inList("creation_time", inits_ms))
               .filter(ee.Filter.inList("forecast_hours", HOURS))
               .select("total_precipitation_surface"))

        def per_img(img):
            fc = img.reduceRegions(collection=SUBS, reducer=ee.Reducer.mean(), scale=27830)
            return fc.map(lambda ft: ft.set("init_ms", img.get("creation_time"),
                                            "fhour", img.get("forecast_hours")))
        out = ee.FeatureCollection(col.map(per_img)).flatten()
        info = out.getInfo()["features"]
        if info:
            parts.append(pd.DataFrame([f["properties"] for f in info]))
    if not parts:
        raise ValueError(f"GFS grid {anio}-{mes:02d}: vacío")
    df = pd.concat(parts, ignore_index=True).rename(columns={"mean": "tp_mm"})
    df["init"] = pd.to_datetime(df["init_ms"], unit="ms")
    df = df[["init", "fhour", "entity_id", "tp_mm"]].sort_values(["init", "entity_id", "fhour"])
    out_csv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)
    log.info(f"  {out_csv.name}: {df['init'].nunique()} inits × {df['entity_id'].nunique()} subs "
             f"× {df['fhour'].nunique()} pasos ({len(df)} filas)")


def construir_bronze():
    files = sorted(RAWDIR.glob("gfs_tp_grid_*.csv"))
    if not files:
        raise FileNotFoundError("No hay raw grid — corre la extracción primero")
    df = pd.concat([pd.read_csv(f, parse_dates=["init"]) for f in files], ignore_index=True)
    df = df.drop_duplicates(["init", "entity_id", "fhour"]).dropna(subset=["tp_mm"])
    h24 = list(range(24, 24 * H_MAX + 1, 24)); out = []
    for (init, ent), g in df.groupby(["init", "entity_id"]):
        tp = g.set_index("fhour")["tp_mm"].reindex(HOURS)
        if init < REGIME_CHANGE:
            t24 = tp.loc[h24]
            if t24.isna().any():
                continue
            dia = np.diff(np.concatenate([[0.0], t24.values]))
        else:
            if tp.isna().any():
                continue
            dia = tp.values.reshape(H_MAX, 4).sum(axis=1)
        dia = np.clip(dia, 0.0, None)
        for L in range(1, H_MAX + 1):
            out.append((init.normalize(), L, ent, round(float(dia[L - 1]), 3)))
    b = pd.DataFrame(out, columns=["init_date", "lead", "entity_id", "pr_gfs_mm"])
    b.to_csv(BRONZE, index=False)
    log.info(f"B12: {b['init_date'].nunique()} inits × {b['entity_id'].nunique()} subs · leads 1-{H_MAX} "
             f"→ {BRONZE.name} ({BRONZE.stat().st_size/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--build", action="store_true")
    ap.add_argument("--test", action="store_true", help="solo 2024-03 para validar")
    a = ap.parse_args()
    if a.build:
        construir_bronze(); return
    init_gee()
    if a.test:
        extraer_mes(2024, 3, RAWDIR / "gfs_tp_grid_2024_03.csv"); return
    s = pd.Timestamp(a.start + "-01"); e = pd.Timestamp(a.end + "-01")
    for d in pd.date_range(s, e, freq="MS"):
        f = RAWDIR / f"gfs_tp_grid_{d.year}_{d.month:02d}.csv"
        if f.exists():
            log.info(f"  skip {f.name}"); continue
        try:
            extraer_mes(d.year, d.month, f)
        except Exception as ex:
            log.warning(f"  {d.year}-{d.month:02d}: {ex}")


if __name__ == "__main__":
    main()
