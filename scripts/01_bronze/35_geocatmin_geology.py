"""
Script 35: GEOCATMIN Geology Static Covariates
HidroAlerta Chancay-Huaral

Queries INGEMMET GEOCATMIN ArcGIS REST API (Geología 1:100,000) for each
sub-cuenca and computes area-weighted lithological class percentages compatible
with HydroATLAS lith_pc scheme (12 classes + karst indicator).

Source: https://geocatmin.ingemmet.gob.pe/arcgis/rest/services/
        SERV_GEOLOGIA_100K_INTEGRADA/MapServer/6 ("Geología")

Output: data/silver/S6_geocatmin_geology.csv
        Columns: entity_id + 12 lith_pct + dom_lith + karst_pct + n_units
"""

import json
import logging
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
GEOJSON = ROOT / "outputs" / "subcuencas_wgs84_simplified.geojson"
OUT_DIR = ROOT / "data" / "silver"
OUT_CSV = OUT_DIR / "S6_geocatmin_geology.csv"

# ── GEOCATMIN REST endpoint ───────────────────────────────────────────────────
GEOCATMIN_BASE = (
    "https://geocatmin.ingemmet.gob.pe/arcgis/rest/services"
    "/SERV_GEOLOGIA_100K_INTEGRADA/MapServer/6/query"
)
REQUEST_TIMEOUT = 60   # seconds per tile request
MAX_RETRIES     = 3
RETRY_DELAY     = 5    # seconds between retries

# ── WGS84 ↔ Web Mercator helpers (no pyproj needed) ─────────────────────────
def _lon_to_x(lon: float) -> float:
    return lon * 20037508.34 / 180.0

def _lat_to_y(lat: float) -> float:
    return math.log(math.tan((90 + lat) * math.pi / 360.0)) / math.pi * 20037508.34

# ── HydroATLAS 12-class lithological scheme ───────────────────────────────────
# Based on GLHYMPS / Hartmann & Moosdorf 2012 + HydroATLAS attribute guide
LITH_CLASSES = {
    1:  "metamorphic",
    2:  "acid_plutonic",
    3:  "basic_plutonic",
    4:  "acid_volcanic",
    5:  "basic_volcanic",
    6:  "pyroclastic",
    7:  "carbonate",        # HIGH karst potential
    8:  "mixed_sedimentary",
    9:  "siliciclastic",
    10: "unconsolidated",
    11: "mixed_rocks",
    12: "water_ice",
}

# ── Lithological classification ───────────────────────────────────────────────
# Two-stage: (1) DESCRIP keyword matching (ground-truth rock names in Spanish),
# (2) NAME code fallback for known GEOCATMIN formation abbreviations.
# DESCRIP is the primary source since formation codes (Ki-ca, PN-c, etc.) are
# ambiguous abbreviations that can represent different lithologies depending on
# the specific local formation name.

def _classify_by_descrip(descrip: str) -> int:
    """Classify using rock-type keywords in the DESCRIP field (Spanish)."""
    d = descrip.lower()

    # Carbonate (check before siliciclastic — "caliza" may co-occur with "arenisca")
    if any(k in d for k in ("caliz", "calizas", "dolomit", "marmol", "mármol",
                             "calcar", "calcár")):
        # Mixed carbonate+clastic → lith8; pure carbonate → lith7
        if any(k in d for k in ("arenisca", "lutita", "limolita", "arcillita",
                                  "conglomer")):
            return 8
        return 7

    # Metamorphic
    if any(k in d for k in ("gneis", "esquisto", "filita", "pizarra metamorfica",
                             "cuarcita metamor", "anfibolit", "migmatit",
                             "metapelita", "metarenisca", "metasediment")):
        return 1

    # Acid plutonic
    if any(k in d for k in ("granodiorit", "granito", "tonalita", "monzogranit",
                             "sienita", "monzonit", "monzodiorit")):
        return 2

    # Basic plutonic
    if any(k in d for k in ("gabro", "gabbro", "diorit", "diabas", "dunita",
                             "piroxenit", "hornblendit", "peridotit", "norita")):
        return 3

    # Pyroclastic (before acid_volcanic — toba may be mentioned with andesita)
    if any(k in d for k in ("toba", "tobas", "piroclás", "ignimbrit", "lapilli",
                             "brecha volc", "aglomerad")):
        # If dominated by lava mention too → keep pyroclastic
        return 6

    # Acid volcanic
    if any(k in d for k in ("andesita", "dacita", "riolita", "riodacita",
                             "traquita", "traquiandesit", "latita")):
        return 4

    # Basic volcanic
    if any(k in d for k in ("basalto", "basalt", "andesita basáltica")):
        return 5

    # Siliciclastic
    if any(k in d for k in ("arenisca", "cuarcita", "lutita", "limolita",
                             "arcillita", "pizarra", "pelita")):
        if "conglomer" in d:
            return 8  # mixed
        return 9

    # Unconsolidated / clastic cover
    if any(k in d for k in ("conglomerado", "grava", "arena", "limo", "arcilla",
                             "aluvial", "aluvion", "fluvial", "morrénico",
                             "morrena", "glaciofluvial", "coluvial", "dep",
                             "acumulaci")):
        return 10

    # Ice / water
    if any(k in d for k in ("glaciar", "nieve", "lago", "laguna", "hielo")):
        return 12

    return 11  # unknown


def _classify_by_name(name: str) -> int:
    """
    Fallback classifier using GEOCATMIN NAME code patterns.
    Only invoked when DESCRIP is unavailable or returns lith11.
    """
    n = name.strip().lower()

    # Quaternary unconsolidated (Q prefix always unconsolidated/glacial)
    if n.startswith("q"):
        if any(s in n for s in ("gl", "mo")):
            return 12  # glacial ice
        return 10

    suffix = n.split("-", 1)[-1].strip() if "-" in n else n

    # Known Peru Cretaceous limestone formations (Jumasha, Celendín, Chúlec, etc.)
    # These are identified by formation code suffix
    _carb_fmts = {"j", "ce", "chu", "ph", "at", "cz", "cal", "cali", "dol",
                  "dl", "ls", "cg", "pt"}  # pt=Pariatambo has calcareous
    # Check exact suffix tokens (split by / and ,)
    suffix_tokens = set(t.strip() for sep in ["/", ","] for t in suffix.split(sep))
    suffix_tokens.add(suffix)
    if suffix_tokens & _carb_fmts:
        return 7

    # Acid plutonic
    if suffix_tokens & {"gr", "gd", "to", "mo", "sy", "qm", "mg", "rh", "grd"}:
        return 2

    # Basic plutonic
    if suffix_tokens & {"gb", "di", "du", "hb", "no", "py", "dia", "gbdi",
                        "gbgd", "gbdi"}:
        return 3

    # Mixed plutonic/intrusive combos (like "bc/p-di,tn")
    if any(s in suffix for s in ("di,tn", "tn,gd", "gd,tn", "gb,di")):
        return 8

    # Pyroclastics
    if suffix_tokens & {"tp", "to", "br", "ag", "tob", "igb", "igt"}:
        if not any(s in suffix for s in ("an", "da", "ri")):
            return 6

    # Acid volcanic — andesita codes often in suffix: "an", "da", "ri"
    # Also PN-c, Nm-c etc. where "c" = Calipuy/other volcanic group
    if suffix_tokens & {"an", "da", "ri", "ry", "tr", "vc", "va"}:
        return 4
    if suffix in ("an", "da", "ri", "ry", "tr", "vc"):
        return 4
    # "c" suffix on Neogene/Paleogene = Calipuy volcanic (acid volcanic)
    if suffix == "c" and any(n.startswith(p) for p in ("nm", "np", "pn", "n")):
        return 4

    # Basic volcanic
    if suffix_tokens & {"ba", "bv", "db", "do"}:
        return 5

    # Siliciclastic
    if suffix_tokens & {"ar", "qu", "cu", "pi", "pe", "li", "fl", "s", "f",
                        "chi", "chu", "oy"}:
        if suffix not in {"chu"}:  # Chúlec is calcareous → handled above
            return 9

    # Mixed sedimentary
    if any(s in suffix for s in ("lu", "mu", "ms", "bc", "sr")):
        return 8

    # Metamorphic
    if suffix_tokens & {"mc", "mi", "gn", "fi", "es", "ml", "am", "mt", "mf"}:
        return 1

    # Water/ice
    if any(s in n for s in ("gl", "nv", "lag", "agua")):
        return 12

    return 11  # unknown/mixed


def classify_geocatmin(name: str, descrip: str = "") -> int:
    """
    Map a GEOCATMIN geological unit to HydroATLAS lith_class 1-12.
    Uses DESCRIP (rock-type keywords in Spanish) as primary source.
    Falls back to NAME-code pattern matching.
    Quaternary prefix always returns lith10 regardless of DESCRIP.
    """
    if not name or not isinstance(name, str):
        name = ""
    if not descrip or not isinstance(descrip, str):
        descrip = ""

    n = name.strip().lower()

    # Quaternary always unconsolidated (may be glacial/morenic but still lith10)
    if n.startswith("q"):
        if "gl" in n or ("mor" in descrip.lower()) or ("nieve" in descrip.lower()):
            return 12
        return 10

    # Try DESCRIP first if available
    if descrip.strip():
        cls = _classify_by_descrip(descrip)
        if cls != 11:
            return cls

    # Fall back to NAME-based if DESCRIP unclear or unavailable
    return _classify_by_name(name)


# ── ArcGIS REST query helper ──────────────────────────────────────────────────
def query_geocatmin_layer(polygon, out_fields="NAME,DESCRIP,UNIDAD", max_tries=MAX_RETRIES):
    """
    Spatial query to GEOCATMIN Layer 6 (Web Mercator, EPSG:3857) intersecting
    a WGS84 polygon. Returns GeoDataFrame in EPSG:4326 or None on failure.

    The service requires:
    - Geometry in Web Mercator (WKID 102100)
    - No resultRecordCount (pagination not supported; maxRecordCount=1000)
    - outSR=4326 to get responses back in WGS84
    """
    minx, miny, maxx, maxy = polygon.bounds  # WGS84
    # Convert bbox to Web Mercator for the spatial query
    geom_bbox = {
        "xmin": _lon_to_x(minx), "ymin": _lat_to_y(miny),
        "xmax": _lon_to_x(maxx), "ymax": _lat_to_y(maxy),
        "spatialReference": {"wkid": 102100}
    }

    params = {
        "where": "1=1",
        "geometry": json.dumps(geom_bbox),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.get(GEOCATMIN_BASE, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                log.warning(f"  GEOCATMIN API error: {data['error']}")
                return None

            features = data.get("features", [])
            if not features:
                log.warning("  No features returned from GEOCATMIN.")
                return None

            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            # Drop null geometries
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
            return gdf

        except Exception as e:
            log.warning(f"  Attempt {attempt}/{max_tries} failed: {e}")
            if attempt < max_tries:
                time.sleep(RETRY_DELAY)

    return None


# ── Per-sub-cuenca lithology computation ─────────────────────────────────────
def compute_geology(entity_id: str, polygon) -> dict:
    """
    Query GEOCATMIN, clip to polygon, compute area-weighted lith_class %.
    Returns a dict of static geology covariates.
    """
    result = {"entity_id": entity_id}
    for cls_id in LITH_CLASSES:
        result[f"lith{cls_id}_pct"] = np.nan

    result["dom_lith"] = "unknown"
    result["dom_lith_cls"] = np.nan
    result["karst_pct"] = np.nan
    result["n_units"] = 0
    result["query_ok"] = False

    log.info(f"  Querying GEOCATMIN for {entity_id}...")
    gdf = query_geocatmin_layer(polygon)

    if gdf is None or gdf.empty:
        log.error(f"  {entity_id}: no geology data retrieved.")
        return result

    # Clip geology polygons to sub-cuenca boundary
    sub_poly = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    try:
        gdf_clip = gpd.overlay(gdf, sub_poly, how="intersection", keep_geom_type=False)
    except Exception as e:
        log.warning(f"  {entity_id}: overlay failed ({e}), using intersection fallback")
        # Fallback: clip each geometry individually
        clipped = []
        for _, row in gdf.iterrows():
            try:
                geom_clipped = row.geometry.intersection(polygon)
                if not geom_clipped.is_empty:
                    r = row.copy()
                    r.geometry = geom_clipped
                    clipped.append(r)
            except Exception:
                pass
        if not clipped:
            log.error(f"  {entity_id}: fallback clip produced no features.")
            return result
        gdf_clip = gpd.GeoDataFrame(clipped, crs="EPSG:4326")

    if gdf_clip.empty:
        log.error(f"  {entity_id}: empty after clip.")
        return result

    # Project to UTM Zone 18S for area computation (Peru)
    gdf_utm = gdf_clip.to_crs("EPSG:32718")
    gdf_utm["area_m2"] = gdf_utm.geometry.area
    total_area = gdf_utm["area_m2"].sum()

    if total_area <= 0:
        log.error(f"  {entity_id}: total area is zero after clip.")
        return result

    # Assign lith_class to each clipped polygon
    name_col = "NAME" if "NAME" in gdf_utm.columns else (
        "name" if "name" in gdf_utm.columns else None
    )
    if name_col is None:
        log.warning(f"  {entity_id}: NAME field not found. Columns: {list(gdf_utm.columns)}")
        for alt in ["UNIDAD", "DESCRIP", "NOMBRE", "CODIGO", "LITOLOGIA"]:
            if alt in gdf_utm.columns:
                name_col = alt
                log.info(f"  Using '{alt}' as name column.")
                break

    if name_col is None:
        log.error(f"  {entity_id}: no usable name column found.")
        return result

    descrip_col = "DESCRIP" if "DESCRIP" in gdf_utm.columns else (
        "descrip" if "descrip" in gdf_utm.columns else None
    )

    def _classify_row(row):
        nm = row[name_col] if name_col else ""
        desc = row[descrip_col] if descrip_col else ""
        return classify_geocatmin(nm or "", desc or "")

    gdf_utm["lith_cls"] = gdf_utm.apply(_classify_row, axis=1)
    gdf_utm["dom_name"] = gdf_utm[name_col].fillna("unknown")

    # Area-weighted percentages per class
    area_by_cls = gdf_utm.groupby("lith_cls")["area_m2"].sum()
    pct_by_cls  = (area_by_cls / total_area * 100).round(2)

    for cls_id in LITH_CLASSES:
        result[f"lith{cls_id}_pct"] = float(pct_by_cls.get(cls_id, 0.0))

    # Dominant lithological unit (by area)
    idx_max = gdf_utm.groupby("dom_name")["area_m2"].sum().idxmax()
    result["dom_lith"]     = idx_max
    result["dom_lith_cls"] = int(gdf_utm.loc[gdf_utm["dom_name"] == idx_max, "lith_cls"].iloc[0])

    # Karst percentage = carbonate class (7)
    result["karst_pct"] = result["lith7_pct"]

    result["n_units"]  = int(gdf_utm[name_col].nunique())
    result["query_ok"] = True

    log.info(
        f"  {entity_id}: OK — {result['n_units']} units, "
        f"dom={result['dom_lith']} (cls {result['dom_lith_cls']}), "
        f"karst={result['karst_pct']:.1f}%"
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading sub-cuenca geometries...")
    subcuencas = gpd.read_file(GEOJSON)
    log.info(f"  {len(subcuencas)} sub-cuencas loaded.")

    records = []
    for _, row in subcuencas.iterrows():
        entity_id = row["entity_id"]
        polygon   = row.geometry
        rec = compute_geology(entity_id, polygon)
        rec["nombre"]  = row.get("nombre", "")
        rec["area_km2"] = row.get("area_km2", np.nan)
        records.append(rec)
        time.sleep(1)  # polite delay between requests

    df = pd.DataFrame(records)

    # Reorder columns
    cols_order = ["entity_id", "nombre", "area_km2"] + \
                 [f"lith{i}_pct" for i in range(1, 13)] + \
                 ["dom_lith", "dom_lith_cls", "karst_pct", "n_units", "query_ok"]
    df = df[[c for c in cols_order if c in df.columns]]

    df.to_csv(OUT_CSV, index=False)
    log.info(f"Saved → {OUT_CSV}")

    # Summary table
    print("\n" + "=" * 70)
    print("GEOCATMIN Geology — Summary by Sub-cuenca")
    print("=" * 70)
    summary_cols = ["entity_id", "dom_lith", "dom_lith_cls",
                    "karst_pct", "lith1_pct", "lith2_pct", "lith3_pct",
                    "lith9_pct", "lith10_pct", "n_units", "query_ok"]
    print(df[[c for c in summary_cols if c in df.columns]].to_string(index=False))
    print("\nLithological class legend:")
    for k, v in LITH_CLASSES.items():
        print(f"  lith{k:2d}: {v}")

    failed = df[~df["query_ok"]] if "query_ok" in df.columns else pd.DataFrame()
    if not failed.empty:
        log.warning(f"\n{len(failed)} sub-cuencas failed: {list(failed['entity_id'])}")
        log.warning("Re-run the script — GEOCATMIN can be flaky on first attempt.")


if __name__ == "__main__":
    main()
