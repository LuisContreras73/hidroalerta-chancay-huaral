"""Calcula savings de clipear a la cuenca real vs bbox completo."""
import geopandas as gpd
import numpy as np
import glob, os

# --- cuenca ---
basin = gpd.read_file(
    "data/raw/shapefiles/cuenca_chancay_huaral/limite/Cuenca_Chancay___Huaral.shp"
).to_crs(epsg=4326)
bounds = basin.total_bounds  # minx miny maxx maxy
dlon = bounds[2] - bounds[0]
dlat = bounds[3] - bounds[1]
lat_c = (bounds[1] + bounds[3]) / 2
km_lon = dlon * 111.32 * np.cos(np.radians(lat_c))
km_lat = dlat * 111.32
bbox_poly_km2 = km_lon * km_lat
BASIN_KM2 = 3062.62

print("=== Geometria cuenca WGS84 ===")
print(f"  Lon: {bounds[0]:.4f} -> {bounds[2]:.4f}  ({dlon:.3f} deg, {km_lon:.1f} km)")
print(f"  Lat: {bounds[1]:.4f} -> {bounds[3]:.4f}  ({dlat:.3f} deg, {km_lat:.1f} km)")
print(f"  Bbox poligono:     {bbox_poly_km2:.0f} km2")
print(f"  Area real cuenca:  {BASIN_KM2:.0f} km2  ({BASIN_KM2/bbox_poly_km2*100:.1f}% del bbox)")

# --- bbox actual descarga ---
full = dict(dlon=0.9, dlat=1.1)
km_lon_full = full["dlon"] * 111.32 * np.cos(np.radians(-11.35))
km_lat_full = full["dlat"] * 111.32
full_km2 = km_lon_full * km_lat_full
print(f"\n=== Bbox descarga actual ===")
print(f"  {km_lon_full:.1f} km x {km_lat_full:.1f} km = {full_km2:.0f} km2")
print(f"  Cuenca / bbox_descarga: {BASIN_KM2/full_km2*100:.1f}%")
print(f"  Bbox_poligono / bbox_descarga: {bbox_poly_km2/full_km2*100:.1f}%")

# --- reduccion de pixeles ---
print("\n=== Pixeles bbox descarga actual vs clip a poligono cuenca ===")
print(f"{'Res':>6} | {'Px actual (full bbox)':>22} | {'Px clip bbox-poligono':>22} | {'Px clip exacto cuenca':>22} | {'Reduccion clip':>15}")
print("-" * 98)
for res_m in [10, 30, 50, 100, 250, 500]:
    # full bbox (312x268 a 500m -> escalar)
    scale_fac = 500 / res_m
    px_full = (312 * scale_fac) * (268 * scale_fac) / 1e6

    # bbox del poligono de cuenca
    px_poly_bbox_lon = km_lon * 1000 / res_m
    px_poly_bbox_lat = km_lat * 1000 / res_m
    px_poly_bbox = px_poly_bbox_lon * px_poly_bbox_lat / 1e6

    # pixeles exactamente dentro de la cuenca
    px_basin = BASIN_KM2 * 1e6 / (res_m * res_m) / 1e6

    reduccion = (1 - px_poly_bbox / px_full) * 100
    print(f"  {res_m:>4}m | {px_full:>20.1f}M | {px_poly_bbox:>20.1f}M | {px_basin:>20.1f}M | {reduccion:>13.0f}%")

# --- proyeccion de tamanos con clip ---
print("\n=== Proyeccion de tamano con clip a bbox poligono cuenca ===")
print("(Factor reduccion vs bbox actual + DEFLATE compression ~3x para NoData exterior)")
print()

sources = [
    ("Sentinel-1",   "data/raw/gee/sentinel1",  "s1_*.tif",       500, 10),
    ("Sentinel-2",   "data/raw/gee/sentinel2",  "s2_*.tif",       500, 10),
    ("Sentinel-1",   "data/raw/gee/sentinel1",  "s1_*.tif",       500, 30),
    ("Landsat",      "data/raw/gee/water_land", "landsat_*.tif",  250, 30),
    ("JRC Water",    "data/raw/gee/water_land", "jrc_*.tif",      500, 30),
    ("ESA+MCD12Q1",  "data/raw/gee/land_cover", "*.tif",          100, 10),
]

GEE_LIMIT_MB = 50.0

print(f"{'Fuente':<13} {'Cur':>5} {'Tgt':>5} | {'MB actual':>10} | {'Factor px':>10} | {'GB clip+DEFLATE':>15} | {'Tasks':>7} | {'Factible?':>10}")
print("-" * 90)

for name, folder, pat, cur_m, tgt_m in sources:
    tifs = sorted(glob.glob(f"{folder}/{pat}"))
    if not tifs:
        continue
    n = len(tifs)
    mb_now = sum(os.path.getsize(f) for f in tifs) / 1e6

    # factor de escala en pixels: resolucion + clip a bbox poligono
    res_factor = (cur_m / tgt_m) ** 2
    clip_factor = bbox_poly_km2 / full_km2      # bbox poligono vs bbox descarga
    deflate_factor = 1 / 3.0                    # DEFLATE compression extra por NoData (~3x)

    total_factor = res_factor * clip_factor * deflate_factor
    mb_proj = mb_now * total_factor
    gb_proj = mb_proj / 1000.0

    avg_mb_proj = (mb_now / n) * total_factor
    tiles = max(1, int(avg_mb_proj / GEE_LIMIT_MB) + 1)
    tasks = n * tiles

    factible = "SI" if gb_proj < 50 else ("BORDE" if gb_proj < 150 else "NO")
    print(f"{name:<13} {cur_m:>4}m {tgt_m:>4}m | {mb_now:>10.1f} | {total_factor:>10.1f}x | {gb_proj:>15.1f} | {tasks:>7} | {factible:>10}")
