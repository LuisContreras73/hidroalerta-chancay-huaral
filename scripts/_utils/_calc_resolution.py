"""Calculo de proyeccion de tamano/tiempo para re-descarga GEE a resolucion nativa."""
import glob, os

GEE_LIMIT_MB = 50.0
MIN_PER_TASK  = (2, 4, 8)  # optimista, realista, pesimista

sources = [
    ("Sentinel-1",    "data/raw/gee/sentinel1",  "s1_*.tif",       500, 10),
    ("Sentinel-2",    "data/raw/gee/sentinel2",  "s2_*.tif",       500, 10),
    ("Landsat",       "data/raw/gee/water_land", "landsat_*.tif",  250, 30),
    ("JRC Water",     "data/raw/gee/water_land", "jrc_*.tif",      500, 30),
    ("ESA WorldCvr",  "data/raw/gee/land_cover", "esa_*.tif",      100, 10),
    ("MCD12Q1",       "data/raw/gee/land_cover", "mcd12q1_*.tif",  100, 10),
]

header = f"{'Fuente':<14} {'N':>4} {'MB actual':>10} {'Cur':>6} {'Tgt':>5} {'Factor':>8} {'GB proy':>8} {'Tiles/f':>8} {'Tasks':>7}"
print(header)
print("-" * len(header))

total_mb_now   = 0.0
total_gb_proj  = 0.0
total_tasks    = 0

rows = []
for name, folder, pat, cur_m, tgt_m in sources:
    tifs = sorted(glob.glob(f"{folder}/{pat}"))
    if not tifs:
        continue
    n       = len(tifs)
    mb_now  = sum(os.path.getsize(f) for f in tifs) / 1e6
    factor  = (cur_m / tgt_m) ** 2
    mb_proj = mb_now * factor
    gb_proj = mb_proj / 1000.0
    avg_mb_proj   = (mb_now / n) * factor
    tiles_per_file = max(1, int(avg_mb_proj / GEE_LIMIT_MB) + 1)
    tasks          = n * tiles_per_file

    total_mb_now  += mb_now
    total_gb_proj += gb_proj
    total_tasks   += tasks

    row = (name, n, mb_now, cur_m, tgt_m, factor, gb_proj, tiles_per_file, tasks)
    rows.append(row)
    print(f"{name:<14} {n:>4} {mb_now:>10.1f} {cur_m:>5}m {tgt_m:>4}m {factor:>8.0f}x {gb_proj:>8.1f} {tiles_per_file:>8} {tasks:>7}")

print("-" * len(header))
print(f"{'TOTAL':<14} {'':>4} {total_mb_now:>10.1f} {'':>6} {'':>5} {'':>8} {total_gb_proj:>8.1f} {'':>8} {total_tasks:>7}")

print()
print("Tiempo estimado:")
for mph in MIN_PER_TASK:
    h = total_tasks * mph / 60
    print(f"  {mph} min/task -> {h:.0f} h = {h/24:.1f} dias")

print()
print(f"Storage adicional: {total_gb_proj:.0f} GB  (actual: {total_mb_now/1000:.1f} GB)")
print(f"Storage total GEE: {(total_gb_proj + 5.91):.0f} GB")
