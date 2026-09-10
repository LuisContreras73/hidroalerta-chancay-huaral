from pathlib import Path
gee = Path("data/raw/gee")
d6  = Path("data/processed")

print("=== GEE disponible para D7 ===")
sources = {
    "CHIRPS":    ("chirps",     "chirps_precip_*.tif"),
    "Snow":      ("snow",       "mod10a1_snow_*.tif"),
    "NDVI/EVI":  ("vegetation", "mod13q1_ndvi_*.tif"),
    "LSWI":      ("vegetation", "mod09ga_lswi_*.tif"),
    "LST":       ("thermal",    "mod11a1_lst_*.tif"),
    "ET/PET":    ("et",         "mod16a2_et_pet_*.tif"),
    "SMAP":      ("smap",       "smap_soil_*.tif"),
    "S1 SAR":    ("sentinel1",  "s1_vv_vh_ratio_*.tif"),
    "S2":        ("sentinel2",  "s2_ndvi_ndwi_*.tif"),
    "Landsat":   ("water_land", "landsat_ndvi_ndwi_*250m.tif"),
    "JRC 30m":   ("water_land", "jrc_water_????_30m.tif"),
    "ESA LC":    ("land_cover", "esa_worldcover*.tif"),
    "MCD12Q1":   ("land_cover", "mcd12q1_igbp_*.tif"),
}

for name, (folder, pat) in sources.items():
    files = sorted((gee / folder).glob(pat))
    if not files:
        print(f"  {name:<14} 0 archivos")
        continue
    sizes = sum(f.stat().st_size for f in files)
    years = []
    for f in files:
        for part in f.stem.split("_"):
            if part.isdigit() and len(part) == 4 and 1980 < int(part) < 2030:
                years.append(int(part)); break
    yr_range = f"{min(years)}-{max(years)}" if years else "?"
    print(f"  {name:<14} {len(files):>4} files  {sizes/1e6:>7.1f} MB  {yr_range}")

print()
print("=== D6 base ===")
for f in sorted(d6.glob("D6_multientity*.csv")):
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
era5 = list(Path("data/raw/era5/daily/parts").glob("*.nc"))
print(f"  ERA5 parts: {len(era5)} archivos")
