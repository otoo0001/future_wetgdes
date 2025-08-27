#!/usr/bin/env python3
import os, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio.features
from affine import Affine

# ── Config ────────────────────────────────────────────────────────────────────
MASK_DIR     = os.environ.get("MASK_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
SCENARIO     = os.environ.get("SCENARIO")  # single scenario for this job
SCENARIOS    = [SCENARIO] if SCENARIO else os.environ.get(
    "SCENARIOS", "historical ssp126 ssp370 ssp585"
).split()

BIOME_SHP    = os.environ.get("BIOME_SHP", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
QA_DIR       = os.environ.get("QA_DIR", "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/")
OUT_DIR      = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_area")
LOG_DIR      = os.environ.get("LOG_DIR", f"{OUT_DIR}/logs")
TEST_MODE    = os.environ.get("TEST", "0").lower() in ("1","true","yes","y")

# toggles and batching
APPLY_QA_MASK = os.environ.get("APPLY_QA_MASK", "1").lower() in ("1","true","yes","y")
TILE_Y        = int(os.environ.get("TILE_Y", "1024"))
TILE_X        = int(os.environ.get("TILE_X", "1024"))
TIME_BATCH    = int(os.environ.get("TIME_BATCH", "6"))  # number of timesteps per batch

ENGINE        = os.environ.get("XR_ENGINE", "h5netcdf")  # falls back if missing

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("gde_area_by_biome_realm_monthly_tiled")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
fh = logging.FileHandler(f"{LOG_DIR}/area_monthly_tiled.log"); fh.setFormatter(fmt); logger.addHandler(fh)

t0 = time.time()
logger.info(
    "START  MASK_DIR=%s  OUT_DIR=%s  SCENARIOS=%s  APPLY_QA_MASK=%s  TILE=%dx%d  TIME_BATCH=%d",
    MASK_DIR, OUT_DIR, " ".join(SCENARIOS), str(APPLY_QA_MASK), TILE_Y, TILE_X, TIME_BATCH
)

# ── Load QA grid and build mask ───────────────────────────────────────────────
qa_paths = sorted([os.path.join(QA_DIR, f) for f in os.listdir(QA_DIR) if f.endswith(".nc")])
if not qa_paths:
    raise FileNotFoundError(f"No .nc files in {QA_DIR}")
logger.info("Loading QA from %d files", len(qa_paths))

qa = xr.open_mfdataset(qa_paths, combine="by_coords", decode_times=False)

lat_name = "lat" if "lat" in qa.coords else ("latitude" if "latitude" in qa.coords else None)
lon_name = "lon" if "lon" in qa.coords else ("longitude" if "longitude" in qa.coords else None)
if lat_name is None or lon_name is None:
    raise RuntimeError("QA files must have lat or latitude and lon or longitude coords")

qa_lat = qa[lat_name].values
qa_lon = qa[lon_name].values
ny, nx = qa_lat.size, qa_lon.size

def _get(var):
    return qa[var].values if var in qa.data_vars else None

mount = _get("mountains_qa")
karst  = _get("karst_qa")
perma  = _get("permafrost_qa")
spin   = _get("spinup_qa")

static_ok = np.ones((ny, nx), dtype=bool)
if mount is not None: static_ok &= (mount == 0)
if karst  is not None: static_ok &= (karst  == 0)
if perma  is not None: static_ok &= (perma  == 0)
spin_ok = np.ones_like(static_ok, dtype=bool)
if spin is not None:
    spin_ok = ~np.isin(spin, [4, 6])
qa_mask = static_ok & spin_ok
logger.info("QA grid %s, good=%.2f%%", (ny, nx), 100.0 * qa_mask.mean())

# ── Pixel areas on QA grid in km^2 ────────────────────────────────────────────
R = 6_371_000.0
lat_res = float(abs(qa_lat[1] - qa_lat[0]))
lon_res = float(abs(qa_lon[1] - qa_lon[0]))
dlam = lon_res / 360.0
band = np.sin(np.deg2rad(qa_lat + lat_res/2.0)) - np.sin(np.deg2rad(qa_lat - lat_res/2.0))
area_per_lat = (2.0 * np.pi * R**2) * band * dlam / 1e6
pixel_area = np.repeat(area_per_lat[:, None], nx, axis=1).astype("float32")

# ── Rasterize biome x realm to QA grid ────────────────────────────────────────
logger.info("Loading biomes %s", BIOME_SHP)
gdf = gpd.read_file(BIOME_SHP).set_crs("EPSG:4326", allow_override=True)
if "BIOME" not in gdf.columns or "REALM" not in gdf.columns:
    raise RuntimeError("Shapefile must contain BIOME and REALM fields")
gdf = gdf.loc[gdf["BIOME"].between(1, 14), ["BIOME", "REALM", "geometry"]].copy()
gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)

unique_combos = sorted(gdf["BIOME_ID_REALM"].unique())
combo_to_code = {cmb: i + 1 for i, cmb in enumerate(unique_combos)}  # 0 reserved for outside
code_to_combo = {v: k for k, v in combo_to_code.items()}
n_codes = len(combo_to_code)
logger.info("Biome realm combos: %d", n_codes)

transform = Affine(lon_res, 0, qa_lon.min(), 0, -lat_res, qa_lat.max())
shapes = ((geom, combo_to_code[cmb]) for geom, cmb in zip(gdf.geometry, gdf["BIOME_ID_REALM"]))
codes_arr = rasterio.features.rasterize(
    shapes,
    out_shape=(ny, nx),
    transform=transform,
    fill=0,
    dtype="int32",
)
# if masking is off, keep the codes as is, otherwise zero out bad QA pixels
if APPLY_QA_MASK:
    codes_arr = np.where(qa_mask, codes_arr, 0).astype(np.int32)

# ── Scenario processing, tiled and batched ────────────────────────────────────
def process_scenario(scenario: str) -> pd.DataFrame:
    path_nc = os.path.join(MASK_DIR, f"wetGDE_{scenario}.nc")
    if not os.path.exists(path_nc):
        logger.warning("Missing file: %s (skipping)", path_nc)
        return pd.DataFrame()

    logger.info("Scenario=%s file=%s", scenario, path_nc)

    # open with minimal chunking, we will control memory by explicit batching
    try:
        ds = xr.open_dataset(path_nc, decode_times=True, engine=ENGINE)
    except Exception:
        ds = xr.open_dataset(path_nc, decode_times=True)

    var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
    wet = ds[var]

    # normalize coordinate names
    rename = {}
    if "lat" not in wet.dims and "latitude" in wet.dims:
        rename["latitude"] = "lat"
    if "lon" not in wet.dims and "longitude" in wet.dims:
        rename["longitude"] = "lon"
    if rename:
        wet = wet.rename(rename)

    if TEST_MODE:
        wet = wet.isel(time=slice(0, 1))
        logger.info("TEST MODE, first timestamp %s", pd.to_datetime(wet["time"].values[0]).isoformat())

    times = pd.to_datetime(wet["time"].values)
    nt = times.size

    # accumulator: sum of area by code for each time, include code 0 then drop later
    sums = np.zeros((nt, n_codes + 1), dtype=np.float64)

    # precompute equal-grid check
    same_lat = np.array_equal(wet["lat"].values, qa_lat)
    same_lon = np.array_equal(wet["lon"].values, qa_lon)

    # iterate in time batches to bound memory
    for t0_idx in range(0, nt, TIME_BATCH):
        t1_idx = min(t0_idx + TIME_BATCH, nt)
        wet_tb = wet.isel(time=slice(t0_idx, t1_idx))

        # regrid per batch, nearest to QA grid
        if same_lat and same_lon:
            wet_on_qa = wet_tb
        else:
            wet_on_qa = wet_tb.interp(lat=qa_lat, lon=qa_lon, method="nearest")

        # iterate spatial tiles, accumulate bincounts
        for y0 in range(0, ny, TILE_Y):
            y1 = min(y0 + TILE_Y, ny)
            for x0 in range(0, nx, TILE_X):
                x1 = min(x0 + TILE_X, nx)

                # fetch tile to memory
                w = wet_on_qa.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values  # shape: tb x ty x tx
                # to 0-1
                w = (w == 1).astype(np.float32)

                # apply QA mask if toggle is off, we still need 0 outside QA if chosen
                if APPLY_QA_MASK is False:
                    # nothing to do here, use all pixels
                    pass

                codes_tile = codes_arr[y0:y1, x0:x1]
                area_tile  = pixel_area[y0:y1, x0:x1]
                if APPLY_QA_MASK is False:
                    # when mask is off, codes outside biomes may be 0, which is fine
                    pass

                # accumulate per time step using bincount over codes
                # shapes: tb, ty*tx
                codes_flat = codes_tile.ravel()
                area_flat  = area_tile.ravel()
                for k in range(w.shape[0]):
                    weights = area_flat * w[k].ravel()
                    # minlength to include code 0..n_codes
                    bc = np.bincount(codes_flat, weights=weights, minlength=n_codes + 1)
                    sums[t0_idx + k, :bc.size] += bc

        logger.info("Accumulated %d/%d timesteps for scenario %s", t1_idx, nt, scenario)

    # build tidy DataFrame, drop code 0
    records = []
    for i, ts in enumerate(times):
        for code in range(1, n_codes + 1):
            area_km2 = float(sums[i, code])
            if area_km2 == 0.0:
                continue
            records.append((scenario, ts, code_to_combo[code], area_km2))

    if not records:
        return pd.DataFrame(columns=["scenario", "time", "BIOME_ID_REALM", "area_km2"])

    df = pd.DataFrame.from_records(records, columns=["scenario", "time", "BIOME_ID_REALM", "area_km2"])
    df = df.sort_values(["scenario", "time", "BIOME_ID_REALM"]).reset_index(drop=True)
    return df

# ── Run for all scenarios and write outputs ───────────────────────────────────
all_parts = []
for scen in SCENARIOS:
    df_scen = process_scenario(scen)
    if not df_scen.empty:
        all_parts.append(df_scen)

if all_parts:
    df = pd.concat(all_parts, ignore_index=True)
    if SCENARIO:
        out_path = os.path.join(OUT_DIR, f"gde_area_by_biome_realm_monthly_{SCENARIO}.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s rows=%d  APPLY_QA_MASK=%s", out_path, len(df), str(APPLY_QA_MASK))
    else:
        out_path = os.path.join(OUT_DIR, "gde_area_by_biome_realm_monthly_all.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s rows=%d  APPLY_QA_MASK=%s", out_path, len(df), str(APPLY_QA_MASK))
else:
    logger.warning("No scenario produced output  APPLY_QA_MASK=%s", str(APPLY_QA_MASK))

logger.info("DONE in %.1f s", time.time() - t0)




###########EXCLUDE AGRICULTURE FROM WETGDE MASKS##########
# Set APPLY_AG_MASK=1, provide AG_PATH with yearly agriculture area, set AG_AREA_VAR and AG_AREA_UNITS.

# Yearly agriculture is handled by reindex(... method="nearest", tolerance=AG_TOL_DAYS), so each monthly timestep uses the nearest year. Adjust AG_TOL_DAYS if your agriculture timestamps are mid year or end year.

# For permanent exclusion after first conversion since a baseline year, set AG_RULE=exclude_after_conversion and choose AG_BASE_YEAR.
#!/usr/bin/env python3
# import os, time, logging
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import xarray as xr
# import geopandas as gpd
# import rasterio.features
# from affine import Affine

# # ── Config ────────────────────────────────────────────────────────────────────
# MASK_DIR      = os.environ.get("MASK_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
# SCENARIO      = os.environ.get("SCENARIO")
# SCENARIOS     = [SCENARIO] if SCENARIO else os.environ.get(
#     "SCENARIOS", "historical ssp126 ssp370 ssp585"
# ).split()

# BIOME_SHP     = os.environ.get("BIOME_SHP", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
# QA_DIR        = os.environ.get("QA_DIR", "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/")
# OUT_DIR       = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_area")
# LOG_DIR       = os.environ.get("LOG_DIR", f"{OUT_DIR}/logs")
# TEST_MODE     = os.environ.get("TEST", "0").lower() in ("1","true","yes","y")

# # batching and tiles
# APPLY_QA_MASK = os.environ.get("APPLY_QA_MASK", "1").lower() in ("1","true","yes","y")
# TILE_Y        = int(os.environ.get("TILE_Y", "1024"))
# TILE_X        = int(os.environ.get("TILE_X", "1024"))
# TIME_BATCH    = int(os.environ.get("TIME_BATCH", "6"))

# ENGINE        = os.environ.get("XR_ENGINE", "h5netcdf")

# # agriculture toggle and settings
# APPLY_AG_MASK  = os.environ.get("APPLY_AG_MASK", "0").lower() in ("1","true","yes","y")
# AG_PATH        = os.environ.get("AG_PATH")                      # NetCDF with yearly agriculture area
# AG_AREA_VAR    = os.environ.get("AG_AREA_VAR", "ag_area")       # variable name that stores area
# AG_AREA_UNITS  = os.environ.get("AG_AREA_UNITS", "km2")         # km2 or m2
# AG_TOL_DAYS    = int(os.environ.get("AG_TOL_DAYS", "366"))      # nearest year tolerance
# AG_RULE        = os.environ.get("AG_RULE", "exclude_when_ag")   # exclude_when_ag or exclude_after_conversion
# AG_BASE_YEAR   = int(os.environ.get("AG_BASE_YEAR", "2000"))    # for exclude_after_conversion

# Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
# Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# # ── Logging ───────────────────────────────────────────────────────────────────
# logger = logging.getLogger("gde_area_by_biome_realm_monthly_tiled")
# logger.setLevel(logging.INFO)
# fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
# fh = logging.FileHandler(f"{LOG_DIR}/area_monthly_tiled.log"); fh.setFormatter(fmt); logger.addHandler(fh)

# t0 = time.time()
# logger.info(
#     "START  MASK_DIR=%s  OUT_DIR=%s  SCENARIOS=%s  APPLY_QA_MASK=%s  APPLY_AG_MASK=%s  TILE=%dx%d  TIME_BATCH=%d",
#     MASK_DIR, OUT_DIR, " ".join(SCENARIOS), str(APPLY_QA_MASK), str(APPLY_AG_MASK), TILE_Y, TILE_X, TIME_BATCH
# )

# # ── Load QA grid and build mask ───────────────────────────────────────────────
# qa_paths = sorted([os.path.join(QA_DIR, f) for f in os.listdir(QA_DIR) if f.endswith(".nc")])
# if not qa_paths:
#     raise FileNotFoundError(f"No .nc files in {QA_DIR}")
# logger.info("Loading QA from %d files", len(qa_paths))

# qa = xr.open_mfdataset(qa_paths, combine="by_coords", decode_times=False)

# lat_name = "lat" if "lat" in qa.coords else ("latitude" if "latitude" in qa.coords else None)
# lon_name = "lon" if "lon" in qa.coords else ("longitude" if "longitude" in qa.coords else None)
# if lat_name is None or lon_name is None:
#     raise RuntimeError("QA files must have lat or latitude and lon or longitude coords")

# qa_lat = qa[lat_name].values
# qa_lon = qa[lon_name].values
# ny, nx = qa_lat.size, qa_lon.size

# def _get(var):
#     return qa[var].values if var in qa.data_vars else None

# mount = _get("mountains_qa")
# karst  = _get("karst_qa")
# perma  = _get("permafrost_qa")
# spin   = _get("spinup_qa")

# static_ok = np.ones((ny, nx), dtype=bool)
# if mount is not None: static_ok &= (mount == 0)
# if karst  is not None: static_ok &= (karst  == 0)
# if perma  is not None: static_ok &= (perma  == 0)
# spin_ok = np.ones_like(static_ok, dtype=bool)
# if spin is not None:
#     spin_ok = ~np.isin(spin, [4, 6])
# qa_mask = static_ok & spin_ok
# logger.info("QA grid %s, good=%.2f%%", (ny, nx), 100.0 * qa_mask.mean())

# # ── Pixel areas on QA grid in km2 ─────────────────────────────────────────────
# R = 6_371_000.0
# lat_res = float(abs(qa_lat[1] - qa_lat[0]))
# lon_res = float(abs(qa_lon[1] - qa_lon[0]))
# dlam = lon_res / 360.0
# band = np.sin(np.deg2rad(qa_lat + lat_res/2.0)) - np.sin(np.deg2rad(qa_lat - lat_res/2.0))
# area_per_lat = (2.0 * np.pi * R**2) * band * dlam / 1e6
# pixel_area = np.repeat(area_per_lat[:, None], nx, axis=1).astype("float32")

# # ── Rasterize biome x realm to QA grid ────────────────────────────────────────
# logger.info("Loading biomes %s", BIOME_SHP)
# gdf = gpd.read_file(BIOME_SHP).set_crs("EPSG:4326", allow_override=True)
# if "BIOME" not in gdf.columns or "REALM" not in gdf.columns:
#     raise RuntimeError("Shapefile must contain BIOME and REALM fields")
# gdf = gdf.loc[gdf["BIOME"].between(1, 14), ["BIOME", "REALM", "geometry"]].copy()
# gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)

# unique_combos = sorted(gdf["BIOME_ID_REALM"].unique())
# combo_to_code = {cmb: i + 1 for i, cmb in enumerate(unique_combos)}  # 0 is outside
# code_to_combo = {v: k for k, v in combo_to_code.items()}
# n_codes = len(combo_to_code)
# logger.info("Biome realm combos: %d", n_codes)

# transform = Affine(lon_res, 0, qa_lon.min(), 0, -lat_res, qa_lat.max())
# shapes = ((geom, combo_to_code[cmb]) for geom, cmb in zip(gdf.geometry, gdf["BIOME_ID_REALM"]))
# codes_arr = rasterio.features.rasterize(
#     shapes,
#     out_shape=(ny, nx),
#     transform=transform,
#     fill=0,
#     dtype="int32",
# )
# if APPLY_QA_MASK:
#     codes_arr = np.where(qa_mask, codes_arr, 0).astype(np.int32)

# # ── Agriculture loader, converts yearly AREA to FRACTION on QA grid ───────────
# ag_frac_full = None
# if APPLY_AG_MASK:
#     if not AG_PATH or not os.path.exists(AG_PATH):
#         logger.warning("APPLY_AG_MASK=1 but AG_PATH missing, disabling ag exclusion")
#         APPLY_AG_MASK = False
#     else:
#         try:
#             ag_ds = xr.open_dataset(AG_PATH, decode_times=True, engine=ENGINE)
#         except Exception:
#             ag_ds = xr.open_dataset(AG_PATH, decode_times=True)

#         if AG_AREA_VAR not in ag_ds.data_vars:
#             logger.warning("AG_AREA_VAR %s not found in %s, disabling ag exclusion", AG_AREA_VAR, AG_PATH)
#             APPLY_AG_MASK = False
#         else:
#             ag = ag_ds[AG_AREA_VAR]
#             rename = {}
#             if "lat" not in ag.dims and "latitude" in ag.dims: rename["latitude"] = "lat"
#             if "lon" not in ag.dims and "longitude" in ag.dims: rename["longitude"] = "lon"
#             if rename: ag = ag.rename(rename)
#             if "time" not in ag.dims:
#                 logger.warning("Agriculture layer has no time, expanding as static")
#                 ag = ag.expand_dims(time=[np.datetime64("1900-01-01")])

#             # units to km2
#             if AG_AREA_UNITS.lower() == "m2":
#                 ag = ag / 1e6
#             elif AG_AREA_UNITS.lower() != "km2":
#                 logger.warning("Unknown AG_AREA_UNITS=%s, assuming km2", AG_AREA_UNITS)

#             # to QA grid
#             if not (np.array_equal(ag["lat"].values, qa_lat) and np.array_equal(ag["lon"].values, qa_lon)):
#                 ag = ag.interp(lat=qa_lat, lon=qa_lon, method="nearest")

#             # convert area to fraction by dividing by pixel_area
#             pa = xr.DataArray(pixel_area, dims=("lat","lon"), coords={"lat": qa_lat, "lon": qa_lon})
#             ag_frac_full = (ag / pa).fillna(0).clip(0, 1)  # time, lat, lon
#             logger.info("Loaded agriculture, will apply rule %s", AG_RULE)

# # ── Scenario processing, tiled and batched ────────────────────────────────────
# def process_scenario(scenario: str) -> pd.DataFrame:
#     path_nc = os.path.join(MASK_DIR, f"wetGDE_{scenario}.nc")
#     if not os.path.exists(path_nc):
#         logger.warning("Missing file: %s (skipping)", path_nc)
#         return pd.DataFrame()

#     logger.info("Scenario=%s file=%s", scenario, path_nc)

#     try:
#         ds = xr.open_dataset(path_nc, decode_times=True, engine=ENGINE)
#     except Exception:
#         ds = xr.open_dataset(path_nc, decode_times=True)

#     var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
#     wet = ds[var]

#     rename = {}
#     if "lat" not in wet.dims and "latitude" in wet.dims: rename["latitude"] = "lat"
#     if "lon" not in wet.dims and "longitude" in wet.dims: rename["longitude"] = "lon"
#     if rename: wet = wet.rename(rename)

#     if TEST_MODE:
#         wet = wet.isel(time=slice(0, 1))
#         logger.info("TEST MODE, first timestamp %s", pd.to_datetime(wet["time"].values[0]).isoformat())

#     times = pd.to_datetime(wet["time"].values)
#     nt = times.size

#     sums = np.zeros((nt, n_codes + 1), dtype=np.float64)

#     same_lat = np.array_equal(wet["lat"].values, qa_lat)
#     same_lon = np.array_equal(wet["lon"].values, qa_lon)

#     # agriculture fraction aligned to scenario time axis
#     ag_frac_on_time = None
#     if APPLY_AG_MASK and ag_frac_full is not None:
#         ag_frac_on_time = ag_frac_full.reindex(
#             time=times, method="nearest", tolerance=np.timedelta64(AG_TOL_DAYS, "D")
#         )
#         if AG_RULE == "exclude_after_conversion":
#             years = xr.DataArray(pd.to_datetime(ag_frac_on_time["time"].values).year, dims=("time",))
#             ag_since = ag_frac_on_time.where(years >= AG_BASE_YEAR, 0)
#             ever = (ag_since > 0).astype("uint8").cummax("time")
#             ag_frac_on_time = ever  # 0 or 1 thereafter

#     for t0_idx in range(0, nt, TIME_BATCH):
#         t1_idx = min(t0_idx + TIME_BATCH, nt)
#         wet_tb = wet.isel(time=slice(t0_idx, t1_idx))

#         if same_lat and same_lon:
#             wet_on_qa = wet_tb
#         else:
#             wet_on_qa = wet_tb.interp(lat=qa_lat, lon=qa_lon, method="nearest")

#         # slice agriculture for this batch and prepare interpolation function
#         if APPLY_AG_MASK and ag_frac_on_time is not None:
#             ag_tb = ag_frac_on_time.isel(time=slice(t0_idx, t1_idx))
#             if not (np.array_equal(ag_tb["lat"].values, qa_lat) and np.array_equal(ag_tb["lon"].values, qa_lon)):
#                 ag_tb = ag_tb.interp(lat=qa_lat, lon=qa_lon, method="nearest")
#         else:
#             ag_tb = None

#         for y0 in range(0, ny, TILE_Y):
#             y1 = min(y0 + TILE_Y, ny)
#             for x0 in range(0, nx, TILE_X):
#                 x1 = min(x0 + TILE_X, nx)

#                 w = wet_on_qa.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
#                 w = (w == 1).astype(np.float32)

#                 codes_tile = codes_arr[y0:y1, x0:x1]
#                 area_tile  = pixel_area[y0:y1, x0:x1]
#                 codes_flat = codes_tile.ravel()
#                 area_flat  = area_tile.ravel()

#                 if ag_tb is not None:
#                     ag_tile = ag_tb.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
#                 else:
#                     ag_tile = None

#                 for k in range(w.shape[0]):
#                     wet_flat = w[k].ravel()
#                     if ag_tile is not None:
#                         # yearly agriculture handled by reindex to nearest year
#                         eff_flat = wet_flat * (1.0 - np.clip(ag_tile[k].ravel(), 0.0, 1.0))
#                     else:
#                         eff_flat = wet_flat

#                     weights = area_flat * eff_flat
#                     bc = np.bincount(codes_flat, weights=weights, minlength=n_codes + 1)
#                     sums[t0_idx + k, :bc.size] += bc

#         logger.info("Accumulated %d/%d timesteps for scenario %s", t1_idx, nt, scenario)

#     records = []
#     for i, ts in enumerate(times):
#         for code in range(1, n_codes + 1):
#             area_km2 = float(sums[i, code])
#             if area_km2 == 0.0:
#                 continue
#             records.append((scenario, ts, code_to_combo[code], area_km2))

#     if not records:
#         return pd.DataFrame(columns=["scenario", "time", "BIOME_ID_REALM", "area_km2"])

#     df = pd.DataFrame.from_records(records, columns=["scenario", "time", "BIOME_ID_REALM", "area_km2"])
#     df = df.sort_values(["scenario", "time", "BIOME_ID_REALM"]).reset_index(drop=True)
#     return df

# # ── Run for all scenarios and write outputs ───────────────────────────────────
# all_parts = []
# for scen in SCENARIOS:
#     df_scen = process_scenario(scen)
#     if not df_scen.empty:
#         all_parts.append(df_scen)

# if all_parts:
#     df = pd.concat(all_parts, ignore_index=True)
#     if SCENARIO:
#         out_path = os.path.join(OUT_DIR, f"gde_area_by_biome_realm_monthly_{SCENARIO}.parquet")
#     else:
#         out_path = os.path.join(OUT_DIR, "gde_area_by_biome_realm_monthly_all.parquet")
#     df.to_parquet(out_path, index=False)
#     logger.info("Wrote %s rows=%d  APPLY_QA_MASK=%s  APPLY_AG_MASK=%s  AG_RULE=%s",
#                 out_path, len(df), str(APPLY_QA_MASK), str(APPLY_AG_MASK), AG_RULE)
# else:
#     logger.warning("No scenario produced output  APPLY_QA_MASK=%s  APPLY_AG_MASK=%s",
#                    str(APPLY_QA_MASK), str(APPLY_AG_MASK))

# logger.info("DONE in %.1f s", time.time() - t0)

#######without agriculture
#!/usr/bin/env python3
import os, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio.features
from affine import Affine

# ── Config ────────────────────────────────────────────────────────────────────
MASK_DIR      = os.environ.get("MASK_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
SCENARIO      = os.environ.get("SCENARIO")
SCENARIOS     = [SCENARIO] if SCENARIO else os.environ.get(
    "SCENARIOS", "historical ssp126 ssp370 ssp585"
).split()

BIOME_SHP     = os.environ.get("BIOME_SHP", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
QA_DIR        = os.environ.get("QA_DIR", "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/")
OUT_DIR       = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_area")
LOG_DIR       = os.environ.get("LOG_DIR", f"{OUT_DIR}/logs")
TEST_MODE     = os.environ.get("TEST", "0").lower() in ("1","true","yes","y")

# batching and tiles
APPLY_QA_MASK = os.environ.get("APPLY_QA_MASK", "1").lower() in ("1","true","yes","y")
TILE_Y        = int(os.environ.get("TILE_Y", "1024"))
TILE_X        = int(os.environ.get("TILE_X", "1024"))
TIME_BATCH    = int(os.environ.get("TIME_BATCH", "6"))
ENGINE        = os.environ.get("XR_ENGINE", "h5netcdf")

# agriculture, LUH2 states input
APPLY_AG_MASK  = os.environ.get("APPLY_AG_MASK", "1").lower() in ("1","true","yes","y")
LUH2_STATES    = os.environ.get("LUH2_STATES", "")  # space separated list of states.nc paths, can include extension file
AG_EXPERIMENTS = os.environ.get("AG_EXPERIMENTS", "crops crops_pasture").split()  # subset of {crops, crops_pasture}
AG_TOL_DAYS    = int(os.environ.get("AG_TOL_DAYS", "366"))
AG_RULE        = os.environ.get("AG_RULE", "exclude_when_ag")  # or exclude_after_conversion
AG_BASE_YEAR   = int(os.environ.get("AG_BASE_YEAR", "2000"))

# transient NetCDF
WRITE_NC        = os.environ.get("WRITE_NC", "1").lower() in ("1","true","yes","y")
NC_TRANSIENT    = os.environ.get("NC_TRANSIENT", "1").lower() in ("1","true","yes","y")
NC_ENGINE       = os.environ.get("NC_ENGINE", "h5netcdf")
NC_TMPDIR       = os.environ.get("NC_TMPDIR", "/scratch")
NC_FILENAME     = os.environ.get("NC_FILENAME", "gde_area_tmp.nc")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("gde_area_by_biome_realm_monthly_tiled")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
fh = logging.FileHandler(f"{LOG_DIR}/area_monthly_tiled.log"); fh.setFormatter(fmt); logger.addHandler(fh)

t0 = time.time()
logger.info(
    "START MASK_DIR=%s OUT_DIR=%s SCENARIOS=%s APPLY_QA_MASK=%s APPLY_AG_MASK=%s AG_EXPERIMENTS=%s",
    MASK_DIR, OUT_DIR, " ".join(SCENARIOS), APPLY_QA_MASK, APPLY_AG_MASK, " ".join(AG_EXPERIMENTS)
)

# ── QA grid and mask ─────────────────────────────────────────────────────────
qa_paths = sorted([os.path.join(QA_DIR, f) for f in os.listdir(QA_DIR) if f.endswith(".nc")])
if not qa_paths:
    raise FileNotFoundError(f"No .nc files in {QA_DIR}")
qa = xr.open_mfdataset(qa_paths, combine="by_coords", decode_times=False)

lat_name = "lat" if "lat" in qa.coords else ("latitude" if "latitude" in qa.coords else None)
lon_name = "lon" if "lon" in qa.coords else ("longitude" if "longitude" in qa.coords else None)
if lat_name is None or lon_name is None:
    raise RuntimeError("QA files must have lat or latitude and lon or longitude coords")

qa_lat = qa[lat_name].values
qa_lon = qa[lon_name].values
ny, nx = qa_lat.size, qa_lon.size

def _get(var):
    return qa[var].values if var in qa.data_vars else None

mount = _get("mountains_qa")
karst  = _get("karst_qa")
perma  = _get("permafrost_qa")
spin   = _get("spinup_qa")

static_ok = np.ones((ny, nx), dtype=bool)
if mount is not None: static_ok &= (mount == 0)
if karst  is not None: static_ok &= (karst  == 0)
if perma  is not None: static_ok &= (perma  == 0)
spin_ok = np.ones_like(static_ok, dtype=bool)
if spin is not None:
    spin_ok = ~np.isin(spin, [4, 6])
qa_mask = static_ok & spin_ok
logger.info("QA grid %s, good=%.2f%%", (ny, nx), 100.0 * qa_mask.mean())

# pixel areas in km2 on QA grid
R = 6_371_000.0
lat_res = float(abs(qa_lat[1] - qa_lat[0]))
lon_res = float(abs(qa_lon[1] - qa_lon[0]))
dlam = lon_res / 360.0
band = np.sin(np.deg2rad(qa_lat + lat_res/2.0)) - np.sin(np.deg2rad(qa_lat - lat_res/2.0))
area_per_lat = (2.0 * np.pi * R**2) * band * dlam / 1e6
pixel_area = np.repeat(area_per_lat[:, None], nx, axis=1).astype("float32")

# ── Biome x realm raster on QA grid ──────────────────────────────────────────
gdf = gpd.read_file(BIOME_SHP).set_crs("EPSG:4326", allow_override=True)
if "BIOME" not in gdf.columns or "REALM" not in gdf.columns:
    raise RuntimeError("Shapefile must contain BIOME and REALM fields")
gdf = gdf.loc[gdf["BIOME"].between(1, 14), ["BIOME", "REALM", "geometry"]].copy()
gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)

unique_combos = sorted(gdf["BIOME_ID_REALM"].unique())
combo_to_code = {cmb: i + 1 for i, cmb in enumerate(unique_combos)}
code_to_combo = {v: k for k, v in combo_to_code.items()}
n_codes = len(combo_to_code)
transform = Affine(lon_res, 0, qa_lon.min(), 0, -lat_res, qa_lat.max())
shapes = ((geom, combo_to_code[cmb]) for geom, cmb in zip(gdf.geometry, gdf["BIOME_ID_REALM"]))
codes_arr = rasterio.features.rasterize(shapes, out_shape=(ny, nx), transform=transform, fill=0, dtype="int32")
if APPLY_QA_MASK:
    codes_arr = np.where(qa_mask, codes_arr, 0).astype(np.int32)

# ── LUH2 loader, returns crops and pasture fractions ─────────────────────────
def load_luh2_crops_pastr(states_files, engine="h5netcdf"):
    if not states_files:
        return None, None
    ds = xr.open_mfdataset(states_files, combine="by_coords", decode_times=True, engine=engine)
    rename = {}
    if "lat" not in ds.dims and "latitude" in ds.dims: rename["latitude"] = "lat"
    if "lon" not in ds.dims and "longitude" in ds.dims: rename["longitude"] = "lon"
    if rename: ds = ds.rename(rename)
    needed = ["c3ann","c4ann","c3per","c4per","c3nfx","pastr"]
    missing = [v for v in needed if v not in ds.data_vars]
    if missing:
        raise KeyError(f"LUH2 states missing variables: {missing}")
    crops = ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]
    pastr = ds["pastr"]
    crops = crops.fillna(0).clip(0, 1)
    pastr = pastr.fillna(0).clip(0, 1)
    return crops, pastr

# load LUH2
crops_full, pastr_full = None, None
if APPLY_AG_MASK:
    state_files = [p for p in LUH2_STATES.split() if p]
    if not state_files:
        logger.warning("APPLY_AG_MASK=1 but LUH2_STATES not set, disabling agricultural exclusion")
        APPLY_AG_MASK = False
    else:
        crops_full, pastr_full = load_luh2_crops_pastr(state_files, engine=ENGINE)
        # align to QA grid once
        if not (np.array_equal(crops_full["lat"].values, qa_lat) and np.array_equal(crops_full["lon"].values, qa_lon)):
            crops_full = crops_full.interp(lat=qa_lat, lon=qa_lon, method="nearest")
            pastr_full = pastr_full.interp(lat=qa_lat, lon=qa_lon, method="nearest")

# ── Scenario processing with two ag experiments in one pass ───────────────────
def process_scenario(scenario: str) -> pd.DataFrame:
    path_nc = os.path.join(MASK_DIR, f"wetGDE_{scenario}.nc")
    if not os.path.exists(path_nc):
        logger.warning("Missing file: %s (skipping)", path_nc)
        return pd.DataFrame()

    ds = xr.open_dataset(path_nc, decode_times=True, engine=ENGINE)
    var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
    wet = ds[var]
    rename = {}
    if "lat" not in wet.dims and "latitude" in wet.dims: rename["latitude"] = "lat"
    if "lon" not in wet.dims and "longitude" in wet.dims: rename["longitude"] = "lon"
    if rename: wet = wet.rename(rename)
    if TEST_MODE:
        wet = wet.isel(time=slice(0, 1))

    times = pd.to_datetime(wet["time"].values)
    nt = times.size
    same_lat = np.array_equal(wet["lat"].values, qa_lat)
    same_lon = np.array_equal(wet["lon"].values, qa_lon)

    # sums per experiment
    exps = [e for e in AG_EXPERIMENTS if e in ("crops","crops_pasture")]
    if not exps:
        exps = ["crops","crops_pasture"]
    sums = {e: np.zeros((nt, n_codes + 1), dtype=np.float64) for e in exps}

    # prepare agriculture time alignment
    if APPLY_AG_MASK and crops_full is not None:
        crops_t = crops_full.reindex(time=times, method="nearest", tolerance=np.timedelta64(AG_TOL_DAYS, "D"))
        pastr_t = pastr_full.reindex(time=times, method="nearest", tolerance=np.timedelta64(AG_TOL_DAYS, "D"))
        if AG_RULE == "exclude_after_conversion":
            years = xr.DataArray(pd.to_datetime(crops_t["time"].values).year, dims=("time",))
            crops_since = crops_t.where(years >= AG_BASE_YEAR, 0)
            pastr_since = pastr_t.where(years >= AG_BASE_YEAR, 0)
            crops_t = (crops_since > 0).astype("uint8").cummax("time")
            pastr_t = (pastr_since > 0).astype("uint8").cummax("time")
    else:
        crops_t, pastr_t = None, None

    for t0_idx in range(0, nt, TIME_BATCH):
        t1_idx = min(t0_idx + TIME_BATCH, nt)
        wet_tb = wet.isel(time=slice(t0_idx, t1_idx))
        wet_on_qa = wet_tb if (same_lat and same_lon) else wet_tb.interp(lat=qa_lat, lon=qa_lon, method="nearest")

        # ag fractions for this batch
        if APPLY_AG_MASK and crops_t is not None:
            crops_tb = crops_t.isel(time=slice(t0_idx, t1_idx))
            pastr_tb = pastr_t.isel(time=slice(t0_idx, t1_idx))
        else:
            crops_tb, pastr_tb = None, None

        for y0 in range(0, ny, TILE_Y):
            y1 = min(y0 + TILE_Y, ny)
            for x0 in range(0, nx, TILE_X):
                x1 = min(x0 + TILE_X, nx)
                w = wet_on_qa.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
                w = (w == 1).astype(np.float32)

                codes_tile = codes_arr[y0:y1, x0:x1]
                area_tile  = pixel_area[y0:y1, x0:x1]
                codes_flat = codes_tile.ravel()
                area_flat  = area_tile.ravel()

                if crops_tb is not None:
                    crops_tile = crops_tb.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
                    pastr_tile = pastr_tb.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values

                for k in range(w.shape[0]):
                    wet_flat = w[k].ravel()

                    # build experiment specific effective wet fractions
                    eff = {}
                    if crops_tb is None:
                        for e in exps:
                            eff[e] = wet_flat
                    else:
                        crops_f = np.clip(crops_tile[k].ravel(), 0.0, 1.0)
                        pastr_f = np.clip(pastr_tile[k].ravel(), 0.0, 1.0)
                        if "crops" in exps:
                            eff["crops"] = wet_flat * (1.0 - crops_f)
                        if "crops_pasture" in exps:
                            ag_cp = np.clip(crops_f + pastr_f, 0.0, 1.0)
                            eff["crops_pasture"] = wet_flat * (1.0 - ag_cp)

                    for e in exps:
                        weights = area_flat * eff[e]
                        bc = np.bincount(codes_flat, weights=weights, minlength=n_codes + 1)
                        sums[e][t0_idx + k, :bc.size] += bc

        logger.info("Scenario %s, accumulated %d/%d timesteps", scenario, t1_idx, nt)

    # build tidy records with ag_exclusion label
    records = []
    for e in exps:
        sarr = sums[e]
        for i, ts in enumerate(times):
            for code in range(1, n_codes + 1):
                area_km2 = float(sarr[i, code])
                if area_km2 == 0.0:
                    continue
                records.append((scenario, ts, code_to_combo[code], e, area_km2))

    if not records:
        return pd.DataFrame(columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])

    df = pd.DataFrame.from_records(records, columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])
    return df.sort_values(["scenario","time","BIOME_ID_REALM","ag_exclusion"]).reset_index(drop=True)

# ── NetCDF writers ────────────────────────────────────────────────────────────
def _tmp_nc_path(name):
    if not WRITE_NC:
        return None
    if NC_TRANSIENT:
        import tempfile
        return tempfile.NamedTemporaryFile(prefix=name.rstrip(".nc")+"_", suffix=".nc",
                                           dir=NC_TMPDIR, delete=False).name
    return os.path.join(OUT_DIR, name)

def write_nc_cube(df_all: pd.DataFrame):
    scenarios = np.array(sorted(df_all["scenario"].unique()), dtype="object")
    biomes    = np.array(sorted(df_all["BIOME_ID_REALM"].unique()), dtype="object")
    exps      = np.array(sorted(df_all["ag_exclusion"].unique()), dtype="object")
    times     = np.array(sorted(pd.to_datetime(df_all["time"].unique())))
    idx_t = {t:i for i,t in enumerate(times)}
    idx_s = {s:i for i,s in enumerate(scenarios)}
    idx_b = {b:i for i,b in enumerate(biomes)}
    idx_e = {e:i for i,e in enumerate(exps)}

    arr = np.full((len(times), len(scenarios), len(biomes), len(exps)), np.nan, dtype="float32")
    for s, t, b, e, a in df_all[["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"]].itertuples(index=False):
        arr[idx_t[pd.to_datetime(t)], idx_s[s], idx_b[b], idx_e[e]] = a

    ds = xr.Dataset(
        {"gde_area_km2": (("time","scenario","biome_realm","ag_exclusion"), arr)},
        coords={"time": times, "scenario": scenarios, "biome_realm": biomes, "ag_exclusion": exps},
        attrs={"notes": "WetGDE area after QA and LUH2 agriculture exclusion, experiments in ag_exclusion"},
    )
    path_nc = _tmp_nc_path(NC_FILENAME)
    if path_nc:
        ds.to_netcdf(path_nc, engine=NC_ENGINE)
        logger.info("NetCDF written: %s", path_nc)
    return path_nc

# ── Run and write ─────────────────────────────────────────────────────────────
all_parts = []
for scen in SCENARIOS:
    df_scen = process_scenario(scen)
    if not df_scen.empty:
        all_parts.append(df_scen)

if all_parts:
    df = pd.concat(all_parts, ignore_index=True)

    # combined parquet with ag_exclusion column
    if SCENARIO:
        base = f"gde_area_by_biome_realm_monthly_{SCENARIO}"
    else:
        base = "gde_area_by_biome_realm_monthly_all"
    out_parq_all = os.path.join(OUT_DIR, f"{base}_with_ag_experiments.parquet")
    df.to_parquet(out_parq_all, index=False)
    logger.info("Parquet written: %s rows=%d", out_parq_all, len(df))

    # separate files per experiment
    for e in sorted(df["ag_exclusion"].unique()):
        dfe = df.loc[df["ag_exclusion"] == e].drop(columns=["ag_exclusion"])
        out_parq_e = os.path.join(OUT_DIR, f"{base}_excl_{e}.parquet")
        dfe.to_parquet(out_parq_e, index=False)
        logger.info("Parquet written: %s rows=%d", out_parq_e, len(dfe))

    # transient NetCDF with ag_exclusion dimension
    if WRITE_NC:
        write_nc_cube(df)
else:
    logger.warning("No scenario produced output")

logger.info("DONE in %.1f s", time.time() - t0)
