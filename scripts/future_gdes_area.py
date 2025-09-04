



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
# WetGDE area by biome×realm with three variants:
#   ag_exclusion = "none"            -> baseline, no agriculture exclusion
#   ag_exclusion = "crops"           -> exclude LUH2 cropland fraction
#   ag_exclusion = "crops_pasture"   -> exclude LUH2 cropland + managed pasture fraction
#!/usr/bin/env python3
# gde_area_by_biome_realm_monthly_tiled.py

# import os, re, time, logging, tempfile
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import xarray as xr
# import geopandas as gpd
# import rasterio.features
# from affine import Affine

# # ── env helpers ───────────────────────────────────────────────────────────────
# def _env_bool(k, default):
#     v = os.getenv(k)
#     if v is None:
#         return default
#     return str(v).strip().lower() in ("1", "true", "t", "yes", "y")

# def _env_int(k, default):
#     v = os.getenv(k)
#     return int(v) if v not in (None, "") else default

# def _env_str(k, default):
#     v = os.getenv(k)
#     return v if v not in (None, "") else default

# # ── CONFIG, uses env if provided ──────────────────────────────────────────────
# MASK_DIR   = _env_str("MASK_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
# OUT_DIR    = _env_str("OUT_DIR",  "/projects/prjs1578/futurewetgde/wetGDEs_area")
# LOG_DIR    = _env_str("LOG_DIR",  f"{OUT_DIR}/logs")

# if _env_str("SCENARIO", None):
#     SCENARIOS  = [_env_str("SCENARIO", "")]
# elif _env_str("SCENARIOS", None):
#     SCENARIOS  = [s.strip() for s in os.getenv("SCENARIOS").split(",") if s.strip()]
# else:
#     SCENARIOS  = ["ssp126", "ssp370", "ssp585"]

# BIOME_SHP  = _env_str("BIOME_SHP", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
# QA_DIR     = _env_str("QA_DIR",    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/")

# ENGINE         = _env_str("XR_ENGINE", "netcdf4")
# TILE_Y         = _env_int("TILE_Y", 1536)
# TILE_X         = _env_int("TILE_X", 1536)
# TIME_BATCH     = _env_int("TIME_BATCH", 12)
# WET_THRESHOLD  = _env_str("WET_THRESHOLD", "gt0")

# SMALL_TEST     = _env_bool("SMALL_TEST", False)
# APPLY_QA_MASK  = _env_bool("APPLY_QA_MASK", True)
# APPLY_AG_MASK  = _env_bool("APPLY_AG_MASK", True)
# WRITE_NC       = _env_bool("WRITE_NC", True)
# NC_TRANSIENT   = _env_bool("NC_TRANSIENT", True)

# AG_RULE        = _env_str("AG_RULE", "exclude_when_ag")
# AG_BASE_YEAR   = _env_int("AG_BASE_YEAR", 2000)

# LUH2_SSP_ROOT  = _env_str("LUH2_SSP_ROOT", "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp")
# LUH2_MAP = {
#     "historical": "Historic_Data/states.nc",
#     "ssp126":    "RCP2_6_SSP1_from_IMAGE/states.nc",
#     "ssp245":    "RCP4_5_SSP2_from_MESSAGE_GLOBIOM/states.nc",
#     "ssp370":    "RCP7_0_SSP3_from_AIM/states.nc",
#     "ssp434":    "RCP3_4_SSP4_from_GCAM/states.nc",
#     "ssp460":    "RCP6_0_SSP4_from_GCAM/states.nc",
#     "ssp585":    "RCP8_5_SSP5_from_REMIND_MAGPIE/states.nc",
# }

# NC_ENGINE    = "netcdf4"
# NC_TMPDIR    = _env_str("NC_TMPDIR", "/scratch")
# NC_FILENAME  = _env_str("NC_FILENAME", "gde_area_tmp.nc")

# # metadata for NetCDF
# NC_TITLE       = _env_str("NC_TITLE", "Global wetland groundwater dependent ecosystem area by biome×realm with agricultural exclusions")
# NC_INSTITUTION = _env_str("NC_INSTITUTION", "Utrecht University")
# NC_AUTHOR      = _env_str("NC_AUTHOR", "Nicole Gyakowah Otoo n.g.otoo@uu.nnl")
# NC_DESCRIPTION = _env_str("NC_DESCRIPTION",
#                           "Monthly wetland GDE area aggregated by WWF terrestrial biome × realm. "
#                           "Values are km^2 after applying QA masks and agricultural exclusion experiments: "
#                           "none, crops, crops+pasture. Computed from wetGDE occurence and LUH2 states.")

# # ── Prep dirs and logging ─────────────────────────────────────────────────────
# Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
# Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# logger = logging.getLogger("gde_area_by_biome_realm_monthly_tiled")
# logger.setLevel(logging.INFO)
# fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
# fh = logging.FileHandler(f"{LOG_DIR}/area_monthly_tiled.log"); fh.setFormatter(fmt); logger.addHandler(fh)

# t0 = time.time()
# logger.info(
#     "START MASK_DIR=%s OUT_DIR=%s SCENARIOS=%s SMALL_TEST=%s APPLY_QA_MASK=%s APPLY_AG_MASK=%s "
#     "WET_THRESHOLD=%s AG_RULE=%s AG_BASE_YEAR=%d LUH2_SSP_ROOT=%s XR_ENGINE=%s",
#     MASK_DIR, OUT_DIR, ",".join(SCENARIOS), SMALL_TEST, APPLY_QA_MASK, APPLY_AG_MASK,
#     WET_THRESHOLD, AG_RULE, AG_BASE_YEAR, LUH2_SSP_ROOT, ENGINE,
# )

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def _std(ds: xr.Dataset) -> xr.Dataset:
#     if "latitude" in ds.coords:  ds = ds.rename({"latitude": "lat"})
#     if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
#     return ds

# def open_qa_merged(qa_paths):
#     base = _std(xr.open_dataset(qa_paths[0], decode_times=False))
#     qa_lat = base["lat"].values
#     qa_lon = base["lon"].values
#     ny, nx = qa_lat.size, qa_lon.size
#     qa_vars = {}
#     for p in qa_paths:
#         ds = _std(xr.open_dataset(p, decode_times=False))
#         if (ds.sizes.get("lat") != ny) or (ds.sizes.get("lon") != nx) or \
#            (not np.array_equal(ds["lat"].values, qa_lat)) or \
#            (not np.array_equal(ds["lon"].values, qa_lon)):
#             ds = ds.interp(lat=qa_lat, lon=qa_lon, method="nearest")
#         for v in ds.data_vars:
#             dv = ds[v]
#             if "time" in dv.dims:
#                 dv = dv.isel(time=0, drop=True)
#             qa_vars[v] = dv
#     return xr.Dataset(qa_vars, coords={"lat": qa_lat, "lon": qa_lon})

# def binarize_wet(arr: np.ndarray, thr_mode: str) -> np.ndarray:
#     if thr_mode == "gt0":    return (arr > 0).astype(np.float32)
#     if thr_mode == "ge0.5":  return (arr >= 0.5).astype(np.float32)
#     if thr_mode == "ge0.25": return (arr >= 0.25).astype(np.float32)
#     return (arr == 1).astype(np.float32)

# def luh2_file_for_scenario(scen: str) -> str | None:
#     rel = LUH2_MAP.get(scen)
#     if not rel:
#         return None
#     path = os.path.join(LUH2_SSP_ROOT, rel)
#     return path if os.path.isfile(path) else None

# # LUH2 open with integer year coordinates (avoid calendar decode)
# def load_luh2_crops_pastr(states_files, engine="netcdf4"):
#     if not states_files:
#         return None, None
#     ds = xr.open_mfdataset(states_files, combine="by_coords", decode_times=False, engine=engine)
#     ds = _std(ds)
#     units = str(ds["time"].attrs.get("units", "years since 2015-01-01"))
#     m = re.search(r"years\s+since\s*(\d{1,4})", units)
#     base_year = int(m.group(1)) if m else 2015
#     offs = np.rint(np.asarray(ds["time"].values)).astype(int)
#     yrs = base_year + offs
#     ds = ds.assign_coords(time=("time", yrs))
#     ds["time"].attrs.clear()

#     need = ["c3ann","c4ann","c3per","c4per","c3nfx","pastr"]
#     miss = [v for v in need if v not in ds.data_vars]
#     if miss:
#         raise KeyError(f"LUH2 states missing variables: {miss}")

#     crops = (ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]).fillna(0).clip(0,1)
#     pastr = ds["pastr"].fillna(0).clip(0,1)
#     return crops, pastr

# # ── Preflight LUH2 ───────────────────────────────────────────────────────────
# if APPLY_AG_MASK:
#     missing = []
#     for scen in SCENARIOS:
#         f = luh2_file_for_scenario(scen)
#         if not f:
#             missing.append(scen)
#     if missing:
#         raise FileNotFoundError("APPLY_AG_MASK=True, LUH2 states.nc missing for: " + ", ".join(missing))
#     else:
#         logger.info("Preflight LUH2 OK for all scenarios: %s", ", ".join(SCENARIOS))
# else:
#     logger.info("APPLY_AG_MASK=False, running without LUH2 exclusion")

# # ── QA grid and mask ─────────────────────────────────────────────────────────
# qa_paths = sorted([os.path.join(QA_DIR, f) for f in os.listdir(QA_DIR) if f.endswith(".nc")])
# if not qa_paths:
#     raise FileNotFoundError(f"No .nc files in {QA_DIR}")
# qa = open_qa_merged(qa_paths)

# qa_lat = qa["lat"].values
# qa_lon = qa["lon"].values
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

# # pixel areas km2 on QA grid
# R = 6_371_000.0
# lat_res = float(abs(qa_lat[1] - qa_lat[0]))
# lon_res = float(abs(qa_lon[1] - qa_lon[0]))
# dlam = lon_res / 360.0
# band = np.sin(np.deg2rad(qa_lat + lat_res/2.0)) - np.sin(np.deg2rad(qa_lat - lat_res/2.0))
# area_per_lat = (2.0 * np.pi * R**2) * band * dlam / 1e6
# pixel_area = np.repeat(area_per_lat[:, None], nx, axis=1).astype("float32")

# # ── Biome × realm raster on QA grid ──────────────────────────────────────────
# gdf = gpd.read_file(BIOME_SHP).set_crs("EPSG:4326", allow_override=True)
# if "BIOME" not in gdf.columns or "REALM" not in gdf.columns:
#     raise RuntimeError("Shapefile must contain BIOME and REALM fields")
# gdf = gdf.loc[gdf["BIOME"].between(1, 14), ["BIOME", "REALM", "geometry"]].copy()
# gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)

# unique_combos = sorted(gdf["BIOME_ID_REALM"].unique())
# combo_to_code = {cmb: i + 1 for i, cmb in enumerate(unique_combos)}  # 0 outside
# code_to_combo = {v: k for k, v in combo_to_code.items()}
# n_codes = len(combo_to_code)

# # rasterize with pixel edge transform
# transform = Affine(lon_res, 0, qa_lon.min() - lon_res/2.0, 0, -lat_res, qa_lat.max() + lat_res/2.0)
# shapes = ((geom, combo_to_code[cmb]) for geom, cmb in zip(gdf.geometry, gdf["BIOME_ID_REALM"]))
# codes_arr = rasterio.features.rasterize(
#     shapes, out_shape=(ny, nx), transform=transform, fill=0, dtype="int32"
# )
# if APPLY_QA_MASK:
#     codes_arr = np.where(qa_mask, codes_arr, 0).astype(np.int32)

# # optional cache
# try:
#     cache_path = os.path.join(OUT_DIR, "biome_realm_mask_cache.npz")
#     if os.path.isfile(cache_path):
#         d = np.load(cache_path)
#         codes_arr = d["codes_arr"].astype(np.int32)
#         pixel_area = d["pixel_area"].astype(np.float32)
#     else:
#         np.savez_compressed(cache_path, codes_arr=codes_arr, pixel_area=pixel_area)
# except Exception:
#     pass

# # ── Core processing ───────────────────────────────────────────────────────────
# def process_scenario(scenario: str) -> pd.DataFrame:
#     path_nc = os.path.join(MASK_DIR, f"wetGDE_{scenario}.nc")
#     if not os.path.exists(path_nc):
#         logger.warning("Missing file: %s  skipping", path_nc)
#         return pd.DataFrame()

#     try:
#         ds = xr.open_dataset(path_nc, decode_times=True, engine=ENGINE)
#     except Exception:
#         ds = xr.open_dataset(path_nc, decode_times=False, engine=ENGINE)
#         ds = ds.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))

#     var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
#     wet = _std(ds[var].to_dataset(name="wet"))["wet"]

#     if SMALL_TEST:
#         wet = wet.isel(time=slice(0, 1))
#         logger.info("TEST mode, first timestamp %s", pd.to_datetime(wet["time"].values[0]).isoformat())

#     times = pd.to_datetime(wet["time"].values)
#     logger.info("WetGDE time coverage: %s .. %s  n=%d", times[0].date(), times[-1].date(), len(times))

#     same_lat = np.array_equal(wet["lat"].values, qa_lat)
#     same_lon = np.array_equal(wet["lon"].values, qa_lon)

#     crops_full = pastr_full = None
#     if APPLY_AG_MASK:
#         f = luh2_file_for_scenario(scenario)
#         logger.info("LUH2 file for %s: %s", scenario, f or "NONE")
#         if f:
#             crops_full, pastr_full = load_luh2_crops_pastr([f], engine=ENGINE)
#         else:
#             logger.warning("No LUH2 states for %s, skipping ag exclusion", scenario)
#     else:
#         logger.info("No LUH2 ag mask will be applied for %s", scenario)

#     ag_same_lat = ag_same_lon = False
#     if (crops_full is not None) and (pastr_full is not None):
#         ag_same_lat = np.array_equal(crops_full["lat"].values, qa_lat)
#         ag_same_lon = np.array_equal(crops_full["lon"].values, qa_lon)

#         years_wet = xr.DataArray(pd.to_datetime(times).year, dims=("time",))
#         crops_t = crops_full.sel(time=years_wet)
#         pastr_t = pastr_full.sel(time=years_wet)

#         if AG_RULE == "exclude_after_conversion":
#             mask_since = (years_wet >= AG_BASE_YEAR)
#             crops_since = xr.where(mask_since, crops_t, 0)
#             pastr_since = xr.where(mask_since, pastr_t, 0)
#             crops_t = (crops_since > 0).astype("uint8").cumsum("time").clip(0,1)
#             pastr_t = (pastr_since > 0).astype("uint8").cumsum("time").clip(0,1)

#         valid_per_t = np.isfinite(crops_t).any(dim=("lat","lon"))
#         sum_da = valid_per_t.sum()
#         try:
#             n_ok = int(sum_da.item())
#         except Exception:
#             n_ok = int(sum_da.compute().item())
#         logger.info("LUH2 year matching available for %d/%d months", n_ok, len(times))
#     else:
#         crops_t = pastr_t = None

#     exps = ["none", "crops", "crops_pasture"]
#     sums = {e: np.zeros((len(times), n_codes + 1), dtype=np.float64) for e in exps}

#     for t0_idx in range(0, len(times), TIME_BATCH):
#         t1_idx = min(t0_idx + TIME_BATCH, len(times))
#         wet_tb = wet.isel(time=slice(t0_idx, t1_idx))

#         if SMALL_TEST:
#             k0 = 0
#             wet0 = (wet_tb.isel(time=k0) if same_lat and same_lon
#                     else wet_tb.isel(time=k0).interp(lat=qa_lat, lon=qa_lon, method="nearest"))
#             wet_bin0 = binarize_wet(wet0.astype("float32").values, WET_THRESHOLD)
#             wet_n = int(np.nan_to_num(wet_bin0, nan=0).sum())
#             if crops_t is not None:
#                 if ag_same_lat and ag_same_lon:
#                     c0 = (crops_t.isel(time=t0_idx).values > 0).astype(np.uint8)
#                     cp0 = ((crops_t.isel(time=t0_idx).values + pastr_t.isel(time=t0_idx).values) > 0).astype(np.uint8)
#                 else:
#                     c0 = (crops_t.isel(time=t0_idx).interp(lat=qa_lat, lon=qa_lon, method="nearest").values > 0).astype(np.uint8)
#                     cp0 = ((crops_t.isel(time=t0_idx).interp(lat=qa_lat, lon=qa_lon, method="nearest").values +
#                             pastr_t.isel(time=t0_idx).interp(lat=qa_lat, lon=qa_lon, method="nearest").values) > 0).astype(np.uint8)
#                 inter_c  = int((wet_bin0 * c0).sum())
#                 inter_cp = int((wet_bin0 * cp0).sum())
#                 logger.info("OVERLAP t=%s wet_n=%d wet∩crops=%d wet∩crops+pasture=%d",
#                             pd.to_datetime(wet_tb.time.values[k0]).date(), wet_n, inter_c, inter_cp)
#             else:
#                 logger.info("OVERLAP t=%s wet_n=%d no LUH2", pd.to_datetime(wet_tb.time.values[k0]).date(), wet_n)

#         for y0 in range(0, ny, TILE_Y):
#             y1 = min(y0 + TILE_Y, ny)
#             lat_win = qa_lat[y0:y1]
#             for x0 in range(0, nx, TILE_X):
#                 x1 = min(x0 + TILE_X, nx)
#                 lon_win = qa_lon[x0:x1]

#                 # wet tile, tile-wise interpolation
#                 if same_lat and same_lon:
#                     w = wet_tb.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
#                 else:
#                     w = wet_tb.interp(lat=lat_win, lon=lon_win, method="nearest").astype("float32").values
#                 w = binarize_wet(w, WET_THRESHOLD)

#                 codes_tile = codes_arr[y0:y1, x0:x1]
#                 area_tile  = pixel_area[y0:y1, x0:x1]
#                 codes_flat = codes_tile.ravel()
#                 area_flat  = area_tile.ravel()

#                 uniq_codes, inv = np.unique(codes_flat, return_inverse=True)
#                 acc_none = np.zeros(uniq_codes.size, dtype=np.float64)
#                 acc_crops = np.zeros_like(acc_none)
#                 acc_cp = np.zeros_like(acc_none)

#                 # LUH2 tile, interpolate per tile only
#                 if crops_t is not None:
#                     crops_tb = crops_t.isel(time=slice(t0_idx, t1_idx))
#                     pastr_tb = pastr_t.isel(time=slice(t0_idx, t1_idx))
#                     if ag_same_lat and ag_same_lon:
#                         crops_tile = crops_tb.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
#                         pastr_tile = pastr_tb.isel(lat=slice(y0, y1), lon=slice(x0, x1)).astype("float32").values
#                     else:
#                         crops_tile = crops_tb.interp(lat=lat_win, lon=lon_win, method="nearest").astype("float32").values
#                         pastr_tile = pastr_tb.interp(lat=lat_win, lon=lon_win, method="nearest").astype("float32").values
#                 else:
#                     crops_tile = pastr_tile = None

#                 for k in range(w.shape[0]):
#                     wet_flat = w[k].ravel()
#                     weights_base = np.nan_to_num(area_flat * wet_flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

#                     np.add.at(acc_none, inv, weights_base)

#                     if crops_tile is None:
#                         np.add.at(acc_crops, inv, weights_base)
#                         np.add.at(acc_cp,    inv, weights_base)
#                     else:
#                         crops_f = np.nan_to_num(crops_tile[k].ravel(), 0.0).clip(0,1)
#                         pastr_f = np.nan_to_num(pastr_tile[k].ravel(), 0.0).clip(0,1)
#                         np.add.at(acc_crops, inv, weights_base * (1.0 - crops_f))
#                         np.add.at(acc_cp,    inv, weights_base * (1.0 - np.clip(crops_f + pastr_f, 0.0, 1.0)))

#                     sums["none"][t0_idx + k, uniq_codes] += acc_none
#                     sums["crops"][t0_idx + k, uniq_codes] += acc_crops
#                     sums["crops_pasture"][t0_idx + k, uniq_codes] += acc_cp

#                     acc_none.fill(0.0); acc_crops.fill(0.0); acc_cp.fill(0.0)

#         logger.info("Scenario %s, accumulated %d/%d timesteps", scenario, t1_idx, len(times))
#         if SMALL_TEST:
#             logger.info("TEST mode, stopping after first batch")
#             break

#     records = []
#     for e in ("none","crops","crops_pasture"):
#         sarr = sums[e]
#         for i, ts in enumerate(times):
#             nz_codes = np.nonzero(sarr[i, 1:] > 0.0)[0] + 1
#             for code in nz_codes:
#                 a = float(sarr[i, code])
#                 records.append((scenario, ts, code_to_combo[code], e, a))

#     if not records:
#         return pd.DataFrame(columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])

#     df = pd.DataFrame.from_records(records, columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])
#     df = df.sort_values(["scenario","time","BIOME_ID_REALM","ag_exclusion"]).reset_index(drop=True)
#     return df

# # ── NetCDF writer, ncview-friendly integer dims + name lookups ───────────────
# def _tmp_nc_path(name):
#     if not WRITE_NC:
#         return None
#     if NC_TRANSIENT:
#         dir_ = NC_TMPDIR if NC_TMPDIR and os.path.isdir(NC_TMPDIR) else OUT_DIR
#         Path(dir_).mkdir(parents=True, exist_ok=True)
#         prefix = name.rstrip(".nc") + "_"
#         fd, path = tempfile.mkstemp(prefix=prefix, suffix=".nc", dir=dir_)
#         os.close(fd)
#         return path
#     Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
#     return os.path.join(OUT_DIR, name)

# def _str_lookup_array(strings, dim_name):
#     """Return a DataArray char array of shape (dim, strlen) for fixed-width storage."""
#     strings = np.asarray(strings, dtype=object).astype(str)
#     L = max(1, max(len(s) for s in strings))
#     arr = np.full((len(strings), L), b" ", dtype="S1")
#     for i, s in enumerate(strings):
#         bs = np.frombuffer(s.encode("utf-8"), dtype="S1")[:L]
#         arr[i, :bs.size] = bs
#     return xr.DataArray(arr, dims=(dim_name, f"{dim_name}_strlen"))

# def write_nc_cube(df_all: pd.DataFrame, name: str | None = None):
#     # categories
#     scenarios = np.array(sorted(df_all["scenario"].unique()), dtype=object).astype(str)
#     biomes    = np.array(sorted(df_all["BIOME_ID_REALM"].unique()), dtype=object).astype(str)
#     exps      = np.array(sorted(df_all["ag_exclusion"].unique()), dtype=object).astype(str)
#     times     = np.array(sorted(pd.to_datetime(df_all["time"].unique())))  # datetime64[ns]

#     # integer-coded dims
#     scen_ids  = np.arange(len(scenarios), dtype="int32")
#     biome_ids = np.arange(len(biomes), dtype="int32")
#     exp_ids   = np.arange(len(exps), dtype="int32")

#     idx_t = {t:i for i,t in enumerate(times)}
#     idx_s = {s:i for i,s in enumerate(scenarios)}
#     idx_b = {b:i for i,b in enumerate(biomes)}
#     idx_e = {e:i for i,e in enumerate(exps)}

#     data = np.full((len(times), len(scenarios), len(biomes), len(exps)), np.nan, dtype="float32")
#     for s, t, b, e, a in df_all[["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"]].itertuples(index=False):
#         data[idx_t[pd.to_datetime(t)], idx_s[str(s)], idx_b[str(b)], idx_e[str(e)]] = a

#     ds = xr.Dataset(
#         data_vars={
#             "gde_area_km2": (("time","scenario","biome_realm","ag_exclusion"), data),
#         },
#         coords={
#             "time": times,                 # encoded to CF numeric below
#             "scenario": scen_ids,          # integers to please ncview
#             "biome_realm": biome_ids,
#             "ag_exclusion": exp_ids,
#         },
#         attrs={
#             "title": NC_TITLE,
#             "institution": NC_INSTITUTION,
#             "source": "wetGDE monthly masks + LUH2 states",
#             "history": f"created {pd.Timestamp.utcnow().isoformat()}Z",
#             "author": NC_AUTHOR,
#             "description": NC_DESCRIPTION,
#             "notes": "Integer coords with label mappings in coord attributes. Time is monthly EOM.",
#         },
#     )

#     # label mappings as attributes on coords
#     ds["scenario"].attrs.update({
#         "long_name": "scenario index",
#         "flag_values": scen_ids,
#         "flag_meanings": " ".join(scenarios)  # space-separated tokens
#     })
#     ds["biome_realm"].attrs.update({
#         "long_name": "WWF terrestrial biome × realm index",
#         "flag_values": biome_ids,
#         "flag_meanings": " ".join(biomes)     # e.g. '1_NT 6_PA ...'
#     })
#     ds["ag_exclusion"].attrs.update({
#         "long_name": "agricultural exclusion (with a base year of 2000)",
#         "flag_values": exp_ids,
#         "flag_meanings": " ".join(exps)       # 'none crops crops_pasture'
#     })

#     # data variable metadata
#     ds["gde_area_km2"].attrs.update({
#         "long_name": "WetlandGDE area aggregated by biome × realm with QA and ag exclusion",
#         "units": "km2",
#         "cell_methods": "time: point",
#     })

#     enc = {
#         "gde_area_km2": {"zlib": True, "complevel": 4,
#                          "chunksizes": (min(len(times), 24), 1, 1, 1)},
#         "time": {"units": "days since 1900-01-01 00:00:00", "calendar": "standard"},
#     }

#     out_name = name or NC_FILENAME
#     path_nc = _tmp_nc_path(out_name)
#     if path_nc:
#         ds.to_netcdf(path_nc, engine="netcdf4", format="NETCDF4_CLASSIC", encoding=enc)
#         logger.info("NetCDF written: %s", path_nc)
#     return path_nc

# # ── Parquet wide per scenario ONLY ────────────────────────────────────────────
# def save_parquet_wide_per_scenario(df: pd.DataFrame, out_dir: str, base_prefix: str):
#     for scen, d in df.groupby("scenario"):
#         w = (d.pivot_table(index=["time","BIOME_ID_REALM"],
#                            columns="ag_exclusion",
#                            values="area_km2",
#                            aggfunc="first")
#                .rename(columns={
#                    "none":"area_none_km2",
#                    "crops":"area_crops_excl_km2",
#                    "crops_pasture":"area_crops_pasture_excl_km2"})
#                .reset_index())
#         for col in ["area_none_km2","area_crops_excl_km2","area_crops_pasture_excl_km2"]:
#             if col not in w.columns:
#                 w[col] = np.nan
#         w = w[["time","BIOME_ID_REALM","area_none_km2","area_crops_excl_km2","area_crops_pasture_excl_km2"]]
#         out_pq = os.path.join(out_dir, f"{base_prefix}_{scen}.parquet")
#         w.to_parquet(out_pq, index=False, compression="zstd")
#         logger.info("Parquet written (wide): %s rows=%d", out_pq, len(w))

# # ── Run ───────────────────────────────────────────────────────────────────────
# def main():
#     all_parts = []
#     for scen in SCENARIOS:
#         if scen == "":
#             continue
#         df_scen = process_scenario(scen)
#         if not df_scen.empty:
#             all_parts.append(df_scen)

#     if all_parts:
#         df = pd.concat(all_parts, ignore_index=True)

#         single_scenario = len(SCENARIOS) == 1
#         scen_tag = SCENARIOS[0] if single_scenario else "all"
#         base = "gde_area_by_biome_realm_monthly"

#         # save ONLY per-scenario wide parquet files
#         save_parquet_wide_per_scenario(df, OUT_DIR, base)

#         # NetCDF, always write, name by scen_tag, integer dims with name lookups
#         if WRITE_NC:
#             nc_name = f"{base}_{scen_tag}.nc"
#             write_nc_cube(df, nc_name)
#     else:
#         logger.warning("No scenario produced output")

#     logger.info("DONE in %.1f s", time.time() - t0)

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # future_gdes_area.py
# # Streaming NetCDF + streaming and final wide Parquet per scenario
# # Aggregates monthly wetGDE masks to area (km^2) by WWF biome × realm
# # with optional LUH2 agricultural exclusions.

# import os, re, time, logging
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import xarray as xr
# import geopandas as gpd
# import rasterio.features
# from affine import Affine
# from netCDF4 import Dataset, date2num

# # ── env helpers ───────────────────────────────────────────────────────────────
# def _env_bool(k, default):
#     v = os.getenv(k)
#     if v is None:
#         return default
#     return str(v).strip().lower() in ("1","true","t","yes","y")

# def _env_int(k, default):
#     v = os.getenv(k)
#     return int(v) if v not in (None,"") else default

# def _env_str(k, default):
#     v = os.getenv(k)
#     return v if v not in (None,"") else default

# # ── CONFIG ────────────────────────────────────────────────────────────────────
# MASK_DIR   = _env_str("MASK_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
# OUT_DIR    = _env_str("OUT_DIR",  "/projects/prjs1578/futurewetgde/wetGDEs_area_test")
# LOG_DIR    = _env_str("LOG_DIR",  f"{OUT_DIR}/logs")

# # scenarios, prefer single SCENARIO env
# if _env_str("SCENARIO", None):
#     SCENARIOS = [_env_str("SCENARIO", "")]
# elif _env_str("SCENARIOS", None):
#     SCENARIOS = [s.strip() for s in os.getenv("SCENARIOS").split(",") if s.strip()]
# else:
#     SCENARIOS = ["historical","ssp126","ssp370","ssp585"]

# BIOME_SHP  = _env_str("BIOME_SHP", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
# QA_DIR     = _env_str("QA_DIR",    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/")

# ENGINE         = _env_str("XR_ENGINE", "netcdf4")
# TILE_Y         = _env_int("TILE_Y", 2048)
# TILE_X         = _env_int("TILE_X", 2048)
# TIME_BATCH     = _env_int("TIME_BATCH", 24)
# WET_THRESHOLD  = _env_str("WET_THRESHOLD", "gt0")   # gt0, ge0.25, ge0.5, eq1

# SMALL_TEST     = _env_bool("SMALL_TEST", False)
# APPLY_QA_MASK  = _env_bool("APPLY_QA_MASK", True)
# APPLY_AG_MASK  = _env_bool("APPLY_AG_MASK", True)

# AG_RULE        = _env_str("AG_RULE", "exclude_when_ag")  # or exclude_after_conversion
# AG_BASE_YEAR   = _env_int("AG_BASE_YEAR", 2000)

# LUH2_SSP_ROOT  = _env_str("LUH2_SSP_ROOT", "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp")
# LUH2_MAP = {
#     "historical": "Historic_Data/states.nc",
#     "ssp126":    "RCP2_6_SSP1_from_IMAGE/states.nc",
#     "ssp245":    "RCP4_5_SSP2_from_MESSAGE_GLOBIOM/states.nc",
#     "ssp370":    "RCP7_0_SSP3_from_AIM/states.nc",
#     "ssp434":    "RCP3_4_SSP4_from_GCAM/states.nc",
#     "ssp460":    "RCP6_0_SSP4_from_GCAM/states.nc",
#     "ssp585":    "RCP8_5_SSP5_from_REMIND_MAGPIE/states.nc",
# }

# # NetCDF output
# WRITE_NC            = _env_bool("WRITE_NC", True)
# WRITE_NC_STREAMING  = _env_bool("WRITE_NC_STREAMING", True)  # live append per TIME_BATCH
# NC_TRANSIENT        = _env_bool("NC_TRANSIENT", False)       # write to NC_TMPDIR if True
# NC_TMPDIR           = _env_str("NC_TMPDIR", "/scratch")
# NC_ENGINE           = "netcdf4"

# # NetCDF metadata
# NC_TITLE       = _env_str("NC_TITLE", "Global wetGDE area by biome × realm with agricultural exclusions")
# NC_INSTITUTION = _env_str("NC_INSTITUTION", "Utrecht University")
# NC_AUTHOR      = _env_str("NC_AUTHOR", "Nicole Gyakowah Otoo <n.g.otoo@uu.nl>")
# NC_DESCRIPTION = _env_str("NC_DESCRIPTION",
#     "Monthly wetland GDE area aggregated by WWF terrestrial biome × realm on a regular lon lat grid. "
#     "Values are area in km^2 after QA masking and agricultural exclusion experiments, none, crops, crops_pasture, derived from LUH2 states.")

# # Parquet
# PARQUET_CODEC            = _env_str("PARQUET_CODEC", "snappy")
# WRITE_PARQUET_STREAMING  = _env_bool("WRITE_PARQUET_STREAMING", True)  # write one part per batch
# PARQUET_LIVE_DIR         = _env_str("PARQUET_LIVE_DIR", OUT_DIR)       # dataset root for parts
# WRITE_PARQUET_FINAL      = _env_bool("WRITE_PARQUET_FINAL", True)      # final per-scenario wide file

# # Optional tiny ncview preview map
# PREVIEW_MAP              = _env_bool("PREVIEW_MAP", False)
# PREVIEW_EVERY_N_MONTHS   = _env_int("PREVIEW_EVERY_N_MONTHS", 12)
# PREVIEW_EXCLUSION        = _env_str("PREVIEW_EXCLUSION", "crops")  # none|crops|crops_pasture

# # ── logging ───────────────────────────────────────────────────────────────────
# Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
# Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# logger = logging.getLogger("gde_area_by_biome_realm_monthly_tiled")
# logger.setLevel(logging.INFO)
# fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
# fh = logging.FileHandler(f"{LOG_DIR}/area_monthly_tiled.log"); fh.setFormatter(fmt); logger.addHandler(fh)

# t0 = time.time()
# logger.info(
#     "START MASK_DIR=%s OUT_DIR=%s SCENARIOS=%s SMALL_TEST=%s APPLY_QA_MASK=%s APPLY_AG_MASK=%s "
#     "WET_THRESHOLD=%s AG_RULE=%s AG_BASE_YEAR=%d LUH2_SSP_ROOT=%s XR_ENGINE=%s",
#     MASK_DIR, OUT_DIR, ",".join(SCENARIOS), SMALL_TEST, APPLY_QA_MASK, APPLY_AG_MASK,
#     WET_THRESHOLD, AG_RULE, AG_BASE_YEAR, LUH2_SSP_ROOT, ENGINE,
# )

# # ── helpers ───────────────────────────────────────────────────────────────────
# def _std(ds: xr.Dataset) -> xr.Dataset:
#     if "latitude" in ds.coords:  ds = ds.rename({"latitude":"lat"})
#     if "longitude" in ds.coords: ds = ds.rename({"longitude":"lon"})
#     return ds

# def binarize_wet(arr: np.ndarray, thr_mode: str) -> np.ndarray:
#     if thr_mode == "gt0":    return (arr > 0).astype(np.float32)
#     if thr_mode == "ge0.5":  return (arr >= 0.5).astype(np.float32)
#     if thr_mode == "ge0.25": return (arr >= 0.25).astype(np.float32)
#     return (arr == 1).astype(np.float32)

# def luh2_file_for_scenario(scen: str) -> str | None:
#     rel = LUH2_MAP.get(scen)
#     if not rel:
#         return None
#     path = os.path.join(LUH2_SSP_ROOT, rel)
#     return path if os.path.isfile(path) else None

# def open_qa_merged(qa_paths):
#     base = _std(xr.open_dataset(qa_paths[0], decode_times=False, mask_and_scale=False))
#     qa_lat = base["lat"].values
#     qa_lon = base["lon"].values
#     ny, nx = qa_lat.size, qa_lon.size
#     qa_vars = {}
#     for p in qa_paths:
#         ds = _std(xr.open_dataset(p, decode_times=False, mask_and_scale=False))
#         if (ds.sizes.get("lat") != ny) or (ds.sizes.get("lon") != nx) or \
#            (not np.array_equal(ds["lat"].values, qa_lat)) or \
#            (not np.array_equal(ds["lon"].values, qa_lon)):
#             ds = ds.interp(lat=qa_lat, lon=qa_lon, method="nearest")
#         for v in ds.data_vars:
#             dv = ds[v]
#             if "time" in dv.dims:
#                 dv = dv.isel(time=0, drop=True)
#             qa_vars[v] = dv
#     return xr.Dataset(qa_vars, coords={"lat": qa_lat, "lon": qa_lon})

# # fast nearest neighbor regrid indices
# def _nearest_index(src, tgt):
#     idx = np.searchsorted(src, tgt)
#     idx = np.clip(idx, 1, len(src)-1)
#     left = src[idx-1]; right = src[idx]
#     use_left = (tgt - left) <= (right - tgt)
#     return (idx - use_left.astype(np.int32)).astype(np.int32)

# def build_regrid_indices(src_lat, src_lon, qa_lat, qa_lon):
#     lat_src = src_lat if src_lat[0] < src_lat[-1] else src_lat[::-1]
#     lat_rev = (src_lat[0] > src_lat[-1])
#     lat_idx = _nearest_index(lat_src, qa_lat)
#     if lat_rev:
#         lat_idx = (len(src_lat) - 1) - lat_idx
#     lon_idx = _nearest_index(src_lon, qa_lon)
#     return lat_idx, lon_idx

# # LUH2 loader, avoid CF calendar decoding, work in integer years
# def load_luh2_crops_pastr(states_files, engine="netcdf4"):
#     if not states_files:
#         return None, None
#     ds = xr.open_mfdataset(states_files, combine="by_coords", decode_times=False,
#                            engine=engine, mask_and_scale=False)
#     ds = _std(ds)
#     units = str(ds["time"].attrs.get("units", "years since 2015-01-01"))
#     m = re.search(r"years\s+since\s*(\d{1,4})", units)
#     base_year = int(m.group(1)) if m else 2015
#     offs = np.rint(np.asarray(ds["time"].values)).astype(int)
#     yrs = base_year + offs
#     ds = ds.assign_coords(time=("time", yrs.astype("int32")))
#     ds["time"].attrs.clear()

#     need = ["c3ann","c4ann","c3per","c4per","c3nfx","pastr"]
#     miss = [v for v in need if v not in ds.data_vars]
#     if miss:
#         raise KeyError(f"LUH2 states missing variables: {miss}")

#     crops = (ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]).fillna(0).clip(0,1)
#     pastr = ds["pastr"].fillna(0).clip(0,1)
#     return crops, pastr

# # ── preflight LUH2 ───────────────────────────────────────────────────────────
# if APPLY_AG_MASK:
#     missing = [s for s in SCENARIOS if not luh2_file_for_scenario(s)]
#     if missing:
#         raise FileNotFoundError("APPLY_AG_MASK=True, LUH2 states.nc missing for: " + ", ".join(missing))
#     logger.info("Preflight LUH2 OK for all scenarios: %s", ", ".join(SCENARIOS))
# else:
#     logger.info("APPLY_AG_MASK=False, running without LUH2 exclusion")

# # ── QA grid and masks ────────────────────────────────────────────────────────
# qa_paths = sorted([os.path.join(QA_DIR, f) for f in os.listdir(QA_DIR) if f.endswith(".nc")])
# if not qa_paths:
#     raise FileNotFoundError(f"No .nc files in {QA_DIR}")
# qa = open_qa_merged(qa_paths)

# qa_lat = qa["lat"].values
# qa_lon = qa["lon"].values
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

# # pixel areas, km^2 on QA grid
# R = 6_371_000.0
# lat_res = float(abs(qa_lat[1] - qa_lat[0]))
# lon_res = float(abs(qa_lon[1] - qa_lon[0]))
# dlam = lon_res / 360.0
# band = np.sin(np.deg2rad(qa_lat + lat_res/2.0)) - np.sin(np.deg2rad(qa_lat - lat_res/2.0))
# area_per_lat = (2.0 * np.pi * R**2) * band * dlam / 1e6
# pixel_area = np.repeat(area_per_lat[:, None], nx, axis=1).astype("float32")

# # ── Biome×realm raster on QA grid ────────────────────────────────────────────
# gdf = gpd.read_file(BIOME_SHP).set_crs("EPSG:4326", allow_override=True)
# if "BIOME" not in gdf.columns or "REALM" not in gdf.columns:
#     raise RuntimeError("Shapefile must contain BIOME and REALM fields")
# gdf = gdf.loc[gdf["BIOME"].between(1, 14), ["BIOME","REALM","geometry"]].copy()
# gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)

# unique_combos = sorted(gdf["BIOME_ID_REALM"].unique())
# combo_to_code = {cmb: i+1 for i, cmb in enumerate(unique_combos)}  # 0 = outside
# code_to_combo = {v:k for k,v in combo_to_code.items()}
# n_codes = len(combo_to_code)

# # rasterize with pixel edge transform aligned to cell centers
# transform = Affine(lon_res, 0, qa_lon.min() - lon_res/2.0, 0, -lat_res, qa_lat.max() + lat_res/2.0)
# shapes = ((geom, combo_to_code[cmb]) for geom, cmb in zip(gdf.geometry, gdf["BIOME_ID_REALM"]))
# codes_arr = rasterio.features.rasterize(
#     shapes, out_shape=(ny, nx), transform=transform, fill=0, dtype="int32"
# )
# if APPLY_QA_MASK:
#     codes_arr = np.where(qa_mask, codes_arr, 0).astype(np.int32)

# # ── NetCDF streaming helpers ─────────────────────────────────────────────────
# def _nc_path_for_scenario(base_prefix: str, scenario: str) -> str:
#     name = f"{base_prefix}_{scenario}.nc"
#     if NC_TRANSIENT:
#         dir_ = NC_TMPDIR if NC_TMPDIR and os.path.isdir(NC_TMPDIR) else OUT_DIR
#         Path(dir_).mkdir(parents=True, exist_ok=True)
#         return os.path.join(dir_, name)
#     Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
#     return os.path.join(OUT_DIR, name)

# def nc_stream_open(path_nc: str, biomes: list[str], exps: list[str], scenario: str):
#     mode = "a" if os.path.exists(path_nc) else "w"
#     ds = Dataset(path_nc, mode, format="NETCDF4_CLASSIC")

#     if mode == "w":
#         # streaming dims
#         ds.createDimension("time", None)  # unlimited
#         ds.createDimension("biome_realm", len(biomes))
#         ds.createDimension("ag_exclusion", len(exps))
#         # static map dims so ncview shows a globe
#         ds.createDimension("lat", ny)
#         ds.createDimension("lon", nx)

#         # time
#         tvar = ds.createVariable("time", "f8", ("time",))
#         tvar.units = "days since 1900-01-01 00:00:00"
#         tvar.calendar = "standard"
#         tvar.standard_name = "time"

#         # category coords with labels in attributes
#         bvar = ds.createVariable("biome_realm", "i4", ("biome_realm",))
#         bvar[:] = np.arange(len(biomes), dtype=np.int32)
#         bvar.long_name = "WWF terrestrial biome x realm index"
#         bvar.flag_values = np.arange(len(biomes), dtype=np.int32)
#         bvar.flag_meanings = " ".join(biomes)

#         evar = ds.createVariable("ag_exclusion", "i4", ("ag_exclusion",))
#         evar[:] = np.arange(len(exps), dtype=np.int32)
#         evar.long_name = "agricultural exclusion experiment index"
#         evar.flag_values = np.arange(len(exps), dtype=np.int32)
#         evar.flag_meanings = " ".join(exps)

#         # static map variables
#         latv = ds.createVariable("lat", "f4", ("lat",))
#         latv[:] = qa_lat
#         latv.standard_name = "latitude"
#         latv.units = "degrees_north"

#         lonv = ds.createVariable("lon", "f4", ("lon",))
#         lonv[:] = qa_lon
#         lonv.standard_name = "longitude"
#         lonv.units = "degrees_east"

#         codev = ds.createVariable("biome_realm_code", "i4", ("lat","lon"), zlib=True, complevel=1, fill_value=0)
#         codev[:] = codes_arr
#         codev.long_name = "WWF terrestrial biome × realm code (0=outside)"
#         codev.flag_values = np.arange(0, n_codes+1, dtype=np.int32)
#         codev.flag_meanings = "none " + " ".join(unique_combos)

#         # main streamed variable (no lat,lon)
#         v = ds.createVariable(
#             "gdeareakm2", "f4",
#             ("time","biome_realm","ag_exclusion"),
#             zlib=True, complevel=0,
#             chunksizes=(max(1, min(24, TIME_BATCH)), len(biomes), 1),
#             fill_value=np.nan
#         )
#         v.long_name = "WetGDE area aggregated by biome x realm with QA and agricultural exclusion"
#         v.units = "km2"
#         v.cell_methods = "time: point"

#         # optional preview variable
#         if PREVIEW_MAP:
#             ds.createDimension("time_preview", None)
#             tp = ds.createVariable("time_preview", "f8", ("time_preview",))
#             tp.units = "days since 1900-01-01 00:00:00"
#             tp.calendar = "standard"
#             vprev = ds.createVariable(
#                 "gdeareakm2_preview", "f4",
#                 ("time_preview", "lat", "lon"),
#                 zlib=True, complevel=1, fill_value=np.nan
#             )
#             vprev.long_name = "Preview map of gdeareakm2, broadcast per biome×realm"
#             vprev.units = "km2"
#         else:
#             tp = vprev = None

#         # global attrs
#         ds.title = NC_TITLE
#         ds.institution = NC_INSTITUTION
#         ds.source = "wetGDE monthly masks + LUH2 states"
#         ds.history = f"streaming created {pd.Timestamp.utcnow().isoformat()}Z"
#         ds.author = NC_AUTHOR
#         ds.description = NC_DESCRIPTION
#         ds.notes = "Integer category coords with label mappings in coord attributes. Time is monthly end of month."
#         ds.Conventions = "CF-1.8"
#         ds.scenario = scenario
#     else:
#         tvar = ds.variables["time"]
#         v = ds.variables["gdeareakm2"]
#         tp = ds.variables["time_preview"] if ("time_preview" in ds.variables and PREVIEW_MAP) else None
#         vprev = ds.variables["gdeareakm2_preview"] if ("gdeareakm2_preview" in ds.variables and PREVIEW_MAP) else None

#     return ds, tvar, v, tp, vprev

# def nc_stream_write_batch(nc_ds, tvar, vvar, times_slice: np.ndarray, sums_batch: dict,
#                           tp=None, vprev=None, t0_idx_global=0):
#     py_dt = pd.to_datetime(times_slice).to_pydatetime()
#     t0 = int(tvar.shape[0])
#     n = len(py_dt)
#     tvar[t0:t0+n] = date2num(py_dt, tvar.units, tvar.calendar)

#     arr_none  = sums_batch["none"][:, 1:]
#     arr_crops = sums_batch["crops"][:, 1:]
#     arr_cp    = sums_batch["crops_pasture"][:, 1:]
#     data_out  = np.stack([arr_none, arr_crops, arr_cp], axis=2)  # (batch, n_codes, 3)
#     vvar[t0:t0+n, :, :] = data_out

#     if PREVIEW_MAP and (tp is not None) and (vprev is not None):
#         which = PREVIEW_EXCLUSION
#         sel = sums_batch[which]  # (batch, n_codes+1)
#         for k in range(sel.shape[0]):
#             global_t = t0_idx_global + k
#             if global_t % PREVIEW_EVERY_N_MONTHS != 0:
#                 continue
#             vals = sel[k, :]        # totals per biome code
#             grid = vals[codes_arr]  # broadcast to grid
#             tpi = int(tp.shape[0])
#             dt = pd.to_datetime(times_slice[k]).to_pydatetime()
#             tp[tpi:tpi+1] = date2num([dt], tp.units, "standard")
#             vprev[tpi, :, :] = grid

#     nc_ds.sync()

# # ── Parquet writers ───────────────────────────────────────────────────────────
# def _wide_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
#     w = (df_long.pivot_table(index=["time","BIOME_ID_REALM"],
#                              columns="ag_exclusion",
#                              values="area_km2",
#                              aggfunc="first")
#                    .rename(columns={
#                        "none":"area_none_km2",
#                        "crops":"area_crops_excl_km2",
#                        "crops_pasture":"area_crops_pasture_excl_km2"})
#                    .reset_index())
#     for col in ["area_none_km2","area_crops_excl_km2","area_crops_pasture_excl_km2"]:
#         if col not in w.columns:
#             w[col] = np.nan
#     return w[["time","BIOME_ID_REALM","area_none_km2","area_crops_excl_km2","area_crops_pasture_excl_km2"]]

# def _write_parquet_wide_batch(scenario: str, df_batch: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp):
#     if df_batch.empty:
#         return
#     w = _wide_from_long(df_batch)
#     w["year"] = pd.to_datetime(w["time"]).dt.year
#     root = os.path.join(PARQUET_LIVE_DIR, f"gde_area_by_biome_realm_monthly_{scenario}")
#     Path(root).mkdir(parents=True, exist_ok=True)
#     for yr, wy in w.groupby("year"):
#         outdir = os.path.join(root, f"year={yr}")
#         Path(outdir).mkdir(parents=True, exist_ok=True)
#         fname = f"part_{pd.to_datetime(t0).strftime('%Y%m')}_{pd.to_datetime(t1).strftime('%Y%m')}.parquet"
#         wy.drop(columns=["year"]).to_parquet(os.path.join(outdir, fname), index=False, compression=PARQUET_CODEC)

# def save_parquet_wide_per_scenario(df: pd.DataFrame, out_dir: str, base_prefix: str):
#     for scen, d in df.groupby("scenario"):
#         w = _wide_from_long(d)
#         out_pq = os.path.join(out_dir, f"{base_prefix}_{scen}.parquet")
#         w.to_parquet(out_pq, index=False, compression=PARQUET_CODEC)
#         logger.info("Parquet written (wide final): %s rows=%d", out_pq, len(w))

# # ── processing ────────────────────────────────────────────────────────────────
# def process_scenario(scenario: str) -> pd.DataFrame:
#     path_nc_in = os.path.join(MASK_DIR, f"wetGDE_{scenario}.nc")
#     if not os.path.exists(path_nc_in):
#         logger.warning("Missing file: %s  skipping", path_nc_in)
#         return pd.DataFrame()

#     try:
#         ds = xr.open_dataset(path_nc_in, decode_times=True, engine=ENGINE, mask_and_scale=False)
#     except Exception:
#         ds = xr.open_dataset(path_nc_in, decode_times=False, engine=ENGINE, mask_and_scale=False)
#         ds = ds.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))

#     var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
#     wet = _std(ds[var].to_dataset(name="wet"))["wet"]

#     if SMALL_TEST:
#         wet = wet.isel(time=slice(0, 1))
#         logger.info("TEST mode, first timestamp %s", pd.to_datetime(wet["time"].values[0]).isoformat())

#     times = pd.to_datetime(wet["time"].values)
#     logger.info("WetGDE time coverage: %s .. %s  n=%d", times[0].date(), times[-1].date(), len(times))

#     # regrid indices to QA grid
#     same_wet_lat = np.array_equal(wet["lat"].values, qa_lat)
#     same_wet_lon = np.array_equal(wet["lon"].values, qa_lon)
#     if not (same_wet_lat and same_wet_lon):
#         wet_lat_idx, wet_lon_idx = build_regrid_indices(wet["lat"].values, wet["lon"].values, qa_lat, qa_lon)

#     # LUH2 to time axis
#     crops_t = pastr_t = None
#     if APPLY_AG_MASK:
#         f = luh2_file_for_scenario(scenario)
#         logger.info("LUH2 file for %s: %s", scenario, f or "NONE")
#         if f:
#             crops_full, pastr_full = load_luh2_crops_pastr([f], engine=ENGINE)
#             years_wet = xr.DataArray(pd.to_datetime(times).year.astype("int32"), dims=("time",))
#             crops_t = crops_full.reindex(time=years_wet, method="nearest", tolerance=1)
#             pastr_t = pastr_full.reindex(time=years_wet, method="nearest", tolerance=1)

#             if AG_RULE == "exclude_after_conversion":
#                 mask_since = (years_wet >= AG_BASE_YEAR)
#                 crops_since = xr.where(mask_since, crops_t, 0)
#                 pastr_since = xr.where(mask_since, pastr_t, 0)
#                 crops_t = (crops_since > 0).astype("uint8").cumsum("time").clip(0,1)
#                 pastr_t = (pastr_since > 0).astype("uint8").cumsum("time").clip(0,1)

#             same_ag_lat = np.array_equal(crops_full["lat"].values, qa_lat)
#             same_ag_lon = np.array_equal(crops_full["lon"].values, qa_lon)
#             if not (same_ag_lat and same_ag_lon):
#                 ag_lat_idx, ag_lon_idx = build_regrid_indices(crops_full["lat"].values, crops_full["lon"].values, qa_lat, qa_lon)

#             valid_per_t = xr.apply_ufunc(np.isfinite, crops_t, dask="allowed").any(dim=("lat","lon"))
#             total = valid_per_t.sum()
#             if hasattr(total, "compute"):
#                 total = total.compute()
#             if hasattr(total, "values"):
#                 total = total.values
#             n_ok = int(np.asarray(total))
#             logger.info("LUH2 year matching available for %d/%d months", n_ok, len(times))
#         else:
#             logger.warning("No LUH2 states for %s, skipping ag exclusion", scenario)

#     exps = ["none","crops","crops_pasture"]

#     # NetCDF stream open
#     nc_ds = tvar_nc = v_nc = tp_nc = vprev_nc = None
#     if WRITE_NC and WRITE_NC_STREAMING:
#         base = "gde_area_by_biome_realm_monthly"
#         nc_path = _nc_path_for_scenario(base, scenario)
#         nc_ds, tvar_nc, v_nc, tp_nc, vprev_nc = nc_stream_open(nc_path, unique_combos, exps, scenario)

#     wet_src_all = wet.astype("float32")
#     all_records = []

#     for t0_idx in range(0, len(times), TIME_BATCH):
#         t1_idx = min(t0_idx + TIME_BATCH, len(times))
#         batch_len = t1_idx - t0_idx

#         wet_tb = wet_src_all.isel(time=slice(t0_idx, t1_idx)).values
#         sums_batch = {e: np.zeros((batch_len, n_codes + 1), dtype=np.float64) for e in exps}

#         if crops_t is not None:
#             crops_tb = crops_t.isel(time=slice(t0_idx, t1_idx)).astype("float32").values
#             pastr_tb = pastr_t.isel(time=slice(t0_idx, t1_idx)).astype("float32").values

#         for y0 in range(0, ny, TILE_Y):
#             y1 = min(y0 + TILE_Y, ny)
#             lat_idx_tile = slice(y0, y1) if (same_wet_lat and same_wet_lon) else wet_lat_idx[y0:y1]
#             for x0 in range(0, nx, TILE_X):
#                 x1 = min(x0 + TILE_X, nx)
#                 lon_idx_tile = slice(x0, x1) if (same_wet_lon and same_wet_lat) else wet_lon_idx[x0:x1]

#                 if same_wet_lat and same_wet_lon:
#                     w = wet_tb[:, y0:y1, x0:x1]
#                 else:
#                     w = np.take(np.take(wet_tb, lat_idx_tile, axis=1), lon_idx_tile, axis=2)
#                 w = binarize_wet(w, WET_THRESHOLD)

#                 codes_tile = codes_arr[y0:y1, x0:x1]
#                 area_tile  = pixel_area[y0:y1, x0:x1]
#                 codes_flat = codes_tile.ravel()
#                 area_flat  = area_tile.ravel()

#                 if APPLY_AG_MASK and (crops_t is not None):
#                     if 'ag_lat_idx' in locals():
#                         c_tile = np.take(np.take(crops_tb, ag_lat_idx[y0:y1], axis=1), ag_lon_idx[x0:x1], axis=2)
#                         p_tile = np.take(np.take(pastr_tb, ag_lat_idx[y0:y1], axis=1), ag_lon_idx[x0:x1], axis=2)
#                     else:
#                         c_tile = crops_tb[:, y0:y1, x0:x1]
#                         p_tile = pastr_tb[:, y0:y1, x0:x1]
#                 else:
#                     c_tile = p_tile = None

#                 for k in range(w.shape[0]):
#                     wet_flat = w[k].ravel()
#                     weights_base = np.nan_to_num(area_flat * wet_flat, nan=0.0, posinf=0.0, neginf=0.0)
#                     if weights_base.sum() == 0.0:
#                         continue

#                     bc = np.bincount(codes_flat, weights=weights_base, minlength=n_codes + 1)
#                     sums_batch["none"][k, :bc.size] += bc

#                     if c_tile is None:
#                         sums_batch["crops"][k, :bc.size] += bc
#                         sums_batch["crops_pasture"][k, :bc.size] += bc
#                     else:
#                         crops_f = np.nan_to_num(c_tile[k].ravel(), 0.0).clip(0, 1)
#                         pastr_f = np.nan_to_num(p_tile[k].ravel(), 0.0).clip(0, 1)
#                         bc = np.bincount(codes_flat, weights=weights_base * (1.0 - crops_f), minlength=n_codes + 1)
#                         sums_batch["crops"][k, :bc.size] += bc
#                         cp = np.clip(crops_f + pastr_f, 0.0, 1.0)
#                         bc = np.bincount(codes_flat, weights=weights_base * (1.0 - cp), minlength=n_codes + 1)
#                         sums_batch["crops_pasture"][k, :bc.size] += bc

#         # build batch df (long) once
#         batch_records = []
#         for e in ("none","crops","crops_pasture"):
#             sarr = sums_batch[e]
#             for k in range(sarr.shape[0]):
#                 ts = times[t0_idx + k]
#                 nz = np.nonzero(sarr[k, 1:] > 0.0)[0] + 1
#                 for code in nz:
#                     batch_records.append((scenario, ts, code_to_combo[code], e, float(sarr[k, code])))
#         df_batch = pd.DataFrame(batch_records, columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])
#         all_records.extend(batch_records)

#         # streaming parquet part
#         if WRITE_PARQUET_STREAMING:
#             _write_parquet_wide_batch(scenario, df_batch, times[t0_idx], times[t1_idx-1])

#         # streaming NetCDF append
#         if WRITE_NC and WRITE_NC_STREAMING:
#             nc_stream_write_batch(nc_ds, tvar_nc, v_nc, times[t0_idx:t1_idx], sums_batch,
#                                   tp=tp_nc, vprev=vprev_nc, t0_idx_global=t0_idx)

#         logger.info("Scenario %s, appended %d/%d timesteps", scenario, t1_idx, len(times))
#         if SMALL_TEST:
#             logger.info("TEST mode, stopping after first batch")
#             break

#     if WRITE_NC and WRITE_NC_STREAMING and (nc_ds is not None):
#         nc_ds.close()

#     if not all_records:
#         return pd.DataFrame(columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])

#     df = pd.DataFrame.from_records(all_records, columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])
#     df = df.sort_values(["scenario","time","BIOME_ID_REALM","ag_exclusion"]).reset_index(drop=True)
#     return df

# # ── main ──────────────────────────────────────────────────────────────────────
# def main():
#     all_parts = []
#     for scen in SCENARIOS:
#         if not scen:
#             continue
#         df_scen = process_scenario(scen)
#         if not df_scen.empty:
#             all_parts.append(df_scen)

#     if all_parts:
#         df = pd.concat(all_parts, ignore_index=True)

#         base = "gde_area_by_biome_realm_monthly"

#         if WRITE_PARQUET_FINAL:
#             save_parquet_wide_per_scenario(df, OUT_DIR, base)

#         # optional non-streaming NC build from final df, usually not needed
#         if WRITE_NC and not WRITE_NC_STREAMING:
#             scen_tag = SCENARIOS[0] if len(SCENARIOS) == 1 else "all"
#             # build dense cube once using xarray
#             scenarios = np.array(sorted(df["scenario"].unique()), dtype=object).astype(str)
#             biomes    = np.array(sorted(df["BIOME_ID_REALM"].unique()), dtype=object).astype(str)
#             exps      = np.array(sorted(df["ag_exclusion"].unique()), dtype=object).astype(str)
#             times     = np.array(sorted(pd.to_datetime(df["time"].unique())))
#             scen_ids  = np.arange(len(scenarios), dtype="int32")
#             biome_ids = np.arange(len(biomes), dtype="int32")
#             exp_ids   = np.arange(len(exps), dtype="int32")
#             idx_t = {t:i for i,t in enumerate(times)}
#             idx_s = {s:i for i,s in enumerate(scenarios)}
#             idx_b = {b:i for i,b in enumerate(biomes)}
#             idx_e = {e:i for i,e in enumerate(exps)}
#             data = np.full((len(times), len(scenarios), len(biomes), len(exps)), np.nan, dtype="float32")
#             for s, t, b, e, a in df[["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"]].itertuples(index=False):
#                 data[idx_t[pd.to_datetime(t)], idx_s[str(s)], idx_b[str(b)], idx_e[str(e)]] = a
#             ds = xr.Dataset(
#                 {"gdeareakm2": (("time","scenario","biome_realm","ag_exclusion"), data)},
#                 coords={"time": times, "scenario": scen_ids, "biome_realm": biome_ids, "ag_exclusion": exp_ids},
#                 attrs={"title": NC_TITLE, "institution": NC_INSTITUTION, "source": "wetGDE monthly masks + LUH2 states",
#                        "history": f"created {pd.Timestamp.utcnow().isoformat()}Z", "author": NC_AUTHOR,
#                        "description": NC_DESCRIPTION, "Conventions": "CF-1.8"}
#             )
#             ds["scenario"].attrs.update({"long_name":"scenario index","flag_values":scen_ids,"flag_meanings":" ".join(scenarios)})
#             ds["biome_realm"].attrs.update({"long_name":"WWF terrestrial biome x realm index","flag_values":biome_ids,"flag_meanings":" ".join(biomes)})
#             ds["ag_exclusion"].attrs.update({"long_name":"agricultural exclusion experiment index","flag_values":exp_ids,"flag_meanings":" ".join(exps)})
#             ds["gdeareakm2"].attrs.update({"long_name":"WetGDE area aggregated by biome x realm with QA and agricultural exclusion","units":"km2","cell_methods":"time: point"})
#             enc = {"gdeareakm2": {"zlib": True, "complevel": 0, "chunksizes": (min(len(times),24),1,1,1)},
#                    "time": {"units": "days since 1900-01-01 00:00:00", "calendar": "standard"}}
#             out_path = os.path.join(OUT_DIR, f"{base}_{scen_tag}.nc")
#             ds.to_netcdf(out_path, engine=NC_ENGINE, format="NETCDF4_CLASSIC", encoding=enc)
#             logger.info("NetCDF written (final cube): %s", out_path)
#     else:
#         logger.warning("No scenario produced output")

#     logger.info("DONE in %.1f s", time.time() - t0)

# if __name__ == "__main__":
#     main()




###Climate only landuse only exclusion
#!/usr/bin/env python3
# future_gdes_area.py
# Streaming NetCDF + streaming and final wide Parquet per scenario
# Aggregates monthly wetGDE masks to area (km^2) by WWF biome × realm
# with optional LUH2 agricultural exclusions and counterfactual families.

import os, re, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio.features
from affine import Affine
from netCDF4 import Dataset, date2num

# ── env helpers ───────────────────────────────────────────────────────────────
def _env_bool(k, default):
    v = os.getenv(k)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","t","yes","y")

def _env_int(k, default):
    v = os.getenv(k)
    return int(v) if v not in (None,"") else default

def _env_str(k, default):
    v = os.getenv(k)
    return v if v not in (None,"") else default

# ── CONFIG ────────────────────────────────────────────────────────────────────
MASK_DIR   = _env_str("MASK_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
OUT_DIR    = _env_str("OUT_DIR",  "/projects/prjs1578/futurewetgde/wetGDEs_area_scenarios")
LOG_DIR    = _env_str("LOG_DIR",  f"{OUT_DIR}/logs")

# scenarios, prefer single SCENARIO env
if _env_str("SCENARIO", None):
    SCENARIOS = [_env_str("SCENARIO", "")]
elif _env_str("SCENARIOS", None):
    SCENARIOS = [s.strip() for s in os.getenv("SCENARIOS").split(",") if s.strip()]
else:
    SCENARIOS = ["historical","ssp126","ssp370","ssp585"]

BIOME_SHP  = _env_str("BIOME_SHP", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
QA_DIR     = _env_str("QA_DIR",    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/")

ENGINE         = _env_str("XR_ENGINE", "netcdf4")
TILE_Y         = _env_int("TILE_Y", 2048)
TILE_X         = _env_int("TILE_X", 2048)
TIME_BATCH     = _env_int("TIME_BATCH", 24)
WET_THRESHOLD  = _env_str("WET_THRESHOLD", "gt0")   # gt0, ge0.25, ge0.5, eq1

SMALL_TEST     = _env_bool("SMALL_TEST", True)
APPLY_QA_MASK  = _env_bool("APPLY_QA_MASK", True)
APPLY_AG_MASK  = _env_bool("APPLY_AG_MASK", True)

AG_RULE        = _env_str("AG_RULE", "exclude_when_ag")  # or exclude_after_conversion
AG_BASE_YEAR   = _env_int("AG_BASE_YEAR", 2000)

LUH2_SSP_ROOT  = _env_str("LUH2_SSP_ROOT", "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp")
LUH2_MAP = {
    "historical": "Historic_Data/states.nc",
    "ssp126":    "RCP2_6_SSP1_from_IMAGE/states.nc",
    "ssp245":    "RCP4_5_SSP2_from_MESSAGE_GLOBIOM/states.nc",
    "ssp370":    "RCP7_0_SSP3_from_AIM/states.nc",
    "ssp434":    "RCP3_4_SSP4_from_GCAM/states.nc",
    "ssp460":    "RCP6_0_SSP4_from_GCAM/states.nc",
    "ssp585":    "RCP8_5_SSP5_from_REMIND_MAGPIE/states.nc",
}

# ── Counterfactual controls ───────────────────────────────────────────────────
# Families:
#   full          → normal run (evolving wetGDE + evolving LUH2)
#   climate_only  → evolving wetGDE + LUH2 frozen to FIX_AG_YEAR
#   landuse_only  → LUH2 evolving + wetGDE frozen to historical monthly climatology
COUNTERFACTUAL   = _env_str("COUNTERFACTUAL", "full")    # full|climate_only|landuse_only
RUN_TAG_DEFAULT  = _env_str("RUN_TAG", COUNTERFACTUAL)   # used in output filenames
FIX_AG_YEAR      = _env_int("FIX_AG_YEAR", 2000)         # for climate_only
HIST_WET_FILE    = _env_str("HIST_WET_FILE", "")         # for landuse_only
HIST_CLIM_WINDOW = _env_str("HIST_CLIM_WINDOW", "1985-01-01,2014-12-31")

# NetCDF output
WRITE_NC            = _env_bool("WRITE_NC", True)
WRITE_NC_STREAMING  = _env_bool("WRITE_NC_STREAMING", True)  # live append per TIME_BATCH
NC_TRANSIENT        = _env_bool("NC_TRANSIENT", False)       # write to NC_TMPDIR if True
NC_TMPDIR           = _env_str("NC_TMPDIR", "/scratch")
NC_ENGINE           = "netcdf4"

# NetCDF metadata
NC_TITLE       = _env_str("NC_TITLE", "Global wetGDE area by biome × realm with agricultural exclusions")
NC_INSTITUTION = _env_str("NC_INSTITUTION", "Utrecht University")
NC_AUTHOR      = _env_str("NC_AUTHOR", "Nicole Gyakowah Otoo <n.g.otoo@uu.nl>")
NC_DESCRIPTION = _env_str("NC_DESCRIPTION",
    "Monthly wetland GDE area aggregated by WWF terrestrial biome × realm on a regular lon lat grid. "
    "Values are area in km^2 after QA masking and agricultural exclusion experiments, none, crops, crops_pasture, derived from LUH2 states.")

# Parquet
PARQUET_CODEC            = _env_str("PARQUET_CODEC", "snappy")
WRITE_PARQUET_STREAMING  = _env_bool("WRITE_PARQUET_STREAMING", True)  # write one part per batch
PARQUET_LIVE_DIR         = _env_str("PARQUET_LIVE_DIR", OUT_DIR)       # dataset root for parts
WRITE_PARQUET_FINAL      = _env_bool("WRITE_PARQUET_FINAL", True)      # final per-scenario wide file

# Optional tiny ncview preview map
PREVIEW_MAP              = _env_bool("PREVIEW_MAP", False)
PREVIEW_EVERY_N_MONTHS   = _env_int("PREVIEW_EVERY_N_MONTHS", 12)
PREVIEW_EXCLUSION        = _env_str("PREVIEW_EXCLUSION", "crops")  # none|crops|crops_pasture

# ── logging ───────────────────────────────────────────────────────────────────
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("gde_area_by_biome_realm_monthly_tiled")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
fh = logging.FileHandler(f"{LOG_DIR}/area_monthly_tiled.log"); fh.setFormatter(fmt); logger.addHandler(fh)

t0 = time.time()
logger.info(
    ("START MASK_DIR=%s OUT_DIR=%s SCENARIOS=%s SMALL_TEST=%s APPLY_QA_MASK=%s APPLY_AG_MASK=%s "
     "WET_THRESHOLD=%s AG_RULE=%s AG_BASE_YEAR=%d LUH2_SSP_ROOT=%s XR_ENGINE=%s | "
     "COUNTERFACTUAL=%s RUN_TAG=%s FIX_AG_YEAR=%s HIST_WET_FILE=%s HIST_CLIM_WINDOW=%s"),
    MASK_DIR, OUT_DIR, ",".join(SCENARIOS), SMALL_TEST, APPLY_QA_MASK, APPLY_AG_MASK,
    WET_THRESHOLD, AG_RULE, AG_BASE_YEAR, LUH2_SSP_ROOT, ENGINE,
    COUNTERFACTUAL, RUN_TAG_DEFAULT, str(FIX_AG_YEAR), HIST_WET_FILE or "(default)", HIST_CLIM_WINDOW
)

# ── helpers ───────────────────────────────────────────────────────────────────
def _std(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords:  ds = ds.rename({"latitude":"lat"})
    if "longitude" in ds.coords: ds = ds.rename({"longitude":"lon"})
    return ds

def binarize_wet(arr: np.ndarray, thr_mode: str) -> np.ndarray:
    if thr_mode == "gt0":    return (arr > 0).astype(np.float32)
    if thr_mode == "ge0.5":  return (arr >= 0.5).astype(np.float32)
    if thr_mode == "ge0.25": return (arr >= 0.25).astype(np.float32)
    return (arr == 1).astype(np.float32)

def luh2_file_for_scenario(scen: str) -> str | None:
    rel = LUH2_MAP.get(scen)
    if not rel:
        return None
    path = os.path.join(LUH2_SSP_ROOT, rel)
    return path if os.path.isfile(path) else None

def open_qa_merged(qa_paths):
    base = _std(xr.open_dataset(qa_paths[0], decode_times=False, mask_and_scale=False))
    qa_lat = base["lat"].values
    qa_lon = base["lon"].values
    ny, nx = qa_lat.size, qa_lon.size
    qa_vars = {}
    for p in qa_paths:
        ds = _std(xr.open_dataset(p, decode_times=False, mask_and_scale=False))
        if (ds.sizes.get("lat") != ny) or (ds.sizes.get("lon") != nx) or \
           (not np.array_equal(ds["lat"].values, qa_lat)) or \
           (not np.array_equal(ds["lon"].values, qa_lon)):
            ds = ds.interp(lat=qa_lat, lon=qa_lon, method="nearest")
        for v in ds.data_vars:
            dv = ds[v]
            if "time" in dv.dims:
                dv = dv.isel(time=0, drop=True)
            qa_vars[v] = dv
    return xr.Dataset(qa_vars, coords={"lat": qa_lat, "lon": qa_lon})

# fast nearest neighbor regrid indices
def _nearest_index(src, tgt):
    idx = np.searchsorted(src, tgt)
    idx = np.clip(idx, 1, len(src)-1)
    left = src[idx-1]; right = src[idx]
    use_left = (tgt - left) <= (right - tgt)
    return (idx - use_left.astype(np.int32)).astype(np.int32)

def build_regrid_indices(src_lat, src_lon, qa_lat, qa_lon):
    lat_src = src_lat if src_lat[0] < src_lat[-1] else src_lat[::-1]
    lat_rev = (src_lat[0] > src_lat[-1])
    lat_idx = _nearest_index(lat_src, qa_lat)
    if lat_rev:
        lat_idx = (len(src_lat) - 1) - lat_idx
    lon_idx = _nearest_index(src_lon, qa_lon)
    return lat_idx, lon_idx

# LUH2 loader, avoid CF calendar decoding, work in integer years
def load_luh2_crops_pastr(states_files, engine="netcdf4"):
    if not states_files:
        return None, None
    ds = xr.open_mfdataset(states_files, combine="by_coords", decode_times=False,
                           engine=engine, mask_and_scale=False)
    ds = _std(ds)
    units = str(ds["time"].attrs.get("units", "years since 2015-01-01"))
    m = re.search(r"years\s+since\s*(\d{1,4})", units)
    base_year = int(m.group(1)) if m else 2015
    offs = np.rint(np.asarray(ds["time"].values)).astype(int)
    yrs = base_year + offs
    ds = ds.assign_coords(time=("time", yrs.astype("int32")))
    ds["time"].attrs.clear()

    need = ["c3ann","c4ann","c3per","c4per","c3nfx","pastr"]
    miss = [v for v in need if v not in ds.data_vars]
    if miss:
        raise KeyError(f"LUH2 states missing variables: {miss}")

    crops = (ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]).fillna(0).clip(0,1)
    pastr = ds["pastr"].fillna(0).clip(0,1)
    return crops, pastr

def load_monthly_climatology(hist_path: str, window: str, engine="netcdf4") -> xr.DataArray:
    """Monthly mean wetness (12, lat, lon) from historical file over date window."""
    if not hist_path or not os.path.isfile(hist_path):
        raise FileNotFoundError(f"HIST_WET_FILE missing: {hist_path}")
    ds = xr.open_dataset(hist_path, decode_times=True, engine=engine, mask_and_scale=False)
    ds = _std(ds)
    var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var].astype("float32")
    try:
        start, end = [pd.to_datetime(s) for s in window.split(",")]
        da = da.sel(time=slice(start, end))
    except Exception:
        pass
    clim = da.groupby("time.month").mean("time", skipna=True).astype("float32")
    clim = clim.assign_coords(month=np.arange(1, 13, dtype="int32"))
    return clim  # dims: month, lat, lon

# ── preflight LUH2 ───────────────────────────────────────────────────────────
if APPLY_AG_MASK:
    missing = [s for s in SCENARIOS if not luh2_file_for_scenario(s)]
    if missing:
        raise FileNotFoundError("APPLY_AG_MASK=True, LUH2 states.nc missing for: " + ", ".join(missing))
    logger.info("Preflight LUH2 OK for all scenarios: %s", ", ".join(SCENARIOS))
else:
    logger.info("APPLY_AG_MASK=False, running without LUH2 exclusion")

# ── QA grid and masks ────────────────────────────────────────────────────────
qa_paths = sorted([os.path.join(QA_DIR, f) for f in os.listdir(QA_DIR) if f.endswith(".nc")])
if not qa_paths:
    raise FileNotFoundError(f"No .nc files in {QA_DIR}")
qa = open_qa_merged(qa_paths)

qa_lat = qa["lat"].values
qa_lon = qa["lon"].values
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

# pixel areas, km^2 on QA grid
R = 6_371_000.0
lat_res = float(abs(qa_lat[1] - qa_lat[0]))
lon_res = float(abs(qa_lon[1] - qa_lon[0]))
dlam = lon_res / 360.0
band = np.sin(np.deg2rad(qa_lat + lat_res/2.0)) - np.sin(np.deg2rad(qa_lat - lat_res/2.0))
area_per_lat = (2.0 * np.pi * R**2) * band * dlam / 1e6
pixel_area = np.repeat(area_per_lat[:, None], nx, axis=1).astype("float32")

# ── Biome×realm raster on QA grid ────────────────────────────────────────────
gdf = gpd.read_file(BIOME_SHP).set_crs("EPSG:4326", allow_override=True)
if "BIOME" not in gdf.columns or "REALM" not in gdf.columns:
    raise RuntimeError("Shapefile must contain BIOME and REALM fields")
gdf = gdf.loc[gdf["BIOME"].between(1, 14), ["BIOME","REALM","geometry"]].copy()
gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)

unique_combos = sorted(gdf["BIOME_ID_REALM"].unique())
combo_to_code = {cmb: i+1 for i, cmb in enumerate(unique_combos)}  # 0 = outside
code_to_combo = {v:k for k,v in combo_to_code.items()}
n_codes = len(combo_to_code)

# rasterize with pixel edge transform aligned to cell centers
transform = Affine(lon_res, 0, qa_lon.min() - lon_res/2.0, 0, -lat_res, qa_lat.max() + lat_res/2.0)
shapes = ((geom, combo_to_code[cmb]) for geom, cmb in zip(gdf.geometry, gdf["BIOME_ID_REALM"]))
codes_arr = rasterio.features.rasterize(
    shapes, out_shape=(ny, nx), transform=transform, fill=0, dtype="int32"
)
if APPLY_QA_MASK:
    codes_arr = np.where(qa_mask, codes_arr, 0).astype(np.int32)

# ── NetCDF streaming helpers ─────────────────────────────────────────────────
def _nc_path_for_scenario(base_prefix: str, scenario: str) -> str:
    name = f"{base_prefix}_{scenario}.nc"
    if NC_TRANSIENT:
        dir_ = NC_TMPDIR if NC_TMPDIR and os.path.isdir(NC_TMPDIR) else OUT_DIR
        Path(dir_).mkdir(parents=True, exist_ok=True)
        return os.path.join(dir_, name)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    return os.path.join(OUT_DIR, name)

def nc_stream_open(path_nc: str, biomes: list[str], exps: list[str], scenario: str, run_tag: str):
    mode = "a" if os.path.exists(path_nc) else "w"
    ds = Dataset(path_nc, mode, format="NETCDF4_CLASSIC")

    if mode == "w":
        # streaming dims
        ds.createDimension("time", None)  # unlimited
        ds.createDimension("biome_realm", len(biomes))
        ds.createDimension("ag_exclusion", len(exps))
        # static map dims so ncview shows a globe
        ds.createDimension("lat", ny)
        ds.createDimension("lon", nx)

        # time
        tvar = ds.createVariable("time", "f8", ("time",))
        tvar.units = "days since 1900-01-01 00:00:00"
        tvar.calendar = "standard"
        tvar.standard_name = "time"

        # category coords with labels in attributes
        bvar = ds.createVariable("biome_realm", "i4", ("biome_realm",))
        bvar[:] = np.arange(len(biomes), dtype=np.int32)
        bvar.long_name = "WWF terrestrial biome x realm index"
        bvar.flag_values = np.arange(len(biomes), dtype=np.int32)
        bvar.flag_meanings = " ".join(biomes)

        evar = ds.createVariable("ag_exclusion", "i4", ("ag_exclusion",))
        evar[:] = np.arange(len(exps), dtype=np.int32)
        evar.long_name = "agricultural exclusion experiment index"
        evar.flag_values = np.arange(len(exps), dtype=np.int32)
        evar.flag_meanings = " ".join(exps)

        # static map variables
        latv = ds.createVariable("lat", "f4", ("lat",))
        latv[:] = qa_lat
        latv.standard_name = "latitude"
        latv.units = "degrees_north"

        lonv = ds.createVariable("lon", "f4", ("lon",))
        lonv[:] = qa_lon
        lonv.standard_name = "longitude"
        lonv.units = "degrees_east"

        codev = ds.createVariable("biome_realm_code", "i4", ("lat","lon"), zlib=True, complevel=1, fill_value=0)
        codev[:] = codes_arr
        codev.long_name = "WWF terrestrial biome × realm code (0=outside)"
        codev.flag_values = np.arange(0, n_codes+1, dtype=np.int32)
        codev.flag_meanings = "none " + " ".join(unique_combos)

        # main streamed variable (no lat,lon)
        v = ds.createVariable(
            "gdeareakm2", "f4",
            ("time","biome_realm","ag_exclusion"),
            zlib=True, complevel=0,
            chunksizes=(max(1, min(24, TIME_BATCH)), len(biomes), 1),
            fill_value=np.nan
        )
        v.long_name = "WetGDE area aggregated by biome x realm with QA and agricultural exclusion"
        v.units = "km2"
        v.cell_methods = "time: point"

        # optional preview variable
        if PREVIEW_MAP:
            ds.createDimension("time_preview", None)
            tp = ds.createVariable("time_preview", "f8", ("time_preview",))
            tp.units = "days since 1900-01-01 00:00:00"
            tp.calendar = "standard"
            vprev = ds.createVariable(
                "gdeareakm2_preview", "f4",
                ("time_preview", "lat", "lon"),
                zlib=True, complevel=1, fill_value=np.nan
            )
            vprev.long_name = "Preview map of gdeareakm2, broadcast per biome×realm"
            vprev.units = "km2"
        else:
            tp = vprev = None

        # global attrs
        ds.title = NC_TITLE
        ds.institution = NC_INSTITUTION
        ds.source = "wetGDE monthly masks + LUH2 states"
        ds.history = f"streaming created {pd.Timestamp.utcnow().isoformat()}Z"
        ds.author = NC_AUTHOR
        ds.description = NC_DESCRIPTION
        ds.notes = ("Integer category coords with label mappings in coord attributes. "
                    "Time is monthly end of month.")
        ds.Conventions = "CF-1.8"
        ds.scenario = scenario
        ds.run_tag = run_tag
    else:
        tvar = ds.variables["time"]
        v = ds.variables["gdeareakm2"]
        tp = ds.variables["time_preview"] if ("time_preview" in ds.variables and PREVIEW_MAP) else None
        vprev = ds.variables["gdeareakm2_preview"] if ("gdeareakm2_preview" in ds.variables and PREVIEW_MAP) else None

    return ds, tvar, v, tp, vprev

def nc_stream_write_batch(nc_ds, tvar, vvar, times_slice: np.ndarray, sums_batch: dict,
                          tp=None, vprev=None, t0_idx_global=0):
    py_dt = pd.to_datetime(times_slice).to_pydatetime()
    t0i = int(tvar.shape[0])
    n = len(py_dt)
    tvar[t0i:t0i+n] = date2num(py_dt, tvar.units, tvar.calendar)

    arr_none  = sums_batch["none"][:, 1:]
    arr_crops = sums_batch["crops"][:, 1:]
    arr_cp    = sums_batch["crops_pasture"][:, 1:]
    data_out  = np.stack([arr_none, arr_crops, arr_cp], axis=2)  # (batch, n_codes, 3)
    vvar[t0i:t0i+n, :, :] = data_out

    if PREVIEW_MAP and (tp is not None) and (vprev is not None):
        which = PREVIEW_EXCLUSION
        sel = sums_batch[which]  # (batch, n_codes+1)
        for k in range(sel.shape[0]):
            global_t = t0_idx_global + k
            if global_t % PREVIEW_EVERY_N_MONTHS != 0:
                continue
            vals = sel[k, :]        # totals per biome code
            grid = vals[codes_arr]  # broadcast to grid
            tpi = int(tp.shape[0])
            dt = pd.to_datetime(times_slice[k]).to_pydatetime()
            tp[tpi:tpi+1] = date2num([dt], tp.units, "standard")
            vprev[tpi, :, :] = grid

    nc_ds.sync()

# ── Parquet writers ───────────────────────────────────────────────────────────
def _wide_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    w = (df_long.pivot_table(index=["time","BIOME_ID_REALM"],
                             columns="ag_exclusion",
                             values="area_km2",
                             aggfunc="first")
                   .rename(columns={
                       "none":"area_none_km2",
                       "crops":"area_crops_excl_km2",
                       "crops_pasture":"area_crops_pasture_excl_km2"})
                   .reset_index())
    for col in ["area_none_km2","area_crops_excl_km2","area_crops_pasture_excl_km2"]:
        if col not in w.columns:
            w[col] = np.nan
    return w[["time","BIOME_ID_REALM","area_none_km2","area_crops_excl_km2","area_crops_pasture_excl_km2"]]

def _write_parquet_wide_batch(scenario: str, df_batch: pd.DataFrame,
                              t0: pd.Timestamp, t1: pd.Timestamp, run_tag: str):
    if df_batch.empty:
        return
    w = _wide_from_long(df_batch)
    w["year"] = pd.to_datetime(w["time"]).dt.year
    root = os.path.join(PARQUET_LIVE_DIR, f"gde_area_by_biome_realm_monthly_{run_tag}_{scenario}")
    Path(root).mkdir(parents=True, exist_ok=True)
    for yr, wy in w.groupby("year"):
        outdir = os.path.join(root, f"year={yr}")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fname = f"part_{pd.to_datetime(t0).strftime('%Y%m')}_{pd.to_datetime(t1).strftime('%Y%m')}.parquet"
        wy.drop(columns=["year"]).to_parquet(os.path.join(outdir, fname), index=False, compression=PARQUET_CODEC)

def save_parquet_wide_per_scenario(df: pd.DataFrame, out_dir: str, base_prefix: str):
    for scen, d in df.groupby("scenario"):
        w = _wide_from_long(d)
        out_pq = os.path.join(out_dir, f"{base_prefix}_{scen}.parquet")
        w.to_parquet(out_pq, index=False, compression=PARQUET_CODEC)
        logger.info("Parquet written (wide final): %s rows=%d", out_pq, len(w))

# ── processing ────────────────────────────────────────────────────────────────
def process_scenario(scenario: str, counterfactual: str, run_tag: str) -> pd.DataFrame:
    path_nc_in = os.path.join(MASK_DIR, f"wetGDE_{scenario}.nc")
    if not os.path.exists(path_nc_in):
        logger.warning("Missing file: %s  skipping", path_nc_in)
        return pd.DataFrame()

    try:
        ds = xr.open_dataset(path_nc_in, decode_times=True, engine=ENGINE, mask_and_scale=False)
    except Exception:
        ds = xr.open_dataset(path_nc_in, decode_times=False, engine=ENGINE, mask_and_scale=False)
        ds = ds.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))

    var = "wetGDE" if "wetGDE" in ds.data_vars else list(ds.data_vars)[0]
    wet = _std(ds[var].to_dataset(name="wet"))["wet"].astype("float32")

    if SMALL_TEST:
        # keep full time in Dataset but we'll only iterate the FIRST batch below
        logger.info("TEST mode active: will process first TIME_BATCH=%d months and first spatial tile only", TIME_BATCH)

    times = pd.to_datetime(wet["time"].values)
    logger.info("WetGDE time coverage (%s): %s .. %s  n=%d", counterfactual, times[0].date(), times[-1].date(), len(times))

    # regrid indices to QA grid
    same_wet_lat = np.array_equal(wet["lat"].values, qa_lat)
    same_wet_lon = np.array_equal(wet["lon"].values, qa_lon)
    if not (same_wet_lat and same_wet_lon):
        wet_lat_idx, wet_lon_idx = build_regrid_indices(wet["lat"].values, wet["lon"].values, qa_lat, qa_lon)

    # landuse_only: precompute 12-month wet climatology on the WET native grid
    wet_clim = None
    if counterfactual == "landuse_only":
        try:
            hist_file = HIST_WET_FILE or os.path.join(MASK_DIR, "wetGDE_historical.nc")
            wet_clim_da = load_monthly_climatology(hist_file, HIST_CLIM_WINDOW, engine=ENGINE)
            if not (np.array_equal(wet_clim_da["lat"].values, wet["lat"].values) and
                    np.array_equal(wet_clim_da["lon"].values, wet["lon"].values)):
                wet_clim_da = wet_clim_da.interp(lat=wet["lat"], lon=wet["lon"], method="nearest")
            wet_clim = wet_clim_da.astype("float32").values  # (12, y, x)
            logger.info("landuse_only: loaded monthly wet climatology [%s | %s]", hist_file, HIST_CLIM_WINDOW)
        except Exception as e:
            logger.warning("landuse_only: failed to load climatology (%s) -> reverting to full", e)
            wet_clim = None
            counterfactual = "full"

    # LUH2 to time axis
    crops_t = pastr_t = None
    if APPLY_AG_MASK:
        f = luh2_file_for_scenario(scenario)
        logger.info("LUH2 file for %s: %s", scenario, f or "NONE")
        if f:
            crops_full, pastr_full = load_luh2_crops_pastr([f], engine=ENGINE)
            years_wet = xr.DataArray(pd.to_datetime(times).year.astype("int32"), dims=("time",))

            if counterfactual == "climate_only":
                # freeze ag to FIX_AG_YEAR across all months
                yfix = int(FIX_AG_YEAR)
                fix_c = crops_full.sel(time=yfix, method="nearest").astype("float32")
                fix_p = pastr_full.sel(time=yfix, method="nearest").astype("float32")
                tile_shape = (len(years_wet), fix_c.shape[0], fix_c.shape[1])
                crops_t = xr.DataArray(np.broadcast_to(fix_c.values, tile_shape),
                                       dims=("time","lat","lon"),
                                       coords={"time": years_wet, "lat": crops_full["lat"], "lon": crops_full["lon"]})
                pastr_t = xr.DataArray(np.broadcast_to(fix_p.values, tile_shape),
                                       dims=("time","lat","lon"),
                                       coords={"time": years_wet, "lat": crops_full["lat"], "lon": crops_full["lon"]})
                logger.info("climate_only: LUH2 frozen to year=%d", yfix)
            else:
                crops_t = crops_full.reindex(time=years_wet, method="nearest", tolerance=1)
                pastr_t = pastr_full.reindex(time=years_wet, method="nearest", tolerance=1)

            if AG_RULE == "exclude_after_conversion":
                mask_since = (years_wet >= AG_BASE_YEAR)
                crops_since = xr.where(mask_since, crops_t, 0)
                pastr_since = xr.where(mask_since, pastr_t, 0)
                crops_t = (crops_since > 0).astype("uint8").cumsum("time").clip(0,1)
                pastr_t = (pastr_since > 0).astype("uint8").cumsum("time").clip(0,1)

            same_ag_lat = np.array_equal(crops_full["lat"].values, qa_lat)
            same_ag_lon = np.array_equal(crops_full["lon"].values, qa_lon)
            if not (same_ag_lat and same_ag_lon):
                ag_lat_idx, ag_lon_idx = build_regrid_indices(crops_full["lat"].values, crops_full["lon"].values, qa_lat, qa_lon)

            valid_per_t = xr.apply_ufunc(np.isfinite, crops_t, dask="allowed").any(dim=("lat","lon"))
            total = valid_per_t.sum()
            if hasattr(total, "compute"):
                total = total.compute()
            if hasattr(total, "values"):
                total = total.values
            n_ok = int(np.asarray(total))
            logger.info("LUH2 year matching available for %d/%d months", n_ok, len(times))
        else:
            logger.warning("No LUH2 states for %s, skipping ag exclusion", scenario)

    exps = ["none","crops","crops_pasture"]

    # NetCDF stream open
    nc_ds = tvar_nc = v_nc = tp_nc = vprev_nc = None
    if WRITE_NC and WRITE_NC_STREAMING:
        base = f"gde_area_by_biome_realm_monthly_{run_tag}"
        nc_path = _nc_path_for_scenario(base, scenario)
        nc_ds, tvar_nc, v_nc, tp_nc, vprev_nc = nc_stream_open(nc_path, unique_combos, exps, scenario, run_tag)

    wet_src_all = wet.astype("float32")
    all_records = []

    # small-test spatial limits: first tile only
    y_starts = range(0, ny, TILE_Y) if not SMALL_TEST else [0]
    x_starts = range(0, nx, TILE_X) if not SMALL_TEST else [0]

    for t0_idx in range(0, len(times), TIME_BATCH):
        t1_idx = min(t0_idx + TIME_BATCH, len(times))
        batch_len = t1_idx - t0_idx

        wet_tb = wet_src_all.isel(time=slice(t0_idx, t1_idx)).values
        sums_batch = {e: np.zeros((batch_len, n_codes + 1), dtype=np.float64) for e in exps}

        if crops_t is not None:
            crops_tb = crops_t.isel(time=slice(t0_idx, t1_idx)).astype("float32").values
            pastr_tb = pastr_t.isel(time=slice(t0_idx, t1_idx)).astype("float32").values

        for y0 in y_starts:
            y1 = min(y0 + TILE_Y, ny)
            lat_idx_tile = slice(y0, y1) if (same_wet_lat and same_wet_lon) else wet_lat_idx[y0:y1]
            for x0 in x_starts:
                x1 = min(x0 + TILE_X, nx)
                lon_idx_tile = slice(x0, x1) if (same_wet_lon and same_wet_lat) else wet_lon_idx[x0:x1]

                if same_wet_lat and same_wet_lon:
                    w = wet_tb[:, y0:y1, x0:x1]
                else:
                    w = np.take(np.take(wet_tb, lat_idx_tile, axis=1), lon_idx_tile, axis=2)

                # landuse_only: override w with monthly climatology prior to binarization
                if (counterfactual == "landuse_only") and (wet_clim is not None):
                    months_batch = pd.DatetimeIndex(times[t0_idx:t1_idx]).month.values  # 1..12
                    if same_wet_lat and same_wet_lon:
                        for k, m in enumerate(months_batch):
                            w[k, :, :] = wet_clim[m-1, y0:y1, x0:x1]
                    else:
                        for k, m in enumerate(months_batch):
                            w[k, :, :] = np.take(
                                np.take(wet_clim[m-1], lat_idx_tile, axis=0),
                                lon_idx_tile, axis=1
                            )

                w = binarize_wet(w, WET_THRESHOLD)

                codes_tile = codes_arr[y0:y1, x0:x1]
                area_tile  = pixel_area[y0:y1, x0:x1]
                codes_flat = codes_tile.ravel()
                area_flat  = area_tile.ravel()

                if APPLY_AG_MASK and (crops_t is not None):
                    if 'ag_lat_idx' in locals():
                        c_tile = np.take(np.take(crops_tb, ag_lat_idx[y0:y1], axis=1), ag_lon_idx[x0:x1], axis=2)
                        p_tile = np.take(np.take(pastr_tb, ag_lat_idx[y0:y1], axis=1), ag_lon_idx[x0:x1], axis=2)
                    else:
                        c_tile = crops_tb[:, y0:y1, x0:x1]
                        p_tile = pastr_tb[:, y0:y1, x0:x1]
                else:
                    c_tile = p_tile = None

                for k in range(w.shape[0]):
                    wet_flat = w[k].ravel()
                    weights_base = np.nan_to_num(area_flat * wet_flat, nan=0.0, posinf=0.0, neginf=0.0)
                    if weights_base.sum() == 0.0:
                        continue

                    bc = np.bincount(codes_flat, weights=weights_base, minlength=n_codes + 1)
                    sums_batch["none"][k, :bc.size] += bc

                    if c_tile is None:
                        sums_batch["crops"][k, :bc.size] += bc
                        sums_batch["crops_pasture"][k, :bc.size] += bc
                    else:
                        crops_f = np.nan_to_num(c_tile[k].ravel(), 0.0).clip(0, 1)
                        pastr_f = np.nan_to_num(p_tile[k].ravel(), 0.0).clip(0, 1)
                        bc = np.bincount(codes_flat, weights=weights_base * (1.0 - crops_f), minlength=n_codes + 1)
                        sums_batch["crops"][k, :bc.size] += bc
                        cp = np.clip(crops_f + pastr_f, 0.0, 1.0)
                        bc = np.bincount(codes_flat, weights=weights_base * (1.0 - cp), minlength=n_codes + 1)
                        sums_batch["crops_pasture"][k, :bc.size] += bc

        # build batch df (long) once
        batch_records = []
        for e in ("none","crops","crops_pasture"):
            sarr = sums_batch[e]
            for k in range(sarr.shape[0]):
                ts = times[t0_idx + k]
                nz = np.nonzero(sarr[k, 1:] > 0.0)[0] + 1
                for code in nz:
                    batch_records.append((scenario, ts, code_to_combo[code], e, float(sarr[k, code])))
        df_batch = pd.DataFrame(batch_records, columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])
        all_records.extend(batch_records)

        # streaming parquet part
        if WRITE_PARQUET_STREAMING:
            _write_parquet_wide_batch(scenario, df_batch, times[t0_idx], times[t1_idx-1], run_tag=run_tag)

        # streaming NetCDF append
        if WRITE_NC and WRITE_NC_STREAMING:
            nc_stream_write_batch(nc_ds, tvar_nc, v_nc, times[t0_idx:t1_idx], sums_batch,
                                  tp=tp_nc, vprev=vprev_nc, t0_idx_global=t0_idx)

        logger.info("Scenario %s [%s], appended %d/%d timesteps", scenario, run_tag, t1_idx, len(times))

        if SMALL_TEST:
            logger.info("TEST mode, stopping after first batch (time) and first spatial tile(s)")
            break  # only first TIME_BATCH months

    if WRITE_NC and WRITE_NC_STREAMING and (nc_ds is not None):
        nc_ds.close()

    if not all_records:
        return pd.DataFrame(columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])

    df = pd.DataFrame.from_records(all_records, columns=["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"])
    df = df.sort_values(["scenario","time","BIOME_ID_REALM","ag_exclusion"]).reset_index(drop=True)
    return df

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # In small-test mode, run ALL families to see end-to-end behavior quickly.
    families = ["full","climate_only","landuse_only"] if SMALL_TEST else [COUNTERFACTUAL]

    for fam in families:
        run_tag = RUN_TAG_DEFAULT if not SMALL_TEST else fam
        all_parts = []
        for scen in SCENARIOS:
            if not scen:
                continue
            df_scen = process_scenario(scen, counterfactual=fam, run_tag=run_tag)
            if not df_scen.empty:
                all_parts.append(df_scen)

        if all_parts:
            df = pd.concat(all_parts, ignore_index=True)
            base = f"gde_area_by_biome_realm_monthly_{run_tag}"

            if WRITE_PARQUET_FINAL:
                save_parquet_wide_per_scenario(df, OUT_DIR, base)

            # optional non-streaming NC build from final df, usually not needed
            if WRITE_NC and not WRITE_NC_STREAMING:
                scen_tag = SCENARIOS[0] if len(SCENARIOS) == 1 else "all"
                # build dense cube once using xarray
                scenarios = np.array(sorted(df["scenario"].unique()), dtype=object).astype(str)
                biomes    = np.array(sorted(df["BIOME_ID_REALM"].unique()), dtype=object).astype(str)
                exps      = np.array(sorted(df["ag_exclusion"].unique()), dtype=object).astype(str)
                times_all = np.array(sorted(pd.to_datetime(df["time"].unique())))
                scen_ids  = np.arange(len(scenarios), dtype="int32")
                biome_ids = np.arange(len(biomes), dtype="int32")
                exp_ids   = np.arange(len(exps), dtype="int32")
                idx_t = {t:i for i,t in enumerate(times_all)}
                idx_s = {s:i for i,s in enumerate(scenarios)}
                idx_b = {b:i for i,b in enumerate(biomes)}
                idx_e = {e:i for i,e in enumerate(exps)}
                data = np.full((len(times_all), len(scenarios), len(biomes), len(exps)), np.nan, dtype="float32")
                for s, t, b, e, a in df[["scenario","time","BIOME_ID_REALM","ag_exclusion","area_km2"]].itertuples(index=False):
                    data[idx_t[pd.to_datetime(t)], idx_s[str(s)], idx_b[str(b)], idx_e[str(e)]] = a
                ds = xr.Dataset(
                    {"gdeareakm2": (("time","scenario","biome_realm","ag_exclusion"), data)},
                    coords={"time": times_all, "scenario": scen_ids, "biome_realm": biome_ids, "ag_exclusion": exp_ids},
                    attrs={"title": NC_TITLE, "institution": NC_INSTITUTION, "source": "wetGDE monthly masks + LUH2 states",
                           "history": f"created {pd.Timestamp.utcnow().isoformat()}Z", "author": NC_AUTHOR,
                           "description": NC_DESCRIPTION, "Conventions": "CF-1.8", "run_tag": run_tag}
                )
                ds["scenario"].attrs.update({"long_name":"scenario index","flag_values":scen_ids,"flag_meanings":" ".join(scenarios)})
                ds["biome_realm"].attrs.update({"long_name":"WWF terrestrial biome x realm index","flag_values":biome_ids,"flag_meanings":" ".join(biomes)})
                ds["ag_exclusion"].attrs.update({"long_name":"agricultural exclusion experiment index","flag_values":exp_ids,"flag_meanings":" ".join(exps)})
                ds["gdeareakm2"].attrs.update({"long_name":"WetGDE area aggregated by biome x realm with QA and agricultural exclusion","units":"km2","cell_methods":"time: point"})
                enc = {"gdeareakm2": {"zlib": True, "complevel": 0, "chunksizes": (min(len(times_all),24),1,1,1)},
                       "time": {"units": "days since 1900-01-01 00:00:00", "calendar": "standard"}}
                out_path = os.path.join(OUT_DIR, f"{base}_{scen_tag}.nc")
                ds.to_netcdf(out_path, engine=NC_ENGINE, format="NETCDF4_CLASSIC", encoding=enc)
                logger.info("NetCDF written (final cube): %s", out_path)
        else:
            logger.warning("No scenario produced output for run_tag=%s", run_tag)

    logger.info("DONE in %.1f s", time.time() - t0)

if __name__ == "__main__":
    main()