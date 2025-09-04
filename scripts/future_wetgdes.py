# ######Original without satAreaFrac variable
# #!/usr/bin/env python3
# """
# wetGDE mask builder at ~1 km global for CMIP6 scenarios.

# Rule:
#   wetGDE = 1 where (satAreaFrac > 0.5) AND (WTD <= 5 m), else 0, 127 means missing inputs
# """

# import os
# import numpy as np
# import xarray as xr
# import dask
# from dask.diagnostics import ProgressBar
# from datetime import datetime

# # ---------------- Paths and parameters ----------------
# SAT_DIR     = "/projects/prjs1578/sat_future"
# MERGED_DIR  = "/projects/prjs1578/future_gw"
# OUT_DIR     = "/projects/prjs1578/futurewetgde/wetGDEs"
# os.makedirs(OUT_DIR, exist_ok=True)

# SCENARIOS   = ["historical", "ssp126", "ssp370", "ssp585"]
# SAT_VAR     = "satAreaFrac"
# WTD_MAIN    = "l1_wtd"
# WTD_FALLBACK= "l2_wtd"

# SAT_THRESHOLD = 0.5
# WTD_THRESHOLD = 5.0

# # large spatial chunks for faster IO, overridable via env SPATIAL_CHUNK
# SZ = int(os.environ.get("SPATIAL_CHUNK", "1024"))  # try 512 or 1024
# CHUNK_TARGETS = {"time": 1, "lat": SZ, "lon": SZ}

# ENGINE   = "netcdf4"
# NC_MODEL = "NETCDF4_CLASSIC"

# ADDITIONAL_AUTHOR = "Nicole Gyakowah Otoo"
# ADDITIONAL_EMAIL  = "n.g.otoo@uu.nl, nicholetylor@gmail.com"

# TEST        = os.environ.get("TEST", "false").lower() in ("true","1","yes","y")
# SMALL_TEST  = os.environ.get("SMALL_TEST", "false").lower() in ("true","1","yes","y")
# REGION_NAME = os.environ.get("REGION", "great_plains").strip().lower()
# ONLY_SCEN   = os.environ.get("SCENARIO", "").strip() or None
# PROGRESS_LOG= os.environ.get("PROGRESS_LOG", "").strip() or None  # file to stream Dask progress bar

# REGIONS = {
#     "great_plains": dict(lat=(25.0, 50.0),  lon=(-106.0, -94.0)),
#     "sahel":        dict(lat=(10.0, 20.0),  lon=(-20.0,   30.0)),
#     "amazon":       dict(lat=(-15.0, 5.0),  lon=(-75.0,  -50.0)),
#     "europe":       dict(lat=(36.0, 60.0),  lon=(-10.0,   30.0)),
#     "australia_se": dict(lat=(-40.0, -25.0),lon=(140.0,  155.0)),
# }

# dask.config.set({
#     "scheduler": "threads",
#     "array.slicing.split_large_chunks": True,
# })

# # ---------------- Helpers ----------------
# def _ascii(s): return str(s).encode("ascii", "ignore").decode("ascii")

# def standardize_coord_names(da: xr.DataArray) -> xr.DataArray:
#     rename = {}
#     if "latitude" in da.dims:  rename["latitude"]  = "lat"
#     if "longitude" in da.dims: rename["longitude"] = "lon"
#     if "y" in da.dims and "lat" not in da.dims: rename["y"] = "lat"
#     if "x" in da.dims and "lon" not in da.dims: rename["x"] = "lon"
#     return da.rename(rename) if rename else da

# def dataset_lon_range(da: xr.DataArray) -> str:
#     if "lon" not in da.dims: return "m180_180"
#     vmin = float(da["lon"].min()); vmax = float(da["lon"].max())
#     eps = 1e-6
#     return "0_360" if vmin >= -eps and vmax <= 360.0 + eps else "m180_180"

# def to_0_360(lon):    return xr.where(lon < 0, lon + 360.0, lon)
# def to_m180_180(lon): return ((lon + 180.0) % 360.0) - 180.0

# def unify_longitudes(da: xr.DataArray, target_range: str) -> xr.DataArray:
#     if "lon" not in da.dims: return da
#     lon = da["lon"]
#     vmin = float(lon.min()); vmax = float(lon.max())
#     if target_range == "0_360":
#         if vmin < 0 or vmax > 360:
#             da = da.assign_coords(lon=to_0_360(lon)).sortby("lon")
#     else:
#         if vmin < -180 or vmax > 180:
#             da = da.assign_coords(lon=to_m180_180(lon)).sortby("lon")
#     return da

# def subset_region(da: xr.DataArray, region_name: str) -> xr.DataArray:
#     if region_name not in REGIONS: region_name = "great_plains"
#     lat_lo, lat_hi = REGIONS[region_name]["lat"]
#     lon_lo, lon_hi = REGIONS[region_name]["lon"]
#     if "lat" in da.dims and np.any(np.diff(da["lat"].values) < 0): da = da.sortby("lat")
#     if "lon" in da.dims and np.any(np.diff(da["lon"].values) < 0): da = da.sortby("lon")
#     if "lon" in da.dims and dataset_lon_range(da) == "0_360":
#         lon_lo = lon_lo if lon_lo >= 0 else lon_lo + 360.0
#         lon_hi = lon_hi if lon_hi >= 0 else lon_hi + 360.0
#         if lon_lo <= lon_hi:
#             return da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
#         left  = da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, 360.0))
#         right = da.sel(lat=slice(lat_lo, lat_hi), lon=slice(0.0, lon_hi))
#         return xr.concat([left, right], dim="lon").sortby("lon")
#     return da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))

# def chunk_map(da: xr.DataArray):
#     out = {}
#     for d in da.dims:
#         size = int(da.sizes[d])
#         if size <= 0:
#             raise ValueError(f"Dimension {d} has size {size}")
#         target = int(CHUNK_TARGETS.get(d, size))
#         out[d] = max(1, min(target, size))
#     return out

# def sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
#     ds.attrs = {k: _ascii(v) for k, v in ds.attrs.items()}
#     ds.attrs.setdefault("institution", "Department of Physical Geography, Utrecht University")
#     ds.attrs.setdefault("title", "PCR-GLOBWB 2 output not coupled to MODFLOW")
#     ds.attrs.setdefault("description", "Post processing PCR-GLOBWB output by Edwin H. Sutanudjaja E.H.Sutanudjaja@UU.NL")
#     if "history" in ds.attrs: ds.attrs["history"] = _ascii(ds.attrs["history"])
#     if "time" in ds.coords:
#         t = ds["time"]
#         t.attrs["standard_name"] = "time"
#         t.attrs.setdefault("long_name", "time")
#     if "lat" in ds.coords:
#         ds["lat"].attrs.update({"standard_name":"latitude","units":"degrees_north","long_name":"latitude"})
#     if "lon" in ds.coords:
#         ds["lon"].attrs.update({"standard_name":"longitude","units":"degrees_east","long_name":"longitude"})
#     return ds

# def align_time_only(a: xr.DataArray, b: xr.DataArray):
#     if "time" not in a.dims or "time" not in b.dims:
#         return a, b
#     common = np.intersect1d(a["time"].values, b["time"].values)
#     if common.size == 0:
#         raise ValueError("No overlapping time steps between sat and wtd")
#     return a.sel(time=common), b.sel(time=common)

# # ---------------- IO ----------------
# def open_sat(scen: str) -> xr.DataArray:
#     path = os.path.join(SAT_DIR, f"satAreaFrac_monthly_ensemble_mean_{scen}.nc")
#     if not os.path.exists(path): raise FileNotFoundError(path)
#     ds = xr.open_dataset(path, decode_times=True, chunks=CHUNK_TARGETS)
#     da = ds[SAT_VAR] if SAT_VAR in ds else ds[list(ds.data_vars)[0]]
#     da = standardize_coord_names(da)
#     if np.any(np.diff(da["lat"].values) < 0): da = da.sortby("lat")
#     if np.any(np.diff(da["lon"].values) < 0): da = da.sortby("lon")
#     if SMALL_TEST: da = subset_region(da, REGION_NAME)
#     if int(da.sizes["lat"]) == 0 or int(da.sizes["lon"]) == 0:
#         raise ValueError("SAT opened with empty spatial dims")
#     return da.astype("float32")

# def open_wtd(scen: str) -> xr.DataArray:
#     zpath = os.path.join(MERGED_DIR, scen, "wtd.zarr")
#     if not os.path.isdir(zpath): raise FileNotFoundError(zpath)
#     try:
#         ds = xr.open_zarr(zpath, consolidated=True, chunks="auto")
#     except Exception:
#         ds = xr.open_zarr(zpath, chunks="auto")
#     if WTD_MAIN not in ds or WTD_FALLBACK not in ds:
#         raise KeyError(f"{zpath} must contain {WTD_MAIN} and {WTD_FALLBACK}, has {list(ds.data_vars)}")
#     l1 = standardize_coord_names(ds[WTD_MAIN])
#     l2 = standardize_coord_names(ds[WTD_FALLBACK])
#     for d in ("lat","lon"):
#         if d in l1.dims and np.any(np.diff(l1[d]) < 0): l1 = l1.sortby(d)
#         if d in l2.dims and np.any(np.diff(l2[d]) < 0): l2 = l2.sortby(d)
#     wtd = xr.where(l1.notnull(), l1, l2)
#     if SMALL_TEST: wtd = subset_region(wtd, REGION_NAME)
#     if int(wtd.sizes["lat"]) == 0 or int(wtd.sizes["lon"]) == 0:
#         raise ValueError("WTD opened with empty spatial dims")
#     return wtd.astype("float32")

# # ---------------- Core ----------------
# def process_scenario(scen: str):
#     print(f"[info] {scen}: open inputs")
#     sat = open_sat(scen)
#     wtd = open_wtd(scen)

#     # unify longitude conventions and align ONLY time
#     target = dataset_lon_range(wtd)
#     sat = unify_longitudes(sat, target)
#     wtd = unify_longitudes(wtd, target)
#     sat, wtd = align_time_only(sat, wtd)

#     # map WTD onto SAT grid
#     try:
#         wtd_on_sat = wtd.reindex(lat=sat["lat"], lon=sat["lon"], method="nearest")
#     except Exception:
#         wtd_on_sat = wtd.interp(lat=sat["lat"], lon=sat["lon"], method="nearest")

#     # chunking
#     cm = chunk_map(sat)
#     sat = sat.chunk(cm)
#     wtd_on_sat = wtd_on_sat.chunk({d: cm.get(d, 1) for d in wtd_on_sat.dims})

#     # rule and dataset
#     valid = sat.notnull() & wtd_on_sat.notnull()
#     wet = xr.where(valid, ((sat > SAT_THRESHOLD) & (wtd_on_sat <= WTD_THRESHOLD)).astype("i1"), np.int8(127))
#     wet.name = "wetGDE"
#     wet = wet.transpose(*[d for d in ("time","lat","lon") if d in wet.dims])

#     for d in wet.dims:
#         if wet.sizes[d] <= 0:
#             raise ValueError(f"Output dimension {d} has size {wet.sizes[d]} after processing")

#     ds_out = xr.Dataset({"wetGDE": wet}, coords={d: wet.coords[d] for d in wet.dims})

#     # metadata
#     sat_hdr_path = os.path.join(SAT_DIR, f"satAreaFrac_monthly_ensemble_mean_{scen}.nc")
#     with xr.open_dataset(sat_hdr_path, decode_times=True) as hdr:
#         ds_out.attrs.update(dict(hdr.attrs))

#     ds_out.attrs["history"] = ds_out.attrs.get("history","") + f"; wetGDE: satAreaFrac>{SAT_THRESHOLD} & WTD<={WTD_THRESHOLD}m, scenario={scen}"
#     ds_out.attrs["additional_author"] = ADDITIONAL_AUTHOR
#     ds_out.attrs["additional_email"]  = ADDITIONAL_EMAIL
#     ds_out.attrs["date_created"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
#     ds_out.attrs["command"] = " ".join([_ascii(k) for k in os.sys.argv]) or "python wetgde_mask.py"
#     ds_out.attrs["thresholds"] = f"sat>{SAT_THRESHOLD}, wtd<={WTD_THRESHOLD}"
#     ds_out["wetGDE"].attrs = {
#         "long_name": "Wetland GDE mask from satAreaFrac and WTD",
#         "units": "1",
#         "values": "0 or 1",
#         "rule": f"satAreaFrac>{SAT_THRESHOLD} & WTD<={WTD_THRESHOLD}m",
#         "scenario": scen,
#         "source_sat": os.path.basename(sat_hdr_path),
#         "source_wtd": f"{scen}/wtd.zarr::{WTD_MAIN}|{WTD_FALLBACK}",
#         "_FillValue_meaning": "127 means missing inputs",
#         "_Storage": "int8",
#         "missing_value": np.int8(127),
#     }
#     ds_out = sanitize_attrs(ds_out)

#     # output
#     out_path = os.path.join(OUT_DIR, f"wetGDE_{scen}.nc")
#     if os.path.exists(out_path):
#         try: os.remove(out_path)
#         except Exception: pass

#     enc_chunks = tuple(max(1, cm.get(d, 1)) for d in ds_out["wetGDE"].dims)
#     enc = {
#         "wetGDE": {
#             "dtype": "i1",
#             "zlib": True,
#             "complevel": 3,  # speed-friendly
#             "chunksizes": enc_chunks,
#             "_FillValue": np.int8(127),
#         },
#         "time": {"zlib": True, "complevel": 1} if "time" in ds_out.dims else {},
#         "lat":  {"zlib": True, "complevel": 1} if "lat"  in ds_out.dims else {},
#         "lon":  {"zlib": True, "complevel": 1} if "lon"  in ds_out.dims else {},
#     }
#     udims = {"time": True} if "time" in ds_out["wetGDE"].dims else None

#     print(f"[write] {scen} -> {out_path} engine={ENGINE} chunks={enc_chunks}")
#     delayed = ds_out.to_netcdf(
#         out_path,
#         engine=ENGINE,
#         format=NC_MODEL,
#         encoding=enc,
#         **({"unlimited_dims": udims} if udims else {}),
#         compute=False,
#     )

#     # progress bar (to file if PROGRESS_LOG is set)
#     if PROGRESS_LOG:
#         print(f"[progress] writing Dask bar to {PROGRESS_LOG}")
#         with open(PROGRESS_LOG, "w", buffering=1) as fh:
#             with ProgressBar(out=fh):
#                 delayed.compute()
#     else:
#         with ProgressBar():
#             delayed.compute()

#     try:
#         size = os.path.getsize(out_path)
#         print(f"[done] {scen} {out_path} size={size}")
#     except Exception:
#         print(f"[done] {scen} {out_path}")

# # ---------------- Main ----------------
# if __name__ == "__main__":
#     scens = [ONLY_SCEN] if ONLY_SCEN else SCENARIOS
#     if TEST: scens = scens[:1]

#     print(f"[setup] OUT_DIR={OUT_DIR}")
#     print(f"[setup] SMALL_TEST={SMALL_TEST} REGION={REGION_NAME if SMALL_TEST else 'full_global'}")
#     print(f"[setup] chunks={CHUNK_TARGETS} ENGINE=netcdf4 MODEL=NETCDF4_CLASSIC")

#     for scen in scens:
#         try:
#             process_scenario(scen)
#         except Exception as e:
#             ts = datetime.now().isoformat(timespec="seconds")
#             print(f"[error] {ts} | {scen} | {e}")
#     print("[complete] all scenarios processed")

# ###donefrom scratch wetGDE with satAreaFrac variable
# #!/usr/bin/env python3
# """
# wetGDE mask builder at ~1 km global for CMIP6 scenarios, plus satAreaFrac stored
# STRICTLY where wetGDE == 1. Recomputes everything from SAT and WTD.

# Rule:
#   wetGDE = 1 where (satAreaFrac > 0.5) AND (WTD <= 5 m), else 0, 127 means missing inputs
#   satAreaFrac_out = satAreaFrac where wetGDE == 1, NaN elsewhere
# """

# import os
# import numpy as np
# import xarray as xr
# import dask
# from dask.diagnostics import ProgressBar
# from datetime import datetime

# # ---------------- Paths and parameters ----------------
# SAT_DIR     = "/projects/prjs1578/sat_future"
# MERGED_DIR  = "/projects/prjs1578/future_gw"
# OUT_DIR     = "/projects/prjs1578/futurewetgde/wetGDEs_rebuilt"
# os.makedirs(OUT_DIR, exist_ok=True)

# SCENARIOS   = ["historical", "ssp126", "ssp370", "ssp585"]
# SAT_VAR     = "satAreaFrac"
# WTD_MAIN    = "l1_wtd"
# WTD_FALLBACK= "l2_wtd"

# SAT_THRESHOLD = 0.5
# WTD_THRESHOLD = 5.0

# # large spatial chunks for faster IO, overridable via env SPATIAL_CHUNK
# SZ = int(os.environ.get("SPATIAL_CHUNK", "1024"))
# CHUNK_TARGETS = {"time": 1, "lat": SZ, "lon": SZ}

# ENGINE   = "netcdf4"
# NC_MODEL = "NETCDF4_CLASSIC"

# ADDITIONAL_AUTHOR = "Nicole Gyakowah Otoo"
# ADDITIONAL_EMAIL  = "n.g.otoo@uu.nl, nicholetylor@gmail.com"

# TEST        = os.environ.get("TEST", "false").lower() in ("true","1","yes","y")
# SMALL_TEST  = os.environ.get("SMALL_TEST", "false").lower() in ("true","1","yes","y")
# REGION_NAME = os.environ.get("REGION", "great_plains").strip().lower()
# ONLY_SCEN   = os.environ.get("SCENARIO", "").strip() or None
# PROGRESS_LOG= os.environ.get("PROGRESS_LOG", "").strip() or None

# REGIONS = {
#     "great_plains": dict(lat=(25.0, 50.0),  lon=(-106.0, -94.0)),
#     "sahel":        dict(lat=(10.0, 20.0),  lon=(-20.0,   30.0)),
#     "amazon":       dict(lat=(-15.0, 5.0),  lon=(-75.0,  -50.0)),
#     "europe":       dict(lat=(36.0, 60.0),  lon=(-10.0,   30.0)),
#     "australia_se": dict(lat=(-40.0, -25.0),lon=(140.0,  155.0)),
# }

# dask.config.set({
#     "scheduler": "threads",
#     "array.slicing.split_large_chunks": True,
# })

# # ---------------- Helpers ----------------
# def _ascii(s): return str(s).encode("ascii", "ignore").decode("ascii")

# def standardize_coord_names(da: xr.DataArray) -> xr.DataArray:
#     rename = {}
#     if "latitude" in da.dims:  rename["latitude"]  = "lat"
#     if "longitude" in da.dims: rename["longitude"] = "lon"
#     if "y" in da.dims and "lat" not in da.dims: rename["y"] = "lat"
#     if "x" in da.dims and "lon" not in da.dims: rename["x"] = "lon"
#     return da.rename(rename) if rename else da

# def dataset_lon_range(da: xr.DataArray) -> str:
#     if "lon" not in da.dims: return "m180_180"
#     vmin = float(da["lon"].min()); vmax = float(da["lon"].max())
#     eps = 1e-6
#     return "0_360" if vmin >= -eps and vmax <= 360.0 + eps else "m180_180"

# def to_0_360(lon):    return xr.where(lon < 0, lon + 360.0, lon)
# def to_m180_180(lon): return ((lon + 180.0) % 360.0) - 180.0

# def unify_longitudes(da: xr.DataArray, target_range: str) -> xr.DataArray:
#     if "lon" not in da.dims: return da
#     lon = da["lon"]
#     vmin = float(lon.min()); vmax = float(lon.max())
#     if target_range == "0_360":
#         if vmin < 0 or vmax > 360:
#             da = da.assign_coords(lon=to_0_360(lon)).sortby("lon")
#     else:
#         if vmin < -180 or vmax > 180:
#             da = da.assign_coords(lon=to_m180_180(lon)).sortby("lon")
#     return da

# def subset_region(da: xr.DataArray, region_name: str) -> xr.DataArray:
#     if region_name not in REGIONS: region_name = "great_plains"
#     lat_lo, lat_hi = REGIONS[region_name]["lat"]
#     lon_lo, lon_hi = REGIONS[region_name]["lon"]
#     if "lat" in da.dims and np.any(np.diff(da["lat"].values) < 0): da = da.sortby("lat")
#     if "lon" in da.dims and np.any(np.diff(da["lon"].values) < 0): da = da.sortby("lon")
#     if "lon" in da.dims and dataset_lon_range(da) == "0_360":
#         lon_lo = lon_lo if lon_lo >= 0 else lon_lo + 360.0
#         lon_hi = lon_hi if lon_hi >= 0 else lon_hi + 360.0
#         if lon_lo <= lon_hi:
#             return da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
#         left  = da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, 360.0))
#         right = da.sel(lat=slice(lat_lo, lat_hi), lon=slice(0.0, lon_hi))
#         return xr.concat([left, right], dim="lon").sortby("lon")
#     return da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))

# def chunk_map(da: xr.DataArray):
#     out = {}
#     for d in da.dims:
#         size = int(da.sizes[d])
#         if size <= 0:
#             raise ValueError(f"Dimension {d} has size {size}")
#         target = int(CHUNK_TARGETS.get(d, size))
#         out[d] = max(1, min(target, size))
#     return out

# def sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
#     ds.attrs = {k: _ascii(v) for k, v in ds.attrs.items()}
#     ds.attrs.setdefault("institution", "Department of Physical Geography, Utrecht University")
#     ds.attrs.setdefault("title", "PCR-GLOBWB 2 output not coupled to MODFLOW")
#     ds.attrs.setdefault("description", "Post processing PCR-GLOBWB output by Edwin H. Sutanudjaja E.H.Sutanudjaja@UU.NL")
#     if "history" in ds.attrs: ds.attrs["history"] = _ascii(ds.attrs["history"])
#     if "time" in ds.coords:
#         t = ds["time"]
#         t.attrs["standard_name"] = "time"
#         t.attrs.setdefault("long_name", "time")
#     if "lat" in ds.coords:
#         ds["lat"].attrs.update({"standard_name":"latitude","units":"degrees_north","long_name":"latitude"})
#     if "lon" in ds.coords:
#         ds["lon"].attrs.update({"standard_name":"longitude","units":"degrees_east","long_name":"longitude"})
#     return ds

# def align_time_only(a: xr.DataArray, b: xr.DataArray):
#     if "time" not in a.dims or "time" not in b.dims:
#         return a, b
#     common = np.intersect1d(a["time"].values, b["time"].values)
#     if common.size == 0:
#         raise ValueError("No overlapping time steps between sat and wtd")
#     return a.sel(time=common), b.sel(time=common)

# # ---------------- IO ----------------
# def open_sat(scen: str) -> xr.DataArray:
#     path = os.path.join(SAT_DIR, f"satAreaFrac_monthly_ensemble_mean_{scen}.nc")
#     if not os.path.exists(path): raise FileNotFoundError(path)
#     ds = xr.open_dataset(path, decode_times=True, chunks=CHUNK_TARGETS)
#     da = ds[SAT_VAR] if SAT_VAR in ds else ds[list(ds.data_vars)[0]]
#     da = standardize_coord_names(da)
#     if np.any(np.diff(da["lat"].values) < 0): da = da.sortby("lat")
#     if np.any(np.diff(da["lon"].values) < 0): da = da.sortby("lon")
#     if SMALL_TEST: da = subset_region(da, REGION_NAME)
#     if int(da.sizes["lat"]) == 0 or int(da.sizes["lon"]) == 0:
#         raise ValueError("SAT opened with empty spatial dims")
#     return da.astype("float32")

# def open_wtd(scen: str) -> xr.DataArray:
#     zpath = os.path.join(MERGED_DIR, scen, "wtd.zarr")
#     if not os.path.isdir(zpath): raise FileNotFoundError(zpath)
#     try:
#         ds = xr.open_zarr(zpath, consolidated=True, chunks="auto")
#     except Exception:
#         ds = xr.open_zarr(zpath, chunks="auto")
#     if WTD_MAIN not in ds or WTD_FALLBACK not in ds:
#         raise KeyError(f"{zpath} must contain {WTD_MAIN} and {WTD_FALLBACK}, has {list(ds.data_vars)}")
#     l1 = standardize_coord_names(ds[WTD_MAIN])
#     l2 = standardize_coord_names(ds[WTD_FALLBACK])
#     for d in ("lat","lon"):
#         if d in l1.dims and np.any(np.diff(l1[d]) < 0): l1 = l1.sortby(d)
#         if d in l2.dims and np.any(np.diff(l2[d]) < 0): l2 = l2.sortby(d)
#     wtd = xr.where(l1.notnull(), l1, l2)
#     if SMALL_TEST: wtd = subset_region(wtd, REGION_NAME)
#     if int(wtd.sizes["lat"]) == 0 or int(wtd.sizes["lon"]) == 0:
#         raise ValueError("WTD opened with empty spatial dims")
#     return wtd.astype("float32")

# # ---------------- Core ----------------
# def process_scenario(scen: str):
#     print(f"[info] {scen}: open inputs")
#     sat = open_sat(scen)
#     wtd = open_wtd(scen)

#     # unify longitude conventions and align ONLY time
#     target = dataset_lon_range(wtd)
#     sat = unify_longitudes(sat, target)
#     wtd = unify_longitudes(wtd, target)
#     sat, wtd = align_time_only(sat, wtd)

#     # map WTD onto SAT grid
#     try:
#         wtd_on_sat = wtd.reindex(lat=sat["lat"], lon=sat["lon"], method="nearest")
#     except Exception:
#         wtd_on_sat = wtd.interp(lat=sat["lat"], lon=sat["lon"], method="nearest")

#     # chunking
#     cm = chunk_map(sat)
#     sat = sat.chunk(cm)
#     wtd_on_sat = wtd_on_sat.chunk({d: cm.get(d, 1) for d in wtd_on_sat.dims})

#     # rule and dataset
#     valid_inputs = sat.notnull() & wtd_on_sat.notnull()
#     wet = xr.where(valid_inputs, ((sat > SAT_THRESHOLD) & (wtd_on_sat <= WTD_THRESHOLD)).astype("i1"), np.int8(127))
#     wet.name = "wetGDE"
#     wet = wet.transpose(*[d for d in ("time","lat","lon") if d in wet.dims])

#     # satAreaFrac strictly where wetGDE == 1, NaN elsewhere
#     sat_keep = xr.where(wet == 1, sat, np.nan).astype("float32")
#     sat_keep.name = "satAreaFrac"
#     sat_keep = sat_keep.transpose(*[d for d in ("time","lat","lon") if d in sat_keep.dims])

#     for d in wet.dims:
#         if wet.sizes[d] <= 0:
#             raise ValueError(f"Output dimension {d} has size {wet.sizes[d]} after processing")

#     ds_out = xr.Dataset(
#         {"wetGDE": wet, "satAreaFrac": sat_keep},
#         coords={d: wet.coords[d] for d in wet.dims},
#     )

#     # metadata
#     sat_hdr_path = os.path.join(SAT_DIR, f"satAreaFrac_monthly_ensemble_mean_{scen}.nc")
#     with xr.open_dataset(sat_hdr_path, decode_times=True) as hdr:
#         ds_out.attrs.update(dict(hdr.attrs))

#     ds_out.attrs["history"] = ds_out.attrs.get("history","") + f"; wetGDE: satAreaFrac>{SAT_THRESHOLD} & WTD<={WTD_THRESHOLD}m, scenario={scen}; satAreaFrac_out kept only where wetGDE==1"
#     ds_out.attrs["additional_author"] = ADDITIONAL_AUTHOR
#     ds_out.attrs["additional_email"]  = ADDITIONAL_EMAIL
#     ds_out.attrs["date_created"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
#     ds_out.attrs["command"] = " ".join([_ascii(k) for k in os.sys.argv]) or "python rebuild_wetgde_strict.py"
#     ds_out.attrs["thresholds"] = f"sat>{SAT_THRESHOLD}, wtd<={WTD_THRESHOLD}"

#     ds_out["wetGDE"].attrs = {
#         "long_name": "Wetland GDE mask from satAreaFrac and WTD",
#         "units": "1",
#         "values": "0 or 1",
#         "rule": f"satAreaFrac>{SAT_THRESHOLD} & WTD<={WTD_THRESHOLD}m",
#         "scenario": scen,
#         "source_sat": os.path.basename(sat_hdr_path),
#         "source_wtd": f"{scen}/wtd.zarr::{WTD_MAIN}|{WTD_FALLBACK}",
#         "_FillValue_meaning": "127 means missing inputs",
#         "_Storage": "int8",
#         "missing_value": np.int8(127),
#     }
#     ds_out["satAreaFrac"].attrs = {
#         "long_name": "Saturated area fraction",
#         "units": "1",
#         "source_sat": os.path.basename(sat_hdr_path),
#         "note": "Values kept only where wetGDE==1, NaN elsewhere",
#     }

#     ds_out = sanitize_attrs(ds_out)

#     # output
#     out_path = os.path.join(OUT_DIR, f"wetGDE_{scen}.nc")
#     if os.path.exists(out_path):
#         try: os.remove(out_path)
#         except Exception: pass

#     enc_chunks = tuple(max(1, cm.get(d, 1)) for d in ds_out["wetGDE"].dims)
#     enc = {
#         "wetGDE": {
#             "dtype": "i1",
#             "zlib": True,
#             "complevel": 3,
#             "chunksizes": enc_chunks,
#             "_FillValue": np.int8(127),
#         },
#         # pack satAreaFrac as uint8 with scale factor to save space
#         "satAreaFrac": {
#             "dtype": "u1",
#             "zlib": True,
#             "complevel": 3,
#             "chunksizes": enc_chunks,
#             "_FillValue": 255,
#             "scale_factor": 1.0 / 255.0,
#             "add_offset": 0.0,
#         },
#         "time": {"zlib": True, "complevel": 1} if "time" in ds_out.dims else {},
#         "lat":  {"zlib": True, "complevel": 1} if "lat"  in ds_out.dims else {},
#         "lon":  {"zlib": True, "complevel": 1} if "lon"  in ds_out.dims else {},
#     }
#     udims = {"time": True} if "time" in ds_out["wetGDE"].dims else None

#     print(f"[write] {scen} -> {out_path} engine={ENGINE} chunks={enc_chunks}")
#     delayed = ds_out.to_netcdf(
#         out_path,
#         engine=ENGINE,
#         format=NC_MODEL,
#         encoding=enc,
#         **({"unlimited_dims": udims} if udims else {}),
#         compute=False,
#     )

#     # progress bar
#     if PROGRESS_LOG:
#         print(f"[progress] writing Dask bar to {PROGRESS_LOG}")
#         with open(PROGRESS_LOG, "w", buffering=1) as fh:
#             with ProgressBar(out=fh):
#                 delayed.compute()
#     else:
#         with ProgressBar():
#             delayed.compute()

#     try:
#         size = os.path.getsize(out_path)
#         print(f"[done] {scen} {out_path} size={size}")
#     except Exception:
#         print(f"[done] {scen} {out_path}")

# # ---------------- Main ----------------
# if __name__ == "__main__":
#     scens = [ONLY_SCEN] if ONLY_SCEN else SCENARIOS
#     if TEST: scens = scens[:1]

#     print(f"[setup] OUT_DIR={OUT_DIR}")
#     print(f"[setup] SMALL_TEST={SMALL_TEST} REGION={REGION_NAME if SMALL_TEST else 'full_global'}")
#     print(f"[setup] chunks={CHUNK_TARGETS} ENGINE=netcdf4 MODEL=NETCDF4_CLASSIC")

#     for scen in scens:
#         try:
#             process_scenario(scen)
#         except Exception as e:
#             ts = datetime.now().isoformat(timespec="seconds")
#             print(f"[error] {ts} | {scen} | {e}")
#     print("[complete] all scenarios processed")


#####WETGDE AUGMENTATION WITH SATAREAFRAC VARIABLE
#!/usr/bin/env python3
"""
Append satAreaFrac to existing wetGDE NetCDFs, but only where wetGDE == 1.
Keeps original wetGDE unchanged. Packs satAreaFrac as uint8 with scale_factor 1/255.
"""

import os
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# -------- Paths --------
SAT_DIR = os.environ.get("SAT_DIR", "/projects/prjs1578/sat_future")
WET_DIR = os.environ.get("WET_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
OUT_DIR = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_augmented")
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = (
    os.environ.get("ONLY_SCEN").split(",")
    if os.environ.get("ONLY_SCEN")
    else ["historical", "ssp126", "ssp370", "ssp585"]
)

ENGINE = "netcdf4"
NC_MODEL = "NETCDF4_CLASSIC"

def _std_names(da):
    r = {}
    if "latitude" in da.dims:  r["latitude"] = "lat"
    if "longitude" in da.dims: r["longitude"] = "lon"
    if "y" in da.dims and "lat" not in da.dims: r["y"] = "lat"
    if "x" in da.dims and "lon" not in da.dims: r["x"] = "lon"
    return da.rename(r) if r else da

def _to_m180(lon): return ((lon + 180.0) % 360.0) - 180.0

def _unify_to_target(da, target_min=-180.0, target_max=180.0):
    if "lon" not in da.dims: return da
    lon = da["lon"]
    if float(lon.min()) < target_min or float(lon.max()) > target_max:
        if target_min == -180.0:
            da = da.assign_coords(lon=_to_m180(lon))
        else:
            da = da.assign_coords(lon=xr.where(lon < 0, lon + 360.0, lon))
        da = da.sortby("lon")
    return da

def augment_one(scen):
    wet_path = os.path.join(WET_DIR, f"wetGDE_{scen}.nc")
    sat_path = os.path.join(SAT_DIR, f"satAreaFrac_monthly_ensemble_mean_{scen}.nc")
    if not os.path.exists(wet_path):
        print(f"[skip] {scen} missing {wet_path}")
        return
    if not os.path.exists(sat_path):
        print(f"[skip] {scen} missing {sat_path}")
        return

    print(f"[open] {scen}")
    wet = xr.open_dataset(wet_path, decode_times=True, chunks={"time": 1, "lat": 1024, "lon": 1024})
    sat_ds = xr.open_dataset(sat_path, decode_times=True, chunks={"time": 1, "lat": 1024, "lon": 1024})

    wet_da = wet["wetGDE"]
    sat_da = _std_names(sat_ds.get("satAreaFrac", list(sat_ds.data_vars.values())[0])).astype("float32")

    # sort
    if "lat" in wet_da.dims and np.any(np.diff(wet_da["lat"]) < 0): wet_da = wet_da.sortby("lat")
    if "lon" in wet_da.dims and np.any(np.diff(wet_da["lon"]) < 0): wet_da = wet_da.sortby("lon")
    if "lat" in sat_da.dims and np.any(np.diff(sat_da["lat"]) < 0): sat_da = sat_da.sortby("lat")
    if "lon" in sat_da.dims and np.any(np.diff(sat_da["lon"]) < 0): sat_da = sat_da.sortby("lon")

    # unify SAT longitudes to match wetGDE convention
    sat_da = (
        _unify_to_target(sat_da, -180.0, 180.0)
        if float(wet_da["lon"].min()) >= -180.0 and float(wet_da["lon"].max()) <= 180.0
        else _unify_to_target(sat_da, 0.0, 360.0)
    )

    # time alignment to wetGDE
    if "time" in sat_da.dims and "time" in wet_da.dims:
        common = np.intersect1d(sat_da["time"].values, wet_da["time"].values)
        if common.size == 0:
            raise ValueError(f"No overlapping time between SAT and wetGDE for {scen}")
        sat_da = sat_da.sel(time=common)
        wet = wet.sel(time=common)
        wet_da = wet_da.sel(time=common)

    # map SAT onto wet grid
    sat_on_wet = sat_da.reindex(lat=wet_da["lat"], lon=wet_da["lon"], method="nearest")

    # keep only where wetGDE == 1
    valid_inputs = (wet_da == 1)
    sat_keep = xr.where(valid_inputs, sat_on_wet, np.nan).astype("float32")
    sat_keep = sat_keep.transpose(*wet_da.dims)
    sat_keep.name = "satAreaFrac"

    # attach to dataset
    out = wet.copy()
    out["satAreaFrac"] = sat_keep
    out["satAreaFrac"].attrs.update({
        "long_name": "Saturated area fraction",
        "units": "1",
        "source_sat": os.path.basename(sat_path),
        "note": "Values kept only where wetGDE==1, NaN elsewhere",
    })

    # pack satAreaFrac as uint8 with scale factor
    chunksizes = out["wetGDE"].encoding.get("chunksizes", None)
    if not chunksizes:
        chunksizes = tuple(max(1, s) for s in out["wetGDE"].shape)
    enc = {
        "wetGDE": {
            "dtype": "i1",
            "zlib": True,
            "complevel": 3,
            "_FillValue": np.int8(127),
            "chunksizes": chunksizes,
        },
        "satAreaFrac": {
            "dtype": "u1",
            "zlib": True,
            "complevel": 3,
            "_FillValue": 255,
            "scale_factor": 1.0 / 255.0,
            "add_offset": 0.0,
            "chunksizes": chunksizes,
        },
        "time": {"zlib": True, "complevel": 1} if "time" in out.dims else {},
        "lat":  {"zlib": True, "complevel": 1} if "lat" in out.dims else {},
        "lon":  {"zlib": True, "complevel": 1} if "lon" in out.dims else {},
    }

    out_path = os.path.join(OUT_DIR, f"wetGDE_{scen}.nc")
    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass

    print(f"[write] {scen} -> {out_path}")
    delayed = out.to_netcdf(out_path, engine=ENGINE, format=NC_MODEL, encoding=enc, compute=False)
    with ProgressBar():
        delayed.compute()
    print(f"[done] {scen}")

if __name__ == "__main__":
    for s in SCENARIOS:
        scen = s.strip()
        try:
            augment_one(scen)
        except Exception as e:
            print(f"[error] {scen} | {e}")
    print("[complete]")

