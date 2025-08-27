#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime

# ---------------- Config (override via env) ----------------
IN_DIR       = os.environ.get("IN_DIR",  "/projects/prjs1578/futurewetgde/wetGDEs")
OUT_DIR      = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
IN_TEMPLATE  = os.environ.get("IN_TEMPLATE",  "wetGDE_{scenario}.nc")
OUT_TEMPLATE = os.environ.get("OUT_TEMPLATE", "wetGDE_months_yearly_{scenario}.nc")
PROGRESS_LOG = os.environ.get("PROGRESS_LOG", "").strip() or None
SZ           = int(os.environ.get("SPATIAL_CHUNK", "1024"))
ENGINE       = "netcdf4"
NC_MODEL     = "NETCDF4_CLASSIC"

ENV_SCENS = os.environ.get("SCENARIOS", "").strip()
SCENARIOS = ENV_SCENS.split() if ENV_SCENS else ["historical", "ssp126", "ssp370", "ssp585"]

os.makedirs(OUT_DIR, exist_ok=True)
READ_CHUNKS = {"time": 1, "lat": SZ, "lon": SZ}

# ---------------- Helpers ----------------
def in_path_for(scen):  return os.path.join(IN_DIR,  IN_TEMPLATE.format(scenario=scen))
def out_path_for(scen): return os.path.join(OUT_DIR, OUT_TEMPLATE.format(scenario=scen))

def progress_open_for(scen):
    if not PROGRESS_LOG: return None, None
    path = PROGRESS_LOG
    if path.endswith(os.sep) or os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"progress_{scen}.log")
    def _open(): return open(path, "w", buffering=1)
    return _open, path

# ---------------- Core ----------------
def build_annual_counts(scen: str):
    in_path  = in_path_for(scen)
    out_path = out_path_for(scen)

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass

    ds  = xr.open_dataset(in_path, chunks=READ_CHUNKS)
    wet = ds["wetGDE"]  # 0, 1, 127(fill)

    # Indicator: 1 for wet, 0 for dry, NaN for missing
    wet01 = xr.where(wet == 1, 1.0, xr.where(wet == 0, 0.0, np.nan))

    # Sum per calendar year, ignoring NaNs (missing months)
    counts_year = wet01.groupby("time.year").sum(dim="time", skipna=True)
    counts_year = counts_year.clip(min=0, max=12).astype("f4")

    # Make CF time at mid year so viewers behave, write as fixed-size time
    years = counts_year["year"].values.astype(int)
    mid_dates = np.array([np.datetime64(f"{y}-07-01") for y in years], dtype="datetime64[ns]")
    counts = counts_year.rename({"year": "time"}).assign_coords(time=mid_dates)
    counts = counts.transpose("time", "lat", "lon")
    counts.name = "wetGDE_months_yearly"
    counts.attrs.update({
        "long_name": "Number of wet months per year",
        "units": "months",
        "valid_min": 0.0,
        "valid_max": 12.0,
        "count_rule": "sum over months of 1{wetGDE==1}, missing months ignored",
        "source_mask": os.path.basename(in_path),
        "date_created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })
    counts["time"].attrs.update({"long_name": "year midpoint", "standard_name": "time"})

    # Uncompressed contiguous write, no chunks, no zlib
    enc = {
        "wetGDE_months_yearly": {
            "dtype": "f4",
            "_FillValue": np.float32(-9999.0),
            # do not set zlib, complevel, or chunksizes
        }
    }

    delayed = counts.to_netcdf(
        out_path,
        engine=ENGINE,
        format=NC_MODEL,
        encoding=enc,
        compute=False,          # fixed-size dims, contiguous layout
    )

    opener, logpath = progress_open_for(scen)
    if opener:
        print(f"[progress] {scen} -> {logpath}")
        with opener() as fh:
            with ProgressBar(out=fh):
                delayed.compute()
    else:
        with ProgressBar(out=sys.stdout):
            delayed.compute()

    try:
        print(f"[done] {scen} -> {out_path} size={os.path.getsize(out_path)}")
    except Exception:
        print(f"[done] {scen} -> {out_path}")

    try: ds.close()
    except Exception: pass

# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"[setup] IN_DIR={IN_DIR}")
    print(f"[setup] OUT_DIR={OUT_DIR}")
    print(f"[setup] IN_TEMPLATE={IN_TEMPLATE} OUT_TEMPLATE={OUT_TEMPLATE}")
    print(f"[setup] SCENARIOS={SCENARIOS}")
    print(f"[setup] read_chunks={READ_CHUNKS} ENGINE={ENGINE} MODEL={NC_MODEL}")

    for s in SCENARIOS:
        try:
            build_annual_counts(s)
        except Exception as e:
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"[error] {ts} | {s} | {e}")
