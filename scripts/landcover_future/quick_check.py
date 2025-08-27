#!/usr/bin/env python3
"""
Time series of global land-cover fractions from LUH2 states

Plots
- Cropland fraction of land (historical 1960–2014 dotted black, futures 2015–2100 colored)
- Cropland + managed pasture fraction of land (same styling)

Outputs
- lc_fraction_timeseries_with_historical.csv
- ts_cropland_with_historical.png
- ts_cropland_pasture_with_historical.png
"""

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# robust CF-time decoding without deprecated kwargs
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# ── Config ───────────────────────────────────────────────────────────────────
HIST_STATES = os.environ.get(
    "HIST_STATES",
    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp/Historic_Data/states.nc",
)
FUTURE_ROOT = os.environ.get(
    "LUH2_SSP_ROOT",
    "/projects/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp",
)
ENGINE = os.environ.get("XR_ENGINE", "netcdf4")      # per request, default netcdf4
STATIC_PATH = os.environ.get("LUH2_STATIC", "")      # optional staticData_quarterdeg.nc
USE_LAND_MASK = os.environ.get("USE_LAND_MASK", "1").lower() in ("1","true","y")
INCLUDE_RANGELAND = os.environ.get("INCLUDE_RANGELAND", "0").lower() in ("1","true","y")

HIST_START, HIST_END = 1960, 2014
FUT_START,  FUT_END  = 2015, 2100

SCENARIOS = ["historical", "ssp126", "ssp370", "ssp585"]
SCEN_COLOR = {
    "historical": "#222222",
    "ssp126": "#1a9850",
    "ssp370": "#fdae61",
    "ssp585": "#d73027",
}
LINESTYLE = {"historical": ":", "ssp126": "-", "ssp370": "-", "ssp585": "-"}
SCEN_LABEL = {
    "historical": "Historical (LUH2 v2h)",
    "ssp126": "SSP1-2.6 (IMAGE)",
    "ssp370": "SSP3-7.0 (AIM)",
    "ssp585": "SSP5-8.5 (REMIND-MAGPIE)",
}
FOLDERS = {
    "ssp126": "RCP2_6_SSP1_from_IMAGE",
    "ssp370": "RCP7_0_SSP3_from_AIM",
    "ssp585": "RCP8_5_SSP5_from_REMIND_MAGPIE",
}
PLOT_DPI = int(os.environ.get("PLOT_DPI", "200"))

def log(s): print(s, flush=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def open_states(path_or_dir: str, year_min: int, year_max: int) -> xr.Dataset:
    """Open LUH2 states from a file or directory, slice to [year_min,year_max] before converting times."""
    if os.path.isdir(path_or_dir):
        cands = [f for f in os.listdir(path_or_dir) if re.search(r"states.*\.nc$", f)]
        if not cands:
            raise FileNotFoundError(f"No states*.nc in {path_or_dir}")
        fn = os.path.join(path_or_dir, sorted(cands)[0])
    else:
        fn = path_or_dir
        if not os.path.isfile(fn):
            raise FileNotFoundError(fn)
    log(f"  Opening {fn}")

    # decode to cftime, slice, then convert to numpy-safe calendar
    try:
        ds = xr.open_dataset(fn, engine=ENGINE, decode_times=TIME_CODER)
        ds = ds.sel(time=slice(f"{year_min}", f"{year_max}"))
        ds = ds.convert_calendar("standard", use_cftime=False)
    except Exception as e:
        log(f"  CF decode failed, falling back, reason: {e}")
        ds = xr.open_dataset(fn, engine=ENGINE, decode_times=False)
        units = str(ds["time"].attrs.get("units", "years since 2015-01-01"))
        m = re.search(r"years since\s*(\d{4})", units)
        base_year = int(m.group(1)) if m else year_min
        offs = np.rint(np.asarray(ds["time"].values)).astype(int)
        yrs = base_year + offs
        keep = (yrs >= year_min) & (yrs <= year_max)
        if not keep.any():
            raise ValueError(f"No years in [{year_min},{year_max}]")
        ds = ds.isel(time=np.where(keep)[0])
        yrs_keep = yrs[keep].astype(int)
        ds = ds.assign_coords(time=("time", pd.to_datetime([f"{y}-01-01" for y in yrs_keep])))

    if "latitude" in ds.dims:  ds = ds.rename({"latitude":"lat"})
    if "longitude" in ds.dims: ds = ds.rename({"longitude":"lon"})
    need = ["c3ann","c4ann","c3per","c4per","c3nfx","pastr"]
    miss = [v for v in need if v not in ds]
    if miss:
        raise KeyError(f"Missing variables: {miss}")

    crops = (ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]).fillna(0).clip(0,1)
    pastr = ds["pastr"].fillna(0).clip(0,1)
    rng   = ds["range"].fillna(0).clip(0,1) if "range" in ds.data_vars else xr.zeros_like(crops)

    out = xr.Dataset({"crops":crops, "pastr":pastr, "range":rng})
    log(f"  Grid lat={out.dims['lat']} lon={out.dims['lon']}, years {pd.to_datetime(out.time.values[0]).year}-{pd.to_datetime(out.time.values[-1]).year}")
    return out

def cell_area_km2(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    R = 6_371_000.0
    dlat = float(abs(lat[1]-lat[0])); dlon = float(abs(lon[1]-lon[0]))
    dlam = dlon/360.0
    band = np.sin(np.deg2rad(lat + dlat/2.0)) - np.sin(np.deg2rad(lat - dlat/2.0))
    a_lat = (2*np.pi*R**2)*band*dlam/1e6
    return xr.DataArray(np.repeat(a_lat.values[:,None], lon.size, axis=1),
                        dims=("lat","lon"), coords={"lat":lat.values,"lon":lon.values}, name="cell_km2")

def load_land_fraction(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    if not (USE_LAND_MASK and STATIC_PATH and os.path.exists(STATIC_PATH)):
        log("No static land mask, using land fraction = 1")
        return xr.DataArray(np.ones((lat.size, lon.size), dtype="float32"),
                            dims=("lat","lon"), coords={"lat":lat.values,"lon":lon.values}, name="land_frac")
    log(f"Opening static mask: {STATIC_PATH}")
    ds = xr.open_dataset(STATIC_PATH, engine=ENGINE, decode_times=False)
    if "latitude" in ds.dims:  ds = ds.rename({"latitude":"lat"})
    if "longitude" in ds.dims: ds = ds.rename({"longitude":"lon"})
    cand = [v for v in ds.data_vars if v.lower() in ("gicew","icew","ice_water_frac","waterfrac","frac_water","ocean")]
    if cand:
        iw = ds[cand[0]].astype("float32")
        if not (np.array_equal(iw["lat"].values, lat.values) and np.array_equal(iw["lon"].values, lon.values)):
            iw = iw.interp(lat=lat, lon=lon, method="nearest")
        return (1.0 - iw).clip(0,1).rename("land_frac")
    log("Static file missing water fraction, using land fraction = 1")
    return xr.DataArray(np.ones((lat.size, lon.size), dtype="float32"),
                        dims=("lat","lon"), coords={"lat":lat.values,"lon":lon.values}, name="land_frac")

# ── Run ───────────────────────────────────────────────────────────────────────
log("Loading historical 1960–2014")
hist = open_states(HIST_STATES, HIST_START, HIST_END)
lat, lon = hist.lat, hist.lon
cell_km2 = cell_area_km2(lat, lon)
land_km2 = cell_km2 * load_land_fraction(lat, lon)
den_land = land_km2.sum(("lat","lon"))

hist_years = pd.date_range(f"{HIST_START}-01-01", f"{HIST_END}-01-01", freq="YS")
hist = hist.reindex(time=hist_years, method="nearest", tolerance=np.timedelta64(366,"D"))

rows = []
frac_crops_hist = ((hist["crops"] * land_km2).sum(("lat","lon")) / den_land).astype("float64")
cp_hist = (hist["crops"] + hist["pastr"]).clip(0,1)
if INCLUDE_RANGELAND:
    cp_hist = (cp_hist + hist["range"]).clip(0,1)
frac_cp_hist = ((cp_hist * land_km2).sum(("lat","lon")) / den_land).astype("float64")
rows.append(pd.DataFrame({
    "time": pd.to_datetime(hist.time.values),
    "scenario": "historical",
    "scenario_label": SCEN_LABEL["historical"],
    "frac_cropland": frac_crops_hist.values,
    "frac_cropland_pasture": frac_cp_hist.values,
}))

log("Loading futures 2015–2100")
future_years = pd.date_range(f"{FUT_START}-01-01", f"{FUT_END}-01-01", freq="YS")
for scen in ["ssp126","ssp370","ssp585"]:
    dpath = os.path.join(FUTURE_ROOT, FOLDERS[scen])
    log(f"  {scen}: {dpath}")
    ds = open_states(dpath, FUT_START, FUT_END)
    if not (np.array_equal(ds.lat.values, lat.values) and np.array_equal(ds.lon.values, lon.values)):
        ds = ds.interp(lat=lat, lon=lon, method="nearest")
    ds = ds.reindex(time=future_years, method="nearest", tolerance=np.timedelta64(366,"D"))

    frac_crops = ((ds["crops"] * land_km2).sum(("lat","lon")) / den_land).astype("float64")
    cp = (ds["crops"] + ds["pastr"]).clip(0,1)
    if INCLUDE_RANGELAND:
        cp = (cp + ds["range"]).clip(0,1)
    frac_cp = ((cp * land_km2).sum(("lat","lon")) / den_land).astype("float64")

    rows.append(pd.DataFrame({
        "time": pd.to_datetime(ds.time.values),
        "scenario": scen,
        "scenario_label": SCEN_LABEL[scen],
        "frac_cropland": frac_crops.values,
        "frac_cropland_pasture": frac_cp.values,
    }))

ts = pd.concat(rows, ignore_index=True).sort_values(["scenario","time"])
ts.to_csv("lc_fraction_timeseries_with_historical.csv", index=False)
log("Wrote lc_fraction_timeseries_with_historical.csv")

# ── Plots: time series only ───────────────────────────────────────────────────
def plot_ts(ycol: str, outfile: str, title: str, ylabel: str):
    plt.figure(figsize=(10,5))

    # historical
    dh = ts[ts["scenario"] == "historical"]
    if not dh.empty:
        plt.plot(dh["time"], dh[ycol]*100.0,
                 label=SCEN_LABEL["historical"],
                 color=SCEN_COLOR["historical"], linestyle=LINESTYLE["historical"], linewidth=2)

    # futures
    for scen in ["ssp126","ssp370","ssp585"]:
        df = ts[ts["scenario"] == scen]
        if df.empty: continue
        plt.plot(df["time"], df[ycol]*100.0,
                 label=SCEN_LABEL[scen],
                 color=SCEN_COLOR[scen], linestyle=LINESTYLE[scen], linewidth=2)

    plt.ylabel(ylabel)
    plt.xlabel("Time")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI)
    plt.close()
    log(f"Wrote {outfile}")

plot_ts("frac_cropland",
        "ts_cropland_with_historical.png",
        "Global fraction of land, Cropland",
        "Percent of land")

plot_ts("frac_cropland_pasture",
        "ts_cropland_pasture_with_historical.png",
        "Global fraction of land, Cropland + managed pasture",
        "Percent of land")

log("Done")
