#!/usr/bin/env python3
"""
LUH2 futures: per–cell change diagnostics + global time series

Outputs
- lc_fraction_timeseries_futures.csv
- panel_timeseries_crops_and_croppastr.png         # two multilines on one page
- panel_trend_cropland_and_cp_robinson.png         # 2x3, trends, pp/decade
- panel_delta_cropland_and_cp_robinson.png         # 2x3, endpoint delta 2100-2015, pp
- hist_delta_cropland.png                          # area-weighted hist of per-cell changes
- hist_delta_croppastr.png
- map_where_crops_vs_pastures_<REF_YEAR>.png       # cropland vs pasture fractions
"""

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# ── Config ───────────────────────────────────────────────────────────────────
FUTURE_ROOT = os.environ.get(
    "LUH2_SSP_ROOT",
    "/projects/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp",
)
ENGINE = os.environ.get("XR_ENGINE", "netcdf4")
STATIC_PATH = os.environ.get(
    "LUH2_STATIC",
    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp/Supportingfiles_and_documentation/staticData_quarterdeg.nc",
)
USE_LAND_MASK = os.environ.get("USE_LAND_MASK", "1").lower() in ("1","true","y")
INCLUDE_RANGELAND = os.environ.get("INCLUDE_RANGELAND", "0").lower() in ("1","true","y")

FUT_START = int(os.environ.get("FUT_START", "2015"))
FUT_END   = int(os.environ.get("FUT_END",   "2100"))
REF_YEAR  = int(os.environ.get("REF_YEAR",  str(FUT_START)))  # for crops vs pastures map

SCENARIOS = ["ssp126", "ssp370", "ssp585"]
SCEN_LABEL = {
    "ssp126": "SSP1-2.6 (IMAGE)",
    "ssp370": "SSP3-7.0 (AIM)",
    "ssp585": "SSP5-8.5 (REMIND-MAGPIE)",
}
SCEN_SHORT = {"ssp126":"SSP1-2.6", "ssp370":"SSP3-7.0", "ssp585":"SSP5-8.5"}
SCEN_COLOR = {"ssp126":"#1a9850","ssp370":"#fdae61","ssp585":"#d73027"}
LINESTYLE  = {"ssp126":"-","ssp370":"-","ssp585":"-"}
FOLDERS = {
    "ssp126": "RCP2_6_SSP1_from_IMAGE",
    "ssp370": "RCP7_0_SSP3_from_AIM",
    "ssp585": "RCP8_5_SSP5_from_REMIND_MAGPIE",
}

PLOT_DPI = int(os.environ.get("PLOT_DPI", "180"))
CMAP_DIVERGING = "RdBu"     # positive = red, negative = blue
CMAP_CROPS     = "YlGn"
CMAP_PASTURE   = "YlOrBr"
SHOW_COASTLINES = os.environ.get("SHOW_COASTLINES","1").lower() in ("1","true","y")

def log(s): print(s, flush=True)

# ── IO and math helpers ──────────────────────────────────────────────────────
def open_states(path_or_dir: str, year_min: int, year_max: int) -> xr.Dataset:
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

    try:
        ds = xr.open_dataset(fn, engine=ENGINE, decode_times=TIME_CODER)
        ds = ds.sel(time=slice(f"{year_min}", f"{year_max}"))
        ds = ds.convert_calendar("standard", use_cftime=False)
    except Exception as e:
        log(f"  CF decode failed, fallback, reason: {e}")
        ds = xr.open_dataset(fn, engine=ENGINE, decode_times=False)
        units = str(ds["time"].attrs.get("units", "years since 2015-01-01"))
        m = re.search(r"years since\s*(\d{4})", units)
        base_year = int(m.group(1)) if m else year_min
        offs = np.rint(np.asarray(ds["time"].values)).astype(int)
        yrs = base_year + offs
        keep = (yrs >= year_min) & (yrs <= year_max)
        if not keep.any():
            raise ValueError(f"No years in [{year_min},{year_max}] after fallback decode")
        ds = ds.isel(time=np.where(keep)[0])
        ds = ds.assign_coords(time=("time", pd.to_datetime([f"{y}-01-01" for y in yrs[keep].astype(int)])))

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
    sizes = out.sizes
    y0 = pd.to_datetime(out.time.values[0]).year
    y1 = pd.to_datetime(out.time.values[-1]).year
    log(f"  Grid lat={sizes['lat']} lon={sizes['lon']}, years {y0}-{y1}")
    return out

def cell_area_km2(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    R = 6_371_000.0
    dlat = float(abs(lat[1]-lat[0])); dlon = float(abs(lon[1]-lon[0]))
    dlam = dlon/360.0
    band = np.sin(np.deg2rad(lat + dlat/2.0)) - np.sin(np.deg2rad(lat - dlat/2.0))
    a_lat = (2*np.pi*R**2)*band*dlam/1e6
    return xr.DataArray(np.repeat(a_lat.values[:,None], lon.size, axis=1),
                        dims=("lat","lon"), coords={"lat":lat.values,"lon":lon.values}, name="cell_km2")

CAND_WATER_VARS = ("gicew","icew","ice_water_frac","waterfrac","frac_water","ocean","sftof","areacello","areacellw")

def load_land_fraction_from_static(lat: xr.DataArray, lon: xr.DataArray):
    if not (USE_LAND_MASK and STATIC_PATH and os.path.exists(STATIC_PATH)):
        return None
    log(f"Opening static mask: {STATIC_PATH}")
    ds = xr.open_dataset(STATIC_PATH, engine=ENGINE, decode_times=False)
    if "latitude" in ds.dims:  ds = ds.rename({"latitude":"lat"})
    if "longitude" in ds.dims: ds = ds.rename({"longitude":"lon"})
    name = next((v for v in ds.data_vars if v.lower() in CAND_WATER_VARS), None)
    if name is None:
        return None
    iw = ds[name].astype("float32")
    if not (np.array_equal(iw["lat"].values, lat.values) and np.array_equal(iw["lon"].values, lon.values)):
        iw = iw.interp(lat=lat, lon=lon, method="nearest")
    return (1.0 - iw).clip(0,1).rename("land_frac")

def derive_land_fraction_from_states(ds_states: xr.Dataset) -> xr.DataArray:
    names = ["primf","primn","secdf","secdn","pastr","range","urban","c3ann","c3per","c4ann","c4per","c3nfx"]
    present = [v for v in names if v in ds_states]
    if not present:
        present = [v for v in ["c3ann","c3per","c4ann","c4per","c3nfx","pastr","range","urban"] if v in ds_states]
    log("Deriving land fraction from states first year")
    lf = sum([ds_states[v].isel(time=0).fillna(0.0) for v in present]).clip(0,1).astype("float32").rename("land_frac")
    return lf

def compute_land_fraction(lat: xr.DataArray, lon: xr.DataArray, ref_states: xr.Dataset, cell_km2: xr.DataArray) -> xr.DataArray:
    lf = load_land_fraction_from_static(lat, lon)
    if lf is None:
        log("Static water fraction not found, deriving from states")
        lf = derive_land_fraction_from_states(ref_states)
    land_total = float((cell_km2 * lf).sum().values)
    if not (1.2e8 <= land_total <= 1.8e8):
        log(f"Static-derived land area {land_total/1e6:.1f} Mkm^2 looks wrong, forcing states-derived")
        lf = derive_land_fraction_from_states(ref_states)
    return lf

def land_nan_mask(land_frac: xr.DataArray) -> xr.DataArray:
    return xr.where(land_frac > 0, 1.0, np.nan)

def trend_pp_per_decade(frac_tyl: xr.DataArray) -> xr.DataArray:
    nt = frac_tyl.sizes["time"]
    t = xr.DataArray(np.arange(nt, dtype="float32"), dims=("time",))
    t0 = t - t.mean("time")
    y0 = frac_tyl - frac_tyl.mean("time")
    slope_per_year = (t0 * y0).mean("time") / (t0**2).mean("time")
    return slope_per_year * 100.0 * 10.0

def coord_edges(x1d):
    x = np.asarray(x1d, dtype=float)
    dx = np.diff(x)
    e = np.empty(x.size+1, dtype=float)
    e[1:-1] = (x[:-1] + x[1:]) / 2.0
    e[0]  = x[0] - dx[0]/2.0
    e[-1] = x[-1] + dx[-1]/2.0
    return e

def grid_for_pcolormesh(da: xr.DataArray):
    lat = da["lat"].values
    lon = da["lon"].values
    lat_e = coord_edges(lat)
    lon_e = coord_edges(lon)
    data = da.values
    if lat_e[0] > lat_e[-1]:
        lat_e = lat_e[::-1]
        data = data[::-1, :]
    return lon_e, lat_e, data

# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_ts_two_panels(ts: pd.DataFrame, outfile: str):
    """One page, two panels: top = cropland, bottom = cropland+pasture, multilines across scenarios."""
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    # top: cropland
    for scen in SCENARIOS:
        d = ts[ts["scenario"] == scen]
        axs[0].plot(d["time"], d["frac_cropland"]*100.0, color=SCEN_COLOR[scen], linestyle=LINESTYLE[scen],
                    linewidth=2, label=SCEN_LABEL[scen])
    axs[0].set_ylabel("Percent of land")
    axs[0].set_title("Global fraction of land, Cropland, 2015–2100")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(ncol=3, fontsize=9, loc="upper left")

    # bottom: cropland + pasture
    for scen in SCENARIOS:
        d = ts[ts["scenario"] == scen]
        axs[1].plot(d["time"], d["frac_cropland_pasture"]*100.0, color=SCEN_COLOR[scen], linestyle=LINESTYLE[scen],
                    linewidth=2, label=SCEN_LABEL[scen])
    axs[1].set_ylabel("Percent of land")
    axs[1].set_title("Global fraction of land, Cropland + managed pasture, 2015–2100")
    axs[1].grid(True, alpha=0.3)
    axs[1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI)
    plt.close()
    log(f"Wrote {outfile}")

def plot_spatial_combo_robinson(slopes_c, slopes_cp, outfile):
    """2x3 panel: top row cropland trend, bottom row cropland+pasture trend. Second row label lifted higher."""
    try:
        import cartopy.crs as ccrs
        vmax_c  = np.nanmax([np.nanpercentile(np.abs(slopes_c[s].values), 99)  for s in SCENARIOS])
        vmax_cp = np.nanmax([np.nanpercentile(np.abs(slopes_cp[s].values), 99) for s in SCENARIOS])
        vlim_c  = (-vmax_c, vmax_c); vlim_cp = (-vmax_cp, vmax_cp)

        fig = plt.figure(figsize=(14, 7.6))
        proj = ccrs.Robinson(); pc = ccrs.PlateCarree()

        axes = []
        for i in range(2*len(SCENARIOS)):
            ax = plt.subplot(2, len(SCENARIOS), i+1, projection=proj)
            ax.set_global()
            if SHOW_COASTLINES: ax.coastlines(linewidth=0.4, color="k")
            ax.axis("off")
            axes.append(ax)

        # top row: cropland
        for j, scen in enumerate(SCENARIOS):
            da = slopes_c[scen]
            lon_e, lat_e, data = grid_for_pcolormesh(da)
            mesh = axes[j].pcolormesh(lon_e, lat_e, data, transform=pc,
                                      cmap=CMAP_DIVERGING, vmin=vlim_c[0], vmax=vlim_c[1], shading="auto")
            axes[j].set_title(SCEN_SHORT[scen], fontsize=10)

        # bottom row: cropland+pasture
        for j, scen in enumerate(SCENARIOS):
            da = slopes_cp[scen]
            lon_e, lat_e, data = grid_for_pcolormesh(da)
            mesh2 = axes[len(SCENARIOS)+j].pcolormesh(lon_e, lat_e, data, transform=pc,
                                                      cmap=CMAP_DIVERGING, vmin=vlim_cp[0], vmax=vlim_cp[1], shading="auto")

        # colorbars without borders
        cax1 = fig.add_axes([0.92, 0.59, 0.015, 0.30]); cb1 = fig.colorbar(mesh,  cax=cax1);  cb1.set_label("pp per decade");  cb1.outline.set_visible(False)
        cax2 = fig.add_axes([0.92, 0.13, 0.015, 0.30]); cb2 = fig.colorbar(mesh2, cax=cax2); cb2.set_label("pp per decade"); cb2.outline.set_visible(False)

        # row labels, second label higher
        fig.text(0.05, 0.93, "Cropland trend", fontsize=11, va="center")
        fig.text(0.05, 0.49, "Cropland + managed pasture trend", fontsize=11, va="center")

        plt.tight_layout(rect=[0.05, 0.06, 0.9, 0.99])
        plt.savefig(outfile, dpi=PLOT_DPI); plt.close(); log(f"Wrote {outfile}")

    except Exception as e:
        log(f"Cartopy unavailable, simple fallback ({e})")

def plot_spatial_delta_panel(deltas_c, deltas_cp, outfile):
    try:
        import cartopy.crs as ccrs
        vmax_c  = np.nanmax([np.nanpercentile(np.abs(deltas_c[s].values), 99)  for s in SCENARIOS])
        vmax_cp = np.nanmax([np.nanpercentile(np.abs(deltas_cp[s].values), 99) for s in SCENARIOS])
        vlim_c, vlim_cp = (-vmax_c, vmax_c), (-vmax_cp, vmax_cp)

        fig = plt.figure(figsize=(14, 7.6))
        proj, pc = ccrs.Robinson(), ccrs.PlateCarree()

        axes = []
        for i in range(2*len(SCENARIOS)):
            ax = plt.subplot(2, len(SCENARIOS), i+1, projection=proj)
            ax.set_global(); ax.axis("off")
            if SHOW_COASTLINES: ax.coastlines(linewidth=0.4, color="k")
            axes.append(ax)

        for j, scen in enumerate(SCENARIOS):
            da = deltas_c[scen]
            lon_e, lat_e, data = grid_for_pcolormesh(da)
            m1 = axes[j].pcolormesh(lon_e, lat_e, data, transform=pc,
                                    cmap=CMAP_DIVERGING, vmin=vlim_c[0], vmax=vlim_c[1], shading="auto")
            axes[j].set_title(SCEN_SHORT[scen], fontsize=10)

        for j, scen in enumerate(SCENARIOS):
            da = deltas_cp[scen]
            lon_e, lat_e, data = grid_for_pcolormesh(da)
            m2 = axes[len(SCENARIOS)+j].pcolormesh(lon_e, lat_e, data, transform=pc,
                                                   cmap=CMAP_DIVERGING, vmin=vlim_cp[0], vmax=vlim_cp[1], shading="auto")

        cax1 = fig.add_axes([0.92, 0.59, 0.015, 0.30]); cb1 = fig.colorbar(m1, cax=cax1); cb1.set_label("pp (2100 − 2015)"); cb1.outline.set_visible(False)
        cax2 = fig.add_axes([0.92, 0.13, 0.015, 0.30]); cb2 = fig.colorbar(m2, cax=cax2); cb2.set_label("pp (2100 − 2015)"); cb2.outline.set_visible(False)

        fig.text(0.05, 0.93, "Cropland Δ", fontsize=11, va="center")
        fig.text(0.05, 0.49, "Cropland + managed pasture Δ", fontsize=11, va="center")

        plt.tight_layout(rect=[0.05, 0.06, 0.9, 0.99])
        plt.savefig(outfile, dpi=PLOT_DPI); plt.close(); log(f"Wrote {outfile}")
    except Exception as e:
        log(f"Cartopy delta panel failed ({e})")

def plot_where_crops_vs_pastures(ds_ref: xr.Dataset, land_mask_nan: xr.DataArray, year: int, outfile: str):
    try:
        import cartopy.crs as ccrs
        proj = ccrs.Robinson(); pc = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)

        cro = (ds_ref["crops"].sel(time=f"{year}-01-01") * land_mask_nan).clip(0,1)
        pas = (ds_ref["pastr"].sel(time=f"{year}-01-01") * land_mask_nan).clip(0,1)

        # compute edges and flip safely for pcolormesh
        def grid_for_pmesh(da):
            lat = da["lat"].values; lon = da["lon"].values
            lat_e = coord_edges(lat); lon_e = coord_edges(lon)
            data = da.values
            if lat_e[0] > lat_e[-1]:
                lat_e = lat_e[::-1]; data = data[::-1, :]
            return lon_e, lat_e, data

        lon_e, lat_e, data_c = grid_for_pmesh(cro)
        _,     _,    data_p = grid_for_pmesh(pas)

        # left: cropland
        ax1 = plt.subplot(1,2,1, projection=proj); ax1.set_global(); ax1.axis("off")
        if SHOW_COASTLINES: ax1.coastlines(linewidth=0.4, color="k")
        m1 = ax1.pcolormesh(lon_e, lat_e, data_c, transform=pc, cmap=CMAP_CROPS, vmin=0, vmax=1, shading="auto")
        ax1.set_title(f"Cropland ")
        cb1 = fig.colorbar(m1, ax=ax1, orientation="vertical", fraction=0.04, pad=0.02,shrink=0.5, aspect=20)
        cb1.set_label("cell fraction"); cb1.outline.set_visible(False)

        # right: pasture
        ax2 = plt.subplot(1,2,2, projection=proj); ax2.set_global(); ax2.axis("off")
        if SHOW_COASTLINES: ax2.coastlines(linewidth=0.4, color="k")
        m2 = ax2.pcolormesh(lon_e, lat_e, data_p, transform=pc, cmap=CMAP_PASTURE, vmin=0, vmax=1, shading="auto")
        ax2.set_title(f"Managed pasture ")
        cb2 = fig.colorbar(m2, ax=ax2, orientation="vertical", fraction=0.04, pad=0.02, shrink=0.5,aspect=20)
        cb2.set_label("cell fraction"); cb2.outline.set_visible(False)

        plt.savefig(outfile, dpi=PLOT_DPI)
        plt.close(); log(f"Wrote {outfile}")

    except Exception as e:
        log(f"Cartopy unavailable, simple fallback ({e})")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        def imshow_panel(ax, da, cmap, title):
            arr = da.transpose("lat","lon").values
            lat, lon = da.lat.values, da.lon.values
            if lat[0] > lat[-1]:
                arr = np.flipud(arr); lat = lat[::-1]
            im = ax.imshow(arr, origin="lower", extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                           vmin=0, vmax=1, cmap=cmap)
            ax.set_title(title); ax.axis("off")
            return im
        cro = (ds_ref["crops"].sel(time=f"{year}-01-01") * land_mask_nan).clip(0,1)
        pas = (ds_ref["pastr"].sel(time=f"{year}-01-01") * land_mask_nan).clip(0,1)
        im1 = imshow_panel(axs[0], cro, CMAP_CROPS, f"Cropland fraction, {year}")
        im2 = imshow_panel(axs[1], pas, CMAP_PASTURE, f"Managed pasture fraction, {year}")
        cb1 = fig.colorbar(im1, ax=axs[0], orientation="vertical", fraction=0.04, pad=0.02,shrink=0.5,aspect=20)
        cb2 = fig.colorbar(im2, ax=axs[1], orientation="vertical", fraction=0.04, pad=0.02,shrink=0.5,aspect=20)
        cb1.set_label("fraction of cell"); cb2.set_label("fraction of cell")
        cb1.outline.set_visible(False); cb2.outline.set_visible(False)
        plt.savefig(outfile, dpi=PLOT_DPI)
        plt.close(); log(f"Wrote {outfile}")


    except Exception as e:
        log(f"Cartopy unavailable, simple fallback ({e})")

def plot_delta_hist(deltas_dict, title, outfile, land_km2, land_mask_nan):
    plt.figure(figsize=(6.8,4.3))
    bins = np.linspace(-40, 40, 81)  # pp bins
    w_all = (land_km2 * land_mask_nan).values.ravel()
    for scen in SCENARIOS:
        x = deltas_dict[scen].values.ravel()
        m = np.isfinite(x) & np.isfinite(w_all) & (w_all > 0)
        hist, edges = np.histogram(x[m], bins=bins, weights=w_all[m])
        share = hist / hist.sum() * 100.0
        plt.step(edges[:-1], share, where="post", label=SCEN_LABEL[scen],
                 linewidth=2, color=SCEN_COLOR[scen])
    plt.xlabel("Change in fraction, pp (2100 − 2015)")
    plt.ylabel("Percent of global land")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI)
    plt.close()
    log(f"Wrote {outfile}")

# ── Run ───────────────────────────────────────────────────────────────────────
log("Config")
log(f"  FUTURE_ROOT={FUTURE_ROOT}")
log(f"  ENGINE={ENGINE}")
log(f"  Years: {FUT_START}-{FUT_END}   REF_YEAR={REF_YEAR}")
log(f"  STATIC={STATIC_PATH or 'None'}  USE_LAND_MASK={USE_LAND_MASK}")
log(f"  INCLUDE_RANGELAND={INCLUDE_RANGELAND}")

# reference grid and land fraction
ref = open_states(os.path.join(FUTURE_ROOT, FOLDERS["ssp126"]), FUT_START, FUT_END)
lat, lon = ref.lat, ref.lon
cell_km2 = cell_area_km2(lat, lon)
land_frac = compute_land_fraction(lat, lon, ref_states=ref, cell_km2=cell_km2)
land_km2 = cell_km2 * land_frac
den_land = land_km2.sum(("lat","lon"))
print(f"Land area used: {float(den_land.values)/1e6:.1f} Mkm^2")

# global time series
years = pd.date_range(f"{FUT_START}-01-01", f"{FUT_END}-01-01", freq="YS")
rows = []
for scen in SCENARIOS:
    dpath = os.path.join(FUTURE_ROOT, FOLDERS[scen])
    log(f"Time series, {scen}: {dpath}")
    ds = open_states(dpath, FUT_START, FUT_END)
    if not (np.array_equal(ds.lat.values, lat.values) and np.array_equal(ds.lon.values, lon.values)):
        ds = ds.interp(lat=lat, lon=lon, method="nearest")
    ds = ds.reindex(time=years, method="nearest", tolerance=np.timedelta64(366,"D"))

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
ts.to_csv("lc_fraction_timeseries_futures.csv", index=False); log("Wrote lc_fraction_timeseries_futures.csv")

# two-panel time series, one page
plot_ts_two_panels(ts, "panel_timeseries_crops_and_croppastr.png")

# per–cell trends and endpoint deltas, then composite spatial panels and histograms
land_mask_nan = land_nan_mask(land_frac)
slopes_cropland, slopes_cp = {}, {}
deltas_cropland, deltas_cp = {}, {}

for scen in SCENARIOS:
    dpath = os.path.join(FUTURE_ROOT, FOLDERS[scen])
    log(f"Trends and deltas, {scen}: {dpath}")
    ds = open_states(dpath, FUT_START, FUT_END)
    if not (np.array_equal(ds.lat.values, lat.values) and np.array_equal(ds.lon.values, lon.values)):
        ds = ds.interp(lat=lat, lon=lon, method="nearest")
    ds = ds.reindex(time=years, method="nearest", tolerance=np.timedelta64(366,"D"))

    cropland = ds["crops"].clip(0,1) * land_mask_nan
    cp = (ds["crops"] + ds["pastr"]).clip(0,1)
    if INCLUDE_RANGELAND: cp = (cp + ds["range"]).clip(0,1)
    cp = cp * land_mask_nan

    # linear trend, pp/decade
    slope_c  = trend_pp_per_decade(cropland).rename("trend_cropland_ppdecade")
    slope_cp_ = trend_pp_per_decade(cp).rename("trend_croppastr_ppdecade")
    slopes_cropland[scen] = slope_c
    slopes_cp[scen] = slope_cp_

    # endpoint delta, pp
    c0 = cropland.sel(time=f"{FUT_START}-01-01")
    c1 = cropland.sel(time=f"{FUT_END}-01-01")
    cp0 = cp.sel(time=f"{FUT_START}-01-01")
    cp1 = cp.sel(time=f"{FUT_END}-01-01")
    dC  = ((c1 - c0) * 100.0).rename("delta_cropland_pp")
    dCP = ((cp1 - cp0) * 100.0).rename("delta_croppastr_pp")
    deltas_cropland[scen] = dC
    deltas_cp[scen] = dCP

# 2x3 panels
plot_spatial_combo_robinson(slopes_cropland, slopes_cp, "panel_trend_cropland_and_cp_robinson.png")
plot_spatial_delta_panel(deltas_cropland, deltas_cp, "panel_delta_cropland_and_cp_robinson.png")

# histograms of per-cell changes (area-weighted)
plot_delta_hist(deltas_cropland, "Per-cell cropland change distribution", "hist_delta_cropland.png", land_km2, land_mask_nan)
plot_delta_hist(deltas_cp,       "Per-cell cropland+pasture change distribution", "hist_delta_croppastr.png", land_km2, land_mask_nan)

# where are crops vs pastures in REF_YEAR
plot_where_crops_vs_pastures(ref, land_mask_nan, REF_YEAR, f"map_where_crops_vs_pastures_{REF_YEAR}.png")

log("Done")
