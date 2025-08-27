# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import os, sys, json, glob
# import numpy as np
# import xarray as xr
# from dask.diagnostics import ProgressBar
# from datetime import datetime

# # ====================== Config (env-overridable) ======================
# IN_DIR        = os.environ.get("IN_DIR",  "/projects/prjs1578/futurewetgde/wetGDEs")
# OUT_DIR       = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
# IN_TEMPLATE   = os.environ.get("IN_TEMPLATE",  "wetGDE_{scenario}.nc")
# OUT_TEMPLATE  = os.environ.get("OUT_TEMPLATE", "wetGDE_months_yearly_{scenario}.nc")
# PROGRESS_LOG  = os.environ.get("PROGRESS_LOG", "").strip() or None
# SZ            = int(os.environ.get("SPATIAL_CHUNK", "1024"))
# ENGINE        = "netcdf4"                                # write engine
# READ_ENGINE   = os.environ.get("READ_ENGINE", "netcdf4") # read engine
# NC_MODEL      = "NETCDF4_CLASSIC"
# FORCE_REBUILD = int(os.environ.get("FORCE_REBUILD", "0"))

# DO_PLOTS   = int(os.environ.get("DO_PLOTS", "1"))
# PLOT_DIR   = os.environ.get("PLOT_DIR", os.path.join(OUT_DIR, "plots"))
# WWF_SHAPE  = os.environ.get("WWF_SHAPE", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")

# # slope map visual limits, months per decade
# SLOPE_VLIM  = float(os.environ.get("SLOPE_VLIM", "0.6"))

# PERIODS = json.loads(os.environ.get(
#     "PERIODS_JSON",
#     '{"near":[null,"2021-01-01","2050-12-31"],'
#     ' "mid":[null,"2041-01-01","2070-12-31"],'
#     ' "late":[null,"2071-01-01","2100-12-31"],'
#     ' "baseline":["historical","1985-01-01","2014-12-31"]}'
# ))

# SCENARIOS  = ["historical", "ssp126", "ssp370", "ssp585"]
# PLOT_ORDER = ["historical", "ssp126", "ssp370", "ssp585"]

# SCEN_COLOR = {
#     "historical": "#222222",
#     "ssp126": "#1a9850",
#     "ssp370": "#fdae61",
#     "ssp585": "#d73027",
# }


# # Map short realm codes to full names, pass-through if already full names
# REALM_NAME_MAP = {
#     "AN": "Antarctic",
#     "AT": "Afrotropic",
#     "AU": "Australasia",
#     "IM": "Indomalaya",
#     "NA": "Nearctic",
#     "NT": "Neotropic",
#     "PA": "Palearctic",
#     "OC": "Oceania",
# }
# def to_full_realm_name(s: str) -> str:
#     s = (s or "").strip()
#     return REALM_NAME_MAP.get(s, s)

# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(PLOT_DIR, exist_ok=True)
# READ_CHUNKS = {"time": 1, "lat": SZ, "lon": SZ}

# # ====================== mpl colormap compatibility ======================
# def get_cmap_compat(name):
#     try:
#         from matplotlib import colormaps as mcolormaps  # mpl ≥ 3.6
#         return mcolormaps[name]
#     except Exception:
#         import matplotlib.pyplot as plt                 # older mpl
#         return plt.get_cmap(name)

# # ====================== IO helpers ======================
# def in_path_for(scen):  return os.path.join(IN_DIR,  IN_TEMPLATE.format(scenario=scen))
# def out_path_for(scen): return os.path.join(OUT_DIR, OUT_TEMPLATE.format(scenario=scen))

# def progress_open_for(scen):
#     if not PROGRESS_LOG: return None, None
#     path = PROGRESS_LOG
#     if path.endswith(os.sep) or os.path.isdir(path):
#         os.makedirs(path, exist_ok=True)
#         path = os.path.join(path, f"progress_{scen}.log")
#     def _open(): return open(path, "w", buffering=1)
#     return _open, path

# # ====================== Step 1: build annual counts per scenario ======================
# def build_annual_counts(scen: str):
#     in_path  = in_path_for(scen)
#     out_path = out_path_for(scen)
#     if not os.path.exists(in_path):
#         raise FileNotFoundError(in_path)
#     if os.path.exists(out_path) and not FORCE_REBUILD:
#         print(f"[skip] {scen} exists -> {out_path}")
#         return

#     ds  = xr.open_dataset(in_path, chunks=READ_CHUNKS, engine=READ_ENGINE, use_cftime="auto")
#     wet = ds["wetGDE"]  # 0 dry, 1 wet, 127 fill

#     wet01 = xr.where(wet == 1, 1.0, xr.where(wet == 0, 0.0, np.nan))
#     counts_year = wet01.groupby("time.year").sum(dim="time", skipna=True)
#     counts_year = counts_year.clip(min=0, max=12).astype("f4")

#     years = counts_year["year"].values.astype(int)
#     mid_dates = np.array([np.datetime64(f"{y}-07-01") for y in years], dtype="datetime64[ns]")
#     counts = counts_year.rename({"year": "time"}).assign_coords(time=mid_dates)
#     counts = counts.transpose("time", "lat", "lon")
#     counts.name = "wetGDE_months_yearly"
#     counts.attrs.update({
#         "long_name": "Number of wet months per year",
#         "units": "months",
#         "valid_min": 0.0,
#         "valid_max": 12.0,
#         "count_rule": "sum over months of 1{wetGDE==1}, missing months ignored",
#         "source_mask": os.path.basename(in_path),
#         "date_created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
#     })
#     counts["time"].attrs.update({"long_name": "year midpoint", "standard_name": "time"})

#     enc = {"wetGDE_months_yearly": {"dtype": "f4", "_FillValue": np.float32(-9999.0)}}
#     delayed = counts.to_netcdf(out_path, engine=ENGINE, format=NC_MODEL, encoding=enc, compute=False)

#     opener, _ = progress_open_for(scen)
#     if opener:
#         with opener() as fh:
#             with ProgressBar(out=fh): delayed.compute()
#     else:
#         with ProgressBar(out=sys.stdout): delayed.compute()
#     try: ds.close()
#     except Exception: pass
#     print(f"[done] {scen} -> {out_path}")

# # ====================== Step 2: load ALIGNED data ======================
# def _round_coords(da, nd=6):
#     return da.assign_coords(lat=np.round(da["lat"].values, nd),
#                             lon=np.round(da["lon"].values, nd))

# def _as_year_dim(da):
#     return da.assign_coords(year=da["time"].dt.year).swap_dims({"time":"year"}).drop_vars("time")

# def _open_member_stack(pattern_or_file):
#     if os.path.exists(pattern_or_file):
#         return xr.open_dataset(pattern_or_file, chunks={}, engine=READ_ENGINE, use_cftime="auto")["wetGDE_months_yearly"].expand_dims(member=["m1"])
#     base, ext = os.path.splitext(pattern_or_file)
#     paths = sorted(glob.glob(base + "_*" + ext))
#     if not paths:
#         raise FileNotFoundError(pattern_or_file)
#     if len(paths) == 1:
#         da = xr.open_dataset(paths[0], chunks={}, engine=READ_ENGINE, use_cftime="auto")["wetGDE_months_yearly"].expand_dims(member=[os.path.basename(paths[0])])
#     else:
#         da = xr.open_mfdataset(paths, combine="nested", concat_dim="member", chunks={}, engine=READ_ENGINE, use_cftime="auto")["wetGDE_months_yearly"]
#         da = da.assign_coords(member=[os.path.basename(p) for p in paths])
#     return da

# def load_members_aligned(scen, lat_ref, lon_ref):
#     da = _open_member_stack(out_path_for(scen)).where(lambda z: z >= 0)
#     da = _round_coords(da).sortby(["lat","lon"])
#     da = _as_year_dim(da).sortby("year")
#     da = da.reindex(lat=lat_ref, lon=lon_ref)
#     return da  # dims: member, year, lat, lon

# def load_all_aligned_with_members():
#     dah = _open_member_stack(out_path_for("historical")).where(lambda z: z >= 0)
#     dah = _round_coords(dah).sortby(["lat","lon"])
#     dah = _as_year_dim(dah).sortby("year")
#     lat_ref, lon_ref = dah["lat"], dah["lon"]
#     by = {"historical": dah}
#     for s in ["ssp126","ssp370","ssp585"]:
#         p = out_path_for(s)
#         if not os.path.exists(p) and not glob.glob(os.path.splitext(p)[0] + "_*.nc"):
#             continue
#         by[s] = load_members_aligned(s, lat_ref, lon_ref)
#     return by

# # ====================== Masks and averaging ======================
# def area_weights(lat): return np.cos(np.deg2rad(lat)).clip(min=0)

# def wet_mask_hist_only(da_hist_mem):
#     return (da_hist_mem.max(dim=("member","year")) > 0)

# def wet_mask_hist_plus_future(h, f):
#     return ((h.max(dim=("member","year")) > 0) | (f.max(dim=("member","year")) > 0))

# # ====================== GDAL-free realm rasterization ======================
# def realm_labels_on(dah_lat, dah_lon, shape_path):
#     from cartopy.io import shapereader as shp
#     from shapely.geometry import shape as shp_shape, GeometryCollection
#     from shapely.ops import unary_union, transform as shp_transform
#     from shapely import prepared
#     try:
#         from shapely import vectorized as shp_vec
#         HAS_VEC = True
#     except Exception:
#         HAS_VEC = False
#     from pyproj import CRS, Transformer
#     import os

#     reader = shp.Reader(shape_path)
#     records = list(reader.records())

#     prj_path = os.path.splitext(shape_path)[0] + ".prj"
#     if os.path.exists(prj_path):
#         wkt = open(prj_path, "r").read()
#         crs_src = CRS.from_wkt(wkt)
#     else:
#         crs_src = CRS.from_epsg(4326)
#     crs_dst = CRS.from_epsg(4326)
#     tfm = None if crs_src == crs_dst else Transformer.from_crs(crs_src, crs_dst, always_xy=True).transform

#     geoms_by_realm = {}
#     for rec in records:
#         geom = shp_shape(rec.geometry)
#         if tfm is not None:
#             geom = shp_transform(tfm, geom)
#         attrs = rec.attributes if hasattr(rec, "attributes") else rec.as_dict()
#         realm_raw = None
#         for key in ("REALM","realm","Realm","REALMS"):
#             if key in attrs:
#                 realm_raw = attrs[key]
#                 break
#         if realm_raw is None:
#             continue
#         realm_name = to_full_realm_name(str(realm_raw))

#         geoms_by_realm.setdefault(realm_name, []).append(geom)

#     dissolved = {}
#     for realm_name, geoms in geoms_by_realm.items():
#         geoms = [g for g in geoms if not g.is_empty]
#         if not geoms: 
#             continue
#         try:
#             merged = unary_union(geoms)
#         except Exception:
#             merged = GeometryCollection(geoms)
#         dissolved[realm_name] = merged

#     lat = np.asarray(dah_lat.values)
#     lon = np.asarray(dah_lon.values)
#     Y, X = np.meshgrid(lat, lon, indexing="ij")  # lat x lon

#     labels = np.zeros((lat.size, lon.size), dtype=np.int16)
#     code2name = {}
#     code = 1
#     for realm_name, geom in dissolved.items():
#         if geom.is_empty: 
#             continue
#         pgeom = prepared.prep(geom)
#         if HAS_VEC:
#             mask = shp_vec.contains(geom, X, Y)
#         else:
#             mask = np.zeros_like(labels, dtype=bool)
#             for j in range(X.shape[1]):
#                 pts = np.column_stack([X[:, j], Y[:, j]])
#                 # point-in-polygon without constructing a shapely Point per loop would be faster,
#                 # but this fallback is rarely used
#                 mask[:, j] = np.array([pgeom.contains(shp_shape({"type":"Point","coordinates":tuple(pt)})) for pt in pts], dtype=bool)
#         labels[mask] = code
#         code2name[code] = realm_name  # store full realm name
#         code += 1

#     lab = xr.DataArray(labels, dims=("lat","lon"), coords={"lat":dah_lat, "lon":dah_lon})
#     return lab, code2name

# # ====================== Boundary alignment for slopes ======================
# def align_future_gridwise(h_mean, f_mean, win=5):
#     yH_last = int(h_mean["year"].max())
#     yF_first = int(f_mean["year"].min())
#     Hbar = h_mean.sel(year=slice(yH_last - (win-1), yH_last)).mean("year", skipna=True)
#     Fbar = f_mean.sel(year=slice(yF_first, yF_first + (win-1))).mean("year", skipna=True)
#     offset = Hbar - Fbar
#     return f_mean + offset

# def slope_months_per_decade(da_year_lat_lon, years_min=20):
#     valid = np.isfinite(da_year_lat_lon)
#     n_ok = valid.sum("year")
#     pf = da_year_lat_lon.polyfit(dim="year", deg=1)
#     slope = pf.polyfit_coefficients.sel(degree=1) * 10.0
#     return slope.where(n_ok >= years_min)

# def slope_map_hist_plus_future(by, scen, years_min=20, align=True):
#     h = by["historical"].mean("member")
#     f = by[scen].mean("member") if scen in by else None
#     if f is None: return None
#     hist_end = int(h["year"].max())
#     if align:
#         f = align_future_gridwise(h, f, win=5)
#     f = f.sel(year=f["year"] > hist_end)          # hard break at boundary
#     combined = xr.concat([h, f], dim="year")      # historical + scenario
#     mask = wet_mask_hist_plus_future(by["historical"], by[scen])
#     return slope_months_per_decade(combined.where(mask), years_min=years_min)

# # ====================== TS helpers ======================
# def smooth_series(y_vals, years, win=5):
#     if win is None or win <= 1: return y_vals
#     da = xr.DataArray(y_vals, coords={"year": years})
#     return da.rolling(year=win, center=True, min_periods=max(1, win//2)).mean().values

# def realm_series(mask_realm, by, smooth_win=5):
#     lat = by["historical"]["lat"]
#     w_lat = area_weights(lat)

#     def member_stack_realm(da_mem, mask_bool):
#         return da_mem.where(mask_bool).weighted(w_lat).mean(("lat","lon"))  # member,year

#     mh = mask_realm & wet_mask_hist_only(by["historical"])
#     h_mem = member_stack_realm(by["historical"], mh)
#     h_mean = h_mem.mean("member")
#     hx = h_mean["year"].values.astype(float)
#     hy = smooth_series(h_mean.values, h_mean["year"].values, smooth_win)
#     hist_end = int(h_mean["year"].max())
#     join_val = float(hy[-1]) if hy.size else np.nan

#     out = {"historical": (hx, hy, None, None), "hist_end": hist_end, "join_val": join_val}

#     for scen in ["ssp126","ssp370","ssp585"]:
#         if scen not in by: continue
#         mf = mask_realm & wet_mask_hist_plus_future(by["historical"], by[scen])
#         f_mem = member_stack_realm(by[scen], mf)
#         f_mean = f_mem.mean("member")
#         sel = f_mean["year"].values > hist_end
#         if not np.any(sel): continue
#         f_mean = f_mean.sel(year=f_mean["year"].values[sel])
#         fx = f_mean["year"].values.astype(float)
#         fy = smooth_series(f_mean.values, f_mean["year"].values, smooth_win)
#         off = (join_val - float(fy[0])) if fy.size else 0.0

#         # shaded 10–90 percentile if members ≥ 2, else none
#         ylo = yhi = None
#         if "member" in f_mem.dims and f_mem.sizes.get("member", 1) >= 2:
#             q10 = f_mem.quantile(0.10, dim="member").sel(year=f_mean["year"])
#             q90 = f_mem.quantile(0.90, dim="member").sel(year=f_mean["year"])
#             ylo = smooth_series(q10.values, q10["year"].values, smooth_win) + off
#             yhi = smooth_series(q90.values, q90["year"].values, smooth_win) + off

#         out[scen] = (fx, fy + off, ylo, yhi)
#     return out

# # ====================== Plotting: all realms on one page, shared y label ======================
# def plot_all_realms_grid(by, labels, code2name, out_png, smooth_win=5, shade=True, ncols=4):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#     from matplotlib.ticker import MultipleLocator

#     # exclude Antarctic from time series and ensure full names in titles
#     items = sorted(
#         [(code, name) for code, name in code2name.items()
#          if name not in ("Antarctic", "AN")],
#         key=lambda kv: kv[1]
#     )
#     n = len(items)
#     ncols = min(ncols, max(1, n))
#     nrows = int(np.ceil(n / ncols))

#     fig = plt.figure(figsize=(4.0*ncols, 2.8*nrows + 0.9), dpi=220)

#     handles = [Line2D([0],[0], color=SCEN_COLOR[s], lw=2) for s in PLOT_ORDER]
#     labels_legend = ["Historical", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]

#     hist_end = int(by["historical"]["year"].max())

#     for idx, (code, full_name) in enumerate(items, start=1):
#         ax = plt.subplot(nrows, ncols, idx)
#         series = realm_series(labels == code, by, smooth_win=smooth_win)

#         hx, hy, _, _ = series["historical"]
#         ax.plot(hx, hy, lw=2, color=SCEN_COLOR["historical"])

#         for scen in ["ssp126","ssp370","ssp585"]:
#             if scen not in series: 
#                 continue
#             fx, fy, ylo, yhi = series[scen]
#             ax.plot(fx, fy, lw=2, color=SCEN_COLOR[scen])
#             if shade and (ylo is not None) and (yhi is not None):
#                 ax.fill_between(fx, ylo, yhi, color=SCEN_COLOR[scen], alpha=0.18, linewidth=0)

#         ax.axvline(hist_end, color="0.6", lw=1)

#         # per-panel y range from its own data, integer ticks
#         ys = []
#         for ln in ax.get_lines():
#             y = np.asarray(ln.get_ydata(), dtype=float)
#             if y.size: ys.append(y)
#         if ys:
#             yall  = np.hstack(ys)
#             ylow  = float(np.nanpercentile(yall, 1))
#             yhigh = float(np.nanpercentile(yall, 99))
#             pad   = 0.2
#             ymin  = max(0.0, ylow  - pad)
#             ymax  = min(12.0, yhigh + pad)
#             if ymax - ymin < 0.5:
#                 ymax = min(12.0, ymin + 0.5)
#             ax.set_ylim(ymin, ymax)
#             yt0, yt1 = ax.get_ylim()
#             low  = int(np.floor(yt0))
#             high = int(np.ceil(yt1))
#             ax.set_yticks(np.arange(low, high + 1, 1))
#             ax.yaxis.set_minor_locator(MultipleLocator(0.5))

#         # per-panel x ticks
#         xs = []
#         for ln in ax.get_lines():
#             x = np.asarray(ln.get_xdata(), dtype=float)
#             if x.size: xs.append(x)
#         if xs:
#             xmin = int(np.floor(min(arr.min() for arr in xs)))
#             xmax = int(np.ceil (max(arr.max() for arr in xs)))
#             step = 20 if (xmax - xmin) > 120 else 10
#             ax.set_xticks(np.arange(xmin - xmin % step, xmax + step, step))

#         ax.set_title(full_name)  # use full realm name as title
#         ax.grid(False, lw=0.3, alpha=0.5)
#         # keep x labels only on last row
#         if idx <= (nrows-1)*ncols:
#             ax.set_xticklabels([])

#     # single shared y-axis label for the whole figure
#     fig.text(0.03, 0.5, "Months per year", rotation="vertical", va="center", ha="left")

#     # common legend at bottom
#     fig.legend(handles, labels_legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015))

#     fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.97])
#     fig.savefig(out_png, bbox_inches="tight")
#     plt.close(fig)

# # ====================== Spatial 1×3, slopes of (historical + each SSP) ======================
# def spatial_three_panel_slopes(by, out_png, years_min=20, align=True, vlim=None):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import cartopy.crs as ccrs, cartopy.feature as cfeature
#     from matplotlib import colors as mcolors
#     import matplotlib.cm as mcm

#     # compute slopes for the three combined series
#     maps = []
#     order = [("ssp126", "SSP1-2.6"), ("ssp370", "SSP3-7.0"), ("ssp585", "SSP5-8.5")]
#     for scen, label in order:
#         if scen not in by: 
#             maps.append((label, None))
#             continue
#         m = slope_map_hist_plus_future(by, scen, years_min=years_min, align=align)
#         maps.append((label, m))

#     # color scale from these three maps only
#     if vlim is None:
#         vals = []
#         for _, m in maps:
#             if m is not None:
#                 a = np.abs(m.values)
#                 if np.any(np.isfinite(a)):
#                     vals.append(a[np.isfinite(a)])
#         vmax = np.nanpercentile(np.concatenate(vals), 99) if vals else SLOPE_VLIM
#         vlim = max(SLOPE_VLIM, float(vmax))
#     norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
#     cmap = get_cmap_compat("RdBu_r")

#     fig = plt.figure(figsize=(12.6, 4.8), dpi=220)
#     proj = ccrs.Robinson()

#     for i in range(3):
#         ax = plt.subplot(1, 3, i+1, projection=proj)
#         ax.set_global(); ax.coastlines(linewidth=0.4)
#         ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.2)
#         if i < len(maps) and maps[i][1] is not None:
#             title, da2d = maps[i]
#             ax.pcolormesh(da2d["lon"], da2d["lat"], da2d,
#                           transform=ccrs.PlateCarree(), shading="auto",
#                           norm=norm, cmap=cmap)
#             ax.set_title(title, pad=2, fontsize=10)
#         else:
#             ax.set_title("missing", pad=2, fontsize=10)
#         ax.set_axis_off()
#         try:
#             ax.outline_patch.set_visible(False); ax.background_patch.set_visible(False)
#         except Exception: pass

#     cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
#     sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
#     cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
#     cb.set_label("trend, months per decade")
#     fig.tight_layout(rect=[0,0.10,1,0.98])
#     fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)

# # ====================== Driver ======================
# def make_plots():
#     by = load_all_aligned_with_members()
#     if "historical" not in by:
#         print("[warn] missing historical, aborting plots"); return

#     dah_mem = by["historical"]
#     lat = dah_mem["lat"]; lon = dah_mem["lon"]

#     labels, code2name = realm_labels_on(lat, lon, WWF_SHAPE)

#     # 1) multi-realm TS page (Antarctic excluded, titles are full realm names)
#     out_grid = os.path.join(PLOT_DIR, "ts_values_all_realms_grid.png")
#     plot_all_realms_grid(by, labels, code2name, out_grid, smooth_win=5, shade=True, ncols=4)

#     # 2) spatial slopes page, hist + each SSP
#     out_slope = os.path.join(PLOT_DIR, "panel_slopes_hist_plus_each_SSP_1x3.png")
#     spatial_three_panel_slopes(by, out_slope, years_min=20, align=True, vlim=SLOPE_VLIM)

# # ====================== Main ======================
# if __name__ == "__main__":
#     print(f"[setup] IN_DIR={IN_DIR}")
#     print(f"[setup] OUT_DIR={OUT_DIR}")
#     print(f"[setup] SCENARIOS={SCENARIOS}")
#     print(f"[setup] plotting -> {PLOT_DIR}")

#     for s in SCENARIOS:
#         try:
#             build_annual_counts(s)
#         except Exception as e:
#             ts = datetime.now().isoformat(timespec="seconds")
#             print(f"[error] {ts} | {s} | {e}")

#     if DO_PLOTS:
#         try:
#             make_plots()
#         except Exception as e:
#             print(f"[warn] plotting skipped: {e}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, glob
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from datetime import datetime

# ====================== Config (env-overridable) ======================
IN_DIR        = os.environ.get("IN_DIR",  "/projects/prjs1578/futurewetgde/wetGDEs")
OUT_DIR       = os.environ.get("OUT_DIR", "/projects/prjs1578/futurewetgde/wetGDEs")
IN_TEMPLATE   = os.environ.get("IN_TEMPLATE",  "wetGDE_{scenario}.nc")
OUT_TEMPLATE  = os.environ.get("OUT_TEMPLATE", "wetGDE_months_yearly_{scenario}.nc")
PROGRESS_LOG  = os.environ.get("PROGRESS_LOG", "").strip() or None
SZ            = int(os.environ.get("SPATIAL_CHUNK", "1024"))
ENGINE        = "netcdf4"                                # write engine
READ_ENGINE   = os.environ.get("READ_ENGINE", "netcdf4") # read engine
NC_MODEL      = "NETCDF4_CLASSIC"
FORCE_REBUILD = int(os.environ.get("FORCE_REBUILD", "0"))

DO_PLOTS   = int(os.environ.get("DO_PLOTS", "1"))
PLOT_DIR   = os.environ.get("PLOT_DIR", os.path.join(OUT_DIR, "plots_new"))
WWF_SHAPE  = os.environ.get("WWF_SHAPE", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")

# slope map visual limits, months per decade
SLOPE_VLIM  = float(os.environ.get("SLOPE_VLIM", "0.6"))

PERIODS = json.loads(os.environ.get(
    "PERIODS_JSON",
    '{"near":[null,"2021-01-01","2050-12-31"],'
    ' "mid":[null,"2041-01-01","2070-12-31"],'
    ' "late":[null,"2071-01-01","2100-12-31"],'
    ' "baseline":["historical","1985-01-01","2014-12-31"]}'
))

SCENARIOS  = ["historical", "ssp126", "ssp370", "ssp585"]
PLOT_ORDER = ["historical", "ssp126", "ssp370", "ssp585"]

SCEN_COLOR = {
    "historical": "#222222",
    "ssp126": "#1a9850",
    "ssp370": "#fdae61",
    "ssp585": "#d73027",
}

# Map short realm codes to full names, pass-through if already full names
REALM_NAME_MAP = {
    "AA": "Australasian",
    "AT": "Afrotropical",
    "IM": "Indo Malayan",
    "NA": "Nearctic",
    "NT": "Neotropical",
    "OC": "Oceanian",
    "PA": "Palearctic",
    "AN": "Antarctic",
}
def to_full_realm_name(s: str) -> str:
    s = (s or "").strip()
    return REALM_NAME_MAP.get(s, s)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
READ_CHUNKS = {"time": 1, "lat": SZ, "lon": SZ}

# xarray CF-time decoding (replacement for deprecated use_cftime)
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# ====================== mpl colormap compatibility ======================
def get_cmap_compat(name):
    try:
        from matplotlib import colormaps as mcolormaps  # mpl ≥ 3.6
        return mcolormaps[name]
    except Exception:
        import matplotlib.pyplot as plt                 # older mpl
        return plt.get_cmap(name)

# ====================== IO helpers ======================
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

# ====================== Step 1: build annual counts per scenario ======================
def build_annual_counts(scen: str):
    in_path  = in_path_for(scen)
    out_path = out_path_for(scen)
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    if os.path.exists(out_path) and not FORCE_REBUILD:
        print(f"[skip] {scen} exists -> {out_path}")
        return

    ds  = xr.open_dataset(in_path, chunks=READ_CHUNKS, engine=READ_ENGINE, decode_times=TIME_CODER)
    wet = ds["wetGDE"]  # 0 dry, 1 wet, 127 fill

    wet01 = xr.where(wet == 1, 1.0, xr.where(wet == 0, 0.0, np.nan))
    counts_year = wet01.groupby("time.year").sum(dim="time", skipna=True)
    counts_year = counts_year.clip(min=0, max=12).astype("f4")

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

    enc = {"wetGDE_months_yearly": {"dtype": "f4", "_FillValue": np.float32(-9999.0)}}
    delayed = counts.to_netcdf(out_path, engine=ENGINE, format=NC_MODEL, encoding=enc, compute=False)

    opener, _ = progress_open_for(scen)
    if opener:
        with opener() as fh:
            with ProgressBar(out=fh): delayed.compute()
    else:
        with ProgressBar(out=sys.stdout): delayed.compute()
    try: ds.close()
    except Exception: pass
    print(f"[done] {scen} -> {out_path}")

# ====================== Step 2: load ALIGNED data ======================
def _round_coords(da, nd=6):
    return da.assign_coords(lat=np.round(da["lat"].values, nd),
                            lon=np.round(da["lon"].values, nd))

def _as_year_dim(da):
    return da.assign_coords(year=da["time"].dt.year).swap_dims({"time":"year"}).drop_vars("time")

def _open_member_stack(pattern_or_file):
    if os.path.exists(pattern_or_file):
        return xr.open_dataset(pattern_or_file, chunks={}, engine=READ_ENGINE, decode_times=TIME_CODER)["wetGDE_months_yearly"].expand_dims(member=["m1"])
    base, ext = os.path.splitext(pattern_or_file)
    paths = sorted(glob.glob(base + "_*" + ext))
    if not paths:
        raise FileNotFoundError(pattern_or_file)
    if len(paths) == 1:
        da = xr.open_dataset(paths[0], chunks={}, engine=READ_ENGINE, decode_times=TIME_CODER)["wetGDE_months_yearly"].expand_dims(member=[os.path.basename(paths[0])])
    else:
        da = xr.open_mfdataset(paths, combine="nested", concat_dim="member", chunks={}, engine=READ_ENGINE, decode_times=TIME_CODER)["wetGDE_months_yearly"]
        da = da.assign_coords(member=[os.path.basename(p) for p in paths])
    return da

def load_members_aligned(scen, lat_ref, lon_ref):
    da = _open_member_stack(out_path_for(scen)).where(lambda z: z >= 0)
    da = _round_coords(da).sortby(["lat","lon"])
    da = _as_year_dim(da).sortby("year")
    da = da.reindex(lat=lat_ref, lon=lon_ref)
    return da  # dims: member, year, lat, lon

def load_all_aligned_with_members():
    dah = _open_member_stack(out_path_for("historical")).where(lambda z: z >= 0)
    dah = _round_coords(dah).sortby(["lat","lon"])
    dah = _as_year_dim(dah).sortby("year")
    lat_ref, lon_ref = dah["lat"], dah["lon"]
    by = {"historical": dah}
    for s in ["ssp126","ssp370","ssp585"]:
        p = out_path_for(s)
        if not os.path.exists(p) and not glob.glob(os.path.splitext(p)[0] + "_*.nc"):
            continue
        by[s] = load_members_aligned(s, lat_ref, lon_ref)
    return by

# ====================== Masks and averaging ======================
def area_weights(lat): return np.cos(np.deg2rad(lat)).clip(min=0)

def wet_mask_hist_only(da_hist_mem):
    return (da_hist_mem.max(dim=("member","year")) > 0)

def wet_mask_hist_plus_future(h, f):
    return ((h.max(dim=("member","year")) > 0) | (f.max(dim=("member","year")) > 0))

# ====================== Realm labels without shapely/geopandas ======================
def realm_labels_on(dah_lat, dah_lon, shape_path):
    from matplotlib.path import Path as MplPath
    import shapefile  # pyshp

    # force latin1 DBF decoding to avoid utf-8 errors
    r = shapefile.Reader(shape_path, encoding="latin1")
    fnames = [f[0] for f in r.fields[1:]]  # skip DeletionFlag

    # collect polygons per realm name
    polys_by_realm = {}
    for sr in r.iterShapeRecords():
        attrs = {fnames[i]: sr.record[i] for i in range(len(fnames))}
        realm_raw = None
        for key in ("REALM","realm","Realm","REALMS"):
            if key in attrs:
                realm_raw = attrs[key]
                break
        if realm_raw is None:
            continue
        realm = to_full_realm_name(str(realm_raw))

        pts = np.asarray(sr.shape.points, dtype=float)
        if pts.size == 0:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        rings = [pts[parts[i]:parts[i+1]] for i in range(len(parts)-1)]
        polys_by_realm.setdefault(realm, []).append(rings)

    # rasterize realms to label grid using even-odd polygon fill
    lat = np.asarray(dah_lat.values)
    lon = np.asarray(dah_lon.values)
    LON2, LAT2 = np.meshgrid(lon, lat)  # lon x lat grids for point tests

    names = sorted(polys_by_realm.keys())
    code_map = {nm: i+1 for i, nm in enumerate(names)}
    labels = np.zeros((lat.size, lon.size), dtype=np.int16)

    for nm in names:
        code = code_map[nm]
        for rings in polys_by_realm[nm]:
            ext = rings[0]
            xmin, ymin = ext.min(axis=0)
            xmax, ymax = ext.max(axis=0)
            inside_bbox = (LON2 >= xmin) & (LON2 <= xmax) & (LAT2 >= ymin) & (LAT2 <= ymax)
            idx = np.where(inside_bbox)
            if idx[0].size == 0:
                continue
            pts = np.column_stack([LON2[idx], LAT2[idx]])
            verts, codes = [], []
            for ring in rings:
                verts.append(tuple(ring[0])); codes.append(MplPath.MOVETO)
                for v in ring[1:]:
                    verts.append(tuple(v)); codes.append(MplPath.LINETO)
                verts.append(tuple(ring[0])); codes.append(MplPath.CLOSEPOLY)
            path = MplPath(verts, codes)
            mask = path.contains_points(pts, radius=0.0)
            labels[idx[0][mask], idx[1][mask]] = code

    lab = xr.DataArray(labels, dims=("lat","lon"), coords={"lat":dah_lat, "lon":dah_lon})
    code2name = {v:k for k,v in code_map.items()}
    return lab, code2name

# ====================== Boundary alignment for slopes ======================
def align_future_gridwise(h_mean, f_mean, win=5):
    yH_last = int(h_mean["year"].max())
    yF_first = int(f_mean["year"].min())
    Hbar = h_mean.sel(year=slice(yH_last - (win-1), yH_last)).mean("year", skipna=True)
    Fbar = f_mean.sel(year=slice(yF_first, yF_first + (win-1))).mean("year", skipna=True)
    offset = Hbar - Fbar
    return f_mean + offset

def slope_months_per_decade(da_year_lat_lon, years_min=20):
    valid = np.isfinite(da_year_lat_lon)
    n_ok = valid.sum("year")
    pf = da_year_lat_lon.polyfit(dim="year", deg=1)
    slope = pf.polyfit_coefficients.sel(degree=1) * 10.0
    return slope.where(n_ok >= years_min)

def slope_map_hist_plus_future(by, scen, years_min=20, align=True):
    h = by["historical"].mean("member")
    f = by[scen].mean("member") if scen in by else None
    if f is None: return None
    hist_end = int(h["year"].max())
    if align:
        f = align_future_gridwise(h, f, win=5)
    f = f.sel(year=f["year"] > hist_end)          # hard break at boundary
    combined = xr.concat([h, f], dim="year")      # historical + scenario
    mask = wet_mask_hist_plus_future(by["historical"], by[scen])
    return slope_months_per_decade(combined.where(mask), years_min=years_min)

# ====================== TS helpers ======================
def smooth_series(y_vals, years, win=5):
    if win is None or win <= 1: return y_vals
    da = xr.DataArray(y_vals, coords={"year": years})
    return da.rolling(year=win, center=True, min_periods=max(1, win//2)).mean().values

def realm_series(mask_realm, by, smooth_win=5):
    lat = by["historical"]["lat"]
    w_lat = area_weights(lat)

    def member_stack_realm(da_mem, mask_bool):
        return da_mem.where(mask_bool).weighted(w_lat).mean(("lat","lon"))  # member,year

    mh = mask_realm & wet_mask_hist_only(by["historical"])
    h_mem = member_stack_realm(by["historical"], mh)
    h_mean = h_mem.mean("member")
    hx = h_mean["year"].values.astype(float)
    hy = smooth_series(h_mean.values, h_mean["year"].values, smooth_win)
    hist_end = int(h_mean["year"].max())
    join_val = float(hy[-1]) if hy.size else np.nan

    out = {"historical": (hx, hy, None, None), "hist_end": hist_end, "join_val": join_val}

    for scen in ["ssp126","ssp370","ssp585"]:
        if scen not in by: continue
        mf = mask_realm & wet_mask_hist_plus_future(by["historical"], by[scen])
        f_mem = member_stack_realm(by[scen], mf)
        f_mean = f_mem.mean("member")
        sel = f_mean["year"].values > hist_end
        if not np.any(sel): continue
        f_mean = f_mean.sel(year=f_mean["year"].values[sel])
        fx = f_mean["year"].values.astype(float)
        fy = smooth_series(f_mean.values, f_mean["year"].values, smooth_win)
        off = (join_val - float(fy[0])) if fy.size else 0.0

        # shaded 10–90 percentile if members ≥ 2, else none
        ylo = yhi = None
        if "member" in f_mem.dims and f_mem.sizes.get("member", 1) >= 2:
            q10 = f_mem.quantile(0.10, dim="member").sel(year=f_mean["year"])
            q90 = f_mem.quantile(0.90, dim="member").sel(year=f_mean["year"])
            ylo = smooth_series(q10.values, q10["year"].values, smooth_win) + off
            yhi = smooth_series(q90.values, q90["year"].values, smooth_win) + off

        out[scen] = (fx, fy + off, ylo, yhi)
    return out

# ====================== Plotting: all realms on one page, shared y label ======================
def plot_all_realms_grid(by, labels, code2name, out_png, smooth_win=5, shade=True, ncols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    # exclude Antarctic from time series and ensure full names in titles
    items = sorted(
        [(code, name) for code, name in code2name.items()
         if name not in ("Antarctic", "AN")],
        key=lambda kv: kv[1]
    )
    n = len(items)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(4.0*ncols, 2.8*nrows + 0.9), dpi=220)

    handles = [Line2D([0],[0], color=SCEN_COLOR[s], lw=2) for s in PLOT_ORDER]
    labels_legend = ["Historical", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]

    hist_end = int(by["historical"]["year"].max())

    for idx, (code, full_name) in enumerate(items, start=1):
        ax = plt.subplot(nrows, ncols, idx)
        series = realm_series(labels == code, by, smooth_win=smooth_win)

        hx, hy, _, _ = series["historical"]
        ax.plot(hx, hy, lw=2, color=SCEN_COLOR["historical"])

        for scen in ["ssp126","ssp370","ssp585"]:
            if scen not in series: 
                continue
            fx, fy, ylo, yhi = series[scen]
            ax.plot(fx, fy, lw=2, color=SCEN_COLOR[scen])
            if shade and (ylo is not None) and (yhi is not None):
                ax.fill_between(fx, ylo, yhi, color=SCEN_COLOR[scen], alpha=0.18, linewidth=0)

        ax.axvline(hist_end, color="0.6", lw=1)

        # per-panel y range from its own data, integer ticks
        ys = []
        for ln in ax.get_lines():
            y = np.asarray(ln.get_ydata(), dtype=float)
            if y.size: ys.append(y)
        if ys:
            yall  = np.hstack(ys)
            ylow  = float(np.nanpercentile(yall, 1))
            yhigh = float(np.nanpercentile(yall, 99))
            pad   = 0.2
            ymin  = max(0.0, ylow  - pad)
            ymax  = min(12.0, yhigh + pad)
            if ymax - ymin < 0.5:
                ymax = min(12.0, ymin + 0.5)
            ax.set_ylim(ymin, ymax)
            yt0, yt1 = ax.get_ylim()
            low  = int(np.floor(yt0))
            high = int(np.ceil(yt1))
            ax.set_yticks(np.arange(low, high + 1, 1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        # per-panel x ticks
        xs = []
        for ln in ax.get_lines():
            x = np.asarray(ln.get_xdata(), dtype=float)
            if x.size: xs.append(x)
        if xs:
            xmin = int(np.floor(min(arr.min() for arr in xs)))
            xmax = int(np.ceil (max(arr.max() for arr in xs)))
            step = 20 if (xmax - xmin) > 120 else 10
            ax.set_xticks(np.arange(xmin - xmin % step, xmax + step, step))

        ax.set_title(full_name)  # use full realm name as title
        ax.grid(False, lw=0.3, alpha=0.5)
        # keep x labels only on last row
        if idx <= (nrows-1)*ncols:
            ax.set_xticklabels([])

    # single shared y-axis label for the whole figure
    fig.text(0.03, 0.5, "Months per year", rotation="vertical", va="center", ha="left")

    # common legend at bottom
    fig.legend(handles, labels_legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015))

    fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.97])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ====================== Spatial 1×3, slopes of (historical + each SSP) ======================
def spatial_three_panel_slopes(by, out_png, years_min=20, align=True, vlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    from matplotlib import colors as mcolors
    import matplotlib.cm as mcm

    # compute slopes for the three combined series
    maps = []
    order = [("ssp126", "SSP1-2.6"), ("ssp370", "SSP3-7.0"), ("ssp585", "SSP5-8.5")]
    for scen, label in order:
        if scen not in by: 
            maps.append((label, None))
            continue
        m = slope_map_hist_plus_future(by, scen, years_min=years_min, align=align)
        maps.append((label, m))

    # color scale from these three maps only
    if vlim is None:
        vals = []
        for _, m in maps:
            if m is not None:
                a = np.abs(m.values)
                if np.any(np.isfinite(a)):
                    vals.append(a[np.isfinite(a)])
        vmax = np.nanpercentile(np.concatenate(vals), 99) if vals else SLOPE_VLIM
        vlim = max(SLOPE_VLIM, float(vmax))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    cmap = get_cmap_compat("RdBu_r")

    fig = plt.figure(figsize=(12.6, 4.8), dpi=220)
    proj = ccrs.Robinson()

    for i in range(3):
        ax = plt.subplot(1, 3, i+1, projection=proj)
        ax.set_global(); ax.coastlines(linewidth=0.1)
        # ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.2)
        if i < len(maps) and maps[i][1] is not None:
            title, da2d = maps[i]
            ax.pcolormesh(da2d["lon"], da2d["lat"], da2d,
                          transform=ccrs.PlateCarree(), shading="auto",
                          norm=norm, cmap=cmap)
            ax.set_title(title, pad=2, fontsize=10)
        else:
            ax.set_title("missing", pad=2, fontsize=10)
        ax.set_axis_off()
        try:
            ax.outline_patch.set_visible(False); ax.background_patch.set_visible(False)
        except Exception: pass

    cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")

    # Label
    cb.set_label("Trend (months per decade)")

    # Make outline visible
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(0.8)

    # Glossy overlay: semi-transparent white gradient
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtransforms

    # Add glossy rectangle covering top half
    trans = mtransforms.blended_transform_factory(cax.transAxes, cax.transAxes)
    rect = Rectangle((0, 0.5), 1, 0.5, transform=trans,
                    facecolor="white", alpha=0.15, zorder=10)
    cax.add_patch(rect)

    # Layout and save
    fig.tight_layout(rect=[0, 0.10, 1, 0.98])
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


# ====================== Driver ======================
def make_plots():
    by = load_all_aligned_with_members()
    if "historical" not in by:
        print("[warn] missing historical, aborting plots"); return

    dah_mem = by["historical"]
    lat = dah_mem["lat"]; lon = dah_mem["lon"]

    labels, code2name = realm_labels_on(lat, lon, WWF_SHAPE)

    # 1) multi-realm TS page (Antarctic excluded, titles are full realm names)
    out_grid = os.path.join(PLOT_DIR, "ts_values_all_realms_grid.png")
    plot_all_realms_grid(by, labels, code2name, out_grid, smooth_win=5, shade=True, ncols=4)

    # 2) spatial slopes page, hist + each SSP
    out_slope = os.path.join(PLOT_DIR, "panel_slopes_hist_plus_each_SSP_1x3.png")
    spatial_three_panel_slopes(by, out_slope, years_min=20, align=True, vlim=SLOPE_VLIM)

# ====================== Main ======================
if __name__ == "__main__":
    print(f"[setup] IN_DIR={IN_DIR}")
    print(f"[setup] OUT_DIR={OUT_DIR}")
    print(f"[setup] SCENARIOS={SCENARIOS}")
    print(f"[setup] plotting -> {PLOT_DIR}")

    for s in SCENARIOS:
        try:
            build_annual_counts(s)
        except Exception as e:
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"[error] {ts} | {s} | {e}")

    if DO_PLOTS:
        try:
            make_plots()
        except Exception as e:
            print(f"[warn] plotting skipped: {e}")
