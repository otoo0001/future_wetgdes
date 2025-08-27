# #!/usr/bin/env python3
# # Time series and spatial plots for GDE area + animation
# # - IPCC time series per realm: historical black, SSPs colored, vertical line at end of historical
# # - Panel variant with percentile shading and a lines only variant
# # - Spatial percent change per BIOME x REALM for each SSP using Theil Sen endpoints
# # - External colorbar without borders, glossy overlay
# # - Realm trend text saved to a .txt file
# # - Multi panel time series animation with y tick labels in scientific notation and legend at bottom
# # - Prints progress to stdout

# import os
# from pathlib import Path
# import shutil
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
# from matplotlib.ticker import ScalarFormatter
# from matplotlib import animation
# from scipy.stats import theilslopes
# import cartopy.crs as ccrs

# # ── Config ────────────────────────────────────────────────────────────────────
# PARQUET_GLOB = os.environ.get(
#     "PARQUET_GLOB",
#     "/projects/prjs1578/futurewetgde/wetGDEs_area/gde_area_by_biome_realm_monthly_*.parquet",
# )
# BIOME_SHP = os.environ.get(
#     "BIOME_SHP",
#     "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp",
# )
# OUT_DIR = os.environ.get("PLOT_OUT", "/projects/prjs1578/futurewetgde/figs_gde_area")

# YEAR_MIN = int(os.environ.get("YEAR_MIN", "1980"))
# YEAR_MAX = int(os.environ.get("YEAR_MAX", "2100"))
# NORMALIZE = os.environ.get("NORMALIZE", "0").lower() in ("1", "true", "yes", "y")
# BASELINE = tuple(int(x) for x in os.environ.get("BASELINE", "1995,2014").split(","))
# TREND_DECIMALS = int(os.environ.get("TREND_DECIMALS", "0"))
# ANIM_FPS = int(os.environ.get("ANIM_FPS", "8"))

# SCEN_COLORS = {"ssp126": "#1a9850", "ssp370": "#fdae61", "ssp585": "#d73027"}
# REALM_NAME = {
#     "AA": "Australasian",
#     "AT": "Afrotropical",
#     "IM": "Indo Malayan",
#     "NA": "Nearctic",
#     "NT": "Neotropical",
#     "OC": "Oceanian",
#     "PA": "Palearctic",
#     "AN": "Antarctic",
# }

# DPI = int(os.environ.get("DPI", "400"))
# Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# # ── IO ────────────────────────────────────────────────────────────────────────
# def read_parquets(glob_pat: str) -> pd.DataFrame:
#     import glob
#     print(f"Loading Parquet files matching: {glob_pat}")
#     paths = sorted(glob.glob(glob_pat))
#     if not paths:
#         raise FileNotFoundError(f"No parquet at {glob_pat}")
#     parts = []
#     for p in paths:
#         print(f"  reading {p}")
#         df = pd.read_parquet(p)
#         df["time"] = pd.to_datetime(df["time"])
#         parts.append(df)
#     out = pd.concat(parts, ignore_index=True)
#     split = out["BIOME_ID_REALM"].str.split("_", n=1, expand=True)
#     out["biome_id"] = split[0].astype(int)
#     out["realm"] = split[1].astype(str)
#     out["year"] = out["time"].dt.year
#     out = out[(out["year"] >= YEAR_MIN) & (out["year"] <= YEAR_MAX)]
#     print(f"Loaded {len(out):,} rows, years {out['year'].min()} to {out['year'].max()}, scenarios: {sorted(out['scenario'].unique())}")
#     return out

# # ── Aggregations ──────────────────────────────────────────────────────────────
# def annual_by_realm(df: pd.DataFrame) -> pd.DataFrame:
#     print("Aggregating to annual realm totals")
#     # first annual mean at BIOME_ID_REALM level, then sum across biomes per realm
#     tmp = (
#         df.groupby(["scenario", "realm", "BIOME_ID_REALM", "year"], observed=True)["area_km2"]
#           .mean()
#           .reset_index()
#     )
#     ann = (
#         tmp.groupby(["scenario", "realm", "year"], observed=True)["area_km2"]
#            .sum()
#            .reset_index()
#     )
#     print(f"Annual realm table has {len(ann):,} rows")
#     return ann

# def annual_by_combo(df: pd.DataFrame) -> pd.DataFrame:
#     print("Aggregating to annual biome x realm means")
#     ann = (
#         df.groupby(
#             ["scenario", "BIOME_ID_REALM", "biome_id", "realm", "year"], observed=True
#         )["area_km2"]
#         .mean()
#         .reset_index()
#     )
#     print(f"Annual biome x realm table has {len(ann):,} rows")
#     return ann

# def anomaly(ann: pd.DataFrame, baseline=(1995, 2014)) -> pd.DataFrame:
#     y0, y1 = baseline
#     print(f"Converting to anomalies relative to {y0}-{y1}")
#     base = (
#         ann[(ann["year"] >= y0) & (ann["year"] <= y1)]
#         .groupby(["scenario", "realm"], observed=True)["area_km2"]
#         .mean()
#         .rename("base")
#     )
#     out = ann.merge(base, on=["scenario", "realm"], how="left")
#     out["area_km2"] = out["area_km2"] - out["base"]
#     return out.drop(columns="base")

# # ── Colormap helper ───────────────────────────────────────────────────────────
# def _get_cmap(name="RdBu"):
#     try:
#         return mpl.colormaps[name]          # mpl >= 3.6
#     except Exception:
#         try:
#             return plt.get_cmap(name)       # fallback
#         except Exception:
#             return getattr(mpl.cm, name)    # last resort

# # ── Trend helpers ─────────────────────────────────────────────────────────────
# def _sen_slope(y: np.ndarray, x: np.ndarray) -> float:
#     y = np.asarray(y); x = np.asarray(x)
#     m = np.isfinite(y) & np.isfinite(x)
#     if m.sum() < 2:
#         return np.nan
#     return float(theilslopes(y[m], x[m])[0])

# def compute_realm_trends_for_ts(ann_realm: pd.DataFrame) -> pd.DataFrame:
#     print("Computing Theil Sen slopes per realm and scenario for time series")
#     data = anomaly(ann_realm) if NORMALIZE else ann_realm.copy()
#     has_hist = "historical" in data["scenario"].unique()
#     hist_end = int(data.loc[data["scenario"] == "historical", "year"].max()) if has_hist else None
#     fut_scens = [s for s in ["ssp126", "ssp370", "ssp585"] if s in data["scenario"].unique()]

#     recs = []
#     for realm, dr in data.groupby("realm", observed=True):
#         if has_hist:
#             dh = dr[dr["scenario"] == "historical"].sort_values("year")

#             if not dh.empty:
#                 slope = _sen_slope(dh["area_km2"].to_numpy(), dh["year"].to_numpy())
#                 y0, y1 = int(dh["year"].min()), int(dh["year"].max())
#                 recs.append((realm, "historical", y0, y1, slope))
#         for s in fut_scens:
#             ds = dr[dr["scenario"] == s].copy()
#             if has_hist:
#                 ds = ds[ds["year"] > hist_end]
#             ds = ds.sort_values("year")
#             if not ds.empty:
#                 slope = _sen_slope(ds["area_km2"].to_numpy(), ds["year"].to_numpy())
#                 y0, y1 = int(ds["year"].min()), int(ds["year"].max())
#                 recs.append((realm, s, y0, y1, slope))
#     out = pd.DataFrame(recs, columns=["realm", "scenario", "year_start", "year_end", "slope_km2_per_year"])
#     print(f"Computed {len(out):,} trend rows")
#     return out

# def save_realm_trends_txt(trends_df: pd.DataFrame, out_txt: str):
#     print(f"Writing trend text to {out_txt}")
#     lines = []
#     hdr = f"Realm trends (Theil Sen slopes)  units: km^2 per year  normalized={int(NORMALIZE)} baseline={BASELINE if NORMALIZE else 'NA'}"
#     lines.append(hdr)
#     lines.append("=" * len(hdr))
#     for realm in sorted(trends_df["realm"].unique()):
#         lines.append(f"\n{REALM_NAME.get(realm, realm)} ({realm})")
#         sub = trends_df[trends_df["realm"] == realm].sort_values(["scenario", "year_start"])
#         for _, row in sub.iterrows():
#             scen = row["scenario"]
#             y0, y1 = int(row["year_start"]), int(row["year_end"])
#             slope = row["slope_km2_per_year"]
#             slope_txt = f"{slope:+,.{TREND_DECIMALS}f}" if np.isfinite(slope) else "NA"
#             lines.append(f"  {scen:11s}  slope {slope_txt} km^2/yr   period {y0} to {y1}  n={y1-y0+1}")
#     Path(out_txt).write_text("\n".join(lines), encoding="utf-8")

# # ── Time series: with percentile shading ──────────────────────────────────────
# def plot_ts_all_realms(ann_realm: pd.DataFrame, out_png: str):
#     print(f"Plotting time series with shading -> {out_png}")
#     data = anomaly(ann_realm) if NORMALIZE else ann_realm.copy()
#     has_hist = "historical" in data["scenario"].unique()
#     hist_end = int(data.loc[data["scenario"] == "historical", "year"].max()) if has_hist else None
#     fut_scens = [s for s in ["ssp126", "ssp370", "ssp585"] if s in data["scenario"].unique()]

#     realms = sorted(data["realm"].unique())
#     n = len(realms)
#     ncols = 4 if n >= 8 else 3 if n >= 6 else 2
#     nrows = int(np.ceil(n / ncols))

#     fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), dpi=DPI, sharex=True)
#     fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.90, wspace=0.10, hspace=0.20)
#     axes = np.array(axes).reshape(-1)

#     for i, r in enumerate(realms):
#         ax = axes[i]
#         ax.tick_params(axis="y", pad=6)
#         dr = data[data["realm"] == r].copy().sort_values(["scenario", "year"])

#         yvals = []
#         if has_hist:
#             yvals.append(dr.loc[dr["scenario"] == "historical", "area_km2"].to_numpy())
#         for s in fut_scens:
#             tmp = dr.loc[dr["scenario"] == s, ["year", "area_km2"]]
#             if has_hist:
#                 tmp = tmp[tmp["year"] > hist_end]
#             yvals.append(tmp["area_km2"].to_numpy())
#         yvals = np.concatenate([v for v in yvals if v.size > 0]) if yvals else np.array([0, 1])
#         ypad = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) if np.isfinite(yvals).any() else 1.0)
#         ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)

#         if has_hist:
#             dh = dr[dr["scenario"] == "historical"]
#             ax.plot(dh["year"], dh["area_km2"], color="black", linewidth=1.8, label="historical")

#         for s in fut_scens:
#             ds = dr[dr["scenario"] == s].copy()
#             if has_hist:
#                 ds = ds[ds["year"] > hist_end]
#             if not ds.empty:
#                 ax.plot(ds["year"], ds["area_km2"], color=SCEN_COLORS.get(s, "0.5"), linewidth=1.5, label=s)

#         if fut_scens:
#             dfut = dr[dr["scenario"].isin(fut_scens)].copy()
#             if has_hist:
#                 dfut = dfut[dfut["year"] > hist_end]
#             if not dfut.empty:
#                 piv = dfut.pivot_table(index="year", columns="scenario", values="area_km2", aggfunc="mean").sort_index()
#                 years = piv.index.values
#                 if piv.shape[1] >= 1:
#                     qlo, qhi = np.nanpercentile(piv.values, [5, 95], axis=1)
#                     ax.fill_between(years, qlo, qhi, alpha=0.25, linewidth=0)

#         if has_hist and np.isfinite(hist_end):
#             ax.axvline(hist_end, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=5)

#         ax.set_title(REALM_NAME.get(r, r), fontsize=9)
#         ax.grid(True, alpha=0.3)
#         if i // ncols == nrows - 1:
#             ax.set_xlabel("Year")
#         if i % ncols == 0:
#             ax.set_ylabel("Area anomaly km²" if NORMALIZE else "Area km²")

#     for j in range(i + 1, len(axes)):
#         axes[j].axis("off")

#     lines, labels = [], []
#     if has_hist:
#         lines.append(mpl.lines.Line2D([], [], color="black", lw=1.8)); labels.append("historical")
#     for s in fut_scens:
#         lines.append(mpl.lines.Line2D([], [], color=SCEN_COLORS.get(s, "0.5"), lw=1.5)); labels.append(s)
#     band = mpl.patches.Patch(facecolor="0.6", alpha=0.25, label="5–95 pct (futures)")
#     lines.append(band); labels.append("5–95 pct (futures)")
#     fig.legend(lines, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.995))

#     fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
#     plt.close(fig)

# # ── Time series: lines only ───────────────────────────────────────────────────
# def plot_ts_all_realms_lines_only(ann_realm: pd.DataFrame, out_png: str):
#     print(f"Plotting time series lines only -> {out_png}")
#     data = anomaly(ann_realm) if NORMALIZE else ann_realm.copy()
#     has_hist = "historical" in data["scenario"].unique()
#     hist_end = int(data.loc[data["scenario"] == "historical", "year"].max()) if has_hist else None
#     fut_scens = [s for s in ["ssp126", "ssp370", "ssp585"] if s in data["scenario"].unique()]

#     realms = sorted(data["realm"].unique())
#     n = len(realms)
#     ncols = 4 if n >= 8 else 3 if n >= 6 else 2
#     nrows = int(np.ceil(n / ncols))

#     fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), dpi=DPI, sharex=True)
#     fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.90, wspace=0.10, hspace=0.20)
#     axes = np.array(axes).reshape(-1)

#     for i, r in enumerate(realms):
#         ax = axes[i]
#         ax.tick_params(axis="y", pad=6)
#         dr = data[data["realm"] == r].copy().sort_values(["scenario", "year"])

#         yvals = []
#         if has_hist:
#             yvals.append(dr.loc[dr["scenario"] == "historical", "area_km2"].to_numpy())
#         for s in fut_scens:
#             tmp = dr.loc[dr["scenario"] == s, ["year", "area_km2"]]
#             if has_hist:
#                 tmp = tmp[tmp["year"] > hist_end]
#             yvals.append(tmp["area_km2"].to_numpy())
#         yvals = np.concatenate([v for v in yvals if v.size > 0]) if yvals else np.array([0, 1])
#         ypad = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) if np.isfinite(yvals).any() else 1.0)
#         ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)

#         if has_hist:
#             dh = dr[dr["scenario"] == "historical"]
#             ax.plot(dh["year"], dh["area_km2"], color="black", linewidth=1.8, label="historical")

#         for s in fut_scens:
#             ds = dr[dr["scenario"] == s].copy()
#             if has_hist:
#                 ds = ds[ds["year"] > hist_end]
#             if not ds.empty:
#                 ax.plot(ds["year"], ds["area_km2"], color=SCEN_COLORS.get(s, "0.5"), linewidth=1.5, label=s)

#         if has_hist and np.isfinite(hist_end):
#             ax.axvline(hist_end, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=5)

#         ax.set_title(REALM_NAME.get(r, r), fontsize=9)
#         ax.grid(True, alpha=0.3)
#         if i // ncols == nrows - 1:
#             ax.set_xlabel("Year")
#         if i % ncols == 0:
#             ax.set_ylabel("Area anomaly km²" if NORMALIZE else "Area km²")

#     for j in range(i + 1, len(axes)):
#         axes[j].axis("off")

#     lines, labels = [], []
#     if has_hist:
#         lines.append(mpl.lines.Line2D([], [], color="black", lw=1.8)); labels.append("historical")
#     for s in fut_scens:
#         lines.append(mpl.lines.Line2D([], [], color=SCEN_COLORS.get(s, "0.5"), lw=1.5)); labels.append(s)
#     fig.legend(lines, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.995))

#     fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
#     plt.close(fig)

# # ── Spatial percent change per SSP ────────────────────────────────────────────
# def pct_change_hist_plus_ssp(ann_combo: pd.DataFrame) -> pd.DataFrame:
#     print("Computing spatial percent change per biome x realm, historical concatenated with each SSP")
#     scens = sorted(ann_combo["scenario"].unique())
#     has_hist = "historical" in scens
#     fut_scens = [s for s in scens if s.startswith("ssp")]
#     recs = []

#     hist_by_combo = {}
#     if has_hist:
#         for combo, sub in ann_combo[ann_combo["scenario"] == "historical"].groupby(
#             "BIOME_ID_REALM", observed=True
#         ):
#             hist_by_combo[combo] = sub.sort_values("year")[["year", "area_km2"]].to_numpy()

#     for s in fut_scens:
#         for combo, sub in ann_combo[ann_combo["scenario"] == s].groupby(
#             "BIOME_ID_REALM", observed=True
#         ):
#             arr_ssp = sub.sort_values("year")[["year", "area_km2"]].to_numpy()
#             if combo in hist_by_combo:
#                 arr = np.vstack([hist_by_combo[combo], arr_ssp])
#                 yrs, vals = arr[:, 0], arr[:, 1]
#                 order = np.argsort(yrs)
#                 yrs, vals = yrs[order], vals[order]
#                 _, idx_last = np.unique(yrs[::-1], return_index=True)
#                 idx = len(yrs) - 1 - np.sort(idx_last)
#                 yrs, vals = yrs[idx], vals[idx]
#             else:
#                 yrs, vals = arr_ssp[:, 0], arr_ssp[:, 1]

#             if len(yrs) < 2 or not np.isfinite(vals).any():
#                 pct = np.nan
#             else:
#                 slope, intercept, _, _ = theilslopes(vals, yrs)
#                 y0 = intercept + slope * yrs.min()
#                 yN = intercept + slope * yrs.max()
#                 pct = np.nan if y0 == 0 else (yN - y0) / y0 * 100.0

#             biome_id = int(sub["biome_id"].iloc[0])
#             realm = str(sub["realm"].iloc[0])
#             recs.append((s, combo, biome_id, realm, pct))

#     out = pd.DataFrame(recs, columns=["scenario", "BIOME_ID_REALM", "biome_id", "realm", "pct_change"])
#     print(f"Computed percent change for {len(out):,} biome x realm x scenario rows")
#     return out

# def merge_pct_to_shapes(pct_df: pd.DataFrame, shp_path: str, scen: str) -> gpd.GeoDataFrame:
#     gdf = gpd.read_file(shp_path)[["BIOME", "REALM", "geometry"]]
#     gdf["BIOME_ID_REALM"] = gdf["BIOME"].astype(int).astype(str) + "_" + gdf["REALM"].astype(str)
#     sub = pct_df[pct_df["scenario"] == scen][["BIOME_ID_REALM", "pct_change"]]
#     gdf = gdf.merge(sub, on="BIOME_ID_REALM", how="left")
#     gdf = gdf.set_crs(4326, allow_override=True)
#     return gdf

# def _add_glossy_overlay(ax):
#     n = 256
#     x = np.linspace(0, 1, n)
#     y = np.linspace(0, 1, n)
#     X, Y = np.meshgrid(x, y)
#     alpha = 0.30 * np.exp(-((X - 0.25) / 0.18) ** 2) + 0.08 * (1 - Y)
#     alpha = np.clip(alpha, 0, 0.35)
#     overlay = np.ones((n, n, 4), dtype=float)
#     overlay[..., 3] = alpha
#     ax.imshow(overlay, extent=(0, 1, 0, 1), transform=ax.transAxes, interpolation="bilinear", zorder=10)

# def plot_spatial_pct_grid(pct_df: pd.DataFrame, shp_path: str, out_png: str):
#     print(f"Plotting spatial percent change grid -> {out_png}")
#     panels = [s for s in ["ssp126", "ssp370", "ssp585"] if s in pct_df["scenario"].unique()]
#     if not panels:
#         print("No SSP panels to plot")
#         return

#     vals = (
#         np.concatenate([pct_df[pct_df["scenario"] == p]["pct_change"].dropna().to_numpy() for p in panels])
#         if panels else np.array([0])
#     )
#     if vals.size == 0:
#         vmin, vmax = -10, 10
#     else:
#         q_low, q_hi = np.nanpercentile(vals, [2, 98])
#         m = max(abs(q_low), abs(q_hi))
#         vmin, vmax = -m, m

#     n = len(panels)
#     ncols = 2 if n == 2 else 3
#     nrows = int(np.ceil(n / ncols))
#     fig = plt.figure(figsize=(3.6 * ncols, 2.7 * nrows), dpi=DPI)

#     fig.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.96, wspace=0.02, hspace=0.08)

#     cmap = _get_cmap("RdBu")
#     norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

#     for i, p in enumerate(panels):
#         ax = plt.subplot(nrows, ncols, i + 1, projection=ccrs.Robinson())
#         gdf = merge_pct_to_shapes(pct_df, shp_path, scen=p)

#         for geom, val in zip(gdf.geometry, gdf["pct_change"].to_numpy()):
#             if geom is None or not np.isfinite(val):
#                 continue
#             ax.add_geometries(
#                 [geom],
#                 crs=ccrs.PlateCarree(),
#                 facecolor=cmap(norm(val)),
#                 edgecolor="none",
#                 linewidth=0,
#             )
#         ax.coastlines(linewidth=0.4, color="black")
#         ax.set_global()
#         ax.set_title(p, fontsize=9)
#         ax.axis("off")

#     cax = fig.add_axes([0.88, 0.15, 0.022, 0.65])  # slimmer colorbar
#     sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
#     cbar.set_label("% change")
#     cbar.outline.set_visible(False)
#     cbar.ax.tick_params(length=0)
#     for spine in cbar.ax.spines.values():
#         spine.set_visible(False)
#     _add_glossy_overlay(cbar.ax)

#     fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
#     plt.close(fig)

# # ── Animation ─────────────────────────────────────────────────────────────────
# def animate_ts_all_realms(ann_realm: pd.DataFrame, out_base: str, fps: int = ANIM_FPS):
#     print("Building time series animation")
#     data = anomaly(ann_realm) if NORMALIZE else ann_realm.copy()
#     has_hist = "historical" in data["scenario"].unique()
#     hist_end = int(data.loc[data["scenario"] == "historical", "year"].max()) if has_hist else None
#     fut_scens = [s for s in ["ssp126", "ssp370", "ssp585"] if s in data["scenario"].unique()]
#     realms = sorted(data["realm"].unique())

#     # gather series
#     series = {}
#     all_years = set()
#     for r in realms:
#         series[r] = {}
#         dr = data[data["realm"] == r]
#         if has_hist:
#             dh = dr[dr]["historical" == dr["scenario"]].sort_values("year")


#             if not dh.empty:
#                 series[r]["historical"] = (dh["year"].to_numpy(), dh["area_km2"].to_numpy())
#                 all_years.update(dh["year"].tolist())
#         for scen in fut_scens:
#             ds = dr[dr["scenario"] == scen].sort_values("year")
#             if has_hist:
#                 ds = ds[ds["year"] > hist_end]
#             if not ds.empty:
#                 series[r][scen] = (ds["year"].to_numpy(), ds["area_km2"].to_numpy())
#                 all_years.update(ds["year"].tolist())

#     years = np.array(sorted(y for y in all_years if YEAR_MIN <= y <= YEAR_MAX))
#     if years.size == 0:
#         print("No years available for animation, skipping")
#         return

#     n = len(realms)
#     ncols = 4 if n >= 8 else 3 if n >= 6 else 2
#     nrows = int(np.ceil(n / ncols))
#     fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), dpi=DPI, sharex=True)

#     # extra margins: bottom for legend, left for y tick labels
#     fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.90, wspace=0.10, hspace=0.22)
#     axes = np.array(axes).reshape(-1)

#     # per realm y-limits
#     ylims = {}
#     for r in realms:
#         vals = []
#         for _, (_, yy) in series[r].items():
#             vals.append(yy)
#         if vals:
#             v = np.concatenate(vals)
#             vmin, vmax = np.nanmin(v), np.nanmax(v)
#             pad = 0.05 * (vmax - vmin if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0)
#             ylims[r] = (vmin - pad, vmax + pad)
#         else:
#             ylims[r] = (0, 1)

#     # prepare legend handles once
#     leg_lines, leg_labels = [], []
#     if has_hist:
#         leg_lines.append(mpl.lines.Line2D([], [], color="black", lw=1.8)); leg_labels.append("historical")
#     for s in fut_scens:
#         leg_lines.append(mpl.lines.Line2D([], [], color=SCEN_COLORS.get(s, "0.5"), lw=1.5)); leg_labels.append(s)
#     # place legend at bottom outside the plot area
#     fig.legend(
#         leg_lines, leg_labels,
#         loc="lower center", bbox_to_anchor=(0.5, 0.04),
#         ncol=len(leg_labels), frameon=False, borderaxespad=0.0
#     )

#     suptitle = fig.suptitle("", y=0.965, fontsize=12, fontweight="bold")

#     # create axes content
#     artists = []
#     for i, r in enumerate(realms):
#         ax = axes[i]
#         ax.tick_params(axis="y", pad=10)
#         # scientific notation on y axis
#         fmt = ScalarFormatter(useMathText=True)
#         fmt.set_powerlimits((0, 0))  # always scientific
#         ax.yaxis.set_major_formatter(fmt)

#         ax.set_title(REALM_NAME.get(r, r), fontsize=9)
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(*ylims[r])
#         if i // ncols == nrows - 1:
#             ax.set_xlabel("Year")
#         if i % ncols == 0:
#             ax.set_ylabel("Area km$^2$")

#         if has_hist and hist_end is not None:
#             ax.axvline(hist_end, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=5)

#         L = {}
#         if "historical" in series[r]:
#             (line_hist,) = ax.plot([], [], color="black", lw=1.8, label="historical")
#             L["historical"] = line_hist
#         for s in fut_scens:
#             if s in series[r]:
#                 (line_s,) = ax.plot([], [], color=SCEN_COLORS.get(s, "0.5"), lw=1.5, label=s)
#                 L[s] = line_s
#         artists.append((ax, L))

#     for j in range(i + 1, len(axes)):
#         axes[j].axis("off")

#     def init():
#         for _, L in artists:
#             for line in L.values():
#                 line.set_data([], [])
#         suptitle.set_text("")
#         return [suptitle] + [ln for _, L in artists for ln in L.values()]

#     def update(frame_idx):
#         yr = years[frame_idx]
#         suptitle.set_text(f"Year {yr}")
#         for (ax, L), r in zip(artists, realms):
#             if "historical" in series[r]:
#                 xh, yh = series[r]["historical"]
#                 m = xh <= yr
#                 L["historical"].set_data(xh[m], yh[m])
#             for s in fut_scens:
#                 if s in series[r]:
#                     xs, ys = series[r][s]
#                     m = xs <= yr
#                     L[s].set_data(xs[m], ys[m])
#             ax.set_xlim(YEAR_MIN, YEAR_MAX)
#         return [suptitle] + [ln for _, L in artists for ln in L.values()]

#     ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(years), blit=False)

#     if shutil.which("ffmpeg"):
#         out_mp4 = f"{out_base}.mp4"
#         print(f"ffmpeg found, writing MP4 to {out_mp4}")
#         writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
#         ani.save(out_mp4, writer=writer, dpi=DPI)
#     else:
#         out_gif = f"{out_base}.gif"
#         print(f"ffmpeg not found, writing GIF to {out_gif}")
#         from matplotlib.animation import PillowWriter
#         writer = PillowWriter(fps=min(fps, 12))
#         ani.save(out_gif, writer=writer, dpi=DPI)

#     plt.close(fig)
#     print("Animation complete")

# # ── Main ──────────────────────────────────────────────────────────────────────
# def main():
#     df = read_parquets(PARQUET_GLOB)

#     ann_realm = annual_by_realm(df)
#     trends = compute_realm_trends_for_ts(ann_realm)
#     trends_txt = f"{OUT_DIR}/timeseries_realm_trends.txt"
#     save_realm_trends_txt(trends, out_txt=trends_txt)

#     ts_with_shading = f"{OUT_DIR}/timeseries_all_realms.png"
#     plot_ts_all_realms(ann_realm, out_png=ts_with_shading)
#     print(f"Wrote {ts_with_shading}")

#     ts_lines_only = f"{OUT_DIR}/timeseries_all_realms_lines_only.png"
#     plot_ts_all_realms_lines_only(ann_realm, out_png=ts_lines_only)
#     print(f"Wrote {ts_lines_only}")

#     ann_combo = annual_by_combo(df)
#     pct_df = pct_change_hist_plus_ssp(ann_combo)
#     spatial_png = f"{OUT_DIR}/spatial_pct_change_grid.png"
#     plot_spatial_pct_grid(pct_df, BIOME_SHP, out_png=spatial_png)
#     print(f"Wrote {spatial_png}")

#     anim_base = f"{OUT_DIR}/timeseries_all_realms_anim"
#     animate_ts_all_realms(ann_realm, out_base=anim_base, fps=ANIM_FPS)

#     print("Done")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# Standalone animation script for GDE area time series per realm

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import animation
import shutil

# ── Config ───────────────────────────────────────────────────────────────
PARQUET_GLOB = os.environ.get(
    "PARQUET_GLOB",
    "/projects/prjs1578/futurewetgde/wetGDEs_area/gde_area_by_biome_realm_monthly_*.parquet",
)
OUT_DIR = os.environ.get("PLOT_OUT", "/projects/prjs1578/futurewetgde/figs_gde_area_anim")
YEAR_MIN = int(os.environ.get("YEAR_MIN", "1980"))
YEAR_MAX = int(os.environ.get("YEAR_MAX", "2100"))
ANIM_FPS = int(os.environ.get("ANIM_FPS", "8"))
DPI = int(os.environ.get("DPI", "300"))

SCEN_COLORS = {"ssp126": "#1a9850", "ssp370": "#fdae61", "ssp585": "#d73027"}
REALM_NAME = {
    "AA": "Australasian",
    "AT": "Afrotropical",
    "IM": "Indo Malayan",
    "NA": "Nearctic",
    "NT": "Neotropical",
    "OC": "Oceanian",
    "PA": "Palearctic",
    "AN": "Antarctic",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── IO ────────────────────────────────────────────────────────────────────
def read_parquets(glob_pat: str) -> pd.DataFrame:
    import glob
    print(f"Loading Parquet files matching: {glob_pat}")
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No parquet at {glob_pat}")
    parts = []
    for p in paths:
        print(f"  reading {p}")
        df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    split = out["BIOME_ID_REALM"].str.split("_", n=1, expand=True)
    out["biome_id"] = split[0].astype(int)
    out["realm"] = split[1].astype(str)
    out["year"] = out["time"].dt.year
    out = out[(out["year"] >= YEAR_MIN) & (out["year"] <= YEAR_MAX)]
    print(f"Loaded {len(out):,} rows, years {out['year'].min()}–{out['year'].max()}, scenarios: {sorted(out['scenario'].unique())}")
    return out

def annual_by_realm(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating to annual realm totals")
    tmp = (
        df.groupby(["scenario", "realm", "BIOME_ID_REALM", "year"], observed=True)["area_km2"]
          .mean()
          .reset_index()
    )
    ann = (
        tmp.groupby(["scenario", "realm", "year"], observed=True)["area_km2"]
           .sum()
           .reset_index()
    )
    return ann

# ── Animation ─────────────────────────────────────────────────────────────
def animate_ts_all_realms(ann_realm: pd.DataFrame, out_base: str, fps: int = ANIM_FPS):
    print("Building time series animation")
    has_hist = "historical" in ann_realm["scenario"].unique()
    hist_end = int(ann_realm.loc[ann_realm["scenario"] == "historical", "year"].max()) if has_hist else None
    fut_scens = [s for s in ["ssp126", "ssp370", "ssp585"] if s in ann_realm["scenario"].unique()]
    realms = sorted(ann_realm["realm"].unique())

    series = {}
    all_years = set()
    for r in realms:
        series[r] = {}
        dr = ann_realm[ann_realm["realm"] == r]
        if has_hist:
            dh = dr[dr["scenario"] == "historical"].sort_values("year")
            if not dh.empty:
                series[r]["historical"] = (dh["year"].to_numpy(), dh["area_km2"].to_numpy())
                all_years.update(dh["year"].tolist())
        for scen in fut_scens:
            ds = dr[dr["scenario"] == scen].sort_values("year")
            if has_hist:
                ds = ds[ds["year"] > hist_end]
            if not ds.empty:
                series[r][scen] = (ds["year"].to_numpy(), ds["area_km2"].to_numpy())
                all_years.update(ds["year"].tolist())

    years = np.array(sorted(y for y in all_years if YEAR_MIN <= y <= YEAR_MAX))
    if years.size == 0:
        print("No years available for animation, skipping")
        return

    n = len(realms)
    ncols = 4 if n >= 8 else 3 if n >= 6 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 2.6 * nrows),  # wider panels
        dpi=DPI,
        sharex=True
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, bottom=0.22, top=0.92,
        wspace=0.25, hspace=0.35  # more side-by-side and vertical space
    )
    axes = np.array(axes).reshape(-1)

    # y limits per realm
    ylims = {}
    for r in realms:
        vals = []
        for _, (_, yy) in series[r].items():
            vals.append(yy)
        if vals:
            v = np.concatenate(vals)
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
            ylims[r] = (vmin - pad, vmax + pad)
        else:
            ylims[r] = (0, 1)

    # legend
    leg_lines, leg_labels = [], []
    if has_hist:
        leg_lines.append(mpl.lines.Line2D([], [], color="black", lw=1.8)); leg_labels.append("historical")
    for s in fut_scens:
        leg_lines.append(mpl.lines.Line2D([], [], color=SCEN_COLORS.get(s, "0.5"), lw=1.5)); leg_labels.append(s)
    fig.legend(
        leg_lines, leg_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.04),
        ncol=len(leg_labels), frameon=False
    )

    suptitle = fig.suptitle("", y=0.97, fontsize=13, fontweight="bold")

    artists = []
    for i, r in enumerate(realms):
        ax = axes[i]
        ax.tick_params(axis="y", pad=6)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)

        ax.set_title(REALM_NAME.get(r, r), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*ylims[r])
        if i // ncols == nrows - 1:
            ax.set_xlabel("Year")
        if i % ncols == 0:
            ax.set_ylabel("Area km$^2$")

        if has_hist and hist_end is not None:
            ax.axvline(hist_end, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=5)

        L = {}
        if "historical" in series[r]:
            (line_hist,) = ax.plot([], [], color="black", lw=1.8)
            L["historical"] = line_hist
        for s in fut_scens:
            if s in series[r]:
                (line_s,) = ax.plot([], [], color=SCEN_COLORS.get(s, "0.5"), lw=1.5)
                L[s] = line_s
        artists.append((ax, L))

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    def init():
        for _, L in artists:
            for line in L.values():
                line.set_data([], [])
        suptitle.set_text("")
        return [suptitle] + [ln for _, L in artists for ln in L.values()]

    def update(frame_idx):
        yr = years[frame_idx]
        suptitle.set_text(f"Year {yr}")
        for (ax, L), r in zip(artists, realms):
            if "historical" in series[r]:
                xh, yh = series[r]["historical"]
                m = xh <= yr
                L["historical"].set_data(xh[m], yh[m])
            for s in fut_scens:
                if s in series[r]:
                    xs, ys = series[r][s]
                    m = xs <= yr
                    L[s].set_data(xs[m], ys[m])
            ax.set_xlim(YEAR_MIN, YEAR_MAX)
        return [suptitle] + [ln for _, L in artists for ln in L.values()]

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(years), blit=False)

    if shutil.which("ffmpeg"):
        out_mp4 = f"{out_base}.mp4"
        print(f"ffmpeg found, writing MP4 to {out_mp4}")
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        ani.save(out_mp4, writer=writer, dpi=DPI)
    else:
        out_gif = f"{out_base}.gif"
        print(f"ffmpeg not found, writing GIF to {out_gif}")
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=min(fps, 12))
        ani.save(out_gif, writer=writer, dpi=DPI)

    plt.close(fig)
    print("Animation complete")

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    df = read_parquets(PARQUET_GLOB)
    ann_realm = annual_by_realm(df)
    anim_base = f"{OUT_DIR}/timeseries_all_realms_anim"
    animate_ts_all_realms(ann_realm, out_base=anim_base, fps=ANIM_FPS)

if __name__ == "__main__":
    main()
