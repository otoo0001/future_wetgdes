#!/usr/bin/env python3
# plot_all_future_gde.py
#
# Produces:
#  1) Realm time-series grids (Australasian+Oceanian merged; Antarctic dropped)
#       • Exclusion: none
#       • Exclusion: crops
#       • Exclusion: crops+pasture
#     Each with 5–95% futures envelope.
#  2) Realm lollipop trends (overlays, separate x-range per scenario):
#       • No exclusion (solid) vs Exclude crops (dashed, plotted slightly below)
#       • No exclusion (solid) vs Exclude crops+pasture (dashed, plotted slightly below)
#     Colors: negative=red, positive=blue; legend at bottom.
#  3) Spatial % change maps (late 2071–2100 vs baseline 1985–2014) on biome×realm:
#     3 rows (none, crops, crops+pasture) × 3 cols (ssp126, ssp370, ssp585),
#     Robinson projection, shared short vertical borderless colorbar.
#
# Inputs: Parquet from future_gdes_area.py
#   <PARQUET_DIR>/gde_area_by_biome_realm_monthly_<scenario>.parquet
#   or hive parts at:
#   <PARQUET_DIR>/gde_area_by_biome_realm_monthly_<scenario>/year=YYYY/part_*.parquet

import os, glob, math, warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs  # for Robinson projection

# ──────────────────────────────────────────────────────────────────────────────
# Config (env-overridable)
# ──────────────────────────────────────────────────────────────────────────────
PARQUET_DIR    = os.environ.get("PARQUET_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_area_test")
PARQUET_PREFIX = os.environ.get("PARQUET_PREFIX", "gde_area_by_biome_realm_monthly")
OUT_DIR        = os.environ.get("PLOT_DIR", os.path.join(PARQUET_DIR, "plots_biome_realm"))
WWF_SHAPE      = os.environ.get("WWF_SHAPE", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")

SCENARIOS    = ["historical", "ssp126", "ssp370", "ssp585"]
FUTURE_SCENS = ["ssp126", "ssp370", "ssp585"]
EXCLUSIONS   = ["none", "crops", "crops_pasture"]  # rows in spatial grid

# column names expected in Parquet
COLMAP = {
    "none":          "area_none_km2",
    "crops":         "area_crops_excl_km2",
    "crops_pasture": "area_crops_pasture_excl_km2",
}

# windows
BASELINE_YR = (1985, 2014)
LATE_YR     = (2071, 2100)

# TS smoothing (years)
SMOOTH_YEARS = int(os.environ.get("SMOOTH_YEARS", "5"))

# realm names & merging: AA + OC → AO; drop Antarctic (AN)
REALM_NAME_MAP = {
    "AA": "Australasian", "AT": "Afrotropical", "IM": "Indo Malayan",
    "NA": "Nearctic", "NT": "Neotropical", "OC": "Oceanian",
    "PA": "Palearctic", "AN": "Antarctic"
}
MERGE_TO_CODE   = {"AA", "OC"}  # → AO
MERGED_CODE     = "AO"
MERGED_NAME     = "Australasian+Oceanian"
DROP_CODES      = {"AN"}        # drop Antarctic

def realm_full_from_code(code: str) -> str:
    if code == MERGED_CODE:
        return MERGED_NAME
    return REALM_NAME_MAP.get(code, code)

# colors
SCEN_COLOR = {"historical":"#222222", "ssp126":"#1a9850", "ssp370":"#fdae61", "ssp585":"#d73027"}
COLOR_NEG  = "#d73027"  # negative (decline) = red
COLOR_POS  = "#1f78b4"  # positive (increase) = blue
CMAP_SPAT  = plt.get_cmap("RdBu")  # negative red, positive blue with TwoSlopeNorm(vcenter=0)

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────
def parquet_sources_for(scen: str) -> List[str]:
    """Return Parquet paths for a scenario: final file or hive parts."""
    final = os.path.join(PARQUET_DIR, f"{PARQUET_PREFIX}_{scen}.parquet")
    if os.path.isfile(final):
        return [final]
    hive_root = os.path.join(PARQUET_DIR, f"{PARQUET_PREFIX}_{scen}")
    return sorted(glob.glob(os.path.join(hive_root, "year=*/*.parquet")))

def load_monthly(excl: str) -> pd.DataFrame:
    """Load monthly biome×realm totals for one exclusion across scenarios → long DF."""
    col = COLMAP[excl]
    pieces = []
    for scen in SCENARIOS:
        srcs = parquet_sources_for(scen)
        if not srcs:
            print(f"[warn] missing Parquet for {scen}: {os.path.join(PARQUET_DIR, f'{PARQUET_PREFIX}_{scen}*')}")
            continue
        use_cols = ["time", "BIOME_ID_REALM", col]
        d = pd.concat((pd.read_parquet(p, columns=use_cols) for p in srcs), ignore_index=True)
        d = d.rename(columns={col: "area_km2"})
        d["scenario"] = scen
        d["time"] = pd.to_datetime(d["time"])
        pieces.append(d)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# Realm aggregation + TS smoothing
# ──────────────────────────────────────────────────────────────────────────────
def monthly_to_realm_annual(dfm: pd.DataFrame, merge_AA_OC=True, drop_antarctic=True) -> pd.DataFrame:
    """Sum monthly biome×realm to annual realm totals (AA+OC merged; AN dropped)."""
    if dfm.empty: return dfm
    df = dfm.copy()
    parts = df["BIOME_ID_REALM"].str.split("_", n=1, expand=True)
    parts.columns = ["biome_str", "realm_code"]
    if drop_antarctic:
        keep = parts["realm_code"] != "AN"
        df = df.loc[keep].copy()
        parts = parts.loc[keep]
    if merge_AA_OC:
        parts["realm_code"] = parts["realm_code"].where(~parts["realm_code"].isin(MERGE_TO_CODE), MERGED_CODE)
    df["realm"] = parts["realm_code"].apply(realm_full_from_code)
    df["year"]  = df["time"].dt.year.astype(int)
    return df.groupby(["scenario","realm","year"], as_index=False)["area_km2"].sum()

def smooth_by_year(df_ann: pd.DataFrame, window_years: int) -> pd.DataFrame:
    """Centered rolling mean (integer years) per scenario×realm."""
    if df_ann.empty or window_years <= 1:
        return df_ann
    out = []
    for (sc, r), d in df_ann.groupby(["scenario","realm"], sort=False):
        years = np.arange(d["year"].min(), d["year"].max() + 1, dtype=int)
        s = pd.Series(d.set_index("year")["area_km2"], index=years, dtype=float)
        y = s.rolling(window_years, center=True, min_periods=max(1, window_years//2)).mean()
        out.append(pd.DataFrame({"scenario": sc, "realm": r, "year": y.index.values, "area_km2": y.values}))
    return pd.concat(out, ignore_index=True)

def futures_envelope(df_ann: pd.DataFrame) -> pd.DataFrame:
    """5–95% envelope across future scenarios, per realm×year."""
    fut = df_ann[df_ann["scenario"].isin(FUTURE_SCENS)].copy()
    if fut.empty:
        return pd.DataFrame(columns=["realm","year","y5","y95"])
    pvt = fut.pivot_table(index=["realm","year"], columns="scenario", values="area_km2", aggfunc="first")
    q = pvt.quantile([0.05, 0.95], axis=1).T
    q.columns = ["y5","y95"]
    return q.reset_index()

# ──────────────────────────────────────────────────────────────────────────────
# Trends
# ──────────────────────────────────────────────────────────────────────────────
def slope_km2_per_year(years: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(years, float); y = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2: return np.nan
    m, _ = np.polyfit(x[ok], y[ok], 1)
    return float(m)

def trend_table(df_ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, r), d in df_ann.groupby(["scenario","realm"]):
        lo, hi = (BASELINE_YR if sc == "historical" else (2015, 2100))
        dd = d[(d["year"] >= lo) & (d["year"] <= hi)]
        rows.append((sc, r, slope_km2_per_year(dd["year"].to_numpy(), dd["area_km2"].to_numpy())))
    return pd.DataFrame(rows, columns=["scenario","realm","slope_km2_per_year"])

# ──────────────────────────────────────────────────────────────────────────────
# Spatial (biome×realm) helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_biome_realm_geoms() -> gpd.GeoDataFrame:
    """Dissolve WWF polygons to BIOME_ID_REALM; drop Antarctic."""
    shp = gpd.read_file(WWF_SHAPE)
    if "BIOME" not in shp.columns or "REALM" not in shp.columns:
        raise RuntimeError("Shapefile must include BIOME and REALM fields.")
    shp = shp.to_crs("EPSG:4326")
    shp = shp[shp["REALM"] != "AN"].copy()  # drop Antarctic
    shp["BIOME_ID_REALM"] = shp["BIOME"].astype(int).astype(str) + "_" + shp["REALM"].astype(str)
    return shp.dissolve(by="BIOME_ID_REALM", as_index=False, aggfunc="first")[["BIOME_ID_REALM","geometry"]]

def pct_change_by_biorealm(dfm: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """For one exclusion DF, compute % change per BIOME_ID_REALM for each future scenario."""
    out: Dict[str, pd.DataFrame] = {}
    if dfm.empty:
        return {sc: pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"]) for sc in FUTURE_SCENS}
    d = dfm.copy()
    d["year"] = d["time"].dt.year.astype(int)

    base = d[(d["scenario"]=="historical") & (d["year"].between(BASELINE_YR[0], BASELINE_YR[1]))]
    if base.empty:
        return {sc: pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"]) for sc in FUTURE_SCENS}
    b_mean = base.groupby("BIOME_ID_REALM", as_index=False)["area_km2"].mean().rename(columns={"area_km2":"V_baseline"})

    for sc in FUTURE_SCENS:
        fut = d[(d["scenario"]==sc) & (d["year"].between(LATE_YR[0], LATE_YR[1]))]
        if fut.empty:
            out[sc] = pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"])
            continue
        f_mean = fut.groupby("BIOME_ID_REALM", as_index=False)["area_km2"].mean().rename(columns={"area_km2":"V_future"})
        m = pd.merge(b_mean, f_mean, on="BIOME_ID_REALM", how="outer")
        m["pct_change"] = np.where(m["V_baseline"]>0, (m["V_future"]-m["V_baseline"])/m["V_baseline"]*100.0, np.nan)
        out[sc] = m[["BIOME_ID_REALM","pct_change"]]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_ts_realms(df_ann: pd.DataFrame, out_png: str, smooth_years: int = SMOOTH_YEARS, title: str | None = None):
    if df_ann.empty:
        print("[warn] TS: no data"); return
    df_s = smooth_by_year(df_ann, smooth_years)
    realms = sorted(df_s["realm"].unique())
    n = len(realms); ncols = 4; nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(4.0*ncols, 2.9*nrows + 1.0), dpi=220)

    env = futures_envelope(df_s)
    hist_end = df_s.loc[df_s["scenario"]=="historical","year"].max()

    for i, realm in enumerate(realms, 1):
        ax = plt.subplot(nrows, ncols, i)
        e = env[env["realm"] == realm]
        if not e.empty:
            ax.fill_between(e["year"], e["y5"], e["y95"], color="0.75", alpha=0.35, label="5–95% futures")
        for sc in SCENARIOS:
            d = df_s[(df_s["scenario"]==sc) & (df_s["realm"]==realm)]
            if d.empty: continue
            ax.plot(d["year"], d["area_km2"], lw=2, color=SCEN_COLOR[sc], label=sc)
        if pd.notna(hist_end):
            ax.axvline(hist_end, color="0.6", lw=1, ls="--")
        ax.set_title(realm, fontsize=10)
        ax.grid(True, alpha=0.2, lw=0.5)
        if i <= (nrows-1)*ncols:
            ax.set_xticklabels([])
        ax.set_ylabel("Area km$^2$" if i in (1, ncols+1) else "")
    if title:
        fig.suptitle(title, y=0.995, fontsize=12)

    handles = [Line2D([0],[0], color=SCEN_COLOR[s], lw=2) for s in SCENARIOS]
    labs    = ["historical","ssp126","ssp370","ssp585"]
    fig.legend(handles, labs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] TS -> {out_png}")

def plot_lollipop_overlay(df_ann_A: pd.DataFrame, df_ann_B: pd.DataFrame,
                          label_A: str, label_B: str, out_png: str):
    """Overlay realm lollipops for two exclusions (separate x-range per scenario; dashed series below)."""
    if df_ann_A.empty or df_ann_B.empty:
        print(f"[warn] lollipop skipped ({label_A} vs {label_B}): missing data"); return

    tt_A = trend_table(df_ann_A)
    tt_B = trend_table(df_ann_B)
    realms = sorted(set(tt_A["realm"]).union(tt_B["realm"]))
    y_main  = np.arange(len(realms))[::-1]
    y_below = y_main - 0.35  # offset for dashed series

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=220)
    axes = axes.ravel()

    for ax, sc in zip(axes, SCENARIOS):
        dA = tt_A[tt_A["scenario"] == sc].set_index("realm").reindex(realms)
        dB = tt_B[tt_B["scenario"] == sc].set_index("realm").reindex(realms)

        # symmetric per-scenario limits
        vals = np.concatenate([
            dA["slope_km2_per_year"].to_numpy(dtype=float),
            dB["slope_km2_per_year"].to_numpy(dtype=float)
        ])
        vals = vals[np.isfinite(vals)]
        xmax = float(np.abs(vals).max())*1.08 if vals.size else 1.0
        ax.set_xlim(-xmax, xmax)

        # A: solid stems + filled circles
        for yi, xi in zip(y_main, dA["slope_km2_per_year"].to_numpy()):
            if not np.isfinite(xi): continue
            color = COLOR_POS if xi > 0 else COLOR_NEG
            ax.plot([0, xi], [yi, yi], color=color, lw=2, ls="-", zorder=3)
            ax.plot([xi], [yi], marker="o", color=color, ms=5, zorder=4)

        # B: dashed stems + open squares, below
        for yi, xi in zip(y_below, dB["slope_km2_per_year"].to_numpy()):
            if not np.isfinite(xi): continue
            color = COLOR_POS if xi > 0 else COLOR_NEG
            ax.plot([0, xi], [yi, yi], color=color, lw=2, ls="--", zorder=2)
            ax.plot([xi], [yi], marker="s", mfc="none", mec=color, ms=5, zorder=3)

        ax.axvline(0, color="0.75", lw=1)
        ax.set_yticks(y_main)
        ax.set_yticklabels(realms, fontsize=8)
        ax.set_title(sc, fontsize=11, pad=6)
        ax.grid(True, axis="x", alpha=0.25, lw=0.6)
        ax.set_xlabel("km$^2$ per year")
        ax.set_ylim(y_below.min()-0.6, y_main.max()+0.6)

    legend_lines = [
        Line2D([0],[0], color="k", lw=2, ls="-"),
        Line2D([0],[0], color="k", lw=2, ls="--"),
        Line2D([0],[0], marker="o", color="k", lw=0),
        Line2D([0],[0], marker="s", color="k", lw=0, mfc="none"),
    ]
    legend_labels = [label_A, label_B, f"{label_A} marker", f"{label_B} marker"]
    fig.legend(legend_lines, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0.05, 0.08, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] lollipop -> {out_png}")

# ──────────────────────────────────────────────────────────────────────────────
# Spatial % change (biome×realm) — EXACT aesthetics requested
# ──────────────────────────────────────────────────────────────────────────────
def plot_spatial_pct_biorealm(dfm_by_excl: Dict[str, pd.DataFrame], out_png: str, vlim=50.0):
    geom = load_biome_realm_geoms()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)

    fig = plt.figure(figsize=(12, 10), dpi=260)
    proj = ccrs.Robinson()

    for r, excl in enumerate(EXCLUSIONS, start=1):
        dfm = dfm_by_excl.get(excl, pd.DataFrame())
        pct_by = pct_change_by_biorealm(dfm) if not dfm.empty else {sc: pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"]) for sc in FUTURE_SCENS}
        for c, scen in enumerate(FUTURE_SCENS, start=1):
            ax = plt.subplot(3, 3, (r-1)*3 + c, projection=proj)
            ax.set_global()
            ax.set_title(f"{excl} • {scen}", fontsize=10, pad=2)
            ax.coastlines(linewidth=0.1, color="0.4")

            dd = pct_by.get(scen, pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"]))
            g = geom.merge(dd, on="BIOME_ID_REALM", how="left")

            for _, row in g.iterrows():
                val = row.get("pct_change", np.nan)
                fc = CMAP_SPAT(norm(val)) if np.isfinite(val) else (0,0,0,0)
                ax.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                                  facecolor=fc, edgecolor="none", linewidth=0.0, zorder=1)

            ax.add_geometries(g["geometry"], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="0.25", linewidth=0, zorder=2)

            ax.set_axis_off()
            try:
                ax.outline_patch.set_visible(False); ax.background_patch.set_visible(False)
            except Exception:
                pass

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_SPAT); sm.set_array([])
    cax = fig.add_axes([0.92, 0.35, 0.02, 0.30])  # short vertical bar
    cb = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("% change")
    cb.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.02, 0.90, 0.98])
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[ok] spatial 3×3 -> {out_png}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[setup] PARQUET_DIR={PARQUET_DIR}")
    print(f"[setup] OUT_DIR={OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load monthly per exclusion (keep BIOME_ID_REALM for spatial)
    dfm_by_excl = {excl: load_monthly(excl) for excl in EXCLUSIONS}

    # === Time-series (one grid per exclusion) ===
    ann_none  = monthly_to_realm_annual(dfm_by_excl.get("none", pd.DataFrame()),  merge_AA_OC=True, drop_antarctic=True)
    ann_crops = monthly_to_realm_annual(dfm_by_excl.get("crops", pd.DataFrame()), merge_AA_OC=True, drop_antarctic=True)
    ann_cp    = monthly_to_realm_annual(dfm_by_excl.get("crops_pasture", pd.DataFrame()), merge_AA_OC=True, drop_antarctic=True)

    plot_ts_realms(ann_none,  os.path.join(OUT_DIR, "ts_realms_none.png"),           smooth_years=SMOOTH_YEARS, title="Realm time series — Exclusion: none")
    plot_ts_realms(ann_crops, os.path.join(OUT_DIR, "ts_realms_crops.png"),          smooth_years=SMOOTH_YEARS, title="Realm time series — Exclusion: crops")
    plot_ts_realms(ann_cp,    os.path.join(OUT_DIR, "ts_realms_crops_pasture.png"),  smooth_years=SMOOTH_YEARS, title="Realm time series — Exclusion: crops+pasture")

    # === Lollipops (realms only; separate x-scale per scenario) ===
    plot_lollipop_overlay(ann_none, ann_crops,
                          label_A="No exclusion", label_B="Exclude crops",
                          out_png=os.path.join(OUT_DIR, "trend_lollipop_realms_none_vs_crops.png"))
    plot_lollipop_overlay(ann_none, ann_cp,
                          label_A="No exclusion", label_B="Exclude crops+pasture",
                          out_png=os.path.join(OUT_DIR, "trend_lollipop_realms_none_vs_crops_pasture.png"))

    # === Spatial 3×3 (biome×realm) with requested aesthetics ===
    plot_spatial_pct_biorealm(dfm_by_excl,
                              out_png=os.path.join(OUT_DIR, "spatial_pct_change_biorealm_3x3.png"),
                              vlim=50.0)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
