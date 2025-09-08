#!/usr/bin/env python3
# future_plots_scenario.py
#
# Plots from per-family Parquet produced by future_gdes_area.py:
#   • Realm time series with trend lines; climate_only (solid) vs landuse_only (dashed)
#   • Realm lollipop trends (km²/yr) per scenario (FULL)
#   • Climate-only vs Land-use-only lollipop overlay (km²/yr), per scenario
#   • Exclusion overlays (None vs Crops, None vs Crops+Pasture) lollipop (km²/yr), per scenario (FULL)
#   • Spatial % change (late vs baseline) 3×3 (rows=exclusions, cols=scenarios), Robinson, shared short colorbar
#   • GIF flipbook 3×3 (rows=exclusions, cols=scenarios; frames=decades), fixed colors with decade label
#
# Non-spatial plots aggregate to *realm* (Australasian+Oceanian merged; Antarctic removed).
# Spatial maps use biome×realm polygons.
#
# Looks for Parquet like:
#   PARQUET_DIR/gde_area_by_biome_realm_monthly_{RUN_TAG}_{SCENARIO}.parquet
# or hive parts:
#   PARQUET_DIR/gde_area_by_biome_realm_monthly_{RUN_TAG}_{SCENARIO}/year=YYYY/part_*.parquet

import os, glob, math, warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation
try:
    from matplotlib.animation import PillowWriter
except Exception:
    PillowWriter = None

import cartopy.crs as ccrs

# ──────────────────────────────────────────────────────────────────────────────
# Config (env-overridable)
# ──────────────────────────────────────────────────────────────────────────────
PARQUET_DIR    = os.environ.get("PARQUET_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_area_scenarios")
PARQUET_PREFIX = os.environ.get("PARQUET_PREFIX", "gde_area_by_biome_realm_monthly")
OUT_DIR        = os.environ.get("PLOT_DIR", os.path.join(PARQUET_DIR, "plots_biome_realm_newnew"))
WWF_SHAPE      = os.environ.get("WWF_SHAPE", "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")

SCENARIOS      = ["historical", "ssp126", "ssp370", "ssp585"]
FUTURE_SCENS   = ["ssp126", "ssp370", "ssp585"]
EXCLUSIONS     = ["none", "crops", "crops_pasture"]
RUN_TAGS       = os.environ.get("RUN_TAGS", "full,climate_only,landuse_only").split(",")

COLMAP = {
    "none":          "area_none_km2",
    "crops":         "area_crops_excl_km2",
    "crops_pasture": "area_crops_pasture_excl_km2",
}

BASELINE_YR   = (1985, 2014)
LATE_YR       = (2071, 2100)
SMOOTH_YEARS  = int(os.environ.get("SMOOTH_YEARS", "5"))

# realm mapping + merge rules
REALM_NAME_MAP = {
    "AA": "Australasian","AT": "Afrotropical","IM":"Indo Malayan",
    "NA":"Nearctic","NT":"Neotropical","OC":"Oceanian","PA":"Palearctic","AN":"Antarctic"
}
MERGE_TO_CODE   = {"AA","OC"}   # → AO
MERGED_CODE     = "AO"
MERGED_NAME     = "Australasian+Oceanian"
DROP_CODES      = {"AN"}

def realm_full_from_code(code: str) -> str:
    if code == MERGED_CODE: return MERGED_NAME
    return REALM_NAME_MAP.get(code, code)

# colors / styles
SCEN_COLOR = {"historical":"#222222","ssp126":"#1a9850","ssp370":"#fdae61","ssp585":"#d73027"}
COLOR_NEG  = "#d73027"  # declines
COLOR_POS  = "#1f78b4"  # increases
CMAP_SPAT  = plt.get_cmap("RdBu")  # used with TwoSlopeNorm(vcenter=0)

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parquet_sources(run_tag: str, scen: str) -> List[str]:
    final = os.path.join(PARQUET_DIR, f"{PARQUET_PREFIX}_{run_tag}_{scen}.parquet")
    if os.path.isfile(final):
        return [final]
    hive_root = os.path.join(PARQUET_DIR, f"{PARQUET_PREFIX}_{run_tag}_{scen}")
    parts = sorted(glob.glob(os.path.join(hive_root, "year=*/*.parquet")))
    return parts

def load_monthly(run_tag: str, excl: str) -> pd.DataFrame:
    col = COLMAP[excl]
    pieces = []
    for scen in SCENARIOS:
        srcs = _parquet_sources(run_tag, scen)
        if not srcs:
            print(f"[warn] missing Parquet for {run_tag}/{scen}: {os.path.join(PARQUET_DIR, f'{PARQUET_PREFIX}_{run_tag}_{scen}*')}")
            continue
        use_cols = ["time", "BIOME_ID_REALM", col]
        d = pd.concat((pd.read_parquet(p, columns=use_cols) for p in srcs), ignore_index=True)
        d = d.rename(columns={col: "area_km2"})
        d["scenario"] = scen
        d["time"] = pd.to_datetime(d["time"])
        pieces.append(d)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# Aggregation & utilities (realm)
# ──────────────────────────────────────────────────────────────────────────────
def monthly_to_realm_annual(dfm: pd.DataFrame, *, merge_AA_OC=True, drop_antarctic=True) -> pd.DataFrame:
    if dfm.empty: return dfm
    df = dfm.copy()
    parts = df["BIOME_ID_REALM"].str.split("_", n=1, expand=True)
    parts.columns = ["biome_str","realm_code"]
    if drop_antarctic:
        keep = parts["realm_code"] != "AN"
        df = df.loc[keep].copy()
        parts = parts.loc[keep]
    if merge_AA_OC:
        parts["realm_code"] = parts["realm_code"].where(~parts["realm_code"].isin(MERGE_TO_CODE), MERGED_CODE)
    df["realm"] = parts["realm_code"].apply(realm_full_from_code)
    df["year"] = df["time"].dt.year.astype(int)
    return df.groupby(["scenario","realm","year"], as_index=False)["area_km2"].sum()

def smooth_by_year(df_ann: pd.DataFrame, window_years: int) -> pd.DataFrame:
    if df_ann.empty or window_years <= 1: return df_ann
    out = []
    for (sc, r), d in df_ann.groupby(["scenario","realm"], sort=False):
        yrs = np.arange(d["year"].min(), d["year"].max()+1, dtype=int)
        s = pd.Series(d.set_index("year")["area_km2"], index=yrs, dtype=float)
        y = s.rolling(window_years, center=True, min_periods=max(1, window_years//2)).mean()
        out.append(pd.DataFrame({"scenario":sc,"realm":r,"year":y.index.values,"area_km2":y.values}))
    return pd.concat(out, ignore_index=True)

def futures_envelope(df_ann: pd.DataFrame) -> pd.DataFrame:
    fut = df_ann[df_ann["scenario"].isin(FUTURE_SCENS)]
    if fut.empty: return pd.DataFrame(columns=["realm","year","y5","y95"])
    p = fut.pivot_table(index=["realm","year"], columns="scenario", values="area_km2", aggfunc="first")
    q = p.quantile([0.05,0.95], axis=1).T
    q.columns = ["y5","y95"]
    return q.reset_index()

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

def pct_change_realm(df_ann: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (sc, r), d in df_ann.groupby(["scenario","realm"]):
        b = d[(d["year"]>=BASELINE_YR[0]) & (d["year"]<=BASELINE_YR[1])]["area_km2"].mean()
        if sc == "historical":
            f = d[(d["year"]>=BASELINE_YR[0]) & (d["year"]<=BASELINE_YR[1])]["area_km2"].mean()
        else:
            f = d[(d["year"]>=LATE_YR[0]) & (d["year"]<=LATE_YR[1])]["area_km2"].mean()
        pct = np.nan if not (b and np.isfinite(b)) else (f-b)/b*100.0
        out.append((sc, r, pct))
    return pd.DataFrame(out, columns=["scenario","realm","pct_change"])

# ──────────────────────────────────────────────────────────────────────────────
# Spatial geometry (biome×realm)
# ──────────────────────────────────────────────────────────────────────────────
def load_biome_realm_geoms() -> gpd.GeoDataFrame:
    shp = gpd.read_file(WWF_SHAPE)
    if "BIOME" not in shp.columns or "REALM" not in shp.columns:
        raise RuntimeError("Shapefile must contain BIOME and REALM")
    shp = shp.to_crs("EPSG:4326")
    shp = shp[shp["REALM"] != "AN"].copy()
    shp["BIOME_ID_REALM"] = shp["BIOME"].astype(int).astype(str) + "_" + shp["REALM"].astype(str)
    diss = shp.dissolve(by="BIOME_ID_REALM", as_index=False, aggfunc="first")[["BIOME_ID_REALM","geometry"]]
    return diss

def pct_change_by_biorealm(dfm: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    if dfm.empty:
        return {sc: pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"]) for sc in FUTURE_SCENS}
    df = dfm.copy()
    df["year"] = df["time"].dt.year.astype(int)
    for sc in FUTURE_SCENS:
        d_f = df[(df["scenario"]==sc) & (df["year"].between(LATE_YR[0], LATE_YR[1]))]
        d_b = df[(df["scenario"]=="historical") & (df["year"].between(BASELINE_YR[0], BASELINE_YR[1]))]
        if d_f.empty or d_b.empty:
            out[sc] = pd.DataFrame(columns=["BIOME_ID_REALM","pct_change"]); continue
        f_mean = d_f.groupby("BIOME_ID_REALM", as_index=False)["area_km2"].mean().rename(columns={"area_km2":"V_future"})
        b_mean = d_b.groupby("BIOME_ID_REALM", as_index=False)["area_km2"].mean().rename(columns={"area_km2":"V_baseline"})
        m = pd.merge(b_mean, f_mean, on="BIOME_ID_REALM", how="outer")
        m["pct_change"] = np.where(m["V_baseline"]>0, (m["V_future"]-m["V_baseline"])/m["V_baseline"]*100.0, np.nan)
        out[sc] = m[["BIOME_ID_REALM","pct_change"]]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def add_trend_line(ax, years, values, color="0.2", lw=1.5, ls=":"):
    ok = np.isfinite(years) & np.isfinite(values)
    if ok.sum() < 2: return
    m, b = np.polyfit(years[ok], values[ok], 1)
    yfit = m*years + b
    ax.plot(years, yfit, lw=lw, ls=ls, color=color, alpha=0.9)

def _remove_layer(lyr):
    if lyr is None: 
        return
    try:
        lyr.remove()
    except Exception:
        try:
            for a in list(lyr):
                try: a.remove()
                except Exception: pass
        except Exception:
            pass

def _save_anim(anim: FuncAnimation, base_path: str, fps=2):
    gif_path = f"{base_path}.gif"
    if PillowWriter is None:
        print("[warn] Pillow writer not available; saving first frame PNG instead.")
        anim._init_draw()
        anim._draw_frame(0)
        anim._fig.savefig(f"{base_path}_frame0.png", dpi=180, bbox_inches="tight")
        return
    try:
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"[ok] GIF -> {gif_path}")
    except Exception as e:
        print(f"[warn] GIF save failed ({e}); saving frame0 PNG instead.")
        anim._init_draw()
        anim._draw_frame(0)
        anim._fig.savefig(f"{base_path}_frame0.png", dpi=180, bbox_inches="tight")

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_ts_realms_overlay(df_ann_A: pd.DataFrame, df_ann_B: pd.DataFrame,
                           label_A: str, label_B: str, out_png: str,
                           smooth_years: int = SMOOTH_YEARS, show_full_env=False, df_ann_full=None):
    if df_ann_A.empty or df_ann_B.empty:
        print(f"[warn] TS overlay skipped ({label_A} vs {label_B}): missing data"); return

    dfA = smooth_by_year(df_ann_A, smooth_years)
    dfB = smooth_by_year(df_ann_B, smooth_years)

    realms = sorted(set(dfA["realm"]).union(dfB["realm"]))
    n = len(realms); ncols = 4; nrows = math.ceil(n/ncols)
    fig = plt.figure(figsize=(4.0*ncols, 2.9*nrows + 1.0), dpi=220)

    env = futures_envelope(smooth_by_year(df_ann_full, smooth_years)) if (show_full_env and (df_ann_full is not None)) else pd.DataFrame()
    hist_end = dfA.loc[dfA["scenario"]=="historical","year"].max()

    for i, realm in enumerate(realms, 1):
        ax = plt.subplot(nrows, ncols, i)

        if not env.empty:
            e = env[env["realm"]==realm]
            if not e.empty:
                ax.fill_between(e["year"], e["y5"], e["y95"], color="0.75", alpha=0.32)

        for sc in SCENARIOS:
            dA = dfA[(dfA["scenario"]==sc) & (dfA["realm"]==realm)]
            dB = dfB[(dfB["scenario"]==sc) & (dfB["realm"]==realm)]
            if not dA.empty:
                ax.plot(dA["year"], dA["area_km2"], lw=2, color=SCEN_COLOR[sc], ls="-", label=f"{label_A} {sc}")
                add_trend_line(ax, dA["year"].values, dA["area_km2"].values, color=SCEN_COLOR[sc], lw=1.0, ls=":")
            if not dB.empty:
                ax.plot(dB["year"], dB["area_km2"], lw=2, color=SCEN_COLOR[sc], ls="--", label=f"{label_B} {sc}")
                add_trend_line(ax, dB["year"].values, dB["area_km2"].values, color=SCEN_COLOR[sc], lw=1.0, ls="--")

        if pd.notna(hist_end):
            ax.axvline(hist_end, color="0.6", lw=1, ls="--")

        ax.set_title(realm, fontsize=10)
        ax.grid(True, alpha=0.2, lw=0.5)
        if i <= (nrows-1)*ncols:
            ax.set_xticklabels([])
        ax.set_ylabel("Area km$^2$" if i in (1, ncols+1) else "")

    handles = [Line2D([0],[0], color="k", lw=2, ls="-"),
               Line2D([0],[0], color="k", lw=2, ls="--")]
    labs = [label_A, label_B]
    fig.legend(handles, labs, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] TS -> {out_png}")

def plot_lollipop_realms(df_ann: pd.DataFrame, out_png: str):
    if df_ann.empty: print("[warn] no data for lollipops"); return
    tt = trend_table(df_ann)
    base = tt[tt["scenario"]=="ssp585"].copy()
    base["abs"] = base["slope_km2_per_year"].abs()
    order = base.sort_values("abs")["realm"].tolist()
    for r in sorted(tt["realm"].unique()):
        if r not in order: order.append(r)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), dpi=220, sharex=False)
    axes = axes.ravel()

    for ax, sc in zip(axes, SCENARIOS):
        d = tt[tt["scenario"]==sc].set_index("realm").reindex(order)
        y = np.arange(len(d))[::-1]
        x = d["slope_km2_per_year"].to_numpy(float)

        ax.axvline(0, color="0.75", lw=1)
        for yi, xi in zip(y, x):
            if not np.isfinite(xi): continue
            color = COLOR_POS if xi > 0 else COLOR_NEG
            ax.plot([0, xi], [yi, yi], color=color, lw=2)
            ax.plot([xi], [yi], marker="o", color=color, ms=5)
        ax.set_yticks(y); ax.set_yticklabels(d.index.tolist(), fontsize=8)
        ax.set_title(sc, fontsize=11)
        ax.grid(True, axis="x", alpha=0.25)
        m = np.nanmax(np.abs(x)) if x.size else 1.0
        ax.set_xlim(-1.08*m, 1.08*m)
        ax.set_xlabel("km$^2$ per year")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] lollipops -> {out_png}")

def plot_lollipop_overlay_runs(df_ann_A: pd.DataFrame, df_ann_B: pd.DataFrame,
                               label_A: str, label_B: str, out_png: str):
    if df_ann_A.empty or df_ann_B.empty:
        print(f"[warn] lollipop overlay skipped ({label_A} vs {label_B}): missing data"); return

    tt_A = trend_table(df_ann_A)
    tt_B = trend_table(df_ann_B)
    realms = sorted(set(tt_A["realm"]).union(tt_B["realm"]))
    y_main  = np.arange(len(realms))[::-1]
    y_below = y_main - 0.35

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=220)
    axes = axes.ravel()

    for ax, sc in zip(axes, SCENARIOS):
        dA = tt_A[tt_A["scenario"] == sc].set_index("realm").reindex(realms)
        dB = tt_B[tt_B["scenario"] == sc].set_index("realm").reindex(realms)

        vals = np.concatenate([
            dA["slope_km2_per_year"].to_numpy(dtype=float),
            dB["slope_km2_per_year"].to_numpy(dtype=float)
        ])
        vals = vals[np.isfinite(vals)]
        xmax = float(np.abs(vals).max())*1.08 if vals.size else 1.0
        ax.set_xlim(-xmax, xmax)

        for yi, xi in zip(y_main, dA["slope_km2_per_year"].to_numpy()):
            if not np.isfinite(xi): continue
            color = COLOR_POS if xi > 0 else COLOR_NEG
            ax.plot([0, xi], [yi, yi], color=color, lw=2, ls="-", zorder=3)
            ax.plot([xi], [yi], marker="o", color=color, ms=5, zorder=4)

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
    print(f"[ok] lollipop overlay -> {out_png}")

def plot_lollipop_overlay_exclusions(df_ann_none: pd.DataFrame, df_ann_excl: pd.DataFrame,
                                     label_excl: str, out_png: str):
    """
    Overlay lollipop trends comparing exclusions within the SAME family (e.g., FULL):
      • 'none' = solid stems + filled circles
      • excluded (crops or crops+pasture) = dashed stems + open squares, slightly offset
    Separate x-range per scenario.
    """
    if df_ann_none.empty or df_ann_excl.empty:
        print(f"[warn] lollipop excl overlay skipped (none vs {label_excl}): missing data"); return

    tt_A = trend_table(df_ann_none)
    tt_B = trend_table(df_ann_excl)
    realms = sorted(set(tt_A["realm"]).union(tt_B["realm"]))
    y_main  = np.arange(len(realms))[::-1]
    y_below = y_main - 0.35

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=220)
    axes = axes.ravel()

    for ax, sc in zip(axes, SCENARIOS):
        dA = tt_A[tt_A["scenario"] == sc].set_index("realm").reindex(realms)
        dB = tt_B[tt_B["scenario"] == sc].set_index("realm").reindex(realms)

        vals = np.concatenate([
            dA["slope_km2_per_year"].to_numpy(dtype=float),
            dB["slope_km2_per_year"].to_numpy(dtype=float)
        ])
        vals = vals[np.isfinite(vals)]
        xmax = float(np.abs(vals).max())*1.08 if vals.size else 1.0
        ax.set_xlim(-xmax, xmax)

        # none (solid)
        for yi, xi in zip(y_main, dA["slope_km2_per_year"].to_numpy()):
            if not np.isfinite(xi): continue
            color = COLOR_POS if xi > 0 else COLOR_NEG
            ax.plot([0, xi], [yi, yi], color=color, lw=2, ls="-", zorder=3)
            ax.plot([xi], [yi], marker="o", color=color, ms=5, zorder=4)

        # excluded (dashed)
        for yi, xi in zip(y_below, dB["slope_km2_per_year"].to_numpy()):
            if not np.isfinite(xi): continue
            color = COLOR_POS if xi > 0 else COLOR_NEG
            ax.plot([0, xi], [yi, yi], color=color, lw=2, ls="--", zorder=2)
            ax.plot([xi], [yi], marker="s", mfc="none", mec=color, ms=5, zorder=3)

        ax.axvline(0, color="0.75", lw=1)
        ax.set_yticks(y_main); ax.set_yticklabels(realms, fontsize=8)
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
    legend_labels = ["No exclusion", label_excl, "No exclusion marker", f"{label_excl} marker"]
    fig.legend(legend_lines, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0.05, 0.08, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] lollipop excl overlay -> {out_png}")

def plot_spatial_pct_biorealm(dfm: pd.DataFrame, out_png: str, vlim=50.0):
    geom = load_biome_realm_geoms()
    pct_by = pct_change_by_biorealm(dfm)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)

    fig = plt.figure(figsize=(12, 10), dpi=260)
    proj = ccrs.Robinson()

    for r, excl in enumerate(EXCLUSIONS, start=1):
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
    cax = fig.add_axes([0.92, 0.35, 0.02, 0.30])
    cb = plt.colorbar(sm, cax=cax, orientation="vertical"); cb.set_label("% change"); cb.outline.set_visible(False)

    fig.tight_layout(rect=[0.02, 0.02, 0.90, 0.98])
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[ok] spatial 3×3 -> {out_png}")

# ── 3×3 Flipbook (GIF): rows=exclusions, cols=scenarios; frames=decades ──────
def anim_flipbook_spatial_3x3(dfm_by_excl: Dict[str, pd.DataFrame], out_base: str,
                              window_years=10, vlim=None):
    geom = load_biome_realm_geoms()

    stats = {}
    buckets = set()
    all_vals = []

    for excl in EXCLUSIONS:
        dfm = dfm_by_excl.get(excl, pd.DataFrame())
        if dfm is None or dfm.empty:
            continue
        df = dfm.copy()
        df["year"] = df["time"].dt.year.astype(int)
        base = df[(df["scenario"]=="historical") & (df["year"].between(BASELINE_YR[0], BASELINE_YR[1]))]
        if base.empty:
            continue
        B = base.groupby("BIOME_ID_REALM", as_index=False)["area_km2"].mean().rename(columns={"area_km2":"B"})
        for scen in FUTURE_SCENS:
            fut = df[df["scenario"]==scen].copy()
            if fut.empty:
                continue
            fut["bucket"] = (fut["year"]//window_years)*window_years
            ST = fut.groupby(["bucket","BIOME_ID_REALM"], as_index=False)["area_km2"].mean().rename(columns={"area_km2":"F"})
            stats[(excl, scen)] = (B, ST)
            buckets.update(ST["bucket"].unique())
            m = pd.merge(B, ST, on="BIOME_ID_REALM", how="right")
            m["pct"] = np.where(m["B"]>0, (m["F"]-m["B"])/m["B"]*100.0, np.nan)
            all_vals.append(m["pct"].to_numpy())

    buckets = sorted(int(b) for b in buckets)
    if not stats or not buckets:
        print("[warn] 3×3 flipbook: nothing to animate"); 
        return

    if vlim is None:
        arr = np.concatenate([v for v in all_vals if v.size])
        vmax = np.nanmax(np.abs(arr)) if arr.size else 50.0
        vlim = float(np.ceil(vmax/10.0)*10.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)

    fig = plt.figure(figsize=(12, 10), dpi=220)
    proj = ccrs.Robinson()
    axes = []
    for r, excl in enumerate(EXCLUSIONS, start=1):
        row_axes = []
        for c, scen in enumerate(FUTURE_SCENS, start=1):
            ax = plt.subplot(3, 3, (r-1)*3 + c, projection=proj)
            ax.set_global()
            ax.set_title(f"{excl} • {scen}", fontsize=10, pad=2)
            ax.coastlines(linewidth=0.1, color="0.4")
            ax.add_geometries(geom["geometry"], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="0.25", linewidth=0, zorder=2)
            ax.set_axis_off()
            try:
                ax.outline_patch.set_visible(False); ax.background_patch.set_visible(False)
            except Exception:
                pass
            row_axes.append(ax)
        axes.append(row_axes)

    title_txt = fig.text(0.06, 0.97, "", fontsize=14, weight="bold")
    dynamic_layers: Dict[Tuple[int,int], object] = {}

    def draw(i):
        bkt = buckets[i]
        title_txt.set_text(f"Decade: {bkt}s")
        for r, excl in enumerate(EXCLUSIONS):
            for c, scen in enumerate(FUTURE_SCENS):
                ax = axes[r][c]
                key = (r,c)
                _remove_layer(dynamic_layers.get(key))
                dynamic_layers[key] = None
                if (excl, scen) not in stats:
                    continue
                B, ST = stats[(excl, scen)]
                dd = ST[ST["bucket"] == bkt]
                if dd.empty:
                    continue
                m = pd.merge(B, dd, on="BIOME_ID_REALM", how="right")
                m["pct"] = np.where(m["B"]>0, (m["F"]-m["B"])/m["B"]*100.0, np.nan)
                g = geom.merge(m[["BIOME_ID_REALM","pct"]], on="BIOME_ID_REALM", how="left")
                geoms = list(g["geometry"])
                fcols = [CMAP_SPAT(norm(v)) if np.isfinite(v) else (0,0,0,0) for v in g["pct"].to_numpy()]
                dynamic_layers[key] = ax.add_geometries(
                    geoms, crs=ccrs.PlateCarree(), facecolor=fcols, edgecolor="none", linewidth=0.0, zorder=1
                )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_SPAT); sm.set_array([])
    cax = fig.add_axes([0.92, 0.35, 0.02, 0.30])
    cb = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("% change")
    cb.outline.set_visible(False)

    anim = FuncAnimation(fig, draw, frames=len(buckets), interval=900, repeat=False)
    _save_anim(anim, os.path.join(OUT_DIR, f"{out_base}_spatial_3x3"), fps=2)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[setup] PARQUET_DIR={PARQUET_DIR}")
    print(f"[setup] OUT_DIR={OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load monthly per run_tag (keep BIOME_ID_REALM for spatial)
    dfm_by_run_excl: Dict[Tuple[str,str], pd.DataFrame] = {}
    for run in RUN_TAGS:
        for excl in EXCLUSIONS:
            dfm_by_run_excl[(run, excl)] = load_monthly(run, excl)

    # === Time series overlays: climate_only (solid) vs landuse_only (dashed)
    for excl in EXCLUSIONS:
        ann_clim = monthly_to_realm_annual(dfm_by_run_excl.get(("climate_only", excl), pd.DataFrame()),
                                           merge_AA_OC=True, drop_antarctic=True)
        ann_lu   = monthly_to_realm_annual(dfm_by_run_excl.get(("landuse_only", excl), pd.DataFrame()),
                                           merge_AA_OC=True, drop_antarctic=True)
        ann_full = monthly_to_realm_annual(dfm_by_run_excl.get(("full", excl), pd.DataFrame()),
                                           merge_AA_OC=True, drop_antarctic=True)
        plot_ts_realms_overlay(
            ann_clim, ann_lu,
            label_A="Climate-only", label_B="Land-use-only",
            out_png=os.path.join(OUT_DIR, f"ts_realms_{excl}_climate_vs_landuse.png"),
            smooth_years=SMOOTH_YEARS, show_full_env=True, df_ann_full=ann_full
        )

    # === Lollipops (realms) for FULL family
    for excl in EXCLUSIONS:
        ann_full = monthly_to_realm_annual(dfm_by_run_excl.get(("full", excl), pd.DataFrame()),
                                           merge_AA_OC=True, drop_antarctic=True)
        plot_lollipop_realms(ann_full, os.path.join(OUT_DIR, f"trend_lollipop_realms_{excl}_full.png"))

    # === Lollipop overlays for Climate-only vs Land-use-only
    for excl in EXCLUSIONS:
        ann_clim = monthly_to_realm_annual(dfm_by_run_excl.get(("climate_only", excl), pd.DataFrame()),
                                           merge_AA_OC=True, drop_antarctic=True)
        ann_lu   = monthly_to_realm_annual(dfm_by_run_excl.get(("landuse_only", excl), pd.DataFrame()),
                                           merge_AA_OC=True, drop_antarctic=True)
        plot_lollipop_overlay_runs(
            ann_clim, ann_lu,
            label_A="Climate-only", label_B="Land-use-only",
            out_png=os.path.join(OUT_DIR, f"trend_lollipop_realms_{excl}_climate_vs_landuse.png")
        )

    # === Lollipop overlays for exclusions within FULL family
    ann_full_none  = monthly_to_realm_annual(dfm_by_run_excl.get(("full", "none"), pd.DataFrame()),
                                             merge_AA_OC=True, drop_antarctic=True)
    ann_full_crops = monthly_to_realm_annual(dfm_by_run_excl.get(("full", "crops"), pd.DataFrame()),
                                             merge_AA_OC=True, drop_antarctic=True)
    ann_full_cp    = monthly_to_realm_annual(dfm_by_run_excl.get(("full", "crops_pasture"), pd.DataFrame()),
                                             merge_AA_OC=True, drop_antarctic=True)

    plot_lollipop_overlay_exclusions(
        ann_full_none, ann_full_crops,
        label_excl="Exclude crops",
        out_png=os.path.join(OUT_DIR, "trend_lollipop_realms_none_vs_crops_full.png")
    )
    plot_lollipop_overlay_exclusions(
        ann_full_none, ann_full_cp,
        label_excl="Exclude crops+pasture",
        out_png=os.path.join(OUT_DIR, "trend_lollipop_realms_none_vs_crops_pasture_full.png")
    )

    # === Spatial 3×3 (% change) for FULL family (static)
    for excl in EXCLUSIONS:
        dfm_full = dfm_by_run_excl.get(("full", excl), pd.DataFrame())
        plot_spatial_pct_biorealm(dfm_full, os.path.join(OUT_DIR, f"spatial_pct_change_biorealm_{excl}_full.png"), vlim=50.0)

    # === Flipbook 3×3 (rows=exclusions, cols=scenarios; frames=decades), FULL family
    dfm_by_excl_full = {excl: dfm_by_run_excl.get(("full", excl), pd.DataFrame()) for excl in EXCLUSIONS}
    anim_flipbook_spatial_3x3(dfm_by_excl_full, out_base="flipbook_full", window_years=10,vlim=50.0)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
