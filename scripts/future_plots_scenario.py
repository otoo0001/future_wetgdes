#!/usr/bin/env python3
# plot_outputs_future_gde.py
#
# Makes:
#  1) Realm time-series grids (per RUN_TAG and per exclusion):
#       • AO merged, Antarctic dropped; 5–95% envelope over futures
#  2) Realm lollipop trends (per RUN_TAG):
#       • None (solid) vs Crops (dashed, offset below)
#       • None (solid) vs Crops+Pasture (dashed, offset below)
#       • Separate x-range per scenario; red=decline, blue=increase; legend at bottom
#  3) Optional comparison lollipops across run tags (if RUN_TAGS has >1)
#
# Expects Parquet created by future_gdes_area.py (new version):
#   <PARQUET_DIR>/gde_area_by_biome_realm_monthly_<RUN_TAG>_<scenario>.parquet
#   or live hive datasets:
#   <PARQUET_DIR>/gde_area_by_biome_realm_monthly_<RUN_TAG>_<scenario>/year=YYYY/part_*.parquet

import os, glob, math, warnings
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# Config via env
# ──────────────────────────────────────────────────────────────────────────────
PARQUET_DIR    = os.environ.get("PARQUET_DIR", "/projects/prjs1578/futurewetgde/wetGDEs_area_scenarios")
PARQUET_PREFIX = os.environ.get("PARQUET_PREFIX", "gde_area_by_biome_realm_monthly")
RUN_TAGS       = os.environ.get("RUN_TAGS", "full").split()
OUT_DIR_ROOT   = os.environ.get("PLOT_DIR", None)  # if None, we’ll use <PARQUET_DIR>/plots_biome_realm_<run_tag>
SMOOTH_YEARS   = int(os.environ.get("SMOOTH_YEARS", "5"))

SCENARIOS      = ["historical","ssp126","ssp370","ssp585"]
FUTURE_SCENS   = ["ssp126","ssp370","ssp585"]
EXCLUSIONS     = ["none","crops","crops_pasture"]
COLMAP = {
    "none":          "area_none_km2",
    "crops":         "area_crops_excl_km2",
    "crops_pasture": "area_crops_pasture_excl_km2",
}

# realm code/name
REALM_NAME_MAP = {
    "AA":"Australasian","AT":"Afrotropical","IM":"Indo Malayan","NA":"Nearctic",
    "NT":"Neotropical","OC":"Oceanian","PA":"Palearctic","AN":"Antarctic"
}
MERGE_TO_CODE   = {"AA","OC"}  # AO
MERGED_CODE     = "AO"
MERGED_NAME     = "Australasian+Oceanian"
DROP_CODES      = {"AN"}       # drop Antarctic

def realm_full_from_code(code: str) -> str:
    if code == MERGED_CODE: return MERGED_NAME
    return REALM_NAME_MAP.get(code, code)

# plots styling
SCEN_COLOR = {"historical":"#222222","ssp126":"#1a9850","ssp370":"#fdae61","ssp585":"#d73027"}
COLOR_NEG  = "#d73027"  # decline
COLOR_POS  = "#1f78b4"  # increase

# baseline & late windows (years) — only for trend selection, not plots themselves
BASELINE_YR = (1985, 2014)

# ──────────────────────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────────────────────
def parquet_sources_for(run_tag: str, scen: str) -> List[str]:
    final = os.path.join(PARQUET_DIR, f"{PARQUET_PREFIX}_{run_tag}_{scen}.parquet")
    if os.path.isfile(final):
        return [final]
    hive_root = os.path.join(PARQUET_DIR, f"{PARQUET_PREFIX}_{run_tag}_{scen}")
    parts = sorted(glob.glob(os.path.join(hive_root, "year=*/*.parquet")))
    return parts

def load_monthly(run_tag: str, excl: str) -> pd.DataFrame:
    """Load monthly biome×realm totals for one exclusion across scenarios → long DF."""
    col = COLMAP[excl]
    pieces = []
    for scen in SCENARIOS:
        srcs = parquet_sources_for(run_tag, scen)
        if not srcs:
            print(f"[warn] missing Parquet for {run_tag}/{scen}")
            continue
        use_cols = ["time","BIOME_ID_REALM", col]
        d = pd.concat((pd.read_parquet(p, columns=use_cols) for p in srcs), ignore_index=True)
        d = d.rename(columns={col: "area_km2"})
        d["scenario"] = scen
        d["time"] = pd.to_datetime(d["time"])
        pieces.append(d)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# Aggregations
# ──────────────────────────────────────────────────────────────────────────────
def monthly_to_realm_annual(dfm: pd.DataFrame, merge_AA_OC=True, drop_antarctic=True) -> pd.DataFrame:
    if dfm.empty: return dfm
    df = dfm.copy()
    parts = df["BIOME_ID_REALM"].str.split("_", n=1, expand=True)
    parts.columns = ["biome","realm_code"]
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
        years = np.arange(d["year"].min(), d["year"].max()+1, dtype=int)
        s = pd.Series(d.set_index("year")["area_km2"], index=years, dtype=float)
        y = s.rolling(window_years, center=True, min_periods=max(1, window_years//2)).mean()
        out.append(pd.DataFrame({"scenario": sc, "realm": r, "year": y.index.values, "area_km2": y.values}))
    return pd.concat(out, ignore_index=True)

def futures_envelope(df_ann: pd.DataFrame) -> pd.DataFrame:
    fut = df_ann[df_ann["scenario"].isin(FUTURE_SCENS)].copy()
    if fut.empty: return pd.DataFrame(columns=["realm","year","y5","y95"])
    pvt = fut.pivot_table(index=["realm","year"], columns="scenario", values="area_km2", aggfunc="first")
    q = pvt.quantile([0.05, 0.95], axis=1).T
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

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_ts_realms(df_ann: pd.DataFrame, out_png: str, title=None, smooth_years=SMOOTH_YEARS):
    if df_ann.empty:
        print(f"[warn] TS: no data for {out_png}")
        return
    df_s = smooth_by_year(df_ann, smooth_years)
    realms = sorted(df_s["realm"].unique())
    n = len(realms); ncols = 4; nrows = math.ceil(n/ncols)
    fig = plt.figure(figsize=(4.0*ncols, 2.9*nrows + 1.0), dpi=220)

    env = futures_envelope(df_s)
    hist_end = df_s.loc[df_s["scenario"]=="historical","year"].max()

    for i, realm in enumerate(realms, 1):
        ax = plt.subplot(nrows, ncols, i)
        e = env[env["realm"] == realm]
        if not e.empty:
            ax.fill_between(e["year"], e["y5"], e["y95"], color="0.75", alpha=0.35)
        for sc in SCENARIOS:
            d = df_s[(df_s["scenario"]==sc) & (df_s["realm"]==realm)]
            if d.empty: continue
            ax.plot(d["year"], d["area_km2"], lw=2, color=SCEN_COLOR[sc])
        if pd.notna(hist_end):
            ax.axvline(hist_end, color="0.6", lw=1, ls="--")
        ax.set_title(realm, fontsize=10)
        ax.grid(True, alpha=0.2, lw=0.5)
        if i <= (nrows-1)*ncols: ax.set_xticklabels([])
        ax.set_ylabel("Area km$^2$" if i in (1, ncols+1) else "")
    if title: fig.suptitle(title, y=0.995, fontsize=12)

    handles = [Line2D([0],[0], color=SCEN_COLOR[s], lw=2) for s in SCENARIOS]
    fig.legend(handles, SCENARIOS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] TS -> {out_png}")

def plot_lollipop_overlay(df_ann_A: pd.DataFrame, df_ann_B: pd.DataFrame,
                          label_A: str, label_B: str, out_png: str):
    """Overlay realm lollipops for two exclusions (separate x-range per scenario; dashed below)."""
    if df_ann_A.empty or df_ann_B.empty:
        print(f"[warn] lollipop skipped ({out_png}): missing data"); return

    tt_A = trend_table(df_ann_A)
    tt_B = trend_table(df_ann_B)
    realms = sorted(set(tt_A["realm"]).union(tt_B["realm"]))
    y_main  = np.arange(len(realms))[::-1]
    y_below = y_main - 0.35  # offset series below

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=220)
    axes = axes.ravel()
    for ax, sc in zip(axes, SCENARIOS):
        dA = tt_A[tt_A["scenario"] == sc].set_index("realm").reindex(realms)
        dB = tt_B[tt_B["scenario"] == sc].set_index("realm").reindex(realms)

        vals = np.concatenate([dA["slope_km2_per_year"].to_numpy(float),
                               dB["slope_km2_per_year"].to_numpy(float)])
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
    legend_labels = [label_A, label_B, f"{label_A} marker", f"{label_B} marker"]
    fig.legend(legend_lines, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0.05, 0.08, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] lollipop -> {out_png}")

def plot_lollipop_compare_run_tags(ann_by_tag: Dict[str,pd.DataFrame], out_png: str):
    """Compare families (run tags) at realm level for the NONE exclusion only (per scenario)."""
    if not ann_by_tag: 
        print(f"[warn] no data for {out_png}"); return
    tags = list(ann_by_tag.keys())
    realms = sorted(set().union(*[set(d["realm"].unique()) for d in ann_by_tag.values()]))

    # colors/linestyles per family
    tag_styles = {}
    base_linestyles = ["-","--","-.",":"]
    for i, tg in enumerate(tags):
        tag_styles[tg] = dict(ls=base_linestyles[i % len(base_linestyles)], color="k")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=220)
    axes = axes.ravel()
    for ax, sc in zip(axes, SCENARIOS):
        # collect all slopes to set symmetric x-limits per scenario
        all_vals = []
        for tg in tags:
            tt = trend_table(ann_by_tag[tg])
            vals = tt.loc[tt["scenario"]==sc, "slope_km2_per_year"].to_numpy(float)
            all_vals.append(vals)
        all_vals = np.concatenate(all_vals) if all_vals else np.array([])
        xmax = float(np.nanmax(np.abs(all_vals))) * 1.08 if all_vals.size else 1.0
        ax.set_xlim(-xmax, xmax)

        y0 = np.arange(len(realms))[::-1]
        for j, tg in enumerate(tags):
            tt = trend_table(ann_by_tag[tg]).set_index(["scenario","realm"])
            x = [tt.loc[(sc, r), "slope_km2_per_year"] if (sc, r) in tt.index else np.nan for r in realms]
            y = y0 - (j * 0.25)  # stagger families
            for yi, xi in zip(y, x):
                if not np.isfinite(xi): continue
                color = COLOR_POS if xi > 0 else COLOR_NEG
                ax.plot([0, xi], [yi, yi], lw=2, color=color, ls=tag_styles[tg]["ls"])
                ax.plot([xi], [yi], marker="o", ms=4, color=color)

        ax.axvline(0, color="0.75", lw=1)
        ax.set_yticks(y0); ax.set_yticklabels(realms, fontsize=8)
        ax.set_title(sc, fontsize=11, pad=6)
        ax.grid(True, axis="x", alpha=0.25, lw=0.6)
        ax.set_xlabel("km$^2$ per year")

    legend_items = [Line2D([0],[0], color="k", lw=2, ls=tag_styles[t]["ls"]) for t in tags]
    fig.legend(legend_items, tags, loc="lower center", ncol=min(4, len(tags)), frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0.05, 0.08, 0.99, 0.98])
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] compare-run-tags lollipop -> {out_png}")

# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[setup] PARQUET_DIR={PARQUET_DIR}")
    print(f"[setup] RUN_TAGS={RUN_TAGS}")

    # For each run_tag, plot TS + lollipops (within-tag overlays of exclusions)
    ann_none_by_tag_for_compare = {}

    for run_tag in RUN_TAGS:
        out_dir = OUT_DIR_ROOT or os.path.join(PARQUET_DIR, f"plots_biome_realm_{run_tag}")
        os.makedirs(out_dir, exist_ok=True)

        # Load monthly per exclusion
        dfm_by_excl = {excl: load_monthly(run_tag, excl) for excl in EXCLUSIONS}

        # Realm TS, 3 separate images (one per exclusion)
        for excl in EXCLUSIONS:
            ann = monthly_to_realm_annual(dfm_by_excl.get(excl, pd.DataFrame()),
                                          merge_AA_OC=True, drop_antarctic=True)
            if ann.empty: 
                print(f"[warn] no data for {run_tag}/{excl}")
                continue
            plot_ts_realms(
                ann,
                out_png=os.path.join(out_dir, f"ts_realms_{run_tag}_{excl}.png"),
                title=f"Realm time series — {run_tag} — exclusion: {excl}"
            )

        # Lollipops (overlay within run_tag)
        ann_none   = monthly_to_realm_annual(dfm_by_excl.get("none", pd.DataFrame()), merge_AA_OC=True, drop_antarctic=True)
        ann_crops  = monthly_to_realm_annual(dfm_by_excl.get("crops", pd.DataFrame()), merge_AA_OC=True, drop_antarctic=True)
        ann_cp     = monthly_to_realm_annual(dfm_by_excl.get("crops_pasture", pd.DataFrame()), merge_AA_OC=True, drop_antarctic=True)

        if not ann_none.empty:
            ann_none_by_tag_for_compare[run_tag] = ann_none

        plot_lollipop_overlay(
            ann_none, ann_crops,
            label_A="No exclusion", label_B="Exclude crops",
            out_png=os.path.join(out_dir, f"trend_lollipop_realms_{run_tag}_none_vs_crops.png")
        )
        plot_lollipop_overlay(
            ann_none, ann_cp,
            label_A="No exclusion", label_B="Exclude crops+pasture",
            out_png=os.path.join(out_dir, f"trend_lollipop_realms_{run_tag}_none_vs_crops_pasture.png")
        )

    # If multiple run tags requested, also make a cross-family comparison lollipop (NONE only)
    if len(RUN_TAGS) > 1  and ann_none_by_tag_for_compare:
        out_dir = OUT_DIR_ROOT or os.path.join(PARQUET_DIR, f"plots_biome_realm_compare_run_tags")
        os.makedirs(out_dir, exist_ok=True)
        plot_lollipop_compare_run_tags(
            ann_by_tag=ann_none_by_tag_for_compare,
            out_png=os.path.join(out_dir, "trend_lollipop_realms_compare_run_tags_none.png")
        )
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    main()
# ──────────────────────────────────────────────────────────────────────────────            

