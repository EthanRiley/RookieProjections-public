#!/usr/bin/env python3
"""
Visualizations for ba_yptpa vs yprr2.0_200rt head-to-head comparison.

Figures:
  5. Side-by-side leak test bars
  6. Bootstrap distribution of residual difference
  7. YPRR grid search heatmap
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

MODEL_FEATS = [
    "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
    "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
]


def _season_age(birthdate, year):
    sept1 = pd.Timestamp(f"{year}-09-01")
    return round((sept1 - birthdate).days / 365.25, 2)


def residual_spearman(df, feat_col, control_cols):
    cols = [feat_col] + control_cols + ["tier_ordinal"]
    sub = df[cols].dropna()
    if len(sub) < 30:
        return np.nan
    rank_feat = rankdata(sub[feat_col].values)
    rank_tier = rankdata(sub["tier_ordinal"].values)
    ctrl_ranks = np.column_stack([rankdata(sub[c].values) for c in control_cols])
    X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
    z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
    resid = rank_feat - X @ z
    sp, _ = spearmanr(resid, rank_tier)
    return sp


def load_data():
    from aggregation.aggregate_college_stats import (
        load_all_grades, get_player_seasons, build_lookups,
    )

    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, _, team_att_lookup, team_games_lookup = build_lookups(all_grades)

    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    dynasty["tier_ordinal"] = dynasty["computed_tier"].map(TIER_ORDER)
    dynasty = dynasty.dropna(subset=["tier_ordinal"]).copy()
    dynasty["tier_ordinal"] = dynasty["tier_ordinal"].astype(int)

    ba_yptpa, mag_yptpa = [], []
    ba_yprr, mag_yprr = [], []

    for _, row in dynasty.iterrows():
        name, draft_year = row["name"], row["draft_year"]
        birthdate = birth_lookup.get((name, draft_year))
        seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)

        _ba_yptpa, _mag_yptpa = np.nan, np.nan
        _ba_yprr, _mag_yprr = np.nan, np.nan

        if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
            for _, s in seasons.sort_values("grade_year").iterrows():
                yr = s["grade_year"]
                yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
                games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
                routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
                team = s.get("team_name", "")
                att = team_att_lookup.get((team, yr))
                tg = team_games_lookup.get((team, yr))

                yptpa_val = (yards / games) / (att / tg) if (att and att > 0 and games > 0 and tg and tg > 0) else None
                yprr_val = yards / routes if routes > 0 else None

                if pd.isna(_ba_yptpa) and games >= 8 and yptpa_val is not None and yptpa_val >= 1.4:
                    _ba_yptpa = _season_age(birthdate, yr)
                    _mag_yptpa = round(yptpa_val, 4)

                if pd.isna(_ba_yprr) and routes >= 200 and yprr_val is not None and yprr_val >= 2.0:
                    _ba_yprr = _season_age(birthdate, yr)
                    _mag_yprr = round(yprr_val, 4)

        ba_yptpa.append(_ba_yptpa)
        mag_yptpa.append(_mag_yptpa)
        ba_yprr.append(_ba_yprr)
        mag_yprr.append(_mag_yprr)

    dynasty["ba_yptpa"] = ba_yptpa
    dynasty["mag_yptpa"] = mag_yptpa
    dynasty["ba_yprr200"] = ba_yprr
    dynasty["mag_yprr"] = mag_yprr

    for col in ["ba_yptpa", "ba_yprr200"]:
        mx = dynasty[col].max()
        dynasty[f"{col}_imp"] = dynasty[col].fillna(round(mx + 1, 2))
    for col in ["mag_yptpa", "mag_yprr"]:
        dynasty[f"{col}_imp"] = dynasty[col].fillna(0)

    return dynasty


def fig5_leak_bars(dynasty):
    """Side-by-side bar chart: residual under progressive controls."""
    tests = {
        "Model\nfeats only": {
            "ba_yptpa_imp": MODEL_FEATS,
            "ba_yprr200_imp": MODEL_FEATS,
        },
        "+ Own\nmagnitude": {
            "ba_yptpa_imp": MODEL_FEATS + ["mag_yptpa_imp"],
            "ba_yprr200_imp": MODEL_FEATS + ["mag_yprr_imp"],
        },
        "+ Other\nmagnitude": {
            "ba_yptpa_imp": MODEL_FEATS + ["mag_yprr_imp"],
            "ba_yprr200_imp": MODEL_FEATS + ["mag_yptpa_imp"],
        },
        "+ Both\nmagnitudes": {
            "ba_yptpa_imp": MODEL_FEATS + ["mag_yptpa_imp", "mag_yprr_imp"],
            "ba_yprr200_imp": MODEL_FEATS + ["mag_yprr_imp", "mag_yptpa_imp"],
        },
    }

    yptpa_vals = []
    yprr_vals = []
    for test_name, ctrl_dict in tests.items():
        for ba_col, ctrl in ctrl_dict.items():
            sp = residual_spearman(dynasty, ba_col, list(dict.fromkeys(ctrl)))
            if ba_col == "ba_yptpa_imp":
                yptpa_vals.append(abs(sp) if pd.notna(sp) else 0)
            else:
                yprr_vals.append(abs(sp) if pd.notna(sp) else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tests))
    width = 0.35

    bars1 = ax.bar(x - width/2, yptpa_vals, width, label="ba_yptpa (1.4 YPTPA, 8 games)",
                   color="#2ca02c", edgecolor="white")
    bars2 = ax.bar(x + width/2, yprr_vals, width, label="yprr2.0_200rt (2.0 YPRR, 200 routes)",
                   color="#1f77b4", edgecolor="white")

    ax.set_ylabel("|Residual Spearman|", fontsize=12)
    ax.set_title("Efficiency Leak Test: ba_yptpa vs yprr2.0_200rt\n(higher = more signal after controls)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(list(tests.keys()), fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(yptpa_vals), max(yprr_vals)) * 1.3)

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def fig6_bootstrap(dynasty):
    """Bootstrap distribution of residual difference."""
    np.random.seed(42)
    n_boot = 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bootstrap Comparison: ba_yptpa vs yprr2.0_200rt\n(negative = ba_yptpa stronger)",
                 fontsize=13, fontweight="bold")

    for ax_i, (test_name, ctrl_fn_yptpa, ctrl_fn_yprr) in enumerate([
        ("Model feats only", MODEL_FEATS, MODEL_FEATS),
        ("+ Own magnitude",
         MODEL_FEATS + ["mag_yptpa_imp"],
         MODEL_FEATS + ["mag_yprr_imp"]),
    ]):
        boot_yptpa = []
        boot_yprr = []
        boot_diffs = []

        for _ in range(n_boot):
            idx = np.random.choice(len(dynasty), size=len(dynasty), replace=True)
            boot = dynasty.iloc[idx]

            sp_yptpa = residual_spearman(boot, "ba_yptpa_imp", ctrl_fn_yptpa)
            sp_yprr = residual_spearman(boot, "ba_yprr200_imp", ctrl_fn_yprr)

            if pd.notna(sp_yptpa) and pd.notna(sp_yprr):
                boot_yptpa.append(sp_yptpa)
                boot_yprr.append(sp_yprr)
                boot_diffs.append(sp_yptpa - sp_yprr)

        diffs = np.array(boot_diffs)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        pct_yptpa_wins = np.mean(diffs < 0) * 100

        ax = axes[ax_i]
        ax.hist(diffs, bins=40, color="#7f7f7f", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="No difference")
        ax.axvline(np.mean(diffs), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(diffs):+.3f}")
        ax.axvline(ci_lo, color="blue", linestyle=":", linewidth=1, label=f"95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        ax.axvline(ci_hi, color="blue", linestyle=":", linewidth=1)

        # Shade regions
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color="blue")

        ax.set_xlabel("Residual difference (yptpa - yprr)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{test_name}\nba_yptpa wins {pct_yptpa_wins:.0f}% of bootstraps", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def fig7_yprr_grid(dynasty):
    """Heatmap of YPRR grid search results."""
    grid_eval = pd.read_csv(os.path.join(DATA_DIR, "yprr_breakout_grid_eval.csv"))

    # Parse config names
    yprr_thresholds = [1.8, 2.0, 2.2, 2.5]
    gates = ["150rt", "200rt", "8gm", "100rt_8gm"]
    gate_labels = ["150 routes", "200 routes", "8 games", "100rt + 8gm"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("YPRR Breakout Grid Search", fontsize=14, fontweight="bold")

    for ax_i, (metric, title, cmap, vmin, vmax) in enumerate([
        ("sp_imp", "Spearman (imputed)", "RdYlGn_r", -0.31, -0.23),
        ("auc", "AUC (imputed)", "RdYlGn", 0.63, 0.69),
        ("residual", "|Residual| (model feats)", "RdYlGn", 0.02, 0.10),
    ]):
        mat = np.full((len(yprr_thresholds), len(gates)), np.nan)
        for i, thresh in enumerate(yprr_thresholds):
            for j, gate in enumerate(gates):
                config = f"yprr{thresh}_{gate}"
                row = grid_eval[grid_eval["config"] == config]
                if len(row) > 0:
                    val = row.iloc[0][metric]
                    if metric == "residual":
                        val = abs(val) if pd.notna(val) else np.nan
                    mat[i, j] = val

        ax = axes[ax_i]
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(gates)))
        ax.set_xticklabels(gate_labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(yprr_thresholds)))
        ax.set_yticklabels([f"{t:.1f}" for t in yprr_thresholds], fontsize=10)
        ax.set_ylabel("YPRR threshold", fontsize=11)
        ax.set_title(title, fontsize=11)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if pd.notna(val):
                    fmt = f"{val:.3f}" if metric != "auc" else f"{val:.3f}"
                    color = "white" if (metric == "residual" and val > 0.08) or (metric == "sp_imp" and val < -0.28) else "black"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=9, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    dynasty = load_data()

    print("Generating Figure 5: Leak test bars...")
    f5 = fig5_leak_bars(dynasty)
    f5.savefig(os.path.join(DATA_DIR, "breakout_fig5_leak_bars.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig5_leak_bars.png")

    print("Generating Figure 6: Bootstrap distributions...")
    f6 = fig6_bootstrap(dynasty)
    f6.savefig(os.path.join(DATA_DIR, "breakout_fig6_bootstrap.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig6_bootstrap.png")

    print("Generating Figure 7: YPRR grid heatmap...")
    f7 = fig7_yprr_grid(dynasty)
    f7.savefig(os.path.join(DATA_DIR, "breakout_fig7_yprr_grid.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig7_yprr_grid.png")

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
