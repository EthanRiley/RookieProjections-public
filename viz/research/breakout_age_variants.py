#!/usr/bin/env python3
"""
Visualizations for breakout age variant comparison.

Generates 4 figures:
  1. Bar chart: Spearman / AUC / Residual across all variants
  2. Efficiency leak heatmap: residual under progressive magnitude controls
  3. Violin plots: tier distributions for top variants
  4. Scatter: qa_zscore_adj vs ba_yptpa colored by tier outcome
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_COLORS = {
    "Bust": "#d62728", "Flex": "#ff7f0e", "Starter": "#bcbd22",
    "Elite": "#2ca02c", "Stud": "#1f77b4", "League-Winner": "#9467bd",
}


def load_data():
    """Load dynasty data with pre-computed breakout age variants."""
    player_path = os.path.join(DATA_DIR, "breakout_age_variants_by_player.csv")
    dynasty_path = os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv")

    players = pd.read_csv(player_path)
    dynasty = pd.read_csv(dynasty_path)

    # Merge the breakout variants onto dynasty (which has the model features)
    variant_cols = [c for c in players.columns if c.startswith("ba_") or c.startswith("qa_") or c.startswith("qy_") or c == "draft_age_feat"]
    merged = dynasty.merge(
        players[["name", "draft_year"] + variant_cols],
        on=["name", "draft_year"], how="left",
    )
    merged["tier_ordinal"] = merged["computed_tier"].map(TIER_ORDER)
    merged = merged.dropna(subset=["tier_ordinal"]).copy()
    merged["tier_ordinal"] = merged["tier_ordinal"].astype(int)
    merged["hit"] = (merged["tier_ordinal"] >= 3).astype(int)
    return merged


def compute_metrics(df, col):
    """Compute Spearman, AUC, and residual for a variant."""
    model_feats = [
        "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
        "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
    ]
    # Impute
    mx = df[col].max()
    imp = df[col].fillna(round(mx + 1, 2) if pd.notna(mx) else 25.0)

    valid = df[["tier_ordinal", "hit"]].copy()
    valid["x"] = imp
    valid = valid.dropna()

    sp, _ = spearmanr(valid["x"], valid["tier_ordinal"])
    try:
        auc = roc_auc_score(valid["hit"], valid["x"])
        if auc < 0.5:
            auc = 1 - auc
    except ValueError:
        auc = np.nan

    # Residual
    sub = df[model_feats + ["tier_ordinal"]].copy()
    sub["x"] = imp
    sub = sub.dropna()
    if len(sub) >= 30:
        rank_feat = rankdata(sub["x"].values)
        rank_tier = rankdata(sub["tier_ordinal"].values)
        ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in model_feats])
        X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
        z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
        resid = rank_feat - X @ z
        sp_resid, _ = spearmanr(resid, rank_tier)
    else:
        sp_resid = np.nan

    return sp, auc, sp_resid


def fig1_metric_comparison(df, variants):
    """Bar chart comparing Spearman, AUC, and residual across variants."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Breakout Age Variants: Metric Comparison", fontsize=14, fontweight="bold")

    metrics = {"Spearman (imputed)": [], "AUC (imputed)": [], "Residual (model feats)": []}
    for col in variants:
        sp, auc, resid = compute_metrics(df, col)
        metrics["Spearman (imputed)"].append(abs(sp))
        metrics["AUC (imputed)"].append(auc)
        metrics["Residual (model feats)"].append(abs(resid) if pd.notna(resid) else 0)

    labels = [v.replace("ba_", "").replace("qa_", "QA:").replace("qy_", "QY:").replace("_feat", "") for v in variants]
    x = np.arange(len(variants))

    # Color by category
    colors = []
    for v in variants:
        if v.startswith("qa_"):
            colors.append("#e377c2")
        elif v.startswith("qy_"):
            colors.append("#17becf")
        elif v == "draft_age_feat":
            colors.append("#7f7f7f")
        else:
            colors.append("#1f77b4")

    for i, (title, vals) in enumerate(metrics.items()):
        axes[i].bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        axes[i].set_ylabel(title, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)
        # Highlight best
        best_idx = np.argmax(vals)
        axes[i].bar(best_idx, vals[best_idx], color=colors[best_idx], edgecolor="black", linewidth=2)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Binary threshold"),
        Patch(facecolor="#e377c2", label="YPTPA quality-adjusted"),
        Patch(facecolor="#17becf", label="YPRR quality-adjusted"),
        Patch(facecolor="#7f7f7f", label="Draft age"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def fig2_efficiency_leak(df, focus_variants):
    """Heatmap: residual signal under progressive magnitude controls."""
    model_feats = [
        "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
        "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
    ]

    # Impute magnitudes
    for mag_col in ["qa_magnitude", "qy_magnitude"]:
        mx = df[mag_col].max()
        df[f"{mag_col}_imp"] = df[mag_col].fillna(round(mx + 1, 2) if pd.notna(mx) else 0.0)

    tests = {
        "Model feats\nonly": model_feats,
        "+ YPTPA\nmagnitude": model_feats + ["qa_magnitude_imp"],
        "+ YPRR\nmagnitude": model_feats + ["qy_magnitude_imp"],
        "+ Both\nmagnitudes": model_feats + ["qa_magnitude_imp", "qy_magnitude_imp"],
    }

    results = np.full((len(focus_variants), len(tests)), np.nan)
    for i, col in enumerate(focus_variants):
        mx = df[col].max()
        imp = df[col].fillna(round(mx + 1, 2) if pd.notna(mx) else 25.0)
        for j, (_, ctrl_feats) in enumerate(tests.items()):
            ctrl = list(dict.fromkeys(ctrl_feats))
            sub = df[ctrl + ["tier_ordinal"]].copy()
            sub["x"] = imp
            sub = sub.dropna()
            if len(sub) < 30:
                continue
            rank_feat = rankdata(sub["x"].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid = rank_feat - X @ z
            sp_resid, _ = spearmanr(resid, rank_tier)
            results[i, j] = sp_resid

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Efficiency Leak Analysis: Residual Signal Under Progressive Controls",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(results, cmap="RdYlGn_r", aspect="auto", vmin=-0.18, vmax=0.1)
    ax.set_xticks(np.arange(len(tests)))
    ax.set_xticklabels(list(tests.keys()), fontsize=10)
    labels = [v.replace("ba_", "").replace("qa_", "QA:").replace("qy_", "QY:").replace("_feat", "") for v in focus_variants]
    ax.set_yticks(np.arange(len(focus_variants)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            val = results[i, j]
            if pd.notna(val):
                color = "white" if abs(val) > 0.12 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=10,
                        fontweight="bold" if abs(val) > 0.12 else "normal", color=color)

    plt.colorbar(im, ax=ax, label="Residual Spearman", shrink=0.8)
    ax.set_xlabel("Control variables", fontsize=11)
    ax.set_ylabel("Breakout age variant", fontsize=11)
    plt.tight_layout()
    return fig


def fig3_tier_violins(df, top_variants):
    """Violin plots showing tier distributions for top variants."""
    n_vars = len(top_variants)
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 6), sharey=False)
    fig.suptitle("Breakout Age Distribution by Tier (Top Variants)", fontsize=14, fontweight="bold", y=1.02)

    tier_names = sorted(TIER_ORDER.keys(), key=lambda t: TIER_ORDER[t])

    for ax_i, col in enumerate(top_variants):
        ax = axes[ax_i] if n_vars > 1 else axes
        data_by_tier = []
        positions = []
        colors_list = []
        for tier_name in tier_names:
            tier_val = TIER_ORDER[tier_name]
            sub = df[df["tier_ordinal"] == tier_val][col].dropna()
            if len(sub) >= 3:
                data_by_tier.append(sub.values)
                positions.append(tier_val)
                colors_list.append(TIER_COLORS[tier_name])

        if data_by_tier:
            parts = ax.violinplot(data_by_tier, positions=positions, showmedians=True, showextrema=False)
            for pc, color in zip(parts["bodies"], colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")

        label = col.replace("ba_", "").replace("qa_", "QA:").replace("qy_", "QY:").replace("_feat", "")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(tier_names)))
        ax.set_xticklabels([t[:4] for t in tier_names], fontsize=8, rotation=45)
        ax.grid(axis="y", alpha=0.3)
        if ax_i == 0:
            ax.set_ylabel("Age / Adjusted Age", fontsize=11)

    plt.tight_layout()
    return fig


def fig4_zscore_vs_binary(df):
    """Scatter: qa_zscore_adj vs ba_yptpa, colored by tier, showing where they diverge."""
    both = df[["ba_yptpa", "qa_zscore_adj", "computed_tier", "tier_ordinal", "name"]].dropna().copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Quality Adjustment Effect: Z-Score Adjusted vs Binary YPTPA Breakout",
                 fontsize=13, fontweight="bold")

    # Left: scatter
    ax = axes[0]
    for tier_name in sorted(TIER_ORDER.keys(), key=lambda t: TIER_ORDER[t]):
        mask = both["computed_tier"] == tier_name
        sub = both[mask]
        ax.scatter(sub["ba_yptpa"], sub["qa_zscore_adj"],
                   c=TIER_COLORS[tier_name], label=tier_name, alpha=0.7, s=40, edgecolors="white", linewidth=0.5)

    # Identity-ish line (qa_zscore = ba_yptpa when magnitude = pop_mean)
    lims = [both["ba_yptpa"].min() - 0.5, both["ba_yptpa"].max() + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No adjustment")
    ax.set_xlabel("ba_yptpa (binary)", fontsize=11)
    ax.set_ylabel("qa_zscore_adj (quality-adjusted)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Points below the line had dominant breakouts", fontsize=10)

    # Right: histogram of the adjustment magnitude
    ax2 = axes[1]
    adjustment = both["ba_yptpa"] - both["qa_zscore_adj"]
    ax2.hist(adjustment, bins=25, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax2.axvline(adjustment.median(), color="red", linestyle="--", label=f"Median: {adjustment.median():.2f}y")
    ax2.set_xlabel("Age discount (ba_yptpa - qa_zscore_adj)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of quality adjustment", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    df = load_data()

    all_variants = [
        "ba_650yards", "ba_45ypg", "ba_yptpa", "ba_45ypg_yprr",
        "ba_yprr_routes", "ba_dominator", "ba_yptpa_yprr", "ba_composite",
        "qa_ratio_scaled", "qa_zscore_adj", "qa_log_magnitude", "qa_magnitude",
        "qy_ratio_scaled", "qy_zscore_adj", "qy_log_magnitude", "qy_magnitude",
        "draft_age_feat",
    ]

    focus_leak = [
        "ba_yptpa", "ba_yprr_routes", "ba_45ypg",
        "qa_zscore_adj", "qa_ratio_scaled", "qa_log_magnitude",
        "qy_zscore_adj", "qy_ratio_scaled", "qy_log_magnitude",
        "draft_age_feat",
    ]

    top_variants = ["ba_yptpa", "qa_zscore_adj", "ba_45ypg", "qy_zscore_adj", "draft_age_feat"]

    print("Generating Figure 1: Metric comparison...")
    f1 = fig1_metric_comparison(df, all_variants)
    f1.savefig(os.path.join(DATA_DIR, "breakout_fig1_metrics.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig1_metrics.png")

    print("Generating Figure 2: Efficiency leak heatmap...")
    f2 = fig2_efficiency_leak(df, focus_leak)
    f2.savefig(os.path.join(DATA_DIR, "breakout_fig2_leak.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig2_leak.png")

    print("Generating Figure 3: Tier violins...")
    f3 = fig3_tier_violins(df, top_variants)
    f3.savefig(os.path.join(DATA_DIR, "breakout_fig3_violins.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig3_violins.png")

    print("Generating Figure 4: Z-score vs binary scatter...")
    f4 = fig4_zscore_vs_binary(df)
    f4.savefig(os.path.join(DATA_DIR, "breakout_fig4_zscore_vs_binary.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig4_zscore_vs_binary.png")

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
