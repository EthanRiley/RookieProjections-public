#!/usr/bin/env python3
"""
Visualizations for breakout magnitude feature evaluation.

Figures:
  8. Magnitude standalone metrics comparison
  9. Feature set total residual comparison
  10. Scatter: breakout YPTPA vs breakout YPRR colored by tier
  11. Violin: magnitude distributions by tier
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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
    mags = pd.read_csv(os.path.join(DATA_DIR, "breakout_magnitudes_by_player.csv"))
    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    # Merge
    mag_cols = [c for c in mags.columns if c not in ["name", "draft_year", "computed_tier", "tier_ordinal"]]
    merged = dynasty.merge(mags[["name", "draft_year"] + mag_cols], on=["name", "draft_year"], how="left",
                           suffixes=("", "_mag"))
    merged["tier_ordinal"] = merged["computed_tier"].map(TIER_ORDER)
    merged = merged.dropna(subset=["tier_ordinal"]).copy()
    merged["tier_ordinal"] = merged["tier_ordinal"].astype(int)
    return merged


def fig8_magnitude_metrics():
    """Bar chart: standalone metrics for each magnitude feature."""
    features = {
        "YPTPA\nat YPTPA\nbreakout": {"sp": 0.190, "auc": 0.585, "resid": -0.056, "drift": 0.057},
        "YPRR\nat YPTPA\nbreakout": {"sp": 0.201, "auc": 0.600, "resid": -0.024, "drift": 0.112},
        "YPTPA\nat YPRR\nbreakout": {"sp": 0.210, "auc": 0.590, "resid": -0.039, "drift": 0.090},
        "YPRR\nat YPRR\nbreakout": {"sp": 0.237, "auc": 0.625, "resid": 0.029, "drift": 0.023},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Breakout Magnitude Features: Standalone Evaluation", fontsize=14, fontweight="bold")

    labels = list(features.keys())
    x = np.arange(len(labels))
    colors = ["#2ca02c", "#1f77b4", "#2ca02c", "#1f77b4"]  # green=YPTPA, blue=YPRR magnitude
    hatches = ["", "", "//", "//"]  # solid=YPTPA breakout, hatched=YPRR breakout

    for ax_i, (metric, title) in enumerate([
        ("sp", "Spearman"),
        ("auc", "AUC"),
        ("resid", "Residual (after model feats)"),
    ]):
        vals = [features[k][metric] for k in labels]
        if metric == "resid":
            vals_abs = [abs(v) for v in vals]
            bars = ax_i_bars = axes[ax_i].bar(x, vals_abs, color=colors, edgecolor="black", linewidth=0.5)
            for bar, h in zip(bars, hatches):
                bar.set_hatch(h)
            # Annotate with actual sign
            for xi, v in zip(x, vals):
                axes[ax_i].text(xi, abs(v) + 0.002, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
        else:
            bars = axes[ax_i].bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
            for bar, h in zip(bars, hatches):
                bar.set_hatch(h)
            for xi, v in zip(x, vals):
                axes[ax_i].text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(labels, fontsize=9)
        axes[ax_i].set_title(title, fontsize=11)
        axes[ax_i].grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="YPTPA magnitude"),
        Patch(facecolor="#1f77b4", label="YPRR magnitude"),
        Patch(facecolor="white", edgecolor="black", label="At YPTPA breakout"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="At YPRR breakout"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8, loc="upper left")

    plt.tight_layout()
    return fig


def fig9_feature_sets():
    """Bar chart: total residual signal for proposed feature sets."""
    sets = {
        "Current\n(ba_yptpa)": 0.788,
        "ba_yprr\nonly": 0.801,
        "ba_yprr\n+ mag_yptpa": 0.819,
        "ba_yprr\n+ mag_yprr": 0.832,
        "ba_yprr\n+ both_mag": 0.911,
        "ba_yptpa\n+ mag_yptpa": 0.816,
        "ba_yptpa\n+ mag_yprr": 0.804,
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(sets.keys())
    vals = list(sets.values())
    x = np.arange(len(labels))

    colors = ["#7f7f7f", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c"]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)

    # Highlight best
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(vals[0], color="gray", linestyle="--", alpha=0.5, label=f"Current: {vals[0]:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Sum |residual Spearman| across all features", fontsize=11)
    ax.set_title("Feature Set Total Predictive Signal\n(higher = more unique information across all features)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#7f7f7f", label="Current model"),
        Patch(facecolor="#1f77b4", label="YPRR breakout sets"),
        Patch(facecolor="#2ca02c", label="YPTPA breakout sets"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    return fig


def fig10_scatter(df):
    """Scatter: breakout YPTPA vs breakout YPRR at the YPRR breakout season."""
    both = df[["mag_yptpa_at_yprr", "mag_yprr_at_yprr", "computed_tier", "name"]].dropna()

    fig, ax = plt.subplots(figsize=(9, 7))
    for tier_name in sorted(TIER_ORDER.keys(), key=lambda t: TIER_ORDER[t]):
        mask = both["computed_tier"] == tier_name
        sub = both[mask]
        ax.scatter(sub["mag_yprr_at_yprr"], sub["mag_yptpa_at_yprr"],
                   c=TIER_COLORS[tier_name], label=tier_name, alpha=0.7, s=50,
                   edgecolors="white", linewidth=0.5)

    sp, _ = spearmanr(both["mag_yprr_at_yprr"], both["mag_yptpa_at_yprr"])
    ax.set_xlabel("YPRR at YPRR breakout", fontsize=12)
    ax.set_ylabel("YPTPA at YPRR breakout", fontsize=12)
    ax.set_title(f"Breakout Season: YPRR vs YPTPA (Spearman = {sp:+.3f})\n"
                 f"These capture different signals — correlation is moderate",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def fig11_violins(df):
    """Violin plots: magnitude by tier for the 4 magnitude features."""
    mag_cols = {
        "mag_yptpa_at_yptpa": "YPTPA mag\n(at YPTPA breakout)",
        "mag_yprr_at_yptpa": "YPRR mag\n(at YPTPA breakout)",
        "mag_yptpa_at_yprr": "YPTPA mag\n(at YPRR breakout)",
        "mag_yprr_at_yprr": "YPRR mag\n(at YPRR breakout)",
    }

    tier_names = sorted(TIER_ORDER.keys(), key=lambda t: TIER_ORDER[t])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Breakout Magnitude Distribution by Tier", fontsize=14, fontweight="bold", y=1.02)

    for ax_i, (col, title) in enumerate(mag_cols.items()):
        ax = axes[ax_i]
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

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(tier_names)))
        ax.set_xticklabels([t[:4] for t in tier_names], fontsize=8, rotation=45)
        ax.grid(axis="y", alpha=0.3)
        if ax_i == 0:
            ax.set_ylabel("Magnitude value", fontsize=11)

    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    df = load_data()

    print("Generating Figure 8: Magnitude standalone metrics...")
    f8 = fig8_magnitude_metrics()
    f8.savefig(os.path.join(DATA_DIR, "breakout_fig8_magnitude_metrics.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig8_magnitude_metrics.png")

    print("Generating Figure 9: Feature set comparison...")
    f9 = fig9_feature_sets()
    f9.savefig(os.path.join(DATA_DIR, "breakout_fig9_feature_sets.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig9_feature_sets.png")

    print("Generating Figure 10: YPTPA vs YPRR scatter...")
    f10 = fig10_scatter(df)
    f10.savefig(os.path.join(DATA_DIR, "breakout_fig10_scatter.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig10_scatter.png")

    print("Generating Figure 11: Magnitude violins by tier...")
    f11 = fig11_violins(df)
    f11.savefig(os.path.join(DATA_DIR, "breakout_fig11_violins.png"), dpi=150, bbox_inches="tight")
    print("  Saved breakout_fig11_violins.png")

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
