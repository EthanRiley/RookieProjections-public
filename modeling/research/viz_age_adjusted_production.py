#!/usr/bin/env python3
"""
Visualize age-adjusted production metric results.

Reads: wr_data/age_adjusted_production_results.csv
Outputs:
  - wr_data/age_adj_heatmaps.png (3 agg windows side by side)
  - wr_data/age_adj_sensitivity.png (senior + freshman sensitivity)
  - wr_data/age_adj_improvement.png (delta bars, scheme comparison, agg comparison)
  - wr_data/age_adj_auc_scatter.png (AUC Elite vs Stud)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")

SCHEME_COLORS = {
    "none": "#9E9E9E",
    "senior": "#F44336",
    "freshman": "#2196F3",
    "both": "#9C27B0",
    "empirical": "#FF9800",
}
SCHEME_LABELS = {
    "none": "Baseline",
    "senior": "Senior Discount",
    "freshman": "Freshman Boost",
    "both": "Both",
    "empirical": "Empirical",
}
METRIC_LABELS = {"yprr": "YPRR", "yptpa": "YPTPA", "ypg": "YPG", "total_yards": "Total Yards"}
METRICS = list(METRIC_LABELS.keys())
AGG_COLORS = {"best2": "#2196F3", "best1": "#FF9800", "career": "#4CAF50"}


def save(fig, name):
    path = os.path.join(DATA_DIR, "charts", f"age_adj_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def chart_heatmaps(df):
    """3 heatmaps side by side: best Spearman per metric x scheme, one per agg window."""
    schemes = ["none", "senior", "freshman", "both", "empirical"]
    aggs = [("best2", "Best 2 Seasons"), ("best1", "Best Single Season"), ("career", "Career")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Best Spearman by Metric x Scheme", fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, (agg, agg_label) in enumerate(aggs):
        ax = axes[ax_idx]
        heat = np.full((len(METRICS), len(schemes)), np.nan)

        for i, m in enumerate(METRICS):
            for j, s in enumerate(schemes):
                sub = df[(df["metric"] == m) & (df["scheme"] == s) & (df["agg"] == agg)]
                if len(sub) > 0:
                    heat[i, j] = sub["spearman"].max()

        im = ax.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0.15, vmax=0.35)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels([SCHEME_LABELS[s] for s in schemes], fontsize=8.5, rotation=20, ha="right")
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
        ax.set_title(agg_label, fontsize=11, fontweight="bold")

        for i in range(len(METRICS)):
            for j in range(len(schemes)):
                val = heat[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.28 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9.5, color=color, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Spearman")
    save(fig, "heatmaps")


def chart_sensitivity(df):
    """Senior discount and freshman boost sensitivity side by side (YPRR best2)."""
    yprr = df[(df["metric"] == "yprr") & (df["agg"] == "best2")]
    baseline = yprr[yprr["scheme"] == "none"]["spearman"].values[0]

    fig, (ax_sr, ax_fr) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("YPRR Best2: Parameter Sensitivity", fontsize=14, fontweight="bold", y=1.02)

    # Senior
    for age_thresh in [21.5, 22.0]:
        sub = yprr[
            (yprr["scheme"] == "senior") &
            (yprr["params"].str.contains(f"age>={age_thresh}"))
        ]
        discounts, spears = [], []
        for _, row in sub.iterrows():
            d = float(row["params"].split("-")[1].replace("%", ""))
            discounts.append(d)
            spears.append(row["spearman"])
        order = np.argsort(discounts)
        ax_sr.plot(np.array(discounts)[order], np.array(spears)[order], "o-",
                   label=f"age >= {age_thresh}", markersize=8, linewidth=2)

    ax_sr.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7, label="No adjustment")
    ax_sr.set_xlabel("Senior Discount (%)", fontsize=11)
    ax_sr.set_ylabel("Spearman", fontsize=11)
    ax_sr.set_title("Senior Discount", fontsize=12, fontweight="bold")
    ax_sr.legend(fontsize=9)
    ax_sr.grid(alpha=0.3)

    # Freshman
    for age_thresh in [19.0, 19.5, 20.0]:
        sub = yprr[
            (yprr["scheme"] == "freshman") &
            (yprr["params"].str.contains(f"age<={age_thresh},"))
        ]
        boosts, spears = [], []
        for _, row in sub.iterrows():
            b = float(row["params"].split("+")[1].replace("%", ""))
            boosts.append(b)
            spears.append(row["spearman"])
        order = np.argsort(boosts)
        ax_fr.plot(np.array(boosts)[order], np.array(spears)[order], "o-",
                   label=f"age <= {age_thresh}", markersize=8, linewidth=2)

    ax_fr.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7, label="No adjustment")
    ax_fr.set_xlabel("Freshman Boost (%)", fontsize=11)
    ax_fr.set_ylabel("Spearman", fontsize=11)
    ax_fr.set_title("Freshman Boost", fontsize=12, fontweight="bold")
    ax_fr.legend(fontsize=9)
    ax_fr.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "sensitivity")


def chart_improvement(df):
    """3 panels: delta bars, scheme comparison, agg window comparison."""
    fig, (ax_delta, ax_scheme, ax_agg) = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle("Age Adjustment Impact on Production Metrics",
                 fontsize=14, fontweight="bold", y=1.02)

    # Panel 1: Delta bars
    aggs = ["best2", "best1", "career"]
    x = np.arange(len(METRICS))
    width = 0.25

    for i, agg in enumerate(aggs):
        deltas = []
        for m in METRICS:
            sub = df[(df["metric"] == m) & (df["agg"] == agg)]
            baseline = sub[sub["scheme"] == "none"]["spearman"].values[0]
            deltas.append(sub["spearman"].max() - baseline)
        ax_delta.bar(x + i * width, deltas, width, label=agg,
                     color=AGG_COLORS[agg], alpha=0.85)

    ax_delta.set_xticks(x + width)
    ax_delta.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax_delta.set_ylabel("Spearman Improvement", fontsize=10)
    ax_delta.set_title("Lift Over Baseline", fontsize=12, fontweight="bold")
    ax_delta.legend(fontsize=9)
    ax_delta.axhline(y=0, color="black", linewidth=0.5)
    ax_delta.grid(axis="y", alpha=0.3)

    # Panel 2: Scheme comparison (best2)
    schemes = ["none", "senior", "freshman", "both", "empirical"]
    sw = 0.15
    for j, s in enumerate(schemes):
        vals = []
        for m in METRICS:
            sub = df[(df["metric"] == m) & (df["scheme"] == s) & (df["agg"] == "best2")]
            vals.append(sub["spearman"].max() if len(sub) > 0 else 0)
        ax_scheme.bar(x + j * sw, vals, sw, label=SCHEME_LABELS[s],
                      color=SCHEME_COLORS[s], alpha=0.85)

    ax_scheme.set_xticks(x + 2 * sw)
    ax_scheme.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax_scheme.set_ylabel("Best Spearman", fontsize=10)
    ax_scheme.set_title("Scheme Comparison (Best2)", fontsize=12, fontweight="bold")
    ax_scheme.legend(fontsize=7.5, loc="upper right")
    ax_scheme.grid(axis="y", alpha=0.3)

    # Panel 3: Agg window comparison (both scheme)
    for i, agg in enumerate(aggs):
        spears = []
        for m in METRICS:
            sub = df[(df["metric"] == m) & (df["agg"] == agg) & (df["scheme"] == "both")]
            spears.append(sub["spearman"].max() if len(sub) > 0 else 0)
        ax_agg.bar(x + i * width, spears, width, label=agg,
                   color=AGG_COLORS[agg], alpha=0.85)

    for i, m in enumerate(METRICS):
        for j, agg in enumerate(aggs):
            sub = df[(df["metric"] == m) & (df["agg"] == agg) & (df["scheme"] == "none")]
            if len(sub) > 0:
                ax_agg.plot(i + j * width, sub["spearman"].values[0], "_",
                            color="black", markersize=14, markeredgewidth=2.5)

    ax_agg.set_xticks(x + width)
    ax_agg.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax_agg.set_ylabel("Spearman", fontsize=10)
    ax_agg.set_title("'Both' by Agg Window\n(black = baseline)", fontsize=12, fontweight="bold")
    ax_agg.legend(fontsize=9)
    ax_agg.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "improvement")


def chart_auc_scatter(df):
    """AUC(Elite) vs AUC(Stud) scatter, colored by scheme, best2 only."""
    best2 = df[df["agg"] == "best2"].copy()
    metric_markers = {"yprr": "o", "yptpa": "s", "ypg": "^", "total_yards": "D"}

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for scheme in ["none", "senior", "freshman", "both", "empirical"]:
        sub = best2[best2["scheme"] == scheme]
        for metric, marker in metric_markers.items():
            msub = sub[sub["metric"] == metric]
            ax.scatter(msub["auc_elite"], msub["auc_stud"],
                       c=SCHEME_COLORS[scheme], marker=marker, s=30, alpha=0.45,
                       edgecolors="none")
        ax.scatter([], [], c=SCHEME_COLORS[scheme], marker="o", s=60,
                   label=SCHEME_LABELS[scheme])

    for metric, marker in metric_markers.items():
        ax.scatter([], [], c="black", marker=marker, s=60, label=METRIC_LABELS[metric])

    for metric, marker in metric_markers.items():
        msub = best2[best2["metric"] == metric]
        best_row = msub.loc[msub["spearman"].idxmax()]
        ax.scatter(best_row["auc_elite"], best_row["auc_stud"],
                   c=SCHEME_COLORS[best_row["scheme"]], marker=marker,
                   s=160, edgecolors="black", linewidth=1.5, zorder=5)

    ax.set_xlabel("AUC (>=Elite)", fontsize=10)
    ax.set_ylabel("AUC (>=Stud)", fontsize=10)
    ax.set_title("AUC Trade-off: Elite vs Stud (Best 2 Seasons)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "auc_scatter")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "outputs", "age_adjusted_production_results.csv"))
    print("Generating charts...")

    chart_heatmaps(df)
    chart_sensitivity(df)
    chart_improvement(df)
    chart_auc_scatter(df)

    print("Done.")


if __name__ == "__main__":
    main()
