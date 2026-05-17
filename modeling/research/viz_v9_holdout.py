#!/usr/bin/env python3
"""Visualize v9 holdout results: version comparison + top prospect rankings."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")


def save(fig, name):
    path = os.path.join(DATA_DIR, f"v9_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def chart_version_comparison():
    """Bar chart comparing v8, best1_both, and graduated across key metrics."""
    versions = ["v8 Baseline\n(best2_yprr)", "v8.5\n(fr+15%, sr-15%)", "v9 Graduated\n(fr+25%, so+5%,\njr-20%, sr-25%)"]
    colors = ["#9E9E9E", "#42A5F5", "#2E7D32"]

    # From holdout results
    data = {
        "LogLoss": [0.799, 0.772, 0.771],
        "Brier": [0.355, 0.345, 0.343],
        "AUC (>=Elite)": [0.963, 0.958, 0.961],
        "AUC (>=Stud)": [0.888, 0.945, 0.953],
        "AUC (>=LW)": [0.908, 1.000, 1.000],
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    fig.suptitle("WR Model Version Comparison — Holdout 2022-2024 (88 players)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (metric, vals) in zip(axes, data.items()):
        lower_better = metric in ("LogLoss", "Brier")
        bars = ax.bar(range(3), vals, color=colors, alpha=0.85, width=0.55)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        vmin, vmax = min(vals), max(vals)
        margin = (vmax - vmin) * 0.5
        ax.set_ylim(vmin - margin, vmax + margin * 1.8)

        ax.set_xticks(range(3))
        ax.set_xticklabels(versions, fontsize=7.5)
        suffix = " (lower=better)" if lower_better else ""
        ax.set_title(f"{metric}{suffix}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Highlight best
        best_idx = np.argmin(vals) if lower_better else np.argmax(vals)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

    fig.tight_layout()
    save(fig, "version_comparison")


def chart_top_prospects():
    """Horizontal bar charts: top 10 prospects per class."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("v9 Top 10 Prospects by Expected Tier Value",
                 fontsize=14, fontweight="bold", y=1.02)

    year_colors = {2024: "#1565C0", 2025: "#2E7D32", 2026: "#E65100"}

    for ax, year in zip(axes, [2024, 2025, 2026]):
        df = pd.read_csv(os.path.join(DATA_DIR, "outputs", f"prospect_predictions_{year}.csv"))
        top = df.head(10).iloc[::-1]  # reverse for horizontal bar

        y = np.arange(len(top))
        bars = ax.barh(y, top["expected_tier"].values, color=year_colors[year],
                       alpha=0.8, height=0.6)

        for i, (_, row) in enumerate(top.iterrows()):
            p_elite_plus = row["P(Elite)"] + row["P(Stud)"] + row["P(League-Winner)"]
            ax.text(row["expected_tier"] + 0.03, i,
                    f"{row['expected_tier']:.2f}  ({p_elite_plus:.0%} Elite+)",
                    va="center", fontsize=8.5, fontweight="bold")

        labels = [f"{row['name']} (#{int(row['pick'])})" for _, row in top.iterrows()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Expected Tier Value", fontsize=10)
        ax.set_title(f"{year} Draft Class", fontsize=12, fontweight="bold")
        ax.set_xlim(0, top["expected_tier"].max() * 1.35)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save(fig, "top_prospects")


def chart_delta_vs_v8():
    """Show improvement trajectory from v8 -> v8.5 -> v9."""
    metrics = ["LogLoss", "Brier", "AUC(Elite)", "AUC(Stud)", "AUC(LW)"]
    v8 =   [0.799, 0.355, 0.963, 0.888, 0.908]
    v85 =  [0.772, 0.345, 0.958, 0.945, 1.000]
    v9 =   [0.771, 0.343, 0.961, 0.953, 1.000]

    # Compute deltas vs v8 (flip sign for LogLoss/Brier so positive = improvement)
    flip = [-1, -1, 1, 1, 1]
    d85 = [(b - a) * f for a, b, f in zip(v8, v85, flip)]
    d9 = [(b - a) * f for a, b, f in zip(v8, v9, flip)]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Improvement Over v8 Baseline (positive = better)",
                 fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(metrics))
    w = 0.3
    bars1 = ax.bar(x - w/2, d85, w, label="v8.5 (fr+15%, sr-15%)",
                   color="#42A5F5", alpha=0.85)
    bars2 = ax.bar(x + w/2, d9, w, label="v9 Graduated (fr+25%, so+5%, jr-20%, sr-25%)",
                   color="#2E7D32", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:+.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Delta vs v8", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "delta_vs_v8")


def main():
    print("Generating v9 charts...")
    chart_version_comparison()
    chart_top_prospects()
    chart_delta_vs_v8()
    print("Done.")


if __name__ == "__main__":
    main()
