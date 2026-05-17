#!/usr/bin/env python3
"""Quick viz: young_yprr variant vs best1_yprr_both vs baseline."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "yprr_substitution_results.csv"))

    variants = ["baseline_best2_yprr", "best1_yprr_both", "young_yprr"]
    labels = ["Baseline\n(best2 YPRR)", "Best1 YPRR\n(both adj)", "Young YPRR\n(FR/SO, -15% JR)"]
    colors = ["#9E9E9E", "#2196F3", "#FF9800"]

    sub = df[df["variant"].isin(variants)].set_index("variant").loc[variants]

    metrics = [
        ("ens_logloss", "LogLoss (lower = better)", True),
        ("ens_brier", "Brier (lower = better)", True),
        ("ens_auc_>=Elite", "AUC (>=Elite)", False),
        ("ens_auc_>=Stud", "AUC (>=Stud)", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Young YPRR (best FR/SO, fallback -15% JR) vs Age-Adjusted Approaches",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (col, title, lower_better) in zip(axes, metrics):
        vals = sub[col].values
        bars = ax.bar(range(len(variants)), vals, color=colors, alpha=0.85, width=0.6)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        # Tight y-axis
        vmin, vmax = vals.min(), vals.max()
        margin = (vmax - vmin) * 0.4
        ax.set_ylim(vmin - margin, vmax + margin * 1.5)

        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Add sample size annotation
    for i, var in enumerate(variants):
        row = sub.loc[var]
        axes[0].text(i, axes[0].get_ylim()[0] + 0.01,
                     f"n={int(row['n_holdout'])}", ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    path = os.path.join(DATA_DIR, "young_yprr_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
