#!/usr/bin/env python3
"""
Layer 2 feature validation: violin plots of top features by computed tier.

Generates a grid of violin plots showing distribution of each feature
conditional on outcome tier. Reveals threshold effects, bimodality,
and interaction candidates.

Reads:
  - wr_data/wr_dynasty_value_with_college.csv
Outputs:
  - wr_data/feature_violins.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = ["Bust", "Flex", "Starter", "Elite", "Stud", "League-Winner"]

# Model features (career) + best2 equivalents + top candidates
TOP_FEATURES = [
    "draft_capital",
    "career_targeted_qb_rating",
    "breakout_age",
    "career_yprr",
    "career_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "career_contested_catch_rate",
    "career_avoided_tackles_pg",
    "breakout_yptpa",
    "breakout_yprr",
    "best2_targeted_qb_rating",
    "best2_yprr",
    "career_caught_percent",
    "best2_caught_percent",
    "career_grades_pass_route",
    "best2_grades_pass_route",
]

df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df = df.dropna(subset=["computed_tier"])

# Impute breakout features
if "breakout_age" in df.columns:
    max_bo = df["breakout_age"].max()
    df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
if "breakout_yptpa" in df.columns:
    df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
if "breakout_yprr" in df.columns:
    df["breakout_yprr"] = df["breakout_yprr"].fillna(0)
df["computed_tier"] = pd.Categorical(df["computed_tier"], categories=TIER_ORDER, ordered=True)

ncols = 4
nrows = (len(TOP_FEATURES) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
axes = axes.flatten()

palette = {
    "Bust": "#d9534f",
    "Flex": "#f0ad4e",
    "Starter": "#5bc0de",
    "Elite": "#428bca",
    "Stud": "#5cb85c",
    "League-Winner": "#8e44ad",
}

for i, feat in enumerate(TOP_FEATURES):
    ax = axes[i]
    plot_df = df[["computed_tier", feat]].dropna()

    sns.violinplot(
        data=plot_df,
        x="computed_tier",
        y=feat,
        order=TIER_ORDER,
        palette=palette,
        inner="quartile",
        cut=0,
        ax=ax,
    )

    # Overlay strip plot for individual points
    sns.stripplot(
        data=plot_df,
        x="computed_tier",
        y=feat,
        order=TIER_ORDER,
        color="black",
        alpha=0.3,
        size=2,
        jitter=True,
        ax=ax,
    )

    ax.set_title(feat.replace("career_", "C: ").replace("best2_", "B2: ").replace("best_", "B: "), fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_ylabel("")

# Hide unused axes
for j in range(len(TOP_FEATURES), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Feature Distributions by Computed Tier", fontsize=14, y=1.01)
plt.tight_layout()

output_path = os.path.join(DATA_DIR, "feature_violins.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
