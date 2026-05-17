#!/usr/bin/env python3
"""
Catch % feature engineering visualizations.

Generates 3 figures:
  1. Summary bar chart: Spearman, AUC, and era drift for all variants
  2. Scatter plots: each variant vs tier ordinal with regression lines
  3. aDOT adjustment explainer: raw catch% vs aDOT with regression line,
     plus before/after violin plots
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = ["Bust", "Flex", "Starter", "Elite", "Stud", "League-Winner"]
TIER_COLORS = {
    "Bust": "#d9534f", "Flex": "#f0ad4e", "Starter": "#5bc0de",
    "Elite": "#428bca", "Stud": "#5cb85c", "League-Winner": "#8e44ad",
}

# All variants to compare
VARIANTS = [
    ("career_caught_percent", "Raw Catch %"),
    ("best2_caught_percent", "Raw Catch % (Best 2)"),
    ("career_catch_pct_adot_adj", "aDOT-Adjusted"),
    ("best2_catch_pct_adot_adj", "aDOT-Adjusted (Best 2)"),
    ("career_catch_pct_above_team", "Above Team Comp %"),
    ("best2_catch_pct_above_team", "Above Team (Best 2)"),
    ("career_catch_pct_double_adj", "Double-Adjusted"),
]


def load_data():
    from aggregation.aggregate_college_stats import load_all_grades, normalize_name, get_player_seasons
    from features.engineer_catch_pct import compute_engineered_features, load_season_data

    all_grades, team_comp_lookup = load_season_data()

    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    dynasty["tier_ordinal"] = dynasty["computed_tier"].map(TIER_ORDER)
    dynasty = dynasty.dropna(subset=["tier_ordinal"]).copy()
    dynasty["tier_ordinal"] = dynasty["tier_ordinal"].astype(int)
    dynasty["hit"] = (dynasty["tier_ordinal"] >= 3).astype(int)

    eng_df = compute_engineered_features(all_grades, team_comp_lookup, dynasty)
    dynasty = pd.concat([dynasty.reset_index(drop=True), eng_df], axis=1)

    return dynasty, all_grades


def compute_metrics(df, col):
    sub = df[[col, "tier_ordinal", "hit", "draft_year"]].dropna()
    if len(sub) < 30:
        return None
    x, y, y_hit = sub[col].values, sub["tier_ordinal"].values, sub["hit"].values
    sp, _ = spearmanr(x, y)
    auc = roc_auc_score(y_hit, x)
    if auc < 0.5:
        auc = 1 - auc
    years = sub["draft_year"].values
    early, late = years <= 2019, years >= 2020
    sp_e, _ = spearmanr(x[early], y[early]) if early.sum() > 10 else (np.nan, np.nan)
    sp_l, _ = spearmanr(x[late], y[late]) if late.sum() > 10 else (np.nan, np.nan)
    drift = abs(sp_e - sp_l) if pd.notna(sp_e) and pd.notna(sp_l) else np.nan
    return {"spearman": sp, "auc": auc, "drift": drift, "n": len(sub)}


# =====================================================================
# Load
# =====================================================================
print("Loading data...")
dynasty, all_grades = load_data()

metrics = {}
for col, label in VARIANTS:
    m = compute_metrics(dynasty, col)
    if m:
        metrics[label] = m

# =====================================================================
# Figure 1: Summary bar chart
# =====================================================================
print("Generating summary bars...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

labels = list(metrics.keys())
x_pos = np.arange(len(labels))

# Colors: highlight aDOT-adjusted career in green, raw career in blue, rest gray
bar_colors = []
for lab in labels:
    if lab == "aDOT-Adjusted":
        bar_colors.append("#5cb85c")
    elif lab == "Raw Catch %":
        bar_colors.append("#428bca")
    else:
        bar_colors.append("#aaaaaa")

# Panel 1: Spearman
vals = [metrics[l]["spearman"] for l in labels]
axes[0].barh(x_pos, vals, color=bar_colors, edgecolor="white", height=0.6)
axes[0].set_yticks(x_pos)
axes[0].set_yticklabels(labels, fontsize=10)
axes[0].set_xlabel("Spearman Correlation with Tier", fontsize=11)
axes[0].set_title("Predictive Signal", fontsize=13, fontweight="bold")
axes[0].axvline(x=0, color="black", linewidth=0.5)
for i, v in enumerate(vals):
    axes[0].text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=9)

# Panel 2: AUC
vals = [metrics[l]["auc"] for l in labels]
axes[1].barh(x_pos, vals, color=bar_colors, edgecolor="white", height=0.6)
axes[1].set_yticks(x_pos)
axes[1].set_yticklabels(labels, fontsize=10)
axes[1].set_xlabel("AUC (Elite+ Classification)", fontsize=11)
axes[1].set_title("Classification Power", fontsize=13, fontweight="bold")
axes[1].set_xlim(0.5, 0.8)
for i, v in enumerate(vals):
    axes[1].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

# Panel 3: Era drift (lower = better)
vals = [metrics[l]["drift"] for l in labels]
axes[2].barh(x_pos, vals, color=bar_colors, edgecolor="white", height=0.6)
axes[2].set_yticks(x_pos)
axes[2].set_yticklabels(labels, fontsize=10)
axes[2].set_xlabel("Era Drift (lower = more stable)", fontsize=11)
axes[2].set_title("Era Stability", fontsize=13, fontweight="bold")
for i, v in enumerate(vals):
    axes[2].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

fig.suptitle("Catch % Variants: Head-to-Head Comparison", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "catch_pct_summary.png"), dpi=150, bbox_inches="tight")
print(f"  Saved catch_pct_summary.png")

# =====================================================================
# Figure 2: aDOT adjustment explainer
# =====================================================================
print("Generating aDOT explainer...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: aDOT vs catch% scatter with regression line (season-level)
from aggregation.aggregate_college_stats import load_all_grades
all_cp = pd.to_numeric(all_grades["caught_percent"], errors="coerce")
all_adot = pd.to_numeric(all_grades["avg_depth_of_target"], errors="coerce")
mask = all_cp.notna() & all_adot.notna()

axes[0].scatter(all_adot[mask], all_cp[mask], alpha=0.08, s=8, color="#666666")
# Regression line
coef = np.polyfit(all_adot[mask].values, all_cp[mask].values, 1)
x_line = np.linspace(all_adot[mask].min(), all_adot[mask].max(), 100)
axes[0].plot(x_line, np.polyval(coef, x_line), color="#d9534f", linewidth=2.5,
             label=f"catch% = {coef[0]:.1f} * aDOT + {coef[1]:.1f}")
axes[0].set_xlabel("Avg Depth of Target (yards)", fontsize=11)
axes[0].set_ylabel("Catch %", fontsize=11)
axes[0].set_title("Deeper Routes = Lower Catch %", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=9, loc="upper right")

# Panel 2: Violin - raw catch% by tier
tier_data_raw = []
tier_labels = []
for tier_name in TIER_NAMES:
    tier_val = TIER_ORDER[tier_name]
    sub = dynasty[dynasty["tier_ordinal"] == tier_val]["career_caught_percent"].dropna()
    if len(sub) > 0:
        tier_data_raw.append(sub.values)
        tier_labels.append(tier_name)

vp = axes[1].violinplot(tier_data_raw, showmedians=True, showextrema=False)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor(TIER_COLORS[tier_labels[i]])
    body.set_alpha(0.7)
vp["cmedians"].set_color("black")
axes[1].set_xticks(range(1, len(tier_labels) + 1))
axes[1].set_xticklabels(tier_labels, rotation=45, fontsize=9)
axes[1].set_ylabel("Catch %", fontsize=11)
axes[1].set_title("Raw Catch % by Tier", fontsize=13, fontweight="bold")

# Panel 3: Violin - aDOT-adjusted catch% by tier
tier_data_adj = []
tier_labels_adj = []
for tier_name in TIER_NAMES:
    tier_val = TIER_ORDER[tier_name]
    sub = dynasty[dynasty["tier_ordinal"] == tier_val]["career_catch_pct_adot_adj"].dropna()
    if len(sub) > 0:
        tier_data_adj.append(sub.values)
        tier_labels_adj.append(tier_name)

vp2 = axes[2].violinplot(tier_data_adj, showmedians=True, showextrema=False)
for i, body in enumerate(vp2["bodies"]):
    body.set_facecolor(TIER_COLORS[tier_labels_adj[i]])
    body.set_alpha(0.7)
vp2["cmedians"].set_color("black")
axes[2].set_xticks(range(1, len(tier_labels_adj) + 1))
axes[2].set_xticklabels(tier_labels_adj, rotation=45, fontsize=9)
axes[2].set_ylabel("aDOT-Adjusted Catch %", fontsize=11)
axes[2].set_title("aDOT-Adjusted Catch % by Tier", fontsize=13, fontweight="bold")

fig.suptitle("Why aDOT Adjustment Improves Catch %", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "catch_pct_adot_explainer.png"), dpi=150, bbox_inches="tight")
print(f"  Saved catch_pct_adot_explainer.png")

# =====================================================================
# Figure 3: Team adjustment failure
# =====================================================================
print("Generating team adjustment comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

compare_pairs = [
    ("career_caught_percent", "Raw Catch %"),
    ("career_catch_pct_adot_adj", "aDOT-Adjusted"),
    ("career_catch_pct_above_team", "Above Team Comp %"),
]

for i, (col, title) in enumerate(compare_pairs):
    sub = dynasty[[col, "tier_ordinal", "computed_tier"]].dropna()

    for tier_name in TIER_NAMES:
        tier_sub = sub[sub["computed_tier"] == tier_name][col]
        if len(tier_sub) > 0:
            jitter = np.random.normal(0, 0.12, len(tier_sub))
            axes[i].scatter(
                TIER_ORDER[tier_name] + jitter, tier_sub.values,
                color=TIER_COLORS[tier_name], alpha=0.5, s=20, edgecolors="none",
            )

    # Tier means
    means = sub.groupby("tier_ordinal")[col].mean()
    axes[i].plot(means.index, means.values, color="black", linewidth=2, marker="o",
                 markersize=6, zorder=5, label="Tier mean")

    sp, _ = spearmanr(sub[col].values, sub["tier_ordinal"].values)
    axes[i].set_title(f"{title}\nSpearman: {sp:+.3f}", fontsize=12, fontweight="bold")
    axes[i].set_xticks(range(6))
    axes[i].set_xticklabels(TIER_NAMES, rotation=45, fontsize=9)
    axes[i].set_ylabel(title, fontsize=10)
    axes[i].legend(fontsize=8, loc="upper left")

fig.suptitle("Catch % Variants vs Outcome Tier", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "catch_pct_variants_scatter.png"), dpi=150, bbox_inches="tight")
print(f"  Saved catch_pct_variants_scatter.png")

print("\nDone.")
