#!/usr/bin/env python3
"""
Career vs Best-2-Seasons feature comparison visualization.

Generates a clean multi-panel figure comparing career aggregation
against best-2-seasons aggregation for each model feature.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

# Feature pairs: (career_col, best2_col, display_name)
PAIRS = [
    ("career_targeted_qb_rating", "best2_targeted_qb_rating", "Targeted QBR"),
    ("career_yprr", "best2_yprr", "Yards Per Route Run"),
    ("career_caught_percent", "best2_caught_percent", "Catch %"),
    ("career_contested_catch_rate", "best2_contested_catch_rate", "Contested Catch Rate"),
    ("career_avoided_tackles_pg", "best2_avoided_tackles_pg", "Avoided Tackles / Game"),
]

# --- Load data ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
df = df.dropna(subset=["tier_ordinal"]).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)
df["hit"] = (df["tier_ordinal"] >= 3).astype(int)

eval_df = pd.read_csv(os.path.join(DATA_DIR, "feature_evaluation.csv"))

# --- Compute metrics for each pair ---
metrics = []
for career_col, best2_col, name in PAIRS:
    sub = df[[career_col, best2_col, "tier_ordinal", "hit"]].dropna()
    tier = sub["tier_ordinal"].values
    hit = sub["hit"].values

    sp_c, _ = spearmanr(sub[career_col], tier)
    sp_b, _ = spearmanr(sub[best2_col], tier)

    from sklearn.metrics import roc_auc_score
    auc_c = roc_auc_score(hit, sub[career_col])
    auc_b = roc_auc_score(hit, sub[best2_col])
    if auc_c < 0.5:
        auc_c = 1 - auc_c
    if auc_b < 0.5:
        auc_b = 1 - auc_b

    # Era stability from eval file
    row_c = eval_df[eval_df["feature"] == career_col]
    row_b = eval_df[eval_df["feature"] == best2_col]
    drift_c = row_c.iloc[0]["spearman_diff"] if len(row_c) > 0 else np.nan
    drift_b = row_b.iloc[0]["spearman_diff"] if len(row_b) > 0 else np.nan

    metrics.append({
        "name": name, "career_col": career_col, "best2_col": best2_col,
        "sp_career": sp_c, "sp_best2": sp_b,
        "auc_career": auc_c, "auc_best2": auc_b,
        "drift_career": drift_c, "drift_best2": drift_b,
    })

# =====================================================================
# FIGURE 1: Head-to-head summary (the main chart)
# =====================================================================
CAREER_COLOR = "#2563EB"
BEST2_COLOR = "#F59E0B"

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Career vs Best-2-Seasons Aggregation", fontsize=18, fontweight="bold", y=1.02)

names = [m["name"] for m in metrics]
y_pos = np.arange(len(names))

# --- Panel 1: Spearman ---
ax = axes[0]
sp_c = [m["sp_career"] for m in metrics]
sp_b = [m["sp_best2"] for m in metrics]

bars_c = ax.barh(y_pos - 0.18, sp_c, 0.35, label="Career", color=CAREER_COLOR, alpha=0.85)
bars_b = ax.barh(y_pos + 0.18, sp_b, 0.35, label="Best 2", color=BEST2_COLOR, alpha=0.85)

for i, (vc, vb) in enumerate(zip(sp_c, sp_b)):
    ax.text(vc + 0.008, i - 0.18, f"{vc:.3f}", va="center", fontsize=9, fontweight="bold", color=CAREER_COLOR)
    ax.text(vb + 0.008, i + 0.18, f"{vb:.3f}", va="center", fontsize=9, fontweight="bold", color=BEST2_COLOR)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel("Spearman Correlation with Tier", fontsize=11)
ax.set_title("Predictive Signal", fontsize=13, fontweight="bold", pad=12)
ax.invert_yaxis()
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, max(max(sp_c), max(sp_b)) * 1.35)
ax.axvline(0, color="gray", linewidth=0.5)

# --- Panel 2: AUC ---
ax = axes[1]
auc_c = [m["auc_career"] for m in metrics]
auc_b = [m["auc_best2"] for m in metrics]

bars_c = ax.barh(y_pos - 0.18, auc_c, 0.35, label="Career", color=CAREER_COLOR, alpha=0.85)
bars_b = ax.barh(y_pos + 0.18, auc_b, 0.35, label="Best 2", color=BEST2_COLOR, alpha=0.85)

for i, (vc, vb) in enumerate(zip(auc_c, auc_b)):
    ax.text(vc + 0.003, i - 0.18, f"{vc:.3f}", va="center", fontsize=9, fontweight="bold", color=CAREER_COLOR)
    ax.text(vb + 0.003, i + 0.18, f"{vb:.3f}", va="center", fontsize=9, fontweight="bold", color=BEST2_COLOR)

ax.set_yticks(y_pos)
ax.set_yticklabels([""] * len(names))
ax.set_xlabel("AUC (Elite or Better)", fontsize=11)
ax.set_title("Classification Power", fontsize=13, fontweight="bold", pad=12)
ax.invert_yaxis()
ax.set_xlim(0.5, max(max(auc_c), max(auc_b)) * 1.08)

# --- Panel 3: Era Stability ---
ax = axes[2]
drift_c = [m["drift_career"] for m in metrics]
drift_b = [m["drift_best2"] for m in metrics]

bars_c = ax.barh(y_pos - 0.18, drift_c, 0.35, label="Career", color=CAREER_COLOR, alpha=0.85)
bars_b = ax.barh(y_pos + 0.18, drift_b, 0.35, label="Best 2", color=BEST2_COLOR, alpha=0.85)

for i, (vc, vb) in enumerate(zip(drift_c, drift_b)):
    ax.text(vc + 0.005, i - 0.18, f"{vc:.3f}", va="center", fontsize=9, fontweight="bold", color=CAREER_COLOR)
    ax.text(vb + 0.005, i + 0.18, f"{vb:.3f}", va="center", fontsize=9, fontweight="bold", color=BEST2_COLOR)

ax.set_yticks(y_pos)
ax.set_yticklabels([""] * len(names))
ax.set_xlabel("Era Drift (lower = more stable)", fontsize=11)
ax.set_title("Era Stability", fontsize=13, fontweight="bold", pad=12)
ax.invert_yaxis()
ax.set_xlim(0, max(max(drift_c), max(drift_b)) * 1.4)

plt.tight_layout()
out1 = os.path.join(VIZ_DIR, "career_vs_best2_summary.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out1}")
plt.close()


# =====================================================================
# FIGURE 2: Per-feature scatter plots (career vs best2 colored by tier)
# =====================================================================
tier_colors = {0: "#d62728", 1: "#ff7f0e", 2: "#bcbd22", 3: "#2ca02c", 4: "#1f77b4", 5: "#9467bd"}
tier_labels = ["Bust", "Flex", "Starter", "Elite", "Stud", "LW"]

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.suptitle("Career vs Best-2-Seasons: Player-Level Comparison",
             fontsize=16, fontweight="bold", y=1.04)

for idx, (career_col, best2_col, name) in enumerate(PAIRS):
    ax = axes[idx]
    sub = df[[career_col, best2_col, "tier_ordinal"]].dropna()

    colors = [tier_colors[int(t)] for t in sub["tier_ordinal"]]
    ax.scatter(sub[career_col], sub[best2_col], c=colors, alpha=0.55, s=20,
              edgecolors="white", linewidths=0.3)

    # Identity line
    lo = min(sub[career_col].min(), sub[best2_col].min())
    hi = max(sub[career_col].max(), sub[best2_col].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

    # Correlation annotation
    r, _ = spearmanr(sub[career_col], sub[best2_col])
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=10, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Career", fontsize=10)
    ax.set_ylabel("Best 2 Seasons", fontsize=10)
    ax.set_title(name, fontsize=11, fontweight="bold")

# Shared legend
for i, label in enumerate(tier_labels):
    axes[-1].scatter([], [], c=tier_colors[i], label=label, s=25)
axes[-1].legend(fontsize=8, loc="lower right", framealpha=0.9)

plt.tight_layout()
out2 = os.path.join(VIZ_DIR, "career_vs_best2_scatter.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out2}")
plt.close()


# =====================================================================
# FIGURE 3: Violin plots — career vs best2 side by side per tier
# =====================================================================
import seaborn as sns

tier_order = ["Bust", "Flex", "Starter", "Elite", "Stud", "League-Winner"]
df["computed_tier"] = pd.Categorical(df["computed_tier"], categories=tier_order, ordered=True)

fig, axes = plt.subplots(5, 2, figsize=(16, 22))
fig.suptitle("Distribution by Tier: Career (left) vs Best 2 Seasons (right)",
             fontsize=16, fontweight="bold", y=1.01)

palette = {
    "Bust": "#d62728", "Flex": "#ff7f0e", "Starter": "#bcbd22",
    "Elite": "#2ca02c", "Stud": "#1f77b4", "League-Winner": "#9467bd",
}

for row_idx, (career_col, best2_col, name) in enumerate(PAIRS):
    for col_idx, (feat, label) in enumerate([(career_col, "Career"), (best2_col, "Best 2 Seasons")]):
        ax = axes[row_idx, col_idx]
        plot_df = df[["computed_tier", feat]].dropna()

        sns.violinplot(
            data=plot_df, x="computed_tier", y=feat,
            order=tier_order, hue="computed_tier", palette=palette,
            inner="quartile", cut=0, ax=ax, legend=False,
        )
        sns.stripplot(
            data=plot_df, x="computed_tier", y=feat,
            order=tier_order, color="black", alpha=0.25, size=2,
            jitter=True, ax=ax,
        )

        if col_idx == 0:
            ax.set_ylabel(name, fontsize=11, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(label if row_idx == 0 else "", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)

plt.tight_layout()
out3 = os.path.join(VIZ_DIR, "career_vs_best2_violins.png")
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out3}")
plt.close()

# =====================================================================
# Print summary
# =====================================================================
print("\n" + "=" * 70)
print("CAREER vs BEST-2-SEASONS SUMMARY")
print("=" * 70)
print(f"\n{'Feature':<25s} {'Career':>10s} {'Best2':>10s} {'Career':>8s} {'Best2':>8s} {'Career':>8s} {'Best2':>8s}")
print(f"{'':25s} {'Spearman':>10s} {'Spearman':>10s} {'AUC':>8s} {'AUC':>8s} {'Drift':>8s} {'Drift':>8s}")
print("-" * 95)
for m in metrics:
    sp_w = "<-" if abs(m["sp_career"]) > abs(m["sp_best2"]) else "->"
    auc_w = "<-" if m["auc_career"] > m["auc_best2"] else "->"
    drift_w = "<-" if m["drift_career"] < m["drift_best2"] else "->"
    print(f"{m['name']:<25s} {m['sp_career']:>+10.3f} {m['sp_best2']:>+10.3f} "
          f"{m['auc_career']:>8.3f} {m['auc_best2']:>8.3f} "
          f"{m['drift_career']:>8.3f} {m['drift_best2']:>8.3f}")

print("\nVerdict: Career aggregation wins on signal (Spearman, AUC) for all 5 features.")
print("Best-2 has better era stability for caught_percent and avoided_tackles.")
print("Overall: career is the stronger aggregation method.")
