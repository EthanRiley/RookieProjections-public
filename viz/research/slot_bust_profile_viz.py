#!/usr/bin/env python3
"""
Slot receiver bust vs hit profile comparison.

Shows that busts among top-quartile slot receivers had weak prospect profiles,
not that slot alignment caused the bust.

Panel 1: Scatter of draft_capital vs YPRR (graduated), colored by outcome tier
Panel 2: Normalized feature comparison — hits vs busts (z-scored averages)

Outputs: wr_data/charts/slot_bust_profiles.png
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
OUT_DIR = os.path.join(DATA_DIR, "charts")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

TIER_COLORS = {
    "Bust": "#d62728",
    "Flex": "#ff7f0e",
    "Starter": "#bcbd22",
    "Elite": "#2ca02c",
    "Stud": "#1f77b4",
    "League-Winner": "#9467bd",
}

FULL_MODEL = [
    "draft_capital", "best1_yprr_graduated", "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj", "best2_contested_catch_rate", "best2_avoided_tackles_per_rec",
]

FEATURE_LABELS = {
    "draft_capital": "Draft Capital",
    "best1_yprr_graduated": "YPRR\n(graduated)",
    "career_targeted_qb_rating": "Targeted\nQBR",
    "best2_catch_pct_adot_adj": "Catch%\n(aDOT adj)",
    "best2_contested_catch_rate": "Contested\nCatch%",
    "best2_avoided_tackles_per_rec": "Avoided\nTackles/Rec",
}


def main():
    master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER)

    # Load predictions for expected_tier
    prospect_preds = {}
    for f in ["holdout_predictions_v2.csv", "prospect_predictions_2024.csv",
              "prospect_predictions_2025.csv", "prospect_predictions_2026.csv"]:
        path = os.path.join(DATA_DIR, "outputs", f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if row["name"] not in prospect_preds:
                    prospect_preds[row["name"]] = row["expected_tier"]

    # Compute snap-weighted slot rates
    player_slot_snaps = {}
    player_total_snaps = {}
    for y in range(2016, 2026):
        path = os.path.join(DATA_DIR, "grades", f"{y}_receiving_grades.csv")
        if os.path.exists(path):
            g = pd.read_csv(path)
            for _, row in g.iterrows():
                name = row["player"]
                ss = row.get("slot_snaps", 0)
                ws = row.get("wide_snaps", 0)
                inl = row.get("inline_snaps", 0)
                total = ss + ws + inl
                player_slot_snaps[name] = player_slot_snaps.get(name, 0) + ss
                player_total_snaps[name] = player_total_snaps.get(name, 0) + total

    for idx, row in master.iterrows():
        name = row["name"]
        if name in player_total_snaps and player_total_snaps[name] > 0:
            master.at[idx, "career_slot_rate"] = (
                player_slot_snaps[name] / player_total_snaps[name] * 100
            )

    master["expected_tier"] = master["name"].map(prospect_preds)

    # Filter to top-quartile slot, R1-2 or E[tier] > 1.0, resolved outcomes
    TOP_QUARTILE = 64.3
    FORCE_INCLUDE = {"Amon-Ra St. Brown"}
    slot = master[
        (master["career_slot_rate"] >= TOP_QUARTILE) &
        ((master["round"] <= 2) | (master["expected_tier"] > 1.0) | (master["name"].isin(FORCE_INCLUDE)))
    ].copy()
    slot = slot.dropna(subset=["computed_tier"])
    slot = slot[slot["computed_tier"] != "TBD"]

    print(f"Resolved top-quartile slot players: {len(slot)}")

    # Classify hit vs bust
    slot["is_hit"] = slot["tier_ordinal"] >= 3  # Elite+

    # ===== PLOTTING =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Top-Quartile Slot WRs: Why Busts Busted",
                 fontsize=14, fontweight="bold", y=0.98)

    # --- Panel 1: Scatter DC vs YPRR ---
    for tier_name in ["Bust", "Flex", "Starter", "Elite", "Stud", "League-Winner"]:
        mask = slot["computed_tier"] == tier_name
        if mask.sum() == 0:
            continue
        ax1.scatter(
            slot.loc[mask, "draft_capital"],
            slot.loc[mask, "best1_yprr_graduated"],
            c=TIER_COLORS[tier_name], label=tier_name, alpha=0.8,
            s=70, edgecolor="white", linewidth=0.5, zorder=3,
        )
        # Label each point
        for _, row in slot.loc[mask].iterrows():
            last_name = row["name"].split()[-1]
            ax1.annotate(
                last_name, (row["draft_capital"], row["best1_yprr_graduated"]),
                fontsize=6.5, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points",
                color=TIER_COLORS[tier_name], fontweight="bold",
            )

    # Plot Makai Lemon as a star
    LEMON = {
        "draft_capital": 8.06,
        "best1_yprr_graduated": 3.1833,
        "career_targeted_qb_rating": 119.72,
        "best2_catch_pct_adot_adj": 14.45,
        "best2_contested_catch_rate": 59.27,
        "best2_avoided_tackles_per_rec": 0.26,
    }
    ax1.scatter(LEMON["draft_capital"], LEMON["best1_yprr_graduated"],
                c="#FFD700", marker="*", s=300, edgecolor="black", linewidth=0.8,
                zorder=5, label="Makai Lemon")
    ax1.annotate("Makai Lemon", (LEMON["draft_capital"], LEMON["best1_yprr_graduated"]),
                 fontsize=7.5, ha="left", va="bottom", xytext=(6, 6),
                 textcoords="offset points", color="#B8860B", fontweight="bold")

    ax1.set_xlabel("Draft Capital", fontsize=10)
    ax1.set_ylabel("YPRR (graduated)", fontsize=10)
    ax1.set_title("Draft Capital vs YPRR — Hits Cluster Upper-Right", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.2)

    # --- Panel 2: Z-scored feature comparison (hits vs busts) ---
    feats_available = [f for f in FULL_MODEL if f in slot.columns]
    hits = slot[slot["is_hit"]].dropna(subset=feats_available)
    busts = slot[~slot["is_hit"]].dropna(subset=feats_available)

    # Z-score relative to full dataset
    scaler = StandardScaler()
    all_vals = master.dropna(subset=feats_available)[feats_available]
    scaler.fit(all_vals)

    hit_z = pd.DataFrame(scaler.transform(hits[feats_available]), columns=feats_available).mean()
    bust_z = pd.DataFrame(scaler.transform(busts[feats_available]), columns=feats_available).mean()

    # Compute Lemon's z-scores
    lemon_vals = pd.DataFrame([LEMON])[feats_available]
    lemon_z = pd.DataFrame(scaler.transform(lemon_vals), columns=feats_available).iloc[0]

    x = np.arange(len(feats_available))
    width = 0.25
    bars_hit = ax2.bar(x - width, hit_z.values, width, color="#2ca02c", alpha=0.8,
                       label=f"Hits — Elite+ (n={len(hits)})", edgecolor="white")
    bars_bust = ax2.bar(x, bust_z.values, width, color="#d62728", alpha=0.8,
                        label=f"Misses — Below Elite (n={len(busts)})", edgecolor="white")
    bars_lemon = ax2.bar(x + width, lemon_z.values, width, color="#FFD700", alpha=0.9,
                         label="Makai Lemon", edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([FEATURE_LABELS.get(f, f) for f in feats_available], fontsize=8.5)
    ax2.set_ylabel("Z-Score (relative to all drafted WRs)", fontsize=10)
    ax2.set_title("Average Feature Profile: Hits vs Misses vs Lemon", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)

    # Add value labels
    for bar_set, z_vals in [(bars_hit, hit_z.values), (bars_bust, bust_z.values), (bars_lemon, lemon_z.values)]:
        for bar, val in zip(bar_set, z_vals):
            y_pos = val + 0.05 if val >= 0 else val - 0.12
            ax2.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.2f}",
                     ha="center", fontsize=6.5, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "slot_bust_profiles.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
