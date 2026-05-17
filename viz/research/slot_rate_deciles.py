#!/usr/bin/env python3
"""
Slot rate decile analysis.

Shows what slot rates look like by decile across all WRs with 200+ FBS routes,
contextualizing where "slot-heavy" actually starts in the data.

Outputs: wr_data/charts/slot_rate_deciles.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
OUT_DIR = os.path.join(DATA_DIR, "charts")

NOTABLE_PLAYERS = [
    "Makai Lemon",
    "Justin Jefferson",
    "Jaxon Smith-Njigba",
    "Amon-Ra St. Brown",
    "Jaylen Waddle",
    "Drake London",
    "Malik Nabers",
    "Jordan Addison",
]


def main():
    # Compute snap-weighted career slot rates and routes from grades
    player_slot_snaps = {}
    player_total_snaps = {}
    player_routes = {}
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
                player_routes[name] = player_routes.get(name, 0) + row["routes"]

    # Build dataframe of all players with 200+ routes
    rows = []
    for name in player_routes:
        if player_routes[name] >= 200 and player_total_snaps.get(name, 0) > 0:
            sr = player_slot_snaps[name] / player_total_snaps[name] * 100
            rows.append({"name": name, "slot_rate": sr, "routes": player_routes[name]})
    df = pd.DataFrame(rows)

    # Compute percentiles
    percentiles = np.arange(0, 101, 10)
    pct_values = np.percentile(df["slot_rate"], percentiles)

    # Print summary
    print(f"Total WRs with 200+ FBS routes: {len(df)}")
    print()
    print("SLOT RATE BY DECILE")
    print("=" * 50)
    for p, v in zip(percentiles, pct_values):
        print(f"  {p:3d}th percentile: {v:5.1f}%")

    print()
    print("NOTABLE PLAYERS IN CONTEXT")
    print("=" * 50)
    for name in NOTABLE_PLAYERS:
        if name in player_total_snaps and player_total_snaps[name] > 0:
            sr = player_slot_snaps[name] / player_total_snaps[name] * 100
            pctile = (df["slot_rate"] < sr).mean() * 100
            print(f"  {name:<25s} {sr:5.1f}% slot  ({pctile:.0f}th percentile)")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Slot Rate Distribution: All WRs with 200+ FBS Routes (2016-2025)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Left: histogram with threshold lines
    ax1.hist(df["slot_rate"], bins=40, color="#4a90d9", alpha=0.7, edgecolor="white")
    ax1.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.5, label="50% (old threshold)")
    ax1.axvline(65, color="#f39c12", linestyle="--", linewidth=1.5, label="65% (new threshold)")
    ax1.axvline(75, color="#8e44ad", linestyle="--", linewidth=1.5, label="75% (Barrett threshold)")

    # Add notable player markers
    colors_notable = plt.cm.Set2(np.linspace(0, 1, len(NOTABLE_PLAYERS)))
    for i, name in enumerate(NOTABLE_PLAYERS):
        if name in player_total_snaps and player_total_snaps[name] > 0:
            sr = player_slot_snaps[name] / player_total_snaps[name] * 100
            ax1.axvline(sr, color=colors_notable[i], linestyle=":", linewidth=1.2, alpha=0.8)
            ax1.text(sr, ax1.get_ylim()[1] * 0.95 - i * ax1.get_ylim()[1] * 0.05,
                     f" {name.split()[-1]}", fontsize=7, color=colors_notable[i],
                     va="top", fontweight="bold")

    ax1.set_xlabel("Career Slot Rate (%)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Distribution + Key Thresholds", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")

    # Right: decile bar chart
    decile_labels = [f"{i*10}-{(i+1)*10}th" for i in range(10)]
    decile_mins = pct_values[:-1]
    decile_maxs = pct_values[1:]
    decile_mids = (decile_mins + decile_maxs) / 2

    bar_colors = ["#4a90d9"] * 10
    # Color deciles that contain the 65% threshold differently
    for i in range(10):
        if decile_maxs[i] >= 65:
            bar_colors[i] = "#f39c12"
        if decile_mins[i] >= 65:
            bar_colors[i] = "#e74c3c"

    bars = ax2.barh(range(10), decile_maxs - decile_mins, left=decile_mins,
                    color=bar_colors, alpha=0.7, edgecolor="white", height=0.7)

    for i in range(10):
        ax2.text(decile_maxs[i] + 1, i,
                 f"{decile_mins[i]:.0f}% - {decile_maxs[i]:.0f}%",
                 va="center", fontsize=8.5)

    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"Decile {i+1}\n(top {100-i*10}%)" for i in range(10)], fontsize=8)
    ax2.set_xlabel("Slot Rate (%)", fontsize=10)
    ax2.set_title("Slot Rate by Decile", fontsize=11)
    ax2.axvline(65, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.text(66, 9.3, "65%", fontsize=8, color="#f39c12", fontweight="bold")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "slot_rate_deciles.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
