#!/usr/bin/env python3
"""
Slot receiver college E[tier] comparison.

Shows slot-heavy WRs (>50% career slot) with college E[tier] >= 1.0,
colored by outcome tier.

Outputs: wr_data/charts/slot_college_etier.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
OUT_DIR = os.path.join(DATA_DIR, "charts")

TIER_COLORS = {
    "Bust": "#d62728",
    "Flex": "#ff7f0e",
    "Starter": "#bcbd22",
    "Elite": "#2ca02c",
    "Stud": "#1f77b4",
    "League-Winner": "#9467bd",
    "TBD": "#999999",
}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "slot_college_etier.csv"))
    df = df.sort_values("college_etier", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Slot WRs (≥50% Career Slot)  —  College E[tier] ≥ 1.0",
                 fontsize=14, fontweight="bold", y=0.98)

    y_pos = np.arange(len(df))
    colors = [TIER_COLORS[t] for t in df["outcome"]]

    ax.barh(y_pos, df["college_etier"], color=colors, alpha=0.85,
            edgecolor="white", height=0.7)

    for i, (_, row) in enumerate(df.iterrows()):
        tier = row["outcome"]
        label = f"{row['name']}  ({row['slot_rate']:.0f}% slot)  [{tier}]"
        ax.text(row["college_etier"] + 0.03, i, label, va="center", ha="left",
                fontsize=9, fontweight="bold", color=TIER_COLORS[tier])

    ax.set_yticks([])
    ax.set_xlabel("College Expected Tier", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(right=df["college_etier"].max() * 1.65)

    legend_patches = [mpatches.Patch(color=TIER_COLORS[t], label=t)
                      for t in ["League-Winner", "Stud", "Elite", "Flex", "Bust", "TBD"]]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right", ncol=2)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "slot_college_etier.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
