#!/usr/bin/env python3
"""
Slot receiver outcome visualization.

Shows R1-2 WRs with >200 career routes and >50% career slot rate,
colored by outcome tier. Includes 2024-2026 prospects as TBD.

Outputs: wr_data/charts/slot_receiver_outcomes.png
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
    master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))

    # Load all predictions to get expected_tier
    prospect_preds = {}
    for f in ["holdout_predictions_v2.csv", "prospect_predictions_2024.csv",
              "prospect_predictions_2025.csv", "prospect_predictions_2026.csv"]:
        path = os.path.join(DATA_DIR, "outputs", f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if row["name"] not in prospect_preds:
                    prospect_preds[row["name"]] = row["expected_tier"]

    # Compute career routes from grades
    routes = {}
    for y in range(2016, 2026):
        path = os.path.join(DATA_DIR, "grades", f"{y}_receiving_grades.csv")
        if os.path.exists(path):
            g = pd.read_csv(path)
            for _, row in g.iterrows():
                name = row["player"]
                routes[name] = routes.get(name, 0) + row["routes"]

    master = master.copy()
    master["career_routes"] = master["name"].map(routes)

    # Compute snap-weighted career slot rates from grades (more accurate than game-weighted)
    player_slots = {}
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

    # Override master career_slot_rate with snap-weighted version
    for idx, row in master.iterrows():
        name = row["name"]
        if name in player_total_snaps and player_total_snaps[name] > 0:
            master.at[idx, "career_slot_rate"] = (
                player_slot_snaps[name] / player_total_snaps[name] * 100
            )

    # Add prospect-only players from prediction files
    prospect_rows = []
    for year in [2024, 2025, 2026]:
        path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_{year}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                name = row["name"]
                if name not in master["name"].values:
                    rnd = row.get("round", 99)
                    cr = routes.get(name, 0)
                    ts = player_total_snaps.get(name, 1)
                    cs = player_slot_snaps.get(name, 0) / ts * 100 if ts > 0 else 0
                    prospect_rows.append({
                        "name": name,
                        "draft_year": row["draft_year"],
                        "round": rnd,
                        "pick": row["pick"],
                        "computed_tier": "TBD",
                        "dynasty_value": 0,
                        "career_slot_rate": cs,
                        "career_routes": cr,
                    })

    if prospect_rows:
        prospect_df = pd.DataFrame(prospect_rows)
        master = pd.concat([master, prospect_df], ignore_index=True)

    # Add expected_tier to master for filtering
    master["expected_tier"] = master["name"].map(prospect_preds)

    # Filter: (R1-2 OR model score >1.0 OR force-include), >200 career routes, >50% career slot
    FORCE_INCLUDE = {"Amon-Ra St. Brown"}
    candidates = master[
        (master["round"] <= 2) | (master["expected_tier"] > 1.0) | (master["name"].isin(FORCE_INCLUDE))
    ].copy()
    slot = candidates[
        (candidates["career_routes"] > 200) & (candidates["career_slot_rate"] > 50)
    ].copy()

    # Mark TBD only for players with no real dynasty value (2025/2026 prospects)
    slot["display_tier"] = slot.apply(
        lambda r: "TBD" if r["computed_tier"] == "TBD" else r["computed_tier"], axis=1)

    # Sort: TBDs at bottom by expected_tier, rest by dynasty_value
    slot["sort_key"] = slot.apply(
        lambda r: -1000 + prospect_preds.get(r["name"], 0) if r["display_tier"] == "TBD"
        else r["dynasty_value"], axis=1)
    slot = slot.sort_values("sort_key", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("WRs with >50% Career Slot Rate  (R1-2 or E[tier] > 1.0, >200 routes)",
                 fontsize=14, fontweight="bold", y=0.98)

    y_pos = np.arange(len(slot))
    colors = [TIER_COLORS[t] for t in slot["display_tier"]]

    # Use dynasty_value for resolved players, no bar for TBDs
    bar_values = []
    for _, row in slot.iterrows():
        if row["display_tier"] == "TBD":
            bar_values.append(0)
        else:
            bar_values.append(row["dynasty_value"])

    ax.barh(y_pos, bar_values, color=colors, alpha=0.85, edgecolor="white", height=0.7)

    # Labels — always to the right of the bar
    for i, (_, row) in enumerate(slot.iterrows()):
        tier_label = row["display_tier"]
        slot_pct = row["career_slot_rate"]
        name = row["name"]
        val = bar_values[i]

        if tier_label == "TBD":
            et = prospect_preds.get(name, 0)
            label = f"{name}  ({slot_pct:.0f}% slot)  E[tier]={et:.2f}"
        else:
            label = f"{name}  ({slot_pct:.0f}% slot)  [{tier_label}]"

        x_pos = max(val, 0) + 5
        ax.text(x_pos, i, label, va="center", ha="left", fontsize=8.5, fontweight="bold",
                color=TIER_COLORS[tier_label])

    ax.set_yticks([])
    ax.set_xlabel("Dynasty Value", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(right=max(bar_values) * 1.55)

    # Legend
    legend_patches = [mpatches.Patch(color=TIER_COLORS[t], label=t)
                      for t in ["League-Winner", "Stud", "Elite", "Starter", "Flex", "Bust", "TBD"]]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right", ncol=2)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "slot_receiver_outcomes.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
