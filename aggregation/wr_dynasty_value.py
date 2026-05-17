#!/usr/bin/env python3
"""
Compute a dynasty value metric for rookie WRs.

For each player, finds their first 4 NFL seasons (rookie contract),
computes points above replacement (WR36 baseline per season), raises
to power k=1.2 to capture the convex dynasty value curve, then
averages the best 2 of 4 seasons.

Reads:
  - wr_data/rookie_wr_2016_2024_classified.csv
  - wr_data/nfl_yearly_ppr_totals_2016_2025.csv
Outputs:
  - wr_data/wr_dynasty_value.csv
"""

import math
import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

K = 1.2
WR_REPLACEMENT_RANK = 36
TOP_N_SEASONS = 2
FIRST_N_YEARS = 4
MAX_PICK = 260  # approximate last pick in a draft

# --- Name normalization (shared logic with join_grades.py) ---
SUFFIXES_RE = re.compile(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)

def draft_capital_score(pick):
    """Sqrt draft capital: pick 1 ≈ 10, last pick ≈ 3. Clusters top picks tightly."""
    return 10 - 7 * math.sqrt(pick / MAX_PICK)


def normalize_name(name):
    n = SUFFIXES_RE.sub('', str(name)).strip()
    n = n.replace('.', '').replace("'", '').lower()
    return ' '.join(n.split())


# --- Load data ---
rookies = pd.read_csv(os.path.join(DATA_DIR, "rookie_wr_2016_2024_classified.csv"))
ppr = pd.read_csv(os.path.join(DATA_DIR, "nfl_yearly_ppr_totals_2016_2025.csv"))

ppr_wr = ppr[ppr["position"] == "WR"].copy()

# --- Compute per-season WR replacement baseline ---
baselines = (
    ppr_wr.groupby("season")["fantasy_points_ppr"]
    .apply(lambda s: s.nlargest(WR_REPLACEMENT_RANK).iloc[-1])
    .rename("baseline")
)
print("WR36 baselines per season:")
print(baselines.to_string())
print()

# --- Join rookies to PPR stats via normalized names ---
rookies["_join_key"] = rookies["name"].apply(normalize_name)
ppr_wr["_join_key"] = ppr_wr["player_display_name"].apply(normalize_name)

# For each rookie, only keep seasons within their first N years
results = []

for _, row in rookies.iterrows():
    name = row["name"]
    draft_year = row["year"]
    key = row["_join_key"]
    tier = row["tier"]

    first_years = list(range(draft_year, draft_year + FIRST_N_YEARS))
    player_seasons = ppr_wr[
        (ppr_wr["_join_key"] == key) & (ppr_wr["season"].isin(first_years))
    ].copy()

    season_values = []
    for yr in first_years:
        season_row = player_seasons[player_seasons["season"] == yr]
        if len(season_row) == 0:
            season_values.append(0.0)
        else:
            pts = season_row["fantasy_points_ppr"].values[0]
            baseline = baselines.get(yr, 0)
            par = max(pts - baseline, 0) ** K
            season_values.append(par)

    # Best 2 of 4
    top_seasons = sorted(season_values, reverse=True)[:TOP_N_SEASONS]
    dynasty_value = sum(top_seasons) / TOP_N_SEASONS

    # Classify based on dynasty value thresholds
    if dynasty_value >= 350:
        computed_tier = "League-Winner"
    elif dynasty_value >= 180:
        computed_tier = "Stud"
    elif dynasty_value >= 75:
        computed_tier = "Elite"
    elif dynasty_value >= 50:
        computed_tier = "Starter"
    elif dynasty_value > 0:
        computed_tier = "Flex"
    else:
        computed_tier = "Bust"

    results.append({
        "name": name,
        "draft_year": draft_year,
        "round": row["round"],
        "pick": row["pick"],
        "draft_capital": round(draft_capital_score(row["pick"]), 2),
        "manual_tier": tier,
        "computed_tier": computed_tier,
        "dynasty_value": round(dynasty_value, 2),
        "season_values": top_seasons,
    })

df = pd.DataFrame(results)
df = df.sort_values("dynasty_value", ascending=False).reset_index(drop=True)

# Print computed tier distribution
print("\nComputed tier distribution:")
tier_order = ["League-Winner", "Stud", "Elite", "Starter", "Flex", "Bust"]
tier_counts = df["computed_tier"].value_counts().reindex(tier_order).fillna(0).astype(int)
print(tier_counts.to_string())

# Print all non-bust players
print("\nAll non-bust players:")
non_bust = df[df["computed_tier"] != "Bust"]
print(non_bust[["name", "draft_year", "computed_tier", "dynasty_value"]].to_string(index=False))

# Save
output_path = os.path.join(DATA_DIR, "wr_dynasty_value.csv")
df.drop(columns=["season_values"]).to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
