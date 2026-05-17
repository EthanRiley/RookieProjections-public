import pandas as pd
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

# --- Name normalization ---
SUFFIXES_RE = re.compile(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)

# Map all variants to one canonical form
NICKNAME_CANONICAL = {
    "josh": "joshua",
    "gabriel": "gabe",
    "gabe": "gabe",
    "joshua": "joshua",
}

def normalize_name(name):
    """Normalize a player name for fuzzy matching:
    - Strip suffixes (Jr., II, III, etc.)
    - Remove periods (D.J. -> DJ)
    - Normalize known nicknames to canonical form
    - Lowercase everything
    """
    n = SUFFIXES_RE.sub('', name).strip()
    n = n.replace('.', '')
    n = n.lower()
    parts = n.split()
    if len(parts) >= 2 and parts[0] in NICKNAME_CANONICAL:
        parts[0] = NICKNAME_CANONICAL[parts[0]]
    return ' '.join(parts)

# Load the filtered per-route data
per_route = pd.read_csv(os.path.join(DATA_DIR, "per_route_by_season_drafted_wrs_only.csv"))
print(f"Per route records: {len(per_route)}")
print(f"Seasons in per_route: {sorted(per_route['season'].unique())}")

# Join receiving grades for each year 2016-2023
years = range(2016, 2024)
merged_parts = []

for year in years:
    grades_file = os.path.join(DATA_DIR, "grades", f"{year}_receiving_grades.csv")
    grades = pd.read_csv(grades_file)

    # Filter per_route to this season
    pr_season = per_route[per_route["season"] == year].copy()

    if len(pr_season) == 0:
        print(f"{year}: no per_route records, skipping")
        continue

    # Add normalized join keys
    pr_season["_join_key"] = pr_season["player"].apply(normalize_name)
    grades["_join_key"] = grades["player"].apply(normalize_name)

    # Drop overlapping columns from grades (except _join_key used for merge)
    grades_cols_to_drop = []
    for col in grades.columns:
        if col in per_route.columns and col not in ("player", "_join_key"):
            grades_cols_to_drop.append(col)

    grades_clean = grades.drop(columns=grades_cols_to_drop, errors='ignore')
    # Drop original player column from grades to avoid collision
    grades_clean = grades_clean.drop(columns=["player"])

    merged = pr_season.merge(grades_clean, on="_join_key", how="left", suffixes=("", "_grades"))
    merged = merged.drop(columns=["_join_key"])

    matched = merged[merged["player_id"].notna()]
    unmatched = merged[merged["player_id"].isna()]
    print(f"{year}: {len(pr_season)} per_route rows, {len(matched)} matched, {len(unmatched)} unmatched")

    if len(unmatched) > 0:
        print(f"  Unmatched: {unmatched['player'].tolist()}")

    merged_parts.append(merged)

# Combine all years
final = pd.concat(merged_parts, ignore_index=True)
print(f"\nFinal dataset: {len(final)} rows, {len(final.columns)} columns")

output_path = os.path.join(DATA_DIR, "wr_by_season_final.csv")
final.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
