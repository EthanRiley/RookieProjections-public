#!/usr/bin/env python3
"""
Filter per_route_by_season data to only include drafted WRs from 2016-2023.
Uses exact matching first, then fuzzy matching for remaining names.

Reads from:
  - ../wr_data/rookie_wr_2016_2023.csv (player names to keep)
  - ../wr_data/Per Route By Season.csv (full per-route data)
Outputs:
  - ../wr_data/per_route_by_season_drafted_wrs_only.csv
  - ../wr_data/missing_rookies.csv (players not found even after fuzzy matching)
"""

import os
import pandas as pd
from thefuzz import process, fuzz

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

FUZZY_THRESHOLD = 95

# Manual overrides for known name mismatches (rookie_name -> per_route_name)
MANUAL_MAP = {
    "D.J. Moore": "DJ Moore",
    "DJ Chark": "DJ Chark Jr.",
    "Gabriel Davis": "Gabe Davis",
    "Gary Jennings Jr": "Gary Jennings",
    "Grant Dubose": "Grant DuBose",
    "JJ Arcega-Whiteside": "J.J. Arcega-Whiteside",
    "James Proche": "James Proche II",
    "John Metchie": "John Metchie III",
    "Josh Palmer": "Joshua Palmer",
    "Marvin Mims": "Marvin Mims Jr.",
    "Mecole Hardman": "Mecole Hardman Jr.",
    "Ray-Ray McCloud": "Ray-Ray McCloud III",
    "Terry Godwin": "Terry Godwin II",
    "DeMario Douglas": "Demario Douglas",
}

rookies = pd.read_csv(os.path.join(DATA_DIR, "rookie_wr_2016_2023.csv"))
per_route = pd.read_csv(os.path.join(DATA_DIR, "Per Route By Season.csv"))

rookie_names = set(rookies["name"].unique())
per_route_names = per_route["player"].unique().tolist()

# --- Step 1: Apply manual overrides first ---
manual_reverse = {v: k for k, v in MANUAL_MAP.items()}  # per_route_name -> rookie_name
manual_matched = set()

for rookie_name, pr_name in MANUAL_MAP.items():
    if pr_name in per_route_names:
        manual_matched.add(rookie_name)

# --- Step 2: Exact matches ---
exact_matches = rookie_names & set(per_route_names)
already_resolved = exact_matches | manual_matched
unmatched = sorted(rookie_names - already_resolved)

print(f"Rookies in list: {len(rookie_names)}")
print(f"Exact matches: {len(exact_matches)}")
print(f"Manual overrides matched: {len(manual_matched)}")
print(f"Attempting fuzzy match on {len(unmatched)} remaining names...")

# --- Step 3: Fuzzy match remaining unmatched names ---
fuzzy_map = {}  # per_route_name -> rookie_name
still_missing = []

for name in unmatched:
    result = process.extractOne(name, per_route_names, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= FUZZY_THRESHOLD:
        fuzzy_map[result[0]] = name
        print(f"  Fuzzy: '{name}' -> '{result[0]}' (score: {result[1]})")
    else:
        still_missing.append(name)
        best = result[0] if result else "N/A"
        score = result[1] if result else 0
        print(f"  No match: '{name}' (best: '{best}', score: {score})")

# --- Step 4: Build filtered dataset ---
# Exact match rows
exact_filtered = per_route[per_route["player"].isin(exact_matches)].copy()

# Manual override rows
manual_pr_names = set(MANUAL_MAP.values()) & set(per_route_names)
manual_filtered = per_route[per_route["player"].isin(manual_pr_names)].copy()
manual_filtered["player"] = manual_filtered["player"].map(manual_reverse)

# Fuzzy match rows
fuzzy_filtered = per_route[per_route["player"].isin(fuzzy_map.keys())].copy()
fuzzy_filtered["player"] = fuzzy_filtered["player"].map(fuzzy_map)

filtered = pd.concat([exact_filtered, manual_filtered, fuzzy_filtered], ignore_index=True)

print(f"\nTotal matched players: {filtered['player'].nunique()}")
print(f"Total rows kept: {len(filtered)}")

# --- Step 4: Save outputs ---
filtered.to_csv(os.path.join(DATA_DIR, "per_route_by_season_drafted_wrs_only.csv"), index=False)
print(f"\nSaved to {os.path.join(DATA_DIR, 'per_route_by_season_drafted_wrs_only.csv')}")

if still_missing:
    print(f"\n{len(still_missing)} players still not found:")
    for name in still_missing:
        print(f"  {name}")
    missing_df = rookies[rookies["name"].isin(still_missing)].sort_values(["year", "name"])
    missing_df.to_csv(os.path.join(DATA_DIR, "missing_rookies.csv"), index=False)
    print(f"Saved missing players to {os.path.join(DATA_DIR, 'missing_rookies.csv')}")
else:
    print("\nAll rookies matched!")
