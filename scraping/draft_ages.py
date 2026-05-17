#!/usr/bin/env python3
"""
Compute precise fractional draft age for all WRs.

Uses nflverse import_ids() for exact birthdates and NFL draft dates
to compute age in years at the time of drafting (day 1 of the draft).

Updates:
  - wr_data/wr_dynasty_value_with_college.csv (overwrites draft_age column)

Also outputs a standalone CSV for prospect use:
  - wr_data/draft_ages.csv
"""

import os

import nfl_data_py as nfl
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

# NFL Draft day 1 dates (Thursday of draft week)
DRAFT_DATES = {
    2016: "2016-04-28",
    2017: "2017-04-27",
    2018: "2018-04-26",
    2019: "2019-04-25",
    2020: "2020-04-23",
    2021: "2021-04-29",
    2022: "2022-04-28",
    2023: "2023-04-27",
    2024: "2024-04-25",
    2025: "2025-04-24",
    2026: "2026-04-23",
}


def compute_fractional_age(birthdate, draft_date):
    """Compute age in fractional years at draft date."""
    if pd.isna(birthdate) or pd.isna(draft_date):
        return np.nan
    bd = pd.Timestamp(birthdate)
    dd = pd.Timestamp(draft_date)
    delta = dd - bd
    return round(delta.days / 365.25, 2)


# Load nflverse IDs (has birthdates)
print("Loading nflverse IDs...")
ids = nfl.import_ids()
ids = ids[ids["birthdate"].notna()].copy()
ids["birthdate"] = pd.to_datetime(ids["birthdate"])

# Build a lookup: pfr_id -> birthdate
pfr_to_birth = dict(zip(ids["pfr_id"], ids["birthdate"]))
print(f"  {len(pfr_to_birth)} players with birthdates")

# Process all draft years
all_ages = []

for year in range(2016, 2027):
    draft_date = DRAFT_DATES.get(year)
    if not draft_date:
        continue

    try:
        draft = nfl.import_draft_picks([year])
    except Exception:
        continue

    wr = draft[draft["category"].str.upper() == "WR"].copy()
    if wr.empty:
        continue

    for _, row in wr.iterrows():
        pfr_id = row.get("pfr_player_id")
        birthdate = pfr_to_birth.get(pfr_id)
        age = compute_fractional_age(birthdate, draft_date)

        all_ages.append({
            "name": row["pfr_player_name"],
            "draft_year": year,
            "pick": row["pick"],
            "pfr_id": pfr_id,
            "birthdate": birthdate,
            "draft_date": draft_date,
            "draft_age": age,
        })

    matched = sum(1 for a in all_ages if a["draft_year"] == year and pd.notna(a["draft_age"]))
    total = len(wr)
    print(f"  {year}: {matched}/{total} WRs with precise age")

ages_df = pd.DataFrame(all_ages)

# Save standalone ages CSV
ages_path = os.path.join(DATA_DIR, "draft_ages.csv")
ages_df.to_csv(ages_path, index=False)
print(f"\nSaved {len(ages_df)} ages to {ages_path}")

# Show some examples to verify
print("\nVerification (2025 round 1):")
r1_25 = ages_df[(ages_df["draft_year"] == 2025) & (ages_df["pick"] <= 40)]
for _, row in r1_25.iterrows():
    bd_str = row["birthdate"].strftime("%Y-%m-%d") if pd.notna(row["birthdate"]) else "???"
    print(f"  {row['name']:<25s} pick {row['pick']:>3d}  born {bd_str}  age {row['draft_age']}")

print("\nVerification (2026 round 1):")
r1_26 = ages_df[(ages_df["draft_year"] == 2026) & (ages_df["pick"] <= 40)]
for _, row in r1_26.iterrows():
    bd_str = row["birthdate"].strftime("%Y-%m-%d") if pd.notna(row["birthdate"]) else "???"
    print(f"  {row['name']:<25s} pick {row['pick']:>3d}  born {bd_str}  age {row['draft_age']}")

# --- Update training data ---
print("\n" + "=" * 70)
print("UPDATING TRAINING DATA")
print("=" * 70)

train_path = os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv")
train = pd.read_csv(train_path)

old_ages = train["draft_age"].copy()

# Match by name + draft_year
ages_lookup = ages_df.set_index(["name", "draft_year"])["draft_age"]

matched = 0
updated = 0
for idx, row in train.iterrows():
    key = (row["name"], row["draft_year"])
    if key in ages_lookup.index:
        new_age = ages_lookup[key]
        if pd.notna(new_age):
            matched += 1
            if row["draft_age"] != new_age:
                updated += 1
            train.at[idx, "draft_age"] = new_age

print(f"  Matched: {matched}/{len(train)}")
print(f"  Updated: {updated}")

# Show biggest changes
train["old_age"] = old_ages
train["age_diff"] = (train["draft_age"] - train["old_age"]).abs()
changed = train[train["age_diff"] > 0.01].sort_values("age_diff", ascending=False)
if len(changed) > 0:
    print(f"\n  Biggest age corrections:")
    for _, row in changed.head(20).iterrows():
        print(f"    {row['name']:<25s} {row['draft_year']}  old={row['old_age']}  new={row['draft_age']:.2f}  diff={row['age_diff']:.2f}")

train = train.drop(columns=["old_age", "age_diff"])
train.to_csv(train_path, index=False)
print(f"\nSaved updated training data to {train_path}")
