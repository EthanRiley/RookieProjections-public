#!/usr/bin/env python3
"""
Build dynasty value datasets with train/test splits for any position.

Modular pipeline:
  1. Fetch draft picks from nflverse for the specified years
  2. Compute dynasty value from NFL PPR totals (best 2 of 4 seasons, convex-transformed)
  3. Classify into ordinal tiers
  4. Aggregate college stats from PFF grades
  5. Split into train (<=max_train_year) and holdout (>max_train_year)

Supports: WR, RB, TE (TE requires its own aggregation module)

Usage:
  python3 aggregation/build_dynasty_dataset.py --position RB
  python3 aggregation/build_dynasty_dataset.py --position RB --train-end 2021
  python3 aggregation/build_dynasty_dataset.py --position WR --train-end 2021 --draft-start 2016
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

# Add project root for imports

# --- Shared constants ---
K = 1.2                # Convex exponent for dynasty value
TOP_N_SEASONS = 2      # Best N of first-contract seasons to average
FIRST_N_YEARS = 4      # Rookie contract length
MAX_PICK = 260         # Approximate last pick in a draft

SUFFIXES_RE = re.compile(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)

# Position-specific replacement levels (rank for baseline PPR)
REPLACEMENT_RANKS = {
    "WR": 36,
    "RB": 24,
    "TE": 12,
    "QB": 12,
}

# Tier thresholds (same across positions — dynasty_value cutoffs)
TIER_THRESHOLDS = [
    ("League-Winner", 350),
    ("Stud", 180),
    ("Elite", 75),
    ("Starter", 30),
    ("Flex", 0.01),  # > 0
    ("Bust", 0),
]

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}


def normalize_name(name):
    n = SUFFIXES_RE.sub('', str(name)).strip()
    n = n.replace('.', '').replace("'", '').lower()
    return ' '.join(n.split())


def draft_capital_score(pick):
    return round(max(10 - (10 / math.log(261)) * math.log(pick + 1), 0), 2)


def classify_tier(dynasty_value):
    for tier_name, threshold in TIER_THRESHOLDS:
        if dynasty_value >= threshold:
            return tier_name
    return "Bust"


def _build_rb_birth_lookup(dynasty_df):
    """Build name -> birthdate lookup for RBs from nflverse seasonal rosters."""
    try:
        import nfl_data_py as nfl
        years = sorted(dynasty_df["draft_year"].unique().tolist())
        rosters = nfl.import_seasonal_rosters(years)
        rbs = rosters[rosters["position"] == "RB"][["player_name", "birth_date"]].drop_duplicates(subset="player_name")
        rbs = rbs[rbs["birth_date"].notna()]
        lookup = {}
        for _, row in rbs.iterrows():
            key = normalize_name(row["player_name"])
            lookup[key] = pd.Timestamp(row["birth_date"])
        print(f"  RB birthdate lookup: {len(lookup)} players")
        return lookup
    except Exception as e:
        print(f"  Warning: could not build RB birth lookup: {e}")
        return {}


# ============================================================
# Step 1: Fetch draft picks
# ============================================================

def fetch_draft_picks(position, draft_years):
    """Fetch draft picks for a position from nflverse."""
    import nfl_data_py as nfl

    all_picks = []
    for year in draft_years:
        df = nfl.import_draft_picks([year])
        pos = df[df["category"].str.upper() == position.upper()].copy()
        pos = pos.sort_values("pick").reset_index(drop=True)
        pos = pos.rename(columns={"pfr_player_name": "name", "season": "draft_year"})
        pos["draft_capital"] = pos["pick"].apply(draft_capital_score)
        all_picks.append(pos[["name", "draft_year", "round", "pick", "draft_capital"]].copy())

    result = pd.concat(all_picks, ignore_index=True)
    result = result.dropna(subset=["name"])
    print(f"  Fetched {len(result)} {position} draft picks across {len(draft_years)} years")
    return result


# ============================================================
# Step 2: Compute dynasty values
# ============================================================

def compute_dynasty_values(picks, position, ppr_path):
    """Compute dynasty value for each player from NFL PPR totals."""
    ppr = pd.read_csv(ppr_path)
    ppr_pos = ppr[ppr["position"] == position.upper()].copy()

    replacement_rank = REPLACEMENT_RANKS.get(position.upper(), 36)

    # Per-season replacement baselines
    baselines = (
        ppr_pos.groupby("season")["fantasy_points_ppr"]
        .apply(lambda s: s.nlargest(replacement_rank).iloc[-1] if len(s) >= replacement_rank else 0)
        .rename("baseline")
    )
    print(f"  {position.upper()}{replacement_rank} baselines computed for {len(baselines)} seasons")

    ppr_pos["_join_key"] = ppr_pos["player_display_name"].apply(normalize_name)
    picks["_join_key"] = picks["name"].apply(normalize_name)

    results = []
    for _, row in picks.iterrows():
        name = row["name"]
        draft_year = row["draft_year"]
        key = row["_join_key"]

        first_years = list(range(int(draft_year), int(draft_year) + FIRST_N_YEARS))
        player_seasons = ppr_pos[
            (ppr_pos["_join_key"] == key) & (ppr_pos["season"].isin(first_years))
        ]

        season_values = []
        for yr in first_years:
            baseline = baselines.get(yr, 0)
            season_data = player_seasons[player_seasons["season"] == yr]
            if len(season_data) > 0:
                pts = season_data["fantasy_points_ppr"].values[0]
                above = max(pts - baseline, 0)
                convex = above ** K
                season_values.append(convex)
            else:
                season_values.append(0)

        # Best N of first-contract seasons
        season_values.sort(reverse=True)
        dynasty_value = round(np.mean(season_values[:TOP_N_SEASONS]), 2) if season_values else 0

        # Check if player is "resolved"
        # Hard stop: rookie contract window has closed (draft_year + 4 <= current year)
        # Soft check: at least 2 NFL seasons of data
        current_year = 2025
        contract_expired = (draft_year + FIRST_N_YEARS) <= current_year
        max_nfl_season = ppr_pos[ppr_pos["_join_key"] == key]["season"].max() if len(
            ppr_pos[ppr_pos["_join_key"] == key]) > 0 else 0
        seasons_played = max_nfl_season - draft_year + 1 if pd.notna(max_nfl_season) else 0
        is_resolved = contract_expired or seasons_played >= 2

        results.append({
            "name": name,
            "draft_year": int(draft_year),
            "round": int(row["round"]),
            "pick": int(row["pick"]),
            "draft_capital": row["draft_capital"],
            "dynasty_value": dynasty_value,
            "computed_tier": classify_tier(dynasty_value) if is_resolved else "TBD",
            "is_resolved": is_resolved,
            "nfl_seasons": int(seasons_played),
        })

    result = pd.DataFrame(results)
    resolved = result[result["is_resolved"]]
    print(f"  Dynasty values computed: {len(result)} players, {len(resolved)} resolved")
    tier_counts = resolved["computed_tier"].value_counts()
    for tier in ["League-Winner", "Stud", "Elite", "Starter", "Flex", "Bust"]:
        print(f"    {tier}: {tier_counts.get(tier, 0)}")

    return result


# ============================================================
# Step 3: Aggregate college stats
# ============================================================

def aggregate_college_stats(dynasty_df, position):
    """Aggregate college stats and merge onto dynasty data."""
    if position.upper() == "WR":
        from aggregation.aggregate_college_stats import (
            load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
        )
        all_grades = load_all_grades(range(2016, 2027))
        birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
        adot_coef = fit_adot_regression(all_grades)

        rows = []
        for _, row in dynasty_df.iterrows():
            result = aggregate_player(
                all_grades, row["name"], row["draft_year"],
                birth_lookup=birth_lookup, team_att_lookup=team_att_lookup,
                draft_age_lookup=draft_age_lookup, team_games_lookup=team_games_lookup,
                adot_coef=adot_coef,
            )
            rows.append(result if result else {})

    elif position.upper() == "RB":
        from aggregation.aggregate_rb_college_stats import (
            load_all_rb_grades, aggregate_player,
        )
        all_grades = load_all_rb_grades(range(2014, 2027))

        # Build birthdate lookup from nflverse rosters
        birth_lookup = _build_rb_birth_lookup(dynasty_df)

        rows = []
        for _, row in dynasty_df.iterrows():
            birthdate = birth_lookup.get(normalize_name(row["name"]))
            result = aggregate_player(all_grades, row["name"], row["draft_year"],
                                      birthdate=birthdate)
            rows.append(result if result else {})

    else:
        raise ValueError(f"Position {position} not yet supported for college aggregation")

    matched = sum(1 for r in rows if r)
    print(f"  College stats matched: {matched}/{len(dynasty_df)}")

    agg_df = pd.DataFrame(rows)
    result = pd.concat([dynasty_df.reset_index(drop=True), agg_df], axis=1)
    return result


# ============================================================
# Step 4: Train/test split
# ============================================================

def split_train_test(df, max_train_year, holdout_years=None):
    """Split into train (resolved, <= max_train_year) and holdout (> max_train_year)."""
    train = df[
        (df["draft_year"] <= max_train_year) & (df["is_resolved"])
    ].copy()

    if holdout_years:
        holdout = df[df["draft_year"].isin(holdout_years)].copy()
    else:
        holdout = df[df["draft_year"] > max_train_year].copy()

    print(f"  Train: {len(train)} players ({train['draft_year'].min()}-{train['draft_year'].max()})")
    print(f"  Holdout: {len(holdout)} players ({holdout['draft_year'].min()}-{holdout['draft_year'].max()})")

    return train, holdout


# ============================================================
# Main pipeline
# ============================================================

def build_dataset(position, draft_start, draft_end, max_train_year,
                  holdout_years=None, ppr_path=None, output_dir=None):
    """Full pipeline: fetch picks -> dynasty values -> college stats -> split."""
    if ppr_path is None:
        ppr_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "wr_data",
            "nfl_yearly_ppr_totals_2016_2025.csv"
        )
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            f"{position.lower()}_data"
        )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    draft_years = list(range(draft_start, draft_end + 1))

    print(f"\n{'='*60}")
    print(f"Building {position.upper()} dynasty dataset")
    print(f"Draft years: {draft_start}-{draft_end}")
    print(f"Train: <={max_train_year}, Holdout: {holdout_years or f'>{max_train_year}'}")
    print(f"{'='*60}")

    # Step 1: Draft picks
    print("\n[1/4] Fetching draft picks...")
    picks = fetch_draft_picks(position, draft_years)

    # Step 2: Dynasty values
    print("\n[2/4] Computing dynasty values...")
    dynasty = compute_dynasty_values(picks, position, ppr_path)

    # Step 3: College stats
    print("\n[3/4] Aggregating college stats...")
    full = aggregate_college_stats(dynasty, position)

    # Step 4: Split
    print("\n[4/4] Splitting train/test...")
    train, holdout = split_train_test(full, max_train_year, holdout_years)

    # Save
    full_path = os.path.join(output_dir, f"{position.lower()}_dynasty_value_with_college.csv")
    train_path = os.path.join(output_dir, "outputs", f"train_{position.lower()}.csv")
    holdout_path = os.path.join(output_dir, "outputs", f"holdout_{position.lower()}.csv")

    full.to_csv(full_path, index=False)
    train.to_csv(train_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    print(f"\n  Saved full dataset:  {full_path} ({len(full)} players, {len(full.columns)} cols)")
    print(f"  Saved train split:   {train_path} ({len(train)} players)")
    print(f"  Saved holdout split: {holdout_path} ({len(holdout)} players)")

    return full, train, holdout


def main():
    parser = argparse.ArgumentParser(description="Build dynasty value dataset with train/test split")
    parser.add_argument("--position", required=True, choices=["WR", "RB", "TE", "QB"],
                        help="Position to build dataset for")
    parser.add_argument("--draft-start", type=int, default=None,
                        help="First draft year (default: WR=2016, RB=2016)")
    parser.add_argument("--draft-end", type=int, default=2024,
                        help="Last draft year (default: 2024)")
    parser.add_argument("--train-end", type=int, default=2021,
                        help="Last year for training data (default: 2021)")
    parser.add_argument("--holdout-years", type=int, nargs="+", default=None,
                        help="Specific holdout years (default: everything after train-end)")
    args = parser.parse_args()

    # Position-specific defaults
    draft_start_defaults = {"WR": 2016, "RB": 2016, "TE": 2016, "QB": 2016}
    draft_start = args.draft_start or draft_start_defaults.get(args.position, 2016)

    holdout_years = args.holdout_years or [2022, 2023, 2024]

    build_dataset(
        position=args.position,
        draft_start=draft_start,
        draft_end=args.draft_end,
        max_train_year=args.train_end,
        holdout_years=holdout_years,
    )


if __name__ == "__main__":
    main()
