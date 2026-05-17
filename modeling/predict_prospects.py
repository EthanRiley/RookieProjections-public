#!/usr/bin/env python3
"""
Generate tier probability predictions for incoming draft classes (2024, 2025, 2026).

Pipeline:
  1. Fetch draft picks from nflverse
  2. Aggregate college stats from PFF grades files
  3. Compute draft_capital and draft_age
  4. Retrain Bayesian + XGBoost models on labeled data (2018-2023)
  5. Generate 50/50 ensemble predictions (full + college-only), log-scaled DC

Outputs:
  - wr_data/outputs/prospect_predictions_2024.csv
  - wr_data/outputs/prospect_predictions_2025.csv
  - wr_data/outputs/prospect_predictions_2026.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root for imports
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

from modeling.wr_model import (
    TIER_ORDER, TIER_NAMES, TIER_COLS, COLLEGE_FEATURES, N_TIERS,
    W_BAYES, W_XGB,
    dc_log, build_catch_composite, apply_catch_composite,
    train_full_and_college, build_pred_df,
)
from aggregation.aggregate_college_stats import (
    load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")


# ---- Step 1: Load draft picks ----
def load_draft_picks(year):
    import nfl_data_py as nfl
    df = nfl.import_draft_picks([year])
    wr = df[df['category'].str.upper() == 'WR'].copy()
    wr = wr.sort_values('pick').reset_index(drop=True)
    wr = wr.rename(columns={'pfr_player_name': 'name', 'season': 'draft_year'})
    wr['draft_capital'] = np.maximum(10 - (10 / np.log(261)) * np.log(wr['pick'] + 1), 0).round(2)
    return wr[['name', 'draft_year', 'round', 'pick', 'draft_capital']].copy()


# ---- Step 2: Aggregate college stats for prospects ----
def aggregate_prospect_college_stats(prospects_df):
    all_grades = load_all_grades(range(2016, 2027))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    rows = []
    matched = 0
    unmatched = []

    for _, row in prospects_df.iterrows():
        result = aggregate_player(
            all_grades, row["name"], row["draft_year"],
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            adot_coef=adot_coef,
            team_games_lookup=team_games_lookup,
        )
        if result:
            matched += 1
        else:
            unmatched.append(row["name"])
        rows.append(result)

    print(f"  College stats matched: {matched}/{len(prospects_df)}")
    if unmatched:
        print(f"  Unmatched: {unmatched}")

    agg_df = pd.DataFrame(rows)
    result = pd.concat([prospects_df.reset_index(drop=True), agg_df], axis=1)
    return result


# ---- Step 3: Load training data ----
def load_training_data(max_year=2023):
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)

    # Re-aggregate to get peak-gated features (not in master CSV)
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    for i, (_, row) in enumerate(df.iterrows()):
        result = aggregate_player(
            all_grades, row["name"], row["draft_year"],
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            team_games_lookup=team_games_lookup,
            adot_coef=adot_coef,
        )
        for col in ["pg_yprr_graduated", "pg_catch_pct_adot_adj_graduated",
                    "career_catch_pct_adot_adj"]:
            if col in result:
                df.at[df.index[i], col] = result[col]

    # Build catch composite (all rows are training data here, no holdout)
    df["catch_composite"], z_params = build_catch_composite(df)

    all_features = ["draft_capital"] + COLLEGE_FEATURES
    df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    # Recompute draft capital with log scaling
    df["draft_capital"] = np.maximum(10 - (10 / np.log(261)) * np.log(df["pick"] + 1), 0)
    df = df[df["draft_year"] <= max_year].copy()
    return df, z_params


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    # Load training data (all labeled players 2016-2024)
    print("Loading training data...")
    train_df, z_params = load_training_data()
    print(f"  Training on {len(train_df)} players ({sorted(train_df['draft_year'].unique())})")

    for year in [2024, 2025, 2026]:
        print(f"\n{'=' * 70}")
        print(f"PREDICTING {year} DRAFT CLASS")
        print(f"{'=' * 70}")

        # Load draft picks
        print(f"\nFetching {year} draft picks...")
        prospects = load_draft_picks(year)
        print(f"  {len(prospects)} WRs drafted")

        # Aggregate college stats
        print(f"\nAggregating college stats...")
        prospects = aggregate_prospect_college_stats(prospects)

        # Build catch composite using training z-score params
        prospects["catch_composite"] = apply_catch_composite(prospects, z_params)

        # Drop prospects missing key features
        required = ["draft_capital"] + COLLEGE_FEATURES
        before = len(prospects)
        prospects = prospects.dropna(subset=required).reset_index(drop=True)
        if len(prospects) < before:
            print(f"  Dropped {before - len(prospects)} prospects with missing features")
        print(f"  {len(prospects)} prospects with complete data")

        # Train and predict
        full_probs, college_probs, _, components = train_full_and_college(train_df, prospects)

        # Build output (prospects don't have computed_tier, add placeholder)
        prospects["computed_tier"] = ""
        out = build_pred_df(prospects, full_probs, college_probs, components=components)

        # Print
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", None)

        display = out[["name", "draft_year", "pick",
                        "P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)",
                        "expected_tier", "college_expected_tier", "edge"]].copy()
        display.columns = ["Name", "Year", "Pick",
                           "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                           "E[full]", "E[college]", "Edge"]

        print(f"\n{year} PROSPECT RANKINGS")
        print("=" * 70)
        print(display.to_string(index=False))

        # Save
        out_path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_{year}.csv")
        out.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
