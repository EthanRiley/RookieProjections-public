#!/usr/bin/env python3
"""
Generate tier probability predictions for incoming RB draft classes.

Pipeline:
  1. Fetch draft picks from nflverse
  2. Aggregate college stats from PFF grades files
  3. Compute draft_capital, composites
  4. Retrain Bayesian + XGBoost models on labeled data (up to MAX_TRAIN_YEAR)
  5. Generate 30/70 ensemble predictions (full + college-only), log-scaled DC

Outputs:
  - rb_data/outputs/prospect_predictions_rb_{year}.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from aggregation.aggregate_rb_college_stats import (
    load_all_rb_grades, aggregate_player, normalize_name,
)
from modeling.rb_model import (
    TIER_ORDER, TIER_NAMES, COLLEGE_FEATURES, N_TIERS,
    dc_log, apply_feature_fallbacks, compute_composites, apply_composites,
    train_full_and_college, build_pred_df,
)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rb_data")

MAX_TRAIN_YEAR = 2024


# ---- Load draft picks ----
def load_draft_picks(year):
    import nfl_data_py as nfl
    df = nfl.import_draft_picks([year])
    rb = df[df['category'].str.upper() == 'RB'].copy()
    rb = rb.sort_values('pick').reset_index(drop=True)
    rb = rb.rename(columns={'pfr_player_name': 'name', 'season': 'draft_year'})
    rb['draft_capital'] = rb['pick'].apply(dc_log).round(2)
    return rb[['name', 'draft_year', 'round', 'pick', 'draft_capital']].copy()


# ---- Aggregate college stats for prospects ----
def aggregate_prospect_stats(prospects_df, all_grades, birth_lookup):
    rows = []
    matched = 0
    unmatched = []

    for _, row in prospects_df.iterrows():
        birthdate = birth_lookup.get(normalize_name(row["name"]))
        result = aggregate_player(all_grades, row["name"], row["draft_year"], birthdate=birthdate)
        if result:
            matched += 1
        else:
            unmatched.append(row["name"])
        rows.append(result if result else {})

    print(f"  College stats matched: {matched}/{len(prospects_df)}")
    if unmatched:
        print(f"  Unmatched: {unmatched}")

    agg_df = pd.DataFrame(rows)
    result = pd.concat([prospects_df.reset_index(drop=True), agg_df], axis=1)
    return result


# ---- Load training data ----
def load_training_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"]).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    df["is_resolved"] = df["is_resolved"].astype(bool)
    df = df[df["is_resolved"]].copy()
    df["draft_capital"] = df["pick"].apply(dc_log)
    df = df[df["draft_year"] <= MAX_TRAIN_YEAR].copy()

    # Apply peak2_ypa fallback
    df = apply_feature_fallbacks(df)

    # Compute composites (all rows are training data here)
    df, scaler_dict = compute_composites(df)
    return df, scaler_dict


# ==================================================================
if __name__ == "__main__":
    print("Loading training data...")
    train_df, scaler_dict = load_training_data()

    # Drop rows missing model features
    all_feats = ["draft_capital"] + COLLEGE_FEATURES
    train_df = train_df.dropna(subset=all_feats).copy()
    print(f"  Training on {len(train_df)} RBs (draft years {sorted(train_df['draft_year'].unique())})")

    # Load all grades and birthdates once
    print("Loading PFF grades...")
    all_grades = load_all_rb_grades(range(2014, 2027))
    print(f"  {len(all_grades)} player-seasons")

    # Build birthdate lookup
    birth_lookup = {}
    try:
        import nfl_data_py as nfl
        for yr in [2024, 2025, 2026]:
            try:
                rosters = nfl.import_seasonal_rosters([yr])
                rbs = rosters[rosters["position"] == "RB"][["player_name", "birth_date"]].drop_duplicates(subset="player_name")
                rbs = rbs[rbs["birth_date"].notna()]
                for _, r in rbs.iterrows():
                    key = normalize_name(r["player_name"])
                    if key not in birth_lookup:
                        birth_lookup[key] = pd.Timestamp(r["birth_date"])
            except Exception:
                pass
        print(f"  {len(birth_lookup)} birthdates loaded")
    except Exception:
        print("  Warning: could not load birthdates")

    for year in [2025, 2026]:
        print(f"\n{'=' * 70}")
        print(f"PREDICTING {year} RB DRAFT CLASS")
        print(f"{'=' * 70}")

        # Load draft picks
        print(f"\nFetching {year} draft picks...")
        try:
            prospects = load_draft_picks(year)
        except Exception as e:
            print(f"  No draft data for {year}: {e}")
            continue
        print(f"  {len(prospects)} RBs drafted")

        # Aggregate college stats
        print(f"\nAggregating college stats...")
        prospects = aggregate_prospect_stats(prospects, all_grades, birth_lookup)

        # Apply peak2_ypa fallback
        prospects = apply_feature_fallbacks(prospects)

        # Compute composites using training scalers
        prospects = apply_composites(prospects, scaler_dict)

        # Drop missing
        required = all_feats
        before = len(prospects)
        prospects = prospects.dropna(subset=required).reset_index(drop=True)
        if len(prospects) < before:
            print(f"  Dropped {before - len(prospects)} prospects with missing features")
        print(f"  {len(prospects)} prospects with complete data")

        if len(prospects) == 0:
            print("  No prospects to predict!")
            continue

        # Train and predict
        full_probs, college_probs, _, components = train_full_and_college(train_df, prospects)

        # Output
        prospects["computed_tier"] = ""
        out = build_pred_df(prospects, full_probs, college_probs, components=components)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", None)

        display = out[["name", "draft_year", "pick",
                        "P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)",
                        "expected_tier", "college_expected_tier", "edge"]].copy()
        display.columns = ["Name", "Year", "Pick",
                           "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                           "E[full]", "E[college]", "Edge"]

        print(f"\n{year} RB PROSPECT RANKINGS")
        print("=" * 70)
        print(display.to_string(index=False))

        out_path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_rb_{year}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
