#!/usr/bin/env python3
"""
Generate a visual WR prospect profile card.

Usage:
    python viz/prospect_profile.py "Travis Hunter"
    python viz/prospect_profile.py "Makai Lemon" --year 2026
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from viz.base_profile import find_player as _find_player, make_profile as _make_profile


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

FEATURES = [
    ("draft_capital", "Draft Capital"),
    ("pg_yprr_graduated", "YPRR (peak-gated, age-adj)"),
    ("catch_composite", "Catching Composite"),
    ("best2_contested_catch_rate", "Best 2 CCR"),
    ("best2_avoided_tackles_per_rec", "Best 2 MTF / Rec"),
]

COMPOSITE_COMPONENTS = {
    "catch_composite": [
        ("pg_catch_pct_adot_adj_graduated", "CPAA"),
        ("career_catch_pct_adot_adj", "Career Catch%"),
    ],
}

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def find_player(name, year=None):
    """Search WR prediction files for the player."""
    return _find_player(
        name, year, DATA_DIR,
        prospect_pattern="prospect_predictions_{}.csv",
        holdout_file="holdout_predictions_v12.csv",
        retro_file="retro_loo_predictions.csv",
    )


def load_training_features():
    """Load training data for percentile computation."""
    from aggregation.aggregate_college_stats import (
        load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
    )
    from modeling.wr_model import build_catch_composite

    train = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    train["draft_capital"] = np.maximum(10 - (10 / np.log(261)) * np.log(train["pick"] + 1), 0)

    # Re-aggregate to get peak-gated features (not in master CSV)
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    for i, (_, row) in enumerate(train.iterrows()):
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
                train.at[train.index[i], col] = result[col]

    train["catch_composite"], _ = build_catch_composite(train)

    feat_cols = [f[0] for f in FEATURES]
    return train.dropna(subset=feat_cols)


def get_prospect_features(name, year, pick=None):
    """Reconstruct prospect features from raw data using shared aggregation module."""
    from aggregation.aggregate_college_stats import (
        load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
    )
    from modeling.wr_model import build_catch_composite, CATCH_COMPOSITE_CPAA_WEIGHT, CATCH_COMPOSITE_CAREER_WEIGHT

    all_grades = load_all_grades(range(2016, 2027))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    result = aggregate_player(
        all_grades, name, year,
        birth_lookup=birth_lookup,
        team_att_lookup=team_att_lookup,
        draft_age_lookup=draft_age_lookup,
        adot_coef=adot_coef,
        team_games_lookup=team_games_lookup,
    )

    if not result:
        return None

    # Draft capital
    if pick is not None:
        result["draft_capital"] = round(max(10 - (10 / np.log(261)) * np.log(pick + 1), 0), 2)
        result["pick"] = pick
    else:
        try:
            import nfl_data_py as nfl
            draft = nfl.import_draft_picks([year])
            wr = draft[draft["pfr_player_name"] == name]
            if len(wr) > 0:
                pick = wr.iloc[0]["pick"]
                result["draft_capital"] = round(max(10 - (10 / np.log(261)) * np.log(pick + 1), 0), 2)
                result["pick"] = pick
        except ImportError:
            pass

    # Build catch composite using training z-score params
    train = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))

    train_agg_grades = load_all_grades(range(2016, 2026))
    train_birth, train_draft_age, train_team_att, train_team_games = build_lookups(train_agg_grades)
    train_adot_coef = fit_adot_regression(train_agg_grades)

    for i, (_, row) in enumerate(train.iterrows()):
        train_result = aggregate_player(
            train_agg_grades, row["name"], row["draft_year"],
            birth_lookup=train_birth,
            team_att_lookup=train_team_att,
            draft_age_lookup=train_draft_age,
            team_games_lookup=train_team_games,
            adot_coef=train_adot_coef,
        )
        for col in ["pg_catch_pct_adot_adj_graduated", "career_catch_pct_adot_adj"]:
            if col in train_result:
                train.at[train.index[i], col] = train_result[col]

    _, z_params = build_catch_composite(train)

    cpaa = result.get("pg_catch_pct_adot_adj_graduated")
    career = result.get("career_catch_pct_adot_adj")
    if cpaa is not None and career is not None:
        result["catch_composite"] = (
            CATCH_COMPOSITE_CPAA_WEIGHT * (cpaa - z_params["cpaa_mean"]) / z_params["cpaa_std"]
            + CATCH_COMPOSITE_CAREER_WEIGHT * (career - z_params["career_mean"]) / z_params["career_std"]
        )

    return result


def make_wr_profile(player_row, class_df, year, prospect_feats, train_df):
    """Generate a WR profile card."""
    output_dir = os.path.join(PROFILE_DIR, str(year))
    _make_profile(
        player_row, class_df, year, prospect_feats, train_df,
        features=FEATURES,
        composite_components=COMPOSITE_COMPONENTS,
        position_label="WRs",
        percentile_label="Percentile vs. Historical WRs (2017-2022)",
        composite_title="Catch Composite Breakdown",
        output_dir=output_dir,
    )


def run_batch(year, top_n=10):
    """Generate profiles for the top N prospects in a given draft year."""
    path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_{year}.csv")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_v12.csv")
        if not os.path.exists(path):
            print(f"No predictions file for {year}")
            return
        class_df = pd.read_csv(path)
        class_df = class_df[class_df["draft_year"] == year].copy()
    else:
        class_df = pd.read_csv(path)

    class_df = class_df.sort_values("expected_tier", ascending=False).reset_index(drop=True)
    top = class_df.head(top_n)

    train_df = load_training_features()

    from aggregation.aggregate_college_stats import (
        load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
    )
    from modeling.wr_model import build_catch_composite, CATCH_COMPOSITE_CPAA_WEIGHT, CATCH_COMPOSITE_CAREER_WEIGHT

    all_grades = load_all_grades(range(2016, 2027))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    # Build z_params from training data for catch composite
    train_master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    train_agg_grades = load_all_grades(range(2016, 2026))
    train_birth, train_draft_age, train_team_att, train_team_games = build_lookups(train_agg_grades)
    train_adot_coef = fit_adot_regression(train_agg_grades)
    for i, (_, row) in enumerate(train_master.iterrows()):
        train_result = aggregate_player(
            train_agg_grades, row["name"], row["draft_year"],
            birth_lookup=train_birth, team_att_lookup=train_team_att,
            draft_age_lookup=train_draft_age, team_games_lookup=train_team_games,
            adot_coef=train_adot_coef,
        )
        for col in ["pg_catch_pct_adot_adj_graduated", "career_catch_pct_adot_adj"]:
            if col in train_result:
                train_master.at[train_master.index[i], col] = train_result[col]
    _, z_params = build_catch_composite(train_master)

    print(f"\n{'='*60}")
    print(f"  Generating top {top_n} profiles for {year}")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(top.iterrows()):
        name = row["name"]
        pick = int(row["pick"])
        print(f"\n  [{i+1}/{top_n}] {name} (pick {pick})")

        result = aggregate_player(
            all_grades, name, year,
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            adot_coef=adot_coef,
            team_games_lookup=team_games_lookup,
        )

        if not result:
            base_name = name.rsplit(" ", 1)[0]
            result = aggregate_player(
                all_grades, base_name, year,
                birth_lookup=birth_lookup,
                team_att_lookup=team_att_lookup,
                draft_age_lookup=draft_age_lookup,
                adot_coef=adot_coef,
                team_games_lookup=team_games_lookup,
            )

        if not result:
            print(f"    Skipped -- could not aggregate college stats")
            continue

        result["draft_capital"] = round(max(10 - (10 / np.log(261)) * np.log(pick + 1), 0), 2)
        result["pick"] = pick

        cpaa = result.get("pg_catch_pct_adot_adj_graduated")
        career = result.get("career_catch_pct_adot_adj")
        if cpaa is not None and career is not None:
            result["catch_composite"] = (
                CATCH_COMPOSITE_CPAA_WEIGHT * (cpaa - z_params["cpaa_mean"]) / z_params["cpaa_std"]
                + CATCH_COMPOSITE_CAREER_WEIGHT * (career - z_params["career_mean"]) / z_params["career_std"]
            )

        make_wr_profile(row, class_df, year, result, train_df)


def main():
    parser = argparse.ArgumentParser(description="Generate a prospect profile card")
    parser.add_argument("name", nargs="?", default=None,
                        help="Player name (partial match OK). Omit for batch mode.")
    parser.add_argument("--year", type=int, default=None,
                        help="Draft year (auto-detected if omitted)")
    parser.add_argument("--batch", action="store_true",
                        help="Generate top 10 profiles for --year (or all years if omitted)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top prospects for batch mode (default: 10)")
    parser.add_argument("--features-json", default=None,
                        help="Path to JSON file with precomputed features")
    args = parser.parse_args()

    if args.batch or args.name is None:
        years = [args.year] if args.year else [2022, 2023, 2024, 2025, 2026]
        for yr in years:
            run_batch(yr, top_n=args.top)
        return

    player_row, class_df, year = find_player(args.name, args.year)
    if player_row is None:
        print(f"Player '{args.name}' not found in prospect predictions.")
        sys.exit(1)

    print(f"Found: {player_row['name']} ({year} draft, pick {int(player_row['pick'])})")

    train_df = load_training_features()

    if args.features_json:
        import json
        with open(args.features_json) as f:
            prospect_feats = json.load(f)
    else:
        pick = int(player_row["pick"]) if pd.notna(player_row.get("pick")) else None
        prospect_feats = get_prospect_features(player_row["name"], year, pick=pick)
        if prospect_feats is None:
            base_name = player_row["name"].rsplit(" ", 1)[0]
            print(f"  Trying base name '{base_name}'...")
            prospect_feats = get_prospect_features(base_name, year, pick=pick)

    if prospect_feats is None:
        print("Could not reconstruct prospect features.")
        sys.exit(1)

    make_wr_profile(player_row, class_df, year, prospect_feats, train_df)


if __name__ == "__main__":
    main()
