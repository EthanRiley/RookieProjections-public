#!/usr/bin/env python3
"""
Generate a visual RB prospect profile card.

Usage:
    python viz/rb_prospect_profile.py "Bijan Robinson"
    python viz/rb_prospect_profile.py --batch --year 2023
    python viz/rb_prospect_profile.py --batch --holdout
    python viz/rb_prospect_profile.py --lookahead --top 15
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

from viz.base_profile import find_player as _find_player, make_profile as _make_profile


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rb_data")

FEATURES = [
    ("draft_capital", "Draft Capital"),
    ("peak2_ypa", "Peak-2 YPA"),
    ("composite_explosive", "Explosive Composite"),
    ("composite_receiving", "Receiving Composite"),
    ("peak_yac_per_att", "Peak YAC/Attempt"),
]

COLLEGE_ONLY_FEATURES = [
    ("peak2_ypa", "Peak-2 YPA"),
    ("composite_explosive", "Explosive Composite"),
    ("composite_receiving", "Receiving Composite"),
    ("peak_yac_per_att", "Peak YAC/Attempt"),
]

COMPOSITE_COMPONENTS = {
    "composite_receiving": [
        ("career_rec_yards_pg", "Rec Yards/G"),
        ("career_yprr", "YPRR"),
        ("career_grades_pass_route", "Route Grade"),
    ],
    "composite_explosive": [
        ("career_explosive_per_att", "Explosive Rate"),
        ("best2_explosive_pg", "Explosive/G"),
    ],
}

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "rb")

SUFFIXES_RE = re.compile(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


def normalize_name(name):
    n = SUFFIXES_RE.sub('', str(name)).strip()
    n = n.replace('.', '').replace("'", '').lower()
    return ' '.join(n.split())


def find_player(name, year=None):
    """Search RB prediction files for the player."""
    return _find_player(
        name, year, DATA_DIR,
        prospect_pattern="prospect_predictions_rb_{}.csv",
        holdout_file="holdout_predictions_rb_v1.csv",
    )


def load_training_features():
    """Load training data for percentile computation."""
    from sklearn.preprocessing import StandardScaler

    train = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
    train["draft_capital"] = np.maximum(10 - (10 / np.log(261)) * np.log(train["pick"] + 1), 0)

    composite_defs = {
        "composite_explosive": ["career_explosive_per_att", "best2_explosive_pg"],
        "composite_receiving": ["career_rec_yards_pg", "career_yprr", "career_grades_pass_route"],
    }
    for comp_name, feats in composite_defs.items():
        sub = train[feats].copy()
        valid = sub.notna().all(axis=1)
        if valid.sum() >= 10:
            scaler = StandardScaler()
            z = pd.DataFrame(
                scaler.fit_transform(sub[valid]),
                index=sub[valid].index, columns=feats,
            )
            train.loc[valid, comp_name] = z.mean(axis=1).round(4)

    return train


def get_prospect_features(name, year, master_df=None):
    """Get features for a prospect from the master dataset or re-aggregate."""
    if master_df is not None:
        key = normalize_name(name)
        match = master_df[master_df["name"].apply(normalize_name) == key]
        if len(match) > 0:
            row = match.iloc[0]
            result = row.to_dict()
            result["draft_capital"] = dc_log(row["pick"])
            return result

    from aggregation.aggregate_rb_college_stats import load_all_rb_grades, aggregate_player

    all_grades = load_all_rb_grades(range(2014, 2027))

    try:
        import nfl_data_py as nfl
        rosters = nfl.import_seasonal_rosters([year])
        rbs = rosters[rosters["position"] == "RB"][["player_name", "birth_date"]].drop_duplicates(subset="player_name")
        rbs = rbs[rbs["birth_date"].notna()]
        bd_lookup = {normalize_name(r["player_name"]): pd.Timestamp(r["birth_date"]) for _, r in rbs.iterrows()}
        birthdate = bd_lookup.get(normalize_name(name))
    except Exception:
        birthdate = None

    result = aggregate_player(all_grades, name, year, birthdate=birthdate)
    if not result:
        return None

    try:
        import nfl_data_py as nfl
        draft = nfl.import_draft_picks([year])
        match = draft[draft["pfr_player_name"] == name]
        if len(match) > 0:
            pick = match.iloc[0]["pick"]
            result["draft_capital"] = dc_log(pick)
            result["pick"] = pick
    except Exception:
        pass

    return result


def make_rb_profile(player_row, class_df, year, prospect_feats, train_df,
                    college_only=False):
    """Generate an RB profile card."""
    feature_list = COLLEGE_ONLY_FEATURES if college_only else FEATURES
    if college_only:
        output_dir = os.path.join(PROFILE_DIR, "lookahead")
    else:
        output_dir = os.path.join(PROFILE_DIR, str(year))

    _make_profile(
        player_row, class_df, year, prospect_feats, train_df,
        features=feature_list,
        composite_components=COMPOSITE_COMPONENTS,
        position_label="RBs",
        percentile_label="Percentile vs. Historical RBs (2016-2021)",
        composite_title="Composite Breakdown",
        output_dir=output_dir,
        college_only=college_only,
    )


def run_batch(pred_df, year, train_df, master_df, top_n=10, college_only=False):
    """Generate profiles for a set of predictions."""
    pred_df = pred_df.sort_values("expected_tier", ascending=False).reset_index(drop=True)
    top = pred_df.head(top_n)

    label = "underclassman" if college_only else f"{year}"
    print(f"\n{'='*60}")
    print(f"  Generating top {min(top_n, len(top))} RB profiles for {label}")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(top.iterrows()):
        name = row["name"]
        pick_label = f" (pick {int(row['pick'])})" if "pick" in row.index and pd.notna(row.get("pick")) else ""
        print(f"\n  [{i+1}/{min(top_n, len(top))}] {name}{pick_label}")

        if college_only:
            feats = row.to_dict()
        else:
            feats = get_prospect_features(name, year, master_df=master_df)
            if feats is None:
                print(f"    Skipped — could not get features")
                continue

            # Compute composites for this player
            from sklearn.preprocessing import StandardScaler
            composite_defs = {
                "composite_explosive": ["career_explosive_per_att", "best2_explosive_pg"],
                "composite_receiving": ["career_rec_yards_pg", "career_yprr", "career_grades_pass_route"],
            }
            for comp_name, comp_feats in composite_defs.items():
                if all(f in feats and pd.notna(feats.get(f)) for f in comp_feats):
                    train_vals = train_df[comp_feats].dropna()
                    scaler = StandardScaler()
                    scaler.fit(train_vals)
                    player_vals = np.array([[feats[f] for f in comp_feats]])
                    z = scaler.transform(player_vals)
                    feats[comp_name] = round(float(z.mean()), 4)

        make_rb_profile(row, pred_df, year, feats, train_df, college_only=college_only)


def main():
    parser = argparse.ArgumentParser(description="Generate RB prospect profile card")
    parser.add_argument("name", nargs="?", default=None,
                        help="Player name (partial match OK). Omit for batch mode.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--batch", action="store_true",
                        help="Generate top N profiles")
    parser.add_argument("--holdout", action="store_true",
                        help="Generate profiles for holdout players")
    parser.add_argument("--lookahead", action="store_true",
                        help="Generate profiles from underclassman lookahead")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    train_df = load_training_features()
    master_df = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))

    if args.lookahead:
        path = os.path.join(DATA_DIR, "outputs", "rb_underclassman_lookahead_2026.csv")
        if not os.path.exists(path):
            print(f"Lookahead file not found: {path}")
            print("Run: python modeling/research/rb_underclassman_lookahead.py")
            sys.exit(1)
        df = pd.read_csv(path)
        run_batch(df, "Lookahead", train_df, master_df, top_n=args.top, college_only=True)
        return

    if args.batch or args.holdout or args.name is None:
        if args.holdout:
            path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_rb_v1.csv")
            if not os.path.exists(path):
                print(f"Holdout file not found: {path}")
                sys.exit(1)
            df = pd.read_csv(path)
            years = [args.year] if args.year else sorted(df["draft_year"].unique())
            for yr in years:
                class_df = df[df["draft_year"] == yr].copy()
                run_batch(class_df, int(yr), train_df, master_df, top_n=args.top)
        else:
            years = [args.year] if args.year else [2024, 2025, 2026]
            for yr in years:
                path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_rb_{yr}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    run_batch(df, yr, train_df, master_df, top_n=args.top)
                else:
                    print(f"No prospect predictions for {yr}")
        return

    player_row, class_df, year = find_player(args.name, args.year)
    if player_row is None:
        print(f"Player '{args.name}' not found.")
        sys.exit(1)

    print(f"Found: {player_row['name']} ({year} draft, pick {int(player_row['pick'])})")
    feats = get_prospect_features(player_row["name"], year, master_df=master_df)
    if feats is None:
        print("Could not get features.")
        sys.exit(1)

    make_rb_profile(player_row, class_df, year, feats, train_df)


if __name__ == "__main__":
    main()
