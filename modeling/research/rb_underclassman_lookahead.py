#!/usr/bin/env python3
"""
RB Underclassman Lookahead Analysis.

Identifies top underclassman RBs (not yet drafted) from the most recent
PFF season and projects their tier probabilities using the college-only
model variant (no draft capital available yet).

Also runs projections at mock draft capital levels (round 1, 2, 3, 4)
to show how landing spot would affect their outlook.

Outputs:
  - rb_data/outputs/rb_underclassman_lookahead_2026.csv
  - Console: ranked table with college-only tier probabilities
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

from aggregation.aggregate_rb_college_stats import (
    load_all_rb_grades, aggregate_player, normalize_name,
)
from modeling.rb_model import (
    TIER_ORDER, TIER_NAMES, COLLEGE_FEATURES, N_TIERS,
    dc_log, apply_feature_fallbacks, compute_composites, apply_composites,
    train_bayesian, train_xgb, blend, cumulative_to_tier_probs,
    W_BAYES, W_XGB,
)
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(PROJECT_ROOT, "rb_data")
MAX_TRAIN_YEAR = 2024
LATEST_SEASON = 2025

# Mock draft picks for scenario analysis
MOCK_PICKS = {
    "Rd1 (pick 20)": 20,
    "Rd2 (pick 50)": 50,
    "Rd3 (pick 80)": 80,
    "Rd4 (pick 115)": 115,
}


def load_training_data():
    """Load and prepare training data."""
    df = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"]).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    df["is_resolved"] = df["is_resolved"].astype(bool)
    df = df[df["is_resolved"]].copy()
    df["draft_capital"] = df["pick"].apply(dc_log)
    df = df[df["draft_year"] <= MAX_TRAIN_YEAR].copy()
    df = apply_feature_fallbacks(df)
    df, scaler_dict = compute_composites(df)
    return df, scaler_dict


def get_undrafted_rbs(all_grades):
    """Identify undrafted RBs with significant 2025 production."""
    # Load drafted RBs to exclude
    drafted_keys = set()
    try:
        import nfl_data_py as nfl
        for yr in [2025, 2026]:
            try:
                picks = nfl.import_draft_picks([yr])
                rbs = picks[picks["category"].str.upper() == "RB"]
                for _, r in rbs.iterrows():
                    drafted_keys.add(normalize_name(r["pfr_player_name"]))
            except Exception:
                pass
    except ImportError:
        print("  Warning: nfl_data_py not available, cannot filter drafted players")

    # Get 2025 season data
    g25 = all_grades[all_grades["grade_year"] == LATEST_SEASON].copy()
    g25["_key"] = g25["player"].apply(normalize_name)
    g25["attempts"] = pd.to_numeric(g25["attempts"], errors="coerce").fillna(0)
    g25["grades_offense"] = pd.to_numeric(g25["grades_offense"], errors="coerce").fillna(0)

    # Filter: undrafted, 100+ attempts, position = RB (exclude QBs with rushing stats)
    # QBs often show up in rushing data — filter by checking if they have scramble-heavy profiles
    undrafted = g25[~g25["_key"].isin(drafted_keys)].copy()
    eligible = undrafted[undrafted["attempts"] >= 100].copy()

    # Filter out likely QBs: scrambles > 30% of attempts is a QB signature
    if "scrambles" in eligible.columns:
        scrambles = pd.to_numeric(eligible["scrambles"], errors="coerce").fillna(0)
        scramble_rate = scrambles / eligible["attempts"]
        eligible = eligible[scramble_rate < 0.30].copy()

    # Sort by grade, take top 30
    eligible = eligible.sort_values("grades_offense", ascending=False).head(30)

    print(f"  {len(eligible)} undrafted RBs with 100+ attempts in {LATEST_SEASON}")
    return eligible


def aggregate_underclassman_stats(prospects_list, all_grades, birth_lookup):
    """Aggregate college stats for underclassman prospects."""
    rows = []
    matched = 0

    for name, team in prospects_list:
        # For underclassmen, use draft_year = LATEST_SEASON + 1 (they'd declare next year)
        birthdate = birth_lookup.get(normalize_name(name))
        result = aggregate_player(all_grades, name, LATEST_SEASON + 1, birthdate=birthdate)
        if result:
            matched += 1
            result["name"] = name
            result["team"] = team
        else:
            result = {"name": name, "team": team}
        rows.append(result)

    print(f"  College stats matched: {matched}/{len(prospects_list)}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("RB UNDERCLASSMAN LOOKAHEAD ANALYSIS")
    print("=" * 70)

    # Load training data
    print("\nLoading training data...")
    train_df, scaler_dict = load_training_data()
    all_feats = ["draft_capital"] + COLLEGE_FEATURES
    train_df = train_df.dropna(subset=all_feats).copy()
    print(f"  Training on {len(train_df)} RBs")

    # Load all grades
    print("\nLoading PFF grades...")
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

    # Identify top underclassmen
    print("\nIdentifying undrafted RBs...")
    top_rbs = get_undrafted_rbs(all_grades)

    # Build prospect list
    prospects_list = [(row["player"], row["team_name"]) for _, row in top_rbs.iterrows()]

    # Aggregate college stats
    print("\nAggregating college stats...")
    prospects = aggregate_underclassman_stats(prospects_list, all_grades, birth_lookup)

    # Apply fallbacks and composites
    prospects = apply_feature_fallbacks(prospects)
    prospects = apply_composites(prospects, scaler_dict)

    # Check feature coverage
    print(f"\nFeature coverage:")
    for feat in COLLEGE_FEATURES:
        n_valid = prospects[feat].notna().sum() if feat in prospects.columns else 0
        print(f"  {feat}: {n_valid}/{len(prospects)}")

    # Drop missing college features
    college_feats = COLLEGE_FEATURES
    before = len(prospects)
    prospects = prospects.dropna(subset=college_feats).reset_index(drop=True)
    if len(prospects) < before:
        print(f"\n  Dropped {before - len(prospects)} prospects with missing features")
    print(f"  {len(prospects)} prospects with complete data")

    if len(prospects) == 0:
        print("No prospects to predict!")
        sys.exit(1)

    # Scale college features
    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(train_df[COLLEGE_FEATURES].values)
    X_college_pred = scaler.transform(prospects[COLLEGE_FEATURES].values)
    y_train = train_df["tier_ordinal"].values

    # === College-only predictions ===
    print("\n" + "=" * 70)
    print("COLLEGE-ONLY MODEL (no draft capital)")
    print("=" * 70)

    print("\nTraining XGBoost College-Only...")
    xgb_college = train_xgb(
        train_df[COLLEGE_FEATURES].values, y_train,
        prospects[COLLEGE_FEATURES].values, random_state=42,
    )

    print("\nTraining Bayesian College-Only...")
    bayes_college = train_bayesian(
        X_college_train, None, y_train,
        X_college_pred, None, False,
        random_seed=42,
    )

    college_probs = blend(bayes_college, xgb_college)

    # Build output
    out = prospects[["name", "team"]].copy()
    for i, tier_name in TIER_NAMES.items():
        out[f"P({tier_name})"] = college_probs[:, i].round(3)
    out["predicted_tier"] = [TIER_NAMES[i] for i in college_probs.argmax(axis=1)]
    out["expected_tier"] = sum(college_probs[:, i] * i for i in range(N_TIERS))

    # Component probabilities
    for i, tier_name in TIER_NAMES.items():
        out[f"xgb_college_P({tier_name})"] = xgb_college[:, i].round(3)
    for i, tier_name in TIER_NAMES.items():
        out[f"bayes_college_P({tier_name})"] = bayes_college[:, i].round(3)

    # Add raw feature values for context
    for feat in COLLEGE_FEATURES:
        out[feat] = prospects[feat].values

    # Add composite component values for profile cards
    for comp_col in ["career_rec_yards_pg", "career_yprr", "career_grades_pass_route",
                     "career_explosive_per_att", "best2_explosive_pg"]:
        if comp_col in prospects.columns:
            out[comp_col] = prospects[comp_col].values

    out = out.sort_values("expected_tier", ascending=False).reset_index(drop=True)

    # Display
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    display = out[["name", "team",
                    "P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)",
                    "expected_tier", "predicted_tier"]].copy()
    display.columns = ["Name", "Team", "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                        "E[tier]", "Predicted"]

    print(f"\nRB UNDERCLASSMAN RANKINGS (College-Only)")
    print("=" * 70)
    print(display.to_string(index=False))

    # === Draft capital scenarios ===
    print(f"\n\n{'=' * 70}")
    print("DRAFT CAPITAL SCENARIO ANALYSIS")
    print(f"How expected tier changes with draft position")
    print(f"{'=' * 70}")

    # For top 10 underclassmen, show expected tier at each mock pick
    top10 = out.head(10).copy()
    scenario_cols = ["Name", "E[college]"]
    scenario_data = []

    for _, row in top10.iterrows():
        name = row["name"]
        p_idx = prospects[prospects["name"] == name].index[0]
        scenario_row = {"Name": name, "E[college]": round(row["expected_tier"], 2)}

        for label, pick in MOCK_PICKS.items():
            dc = dc_log(pick)
            # Create single-row prediction with draft capital
            pred_feats = prospects.iloc[[p_idx]][COLLEGE_FEATURES].values
            pred_all = np.column_stack([np.array([dc]), pred_feats])

            train_all = np.column_stack([
                train_df["draft_capital"].values.reshape(-1, 1),
                train_df[COLLEGE_FEATURES].values,
            ])

            # Quick XGBoost prediction (skip Bayesian for speed in scenarios)
            xgb_full_scenario = train_xgb(
                train_all, y_train, pred_all, random_state=42,
            )
            # Bayesian with DC
            X_cp = scaler.transform(pred_feats)
            bayes_full_scenario = train_bayesian(
                X_college_train, train_df["draft_capital"].values, y_train,
                X_cp, np.array([dc]), True, random_seed=42,
            )
            full_scenario = blend(bayes_full_scenario, xgb_full_scenario)
            et = sum(full_scenario[0, i] * i for i in range(N_TIERS))
            scenario_row[label] = round(et, 2)

        scenario_data.append(scenario_row)

    scenario_df = pd.DataFrame(scenario_data)
    print(f"\n{'Name':25s} {'College':>8s} {'Rd1(20)':>8s} {'Rd2(50)':>8s} {'Rd3(80)':>8s} {'Rd4(115)':>8s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in scenario_df.iterrows():
        print(f"{row['Name']:25s} {row['E[college]']:>8.2f} "
              f"{row.get('Rd1 (pick 20)', 0):>8.2f} {row.get('Rd2 (pick 50)', 0):>8.2f} "
              f"{row.get('Rd3 (pick 80)', 0):>8.2f} {row.get('Rd4 (pick 115)', 0):>8.2f}")

    # Save full output
    out_path = os.path.join(DATA_DIR, "outputs", "rb_underclassman_lookahead_2026.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
