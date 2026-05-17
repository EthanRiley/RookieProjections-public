#!/usr/bin/env python3
"""Generate LOO retro predictions for training-set players."""

import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")


from modeling.predict_prospects import (
    load_training_data, train_xgb_predict, train_bayesian_predict,
    blend, COLLEGE_FEATURES, TIER_NAMES,
)
from aggregation.aggregate_college_stats import (
    load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")

PLAYERS = [
    ("Ja'Marr Chase", 2021, "League-Winner"),
    ("Justin Jefferson", 2020, "League-Winner"),
    ("Amon-Ra St. Brown", 2021, "League-Winner"),
    ("Puka Nacua", 2023, "Elite"),
    ("CeeDee Lamb", 2020, "League-Winner"),
    ("Jerry Jeudy", 2020, "Starter"),
]

def main():
    train_df = load_training_data(max_year=2024)
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    # Re-aggregate peak-gated features for all training data
    for i, (_, row) in enumerate(train_df.iterrows()):
        result = aggregate_player(
            all_grades, row["name"], row["draft_year"],
            birth_lookup=birth_lookup, team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup, team_games_lookup=team_games_lookup,
            adot_coef=adot_coef,
        )
        for col in ["pg_yprr_graduated", "pg_catch_pct_adot_adj_graduated"]:
            if col in result:
                train_df.at[train_df.index[i], col] = result[col]

    train_df["draft_capital"] = np.maximum(
        10 - (10 / np.log(261)) * np.log(train_df["pick"] + 1), 0
    )

    features = ["draft_capital"] + COLLEGE_FEATURES
    all_rows = []

    for name, draft_year, actual_tier in PLAYERS:
        print(f"\n{'='*60}")
        print(f"  {name} ({draft_year})")
        print(f"{'='*60}")

        mask = (train_df["name"] == name) & (train_df["draft_year"] == draft_year)
        player_df = train_df[mask].copy()
        rest_df = train_df[~mask].copy()

        if len(player_df) == 0:
            print(f"  NOT FOUND")
            continue

        print(f"  Training on {len(rest_df)} players, predicting 1")

        xgb_full = train_xgb_predict(rest_df, player_df, features)
        xgb_col = train_xgb_predict(rest_df, player_df, COLLEGE_FEATURES)
        print("  Training Bayesian Full...")
        bayes_full = train_bayesian_predict(rest_df, player_df, features, use_draft_capital=True)
        print("  Training Bayesian College...")
        bayes_col = train_bayesian_predict(rest_df, player_df, COLLEGE_FEATURES, use_draft_capital=False)

        full_probs = blend(bayes_full, xgb_full)
        col_probs = blend(bayes_col, xgb_col)

        out = player_df[["name", "draft_year", "pick", "draft_capital"]].copy()
        out["round"] = 1
        for i, tn in TIER_NAMES.items():
            out[f"P({tn})"] = full_probs[0, i].round(3)
        out["predicted_tier"] = TIER_NAMES[full_probs.argmax(axis=1)[0]]
        out["expected_tier"] = sum(full_probs[0, i] * i for i in range(6))
        for i, tn in TIER_NAMES.items():
            out[f"college_P({tn})"] = col_probs[0, i].round(3)
        out["college_predicted_tier"] = TIER_NAMES[col_probs.argmax(axis=1)[0]]
        out["college_expected_tier"] = sum(col_probs[0, i] * i for i in range(6))
        out["edge"] = (out["college_expected_tier"] - out["expected_tier"]).round(3)
        out["computed_tier"] = actual_tier

        print(f"  E[full]={out['expected_tier'].values[0]:.3f}  "
              f"E[college]={out['college_expected_tier'].values[0]:.3f}  "
              f"edge={out['edge'].values[0]:.3f}")

        all_rows.append(out)

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "outputs", "retro_loo_predictions.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} retro predictions to {out_path}")

if __name__ == "__main__":
    main()
