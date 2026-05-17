#!/usr/bin/env python3
"""Grid search the peak-gated quality gate threshold.

Tests grades_offense thresholds from 70 to 85 in 2.5 increments for
peak-gated feature selection (pg_yprr_graduated, pg_catch_pct_adot_adj_graduated).

Evaluates on holdout (2022-2024). Monkey-patches PEAK_GATED_QUALITY_GATE
in aggregate_college_stats to test each threshold.

Outputs:
  - wr_data/outputs/grade_gate_grid_search.csv
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


from modeling.predict_prospects import (
    train_xgb_predict, train_bayesian_predict,
    blend, COLLEGE_FEATURES, TIER_NAMES, TIER_ORDER,
)
import aggregation.aggregate_college_stats as agg
from aggregation.aggregate_college_stats import (
    load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
)
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")

GATES = [70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0]


def main():
    master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER)

    all_grades = load_all_grades(range(2016, 2026))
    bl, dal, tal, tgl = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    results = []

    for gate in GATES:
        print(f"\n===== Grade Gate: {gate} =====")

        # Monkey-patch the gate
        agg.PEAK_GATED_QUALITY_GATE = gate

        df = master.copy()
        for i, (_, row) in enumerate(df.iterrows()):
            result = aggregate_player(
                all_grades, row["name"], row["draft_year"],
                birth_lookup=bl, team_att_lookup=tal,
                draft_age_lookup=dal, team_games_lookup=tgl,
                adot_coef=adot_coef,
            )
            for col in ["pg_yprr_graduated", "pg_catch_pct_adot_adj_graduated"]:
                if col in result:
                    df.at[df.index[i], col] = result[col]

        df["draft_capital"] = np.maximum(
            10 - (10 / np.log(261)) * np.log(df["pick"] + 1), 0
        )

        all_features = ["draft_capital"] + COLLEGE_FEATURES
        df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
        df["tier_ordinal"] = df["tier_ordinal"].astype(int)

        train_df = df[df["draft_year"] <= 2021].copy()
        holdout_df = df[df["draft_year"] >= 2022].copy()
        print(f"  Train: {len(train_df)}, Holdout: {len(holdout_df)}")

        features = ["draft_capital"] + COLLEGE_FEATURES

        xgb = train_xgb_predict(train_df, holdout_df, features)
        print("  Bayesian...")
        bayes = train_bayesian_predict(train_df, holdout_df, features, use_draft_capital=True)
        probs = blend(bayes, xgb)

        y_true = holdout_df["tier_ordinal"].values
        ll = log_loss(y_true, probs, labels=list(range(6)))
        brier = np.mean(np.sum((probs - np.eye(6)[y_true]) ** 2, axis=1))
        elite_auc = roc_auc_score((y_true >= 3).astype(int), probs[:, 3:].sum(axis=1))

        print(f"  LL={ll:.4f}  Brier={brier:.4f}  Elite AUC={elite_auc:.4f}")

        results.append({
            "gate": gate, "n_train": len(train_df), "n_holdout": len(holdout_df),
            "log_loss": round(ll, 4), "brier": round(brier, 4),
            "elite_auc": round(elite_auc, 4),
        })

    # Reset gate
    agg.PEAK_GATED_QUALITY_GATE = 80.0

    results_df = pd.DataFrame(results)

    # Compute equal-weighted composite (normalize each to [0,1], 0=best)
    for col in ["log_loss", "brier"]:
        mn, mx = results_df[col].min(), results_df[col].max()
        results_df[f"{col}_norm"] = (results_df[col] - mn) / (mx - mn) if mx > mn else 0
    mn, mx = results_df["elite_auc"].min(), results_df["elite_auc"].max()
    results_df["elite_auc_norm"] = (mx - results_df["elite_auc"]) / (mx - mn) if mx > mn else 0

    results_df["composite"] = (
        (results_df["log_loss_norm"] + results_df["brier_norm"] + results_df["elite_auc_norm"]) / 3
    ).round(4)

    print("\n\n===== SUMMARY =====")
    print(results_df[["gate", "n_train", "n_holdout", "log_loss", "brier",
                       "elite_auc", "composite"]].to_string(index=False))
    print(f"\nBest gate: {results_df.loc[results_df['composite'].idxmin(), 'gate']}")

    csv_path = os.path.join(DATA_DIR, "outputs", "grade_gate_grid_search.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
