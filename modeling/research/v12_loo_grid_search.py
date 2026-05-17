#!/usr/bin/env python3
"""
Comprehensive LOO grid search across all v12 feature configurations.

Tests all combinations of:
  - YPRR variant: best1_yprr_graduated, pg_yprr_graduated
  - 5th feature (catch% slot): pg_catch_pct_adot_adj_graduated, best1_catch_pct_adot_adj_graduated,
    pg_clean_catch_rate_graduated, best1_clean_catch_rate_graduated,
    pg_catch_minus_drops_graduated, career_targeted_qb_rating, (none)
  - 6th feature: best2_catch_pct_adot_adj, best1_grades_pass_route, (none)

Outputs the best config per metric and all configs that tie.
"""

import os
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}


def _loo_ordinal_scores(d, features, years, n_tiers=6):
    thresholds = list(range(1, n_tiers))
    n = len(d)
    cum_probs = np.zeros((n, len(thresholds)))
    player_indices = np.arange(n)
    mask_predicted = np.zeros(n, dtype=bool)
    y = d["tier_num"].values
    years_arr = d["draft_year"].values

    for yr in years:
        train_mask = years_arr != yr
        test_mask = years_arr == yr
        test_idx = player_indices[test_mask]
        if test_idx.sum() == 0:
            continue
        X_train = d.iloc[train_mask][features].values
        X_test = d.iloc[test_mask][features].values
        y_train = y[train_mask]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
        for ti, thresh in enumerate(thresholds):
            y_bin = (y_train >= thresh).astype(int)
            if y_bin.sum() < 2 or y_bin.sum() == len(y_bin):
                cum_probs[test_idx, ti] = y_bin.mean()
                continue
            lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
            lr.fit(X_tr_s, y_bin)
            cum_probs[test_idx, ti] = lr.predict_proba(X_te_s)[:, 1]
        mask_predicted[test_mask] = True

    if not mask_predicted.any():
        return {}

    idx = mask_predicted
    n_pred = idx.sum()
    cp = cum_probs[idx]
    for ti in range(len(thresholds) - 1, 0, -1):
        cp[:, ti - 1] = np.maximum(cp[:, ti - 1], cp[:, ti])
    tier_probs = np.zeros((n_pred, n_tiers))
    tier_probs[:, 0] = 1 - cp[:, 0]
    for k in range(1, n_tiers - 1):
        tier_probs[:, k] = cp[:, k - 1] - cp[:, k]
    tier_probs[:, n_tiers - 1] = cp[:, -1]
    tier_probs = np.clip(tier_probs, 1e-8, 1.0)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)

    y_pred = y[idx]
    ll = -np.mean(np.log(tier_probs[np.arange(n_pred), y_pred]))
    one_hot = np.zeros((n_pred, n_tiers))
    one_hot[np.arange(n_pred), y_pred] = 1
    brier = np.mean(np.sum((tier_probs - one_hot) ** 2, axis=1))

    results = {"log_loss": ll, "brier": brier}
    for ti, thresh in enumerate(thresholds):
        y_bin = (y_pred >= thresh).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            results[f"auc_{thresh}"] = roc_auc_score(y_bin, cp[:, ti])
        else:
            results[f"auc_{thresh}"] = np.nan
    return results


def main():
    # Load and prepare data
    from aggregation.aggregate_college_stats import (
        load_all_grades, build_lookups, aggregate_player, fit_adot_regression,
    )

    print("Loading data and aggregating features...")
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_num"] = df["computed_tier"].map(TIER_ORDER)

    # Re-aggregate to get all candidate features
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
                     "best1_catch_pct_adot_adj_graduated"]:
            if col in result:
                df.at[df.index[i], col] = result[col]

    # Recompute draft capital with log scaling
    df["draft_capital"] = np.maximum(10 - (10 / np.log(261)) * np.log(df["pick"] + 1), 0)

    # Define the search space
    CORE = ["draft_capital", "best2_contested_catch_rate", "best2_avoided_tackles_per_rec"]

    yprr_options = ["best1_yprr_graduated", "pg_yprr_graduated"]

    catch_options = [
        "pg_catch_pct_adot_adj_graduated",
        "best1_catch_pct_adot_adj_graduated",
        "career_targeted_qb_rating",
        None,  # no 5th catch feature
    ]

    sixth_options = [
        "best2_catch_pct_adot_adj",
        None,  # no 6th feature
    ]

    # Build all configs
    configs = []
    for yprr, catch, sixth in product(yprr_options, catch_options, sixth_options):
        features = CORE + [yprr]
        name_parts = [yprr.replace("_graduated", "").replace("best1_", "b1_").replace("pg_", "pg_")]

        if catch is not None:
            features.append(catch)
            name_parts.append(catch.replace("_graduated", "").replace("best1_", "b1_")
                              .replace("pg_", "pg_").replace("catch_pct_adot_adj", "cpaa")
                              .replace("career_targeted_qb_rating", "QBR"))

        if sixth is not None:
            features.append(sixth)
            name_parts.append("b2_cpaa")

        name = " + ".join(name_parts)
        configs.append((name, features))

    # Check all features exist
    valid_configs = []
    for name, features in configs:
        missing = [f for f in features if f not in df.columns or df[f].isna().all()]
        if not missing:
            valid_configs.append((name, features))

    print(f"\nTotal valid configurations: {len(valid_configs)}")

    # Prepare data
    all_feat_cols = list(set(f for _, feats in valid_configs for f in feats))
    d = df.dropna(subset=["tier_num"] + all_feat_cols).copy()
    years = sorted(d["draft_year"].unique())
    print(f"Players with all features: {len(d)}")
    print(f"Years: {[int(y) for y in years]}")

    # Run LOO for each config
    print(f"\nRunning LOO for {len(valid_configs)} configurations...")
    results = []
    for i, (name, features) in enumerate(valid_configs):
        d_sub = df.dropna(subset=["tier_num"] + features).copy()
        sub_years = sorted(d_sub["draft_year"].unique())
        scores = _loo_ordinal_scores(d_sub, features, sub_years)
        if not scores:
            continue
        row = {"config": name, "n_feats": len(features), "n_players": len(d_sub)}
        row.update(scores)
        results.append(row)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(valid_configs)}] {name}: LL={scores.get('log_loss', np.nan):.3f}")

    results_df = pd.DataFrame(results)

    # Rename AUC columns for clarity
    results_df = results_df.rename(columns={
        "auc_2": "starter_auc",
        "auc_3": "elite_auc",
        "auc_4": "stud_auc",
        "auc_5": "lw_auc",
        "auc_1": "flex_auc",
    })

    # Print full results table
    print(f"\n{'=' * 130}")
    print(f"  ALL CONFIGURATIONS (sorted by LogLoss)")
    print(f"{'=' * 130}")
    print(f"  {'Config':<65s} {'#F':>3s} {'LL':>7s} {'Brier':>7s} {'Elite':>7s} {'Stud':>7s} {'Start':>7s}")
    print(f"  {'-'*65} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for _, r in results_df.sort_values("log_loss").iterrows():
        print(f"  {r['config']:<65s} {r['n_feats']:>3.0f} {r['log_loss']:>7.3f} {r['brier']:>7.3f} "
              f"{r.get('elite_auc', np.nan):>7.3f} {r.get('stud_auc', np.nan):>7.3f} "
              f"{r.get('starter_auc', np.nan):>7.3f}")

    # Find best per metric
    metrics = {
        "LogLoss": ("log_loss", "min"),
        "Brier": ("brier", "min"),
        ">=Elite AUC": ("elite_auc", "max"),
        ">=Stud AUC": ("stud_auc", "max"),
        ">=Starter AUC": ("starter_auc", "max"),
    }

    print(f"\n{'=' * 130}")
    print(f"  BEST PER METRIC (ties within 0.001)")
    print(f"{'=' * 130}")

    for label, (col, direction) in metrics.items():
        if col not in results_df.columns:
            continue
        if direction == "min":
            best_val = results_df[col].min()
            ties = results_df[results_df[col] <= best_val + 0.001]
        else:
            best_val = results_df[col].max()
            ties = results_df[results_df[col] >= best_val - 0.001]

        print(f"\n  {label} (best = {best_val:.3f}):")
        for _, r in ties.sort_values(col, ascending=(direction == "min")).iterrows():
            marker = " <-- v12" if ("pg_yprr" in r["config"] and "pg_cpaa" in r["config"]
                                     and "b2_cpaa" in r["config"]) else ""
            print(f"    {r['config']:<65s} {col}={r[col]:.3f}  "
                  f"LL={r['log_loss']:.3f} Br={r['brier']:.3f} "
                  f"Elite={r.get('elite_auc', np.nan):.3f} Stud={r.get('stud_auc', np.nan):.3f} "
                  f"Start={r.get('starter_auc', np.nan):.3f}{marker}")

    # Save results
    out_path = os.path.join(DATA_DIR, "outputs", "v12_loo_grid_search.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
