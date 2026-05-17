#!/usr/bin/env python3
"""
v12 WR Holdout Evaluation: Peak-gated features + 50/50 Ensemble.

Feature set (5 features, 4 college + draft_capital):
  - draft_capital (log-scaled)
  - pg_yprr_graduated
  - catch_composite (67% CPAA + 33% career aDOT-adj catch%)
  - best2_contested_catch_rate
  - best2_avoided_tackles_per_rec

Trains on 2018-2021, predicts 2022-2024 (same protocol as v11).

Outputs:
  - wr_data/outputs/holdout_predictions_v12.csv
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
    TIER_ORDER, TIER_NAMES, COLLEGE_FEATURES, N_TIERS,
    THRESHOLDS, THRESHOLD_LABELS,
    W_BAYES, W_XGB,
    dc_log, build_catch_composite, evaluate, compute_metrics,
    train_full_and_college, build_pred_df,
)
from aggregation.aggregate_college_stats import (
    load_all_grades, build_lookups, aggregate_player, fit_adot_regression,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")
HOLDOUT_YEARS = [2022, 2023, 2024]

# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("v12 HOLDOUT EVALUATION")
print(f"Features: draft_capital + {COLLEGE_FEATURES}")
print(f"Ensemble: {W_BAYES:.0%} Bayesian / {W_XGB:.0%} XGBoost")
print("=" * 70)

# --- Load + engineer data ---
print("\nLoading grades and aggregating features...")
all_grades = load_all_grades(range(2016, 2026))
birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
adot_coef = fit_adot_regression(all_grades)
print(f"  aDOT regression: catch% = {adot_coef[0]:.2f} * aDOT + {adot_coef[1]:.2f}")

df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)

# Re-aggregate to get peak-gated features
print("Re-aggregating with peak-gated features...")
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

# Split BEFORE computing catch composite to prevent z-score leakage
train_mask = ~df["draft_year"].isin(HOLDOUT_YEARS)
df["catch_composite"], z_params = build_catch_composite(df, train_mask=train_mask)

all_features = ["draft_capital"] + COLLEGE_FEATURES
df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)

# Recompute draft capital with log scaling
df["draft_capital"] = df["pick"].apply(dc_log)

train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

print(f"\nTraining set: {len(train_df)} players ({sorted(train_df['draft_year'].unique())})")
print(f"Holdout set:  {len(holdout_df)} players ({sorted(holdout_df['draft_year'].unique())})")
print(f"\nTier distribution (train):")
for tier_name in TIER_ORDER:
    n = (train_df["computed_tier"] == tier_name).sum()
    print(f"  {tier_name:15s} {n:3d} ({n/len(train_df):.1%})")

# --- Train models ---
full_probs, college_probs, scaler, components = train_full_and_college(train_df, holdout_df)
actual = holdout_df["tier_ordinal"].values

# --- Evaluate all models ---
print("\n" + "=" * 70)
print("v12 HOLDOUT EVALUATION (2022-2024)")
print("=" * 70)

evaluate(full_probs, actual, "ENSEMBLE FULL")
evaluate(college_probs, actual, "ENSEMBLE COLLEGE-ONLY")

# --- v11 comparison ---
print("\n" + "=" * 70)
print("v11 vs v12 COMPARISON (Ensemble Full)")
print("=" * 70)
v11_metrics = {
    "LogLoss": 0.773, "Brier": 0.340,
    ">=Elite AUC": 0.970, ">=Stud AUC": 0.941,
    ">=Starter AUC": 0.920, ">=LW AUC": 0.989,
}
m = compute_metrics(full_probs, actual)

print(f"\n  {'Metric':<20s} {'v11':>15s} {'v12':>15s} {'Delta':>10s}")
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")
print(f"  {'LogLoss':<20s} {0.773:>15.4f} {m['log_loss']:>15.4f} {m['log_loss'] - 0.773:>+10.4f}")
print(f"  {'Brier':<20s} {0.340:>15.4f} {m['brier']:>15.4f} {m['brier'] - 0.340:>+10.4f}")
for tlabel in THRESHOLD_LABELS:
    v11_key = f"{tlabel} AUC"
    v11_val = v11_metrics.get(v11_key, None)
    v12_val = m.get(f"{tlabel}_auc", float("nan"))
    if v11_val is not None:
        print(f"  {v11_key:<20s} {v11_val:>15.3f} {v12_val:>15.3f} {v12_val - v11_val:>+10.3f}")
    else:
        print(f"  {tlabel + ' AUC':<20s} {'--':>15s} {v12_val:>15.3f}")

# --- Build output ---
out = build_pred_df(holdout_df, full_probs, college_probs, components=components)

# Print
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

display = out[["name", "draft_year", "pick", "computed_tier",
               "P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)",
               "expected_tier", "college_expected_tier", "edge"]].copy()
display.columns = ["Name", "Year", "Pick", "Actual",
                    "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                    "E[full]", "E[college]", "Edge"]

print("\n" + "=" * 70)
print("HOLDOUT PREDICTIONS (sorted by E[full])")
print("=" * 70)
print(display.to_string(index=False))

# Save
out_path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_v12.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
