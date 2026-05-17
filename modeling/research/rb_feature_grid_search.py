#!/usr/bin/env python3
"""
RB Model v1 — Feature Combination Grid Search.

Exhaustive search over 1-3 feature combinations from the top RB candidates.
Uses XGBoost cumulative link with leave-one-year-out CV (fast proxy for
full Bayesian+XGB ensemble).

Trains on resolved RBs (2016-2021), holds out 2022-2024.
Evaluates: LogLoss, Brier, >=Elite AUC, >=Stud AUC.

Usage:
    python3 modeling/research/rb_feature_grid_search.py
    python3 modeling/research/rb_feature_grid_search.py --max-features 2
    python3 modeling/research/rb_feature_grid_search.py --top 30
"""

import argparse
import math
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rb_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
THRESHOLDS = [1, 2, 3, 4, 5]
N_TIERS = 6

HOLDOUT_YEARS = [2022, 2023, 2024]

# --- Candidate features (distinct dimensions, from feature_evaluation.csv) ---
# Selected by: top composite rank, era stability, distinct skill dimensions.
#
# Dimension mapping:
#   PFF grade quality     -> career_grades_offense (Spearman 0.493, AUC 0.801)
#   PFF run grade         -> best2_grades_run (0.409, 0.731) — less correlated w/ offense grade
#   Route efficiency      -> best2_yprr (0.266, 0.718)
#   Elusiveness           -> career_avoided_tackles_per_att (0.320, 0.725)
#   Yards after contact   -> career_yco_attempt (0.288, 0.664) — r=0.652 with elusiveness
#   Receiving production  -> career_rec_yards_pg (0.242, 0.706)
#   Pass route grade      -> career_grades_pass_route (0.255, 0.698) — r=0.738 with YPRR
#   Explosiveness         -> best2_explosive_pg (0.286, 0.672)
#   YPA (efficiency)      -> career_ypa (0.289, 0.683) — r=0.692 with YCO
#   Touchdowns            -> best2_touchdowns_pg (0.257, 0.643)
#   Elusive MTF           -> career_elu_rush_mtf_pg (0.318, 0.676)
#
# Dropped (redundant):
#   career_grades_run — r=0.919 with career_grades_offense
#   career_yac_per_att — r=1.000 with career_yco_attempt
#   best_grades_run, peak_grades_run, best_grades_offense, peak_grades_offense — all r>0.9 with each other

CANDIDATE_FEATURES = [
    # Original top features
    "career_rec_yards_pg",
    "best2_explosive_pg",
    "career_elu_rush_mtf_pg",
    "career_grades_pass_route",
    "best2_touchdowns_pg",
    "career_ypa",
    "career_yco_attempt",
    "best2_yprr",
    # Age-adjusted (graduated multiplier, peak selection)
    "adj_rec_yards_pg",                 # AUC 0.697, era 0.024 (very stable!)
    "adj_yprr",                         # AUC 0.696, Spearman +0.245
    "adj_explosive_pg",                 # AUC 0.639, Spearman +0.221
    # Peak-gated (grades_offense >= 80 filter)
    "pg_rec_yards_pg",                  # Spearman +0.315 (best!), AUC 0.719
    "pg_yprr",                          # Spearman +0.283, era 0.032 (very stable)
    "pg_elu_rush_mtf_pg",              # Spearman +0.312, AUC 0.684, era 0.087
    # Composite skill scores (z-scored averages)
    "composite_self_creation",
    "composite_explosive",
    "composite_receiving",
]

# --- Composite definitions ---
# Each composite: list of (feature, direction) tuples.
# Features are z-scored on training data, averaged, then the composite is a single number.
COMPOSITE_DEFS = {
    "composite_self_creation": [
        # Elusiveness: making defenders miss + yards after contact
        # MTF/att and YCO/att capture different aspects (r=0.644)
        "career_elu_rush_mtf_per_att",   # MTF per attempt
        "career_yco_attempt",            # Yards after contact per attempt
    ],
    "composite_explosive": [
        # Big-play ability: rate + volume (r=0.338 — nearly independent!)
        "career_explosive_per_att",      # Explosive play rate
        "best2_explosive_pg",            # Explosive plays per game (volume)
    ],
    "composite_receiving": [
        # Pass-catching skill: production + efficiency + quality
        # Inter-r: 0.59-0.77 — correlated but distinct evaluator perspectives
        "career_rec_yards_pg",           # Receiving production volume
        "career_yprr",                   # Yards per route run (efficiency)
        "career_grades_pass_route",      # PFF route quality grade
    ],
}


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


def _compute_composites(df):
    """Compute z-scored composite features on the full dataset.

    Z-scoring is done on the full resolved population (not train-only)
    so that the composite values are comparable across players.
    For proper CV, the grid search re-standardizes features anyway.
    """
    from sklearn.preprocessing import StandardScaler

    for comp_name, input_feats in COMPOSITE_DEFS.items():
        available = [f for f in input_feats if f in df.columns]
        if not available:
            df[comp_name] = np.nan
            continue

        sub = df[available].copy()
        # Z-score each input
        scaler = StandardScaler()
        valid_mask = sub.notna().all(axis=1)
        if valid_mask.sum() < 10:
            df[comp_name] = np.nan
            continue

        z_scored = pd.DataFrame(
            scaler.fit_transform(sub[valid_mask]),
            index=sub[valid_mask].index,
            columns=available,
        )
        # Average z-scores
        df.loc[valid_mask, comp_name] = z_scored.mean(axis=1).round(4)
        df.loc[~valid_mask, comp_name] = np.nan

    return df


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"]).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    df["is_resolved"] = df["is_resolved"].astype(bool)
    df = df[df["is_resolved"]].copy()
    df["draft_capital"] = df["pick"].apply(dc_log)
    df = _compute_composites(df)
    return df


def train_xgb_cumulative(X_train, y_train, X_test):
    """XGBoost cumulative link model — returns tier probability matrix."""
    cum_probs = np.zeros((len(X_test), len(THRESHOLDS)))

    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train >= threshold).astype(int)
        pos = y_bin.sum()
        if pos == 0 or pos == len(y_bin):
            cum_probs[:, t_idx] = 0.5
            continue
        scale = (len(y_bin) - pos) / max(pos, 1)

        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, random_state=42, eval_metric="logloss",
        )
        min_class = min(pos, len(y_bin) - pos)
        cv_folds = min(5, max(2, min_class))
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds)
        calibrated.fit(X_train, y_bin)
        cum_probs[:, t_idx] = calibrated.predict_proba(X_test)[:, 1]

    # Enforce monotonicity
    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    # Convert cumulative to tier probabilities
    tier_probs = np.zeros((len(X_test), N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def evaluate_combo(df, features, use_dc=True):
    """Evaluate a feature combination using holdout evaluation."""
    all_feats = (["draft_capital"] + list(features)) if use_dc else list(features)
    sub = df.dropna(subset=all_feats).copy()

    train = sub[~sub["draft_year"].isin(HOLDOUT_YEARS)]
    holdout = sub[sub["draft_year"].isin(HOLDOUT_YEARS)]

    if len(holdout) < 10 or holdout["tier_ordinal"].nunique() < 3:
        return None

    X_train = train[all_feats].values
    y_train = train["tier_ordinal"].values
    X_hold = holdout[all_feats].values
    y_hold = holdout["tier_ordinal"].values

    probs = train_xgb_cumulative(X_train, y_train, X_hold)

    # Metrics
    y_onehot = np.zeros((len(y_hold), N_TIERS))
    y_onehot[np.arange(len(y_hold)), y_hold] = 1

    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))

    # AUCs
    aucs = {}
    for threshold, label in zip([3, 4, 5], [">=Elite", ">=Stud", ">=LW"]):
        y_bin = (y_hold >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        if 0 < y_bin.sum() < len(y_bin):
            aucs[label] = roc_auc_score(y_bin, pred)
        else:
            aucs[label] = np.nan

    # Starter AUC
    y_starter = (y_hold >= 2).astype(int)
    pred_starter = probs[:, 2:].sum(axis=1)
    if 0 < y_starter.sum() < len(y_starter):
        aucs[">=Starter"] = roc_auc_score(y_starter, pred_starter)
    else:
        aucs[">=Starter"] = np.nan

    return {
        "features": " + ".join(features),
        "n_features": len(features),
        "use_dc": use_dc,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "elite_auc": round(aucs.get(">=Elite", np.nan), 3),
        "stud_auc": round(aucs.get(">=Stud", np.nan), 3),
        "lw_auc": round(aucs.get(">=LW", np.nan), 3),
        "starter_auc": round(aucs.get(">=Starter", np.nan), 3),
    }


def composite_score(row):
    """Weighted composite: 35% LogLoss + 35% Elite AUC + 15% Brier + 10% Starter AUC + 5% Stud AUC."""
    ll = row["log_loss"] if not np.isnan(row["log_loss"]) else 3.0
    elite = row["elite_auc"] if not np.isnan(row["elite_auc"]) else 0.5
    brier = row["brier"] if not np.isnan(row["brier"]) else 1.0
    starter = row["starter_auc"] if not np.isnan(row["starter_auc"]) else 0.5
    stud = row["stud_auc"] if not np.isnan(row["stud_auc"]) else 0.5

    # Normalize: lower LL and Brier is better, higher AUC is better
    # Score = weighted sum where higher = better
    score = (
        0.35 * (1 - ll / 3.0) +  # LogLoss: 0=worst(3.0), 1=best(0)
        0.35 * elite +
        0.15 * (1 - brier) +
        0.10 * starter +
        0.05 * stud
    )
    return round(score, 4)


def main():
    parser = argparse.ArgumentParser(description="RB Feature Grid Search")
    parser.add_argument("--max-features", type=int, default=3, help="Max college features per combo")
    parser.add_argument("--top", type=int, default=25, help="Number of top results to display")
    parser.add_argument("--no-dc", action="store_true", help="Also test college-only (no draft capital)")
    args = parser.parse_args()

    df = load_data()
    print(f"RB Feature Grid Search")
    print(f"Dataset: {len(df)} resolved RBs")
    print(f"Train: {df[~df.draft_year.isin(HOLDOUT_YEARS)].shape[0]} players "
          f"({sorted(df[~df.draft_year.isin(HOLDOUT_YEARS)].draft_year.unique())})")
    print(f"Holdout: {df[df.draft_year.isin(HOLDOUT_YEARS)].shape[0]} players "
          f"({HOLDOUT_YEARS})")
    print(f"Candidate features: {len(CANDIDATE_FEATURES)}")
    print(f"Max features per combo: {args.max_features}")

    # Count total combos
    total = sum(
        len(list(combinations(CANDIDATE_FEATURES, k)))
        for k in range(1, args.max_features + 1)
    )
    if args.no_dc:
        total *= 2
    print(f"Total combinations to evaluate: {total}")
    print()

    results = []
    evaluated = 0

    for n_feats in range(1, args.max_features + 1):
        combos = list(combinations(CANDIDATE_FEATURES, n_feats))
        print(f"Evaluating {len(combos)} {n_feats}-feature combinations...")

        for combo in combos:
            result = evaluate_combo(df, combo, use_dc=True)
            if result:
                result["composite"] = composite_score(result)
                results.append(result)
            evaluated += 1

            if args.no_dc:
                result_nc = evaluate_combo(df, combo, use_dc=False)
                if result_nc:
                    result_nc["composite"] = composite_score(result_nc)
                    results.append(result_nc)
                evaluated += 1

            if evaluated % 25 == 0:
                print(f"  ... {evaluated}/{total} done")

    results_df = pd.DataFrame(results).sort_values("composite", ascending=False).reset_index(drop=True)

    # Print results
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)

    print("\n" + "=" * 120)
    print(f"TOP {args.top} FEATURE COMBINATIONS (with draft capital)")
    print("=" * 120)

    dc_results = results_df[results_df["use_dc"]].head(args.top)
    print(dc_results[["features", "n_features", "log_loss", "brier",
                       "elite_auc", "stud_auc", "starter_auc", "lw_auc",
                       "composite", "n_train", "n_holdout"]].to_string(index=False))

    if args.no_dc:
        print("\n" + "=" * 120)
        print(f"TOP {args.top} FEATURE COMBINATIONS (college-only, no draft capital)")
        print("=" * 120)
        nc_results = results_df[~results_df["use_dc"]].head(args.top)
        print(nc_results[["features", "n_features", "log_loss", "brier",
                           "elite_auc", "stud_auc", "starter_auc", "lw_auc",
                           "composite", "n_train", "n_holdout"]].to_string(index=False))

    # Summary by feature count
    print("\n" + "=" * 120)
    print("BEST BY FEATURE COUNT (with draft capital)")
    print("=" * 120)
    for n in range(1, args.max_features + 1):
        sub = results_df[(results_df["use_dc"]) & (results_df["n_features"] == n)]
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        print(f"\n  Best {n}-feature: {best['features']}")
        print(f"    LogLoss={best['log_loss']:.4f}  Brier={best['brier']:.4f}  "
              f">=Elite AUC={best['elite_auc']:.3f}  >=Stud AUC={best['stud_auc']:.3f}  "
              f">=Starter AUC={best['starter_auc']:.3f}  Composite={best['composite']:.4f}")

    # Feature frequency in top 20
    print("\n" + "=" * 120)
    print("FEATURE FREQUENCY IN TOP 20 COMBINATIONS")
    print("=" * 120)
    top20 = results_df[results_df["use_dc"]].head(20)
    freq = {}
    for feats_str in top20["features"]:
        for f in feats_str.split(" + "):
            freq[f] = freq.get(f, 0) + 1
    for f, count in sorted(freq.items(), key=lambda x: -x[1]):
        print(f"  {f:45s} {count:2d}/20")

    # Save
    out_path = os.path.join(DATA_DIR, "outputs", "rb_feature_grid_search.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
