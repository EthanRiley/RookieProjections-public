#!/usr/bin/env python3
"""
Model 1: XGBoost with Cumulative Link (ordinal classification).

Runs two variants:
  A) Full model (draft capital + college features)
  B) College-only model (no draft capital)

For each threshold k in {Flex, Starter, Elite, Stud, League-Winner},
trains a binary classifier P(tier >= k). Tier probabilities are recovered
by differencing adjacent cumulative probabilities.

Cross-validation: leave-one-year-out on 2016-2021, holdout on 2022-2024.
Calibration: Platt scaling on out-of-fold predictions.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0,
    "Flex": 1,
    "Starter": 2,
    "Elite": 3,
    "Stud": 4,
    "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]

COLLEGE_FEATURES = [
    "best2_yprr",
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

FEATURE_SETS = {
    "full": ["draft_capital"] + COLLEGE_FEATURES,
    "college_only": COLLEGE_FEATURES,
}

HOLDOUT_YEARS = [2022, 2023, 2024]

# --- Load data ---
all_features = ["draft_capital"] + COLLEGE_FEATURES
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)

train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

print(f"Training set: {len(train_df)} players ({sorted(train_df['draft_year'].unique())})")
print(f"Holdout set:  {len(holdout_df)} players ({sorted(holdout_df['draft_year'].unique())})")
print(f"Tier distribution (train):")
for tier_name in TIER_ORDER:
    n = (train_df["computed_tier"] == tier_name).sum()
    print(f"  {tier_name:15s} {n:3d} ({n/len(train_df):.1%})")


def cumulative_probs_to_tier_probs(cum_probs):
    n = cum_probs.shape[0]
    tier_probs = np.zeros((n, 6))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    row_sums = tier_probs.sum(axis=1, keepdims=True)
    tier_probs = tier_probs / np.where(row_sums > 0, row_sums, 1)
    return tier_probs


def evaluate_predictions(tier_probs, cum_probs, y_true, label=""):
    print(f"\n  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s} {'Pos rate':>10s}")
    for t_idx, (threshold, tlabel) in enumerate(zip(THRESHOLDS, THRESHOLD_LABELS)):
        y_bin = (y_true >= threshold).astype(int)
        pred = cum_probs[:, t_idx]
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        print(f"  {tlabel:<15s} {auc:>8.3f} {brier:>8.4f} {y_bin.mean():>10.1%}")

    y_onehot = np.zeros((len(y_true), 6))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ml = log_loss(y_onehot, tier_probs)
    mb = np.mean(np.sum((y_onehot - tier_probs) ** 2, axis=1))
    print(f"\n  Multi-class log loss:  {ml:.4f}")
    print(f"  Multi-class Brier:     {mb:.4f}")
    return ml, mb


def run_variant(variant_name, features):
    print(f"\n{'#' * 70}")
    print(f"# XGBoost Cumulative Link — {variant_name.upper()} ({len(features)} features)")
    print(f"# Features: {features}")
    print(f"{'#' * 70}")

    # --- Leave-One-Year-Out CV ---
    print("\n" + "=" * 70)
    print("LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
    print("=" * 70)

    train_years = sorted(train_df["draft_year"].unique())
    oof_cum_probs = np.zeros((len(train_df), len(THRESHOLDS)))

    for fold_year in train_years:
        fold_train = train_df[train_df["draft_year"] != fold_year]
        fold_val = train_df[train_df["draft_year"] == fold_year]

        X_tr = fold_train[features].values
        X_val = fold_val[features].values
        y_tr = fold_train["tier_ordinal"].values
        val_idx = np.where(train_df["draft_year"] == fold_year)[0]

        for t_idx, threshold in enumerate(THRESHOLDS):
            y_bin = (y_tr >= threshold).astype(int)
            pos = y_bin.sum()
            scale = (len(y_bin) - pos) / max(pos, 1)

            model = XGBClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                scale_pos_weight=scale, random_state=42, eval_metric="logloss",
            )
            model.fit(X_tr, y_bin, verbose=False)
            oof_cum_probs[val_idx, t_idx] = model.predict_proba(X_val)[:, 1]

        n_val = len(fold_val)
        n_hits = (fold_val["tier_ordinal"] >= 3).sum()
        print(f"  Fold {fold_year}: {n_val} players, {n_hits} hits")

    # Enforce monotonicity
    for i in range(len(THRESHOLDS) - 1, 0, -1):
        oof_cum_probs[:, i] = np.minimum(oof_cum_probs[:, i], oof_cum_probs[:, i - 1])
    oof_tier_probs = cumulative_probs_to_tier_probs(oof_cum_probs)

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD EVALUATION")
    print("=" * 70)
    y_train = train_df["tier_ordinal"].values
    oof_ll, oof_brier = evaluate_predictions(oof_tier_probs, oof_cum_probs, y_train)

    # --- Train final models ---
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS")
    print("=" * 70)

    X_train_full = train_df[features].values
    y_train_full = train_df["tier_ordinal"].values
    final_models = {}

    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train_full >= threshold).astype(int)
        pos = y_bin.sum()
        scale = (len(y_bin) - pos) / max(pos, 1)

        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, random_state=42, eval_metric="logloss",
        )
        min_class = min(y_bin.sum(), len(y_bin) - y_bin.sum())
        cv_folds = min(5, max(2, min_class))
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds)
        calibrated.fit(X_train_full, y_bin)
        final_models[threshold] = calibrated

    # --- Holdout ---
    print("\n" + "=" * 70)
    print(f"HOLDOUT EVALUATION ({HOLDOUT_YEARS})")
    print("=" * 70)

    X_holdout = holdout_df[features].values
    y_holdout = holdout_df["tier_ordinal"].values

    hold_cum = np.zeros((len(holdout_df), len(THRESHOLDS)))
    for t_idx, threshold in enumerate(THRESHOLDS):
        hold_cum[:, t_idx] = final_models[threshold].predict_proba(X_holdout)[:, 1]
    for i in range(len(THRESHOLDS) - 1, 0, -1):
        hold_cum[:, i] = np.minimum(hold_cum[:, i], hold_cum[:, i - 1])
    hold_tier_probs = cumulative_probs_to_tier_probs(hold_cum)

    hold_ll, hold_brier = evaluate_predictions(hold_tier_probs, hold_cum, y_holdout)

    # --- Player predictions ---
    print("\n" + "=" * 70)
    print("HOLDOUT PLAYER PREDICTIONS")
    print("=" * 70)

    pred_df = holdout_df[["name", "draft_year", "pick", "computed_tier"]].copy()
    for i, tier_name in TIER_NAMES.items():
        pred_df[f"P({tier_name})"] = hold_tier_probs[:, i].round(3)
    pred_df["predicted_tier"] = [TIER_NAMES[i] for i in hold_tier_probs.argmax(axis=1)]
    pred_df["expected_tier"] = sum(hold_tier_probs[:, i] * i for i in range(6))
    pred_df = pred_df.sort_values("expected_tier", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(pred_df.head(25).to_string(index=False))

    out_path = os.path.join(DATA_DIR, "outputs", f"xgb_{variant_name}_holdout_predictions.csv")
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return {"oof_ll": oof_ll, "oof_brier": oof_brier, "hold_ll": hold_ll, "hold_brier": hold_brier}


# ============================================================
# Run both variants
# ============================================================
results = {}
for name, feats in FEATURE_SETS.items():
    results[name] = run_variant(name, feats)

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print(f"\n  {'Metric':<25s} {'Full':>12s} {'College-Only':>14s}")
print(f"  {'-'*25} {'-'*12} {'-'*14}")
for metric in ["oof_ll", "oof_brier", "hold_ll", "hold_brier"]:
    label = metric.replace("_", " ").title()
    print(f"  {label:<25s} {results['full'][metric]:>12.4f} {results['college_only'][metric]:>14.4f}")
