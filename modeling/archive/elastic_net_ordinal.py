#!/usr/bin/env python3
"""
Model 3: Elastic Net Ordinal Logistic Regression.

Runs two variants:
  A) Full model (draft capital + college features)
  B) College-only model (no draft capital)

Cumulative link approach: K-1 binary logistic regressions with elastic net
regularization. L1 does feature selection, L2 handles correlated features.
Interpretable coefficients act as a sanity check on the ensemble.

Cross-validation: leave-one-year-out on 2016-2021, holdout on 2022-2024.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


def evaluate_predictions(tier_probs, cum_probs, y_true):
    print(f"\n  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s} {'Pos rate':>10s}")
    for t_idx, (threshold, label) in enumerate(zip(THRESHOLDS, THRESHOLD_LABELS)):
        y_bin = (y_true >= threshold).astype(int)
        pred = cum_probs[:, t_idx]
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        print(f"  {label:<15s} {auc:>8.3f} {brier:>8.4f} {y_bin.mean():>10.1%}")

    y_onehot = np.zeros((len(y_true), 6))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ml = log_loss(y_onehot, tier_probs)
    mb = np.mean(np.sum((y_onehot - tier_probs) ** 2, axis=1))
    print(f"\n  Multi-class log loss:  {ml:.4f}")
    print(f"  Multi-class Brier:     {mb:.4f}")
    return ml, mb


def run_variant(variant_name, features):
    print(f"\n{'#' * 70}")
    print(f"# Elastic Net Ordinal — {variant_name.upper()} ({len(features)} features)")
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

        # Scale features per fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(fold_train[features].values)
        X_val = scaler.transform(fold_val[features].values)
        y_tr = fold_train["tier_ordinal"].values
        val_idx = np.where(train_df["draft_year"] == fold_year)[0]

        for t_idx, threshold in enumerate(THRESHOLDS):
            y_bin = (y_tr >= threshold).astype(int)
            pos = y_bin.sum()

            if pos < 2 or (len(y_bin) - pos) < 2:
                # Too few positives/negatives — use base rate
                oof_cum_probs[val_idx, t_idx] = pos / len(y_bin)
                continue

            min_class = min(pos, len(y_bin) - pos)
            cv_folds = min(3, max(2, min_class))

            model = LogisticRegressionCV(
                penalty="elasticnet",
                l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
                Cs=10,
                cv=cv_folds,
                solver="saga",
                max_iter=10000,
                random_state=42,
                class_weight="balanced",
                scoring="roc_auc",
            )
            model.fit(X_tr, y_bin)
            oof_cum_probs[val_idx, t_idx] = model.predict_proba(X_val)[:, 1]

        n_val = len(fold_val)
        n_hits = (fold_val["tier_ordinal"] >= 3).sum()
        print(f"  Fold {fold_year}: {n_val} players, {n_hits} hits")

    # Enforce monotonicity
    for i in range(len(THRESHOLDS) - 1, 0, -1):
        oof_cum_probs[:, i] = np.minimum(oof_cum_probs[:, i], oof_cum_probs[:, i - 1])
    oof_tier_probs = cumulative_probs_to_tier_probs(oof_cum_probs)

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD EVALUATION (before calibration)")
    print("=" * 70)
    y_train = train_df["tier_ordinal"].values
    oof_ll_raw, oof_brier_raw = evaluate_predictions(oof_tier_probs, oof_cum_probs, y_train)

    # --- Temperature scaling on OOF cumulative probabilities ---
    # Find one T per threshold that minimizes Brier score on OOF
    temperatures = np.ones(len(THRESHOLDS))
    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train >= threshold).astype(int)
        raw_probs = oof_cum_probs[:, t_idx]
        # Clip to avoid log(0)
        raw_probs = np.clip(raw_probs, 1e-8, 1 - 1e-8)
        logits = np.log(raw_probs / (1 - raw_probs))

        def neg_brier(T):
            cal = 1.0 / (1.0 + np.exp(-logits / T))
            return brier_score_loss(y_bin, cal)

        result = minimize_scalar(neg_brier, bounds=(0.1, 10.0), method="bounded")
        temperatures[t_idx] = result.x

    print(f"\n  Learned temperatures: {['%.2f' % t for t in temperatures]}")

    # Apply temperature scaling to OOF
    oof_cum_cal = np.copy(oof_cum_probs)
    for t_idx in range(len(THRESHOLDS)):
        raw = np.clip(oof_cum_probs[:, t_idx], 1e-8, 1 - 1e-8)
        logits = np.log(raw / (1 - raw))
        oof_cum_cal[:, t_idx] = 1.0 / (1.0 + np.exp(-logits / temperatures[t_idx]))

    # Enforce monotonicity on calibrated
    for i in range(len(THRESHOLDS) - 1, 0, -1):
        oof_cum_cal[:, i] = np.minimum(oof_cum_cal[:, i], oof_cum_cal[:, i - 1])
    oof_tier_probs_cal = cumulative_probs_to_tier_probs(oof_cum_cal)

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD EVALUATION (after temperature scaling)")
    print("=" * 70)
    oof_ll, oof_brier = evaluate_predictions(oof_tier_probs_cal, oof_cum_cal, y_train)

    # --- Train final models ---
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS + COEFFICIENTS")
    print("=" * 70)

    final_scaler = StandardScaler()
    X_train_full = final_scaler.fit_transform(train_df[features].values)
    y_train_full = train_df["tier_ordinal"].values
    final_models = {}

    print(f"\n  {'Feature':<40s}", end="")
    for label in THRESHOLD_LABELS:
        print(f" {label:>10s}", end="")
    print()
    print(f"  {'-'*40}", end="")
    for _ in THRESHOLD_LABELS:
        print(f" {'-'*10}", end="")
    print()

    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train_full >= threshold).astype(int)
        pos = y_bin.sum()

        model = LogisticRegressionCV(
            penalty="elasticnet",
            l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
            Cs=10,
            cv=min(5, max(2, min(pos, len(y_bin) - pos))),
            solver="saga",
            max_iter=10000,
            random_state=42,
            class_weight="balanced",
            scoring="neg_log_loss",
        )
        model.fit(X_train_full, y_bin)
        final_models[threshold] = (model, final_scaler)

        if t_idx == 0:
            coef_matrix = np.zeros((len(features), len(THRESHOLDS)))
        coef_matrix[:, t_idx] = model.coef_[0]

    for i, feat in enumerate(features):
        print(f"  {feat:<40s}", end="")
        for t_idx in range(len(THRESHOLDS)):
            print(f" {coef_matrix[i, t_idx]:>+10.3f}", end="")
        print()

    print(f"\n  Best C:      ", end="")
    for threshold in THRESHOLDS:
        model = final_models[threshold][0]
        print(f" {model.C_[0]:>10.3f}", end="")
    print()
    print(f"  Best l1_ratio:", end="")
    for threshold in THRESHOLDS:
        model = final_models[threshold][0]
        print(f" {model.l1_ratio_[0]:>10.1f}", end="")
    print()

    # --- Holdout ---
    print("\n" + "=" * 70)
    print(f"HOLDOUT EVALUATION ({HOLDOUT_YEARS})")
    print("=" * 70)

    X_holdout = final_scaler.transform(holdout_df[features].values)
    y_holdout = holdout_df["tier_ordinal"].values

    hold_cum_raw = np.zeros((len(holdout_df), len(THRESHOLDS)))
    for t_idx, threshold in enumerate(THRESHOLDS):
        model = final_models[threshold][0]
        hold_cum_raw[:, t_idx] = model.predict_proba(X_holdout)[:, 1]

    # Apply temperature scaling learned from OOF
    hold_cum = np.copy(hold_cum_raw)
    for t_idx in range(len(THRESHOLDS)):
        raw = np.clip(hold_cum_raw[:, t_idx], 1e-8, 1 - 1e-8)
        logits = np.log(raw / (1 - raw))
        hold_cum[:, t_idx] = 1.0 / (1.0 + np.exp(-logits / temperatures[t_idx]))

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

    out_path = os.path.join(DATA_DIR, "outputs", f"enet_{variant_name}_holdout_predictions.csv")
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
