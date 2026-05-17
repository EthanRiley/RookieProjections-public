#!/usr/bin/env python3
"""
Ensemble: Weighted average of Bayesian (75%) + XGBoost (25%) models.

Runs both full (draft capital + college) and college-only variants.
Produces final tier probability distributions for holdout players.

Reads prediction CSVs from the individual models.
Outputs: wr_data/ensemble_holdout_predictions.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

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
TIER_COLS = ["P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)"]
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]

W_BAYES = 0.75
W_XGB = 0.25


def load_model(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename)).set_index("name")


def blend(bayes_df, xgb_df):
    """Weighted average of tier probabilities, renormalized."""
    b_probs = bayes_df[TIER_COLS].values
    x_probs = xgb_df.loc[bayes_df.index, TIER_COLS].values
    combo = W_BAYES * b_probs + W_XGB * x_probs
    combo = combo / combo.sum(axis=1, keepdims=True)
    return combo


def evaluate(probs, y_true, label):
    print(f"\n  {label}")
    print(f"  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s} {'Pos rate':>10s}")
    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        print(f"  {tlabel:<15s} {auc:>8.3f} {brier:>8.4f} {y_bin.mean():>10.1%}")

    y_onehot = np.zeros((len(y_true), 6))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    print(f"\n  Multi-class log loss:  {ll:.4f}")
    print(f"  Multi-class Brier:     {brier:.4f}")
    return ll, brier


def build_pred_df(base_df, probs):
    pred = base_df[["draft_year", "pick", "computed_tier"]].copy()
    for i, tier_name in TIER_NAMES.items():
        pred[f"P({tier_name})"] = probs[:, i].round(3)
    pred["predicted_tier"] = [TIER_NAMES[i] for i in probs.argmax(axis=1)]
    pred["expected_tier"] = sum(probs[:, i] * i for i in range(6))
    return pred


# --- Load all models ---
bf = load_model("bayesian_full_holdout_predictions.csv")
bc = load_model("bayesian_college_only_holdout_predictions.csv")
xf = load_model("xgb_full_holdout_predictions.csv")
xc = load_model("xgb_college_only_holdout_predictions.csv")

actual = bf["computed_tier"].map(TIER_ORDER).values

print(f"Ensemble weights: Bayesian={W_BAYES:.0%}, XGBoost={W_XGB:.0%}")
print(f"Holdout players: {len(bf)}")

# --- Full ensemble ---
full_probs = blend(bf, xf)

# --- College-only ensemble ---
college_probs = blend(bc, xc)

# --- Evaluate ---
print("\n" + "=" * 70)
print("HOLDOUT EVALUATION")
print("=" * 70)

evaluate(full_probs, actual, "ENSEMBLE FULL")
evaluate(college_probs, actual, "ENSEMBLE COLLEGE-ONLY")

# Compare to individual models
print("\n" + "=" * 70)
print("COMPARISON TO INDIVIDUAL MODELS")
print("=" * 70)

y_onehot = np.zeros((len(actual), 6))
y_onehot[np.arange(len(actual)), actual] = 1

models = {
    "Bayesian Full": bf[TIER_COLS].values,
    "XGBoost Full": xf.loc[bf.index, TIER_COLS].values,
    "Ensemble Full": full_probs,
    "Bayesian College": bc[TIER_COLS].values,
    "XGBoost College": xc.loc[bf.index, TIER_COLS].values,
    "Ensemble College": college_probs,
}

print(f"\n  {'Model':<25s} {'LogLoss':>10s} {'Brier':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
for name, probs in models.items():
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    print(f"  {name:<25s} {ll:>10.4f} {brier:>10.4f}")

# --- Build output ---
full_pred = build_pred_df(bf, full_probs)
college_pred = build_pred_df(bc, college_probs)

# Combine into single output
out = full_pred.copy()
out = out.rename(columns={c: f"full_{c}" for c in TIER_COLS + ["predicted_tier", "expected_tier"]})

for i, tier_name in TIER_NAMES.items():
    out[f"college_P({tier_name})"] = college_probs[:, i].round(3)
out["college_predicted_tier"] = [TIER_NAMES[i] for i in college_probs.argmax(axis=1)]
out["college_expected_tier"] = sum(college_probs[:, i] * i for i in range(6))

out["edge"] = (out["college_expected_tier"] - out["full_expected_tier"]).round(3)
out = out.sort_values("full_expected_tier", ascending=False)

# --- Print ---
print("\n" + "=" * 70)
print("ENSEMBLE PREDICTIONS (sorted by full expected tier)")
print("=" * 70)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

display = out[["draft_year", "pick", "computed_tier",
               "full_P(Bust)", "full_P(Flex)", "full_P(Starter)",
               "full_P(Elite)", "full_P(Stud)", "full_P(League-Winner)",
               "full_expected_tier",
               "college_expected_tier", "edge"]].copy()

display.columns = ["Year", "Pick", "Actual",
                    "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                    "E[full]", "E[college]", "Edge"]

print(display.to_string())

# --- Save ---
out_path = os.path.join(DATA_DIR, "outputs", "ensemble_holdout_predictions.csv")
out.to_csv(out_path)
print(f"\nSaved to {out_path}")
