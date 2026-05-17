#!/usr/bin/env python3
"""
Evaluate the new feature set on 2022-2024 holdout.

Trains on 2017-2021, predicts 2022-2024 (same protocol as original models).
Produces ensemble predictions with actual tiers for comparison.

Outputs:
  - wr_data/holdout_predictions_v2.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

COLLEGE_FEATURES = [
    "best1_yprr_graduated",
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

HOLDOUT_YEARS = [2022, 2023, 2024]
W_BAYES = 0.75
W_XGB = 0.25


# --- Load data ---
all_features = ["draft_capital"] + COLLEGE_FEATURES
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
# Impute breakout features: never broke out -> max+1 for age, 0 for magnitudes
df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)

train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

print(f"Training set: {len(train_df)} players ({sorted(train_df['draft_year'].unique())})")
print(f"Holdout set:  {len(holdout_df)} players ({sorted(holdout_df['draft_year'].unique())})")
print(f"Features: {all_features}")
print(f"\nTier distribution (train):")
for tier_name in TIER_ORDER:
    n = (train_df["computed_tier"] == tier_name).sum()
    print(f"  {tier_name:15s} {n:3d} ({n/len(train_df):.1%})")


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


# --- XGBoost ---
def train_xgb(features, label):
    print(f"\nTraining XGBoost {label}...")
    X_train = train_df[features].values
    y_train = train_df["tier_ordinal"].values
    X_hold = holdout_df[features].values

    cum_probs = np.zeros((len(holdout_df), len(THRESHOLDS)))
    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train >= threshold).astype(int)
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
        calibrated.fit(X_train, y_bin)
        cum_probs[:, t_idx] = calibrated.predict_proba(X_hold)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((len(holdout_df), 6))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


# --- Bayesian ---
def train_bayesian(features, use_dc, label):
    print(f"\nTraining Bayesian {label}...")
    college_feats = [f for f in features if f != "draft_capital"]
    n_college = len(college_feats)

    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(train_df[college_feats].values)
    X_college_hold = scaler.transform(holdout_df[college_feats].values)
    y_train = train_df["tier_ordinal"].values

    dc_train = train_df["draft_capital"].values if use_dc else None
    dc_hold = holdout_df["draft_capital"].values if use_dc else None

    with pm.Model() as model:
        beta_college = pm.Normal("beta_college", mu=0.0, sigma=0.5, shape=n_college)
        eta = pt.dot(X_college_train, beta_college)
        if use_dc:
            beta_dc = pm.Normal("beta_dc", mu=0.5, sigma=0.3)
            eta = eta + beta_dc * dc_train
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-2, 3, N_CUTPOINTS),
            sigma=1.5, shape=N_CUTPOINTS,
            transform=pm.distributions.transforms.ordered,
        )
        pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y_train)

    with model:
        trace = pm.sample(
            3000, tune=2000, chains=4, cores=1,
            random_seed=42, progressbar=True, target_accept=0.9,
        )

    # Predict
    beta_college_samples = trace.posterior["beta_college"].values.reshape(-1, n_college)
    cutpoints_samples = trace.posterior["cutpoints"].values.reshape(-1, N_CUTPOINTS)
    n_samples = len(cutpoints_samples)
    n_obs = X_college_hold.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    has_dc = "beta_dc" in trace.posterior
    if has_dc:
        beta_dc_samples = trace.posterior["beta_dc"].values.flatten()

    for i in range(n_samples):
        eta = X_college_hold @ beta_college_samples[i]
        if has_dc:
            eta = eta + beta_dc_samples[i] * dc_hold
        cum_probs = 1.0 / (1.0 + np.exp(-(cutpoints_samples[i] - eta[:, None])))
        sample_probs = np.zeros((n_obs, N_TIERS))
        sample_probs[:, 0] = cum_probs[:, 0]
        for k in range(1, N_CUTPOINTS):
            sample_probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        sample_probs[:, N_TIERS - 1] = 1 - cum_probs[:, N_CUTPOINTS - 1]
        tier_probs += sample_probs

    tier_probs /= n_samples
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs /= tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


# === Run all models ===
xgb_full = train_xgb(["draft_capital"] + COLLEGE_FEATURES, "Full")
xgb_college = train_xgb(COLLEGE_FEATURES, "College-Only")
bayes_full = train_bayesian(["draft_capital"] + COLLEGE_FEATURES, True, "Full")
bayes_college = train_bayesian(COLLEGE_FEATURES, False, "College-Only")

# === Ensemble ===
def blend(b, x):
    combo = W_BAYES * b + W_XGB * x
    return combo / combo.sum(axis=1, keepdims=True)

full_probs = blend(bayes_full, xgb_full)
college_probs = blend(bayes_college, xgb_college)

actual = holdout_df["tier_ordinal"].values

# === Evaluate ===
print("\n" + "=" * 70)
print("HOLDOUT EVALUATION (2022-2024)")
print("=" * 70)

results = {}
models = {
    "Bayesian Full": bayes_full,
    "XGBoost Full": xgb_full,
    "Ensemble Full": full_probs,
    "Bayesian College": bayes_college,
    "XGBoost College": xgb_college,
    "Ensemble College": college_probs,
}

y_onehot = np.zeros((len(actual), 6))
y_onehot[np.arange(len(actual)), actual] = 1

print(f"\n  {'Model':<25s} {'LogLoss':>10s} {'Brier':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
for name, probs in models.items():
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    results[name] = {"ll": ll, "brier": brier}
    print(f"  {name:<25s} {ll:>10.4f} {brier:>10.4f}")

# Detailed evaluation for ensemble
evaluate(full_probs, actual, "ENSEMBLE FULL (detailed)")
evaluate(college_probs, actual, "ENSEMBLE COLLEGE-ONLY (detailed)")

# === Build output ===
out = holdout_df[["name", "draft_year", "pick", "computed_tier", "draft_age"]].copy()

for i, tier_name in TIER_NAMES.items():
    out[f"P({tier_name})"] = full_probs[:, i].round(3)
out["predicted_tier"] = [TIER_NAMES[i] for i in full_probs.argmax(axis=1)]
out["expected_tier"] = sum(full_probs[:, i] * i for i in range(6))

for i, tier_name in TIER_NAMES.items():
    out[f"college_P({tier_name})"] = college_probs[:, i].round(3)
out["college_predicted_tier"] = [TIER_NAMES[i] for i in college_probs.argmax(axis=1)]
out["college_expected_tier"] = sum(college_probs[:, i] * i for i in range(6))

out["edge"] = (out["college_expected_tier"] - out["expected_tier"]).round(3)
out = out.sort_values("expected_tier", ascending=False).reset_index(drop=True)

# Print
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

display = out[["name", "draft_year", "pick", "computed_tier", "draft_age",
               "P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)",
               "expected_tier", "college_expected_tier", "edge"]].copy()
display.columns = ["Name", "Year", "Pick", "Actual", "Age",
                    "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                    "E[full]", "E[college]", "Edge"]

print("\n" + "=" * 70)
print("HOLDOUT PREDICTIONS (sorted by E[full])")
print("=" * 70)
print(display.to_string(index=False))

# Save
out_path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_v2.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
