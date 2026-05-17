#!/usr/bin/env python3
"""
Evaluate WR holdout with log-scaled draft capital vs sqrt baseline.

Same protocol as evaluate_holdout.py (train 2017-2021, predict 2022-2024),
but recomputes draft_capital using log formula instead of stored sqrt values.

Log formula:  DC = 10 - (10 / ln(261)) * ln(pick + 1)
Sqrt formula: DC = 10 - 7 * sqrt(pick / 260)

Outputs:
  - wr_data/outputs/holdout_predictions_log_dc.csv
  - Side-by-side comparison of sqrt vs log holdout metrics
"""

import math
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


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


def dc_sqrt(pick):
    return 10 - 7 * math.sqrt(pick / 260)


# --- Load data ---
all_features = ["draft_capital"] + COLLEGE_FEATURES
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)

# Recompute draft capital with log scaling
df["dc_sqrt"] = df["pick"].apply(dc_sqrt)
df["dc_log"] = df["pick"].apply(dc_log)

# Verify stored values match sqrt
stored_vs_computed = np.abs(df["draft_capital"] - df["dc_sqrt"]).max()
print(f"Max diff between stored DC and computed sqrt: {stored_vs_computed:.6f}")

train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

print(f"Training set: {len(train_df)} players ({sorted(train_df['draft_year'].unique())})")
print(f"Holdout set:  {len(holdout_df)} players ({sorted(holdout_df['draft_year'].unique())})")


def evaluate_detailed(probs, y_true, label):
    print(f"\n  {label}")
    print(f"  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s} {'Pos rate':>10s}")
    aucs = {}
    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        aucs[tlabel] = auc
        print(f"  {tlabel:<15s} {auc:>8.3f} {brier:>8.4f} {y_bin.mean():>10.1%}")

    y_onehot = np.zeros((len(y_true), 6))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    print(f"\n  Multi-class log loss:  {ll:.4f}")
    print(f"  Multi-class Brier:     {brier:.4f}")
    return ll, brier, aucs


# --- XGBoost ---
def train_xgb(X_train, y_train, X_hold, label):
    print(f"\nTraining XGBoost {label}...")
    cum_probs = np.zeros((len(X_hold), len(THRESHOLDS)))
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

    tier_probs = np.zeros((len(X_hold), 6))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


# --- Bayesian ---
def train_bayesian(college_feats, dc_train, dc_hold, y_train, X_college_train, X_college_hold, label):
    print(f"\nTraining Bayesian {label}...")
    n_college = len(college_feats)
    use_dc = dc_train is not None

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


def blend(b, x):
    combo = W_BAYES * b + W_XGB * x
    return combo / combo.sum(axis=1, keepdims=True)


# === Shared data ===
y_train = train_df["tier_ordinal"].values
actual = holdout_df["tier_ordinal"].values

scaler = StandardScaler()
X_college_train = scaler.fit_transform(train_df[COLLEGE_FEATURES].values)
X_college_hold = scaler.transform(holdout_df[COLLEGE_FEATURES].values)


# ==========================================================
# SQRT (baseline) — using stored draft_capital values
# ==========================================================
print("\n" + "=" * 70)
print("TRAINING WITH SQRT DRAFT CAPITAL (baseline)")
print("=" * 70)

dc_sqrt_train = train_df["draft_capital"].values
dc_sqrt_hold = holdout_df["draft_capital"].values

X_train_sqrt = np.column_stack([dc_sqrt_train, X_college_train])
X_hold_sqrt = np.column_stack([dc_sqrt_hold, X_college_hold])

xgb_sqrt = train_xgb(X_train_sqrt, y_train, X_hold_sqrt, "Sqrt Full")
bayes_sqrt = train_bayesian(COLLEGE_FEATURES, dc_sqrt_train, dc_sqrt_hold,
                            y_train, X_college_train, X_college_hold, "Sqrt Full")
ensemble_sqrt = blend(bayes_sqrt, xgb_sqrt)


# ==========================================================
# LOG — recomputed draft capital
# ==========================================================
print("\n" + "=" * 70)
print("TRAINING WITH LOG DRAFT CAPITAL")
print("=" * 70)

dc_log_train = train_df["dc_log"].values
dc_log_hold = holdout_df["dc_log"].values

X_train_log = np.column_stack([dc_log_train, X_college_train])
X_hold_log = np.column_stack([dc_log_hold, X_college_hold])

xgb_log = train_xgb(X_train_log, y_train, X_hold_log, "Log Full")
bayes_log = train_bayesian(COLLEGE_FEATURES, dc_log_train, dc_log_hold,
                           y_train, X_college_train, X_college_hold, "Log Full")
ensemble_log = blend(bayes_log, xgb_log)


# ==========================================================
# COMPARISON
# ==========================================================
print("\n" + "=" * 70)
print("HOLDOUT COMPARISON: SQRT vs LOG DRAFT CAPITAL")
print("=" * 70)

ll_sqrt, brier_sqrt, aucs_sqrt = evaluate_detailed(ensemble_sqrt, actual, "ENSEMBLE SQRT (baseline)")
ll_log, brier_log, aucs_log = evaluate_detailed(ensemble_log, actual, "ENSEMBLE LOG")

print("\n" + "=" * 70)
print("SIDE-BY-SIDE SUMMARY")
print("=" * 70)
print(f"\n  {'Metric':<20s} {'Sqrt':>10s} {'Log':>10s} {'Delta':>10s} {'Winner':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

delta_ll = ll_log - ll_sqrt
delta_brier = brier_log - brier_sqrt
print(f"  {'LogLoss':<20s} {ll_sqrt:>10.4f} {ll_log:>10.4f} {delta_ll:>+10.4f} {'Log' if delta_ll < 0 else 'Sqrt':>10s}")
print(f"  {'Brier':<20s} {brier_sqrt:>10.4f} {brier_log:>10.4f} {delta_brier:>+10.4f} {'Log' if delta_brier < 0 else 'Sqrt':>10s}")

for label in THRESHOLD_LABELS:
    a_sqrt = aucs_sqrt[label]
    a_log = aucs_log[label]
    delta = a_log - a_sqrt
    print(f"  {label + ' AUC':<20s} {a_sqrt:>10.3f} {a_log:>10.3f} {delta:>+10.3f} {'Log' if delta > 0 else 'Sqrt':>10s}")


# === Save log DC predictions ===
out = holdout_df[["name", "draft_year", "pick", "computed_tier", "draft_age"]].copy()

for i, tier_name in TIER_NAMES.items():
    out[f"P({tier_name})"] = ensemble_log[:, i].round(3)
out["predicted_tier"] = [TIER_NAMES[i] for i in ensemble_log.argmax(axis=1)]
out["expected_tier"] = sum(ensemble_log[:, i] * i for i in range(6))

for i, tier_name in TIER_NAMES.items():
    out[f"college_P({tier_name})"] = 0.0  # college-only not rerun here
out["college_predicted_tier"] = ""
out["college_expected_tier"] = 0.0
out["edge"] = 0.0

out = out.sort_values("expected_tier", ascending=False).reset_index(drop=True)

out_path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_log_dc.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out.to_csv(out_path, index=False)
print(f"\nSaved log DC predictions to {out_path}")
