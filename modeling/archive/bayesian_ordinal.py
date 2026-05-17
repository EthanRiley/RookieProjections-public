#!/usr/bin/env python3
"""
Model 2: Bayesian Ordinal Regression with Draft Capital Prior.

Runs two variants:
  A) Full model (draft capital + college features)
  B) College-only model (no draft capital)

Cumulative logit (proportional odds) model.
Cross-validation: leave-one-year-out on 2016-2021, holdout on 2022-2024.
"""

import os
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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

HOLDOUT_YEARS = [2022, 2023, 2024]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

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


def build_model(college_features, n_college, draft_capital=None, y_obs=None):
    with pm.Model() as model:
        # College feature coefficients
        beta_college = pm.Normal(
            "beta_college", mu=0.0, sigma=0.5, shape=n_college,
        )

        # Latent score
        eta = pt.dot(college_features, beta_college)

        # Add draft capital if provided
        if draft_capital is not None:
            beta_dc = pm.Normal("beta_dc", mu=0.5, sigma=0.3)
            eta = eta + beta_dc * draft_capital

        # Ordered cutpoints
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(-2, 3, N_CUTPOINTS),
            sigma=1.5,
            shape=N_CUTPOINTS,
            transform=pm.distributions.transforms.ordered,
        )

        pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y_obs)

    return model


def predict_tier_probs(trace, college_features, n_college, draft_capital=None):
    beta_college = trace.posterior["beta_college"].values.reshape(-1, n_college)
    cutpoints = trace.posterior["cutpoints"].values.reshape(-1, N_CUTPOINTS)

    n_samples = len(cutpoints)
    n_obs = college_features.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    has_dc = "beta_dc" in trace.posterior
    if has_dc:
        beta_dc = trace.posterior["beta_dc"].values.flatten()

    for i in range(n_samples):
        eta = college_features @ beta_college[i]
        if has_dc:
            eta = eta + beta_dc[i] * draft_capital

        cum_probs = 1.0 / (1.0 + np.exp(-(cutpoints[i] - eta[:, None])))
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


def evaluate_predictions(tier_probs, y_true):
    print(f"\n  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s} {'Pos rate':>10s}")
    for threshold, label in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = tier_probs[:, threshold:].sum(axis=1)
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        print(f"  {label:<15s} {auc:>8.3f} {brier:>8.4f} {y_bin.mean():>10.1%}")

    y_onehot = np.zeros((len(y_true), N_TIERS))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ml = log_loss(y_onehot, tier_probs)
    mb = np.mean(np.sum((y_onehot - tier_probs) ** 2, axis=1))
    print(f"\n  Multi-class log loss:  {ml:.4f}")
    print(f"  Multi-class Brier:     {mb:.4f}")
    return ml, mb


def run_variant(variant_name, use_draft_capital):
    n_college = len(COLLEGE_FEATURES)

    print(f"\n{'#' * 70}")
    print(f"# Bayesian Ordinal — {variant_name.upper()}")
    if use_draft_capital:
        print(f"# Features: draft_capital + {COLLEGE_FEATURES}")
    else:
        print(f"# Features: {COLLEGE_FEATURES}")
    print(f"{'#' * 70}")

    # --- Leave-One-Year-Out CV ---
    print("\n" + "=" * 70)
    print("LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
    print("=" * 70)

    train_years = sorted(train_df["draft_year"].unique())
    oof_tier_probs = np.zeros((len(train_df), N_TIERS))

    for fold_year in train_years:
        print(f"\n  Fold {fold_year}...")

        fold_train_mask = train_df["draft_year"] != fold_year
        fold_val_mask = train_df["draft_year"] == fold_year

        scaler = StandardScaler()
        X_college_tr = scaler.fit_transform(train_df.loc[fold_train_mask, COLLEGE_FEATURES].values)
        X_college_val = scaler.transform(train_df.loc[fold_val_mask, COLLEGE_FEATURES].values)
        y_tr = train_df.loc[fold_train_mask, "tier_ordinal"].values

        dc_tr = train_df.loc[fold_train_mask, "draft_capital"].values if use_draft_capital else None
        dc_val = train_df.loc[fold_val_mask, "draft_capital"].values if use_draft_capital else None

        model = build_model(X_college_tr, n_college, draft_capital=dc_tr, y_obs=y_tr)
        with model:
            trace = pm.sample(
                2000, tune=1500, chains=2, cores=1,
                random_seed=42, progressbar=False, target_accept=0.9,
            )

        val_probs = predict_tier_probs(trace, X_college_val, n_college, draft_capital=dc_val)
        val_idx = np.where(fold_val_mask.values)[0]
        oof_tier_probs[val_idx] = val_probs

        n_val = fold_val_mask.sum()
        n_hits = (train_df.loc[fold_val_mask, "tier_ordinal"] >= 3).sum()
        print(f"    {n_val} players, {n_hits} hits")

    # OOF evaluation
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD EVALUATION")
    print("=" * 70)
    y_train = train_df["tier_ordinal"].values
    oof_ll, oof_brier = evaluate_predictions(oof_tier_probs, y_train)

    # --- Train final model ---
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL (full training set)")
    print("=" * 70)

    final_scaler = StandardScaler()
    X_college_full = final_scaler.fit_transform(train_df[COLLEGE_FEATURES].values)
    y_train_full = train_df["tier_ordinal"].values
    dc_full = train_df["draft_capital"].values if use_draft_capital else None

    final_model = build_model(X_college_full, n_college, draft_capital=dc_full, y_obs=y_train_full)
    with final_model:
        final_trace = pm.sample(
            3000, tune=2000, chains=4, cores=1,
            random_seed=42, progressbar=True, target_accept=0.9,
        )

    # Coefficient summary
    print("\nCoefficient summary:")
    var_names = ["beta_college", "cutpoints"]
    if use_draft_capital:
        var_names = ["beta_dc"] + var_names
    summary = az.summary(final_trace, var_names=var_names)
    college_labels = {f"beta_college[{i}]": f"beta_{feat}" for i, feat in enumerate(COLLEGE_FEATURES)}
    summary = summary.rename(index=college_labels)
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())

    # --- Holdout ---
    print("\n" + "=" * 70)
    print(f"HOLDOUT EVALUATION ({HOLDOUT_YEARS})")
    print("=" * 70)

    X_college_hold = final_scaler.transform(holdout_df[COLLEGE_FEATURES].values)
    y_holdout = holdout_df["tier_ordinal"].values
    dc_hold = holdout_df["draft_capital"].values if use_draft_capital else None

    hold_probs = predict_tier_probs(final_trace, X_college_hold, n_college, draft_capital=dc_hold)
    hold_ll, hold_brier = evaluate_predictions(hold_probs, y_holdout)

    # --- Player predictions ---
    print("\n" + "=" * 70)
    print("HOLDOUT PLAYER PREDICTIONS")
    print("=" * 70)

    pred_df = holdout_df[["name", "draft_year", "pick", "computed_tier"]].copy()
    for i, tier_name in TIER_NAMES.items():
        pred_df[f"P({tier_name})"] = hold_probs[:, i].round(3)
    pred_df["predicted_tier"] = [TIER_NAMES[i] for i in hold_probs.argmax(axis=1)]
    pred_df["expected_tier"] = sum(hold_probs[:, i] * i for i in range(N_TIERS))
    pred_df = pred_df.sort_values("expected_tier", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(pred_df.head(25).to_string(index=False))

    out_path = os.path.join(DATA_DIR, "outputs", f"bayesian_{variant_name}_holdout_predictions.csv")
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return {"oof_ll": oof_ll, "oof_brier": oof_brier, "hold_ll": hold_ll, "hold_brier": hold_brier}


# ============================================================
# Run both variants
# ============================================================
results = {}
results["full"] = run_variant("full", use_draft_capital=True)
results["college_only"] = run_variant("college_only", use_draft_capital=False)

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
