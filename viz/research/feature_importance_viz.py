#!/usr/bin/env python3
"""
Feature importance visualization — multi-panel chart showing:
  1. Bayesian posterior coefficients (standardized) with 94% HDI
  2. XGBoost gain-based importance
  3. Permutation importance on holdout (log-loss delta)
  4. Univariate metrics (Spearman, AUC for >=Elite)
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
OUT_DIR = os.path.join(DATA_DIR, "charts")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
THRESHOLDS = [1, 2, 3, 4, 5]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

COLLEGE_FEATURES = [
    "best1_yprr_graduated",
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

ALL_FEATURES = ["draft_capital"] + COLLEGE_FEATURES

FEATURE_LABELS = {
    "draft_capital": "Draft Capital",
    "best1_yprr_graduated": "YPRR (age-adj)",
    "career_targeted_qb_rating": "Targeted QBR",
    "best2_catch_pct_adot_adj": "Catch% (aDOT adj)",
    "best2_contested_catch_rate": "Contested Catch Rate",
    "best2_avoided_tackles_per_rec": "MTF / Reception",
}

HOLDOUT_YEARS = [2022, 2023, 2024]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"] + ALL_FEATURES).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    train = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
    holdout = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()
    return train, holdout


def get_bayesian_posteriors(train):
    """Train Bayesian ordinal model, return posterior coefficient samples."""
    scaler = StandardScaler()
    X_college = scaler.fit_transform(train[COLLEGE_FEATURES].values)
    dc = train["draft_capital"].values
    y = train["tier_ordinal"].values
    n_college = len(COLLEGE_FEATURES)

    with pm.Model():
        beta_college = pm.Normal("beta_college", mu=0.0, sigma=0.5, shape=n_college)
        beta_dc = pm.Normal("beta_dc", mu=0.5, sigma=0.3)
        eta = pt.dot(X_college, beta_college) + beta_dc * dc
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-2, 3, N_CUTPOINTS),
            sigma=1.5, shape=N_CUTPOINTS,
            transform=pm.distributions.transforms.ordered,
        )
        pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y)
        trace = pm.sample(3000, tune=2000, chains=4, cores=1,
                          random_state=42, progressbar=True, target_accept=0.9)

    # Extract posteriors — dc goes first to match ALL_FEATURES order
    beta_dc_samples = trace.posterior["beta_dc"].values.flatten()
    beta_college_samples = trace.posterior["beta_college"].values.reshape(-1, n_college)
    # Combine: [draft_capital, college_feat_1, ..., college_feat_5]
    all_samples = np.column_stack([beta_dc_samples, beta_college_samples])
    return all_samples


def get_xgb_importance(train):
    """Train XGBoost cumulative link models, return average gain importance."""
    X = train[ALL_FEATURES].values
    y = train["tier_ordinal"].values
    importances = np.zeros(len(ALL_FEATURES))

    for threshold in THRESHOLDS:
        y_bin = (y >= threshold).astype(int)
        pos = y_bin.sum()
        scale = (len(y_bin) - pos) / max(pos, 1)
        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, random_state=42, eval_metric="logloss",
            importance_type="gain",
        )
        model.fit(X, y_bin)
        importances += model.feature_importances_

    importances /= len(THRESHOLDS)
    return importances


def xgb_predict_proba(train, X_test):
    """XGBoost cumulative link predictions."""
    X_train = train[ALL_FEATURES].values
    y_train = train["tier_ordinal"].values
    cum_probs = np.zeros((X_test.shape[0], len(THRESHOLDS)))

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
        cum_probs[:, t_idx] = calibrated.predict_proba(X_test)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((X_test.shape[0], N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs /= tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def bayes_predict_proba(trace_samples, train, X_test_raw):
    """Bayesian ordinal predictions using stored posterior samples."""
    scaler = StandardScaler()
    scaler.fit(train[COLLEGE_FEATURES].values)
    X_college = scaler.transform(X_test_raw[:, 1:])  # skip draft_capital
    dc = X_test_raw[:, 0]

    n_college = len(COLLEGE_FEATURES)
    n_samples = trace_samples.shape[0]
    n_obs = X_college.shape[0]

    # We need cutpoints too — let's just use XGB for permutation importance
    # since it's more straightforward
    return None


def get_permutation_importance(train, holdout):
    """Permutation importance on holdout using ensemble log-loss."""
    X_holdout = holdout[ALL_FEATURES].values
    y_holdout = holdout["tier_ordinal"].values

    # Baseline predictions (XGBoost — faster than full ensemble)
    baseline_probs = xgb_predict_proba(train, X_holdout)
    baseline_ll = log_loss(y_holdout, baseline_probs, labels=list(range(N_TIERS)))

    importances = np.zeros(len(ALL_FEATURES))
    n_repeats = 20

    for feat_idx in range(len(ALL_FEATURES)):
        deltas = []
        for r in range(n_repeats):
            X_perm = X_holdout.copy()
            rng = np.random.RandomState(r)
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            perm_probs = xgb_predict_proba(train, X_perm)
            perm_ll = log_loss(y_holdout, perm_probs, labels=list(range(N_TIERS)))
            deltas.append(perm_ll - baseline_ll)
        importances[feat_idx] = np.mean(deltas)
        print(f"  Permutation {FEATURE_LABELS[ALL_FEATURES[feat_idx]]}: "
              f"delta LogLoss = {importances[feat_idx]:+.4f}")

    return importances, baseline_ll


def get_univariate_metrics(df):
    """Spearman correlation and >=Elite AUC for each feature."""
    y_ord = df["tier_ordinal"].values
    y_elite = (y_ord >= 3).astype(int)

    spearmans = []
    aucs = []
    for feat in ALL_FEATURES:
        x = df[feat].values
        rho, _ = spearmanr(x, y_ord)
        spearmans.append(rho)
        auc = roc_auc_score(y_elite, x)
        if auc < 0.5:
            auc = 1 - auc
        aucs.append(auc)

    return np.array(spearmans), np.array(aucs)


def make_plot(posteriors, xgb_imp, perm_imp, spearmans, aucs):
    """4-panel feature importance visualization."""
    labels = [FEATURE_LABELS[f] for f in ALL_FEATURES]
    n_feat = len(ALL_FEATURES)
    y_pos = np.arange(n_feat)

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    fig.suptitle("Feature Importance — WR Dynasty Tier Model (v9)",
                 fontsize=16, fontweight="bold", y=1.02)

    # --- Panel 1: Bayesian posteriors ---
    ax = axes[0]
    medians = np.median(posteriors, axis=0)
    hdi_low = np.percentile(posteriors, 3, axis=0)
    hdi_high = np.percentile(posteriors, 97, axis=0)

    # Sort by median for this panel but keep consistent ordering
    sort_idx = np.argsort(medians)
    colors = ["#2ca02c" if m > 0 else "#d62728" for m in medians[sort_idx]]

    ax.barh(y_pos, medians[sort_idx], color=colors, alpha=0.8, edgecolor="white", height=0.6)
    ax.errorbar(medians[sort_idx], y_pos,
                xerr=[medians[sort_idx] - hdi_low[sort_idx],
                      hdi_high[sort_idx] - medians[sort_idx]],
                fmt="none", ecolor="black", capsize=3, linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in sort_idx])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Bayesian Posterior\n(Median + 94% HDI)", fontweight="bold")

    # --- Panel 2: XGBoost gain ---
    ax = axes[1]
    sort_idx2 = np.argsort(xgb_imp)
    ax.barh(y_pos, xgb_imp[sort_idx2], color="#1f77b4", alpha=0.8,
            edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in sort_idx2])
    for i, v in enumerate(xgb_imp[sort_idx2]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Avg Gain (across 5 thresholds)")
    ax.set_title("XGBoost\nFeature Importance", fontweight="bold")

    # --- Panel 3: Permutation importance ---
    ax = axes[2]
    sort_idx3 = np.argsort(perm_imp)
    colors3 = ["#ff7f0e" if v > 0 else "#cccccc" for v in perm_imp[sort_idx3]]
    ax.barh(y_pos, perm_imp[sort_idx3], color=colors3, alpha=0.8,
            edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in sort_idx3])
    for i, v in enumerate(perm_imp[sort_idx3]):
        ax.text(max(v, 0) + 0.002, i, f"{v:+.4f}", va="center", fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Delta LogLoss (higher = more important)")
    ax.set_title("Permutation Importance\n(Holdout, 20 repeats)", fontweight="bold")

    # --- Panel 4: Univariate metrics ---
    ax = axes[3]
    sort_idx4 = np.argsort(aucs)
    width = 0.35

    ax.barh(y_pos - width/2, spearmans[sort_idx4], height=width,
            color="#9467bd", alpha=0.8, label="Spearman rho")
    ax.barh(y_pos + width/2, aucs[sort_idx4], height=width,
            color="#2ca02c", alpha=0.6, label=">=Elite AUC")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in sort_idx4])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Metric Value")
    ax.set_title("Univariate Metrics\n(Full Dataset)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    for i in range(n_feat):
        ax.text(aucs[sort_idx4[i]] + 0.01, i + width/2,
                f"{aucs[sort_idx4[i]]:.3f}", va="center", fontsize=7.5)
        ax.text(max(spearmans[sort_idx4[i]], 0) + 0.01, i - width/2,
                f"{spearmans[sort_idx4[i]]:.3f}", va="center", fontsize=7.5)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {out_path}")
    plt.close()


def main():
    print("Loading data...")
    train, holdout = load_data()
    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    # Full dataset for univariate metrics
    full = pd.concat([train, holdout])

    print("\nComputing univariate metrics...")
    spearmans, aucs = get_univariate_metrics(full)
    for i, f in enumerate(ALL_FEATURES):
        print(f"  {FEATURE_LABELS[f]:25s}  Spearman={spearmans[i]:+.3f}  AUC={aucs[i]:.3f}")

    print("\nTraining Bayesian model for posteriors...")
    posteriors = get_bayesian_posteriors(train)

    print("\nComputing XGBoost gain importance...")
    xgb_imp = get_xgb_importance(train)

    print("\nComputing permutation importance on holdout...")
    perm_imp, baseline_ll = get_permutation_importance(train, holdout)
    print(f"  Baseline holdout LogLoss: {baseline_ll:.4f}")

    print("\nGenerating visualization...")
    make_plot(posteriors, xgb_imp, perm_imp, spearmans, aucs)


if __name__ == "__main__":
    main()
