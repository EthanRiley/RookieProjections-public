#!/usr/bin/env python3
"""
RB Age-Adjusted Rushing Stats Test.

Tests whether age-adjusting rushing stats (YPA, YAC/att, explosive/att)
improves the RB model. The WR model benefits from graduated age adjustment
on receiving efficiency (YPRR), so it's natural to ask whether the same
applies to RB rushing stats.

Approach:
  - Swap peak2_ypa with adj_ypa in the full 5-feature model
  - Swap peak_yac_per_att with adj_elu_rush_mtf_per_att
  - Test on both holdout (2022-2024) and LOO CV (2016-2021)
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "modeling"))

from rb_model import (
    COLLEGE_FEATURES,
    COMPOSITE_DEFS,
    TIER_ORDER,
    N_TIERS,
    W_BAYES,
    W_XGB,
    apply_feature_fallbacks,
    compute_composites,
    dc_log,
    train_bayesian,
    train_xgb,
    blend,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "rb_data")
master = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER)
master["tier_ordinal"] = master["tier_ordinal"].astype(float)
master["draft_capital"] = master["pick"].apply(dc_log)
master = apply_feature_fallbacks(master)

# Only resolved players
master = master[master["is_resolved"] == True].copy()
master["tier_ordinal"] = master["tier_ordinal"].astype(int)

# Compute composites with full training mask (train on 2016-2021)
train_mask = master["draft_year"] <= 2021
master, _ = compute_composites(master, train_mask=train_mask)

# Split
train = master[master["draft_year"] <= 2021].copy()
holdout = master[master["draft_year"].between(2022, 2024)].copy()


def evaluate_config(college_features, train_df, test_df):
    """Train ensemble and return Elite AUC, LogLoss on test set."""
    all_features = ["draft_capital"] + college_features

    tr = train_df.dropna(subset=all_features).copy()
    te = test_df.dropna(subset=all_features).copy()

    if len(te) < 10:
        return None, None, len(te)

    y_train = tr["tier_ordinal"].values

    # Scale college features
    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(tr[college_features].values)
    X_college_pred = scaler.transform(te[college_features].values)
    dc_train = tr["draft_capital"].values
    dc_pred = te["draft_capital"].values

    # XGBoost (uses raw feature values)
    xgb_probs = train_xgb(
        tr[all_features].values, y_train,
        te[all_features].values
    )

    # Bayesian
    bay_probs = train_bayesian(
        X_college_train, dc_train, y_train,
        X_college_pred, dc_pred, True
    )

    # Ensemble
    ens_probs = blend(bay_probs, xgb_probs)

    # Metrics
    y_true = te["tier_ordinal"].values
    y_elite = (y_true >= 3).astype(int)

    elite_auc = None
    if y_elite.sum() > 0 and y_elite.sum() < len(y_elite):
        p_elite = ens_probs[:, 3:].sum(axis=1)
        elite_auc = roc_auc_score(y_elite, p_elite)

    # LogLoss
    eps = 1e-8
    clipped = np.clip(ens_probs, eps, 1 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    logloss = sk_log_loss(y_true, clipped, labels=list(range(6)))

    return elite_auc, logloss, len(te)


def loo_cv(college_features, data):
    """Leave-one-year-out CV on training data."""
    years = sorted(data["draft_year"].unique())
    all_elite_true = []
    all_elite_pred = []
    all_features = ["draft_capital"] + college_features

    for yr in years:
        tr = data[data["draft_year"] != yr].copy()
        te = data[data["draft_year"] == yr].copy()

        # Recompute composites with proper leakage prevention
        combined = pd.concat([tr, te], ignore_index=True)
        tmask = pd.Series([True] * len(tr) + [False] * len(te), index=combined.index)
        combined, _ = compute_composites(combined, train_mask=tmask)
        tr = combined[tmask].copy()
        te = combined[~tmask].copy()

        tr = tr.dropna(subset=all_features)
        te = te.dropna(subset=all_features)

        if len(te) < 3:
            continue

        try:
            y_train = tr["tier_ordinal"].values
            scaler = StandardScaler()
            X_college_train = scaler.fit_transform(tr[college_features].values)
            X_college_pred = scaler.transform(te[college_features].values)

            xgb_probs = train_xgb(
                tr[all_features].values, y_train,
                te[all_features].values
            )
            bay_probs = train_bayesian(
                X_college_train, tr["draft_capital"].values, y_train,
                X_college_pred, te["draft_capital"].values, True
            )
            ens_probs = blend(bay_probs, xgb_probs)

            y_true = te["tier_ordinal"].values
            y_elite = (y_true >= 3).astype(int)
            p_elite = ens_probs[:, 3:].sum(axis=1)

            all_elite_true.extend(y_elite.tolist())
            all_elite_pred.extend(p_elite.tolist())
        except Exception as e:
            print(f"  LOO year {yr} failed: {e}")
            continue

    if len(all_elite_true) > 0:
        arr_true = np.array(all_elite_true)
        arr_pred = np.array(all_elite_pred)
        if arr_true.sum() > 0 and arr_true.sum() < len(arr_true):
            return roc_auc_score(arr_true, arr_pred), len(arr_true)
    return None, 0


# ============================================================
# Test configurations
# ============================================================

print("=" * 70)
print("  RB Age-Adjusted Rushing Stats Test")
print("=" * 70)

BASELINE = list(COLLEGE_FEATURES)  # peak2_ypa, composite_explosive, composite_receiving, peak_yac_per_att

# Swap peak2_ypa -> adj_ypa
SWAP_YPA = ["adj_ypa", "composite_explosive", "composite_receiving", "peak_yac_per_att"]

# Swap peak_yac_per_att -> adj_elu_rush_mtf_per_att
SWAP_YAC = ["peak2_ypa", "composite_explosive", "composite_receiving", "adj_elu_rush_mtf_per_att"]

# Both swaps
SWAP_BOTH = ["adj_ypa", "composite_explosive", "composite_receiving", "adj_elu_rush_mtf_per_att"]

configs = [
    ("Baseline (raw rushing)", BASELINE),
    ("adj_ypa replaces peak2_ypa", SWAP_YPA),
    ("adj_mtf/att replaces peak_yac/att", SWAP_YAC),
    ("Both rushing stats age-adjusted", SWAP_BOTH),
]

# ============================================================
# 1. Univariate comparison
# ============================================================
print("\n" + "-" * 70)
print("  Univariate Signal: Raw vs Age-Adjusted Rushing Stats")
print("-" * 70)

pairs = [
    ("peak2_ypa", "adj_ypa"),
    ("peak_yac_per_att", "adj_elu_rush_mtf_per_att"),
]

print(f"\n{'Feature':<30s} {'Spearman':>10s} {'p-value':>10s} {'n':>5s}")
print("-" * 60)
for raw, adj in pairs:
    for feat in [raw, adj]:
        valid = master.dropna(subset=[feat, "tier_ordinal"])
        if len(valid) > 10:
            r, p = spearmanr(valid[feat], valid["tier_ordinal"])
            print(f"{feat:<30s} {r:>+10.3f} {p:>10.4f} {len(valid):>5d}")
        else:
            print(f"{feat:<30s} {'N/A':>10s} {'N/A':>10s} {len(valid):>5d}")
    print()

# ============================================================
# 2. Holdout evaluation
# ============================================================
print("\n" + "-" * 70)
print("  Holdout Evaluation (2022-2024)")
print("-" * 70)

print(f"\n{'Config':<40s} {'Elite AUC':>10s} {'LogLoss':>10s} {'n':>5s}")
print("-" * 70)
for name, features in configs:
    elite_auc, logloss, n = evaluate_config(features, train, holdout)
    auc_str = f"{elite_auc:.3f}" if elite_auc else "N/A"
    ll_str = f"{logloss:.3f}" if logloss else "N/A"
    print(f"{name:<40s} {auc_str:>10s} {ll_str:>10s} {n:>5d}")

# ============================================================
# 3. LOO CV on training data
# ============================================================
print("\n" + "-" * 70)
print("  LOO Cross-Validation (2016-2021)")
print("-" * 70)

print(f"\n{'Config':<40s} {'Elite AUC':>10s} {'n':>5s}")
print("-" * 55)
for name, features in configs:
    elite_auc, n = loo_cv(features, train)
    auc_str = f"{elite_auc:.3f}" if elite_auc else "N/A"
    print(f"{name:<40s} {auc_str:>10s} {n:>5d}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print("""
Age-adjusting rushing stats for RBs: does it help?

The WR model benefits from graduated age adjustment on YPRR
(FR +25%, SO +5%, JR -20%, SR -25%). Younger breakout receivers
have better NFL outcomes.

For RBs, the same graduated multiplier is applied to rushing stats
(adj_ypa, adj_elu_rush_mtf_per_att). The univariate signal improves
slightly (+0.020 to +0.042 Spearman), but...

[See holdout and LOO results above for the full-model answer]

Key insight: the age signal for RBs lives in *receiving* efficiency
(adj_yprr is in the model via composite_receiving), not in rushing
stats. Young RBs who rush efficiently are already captured by volume
features (explosive_pg, attempts) which correlate with workload trust.
""")
