#!/usr/bin/env python3
"""
Test dropping peak2_ypa from the model, keeping only peak_yac_per_att.

Configs:
  5F (current): peak2_ypa + composite_explosive + composite_receiving + peak_yac_per_att
  4F (drop YPA): composite_explosive + composite_receiving + peak_yac_per_att
  3F (grid winner): best2_explosive_pg + adj_yprr + composite_receiving
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "modeling"))

from rb_model import (
    COLLEGE_FEATURES,
    COMPOSITE_DEFS,
    TIER_ORDER,
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
master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER).astype(float)
master["draft_capital"] = master["pick"].apply(dc_log)
master = apply_feature_fallbacks(master)
master = master[master["is_resolved"] == True].copy()
master["tier_ordinal"] = master["tier_ordinal"].astype(int)

train_mask = master["draft_year"] <= 2021
master, _ = compute_composites(master, train_mask=train_mask)

train = master[master["draft_year"] <= 2021].copy()
holdout = master[master["draft_year"].between(2022, 2024)].copy()


def evaluate_config(college_features, train_df, test_df):
    all_features = ["draft_capital"] + college_features
    tr = train_df.dropna(subset=all_features).copy()
    te = test_df.dropna(subset=all_features).copy()

    y_train = tr["tier_ordinal"].values
    y_true = te["tier_ordinal"].values

    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(tr[college_features].values)
    X_college_pred = scaler.transform(te[college_features].values)

    xgb_probs = train_xgb(tr[all_features].values, y_train, te[all_features].values)
    bay_probs = train_bayesian(
        X_college_train, tr["draft_capital"].values, y_train,
        X_college_pred, te["draft_capital"].values, True
    )
    ens_probs = blend(bay_probs, xgb_probs)

    eps = 1e-8
    clipped = np.clip(ens_probs, eps, 1 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    logloss = sk_log_loss(y_true, clipped, labels=list(range(6)))

    aucs = {}
    for thresh, name in [(3, "Elite"), (4, "Stud"), (2, "Starter"), (5, "LW")]:
        y_bin = (y_true >= thresh).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            aucs[name] = roc_auc_score(y_bin, ens_probs[:, thresh:].sum(axis=1))

    return logloss, aucs, len(te)


def loo_cv(college_features, data):
    years = sorted(data["draft_year"].unique())
    all_true, all_probs = [], []
    all_features = ["draft_capital"] + college_features

    for yr in years:
        tr = data[data["draft_year"] != yr].copy()
        te = data[data["draft_year"] == yr].copy()
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
            xgb_probs = train_xgb(tr[all_features].values, y_train, te[all_features].values)
            bay_probs = train_bayesian(
                X_college_train, tr["draft_capital"].values, y_train,
                X_college_pred, te["draft_capital"].values, True
            )
            ens_probs = blend(bay_probs, xgb_probs)
            all_true.extend(te["tier_ordinal"].values.tolist())
            all_probs.extend(ens_probs.tolist())
        except Exception as e:
            print(f"  LOO {yr} failed: {e}")

    all_true = np.array(all_true)
    all_probs = np.array(all_probs)
    eps = 1e-8
    clipped = np.clip(all_probs, eps, 1 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    logloss = sk_log_loss(all_true, clipped, labels=list(range(6)))

    aucs = {}
    for thresh, name in [(3, "Elite"), (4, "Stud"), (2, "Starter")]:
        y_bin = (all_true >= thresh).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            aucs[name] = roc_auc_score(y_bin, all_probs[:, thresh:].sum(axis=1))

    return logloss, aucs, len(all_true)


configs = [
    ("5F current", list(COLLEGE_FEATURES)),
    ("4F drop YPA", ["composite_explosive", "composite_receiving", "peak_yac_per_att"]),
    ("3F grid winner", ["best2_explosive_pg", "adj_yprr", "composite_receiving"]),
]

print("=" * 70)
print("  Drop YPA Test")
print("=" * 70)

print("\n  HOLDOUT (2022-2024)")
print(f"  {'Config':<20s} {'LogLoss':>8s} {'Elite':>8s} {'Stud':>8s} {'Starter':>8s} {'n':>5s}")
print("  " + "-" * 55)
for label, feats in configs:
    ll, aucs, n = evaluate_config(feats, train, holdout)
    print(f"  {label:<20s} {ll:>8.3f} {aucs.get('Elite',0):>8.3f} {aucs.get('Stud',0):>8.3f} {aucs.get('Starter',0):>8.3f} {n:>5d}")

print("\n  LOO CV (2016-2021)")
print(f"  {'Config':<20s} {'LogLoss':>8s} {'Elite':>8s} {'Stud':>8s} {'Starter':>8s} {'n':>5s}")
print("  " + "-" * 55)
for label, feats in configs:
    ll, aucs, n = loo_cv(feats, train)
    print(f"  {label:<20s} {ll:>8.3f} {aucs.get('Elite',0):>8.3f} {aucs.get('Stud',0):>8.3f} {aucs.get('Starter',0):>8.3f} {n:>5d}")
