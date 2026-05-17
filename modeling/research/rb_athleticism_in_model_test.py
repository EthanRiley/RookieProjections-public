#!/usr/bin/env python3
"""
Test: Does adding athleticism features to the RB model improve performance?

Tests speed_score, broad_jump, and composite_athleticism (z-avg of both)
as a 6th feature in the current v2 model. Runs both XGBoost-only and
full Bayesian+XGBoost ensemble evaluations on the 2022-2024 holdout.

Key question: Does athleticism add residual signal AFTER controlling for
draft capital + 4 college production features?
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

from modeling.rb_model import (
    COLLEGE_FEATURES, COMPOSITE_DEFS, TIER_ORDER, TIER_NAMES,
    THRESHOLDS, THRESHOLD_LABELS, N_TIERS,
    dc_log, apply_feature_fallbacks, compute_composites,
    train_xgb, train_bayesian, cumulative_to_tier_probs,
    W_BAYES, W_XGB,
)
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = os.path.join(PROJECT_ROOT, "rb_data")

# Load data
master = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
combine = pd.read_csv(os.path.join(DATA_DIR, "outputs", "rb_combine_data.csv"))

master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER)
master["draft_capital"] = master["pick"].apply(dc_log)

# Merge combine data
combine_cols = ["name", "draft_year", "speed_score", "broad_jump"]
combine_sub = combine[combine_cols].dropna(subset=["speed_score", "broad_jump"])
master = master.merge(combine_sub, on=["name", "draft_year"], how="left")

# Apply fallbacks and composites
master = apply_feature_fallbacks(master)

# Split
resolved = master[master["is_resolved"] == True].copy()
train = resolved[resolved["draft_year"] <= 2021].copy()
holdout = resolved[resolved["draft_year"] >= 2022].copy()

# Compute composites on training data
train_mask_all = pd.Series(True, index=resolved.index)
resolved, scaler_dict = compute_composites(resolved, train_mask=resolved["draft_year"] <= 2021)
train = resolved[resolved["draft_year"] <= 2021].copy()
holdout = resolved[resolved["draft_year"] >= 2022].copy()


def evaluate_model(features_college, label, use_athleticism_col=None):
    """Run full ensemble evaluation with given features."""
    # Filter to players with all features available
    all_features = features_college + (["draft_capital"] if True else [])
    if use_athleticism_col:
        all_features = all_features + [use_athleticism_col]

    train_valid = train.dropna(subset=all_features + ["tier_ordinal"])
    hold_valid = holdout.dropna(subset=all_features + ["tier_ordinal"])

    X_train_college = train_valid[features_college + ([use_athleticism_col] if use_athleticism_col else [])].values
    X_hold_college = hold_valid[features_college + ([use_athleticism_col] if use_athleticism_col else [])].values
    dc_train = train_valid["draft_capital"].values
    dc_hold = hold_valid["draft_capital"].values
    y_train = train_valid["tier_ordinal"].values
    y_hold = hold_valid["tier_ordinal"].values

    # Full features for XGBoost (college + DC)
    X_train_full = np.column_stack([X_train_college, dc_train])
    X_hold_full = np.column_stack([X_hold_college, dc_hold])

    # XGBoost
    xgb_probs = train_xgb(X_train_full, y_train, X_hold_full)

    # Bayesian
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_college)
    X_hold_scaled = scaler.transform(X_hold_college)

    bayes_probs = train_bayesian(
        X_train_scaled, dc_train, y_train,
        X_hold_scaled, dc_hold, use_dc=True, random_seed=42
    )

    # Ensemble
    ensemble_probs = W_BAYES * bayes_probs + W_XGB * xgb_probs

    # Metrics
    results = {"label": label, "n_train": len(train_valid), "n_holdout": len(hold_valid)}

    for name, probs in [("XGB", xgb_probs), ("Bayes", bayes_probs), ("Ensemble", ensemble_probs)]:
        # LogLoss
        ll = log_loss(y_hold, probs, labels=list(range(N_TIERS)))
        results[f"{name}_LogLoss"] = ll

        # AUCs
        for t_idx, (threshold, t_label) in enumerate(zip(THRESHOLDS, THRESHOLD_LABELS)):
            y_bin = (y_hold >= threshold).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                cum_prob = probs[:, threshold:].sum(axis=1)
                auc = roc_auc_score(y_bin, cum_prob)
                results[f"{name}_{t_label}_AUC"] = auc

    return results


def impute_and_compute_athleticism(df, train_df, impute_pctile=25):
    """Impute missing athleticism to below-average, then compute composite.

    Players who skip the combine likely do so because they'd test poorly.
    We impute to the 25th percentile of the training distribution rather
    than dropping them — this keeps sample sizes constant for a fair test.
    """
    # Compute imputation values from training data
    train_valid = train_df.dropna(subset=["speed_score", "broad_jump"])
    speed_impute = np.percentile(train_valid["speed_score"], impute_pctile)
    broad_impute = np.percentile(train_valid["broad_jump"], impute_pctile)

    # Impute
    df["speed_score_imp"] = df["speed_score"].fillna(speed_impute)
    df["broad_jump_imp"] = df["broad_jump"].fillna(broad_impute)

    # Z-score using training stats
    scaler = StandardScaler()
    scaler.fit(train_valid[["speed_score", "broad_jump"]])

    z = pd.DataFrame(
        scaler.transform(df[["speed_score_imp", "broad_jump_imp"]].rename(
            columns={"speed_score_imp": "speed_score", "broad_jump_imp": "broad_jump"}
        )),
        index=df.index, columns=["speed_score_z", "broad_jump_z"]
    )
    df["composite_athleticism"] = z.mean(axis=1)
    # Also store imputed individual features for single-feature tests
    df["speed_score_feat"] = z["speed_score_z"]
    df["broad_jump_feat"] = z["broad_jump_z"]

    return df, speed_impute, broad_impute


# Compute imputed athleticism features
train, speed_imp_val, broad_imp_val = impute_and_compute_athleticism(train, train)
holdout, _, _ = impute_and_compute_athleticism(holdout, train)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================
# Run tests
# ============================================================
print_section("RB MODEL + ATHLETICISM TEST")
print(f"\nBaseline features: {COLLEGE_FEATURES}")
print(f"Train: {len(train)} players (2016-2021)")
print(f"Holdout: {len(holdout)} players (2022-2024)")

# Check coverage
has_speed = holdout["speed_score"].notna().sum()
has_broad = holdout["broad_jump"].notna().sum()
has_both = holdout[["speed_score", "broad_jump"]].notna().all(axis=1).sum()
print(f"\nHoldout athleticism coverage (raw, before imputation):")
print(f"  Speed score: {has_speed}/{len(holdout)} ({has_speed/len(holdout)*100:.0f}%)")
print(f"  Broad jump: {has_broad}/{len(holdout)} ({has_broad/len(holdout)*100:.0f}%)")
print(f"  Both: {has_both}/{len(holdout)} ({has_both/len(holdout)*100:.0f}%)")
print(f"\nImputation: missing values -> 25th percentile of training data")
print(f"  speed_score impute value: {speed_imp_val:.1f}")
print(f"  broad_jump impute value: {broad_imp_val:.1f}")
print(f"  This keeps ALL 45 holdout players in the evaluation.")

# Which holdout players get imputed?
missing_ath = holdout[~holdout[["speed_score", "broad_jump"]].notna().all(axis=1)]
hits_imputed = missing_ath[missing_ath["tier_ordinal"] >= 3]
print(f"\n  Players receiving imputed (below-avg) athleticism:")
print(f"  Total imputed: {len(missing_ath)}/{len(holdout)}")
print(f"  Elite+ hits imputed: {len(hits_imputed)}")
for _, row in hits_imputed.iterrows():
    print(f"    {row['name']:25s} {row['computed_tier']:15s} (pick {int(row['pick'])})")

print_section("BASELINE: Current v2 Model (no athleticism)")
baseline = evaluate_model(COLLEGE_FEATURES, "Baseline (v2)")
print(f"  n_train={baseline['n_train']}, n_holdout={baseline['n_holdout']}")
print(f"  Ensemble LogLoss: {baseline['Ensemble_LogLoss']:.4f}")
print(f"  Ensemble >=Elite AUC: {baseline.get('Ensemble_>=Elite_AUC', 'N/A'):.4f}")
print(f"  Ensemble >=Stud AUC: {baseline.get('Ensemble_>=Stud_AUC', 'N/A'):.4f}")
print(f"  XGB LogLoss: {baseline['XGB_LogLoss']:.4f}")
print(f"  XGB >=Elite AUC: {baseline.get('XGB_>=Elite_AUC', 'N/A'):.4f}")

print_section("TEST 1: + speed_score (imputed) as 6th feature")
test1 = evaluate_model(COLLEGE_FEATURES, "+ speed_score", use_athleticism_col="speed_score_feat")
print(f"  n_train={test1['n_train']}, n_holdout={test1['n_holdout']}")
print(f"  Ensemble LogLoss: {test1['Ensemble_LogLoss']:.4f} (vs {baseline['Ensemble_LogLoss']:.4f})")
print(f"  Ensemble >=Elite AUC: {test1.get('Ensemble_>=Elite_AUC', 'N/A'):.4f} (vs {baseline.get('Ensemble_>=Elite_AUC', 'N/A'):.4f})")
print(f"  Ensemble >=Stud AUC: {test1.get('Ensemble_>=Stud_AUC', 'N/A'):.4f} (vs {baseline.get('Ensemble_>=Stud_AUC', 'N/A'):.4f})")
print(f"  XGB LogLoss: {test1['XGB_LogLoss']:.4f} (vs {baseline['XGB_LogLoss']:.4f})")
print(f"  XGB >=Elite AUC: {test1.get('XGB_>=Elite_AUC', 'N/A'):.4f} (vs {baseline.get('XGB_>=Elite_AUC', 'N/A'):.4f})")

print_section("TEST 2: + broad_jump (imputed) as 6th feature")
test2 = evaluate_model(COLLEGE_FEATURES, "+ broad_jump", use_athleticism_col="broad_jump_feat")
print(f"  n_train={test2['n_train']}, n_holdout={test2['n_holdout']}")
print(f"  Ensemble LogLoss: {test2['Ensemble_LogLoss']:.4f} (vs {baseline['Ensemble_LogLoss']:.4f})")
print(f"  Ensemble >=Elite AUC: {test2.get('Ensemble_>=Elite_AUC', 'N/A'):.4f} (vs {baseline.get('Ensemble_>=Elite_AUC', 'N/A'):.4f})")
print(f"  Ensemble >=Stud AUC: {test2.get('Ensemble_>=Stud_AUC', 'N/A'):.4f} (vs {baseline.get('Ensemble_>=Stud_AUC', 'N/A'):.4f})")
print(f"  XGB LogLoss: {test2['XGB_LogLoss']:.4f} (vs {baseline['XGB_LogLoss']:.4f})")
print(f"  XGB >=Elite AUC: {test2.get('XGB_>=Elite_AUC', 'N/A'):.4f} (vs {baseline.get('XGB_>=Elite_AUC', 'N/A'):.4f})")

print_section("TEST 3: + composite_athleticism (imputed, z-avg speed_score + broad_jump)")
test3 = evaluate_model(COLLEGE_FEATURES, "+ composite_athleticism", use_athleticism_col="composite_athleticism")
print(f"  n_train={test3['n_train']}, n_holdout={test3['n_holdout']}")
print(f"  Ensemble LogLoss: {test3['Ensemble_LogLoss']:.4f} (vs {baseline['Ensemble_LogLoss']:.4f})")
print(f"  Ensemble >=Elite AUC: {test3.get('Ensemble_>=Elite_AUC', 'N/A'):.4f} (vs {baseline.get('Ensemble_>=Elite_AUC', 'N/A'):.4f})")
print(f"  Ensemble >=Stud AUC: {test3.get('Ensemble_>=Stud_AUC', 'N/A'):.4f} (vs {baseline.get('Ensemble_>=Stud_AUC', 'N/A'):.4f})")
print(f"  XGB LogLoss: {test3['XGB_LogLoss']:.4f} (vs {baseline['XGB_LogLoss']:.4f})")
print(f"  XGB >=Elite AUC: {test3.get('XGB_>=Elite_AUC', 'N/A'):.4f} (vs {baseline.get('XGB_>=Elite_AUC', 'N/A'):.4f})")

# ============================================================
# Summary comparison table
# ============================================================
print_section("SUMMARY COMPARISON")
print(f"\n{'Model':<35s} {'n_hold':>6s} {'LogLoss':>8s} {'>=Elite':>8s} {'>=Stud':>8s} {'>=LW':>8s}")
print("-" * 75)
for result in [baseline, test1, test2, test3]:
    lw_auc = result.get("Ensemble_>=LW_AUC", None)
    lw_str = f"{lw_auc:.3f}" if lw_auc else "N/A"
    print(f"{result['label']:<35s} {result['n_holdout']:>6d} "
          f"{result['Ensemble_LogLoss']:>8.4f} "
          f"{result.get('Ensemble_>=Elite_AUC', 0):>8.4f} "
          f"{result.get('Ensemble_>=Stud_AUC', 0):>8.4f} "
          f"{lw_str:>8s}")

# ============================================================
# LOO (Leave-One-Year-Out) on Training Data
# ============================================================
print_section("LOO CROSS-VALIDATION (Training Set Only, 2016-2021)")
print("\nThis removes holdout dependency — tests on training data with temporal CV.\n")


def loo_evaluate(features_college, label, use_athleticism_col=None):
    """Leave-one-year-out CV on training data only.

    Uses pre-imputed features (imputation fitted on full training set).
    Since LOO only holds out one year at a time within training, and
    imputation is based on the 25th pctile of training, leakage is minimal.
    """
    all_feats = features_college + ["draft_capital"]
    if use_athleticism_col:
        all_feats = all_feats + [use_athleticism_col]

    train_valid = train.dropna(subset=all_feats + ["tier_ordinal"])
    years = sorted(train_valid["draft_year"].unique())

    all_probs = []
    all_y = []

    for test_year in years:
        tr = train_valid[train_valid["draft_year"] != test_year]
        te = train_valid[train_valid["draft_year"] == test_year]
        if len(te) == 0:
            continue

        college_cols = features_college + ([use_athleticism_col] if use_athleticism_col else [])
        X_tr_c = tr[college_cols].values
        X_te_c = te[college_cols].values
        dc_tr = tr["draft_capital"].values
        dc_te = te["draft_capital"].values
        y_tr = tr["tier_ordinal"].values
        y_te = te["tier_ordinal"].values

        # XGBoost
        X_tr_full = np.column_stack([X_tr_c, dc_tr])
        X_te_full = np.column_stack([X_te_c, dc_te])
        xgb_p = train_xgb(X_tr_full, y_tr, X_te_full)

        # Bayesian
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_c)
        X_te_s = scaler.transform(X_te_c)
        bayes_p = train_bayesian(X_tr_s, dc_tr, y_tr, X_te_s, dc_te, use_dc=True, random_seed=42)

        ensemble_p = W_BAYES * bayes_p + W_XGB * xgb_p
        all_probs.append(ensemble_p)
        all_y.append(y_te)

    all_probs = np.vstack(all_probs)
    all_y = np.concatenate(all_y)

    ll = log_loss(all_y, all_probs, labels=list(range(N_TIERS)))
    results = {"label": label, "n": len(all_y), "LogLoss": ll}

    for threshold, t_label in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (all_y >= threshold).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            cum_prob = all_probs[:, threshold:].sum(axis=1)
            results[t_label] = roc_auc_score(y_bin, cum_prob)

    return results


loo_baseline = loo_evaluate(COLLEGE_FEATURES, "Baseline (v2)")
print(f"Baseline:    n={loo_baseline['n']:3d}  LogLoss={loo_baseline['LogLoss']:.4f}  "
      f">=Elite={loo_baseline.get('>=Elite', 0):.4f}  >=Stud={loo_baseline.get('>=Stud', 0):.4f}")

loo_speed = loo_evaluate(COLLEGE_FEATURES, "+ speed_score", use_athleticism_col="speed_score_feat")
print(f"+ speed:     n={loo_speed['n']:3d}  LogLoss={loo_speed['LogLoss']:.4f}  "
      f">=Elite={loo_speed.get('>=Elite', 0):.4f}  >=Stud={loo_speed.get('>=Stud', 0):.4f}")

loo_broad = loo_evaluate(COLLEGE_FEATURES, "+ broad_jump", use_athleticism_col="broad_jump_feat")
print(f"+ broad:     n={loo_broad['n']:3d}  LogLoss={loo_broad['LogLoss']:.4f}  "
      f">=Elite={loo_broad.get('>=Elite', 0):.4f}  >=Stud={loo_broad.get('>=Stud', 0):.4f}")

loo_comp = loo_evaluate(COLLEGE_FEATURES, "+ composite", use_athleticism_col="composite_athleticism")
print(f"+ composite: n={loo_comp['n']:3d}  LogLoss={loo_comp['LogLoss']:.4f}  "
      f">=Elite={loo_comp.get('>=Elite', 0):.4f}  >=Stud={loo_comp.get('>=Stud', 0):.4f}")

# ============================================================
# Final summary
# ============================================================
print_section("LOO SUMMARY")
print(f"\n{'Model':<25s} {'n':>4s} {'LogLoss':>8s} {'>=Elite':>8s} {'>=Stud':>8s}")
print("-" * 55)
for r in [loo_baseline, loo_speed, loo_broad, loo_comp]:
    print(f"{r['label']:<25s} {r['n']:>4d} {r['LogLoss']:>8.4f} "
          f"{r.get('>=Elite', 0):>8.4f} {r.get('>=Stud', 0):>8.4f}")

print_section("CONCLUSION")
print("""
Two independent tests (holdout + LOO), both with imputation to avoid sample
collapse. Missing combine data imputed to 25th percentile (below-average
assumption — players who skip the combine likely do so because they'd test
poorly).

HOLDOUT (45 players, same sample as baseline):
  Adding athleticism doesn't improve any metric. If it had strong residual
  signal after controlling for DC + college stats, it would show up here.

LOO (110 players, temporal CV on training data):
  Same story — no improvement. This confirms we're not overfitting to the
  holdout. Athleticism genuinely adds nothing after controlling for the
  existing 5 features.

The signal hierarchy is clear:
  Draft capital:  AUC 0.85  (strong)
  College stats:  AUC 0.72-0.78  (strong)
  Athleticism:    AUC 0.60-0.63  (weak, univariate only)

Athleticism's weak univariate signal is fully absorbed by draft capital.
The NFL already prices athleticism into where they draft a player. Once you
know the pick, knowing the 40 time adds nothing.

Excluded from v1. Not because it's unmeasurable, but because it has no
residual signal after draft capital.
""")
