#!/usr/bin/env python3
"""
Head-to-head analysis: career_yprr vs best2_yprr vs peak_yprr vs breakout_yprr.

8-part comprehensive comparison:
  1. Univariate screens (Spearman, MI, AUC)
  2. Era stability
  3. Residual signal after other model features
  4. Correlation matrix between variants
  5. Correlation with other model features
  6. Bootstrap head-to-head (residual Spearman)
  7. Total feature set signal (sum |residual Spearman|)
  8. Leave-one-year-out AUC comparison
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

# Current model features EXCLUDING any yprr variant and breakout_yprr.
# We test each YPRR variant's contribution against this base.
BASE_FEATURES = [
    "draft_capital",
    "career_targeted_qb_rating",
    "breakout_age",
    "peak_contested_catch_rate",
    "peak2_avoided_tackles_per_rec",
    "breakout_yptpa",
]

YPRR_VARIANTS = [
    "career_yprr",
    "best2_yprr",
    "peak_yprr",
    "breakout_yprr",
]

# --- Load data ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
max_bo = df["breakout_age"].max()
df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

# Need all variants + base features + tier
required = BASE_FEATURES + YPRR_VARIANTS + ["tier_ordinal"]
df = df.dropna(subset=required).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)

y = df["tier_ordinal"].values
hit = (y >= 3).astype(int)  # Elite+

print(f"Dataset: {len(df)} players")
print(f"Draft years: {sorted(df['draft_year'].unique())}")
print()


# ============================================================
# Part 1: Univariate Screens
# ============================================================
print("=" * 70)
print("PART 1: UNIVARIATE SCREENS")
print("=" * 70)
print(f"\n  {'Variant':<25s} {'Spearman':>10s} {'MI':>10s} {'AUC':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

for variant in YPRR_VARIANTS:
    vals = df[variant].values
    sp, _ = spearmanr(vals, y)
    mi = mutual_info_classif(vals.reshape(-1, 1), y, random_state=42)[0]
    auc = roc_auc_score(hit, vals)
    print(f"  {variant:<25s} {sp:>+10.3f} {mi:>10.3f} {auc:>10.3f}")


# ============================================================
# Part 2: Era Stability
# ============================================================
print(f"\n{'=' * 70}")
print("PART 2: ERA STABILITY")
print("=" * 70)

years = sorted(df["draft_year"].unique())
mid = years[len(years) // 2]
early = df[df["draft_year"] <= mid]
late = df[df["draft_year"] > mid]

print(f"\n  Early: {sorted(early['draft_year'].unique())} ({len(early)} players)")
print(f"  Late:  {sorted(late['draft_year'].unique())} ({len(late)} players)")
print(f"\n  {'Variant':<25s} {'Early Sp':>10s} {'Late Sp':>10s} {'Drift':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

for variant in YPRR_VARIANTS:
    sp_early, _ = spearmanr(early[variant].values, early["tier_ordinal"].values)
    sp_late, _ = spearmanr(late[variant].values, late["tier_ordinal"].values)
    drift = abs(sp_early - sp_late)
    print(f"  {variant:<25s} {sp_early:>+10.3f} {sp_late:>+10.3f} {drift:>10.3f}")


# ============================================================
# Part 3: Residual Signal After Base Features
# ============================================================
print(f"\n{'=' * 70}")
print("PART 3: RESIDUAL SIGNAL (after base features, no YPRR variant)")
print("=" * 70)

scaler = StandardScaler()
X_base = scaler.fit_transform(df[BASE_FEATURES].values)

ridge = Ridge(alpha=1.0)
ridge.fit(X_base, y)
residuals = y - ridge.predict(X_base)

print(f"\n  {'Variant':<25s} {'Residual Sp':>12s} {'p-value':>10s}")
print(f"  {'-'*25} {'-'*12} {'-'*10}")

for variant in YPRR_VARIANTS:
    sp, p = spearmanr(df[variant].values, residuals)
    print(f"  {variant:<25s} {sp:>+12.3f} {p:>10.4f}")


# ============================================================
# Part 4: Correlation Matrix Between Variants
# ============================================================
print(f"\n{'=' * 70}")
print("PART 4: CORRELATION MATRIX (between YPRR variants)")
print("=" * 70)

short_names = {v: v.replace("_yprr", "") for v in YPRR_VARIANTS}

print(f"\n  {'':>25s}", end="")
for v in YPRR_VARIANTS:
    print(f" {short_names[v]:>12s}", end="")
print()
print(f"  {'-'*25}", end="")
for _ in YPRR_VARIANTS:
    print(f" {'-'*12}", end="")
print()

for v1 in YPRR_VARIANTS:
    print(f"  {short_names[v1]:<25s}", end="")
    for v2 in YPRR_VARIANTS:
        corr = float(spearmanr(df[v1].values, df[v2].values)[0])
        print(f" {corr:>+12.3f}", end="")
    print()


# ============================================================
# Part 5: Correlation With Other Model Features
# ============================================================
print(f"\n{'=' * 70}")
print("PART 5: CORRELATION WITH BASE MODEL FEATURES")
print("=" * 70)

print(f"\n  {'Feature':<35s}", end="")
for v in YPRR_VARIANTS:
    print(f" {short_names[v]:>12s}", end="")
print()
print(f"  {'-'*35}", end="")
for _ in YPRR_VARIANTS:
    print(f" {'-'*12}", end="")
print()

for feat in BASE_FEATURES:
    print(f"  {feat:<35s}", end="")
    for variant in YPRR_VARIANTS:
        corr = float(spearmanr(df[feat].values, df[variant].values)[0])
        print(f" {corr:>+12.3f}", end="")
    print()

# Max collinearity
print(f"\n  {'Max |corr| with base features':<35s}", end="")
for variant in YPRR_VARIANTS:
    max_corr = max(abs(float(spearmanr(df[feat].values, df[variant].values)[0]))
                   for feat in BASE_FEATURES)
    print(f" {max_corr:>12.3f}", end="")
print()


# ============================================================
# Part 6: Bootstrap Head-to-Head (Residual Spearman)
# ============================================================
print(f"\n{'=' * 70}")
print("PART 6: BOOTSTRAP HEAD-TO-HEAD (1000 iterations)")
print("=" * 70)

n_boot = 1000
rng = np.random.RandomState(42)
boot_results = {v: [] for v in YPRR_VARIANTS}

for _ in range(n_boot):
    idx = rng.choice(len(df), size=len(df), replace=True)
    X_boot = scaler.fit_transform(df[BASE_FEATURES].values[idx])
    y_boot = y[idx]
    ridge.fit(X_boot, y_boot)
    resid_boot = y_boot - ridge.predict(X_boot)

    for variant in YPRR_VARIANTS:
        sp, _ = spearmanr(df[variant].values[idx], resid_boot)
        boot_results[variant].append(sp)

print(f"\n  {'Variant':<25s} {'Mean':>8s} {'Std':>8s} {'2.5%':>8s} {'97.5%':>8s} {'% > 0':>8s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for variant in YPRR_VARIANTS:
    arr = np.array(boot_results[variant])
    print(f"  {variant:<25s} {arr.mean():>+8.3f} {arr.std():>8.3f} "
          f"{np.percentile(arr, 2.5):>+8.3f} {np.percentile(arr, 97.5):>+8.3f} "
          f"{(arr > 0).mean():>8.1%}")

# Pairwise win rates
print(f"\n  Pairwise win rates (row beats column in residual Spearman):")
boot_arr = {v: np.array(boot_results[v]) for v in YPRR_VARIANTS}
print(f"  {'':>25s}", end="")
for v in YPRR_VARIANTS:
    print(f" {short_names[v]:>12s}", end="")
print()

for v1 in YPRR_VARIANTS:
    print(f"  {short_names[v1]:<25s}", end="")
    for v2 in YPRR_VARIANTS:
        if v1 == v2:
            print(f" {'---':>12s}", end="")
        else:
            win_rate = (boot_arr[v1] > boot_arr[v2]).mean()
            print(f" {win_rate:>12.1%}", end="")
    print()


# ============================================================
# Part 7: Total Feature Set Signal
# ============================================================
print(f"\n{'=' * 70}")
print("PART 7: TOTAL FEATURE SET SIGNAL (sum |residual Spearman|)")
print("=" * 70)

for variant in YPRR_VARIANTS:
    full_features = BASE_FEATURES + [variant]
    total_signal = 0.0
    print(f"\n  Feature set with {short_names[variant]}_yprr:")
    print(f"  {'Feature':<35s} {'Residual |Sp|':>14s}")
    print(f"  {'-'*35} {'-'*14}")

    for feat in full_features:
        other = [f for f in full_features if f != feat]
        X_o = scaler.fit_transform(df[other].values)
        ridge.fit(X_o, y)
        resid = y - ridge.predict(X_o)
        sp = abs(float(spearmanr(df[feat].values, resid)[0]))
        total_signal += sp
        print(f"  {feat:<35s} {sp:>14.3f}")

    print(f"  {'TOTAL':<35s} {total_signal:>14.3f}")

# Also compute baseline (no YPRR variant at all)
print(f"\n  Baseline (no YPRR variant):")
total_baseline = 0.0
print(f"  {'Feature':<35s} {'Residual |Sp|':>14s}")
print(f"  {'-'*35} {'-'*14}")
for feat in BASE_FEATURES:
    other = [f for f in BASE_FEATURES if f != feat]
    X_o = scaler.fit_transform(df[other].values)
    ridge.fit(X_o, y)
    resid = y - ridge.predict(X_o)
    sp = abs(float(spearmanr(df[feat].values, resid)[0]))
    total_baseline += sp
    print(f"  {feat:<35s} {sp:>14.3f}")
print(f"  {'TOTAL':<35s} {total_baseline:>14.3f}")


# ============================================================
# Part 8: Leave-One-Year-Out AUC Comparison
# ============================================================
print(f"\n{'=' * 70}")
print("PART 8: LEAVE-ONE-YEAR-OUT AUC (Elite+ threshold)")
print("=" * 70)

# Baseline: no YPRR variant
all_preds_base = []
all_true_base = []
for fold_year in years:
    train_mask = df["draft_year"] != fold_year
    val_mask = df["draft_year"] == fold_year

    sc = StandardScaler()
    X_tr = sc.fit_transform(df.loc[train_mask, BASE_FEATURES].values)
    X_val = sc.transform(df.loc[val_mask, BASE_FEATURES].values)

    y_tr = (df.loc[train_mask, "tier_ordinal"].values >= 3).astype(int)
    y_val = (df.loc[val_mask, "tier_ordinal"].values >= 3).astype(int)

    if y_tr.sum() < 2 or (len(y_tr) - y_tr.sum()) < 2:
        continue
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        continue

    model = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)[:, 1]

    all_preds_base.extend(preds)
    all_true_base.extend(y_val)

base_auc = roc_auc_score(np.array(all_true_base), np.array(all_preds_base))
print(f"\n  {'[baseline: no YPRR]':<25s}  LOO-AUC: {base_auc:.3f}")

for variant in YPRR_VARIANTS:
    full_features = BASE_FEATURES + [variant]
    all_preds = []
    all_true = []

    for fold_year in years:
        train_mask = df["draft_year"] != fold_year
        val_mask = df["draft_year"] == fold_year

        sc = StandardScaler()
        X_tr = sc.fit_transform(df.loc[train_mask, full_features].values)
        X_val = sc.transform(df.loc[val_mask, full_features].values)

        y_tr = (df.loc[train_mask, "tier_ordinal"].values >= 3).astype(int)
        y_val = (df.loc[val_mask, "tier_ordinal"].values >= 3).astype(int)

        if y_tr.sum() < 2 or (len(y_tr) - y_tr.sum()) < 2:
            continue
        if y_val.sum() == 0 or y_val.sum() == len(y_val):
            continue

        model = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]

        all_preds.extend(preds)
        all_true.extend(y_val)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    auc = roc_auc_score(all_true, all_preds)
    delta = auc - base_auc
    print(f"  {variant:<25s}  LOO-AUC: {auc:.3f}  (delta: {delta:+.3f})")


# ============================================================
# Elastic Net Survival
# ============================================================
print(f"\n{'=' * 70}")
print("BONUS: ELASTIC NET SURVIVAL")
print("=" * 70)

from sklearn.linear_model import SGDClassifier

print(f"\n  {'Variant':<25s} {'C=0.01':>8s} {'C=0.1':>8s} {'C=1.0':>8s} {'Survives':>10s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

for variant in YPRR_VARIANTS:
    full_features = BASE_FEATURES + [variant]
    X_all = StandardScaler().fit_transform(df[full_features].values)
    y_bin = (y >= 3).astype(int)
    survivals = []

    for C in [0.01, 0.1, 1.0]:
        lr = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            C=C, max_iter=10000, random_state=42
        )
        lr.fit(X_all, y_bin)
        coef_idx = len(BASE_FEATURES)  # the YPRR variant is the last feature
        survived = abs(lr.coef_[0, coef_idx]) > 1e-6
        survivals.append(survived)

    survive_str = f"{sum(survivals)}/3"
    marks = ["Y" if s else "." for s in survivals]
    print(f"  {variant:<25s} {marks[0]:>8s} {marks[1]:>8s} {marks[2]:>8s} {survive_str:>10s}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print("""
  Comparison of 4 YPRR variants against a base model that includes:
    draft_capital, career_targeted_qb_rating, breakout_age,
    peak_contested_catch_rate, peak2_avoided_tackles_per_rec, breakout_yptpa

  Each variant is tested for its marginal contribution on top of this base.

  Key metrics:
    - Part 1: Raw univariate signal
    - Part 2: Era stability (lower drift = better)
    - Part 3: Residual signal after base (higher = more unique info)
    - Part 5: Max collinearity with base (lower = more independent)
    - Part 6: Bootstrap reliability (% positive = how often signal is real)
    - Part 7: Total feature set signal (higher = better overall)
    - Part 8: LOO-AUC (higher = better prediction; delta vs no-YPRR baseline)
    - Elastic Net: survives regularization (more = more robust)
""")
