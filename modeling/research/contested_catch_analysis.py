#!/usr/bin/env python3
"""
Head-to-head analysis: career_contested_catch_rate vs best2_contested_catch_rate vs best_contested_catch_rate.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

# Other model features (excluding the contested catch rate variant being tested)
OTHER_FEATURES = [
    "draft_capital",
    "career_targeted_qb_rating",
    "breakout_age",
    "career_yprr",
    "career_catch_pct_adot_adj",
    "career_avoided_tackles_pg",
    "breakout_yptpa",
    "breakout_yprr",
]

CCR_VARIANTS = [
    "career_contested_catch_rate",
    "best2_contested_catch_rate",
    "best_contested_catch_rate",
]

# --- Load data ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
max_bo = df["breakout_age"].max()
df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

# Need all 3 variants + other features + tier
required = OTHER_FEATURES + CCR_VARIANTS + ["tier_ordinal"]
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
print(f"\n  {'Variant':<35s} {'Spearman':>10s} {'MI':>10s} {'AUC':>10s}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

for variant in CCR_VARIANTS:
    vals = df[variant].values
    sp, _ = spearmanr(vals, y)
    mi = mutual_info_classif(vals.reshape(-1, 1), y, random_state=42)[0]
    auc = roc_auc_score(hit, vals)
    print(f"  {variant:<35s} {sp:>+10.3f} {mi:>10.3f} {auc:>10.3f}")


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
print(f"\n  {'Variant':<35s} {'Early Sp':>10s} {'Late Sp':>10s} {'Drift':>10s}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

for variant in CCR_VARIANTS:
    sp_early, _ = spearmanr(early[variant].values, early["tier_ordinal"].values)
    sp_late, _ = spearmanr(late[variant].values, late["tier_ordinal"].values)
    drift = abs(sp_early - sp_late)
    print(f"  {variant:<35s} {sp_early:>+10.3f} {sp_late:>+10.3f} {drift:>10.3f}")


# ============================================================
# Part 3: Residual Signal After Other Model Features
# ============================================================
print(f"\n{'=' * 70}")
print("PART 3: RESIDUAL SIGNAL (after other model features)")
print("=" * 70)

scaler = StandardScaler()
X_other = scaler.fit_transform(df[OTHER_FEATURES].values)

# Fit OLS-style ordinal proxy: predict tier_ordinal from other features
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_other, y)
residuals = y - ridge.predict(X_other)

print(f"\n  {'Variant':<35s} {'Residual Sp':>12s} {'p-value':>10s}")
print(f"  {'-'*35} {'-'*12} {'-'*10}")

for variant in CCR_VARIANTS:
    sp, p = spearmanr(df[variant].values, residuals)
    print(f"  {variant:<35s} {sp:>+12.3f} {p:>10.4f}")


# ============================================================
# Part 4: Correlation Matrix Between Variants
# ============================================================
print(f"\n{'=' * 70}")
print("PART 4: CORRELATION MATRIX (between CCR variants)")
print("=" * 70)

print(f"\n  {'':>35s}", end="")
for v in CCR_VARIANTS:
    short = v.replace("_contested_catch_rate", "")
    print(f" {short:>12s}", end="")
print()
print(f"  {'-'*35}", end="")
for _ in CCR_VARIANTS:
    print(f" {'-'*12}", end="")
print()

for v1 in CCR_VARIANTS:
    short1 = v1.replace("_contested_catch_rate", "")
    print(f"  {short1:<35s}", end="")
    for v2 in CCR_VARIANTS:
        corr = float(spearmanr(df[v1].values, df[v2].values)[0])
        print(f" {corr:>+12.3f}", end="")
    print()


# ============================================================
# Part 5: Correlation With Other Model Features
# ============================================================
print(f"\n{'=' * 70}")
print("PART 5: CORRELATION WITH OTHER MODEL FEATURES")
print("=" * 70)

print(f"\n  {'Feature':<35s}", end="")
for v in CCR_VARIANTS:
    short = v.replace("_contested_catch_rate", "")
    print(f" {short:>12s}", end="")
print()
print(f"  {'-'*35}", end="")
for _ in CCR_VARIANTS:
    print(f" {'-'*12}", end="")
print()

for feat in OTHER_FEATURES:
    print(f"  {feat:<35s}", end="")
    for variant in CCR_VARIANTS:
        corr = float(spearmanr(df[feat].values, df[variant].values)[0])
        print(f" {corr:>+12.3f}", end="")
    print()

# Max collinearity
print(f"\n  {'Max |corr| with model features':<35s}", end="")
for variant in CCR_VARIANTS:
    max_corr = max(abs(float(spearmanr(df[feat].values, df[variant].values)[0]))
                   for feat in OTHER_FEATURES)
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
boot_results = {v: [] for v in CCR_VARIANTS}

for _ in range(n_boot):
    idx = rng.choice(len(df), size=len(df), replace=True)
    X_boot = scaler.fit_transform(df[OTHER_FEATURES].values[idx])
    y_boot = y[idx]
    ridge.fit(X_boot, y_boot)
    resid_boot = y_boot - ridge.predict(X_boot)

    for variant in CCR_VARIANTS:
        sp, _ = spearmanr(df[variant].values[idx], resid_boot)
        boot_results[variant].append(sp)

print(f"\n  {'Variant':<35s} {'Mean':>8s} {'Std':>8s} {'2.5%':>8s} {'97.5%':>8s} {'% > 0':>8s}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for variant in CCR_VARIANTS:
    arr = np.array(boot_results[variant])
    print(f"  {variant:<35s} {arr.mean():>+8.3f} {arr.std():>8.3f} "
          f"{np.percentile(arr, 2.5):>+8.3f} {np.percentile(arr, 97.5):>+8.3f} "
          f"{(arr > 0).mean():>8.1%}")

# Pairwise win rates
print(f"\n  Pairwise win rates (row beats column in residual Spearman):")
boot_arr = {v: np.array(boot_results[v]) for v in CCR_VARIANTS}
print(f"  {'':>35s}", end="")
for v in CCR_VARIANTS:
    short = v.replace("_contested_catch_rate", "")
    print(f" {short:>12s}", end="")
print()

for v1 in CCR_VARIANTS:
    short1 = v1.replace("_contested_catch_rate", "")
    print(f"  {short1:<35s}", end="")
    for v2 in CCR_VARIANTS:
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

for variant in CCR_VARIANTS:
    full_features = OTHER_FEATURES + [variant]
    total_signal = 0.0
    print(f"\n  Feature set with {variant}:")
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


# ============================================================
# Part 8: Leave-One-Year-Out AUC Comparison
# ============================================================
print(f"\n{'=' * 70}")
print("PART 8: LEAVE-ONE-YEAR-OUT AUC (Elite+ threshold)")
print("=" * 70)

for variant in CCR_VARIANTS:
    full_features = OTHER_FEATURES + [variant]
    all_preds = []
    all_true = []

    for fold_year in years:
        train_mask = df["draft_year"] != fold_year
        val_mask = df["draft_year"] == fold_year

        X_tr = StandardScaler().fit_transform(df.loc[train_mask, full_features].values)
        X_val = StandardScaler().fit_transform(df.loc[val_mask, full_features].values)
        # Re-fit scaler properly
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
    print(f"  {variant:<35s}  LOO-AUC: {auc:.3f}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print("""
Key metrics to compare:
  - Part 1: Raw univariate signal
  - Part 2: Era stability (lower drift = better)
  - Part 3: Residual signal (higher = more unique info)
  - Part 5: Max collinearity (lower = more independent)
  - Part 6: Bootstrap win rate (who wins more often)
  - Part 7: Total feature set signal (higher = better overall set)
  - Part 8: LOO-AUC (higher = better predictive power)
""")
