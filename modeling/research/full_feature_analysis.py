#!/usr/bin/env python3
"""
Comprehensive 8-part analysis for every candidate feature dimension.

Base model: draft_capital + breakout_age (the only 2 locked features).
Tests every plausible variant for each dimension against this base.

Dimensions:
  A. YPRR (route efficiency)
  B. YPTPA / market share
  C. QB trust / route quality
  D. Contested catch rate
  E. Catch reliability
  F. Elusiveness / YAC
  G. Production volume
"""

import os
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

# Only 2 locked features
BASE_FEATURES = ["draft_capital", "breakout_age"]

# All candidate features grouped by dimension
DIMENSIONS = {
    "A: YPRR / Route Efficiency": [
        "career_yprr",
        "best2_yprr",
        "peak_yprr",
        "breakout_yprr",
    ],
    "B: YPTPA / Market Share": [
        "breakout_yptpa",
        "best_yards_per_team_pass_att",
        "career_yards_pg",
        "best2_yards_pg",
    ],
    "C: QB Trust / Route Quality": [
        "career_targeted_qb_rating",
        "best2_targeted_qb_rating",
        "best_targeted_qb_rating",
        "career_grades_pass_route",
        "best2_grades_pass_route",
        "career_grades_offense",
        "best2_grades_offense",
    ],
    "D: Contested Catch Rate": [
        "peak_contested_catch_rate",
        "career_contested_catch_rate",
        "best2_contested_catch_rate",
        "best_contested_catch_rate",
    ],
    "E: Catch Reliability": [
        "career_catch_pct_adot_adj",
        "best2_catch_pct_adot_adj",
        "career_caught_percent",
        "best2_caught_percent",
        "best_caught_percent",
    ],
    "F: Elusiveness / YAC": [
        "peak2_avoided_tackles_per_rec",
        "career_avoided_tackles_per_rec",
        "career_avoided_tackles_pg",
        "best2_avoided_tackles_pg",
        "best2_avoided_tackles_per_rec",
        "career_yards_after_catch_pg",
        "best2_yards_after_catch_pg",
        "career_yards_after_catch_per_reception",
    ],
    "G: Production Volume": [
        "career_targets_pg",
        "best2_targets_pg",
        "career_receptions_pg",
        "best2_receptions_pg",
        "career_first_downs_per_route",
        "best2_first_downs_per_route",
    ],
}


def run_dimension_analysis(dim_name, candidates, df, base_features, y, hit, years):
    """Run 8-part analysis for a single dimension."""
    print(f"\n{'#' * 74}")
    print(f"#  DIMENSION: {dim_name}")
    print(f"{'#' * 74}")

    # Filter to candidates that exist in the data
    valid_candidates = [c for c in candidates if c in df.columns and df[c].notna().sum() > 50]
    if not valid_candidates:
        print("  No valid candidates with sufficient data.")
        return {}

    # Create complete-cases subset for this dimension (all candidates + base + tier)
    all_cols = base_features + valid_candidates + ["tier_ordinal", "draft_year"]
    df_dim = df.dropna(subset=all_cols).copy()
    y_dim = df_dim["tier_ordinal"].values
    hit_dim = (y_dim >= 3).astype(int)
    years_dim = sorted(df_dim["draft_year"].unique())
    print(f"\n  Complete cases: {len(df_dim)} players (years: {[int(x) for x in years_dim]})")

    # Use df_dim, y_dim, hit_dim, years_dim throughout
    d = df_dim
    yv = y_dim
    hv = hit_dim
    yrs = years_dim

    # --- Part 1: Univariate ---
    print(f"\n  --- Part 1: Univariate Screens ---")
    print(f"  {'Candidate':<40s} {'Spearman':>9s} {'MI':>7s} {'AUC':>7s}")
    print(f"  {'-'*40} {'-'*9} {'-'*7} {'-'*7}")

    for c in valid_candidates:
        vals = d[c].values
        try:
            sp, _ = spearmanr(vals, yv)
            mi = mutual_info_classif(vals.reshape(-1, 1), yv, random_state=42)[0]
            auc = roc_auc_score(hv, vals)
            print(f"  {c:<40s} {sp:>+9.3f} {mi:>7.3f} {auc:>7.3f}")
        except Exception as e:
            print(f"  {c:<40s}  [error: {e}]")

    # --- Part 2: Era Stability ---
    print(f"\n  --- Part 2: Era Stability ---")
    mid = yrs[len(yrs) // 2]
    early = d[d["draft_year"] <= mid]
    late = d[d["draft_year"] > mid]

    print(f"  {'Candidate':<40s} {'Early Sp':>9s} {'Late Sp':>9s} {'Drift':>7s}")
    print(f"  {'-'*40} {'-'*9} {'-'*9} {'-'*7}")

    for c in valid_candidates:
        try:
            sp_e, _ = spearmanr(early[c].values, early["tier_ordinal"].values)
            sp_l, _ = spearmanr(late[c].values, late["tier_ordinal"].values)
            print(f"  {c:<40s} {sp_e:>+9.3f} {sp_l:>+9.3f} {abs(sp_e - sp_l):>7.3f}")
        except Exception:
            print(f"  {c:<40s}  [error]")

    # --- Part 3: Residual Signal ---
    print(f"\n  --- Part 3: Residual Signal (after base: {', '.join(base_features)}) ---")
    scaler = StandardScaler()
    X_base = scaler.fit_transform(d[base_features].values)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_base, yv)
    residuals = yv - ridge.predict(X_base)

    print(f"  {'Candidate':<40s} {'Residual Sp':>12s} {'p-value':>9s}")
    print(f"  {'-'*40} {'-'*12} {'-'*9}")

    for c in valid_candidates:
        sp, p = spearmanr(d[c].values, residuals)
        print(f"  {c:<40s} {sp:>+12.3f} {p:>9.4f}")

    # --- Part 4: Max Collinearity with Base ---
    print(f"\n  --- Part 4: Collinearity with Base Features ---")
    print(f"  {'Candidate':<40s}", end="")
    for bf in base_features:
        short = bf[:12]
        print(f" {short:>12s}", end="")
    print(f" {'Max |corr|':>10s}")
    print(f"  {'-'*40}", end="")
    for _ in base_features:
        print(f" {'-'*12}", end="")
    print(f" {'-'*10}")

    for c in valid_candidates:
        print(f"  {c:<40s}", end="")
        max_corr = 0
        for bf in base_features:
            corr = float(spearmanr(d[bf].values, d[c].values)[0])
            max_corr = max(max_corr, abs(corr))
            print(f" {corr:>+12.3f}", end="")
        print(f" {max_corr:>10.3f}")

    # --- Part 5: Bootstrap ---
    print(f"\n  --- Part 5: Bootstrap (1000 iterations) ---")
    n_boot = 1000
    rng = np.random.RandomState(42)
    boot_results = {c: [] for c in valid_candidates}

    for _ in range(n_boot):
        idx = rng.choice(len(d), size=len(d), replace=True)
        X_boot = scaler.fit_transform(d[base_features].values[idx])
        y_boot = yv[idx]
        ridge.fit(X_boot, y_boot)
        resid_boot = y_boot - ridge.predict(X_boot)
        for c in valid_candidates:
            sp, _ = spearmanr(d[c].values[idx], resid_boot)
            boot_results[c].append(sp)

    print(f"  {'Candidate':<40s} {'Mean':>7s} {'Std':>7s} {'% > 0':>7s}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7}")

    for c in valid_candidates:
        arr = np.array(boot_results[c])
        print(f"  {c:<40s} {arr.mean():>+7.3f} {arr.std():>7.3f} {(arr > 0).mean():>7.1%}")

    # --- Part 6: LOO-AUC ---
    print(f"\n  --- Part 6: Leave-One-Year-Out AUC ---")

    # Baseline
    all_p, all_t = [], []
    for fold_year in yrs:
        train_mask = d["draft_year"] != fold_year
        val_mask = d["draft_year"] == fold_year
        sc = StandardScaler()
        X_tr = sc.fit_transform(d.loc[train_mask, base_features].values)
        X_val = sc.transform(d.loc[val_mask, base_features].values)
        y_tr = (d.loc[train_mask, "tier_ordinal"].values >= 3).astype(int)
        y_val = (d.loc[val_mask, "tier_ordinal"].values >= 3).astype(int)
        if y_tr.sum() < 2 or y_val.sum() == 0 or y_val.sum() == len(y_val):
            continue
        m = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        m.fit(X_tr, y_tr)
        all_p.extend(m.predict_proba(X_val)[:, 1])
        all_t.extend(y_val)
    base_auc = roc_auc_score(np.array(all_t), np.array(all_p))
    print(f"  {'[baseline: base only]':<40s}  LOO-AUC: {base_auc:.3f}")

    results = {}
    for c in valid_candidates:
        feats = base_features + [c]
        all_p, all_t = [], []
        for fold_year in yrs:
            train_mask = d["draft_year"] != fold_year
            val_mask = d["draft_year"] == fold_year
            sc = StandardScaler()
            X_tr = sc.fit_transform(d.loc[train_mask, feats].values)
            X_val = sc.transform(d.loc[val_mask, feats].values)
            y_tr = (d.loc[train_mask, "tier_ordinal"].values >= 3).astype(int)
            y_val = (d.loc[val_mask, "tier_ordinal"].values >= 3).astype(int)
            if y_tr.sum() < 2 or y_val.sum() == 0 or y_val.sum() == len(y_val):
                continue
            m = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
            m.fit(X_tr, y_tr)
            all_p.extend(m.predict_proba(X_val)[:, 1])
            all_t.extend(y_val)
        auc = roc_auc_score(np.array(all_t), np.array(all_p))
        delta = auc - base_auc
        print(f"  {c:<40s}  LOO-AUC: {auc:.3f}  (delta: {delta:+.3f})")
        results[c] = {
            "auc": auc,
            "delta": delta,
            "residual_mean": np.array(boot_results[c]).mean(),
            "boot_pct_pos": (np.array(boot_results[c]) > 0).mean(),
        }

    # --- Part 7: Elastic Net Survival ---
    print(f"\n  --- Part 7: Elastic Net Survival ---")
    print(f"  {'Candidate':<40s} {'C=0.01':>7s} {'C=0.1':>7s} {'C=1.0':>7s} {'Count':>7s}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    y_bin = (yv >= 3).astype(int)
    for c in valid_candidates:
        feats = base_features + [c]
        X_all = StandardScaler().fit_transform(d[feats].values)
        marks = []
        for C in [0.01, 0.1, 1.0]:
            lr = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=C, max_iter=10000, random_state=42
            )
            lr.fit(X_all, y_bin)
            survived = abs(lr.coef_[0, -1]) > 1e-6
            marks.append("Y" if survived else ".")
        print(f"  {c:<40s} {marks[0]:>7s} {marks[1]:>7s} {marks[2]:>7s} {sum(m == 'Y' for m in marks):>5d}/3")

    return results


# --- Main ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
max_bo = df["breakout_age"].max()
df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

# Drop rows missing base features or tier
required_base = BASE_FEATURES + ["tier_ordinal"]
df = df.dropna(subset=required_base).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)

y = df["tier_ordinal"].values
hit = (y >= 3).astype(int)
years = sorted(df["draft_year"].unique())

print(f"Dataset: {len(df)} players")
print(f"Draft years: {years}")
print(f"Base features: {BASE_FEATURES}")
print(f"Tier distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

all_results = {}
for dim_name, candidates in DIMENSIONS.items():
    results = run_dimension_analysis(dim_name, candidates, df, BASE_FEATURES, y, hit, years)
    all_results[dim_name] = results

# ============================================================
# Grand Summary
# ============================================================
print(f"\n\n{'=' * 74}")
print("GRAND SUMMARY: ALL DIMENSIONS")
print(f"{'=' * 74}")
print(f"\n  {'Candidate':<40s} {'LOO-AUC':>8s} {'Delta':>8s} {'Resid':>8s} {'Boot%+':>8s}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

# Flatten and sort by LOO-AUC delta
flat = []
for dim_name, results in all_results.items():
    for feat, stats in results.items():
        flat.append((feat, dim_name, stats))

flat.sort(key=lambda x: x[2]["delta"], reverse=True)
for feat, dim, stats in flat:
    print(f"  {feat:<40s} {stats['auc']:>8.3f} {stats['delta']:>+8.3f} "
          f"{stats['residual_mean']:>+8.3f} {stats['boot_pct_pos']:>8.1%}")
