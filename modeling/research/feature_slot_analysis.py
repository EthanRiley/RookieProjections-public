#!/usr/bin/env python3
"""
Systematic feature slot analysis.

Locked features (not contested):
  - draft_capital
  - breakout_age
  - breakout_yprr
  - breakout_yptpa
  - best_contested_catch_rate

Contested slots — test all plausible alternatives:
  Slot A (QB confidence / efficiency): career_targeted_qb_rating vs alternatives
  Slot B (route efficiency): career_yprr vs alternatives
  Slot C (catch reliability): career_catch_pct_adot_adj vs alternatives
  Slot D (elusiveness / YAC): career_avoided_tackles_pg vs alternatives

For each candidate:
  1. Univariate signal (Spearman, AUC)
  2. Era stability
  3. Residual signal after locked features
  4. Residual signal after locked + other slot winners
  5. Collinearity with locked features
  6. Elastic net survival
  7. Bootstrap residual (1000 iterations)
  8. Total feature set signal when swapped in
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

# --- Load data ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
max_bo = df["breakout_age"].max()
df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

LOCKED = [
    "draft_capital",
    "breakout_age",
    "breakout_yprr",
    "breakout_yptpa",
    "best_contested_catch_rate",
]

# Slots and their candidates
SLOTS = {
    "A: QB Confidence / Route Quality": [
        "career_targeted_qb_rating",
        "best_targeted_qb_rating",
        "best2_targeted_qb_rating",
        "career_grades_pass_route",
        "best2_grades_pass_route",
        "best_grades_pass_route",
        "career_grades_offense",
        "best2_grades_offense",
        "best_grades_offense",
    ],
    "B: Route Efficiency": [
        "career_yprr",
        "best2_yprr",
        "career_yards_pg",
        "best2_yards_pg",
        "best_yards_pg",
        "career_first_downs_per_route",
        "best2_first_downs_per_route",
        "best_yards_per_team_pass_att",
    ],
    "C: Catch Reliability": [
        "career_catch_pct_adot_adj",
        "best2_catch_pct_adot_adj",
        "career_caught_percent",
        "best2_caught_percent",
        "best_caught_percent",
    ],
    "D: Elusiveness / YAC": [
        "career_avoided_tackles_pg",
        "best_avoided_tackles_pg",
        "best2_avoided_tackles_pg",
        "career_avoided_tackles_per_rec",
        "best2_avoided_tackles_per_rec",
        "career_yards_after_catch_pg",
        "best_yards_after_catch_pg",
        "best2_yards_after_catch_pg",
    ],
}

# Also test: should we even HAVE 4 slots? Test "none" for each slot.

# Filter to complete cases for all candidates
all_candidates = LOCKED.copy()
for candidates in SLOTS.values():
    all_candidates.extend(candidates)
all_candidates = list(set(all_candidates))
all_candidates = [c for c in all_candidates if c in df.columns]

df_full = df.dropna(subset=["tier_ordinal"] + all_candidates).copy()
df_full["tier_ordinal"] = df_full["tier_ordinal"].astype(int)
y = df_full["tier_ordinal"].values
hit = (y >= 3).astype(int)
years = df_full["draft_year"].values

print(f"Dataset: {len(df_full)} players")
print(f"Draft years: {sorted(df_full['draft_year'].unique())}")
print(f"Locked features: {LOCKED}")
print()

scaler = StandardScaler()
ridge = Ridge(alpha=1.0)
rng = np.random.RandomState(42)


def univariate(feat):
    vals = df_full[feat].values
    sp, sp_p = spearmanr(vals, y)
    auc = roc_auc_score(hit, vals)
    if auc < 0.5:
        auc = 1 - auc
    return float(sp), float(auc)


def era_stability(feat):
    mid_years = sorted(df_full["draft_year"].unique())
    mid = mid_years[len(mid_years) // 2]
    early = df_full[df_full["draft_year"] <= mid]
    late = df_full[df_full["draft_year"] > mid]
    sp_e, _ = spearmanr(early[feat].values, early["tier_ordinal"].values)
    sp_l, _ = spearmanr(late[feat].values, late["tier_ordinal"].values)
    return float(sp_e), float(sp_l), abs(float(sp_e) - float(sp_l))


def residual_signal(feat, control_features):
    """Spearman of feat with residuals after regressing y on control_features."""
    ctrl = [c for c in control_features if c in df_full.columns]
    if not ctrl:
        sp, _ = spearmanr(df_full[feat].values, y)
        return float(sp)
    X_ctrl = scaler.fit_transform(df_full[ctrl].values)
    ridge.fit(X_ctrl, y)
    resid = y - ridge.predict(X_ctrl)
    sp, _ = spearmanr(df_full[feat].values, resid)
    return float(sp)


def max_collinearity(feat, reference_features):
    """Max absolute Spearman correlation with reference features."""
    ref = [c for c in reference_features if c in df_full.columns and c != feat]
    if not ref:
        return 0.0
    return max(abs(float(spearmanr(df_full[feat].values, df_full[r].values)[0])) for r in ref)


def bootstrap_residual(feat, control_features, n_boot=1000):
    """Bootstrap distribution of residual Spearman."""
    ctrl = [c for c in control_features if c in df_full.columns]
    results = []
    for _ in range(n_boot):
        idx = rng.choice(len(df_full), size=len(df_full), replace=True)
        if ctrl:
            X_b = scaler.fit_transform(df_full[ctrl].values[idx])
            y_b = y[idx]
            ridge.fit(X_b, y_b)
            resid = y_b - ridge.predict(X_b)
        else:
            resid = y[idx]
        sp, _ = spearmanr(df_full[feat].values[idx], resid)
        results.append(sp)
    arr = np.array(results)
    return float(arr.mean()), float(arr.std()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr > 0).mean())


def enet_survival(feat, all_feats_for_enet):
    """How many of 3 regularization strengths does feat survive?"""
    feats = [f for f in all_feats_for_enet if f in df_full.columns]
    idx = feats.index(feat) if feat in feats else -1
    if idx == -1:
        return 0, []
    X = scaler.fit_transform(df_full[feats].values)
    count = 0
    coefs = []
    for C in [0.01, 0.1, 1.0]:
        model = LogisticRegression(
            penalty="elasticnet", C=C, l1_ratio=0.5,
            solver="saga", max_iter=10000, random_state=42, class_weight="balanced",
        )
        model.fit(X, hit)
        c = model.coef_[0][idx]
        coefs.append(c)
        if abs(c) > 1e-6:
            count += 1
    return count, coefs


def loo_auc(feature_set):
    """Leave-one-year-out AUC for Elite+ threshold."""
    feats = [f for f in feature_set if f in df_full.columns]
    all_preds, all_true = [], []
    for fold_year in sorted(df_full["draft_year"].unique()):
        tr = df_full["draft_year"] != fold_year
        val = df_full["draft_year"] == fold_year
        y_tr = (df_full.loc[tr, "tier_ordinal"].values >= 3).astype(int)
        y_val = (df_full.loc[val, "tier_ordinal"].values >= 3).astype(int)
        if y_tr.sum() < 2 or y_val.sum() == 0 or y_val.sum() == len(y_val):
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_full.loc[tr, feats].values)
        X_val = sc.transform(df_full.loc[val, feats].values)
        model = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        model.fit(X_tr, y_tr)
        all_preds.extend(model.predict_proba(X_val)[:, 1])
        all_true.extend(y_val)
    return roc_auc_score(np.array(all_true), np.array(all_preds))


def total_feature_set_signal(feature_set):
    """Sum of |residual Spearman| for each feature after all others."""
    feats = [f for f in feature_set if f in df_full.columns]
    total = 0.0
    for feat in feats:
        others = [f for f in feats if f != feat]
        sp = abs(residual_signal(feat, others))
        total += sp
    return total


# ============================================================
# Run analysis for each slot
# ============================================================

# First: baseline with just locked features
print("=" * 80)
print("BASELINE: LOCKED FEATURES ONLY")
print("=" * 80)
base_auc = loo_auc(LOCKED)
base_signal = total_feature_set_signal(LOCKED)
print(f"  LOO-AUC: {base_auc:.3f}")
print(f"  Total signal: {base_signal:.3f}")
print()

for slot_name, candidates in SLOTS.items():
    print("=" * 80)
    print(f"SLOT {slot_name}")
    print("=" * 80)

    # Filter to candidates that exist
    candidates = [c for c in candidates if c in df_full.columns]

    # All features for enet context
    enet_features = LOCKED + candidates

    print(f"\n  {'Candidate':<40s} {'Sp':>6s} {'AUC':>6s} {'Drift':>6s} "
          f"{'Resid':>6s} {'MaxCor':>6s} {'Enet':>5s} "
          f"{'Boot+':>6s} {'SetSig':>7s} {'LOOAUC':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*6} {'-'*5} "
          f"{'-'*6} {'-'*7} {'-'*7}")

    slot_results = []

    for cand in candidates:
        sp, auc = univariate(cand)
        sp_e, sp_l, drift = era_stability(cand)
        resid = residual_signal(cand, LOCKED)
        max_col = max_collinearity(cand, LOCKED)
        enet_count, enet_coefs = enet_survival(cand, enet_features)
        boot_mean, boot_std, boot_lo, boot_hi, boot_pos = bootstrap_residual(cand, LOCKED)

        # Test this candidate added to locked set
        test_set = LOCKED + [cand]
        sig = total_feature_set_signal(test_set)
        la = loo_auc(test_set)

        slot_results.append({
            "candidate": cand,
            "sp": sp, "auc": auc, "drift": drift,
            "resid": resid, "max_col": max_col, "enet": enet_count,
            "boot_pos": boot_pos, "set_sig": sig, "loo_auc": la,
            "sp_e": sp_e, "sp_l": sp_l,
            "boot_mean": boot_mean, "boot_lo": boot_lo, "boot_hi": boot_hi,
        })

        print(f"  {cand:<40s} {sp:>+6.3f} {auc:>6.3f} {drift:>6.3f} "
              f"{resid:>+6.3f} {max_col:>6.3f} {enet_count:>5d} "
              f"{boot_pos:>6.1%} {sig:>7.3f} {la:>7.3f}")

    # Also test "NONE" - just locked features for this slot
    print(f"  {'[NONE - locked only]':<40s} {'':>6s} {'':>6s} {'':>6s} "
          f"{'':>6s} {'':>6s} {'':>5s} "
          f"{'':>6s} {base_signal:>7.3f} {base_auc:>7.3f}")

    # Detailed bootstrap for top 3
    sorted_results = sorted(slot_results, key=lambda x: x["set_sig"], reverse=True)
    print(f"\n  Top 3 by total feature set signal (bootstrap details):")
    for r in sorted_results[:3]:
        print(f"    {r['candidate']:<38s}  resid={r['boot_mean']:+.3f} "
              f"[{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}]  "
              f"pos={r['boot_pos']:.1%}  "
              f"era: early={r['sp_e']:+.3f} late={r['sp_l']:+.3f}")
    print()


# ============================================================
# Cross-slot interactions: test all combinations of slot winners
# ============================================================
print("=" * 80)
print("CROSS-SLOT INTERACTION: BEST COMBINATIONS")
print("=" * 80)

# Get top 2 from each slot by set_sig
slot_tops = {}
for slot_name, candidates in SLOTS.items():
    candidates = [c for c in candidates if c in df_full.columns]
    results = []
    for cand in candidates:
        test_set = LOCKED + [cand]
        sig = total_feature_set_signal(test_set)
        results.append((cand, sig))
    results.sort(key=lambda x: x[1], reverse=True)
    slot_tops[slot_name] = [r[0] for r in results[:3]]

# Test combinations
from itertools import product

slot_keys = list(SLOTS.keys())
combos = list(product(*[slot_tops[k] + [None] for k in slot_keys]))

print(f"\n  Testing {len(combos)} combinations of top candidates + None...")
print(f"\n  {'SlotA':<30s} {'SlotB':<25s} {'SlotC':<25s} {'SlotD':<25s} {'Signal':>7s} {'LOOAUC':>7s}")
print(f"  {'-'*30} {'-'*25} {'-'*25} {'-'*25} {'-'*7} {'-'*7}")

combo_results = []
for combo in combos:
    feature_set = LOCKED + [c for c in combo if c is not None]
    sig = total_feature_set_signal(feature_set)
    la = loo_auc(feature_set)
    combo_results.append((combo, sig, la, len(feature_set)))

# Sort by LOO-AUC
combo_results.sort(key=lambda x: x[2], reverse=True)

for combo, sig, la, n_feats in combo_results[:25]:
    labels = []
    for c in combo:
        if c is None:
            labels.append("[none]")
        else:
            labels.append(c.replace("career_", "c_").replace("best2_", "b2_").replace("best_", "b_"))
    print(f"  {labels[0]:<30s} {labels[1]:<25s} {labels[2]:<25s} {labels[3]:<25s} {sig:>7.3f} {la:>7.3f}")

print(f"\n  ...bottom 5:")
for combo, sig, la, n_feats in combo_results[-5:]:
    labels = []
    for c in combo:
        if c is None:
            labels.append("[none]")
        else:
            labels.append(c.replace("career_", "c_").replace("best2_", "b2_").replace("best_", "b_"))
    print(f"  {labels[0]:<30s} {labels[1]:<25s} {labels[2]:<25s} {labels[3]:<25s} {sig:>7.3f} {la:>7.3f}")

# Locked-only baseline
print(f"\n  {'[locked only]':<107s} {base_signal:>7.3f} {base_auc:>7.3f}")
