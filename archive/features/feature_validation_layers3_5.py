#!/usr/bin/env python3
"""
Feature validation Layers 3-5 from design doc.

Layer 3: Elastic Net ordinal regression at multiple regularization strengths
Layer 4: XGBoost permutation importance
Layer 5: Era stability (early vs late draft classes)

Reads:  wr_data/wr_dynasty_value_with_college.csv
Outputs: Prints combined feature evaluation table + saves to wr_data/feature_evaluation.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0,
    "Flex": 1,
    "Starter": 2,
    "Elite": 3,
    "Stud": 4,
    "League-Winner": 5,
}

BINARY_THRESHOLD = 3  # Elite or better

# --- Load and prep data ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
df = df.dropna(subset=["tier_ordinal"])
df["is_hit"] = (df["tier_ordinal"] >= BINARY_THRESHOLD).astype(int)

feature_cols = [
    c for c in df.columns
    if (c.startswith("career_") or c.startswith("best_") or c.startswith("best2_"))
    and pd.api.types.is_numeric_dtype(df[c])
]
# Include draft_capital, breakout_age
extra = [c for c in ["draft_capital", "draft_age", "breakout_age", "breakout_yptpa", "breakout_yprr"]
         if c in df.columns and c not in feature_cols]
all_features = extra + feature_cols

# Impute breakout features
if "breakout_age" in df.columns:
    max_bo = df["breakout_age"].max()
    df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
if "breakout_yptpa" in df.columns:
    df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
if "breakout_yprr" in df.columns:
    df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

# Complete cases only
df_complete = df[all_features + ["tier_ordinal", "is_hit", "draft_year", "name"]].dropna().copy()
print(f"Complete cases: {len(df_complete)}")
print(f"Features: {len(all_features)}")
print(f"Hit rate: {df_complete['is_hit'].mean():.1%}\n")

X = df_complete[all_features].values
y_ord = df_complete["tier_ordinal"].values.astype(int)
y_bin = df_complete["is_hit"].values.astype(int)
draft_years = df_complete["draft_year"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# LAYER 3: Elastic Net at multiple regularization strengths
# ============================================================
print("=" * 70)
print("LAYER 3: Elastic Net Ordinal Regression")
print("=" * 70)

# Use cumulative binary approach: fit P(tier >= k) for each threshold
# Then look at which features survive across thresholds and lambdas
alphas = [0.01, 0.1, 1.0]  # C = 1/alpha for LogisticRegression
l1_ratios = [0.5]  # elastic net mixing

layer3_results = {feat: {f"enet_C{C}": 0.0 for C in alphas} for feat in all_features}

for C_val in alphas:
    # Fit binary classifier for "Elite or better"
    model = LogisticRegression(
        penalty="elasticnet",
        C=C_val,
        l1_ratio=0.5,
        solver="saga",
        max_iter=10000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_scaled, y_bin)
    coefs = model.coef_[0]

    for i, feat in enumerate(all_features):
        layer3_results[feat][f"enet_C{C_val}"] = round(coefs[i], 4)

layer3_df = pd.DataFrame(layer3_results).T
layer3_df.index.name = "feature"
layer3_df = layer3_df.reset_index()

# Count how many lambda settings each feature survives (non-zero coef)
layer3_df["enet_survive_count"] = sum(
    (layer3_df[f"enet_C{C}"].abs() > 1e-6).astype(int) for C in alphas
)

print("\nFeatures surviving at each regularization strength:")
for C_val in alphas:
    col = f"enet_C{C_val}"
    surviving = layer3_df[layer3_df[col].abs() > 1e-6]
    print(f"\n  C={C_val} ({len(surviving)} features survive):")
    top = surviving.reindex(surviving[col].abs().sort_values(ascending=False).index)
    for _, row in top.head(15).iterrows():
        print(f"    {row['feature']:45s} coef={row[col]:+.4f}")

# ============================================================
# LAYER 4: XGBoost Permutation Importance
# ============================================================
print("\n" + "=" * 70)
print("LAYER 4: XGBoost Permutation Importance")
print("=" * 70)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    scale_pos_weight=(y_bin == 0).sum() / max((y_bin == 1).sum(), 1),
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
xgb.fit(X, y_bin)

# Permutation importance (not built-in feature importance)
perm_result = permutation_importance(
    xgb, X, y_bin, n_repeats=30, random_state=42, scoring="roc_auc"
)

perm_df = pd.DataFrame({
    "feature": all_features,
    "perm_importance_mean": np.round(perm_result.importances_mean, 4),
    "perm_importance_std": np.round(perm_result.importances_std, 4),
})

perm_df = perm_df.sort_values("perm_importance_mean", ascending=False)
print("\nTop 20 features by permutation importance:")
for _, row in perm_df.head(20).iterrows():
    print(f"  {row['feature']:45s} {row['perm_importance_mean']:+.4f} +/- {row['perm_importance_std']:.4f}")

# ============================================================
# LAYER 5: Era Stability
# ============================================================
print("\n" + "=" * 70)
print("LAYER 5: Era Stability (2016-2019 vs 2020-2024)")
print("=" * 70)

early_mask = draft_years <= 2019
late_mask = draft_years >= 2020

stability_results = []

for i, feat in enumerate(all_features):
    x_all = X[:, i]

    # Early era
    x_early = x_all[early_mask]
    y_early = y_ord[early_mask]
    y_bin_early = y_bin[early_mask]

    # Late era
    x_late = x_all[late_mask]
    y_late = y_ord[late_mask]
    y_bin_late = y_bin[late_mask]

    # Spearman in each era
    sp_early, _ = spearmanr(x_early, y_early) if len(x_early) > 10 else (np.nan, np.nan)
    sp_late, _ = spearmanr(x_late, y_late) if len(x_late) > 10 else (np.nan, np.nan)

    # AUC in each era
    def safe_auc(y_true, x_score):
        if y_true.sum() == 0 or y_true.sum() == len(y_true) or len(y_true) < 10:
            return np.nan
        auc = roc_auc_score(y_true, x_score)
        return auc if auc >= 0.5 else 1 - auc

    auc_early = safe_auc(y_bin_early, x_early)
    auc_late = safe_auc(y_bin_late, x_late)

    # Stability = 1 - abs difference in Spearman between eras
    sp_diff = abs(sp_early - sp_late) if not (np.isnan(sp_early) or np.isnan(sp_late)) else np.nan

    stability_results.append({
        "feature": feat,
        "spearman_early": round(sp_early, 3) if not np.isnan(sp_early) else np.nan,
        "spearman_late": round(sp_late, 3) if not np.isnan(sp_late) else np.nan,
        "spearman_diff": round(sp_diff, 3) if not np.isnan(sp_diff) else np.nan,
        "auc_early": round(auc_early, 3) if not np.isnan(auc_early) else np.nan,
        "auc_late": round(auc_late, 3) if not np.isnan(auc_late) else np.nan,
    })

stability_df = pd.DataFrame(stability_results)

print(f"\nEarly era (<=2019): {early_mask.sum()} players, {y_bin[early_mask].sum()} hits")
print(f"Late era  (>=2020): {late_mask.sum()} players, {y_bin[late_mask].sum()} hits")

print("\nMost STABLE features (smallest Spearman difference across eras):")
stable = stability_df.dropna(subset=["spearman_diff"]).sort_values("spearman_diff")
for _, row in stable.head(15).iterrows():
    print(f"  {row['feature']:45s} early={row['spearman_early']:+.3f}  late={row['spearman_late']:+.3f}  diff={row['spearman_diff']:.3f}")

print("\nMost UNSTABLE features (largest Spearman difference across eras):")
for _, row in stable.tail(10).iloc[::-1].iterrows():
    print(f"  {row['feature']:45s} early={row['spearman_early']:+.3f}  late={row['spearman_late']:+.3f}  diff={row['spearman_diff']:.3f}")

# ============================================================
# COMBINED FEATURE EVALUATION TABLE
# ============================================================
print("\n" + "=" * 70)
print("COMBINED FEATURE EVALUATION TABLE")
print("=" * 70)

# Recompute Layer 1 metrics on complete cases for consistency
layer1_results = []
for i, feat in enumerate(all_features):
    x = X[:, i]
    sp, sp_p = spearmanr(x, y_ord)
    mi = mutual_info_classif(x.reshape(-1, 1), y_ord, discrete_features=False, random_state=42)[0]
    auc = roc_auc_score(y_bin, x) if y_bin.sum() > 0 else np.nan
    if auc < 0.5:
        auc = 1 - auc
    layer1_results.append({
        "feature": feat,
        "spearman": round(sp, 3),
        "mutual_info": round(mi, 4),
        "auc": round(auc, 3),
    })

combined = pd.DataFrame(layer1_results)

# Merge Layer 3
enet_cols = ["feature", "enet_survive_count"] + [f"enet_C{C}" for C in alphas]
combined = combined.merge(layer3_df[enet_cols], on="feature")

# Merge Layer 4
combined = combined.merge(perm_df[["feature", "perm_importance_mean"]], on="feature")

# Merge Layer 5
combined = combined.merge(stability_df[["feature", "spearman_diff", "spearman_early", "spearman_late"]], on="feature")

# Composite ranking
for metric, ascending in [
    ("spearman", True),      # rank by abs value, descending -> ascending=True for rank
    ("mutual_info", True),
    ("auc", True),
    ("perm_importance_mean", True),
    ("spearman_diff", False),  # lower diff = more stable = better
]:
    vals = combined[metric].abs() if metric == "spearman" else combined[metric]
    if ascending:
        combined[f"{metric}_rank"] = vals.rank(ascending=False)
    else:
        combined[f"{metric}_rank"] = vals.rank(ascending=True)

combined["composite_rank"] = (
    combined["spearman_rank"]
    + combined["mutual_info_rank"]
    + combined["auc_rank"]
    + combined["perm_importance_mean_rank"]
    + combined["spearman_diff_rank"]
) / 5

combined = combined.sort_values("composite_rank").reset_index(drop=True)

# Print final table
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print_cols = [
    "feature", "spearman", "mutual_info", "auc",
    "enet_survive_count", "enet_C0.1",
    "perm_importance_mean", "spearman_diff",
    "composite_rank",
]
print("\n" + combined[print_cols].to_string(index=False))

# Save full table
out_path = os.path.join(DATA_DIR, "feature_evaluation.csv")
combined.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
