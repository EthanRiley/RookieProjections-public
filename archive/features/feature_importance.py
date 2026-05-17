#!/usr/bin/env python3
"""
Feature importance screening (Layer 1 from design doc).

For each candidate feature, computes:
  - Spearman rank correlation with ordinal tier
  - Mutual information with tier
  - Standalone AUC for "Elite or better" vs rest

Reads:
  - wr_data/wr_dynasty_value_with_college.csv
Outputs:
  - Prints ranked feature table
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0,
    "Flex": 1,
    "Starter": 2,
    "Elite": 3,
    "Stud": 4,
    "League-Winner": 5,
}

# Binary threshold: "Elite or better" (Elite, Stud, League-Winner)
BINARY_THRESHOLD = 3

df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))

# Map tier to ordinal
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)

# Drop rows with no tier
df = df.dropna(subset=["tier_ordinal"])

# Binary target
df["is_hit"] = (df["tier_ordinal"] >= BINARY_THRESHOLD).astype(int)

# Identify feature columns (career_ and best_ prefixed, numeric only)
feature_cols = [
    c for c in df.columns
    if (c.startswith("career_") or c.startswith("best_") or c.startswith("best2_"))
    and pd.api.types.is_numeric_dtype(df[c])
]
# Include non-prefixed features
for extra in ["draft_capital", "breakout_age", "breakout_yptpa", "breakout_yprr",
              "best_yards_per_team_pass_att"]:
    if extra in df.columns and extra not in feature_cols:
        feature_cols.append(extra)

# Impute breakout features
if "breakout_age" in df.columns:
    max_bo = df["breakout_age"].max()
    df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
if "breakout_yptpa" in df.columns:
    df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
if "breakout_yprr" in df.columns:
    df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

print(f"Players: {len(df)}")
print(f"Hit rate (Elite+): {df['is_hit'].mean():.1%}")
print(f"Candidate features: {len(feature_cols)}\n")

results = []

for col in feature_cols:
    valid = df[[col, "tier_ordinal", "is_hit"]].dropna()
    if len(valid) < 30:
        continue

    x = valid[col].values
    y_ord = valid["tier_ordinal"].values
    y_bin = valid["is_hit"].values

    # Spearman rank correlation
    spear_corr, spear_p = spearmanr(x, y_ord)

    # Mutual information (discrete target)
    mi = mutual_info_classif(
        x.reshape(-1, 1), y_ord, discrete_features=False, random_state=42
    )[0]

    # Standalone AUC
    if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
        auc = roc_auc_score(y_bin, x)
        # Flip if negative correlation
        if auc < 0.5:
            auc = 1 - auc
    else:
        auc = np.nan

    results.append({
        "feature": col,
        "spearman": round(spear_corr, 3),
        "spearman_p": round(spear_p, 4),
        "mutual_info": round(mi, 4),
        "auc": round(auc, 3) if not np.isnan(auc) else np.nan,
        "n": len(valid),
    })

results_df = pd.DataFrame(results)

# Composite score: average of normalized ranks
for metric in ["spearman", "mutual_info", "auc"]:
    col = f"{metric}_rank"
    vals = results_df[metric].abs() if metric == "spearman" else results_df[metric]
    results_df[col] = vals.rank(ascending=False)

results_df["composite_rank"] = (
    results_df["spearman_rank"] + results_df["mutual_info_rank"] + results_df["auc_rank"]
) / 3

results_df = results_df.sort_values("composite_rank").reset_index(drop=True)

# Print full table
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 160)
print(results_df[["feature", "spearman", "mutual_info", "auc", "composite_rank", "n"]].to_string(index=False))
