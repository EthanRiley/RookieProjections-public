#!/usr/bin/env python3
"""
Combine predictions from all 6 model variants into a single CSV.

Models:
  - Bayesian Full (BF) / College-Only (BC)
  - XGBoost Full (XF) / College-Only (XC)
  - Elastic Net Full (EF) / College-Only (EC)

Outputs:
  - wr_data/combined_holdout_predictions.csv
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_COLS = ["P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)"]
TIER_SHORT = ["Bust", "Flex", "Starter", "Elite", "Stud", "LW"]

MODELS = {
    "BF": "bayesian_full_holdout_predictions.csv",
    "BC": "bayesian_college_only_holdout_predictions.csv",
    "XF": "xgb_full_holdout_predictions.csv",
    "XC": "xgb_college_only_holdout_predictions.csv",
    "EF": "enet_full_holdout_predictions.csv",
    "EC": "enet_college_only_holdout_predictions.csv",
}

# Load all model predictions, index by name
dfs = {}
for key, filename in MODELS.items():
    path = os.path.join(DATA_DIR, filename)
    dfs[key] = pd.read_csv(path).set_index("name")

# Use Bayesian Full as the base (has player info)
base_key = "BF"
base = dfs[base_key][["draft_year", "pick", "computed_tier"]].copy()

# Add each model's tier probabilities and expected tier
for key, mdf in dfs.items():
    for tier_col, short in zip(TIER_COLS, TIER_SHORT):
        base[f"{key}_{short}"] = mdf[tier_col]
    base[f"{key}_expected"] = mdf["expected_tier"]
    base[f"{key}_predicted"] = mdf["predicted_tier"]

# Edge: college-only expected minus full expected (per model family)
base["bayes_edge"] = (base["BC_expected"] - base["BF_expected"]).round(3)
base["xgb_edge"] = (base["XC_expected"] - base["XF_expected"]).round(3)
base["enet_edge"] = (base["EC_expected"] - base["EF_expected"]).round(3)

# Sort by Bayesian Full expected tier
base = base.sort_values("BF_expected", ascending=False)

# Save
out_path = os.path.join(DATA_DIR, "outputs", "combined_holdout_predictions.csv")
base.to_csv(out_path)
print(f"Saved {len(base)} players to {out_path}")
print(f"Columns: {len(base.columns)}")

# Print summary view
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

summary = base[["draft_year", "pick", "computed_tier",
                "BF_expected", "BC_expected", "bayes_edge",
                "XF_expected", "XC_expected", "xgb_edge",
                "EF_expected", "EC_expected", "enet_edge"]].copy()

summary.columns = ["Year", "Pick", "Actual",
                    "BF", "BC", "B_edge",
                    "XF", "XC", "X_edge",
                    "EF", "EC", "E_edge"]

for col in ["BF", "BC", "B_edge", "XF", "XC", "X_edge", "EF", "EC", "E_edge"]:
    summary[col] = summary[col].round(2)

print("\n" + summary.to_string())
