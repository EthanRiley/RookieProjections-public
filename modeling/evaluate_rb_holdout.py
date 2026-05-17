#!/usr/bin/env python3
"""
RB Holdout Evaluation with ensemble weight grid search.

Trains on 2016-2021, evaluates on 2022-2024 holdout.
Uses shared rb_model.py module for all model logic.

Fixes from v1 audit:
  - Composite z-scores fit on training data only (no holdout leakage)
  - peak2_ypa fallback to peak_ypa for single-season players
  - Component probability columns in output
  - Consistent output schema with predict_rb_prospects.py

Outputs:
  - rb_data/outputs/holdout_predictions_rb_v1.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

from modeling.rb_model import (
    TIER_ORDER, TIER_NAMES, COLLEGE_FEATURES, COMPOSITE_DEFS, N_TIERS,
    THRESHOLDS, THRESHOLD_LABELS,
    W_BAYES, W_XGB,
    dc_log, apply_feature_fallbacks, compute_composites,
    train_xgb, train_bayesian, blend,
    evaluate, compute_metrics, composite_score,
    train_full_and_college, build_pred_df,
)
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(PROJECT_ROOT, "rb_data")
HOLDOUT_YEARS = [2022, 2023, 2024]

# Grid search weights
WEIGHT_GRID = [round(w / 100, 2) for w in range(0, 105, 5)]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"]).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    df["is_resolved"] = df["is_resolved"].astype(bool)
    df = df[df["is_resolved"]].copy()
    df["draft_capital"] = df["pick"].apply(dc_log)

    # Apply peak2_ypa fallback BEFORE composite computation
    df = apply_feature_fallbacks(df)

    return df


def main():
    df = load_data()

    print("=" * 70)
    print("RB HOLDOUT EVALUATION (with z-score leakage fix)")
    print(f"Features: draft_capital + {COLLEGE_FEATURES}")
    print(f"Ensemble weights: grid searched (0% to 100% Bayesian by 5%)")
    print("=" * 70)

    # Split BEFORE computing composites to prevent z-score leakage
    train_mask = ~df["draft_year"].isin(HOLDOUT_YEARS)
    df, scaler_dict = compute_composites(df, train_mask=train_mask)

    all_feats = ["draft_capital"] + COLLEGE_FEATURES
    sub = df.dropna(subset=all_feats).copy()
    train = sub[~sub["draft_year"].isin(HOLDOUT_YEARS)].copy()
    holdout = sub[sub["draft_year"].isin(HOLDOUT_YEARS)].copy()

    print(f"\nTraining set: {len(train)} players ({sorted(train['draft_year'].unique())})")
    print(f"Holdout set:  {len(holdout)} players ({sorted(holdout['draft_year'].unique())})")
    print(f"\nTier distribution (train):")
    for tier_name in TIER_ORDER:
        n = (train["computed_tier"] == tier_name).sum()
        print(f"  {tier_name:15s} {n:3d} ({n/len(train):.1%})")

    # --- Train models ---
    full_probs, college_probs, scaler, components = train_full_and_college(train, holdout)
    actual = holdout["tier_ordinal"].values

    # --- Evaluate ensemble ---
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION (2022-2024)")
    print("=" * 70)

    evaluate(full_probs, actual, "ENSEMBLE FULL (30/70)")
    evaluate(college_probs, actual, "ENSEMBLE COLLEGE-ONLY (30/70)")

    # --- Also evaluate individual components ---
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL COMPONENTS")
    print("=" * 70)
    print(f"\n  {'Model':<30s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} {'>=Stud':>8s} {'>=Start':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, probs in [("Bayesian Full", components["bayes_full"]),
                        ("XGBoost Full", components["xgb_full"]),
                        ("Bayesian College", components["bayes_college"]),
                        ("XGBoost College", components["xgb_college"])]:
        m = compute_metrics(probs, actual)
        print(f"  {name:<30s} {m['log_loss']:>8.4f} {m['brier']:>8.4f} "
              f"{m.get('>=Elite_auc', float('nan')):>8.3f} {m.get('>=Stud_auc', float('nan')):>8.3f} "
              f"{m.get('>=Starter_auc', float('nan')):>8.3f}")

    # --- Weight grid search ---
    print("\n" + "=" * 70)
    print("ENSEMBLE WEIGHT GRID SEARCH")
    print(f"Testing {len(WEIGHT_GRID)} Bayesian weights (0.00 to 1.00 by 0.05)")
    print("=" * 70)

    print(f"\n  {'W_Bayes':>8s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} {'>=Stud':>8s} {'>=Start':>8s} {'Composite':>10s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    best_w = None
    best_score = -1

    for w_bayes in WEIGHT_GRID:
        w_xgb = 1.0 - w_bayes
        combo = blend(components["bayes_full"], components["xgb_full"], w_bayes, w_xgb)
        m = compute_metrics(combo, actual)
        score = composite_score(m)

        if score > best_score:
            best_score = score
            best_w = w_bayes

        marker = " <--" if w_bayes == best_w else ""
        print(f"  {w_bayes:>8.2f} {m['log_loss']:>8.4f} {m['brier']:>8.4f} "
              f"{m.get('>=Elite_auc', float('nan')):>8.3f} {m.get('>=Stud_auc', float('nan')):>8.3f} "
              f"{m.get('>=Starter_auc', float('nan')):>8.3f} {score:>10.4f}{marker}")

    print(f"\n  --> Best weight: {best_w:.0%} Bayesian / {1 - best_w:.0%} XGBoost")

    # Recompute best ensemble (may differ from default 30/70)
    best_full = blend(components["bayes_full"], components["xgb_full"], best_w, 1 - best_w)
    best_college = blend(components["bayes_college"], components["xgb_college"], best_w, 1 - best_w)
    m_best = compute_metrics(best_full, actual)
    print(f"      LogLoss={m_best['log_loss']:.4f}  Brier={m_best['brier']:.4f}  "
          f">=Elite={m_best.get('>=Elite_auc', float('nan')):.3f}  Composite={composite_score(m_best):.4f}")

    # --- Build and save output using default weights ---
    holdout["computed_tier"] = holdout["computed_tier"]  # already present
    out = build_pred_df(holdout, full_probs, college_probs, components=components)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*70}")
    print(f"HOLDOUT PREDICTIONS ({W_BAYES:.0%} Bayesian / {W_XGB:.0%} XGBoost)")
    print(f"{'='*70}")

    display = out[["name", "draft_year", "pick", "computed_tier",
                    "P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)",
                    "expected_tier", "college_expected_tier", "edge"]].copy()
    display.columns = ["Name", "Year", "Pick", "Actual",
                        "Bust", "Flex", "Start", "Elite", "Stud", "LW",
                        "E[full]", "E[college]", "Edge"]
    print(display.to_string(index=False))

    out_path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_rb_v1.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    main()
