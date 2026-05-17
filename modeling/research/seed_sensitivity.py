#!/usr/bin/env python3
"""
Seed sensitivity analysis for WR v12 model.

Runs the full holdout evaluation with multiple random seeds to quantify
how stable the reported metrics are across the randomness space.

Outputs:
  - wr_data/reports/seed_sensitivity_report.md
  - wr_data/outputs/seed_sensitivity_results.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

from modeling.wr_model import (
    TIER_ORDER, COLLEGE_FEATURES, N_TIERS,
    dc_log, build_catch_composite, compute_metrics,
    train_full_and_college,
)
from aggregation.aggregate_college_stats import (
    load_all_grades, build_lookups, aggregate_player, fit_adot_regression,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")
HOLDOUT_YEARS = [2022, 2023, 2024]
SEEDS = [42, 123, 456, 789, 1337, 2024, 7777, 9999, 31415, 54321]


def load_and_prepare():
    """Load data once, return train_df and holdout_df."""
    print("Loading grades and aggregating features...")
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)

    print("Re-aggregating with peak-gated features...")
    for i, (_, row) in enumerate(df.iterrows()):
        result = aggregate_player(
            all_grades, row["name"], row["draft_year"],
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            team_games_lookup=team_games_lookup,
            adot_coef=adot_coef,
        )
        for col in ["pg_yprr_graduated", "pg_catch_pct_adot_adj_graduated",
                    "career_catch_pct_adot_adj"]:
            if col in result:
                df.at[df.index[i], col] = result[col]

    train_mask = ~df["draft_year"].isin(HOLDOUT_YEARS)
    df["catch_composite"], _ = build_catch_composite(df, train_mask=train_mask)

    all_features = ["draft_capital"] + COLLEGE_FEATURES
    df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    df["draft_capital"] = df["pick"].apply(dc_log)

    train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
    holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

    print(f"Training: {len(train_df)}, Holdout: {len(holdout_df)}")
    return train_df, holdout_df


def run_seed(train_df, holdout_df, seed):
    """Run full evaluation with a given seed."""
    full_probs, college_probs, _, _ = train_full_and_college(
        train_df, holdout_df, random_seed=seed,
    )
    actual = holdout_df["tier_ordinal"].values
    full_m = compute_metrics(full_probs, actual)
    college_m = compute_metrics(college_probs, actual)
    return {
        "seed": seed,
        **{f"full_{k}": v for k, v in full_m.items()},
        **{f"college_{k}": v for k, v in college_m.items()},
    }


if __name__ == "__main__":
    train_df, holdout_df = load_and_prepare()

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'=' * 70}")
        row = run_seed(train_df, holdout_df, seed)
        results.append(row)

        # Print running summary
        print(f"  LogLoss={row['full_log_loss']:.4f}  Brier={row['full_brier']:.4f}  "
              f"Elite AUC={row['full_>=Elite_auc']:.3f}")

    results_df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 70)
    print("SEED SENSITIVITY SUMMARY")
    print("=" * 70)

    key_metrics = ["full_log_loss", "full_brier", "full_>=Elite_auc", "full_>=Stud_auc", "full_>=LW_auc"]
    print(f"\n  {'Metric':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Range':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for col in key_metrics:
        vals = results_df[col]
        label = col.replace("full_", "")
        print(f"  {label:<25s} {vals.mean():>8.4f} {vals.std():>8.4f} "
              f"{vals.min():>8.4f} {vals.max():>8.4f} {vals.max()-vals.min():>8.4f}")

    # Save results
    out_csv = os.path.join(DATA_DIR, "outputs", "seed_sensitivity_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    # Generate report
    report_lines = [
        "# Seed Sensitivity Report",
        "",
        f"**Seeds tested**: {len(SEEDS)}",
        f"**Seeds**: {SEEDS}",
        "",
        "## Full Model (Ensemble)",
        "",
        "| Metric | Mean | Std | Min | Max | Range |",
        "|--------|------|-----|-----|-----|-------|",
    ]
    for col in key_metrics:
        vals = results_df[col]
        label = col.replace("full_", "")
        report_lines.append(
            f"| {label} | {vals.mean():.4f} | {vals.std():.4f} | "
            f"{vals.min():.4f} | {vals.max():.4f} | {vals.max()-vals.min():.4f} |"
        )

    report_lines += [
        "",
        "## Per-Seed Results",
        "",
        "| Seed | LogLoss | Brier | >=Elite AUC | >=Stud AUC |",
        "|------|---------|-------|-------------|------------|",
    ]
    for _, row in results_df.iterrows():
        report_lines.append(
            f"| {int(row['seed'])} | {row['full_log_loss']:.4f} | {row['full_brier']:.4f} | "
            f"{row['full_>=Elite_auc']:.3f} | {row['full_>=Stud_auc']:.3f} |"
        )

    report_path = os.path.join(DATA_DIR, "reports", "seed_sensitivity_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved report to {report_path}")
