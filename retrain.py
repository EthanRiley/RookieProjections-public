#!/usr/bin/env python3
"""
Full retrain pipeline.

Steps:
  1. Regenerate master dataset (aggregate_college_stats.py)
  2. Evaluate holdout (train 2018-2021, test 2022-2024) → holdout_predictions_v2.csv
  3. Predict prospects (retrain on all labeled data, predict 2024-2026) → prospect_predictions_{year}.csv
  4. Regenerate prospect/holdout profiles (batch mode)

Usage:
  python retrain.py                    # Full pipeline
  python retrain.py --step aggregate   # Just step 1
  python retrain.py --step holdout     # Just step 2
  python retrain.py --step prospects   # Just step 3
  python retrain.py --step profiles    # Just step 4
  python retrain.py --skip-profiles    # Steps 1-3 only (faster)
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.0f}s (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n  Done in {elapsed:.0f}s")
    return result


def step_aggregate():
    run("python3 -m aggregation.aggregate_college_stats",
        "Step 1: Regenerate master dataset (wr_dynasty_value_with_college.csv)")


def step_holdout():
    run("python3 -m modeling.evaluate_holdout",
        "Step 2: Evaluate holdout (train 2018-2021, test 2022-2024)")


def step_prospects():
    run("python3 -m modeling.predict_prospects",
        "Step 3: Predict prospects (retrain on all data, predict 2024-2026)")


def step_profiles():
    # Holdout profiles
    for year in [2022, 2023, 2024]:
        run(f"python3 -m viz.prospect_profile --batch --year {year} --top 15",
            f"Step 4a: Generate {year} holdout profiles")

    # Prospect profiles
    for year in [2025, 2026]:
        run(f"python3 -m viz.prospect_profile --batch --year {year} --top 15",
            f"Step 4b: Generate {year} prospect profiles")


def main():
    parser = argparse.ArgumentParser(description="Full retrain pipeline")
    parser.add_argument("--step", choices=["aggregate", "holdout", "prospects", "profiles"],
                        help="Run only a specific step")
    parser.add_argument("--skip-profiles", action="store_true",
                        help="Skip profile generation (steps 1-3 only)")
    args = parser.parse_args()

    start = time.time()

    if args.step:
        {"aggregate": step_aggregate, "holdout": step_holdout,
         "prospects": step_prospects, "profiles": step_profiles}[args.step]()
    else:
        step_aggregate()
        step_holdout()
        step_prospects()
        if not args.skip_profiles:
            step_profiles()

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
