#!/usr/bin/env python3
"""
Interactive player classification tool.
Walks through each player in rookie_wr_2016_2024.csv and prompts for a tier label.
Saves progress after each entry so you can quit and resume later.

Tiers:
  1 = League-Winner
  2 = Elite
  3 = Starter
  4 = Flex
  5 = Stash
  6 = Bust

Usage:
  python classify_players.py
"""

import pandas as pd
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")
INPUT_PATH = os.path.join(DATA_DIR, "rookie_wr_2016_2024.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "rookie_wr_2016_2024_classified.csv")

TIERS = {
    "1": "League-Winner",
    "2": "Elite",
    "3": "Starter",
    "4": "Flex",
    "5": "Stash",
    "6": "Bust",
}

TIER_DISPLAY = "  ".join(f"[{k}] {v}" for k, v in TIERS.items())


def load_progress():
    """Load existing classifications if resuming."""
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        if "tier" in df.columns:
            classified = df[df["tier"].notna() & (df["tier"] != "")]
            return dict(zip(classified["name"], classified["tier"]))
    return {}


def main():
    rookies = pd.read_csv(INPUT_PATH)
    progress = load_progress()

    if progress:
        print(f"Resuming — {len(progress)} players already classified.\n")

    total = len(rookies)
    current_year = None

    for idx, row in rookies.iterrows():
        name = row["name"]

        if name in progress:
            continue

        if row["year"] != current_year:
            current_year = row["year"]
            print(f"\n{'='*50}")
            print(f"  {current_year} Draft Class")
            print(f"{'='*50}\n")

        remaining = total - len(progress)
        print(f"[{len(progress)+1}/{total}]  {name}  (Rd {row['round']}, Pick {row['pick']})")
        print(f"  {TIER_DISPLAY}")

        while True:
            choice = input("  > ").strip()

            if choice.lower() in ("q", "quit", "exit"):
                save(rookies, progress)
                print(f"\nSaved {len(progress)} classifications to {OUTPUT_PATH}")
                sys.exit(0)
            elif choice.lower() in ("s", "skip"):
                print("  Skipped.\n")
                break
            elif choice in TIERS:
                progress[name] = TIERS[choice]
                print(f"  -> {TIERS[choice]}\n")
                break
            else:
                print(f"  Invalid. Enter 1-6, 's' to skip, or 'q' to quit.")

    save(rookies, progress)
    print(f"\nDone! All {len(progress)} classifications saved to {OUTPUT_PATH}")


def save(rookies, progress):
    df = rookies.copy()
    df["tier"] = df["name"].map(progress).fillna("")
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
