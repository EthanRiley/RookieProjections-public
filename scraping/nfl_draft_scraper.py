#!/usr/bin/env python3
"""
NFL Draft Lookup
Uses nfl_data_py (nflverse) to fetch all players drafted in a given year
at a given position. No scraping, no 403s.

Usage:
    python nfl_draft_lookup.py <year> <position>

Examples:
    python nfl_draft_lookup.py 2016 WR
    python nfl_draft_lookup.py 2023 QB --csv qb_2023.csv
    python nfl_draft_lookup.py 2020 DB        # all defensive backs (CB/S/etc)
    python nfl_draft_lookup.py 2022 OL        # all offensive linemen
    python nfl_draft_lookup.py 2021 EDGE      # edge rushers (2019+ only)

Position arg can match either the granular `position` field (QB, RB, WR, TE,
CB, FS, SS, DE, DT, OT, OG, C, ILB, OLB, K, P, LS, ...) or the broader
`category` field (QB, RB, WR, TE, OL, DL, LB, DB, K, P, ED).

Requirements:
    pip install nfl_data_py pandas
"""

import argparse
import os
import sys

import nfl_data_py as nfl
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

# Position aliases -> (category_value, position_value)
# If a tuple entry is None, we don't filter on that field.
ALIASES = {
    # Offensive line umbrella
    "OL": ("OL", None),
    "OT": (None, "T"),
    "T": (None, "T"),
    "OG": (None, "G"),
    "G": (None, "G"),
    "C": (None, "C"),
    # Skill
    "QB": ("QB", None),
    "RB": ("RB", None),
    "HB": ("RB", None),
    "FB": (None, "FB"),
    "WR": ("WR", None),
    "TE": ("TE", None),
    # Defensive line / edge
    "DL": ("DL", None),
    "DE": (None, "DE"),
    "DT": (None, "DT"),
    "NT": (None, "NT"),
    "EDGE": ("ED", None),  # only populated from 2019+
    "ED": ("ED", None),
    # Linebacker
    "LB": ("LB", None),
    "ILB": (None, "ILB"),
    "OLB": (None, "OLB"),
    "MLB": (None, "ILB"),
    # Secondary
    "DB": ("DB", None),
    "CB": (None, "CB"),
    "S": (None, "S"),
    "FS": (None, "FS"),
    "SS": (None, "SS"),
    "SAF": (None, "S"),
    # Special teams
    "K": ("K", None),
    "PK": ("K", None),
    "P": ("P", None),
    "LS": ("LS", None),
}


def resolve_position_filter(pos_arg: str):
    pos = pos_arg.upper().strip()
    if pos not in ALIASES:
        valid = ", ".join(sorted(ALIASES.keys()))
        raise ValueError(
            f"Unknown position '{pos_arg}'.\nValid options: {valid}"
        )
    return ALIASES[pos]


def filter_df(df: pd.DataFrame, category, position) -> pd.DataFrame:
    """
    Apply category/position filter. Matches if EITHER field matches the
    corresponding value (so ambiguous codes like 'S' still work for years
    where PFR labeled it FS/SS).
    """
    masks = []
    if category is not None and "category" in df.columns:
        masks.append(df["category"].astype(str).str.upper() == category.upper())
    if position is not None and "position" in df.columns:
        pos_upper = df["position"].astype(str).str.upper()
        # Handle the 'S' family: if user asked for S, catch S/FS/SS
        if position.upper() == "S":
            masks.append(pos_upper.isin(["S", "FS", "SS"]))
        else:
            masks.append(pos_upper == position.upper())

    if not masks:
        return df.iloc[0:0]  # empty

    # Combine with OR so either matching category or matching position keeps the row
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return df[combined].reset_index(drop=True)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "season", "round", "pick", "team", "pfr_player_name", "position",
        "category", "age", "college",
        "games", "seasons_started",
        "pass_yards", "pass_tds",
        "rush_yards", "rush_tds",
        "rec", "rec_yards", "rec_tds",
        "def_solo_tackles", "def_ints", "def_sacks",
        "allpro", "probowls", "car_av",
    ]
    cols = [c for c in preferred if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def main():
    years = list(range(2016, 2024))
    position_arg = "WR"
    csv_path = os.path.join(DATA_DIR, "rookie_wr_2016_2024.csv")

    category, position = resolve_position_filter(position_arg)

    print(f"Fetching {years[0]}-{years[-1]} NFL draft picks from nflverse...", file=sys.stderr)
    try:
        df = nfl.import_draft_picks(years)
    except Exception as e:
        print(f"Failed to load draft data: {e}", file=sys.stderr)
        sys.exit(1)

    filtered = filter_df(df, category, position)

    if filtered.empty:
        print("No WR players found.", file=sys.stderr)
        sys.exit(0)

    if "pick" in filtered.columns:
        filtered = filtered.sort_values(["season", "pick"]).reset_index(drop=True)

    names = filtered[["season", "pfr_player_name", "round", "pick"]].rename(
        columns={"season": "year", "pfr_player_name": "name"}
    )

    print(f"\n{len(names)} WR(s) drafted from {years[0]}-{years[-1]}:\n")
    print(names.to_string(index=False))

    names.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()