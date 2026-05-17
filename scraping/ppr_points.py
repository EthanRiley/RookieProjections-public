# yearly_ppr_totals.py

import os
import nflreadpy as nfl
import polars as pl

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

START_YEAR = 2016
END_YEAR = 2025  # last completed NFL season as of 2026

years = list(range(START_YEAR, END_YEAR + 1))

# "reg" = regular-season yearly totals
df = nfl.load_player_stats(years, summary_level="reg")

# Keep useful fantasy columns if they exist
wanted_cols = [
    "season",
    "player_id",
    "player_display_name",
    "player_name",
    "position",
    "position_group",
    "recent_team",

    # fantasy scoring
    "fantasy_points",
    "fantasy_points_ppr",

    # passing
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",

    # rushing
    "carries",
    "rushing_yards",
    "rushing_tds",

    # receiving
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",

    # misc
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
]

existing_cols = [c for c in wanted_cols if c in df.columns]

yearly = (
    df.select(existing_cols)
      .sort(["season", "fantasy_points_ppr"], descending=[False, True])
)

output_path = os.path.join(DATA_DIR, "nfl_yearly_ppr_totals_2016_2025.csv")
yearly.write_csv(output_path)

print(yearly.head(25))
print(f"Saved to {output_path}")