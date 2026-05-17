#!/usr/bin/env python3
"""
Aggregate college receiving grades onto the dynasty value file.

For each player, computes:
  - Career stats: game-weighted averages for rate stats, per-game totals for counting stats
  - Best-2-seasons stats: same aggregation on the top 2 seasons by grades_offense
  - Best season stats: season with highest grades_offense (same normalization)

All counting stats are per-game to account for varying game counts.

Reads:
  - wr_data/wr_dynasty_value.csv
  - wr_data/{2016..2025}_receiving_grades.csv
Outputs:
  - wr_data/wr_dynasty_value_with_college.csv
"""

import os
import re

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

# --- Name normalization ---
SUFFIXES_RE = re.compile(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)

def normalize_name(name):
    n = SUFFIXES_RE.sub('', str(name)).strip()
    n = n.replace('.', '').replace("'", '').lower()
    return ' '.join(n.split())


# Stats that are already rates/grades — aggregate via weighted average
# Game-weighted: PFF grades and route_rate (snap-level metrics where games is reasonable)
GAME_WEIGHTED_COLS = [
    "grades_hands_drop",
    "grades_hands_fumble",
    "grades_offense",
    "grades_pass_route",
    "route_rate",
]

# Target-weighted: stats that are per-target by nature
TARGET_WEIGHTED_COLS = [
    "avg_depth_of_target",
    "caught_percent",
    "drop_rate",
    "targeted_qb_rating",
]

# Reception-weighted: stats that are per-reception by nature
RECEPTION_WEIGHTED_COLS = [
    "yards_after_catch_per_reception",
    "yards_per_reception",
]

# Contested-target-weighted
CONTESTED_TARGET_WEIGHTED_COLS = [
    "contested_catch_rate",
]

# Snap-weighted: alignment rates
SNAP_WEIGHTED_COLS = [
    "slot_rate",
    "wide_rate",
    "inline_rate",
]

# Combined list for backwards compat (used in other checks)
RATE_COLS = (GAME_WEIGHTED_COLS + TARGET_WEIGHTED_COLS + RECEPTION_WEIGHTED_COLS
             + CONTESTED_TARGET_WEIGHTED_COLS + SNAP_WEIGHTED_COLS)

# Counting stats — aggregate as per-game
COUNTING_COLS = [
    "avoided_tackles",
    "contested_receptions",
    "contested_targets",
    "drops",
    "first_downs",
    "fumbles",
    "receptions",
    "routes",
    "targets",
    "touchdowns",
    "yards",
    "yards_after_catch",
]

# Seasons to exclude: (normalized_name, team_name, grade_year)
SEASON_EXCLUSIONS = [
    ("elijah sarratt", "JAMES MAD", 2023),
    ("kyle williams", "UNLV", 2020),  # Different Kyle Williams (not 2025 prospect)
    ("kyle williams", "UNLV", 2021),
    ("kyle williams", "UNLV", 2022),
]


def _should_exclude(row):
    """Check if a season row matches any exclusion rule."""
    key = row.get("_join_key", "")
    team = row.get("team_name", "")
    year = row.get("grade_year", 0)
    return any(key == n and team == t and year == y for n, t, y in SEASON_EXCLUSIONS)


def fit_adot_regression(all_grades):
    """Fit a linear regression of catch% on aDOT across all player-seasons.

    Returns (slope, intercept) coefficients: catch% = slope * aDOT + intercept
    """
    cp = pd.to_numeric(all_grades["caught_percent"], errors="coerce")
    adot = pd.to_numeric(all_grades["avg_depth_of_target"], errors="coerce")
    mask = cp.notna() & adot.notna()
    coef = np.polyfit(adot[mask].values, cp[mask].values, 1)
    return coef


def aggregate_seasons(seasons: pd.DataFrame, prefix: str = "career",
                      adot_coef=None) -> dict:
    """Compute game-weighted averages and per-game counting stats for a set of seasons.

    Args:
        seasons: DataFrame of season rows (must have player_game_count column)
        prefix: column name prefix (e.g. "career", "best2")
        adot_coef: optional (slope, intercept) from fit_adot_regression for aDOT-adjusted catch%
    """
    games = seasons["player_game_count"].values
    total_games = games.sum()
    if total_games == 0:
        return {}

    result = {f"{prefix}_games": int(total_games), f"{prefix}_seasons": len(seasons)}

    # Helper: weighted average with fallback to game-weighted
    def _wavg(col, weight_col):
        vals = pd.to_numeric(seasons[col], errors="coerce") if col in seasons.columns else None
        if vals is None:
            return
        wts_raw = pd.to_numeric(seasons[weight_col], errors="coerce") if weight_col in seasons.columns else None
        mask = vals.notna()
        if wts_raw is not None:
            mask = mask & wts_raw.notna()
            wts = wts_raw[mask].values
            if wts.sum() > 0:
                result[f"{prefix}_{col}"] = round(np.average(vals[mask], weights=wts), 2)
                return
        # Fallback to game-weighted if weight column missing or all zeros
        if mask.any():
            result[f"{prefix}_{col}"] = round(np.average(vals[mask], weights=games[mask]), 2)

    # Game-weighted: PFF grades and route_rate
    for col in GAME_WEIGHTED_COLS:
        _wavg(col, "player_game_count")

    # Target-weighted: targeted_qb_rating, caught_percent, drop_rate, avg_depth_of_target
    for col in TARGET_WEIGHTED_COLS:
        _wavg(col, "targets")

    # Reception-weighted: yards_per_reception, yards_after_catch_per_reception
    for col in RECEPTION_WEIGHTED_COLS:
        _wavg(col, "receptions")

    # Contested-target-weighted: contested_catch_rate
    for col in CONTESTED_TARGET_WEIGHTED_COLS:
        _wavg(col, "contested_targets")

    # Snap-weighted: slot_rate, wide_rate, inline_rate
    # Use total alignment snaps (slot + wide + inline) as weight
    if all(c in seasons.columns for c in ["slot_snaps", "wide_snaps", "inline_snaps"]):
        total_align = (pd.to_numeric(seasons["slot_snaps"], errors="coerce").fillna(0)
                       + pd.to_numeric(seasons["wide_snaps"], errors="coerce").fillna(0)
                       + pd.to_numeric(seasons["inline_snaps"], errors="coerce").fillna(0))
        for col in SNAP_WEIGHTED_COLS:
            if col in seasons.columns:
                vals = pd.to_numeric(seasons[col], errors="coerce")
                mask = vals.notna() & (total_align > 0)
                if mask.any():
                    result[f"{prefix}_{col}"] = round(
                        np.average(vals[mask], weights=total_align[mask]), 2)
    else:
        for col in SNAP_WEIGHTED_COLS:
            _wavg(col, "player_game_count")

    # Per-game for counting stats
    for col in COUNTING_COLS:
        if col in seasons.columns:
            vals = pd.to_numeric(seasons[col], errors="coerce").fillna(0)
            result[f"{prefix}_{col}_pg"] = round(vals.sum() / total_games, 2)

    # Route-based rates: compute total routes once
    if "routes" in seasons.columns:
        total_routes = pd.to_numeric(seasons["routes"], errors="coerce").fillna(0).sum()
        if total_routes > 0:
            if "yards" in seasons.columns:
                total_yards = pd.to_numeric(seasons["yards"], errors="coerce").fillna(0).sum()
                result[f"{prefix}_yprr"] = round(total_yards / total_routes, 2)
            if "first_downs" in seasons.columns:
                total_fd = pd.to_numeric(seasons["first_downs"], errors="coerce").fillna(0).sum()
                result[f"{prefix}_first_downs_per_route"] = round(total_fd / total_routes, 4)

    # Avoided tackles per reception (not per game)
    if "avoided_tackles" in seasons.columns and "receptions" in seasons.columns:
        total_at = pd.to_numeric(seasons["avoided_tackles"], errors="coerce").fillna(0).sum()
        total_rec = pd.to_numeric(seasons["receptions"], errors="coerce").fillna(0).sum()
        if total_rec > 0:
            result[f"{prefix}_avoided_tackles_per_rec"] = round(total_at / total_rec, 2)

    # aDOT-adjusted catch%: residual of catch% after regressing on avg_depth_of_target
    # Weight by targets (not games) since catch% is a per-target metric
    if adot_coef is not None and "caught_percent" in seasons.columns and "avg_depth_of_target" in seasons.columns:
        cp_vals = pd.to_numeric(seasons["caught_percent"], errors="coerce")
        adot_vals = pd.to_numeric(seasons["avg_depth_of_target"], errors="coerce")
        tgt_vals = pd.to_numeric(seasons["targets"], errors="coerce") if "targets" in seasons.columns else None
        valid = cp_vals.notna() & adot_vals.notna()
        if tgt_vals is not None:
            valid = valid & tgt_vals.notna() & (tgt_vals > 0)
        if valid.any():
            residuals = cp_vals[valid] - np.polyval(adot_coef, adot_vals[valid])
            wts = tgt_vals[valid].values if tgt_vals is not None else games[valid]
            if wts.sum() > 0:
                result[f"{prefix}_catch_pct_adot_adj"] = round(
                    np.average(residuals, weights=wts), 2
                )
            else:
                result[f"{prefix}_catch_pct_adot_adj"] = round(
                    np.average(residuals, weights=games[valid]), 2
                )

    return result


CCR_MIN_CONTESTED_TARGETS = 10
CCR_GROUP_AVG = 45.0
CCR_PENALTY_PP = 5.0

P5_TEAMS = {
    # SEC
    'ALABAMA', 'ARKANSAS', 'AUBURN', 'FLORIDA', 'GEORGIA', 'KENTUCKY', 'LSU',
    'MISS STATE', 'MISSOURI', 'OLE MISS', 'S CAROLINA', 'TENNESSEE', 'TEXAS',
    'TEXAS A&M', 'OKLAHOMA', 'VANDERBILT',
    # Big Ten
    'ILLINOIS', 'INDIANA', 'IOWA', 'MARYLAND', 'MICHIGAN', 'MICH STATE',
    'MINNESOTA', 'NEBRASKA', 'NWESTERN', 'OHIO STATE', 'PENN STATE', 'PURDUE',
    'RUTGERS', 'WISCONSIN', 'UCLA', 'USC', 'OREGON', 'WASHINGTON',
    # Big 12
    'ARIZONA', 'ARIZONA ST', 'BAYLOR', 'BYU', 'CINCINNATI', 'COLORADO',
    'HOUSTON', 'IOWA STATE', 'KANSAS', 'KANSAS ST', 'OKLA STATE', 'TCU',
    'TEXAS TECH', 'UCF', 'W VIRGINIA', 'UTAH',
    # ACC
    'BOSTON COL', 'CLEMSON', 'DUKE', 'FLORIDA ST', 'GA TECH', 'LOUISVILLE',
    'MIAMI FL', 'N CAROLINA', 'NC STATE', 'PITTSBURGH', 'SYRACUSE', 'VA TECH',
    'VIRGINIA', 'WAKE', 'SMU', 'CAL', 'STANFORD',
    # Independent P5
    'NOTRE DAME',
    # Former P5 (pre-realignment, count historical seasons)
    'OREGON ST', 'WASH STATE',
}


SENIOR_AGE_THRESHOLD = 21.5
SENIOR_DISCOUNT_PP = 10.0
SENIOR_DISCOUNT_COLS = ["contested_catch_rate", "caught_percent", "targeted_qb_rating"]

# Graduated age adjustment: per-age-class multiplicative adjustment for YPRR and catch%
GRADUATED_ADJ = {
    # (age_lower, age_upper): multiplier
    (0, 19.5): 1.25,      # freshman +25%
    (19.5, 20.5): 1.05,   # sophomore +5%
    (20.5, 21.5): 0.80,   # junior -20%
    (21.5, 22.5): 0.75,   # senior -25%
    (22.5, 99): 0.50,     # super senior -50%
}
GRADUATED_YPRR_ADJ = GRADUATED_ADJ  # backwards compat alias

# Peak-gated selection: minimum grades_offense to be eligible for stat-specific peak selection
PEAK_GATED_QUALITY_GATE = 80.0


def _apply_senior_discount(seasons: pd.DataFrame, birthdate) -> pd.DataFrame:
    """Discount percentage-based rate stats by SENIOR_DISCOUNT_PP for seasons
    where the player is >= SENIOR_AGE_THRESHOLD on Sept 1.

    Returns a copy with discounted values. Non-percentage stats (YPRR, avoided
    tackles per rec) are not affected.
    """
    if birthdate is None or pd.isna(birthdate):
        return seasons
    if "grade_year" not in seasons.columns:
        return seasons

    seasons = seasons.copy()
    for _, row in seasons.iterrows():
        yr = row.get("grade_year")
        if pd.isna(yr):
            continue
        sept1 = pd.Timestamp(f"{int(yr)}-09-01")
        age = (sept1 - birthdate).days / 365.25
        if age >= SENIOR_AGE_THRESHOLD:
            for col in SENIOR_DISCOUNT_COLS:
                if col in seasons.columns:
                    val = pd.to_numeric(row.get(col), errors="coerce")
                    if pd.notna(val):
                        seasons.at[row.name, col] = val - SENIOR_DISCOUNT_PP

    return seasons


def compute_best1_yprr_graduated(seasons: pd.DataFrame, birthdate) -> float:
    """Best single season YPRR with graduated age adjustment.

    Selects the best season by grades_offense among P5 seasons with 200+ routes,
    applies a per-age-class multiplicative adjustment to YPRR, and returns the
    adjusted value.
    """
    routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)),
                           errors="coerce").fillna(0)
    eligible = seasons[routes >= 200]

    if "team_name" in eligible.columns:
        p5 = eligible[eligible["team_name"].isin(P5_TEAMS)]
        if len(p5) >= 1:
            eligible = p5

    if len(eligible) == 0:
        return np.nan

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().any():
        best = eligible.loc[grades.idxmax()]
    else:
        best = eligible.iloc[0]

    yards = pd.to_numeric(best.get("yards", 0), errors="coerce") or 0
    rts = pd.to_numeric(best.get("routes", 0), errors="coerce") or 0
    if rts == 0:
        return np.nan
    yprr = yards / rts

    # Apply graduated age adjustment
    if birthdate is not None and pd.notna(birthdate):
        yr = best.get("grade_year")
        if pd.notna(yr):
            sept1 = pd.Timestamp(f"{int(yr)}-09-01")
            age = (sept1 - birthdate).days / 365.25
            for (lo, hi), mult in GRADUATED_YPRR_ADJ.items():
                if lo <= age < hi:
                    yprr *= mult
                    break

    return round(yprr, 4)


def compute_peak_yprr_graduated(seasons: pd.DataFrame, birthdate) -> float:
    """Peak age-adjusted YPRR: compute adjusted YPRR for all eligible seasons, take max."""
    routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)),
                           errors="coerce").fillna(0)
    eligible = seasons[routes >= 200]

    if "team_name" in eligible.columns:
        p5 = eligible[eligible["team_name"].isin(P5_TEAMS)]
        if len(p5) >= 1:
            eligible = p5

    if len(eligible) == 0:
        return np.nan

    best_adj_yprr = np.nan
    for _, row in eligible.iterrows():
        yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
        rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
        if rts == 0:
            continue
        yprr = yards / rts

        if birthdate is not None and pd.notna(birthdate):
            yr = row.get("grade_year")
            if pd.notna(yr):
                sept1 = pd.Timestamp(f"{int(yr)}-09-01")
                age = (sept1 - birthdate).days / 365.25
                for (lo, hi), mult in GRADUATED_YPRR_ADJ.items():
                    if lo <= age < hi:
                        yprr *= mult
                        break

        if np.isnan(best_adj_yprr) or yprr > best_adj_yprr:
            best_adj_yprr = yprr

    return round(best_adj_yprr, 4) if not np.isnan(best_adj_yprr) else np.nan


def compute_pg_yprr_graduated(seasons: pd.DataFrame, birthdate) -> float:
    """Peak-gated YPRR graduated: max YPRR from seasons with grade >= 80, with age adjustment.

    Picks the season where YPRR is highest among quality-gated seasons
    (grades_offense >= PEAK_GATED_QUALITY_GATE). Falls back to best1 if
    no season meets the gate.
    """
    routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)),
                           errors="coerce").fillna(0)
    eligible = seasons[routes >= 200]

    if "team_name" in eligible.columns:
        p5 = eligible[eligible["team_name"].isin(P5_TEAMS)]
        if len(p5) >= 1:
            eligible = p5

    if len(eligible) == 0:
        return np.nan

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if not grades.notna().any():
        return np.nan

    # Quality-gated seasons
    gated = eligible[grades >= PEAK_GATED_QUALITY_GATE]

    def _age_adjust_yprr(raw_yprr, yr):
        """Apply graduated age adjustment to a YPRR value."""
        if birthdate is None or not pd.notna(birthdate) or yr is None or not pd.notna(yr):
            return raw_yprr
        sept1 = pd.Timestamp(f"{int(yr)}-09-01")
        age = (sept1 - birthdate).days / 365.25
        for (lo, hi), mult in GRADUATED_ADJ.items():
            if lo <= age < hi:
                return raw_yprr * mult
        return raw_yprr

    if len(gated) > 0:
        # Find season with max age-adjusted YPRR among gated seasons
        best_val = np.nan
        for _, row in gated.iterrows():
            yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
            rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
            if rts == 0:
                continue
            yprr = yards / rts
            adj_yprr = _age_adjust_yprr(yprr, row.get("grade_year"))
            if np.isnan(best_val) or adj_yprr > best_val:
                best_val = adj_yprr
    else:
        # Fall back to best1 (highest grade season)
        best_row = eligible.loc[grades.idxmax()]
        yards = pd.to_numeric(best_row.get("yards", 0), errors="coerce") or 0
        rts = pd.to_numeric(best_row.get("routes", 0), errors="coerce") or 0
        if rts == 0:
            return np.nan
        best_val = _age_adjust_yprr(yards / rts, best_row.get("grade_year"))

    if np.isnan(best_val):
        return np.nan

    return round(best_val, 4)


def compute_pg_catch_pct_adot_adj_graduated(seasons: pd.DataFrame, birthdate,
                                             adot_coef=None) -> float:
    """Peak-gated catch% aDOT-adjusted graduated: max aDOT-adjusted catch% from
    quality-gated seasons, with graduated age adjustment.

    Picks the season where catch_pct_adot_adj is highest among seasons with
    grades_offense >= PEAK_GATED_QUALITY_GATE. Falls back to best1 if no
    season meets the gate.
    """
    if adot_coef is None:
        return np.nan

    routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)),
                           errors="coerce").fillna(0)
    eligible = seasons[routes >= 200]

    if "team_name" in eligible.columns:
        p5 = eligible[eligible["team_name"].isin(P5_TEAMS)]
        if len(p5) >= 1:
            eligible = p5

    if len(eligible) == 0:
        return np.nan

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if not grades.notna().any():
        return np.nan

    # Compute aDOT-adjusted catch% per season
    cp = pd.to_numeric(eligible["caught_percent"], errors="coerce")
    adot = pd.to_numeric(eligible["avg_depth_of_target"], errors="coerce")
    cpaa = cp - np.polyval(adot_coef, adot)

    def _age_adjust_cpaa(raw_cpaa, yr):
        """Apply graduated age adjustment to an aDOT-adjusted catch% value."""
        if birthdate is None or not pd.notna(birthdate) or yr is None or not pd.notna(yr):
            return raw_cpaa
        sept1 = pd.Timestamp(f"{int(yr)}-09-01")
        age = (sept1 - birthdate).days / 365.25
        for (lo, hi), mult in GRADUATED_ADJ.items():
            if lo <= age < hi:
                return raw_cpaa * mult
        return raw_cpaa

    # Quality-gated seasons
    gated_mask = grades >= PEAK_GATED_QUALITY_GATE
    gated_cpaa = cpaa[gated_mask].dropna()

    if len(gated_cpaa) > 0:
        # Select season with max age-adjusted cpaa among gated seasons
        best_val = np.nan
        for idx in gated_cpaa.index:
            yr = eligible.loc[idx].get("grade_year")
            adj = _age_adjust_cpaa(float(gated_cpaa.loc[idx]), yr)
            if np.isnan(best_val) or adj > best_val:
                best_val = adj
    else:
        # Fall back to best1 (highest grade season)
        best1_idx = grades.idxmax()
        raw = cpaa.loc[best1_idx]
        if pd.isna(raw):
            return np.nan
        best_val = _age_adjust_cpaa(float(raw), eligible.loc[best1_idx].get("grade_year"))

    if np.isnan(best_val):
        return np.nan

    return round(best_val, 4)


def best2_stats(seasons: pd.DataFrame, adot_coef=None, birthdate=None) -> dict:
    """Compute stats using only the best 2 seasons by grades_offense.

    Only seasons with 200+ routes at P5 schools are eligible, to avoid
    small-sample noise and inflated stats from weaker competition. Falls
    back to all 200+ route seasons (any school) if fewer than 2 P5 seasons.

    Senior season discount: percentage-based rate stats are reduced by
    SENIOR_DISCOUNT_PP for seasons where the player is >= 22 on Sept 1.

    Small-sample filter for contested catch rate: if total contested targets
    across the selected seasons is below CCR_MIN_CONTESTED_TARGETS, replace
    with group average minus CCR_PENALTY_PP percentage points.
    """
    routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    eligible = seasons[routes >= 200]

    # Prefer P5 seasons; fall back to all eligible if fewer than 2 P5
    if "team_name" in eligible.columns:
        p5_eligible = eligible[eligible["team_name"].isin(P5_TEAMS)]
        if len(p5_eligible) >= 2:
            eligible = p5_eligible
        # If only 0-1 P5 seasons, use all eligible (don't penalize players
        # who only played at non-P5 schools)

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce") if len(eligible) > 0 else pd.Series(dtype=float)
    if grades.notna().sum() < 2:
        # Fall back to all seasons with 200+ routes (may be 0 or 1)
        if len(eligible) > 0:
            selected = eligible
        else:
            selected = seasons
    else:
        top2_idx = grades.nlargest(2).index
        selected = eligible.loc[top2_idx]

    # Apply senior season discount before aggregation
    selected = _apply_senior_discount(selected, birthdate)

    result = aggregate_seasons(selected, prefix="best2", adot_coef=adot_coef)

    # Small-sample CCR filter: replace with penalized group average if too few contested targets.
    # Only applies when CCR data exists but sample is small; players with no CCR data stay NaN.
    if "contested_targets" in selected.columns and "contested_catch_rate" in selected.columns:
        ct = pd.to_numeric(selected["contested_targets"], errors="coerce").fillna(0)
        ccr = pd.to_numeric(selected["contested_catch_rate"], errors="coerce")
        total_ct = ct.sum()
        has_ccr_data = ccr.notna().any()
        if has_ccr_data and total_ct < CCR_MIN_CONTESTED_TARGETS:
            result["best2_contested_catch_rate"] = round(CCR_GROUP_AVG - CCR_PENALTY_PP, 2)

    return result


def best_season_stats(seasons: pd.DataFrame) -> dict:
    """Pick season with highest grades_offense, return per-game counting stats and rates."""
    grades = pd.to_numeric(seasons["grades_offense"], errors="coerce")
    if grades.isna().all():
        return {}

    best = seasons.loc[grades.idxmax()]
    games = best["player_game_count"]
    if pd.isna(games) or games == 0:
        return {}

    result = {"best_season_games": int(games)}

    for col in RATE_COLS:
        if col in seasons.columns:
            val = pd.to_numeric(best.get(col), errors="coerce")
            if pd.notna(val):
                result[f"best_{col}"] = round(val, 2)

    for col in COUNTING_COLS:
        if col in seasons.columns:
            val = pd.to_numeric(best.get(col, 0), errors="coerce")
            if pd.isna(val):
                val = 0
            result[f"best_{col}_pg"] = round(val / games, 2)

    return result


def load_all_grades(year_range=range(2016, 2027)):
    """Load and concatenate all receiving grades files."""
    all_grades = []
    for yr in year_range:
        path = os.path.join(DATA_DIR, "grades", f"{yr}_receiving_grades.csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
            d["grade_year"] = yr
            all_grades.append(d)
    if not all_grades:
        return pd.DataFrame()
    result = pd.concat(all_grades, ignore_index=True)
    result["_join_key"] = result["player"].apply(normalize_name)
    return result


def get_player_seasons(all_grades, name, draft_year, apply_exclusions=True,
                       birthdate=None):
    """Get a player's college seasons, optionally applying exclusion rules.

    Handles name collisions using birthdate (player must be >= 17 on Sept 1
    of the season) and a 6-season cap on career length.
    """
    key = normalize_name(name)
    seasons = all_grades[
        (all_grades["_join_key"] == key) & (all_grades["grade_year"] <= draft_year)
    ]
    if apply_exclusions and len(seasons) > 0:
        mask = ~seasons.apply(_should_exclude, axis=1)
        seasons = seasons[mask]

    # Handle name collisions using two filters:
    # 1. Player must be >= 18 on Sept 1 of the season
    # 2. Season must be within 5 years of draft (covers redshirt + COVID)
    if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
        min_year = birthdate.year + 18 if hasattr(birthdate, 'year') else None
        if min_year is not None:
            sept1_min = pd.Timestamp(f"{min_year}-09-01")
            if sept1_min < birthdate + pd.DateOffset(years=18):
                min_year += 1
            seasons = seasons[seasons["grade_year"] >= min_year]

    if len(seasons) > 0:
        earliest = draft_year - 5
        seasons = seasons[seasons["grade_year"] >= earliest]

    return seasons


def compute_breakout_age(seasons, birthdate, team_att_lookup=None, team_games_lookup=None):
    """Compute YPRR-based breakout age and magnitudes at breakout.

    Breakout = first season with 2.0+ YPRR and 200+ routes.
    Returns (breakout_age, breakout_yptpa, breakout_yprr).
    breakout_yptpa = YPTPA value at the breakout season.
    breakout_yprr = YPRR value at the breakout season.
    """
    result = (np.nan, np.nan, np.nan)
    if len(seasons) == 0 or pd.isna(birthdate):
        return result

    for _, s in seasons.sort_values("grade_year").iterrows():
        yr = s["grade_year"]
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        team = s.get("team_name", "")

        if routes >= 200:
            yprr_val = yards / routes
            if yprr_val >= 2.0:
                sept1 = pd.Timestamp(f"{yr}-09-01")
                age = round((sept1 - birthdate).days / 365.25, 2)

                # Compute YPTPA at breakout
                yptpa_val = np.nan
                if team_att_lookup and team_games_lookup:
                    att = team_att_lookup.get((team, yr))
                    tg = team_games_lookup.get((team, yr))
                    if att and att > 0 and games > 0 and tg and tg > 0:
                        yptpa_val = round((yards / games) / (att / tg), 4)

                return (age, yptpa_val, round(yprr_val, 4))

    return result


def compute_peak_stats(seasons: pd.DataFrame) -> dict:
    """Compute true per-stat peaks (not tied to best offensive grade season).

    - peak_contested_catch_rate: highest single-season contested catch rate
      (min 3 contested targets). If career game-weighted rate is higher
      (due to low-volume seasons not meeting the threshold), use career as floor.
    - peak2_avoided_tackles_per_rec: avoided tackles / receptions averaged
      across the 2 seasons with the highest per-rec rate (min 10 receptions)
    """
    result = {}

    # Peak contested catch rate (min 3 contested targets, floored by career rate)
    if "contested_catch_rate" in seasons.columns and "contested_targets" in seasons.columns:
        ccr = pd.to_numeric(seasons["contested_catch_rate"], errors="coerce")
        ct = pd.to_numeric(seasons["contested_targets"], errors="coerce")
        valid = ccr.notna() & ct.notna() & (ct >= 3)
        if valid.any():
            peak_val = float(ccr[valid].max())
            # Floor by career game-weighted average (career may exceed peak when
            # low-volume seasons with high rates don't meet the min 3 filter)
            games = pd.to_numeric(seasons["player_game_count"], errors="coerce").fillna(1)
            all_valid = ccr.notna()
            if all_valid.any():
                career_val = float(np.average(ccr[all_valid], weights=games[all_valid]))
                peak_val = max(peak_val, career_val)
            result["peak_contested_catch_rate"] = round(peak_val, 2)

    # Peak2 avoided tackles per reception
    if "avoided_tackles" in seasons.columns and "receptions" in seasons.columns:
        at = pd.to_numeric(seasons["avoided_tackles"], errors="coerce").fillna(0)
        rec = pd.to_numeric(seasons["receptions"], errors="coerce").fillna(0)
        valid = rec >= 10
        if valid.sum() >= 2:
            rates = (at[valid] / rec[valid])
            top2 = rates.nlargest(2)
            # Weighted by receptions for the average
            top2_at = at[valid].loc[top2.index].sum()
            top2_rec = rec[valid].loc[top2.index].sum()
            result["peak2_avoided_tackles_per_rec"] = round(float(top2_at / top2_rec), 4)
        elif valid.sum() == 1:
            idx = valid[valid].index
            result["peak2_avoided_tackles_per_rec"] = round(float(at.loc[idx].sum() / rec.loc[idx].sum()), 4)

    # Peak YPRR (single best season, min 100 routes)
    if "yards" in seasons.columns and "routes" in seasons.columns:
        yards = pd.to_numeric(seasons["yards"], errors="coerce").fillna(0)
        routes = pd.to_numeric(seasons["routes"], errors="coerce").fillna(0)
        valid = routes >= 100
        if valid.any():
            yprr_per_season = yards[valid] / routes[valid]
            result["peak_yprr"] = round(float(yprr_per_season.max()), 4)

    return result


def compute_best_yptpa(seasons, team_att_lookup):
    """Compute best single-season yards per team pass attempt."""
    best_yptpa = np.nan
    for _, s in seasons.iterrows():
        yards = pd.to_numeric(s.get("yards"), errors="coerce")
        team = s.get("team_name")
        yr = s.get("grade_year")
        att = team_att_lookup.get((team, yr))
        if pd.notna(yards) and att and att > 0:
            yptpa = yards / att
            if pd.isna(best_yptpa) or yptpa > best_yptpa:
                best_yptpa = yptpa
    return round(best_yptpa, 4) if pd.notna(best_yptpa) else np.nan


def aggregate_player(all_grades, name, draft_year, birth_lookup=None,
                     team_att_lookup=None, draft_age_lookup=None,
                     adot_coef=None, team_games_lookup=None,
                     apply_exclusions=True):
    """Full aggregation for a single player. Returns a dict of all computed features."""
    birthdate = birth_lookup.get((name, draft_year)) if birth_lookup else None
    seasons = get_player_seasons(all_grades, name, draft_year, apply_exclusions,
                                 birthdate=birthdate)

    if len(seasons) == 0:
        return {}

    result = {}
    result.update(aggregate_seasons(seasons, prefix="career", adot_coef=adot_coef))
    result.update(best2_stats(seasons, adot_coef=adot_coef, birthdate=birthdate))
    result.update(best_season_stats(seasons))

    # Graduated age-adjusted YPRR (best single season)
    result["best1_yprr_graduated"] = compute_best1_yprr_graduated(seasons, birthdate)
    result["peak_yprr_graduated"] = compute_peak_yprr_graduated(seasons, birthdate)

    # Peak-gated features (v12)
    result["pg_yprr_graduated"] = compute_pg_yprr_graduated(seasons, birthdate)
    result["pg_catch_pct_adot_adj_graduated"] = compute_pg_catch_pct_adot_adj_graduated(
        seasons, birthdate, adot_coef=adot_coef
    )

    # Draft age
    if draft_age_lookup:
        da = draft_age_lookup.get((name, draft_year))
        if da is not None:
            result["draft_age"] = da

    # Breakout age (YPRR-based) + magnitudes at breakout
    if birth_lookup:
        birthdate = birth_lookup.get((name, draft_year))
        if birthdate is not None:
            ba, bp_yptpa, bp_yprr = compute_breakout_age(
                seasons, birthdate, team_att_lookup, team_games_lookup
            )
            result["breakout_age"] = ba
            result["breakout_yptpa"] = bp_yptpa
            result["breakout_yprr"] = bp_yprr

    # Peak per-stat features
    result.update(compute_peak_stats(seasons))

    # Best yards per team pass attempt
    if team_att_lookup:
        result["best_yards_per_team_pass_att"] = compute_best_yptpa(seasons, team_att_lookup)

    return result


def build_lookups(all_grades=None):
    """Build lookup dicts for draft ages, birthdates, team pass attempts, and team games."""
    ages = pd.read_csv(os.path.join(DATA_DIR, "draft_ages.csv"))
    ages["birthdate"] = pd.to_datetime(ages["birthdate"])

    # Impute missing birthdates from draft_age where possible:
    # birthdate ≈ draft_date - draft_age_years
    for i, row in ages.iterrows():
        if pd.isna(row["birthdate"]) and pd.notna(row.get("draft_age")):
            draft_date = pd.to_datetime(row.get("draft_date"))
            if pd.notna(draft_date):
                ages.at[i, "birthdate"] = draft_date - pd.Timedelta(days=row["draft_age"] * 365.25)

    birth_lookup = dict(zip(
        zip(ages["name"], ages["draft_year"]),
        ages["birthdate"],
    ))
    draft_age_lookup = dict(zip(
        zip(ages["name"], ages["draft_year"]),
        ages["draft_age"],
    ))

    team_att = pd.read_csv(os.path.join(DATA_DIR, "team_pass_attempts.csv"))
    team_att_lookup = dict(zip(
        zip(team_att["team_pff"], team_att["year"]),
        team_att["pass_att"],
    ))

    # Team games: max player_game_count per team-year as proxy
    team_games_lookup = {}
    if all_grades is not None and len(all_grades) > 0:
        team_games_lookup = all_grades.groupby(
            ["team_name", "grade_year"]
        )["player_game_count"].max().to_dict()

    return birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup


def run_dynasty_aggregation():
    """Main pipeline: aggregate college stats onto dynasty value file."""
    print("Loading grades files...")
    all_grades = load_all_grades(range(2016, 2026))

    print("Loading dynasty value file...")
    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value.csv"))

    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)

    print("Fitting aDOT regression...")
    adot_coef = fit_adot_regression(all_grades)
    print(f"  catch% = {adot_coef[0]:.2f} * aDOT + {adot_coef[1]:.2f}")

    print("Aggregating per player...")
    rows = []
    matched = 0
    unmatched_names = []

    for _, row in dynasty.iterrows():
        result = aggregate_player(
            all_grades, row["name"], row["draft_year"],
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            team_games_lookup=team_games_lookup,
            adot_coef=adot_coef,
        )
        if result:
            matched += 1
        else:
            unmatched_names.append(row["name"])
        rows.append(result)

    print(f"Matched: {matched}/{len(dynasty)}")
    if unmatched_names:
        print(f"Unmatched ({len(unmatched_names)}): {unmatched_names[:10]}...")

    # Draft age: set on dynasty for unmatched players too (aggregate_player handles matched ones)
    dynasty["draft_age"] = [
        draft_age_lookup.get((row["name"], row["draft_year"]), np.nan)
        for _, row in dynasty.iterrows()
    ]
    da_count = dynasty["draft_age"].notna().sum()
    print(f"Draft age matched: {da_count}/{len(dynasty)}")

    # Merge all aggregated features
    agg_df = pd.DataFrame(rows)

    # Remove draft_age from agg_df to avoid duplicate column (dynasty already has it)
    for drop_col in ["draft_age"]:
        if drop_col in agg_df.columns:
            agg_df = agg_df.drop(columns=[drop_col])
    result = pd.concat([dynasty, agg_df], axis=1)

    # Save
    output_path = os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv")
    result.to_csv(output_path, index=False)

    print(f"\nTotal columns: {len(result.columns)}")
    print(f"Saved to {output_path}")

    # Show a sample
    print("\nSample (Justin Jefferson):")
    jj = result[result["name"] == "Justin Jefferson"]
    if len(jj) > 0:
        row = jj.iloc[0]
        for col in result.columns:
            val = row[col]
            if not isinstance(val, (list, np.ndarray)) and pd.notna(val):
                print(f"  {col}: {val}")

    return result


if __name__ == "__main__":
    run_dynasty_aggregation()
