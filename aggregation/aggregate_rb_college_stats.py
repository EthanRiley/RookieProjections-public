#!/usr/bin/env python3
"""
Aggregate college rushing grades for RBs.

For each player, computes:
  - Career stats: weighted averages for rate stats, per-game totals for counting stats
  - Best season stats: season with highest grades_offense
  - Best 2 season stats: top 2 seasons by grades_offense
  - Peak stats: per-stat peaks (not tied to best grade season)
  - Peak 2 stats: top 2 seasons per individual stat

All counting stats are per-game to account for varying game counts.

Reads:
  - rb_data/pff_rb_{2014..2025}.csv
Outputs:
  - Aggregated dict per player (used by downstream pipelines)
"""

import os
import re

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rb_data")
WR_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

# --- Name normalization ---
SUFFIXES_RE = re.compile(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)


def normalize_name(name):
    n = SUFFIXES_RE.sub('', str(name)).strip()
    n = n.replace('.', '').replace("'", '').lower()
    return ' '.join(n.split())


# ============================================================
# Column classification for weighted aggregation
# ============================================================

# Game-weighted: PFF grades (snap-level metrics)
GAME_WEIGHTED_COLS = [
    "grades_offense",
    "grades_run",
    "grades_pass",
    "grades_pass_route",
    "grades_pass_block",
    "grades_run_block",
    "grades_hands_fumble",
    "grades_offense_penalty",
]

# Attempt-weighted: per-rush stats
ATTEMPT_WEIGHTED_COLS = [
    "ypa",
    "yco_attempt",
    "breakaway_percent",
]

# Target-weighted: receiving rate stats
TARGET_WEIGHTED_COLS = [
    "yprr",
]

# Counting stats — aggregate as per-game
COUNTING_COLS = [
    "attempts",
    "avoided_tackles",
    "breakaway_attempts",
    "breakaway_yards",
    "designed_yards",
    "drops",
    "elu_recv_mtf",
    "elu_rush_mtf",
    "elu_yco",
    "explosive",
    "first_downs",
    "fumbles",
    "gap_attempts",
    "rec_yards",
    "receptions",
    "routes",
    "run_plays",
    "scramble_yards",
    "scrambles",
    "targets",
    "total_touches",
    "touchdowns",
    "yards",
    "yards_after_contact",
    "zone_attempts",
]

# Stats that are already rates
RATE_COLS = GAME_WEIGHTED_COLS + ATTEMPT_WEIGHTED_COLS + TARGET_WEIGHTED_COLS

# Minimum attempts for season eligibility (analogous to 200-route minimum for WRs)
MIN_ATTEMPTS = 100

# Peak-gated: minimum PFF grade to qualify for peak selection
PEAK_GATE_GRADE = 80.0

# Graduated age adjustment: multiplicative adjustment per age class on Sept 1
# Same structure as WR YPRR adjustment, tuned for RB rate stats
GRADUATED_AGE_ADJ = {
    # (age_lower, age_upper): multiplier
    (0, 19.5): 1.25,      # freshman +25%
    (19.5, 20.5): 1.05,   # sophomore +5%
    (20.5, 21.5): 0.80,   # junior -20%
    (21.5, 99): 0.75,     # senior -25%
}

# P5 teams (shared with WR pipeline)
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
    # Former P5
    'OREGON ST', 'WASH STATE',
}


# ============================================================
# Core aggregation functions
# ============================================================

def aggregate_seasons(seasons: pd.DataFrame, prefix: str = "career") -> dict:
    """Compute weighted averages and per-game counting stats for a set of seasons."""
    games = pd.to_numeric(seasons["player_game_count"], errors="coerce").fillna(0).values
    total_games = games.sum()
    if total_games == 0:
        return {}

    result = {f"{prefix}_games": int(total_games), f"{prefix}_seasons": len(seasons)}

    def _wavg(col, weight_col):
        if col not in seasons.columns:
            return
        vals = pd.to_numeric(seasons[col], errors="coerce")
        wts_raw = pd.to_numeric(seasons[weight_col], errors="coerce") if weight_col in seasons.columns else None
        mask = vals.notna()
        if wts_raw is not None:
            mask = mask & wts_raw.notna()
            wts = wts_raw[mask].values
            if wts.sum() > 0:
                result[f"{prefix}_{col}"] = round(float(np.average(vals[mask], weights=wts)), 2)
                return
        if mask.any():
            result[f"{prefix}_{col}"] = round(float(np.average(vals[mask], weights=games[mask])), 2)

    # Game-weighted: PFF grades
    for col in GAME_WEIGHTED_COLS:
        _wavg(col, "player_game_count")

    # Attempt-weighted: per-rush stats
    for col in ATTEMPT_WEIGHTED_COLS:
        _wavg(col, "attempts")

    # Target-weighted: receiving efficiency
    for col in TARGET_WEIGHTED_COLS:
        _wavg(col, "targets")

    # Per-game for counting stats
    for col in COUNTING_COLS:
        if col in seasons.columns:
            vals = pd.to_numeric(seasons[col], errors="coerce").fillna(0)
            result[f"{prefix}_{col}_pg"] = round(float(vals.sum() / total_games), 2)

    # Derived rates computed from totals (more accurate than averaging rates)
    attempts_total = pd.to_numeric(seasons.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    yards_total = pd.to_numeric(seasons.get("yards", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    yac_total = pd.to_numeric(seasons.get("yards_after_contact", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    at_total = pd.to_numeric(seasons.get("avoided_tackles", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    touches_total = pd.to_numeric(seasons.get("total_touches", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    rec_total = pd.to_numeric(seasons.get("receptions", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    targets_total = pd.to_numeric(seasons.get("targets", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    routes_total = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    rec_yards_total = pd.to_numeric(seasons.get("rec_yards", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    fd_total = pd.to_numeric(seasons.get("first_downs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    td_total = pd.to_numeric(seasons.get("touchdowns", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    fumbles_total = pd.to_numeric(seasons.get("fumbles", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

    explosive_total = pd.to_numeric(seasons.get("explosive", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    elu_rush_mtf_total = pd.to_numeric(seasons.get("elu_rush_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

    if attempts_total > 0:
        result[f"{prefix}_ypa_total"] = round(float(yards_total / attempts_total), 2)
        result[f"{prefix}_yac_per_att"] = round(float(yac_total / attempts_total), 2)
        result[f"{prefix}_avoided_tackles_per_att"] = round(float(at_total / attempts_total), 4)
        result[f"{prefix}_explosive_per_att"] = round(float(explosive_total / attempts_total), 4)
        result[f"{prefix}_elu_rush_mtf_per_att"] = round(float(elu_rush_mtf_total / attempts_total), 4)
    if touches_total > 0:
        result[f"{prefix}_avoided_tackles_per_touch"] = round(float(at_total / touches_total), 4)
        result[f"{prefix}_fumble_rate"] = round(float(fumbles_total / touches_total), 4)
        result[f"{prefix}_td_per_touch"] = round(float(td_total / touches_total), 4)
        result[f"{prefix}_fd_per_touch"] = round(float(fd_total / touches_total), 4)
    if targets_total > 0:
        result[f"{prefix}_catch_rate"] = round(float(rec_total / targets_total * 100), 2)
    if routes_total > 0:
        result[f"{prefix}_yprr_total"] = round(float(rec_yards_total / routes_total), 4)
    if rec_total > 0:
        result[f"{prefix}_rec_yards_per_rec"] = round(float(rec_yards_total / rec_total), 2)

    # Elusive rating from totals
    elu_yco = pd.to_numeric(seasons.get("elu_yco", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    elu_mtf = (pd.to_numeric(seasons.get("elu_rush_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
               + pd.to_numeric(seasons.get("elu_recv_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    if touches_total > 0:
        result[f"{prefix}_elusive_rating_total"] = round(float((elu_yco + elu_mtf) / touches_total), 2)

    return result


def _select_best_n(seasons: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select the top n seasons by grades_offense, with P5 preference and min attempts."""
    attempts = pd.to_numeric(seasons.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
    eligible = seasons[attempts >= MIN_ATTEMPTS]

    # Prefer P5 seasons
    if "team_name" in eligible.columns:
        p5 = eligible[eligible["team_name"].isin(P5_TEAMS)]
        if len(p5) >= n:
            eligible = p5

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce") if len(eligible) > 0 else pd.Series(dtype=float)
    if grades.notna().sum() >= n:
        top_idx = grades.nlargest(n).index
        return eligible.loc[top_idx]
    elif len(eligible) > 0:
        return eligible
    else:
        return seasons


def best_season_stats(seasons: pd.DataFrame) -> dict:
    """Stats from the single best season by grades_offense."""
    selected = _select_best_n(seasons, 1)
    if len(selected) == 0:
        return {}
    grades = pd.to_numeric(selected["grades_offense"], errors="coerce")
    if grades.notna().any():
        best = selected.loc[grades.idxmax()]
    else:
        best = selected.iloc[0]

    games = pd.to_numeric(best.get("player_game_count", 0), errors="coerce")
    if pd.isna(games) or games == 0:
        return {}

    result = {"best_season_games": int(games)}

    for col in RATE_COLS:
        if col in seasons.columns:
            val = pd.to_numeric(best.get(col), errors="coerce")
            if pd.notna(val):
                result[f"best_{col}"] = round(float(val), 2)

    for col in COUNTING_COLS:
        if col in seasons.columns:
            val = pd.to_numeric(best.get(col, 0), errors="coerce")
            if pd.isna(val):
                val = 0
            result[f"best_{col}_pg"] = round(float(val / games), 2)

    # Derived per-touch/per-attempt rates for best season
    att = pd.to_numeric(best.get("attempts", 0), errors="coerce") or 0
    yards = pd.to_numeric(best.get("yards", 0), errors="coerce") or 0
    yac = pd.to_numeric(best.get("yards_after_contact", 0), errors="coerce") or 0
    at = pd.to_numeric(best.get("avoided_tackles", 0), errors="coerce") or 0
    touches = pd.to_numeric(best.get("total_touches", 0), errors="coerce") or 0
    targets = pd.to_numeric(best.get("targets", 0), errors="coerce") or 0
    rec = pd.to_numeric(best.get("receptions", 0), errors="coerce") or 0
    rec_yards = pd.to_numeric(best.get("rec_yards", 0), errors="coerce") or 0
    routes = pd.to_numeric(best.get("routes", 0), errors="coerce") or 0

    explosive = pd.to_numeric(best.get("explosive", 0), errors="coerce") or 0
    elu_rush_mtf = pd.to_numeric(best.get("elu_rush_mtf", 0), errors="coerce") or 0

    if att > 0:
        result["best_ypa_total"] = round(float(yards / att), 2)
        result["best_yac_per_att"] = round(float(yac / att), 2)
        result["best_avoided_tackles_per_att"] = round(float(at / att), 4)
        result["best_explosive_per_att"] = round(float(explosive / att), 4)
        result["best_elu_rush_mtf_per_att"] = round(float(elu_rush_mtf / att), 4)
    if touches > 0:
        result["best_avoided_tackles_per_touch"] = round(float(at / touches), 4)
    if targets > 0:
        result["best_catch_rate"] = round(float(rec / targets * 100), 2)
    if routes > 0:
        result["best_yprr_total"] = round(float(rec_yards / routes), 4)

    return result


def best2_stats(seasons: pd.DataFrame) -> dict:
    """Stats from the top 2 seasons by grades_offense."""
    selected = _select_best_n(seasons, 2)
    return aggregate_seasons(selected, prefix="best2")


def peak_stats(seasons: pd.DataFrame) -> dict:
    """Per-stat peaks across all eligible seasons (not tied to best grade season).

    For rate stats: best single-season value (with minimum volume filters).
    For efficiency: compute from season totals then take the max.
    """
    result = {}
    attempts = pd.to_numeric(seasons.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
    eligible = seasons[attempts >= MIN_ATTEMPTS]

    if len(eligible) == 0:
        return result

    # Peak grades
    for col in GAME_WEIGHTED_COLS:
        if col in eligible.columns:
            vals = pd.to_numeric(eligible[col], errors="coerce")
            if vals.notna().any():
                result[f"peak_{col}"] = round(float(vals.max()), 2)

    # Peak YPA
    yards = pd.to_numeric(eligible["yards"], errors="coerce").fillna(0)
    att = pd.to_numeric(eligible["attempts"], errors="coerce").fillna(0)
    valid = att >= MIN_ATTEMPTS
    if valid.any():
        ypa = yards[valid] / att[valid]
        result["peak_ypa"] = round(float(ypa.max()), 2)

    # Peak YAC/attempt
    yac = pd.to_numeric(eligible.get("yards_after_contact", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if valid.any():
        yac_pa = yac[valid] / att[valid]
        result["peak_yac_per_att"] = round(float(yac_pa.max()), 2)

    # Peak avoided tackles per attempt
    at = pd.to_numeric(eligible.get("avoided_tackles", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if valid.any():
        at_pa = at[valid] / att[valid]
        result["peak_avoided_tackles_per_att"] = round(float(at_pa.max()), 4)

    # Peak avoided tackles per touch
    touches = pd.to_numeric(eligible.get("total_touches", pd.Series(dtype=float)), errors="coerce").fillna(0)
    valid_touches = touches >= MIN_ATTEMPTS
    if valid_touches.any():
        at_pt = at[valid_touches] / touches[valid_touches]
        result["peak_avoided_tackles_per_touch"] = round(float(at_pt.max()), 4)

    # Peak explosive per attempt
    explosive = pd.to_numeric(eligible.get("explosive", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if valid.any():
        exp_pa = explosive[valid] / att[valid]
        result["peak_explosive_per_att"] = round(float(exp_pa.max()), 4)

    # Peak elu_rush_mtf per attempt
    elu_rush_mtf = pd.to_numeric(eligible.get("elu_rush_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if valid.any():
        mtf_pa = elu_rush_mtf[valid] / att[valid]
        result["peak_elu_rush_mtf_per_att"] = round(float(mtf_pa.max()), 4)

    # Peak elusive rating
    if "elusive_rating" in eligible.columns:
        er = pd.to_numeric(eligible["elusive_rating"], errors="coerce")
        if er.notna().any():
            result["peak_elusive_rating"] = round(float(er.max()), 2)

    # Peak catch rate (min 15 targets)
    targets = pd.to_numeric(eligible.get("targets", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rec = pd.to_numeric(eligible.get("receptions", pd.Series(dtype=float)), errors="coerce").fillna(0)
    valid_tgt = targets >= 15
    if valid_tgt.any():
        cr = rec[valid_tgt] / targets[valid_tgt] * 100
        result["peak_catch_rate"] = round(float(cr.max()), 2)

    # Peak YPRR (min 50 routes)
    routes = pd.to_numeric(eligible.get("routes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rec_yards = pd.to_numeric(eligible.get("rec_yards", pd.Series(dtype=float)), errors="coerce").fillna(0)
    valid_routes = routes >= 50
    if valid_routes.any():
        yprr = rec_yards[valid_routes] / routes[valid_routes]
        result["peak_yprr"] = round(float(yprr.max()), 4)

    # Peak breakaway rate
    ba_att = pd.to_numeric(eligible.get("breakaway_attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if valid.any() and ba_att[valid].sum() > 0:
        bp = pd.to_numeric(eligible.get("breakaway_percent", pd.Series(dtype=float)), errors="coerce")
        if bp.notna().any():
            result["peak_breakaway_percent"] = round(float(bp[valid].max()), 2)

    return result


def peak2_stats(seasons: pd.DataFrame) -> dict:
    """Top 2 seasons per individual stat (not tied to a single 'best' season)."""
    result = {}
    attempts = pd.to_numeric(seasons.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
    eligible = seasons[attempts >= MIN_ATTEMPTS]

    if len(eligible) < 2:
        return result

    # Peak2 YPA: top 2 seasons by YPA, weighted by attempts
    yards = pd.to_numeric(eligible["yards"], errors="coerce").fillna(0)
    att = pd.to_numeric(eligible["attempts"], errors="coerce").fillna(0)
    ypa = yards / att.replace(0, np.nan)
    if ypa.notna().sum() >= 2:
        top2_idx = ypa.nlargest(2).index
        result["peak2_ypa"] = round(float(yards.loc[top2_idx].sum() / att.loc[top2_idx].sum()), 2)

    # Peak2 avoided tackles per attempt
    at = pd.to_numeric(eligible.get("avoided_tackles", pd.Series(dtype=float)), errors="coerce").fillna(0)
    at_pa = at / att.replace(0, np.nan)
    if at_pa.notna().sum() >= 2:
        top2_idx = at_pa.nlargest(2).index
        result["peak2_avoided_tackles_per_att"] = round(float(at.loc[top2_idx].sum() / att.loc[top2_idx].sum()), 4)

    # Peak2 elusive rating
    if "elusive_rating" in eligible.columns:
        er = pd.to_numeric(eligible["elusive_rating"], errors="coerce")
        if er.notna().sum() >= 2:
            top2 = er.nlargest(2)
            games = pd.to_numeric(eligible["player_game_count"], errors="coerce").fillna(0)
            result["peak2_elusive_rating"] = round(float(np.average(top2, weights=games.loc[top2.index])), 2)

    # Peak2 grades_offense
    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().sum() >= 2:
        top2 = grades.nlargest(2)
        games = pd.to_numeric(eligible["player_game_count"], errors="coerce").fillna(0)
        result["peak2_grades_offense"] = round(float(np.average(top2, weights=games.loc[top2.index])), 2)

    # Peak2 grades_run
    if "grades_run" in eligible.columns:
        gr = pd.to_numeric(eligible["grades_run"], errors="coerce")
        if gr.notna().sum() >= 2:
            top2 = gr.nlargest(2)
            games = pd.to_numeric(eligible["player_game_count"], errors="coerce").fillna(0)
            result["peak2_grades_run"] = round(float(np.average(top2, weights=games.loc[top2.index])), 2)

    # Peak2 YAC per attempt
    yac = pd.to_numeric(eligible.get("yards_after_contact", pd.Series(dtype=float)), errors="coerce").fillna(0)
    yac_pa = yac / att.replace(0, np.nan)
    if yac_pa.notna().sum() >= 2:
        top2_idx = yac_pa.nlargest(2).index
        result["peak2_yac_per_att"] = round(float(yac.loc[top2_idx].sum() / att.loc[top2_idx].sum()), 2)

    # Peak2 explosive per attempt
    explosive = pd.to_numeric(eligible.get("explosive", pd.Series(dtype=float)), errors="coerce").fillna(0)
    exp_pa = explosive / att.replace(0, np.nan)
    if exp_pa.notna().sum() >= 2:
        top2_idx = exp_pa.nlargest(2).index
        result["peak2_explosive_per_att"] = round(float(explosive.loc[top2_idx].sum() / att.loc[top2_idx].sum()), 4)

    # Peak2 elu_rush_mtf per attempt
    elu_rush_mtf = pd.to_numeric(eligible.get("elu_rush_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0)
    mtf_pa = elu_rush_mtf / att.replace(0, np.nan)
    if mtf_pa.notna().sum() >= 2:
        top2_idx = mtf_pa.nlargest(2).index
        result["peak2_elu_rush_mtf_per_att"] = round(float(elu_rush_mtf.loc[top2_idx].sum() / att.loc[top2_idx].sum()), 4)

    # Peak2 catch rate (min 15 targets per season)
    targets = pd.to_numeric(eligible.get("targets", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rec = pd.to_numeric(eligible.get("receptions", pd.Series(dtype=float)), errors="coerce").fillna(0)
    valid_tgt = targets >= 15
    if valid_tgt.sum() >= 2:
        cr = rec[valid_tgt] / targets[valid_tgt]
        top2_idx = cr.nlargest(2).index
        result["peak2_catch_rate"] = round(float(rec.loc[top2_idx].sum() / targets.loc[top2_idx].sum() * 100), 2)

    return result


# ============================================================
# Data loading and player-level aggregation
# ============================================================

def load_all_rb_grades(year_range=range(2014, 2026)):
    """Load and concatenate all RB grades files."""
    all_grades = []
    for yr in year_range:
        path = os.path.join(DATA_DIR, f"pff_rb_{yr}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
            d["grade_year"] = yr
            all_grades.append(d)
    if not all_grades:
        return pd.DataFrame()
    result = pd.concat(all_grades, ignore_index=True)
    result["_join_key"] = result["player"].apply(normalize_name)
    return result


def get_player_seasons(all_grades, name, draft_year, birthdate=None):
    """Get a player's college seasons before their draft year."""
    key = normalize_name(name)
    seasons = all_grades[
        (all_grades["_join_key"] == key) & (all_grades["grade_year"] <= draft_year)
    ].copy()

    # Age-based filtering for name collisions
    if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
        min_year = birthdate.year + 18 if hasattr(birthdate, 'year') else None
        if min_year is not None:
            sept1_min = pd.Timestamp(f"{min_year}-09-01")
            if sept1_min < birthdate + pd.DateOffset(years=18):
                min_year += 1
            seasons = seasons[seasons["grade_year"] >= min_year]

    # Cap at 5 years before draft
    if len(seasons) > 0:
        earliest = draft_year - 5
        seasons = seasons[seasons["grade_year"] >= earliest]

    return seasons


def peak_gated_stats(seasons: pd.DataFrame) -> dict:
    """Peak stats gated by PFF grade >= PEAK_GATE_GRADE.

    Only considers seasons where grades_offense >= 80. If no season qualifies,
    returns empty dict (feature will be NaN — that's informative).
    """
    result = {}
    attempts = pd.to_numeric(seasons.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
    grades = pd.to_numeric(seasons.get("grades_offense", pd.Series(dtype=float)), errors="coerce").fillna(0)
    eligible = seasons[(attempts >= MIN_ATTEMPTS) & (grades >= PEAK_GATE_GRADE)]

    if len(eligible) == 0:
        return result

    att = pd.to_numeric(eligible["attempts"], errors="coerce").fillna(0)
    valid = att >= MIN_ATTEMPTS

    # Peak-gated rate stats (per-attempt)
    if valid.any():
        explosive = pd.to_numeric(eligible.get("explosive", pd.Series(dtype=float)), errors="coerce").fillna(0)
        result["pg_explosive_per_att"] = round(float((explosive[valid] / att[valid]).max()), 4)

        elu_rush_mtf = pd.to_numeric(eligible.get("elu_rush_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0)
        result["pg_elu_rush_mtf_per_att"] = round(float((elu_rush_mtf[valid] / att[valid]).max()), 4)

        at = pd.to_numeric(eligible.get("avoided_tackles", pd.Series(dtype=float)), errors="coerce").fillna(0)
        result["pg_avoided_tackles_per_att"] = round(float((at[valid] / att[valid]).max()), 4)

        yards = pd.to_numeric(eligible.get("yards", pd.Series(dtype=float)), errors="coerce").fillna(0)
        result["pg_ypa"] = round(float((yards[valid] / att[valid]).max()), 2)

        yac = pd.to_numeric(eligible.get("yards_after_contact", pd.Series(dtype=float)), errors="coerce").fillna(0)
        result["pg_yco_attempt"] = round(float((yac[valid] / att[valid]).max()), 2)

    # Peak-gated per-game counting stats
    games = pd.to_numeric(eligible["player_game_count"], errors="coerce").fillna(0)
    valid_games = games > 0
    if valid_games.any():
        for col in ["explosive", "elu_rush_mtf", "rec_yards"]:
            vals = pd.to_numeric(eligible.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0)
            result[f"pg_{col}_pg"] = round(float((vals[valid_games] / games[valid_games]).max()), 2)

    # Peak-gated receiving rate
    routes = pd.to_numeric(eligible.get("routes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rec_yards = pd.to_numeric(eligible.get("rec_yards", pd.Series(dtype=float)), errors="coerce").fillna(0)
    valid_routes = routes >= 50
    if valid_routes.any():
        result["pg_yprr"] = round(float((rec_yards[valid_routes] / routes[valid_routes]).max()), 4)

    return result


def _get_season_age(season_row, birthdate):
    """Compute age on Sept 1 of the season year."""
    if birthdate is None or pd.isna(birthdate):
        return None
    yr = season_row.get("grade_year")
    if pd.isna(yr):
        return None
    sept1 = pd.Timestamp(f"{int(yr)}-09-01")
    return (sept1 - pd.Timestamp(birthdate)).days / 365.25


def _age_multiplier(age):
    """Look up graduated age adjustment multiplier."""
    if age is None:
        return 1.0
    for (lo, hi), mult in GRADUATED_AGE_ADJ.items():
        if lo <= age < hi:
            return mult
    return 1.0


def age_adjusted_stats(seasons: pd.DataFrame, birthdate) -> dict:
    """Compute age-adjusted versions of key rate stats.

    For each eligible season, compute the rate stat, apply the graduated age
    multiplier, then take the best adjusted value (peak selection with age penalty).
    """
    result = {}
    if birthdate is None or pd.isna(birthdate):
        return result

    attempts = pd.to_numeric(seasons.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
    eligible = seasons[attempts >= MIN_ATTEMPTS].copy()

    if len(eligible) == 0:
        return result

    att = pd.to_numeric(eligible["attempts"], errors="coerce").fillna(0)

    # Compute age multiplier for each season
    mults = []
    for _, row in eligible.iterrows():
        age = _get_season_age(row, birthdate)
        mults.append(_age_multiplier(age))
    mults = np.array(mults)

    # Age-adjusted explosive per attempt (best season after adjustment)
    explosive = pd.to_numeric(eligible.get("explosive", pd.Series(dtype=float)), errors="coerce").fillna(0)
    exp_rate = (explosive / att).values
    adjusted = exp_rate * mults
    result["adj_explosive_per_att"] = round(float(np.nanmax(adjusted)), 4)

    # Age-adjusted MTF per attempt
    elu_rush_mtf = pd.to_numeric(eligible.get("elu_rush_mtf", pd.Series(dtype=float)), errors="coerce").fillna(0)
    mtf_rate = (elu_rush_mtf / att).values
    adjusted = mtf_rate * mults
    result["adj_elu_rush_mtf_per_att"] = round(float(np.nanmax(adjusted)), 4)

    # Age-adjusted YPA
    yards = pd.to_numeric(eligible.get("yards", pd.Series(dtype=float)), errors="coerce").fillna(0)
    ypa = (yards / att).values
    adjusted = ypa * mults
    result["adj_ypa"] = round(float(np.nanmax(adjusted)), 2)

    # Age-adjusted YPRR
    routes = pd.to_numeric(eligible.get("routes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rec_yards = pd.to_numeric(eligible.get("rec_yards", pd.Series(dtype=float)), errors="coerce").fillna(0)
    valid_routes = routes >= 50
    if valid_routes.any():
        yprr = (rec_yards[valid_routes] / routes[valid_routes]).values
        adjusted = yprr * mults[valid_routes.values]
        result["adj_yprr"] = round(float(np.nanmax(adjusted)), 4)

    # Age-adjusted rec_yards per game
    games = pd.to_numeric(eligible["player_game_count"], errors="coerce").fillna(0)
    valid_games = games > 0
    if valid_games.any():
        rypg = (pd.to_numeric(eligible.get("rec_yards", pd.Series(dtype=float)), errors="coerce").fillna(0)[valid_games] / games[valid_games]).values
        adjusted = rypg * mults[valid_games.values]
        result["adj_rec_yards_pg"] = round(float(np.nanmax(adjusted)), 2)

    # Age-adjusted explosive per game
    if valid_games.any():
        epg = (explosive[valid_games] / games[valid_games]).values
        adjusted = epg * mults[valid_games.values]
        result["adj_explosive_pg"] = round(float(np.nanmax(adjusted)), 2)

    return result



def aggregate_player(all_grades, name, draft_year, birthdate=None):
    """Full aggregation for a single RB. Returns a dict of all computed features."""
    seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)

    if len(seasons) == 0:
        return {}

    result = {}

    # Career stats (all seasons)
    result.update(aggregate_seasons(seasons, prefix="career"))

    # Best season (single best by grades_offense)
    result.update(best_season_stats(seasons))

    # Best 2 seasons (top 2 by grades_offense)
    result.update(best2_stats(seasons))

    # Peak stats (per-stat peaks)
    result.update(peak_stats(seasons))

    # Peak 2 stats (top 2 per individual stat)
    result.update(peak2_stats(seasons))

    # Peak-gated stats (peak values from seasons with grades_offense >= 80)
    result.update(peak_gated_stats(seasons))

    # Age-adjusted stats (graduated multiplier on rate stats)
    result.update(age_adjusted_stats(seasons, birthdate))

    return result


def run_rb_aggregation_sample():
    """Quick test: aggregate a few notable RBs."""
    print("Loading RB grades files...")
    all_grades = load_all_rb_grades()
    print(f"Loaded {len(all_grades)} player-seasons across {all_grades['grade_year'].nunique()} years")

    test_players = [
        ("Ashton Jeanty", 2025),
        ("Bijan Robinson", 2023),
        ("Breece Hall", 2022),
        ("Jahmyr Gibbs", 2023),
        ("Kenneth Walker", 2022),
    ]

    for name, draft_year in test_players:
        result = aggregate_player(all_grades, name, draft_year)
        if result:
            print(f"\n{name} ({draft_year}):")
            for k, v in sorted(result.items()):
                print(f"  {k}: {v}")
        else:
            print(f"\n{name} ({draft_year}): NOT FOUND")


if __name__ == "__main__":
    run_rb_aggregation_sample()
