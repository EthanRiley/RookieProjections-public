#!/usr/bin/env python3
"""
Compare breakout age variants + quality-adjusted variants + draft age head-to-head.

Binary threshold variants:
  1. 650 total yards (volume)
  2. 45 ypg + 8 games (per-game volume)
  3. 1.4 game-normalized YPTPA + 8 games (team-adjusted volume) [current]
  4. 45 ypg + 8 games + 2.0 YPRR (volume + efficiency)
  5. Draft age (no breakout — baseline)
  6. 2.0 YPRR + 150 routes (pure efficiency)
  7. 25%+ dominator (team yard share) + 8 games (market share)
  8. 1.4 YPTPA + 2.0 YPRR + 8 games (strictest gate)
  9. Composite: earliest of (YPTPA >= 1.4 OR YPRR >= 2.2) + 8 games

Quality-adjusted variants (based on YPTPA breakout):
 10. Ratio-scaled: age * (1.4 / actual_yptpa) — compresses age for dominant breakouts
 11. Z-score adjusted: age - z_score * 0.5 — discounts age by how exceptional the breakout was
 12. Log-magnitude weighted: age / log2(actual_yptpa / 1.4 + 1) — diminishing returns on excess
 13. Breakout magnitude: just the YPTPA value at breakout (standalone, no age component)

Runs Layer 1 evaluation: Spearman, AUC, era stability, residual analysis.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}


# ---------------------------------------------------------------------------
# Breakout age computation functions
# ---------------------------------------------------------------------------

def _season_age(birthdate, year):
    """Age on Sept 1 of the given year."""
    sept1 = pd.Timestamp(f"{year}-09-01")
    return round((sept1 - birthdate).days / 365.25, 2)


def breakout_650_yards(seasons, birthdate, **_):
    """First season with 650+ total receiving yards."""
    for _, s in seasons.sort_values("grade_year").iterrows():
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        if yards >= 650:
            return _season_age(birthdate, s["grade_year"])
    return np.nan


def breakout_45ypg(seasons, birthdate, **_):
    """First season with 45+ ypg and 8+ games."""
    for _, s in seasons.sort_values("grade_year").iterrows():
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        if games >= 8 and yards / games >= 45:
            return _season_age(birthdate, s["grade_year"])
    return np.nan


def breakout_yptpa(seasons, birthdate, team_att_lookup=None, team_games_lookup=None, **_):
    """First season with game-normalized YPTPA >= 1.4 and 8+ games. [Current model]"""
    if team_att_lookup is None or team_games_lookup is None:
        return np.nan
    for _, s in seasons.sort_values("grade_year").iterrows():
        yr = s["grade_year"]
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        team = s.get("team_name", "")
        att = team_att_lookup.get((team, yr))
        tg = team_games_lookup.get((team, yr))
        if att and att > 0 and games >= 8 and tg and tg > 0:
            ypg = yards / games
            att_pg = att / tg
            if ypg / att_pg >= 1.4:
                return _season_age(birthdate, yr)
    return np.nan


def breakout_45ypg_yprr(seasons, birthdate, **_):
    """First season with 45+ ypg, 8+ games, AND 2.0+ YPRR."""
    for _, s in seasons.sort_values("grade_year").iterrows():
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
        if games >= 8 and yards / games >= 45 and routes > 0 and yards / routes >= 2.0:
            return _season_age(birthdate, s["grade_year"])
    return np.nan


def breakout_yprr_routes(seasons, birthdate, **_):
    """First season with 2.0+ YPRR and 150+ routes."""
    for _, s in seasons.sort_values("grade_year").iterrows():
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
        if routes >= 150 and yards / routes >= 2.0:
            return _season_age(birthdate, s["grade_year"])
    return np.nan


def breakout_dominator(seasons, birthdate, team_yards_lookup=None, **_):
    """First season with 25%+ share of team receiving yards and 8+ games."""
    if team_yards_lookup is None:
        return np.nan
    for _, s in seasons.sort_values("grade_year").iterrows():
        yr = s["grade_year"]
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        team = s.get("team_name", "")
        team_yards = team_yards_lookup.get((team, yr), 0)
        if games >= 8 and team_yards > 0 and yards / team_yards >= 0.25:
            return _season_age(birthdate, yr)
    return np.nan


def breakout_yptpa_yprr(seasons, birthdate, team_att_lookup=None, team_games_lookup=None, **_):
    """First season with YPTPA >= 1.4 AND YPRR >= 2.0, 8+ games. Strictest."""
    if team_att_lookup is None or team_games_lookup is None:
        return np.nan
    for _, s in seasons.sort_values("grade_year").iterrows():
        yr = s["grade_year"]
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
        team = s.get("team_name", "")
        att = team_att_lookup.get((team, yr))
        tg = team_games_lookup.get((team, yr))
        if att and att > 0 and games >= 8 and tg and tg > 0 and routes > 0:
            ypg = yards / games
            att_pg = att / tg
            yprr = yards / routes
            if ypg / att_pg >= 1.4 and yprr >= 2.0:
                return _season_age(birthdate, yr)
    return np.nan


def _season_yptpa(s, team_att_lookup, team_games_lookup):
    """Compute game-normalized YPTPA for a single season. Returns None if not computable."""
    yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
    games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
    team = s.get("team_name", "")
    yr = s["grade_year"]
    att = team_att_lookup.get((team, yr))
    tg = team_games_lookup.get((team, yr))
    if att and att > 0 and games > 0 and tg and tg > 0:
        return (yards / games) / (att / tg)
    return None


def breakout_composite(seasons, birthdate, team_att_lookup=None, team_games_lookup=None, **_):
    """Earliest season with (YPTPA >= 1.4 OR YPRR >= 2.2) and 8+ games."""
    if team_att_lookup is None or team_games_lookup is None:
        return np.nan
    for _, s in seasons.sort_values("grade_year").iterrows():
        yr = s["grade_year"]
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
        team = s.get("team_name", "")
        att = team_att_lookup.get((team, yr))
        tg = team_games_lookup.get((team, yr))
        if games < 8:
            continue
        yptpa_pass = False
        if att and att > 0 and tg and tg > 0:
            ypg = yards / games
            att_pg = att / tg
            yptpa_pass = (ypg / att_pg >= 1.4)
        yprr_pass = (routes > 0 and yards / routes >= 2.2)
        if yptpa_pass or yprr_pass:
            return _season_age(birthdate, yr)
    return np.nan


def _find_breakout_yptpa_details(seasons, birthdate, team_att_lookup, team_games_lookup):
    """Find first YPTPA breakout season and return (age, yptpa_value) or (nan, nan)."""
    if team_att_lookup is None or team_games_lookup is None:
        return np.nan, np.nan
    for _, s in seasons.sort_values("grade_year").iterrows():
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
        if games < 8:
            continue
        yptpa = _season_yptpa(s, team_att_lookup, team_games_lookup)
        if yptpa is not None and yptpa >= 1.4:
            age = _season_age(birthdate, s["grade_year"])
            return age, yptpa
    return np.nan, np.nan


def breakout_ratio_scaled(seasons, birthdate, team_att_lookup=None, team_games_lookup=None, **_):
    """Ratio-scaled: age * (threshold / actual_yptpa).

    A 19yo breakout at 2.1 YPTPA -> 19 * (1.4/2.1) = 12.67
    A 19yo breakout at 1.4 YPTPA -> 19 * (1.4/1.4) = 19.0
    Lower is better (younger + more dominant).
    """
    age, yptpa = _find_breakout_yptpa_details(seasons, birthdate, team_att_lookup, team_games_lookup)
    if pd.isna(age):
        return np.nan
    return round(age * (1.4 / yptpa), 2)


def breakout_zscore_adj(seasons, birthdate, team_att_lookup=None, team_games_lookup=None,
                        yptpa_mean=None, yptpa_std=None, **_):
    """Z-score adjusted: age - z_score * 0.5.

    z_score = (breakout_yptpa - population_mean) / population_std
    Discounts age by how exceptional the breakout was relative to all seasons.
    The 0.5 scaling factor keeps the adjustment in a reasonable range (~0.5-2 years).
    """
    age, yptpa = _find_breakout_yptpa_details(seasons, birthdate, team_att_lookup, team_games_lookup)
    if pd.isna(age) or yptpa_mean is None or yptpa_std is None or yptpa_std == 0:
        return np.nan
    z = (yptpa - yptpa_mean) / yptpa_std
    return round(age - z * 0.5, 2)


def breakout_log_magnitude(seasons, birthdate, team_att_lookup=None, team_games_lookup=None, **_):
    """Log-magnitude weighted: age / log2(actual_yptpa / threshold + 1).

    Diminishing returns on excess YPTPA above threshold.
    A 19yo at 1.4 -> 19 / log2(2.0) = 19.0
    A 19yo at 2.8 -> 19 / log2(3.0) = 11.99
    A 19yo at 4.2 -> 19 / log2(4.0) = 9.5
    """
    age, yptpa = _find_breakout_yptpa_details(seasons, birthdate, team_att_lookup, team_games_lookup)
    if pd.isna(age):
        return np.nan
    return round(age / np.log2(yptpa / 1.4 + 1), 2)


def breakout_magnitude(seasons, birthdate, team_att_lookup=None, team_games_lookup=None, **_):
    """Just the YPTPA value at breakout (no age component). Higher is better."""
    _, yptpa = _find_breakout_yptpa_details(seasons, birthdate, team_att_lookup, team_games_lookup)
    return round(yptpa, 4) if pd.notna(yptpa) else np.nan


# --- YPRR-based quality-adjusted variants ---

def _find_breakout_yprr_details(seasons, birthdate):
    """Find first YPRR breakout season (2.0+ YPRR, 150+ routes) and return (age, yprr_value)."""
    for _, s in seasons.sort_values("grade_year").iterrows():
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
        if routes >= 150:
            yprr = yards / routes
            if yprr >= 2.0:
                age = _season_age(birthdate, s["grade_year"])
                return age, yprr
    return np.nan, np.nan


def breakout_yprr_ratio(seasons, birthdate, **_):
    """YPRR ratio-scaled: age * (2.0 / actual_yprr).

    A 19yo at 3.0 YPRR -> 19 * (2.0/3.0) = 12.67
    A 19yo at 2.0 YPRR -> 19 * (2.0/2.0) = 19.0
    """
    age, yprr = _find_breakout_yprr_details(seasons, birthdate)
    if pd.isna(age):
        return np.nan
    return round(age * (2.0 / yprr), 2)


def breakout_yprr_zscore(seasons, birthdate, yprr_mean=None, yprr_std=None, **_):
    """YPRR z-score adjusted: age - z * 0.5 where z = (yprr - pop_mean) / pop_std."""
    age, yprr = _find_breakout_yprr_details(seasons, birthdate)
    if pd.isna(age) or yprr_mean is None or yprr_std is None or yprr_std == 0:
        return np.nan
    z = (yprr - yprr_mean) / yprr_std
    return round(age - z * 0.5, 2)


def breakout_yprr_log(seasons, birthdate, **_):
    """YPRR log-magnitude: age / log2(actual_yprr / 2.0 + 1)."""
    age, yprr = _find_breakout_yprr_details(seasons, birthdate)
    if pd.isna(age):
        return np.nan
    return round(age / np.log2(yprr / 2.0 + 1), 2)


def breakout_yprr_magnitude(seasons, birthdate, **_):
    """Just the YPRR value at breakout (no age component)."""
    _, yprr = _find_breakout_yprr_details(seasons, birthdate)
    return round(yprr, 4) if pd.notna(yprr) else np.nan


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VARIANTS = {
    # Binary threshold variants
    "ba_650yards": breakout_650_yards,
    "ba_45ypg": breakout_45ypg,
    "ba_yptpa": breakout_yptpa,
    "ba_45ypg_yprr": breakout_45ypg_yprr,
    "ba_yprr_routes": breakout_yprr_routes,
    "ba_dominator": breakout_dominator,
    "ba_yptpa_yprr": breakout_yptpa_yprr,
    "ba_composite": breakout_composite,
    # Quality-adjusted variants (YPTPA-based)
    "qa_ratio_scaled": breakout_ratio_scaled,
    "qa_zscore_adj": breakout_zscore_adj,
    "qa_log_magnitude": breakout_log_magnitude,
    "qa_magnitude": breakout_magnitude,
    # Quality-adjusted variants (YPRR-based)
    "qy_ratio_scaled": breakout_yprr_ratio,
    "qy_zscore_adj": breakout_yprr_zscore,
    "qy_log_magnitude": breakout_yprr_log,
    "qy_magnitude": breakout_yprr_magnitude,
}


def load_data():
    """Load dynasty data and compute all breakout age variants."""
    from aggregation.aggregate_college_stats import (
        load_all_grades, get_player_seasons, build_lookups,
    )

    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)

    # Build team receiving yards lookup for dominator metric
    grades_numeric = all_grades.copy()
    grades_numeric["yards_num"] = pd.to_numeric(grades_numeric["yards"], errors="coerce").fillna(0)
    team_yards_lookup = grades_numeric.groupby(
        ["team_name", "grade_year"]
    )["yards_num"].sum().to_dict()

    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    dynasty["tier_ordinal"] = dynasty["computed_tier"].map(TIER_ORDER)
    dynasty = dynasty.dropna(subset=["tier_ordinal"]).copy()
    dynasty["tier_ordinal"] = dynasty["tier_ordinal"].astype(int)
    dynasty["hit"] = (dynasty["tier_ordinal"] >= 3).astype(int)

    # Compute YPTPA distribution across all player-seasons (for z-score variant)
    print("  Computing YPTPA distribution for z-score baseline...")
    all_yptpa = []
    for _, s in all_grades.iterrows():
        yptpa = _season_yptpa(s, team_att_lookup, team_games_lookup)
        if yptpa is not None:
            all_yptpa.append(yptpa)
    yptpa_mean = np.mean(all_yptpa)
    yptpa_std = np.std(all_yptpa)
    print(f"    YPTPA distribution: mean={yptpa_mean:.3f}, std={yptpa_std:.3f}, n={len(all_yptpa)}")

    # Compute YPRR distribution across all player-seasons (for YPRR z-score variant)
    print("  Computing YPRR distribution for z-score baseline...")
    yprr_yards = pd.to_numeric(all_grades["yards"], errors="coerce").fillna(0)
    yprr_routes = pd.to_numeric(all_grades["routes"], errors="coerce").fillna(0)
    yprr_mask = yprr_routes > 0
    all_yprr = (yprr_yards[yprr_mask] / yprr_routes[yprr_mask]).values
    yprr_mean = np.mean(all_yprr)
    yprr_std = np.std(all_yprr)
    print(f"    YPRR distribution: mean={yprr_mean:.3f}, std={yprr_std:.3f}, n={len(all_yprr)}")

    kwargs = dict(
        team_att_lookup=team_att_lookup,
        team_games_lookup=team_games_lookup,
        team_yards_lookup=team_yards_lookup,
        yptpa_mean=yptpa_mean,
        yptpa_std=yptpa_std,
        yprr_mean=yprr_mean,
        yprr_std=yprr_std,
    )

    # Compute all breakout age variants
    for var_name, func in VARIANTS.items():
        print(f"  Computing {var_name}...")
        ages = []
        for _, row in dynasty.iterrows():
            name, draft_year = row["name"], row["draft_year"]
            birthdate = birth_lookup.get((name, draft_year))
            seasons = get_player_seasons(all_grades, name, draft_year,
                                         birthdate=birthdate)
            if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
                ages.append(func(seasons, birthdate, **kwargs))
            else:
                ages.append(np.nan)
        dynasty[var_name] = ages

    # Draft age (from lookup)
    dynasty["draft_age_feat"] = [
        draft_age_lookup.get((row["name"], row["draft_year"]), np.nan)
        for _, row in dynasty.iterrows()
    ]

    return dynasty


def evaluate_feature(df, col):
    """Layer 1 evaluation for a single feature."""
    valid = df[[col, "tier_ordinal", "hit", "draft_year"]].dropna()
    if len(valid) < 30:
        return None

    x = valid[col].values
    y = valid["tier_ordinal"].values
    y_hit = valid["hit"].values

    sp, _ = spearmanr(x, y)
    try:
        auc = roc_auc_score(y_hit, x)
        if auc < 0.5:
            auc = 1 - auc
    except ValueError:
        auc = np.nan

    # Era stability
    years = valid["draft_year"].values
    early = years <= 2019
    late = years >= 2020
    sp_e, _ = spearmanr(x[early], y[early]) if early.sum() > 10 else (np.nan, np.nan)
    sp_l, _ = spearmanr(x[late], y[late]) if late.sum() > 10 else (np.nan, np.nan)
    drift = abs(sp_e - sp_l) if pd.notna(sp_e) and pd.notna(sp_l) else np.nan

    return {
        "feature": col, "spearman": round(sp, 3), "auc": round(auc, 3),
        "n": len(valid), "n_missing": len(df) - len(valid),
        "coverage": round(len(valid) / len(df), 3),
        "drift": round(drift, 3) if pd.notna(drift) else np.nan,
        "sp_early": round(sp_e, 3) if pd.notna(sp_e) else np.nan,
        "sp_late": round(sp_l, 3) if pd.notna(sp_l) else np.nan,
    }


def main():
    print("Loading data and computing breakout age variants...")
    dynasty = load_data()

    all_cols = list(VARIANTS.keys()) + ["draft_age_feat"]

    # --- Coverage ---
    print("\n" + "=" * 80)
    print("COVERAGE")
    print("=" * 80)
    print(f"\n  {'Variant':<25s} {'Valid':>6s} {'Missing':>8s} {'Coverage':>10s}")
    print("  " + "-" * 55)
    for col in all_cols:
        valid = dynasty[col].notna().sum()
        missing = dynasty[col].isna().sum()
        print(f"  {col:<25s} {valid:>6d} {missing:>8d} {valid/len(dynasty):>10.1%}")

    # Impute NaNs with max+1 for evaluation (same as model pipeline)
    imp_cols = {}
    for col in all_cols:
        mx = dynasty[col].max()
        imp_name = f"{col}_imp"
        dynasty[imp_name] = dynasty[col].fillna(round(mx + 1, 2) if pd.notna(mx) else 25.0)
        imp_cols[col] = imp_name

    # --- Layer 1: Raw (no imputation) ---
    print("\n" + "=" * 80)
    print("LAYER 1: UNIVARIATE EVALUATION (raw, no imputation)")
    print("=" * 80)
    print(f"\n  {'Variant':<25s} {'Spearman':>10s} {'AUC':>8s} {'N':>6s} {'Drift':>8s}")
    print("  " + "-" * 60)

    raw_results = []
    for col in all_cols:
        res = evaluate_feature(dynasty, col)
        if res:
            raw_results.append(res)
            drift_str = f"{res['drift']:.3f}" if pd.notna(res['drift']) else "N/A"
            print(f"  {col:<25s} {res['spearman']:>+10.3f} {res['auc']:>8.3f} {res['n']:>6d} {drift_str:>8s}")

    # --- Layer 1: Imputed ---
    print("\n" + "=" * 80)
    print("LAYER 1: UNIVARIATE EVALUATION (imputed — NaN = max+1)")
    print("=" * 80)
    print(f"\n  {'Variant':<25s} {'Spearman':>10s} {'AUC':>8s} {'N':>6s} {'Drift':>8s}")
    print("  " + "-" * 60)

    imp_results = []
    for col in all_cols:
        res = evaluate_feature(dynasty, imp_cols[col])
        if res:
            res["feature"] = col  # use original name
            imp_results.append(res)
            drift_str = f"{res['drift']:.3f}" if pd.notna(res['drift']) else "N/A"
            print(f"  {col:<25s} {res['spearman']:>+10.3f} {res['auc']:>8.3f} {res['n']:>6d} {drift_str:>8s}")

    # --- Distribution by tier ---
    print("\n" + "=" * 80)
    print("DISTRIBUTION BY TIER (median age)")
    print("=" * 80)
    tier_names = [t for t, _ in sorted(TIER_ORDER.items(), key=lambda x: x[1])]
    print(f"\n  {'Variant':<25s}", end="")
    for t in tier_names:
        print(f" {t:>8s}", end="")
    print()
    print("  " + "-" * 75)

    for col in all_cols:
        print(f"  {col:<25s}", end="")
        for tier_name in tier_names:
            tier_val = TIER_ORDER[tier_name]
            sub = dynasty[dynasty["tier_ordinal"] == tier_val][col].dropna()
            if len(sub) > 0:
                print(f" {sub.median():>8.2f}", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print()

    # --- Residual analysis after controlling model features ---
    print("\n" + "=" * 80)
    print("RESIDUAL ANALYSIS (after controlling all other model features)")
    print("=" * 80)
    model_feats = [
        "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
        "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
    ]

    print(f"\n  {'Variant':<25s} {'Residual Spearman':>20s} {'N':>6s}")
    print("  " + "-" * 55)

    residual_results = []
    for col in all_cols:
        imp_name = imp_cols[col]
        sub = dynasty[[imp_name] + model_feats + ["tier_ordinal"]].dropna()
        if len(sub) < 30:
            continue
        rank_feat = rankdata(sub[imp_name].values)
        rank_tier = rankdata(sub["tier_ordinal"].values)
        all_ranks = np.column_stack([rankdata(sub[f].values) for f in model_feats])
        X = np.column_stack([all_ranks, np.ones(len(sub))])
        z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
        resid = rank_feat - X @ z
        sp_resid, _ = spearmanr(resid, rank_tier)
        residual_results.append({"feature": col, "residual_spearman": round(sp_resid, 3), "n": len(sub)})
        print(f"  {col:<25s} {sp_resid:>+20.3f} {len(sub):>6d}")

    # --- Efficiency leak analysis ---
    print("\n" + "=" * 80)
    print("EFFICIENCY LEAK ANALYSIS")
    print("=" * 80)
    print("\n  How much of each variant's signal is genuine age signal vs leaked efficiency?")
    print("  We test residual signal after progressively controlling for efficiency sources.\n")

    # Columns we'll control for in each test
    leak_tests = {
        "model_feats_only": model_feats,
        "+ breakout_yptpa_mag": model_feats + ["qa_magnitude_imp"],
        "+ breakout_yprr_mag": model_feats + ["qy_magnitude_imp"],
        "+ both_magnitudes": model_feats + ["qa_magnitude_imp", "qy_magnitude_imp"],
        "+ career_yprr only": model_feats + ["career_yprr"],  # sanity: already in model_feats
    }

    # Focus on the interesting variants
    focus_variants = [
        "ba_yptpa", "ba_yprr_routes", "ba_45ypg",
        "qa_zscore_adj", "qa_ratio_scaled", "qa_log_magnitude",
        "qy_zscore_adj", "qy_ratio_scaled", "qy_log_magnitude",
        "draft_age_feat",
    ]

    # Make sure magnitude imputed columns exist
    for mag_col in ["qa_magnitude", "qy_magnitude"]:
        imp_name = imp_cols.get(mag_col)
        if imp_name is None:
            mx = dynasty[mag_col].max()
            imp_name = f"{mag_col}_imp"
            dynasty[imp_name] = dynasty[mag_col].fillna(round(mx + 1, 2) if pd.notna(mx) else 0.0)
            imp_cols[mag_col] = imp_name

    # Header
    test_names = list(leak_tests.keys())
    print(f"  {'Variant':<25s}", end="")
    for tn in test_names:
        print(f" {tn:>22s}", end="")
    print()
    print("  " + "-" * (25 + 23 * len(test_names)))

    for col in focus_variants:
        imp_name = imp_cols[col]
        print(f"  {col:<25s}", end="")
        for test_name, control_feats in leak_tests.items():
            # Deduplicate control features
            ctrl = list(dict.fromkeys(control_feats))
            needed = [imp_name] + ctrl + ["tier_ordinal"]
            sub = dynasty[needed].dropna()
            if len(sub) < 30:
                print(f" {'N/A':>22s}", end="")
                continue
            rank_feat = rankdata(sub[imp_name].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid = rank_feat - X @ z
            sp_resid, _ = spearmanr(resid, rank_tier)
            print(f" {sp_resid:>+22.3f}", end="")
        print()

    # Summary interpretation
    print("\n  INTERPRETATION:")
    print("  - 'model_feats_only' = baseline residual (same as above)")
    print("  - '+ breakout_yptpa_mag' = after also controlling for YPTPA at breakout")
    print("  - '+ breakout_yprr_mag' = after also controlling for YPRR at breakout")
    print("  - '+ both_magnitudes' = after controlling for both magnitudes")
    print("  - If a quality-adjusted variant's residual drops to match the binary variant,")
    print("    the quality adjustment was just leaking magnitude info already in the model.")
    print("  - If it stays above the binary variant, the quality adjustment adds real signal.")

    # --- Correlation matrix between variants ---
    print("\n" + "=" * 80)
    print("CORRELATION BETWEEN VARIANTS (Spearman, pairwise complete)")
    print("=" * 80)
    print(f"\n  {'':>25s}", end="")
    short_names = {col: col.replace("ba_", "").replace("_feat", "")[:10] for col in all_cols}
    for col in all_cols:
        print(f" {short_names[col]:>10s}", end="")
    print()

    for col_a in all_cols:
        print(f"  {short_names[col_a]:>25s}", end="")
        for col_b in all_cols:
            both = dynasty[[col_a, col_b]].dropna()
            if len(both) > 10 and col_a != col_b:
                sp, _ = spearmanr(both[col_a], both[col_b])
                print(f" {sp:>+10.3f}", end="")
            elif col_a == col_b:
                print(f" {'1.000':>10s}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    # --- Summary ranking ---
    print("\n" + "=" * 80)
    print("SUMMARY RANKING")
    print("=" * 80)

    # Merge raw, imputed, and residual results
    summary = []
    for imp_res in imp_results:
        row = {"feature": imp_res["feature"]}
        row["spearman_imp"] = imp_res["spearman"]
        row["auc_imp"] = imp_res["auc"]
        row["drift"] = imp_res["drift"]
        row["coverage"] = imp_res["coverage"]
        # Find raw
        raw_match = [r for r in raw_results if r["feature"] == imp_res["feature"]]
        if raw_match:
            row["spearman_raw"] = raw_match[0]["spearman"]
            row["auc_raw"] = raw_match[0]["auc"]
            row["n_raw"] = raw_match[0]["n"]
        # Find residual
        res_match = [r for r in residual_results if r["feature"] == imp_res["feature"]]
        if res_match:
            row["residual_sp"] = res_match[0]["residual_spearman"]
        summary.append(row)

    # Composite score: |spearman_imp| * 0.3 + auc_imp * 0.3 + |residual_sp| * 0.2 + (1-drift) * 0.1 + coverage * 0.1
    for row in summary:
        sp = abs(row.get("spearman_imp", 0))
        auc = row.get("auc_imp", 0.5) - 0.5  # normalize AUC to 0-0.5 range
        res = abs(row.get("residual_sp", 0))
        drift = row.get("drift", 0.5) if pd.notna(row.get("drift")) else 0.5
        cov = row.get("coverage", 0)
        row["composite"] = round(sp * 0.3 + auc * 0.6 + res * 0.2 + (1 - drift) * 0.1 + cov * 0.1, 4)

    summary.sort(key=lambda x: x.get("composite", 0), reverse=True)

    print(f"\n  {'Rank':<6s} {'Variant':<25s} {'Sp(imp)':>8s} {'AUC':>8s} {'Resid':>8s} {'Drift':>8s} {'Cov':>6s} {'Score':>8s}")
    print("  " + "-" * 80)
    for i, row in enumerate(summary, 1):
        drift_str = f"{row['drift']:.3f}" if pd.notna(row.get('drift')) else "N/A"
        res_str = f"{row['residual_sp']:+.3f}" if 'residual_sp' in row else "N/A"
        print(f"  {i:<6d} {row['feature']:<25s} {row.get('spearman_imp', 0):>+8.3f} "
              f"{row.get('auc_imp', 0):>8.3f} {res_str:>8s} {drift_str:>8s} "
              f"{row.get('coverage', 0):>6.1%} {row.get('composite', 0):>8.4f}")

    # Save full results
    summary_df = pd.DataFrame(summary)
    out_path = os.path.join(DATA_DIR, "breakout_age_variants_eval.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved evaluation to {out_path}")

    # Save per-player breakout ages for inspection
    player_cols = ["name", "draft_year", "computed_tier"] + all_cols
    player_out = dynasty[player_cols].copy()
    player_out.to_csv(os.path.join(DATA_DIR, "breakout_age_variants_by_player.csv"), index=False)
    print(f"Saved per-player breakout ages to breakout_age_variants_by_player.csv")


if __name__ == "__main__":
    main()
