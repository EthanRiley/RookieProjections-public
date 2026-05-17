#!/usr/bin/env python3
"""
2027 Draft Class Lookahead — Sophomore WR class of 2026.

Identifies WRs who will be sophomores or juniors in 2026 and could declare
for the 2027 NFL Draft. Includes:
  - Players who first appeared in PFF data in 2024 (sophomores in 2025, juniors in 2026)
  - Players who first appeared in PFF data in 2025 (freshmen in 2025, sophomores in 2026)

Excludes players already in the 2026 draft class predictions.

For multi-season players, uses proper aggregation (best2, career stats, peak-gated
features with graduated age adjustments) matching the production model pipeline.
For single-season players, uses that season with appropriate age adjustment.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

from aggregation.aggregate_college_stats import (
    load_all_grades,
    normalize_name,
    aggregate_seasons,
    best2_stats,
    P5_TEAMS,
    GRADUATED_ADJ,
    fit_adot_regression,
    PEAK_GATED_QUALITY_GATE,
    build_lookups,
    compute_pg_yprr_graduated,
    compute_pg_catch_pct_adot_adj_graduated,
    get_player_seasons,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")
ROUTE_MIN = 195  # Relaxed from 200 for freshmen

# Manual exclusions: players whose first PFF appearance is 2024/2025 but are NOT
# true freshmen/sophomores (e.g. seniors who transferred or only got late playing time).
# Format: normalized player name
MANUAL_EXCLUSIONS = {
    normalize_name("Cameron Dorner"),      # Senior, N Texas
    normalize_name("Jmariyae Robinson"),   # Junior, Missouri State — not getting drafted
    normalize_name("Landon Ellis"),        # Junior, James Madison — not getting drafted
    normalize_name("Caleb Hawkins"),       # RB (HB), not a WR
    normalize_name("Jacory Thomas"),       # Junior
    normalize_name("DeAree Rogers"),       # Junior
}

# Manual class year overrides: players whose PFF debut year doesn't match their
# actual class year (e.g. redshirt players whose first PFF data is a year late).
# Format: normalized_name -> actual first_year_of_college (determines age adjustment)
# A player who enrolled in 2023 but first appeared in PFF in 2025 should map to 2023.
CLASS_YEAR_OVERRIDES = {
    normalize_name("Charlie Becker"): 2024,   # RS freshman in 2024, sophomore in 2025
}

# --- Load all grades ---
print("Loading grades data...")
all_grades = load_all_grades(year_range=range(2016, 2026))
adot_coef = fit_adot_regression(all_grades)

# --- Identify 2027 class candidates ---
# Players whose first PFF appearance was 2024 or 2025 (at most 2 seasons of data)
print("Identifying 2027 draft class candidates...")
player_first_year = all_grades.groupby("_join_key")["grade_year"].min()
candidate_keys = set(player_first_year[player_first_year.isin([2024, 2025])].index)

# Filter to WR position only (PFF receiving grades include HB, TE, etc.)
all_grades = all_grades[all_grades["position"] == "WR"]

# Count seasons per player
player_seasons = all_grades[all_grades["_join_key"].isin(candidate_keys)].copy()

# Exclude players already in the 2026 draft class
draft_2026_path = os.path.join(DATA_DIR, "outputs", "prospect_predictions_2026.csv")
if os.path.exists(draft_2026_path):
    draft_2026 = pd.read_csv(draft_2026_path)
    draft_2026_keys = set(draft_2026["name"].apply(normalize_name))
    before = len(candidate_keys)
    candidate_keys -= draft_2026_keys
    print(f"Excluded {before - len(candidate_keys)} players already in 2026 draft class")

# Also exclude players in 2025 and 2024 draft classes
for yr in [2024, 2025]:
    path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_{yr}.csv")
    if os.path.exists(path):
        drafted = pd.read_csv(path)
        drafted_keys = set(drafted["name"].apply(normalize_name))
        candidate_keys -= drafted_keys

# Exclude players in holdout (already drafted 2022-2024)
holdout_path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_v12.csv")
if os.path.exists(holdout_path):
    holdout = pd.read_csv(holdout_path)
    holdout_keys = set(holdout["name"].apply(normalize_name))
    candidate_keys -= holdout_keys

# Also exclude from master dynasty value file (all historically drafted WRs)
master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
master_keys = set(master["name"].apply(normalize_name))
candidate_keys -= master_keys

# Manual exclusions (upperclassmen misidentified as freshmen/sophomores)
before = len(candidate_keys)
candidate_keys -= MANUAL_EXCLUSIONS
if before - len(candidate_keys) > 0:
    print(f"Excluded {before - len(candidate_keys)} manually flagged upperclassmen")

print(f"Total 2027 class candidates: {len(candidate_keys)}")

# --- Aggregate each candidate using the proper pipeline ---
print("Aggregating candidate features...")

NUMERIC_COLS = [
    "routes", "yards", "targets", "receptions", "player_game_count",
    "grades_offense", "grades_pass_route", "caught_percent",
    "avg_depth_of_target", "contested_catch_rate", "contested_targets",
    "contested_receptions", "avoided_tackles", "yprr",
    "yards_after_catch_per_reception", "drop_rate", "first_downs",
    "touchdowns", "slot_rate", "wide_rate",
]

rows = []
for key in sorted(candidate_keys):
    seasons = all_grades[all_grades["_join_key"] == key].copy()
    if len(seasons) == 0:
        continue

    for col in NUMERIC_COLS:
        if col in seasons.columns:
            seasons[col] = pd.to_numeric(seasons[col], errors="coerce")

    # P5 filter: must have at least one P5 season
    if not seasons["team_name"].isin(P5_TEAMS).any():
        continue

    # Check route minimum: need at least one season with ROUTE_MIN+ routes
    season_routes = seasons["routes"].fillna(0)
    if season_routes.max() < ROUTE_MIN:
        continue

    player_name = seasons.iloc[0]["player"]
    first_year = int(seasons["grade_year"].min())
    n_seasons = len(seasons)
    latest_team = seasons.sort_values("grade_year").iloc[-1]["team_name"]

    # Apply class year override if available
    actual_start = CLASS_YEAR_OVERRIDES.get(key, first_year)

    # Determine class year label based on actual enrollment year
    # 2025 season: if enrolled 2025 -> FR, 2024 -> SO, 2023 -> JR
    years_in_school_2025 = 2025 - actual_start + 1
    if years_in_school_2025 == 1:
        class_label = "FR (2025)"
    elif years_in_school_2025 == 2:
        class_label = "SO (2024-25)"
    elif years_in_school_2025 == 3:
        class_label = "JR (2023-25)"
    else:
        class_label = f"SR ({actual_start}-25)"

    # Filter to eligible seasons (ROUTE_MIN+ routes)
    eligible_seasons = seasons[seasons["routes"] >= ROUTE_MIN]
    if len(eligible_seasons) == 0:
        eligible_seasons = seasons  # Fall back to all seasons

    # --- Compute features using proper pipeline logic ---

    # Best season by grade (for single-season or fallback)
    grades = pd.to_numeric(eligible_seasons["grades_offense"], errors="coerce")
    best_idx = grades.idxmax() if grades.notna().any() else eligible_seasons.index[0]
    best_season = eligible_seasons.loc[best_idx]

    # For multi-season players, compute best2 stats
    if len(eligible_seasons) >= 2:
        b2 = best2_stats(eligible_seasons, adot_coef=adot_coef)
        career = aggregate_seasons(eligible_seasons, prefix="career", adot_coef=adot_coef)
    else:
        b2 = aggregate_seasons(eligible_seasons, prefix="best2", adot_coef=adot_coef)
        career = aggregate_seasons(eligible_seasons, prefix="career", adot_coef=adot_coef)

    # YPRR: use best season raw YPRR
    best_yards = pd.to_numeric(best_season.get("yards", 0), errors="coerce") or 0
    best_routes = pd.to_numeric(best_season.get("routes", 0), errors="coerce") or 0
    raw_yprr = best_yards / best_routes if best_routes > 0 else np.nan

    # Age adjustment: estimate from actual class year
    # Freshmen get +25%, sophomores get +5%, juniors -20%, seniors -25%
    if years_in_school_2025 == 1:
        age_mult = 1.25  # freshman
    elif years_in_school_2025 == 2:
        age_mult = 1.05  # sophomore
    elif years_in_school_2025 == 3:
        age_mult = 0.80  # junior
    else:
        age_mult = 0.75  # senior

    adj_yprr = raw_yprr * age_mult if pd.notna(raw_yprr) else np.nan

    # For multi-season players, also compute peak adjusted YPRR across seasons
    if n_seasons >= 2:
        peak_adj_yprr = np.nan
        for _, s in eligible_seasons.iterrows():
            sy = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
            sr = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
            if sr == 0:
                continue
            s_yprr = sy / sr
            s_year = int(s["grade_year"])
            # Apply age adjustment based on actual year in school
            s_years_in = s_year - actual_start + 1
            if s_years_in == 1:
                s_mult = 1.25
            elif s_years_in == 2:
                s_mult = 1.05
            elif s_years_in == 3:
                s_mult = 0.80
            else:
                s_mult = 0.75
            s_adj = s_yprr * s_mult
            if pd.isna(peak_adj_yprr) or s_adj > peak_adj_yprr:
                peak_adj_yprr = s_adj
        adj_yprr = peak_adj_yprr if pd.notna(peak_adj_yprr) else adj_yprr

    # aDOT-adjusted catch%
    best_cp = pd.to_numeric(best_season.get("caught_percent"), errors="coerce")
    best_adot = pd.to_numeric(best_season.get("avg_depth_of_target"), errors="coerce")
    if pd.notna(best_cp) and pd.notna(best_adot):
        cpaa = best_cp - np.polyval(adot_coef, best_adot)
        cpaa_graduated = cpaa * age_mult
    else:
        cpaa = np.nan
        cpaa_graduated = np.nan

    # For multi-season: take peak CPAA across seasons
    if n_seasons >= 2:
        peak_cpaa = np.nan
        for _, s in eligible_seasons.iterrows():
            s_cp = pd.to_numeric(s.get("caught_percent"), errors="coerce")
            s_adot = pd.to_numeric(s.get("avg_depth_of_target"), errors="coerce")
            if pd.isna(s_cp) or pd.isna(s_adot):
                continue
            s_cpaa = s_cp - np.polyval(adot_coef, s_adot)
            s_year = int(s["grade_year"])
            s_years_in = s_year - actual_start + 1
            if s_years_in == 1:
                s_mult = 1.25
            elif s_years_in == 2:
                s_mult = 1.05
            elif s_years_in == 3:
                s_mult = 0.80
            else:
                s_mult = 0.75
            s_adj = s_cpaa * s_mult
            if pd.isna(peak_cpaa) or s_adj > peak_cpaa:
                peak_cpaa = s_adj
        cpaa_graduated = peak_cpaa if pd.notna(peak_cpaa) else cpaa_graduated

    # Contested targets (total across eligible seasons)
    total_ct = pd.to_numeric(eligible_seasons["contested_targets"], errors="coerce").fillna(0).sum()

    # CCR: use best2 if available, else best season
    # Small-sample filter: <10 CT → impute with 50th percentile of historical CCR
    CCR_MIN_CT = 10
    ccr = b2.get("best2_contested_catch_rate", np.nan)
    if pd.isna(ccr):
        ccr = pd.to_numeric(best_season.get("contested_catch_rate"), errors="coerce")
    if total_ct < CCR_MIN_CT:
        ccr = np.nan  # Will be imputed with p50 after historical distributions are computed

    # Avoided tackles per rec: use best2 if available
    # 20% relative discount for single-season players (inflated by small sample)
    at_per_rec = b2.get("best2_avoided_tackles_per_rec", np.nan)
    if pd.isna(at_per_rec):
        total_at = pd.to_numeric(eligible_seasons["avoided_tackles"], errors="coerce").fillna(0).sum()
        total_rec = pd.to_numeric(eligible_seasons["receptions"], errors="coerce").fillna(0).sum()
        at_per_rec = total_at / total_rec if total_rec > 0 else np.nan
    if n_seasons == 1 and pd.notna(at_per_rec):
        at_per_rec *= 0.80

    # Best season stats for display
    best_grade = pd.to_numeric(best_season.get("grades_offense"), errors="coerce")
    best_rte_grade = pd.to_numeric(best_season.get("grades_pass_route"), errors="coerce")
    best_cp_raw = pd.to_numeric(best_season.get("caught_percent"), errors="coerce")
    best_adot_raw = pd.to_numeric(best_season.get("avg_depth_of_target"), errors="coerce")
    best_slot = pd.to_numeric(best_season.get("slot_rate"), errors="coerce")
    best_wide = pd.to_numeric(best_season.get("wide_rate"), errors="coerce")

    # Per-game totals (across all eligible seasons)
    total_games = pd.to_numeric(eligible_seasons["player_game_count"], errors="coerce").fillna(0).sum()
    total_yards = pd.to_numeric(eligible_seasons["yards"], errors="coerce").fillna(0).sum()
    total_recs = pd.to_numeric(eligible_seasons["receptions"], errors="coerce").fillna(0).sum()
    total_tds = pd.to_numeric(eligible_seasons["touchdowns"], errors="coerce").fillna(0).sum()
    total_routes_all = pd.to_numeric(eligible_seasons["routes"], errors="coerce").fillna(0).sum()

    yards_pg = total_yards / total_games if total_games > 0 else np.nan
    recs_pg = total_recs / total_games if total_games > 0 else np.nan
    tds_pg = total_tds / total_games if total_games > 0 else np.nan

    # P5 flag (any P5 season)
    is_p5 = eligible_seasons["team_name"].isin(P5_TEAMS).any()

    # Career aDOT-adjusted catch% (from aggregate_seasons output)
    career_cpaa = career.get("career_catch_pct_adot_adj", np.nan)

    # Quality gate
    quality_gated = (grades >= PEAK_GATED_QUALITY_GATE).any()

    rows.append({
        "player": player_name,
        "team_name": latest_team,
        "class_label": class_label,
        "n_seasons": n_seasons,
        "is_p5": is_p5,
        "total_games": int(total_games),
        "total_routes": int(total_routes_all),
        "best_grades_offense": best_grade,
        "best_grades_pass_route": best_rte_grade,
        "raw_yprr": round(raw_yprr, 4) if pd.notna(raw_yprr) else np.nan,
        "adj_yprr": round(adj_yprr, 4) if pd.notna(adj_yprr) else np.nan,
        "caught_percent": best_cp_raw,
        "avg_depth_of_target": best_adot_raw,
        "catch_pct_adot_adj": round(cpaa, 2) if pd.notna(cpaa) else np.nan,
        "catch_pct_adot_adj_graduated": round(cpaa_graduated, 2) if pd.notna(cpaa_graduated) else np.nan,
        "career_catch_pct_adot_adj": round(career_cpaa, 2) if pd.notna(career_cpaa) else np.nan,
        "contested_catch_rate": ccr,
        "contested_targets": int(total_ct),
        "avoided_tackles_per_rec": round(at_per_rec, 4) if pd.notna(at_per_rec) else np.nan,
        "yards_pg": round(yards_pg, 1) if pd.notna(yards_pg) else np.nan,
        "receptions_pg": round(recs_pg, 1) if pd.notna(recs_pg) else np.nan,
        "touchdowns_pg": round(tds_pg, 2) if pd.notna(tds_pg) else np.nan,
        "slot_rate": best_slot,
        "wide_rate": best_wide,
        "quality_gated": quality_gated,
    })

eligible = pd.DataFrame(rows)
print(f"Candidates with {ROUTE_MIN}+ routes in at least one season: {len(eligible)}")

# --- Historical distributions for percentile context ---
print("Computing historical feature distributions from drafted WRs...")
birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)

hist_pg_yprr = []
hist_pg_cpaa = []
for _, row in master.iterrows():
    name = row["name"]
    dy = int(row["draft_year"])
    birthdate = birth_lookup.get((name, dy))
    seasons = get_player_seasons(all_grades, name, dy, birthdate=birthdate)
    if len(seasons) == 0:
        continue
    val = compute_pg_yprr_graduated(seasons, birthdate)
    if not np.isnan(val):
        hist_pg_yprr.append(val)
    val2 = compute_pg_catch_pct_adot_adj_graduated(seasons, birthdate, adot_coef=adot_coef)
    if not np.isnan(val2):
        hist_pg_cpaa.append(val2)

hist_yprr = pd.Series(hist_pg_yprr)
hist_cpaa = pd.Series(hist_pg_cpaa) if hist_pg_cpaa else None
hist_ccr = master["best2_contested_catch_rate"].dropna()
hist_at = master["best2_avoided_tackles_per_rec"].dropna()
print(f"Historical baselines: {len(hist_yprr)} YPRR, {len(hist_cpaa) if hist_cpaa is not None else 0} CPAA, "
      f"{len(hist_ccr)} CCR, {len(hist_at)} AT/R")

def pctile(val, series):
    if pd.isna(val) or len(series) == 0:
        return np.nan
    return round((series < val).mean() * 100, 0)

def zscore(val, series):
    if pd.isna(val) or len(series) == 0 or series.std() == 0:
        return np.nan
    return (val - series.mean()) / series.std()

# Impute CCR for small-sample players with 50th percentile of historical distribution
ccr_p50 = hist_ccr.median()
small_ccr_mask = eligible["contested_catch_rate"].isna()
eligible.loc[small_ccr_mask, "contested_catch_rate"] = ccr_p50
print(f"Imputed {small_ccr_mask.sum()} players with <10 CT to CCR p50 = {ccr_p50:.1f}%")

# Build catch_composite matching production model: z-avg(67% CPAA + 33% career CPAA)
# Fit z-score params on historical drafted WRs
hist_career_cpaa = master["career_catch_pct_adot_adj"].dropna() if "career_catch_pct_adot_adj" in master.columns else pd.Series(dtype=float)
cpaa_mean = pd.Series(hist_pg_cpaa).mean()
cpaa_std = pd.Series(hist_pg_cpaa).std()
career_cpaa_mean = hist_career_cpaa.mean()
career_cpaa_std = hist_career_cpaa.std()

CATCH_COMPOSITE_CPAA_WEIGHT = 0.67
CATCH_COMPOSITE_CAREER_WEIGHT = 0.33

def compute_catch_composite(pg_cpaa, career_cpaa_val):
    if pd.isna(pg_cpaa) or pd.isna(career_cpaa_val):
        return np.nan
    return (CATCH_COMPOSITE_CPAA_WEIGHT * (pg_cpaa - cpaa_mean) / cpaa_std
            + CATCH_COMPOSITE_CAREER_WEIGHT * (career_cpaa_val - career_cpaa_mean) / career_cpaa_std)

eligible["catch_composite"] = eligible.apply(
    lambda r: compute_catch_composite(r["catch_pct_adot_adj_graduated"], r["career_catch_pct_adot_adj"]),
    axis=1
)

# Build historical catch_composite for percentile context
hist_catch_composite = []
for _, row in master.dropna(subset=["career_catch_pct_adot_adj"]).iterrows():
    name = row["name"]
    dy = int(row["draft_year"])
    birthdate = birth_lookup.get((name, dy))
    seasons = get_player_seasons(all_grades, name, dy, birthdate=birthdate)
    if len(seasons) == 0:
        continue
    pg_val = compute_pg_catch_pct_adot_adj_graduated(seasons, birthdate, adot_coef=adot_coef)
    if not np.isnan(pg_val):
        cc = compute_catch_composite(pg_val, row["career_catch_pct_adot_adj"])
        if pd.notna(cc):
            hist_catch_composite.append(cc)
hist_cc = pd.Series(hist_catch_composite)
print(f"Historical catch_composite: {len(hist_cc)} values, mean={hist_cc.mean():.3f}, std={hist_cc.std():.3f}")

eligible["z_yprr"] = eligible["adj_yprr"].apply(lambda x: zscore(x, hist_yprr))
eligible["z_catch_composite"] = eligible["catch_composite"].apply(lambda x: zscore(x, hist_cc))
eligible["z_ccr"] = eligible["contested_catch_rate"].apply(lambda x: zscore(x, hist_ccr))
eligible["z_at"] = eligible["avoided_tackles_per_rec"].apply(lambda x: zscore(x, hist_at))

z_cols = ["z_yprr", "z_catch_composite", "z_ccr", "z_at"]
eligible["college_composite"] = eligible[z_cols].mean(axis=1)

# --- Sort and display ---
eligible = eligible.sort_values("college_composite", ascending=False).reset_index(drop=True)

print("\n" + "=" * 120)
print("2027 DRAFT CLASS LOOKAHEAD — Sophomore & Junior WRs (first appeared 2024 or 2025)")
print("Ranked by college composite = avg z-score of adj_yprr, catch_composite, CCR, avoided_tackles_per_rec")
print("YPRR/catch% include graduated age adjustment (FR +25%, SO +5%)")
print("=" * 120)

print(f"\n{'Rank':<5} {'Player':<25} {'Team':<12} {'Cl':>10} {'P5':>3} {'G':>3} {'Rte':>5} "
      f"{'Grade':>6} {'RteGr':>6} {'YPRR':>6} {'aYPRR':>6} "
      f"{'Ct%':>6} {'aDOT':>5} {'CComp':>6} "
      f"{'CCR':>6} {'CT':>3} {'AT/R':>5} "
      f"{'Y/G':>6} {'R/G':>5} {'TD/G':>5} "
      f"{'Comp':>6}")
print("-" * 175)

for i, (_, row) in enumerate(eligible.head(30).iterrows(), 1):
    p5 = "Y" if row["is_p5"] else "N"
    cc = row.get("catch_composite", np.nan)
    cc_str = f"{cc:>6.2f}" if pd.notna(cc) else "   N/A"
    print(f"{i:<5} {row['player']:<25} {row['team_name']:<12} {row['class_label']:>10} {p5:>3} "
          f"{int(row['total_games']):>3} {int(row['total_routes']):>5} "
          f"{row['best_grades_offense']:>6.1f} {row['best_grades_pass_route']:>6.1f} "
          f"{row['raw_yprr']:>6.2f} {row['adj_yprr']:>6.2f} "
          f"{row['caught_percent']:>6.1f} {row['avg_depth_of_target']:>5.1f} "
          f"{cc_str} "
          f"{row['contested_catch_rate']:>6.1f} {int(row['contested_targets']):>3} "
          f"{row['avoided_tackles_per_rec']:>5.2f} "
          f"{row['yards_pg']:>6.1f} {row['receptions_pg']:>5.1f} {row['touchdowns_pg']:>5.2f} "
          f"{row['college_composite']:>6.2f}")

# --- Percentile context ---
print("\n\nPERCENTILE CONTEXT (vs. historical drafted WRs 2018-2024)")
print(f"{'Rank':<5} {'Player':<25} {'Class':>10} {'aYPRR%':>7} {'CComp%':>7} {'CCR%':>7} {'AT/R%':>7}")
print("-" * 75)

for i, (_, row) in enumerate(eligible.head(20).iterrows(), 1):
    p_yprr = pctile(row["adj_yprr"], hist_yprr)
    p_cc = pctile(row.get("catch_composite", np.nan), hist_cc)
    p_ccr = pctile(row["contested_catch_rate"], hist_ccr)
    p_at = pctile(row["avoided_tackles_per_rec"], hist_at)
    print(f"{i:<5} {row['player']:<25} {row['class_label']:>10} "
          f"{p_yprr:>6.0f}% {p_cc:>6.0f}% {p_ccr:>6.0f}% {p_at:>6.0f}%")

# --- Quality-gated ---
qg = eligible[eligible["quality_gated"]].sort_values("college_composite", ascending=False)
print(f"\n\nQUALITY-GATED (grades_offense >= 80 in at least one season): {len(qg)} of {len(eligible)}")

# --- Save output ---
output_path = os.path.join(DATA_DIR, "outputs", "sophomore_2026_lookahead.csv")
eligible.to_csv(output_path, index=False)
print(f"\nSaved {len(eligible)} players to {output_path}")
