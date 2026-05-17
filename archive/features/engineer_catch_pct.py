#!/usr/bin/env python3
"""
Engineer receiver-isolated catch percentage variants and evaluate them.

Three features:
  1. catch_pct_above_team: WR catch% minus team completion% (controls for QB/scheme)
  2. catch_pct_adot_adjusted: residual of catch% after regressing on avg_depth_of_target
  3. catch_pct_double_adjusted: residual of (catch% - team_comp%) after regressing on aDOT

Computes career, best2, and best-single-season versions of each.
Evaluates against raw career_caught_percent on all 5 validation layers.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viz")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}


def load_season_data():
    """Load all grades + team stats at the season level."""
    from aggregation.aggregate_college_stats import load_all_grades, normalize_name

    all_grades = load_all_grades(range(2016, 2026))

    # Team completion %
    team_stats_path = os.path.join(DATA_DIR, "team_pass_stats.csv")
    if os.path.exists(team_stats_path):
        team_stats = pd.read_csv(team_stats_path)
        team_comp_lookup = {}
        for _, row in team_stats.iterrows():
            if pd.notna(row.get("completions")) and pd.notna(row.get("pass_att")) and row["pass_att"] > 0:
                team_comp_lookup[(row["team_pff"], row["year"])] = row["completions"] / row["pass_att"] * 100
        print(f"Team completion % loaded: {len(team_comp_lookup)} team-seasons")
    else:
        team_comp_lookup = {}
        print("WARNING: team_pass_stats.csv not found — feature #1 will be skipped")

    return all_grades, team_comp_lookup


def compute_engineered_features(all_grades, team_comp_lookup, dynasty):
    """Compute all 3 engineered catch % features at career level."""
    from aggregation.aggregate_college_stats import normalize_name, get_player_seasons

    # First, fit the aDOT regression on ALL player-seasons
    all_cp = pd.to_numeric(all_grades["caught_percent"], errors="coerce")
    all_adot = pd.to_numeric(all_grades["avg_depth_of_target"], errors="coerce")
    mask = all_cp.notna() & all_adot.notna()
    adot_coef = np.polyfit(all_adot[mask].values, all_cp[mask].values, 1)
    print(f"aDOT regression: catch% = {adot_coef[0]:.2f} * aDOT + {adot_coef[1]:.2f}")

    results = []
    for _, row in dynasty.iterrows():
        name = row["name"]
        draft_year = row["draft_year"]
        seasons = get_player_seasons(all_grades, name, draft_year)

        if len(seasons) == 0:
            results.append({})
            continue

        games = seasons["player_game_count"].values
        total_games = games.sum()
        if total_games == 0:
            results.append({})
            continue

        # Per-season values
        cp_vals = pd.to_numeric(seasons["caught_percent"], errors="coerce")
        adot_vals = pd.to_numeric(seasons["avg_depth_of_target"], errors="coerce")
        team_names = seasons["team_name"].values
        grade_years = seasons["grade_year"].values

        res = {}

        # --- Feature 1: catch% above team completion% ---
        above_team = []
        weights_above = []
        for j, (cp, team, yr, g) in enumerate(zip(cp_vals, team_names, grade_years, games)):
            if pd.notna(cp):
                team_comp = team_comp_lookup.get((team, yr))
                if team_comp is not None:
                    above_team.append(cp - team_comp)
                    weights_above.append(g)

        if len(above_team) > 0:
            res["career_catch_pct_above_team"] = round(
                np.average(above_team, weights=weights_above), 2
            )
            # Best 2 seasons version
            if len(above_team) >= 2:
                grades = pd.to_numeric(seasons["grades_offense"], errors="coerce")
                valid_idx = [j for j in range(len(seasons)) if pd.notna(cp_vals.iloc[j])
                             and team_comp_lookup.get((team_names[j], grade_years[j])) is not None]
                if len(valid_idx) >= 2:
                    grade_vals = [(grades.iloc[j] if pd.notna(grades.iloc[j]) else 0, j) for j in valid_idx]
                    grade_vals.sort(reverse=True)
                    top2_idx = [idx for _, idx in grade_vals[:2]]
                    top2_above = [above_team[valid_idx.index(j)] for j in top2_idx if j in valid_idx]
                    top2_weights = [games[j] for j in top2_idx]
                    if top2_above:
                        res["best2_catch_pct_above_team"] = round(
                            np.average(top2_above, weights=top2_weights), 2
                        )

        # --- Feature 2: aDOT-adjusted catch% ---
        adot_adj = []
        weights_adot = []
        for j, (cp, adot, g) in enumerate(zip(cp_vals, adot_vals, games)):
            if pd.notna(cp) and pd.notna(adot):
                expected_cp = np.polyval(adot_coef, adot)
                adot_adj.append(cp - expected_cp)
                weights_adot.append(g)

        if len(adot_adj) > 0:
            res["career_catch_pct_adot_adj"] = round(
                np.average(adot_adj, weights=weights_adot), 2
            )
            # Best 2 seasons version
            if len(adot_adj) >= 2:
                grades = pd.to_numeric(seasons["grades_offense"], errors="coerce")
                valid_idx = [j for j in range(len(seasons))
                             if pd.notna(cp_vals.iloc[j]) and pd.notna(adot_vals.iloc[j])]
                if len(valid_idx) >= 2:
                    grade_vals = [(grades.iloc[j] if pd.notna(grades.iloc[j]) else 0, j) for j in valid_idx]
                    grade_vals.sort(reverse=True)
                    top2_idx = [idx for _, idx in grade_vals[:2]]
                    top2_adj = [adot_adj[valid_idx.index(j)] for j in top2_idx if j in valid_idx]
                    top2_weights = [games[j] for j in top2_idx]
                    if top2_adj:
                        res["best2_catch_pct_adot_adj"] = round(
                            np.average(top2_adj, weights=top2_weights), 2
                        )

        # --- Feature 3: double-adjusted (above team + aDOT) ---
        double_adj = []
        weights_double = []
        for j, (cp, adot, team, yr, g) in enumerate(
            zip(cp_vals, adot_vals, team_names, grade_years, games)
        ):
            if pd.notna(cp) and pd.notna(adot):
                team_comp = team_comp_lookup.get((team, yr))
                if team_comp is not None:
                    above = cp - team_comp
                    expected_above = np.polyval(adot_coef, adot) - team_comp
                    # Simpler: just subtract both adjustments
                    expected_cp = np.polyval(adot_coef, adot)
                    double_adj.append(cp - expected_cp - (team_comp - np.mean(list(team_comp_lookup.values()))))
                    weights_double.append(g)

        if len(double_adj) > 0:
            res["career_catch_pct_double_adj"] = round(
                np.average(double_adj, weights=weights_double), 2
            )

        results.append(res)

    return pd.DataFrame(results)


def evaluate_features(df, feature_cols):
    """Run Layer 1 evaluation on a set of features."""
    tier = df["tier_ordinal"].values
    hit = df["hit"].values

    print(f"\n{'Feature':<35s} {'Spearman':>10s} {'AUC':>8s} {'N':>6s}")
    print("-" * 65)

    results = []
    for col in feature_cols:
        valid = df[[col, "tier_ordinal", "hit"]].dropna()
        if len(valid) < 30:
            print(f"{col:<35s} {'(too few)':>10s}")
            continue

        x = valid[col].values
        y = valid["tier_ordinal"].values
        y_hit = valid["hit"].values

        sp, _ = spearmanr(x, y)
        auc = roc_auc_score(y_hit, x)
        if auc < 0.5:
            auc = 1 - auc

        # Era stability
        years = df.loc[valid.index, "draft_year"].values
        early = years <= 2019
        late = years >= 2020
        sp_e, _ = spearmanr(x[early], y[early]) if early.sum() > 10 else (np.nan, np.nan)
        sp_l, _ = spearmanr(x[late], y[late]) if late.sum() > 10 else (np.nan, np.nan)
        drift = abs(sp_e - sp_l) if pd.notna(sp_e) and pd.notna(sp_l) else np.nan

        print(f"{col:<35s} {sp:>+10.3f} {auc:>8.3f} {len(valid):>6d}  drift={drift:.3f}" if pd.notna(drift)
              else f"{col:<35s} {sp:>+10.3f} {auc:>8.3f} {len(valid):>6d}")

        results.append({
            "feature": col, "spearman": sp, "auc": auc, "n": len(valid),
            "drift": drift, "sp_early": sp_e, "sp_late": sp_l,
        })

    return pd.DataFrame(results)


# =====================================================================
# Main
# =====================================================================
print("Loading data...")
all_grades, team_comp_lookup = load_season_data()

dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
dynasty["tier_ordinal"] = dynasty["computed_tier"].map(TIER_ORDER)
dynasty = dynasty.dropna(subset=["tier_ordinal"]).copy()
dynasty["tier_ordinal"] = dynasty["tier_ordinal"].astype(int)
dynasty["hit"] = (dynasty["tier_ordinal"] >= 3).astype(int)

print("Computing engineered features...")
eng_df = compute_engineered_features(all_grades, team_comp_lookup, dynasty)
dynasty = pd.concat([dynasty.reset_index(drop=True), eng_df], axis=1)

# =====================================================================
# Evaluate
# =====================================================================
print("\n" + "=" * 70)
print("EVALUATION: Raw vs Engineered Catch %")
print("=" * 70)

baseline_cols = ["career_caught_percent", "best2_caught_percent"]
engineered_cols = [c for c in eng_df.columns if eng_df[c].notna().sum() > 30]
all_cols = baseline_cols + sorted(engineered_cols)

eval_results = evaluate_features(dynasty, all_cols)

# =====================================================================
# Residual analysis: do engineered features add signal beyond raw?
# =====================================================================
print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

from scipy.stats import rankdata

model_feats = ["career_targeted_qb_rating", "career_yprr", "best2_contested_catch_rate",
               "career_avoided_tackles_pg", "breakout_age", "draft_capital"]

# Impute breakout_age
mx = dynasty["breakout_age"].max()
dynasty["breakout_age"] = dynasty["breakout_age"].fillna(round(mx + 1, 2))

for col in all_cols:
    sub = dynasty[[col] + model_feats + ["tier_ordinal"]].dropna()
    if len(sub) < 50:
        continue

    rank_feat = rankdata(sub[col].values)
    rank_tier = rankdata(sub["tier_ordinal"].values)

    # Residual after removing QBR + YPRR
    rank_qbr = rankdata(sub["career_targeted_qb_rating"].values)
    rank_yprr = rankdata(sub["career_yprr"].values)
    X = np.column_stack([rank_qbr, rank_yprr, np.ones(len(sub))])
    z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
    resid = rank_feat - X @ z
    sp_resid, _ = spearmanr(resid, rank_tier)

    # Residual after removing ALL other model features
    all_ranks = np.column_stack([rankdata(sub[f].values) for f in model_feats])
    X_all = np.column_stack([all_ranks, np.ones(len(sub))])
    z_all = np.linalg.lstsq(X_all, rank_feat, rcond=None)[0]
    resid_all = rank_feat - X_all @ z_all
    sp_resid_all, _ = spearmanr(resid_all, rank_tier)

    print(f"  {col:<35s} after QBR+YPRR: {sp_resid:+.3f}  after ALL: {sp_resid_all:+.3f}")

# =====================================================================
# Save results
# =====================================================================
if len(eval_results) > 0:
    out_path = os.path.join(DATA_DIR, "catch_pct_engineering_eval.csv")
    eval_results.to_csv(out_path, index=False)
    print(f"\nSaved evaluation to {out_path}")
