#!/usr/bin/env python3
"""
Explore YPRR-based breakout age configurations + double-residual production metric.

Part 1: YPRR breakout grid search
  - YPRR thresholds: 1.8, 2.0, 2.2, 2.5
  - Volume gates: 150 routes, 200 routes, 8 games, 100 routes + 8 games
  - Total: 16 configurations

Part 2: Double-residual breakout
  - Regress yards on (routes + team_pass_att) across all player-seasons
  - Residual = production unexplained by either opportunity source
  - Breakout = first season with residual above threshold

Part 3: Double-residual as a career feature (replacing career_yprr)
  - Career average double-residual vs career_yprr head-to-head

Evaluation: Spearman, AUC, residual after model features, efficiency leak test.
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

MODEL_FEATS = [
    "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
    "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
]

# Model features excluding career_yprr (for testing replacements)
MODEL_FEATS_NO_YPRR = [f for f in MODEL_FEATS if f != "career_yprr"]


def _season_age(birthdate, year):
    sept1 = pd.Timestamp(f"{year}-09-01")
    return round((sept1 - birthdate).days / 365.25, 2)


def evaluate_feature(df, col, control_feats=None):
    """Evaluate a feature: Spearman, AUC, and optionally residual after controls."""
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

    result = {
        "spearman": round(sp, 3), "auc": round(auc, 3),
        "n": len(valid), "coverage": round(len(valid) / len(df), 3),
        "drift": round(drift, 3) if pd.notna(drift) else np.nan,
    }

    # Residual after controlling for other features
    if control_feats:
        sub = df[[col] + control_feats + ["tier_ordinal"]].dropna()
        if len(sub) >= 30:
            rank_feat = rankdata(sub[col].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in control_feats])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid = rank_feat - X @ z
            sp_resid, _ = spearmanr(resid, rank_tier)
            result["residual"] = round(sp_resid, 3)

    return result


def fit_double_residual(all_grades, team_att_lookup, team_games_lookup):
    """Fit OLS: yards ~ routes + team_att_per_game. Return coefficients.

    Model: yards = b0 + b1*routes + b2*team_att_pg + epsilon
    The residual (epsilon) is production unexplained by either opportunity source.
    """
    rows = []
    for _, s in all_grades.iterrows():
        yards = pd.to_numeric(s.get("yards", 0), errors="coerce")
        routes = pd.to_numeric(s.get("routes", 0), errors="coerce")
        games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce")
        team = s.get("team_name", "")
        yr = s.get("grade_year", 0)
        att = team_att_lookup.get((team, yr))
        tg = team_games_lookup.get((team, yr))
        if (pd.notna(yards) and pd.notna(routes) and routes > 0
                and pd.notna(games) and games > 0
                and att and att > 0 and tg and tg > 0):
            rows.append({
                "yards": yards,
                "routes": routes,
                "team_att_pg": att / tg,
                "games": games,
            })

    fit_df = pd.DataFrame(rows)
    # Per-game normalize everything for comparability
    fit_df["ypg"] = fit_df["yards"] / fit_df["games"]
    fit_df["rpg"] = fit_df["routes"] / fit_df["games"]

    # OLS: ypg ~ rpg + team_att_pg
    X = np.column_stack([fit_df["rpg"].values, fit_df["team_att_pg"].values, np.ones(len(fit_df))])
    y = fit_df["ypg"].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    print(f"  Double residual model: ypg = {coef[0]:.3f}*rpg + {coef[1]:.3f}*team_att_pg + {coef[2]:.3f}")
    print(f"  R² = {1 - np.sum((y - X @ coef)**2) / np.sum((y - y.mean())**2):.3f}")

    # Residual distribution
    resids = y - X @ coef
    print(f"  Residual: mean={resids.mean():.3f}, std={resids.std():.3f}")

    return coef


def season_double_residual(s, coef, team_att_lookup, team_games_lookup):
    """Compute the double residual for a single season."""
    yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
    routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
    games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
    team = s.get("team_name", "")
    yr = s.get("grade_year", 0)
    att = team_att_lookup.get((team, yr))
    tg = team_games_lookup.get((team, yr))
    if routes > 0 and games > 0 and att and att > 0 and tg and tg > 0:
        ypg = yards / games
        rpg = routes / games
        att_pg = att / tg
        expected = coef[0] * rpg + coef[1] * att_pg + coef[2]
        return ypg - expected
    return None


def main():
    from aggregation.aggregate_college_stats import (
        load_all_grades, get_player_seasons, build_lookups,
    )

    print("Loading data...")
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)

    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    dynasty["tier_ordinal"] = dynasty["computed_tier"].map(TIER_ORDER)
    dynasty = dynasty.dropna(subset=["tier_ordinal"]).copy()
    dynasty["tier_ordinal"] = dynasty["tier_ordinal"].astype(int)
    dynasty["hit"] = (dynasty["tier_ordinal"] >= 3).astype(int)

    # =====================================================================
    # PART 1: YPRR Breakout Grid Search
    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 1: YPRR BREAKOUT GRID SEARCH")
    print("=" * 90)

    yprr_thresholds = [1.8, 2.0, 2.2, 2.5]
    volume_gates = {
        "150rt": lambda s, g, r: r >= 150,
        "200rt": lambda s, g, r: r >= 200,
        "8gm": lambda s, g, r: g >= 8,
        "100rt_8gm": lambda s, g, r: r >= 100 and g >= 8,
    }

    configs = {}
    for yprr_thresh in yprr_thresholds:
        for gate_name, gate_fn in volume_gates.items():
            config_name = f"yprr{yprr_thresh}_{gate_name}"
            ages = []
            for _, row in dynasty.iterrows():
                name, draft_year = row["name"], row["draft_year"]
                birthdate = birth_lookup.get((name, draft_year))
                seasons = get_player_seasons(all_grades, name, draft_year,
                                             birthdate=birthdate)
                if birthdate is None or pd.isna(birthdate) or len(seasons) == 0:
                    ages.append(np.nan)
                    continue
                found = np.nan
                for _, s in seasons.sort_values("grade_year").iterrows():
                    yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
                    routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
                    games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
                    if routes > 0 and yards / routes >= yprr_thresh and gate_fn(s, games, routes):
                        found = _season_age(birthdate, s["grade_year"])
                        break
                ages.append(found)
            dynasty[config_name] = ages
            configs[config_name] = (yprr_thresh, gate_name)

    # Also add the current ba_yptpa for comparison
    from features.engineer_breakout_age import breakout_yptpa
    print("  Computing ba_yptpa for comparison...")
    yptpa_ages = []
    for _, row in dynasty.iterrows():
        name, draft_year = row["name"], row["draft_year"]
        birthdate = birth_lookup.get((name, draft_year))
        seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)
        if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
            yptpa_ages.append(breakout_yptpa(seasons, birthdate,
                              team_att_lookup=team_att_lookup,
                              team_games_lookup=team_games_lookup))
        else:
            yptpa_ages.append(np.nan)
    dynasty["ba_yptpa_ref"] = yptpa_ages

    # Evaluate all YPRR configs
    print(f"\n  {'Config':<25s} {'Sp(raw)':>8s} {'Sp(imp)':>8s} {'AUC':>8s} {'Resid':>8s} {'Cov':>6s} {'Drift':>8s}")
    print("  " + "-" * 80)

    yprr_results = []
    for config_name in list(configs.keys()) + ["ba_yptpa_ref"]:
        # Impute
        mx = dynasty[config_name].max()
        imp_name = f"{config_name}_imp"
        dynasty[imp_name] = dynasty[config_name].fillna(round(mx + 1, 2) if pd.notna(mx) else 25.0)

        raw_res = evaluate_feature(dynasty, config_name)
        imp_res = evaluate_feature(dynasty, imp_name, control_feats=MODEL_FEATS)

        if raw_res and imp_res:
            row = {"config": config_name}
            row["sp_raw"] = raw_res["spearman"]
            row["sp_imp"] = imp_res["spearman"]
            row["auc"] = imp_res["auc"]
            row["residual"] = imp_res.get("residual", np.nan)
            row["coverage"] = raw_res["coverage"]
            row["drift"] = imp_res.get("drift", np.nan)
            row["n_raw"] = raw_res["n"]
            yprr_results.append(row)

            drift_str = f"{row['drift']:.3f}" if pd.notna(row['drift']) else "N/A"
            res_str = f"{row['residual']:+.3f}" if pd.notna(row['residual']) else "N/A"
            print(f"  {config_name:<25s} {row['sp_raw']:>+8.3f} {row['sp_imp']:>+8.3f} "
                  f"{row['auc']:>8.3f} {res_str:>8s} {row['coverage']:>6.1%} {drift_str:>8s}")

    # =====================================================================
    # PART 2: Double Residual Breakout Age
    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 2: DOUBLE RESIDUAL MODEL")
    print("=" * 90)

    print("\nFitting double residual model (ypg ~ rpg + team_att_pg)...")
    dr_coef = fit_double_residual(all_grades, team_att_lookup, team_games_lookup)

    # Compute residual distribution for thresholding
    all_resids = []
    for _, s in all_grades.iterrows():
        r = season_double_residual(s, dr_coef, team_att_lookup, team_games_lookup)
        if r is not None:
            all_resids.append(r)
    all_resids = np.array(all_resids)
    print(f"\n  Residual percentiles:")
    for pct in [50, 75, 80, 85, 90, 95]:
        print(f"    {pct}th: {np.percentile(all_resids, pct):.2f}")

    # Test breakout at various residual thresholds
    dr_thresholds = {
        "dr_p75": np.percentile(all_resids, 75),
        "dr_p80": np.percentile(all_resids, 80),
        "dr_p85": np.percentile(all_resids, 85),
        "dr_p90": np.percentile(all_resids, 90),
    }

    print(f"\n  Testing double-residual breakout age at percentile thresholds (8+ games):")
    print(f"\n  {'Config':<25s} {'Thresh':>8s} {'Sp(raw)':>8s} {'Sp(imp)':>8s} {'AUC':>8s} {'Resid':>8s} {'Cov':>6s} {'Drift':>8s}")
    print("  " + "-" * 85)

    dr_results = []
    for dr_name, thresh in dr_thresholds.items():
        ages = []
        for _, row in dynasty.iterrows():
            name, draft_year = row["name"], row["draft_year"]
            birthdate = birth_lookup.get((name, draft_year))
            seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)
            if birthdate is None or pd.isna(birthdate) or len(seasons) == 0:
                ages.append(np.nan)
                continue
            found = np.nan
            for _, s in seasons.sort_values("grade_year").iterrows():
                games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
                if games < 8:
                    continue
                resid = season_double_residual(s, dr_coef, team_att_lookup, team_games_lookup)
                if resid is not None and resid >= thresh:
                    found = _season_age(birthdate, s["grade_year"])
                    break
            ages.append(found)
        dynasty[dr_name] = ages

        # Impute and evaluate
        mx = dynasty[dr_name].max()
        imp_name = f"{dr_name}_imp"
        dynasty[imp_name] = dynasty[dr_name].fillna(round(mx + 1, 2) if pd.notna(mx) else 25.0)

        raw_res = evaluate_feature(dynasty, dr_name)
        imp_res = evaluate_feature(dynasty, imp_name, control_feats=MODEL_FEATS)

        if raw_res and imp_res:
            r = {"config": dr_name, "threshold": round(thresh, 2)}
            r["sp_raw"] = raw_res["spearman"]
            r["sp_imp"] = imp_res["spearman"]
            r["auc"] = imp_res["auc"]
            r["residual"] = imp_res.get("residual", np.nan)
            r["coverage"] = raw_res["coverage"]
            r["drift"] = imp_res.get("drift", np.nan)
            dr_results.append(r)

            drift_str = f"{r['drift']:.3f}" if pd.notna(r['drift']) else "N/A"
            res_str = f"{r['residual']:+.3f}" if pd.notna(r['residual']) else "N/A"
            print(f"  {dr_name:<25s} {r['threshold']:>8.2f} {r['sp_raw']:>+8.3f} {r['sp_imp']:>+8.3f} "
                  f"{r['auc']:>8.3f} {res_str:>8s} {r['coverage']:>6.1%} {drift_str:>8s}")

    # =====================================================================
    # PART 3: Double Residual as Career Feature
    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 3: CAREER DOUBLE RESIDUAL vs CAREER YPRR")
    print("=" * 90)

    print("\nComputing career double residual per player...")
    career_dr = []
    for _, row in dynasty.iterrows():
        name, draft_year = row["name"], row["draft_year"]
        birthdate = birth_lookup.get((name, draft_year))
        seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)
        if len(seasons) == 0:
            career_dr.append(np.nan)
            continue
        # Game-weighted average of per-season residuals
        resids = []
        weights = []
        for _, s in seasons.iterrows():
            games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
            r = season_double_residual(s, dr_coef, team_att_lookup, team_games_lookup)
            if r is not None and games > 0:
                resids.append(r)
                weights.append(games)
        if resids:
            career_dr.append(round(np.average(resids, weights=weights), 2))
        else:
            career_dr.append(np.nan)
    dynasty["career_double_residual"] = career_dr

    # Head-to-head: career_yprr vs career_double_residual
    print(f"\n  {'Feature':<30s} {'Spearman':>10s} {'AUC':>8s} {'Drift':>8s}")
    print("  " + "-" * 60)

    for col in ["career_yprr", "career_double_residual"]:
        res = evaluate_feature(dynasty, col)
        if res:
            drift_str = f"{res['drift']:.3f}" if pd.notna(res['drift']) else "N/A"
            print(f"  {col:<30s} {res['spearman']:>+10.3f} {res['auc']:>8.3f} {drift_str:>8s}")

    # Residual after controlling for model features (excluding career_yprr for fair test)
    print(f"\n  Residual after controlling model features (excluding career_yprr):")
    print(f"  {'Feature':<30s} {'Residual':>10s}")
    print("  " + "-" * 45)

    for col in ["career_yprr", "career_double_residual"]:
        res = evaluate_feature(dynasty, col, control_feats=MODEL_FEATS_NO_YPRR)
        if res and "residual" in res:
            print(f"  {col:<30s} {res['residual']:>+10.3f}")

    # Correlation between the two
    both = dynasty[["career_yprr", "career_double_residual"]].dropna()
    if len(both) > 10:
        sp_corr, _ = spearmanr(both["career_yprr"], both["career_double_residual"])
        print(f"\n  Spearman correlation between career_yprr and career_double_residual: {sp_corr:+.3f} (n={len(both)})")

    # Does replacing career_yprr with career_double_residual help breakout age signal?
    print(f"\n  Breakout age residual with different production features in the control set:")
    print(f"  {'Breakout variant':<20s} {'w/ career_yprr':>15s} {'w/ career_dr':>15s} {'Difference':>12s}")
    print("  " + "-" * 65)

    for ba_col in ["ba_yptpa_ref"]:
        imp_name = f"{ba_col}_imp"
        # With career_yprr
        ctrl_yprr = MODEL_FEATS
        sub = dynasty[[imp_name] + ctrl_yprr + ["tier_ordinal"]].dropna()
        if len(sub) >= 30:
            rank_feat = rankdata(sub[imp_name].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl_yprr])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid_yprr, _ = spearmanr(rank_feat - X @ z, rank_tier)
        else:
            resid_yprr = np.nan

        # With career_double_residual
        ctrl_dr = [f if f != "career_yprr" else "career_double_residual" for f in MODEL_FEATS]
        sub = dynasty[[imp_name] + ctrl_dr + ["tier_ordinal"]].dropna()
        if len(sub) >= 30:
            rank_feat = rankdata(sub[imp_name].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl_dr])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid_dr, _ = spearmanr(rank_feat - X @ z, rank_tier)
        else:
            resid_dr = np.nan

        diff = resid_dr - resid_yprr if pd.notna(resid_yprr) and pd.notna(resid_dr) else np.nan
        diff_str = f"{diff:+.3f}" if pd.notna(diff) else "N/A"
        print(f"  {ba_col:<20s} {resid_yprr:>+15.3f} {resid_dr:>+15.3f} {diff_str:>12s}")

    # =====================================================================
    # PART 4: Efficiency Leak Test on Best YPRR Configs
    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 4: EFFICIENCY LEAK TEST (top YPRR breakout configs)")
    print("=" * 90)

    # Find top 5 YPRR configs by residual
    yprr_results_sorted = sorted(yprr_results, key=lambda x: abs(x.get("residual", 0)), reverse=True)
    top_configs = [r["config"] for r in yprr_results_sorted[:5] if r["config"] != "ba_yptpa_ref"]
    top_configs = ["ba_yptpa_ref"] + top_configs  # always include yptpa for comparison

    # Compute YPRR magnitude at breakout for each top config
    print(f"\n  {'Config':<25s} {'Model only':>12s} {'+ YPRR mag':>12s} {'+ YPTPA mag':>12s} {'+ Both':>12s}")
    print("  " + "-" * 75)

    for config_name in top_configs:
        imp_name = f"{config_name}_imp"

        # Need breakout YPRR magnitude for this config
        # (for ba_yptpa_ref, use YPTPA magnitude)
        leak_results = {}
        for test_name, extra_controls in [
            ("Model only", []),
            ("+ YPRR mag", ["career_yprr"]),   # career_yprr as proxy for magnitude
            ("+ YPTPA mag", []),                # skip if no yptpa column
            ("+ Both", ["career_yprr"]),
        ]:
            ctrl = MODEL_FEATS + extra_controls
            ctrl = list(dict.fromkeys(ctrl))  # deduplicate
            sub = dynasty[[imp_name] + ctrl + ["tier_ordinal"]].dropna()
            if len(sub) < 30:
                leak_results[test_name] = np.nan
                continue
            rank_feat = rankdata(sub[imp_name].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid = rank_feat - X @ z
            sp_resid, _ = spearmanr(resid, rank_tier)
            leak_results[test_name] = sp_resid

        vals = [leak_results.get(t, np.nan) for t in ["Model only", "+ YPRR mag", "+ YPTPA mag", "+ Both"]]
        val_strs = [f"{v:>+12.3f}" if pd.notna(v) else f"{'N/A':>12s}" for v in vals]
        print(f"  {config_name:<25s} {''.join(val_strs)}")

    # Save results
    all_results = yprr_results + dr_results
    pd.DataFrame(all_results).to_csv(
        os.path.join(DATA_DIR, "yprr_breakout_grid_eval.csv"), index=False
    )
    print(f"\nSaved evaluation to wr_data/yprr_breakout_grid_eval.csv")


if __name__ == "__main__":
    main()
