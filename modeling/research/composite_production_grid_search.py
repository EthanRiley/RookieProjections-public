#!/usr/bin/env python3
"""
Expansive grid search: composite production metrics to replace pg_yprr_graduated.

Goal: find a composite that has YPRR's predictive power with YPTPA's era stability.

Candidate raw production metrics (per-season, then aggregated):
  - YPRR (yards per route run)
  - YPTPA (yards per team pass attempt)
  - yards_per_game
  - yards_per_reception
  - first_downs_per_route
  - first_downs_per_game
  - touchdowns_per_game
  - targets_per_game
  - receptions_per_game
  - yac_per_route (yards after catch per route run)
  - target_share (targets per route)

Composite techniques:
  1. Weighted averages (z-score normalized, various weights)
  2. PCA (first principal component)
  3. Geometric mean (sqrt(A * B))
  4. Rank-averaged composites
  5. Individual peak-gated age-adjusted variants (for comparison)

All composites use peak-gated, age-adjusted selection (same as pg_yprr_graduated).
LOO evaluation on 2018-2021 training data, then holdout confirmation on best configs.
"""

import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

GRADUATED_ADJ = {
    (0, 19.5): 1.25,
    (19.5, 20.5): 1.05,
    (20.5, 21.5): 0.80,
    (21.5, 22.5): 0.75,
    (22.5, 99): 0.50,
}

PEAK_GATED_QUALITY_GATE = 80

P5_TEAMS = {
    'ALABAMA', 'ARKANSAS', 'AUBURN', 'FLORIDA', 'GEORGIA', 'KENTUCKY', 'LSU',
    'MISS STATE', 'MISSOURI', 'OLE MISS', 'S CAROLINA', 'TENNESSEE', 'TEXAS',
    'TEXAS A&M', 'OKLAHOMA', 'VANDERBILT',
    'ILLINOIS', 'INDIANA', 'IOWA', 'MARYLAND', 'MICHIGAN', 'MICH STATE',
    'MINNESOTA', 'NEBRASKA', 'NWESTERN', 'OHIO STATE', 'PENN STATE', 'PURDUE',
    'RUTGERS', 'WISCONSIN', 'UCLA', 'USC', 'OREGON', 'WASHINGTON',
    'ARIZONA', 'ARIZONA ST', 'BAYLOR', 'BYU', 'CINCINNATI', 'COLORADO',
    'HOUSTON', 'IOWA STATE', 'KANSAS', 'KANSAS ST', 'OKLAHOMA ST', 'TCU',
    'TEXAS TECH', 'UCF', 'UTAH', 'WVU',
    'BC', 'CLEMSON', 'DUKE', 'FL STATE', 'GEORGIA TECH', 'LOUISVILLE',
    'MIAMI', 'NC STATE', 'NORTH CAROLINA', 'PITT', 'SMU', 'SYRACUSE',
    'VIRGINIA', 'VA TECH', 'WAKE FOREST', 'STANFORD', 'CAL',
    'NOTRE DAME', 'OREGON ST', 'WASH STATE',
}


def get_age_mult(birthdate, year):
    """Get graduated age adjustment multiplier."""
    if birthdate is None or pd.isna(birthdate) or year is None or pd.isna(year):
        return 1.0
    sept1 = pd.Timestamp(f"{int(year)}-09-01")
    age = (sept1 - birthdate).days / 365.25
    for (lo, hi), mult in GRADUATED_ADJ.items():
        if lo <= age < hi:
            return mult
    return 1.0


def compute_season_production_stats(row, team_att_lookup, team_games_lookup):
    """Compute all production metrics for a single season."""
    stats = {}

    yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
    routes = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
    targets = pd.to_numeric(row.get("targets", 0), errors="coerce") or 0
    receptions = pd.to_numeric(row.get("receptions", 0), errors="coerce") or 0
    first_downs = pd.to_numeric(row.get("first_downs", 0), errors="coerce") or 0
    touchdowns = pd.to_numeric(row.get("touchdowns", 0), errors="coerce") or 0
    yac = pd.to_numeric(row.get("yards_after_catch", 0), errors="coerce") or 0
    games = pd.to_numeric(row.get("player_game_count", 0), errors="coerce") or 0

    team = row.get("team_name")
    yr = row.get("grade_year")
    team_att = team_att_lookup.get((team, yr), 0) if team_att_lookup else 0
    team_gm = team_games_lookup.get((team, yr), 0) if team_games_lookup else 0

    # Route-based rates
    if routes > 0:
        stats["yprr"] = yards / routes
        stats["first_downs_per_route"] = first_downs / routes
        stats["yac_per_route"] = yac / routes
        stats["target_share"] = targets / routes
    else:
        stats["yprr"] = 0
        stats["first_downs_per_route"] = 0
        stats["yac_per_route"] = 0
        stats["target_share"] = 0

    # Per-game rates
    if games > 0:
        stats["yards_per_game"] = yards / games
        stats["first_downs_per_game"] = first_downs / games
        stats["touchdowns_per_game"] = touchdowns / games
        stats["targets_per_game"] = targets / games
        stats["receptions_per_game"] = receptions / games
        stats["yac_per_game"] = yac / games
    else:
        for k in ["yards_per_game", "first_downs_per_game", "touchdowns_per_game",
                   "targets_per_game", "receptions_per_game", "yac_per_game"]:
            stats[k] = 0

    # Per-reception rates
    if receptions > 0:
        stats["yards_per_reception"] = yards / receptions
        stats["yac_per_reception"] = yac / receptions
    else:
        stats["yards_per_reception"] = 0
        stats["yac_per_reception"] = 0

    # Team-context rates
    if team_att and team_att > 0:
        stats["yptpa"] = yards / team_att
    else:
        stats["yptpa"] = 0

    # First downs per target (conversion efficiency)
    if targets > 0:
        stats["first_downs_per_target"] = first_downs / targets
    else:
        stats["first_downs_per_target"] = 0

    return stats


ALL_PRODUCTION_METRICS = [
    "yprr",
    "yptpa",
    "yards_per_game",
    "yards_per_reception",
    "first_downs_per_route",
    "first_downs_per_game",
    "first_downs_per_target",
    "touchdowns_per_game",
    "targets_per_game",
    "receptions_per_game",
    "yac_per_route",
    "yac_per_game",
    "yac_per_reception",
    "target_share",
]


def compute_pg_metric(seasons, birthdate, metric_name, team_att_lookup, team_games_lookup):
    """Peak-gated, age-adjusted version of any production metric.

    Same logic as pg_yprr_graduated: pick the season where the age-adjusted
    metric peaks among grade >= 80 seasons, fallback to best1.
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

    gated = eligible[grades >= PEAK_GATED_QUALITY_GATE]

    def _get_adj_val(row):
        stats = compute_season_production_stats(row, team_att_lookup, team_games_lookup)
        raw = stats.get(metric_name, 0)
        if raw == 0:
            return np.nan
        mult = get_age_mult(birthdate, row.get("grade_year"))
        return raw * mult

    search_set = gated if len(gated) > 0 else eligible.loc[[grades.idxmax()]]

    best_val = np.nan
    for _, row in search_set.iterrows():
        adj = _get_adj_val(row)
        if pd.notna(adj) and (np.isnan(best_val) or adj > best_val):
            best_val = adj

    return round(best_val, 4) if pd.notna(best_val) else np.nan


def aggregate_all_pg_metrics(all_grades, df, birth_lookup, team_att_lookup,
                              team_games_lookup):
    """Compute all peak-gated age-adjusted production metrics for every player."""
    from aggregation.aggregate_college_stats import get_player_seasons

    results = {m: [] for m in ALL_PRODUCTION_METRICS}

    for _, row in df.iterrows():
        name, draft_year = row["name"], row["draft_year"]
        birthdate = birth_lookup.get((name, draft_year)) if birth_lookup else None
        seasons = get_player_seasons(all_grades, name, draft_year,
                                      apply_exclusions=True, birthdate=birthdate)

        for metric in ALL_PRODUCTION_METRICS:
            val = compute_pg_metric(seasons, birthdate, metric,
                                     team_att_lookup, team_games_lookup)
            results[metric].append(val)

    for metric in ALL_PRODUCTION_METRICS:
        df[f"pg_{metric}"] = results[metric]

    return df


def _loo_ordinal_scores(d, features, years, n_tiers=6):
    """Leave-one-year-out ordinal logistic regression."""
    thresholds = list(range(1, n_tiers))
    n = len(d)
    cum_probs = np.zeros((n, len(thresholds)))
    player_indices = np.arange(n)
    mask_predicted = np.zeros(n, dtype=bool)
    y = d["tier_num"].values
    years_arr = d["draft_year"].values

    for yr in years:
        train_mask = years_arr != yr
        test_mask = years_arr == yr
        test_idx = player_indices[test_mask]
        if test_idx.sum() == 0:
            continue
        X_train = d.iloc[train_mask][features].values
        X_test = d.iloc[test_mask][features].values
        y_train = y[train_mask]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
        for ti, thresh in enumerate(thresholds):
            y_bin = (y_train >= thresh).astype(int)
            if y_bin.sum() < 2 or y_bin.sum() == len(y_bin):
                cum_probs[test_idx, ti] = y_bin.mean()
                continue
            lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
            lr.fit(X_tr_s, y_bin)
            cum_probs[test_idx, ti] = lr.predict_proba(X_te_s)[:, 1]
        mask_predicted[test_mask] = True

    if not mask_predicted.any():
        return {}

    idx = mask_predicted
    n_pred = idx.sum()
    cp = cum_probs[idx]
    for ti in range(len(thresholds) - 1, 0, -1):
        cp[:, ti - 1] = np.maximum(cp[:, ti - 1], cp[:, ti])
    tier_probs = np.zeros((n_pred, n_tiers))
    tier_probs[:, 0] = 1 - cp[:, 0]
    for k in range(1, n_tiers - 1):
        tier_probs[:, k] = cp[:, k - 1] - cp[:, k]
    tier_probs[:, n_tiers - 1] = cp[:, -1]
    tier_probs = np.clip(tier_probs, 1e-8, 1.0)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)

    y_pred = y[idx]
    ll = -np.mean(np.log(tier_probs[np.arange(n_pred), y_pred]))
    one_hot = np.zeros((n_pred, n_tiers))
    one_hot[np.arange(n_pred), y_pred] = 1
    brier = np.mean(np.sum((tier_probs - one_hot) ** 2, axis=1))

    results = {"log_loss": ll, "brier": brier}
    for ti, thresh in enumerate(thresholds):
        y_bin = (y_pred >= thresh).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            results[f"auc_{thresh}"] = roc_auc_score(y_bin, cp[:, ti])
        else:
            results[f"auc_{thresh}"] = np.nan
    return results


def compute_era_stability(df, col, years):
    """Spearman correlation with tier_num, split early/late, return drift."""
    from scipy.stats import spearmanr
    valid = df.dropna(subset=[col, "tier_num"])
    if len(valid) < 10:
        return np.nan, np.nan, np.nan

    mid = np.median(years)
    early = valid[valid["draft_year"] <= mid]
    late = valid[valid["draft_year"] > mid]

    if len(early) < 5 or len(late) < 5:
        return np.nan, np.nan, np.nan

    r_early = spearmanr(early[col], early["tier_num"])[0]
    r_late = spearmanr(late[col], late["tier_num"])[0]
    overall = spearmanr(valid[col], valid["tier_num"])[0]

    return overall, abs(r_late - r_early), (r_early, r_late)


def main():
    from aggregation.aggregate_college_stats import (
        load_all_grades, build_lookups, fit_adot_regression,
    )

    print("=" * 100)
    print("  COMPOSITE PRODUCTION METRIC GRID SEARCH")
    print("  Goal: YPRR's predictive power + YPTPA's era stability")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)

    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_num"] = df["computed_tier"].map(TIER_ORDER)
    df["draft_capital"] = np.maximum(10 - (10 / np.log(261)) * np.log(df["pick"] + 1), 0)

    # Step 1: Compute all peak-gated age-adjusted production metrics
    print("Computing peak-gated age-adjusted production metrics for all players...")
    df = aggregate_all_pg_metrics(all_grades, df, birth_lookup, team_att_lookup,
                                   team_games_lookup)

    pg_cols = [f"pg_{m}" for m in ALL_PRODUCTION_METRICS]
    available = [c for c in pg_cols if df[c].notna().sum() >= 50]
    print(f"\nAvailable metrics ({len(available)}):")
    for c in available:
        n_valid = df[c].notna().sum()
        print(f"  {c}: {n_valid} non-null")

    # Step 2: Era stability + univariate signal for each metric
    years = sorted(df["draft_year"].unique())
    print(f"\n{'=' * 100}")
    print(f"  INDIVIDUAL METRIC ANALYSIS")
    print(f"{'=' * 100}")
    print(f"  {'Metric':<35s} {'Spearman':>8s} {'Drift':>8s} {'Early':>8s} {'Late':>8s} {'N':>5s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    metric_stats = {}
    for col in available:
        overall, drift, splits = compute_era_stability(df, col, years)
        n = df[col].notna().sum()
        if splits is not None and not np.isnan(overall):
            print(f"  {col:<35s} {overall:>8.3f} {drift:>8.3f} {splits[0]:>8.3f} {splits[1]:>8.3f} {n:>5d}")
            metric_stats[col] = {"spearman": overall, "drift": drift,
                                  "early": splits[0], "late": splits[1]}

    # Step 3: Build composite features
    print(f"\n{'=' * 100}")
    print(f"  BUILDING COMPOSITE FEATURES")
    print(f"{'=' * 100}")

    # Only use metrics with at least moderate signal and enough data
    candidate_metrics = [c for c in available
                         if c in metric_stats and metric_stats[c]["spearman"] > 0.05
                         and df[c].notna().sum() >= 100]
    print(f"\nCandidate metrics for composites ({len(candidate_metrics)}):")
    for c in candidate_metrics:
        s = metric_stats[c]
        print(f"  {c}: spearman={s['spearman']:.3f}, drift={s['drift']:.3f}")

    # Z-score normalize all candidates (for composites)
    scaler = StandardScaler()
    z_cols = {}
    for col in candidate_metrics:
        valid_mask = df[col].notna()
        z_name = f"z_{col}"
        df[z_name] = np.nan
        if valid_mask.sum() > 2:
            df.loc[valid_mask, z_name] = scaler.fit_transform(
                df.loc[valid_mask, [col]]).flatten()
            z_cols[col] = z_name

    composite_features = {}

    # --- Technique 1: Weighted averages of pairs ---
    print("\nBuilding weighted average composites...")
    weights_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    yprr_z = z_cols.get("pg_yprr")
    if yprr_z:
        for other_col in candidate_metrics:
            if other_col == "pg_yprr":
                continue
            other_z = z_cols.get(other_col)
            if not other_z:
                continue
            other_short = other_col.replace("pg_", "")
            for w in weights_to_test:
                name = f"wavg_{w:.0%}yprr_{1-w:.0%}{other_short}"
                valid = df[yprr_z].notna() & df[other_z].notna()
                df[name] = np.nan
                df.loc[valid, name] = w * df.loc[valid, yprr_z] + (1 - w) * df.loc[valid, other_z]
                composite_features[name] = name

    # --- Technique 2: Weighted averages of triples (YPRR + 2 others) ---
    print("Building triple weighted average composites...")
    stable_metrics = [c for c in candidate_metrics
                      if c != "pg_yprr" and c in metric_stats
                      and metric_stats[c]["drift"] < 0.20]
    for pair in combinations(stable_metrics, 2):
        z1 = z_cols.get(pair[0])
        z2 = z_cols.get(pair[1])
        if not z1 or not z2 or not yprr_z:
            continue
        s1 = pair[0].replace("pg_", "")
        s2 = pair[1].replace("pg_", "")
        # 50% YPRR + 25% each
        name = f"triple_50yprr_25{s1}_25{s2}"
        valid = df[yprr_z].notna() & df[z1].notna() & df[z2].notna()
        df[name] = np.nan
        df.loc[valid, name] = 0.5 * df.loc[valid, yprr_z] + 0.25 * df.loc[valid, z1] + 0.25 * df.loc[valid, z2]
        composite_features[name] = name

    # --- Technique 3: PCA composites ---
    print("Building PCA composites...")
    # PCA on all candidate metrics
    pca_sets = []
    # All candidates
    pca_sets.append(("pca_all", candidate_metrics))
    # YPRR + each other metric (2-metric PCA)
    for other in candidate_metrics:
        if other == "pg_yprr":
            continue
        short = other.replace("pg_", "")
        pca_sets.append((f"pca_yprr_{short}", ["pg_yprr", other]))
    # YPRR + all stable metrics
    if len(stable_metrics) >= 2:
        pca_sets.append(("pca_yprr_stable", ["pg_yprr"] + stable_metrics))
    # Efficiency only (per-route metrics)
    eff_metrics = [c for c in candidate_metrics if "per_route" in c or "yprr" in c or "target_share" in c]
    if len(eff_metrics) >= 2:
        pca_sets.append(("pca_efficiency", eff_metrics))
    # Volume only (per-game metrics)
    vol_metrics = [c for c in candidate_metrics if "per_game" in c]
    if len(vol_metrics) >= 2:
        pca_sets.append(("pca_volume", vol_metrics))
    # Efficiency + volume
    if len(eff_metrics) >= 1 and len(vol_metrics) >= 1:
        pca_sets.append(("pca_eff_vol", eff_metrics + vol_metrics))

    for pca_name, metric_list in pca_sets:
        cols = [c for c in metric_list if c in df.columns]
        if len(cols) < 2:
            continue
        valid = df[cols].notna().all(axis=1)
        if valid.sum() < 20:
            continue
        pca = PCA(n_components=1)
        vals = scaler.fit_transform(df.loc[valid, cols])
        pc1 = pca.fit_transform(vals).flatten()
        # Ensure positive correlation with tier_num
        from scipy.stats import spearmanr
        corr = spearmanr(pc1, df.loc[valid, "tier_num"])[0]
        if corr < 0:
            pc1 = -pc1
        df[pca_name] = np.nan
        df.loc[valid, pca_name] = pc1
        composite_features[pca_name] = pca_name
        loadings = pca.components_[0]
        var_exp = pca.explained_variance_ratio_[0]
        print(f"  {pca_name}: var_explained={var_exp:.1%}, loadings={dict(zip([c.replace('pg_','') for c in cols], loadings.round(3)))}")

    # --- Technique 4: Geometric means ---
    print("Building geometric mean composites...")
    if yprr_z:
        for other_col in candidate_metrics:
            if other_col == "pg_yprr":
                continue
            short = other_col.replace("pg_", "")
            # Use raw values (not z-scores) for geometric mean
            valid = df["pg_yprr"].notna() & df[other_col].notna() & (df["pg_yprr"] > 0) & (df[other_col] > 0)
            if valid.sum() < 20:
                continue
            name = f"geomean_yprr_{short}"
            df[name] = np.nan
            df.loc[valid, name] = np.sqrt(df.loc[valid, "pg_yprr"] * df.loc[valid, other_col])
            composite_features[name] = name

    # --- Technique 5: Rank-averaged composites ---
    print("Building rank-averaged composites...")
    for other_col in candidate_metrics:
        if other_col == "pg_yprr":
            continue
        short = other_col.replace("pg_", "")
        valid = df["pg_yprr"].notna() & df[other_col].notna()
        if valid.sum() < 20:
            continue
        name = f"rankavg_yprr_{short}"
        df[name] = np.nan
        r1 = df.loc[valid, "pg_yprr"].rank(pct=True)
        r2 = df.loc[valid, other_col].rank(pct=True)
        df.loc[valid, name] = (r1 + r2) / 2
        composite_features[name] = name

    # --- Technique 6: YPRR residualized on YPTPA (and vice versa) ---
    print("Building residualized composites...")
    for base, residual_on in [("pg_yprr", "pg_yptpa"), ("pg_yptpa", "pg_yprr")]:
        if base not in df.columns or residual_on not in df.columns:
            continue
        valid = df[base].notna() & df[residual_on].notna()
        if valid.sum() < 20:
            continue
        from numpy.polynomial.polynomial import polyfit
        coeffs = np.polyfit(df.loc[valid, residual_on], df.loc[valid, base], 1)
        predicted = np.polyval(coeffs, df.loc[valid, residual_on])
        residuals = df.loc[valid, base] - predicted
        base_short = base.replace("pg_", "")
        res_short = residual_on.replace("pg_", "")
        name = f"resid_{base_short}_on_{res_short}"
        df[name] = np.nan
        df.loc[valid, name] = residuals
        composite_features[name] = name

    print(f"\nTotal composite features built: {len(composite_features)}")

    # Step 4: Era stability for all composites
    print(f"\n{'=' * 100}")
    print(f"  COMPOSITE ERA STABILITY")
    print(f"{'=' * 100}")
    print(f"  {'Composite':<55s} {'Spearman':>8s} {'Drift':>8s}")
    print(f"  {'-'*55} {'-'*8} {'-'*8}")

    composite_stability = {}
    for name in sorted(composite_features.keys()):
        overall, drift, splits = compute_era_stability(df, name, years)
        if not np.isnan(overall):
            composite_stability[name] = {"spearman": overall, "drift": drift}
            print(f"  {name:<55s} {overall:>8.3f} {drift:>8.3f}")

    # Step 5: LOO grid search — test each as replacement for pg_yprr_graduated
    print(f"\n{'=' * 100}")
    print(f"  LOO GRID SEARCH: REPLACING pg_yprr_graduated")
    print(f"{'=' * 100}")

    CORE = ["draft_capital", "pg_catch_pct_adot_adj_graduated",
            "best2_contested_catch_rate", "best2_avoided_tackles_per_rec"]

    # Re-aggregate pg_catch_pct_adot_adj_graduated (not in master CSV)
    from aggregation.aggregate_college_stats import aggregate_player
    adot_coef = fit_adot_regression(all_grades)
    for i, (_, row) in enumerate(df.iterrows()):
        result = aggregate_player(
            all_grades, row["name"], row["draft_year"],
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            team_games_lookup=team_games_lookup,
            adot_coef=adot_coef,
        )
        for col in ["pg_yprr_graduated", "pg_catch_pct_adot_adj_graduated"]:
            if col in result:
                df.at[df.index[i], col] = result[col]

    # Configs to test:
    # 1. Baseline: pg_yprr_graduated (current v12)
    # 2. Each individual pg_ metric
    # 3. Each composite
    configs = []

    # Baseline
    configs.append(("BASELINE: pg_yprr_graduated", CORE + ["pg_yprr_graduated"]))

    # Individual pg_ metrics
    for col in available:
        if col == "pg_yprr":  # pg_yprr without age adj is different from pg_yprr_graduated
            continue
        configs.append((f"individual: {col}", CORE + [col]))

    # Also test pg_yprr (without graduated suffix — raw peak-gated)
    if "pg_yprr" in df.columns:
        configs.append(("individual: pg_yprr (no age adj)", CORE + ["pg_yprr"]))

    # Composites
    for name in composite_features:
        configs.append((f"composite: {name}", CORE + [name]))

    # Filter to configs where all features have enough data
    valid_configs = []
    for label, features in configs:
        d_sub = df.dropna(subset=["tier_num"] + features)
        if len(d_sub) >= 80:
            valid_configs.append((label, features))

    print(f"\nTotal configs to test: {len(valid_configs)}")
    print("Running LOO evaluation...\n")

    results = []
    for i, (label, features) in enumerate(valid_configs):
        d_sub = df.dropna(subset=["tier_num"] + features).copy()
        sub_years = sorted(d_sub["draft_year"].unique())
        scores = _loo_ordinal_scores(d_sub, features, sub_years)
        if not scores:
            continue

        # Get era stability of the production metric (last feature = the one replacing YPRR)
        prod_feat = features[-1]
        stab = composite_stability.get(prod_feat, metric_stats.get(prod_feat, {}))

        row = {
            "config": label,
            "prod_feature": prod_feat,
            "n_feats": len(features),
            "n_players": len(d_sub),
            "spearman": stab.get("spearman", np.nan),
            "drift": stab.get("drift", np.nan),
        }
        row.update(scores)
        results.append(row)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(valid_configs)}] {label[:60]}: "
                  f"LL={scores.get('log_loss', np.nan):.3f} "
                  f"Brier={scores.get('brier', np.nan):.3f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={
        "auc_1": "flex_auc", "auc_2": "starter_auc",
        "auc_3": "elite_auc", "auc_4": "stud_auc", "auc_5": "lw_auc",
    })

    # Print results sorted by LogLoss
    print(f"\n{'=' * 140}")
    print(f"  ALL RESULTS (sorted by LogLoss)")
    print(f"{'=' * 140}")
    print(f"  {'Config':<60s} {'LL':>7s} {'Brier':>7s} {'Elite':>7s} {'Stud':>7s} {'Drift':>7s} {'Spear':>7s}")
    print(f"  {'-'*60} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for _, r in results_df.sort_values("log_loss").iterrows():
        marker = " ***" if "BASELINE" in r["config"] else ""
        print(f"  {r['config']:<60s} {r['log_loss']:>7.3f} {r['brier']:>7.3f} "
              f"{r.get('elite_auc', np.nan):>7.3f} {r.get('stud_auc', np.nan):>7.3f} "
              f"{r.get('drift', np.nan):>7.3f} {r.get('spearman', np.nan):>7.3f}{marker}")

    # Pareto analysis: configs that improve on BOTH LogLoss and drift vs baseline
    baseline = results_df[results_df["config"].str.contains("BASELINE")]
    if len(baseline) > 0:
        bl_ll = baseline.iloc[0]["log_loss"]
        bl_drift = baseline.iloc[0].get("drift", np.nan)

        print(f"\n{'=' * 140}")
        print(f"  PARETO FRONTIER: Better LogLoss AND Better Era Stability than baseline")
        print(f"  Baseline: LL={bl_ll:.3f}, Drift={bl_drift:.3f}")
        print(f"{'=' * 140}")

        pareto = results_df[
            (results_df["log_loss"] <= bl_ll) &
            (results_df["drift"] <= bl_drift) &
            (~results_df["config"].str.contains("BASELINE"))
        ].sort_values("log_loss")

        if len(pareto) > 0:
            for _, r in pareto.iterrows():
                ll_delta = r["log_loss"] - bl_ll
                drift_delta = r.get("drift", np.nan) - bl_drift
                print(f"  {r['config']:<60s} LL={r['log_loss']:.3f} ({ll_delta:+.3f}) "
                      f"Drift={r.get('drift', np.nan):.3f} ({drift_delta:+.3f}) "
                      f"Elite={r.get('elite_auc', np.nan):.3f} Stud={r.get('stud_auc', np.nan):.3f}")
        else:
            print("  No configs dominate the baseline on both dimensions.")

        # Also show: best LL regardless of drift, best drift with LL within 5%
        print(f"\n  TOP 10 by LogLoss (regardless of drift):")
        for _, r in results_df.sort_values("log_loss").head(10).iterrows():
            marker = " <-- BASELINE" if "BASELINE" in r["config"] else ""
            print(f"    {r['config']:<60s} LL={r['log_loss']:.3f} Drift={r.get('drift', np.nan):.3f} "
                  f"Elite={r.get('elite_auc', np.nan):.3f}{marker}")

        print(f"\n  TOP 10 by Era Stability (drift < baseline, LL within 10%):")
        stable = results_df[results_df["log_loss"] <= bl_ll * 1.10].sort_values("drift")
        for _, r in stable.head(10).iterrows():
            marker = " <-- BASELINE" if "BASELINE" in r["config"] else ""
            print(f"    {r['config']:<60s} Drift={r.get('drift', np.nan):.3f} LL={r['log_loss']:.3f} "
                  f"Elite={r.get('elite_auc', np.nan):.3f}{marker}")

    # Save
    out_path = os.path.join(DATA_DIR, "outputs", "composite_production_grid_search.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} results to {out_path}")


if __name__ == "__main__":
    main()
