#!/usr/bin/env python3
"""
Feature engineering investigation: replacing career_targeted_qb_rating.

QBR is a composite stat (touchdowns + catch rate + route quality) with high
collinearity to draft capital (rho=0.393) and catch_pct_adot_adj (rho=0.575).
This script engineers ~30 candidate features that capture "target outcome quality"
and tests them using the same 7-part protocol as full_feature_analysis.py.

Candidates include:
  - QBR variants: aDOT-adjusted, senior-discounted, graduated, peak
  - Target-level outcomes: yards/target, value/target, first downs/target
  - Catch ability metrics: clean catch rate, no-negative rate, catch-minus-drops
  - PCA composites: PC1 of target outcomes, PC1 of catch ability
  - Z-score composites: target outcome blend, catch ability blend

Tested in three contexts:
  1. Univariate (raw correlation with tier)
  2. DC-controlled (residual after draft_capital)
  3. Full-model (residual after all other v11 features)

Reads:
  - wr_data/wr_dynasty_value_with_college.csv
  - wr_data/grades/{year}_receiving_grades.csv (2016-2025)
  - wr_data/draft_ages.csv

Outputs:
  - wr_data/outputs/qbr_engineering_candidates.csv (all candidates per player)
  - wr_data/charts/qbr_engineering_*.png (visualizations)
  - wr_data/reports/qbr_engineering_report.md (full report)
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import spearmanr, zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

SUFFIXES_RE = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", re.IGNORECASE)

P5_TEAMS = {
    "ALABAMA", "ARKANSAS", "AUBURN", "FLORIDA", "GEORGIA", "KENTUCKY", "LSU",
    "MISS STATE", "MISSOURI", "OLE MISS", "S CAROLINA", "TENNESSEE", "TEXAS",
    "TEXAS A&M", "OKLAHOMA", "VANDERBILT",
    "ILLINOIS", "INDIANA", "IOWA", "MARYLAND", "MICHIGAN", "MICH STATE",
    "MINNESOTA", "NEBRASKA", "NWESTERN", "OHIO STATE", "PENN STATE", "PURDUE",
    "RUTGERS", "WISCONSIN", "UCLA", "USC", "OREGON", "WASHINGTON",
    "ARIZONA", "ARIZONA ST", "BAYLOR", "BYU", "CINCINNATI", "COLORADO",
    "HOUSTON", "IOWA STATE", "KANSAS", "KANSAS ST", "OKLA STATE", "TCU",
    "TEXAS TECH", "UCF", "W VIRGINIA", "UTAH",
    "BOSTON COL", "CLEMSON", "DUKE", "FLORIDA ST", "GA TECH", "LOUISVILLE",
    "MIAMI FL", "N CAROLINA", "NC STATE", "PITTSBURGH", "SYRACUSE", "VA TECH",
    "VIRGINIA", "WAKE", "SMU", "CAL", "STANFORD",
    "NOTRE DAME", "OREGON ST", "WASH STATE",
}

SEASON_EXCLUSIONS = {
    ("elijah sarratt", "JAMES MAD", 2023),
    ("kyle williams", "UNLV", 2020),
    ("kyle williams", "UNLV", 2021),
    ("kyle williams", "UNLV", 2022),
}

GRADUATED_ADJ = {
    (0, 19.5): 1.25,
    (19.5, 20.5): 1.05,
    (20.5, 21.5): 0.80,
    (21.5, 99): 0.75,
}

SENIOR_AGE_THRESHOLD = 21.5
SENIOR_DISCOUNT_PP = 10.0

# v11 features (excluding the two under investigation)
ANCHOR_FEATURES = [
    "draft_capital",
    "best1_yprr_graduated",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

# The two features we're potentially replacing
INCUMBENT_FEATURES = [
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
]


def normalize_name(name):
    n = SUFFIXES_RE.sub("", str(name)).strip()
    n = n.replace(".", "").replace("'", "").lower()
    return " ".join(n.split())


def get_age_on_sept1(birthdate, year):
    if birthdate is None or pd.isna(birthdate):
        return None
    sept1 = pd.Timestamp(f"{int(year)}-09-01")
    return (sept1 - birthdate).days / 365.25


# ============================================================
# STEP 1: Load data and engineer season-level features
# ============================================================

def load_grades():
    """Load all receiving grades files."""
    all_grades = []
    for yr in range(2016, 2026):
        path = os.path.join(DATA_DIR, "grades", f"{yr}_receiving_grades.csv")
        if os.path.exists(path):
            g = pd.read_csv(path)
            g["grade_year"] = yr
            all_grades.append(g)
    ag = pd.concat(all_grades, ignore_index=True)
    ag["_join_key"] = ag["player"].apply(normalize_name)

    num_cols = [
        "routes", "targets", "receptions", "yards", "touchdowns", "first_downs",
        "avoided_tackles", "drops", "yards_after_catch", "contested_targets",
        "contested_receptions", "player_game_count", "targeted_qb_rating",
        "caught_percent", "drop_rate", "avg_depth_of_target", "yards_per_reception",
        "yards_after_catch_per_reception", "yprr", "grades_offense",
        "grades_pass_route", "grades_hands_drop", "contested_catch_rate",
        "interceptions",
    ]
    for c in num_cols:
        if c in ag.columns:
            ag[c] = pd.to_numeric(ag[c], errors="coerce")
    return ag


def fit_regressions(ag):
    """Fit aDOT adjustment regressions on full population."""
    # catch% ~ aDOT
    cp, adot = ag["caught_percent"], ag["avg_depth_of_target"]
    m = cp.notna() & adot.notna()
    adot_catch_coef = np.polyfit(adot[m].values, cp[m].values, 1)

    # QBR ~ aDOT
    qbr = ag["targeted_qb_rating"]
    m2 = qbr.notna() & adot.notna()
    adot_qbr_coef = np.polyfit(adot[m2].values, qbr[m2].values, 1)

    # YPR ~ aDOT
    ypr = ag["yards_per_reception"]
    m3 = ypr.notna() & adot.notna()
    adot_ypr_coef = np.polyfit(adot[m3].values, ypr[m3].values, 1)

    return adot_catch_coef, adot_qbr_coef, adot_ypr_coef


def engineer_season_features(qual, adot_catch_coef, adot_qbr_coef, adot_ypr_coef):
    """Add all engineered features at the season level."""
    q = qual.copy()

    # aDOT-adjusted QBR
    q["qbr_adot_adj"] = q["targeted_qb_rating"] - np.polyval(adot_qbr_coef, q["avg_depth_of_target"])

    # aDOT-adjusted catch% (production version)
    q["catch_pct_adot_adj"] = q["caught_percent"] - np.polyval(adot_catch_coef, q["avg_depth_of_target"])

    # aDOT-adjusted YPR
    q["ypr_adot_adj"] = q["yards_per_reception"] - np.polyval(adot_ypr_coef, q["avg_depth_of_target"])

    # Target-outcome rates
    q["yards_per_target"] = q["yards"] / q["targets"]
    q["first_downs_per_target"] = q["first_downs"] / q["targets"]
    q["td_per_target"] = q["touchdowns"] / q["targets"]
    q["value_per_target"] = (q["first_downs"] + q["touchdowns"]) / q["targets"]
    q["yac_per_target"] = q["yards_after_catch"] / q["targets"]

    # Catch reliability variants
    non_ct = q["targets"] - q["contested_targets"]
    non_cr = q["receptions"] - q["contested_receptions"]
    q["clean_catch_rate"] = np.where(non_ct > 0, non_cr / non_ct * 100, np.nan)
    q["catch_minus_drops"] = q["caught_percent"] - q["drop_rate"]
    q["no_negative_rate"] = 1 - (q["drops"] + q["interceptions"]) / q["targets"]

    # Z-score composites (computed within the qualified population)
    z_cols = [
        "catch_pct_adot_adj", "qbr_adot_adj", "yards_per_target",
        "value_per_target", "yac_per_target", "no_negative_rate",
        "clean_catch_rate", "grades_hands_drop", "first_downs_per_target",
        "catch_minus_drops", "targeted_qb_rating", "caught_percent",
    ]
    for col in z_cols:
        valid_mask = q[col].notna()
        if valid_mask.sum() > 10:
            q.loc[valid_mask, f"z_{col}"] = zscore(q.loc[valid_mask, col])

    # Composite z-scores
    q["z_target_outcome"] = q[["z_catch_pct_adot_adj", "z_qbr_adot_adj", "z_yards_per_target"]].mean(axis=1)
    q["z_catch_ability"] = q[["z_catch_pct_adot_adj", "z_no_negative_rate", "z_grades_hands_drop"]].mean(axis=1)
    q["z_reception_value"] = q[["z_yards_per_target", "z_yac_per_target", "z_first_downs_per_target"]].mean(axis=1)
    q["z_target_quality_full"] = q[[
        "z_catch_pct_adot_adj", "z_qbr_adot_adj", "z_yards_per_target",
        "z_yac_per_target", "z_no_negative_rate",
    ]].mean(axis=1)

    # PCA: target outcome (6 components -> PC1, PC2)
    pca_cols = [
        "catch_pct_adot_adj", "qbr_adot_adj", "yards_per_target",
        "yac_per_target", "value_per_target", "no_negative_rate",
    ]
    pca_valid = q[pca_cols].dropna()
    if len(pca_valid) > 50:
        X_pca = StandardScaler().fit_transform(pca_valid)
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(X_pca)
        q.loc[pca_valid.index, "pca_target_outcome_1"] = pcs[:, 0]
        q.loc[pca_valid.index, "pca_target_outcome_2"] = pcs[:, 1]
        print(f"  Target outcome PCA variance explained: {pca.explained_variance_ratio_}")
        print(f"  PC1 loadings: {dict(zip(pca_cols, pca.components_[0].round(3)))}")

    # PCA: catch ability (4 components -> PC1)
    catch_pca_cols = ["catch_pct_adot_adj", "no_negative_rate", "clean_catch_rate", "grades_hands_drop"]
    catch_valid = q[catch_pca_cols].dropna()
    if len(catch_valid) > 50:
        X_catch = StandardScaler().fit_transform(catch_valid)
        catch_pca = PCA(n_components=2)
        catch_pcs = catch_pca.fit_transform(X_catch)
        q.loc[catch_valid.index, "pca_catch_ability"] = catch_pcs[:, 0]
        print(f"  Catch ability PCA variance explained: {catch_pca.explained_variance_ratio_}")
        print(f"  PC1 loadings: {dict(zip(catch_pca_cols, catch_pca.components_[0].round(3)))}")

    return q


# ============================================================
# STEP 1b: Supervised composites (Ridge-weighted catch metrics)
# ============================================================

def build_supervised_composites(df, qual_seasons, birth_lookup):
    """Build supervised composites by fitting Ridge on catch-metric components
    against tier outcome, then projecting back to season level.

    This finds the optimal weighting of catch-quality stats to predict dynasty
    outcome, creating a single 'supervised catch composite' feature.
    """
    # Use whichever catch-quality columns actually exist in df for each window.
    catch_components = [
        "catch_pct_adot_adj", "clean_catch_rate", "no_negative_rate",
        "qbr_adot_adj", "grades_hands_drop", "catch_minus_drops",
    ]

    composites = {}

    for prefix in ["career", "best1", "best2"]:
        cols = [f"{prefix}_{c}" for c in catch_components if f"{prefix}_{c}" in df.columns]
        if len(cols) < 3:
            print(f"  Skipping {prefix} supervised composite ({len(cols)} cols available)")
            continue
        valid = df[cols + ["tier_num"]].dropna()
        if len(valid) < 30:
            continue

        X = valid[cols].values
        y = valid["tier_num"].values

        # Fit Ridge to find optimal weights
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        ridge = Ridge(alpha=1.0).fit(X_s, y)
        weights = ridge.coef_

        print(f"  Supervised composite ({prefix}) weights:")
        for c, w in zip(catch_components, weights):
            print(f"    {c:30s}: {w:+.3f}")

        # Project all players (including those that were in test above)
        all_valid = df[cols].dropna()
        X_all = scaler.transform(all_valid.values)
        projection = X_all @ weights
        composites[f"{prefix}_supervised_catch"] = pd.Series(
            projection, index=all_valid.index
        )

    # Also build LOO-supervised composite (avoid target leakage)
    # Fit weights on each LOO-year fold, predict held-out year
    if "draft_year" in df.columns:
        for prefix in ["career", "best1"]:
            cols = [f"{prefix}_{c}" for c in catch_components if f"{prefix}_{c}" in df.columns]
            if len(cols) < 3:
                continue
            valid = df[cols + ["tier_num", "draft_year"]].dropna()
            if len(valid) < 30:
                continue

            loo_preds = pd.Series(np.nan, index=valid.index)
            for yr in valid["draft_year"].unique():
                train = valid[valid["draft_year"] != yr]
                test = valid[valid["draft_year"] == yr]
                if len(train) < 20 or len(test) == 0:
                    continue
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(train[cols].values)
                ridge = Ridge(alpha=1.0).fit(X_tr, train["tier_num"].values)
                X_te = scaler.transform(test[cols].values)
                loo_preds.loc[test.index] = X_te @ ridge.coef_

            composites[f"{prefix}_supervised_catch_loo"] = loo_preds

    return composites


# ============================================================
# STEP 2: Aggregate per player (career, best2, best1, peak)
# ============================================================

# Features to aggregate across all temporal windows
ENGINEERED_COLS = [
    "qbr_adot_adj", "yards_per_target", "value_per_target", "first_downs_per_target",
    "clean_catch_rate", "catch_minus_drops", "yac_per_target", "ypr_adot_adj",
    "td_per_target", "no_negative_rate",
    "z_target_outcome", "z_catch_ability", "z_reception_value", "z_target_quality_full",
    "pca_target_outcome_1", "pca_target_outcome_2", "pca_catch_ability",
]

# Features that are rate-per-target (weight by targets)
TARGET_WEIGHTED = {
    "qbr_adot_adj", "yards_per_target", "value_per_target", "first_downs_per_target",
    "clean_catch_rate", "catch_minus_drops", "yac_per_target", "ypr_adot_adj",
    "td_per_target", "no_negative_rate", "targeted_qb_rating", "catch_pct_adot_adj",
}


def _wavg(series, weights):
    """Weighted average with NaN handling."""
    mask = series.notna() & pd.Series(weights).notna()
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    return np.average(series[mask], weights=weights[mask])


def aggregate_player_engineered(player_key, draft_year, birthdate, qual_seasons):
    """Aggregate all engineered features for a single player."""
    seasons = qual_seasons[
        (qual_seasons["_join_key"] == player_key) &
        (qual_seasons["grade_year"] <= draft_year)
    ].copy()

    # Apply exclusions
    if len(seasons) > 0:
        excl = seasons.apply(
            lambda r: (r["_join_key"], r.get("team_name", ""), r.get("grade_year", 0))
            in SEASON_EXCLUSIONS, axis=1
        )
        seasons = seasons[~excl]

    # Age filter
    if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
        min_year = birthdate.year + 18
        seasons = seasons[seasons["grade_year"] >= min_year]
    if len(seasons) > 0:
        seasons = seasons[seasons["grade_year"] >= draft_year - 5]

    if len(seasons) == 0:
        return {}

    # P5-filtered eligible seasons for best2/best1
    p5 = seasons[seasons["team_name"].isin(P5_TEAMS)] if "team_name" in seasons.columns else seasons
    eligible = p5 if len(p5) >= 2 else seasons

    # Best2: top 2 by grades_offense
    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().sum() >= 2:
        best2 = eligible.loc[grades.nlargest(2).index].copy()
    else:
        best2 = eligible.copy()

    # Best1: top 1 by grades_offense
    if grades.notna().any():
        best1_row = eligible.loc[grades.idxmax()]
    else:
        best1_row = eligible.iloc[0]

    result = {}

    # --- Career aggregation ---
    for feat in ENGINEERED_COLS:
        if feat not in seasons.columns:
            continue
        wt_col = "targets" if feat in TARGET_WEIGHTED else "player_game_count"
        val = _wavg(seasons[feat], seasons[wt_col].values)
        if pd.notna(val):
            result[f"career_{feat}"] = round(val, 4)

    # --- Best2 aggregation ---
    for feat in ENGINEERED_COLS:
        if feat not in best2.columns:
            continue
        wt_col = "targets" if feat in TARGET_WEIGHTED else "player_game_count"
        val = _wavg(best2[feat], best2[wt_col].values)
        if pd.notna(val):
            result[f"best2_{feat}"] = round(val, 4)

    # --- Best1 values ---
    for feat in ENGINEERED_COLS:
        if feat in best1_row.index and pd.notna(best1_row[feat]):
            result[f"best1_{feat}"] = round(float(best1_row[feat]), 4)

    # --- Senior-discounted QBR and QBR-aDOT-adj ---
    if birthdate is not None and pd.notna(birthdate):
        for prefix, source in [("career", seasons), ("best2", best2)]:
            disc = source.copy()
            for idx, row in disc.iterrows():
                age = get_age_on_sept1(birthdate, row["grade_year"])
                if age is not None and age >= SENIOR_AGE_THRESHOLD:
                    for col in ["targeted_qb_rating", "qbr_adot_adj"]:
                        if pd.notna(disc.at[idx, col]):
                            disc.at[idx, col] -= SENIOR_DISCOUNT_PP
            for feat in ["targeted_qb_rating", "qbr_adot_adj"]:
                val = _wavg(disc[feat], disc["targets"].values)
                if pd.notna(val):
                    result[f"{prefix}_{feat}_sr_disc"] = round(val, 4)

    # --- Graduated age-adjusted QBR (like YPRR graduated) ---
    if birthdate is not None and pd.notna(birthdate) and grades.notna().any():
        best_row = eligible.loc[grades.idxmax()]
        age = get_age_on_sept1(birthdate, best_row["grade_year"])
        if age is not None:
            for feat, center in [("targeted_qb_rating", 110), ("qbr_adot_adj", 0)]:
                raw = best_row.get(feat)
                if pd.notna(raw):
                    for (lo, hi), mult in GRADUATED_ADJ.items():
                        if lo <= age < hi:
                            adjusted = (raw - center) * mult + center
                            suffix = "qbr_graduated" if feat == "targeted_qb_rating" else "qbr_adot_adj_graduated"
                            result[f"best1_{suffix}"] = round(adjusted, 4)
                            break

    # --- Peak QBR (max single season) ---
    for feat, out_name in [
        ("targeted_qb_rating", "peak_qbr"),
        ("qbr_adot_adj", "peak_qbr_adot_adj"),
    ]:
        vals = eligible[feat].dropna()
        if len(vals) > 0:
            result[out_name] = round(float(vals.max()), 4)

    return result


# ============================================================
# STEP 3: 7-part analysis protocol
# ============================================================

def run_analysis(candidates, df, base_features, label=""):
    """Run full 7-part analysis for a set of candidates against a base."""
    valid_candidates = [c for c in candidates if c in df.columns and df[c].notna().sum() > 50]
    if not valid_candidates:
        print(f"  No valid candidates for {label}")
        return pd.DataFrame()

    all_cols = base_features + valid_candidates + ["tier_num", "draft_year"]
    d = df.dropna(subset=[c for c in all_cols if c in df.columns]).copy()
    y = d["tier_num"].values
    hit = (y >= 3).astype(int)
    years = sorted(d["draft_year"].unique())

    print(f"\n{'=' * 74}")
    print(f"  {label} | n={len(d)} | base={base_features}")
    print(f"{'=' * 74}")

    results = []

    # Prepare base residuals
    scaler = StandardScaler()
    X_base = scaler.fit_transform(d[base_features].values)
    ridge = Ridge(alpha=1.0).fit(X_base, y)
    residuals = y - ridge.predict(X_base)

    # Base LOO-AUC
    base_auc = _loo_auc(d, base_features, years)

    for c in valid_candidates:
        row = {"feature": c}

        # Part 1: Univariate
        sp, sp_p = spearmanr(d[c].values, y)
        auc = roc_auc_score(hit, d[c].values) if d[c].nunique() > 1 else 0.5
        row["spearman"] = round(sp, 3)
        row["auc"] = round(auc, 3)

        # Part 2: Era stability
        mid = years[len(years) // 2]
        early = d[d["draft_year"] <= mid]
        late = d[d["draft_year"] > mid]
        sp_e, _ = spearmanr(early[c].values, early["tier_num"].values) if len(early) > 10 else (np.nan, np.nan)
        sp_l, _ = spearmanr(late[c].values, late["tier_num"].values) if len(late) > 10 else (np.nan, np.nan)
        row["era_drift"] = round(abs(sp_e - sp_l), 3) if pd.notna(sp_e) and pd.notna(sp_l) else np.nan

        # Part 3: Residual signal
        sp_res, p_res = spearmanr(d[c].values, residuals)
        row["residual"] = round(sp_res, 3)

        # Part 4: Max collinearity with base
        max_corr = 0
        for bf in base_features:
            corr = abs(spearmanr(d[bf].values, d[c].values)[0])
            max_corr = max(max_corr, corr)
        row["max_collinearity"] = round(max_corr, 3)

        # Part 5: Bootstrap
        n_boot = 1000
        rng = np.random.RandomState(42)
        boot_rhos = []
        for _ in range(n_boot):
            idx = rng.choice(len(d), size=len(d), replace=True)
            X_b = scaler.fit_transform(d[base_features].values[idx])
            ridge.fit(X_b, y[idx])
            res_b = y[idx] - ridge.predict(X_b)
            r, _ = spearmanr(d[c].values[idx], res_b)
            boot_rhos.append(r)
        boot_arr = np.array(boot_rhos)
        row["boot_pct_pos"] = round((boot_arr > 0).mean(), 3)
        row["boot_mean"] = round(boot_arr.mean(), 3)

        # Part 6: LOO-AUC
        feat_auc = _loo_auc(d, base_features + [c], years)
        row["loo_auc"] = round(feat_auc, 3)
        row["loo_delta"] = round(feat_auc - base_auc, 3)

        # Part 7: Elastic net survival
        enet_count = 0
        y_bin = hit
        for C in [0.01, 0.1, 1.0]:
            X_all = StandardScaler().fit_transform(d[base_features + [c]].values)
            lr = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=C, max_iter=10000, random_state=42,
            )
            lr.fit(X_all, y_bin)
            if abs(lr.coef_[0, -1]) > 1e-6:
                enet_count += 1
        row["enet_survive"] = enet_count

        results.append(row)

    results_df = pd.DataFrame(results)

    # Print summary table
    print(f"\n  {'Feature':<42s} {'Sp':>6s} {'AUC':>6s} {'Drift':>6s} {'Resid':>6s} "
          f"{'Boot%':>6s} {'LOO':>6s} {'Delta':>7s} {'Enet':>5s} {'Collin':>6s}")
    print(f"  {'-'*42} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*6}")
    for _, r in results_df.sort_values("loo_delta", ascending=False).iterrows():
        print(f"  {r['feature']:<42s} {r['spearman']:>+6.3f} {r['auc']:>6.3f} "
              f"{r['era_drift']:>6.3f} {r['residual']:>+6.3f} {r['boot_pct_pos']:>6.1%} "
              f"{r['loo_auc']:>6.3f} {r['loo_delta']:>+7.3f} {r['enet_survive']:>3d}/3 "
              f"{r['max_collinearity']:>6.3f}")

    return results_df


def _loo_auc(d, features, years):
    """Leave-one-year-out AUC for >=Elite threshold."""
    all_p, all_t = [], []
    for yr in years:
        train = d[d["draft_year"] != yr]
        test = d[d["draft_year"] == yr]
        y_tr = (train["tier_num"].values >= 3).astype(int)
        y_te = (test["tier_num"].values >= 3).astype(int)
        if y_tr.sum() < 2 or y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(train[features].values)
        X_te = sc.transform(test[features].values)
        lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        lr.fit(X_tr, y_tr)
        all_p.extend(lr.predict_proba(X_te)[:, 1])
        all_t.extend(y_te)
    return roc_auc_score(np.array(all_t), np.array(all_p)) if all_t else np.nan


# ============================================================
# STEP 4: Combination testing (replace QBR, replace both, etc.)
# ============================================================

def _loo_ordinal_scores(d, features, years, n_tiers=6):
    """LOO ordinal LogLoss and Brier score using K-1 cumulative binary classifiers.

    Mirrors the XGBoost cumulative link approach: train K-1 binary classifiers
    (>=1, >=2, ..., >=5), then convert to ordinal probabilities.
    Returns (log_loss, brier, auc_elite, auc_stud, auc_starter).
    """
    thresholds = list(range(1, n_tiers))  # [1, 2, 3, 4, 5]
    n = len(d)

    # Collect per-threshold probabilities for each held-out player
    cum_probs = np.zeros((n, len(thresholds)))  # P(tier >= k)
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
                # Use class prior
                cum_probs[test_idx, ti] = y_bin.mean()
                continue
            lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
            lr.fit(X_tr_s, y_bin)
            cum_probs[test_idx, ti] = lr.predict_proba(X_te_s)[:, 1]

        mask_predicted[test_mask] = True

    if not mask_predicted.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Convert cumulative probs to ordinal tier probabilities
    # P(tier=0) = 1 - P(>=1)
    # P(tier=k) = P(>=k) - P(>=k+1)  for k=1..4
    # P(tier=5) = P(>=5)
    idx = mask_predicted
    n_pred = idx.sum()
    tier_probs = np.zeros((n_pred, n_tiers))

    cp = cum_probs[idx]
    # Enforce monotonicity: P(>=k) >= P(>=k+1)
    for ti in range(len(thresholds) - 1, 0, -1):
        cp[:, ti - 1] = np.maximum(cp[:, ti - 1], cp[:, ti])

    tier_probs[:, 0] = 1 - cp[:, 0]
    for k in range(1, n_tiers - 1):
        tier_probs[:, k] = cp[:, k - 1] - cp[:, k]
    tier_probs[:, n_tiers - 1] = cp[:, -1]

    # Clip and renormalize
    tier_probs = np.clip(tier_probs, 1e-8, 1.0)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)

    y_pred = y[idx]

    # Ordinal LogLoss
    ll = -np.mean(np.log(tier_probs[np.arange(n_pred), y_pred]))

    # Brier score (one-hot vs predicted probs, averaged across classes)
    one_hot = np.zeros((n_pred, n_tiers))
    one_hot[np.arange(n_pred), y_pred] = 1
    brier = np.mean(np.sum((tier_probs - one_hot) ** 2, axis=1))

    # Binary AUCs at key thresholds
    auc_elite = roc_auc_score((y_pred >= 3).astype(int), cp[:, 2]) if (y_pred >= 3).sum() > 0 and (y_pred < 3).sum() > 0 else np.nan
    auc_stud = roc_auc_score((y_pred >= 4).astype(int), cp[:, 3]) if (y_pred >= 4).sum() > 0 and (y_pred < 4).sum() > 0 else np.nan
    auc_starter = roc_auc_score((y_pred >= 2).astype(int), cp[:, 1]) if (y_pred >= 2).sum() > 0 and (y_pred < 2).sum() > 0 else np.nan

    return ll, brier, auc_elite, auc_stud, auc_starter


def test_combinations(top_candidates, df, label=""):
    """Test specific feature combinations with ordinal LogLoss, Brier, and AUC."""
    print(f"\n{'=' * 74}")
    print(f"  COMBINATION TESTS: {label}")
    print(f"{'=' * 74}")

    all_cols = ANCHOR_FEATURES + INCUMBENT_FEATURES + top_candidates + ["tier_num", "draft_year"]
    existing = [c for c in all_cols if c in df.columns]
    d = df.dropna(subset=existing).copy()
    years = sorted(d["draft_year"].unique())

    combos = {}

    # Baseline: current v11
    v11_feats = ANCHOR_FEATURES + INCUMBENT_FEATURES
    v11_feats = [f for f in v11_feats if f in d.columns]
    combos["v11 (current)"] = v11_feats

    # Drop QBR only
    combos["v11 minus QBR"] = [f for f in v11_feats if f != "career_targeted_qb_rating"]

    # Drop both QBR and CPA
    combos["v11 minus QBR+CPA"] = ANCHOR_FEATURES[:]

    # Replace QBR with each top candidate
    for cand in top_candidates:
        if cand in d.columns:
            feats = [f for f in v11_feats if f != "career_targeted_qb_rating"] + [cand]
            combos[f"replace QBR -> {cand}"] = feats

    # Replace BOTH QBR and CPA with single composite
    for cand in top_candidates:
        if cand in d.columns:
            feats = ANCHOR_FEATURES + [cand]
            combos[f"replace QBR+CPA -> {cand}"] = feats

    # Replace both with two new features
    for i, c1 in enumerate(top_candidates):
        for c2 in top_candidates[i + 1:]:
            if c1 in d.columns and c2 in d.columns:
                feats = ANCHOR_FEATURES + [c1, c2]
                short1 = c1.split("_", 1)[-1][:15]
                short2 = c2.split("_", 1)[-1][:15]
                combos[f"replace both -> {short1} + {short2}"] = feats

    print(f"\n  n={len(d)} | years={[int(y) for y in years]}")
    print(f"\n  {'Combination':<55s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} "
          f"{'>=Stud':>8s} {'>=Start':>8s} {'#F':>4s}")
    print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    combo_results = []
    for name, feats in combos.items():
        valid_feats = [f for f in feats if f in d.columns]
        ll, brier, auc_e, auc_s, auc_st = _loo_ordinal_scores(d, valid_feats, years)
        row = {
            "combo": name, "features": valid_feats, "n_feats": len(valid_feats),
            "log_loss": round(ll, 3) if pd.notna(ll) else np.nan,
            "brier": round(brier, 3) if pd.notna(brier) else np.nan,
            "elite_auc": round(auc_e, 3) if pd.notna(auc_e) else np.nan,
            "stud_auc": round(auc_s, 3) if pd.notna(auc_s) else np.nan,
            "starter_auc": round(auc_st, 3) if pd.notna(auc_st) else np.nan,
        }
        combo_results.append(row)
        print(f"  {name:<55s} {row['log_loss']:>8.3f} {row['brier']:>8.3f} "
              f"{row['elite_auc']:>8.3f} {row['stud_auc']:>8.3f} "
              f"{row['starter_auc']:>8.3f} {row['n_feats']:>4d}")

    return pd.DataFrame(combo_results)


# ============================================================
# STEP 5: Visualization
# ============================================================

def generate_visualizations(results_dc, results_full, combo_results, df):
    """Generate all charts for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d",
        "grid.color": "#21262d",
        "font.family": "monospace",
        "font.size": 10,
    })

    charts_dir = os.path.join(DATA_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # --- Figure 1: Candidate comparison (DC context) ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Panel A: Spearman vs LOO-AUC delta (DC base)
    ax = axes[0, 0]
    r = results_dc.copy()
    colors = ["#f78166" if "qbr" in f or "targeted" in f else
              "#58a6ff" if "catch" in f or "clean" in f or "no_neg" in f or "pca_catch" in f else
              "#3fb950" if "pca_target" in f or "z_target" in f or "z_catch" in f else
              "#d29922" for f in r["feature"]]
    ax.scatter(r["spearman"], r["loo_delta"], c=colors, alpha=0.7, s=60, edgecolors="none")
    for _, row in r.iterrows():
        short = row["feature"].replace("career_", "c_").replace("best2_", "b2_").replace("best1_", "b1_")
        if abs(row["loo_delta"]) > 0.005 or abs(row["spearman"]) > 0.25:
            ax.annotate(short, (row["spearman"], row["loo_delta"]),
                        fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    # Mark incumbents
    for inc in INCUMBENT_FEATURES:
        if inc in r["feature"].values:
            ir = r[r["feature"] == inc].iloc[0]
            ax.scatter([ir["spearman"]], [ir["loo_delta"]], c="white", s=100,
                       marker="*", zorder=10, edgecolors="none")
    ax.axhline(0, color="#f0883e", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Univariate Spearman")
    ax.set_ylabel("LOO-AUC Delta (vs DC-only base)")
    ax.set_title("A. Univariate Signal vs Predictive Lift (DC base)", fontweight="bold")

    # Panel B: Residual vs Bootstrap % positive (DC base)
    ax = axes[0, 1]
    ax.scatter(r["residual"], r["boot_pct_pos"], c=colors, alpha=0.7, s=60, edgecolors="none")
    for _, row in r.iterrows():
        short = row["feature"].replace("career_", "c_").replace("best2_", "b2_").replace("best1_", "b1_")
        if row["boot_pct_pos"] > 0.7 or row["boot_pct_pos"] < 0.3:
            ax.annotate(short, (row["residual"], row["boot_pct_pos"]),
                        fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="#f0883e", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="#8b949e", linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Residual Spearman (after DC)")
    ax.set_ylabel("Bootstrap % Positive")
    ax.set_title("B. Residual Signal Reliability (DC base)", fontweight="bold")

    # Panel C: Top candidates ranked by composite score (DC)
    ax = axes[1, 0]
    r2 = r.copy()
    # Composite: normalize each metric to [0,1], average
    for col in ["spearman", "loo_delta", "residual", "boot_pct_pos"]:
        mn, mx = r2[col].min(), r2[col].max()
        r2[f"{col}_norm"] = (r2[col] - mn) / (mx - mn + 1e-9)
    r2["composite"] = (r2["spearman_norm"] * 0.2 + r2["loo_delta_norm"] * 0.3 +
                        r2["residual_norm"] * 0.25 + r2["boot_pct_pos_norm"] * 0.25)
    top15 = r2.nlargest(15, "composite")

    y_pos = range(len(top15))
    bar_colors = ["#f78166" if "qbr" in f or "targeted" in f else
                  "#58a6ff" if "catch" in f or "clean" in f or "no_neg" in f else
                  "#3fb950" for f in top15["feature"]]
    ax.barh(y_pos, top15["composite"].values, color=bar_colors, alpha=0.7)
    ax.set_yticks(list(y_pos))
    short_names = [f.replace("career_", "c:").replace("best2_", "b2:").replace("best1_", "b1:").replace("peak_", "pk:")
                   for f in top15["feature"]]
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Composite Score (weighted)")
    ax.set_title("C. Top 15 Candidates (DC base, composite rank)", fontweight="bold")
    ax.invert_yaxis()

    # Panel D: Full-model context top candidates
    ax = axes[1, 1]
    if results_full is not None and len(results_full) > 0:
        rf = results_full.copy()
        for col in ["spearman", "loo_delta", "residual", "boot_pct_pos"]:
            mn, mx = rf[col].min(), rf[col].max()
            rf[f"{col}_norm"] = (rf[col] - mn) / (mx - mn + 1e-9)
        rf["composite"] = (rf["spearman_norm"] * 0.2 + rf["loo_delta_norm"] * 0.3 +
                           rf["residual_norm"] * 0.25 + rf["boot_pct_pos_norm"] * 0.25)
        top15f = rf.nlargest(15, "composite")
        y_pos2 = range(len(top15f))
        bar_colors2 = ["#f78166" if "qbr" in f or "targeted" in f else
                       "#58a6ff" if "catch" in f or "clean" in f or "no_neg" in f else
                       "#3fb950" for f in top15f["feature"]]
        ax.barh(y_pos2, top15f["composite"].values, color=bar_colors2, alpha=0.7)
        ax.set_yticks(list(y_pos2))
        short_names2 = [f.replace("career_", "c:").replace("best2_", "b2:").replace("best1_", "b1:").replace("peak_", "pk:")
                        for f in top15f["feature"]]
        ax.set_yticklabels(short_names2, fontsize=8)
        ax.set_xlabel("Composite Score (weighted)")
        ax.set_title("D. Top 15 Candidates (full model base)", fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle("Target Outcome Feature Engineering: Candidate Analysis", fontsize=14, fontweight="bold", y=0.98)
    path1 = os.path.join(charts_dir, "qbr_engineering_candidates.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")
    plt.close(fig)

    # --- Figure 2: Combination results (3-panel: LogLoss, Brier, AUC) ---
    if combo_results is not None and len(combo_results) > 0:
        # Filter to key combos to keep readable
        key_patterns = [
            "v11 (current)", "v11 minus QBR", "v11 minus QBR+CPA",
            "replace QBR ->", "replace QBR+CPA ->",
        ]
        # Also include supervised composites and top "replace both" combos
        key_mask = combo_results["combo"].apply(
            lambda x: any(p in x for p in key_patterns[:3]) or
            (any(p in x for p in key_patterns[3:]) and "replace both" not in x)
        )
        # Add top 5 "replace both" by log_loss
        both_combos = combo_results[combo_results["combo"].str.startswith("replace both")]
        if len(both_combos) > 0 and "log_loss" in both_combos.columns:
            top_both = both_combos.nsmallest(5, "log_loss")
            key_mask = key_mask | combo_results.index.isin(top_both.index)

        cr = combo_results[key_mask].copy()

        fig2, axes2 = plt.subplots(1, 3, figsize=(24, max(8, len(cr) * 0.35)))

        v11_row = cr[cr["combo"] == "v11 (current)"]
        v11_ll = v11_row["log_loss"].values[0] if len(v11_row) > 0 and "log_loss" in cr.columns else np.nan
        v11_br = v11_row["brier"].values[0] if len(v11_row) > 0 and "brier" in cr.columns else np.nan
        v11_auc = v11_row["elite_auc"].values[0] if len(v11_row) > 0 else np.nan

        for ax, metric, v11_val, title, lower_better in [
            (axes2[0], "log_loss", v11_ll, "Ordinal LogLoss (lower = better)", True),
            (axes2[1], "brier", v11_br, "Ordinal Brier Score (lower = better)", True),
            (axes2[2], "elite_auc", v11_auc, ">=Elite AUC (higher = better)", False),
        ]:
            if metric not in cr.columns:
                continue
            sorted_cr = cr.sort_values(metric, ascending=not lower_better).copy()
            y_pos = range(len(sorted_cr))

            if lower_better:
                bar_colors = ["#3fb950" if v <= v11_val else "#f85149"
                              for v in sorted_cr[metric]]
            else:
                bar_colors = ["#3fb950" if v >= v11_val else "#f85149"
                              for v in sorted_cr[metric]]

            bars = ax.barh(y_pos, sorted_cr[metric].values, color=bar_colors, alpha=0.7)
            ax.axvline(v11_val, color="#f0883e", linewidth=2, linestyle="--", alpha=0.8)
            ax.set_yticks(list(y_pos))
            short_labels = sorted_cr["combo"].str.replace("replace QBR\\+CPA -> ", "QBR+CPA=> ", regex=True)
            short_labels = short_labels.str.replace("replace QBR -> ", "QBR=> ", regex=True)
            short_labels = short_labels.str.replace("replace both -> ", "both=> ", regex=True)
            ax.set_yticklabels(short_labels.values, fontsize=7)
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_title(title, fontweight="bold", fontsize=10)

            for bar, val in zip(bars, sorted_cr[metric].values):
                delta = val - v11_val
                ax.text(val + (0.001 if not lower_better else 0.002),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f} ({delta:+.3f})", va="center", fontsize=7)

        fig2.suptitle("Combination Results: LogLoss, Brier, and AUC", fontsize=13, fontweight="bold")
        path2 = os.path.join(charts_dir, "qbr_engineering_combos.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path2}")
        plt.close(fig2)

    # --- Figure 3: Feature family comparison ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(20, 8))

    families = {
        "QBR variants": [f for f in results_dc["feature"] if "qbr" in f or "targeted" in f],
        "Catch/reliability": [f for f in results_dc["feature"] if any(x in f for x in ["catch", "clean", "no_neg", "drop"])],
        "Composites/PCA": [f for f in results_dc["feature"] if any(x in f for x in ["pca", "z_target", "z_catch", "z_recep"])],
    }

    for ax, (fam_name, fam_feats) in zip(axes3, families.items()):
        fam_data = results_dc[results_dc["feature"].isin(fam_feats)].sort_values("loo_delta", ascending=True)
        if len(fam_data) == 0:
            continue
        y_pos = range(len(fam_data))
        bc = ["#3fb950" if d >= 0 else "#f85149" for d in fam_data["loo_delta"]]
        ax.barh(y_pos, fam_data["loo_delta"].values, color=bc, alpha=0.7)
        ax.set_yticks(list(y_pos))
        short = [f.replace("career_", "c:").replace("best2_", "b2:").replace("best1_", "b1:").replace("peak_", "pk:")
                 for f in fam_data["feature"]]
        ax.set_yticklabels(short, fontsize=7)
        ax.axvline(0, color="#f0883e", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("LOO-AUC Delta (vs DC-only)")
        ax.set_title(fam_name, fontweight="bold", fontsize=11)

    fig3.suptitle("LOO-AUC Delta by Feature Family (DC base)", fontsize=13, fontweight="bold")
    path3 = os.path.join(charts_dir, "qbr_engineering_families.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path3}")
    plt.close(fig3)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 74)
    print("  TARGET OUTCOME FEATURE ENGINEERING")
    print("  Replacing career_targeted_qb_rating (and possibly best2_catch_pct_adot_adj)")
    print("=" * 74)

    # --- Load data ---
    print("\n[1/6] Loading grades data...")
    ag = load_grades()
    qual = ag[ag["routes"] >= 200].copy()
    print(f"  Total seasons: {len(ag)}, qualified (200+ routes): {len(qual)}")

    # --- Fit regressions ---
    print("\n[2/6] Fitting aDOT adjustment regressions...")
    adot_catch_coef, adot_qbr_coef, adot_ypr_coef = fit_regressions(ag)
    print(f"  QBR ~ aDOT: QBR = {adot_qbr_coef[0]:.2f} * aDOT + {adot_qbr_coef[1]:.2f}")
    print(f"  Catch% ~ aDOT: CP = {adot_catch_coef[0]:.2f} * aDOT + {adot_catch_coef[1]:.2f}")

    # --- Engineer season features ---
    print("\n[3/6] Engineering season-level features...")
    qual = engineer_season_features(qual, adot_catch_coef, adot_qbr_coef, adot_ypr_coef)

    # --- Aggregate per player ---
    print("\n[4/6] Aggregating per player...")
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_num"] = df["computed_tier"].map(TIER_ORDER)
    df["_join_key"] = df["name"].apply(normalize_name)

    ages = pd.read_csv(os.path.join(DATA_DIR, "draft_ages.csv"))
    ages["birthdate"] = pd.to_datetime(ages["birthdate"])
    birth_lookup = dict(zip(zip(ages["name"], ages["draft_year"]), ages["birthdate"]))

    results = []
    for _, row in df.iterrows():
        birthdate = birth_lookup.get((row["name"], row["draft_year"]))
        res = aggregate_player_engineered(
            normalize_name(row["name"]), row["draft_year"], birthdate, qual
        )
        results.append(res)

    eng_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), eng_df.reset_index(drop=True)], axis=1)
    df = df.copy()

    # --- Build supervised composites ---
    print("\n[4b/7] Building supervised composites...")
    sup_composites = build_supervised_composites(df, qual, birth_lookup)
    for col_name, series in sup_composites.items():
        df[col_name] = np.nan
        df.loc[series.index, col_name] = series.values
        n_valid = series.notna().sum()
        print(f"  {col_name}: n={n_valid}")

    # Save candidates
    out_path = os.path.join(DATA_DIR, "outputs", "qbr_engineering_candidates.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(eng_df.columns)} engineered + {len(sup_composites)} supervised columns to {out_path}")

    # Identify all candidate features (new + incumbents + supervised)
    new_features = [c for c in eng_df.columns if eng_df[c].notna().sum() > 50]
    sup_features = [c for c in sup_composites.keys() if df[c].notna().sum() > 50]
    all_candidates = INCUMBENT_FEATURES + new_features + sup_features
    print(f"  Total candidates to test: {len(all_candidates)}")

    # --- Context 1: DC-only base ---
    print("\n[5/7] Running 7-part analysis...")
    results_dc = run_analysis(
        all_candidates, df,
        base_features=["draft_capital"],
        label="CONTEXT 1: DC-only base"
    )

    # --- Context 2: Full model base (anchors minus QBR/CPA) ---
    results_full = run_analysis(
        all_candidates, df,
        base_features=ANCHOR_FEATURES,
        label="CONTEXT 2: Full model base (4 anchors)"
    )

    # --- Combination testing ---
    print("\n[6/7] Testing feature combinations...")

    # Pick top candidates from each context for combo testing
    if len(results_dc) > 0 and len(results_full) > 0:
        # Merge rankings
        dc_top = set(results_dc.nlargest(8, "loo_delta")["feature"].tolist())
        full_top = set(results_full.nlargest(8, "loo_delta")["feature"].tolist())
        # Also include features with high bootstrap in full context
        full_boot = set(results_full[results_full["boot_pct_pos"] > 0.65]["feature"].tolist())
        top_pool = list(dc_top | full_top | full_boot | set(INCUMBENT_FEATURES))
        # Cap at 10 to keep combinations manageable
        top_pool = top_pool[:12]
        print(f"  Top candidate pool for combos: {top_pool}")

        combo_results = test_combinations(top_pool, df, label="Replacement combinations")
    else:
        combo_results = pd.DataFrame()

    # --- Visualizations ---
    print("\n[7/7] Generating visualizations...")
    generate_visualizations(results_dc, results_full, combo_results, df)

    # --- Save analysis results ---
    if len(results_dc) > 0:
        results_dc.to_csv(os.path.join(DATA_DIR, "outputs", "qbr_eng_results_dc.csv"), index=False)
    if len(results_full) > 0:
        results_full.to_csv(os.path.join(DATA_DIR, "outputs", "qbr_eng_results_full.csv"), index=False)
    if len(combo_results) > 0:
        combo_results.to_csv(os.path.join(DATA_DIR, "outputs", "qbr_eng_combos.csv"), index=False)

    print("\n" + "=" * 74)
    print("  DONE. Output files:")
    print(f"    {os.path.join(DATA_DIR, 'outputs', 'qbr_engineering_candidates.csv')}")
    print(f"    {os.path.join(DATA_DIR, 'outputs', 'qbr_eng_results_dc.csv')}")
    print(f"    {os.path.join(DATA_DIR, 'outputs', 'qbr_eng_results_full.csv')}")
    print(f"    {os.path.join(DATA_DIR, 'outputs', 'qbr_eng_combos.csv')}")
    print(f"    {os.path.join(DATA_DIR, 'charts', 'qbr_engineering_candidates.png')}")
    print(f"    {os.path.join(DATA_DIR, 'charts', 'qbr_engineering_combos.png')}")
    print(f"    {os.path.join(DATA_DIR, 'charts', 'qbr_engineering_families.png')}")
    print("=" * 74)


if __name__ == "__main__":
    main()
