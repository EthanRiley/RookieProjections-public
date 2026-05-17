#!/usr/bin/env python3
"""
Focused test: catch_pct_adot_adj - drop_rate as a replacement feature.

Engineers several variants of "aDOT-adjusted catch% minus drops" and tests
them in the full 7-part protocol + ordinal combo scoring.

Variants:
  - catch_pct_adot_adj_minus_drops: direct subtraction (double-penalizes drops)
  - catch_pct_adot_adj_minus_2x_drops: stronger drop penalty
  - catch_pct_adot_adj_minus_drops_graduated: with age adjustment
"""

import os
import sys
import warnings
import re

import numpy as np
import pandas as pd
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

ANCHOR_FEATURES = [
    "draft_capital",
    "best1_yprr_graduated",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

V11_FEATURES = ANCHOR_FEATURES + [
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


def load_grades():
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


def engineer_cpaa_minus_drops(qual, adot_catch_coef):
    """Add catch_pct_adot_adj_minus_drops variants at season level."""
    q = qual.copy()

    # Recompute catch_pct_adot_adj for this qualified subset
    q["catch_pct_adot_adj"] = q["caught_percent"] - np.polyval(adot_catch_coef, q["avg_depth_of_target"])

    # Core feature: aDOT-adjusted catch% minus drop rate
    q["cpaa_minus_drops"] = q["catch_pct_adot_adj"] - q["drop_rate"]

    # Stronger drop penalty variant
    q["cpaa_minus_2x_drops"] = q["catch_pct_adot_adj"] - 2 * q["drop_rate"]

    # Also: raw catch_minus_drops for comparison (already in prior analysis)
    q["catch_minus_drops"] = q["caught_percent"] - q["drop_rate"]

    # Clean catch rate (for comparison)
    non_ct = q["targets"] - q["contested_targets"]
    non_cr = q["receptions"] - q["contested_receptions"]
    q["clean_catch_rate"] = np.where(non_ct > 0, non_cr / non_ct * 100, np.nan)

    return q


def _wavg(series, weights):
    mask = series.notna() & pd.Series(weights).notna()
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    return np.average(series[mask], weights=weights[mask])


def aggregate_new_features(player_key, draft_year, birthdate, qual_seasons):
    """Aggregate the new features for one player."""
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

    # P5-filtered eligible seasons
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
    new_cols = ["cpaa_minus_drops", "cpaa_minus_2x_drops", "catch_minus_drops",
                "clean_catch_rate", "catch_pct_adot_adj"]

    # Career aggregation (target-weighted)
    for feat in new_cols:
        if feat in seasons.columns:
            val = _wavg(seasons[feat], seasons["targets"].values)
            if pd.notna(val):
                result[f"career_{feat}"] = round(val, 4)

    # Best2 aggregation (target-weighted)
    for feat in new_cols:
        if feat in best2.columns:
            val = _wavg(best2[feat], best2["targets"].values)
            if pd.notna(val):
                result[f"best2_{feat}"] = round(val, 4)

    # Best1 values
    for feat in new_cols:
        if feat in best1_row.index and pd.notna(best1_row[feat]):
            result[f"best1_{feat}"] = round(float(best1_row[feat]), 4)

    # Graduated variants (best1, age-adjusted)
    if birthdate is not None and pd.notna(birthdate) and grades.notna().any():
        age = get_age_on_sept1(birthdate, best1_row["grade_year"])
        if age is not None:
            for feat, center in [("cpaa_minus_drops", 0), ("catch_pct_adot_adj", 0)]:
                raw = best1_row.get(feat)
                if pd.notna(raw):
                    for (lo, hi), mult in GRADUATED_ADJ.items():
                        if lo <= age < hi:
                            adjusted = (raw - center) * mult + center
                            result[f"best1_{feat}_graduated"] = round(adjusted, 4)
                            break

    # Senior-discounted variants (career, best2)
    if birthdate is not None and pd.notna(birthdate):
        for prefix, source in [("career", seasons), ("best2", best2)]:
            disc = source.copy()
            for idx, row in disc.iterrows():
                age = get_age_on_sept1(birthdate, row["grade_year"])
                if age is not None and age >= SENIOR_AGE_THRESHOLD:
                    for col in ["cpaa_minus_drops", "catch_pct_adot_adj"]:
                        if col in disc.columns and pd.notna(disc.at[idx, col]):
                            disc.at[idx, col] -= SENIOR_DISCOUNT_PP
            for feat in ["cpaa_minus_drops", "catch_pct_adot_adj"]:
                if feat in disc.columns:
                    val = _wavg(disc[feat], disc["targets"].values)
                    if pd.notna(val):
                        result[f"{prefix}_{feat}_sr_disc"] = round(val, 4)

    return result


# ============================================================
# Analysis functions (from target_outcome_engineering.py)
# ============================================================

def _loo_auc(d, features, years):
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


def _loo_ordinal_scores(d, features, years, n_tiers=6):
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
        return np.nan, np.nan, np.nan, np.nan, np.nan

    idx = mask_predicted
    n_pred = idx.sum()
    tier_probs = np.zeros((n_pred, n_tiers))

    cp = cum_probs[idx]
    for ti in range(len(thresholds) - 1, 0, -1):
        cp[:, ti - 1] = np.maximum(cp[:, ti - 1], cp[:, ti])

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

    auc_elite = roc_auc_score((y_pred >= 3).astype(int), cp[:, 2]) if (y_pred >= 3).sum() > 0 and (y_pred < 3).sum() > 0 else np.nan
    auc_stud = roc_auc_score((y_pred >= 4).astype(int), cp[:, 3]) if (y_pred >= 4).sum() > 0 and (y_pred < 4).sum() > 0 else np.nan
    auc_starter = roc_auc_score((y_pred >= 2).astype(int), cp[:, 1]) if (y_pred >= 2).sum() > 0 and (y_pred < 2).sum() > 0 else np.nan

    return ll, brier, auc_elite, auc_stud, auc_starter


def run_7part(candidates, df, base_features, label=""):
    """Run 7-part analysis."""
    valid_candidates = [c for c in candidates if c in df.columns and df[c].notna().sum() > 50]
    if not valid_candidates:
        print(f"  No valid candidates for {label}")
        return pd.DataFrame()

    all_cols = base_features + valid_candidates + ["tier_num", "draft_year"]
    d = df.dropna(subset=[c for c in all_cols if c in df.columns]).copy()
    y = d["tier_num"].values
    hit = (y >= 3).astype(int)
    years = sorted(d["draft_year"].unique())

    print(f"\n{'=' * 80}")
    print(f"  {label} | n={len(d)} | base={base_features}")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X_base = scaler.fit_transform(d[base_features].values)
    ridge = Ridge(alpha=1.0).fit(X_base, y)
    residuals = y - ridge.predict(X_base)
    base_auc = _loo_auc(d, base_features, years)

    results = []
    for c in valid_candidates:
        row = {"feature": c}

        # Part 1: Univariate
        sp, _ = spearmanr(d[c].values, y)
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
        sp_res, _ = spearmanr(d[c].values, residuals)
        row["residual"] = round(sp_res, 3)

        # Part 4: Max collinearity
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
        for C in [0.01, 0.1, 1.0]:
            X_all = StandardScaler().fit_transform(d[base_features + [c]].values)
            lr = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=C, max_iter=10000, random_state=42,
            )
            lr.fit(X_all, hit)
            if abs(lr.coef_[0, -1]) > 1e-6:
                enet_count += 1
        row["enet_survive"] = enet_count

        results.append(row)

    results_df = pd.DataFrame(results)

    print(f"\n  {'Feature':<45s} {'Sp':>6s} {'AUC':>6s} {'Drift':>6s} {'Resid':>6s} "
          f"{'Boot%':>6s} {'LOO':>6s} {'Delta':>7s} {'Enet':>5s} {'Collin':>6s}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*6}")
    for _, r in results_df.sort_values("loo_delta", ascending=False).iterrows():
        print(f"  {r['feature']:<45s} {r['spearman']:>+6.3f} {r['auc']:>6.3f} "
              f"{r['era_drift']:>6.3f} {r['residual']:>+6.3f} {r['boot_pct_pos']:>6.1%} "
              f"{r['loo_auc']:>6.3f} {r['loo_delta']:>+7.3f} {r['enet_survive']:>3d}/3 "
              f"{r['max_collinearity']:>6.3f}")

    return results_df


def test_combos(new_feats, df):
    """Test combinations with ordinal scoring."""
    print(f"\n{'=' * 80}")
    print(f"  COMBINATION TESTS (ordinal LogLoss + Brier + AUC)")
    print(f"{'=' * 80}")

    # Only require core columns to be non-null; handle NaN per-combo
    core_cols = ANCHOR_FEATURES + ["tier_num", "draft_year"]
    d = df.dropna(subset=core_cols).copy()
    years = sorted(d["draft_year"].unique())
    print(f"  n={len(d)} | years={[int(y) for y in years]}")

    combos = {}

    # Baseline: current v11
    combos["v11 (career_targeted_qb_rating + best2_catch_pct_adot_adj)"] = V11_FEATURES[:]

    # Drop QBR only (keep aDOT-adj catch%)
    combos["v11 minus QBR"] = [f for f in V11_FEATURES if f != "career_targeted_qb_rating"]

    # Drop both
    combos["4 anchors only"] = ANCHOR_FEATURES[:]

    # Replace QBR with each new feature (keeping aDOT-adj catch%)
    for cand in new_feats:
        if cand in d.columns:
            feats = [f for f in V11_FEATURES if f != "career_targeted_qb_rating"] + [cand]
            combos[f"QBR => {cand}"] = feats

    # Replace aDOT-adj catch% with each new feature (keeping QBR)
    for cand in new_feats:
        if cand in d.columns:
            feats = [f for f in V11_FEATURES if f != "best2_catch_pct_adot_adj"] + [cand]
            combos[f"catch%_adot => {cand}"] = feats

    # Replace BOTH with single new feature
    for cand in new_feats:
        if cand in d.columns:
            feats = ANCHOR_FEATURES + [cand]
            combos[f"both => {cand}"] = feats

    # Replace BOTH with two new features (pairs of new feats)
    for i, c1 in enumerate(new_feats):
        for c2 in new_feats[i + 1:]:
            if c1 in d.columns and c2 in d.columns:
                feats = ANCHOR_FEATURES + [c1, c2]
                combos[f"both => {c1} + {c2}"] = feats

    # Key comparisons from prior analysis
    if "best1_clean_catch_rate" in d.columns:
        combos["both => best1_clean_catch_rate (prior best)"] = ANCHOR_FEATURES + ["best1_clean_catch_rate"]
    if "best1_catch_minus_drops" in d.columns:
        combos["both => best1_catch_minus_drops (prior best LL)"] = ANCHOR_FEATURES + ["best1_catch_minus_drops"]

    print(f"\n  {'Combination':<65s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} "
          f"{'>=Stud':>8s} {'>=Start':>8s} {'#F':>4s}")
    print(f"  {'-'*65} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    results = []
    for name, feats in combos.items():
        valid_feats = [f for f in feats if f in d.columns]
        # Subset to rows with non-null values for this combo's features
        d_combo = d.dropna(subset=valid_feats)
        combo_years = sorted(d_combo["draft_year"].unique())
        ll, brier, auc_e, auc_s, auc_st = _loo_ordinal_scores(d_combo, valid_feats, combo_years)
        row = {
            "combo": name, "features": str(valid_feats), "n_feats": len(valid_feats),
            "log_loss": round(ll, 3) if pd.notna(ll) else np.nan,
            "brier": round(brier, 3) if pd.notna(brier) else np.nan,
            "elite_auc": round(auc_e, 3) if pd.notna(auc_e) else np.nan,
            "stud_auc": round(auc_s, 3) if pd.notna(auc_s) else np.nan,
            "starter_auc": round(auc_st, 3) if pd.notna(auc_st) else np.nan,
        }
        results.append(row)
        print(f"  {name:<65s} {row['log_loss']:>8.3f} {row['brier']:>8.3f} "
              f"{row['elite_auc']:>8.3f} {row['stud_auc']:>8.3f} "
              f"{row['starter_auc']:>8.3f} {row['n_feats']:>4d}")

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("  CATCH% aDOT ADJ MINUS DROPS: Feature Engineering Test")
    print("=" * 80)

    # Load grades
    print("\n[1/5] Loading grades data...")
    ag = load_grades()
    qual = ag[ag["routes"] >= 200].copy()
    print(f"  Qualified seasons: {len(qual)}")

    # Fit aDOT regression
    print("\n[2/5] Fitting aDOT regression...")
    cp, adot = ag["caught_percent"], ag["avg_depth_of_target"]
    m = cp.notna() & adot.notna()
    adot_catch_coef = np.polyfit(adot[m].values, cp[m].values, 1)
    print(f"  Catch% ~ aDOT: slope={adot_catch_coef[0]:.3f}, intercept={adot_catch_coef[1]:.3f}")

    # Engineer at season level
    print("\n[3/5] Engineering season features...")
    qual = engineer_cpaa_minus_drops(qual, adot_catch_coef)

    # Show season-level stats
    for col in ["cpaa_minus_drops", "cpaa_minus_2x_drops", "catch_pct_adot_adj", "catch_minus_drops"]:
        vals = qual[col].dropna()
        print(f"  {col}: n={len(vals)}, mean={vals.mean():.2f}, std={vals.std():.2f}")

    # Show correlations between variants at season level
    print("\n  Season-level correlations:")
    season_feats = ["cpaa_minus_drops", "catch_pct_adot_adj", "catch_minus_drops",
                    "caught_percent", "drop_rate", "clean_catch_rate",
                    "targeted_qb_rating", "yprr"]
    for f1 in ["cpaa_minus_drops"]:
        for f2 in season_feats:
            if f1 != f2:
                valid = qual[[f1, f2]].dropna()
                if len(valid) > 30:
                    r, _ = spearmanr(valid[f1], valid[f2])
                    print(f"    {f1} vs {f2}: rho={r:+.3f}")

    # Load master + aggregate
    print("\n[4/5] Aggregating per player...")
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_num"] = df["computed_tier"].map(TIER_ORDER)
    df["_join_key"] = df["name"].apply(normalize_name)

    ages = pd.read_csv(os.path.join(DATA_DIR, "draft_ages.csv"))
    ages["birthdate"] = pd.to_datetime(ages["birthdate"])
    birth_lookup = dict(zip(zip(ages["name"], ages["draft_year"]), ages["birthdate"]))

    results = []
    for _, row in df.iterrows():
        birthdate = birth_lookup.get((row["name"], row["draft_year"]))
        res = aggregate_new_features(
            normalize_name(row["name"]), row["draft_year"], birthdate, qual
        )
        results.append(res)

    eng_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), eng_df.reset_index(drop=True)], axis=1)

    # Deduplicate columns — keep master (first) version for existing columns
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Show available new features
    new_cols = [c for c in eng_df.columns if eng_df[c].notna().sum() > 50]
    print(f"  New features available: {new_cols}")

    for c in new_cols:
        n = df[c].notna().sum()
        print(f"    {c}: n={n}, mean={df[c].mean():.2f}")

    # ======= 7-part analysis =======
    print("\n[5/5] Running analysis...")

    # All candidates: new features + incumbents + prior best
    candidates = new_cols + [
        "career_targeted_qb_rating", "best2_catch_pct_adot_adj",
        "best1_clean_catch_rate", "best1_catch_minus_drops",
    ]
    candidates = [c for c in candidates if c in df.columns]

    # DC-only context
    results_dc = run_7part(candidates, df, base_features=["draft_capital"],
                           label="DC-ONLY BASE")

    # Full model context (4 anchors)
    results_full = run_7part(candidates, df, base_features=ANCHOR_FEATURES,
                             label="FULL MODEL BASE (4 anchors)")

    # ======= Combination tests =======
    new_feats_for_combos = [c for c in new_cols if c in df.columns and df[c].notna().sum() > 100]
    # Add prior best for comparison
    for extra in ["best1_clean_catch_rate", "best1_catch_minus_drops"]:
        if extra in df.columns and extra not in new_feats_for_combos:
            new_feats_for_combos.append(extra)

    combo_results = test_combos(new_feats_for_combos, df)

    # Save results
    out_dir = os.path.join(DATA_DIR, "outputs")
    combo_results.to_csv(os.path.join(out_dir, "cpaa_minus_drops_combos.csv"), index=False)
    if len(results_dc) > 0:
        results_dc.to_csv(os.path.join(out_dir, "cpaa_minus_drops_dc.csv"), index=False)
    if len(results_full) > 0:
        results_full.to_csv(os.path.join(out_dir, "cpaa_minus_drops_full.csv"), index=False)

    print(f"\n  Results saved to {out_dir}/cpaa_minus_drops_*.csv")

    # ======= Summary =======
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if len(combo_results) > 0:
        v11_row = combo_results[combo_results["combo"].str.contains("v11.*career")]
        if len(v11_row) > 0:
            v11 = v11_row.iloc[0]
            print(f"\n  v11 baseline: LogLoss={v11['log_loss']:.3f} Brier={v11['brier']:.3f} "
                  f"Elite={v11['elite_auc']:.3f} Stud={v11['stud_auc']:.3f} Starter={v11['starter_auc']:.3f}")

        print(f"\n  Best by LogLoss:")
        best_ll = combo_results.nsmallest(3, "log_loss")
        for _, r in best_ll.iterrows():
            print(f"    {r['combo']:<60s} LL={r['log_loss']:.3f}")

        print(f"\n  Best by Brier:")
        best_br = combo_results.nsmallest(3, "brier")
        for _, r in best_br.iterrows():
            print(f"    {r['combo']:<60s} Br={r['brier']:.3f}")

        print(f"\n  Best by >=Elite AUC:")
        best_auc = combo_results.nlargest(3, "elite_auc")
        for _, r in best_auc.iterrows():
            print(f"    {r['combo']:<60s} AUC={r['elite_auc']:.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
