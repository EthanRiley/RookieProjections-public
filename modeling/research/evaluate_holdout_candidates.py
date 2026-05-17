#!/usr/bin/env python3
"""
Production holdout evaluation of candidate feature sets through the full
Bayesian ordinal + XGBoost cumulative link ensemble (60/40).

Tests:
  A: v11 baseline (6 feats)
  B: 5 feats (pg_yprr_grad + pg_cpaa_grad, no best2_cpaa)
  C: 6 feats (B + best2_catch_pct_adot_adj)
  D: 6 feats (B + best1_grades_pass_route)

Train: 2018-2021, Holdout: 2022-2024
Ensemble: 60% Bayesian + 40% XGBoost (same as v11)
"""

import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

HOLDOUT_YEARS = [2022, 2023, 2024]
W_BAYES = 0.60
W_XGB = 0.40

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
QUALITY_GATE = 80.0


def normalize_name(name):
    n = SUFFIXES_RE.sub("", str(name)).strip()
    n = n.replace(".", "").replace("'", "").lower()
    return " ".join(n.split())


def get_age_on_sept1(birthdate, year):
    if birthdate is None or pd.isna(birthdate):
        return None
    sept1 = pd.Timestamp(f"{int(year)}-09-01")
    return (sept1 - birthdate).days / 365.25


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


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
    ]
    for c in num_cols:
        if c in ag.columns:
            ag[c] = pd.to_numeric(ag[c], errors="coerce")
    return ag


def aggregate_pg_features(player_key, draft_year, birthdate, qual_seasons, adot_coef):
    """Compute peak-gated YPRR graduated, CPAA graduated, and best1_grades_pass_route."""
    seasons = qual_seasons[
        (qual_seasons["_join_key"] == player_key) &
        (qual_seasons["grade_year"] <= draft_year)
    ].copy()

    if len(seasons) > 0:
        excl = seasons.apply(
            lambda r: (r["_join_key"], r.get("team_name", ""), r.get("grade_year", 0))
            in SEASON_EXCLUSIONS, axis=1
        )
        seasons = seasons[~excl]
    if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
        min_year = birthdate.year + 18
        seasons = seasons[seasons["grade_year"] >= min_year]
    if len(seasons) > 0:
        seasons = seasons[seasons["grade_year"] >= draft_year - 5]
    if len(seasons) == 0:
        return {}

    # P5 filter
    p5 = seasons[seasons["team_name"].isin(P5_TEAMS)] if "team_name" in seasons.columns else seasons
    eligible = p5 if len(p5) >= 1 else seasons

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if not grades.notna().any():
        return {}

    result = {}

    # Best1 row (highest grade)
    best1_row = eligible.loc[grades.idxmax()]

    # best1_grades_pass_route
    gpr = pd.to_numeric(best1_row.get("grades_pass_route", np.nan), errors="coerce")
    if pd.notna(gpr):
        result["best1_grades_pass_route"] = round(float(gpr), 2)

    # Quality-gated seasons
    gated = eligible[grades >= QUALITY_GATE]
    has_gated = len(gated) > 0

    # Compute YPRR per season
    for idx, row in eligible.iterrows():
        yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
        rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
        eligible.at[idx, "yprr_raw"] = yards / rts if rts > 0 else np.nan

    # Compute CPAA per season
    eligible["catch_pct_adot_adj"] = eligible["caught_percent"] - np.polyval(
        adot_coef, eligible["avg_depth_of_target"]
    )

    if has_gated:
        for idx, row in gated.iterrows():
            yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
            rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
            gated.at[idx, "yprr_raw"] = yards / rts if rts > 0 else np.nan
        gated["catch_pct_adot_adj"] = gated["caught_percent"] - np.polyval(
            adot_coef, gated["avg_depth_of_target"]
        )

    # --- Peak-gated YPRR ---
    best1_yprr_raw = eligible.at[grades.idxmax(), "yprr_raw"] if "yprr_raw" in eligible.columns else np.nan

    if has_gated and "yprr_raw" in gated.columns:
        gated_yprr = gated["yprr_raw"].dropna()
        if len(gated_yprr) > 0:
            pg_yprr_idx = gated_yprr.idxmax()
            pg_yprr_val = float(gated_yprr.max())
            pg_yprr_row = gated.loc[pg_yprr_idx]
        else:
            pg_yprr_val = best1_yprr_raw
            pg_yprr_row = best1_row
    else:
        pg_yprr_val = best1_yprr_raw
        pg_yprr_row = best1_row

    # --- Peak-gated CPAA ---
    best1_cpaa = eligible.at[grades.idxmax(), "catch_pct_adot_adj"] if "catch_pct_adot_adj" in eligible.columns else np.nan

    if has_gated and "catch_pct_adot_adj" in gated.columns:
        gated_cpaa = gated["catch_pct_adot_adj"].dropna()
        if len(gated_cpaa) > 0:
            pg_cpaa_idx = gated_cpaa.idxmax()
            pg_cpaa_val = float(gated_cpaa.max())
            pg_cpaa_row = gated.loc[pg_cpaa_idx]
        else:
            pg_cpaa_val = best1_cpaa
            pg_cpaa_row = best1_row
    else:
        pg_cpaa_val = best1_cpaa
        pg_cpaa_row = best1_row

    # --- Graduated adjustments ---
    if birthdate is not None and pd.notna(birthdate):
        for prefix, val, row_source, center in [
            ("pg_yprr", pg_yprr_val, pg_yprr_row, 0),
            ("pg_catch_pct_adot_adj", pg_cpaa_val, pg_cpaa_row, 0),
        ]:
            if pd.isna(val):
                continue
            yr = row_source.get("grade_year") if hasattr(row_source, "get") else None
            if yr is None or pd.isna(yr):
                continue
            age = get_age_on_sept1(birthdate, yr)
            if age is None:
                continue
            for (lo, hi), mult in GRADUATED_ADJ.items():
                if lo <= age < hi:
                    adjusted = (val - center) * mult + center
                    result[f"{prefix}_graduated"] = round(adjusted, 4)
                    break

    return result


# ============================================================
# Model training functions (same as evaluate_holdout_v11.py)
# ============================================================

def train_xgb(X_train, y_train, X_hold):
    cum_probs = np.zeros((len(X_hold), len(THRESHOLDS)))
    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train >= threshold).astype(int)
        pos = y_bin.sum()
        scale = (len(y_bin) - pos) / max(pos, 1)
        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, random_state=42, eval_metric="logloss",
        )
        min_class = min(y_bin.sum(), len(y_bin) - y_bin.sum())
        cv_folds = min(5, max(2, min_class))
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds)
        calibrated.fit(X_train, y_bin)
        cum_probs[:, t_idx] = calibrated.predict_proba(X_hold)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((len(X_hold), N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def train_bayesian(X_college_train, dc_train, y_train, X_college_hold, dc_hold, use_dc):
    n_college = X_college_train.shape[1]

    with pm.Model() as model:
        beta_college = pm.Normal("beta_college", mu=0.0, sigma=0.5, shape=n_college)
        eta = pt.dot(X_college_train, beta_college)
        if use_dc:
            beta_dc = pm.Normal("beta_dc", mu=0.5, sigma=0.3)
            eta = eta + beta_dc * dc_train
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-2, 3, N_CUTPOINTS),
            sigma=1.5, shape=N_CUTPOINTS,
            transform=pm.distributions.transforms.ordered,
        )
        pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y_train)

    with model:
        trace = pm.sample(
            3000, tune=2000, chains=4, cores=1,
            random_seed=42, progressbar=False, target_accept=0.9,
        )

    beta_college_samples = trace.posterior["beta_college"].values.reshape(-1, n_college)
    cutpoints_samples = trace.posterior["cutpoints"].values.reshape(-1, N_CUTPOINTS)
    n_samples = len(cutpoints_samples)
    n_obs = X_college_hold.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    has_dc = "beta_dc" in trace.posterior
    if has_dc:
        beta_dc_samples = trace.posterior["beta_dc"].values.flatten()

    for i in range(n_samples):
        eta = X_college_hold @ beta_college_samples[i]
        if has_dc:
            eta = eta + beta_dc_samples[i] * dc_hold
        cum_probs = 1.0 / (1.0 + np.exp(-(cutpoints_samples[i] - eta[:, None])))
        sample_probs = np.zeros((n_obs, N_TIERS))
        sample_probs[:, 0] = cum_probs[:, 0]
        for k in range(1, N_CUTPOINTS):
            sample_probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        sample_probs[:, N_TIERS - 1] = 1 - cum_probs[:, N_CUTPOINTS - 1]
        tier_probs += sample_probs

    tier_probs /= n_samples
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs /= tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def evaluate_config(train_df, holdout_df, college_features, config_name):
    """Run full Bayesian+XGBoost ensemble for a feature configuration."""
    print(f"\n{'='*70}")
    print(f"  {config_name}")
    print(f"  Features: {['draft_capital'] + college_features}")
    print(f"{'='*70}")

    all_features = ["draft_capital"] + college_features
    y_train = train_df["tier_ordinal"].values
    y_hold = holdout_df["tier_ordinal"].values

    # XGBoost (full)
    print(f"  Training XGBoost Full...")
    xgb_full = train_xgb(
        train_df[all_features].values, y_train,
        holdout_df[all_features].values
    )

    # XGBoost (college only)
    print(f"  Training XGBoost College...")
    xgb_college = train_xgb(
        train_df[college_features].values, y_train,
        holdout_df[college_features].values
    )

    # Bayesian (full)
    print(f"  Training Bayesian Full...")
    scaler_full = StandardScaler()
    X_c_train = scaler_full.fit_transform(train_df[college_features].values)
    X_c_hold = scaler_full.transform(holdout_df[college_features].values)
    bayes_full = train_bayesian(
        X_c_train, train_df["draft_capital"].values, y_train,
        X_c_hold, holdout_df["draft_capital"].values, use_dc=True
    )

    # Bayesian (college only)
    print(f"  Training Bayesian College...")
    scaler_col = StandardScaler()
    X_cc_train = scaler_col.fit_transform(train_df[college_features].values)
    X_cc_hold = scaler_col.transform(holdout_df[college_features].values)
    bayes_college = train_bayesian(
        X_cc_train, None, y_train,
        X_cc_hold, None, use_dc=False
    )

    # Ensemble
    def blend(b, x):
        combo = W_BAYES * b + W_XGB * x
        return combo / combo.sum(axis=1, keepdims=True)

    full_probs = blend(bayes_full, xgb_full)
    college_probs = blend(bayes_college, xgb_college)

    # Evaluate
    y_onehot = np.zeros((len(y_hold), N_TIERS))
    y_onehot[np.arange(len(y_hold)), y_hold] = 1

    results = {}
    for label, probs in [("Full", full_probs), ("College", college_probs)]:
        ll = log_loss(y_onehot, probs)
        brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
        aucs = {}
        for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
            y_bin = (y_hold >= threshold).astype(int)
            pred = probs[:, threshold:].sum(axis=1)
            aucs[tlabel] = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        results[label] = {"ll": ll, "brier": brier, "aucs": aucs}

    # Print
    for label in ["Full", "College"]:
        r = results[label]
        print(f"\n  {label}: LogLoss={r['ll']:.4f}  Brier={r['brier']:.4f}")
        for tlabel in [">=Elite", ">=Stud", ">=Starter", ">=LW"]:
            print(f"    {tlabel} AUC: {r['aucs'][tlabel]:.3f}")

    # Also evaluate individual models
    for model_name, probs in [("Bayesian Full", bayes_full), ("XGBoost Full", xgb_full),
                               ("Bayesian College", bayes_college), ("XGBoost College", xgb_college)]:
        ll = log_loss(y_onehot, probs)
        brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
        print(f"  {model_name}: LL={ll:.4f} Brier={brier:.4f}")

    return results, full_probs, college_probs


def main():
    print("=" * 70)
    print("  PRODUCTION HOLDOUT EVALUATION: CANDIDATE FEATURE SETS")
    print("  Train: 2018-2021 | Holdout: 2022-2024")
    print("  Ensemble: 60% Bayesian + 40% XGBoost")
    print("=" * 70)

    # Load grades + master
    print("\n[1/3] Loading and engineering data...")
    ag = load_grades()
    qual = ag[ag["routes"] >= 200].copy()

    # aDOT regression
    cp, adot = ag["caught_percent"], ag["avg_depth_of_target"]
    m = cp.notna() & adot.notna()
    adot_coef = np.polyfit(adot[m].values, cp[m].values, 1)

    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df["_join_key"] = df["name"].apply(normalize_name)

    ages = pd.read_csv(os.path.join(DATA_DIR, "draft_ages.csv"))
    ages["birthdate"] = pd.to_datetime(ages["birthdate"])
    birth_lookup = dict(zip(zip(ages["name"], ages["draft_year"]), ages["birthdate"]))

    # Engineer peak-gated features
    print("  Engineering peak-gated features...")
    eng_results = []
    for _, row in df.iterrows():
        birthdate = birth_lookup.get((row["name"], row["draft_year"]))
        res = aggregate_pg_features(
            normalize_name(row["name"]), row["draft_year"], birthdate, qual, adot_coef
        )
        eng_results.append(res)

    eng_df = pd.DataFrame(eng_results)
    df = pd.concat([df.reset_index(drop=True), eng_df.reset_index(drop=True)], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # Recompute draft capital with log scaling
    df["draft_capital"] = df["pick"].apply(dc_log)

    # Check coverage
    for col in ["pg_yprr_graduated", "pg_catch_pct_adot_adj_graduated", "best1_grades_pass_route"]:
        n = df[col].notna().sum() if col in df.columns else 0
        print(f"  {col}: {n}/{len(df)} non-null")

    # Define configs
    configs = {
        "A: v11 baseline": [
            "best1_yprr_graduated",
            "career_targeted_qb_rating",
            "best2_catch_pct_adot_adj",
            "best2_contested_catch_rate",
            "best2_avoided_tackles_per_rec",
        ],
        "B: 5-feat (pg_yprr + pg_cpaa)": [
            "pg_yprr_graduated",
            "pg_catch_pct_adot_adj_graduated",
            "best2_contested_catch_rate",
            "best2_avoided_tackles_per_rec",
        ],
        "C: 6-feat (B + best2_cpaa)": [
            "pg_yprr_graduated",
            "pg_catch_pct_adot_adj_graduated",
            "best2_catch_pct_adot_adj",
            "best2_contested_catch_rate",
            "best2_avoided_tackles_per_rec",
        ],
        "D: 6-feat (B + grades_pass_route)": [
            "pg_yprr_graduated",
            "pg_catch_pct_adot_adj_graduated",
            "best1_grades_pass_route",
            "best2_contested_catch_rate",
            "best2_avoided_tackles_per_rec",
        ],
    }

    # For each config, determine which rows have complete data
    print("\n[2/3] Running holdout evaluations...")
    all_results = {}

    for config_name, college_feats in configs.items():
        all_feats = ["draft_capital"] + college_feats
        valid = df.dropna(subset=all_feats + ["tier_ordinal"]).copy()
        valid["tier_ordinal"] = valid["tier_ordinal"].astype(int)

        train = valid[~valid["draft_year"].isin(HOLDOUT_YEARS)]
        holdout = valid[valid["draft_year"].isin(HOLDOUT_YEARS)]

        print(f"\n  Config: {config_name}")
        print(f"  Train: {len(train)} | Holdout: {len(holdout)}")

        results, full_probs, college_probs = evaluate_config(
            train, holdout, college_feats, config_name
        )
        all_results[config_name] = results

    # Summary comparison
    print("\n\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n  {'Config':<40s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} "
          f"{'>=Stud':>8s} {'>=Start':>8s} {'>=LW':>8s}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for config_name, results in all_results.items():
        r = results["Full"]
        print(f"  {config_name:<40s} {r['ll']:>8.4f} {r['brier']:>8.4f} "
              f"{r['aucs']['>=Elite']:>8.3f} {r['aucs']['>=Stud']:>8.3f} "
              f"{r['aucs']['>=Starter']:>8.3f} {r['aucs']['>=LW']:>8.3f}")

    print(f"\n  College-only:")
    print(f"  {'Config':<40s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} "
          f"{'>=Stud':>8s} {'>=Start':>8s} {'>=LW':>8s}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for config_name, results in all_results.items():
        r = results["College"]
        print(f"  {config_name:<40s} {r['ll']:>8.4f} {r['brier']:>8.4f} "
              f"{r['aucs']['>=Elite']:>8.3f} {r['aucs']['>=Stud']:>8.3f} "
              f"{r['aucs']['>=Starter']:>8.3f} {r['aucs']['>=LW']:>8.3f}")

    # Deltas vs v11
    if "A: v11 baseline" in all_results:
        v11 = all_results["A: v11 baseline"]["Full"]
        print(f"\n  Deltas vs v11 (Full):")
        print(f"  {'Config':<40s} {'dLL':>8s} {'dBrier':>8s} {'dElite':>8s} "
              f"{'dStud':>8s} {'dStart':>8s} {'dLW':>8s}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for config_name, results in all_results.items():
            if "v11" in config_name:
                continue
            r = results["Full"]
            dll = r["ll"] - v11["ll"]
            dbr = r["brier"] - v11["brier"]
            dae = r["aucs"][">=Elite"] - v11["aucs"][">=Elite"]
            das = r["aucs"][">=Stud"] - v11["aucs"][">=Stud"]
            dst = r["aucs"][">=Starter"] - v11["aucs"][">=Starter"]
            dlw = r["aucs"][">=LW"] - v11["aucs"][">=LW"]
            print(f"  {config_name:<40s} {dll:>+8.4f} {dbr:>+8.4f} "
                  f"{dae:>+8.3f} {das:>+8.3f} {dst:>+8.3f} {dlw:>+8.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
