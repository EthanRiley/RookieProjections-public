#!/usr/bin/env python3
"""
Holdout evaluation: geomean(pg_yprr, pg_yac_per_game) replacing pg_yprr_graduated.

Same protocol as evaluate_holdout_v12.py: train 2018-2021, test 2022-2024.
60/40 Bayesian/XGBoost ensemble.
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")

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


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


def get_age_mult(birthdate, year):
    if birthdate is None or pd.isna(birthdate) or year is None or pd.isna(year):
        return 1.0
    sept1 = pd.Timestamp(f"{int(year)}-09-01")
    age = (sept1 - birthdate).days / 365.25
    for (lo, hi), mult in GRADUATED_ADJ.items():
        if lo <= age < hi:
            return mult
    return 1.0


def compute_pg_yac_per_game(seasons, birthdate):
    """Peak-gated, age-adjusted YAC per game."""
    from aggregation.aggregate_college_stats import get_player_seasons

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
        yac = pd.to_numeric(row.get("yards_after_catch", 0), errors="coerce") or 0
        games = pd.to_numeric(row.get("player_game_count", 0), errors="coerce") or 0
        if games == 0:
            return np.nan
        raw = yac / games
        mult = get_age_mult(birthdate, row.get("grade_year"))
        return raw * mult

    search_set = gated if len(gated) > 0 else eligible.loc[[grades.idxmax()]]

    best_val = np.nan
    for _, row in search_set.iterrows():
        adj = _get_adj_val(row)
        if pd.notna(adj) and (np.isnan(best_val) or adj > best_val):
            best_val = adj

    return round(best_val, 4) if pd.notna(best_val) else np.nan


def train_xgb(X_train, y_train, X_hold, label):
    print(f"\nTraining XGBoost {label}...")
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


def train_bayesian(X_college_train, dc_train, y_train, X_college_hold, dc_hold, use_dc, label):
    print(f"\nTraining Bayesian {label}...")
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
            random_seed=42, progressbar=True, target_accept=0.9,
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


def evaluate(probs, y_true, label):
    print(f"\n  {label}")
    print(f"  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s}")
    aucs = {}
    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        aucs[tlabel] = auc
        print(f"  {tlabel:<15s} {auc:>8.3f} {brier:>8.4f}")

    y_onehot = np.zeros((len(y_true), 6))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    print(f"\n  LogLoss:  {ll:.4f}")
    print(f"  Brier:    {brier:.4f}")
    return ll, brier, aucs


# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("COMPOSITE HOLDOUT EVALUATION")
print("geomean(pg_yprr, pg_yac_per_game) vs pg_yprr_graduated")
print("=" * 70)

from aggregation.aggregate_college_stats import (
    load_all_grades, build_lookups, aggregate_player, fit_adot_regression,
    get_player_seasons,
)

print("\nLoading grades and aggregating features...")
all_grades = load_all_grades(range(2016, 2026))
birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
adot_coef = fit_adot_regression(all_grades)

df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)

# Re-aggregate peak-gated features + compute pg_yac_per_game
print("Re-aggregating with peak-gated features + pg_yac_per_game...")
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

    # Compute pg_yac_per_game
    birthdate = birth_lookup.get((row["name"], row["draft_year"])) if birth_lookup else None
    seasons = get_player_seasons(all_grades, row["name"], row["draft_year"],
                                  apply_exclusions=True, birthdate=birthdate)
    df.at[df.index[i], "pg_yac_per_game"] = compute_pg_yac_per_game(seasons, birthdate)

# Compute geomean
valid = df["pg_yprr_graduated"].notna() & df["pg_yac_per_game"].notna() & (df["pg_yprr_graduated"] > 0) & (df["pg_yac_per_game"] > 0)
df["geomean_yprr_yac_pg"] = np.nan
df.loc[valid, "geomean_yprr_yac_pg"] = np.sqrt(
    df.loc[valid, "pg_yprr_graduated"] * df.loc[valid, "pg_yac_per_game"]
)

# Recompute draft capital
df["draft_capital"] = df["pick"].apply(dc_log)

# ============================================================
# Run BOTH feature sets through holdout
# ============================================================
CONFIGS = {
    "v12 (pg_yprr_graduated)": {
        "college_features": [
            "pg_yprr_graduated",
            "pg_catch_pct_adot_adj_graduated",
            "best2_contested_catch_rate",
            "best2_avoided_tackles_per_rec",
        ],
    },
    "COMPOSITE (geomean_yprr_yac_pg)": {
        "college_features": [
            "geomean_yprr_yac_pg",
            "pg_catch_pct_adot_adj_graduated",
            "best2_contested_catch_rate",
            "best2_avoided_tackles_per_rec",
        ],
    },
}

for config_name, config in CONFIGS.items():
    college_features = config["college_features"]
    all_features = ["draft_capital"] + college_features

    d = df.dropna(subset=["tier_ordinal"] + all_features).copy()
    d["tier_ordinal"] = d["tier_ordinal"].astype(int)

    train_df = d[~d["draft_year"].isin(HOLDOUT_YEARS)].copy()
    holdout_df = d[d["draft_year"].isin(HOLDOUT_YEARS)].copy()

    print(f"\n{'=' * 70}")
    print(f"  {config_name}")
    print(f"  Train: {len(train_df)}, Holdout: {len(holdout_df)}")
    print(f"  Features: {all_features}")
    print(f"{'=' * 70}")

    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(train_df[college_features].values)
    X_college_hold = scaler.transform(holdout_df[college_features].values)
    y_train = train_df["tier_ordinal"].values
    y_hold = holdout_df["tier_ordinal"].values

    xgb_full = train_xgb(train_df[all_features].values, y_train,
                          holdout_df[all_features].values, "Full")
    xgb_college = train_xgb(train_df[college_features].values, y_train,
                             holdout_df[college_features].values, "College")
    bayes_full = train_bayesian(X_college_train, train_df["draft_capital"].values, y_train,
                                 X_college_hold, holdout_df["draft_capital"].values, True, "Full")
    bayes_college = train_bayesian(X_college_train, None, y_train,
                                    X_college_hold, None, False, "College")

    def blend(b, x):
        combo = W_BAYES * b + W_XGB * x
        return combo / combo.sum(axis=1, keepdims=True)

    full_probs = blend(bayes_full, xgb_full)
    college_probs = blend(bayes_college, xgb_college)

    print(f"\n  --- ENSEMBLE FULL ---")
    ll, brier, aucs = evaluate(full_probs, y_hold, f"{config_name} FULL")
    config["full_ll"] = ll
    config["full_brier"] = brier
    config["full_aucs"] = aucs

    print(f"\n  --- ENSEMBLE COLLEGE ---")
    cll, cbrier, caucs = evaluate(college_probs, y_hold, f"{config_name} COLLEGE")
    config["college_ll"] = cll
    config["college_brier"] = cbrier
    config["college_aucs"] = caucs

# ============================================================
# Comparison
# ============================================================
print(f"\n{'=' * 70}")
print(f"  HEAD-TO-HEAD COMPARISON")
print(f"{'=' * 70}")

v12 = CONFIGS["v12 (pg_yprr_graduated)"]
comp = CONFIGS["COMPOSITE (geomean_yprr_yac_pg)"]

print(f"\n  FULL MODEL:")
print(f"  {'Metric':<20s} {'v12':>10s} {'Composite':>10s} {'Delta':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'LogLoss':<20s} {v12['full_ll']:>10.4f} {comp['full_ll']:>10.4f} {comp['full_ll'] - v12['full_ll']:>+10.4f}")
print(f"  {'Brier':<20s} {v12['full_brier']:>10.4f} {comp['full_brier']:>10.4f} {comp['full_brier'] - v12['full_brier']:>+10.4f}")
for tlabel in THRESHOLD_LABELS:
    v12_auc = v12["full_aucs"].get(tlabel, np.nan)
    comp_auc = comp["full_aucs"].get(tlabel, np.nan)
    print(f"  {tlabel + ' AUC':<20s} {v12_auc:>10.3f} {comp_auc:>10.3f} {comp_auc - v12_auc:>+10.3f}")

print(f"\n  COLLEGE-ONLY MODEL:")
print(f"  {'Metric':<20s} {'v12':>10s} {'Composite':>10s} {'Delta':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'LogLoss':<20s} {v12['college_ll']:>10.4f} {comp['college_ll']:>10.4f} {comp['college_ll'] - v12['college_ll']:>+10.4f}")
print(f"  {'Brier':<20s} {v12['college_brier']:>10.4f} {comp['college_brier']:>10.4f} {comp['college_brier'] - v12['college_brier']:>+10.4f}")
for tlabel in THRESHOLD_LABELS:
    v12_auc = v12["college_aucs"].get(tlabel, np.nan)
    comp_auc = comp["college_aucs"].get(tlabel, np.nan)
    print(f"  {tlabel + ' AUC':<20s} {v12_auc:>10.3f} {comp_auc:>10.3f} {comp_auc - v12_auc:>+10.3f}")
