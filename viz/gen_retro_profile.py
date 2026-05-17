#!/usr/bin/env python3
"""
Generate a "for fun" retro/hypothetical profile for a player not in predictions.

Usage:
    python viz/gen_retro_profile.py "Jeremiah Smith" --year 2026 --pick 2
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
THRESHOLDS = [1, 2, 3, 4, 5]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

COLLEGE_FEATURES = [
    "best1_yprr_graduated",
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

W_BAYES = 0.75
W_XGB = 0.25


def load_training_data():
    all_features = ["draft_capital"] + COLLEGE_FEATURES
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)
    return df


def train_xgb_predict(train_df, X_prospect, features):
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    X_train = train_df[features].values
    y_train = train_df["tier_ordinal"].values
    cum_probs = np.zeros((X_prospect.shape[0], len(THRESHOLDS)))

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
        cum_probs[:, t_idx] = calibrated.predict_proba(X_prospect)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((X_prospect.shape[0], 6))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def train_bayesian_predict(train_df, X_prospect_college, dc_prospect, features, use_dc):
    import pymc as pm
    import pytensor.tensor as pt
    from sklearn.preprocessing import StandardScaler

    college_feats = [f for f in features if f != "draft_capital"]
    n_college = len(college_feats)

    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(train_df[college_feats].values)
    X_college_pred = scaler.transform(X_prospect_college)
    y_train = train_df["tier_ordinal"].values

    dc_train = train_df["draft_capital"].values if use_dc else None

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
    n_obs = X_college_pred.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    has_dc = "beta_dc" in trace.posterior
    if has_dc:
        beta_dc_samples = trace.posterior["beta_dc"].values.flatten()

    for i in range(n_samples):
        eta = X_college_pred @ beta_college_samples[i]
        if has_dc:
            eta = eta + beta_dc_samples[i] * dc_prospect
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Player name")
    parser.add_argument("--year", type=int, required=True, help="Hypothetical draft year")
    parser.add_argument("--pick", type=int, required=True, help="Hypothetical draft pick")
    args = parser.parse_args()

    name = args.name
    year = args.year
    pick = args.pick
    draft_capital = round(10 - 7 * np.sqrt(pick / 260), 2)

    print(f"Generating retro profile for {name}")
    print(f"  Hypothetical: {year} draft, pick #{pick}, DC={draft_capital}")

    # Step 1: Aggregate college stats
    from aggregation.aggregate_college_stats import (
        load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
    )

    all_grades = load_all_grades(range(2016, 2027))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    result = aggregate_player(
        all_grades, name, year,
        birth_lookup=birth_lookup,
        team_att_lookup=team_att_lookup,
        draft_age_lookup=draft_age_lookup,
        adot_coef=adot_coef,
        team_games_lookup=team_games_lookup,
    )

    if not result:
        print(f"Could not aggregate college stats for {name}")
        sys.exit(1)

    result["draft_capital"] = draft_capital
    result["pick"] = pick

    print(f"\n  College features:")
    for f in COLLEGE_FEATURES:
        print(f"    {f}: {result.get(f, 'MISSING')}")
    print(f"    draft_capital: {draft_capital}")

    # Check for missing features
    all_features = ["draft_capital"] + COLLEGE_FEATURES
    missing = [f for f in all_features if f not in result or pd.isna(result.get(f))]
    if missing:
        print(f"\n  WARNING: Missing features: {missing}")
        sys.exit(1)

    # Step 2: Load training data
    print(f"\nLoading training data...")
    train_df = load_training_data()
    print(f"  {len(train_df)} training players")

    # Step 3: Build prospect feature arrays
    prospect_full = np.array([[result[f] for f in all_features]])
    prospect_college = np.array([[result[f] for f in COLLEGE_FEATURES]])
    dc_array = np.array([draft_capital])

    # Step 4: XGBoost
    print(f"\nTraining XGBoost Full...")
    xgb_full = train_xgb_predict(train_df, prospect_full, all_features)
    print(f"Training XGBoost College-Only...")
    xgb_college = train_xgb_predict(train_df, prospect_college, COLLEGE_FEATURES)

    # Step 5: Bayesian
    print(f"\nTraining Bayesian Full...")
    bayes_full = train_bayesian_predict(train_df, prospect_college, dc_array, all_features, True)
    print(f"\nTraining Bayesian College-Only...")
    bayes_college = train_bayesian_predict(train_df, prospect_college, None, COLLEGE_FEATURES, False)

    # Step 6: Ensemble
    full_probs = W_BAYES * bayes_full + W_XGB * xgb_full
    full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)
    college_probs = W_BAYES * bayes_college + W_XGB * xgb_college
    college_probs = college_probs / college_probs.sum(axis=1, keepdims=True)

    fp = full_probs[0]
    cp = college_probs[0]

    print(f"\n{'='*60}")
    print(f"  RESULTS: {name}")
    print(f"{'='*60}")
    for i, tn in TIER_NAMES.items():
        print(f"  P({tn:15s})  Full: {fp[i]:.3f}  College: {cp[i]:.3f}")
    e_full = sum(fp[i] * i for i in range(6))
    e_college = sum(cp[i] * i for i in range(6))
    edge = e_college - e_full
    print(f"\n  E[tier] Full:    {e_full:.3f}")
    print(f"  E[tier] College: {e_college:.3f}")
    print(f"  Edge:            {edge:+.3f}")

    # Step 7: Build a fake player_row and class_df for profile generation
    player_row = pd.Series({
        "name": name,
        "draft_year": year,
        "pick": pick,
        "expected_tier": e_full,
        "college_expected_tier": e_college,
        "edge": edge,
        "predicted_tier": TIER_NAMES[int(np.argmax(fp))],
        "college_predicted_tier": TIER_NAMES[int(np.argmax(cp))],
    })
    for i, tn in TIER_NAMES.items():
        player_row[f"P({tn})"] = fp[i]
        player_row[f"college_P({tn})"] = cp[i]

    # Use this player as the only member of the "class" for ranking
    class_df = pd.DataFrame([player_row])

    # Step 8: Generate profile (pass year as "YYYY (Retro)" string for folder/title)
    from viz.prospect_profile import make_profile, load_training_features
    train_feat_df = load_training_features()

    make_profile(player_row, class_df, f"{year} (Retro)", result, train_feat_df)


if __name__ == "__main__":
    main()
