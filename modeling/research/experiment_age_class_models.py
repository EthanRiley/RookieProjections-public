#!/usr/bin/env python3
"""
Test production metrics by age class in bivariate and trivariate models.

Investigates the claim that production metrics lose predictive power at
older ages, especially when combined with draft capital.

For each age class (freshman/sophomore/junior/senior), computes single-season
YPRR, YPTPA, and YPG. Then runs:
  - Bivariate: draft_capital + metric
  - Trivariate: draft_capital + metric + career_targeted_qb_rating

Also tests the age-adjusted "winner" metrics from the substitution study.

Baselines:
  - draft_capital alone
  - draft_capital + career_targeted_qb_rating

All models: XGBoost + Bayesian ensemble (75/25), holdout 2022-2024.

Usage:
  python modeling/test_age_class_models.py
"""

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from aggregation.aggregate_college_stats import (
    normalize_name, load_all_grades, build_lookups, P5_TEAMS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

HOLDOUT_YEARS = [2022, 2023, 2024]
W_BAYES = 0.75
W_XGB = 0.25

# Age class bins (age on Sept 1 of season)
AGE_CLASSES = {
    "freshman": (0, 19.5),
    "sophomore": (19.5, 20.5),
    "junior": (20.5, 21.5),
    "senior": (21.5, 99),
}

BASE_METRICS = ["yprr", "yptpa", "ypg"]

# Winner variants from substitution study
WINNER_VARIANTS = {
    "best1_yprr_both": {
        "metric": "yprr", "agg": "best1",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 19.5, "freshman_boost": 0.15,
    },
    "best2_yprr_both": {
        "metric": "yprr", "agg": "best2",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 20.0, "freshman_boost": 0.15,
    },
    "best1_yprr_senior": {
        "metric": "yprr", "agg": "best1",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
    },
    "best2_yptpa_both": {
        "metric": "yptpa", "agg": "best2",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 20.0, "freshman_boost": 0.15,
    },
    "best2_yprr_unadj": {
        "metric": "yprr", "agg": "best2",
    },
}


def compute_season_age(grade_year, birthdate):
    if birthdate is None or pd.isna(birthdate) or pd.isna(grade_year):
        return np.nan
    sept1 = pd.Timestamp(f"{int(grade_year)}-09-01")
    return (sept1 - birthdate).days / 365.25


def build_player_seasons(all_grades, base_df, birth_lookup, team_att_lookup):
    """Build per-player, per-season metric values with age."""
    records = []
    for _, row in base_df.iterrows():
        name = row["name"]
        draft_year = row["draft_year"]
        bd = birth_lookup.get((name, draft_year))

        key = normalize_name(name)
        seasons = all_grades[
            (all_grades["_join_key"] == key) & (all_grades["grade_year"] <= draft_year)
        ].copy()

        if bd is not None and pd.notna(bd):
            min_year = bd.year + 18
            sept1_min = pd.Timestamp(f"{min_year}-09-01")
            if sept1_min < bd + pd.DateOffset(years=18):
                min_year += 1
            seasons = seasons[seasons["grade_year"] >= min_year]
        seasons = seasons[seasons["grade_year"] >= draft_year - 5]

        routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)),
                               errors="coerce").fillna(0)
        seasons = seasons[routes >= 200]

        for _, s in seasons.iterrows():
            yr = s["grade_year"]
            r = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
            y = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
            g = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
            go = pd.to_numeric(s.get("grades_offense", np.nan), errors="coerce")
            team = s.get("team_name", "")
            age = compute_season_age(yr, bd) if bd is not None else np.nan

            att = team_att_lookup.get((team, yr))

            records.append({
                "name": name, "draft_year": draft_year,
                "grade_year": yr, "age": age,
                "yprr": y / r if r > 0 else np.nan,
                "yptpa": y / att if att and att > 0 else np.nan,
                "ypg": y / g if g > 0 else np.nan,
                "yards": y, "routes": r, "games": g,
                "grades_offense": go,
                "team_name": team,
                "is_p5": team in P5_TEAMS,
            })

    return pd.DataFrame(records)


def get_age_class_feature(player_seasons, name, draft_year, metric, age_lo, age_hi):
    """Get a player's metric value for a specific age class.
    If multiple seasons in the class, take the one with highest grades_offense."""
    ps = player_seasons[
        (player_seasons["name"] == name) &
        (player_seasons["draft_year"] == draft_year) &
        (player_seasons["age"] >= age_lo) &
        (player_seasons["age"] < age_hi)
    ]
    if len(ps) == 0:
        return np.nan
    if len(ps) > 1:
        grades = pd.to_numeric(ps["grades_offense"], errors="coerce")
        if grades.notna().any():
            ps = ps.loc[[grades.idxmax()]]
        else:
            ps = ps.head(1)
    return ps[metric].values[0]


def get_winner_feature(player_seasons, name, draft_year, variant):
    """Compute an age-adjusted aggregated feature for a player."""
    ps = player_seasons[
        (player_seasons["name"] == name) &
        (player_seasons["draft_year"] == draft_year)
    ].copy()
    if len(ps) == 0:
        return np.nan

    metric = variant["metric"]
    agg = variant["agg"]
    ps["_val"] = ps[metric].astype(float)

    # Apply adjustments
    if "senior_age_thresh" in variant:
        mask = ps["age"].notna() & (ps["age"] >= variant["senior_age_thresh"])
        ps.loc[mask, "_val"] = ps.loc[mask, "_val"] * (1 - variant["senior_discount"])
    if "freshman_age_thresh" in variant:
        mask = ps["age"].notna() & (ps["age"] <= variant["freshman_age_thresh"])
        ps.loc[mask, "_val"] = ps.loc[mask, "_val"] * (1 + variant["freshman_boost"])

    # Aggregate
    if agg in ("best2", "best1"):
        min_p5 = 2 if agg == "best2" else 1
        p5 = ps[ps["is_p5"]]
        eligible = p5 if len(p5) >= min_p5 else ps
        grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
        n_select = 2 if agg == "best2" else 1
        if grades.notna().sum() >= n_select:
            selected = eligible.loc[grades.nlargest(n_select).index]
        else:
            selected = eligible
    else:
        selected = ps

    vals = selected["_val"]
    routes = pd.to_numeric(selected["routes"], errors="coerce").fillna(0)
    mask = vals.notna()
    if not mask.any():
        return np.nan
    return np.average(vals[mask], weights=routes[mask])


# ---- Model training (simplified from evaluate_holdout.py) ----

def train_xgb(X_train, y_train, X_hold):
    cum_probs = np.zeros((X_hold.shape[0], len(THRESHOLDS)))
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
        if min_class < 2:
            # Too few examples for calibrated CV, use raw model
            model.fit(X_train, y_bin)
            cum_probs[:, t_idx] = model.predict_proba(X_hold)[:, 1]
        else:
            cv_folds = min(5, max(2, min_class))
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds)
            calibrated.fit(X_train, y_bin)
            cum_probs[:, t_idx] = calibrated.predict_proba(X_hold)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((X_hold.shape[0], N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def train_bayesian(X_train, y_train, X_hold, has_dc=True):
    """Train Bayesian ordinal model. Assumes first column is draft_capital if has_dc."""
    if has_dc:
        dc_train = X_train[:, 0]
        dc_hold = X_hold[:, 0]
        college_train = X_train[:, 1:]
        college_hold = X_hold[:, 1:]
    else:
        college_train = X_train
        college_hold = X_hold

    n_college = college_train.shape[1]

    if n_college > 0:
        scaler = StandardScaler()
        college_train_s = scaler.fit_transform(college_train)
        college_hold_s = scaler.transform(college_hold)
    else:
        college_train_s = college_train
        college_hold_s = college_hold

    with pm.Model() as model:
        if n_college > 0:
            beta_college = pm.Normal("beta_college", mu=0.0, sigma=0.5, shape=n_college)
            eta = pt.dot(college_train_s, beta_college)
        else:
            eta = pt.zeros(len(y_train))
        if has_dc:
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
            2000, tune=1500, chains=4, cores=1,
            random_seed=42, progressbar=False, target_accept=0.9,
        )

    cutpoints_samples = trace.posterior["cutpoints"].values.reshape(-1, N_CUTPOINTS)
    n_samples = len(cutpoints_samples)
    n_obs = college_hold_s.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    has_college = n_college > 0 and "beta_college" in trace.posterior
    if has_college:
        beta_college_samples = trace.posterior["beta_college"].values.reshape(-1, n_college)

    has_dc_param = "beta_dc" in trace.posterior
    if has_dc_param:
        beta_dc_samples = trace.posterior["beta_dc"].values.flatten()

    for i in range(n_samples):
        if has_college:
            eta = college_hold_s @ beta_college_samples[i]
        else:
            eta = np.zeros(n_obs)
        if has_dc_param:
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


def evaluate_probs(probs, y_true):
    y_onehot = np.zeros((len(y_true), N_TIERS))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    result = {"logloss": round(ll, 4), "brier": round(brier, 4)}
    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        if 0 < y_bin.sum() < len(y_bin):
            result[f"auc_{tlabel}"] = round(roc_auc_score(y_bin, pred), 4)
    return result


def run_model(df, features, label):
    """Train XGB + Bayesian ensemble, evaluate on holdout. Returns result dict."""
    sub = df.dropna(subset=["tier_ordinal"] + features).copy()
    train = sub[~sub["draft_year"].isin(HOLDOUT_YEARS)]
    hold = sub[sub["draft_year"].isin(HOLDOUT_YEARS)]

    if len(train) < 20 or len(hold) < 10:
        return {"label": label, "n_train": len(train), "n_holdout": len(hold),
                "logloss": np.nan, "brier": np.nan}

    X_train = train[features].values.astype(float)
    y_train = train["tier_ordinal"].values
    X_hold = hold[features].values.astype(float)
    y_hold = hold["tier_ordinal"].values

    has_dc = "draft_capital" in features

    xgb_probs = train_xgb(X_train, y_train, X_hold)
    bayes_probs = train_bayesian(X_train, y_train, X_hold, has_dc=has_dc)

    ens_probs = W_BAYES * bayes_probs + W_XGB * xgb_probs
    ens_probs = ens_probs / ens_probs.sum(axis=1, keepdims=True)

    ev = evaluate_probs(ens_probs, y_hold)
    return {"label": label, "n_train": len(train), "n_holdout": len(hold), **ev}


def main():
    print("Loading data...")
    all_grades = load_all_grades(range(2016, 2026))
    base_df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    base_df["tier_ordinal"] = base_df["computed_tier"].map(TIER_ORDER)

    birth_lookup, _, team_att_lookup, _ = build_lookups(all_grades)

    print("Building player-season data...")
    player_seasons = build_player_seasons(all_grades, base_df, birth_lookup, team_att_lookup)
    print(f"  {len(player_seasons)} player-seasons")

    # ---- Compute age-class features ----
    print("Computing age-class features...")
    for age_class, (age_lo, age_hi) in AGE_CLASSES.items():
        for metric in BASE_METRICS:
            col = f"{age_class}_{metric}"
            base_df[col] = [
                get_age_class_feature(player_seasons, row["name"], row["draft_year"],
                                      metric, age_lo, age_hi)
                for _, row in base_df.iterrows()
            ]
            n_valid = base_df[col].notna().sum()
            print(f"  {col}: {n_valid} players with data")

    # ---- Compute winner features ----
    print("Computing winner features...")
    for var_name, var_config in WINNER_VARIANTS.items():
        base_df[var_name] = [
            get_winner_feature(player_seasons, row["name"], row["draft_year"], var_config)
            for _, row in base_df.iterrows()
        ]
        n_valid = base_df[var_name].notna().sum()
        print(f"  {var_name}: {n_valid} players with data")

    # ---- Run models ----
    all_results = []
    t_total = time.time()

    # Baselines
    print("\n--- BASELINES ---")
    t0 = time.time()
    r = run_model(base_df, ["draft_capital"], "DC only")
    all_results.append({"group": "baseline", "model_type": "bivariate", **r})
    print(f"  DC only: LL={r['logloss']:.4f} Brier={r['brier']:.4f} "
          f"AUC(E)={r.get('auc_>=Elite', 'N/A')} AUC(S)={r.get('auc_>=Stud', 'N/A')} "
          f"({time.time()-t0:.0f}s)")

    t0 = time.time()
    r = run_model(base_df, ["draft_capital", "career_targeted_qb_rating"],
                  "DC + career_tqbr")
    all_results.append({"group": "baseline", "model_type": "trivariate", **r})
    print(f"  DC + tQBR: LL={r['logloss']:.4f} Brier={r['brier']:.4f} "
          f"AUC(E)={r.get('auc_>=Elite', 'N/A')} AUC(S)={r.get('auc_>=Stud', 'N/A')} "
          f"({time.time()-t0:.0f}s)")

    # Age-class models
    for age_class in AGE_CLASSES:
        print(f"\n--- {age_class.upper()} ---")
        for metric in BASE_METRICS:
            col = f"{age_class}_{metric}"

            # Bivariate: DC + metric
            t0 = time.time()
            r = run_model(base_df, ["draft_capital", col],
                          f"DC + {col}")
            all_results.append({"group": age_class, "model_type": "bivariate",
                                "metric": metric, **r})
            auc_e = r.get("auc_>=Elite", "N/A")
            auc_s = r.get("auc_>=Stud", "N/A")
            print(f"  DC + {col}: n={r['n_train']}/{r['n_holdout']} "
                  f"LL={r['logloss']:.4f} Brier={r['brier']:.4f} "
                  f"AUC(E)={auc_e} AUC(S)={auc_s} ({time.time()-t0:.0f}s)")

            # Trivariate: DC + metric + career_tqbr
            t0 = time.time()
            r = run_model(base_df, ["draft_capital", col, "career_targeted_qb_rating"],
                          f"DC + {col} + tQBR")
            all_results.append({"group": age_class, "model_type": "trivariate",
                                "metric": metric, **r})
            auc_e = r.get("auc_>=Elite", "N/A")
            auc_s = r.get("auc_>=Stud", "N/A")
            print(f"  DC + {col} + tQBR: n={r['n_train']}/{r['n_holdout']} "
                  f"LL={r['logloss']:.4f} Brier={r['brier']:.4f} "
                  f"AUC(E)={auc_e} AUC(S)={auc_s} ({time.time()-t0:.0f}s)")

    # Winner variants
    print(f"\n--- WINNER VARIANTS ---")
    for var_name in WINNER_VARIANTS:
        # Bivariate
        t0 = time.time()
        r = run_model(base_df, ["draft_capital", var_name],
                      f"DC + {var_name}")
        all_results.append({"group": "winner", "model_type": "bivariate",
                            "metric": var_name, **r})
        auc_e = r.get("auc_>=Elite", "N/A")
        auc_s = r.get("auc_>=Stud", "N/A")
        print(f"  DC + {var_name}: LL={r['logloss']:.4f} Brier={r['brier']:.4f} "
              f"AUC(E)={auc_e} AUC(S)={auc_s} ({time.time()-t0:.0f}s)")

        # Trivariate
        t0 = time.time()
        r = run_model(base_df, ["draft_capital", var_name, "career_targeted_qb_rating"],
                      f"DC + {var_name} + tQBR")
        all_results.append({"group": "winner", "model_type": "trivariate",
                            "metric": var_name, **r})
        auc_e = r.get("auc_>=Elite", "N/A")
        auc_s = r.get("auc_>=Stud", "N/A")
        print(f"  DC + {var_name} + tQBR: LL={r['logloss']:.4f} Brier={r['brier']:.4f} "
              f"AUC(E)={auc_e} AUC(S)={auc_s} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s")

    # ---- Save ----
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(DATA_DIR, "age_class_model_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("SUMMARY: Bivariate models (DC + metric)")
    print(f"{'='*80}")
    bi = results_df[results_df["model_type"] == "bivariate"]
    print(f"{'Label':<35s} {'n':>5s} {'LL':>8s} {'Brier':>8s} "
          f"{'AUC(E)':>8s} {'AUC(S)':>8s}")
    print("-" * 80)
    for _, r in bi.iterrows():
        print(f"{r['label']:<35s} {r['n_holdout']:>5.0f} {r['logloss']:>8.4f} "
              f"{r['brier']:>8.4f} {r.get('auc_>=Elite', np.nan):>8.4f} "
              f"{r.get('auc_>=Stud', np.nan):>8.4f}")

    print(f"\n{'='*80}")
    print("SUMMARY: Trivariate models (DC + metric + career_tQBR)")
    print(f"{'='*80}")
    tri = results_df[results_df["model_type"] == "trivariate"]
    print(f"{'Label':<35s} {'n':>5s} {'LL':>8s} {'Brier':>8s} "
          f"{'AUC(E)':>8s} {'AUC(S)':>8s}")
    print("-" * 80)
    for _, r in tri.iterrows():
        print(f"{r['label']:<35s} {r['n_holdout']:>5.0f} {r['logloss']:>8.4f} "
              f"{r['brier']:>8.4f} {r.get('auc_>=Elite', np.nan):>8.4f} "
              f"{r.get('auc_>=Stud', np.nan):>8.4f}")


if __name__ == "__main__":
    main()
