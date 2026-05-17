#!/usr/bin/env python3
"""
Test substituting age-adjusted production metrics for best2_yprr in the full model.

Variants tested (each replaces best2_yprr):
  1. best2_yptpa_both    -- best2 YPTPA, sr>=21.5/-15%, fr<=20.0/+15%
  2. best1_yprr_senior   -- best1 YPRR, sr>=21.5/-15%
  3. best1_yprr_both     -- best1 YPRR, sr>=21.5/-15%, fr<=19.5/+15%
  4. best2_yprr_both     -- best2 YPRR, sr>=21.5/-15%, fr<=20.0/+15%
  5. career_yptpa_both   -- career YPTPA, sr>=21.5/-15%, fr<=20.0/+15%
  6. young_yprr          -- best of FR/SO YPRR, fallback to -15% JR YPRR

For each variant: trains XGBoost + Bayesian on 2018-2021, evaluates ensemble
on 2022-2024 holdout. Saves results to wr_data/yprr_substitution_results.csv.

Usage:
  python modeling/test_yprr_substitutions.py
"""

import os
import sys
import json
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
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

OTHER_COLLEGE_FEATURES = [
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

HOLDOUT_YEARS = [2022, 2023, 2024]
W_BAYES = 0.75
W_XGB = 0.25

# --- Variant definitions ---
VARIANTS = {
    "best2_yptpa_both": {
        "metric": "yptpa", "agg": "best2",
        "scheme": "both",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 20.0, "freshman_boost": 0.15,
    },
    "best1_yprr_senior": {
        "metric": "yprr", "agg": "best1",
        "scheme": "senior",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
    },
    "best1_yprr_both": {
        "metric": "yprr", "agg": "best1",
        "scheme": "both",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 19.5, "freshman_boost": 0.15,
    },
    "best2_yprr_both": {
        "metric": "yprr", "agg": "best2",
        "scheme": "both",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 20.0, "freshman_boost": 0.15,
    },
    "career_yptpa_both": {
        "metric": "yptpa", "agg": "career",
        "scheme": "both",
        "senior_age_thresh": 21.5, "senior_discount": 0.15,
        "freshman_age_thresh": 20.0, "freshman_boost": 0.15,
    },
    "young_yprr": {
        "custom": "young_yprr",
        "junior_discount": 0.15,
    },
    "best1_yprr_grad_sp": {
        "metric": "yprr", "agg": "best1",
        "scheme": "graduated",
        "adj": {"freshman": 0.25, "sophomore": 0.05, "junior": 0, "senior": -0.25},
    },
    "best1_yprr_grad_auc": {
        "metric": "yprr", "agg": "best1",
        "scheme": "graduated",
        "adj": {"freshman": 0.25, "sophomore": 0.05, "junior": -0.20, "senior": -0.25},
    },
    "best2_yprr_grad_sp": {
        "metric": "yprr", "agg": "best2",
        "scheme": "graduated",
        "adj": {"freshman": 0.25, "sophomore": 0.10, "junior": 0, "senior": -0.25},
    },
    "best2_yprr_grad_auc": {
        "metric": "yprr", "agg": "best2",
        "scheme": "graduated",
        "adj": {"freshman": 0.25, "sophomore": 0.10, "junior": -0.20, "senior": -0.25},
    },
}


# ---- Feature computation ----

def compute_season_age(grade_year, birthdate):
    if birthdate is None or pd.isna(birthdate) or pd.isna(grade_year):
        return np.nan
    sept1 = pd.Timestamp(f"{int(grade_year)}-09-01")
    return (sept1 - birthdate).days / 365.25


def get_player_seasons(all_grades, name, draft_year, birth_lookup):
    """Get eligible seasons for a player (200+ routes, age/year filters)."""
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

    routes = pd.to_numeric(seasons.get("routes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    seasons = seasons[routes >= 200].copy()

    # Compute age for each season
    if bd is not None and pd.notna(bd):
        seasons["_age"] = seasons["grade_year"].apply(lambda yr: compute_season_age(yr, bd))
    else:
        seasons["_age"] = np.nan

    return seasons


def compute_season_metrics(seasons, team_att_lookup):
    """Add per-season yprr and yptpa columns."""
    s = seasons.copy()
    yards = pd.to_numeric(s["yards"], errors="coerce").fillna(0)
    routes = pd.to_numeric(s["routes"], errors="coerce").fillna(0)
    games = pd.to_numeric(s["player_game_count"], errors="coerce").fillna(0)

    s["_yprr"] = np.where(routes > 0, yards / routes, np.nan)

    yptpa_vals = []
    for _, row in s.iterrows():
        team = row.get("team_name", "")
        yr = row.get("grade_year")
        att = team_att_lookup.get((team, yr))
        y = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
        yptpa_vals.append(y / att if att and att > 0 else np.nan)
    s["_yptpa"] = yptpa_vals

    s["_is_p5"] = s["team_name"].isin(P5_TEAMS)
    return s


AGE_BINS = {
    "freshman": (0, 19.5),
    "sophomore": (19.5, 20.5),
    "junior": (20.5, 21.5),
    "senior": (21.5, 99),
}


def apply_age_adjustment(seasons, metric_col, variant):
    """Apply age discount/boost to a metric column."""
    s = seasons.copy()
    s[metric_col] = s[metric_col].astype(float)
    scheme = variant["scheme"]

    if scheme == "graduated":
        adj = variant["adj"]
        for cls, (lo, hi) in AGE_BINS.items():
            a = adj.get(cls, 0)
            if a == 0:
                continue
            mask = s["_age"].notna() & (s["_age"] >= lo) & (s["_age"] < hi)
            s.loc[mask, metric_col] = s.loc[mask, metric_col] * (1 + a)
        return s

    if scheme in ("senior", "both"):
        thresh = variant["senior_age_thresh"]
        disc = variant["senior_discount"]
        mask = s["_age"].notna() & (s["_age"] >= thresh)
        s.loc[mask, metric_col] = s.loc[mask, metric_col] * (1 - disc)

    if scheme in ("freshman", "both"):
        thresh = variant["freshman_age_thresh"]
        boost = variant["freshman_boost"]
        mask = s["_age"].notna() & (s["_age"] <= thresh)
        s.loc[mask, metric_col] = s.loc[mask, metric_col] * (1 + boost)

    return s


def aggregate_variant(seasons, metric_col, agg_type):
    """Aggregate adjusted metric across seasons using the specified window."""
    if len(seasons) == 0:
        return np.nan

    vals = seasons[metric_col]
    routes = pd.to_numeric(seasons["routes"], errors="coerce").fillna(0)

    # P5 filter for best2/best1
    if agg_type in ("best2", "best1"):
        min_p5 = 2 if agg_type == "best2" else 1
        p5 = seasons[seasons["_is_p5"]]
        eligible = p5 if len(p5) >= min_p5 else seasons
    else:
        eligible = seasons

    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    vals_e = eligible[metric_col]
    routes_e = pd.to_numeric(eligible["routes"], errors="coerce").fillna(0)

    if agg_type == "best2":
        if grades.notna().sum() >= 2:
            selected = eligible.loc[grades.nlargest(2).index]
        else:
            selected = eligible
    elif agg_type == "best1":
        if grades.notna().sum() >= 1:
            selected = eligible.loc[[grades.idxmax()]]
        else:
            selected = eligible.head(1)
    else:  # career
        selected = eligible

    v = selected[metric_col]
    r = pd.to_numeric(selected["routes"], errors="coerce").fillna(0)
    mask = v.notna()
    if not mask.any():
        return np.nan
    return np.average(v[mask], weights=r[mask])


def compute_young_yprr(seasons, junior_discount=0.15):
    """Best of freshman/sophomore YPRR; fallback to 15%-discounted junior YPRR.

    Age classes:
      - Freshman: age < 19.5 on Sept 1
      - Sophomore: 19.5 <= age < 20.5
      - Junior: 20.5 <= age < 21.5

    Selection: best single season by grades_offense among freshman/sophomore.
    If no freshman or sophomore season available, use best junior season * (1 - discount).
    If no junior season either, return NaN.
    """
    if len(seasons) == 0 or "_age" not in seasons.columns:
        return np.nan

    # P5 filter
    p5 = seasons[seasons["_is_p5"]]
    eligible = p5 if len(p5) >= 1 else seasons

    young = eligible[eligible["_age"].notna() & (eligible["_age"] < 20.5)]
    if len(young) > 0:
        grades = pd.to_numeric(young["grades_offense"], errors="coerce")
        if grades.notna().any():
            best = young.loc[grades.idxmax()]
        else:
            best = young.iloc[0]
        yprr = best["_yprr"]
        return yprr if pd.notna(yprr) else np.nan

    # Fallback to junior (20.5 <= age < 21.5)
    juniors = eligible[
        eligible["_age"].notna() & (eligible["_age"] >= 20.5) & (eligible["_age"] < 21.5)
    ]
    if len(juniors) > 0:
        grades = pd.to_numeric(juniors["grades_offense"], errors="coerce")
        if grades.notna().any():
            best = juniors.loc[grades.idxmax()]
        else:
            best = juniors.iloc[0]
        yprr = best["_yprr"]
        if pd.notna(yprr):
            return yprr * (1 - junior_discount)

    return np.nan


def compute_variant_feature(all_grades, dynasty_df, birth_lookup, team_att_lookup, variant):
    """Compute a variant feature for all players in the dynasty DataFrame."""
    is_custom = "custom" in variant

    values = {}
    for _, row in dynasty_df.iterrows():
        name = row["name"]
        draft_year = row["draft_year"]
        seasons = get_player_seasons(all_grades, name, draft_year, birth_lookup)
        if len(seasons) == 0:
            values[(name, draft_year)] = np.nan
            continue

        seasons = compute_season_metrics(seasons, team_att_lookup)

        if is_custom and variant["custom"] == "young_yprr":
            values[(name, draft_year)] = compute_young_yprr(
                seasons, junior_discount=variant.get("junior_discount", 0.15)
            )
        else:
            metric_key = f"_{variant['metric']}"
            agg_type = variant["agg"]
            seasons = apply_age_adjustment(seasons, metric_key, variant)
            values[(name, draft_year)] = aggregate_variant(seasons, metric_key, agg_type)

    return values


# ---- Model training ----

def train_xgb(train_df, holdout_df, features):
    X_train = train_df[features].values
    y_train = train_df["tier_ordinal"].values
    X_hold = holdout_df[features].values

    cum_probs = np.zeros((len(holdout_df), len(THRESHOLDS)))
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

    tier_probs = np.zeros((len(holdout_df), N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def train_bayesian(train_df, holdout_df, features):
    college_feats = [f for f in features if f != "draft_capital"]
    n_college = len(college_feats)
    use_dc = "draft_capital" in features

    scaler = StandardScaler()
    X_college_train = scaler.fit_transform(train_df[college_feats].values)
    X_college_hold = scaler.transform(holdout_df[college_feats].values)
    y_train = train_df["tier_ordinal"].values

    dc_train = train_df["draft_capital"].values if use_dc else None
    dc_hold = holdout_df["draft_capital"].values if use_dc else None

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


def evaluate_model(probs, y_true):
    """Return dict of evaluation metrics."""
    y_onehot = np.zeros((len(y_true), N_TIERS))
    y_onehot[np.arange(len(y_true)), y_true] = 1

    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))

    result = {"logloss": round(ll, 4), "brier": round(brier, 4)}

    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        if 0 < y_bin.sum() < len(y_bin):
            auc = roc_auc_score(y_bin, pred)
            result[f"auc_{tlabel}"] = round(auc, 4)
        brier_t = brier_score_loss(y_bin, pred)
        result[f"brier_{tlabel}"] = round(brier_t, 4)

    return result


# ---- Main ----

def main():
    print("Loading data...")
    all_grades = load_all_grades(range(2016, 2026))
    base_df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    base_df["tier_ordinal"] = base_df["computed_tier"].map(TIER_ORDER)

    birth_lookup, _, team_att_lookup, _ = build_lookups(all_grades)

    # ---- Run baseline (current v8 model) ----
    print("\n" + "=" * 70)
    print("BASELINE: current v8 (best2_yprr, no age adjustment)")
    print("=" * 70)

    baseline_features = ["draft_capital", "best2_yprr"] + OTHER_COLLEGE_FEATURES
    df_base = base_df.dropna(subset=["tier_ordinal"] + baseline_features).copy()
    df_base["tier_ordinal"] = df_base["tier_ordinal"].astype(int)

    train_base = df_base[~df_base["draft_year"].isin(HOLDOUT_YEARS)]
    hold_base = df_base[df_base["draft_year"].isin(HOLDOUT_YEARS)]
    print(f"  Train: {len(train_base)}, Holdout: {len(hold_base)}")

    t0 = time.time()
    print("  Training XGBoost...")
    xgb_base = train_xgb(train_base, hold_base, baseline_features)
    print("  Training Bayesian...")
    bayes_base = train_bayesian(train_base, hold_base, baseline_features)

    ens_base = W_BAYES * bayes_base + W_XGB * xgb_base
    ens_base = ens_base / ens_base.sum(axis=1, keepdims=True)

    actual = hold_base["tier_ordinal"].values
    baseline_results = {
        "xgb": evaluate_model(xgb_base, actual),
        "bayes": evaluate_model(bayes_base, actual),
        "ensemble": evaluate_model(ens_base, actual),
    }
    elapsed = time.time() - t0
    print(f"  Baseline done in {elapsed:.0f}s")

    print(f"\n  Ensemble: LogLoss={baseline_results['ensemble']['logloss']:.4f}  "
          f"Brier={baseline_results['ensemble']['brier']:.4f}  "
          f"AUC(Elite)={baseline_results['ensemble'].get('auc_>=Elite', 'N/A')}  "
          f"AUC(Stud)={baseline_results['ensemble'].get('auc_>=Stud', 'N/A')}")

    all_results = [{
        "variant": "baseline_best2_yprr",
        "description": "Current v8 (best2_yprr, no age adj)",
        **{f"xgb_{k}": v for k, v in baseline_results["xgb"].items()},
        **{f"bayes_{k}": v for k, v in baseline_results["bayes"].items()},
        **{f"ens_{k}": v for k, v in baseline_results["ensemble"].items()},
        "n_train": len(train_base),
        "n_holdout": len(hold_base),
    }]

    # ---- Run each variant ----
    for var_name, var_config in VARIANTS.items():
        print(f"\n{'=' * 70}")
        print(f"VARIANT: {var_name}")
        print(f"  {var_config}")
        print(f"{'=' * 70}")

        t0 = time.time()
        print("  Computing variant feature...")
        feat_values = compute_variant_feature(
            all_grades, base_df, birth_lookup, team_att_lookup, var_config
        )

        # Add variant feature to DataFrame
        df_var = base_df.copy()
        df_var["_variant_feat"] = [
            feat_values.get((row["name"], row["draft_year"]), np.nan)
            for _, row in df_var.iterrows()
        ]

        # Use variant feature in place of best2_yprr
        var_features = ["draft_capital", "_variant_feat"] + OTHER_COLLEGE_FEATURES
        df_var = df_var.dropna(subset=["tier_ordinal"] + var_features).copy()
        df_var["tier_ordinal"] = df_var["tier_ordinal"].astype(int)

        train_var = df_var[~df_var["draft_year"].isin(HOLDOUT_YEARS)]
        hold_var = df_var[df_var["draft_year"].isin(HOLDOUT_YEARS)]
        print(f"  Train: {len(train_var)}, Holdout: {len(hold_var)}")

        print("  Training XGBoost...")
        xgb_var = train_xgb(train_var, hold_var, var_features)
        print("  Training Bayesian...")
        bayes_var = train_bayesian(train_var, hold_var, var_features)

        ens_var = W_BAYES * bayes_var + W_XGB * xgb_var
        ens_var = ens_var / ens_var.sum(axis=1, keepdims=True)

        actual_var = hold_var["tier_ordinal"].values
        var_results = {
            "xgb": evaluate_model(xgb_var, actual_var),
            "bayes": evaluate_model(bayes_var, actual_var),
            "ensemble": evaluate_model(ens_var, actual_var),
        }
        elapsed = time.time() - t0

        print(f"\n  Ensemble: LogLoss={var_results['ensemble']['logloss']:.4f}  "
              f"Brier={var_results['ensemble']['brier']:.4f}  "
              f"AUC(Elite)={var_results['ensemble'].get('auc_>=Elite', 'N/A')}  "
              f"AUC(Stud)={var_results['ensemble'].get('auc_>=Stud', 'N/A')}  "
              f"({elapsed:.0f}s)")

        if "custom" in var_config:
            desc = var_config["custom"]
        else:
            desc = (f"{var_config['agg']} {var_config['metric']} "
                    f"{var_config['scheme']}")
        all_results.append({
            "variant": var_name,
            "description": desc,
            **{f"xgb_{k}": v for k, v in var_results["xgb"].items()},
            **{f"bayes_{k}": v for k, v in var_results["bayes"].items()},
            **{f"ens_{k}": v for k, v in var_results["ensemble"].items()},
            "n_train": len(train_var),
            "n_holdout": len(hold_var),
        })

        # Save incrementally
        pd.DataFrame(all_results).to_csv(
            os.path.join(DATA_DIR, "yprr_substitution_results.csv"), index=False
        )

    # ---- Final summary ----
    results_df = pd.DataFrame(all_results)
    print(f"\n\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")

    key_cols = ["variant", "ens_logloss", "ens_brier",
                "ens_auc_>=Elite", "ens_auc_>=Stud", "ens_auc_>=LW"]
    available = [c for c in key_cols if c in results_df.columns]
    print(results_df[available].to_string(index=False))

    # Deltas vs baseline
    baseline_row = results_df.iloc[0]
    print(f"\n  DELTAS vs BASELINE:")
    for _, row in results_df.iloc[1:].iterrows():
        d_ll = row["ens_logloss"] - baseline_row["ens_logloss"]
        d_br = row["ens_brier"] - baseline_row["ens_brier"]
        d_elite = row.get("ens_auc_>=Elite", 0) - baseline_row.get("ens_auc_>=Elite", 0)
        d_stud = row.get("ens_auc_>=Stud", 0) - baseline_row.get("ens_auc_>=Stud", 0)
        print(f"    {row['variant']:25s}  dLL={d_ll:+.4f}  dBrier={d_br:+.4f}  "
              f"dAUC(E)={d_elite:+.4f}  dAUC(S)={d_stud:+.4f}")

    out_path = os.path.join(DATA_DIR, "yprr_substitution_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
