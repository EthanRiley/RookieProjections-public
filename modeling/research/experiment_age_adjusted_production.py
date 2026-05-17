#!/usr/bin/env python3
"""
Test age-adjusted versions of production metrics across aggregation windows.

Base metrics:
  - YPRR (yards / routes)
  - YPTPA (yards / team_pass_attempts)
  - Yards per game
  - Total yards (season)

Aggregation windows:
  - best2: top 2 seasons by grades_offense (P5 filter, 200+ routes)
  - best1: single best season by grades_offense
  - career: all eligible seasons

Adjustment schemes:
  1. None (baseline)
  2. Senior discount: multiplicative penalty for age >= threshold
  3. Freshman boost: multiplicative bonus for age <= threshold
  4. Senior discount + freshman boost combined
  5. Empirical age normalization: scale each season to a reference age
     using population-average production at each age

Evaluation: Spearman correlation with dynasty_value, AUC for >=Elite,
AUC for >=Stud.

Usage:
  python modeling/test_age_adjusted_production.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from aggregation.aggregate_college_stats import (
    normalize_name, load_all_grades, build_lookups, P5_TEAMS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

# --- Age adjustment parameters to grid search ---
SENIOR_AGE_THRESHOLDS = [21.5, 22.0]
SENIOR_DISCOUNTS = [0.05, 0.10, 0.15]

FRESHMAN_AGE_THRESHOLDS = [19.0, 19.5, 20.0]
FRESHMAN_BOOSTS = [0.05, 0.10, 0.15]

REFERENCE_AGE = 20.5


def compute_season_age(grade_year, birthdate):
    """Age on Sept 1 of the season."""
    if birthdate is None or pd.isna(birthdate) or pd.isna(grade_year):
        return np.nan
    sept1 = pd.Timestamp(f"{int(grade_year)}-09-01")
    return (sept1 - birthdate).days / 365.25


def get_player_season_data(all_grades, dynasty, birth_lookup, team_att_lookup):
    """Build a DataFrame of all eligible player-seasons with raw production metrics."""
    records = []

    for _, row in dynasty.iterrows():
        name = row["name"]
        draft_year = row["draft_year"]
        dv = row.get("dynasty_value", 0)
        bd = birth_lookup.get((name, draft_year))

        key = normalize_name(name)
        seasons = all_grades[
            (all_grades["_join_key"] == key) & (all_grades["grade_year"] <= draft_year)
        ]

        if bd is not None and pd.notna(bd):
            min_year = bd.year + 18
            sept1_min = pd.Timestamp(f"{min_year}-09-01")
            if sept1_min < bd + pd.DateOffset(years=18):
                min_year += 1
            seasons = seasons[seasons["grade_year"] >= min_year]
        seasons = seasons[seasons["grade_year"] >= draft_year - 5]

        for _, s in seasons.iterrows():
            yr = s["grade_year"]
            routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
            yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
            games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
            grades_off = pd.to_numeric(s.get("grades_offense", np.nan), errors="coerce")
            team = s.get("team_name", "")

            if routes < 200:
                continue

            age = compute_season_age(yr, bd) if bd is not None else np.nan
            yprr = yards / routes if routes > 0 else np.nan
            att = team_att_lookup.get((team, yr))
            yptpa = yards / att if att and att > 0 else np.nan
            ypg = yards / games if games > 0 else np.nan
            is_p5 = team in P5_TEAMS

            records.append({
                "name": name,
                "draft_year": draft_year,
                "grade_year": yr,
                "team_name": team,
                "age": round(age, 2) if not np.isnan(age) else np.nan,
                "grades_offense": grades_off,
                "routes": int(routes),
                "yards": float(yards),
                "games": int(games),
                "yprr": round(yprr, 4) if yprr is not None else np.nan,
                "yptpa": round(yptpa, 4) if yptpa is not None else np.nan,
                "ypg": round(ypg, 2) if ypg is not None else np.nan,
                "total_yards": float(yards),
                "is_p5": is_p5,
                "dynasty_value": dv,
            })

    return pd.DataFrame(records)


# ---- Aggregation functions ----

def _aggregate_metric(selected, metric_col):
    """Aggregate a metric across selected seasons using appropriate weighting."""
    vals = selected[metric_col]
    if vals.isna().all():
        return np.nan
    mask = vals.notna()
    if metric_col in ("yprr", "yptpa"):
        routes = selected["routes"]
        return np.average(vals[mask], weights=routes[mask]) if mask.any() else np.nan
    elif metric_col == "ypg":
        games = selected["games"]
        return np.average(vals[mask], weights=games[mask]) if mask.any() else np.nan
    elif metric_col == "total_yards":
        return vals.mean()
    return vals.mean()


def _apply_p5_filter(player_seasons, min_seasons=2):
    """Apply P5 filter: prefer P5 seasons if enough are available."""
    p5 = player_seasons[player_seasons["is_p5"]]
    return p5 if len(p5) >= min_seasons else player_seasons


def agg_best2(player_seasons, metric_col):
    """Best 2 seasons by grades_offense (P5 filter, 200+ routes already applied)."""
    if len(player_seasons) == 0:
        return np.nan
    eligible = _apply_p5_filter(player_seasons, min_seasons=2)
    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().sum() < 2:
        selected = eligible
    else:
        selected = eligible.loc[grades.nlargest(2).index]
    return _aggregate_metric(selected, metric_col)


def agg_best1(player_seasons, metric_col):
    """Single best season by grades_offense (P5 filter)."""
    if len(player_seasons) == 0:
        return np.nan
    eligible = _apply_p5_filter(player_seasons, min_seasons=1)
    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().sum() == 0:
        selected = eligible.head(1)
    else:
        selected = eligible.loc[[grades.idxmax()]]
    return _aggregate_metric(selected, metric_col)


def agg_career(player_seasons, metric_col):
    """All eligible seasons."""
    if len(player_seasons) == 0:
        return np.nan
    return _aggregate_metric(player_seasons, metric_col)


AGG_FUNCS = {
    "best2": agg_best2,
    "best1": agg_best1,
    "career": agg_career,
}


# ---- Adjustment ----

def apply_adjustment(season_df, metric_col, scheme, params):
    """Apply an age adjustment to a metric column, returning a modified copy."""
    df = season_df.copy()
    if scheme == "none":
        return df

    # Ensure the metric column is float so assignment doesn't fail
    df[metric_col] = df[metric_col].astype(float)

    if scheme in ("senior", "both"):
        age_thresh = params["senior_age_thresh"]
        discount = params["senior_discount"]
        mask = df["age"].notna() & (df["age"] >= age_thresh)
        df.loc[mask, metric_col] = df.loc[mask, metric_col] * (1 - discount)

    if scheme in ("freshman", "both"):
        age_thresh = params["freshman_age_thresh"]
        boost = params["freshman_boost"]
        mask = df["age"].notna() & (df["age"] <= age_thresh)
        df.loc[mask, metric_col] = df.loc[mask, metric_col] * (1 + boost)

    if scheme == "empirical":
        age_curve = params["age_curve"]
        ref_val = params["ref_val"]
        ages = df["age"].values
        vals = df[metric_col].values.astype(float)
        for i in range(len(df)):
            if pd.notna(ages[i]):
                expected = np.polyval(age_curve, ages[i])
                if expected > 0:
                    vals[i] = vals[i] * (ref_val / expected)
        df[metric_col] = vals

    return df


def fit_age_curve(season_df, metric_col):
    """Fit a quadratic age curve. Returns (coefficients, reference_value)."""
    valid = season_df[["age", metric_col]].dropna()
    if len(valid) < 20:
        return None, None
    coeffs = np.polyfit(valid["age"].values, valid[metric_col].values, 2)
    ref_val = np.polyval(coeffs, REFERENCE_AGE)
    return coeffs, ref_val


# ---- Evaluation ----

def evaluate_metric(dynasty, player_agg, metric_name):
    """Spearman with dynasty_value, AUC for >=Elite, AUC for >=Stud."""
    cols = ["name", "draft_year", "dynasty_value"]
    merged = dynasty[cols].merge(player_agg, on=["name", "draft_year"], how="inner")
    merged = merged.dropna(subset=[metric_name])

    if len(merged) < 20:
        return {"n": len(merged), "spearman": np.nan, "auc_elite": np.nan,
                "auc_stud": np.nan}

    dv = merged["dynasty_value"].values
    feat = merged[metric_name].values

    sp, _ = spearmanr(feat, dv)

    elite_label = (dv >= 75).astype(int)
    auc_elite = np.nan
    if 0 < elite_label.sum() < len(elite_label):
        auc_elite = roc_auc_score(elite_label, feat)

    stud_label = (dv >= 180).astype(int)
    auc_stud = np.nan
    if 0 < stud_label.sum() < len(stud_label):
        auc_stud = roc_auc_score(stud_label, feat)

    return {"n": len(merged), "spearman": round(sp, 4),
            "auc_elite": round(auc_elite, 4) if not np.isnan(auc_elite) else np.nan,
            "auc_stud": round(auc_stud, 4) if not np.isnan(auc_stud) else np.nan}


# ---- Main test loop ----

def compute_agg_values(season_df, metric_col, agg_name, agg_func):
    """Group by player and compute aggregated metric."""
    return (
        season_df.groupby(["name", "draft_year"])
        .apply(lambda g: agg_func(g, metric_col), include_groups=False)
        .reset_index(name=metric_col)
    )


def run_test(season_df, dynasty, base_metric):
    """Run all adjustment schemes x aggregation windows for one base metric."""
    results = []

    # Pre-fit empirical curve on unadjusted data
    coeffs, ref_val = fit_age_curve(season_df, base_metric)

    # Build list of (scheme_name, params, label) combos
    scheme_configs = [("none", {}, "-")]

    for age_thresh in SENIOR_AGE_THRESHOLDS:
        for discount in SENIOR_DISCOUNTS:
            p = {"senior_age_thresh": age_thresh, "senior_discount": discount}
            label = f"age>={age_thresh}, -{discount*100:.0f}%"
            scheme_configs.append(("senior", p, label))

    for age_thresh in FRESHMAN_AGE_THRESHOLDS:
        for boost in FRESHMAN_BOOSTS:
            p = {"freshman_age_thresh": age_thresh, "freshman_boost": boost}
            label = f"age<={age_thresh}, +{boost*100:.0f}%"
            scheme_configs.append(("freshman", p, label))

    for sr_age in SENIOR_AGE_THRESHOLDS:
        for sr_disc in SENIOR_DISCOUNTS:
            for fr_age in FRESHMAN_AGE_THRESHOLDS:
                for fr_boost in FRESHMAN_BOOSTS:
                    if fr_age >= sr_age:
                        continue
                    p = {
                        "senior_age_thresh": sr_age, "senior_discount": sr_disc,
                        "freshman_age_thresh": fr_age, "freshman_boost": fr_boost,
                    }
                    label = (f"sr>={sr_age}/-{sr_disc*100:.0f}%, "
                             f"fr<={fr_age}/+{fr_boost*100:.0f}%")
                    scheme_configs.append(("both", p, label))

    if coeffs is not None:
        peak_age = -coeffs[1] / (2 * coeffs[0])
        p = {"age_curve": coeffs, "ref_val": ref_val}
        label = f"quad norm to {REFERENCE_AGE} (peak~{peak_age:.1f})"
        scheme_configs.append(("empirical", p, label))

    for scheme, params, label in scheme_configs:
        adj = apply_adjustment(season_df, base_metric, scheme, params)

        for agg_name, agg_func in AGG_FUNCS.items():
            agg_vals = compute_agg_values(adj, base_metric, agg_name, agg_func)
            ev = evaluate_metric(dynasty, agg_vals, base_metric)
            results.append({
                "metric": base_metric,
                "agg": agg_name,
                "scheme": scheme,
                "params": label,
                **ev,
            })

    return pd.DataFrame(results)


def print_results_table(res, base_metric):
    """Print top results per aggregation window x scheme."""
    for agg_name in ["best2", "best1", "career"]:
        agg_res = res[res["agg"] == agg_name]
        if len(agg_res) == 0:
            continue
        print(f"\n  === {agg_name.upper()} ===")
        for scheme in ["none", "senior", "freshman", "both", "empirical"]:
            sub = agg_res[agg_res["scheme"] == scheme]
            if len(sub) == 0:
                continue
            best = sub.sort_values("spearman", ascending=False).head(2)
            print(f"\n    --- {scheme.upper()} ---")
            for _, r in best.iterrows():
                print(f"      {r['params']:45s}  Sp={r['spearman']:.4f}  "
                      f"AUC(E)={r['auc_elite']:.4f}  AUC(S)={r['auc_stud']:.4f}")


def main():
    print("Loading data...")
    all_grades = load_all_grades(range(2016, 2026))
    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value.csv"))
    dynasty = dynasty[dynasty["draft_year"].between(2018, 2024)].copy()

    birth_lookup, _, team_att_lookup, _ = build_lookups(all_grades)

    print("Building player-season data...")
    season_df = get_player_season_data(all_grades, dynasty, birth_lookup,
                                        team_att_lookup)
    print(f"  {len(season_df)} eligible player-seasons, "
          f"{season_df['name'].nunique()} players")

    ages = season_df["age"].dropna()
    print(f"  Age range: {ages.min():.1f} - {ages.max():.1f}, "
          f"median {ages.median():.1f}")
    print(f"  Seasons age <= 19: {(ages <= 19).sum()},  "
          f"age 19-21: {((ages > 19) & (ages < 21)).sum()},  "
          f"age >= 22: {(ages >= 22).sum()}")

    base_metrics = ["yprr", "yptpa", "ypg", "total_yards"]
    all_results = []

    for metric in base_metrics:
        valid = season_df[metric].notna().sum()
        print(f"\n{'='*75}")
        print(f"  {metric.upper()} ({valid} seasons with data)")
        print(f"{'='*75}")

        res = run_test(season_df, dynasty, metric)
        all_results.append(res)
        print_results_table(res, metric)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "age_adjusted_production_results.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n\nFull results saved to {out_path}")
    print(f"Total rows: {len(combined)}")

    # ---- Grand summary: best variant per metric x agg ----
    print(f"\n{'='*75}")
    print("GRAND SUMMARY: Best variant per metric x aggregation (by Spearman)")
    print(f"{'='*75}")
    for metric in base_metrics:
        print(f"\n  {metric.upper()}:")
        sub = combined[combined["metric"] == metric]
        for agg_name in ["best2", "best1", "career"]:
            agg_sub = sub[sub["agg"] == agg_name]
            baseline = agg_sub[agg_sub["scheme"] == "none"].iloc[0]
            best = agg_sub.sort_values("spearman", ascending=False).iloc[0]
            delta_sp = best["spearman"] - baseline["spearman"]
            print(f"    {agg_name:8s}  base Sp={baseline['spearman']:.4f}  "
                  f"best Sp={best['spearman']:.4f} ({delta_sp:+.4f})  "
                  f"AUC(E)={best['auc_elite']:.4f}  AUC(S)={best['auc_stud']:.4f}  "
                  f"[{best['scheme']}: {best['params']}]")

    # ---- Also by AUC(Elite) ----
    print(f"\n{'='*75}")
    print("GRAND SUMMARY: Best variant per metric x aggregation (by AUC Elite)")
    print(f"{'='*75}")
    for metric in base_metrics:
        print(f"\n  {metric.upper()}:")
        sub = combined[combined["metric"] == metric]
        for agg_name in ["best2", "best1", "career"]:
            agg_sub = sub[sub["agg"] == agg_name]
            baseline = agg_sub[agg_sub["scheme"] == "none"].iloc[0]
            best = agg_sub.sort_values("auc_elite", ascending=False).iloc[0]
            delta = best["auc_elite"] - baseline["auc_elite"]
            print(f"    {agg_name:8s}  base AUC(E)={baseline['auc_elite']:.4f}  "
                  f"best AUC(E)={best['auc_elite']:.4f} ({delta:+.4f})  "
                  f"Sp={best['spearman']:.4f}  "
                  f"[{best['scheme']}: {best['params']}]")


if __name__ == "__main__":
    main()
