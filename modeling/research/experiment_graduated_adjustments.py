#!/usr/bin/env python3
"""
Expanded age adjustment grid search with graduated per-age-class adjustments.

Tests wider discount/boost ranges (5-30%) and applies adjustments to ALL age
classes (freshman, sophomore, junior, senior), not just the extremes.

Focuses on the 3 strongest metric/agg combos:
  - best1 YPRR
  - best2 YPRR
  - career YPTPA

Age classes:
  - Freshman: age < 19.5
  - Sophomore: 19.5 <= age < 20.5
  - Junior: 20.5 <= age < 21.5
  - Senior: age >= 21.5

Usage:
  python modeling/test_graduated_adjustments.py
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

# Age class boundaries
AGE_BINS = {
    "freshman": (0, 19.5),
    "sophomore": (19.5, 20.5),
    "junior": (20.5, 21.5),
    "senior": (21.5, 99),
}

# Adjustment magnitudes to test per age class
# Positive = boost, negative = discount
FRESHMAN_ADJS = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SOPHOMORE_ADJS = [0, 0.05, 0.10, 0.15, -0.05, -0.10]
JUNIOR_ADJS = [0, -0.05, -0.10, -0.15, -0.20, -0.25]
SENIOR_ADJS = [0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]

# Metric/agg combos to test
COMBOS = [
    ("yprr", "best1"),
    ("yprr", "best2"),
    ("yptpa", "career"),
]


def compute_season_age(grade_year, birthdate):
    if birthdate is None or pd.isna(birthdate) or pd.isna(grade_year):
        return np.nan
    sept1 = pd.Timestamp(f"{int(grade_year)}-09-01")
    return (sept1 - birthdate).days / 365.25


def get_player_season_data(all_grades, dynasty, birth_lookup, team_att_lookup):
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

            records.append({
                "name": name, "draft_year": draft_year,
                "grade_year": yr, "team_name": team,
                "age": round(age, 2) if not np.isnan(age) else np.nan,
                "grades_offense": grades_off,
                "routes": int(routes), "yards": float(yards),
                "games": int(games),
                "yprr": round(yprr, 4) if yprr is not None else np.nan,
                "yptpa": round(yptpa, 4) if yptpa is not None else np.nan,
                "is_p5": team in P5_TEAMS,
                "dynasty_value": dv,
            })

    return pd.DataFrame(records)


def get_age_class(age):
    if pd.isna(age):
        return None
    for cls, (lo, hi) in AGE_BINS.items():
        if lo <= age < hi:
            return cls
    return None


def apply_graduated_adjustment(season_df, metric_col, adj_dict):
    """Apply per-age-class multiplicative adjustments.

    adj_dict: {"freshman": 0.15, "sophomore": 0.05, "junior": -0.10, "senior": -0.15}
    Positive = boost (multiply by 1+adj), negative = discount (multiply by 1+adj).
    """
    df = season_df.copy()
    df[metric_col] = df[metric_col].astype(float)

    for cls, (lo, hi) in AGE_BINS.items():
        adj = adj_dict.get(cls, 0)
        if adj == 0:
            continue
        mask = df["age"].notna() & (df["age"] >= lo) & (df["age"] < hi)
        df.loc[mask, metric_col] = df.loc[mask, metric_col] * (1 + adj)

    return df


def _apply_p5_filter(player_seasons, min_seasons=2):
    p5 = player_seasons[player_seasons["is_p5"]]
    return p5 if len(p5) >= min_seasons else player_seasons


def _aggregate_metric(selected, metric_col):
    vals = selected[metric_col]
    if vals.isna().all():
        return np.nan
    mask = vals.notna()
    if metric_col in ("yprr", "yptpa"):
        routes = selected["routes"]
        return np.average(vals[mask], weights=routes[mask]) if mask.any() else np.nan
    return vals.mean()


def agg_best1(player_seasons, metric_col):
    if len(player_seasons) == 0:
        return np.nan
    eligible = _apply_p5_filter(player_seasons, min_seasons=1)
    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().sum() == 0:
        selected = eligible.head(1)
    else:
        selected = eligible.loc[[grades.idxmax()]]
    return _aggregate_metric(selected, metric_col)


def agg_best2(player_seasons, metric_col):
    if len(player_seasons) == 0:
        return np.nan
    eligible = _apply_p5_filter(player_seasons, min_seasons=2)
    grades = pd.to_numeric(eligible["grades_offense"], errors="coerce")
    if grades.notna().sum() < 2:
        selected = eligible
    else:
        selected = eligible.loc[grades.nlargest(2).index]
    return _aggregate_metric(selected, metric_col)


def agg_career(player_seasons, metric_col):
    if len(player_seasons) == 0:
        return np.nan
    return _aggregate_metric(player_seasons, metric_col)


AGG_FUNCS = {"best1": agg_best1, "best2": agg_best2, "career": agg_career}


def compute_agg_values(season_df, metric_col, agg_func):
    return (
        season_df.groupby(["name", "draft_year"])
        .apply(lambda g: agg_func(g, metric_col), include_groups=False)
        .reset_index(name=metric_col)
    )


def evaluate_metric(dynasty, player_agg, metric_name):
    merged = dynasty[["name", "draft_year", "dynasty_value"]].merge(
        player_agg, on=["name", "draft_year"], how="inner"
    )
    merged = merged.dropna(subset=[metric_name])

    if len(merged) < 20:
        return {"n": len(merged), "spearman": np.nan, "auc_elite": np.nan, "auc_stud": np.nan}

    dv = merged["dynasty_value"].values
    feat = merged[metric_name].values
    sp, _ = spearmanr(feat, dv)

    elite_label = (dv >= 75).astype(int)
    auc_elite = roc_auc_score(elite_label, feat) if 0 < elite_label.sum() < len(elite_label) else np.nan

    stud_label = (dv >= 180).astype(int)
    auc_stud = roc_auc_score(stud_label, feat) if 0 < stud_label.sum() < len(stud_label) else np.nan

    return {
        "n": len(merged),
        "spearman": round(sp, 4),
        "auc_elite": round(auc_elite, 4) if not np.isnan(auc_elite) else np.nan,
        "auc_stud": round(auc_stud, 4) if not np.isnan(auc_stud) else np.nan,
    }


def main():
    print("Loading data...")
    all_grades = load_all_grades(range(2016, 2026))
    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value.csv"))
    dynasty = dynasty[dynasty["draft_year"].between(2018, 2024)].copy()

    birth_lookup, _, team_att_lookup, _ = build_lookups(all_grades)

    print("Building player-season data...")
    season_df = get_player_season_data(all_grades, dynasty, birth_lookup, team_att_lookup)
    print(f"  {len(season_df)} eligible player-seasons, {season_df['name'].nunique()} players")

    # Age class distribution
    ages = season_df["age"].dropna()
    for cls, (lo, hi) in AGE_BINS.items():
        n = ((ages >= lo) & (ages < hi)).sum()
        print(f"  {cls}: {n} seasons")

    all_results = []

    # ---- Phase 1: Single-class sweeps ----
    # Hold other classes at 0, sweep one class at a time
    print("\n" + "=" * 75)
    print("PHASE 1: Single-class sweeps")
    print("=" * 75)

    class_sweeps = {
        "freshman": FRESHMAN_ADJS,
        "sophomore": SOPHOMORE_ADJS,
        "junior": JUNIOR_ADJS,
        "senior": SENIOR_ADJS,
    }

    for metric, agg_name in COMBOS:
        agg_func = AGG_FUNCS[agg_name]
        print(f"\n  --- {metric.upper()} {agg_name.upper()} ---")

        # Baseline
        agg_vals = compute_agg_values(season_df, metric, agg_func)
        ev = evaluate_metric(dynasty, agg_vals, metric)
        all_results.append({
            "metric": metric, "agg": agg_name, "scheme": "baseline",
            "fr_adj": 0, "so_adj": 0, "jr_adj": 0, "sr_adj": 0,
            "label": "no adjustment", **ev,
        })
        print(f"    baseline: Sp={ev['spearman']:.4f}  AUC(E)={ev['auc_elite']:.4f}  AUC(S)={ev['auc_stud']:.4f}")

        for cls, adjs in class_sweeps.items():
            for adj in adjs:
                if adj == 0:
                    continue
                adj_dict = {"freshman": 0, "sophomore": 0, "junior": 0, "senior": 0}
                adj_dict[cls] = adj
                adjusted = apply_graduated_adjustment(season_df, metric, adj_dict)
                agg_vals = compute_agg_values(adjusted, metric, agg_func)
                ev = evaluate_metric(dynasty, agg_vals, metric)
                label = f"{cls}={adj:+.0%}"
                all_results.append({
                    "metric": metric, "agg": agg_name, "scheme": f"single_{cls}",
                    "fr_adj": adj_dict["freshman"], "so_adj": adj_dict["sophomore"],
                    "jr_adj": adj_dict["junior"], "sr_adj": adj_dict["senior"],
                    "label": label, **ev,
                })

            # Print best for this class
            cls_results = [r for r in all_results
                           if r["metric"] == metric and r["agg"] == agg_name
                           and r["scheme"] == f"single_{cls}"]
            if cls_results:
                best = max(cls_results, key=lambda x: x["spearman"] if pd.notna(x["spearman"]) else -1)
                print(f"    best {cls}: {best['label']:20s}  Sp={best['spearman']:.4f}  "
                      f"AUC(E)={best['auc_elite']:.4f}  AUC(S)={best['auc_stud']:.4f}")

    # ---- Phase 2: Graduated combos ----
    # Test combinations of the best single-class adjustments
    print("\n" + "=" * 75)
    print("PHASE 2: Graduated combinations (all 4 classes)")
    print("=" * 75)

    # Smaller grid for combos: pick promising values from phase 1
    FR_COMBO = [0, 0.10, 0.15, 0.20, 0.25]
    SO_COMBO = [0, 0.05, 0.10, -0.05]
    JR_COMBO = [0, -0.05, -0.10, -0.15, -0.20]
    SR_COMBO = [0, -0.10, -0.15, -0.20, -0.25]

    total_combos = len(FR_COMBO) * len(SO_COMBO) * len(JR_COMBO) * len(SR_COMBO)
    print(f"  Testing {total_combos} combinations x {len(COMBOS)} metric/agg combos...")

    combo_count = 0
    for metric, agg_name in COMBOS:
        agg_func = AGG_FUNCS[agg_name]
        best_sp = -1
        best_config = None

        for fr in FR_COMBO:
            for so in SO_COMBO:
                for jr in JR_COMBO:
                    for sr in SR_COMBO:
                        # Skip all-zeros (already have baseline)
                        if fr == 0 and so == 0 and jr == 0 and sr == 0:
                            continue

                        adj_dict = {"freshman": fr, "sophomore": so,
                                    "junior": jr, "senior": sr}
                        adjusted = apply_graduated_adjustment(season_df, metric, adj_dict)
                        agg_vals = compute_agg_values(adjusted, metric, agg_func)
                        ev = evaluate_metric(dynasty, agg_vals, metric)

                        parts = []
                        if fr != 0: parts.append(f"fr={fr:+.0%}")
                        if so != 0: parts.append(f"so={so:+.0%}")
                        if jr != 0: parts.append(f"jr={jr:+.0%}")
                        if sr != 0: parts.append(f"sr={sr:+.0%}")
                        label = ", ".join(parts)

                        all_results.append({
                            "metric": metric, "agg": agg_name, "scheme": "graduated",
                            "fr_adj": fr, "so_adj": so, "jr_adj": jr, "sr_adj": sr,
                            "label": label, **ev,
                        })

                        if pd.notna(ev["spearman"]) and ev["spearman"] > best_sp:
                            best_sp = ev["spearman"]
                            best_config = {**ev, "label": label,
                                           "fr": fr, "so": so, "jr": jr, "sr": sr}

                        combo_count += 1
                        if combo_count % 500 == 0:
                            print(f"    ...{combo_count} combos tested")

        print(f"\n  {metric.upper()} {agg_name.upper()} best graduated:")
        print(f"    {best_config['label']}")
        print(f"    Sp={best_config['spearman']:.4f}  AUC(E)={best_config['auc_elite']:.4f}  "
              f"AUC(S)={best_config['auc_stud']:.4f}")

    # Save all results
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(DATA_DIR, "graduated_adjustment_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {out_path}")

    # ---- Grand summary ----
    print(f"\n{'=' * 75}")
    print("GRAND SUMMARY: Top 10 overall by Spearman")
    print(f"{'=' * 75}")
    top10 = results_df.sort_values("spearman", ascending=False).head(10)
    for _, r in top10.iterrows():
        print(f"  {r['metric']:6s} {r['agg']:7s}  {r['label']:45s}  "
              f"Sp={r['spearman']:.4f}  AUC(E)={r['auc_elite']:.4f}  AUC(S)={r['auc_stud']:.4f}")

    print(f"\n{'=' * 75}")
    print("GRAND SUMMARY: Top 10 overall by AUC(Stud)")
    print(f"{'=' * 75}")
    top10_s = results_df.sort_values("auc_stud", ascending=False).head(10)
    for _, r in top10_s.iterrows():
        print(f"  {r['metric']:6s} {r['agg']:7s}  {r['label']:45s}  "
              f"Sp={r['spearman']:.4f}  AUC(E)={r['auc_elite']:.4f}  AUC(S)={r['auc_stud']:.4f}")

    # Per-combo best
    print(f"\n{'=' * 75}")
    print("BEST PER METRIC/AGG")
    print(f"{'=' * 75}")
    for metric, agg_name in COMBOS:
        sub = results_df[(results_df["metric"] == metric) & (results_df["agg"] == agg_name)]
        baseline = sub[sub["scheme"] == "baseline"].iloc[0]
        best_sp = sub.sort_values("spearman", ascending=False).iloc[0]
        best_auc_s = sub.sort_values("auc_stud", ascending=False).iloc[0]
        print(f"\n  {metric.upper()} {agg_name.upper()}:")
        print(f"    baseline:    Sp={baseline['spearman']:.4f}  AUC(E)={baseline['auc_elite']:.4f}  AUC(S)={baseline['auc_stud']:.4f}")
        print(f"    best(Sp):    Sp={best_sp['spearman']:.4f}  AUC(E)={best_sp['auc_elite']:.4f}  AUC(S)={best_sp['auc_stud']:.4f}  [{best_sp['label']}]")
        print(f"    best(AUC-S): Sp={best_auc_s['spearman']:.4f}  AUC(E)={best_auc_s['auc_elite']:.4f}  AUC(S)={best_auc_s['auc_stud']:.4f}  [{best_auc_s['label']}]")


if __name__ == "__main__":
    main()
