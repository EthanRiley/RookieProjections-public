#!/usr/bin/env python3
"""
Head-to-head efficiency leak test: ba_yptpa vs yprr2.0_200rt.

For each variant, compute residual signal after progressively controlling for:
  1. Model features only
  2. + own breakout magnitude (YPTPA mag for ba_yptpa, YPRR mag for yprr)
  3. + the OTHER metric's breakout magnitude
  4. + both magnitudes
  5. + career_yprr
  6. + career_yprr + own magnitude

This isolates whether the suppressor effect is unique to YPTPA or applies equally to YPRR.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

MODEL_FEATS = [
    "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
    "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
]


def _season_age(birthdate, year):
    sept1 = pd.Timestamp(f"{year}-09-01")
    return round((sept1 - birthdate).days / 365.25, 2)


def residual_spearman(df, feat_col, control_cols):
    """Compute Spearman of feat_col residual (after removing controls) with tier."""
    cols = [feat_col] + control_cols + ["tier_ordinal"]
    sub = df[cols].dropna()
    if len(sub) < 30:
        return np.nan, 0
    rank_feat = rankdata(sub[feat_col].values)
    rank_tier = rankdata(sub["tier_ordinal"].values)
    ctrl_ranks = np.column_stack([rankdata(sub[c].values) for c in control_cols])
    X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
    z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
    resid = rank_feat - X @ z
    sp, _ = spearmanr(resid, rank_tier)
    return round(sp, 3), len(sub)


def main():
    from aggregation.aggregate_college_stats import (
        load_all_grades, get_player_seasons, build_lookups,
    )

    print("Loading data...")
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, _, team_att_lookup, team_games_lookup = build_lookups(all_grades)

    dynasty = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    dynasty["tier_ordinal"] = dynasty["computed_tier"].map(TIER_ORDER)
    dynasty = dynasty.dropna(subset=["tier_ordinal"]).copy()
    dynasty["tier_ordinal"] = dynasty["tier_ordinal"].astype(int)

    # Compute both breakout ages + magnitudes at breakout
    ba_yptpa = []
    mag_yptpa = []  # YPTPA value at breakout season
    ba_yprr = []
    mag_yprr = []   # YPRR value at breakout season
    mag_yprr_at_yptpa = []  # YPRR at the YPTPA breakout season
    mag_yptpa_at_yprr = []  # YPTPA at the YPRR breakout season

    for _, row in dynasty.iterrows():
        name, draft_year = row["name"], row["draft_year"]
        birthdate = birth_lookup.get((name, draft_year))
        seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)

        _ba_yptpa = np.nan
        _mag_yptpa = np.nan
        _mag_yprr_at_yptpa = np.nan
        _ba_yprr = np.nan
        _mag_yprr = np.nan
        _mag_yptpa_at_yprr = np.nan

        if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
            for _, s in seasons.sort_values("grade_year").iterrows():
                yr = s["grade_year"]
                yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
                games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
                routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
                team = s.get("team_name", "")
                att = team_att_lookup.get((team, yr))
                tg = team_games_lookup.get((team, yr))

                # Compute metrics for this season
                yptpa_val = None
                yprr_val = None
                if att and att > 0 and games > 0 and tg and tg > 0:
                    yptpa_val = (yards / games) / (att / tg)
                if routes > 0:
                    yprr_val = yards / routes

                # YPTPA breakout: >= 1.4, 8+ games
                if pd.isna(_ba_yptpa) and games >= 8 and yptpa_val is not None and yptpa_val >= 1.4:
                    _ba_yptpa = _season_age(birthdate, yr)
                    _mag_yptpa = round(yptpa_val, 4)
                    _mag_yprr_at_yptpa = round(yprr_val, 4) if yprr_val is not None else np.nan

                # YPRR breakout: >= 2.0, 200+ routes
                if pd.isna(_ba_yprr) and routes >= 200 and yprr_val is not None and yprr_val >= 2.0:
                    _ba_yprr = _season_age(birthdate, yr)
                    _mag_yprr = round(yprr_val, 4)
                    _mag_yptpa_at_yprr = round(yptpa_val, 4) if yptpa_val is not None else np.nan

        ba_yptpa.append(_ba_yptpa)
        mag_yptpa.append(_mag_yptpa)
        mag_yprr_at_yptpa.append(_mag_yprr_at_yptpa)
        ba_yprr.append(_ba_yprr)
        mag_yprr.append(_mag_yprr)
        mag_yptpa_at_yprr.append(_mag_yptpa_at_yprr)

    dynasty["ba_yptpa"] = ba_yptpa
    dynasty["mag_yptpa"] = mag_yptpa
    dynasty["mag_yprr_at_yptpa"] = mag_yprr_at_yptpa
    dynasty["ba_yprr200"] = ba_yprr
    dynasty["mag_yprr"] = mag_yprr
    dynasty["mag_yptpa_at_yprr"] = mag_yptpa_at_yprr

    # Impute breakout ages (NaN = max + 1)
    for col in ["ba_yptpa", "ba_yprr200"]:
        mx = dynasty[col].max()
        dynasty[f"{col}_imp"] = dynasty[col].fillna(round(mx + 1, 2))

    # Impute magnitudes (NaN = 0 for magnitude — didn't break out)
    for col in ["mag_yptpa", "mag_yprr", "mag_yprr_at_yptpa", "mag_yptpa_at_yprr"]:
        dynasty[f"{col}_imp"] = dynasty[col].fillna(0)

    # =====================================================================
    # Coverage comparison
    # =====================================================================
    print("\n" + "=" * 90)
    print("COVERAGE")
    print("=" * 90)
    for col in ["ba_yptpa", "ba_yprr200", "mag_yptpa", "mag_yprr"]:
        valid = dynasty[col].notna().sum()
        print(f"  {col:<25s}: {valid}/{len(dynasty)} ({valid/len(dynasty):.1%})")

    # =====================================================================
    # Head-to-head efficiency leak test
    # =====================================================================
    print("\n" + "=" * 90)
    print("EFFICIENCY LEAK TEST: ba_yptpa vs yprr2.0_200rt")
    print("=" * 90)

    breakout_variants = {
        "ba_yptpa_imp": {
            "own_mag": "mag_yptpa_imp",
            "cross_mag": "mag_yprr_at_yptpa_imp",
            "other_ba_mag": "mag_yprr_imp",
        },
        "ba_yprr200_imp": {
            "own_mag": "mag_yprr_imp",
            "cross_mag": "mag_yptpa_at_yprr_imp",
            "other_ba_mag": "mag_yptpa_imp",
        },
    }

    tests = [
        ("1. Model feats only", lambda v: MODEL_FEATS),
        ("2. + own magnitude", lambda v: MODEL_FEATS + [v["own_mag"]]),
        ("3. + cross magnitude", lambda v: MODEL_FEATS + [v["cross_mag"]]),
        ("4. + other BA magnitude", lambda v: MODEL_FEATS + [v["other_ba_mag"]]),
        ("5. + own + other mag", lambda v: MODEL_FEATS + [v["own_mag"], v["other_ba_mag"]]),
        ("6. + all magnitudes", lambda v: MODEL_FEATS + [v["own_mag"], v["cross_mag"], v["other_ba_mag"]]),
    ]

    print(f"\n  {'Test':<30s} {'ba_yptpa':>12s} {'yprr2.0_200rt':>15s} {'Difference':>12s}")
    print("  " + "-" * 75)

    for test_name, ctrl_fn in tests:
        results = {}
        for ba_col, variant_info in breakout_variants.items():
            ctrl = list(dict.fromkeys(ctrl_fn(variant_info)))
            sp, n = residual_spearman(dynasty, ba_col, ctrl)
            results[ba_col] = sp

        sp_yptpa = results["ba_yptpa_imp"]
        sp_yprr = results["ba_yprr200_imp"]
        diff = sp_yprr - sp_yptpa if pd.notna(sp_yptpa) and pd.notna(sp_yprr) else np.nan
        diff_str = f"{diff:+.3f}" if pd.notna(diff) else "N/A"
        print(f"  {test_name:<30s} {sp_yptpa:>+12.3f} {sp_yprr:>+15.3f} {diff_str:>12s}")

    # =====================================================================
    # Magnitude correlation analysis
    # =====================================================================
    print("\n" + "=" * 90)
    print("MAGNITUDE CORRELATIONS")
    print("=" * 90)

    mag_cols = ["mag_yptpa", "mag_yprr", "mag_yprr_at_yptpa", "mag_yptpa_at_yprr",
                "career_yprr", "career_targeted_qb_rating"]

    print(f"\n  How correlated are the magnitude signals with existing model features?\n")
    print(f"  {'Magnitude':<25s} {'career_yprr':>12s} {'career_tqbr':>12s} {'draft_cap':>10s}")
    print("  " + "-" * 65)

    for col in ["mag_yptpa_imp", "mag_yprr_imp", "mag_yprr_at_yptpa_imp", "mag_yptpa_at_yprr_imp"]:
        row_vals = []
        for model_col in ["career_yprr", "career_targeted_qb_rating", "draft_capital"]:
            both = dynasty[[col, model_col]].dropna()
            if len(both) > 10:
                sp, _ = spearmanr(both[col], both[model_col])
                row_vals.append(f"{sp:>+12.3f}")
            else:
                row_vals.append(f"{'N/A':>12s}")
        label = col.replace("_imp", "")
        print(f"  {label:<25s} {''.join(row_vals)}")

    # =====================================================================
    # Suppressor effect size
    # =====================================================================
    print("\n" + "=" * 90)
    print("SUPPRESSOR EFFECT SIZE")
    print("=" * 90)
    print("\n  How much does each breakout age improve when controlling for its own magnitude?\n")

    for ba_col, variant_info in breakout_variants.items():
        label = ba_col.replace("_imp", "")
        sp_base, n = residual_spearman(dynasty, ba_col, MODEL_FEATS)
        sp_supp, _ = residual_spearman(dynasty, ba_col, MODEL_FEATS + [variant_info["own_mag"]])
        improvement = sp_supp - sp_base if pd.notna(sp_base) and pd.notna(sp_supp) else np.nan
        pct = (improvement / abs(sp_base) * 100) if pd.notna(improvement) and sp_base != 0 else np.nan
        print(f"  {label:<20s}: {sp_base:+.3f} -> {sp_supp:+.3f}  "
              f"(improvement: {improvement:+.3f}, {pct:+.1f}%)")

    # =====================================================================
    # Bootstrap confidence intervals
    # =====================================================================
    print("\n" + "=" * 90)
    print("BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
    print("=" * 90)
    print("\n  Can we distinguish ba_yptpa from yprr2.0_200rt given sampling noise?\n")

    np.random.seed(42)
    n_boot = 1000

    for test_name, ctrl_fn in [
        ("Model feats only", lambda v: MODEL_FEATS),
        ("+ own magnitude", lambda v: MODEL_FEATS + [v["own_mag"]]),
    ]:
        boot_diffs = []
        for _ in range(n_boot):
            idx = np.random.choice(len(dynasty), size=len(dynasty), replace=True)
            boot = dynasty.iloc[idx].copy()

            results = {}
            for ba_col, variant_info in breakout_variants.items():
                ctrl = list(dict.fromkeys(ctrl_fn(variant_info)))
                sp, _ = residual_spearman(boot, ba_col, ctrl)
                results[ba_col] = sp

            if pd.notna(results["ba_yptpa_imp"]) and pd.notna(results["ba_yprr200_imp"]):
                boot_diffs.append(results["ba_yptpa_imp"] - results["ba_yprr200_imp"])

        diffs = np.array(boot_diffs)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        mean_diff = np.mean(diffs)
        pct_yptpa_wins = np.mean(diffs < 0) * 100  # negative = stronger (lower is better for age)

        print(f"  {test_name}:")
        print(f"    Mean difference (yptpa - yprr): {mean_diff:+.3f}")
        print(f"    95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        print(f"    ba_yptpa wins: {pct_yptpa_wins:.1f}% of bootstraps")
        print()


if __name__ == "__main__":
    main()
