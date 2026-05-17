#!/usr/bin/env python3
"""
Evaluate breakout magnitude features: YPRR at breakout and YPTPA at breakout.

Tests with yprr2.0_200rt as the breakout age metric.
Evaluates mag_yprr and mag_yptpa as standalone features and in combination.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

# Current model features
MODEL_FEATS = [
    "career_targeted_qb_rating", "career_yprr", "career_catch_pct_adot_adj",
    "best2_contested_catch_rate", "career_avoided_tackles_pg", "draft_capital",
]


def _season_age(birthdate, year):
    sept1 = pd.Timestamp(f"{year}-09-01")
    return round((sept1 - birthdate).days / 365.25, 2)


def evaluate_feature(df, col, control_feats=None):
    """Spearman, AUC, drift, and optionally residual."""
    valid = df[[col, "tier_ordinal", "hit", "draft_year"]].dropna()
    if len(valid) < 30:
        return None

    x = valid[col].values
    y = valid["tier_ordinal"].values
    y_hit = valid["hit"].values

    sp, _ = spearmanr(x, y)
    try:
        auc = roc_auc_score(y_hit, x)
        if auc < 0.5:
            auc = 1 - auc
    except ValueError:
        auc = np.nan

    years = valid["draft_year"].values
    early, late = years <= 2019, years >= 2020
    sp_e, _ = spearmanr(x[early], y[early]) if early.sum() > 10 else (np.nan, np.nan)
    sp_l, _ = spearmanr(x[late], y[late]) if late.sum() > 10 else (np.nan, np.nan)
    drift = abs(sp_e - sp_l) if pd.notna(sp_e) and pd.notna(sp_l) else np.nan

    result = {
        "spearman": round(sp, 3), "auc": round(auc, 3),
        "n": len(valid), "coverage": round(len(valid) / len(df), 3),
        "drift": round(drift, 3) if pd.notna(drift) else np.nan,
    }

    if control_feats:
        sub = df[[col] + control_feats + ["tier_ordinal"]].dropna()
        if len(sub) >= 30:
            rank_feat = rankdata(sub[col].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in control_feats])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid = rank_feat - X @ z
            sp_resid, _ = spearmanr(resid, rank_tier)
            result["residual"] = round(sp_resid, 3)

    return result


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
    dynasty["hit"] = (dynasty["tier_ordinal"] >= 3).astype(int)

    # Compute breakout ages + magnitudes at breakout for both metrics
    print("Computing breakout ages and magnitudes...")
    ba_yptpa, mag_yptpa_at_yptpa, mag_yprr_at_yptpa = [], [], []
    ba_yprr, mag_yptpa_at_yprr, mag_yprr_at_yprr = [], [], []

    for _, row in dynasty.iterrows():
        name, draft_year = row["name"], row["draft_year"]
        birthdate = birth_lookup.get((name, draft_year))
        seasons = get_player_seasons(all_grades, name, draft_year, birthdate=birthdate)

        _ba_yptpa, _mag_yptpa_ay, _mag_yprr_ay = np.nan, np.nan, np.nan
        _ba_yprr, _mag_yptpa_yr, _mag_yprr_yr = np.nan, np.nan, np.nan

        if birthdate is not None and pd.notna(birthdate) and len(seasons) > 0:
            for _, s in seasons.sort_values("grade_year").iterrows():
                yr = s["grade_year"]
                yards = pd.to_numeric(s.get("yards", 0), errors="coerce") or 0
                games = pd.to_numeric(s.get("player_game_count", 0), errors="coerce") or 0
                routes = pd.to_numeric(s.get("routes", 0), errors="coerce") or 0
                team = s.get("team_name", "")
                att = team_att_lookup.get((team, yr))
                tg = team_games_lookup.get((team, yr))

                yptpa_val = (yards / games) / (att / tg) if (att and att > 0 and games > 0 and tg and tg > 0) else None
                yprr_val = yards / routes if routes > 0 else None

                # YPTPA breakout
                if pd.isna(_ba_yptpa) and games >= 8 and yptpa_val is not None and yptpa_val >= 1.4:
                    _ba_yptpa = _season_age(birthdate, yr)
                    _mag_yptpa_ay = round(yptpa_val, 4)
                    _mag_yprr_ay = round(yprr_val, 4) if yprr_val is not None else np.nan

                # YPRR breakout
                if pd.isna(_ba_yprr) and routes >= 200 and yprr_val is not None and yprr_val >= 2.0:
                    _ba_yprr = _season_age(birthdate, yr)
                    _mag_yptpa_yr = round(yptpa_val, 4) if yptpa_val is not None else np.nan
                    _mag_yprr_yr = round(yprr_val, 4)

        ba_yptpa.append(_ba_yptpa)
        mag_yptpa_at_yptpa.append(_mag_yptpa_ay)
        mag_yprr_at_yptpa.append(_mag_yprr_ay)
        ba_yprr.append(_ba_yprr)
        mag_yptpa_at_yprr.append(_mag_yptpa_yr)
        mag_yprr_at_yprr.append(_mag_yprr_yr)

    dynasty["ba_yptpa"] = ba_yptpa
    dynasty["mag_yptpa_at_yptpa"] = mag_yptpa_at_yptpa
    dynasty["mag_yprr_at_yptpa"] = mag_yprr_at_yptpa
    dynasty["ba_yprr"] = ba_yprr
    dynasty["mag_yptpa_at_yprr"] = mag_yptpa_at_yprr
    dynasty["mag_yprr_at_yprr"] = mag_yprr_at_yprr

    # Impute breakout ages with max+1, magnitudes with 0
    for col in ["ba_yptpa", "ba_yprr"]:
        mx = dynasty[col].max()
        dynasty[f"{col}_imp"] = dynasty[col].fillna(round(mx + 1, 2))
    for col in ["mag_yptpa_at_yptpa", "mag_yprr_at_yptpa",
                "mag_yptpa_at_yprr", "mag_yprr_at_yprr"]:
        dynasty[f"{col}_imp"] = dynasty[col].fillna(0)

    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 1: COVERAGE")
    print("=" * 90)
    for col in ["ba_yptpa", "ba_yprr",
                "mag_yptpa_at_yptpa", "mag_yprr_at_yptpa",
                "mag_yptpa_at_yprr", "mag_yprr_at_yprr"]:
        valid = dynasty[col].notna().sum()
        print(f"  {col:<25s}: {valid}/{len(dynasty)} ({valid/len(dynasty):.1%})")

    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 2: STANDALONE EVALUATION OF MAGNITUDE FEATURES")
    print("=" * 90)

    mag_features = {
        "mag_yptpa_at_yptpa_imp": "YPTPA at YPTPA-breakout",
        "mag_yprr_at_yptpa_imp": "YPRR at YPTPA-breakout",
        "mag_yptpa_at_yprr_imp": "YPTPA at YPRR-breakout",
        "mag_yprr_at_yprr_imp": "YPRR at YPRR-breakout",
    }

    print(f"\n  {'Feature':<30s} {'Spearman':>10s} {'AUC':>8s} {'Resid':>8s} {'Drift':>8s} {'Cov':>6s}")
    print("  " + "-" * 75)

    for col, label in mag_features.items():
        res = evaluate_feature(dynasty, col, control_feats=MODEL_FEATS)
        if res:
            drift_str = f"{res['drift']:.3f}" if pd.notna(res.get('drift')) else "N/A"
            res_str = f"{res['residual']:+.3f}" if 'residual' in res else "N/A"
            print(f"  {label:<30s} {res['spearman']:>+10.3f} {res['auc']:>8.3f} "
                  f"{res_str:>8s} {drift_str:>8s} {res['coverage']:>6.1%}")

    # Also evaluate existing breakout ages for reference
    print("\n  Reference (existing features):")
    # career_yprr is already in MODEL_FEATS, so compute its residual excluding itself
    model_no_yprr = [f for f in MODEL_FEATS if f != "career_yprr"]
    for col, label, ctrl in [
        ("career_yprr", "career_yprr", model_no_yprr),
        ("ba_yptpa_imp", "ba_yptpa (imputed)", MODEL_FEATS),
        ("ba_yprr_imp", "ba_yprr (imputed)", MODEL_FEATS),
    ]:
        res = evaluate_feature(dynasty, col, control_feats=ctrl)
        if res:
            drift_str = f"{res['drift']:.3f}" if pd.notna(res.get('drift')) else "N/A"
            res_str = f"{res['residual']:+.3f}" if 'residual' in res else "N/A"
            print(f"  {label:<30s} {res['spearman']:>+10.3f} {res['auc']:>8.3f} "
                  f"{res_str:>8s} {drift_str:>8s} {res['coverage']:>6.1%}")

    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 3: RESIDUAL OF BREAKOUT AGE WITH MAGNITUDE IN MODEL")
    print("=" * 90)
    print("\n  How does adding a magnitude feature change breakout age's residual?")
    print("  (Using yprr2.0_200rt as breakout age)\n")

    ba_col = "ba_yprr_imp"
    print(f"  {'Control set':<50s} {'Residual':>10s}")
    print("  " + "-" * 65)

    test_sets = [
        ("Model feats only", MODEL_FEATS),
        ("+ mag_yptpa_at_yprr", MODEL_FEATS + ["mag_yptpa_at_yprr_imp"]),
        ("+ mag_yprr_at_yprr", MODEL_FEATS + ["mag_yprr_at_yprr_imp"]),
        ("+ both magnitudes at YPRR breakout", MODEL_FEATS + ["mag_yptpa_at_yprr_imp", "mag_yprr_at_yprr_imp"]),
        ("+ mag_yptpa_at_yptpa (cross-breakout)", MODEL_FEATS + ["mag_yptpa_at_yptpa_imp"]),
        ("+ mag_yprr_at_yptpa (cross-breakout)", MODEL_FEATS + ["mag_yprr_at_yptpa_imp"]),
        ("+ all 4 magnitudes", MODEL_FEATS + ["mag_yptpa_at_yprr_imp", "mag_yprr_at_yprr_imp",
                                                "mag_yptpa_at_yptpa_imp", "mag_yprr_at_yptpa_imp"]),
    ]

    for label, ctrl in test_sets:
        ctrl_dedup = list(dict.fromkeys(ctrl))
        sub = dynasty[[ba_col] + ctrl_dedup + ["tier_ordinal"]].dropna()
        if len(sub) < 30:
            print(f"  {label:<50s} {'N/A':>10s}")
            continue
        rank_feat = rankdata(sub[ba_col].values)
        rank_tier = rankdata(sub["tier_ordinal"].values)
        ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl_dedup])
        X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
        z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
        resid = rank_feat - X @ z
        sp_resid, _ = spearmanr(resid, rank_tier)
        print(f"  {label:<50s} {sp_resid:>+10.3f}")

    # Same for ba_yptpa
    print(f"\n  (Using ba_yptpa as breakout age)\n")
    ba_col = "ba_yptpa_imp"
    print(f"  {'Control set':<50s} {'Residual':>10s}")
    print("  " + "-" * 65)

    test_sets_yptpa = [
        ("Model feats only", MODEL_FEATS),
        ("+ mag_yptpa_at_yptpa", MODEL_FEATS + ["mag_yptpa_at_yptpa_imp"]),
        ("+ mag_yprr_at_yptpa", MODEL_FEATS + ["mag_yprr_at_yptpa_imp"]),
        ("+ both magnitudes at YPTPA breakout", MODEL_FEATS + ["mag_yptpa_at_yptpa_imp", "mag_yprr_at_yptpa_imp"]),
    ]

    for label, ctrl in test_sets_yptpa:
        ctrl_dedup = list(dict.fromkeys(ctrl))
        sub = dynasty[[ba_col] + ctrl_dedup + ["tier_ordinal"]].dropna()
        if len(sub) < 30:
            print(f"  {label:<50s} {'N/A':>10s}")
            continue
        rank_feat = rankdata(sub[ba_col].values)
        rank_tier = rankdata(sub["tier_ordinal"].values)
        ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in ctrl_dedup])
        X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
        z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
        resid = rank_feat - X @ z
        sp_resid, _ = spearmanr(resid, rank_tier)
        print(f"  {label:<50s} {sp_resid:>+10.3f}")

    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 4: CORRELATION BETWEEN MAGNITUDE FEATURES AND MODEL FEATURES")
    print("=" * 90)

    all_mag = ["mag_yptpa_at_yptpa_imp", "mag_yprr_at_yptpa_imp",
               "mag_yptpa_at_yprr_imp", "mag_yprr_at_yprr_imp"]
    ref_feats = MODEL_FEATS + ["ba_yptpa_imp", "ba_yprr_imp"]

    print(f"\n  {'Magnitude':<25s}", end="")
    for f in ref_feats:
        short = f.replace("career_", "c_").replace("best2_", "b2_").replace("_imp", "")[:12]
        print(f" {short:>12s}", end="")
    print()
    print("  " + "-" * (25 + 13 * len(ref_feats)))

    for mag in all_mag:
        label = mag.replace("_imp", "").replace("mag_", "")
        print(f"  {label:<25s}", end="")
        for f in ref_feats:
            both = dynasty[[mag, f]].dropna()
            if len(both) > 10:
                sp, _ = spearmanr(both[mag], both[f])
                print(f" {sp:>+12.3f}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()

    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 5: PROPOSED NEW FEATURE SET EVALUATION")
    print("=" * 90)
    print("\n  Testing: replace ba_yptpa with ba_yprr, add magnitude features\n")

    # Current feature set residual (ba_yptpa as breakout)
    current_feats = MODEL_FEATS + ["ba_yptpa_imp"]

    # Proposed sets
    proposed_sets = {
        "Current (ba_yptpa)": MODEL_FEATS + ["ba_yptpa_imp"],
        "ba_yprr only": MODEL_FEATS + ["ba_yprr_imp"],
        "ba_yprr + mag_yptpa": MODEL_FEATS + ["ba_yprr_imp", "mag_yptpa_at_yprr_imp"],
        "ba_yprr + mag_yprr": MODEL_FEATS + ["ba_yprr_imp", "mag_yprr_at_yprr_imp"],
        "ba_yprr + both_mag": MODEL_FEATS + ["ba_yprr_imp", "mag_yptpa_at_yprr_imp", "mag_yprr_at_yprr_imp"],
        "ba_yptpa + mag_yptpa": MODEL_FEATS + ["ba_yptpa_imp", "mag_yptpa_at_yptpa_imp"],
        "ba_yptpa + mag_yprr": MODEL_FEATS + ["ba_yptpa_imp", "mag_yprr_at_yptpa_imp"],
    }

    # For each proposed set, compute total predictive power via multivariate Spearman
    # (sum of absolute residual correlations of each feature)
    print(f"  {'Feature set':<35s} {'N feats':>8s} {'Sum |resid|':>12s}")
    print("  " + "-" * 60)

    for set_name, feat_set in proposed_sets.items():
        feat_set = list(dict.fromkeys(feat_set))
        sub = dynasty[feat_set + ["tier_ordinal"]].dropna()
        if len(sub) < 30:
            continue
        # For each feature, compute its residual after all others
        total_resid = 0
        for feat in feat_set:
            others = [f for f in feat_set if f != feat]
            rank_feat = rankdata(sub[feat].values)
            rank_tier = rankdata(sub["tier_ordinal"].values)
            ctrl_ranks = np.column_stack([rankdata(sub[f].values) for f in others])
            X = np.column_stack([ctrl_ranks, np.ones(len(sub))])
            z = np.linalg.lstsq(X, rank_feat, rcond=None)[0]
            resid = rank_feat - X @ z
            sp_resid, _ = spearmanr(resid, rank_tier)
            total_resid += abs(sp_resid)
        print(f"  {set_name:<35s} {len(feat_set):>8d} {total_resid:>12.3f}")

    # Save magnitudes for visualization
    save_cols = ["name", "draft_year", "computed_tier", "tier_ordinal",
                 "ba_yptpa", "ba_yprr",
                 "mag_yptpa_at_yptpa", "mag_yprr_at_yptpa",
                 "mag_yptpa_at_yprr", "mag_yprr_at_yprr"]
    dynasty[save_cols].to_csv(os.path.join(DATA_DIR, "breakout_magnitudes_by_player.csv"), index=False)
    print(f"\nSaved per-player magnitudes to breakout_magnitudes_by_player.csv")


if __name__ == "__main__":
    main()
