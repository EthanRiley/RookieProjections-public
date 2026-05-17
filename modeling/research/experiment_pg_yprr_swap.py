#!/usr/bin/env python3
"""
Test replacing best1_yprr_graduated with pg_yprr_graduated in the feature set.

The prior peak-gated analysis added YPRR variants as *extra* features alongside
the existing best1_yprr_graduated anchor, creating duplicates. This script
properly *swaps* the YPRR anchor to test peak-gated YPRR selection.

Uses the same data engineering pipeline as test_peak_gated_selection.py.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
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


def engineer_season_features(qual, adot_catch_coef):
    q = qual.copy()
    q["catch_pct_adot_adj"] = q["caught_percent"] - np.polyval(adot_catch_coef, q["avg_depth_of_target"])
    return q


def aggregate_yprr_and_cpaa(player_key, draft_year, birthdate, qual_seasons):
    """Aggregate YPRR + catch_pct_adot_adj using best1, peak-gated, and pure peak."""
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

    # Compute YPRR per season
    for idx, row in eligible.iterrows():
        yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
        rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
        eligible.at[idx, "yprr_raw"] = yards / rts if rts > 0 else np.nan

    result = {}

    # --- best1: highest grades_offense season ---
    best1_row = eligible.loc[grades.idxmax()]
    yards = pd.to_numeric(best1_row.get("yards", 0), errors="coerce") or 0
    rts = pd.to_numeric(best1_row.get("routes", 0), errors="coerce") or 0
    best1_yprr = yards / rts if rts > 0 else np.nan
    if pd.notna(best1_yprr):
        result["best1_yprr"] = round(float(best1_yprr), 4)

    # --- peak-gated: max age-adjusted YPRR from seasons with grade >= 80, fallback to best1 ---
    gated = eligible[grades >= QUALITY_GATE]
    if len(gated) > 0:
        for idx, row in gated.iterrows():
            yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
            rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
            gated.at[idx, "yprr_raw"] = yards / rts if rts > 0 else np.nan
        gated_vals = gated["yprr_raw"].dropna()
        if len(gated_vals) > 0:
            # Select based on age-adjusted YPRR
            if birthdate is not None and pd.notna(birthdate):
                best_adj = np.nan
                pg_idx = None
                for idx in gated_vals.index:
                    yr = gated.loc[idx].get("grade_year") if hasattr(gated.loc[idx], "get") else None
                    age = get_age_on_sept1(birthdate, yr) if yr is not None else None
                    adj = float(gated_vals.loc[idx])
                    if age is not None:
                        for (lo, hi), mult in GRADUATED_ADJ.items():
                            if lo <= age < hi:
                                adj = adj * mult
                                break
                    if np.isnan(best_adj) or adj > best_adj:
                        best_adj = adj
                        pg_idx = idx
            else:
                pg_idx = gated_vals.idxmax()
            pg_yprr = float(gated_vals.loc[pg_idx])
            pg_row = gated.loc[pg_idx]
        else:
            pg_yprr = best1_yprr
            pg_row = best1_row
    else:
        pg_yprr = best1_yprr
        pg_row = best1_row

    if pd.notna(pg_yprr):
        result["pg_yprr"] = round(float(pg_yprr), 4)

    # --- pure peak: max YPRR from ALL eligible seasons ---
    all_vals = eligible["yprr_raw"].dropna()
    if len(all_vals) > 0:
        peak_idx = all_vals.idxmax()
        peak_yprr = float(all_vals.max())
        peak_row = eligible.loc[peak_idx]
    else:
        peak_yprr = best1_yprr
        peak_row = best1_row

    if pd.notna(peak_yprr):
        result["peak_yprr"] = round(float(peak_yprr), 4)

    # --- Graduated variants ---
    if birthdate is not None and pd.notna(birthdate):
        for prefix, val, row_source in [
            ("best1", best1_yprr, best1_row),
            ("pg", pg_yprr, pg_row),
            ("peak", peak_yprr, peak_row),
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
                    adjusted = val * mult  # YPRR center = 0
                    result[f"{prefix}_yprr_graduated"] = round(adjusted, 4)
                    break

    # --- catch_pct_adot_adj: peak-gated ---
    if "catch_pct_adot_adj" in eligible.columns:
        # best1 cpaa
        if "catch_pct_adot_adj" in best1_row.index:
            b1_cpaa = best1_row["catch_pct_adot_adj"]
            if pd.notna(b1_cpaa):
                result["best1_catch_pct_adot_adj"] = round(float(b1_cpaa), 4)

        # pg cpaa (select based on age-adjusted value)
        if len(gated) > 0 and "catch_pct_adot_adj" in gated.columns:
            cpaa_vals = gated["catch_pct_adot_adj"].dropna()
            if len(cpaa_vals) > 0:
                if birthdate is not None and pd.notna(birthdate):
                    best_adj = np.nan
                    cpaa_pg_idx = None
                    for idx in cpaa_vals.index:
                        yr = gated.loc[idx].get("grade_year") if hasattr(gated.loc[idx], "get") else None
                        adj = float(cpaa_vals.loc[idx])
                        if yr is not None:
                            age = get_age_on_sept1(birthdate, yr)
                            if age is not None:
                                for (lo, hi), mult in GRADUATED_ADJ.items():
                                    if lo <= age < hi:
                                        adj = adj * mult
                                        break
                        if np.isnan(best_adj) or adj > best_adj:
                            best_adj = adj
                            cpaa_pg_idx = idx
                else:
                    cpaa_pg_idx = cpaa_vals.idxmax()
                cpaa_pg_val = float(cpaa_vals.loc[cpaa_pg_idx])
                cpaa_pg_row = gated.loc[cpaa_pg_idx]
            else:
                cpaa_pg_val = b1_cpaa if pd.notna(b1_cpaa) else np.nan
                cpaa_pg_row = best1_row
        else:
            cpaa_pg_val = best1_row.get("catch_pct_adot_adj", np.nan)
            cpaa_pg_row = best1_row

        if pd.notna(cpaa_pg_val):
            result["pg_catch_pct_adot_adj"] = round(float(cpaa_pg_val), 4)

        # Graduated variants for cpaa
        if birthdate is not None and pd.notna(birthdate):
            for prefix, val, row_source in [
                ("best1", best1_row.get("catch_pct_adot_adj", np.nan), best1_row),
                ("pg", cpaa_pg_val, cpaa_pg_row),
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
                        adjusted = val * mult  # center = 0 for aDOT-adjusted
                        result[f"{prefix}_catch_pct_adot_adj_graduated"] = round(adjusted, 4)
                        break

    # Track divergence
    result["pg_yprr_same_as_best1"] = 1 if (
        pd.notna(best1_yprr) and pd.notna(pg_yprr) and abs(best1_yprr - pg_yprr) < 1e-6
    ) else 0
    result["n_gated_seasons"] = len(gated) if len(gated) > 0 else 0
    result["best1_grade_for_yprr"] = round(float(grades.max()), 1)
    if len(gated) > 0 and len(gated_vals) > 0:
        result["pg_yprr_grade"] = round(float(gated.loc[pg_idx, "grades_offense"]), 1)

    return result


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
    cp = cum_probs[idx]
    for ti in range(len(thresholds) - 1, 0, -1):
        cp[:, ti - 1] = np.maximum(cp[:, ti - 1], cp[:, ti])
    tier_probs = np.zeros((n_pred, n_tiers))
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


def main():
    print("=" * 85)
    print("  PEAK-GATED YPRR SWAP TEST")
    print(f"  Quality gate: grades_offense >= {QUALITY_GATE}")
    print("=" * 85)

    # Load data
    print("\n[1/4] Loading data...")
    ag = load_grades()
    qual = ag[ag["routes"] >= 200].copy()

    # aDOT regression for catch% adjustment
    cp, adot = ag["caught_percent"], ag["avg_depth_of_target"]
    m = cp.notna() & adot.notna()
    adot_catch_coef = np.polyfit(adot[m].values, cp[m].values, 1)

    # Engineer season features
    qual = engineer_season_features(qual, adot_catch_coef)

    # Load master
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_num"] = df["computed_tier"].map(TIER_ORDER)
    df["_join_key"] = df["name"].apply(normalize_name)

    ages = pd.read_csv(os.path.join(DATA_DIR, "draft_ages.csv"))
    ages["birthdate"] = pd.to_datetime(ages["birthdate"])
    birth_lookup = dict(zip(zip(ages["name"], ages["draft_year"]), ages["birthdate"]))

    # Aggregate YPRR + CPAA
    print("\n[2/4] Aggregating YPRR + CPAA per player...")
    results = []
    for _, row in df.iterrows():
        birthdate = birth_lookup.get((row["name"], row["draft_year"]))
        res = aggregate_yprr_and_cpaa(normalize_name(row["name"]), row["draft_year"], birthdate, qual)
        results.append(res)

    eng_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), eng_df.reset_index(drop=True)], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # Diagnostics
    if "pg_yprr_same_as_best1" in df.columns:
        diag = df.dropna(subset=["pg_yprr_same_as_best1"])
        same = int(diag["pg_yprr_same_as_best1"].sum())
        diff = len(diag) - same
        print(f"\n  Peak-gated YPRR selection divergence:")
        print(f"    Same as best1: {same}/{len(diag)} ({same/len(diag)*100:.0f}%)")
        print(f"    Different season: {diff}/{len(diag)} ({diff/len(diag)*100:.0f}%)")

    # Correlation between best1 and pg YPRR graduated
    if "best1_yprr_graduated" in df.columns and "pg_yprr_graduated" in df.columns:
        both = df[["best1_yprr_graduated", "pg_yprr_graduated"]].dropna()
        corr = both["best1_yprr_graduated"].corr(both["pg_yprr_graduated"])
        sp, _ = spearmanr(both["best1_yprr_graduated"], both["pg_yprr_graduated"])
        print(f"\n  Correlation best1 vs pg YPRR graduated:")
        print(f"    Pearson: {corr:.4f}")
        print(f"    Spearman: {sp:.4f}")

    # Combo tests: actually SWAP YPRR in the feature set
    print("\n[3/4] Running YPRR swap combo tests...")
    print("  (properly replacing best1_yprr_graduated in anchor features)")

    V11_FEATURES = [
        "draft_capital",
        "best1_yprr_graduated",
        "best2_contested_catch_rate",
        "best2_avoided_tackles_per_rec",
        "career_targeted_qb_rating",
        "best2_catch_pct_adot_adj",
    ]
    OTHER_ANCHORS = [
        "draft_capital",
        "best2_contested_catch_rate",
        "best2_avoided_tackles_per_rec",
    ]

    combos = {}

    # v11 baseline
    combos["v11 (best1_yprr_graduated)"] = V11_FEATURES[:]

    # Swap YPRR only, keep everything else
    for yprr_variant in ["pg_yprr_graduated", "peak_yprr_graduated",
                         "pg_yprr", "peak_yprr", "best1_yprr"]:
        if yprr_variant in df.columns and df[yprr_variant].notna().sum() > 50:
            feats = [yprr_variant if f == "best1_yprr_graduated" else f for f in V11_FEATURES]
            combos[f"YPRR => {yprr_variant}"] = feats

    # Also test: swap YPRR + swap QBR with pg_cpaa_graduated (the winning catch config)
    pg_cpaa_col = "pg_catch_pct_adot_adj_graduated"
    if pg_cpaa_col in df.columns:
        for yprr_variant in ["pg_yprr_graduated", "peak_yprr_graduated"]:
            if yprr_variant in df.columns:
                feats = OTHER_ANCHORS + [yprr_variant, pg_cpaa_col, "best2_catch_pct_adot_adj"]
                combos[f"YPRR => {yprr_variant} + QBR => pg_cpaa_grad"] = feats
        # Best known config from prior test: best1_yprr + pg_cpaa_grad (for reference)
        feats = OTHER_ANCHORS + ["best1_yprr_graduated", pg_cpaa_col, "best2_catch_pct_adot_adj"]
        combos[f"best1_yprr + QBR => pg_cpaa_grad (prior best)"] = feats
    else:
        print(f"  Note: {pg_cpaa_col} not in df — compound swap tests skipped.")

    # 4 anchors with swapped YPRR (no QBR, no catch%)
    for yprr_variant in ["pg_yprr_graduated", "peak_yprr_graduated"]:
        if yprr_variant in df.columns:
            feats = OTHER_ANCHORS + [yprr_variant]
            combos[f"4 anchors ({yprr_variant})"] = feats

    # Standard 4 anchors
    combos["4 anchors (best1_yprr_graduated)"] = OTHER_ANCHORS + ["best1_yprr_graduated"]

    core_cols = OTHER_ANCHORS + ["tier_num", "draft_year"]
    d = df.dropna(subset=core_cols).copy()
    years = sorted(d["draft_year"].unique())
    print(f"  n={len(d)} | years={[int(y) for y in years]}")

    print(f"\n  {'Combination':<60s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} "
          f"{'>=Stud':>8s} {'>=Start':>8s} {'#F':>4s}")
    print(f"  {'-'*60} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    results = []
    for name, feats in combos.items():
        valid_feats = [f for f in feats if f in d.columns]
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
        ll_str = f"{row['log_loss']:>8.3f}" if pd.notna(row['log_loss']) else "     nan"
        br_str = f"{row['brier']:>8.3f}" if pd.notna(row['brier']) else "     nan"
        ae_str = f"{row['elite_auc']:>8.3f}" if pd.notna(row['elite_auc']) else "     nan"
        as_str = f"{row['stud_auc']:>8.3f}" if pd.notna(row['stud_auc']) else "     nan"
        at_str = f"{row['starter_auc']:>8.3f}" if pd.notna(row['starter_auc']) else "     nan"
        print(f"  {name:<60s} {ll_str} {br_str} {ae_str} {as_str} {at_str} {row['n_feats']:>4d}")

    combo_df = pd.DataFrame(results)

    # Save results
    out_path = os.path.join(DATA_DIR, "outputs", "pg_yprr_swap_combos.csv")
    combo_df.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")

    # Summary
    print(f"\n{'=' * 85}")
    print("  SUMMARY")
    print(f"{'=' * 85}")

    v11_row = combo_df[combo_df["combo"].str.contains("v11")]
    if len(v11_row) > 0:
        v = v11_row.iloc[0]
        print(f"\n  v11 baseline: LL={v['log_loss']:.3f} Br={v['brier']:.3f} "
              f"Elite={v['elite_auc']:.3f} Stud={v['stud_auc']:.3f} Start={v['starter_auc']:.3f}")

    for _, r in combo_df.iterrows():
        if "v11" in r["combo"]:
            continue
        v11_ll = v11_row.iloc[0]["log_loss"] if len(v11_row) > 0 else np.nan
        if pd.notna(r["log_loss"]) and pd.notna(v11_ll):
            delta_pct = (r["log_loss"] - v11_ll) / v11_ll * 100
            print(f"  {r['combo']:<60s} LL={r['log_loss']:.3f} ({delta_pct:+.1f}%)")

    print(f"\n{'=' * 85}")


if __name__ == "__main__":
    main()
