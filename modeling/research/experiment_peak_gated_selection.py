#!/usr/bin/env python3
"""
Test "peak-gated" season selection: pick the season where a stat peaks,
but only from seasons with grades_offense >= 80. If no season meets
the quality gate, fall back to highest grades_offense season.

This replaces the current "best1" approach (always highest grades_offense)
with a stat-specific peak that's gated behind a quality threshold.

Tests: catch_pct_adot_adj_graduated, cpaa_minus_drops_graduated,
       catch_minus_drops, clean_catch_rate, yprr_graduated, and
       the current best1 variants for comparison.
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
SENIOR_AGE_THRESHOLD = 21.5
SENIOR_DISCOUNT_PP = 10.0
QUALITY_GATE = 80.0  # Minimum grades_offense for peak selection

ANCHOR_FEATURES = [
    "draft_capital",
    "best1_yprr_graduated",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]
V11_FEATURES = ANCHOR_FEATURES + [
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
]


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
    """Add features at season level."""
    q = qual.copy()
    q["catch_pct_adot_adj"] = q["caught_percent"] - np.polyval(adot_catch_coef, q["avg_depth_of_target"])
    q["cpaa_minus_drops"] = q["catch_pct_adot_adj"] - q["drop_rate"]
    q["catch_minus_drops"] = q["caught_percent"] - q["drop_rate"]
    non_ct = q["targets"] - q["contested_targets"]
    non_cr = q["receptions"] - q["contested_receptions"]
    q["clean_catch_rate"] = np.where(non_ct > 0, non_cr / non_ct * 100, np.nan)
    return q


def aggregate_player(player_key, draft_year, birthdate, qual_seasons):
    """Aggregate features using three selection methods: best1, peak, peak_gated."""
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

    # BEST1: highest grades_offense season (current method)
    best1_row = eligible.loc[grades.idxmax()]

    # QUALITY-GATED seasons: grades_offense >= 80
    gated = eligible[grades >= QUALITY_GATE]
    has_gated = len(gated) > 0

    result = {}

    # Stats to compute for each selection method
    stat_defs = {
        "catch_pct_adot_adj": {"col": "catch_pct_adot_adj", "center": 0, "graduated": True},
        "cpaa_minus_drops": {"col": "cpaa_minus_drops", "center": 0, "graduated": True},
        "catch_minus_drops": {"col": "catch_minus_drops", "center": 55, "graduated": True},
        "clean_catch_rate": {"col": "clean_catch_rate", "center": 68, "graduated": True},
        "yprr": {"col": "yprr_raw", "center": 0, "graduated": True},
    }

    # Compute raw YPRR from yards/routes at season level
    for idx, row in eligible.iterrows():
        yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
        rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
        eligible.at[idx, "yprr_raw"] = yards / rts if rts > 0 else np.nan

    if has_gated:
        for idx, row in gated.iterrows():
            yards = pd.to_numeric(row.get("yards", 0), errors="coerce") or 0
            rts = pd.to_numeric(row.get("routes", 0), errors="coerce") or 0
            gated.at[idx, "yprr_raw"] = yards / rts if rts > 0 else np.nan

    for stat_name, sdef in stat_defs.items():
        col = sdef["col"]

        # --- best1: value from highest-grade season ---
        if col == "yprr_raw":
            yards = pd.to_numeric(best1_row.get("yards", 0), errors="coerce") or 0
            rts = pd.to_numeric(best1_row.get("routes", 0), errors="coerce") or 0
            best1_val = yards / rts if rts > 0 else np.nan
        elif col in best1_row.index:
            best1_val = best1_row[col]
        else:
            best1_val = np.nan

        if pd.notna(best1_val):
            result[f"best1_{stat_name}"] = round(float(best1_val), 4)

        # Helper to age-adjust a value given a season year
        def _age_adjust(val, yr, center):
            if birthdate is None or pd.isna(birthdate) or yr is None or pd.isna(yr):
                return val
            age = get_age_on_sept1(birthdate, yr)
            if age is None:
                return val
            for (lo, hi), mult in GRADUATED_ADJ.items():
                if lo <= age < hi:
                    return (val - center) * mult + center
            return val

        # --- peak_gated: max of stat among quality-gated seasons, fallback to best1 ---
        # For graduated stats, select based on age-adjusted value
        if has_gated and col in gated.columns:
            gated_vals = gated[col].dropna()
            if len(gated_vals) > 0:
                if sdef["graduated"] and birthdate is not None and pd.notna(birthdate):
                    # Select season with max age-adjusted value
                    best_adj = np.nan
                    peak_idx = None
                    for idx in gated_vals.index:
                        yr = gated.loc[idx].get("grade_year") if hasattr(gated.loc[idx], "get") else None
                        adj = _age_adjust(float(gated_vals.loc[idx]), yr, sdef["center"])
                        if np.isnan(best_adj) or adj > best_adj:
                            best_adj = adj
                            peak_idx = idx
                    peak_val = float(gated_vals.loc[peak_idx])
                    peak_row = gated.loc[peak_idx]
                else:
                    peak_idx = gated_vals.idxmax()
                    peak_val = float(gated_vals.max())
                    peak_row = gated.loc[peak_idx]
            else:
                peak_val = best1_val
                peak_row = best1_row
        else:
            peak_val = best1_val
            peak_row = best1_row

        if pd.notna(peak_val):
            result[f"pg_{stat_name}"] = round(float(peak_val), 4)

        # --- pure peak: max of stat among ALL eligible seasons (no quality gate) ---
        if col in eligible.columns:
            all_vals = eligible[col].dropna()
            if len(all_vals) > 0:
                pure_peak_idx = all_vals.idxmax()
                pure_peak_val = float(all_vals.max())
                pure_peak_row = eligible.loc[pure_peak_idx]
            else:
                pure_peak_val = best1_val
                pure_peak_row = best1_row
        else:
            pure_peak_val = best1_val
            pure_peak_row = best1_row

        if pd.notna(pure_peak_val):
            result[f"peak_{stat_name}"] = round(float(pure_peak_val), 4)

        # --- Graduated variants (age-adjusted) ---
        if sdef["graduated"] and birthdate is not None and pd.notna(birthdate):
            center = sdef["center"]

            for prefix, val, row_source in [
                ("best1", best1_val, best1_row),
                ("pg", peak_val, peak_row),
                ("peak", pure_peak_val, pure_peak_row),
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
                        result[f"{prefix}_{stat_name}_graduated"] = round(adjusted, 4)
                        break

    # Track which season was selected for each method (for diagnostics)
    result["best1_grade"] = round(float(grades.max()), 1) if grades.notna().any() else np.nan
    if has_gated:
        # For the main feature (catch_pct_adot_adj), which season did peak_gated pick?
        col = "catch_pct_adot_adj"
        if col in gated.columns:
            gated_vals = gated[col].dropna()
            if len(gated_vals) > 0:
                pg_idx = gated_vals.idxmax()
                result["pg_cpaa_grade"] = round(float(gated.loc[pg_idx, "grades_offense"]), 1)
                result["pg_cpaa_same_as_best1"] = 1 if pg_idx == grades.idxmax() else 0
        result["n_gated_seasons"] = len(gated)
    else:
        result["pg_cpaa_same_as_best1"] = 1
        result["n_gated_seasons"] = 0

    return result


# ============================================================
# Analysis functions
# ============================================================

def _loo_auc(d, features, years):
    all_p, all_t = [], []
    for yr in years:
        train = d[d["draft_year"] != yr]
        test = d[d["draft_year"] == yr]
        y_tr = (train["tier_num"].values >= 3).astype(int)
        y_te = (test["tier_num"].values >= 3).astype(int)
        if y_tr.sum() < 2 or y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(train[features].values)
        X_te = sc.transform(test[features].values)
        lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        lr.fit(X_tr, y_tr)
        all_p.extend(lr.predict_proba(X_te)[:, 1])
        all_t.extend(y_te)
    return roc_auc_score(np.array(all_t), np.array(all_p)) if all_t else np.nan


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


def run_7part(candidates, df, base_features, label=""):
    valid_candidates = [c for c in candidates if c in df.columns and df[c].notna().sum() > 50]
    if not valid_candidates:
        print(f"  No valid candidates for {label}")
        return pd.DataFrame()

    all_cols = base_features + valid_candidates + ["tier_num", "draft_year"]
    d = df.dropna(subset=[c for c in all_cols if c in df.columns]).copy()
    y = d["tier_num"].values
    hit = (y >= 3).astype(int)
    years = sorted(d["draft_year"].unique())

    print(f"\n{'=' * 85}")
    print(f"  {label} | n={len(d)} | base={base_features}")
    print(f"{'=' * 85}")

    scaler = StandardScaler()
    X_base = scaler.fit_transform(d[base_features].values)
    ridge = Ridge(alpha=1.0).fit(X_base, y)
    residuals = y - ridge.predict(X_base)
    base_auc = _loo_auc(d, base_features, years)

    results = []
    for c in valid_candidates:
        row = {"feature": c}
        sp, _ = spearmanr(d[c].values, y)
        auc = roc_auc_score(hit, d[c].values) if d[c].nunique() > 1 else 0.5
        row["spearman"] = round(sp, 3)
        row["auc"] = round(auc, 3)

        mid = years[len(years) // 2]
        early = d[d["draft_year"] <= mid]
        late = d[d["draft_year"] > mid]
        sp_e, _ = spearmanr(early[c].values, early["tier_num"].values) if len(early) > 10 else (np.nan, np.nan)
        sp_l, _ = spearmanr(late[c].values, late["tier_num"].values) if len(late) > 10 else (np.nan, np.nan)
        row["era_drift"] = round(abs(sp_e - sp_l), 3) if pd.notna(sp_e) and pd.notna(sp_l) else np.nan

        sp_res, _ = spearmanr(d[c].values, residuals)
        row["residual"] = round(sp_res, 3)

        max_corr = 0
        for bf in base_features:
            corr = abs(spearmanr(d[bf].values, d[c].values)[0])
            max_corr = max(max_corr, corr)
        row["max_collinearity"] = round(max_corr, 3)

        n_boot = 1000
        rng = np.random.RandomState(42)
        boot_rhos = []
        for _ in range(n_boot):
            idx = rng.choice(len(d), size=len(d), replace=True)
            X_b = scaler.fit_transform(d[base_features].values[idx])
            ridge.fit(X_b, y[idx])
            res_b = y[idx] - ridge.predict(X_b)
            r, _ = spearmanr(d[c].values[idx], res_b)
            boot_rhos.append(r)
        boot_arr = np.array(boot_rhos)
        row["boot_pct_pos"] = round((boot_arr > 0).mean(), 3)
        row["boot_mean"] = round(boot_arr.mean(), 3)

        feat_auc = _loo_auc(d, base_features + [c], years)
        row["loo_auc"] = round(feat_auc, 3)
        row["loo_delta"] = round(feat_auc - base_auc, 3)

        enet_count = 0
        for C in [0.01, 0.1, 1.0]:
            X_all = StandardScaler().fit_transform(d[base_features + [c]].values)
            lr = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=C, max_iter=10000, random_state=42,
            )
            lr.fit(X_all, hit)
            if abs(lr.coef_[0, -1]) > 1e-6:
                enet_count += 1
        row["enet_survive"] = enet_count
        results.append(row)

    results_df = pd.DataFrame(results)
    print(f"\n  {'Feature':<48s} {'Sp':>6s} {'AUC':>6s} {'Drift':>6s} {'Resid':>6s} "
          f"{'Boot%':>6s} {'LOO':>6s} {'Delta':>7s} {'Enet':>5s} {'Collin':>6s}")
    print(f"  {'-'*48} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*6}")
    for _, r in results_df.sort_values("loo_delta", ascending=False).iterrows():
        print(f"  {r['feature']:<48s} {r['spearman']:>+6.3f} {r['auc']:>6.3f} "
              f"{r['era_drift']:>6.3f} {r['residual']:>+6.3f} {r['boot_pct_pos']:>6.1%} "
              f"{r['loo_auc']:>6.3f} {r['loo_delta']:>+7.3f} {r['enet_survive']:>3d}/3 "
              f"{r['max_collinearity']:>6.3f}")
    return results_df


def test_combos(new_feats, df):
    print(f"\n{'=' * 85}")
    print(f"  COMBINATION TESTS (ordinal LogLoss + Brier + AUC)")
    print(f"{'=' * 85}")

    core_cols = ANCHOR_FEATURES + ["tier_num", "draft_year"]
    d = df.dropna(subset=core_cols).copy()
    years = sorted(d["draft_year"].unique())
    print(f"  n={len(d)} | years={[int(y) for y in years]}")

    combos = {}
    combos["v11 (QBR + catch%_adot_adj)"] = V11_FEATURES[:]
    combos["v11 minus QBR"] = [f for f in V11_FEATURES if f != "career_targeted_qb_rating"]
    combos["4 anchors only"] = ANCHOR_FEATURES[:]

    # Replace QBR only (keep best2_catch_pct_adot_adj)
    for cand in new_feats:
        if cand in d.columns:
            feats = [f for f in V11_FEATURES if f != "career_targeted_qb_rating"] + [cand]
            combos[f"QBR => {cand}"] = feats

    # Replace catch%_adot only (keep QBR)
    for cand in new_feats:
        if cand in d.columns:
            feats = [f for f in V11_FEATURES if f != "best2_catch_pct_adot_adj"] + [cand]
            combos[f"catch%_adot => {cand}"] = feats

    # Replace BOTH with single new feature
    for cand in new_feats:
        if cand in d.columns:
            combos[f"both => {cand}"] = ANCHOR_FEATURES + [cand]

    print(f"\n  {'Combination':<65s} {'LogLoss':>8s} {'Brier':>8s} {'>=Elite':>8s} "
          f"{'>=Stud':>8s} {'>=Start':>8s} {'#F':>4s}")
    print(f"  {'-'*65} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

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
        print(f"  {name:<65s} {ll_str} {br_str} {ae_str} {as_str} {at_str} {row['n_feats']:>4d}")

    return pd.DataFrame(results)


def generate_report(combo_results, results_dc, results_full, df, diagnostics):
    """Generate markdown report and visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d",
        "grid.color": "#21262d",
        "font.family": "monospace",
        "font.size": 10,
    })

    charts_dir = os.path.join(DATA_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # --- Figure: 6-panel report ---
    fig, axes = plt.subplots(2, 3, figsize=(28, 16))

    # Panel A: Selection method comparison (full model 7-part)
    ax = axes[0, 0]
    if len(results_full) > 0:
        # Group by selection method
        rf = results_full.copy()
        def get_method(f):
            if f.startswith("pg_"):
                return "peak-gated"
            elif f.startswith("peak_"):
                return "pure peak"
            elif f.startswith("best1_"):
                return "best1 (current)"
            return "other"
        rf["method"] = rf["feature"].apply(get_method)
        rf["stat"] = rf["feature"].apply(lambda f: f.split("_", 1)[-1] if "_" in f else f)

        # Plot LOO delta by method, colored by stat family
        method_order = ["best1 (current)", "peak-gated", "pure peak"]
        colors_map = {"best1 (current)": "#58a6ff", "peak-gated": "#3fb950", "pure peak": "#f78166"}
        x_pos = 0
        tick_labels, tick_positions = [], []
        for _, row in rf.sort_values("loo_delta", ascending=True).iterrows():
            color = colors_map.get(row["method"], "#8b949e")
            ax.barh(x_pos, row["loo_delta"], color=color, alpha=0.7)
            short = row["feature"].replace("_graduated", "_gr")
            tick_labels.append(short)
            tick_positions.append(x_pos)
            x_pos += 1
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.axvline(0, color="#f0883e", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("LOO-AUC Delta (vs 4-anchor base)")
        ax.set_title("A. Selection Method Comparison\n(Full Model Base)", fontweight="bold", fontsize=10)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#58a6ff", alpha=0.7, label="best1 (best grade)"),
            Patch(facecolor="#3fb950", alpha=0.7, label="peak-gated (grade≥80)"),
            Patch(facecolor="#f78166", alpha=0.7, label="pure peak (any grade)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Panel B: Combo results - LogLoss
    ax = axes[0, 1]
    if len(combo_results) > 0:
        # Filter to key combos
        key = combo_results[
            combo_results["combo"].str.contains("v11|4 anchors|QBR =>.*graduated|both =>.*graduated|both =>.*clean|QBR =>.*clean")
        ].copy()
        if len(key) == 0:
            key = combo_results.head(15)
        key = key.sort_values("log_loss", ascending=False)
        v11_ll = combo_results[combo_results["combo"].str.contains("v11.*QBR")]["log_loss"].values
        v11_ll = v11_ll[0] if len(v11_ll) > 0 else np.nan

        y_pos = range(len(key))
        bar_colors = ["#3fb950" if v <= v11_ll else "#f85149" for v in key["log_loss"]]
        bars = ax.barh(list(y_pos), key["log_loss"].values, color=bar_colors, alpha=0.7)
        if pd.notna(v11_ll):
            ax.axvline(v11_ll, color="#f0883e", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_yticks(list(y_pos))
        labels = key["combo"].str.replace("catch_pct_adot_adj", "cpaa", regex=False)
        ax.set_yticklabels(labels.values, fontsize=7)
        ax.set_xlabel("Ordinal LogLoss (lower = better)")
        ax.set_title("B. Combo LogLoss\n(orange line = v11)", fontweight="bold", fontsize=10)
        for bar, val in zip(bars, key["log_loss"].values):
            if pd.notna(val) and pd.notna(v11_ll):
                delta = val - v11_ll
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f} ({delta:+.3f})", va="center", fontsize=7)

    # Panel C: Combo results - Brier
    ax = axes[0, 2]
    if len(combo_results) > 0:
        key2 = key.sort_values("brier", ascending=False)
        v11_br = combo_results[combo_results["combo"].str.contains("v11.*QBR")]["brier"].values
        v11_br = v11_br[0] if len(v11_br) > 0 else np.nan

        y_pos = range(len(key2))
        bar_colors = ["#3fb950" if v <= v11_br else "#f85149" for v in key2["brier"]]
        bars = ax.barh(list(y_pos), key2["brier"].values, color=bar_colors, alpha=0.7)
        if pd.notna(v11_br):
            ax.axvline(v11_br, color="#f0883e", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_yticks(list(y_pos))
        labels2 = key2["combo"].str.replace("catch_pct_adot_adj", "cpaa", regex=False)
        ax.set_yticklabels(labels2.values, fontsize=7)
        ax.set_xlabel("Ordinal Brier (lower = better)")
        ax.set_title("C. Combo Brier Score\n(orange line = v11)", fontweight="bold", fontsize=10)
        for bar, val in zip(bars, key2["brier"].values):
            if pd.notna(val) and pd.notna(v11_br):
                delta = val - v11_br
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f} ({delta:+.3f})", va="center", fontsize=7)

    # Panel D: Combo results - Elite AUC
    ax = axes[1, 0]
    if len(combo_results) > 0:
        key3 = key.sort_values("elite_auc", ascending=True)
        v11_auc = combo_results[combo_results["combo"].str.contains("v11.*QBR")]["elite_auc"].values
        v11_auc = v11_auc[0] if len(v11_auc) > 0 else np.nan

        y_pos = range(len(key3))
        bar_colors = ["#3fb950" if v >= v11_auc else "#f85149" for v in key3["elite_auc"]]
        bars = ax.barh(list(y_pos), key3["elite_auc"].values, color=bar_colors, alpha=0.7)
        if pd.notna(v11_auc):
            ax.axvline(v11_auc, color="#f0883e", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_yticks(list(y_pos))
        labels3 = key3["combo"].str.replace("catch_pct_adot_adj", "cpaa", regex=False)
        ax.set_yticklabels(labels3.values, fontsize=7)
        ax.set_xlabel(">=Elite AUC (higher = better)")
        ax.set_title("D. Combo Elite AUC\n(orange line = v11)", fontweight="bold", fontsize=10)
        for bar, val in zip(bars, key3["elite_auc"].values):
            if pd.notna(val) and pd.notna(v11_auc):
                delta = val - v11_auc
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f} ({delta:+.3f})", va="center", fontsize=7)

    # Panel E: Diagnostics - how often does peak-gated differ from best1?
    ax = axes[1, 1]
    if diagnostics is not None and len(diagnostics) > 0:
        same = diagnostics["pg_cpaa_same_as_best1"].sum()
        diff = len(diagnostics) - same
        n_gated = diagnostics["n_gated_seasons"]

        ax.bar(["Same as\nbest1", "Different\nseason"], [same, diff],
               color=["#58a6ff", "#3fb950"], alpha=0.7)
        ax.set_ylabel("Number of players")
        ax.set_title(f"E. Peak-Gated vs Best1 Selection\n(n={len(diagnostics)}, gate=grade≥{QUALITY_GATE})",
                     fontweight="bold", fontsize=10)
        ax.text(0, same + 1, f"{same} ({same/len(diagnostics)*100:.0f}%)", ha="center", fontsize=11)
        ax.text(1, diff + 1, f"{diff} ({diff/len(diagnostics)*100:.0f}%)", ha="center", fontsize=11)

        # Inset: distribution of gated seasons count
        ax_inset = ax.inset_axes([0.55, 0.45, 0.4, 0.45])
        ax_inset.hist(n_gated, bins=range(0, int(n_gated.max()) + 2), color="#d29922", alpha=0.7,
                      edgecolor="#30363d")
        ax_inset.set_xlabel("# seasons ≥ 80 grade", fontsize=7)
        ax_inset.set_ylabel("Players", fontsize=7)
        ax_inset.set_title("Gated season count", fontsize=8)
        ax_inset.tick_params(labelsize=7)

    # Panel F: Head-to-head scatter (best1 vs peak-gated, cpaa_graduated)
    ax = axes[1, 2]
    b1_col = "best1_catch_pct_adot_adj_graduated"
    pg_col = "pg_catch_pct_adot_adj_graduated"
    if b1_col in df.columns and pg_col in df.columns:
        valid = df[[b1_col, pg_col, "computed_tier"]].dropna()
        tier_colors = {
            "Bust": "#8b949e", "Flex": "#58a6ff", "Starter": "#3fb950",
            "Elite": "#d29922", "Stud": "#f78166", "League-Winner": "#f85149",
        }
        for tier in ["Bust", "Flex", "Starter", "Elite", "Stud", "League-Winner"]:
            subset = valid[valid["computed_tier"] == tier]
            ax.scatter(subset[b1_col], subset[pg_col],
                       c=tier_colors.get(tier, "#8b949e"), label=tier,
                       alpha=0.6, s=30, edgecolors="none")
        # Identity line
        lims = [min(valid[b1_col].min(), valid[pg_col].min()) - 2,
                max(valid[b1_col].max(), valid[pg_col].max()) + 2]
        ax.plot(lims, lims, color="#f0883e", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"best1 (by grade) cpaa_graduated")
        ax.set_ylabel(f"peak-gated (grade≥{QUALITY_GATE}) cpaa_graduated")
        ax.set_title("F. best1 vs peak-gated\n(catch% aDOT adj graduated)", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Peak-Gated Season Selection: Analysis", fontsize=14, fontweight="bold", y=0.98)
    path = os.path.join(charts_dir, "peak_gated_selection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return path


def write_report(combo_results, results_dc, results_full, diagnostics, chart_path):
    """Write markdown report."""
    reports_dir = os.path.join(DATA_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Extract key numbers
    v11 = combo_results[combo_results["combo"].str.contains("v11.*QBR")]
    v11_ll = v11["log_loss"].values[0] if len(v11) > 0 else np.nan
    v11_br = v11["brier"].values[0] if len(v11) > 0 else np.nan
    v11_ae = v11["elite_auc"].values[0] if len(v11) > 0 else np.nan
    v11_as = v11["stud_auc"].values[0] if len(v11) > 0 else np.nan
    v11_at = v11["starter_auc"].values[0] if len(v11) > 0 else np.nan

    same = int(diagnostics["pg_cpaa_same_as_best1"].sum()) if diagnostics is not None else 0
    total = len(diagnostics) if diagnostics is not None else 0
    diff = total - same

    # Get best combos
    best_ll_row = combo_results.loc[combo_results["log_loss"].idxmin()] if combo_results["log_loss"].notna().any() else None
    best_br_row = combo_results.loc[combo_results["brier"].idxmin()] if combo_results["brier"].notna().any() else None
    best_ae_row = combo_results.loc[combo_results["elite_auc"].idxmax()] if combo_results["elite_auc"].notna().any() else None

    lines = [
        "# Peak-Gated Season Selection: Investigation Report",
        "",
        "**Date**: 2026-05-13",
        f"**Script**: `modeling/research/test_peak_gated_selection.py`",
        f"**Quality Gate**: grades_offense >= {QUALITY_GATE}",
        "",
        "---",
        "",
        "## Concept",
        "",
        "The current `best1` selection picks the season with the highest PFF offensive grade.",
        "This investigation tests a hybrid: **peak-gated** selection picks the season where",
        "a specific stat peaks, but only from seasons with grades_offense >= 80. If no season",
        "meets the quality gate, it falls back to the current best-by-grade selection.",
        "",
        "The hypothesis: a receiver's best *catching* season may not be their best *overall grade*",
        "season. A sophomore with a 82-grade season but elite catch metrics could be more",
        "informative than their 88-grade junior season where they ran better routes but caught worse.",
        "",
        "---",
        "",
        "## Selection Divergence",
        "",
        f"For `catch_pct_adot_adj`, peak-gated selected a **different season** than best1 for",
        f"**{diff} of {total} players ({diff/total*100:.0f}%)**. For the remaining {same} ({same/total*100:.0f}%),",
        "the best-grade season was also the peak catch% season.",
        "",
        "---",
        "",
        "## 7-Part Analysis Results",
        "",
        "### Full Model Base (4 anchors)",
        "",
    ]

    if len(results_full) > 0:
        lines.append("| Rank | Feature | Spearman | LOO Delta | Residual | Boot %+ | Collinearity | Era Drift |")
        lines.append("|------|---------|----------|-----------|----------|---------|-------------|-----------|")
        for i, (_, r) in enumerate(results_full.sort_values("loo_delta", ascending=False).iterrows(), 1):
            lines.append(
                f"| {i} | {r['feature']} | {r['spearman']:+.3f} | {r['loo_delta']:+.3f} | "
                f"{r['residual']:+.3f} | {r['boot_pct_pos']:.1%} | {r['max_collinearity']:.3f} | {r['era_drift']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Combination Results",
        "",
        "| Configuration | LogLoss | Brier | >=Elite | >=Stud | >=Starter | # Feats |",
        "|---------------|---------|-------|---------|--------|-----------|---------|",
    ])

    for _, r in combo_results.iterrows():
        ll = f"{r['log_loss']:.3f}" if pd.notna(r['log_loss']) else "n/a"
        br = f"{r['brier']:.3f}" if pd.notna(r['brier']) else "n/a"
        ae = f"{r['elite_auc']:.3f}" if pd.notna(r['elite_auc']) else "n/a"
        a_s = f"{r['stud_auc']:.3f}" if pd.notna(r['stud_auc']) else "n/a"
        at = f"{r['starter_auc']:.3f}" if pd.notna(r['starter_auc']) else "n/a"
        lines.append(f"| {r['combo']} | {ll} | {br} | {ae} | {a_s} | {at} | {r['n_feats']} |")

    lines.extend([
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ])

    if best_ll_row is not None:
        lines.append(f"**Best LogLoss**: {best_ll_row['combo']} ({best_ll_row['log_loss']:.3f} vs v11 {v11_ll:.3f}, "
                     f"delta {best_ll_row['log_loss'] - v11_ll:+.3f})")
    if best_br_row is not None:
        lines.append(f"**Best Brier**: {best_br_row['combo']} ({best_br_row['brier']:.3f} vs v11 {v11_br:.3f}, "
                     f"delta {best_br_row['brier'] - v11_br:+.3f})")
    if best_ae_row is not None:
        lines.append(f"**Best Elite AUC**: {best_ae_row['combo']} ({best_ae_row['elite_auc']:.3f} vs v11 {v11_ae:.3f}, "
                     f"delta {best_ae_row['elite_auc'] - v11_ae:+.3f})")

    lines.extend([
        "",
        "---",
        "",
        "## Visualizations",
        "",
        f"![Peak-Gated Selection Analysis]({os.path.basename(chart_path)})",
        "",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `wr_data/charts/peak_gated_selection.png` | 6-panel analysis: method comparison, combo results, diagnostics |",
    ])

    report_path = os.path.join(reports_dir, "peak_gated_selection_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {report_path}")
    return report_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 85)
    print("  PEAK-GATED SEASON SELECTION TEST")
    print(f"  Quality gate: grades_offense >= {QUALITY_GATE}")
    print("=" * 85)

    # Load data
    print("\n[1/6] Loading data...")
    ag = load_grades()
    qual = ag[ag["routes"] >= 200].copy()
    print(f"  Qualified seasons: {len(qual)}")

    # aDOT regression
    cp, adot = ag["caught_percent"], ag["avg_depth_of_target"]
    m = cp.notna() & adot.notna()
    adot_catch_coef = np.polyfit(adot[m].values, cp[m].values, 1)

    # Engineer season features
    print("\n[2/6] Engineering season features...")
    qual = engineer_season_features(qual, adot_catch_coef)

    # How many seasons pass the quality gate?
    grades = pd.to_numeric(qual["grades_offense"], errors="coerce")
    n_above = (grades >= QUALITY_GATE).sum()
    print(f"  Seasons with grades_offense >= {QUALITY_GATE}: {n_above}/{len(qual)} ({n_above/len(qual)*100:.1f}%)")

    # Load master
    print("\n[3/6] Aggregating per player...")
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_num"] = df["computed_tier"].map(TIER_ORDER)
    df["_join_key"] = df["name"].apply(normalize_name)

    ages = pd.read_csv(os.path.join(DATA_DIR, "draft_ages.csv"))
    ages["birthdate"] = pd.to_datetime(ages["birthdate"])
    birth_lookup = dict(zip(zip(ages["name"], ages["draft_year"]), ages["birthdate"]))

    results = []
    for _, row in df.iterrows():
        birthdate = birth_lookup.get((row["name"], row["draft_year"]))
        res = aggregate_player(normalize_name(row["name"]), row["draft_year"], birthdate, qual)
        results.append(res)

    eng_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), eng_df.reset_index(drop=True)], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Diagnostics
    diag_cols = ["pg_cpaa_same_as_best1", "n_gated_seasons", "best1_grade", "pg_cpaa_grade"]
    diagnostics = df[["name"] + [c for c in diag_cols if c in df.columns]].dropna(subset=["pg_cpaa_same_as_best1"])

    same = int(diagnostics["pg_cpaa_same_as_best1"].sum())
    diff = len(diagnostics) - same
    print(f"\n  Peak-gated selection divergence:")
    print(f"    Same as best1: {same}/{len(diagnostics)} ({same/len(diagnostics)*100:.0f}%)")
    print(f"    Different season: {diff}/{len(diagnostics)} ({diff/len(diagnostics)*100:.0f}%)")
    print(f"    Mean # gated seasons: {diagnostics['n_gated_seasons'].mean():.1f}")

    # Show available new features
    new_cols = [c for c in eng_df.columns
                if c not in diag_cols and c not in ["best1_grade"]
                and eng_df[c].notna().sum() > 50
                and not c.startswith("pg_cpaa_")]
    print(f"\n  Features for testing: {len(new_cols)}")
    for c in sorted(new_cols):
        n = df[c].notna().sum()
        print(f"    {c}: n={n}")

    # Add incumbents + key existing features
    candidates = new_cols + [
        "career_targeted_qb_rating", "best2_catch_pct_adot_adj",
    ]
    candidates = list(dict.fromkeys(c for c in candidates if c in df.columns))

    # Run 7-part analysis
    print("\n[4/6] Running 7-part analysis...")
    results_dc = run_7part(candidates, df, base_features=["draft_capital"],
                           label="DC-ONLY BASE")
    results_full = run_7part(candidates, df, base_features=ANCHOR_FEATURES,
                             label="FULL MODEL BASE (4 anchors)")

    # Combo tests
    print("\n[5/6] Running combination tests...")
    combo_feats = [c for c in new_cols if "graduated" in c or "clean_catch_rate" in c
                   or c in ["best1_catch_pct_adot_adj", "pg_catch_pct_adot_adj",
                            "best1_cpaa_minus_drops", "pg_cpaa_minus_drops"]]
    combo_results = test_combos(combo_feats, df)

    # Save combo results
    out_dir = os.path.join(DATA_DIR, "outputs")
    combo_results.to_csv(os.path.join(out_dir, "peak_gated_combos.csv"), index=False)
    if len(results_dc) > 0:
        results_dc.to_csv(os.path.join(out_dir, "peak_gated_dc.csv"), index=False)
    if len(results_full) > 0:
        results_full.to_csv(os.path.join(out_dir, "peak_gated_full.csv"), index=False)

    # Generate report + viz
    print("\n[6/6] Generating report and visualizations...")
    chart_path = generate_report(combo_results, results_dc, results_full, df, diagnostics)
    report_path = write_report(combo_results, results_dc, results_full, diagnostics, chart_path)

    # Summary
    print("\n" + "=" * 85)
    print("  SUMMARY")
    print("=" * 85)

    if len(combo_results) > 0:
        v11 = combo_results[combo_results["combo"].str.contains("v11.*QBR")]
        if len(v11) > 0:
            v = v11.iloc[0]
            print(f"\n  v11 baseline: LL={v['log_loss']:.3f} Br={v['brier']:.3f} "
                  f"Elite={v['elite_auc']:.3f} Stud={v['stud_auc']:.3f} Start={v['starter_auc']:.3f}")

        for metric, ascending, label in [
            ("log_loss", True, "Best LogLoss"),
            ("brier", True, "Best Brier"),
            ("elite_auc", False, "Best Elite AUC"),
        ]:
            best = combo_results.nsmallest(3, metric) if ascending else combo_results.nlargest(3, metric)
            print(f"\n  {label}:")
            for _, r in best.iterrows():
                val = r[metric]
                print(f"    {r['combo']:<60s} {metric}={val:.3f}")

    print(f"\n  Report: {report_path}")
    print(f"  Chart:  {chart_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
