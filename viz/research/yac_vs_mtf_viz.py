#!/usr/bin/env python3
"""
YAC vs MTF/Reception visualization.

Shows that YAC variants have negative residual signal once better features
are in the model, while missed tackles per reception retains positive signal.
The useful part of post-catch ability is captured by MTF/rec — YAC is redundant.

Outputs: wr_data/charts/yac_vs_mtf.png
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
OUT_DIR = os.path.join(DATA_DIR, "charts")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
THRESHOLDS = [1, 2, 3, 4, 5]
N_TIERS = 6

# Locked features (the base model without any post-catch feature)
LOCKED_FEATURES = [
    "draft_capital",
    "best1_yprr_graduated",
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
]

HOLDOUT_YEARS = [2022, 2023, 2024]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"] + LOCKED_FEATURES + ["best2_avoided_tackles_per_rec"]).copy()
    df["tier_ordinal"] = df["tier_ordinal"].astype(int)

    # Aggregate YAC features from raw grades
    grades_dfs = []
    for year in range(2016, 2026):
        path = os.path.join(DATA_DIR, "grades", f"{year}_receiving_grades.csv")
        if os.path.exists(path):
            gdf = pd.read_csv(path)
            gdf["season"] = year
            grades_dfs.append(gdf)
    grades = pd.concat(grades_dfs, ignore_index=True)

    # Compute per-player YAC aggregates
    yac_aggs = []
    for name in df["name"].unique():
        pg = grades[grades["player"].str.lower() == name.lower()]
        if len(pg) == 0:
            pg = grades[grades["player"] == name]
        if len(pg) == 0:
            continue
        pg = pg.sort_values("yards_after_catch_per_reception", ascending=False)
        best2 = pg.head(2)
        yac_aggs.append({
            "name": name,
            "best2_yac_per_rec": best2["yards_after_catch_per_reception"].mean(),
            "career_yac_per_rec": pg["yards_after_catch_per_reception"].mean(),
            "best2_yac_total": best2["yards_after_catch"].mean(),
        })

    yac_df = pd.DataFrame(yac_aggs)
    df = df.merge(yac_df, on="name", how="inner")

    train = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
    holdout = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()
    return df, train, holdout


def xgb_predict_proba(X_train, y_train, X_test):
    cum_probs = np.zeros((X_test.shape[0], len(THRESHOLDS)))
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
        cum_probs[:, t_idx] = calibrated.predict_proba(X_test)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((X_test.shape[0], N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs /= tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def permutation_importance(train, holdout, features, n_repeats=20):
    X_train = train[features].values
    y_train = train["tier_ordinal"].values
    X_holdout = holdout[features].values
    y_holdout = holdout["tier_ordinal"].values

    baseline_probs = xgb_predict_proba(X_train, y_train, X_holdout)
    baseline_ll = log_loss(y_holdout, baseline_probs, labels=list(range(N_TIERS)))

    importances = {}
    for feat_idx, feat in enumerate(features):
        deltas = []
        for r in range(n_repeats):
            X_perm = X_holdout.copy()
            rng = np.random.RandomState(r)
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            perm_probs = xgb_predict_proba(X_train, y_train, X_perm)
            perm_ll = log_loss(y_holdout, perm_probs, labels=list(range(N_TIERS)))
            deltas.append(perm_ll - baseline_ll)
        importances[feat] = np.mean(deltas)
        print(f"  {feat}: {importances[feat]:+.4f}")

    return importances, baseline_ll


def residual_analysis(train, holdout, base_features, candidate_feature, n_bootstrap=500):
    """Compute residual Spearman of candidate after controlling for base features."""
    X_train = train[base_features].values
    y_train = train["tier_ordinal"].values
    X_holdout = holdout[base_features].values
    y_holdout = holdout["tier_ordinal"].values

    # Get base model predictions (expected tier)
    base_probs = xgb_predict_proba(X_train, y_train, X_holdout)
    base_expected = np.sum(base_probs * np.arange(N_TIERS), axis=1)

    # Residual = actual tier - predicted tier
    residuals = y_holdout - base_expected

    # Candidate feature values on holdout
    cand_vals = holdout[candidate_feature].values

    # Spearman of candidate vs residuals
    rho, _ = spearmanr(cand_vals, residuals)

    # Bootstrap
    boot_rhos = []
    for b in range(n_bootstrap):
        rng = np.random.RandomState(b)
        idx = rng.choice(len(residuals), size=len(residuals), replace=True)
        r, _ = spearmanr(cand_vals[idx], residuals[idx])
        boot_rhos.append(r)

    pct_positive = np.mean(np.array(boot_rhos) > 0)
    return rho, np.array(boot_rhos), pct_positive


def make_plot(candidates, residuals_dict, boot_dict, pct_pos_dict, univariate_dict,
              title_suffix="", filename="yac_vs_mtf.png"):
    """
    3-panel viz:
      1. Univariate Spearman (all post-catch features)
      2. Residual signal after locked features (with bootstrap CI)
      3. Bootstrap distributions for MTF/rec vs best YAC variant
    """
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)
    fig.suptitle(f"Post-Catch Features: YAC vs Missed Tackles / Reception\n{title_suffix}",
                 fontsize=15, fontweight="bold", y=1.04)

    labels = {
        "best2_avoided_tackles_per_rec": "MTF / Rec (best2)",
        "best2_yac_per_rec": "YAC / Rec (best2)",
        "career_yac_per_rec": "YAC / Rec (career)",
        "best2_yac_total": "YAC Total (best2)",
        "best2_catch_pct_adot_adj": "Catch% aDOT-adj (best2)",
    }

    # Sort by residual
    sorted_cands = sorted(candidates, key=lambda c: residuals_dict[c])
    colors = ["#2ca02c" if residuals_dict[c] > 0 else "#d62728" for c in sorted_cands]
    y_pos = np.arange(len(sorted_cands))

    # --- Panel 1: Univariate Spearman ---
    ax1 = fig.add_subplot(gs[0])
    uni_vals = [univariate_dict[c] for c in sorted_cands]
    uni_colors = ["#1f77b4" for _ in sorted_cands]
    ax1.barh(y_pos, uni_vals, color=uni_colors, alpha=0.8, edgecolor="white", height=0.55)
    for i, v in enumerate(uni_vals):
        ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([labels[c] for c in sorted_cands], fontsize=10)
    ax1.set_xlabel("Spearman rho", fontsize=10)
    ax1.set_title("Univariate Signal\n(vs Tier Ordinal)", fontweight="bold", fontsize=11)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.grid(axis="x", alpha=0.3)

    # --- Panel 2: Residual Spearman ---
    ax2 = fig.add_subplot(gs[1])

    # Bootstrap CI
    ci_low = [np.percentile(boot_dict[c], 5) for c in sorted_cands]
    ci_high = [np.percentile(boot_dict[c], 95) for c in sorted_cands]
    res_vals = [residuals_dict[c] for c in sorted_cands]

    ax2.barh(y_pos, res_vals, color=colors, alpha=0.8, edgecolor="white", height=0.55)
    ax2.errorbar(res_vals, y_pos,
                 xerr=[np.array(res_vals) - np.array(ci_low),
                       np.array(ci_high) - np.array(res_vals)],
                 fmt="none", ecolor="black", capsize=3, linewidth=1.2)

    for i, c in enumerate(sorted_cands):
        pct = pct_pos_dict[c]
        txt = f"{res_vals[i]:+.3f} ({pct:.0%}+)"
        x_offset = max(res_vals[i], ci_high[i]) + 0.015
        ax2.text(x_offset, i, txt, va="center", fontsize=8.5,
                 color="#2ca02c" if res_vals[i] > 0 else "#d62728", fontweight="bold")

    ax2.set_xlim(right=ax2.get_xlim()[1] + 0.12)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([labels[c] for c in sorted_cands], fontsize=10)
    ax2.set_xlabel("Residual Spearman (after locked features)", fontsize=10)
    ax2.set_title("Residual Signal\n(90% Bootstrap CI)", fontweight="bold", fontsize=11)
    ax2.axvline(0, color="black", linewidth=1, linestyle="-")
    ax2.grid(axis="x", alpha=0.3)

    # --- Panel 3: Bootstrap distributions ---
    ax3 = fig.add_subplot(gs[2])

    mtf_key = "best2_avoided_tackles_per_rec"
    # Pick the best YAC variant by residual
    yac_keys = [c for c in candidates if c != mtf_key]
    best_yac = max(yac_keys, key=lambda c: residuals_dict[c])

    ax3.hist(boot_dict[mtf_key], bins=40, alpha=0.7, color="#2ca02c",
             label=f"{labels[mtf_key]}", density=True, edgecolor="white")
    ax3.hist(boot_dict[best_yac], bins=40, alpha=0.55, color="#d62728",
             label=f"{labels[best_yac]} (best YAC)", density=True, edgecolor="white")

    ax3.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax3.axvline(residuals_dict[mtf_key], color="#2ca02c", linewidth=2, linestyle="-", alpha=0.8)
    ax3.axvline(residuals_dict[best_yac], color="#d62728", linewidth=2, linestyle="-", alpha=0.8)

    ax3.set_xlabel("Bootstrap Residual Spearman", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Bootstrap Distributions\nMTF/Rec vs Best YAC Variant", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(axis="both", alpha=0.3)

    # Annotate percentages
    mtf_pct = pct_pos_dict[mtf_key]
    yac_pct = pct_pos_dict[best_yac]
    ax3.text(0.95, 0.92, f"MTF/Rec: {mtf_pct:.0%} positive\n{labels[best_yac]}: {yac_pct:.0%} positive",
             transform=ax3.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved to {out_path}")


def run_analysis(full, train, holdout, candidates, locked, title_suffix, filename):
    """Run residual analysis and generate chart for a given locked feature set."""
    # Drop rows with NaN in any candidate
    tr = train.copy()
    ho = holdout.copy()
    for c in candidates:
        tr = tr.dropna(subset=[c])
        ho = ho.dropna(subset=[c])
    print(f"  After filter: Train {len(tr)}, Holdout {len(ho)}")

    # Univariate Spearman
    print(f"\n  Univariate Spearman:")
    univariate = {}
    for c in candidates:
        sub = full.dropna(subset=[c])
        rho, _ = spearmanr(sub[c], sub["tier_ordinal"])
        univariate[c] = rho
        print(f"    {c}: {rho:+.3f}")

    # Residual analysis
    print(f"\n  Residual analysis (locked: {[f.split('_')[0] for f in locked]})...")
    residuals_dict = {}
    boot_dict = {}
    pct_pos_dict = {}
    for c in candidates:
        print(f"    {c}...")
        rho, boots, pct = residual_analysis(tr, ho, locked, c)
        residuals_dict[c] = rho
        boot_dict[c] = boots
        pct_pos_dict[c] = pct
        print(f"      residual={rho:+.3f}, {pct:.0%} positive")

    print(f"\n  Generating {filename}...")
    make_plot(candidates, residuals_dict, boot_dict, pct_pos_dict, univariate,
              title_suffix=title_suffix, filename=filename)


def main():
    print("Loading data...")
    full, train, holdout = load_data()
    print(f"  Full: {len(full)}, Train: {len(train)}, Holdout: {len(holdout)}")

    yac_mtf_candidates = [
        "best2_avoided_tackles_per_rec",
        "best2_yac_per_rec",
        "career_yac_per_rec",
        "best2_yac_total",
    ]

    all_candidates = yac_mtf_candidates + ["best2_catch_pct_adot_adj"]

    # --- Chart 1: Original (catch% in locked set, YAC/MTF as candidates) ---
    print("\n" + "=" * 60)
    print("CHART 1: Catch% in locked set")
    print("=" * 60)
    run_analysis(full, train, holdout, yac_mtf_candidates, LOCKED_FEATURES,
                 title_suffix="(Catch% in locked set)",
                 filename="yac_vs_mtf.png")

    # --- Chart 2: Catch% removed from locked, added as candidate ---
    print("\n" + "=" * 60)
    print("CHART 2: Catch% as candidate (removed from locked set)")
    print("=" * 60)
    locked_no_catch = [f for f in LOCKED_FEATURES if f != "best2_catch_pct_adot_adj"]
    run_analysis(full, train, holdout, all_candidates, locked_no_catch,
                 title_suffix="(Catch% as candidate)",
                 filename="yac_vs_mtf_with_catch_pct.png")


if __name__ == "__main__":
    main()
