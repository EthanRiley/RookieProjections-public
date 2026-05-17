#!/usr/bin/env python3
"""Grid search catch composite features: CPAA + career aDOT-adjusted catch%.

Tests different weightings and composition methods for combining
pg_catch_pct_adot_adj_graduated (CPAA) with career_catch_pct_adot_adj
as a replacement for standalone CPAA in the 5-feature model.

Evaluates on both holdout (2022-2024) and LOO (2018-2021).

Outputs:
  - wr_data/outputs/catch_composite_grid_search.csv
  - wr_data/charts/catch_composite_grid_search.png
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore

warnings.filterwarnings("ignore")


from modeling.predict_prospects import (
    load_training_data, train_xgb_predict, train_bayesian_predict,
    blend, COLLEGE_FEATURES, TIER_NAMES,
)
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")


def evaluate_holdout(train_df, features):
    train = train_df[train_df["draft_year"] <= 2021].dropna(subset=features).copy()
    holdout = train_df[train_df["draft_year"] >= 2022].dropna(subset=features).copy()

    xgb = train_xgb_predict(train, holdout, features)
    bayes = train_bayesian_predict(train, holdout, features, use_draft_capital=True)
    probs = blend(bayes, xgb)

    y = holdout["tier_ordinal"].values
    ll = log_loss(y, probs, labels=list(range(6)))
    brier = np.mean(np.sum((probs - np.eye(6)[y]) ** 2, axis=1))
    elite_auc = roc_auc_score((y >= 3).astype(int), probs[:, 3:].sum(axis=1))
    return ll, brier, elite_auc


def evaluate_loo(train_df, features):
    loo_df = train_df[train_df["draft_year"] <= 2021].copy()
    years = sorted(loo_df["draft_year"].unique())

    all_probs, all_y = [], []
    for yr in years:
        t = loo_df[loo_df["draft_year"] != yr].dropna(subset=features).copy()
        h = loo_df[loo_df["draft_year"] == yr].dropna(subset=features).copy()

        xgb = train_xgb_predict(t, h, features)
        bayes = train_bayesian_predict(t, h, features, use_draft_capital=True)
        p = blend(bayes, xgb)
        all_probs.append(p)
        all_y.append(h["tier_ordinal"].values)

    all_probs = np.vstack(all_probs)
    all_y = np.concatenate(all_y)

    ll = log_loss(all_y, all_probs, labels=list(range(6)))
    brier = np.mean(np.sum((all_probs - np.eye(6)[all_y]) ** 2, axis=1))
    elite_auc = roc_auc_score((all_y >= 3).astype(int), all_probs[:, 3:].sum(axis=1))
    return ll, brier, elite_auc


def build_composites(train_df):
    """Build all composite columns and return config dict mapping name -> column."""
    z_cpaa = zscore(train_df["pg_catch_pct_adot_adj_graduated"])
    z_career = zscore(train_df["career_catch_pct_adot_adj"])

    configs = {}

    # Z-avg at different weights (CPAA weight / career weight)
    for cpaa_w in [0.25, 0.33, 0.50, 0.67, 0.75]:
        career_w = 1 - cpaa_w
        col = f"comp_{int(cpaa_w * 100)}_{int(career_w * 100)}"
        train_df[col] = cpaa_w * z_cpaa + career_w * z_career
        configs[f"z-avg {int(cpaa_w * 100)}/{int(career_w * 100)}"] = col

    # Geometric mean
    train_df["comp_geomean"] = np.sqrt(
        train_df["pg_catch_pct_adot_adj_graduated"].clip(lower=0.01)
        * train_df["career_catch_pct_adot_adj"].clip(lower=0.01)
    )
    configs["geomean"] = "comp_geomean"

    # Rank average
    train_df["comp_rank_avg"] = (
        train_df["pg_catch_pct_adot_adj_graduated"].rank()
        + train_df["career_catch_pct_adot_adj"].rank()
    ) / 2
    configs["rank-avg"] = "comp_rank_avg"

    # Z-max
    train_df["comp_z_max"] = np.maximum(z_cpaa, z_career)
    configs["z-max"] = "comp_z_max"

    return configs


def make_chart(results_df):
    """Generate the 2x3 grid search comparison chart."""
    import matplotlib.pyplot as plt

    names = results_df["config"].tolist()
    x = np.arange(len(names))

    # Add line breaks for long names
    display_names = []
    for n in names:
        if n == "5F baseline":
            display_names.append("5F baseline\n(CPAA only)")
        elif "25/75" in n:
            display_names.append(n + "\n(career heavy)")
        elif "67/" in n:
            display_names.append(n + "\n(CPAA heavy)")
        else:
            display_names.append(n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Catch Composite Grid Search — CPAA + Career aDOT-Adj Catch%",
        fontsize=15, fontweight="bold", y=0.98,
    )

    def plot_metric(ax, col, title, lower_better=True):
        values = results_df[col].tolist()
        baseline = values[0]
        colors = []
        for v in values:
            if lower_better:
                colors.append("#2ca02c" if v < baseline else "#d62728" if v > baseline else "#1f77b4")
            else:
                colors.append("#2ca02c" if v > baseline else "#d62728" if v < baseline else "#1f77b4")
        colors[0] = "#1f77b4"

        best_idx = int(np.argmin(values)) if lower_better else int(np.argmax(values))
        colors[best_idx] = "#9467bd"

        ax.bar(x, values, color=colors, edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=7.5, rotation=30, ha="right")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.axhline(baseline, color="#1f77b4", linestyle="--", alpha=0.4, linewidth=1)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        mn, mx = min(values), max(values)
        margin = (mx - mn) * 0.3
        ax.set_ylim(mn - margin, mx + margin)

    plot_metric(axes[0, 0], "h_ll", "Holdout LogLoss", lower_better=True)
    plot_metric(axes[0, 1], "h_brier", "Holdout Brier", lower_better=True)
    plot_metric(axes[0, 2], "h_elite_auc", "Holdout Elite AUC", lower_better=False)
    plot_metric(axes[1, 0], "l_ll", "LOO LogLoss", lower_better=True)
    plot_metric(axes[1, 1], "l_brier", "LOO Brier", lower_better=True)
    plot_metric(axes[1, 2], "l_elite_auc", "LOO Elite AUC", lower_better=False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Baseline (CPAA only)"),
        Patch(facecolor="#2ca02c", label="Better than baseline"),
        Patch(facecolor="#d62728", label="Worse than baseline"),
        Patch(facecolor="#9467bd", label="Best config"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    chart_path = os.path.join(DATA_DIR, "charts", "catch_composite_grid_search.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved chart to {chart_path}")
    plt.close()


def main():
    print("Loading training data...")
    train_df = load_training_data(max_year=2024)
    train_df["draft_capital"] = np.maximum(
        10 - (10 / np.log(261)) * np.log(train_df["pick"] + 1), 0
    )

    # Build composite columns
    composite_configs = build_composites(train_df)

    base_features = [
        "draft_capital", "pg_yprr_graduated", "PLACEHOLDER",
        "best2_contested_catch_rate", "best2_avoided_tackles_per_rec",
    ]

    results = []

    # Baseline: CPAA only
    print("\n===== 5F baseline (CPAA only) =====")
    features = ["draft_capital"] + COLLEGE_FEATURES
    h_ll, h_brier, h_elite = evaluate_holdout(train_df, features)
    l_ll, l_brier, l_elite = evaluate_loo(train_df, features)
    print(f"  H: LL={h_ll:.4f} B={h_brier:.4f} E={h_elite:.4f}  "
          f"L: LL={l_ll:.4f} B={l_brier:.4f} E={l_elite:.4f}")
    results.append({
        "config": "5F baseline", "h_ll": round(h_ll, 4), "h_brier": round(h_brier, 4),
        "h_elite_auc": round(h_elite, 4), "l_ll": round(l_ll, 4),
        "l_brier": round(l_brier, 4), "l_elite_auc": round(l_elite, 4),
    })

    # 6F: add career as separate feature
    print("\n===== 6F + career_adot_adj =====")
    features_6f = ["draft_capital"] + COLLEGE_FEATURES + ["career_catch_pct_adot_adj"]
    h_ll, h_brier, h_elite = evaluate_holdout(train_df, features_6f)
    l_ll, l_brier, l_elite = evaluate_loo(train_df, features_6f)
    print(f"  H: LL={h_ll:.4f} B={h_brier:.4f} E={h_elite:.4f}  "
          f"L: LL={l_ll:.4f} B={l_brier:.4f} E={l_elite:.4f}")
    results.append({
        "config": "6F + career", "h_ll": round(h_ll, 4), "h_brier": round(h_brier, 4),
        "h_elite_auc": round(h_elite, 4), "l_ll": round(l_ll, 4),
        "l_brier": round(l_brier, 4), "l_elite_auc": round(l_elite, 4),
    })

    # Composite configs
    for name, col in composite_configs.items():
        print(f"\n===== {name} =====")
        feats = base_features.copy()
        feats[2] = col

        h_ll, h_brier, h_elite = evaluate_holdout(train_df, feats)
        l_ll, l_brier, l_elite = evaluate_loo(train_df, feats)
        print(f"  H: LL={h_ll:.4f} B={h_brier:.4f} E={h_elite:.4f}  "
              f"L: LL={l_ll:.4f} B={l_brier:.4f} E={l_elite:.4f}")

        results.append({
            "config": name, "h_ll": round(h_ll, 4), "h_brier": round(h_brier, 4),
            "h_elite_auc": round(h_elite, 4), "l_ll": round(l_ll, 4),
            "l_brier": round(l_brier, 4), "l_elite_auc": round(l_elite, 4),
        })

    # Save results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(DATA_DIR, "outputs", "catch_composite_grid_search.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    print("\n===== SUMMARY =====")
    print(results_df.to_string(index=False))

    # Generate chart
    make_chart(results_df)


if __name__ == "__main__":
    main()
