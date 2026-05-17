#!/usr/bin/env python3
"""
Visualize YPRR substitution model comparison results.

Reads: wr_data/yprr_substitution_results.csv
Outputs:
  - wr_data/yprr_sub_deltas.png        -- delta vs baseline for key metrics
  - wr_data/yprr_sub_abs_metrics.png   -- absolute metric values side by side
  - wr_data/yprr_sub_model_breakdown.png -- XGB vs Bayes vs Ensemble per variant
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

VARIANT_LABELS = {
    "baseline_best2_yprr": "Baseline\n(best2 YPRR)",
    "best2_yptpa_both": "Best2 YPTPA\n(both)",
    "best1_yprr_senior": "Best1 YPRR\n(senior)",
    "best1_yprr_both": "Best1 YPRR\n(both)",
    "best2_yprr_both": "Best2 YPRR\n(both)",
    "career_yptpa_both": "Career YPTPA\n(both)",
}

VARIANT_COLORS = {
    "baseline_best2_yprr": "#9E9E9E",
    "best2_yptpa_both": "#FF9800",
    "best1_yprr_senior": "#F44336",
    "best1_yprr_both": "#9C27B0",
    "best2_yprr_both": "#2196F3",
    "career_yptpa_both": "#4CAF50",
}


def save(fig, name):
    path = os.path.join(DATA_DIR, f"yprr_sub_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def chart_deltas(df):
    """Bar chart: delta vs baseline for LogLoss, Brier, AUC(Elite), AUC(Stud)."""
    baseline = df.iloc[0]
    variants = df.iloc[1:]

    metrics = [
        ("ens_logloss", "LogLoss", -1),       # negative = better, flip sign for visual
        ("ens_brier", "Brier", -1),
        ("ens_auc_>=Elite", "AUC (Elite)", 1),
        ("ens_auc_>=Stud", "AUC (Stud)", 1),
        ("ens_auc_>=LW", "AUC (LW)", 1),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("Improvement Over Baseline (best2_yprr, no age adjustment)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (col, label, direction) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        base_val = baseline[col]
        names = []
        deltas = []
        colors = []
        for _, row in variants.iterrows():
            raw_delta = row[col] - base_val
            # For loss metrics, improvement = negative delta, so flip for display
            display_delta = raw_delta * direction
            names.append(VARIANT_LABELS.get(row["variant"], row["variant"]))
            deltas.append(display_delta)
            colors.append(VARIANT_COLORS.get(row["variant"], "#666"))

        bars = ax.barh(range(len(names)), deltas, color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Annotate values
        for i, (bar, d) in enumerate(zip(bars, deltas)):
            raw = variants.iloc[i][col] - base_val
            ax.text(d + 0.001 * np.sign(d), i,
                    f"{raw:+.4f}", va="center", fontsize=8, fontweight="bold")

        if direction == -1:
            ax.set_xlabel("Improvement (lower is better)", fontsize=8)
        else:
            ax.set_xlabel("Improvement (higher is better)", fontsize=8)

    fig.tight_layout()
    save(fig, "deltas")


def chart_absolute(df):
    """Grouped bars: absolute metric values for each variant."""
    metrics = [
        ("ens_logloss", "LogLoss"),
        ("ens_brier", "Brier"),
        ("ens_auc_>=Elite", "AUC (Elite)"),
        ("ens_auc_>=Stud", "AUC (Stud)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("Absolute Ensemble Metrics by Variant",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (col, label) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        names = [VARIANT_LABELS.get(v, v) for v in df["variant"]]
        vals = df[col].values
        colors = [VARIANT_COLORS.get(v, "#666") for v in df["variant"]]

        bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Tight x-axis
        vmin, vmax = vals.min(), vals.max()
        margin = (vmax - vmin) * 0.3
        if "logloss" in col or "brier" in col:
            ax.set_xlim(vmin - margin, vmax + margin * 0.5)
        else:
            ax.set_xlim(vmin - margin * 0.5, min(vmax + margin, 1.0))

        for bar, v in zip(bars, vals):
            ax.text(v + margin * 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    save(fig, "abs_metrics")


def chart_model_breakdown(df):
    """For each variant: compare XGB vs Bayes vs Ensemble on key metrics."""
    key_metrics = [
        ("logloss", "LogLoss"),
        ("brier", "Brier"),
        ("auc_>=Elite", "AUC (Elite)"),
        ("auc_>=Stud", "AUC (Stud)"),
    ]
    model_colors = {"XGBoost": "#FF9800", "Bayesian": "#2196F3", "Ensemble": "#9C27B0"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Component Breakdown by Variant",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (metric_suffix, metric_label) in zip(axes.flat, key_metrics):
        x = np.arange(len(df))
        width = 0.25

        for i, (prefix, model_name) in enumerate([
            ("xgb_", "XGBoost"), ("bayes_", "Bayesian"), ("ens_", "Ensemble")
        ]):
            col = f"{prefix}{metric_suffix}"
            if col not in df.columns:
                continue
            vals = df[col].values
            ax.bar(x + i * width, vals, width, label=model_name,
                   color=model_colors[model_name], alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [VARIANT_LABELS.get(v, v) for v in df["variant"]],
            fontsize=7, rotation=15, ha="right"
        )
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Tight y-axis
        all_vals = []
        for prefix in ["xgb_", "bayes_", "ens_"]:
            col = f"{prefix}{metric_suffix}"
            if col in df.columns:
                all_vals.extend(df[col].values)
        if all_vals:
            vmin, vmax = min(all_vals), max(all_vals)
            margin = (vmax - vmin) * 0.3
            ax.set_ylim(vmin - margin, vmax + margin)

    fig.tight_layout()
    save(fig, "model_breakdown")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "yprr_substitution_results.csv"))
    print("Generating charts...")

    chart_deltas(df)
    chart_absolute(df)
    chart_model_breakdown(df)

    print("Done.")


if __name__ == "__main__":
    main()
