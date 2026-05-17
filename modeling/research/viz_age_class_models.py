#!/usr/bin/env python3
"""
Visualize age-class model results.

Reads: wr_data/age_class_model_results.csv
Outputs:
  - wr_data/age_class_bivariate.png   -- AUC by age class (bivariate)
  - wr_data/age_class_trivariate.png  -- AUC by age class (trivariate)
  - wr_data/age_class_winners.png     -- Winners vs baselines vs age classes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

METRIC_COLORS = {"yprr": "#2196F3", "yptpa": "#FF9800", "ypg": "#4CAF50"}
METRIC_LABELS = {"yprr": "YPRR", "yptpa": "YPTPA", "ypg": "YPG"}
AGE_ORDER = ["freshman", "sophomore", "junior", "senior"]
AGE_LABELS = {"freshman": "Freshman\n(<19.5)", "sophomore": "Sophomore\n(19.5-20.5)",
              "junior": "Junior\n(20.5-21.5)", "senior": "Senior\n(21.5+)"}


def save(fig, name):
    path = os.path.join(DATA_DIR, f"age_class_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def chart_by_age_class(df, model_type, title_suffix):
    """AUC(Elite) and AUC(Stud) by age class, grouped by metric."""
    sub = df[(df["group"].isin(AGE_ORDER)) & (df["model_type"] == model_type)]

    # Get baselines
    baselines = df[df["group"] == "baseline"]
    if model_type == "bivariate":
        bl = baselines[baselines["label"] == "DC only"].iloc[0]
        bl_label = "DC only"
    else:
        bl = baselines[baselines["label"] == "DC + career_tqbr"].iloc[0]
        bl_label = "DC + tQBR"

    fig, (ax_elite, ax_stud) = plt.subplots(1, 2, figsize=(14, 5.5))
    title_prefix = "Bivariate (DC + metric)" if model_type == "bivariate" else "Trivariate (DC + metric + tQBR)"
    fig.suptitle(f"{title_prefix}: Predictive Power by Age Class",
                 fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(AGE_ORDER))
    width = 0.25

    for ax, auc_col, auc_label in [
        (ax_elite, "auc_>=Elite", "AUC (>=Elite)"),
        (ax_stud, "auc_>=Stud", "AUC (>=Stud)"),
    ]:
        for i, metric in enumerate(["yprr", "yptpa", "ypg"]):
            vals = []
            for age_class in AGE_ORDER:
                row = sub[(sub["group"] == age_class) & (sub["metric"] == metric)]
                if len(row) > 0 and auc_col in row.columns:
                    v = row[auc_col].values[0]
                    vals.append(v if pd.notna(v) else np.nan)
                else:
                    vals.append(np.nan)

            # Plot bars, handling NaN
            valid_mask = [not np.isnan(v) for v in vals]
            bar_vals = [v if not np.isnan(v) else 0 for v in vals]
            bars = ax.bar(x + i * width, bar_vals, width,
                          label=METRIC_LABELS[metric], color=METRIC_COLORS[metric],
                          alpha=0.85)
            # Gray out NaN bars
            for j, (bar, valid) in enumerate(zip(bars, valid_mask)):
                if not valid:
                    bar.set_alpha(0.1)

            # Annotate
            for j, (v, valid) in enumerate(zip(vals, valid_mask)):
                if valid:
                    ax.text(x[j] + i * width, v + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7, fontweight="bold")
                else:
                    ax.text(x[j] + i * width, 0.05, "N/A",
                            ha="center", va="bottom", fontsize=7, color="gray")

        # Baseline line
        bl_val = bl.get(auc_col, np.nan)
        if pd.notna(bl_val):
            ax.axhline(y=bl_val, color="gray", linestyle="--", linewidth=1.5,
                       label=f"{bl_label} ({bl_val:.3f})", alpha=0.7)

        # Sample size annotations at bottom
        for j, age_class in enumerate(AGE_ORDER):
            row = sub[(sub["group"] == age_class) & (sub["metric"] == "yprr")]
            if len(row) > 0:
                n = row["n_holdout"].values[0]
                ax.text(x[j] + width, -0.08, f"n={n:.0f}",
                        ha="center", va="top", fontsize=8, color="gray",
                        transform=ax.get_xaxis_transform())

        ax.set_xticks(x + width)
        ax.set_xticklabels([AGE_LABELS[a] for a in AGE_ORDER], fontsize=9)
        ax.set_ylabel(auc_label, fontsize=10)
        ax.set_title(auc_label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, title_suffix)


def chart_winners_comparison(df):
    """Compare winners vs baselines vs best age-class models."""
    fig, (ax_ll, ax_elite, ax_stud) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Full Comparison: Baselines vs Age Classes vs Age-Adjusted Winners",
                 fontsize=14, fontweight="bold", y=1.02)

    # Collect rows to show
    rows = []

    # Baselines
    bl_dc = df[df["label"] == "DC only"].iloc[0]
    rows.append(("DC only", bl_dc, "#9E9E9E"))
    bl_tqbr = df[df["label"] == "DC + career_tqbr"].iloc[0]
    rows.append(("DC + tQBR", bl_tqbr, "#757575"))

    # Best per age class (bivariate, yprr)
    for age_class in AGE_ORDER:
        sub = df[(df["group"] == age_class) & (df["model_type"] == "bivariate")
                 & (df["metric"] == "yprr")]
        if len(sub) > 0:
            r = sub.iloc[0]
            color = {"freshman": "#81D4FA", "sophomore": "#4FC3F7",
                     "junior": "#29B6F6", "senior": "#0288D1"}[age_class]
            rows.append((f"DC + {age_class} YPRR", r, color))

    # Winners (bivariate)
    winner_colors = {
        "best2_yprr_unadj": "#BDBDBD",
        "best1_yprr_senior": "#F44336",
        "best1_yprr_both": "#9C27B0",
        "best2_yprr_both": "#E040FB",
        "best2_yptpa_both": "#FF9800",
    }
    for var_name in ["best2_yprr_unadj", "best1_yprr_senior", "best1_yprr_both",
                     "best2_yprr_both", "best2_yptpa_both"]:
        sub = df[(df["group"] == "winner") & (df["model_type"] == "bivariate")
                 & (df["metric"] == var_name)]
        if len(sub) > 0:
            short = var_name.replace("best", "b").replace("_yprr", " YPRR").replace(
                "_yptpa", " YPTPA").replace("_both", " (both)").replace(
                "_senior", " (sr)").replace("_unadj", " (raw)")
            rows.append((f"DC + {short}", sub.iloc[0], winner_colors.get(var_name, "#666")))

    # Winners (trivariate)
    for var_name in ["best2_yprr_unadj", "best1_yprr_both", "best2_yprr_both"]:
        sub = df[(df["group"] == "winner") & (df["model_type"] == "trivariate")
                 & (df["metric"] == var_name)]
        if len(sub) > 0:
            short = var_name.replace("best", "b").replace("_yprr", " YPRR").replace(
                "_both", " (both)").replace("_unadj", " (raw)")
            rows.append((f"DC + {short} + tQBR", sub.iloc[0],
                         winner_colors.get(var_name, "#666")))

    labels = [r[0] for r in rows]
    y = np.arange(len(labels))

    # LogLoss
    vals = [r[1]["logloss"] for r in rows]
    colors = [r[2] for r in rows]
    ax_ll.barh(y, vals, color=colors, alpha=0.85, height=0.7)
    for i, v in enumerate(vals):
        ax_ll.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7.5, fontweight="bold")
    ax_ll.set_yticks(y)
    ax_ll.set_yticklabels(labels, fontsize=8)
    ax_ll.set_title("LogLoss (lower = better)", fontsize=11, fontweight="bold")
    ax_ll.invert_yaxis()
    ax_ll.grid(axis="x", alpha=0.3)

    # AUC Elite
    vals = [r[1].get("auc_>=Elite", np.nan) for r in rows]
    valid = [not (isinstance(v, float) and np.isnan(v)) for v in vals]
    ax_elite.barh(y, [v if vv else 0 for v, vv in zip(vals, valid)],
                  color=colors, alpha=0.85, height=0.7)
    for i, (v, vv) in enumerate(zip(vals, valid)):
        if vv:
            ax_elite.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5,
                          fontweight="bold")
        else:
            ax_elite.text(0.05, i, "N/A", va="center", fontsize=7.5, color="gray")
    ax_elite.set_yticks(y)
    ax_elite.set_yticklabels(labels, fontsize=8)
    ax_elite.set_title("AUC (>=Elite)", fontsize=11, fontweight="bold")
    ax_elite.invert_yaxis()
    ax_elite.grid(axis="x", alpha=0.3)

    # AUC Stud
    vals = [r[1].get("auc_>=Stud", np.nan) for r in rows]
    valid = [not (isinstance(v, float) and np.isnan(v)) for v in vals]
    ax_stud.barh(y, [v if vv else 0 for v, vv in zip(vals, valid)],
                 color=colors, alpha=0.85, height=0.7)
    for i, (v, vv) in enumerate(zip(vals, valid)):
        if vv:
            ax_stud.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5,
                         fontweight="bold")
        else:
            ax_stud.text(0.05, i, "N/A", va="center", fontsize=7.5, color="gray")
    ax_stud.set_yticks(y)
    ax_stud.set_yticklabels(labels, fontsize=8)
    ax_stud.set_title("AUC (>=Stud)", fontsize=11, fontweight="bold")
    ax_stud.invert_yaxis()
    ax_stud.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save(fig, "winners")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "age_class_model_results.csv"))
    print("Generating charts...")

    chart_by_age_class(df, "bivariate", "bivariate")
    chart_by_age_class(df, "trivariate", "trivariate")
    chart_winners_comparison(df)

    print("Done.")


if __name__ == "__main__":
    main()
