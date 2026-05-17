#!/usr/bin/env python3
"""
Visualize graduated age adjustment results.

Reads: wr_data/graduated_adjustment_results.csv
Outputs:
  - wr_data/grad_adj_single_sweeps.png   -- per-class sweep curves
  - wr_data/grad_adj_top_combos.png      -- top graduated combos vs baselines
  - wr_data/grad_adj_class_heatmap.png   -- sophomore x junior heatmap (fr/sr fixed)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

CLASS_COLORS = {
    "freshman": "#2196F3",
    "sophomore": "#4CAF50",
    "junior": "#FF9800",
    "senior": "#F44336",
}
COMBO_LABELS = {
    ("yprr", "best1"): "YPRR Best1",
    ("yprr", "best2"): "YPRR Best2",
    ("yptpa", "career"): "YPTPA Career",
}
COMBO_COLORS = {
    ("yprr", "best1"): "#2196F3",
    ("yprr", "best2"): "#9C27B0",
    ("yptpa", "career"): "#FF9800",
}


def save(fig, name):
    path = os.path.join(DATA_DIR, f"grad_adj_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def chart_single_sweeps(df):
    """Per-class sweep curves: how each age class adjustment affects Spearman and AUC(Stud)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Single-Class Age Adjustment Sweeps", fontsize=14, fontweight="bold", y=1.02)

    col_map = {
        "single_freshman": ("fr_adj", "Freshman Boost"),
        "single_sophomore": ("so_adj", "Sophomore Adj"),
        "single_junior": ("jr_adj", "Junior Discount"),
        "single_senior": ("sr_adj", "Senior Discount"),
    }

    for row_idx, (eval_col, ylabel) in enumerate([("spearman", "Spearman"), ("auc_stud", "AUC (>=Stud)")]):
        for col_idx, (metric, agg) in enumerate([("yprr", "best1"), ("yprr", "best2"), ("yptpa", "career")]):
            ax = axes[row_idx, col_idx]
            combo_label = COMBO_LABELS[(metric, agg)]

            sub = df[(df["metric"] == metric) & (df["agg"] == agg)]
            baseline = sub[sub["scheme"] == "baseline"][eval_col].values[0]

            for scheme_name, (adj_col, cls_label) in col_map.items():
                cls_data = sub[sub["scheme"] == scheme_name].copy()
                if len(cls_data) == 0:
                    continue

                # Add baseline point
                cls_data = pd.concat([
                    pd.DataFrame([{adj_col: 0, eval_col: baseline}]),
                    cls_data[[adj_col, eval_col]]
                ]).sort_values(adj_col)

                cls_name = scheme_name.replace("single_", "")
                ax.plot(cls_data[adj_col] * 100, cls_data[eval_col], "o-",
                        color=CLASS_COLORS[cls_name], label=cls_label,
                        markersize=6, linewidth=2)

            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_xlabel("Adjustment (%)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{combo_label} — {ylabel}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "single_sweeps")


def chart_top_combos(df):
    """Horizontal bar chart: top graduated combos vs baseline and previous best."""
    fig, (ax_sp, ax_auc) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Top Graduated Adjustments vs Previous Best",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, col, title in [
        (ax_sp, "spearman", "Spearman (higher = better)"),
        (ax_auc, "auc_stud", "AUC >=Stud (higher = better)"),
    ]:
        rows = []

        for metric, agg in [("yprr", "best1"), ("yprr", "best2"), ("yptpa", "career")]:
            combo_label = COMBO_LABELS[(metric, agg)]
            sub = df[(df["metric"] == metric) & (df["agg"] == agg)]

            # Baseline
            bl = sub[sub["scheme"] == "baseline"].iloc[0]
            rows.append((f"{combo_label}\nbaseline", bl[col], "#BDBDBD"))

            # Previous best (sr=-15%, fr=+15% equivalent -- scheme=graduated closest)
            prev = sub[
                (sub["scheme"] == "graduated") &
                (sub["fr_adj"] == 0.15) & (sub["sr_adj"] == -0.15) &
                (sub["so_adj"] == 0) & (sub["jr_adj"] == 0)
            ]
            if len(prev) > 0:
                rows.append((f"{combo_label}\nprev best (fr+15/sr-15)",
                             prev.iloc[0][col], "#78909C"))

            # Best by this metric
            best = sub.sort_values(col, ascending=False).iloc[0]
            rows.append((f"{combo_label}\n{best['label']}", best[col],
                         COMBO_COLORS[(metric, agg)]))

            # Spacer
            rows.append(("", 0, "white"))

        # Remove trailing spacer
        rows = rows[:-1]

        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [r[2] for r in rows]
        y = np.arange(len(labels))

        bars = ax.barh(y, vals, color=colors, alpha=0.85, height=0.7)
        for i, (v, label) in enumerate(zip(vals, labels)):
            if label and v > 0:
                ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Tight x range
        valid_vals = [v for v in vals if v > 0]
        if valid_vals:
            vmin, vmax = min(valid_vals), max(valid_vals)
            margin = (vmax - vmin) * 0.3
            ax.set_xlim(vmin - margin, vmax + margin * 1.5)

    fig.tight_layout()
    save(fig, "top_combos")


def chart_class_heatmap(df):
    """Heatmaps: sophomore x junior adjustment, with fr=+25% and sr=-25% fixed. YPRR best1 only."""
    sub = df[
        (df["metric"] == "yprr") & (df["agg"] == "best1") &
        (df["scheme"] == "graduated") &
        (df["fr_adj"] == 0.25) & (df["sr_adj"] == -0.25)
    ]

    so_vals = sorted(sub["so_adj"].unique())
    jr_vals = sorted(sub["jr_adj"].unique())

    fig, (ax_sp, ax_auc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("YPRR Best1: Sophomore x Junior Adjustment\n(Freshman +25%, Senior -25% fixed)",
                 fontsize=13, fontweight="bold", y=1.05)

    for ax, col, title, cmap in [
        (ax_sp, "spearman", "Spearman", "YlOrRd"),
        (ax_auc, "auc_stud", "AUC (>=Stud)", "YlGnBu"),
    ]:
        heat = np.full((len(so_vals), len(jr_vals)), np.nan)
        for i, so in enumerate(so_vals):
            for j, jr in enumerate(jr_vals):
                row = sub[(sub["so_adj"] == so) & (sub["jr_adj"] == jr)]
                if len(row) > 0:
                    heat[i, j] = row.iloc[0][col]

        im = ax.imshow(heat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(jr_vals)))
        ax.set_xticklabels([f"{v*100:+.0f}%" for v in jr_vals], fontsize=9)
        ax.set_yticks(range(len(so_vals)))
        ax.set_yticklabels([f"{v*100:+.0f}%" for v in so_vals], fontsize=9)
        ax.set_xlabel("Junior Adjustment", fontsize=10)
        ax.set_ylabel("Sophomore Adjustment", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i in range(len(so_vals)):
            for j in range(len(jr_vals)):
                val = heat[i, j]
                if not np.isnan(val):
                    vmin, vmax = np.nanmin(heat), np.nanmax(heat)
                    mid = (vmin + vmax) / 2
                    color = "white" if val > mid else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8.5, color=color, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.tight_layout()
    save(fig, "class_heatmap")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "graduated_adjustment_results.csv"))
    print("Generating charts...")

    chart_single_sweeps(df)
    chart_top_combos(df)
    chart_class_heatmap(df)

    print("Done.")


if __name__ == "__main__":
    main()
