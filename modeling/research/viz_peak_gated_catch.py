#!/usr/bin/env python3
"""
Generate publication-quality visualizations for the peak-gated catch metric report.

Charts:
  1. Engineering progression: stacking effects of each adjustment
  2. Selection method comparison (best1 vs pg vs peak) across stats
  3. Top combo results across all 5 metrics
  4. 7-part analysis comparison for catch_pct_adot_adj variants
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
CHARTS_DIR = os.path.join(DATA_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Dark theme
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

# Colors
C_GREEN = "#3fb950"
C_RED = "#f85149"
C_BLUE = "#58a6ff"
C_ORANGE = "#f0883e"
C_YELLOW = "#d29922"
C_PURPLE = "#bc8cff"
C_GRAY = "#8b949e"
C_LIGHT = "#e6edf3"

combos = pd.read_csv(os.path.join(DATA_DIR, "outputs", "peak_gated_combos.csv"))
full = pd.read_csv(os.path.join(DATA_DIR, "outputs", "peak_gated_full.csv"))


# ======================================================================
# FIGURE 1: Engineering Progression — Stacking Effects
# ======================================================================

def fig1_progression():
    """Show the cumulative effect of each engineering step on LogLoss."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    # --- Panel A: LogLoss progression (replace QBR) ---
    ax = axes[0]
    stages = [
        ("v11 (career QBR)", 2.347, C_RED),
        ("+ aDOT adjust\n(best1_cpaa)", 2.142, C_ORANGE),
        ("+ graduated age adj\n(best1_cpaa_grad)", 1.972, C_YELLOW),
        ("+ peak-gated select\n(pg_cpaa_grad)", 1.670, C_GREEN),
    ]
    x = range(len(stages))
    bars = ax.bar(x, [s[1] for s in stages], color=[s[2] for s in stages],
                  alpha=0.8, width=0.6, edgecolor="#30363d", linewidth=1.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax.set_ylabel("Ordinal LogLoss (lower = better)", fontsize=11)
    ax.set_title("A. Cumulative Engineering Effect on LogLoss\n(replacing QBR, keeping best2_catch%_adot_adj)",
                 fontweight="bold", fontsize=12)

    # Annotations showing delta
    for i in range(1, len(stages)):
        delta = stages[i][1] - stages[i-1][1]
        pct = delta / stages[i-1][1] * 100
        ax.annotate(f"{delta:+.3f}\n({pct:+.1f}%)",
                    xy=(i, stages[i][1]), xytext=(i - 0.4, stages[i][1] + 0.06),
                    fontsize=9, color=C_LIGHT, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1))

    # Value labels on bars
    for bar, (_, val, _) in zip(bars, stages):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.05,
                f"{val:.3f}", ha="center", va="top", fontsize=11,
                fontweight="bold", color="#0d1117")

    ax.set_ylim(0, 2.6)
    ax.axhline(2.347, color=C_RED, linewidth=1, linestyle="--", alpha=0.4, label="v11 baseline")

    # Total improvement annotation
    total_delta = stages[-1][1] - stages[0][1]
    total_pct = total_delta / stages[0][1] * 100
    ax.text(0.95, 0.95, f"Total: {total_delta:+.3f} ({total_pct:+.1f}%)",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=C_GREEN, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor=C_GREEN, alpha=0.9))

    # --- Panel B: All 5 metrics progression ---
    ax = axes[1]
    metrics = {
        "LogLoss": [2.347, 2.142, 1.972, 1.670],
        "Brier": [0.515, 0.521, 0.499, 0.494],
        "Elite AUC": [0.842, 0.838, 0.861, 0.863],
        "Stud AUC": [0.778, 0.768, 0.773, 0.780],
        "Starter AUC": [0.833, 0.834, 0.852, 0.849],
    }
    stage_labels = ["v11", "+ aDOT adj", "+ graduated", "+ peak-gated"]
    lower_better = {"LogLoss": True, "Brier": True, "Elite AUC": False, "Stud AUC": False, "Starter AUC": False}

    # Normalize each metric to show relative improvement (0 = v11, 1 = best possible)
    x_pos = np.arange(len(stage_labels))
    width = 0.15
    metric_colors = [C_RED, C_ORANGE, C_BLUE, C_PURPLE, C_GREEN]

    for i, (metric, vals) in enumerate(metrics.items()):
        # Normalize: how much of the gap from v11 to best has been closed?
        v11_val = vals[0]
        best_val = min(vals) if lower_better[metric] else max(vals)
        if abs(best_val - v11_val) < 1e-9:
            normalized = [0] * len(vals)
        else:
            if lower_better[metric]:
                normalized = [(v11_val - v) / (v11_val - best_val) for v in vals]
            else:
                normalized = [(v - v11_val) / (best_val - v11_val) for v in vals]
        ax.bar(x_pos + i * width - 2 * width, normalized, width * 0.9,
               label=metric, color=metric_colors[i], alpha=0.7, edgecolor="#30363d")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylabel("Fraction of Maximum Improvement", fontsize=11)
    ax.set_title("B. All Metrics: Fraction of Improvement Captured at Each Stage",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.axhline(0, color=C_GRAY, linewidth=1, linestyle="--", alpha=0.5)
    ax.axhline(1, color=C_GREEN, linewidth=1, linestyle="--", alpha=0.3)
    ax.set_ylim(-0.3, 1.15)

    fig.suptitle("Peak-Gated Catch% aDOT Adjusted Graduated: Engineering Progression",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "pg_cpaa_progression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ======================================================================
# FIGURE 2: Selection Method Comparison
# ======================================================================

def fig2_selection_methods():
    """Compare best1, peak-gated, and pure peak across catch metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Data: LOO delta from full model base
    stats = {
        "catch_pct_adot_adj\n(graduated)": {
            "best1": 0.017, "peak-gated": 0.021, "pure peak": 0.003,
        },
        "catch_minus_drops\n(graduated)": {
            "best1": 0.015, "peak-gated": 0.022, "pure peak": -0.002,
        },
        "clean_catch_rate\n(graduated)": {
            "best1": 0.013, "peak-gated": 0.015, "pure peak": -0.003,
        },
        "cpaa_minus_drops\n(graduated)": {
            "best1": 0.016, "peak-gated": None, "pure peak": -0.001,
        },
    }

    # Panel A: LOO-AUC delta
    ax = axes[0]
    stat_names = list(stats.keys())
    x = np.arange(len(stat_names))
    width = 0.25
    method_colors = {"best1": C_BLUE, "peak-gated": C_GREEN, "pure peak": C_ORANGE}

    for i, (method, color) in enumerate(method_colors.items()):
        vals = [stats[s].get(method) for s in stat_names]
        valid_x = [xx for xx, v in zip(x, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        ax.bar([xx + (i - 1) * width for xx in valid_x], valid_v,
               width * 0.9, label=method, color=color, alpha=0.7, edgecolor="#30363d")

    ax.set_xticks(x)
    ax.set_xticklabels(stat_names, fontsize=9)
    ax.set_ylabel("LOO-AUC Delta (vs 4-anchor base)")
    ax.set_title("A. LOO-AUC: Peak-Gated Wins on Every Stat", fontweight="bold", fontsize=11)
    ax.axhline(0, color=C_GRAY, linewidth=1, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel B: Era drift
    ax = axes[1]
    era_data = {
        "catch_pct_adot_adj\n(graduated)": {"best1": 0.028, "peak-gated": 0.019, "pure peak": 0.088},
        "catch_minus_drops\n(graduated)": {"best1": 0.033, "peak-gated": 0.005, "pure peak": 0.074},
        "clean_catch_rate\n(graduated)": {"best1": 0.029, "peak-gated": 0.125, "pure peak": 0.109},
    }
    stat_names_era = list(era_data.keys())
    x2 = np.arange(len(stat_names_era))
    for i, (method, color) in enumerate(method_colors.items()):
        vals = [era_data[s].get(method, None) for s in stat_names_era]
        valid_x = [xx for xx, v in zip(x2, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        ax.bar([xx + (i - 1) * width for xx in valid_x], valid_v,
               width * 0.9, label=method, color=color, alpha=0.7, edgecolor="#30363d")

    ax.set_xticks(x2)
    ax.set_xticklabels(stat_names_era, fontsize=9)
    ax.set_ylabel("Era Drift (lower = more stable)")
    ax.set_title("B. Era Stability: Pure Peak is Noisy", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)

    # Panel C: Collinearity with anchors
    ax = axes[2]
    collin_data = {
        "catch_pct_adot_adj\n(graduated)": {"best1": 0.436, "peak-gated": 0.451, "pure peak": 0.458},
        "catch_minus_drops\n(graduated)": {"best1": 0.340, "peak-gated": 0.330, "pure peak": 0.273},
        "clean_catch_rate\n(graduated)": {"best1": 0.302, "peak-gated": 0.300, "pure peak": 0.298},
    }
    stat_names_col = list(collin_data.keys())
    x3 = np.arange(len(stat_names_col))
    for i, (method, color) in enumerate(method_colors.items()):
        vals = [collin_data[s].get(method, None) for s in stat_names_col]
        valid_x = [xx for xx, v in zip(x3, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        ax.bar([xx + (i - 1) * width for xx in valid_x], valid_v,
               width * 0.9, label=method, color=color, alpha=0.7, edgecolor="#30363d")

    ax.set_xticks(x3)
    ax.set_xticklabels(stat_names_col, fontsize=9)
    ax.set_ylabel("Max Collinearity with Anchors")
    ax.set_title("C. Collinearity: All Methods Similar", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Selection Method Comparison: best1 vs peak-gated vs pure peak",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "pg_cpaa_selection_methods.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ======================================================================
# FIGURE 3: Top Combinations — Multi-Metric Dashboard
# ======================================================================

def fig3_combo_dashboard():
    """Show top combos across LogLoss, Brier, Elite AUC, Stud AUC, Starter AUC."""
    fig = plt.figure(figsize=(26, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Key combos to highlight
    key_combos = [
        "v11 (QBR + catch%_adot_adj)",
        "v11 minus QBR",
        "4 anchors only",
        "QBR => pg_catch_pct_adot_adj_graduated",
        "QBR => best1_catch_pct_adot_adj_graduated",
        "QBR => pg_clean_catch_rate_graduated",
        "QBR => pg_catch_minus_drops_graduated",
        "both => pg_catch_pct_adot_adj_graduated",
        "both => best1_clean_catch_rate_graduated",
        "both => pg_catch_minus_drops_graduated",
        "catch%_adot => pg_catch_pct_adot_adj_graduated",
    ]
    cr = combos[combos["combo"].isin(key_combos)].copy()

    v11_vals = combos[combos["combo"].str.contains("v11.*QBR")].iloc[0]

    metric_configs = [
        (gs[0, 0], "log_loss", v11_vals["log_loss"], "Ordinal LogLoss", True),
        (gs[0, 1], "brier", v11_vals["brier"], "Ordinal Brier Score", True),
        (gs[0, 2], "elite_auc", v11_vals["elite_auc"], ">=Elite AUC", False),
        (gs[1, 0], "stud_auc", v11_vals["stud_auc"], ">=Stud AUC", False),
        (gs[1, 1], "starter_auc", v11_vals["starter_auc"], ">=Starter AUC", False),
    ]

    for gs_pos, metric, v11_val, title, lower_better in metric_configs:
        ax = fig.add_subplot(gs_pos)
        sorted_cr = cr.sort_values(metric, ascending=not lower_better).copy()
        y_pos = range(len(sorted_cr))

        bar_colors = []
        for v in sorted_cr[metric]:
            if lower_better:
                bar_colors.append(C_GREEN if v <= v11_val else C_RED)
            else:
                bar_colors.append(C_GREEN if v >= v11_val else C_RED)

        bars = ax.barh(list(y_pos), sorted_cr[metric].values, color=bar_colors, alpha=0.7,
                       edgecolor="#30363d", linewidth=0.5)
        ax.axvline(v11_val, color=C_ORANGE, linewidth=2, linestyle="--", alpha=0.8)

        # Short labels
        short = sorted_cr["combo"].str.replace("catch_pct_adot_adj", "cpaa", regex=False)
        short = short.str.replace("clean_catch_rate", "ccr", regex=False)
        short = short.str.replace("catch_minus_drops", "cmd", regex=False)
        short = short.str.replace("_graduated", "_gr", regex=False)
        short = short.str.replace("catch%_adot", "cpaa_b2", regex=False)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(short.values, fontsize=8)

        direction = "lower" if lower_better else "higher"
        ax.set_title(f"{title} ({direction} = better)", fontweight="bold", fontsize=10)

        for bar, val in zip(bars, sorted_cr[metric].values):
            delta = val - v11_val
            ax.text(val + (0.003 if not lower_better else 0.008),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f} ({delta:+.3f})", va="center", fontsize=7,
                    color=C_GREEN if (lower_better and delta < 0) or (not lower_better and delta > 0) else C_RED)

    # Panel F: Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    summary_data = [
        ["Metric", "v11", "pg_cpaa_gr\n(QBR=>)", "Delta", "% Chg"],
        ["LogLoss", "2.347", "1.670", "-0.677", "-29%"],
        ["Brier", "0.515", "0.494", "-0.021", "-4%"],
        ["Elite AUC", "0.842", "0.863", "+0.021", "+2.5%"],
        ["Stud AUC", "0.778", "0.780", "+0.002", "+0.3%"],
        ["Starter AUC", "0.833", "0.849", "+0.016", "+1.9%"],
    ]

    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(fontweight="bold", color=C_LIGHT)
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color=C_LIGHT)
            # Color the delta column
            if col == 3 or col == 4:
                text = cell.get_text().get_text()
                if text.startswith("-") and col == 3:
                    cell.set_text_props(color=C_GREEN, fontweight="bold")
                elif text.startswith("+"):
                    cell.set_text_props(color=C_GREEN, fontweight="bold")

    ax.set_title("F. Head-to-Head: v11 vs Best Replacement", fontweight="bold", fontsize=11, pad=20)

    fig.suptitle("Peak-Gated Catch% aDOT Adj Graduated: Multi-Metric Results",
                 fontsize=15, fontweight="bold", y=1.0)
    path = os.path.join(CHARTS_DIR, "pg_cpaa_combo_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ======================================================================
# FIGURE 4: 7-Part Analysis — catch_pct_adot_adj variants
# ======================================================================

def fig4_seven_part():
    """Detailed 7-part analysis for catch_pct_adot_adj family."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # Focus on cpaa variants only
    cpaa = full[full["feature"].str.contains("catch_pct_adot_adj|career_targeted|best2_catch_pct")].copy()
    cpaa = cpaa.sort_values("loo_delta", ascending=True)

    def short_name(f):
        return (f.replace("catch_pct_adot_adj", "cpaa")
                 .replace("_graduated", "_gr")
                 .replace("career_targeted_qb_rating", "career_QBR")
                 .replace("best2_catch_pct_adot_adj", "best2_cpaa (v11)"))

    cpaa["short"] = cpaa["feature"].apply(short_name)

    def color_by_method(f):
        if f.startswith("pg_"):
            return C_GREEN
        elif f.startswith("peak_"):
            return C_ORANGE
        elif f.startswith("best1_"):
            return C_BLUE
        elif "career" in f:
            return C_RED
        return C_YELLOW

    colors = [color_by_method(f) for f in cpaa["feature"]]

    # Panel A: LOO-AUC Delta
    ax = axes[0, 0]
    y_pos = range(len(cpaa))
    ax.barh(list(y_pos), cpaa["loo_delta"].values, color=colors, alpha=0.7, edgecolor="#30363d")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cpaa["short"].values, fontsize=8)
    ax.axvline(0, color=C_GRAY, linewidth=1, linestyle="--")
    ax.set_xlabel("LOO-AUC Delta")
    ax.set_title("A. LOO-AUC Delta\n(full model base)", fontweight="bold")
    for y, val in zip(y_pos, cpaa["loo_delta"].values):
        ax.text(val + 0.0005, y, f"{val:+.3f}", va="center", fontsize=8,
                color=C_GREEN if val > 0 else C_RED)

    # Panel B: Spearman
    ax = axes[0, 1]
    ax.barh(list(y_pos), cpaa["spearman"].values, color=colors, alpha=0.7, edgecolor="#30363d")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cpaa["short"].values, fontsize=8)
    ax.set_xlabel("Spearman rho")
    ax.set_title("B. Univariate Spearman\n(with tier outcome)", fontweight="bold")

    # Panel C: Era Drift
    ax = axes[0, 2]
    drift_colors = [C_GREEN if d < 0.05 else C_YELLOW if d < 0.1 else C_RED for d in cpaa["era_drift"]]
    ax.barh(list(y_pos), cpaa["era_drift"].values, color=drift_colors, alpha=0.7, edgecolor="#30363d")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cpaa["short"].values, fontsize=8)
    ax.set_xlabel("Era Drift (lower = more stable)")
    ax.set_title("C. Era Stability\n(|early - late| Spearman)", fontweight="bold")
    ax.axvline(0.05, color=C_GREEN, linewidth=1, linestyle=":", alpha=0.5, label="good (<0.05)")
    ax.axvline(0.10, color=C_RED, linewidth=1, linestyle=":", alpha=0.5, label="concerning (>0.10)")
    ax.legend(fontsize=8)

    # Panel D: Collinearity
    ax = axes[1, 0]
    ax.barh(list(y_pos), cpaa["max_collinearity"].values, color=colors, alpha=0.7, edgecolor="#30363d")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cpaa["short"].values, fontsize=8)
    ax.set_xlabel("Max Collinearity with Anchors")
    ax.set_title("D. Collinearity\n(lower = more independent)", fontweight="bold")
    ax.axvline(0.5, color=C_RED, linewidth=1, linestyle=":", alpha=0.5, label="high (>0.5)")
    ax.legend(fontsize=8)

    # Panel E: Residual + Bootstrap
    ax = axes[1, 1]
    ax.scatter(cpaa["residual"], cpaa["boot_pct_pos"], c=colors, s=80, alpha=0.7,
               edgecolors="#30363d", linewidths=1)
    for _, row in cpaa.iterrows():
        ax.annotate(short_name(row["feature"]),
                    (row["residual"], row["boot_pct_pos"]),
                    fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0, color=C_GRAY, linewidth=1, linestyle="--")
    ax.axhline(0.5, color=C_GRAY, linewidth=1, linestyle=":")
    ax.set_xlabel("Residual Spearman (after anchors)")
    ax.set_ylabel("Bootstrap % Positive")
    ax.set_title("E. Residual Signal Reliability\n(right+top = genuine signal)", fontweight="bold")

    # Panel F: Legend / Method Guide
    ax = axes[1, 2]
    ax.axis("off")

    legend_items = [
        (C_GREEN, "pg_ (peak-gated)", "Peak stat from seasons with grade >= 80.\nFalls back to best1 if none qualify."),
        (C_BLUE, "best1_ (current)", "Season with highest PFF offensive grade."),
        (C_ORANGE, "peak_ (pure peak)", "Season with highest stat value,\nregardless of grade. No quality gate."),
        (C_RED, "career_ (QBR incumbent)", "Target-weighted career average."),
        (C_YELLOW, "best2_ (catch% incumbent)", "Best 2 seasons by grade, averaged."),
    ]

    y_start = 0.9
    for color, name, desc in legend_items:
        ax.add_patch(FancyBboxPatch((0.02, y_start - 0.04), 0.04, 0.06,
                                     boxstyle="round,pad=0.01", facecolor=color, alpha=0.7))
        ax.text(0.1, y_start, name, fontsize=11, fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(0.1, y_start - 0.06, desc, fontsize=9, va="center", transform=ax.transAxes, color=C_GRAY)
        y_start -= 0.18

    ax.set_title("F. Selection Method Guide", fontweight="bold", fontsize=11)

    fig.suptitle("7-Part Analysis: catch_pct_adot_adj Family (Full Model Base)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "pg_cpaa_7part.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print("Generating peak-gated catch metric visualizations...")
    fig1_progression()
    fig2_selection_methods()
    fig3_combo_dashboard()
    fig4_seven_part()
    print("\nDone!")
