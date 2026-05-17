#!/usr/bin/env python3
"""
WR Holdout Comparison: Sqrt vs Log Draft Capital.

Two-panel chart:
  Left:  Grouped bar chart of AUC by threshold (sqrt vs log)
  Right: LogLoss + Brier comparison with delta annotations

Outputs: wr_data/charts/wr_sqrt_vs_log_holdout.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Results from evaluate_holdout_log_dc.py
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
AUC_SQRT = [0.868, 0.866, 0.953, 0.957, 1.000]
AUC_LOG  = [0.892, 0.916, 0.963, 0.929, 0.989]

LOGLOSS_SQRT = 0.7845
LOGLOSS_LOG  = 0.7679
BRIER_SQRT   = 0.3477
BRIER_LOG    = 0.3377

C_SQRT = "#d62728"
C_LOG  = "#2ca02c"


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("WR Holdout Evaluation: Sqrt vs Log Draft Capital (n=88, 2022-2024)",
                 fontsize=13, fontweight="bold", y=0.98)

    # ── Left panel: AUC by threshold ──
    x = np.arange(len(THRESHOLD_LABELS))
    w = 0.32

    bars_sqrt = ax1.bar(x - w/2, AUC_SQRT, w, color=C_SQRT, alpha=0.85,
                        label="Sqrt (baseline)", edgecolor="white", linewidth=0.5)
    bars_log = ax1.bar(x + w/2, AUC_LOG, w, color=C_LOG, alpha=0.85,
                       label="Log", edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [bars_sqrt, bars_log]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Delta annotations
    for i in range(len(THRESHOLD_LABELS)):
        delta = AUC_LOG[i] - AUC_SQRT[i]
        color = C_LOG if delta > 0 else C_SQRT
        y_pos = max(AUC_SQRT[i], AUC_LOG[i]) + 0.025
        ax1.text(x[i], y_pos, f"{delta:+.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=color,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=color, alpha=0.8))

    ax1.set_xticks(x)
    ax1.set_xticklabels(THRESHOLD_LABELS, fontsize=10)
    ax1.set_ylabel("AUC", fontsize=11)
    ax1.set_title("AUC by Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.8, 1.06)
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(axis="y", alpha=0.2)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

    # ── Right panel: LogLoss + Brier ──
    metrics = ["LogLoss", "Brier"]
    sqrt_vals = [LOGLOSS_SQRT, BRIER_SQRT]
    log_vals = [LOGLOSS_LOG, BRIER_LOG]

    x2 = np.arange(len(metrics))
    w2 = 0.32

    bars_s = ax2.bar(x2 - w2/2, sqrt_vals, w2, color=C_SQRT, alpha=0.85,
                     label="Sqrt", edgecolor="white", linewidth=0.5)
    bars_l = ax2.bar(x2 + w2/2, log_vals, w2, color=C_LOG, alpha=0.85,
                     label="Log", edgecolor="white", linewidth=0.5)

    for bars in [bars_s, bars_l]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                     f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Delta annotations
    for i in range(len(metrics)):
        delta = log_vals[i] - sqrt_vals[i]
        y_pos = max(sqrt_vals[i], log_vals[i]) + 0.03
        ax2.text(x2[i], y_pos, f"{delta:+.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_LOG,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=C_LOG, alpha=0.8))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylabel("Score (lower = better)", fontsize=11)
    ax2.set_title("Calibration Metrics", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.95)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.2)

    # Footnote
    fig.text(0.5, 0.01,
             "Log formula: DC = 10 - (10 / ln(261)) * ln(pick + 1)    |    "
             "Sqrt formula: DC = 10 - 7 * sqrt(pick / 260)    |    "
             "Ensemble: 75% Bayesian + 25% XGBoost",
             ha="center", fontsize=8, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_dir = os.path.join(PROJECT_ROOT, "wr_data", "charts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wr_sqrt_vs_log_holdout.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
