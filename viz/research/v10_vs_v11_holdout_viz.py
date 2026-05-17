#!/usr/bin/env python3
"""
v10 vs v11 WR Holdout Comparison.

v10: sqrt DC + 75/25 Bayesian/XGBoost
v11: log DC + 60/40 Bayesian/XGBoost

Outputs: wr_data/charts/v10_vs_v11_holdout.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# v10 results (sqrt DC, 75/25) from evaluate_holdout_log_dc.py sqrt baseline
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
AUC_V10 = [0.868, 0.866, 0.953, 0.957, 1.000]
AUC_V11 = [0.888, 0.920, 0.970, 0.941, 0.989]

LL_V10 = 0.7845
LL_V11 = 0.7731
BR_V10 = 0.3477
BR_V11 = 0.3404

C_V10 = "#d62728"
C_V11 = "#2ca02c"


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("WR Holdout: v10 (sqrt 75/25) vs v11 (log 60/40)  —  n=88, 2022-2024",
                 fontsize=13, fontweight="bold", y=0.98)

    # ── Left panel: AUC by threshold ──
    x = np.arange(len(THRESHOLD_LABELS))
    w = 0.32

    bars_v10 = ax1.bar(x - w/2, AUC_V10, w, color=C_V10, alpha=0.85,
                        label="v10 (sqrt, 75/25)", edgecolor="white", linewidth=0.5)
    bars_v11 = ax1.bar(x + w/2, AUC_V11, w, color=C_V11, alpha=0.85,
                        label="v11 (log, 60/40)", edgecolor="white", linewidth=0.5)

    for bars in [bars_v10, bars_v11]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Delta annotations
    for i in range(len(THRESHOLD_LABELS)):
        delta = AUC_V11[i] - AUC_V10[i]
        color = C_V11 if delta > 0 else C_V10
        y_pos = max(AUC_V10[i], AUC_V11[i]) + 0.025
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
    v10_vals = [LL_V10, BR_V10]
    v11_vals = [LL_V11, BR_V11]

    x2 = np.arange(len(metrics))
    w2 = 0.32

    bars_s = ax2.bar(x2 - w2/2, v10_vals, w2, color=C_V10, alpha=0.85,
                     label="v10", edgecolor="white", linewidth=0.5)
    bars_l = ax2.bar(x2 + w2/2, v11_vals, w2, color=C_V11, alpha=0.85,
                     label="v11", edgecolor="white", linewidth=0.5)

    for bars in [bars_s, bars_l]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                     f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Delta annotations
    for i in range(len(metrics)):
        delta = v11_vals[i] - v10_vals[i]
        y_pos = max(v10_vals[i], v11_vals[i]) + 0.03
        ax2.text(x2[i], y_pos, f"{delta:+.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_V11,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=C_V11, alpha=0.8))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylabel("Score (lower = better)", fontsize=11)
    ax2.set_title("Calibration Metrics", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.95)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.2)

    # Footnote
    fig.text(0.5, 0.01,
             "v10: DC = 10 - 7 * sqrt(pick / 260), 75% Bayes / 25% XGB    |    "
             "v11: DC = 10 - (10 / ln(261)) * ln(pick + 1), 60% Bayes / 40% XGB",
             ha="center", fontsize=8, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_dir = os.path.join(PROJECT_ROOT, "wr_data", "charts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v10_vs_v11_holdout.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
