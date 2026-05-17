#!/usr/bin/env python3
"""
Draft Capital Curve Comparison: Log vs Jimmy Johnson vs Sqrt.

4-panel visualization:
  1. Curve shapes (value vs pick number)
  2. R² comparison across positions (RB + WR)
  3. Round-level value distribution vs actual dynasty value
  4. Residual analysis by round

Outputs: rb_data/charts/draft_capital_curves.png
"""

import math
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# === Define curves ===

JJ_PICKS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
            61,62,63,64,65,96,97,100,128,150,175,200,224,256]
JJ_VALUES = [3000,2600,2200,1800,1700,1600,1500,1400,1350,1300,
             1250,1200,1150,1100,1050,1000,950,900,875,850,
             800,780,760,740,720,700,680,660,640,620,600,590,
             580,560,550,540,530,520,510,500,490,480,470,460,
             450,440,430,420,410,400,390,380,370,360,350,340,
             330,320,310,300,292,284,276,270,265,116,112,100,
             44,31.4,21.4,11,1.8,1]
JJ_INTERP = interp1d(JJ_PICKS, JJ_VALUES, kind="linear", fill_value="extrapolate")

TIER_ORDER = {"Bust": 0, "Flex": 1, "Starter": 2, "Elite": 3, "Stud": 4, "League-Winner": 5}


def dc_jj_norm(pick):
    """Jimmy Johnson value normalized to 0-10 scale."""
    raw = float(JJ_INTERP(min(max(pick, 1), 256)))
    mn, mx = float(JJ_INTERP(256)), float(JJ_INTERP(1))
    return (raw - mn) / (mx - mn) * 10


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


def dc_sqrt(pick):
    return 10 - 7 * math.sqrt(pick / 260)


def load_position(pos):
    """Load resolved data for a position."""
    if pos == "WR":
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "wr_data", "wr_dynasty_value_with_college.csv"))
        df = df[df["computed_tier"] != "TBD"].copy()
    else:
        train_path = os.path.join(PROJECT_ROOT, "rb_data", "outputs", "train_rb.csv")
        df = pd.read_csv(train_path)
    df["tier_ord"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ord"])
    df["is_hit"] = (df["tier_ord"] >= 3).astype(int)
    df["dc_jj"] = df["pick"].apply(dc_jj_norm)
    df["dc_log"] = df["pick"].apply(dc_log)
    df["dc_sqrt"] = df["pick"].apply(dc_sqrt)
    return df


def main():
    rb = load_position("RB")
    wr = load_position("WR")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Draft Capital Curve Analysis: Log vs Jimmy Johnson vs Sqrt",
                 fontsize=14, fontweight="bold", y=0.98)

    COLORS = {"JJ": "#1f77b4", "Log": "#2ca02c", "Sqrt": "#d62728"}

    # ==============================================
    # Panel 1: Curve shapes
    # ==============================================
    ax = axes[0, 0]
    picks = np.arange(1, 261)
    jj_vals = [dc_jj_norm(p) for p in picks]
    log_vals = [dc_log(p) for p in picks]
    sqrt_vals = [dc_sqrt(p) for p in picks]

    ax.plot(picks, jj_vals, color=COLORS["JJ"], linewidth=2.5, label="Jimmy Johnson (norm)")
    ax.plot(picks, log_vals, color=COLORS["Log"], linewidth=2.5, label="Log")
    ax.plot(picks, sqrt_vals, color=COLORS["Sqrt"], linewidth=2.5, label="Sqrt (current)", linestyle="--")

    # Round demarcations
    for rd_pick, rd_label in [(32, "R1|R2"), (64, "R2|R3"), (100, "R3|R4")]:
        ax.axvline(rd_pick, color="gray", linestyle=":", alpha=0.4)
        ax.text(rd_pick + 2, 9.5, rd_label, fontsize=7, color="gray")

    ax.set_xlabel("Pick Number", fontsize=10)
    ax.set_ylabel("Draft Capital Score (0-10)", fontsize=10)
    ax.set_title("Curve Shapes", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(-0.5, 10.5)

    # ==============================================
    # Panel 2: R² comparison (grouped bars, RB + WR)
    # ==============================================
    ax = axes[0, 1]

    metrics = {}
    for pos, df in [("RB", rb), ("WR", wr)]:
        metrics[pos] = {}
        for name, col in [("JJ", "dc_jj"), ("Log", "dc_log"), ("Sqrt", "dc_sqrt")]:
            x = df[col].values
            r2_dv = np.corrcoef(x, df["dynasty_value"].values)[0, 1] ** 2
            r2_tier = np.corrcoef(x, df["tier_ord"].values)[0, 1] ** 2
            metrics[pos][name] = {"r2_dv": r2_dv, "r2_tier": r2_tier}

    x_pos = np.arange(3)  # JJ, Log, Sqrt
    width = 0.18
    curve_names = ["JJ", "Log", "Sqrt"]

    for i, (metric_key, metric_label) in enumerate([("r2_dv", "R² (dynasty value)"), ("r2_tier", "R² (tier ordinal)")]):
        for j, pos in enumerate(["RB", "WR"]):
            vals = [metrics[pos][c][metric_key] for c in curve_names]
            offset = (i * 2 + j - 1.5) * width
            color = COLORS[["JJ", "Log", "Sqrt"][0]]  # base color
            alpha = 0.9 if pos == "RB" else 0.5
            hatch = "" if pos == "RB" else "///"
            bars = ax.bar(x_pos + offset, vals, width, alpha=alpha, hatch=hatch,
                         color=[COLORS[c] for c in curve_names],
                         edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                       f"{val:.3f}", ha="center", fontsize=6, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_names, fontsize=10)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("R² with Dynasty Value & Tier Ordinal", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    # Manual legend
    rb_patch = mpatches.Patch(facecolor="gray", alpha=0.9, label="RB (solid)")
    wr_patch = mpatches.Patch(facecolor="gray", alpha=0.5, hatch="///", label="WR (hatched)")
    ax.legend(handles=[rb_patch, wr_patch], fontsize=8, loc="upper right",
             title="Left pair = R²(dv), Right pair = R²(tier)", title_fontsize=7)

    # ==============================================
    # Panel 3: Round-level value distribution vs actual
    # ==============================================
    ax = axes[1, 0]

    rounds = [1, 2, 3, "4+"]
    x_rd = np.arange(len(rounds))
    width_rd = 0.18

    # Compute shares for RB
    total_dv = rb["dynasty_value"].sum()
    actual_shares = []
    curve_shares = {name: [] for name in ["JJ", "Log", "Sqrt"]}
    for rd in [1, 2, 3, "4+"]:
        mask = rb["round"] == rd if isinstance(rd, int) else rb["round"] >= 4
        actual_shares.append(rb.loc[mask, "dynasty_value"].sum() / total_dv * 100)
        for name, col in [("JJ", "dc_jj"), ("Log", "dc_log"), ("Sqrt", "dc_sqrt")]:
            total_col = rb[col].sum()
            curve_shares[name].append(rb.loc[mask, col].sum() / total_col * 100)

    # Plot
    ax.bar(x_rd - 1.5 * width_rd, actual_shares, width_rd, color="#333333", alpha=0.9, label="Actual DV", edgecolor="white")
    for i, (name, shares) in enumerate(curve_shares.items()):
        ax.bar(x_rd + (i - 0.5) * width_rd, shares, width_rd,
               color=COLORS[name], alpha=0.8, label=name, edgecolor="white")

    # Value labels
    for i, val in enumerate(actual_shares):
        ax.text(x_rd[i] - 1.5 * width_rd, val + 1, f"{val:.0f}%", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x_rd)
    ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=10)
    ax.set_ylabel("% of Total Value", fontsize=10)
    ax.set_title("RB Value Distribution by Round (Curve vs Actual)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # ==============================================
    # Panel 4: Residual by round (RB)
    # ==============================================
    ax = axes[1, 1]

    from numpy.polynomial.polynomial import polyfit

    round_labels = ["R1", "R2", "R3", "R4+"]
    round_masks = [rb["round"] == 1, rb["round"] == 2, rb["round"] == 3, rb["round"] >= 4]
    x_rd = np.arange(len(round_labels))
    width_rd = 0.25

    for i, (name, col) in enumerate([("JJ", "dc_jj"), ("Log", "dc_log")]):
        coefs = polyfit(rb[col].values, rb["tier_ord"].values, 1)
        predicted = coefs[0] + coefs[1] * rb[col].values
        residuals = rb["tier_ord"].values - predicted

        avg_resid = []
        for mask in round_masks:
            avg_resid.append(residuals[mask].mean())

        offset = (i - 0.5) * width_rd
        bars = ax.bar(x_rd + offset, avg_resid, width_rd, color=COLORS[name],
                     alpha=0.8, label=name, edgecolor="white")
        for bar, val in zip(bars, avg_resid):
            y_pos = val + 0.02 if val >= 0 else val - 0.05
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.2f}",
                   ha="center", fontsize=8, fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_rd)
    ax.set_xticklabels(round_labels, fontsize=10)
    ax.set_ylabel("Mean Residual (tier ordinal)", fontsize=10)
    ax.set_title("RB Prediction Residuals by Round (0 = perfect calibration)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.join(PROJECT_ROOT, "rb_data", "charts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "draft_capital_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
