#!/usr/bin/env python3
"""
Visualize top 10 features with per-layer rank breakdown for any position.

Two-panel chart:
  Left: Horizontal bar chart of composite rank (lower = better)
  Right: Heatmap of per-layer ranks (Spearman, MI, AUC, Era Stability) + Enet survival

Usage:
  python3 features/rb_feature_top10_viz.py            # defaults to RB
  python3 features/rb_feature_top10_viz.py --position TE
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def clean_feature_name(name):
    return (name.replace("_", " ")
            .replace("career ", "C: ")
            .replace("best2 ", "B2: ")
            .replace("peak2 ", "P2: ")
            .replace("best ", "B: ")
            .replace("peak ", "P: "))


def plot_feature_top10(position, n=10):
    """Generate top-N feature rank visualization for a position."""
    pos_lower = position.lower()
    data_dir = os.path.join(PROJECT_ROOT, f"{pos_lower}_data")
    eval_path = os.path.join(data_dir, "feature_evaluation.csv")

    if not os.path.exists(eval_path):
        print(f"No feature_evaluation.csv found at {eval_path}")
        return

    df = pd.read_csv(eval_path)

    top = df.sort_values("composite_rank").head(n).copy()
    top = top.iloc[::-1]  # Reverse for horizontal bar (best at top)

    # Use layers that exist in the data (skip perm importance if all tied)
    rank_cols = ["spearman_rank", "mutual_info_rank", "auc_rank", "spearman_diff_rank"]
    rank_labels = ["Spearman\n(Layer 1)", "Mutual Info\n(Layer 1)",
                   "AUC\n(Layer 1)", "Era Stability\n(Layer 5)"]

    max_feats = len(df)

    top["display_name"] = top["feature"].apply(clean_feature_name)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1, 1.5]})
    fig.suptitle(f"{position.upper()} Feature Investigation — Top {n} Features by Composite Rank",
                 fontsize=13, fontweight="bold", y=0.98)

    # --- Left panel: Composite rank bar chart ---
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, n))
    bars = ax1.barh(range(n), top["composite_rank"].values, color=colors,
                    edgecolor="white", linewidth=0.5)

    ax1.set_yticks(range(n))
    ax1.set_yticklabels(top["display_name"].values, fontsize=9)
    ax1.set_xlabel("Composite Rank (lower = better)", fontsize=10)
    ax1.set_title("Composite Rank", fontsize=11, fontweight="bold")
    ax1.invert_xaxis()
    ax1.grid(axis="x", alpha=0.2)

    for i, (bar, val) in enumerate(zip(bars, top["composite_rank"].values)):
        ax1.text(val - 0.5, i, f"{val:.1f}", va="center", ha="right",
                fontsize=8, fontweight="bold", color="white")

    for i, (_, row) in enumerate(top.iterrows()):
        ax1.text(ax1.get_xlim()[0] + 1, i,
                f"Sp={row['spearman']:+.3f}  AUC={row['auc']:.3f}",
                va="center", fontsize=7, color="gray")

    # --- Right panel: Per-layer rank heatmap ---
    rank_data = top[rank_cols].values

    im = ax2.imshow(rank_data, aspect="auto", cmap="RdYlGn_r",
                    vmin=1, vmax=max_feats * 0.6)

    ax2.set_yticks(range(n))
    ax2.set_yticklabels(top["display_name"].values, fontsize=9)
    ax2.set_title("Per-Layer Ranks (lower = better)", fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(len(rank_cols)):
            val = rank_data[i, j]
            text_color = "white" if val > max_feats * 0.35 else "black"
            ax2.text(j, i, f"{int(val)}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    # Enet survival as bonus column
    enet_x = len(rank_cols)
    ax2.set_xlim(-0.5, enet_x + 0.5)

    for i, (_, row) in enumerate(top.iterrows()):
        survive = int(row.get("enet_survive_count", 0))
        color = "#2ca02c" if survive >= 2 else ("#ff7f0e" if survive == 1 else "#d62728")
        ax2.text(enet_x, i, f"{survive}/3", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

    all_labels = rank_labels + ["Enet\nSurvival\n(Layer 3)"]
    ax2.set_xticks(range(len(all_labels)))
    ax2.set_xticklabels(all_labels, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax2, shrink=0.6, pad=0.02)
    cbar.set_label(f"Rank (out of ~{max_feats} features)", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.join(data_dir, "charts"), exist_ok=True)
    out_path = os.path.join(data_dir, "charts", f"{pos_lower}_feature_top{n}_ranks.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")

    # Print table
    print(f"\nTop {n} {position.upper()} Features — Per-Layer Ranks:")
    print(f"{'Feature':<40} {'Composite':>9} {'Spearman':>8} {'MI':>5} {'AUC':>5} {'Era':>5} {'Enet':>5}")
    print("-" * 85)
    for _, row in top.iloc[::-1].iterrows():
        print(f"{row['feature']:<40} {row['composite_rank']:>9.1f} "
              f"{int(row['spearman_rank']):>8} {int(row['mutual_info_rank']):>5} "
              f"{int(row['auc_rank']):>5} {int(row['spearman_diff_rank']):>5} "
              f"{int(row.get('enet_survive_count', 0)):>3}/3")


def main():
    parser = argparse.ArgumentParser(description="Feature top-N rank visualization")
    parser.add_argument("--position", default="RB", help="Position (default: RB)")
    parser.add_argument("--top", type=int, default=10, help="Number of features (default: 10)")
    args = parser.parse_args()

    plot_feature_top10(args.position, n=args.top)


if __name__ == "__main__":
    main()
