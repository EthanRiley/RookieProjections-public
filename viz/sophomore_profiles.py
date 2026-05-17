#!/usr/bin/env python3
"""
Generate profile cards for top 2027 draft class WR prospects (2026 lookahead).

Includes players who first appeared in PFF data in 2024 or 2025 (sophomores
and freshmen who will be juniors/sophomores in 2026). Since they have no draft
capital or model predictions, profiles show feature percentiles, season stats,
class composite ranking, and alignment.

Usage:
    python viz/sophomore_profiles.py              # top 15
    python viz/sophomore_profiles.py --top 20     # top 20
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

from aggregation.aggregate_college_stats import (
    load_all_grades, build_lookups, fit_adot_regression,
    compute_pg_yprr_graduated, compute_pg_catch_pct_adot_adj_graduated,
)

# Feature definitions for percentile bars
FEATURES = [
    ("adj_yprr", "YPRR (age-adjusted)"),
    ("catch_composite", "Catch Composite"),
    ("contested_catch_rate", "Best 2 CCR"),
    ("avoided_tackles_per_rec", "Best 2 MTF / Rec"),
]


def pctile_color(pct):
    if pd.isna(pct):
        return "#cccccc"
    if pct >= 80:
        return "#2ca02c"
    if pct >= 60:
        return "#98df8a"
    if pct >= 40:
        return "#ffbb78"
    if pct >= 20:
        return "#ff7f0e"
    return "#d62728"


def make_sophomore_profile(player, rank, class_df, hist, out_dir):
    """Generate a single prospect profile card."""
    name = player["player"]
    team = player["team_name"]
    is_p5 = player["is_p5"]
    composite = player["college_composite"]
    class_label = player.get("class_label", "??")
    n_seasons = int(player.get("n_seasons", 1))

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 0.8, 1.0], height_ratios=[1, 1],
                           hspace=0.35, wspace=0.35)

    # --- Header ---
    p5_tag = "P5" if is_p5 else "G5"
    qg_tag = " | Quality-Gated" if player.get("quality_gated", False) else ""
    fig.suptitle(
        f"{name}  |  {team} ({p5_tag})  |  {class_label}  |  "
        f"Composite Rank #{rank}{qg_tag}",
        fontsize=16, fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.945, "2027 Draft Class Lookahead — Projected Draft Prospect",
             ha="center", fontsize=11, fontstyle="italic", color="#555555")

    # --- Panel 1: Percentile bars (top left) ---
    ax1 = fig.add_subplot(gs[0, 0])

    feat_labels = []
    feat_pcts = []
    feat_raws = []
    for feat_col, feat_label in FEATURES:
        val = player.get(feat_col, np.nan)
        hist_series = hist.get(feat_col)
        if hist_series is not None and pd.notna(val) and len(hist_series) > 0:
            pct = (hist_series < val).mean() * 100
        else:
            pct = np.nan
        feat_labels.append(feat_label)
        feat_pcts.append(pct)
        feat_raws.append(val)

    y_pos = np.arange(len(feat_labels))
    colors = [pctile_color(p) for p in feat_pcts]

    bars = ax1.barh(y_pos, [v if pd.notna(v) else 0 for v in feat_pcts],
                    color=colors, edgecolor="white", height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feat_labels)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Percentile vs. Historical Drafted WRs (2018-2024)")
    ax1.set_title("Feature Percentiles", fontweight="bold", pad=10)
    ax1.invert_yaxis()
    ax1.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    for i, (bar, pct, raw) in enumerate(zip(bars, feat_pcts, feat_raws)):
        if pd.notna(pct) and pd.notna(raw):
            label_text = f"{raw:.2f}  ({pct:.0f}%)"
            x = pct + 1.5
            ha = "left"
            if pct > 72:
                x = pct - 1.5
                ha = "right"
            ax1.text(x, i, label_text, va="center", ha=ha, fontsize=9)

    # --- Panel 2: Stats summary (top middle) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    # Determine age adjustment label
    if "FR" in class_label:
        adj_label = "Adj YPRR (FR +25%)"
    elif "SO" in class_label:
        adj_label = "Adj YPRR (SO +5%)"
    else:
        adj_label = "Adj YPRR"

    stats = [
        ("Seasons", f"{n_seasons}"),
        ("Games", f"{int(player['total_games'])}"),
        ("Total Routes", f"{int(player['total_routes'])}"),
        ("", ""),
        ("Best PFF Grade", f"{player['best_grades_offense']:.1f}"),
        ("Best Route Grade", f"{player['best_grades_pass_route']:.1f}"),
        ("", ""),
        ("Raw YPRR (best)", f"{player['raw_yprr']:.2f}"),
        (adj_label, f"{player['adj_yprr']:.2f}"),
        ("", ""),
        ("Best Catch %", f"{player['caught_percent']:.1f}%"),
        ("Best aDOT", f"{player['avg_depth_of_target']:.1f}"),
        ("", ""),
        ("Best 2 CCR", f"{player['contested_catch_rate']:.1f}%"),
        ("Total Contested Tgt", f"{int(player['contested_targets'])}"),
        ("", ""),
        ("Yards / Game", f"{player['yards_pg']:.1f}"),
        ("Rec / Game", f"{player['receptions_pg']:.1f}"),
        ("TD / Game", f"{player['touchdowns_pg']:.2f}"),
    ]

    y_start = 0.98
    for i, (label, value) in enumerate(stats):
        y = y_start - i * 0.052
        if label == "":
            continue
        ax2.text(0.05, y, label, fontsize=9.5, fontweight="bold",
                 transform=ax2.transAxes, va="top")
        ax2.text(0.95, y, value, fontsize=9.5, ha="right",
                 transform=ax2.transAxes, va="top")

    ax2.set_title("Career Stats", fontweight="bold", pad=10)

    # --- Panel 3: Z-score bar (top right) ---
    ax3 = fig.add_subplot(gs[0, 2])

    z_features = [
        ("z_yprr", "YPRR"),
        ("z_catch_composite", "Catch Comp"),
        ("z_ccr", "CCR"),
        ("z_at", "AT/R"),
    ]
    z_labels = [f[1] for f in z_features]
    z_vals = [player.get(f[0], 0) for f in z_features]

    z_colors = ["#2ca02c" if v >= 0.5 else "#ff7f0e" if v >= 0 else "#d62728" for v in z_vals]
    y_z = np.arange(len(z_labels))
    ax3.barh(y_z, z_vals, color=z_colors, edgecolor="white", height=0.55)
    ax3.set_yticks(y_z)
    ax3.set_yticklabels(z_labels)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("Z-Score vs. Historical Drafted WRs")
    ax3.set_title("Feature Z-Scores", fontweight="bold", pad=10)
    ax3.invert_yaxis()

    for i, v in enumerate(z_vals):
        if pd.notna(v):
            offset = 0.05 if v >= 0 else -0.05
            ha = "left" if v >= 0 else "right"
            ax3.text(v + offset, i, f"{v:+.2f}", va="center", ha=ha, fontsize=9)

    ax3.text(0.95, 0.02, f"Composite: {composite:.2f}",
             transform=ax3.transAxes, fontsize=12, fontweight="bold",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8", alpha=0.8))

    # --- Panel 4: Class ranking (bottom left + middle) ---
    ax4 = fig.add_subplot(gs[1, 0:2])

    top_n_rank = min(15, len(class_df))
    top_rank = class_df.head(top_n_rank).copy()
    top_names = top_rank["player"].tolist()
    top_comp = top_rank["college_composite"].tolist()
    top_teams = top_rank["team_name"].tolist()

    bar_colors = ["#1f77b4" if n != name else "#d62728" for n in top_names]
    y_rank = np.arange(top_n_rank)

    ax4.barh(y_rank, top_comp, color=bar_colors, edgecolor="white", height=0.7)
    ax4.set_yticks(y_rank)
    ax4.set_yticklabels([f"#{i+1} {n} ({t})" for i, (n, t) in
                         enumerate(zip(top_names, top_teams))], fontsize=8.5)
    ax4.set_xlabel("College Composite Score")
    ax4.set_title("2027 Class Rankings (Top 15)", fontweight="bold", pad=10)
    ax4.invert_yaxis()

    for i, v in enumerate(top_comp):
        ax4.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    # --- Panel 5: Notes + small alignment (bottom right) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    # Notes
    notes = []
    if player.get("quality_gated", False):
        notes.append("Quality-gated (grade >= 80) in at least one season")
    ct = int(player.get("contested_targets", 0))
    if ct < 10:
        notes.append(f"Small CCR sample ({ct} CT) — imputed to p50")
    if not is_p5:
        notes.append("Non-P5 school — production context matters")
    if n_seasons == 1:
        notes.append("Single-season profile — will evolve with more data")
    elif n_seasons == 2:
        notes.append("2-season profile — best2 aggregation applied")
    notes.append(f"Age adj: {'FR +25%' if 'FR' in class_label else 'SO +5%' if 'SO' in class_label else '??'}")

    ax5.text(0.05, 0.95, "Notes", fontsize=11, fontweight="bold",
             transform=ax5.transAxes, va="top")
    for i, note in enumerate(notes):
        ax5.text(0.05, 0.85 - i * 0.1, f"• {note}", fontsize=9,
                 transform=ax5.transAxes, va="top", color="#444444")

    # Small alignment summary (text-based, no pie chart)
    slot = player.get("slot_rate", 0) or 0
    wide = player.get("wide_rate", 0) or 0
    other = max(0, 100 - slot - wide) if (slot + wide) <= 100 else 0

    y_align = 0.85 - len(notes) * 0.1 - 0.05
    ax5.text(0.05, y_align, "Alignment", fontsize=11, fontweight="bold",
             transform=ax5.transAxes, va="top")
    ax5.text(0.05, y_align - 0.1,
             f"Wide: {wide:.0f}%   |   Slot: {slot:.0f}%   |   Other: {other:.0f}%",
             fontsize=10, transform=ax5.transAxes, va="top", color="#333333")

    # Save PNG
    safe_name = name.replace(" ", "_").replace(".", "").replace("'", "")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {out_path}")
    return out_path


def get_player_seasons(all_grades, name, draft_year, birthdate=None):
    """Wrapper for import."""
    from aggregation.aggregate_college_stats import get_player_seasons as _gps
    return _gps(all_grades, name, draft_year, birthdate=birthdate)


def run(top_n=15):
    """Generate profiles and stitch into PDF."""
    lookahead_path = os.path.join(DATA_DIR, "outputs", "sophomore_2026_lookahead.csv")
    if not os.path.exists(lookahead_path):
        print("Run modeling/research/sophomore_2026_lookahead.py first to generate the data.")
        sys.exit(1)

    df = pd.read_csv(lookahead_path)
    df = df.sort_values("college_composite", ascending=False).reset_index(drop=True)

    # Build historical distributions for percentiles
    print("Building historical distributions...")
    all_grades = load_all_grades(range(2016, 2026))
    birth_lookup, _, _, _ = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)
    master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))

    hist_pg_yprr = []
    hist_pg_cpaa = []
    for _, row in master.iterrows():
        name = row["name"]
        dy = int(row["draft_year"])
        birthdate = birth_lookup.get((name, dy))
        seasons = get_player_seasons(all_grades, name, dy, birthdate=birthdate)
        if len(seasons) == 0:
            continue
        val = compute_pg_yprr_graduated(seasons, birthdate)
        if not np.isnan(val):
            hist_pg_yprr.append(val)
        val2 = compute_pg_catch_pct_adot_adj_graduated(seasons, birthdate, adot_coef=adot_coef)
        if not np.isnan(val2):
            hist_pg_cpaa.append(val2)

    # Build historical catch_composite using production formula
    CATCH_COMPOSITE_CPAA_WEIGHT = 0.67
    CATCH_COMPOSITE_CAREER_WEIGHT = 0.33
    cpaa_mean = pd.Series(hist_pg_cpaa).mean()
    cpaa_std = pd.Series(hist_pg_cpaa).std()
    hist_career_cpaa = master["career_catch_pct_adot_adj"].dropna()
    career_cpaa_mean = hist_career_cpaa.mean()
    career_cpaa_std = hist_career_cpaa.std()

    hist_catch_composite = []
    for _, row in master.dropna(subset=["career_catch_pct_adot_adj"]).iterrows():
        name = row["name"]
        dy = int(row["draft_year"])
        birthdate = birth_lookup.get((name, dy))
        seasons = get_player_seasons(all_grades, name, dy, birthdate=birthdate)
        if len(seasons) == 0:
            continue
        pg_val = compute_pg_catch_pct_adot_adj_graduated(seasons, birthdate, adot_coef=adot_coef)
        if not np.isnan(pg_val):
            cc = (CATCH_COMPOSITE_CPAA_WEIGHT * (pg_val - cpaa_mean) / cpaa_std
                  + CATCH_COMPOSITE_CAREER_WEIGHT * (row["career_catch_pct_adot_adj"] - career_cpaa_mean) / career_cpaa_std)
            hist_catch_composite.append(cc)

    hist = {
        "adj_yprr": pd.Series(hist_pg_yprr),
        "catch_composite": pd.Series(hist_catch_composite),
        "contested_catch_rate": master["best2_contested_catch_rate"].dropna(),
        "avoided_tackles_per_rec": master["best2_avoided_tackles_per_rec"].dropna(),
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "profiles", "sophomore_2026")

    print(f"\nGenerating top {top_n} prospect profiles...")
    png_paths = []
    top = df.head(top_n)
    for i, (_, player) in enumerate(top.iterrows()):
        rank = i + 1
        print(f"  [{rank}/{top_n}] {player['player']} ({player['team_name']}) — {player.get('class_label', '??')}")
        path = make_sophomore_profile(player, rank, df, hist, out_dir)
        if path:
            png_paths.append(path)

    # Stitch into PDF using PIL to avoid imshow axis flipping
    if png_paths:
        from PIL import Image
        pdf_path = os.path.join(DATA_DIR, "pdfs", "sophomore_2026_lookahead.pdf")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        images = [Image.open(p).convert("RGB") for p in png_paths]
        images[0].save(pdf_path, save_all=True, append_images=images[1:],
                       resolution=150)
        print(f"\nStitched PDF saved to {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    run(top_n=args.top)
