#!/usr/bin/env python3
"""
Generate top-10 PDF summaries and holdout profile cards.

Produces:
  - wr_data/top10_2024.pdf
  - wr_data/top10_2025.pdf
  - wr_data/top10_2026.pdf
  - wr_data/top10_holdout_2022_2023.pdf
  - viz/profiles/holdout/*.png (top 10 holdout profiles, 2022-2023 only)
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_COLORS_MAP = {
    "Bust": "#d62728",
    "Flex": "#ff7f0e",
    "Starter": "#bcbd22",
    "Elite": "#2ca02c",
    "Stud": "#1f77b4",
    "League-Winner": "#9467bd",
}


def load_prospect(year):
    path = os.path.join(DATA_DIR, "outputs", f"prospect_predictions_{year}.csv")
    df = pd.read_csv(path)
    df = df.sort_values("expected_tier", ascending=False).reset_index(drop=True)
    return df


def load_holdout_2022_2023():
    path = os.path.join(DATA_DIR, "outputs", "holdout_predictions_v2.csv")
    df = pd.read_csv(path)
    df = df[df["draft_year"].isin([2022, 2023])].copy()
    df = df.sort_values("expected_tier", ascending=False).reset_index(drop=True)
    return df


def make_top10_pdf(df, title, out_path, show_actual=False):
    """Create a clean PDF table of top 10 prospects."""
    top = df.head(10).copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)

    # Build table data
    if show_actual:
        col_labels = ["Rank", "Name", "Year", "Pick", "Actual", "E[full]",
                       "P(Elite+)", "P(Stud+)", "P(LW)", "Edge"]
    else:
        col_labels = ["Rank", "Name", "Pick", "E[full]",
                       "P(Elite+)", "P(Stud+)", "P(LW)", "Edge"]

    rows = []
    cell_colors = []
    for i, (_, row) in enumerate(top.iterrows()):
        p_elite_plus = row["P(Elite)"] + row["P(Stud)"] + row["P(League-Winner)"]
        p_stud_plus = row["P(Stud)"] + row["P(League-Winner)"]
        p_lw = row["P(League-Winner)"]
        edge = row["edge"]

        if show_actual:
            actual = row.get("computed_tier", "")
            rows.append([
                f"WR{i+1}",
                row["name"],
                str(int(row["draft_year"])),
                str(int(row["pick"])),
                actual,
                f"{row['expected_tier']:.2f}",
                f"{p_elite_plus:.1%}",
                f"{p_stud_plus:.1%}",
                f"{p_lw:.1%}",
                f"{edge:+.2f}",
            ])
        else:
            rows.append([
                f"WR{i+1}",
                row["name"],
                str(int(row["pick"])),
                f"{row['expected_tier']:.2f}",
                f"{p_elite_plus:.1%}",
                f"{p_stud_plus:.1%}",
                f"{p_lw:.1%}",
                f"{edge:+.2f}",
            ])

        # Row colors - alternate light gray / white
        base = "#f7f7f7" if i % 2 == 0 else "#ffffff"
        row_colors = [base] * len(col_labels)

        # Color the actual tier cell if shown
        if show_actual:
            tier = row.get("computed_tier", "")
            if tier in TIER_COLORS_MAP:
                row_colors[4] = TIER_COLORS_MAP[tier] + "40"  # alpha via hex

        cell_colors.append(row_colors)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#4472C4"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_fontsize(11)

    # Name column left-aligned
    name_col = 1
    for i in range(len(rows)):
        table[i + 1, name_col].set_text_props(ha="left")
        table[i + 1, name_col]._loc = "left"

    # Adjust column widths
    if show_actual:
        widths = [0.06, 0.22, 0.06, 0.06, 0.12, 0.08, 0.1, 0.1, 0.08, 0.08]
    else:
        widths = [0.07, 0.28, 0.07, 0.09, 0.12, 0.12, 0.09, 0.09]
    for j, w in enumerate(widths):
        for i in range(len(rows) + 1):
            table[i, j].set_width(w)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


def generate_holdout_profiles(df, top_n=10):
    """Generate profile PNGs for top N holdout players (2022-2023)."""
    from aggregation.aggregate_college_stats import (
        load_all_grades, aggregate_player, build_lookups, fit_adot_regression,
    )
    from viz.prospect_profile import make_profile, load_training_features

    top = df.head(top_n)
    train_df = load_training_features()

    all_grades = load_all_grades(range(2016, 2027))
    birth_lookup, draft_age_lookup, team_att_lookup, team_games_lookup = build_lookups(all_grades)
    adot_coef = fit_adot_regression(all_grades)

    import nfl_data_py as nfl
    draft_2022 = nfl.import_draft_picks([2022])
    draft_2023 = nfl.import_draft_picks([2023])
    drafts = {2022: draft_2022, 2023: draft_2023}

    print(f"\n{'='*60}")
    print(f"  Generating top {top_n} holdout profiles (2022-2023)")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(top.iterrows()):
        name = row["name"]
        year = int(row["draft_year"])
        print(f"\n  [{i+1}/{top_n}] {name} ({year}, pick {int(row['pick'])})")

        result = aggregate_player(
            all_grades, name, year,
            birth_lookup=birth_lookup,
            team_att_lookup=team_att_lookup,
            draft_age_lookup=draft_age_lookup,
            adot_coef=adot_coef,
            team_games_lookup=team_games_lookup,
        )

        if not result:
            base_name = name.rsplit(" ", 1)[0]
            result = aggregate_player(
                all_grades, base_name, year,
                birth_lookup=birth_lookup,
                team_att_lookup=team_att_lookup,
                draft_age_lookup=draft_age_lookup,
                adot_coef=adot_coef,
                team_games_lookup=team_games_lookup,
            )

        if not result:
            print(f"    Skipped -- could not aggregate college stats")
            continue

        draft = drafts.get(year)
        if draft is not None:
            wr = draft[draft["pfr_player_name"] == name]
            if len(wr) > 0:
                pick = wr.iloc[0]["pick"]
                result["draft_capital"] = round(10 - 7 * np.sqrt(pick / 260), 2)
                result["pick"] = pick

        # Use the full holdout df as class context so rankings are across 2022-2023
        make_profile(row, df, f"2022-2023", result, train_df)


if __name__ == "__main__":
    # Generate PDFs
    for year in [2024, 2025, 2026]:
        df = load_prospect(year)
        make_top10_pdf(df, f"WR Dynasty Model v9 — {year} Top 10",
                       os.path.join(DATA_DIR, "pdfs", f"top10_{year}.pdf"))

    holdout = load_holdout_2022_2023()
    make_top10_pdf(holdout, "WR Dynasty Model v9 — Holdout Top 10 (2022-2023)",
                   os.path.join(DATA_DIR, "pdfs", "top10_holdout_2022_2023.pdf"),
                   show_actual=True)

    # Generate holdout profiles
    generate_holdout_profiles(holdout, top_n=10)
