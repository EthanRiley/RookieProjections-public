"""Position-agnostic profile card rendering.

Shared constants, helper functions, and the make_profile() renderer used
by both WR and RB profile generators. Position modules provide config
(features, composites, labels, output paths) and data loading.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


# --- Shared constants ---

TIER_NAMES_DISPLAY = ["Bust", "Flex", "Starter", "Elite", "Stud", "League-\nWinner"]
TIER_NAMES_FLAT = ["Bust", "Flex", "Starter", "Elite", "Stud", "League-Winner"]
TIER_COLORS = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4", "#9467bd"]

COMPONENT_PREFIXES = [
    ("xgb_full", "XGBoost Full"),
    ("bayes_full", "Bayesian Full"),
    ("xgb_college", "XGBoost College"),
    ("bayes_college", "Bayesian College"),
]


# --- Helpers ---

def pctile_color(v):
    """Color for a percentile value."""
    if pd.isna(v):
        return "#cccccc"
    elif v >= 80:
        return "#2ca02c"
    elif v >= 60:
        return "#98df8a"
    elif v >= 40:
        return "#ffbb78"
    elif v >= 20:
        return "#ff7f0e"
    else:
        return "#d62728"


def has_component_probs(player_row, prefix="xgb_full"):
    """Check if the prediction row has component-level probabilities."""
    return f"{prefix}_P(Bust)" in player_row.index


def compute_percentiles(prospect_feats, train_df, feature_list):
    """Compute percentiles for a prospect's features against training data.

    Args:
        prospect_feats: Dict of feature values for the prospect.
        train_df: Training DataFrame for percentile computation.
        feature_list: List of (column_name, display_label) tuples.

    Returns:
        List of (label, raw_value, percentile) tuples.
    """
    pctiles = []
    for feat_col, feat_label in feature_list:
        val = prospect_feats.get(feat_col, np.nan) if isinstance(prospect_feats, dict) else np.nan
        if pd.notna(val) and feat_col in train_df.columns:
            pct = (train_df[feat_col].dropna() < val).mean() * 100
        else:
            pct = np.nan
        pctiles.append((feat_label, val, pct))
    return pctiles


def find_player(name, year, data_dir, prospect_pattern, holdout_file,
                prospect_years=None, retro_file=None):
    """Search prediction files for a player.

    Args:
        name: Player name (partial match OK).
        year: Draft year filter (None to search all).
        data_dir: Base data directory.
        prospect_pattern: Format string for prospect files, e.g. "prospect_predictions_{}.csv".
        holdout_file: Holdout predictions filename.
        prospect_years: Years to search if year is None (default [2024, 2025, 2026]).
        retro_file: Optional retro LOO predictions filename.

    Returns:
        (player_row, class_df, year) or (None, None, None).
    """
    if prospect_years is None:
        prospect_years = [2024, 2025, 2026]

    # Search prospect files
    for yr in ([year] if year else prospect_years):
        path = os.path.join(data_dir, "outputs", prospect_pattern.format(yr))
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        exact = df[df["name"].str.lower() == name.lower()]
        if len(exact) == 1:
            return exact.iloc[0], df, yr
        match = df[df["name"].str.lower().str.contains(name.lower())]
        if len(match) == 1:
            return match.iloc[0], df, yr
        elif len(match) > 1:
            print(f"Multiple matches in {yr}: {match['name'].tolist()}")
            print("Please be more specific.")
            sys.exit(1)

    # Search holdout file
    holdout_path = os.path.join(data_dir, "outputs", holdout_file)
    if os.path.exists(holdout_path):
        df = pd.read_csv(holdout_path)
        if year:
            df = df[df["draft_year"] == year]
        match = df[df["name"].str.lower().str.contains(name.lower())]
        if len(match) == 1:
            row = match.iloc[0]
            yr = int(row["draft_year"])
            class_df = df[df["draft_year"] == yr]
            return row, class_df, yr
        elif len(match) > 1:
            print(f"Multiple matches in holdout: {match['name'].tolist()}")
            print("Please specify --year or be more specific.")
            sys.exit(1)

    # Search retro LOO predictions (WR only)
    if retro_file:
        retro_path = os.path.join(data_dir, "outputs", retro_file)
        if os.path.exists(retro_path):
            df = pd.read_csv(retro_path)
            if year:
                df = df[df["draft_year"] == year]
            match = df[df["name"].str.lower().str.contains(name.lower())]
            if len(match) == 1:
                row = match.iloc[0]
                yr = int(row["draft_year"])
                return row, df, yr
            elif len(match) > 1:
                print(f"Multiple matches in retro LOO: {match['name'].tolist()}")
                print("Please specify --year or be more specific.")
                sys.exit(1)

    return None, None, None


# --- Profile rendering ---

def make_profile(player_row, class_df, year, prospect_feats, train_df, *,
                 features, composite_components, position_label,
                 percentile_label, composite_title, output_dir,
                 college_only=False):
    """Generate a prospect profile card.

    Args:
        player_row: Series from predictions CSV.
        class_df: Full class DataFrame for ranking context.
        year: Draft year (or label like "Lookahead").
        prospect_feats: Dict of raw feature values.
        train_df: Training DataFrame for percentile computation.
        features: List of (column, label) tuples for the feature panel.
        composite_components: Dict mapping composite name to [(col, label), ...].
        position_label: e.g. "WRs" or "RBs".
        percentile_label: e.g. "Percentile vs. Historical WRs (2017-2022)".
        composite_title: e.g. "Catch Composite Breakdown" or "Composite Breakdown".
        output_dir: Directory to save the PNG.
        college_only: If True, adjusts labels for college-only mode.
    """
    name = player_row["name"]
    has_pick = "pick" in player_row.index and pd.notna(player_row.get("pick"))
    pick = int(player_row["pick"]) if has_pick else None

    # Rank in class (full model)
    rank = class_df.sort_values("expected_tier", ascending=False).reset_index(drop=True)
    rank["rank"] = rank.index + 1
    player_rank = rank[rank["name"] == name]["rank"].values[0]
    class_size = len(class_df)

    # College-only rank
    has_college_tier = ("college_expected_tier" in player_row.index
                        and pd.notna(player_row.get("college_expected_tier")))
    if has_college_tier:
        college_rank = class_df.sort_values("college_expected_tier", ascending=False).reset_index(drop=True)
        college_rank["rank"] = college_rank.index + 1
        player_college_rank = college_rank[college_rank["name"] == name]["rank"].values[0]
    else:
        player_college_rank = player_rank

    # Compute percentiles
    pctiles = compute_percentiles(prospect_feats, train_df, features)

    # Tier probabilities
    tier_probs_full = [player_row.get(f"P({t})", 0) for t in TIER_NAMES_FLAT]

    e_full = player_row["expected_tier"]
    e_college = player_row.get("college_expected_tier", None)
    edge = player_row.get("edge", None)

    # Component probabilities
    has_full_components = has_component_probs(player_row, "xgb_full")
    has_college_components = has_component_probs(player_row, "xgb_college")
    component_probs = {}
    if has_full_components:
        for prefix, label in COMPONENT_PREFIXES:
            component_probs[label] = [player_row.get(f"{prefix}_P({t})", 0) for t in TIER_NAMES_FLAT]
    elif has_college_components:
        for prefix, label in [("xgb_college", "XGBoost College"), ("bayes_college", "Bayesian College")]:
            component_probs[label] = [player_row.get(f"{prefix}_P({t})", 0) for t in TIER_NAMES_FLAT]

    # Composite component percentiles
    comp_pctiles = {}
    for comp_name, components in composite_components.items():
        comp_pctiles[comp_name] = compute_percentiles(prospect_feats, train_df, components)

    # --- Plot ---
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 0.7, 1], height_ratios=[1, 1],
                           hspace=0.35, wspace=0.35)

    # --- Header ---
    actual_tier = player_row.get("computed_tier", None)
    actual_label = f"  |  Actual: {actual_tier}" if pd.notna(actual_tier) and actual_tier else ""
    team_label = ""
    if "team" in player_row.index and pd.notna(player_row.get("team")):
        team_label = f"  |  {player_row['team']}"

    pick_label = f"  |  Pick {pick}" if pick else ""
    rank_label = f"Rank #{player_rank}"
    if has_college_tier:
        rank_label += f" (College #{player_college_rank})"

    mode_label = "College-Only" if college_only else f"{year} Draft"

    fig.suptitle(f"{name}{team_label}  |  {mode_label}{pick_label}  |  "
                 f"{rank_label}  |  {class_size} {position_label}{actual_label}",
                 fontsize=16, fontweight="bold", y=0.98)

    # Subtitle
    if e_college is not None and edge is not None and pd.notna(e_college):
        fig.text(0.5, 0.935,
                 f"E[Full]: {e_full:.2f}   |   E[College]: {e_college:.2f}   |   Edge: {edge:+.3f}",
                 ha="center", fontsize=11, color="#444444")
    else:
        fig.text(0.5, 0.935,
                 f"E[Tier]: {e_full:.2f}",
                 ha="center", fontsize=11, color="#444444")

    # --- Panel 1: Percentile bars (top left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    labels = [p[0] for p in pctiles]
    vals = [p[2] for p in pctiles]
    raw_vals = [p[1] for p in pctiles]

    y_pos = np.arange(len(labels))
    colors = [pctile_color(v) for v in vals]

    bars = ax1.barh(y_pos, [v if pd.notna(v) else 0 for v in vals], color=colors,
                    edgecolor="white", height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel(percentile_label)
    ax1.set_title("Feature Percentiles", fontweight="bold", pad=10)
    ax1.invert_yaxis()
    ax1.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    for i, (bar, val, raw) in enumerate(zip(bars, vals, raw_vals)):
        if pd.notna(val) and pd.notna(raw):
            label_text = f"{raw:.2f}  ({val:.0f}%)"
            x_pos = val + 1.5
            ha = "left"
            if val > 75:
                x_pos = val - 1.5
                ha = "right"
            ax1.text(x_pos, i, label_text, va="center", ha=ha, fontsize=9)

    # --- Panel 2: Composite Breakdown (top middle) ---
    ax_comp = fig.add_subplot(gs[0, 1])

    all_comp_labels = []
    all_comp_pcts = []
    all_comp_raws = []
    all_comp_colors = []
    section_dividers = []

    for comp_name in composite_components:
        if comp_name in comp_pctiles:
            if all_comp_labels:
                section_dividers.append(len(all_comp_labels))
            for feat_label, val, pct in comp_pctiles[comp_name]:
                all_comp_labels.append(feat_label)
                all_comp_pcts.append(pct)
                all_comp_raws.append(val)
                all_comp_colors.append(pctile_color(pct))

    if all_comp_labels:
        y_pos_comp = np.arange(len(all_comp_labels))
        comp_bars = ax_comp.barh(y_pos_comp,
                                 [v if pd.notna(v) else 0 for v in all_comp_pcts],
                                 color=all_comp_colors, edgecolor="white", height=0.6)
        ax_comp.set_yticks(y_pos_comp)
        ax_comp.set_yticklabels(all_comp_labels, fontsize=9)
        ax_comp.set_xlim(0, 100)
        ax_comp.set_title(composite_title, fontweight="bold", pad=10)
        ax_comp.invert_yaxis()
        ax_comp.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        for i, (val, raw) in enumerate(zip(all_comp_pcts, all_comp_raws)):
            if pd.notna(val) and pd.notna(raw):
                label_text = f"{raw:.2f} ({val:.0f}%)"
                x_pos = val + 1.5
                ha = "left"
                if val > 70:
                    x_pos = val - 1.5
                    ha = "right"
                ax_comp.text(x_pos, i, label_text, va="center", ha=ha, fontsize=8)

        # Section dividers between composite groups
        if len(section_dividers) > 0:
            for div_idx in section_dividers:
                ax_comp.axhline(div_idx - 0.5, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    else:
        ax_comp.text(0.5, 0.5, "No composite data", ha="center", va="center",
                     transform=ax_comp.transAxes, fontsize=10, color="gray")
        ax_comp.axis("off")

    # --- Panel 3: Tier probability distribution (top right) ---
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(TIER_NAMES_DISPLAY))
    ax2.bar(x, tier_probs_full, 0.6, color=TIER_COLORS, edgecolor="white", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(TIER_NAMES_DISPLAY, fontsize=9)
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, max(tier_probs_full) * 1.25)
    ax2.set_title("Tier Probability Distribution", fontweight="bold", pad=10)

    for i, v in enumerate(tier_probs_full):
        if v > 0.02:
            ax2.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=8)

    # --- Panel 4: Class ranking context (bottom left+middle) ---
    ax3 = fig.add_subplot(gs[1, 0:2])
    top_n = min(15, len(rank))
    top = rank.head(top_n).copy()
    top_names = top["name"].tolist()
    top_e = top["expected_tier"].tolist()

    bar_colors = ["#1f77b4" if n != name else "#d62728" for n in top_names]
    y_pos_rank = np.arange(top_n)
    ax3.barh(y_pos_rank, top_e, color=bar_colors, edgecolor="white", height=0.7)
    ax3.set_yticks(y_pos_rank)
    ax3.set_yticklabels([f"#{i+1} {n}" for i, n in enumerate(top_names)], fontsize=8.5)
    et_label = "Expected Tier (College-Only)" if college_only else "Expected Tier (Full Model)"
    ax3.set_xlabel(et_label)
    class_label = "Underclassman Rankings" if college_only else f"{year} Class Rankings"
    ax3.set_title(f"{class_label} (Top {top_n})", fontweight="bold", pad=10)
    ax3.invert_yaxis()

    for i, v in enumerate(top_e):
        ax3.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    # --- Panel 5: Component model comparison (bottom right) ---
    ax5 = fig.add_subplot(gs[1, 2])

    if component_probs:
        x = np.arange(len(TIER_NAMES_DISPLAY))
        width = 0.35

        if "XGBoost Full" in component_probs:
            xgb_vals = component_probs["XGBoost Full"]
            bayes_vals = component_probs["Bayesian Full"]
            title = "XGBoost vs. Bayesian (Full)"
        else:
            xgb_vals = component_probs.get("XGBoost College", [0]*6)
            bayes_vals = component_probs.get("Bayesian College", [0]*6)
            title = "XGBoost vs. Bayesian (College)"

        ax5.bar(x - width/2, xgb_vals, width, label="XGBoost",
                color="#ff7f0e", edgecolor="white", alpha=0.8)
        ax5.bar(x + width/2, bayes_vals, width, label="Bayesian",
                color="#9467bd", edgecolor="white", alpha=0.8)

        ax5.set_xticks(x)
        ax5.set_xticklabels(TIER_NAMES_DISPLAY, fontsize=8)
        ax5.set_ylabel("Probability")
        all_vals = list(xgb_vals) + list(bayes_vals)
        ax5.set_ylim(0, max(all_vals) * 1.25 if max(all_vals) > 0 else 1)
        ax5.set_title(title, fontweight="bold", pad=10)
        ax5.legend(fontsize=8, loc="upper right")
    else:
        ax5.text(0.5, 0.5, "No component data\n(re-run predictions to generate)",
                 ha="center", va="center", transform=ax5.transAxes,
                 fontsize=10, color="gray")
        ax5.axis("off")

    # Save
    safe_name = name.replace(" ", "_").replace(".", "").replace("'", "")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close()
