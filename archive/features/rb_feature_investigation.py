#!/usr/bin/env python3
"""
RB Feature Investigation — 5-Layer Validation Pipeline.

Same methodology as WR feature validation:
  Layer 1: Univariate screens (Spearman, MI, AUC)
  Layer 2: Visual inspection (scatter plots)
  Layer 3: Elastic Net ordinal regression at multiple regularization strengths
  Layer 4: XGBoost permutation importance
  Layer 5: Era stability (early vs late draft classes)

Reads:  rb_data/outputs/train_rb.csv (resolved players only)
Outputs:
  - rb_data/feature_evaluation.csv (combined feature table)
  - rb_data/charts/rb_feature_scatter.png (Layer 2 viz)
  - Prints full ranked feature table

Usage:
  python3 features/rb_feature_investigation.py
  python3 features/rb_feature_investigation.py --top 20
  python3 features/rb_feature_investigation.py --layer 1
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rb_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

BINARY_THRESHOLD = 3  # Elite or better


def load_data():
    """Load train data (resolved players only)."""
    train_path = os.path.join(DATA_DIR, "outputs", "train_rb.csv")
    if not os.path.exists(train_path):
        # Fall back to full dataset, filter to resolved
        full_path = os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv")
        df = pd.read_csv(full_path)
        df = df[df["is_resolved"] == True].copy()
    else:
        df = pd.read_csv(train_path)

    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df = df.dropna(subset=["tier_ordinal"])
    df["is_hit"] = (df["tier_ordinal"] >= BINARY_THRESHOLD).astype(int)
    return df


def get_feature_cols(df):
    """Identify all candidate feature columns."""
    prefixes = ("career_", "best_", "best2_", "peak_", "peak2_")
    feature_cols = [
        c for c in df.columns
        if c.startswith(prefixes) and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Add non-prefixed features
    extras = ["draft_capital"]
    for extra in extras:
        if extra in df.columns and extra not in feature_cols:
            feature_cols.append(extra)
    return feature_cols


# ============================================================
# LAYER 1: Univariate Screens
# ============================================================

def layer1_univariate(df, feature_cols):
    """Spearman correlation, mutual information, standalone AUC."""
    print("=" * 70)
    print("LAYER 1: Univariate Screens")
    print("=" * 70)

    results = []
    for col in feature_cols:
        valid = df[[col, "tier_ordinal", "is_hit"]].dropna()
        if len(valid) < 20:
            continue

        x = valid[col].values
        y_ord = valid["tier_ordinal"].values
        y_bin = valid["is_hit"].values

        # Spearman rank correlation
        spear_corr, spear_p = spearmanr(x, y_ord)

        # Mutual information
        mi = mutual_info_classif(
            x.reshape(-1, 1), y_ord, discrete_features=False, random_state=42
        )[0]

        # Standalone AUC for Elite+
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            auc = roc_auc_score(y_bin, x)
            if auc < 0.5:
                auc = 1 - auc
        else:
            auc = np.nan

        results.append({
            "feature": col,
            "spearman": round(spear_corr, 3),
            "spearman_p": round(spear_p, 4),
            "mutual_info": round(mi, 4),
            "auc": round(auc, 3) if not np.isnan(auc) else np.nan,
            "n": len(valid),
        })

    results_df = pd.DataFrame(results)

    # Composite ranking
    for metric in ["spearman", "mutual_info", "auc"]:
        vals = results_df[metric].abs() if metric == "spearman" else results_df[metric]
        results_df[f"{metric}_rank"] = vals.rank(ascending=False)

    results_df["composite_rank"] = (
        results_df["spearman_rank"] + results_df["mutual_info_rank"] + results_df["auc_rank"]
    ) / 3
    results_df = results_df.sort_values("composite_rank").reset_index(drop=True)

    print(f"\nPlayers: {len(df)}, Hit rate (Elite+): {df['is_hit'].mean():.1%}")
    print(f"Candidate features: {len(results_df)}\n")
    print("Top 30 features by composite rank:")
    print(results_df[["feature", "spearman", "mutual_info", "auc", "composite_rank", "n"]].head(30).to_string(index=False))

    return results_df


# ============================================================
# LAYER 2: Visual Inspection (scatter plots)
# ============================================================

def layer2_visual(df, top_features, n=12):
    """Scatter plots of top features vs dynasty_value."""
    import matplotlib.pyplot as plt

    features_to_plot = top_features[:n]
    ncols = 4
    nrows = (len(features_to_plot) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    tier_colors = {
        "Bust": "#d62728", "Flex": "#ff7f0e", "Starter": "#bcbd22",
        "Elite": "#2ca02c", "Stud": "#1f77b4", "League-Winner": "#9467bd",
    }

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        valid = df[[feat, "dynasty_value", "computed_tier"]].dropna()
        for tier, color in tier_colors.items():
            mask = valid["computed_tier"] == tier
            if mask.sum() > 0:
                ax.scatter(valid.loc[mask, feat], valid.loc[mask, "dynasty_value"],
                          c=color, alpha=0.6, s=30, label=tier)
        ax.set_xlabel(feat.replace("_", " "), fontsize=8)
        ax.set_ylabel("Dynasty Value")
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.2)

    # Hide extra axes
    for j in range(len(features_to_plot), len(axes)):
        axes[j].set_visible(False)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle("RB Feature Investigation — Top Features vs Dynasty Value", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.join(DATA_DIR, "charts"), exist_ok=True)
    out_path = os.path.join(DATA_DIR, "charts", "rb_feature_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved scatter plots: {out_path}")


# ============================================================
# LAYER 3: Elastic Net
# ============================================================

def layer3_elastic_net(X_scaled, y_bin, all_features):
    """Elastic Net at multiple regularization strengths."""
    print("\n" + "=" * 70)
    print("LAYER 3: Elastic Net Ordinal Regression")
    print("=" * 70)

    alphas = [0.01, 0.1, 1.0]
    layer3_results = {feat: {} for feat in all_features}

    for C_val in alphas:
        model = LogisticRegression(
            penalty="elasticnet", C=C_val, l1_ratio=0.5,
            solver="saga", max_iter=10000, random_state=42,
            class_weight="balanced",
        )
        model.fit(X_scaled, y_bin)
        coefs = model.coef_[0]
        for i, feat in enumerate(all_features):
            layer3_results[feat][f"enet_C{C_val}"] = round(coefs[i], 4)

    layer3_df = pd.DataFrame(layer3_results).T
    layer3_df.index.name = "feature"
    layer3_df = layer3_df.reset_index()

    layer3_df["enet_survive_count"] = sum(
        (layer3_df[f"enet_C{C}"].abs() > 1e-6).astype(int) for C in alphas
    )

    print("\nFeatures surviving at each regularization strength:")
    for C_val in alphas:
        col = f"enet_C{C_val}"
        surviving = layer3_df[layer3_df[col].abs() > 1e-6]
        print(f"\n  C={C_val} ({len(surviving)} features survive):")
        top = surviving.reindex(surviving[col].abs().sort_values(ascending=False).index)
        for _, row in top.head(10).iterrows():
            print(f"    {row['feature']:45s} coef={row[col]:+.4f}")

    return layer3_df


# ============================================================
# LAYER 4: XGBoost Permutation Importance
# ============================================================

def layer4_xgboost(X, y_bin, all_features):
    """XGBoost permutation importance."""
    print("\n" + "=" * 70)
    print("LAYER 4: XGBoost Permutation Importance")
    print("=" * 70)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        scale_pos_weight=(y_bin == 0).sum() / max((y_bin == 1).sum(), 1),
        random_state=42, use_label_encoder=False, eval_metric="logloss",
    )
    xgb.fit(X, y_bin)

    perm_result = permutation_importance(
        xgb, X, y_bin, n_repeats=30, random_state=42, scoring="roc_auc"
    )

    perm_df = pd.DataFrame({
        "feature": all_features,
        "perm_importance_mean": np.round(perm_result.importances_mean, 4),
        "perm_importance_std": np.round(perm_result.importances_std, 4),
    })
    perm_df = perm_df.sort_values("perm_importance_mean", ascending=False)

    print("\nTop 20 features by permutation importance:")
    for _, row in perm_df.head(20).iterrows():
        print(f"  {row['feature']:45s} {row['perm_importance_mean']:+.4f} +/- {row['perm_importance_std']:.4f}")

    return perm_df


# ============================================================
# LAYER 5: Era Stability
# ============================================================

def layer5_era_stability(X, y_ord, y_bin, draft_years, all_features):
    """Spearman and AUC stability across eras."""
    print("\n" + "=" * 70)
    print("LAYER 5: Era Stability (2016-2018 vs 2019-2021)")
    print("=" * 70)

    early_mask = draft_years <= 2018
    late_mask = draft_years >= 2019

    stability_results = []
    for i, feat in enumerate(all_features):
        x_all = X[:, i]
        x_early, y_early, y_bin_early = x_all[early_mask], y_ord[early_mask], y_bin[early_mask]
        x_late, y_late, y_bin_late = x_all[late_mask], y_ord[late_mask], y_bin[late_mask]

        sp_early, _ = spearmanr(x_early, y_early) if len(x_early) > 10 else (np.nan, np.nan)
        sp_late, _ = spearmanr(x_late, y_late) if len(x_late) > 10 else (np.nan, np.nan)

        def safe_auc(y_true, x_score):
            if y_true.sum() == 0 or y_true.sum() == len(y_true) or len(y_true) < 10:
                return np.nan
            auc = roc_auc_score(y_true, x_score)
            return auc if auc >= 0.5 else 1 - auc

        sp_diff = abs(sp_early - sp_late) if not (np.isnan(sp_early) or np.isnan(sp_late)) else np.nan

        stability_results.append({
            "feature": feat,
            "spearman_early": round(sp_early, 3) if not np.isnan(sp_early) else np.nan,
            "spearman_late": round(sp_late, 3) if not np.isnan(sp_late) else np.nan,
            "spearman_diff": round(sp_diff, 3) if not np.isnan(sp_diff) else np.nan,
        })

    stability_df = pd.DataFrame(stability_results)

    print(f"\nEarly era (<=2018): {early_mask.sum()} players, {y_bin[early_mask].sum()} hits")
    print(f"Late era  (>=2019): {late_mask.sum()} players, {y_bin[late_mask].sum()} hits")

    print("\nMost STABLE features (smallest Spearman difference across eras):")
    stable = stability_df.dropna(subset=["spearman_diff"]).sort_values("spearman_diff")
    for _, row in stable.head(15).iterrows():
        print(f"  {row['feature']:45s} early={row['spearman_early']:+.3f}  late={row['spearman_late']:+.3f}  diff={row['spearman_diff']:.3f}")

    return stability_df


# ============================================================
# Combined Table
# ============================================================

def build_combined_table(layer1_df, layer3_df, perm_df, stability_df):
    """Merge all layers into a single ranked feature table."""
    print("\n" + "=" * 70)
    print("COMBINED FEATURE EVALUATION TABLE")
    print("=" * 70)

    combined = layer1_df[["feature", "spearman", "mutual_info", "auc"]].copy()

    # Merge Layer 3
    alphas = [0.01, 0.1, 1.0]
    enet_cols = ["feature", "enet_survive_count"] + [f"enet_C{C}" for C in alphas]
    combined = combined.merge(layer3_df[enet_cols], on="feature", how="left")

    # Merge Layer 4
    combined = combined.merge(perm_df[["feature", "perm_importance_mean"]], on="feature", how="left")

    # Merge Layer 5
    combined = combined.merge(stability_df[["feature", "spearman_diff", "spearman_early", "spearman_late"]], on="feature", how="left")

    # Fill NaN for features that couldn't be included in layers 3-5 (due to missingness)
    combined["perm_importance_mean"] = combined["perm_importance_mean"].fillna(0)
    combined["enet_survive_count"] = combined["enet_survive_count"].fillna(0)
    combined["spearman_diff"] = combined["spearman_diff"].fillna(1.0)

    # Composite ranking across all layers (excluding Layer 4 — too few complete cases for XGB)
    for metric, ascending in [
        ("spearman", True), ("mutual_info", True), ("auc", True),
        ("perm_importance_mean", True), ("spearman_diff", False),
    ]:
        vals = combined[metric].abs() if metric == "spearman" else combined[metric]
        if ascending:
            combined[f"{metric}_rank"] = vals.rank(ascending=False)
        else:
            combined[f"{metric}_rank"] = vals.rank(ascending=True)

    combined["composite_rank"] = (
        combined["spearman_rank"] + combined["mutual_info_rank"]
        + combined["auc_rank"] + combined["spearman_diff_rank"]
    ) / 4

    combined = combined.sort_values("composite_rank").reset_index(drop=True)

    print_cols = [
        "feature", "spearman", "mutual_info", "auc",
        "enet_survive_count", "perm_importance_mean", "spearman_diff",
        "composite_rank",
    ]
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print("\n" + combined[print_cols].head(40).to_string(index=False))

    return combined


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RB Feature Investigation (5-Layer Pipeline)")
    parser.add_argument("--top", type=int, default=30, help="Number of top features to display")
    parser.add_argument("--layer", type=int, default=None, help="Run only a specific layer (1-5)")
    parser.add_argument("--no-viz", action="store_true", help="Skip Layer 2 visualizations")
    args = parser.parse_args()

    df = load_data()
    feature_cols = get_feature_cols(df)

    print(f"\nRB Feature Investigation")
    print(f"Players: {len(df)} (resolved, train set)")
    print(f"Hit rate (Elite+): {df['is_hit'].mean():.1%}")
    print(f"Tier distribution:")
    for tier in ["League-Winner", "Stud", "Elite", "Starter", "Flex", "Bust"]:
        count = (df["computed_tier"] == tier).sum()
        print(f"  {tier}: {count}")
    print(f"Candidate features: {len(feature_cols)}\n")

    # Layer 1
    if args.layer is None or args.layer == 1:
        layer1_df = layer1_univariate(df, feature_cols)

    # For layers 3-5, need complete-case matrix
    # Use top features from Layer 1 to avoid too much missingness
    if args.layer is None or args.layer >= 3:
        # Get features with enough data
        usable_features = []
        for col in feature_cols:
            if df[col].notna().sum() >= len(df) * 0.7:  # At least 70% non-null
                usable_features.append(col)

        df_complete = df[usable_features + ["tier_ordinal", "is_hit", "draft_year", "name"]].dropna().copy()
        print(f"\nComplete cases for Layers 3-5: {len(df_complete)} players, {len(usable_features)} features")

        X = df_complete[usable_features].values
        y_ord = df_complete["tier_ordinal"].values.astype(int)
        y_bin = df_complete["is_hit"].values.astype(int)
        draft_years = df_complete["draft_year"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # Layer 2
    if (args.layer is None or args.layer == 2) and not args.no_viz:
        top_feats = layer1_df["feature"].head(12).tolist()
        layer2_visual(df, top_feats)

    # Layer 3
    if args.layer is None or args.layer == 3:
        layer3_df = layer3_elastic_net(X_scaled, y_bin, usable_features)

    # Layer 4
    if args.layer is None or args.layer == 4:
        perm_df = layer4_xgboost(X, y_bin, usable_features)

    # Layer 5
    if args.layer is None or args.layer == 5:
        stability_df = layer5_era_stability(X, y_ord, y_bin, draft_years, usable_features)

    # Combined table
    if args.layer is None:
        # Recompute layer1 on usable_features for consistency
        layer1_usable = layer1_univariate(df, usable_features)
        combined = build_combined_table(layer1_usable, layer3_df, perm_df, stability_df)

        # Save
        out_path = os.path.join(DATA_DIR, "feature_evaluation.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
