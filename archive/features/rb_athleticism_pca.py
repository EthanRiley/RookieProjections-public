"""
RB Athleticism PCA Analysis

Generalized script to evaluate combine athleticism composites (RAS, PCA, supervised)
against dynasty RB outcomes. Supports both the modern dataset (2016+) and the extended
historical dataset (2000+).

Usage:
    python features/rb_athleticism_pca.py                    # default: historical
    python features/rb_athleticism_pca.py --dataset modern   # 2016+ only
    python features/rb_athleticism_pca.py --save             # save updated CSV + JSON
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Config ---
PCA_FEATURES = ["forty_neg", "vertical", "broad_jump", "speed_score", "wt", "ht"]
RAW_METRICS = ["forty", "vertical", "broad_jump", "speed_score", "wt", "ht", "bench", "cone", "shuttle"]
TIER_ORDER = {"Bust": 0, "Flex": 1, "Starter": 2, "Elite": 3, "Stud": 4, "League-Winner": 5}
HIT_THRESHOLD = 3  # >= Elite


def load_data(dataset="historical"):
    if dataset == "historical":
        path = PROJECT_ROOT / "rb_data" / "outputs" / "rb_historical_athleticism.csv"
        df = pd.read_csv(path)
        # Standardize column names
        df = df.rename(columns={"wt_kg": "wt", "ht_cm": "ht"})
    else:
        path = PROJECT_ROOT / "rb_data" / "outputs" / "rb_combine_data.csv"
        df = pd.read_csv(path)
        df = df.rename(columns={"wt_kg": "wt", "ht_cm": "ht"})

    # Filter to resolved only
    df = df[df["computed_tier"] != "TBD"].copy()
    df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
    df["is_hit"] = (df["tier_ordinal"] >= HIT_THRESHOLD).astype(int)
    df["forty_neg"] = -df["forty"]  # negate so higher = faster
    df["wt"] = df.get("wt", df.get("weight", None))
    df["ht"] = df.get("ht", df.get("height_in", None))

    return df


def univariate_analysis(df):
    """Test each raw combine metric against tier ordinal."""
    results = {}
    for metric in RAW_METRICS:
        subset = df.dropna(subset=[metric])
        if len(subset) < 20:
            continue
        vals = subset[metric].values
        # Negate forty so positive = better
        if metric == "forty":
            vals = -vals
        tiers = subset["tier_ordinal"].values
        hits = subset["is_hit"].values

        r, p = spearmanr(vals, tiers)
        try:
            auc = roc_auc_score(hits, vals)
        except ValueError:
            auc = np.nan

        results[metric] = {"spearman": round(r, 4), "p": round(p, 4), "auc": round(auc, 3), "n": len(subset)}

    return pd.DataFrame(results).T.sort_values("auc", ascending=False)


def compute_ras(df, ras_population=None):
    """Compute RAS as percentile rank across available metrics, scaled 0-10."""
    metrics = ["forty", "vertical", "broad_jump", "speed_score", "wt", "ht", "bench", "cone", "shuttle"]
    negate = {"forty", "cone", "shuttle"}  # lower = better

    if ras_population is None:
        ras_population = df

    percentiles = pd.DataFrame(index=df.index)
    for m in metrics:
        if m not in df.columns or df[m].isna().all():
            continue
        pop_vals = ras_population[m].dropna()
        if len(pop_vals) < 10:
            continue
        if m in negate:
            percentiles[m] = df[m].apply(
                lambda x: (pop_vals >= x).mean() * 10 if pd.notna(x) else np.nan
            )
        else:
            percentiles[m] = df[m].apply(
                lambda x: (pop_vals <= x).mean() * 10 if pd.notna(x) else np.nan
            )

    df["ras"] = percentiles.mean(axis=1)
    return df


def run_pca(df):
    """Run PCA on the 6 core metrics (complete cases only)."""
    complete = df.dropna(subset=PCA_FEATURES).copy()
    X = complete[PCA_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    complete["pc1"] = scores[:, 0]
    complete["pc2"] = scores[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_FEATURES,
        columns=[f"PC{i+1}" for i in range(len(PCA_FEATURES))]
    )

    return complete, pca, scaler, loadings


def supervised_composite(df, features=None):
    """Weight each metric by its Spearman correlation with tier ordinal."""
    if features is None:
        features = PCA_FEATURES
    complete = df.dropna(subset=features).copy()
    tiers = complete["tier_ordinal"].values

    weights = {}
    for f in features:
        r, _ = spearmanr(complete[f].values, tiers)
        weights[f] = max(r, 0)

    total = sum(weights.values())
    if total == 0:
        complete["supervised_score"] = 0
        return complete, weights

    norm_weights = {f: w / total for f, w in weights.items()}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(complete[features].values)
    weight_vec = np.array([norm_weights[f] for f in features])
    complete["supervised_score"] = X_scaled @ weight_vec

    return complete, norm_weights


def supervised_loocv(df, features=None):
    """Leave-one-out cross-validation for supervised composite."""
    if features is None:
        features = PCA_FEATURES
    complete = df.dropna(subset=features).copy()
    tiers = complete["tier_ordinal"].values
    X = complete[features].values
    n = len(complete)
    predictions = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], tiers[mask]

        # Fit scaler on training fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X[i:i+1])

        # Compute weights from training fold
        weights = []
        for j, f in enumerate(features):
            r, _ = spearmanr(X_train_scaled[:, j], y_train)
            weights.append(max(r, 0))
        total = sum(weights)
        if total == 0:
            predictions[i] = 0
            continue
        weight_vec = np.array([w / total for w in weights])
        predictions[i] = (X_test_scaled @ weight_vec)[0]

    complete["supervised_loocv"] = predictions
    return complete


def pca_loocv(df, features=None):
    """Leave-one-out cross-validation for PCA PC1."""
    if features is None:
        features = PCA_FEATURES
    complete = df.dropna(subset=features).copy()
    X = complete[features].values
    n = len(complete)
    predictions = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[mask])
        X_test_scaled = scaler.transform(X[i:i+1])

        pca = PCA(n_components=1)
        pca.fit(X_train_scaled)
        predictions[i] = pca.transform(X_test_scaled)[0, 0]

    complete["pc1_loocv"] = predictions
    return complete


def evaluate(values, tiers, hits, label=""):
    """Compute Spearman, p-value, and AUC for a scoring method."""
    valid = ~np.isnan(values)
    values, tiers, hits = values[valid], tiers[valid], hits[valid]
    r, p = spearmanr(values, tiers)
    try:
        auc = roc_auc_score(hits, values)
    except ValueError:
        auc = np.nan
    return {"method": label, "spearman": round(r, 4), "p": round(p, 4), "auc": round(auc, 3), "n": int(valid.sum())}


def ras_quartile_hit_rates(df):
    """Hit rates by RAS quartile."""
    subset = df.dropna(subset=["ras"]).copy()
    subset["ras_q"] = pd.qcut(subset["ras"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    result = subset.groupby("ras_q", observed=True).agg(
        n=("is_hit", "count"),
        hits=("is_hit", "sum")
    )
    result["hit_rate"] = (result["hits"] / result["n"]).round(3)
    return result


def feature_subset_search(df, core_features=None, extra_features=None, min_subset=2, max_subset=6):
    """Exhaustive search over feature subsets, evaluated with PCA PC1 + LOOCV."""
    from itertools import combinations

    if core_features is None:
        core_features = ["speed_score", "broad_jump", "vertical", "forty_neg", "wt", "ht"]
    if extra_features is None:
        extra_features = []

    # Negate metrics where lower = better
    if "cone" in df.columns and "cone_neg" not in df.columns:
        df = df.copy()
        df["cone_neg"] = -df["cone"]
    if "shuttle" in df.columns and "shuttle_neg" not in df.columns:
        df = df.copy()
        df["shuttle_neg"] = -df["shuttle"]

    results = []

    def eval_subset(features):
        sub = df.dropna(subset=features).copy()
        if len(sub) < 50:
            return None
        X = sub[features].values
        tiers = sub["tier_ordinal"].values
        hits = sub["is_hit"].values
        n = len(sub)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled).ravel()
        r_in, p_in = spearmanr(pc1, tiers)
        auc_in = roc_auc_score(hits, pc1)

        # LOOCV
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            s = StandardScaler()
            Xt = s.fit_transform(X[mask])
            Xte = s.transform(X[i:i+1])
            p = PCA(n_components=1)
            p.fit(Xt)
            preds[i] = p.transform(Xte)[0, 0]

        r_cv, p_cv = spearmanr(preds, tiers)
        auc_cv = roc_auc_score(hits, preds)

        return {
            "features": "+".join(features),
            "n_features": len(features),
            "n": n,
            "spearman_in": round(r_in, 4),
            "spearman_cv": round(r_cv, 4),
            "p_cv": round(p_cv, 4),
            "auc_in": round(auc_in, 3),
            "auc_cv": round(auc_cv, 3),
            "cv_drop": round(r_in - r_cv, 4),
        }

    # All subsets of core features
    for size in range(min_subset, min(max_subset + 1, len(core_features) + 1)):
        for combo in combinations(core_features, size):
            r = eval_subset(list(combo))
            if r:
                results.append(r)

    # Core subsets + one extra feature
    for extra in extra_features:
        for size in range(min_subset, min(max_subset, len(core_features) + 1)):
            for combo in combinations(core_features, size):
                r = eval_subset(list(combo) + [extra])
                if r:
                    results.append(r)

    return pd.DataFrame(results).sort_values("auc_cv", ascending=False)


def run_full_analysis(dataset="historical", save=False):
    print(f"Loading {dataset} dataset...")
    df = load_data(dataset)
    print(f"  {len(df)} resolved players, {df.draft_year.min()}-{df.draft_year.max()}")
    print(f"  Tier distribution: {df.computed_tier.value_counts().to_dict()}")
    print()

    # --- Univariate ---
    print("=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)
    uni = univariate_analysis(df)
    print(uni.to_string())
    print()

    # --- RAS ---
    print("=" * 60)
    print("RAS (Relative Athletic Score)")
    print("=" * 60)
    df = compute_ras(df)
    ras_eval = evaluate(df["ras"].values, df["tier_ordinal"].values, df["is_hit"].values, "RAS")
    print(f"  Spearman: {ras_eval['spearman']}, p: {ras_eval['p']}, AUC: {ras_eval['auc']}, n: {ras_eval['n']}")

    quartiles = ras_quartile_hit_rates(df)
    print("\n  RAS Quartile Hit Rates:")
    print(quartiles.to_string())
    print()

    # --- PCA ---
    print("=" * 60)
    print("PCA (Unsupervised)")
    print("=" * 60)
    pca_df, pca_model, scaler, loadings = run_pca(df)
    print(f"  Complete cases: {len(pca_df)}")
    print(f"\n  Explained variance:")
    for i, v in enumerate(pca_model.explained_variance_ratio_):
        print(f"    PC{i+1}: {v:.1%}")
    print(f"\n  PC1 Loadings:")
    print(loadings["PC1"].sort_values(ascending=False).to_string())
    print(f"\n  PC2 Loadings:")
    print(loadings["PC2"].sort_values(ascending=False).to_string())

    pc1_eval = evaluate(pca_df["pc1"].values, pca_df["tier_ordinal"].values, pca_df["is_hit"].values, "PCA PC1 (in-sample)")
    pc2_eval = evaluate(pca_df["pc2"].values, pca_df["tier_ordinal"].values, pca_df["is_hit"].values, "PCA PC2 (in-sample)")
    print(f"\n  PC1 — Spearman: {pc1_eval['spearman']}, p: {pc1_eval['p']}, AUC: {pc1_eval['auc']}")
    print(f"  PC2 — Spearman: {pc2_eval['spearman']}, p: {pc2_eval['p']}, AUC: {pc2_eval['auc']}")

    # PCA LOOCV
    print("  Running PCA LOOCV...")
    pca_cv_df = pca_loocv(df)
    pc1_cv_eval = evaluate(pca_cv_df["pc1_loocv"].values, pca_cv_df["tier_ordinal"].values, pca_cv_df["is_hit"].values, "PCA PC1 (LOOCV)")
    print(f"  PC1 LOOCV — Spearman: {pc1_cv_eval['spearman']}, p: {pc1_cv_eval['p']}, AUC: {pc1_cv_eval['auc']}")

    # Top/bottom PC1
    print(f"\n  Top 10 PC1:")
    top10 = pca_df.nlargest(10, "pc1")[["name", "draft_year", "pc1", "computed_tier"]]
    print(top10.to_string(index=False))
    print(f"\n  Bottom 10 PC1:")
    bot10 = pca_df.nsmallest(10, "pc1")[["name", "draft_year", "pc1", "computed_tier"]]
    print(bot10.to_string(index=False))
    print()

    # --- Supervised ---
    print("=" * 60)
    print("SUPERVISED COMPOSITE (Correlation-Weighted)")
    print("=" * 60)
    sup_df, sup_weights = supervised_composite(df)
    print(f"  Complete cases: {len(sup_df)}")
    print(f"\n  Normalized weights:")
    for f, w in sorted(sup_weights.items(), key=lambda x: -x[1]):
        print(f"    {f}: {w:.3f}")

    sup_eval = evaluate(sup_df["supervised_score"].values, sup_df["tier_ordinal"].values, sup_df["is_hit"].values, "Supervised (in-sample)")
    print(f"\n  In-sample — Spearman: {sup_eval['spearman']}, p: {sup_eval['p']}, AUC: {sup_eval['auc']}")

    # Supervised LOOCV
    print("  Running Supervised LOOCV...")
    sup_cv_df = supervised_loocv(df)
    sup_cv_eval = evaluate(sup_cv_df["supervised_loocv"].values, sup_cv_df["tier_ordinal"].values, sup_cv_df["is_hit"].values, "Supervised (LOOCV)")
    print(f"  LOOCV — Spearman: {sup_cv_eval['spearman']}, p: {sup_cv_eval['p']}, AUC: {sup_cv_eval['auc']}")

    # Top supervised
    print(f"\n  Top 10 Supervised (in-sample):")
    top10_sup = sup_df.nlargest(10, "supervised_score")[["name", "draft_year", "supervised_score", "computed_tier"]]
    print(top10_sup.to_string(index=False))
    print()

    # --- Head-to-head comparison ---
    print("=" * 60)
    print("HEAD-TO-HEAD (same complete-case players)")
    print("=" * 60)
    # Use the PCA complete cases for apples-to-apples
    common = pca_df.dropna(subset=["ras"]).copy()
    # Re-run supervised on same subset
    sup_common, _ = supervised_composite(common)
    sup_cv_common = supervised_loocv(common)

    tiers = common["tier_ordinal"].values
    hits = common["is_hit"].values
    n = len(common)

    comparisons = []
    comparisons.append(evaluate(common["ras"].values, tiers, hits, "RAS"))
    comparisons.append(evaluate(common["pc1"].values, tiers, hits, "PCA PC1 (in-sample)"))

    # PCA LOOCV on common subset
    pca_cv_common = pca_loocv(common)
    comparisons.append(evaluate(pca_cv_common["pc1_loocv"].values, tiers, hits, "PCA PC1 (LOOCV)"))

    comparisons.append(evaluate(sup_common["supervised_score"].values, sup_common["tier_ordinal"].values, sup_common["is_hit"].values, "Supervised (in-sample)"))
    comparisons.append(evaluate(sup_cv_common["supervised_loocv"].values, sup_cv_common["tier_ordinal"].values, sup_cv_common["is_hit"].values, "Supervised (LOOCV)"))

    # Speed score on same subset
    comparisons.append(evaluate(common["speed_score"].values, tiers, hits, "Speed score"))

    comp_df = pd.DataFrame(comparisons).set_index("method")
    print(f"  n = {n} players")
    print(comp_df.to_string())
    print()

    # --- Feature Subset Search ---
    print("=" * 60)
    print("FEATURE SUBSET SEARCH")
    print("=" * 60)
    subset_df = feature_subset_search(
        df,
        core_features=["speed_score", "broad_jump", "vertical", "forty_neg", "wt", "ht"],
        extra_features=["bench", "cone_neg", "shuttle_neg"],
    )
    print(f"\n  Top 15 subsets by LOOCV AUC:")
    print(subset_df.head(15)[["features", "n_features", "n", "spearman_cv", "p_cv", "auc_cv", "cv_drop"]].to_string(index=False))
    print()

    # --- Save ---
    if save:
        output_dir = PROJECT_ROOT / "rb_data" / "outputs"
        report_dir = PROJECT_ROOT / "rb_data" / "reports"

        # Save updated JSON
        results = {
            "dataset": dataset,
            "n_resolved": len(df),
            "year_range": f"{df.draft_year.min()}-{df.draft_year.max()}",
            "tier_distribution": df.computed_tier.value_counts().to_dict(),
            "univariate": uni.to_dict(orient="index"),
            "ras": ras_eval,
            "ras_quartiles": quartiles.to_dict(orient="index"),
            "pca": {
                "n_complete": len(pca_df),
                "explained_variance": pca_model.explained_variance_ratio_.tolist(),
                "loadings_pc1": loadings["PC1"].to_dict(),
                "loadings_pc2": loadings["PC2"].to_dict(),
                "pc1_in_sample": pc1_eval,
                "pc1_loocv": pc1_cv_eval,
                "pc2_in_sample": pc2_eval,
            },
            "supervised": {
                "weights": sup_weights,
                "in_sample": sup_eval,
                "loocv": sup_cv_eval,
            },
            "head_to_head": comp_df.reset_index().to_dict(orient="records"),
            "feature_subset_search": {
                "top_15": subset_df.head(15).to_dict(orient="records"),
                "best_subset": subset_df.iloc[0].to_dict() if len(subset_df) > 0 else None,
            },
        }

        json_path = output_dir / "rb_athleticism_pca_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results to {json_path}")

        # Update the historical CSV with fresh scores
        pca_cv_full = pca_loocv(df)
        sup_full, _ = supervised_composite(df)
        sup_cv_full = supervised_loocv(df)

        df_out = df.copy()
        # Merge PCA scores
        df_out = df_out.drop(columns=["pc1", "supervised_score", "supervised_loocv"], errors="ignore")
        df_out = df_out.merge(pca_df[["name", "draft_year", "pc1"]].rename(columns={"pc1": "pc1"}), on=["name", "draft_year"], how="left")
        df_out = df_out.merge(pca_cv_full[["name", "draft_year", "pc1_loocv"]], on=["name", "draft_year"], how="left")
        df_out = df_out.merge(sup_full[["name", "draft_year", "supervised_score"]], on=["name", "draft_year"], how="left")
        df_out = df_out.merge(sup_cv_full[["name", "draft_year", "supervised_loocv"]], on=["name", "draft_year"], how="left")

        # Rename columns back for compatibility
        df_out = df_out.rename(columns={"wt": "wt_kg", "ht": "ht_cm"})
        csv_path = output_dir / "rb_historical_athleticism.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"Saved updated CSV to {csv_path}")

    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RB Athleticism PCA Analysis")
    parser.add_argument("--dataset", choices=["historical", "modern"], default="historical")
    parser.add_argument("--save", action="store_true", help="Save updated CSV and JSON results")
    args = parser.parse_args()

    run_full_analysis(dataset=args.dataset, save=args.save)
