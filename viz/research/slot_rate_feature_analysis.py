#!/usr/bin/env python3
"""
Slot rate feature analysis visualization.

Shows that slot rate has no predictive value for dynasty outcomes:
  Panel 1: Univariate AUC comparison (slot rate vs model features)
  Panel 2: Bivariate with DC (what does adding each feature to DC do?)
  Panel 3: Bootstrap residual distributions after full model
  Panel 4: Scatter of slot rate vs dynasty value

Outputs: wr_data/charts/slot_rate_feature_analysis.png
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wr_data")
OUT_DIR = os.path.join(DATA_DIR, "charts")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

TIER_COLORS = {
    "Bust": "#d62728",
    "Flex": "#ff7f0e",
    "Starter": "#bcbd22",
    "Elite": "#2ca02c",
    "Stud": "#1f77b4",
    "League-Winner": "#9467bd",
}

FULL_MODEL = [
    "draft_capital", "best1_yprr_graduated", "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj", "best2_contested_catch_rate", "best2_avoided_tackles_per_rec",
]

FEATURE_LABELS = {
    "snap_weighted_slot_rate": "Slot Rate",
    "draft_capital": "Draft Capital",
    "best1_yprr_graduated": "YPRR (graduated)",
    "career_targeted_qb_rating": "Targeted QBR",
    "best2_catch_pct_adot_adj": "Catch% (aDOT adj)",
    "best2_contested_catch_rate": "Contested Catch%",
    "best2_avoided_tackles_per_rec": "Avoided Tackles/Rec",
}


def load_data():
    master = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
    master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER)

    player_slot_snaps = {}
    player_total_snaps = {}
    for y in range(2016, 2026):
        path = os.path.join(DATA_DIR, "grades", f"{y}_receiving_grades.csv")
        if os.path.exists(path):
            g = pd.read_csv(path)
            for _, row in g.iterrows():
                name = row["player"]
                ss = row.get("slot_snaps", 0)
                ws = row.get("wide_snaps", 0)
                inl = row.get("inline_snaps", 0)
                total = ss + ws + inl
                player_slot_snaps[name] = player_slot_snaps.get(name, 0) + ss
                player_total_snaps[name] = player_total_snaps.get(name, 0) + total

    for idx, row in master.iterrows():
        name = row["name"]
        if name in player_total_snaps and player_total_snaps[name] > 0:
            master.at[idx, "snap_weighted_slot_rate"] = (
                player_slot_snaps[name] / player_total_snaps[name] * 100
            )

    all_cols = FULL_MODEL + ["snap_weighted_slot_rate", "tier_ordinal"]
    df = master.dropna(subset=[c for c in all_cols if c in master.columns]).copy()
    return df


def loo_auc(df, feature_set):
    feats = [f for f in feature_set if f in df.columns]
    all_preds, all_true = [], []
    for fold_year in sorted(df["draft_year"].unique()):
        tr = df["draft_year"] != fold_year
        val = df["draft_year"] == fold_year
        y_tr = (df.loc[tr, "tier_ordinal"].values >= 3).astype(int)
        y_val = (df.loc[val, "tier_ordinal"].values >= 3).astype(int)
        if y_tr.sum() < 2 or y_val.sum() == 0 or y_val.sum() == len(y_val):
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(df.loc[tr, feats].values)
        X_val = sc.transform(df.loc[val, feats].values)
        model = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        all_preds.extend(probs)
        all_true.extend(y_val)
    return roc_auc_score(np.array(all_true), np.array(all_preds))


def main():
    df = load_data()
    y = df["tier_ordinal"].values.astype(int)
    hit = (y >= 3).astype(int)

    features_to_test = ["snap_weighted_slot_rate"] + FULL_MODEL

    # --- Panel 1: Univariate AUC ---
    uni_aucs = {}
    for feat in features_to_test:
        vals = df[feat].values
        auc = roc_auc_score(hit, vals)
        if auc < 0.5:
            auc = 1 - auc
        uni_aucs[feat] = auc

    # --- Panel 2: Targeted bivariate comparisons ---
    biv_combos = [
        ("DC Only", ["draft_capital"]),
        ("DC + QBR", ["draft_capital", "career_targeted_qb_rating"]),
        ("DC + Slot Rate", ["draft_capital", "snap_weighted_slot_rate"]),
        ("DC + Slot Rate + QBR", ["draft_capital", "snap_weighted_slot_rate", "career_targeted_qb_rating"]),
    ]
    biv_results = [(label, loo_auc(df, feats)) for label, feats in biv_combos]

    # --- Panel 3: Bootstrap residuals ---
    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0)
    rng = np.random.RandomState(42)

    boot_results = {}
    for feat in features_to_test:
        if feat == "snap_weighted_slot_rate":
            results = []
            for _ in range(1000):
                idx = rng.choice(len(df), size=len(df), replace=True)
                X_b = scaler.fit_transform(df[FULL_MODEL].values[idx])
                y_b = y[idx]
                ridge.fit(X_b, y_b)
                resid = y_b - ridge.predict(X_b)
                sp, _ = spearmanr(df[feat].values[idx], resid)
                results.append(sp)
            boot_results[feat] = np.array(results)
        else:
            others = [f for f in FULL_MODEL if f != feat]
            results = []
            for _ in range(1000):
                idx = rng.choice(len(df), size=len(df), replace=True)
                X_b = scaler.fit_transform(df[others].values[idx])
                y_b = y[idx]
                ridge.fit(X_b, y_b)
                resid = y_b - ridge.predict(X_b)
                sp, _ = spearmanr(df[feat].values[idx], resid)
                results.append(sp)
            boot_results[feat] = np.array(results)

    # --- Compute R² for full model vs full model + slot rate ---
    from sklearn.linear_model import LinearRegression
    y_dv = df["dynasty_value"].values
    r2_full = LinearRegression().fit(df[FULL_MODEL].values, y_dv).score(df[FULL_MODEL].values, y_dv)
    r2_plus = LinearRegression().fit(
        df[FULL_MODEL + ["snap_weighted_slot_rate"]].values, y_dv
    ).score(df[FULL_MODEL + ["snap_weighted_slot_rate"]].values, y_dv)

    # Univariate R² for each feature
    uni_r2 = {}
    for feat in features_to_test:
        X_uni = df[[feat]].values
        uni_r2[feat] = LinearRegression().fit(X_uni, y_dv).score(X_uni, y_dv)

    # ===== PLOTTING =====
    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    fig.suptitle("Slot Rate as a Predictive Feature: Full Analysis",
                 fontsize=15, fontweight="bold", y=0.99)

    # --- Panel 1: Univariate AUC bars ---
    ax1 = axes[0, 0]
    labels_1 = [FEATURE_LABELS.get(f, f) for f in features_to_test]
    values_1 = [uni_aucs[f] for f in features_to_test]
    colors_1 = ["#d62728" if f == "snap_weighted_slot_rate" else "#4a90d9" for f in features_to_test]
    bars = ax1.barh(range(len(features_to_test)), values_1, color=colors_1, alpha=0.8,
                    edgecolor="white", height=0.6)
    ax1.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Coin flip (0.50)")
    ax1.set_yticks(range(len(features_to_test)))
    ax1.set_yticklabels(labels_1, fontsize=9)
    ax1.set_xlabel("AUC (Elite+ threshold)", fontsize=10)
    ax1.set_title("Univariate AUC: Slot Rate vs Model Features", fontsize=11, fontweight="bold")
    ax1.set_xlim(0.35, 0.95)
    ax1.legend(fontsize=8)
    for i, v in enumerate(values_1):
        ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold",
                 color=colors_1[i])

    # --- Panel 2: Targeted bivariate comparisons ---
    ax2 = axes[0, 1]
    biv_labels_plot = [label for label, _ in biv_results]
    biv_values_plot = [auc for _, auc in biv_results]
    colors_2 = ["#4a90d9", "#4a90d9", "#d62728", "#d62728"]
    ax2.barh(range(len(biv_results)), biv_values_plot, color=colors_2, alpha=0.8,
             edgecolor="white", height=0.6)
    ax2.set_yticks(range(len(biv_results)))
    ax2.set_yticklabels(biv_labels_plot, fontsize=9)
    ax2.set_xlabel("LOO-CV AUC (Elite+ threshold)", fontsize=10)
    ax2.set_title("Does Slot Rate Add Value to Draft Capital?", fontsize=11, fontweight="bold")
    ax2.set_xlim(0.82, 0.9)
    for i, v in enumerate(biv_values_plot):
        ax2.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold",
                 color=colors_2[i])

    # --- Panel 3: Univariate R² bars ---
    ax_r2 = axes[1, 0]
    r2_labels = [FEATURE_LABELS.get(f, f) for f in features_to_test]
    r2_values = [uni_r2[f] for f in features_to_test]
    r2_colors = ["#d62728" if f == "snap_weighted_slot_rate" else "#4a90d9" for f in features_to_test]
    ax_r2.barh(range(len(features_to_test)), r2_values, color=r2_colors, alpha=0.8,
               edgecolor="white", height=0.6)
    ax_r2.set_yticks(range(len(features_to_test)))
    ax_r2.set_yticklabels(r2_labels, fontsize=9)
    ax_r2.set_xlabel("R² with Dynasty Value", fontsize=10)
    ax_r2.set_title("Univariate R²: Slot Rate vs Model Features", fontsize=11, fontweight="bold")
    for i, v in enumerate(r2_values):
        ax_r2.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold",
                   color=r2_colors[i])

    # --- Panel 4: Full Model vs Full Model + Slot Rate ---
    ax_fm = axes[1, 1]
    fm_labels = ["Full Model\n(6 features)", "Full Model\n+ Slot Rate"]
    fm_values = [r2_full, r2_plus]
    fm_colors = ["#4a90d9", "#d62728"]
    bars_fm = ax_fm.bar(range(2), fm_values, color=fm_colors, alpha=0.8,
                        edgecolor="white", width=0.5)
    ax_fm.set_xticks(range(2))
    ax_fm.set_xticklabels(fm_labels, fontsize=10)
    ax_fm.set_ylabel("R² with Dynasty Value", fontsize=10)
    ax_fm.set_title("Full Model vs Full Model + Slot Rate", fontsize=11, fontweight="bold")
    ax_fm.set_ylim(0, max(fm_values) * 1.25)
    for i, v in enumerate(fm_values):
        ax_fm.text(i, v + 0.005, f"R² = {v:.4f}", ha="center", fontsize=11, fontweight="bold",
                   color=fm_colors[i])
    delta = r2_plus - r2_full
    ax_fm.text(0.5, 0.85, f"Adding slot rate: {delta:+.4f} R²",
               transform=ax_fm.transAxes, ha="center", fontsize=10, fontstyle="italic",
               color="gray")

    # --- Panel 5: Bootstrap residual distributions ---
    ax3 = axes[2, 0]
    boot_features = features_to_test
    boot_labels = [FEATURE_LABELS.get(f, f) for f in boot_features]
    boot_colors = ["#d62728" if f == "snap_weighted_slot_rate" else "#4a90d9" for f in boot_features]

    positions = range(len(boot_features))
    for i, feat in enumerate(boot_features):
        data = boot_results[feat]
        parts = ax3.violinplot([data], positions=[i], vert=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(boot_colors[i])
            pc.set_alpha(0.6)
        lo, med, hi = np.percentile(data, [2.5, 50, 97.5])
        ax3.plot([lo, hi], [i, i], color=boot_colors[i], linewidth=2)
        ax3.plot(med, i, "o", color=boot_colors[i], markersize=6)
        pct_pos = (data > 0).mean() * 100
        ax3.text(hi + 0.01, i, f"{pct_pos:.0f}% +", va="center", fontsize=8,
                 color=boot_colors[i], fontweight="bold")

    ax3.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax3.set_yticks(positions)
    ax3.set_yticklabels(boot_labels, fontsize=9)
    ax3.set_xlabel("Residual Spearman (bootstrap, 1000 iters)", fontsize=10)
    ax3.set_title("Bootstrap Residual Signal After Other Features", fontsize=11, fontweight="bold")
    ax3.set_xlim(-0.35, 0.40)

    # --- Panel 6: Scatter of slot rate vs dynasty value ---
    ax4 = axes[2, 1]
    for tier_name, tier_val in TIER_ORDER.items():
        mask = df["computed_tier"] == tier_name
        ax4.scatter(
            df.loc[mask, "snap_weighted_slot_rate"],
            df.loc[mask, "dynasty_value"],
            c=TIER_COLORS[tier_name], label=tier_name, alpha=0.7,
            s=40, edgecolor="white", linewidth=0.5,
        )

    # Trend line
    x_slot = df["snap_weighted_slot_rate"].values
    y_dv = df["dynasty_value"].values
    z = np.polyfit(x_slot, y_dv, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_slot.min(), x_slot.max(), 100)
    ax4.plot(x_line, p(x_line), "--", color="gray", linewidth=1.5, alpha=0.7)

    sp, sp_p = spearmanr(x_slot, y_dv)
    from sklearn.linear_model import LinearRegression
    r2 = LinearRegression().fit(x_slot.reshape(-1, 1), y_dv).score(x_slot.reshape(-1, 1), y_dv)
    ax4.text(0.05, 0.95, f"Spearman: {sp:+.3f} (p={sp_p:.3f})\nR²: {r2:.4f}",
             transform=ax4.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax4.set_xlabel("Career Slot Rate (%)", fontsize=10)
    ax4.set_ylabel("Dynasty Value", fontsize=10)
    ax4.set_title("Slot Rate vs Dynasty Value", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "slot_rate_feature_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
