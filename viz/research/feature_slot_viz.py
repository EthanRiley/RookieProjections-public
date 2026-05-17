#!/usr/bin/env python3
"""
Visualize the feature slot analysis results.

4-panel layout:
  1. Per-slot candidate comparison (LOO-AUC vs baseline)
  2. Top combo heatmap
  3. Residual signal + bootstrap for each candidate
  4. Radar chart of slot winner profiles
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}

# --- Load data ---
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
max_bo = df["breakout_age"].max()
df["breakout_age"] = df["breakout_age"].fillna(round(max_bo + 1, 2))
df["breakout_yptpa"] = df["breakout_yptpa"].fillna(0)
df["breakout_yprr"] = df["breakout_yprr"].fillna(0)

LOCKED = [
    "draft_capital",
    "breakout_age",
    "breakout_yprr",
    "breakout_yptpa",
    "best_contested_catch_rate",
]

SLOTS = {
    "A: QB Confidence": [
        "career_targeted_qb_rating",
        "best_targeted_qb_rating",
        "best2_targeted_qb_rating",
        "career_grades_pass_route",
        "best2_grades_pass_route",
        "career_grades_offense",
    ],
    "B: Route Efficiency": [
        "career_yprr",
        "best2_yprr",
        "career_yards_pg",
        "best2_yards_pg",
        "career_first_downs_per_route",
        "best_yards_per_team_pass_att",
    ],
    "C: Catch Reliability": [
        "career_catch_pct_adot_adj",
        "best2_catch_pct_adot_adj",
        "career_caught_percent",
        "best2_caught_percent",
        "best_caught_percent",
    ],
    "D: Elusiveness / YAC": [
        "career_avoided_tackles_pg",
        "best_avoided_tackles_pg",
        "best2_avoided_tackles_pg",
        "career_avoided_tackles_per_rec",
        "best2_avoided_tackles_per_rec",
        "career_yards_after_catch_pg",
    ],
}

# Filter to complete cases
all_candidates = LOCKED.copy()
for candidates in SLOTS.values():
    all_candidates.extend(candidates)
all_candidates = list(set(c for c in all_candidates if c in df.columns))
df_full = df.dropna(subset=["tier_ordinal"] + all_candidates).copy()
df_full["tier_ordinal"] = df_full["tier_ordinal"].astype(int)
y = df_full["tier_ordinal"].values
hit = (y >= 3).astype(int)

scaler = StandardScaler()
ridge = Ridge(alpha=1.0)
rng = np.random.RandomState(42)


def loo_auc(feature_set):
    feats = [f for f in feature_set if f in df_full.columns]
    all_preds, all_true = [], []
    for fold_year in sorted(df_full["draft_year"].unique()):
        tr = df_full["draft_year"] != fold_year
        val = df_full["draft_year"] == fold_year
        y_tr = (df_full.loc[tr, "tier_ordinal"].values >= 3).astype(int)
        y_val = (df_full.loc[val, "tier_ordinal"].values >= 3).astype(int)
        if y_tr.sum() < 2 or y_val.sum() == 0 or y_val.sum() == len(y_val):
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_full.loc[tr, feats].values)
        X_val = sc.transform(df_full.loc[val, feats].values)
        model = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
        model.fit(X_tr, y_tr)
        all_preds.extend(model.predict_proba(X_val)[:, 1])
        all_true.extend(y_val)
    return roc_auc_score(np.array(all_true), np.array(all_preds))


def residual_signal(feat, control_features):
    ctrl = [c for c in control_features if c in df_full.columns]
    if not ctrl:
        sp, _ = spearmanr(df_full[feat].values, y)
        return float(sp)
    X_ctrl = scaler.fit_transform(df_full[ctrl].values)
    ridge.fit(X_ctrl, y)
    resid = y - ridge.predict(X_ctrl)
    sp, _ = spearmanr(df_full[feat].values, resid)
    return float(sp)


def bootstrap_residual(feat, control_features, n_boot=500):
    ctrl = [c for c in control_features if c in df_full.columns]
    results = []
    for _ in range(n_boot):
        idx = rng.choice(len(df_full), size=len(df_full), replace=True)
        if ctrl:
            X_b = scaler.fit_transform(df_full[ctrl].values[idx])
            y_b = y[idx]
            ridge.fit(X_b, y_b)
            resid = y_b - ridge.predict(X_b)
        else:
            resid = y[idx]
        sp, _ = spearmanr(df_full[feat].values[idx], resid)
        results.append(sp)
    return np.array(results)


def era_stability(feat):
    mid_years = sorted(df_full["draft_year"].unique())
    mid = mid_years[len(mid_years) // 2]
    early = df_full[df_full["draft_year"] <= mid]
    late = df_full[df_full["draft_year"] > mid]
    sp_e, _ = spearmanr(early[feat].values, early["tier_ordinal"].values)
    sp_l, _ = spearmanr(late[feat].values, late["tier_ordinal"].values)
    return abs(float(sp_e) - float(sp_l))


def max_collinearity(feat):
    ref = [c for c in LOCKED if c != feat]
    if not ref:
        return 0.0
    return max(abs(float(spearmanr(df_full[feat].values, df_full[r].values)[0])) for r in ref)


def univariate(feat):
    sp, _ = spearmanr(df_full[feat].values, y)
    auc = roc_auc_score(hit, df_full[feat].values)
    if auc < 0.5:
        auc = 1 - auc
    return float(sp), float(auc)


# --- Compute all metrics ---
print("Computing metrics...")
baseline_auc = loo_auc(LOCKED)

slot_data = {}
for slot_name, candidates in SLOTS.items():
    candidates = [c for c in candidates if c in df_full.columns]
    slot_data[slot_name] = []
    for cand in candidates:
        sp, auc = univariate(cand)
        resid = residual_signal(cand, LOCKED)
        boot = bootstrap_residual(cand, LOCKED)
        drift = era_stability(cand)
        col = max_collinearity(cand)
        la = loo_auc(LOCKED + [cand])
        slot_data[slot_name].append({
            "name": cand, "sp": sp, "auc": auc,
            "resid": resid, "boot": boot,
            "drift": drift, "col": col, "loo_auc": la,
        })
    # Sort by LOO-AUC
    slot_data[slot_name].sort(key=lambda x: x["loo_auc"], reverse=True)

print("Plotting...")

# --- Short display names ---
def short_name(name):
    return (name
            .replace("career_", "c_")
            .replace("best2_", "b2_")
            .replace("best_", "b_")
            .replace("_per_route", "/rt")
            .replace("_per_rec", "/rec")
            .replace("_pg", "/g")
            .replace("_pct_adot_adj", "%_adj")
            .replace("_catch_rate", "_cr")
            .replace("_pass_route", "_pr")
            .replace("_per_team_pass_att", "/tpa")
            .replace("_after_catch", "_ac")
            .replace("avoided_tackles", "at")
            .replace("caught_percent", "catch%")
            .replace("first_downs", "1d")
            .replace("targeted_qb_rating", "tqbr")
            .replace("grades_offense", "off_grade")
            .replace("yards", "yds")
            )


# ============================================================
# FIGURE 1: LOO-AUC comparison per slot
# ============================================================
slot_keys = list(SLOTS.keys())
n_slots = len(slot_keys)

fig, axes = plt.subplots(1, n_slots, figsize=(20, 7), sharey=True)
fig.suptitle("Feature Slot Analysis: LOO-AUC When Added to Locked Features",
             fontsize=14, fontweight="bold", y=1.02)

slot_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, (slot_name, ax) in enumerate(zip(slot_keys, axes)):
    data = slot_data[slot_name]
    names = [short_name(d["name"]) for d in data]
    aucs = [d["loo_auc"] for d in data]

    colors = ["#2ca02c" if a > baseline_auc else "#ff7f0e" if a > baseline_auc - 0.005 else "#d62728"
              for a in aucs]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor="white", height=0.6, alpha=0.85)
    ax.axvline(baseline_auc, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Baseline ({baseline_auc:.3f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(slot_name, fontsize=11, fontweight="bold")
    ax.set_xlim(min(min(aucs), baseline_auc) - 0.01, max(max(aucs), baseline_auc) + 0.01)
    ax.invert_yaxis()

    for j, (bar, auc_val) in enumerate(zip(bars, aucs)):
        delta = auc_val - baseline_auc
        label = f"{auc_val:.3f} ({delta:+.3f})"
        ax.text(auc_val + 0.001, j, label, va="center", fontsize=8)

    if i == 0:
        ax.set_ylabel("Candidate Feature")
    ax.legend(fontsize=8, loc="lower right")

axes[-1].set_xlabel("LOO-AUC (Elite+ threshold)")
plt.tight_layout()
out1 = os.path.join(DATA_DIR, "slot_analysis_loo_auc.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out1}")


# ============================================================
# FIGURE 2: Bootstrap residual signal distributions
# ============================================================
fig2, axes2 = plt.subplots(1, n_slots, figsize=(20, 7), sharey=False)
fig2.suptitle("Residual Signal After Locked Features (Bootstrap Distribution)",
              fontsize=14, fontweight="bold", y=1.02)

for i, (slot_name, ax) in enumerate(zip(slot_keys, axes2)):
    data = slot_data[slot_name]
    names = [short_name(d["name"]) for d in data]

    positions = []
    for j, d in enumerate(data):
        boot = d["boot"]
        bp = ax.boxplot([boot], positions=[j], vert=False, widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        pct_pos = (boot > 0).mean()
        color = "#2ca02c" if pct_pos > 0.75 else "#ff7f0e" if pct_pos > 0.5 else "#d62728"
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.6)

    ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(slot_name, fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # Annotate with % positive
    for j, d in enumerate(data):
        pct = (d["boot"] > 0).mean()
        ax.text(ax.get_xlim()[1], j, f" {pct:.0%}+", va="center", fontsize=8, color="gray")

    if i == 0:
        ax.set_ylabel("Candidate Feature")

axes2[-1].set_xlabel("Residual Spearman (after locked features)")

# Legend
green = mpatches.Patch(color="#2ca02c", alpha=0.6, label=">75% positive")
orange = mpatches.Patch(color="#ff7f0e", alpha=0.6, label="50-75% positive")
red = mpatches.Patch(color="#d62728", alpha=0.6, label="<50% positive")
axes2[-1].legend(handles=[green, orange, red], fontsize=8, loc="lower right")

plt.tight_layout()
out2 = os.path.join(DATA_DIR, "slot_analysis_residual.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out2}")


# ============================================================
# FIGURE 3: Multi-metric radar per slot (best candidate vs baseline)
# ============================================================
fig3, axes3 = plt.subplots(1, n_slots, figsize=(20, 5.5), subplot_kw=dict(projection="polar"))
fig3.suptitle("Slot Winner Profiles: 5 Metrics (normalized 0-1, higher=better)",
              fontsize=14, fontweight="bold", y=1.05)

metrics = ["Spearman", "AUC", "Stability", "Independence", "Residual"]
n_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]

for i, (slot_name, ax) in enumerate(zip(slot_keys, axes3)):
    data = slot_data[slot_name]

    # Normalize metrics across candidates in this slot
    all_sp = [abs(d["sp"]) for d in data]
    all_auc = [d["auc"] for d in data]
    all_stab = [1 - d["drift"] for d in data]  # invert: lower drift = better
    all_indep = [1 - d["col"] for d in data]  # invert: lower collinearity = better
    all_resid = [d["resid"] for d in data]

    def norm(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    sp_n = norm(all_sp)
    auc_n = norm(all_auc)
    stab_n = norm(all_stab)
    indep_n = norm(all_indep)
    resid_n = norm(all_resid)

    # Plot top 3
    top_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for j in range(min(3, len(data))):
        values = [sp_n[j], auc_n[j], stab_n[j], indep_n[j], resid_n[j]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, color=top_colors[j], markersize=4,
                label=short_name(data[j]["name"]))
        ax.fill(angles, values, alpha=0.1, color=top_colors[j])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(slot_name, fontsize=10, fontweight="bold", pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.15))

plt.tight_layout()
out3 = os.path.join(DATA_DIR, "slot_analysis_radar.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out3}")


# ============================================================
# FIGURE 4: Summary — delta AUC heatmap for all candidates
# ============================================================
fig4, ax4 = plt.subplots(figsize=(10, 12))
fig4.suptitle("LOO-AUC Delta vs Locked-Only Baseline (0.853)",
              fontsize=14, fontweight="bold")

all_entries = []
for slot_name in slot_keys:
    data = slot_data[slot_name]
    for d in data:
        all_entries.append({
            "slot": slot_name.split(":")[0].strip(),
            "name": short_name(d["name"]),
            "delta": d["loo_auc"] - baseline_auc,
            "resid": d["resid"],
        })

entries_df = pd.DataFrame(all_entries)
entries_df = entries_df.sort_values("delta", ascending=True)

y_pos = np.arange(len(entries_df))
deltas = entries_df["delta"].values
colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]

slot_label_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}

ax4.barh(y_pos, deltas, color=colors, edgecolor="white", height=0.7, alpha=0.8)
ax4.axvline(0, color="black", linewidth=1.5)

labels = []
for _, row in entries_df.iterrows():
    labels.append(f"[{row['slot']}] {row['name']}")
ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel("LOO-AUC Delta vs Baseline (locked features only)", fontsize=11)

# Annotate
for j, (delta, resid) in enumerate(zip(deltas, entries_df["resid"].values)):
    sign = "+" if delta >= 0 else ""
    ax4.text(delta + (0.0005 if delta >= 0 else -0.0005), j,
             f"{sign}{delta:.3f}  (resid={resid:+.3f})",
             va="center", ha="left" if delta >= 0 else "right", fontsize=8)

plt.tight_layout()
out4 = os.path.join(DATA_DIR, "slot_analysis_delta.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out4}")

print("\nDone! All visualizations saved to wr_data/")
