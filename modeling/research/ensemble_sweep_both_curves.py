#!/usr/bin/env python3
"""
Ensemble weight sweep for BOTH sqrt and log DC on WR holdout.

Trains 4 models (Bayesian + XGBoost for each curve), sweeps weights,
applies composite ranking, and compares each curve at its own optimal split.

Composite: 35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC

Outputs:
  - wr_data/charts/ensemble_sweep_both_curves.png
"""

import math
import os
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1

COLLEGE_FEATURES = [
    "best1_yprr_graduated",
    "career_targeted_qb_rating",
    "best2_catch_pct_adot_adj",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

HOLDOUT_YEARS = [2022, 2023, 2024]

COMPOSITE_WEIGHTS = {
    "logloss": 0.35,
    ">=Elite": 0.35,
    "brier": 0.15,
    ">=Starter": 0.10,
    ">=Stud": 0.05,
}


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


def dc_sqrt(pick):
    return 10 - 7 * math.sqrt(pick / 260)


# --- Load data ---
all_features = ["draft_capital"] + COLLEGE_FEATURES
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)
df["dc_log"] = df["pick"].apply(dc_log)
df["dc_sqrt"] = df["pick"].apply(dc_sqrt)

train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

print(f"Training: {len(train_df)} | Holdout: {len(holdout_df)}")

y_train = train_df["tier_ordinal"].values
actual = holdout_df["tier_ordinal"].values

scaler = StandardScaler()
X_college_train = scaler.fit_transform(train_df[COLLEGE_FEATURES].values)
X_college_hold = scaler.transform(holdout_df[COLLEGE_FEATURES].values)


def train_xgb(dc_train, dc_hold, label):
    print(f"  Training XGBoost {label}...")
    X_train = np.column_stack([dc_train, X_college_train])
    X_hold = np.column_stack([dc_hold, X_college_hold])

    cum_probs = np.zeros((len(holdout_df), len(THRESHOLDS)))
    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train >= threshold).astype(int)
        pos = y_bin.sum()
        scale = (len(y_bin) - pos) / max(pos, 1)
        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, random_state=42, eval_metric="logloss",
        )
        min_class = min(y_bin.sum(), len(y_bin) - y_bin.sum())
        cv_folds = min(5, max(2, min_class))
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds)
        calibrated.fit(X_train, y_bin)
        cum_probs[:, t_idx] = calibrated.predict_proba(X_hold)[:, 1]

    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    tier_probs = np.zeros((len(holdout_df), 6))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def train_bayesian(dc_train, dc_hold, label):
    print(f"  Training Bayesian {label}...")
    n_college = len(COLLEGE_FEATURES)

    with pm.Model() as model:
        beta_college = pm.Normal("beta_college", mu=0.0, sigma=0.5, shape=n_college)
        eta = pt.dot(X_college_train, beta_college)
        beta_dc = pm.Normal("beta_dc", mu=0.5, sigma=0.3)
        eta = eta + beta_dc * dc_train
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-2, 3, N_CUTPOINTS),
            sigma=1.5, shape=N_CUTPOINTS,
            transform=pm.distributions.transforms.ordered,
        )
        pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y_train)

    with model:
        trace = pm.sample(
            3000, tune=2000, chains=4, cores=1,
            random_seed=42, progressbar=True, target_accept=0.9,
        )

    beta_college_samples = trace.posterior["beta_college"].values.reshape(-1, n_college)
    cutpoints_samples = trace.posterior["cutpoints"].values.reshape(-1, N_CUTPOINTS)
    beta_dc_samples = trace.posterior["beta_dc"].values.flatten()
    n_samples = len(cutpoints_samples)
    n_obs = X_college_hold.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    for i in range(n_samples):
        eta = X_college_hold @ beta_college_samples[i] + beta_dc_samples[i] * dc_hold
        cum_probs = 1.0 / (1.0 + np.exp(-(cutpoints_samples[i] - eta[:, None])))
        sample_probs = np.zeros((n_obs, N_TIERS))
        sample_probs[:, 0] = cum_probs[:, 0]
        for k in range(1, N_CUTPOINTS):
            sample_probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        sample_probs[:, N_TIERS - 1] = 1 - cum_probs[:, N_CUTPOINTS - 1]
        tier_probs += sample_probs

    tier_probs /= n_samples
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs /= tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


def sweep_weights(bayes_probs, xgb_probs, curve_name):
    y_onehot = np.zeros((len(actual), 6))
    y_onehot[np.arange(len(actual)), actual] = 1

    weights = np.arange(0.0, 1.01, 0.05)
    results = []

    for w_bayes in weights:
        combo = w_bayes * bayes_probs + (1.0 - w_bayes) * xgb_probs
        combo = combo / combo.sum(axis=1, keepdims=True)

        ll = log_loss(y_onehot, combo)
        brier = np.mean(np.sum((y_onehot - combo) ** 2, axis=1))

        aucs = {}
        for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
            y_bin = (actual >= threshold).astype(int)
            pred = combo[:, threshold:].sum(axis=1)
            auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
            aucs[tlabel] = auc

        results.append({
            "curve": curve_name,
            "w_bayes": w_bayes,
            "w_xgb": 1.0 - w_bayes,
            "logloss": ll,
            "brier": brier,
            **aucs,
        })

    return pd.DataFrame(results)


# === Train all 4 models ===
print("\n=== SQRT ===")
sqrt_xgb = train_xgb(train_df["dc_sqrt"].values, holdout_df["dc_sqrt"].values, "Sqrt")
sqrt_bayes = train_bayesian(train_df["dc_sqrt"].values, holdout_df["dc_sqrt"].values, "Sqrt")

print("\n=== LOG ===")
log_xgb = train_xgb(train_df["dc_log"].values, holdout_df["dc_log"].values, "Log")
log_bayes = train_bayesian(train_df["dc_log"].values, holdout_df["dc_log"].values, "Log")

# === Sweep both ===
sqrt_results = sweep_weights(sqrt_bayes, sqrt_xgb, "Sqrt")
log_results = sweep_weights(log_bayes, log_xgb, "Log")

# === Composite scoring (normalized across ALL configs from both curves) ===
all_results = pd.concat([sqrt_results, log_results], ignore_index=True)

for col in COMPOSITE_WEIGHTS:
    mn, mx = all_results[col].min(), all_results[col].max()
    if mx == mn:
        all_results[f"{col}_norm"] = 0.5
    elif col in ("logloss", "brier"):
        all_results[f"{col}_norm"] = (mx - all_results[col]) / (mx - mn)
    else:
        all_results[f"{col}_norm"] = (all_results[col] - mn) / (mx - mn)

all_results["composite"] = sum(
    COMPOSITE_WEIGHTS[col] * all_results[f"{col}_norm"]
    for col in COMPOSITE_WEIGHTS
)
all_results["rank"] = all_results["composite"].rank(ascending=False).astype(int)
all_results = all_results.sort_values("composite", ascending=False)

# === Print full ranking ===
print("\n" + "=" * 130)
print("FULL COMPOSITE RANKING — BOTH CURVES (35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC)")
print("=" * 130)
print(f"  {'Rank':>4s}  {'Curve':<5s} {'Bayes':>6s} {'XGB':>6s}  {'Composite':>9s}  "
      f"{'LogLoss':>8s} {'>=Elite':>7s} {'Brier':>8s} {'>=Start':>7s} {'>=Stud':>7s}")
print(f"  {'-'*4}  {'-'*5} {'-'*6} {'-'*6}  {'-'*9}  {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

for _, row in all_results.iterrows():
    marker = ""
    if row["curve"] == "Sqrt" and row["w_bayes"] == 0.75:
        marker = "  <-- current (sqrt 75/25)"
    print(f"  {int(row['rank']):>4d}  {row['curve']:<5s} {row['w_bayes']:>5.0%} {row['w_xgb']:>5.0%}  "
          f"{row['composite']:>9.4f}  "
          f"{row['logloss']:>8.4f} {row['>=Elite']:>7.3f} {row['brier']:>8.4f} "
          f"{row['>=Starter']:>7.3f} {row['>=Stud']:>7.3f}{marker}")

# === Best per curve ===
for curve in ["Sqrt", "Log"]:
    mask = all_results["curve"] == curve
    best = all_results.loc[mask].iloc[0]
    print(f"\nBest {curve}: {best['w_bayes']:.0%}/{best['w_xgb']:.0%}  "
          f"(composite={best['composite']:.4f}, rank #{int(best['rank'])})")

# === Head-to-head at optimal splits ===
best_sqrt = all_results.loc[all_results["curve"] == "Sqrt"].iloc[0]
best_log = all_results.loc[all_results["curve"] == "Log"].iloc[0]

print("\n" + "=" * 90)
print("HEAD-TO-HEAD: Each Curve at Its Optimal Ensemble Split")
print("=" * 90)
print(f"  {'Metric':<20s} {'Sqrt':>20s} {'Log':>20s} {'Delta':>10s} {'Winner':>8s}")
print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*10} {'-'*8}")

sqrt_label = f"{best_sqrt['w_bayes']:.0%}/{best_sqrt['w_xgb']:.0%}"
log_label = f"{best_log['w_bayes']:.0%}/{best_log['w_xgb']:.0%}"

print(f"  {'Config':<20s} {sqrt_label:>20s} {log_label:>20s}")
print(f"  {'Composite':<20s} {best_sqrt['composite']:>20.4f} {best_log['composite']:>20.4f} "
      f"{best_log['composite'] - best_sqrt['composite']:>+10.4f} "
      f"{'Log' if best_log['composite'] > best_sqrt['composite'] else 'Sqrt':>8s}")

for metric in ["logloss", "brier", ">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]:
    s, l = best_sqrt[metric], best_log[metric]
    delta = l - s
    if metric in ("logloss", "brier"):
        winner = "Log" if delta < 0 else "Sqrt"
    else:
        winner = "Log" if delta > 0 else "Sqrt"
    print(f"  {metric:<20s} {s:>20.4f} {l:>20.4f} {delta:>+10.4f} {winner:>8s}")


# === Visualization ===
import matplotlib.pyplot as plt

sqrt_plot = sqrt_results.sort_values("w_bayes")
log_plot = log_results.sort_values("w_bayes")
bayes_pcts = sqrt_plot["w_bayes"].values * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("WR Ensemble Weight Sweep — Sqrt vs Log Draft Capital (Holdout n=88, 2022-2024)",
             fontsize=14, fontweight="bold", y=0.98)

C_SQRT = "#d62728"
C_LOG = "#2ca02c"

best_sqrt_pct = best_sqrt["w_bayes"] * 100
best_log_pct = best_log["w_bayes"] * 100

# Recompute composite per-curve for plotting (normalized within combined pool)
sqrt_composites = all_results.loc[all_results["curve"] == "Sqrt"].sort_values("w_bayes")["composite"].values
log_composites = all_results.loc[all_results["curve"] == "Log"].sort_values("w_bayes")["composite"].values

# Panel 1: Composite score
ax = axes[0, 0]
ax.plot(bayes_pcts, sqrt_composites, "o-", color=C_SQRT, linewidth=2.5,
        markersize=5, label="Sqrt", alpha=0.85)
ax.plot(bayes_pcts, log_composites, "s-", color=C_LOG, linewidth=2.5,
        markersize=5, label="Log", alpha=0.85)

ax.axvline(best_sqrt_pct, color=C_SQRT, linestyle=":", alpha=0.5)
ax.axvline(best_log_pct, color=C_LOG, linestyle=":", alpha=0.5)
ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)

ax.annotate(f"Sqrt best: {best_sqrt_pct:.0f}%\n({best_sqrt['composite']:.4f})",
            xy=(best_sqrt_pct, best_sqrt["composite"]),
            xytext=(best_sqrt_pct + 10, best_sqrt["composite"] - 0.04),
            fontsize=8, fontweight="bold", color=C_SQRT,
            arrowprops=dict(arrowstyle="->", color=C_SQRT, lw=1.2))
ax.annotate(f"Log best: {best_log_pct:.0f}%\n({best_log['composite']:.4f})",
            xy=(best_log_pct, best_log["composite"]),
            xytext=(best_log_pct + 10, best_log["composite"] + 0.02),
            fontsize=8, fontweight="bold", color=C_LOG,
            arrowprops=dict(arrowstyle="->", color=C_LOG, lw=1.2))

ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("Composite Score (higher = better)", fontsize=10)
ax.set_title("Composite Ranking", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

# Panel 2: LogLoss
ax = axes[0, 1]
ax.plot(bayes_pcts, sqrt_plot["logloss"], "o-", color=C_SQRT, linewidth=2.5,
        markersize=5, label="Sqrt", alpha=0.85)
ax.plot(bayes_pcts, log_plot["logloss"], "s-", color=C_LOG, linewidth=2.5,
        markersize=5, label="Log", alpha=0.85)
ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("LogLoss (lower = better)", fontsize=10)
ax.set_title("LogLoss (35% weight)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

# Panel 3: >=Elite AUC
ax = axes[1, 0]
ax.plot(bayes_pcts, sqrt_plot[">=Elite"], "o-", color=C_SQRT, linewidth=2.5,
        markersize=5, label="Sqrt", alpha=0.85)
ax.plot(bayes_pcts, log_plot[">=Elite"], "s-", color=C_LOG, linewidth=2.5,
        markersize=5, label="Log", alpha=0.85)
ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("AUC (higher = better)", fontsize=10)
ax.set_title(">=Elite AUC (35% weight)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

# Panel 4: >=Stud AUC (the concern)
ax = axes[1, 1]
ax.plot(bayes_pcts, sqrt_plot[">=Stud"], "o-", color=C_SQRT, linewidth=2.5,
        markersize=5, label="Sqrt", alpha=0.85)
ax.plot(bayes_pcts, log_plot[">=Stud"], "s-", color=C_LOG, linewidth=2.5,
        markersize=5, label="Log", alpha=0.85)
ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("AUC (higher = better)", fontsize=10)
ax.set_title(">=Stud AUC (5% weight, n=3 positives)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

# Footnote
fig.text(0.5, 0.01,
         "Composite = 35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC  |  "
         "Gray dashed = current 75/25 split",
         ha="center", fontsize=9, color="gray", style="italic")

fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out_dir = os.path.join(DATA_DIR, "charts")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ensemble_sweep_both_curves.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved chart to {out_path}")
