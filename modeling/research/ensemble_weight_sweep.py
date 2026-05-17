#!/usr/bin/env python3
"""
Grid-search ensemble weights for Bayesian vs XGBoost on WR holdout.

Trains both models once with log-scaled draft capital, then sweeps
Bayesian weight from 0.0 to 1.0 in steps of 0.05 and reports:
  - Multi-class LogLoss
  - Multi-class Brier
  - AUC at each threshold (>=Flex through >=LW)

Outputs:
  - wr_data/charts/ensemble_weight_sweep.png
  - Console table of all results
"""

import math
import os
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "wr_data")

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
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


def dc_log(pick):
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


# --- Load data ---
all_features = ["draft_capital"] + COLLEGE_FEATURES
df = pd.read_csv(os.path.join(DATA_DIR, "wr_dynasty_value_with_college.csv"))
df["tier_ordinal"] = df["computed_tier"].map(TIER_ORDER)
df = df.dropna(subset=["tier_ordinal"] + all_features).copy()
df["tier_ordinal"] = df["tier_ordinal"].astype(int)
df["dc_log"] = df["pick"].apply(dc_log)

train_df = df[~df["draft_year"].isin(HOLDOUT_YEARS)].copy()
holdout_df = df[df["draft_year"].isin(HOLDOUT_YEARS)].copy()

print(f"Training: {len(train_df)} | Holdout: {len(holdout_df)}")

y_train = train_df["tier_ordinal"].values
actual = holdout_df["tier_ordinal"].values

scaler = StandardScaler()
X_college_train = scaler.fit_transform(train_df[COLLEGE_FEATURES].values)
X_college_hold = scaler.transform(holdout_df[COLLEGE_FEATURES].values)

dc_train = train_df["dc_log"].values
dc_hold = holdout_df["dc_log"].values


# --- Train XGBoost (once) ---
def train_xgb():
    print("\nTraining XGBoost...")
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


# --- Train Bayesian (once) ---
def train_bayesian():
    print("Training Bayesian...")
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


# === Train both models ===
xgb_probs = train_xgb()
bayes_probs = train_bayesian()

# === Sweep weights ===
y_onehot = np.zeros((len(actual), 6))
y_onehot[np.arange(len(actual)), actual] = 1

weights = np.arange(0.0, 1.01, 0.05)
results = []

for w_bayes in weights:
    w_xgb = 1.0 - w_bayes
    combo = w_bayes * bayes_probs + w_xgb * xgb_probs
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
        "w_bayes": w_bayes,
        "w_xgb": w_xgb,
        "logloss": ll,
        "brier": brier,
        **aucs,
    })

results_df = pd.DataFrame(results)

# === Print table ===
print("\n" + "=" * 110)
print("ENSEMBLE WEIGHT SWEEP (Log DC, WR Holdout n=88)")
print("=" * 110)
print(f"  {'Bayes':>6s} {'XGB':>6s}  {'LogLoss':>8s} {'Brier':>8s}  "
      f"{'>=Flex':>7s} {'>=Start':>7s} {'>=Elite':>7s} {'>=Stud':>7s} {'>=LW':>7s}")
print(f"  {'-'*6} {'-'*6}  {'-'*8} {'-'*8}  {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

best_ll_idx = results_df["logloss"].idxmin()
best_brier_idx = results_df["brier"].idxmin()
best_elite_idx = results_df[">=Elite"].idxmax()

for i, row in results_df.iterrows():
    markers = []
    if i == best_ll_idx:
        markers.append("LL")
    if i == best_brier_idx:
        markers.append("BR")
    if i == best_elite_idx:
        markers.append("EL")
    marker_str = f"  <-- best {','.join(markers)}" if markers else ""

    print(f"  {row['w_bayes']:>5.0%} {row['w_xgb']:>5.0%}  "
          f"{row['logloss']:>8.4f} {row['brier']:>8.4f}  "
          f"{row['>=Flex']:>7.3f} {row['>=Starter']:>7.3f} {row['>=Elite']:>7.3f} "
          f"{row['>=Stud']:>7.3f} {row['>=LW']:>7.3f}{marker_str}")

# === Composite ranking ===
# Weights: LogLoss 35%, >=Elite AUC 35%, Brier 15%, >=Starter AUC 10%, >=Stud AUC 5%
# For LogLoss/Brier, lower is better -> invert so higher = better for all metrics
COMPOSITE_WEIGHTS = {
    "logloss": 0.35,
    ">=Elite": 0.35,
    "brier": 0.15,
    ">=Starter": 0.10,
    ">=Stud": 0.05,
}

# Min-max normalize each metric to 0-1 (higher = better for all after normalization)
for col, weight in COMPOSITE_WEIGHTS.items():
    mn, mx = results_df[col].min(), results_df[col].max()
    if mx == mn:
        results_df[f"{col}_norm"] = 0.5
    elif col in ("logloss", "brier"):
        # Invert: lower raw = higher normalized
        results_df[f"{col}_norm"] = (mx - results_df[col]) / (mx - mn)
    else:
        results_df[f"{col}_norm"] = (results_df[col] - mn) / (mx - mn)

results_df["composite"] = sum(
    COMPOSITE_WEIGHTS[col] * results_df[f"{col}_norm"]
    for col in COMPOSITE_WEIGHTS
)
results_df["rank"] = results_df["composite"].rank(ascending=False).astype(int)
results_df = results_df.sort_values("composite", ascending=False)

print("\n" + "=" * 120)
print("COMPOSITE RANKING (35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC)")
print("=" * 120)
print(f"  {'Rank':>4s}  {'Bayes':>6s} {'XGB':>6s}  {'Composite':>9s}  "
      f"{'LogLoss':>8s} {'>=Elite':>7s} {'Brier':>8s} {'>=Start':>7s} {'>=Stud':>7s}")
print(f"  {'-'*4}  {'-'*6} {'-'*6}  {'-'*9}  {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

for _, row in results_df.iterrows():
    marker = ""
    if row["w_bayes"] == 0.75:
        marker = "  <-- current"
    elif row["rank"] == 1:
        marker = "  <-- BEST"
    print(f"  {int(row['rank']):>4d}  {row['w_bayes']:>5.0%} {row['w_xgb']:>5.0%}  "
          f"{row['composite']:>9.4f}  "
          f"{row['logloss']:>8.4f} {row['>=Elite']:>7.3f} {row['brier']:>8.4f} "
          f"{row['>=Starter']:>7.3f} {row['>=Stud']:>7.3f}{marker}")

best_row = results_df.loc[results_df["rank"] == 1].iloc[0]
current_row = results_df.loc[results_df["w_bayes"] == 0.75].iloc[0]
print(f"\nOptimal: {best_row['w_bayes']:.0%} Bayes / {best_row['w_xgb']:.0%} XGB  "
      f"(composite={best_row['composite']:.4f})")
print(f"Current: 75% Bayes / 25% XGB  "
      f"(composite={current_row['composite']:.4f}, rank #{int(current_row['rank'])})")


# === Visualization ===
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("WR Ensemble Weight Sweep — Log Draft Capital (Holdout n=88, 2022-2024)",
             fontsize=14, fontweight="bold", y=0.98)

# Sort back by weight for plotting
plot_df = results_df.sort_values("w_bayes")
bayes_pcts = plot_df["w_bayes"] * 100

# Panel 1: Composite score
ax = axes[0, 0]
colors = ["#2ca02c" if r == 1 else ("#d62728" if w == 0.75 else "#1f77b4")
          for r, w in zip(plot_df["rank"], plot_df["w_bayes"])]
bars = ax.bar(bayes_pcts, plot_df["composite"], width=4, color=colors,
              edgecolor="white", linewidth=0.5, alpha=0.85)

best_pct = best_row["w_bayes"] * 100
ax.axvline(best_pct, color="#2ca02c", linestyle=":", alpha=0.6)
ax.axvline(75, color="#d62728", linestyle="--", alpha=0.6, linewidth=1.5)

# Annotate best and current
for _, row in plot_df.iterrows():
    pct = row["w_bayes"] * 100
    if row["rank"] == 1:
        ax.annotate(f"#{int(row['rank'])} — {pct:.0f}/{100-pct:.0f}\n({row['composite']:.4f})",
                    xy=(pct, row["composite"]),
                    xytext=(pct + 10, row["composite"] + 0.02),
                    fontsize=8, fontweight="bold", color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
    elif row["w_bayes"] == 0.75:
        ax.annotate(f"current 75/25\n#{int(row['rank'])} ({row['composite']:.4f})",
                    xy=(pct, row["composite"]),
                    xytext=(pct - 18, row["composite"] - 0.06),
                    fontsize=8, fontweight="bold", color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("Composite Score (higher = better)", fontsize=10)
ax.set_title("Composite Ranking", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.2)

# Panel 2: LogLoss + Brier
ax = axes[0, 1]
color_ll = "#1f77b4"
color_br = "#ff7f0e"

ax.plot(bayes_pcts, plot_df["logloss"], "o-", color=color_ll, linewidth=2,
        markersize=4, label="LogLoss (35%)")
ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("LogLoss", fontsize=10, color=color_ll)
ax.tick_params(axis="y", labelcolor=color_ll)

ax2 = ax.twinx()
ax2.plot(bayes_pcts, plot_df["brier"], "s-", color=color_br, linewidth=2,
         markersize=4, label="Brier (15%)")
ax2.set_ylabel("Brier", fontsize=10, color=color_br)
ax2.tick_params(axis="y", labelcolor=color_br)

ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
ax.axvline(best_pct, color="#2ca02c", linestyle=":", alpha=0.4)
ax.set_title("Calibration Metrics (lower = better)", fontsize=11, fontweight="bold")
ax.grid(alpha=0.2)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

# Panel 3: Mid-tier AUCs
ax = axes[1, 0]
for label, color, lw in [(">=Elite", "#d62728", 2.5), (">=Starter", "#9467bd", 2), (">=Flex", "#2ca02c", 1.5)]:
    w = {">=Elite": "35%", ">=Starter": "10%", ">=Flex": "0%"}[label]
    ax.plot(bayes_pcts, plot_df[label], "o-", color=color, linewidth=lw,
            markersize=4, label=f"{label} ({w})")

ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
ax.axvline(best_pct, color="#2ca02c", linestyle=":", alpha=0.4)
ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("AUC", fontsize=10)
ax.set_title("Discrimination — Mid-Tier AUC (higher = better)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# Panel 4: Top-tier AUCs
ax = axes[1, 1]
for label, color in [(">=Stud", "#e377c2"), (">=LW", "#17becf")]:
    w = {">=Stud": "5%", ">=LW": "0%"}[label]
    ax.plot(bayes_pcts, plot_df[label], "o-", color=color, linewidth=2,
            markersize=4, label=f"{label} ({w})")

ax.axvline(75, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
ax.axvline(best_pct, color="#2ca02c", linestyle=":", alpha=0.4)
ax.set_xlabel("Bayesian Weight (%)", fontsize=10)
ax.set_ylabel("AUC", fontsize=10)
ax.set_title("Discrimination — Top-Tier AUC (noisy, small n)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# Footnote
fig.text(0.5, 0.01,
         "Composite = 35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC  "
         "(min-max normalized, higher = better)",
         ha="center", fontsize=9, color="gray", style="italic")

fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out_dir = os.path.join(DATA_DIR, "charts")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ensemble_weight_sweep.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved chart to {out_path}")
