"""Position-agnostic model base: shared constants, training, prediction, and evaluation.

All position-specific logic (features, weights, composites) lives in the
position modules (wr_model.py, rb_model.py). This module provides the
shared ordinal classification infrastructure.
"""

import math
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

# --- Constants (same for all positions) ---

TIER_ORDER = {
    "Bust": 0, "Flex": 1, "Starter": 2,
    "Elite": 3, "Stud": 4, "League-Winner": 5,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}
TIER_COLS = ["P(Bust)", "P(Flex)", "P(Starter)", "P(Elite)", "P(Stud)", "P(League-Winner)"]
THRESHOLDS = [1, 2, 3, 4, 5]
THRESHOLD_LABELS = [">=Flex", ">=Starter", ">=Elite", ">=Stud", ">=LW"]
N_TIERS = 6
N_CUTPOINTS = N_TIERS - 1


# --- Draft capital ---

def dc_log(pick):
    """Log-scaled draft capital: pick 1 ~ 8.7, pick 260 ~ 0."""
    return max(10 - (10 / math.log(261)) * math.log(pick + 1), 0)


# --- Cumulative-to-tier conversion ---

def cumulative_to_tier_probs(cum_probs):
    """Convert K-1 cumulative probabilities to K tier probabilities."""
    n = cum_probs.shape[0]
    tier_probs = np.zeros((n, N_TIERS))
    tier_probs[:, 0] = 1 - cum_probs[:, 0]
    for i in range(len(THRESHOLDS) - 1):
        tier_probs[:, THRESHOLDS[i]] = cum_probs[:, i] - cum_probs[:, i + 1]
    tier_probs[:, 5] = cum_probs[:, -1]
    tier_probs = np.clip(tier_probs, 0, 1)
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    return tier_probs


# --- XGBoost training + prediction ---

def train_xgb(X_train, y_train, X_pred, random_state=42):
    """Train K-1 binary XGBoost classifiers with Platt calibration.

    Returns tier probability array (n_pred, N_TIERS).
    """
    cum_probs = np.zeros((len(X_pred), len(THRESHOLDS)))
    for t_idx, threshold in enumerate(THRESHOLDS):
        y_bin = (y_train >= threshold).astype(int)
        pos = y_bin.sum()
        if pos == 0 or pos == len(y_bin):
            cum_probs[:, t_idx] = 0.5
            continue
        scale = (len(y_bin) - pos) / max(pos, 1)

        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale, random_state=random_state, eval_metric="logloss",
        )
        min_class = min(y_bin.sum(), len(y_bin) - y_bin.sum())
        cv_folds = min(5, max(2, min_class))
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds)
        calibrated.fit(X_train, y_bin)
        cum_probs[:, t_idx] = calibrated.predict_proba(X_pred)[:, 1]

    # Enforce monotonicity
    for i in range(len(THRESHOLDS) - 1, 0, -1):
        cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])

    return cumulative_to_tier_probs(cum_probs)


# --- Bayesian training + prediction ---

def train_bayesian(X_college_train, dc_train, y_train,
                   X_college_pred, dc_pred, use_dc, random_seed=42):
    """Train Bayesian ordinal model and predict tier probabilities."""
    n_college = X_college_train.shape[1]

    with pm.Model() as model:
        beta_college = pm.Normal("beta_college", mu=0.0, sigma=0.5, shape=n_college)
        eta = pt.dot(X_college_train, beta_college)
        if use_dc:
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
            random_seed=random_seed, progressbar=True, target_accept=0.9,
        )

    return predict_from_trace(trace, X_college_pred, dc_pred, n_college)


def predict_from_trace(trace, X_college_pred, dc_pred, n_college):
    """Generate tier probabilities from a PyMC trace."""
    beta_college_samples = trace.posterior["beta_college"].values.reshape(-1, n_college)
    cutpoints_samples = trace.posterior["cutpoints"].values.reshape(-1, N_CUTPOINTS)
    n_samples = len(cutpoints_samples)
    n_obs = X_college_pred.shape[0]
    tier_probs = np.zeros((n_obs, N_TIERS))

    has_dc = "beta_dc" in trace.posterior
    if has_dc:
        beta_dc_samples = trace.posterior["beta_dc"].values.flatten()

    for i in range(n_samples):
        eta = X_college_pred @ beta_college_samples[i]
        if has_dc:
            eta = eta + beta_dc_samples[i] * dc_pred
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


# --- Ensemble ---

def blend(bayes_probs, xgb_probs, w_bayes, w_xgb):
    """Weighted ensemble of Bayesian and XGBoost tier probabilities."""
    combo = w_bayes * bayes_probs + w_xgb * xgb_probs
    combo = combo / combo.sum(axis=1, keepdims=True)
    return combo


# --- Evaluation ---

def evaluate(probs, y_true, label=""):
    """Print detailed evaluation metrics. Returns (log_loss, brier, aucs)."""
    print(f"\n  {label}")
    print(f"  {'Threshold':<15s} {'AUC':>8s} {'Brier':>8s} {'Pos rate':>10s}")
    aucs = {}
    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        brier = brier_score_loss(y_bin, pred)
        aucs[tlabel] = auc
        print(f"  {tlabel:<15s} {auc:>8.3f} {brier:>8.4f} {y_bin.mean():>10.1%}")

    y_onehot = np.zeros((len(y_true), N_TIERS))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))
    print(f"\n  Multi-class log loss:  {ll:.4f}")
    print(f"  Multi-class Brier:     {brier:.4f}")
    return ll, brier, aucs


def compute_metrics(probs, y_true):
    """Compute metrics without printing. Returns dict."""
    y_onehot = np.zeros((len(y_true), N_TIERS))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_onehot, probs)
    brier = np.mean(np.sum((y_onehot - probs) ** 2, axis=1))

    aucs = {}
    for threshold, tlabel in zip(THRESHOLDS, THRESHOLD_LABELS):
        y_bin = (y_true >= threshold).astype(int)
        pred = probs[:, threshold:].sum(axis=1)
        aucs[tlabel] = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float("nan")

    return {"log_loss": ll, "brier": brier, **{f"{k}_auc": v for k, v in aucs.items()}}


# --- Output builders ---

def build_pred_df(base_df, full_probs, college_probs, components=None, extra_cols=None):
    """Build prediction output DataFrame with full + college-only predictions and edge.

    Args:
        base_df: DataFrame with name, draft_year, pick, computed_tier columns.
        full_probs: Full model tier probabilities (n, N_TIERS).
        college_probs: College-only tier probabilities (n, N_TIERS).
        components: Optional dict of {prefix: probs} for component models.
        extra_cols: Optional list of additional columns to include from base_df.
    """
    out = base_df[["name", "draft_year", "pick", "computed_tier"]].copy()

    if extra_cols:
        for col in extra_cols:
            if col in base_df.columns:
                out[col] = base_df[col].values

    for i, tier_name in TIER_NAMES.items():
        out[f"P({tier_name})"] = full_probs[:, i].round(3)
    out["predicted_tier"] = [TIER_NAMES[i] for i in full_probs.argmax(axis=1)]
    out["expected_tier"] = sum(full_probs[:, i] * i for i in range(N_TIERS))

    for i, tier_name in TIER_NAMES.items():
        out[f"college_P({tier_name})"] = college_probs[:, i].round(3)
    out["college_predicted_tier"] = [TIER_NAMES[i] for i in college_probs.argmax(axis=1)]
    out["college_expected_tier"] = sum(college_probs[:, i] * i for i in range(N_TIERS))

    out["edge"] = (out["college_expected_tier"] - out["expected_tier"]).round(3)

    if components is not None:
        for prefix, probs in components.items():
            for i, tier_name in TIER_NAMES.items():
                out[f"{prefix}_P({tier_name})"] = probs[:, i].round(3)

    out = out.sort_values("expected_tier", ascending=False).reset_index(drop=True)
    return out


# --- Training pipeline ---

def train_full_and_college(train_df, pred_df, college_features, w_bayes, w_xgb,
                           scaler=None, random_seed=42):
    """Train both full (DC + college) and college-only model variants.

    Args:
        train_df: Training DataFrame with all features + tier_ordinal.
        pred_df: Prediction DataFrame with all features.
        college_features: List of college feature column names.
        w_bayes: Bayesian ensemble weight.
        w_xgb: XGBoost ensemble weight.
        scaler: Optional pre-fit StandardScaler. If None, fits on train_df.
        random_seed: Random seed for reproducibility.

    Returns:
        (full_probs, college_probs, scaler, components)
    """
    all_features = ["draft_capital"] + college_features

    if scaler is None:
        scaler = StandardScaler()
        X_college_train = scaler.fit_transform(train_df[college_features].values)
    else:
        X_college_train = scaler.transform(train_df[college_features].values)

    X_college_pred = scaler.transform(pred_df[college_features].values)
    y_train = train_df["tier_ordinal"].values

    # XGBoost
    print(f"\nTraining XGBoost Full...")
    xgb_full = train_xgb(train_df[all_features].values, y_train,
                          pred_df[all_features].values, random_state=random_seed)
    print(f"Training XGBoost College-Only...")
    xgb_college = train_xgb(train_df[college_features].values, y_train,
                             pred_df[college_features].values, random_state=random_seed)

    # Bayesian
    print(f"\nTraining Bayesian Full...")
    bayes_full = train_bayesian(
        X_college_train, train_df["draft_capital"].values, y_train,
        X_college_pred, pred_df["draft_capital"].values, True,
        random_seed=random_seed,
    )
    print(f"\nTraining Bayesian College-Only...")
    bayes_college = train_bayesian(
        X_college_train, None, y_train,
        X_college_pred, None, False,
        random_seed=random_seed,
    )

    full_probs = blend(bayes_full, xgb_full, w_bayes, w_xgb)
    college_probs = blend(bayes_college, xgb_college, w_bayes, w_xgb)

    components = {
        "xgb_full": xgb_full,
        "xgb_college": xgb_college,
        "bayes_full": bayes_full,
        "bayes_college": bayes_college,
    }

    return full_probs, college_probs, scaler, components
