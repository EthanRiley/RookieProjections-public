"""RB-specific model module: constants, composites, fallbacks, and training wrappers.

Delegates shared infrastructure (tier math, XGBoost, Bayesian, evaluation)
to base_model.py. All public names are re-exported for backward compatibility.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Re-export shared infrastructure so existing imports keep working
from modeling.base_model import (
    TIER_ORDER,
    TIER_NAMES,
    TIER_COLS,
    THRESHOLDS,
    THRESHOLD_LABELS,
    N_TIERS,
    N_CUTPOINTS,
    dc_log,
    cumulative_to_tier_probs,
    train_xgb,
    train_bayesian,
    predict_from_trace,
    evaluate,
    compute_metrics,
)
from modeling.base_model import blend as _blend
from modeling.base_model import build_pred_df as _build_pred_df
from modeling.base_model import train_full_and_college as _train_full_and_college

# --- RB-specific constants ---

COLLEGE_FEATURES = [
    "peak2_ypa",
    "composite_explosive",
    "composite_receiving",
    "peak_yac_per_att",
]

COMPOSITE_DEFS = {
    "composite_receiving": ["career_rec_yards_pg", "career_yprr", "career_grades_pass_route"],
    "composite_explosive": ["career_explosive_per_att", "best2_explosive_pg"],
}

W_BAYES = 0.45
W_XGB = 0.55


# --- Feature fallbacks ---

def apply_feature_fallbacks(df):
    """Fill missing peak2_ypa from peak_ypa (single-season fallback)."""
    if "peak_ypa" in df.columns:
        mask = df["peak2_ypa"].isna() & df["peak_ypa"].notna()
        df.loc[mask, "peak2_ypa"] = df.loc[mask, "peak_ypa"]
    return df


# --- Composites ---

def compute_composites(df, train_mask=None):
    """Compute z-scored composite features with proper leakage prevention.

    Args:
        df: DataFrame with raw component columns.
        train_mask: Boolean mask for training rows. If provided, z-score params
                    are fit on training data only. If None, fits on all data
                    (appropriate when all rows are training data).

    Returns:
        (df, scaler_dict) where scaler_dict maps composite name to fitted StandardScaler.
    """
    scaler_dict = {}
    for comp_name, input_feats in COMPOSITE_DEFS.items():
        available = [f for f in input_feats if f in df.columns]
        sub = df[available].copy()
        valid = sub.notna().all(axis=1)
        if valid.sum() < 5:
            df[comp_name] = np.nan
            continue

        if train_mask is not None:
            fit_mask = valid & train_mask
        else:
            fit_mask = valid

        scaler = StandardScaler()
        scaler.fit(sub[fit_mask])
        scaler_dict[comp_name] = scaler

        z = pd.DataFrame(
            scaler.transform(sub[valid]),
            index=sub[valid].index, columns=available,
        )
        df.loc[valid, comp_name] = z.mean(axis=1).round(4)
        df.loc[~valid, comp_name] = np.nan
    return df, scaler_dict


def apply_composites(df, scaler_dict):
    """Apply pre-fit scalers to compute composites for new data."""
    for comp_name, input_feats in COMPOSITE_DEFS.items():
        available = [f for f in input_feats if f in df.columns]
        sub = df[available].copy()
        valid = sub.notna().all(axis=1)
        scaler = scaler_dict.get(comp_name)
        if scaler is None or valid.sum() == 0:
            df[comp_name] = np.nan
            continue
        z = pd.DataFrame(
            scaler.transform(sub[valid]),
            index=sub[valid].index, columns=available,
        )
        df.loc[valid, comp_name] = z.mean(axis=1).round(4)
        df.loc[~valid, comp_name] = np.nan
    return df


# --- Composite optimization score ---

def composite_score(m):
    """Composite optimization score for weight grid search."""
    ll = m["log_loss"]
    elite = m.get(">=Elite_auc", m.get(">=Elite", 0.5))
    brier = m["brier"]
    starter = m.get(">=Starter_auc", m.get(">=Starter", 0.5))
    stud = m.get(">=Stud_auc", m.get(">=Stud", 0.5))
    return 0.35 * (1 - ll / 3.0) + 0.35 * elite + 0.15 * (1 - brier) + 0.10 * starter + 0.05 * stud


# --- RB wrappers with position-specific defaults ---

def blend(bayes_probs, xgb_probs, w_bayes=W_BAYES, w_xgb=W_XGB):
    """Weighted ensemble with RB default weights (45/55)."""
    return _blend(bayes_probs, xgb_probs, w_bayes, w_xgb)


def build_pred_df(base_df, full_probs, college_probs, components=None):
    """Build prediction DataFrame, including draft_capital."""
    return _build_pred_df(base_df, full_probs, college_probs, components=components,
                          extra_cols=["draft_capital"])


def train_full_and_college(train_df, pred_df, scaler=None, random_seed=42):
    """Train both full and college-only RB model variants.

    Returns:
        (full_probs, college_probs, scaler, components)
    """
    return _train_full_and_college(
        train_df, pred_df, COLLEGE_FEATURES, W_BAYES, W_XGB,
        scaler=scaler, random_seed=random_seed,
    )
