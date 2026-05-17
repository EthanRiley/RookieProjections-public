"""WR-specific model module: constants, catch composite, and training wrappers.

Delegates shared infrastructure (tier math, XGBoost, Bayesian, evaluation)
to base_model.py. All public names are re-exported for backward compatibility.
"""

import numpy as np
import pandas as pd

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

# --- WR-specific constants ---

COLLEGE_FEATURES = [
    "pg_yprr_graduated",
    "catch_composite",
    "best2_contested_catch_rate",
    "best2_avoided_tackles_per_rec",
]

CATCH_COMPOSITE_CPAA_WEIGHT = 0.67
CATCH_COMPOSITE_CAREER_WEIGHT = 0.33

W_BAYES = 0.50
W_XGB = 0.50


# --- Catch composite ---

def build_catch_composite(df, train_mask=None):
    """Build catch_composite column with proper z-score handling.

    Args:
        df: DataFrame with pg_catch_pct_adot_adj_graduated and career_catch_pct_adot_adj columns.
        train_mask: Boolean mask for training rows. If provided, z-score params are fit
                    on training data only (preventing holdout leakage). If None, uses
                    all non-NaN rows (appropriate when all rows are training data).

    Returns:
        (composite_series, z_params) where z_params is a dict with means/stds for
        reuse on prospect data.
    """
    cpaa_cols = ["pg_catch_pct_adot_adj_graduated", "career_catch_pct_adot_adj"]
    valid = df.dropna(subset=cpaa_cols)

    if train_mask is not None:
        train_valid = valid[train_mask.reindex(valid.index, fill_value=False)]
    else:
        train_valid = valid

    cpaa_mean = train_valid["pg_catch_pct_adot_adj_graduated"].mean()
    cpaa_std = train_valid["pg_catch_pct_adot_adj_graduated"].std()
    career_mean = train_valid["career_catch_pct_adot_adj"].mean()
    career_std = train_valid["career_catch_pct_adot_adj"].std()

    composite = pd.Series(np.nan, index=df.index)
    composite.loc[valid.index] = (
        CATCH_COMPOSITE_CPAA_WEIGHT * (valid["pg_catch_pct_adot_adj_graduated"] - cpaa_mean) / cpaa_std
        + CATCH_COMPOSITE_CAREER_WEIGHT * (valid["career_catch_pct_adot_adj"] - career_mean) / career_std
    )

    z_params = {
        "cpaa_mean": cpaa_mean, "cpaa_std": cpaa_std,
        "career_mean": career_mean, "career_std": career_std,
    }
    return composite, z_params


def apply_catch_composite(df, z_params):
    """Apply pre-computed z-score params to build catch_composite for new data."""
    cpaa_cols = ["pg_catch_pct_adot_adj_graduated", "career_catch_pct_adot_adj"]
    valid = df.dropna(subset=cpaa_cols)
    composite = pd.Series(np.nan, index=df.index)
    composite.loc[valid.index] = (
        CATCH_COMPOSITE_CPAA_WEIGHT * (valid["pg_catch_pct_adot_adj_graduated"] - z_params["cpaa_mean"]) / z_params["cpaa_std"]
        + CATCH_COMPOSITE_CAREER_WEIGHT * (valid["career_catch_pct_adot_adj"] - z_params["career_mean"]) / z_params["career_std"]
    )
    return composite


# --- WR wrappers with position-specific defaults ---

def blend(bayes_probs, xgb_probs, w_bayes=W_BAYES, w_xgb=W_XGB):
    """Weighted ensemble with WR default weights (50/50)."""
    return _blend(bayes_probs, xgb_probs, w_bayes, w_xgb)


def build_pred_df(base_df, full_probs, college_probs, components=None):
    """Build prediction DataFrame, including draft_age if available."""
    extra_cols = ["draft_age"] if "draft_age" in base_df.columns else None
    return _build_pred_df(base_df, full_probs, college_probs, components=components,
                          extra_cols=extra_cols)


def train_full_and_college(train_df, pred_df, scaler=None, random_seed=42):
    """Train both full and college-only WR model variants.

    Returns:
        (full_probs, college_probs, scaler, components)
    """
    return _train_full_and_college(
        train_df, pred_df, COLLEGE_FEATURES, W_BAYES, W_XGB,
        scaler=scaler, random_seed=random_seed,
    )
