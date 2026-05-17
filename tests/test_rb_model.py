"""Tests for RB-specific model logic (composites, fallbacks, feature config).

Pins RB model behavior before the base_model refactor.
"""

import numpy as np
import pandas as pd
import pytest

from modeling.rb_model import (
    COLLEGE_FEATURES,
    COMPOSITE_DEFS,
    W_BAYES,
    W_XGB,
    apply_feature_fallbacks,
    compute_composites,
    apply_composites,
)


class TestRBConstants:
    def test_feature_count(self):
        assert len(COLLEGE_FEATURES) == 4

    def test_feature_names(self):
        assert "peak2_ypa" in COLLEGE_FEATURES
        assert "composite_explosive" in COLLEGE_FEATURES
        assert "composite_receiving" in COLLEGE_FEATURES
        assert "peak_yac_per_att" in COLLEGE_FEATURES

    def test_ensemble_weights(self):
        assert W_BAYES == 0.45
        assert W_XGB == 0.55
        assert abs(W_BAYES + W_XGB - 1.0) < 1e-10

    def test_composite_defs(self):
        assert "composite_receiving" in COMPOSITE_DEFS
        assert "composite_explosive" in COMPOSITE_DEFS
        assert len(COMPOSITE_DEFS["composite_receiving"]) == 3
        assert len(COMPOSITE_DEFS["composite_explosive"]) == 2


class TestFeatureFallbacks:
    def test_fills_missing_peak2_from_peak(self):
        df = pd.DataFrame({
            "peak2_ypa": [4.5, np.nan, 5.0],
            "peak_ypa": [4.3, 4.8, 4.9],
        })
        result = apply_feature_fallbacks(df)
        assert result["peak2_ypa"].iloc[1] == 4.8

    def test_preserves_existing_peak2(self):
        df = pd.DataFrame({
            "peak2_ypa": [4.5, 5.2, 5.0],
            "peak_ypa": [4.3, 4.8, 4.9],
        })
        result = apply_feature_fallbacks(df)
        assert result["peak2_ypa"].iloc[1] == 5.2

    def test_handles_both_missing(self):
        df = pd.DataFrame({
            "peak2_ypa": [4.5, np.nan, 5.0],
            "peak_ypa": [4.3, np.nan, 4.9],
        })
        result = apply_feature_fallbacks(df)
        assert pd.isna(result["peak2_ypa"].iloc[1])

    def test_handles_no_peak_column(self):
        df = pd.DataFrame({"peak2_ypa": [4.5, np.nan, 5.0]})
        result = apply_feature_fallbacks(df)
        assert pd.isna(result["peak2_ypa"].iloc[1])


class TestComputeComposites:
    def _make_df(self, n=30):
        np.random.seed(42)
        return pd.DataFrame({
            "career_rec_yards_pg": np.random.normal(40, 10, n),
            "career_yprr": np.random.normal(1.5, 0.3, n),
            "career_grades_pass_route": np.random.normal(65, 8, n),
            "career_explosive_per_att": np.random.normal(0.12, 0.03, n),
            "best2_explosive_pg": np.random.normal(2.0, 0.5, n),
        })

    def test_creates_composite_columns(self):
        df = self._make_df()
        result, scalers = compute_composites(df)
        assert "composite_receiving" in result.columns
        assert "composite_explosive" in result.columns

    def test_returns_scalers(self):
        df = self._make_df()
        _, scalers = compute_composites(df)
        assert "composite_receiving" in scalers
        assert "composite_explosive" in scalers

    def test_z_scored_mean_near_zero(self):
        df = self._make_df(100)
        result, _ = compute_composites(df)
        assert abs(result["composite_receiving"].mean()) < 0.2
        assert abs(result["composite_explosive"].mean()) < 0.2

    def test_train_mask_prevents_leakage(self):
        df = self._make_df(30)
        train_mask = pd.Series([True] * 20 + [False] * 10, index=df.index)

        _, scalers_all = compute_composites(df)
        _, scalers_train = compute_composites(df, train_mask=train_mask)

        # Scaler means should differ
        m_all = scalers_all["composite_receiving"].mean_
        m_train = scalers_train["composite_receiving"].mean_
        assert not np.allclose(m_all, m_train)

    def test_handles_nans(self):
        df = self._make_df(10)
        df.loc[0, "career_rec_yards_pg"] = np.nan
        result, _ = compute_composites(df)
        assert pd.isna(result["composite_receiving"].iloc[0])
        assert result["composite_receiving"].iloc[1:].notna().all()
        # Explosive should be unaffected
        assert result["composite_explosive"].notna().all()


class TestApplyComposites:
    def test_apply_matches_compute(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "career_rec_yards_pg": np.random.normal(40, 10, 20),
            "career_yprr": np.random.normal(1.5, 0.3, 20),
            "career_grades_pass_route": np.random.normal(65, 8, 20),
            "career_explosive_per_att": np.random.normal(0.12, 0.03, 20),
            "best2_explosive_pg": np.random.normal(2.0, 0.5, 20),
        })
        df_computed, scalers = compute_composites(df.copy())

        df_applied = apply_composites(df.copy(), scalers)

        np.testing.assert_allclose(
            df_computed["composite_receiving"].values,
            df_applied["composite_receiving"].values,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            df_computed["composite_explosive"].values,
            df_applied["composite_explosive"].values,
            atol=1e-10,
        )

    def test_apply_with_new_data(self):
        np.random.seed(42)
        train_df = pd.DataFrame({
            "career_rec_yards_pg": np.random.normal(40, 10, 20),
            "career_yprr": np.random.normal(1.5, 0.3, 20),
            "career_grades_pass_route": np.random.normal(65, 8, 20),
            "career_explosive_per_att": np.random.normal(0.12, 0.03, 20),
            "best2_explosive_pg": np.random.normal(2.0, 0.5, 20),
        })
        _, scalers = compute_composites(train_df)

        new_df = pd.DataFrame({
            "career_rec_yards_pg": [60.0, 20.0],
            "career_yprr": [2.0, 1.0],
            "career_grades_pass_route": [80.0, 50.0],
            "career_explosive_per_att": [0.18, 0.08],
            "best2_explosive_pg": [3.0, 1.0],
        })
        result = apply_composites(new_df, scalers)
        # Higher raw values → higher composite
        assert result["composite_receiving"].iloc[0] > result["composite_receiving"].iloc[1]
        assert result["composite_explosive"].iloc[0] > result["composite_explosive"].iloc[1]
