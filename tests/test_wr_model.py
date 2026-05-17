"""Tests for WR-specific model logic (catch composite, feature config).

Pins WR model behavior before the base_model refactor.
"""

import numpy as np
import pandas as pd
import pytest

from modeling.wr_model import (
    COLLEGE_FEATURES,
    CATCH_COMPOSITE_CPAA_WEIGHT,
    CATCH_COMPOSITE_CAREER_WEIGHT,
    W_BAYES,
    W_XGB,
    build_catch_composite,
    apply_catch_composite,
)


class TestWRConstants:
    def test_feature_count(self):
        assert len(COLLEGE_FEATURES) == 4

    def test_feature_names(self):
        assert "pg_yprr_graduated" in COLLEGE_FEATURES
        assert "catch_composite" in COLLEGE_FEATURES
        assert "best2_contested_catch_rate" in COLLEGE_FEATURES
        assert "best2_avoided_tackles_per_rec" in COLLEGE_FEATURES

    def test_ensemble_weights(self):
        assert W_BAYES == 0.50
        assert W_XGB == 0.50
        assert abs(W_BAYES + W_XGB - 1.0) < 1e-10

    def test_catch_composite_weights(self):
        assert CATCH_COMPOSITE_CPAA_WEIGHT == 0.67
        assert CATCH_COMPOSITE_CAREER_WEIGHT == 0.33
        assert abs(CATCH_COMPOSITE_CPAA_WEIGHT + CATCH_COMPOSITE_CAREER_WEIGHT - 1.0) < 1e-10


class TestBuildCatchComposite:
    def _make_df(self, n=20):
        np.random.seed(42)
        return pd.DataFrame({
            "pg_catch_pct_adot_adj_graduated": np.random.normal(65, 5, n),
            "career_catch_pct_adot_adj": np.random.normal(60, 4, n),
        })

    def test_returns_series_and_params(self):
        df = self._make_df()
        composite, z_params = build_catch_composite(df)
        assert isinstance(composite, pd.Series)
        assert isinstance(z_params, dict)
        assert len(composite) == len(df)

    def test_z_params_keys(self):
        df = self._make_df()
        _, z_params = build_catch_composite(df)
        assert "cpaa_mean" in z_params
        assert "cpaa_std" in z_params
        assert "career_mean" in z_params
        assert "career_std" in z_params

    def test_mean_near_zero(self):
        df = self._make_df(100)
        composite, _ = build_catch_composite(df)
        # z-scored composite should have mean near zero
        assert abs(composite.mean()) < 0.3

    def test_train_mask_prevents_leakage(self):
        df = self._make_df(30)
        train_mask = pd.Series([True] * 20 + [False] * 10, index=df.index)

        _, params_all = build_catch_composite(df)
        _, params_train = build_catch_composite(df, train_mask=train_mask)

        # Params should differ when train_mask excludes rows
        assert params_all["cpaa_mean"] != params_train["cpaa_mean"]

    def test_handles_nans(self):
        df = self._make_df(10)
        df.loc[0, "pg_catch_pct_adot_adj_graduated"] = np.nan
        composite, _ = build_catch_composite(df)
        assert pd.isna(composite.iloc[0])
        assert composite.iloc[1:].notna().all()


class TestApplyCatchComposite:
    def test_apply_matches_build(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "pg_catch_pct_adot_adj_graduated": np.random.normal(65, 5, 20),
            "career_catch_pct_adot_adj": np.random.normal(60, 4, 20),
        })
        composite_built, z_params = build_catch_composite(df)
        composite_applied = apply_catch_composite(df, z_params)
        np.testing.assert_allclose(
            composite_built.values, composite_applied.values, atol=1e-10
        )

    def test_apply_with_new_data(self):
        np.random.seed(42)
        train_df = pd.DataFrame({
            "pg_catch_pct_adot_adj_graduated": np.random.normal(65, 5, 20),
            "career_catch_pct_adot_adj": np.random.normal(60, 4, 20),
        })
        _, z_params = build_catch_composite(train_df)

        new_df = pd.DataFrame({
            "pg_catch_pct_adot_adj_graduated": [70.0, 55.0],
            "career_catch_pct_adot_adj": [65.0, 50.0],
        })
        result = apply_catch_composite(new_df, z_params)
        assert len(result) == 2
        assert result.notna().all()
        # Higher values should produce higher composite
        assert result.iloc[0] > result.iloc[1]
