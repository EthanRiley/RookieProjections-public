"""Tests for shared model logic (draft capital, tier math, blending, metrics).

These tests pin the behavior of functions shared between WR and RB models.
Run before and after the base_model refactor to verify nothing breaks.
"""

import numpy as np
import pandas as pd
import pytest

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
    blend,
    compute_metrics,
    evaluate,
    build_pred_df,
)


# --- Constants ---

class TestConstants:
    def test_tier_order_values(self):
        assert TIER_ORDER == {
            "Bust": 0, "Flex": 1, "Starter": 2,
            "Elite": 3, "Stud": 4, "League-Winner": 5,
        }

    def test_tier_names_is_inverse(self):
        for name, idx in TIER_ORDER.items():
            assert TIER_NAMES[idx] == name

    def test_tier_cols_count(self):
        assert len(TIER_COLS) == N_TIERS == 6

    def test_thresholds(self):
        assert THRESHOLDS == [1, 2, 3, 4, 5]
        assert len(THRESHOLD_LABELS) == len(THRESHOLDS)

    def test_n_cutpoints(self):
        assert N_CUTPOINTS == N_TIERS - 1 == 5


# --- Draft capital ---

class TestDraftCapital:
    def test_pick_1_highest(self):
        assert dc_log(1) > 8.5

    def test_pick_260_near_zero(self):
        assert dc_log(260) < 0.2

    def test_monotonically_decreasing(self):
        vals = [dc_log(p) for p in range(1, 261)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_never_negative(self):
        for p in range(1, 300):
            assert dc_log(p) >= 0

    def test_known_values(self):
        # Pick 1 should be about 8.75
        assert abs(dc_log(1) - 8.753) < 0.01
        # Pick 32 (end of round 1) should be about 3.75
        assert abs(dc_log(32) - 3.721) < 0.05


# --- Cumulative to tier conversion ---

class TestCumulativeToTierProbs:
    def test_output_shape(self):
        cum = np.array([[0.9, 0.7, 0.4, 0.2, 0.05]])
        result = cumulative_to_tier_probs(cum)
        assert result.shape == (1, N_TIERS)

    def test_sums_to_one(self):
        cum = np.array([[0.95, 0.8, 0.5, 0.3, 0.1],
                        [0.7, 0.5, 0.3, 0.1, 0.02]])
        result = cumulative_to_tier_probs(cum)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_all_nonnegative(self):
        cum = np.array([[0.99, 0.9, 0.6, 0.3, 0.05]])
        result = cumulative_to_tier_probs(cum)
        assert (result >= 0).all()

    def test_certain_bust(self):
        # Very low cumulative probs → mostly bust
        cum = np.array([[0.05, 0.01, 0.001, 0.0001, 0.00001]])
        result = cumulative_to_tier_probs(cum)
        assert result[0, 0] > 0.9  # P(Bust) should dominate

    def test_certain_league_winner(self):
        # Very high cumulative probs → mostly league-winner
        cum = np.array([[0.999, 0.999, 0.999, 0.999, 0.99]])
        result = cumulative_to_tier_probs(cum)
        assert result[0, 5] > 0.9  # P(League-Winner) should dominate

    def test_monotonicity_enforced(self):
        # Even with non-monotonic input, output should be valid probs
        cum = np.array([[0.8, 0.85, 0.5, 0.3, 0.1]])  # 0.85 > 0.8 violates
        result = cumulative_to_tier_probs(cum)
        assert (result >= 0).all()
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_batch(self):
        cum = np.random.rand(20, 5)
        cum.sort(axis=1)
        cum = cum[:, ::-1]  # Descending
        result = cumulative_to_tier_probs(cum)
        assert result.shape == (20, N_TIERS)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


# --- Blend ---

class TestBlend:
    def test_equal_weights(self):
        a = np.array([[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]])
        b = np.array([[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]])
        result = blend(a, b, w_bayes=0.5, w_xgb=0.5)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_full_weight_to_bayes(self):
        a = np.array([[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]])
        b = np.array([[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]])
        result = blend(a, b, w_bayes=1.0, w_xgb=0.0)
        np.testing.assert_allclose(result, a / a.sum(axis=1, keepdims=True), atol=1e-6)

    def test_normalizes(self):
        a = np.array([[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]])
        b = np.array([[0.1, 0.3, 0.3, 0.15, 0.1, 0.05]])
        result = blend(a, b, w_bayes=0.45, w_xgb=0.55)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_batch(self):
        n = 10
        a = np.random.dirichlet(np.ones(N_TIERS), n)
        b = np.random.dirichlet(np.ones(N_TIERS), n)
        result = blend(a, b, w_bayes=0.5, w_xgb=0.5)
        assert result.shape == (n, N_TIERS)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


# --- Compute metrics ---

class TestComputeMetrics:
    def _make_perfect_probs(self, y_true):
        """Create one-hot probs matching y_true exactly."""
        probs = np.zeros((len(y_true), N_TIERS))
        probs[np.arange(len(y_true)), y_true] = 1.0
        return probs

    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 3, 4, 5])
        probs = self._make_perfect_probs(y)
        # Nudge away from 0/1 to avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        probs = probs / probs.sum(axis=1, keepdims=True)
        m = compute_metrics(probs, y)
        assert m["log_loss"] < 0.01
        assert m["brier"] < 0.01

    def test_returns_expected_keys(self):
        y = np.array([0, 1, 2, 3, 4, 5])
        probs = np.random.dirichlet(np.ones(N_TIERS), 6)
        m = compute_metrics(probs, y)
        assert "log_loss" in m
        assert "brier" in m
        for label in THRESHOLD_LABELS:
            assert f"{label}_auc" in m

    def test_random_worse_than_informed(self):
        np.random.seed(42)
        y = np.array([0, 0, 0, 1, 2, 3, 4, 5])
        random_probs = np.random.dirichlet(np.ones(N_TIERS), len(y))
        informed_probs = np.zeros((len(y), N_TIERS))
        informed_probs[np.arange(len(y)), y] = 0.8
        informed_probs += 0.04
        informed_probs = informed_probs / informed_probs.sum(axis=1, keepdims=True)
        m_random = compute_metrics(random_probs, y)
        m_informed = compute_metrics(informed_probs, y)
        assert m_informed["log_loss"] < m_random["log_loss"]
        assert m_informed["brier"] < m_random["brier"]


# --- Build pred df ---

class TestBuildPredDf:
    def _make_base_df(self, n=5):
        return pd.DataFrame({
            "name": [f"Player_{i}" for i in range(n)],
            "draft_year": [2025] * n,
            "pick": list(range(1, n + 1)),
            "computed_tier": [0, 1, 2, 3, 4],
            "draft_age": [21.0 + i * 0.5 for i in range(n)],
            "draft_capital": [dc_log(p) for p in range(1, n + 1)],
        })

    def test_output_columns(self):
        base = self._make_base_df()
        probs = np.random.dirichlet(np.ones(N_TIERS), 5)
        college = np.random.dirichlet(np.ones(N_TIERS), 5)
        out = build_pred_df(base, probs, college)
        for col in TIER_COLS:
            assert col in out.columns
        assert "expected_tier" in out.columns
        assert "edge" in out.columns
        assert "college_expected_tier" in out.columns

    def test_sorted_by_expected_tier(self):
        base = self._make_base_df()
        probs = np.random.dirichlet(np.ones(N_TIERS), 5)
        college = np.random.dirichlet(np.ones(N_TIERS), 5)
        out = build_pred_df(base, probs, college)
        assert out["expected_tier"].is_monotonic_decreasing

    def test_probs_sum_to_one(self):
        base = self._make_base_df()
        probs = np.random.dirichlet(np.ones(N_TIERS), 5)
        college = np.random.dirichlet(np.ones(N_TIERS), 5)
        out = build_pred_df(base, probs, college)
        row_sums = out[TIER_COLS].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_edge_is_college_minus_full(self):
        base = self._make_base_df()
        probs = np.random.dirichlet(np.ones(N_TIERS), 5)
        college = np.random.dirichlet(np.ones(N_TIERS), 5)
        out = build_pred_df(base, probs, college)
        expected_edge = out["college_expected_tier"] - out["expected_tier"]
        np.testing.assert_allclose(out["edge"], expected_edge, atol=0.01)

    def test_components_included(self):
        base = self._make_base_df()
        probs = np.random.dirichlet(np.ones(N_TIERS), 5)
        college = np.random.dirichlet(np.ones(N_TIERS), 5)
        components = {"xgb_full": probs, "bayes_full": college}
        out = build_pred_df(base, probs, college, components=components)
        assert "xgb_full_P(Bust)" in out.columns
        assert "bayes_full_P(Bust)" in out.columns
