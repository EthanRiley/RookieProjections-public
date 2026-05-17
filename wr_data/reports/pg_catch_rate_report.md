# Peak-Gated Age-Adjusted aDOT-Adjusted Catch Rate

**Date**: 2026-05-13
**Scripts**: `modeling/research/test_peak_gated_selection.py`, `test_cpaa_minus_drops.py`, `test_pg_yprr_swap.py`

---

## Executive Summary

Three engineering layers compound to produce the single strongest feature improvement found in this project:

1. **aDOT adjustment** — regress out depth-of-target difficulty from catch%
2. **Graduated age adjustment** — weight younger players' catch% more heavily (same multipliers as YPRR)
3. **Peak-gated season selection** — pick the season where catch% peaks, but only from seasons with PFF grades_offense >= 80

The result: `pg_catch_pct_adot_adj_graduated` replaces `career_targeted_qb_rating` and dominates v11 on every metric.

| Metric | v11 (6 feats) | QBR => pg_cpaa_grad (6 feats) | 5 feats (drop best2_cpaa) | Change vs v11 |
|--------|---------------|-------------------------------|---------------------------|---------------|
| **LogLoss** | 2.347 | **1.670** | 1.818 | **-29%** |
| **Brier** | 0.515 | **0.494** | **0.493** | **-4%** |
| **>=Elite AUC** | 0.842 | **0.863** | 0.860 | **+0.021** |
| **>=Stud AUC** | 0.778 | 0.780 | **0.793** | **+0.015** |
| **>=Starter AUC** | 0.833 | **0.849** | 0.837 | **+0.016** |

---

## 1. Engineering Progression

Each layer's contribution, measured by LogLoss when replacing QBR (keeping best2_catch_pct_adot_adj):

| Stage | Feature | LogLoss | Delta | % Change |
|-------|---------|---------|-------|----------|
| v11 baseline | career_targeted_qb_rating | 2.347 | — | — |
| + aDOT adjust | best1_catch_pct_adot_adj | 2.142 | -0.205 | -8.7% |
| + graduated age | best1_catch_pct_adot_adj_graduated | 1.972 | -0.170 | -7.9% |
| + peak-gated | pg_catch_pct_adot_adj_graduated | 1.670 | -0.302 | -15.3% |
| **Total** | | | **-0.677** | **-28.8%** |

The peak-gated selection step provides the largest single improvement (-15.3%), despite only changing the season selected for 15 of 237 players (6%).

---

## 2. Peak-Gated Selection Method

### Concept

**Current (`best1`)**: Select the season with the highest PFF offensive grade. Extract stats from that season.

**Peak-gated (`pg`)**: Select the season where a specific stat peaks, but only from seasons with `grades_offense >= 80`. If no season meets the threshold, fall back to `best1`.

**Pure peak**: Select the season where a stat is highest, regardless of grade. No quality gate.

### Why It Works

The quality gate is the key innovation. Pure peak overfits — it picks whichever season happened to have the highest stat value, including noisy low-grade seasons. Peak-gated constrains the search to quality seasons, then finds the stat-specific peak within that set.

Only 10.4% of qualified seasons (502/4844) have grades_offense >= 80. The gate is selective enough to filter noise while permissive enough to allow stat-specific selection.

### Selection Divergence

For `catch_pct_adot_adj`, peak-gated selects a different season than best1 for 15/237 players (6%). These edge cases disproportionately affect model performance because they're the players where overall grade and catch quality diverge — exactly where the model needs accurate signal.

### Three-Way Comparison (Full Model Base, 4 anchors)

| Method | Stat | LOO Delta | Era Drift | Collinearity |
|--------|------|-----------|-----------|--------------|
| **peak-gated** | cpaa_graduated | **+0.021** | **0.019** | 0.451 |
| best1 | cpaa_graduated | +0.017 | 0.028 | 0.436 |
| pure peak | cpaa_graduated | +0.003 | 0.088 | 0.458 |
| **peak-gated** | catch_minus_drops_grad | **+0.022** | **0.005** | 0.330 |
| best1 | catch_minus_drops_grad | +0.015 | 0.033 | 0.340 |
| pure peak | catch_minus_drops_grad | -0.002 | 0.074 | 0.273 |

Peak-gated wins on LOO-AUC delta for every stat tested. Pure peak consistently has the worst era drift, confirming it captures noise rather than signal.

---

## 3. The Graduated Age Adjustment

The same multipliers used for YPRR apply naturally to catch%:

| Age Class | Multiplier | Logic |
|-----------|------------|-------|
| Freshman (< 19.5) | +25% | Catching 70% aDOT-adjusted as a freshman is more impressive |
| Sophomore (19.5-20.5) | +5% | Slight boost |
| Junior (20.5-21.5) | -20% | Expected to perform at this level |
| Senior (>= 21.5) | -25% | Playing against younger competition |

Impact: best1_cpaa (LogLoss 2.142) → best1_cpaa_graduated (1.972), a 7.9% improvement. The adjustment rewards young breakouts and penalizes seniors who accumulated stats against less mature competition.

---

## 4. The best2_catch_pct_adot_adj Question

### The Problem

The v11 model has `best2_catch_pct_adot_adj` alongside `career_targeted_qb_rating`. When QBR is replaced with `pg_catch_pct_adot_adj_graduated`, we now have two aDOT-adjusted catch% features (Spearman ~0.774). Is this redundant?

### Evidence for Dropping It

| Signal | Value |
|--------|-------|
| Residual Spearman (vs 5-feature base) | -0.055 |
| Bootstrap % positive | 22.9% |
| Brier improves when dropped | 0.494 → 0.493 (90% of configs improve) |
| Stud AUC improves when dropped | 0.780 → 0.793 |

The feature acts as a **confidence amplifier** — two correlated features push probability mass toward sharper distributions. This helps LogLoss (unbounded reward for confident correct predictions) but hurts calibration (Brier). The Stud AUC improvement when dropping suggests multicollinearity-induced coefficient instability at the sparse tail of the tier distribution.

### Evidence for Keeping It

| Signal | Value |
|--------|-------|
| LogLoss improvement | 1.818 → 1.670 (-0.149) |
| Elite AUC improvement | 0.860 → 0.863 |
| Starter AUC improvement | 0.837 → 0.849 |

No other 6th feature comes close to this LogLoss improvement. The two features encode different aggregation windows — peak-gated best1 with age adjustment vs best2 averaged — and the ordinal model benefits from having both perspectives.

### Verdict

The decision is sharpness (LogLoss, keep it) vs calibration (Brier, drop it). For a dynasty draft tool where users need trustworthy probability distributions, calibration matters more. **Recommend dropping `best2_catch_pct_adot_adj`.**

---

## 5. Alternative 6th Feature: best1_grades_pass_route

If a 6th feature is desired, `best1_grades_pass_route` (PFF route-running technique grade) is the strongest candidate that isn't another catch% variant:

| Config | LogLoss | Brier | >=Elite | >=Stud |
|--------|---------|-------|---------|--------|
| 5-feature base | 1.818 | 0.493 | 0.860 | 0.793 |
| + best2_catch_pct_adot_adj | **1.670** | 0.494 | 0.863 | 0.780 |
| + best1_grades_pass_route | 1.812 | **0.483** | **0.870** | **0.810** |

`best1_grades_pass_route` measures route-running *technique* — mechanistically distinct from YPRR (production/efficiency) and catch% (reliability). It achieves the best Brier, Elite AUC, and Stud AUC of any configuration tested, with only a tiny LogLoss penalty vs the base.

---

## 6. Peak-Gated YPRR

Peak-gated selection was also tested for YPRR. Results are minimal because YPRR and overall grade are tightly coupled:

| Config | LogLoss | Brier | Elite | Stud | Starter |
|--------|---------|-------|-------|------|---------|
| v11 (best1_yprr_graduated) | 2.347 | 0.515 | 0.842 | 0.778 | 0.833 |
| pg_yprr_graduated | 2.288 | 0.514 | 0.842 | 0.779 | 0.833 |

Spearman correlation between best1 and pg YPRR graduated: **0.985**. Only 11/237 players (5%) get a different season. The user prefers peak-gated for conceptual consistency, and the metrics are essentially identical.

---

## 7. Proposed Feature Set

### Option A: 5 Features (Recommended)

| Feature | Selection | Dimension |
|---------|-----------|-----------|
| draft_capital | — | NFL talent consensus |
| pg_yprr_graduated | peak-gated | Peak route efficiency (age-adjusted) |
| pg_catch_pct_adot_adj_graduated | peak-gated | Peak catching ability (age+aDOT adjusted) |
| best2_contested_catch_rate | best2 by grade | Contested ball skills |
| best2_avoided_tackles_per_rec | best2 by grade | Post-catch elusiveness |

Better calibrated (Brier 0.493, Stud AUC 0.793). Fewer features, less overfitting risk.

### Option B: 6 Features (with grades_pass_route)

| Feature | Selection | Dimension |
|---------|-----------|-----------|
| draft_capital | — | NFL talent consensus |
| pg_yprr_graduated | peak-gated | Peak route efficiency (age-adjusted) |
| pg_catch_pct_adot_adj_graduated | peak-gated | Peak catching ability (age+aDOT adjusted) |
| best1_grades_pass_route | best1 by grade | Route-running technique |
| best2_contested_catch_rate | best2 by grade | Contested ball skills |
| best2_avoided_tackles_per_rec | best2 by grade | Post-catch elusiveness |

Best discrimination (Elite AUC 0.870, Stud AUC 0.810, Brier 0.483).

---

## 8. Visualizations

| File | Description |
|------|-------------|
| `wr_data/charts/pg_cpaa_progression.png` | Engineering progression: cumulative LogLoss improvement 2.347 → 1.670 |
| `wr_data/charts/pg_cpaa_selection_methods.png` | best1 vs peak-gated vs pure peak across LOO-AUC, era drift, collinearity |
| `wr_data/charts/pg_cpaa_combo_dashboard.png` | Multi-metric dashboard across all key configurations |
| `wr_data/charts/pg_cpaa_7part.png` | 7-part analysis for catch_pct_adot_adj family |
| `wr_data/charts/peak_gated_selection.png` | Auto-generated 6-panel analysis from test script |

## 9. Data Files

| File | Description |
|------|-------------|
| `wr_data/outputs/peak_gated_combos.csv` | All catch metric combination results |
| `wr_data/outputs/peak_gated_full.csv` | Full model base 7-part results |
| `wr_data/outputs/peak_gated_dc.csv` | DC-only base 7-part results |
| `wr_data/outputs/pg_yprr_swap_combos.csv` | YPRR swap combination results |
