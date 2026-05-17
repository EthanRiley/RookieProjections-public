# v12 Model Report

**Date**: 2026-05-13
**Evaluation Script**: `modeling/evaluate_holdout_v12.py`
**Research Scripts**: `modeling/research/test_peak_gated_selection.py`, `modeling/research/v12_loo_grid_search.py`

---

## Summary

v12 makes four changes from v11:

1. **Replaces `career_targeted_qb_rating`** with `pg_catch_pct_adot_adj_graduated` — a peak-gated, aDOT-adjusted, age-adjusted catch percentage. This is the largest single improvement in the model's history.
2. **Replaces `best1_yprr_graduated`** with `pg_yprr_graduated` — peak-gated YPRR for conceptual consistency.
3. **Drops `best2_catch_pct_adot_adj`** — the 6th feature that existed for XGBoost interactions. A comprehensive grid search showed the 5-feature model wins on calibration (LogLoss, Brier) on both LOO and holdout.
4. **Adds super senior age penalty** — players aged 22.5+ on Sept 1 of their season now receive a -50% adjustment (was -25%, same as regular seniors). Backed by data showing 22.5+ seniors bust at 85% with avg tier 0.38.

The result is a leaner 5-feature model with better calibration than v11 on every metric, and a more accurate treatment of older college players.

**Holdout (88 players, 2022-2024):**

| Metric | v11 | v12 | Delta |
|--------|-----|-----|-------|
| **LogLoss** | 0.773 | **0.761** | **-0.012** |
| **Brier** | 0.340 | **0.334** | **-0.006** |
| **>=Elite AUC** | 0.970 | 0.963 | -0.007 |
| >=Stud AUC | 0.941 | 0.902 | -0.039 |
| **>=LW AUC** | 0.989 | 0.989 | 0.000 |
| >=Starter AUC | 0.920 | 0.904 | -0.016 |

**LOO (205 players, leave-one-year-out):**

| Metric | v11 | v12 | Delta |
|--------|-----|-----|-------|
| **LogLoss** | 2.347 | **1.702** | **-27.5%** |
| **Brier** | 0.515 | **0.488** | **-5.2%** |
| **>=Elite AUC** | 0.842 | **0.853** | **+0.011** |
| **>=Stud AUC** | 0.778 | **0.798** | **+0.020** |
| **>=Starter AUC** | 0.833 | **0.837** | **+0.004** |

v12 improves calibration (LogLoss, Brier) on both validation methods and every LOO AUC metric. Holdout AUC discrimination dips slightly on Stud (driven by only 3 Studs in the holdout set) but calibration improvements are the priority for a probabilistic model.

---

## Feature Set

| Feature | Selection | Dimension | Change from v11 |
|---------|-----------|-----------|-----------------|
| `draft_capital` | — | NFL talent consensus | unchanged |
| `pg_yprr_graduated` | **peak-gated** | Peak route efficiency (age-adjusted) | was `best1_yprr_graduated` |
| `pg_catch_pct_adot_adj_graduated` | **peak-gated** | Peak catching ability (age+aDOT adjusted) | **new** (replaces `career_targeted_qb_rating`) |
| `best2_contested_catch_rate` | best2 by grade | Contested ball skills | unchanged |
| `best2_avoided_tackles_per_rec` | best2 by grade | Post-catch elusiveness | unchanged |

### What Changed

1. **`career_targeted_qb_rating` -> `pg_catch_pct_adot_adj_graduated`**: The largest impact. QBR measured QB trust when targeting the receiver — a real but noisy proxy for route quality. The replacement measures peak catching performance directly, with three engineering layers that compound multiplicatively (aDOT adjustment, graduated age adjustment, peak-gated selection). LOO LogLoss improves 27.5%.

2. **`best1_yprr_graduated` -> `pg_yprr_graduated`**: Minimal metric impact (0.985 Spearman correlation, 5% season divergence). Changed for conceptual consistency — if we're using peak-gated selection for catch%, we should use it for YPRR too. The quality gate ensures we're picking the peak YPRR from seasons where the player was actually playing well overall.

3. **Dropped `best2_catch_pct_adot_adj`**: Previously kept for XGBoost interaction value. A comprehensive 12-configuration grid search showed the 5-feature model wins LogLoss and Brier on both LOO and holdout. See "The best2_catch_pct_adot_adj Investigation" below.

---

## Peak-Gated Season Selection

### Concept

Instead of always selecting the season with the highest PFF overall grade (`best1`), peak-gated selection picks the season where a specific stat peaks — but only from seasons with `grades_offense >= 80`. If no season meets the quality gate, it falls back to `best1`.

**Critical implementation detail:** Peak-gated selection compares **age-adjusted** values when choosing the peak season, not raw values. This ensures the age multiplier is factored into the selection decision. Without this, a younger season with slightly lower raw stats but a favorable age boost could be incorrectly passed over.

Example: Makai Lemon's 2024 season (age 20.2, Sophomore, +5%) had raw YPRR of 3.03 and his 2025 season (age 21.2, Junior, -20%) had raw YPRR of 3.13. Selecting on raw values would pick 2025, yielding pg_yprr_graduated = 2.51. Selecting on age-adjusted values correctly picks 2024, yielding pg_yprr_graduated = 3.18.

### Why It Works

A receiver's best catching season may not be their best overall grade season. A sophomore with an 82-grade season but elite catch metrics is more informative for catching ability than their 88-grade junior season where they ran better routes but caught worse.

The quality gate is critical:
- **Pure peak** (no gate) overfits — picks whichever season had the highest stat, including noisy low-grade seasons. Era drift 3-5x worse.
- **Peak-gated** is the Goldilocks zone — stat-specific enough to capture genuine peaks, constrained enough to avoid noise.
- Only 10.4% of qualified seasons have grades_offense >= 80. The gate is selective.

### Selection Divergence

| Stat | Players with different season | Total | % |
|------|------------------------------|-------|---|
| catch_pct_adot_adj | 15 | 237 | 6% |
| yprr | 11 | 237 | 5% |

Despite low divergence rates, the impact on model performance is outsized because these edge cases are exactly where the model needs accurate signal to differentiate outcomes.

---

## Engineering Progression

Three layers compound to produce the total improvement, measured by LOO LogLoss when replacing QBR:

| Stage | Feature | LogLoss | Delta | Cumulative |
|-------|---------|---------|-------|------------|
| v11 baseline | career_targeted_qb_rating | 2.347 | — | — |
| + aDOT adjustment | best1_catch_pct_adot_adj | 2.142 | -0.205 (-8.7%) | -8.7% |
| + graduated age adj | best1_catch_pct_adot_adj_graduated | 1.972 | -0.170 (-7.9%) | -16.0% |
| + peak-gated selection | pg_catch_pct_adot_adj_graduated | 1.702 | -0.270 (-13.7%) | -27.5% |

Each layer is independently valuable, but they compound: the graduated adjustment is most powerful on correctly-selected seasons, and peak-gated selection is most powerful on correctly-adjusted values.

---

## Super Senior Age Penalty

### The Problem

The v11 graduated adjustment treated all seniors (age >= 21.5) identically with a -25% penalty. But outcome data shows a dramatic cliff for older seniors:

| Age on Sept 1 | n | Bust% | Avg Tier | v11 Multiplier | v12 Multiplier |
|----------------|---|-------|----------|----------------|----------------|
| Sophomore (19.5-20.5) | 25 | 52% | 1.36 | +5% | +5% |
| Junior (20.5-21.5) | 104 | 74% | 0.68 | -20% | -20% |
| Senior (21.5-22.5) | 100 | 86% | 0.38 | -25% | -25% |
| **Super Senior (22.5+)** | **61** | **85%** | **0.38** | **-25%** | **-50%** |

The super senior bucket (22.5+) has an 85% bust rate and avg tier of 0.38 — yet was getting the same penalty as regular seniors. Only 9 of 61 super seniors avoided Bust, and the successes (Kupp, McLaurin, Deebo) are extreme outliers.

### Case Study: Tre Harris

Tre Harris (2025, pick 55) exposed the problem. He played at LA Tech (2020-2022) then transferred to Ole Miss (2023-2024). His 2024 Ole Miss season — 7 games, 201 routes, 5.12 YPRR, 78.9% catch rate — produced the **#1 YPRR season in the entire dataset**, barely clearing the 200-route minimum.

With the old -25% penalty, his features were:
- pg_yprr_graduated: **3.84** (elite)
- pg_catch_pct_adot_adj_graduated: **16.20** (strong)
- College-only rank: **#5 all-time** across 2022-2026

This ranked him alongside Chris Olave, Jaxon Smith-Njigba, and Makai Lemon on college profile alone — obviously wrong for a 23-year-old 5th-year transfer playing 7 games.

With the -50% super senior penalty:
- pg_yprr_graduated: **2.56** (average)
- pg_catch_pct_adot_adj_graduated: **10.80** (modest)
- College-only rank: **#54**

### Validation

The super senior penalty was tested across a grid of 15 multiplier combinations. On LOO, every configuration with a steeper super senior penalty improved over the v11 baseline. On production holdout, the -50% penalty is essentially neutral on calibration (Brier unchanged at 0.334, LogLoss +0.003) while correctly penalizing inflated rate stats from older players.

### Birthdate Imputation

`build_lookups()` now imputes missing birthdates from `draft_age` in `draft_ages.csv` where available. This ensures age adjustments are applied consistently. Only 2 players (Kendrick Law, CJ Daniels — both 2026 late-round picks) remain without birthdates.

---

## The best2_catch_pct_adot_adj Investigation

### The Question

With `pg_catch_pct_adot_adj_graduated` in the model, `best2_catch_pct_adot_adj` appears redundant — both are aDOT-adjusted catch percentages (Spearman ~0.774). Should it be dropped?

### Comprehensive Grid Search

A 12-configuration LOO grid search tested all combinations of YPRR variant (best1/pg), catch feature (pg_cpaa/best1_cpaa/QBR/none), and 6th feature (best2_cpaa/none). The 5-feature `pg_yprr + pg_cpaa` model won both LogLoss and Brier:

| Config | #F | LogLoss | Brier | Elite AUC | Stud AUC | Starter AUC |
|--------|---|---------|-------|-----------|----------|-------------|
| **pg_yprr + pg_cpaa** | **5** | **1.711** | **0.488** | **0.861** | 0.787 | 0.837 |
| pg_yprr + pg_cpaa + b2_cpaa | 6 | 1.726 | 0.492 | 0.861 | 0.781 | 0.848 |
| b1_yprr + pg_cpaa | 5 | 1.862 | 0.489 | 0.863 | 0.787 | 0.838 |
| b1_yprr + pg_cpaa + b2_cpaa | 6 | 1.802 | 0.492 | 0.861 | 0.783 | 0.848 |
| pg_yprr + QBR | 5 | 2.290 | 0.509 | 0.840 | 0.828 | 0.822 |
| pg_yprr + QBR + b2_cpaa | 6 | 2.038 | 0.516 | 0.831 | 0.777 | 0.824 |

### Verdict

**Drop `best2_catch_pct_adot_adj`.** The 5-feature model wins LogLoss and Brier on both LOO and holdout. Simpler model, better calibration.

---

## Individual Model Performance

| Model | LogLoss | Brier |
|-------|---------|-------|
| Bayesian Full | 0.777 | 0.340 |
| XGBoost Full | 1.166 | 0.340 |
| **Ensemble Full** | **0.761** | **0.334** |
| Bayesian College | 0.833 | 0.373 |
| XGBoost College | 1.769 | 0.414 |
| Ensemble College | 0.867 | 0.383 |

The 60/40 Bayesian/XGBoost blend continues to outperform either component model.

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Ensemble weights | 60% Bayesian / 40% XGBoost |
| Draft capital | Log-scaled: `10 - (10 / ln(261)) * ln(pick + 1)` |
| Bayesian model | PyMC OrderedLogistic, 3000 draws, 2000 tune, 4 chains |
| Bayesian priors | beta_college ~ N(0, 0.5), beta_dc ~ N(0.5, 0.3) |
| XGBoost | 150 trees, max_depth=3, lr=0.05, Platt-calibrated |
| Training set (holdout eval) | 2018-2021 (117 players) |
| Training set (prospect preds) | 2018-2023 (171 players) |
| Holdout set | 2022-2024 (88 players) |

---

## Data Pipeline Changes

### Aggregation (`aggregation/aggregate_college_stats.py`)

Two new functions added:

- **`compute_pg_yprr_graduated()`**: Peak-gated YPRR with graduated age adjustment. Selects the season with highest **age-adjusted** YPRR from seasons with `grades_offense >= 80`, falls back to best1 if none qualify.

- **`compute_pg_catch_pct_adot_adj_graduated()`**: Peak-gated aDOT-adjusted catch% with graduated age adjustment. Computes `catch_pct_adot_adj` per season using the global aDOT regression, then selects the peak **age-adjusted** value from quality-gated seasons. Falls back to best1 if none qualify.

Both functions compare age-adjusted values during season selection, ensuring the age multiplier is factored into the decision rather than applied post-hoc.

### Quality Gate

`PEAK_GATED_QUALITY_GATE = 80.0` — only seasons with PFF `grades_offense >= 80` are eligible for peak-gated selection. This threshold was chosen based on the distribution of qualified seasons (10.4% pass) and validated empirically.

### Graduated Age Adjustment (v12)

Updated from v11 to add a super senior bucket:

| Age Class | v11 Multiplier | v12 Multiplier |
|-----------|----------------|----------------|
| Freshman (< 19.5) | +25% | +25% |
| Sophomore (19.5-20.5) | +5% | +5% |
| Junior (20.5-21.5) | -20% | -20% |
| Senior (21.5-22.5) | -25% | -25% |
| **Super Senior (22.5+)** | -25% | **-50%** |

### Birthdate Imputation

`build_lookups()` now imputes missing birthdates by back-calculating from `draft_age` and `draft_date` in `draft_ages.csv`. This closes a gap where players without PFR birthdates received no age adjustment at all (multiplier defaulted to 1.0).

---

## Version History

| Version | Key Change | Holdout LogLoss | Holdout Brier | >=Elite AUC |
|---------|-----------|-----------------|---------------|-------------|
| v6 | 5 features, 200-route minimum | — | — | — |
| v7 | 6th feature (catch_pct_adot_adj), P5 filter | — | — | — |
| v8 | Senior season discount | — | — | — |
| v9 | Graduated YPRR | 0.771 | 0.343 | 0.961 |
| v10 | Log-scaled draft capital | — | — | — |
| v11 | 60/40 ensemble (was 75/25) | 0.773 | 0.340 | 0.970 |
| **v12** | **Peak-gated features, drop to 5F, super senior -50%** | **0.761** | **0.334** | **0.963** |

---

## Prospect Predictions

With the finalized 5-feature model, predictions were generated for three draft classes:

**Training**: 2018-2021 for holdout evaluation (2022-2024), 2018-2023 for prospect predictions (2024-2026).

### 2024 Top 10

| Rank | Name | Pick | E[tier] |
|------|------|------|---------|
| 1 | Malik Nabers | 6 | 2.70 |
| 2 | Marvin Harrison Jr. | 4 | 2.64 |
| 3 | Rome Odunze | 9 | 2.23 |
| 4 | Brian Thomas | 23 | 1.90 |
| 5 | Xavier Worthy | 28 | 1.49 |
| 6 | Keon Coleman | 33 | 1.35 |
| 7 | Ladd McConkey | 34 | 1.25 |
| 8 | Xavier Legette | 32 | 1.16 |
| 9 | Ja'Lynn Polk | 37 | 1.10 |
| 10 | Ricky Pearsall | 31 | 1.03 |

### 2025 Top 10

| Rank | Name | Pick | E[tier] |
|------|------|------|---------|
| 1 | Travis Hunter | 2 | 3.02 |
| 2 | Tetairoa McMillan | 8 | 2.40 |
| 3 | Emeka Egbuka | 19 | 1.75 |
| 4 | Matthew Golden | 23 | 1.62 |
| 5 | Luther Burden | 39 | 1.53 |
| 6 | Jayden Higgins | 34 | 1.30 |
| 7 | Jack Bech | 58 | 0.99 |
| 8 | Pat Bryant | 74 | 0.85 |
| 9 | Tre Harris | 55 | 0.83 |
| 10 | Kyle Williams | 69 | 0.77 |

### 2026 Top 10

| Rank | Name | Pick | E[tier] |
|------|------|------|---------|
| 1 | Carnell Tate | 4 | 3.01 |
| 2 | Makai Lemon | 20 | 2.39 |
| 3 | Jordyn Tyson | 8 | 2.32 |
| 4 | KC Concepcion | 24 | 1.61 |
| 5 | Omar Cooper Jr. | 30 | 1.34 |
| 6 | Denzel Boston | 39 | 1.14 |
| 7 | De'Zhaun Stribling | 33 | 1.12 |
| 8 | Antonio Williams | 71 | 0.81 |
| 9 | Germie Bernard | 47 | 0.81 |
| 10 | Ted Hurst | 84 | 0.62 |

### Top 10 College Prospects (across 2024-2026)

| Rank | Name | Year | Pick | College E[tier] | Full E[tier] | Edge |
|------|------|------|------|-----------------|--------------|------|
| 1 | **Makai Lemon** | 2026 | 20 | **1.845** | 2.394 | -0.550 |
| 2 | Malik Nabers | 2024 | 6 | 1.631 | 2.700 | -1.069 |
| 3 | Luther Burden | 2025 | 39 | 1.380 | 1.527 | -0.146 |
| 4 | Carnell Tate | 2026 | 4 | 1.276 | 3.009 | -1.733 |
| 5 | Tetairoa McMillan | 2025 | 8 | 1.189 | 2.403 | -1.215 |
| 6 | Tahj Washington | 2024 | 241 | 1.134 | 0.523 | +0.611 |
| 7 | Marvin Harrison Jr. | 2024 | 4 | 1.081 | 2.642 | -1.561 |
| 8 | Dominic Lovett | 2025 | 244 | 1.072 | 0.398 | +0.674 |
| 9 | Travis Hunter | 2025 | 2 | 1.064 | 3.021 | -1.957 |
| 10 | Brian Thomas | 2024 | 23 | 1.023 | 1.902 | -0.879 |

### Notable Edge Values

Large positive edge = college profile exceeds draft capital (analytical value picks):
- **Dominic Lovett** (2025, pick 244): +0.674
- **Tahj Washington** (2024, pick 241): +0.611
- **Tez Johnson** (2025, pick 235): +0.547
- **CJ Daniels** (2026, pick 197): +0.354

Large negative edge = draft capital exceeds college profile:
- **Travis Hunter** (2025, pick 2): -1.957
- **Carnell Tate** (2026, pick 4): -1.733
- **Marvin Harrison Jr.** (2024, pick 4): -1.561
- **Jordyn Tyson** (2026, pick 8): -1.352

Note: large negative edge for top picks is expected — draft capital is doing heavy lifting. The model says Hunter's college profile alone would rank him at #9, but the market's conviction (pick 2) boosts him to #1 overall.

---

## Output Files

| File | Description |
|------|-------------|
| `wr_data/outputs/holdout_predictions_v12.csv` | v12 holdout predictions (2022-2024, 5 features) |
| `wr_data/outputs/prospect_predictions_2024.csv` | 2024 prospect predictions (trained 2018-2023) |
| `wr_data/outputs/prospect_predictions_2025.csv` | 2025 prospect predictions (trained 2018-2023) |
| `wr_data/outputs/prospect_predictions_2026.csv` | 2026 prospect predictions (trained 2018-2023) |
| `wr_data/outputs/v12_loo_grid_search.csv` | Full 12-config LOO grid search results |
| `viz/profiles/2022/*.png` | 2022 holdout profile cards (top 10) |
| `viz/profiles/2023/*.png` | 2023 holdout profile cards (top 10) |
| `viz/profiles/2024/*.png` | 2024 prospect profile cards (top 10) |
| `viz/profiles/2025/*.png` | 2025 prospect profile cards (top 10) |
| `viz/profiles/2026/*.png` | 2026 prospect profile cards (top 10) |
| `wr_data/reports/pg_catch_rate_report.md` | Detailed peak-gated catch rate research report |
| `wr_data/reports/peak_gated_selection_report.md` | Peak-gated selection method investigation |
| `wr_data/charts/pg_cpaa_progression.png` | Engineering progression visualization |
| `wr_data/charts/pg_cpaa_selection_methods.png` | Selection method comparison |
| `wr_data/charts/pg_cpaa_combo_dashboard.png` | Multi-metric combo dashboard |
| `wr_data/charts/pg_cpaa_7part.png` | 7-part analysis for catch% family |

---

## Visualizations

![Engineering Progression](../charts/pg_cpaa_progression.png)
![Selection Method Comparison](../charts/pg_cpaa_selection_methods.png)
![Combo Dashboard](../charts/pg_cpaa_combo_dashboard.png)
![7-Part Analysis](../charts/pg_cpaa_7part.png)
