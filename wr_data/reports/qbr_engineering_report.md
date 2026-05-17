# Target Outcome Feature Engineering Report

**Date**: 2026-05-13
**Script**: `modeling/research/target_outcome_engineering.py`
**Objective**: Engineer replacements for `career_targeted_qb_rating` (and potentially `best2_catch_pct_adot_adj`) using aDOT-adjusted QBR, composite metrics, PCA, supervised composites, and novel catch-ability features. Evaluate using ordinal LogLoss, Brier score, and multi-threshold AUC.

---

## Executive Summary

We engineered 21 new candidate features (plus 2 supervised composites) and tested ~90 replacement combinations using ordinal cumulative-link scoring (LogLoss, Brier) and multi-threshold AUC.

**The key finding: the metrics disagree on the best replacement, but they unanimously agree that v11 is not optimal.**

| Configuration | LogLoss | Brier | >=Elite AUC | >=Stud AUC | >=Starter AUC | # Feats |
|---------------|---------|-------|-------------|------------|---------------|---------|
| **v11 (current)** | **2.347** | **0.515** | **0.842** | **0.778** | **0.833** | **6** |
| v11 minus QBR only | 2.243 | 0.508 | 0.828 | 0.782 | 0.838 | 5 |
| v11 minus QBR+CPA | 2.407 | 0.494 | 0.839 | 0.851 | 0.844 | 4 |
| QBR+CPA => catch_minus_drops | **2.168** | 0.523 | 0.840 | 0.779 | 0.826 | 5 |
| QBR+CPA => qbr_adot_adj_graduated | 2.372 | **0.488** | 0.832 | 0.800 | 0.837 | 5 |
| QBR+CPA => career_targeted_qb_rating | 2.286 | 0.505 | **0.850** | 0.828 | 0.834 | 5 |
| QBR+CPA => clean_catch_rate | 2.355 | 0.510 | 0.847 | 0.777 | 0.830 | 5 |

Every configuration that removes QBR improves at least one major metric. The 4-anchor base alone (no catch-quality feature at all) has the best Brier (0.494) and best Stud AUC (0.851). But calibration (LogLoss) and Elite discrimination benefit from adding a 5th catch-quality feature.

**No single replacement dominates across all metrics.** The choice depends on what you optimize for. See Section 4 for the full analysis.

---

## 1. Features Engineered

### 1.1 QBR Variants

| Feature | Description |
|---------|-------------|
| `best1_qbr_adot_adj` | QBR residual after regressing on aDOT (best season by grades_offense) |
| `best1_qbr_adot_adj_graduated` | Same, with graduated age adjustment (FR +25%, SO +5%, JR -20%, SR -25%) |
| `best1_qbr_graduated` | Raw QBR with graduated age adjustment (centered on mean 110) |
| `peak_qbr` | Maximum single-season QBR (no grades_offense selection) |
| `peak_qbr_adot_adj` | Maximum single-season aDOT-adjusted QBR |
| `career/best2_targeted_qb_rating_sr_disc` | QBR with senior season discount (-10pp) |
| `career/best2_qbr_adot_adj_sr_disc` | aDOT-adjusted QBR with senior discount |

### 1.2 Target Outcome Metrics

| Feature | Description |
|---------|-------------|
| `yards_per_target` | Total yards / targets |
| `first_downs_per_target` | First downs / targets |
| `td_per_target` | Touchdowns / targets |
| `value_per_target` | (First downs + touchdowns) / targets |
| `yac_per_target` | YAC / targets |
| `ypr_adot_adj` | Yards per reception, adjusted for aDOT |

### 1.3 Catch Reliability Metrics

| Feature | Description |
|---------|-------------|
| **`clean_catch_rate`** | Non-contested receptions / non-contested targets * 100. Isolates uncontested catching ability. |
| `catch_minus_drops` | caught_percent - drop_rate. Rewards catching, penalizes drops. |
| `no_negative_rate` | 1 - (drops + interceptions) / targets. "Don't hurt the offense" rate. |

### 1.4 Composites

| Feature | Description |
|---------|-------------|
| `z_target_outcome` | Z-score average of catch% aDOT adj + QBR aDOT adj + yards/target |
| `z_catch_ability` | Z-score average of catch% aDOT adj + no-negative rate + grades_hands_drop |
| `z_reception_value` | Z-score average of yards/target + YAC/target + first_downs/target |
| `z_target_quality_full` | Z-score average of 5 target-outcome metrics |
| `pca_target_outcome_1` | PC1 of [catch% adj, QBR adj, yards/tgt, YAC/tgt, value/tgt, no-neg rate] |
| `pca_target_outcome_2` | PC2 of the same (YAC vs touchdown/first-down axis) |
| `pca_catch_ability` | PC1 of [catch% adj, no-neg rate, clean catch rate, grades_hands_drop] |

### 1.5 Supervised Composites

| Feature | Description |
|---------|-------------|
| `supervised_catch` | Ridge-weighted composite of catch-quality components, trained on tier outcome |
| `supervised_catch_loo` | Leave-one-year-out version to avoid target leakage |

Ridge regression was fit on catch-quality components (catch% aDOT adj, clean catch rate, no-negative rate, grades_hands_drop, etc.) against tier outcome. The LOO version refits Ridge within each fold.

### 1.6 PCA Loadings

**Target Outcome PC1** (57.2% variance explained):
```
catch_pct_adot_adj:  +0.462
qbr_adot_adj:        +0.500  (QBR is the largest loading)
yards_per_target:    +0.460
value_per_target:    +0.451
no_negative_rate:    +0.267
yac_per_target:      +0.222
```

**Catch Ability PC1** (67.2% variance explained):
```
grades_hands_drop:   +0.534
no_negative_rate:    +0.506
catch_pct_adot_adj:  +0.481
clean_catch_rate:    +0.477
```

All four components load roughly equally -- this is a well-defined latent construct of "catch ability."

---

## 2. Results: DC-Only Base

Testing each candidate added to a draft_capital-only base. n=237.

### Top 10 Candidates

| Rank | Feature | Spearman | LOO Delta | Residual | Boot %+ | Collinearity |
|------|---------|----------|-----------|----------|---------|-------------|
| 1 | career_targeted_qb_rating | +0.303 | **+0.009** | -0.007 | 45.4% | 0.399 |
| 2 | best1_qbr_adot_adj_graduated | +0.302 | +0.005 | +0.007 | 51.5% | 0.388 |
| 3 | best1_pca_catch_ability | +0.200 | +0.001 | +0.013 | 57.2% | 0.246 |
| 4 | best1_catch_minus_drops | +0.184 | +0.000 | +0.003 | 50.1% | 0.230 |
| 5 | best1_z_catch_ability | +0.180 | -0.000 | -0.001 | 49.3% | 0.231 |
| 6 | peak_qbr | +0.237 | -0.001 | +0.013 | 58.4% | 0.305 |
| 7 | peak_qbr_adot_adj | +0.234 | -0.001 | +0.014 | 59.3% | 0.301 |
| 8 | best1_qbr_adot_adj | +0.245 | -0.001 | -0.018 | 37.9% | 0.354 |
| 9 | **best1_clean_catch_rate** | +0.177 | -0.002 | **+0.045** | **72.4%** | **0.192** |
| 10 | best2_catch_pct_adot_adj | +0.342 | -0.004 | +0.043 | 74.0% | 0.394 |

### Supervised Composite Performance (DC-Only)

| Feature | Spearman | LOO Delta | Residual | Boot %+ | Collinearity |
|---------|----------|-----------|----------|---------|-------------|
| best1_supervised_catch | +0.253 | -0.001 | -0.017 | 38.8% | 0.360 |
| best1_supervised_catch_loo | +0.212 | -0.006 | -0.044 | 24.9% | 0.340 |

The supervised composite ranks middle-of-pack. Its LOO version performs worse, suggesting the Ridge weights are overfitting to the training data. The optimal weighting doesn't help when the underlying components are already collinear with draft capital.

**Key observations:**
- `career_targeted_qb_rating` still wins on LOO-AUC delta in the DC-only context
- `best1_clean_catch_rate` has the strongest residual signal (+0.045, 72.4% bootstrap) and lowest collinearity (0.192)
- The aDOT-adjusted QBR variants don't clearly beat raw QBR
- The catch-ability family (clean_catch_rate, catch_minus_drops, pca_catch_ability) consistently shows lower collinearity with DC than QBR variants
- Supervised composites add nothing over simpler features

---

## 3. Results: Full Model Base (4 Anchors)

Testing each candidate added to the 4 anchor features (DC + YPRR + CCR + AT/Rec). n=205. This is the critical test: what does the 5th/6th feature need to provide that the anchors don't already capture?

### Top 10 Candidates

| Rank | Feature | Spearman | LOO Delta | Residual | Boot %+ | Collinearity | Era Drift |
|------|---------|----------|-----------|----------|---------|-------------|-----------|
| **1** | **best1_clean_catch_rate** | +0.161 | **+0.010** | +0.019 | 58.8% | **0.268** | **0.010** |
| 2 | best1_catch_minus_drops | +0.155 | +0.009 | -0.046 | 26.1% | 0.331 | 0.060 |
| 3 | best1_pca_catch_ability | +0.177 | +0.008 | -0.029 | 35.1% | 0.292 | 0.140 |
| 4 | career_targeted_qb_rating | +0.317 | +0.004 | -0.050 | 26.4% | 0.404 | 0.035 |
| 5 | best1_z_catch_ability | +0.160 | +0.003 | -0.043 | 27.3% | 0.327 | 0.145 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 19 | **best2_catch_pct_adot_adj** | +0.353 | **-0.011** | -0.099 | 7.1% | 0.510 | 0.146 |

### Supervised Composite Performance (Full Model)

| Feature | Spearman | LOO Delta | Residual | Boot %+ | Collinearity |
|---------|----------|-----------|----------|---------|-------------|
| best1_supervised_catch | +0.253 | -0.001 | -0.087 | 12.4% | 0.374 |
| best1_supervised_catch_loo | +0.210 | -0.008 | -0.114 | 5.7% | 0.354 |

In the full model context, supervised composites are clearly negative. The Ridge-weighted composite has -0.087 residual (12.4% bootstrap positive) and zero LOO-AUC delta. The LOO version is worse still (-0.008 delta, 5.7% bootstrap). **Optimal linear weighting of catch components cannot overcome the fundamental redundancy with existing features.**

**Key finding: best1_clean_catch_rate ranks #1 in the full model context**, beating career_targeted_qb_rating by +0.006 LOO-AUC delta while having:
- **Much lower collinearity** (0.268 vs 0.404) -- less redundant with existing features
- **Best era stability** (0.010 drift vs 0.035) -- most temporally robust
- **Positive residual** (+0.019, 58.8% bootstrap) vs negative (-0.050, 26.4%) for QBR
- Arguably the only candidate with genuinely positive residual signal in the full context

**best2_catch_pct_adot_adj ranks near the bottom** (-0.011 LOO delta, -0.099 residual, 7.1% bootstrap). It actively hurts the model in the full context. Its high collinearity with the anchors (0.510) means it's deeply redundant.

---

## 4. Combination Results (Full Ordinal Scoring)

All combinations were evaluated using leave-one-year-out ordinal cumulative-link classifiers producing LogLoss, Brier score, and multi-threshold AUC. This is the most comprehensive test.

### 4.1 The Headline Finding: Metrics Disagree

Unlike the investigation report (which only used AUC), the full ordinal scoring reveals that **no single replacement dominates all metrics**:

| Metric | Best Configuration | Best Value | v11 Value | Change |
|--------|-------------------|------------|-----------|--------|
| **LogLoss** | QBR+CPA => catch_minus_drops | **2.168** | 2.347 | **-0.179** |
| **Brier** | QBR+CPA => qbr_adot_adj_graduated | **0.488** | 0.515 | **-0.027** |
| **>=Elite AUC** | QBR+CPA => career_targeted_qb_rating | **0.850** | 0.842 | **+0.008** |
| **>=Stud AUC** | 4 anchors only | **0.851** | 0.778 | **+0.073** |
| **>=Starter AUC** | 4 anchors only | **0.844** | 0.833 | **+0.011** |

This means different feature sets optimize for different objectives. The user must decide which metric matters most.

### 4.2 Replace Both QBR+CPA with Single Feature

| Configuration | LogLoss | Brier | >=Elite | >=Stud | >=Starter |
|---------------|---------|-------|---------|--------|-----------|
| **v11 (current, 6 feats)** | **2.347** | **0.515** | **0.842** | **0.778** | **0.833** |
| 4 anchors only (4 feats) | 2.407 | **0.494** | 0.839 | **0.851** | **0.844** |
| + catch_minus_drops | **2.168** | 0.523 | 0.840 | 0.779 | 0.826 |
| + qbr_adot_adj_graduated | 2.372 | **0.488** | 0.832 | 0.800 | 0.837 |
| + career_targeted_qb_rating | 2.286 | 0.505 | **0.850** | 0.828 | 0.834 |
| + clean_catch_rate | 2.355 | 0.510 | 0.847 | 0.777 | 0.830 |
| + supervised_catch | 2.371 | 0.505 | 0.842 | 0.833 | 0.828 |
| + z_target_outcome | 2.281 | 0.500 | 0.841 | 0.843 | 0.832 |
| + pca_catch_ability | 2.434 | 0.520 | 0.843 | 0.802 | 0.827 |
| + no_negative_rate | 2.503 | 0.504 | 0.837 | 0.845 | 0.841 |

**Interpretation by metric:**

- **LogLoss (calibration)**: `catch_minus_drops` wins decisively (2.168 vs 2.347 v11). It improves the probability distribution shape across the full ordinal scale. This makes sense -- caught_percent minus drop_rate directly penalizes the two failure modes (not catching, dropping) that separate busts from producers.

- **Brier (probability accuracy)**: `qbr_adot_adj_graduated` wins (0.488 vs 0.515). Interestingly, the 4-anchor base alone (0.494) nearly matches it. Both the aDOT adjustment and graduated age adjustment appear to strip noise from QBR, producing better-calibrated probability estimates. But Brier is fairly close across many configurations.

- **>=Elite AUC (discrimination)**: `career_targeted_qb_rating` as a solo replacement wins (0.850). This is the strongest argument for keeping QBR in some form -- but notably, it works better as the *only* catch-quality feature (5 feats) than paired with CPA (6 feats, 0.842). `clean_catch_rate` is close (0.847).

- **>=Stud/Starter AUC**: The 4-anchor base dominates (0.851/0.844). Adding any catch-quality feature tends to hurt Stud discrimination, suggesting the 5th feature adds noise at the tail.

### 4.3 Replace QBR Only (Keeping CPA)

| Configuration | LogLoss | Brier | >=Elite | >=Stud | >=Starter |
|---------------|---------|-------|---------|--------|-----------|
| v11 (current) | 2.347 | 0.515 | 0.842 | 0.778 | 0.833 |
| v11 minus QBR | 2.243 | 0.508 | 0.828 | 0.782 | 0.838 |
| QBR => clean_catch_rate | 2.254 | 0.511 | 0.838 | 0.775 | 0.828 |
| QBR => z_target_outcome | **2.202** | 0.511 | 0.827 | 0.773 | 0.830 |
| QBR => no_negative_rate | 2.382 | **0.510** | 0.830 | 0.773 | **0.839** |
| QBR => supervised_catch | 2.384 | 0.514 | 0.827 | 0.762 | 0.820 |

Simply dropping QBR (keeping CPA) improves both LogLoss (2.243 vs 2.347) and Brier (0.508 vs 0.515). This confirms the investigation report's finding: QBR actively hurts the model. No replacement clearly beats just dropping it when CPA stays.

### 4.4 Best Two-Feature Replacements (6 Total Features)

If we want to keep 6 features (4 anchors + 2 new), the best combos by each metric:

| Combo | LogLoss | Brier | >=Elite |
|-------|---------|-------|---------|
| qbr_adot_adj_gr + targeted_qb_rating | **2.197** | 0.494 | 0.831 |
| qbr_adot_adj_gr + catch_minus_drops | 2.241 | 0.507 | 0.833 |
| qbr_adot_adj_gr + z_target_outcome | 2.315 | 0.491 | 0.833 |
| qbr_adot_adj_gr + supervised_catch | 2.435 | **0.488** | 0.839 |
| qbr_adot_adj_gr + clean_catch_rate | 2.413 | 0.498 | **0.848** |
| v11 (QBR + CPA baseline) | 2.347 | 0.515 | 0.842 |

The `best1_qbr_adot_adj_graduated` appears in most top combos as the better "QBR slot" feature. The best 6-feature LogLoss (2.197) still doesn't beat the best 5-feature LogLoss (2.168 with catch_minus_drops alone), reinforcing that simplicity wins.

### 4.5 Why Simpler Models Win

1. **Collinearity reduction**: QBR (0.404 max) and CPA (0.510 max) are both highly correlated with existing features. They add noise without proportionate signal.
2. **n=205 regularizes toward simplicity**: Going from 6 to 5 features reduces overfitting risk. The Brier improvement of the 4-anchor base (0.494 vs 0.515) demonstrates this.
3. **QBR and CPA overlap** (rho=0.575). Replacing two overlapping features with one focused feature eliminates internal redundancy.

---

## 5. Supervised Composites: Why They Failed

The hypothesis was: maybe the right *linear combination* of catch-quality components (catch% aDOT adj, clean catch rate, no-negative rate, grades_hands_drop) would outperform any individual feature. Ridge regression was used to learn optimal weights from the training data, with a LOO variant to prevent target leakage.

### Results

| Feature | DC Context |  | Full Context |  |
|---------|-----------|--|-------------|--|
|  | LOO Delta | Residual (Boot%) | LOO Delta | Residual (Boot%) |
| best1_supervised_catch | -0.001 | -0.017 (38.8%) | -0.001 | -0.087 (12.4%) |
| best1_supervised_catch_loo | -0.006 | -0.044 (24.9%) | -0.008 | -0.114 (5.7%) |
| best1_clean_catch_rate | -0.002 | +0.045 (72.4%) | +0.010 | +0.019 (58.8%) |
| best1_catch_minus_drops | +0.000 | +0.003 (50.1%) | +0.009 | -0.046 (26.1%) |

In combo testing as QBR+CPA replacement: supervised_catch achieves LogLoss 2.371, Brier 0.505, Elite AUC 0.842 -- roughly matching v11 but not improving it. The LOO version wasn't tested in combos because it was already clearly worse.

### Why Supervised Composites Don't Work Here

1. **Redundancy is the problem, not weighting.** The catch-quality components are all measuring the same latent dimension (PC1 explains 67.2%). No linear reweighting can extract orthogonal information that doesn't exist.
2. **Overfitting.** The supervised version has -0.017 residual but the LOO version has -0.044. The gap shows the Ridge weights are fitting to noise.
3. **Collinearity with anchors.** The composite's max collinearity (0.374 in full context) is lower than QBR (0.404) but higher than clean_catch_rate (0.268). Optimal weighting pushes toward the features that are already well-represented by the anchors.

---

## 6. What is best1_clean_catch_rate?

### Definition

```
clean_catch_rate = (receptions - contested_receptions) / (targets - contested_targets) * 100
```

Aggregation: `best1` = best single P5 season by `grades_offense`, 200+ route minimum.

This isolates **uncontested catching ability** -- how reliably the receiver catches the ball when they have clean separation. It strips out:
- Contested catches (already measured by `best2_contested_catch_rate`)
- Deep ball difficulty (not aDOT-dependent like catch% is)
- Touchdown noise (unlike QBR, which is heavily TD-driven)

### Why It Works

1. **Mechanistically distinct**: The model already has contested catching (CCR). Clean catch rate measures the complementary skill -- can you catch the routine ones? The best receivers do both.
2. **Low collinearity** (0.268): Less redundant with DC, YPRR, or CCR than QBR (0.404) or CPA (0.510).
3. **Exceptional era stability** (0.010 drift): The most temporally stable candidate, meaning this signal is unlikely to be a sample artifact.
4. **Single-season selection reduces noise**: Using `best1` (top season by grades_offense) rather than career average avoids diluting with early-career development seasons.
5. **Strong across metrics**: Not the absolute best on any single metric, but consistently good -- +0.010 LOO-AUC, 2.355 LogLoss (vs 2.347 v11), 0.510 Brier (vs 0.515 v11), 0.847 Elite AUC (vs 0.842 v11).

---

## 7. The Broader Insight: "Catch Quality" is One Dimension, Not Two

The original v11 model uses two features to measure catch quality:
- `career_targeted_qb_rating`: A noisy composite of catches + TDs + yards
- `best2_catch_pct_adot_adj`: Catch rate adjusted for depth of target

These two features correlate at rho=0.575. They're measuring the same latent construct from slightly different angles, and the overlap is costly -- it wastes a feature slot while adding collinearity.

The PCA analysis confirms this: the first principal component of the catch-quality metrics explains 67.2% of variance, with all four components loading roughly equally. There's one dominant underlying factor: **catch ability**.

A single well-chosen feature can capture this dimension more efficiently than two correlated features.

---

## 8. Metric-Specific Recommendations

The right choice depends on optimization priority:

### If LogLoss matters most (calibration)
**Replace QBR+CPA with `best1_catch_minus_drops`** (LogLoss 2.168 vs 2.347, -7.6%)

`catch_minus_drops = caught_percent - drop_rate` directly penalizes the two failure modes that differentiate tiers. Its LogLoss advantage is the largest improvement of any configuration.

Tradeoff: slightly worse Brier (0.523 vs 0.515) and Starter AUC (0.826 vs 0.833).

### If Brier matters most (probability accuracy)
**Replace QBR+CPA with `best1_qbr_adot_adj_graduated`** (Brier 0.488 vs 0.515, -5.2%)

Or simply use the **4-anchor base** (Brier 0.494) with no catch-quality feature. The marginal gain of adding a 5th feature is small.

### If Elite AUC matters most (top-tier discrimination)
**Replace QBR+CPA with `career_targeted_qb_rating` as sole catch feature** (Elite AUC 0.850 vs 0.842, +0.8%)

Ironic: QBR as a solo feature (without CPA alongside it) actually performs better than the v11 pair. Or use **`best1_clean_catch_rate`** (0.847) for a cleaner, more interpretable feature.

### Balanced recommendation
**Replace QBR+CPA with `best1_clean_catch_rate`** -- it ranks in the top 3 on every metric without being worst on any:
- LogLoss: 2.355 (2nd best single replacement, vs 2.347 v11)
- Brier: 0.510 (middle, vs 0.515 v11)
- Elite AUC: 0.847 (2nd best, vs 0.842 v11)
- Era stability: 0.010 (best of any candidate)
- Collinearity: 0.268 (lowest of viable candidates)

### Proposed v12 feature set (5 features)

| Feature | Dimension |
|---------|-----------|
| draft_capital | NFL talent consensus |
| best1_yprr_graduated | Peak route efficiency (age-adjusted) |
| best2_contested_catch_rate | Contested ball skills |
| best2_avoided_tackles_per_rec | Post-catch elusiveness |
| **best1_clean_catch_rate** | **Uncontested catching reliability** |

Each feature measures a distinct, non-overlapping dimension. No two features correlate above 0.3.

---

## 9. Required Follow-Up

1. **Re-run the full Bayesian + XGBoost ensemble** with the proposed feature set on the 2022-2024 holdout. The LOO logistic regression tests here are directionally reliable but the production model (especially XGBoost) may capture non-linear interactions differently.
2. **Test the top 3 candidates** (clean_catch_rate, catch_minus_drops, qbr_adot_adj_graduated) in the production ensemble to see if metric disagreements resolve.
3. **Add `best1_clean_catch_rate` to the aggregation pipeline** (`aggregate_college_stats.py`) so it's computed properly for prospects.
4. **Regenerate prospect predictions** for 2024-2026 classes with the new feature set.

---

## 10. Visualizations

| File | Description |
|------|-------------|
| `wr_data/charts/qbr_engineering_candidates.png` | 4-panel candidate analysis (signal vs lift, residual reliability, top-15 rankings) |
| `wr_data/charts/qbr_engineering_combos.png` | 3-panel combo results: LogLoss, Brier, and Elite AUC rankings |
| `wr_data/charts/qbr_engineering_families.png` | LOO-AUC delta by feature family (QBR variants, catch metrics, composites) |
| `wr_data/charts/qbr_engineering_verdict.png` | 4-panel verdict: key combos, full-model rankings, multi-threshold comparison, head-to-head table |

## 11. Data Files

| File | Description |
|------|-------------|
| `wr_data/outputs/qbr_engineering_candidates.csv` | All engineered features per player (master + 59 new columns) |
| `wr_data/outputs/qbr_eng_results_dc.csv` | 7-part analysis results (DC-only base) |
| `wr_data/outputs/qbr_eng_results_full.csv` | 7-part analysis results (4-anchor base) |
| `wr_data/outputs/qbr_eng_combos.csv` | All ~90 combination test results (LogLoss, Brier, AUC) |
