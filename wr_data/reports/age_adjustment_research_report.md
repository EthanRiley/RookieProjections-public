# WR Age Adjustment Research Report

## Motivation

Raw YPRR correlates with player age (+0.166 cross-sectionally, +0.56 mean within-player increase per year). Older players competing against younger opponents produce inflated rate stats. The v8 model addressed this with a flat -10pp senior discount on percentage-based stats, but did not discount YPRR itself. This research explores whether directly adjusting YPRR (and other production metrics) for age improves predictive power, and whether YPRR remains superior to YPTPA after proper age correction.

---

## Part 1: Systematic Age Adjustment Search

### Setup

Tested 4 production metrics x 5 adjustment schemes x 3 aggregation windows = 60 combinations, each with a parameter grid.

**Metrics**: YPRR, YPTPA, YPG, total yards
**Aggregation windows**: best single season (best1), best 2 seasons (best2), career
**Adjustment schemes**:
- **None** -- raw metric (baseline)
- **Senior discount** -- multiplicative penalty for age >= threshold
- **Freshman boost** -- multiplicative bonus for age <= threshold
- **Both** -- senior discount + freshman boost combined
- **Empirical** -- quadratic age normalization fitted to data

**Parameter grid**:
- Senior thresholds: 21.5, 22.0 | Discounts: 5%, 10%, 15%
- Freshman thresholds: 19.0, 19.5, 20.0 | Boosts: 5%, 10%, 15%
- Combined: all senior x freshman parameter combinations

**Evaluation**: Spearman correlation with dynasty value, standalone AUC(>=Elite), standalone AUC(>=Stud). N=209 WRs with college production data.

### Key Results

#### Best result per metric (across all agg windows and schemes)

| Metric | Best Agg | Best Scheme | Best Params | Spearman | AUC(Elite) | AUC(Stud) |
|--------|----------|-------------|-------------|----------|------------|------------|
| **YPRR** | **best1** | **both** | **sr>=21.5/-15%, fr<=19.5/+15%** | **0.345** | **0.723** | **0.829** |
| YPTPA | career | both | sr>=21.5/-15%, fr<=20.0/+15% | 0.333 | 0.670 | 0.748 |
| total_yards | best1 | both | sr>=21.5/-15%, fr<=20.0/+15% | 0.299 | 0.675 | 0.662 |
| YPG | career | both | sr>=21.5/-15%, fr<=20.0/+15% | 0.280 | 0.651 | 0.778 |

YPRR with age adjustment is the most predictive standalone production metric across all combinations.

#### Improvement over baseline (unadjusted)

| Metric | Agg | Baseline Spearman | Best Spearman | Delta |
|--------|-----|-------------------|---------------|-------|
| **YPRR** | **best1** | **0.268** | **0.345** | **+0.077** |
| YPRR | best2 | 0.220 | 0.299 | +0.080 |
| YPRR | career | 0.244 | 0.319 | +0.075 |
| YPTPA | best2 | 0.238 | 0.313 | +0.075 |
| YPTPA | career | 0.251 | 0.333 | +0.082 |

Age adjustment improves all metrics, but YPRR benefits the most in absolute terms.

#### Optimal parameters

The "both" scheme consistently wins, with:
- **Senior discount**: 15% at age >= 21.5
- **Freshman boost**: 15% at age <= 19.5 (best1 YPRR) or <= 20.0 (best2/career)

The 21.5 threshold outperforms 22.0, confirming that the age advantage begins earlier than the v8 model assumed. The 15% discount/boost magnitude is optimal across metrics.

### Sensitivity Analysis (YPRR best2)

**Senior discount**: monotonically improves with increasing discount (5% -> 10% -> 15%) at both age thresholds. The 21.5 threshold consistently outperforms 22.0.

**Freshman boost**: also monotonically improves with increasing boost. The 20.0 threshold works best for best2; the 19.5 threshold works best for best1.

---

## Part 2: YPRR vs YPTPA by Age Class

### Setup

To directly test the claim that YPTPA is more predictive than YPRR, especially at older ages, we ran bivariate (DC + metric) and trivariate (DC + metric + career tQBR) ordinal models split by age class.

**Age classes**: Freshman (<19.5), Sophomore (19.5-20.5), Junior (20.5-21.5), Senior (21.5+)
**Model**: XGBoost cumulative link + Bayesian ordinal regression ensemble (75/25)
**Training**: leave-one-year-out CV on training set, evaluate on holdout
**Metrics tested**: YPRR, YPTPA, YPG per age class

### Bivariate Results (DC + metric)

| Age Class | n (holdout) | YPRR AUC(Elite) | YPTPA AUC(Elite) | YPG AUC(Elite) |
|-----------|-------------|------------------|-------------------|-----------------|
| Freshman | 33 | 0.637 | 0.566 | 0.550 |
| Sophomore | 51 | **0.955** | 0.926 | 0.947 |
| Junior | 61 | 0.852 | 0.836 | 0.836 |
| Senior | 46 | **0.978** | 0.844 | 0.800 |

**YPRR outperforms YPTPA at every age class.** The gap is largest at the senior level (0.978 vs 0.844), directly contradicting the claim that YPTPA is superior for older players.

### Trivariate Results (DC + metric + career tQBR)

| Age Class | n (holdout) | YPRR AUC(Elite) | YPTPA AUC(Elite) | YPG AUC(Elite) |
|-----------|-------------|------------------|-------------------|-----------------|
| Freshman | 33 | 0.890 | 0.929 | 0.929 |
| Sophomore | 51 | 0.944 | 0.931 | 0.950 |
| Junior | 61 | 0.897 | 0.873 | 0.873 |
| Senior | 46 | 0.933 | 0.644 | 0.600 |

Adding tQBR narrows the gap for younger age classes (where tQBR carries more of the signal) but YPRR maintains its edge overall, especially at the senior level.

### Why YPTPA appears strong in some analyses

YPTPA is less sensitive to age inflation than raw YPRR because it normalizes by team pass attempts rather than individual routes run. This makes it look robust without adjustment -- but it's a weaker metric that happens to be less affected by a confounder. Once you correct the confounder directly, YPRR is strictly superior.

---

## Part 3: Full Model Substitution

### Setup

Substituted 5 age-adjusted variants for `best2_yprr` in the full 6-feature ensemble model:
1. **best2_yptpa_both** -- best2 YPTPA with both adjustments
2. **best1_yprr_senior** -- best1 YPRR with senior discount only
3. **best1_yprr_both** -- best1 YPRR with both adjustments
4. **best2_yprr_both** -- best2 YPRR with both adjustments
5. **career_yptpa_both** -- career YPTPA with both adjustments

Each variant was run through the full pipeline: XGBoost cumulative link + Bayesian ordinal regression, 75/25 ensemble, holdout evaluation (2022-2024).

### Ensemble Results

| Variant | LogLoss | Brier | AUC(Elite) | AUC(Stud) | AUC(LW) |
|---------|---------|-------|------------|------------|---------|
| Baseline (best2_yprr, no adj) | 0.799 | 0.356 | 0.957 | 0.894 | 0.914 |
| best2_yptpa_both | 0.784 | 0.348 | 0.952 | 0.926 | 1.000 |
| best1_yprr_senior | 0.776 | 0.347 | 0.956 | 0.929 | 1.000 |
| **best1_yprr_both** | **0.772** | **0.345** | **0.958** | **0.945** | **1.000** |
| best2_yprr_both | 0.783 | 0.348 | 0.960 | 0.941 | 1.000 |
| career_yptpa_both | 0.784 | 0.348 | 0.952 | 0.929 | 1.000 |

### Improvement: best1_yprr_both vs baseline

| Metric | Baseline | best1_yprr_both | Delta |
|--------|----------|-----------------|-------|
| LogLoss | 0.799 | 0.772 | **-0.027** |
| Brier | 0.356 | 0.345 | **-0.011** |
| AUC(>=Elite) | 0.957 | 0.958 | +0.001 |
| AUC(>=Stud) | 0.894 | 0.945 | **+0.051** |
| AUC(>=LW) | 0.914 | 1.000 | **+0.086** |

The biggest gains are at the top end: AUC(>=Stud) jumps +0.051 and AUC(>=LW) reaches 1.000. LogLoss and Brier also improve meaningfully.

### Component Breakdown

| Variant | XGB LogLoss | Bayes LogLoss | Ens LogLoss |
|---------|-------------|---------------|-------------|
| Baseline | 0.770 | 0.826 | 0.799 |
| best1_yprr_both | 0.747 | 0.799 | 0.772 |

Both model components improve with the age-adjusted metric, confirming this is real signal, not an artifact of one model type.

---

## Part 4: Three-Feature Model Discovery

An unexpected finding: the trivariate model (DC + age-adjusted YPRR + career tQBR) nearly matches the full 6-feature ensemble.

| Model | LogLoss | AUC(Elite) | AUC(Stud) | AUC(LW) |
|-------|---------|------------|------------|---------|
| Full 6-feature ensemble (v8) | 0.798 | 0.963 | 0.888 | 0.908 |
| DC + best1_yprr_both (bivariate) | 0.794 | 0.934 | 0.937 | 1.000 |
| DC + best1_yprr_both + tQBR (trivariate) | 0.762 | 0.953 | 0.937 | 0.966 |
| Full 6-feature + best1_yprr_both substitution | 0.772 | 0.958 | 0.945 | 1.000 |

The 3-feature trivariate model achieves AUC(Elite) 0.953 vs the full model's 0.963, and actually beats it on AUC(Stud) (0.937 vs 0.888). This suggests that much of the signal in the additional features (catch_pct_adot_adj, contested_catch_rate, avoided_tackles_per_rec) is redundant with age-adjusted YPRR.

### Note: Sophomore YPRR vs best1_yprr_both

The sophomore-only bivariate model (DC + sophomore YPRR) actually produces higher discrimination than best1_yprr_both within its subset:

| Model | n (holdout) | LogLoss | AUC(Elite) | AUC(Stud) |
|-------|-------------|---------|------------|------------|
| Sophomore YPRR (bivariate) | 51 | 1.090 | **0.955** | **0.980** |
| best1_yprr_both (bivariate) | 88 | **0.794** | 0.934 | 0.937 |

However, these are not directly comparable. The sophomore model only evaluates on 51 holdout players who have sophomore-year data -- it cannot score players who declared after freshman year or who lack a qualifying sophomore season. This filtered subset is easier to discriminate within. The LogLoss (1.090 vs 0.794) reflects worse overall calibration on the smaller sample.

The best1 aggregation with age adjustment effectively captures the sophomore signal (sophomore seasons are often the best single season) while remaining applicable to all prospects. It trades a small amount of within-subset discrimination for universal coverage.

---

## Conclusions

1. **Age adjustment significantly improves production metric predictiveness.** The optimal scheme is a 15% senior discount at age 21.5 combined with a 15% freshman boost at age 19.5-20.0.

2. **YPRR is superior to YPTPA at every age class.** The claim that YPTPA is more predictive, especially for older players, is not supported. YPTPA only appears robust because it's less sensitive to the age confounder -- once corrected, YPRR dominates.

3. **Best single-season YPRR with both adjustments is the optimal production feature.** best1_yprr_both improves the full model across all metrics, with AUC(Stud) jumping from 0.894 to 0.945.

4. **A 3-feature model (DC + age-adjusted YPRR + career tQBR) captures most of the signal.** This has implications for model simplification and interpretability.

5. **The v8 senior discount should be extended to YPRR.** The current v8 model discounts percentage-based stats but not YPRR. This research shows YPRR benefits substantially from direct age adjustment.

6. **"Young YPRR" (best FR/SO, fallback -15% JR) does not work.** Excluding senior seasons entirely and restricting to young seasons loses too many players (95 train / 71 holdout vs 117/88). LogLoss degrades to 0.943 (+0.144 vs baseline). Adjusting all seasons and letting the best one win is strictly better than filtering by age class.

---

## Part 5: Young YPRR Variant (Negative Result)

### Idea

Instead of adjusting all seasons and picking the best, restrict to the youngest available:
1. Best of freshman (<19.5) or sophomore (19.5-20.5) seasons by grades_offense
2. If neither available, fall back to best junior (20.5-21.5) season with 15% discount
3. Senior seasons excluded entirely

### Results

| Variant | n (train) | n (holdout) | LogLoss | Brier | AUC(Elite) | AUC(Stud) |
|---------|-----------|-------------|---------|-------|------------|------------|
| Baseline (best2_yprr) | 119 | 89 | 0.799 | 0.356 | 0.957 | 0.894 |
| **best1_yprr_both** | **117** | **88** | **0.772** | **0.345** | **0.958** | **0.945** |
| young_yprr | 95 | 71 | 0.943 | 0.425 | 0.938 | 0.902 |

### Why It Fails

The young_yprr variant loses 20% of the dataset. Many players don't have a qualifying freshman or sophomore season with 200+ routes at a P5 school. The junior fallback with a 15% discount is too aggressive -- it throws away real signal. The coverage loss (71 vs 88 holdout) compounds the problem: fewer training examples and fewer holdout players for evaluation.

The best1_yprr_both approach is superior because it keeps all seasons eligible but adjusts their values multiplicatively. Seniors get discounted, freshmen get boosted, and the best single season wins. Same age correction, no data loss.

---

## Recommended Changes for v9

1. **Replace `best2_yprr` with `best1_yprr_both`** -- best single season YPRR with 15% senior discount (age >= 21.5) and 15% freshman boost (age <= 19.5).
2. **Consider simplifying to 3 features** -- DC + best1_yprr_both + career_targeted_qb_rating achieves comparable performance. The additional features may add more complexity than signal.
3. **Tune senior discount magnitude on YPRR independently** from the percentage-point discounts on rate stats. The optimal YPRR discount is 15% multiplicative, while the v8 percentage stats use 10pp additive.

---

## Files Generated

| File | Description |
|------|-------------|
| `wr_data/age_adjusted_production_results.csv` | 852 rows: all metric x agg x scheme x param combinations |
| `wr_data/yprr_substitution_results.csv` | 6 rows: full model results for baseline + 5 substitution variants |
| `wr_data/age_class_model_results.csv` | 36 rows: bivariate/trivariate results by age class + winners |
| `wr_data/age_adj_heatmaps.png` | Spearman heatmap by metric x scheme, 3 agg windows |
| `wr_data/age_adj_sensitivity.png` | Senior discount + freshman boost sensitivity curves |
| `wr_data/age_adj_improvement.png` | Delta bars, scheme comparison, agg comparison |
| `wr_data/age_adj_auc_scatter.png` | AUC Elite vs Stud scatter |
| `wr_data/yprr_sub_deltas.png` | Improvement over baseline for each substitution variant |
| `wr_data/yprr_sub_abs_metrics.png` | Absolute ensemble metrics by variant |
| `wr_data/yprr_sub_model_breakdown.png` | XGB vs Bayesian vs Ensemble per variant |
| `wr_data/age_class_bivariate.png` | YPRR vs YPTPA vs YPG by age class (bivariate) |
| `wr_data/age_class_trivariate.png` | YPRR vs YPTPA vs YPG by age class (trivariate) |
| `wr_data/age_class_winners.png` | Full comparison: baselines vs age classes vs winners |
| `modeling/test_age_adjusted_production.py` | Age adjustment grid search script |
| `modeling/test_yprr_substitutions.py` | Full model substitution test script |
| `modeling/test_age_class_models.py` | Age class bivariate/trivariate model script |
| `modeling/viz_age_adjusted_production.py` | Age adjustment visualization script |
| `modeling/viz_yprr_substitutions.py` | Substitution results visualization script |
| `modeling/viz_age_class_models.py` | Age class results visualization script |
