# Breakout Age Engineering Report

## Overview

We evaluated 17 breakout age variants (9 binary threshold, 4 YPTPA quality-adjusted, 4 YPRR quality-adjusted) plus draft age as a baseline, then further explored 16 YPRR breakout configurations via grid search and a double-residual production metric. The goal: find the best single feature to capture "how early did this player prove he was good in college?"

The dataset covers 291 WRs drafted 2016-2024 with PFF college grades. Evaluation uses Layer 1 univariate screens (Spearman correlation with tier outcome, AUC for Elite+ classification, era stability) plus residual analysis after controlling for all other model features (draft_capital, career_targeted_qb_rating, career_yprr, career_catch_pct_adot_adj, best2_contested_catch_rate, career_avoided_tackles_pg).

---

## Variants Tested

### Binary Threshold Variants

These answer "at what age did the player first cross threshold X?"

| Variant | Definition | Coverage |
|---------|-----------|----------|
| ba_650yards | First season with 650+ total receiving yards | 71.5% |
| ba_45ypg | First season with 45+ ypg and 8+ games | 76.6% |
| ba_yptpa | First season with game-normalized YPTPA >= 1.4 and 8+ games | 77.3% |
| ba_45ypg_yprr | First season with 45+ ypg, 8+ games, AND 2.0+ YPRR | 68.0% |
| ba_yprr_routes | First season with 2.0+ YPRR and 150+ routes | 71.1% |
| ba_dominator | First season with 25%+ team receiving yard share and 8+ games | 56.4% |
| ba_yptpa_yprr | First season with YPTPA >= 1.4 AND YPRR >= 2.0, 8+ games | 68.7% |
| ba_composite | First season with (YPTPA >= 1.4 OR YPRR >= 2.2) and 8+ games | 79.0% |

### Quality-Adjusted Variants

These build on the YPTPA breakout (variant 3) but incorporate *how dominant* the breakout was, not just whether it happened. All have the same 77.3% coverage as ba_yptpa since they require a YPTPA breakout to exist.

| Variant | Definition |
|---------|-----------|
| qa_ratio_scaled | `age * (1.4 / actual_yptpa)` — compresses age proportionally to dominance |
| qa_zscore_adj | `age - z * 0.5` where z = (breakout_yptpa - pop_mean) / pop_std — discounts age by statistical exceptionalism |
| qa_log_magnitude | `age / log2(actual_yptpa / 1.4 + 1)` — diminishing returns on excess YPTPA |
| qa_magnitude | Just the YPTPA value at breakout (no age component) |

The z-score is computed against the full population of all college WR seasons (mean YPTPA = 0.529, std = 0.638, n = 21,669).

### YPRR Quality-Adjusted Variants

Same approach as the YPTPA quality-adjusted variants, but using the YPRR breakout (2.0+ YPRR, 150+ routes) as the trigger and the YPRR value as the quality metric. Coverage is 71.1% (matching ba_yprr_routes).

| Variant | Definition |
|---------|-----------|
| qy_ratio_scaled | `age * (2.0 / actual_yprr)` — compresses age proportionally to YPRR dominance |
| qy_zscore_adj | `age - z * 0.5` where z = (breakout_yprr - pop_mean) / pop_std |
| qy_log_magnitude | `age / log2(actual_yprr / 2.0 + 1)` — diminishing returns on excess YPRR |
| qy_magnitude | Just the YPRR value at breakout (no age component) |

The YPRR z-score is computed against the full population (mean YPRR = 1.441, std = 2.307, n = 21,573). Note the much higher variance compared to YPTPA — this matters for how discriminating the z-scores are.

---

## Results

### Raw Signal (No Imputation)

Evaluating only players who have a breakout age (dropping NaN):

| Variant | Spearman | AUC | N | Drift |
|---------|----------|-----|---|-------|
| **qa_zscore_adj** | **-0.385** | **0.721** | 225 | 0.099 |
| ba_yptpa | -0.315 | 0.684 | 225 | 0.136 |
| ba_yptpa_yprr | -0.315 | 0.698 | 200 | 0.222 |
| ba_log_magnitude | -0.290 | 0.669 | 225 | 0.009 |
| ba_650yards | -0.289 | 0.683 | 208 | 0.188 |
| ba_45ypg_yprr | -0.287 | 0.674 | 198 | 0.334 |
| ba_45ypg | -0.278 | 0.654 | 223 | 0.219 |
| ba_composite | -0.267 | 0.672 | 230 | 0.128 |
| ba_dominator | -0.264 | 0.672 | 164 | 0.158 |
| ba_yprr_routes | -0.249 | 0.668 | 207 | 0.252 |
| qa_ratio_scaled | -0.246 | 0.641 | 225 | 0.006 |
| draft_age | -0.230 | 0.651 | 290 | 0.359 |
| qa_magnitude | +0.175 | 0.603 | 225 | 0.026 |
| qy_zscore_adj | -0.265 | 0.676 | 207 | 0.241 |
| qy_log_magnitude | -0.257 | 0.647 | 207 | 0.113 |
| qy_ratio_scaled | -0.230 | 0.630 | 207 | 0.102 |
| qy_magnitude | +0.151 | 0.584 | 207 | 0.023 |

The YPTPA z-score adjusted variant dominates on raw signal: +22% better Spearman than ba_yptpa (-0.385 vs -0.315) and +5% better AUC (0.721 vs 0.684). The YPRR quality-adjusted variants are weaker across the board — the YPRR distribution has too much variance (std=2.307 vs 0.638 for YPTPA) for the z-scores to be discriminating.

### Imputed Signal (NaN = max + 1)

After imputation (matching the model pipeline), rankings shift somewhat because coverage matters:

| Variant | Spearman | AUC | Drift |
|---------|----------|-----|-------|
| **qa_zscore_adj** | **-0.319** | 0.659 | 0.208 |
| ba_45ypg_yprr | -0.293 | 0.668 | 0.231 |
| ba_650yards | -0.289 | 0.659 | 0.211 |
| ba_45ypg | -0.288 | 0.657 | 0.169 |
| ba_yptpa_yprr | -0.287 | 0.660 | 0.234 |
| ba_yprr_routes | -0.286 | 0.670 | 0.175 |
| ba_yptpa | -0.275 | 0.636 | 0.205 |

Z-score adjusted still leads on Spearman but the gap narrows because imputation dilutes the quality-adjustment signal.

### Residual Analysis

The critical test: how much signal remains after controlling for all other model features?

| Variant | Residual Spearman |
|---------|-------------------|
| **qa_zscore_adj** | **-0.131** |
| ba_yptpa | -0.109 |
| ba_45ypg | -0.096 |
| ba_yptpa_yprr | -0.093 |
| ba_45ypg_yprr | -0.086 |
| ba_yprr_routes | -0.079 |
| ba_composite | -0.073 |
| draft_age | -0.073 |
| ba_650yards | -0.038 |
| ba_dominator | -0.034 |
| qa_log_magnitude | -0.007 |
| qa_ratio_scaled | +0.016 |
| qa_magnitude | +0.070 |
| qy_zscore_adj | -0.086 |
| qy_log_magnitude | -0.060 |
| qy_ratio_scaled | -0.054 |
| qy_magnitude | +0.045 |

Z-score adjusted has the strongest residual signal (-0.131), meaning it contributes the most *new* information on top of the existing feature set. This is 20% more residual signal than ba_yptpa (-0.109).

However, residual analysis alone can be misleading for quality-adjusted variants — see the Efficiency Leak Analysis below.

### Era Stability

| Variant | Drift (|early - late| Spearman) |
|---------|------|
| qa_ratio_scaled | 0.006 |
| qa_log_magnitude | 0.009 |
| qa_magnitude | 0.026 |
| ba_dominator | 0.076 |
| **qa_zscore_adj** | **0.099** |
| ba_composite | 0.128 |
| ba_yptpa | 0.136 |
| ba_45ypg | 0.169 |
| draft_age | 0.359 |

The quality-adjusted variants are generally more era-stable than the binary ones. Z-score adjusted has moderate drift (0.099) — better than most binary variants and much better than draft age (0.359).

### Tier Distribution (Median Breakout Age)

How well does each variant separate tiers?

| Variant | Bust | Flex | Starter | Elite | Stud | League-Winner |
|---------|------|------|---------|-------|------|---------------|
| ba_yptpa | 20.78 | 20.06 | 19.84 | 20.27 | 19.77 | 19.21 |
| qa_zscore_adj | 19.55 | 18.38 | 18.54 | 18.80 | 18.65 | 17.43 |
| draft_age | 22.41 | 21.76 | 22.48 | 22.11 | 21.82 | 21.36 |

Z-score adjusted shows the widest Bust-to-League-Winner gap (19.55 vs 17.43 = 2.12 year spread) compared to ba_yptpa (20.78 vs 19.21 = 1.57 year spread). The quality adjustment amplifies the separation at the top end.

---

## Correlation Structure

Key finding: the binary breakout variants are all highly correlated (0.75-0.98 Spearman with each other). They're measuring essentially the same thing with minor threshold differences.

The quality-adjusted variants introduce genuinely new signal:
- qa_zscore_adj correlates 0.93 with ba_yptpa — it preserves the age signal while adding magnitude
- qa_ratio_scaled correlates only 0.12 with ba_yptpa — it's almost entirely magnitude-driven, losing the age signal
- qa_magnitude correlates 0.10 with ba_yptpa — pure magnitude, nearly orthogonal to age

This explains why qa_zscore_adj wins: it's the only quality-adjusted variant that keeps the strong age signal (which all binary variants agree on) while layering in the magnitude bonus.

---

## Why Quality Adjustment Helps

Consider two players who both break out at age 20 with YPTPA >= 1.4:

- **Player A**: 20 years old, YPTPA = 2.5 (z = 3.09). Adjusted age = 20 - 3.09 * 0.5 = **18.45**
- **Player B**: 20 years old, YPTPA = 1.45 (z = 1.44). Adjusted age = 20 - 1.44 * 0.5 = **19.28**

Binary breakout age treats these identically (both = 20). But Player A was producing at an elite level while Player B barely crossed the threshold. The z-score adjustment distinguishes them by ~0.8 years of "effective age," which is meaningful in a range where the Bust-to-League-Winner gap is only ~1.5 years.

---

## Efficiency Leak Analysis

The residual analysis above has a blind spot: quality-adjusted variants bake efficiency (YPTPA or YPRR magnitude) directly into the feature value. If that efficiency is already captured by other model features (career_yprr, career_targeted_qb_rating), the quality adjustment is double-counting — inflating the residual without adding genuinely new signal.

### Initial Test (Flawed)

An initial test appeared to show a large suppressor effect: ba_yptpa improving from -0.109 to -0.155 when controlling for magnitude. **This was an imputation artifact.** Magnitude NaN values (players who never broke out) were imputed with `max+1`, creating a fake high-magnitude signal that correlated with bust outcomes. This inflated the apparent suppressor effect.

### Corrected Test: ba_yptpa vs yprr2.0_200rt

With proper 0-imputation for magnitude (you didn't break out -> magnitude is 0), we ran a head-to-head leak test:

| Test | ba_yptpa | yprr2.0_200rt | Difference |
|------|----------|---------------|------------|
| 1. Model feats only | -0.109 | -0.098 | +0.011 |
| 2. + own magnitude | -0.111 | -0.100 | +0.011 |
| 3. + cross magnitude | -0.115 | -0.099 | +0.016 |
| 4. + other BA magnitude | -0.109 | -0.102 | +0.007 |
| 5. + own + other mag | -0.110 | -0.104 | +0.006 |
| 6. + all magnitudes | -0.113 | -0.101 | +0.012 |

### Key Findings

**No meaningful suppressor effect for either variant.** ba_yptpa goes from -0.109 to -0.111 with own magnitude (+1.8%). yprr2.0_200rt goes from -0.098 to -0.100 (+2.0%). The quality adjustment story from the initial test was an artifact.

**The gap between variants is tiny and stable.** ba_yptpa leads by ~0.01 across all control conditions. The difference doesn't grow or shrink when controlling for magnitude — these are simply two slightly different measures of the same underlying signal.

**Magnitude correlations with existing features are moderate:**

| Magnitude | career_yprr | career_tqbr | draft_capital |
|-----------|-------------|-------------|---------------|
| YPTPA at breakout | +0.534 | +0.175 | +0.334 |
| YPRR at breakout | +0.665 | +0.356 | +0.360 |

YPRR magnitude is more correlated with existing features than YPTPA magnitude, but neither is redundant enough to create a large leak.

### Bootstrap Confidence Intervals

To determine whether the ba_yptpa vs yprr2.0_200rt difference is statistically meaningful:

| Test | Mean diff | 95% CI | ba_yptpa wins |
|------|-----------|--------|---------------|
| Model feats only | -0.015 | [-0.102, +0.060] | 63.2% |
| + own magnitude | -0.011 | [-0.089, +0.065] | 60.4% |

**The 95% CI includes zero in both cases.** These two variants are statistically indistinguishable. The difference of ~0.01 in residual Spearman could easily flip with one or two players changing outcomes.

### YPRR Quality Variants

The YPRR quality-adjusted variants are weaker than their YPTPA counterparts across every test. The core issue is distributional: YPRR has std=2.307 (vs 0.638 for YPTPA), so z-scores compress into a narrow range and don't discriminate well. qy_zscore_adj correlates 0.995 with the raw ba_yprr_routes, meaning the quality adjustment adds almost nothing.

---

## YPRR Breakout Grid Search

The original YPRR breakout (2.0 YPRR + 150 routes) used a single configuration. To determine whether a different YPRR configuration could compete with ba_yptpa, we tested 16 combinations: 4 YPRR thresholds (1.8, 2.0, 2.2, 2.5) x 4 volume gates (150 routes, 200 routes, 8 games, 100 routes + 8 games).

### Top YPRR Configs vs ba_yptpa

| Config | Sp(raw) | Sp(imp) | AUC | Residual | Coverage | Drift |
|--------|---------|---------|-----|----------|----------|-------|
| **ba_yptpa** | **-0.315** | -0.275 | 0.636 | **-0.109** | **77.3%** | 0.205 |
| yprr2.0_200rt | -0.287 | -0.299 | 0.682 | -0.098 | 70.1% | 0.184 |
| yprr2.5_200rt | -0.362 | -0.285 | 0.653 | -0.089 | 47.8% | 0.192 |
| yprr1.8_200rt | -0.297 | -0.280 | 0.663 | -0.078 | 74.6% | 0.222 |
| yprr2.2_200rt | -0.294 | -0.274 | 0.658 | -0.084 | 64.3% | 0.187 |
| yprr2.5_8gm | -0.302 | -0.281 | 0.668 | -0.080 | 50.5% | 0.292 |

### Key Findings

**200-route gate is consistently the best volume filter.** Across all YPRR thresholds, the 200-route gate outperforms 150 routes, 8 games, and 100 routes + 8 games. It filters out small-sample fluky seasons more effectively than a game count.

**Higher YPRR thresholds have strong raw signal but low coverage.** yprr2.5_200rt has the best raw Spearman (-0.362) but only 47.8% coverage — too many NaN values dilute the imputed signal and make it impractical.

**No YPRR configuration beats ba_yptpa on residual.** The best YPRR residual is -0.098 (yprr2.0_200rt) vs -0.109 for ba_yptpa. The team-volume normalization in YPTPA genuinely adds signal that YPRR-only metrics can't match, even with optimized thresholds.

**YPRR breakouts show zero efficiency leak.** Residuals don't change when controlling for YPRR magnitude, confirming they're clean age signals — just weaker ones than ba_yptpa.

---

## Double-Residual Production Metric

We explored whether a metric controlling for *both* team passing volume and player route participation could outperform YPRR or YPTPA.

### Model

A linear model fit on all college WR seasons:

```
ypg = 1.787 * routes_per_game + (-0.137) * team_att_per_game + 0.057
R² = 0.706
```

The residual represents production unexplained by either opportunity source. Routes per game alone explain most variance — the team volume coefficient is small (-0.137), confirming that once you account for route participation, team passing volume adds little.

### As a Breakout Age Metric

Breakout = first season with double-residual above a percentile threshold and 8+ games:

| Config | Threshold | Sp(raw) | Sp(imp) | AUC | Residual | Coverage | Drift |
|--------|-----------|---------|---------|-----|----------|----------|-------|
| dr_p75 | 3.94 | -0.259 | -0.247 | 0.647 | -0.033 | 76.6% | 0.300 |
| dr_p80 | 5.31 | -0.270 | -0.259 | 0.653 | -0.041 | 75.6% | 0.289 |
| dr_p85 | 7.33 | -0.268 | -0.266 | 0.657 | -0.046 | 74.2% | 0.255 |
| dr_p90 | 10.82 | -0.275 | -0.287 | 0.676 | -0.076 | 71.1% | 0.264 |

The best double-residual breakout (dr_p90) reaches -0.076 residual — weaker than even the YPRR breakouts, with worse drift.

### As a Career Feature (Replacing career_yprr)

Career double-residual (game-weighted average across all seasons) vs career_yprr:

| Feature | Spearman | AUC | Residual (excl. career_yprr) | Drift |
|---------|----------|-----|------|-------|
| career_yprr | +0.291 | 0.659 | +0.024 | 0.209 |
| career_double_residual | +0.286 | 0.652 | +0.033 | 0.202 |

Spearman correlation between the two: **+0.961**. They're nearly the same feature. Swapping career_yprr for career_double_residual changes ba_yptpa's residual by exactly **0.000**.

### Why the Double Residual Doesn't Help

The model `ypg ~ rpg + team_att_pg` has R²=0.706, meaning routes alone explain most of the yards variance. The team volume adjustment contributes very little incremental information. Once you normalize by routes (which YPRR already does), further normalizing by team volume is redundant — YPRR already captures what the double residual captures, plus it's simpler and more interpretable.

---

## Breakout Magnitude as Standalone Features

The quality-adjusted variants failed because they *mixed* age and magnitude into one number, double-counting efficiency. But what about using breakout magnitude as a **separate, standalone feature** alongside a binary breakout age? This lets the model learn the age-magnitude interaction without baking efficiency into the age signal.

We evaluated 4 magnitude features — the YPTPA and YPRR values at each breakout type — as potential additions to the feature set.

### Magnitude Feature Definitions

At the moment a player first crosses a breakout threshold, we record both their YPTPA and YPRR values for that season:

| Feature | Definition |
|---------|-----------|
| mag_yptpa_at_yptpa | YPTPA value at the YPTPA breakout season (1.4+ YPTPA, 8+ games) |
| mag_yprr_at_yptpa | YPRR value at the YPTPA breakout season |
| mag_yptpa_at_yprr | YPTPA value at the YPRR breakout season (2.0+ YPRR, 200+ routes) |
| mag_yprr_at_yprr | YPRR value at the YPRR breakout season |

### Standalone Evaluation

| Feature | Spearman | AUC | Residual (after model feats) | Drift |
|---------|----------|-----|------------------------------|-------|
| mag_yptpa_at_yptpa | +0.190 | 0.585 | -0.056 | 0.057 |
| mag_yprr_at_yptpa | +0.201 | 0.600 | -0.024 | 0.112 |
| mag_yptpa_at_yprr | +0.210 | 0.590 | -0.039 | 0.090 |
| mag_yprr_at_yprr | +0.237 | 0.625 | +0.029 | 0.023 |

**YPTPA at YPTPA-breakout has the strongest residual signal (-0.056).** This means it contributes the most genuinely new information after controlling for all current model features. YPRR at YPRR-breakout (+0.029) is actually slightly redundant — its information is already captured by career_yprr.

The negative residuals for the YPTPA-based magnitudes confirm that these features add unique signal: players with higher YPTPA at breakout tend to land in higher tiers, even after accounting for everything else in the model.

### Cross-Breakout Correlation

YPRR at breakout and YPTPA at breakout are moderately correlated (Spearman = +0.661) — they capture overlapping but distinct signals. A player can dominate on one metric but not the other, so including both adds information.

### Feature Set Total Signal Comparison

To determine the best combination, we computed the sum of |residual Spearman| across all features in the model for each proposed feature set. Higher = more total unique information.

| Feature Set | Sum |residual Spearman| |
|-------------|------------------------------|
| Current (ba_yptpa) | 0.788 |
| ba_yprr only | 0.801 |
| ba_yprr + mag_yptpa | 0.819 |
| ba_yprr + mag_yprr | 0.832 |
| **ba_yprr + both magnitudes** | **0.911** |
| ba_yptpa + mag_yptpa | 0.816 |
| ba_yptpa + mag_yprr | 0.804 |

**Switching to ba_yprr and adding both magnitude features increases total predictive signal by 15.6%** (0.911 vs 0.788). This is a substantial improvement — by far the largest gain we've found in any feature engineering exploration.

The ba_yprr base outperforms ba_yptpa in this context because YPRR-based breakout age is less correlated with the magnitude features than YPTPA-based breakout age, creating more complementary signal.

### Why Both Magnitudes Help

The two magnitudes capture different aspects of breakout quality:
- **YPTPA at breakout** captures team-volume-normalized production efficiency. It tells you "how well did this player produce relative to his team's passing environment?" This is partially orthogonal to career_yprr (residual -0.056).
- **YPRR at breakout** captures raw per-route efficiency. While more correlated with career_yprr (residual +0.029 alone), it adds value *in combination* with YPTPA magnitude because together they describe the full efficiency picture at the breakout moment.

---

## Recommendations

1. **Switch from ba_yptpa to ba_yprr (yprr2.0_200rt) as the breakout age feature.** While ba_yptpa has slightly stronger standalone residual (-0.109 vs -0.098), ba_yprr creates a more complementary feature set when paired with magnitude features. The two variants are statistically indistinguishable on their own (bootstrap 95% CI includes zero), so the deciding factor is how well they compose with the rest of the model.

2. **Add breakout YPTPA magnitude and breakout YPRR magnitude as standalone features.** Together with ba_yprr, this increases total feature set signal by 15.6% (0.788 -> 0.911). These features capture *how dominant* the breakout was — information that's partially orthogonal to existing model features.

3. **Do not use quality-adjusted variants.** They look better in naive evaluation but the improvement comes from baking in efficiency signal that's already available through career_yprr and career_targeted_qb_rating. Separate magnitude features achieve the same goal without double-counting.

4. **Do not replace career_yprr with a double-residual metric.** They correlate 0.961 and produce identical model behavior. The team-volume adjustment adds nothing on top of route normalization for career features. YPRR is simpler and more interpretable.

5. **Draft age adds nothing.** It ranks near the bottom on residual signal (-0.073) and has the worst era drift of any variant (0.359). It's fully dominated by breakout age.

6. **The binary variants are largely redundant with each other.** All correlate 0.75-0.98 with each other. The choice between ba_yptpa and ba_yprr matters less than what you pair it with.

7. **Use 200-route gate for YPRR breakout.** It consistently outperforms 150 routes, 8 games, and 100rt+8gm across all thresholds in the grid search.

8. **Proposed new feature set:**
   - `draft_capital`
   - `career_targeted_qb_rating`
   - `career_yprr`
   - `career_catch_pct_adot_adj`
   - `best2_contested_catch_rate`
   - `career_avoided_tackles_pg`
   - `breakout_age` (ba_yprr: age at first season with 2.0+ YPRR and 200+ routes)
   - `breakout_yptpa` (YPTPA value at the YPRR breakout season) **NEW**
   - `breakout_yprr` (YPRR value at the YPRR breakout season) **NEW**
