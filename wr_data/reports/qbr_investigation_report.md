# Career Targeted QBR: A Critical Investigation

**Date**: 2026-05-13
**Scope**: Is `career_targeted_qb_rating` warranted as a feature in the v11 WR dynasty model?
**Verdict**: The feature has genuine signal, but it is not the second most important feature. It is the **least important** of the six v11 features by most rigorous measures. Its univariate signal is almost entirely confounded with draft capital, and after controlling for the other model features, it contributes zero or negative incremental information.

---

## Executive Summary

The claim under investigation: *"Aside from draft capital, career_targeted_qb_rating is THE MOST important feature in the model."*

This claim is **not supported** by the evidence. Across seven independent analyses:

| Test | Result | Implication |
|------|--------|-------------|
| Univariate Spearman | +0.313 (p<0.001) | Strong raw signal -- but weaker than YPRR (+0.344) and catch% aDOT adj (+0.346) |
| Residual after DC | +0.001 (51.6% bootstrap positive) | **Zero incremental signal** once draft capital is controlled |
| Residual after all other features | -0.077 (16.4% bootstrap positive) | **Negative** -- other features already capture what QBR measures |
| Partial Spearman (controlling DC) | +0.134 (p=0.038) | Weak but significant; the only positive result |
| LOO-AUC ablation | Removing QBR **improves** AUC at all thresholds | QBR actively hurts logistic regression models |
| XGBoost importance | **Last place** (0.097 vs 0.332 for DC) | Tree-based models use it the least |
| Permutation importance | -0.004 (shuffling QBR doesn't hurt) | Model performance is invariant to QBR |

The feature was selected during v4 model development using a different base feature set (draft_capital + breakout_age + breakout magnitudes). That selection decision was sound *for that context* -- the evidence showed +0.041 residual after DC + breakout_age with 74% bootstrap positive. But the feature set has since changed to v11 (breakout features replaced by best1_yprr_graduated and best2_catch_pct_adot_adj), and **the evidence does not transfer**. In the v11 feature set, QBR is redundant.

---

## 1. What is Targeted QBR?

Targeted QBR is the passer rating on attempts where a specific receiver was targeted. It's a PFF-computed metric that captures:
- Whether the pass was completed (catch rate)
- How many yards were gained
- Whether it resulted in a touchdown
- Whether it was intercepted

At the season level (n=4,844 qualified seasons with 200+ routes), QBR is driven by:

| Component | Correlation with QBR |
|-----------|---------------------|
| Touchdowns | **+0.628** |
| Caught percent | **+0.546** |
| PFF pass route grade | **+0.512** |
| YPRR | **+0.467** |
| YAC per reception | +0.313 |
| Drop rate | -0.278 |
| aDOT | -0.034 |

**QBR is not a standalone metric.** It's a weighted composite of touchdowns, catch rate, and route quality. This matters because other model features (YPRR, catch% aDOT adj, CCR) already measure the same underlying skills through different lenses.

*See: Panel E in `wr_data/charts/qbr_investigation.png`*

---

## 2. The Raw Signal Is Real But Confounded

### 2.1 Univariate Signal

QBR has a genuine univariate relationship with dynasty outcomes:

| Tier | n | Mean QBR | Median QBR |
|------|---|----------|------------|
| Bust | 184 | 107.9 | 107.8 |
| Flex | 17 | 117.2 | 112.8 |
| Starter | 7 | 105.8 | 103.7 |
| Elite | 20 | 122.0 | 125.3 |
| Stud | 7 | 123.9 | 121.0 |
| League-Winner | 6 | **133.9** | **136.7** |

There's a clear monotonic trend from Bust to League-Winner (the Starter dip aside, which is a small-sample artifact at n=7). Spearman rho = +0.313 (p < 0.001).

But this signal is largely *borrowed from draft capital*. QBR correlates +0.393 with draft capital. Players drafted high (who are more likely to succeed) also tend to have higher QBR. The question is: does QBR tell us anything *beyond* what draft capital already tells us?

*See: Panels A and B in `wr_data/charts/qbr_investigation.png`*

### 2.2 The Confounding Evidence

**Residual after draft capital alone: rho = +0.001 (p = 0.987)**

This is the most damning finding. After regressing tier outcome on draft capital and computing residuals, QBR has *zero* correlation with the residuals. In 5,000 bootstrap iterations, only 51.6% are positive -- coin-flip territory.

What this means: if you already know where a player was drafted, knowing their QBR gives you **no additional information** about their dynasty outcome. The entire QBR-to-tier signal flows through the DC channel.

**Partial Spearman (controlling DC): +0.134 (p = 0.038)**

This is the one positive result. The partial correlation is statistically significant, suggesting QBR carries *some* independent information. However:
- The effect is small (0.134 vs 0.313 raw -- QBR loses 57% of its signal after DC control)
- It doesn't survive the residual test, which is more stringent
- A p-value of 0.038 with no multiple testing correction on a dataset of 205 players is not definitive

### 2.3 Residual After Full Model

When all other v11 features are used as the base (DC + YPRR + CCR + AT/Rec + CPA), QBR's residual drops to **-0.077 (16.4% bootstrap positive)**. This is clearly negative -- the other features already capture everything QBR measures, plus more.

*See: Panel C in `wr_data/charts/qbr_investigation.png`*

---

## 3. The Interaction Effect: QBR Only Matters for Early Picks

One genuinely interesting finding: QBR has a strong interaction with draft capital.

| DC Quartile | QBR vs Tier rho | n |
|-------------|-----------------|---|
| Bottom 25% (late picks) | +0.034 (n.s.) | 52 |
| 25-50% | +0.172 | 51 |
| 50-75% | +0.219 | 51 |
| Top 25% (early picks) | **+0.302** (p=0.002) | 51 |

Among late-round picks, QBR is noise. Among first/second-round picks, QBR meaningfully differentiates outcomes. Elite+ hit rate within high-DC players:
- Above-median QBR: **50.0%**
- Below-median QBR: **11.5%**

This is a real finding and suggests QBR's value (if any) is as an *interaction term* with draft capital, not as a standalone linear predictor. XGBoost can potentially capture this, which is why the feature may have survived despite looking weak in linear analyses.

However, permutation importance in XGBoost was -0.004, suggesting even the tree model doesn't meaningfully use this interaction in practice.

*See: Panel D in `wr_data/charts/qbr_verdict.png` and Panel B in `wr_data/charts/qbr_confounding.png`*

---

## 4. XGBoost Says: Least Important Feature

The v11 ensemble uses XGBoost for 40% of its weight. XGBoost feature importance rankings:

| Rank | Feature | XGB Importance |
|------|---------|---------------|
| 1 | draft_capital | **0.332** |
| 2 | best2_catch_pct_adot_adj | 0.167 |
| 3 | best1_yprr_graduated | 0.162 |
| 4 | best2_contested_catch_rate | 0.141 |
| 5 | best2_avoided_tackles_per_rec | 0.101 |
| **6** | **career_targeted_qb_rating** | **0.097** |

QBR ranks **dead last**. It gets the least split time of any feature in the tree ensemble.

Permutation importance (shuffling each feature and measuring AUC drop, 50 iterations per feature):

| Feature | Mean AUC Drop | Interpretation |
|---------|---------------|----------------|
| draft_capital | +0.174 | Critical |
| best2_contested_catch_rate | +0.018 | Helpful |
| career_targeted_qb_rating | **-0.004** | **Irrelevant** |
| best1_yprr_graduated | -0.004 | Borderline |
| best2_catch_pct_adot_adj | -0.000 | Neutral |
| best2_avoided_tackles_per_rec | -0.027 | Overfitting |

Shuffling QBR doesn't hurt the model. The XGBoost component appears to have learned workarounds using the other features.

---

## 5. LOO-AUC: QBR Consistently Hurts

Leave-one-year-out AUC with logistic regression across multiple thresholds:

| Threshold | DC Only | DC + QBR | DC + YPRR | Full (no QBR) | Full (with QBR) |
|-----------|---------|----------|-----------|---------------|-----------------|
| >=Flex | 0.877 | 0.865 (-0.012) | 0.885 (+0.008) | 0.884 | 0.877 (-0.007) |
| >=Starter | 0.880 | 0.867 (-0.014) | 0.884 (+0.003) | 0.873 | 0.845 (-0.028) |
| >=Elite | 0.901 | 0.889 (-0.012) | 0.905 (+0.004) | 0.874 | 0.863 (-0.011) |
| >=Stud | 0.882 | 0.874 (-0.008) | 0.910 (+0.029) | 0.838 | 0.841 (+0.002) |

Adding QBR to DC hurts at every threshold. Adding QBR to the full model hurts at 3 of 4 thresholds (only >=Stud shows a tiny +0.002 gain).

For comparison, adding YPRR to DC helps at every threshold, often substantially.

**Important caveat**: These tests use logistic regression, not the Bayesian ordinal + XGBoost ensemble. The production model may capture non-linear patterns that logistic regression misses. But the linear evidence is uniformly negative.

---

## 6. The Redundancy Problem

QBR has Spearman rho = +0.575 with `best2_catch_pct_adot_adj`. This is the highest inter-feature correlation in the model. These two features are measuring nearly the same thing: *does the receiver catch the ball?*

The other features already cover QBR's components:
- Catch rate -> `best2_catch_pct_adot_adj` (direct measure, aDOT-adjusted)
- Route quality/YPRR -> `best1_yprr_graduated` (direct measure, age-adjusted)
- Contested catches -> `best2_contested_catch_rate` (direct measure)

What's left that's unique to QBR? Primarily touchdowns (the strongest season-level correlate at +0.628). But touchdowns are a noisy, opportunity-dependent stat that doesn't cleanly separate talent from situation.

R^2 of QBR reconstructed from other college features: **0.370** (0.402 when including DC). After removing the portion of QBR explainable by other features, the residual (unique QBR information) predicts tier at rho = +0.065 (p = 0.356). **The unique information in QBR is not predictive.**

*See: Panel G in `wr_data/charts/qbr_investigation.png`*

---

## 7. Notable Cases

### QBR Got It Right (High QBR, Elite+ Outcome)
- Justin Jefferson: QBR 137, DC 8.0 -> League-Winner
- Ja'Marr Chase: QBR 136, DC 9.0 -> League-Winner
- CeeDee Lamb: QBR 144, DC 8.2 -> League-Winner
- Jaylen Waddle: QBR 151, DC 8.9 -> Stud

These are *all* high-DC players. QBR confirms draft capital but doesn't differentiate beyond it.

### QBR Missed (Low QBR, Elite+ Outcome)
- **Amon-Ra St. Brown**: QBR 107 -> League-Winner (the biggest QBR miss)
- **D.J. Moore**: QBR 96 -> Elite
- **Zay Flowers**: QBR 100 -> Elite
- Calvin Ridley: QBR 101 -> Elite

### QBR Fooled (High QBR, Bust)
- **Henry Ruggs III**: QBR 149, DC 8.5 -> Bust
- Mecole Hardman: QBR 135, DC 6.8 -> Bust
- Marvin Mims: QBR 134, DC 6.5 -> Bust
- Terrace Marshall Jr.: QBR 131, DC 6.7 -> Bust
- Jalin Hyatt: QBR 131, DC 6.3 -> Bust

High-QBR busts often share a profile: deep-threat speedsters who generated touchdowns through scheme/speed rather than route-running craft.

*See: `wr_data/charts/qbr_player_scatter.png`*

---

## 8. Era Stability

| Feature | Early (<=2020) | Late (>=2022) | Drift |
|---------|----------------|---------------|-------|
| draft_capital | +0.555 | +0.534 | 0.021 |
| best2_avoided_tackles_per_rec | +0.152 | +0.054 | 0.098 |
| best1_yprr_graduated | +0.285 | +0.400 | 0.115 |
| **career_targeted_qb_rating** | **+0.216** | **+0.370** | **0.155** |
| best2_catch_pct_adot_adj | +0.228 | +0.438 | 0.210 |
| best2_contested_catch_rate | +0.076 | +0.345 | 0.269 |

QBR's drift of 0.155 is moderate. The signal appears stronger in recent classes, which is encouraging for forward-looking validity. But with only 3 holdout years (2022-2024) and small class sizes, this could be noise.

*See: Panel D in `wr_data/charts/qbr_confounding.png`*

---

## 9. Why the Original Research Found Different Results

The v4 feature selection report found career_targeted_qb_rating had +0.041 residual after DC + breakout_age (74% bootstrap positive) and +0.023 LOO-AUC delta. That analysis was sound but used a different base:

| v4 Base | v11 Base (current) |
|---------|-------------------|
| draft_capital | draft_capital |
| breakout_age | best1_yprr_graduated |
| breakout_yprr | best2_catch_pct_adot_adj |
| breakout_yptpa | best2_contested_catch_rate |
| best_contested_catch_rate | best2_avoided_tackles_per_rec |

The v4 base lacked any direct catch-rate or efficiency metric — breakout features measure *when* production happened, not *how well* the receiver caught the ball. QBR filled a genuine gap in that feature set.

In v11, `best2_catch_pct_adot_adj` and `best1_yprr_graduated` now directly measure catch reliability and route efficiency. These features cover the same ground QBR covered in v4, making QBR redundant.

**The feature selection was never re-validated after the v4->v11 feature set changes.** This is the root cause of the discrepancy.

---

## 10. Conclusions and Recommendations

### Is QBR warranted?

**As a standalone feature**: Barely. It has genuine univariate signal and a meaningful interaction with draft capital among early picks. But its incremental value after the other v11 features is zero or negative by every rigorous test.

**As "the second most important feature"**: No. By XGBoost importance, permutation importance, LOO-AUC ablation, and residual analysis, it ranks **5th or 6th** out of 6 features. The true importance ranking in v11 is approximately:

1. draft_capital (dominant, irreplaceable)
2. best1_yprr_graduated (strongest college predictor univariately)
3. best2_catch_pct_adot_adj (strong univariate + covers QBR's catch-rate dimension)
4. best2_contested_catch_rate (strongest orthogonal signal, 97.9% bootstrap)
5. best2_avoided_tackles_per_rec (weakest but genuinely orthogonal)
6. career_targeted_qb_rating (redundant with the above)

### Recommendations

1. **Re-run full holdout evaluation with QBR removed** using the actual Bayesian + XGBoost ensemble. The logistic regression tests above may understate QBR's interaction value in the tree model. If holdout metrics (LogLoss, Brier, AUC) are unchanged or improve, drop QBR.

2. **If keeping QBR**, do not call it the second most important feature. It is the least important feature in the current feature set, and its signal is almost entirely subsumed by draft capital and catch% aDOT adjusted.

3. **Consider replacing QBR** with something genuinely orthogonal. The model currently lacks: a speed/athleticism dimension, a target market-share metric, or a route-tree diversity measure. Any of these would add more novel information than QBR provides.

4. **Re-validate all features** whenever the feature set changes. The v4 selection results do not transfer to v11.

---

## Visualizations

| File | Description |
|------|-------------|
| `wr_data/charts/qbr_investigation.png` | 8-panel overview: distributions, correlations, residuals, collinearity |
| `wr_data/charts/qbr_player_scatter.png` | Named player scatters showing hits, misses, and false positives |
| `wr_data/charts/qbr_confounding.png` | 4-panel confounding analysis: quartile effects, DC interaction, era stability |
| `wr_data/charts/qbr_verdict.png` | 4-panel verdict: rankings, signal decomposition, components, interaction |
