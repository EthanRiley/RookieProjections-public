# Career Caught Percent: Model Inclusion Report

## The Feature

`career_caught_percent` is the game-weighted average of a receiver's catch rate across all college seasons. It measures hands reliability — what percentage of catchable targets does this receiver actually haul in?

## Univariate Signal

| Metric | Value |
|--------|-------|
| Spearman w/ tier | **+0.238** |
| AUC (>=Elite) | **0.692** |
| Elastic net survival | **2/3** regularization strengths |
| Era drift | **0.095** (very stable) |
| Composite rank | **#11** of 95 features |

This is a top-tier feature. The Spearman of +0.238 is 8th among all features. The AUC of 0.692 is 6th. It survives elastic net at 2 of 3 regularization strengths, which only 8 features achieve. And critically, the era stability is excellent — a drift of just 0.095, meaning catch percentage predicts outcomes equally well in the 2016-2019 era and the 2020-2024 era.

## Tier Separation

| Tier | Median Catch % | Mean Catch % |
|------|---------------|-------------|
| Bust | 63.5 | 62.9 |
| Flex | 66.7 | 67.2 |
| Starter | 62.8 | 63.0 |
| Elite | 65.7 | 66.3 |
| Stud | 67.7 | 68.4 |
| League-Winner | 68.6 | 69.8 |

The trend is clear: higher-tier outcomes correlate with higher catch rates. League-Winners average 69.8% vs Busts at 62.9%. The Starter tier is a mild exception (small sample of 7 players), but the monotonic trend from Elite through League-Winner is clean.

## Collinearity with Other Model Features

| Model Feature | Spearman with Catch % |
|--------------|----------------------|
| career_targeted_qb_rating | +0.549 |
| career_avoided_tackles_pg | +0.376 |
| career_yprr | +0.330 |
| draft_capital | +0.257 |
| breakout_age | -0.236 |
| best2_contested_catch_rate | +0.165 |

The highest correlation is with targeted QBR (+0.549) — this makes sense because catching more passes directly improves the quarterback's rating when targeting you. But 0.549 is moderate, not redundant. The correlation with YPRR (+0.330) is even lower — catch rate and yards per route measure different things (reliability vs. explosiveness).

## Residual Analysis

This is where the case gets nuanced.

| Controlled For | Catch % Residual Spearman w/ Tier |
|---------------|----------------------------------|
| Nothing (raw) | +0.238 |
| QBR only | +0.038 |
| YPRR only | +0.119 |
| QBR + YPRR | +0.027 |
| ALL other model features | -0.048 |

After controlling for targeted QBR and YPRR together, catch percentage retains only +0.027 residual correlation — close to zero. After controlling for ALL other model features, it's -0.048. This means the unique predictive information in catch percentage is largely captured by the combination of other features already in the model.

However, the reverse analysis tells an important story:

| Feature | Residual After Removing Catch % |
|---------|-------------------------------|
| QBR | +0.252 (from +0.306) |
| YPRR | +0.230 (from +0.292) |

QBR and YPRR both retain substantial signal after controlling for catch percentage. This confirms that catch percentage is the more redundant of the three — not the other way around.

## Why It Stays In the Model

Despite the low residual signal, catch percentage earns its place for three reasons:

1. **Elastic net survival (2/3 strengths).** This is the strongest multivariate endorsement possible from our validation pipeline. When regularized models retain a feature at multiple penalty levels, it means the feature contributes to prediction even in the presence of all other features. Only 8 of 95 features survive at 2+ strengths.

2. **Exceptional era stability (drift = 0.095).** This is the 5th most stable feature in the entire dataset. In a model where several features (QBR, YPRR, contested catch rate) have concerning era drift (0.275-0.318), having a stable feature that captures a related but distinct signal provides robustness. If the modern-era inflation of QBR and YPRR signal turns out to be noise, catch percentage is the backstop.

3. **Distinct conceptual signal.** Catch percentage measures hands reliability — a physical trait (hand size, hand-eye coordination, concentration) that is distinct from route-running efficiency (YPRR), quarterback trust (QBR), contested-catch ability, and elusiveness. Even if the statistical residual is small, the underlying trait being measured is orthogonal enough to justify inclusion in a 7-feature model.

## Comparison: Career vs Best-2 vs Best Single Season

| Aggregation | Spearman | AUC | Enet | Drift |
|------------|----------|-----|------|-------|
| Career | **+0.238** | **0.692** | 2/3 | 0.095 |
| Best 2 | +0.215 | 0.680 | 2/3 | **0.006** |
| Best single | +0.127 | 0.659 | 2/3 | 0.050 |

Career wins on signal. Best-2 has remarkably stable era drift (0.006), but career's 0.095 is already excellent. Career is retained because the stronger signal outweighs the marginal stability gain.

## Conclusion

Catch percentage is a moderately strong univariate predictor with excellent era stability and multivariate survival. Its residual signal after controlling for QBR and YPRR is low (+0.027), but its elastic net survival, conceptual distinctness, and temporal robustness justify its inclusion. It serves as a stable anchor in a feature set where several other features have era instability concerns.
