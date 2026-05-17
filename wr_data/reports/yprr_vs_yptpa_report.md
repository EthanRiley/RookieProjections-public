# YPRR vs Yards Per Team Pass Attempt

## Overview

This report compares two candidate efficiency features for predicting WR dynasty tier outcomes:

- **Career YPRR** (Yards Per Route Run): total career receiving yards / total career routes run. A per-route efficiency metric computed from PFF charting data.
- **Best YPTPA** (Yards Per Team Pass Attempt): best single-season receiving yards / team pass attempts that season. A market-share-style metric that normalizes production against team passing volume.

Both features attempt to measure "how efficient is this receiver?" but from different angles. YPRR measures per-route efficiency directly; YPTPA measures what fraction of a team's passing offense a receiver commanded per attempt.

## Head-to-Head Comparison

| Metric | YPRR | YPTPA | Winner |
|--------|------|-------|--------|
| Spearman w/ tier | +0.292 | +0.212 | YPRR |
| Standalone AUC (>=Elite) | 0.660 | 0.587 | YPRR |
| Elastic net survival | 0/3 | 2/3 | YPTPA |
| Era stability (lower = better) | 0.305 | 0.145 | YPTPA |
| Feature eval composite rank | #4 | #21 | YPRR |

YPRR wins on the two metrics that matter most for prediction (Spearman and AUC) by a wide margin. YPTPA has slightly better era stability and survives elastic net at more regularization strengths, but these advantages are modest.

## Correlation Between Features

Spearman correlation between YPRR and YPTPA: **0.676**

This is high but not extreme — they share substantial information but are not identical. The question is whether the non-shared portion of YPTPA carries any predictive signal.

## Residual Analysis: The Key Finding

This is where the case becomes clear-cut.

**After removing YPRR's signal from YPTPA:**
- Residual Spearman with tier: **+0.067**
- Interpretation: once you know a player's YPRR, knowing their YPTPA tells you essentially **nothing additional** about their dynasty outcome.

**After removing YPTPA's signal from YPRR:**
- Residual Spearman with tier: **+0.198**
- Interpretation: even after controlling for YPTPA, YPRR retains **meaningful predictive signal** that YPTPA cannot capture.

This asymmetry is the core result. YPTPA's information is almost entirely a subset of YPRR's. YPRR contains signal that YPTPA misses, but not vice versa.

## Why YPRR Is the Superior Metric

1. **Direct measurement vs. proxy.** YPRR directly measures what we care about: how many yards does this receiver produce per opportunity to catch a pass? YPTPA is a proxy — it divides by team pass attempts, which includes plays where the receiver wasn't even on the field or wasn't targeted. YPRR normalizes against the receiver's actual route volume.

2. **Controls for snap share.** A receiver who runs 25 routes per game and produces 2.5 YPRR is demonstrably efficient. The same receiver might have a mediocre YPTPA if his team throws 45 times per game but he only runs routes on 60% of pass plays. YPRR correctly credits his efficiency; YPTPA dilutes it.

3. **Robust to team context.** YPTPA is confounded by team passing volume. A receiver on a run-heavy team gets inflated YPTPA (fewer team pass attempts in the denominator). YPRR doesn't have this problem because routes run already reflects the receiver's actual involvement.

4. **Career vs. single season.** YPRR is computed over the full career (total yards / total routes), smoothing out single-season noise. YPTPA uses only the best single season, making it more susceptible to outlier performances.

## Why Not Include Both?

Given that YPTPA survives elastic net and has better era stability, one might argue for including both. The residual analysis rules this out:

- Adding YPTPA to a model that already has YPRR contributes a residual Spearman of only +0.067 — indistinguishable from noise.
- An additional feature with near-zero incremental signal only adds model complexity and overfitting risk.
- With only ~200 complete cases, every unnecessary feature costs statistical power.

## Conclusion

YPRR is strictly superior to YPTPA as a predictive feature. YPTPA is a noisier, less direct proxy for the same underlying signal (receiver efficiency). All of YPTPA's predictive value is subsumed by YPRR, while YPRR carries meaningful signal that YPTPA cannot capture. Only YPRR belongs in the model.
