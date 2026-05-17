# RB Athleticism & Combine Data Analysis

## Summary

We investigated whether NFL Combine measurements predict dynasty RB outcomes, testing raw metrics, RAS (Relative Athletic Score), unsupervised PCA, and a supervised correlation-weighted composite. The short answer: athleticism has weak signal for RBs, concentrated in size-speed composites rather than any single drill. The effect is small compared to college production features and largely evaporates under cross-validation. With updated resolution logic (186 resolved players, up from 106), the signal is even weaker than initially estimated.

---

## 1. Data Coverage

Combine measurements were matched to our 197-player RB dataset (2016-2024 draft classes) via nflverse, using normalized name matching with +/-1 year tolerance.

| Metric | All Players (197) | Resolved (186) |
|--------|:-----------------:|:--------------:|
| Height | 159 | 150 |
| Weight | 160 | 151 |
| 40-yard dash | 136 | 132 |
| Vertical jump | 138 | 134 |
| Broad jump | 126 | 123 |
| Bench press | 103 | 100 |
| 3-cone drill | 61 | 61 |
| Shuttle | 65 | 65 |
| **Any metric** | **161** | **152** |

36 players have no combine data at all. Notable missing players include Joe Mixon (skipped combine) and Najee Harris (limited participation not captured in dataset).

**Key limitation:** Cone and shuttle are missing for >55% of players, making them unusable in any composite that requires complete cases.

---

## 2. Univariate Signal

Spearman correlation with tier ordinal (Bust=0 through League-Winner=5), resolved players only.

| Metric | Spearman | p-value | AUC (>=Elite) | n |
|--------|:--------:|:-------:|:-------------:|:-:|
| Speed score | +0.159 | 0.069 | 0.632 | 131 |
| Broad jump | +0.102 | 0.262 | 0.627 | 123 |
| Weight | +0.090 | 0.272 | 0.614 | 151 |
| Shuttle | -0.089 | 0.482 | 0.595 | 65 |
| 3-cone drill | +0.068 | 0.603 | 0.546 | 61 |
| 40-yard dash | -0.066 | 0.452 | 0.538 | 132 |
| Vertical | +0.043 | 0.625 | 0.539 | 134 |
| Height | +0.015 | 0.858 | 0.591 | 150 |
| Bench press | -0.133 | 0.188 | 0.437 | 100 |

**No individual combine metric reaches statistical significance.** Speed score (weight * 200 / forty^4) is the strongest at p=0.069 — the only metric approaching significance. It combines mass and speed into a single measure, which matters because raw 40 time is nearly useless (AUC 0.538) — heavier backs run slower, and the 40 doesn't adjust for that.

Bench press is *negatively* correlated with outcomes. Vertical, which had marginal signal in the smaller sample (+0.156), dropped to near-zero (+0.043) with more data.

### Tier Means

| Metric | League-Winner | Stud | Elite | Bust |
|--------|:------------:|:----:|:-----:|:----:|
| 40-yard dash | 4.46 (n=6) | 4.55 (n=6) | 4.53 (n=12) | 4.52 (n=45) |
| Weight | 218.3 (n=6) | 217.7 (n=7) | 219.8 (n=12) | 213.3 (n=50) |
| Speed score | 110.2 (n=6) | 101.8 (n=6) | 104.5 (n=12) | 101.7 (n=45) |
| Vertical | 36.2 (n=6) | 34.1 (n=6) | 34.6 (n=12) | 34.0 (n=43) |
| Broad jump | 121.8 (n=5) | 123.2 (n=5) | 122.8 (n=12) | 121.0 (n=42) |

The differences are small. League-Winners run a 4.46 average 40; Busts run a 4.52. That's 0.06 seconds — within normal measurement noise. The clearest separation is in speed score (League-Winners at 110.2 vs everyone else around 101-104), driven by League-Winners being both heavier and faster.

---

## 3. Composite Approaches

### 3a. RAS (Relative Athletic Score)

RAS computes percentile rank for each metric among all combine RBs (2000-2026, n=779), then averages available percentiles on a 0-10 scale. It handles missing data gracefully by averaging only the metrics a player tested in.

**Hit rate by RAS quartile (152 resolved players with RAS):**

| Quartile | Hit Rate | n |
|----------|:--------:|:-:|
| Q1 (low) | 15.8% | 6/38 |
| Q2 | 10.5% | 4/38 |
| Q3 | 23.7% | 9/38 |
| Q4 (high) | **26.3%** | 10/38 |

There's a gradient from bottom to top quartile (15.8% to 26.3%), but it's modest — top-quartile backs hit at only 1.7x the rate of bottom-quartile. The signal is weaker than initially estimated with the smaller sample (which showed 43.5% vs 17.4%).

**Top 10 RAS in our dataset:**

| Player | Year | RAS | Outcome |
|--------|:----:|:---:|---------|
| Hassan Haskins | 2022 | 9.3 | Bust |
| Saquon Barkley | 2018 | 8.6 | League-Winner |
| Isaac Guerendo | 2024 | 8.5 | TBD |
| Rodney Anderson | 2019 | 8.3 | Bust |
| Breece Hall | 2022 | 8.0 | Elite |
| Tyrone Tracy Jr. | 2024 | 7.7 | TBD |
| AJ Dillon | 2020 | 7.6 | Flex |
| Jaylen Wright | 2024 | 7.6 | TBD |
| Rachaad White | 2022 | 7.6 | Elite |
| Chris Evans | 2021 | 7.4 | Bust |

### 3b. PCA (Unsupervised)

PCA was run on 6 metrics with sufficient coverage: 40, vertical, broad jump, speed score, weight, height. Requires complete cases (n=114 resolved).

**Explained variance:**

| Component | Variance | Cumulative |
|-----------|:--------:|:----------:|
| PC1 | 40.2% | 40.2% |
| PC2 | 29.2% | 69.4% |
| PC3 | 19.3% | 88.7% |
| PC4-6 | 11.3% | 100% |

**PC1 loadings** (general athleticism — higher = more athletic):

| Feature | Loading |
|---------|:-------:|
| Speed score | +0.535 |
| Broad jump | +0.494 |
| Vertical | +0.409 |
| 40 (negated) | +0.375 |
| Height | +0.320 |
| Weight | +0.242 |

PC1 is a balanced general athleticism factor. Everything loads positively — faster, more explosive, bigger players score higher. Speed score and broad jump dominate.

**PC2 loadings** (size vs speed tradeoff):

| Feature | Loading |
|---------|:-------:|
| Weight | +0.644 |
| Height | +0.539 |
| Speed score | +0.041 |
| Broad jump | -0.151 |
| Vertical | -0.264 |
| 40 (negated) | -0.448 |

PC2 separates large-framed backs (Derrick Henry, AJ Dillon) from small explosive backs (Devin Singletary, Nyheim Hines). Weight and height load positive, speed/explosiveness load negative.

### 3c. Supervised Composite (Correlation-Weighted)

Rather than equal PCA weighting, weight each metric by its Spearman correlation with tier ordinal. Only positive correlations contribute.

**Weights:**

| Feature | Raw Spearman | Normalized Weight |
|---------|:------------:|:-----------------:|
| Weight | +0.131 | 0.307 |
| Speed score | +0.096 | 0.224 |
| Broad jump | +0.094 | 0.219 |
| Vertical | +0.058 | 0.134 |
| Height | +0.050 | 0.117 |
| 40 (negated) | -0.011 | 0.000 |

With the larger sample, weight dominates the supervised composite. Raw 40 time now has *negative* correlation and gets zero weight. Vertical, which was the top-weighted feature in the smaller sample, dropped substantially.

### 3d. Apples-to-Apples Comparison (same 114 resolved players)

Every method evaluated on the identical 114 resolved players with complete data on all 6 PCA features, with LOOCV where applicable:

| Method | Spearman | p-value | AUC | Needs CV? |
|--------|:--------:|:-------:|:---:|:---------:|
| Speed score | +0.096 | 0.310 | 0.579 | No (single metric) |
| RAS | +0.098 | 0.299 | 0.638 | No (fixed weights) |
| PCA PC1 (in-sample) | +0.125 | 0.184 | 0.628 | -- |
| **PCA PC1 (LOOCV)** | **+0.129** | **0.170** | **0.629** | **Stable** |
| Supervised (in-sample) | +0.170 | 0.071 | 0.684 | -- |
| Supervised (LOOCV) | +0.046 | 0.628 | 0.606 | Overfit |

Key observations:

- **PCA PC1 is stable under LOOCV** (AUC 0.628 vs 0.629). PCA finds variance directions in feature space, not outcome-correlated directions, so holding one player out barely changes the components. No overfitting concern.
- **Supervised composite collapses** (AUC 0.684 vs 0.606). Even with 114 players (up from 77), removing one shifts the correlation-derived weights enough to degrade predictions. Spearman drops from +0.170 to +0.046 (p=0.628).
- **RAS doesn't need LOOCV** because its weights are fixed by design (equal-weight percentiles against the full 779-player combine population). Nothing is fit to our outcome data.
- **RAS vs PCA PC1 is close.** RAS wins on AUC (0.638 vs 0.629), PCA wins on Spearman (+0.129 vs +0.098). Both are in the same noise band.
- **Nothing reaches significance.** The best honest p-value is PCA PC1 at 0.170.

---

## 4. Context: Athleticism vs College Production

For comparison, here's how athleticism metrics rank against the top college production features from the updated 5-layer feature evaluation (137 training players):

| Feature | Spearman | AUC | Source |
|---------|:--------:|:---:|--------|
| draft_capital | +0.568 | 0.846 | NFL consensus |
| career_grades_offense | +0.493 | 0.801 | College |
| peak2_grades_offense | +0.529 | 0.814 | College |
| best2_grades_offense | +0.447 | 0.757 | College |
| career_grades_run | +0.473 | 0.790 | College |
| ... | | | |
| **RAS** | **+0.098** | **0.638** | **Combine** |
| **PCA PC1 (LOOCV)** | **+0.129** | **0.629** | **Combine** |
| **Supervised (LOOCV)** | **+0.046** | **0.606** | **Combine** |
| Speed score | +0.096 | 0.579 | Combine |

All athleticism numbers are on the same 114 complete-case resolved players. The best honest athleticism composite (RAS, AUC 0.638) has less than half the Spearman correlation of the top college feature and an AUC gap of 16+ points. Draft capital alone (AUC 0.846) outperforms every athleticism metric by a wide margin.

---

## 5. Conclusions

1. **No individual combine metric significantly predicts RB dynasty outcomes.** Speed score is the only metric approaching significance (p=0.069). The 40-yard dash is near-useless (AUC 0.538).

2. **Composite approaches help modestly.** RAS (0.638 AUC) and PCA PC1 (0.629 AUC) both outperform individual drills by aggregating across multiple measurements, but neither reaches statistical significance.

3. **The supervised composite is overfit.** It reaches p=0.071 in-sample but collapses to p=0.628 under LOOCV, even with 114 observations. Correlation-weighted composites remain unreliable at this sample size.

4. **The signal weakened with more data.** Adding 80 resolved players (mostly late-round Busts who washed out) reduced RAS quartile hit rates from 43.5%/17.4% (top/bottom) to 26.3%/15.8%. Athleticism doesn't distinguish between different flavors of Bust, which is most of the sample.

5. **Athleticism is a distant second to college production.** The best athleticism metric has less than half the Spearman correlation and 16+ fewer AUC points than the top college feature. It may add marginal value as a supplementary feature, but it cannot anchor a model.

6. **Coverage limits usefulness.** Cone and shuttle are missing for >55% of players. Even the 40 is missing for ~29% of resolved players. Any feature built on combine data will have missing values for a significant fraction of the sample.

### Recommendation

RAS or speed score could be explored as a supplementary feature in the RB model, but expectations should be modest. The signal is real but small (Spearman ~0.10-0.13, AUC ~0.63), and coverage gaps will require imputation or a fallback mechanism. PCA offers no meaningful advantage over RAS. The supervised composite should not be used — its in-sample performance is an artifact of overfitting.

If athleticism enters the model, it should be as a single composite (likely RAS or speed score) rather than multiple raw metrics, to minimize feature bloat in an already small-sample problem. A longer-term approach — computing dynasty values for the full ~400+ resolved combine RBs going back to 2000 and learning athleticism weights from that larger sample — may yield a more reliable composite.

---

## Appendix: Methods

- **Data source**: NFL Combine via nflverse (`nfl_data_py.import_combine_data()`)
- **Matching**: Unicode-normalized names with suffix stripping (Jr., II, etc.), +/-1 year tolerance
- **RAS**: Percentile rank among all RB combine participants (2000-2026, n=779), averaged across available metrics, scaled 0-10
- **PCA**: StandardScaler -> PCA on 6 metrics (40, vertical, broad jump, speed score, weight, height). 40 negated so higher = better.
- **Supervised composite**: Each metric weighted by max(Spearman with tier, 0), normalized. Cross-validated via LOOCV.
- **Hit definition**: Elite or better (tier ordinal >= 3)
- **Resolved players**: 186 of 197 total. Resolution criteria: rookie contract window closed (draft_year + 4 <= 2025) or 2+ NFL seasons of data.
- **Tier thresholds**: Bust=0, Flex>0, Starter>=30, Elite>=75, Stud>=180, League-Winner>=350
