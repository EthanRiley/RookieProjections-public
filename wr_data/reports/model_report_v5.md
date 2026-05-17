# WR Dynasty Tier Model Report (v5)

## Model Overview

Ordinal classification model predicting dynasty fantasy football outcomes for rookie WRs. Outputs a full probability distribution across 6 tiers (Bust / Flex / Starter / Elite / Stud / League-Winner) based on college production analytics and NFL draft capital.

**Ensemble**: 75% Bayesian ordinal regression + 25% XGBoost cumulative link.

**Training data**: 2018-2021 draft classes (122 players). Holdout: 2022-2024 (88 players).

**Prospect predictions**: 2024, 2025, and 2026 classes (retrained on 2018-2023, 176 players).

---

## Feature Set (v5)

7 features, each measuring a distinct dimension of prospect evaluation:

| # | Feature | Dimension | Description |
|---|---------|-----------|-------------|
| 1 | `draft_capital` | Market price | Sqrt-scaled pick score. 32 front offices' consensus. |
| 2 | `breakout_age` | Production timing | Age at first season with 2.0+ YPRR and 200+ routes |
| 3 | `breakout_yprr` | Breakout efficiency | Yards per route run at the breakout season |
| 4 | `breakout_yptpa` | Breakout market share | Yards per team pass attempt at the breakout season |
| 5 | `peak_contested_catch_rate` | Contested catching | Highest single-season CCR (min 3 targets, floored by career) |
| 6 | `career_targeted_qb_rating` | QB trust | Career game-weighted passer rating when targeted |
| 7 | `peak2_avoided_tackles_per_rec` | Elusiveness | Avoided tackles per reception, top 2 seasons |

### Key change from v4: peak stat correction

`best_contested_catch_rate` and `best2_avoided_tackles_per_rec` were discovered to be buggy -- they selected the season with the highest `grades_offense` and reported that season's stats, rather than selecting by the stat itself. 35% of players had a "best" CCR lower than their career rate (e.g., Kadarius Toney: "best" 42.9 vs career 66.8).

Replaced with true peak stats:
- **`peak_contested_catch_rate`**: Max single-season CCR among seasons with 3+ contested targets, floored by career game-weighted average. The floor handles cases where low-volume high-rate seasons contribute to career but don't meet the 3-target threshold.
- **`peak2_avoided_tackles_per_rec`**: Top 2 seasons by per-reception rate (min 10 receptions/season), weighted by reception volume.

---

## Holdout Results (2022-2024)

### Aggregate Metrics

| Model | Log Loss | Brier Score |
|-------|----------|-------------|
| Bayesian Full | 0.8153 | 0.3571 |
| XGBoost Full | 1.6358 | 0.3497 |
| **Ensemble Full** | **0.8026** | **0.3503** |
| Bayesian College | 0.8331 | 0.3694 |
| XGBoost College | 2.7585 | 0.3965 |
| Ensemble College | 0.8523 | 0.3735 |

### Per-Threshold AUC (Ensemble Full)

| Threshold | AUC | Brier | Positive Rate |
|-----------|-----|-------|---------------|
| >=Flex | 0.876 | 0.1330 | 25.0% |
| >=Starter | 0.882 | 0.1005 | 15.9% |
| >=Elite | **0.961** | 0.0759 | 12.5% |
| >=Stud | **0.969** | 0.0298 | 3.4% |
| >=LW | 0.943 | 0.0120 | 1.1% |

The model is strongest at separating Elite+ from the rest (AUC 0.961) and identifying Stud+ upside (AUC 0.969).

### Ranking Quality

| Metric | Value |
|--------|-------|
| Spearman(E[full], tier outcome) | **0.578** |
| Spearman(E[college], tier outcome) | 0.505 |
| Elite+ captured in top 20 | **10 of 11** (91%) |
| Busts correctly in bottom half | 41 of 44 (93%) |

### Calibration by Quartile

| Quartile | Predicted P(Bust) | Actual Bust % | Predicted P(Elite+) | Actual Elite+ % |
|----------|-------------------|---------------|---------------------|-----------------|
| Q1 (top 22) | 53.2% | 31.8% | 35.7% | **45.5%** |
| Q2 | 71.0% | 81.8% | 19.1% | 4.5% |
| Q3 | 80.4% | 90.9% | 12.6% | 0.0% |
| Q4 (bottom 22) | 89.0% | 95.5% | 7.3% | 0.0% |

The model is slightly conservative in Q1 -- it overpredicts bust probability for top prospects. In reality, top-quartile prospects hit Elite+ at 45.5% vs the model's 35.7% predicted rate. Q2-Q4 calibration is tighter.

---

## Holdout: Hits and Misses

### Top 10 Predictions vs Actuals

| Rank | Player | Pick | E[full] | Actual Tier |
|------|--------|------|---------|-------------|
| 1 | Chris Olave | 11 | 2.076 | **Stud** |
| 2 | Garrett Wilson | 10 | 2.067 | **Elite** |
| 3 | Marvin Harrison Jr. | 4 | 1.967 | Flex* |
| 4 | Jaxon Smith-Njigba | 20 | 1.834 | **League-Winner** |
| 5 | Jordan Addison | 23 | 1.680 | **Elite** |
| 6 | Jameson Williams | 12 | 1.668 | **Elite** |
| 7 | Marvin Mims | 63 | 1.621 | Bust |
| 8 | Ladd McConkey | 34 | 1.594 | **Elite** |
| 9 | Malik Nabers | 6 | 1.562 | **Elite** |
| 10 | Treylon Burks | 18 | 1.520 | Bust |

**7 of the top 10 are Elite or better.** MHJ is classified as Flex with only 1 NFL season completed -- this is likely to improve. The two misses (Mims, Burks) are both early-to-mid 1st/2nd round picks with strong college profiles who haven't produced in the NFL.

*MHJ has only played 1 of 4 rookie contract seasons. His current "Flex" tier is based on incomplete data.

### Biggest Miss

**Zay Flowers** (Pick 22, actual Elite) ranked #28 with E=0.926. The model was skeptical: late breakout age (22.62), low peak CCR relative to his draft position. Despite drafting late in the 1st round, he produced immediately -- the model's college profile didn't flag him as a likely hit.

### Notable Busts Ranked High

- **Treylon Burks** (#10, E=1.520): Pick 18, strong college profile, but NFL opportunity collapsed.
- **Marvin Mims** (#7, E=1.621): Pick 63 but elite college analytics. Hasn't received opportunity.
- **Rome Odunze** (#15, E=1.266): Pick 9 with poor college profile (biggest negative edge: -0.487). Draft capital alone pushed him up. 1 NFL season.
- **Keon Coleman** (#14, E=1.339): Pick 33, decent profile. 1 NFL season on a bad offense.

Several of these "busts" have only 1 NFL season and may yet improve. The model's 4-season evaluation window means early busts are provisional.

---

## Draft Capital Edge Analysis

The "edge" column measures E[full] - E[college] -- how much draft capital shifts the prediction vs college analytics alone.

| Stat | Value |
|------|-------|
| Mean edge | -0.081 |
| Spearman(edge, tier outcome) | **-0.526** |

The negative Spearman is striking: **the more draft capital boosts a player's ranking above their college profile, the worse they tend to do.** This suggests the market (NFL teams) may systematically overvalue certain prospect archetypes relative to what college analytics predict.

### Biggest Positive Edges (college profile >> draft slot)

| Player | Pick | Edge | Actual |
|--------|------|------|--------|
| Ronnie Bell | 253 | +0.215 | Bust |
| Kayshon Boutte | 187 | +0.173 | Bust |
| Samori Toure | 258 | +0.139 | Bust |

Late-round picks with college profiles that suggest more talent than their draft slot implies. In practice, these players couldn't overcome the opportunity deficit of being late picks.

### Biggest Negative Edges (draft slot >> college profile)

| Player | Pick | Edge | Actual |
|--------|------|------|--------|
| Rome Odunze | 9 | -0.487 | Bust (1 yr) |
| Marvin Harrison Jr. | 4 | -0.469 | Flex (1 yr) |
| Drake London | 8 | -0.469 | **Elite** |
| Chris Olave | 11 | -0.447 | **Stud** |
| Malik Nabers | 6 | -0.443 | **Elite** |

Top-10 picks whose college profiles didn't match their draft slot. The results are mixed: Olave, London, Nabers hit despite mediocre college analytics, suggesting draft capital captures information (medicals, interviews, tape) invisible to our features. But Odunze and MHJ (so far) haven't produced, suggesting the market can also overpay.

---

## 2024-2026 Prospect Rankings

Models retrained on all labeled data (2018-2023, 176 players) for forward predictions.

### 2024 Class (34 prospects)

| Rank | Player | Pick | P(Bust) | P(Elite+) | E[full] | Early Returns |
|------|--------|------|---------|-----------|---------|---------------|
| 1 | Marvin Harrison Jr. | 4 | 33.2% | 50.4% | 2.122 | Flex (1 yr) |
| 2 | Ladd McConkey | 34 | 36.4% | 46.9% | 1.927 | **Elite** (1 yr) |
| 3 | Malik Nabers | 6 | 42.5% | 42.5% | 1.875 | **Elite** (1 yr) |
| 4 | Brian Thomas | 23 | 51.2% | 34.0% | 1.430 | **Elite** (1 yr) |
| 5 | Keon Coleman | 33 | 49.5% | 33.7% | 1.425 | Bust (1 yr) |

**Early validation**: McConkey (#2), Nabers (#3), and Brian Thomas (#4) all hit Elite in year 1. McConkey at pick 34 was a strong model favorite due to his college profile -- the model correctly identified him as a value pick.

### 2025 Class (30 prospects)

| Rank | Player | Pick | P(Bust) | P(Elite+) | E[full] |
|------|--------|------|---------|-----------|---------|
| 1 | Travis Hunter | 2 | 33.2% | 50.6% | 2.093 |
| 2 | Emeka Egbuka | 19 | 44.0% | 39.7% | 1.645 |
| 3 | Tre Harris | 55 | 49.7% | 33.6% | 1.413 |
| 4 | Tetairoa McMillan | 8 | 49.9% | 32.3% | 1.374 |
| 5 | Matthew Golden | 23 | 51.0% | 30.6% | 1.318 |

Travis Hunter is the clear WR1 with a profile similar to the top picks in prior classes. Tre Harris at pick 55 is a notable value flag -- his college profile significantly outperforms his draft slot (edge = -0.210, moderate), suggesting the model sees more talent than the market priced in.

### 2026 Class (34 prospects)

| Rank | Player | Pick | P(Bust) | P(Elite+) | E[full] |
|------|--------|------|---------|-----------|---------|
| 1 | Carnell Tate | 4 | 34.5% | 49.2% | 2.053 |
| 2 | Makai Lemon | 20 | 42.4% | 41.2% | 1.735 |
| 3 | Omar Cooper Jr. | 30 | 45.1% | 39.6% | 1.684 |
| 4 | Jordyn Tyson | 8 | 54.4% | 30.3% | 1.276 |
| 5 | Caleb Douglas | 75 | 53.6% | 27.4% | 1.240 |

Carnell Tate has an elite combined profile (pick 4, early breakout, strong analytics). Omar Cooper Jr. at pick 30 is a value target -- the model's 3rd-ranked WR despite being a late 1st rounder, driven by strong college production metrics.

---

## Model Architecture

### Ensemble Components

| Component | Weight | Strengths |
|-----------|--------|-----------|
| Bayesian ordinal (PyMC) | 75% | Calibrated uncertainty, handles small-n, draft capital as informed prior |
| XGBoost cumulative link | 25% | Captures non-linearities and interactions |
| Elastic Net ordinal | 0% (sanity check) | Interpretable baseline, unstable on this data |

Bayesian dominates on log loss (0.8153 vs 1.6358) due to better-calibrated probabilities. XGBoost has a slight Brier score edge (0.3497 vs 0.3571), suggesting sharper point predictions. The 75/25 blend captures both strengths.

### Cross-Validation

Leave-one-year-out on 2018-2021 (training). Each year is held out in turn, and the model is trained on the remaining 3 years. This prevents information leakage across draft classes.

### Imputation

| Feature | Missing means | Imputed value |
|---------|---------------|---------------|
| breakout_age | Never broke out | max observed + 1 |
| breakout_yptpa | Never broke out | 0 |
| breakout_yprr | Never broke out | 0 |
| peak_contested_catch_rate | No qualifying seasons | NaN (player dropped) |

---

## Limitations and Open Questions

1. **Small holdout with incomplete careers.** The 2024 class has only 1 NFL season. Several "Bust" labels (MHJ, Keon Coleman, Rome Odunze) are provisional. The model's true accuracy on 2024 won't be known until 2027.

2. **Conservative top-quartile calibration.** The model predicts 35.7% Elite+ rate for top prospects; actual is 45.5%. This means the model is a useful ranking tool but underestimates upside for top prospects.

3. **Opportunity not modeled.** Landing spot, depth chart, and scheme fit are not features. Players like Marvin Mims (elite profile, buried on depth chart) are systematically misranked. This is the largest source of unexplained variance.

4. **Edge paradox.** The negative Spearman between edge and outcome (-0.526) suggests draft capital may introduce more noise than signal for prospects whose college profile doesn't match their slot. The college-only model may be preferable for evaluating "overdrafted" players.

5. **Era sensitivity.** `career_targeted_qb_rating` showed era instability (drift=0.087) in earlier validation. Offensive scheme evolution could degrade this feature's predictive power over time.

6. **Single position.** Only WR is modeled. RB, TE, and QB pipelines not started.

---

## File Index

| File | Description |
|------|-------------|
| `wr_data/holdout_predictions_v2.csv` | Ensemble predictions on 2022-2024 holdout |
| `wr_data/prospect_predictions_2024.csv` | 2024 class predictions (retrained on 2018-2023) |
| `wr_data/prospect_predictions_2025.csv` | 2025 class predictions |
| `wr_data/prospect_predictions_2026.csv` | 2026 class predictions |
| `wr_data/bayesian_full_holdout_predictions.csv` | Bayesian component holdout predictions |
| `wr_data/xgb_full_holdout_predictions.csv` | XGBoost component holdout predictions |
| `wr_data/enet_full_holdout_predictions.csv` | Elastic Net component holdout predictions |
| `wr_data/feature_selection_report.md` | v4 feature selection report (slot analysis) |
| `wr_data/wr_dynasty_value_with_college.csv` | Master dataset with all features |
| `modeling/evaluate_holdout.py` | Holdout evaluation script |
| `modeling/predict_prospects.py` | Prospect prediction script |
| `aggregation/aggregate_college_stats.py` | Feature engineering pipeline |
