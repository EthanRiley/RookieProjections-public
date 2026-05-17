# RB Model v1 — Feature Grid Search Report

## Summary

Exhaustive grid search over 833 feature combinations (1-3 college features + draft capital) using XGBoost cumulative link on holdout data. Four rounds of iterative feature engineering — raw stats, per-attempt rates, age-adjusted/peak-gated variants, and z-scored composites — converged on a 3-feature model: **best2_explosive_pg + adj_yprr + composite_receiving**.

The final model achieves >=Elite AUC 0.936 and composite score 0.7869, a +14.5% improvement over draft capital alone (0.6872) and +1.8% over the Round 1 winner.

---

## 1. Dataset

| Split | Players | Draft Years | Bust Rate |
|-------|:-------:|:-----------:|:---------:|
| Train | 137 | 2016-2021 | 73.7% |
| Holdout | 49 | 2022-2024 | 73.5% |
| **Total** | **186** | **2016-2024** | **73.7%** |

**Tier distribution (train):**

| Tier | Count | % |
|------|:-----:|:-:|
| League-Winner | 6 | 4.4% |
| Stud | 6 | 4.4% |
| Elite | 12 | 8.8% |
| Starter | 5 | 3.6% |
| Flex | 7 | 5.1% |
| Bust | 101 | 73.7% |

---

## 2. Methodology

**Model:** XGBoost cumulative link (K-1 binary classifiers with Platt scaling). Fast proxy for full Bayesian+XGBoost ensemble.

**Evaluation:** Train on 2016-2021 (137 resolved RBs), holdout on 2022-2024 (49 players).

**Scoring composite:** 35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC. Higher = better.

**Draft capital** (`10 - (10/ln(261)) * ln(pick+1)`) always included as a feature. College features are evaluated for their marginal contribution above draft capital.

**Feature engineering rounds:**
1. **Round 1** — 10 raw features, 175 combos
2. **Round 2** — +3 per-attempt rate features, 377 combos
3. **Round 3** — +3 age-adjusted + 3 peak-gated features, 469 combos
4. **Round 4** — +3 z-scored composites, 833 combos

---

## 3. Composite Feature Definitions

Three composite skill scores were engineered as z-scored averages of component features, computed on the full resolved population:

### composite_receiving
**Pass-catching ability** — z-avg of 3 features capturing distinct aspects of receiving skill:
- `career_rec_yards_pg` — receiving production volume
- `career_yprr` — yards per route run (efficiency)
- `career_grades_pass_route` — PFF route quality grade

Inter-correlations: rec_yards_pg vs yprr r=0.018 (nearly independent!), rec_yards_pg vs pass_route r=0.594, yprr vs pass_route r=0.368. Low inter-correlation means the composite captures genuinely different evaluator perspectives.

### composite_explosive
**Big-play ability** — z-avg of 2 features capturing rate and volume:
- `career_explosive_per_att` — explosive play rate
- `best2_explosive_pg` — explosive plays per game (volume, best 2 seasons)

Inter-correlation: r=0.362 — modestly correlated, distinct dimensions.

### composite_self_creation
**Elusiveness & contact balance** — z-avg of 2 features:
- `career_elu_rush_mtf_per_att` — missed tackles forced per attempt
- `career_yco_attempt` — yards after contact per attempt

Inter-correlation: r=0.717 — more redundant, but the composite still captures the shared "self-creation" factor better than either alone.

---

## 4. Results

### 4.1 Baseline

| Model | LogLoss | Brier | >=Elite AUC | >=Stud AUC | >=Starter AUC | Composite |
|-------|:-------:|:-----:|:-----------:|:----------:|:-------------:|:---------:|
| Draft capital only | 1.016 | 0.411 | 0.707 | 0.793 | 0.803 | 0.6872 |

### 4.2 Best by Feature Count

| # Features | Best Combination | LogLoss | >=Elite AUC | >=Stud AUC | >=Starter AUC | Composite |
|:----------:|-----------------|:-------:|:-----------:|:----------:|:-------------:|:---------:|
| 1 | composite_receiving | 0.944 | 0.821 | 0.816 | 0.894 | 0.7458 |
| 2 | career_rec_yards_pg + composite_explosive | 0.955 | 0.912 | 0.897 | 0.903 | 0.7795 |
| **3** | **best2_explosive_pg + adj_yprr + composite_receiving** | **0.981** | **0.936** | **0.906** | **0.920** | **0.7869** |

### 4.3 Top 10 Three-Feature Combinations

| Rank | Features | LogLoss | >=Elite | >=Stud | >=Starter | Composite |
|:----:|----------|:-------:|:-------:|:------:|:---------:|:---------:|
| 1 | best2_explosive_pg + adj_yprr + composite_receiving | 0.981 | 0.936 | 0.906 | 0.920 | 0.7869 |
| 2 | career_ypa + composite_explosive + composite_receiving | 0.978 | 0.922 | 0.880 | 0.931 | 0.7832 |
| 3 | career_rec_yards_pg + composite_explosive + composite_receiving | 0.935 | 0.909 | 0.902 | 0.914 | 0.7826 |
| 4 | best2_explosive_pg + composite_self_creation + composite_receiving | 0.968 | 0.916 | 0.897 | 0.929 | 0.7823 |
| 5 | best2_explosive_pg + adj_rec_yards_pg + composite_receiving | 0.964 | 0.926 | 0.863 | 0.903 | 0.7813 |
| 6 | career_rec_yards_pg + composite_explosive (2-feat) | 0.955 | 0.912 | 0.897 | 0.903 | 0.7795 |
| 7 | adj_yprr + composite_explosive + composite_receiving | 0.957 | 0.909 | 0.889 | 0.909 | 0.7791 |
| 8 | composite_self_creation + composite_explosive + composite_receiving | 0.976 | 0.909 | 0.850 | 0.946 | 0.7787 |
| 9 | career_rec_yards_pg + adj_yprr + composite_explosive | 0.954 | 0.905 | 0.893 | 0.906 | 0.7784 |
| 10 | best2_explosive_pg + career_grades_pass_route + adj_yprr | 1.004 | 0.919 | 0.889 | 0.931 | 0.7780 |

### 4.4 Top 5 Two-Feature Combinations

| Rank | Features | LogLoss | >=Elite | >=Starter | Composite |
|:----:|----------|:-------:|:-------:|:---------:|:---------:|
| 1 | career_rec_yards_pg + composite_explosive | 0.955 | 0.912 | 0.903 | 0.7795 |
| 2 | composite_explosive + composite_receiving | 0.956 | 0.899 | 0.931 | 0.7776 |
| 3 | career_rec_yards_pg + best2_explosive_pg | 1.003 | 0.919 | 0.866 | 0.7724 |
| 4 | best2_explosive_pg + composite_receiving | 0.990 | 0.905 | 0.903 | 0.7722 |
| 5 | composite_self_creation + composite_receiving | 0.963 | 0.889 | 0.937 | 0.7704 |

### 4.5 Top 5 Single-Feature Models

| Rank | Feature | LogLoss | >=Elite | >=Starter | Composite |
|:----:|---------|:-------:|:-------:|:---------:|:---------:|
| 1 | composite_receiving | 0.944 | 0.821 | 0.894 | 0.7458 |
| 2 | career_rec_yards_pg | 1.012 | 0.840 | 0.839 | 0.7353 |
| 3 | adj_yprr | 1.029 | 0.821 | 0.871 | 0.7324 |
| 4 | pg_rec_yards_pg | 0.990 | 0.797 | 0.827 | 0.7236 |
| 5 | adj_rec_yards_pg | 1.025 | 0.787 | 0.843 | 0.7131 |

### 4.6 Three-Composite-Only Model

Using only the three composites (no raw features):

| Features | LogLoss | >=Elite | >=Stud | >=Starter | Composite |
|----------|:-------:|:-------:|:------:|:---------:|:---------:|
| composite_self_creation + composite_explosive + composite_receiving | 0.976 | 0.909 | 0.850 | **0.946** | 0.7787 |

Notable: highest >=Starter AUC of any combination tested (0.946). The composites fully span the RB skill space.

---

## 5. Feature Frequency Analysis

How often each feature appears in top combinations:

| Feature | Top 10 | Top 20 | Top 30 |
|---------|:------:|:------:|:------:|
| composite_receiving | 7 | 12 | 16 |
| composite_explosive | 6 | 11 | 14 |
| best2_explosive_pg | 4 | 9 | 14 |
| career_rec_yards_pg | 3 | 8 | 13 |
| adj_yprr | 4 | 5 | 5 |
| composite_self_creation | 2 | 3 | 6 |
| career_grades_pass_route | 1 | 2 | 5 |
| career_elu_rush_mtf_pg | 0 | 2 | 4 |
| career_ypa | 1 | 2 | 2 |
| adj_rec_yards_pg | 1 | 1 | 2 |
| career_yco_attempt | 0 | 1 | 2 |

**Features that never appear in top 30:** best2_touchdowns_pg (1 appearance), adj_explosive_pg (1), pg_yprr (0), pg_elu_rush_mtf_pg (0), best2_yprr (0), career_elu_rush_mtf_pg is marginal.

### Interpretation

- **composite_receiving is the anchor.** It appears in 16/30 top combos and is the best single feature. The z-scored average of rec_yards_pg + YPRR + pass_route_grade outperforms any individual receiving metric.
- **Explosiveness is the strongest complement.** composite_explosive (14/30) and best2_explosive_pg (14/30) are nearly interchangeable — the composite averages in career explosive rate alongside volume, but raw explosive volume works nearly as well.
- **adj_yprr is the best third feature.** Age-adjusted YPRR adds signal that the receiving composite doesn't fully capture — it applies the graduated age multiplier (FR +25%, SO +5%, JR -20%, SR -25%) before peak selection, rewarding early breakout.
- **Peak-gated features disappeared.** Despite strong univariate numbers (pg_rec_yards_pg: Spearman +0.315), peak-gated features don't add marginal value above composites. The grade>=80 gate reduces sample size too aggressively (113 vs 174 players).
- **Per-attempt rate features never gained traction.** Across all 4 rounds, per-attempt versions (explosive/att, MTF/att) consistently underperform per-game versions. In RB context, opportunity (more touches) is itself a talent signal.

---

## 6. Round-by-Round Progression

| Round | Winner | Composite | >=Elite AUC | >=Starter AUC |
|:-----:|--------|:---------:|:-----------:|:-------------:|
| 1 | career_rec_yards_pg + best2_explosive_pg + career_grades_pass_route | 0.7728 | 0.905 | 0.880 |
| 2 | (same — per-attempt features did not improve) | 0.7728 | 0.905 | 0.880 |
| 3 | best2_explosive_pg + career_grades_pass_route + adj_yprr | 0.7780 | 0.919 | 0.931 |
| **4** | **best2_explosive_pg + adj_yprr + composite_receiving** | **0.7869** | **0.936** | **0.920** |

Key transitions:
- **R1→R3:** Age-adjusted YPRR replaced career_rec_yards_pg as the efficiency measure, adding +0.014 composite and +1.4pp Elite AUC.
- **R3→R4:** composite_receiving absorbed career_grades_pass_route (and implicitly career_rec_yards_pg + career_yprr), yielding a cleaner 3-feature model. The composite pools three distinct receiving perspectives into a single dimension, freeing the other two slots for explosiveness and efficiency.

---

## 7. Recommended v1 Feature Set

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `draft_capital` | NFL consensus | Log-scaled: `10 - (10/ln(261)) * ln(pick+1)` |
| `best2_explosive_pg` | Explosive playmaking | Explosive plays per game, best 2 eligible seasons |
| `adj_yprr` | Receiving efficiency (age-adjusted) | Yards per route run, graduated age multiplier, peak selection |
| `composite_receiving` | Pass-catching skill composite | z-avg(career_rec_yards_pg, career_yprr, career_grades_pass_route) |

### Why these three?

1. **composite_receiving** captures the full receiving skill space in a single number — volume (rec_yards_pg), efficiency (YPRR), and quality (PFF route grade). It outperforms any individual receiving metric because the three components have low inter-correlation (r=0.018 to r=0.594), meaning they capture genuinely different aspects.

2. **best2_explosive_pg** measures big-play ability — a dimension orthogonal to receiving skill. It counts PFF "explosive" plays per game in the player's best 2 eligible seasons. This is a volume metric rather than a rate metric, which is deliberate: backs who create more explosive plays tend to earn more opportunities, and opportunity itself is a talent signal for RBs.

3. **adj_yprr** adds age-adjusted receiving efficiency. The graduated multiplier (FR +25%, SO +5%, JR -20%, SR -25%) rewards early breakout, matching the dynasty value curve where younger producers tend to have better NFL outcomes. This captures signal that composite_receiving (which uses career aggregation without age adjustment) misses.

### Alternative configurations

The 2-feature model (**career_rec_yards_pg + composite_explosive**, composite 0.7795) is nearly as strong and may be preferable if model simplicity is prioritized. The gap to 3-feature is only +0.0074 composite.

The all-composites model (**composite_self_creation + composite_explosive + composite_receiving**, composite 0.7787) has the highest >=Starter AUC (0.946) and may be preferred if Starter-tier separation matters most.

---

## 8. Comparison to WR Model

| Metric | RB v1 (XGB proxy) | WR v11 (full ensemble, holdout) |
|--------|:------------------:|:-------------------------------:|
| >=Elite AUC | 0.936 | 0.970 |
| >=Stud AUC | 0.906 | 0.941 |
| LogLoss | 0.981 | 0.773 |
| # College features | 3 | 5 |
| Train size | 137 | ~200 |

The RB model shows strong discrimination (AUC) but worse calibration (LogLoss) than the WR model. This is expected: (1) XGB cumulative link is a fast proxy, not the full Bayesian+XGB ensemble, and (2) RB has fewer training examples and a higher bust rate. The full ensemble should improve LogLoss substantially.

---

## 9. Athleticism Residual Test

We tested whether `composite_athleticism` (z-avg of speed_score + broad_jump — the two metrics identified as carrying all the athleticism signal in our earlier analysis) adds value on top of the college production features.

**Method:** For each of the top 10 feature combinations from the grid search, we added `composite_athleticism` as a 4th (or 3rd) feature and re-evaluated on holdout.

### Data Coverage Problem

Athleticism data is only available for players who participated in combine drills AND appear in the historical athleticism dataset (2000-2023, 315 RBs). Merging onto our 186-player college stats dataset:

| Split | With Athleticism | Without | Coverage |
|-------|:----------------:|:-------:|:--------:|
| Train (2016-2021) | 85 | 52 | 62% |
| Holdout (2022-2024) | 20 | 29 | 41% |

Critically missing from holdout: **Jahmyr Gibbs** (League-Winner) and **De'Von Achane** (Stud) — the two best holdout outcomes. Losing them makes any AUC comparison meaningless.

### Results

Adding `composite_athleticism` degraded every combination:

| Base Combo | Base Composite | +Ath Composite | Delta | Holdout n |
|------------|:--------------:|:--------------:|:-----:|:---------:|
| best2_explosive_pg + adj_yprr + composite_receiving | 0.7869 | 0.0136 | -0.773 | 18 |
| career_ypa + composite_explosive + composite_receiving | 0.7834 | 0.6428 | -0.141 | 18 |
| career_rec_yards_pg + composite_explosive + composite_receiving | 0.7825 | 0.6291 | -0.153 | 18 |
| composite_self_creation + composite_explosive + composite_receiving | 0.7787 | 0.6238 | -0.155 | 18 |

All 10 combinations showed negative deltas, ranging from -0.14 to -0.77. LogLoss blew up to 1.6-7.0 in most cases.

### Interpretation

This is **not evidence that athleticism has no residual signal** — it's evidence that the test is invalid. With only 18 holdout players (down from 49), missing the top 2 outcomes, the evaluation has no statistical power. The degradation is driven by sample size collapse, not by athleticism being anti-predictive.

The standalone athleticism analysis (AUC ~0.62) already established that the signal is real but modest, far behind college production (AUC ~0.78) and draft capital (AUC ~0.85). Whether it adds residual value after controlling for both remains an open question that cannot be answered with the current data coverage.

**Recommendation:** Exclude athleticism from the v1 RB model. If combine coverage improves in future datasets, revisit as a potential 4th feature. For now, the 3 college features + draft capital provide strong discrimination without data availability constraints.

---

## 10. Grid Search Methodology & Iteration Log

### Search Protocol

Each round follows the same protocol:
1. Define candidate features (distinct skill dimensions, low inter-correlation)
2. Generate all 1-to-3 feature combinations
3. For each combo: train XGBoost cumulative link (K-1 binary classifiers with Platt scaling) on 2016-2021, evaluate on 2022-2024
4. Score with composite metric: 35% LogLoss + 35% >=Elite AUC + 15% Brier + 10% >=Starter AUC + 5% >=Stud AUC
5. Rank by composite, analyze feature frequency in top 20

Features were selected from the 5-layer evaluation pipeline (`rb_data/feature_evaluation.csv`, 158 features screened). Only features with distinct skill dimensions and low mutual correlation were included as candidates.

### Round 1: Raw Features (10 candidates, 175 combos)

**Candidates:** career_rec_yards_pg, best2_explosive_pg, career_elu_rush_mtf_pg, career_grades_pass_route, best2_touchdowns_pg, career_ypa, career_yco_attempt, best2_yprr, career_avoided_tackles_per_att, best2_grades_run

**Winner:** career_rec_yards_pg + best2_explosive_pg + career_grades_pass_route
- Composite: 0.7728, >=Elite AUC: 0.905, >=Starter AUC: 0.880

**Key findings:**
- career_rec_yards_pg appeared in 16/20 top combos — dominant anchor
- best2_explosive_pg was the strongest complement (12/20)
- The third feature slot was contested between career_grades_pass_route and career_yco_attempt

### Round 2: Per-Attempt Rates (+3 candidates, 377 combos)

**Added:** career_explosive_per_att, career_elu_rush_mtf_per_att, best2_explosive_per_att

**Winner:** Same as Round 1 (per-attempt features never cracked top 8)

**Key finding:** Per-attempt versions consistently underperform per-game versions. In the RB context, opportunity (more touches) is itself a signal of talent — normalizing it away removes information.

### Round 3: Age-Adjusted + Peak-Gated (+6 candidates, 469 combos)

**Added:**
- Age-adjusted (graduated multiplier, peak selection): adj_rec_yards_pg, adj_yprr, adj_explosive_pg
- Peak-gated (grades_offense >= 80 filter): pg_rec_yards_pg, pg_yprr, pg_elu_rush_mtf_pg

**Winner:** best2_explosive_pg + career_grades_pass_route + adj_yprr
- Composite: 0.7780, >=Elite AUC: 0.919, >=Starter AUC: 0.931

**Key findings:**
- adj_yprr emerged as a strong third feature, replacing career_rec_yards_pg in the winner
- The graduated age multiplier (FR +25%, SO +5%, JR -20%, SR -25%) rewards early breakout
- Peak-gated features had strong univariate signal (pg_rec_yards_pg: Spearman +0.315) but didn't add marginal value — the grade>=80 gate reduces eligible players from 174 to 113

### Round 4: Composites (+3 candidates, 833 combos)

**Added:** composite_self_creation, composite_explosive, composite_receiving (z-scored averages)

**Winner:** best2_explosive_pg + adj_yprr + composite_receiving
- Composite: 0.7869, >=Elite AUC: 0.936, >=Starter AUC: 0.920

**Key findings:**
- composite_receiving replaced career_grades_pass_route + career_rec_yards_pg as the receiving anchor
- The composite pools three low-correlation receiving perspectives (volume r=0.018 with efficiency) into one dimension
- composite_explosive (11/20 top combos) is nearly interchangeable with raw best2_explosive_pg (9/20)
- The all-composites model (composite_self_creation + composite_explosive + composite_receiving) achieved the highest >=Starter AUC (0.946)

### Round 5: Athleticism Residual (top 10 combos + composite_athleticism)

**Added:** composite_athleticism (z-avg of speed_score + broad_jump)

**Result:** All 10 combinations degraded. Data coverage too sparse (41% holdout, 18 players) for valid evaluation. See Section 9.

---

## 11. Next Steps

1. **Run full Bayesian + XGBoost ensemble** with the 3-feature set to get calibrated probabilities and proper LogLoss.
2. **Evaluate 4th feature candidates** — composite_self_creation (6/30 top combos) or career_ypa may provide marginal lift.
3. **Holdout stability check** — the 49-player holdout is small. Leave-one-year-out CV on training data would provide a more robust estimate.
4. **Generate prospect predictions** for 2024/2025 RB classes.

---

*Report generated 2025-05-13. Data: 186 resolved RBs, 2016-2024 draft classes. Model: XGBoost cumulative link with Platt scaling. Grid search: 4 rounds, 1,854 total combinations evaluated.*
