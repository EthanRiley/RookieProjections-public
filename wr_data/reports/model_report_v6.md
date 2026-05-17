# WR Dynasty Model v6 Report

## What Changed From v5

v6 is a ground-up simplification driven by a comprehensive 42-feature, 7-dimension analysis. The model drops from 7 features to 5, replaces breakout age with best-2-season YPRR, and adds a 200-route minimum filter to prevent small-sample noise.

### v5 Feature Set (7 features)
- draft_capital
- career_targeted_qb_rating
- breakout_age (YPRR-based)
- peak_contested_catch_rate
- peak2_avoided_tackles_per_rec
- breakout_yptpa
- breakout_yprr

### v6 Feature Set (5 features)
- **draft_capital** -- NFL consensus (sqrt-scaled pick score)
- **best2_yprr** -- best 2 seasons YPRR (min 200 routes per season)
- **career_targeted_qb_rating** -- game-weighted career targeted passer rating
- **best2_contested_catch_rate** -- best 2 seasons contested catch rate
- **best2_avoided_tackles_per_rec** -- best 2 seasons missed tackles forced per reception

### Why These Changes

**Replaced breakout_age with best2_yprr.** The full dimension analysis showed breakout age has -0.067 residual signal after draft_capital + best2_yprr, with only 24.7% bootstrap positive. Its information is absorbed by best2_yprr (r=-0.497) and career_tqbr. Swapping breakout_age for best2_yprr in the core improved LOO-AUC from 0.864 to 0.882, winning 81% of 500 bootstrap iterations.

**Dropped breakout_yptpa and breakout_yprr.** Both magnitude features had deeply negative residual signal: breakout_yptpa at -0.164 (1.5% bootstrap positive), breakout_yprr at -0.116 (4.8%). The "15.6% signal improvement" from the breakout age engineering report was based on a flawed metric (sum of absolute residuals treats redundancy as signal). LOO-AUC confirmed both hurt prediction in every configuration.

**Switched peak_contested_catch_rate to best2.** best2_contested_catch_rate had the strongest residual of any feature in any dimension (+0.137, 97.9% bootstrap positive against a 2-feature base). peak_ccr was weaker (+0.084, 88.3%).

**Switched peak2_avoided_tackles_per_rec to best2.** best2_avoided_tackles_per_rec showed +0.066 residual and 84.2% bootstrap positive against the 4-feature core -- genuine orthogonal signal capturing elusiveness after the catch. It has the lowest collinearity with any base feature (max r=0.222) and the best era stability of any elusiveness metric (drift=0.051). It measures a mechanistically distinct skill dimension: what a receiver does after the catch. Draft capital, YPRR, QB trust, and contested catch ability don't capture this.

**Added 200-route minimum filter on best2 season selection.** Without this filter, players with tiny-sample seasons (e.g., Arian Smith: 5 routes, 86 yards = 17.2 YPRR) produced absurdly inflated best2_yprr. This was discovered when Arian Smith (2025, pick 110) appeared as WR2 in the initial model run. The filter ensures only seasons with meaningful volume contribute to best2 metrics.

---

## Model Architecture

Unchanged from v5:
- **Ensemble**: 75% Bayesian ordinal regression + 25% XGBoost cumulative link
- **Variants**: Full model (draft_capital + college features) and College-only (college features only)
- **Training**: 2018-2021 for holdout evaluation; all labeled data for prospect predictions
- **Cross-validation**: Leave-one-year-out
- **Calibration**: Platt scaling on XGBoost; Bayesian model is self-calibrating

---

## Holdout Results (2022-2024, 89 players)

### Ensemble Metrics

| Model | LogLoss | Brier |
|-------|---------|-------|
| Bayesian Full | 0.835 | 0.368 |
| XGBoost Full | 1.184 | 0.368 |
| **Ensemble Full** | **0.813** | **0.362** |
| Ensemble College | 0.883 | 0.394 |

### Per-Threshold AUC (Ensemble Full)

| Threshold | AUC | Brier |
|-----------|-----|-------|
| >=Flex | 0.868 | 0.141 |
| >=Starter | 0.889 | 0.106 |
| **>=Elite** | **0.955** | 0.080 |
| >=Stud | 0.865 | 0.040 |
| >=LW | 0.948 | 0.020 |

### v5 vs v6 Comparison

| Metric | v5 (7 features) | v6 (5 features) |
|--------|-----------------|-----------------|
| Ensemble LogLoss | 0.803 | 0.813 |
| Ensemble Brier | 0.350 | 0.362 |
| >=Elite AUC | 0.961 | 0.955 |

Holdout metrics are slightly worse on paper, but with 89 players these differences are noise. The structural advantage -- 5 features instead of 7, each measuring a distinct skill dimension, less overfitting risk -- is the real win.

---

## Feature Dimensions

Each feature captures a distinct, non-redundant skill:

| Feature | Dimension | Evidence |
|---------|-----------|----------|
| draft_capital | NFL talent consensus | Spearman 0.529, AUC 0.865. Dominant predictor. |
| best2_yprr | Peak route efficiency | Replaces breakout_age. LOO-AUC 0.882 vs 0.864 with ba. 81% bootstrap win rate. |
| career_targeted_qb_rating | QB trust / route quality | +0.041 residual, 74% bootstrap. Only feature with both LOO-AUC lift and positive residual vs 2-feature base. |
| best2_contested_catch_rate | Contested ball skills | +0.137 residual, 97.9% bootstrap. Strongest orthogonal signal of any feature tested. |
| best2_avoided_tackles_per_rec | Elusiveness after the catch | +0.066 residual, 84.2% bootstrap vs 4-feature core. Max collinearity 0.222 with any base feature. Era drift 0.051. |

### What Was Tested and Rejected

42 candidates across 7 dimensions were tested against draft_capital + breakout_age:

- **YPRR variants** (career, best2, peak, breakout): All negative residual. breakout_age is defined by YPRR, so adding YPRR is redundant. However, *replacing* breakout_age with best2_yprr was an improvement.
- **YPTPA / market share**: No genuine orthogonal signal. LOO-AUC improvements without residual support = overfitting.
- **Catch reliability** (catch_pct_adot_adj, caught_percent): 0.765 correlated with career_tqbr. Fully absorbed once tqbr is in the model.
- **Production volume** (targets_pg, receptions_pg, first_downs_per_route): All negative residual. No feature should be in the model.
- **PFF grades** (grades_pass_route, grades_offense): Weaker than targeted QBR on every metric.
- **Breakout magnitudes** (breakout_yptpa, breakout_yprr): The breakout age engineering report's "15.6% improvement" was an artifact of a flawed metric. LOO-AUC testing showed both hurt prediction.

---

## Holdout Highlights

### Hits
- **Jaxon Smith-Njigba** (League-Winner): E[full] = 2.05. Model's WR2 in 2023.
- **Garrett Wilson** (Elite): E[full] = 2.06. Model's WR1 in 2022.
- **Chris Olave** (Stud): E[full] = 1.93. High draft capital + strong college profile.
- **Ladd McConkey** (Elite): E[full] = 1.56 despite pick 34. College stats carried him.
- **Jordan Addison** (Elite): E[full] = 1.48. Correctly flagged.
- **Drake London** (Elite): E[full] = 1.44.

### Misses
- **Treylon Burks** (Bust): E[full] = 1.67. Pick 18 + good profile, but busted. Model can't predict injuries/situation.
- **Puka Nacua** (League-Winner): E[full] = 1.43, but college-only E = 1.86 and edge = +0.43. The college profile screamed talent; pick 177 dragged the full model down. This is exactly the kind of signal the edge column is designed to flag.
- **George Pickens** (Stud): E[full] = 0.91. Pick 52 + modest college volume. Model underrated him.
- **Marvin Harrison Jr.** (Flex after 1 season): E[full] = 1.99. Model loved him. Only 1 season of NFL data.

---

## 2024 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Marvin Harrison Jr. | 4 | 2.25 | 54.6% | -0.55 |
| Ladd McConkey | 34 | 1.77 | 41.3% | -0.33 |
| Malik Nabers | 6 | 1.76 | 41.2% | -0.62 |
| Brian Thomas | 23 | 1.58 | 36.3% | -0.47 |
| Rome Odunze | 9 | 1.31 | 29.7% | -0.60 |

---

## 2025 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Travis Hunter | 2 | 2.57 | 58.7% | -0.75 |
| Emeka Egbuka | 19 | 1.51 | 35.1% | -0.54 |
| Tetairoa McMillan | 8 | 1.50 | 34.7% | -0.57 |
| Matthew Golden | 23 | 1.38 | 32.7% | -0.57 |
| Tre Harris | 55 | 1.35 | 29.2% | +0.14 |
| Kyle Williams | 69 | 1.21 | 27.1% | -0.22 |
| Jayden Higgins | 34 | 1.19 | 26.9% | -0.40 |
| Luther Burden | 39 | 1.09 | 22.2% | -0.26 |
| Jack Bech | 58 | 0.98 | 18.1% | -0.28 |
| Savion Williams | 87 | 0.93 | 17.1% | -0.30 |

**Travis Hunter** is the clear WR1 with 59% P(Elite+) -- the highest of any prospect in any class. Edge of -0.75 means draft capital at pick 2 is doing heavy lifting beyond his college analytics.

**Tre Harris** (pick 55) has the only positive edge in the top 10 (+0.14). His college analytics outperform his draft slot -- a value target.

**Egbuka vs McMillan**: Essentially tied at E ~1.51, but Egbuka has the better college profile (college E = 0.97 vs 0.94) while McMillan has the higher pick (8 vs 19).

---

## 2026 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Carnell Tate | 4 | 2.06 | 48.1% | -0.86 |
| Makai Lemon | 20 | 1.81 | 42.2% | -0.44 |
| Omar Cooper Jr. | 30 | 1.63 | 38.4% | -0.35 |
| Jordyn Tyson | 8 | 1.10 | 23.3% | -0.52 |
| Antonio Williams | 71 | 1.07 | 22.8% | -0.18 |

**Carnell Tate** leads 2026 at pick 4 with 48% P(Elite+).

---

## The Edge Column

The "edge" column (E[full] - E[college]) measures how much draft capital changes the prediction. Key interpretations:

- **Large negative edge** (e.g., -0.85): Draft capital is boosting the player significantly. The market agrees with or exceeds the college analytics. These are consensus top picks.
- **Large positive edge** (e.g., +0.40): College analytics far outperform draft slot. The market underpriced this player. This is where dynasty edge lives -- Puka Nacua had edge +0.43 in the holdout.
- **Near-zero edge**: Draft capital and college analytics agree. No market inefficiency to exploit.

Caution: positive edge can also mean the college stats are noisy for that player (small sample, inflated metrics). Context matters.

---

## Data Pipeline Notes

### 200-Route Filter
The `best2_stats()` function in `aggregation/aggregate_college_stats.py` now requires seasons to have 200+ routes to be eligible for best-2 selection. This prevents small-sample inflation (e.g., 5 routes / 86 yards = 17.2 YPRR). Players with fewer than 2 eligible seasons fall back to their available 200+ route seasons, or all seasons if none qualify.

### CCR Data Availability
Contested catch rate data starts in 2018, limiting training data to 7 draft classes (2018-2024). This is why the model trains on ~120 players for holdout evaluation (2018-2021) and ~210 for prospect predictions (2018-2024).

---

## Known Limitations

1. **Small sample.** ~210 complete cases (2018-2024). Every feature addition risks overfitting.
2. **Holdout is only 89 players.** v5-to-v6 differences are within noise. The structural argument for simplicity matters more.
3. **No opportunity modeling.** Landing spot, depth chart, and scheme fit are not included. This is why some busts have good profiles (Treylon Burks) and some hits have bad profiles (Rashee Rice).
4. **best2_yprr Bayesian posterior crosses zero.** HDI [-0.317, +0.465]. The feature helps in LOO-AUC testing but the posterior is wide. More data will tighten this.
5. **best2_avoided_tackles_per_rec has marginal LOO-AUC impact.** It doesn't improve LOO-AUC (-0.003 vs 4-feature core), but it has genuine orthogonal signal (84.2% bootstrap positive, max collinearity 0.222, era drift 0.051). It captures a distinct skill dimension -- post-catch elusiveness -- that the other 4 features cannot measure.
6. **Edge interpretation requires caution.** Large positive edge can mean "undervalued by the market" or "noisy college stats." Always check the raw feature values.
