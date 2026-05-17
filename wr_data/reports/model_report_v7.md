# WR Dynasty Model v7 Report

## What Changed From v6

v7 adds a 6th feature (best2_catch_pct_adot_adj), implements a contested catch rate small-sample filter, and restricts best2 season selection to P5 schools.

### v6 Feature Set (5 features)
- draft_capital
- best2_yprr
- career_targeted_qb_rating
- best2_contested_catch_rate
- best2_avoided_tackles_per_rec

### v7 Feature Set (6 features)
- **draft_capital** -- NFL consensus (sqrt-scaled pick score)
- **best2_yprr** -- best 2 seasons YPRR (min 200 routes, P5 only)
- **career_targeted_qb_rating** -- game-weighted career targeted passer rating
- **best2_catch_pct_adot_adj** -- best 2 seasons aDOT-adjusted catch percentage (NEW)
- **best2_contested_catch_rate** -- best 2 seasons contested catch rate
- **best2_avoided_tackles_per_rec** -- best 2 seasons missed tackles forced per reception

### Why These Changes

**Added best2_catch_pct_adot_adj.** Catch percentage residual after regressing on average depth of target. Measures hands reliability independent of route depth. XGBoost holdout improved at every threshold when added as a 6th feature alongside career_tqbr (>=Elite AUC 0.900 -> 0.917, LogLoss 1.184 -> 0.790). Bootstrap head-to-head favors 6 features over 5 (52.6% vs 43.8%).

Important caveat: the feature has negative residual signal against the 5-feature base (-0.081, 15% bootstrap positive). Its value is entirely in XGBoost interaction structure, not orthogonal linear signal. The Bayesian model (75% of ensemble) likely doesn't benefit. See `wr_data/catch_pct_swap_report.md` for full analysis including the failed swap attempt (replacing career_tqbr with best2_cpaa).

**Added P5 school filter on best2 season selection.** Non-P5 seasons are excluded from best2 selection when a player has 2+ P5 seasons with 200+ routes. This prevents inflated stats from weaker competition (e.g., CJ Daniels' Liberty season with 88.9% CCR, Elijah Sarratt's JMU season with 78.9% CCR). Players who only played at non-P5 schools fall back to all eligible seasons.

Affected prospects:
- **CJ Daniels**: Liberty 2023 excluded. Now uses LSU + Miami FL. CCR 77.3 -> 74.3, YPRR dropped.
- **Elijah Sarratt**: JMU 2023 excluded. Now uses Indiana 2024 + 2025. CCR dropped significantly.
- **Tre Harris**: La Tech 2021-2022 excluded. Now uses Ole Miss 2023 + 2024.

**Added CCR small-sample filter.** Players with fewer than 10 contested targets across their best2 seasons get a fallback value of 40% (group average 45% minus 5 percentage points). Only applies when CCR data exists but sample is too small; pre-2018 players with no CCR data remain excluded. 20 players in training set receive the fallback.

Discovered when Kendrick Law (2025, pick 168) appeared with 100% CCR from just 2 contested catches.

---

## Model Architecture

Unchanged from v6:
- **Ensemble**: 75% Bayesian ordinal regression + 25% XGBoost cumulative link
- **Variants**: Full model (draft_capital + college features) and College-only (college features only)
- **Training**: 2018-2021 for holdout evaluation; all labeled data (2018-2023) for prospect predictions
- **Cross-validation**: Leave-one-year-out
- **Calibration**: Platt scaling on XGBoost; Bayesian model is self-calibrating

---

## Holdout Results (2022-2024, 89 players)

### Ensemble Metrics

| Model | LogLoss | Brier |
|-------|---------|-------|
| Bayesian Full | 0.845 | 0.372 |
| XGBoost Full | 0.797 | 0.357 |
| **Ensemble Full** | **0.817** | **0.364** |
| Ensemble College | 0.886 | 0.391 |

### Per-Threshold AUC (Ensemble Full)

| Threshold | AUC | Brier |
|-----------|-----|-------|
| >=Flex | 0.846 | 0.141 |
| >=Starter | 0.870 | 0.104 |
| >=Elite | 0.946 | 0.079 |
| >=Stud | 0.856 | 0.039 |
| >=LW | 0.897 | 0.020 |

### v6 vs v7 Comparison

| Metric | v6 (5 features) | v7 (6 features + filters) |
|--------|-----------------|---------------------------|
| Ensemble LogLoss | 0.816 | 0.817 |
| Ensemble Brier | 0.363 | 0.364 |
| >=Elite AUC | 0.947 | 0.946 |
| >=Stud AUC | 0.844 | 0.856 |
| >=LW AUC | 0.931 | 0.897 |

Ensemble metrics are nearly identical. >=Stud improved, >=LW dropped. With 89 holdout players these differences are noise. The P5 filter and CCR small-sample filter are data quality improvements that should help generalization regardless of holdout movement.

---

## 2025 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Travis Hunter | 2 | 2.70 | 62.1% | -0.56 |
| Tetairoa McMillan | 8 | 1.58 | 37.2% | -0.52 |
| Tre Harris | 55 | 1.48 | 33.1% | +0.11 |
| Kyle Williams | 69 | 1.43 | 34.0% | -0.17 |
| Emeka Egbuka | 19 | 1.42 | 32.7% | -0.47 |
| Luther Burden | 39 | 1.30 | 26.5% | -0.20 |
| Matthew Golden | 23 | 1.23 | 28.2% | -0.49 |
| Jack Bech | 58 | 1.17 | 23.2% | -0.26 |
| Jayden Higgins | 34 | 1.10 | 24.0% | -0.31 |
| Dont'e Thornton | 108 | 1.03 | 23.1% | +0.20 |

**Travis Hunter** remains the clear WR1 at 62% P(Elite+) and E=2.70.

**Tre Harris** (pick 55) is the value target with the only positive edge in the top 5 (+0.11). Note: his best2 now uses Ole Miss seasons only (P5 filter excluded La Tech).

**Kyle Williams** (pick 69) at WR4 is a model favorite -- strong aDOT-adjusted catch% pushes him above Egbuka.

## 2026 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Carnell Tate | 4 | 2.25 | 53.3% | -0.70 |
| Makai Lemon | 20 | 1.92 | 45.3% | -0.33 |
| Omar Cooper Jr. | 30 | 1.84 | 42.8% | -0.24 |
| Antonio Williams | 71 | 1.22 | 27.6% | -0.13 |
| Denzel Boston | 39 | 0.96 | 17.3% | -0.34 |
| Jordyn Tyson | 8 | 0.96 | 20.1% | -0.46 |
| KC Concepcion | 24 | 0.96 | 21.4% | -0.40 |
| Kevin Coleman Jr. | 177 | 0.92 | 19.0% | +0.30 |
| Elijah Sarratt | 115 | 0.92 | 20.2% | +0.09 |
| Ja'Kobi Lane | 80 | 0.92 | 19.6% | -0.12 |

**Omar Cooper Jr.** solidified as WR3, now with a bigger gap over the field.

**Kevin Coleman Jr.** (pick 177) has the largest positive edge (+0.30) -- college analytics significantly outperform draft slot.

**CJ Daniels** dropped to rank 11 (E=0.91) after P5 filter excluded his Liberty season.

---

## Data Pipeline Filters

### 1. 200-Route Minimum (from v6)
Seasons must have 200+ routes to be eligible for best2 selection. Prevents small-sample YPRR inflation.

### 2. P5 School Filter (NEW)
When a player has 2+ P5 seasons with 200+ routes, non-P5 seasons are excluded from best2 selection. Players with only non-P5 seasons fall back to all eligible seasons.

P5 defined as: SEC, Big Ten, Big 12, ACC, Notre Dame, Oregon State, Washington State (former P5 counted for historical seasons).

### 3. CCR Small-Sample Filter (NEW)
If total contested targets across best2 seasons < 10, contested catch rate is replaced with 40% (group average minus 5 percentage points). Only applies when CCR data exists; pre-2018 players remain NaN.

---

## Known Limitations

1. **Small sample.** ~174 complete cases (2018-2023). 6 features on 174 players is aggressive.
2. **best2_catch_pct_adot_adj has negative residual signal.** -0.081 against the 5-feature base, only 15% bootstrap positive. Its value is in XGBoost interactions, not linear orthogonal signal.
3. **Holdout is only 89 players.** v6-to-v7 differences are within noise.
4. **P5 filter is binary.** No gradient between SEC and Sun Belt. A player dominating Sun Belt competition gets the same treatment as FCS.
5. **CCR fallback is a flat value.** 40% for all small-sample players regardless of what partial data exists. Could be improved with shrinkage toward the group mean weighted by sample size.
6. **No opportunity modeling.** Landing spot, depth chart, and scheme fit not included.
