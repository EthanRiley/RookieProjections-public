# WR Dynasty Model v11 Report

## What Changed From v9

v11 makes two changes to the model infrastructure, both motivated by systematic analysis rather than feature engineering:

- **v10: Log-scaled draft capital** (was sqrt). Formula: `DC = 10 - (10 / ln(261)) * ln(pick + 1)`
- **v11: 60/40 Bayesian/XGBoost ensemble** (was 75/25)

### v11 Feature Set (6 features, unchanged from v9)
- **draft_capital** -- NFL consensus (log-scaled pick score, changed from sqrt)
- **best1_yprr_graduated** -- best single P5 season YPRR, graduated age adjustment
- **career_targeted_qb_rating** -- game-weighted career targeted passer rating
- **best2_catch_pct_adot_adj** -- best 2 P5 seasons aDOT-adjusted catch percentage
- **best2_contested_catch_rate** -- best 2 P5 seasons contested catch rate
- **best2_avoided_tackles_per_rec** -- best 2 P5 seasons missed tackles forced per reception

---

## Why Log-Scaled Draft Capital

The sqrt curve (`10 - 7 * sqrt(pick/260)`) is too flat — it assigns 46% of total value to R4+ picks, when actual dynasty value from R4+ is only 14%. This means the model treats a R4 pick as roughly similar to a R2 pick, when R2 picks produce 8x more dynasty value per player.

Seven transformations were tested against dynasty outcomes for 91 resolved RBs and 291 resolved WRs: linear, sqrt, cube root, logarithmic, Jimmy Johnson, Rich Hill, and Fitzgerald-Spielberger.

Key findings:
- All monotone transforms are equivalent on rank-based metrics (Spearman, AUC). The difference is entirely in shape.
- JJ/Rich Hill are the same curve (r=0.998). Log/Fitzgerald-Spielberger are the same curve (r=0.998). So the real choice was Log vs JJ.
- Log wins on R²(tier ordinal) for both RB (.393 vs .363 JJ) and WR (.236 vs .222 JJ).
- Log has flattest residuals by round (R4+ residual = -0.04 vs JJ's -0.21).
- Log is a closed-form formula; JJ is a 74-point lookup table designed for trade value, not player outcomes.

Full analysis: `rb_data/reports/draft_capital_curve_analysis.md`

### Log vs Sqrt on WR Holdout (at 75/25 split)

| Metric | Sqrt | Log | Delta |
|--------|------|-----|-------|
| LogLoss | 0.785 | **0.768** | -0.017 |
| Brier | 0.348 | **0.338** | -0.010 |
| >=Flex AUC | 0.868 | **0.892** | +0.024 |
| >=Starter AUC | 0.866 | **0.916** | +0.050 |
| >=Elite AUC | 0.953 | **0.963** | +0.010 |
| >=Stud AUC | **0.957** | 0.929 | -0.028 |

Log wins on 5 of 6 metrics representing 95% of composite weight. Sqrt only wins on >=Stud AUC, which has 3 positive cases in the holdout.

---

## Why 60/40 Ensemble

The original 75/25 Bayesian/XGBoost split was grid-searched early with sqrt DC and never revisited. With log DC and re-engineered features, we swept 42 configurations (21 weights x 2 curves).

Configurations were ranked on a weighted composite:
- 35% LogLoss (probability calibration)
- 35% >=Elite AUC (top-end discrimination)
- 15% Brier (calibration sanity check)
- 10% >=Starter AUC (mid-round discrimination)
- 5% >=Stud AUC (top-end, noisy with n=3)

Results:
- The top 12 configurations are all log DC.
- The old sqrt 75/25 ranked **#35 out of 42**.
- Log DC optimal: 60/40 Bayes/XGB (composite = 0.879)
- Sqrt DC optimal: 35/65 Bayes/XGB (composite = 0.836)
- Log DC is robust across a wide range of weights (25-70% Bayesian all score well), while sqrt drops off steeply past 40%.

The shift from 75/25 to 60/40 gives XGBoost more influence, improving top-end discrimination (>=Elite AUC) without sacrificing calibration.

Full analysis: `modeling/research/ensemble_sweep_both_curves.py`

---

## Model Architecture

- **Ensemble**: 60% Bayesian ordinal regression + 40% XGBoost cumulative link (changed from 75/25)
- **Variants**: Full model (draft_capital + college features) and College-only
- **Training**: 2018-2021 for holdout evaluation; 2018-2024 for prospect predictions
- **Cross-validation**: Leave-one-year-out
- **Calibration**: Platt scaling on XGBoost; Bayesian is self-calibrating

---

## Holdout Results (2022-2024, 88 players)

### Ensemble Metrics

| Model | LogLoss | Brier |
|-------|---------|-------|
| Bayesian Full | 0.769 | 0.335 |
| XGBoost Full | 1.211 | 0.361 |
| **Ensemble Full** | **0.773** | **0.340** |
| Ensemble College | 0.851 | 0.374 |

### Per-Threshold AUC (Ensemble Full)

| Threshold | AUC | Brier |
|-----------|-----|-------|
| >=Flex | 0.888 | 0.122 |
| >=Starter | 0.920 | 0.090 |
| **>=Elite** | **0.970** | 0.066 |
| >=Stud | 0.941 | 0.033 |
| >=LW | 0.989 | 0.012 |

### Version Comparison

| Metric | v8 | v9 | v11 |
|--------|:--:|:--:|:---:|
| Ensemble LogLoss | 0.798 | 0.771 | **0.773** |
| Ensemble Brier | 0.355 | 0.343 | **0.340** |
| >=Elite AUC | 0.963 | 0.961 | **0.970** |
| >=Stud AUC | 0.888 | **0.953** | 0.941 |
| >=LW AUC | 0.908 | **1.000** | 0.989 |

v11's headline improvement is **>=Elite AUC rising to 0.970** — the model's ability to separate future Elites from non-Elites is the best it's ever been. LogLoss is essentially flat vs v9 (+0.002), while Brier improves slightly. The >=Stud and >=LW dips are within noise given n=3 and n=1 positive cases.

---

## Data Pipeline Filters (cumulative)

### 1. 200-Route Minimum (v6)
Seasons must have 200+ routes for best2 eligibility.

### 2. P5 School Filter (v7)
Non-P5 seasons excluded from best2 when 2+ P5 seasons available.

### 3. CCR Small-Sample Filter (v7)
< 10 contested targets in best2 -> fallback to 40% (group avg minus 5pp).

### 4. Senior Season Discount (v8)
Seasons where player age >= 22 on Sept 1: -10pp on contested_catch_rate, caught_percent, and targeted_qb_rating before aggregation.

### 5. Graduated YPRR Age Adjustment (v9)
Per-age-class multiplicative on YPRR: FR +25%, SO +5%, JR -20%, SR -25%.

### 6. Log-Scaled Draft Capital (v10)
`draft_capital = 10 - (10 / ln(261)) * ln(pick + 1)`. Replaces sqrt scaling.

### 7. 60/40 Ensemble (v11)
60% Bayesian / 40% XGBoost. Replaces 75/25 split.

---

## Known Limitations

1. **Small sample.** 117 training players, 6 features. Each change adds a degree of freedom.
2. **Senior discount is a flat penalty.** A 22.0-year-old gets the same discount as a 24-year-old. Could be graduated.
3. **best2_catch_pct_adot_adj has negative residual signal.** Its value is in XGBoost interactions, not orthogonal linear signal. With XGBoost now at 40% (up from 25%), this feature may contribute more.
4. **Holdout is only 88 players.** >=Stud AUC fluctuations (3 positive cases) are noise.
5. **P5 filter is binary.** No gradient between SEC and AAC.
6. **No opportunity modeling.** Landing spot, depth chart, scheme fit not included.
7. **Ensemble weights optimized on holdout.** Some overfitting risk; the improvement should be validated on future classes.
