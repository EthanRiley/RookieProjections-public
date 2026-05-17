# WR Dynasty Model v8 Report

## What Changed From v7

v8 adds a senior season discount: percentage-based rate stats are penalized by 10 percentage points for any season where the player is 22 or older on September 1st. This corrects for the systematic advantage that older players have when competing against younger opponents.

### v8 Feature Set (6 features, unchanged from v7)
- **draft_capital** -- NFL consensus (sqrt-scaled pick score)
- **best2_yprr** -- best 2 P5 seasons YPRR (min 200 routes)
- **career_targeted_qb_rating** -- game-weighted career targeted passer rating
- **best2_catch_pct_adot_adj** -- best 2 P5 seasons aDOT-adjusted catch percentage
- **best2_contested_catch_rate** -- best 2 P5 seasons contested catch rate
- **best2_avoided_tackles_per_rec** -- best 2 P5 seasons missed tackles forced per reception

### Why the Senior Discount

Non-early-declare players get the benefit of their senior (age 22+) season in best2 selection. By that point they are physically mature adults competing against 18-19 year olds. Their rate stats are systematically inflated relative to early declares who entered the NFL at 20-21.

The discount applies to percentage-based rate stats only:
- `contested_catch_rate` (-10pp)
- `caught_percent` (-10pp, which flows into `catch_pct_adot_adj`)
- `targeted_qb_rating` (-10 points)

Non-percentage stats (YPRR, avoided_tackles_per_rec) are not discounted -- route efficiency and elusiveness don't inflate the same way from age advantage.

**Threshold**: age >= 22 on September 1st of the season year. Computed from nflverse birthdates.

### Impact Examples

| Player | Draft Age | Affected Season | CCR Before | CCR After |
|--------|-----------|-----------------|:----------:|:---------:|
| Tre Harris | 23.15 | 2024 Ole Miss (age 22.5) | 61.5% | 51.5% |
| Elijah Sarratt | 22.9 | 2025 Indiana (age 22.6) | 40.0% | 30.0% |
| Travis Hunter | 21.94 | None | -- | -- |
| Tetairoa McMillan | 20.9 | None | -- | -- |

Early declares (Hunter, McMillan, Burden, etc.) are unaffected. The discount only hits players who stayed in school past age 22.

---

## Model Architecture

Unchanged:
- **Ensemble**: 75% Bayesian ordinal regression + 25% XGBoost cumulative link
- **Variants**: Full model (draft_capital + college features) and College-only
- **Training**: 2018-2021 for holdout evaluation; 2018-2023 for prospect predictions
- **Cross-validation**: Leave-one-year-out
- **Calibration**: Platt scaling on XGBoost; Bayesian is self-calibrating

---

## Holdout Results (2022-2024, 89 players)

### Ensemble Metrics

| Model | LogLoss | Brier |
|-------|---------|-------|
| Bayesian Full | 0.830 | 0.365 |
| XGBoost Full | 1.131 | 0.345 |
| **Ensemble Full** | **0.798** | **0.355** |
| Ensemble College | 0.861 | 0.384 |

### Per-Threshold AUC (Ensemble Full)

| Threshold | AUC | Brier |
|-----------|-----|-------|
| >=Flex | 0.885 | 0.135 |
| >=Starter | 0.895 | 0.100 |
| **>=Elite** | **0.963** | 0.077 |
| >=Stud | 0.888 | 0.039 |
| >=LW | 0.908 | 0.020 |

### Version Comparison

| Metric | v6 (5 feat) | v7 (6 feat + filters) | v8 (+ senior discount) |
|--------|:-:|:-:|:-:|
| Ensemble LogLoss | 0.816 | 0.817 | **0.798** |
| Ensemble Brier | 0.363 | 0.364 | **0.355** |
| >=Elite AUC | 0.947 | 0.946 | **0.963** |
| >=Stud AUC | 0.844 | 0.856 | **0.888** |
| >=LW AUC | 0.931 | 0.897 | **0.908** |

The senior discount is the biggest single improvement since the v6 rebuild. Every metric improved, with >=Stud AUC jumping +0.044 and >=Elite AUC reaching 0.963.

---

## 2024 Prospect Predictions (Holdout Validation)

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Malik Nabers | 6 | 2.02 | 48.2% | -0.40 |
| Ladd McConkey | 34 | 1.92 | 44.0% | -0.25 |
| Brian Thomas | 23 | 1.83 | 42.6% | -0.42 |
| Marvin Harrison Jr. | 4 | 1.71 | 39.9% | -0.56 |
| Rome Odunze | 9 | 1.46 | 34.6% | -0.50 |
| Troy Franklin | 102 | 1.25 | 28.5% | +0.27 |
| Jermaine Burton | 80 | 1.06 | 23.6% | +0.03 |
| Ja'Lynn Polk | 37 | 1.05 | 23.0% | -0.36 |
| Keon Coleman | 33 | 0.91 | 20.4% | -0.37 |
| Tahj Washington | 241 | 0.84 | 18.0% | +0.40 |

MHJ dropped from WR1 to WR4 in 2024 -- the model's college analytics prefer Nabers, McConkey, and Brian Thomas. With only 1 NFL season of data, MHJ's actual tier is still TBD.

---

## 2025 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Travis Hunter | 2 | 2.76 | 63.2% | -0.48 |
| Tetairoa McMillan | 8 | 1.61 | 37.8% | -0.48 |
| Kyle Williams | 69 | 1.47 | 34.4% | -0.15 |
| Emeka Egbuka | 19 | 1.44 | 32.9% | -0.44 |
| Luther Burden | 39 | 1.34 | 27.4% | -0.19 |
| Matthew Golden | 23 | 1.25 | 28.7% | -0.45 |
| Jack Bech | 58 | 1.20 | 24.1% | -0.24 |
| Tre Harris | 55 | 1.18 | 25.7% | +0.03 |
| Jayden Higgins | 34 | 1.16 | 25.2% | -0.31 |
| Dont'e Thornton | 108 | 1.11 | 24.7% | +0.23 |

**Travis Hunter** is the clear WR1 at 63.2% P(Elite+) and E=2.76 -- the highest of any prospect across all classes.

**Kyle Williams** (pick 69) at WR3 is a value play -- his aDOT-adjusted catch% and CCR are strong, and his edge of -0.15 means the market is close to fair.

**Tre Harris** dropped from WR3 (pre-discount) to WR8. His age-22 Ole Miss season was discounted, which aligns with his pick 55 draft position. His edge is near zero (+0.03), meaning the full and college-only models now agree.

---

## 2026 Prospect Predictions

| Name | Pick | E[full] | P(Elite+) | Edge |
|------|------|---------|-----------|------|
| Carnell Tate | 4 | 2.31 | 54.4% | -0.64 |
| Makai Lemon | 20 | 1.96 | 46.2% | -0.30 |
| Omar Cooper Jr. | 30 | 1.88 | 43.3% | -0.23 |
| Antonio Williams | 71 | 1.31 | 30.1% | -0.14 |
| Denzel Boston | 39 | 1.00 | 18.6% | -0.33 |
| CJ Daniels | 197 | 0.99 | 22.1% | +0.21 |
| Kevin Coleman Jr. | 177 | 0.99 | 19.8% | +0.30 |
| Jordyn Tyson | 8 | 0.98 | 20.5% | -0.45 |
| Ja'Kobi Lane | 80 | 0.94 | 20.3% | -0.09 |
| KC Concepcion | 24 | 0.93 | 19.3% | -0.36 |

**Omar Cooper Jr.** continues to solidify as WR3, with a clear gap over the field after Antonio Williams.

**Kevin Coleman Jr.** (pick 177) has the biggest positive edge (+0.30) -- college analytics far outperform his late-round draft slot.

**CJ Daniels** (pick 197) at WR6 despite pick 197 -- his P5 seasons at LSU and Miami FL are strong even after the P5 filter excluded his Liberty season.

---

## Data Pipeline Filters (cumulative)

### 1. 200-Route Minimum (v6)
Seasons must have 200+ routes for best2 eligibility.

### 2. P5 School Filter (v7)
Non-P5 seasons excluded from best2 when 2+ P5 seasons available.

### 3. CCR Small-Sample Filter (v7)
< 10 contested targets in best2 -> fallback to 40% (group avg minus 5pp).

### 4. Senior Season Discount (v8)
Seasons where player age >= 22 on Sept 1: -10pp on contested_catch_rate, caught_percent, and targeted_qb_rating before aggregation. Does not affect YPRR or avoided_tackles_per_rec.

---

## Known Limitations

1. **Small sample.** 174 training players, 6 features. Each filter/discount adds a degree of freedom.
2. **Senior discount is a flat penalty.** A 22.0-year-old gets the same discount as a 24-year-old. Could be graduated.
3. **best2_catch_pct_adot_adj has negative residual signal.** Its value is in XGBoost interactions, not orthogonal linear signal. The Bayesian model (75% of ensemble) may not benefit from it.
4. **Holdout is only 89 players.** v7-to-v8 improvements look strong but could be noise.
5. **P5 filter is binary.** No gradient between SEC and AAC. Players who only played at G5 schools get no filter applied.
6. **No opportunity modeling.** Landing spot, depth chart, scheme fit not included.
7. **targeted_qb_rating discount.** -10 points on a 0-158.3 scale is smaller relative to -10pp on a 0-100% scale. May want to tune this separately.
