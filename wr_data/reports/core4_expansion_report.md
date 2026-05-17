# Testing best2_yprr and best2_avoided_tackles_per_rec Against the 4-Feature Core

## Setup

**Base model (4-feature core):**
- draft_capital
- breakout_age
- career_targeted_qb_rating
- best2_contested_catch_rate

**Complete cases:** 211 players (2018-2024). Requires CCR data (available 2018+).

**Base4 LOO-AUC: 0.864**

---

## Results

### best2_yprr

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Residual Spearman | **-0.133** | Negative -- base already captures this |
| Bootstrap %+ | **2.5%** | Signal is negative in 97.5% of resamples |
| LOO-AUC delta | +0.002 | Trivial; contradicts residual (overfitting pattern) |
| Era drift | 0.148 | Unstable across eras |

**Collinearity with base features:**

| Base Feature | Correlation |
|-------------|-------------|
| draft_capital | +0.393 |
| breakout_age | **-0.497** |
| career_targeted_qb_rating | **+0.471** |
| best2_contested_catch_rate | +0.044 |

**Verdict: No.** best2_yprr is heavily redundant with both breakout_age (r=0.497) and career_tqbr (r=0.471). Those two features already capture route efficiency from different angles -- breakout_age measures when a player first hit 2.0+ YPRR, and career_tqbr reflects the downstream QB outcomes of running good routes. Adding best2_yprr on top provides zero new information. The 2.5% bootstrap positive is as close to "definitively redundant" as you can get.

The +0.002 LOO-AUC is the dangerous pattern flagged throughout the full analysis: a tiny AUC improvement without residual support. The logistic regression found a spurious weight that happens to help in-sample but has no genuine orthogonal basis.

---

### best2_avoided_tackles_per_rec

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Residual Spearman | **+0.066** | Positive -- genuinely orthogonal signal |
| Bootstrap %+ | **84.2%** | Reliable across resamples |
| LOO-AUC delta | -0.003 | Doesn't help prediction |
| Era drift | 0.051 | Very stable |

**Collinearity with base features:**

| Base Feature | Correlation |
|-------------|-------------|
| draft_capital | +0.142 |
| breakout_age | -0.222 |
| career_targeted_qb_rating | +0.129 |
| best2_contested_catch_rate | -0.047 |

**Verdict: Real signal, but too small to justify inclusion.** best2_avoided_tackles_per_rec is the most independent feature tested against the 4-feature core -- max collinearity of just 0.222, and essentially zero overlap with CCR (-0.047). The 84.2% bootstrap positive confirms this isn't noise.

What it measures is mechanistically distinct: the ability to make defenders miss after the catch. Draft capital, breakout age, QB trust, and contested catch ability don't capture this. It's a real, stable, orthogonal skill dimension.

The problem is magnitude. At +0.066 residual Spearman and -0.003 LOO-AUC, the signal exists but doesn't move the needle with 211 players. Adding a 5th feature to chase a 0.066 effect risks overfitting more than it adds predictive value.

---

### Both Together

| Configuration | LOO-AUC | Delta vs Base4 |
|--------------|---------|----------------|
| Base4 only | 0.864 | -- |
| + best2_yprr | 0.867 | +0.002 |
| + best2_avoided_tackles_per_rec | 0.862 | -0.003 |
| + both | 0.864 | +0.000 |

Adding both features simultaneously nets exactly zero improvement. The tiny positive from best2_yprr and tiny negative from best2_mtf cancel out perfectly. The model is already saturated at 4 features for this sample size.

---

## Implications for the Model

The 4-feature core (draft_capital + breakout_age + career_tqbr + best2_ccr) appears to be the right stopping point. Each feature captures a distinct dimension:

1. **draft_capital** -- aggregate NFL talent evaluation
2. **breakout_age** -- production timing / early dominance
3. **career_targeted_qb_rating** -- QB trust / route quality
4. **best2_contested_catch_rate** -- contested ball skills

The only remaining candidate with genuine orthogonal signal is best2_avoided_tackles_per_rec (elusiveness after the catch), but the effect is too small to justify the added complexity. If the dataset grows substantially in future years, this is the first feature to re-evaluate.

best2_yprr should not be considered further. Its information is fully absorbed by breakout_age and career_tqbr.
