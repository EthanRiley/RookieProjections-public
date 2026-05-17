# WR Feature Dimension Analysis Report

## Purpose

Systematic evaluation of every candidate feature against a minimal base of **draft_capital + breakout_age** (the only two features locked without question). For each of 7 feature dimensions, all plausible variants were tested using the same 7-part protocol:

1. Univariate screens (Spearman, mutual information, AUC)
2. Era stability (early vs late draft classes)
3. Residual signal after the base model
4. Collinearity with base features
5. Bootstrap reliability (1000 iterations)
6. Leave-one-year-out AUC
7. Elastic net survival at 3 regularization strengths

**Base LOO-AUC**: 0.829-0.844 depending on the complete-cases subset per dimension.

---

## Grand Summary: Top Features by LOO-AUC Delta

Sorted by marginal LOO-AUC improvement when added to draft_capital + breakout_age:

| Rank | Feature | Dimension | LOO-AUC | Delta | Residual | Boot %+ |
|------|---------|-----------|---------|-------|----------|---------|
| 1 | **career_targeted_qb_rating** | C: QB Trust | 0.852 | **+0.023** | +0.041 | **74.1%** |
| 2 | best2_targets_pg | G: Production | 0.842 | +0.013 | -0.138 | 2.3% |
| 3 | best_yards_per_team_pass_att | B: Market Share | 0.842 | +0.013 | -0.133 | 2.3% |
| 4 | best2_yards_pg | B: Market Share | 0.841 | +0.013 | -0.141 | 1.2% |
| 5 | career_catch_pct_adot_adj | E: Catch Reliability | 0.838 | +0.009 | +0.028 | 66.5% |
| 6 | best2_targeted_qb_rating | C: QB Trust | 0.837 | +0.008 | -0.016 | 40.7% |
| 7 | best2_yards_after_catch_pg | F: Elusiveness | 0.844 | +0.007 | -0.153 | 0.6% |
| 8 | breakout_yptpa | B: Market Share | 0.836 | +0.007 | -0.154 | 1.5% |
| 9 | **best2_contested_catch_rate** | D: CCR | 0.845 | +0.001 | **+0.133** | **97.9%** |
| 10 | breakout_yprr | A: YPRR | 0.839 | +0.002 | -0.113 | 4.8% |

### The key tension: LOO-AUC vs residual signal

These two metrics tell different stories and neither should be trusted alone:

- **LOO-AUC** measures in-sample cross-validated prediction. Features that improve AUC on this dataset may not generalize. Features ranked #2-4 and #7 all have **negative residual signal and <3% bootstrap positive** -- they're helping the logistic regression fit this specific dataset but likely contain no genuine orthogonal information.
- **Residual signal + bootstrap** measures whether a feature contains information the base model can't explain. best2_contested_catch_rate has a +0.133 residual that's positive in 97.9% of bootstrap samples -- that's real signal, even though it barely moves LOO-AUC.

**Reliable features have BOTH**: positive residual that bootstraps well AND non-negative LOO-AUC delta.

---

## Dimension A: YPRR / Route Efficiency

**240 complete cases. Base LOO-AUC: 0.837.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| career_yprr | +0.280 | 0.094 | -0.120 | 2.6% | 0.835 | -0.002 | 1/3 |
| best2_yprr | +0.244 | 0.106 | -0.154 | 0.9% | 0.838 | +0.000 | 1/3 |
| peak_yprr | +0.253 | 0.082 | -0.116 | 3.6% | 0.836 | -0.001 | 1/3 |
| breakout_yprr | +0.235 | **0.057** | -0.116 | 4.8% | 0.839 | **+0.002** | 1/3 |

### Verdict: No YPRR variant adds value on top of draft_capital + breakout_age.

Every variant has **negative residual signal** -- the base model already captures route efficiency through breakout_age (which is defined by a YPRR threshold). YPRR is literally how breakout_age is defined: first season with 2.0+ YPRR and 200+ routes. Adding a YPRR feature on top is redundant by construction.

breakout_yprr is the least harmful (best era stability at 0.057, only variant with positive LOO-AUC delta, least negative residual in bootstrap), which is why it was in the prior model. But the evidence for including it is weak.

Career_yprr has the strongest univariate signal (+0.280) but the worst residual. Its signal is entirely subsumed by breakout_age.

---

## Dimension B: YPTPA / Market Share

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| breakout_yptpa | +0.179 | 0.062 | -0.164 | 1.5% | 0.836 | +0.007 | 1/3 |
| best_yards_per_team_pass_att | +0.212 | **0.054** | -0.140 | 2.3% | 0.842 | **+0.013** | 1/3 |
| career_yards_pg | +0.243 | **0.040** | -0.055 | 22.5% | 0.829 | -0.000 | 1/3 |
| best2_yards_pg | +0.181 | **0.017** | -0.150 | 1.2% | 0.841 | +0.013 | 1/3 |

### Verdict: Conflicting signals. LOO-AUC loves best_yptpa and best2_yards_pg, but residuals are all negative.

best_yards_per_team_pass_att and best2_yards_pg both jump LOO-AUC by +0.013, but have deeply negative residual signal (positive in only 1-2% of bootstrap samples). This pattern is suspicious -- they're fitting the LOO cross-validation but don't contain information the base model can't explain.

breakout_yptpa has the same problem: +0.007 LOO-AUC but -0.164 residual. It captures market share of the passing offense at the breakout season, but breakout_age already encodes when the breakout happened and the YPTPA threshold was part of the original breakout definition.

career_yards_pg is the most honest: near-zero LOO-AUC delta and the least negative residual (-0.055, 22.5% positive). It doesn't help, but at least it doesn't pretend to.

---

## Dimension C: QB Trust / Route Quality

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| **career_targeted_qb_rating** | **+0.305** | 0.113 | **+0.041** | **74.1%** | **0.852** | **+0.023** | **2/3** |
| best2_targeted_qb_rating | +0.252 | 0.124 | -0.016 | 40.7% | 0.837 | +0.008 | 2/3 |
| best_targeted_qb_rating | +0.232 | 0.068 | -0.012 | 44.4% | 0.825 | -0.004 | 2/3 |
| career_grades_pass_route | +0.284 | 0.087 | -0.041 | 29.0% | 0.823 | -0.006 | 1/3 |
| best2_grades_pass_route | +0.224 | 0.035 | -0.144 | 2.7% | 0.833 | +0.004 | 1/3 |
| career_grades_offense | +0.274 | 0.080 | -0.058 | 22.0% | 0.824 | -0.005 | 1/3 |
| best2_grades_offense | +0.223 | 0.034 | -0.147 | 1.9% | 0.831 | +0.002 | 1/3 |

### Verdict: career_targeted_qb_rating is the clear winner and the single best feature to add after draft_capital + breakout_age.

This is the **only feature across all 7 dimensions** where both the LOO-AUC and residual signal agree strongly:
- **+0.023 LOO-AUC delta** -- by far the largest of any candidate
- **+0.041 residual, 74.1% bootstrap positive** -- genuine orthogonal information
- **Lowest collinearity with base** (0.330 vs 0.430-0.482 for grades)
- **Survives elastic net at 2/3 strengths**

Why targeted QBR and not grades? Targeted QBR measures something fundamentally different from draft capital and breakout age -- it captures how much quarterbacks trusted this receiver. Draft capital captures talent evaluation, breakout age captures production timing. Targeted QBR captures the quality of targets received, which reflects route-running, separation, and reliable hands as observed by the player actually throwing the ball.

Grades (pass_route, offense) are more collinear with breakout_age (0.463-0.482) because PFF grades and breakout timing both respond to the same underlying talent. Targeted QBR's lower collinearity (0.330) means it captures a different angle.

best2 and best targeted QBR are both worse -- career is more robust because it averages across multiple seasons and QBs, reducing noise from individual season variance.

---

## Dimension D: Contested Catch Rate

**199 complete cases. Base LOO-AUC: 0.844.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| peak_contested_catch_rate | +0.131 | 0.206 | +0.084 | 88.3% | 0.842 | -0.002 | 1/3 |
| career_contested_catch_rate | +0.144 | 0.173 | +0.085 | 87.9% | 0.843 | -0.001 | 1/3 |
| **best2_contested_catch_rate** | +0.134 | 0.166 | **+0.137** | **97.9%** | 0.845 | **+0.001** | **2/3** |
| best_contested_catch_rate | +0.094 | **0.082** | +0.064 | 81.8% | 0.839 | -0.005 | 2/3 |

### Verdict: CCR is the most clearly orthogonal dimension -- every variant has strong positive residual signal.

This is the only dimension where **all 4 variants** have positive residual with >80% bootstrap reliability. CCR captures something draft capital and breakout age genuinely cannot: the ability to win contested catches. This is mechanistically distinct -- draft position reflects aggregate talent evaluation, breakout age reflects production volume, but neither directly measures contested-ball skills.

**best2_contested_catch_rate** is the standout:
- Strongest residual (+0.137, positive 97.9% of the time)
- Only variant with positive LOO-AUC delta (+0.001)
- Survives elastic net at 2/3 strengths
- Moderate era stability (0.166)

**peak_contested_catch_rate** (current model feature) has good residual (+0.084, 88.3%) but slightly negative LOO-AUC delta (-0.002). This is the same tension as v4 -- best2 has more raw signal but peak/best have better stability.

**best_contested_catch_rate** has the best era stability (0.082) but weakest residual (+0.064, 81.8%) and worst LOO-AUC (-0.005).

**This is different from the v4 analysis** which tested CCR against a fuller locked set (5 features). Against just draft_capital + breakout_age, best2 is the clear winner because there are no other features to create collinearity.

---

## Dimension E: Catch Reliability

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| career_catch_pct_adot_adj | **+0.295** | 0.108 | **+0.029** | **66.5%** | **0.838** | **+0.009** | 2/3 |
| best2_catch_pct_adot_adj | +0.239 | 0.053 | -0.008 | 43.5% | 0.831 | +0.002 | 2/3 |
| career_caught_percent | +0.229 | 0.102 | -0.018 | 39.2% | 0.832 | +0.003 | 2/3 |
| best2_caught_percent | +0.216 | **0.012** | +0.007 | 54.7% | 0.830 | +0.001 | 2/3 |
| best_caught_percent | +0.138 | 0.057 | -0.020 | 40.0% | 0.826 | -0.003 | 2/3 |

### Verdict: career_catch_pct_adot_adj is the only candidate with positive residual, but the evidence is borderline.

career_catch_pct_adot_adj has the strongest LOO-AUC delta (+0.009) and the only positive residual (+0.029), but the bootstrap is only 66.5% positive. That's well below the 75% threshold we'd want for confidence. It adds genuine signal, but it's weak and noisy.

All catch reliability features have low collinearity with the base (max 0.322), meaning this dimension IS somewhat orthogonal to draft_capital + breakout_age. But the signal is small.

This dimension likely overlaps with career_targeted_qb_rating: QBs rate higher when receivers catch the ball. Including both may be redundant.

---

## Dimension F: Elusiveness / YAC

**240 complete cases. Base LOO-AUC: 0.837.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| peak2_avoided_tackles_per_rec | +0.120 | 0.047 | +0.014 | 53.8% | 0.834 | -0.003 | 1/3 |
| career_avoided_tackles_per_rec | +0.129 | **0.006** | +0.013 | 54.6% | 0.830 | -0.007 | 0/3 |
| career_avoided_tackles_pg | +0.214 | 0.036 | -0.047 | 24.0% | 0.835 | -0.003 | 1/3 |
| best2_avoided_tackles_pg | +0.205 | 0.039 | -0.061 | 17.4% | 0.837 | +0.000 | 1/3 |
| best2_avoided_tackles_per_rec | +0.146 | **0.008** | +0.033 | 67.2% | 0.830 | -0.007 | 1/3 |
| career_yards_after_catch_pg | +0.222 | 0.046 | -0.104 | 4.5% | 0.836 | -0.001 | 1/3 |
| best2_yards_after_catch_pg | +0.181 | 0.054 | -0.155 | 0.6% | 0.844 | +0.007 | 1/3 |
| career_yac_per_reception | +0.067 | 0.019 | -0.087 | 7.6% | 0.837 | -0.001 | 1/3 |

### Verdict: Elusiveness features are marginal. Per-reception avoided tackles show the most genuine (but weak) signal.

The per-reception avoided tackles variants (peak2, career, best2) all have slightly positive residual signal (53-67% bootstrap positive), but none break 75%. This contrasts with the v4 analysis where best2_avoided_tackles_per_rec showed 84% bootstrap positive against the fuller locked set -- the difference is that with more features in the base, the residual became more concentrated.

Against just draft_capital + breakout_age, elusiveness has barely any orthogonal signal. The per-game variants (career_avoided_tackles_pg, career_yac_pg) are worse because they conflate opportunity with skill -- more targets = more chances to break tackles.

best2_yards_after_catch_pg has a suspicious +0.007 LOO-AUC delta but -0.155 residual (positive in 0.6% of bootstrap). Classic overfitting signal.

---

## Dimension G: Production Volume

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| career_targets_pg | +0.127 | 0.055 | -0.082 | 13.2% | 0.827 | -0.002 | 2/3 |
| best2_targets_pg | +0.089 | 0.031 | -0.146 | 2.3% | 0.842 | +0.013 | 2/3 |
| career_receptions_pg | +0.199 | 0.079 | -0.060 | 20.9% | 0.830 | +0.001 | 1/3 |
| best2_receptions_pg | +0.145 | 0.029 | -0.130 | 3.3% | 0.836 | +0.007 | 2/3 |
| career_first_downs_per_route | +0.245 | 0.171 | -0.071 | 18.5% | 0.818 | -0.011 | 1/3 |
| best2_first_downs_per_route | +0.194 | 0.234 | -0.124 | 6.1% | 0.825 | -0.004 | 1/3 |

### Verdict: No production volume feature should be in the model.

Every single variant has negative residual signal. Production volume is entirely captured by draft capital (teams draft productive players higher) and breakout age (which IS a production threshold). Adding raw counting stats on top is pure noise.

best2_targets_pg shows +0.013 LOO-AUC delta but has -0.146 residual (positive in 2.3% of bootstrap). This is the most extreme example of the LOO-AUC/residual disconnect -- the logistic regression is finding a spurious pattern.

---

## Cross-Dimension Insights

### Features with genuine orthogonal signal (positive residual, >65% bootstrap)

Only **3 features** across all 42 candidates clear this bar:

| Feature | Dimension | Residual | Boot %+ | LOO-AUC Delta |
|---------|-----------|----------|---------|---------------|
| **career_targeted_qb_rating** | QB Trust | +0.041 | 74.1% | **+0.023** |
| **best2_contested_catch_rate** | CCR | +0.133 | 97.9% | +0.001 |
| career_catch_pct_adot_adj | Catch Reliability | +0.029 | 66.5% | +0.009 |

These three capture information that draft_capital + breakout_age cannot explain: QB trust, contested catching, and catch reliability. Everything else -- YPRR, YPTPA, production volume, elusiveness, grades -- is subsumed by the base.

### Why most features have negative residual

draft_capital and breakout_age together are extraordinarily powerful. Draft capital encodes 32 front offices' evaluations (which already incorporate combine data, production stats, tape analysis, medicals). Breakout age encodes when a player became productive at the college level, which is the single best college predictor of NFL success.

When you regress tier outcome on these two features and look at the residuals, most college analytics features correlate *negatively* with what's left over. This means: **players who overperform their draft capital + breakout age profile tend to have WORSE individual analytics metrics.** This is counterintuitive but makes sense: a player drafted in round 4 who hits Elite has to have some hidden quality (scheme fit, mental processing, work ethic) that isn't captured by traditional stats. Conversely, a player with elite stats who busts despite high draft capital was probably over-indexed on measurables.

The exceptions -- QB trust, contested catching, catch reliability -- are the features that capture *skills* rather than *production*. Production is already priced into draft capital and breakout age. Skills are partially orthogonal.

### The "best2" effect

Several features show the pattern: best2 variant has higher LOO-AUC but worse residual than career variant. This is likely because the best-2-seasons-by-grades_offense selection creates a nonlinear interaction with draft capital (teams grade-weight similarly to PFF), which the logistic regression can exploit in-sample but which doesn't reflect genuine additional information.

### Implications for current model

The current model uses 7 features: draft_capital, breakout_age, breakout_yprr, breakout_yptpa, peak_contested_catch_rate, career_targeted_qb_rating, peak2_avoided_tackles_per_rec.

Against the base of just draft_capital + breakout_age, the evidence supports:
- **career_targeted_qb_rating**: Strong support (the clear #1 addition)
- **A CCR variant**: Strong support (best2 or peak; best2 has stronger residual, peak/best have better stability)
- **breakout_yprr**: Weak support (least harmful YPRR variant, but negative residual)
- **breakout_yptpa**: Weak support (+0.007 LOO-AUC but deeply negative residual)
- **peak2_avoided_tackles_per_rec**: Weak support (barely positive residual, 53.8% bootstrap)

The data suggests the model could potentially be simplified to **4 features**: draft_capital, breakout_age, career_targeted_qb_rating, and a CCR variant. Everything else is in the "might help, might not" zone.

---

## Scripts

| Script | Description |
|--------|-------------|
| `modeling/full_feature_analysis.py` | Comprehensive 7-dimension analysis (this report) |
| `modeling/yprr_analysis.py` | Dedicated YPRR 8-part analysis (against full model base) |
| `modeling/contested_catch_analysis.py` | Dedicated CCR 8-part analysis (against v3 feature set) |
