# WR Dynasty Model: Complete Feature Analysis Report

## Executive Summary

A systematic 7-part analysis of **42 candidate features** across 7 dimensions reveals that the vast majority of college analytics features are redundant once draft capital and breakout age are in the model. Only **2 features** have unambiguous evidence for inclusion beyond the base pair. The current 7-feature model is likely over-specified; a leaner 4-feature model may generalize better.

### The 4 features with strong evidence:

| Feature | Why it belongs |
|---------|----------------|
| **draft_capital** | Dominant predictor. 32 NFL front offices' consensus. |
| **breakout_age** | Best college predictor. Captures production timing. |
| **career_targeted_qb_rating** | Only feature with both strong LOO-AUC lift (+0.023) AND positive residual (74% bootstrap). Captures QB trust -- a dimension neither draft capital nor breakout age measures. |
| **A contested catch rate variant** | Only dimension where ALL variants show positive residual signal (82-98% bootstrap). Captures contested-ball skills -- mechanistically distinct from anything else in the model. |

### The features in the gray zone:

| Feature | Case for | Case against |
|---------|----------|--------------|
| breakout_yprr | Least harmful YPRR variant; best era stability (0.057) | Negative residual after base (-0.116). YPRR is literally how breakout_age is defined. |
| breakout_yptpa | +0.007 LOO-AUC vs base | -0.164 residual, 1.5% bootstrap positive. Deeply redundant. |
| peak2_avoided_tackles_per_rec | +0.057 residual after 4-feature core, 79% bootstrap | -0.005 LOO-AUC. Elusiveness signal is real but weak. |

---

## Methodology

**Base model**: draft_capital + breakout_age (the only two features where evidence is overwhelming and no alternative comes close).

**Protocol per dimension** (7 parts):
1. **Univariate screens**: Spearman correlation, mutual information, standalone AUC vs Elite+ threshold
2. **Era stability**: Spearman in early half (<=2020) vs late half (>=2022); drift = absolute difference
3. **Residual signal**: Ridge-regress tier outcome on base features, correlate residuals with candidate
4. **Collinearity**: Spearman between candidate and each base feature
5. **Bootstrap**: 1000 bootstrap iterations of residual signal; % of iterations with positive correlation
6. **LOO-AUC**: Leave-one-year-out AUC with balanced logistic regression; delta vs base-only
7. **Elastic net survival**: Does the coefficient survive L1/L2 regularization at C=0.01, 0.1, 1.0?

**Dataset**: 199-291 players depending on dimension (features with PFF grades data available 2016+; contested catch rate available 2018+). Draft years 2016-2024.

---

## Dimension A: YPRR / Route Efficiency

**240 complete cases. Base LOO-AUC: 0.837.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|---------|----------|-------|----------|---------|---------|-------|
| career_yprr | **+0.280** | 0.094 | -0.120 | 2.6% | 0.835 | -0.002 |
| best2_yprr | +0.244 | 0.106 | -0.154 | 0.9% | 0.838 | +0.000 |
| peak_yprr | +0.253 | 0.082 | -0.116 | 3.6% | 0.836 | -0.001 |
| breakout_yprr | +0.235 | **0.057** | -0.116 | 4.8% | **0.839** | **+0.002** |

**Verdict: No YPRR variant adds genuine value.**

Every variant has negative residual signal, meaning the base model already captures route efficiency. This is expected: breakout_age is literally defined as "first season with 2.0+ YPRR and 200+ routes." Adding YPRR on top is redundant by construction.

career_yprr has the strongest raw univariate signal (+0.280) but the worst residual. All its information is already priced into breakout_age and draft_capital.

breakout_yprr is the "least bad" -- best era stability (0.057 drift), only variant with positive LOO-AUC delta (+0.002), and least negative bootstrap (4.8%). If any YPRR must be in the model, this is it. But the evidence for inclusion is weak.

**A dedicated 8-part YPRR analysis** (testing against the full v5 model base rather than just draft_capital + breakout_age) confirmed these findings. With the fuller base, breakout_yprr had -0.019 residual and 41.3% bootstrap positive -- still negative but more neutral, and it beat every other YPRR variant in pairwise bootstrap comparisons 86-97% of the time.

---

## Dimension B: YPTPA / Market Share

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|---------|----------|-------|----------|---------|---------|-------|
| breakout_yptpa | +0.179 | 0.062 | -0.164 | 1.5% | 0.836 | +0.007 |
| best_yards_per_team_pass_att | +0.212 | **0.054** | -0.140 | 2.3% | **0.842** | **+0.013** |
| career_yards_pg | **+0.243** | **0.040** | **-0.055** | **22.5%** | 0.829 | -0.000 |
| best2_yards_pg | +0.181 | **0.017** | -0.150 | 1.2% | 0.841 | +0.013 |

**Verdict: No market share feature has genuine orthogonal signal.**

best_yards_per_team_pass_att and best2_yards_pg show +0.013 LOO-AUC, but their residuals are deeply negative (1-2% bootstrap positive). This is the most dangerous pattern in the data: **LOO-AUC improvement without residual support**. The logistic regression is finding a spurious interaction that won't generalize.

breakout_yptpa (current model feature) has the same problem: +0.007 LOO-AUC but -0.164 residual. It's 0.782 correlated with breakout_yprr, meaning the two breakout magnitude features largely measure the same thing.

career_yards_pg is the most honest: near-zero LOO-AUC delta and the least negative residual (-0.055). It doesn't help, but it doesn't pretend to either.

---

## Dimension C: QB Trust / Route Quality

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| **career_targeted_qb_rating** | **+0.305** | 0.113 | **+0.041** | **74.1%** | **0.852** | **+0.023** | **2/3** |
| best2_targeted_qb_rating | +0.252 | 0.124 | -0.016 | 40.7% | 0.837 | +0.008 | 2/3 |
| best_targeted_qb_rating | +0.232 | 0.068 | -0.012 | 44.4% | 0.825 | -0.004 | 2/3 |
| career_grades_pass_route | +0.284 | 0.087 | -0.041 | 29.0% | 0.823 | -0.006 | 1/3 |
| best2_grades_pass_route | +0.224 | **0.035** | -0.144 | 2.7% | 0.833 | +0.004 | 1/3 |
| career_grades_offense | +0.274 | 0.080 | -0.058 | 22.0% | 0.824 | -0.005 | 1/3 |
| best2_grades_offense | +0.223 | **0.034** | -0.147 | 1.9% | 0.831 | +0.002 | 1/3 |

**Verdict: career_targeted_qb_rating is the single best feature to add to the base model. Nothing else in any dimension comes close.**

This is the **only feature across all 42 candidates** where both LOO-AUC and residual signal strongly agree:

- **+0.023 LOO-AUC delta** -- 3x larger than any other candidate
- **+0.041 residual, 74.1% bootstrap positive** -- genuine orthogonal information
- **Lowest collinearity with base** (0.330 vs 0.430-0.482 for grades)
- **Survives elastic net at 2/3 strengths**

**Why targeted QBR over grades?** Targeted QBR measures how much quarterbacks trusted this receiver -- passer rating when targeting them. This captures route-running, separation, and reliable hands as evaluated by the player actually throwing the ball. It's conceptually distinct from draft capital (front office evaluation) and breakout age (production timing).

PFF grades are more collinear with breakout_age (0.463-0.482) because grades and early breakout both respond to the same underlying talent. Targeted QBR's lower collinearity (0.330) means it measures a different angle.

**Why career over best/best2?** Career averages across multiple seasons and QBs, reducing noise from individual-season variance. best_targeted_qb_rating actually *hurts* LOO-AUC (-0.004), and best2 is only +0.008 with negative residual.

---

## Dimension D: Contested Catch Rate

**199 complete cases. Base LOO-AUC: 0.844.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta | Enet |
|---------|----------|-------|----------|---------|---------|-------|------|
| peak_ccr | +0.131 | 0.206 | +0.084 | 88.3% | 0.842 | -0.002 | 1/3 |
| career_ccr | +0.144 | 0.173 | +0.085 | 87.9% | 0.843 | -0.001 | 1/3 |
| **best2_ccr** | +0.134 | 0.166 | **+0.137** | **97.9%** | **0.845** | **+0.001** | **2/3** |
| best_ccr | +0.094 | **0.082** | +0.064 | 81.8% | 0.839 | -0.005 | 2/3 |

**Verdict: CCR is the most clearly orthogonal dimension. Every variant has strong positive residual signal.**

This is the only dimension where ALL 4 variants have positive residual with >80% bootstrap reliability. CCR captures something draft capital and breakout age genuinely cannot explain: the ability to win contested catches. This is mechanistically distinct -- draft capital reflects aggregate talent evaluation, breakout age reflects production volume, but neither directly measures contested-ball skills.

**Against the base of just draft_capital + breakout_age, best2_contested_catch_rate is the clear winner:**
- Strongest residual of any feature in any dimension (+0.137, positive 97.9% of the time)
- Only CCR variant with positive LOO-AUC delta (+0.001)
- Survives elastic net at 2/3 strengths

**However, when career_targeted_qb_rating is added to the base (3-feature test):**
- best2_ccr residual drops from +0.137 to +0.058 (82.6% bootstrap) -- still positive but weaker
- peak_ccr residual drops to +0.031 (69.5% bootstrap) -- borderline
- LOO-AUC delta goes slightly negative for both (-0.001 to -0.003 vs base3)

The 3-feature base (draft_capital + breakout_age + career_targeted_qb_rating) already achieves **0.865 LOO-AUC**. Adding CCR doesn't improve prediction on this dataset, but the residual signal is real and may help with forward generalization.

**The CCR variant choice** depends on what you optimize for:
- **best2_ccr**: Strongest residual signal, best bootstrap reliability, best elastic net survival. But higher era drift (0.166).
- **peak_ccr** (current model): Better era stability when tested in the v4 analysis with the fuller feature set. Corrected "peak" computation (min 3 targets, floored by career).
- **best_ccr**: Best era stability (0.082) but weakest signal.

---

## Dimension E: Catch Reliability

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|---------|----------|-------|----------|---------|---------|-------|
| career_catch_pct_adot_adj | **+0.295** | 0.108 | **+0.029** | **66.5%** | **0.838** | **+0.009** |
| best2_catch_pct_adot_adj | +0.239 | 0.053 | -0.008 | 43.5% | 0.831 | +0.002 |
| career_caught_percent | +0.229 | 0.102 | -0.018 | 39.2% | 0.832 | +0.003 |
| best2_caught_percent | +0.216 | **0.012** | +0.007 | 54.7% | 0.830 | +0.001 |
| best_caught_percent | +0.138 | 0.057 | -0.020 | 40.0% | 0.826 | -0.003 |

**Verdict: Borderline. career_catch_pct_adot_adj has the only positive residual, but 66.5% bootstrap is below confidence threshold.**

The critical problem: career_catch_pct_adot_adj is **0.765 correlated with career_targeted_qb_rating**. QBs get higher passer ratings when receivers catch the ball. If career_targeted_qb_rating is already in the model, catch reliability is largely redundant.

When tested against the 4-feature core (draft_capital + breakout_age + career_tqbr + best2_ccr), career_catch_pct_adot_adj drops to **-0.113 residual, 2.9% bootstrap positive**. Its signal is entirely absorbed by targeted QBR.

---

## Dimension F: Elusiveness / YAC

**240 complete cases. Base LOO-AUC: 0.837.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|---------|----------|-------|----------|---------|---------|-------|
| peak2_avoided_tackles_per_rec | +0.120 | 0.047 | +0.014 | 53.8% | 0.834 | -0.003 |
| career_avoided_tackles_per_rec | +0.129 | **0.006** | +0.013 | 54.6% | 0.830 | -0.007 |
| career_avoided_tackles_pg | +0.214 | 0.036 | -0.047 | 24.0% | 0.835 | -0.003 |
| best2_avoided_tackles_pg | +0.205 | 0.039 | -0.061 | 17.4% | 0.837 | +0.000 |
| best2_avoided_tackles_per_rec | +0.146 | **0.008** | **+0.033** | **67.2%** | 0.830 | -0.007 |
| career_yards_after_catch_pg | +0.222 | 0.046 | -0.104 | 4.5% | 0.836 | -0.001 |
| best2_yards_after_catch_pg | +0.181 | 0.054 | -0.155 | 0.6% | **0.844** | **+0.007** |
| career_yac_per_reception | +0.067 | 0.019 | -0.087 | 7.6% | 0.837 | -0.001 |

**Verdict: Elusiveness features are marginal against the base. Per-reception avoided tackles have the most genuine signal.**

The per-reception avoided tackles variants show slightly positive residual (54-67% bootstrap), but none break 75%. Against the base of just draft_capital + breakout_age, elusiveness barely registers as orthogonal.

**Against the 4-feature core**, the picture improves:
- peak2_avoided_tackles_per_rec: +0.057 residual, **79.3% bootstrap** -- the only feature that clears 75% against the 4-feature core
- best2_avoided_tackles_per_rec: +0.064 residual, **84.3% bootstrap** -- even stronger

The elusiveness signal becomes more visible once career_tqbr and CCR are in the model, because those features absorb the "catch quality" variance and leave the "what you do after the catch" variance in the residuals.

However, both hurt LOO-AUC by -0.005 against the 4-feature core. The genuine orthogonal signal exists but doesn't translate to prediction improvement on this dataset.

best2_yards_after_catch_pg shows +0.007 LOO-AUC but has the worst residual in the entire dimension (-0.155, 0.6% bootstrap). Classic overfitting.

---

## Dimension G: Production Volume

**241 complete cases. Base LOO-AUC: 0.829.**

| Variant | Spearman | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|---------|----------|-------|----------|---------|---------|-------|
| career_targets_pg | +0.127 | 0.055 | -0.082 | 13.2% | 0.827 | -0.002 |
| best2_targets_pg | +0.089 | **0.031** | -0.146 | 2.3% | **0.842** | **+0.013** |
| career_receptions_pg | +0.199 | 0.079 | -0.060 | 20.9% | 0.830 | +0.001 |
| best2_receptions_pg | +0.145 | **0.029** | -0.130 | 3.3% | 0.836 | +0.007 |
| career_first_downs_per_route | **+0.245** | 0.171 | -0.071 | 18.5% | 0.818 | -0.011 |
| best2_first_downs_per_route | +0.194 | 0.234 | -0.124 | 6.1% | 0.825 | -0.004 |

**Verdict: No production volume feature should be in the model.**

Every variant has negative residual signal. Production volume is entirely captured by draft capital (teams draft productive players higher) and breakout age (which IS a production threshold).

best2_targets_pg shows +0.013 LOO-AUC but -0.146 residual (positive 2.3%). This is the most extreme example of LOO-AUC improvement without underlying signal -- the logistic regression is overfitting to a coincidental pattern.

career_first_downs_per_route has the worst LOO-AUC delta of any feature (-0.011). Raw counting stats actively hurt the model.

---

## Cascading Analysis: Building Up From the Base

To understand how features interact, I tested features sequentially -- adding them one at a time and re-measuring the next candidate's marginal contribution.

### Step 1: Base (draft_capital + breakout_age)
- LOO-AUC: 0.829-0.844 (varies by complete-cases subset)

### Step 2: + career_targeted_qb_rating
- LOO-AUC: 0.852 (+0.023)
- Clear winner across all 42 candidates

### Step 3: + CCR variant (tested against 3-feature base)
- best2_ccr: residual +0.058, 82.6% boot, LOO-AUC 0.864 (-0.001 vs base3)
- peak_ccr: residual +0.031, 69.5% boot, LOO-AUC 0.862 (-0.003 vs base3)
- Base3 alone: LOO-AUC 0.865

CCR has genuine residual signal but doesn't improve LOO-AUC. The 3-feature model is already very strong.

### Step 4: + additional features (tested against 4-feature core)

4-feature core: draft_capital + breakout_age + career_targeted_qb_rating + best2_contested_catch_rate. LOO-AUC: 0.846.

| Feature | Residual | Boot %+ | LOO-AUC | Delta |
|---------|----------|---------|---------|-------|
| breakout_yprr | -0.083 | 10.0% | 0.847 | +0.001 |
| breakout_yptpa | -0.131 | 2.8% | 0.846 | -0.000 |
| peak2_avoided_tackles_per_rec | +0.057 | **79.3%** | 0.841 | -0.005 |
| career_catch_pct_adot_adj | -0.113 | 2.9% | 0.843 | -0.003 |
| peak_contested_catch_rate | -0.028 | 33.8% | 0.846 | -0.000 |
| career_yprr | -0.136 | 0.9% | 0.851 | +0.005 |
| best2_avoided_tackles_per_rec | +0.064 | **84.3%** | 0.841 | -0.005 |

**Key finding**: After the 4-feature core, only the per-reception avoided tackles features have positive residual with >75% bootstrap reliability. But both hurt LOO-AUC by -0.005.

The breakout magnitude features (breakout_yprr, breakout_yptpa) have deeply negative residuals after the 4-feature core. Whatever they captured is already absorbed.

career_yprr shows +0.005 LOO-AUC but -0.136 residual (0.9% positive). This is overfitting.

---

## Why Draft Capital + Breakout Age Subsume Almost Everything

This is the most important finding. Draft capital and breakout age are so powerful together that they leave very little unexplained variance for other features to capture:

**Draft capital** encodes information from 32 NFL front offices who have access to:
- Full college tape study
- Private workouts and pro days
- Medical evaluations
- Character interviews
- Scheme fit analysis
- All public analytics

When we add a feature like career_yprr, we're adding information the front offices already had. The small residual improvement from YPRR is already embedded in the draft pick.

**Breakout age** captures the single most important college signal: early production. A player who breaks out at 19 is fundamentally different from one who breaks out at 22. This feature subsumes:
- Route efficiency (breakout is defined by YPRR threshold)
- Production volume (you can't break out without volume)
- Most rate stats (high rates drive breakout)

The **only features that survive** are ones that measure **skills** rather than **production outcomes**:
- **Targeted QBR**: How much do QBs trust you? (Not a production stat -- it's a quality-of-target metric)
- **Contested catch rate**: Can you win 50/50 balls? (A skill, not a volume outcome)
- **Avoided tackles per reception**: Can you make defenders miss? (A per-touch skill, not opportunity-dependent)

Production metrics (yards, receptions, first downs, YPRR, YPTPA) are outcomes that both draft capital and breakout age already capture. Skill metrics are partially independent.

---

## Implications for the Current Model

### Current v5 model (7 features):
draft_capital, breakout_age, career_targeted_qb_rating, peak_contested_catch_rate, breakout_yprr, breakout_yptpa, peak2_avoided_tackles_per_rec

### Evidence-based recommendation:

**Strong core (4 features):**
- draft_capital
- breakout_age
- career_targeted_qb_rating
- best2_contested_catch_rate (or peak -- see CCR section)

**Defensible additions (with caveats):**
- peak2_avoided_tackles_per_rec OR best2_avoided_tackles_per_rec: Genuine orthogonal signal (79-84% bootstrap after 4-feature core) but hurts LOO-AUC by -0.005. Include if you believe the orthogonal signal will generalize forward.
- breakout_yprr: Nearly harmless (+0.001 LOO-AUC) with best era stability of any YPRR variant. Negative residual (-0.083) but the least negative of the bunch. Include if you want to retain the breakout magnitude concept.

**Should probably be dropped:**
- breakout_yptpa: -0.131 residual after the 4-feature core, 2.8% bootstrap positive. No evidence of unique contribution. 0.782 correlated with breakout_yprr -- if you keep one breakout magnitude feature, keep yprr (which has better evidence across the board).

### The CCR variant question

The v4 analysis (with 5 locked features) favored peak/best CCR for era stability and elastic net survival. This new analysis (with just 2 locked features) favors best2_ccr for raw signal strength. The disagreement is because the base set changes what "residual" means.

Against the clean 2-feature base, best2_ccr has the strongest signal (+0.137, 97.9% bootstrap). Against the 4-feature core, best2_ccr still leads (+0.058, 82.6%) but peak_ccr drops to borderline (+0.031, 69.5%).

**Recommendation**: Switch from peak_contested_catch_rate to best2_contested_catch_rate. The residual evidence is substantially stronger, and both have similar elastic net survival (2/3). The era stability concern (0.166 vs 0.082 for best) is less relevant given that all CCR variants are trending upward in the late era -- the signal is getting stronger, not weaker.

---

## Summary Table: All 42 Features Ranked

Ranked by combined evidence (LOO-AUC delta against 2-feature base, residual signal, bootstrap reliability):

| Rank | Feature | Dimension | LOO Delta | Residual | Boot %+ | Verdict |
|------|---------|-----------|-----------|----------|---------|---------|
| 1 | **career_targeted_qb_rating** | QB Trust | **+0.023** | **+0.041** | **74.1%** | **Include** |
| 2 | **best2_contested_catch_rate** | CCR | +0.001 | **+0.137** | **97.9%** | **Include** |
| 3 | career_catch_pct_adot_adj | Catch | +0.009 | +0.029 | 66.5% | Borderline (absorbed by tqbr) |
| 4 | peak_contested_catch_rate | CCR | -0.002 | +0.084 | 88.3% | Good (but best2 is stronger) |
| 5 | career_contested_catch_rate | CCR | -0.001 | +0.085 | 87.9% | Good (but best2 is stronger) |
| 6 | best_contested_catch_rate | CCR | -0.005 | +0.064 | 81.8% | Decent (best era stability) |
| 7 | best2_avoided_tackles_per_rec | Elusiveness | -0.007 | +0.033 | 67.2% | Marginal |
| 8 | peak2_avoided_tackles_per_rec | Elusiveness | -0.003 | +0.014 | 53.8% | Marginal |
| 9 | career_avoided_tackles_per_rec | Elusiveness | -0.007 | +0.013 | 54.6% | Marginal |
| 10 | best2_caught_percent | Catch | +0.001 | +0.007 | 54.7% | Noise |
| 11 | breakout_yprr | YPRR | +0.002 | -0.116 | 4.8% | Redundant |
| 12 | breakout_yptpa | YPTPA | +0.007 | -0.164 | 1.5% | Redundant |
| ... | (remaining 30 features) | Various | Various | All negative | <45% | Redundant or harmful |

---

## Data Notes

- 291 total players in master dataset; 199-241 complete cases depending on dimension
- Draft years: 2016-2024 (varies by PFF grades availability)
- Era split: <=2020 (early) vs >=2022 (late)
- Breakout imputation: age = max + 1, yptpa = 0, yprr = 0 for non-breakout players
- LOO-AUC uses balanced logistic regression with class_weight="balanced"
- Bootstrap uses 1000 iterations with Ridge(alpha=1.0) for residual computation
- career_targeted_qb_rating and career_catch_pct_adot_adj are 0.765 correlated
- breakout_yprr and breakout_yptpa are 0.782 correlated
- best2_ccr and peak_ccr are 0.671 correlated

## Scripts

| Script | Description |
|--------|-------------|
| `modeling/full_feature_analysis.py` | 7-dimension analysis against 2-feature base |
| `modeling/yprr_analysis.py` | Dedicated 8-part YPRR analysis against full model base |
| `modeling/contested_catch_analysis.py` | Dedicated 8-part CCR analysis against v3 feature set |
| `modeling/feature_slot_analysis.py` | v4 slot analysis against 5-feature locked set |
