# aDOT-Adjusted Catch Percentage Investigation

## Question

Can `best2_catch_pct_adot_adj` (aDOT-adjusted completion percentage, best 2 seasons) replace or supplement `career_targeted_qb_rating` in the v6 feature set?

---

## Test 1: Swap (replace career_tqbr with catch%)

Three variants tested as a direct replacement for career_targeted_qb_rating in the 5-feature model:

| Metric | career_tqbr (v6) | career_cpaa | best2_cpaa |
|--------|:-:|:-:|:-:|
| >=Starter LOO-AUC | 0.699 | 0.790 | **0.801** |
| >=Elite LOO-AUC | 0.723 | 0.735 | **0.736** |
| >=Stud LOO-AUC | **0.646** | 0.596 | 0.687 |
| Residual Spearman (vs 4-feat base) | +0.025 | +0.013 | -0.041 |
| Bootstrap % positive | 62.9% | 54.7% | 28.7% |

### Bootstrap Head-to-Head (swap, 500 iterations)

| Matchup | Win Rate |
|---------|----------|
| career_tqbr vs career_cpaa | **55.6%** vs 42.6% (tqbr wins) |
| career_tqbr vs best2_cpaa | 42.8% vs **55.8%** (cpaa wins) |

### Holdout Evaluation (swap best2_cpaa for career_tqbr)

| Metric | v6 (career_tqbr) | swap (best2_cpaa) |
|--------|:-:|:-:|
| **Ensemble LogLoss** | **0.816** | 0.837 |
| **Ensemble Brier** | **0.363** | 0.371 |
| **>=Elite AUC** | **0.947** | 0.904 |
| **>=Stud AUC** | **0.844** | 0.759 |
| **>=LW AUC** | **0.931** | 0.718 |

**Verdict: Swap fails.** best2_cpaa won the LOO-AUC and bootstrap head-to-head but collapsed on holdout, especially at the top end. The negative residual signal (-0.041, only 28.7% bootstrap positive) was the correct warning: the LOO-AUC improvement came from XGBoost exploiting redundant structure that doesn't generalize.

### Why the Swap Fails

best2_cpaa correlates 0.484 with career_tqbr, 0.387 with best2_yprr, and 0.362 with best2_ccr. When it replaces career_tqbr, the model loses tqbr's genuinely orthogonal signal and gains a feature that mostly duplicates information already in the model. XGBoost can find interactions in-sample that exploit this redundancy, inflating LOO-AUC, but they don't hold on unseen data.

career_tqbr captures something mechanistically distinct: how well QBs trust the receiver (evidenced by passer rating when targeting them). Catch percentage after aDOT adjustment is more of a reliability metric that overlaps with route efficiency (YPRR) and contested catch rate.

---

## Test 2: Add (keep career_tqbr, add best2_cpaa as 6th feature)

| Metric | v6 (5 feat) | v6 + cpaa (6 feat) |
|--------|:-:|:-:|
| XGB >=Flex AUC | 0.832 | **0.841** |
| XGB >=Starter AUC | 0.864 | **0.869** |
| XGB >=Elite AUC | 0.900 | **0.917** |
| XGB >=Stud AUC | 0.635 | **0.691** |
| XGB >=LW AUC | 0.569 | **0.586** |
| XGB LogLoss | 1.184 | **0.790** |
| XGB Brier | 0.369 | **0.357** |
| LOO-AUC (>=Elite) | **0.726** | 0.725 |

### Residual Signal (best2_cpaa vs full 5-feature base)

| Metric | Value |
|--------|-------|
| Residual Spearman | **-0.081** |
| Bootstrap % positive | **15.0%** |
| Max collinearity | 0.484 (with career_tqbr) |

After partialing out career_tqbr, the remaining cpaa signal has Spearman +0.025 with tier residuals -- marginal at best.

### Bootstrap Head-to-Head (5-feat vs 6-feat, 500 iterations)

| Config | Win Rate |
|--------|----------|
| 5-feat (v6) | 43.8% |
| 6-feat (v6 + cpaa) | 52.6% |
| ties | 3.6% |

---

## Interpretation

**Adding best2_cpaa as a 6th feature improves XGBoost holdout across every threshold** -- sometimes substantially (>=Stud AUC 0.635 -> 0.691, LogLoss 1.184 -> 0.790). The bootstrap head-to-head also slightly favors 6 features (52.6% vs 43.8%).

**However, the residual analysis is deeply negative** (-0.081, only 15% bootstrap positive). This means the 6th feature adds zero orthogonal linear signal -- its value is entirely in XGBoost interaction structure. This is the same pattern that made the swap look good in LOO-AUC but fail on holdout.

**The key question: does adding a redundant-but-interaction-useful feature help or hurt the ensemble?** XGBoost clearly benefits from it. But the Bayesian ordinal model (75% of the ensemble) likely won't, since it's linear and the feature has negative residual. The net ensemble effect is uncertain.

**Risk assessment:**
- Adding a 6th feature to a 174-player training set increases overfitting risk
- The feature is the most collinear addition possible (r=0.484 with career_tqbr, r=0.387 with best2_yprr, r=0.362 with best2_ccr)
- v6's design philosophy was simplification (7 features -> 5) to reduce overfitting. Going to 6 contradicts this.
- The holdout is only 89 players -- XGBoost improvements on 89 players are noisy

---

## Recommendation

**Do not add.** The XGBoost holdout improvement is real but the feature has deeply negative residual signal against the 5-feature base. This is the same profile that made the swap look good in LOO-AUC but fail on holdout. The conservative choice -- maintaining 5 orthogonal features -- is more likely to generalize to future classes.

If the feature is added despite this, it should be XGBoost-only (not fed to the Bayesian model), which would require restructuring the ensemble pipeline.

---

## CCR Small-Sample Filter (implemented)

During this investigation, Kendrick Law (2025, pick 168) was found to have 100% contested catch rate from just 2 contested catches. A small-sample filter was added to `best2_stats()` in `aggregate_college_stats.py`:

- **Threshold**: 10 contested targets minimum across best2 seasons
- **Fallback**: Group average (45%) minus 5 percentage points = 40%
- **Scope**: Only applies to players who have CCR data but below the threshold. Players with no CCR data at all (pre-2018) remain NaN and are excluded from training.
- **Impact**: 20 players in the training set receive the fallback value. Training set size unchanged at 174.
