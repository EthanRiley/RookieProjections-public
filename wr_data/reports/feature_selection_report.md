# WR Feature Validation & Selection Report (v4)

## Overview

This report documents the complete feature selection process for the WR dynasty tier prediction model. The process used a 5-layer validation pipeline on 99 candidate features across 199-291 players, an 8-part contested catch rate variant analysis, and a systematic slot-based comparison that tested every plausible alternative for each feature dimension.

The key insight from v4: **fewer features, more signal**. A rigorous slot analysis revealed that only 2 of the 4 "additional" feature slots (beyond the 5 locked features) carry genuine orthogonal information. The final model uses 7 features, down from 9 in v3.

## Validation Methods

| Layer | Method | Purpose |
|-------|--------|---------|
| 1 | Spearman correlation, mutual information, standalone AUC | Univariate signal detection |
| 2 | Violin plots by tier | Visual inspection for threshold effects and non-linearity |
| 3 | Elastic net at C=0.01, 0.1, 1.0 | Multivariate feature selection under regularization |
| 4 | XGBoost permutation importance | Non-linear interaction detection |
| 5 | Era stability (<=2019 vs >=2020) | Temporal robustness |
| 6 | 8-part head-to-head (contested catch rate) | Deep-dive variant comparison |
| 7 | Systematic slot analysis | Test all alternatives per feature dimension |

## Final Feature Set

**Full model (7 features):**
`draft_capital`, `career_targeted_qb_rating`, `breakout_age`, `best_contested_catch_rate`, `career_avoided_tackles_per_rec`, `breakout_yptpa`, `breakout_yprr`

**College-only model (6 features):** Same minus `draft_capital`

## Slot Analysis

The slot analysis treated 5 features as "locked" — features where the evidence is strong and no plausible alternative comes close — and then systematically tested every candidate for 4 additional feature dimensions.

### Locked Features (not contested)

| Feature | Spearman | AUC | Era Drift | Why Locked |
|---------|----------|-----|-----------|------------|
| **draft_capital** | 0.529 | 0.865 | 0.025 | Dominant on every metric. Only feature with meaningful permutation importance (0.066). Survives all 3 elastic net strengths. |
| **breakout_age** | -0.330 | 0.689 | 0.139 | Third-highest absolute Spearman. YPRR-based definition (2.0+ YPRR, 200+ routes). Captures when production happened, not just physical maturity. |
| **breakout_yprr** | 0.200 | 0.652 | 0.072 | Strong univariate signal with good era stability. Captures magnitude of breakout — how dominant the breakout season was on a per-route basis. |
| **breakout_yptpa** | 0.139 | 0.572 | 0.134 | Market share dimension of breakout magnitude. Partially orthogonal to breakout_yprr. Together with breakout_yprr, adds 15.6% to total feature set signal vs breakout_age alone. |
| **best_contested_catch_rate** | 0.094 | 0.556 | 0.161 | Won 8-part head-to-head analysis. Best era stability of all 3 variants (0.161 vs 0.266/0.320), lowest collinearity with model features (0.176 vs 0.294/0.314), only variant that survives elastic net (2/3 strengths). See detailed analysis below. |

### Baseline Performance

With locked features only: **LOO-AUC = 0.853**, total feature set signal = 0.492.

This is the bar every additional feature must clear. Adding a feature that hurts LOO-AUC isn't automatically disqualifying (the dataset is only 199 players, so a swing of 0.003 could be 1-2 players), but the feature must show reliable orthogonal signal to justify inclusion.

### Slot A: QB Confidence / Route Quality

**Winner: `career_targeted_qb_rating`**

| Candidate | Sp | AUC | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|-----------|-----|-----|-------|----------|---------|---------|-------|
| **career_targeted_qb_rating** | +0.330 | 0.793 | 0.087 | **+0.054** | **79%** | **0.856** | **+0.003** |
| best_targeted_qb_rating | +0.248 | 0.723 | 0.074 | -0.037 | 33% | 0.837 | -0.016 |
| best2_targeted_qb_rating | +0.262 | 0.740 | 0.109 | -0.014 | 47% | 0.844 | -0.010 |
| career_grades_pass_route | +0.293 | 0.680 | 0.111 | +0.019 | 64% | 0.846 | -0.007 |
| career_grades_offense | +0.284 | 0.686 | 0.100 | +0.004 | 55% | 0.846 | -0.007 |
| best2_grades_pass_route | +0.199 | 0.613 | 0.092 | -0.080 | 17% | 0.848 | -0.005 |

**Why career_targeted_qb_rating:** The only candidate in any slot that actually beats the locked-only baseline on LOO-AUC (+0.003). Strongest univariate signal of any college feature (Sp=0.330, AUC=0.793). Residual of +0.054 is reliably positive (79% of bootstrap samples). Survives elastic net at 2/3 strengths. Measures something distinct from the locked set: how much quarterbacks trust this receiver, which encodes route-running quality, separation ability, and hands reliability as evaluated by the player throwing them the ball.

**Why not the grades alternatives:** All grades features have negative or near-zero residual signal after the locked set. They're highly correlated with the breakout features (max collinearity 0.53-0.58 vs 0.34 for targeted QBR), meaning the locked set already captures what grades measure.

### Slot B: Route Efficiency

**Winner: NONE**

| Candidate | Sp | AUC | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|-----------|-----|-----|-------|----------|---------|---------|-------|
| career_yprr | +0.299 | 0.701 | 0.144 | -0.014 | 45% | 0.826 | -0.027 |
| best2_yprr | +0.245 | 0.678 | 0.169 | -0.052 | 29% | 0.832 | -0.022 |
| career_yards_pg | +0.237 | 0.643 | 0.055 | +0.021 | 62% | 0.845 | -0.008 |
| best2_yards_pg | +0.144 | 0.572 | 0.074 | -0.090 | 10% | 0.851 | -0.002 |
| best_yards_per_team_pass_att | +0.178 | 0.585 | 0.002 | -0.049 | 25% | 0.852 | -0.001 |
| career_first_downs_per_route | +0.240 | 0.663 | 0.233 | -0.017 | 42% | 0.841 | -0.013 |

**Why none:** Every route efficiency feature hurts LOO-AUC, most substantially. career_yprr — previously a core model feature — is the single worst addition at -0.027. The reason is clear: **breakout_yprr already captures route efficiency**. It measures YPRR at the breakout season, which is the most predictive slice of the career. Adding career_yprr on top is redundant (max collinearity 0.590 with the locked set) and introduces noise from early-career development seasons that dilute the signal.

This is the biggest change from v3: career_yprr is dropped entirely. The breakout features subsume it.

### Slot C: Catch Reliability

**Winner: NONE**

| Candidate | Sp | AUC | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|-----------|-----|-----|-------|----------|---------|---------|-------|
| career_catch_pct_adot_adj | +0.307 | 0.751 | 0.121 | +0.037 | 72% | 0.846 | -0.007 |
| career_caught_percent | +0.237 | 0.691 | 0.134 | -0.024 | 42% | 0.851 | -0.003 |
| best2_caught_percent | +0.218 | 0.682 | 0.010 | +0.017 | 64% | 0.845 | -0.009 |
| best2_catch_pct_adot_adj | +0.237 | 0.694 | 0.053 | +0.005 | 60% | 0.844 | -0.010 |
| best_caught_percent | +0.131 | 0.662 | 0.005 | -0.048 | 31% | 0.844 | -0.009 |

**Why none:** career_catch_pct_adot_adj has decent residual (+0.037, 72% positive) but still hurts LOO-AUC by -0.007. The 72% bootstrap positive rate doesn't clear the 75% threshold we'd want for confidence. More importantly, catch reliability is partially captured by career_targeted_qb_rating (QBs trust receivers who catch the ball) and best_contested_catch_rate (catching in traffic is the hardest version of catching). The marginal information from a dedicated catch % feature isn't enough to justify the added complexity.

career_caught_percent is the closest to harmless (LOO-AUC delta only -0.003) but its residual is actually negative (-0.024), meaning it's adding noise rather than signal after the locked features.

### Slot D: Elusiveness / YAC

**Winner: `best2_avoided_tackles_per_rec`**

| Candidate | Sp | AUC | Drift | Residual | Boot %+ | LOO-AUC | Delta |
|-----------|-----|-----|-------|----------|---------|---------|-------|
| **best2_avoided_tackles_per_rec** | +0.150 | 0.607 | 0.050 | **+0.069** | **84%** | 0.850 | -0.003 |
| career_avoided_tackles_per_rec | +0.133 | 0.613 | 0.029 | +0.041 | 70% | 0.848 | -0.006 |
| career_avoided_tackles_pg | +0.185 | 0.615 | 0.009 | +0.017 | 60% | 0.846 | -0.007 |
| best2_avoided_tackles_pg | +0.168 | 0.588 | 0.019 | +0.004 | 54% | 0.850 | -0.003 |
| best_avoided_tackles_pg | +0.142 | 0.602 | 0.049 | -0.003 | 49% | 0.847 | -0.007 |
| career_yards_after_catch_pg | +0.214 | 0.635 | 0.007 | -0.032 | 35% | 0.842 | -0.012 |

**Why best2_avoided_tackles_per_rec:** Strongest residual signal of any candidate in any slot (+0.069). Most reliably positive bootstrap distribution (84% — the highest of any non-locked candidate). The -0.003 LOO-AUC delta is within noise for 199 players (1-2 player swing). This feature captures something genuinely orthogonal: per-touch elusiveness, the ability to make defenders miss after the catch. None of the locked features measure this — they cover draft position, breakout timing/magnitude, contested catching, and QB trust, but not what a receiver does with the ball in space. The lowest collinearity with the locked set of any avoided tackles variant (0.214).

**Why not career_avoided_tackles_pg (the v3 pick):** career_avoided_tackles_pg was selected in prior versions for its exceptional era stability (drift=0.009). But the slot analysis reveals it has weak residual signal (+0.017, only 60% bootstrap positive) and a worse LOO-AUC delta (-0.007). The per-game normalization conflates opportunity with ability — a receiver with more targets per game gets more chances to break tackles. Per-reception normalization isolates the actual elusiveness skill.

**Why not YAC features:** All yards_after_catch features have negative residual signal. YAC is too correlated with the breakout magnitude features (breakout_yptpa captures scoring-relevant production which includes YAC).

### Cross-Slot Combination Results

256 combinations of top candidates (including "none" for each slot) were tested. Key finding: the combination analysis confirmed that most features hurt when added together. The top-performing combinations consistently used few additional features:

| Rank | Slot A | Slot B | Slot C | Slot D | LOO-AUC |
|------|--------|--------|--------|--------|---------|
| 1 | [none] | best2_yards_pg | career_caught% | [none] | 0.856 |
| 2 | [none] | best2_1d/rt | career_caught% | [none] | 0.856 |
| 7 | **[none]** | **[none]** | **[none]** | **[none]** | **0.853** |

The locked-only baseline (0.853) ranks 7th out of 256 combinations. Most combinations that beat it do so by <0.003 and use features that lack reliable residual signal. This confirms that the locked features carry the vast majority of predictive power.

## Contested Catch Rate: 8-Part Deep Dive

Three variants were compared in a dedicated analysis:

### Summary Table

| Metric | career | best2 | best |
|--------|--------|-------|------|
| Univariate Spearman | **+0.155** | +0.146 | +0.094 |
| Era drift | 0.320 | 0.266 | **0.161** |
| Elastic net survival | 1/3 | 0/3 | **2/3** |
| Max collinearity | 0.314 | 0.294 | **0.176** |
| Residual Spearman | +0.020 | **+0.079** | +0.038 |
| Bootstrap % positive | 62% | **89%** | 71% |
| Bootstrap: beats career | — | **90%** | 60% |
| Total feature set signal | 0.644 | **0.717** | 0.675 |
| LOO-AUC | **0.832** | 0.831 | 0.829 |

### Decision: `best_contested_catch_rate`

best2 wins on raw residual signal, but best wins on the metrics that matter for forward generalization:

- **Era stability (0.161):** Only variant that doesn't flip sign across eras. career (0.320) is one of the most era-unstable features in the entire candidate pool.
- **Elastic net survival (2/3):** Only variant a regularized model considers worth keeping. best2 gets zeroed out at all 3 strengths, suggesting its residual "signal" may be noise that doesn't survive penalization.
- **Independence (0.176):** Lowest correlation with other model features. career's high collinearity with catch_pct_adot_adj (0.314) explains why its residual drops to near-zero.
- **LOO-AUC (~tied):** All three are within 0.003, so predictive accuracy doesn't differentiate them.

The single best season captures peak contested-catch ability without dilution from early development (career) or small-sample averaging artifacts (best2).

## Breakout Age Engineering

### YPRR-Based Definition

Breakout = first season with **2.0+ YPRR and 200+ routes**. Age computed as of September 1 of that season.

This replaced the prior YPTPA-based definition (1.4+ YPTPA, 8+ games). The two are statistically indistinguishable standalone, but YPRR-based composes better with the magnitude features because the breakout threshold is on the same axis as breakout_yprr.

### Magnitude Features

| Feature | Description | Imputation |
|---------|-------------|------------|
| breakout_yptpa | Yards per team pass attempt at breakout season | 0 (never broke out = no market share) |
| breakout_yprr | YPRR at breakout season | 0 (never broke out = no efficiency) |
| breakout_age | Age at breakout | max + 1 (penalized for never breaking out) |

Together, the three breakout features capture *whether* a player broke out (age), *when* (age), and *how dominantly* (yptpa + yprr). Adding the magnitude features increases total feature set signal by 15.6% vs breakout_age alone.

## Key Changes from v3

| Change | v3 | v4 | Evidence |
|--------|-----|-----|---------|
| **Dropped career_yprr** | In model | **Removed** | Worst LOO-AUC delta of any candidate (-0.027). Redundant with breakout_yprr (collinearity 0.590). Every route efficiency feature hurts prediction. |
| **Dropped career_catch_pct_adot_adj** | In model | **Removed** | LOO-AUC delta -0.007. Bootstrap only 72% positive — below confidence threshold. Signal partially captured by career_targeted_qb_rating and best_contested_catch_rate. |
| **Switched avoided tackles** | career_avoided_tackles_pg | **best2_avoided_tackles_per_rec** | Residual +0.069 vs +0.017. Bootstrap 84% vs 60% positive. Per-reception normalization isolates elusiveness skill from opportunity. |
| **Total features** | 9 | **7** | Fewer features, less overfitting risk. Locked features carry the vast majority of predictive power. |

## Feature Comparison Notes

### Why career_yprr Was Dropped

career_yprr was a core feature in v1-v3 with strong univariate signal (Sp=0.299, AUC=0.701). But the slot analysis revealed it's the most damaging addition to the locked feature set (-0.027 LOO-AUC). The reason: breakout_yprr (YPRR at the breakout season) captures the same underlying signal — per-route efficiency — but at the most predictive moment in a prospect's career. Adding career-averaged YPRR on top introduces noise from early development seasons and creates collinearity (0.590) without adding information.

### Why career_catch_pct_adot_adj Was Dropped

aDOT-adjusted catch % has impressive univariate numbers (Sp=0.307, AUC=0.751) but weak residual signal after the locked features (+0.037, only 72% bootstrap positive). career_targeted_qb_rating already captures "does the QB trust this receiver" which subsumes much of what catch reliability measures. And best_contested_catch_rate captures the hardest version of catching. The marginal information from a dedicated catch metric isn't worth the complexity.

### Why best2_avoided_tackles_per_rec Over career_avoided_tackles_pg

Prior versions used career_avoided_tackles_pg for its exceptional era stability (drift=0.009). But the slot analysis shows its residual signal is weak (+0.017, 60% positive) — the locked features already capture most of what it measures. best2_avoided_tackles_per_rec has 4x the residual signal (+0.069, 84% positive) and decent stability (drift=0.050). The per-reception normalization is also more principled: it isolates elusiveness skill from target volume, which matters because the locked features (particularly draft_capital) already encode opportunity.

## Visualization Outputs

| File | Description |
|------|-------------|
| `wr_data/slot_analysis_loo_auc.png` | LOO-AUC per candidate vs locked-only baseline. Nearly all candidates fall below baseline. |
| `wr_data/slot_analysis_residual.png` | Bootstrap residual distributions. Only career_targeted_qb_rating (79%) and best2_avoided_tackles_per_rec (84%) show reliably positive signal. |
| `wr_data/slot_analysis_delta.png` | Waterfall chart of LOO-AUC deltas. 22 of 23 candidates are red (below baseline). |
| `wr_data/slot_analysis_radar.png` | Multi-metric radar profiles for top candidates per slot. |
| `wr_data/feature_violins.png` | Violin plots of feature distributions by tier outcome. |
| `wr_data/feature_evaluation.csv` | Full 5-layer evaluation table for all 99 candidate features. |

## Final Feature Set Summary

| # | Feature | Dimension | Spearman | Residual (after rest) | Why |
|---|---------|-----------|----------|----------------------|-----|
| 1 | draft_capital | Market price | +0.529 | — | 32 NFL front offices' consensus |
| 2 | breakout_age | Production timing | -0.330 | — | When did they become productive |
| 3 | breakout_yprr | Breakout magnitude (efficiency) | +0.200 | — | How dominant per route at breakout |
| 4 | breakout_yptpa | Breakout magnitude (market share) | +0.139 | — | How large a share of passing offense |
| 5 | best_contested_catch_rate | Contested catching | +0.094 | — | Peak ability to win in traffic |
| 6 | career_targeted_qb_rating | QB trust | +0.330 | +0.054 (79%+) | How much QBs trusted them |
| 7 | best2_avoided_tackles_per_rec | Elusiveness | +0.150 | +0.069 (84%+) | Per-touch ability to make defenders miss |

Each feature captures a distinct dimension of prospect evaluation. No two features are measuring the same thing. The model asks: Where was he drafted? How early did he break out? How dominant was the breakout? Can he catch in traffic? Do quarterbacks trust him? Can he make defenders miss?

## Data Notes

- 291 total players in dataset, 199 with all features complete
- Draft years: 2018-2024 (complete cases)
- Era split for stability: <=2020 (112 players) vs >=2022 (87 players)
- Breakout imputation: age = max + 1, yptpa = 0, yprr = 0 for players who never broke out
- Full feature evaluation table: `wr_data/feature_evaluation.csv`
- Contested catch analysis: `modeling/contested_catch_analysis.py`
- Slot analysis: `modeling/feature_slot_analysis.py`
- Slot visualizations: `viz/feature_slot_viz.py`
