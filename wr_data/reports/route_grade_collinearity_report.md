# Career Route Grade: Collinearity Analysis

## Overview

`career_grades_pass_route` (PFF's route-running grade, career game-weighted average) has strong univariate signal — Spearman +0.296 with tier, AUC 0.682. On paper it looks like a top-5 feature. This report examines why it was excluded from the final model: its signal is almost entirely redundant with features already selected.

## Most Collinear Features

Spearman correlations between `career_grades_pass_route` and all other candidate features (N=241), sorted by absolute correlation:

| Feature | Spearman w/ Route Grade | Spearman w/ Tier |
|---------|------------------------|-----------------|
| **career_grades_offense** | **+0.981** | +0.277 |
| career_first_downs_per_route | +0.913 | +0.250 |
| **career_yprr** | **+0.847** | +0.292 |
| best_grades_offense | +0.805 | +0.218 |
| career_first_downs_pg | +0.804 | +0.210 |
| best_grades_pass_route | +0.799 | +0.228 |
| career_yards_pg | +0.794 | +0.246 |
| career_receptions_pg | +0.754 | +0.203 |
| career_targets_pg | +0.689 | +0.132 |
| career_touchdowns_pg | +0.678 | +0.256 |
| career_yards_after_catch_pg | +0.674 | +0.211 |
| **career_avoided_tackles_pg** | **+0.596** | +0.198 |
| best_yards_per_team_pass_att | +0.590 | +0.212 |
| **career_targeted_qb_rating** | **+0.488** | +0.306 |
| **career_caught_percent** | **+0.438** | +0.229 |
| **draft_capital** | **+0.352** | +0.529 |
| **career_contested_catch_rate** | **+0.145** | +0.145 |

Features in **bold** are in the final model. Route grade is meaningfully correlated with 5 of the 7 selected features (r > 0.35), and extremely correlated with YPRR (r = 0.847).

## Key Collinearity Relationships

### Route Grade vs Career YPRR (r = 0.847)

This is the critical overlap. PFF's route-running grade is heavily informed by the same outcomes that drive YPRR — yards gained on routes run. A receiver who generates high yards per route will almost always earn a high route grade. These features measure nearly the same thing from different angles (subjective PFF grade vs objective per-route production).

### Route Grade vs Career Grades Offense (r = 0.981)

These two are effectively the same feature. For wide receivers, PFF's overall offense grade is almost entirely determined by the route-running grade — blocking grades contribute minimally. Including both would be pure redundancy.

### Route Grade vs Targeted QBR (r = 0.488)

Moderate correlation. Both capture "how good is this receiver when the ball comes his way" but from different perspectives — route grade includes route-running quality regardless of targets, while QBR focuses on what happens when targeted. This is the pair with the most distinct information.

## Residual Analysis

The definitive test: after controlling for features already in the model, how much signal does route grade retain?

| Controlled For | Route Grade Residual Spearman w/ Tier |
|---------------|--------------------------------------|
| Nothing (raw) | +0.289 |
| career_targeted_qb_rating | +0.157 |
| career_yprr | +0.087 |
| **Both QBR + YPRR** | **+0.068** |

After removing just YPRR, route grade's correlation with tier drops from +0.289 to +0.087 — a 70% reduction. After removing both YPRR and targeted QBR, route grade retains only +0.068 of signal — essentially noise for a 211-player sample.

This means the information in route grade that predicts dynasty outcomes is almost entirely captured by the combination of YPRR and targeted QBR. The remaining features in the model (caught %, contested catch rate, avoided tackles, breakout age, draft capital) further absorb whatever scraps might remain.

## Why Not Replace YPRR With Route Grade?

Route grade and YPRR have similar univariate signal (Spearman +0.296 vs +0.292, AUC 0.682 vs 0.660). One might ask: why not use route grade instead of YPRR?

1. **YPRR is objective.** It's a ratio of two countable quantities (yards, routes). Route grade is a subjective PFF assessment that can vary by grader.
2. **Route grade fails multivariate selection.** It survives 0/3 elastic net regularization strengths and has zero permutation importance — the model finds no use for it once other features are present.
3. **YPRR has unique residual signal.** After controlling for route grade, YPRR retains meaningful residual correlation with tier. The reverse is not true — route grade after controlling for YPRR is near-zero.

## Validation Metrics Summary

| Metric | Route Grade | Notes |
|--------|------------|-------|
| Spearman w/ tier | +0.296 | Strong univariate — 5th overall |
| AUC (>=Elite) | 0.682 | Solid |
| Elastic net survival | 0/3 | Zeroed out at all regularization strengths |
| Permutation importance | 0.000 | No value in tree model with other features present |
| Era drift | 0.168 | Moderate stability |
| Residual after QBR + YPRR | +0.068 | Near-zero — redundant |

## Conclusion

`career_grades_pass_route` is a strong feature in isolation but is almost entirely redundant with `career_yprr` (r = 0.847) and `career_targeted_qb_rating` (r = 0.488). After controlling for both, only +0.068 residual correlation remains. Adding it to the model would introduce multicollinearity without meaningful incremental signal, increasing overfitting risk on a 200-player dataset. It is correctly excluded from the final feature set.
