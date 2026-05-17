# Draft Capital Curve Analysis: Log vs Jimmy Johnson vs Sqrt

## Summary

We evaluated three draft capital transformations against dynasty fantasy outcomes for running backs (n=91 resolved, 2016-2021) and wide receivers (n=291 resolved):

1. **Sqrt (current):** `10 - 7 * sqrt(pick / 260)` — our v1 approach
2. **Log:** `10 - (10 / ln(261)) * ln(pick + 1)` — steeper decay, more separation at top
3. **Jimmy Johnson (normalized):** The classic NFL trade value chart (1990s), rescaled to 0-10

**Recommendation:** Switch from sqrt to **log** as the default draft capital transformation. It materially outperforms sqrt, closely approximates Jimmy Johnson's shape with a clean closed-form formula, and produces the flattest residuals across rounds.

---

## Key Findings

### 1. Sqrt is too flat — it doesn't differentiate enough at the top

The sqrt curve assigns 46% of its total value to R4+ picks for RBs. Actual dynasty value from R4+ is only 14%. This means the model treats a R4 pick as roughly similar to a R2 pick, when in reality R2 picks produce 8x more dynasty value per player.

| Curve | R1 share | R2 share | R3 share | R4+ share |
|-------|----------|----------|----------|-----------|
| **Actual DV** | **40%** | **24%** | **22%** | **14%** |
| Jimmy Johnson | 53% | 25% | 14% | 8% |
| Log | 30% | 22% | 21% | 27% |
| Sqrt | 16% | 17% | 20% | 46% |

### 2. Jimmy Johnson has the best R² with dynasty value, Log has the best R² with tier ordinal

| Curve | R²(dynasty value) | R²(tier ordinal) |
|-------|-------------------|-------------------|
| **RB** | | |
| Jimmy Johnson | **0.395** | 0.363 |
| Log | 0.384 | **0.393** |
| Sqrt | 0.305 | 0.351 |
| **WR** | | |
| Jimmy Johnson | 0.144 | 0.222 |
| Log | **0.147** | **0.236** |
| Sqrt | 0.133 | 0.221 |

JJ wins on raw dynasty value for RBs (+0.011 over log), but log wins on tier ordinal for both positions. Since our model predicts **ordinal tiers** (not raw dynasty value), R²(tier) is the more relevant metric.

### 3. Log produces flatter residuals — better calibration across rounds

When fitting a simple linear regression from draft capital to tier ordinal, the residuals by round tell us where each curve systematically over- or under-predicts:

| Round | JJ residual | Log residual |
|-------|-------------|--------------|
| R1 | -0.28 | -0.31 |
| R2 | **+0.68** | +0.33 |
| R3 | +0.34 | +0.08 |
| R4+ | -0.21 | **-0.04** |

JJ has a large positive residual in R2 (+0.68), meaning R2 RBs consistently outperform what JJ predicts. This is because JJ was designed for **trade value** (what teams pay), not **player outcomes** (what players produce). JJ overvalues R1 relative to R2-R3.

Log's residuals are much flatter — the largest is R2 at +0.33, and R3/R4+ are nearly zero. This means the log curve better captures the actual relationship between draft position and outcomes.

### 4. All curves are equivalent on rank-based metrics

Spearman correlation (0.561 for RB, 0.507 for WR) and AUC (0.795 for RB, 0.841 for WR) are identical across all three curves. This is expected — they're all monotone transforms of pick number, so rank ordering is preserved. The difference is entirely in **shape**, which affects parametric models.

### 5. JJ and Log are highly correlated (r=0.945)

Despite different origins, the two curves track closely. The main disagreement is in the top 5 picks (JJ is more aggressive) and R4+ (JJ drops to near-zero while log retains some value). For practical purposes, log is a smooth analytical approximation of the JJ shape.

---

## Why Not Jimmy Johnson?

JJ is tempting given its R²(dv) advantage, but:

1. **JJ is a lookup table, not a formula.** It requires interpolation between 74 discrete points. Log is a single closed-form expression.
2. **JJ was designed for trade value, not outcomes.** It reflects what teams *pay* for picks, which overweights R1 relative to actual production.
3. **JJ's R2 advantage disappears for the target we actually predict.** For tier ordinal (our model's target), log wins.
4. **JJ's R2-R3 residuals are large.** The curve systematically underpredicts mid-round outcomes, which is exactly where dynasty edge lives.

---

## Why Log Over Sqrt?

| Metric | Sqrt | Log | Delta |
|--------|------|-----|-------|
| R²(dv) RB | 0.305 | 0.384 | **+0.079** |
| R²(tier) RB | 0.351 | 0.393 | **+0.042** |
| R²(dv) WR | 0.133 | 0.147 | +0.014 |
| R²(tier) WR | 0.221 | 0.236 | +0.015 |
| R4+ residual | -0.21 | -0.04 | much flatter |
| R1 value share | 16% | 30% | closer to 40% actual |

Log improves R² substantially for RBs and modestly for WRs. The improvement is largest for the metric that matters most (tier ordinal) in the position we're actively building (RB).

The formula: `draft_capital = 10 - (10 / ln(261)) * ln(pick + 1)`

Scale: pick 1 = 8.75, pick 32 = 3.72, pick 64 = 2.50, pick 256 = 0.03.

---

## Impact on WR Model

Switching from sqrt to log for WRs would be a minor improvement (+0.015 R² on tier ordinal). The WR model is already validated and performing well on holdout (LogLoss 0.771, >=Elite AUC 0.961). A curve change would require full retraining and revalidation. **Recommend testing on WR holdout before committing to the switch for WR.**

---

## Visualization

See `rb_data/charts/draft_capital_curves.png` for the full 4-panel comparison:
- Panel 1: Curve shapes (pick → score)
- Panel 2: R² comparison (RB solid, WR hatched)
- Panel 3: Round-level value distribution vs actual dynasty value
- Panel 4: Residual analysis by round (JJ vs Log)
