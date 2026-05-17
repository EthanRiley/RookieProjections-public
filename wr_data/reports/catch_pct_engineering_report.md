# Catch Percentage Feature Engineering Report

## Motivation

Career caught percentage (Spearman +0.229, AUC 0.662) is included in the model as a measure of hands reliability. A common objection is that catch rate reflects quarterback accuracy as much as receiver skill — a WR catching passes from a poor QB will have a deflated catch %, regardless of his hands.

We engineered three variants designed to isolate the receiver-dependent component of catch rate, then evaluated all of them against the raw metric.

## Engineered Variants

### 1. Catch % Above Team Completion %
**Definition:** WR catch % minus team completion % for that season.

**Rationale:** If a team completes 58% of all passes, and a WR catches 68% of his targets, the +10% delta is plausibly receiver-driven. This controls for QB accuracy and scheme at the team level.

**Result:** This was the weakest variant. Career Spearman dropped from +0.229 to +0.093; AUC dropped from 0.662 to 0.566. The team-relative adjustment *removed* signal rather than isolating it. This likely means that playing on a team with a high completion rate is itself predictive of NFL success — better programs produce better receivers, and controlling for that context throws away real information.

### 2. aDOT-Adjusted Catch % (Residual)
**Definition:** Catch % minus expected catch % given average depth of target, where expected catch % comes from a linear regression of catch % on aDOT across all player-seasons.

**Regression:** catch% = -1.73 * aDOT + 78.06 (i.e., each additional yard of depth costs ~1.7 percentage points of catch rate).

**Rationale:** Deep threats naturally have lower catch rates because deep balls are harder to complete. A receiver with a 62% catch rate on a 14-yard aDOT is more impressive than one with 68% on a 9-yard aDOT. The residual captures "hands above expectation given route tree depth."

**Result:** This was the strongest variant by a significant margin. Career Spearman improved from +0.229 to +0.295 (+29%); AUC improved from 0.662 to 0.725 (+10%). Adjusting for target depth meaningfully sharpens the signal. The raw metric penalizes deep threats and rewards slot receivers who run short routes — the aDOT adjustment corrects this bias.

### 3. Double-Adjusted (Team + aDOT)
**Definition:** Catch % adjusted for both team completion rate and aDOT simultaneously.

**Result:** Spearman +0.118, AUC 0.584. Worse than raw and far worse than aDOT-only. The team adjustment drags down the signal just as it did in variant #1. Combining two adjustments does not compound their value — the team adjustment is net-negative regardless of context.

## Full Comparison Table

| Feature | Spearman | AUC | Era Drift | Resid after QBR+YPRR | Resid after ALL |
|---------|----------|-----|-----------|----------------------|-----------------|
| career_caught_percent | +0.229 | 0.662 | 0.061 | +0.021 | -0.032 |
| best2_caught_percent | +0.213 | 0.656 | 0.008 | +0.060 | -0.005 |
| **career_catch_pct_adot_adj** | **+0.295** | **0.725** | 0.162 | +0.048 | +0.004 |
| best2_catch_pct_adot_adj | +0.230 | 0.702 | 0.117 | +0.053 | -0.022 |
| career_catch_pct_above_team | +0.093 | 0.566 | 0.078 | -0.059 | -0.059 |
| best2_catch_pct_above_team | +0.086 | 0.604 | 0.120 | -0.015 | -0.027 |
| career_catch_pct_double_adj | +0.118 | 0.584 | 0.042 | -0.073 | -0.031 |

## Residual Analysis

After controlling for all other model features (QBR, YPRR, contested catch rate, avoided tackles, breakout age, draft capital), residual signal is thin across all catch % variants:

- **career_catch_pct_adot_adj: +0.004** — the only variant that stays positive
- career_caught_percent: -0.032
- best2_caught_percent: -0.005
- All team-relative variants: negative

This confirms the finding from the original catch % report: the unique information in catch rate beyond the other model features is minimal. The value of including it comes from elastic net survival, era stability, and conceptual distinctness — not from a large independent signal.

The aDOT-adjusted version does marginally better here (+0.004 vs -0.032), consistent with it capturing a cleaner signal.

## Era Stability Tradeoff

The main cost of the aDOT adjustment is increased era drift:

| Feature | Era Drift |
|---------|-----------|
| career_caught_percent | 0.061 |
| best2_caught_percent | 0.008 |
| career_catch_pct_adot_adj | 0.162 |
| best2_catch_pct_adot_adj | 0.117 |

Raw career catch % has excellent stability (0.061). The aDOT-adjusted version drifts more (0.162), likely because the relationship between aDOT and catch rate has shifted as offensive schemes have evolved. The best-2-seasons aDOT-adjusted version (0.117) is moderately better on stability.

For context, the model's least stable features have drift around 0.275-0.318 (QBR, YPRR). A drift of 0.162 is elevated but not alarming.

## Addressing the QB Objection

The team-relative adjustment was designed to directly address the "catch rate is about the QB" concern. Its failure is informative: the data does not support the premise that controlling for quarterback/team passing quality improves catch rate's predictive value. This suggests one of two things:

1. **Catch rate is already more receiver-dependent than assumed.** Targets to a specific WR may be on a different difficulty distribution than the team's overall pass attempts — the QB throws differently to his best receiver than to his checkdown.
2. **Team context is itself informative.** Playing on a high-completion-rate team correlates with being in a well-coached program with good quarterback play, which correlates with NFL readiness. Removing that signal is counterproductive.

The aDOT adjustment, by contrast, *does* improve the signal. This tells us the main confound in catch rate is not QB quality but **route tree depth**. Deep threats look worse on catch rate not because they have bad hands, but because deep balls are inherently less catchable. Correcting for this is the right adjustment.

## Recommendation

**Replace `career_caught_percent` with `career_catch_pct_adot_adj` in the model.**

The case:
- +29% improvement in Spearman (+0.295 vs +0.229)
- +10% improvement in AUC (0.725 vs 0.662)
- Only positive residual after controlling all model features (+0.004 vs -0.032)
- Corrects a real confound (route depth) rather than removing real signal (team context)

The cost:
- Era drift increases from 0.061 to 0.162 — elevated but manageable, and well within the range of other model features

This is a meaningful upgrade. The aDOT adjustment produces a purer measure of hands reliability that doesn't penalize receivers for running deep routes, and that improvement shows up across every evaluation metric.
