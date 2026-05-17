#!/usr/bin/env python3
"""
RB Draft Capital Era Analysis.

Investigates whether draft capital has become more or less predictive
of RB dynasty outcomes over time. Splits 186 resolved RBs (2016-2024)
into three eras and compares hit rates, AUC, and Spearman correlations.

Key finding: No clear trend of increasing DC predictiveness. The main
story is Round 3's collapse — 57% Elite+ in 2016-2018 (Kamara, Hunt,
Conner) vs ~20% since. Late-round hit rates are constant at ~5%.

Outputs:
  - Console: full era breakdown tables
  - research/rb_model_debut/charts/rb_dc_era_*.png
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

DATA_DIR = os.path.join(PROJECT_ROOT, "rb_data")

# Load data
master = pd.read_csv(os.path.join(DATA_DIR, "rb_dynasty_value_with_college.csv"))
TIER_ORDER = {"Bust": 0, "Flex": 1, "Starter": 2, "Elite": 3, "Stud": 4, "League-Winner": 5}
master["tier_ordinal"] = master["computed_tier"].map(TIER_ORDER)
resolved = master[master["is_resolved"] == True].copy()
resolved["dc"] = resolved["pick"].apply(
    lambda p: max(10 - (10 / math.log(261)) * math.log(p + 1), 0)
)

ERAS = [
    ("2016-2018", [2016, 2017, 2018]),
    ("2019-2021", [2019, 2020, 2021]),
    ("2022-2024", [2022, 2023, 2024]),
]


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
# 1. Hit Rate by Round, by Era
# ============================================================
print_section("Elite+ Hit Rate by Round, by Era")
for era_name, years in ERAS:
    sub = resolved[resolved["draft_year"].isin(years)]
    print(f"\n{era_name} (n={len(sub)}):")
    for r in [1, 2, 3, 4, 5, 6, 7]:
        rsub = sub[sub["round"] == r]
        n = len(rsub)
        if n == 0:
            continue
        hits = (rsub["tier_ordinal"] >= 3).sum()
        pct = hits / n * 100
        print(f"  Rd{r}: {hits:2d}/{n:2d} = {pct:5.1f}%")


# ============================================================
# 2. DC AUC by Era
# ============================================================
print_section("Draft Capital AUC (>=Elite) by Era")
for era_name, years in ERAS:
    sub = resolved[resolved["draft_year"].isin(years)]
    y = (sub["tier_ordinal"] >= 3).astype(int)
    if 0 < y.sum() < len(y):
        auc = roc_auc_score(y, sub["dc"])
        print(f"{era_name}: n={len(sub):3d}, Elite+ rate={y.mean():.1%}, AUC={auc:.3f}")
    else:
        print(f"{era_name}: n={len(sub):3d}, all same class — AUC undefined")


# ============================================================
# 3. Spearman Correlation by Era
# ============================================================
print_section("Spearman Correlation (DC vs Tier) by Era")
for era_name, years in ERAS:
    sub = resolved[resolved["draft_year"].isin(years)]
    r, p = spearmanr(sub["dc"], sub["tier_ordinal"])
    print(f"{era_name}: Spearman={r:.3f}, p={p:.4f}, n={len(sub)}")


# ============================================================
# 4. Per-Year DC AUC
# ============================================================
print_section("Per-Year DC AUC (>=Elite)")
for yr in sorted(resolved["draft_year"].unique()):
    sub = resolved[resolved["draft_year"] == yr]
    y = (sub["tier_ordinal"] >= 3).astype(int)
    if 0 < y.sum() < len(y):
        auc = roc_auc_score(y, sub["dc"])
        print(f"{yr}: n={len(sub):2d}, Elite+={y.sum():2d}/{len(sub)}, AUC={auc:.3f}")
    else:
        print(f"{yr}: n={len(sub):2d}, all same class — AUC undefined")


# ============================================================
# 5. Round 3 Deep Dive
# ============================================================
print_section("Round 3 Breakdown by Era")
for era_name, years in ERAS:
    sub = resolved[(resolved["draft_year"].isin(years)) & (resolved["round"] == 3)]
    hits = (sub["tier_ordinal"] >= 3).sum()
    pct = hits / len(sub) * 100 if len(sub) > 0 else 0
    print(f"\n{era_name}: {hits}/{len(sub)} Elite+ ({pct:.1f}%)")
    for _, row in sub.sort_values("pick").iterrows():
        marker = " ***" if row["tier_ordinal"] >= 3 else ""
        print(f"    {row['name']:25s} pick {int(row['pick']):3d}  {row['computed_tier']}{marker}")


# ============================================================
# 6. Late-Round Hits
# ============================================================
print_section("Rd4-7 Elite+ Hits by Era")
for era_name, years in ERAS:
    sub = resolved[(resolved["draft_year"].isin(years)) & (resolved["round"] >= 4)]
    hits = sub[sub["tier_ordinal"] >= 3]
    print(f"\n{era_name}: {len(hits)}/{len(sub)} hits ({len(hits)/len(sub)*100:.1f}%)")
    for _, row in hits.iterrows():
        print(f"    {row['name']:25s} Rd{int(row['round'])} pick {int(row['pick']):3d}  {row['computed_tier']}")


# ============================================================
# 7. Rd1-2 vs Rd3 vs Rd4+ Gap
# ============================================================
print_section("Early vs Mid vs Late Round Hit Rates")
print(f"{'Era':<12s} {'Rd1-2':>12s} {'Rd3':>12s} {'Rd4+':>12s} {'Gap (1-2 vs 4+)':>16s}")
print("-" * 65)
for era_name, years in ERAS:
    sub = resolved[resolved["draft_year"].isin(years)]
    early = sub[sub["round"] <= 2]
    mid = sub[sub["round"] == 3]
    late = sub[sub["round"] >= 4]

    e_hit = (early["tier_ordinal"] >= 3).mean() * 100 if len(early) > 0 else 0
    m_hit = (mid["tier_ordinal"] >= 3).mean() * 100 if len(mid) > 0 else 0
    l_hit = (late["tier_ordinal"] >= 3).mean() * 100 if len(late) > 0 else 0
    gap = e_hit - l_hit

    print(f"{era_name:<12s} {e_hit:>5.0f}% (n={len(early):<2d}) {m_hit:>5.0f}% (n={len(mid):<2d}) {l_hit:>5.1f}% (n={len(late):<2d}) {gap:>12.0f}pp")


# ============================================================
# Summary
# ============================================================
print_section("SUMMARY")
print("""
Is draft capital becoming more predictive over time?

  DC AUC by era:   0.824 -> 0.871 -> 0.814  (no trend)
  Spearman by era: 0.513 -> 0.629 -> 0.397  (no trend, dropped recently)
  Rd4+ hit rate:   5.6%  -> 4.8%  -> 5.9%   (constant ~5%)

The answer is NO — draft capital has always been strongly predictive,
and there's no evidence it's becoming more so. The samples per era
(49-74 players) are too small to detect subtle shifts.

The interesting finding: Round 3 collapsed.
  2016-2018: 57% Elite+ (Kamara, Hunt, Conner, Drake)
  2019-2021: 20% Elite+ (Montgomery, Gibson)
  2022-2024: 22% Elite+ (Achane, Rachaad White)

The 2016-2018 Rd3 class was anomalous — 4 of 7 hit Elite+, including
a League-Winner (Kamara). Since then, Rd3 looks like Rd4.
""")
