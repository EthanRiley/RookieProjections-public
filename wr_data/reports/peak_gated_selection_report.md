# Peak-Gated Season Selection: Investigation Report

**Date**: 2026-05-13
**Script**: `modeling/research/test_peak_gated_selection.py`
**Quality Gate**: grades_offense >= 80.0

---

## Concept

The current `best1` selection picks the season with the highest PFF offensive grade.
This investigation tests a hybrid: **peak-gated** selection picks the season where
a specific stat peaks, but only from seasons with grades_offense >= 80. If no season
meets the quality gate, it falls back to the current best-by-grade selection.

The hypothesis: a receiver's best *catching* season may not be their best *overall grade*
season. A sophomore with a 82-grade season but elite catch metrics could be more
informative than their 88-grade junior season where they ran better routes but caught worse.

---

## Selection Divergence

For `catch_pct_adot_adj`, peak-gated selected a **different season** than best1 for
**15 of 237 players (6%)**. For the remaining 222 (94%),
the best-grade season was also the peak catch% season.

---

## 7-Part Analysis Results

### Full Model Base (4 anchors)

| Rank | Feature | Spearman | LOO Delta | Residual | Boot %+ | Collinearity | Era Drift |
|------|---------|----------|-----------|----------|---------|-------------|-----------|
| 1 | pg_catch_pct_adot_adj_graduated | +0.306 | +0.022 | -0.063 | 18.2% | 0.457 | 0.024 |
| 2 | pg_catch_minus_drops_graduated | +0.241 | +0.022 | -0.021 | 38.0% | 0.330 | 0.006 |
| 3 | best1_catch_pct_adot_adj_graduated | +0.294 | +0.017 | -0.050 | 23.0% | 0.436 | 0.028 |
| 4 | best1_cpaa_minus_drops_graduated | +0.240 | +0.016 | -0.028 | 34.2% | 0.410 | 0.095 |
| 5 | pg_clean_catch_rate_graduated | +0.225 | +0.015 | +0.038 | 68.9% | 0.300 | 0.121 |
| 6 | best1_catch_minus_drops_graduated | +0.202 | +0.015 | -0.031 | 32.4% | 0.340 | 0.033 |
| 7 | pg_catch_minus_drops | +0.196 | +0.014 | -0.032 | 32.3% | 0.313 | 0.010 |
| 8 | pg_clean_catch_rate | +0.200 | +0.014 | +0.027 | 63.4% | 0.282 | 0.099 |
| 9 | best1_clean_catch_rate_graduated | +0.181 | +0.013 | +0.042 | 69.1% | 0.302 | 0.029 |
| 10 | best1_clean_catch_rate | +0.155 | +0.010 | +0.025 | 61.2% | 0.285 | 0.005 |
| 11 | best1_catch_minus_drops | +0.154 | +0.009 | -0.046 | 26.5% | 0.334 | 0.053 |
| 12 | best1_cpaa_minus_drops | +0.199 | +0.007 | -0.043 | 27.6% | 0.410 | 0.107 |
| 13 | pg_catch_pct_adot_adj | +0.228 | +0.006 | -0.065 | 18.5% | 0.400 | 0.051 |
| 14 | career_targeted_qb_rating | +0.317 | +0.004 | -0.050 | 26.4% | 0.404 | 0.035 |
| 15 | best1_catch_pct_adot_adj | +0.218 | +0.003 | -0.052 | 23.4% | 0.408 | 0.038 |
| 16 | pg_yprr | +0.271 | +0.003 | -0.192 | 0.2% | 0.834 | 0.058 |
| 17 | peak_yprr | +0.249 | +0.003 | -0.162 | 0.1% | 0.774 | 0.097 |
| 18 | peak_catch_pct_adot_adj_graduated | +0.318 | +0.003 | -0.060 | 18.4% | 0.458 | 0.088 |
| 19 | peak_yprr_graduated | +0.356 | +0.002 | -0.181 | 0.0% | 0.927 | 0.168 |
| 20 | best1_yprr | +0.262 | +0.001 | -0.200 | 0.3% | 0.844 | 0.018 |
| 21 | peak_catch_pct_adot_adj | +0.246 | +0.001 | -0.039 | 28.4% | 0.379 | 0.037 |
| 22 | peak_catch_minus_drops | +0.187 | +0.000 | -0.032 | 31.4% | 0.252 | 0.041 |
| 23 | best1_yprr_graduated | +0.345 | -0.001 | -0.182 | 0.0% | 1.000 | 0.095 |
| 24 | peak_cpaa_minus_drops_graduated | +0.284 | -0.001 | -0.014 | 44.2% | 0.364 | 0.093 |
| 25 | pg_yprr_graduated | +0.363 | -0.002 | -0.172 | 0.0% | 0.986 | 0.150 |
| 26 | peak_catch_minus_drops_graduated | +0.236 | -0.002 | -0.023 | 37.7% | 0.273 | 0.074 |
| 27 | peak_cpaa_minus_drops | +0.243 | -0.002 | -0.021 | 40.2% | 0.359 | 0.045 |
| 28 | peak_clean_catch_rate_graduated | +0.191 | -0.003 | -0.041 | 28.7% | 0.298 | 0.109 |
| 29 | peak_clean_catch_rate | +0.155 | -0.003 | -0.049 | 25.8% | 0.279 | 0.072 |
| 30 | best2_catch_pct_adot_adj | +0.353 | -0.011 | -0.099 | 7.1% | 0.510 | 0.146 |

---

## Combination Results

| Configuration | LogLoss | Brier | >=Elite | >=Stud | >=Starter | # Feats |
|---------------|---------|-------|---------|--------|-----------|---------|
| v11 (QBR + catch%_adot_adj) | 2.347 | 0.515 | 0.842 | 0.778 | 0.833 | 6 |
| v11 minus QBR | 2.243 | 0.508 | 0.828 | 0.782 | 0.838 | 5 |
| 4 anchors only | 2.407 | 0.494 | 0.839 | 0.851 | 0.844 | 4 |
| QBR => best1_catch_pct_adot_adj | 2.142 | 0.521 | 0.838 | 0.768 | 0.834 | 6 |
| QBR => pg_catch_pct_adot_adj | 2.129 | 0.518 | 0.841 | 0.775 | 0.833 | 6 |
| QBR => best1_catch_pct_adot_adj_graduated | 1.972 | 0.499 | 0.861 | 0.773 | 0.852 | 6 |
| QBR => pg_catch_pct_adot_adj_graduated | 1.741 | 0.494 | 0.863 | 0.780 | 0.851 | 6 |
| QBR => peak_catch_pct_adot_adj_graduated | 2.241 | 0.512 | 0.832 | 0.771 | 0.829 | 6 |
| QBR => best1_cpaa_minus_drops | 2.553 | 0.531 | 0.837 | 0.773 | 0.829 | 6 |
| QBR => best1_cpaa_minus_drops_graduated | 2.287 | 0.521 | 0.853 | 0.778 | 0.835 | 6 |
| QBR => peak_cpaa_minus_drops_graduated | 2.418 | 0.514 | 0.836 | 0.779 | 0.827 | 6 |
| QBR => best1_catch_minus_drops_graduated | 2.341 | 0.512 | 0.843 | 0.777 | 0.830 | 6 |
| QBR => pg_catch_minus_drops_graduated | 2.000 | 0.507 | 0.848 | 0.783 | 0.832 | 6 |
| QBR => peak_catch_minus_drops_graduated | 2.346 | 0.517 | 0.826 | 0.773 | 0.826 | 6 |
| QBR => best1_clean_catch_rate | 2.256 | 0.511 | 0.839 | 0.776 | 0.828 | 6 |
| QBR => pg_clean_catch_rate | 2.304 | 0.506 | 0.843 | 0.781 | 0.831 | 6 |
| QBR => peak_clean_catch_rate | 2.529 | 0.514 | 0.826 | 0.779 | 0.829 | 6 |
| QBR => best1_clean_catch_rate_graduated | 2.219 | 0.496 | 0.843 | 0.780 | 0.838 | 6 |
| QBR => pg_clean_catch_rate_graduated | 1.982 | 0.493 | 0.846 | 0.784 | 0.838 | 6 |
| QBR => peak_clean_catch_rate_graduated | 2.643 | 0.514 | 0.824 | 0.776 | 0.829 | 6 |
| QBR => best1_yprr_graduated | 2.319 | 0.510 | 0.828 | 0.782 | 0.838 | 6 |
| QBR => pg_yprr_graduated | 2.378 | 0.512 | 0.829 | 0.781 | 0.838 | 6 |
| QBR => peak_yprr_graduated | 2.166 | 0.515 | 0.829 | 0.777 | 0.844 | 6 |
| catch%_adot => best1_catch_pct_adot_adj | 2.294 | 0.527 | 0.837 | 0.770 | 0.821 | 6 |
| catch%_adot => pg_catch_pct_adot_adj | 2.278 | 0.518 | 0.844 | 0.802 | 0.822 | 6 |
| catch%_adot => best1_catch_pct_adot_adj_graduated | 2.106 | 0.508 | 0.849 | 0.773 | 0.824 | 6 |
| catch%_adot => pg_catch_pct_adot_adj_graduated | 1.951 | 0.498 | 0.852 | 0.789 | 0.828 | 6 |
| catch%_adot => peak_catch_pct_adot_adj_graduated | 2.439 | 0.515 | 0.845 | 0.823 | 0.824 | 6 |
| catch%_adot => best1_cpaa_minus_drops | 2.400 | 0.530 | 0.845 | 0.776 | 0.821 | 6 |
| catch%_adot => best1_cpaa_minus_drops_graduated | 2.281 | 0.522 | 0.860 | 0.788 | 0.824 | 6 |
| catch%_adot => peak_cpaa_minus_drops_graduated | 2.378 | 0.516 | 0.847 | 0.821 | 0.824 | 6 |
| catch%_adot => best1_catch_minus_drops_graduated | 2.273 | 0.513 | 0.849 | 0.784 | 0.820 | 6 |
| catch%_adot => pg_catch_minus_drops_graduated | 2.008 | 0.506 | 0.854 | 0.820 | 0.819 | 6 |
| catch%_adot => peak_catch_minus_drops_graduated | 2.355 | 0.509 | 0.845 | 0.803 | 0.827 | 6 |
| catch%_adot => best1_clean_catch_rate | 2.501 | 0.515 | 0.845 | 0.776 | 0.821 | 6 |
| catch%_adot => pg_clean_catch_rate | 2.311 | 0.512 | 0.851 | 0.796 | 0.823 | 6 |
| catch%_adot => peak_clean_catch_rate | 2.545 | 0.512 | 0.848 | 0.787 | 0.835 | 6 |
| catch%_adot => best1_clean_catch_rate_graduated | 2.266 | 0.497 | 0.854 | 0.778 | 0.826 | 6 |
| catch%_adot => pg_clean_catch_rate_graduated | 2.113 | 0.501 | 0.855 | 0.796 | 0.827 | 6 |
| catch%_adot => peak_clean_catch_rate_graduated | 2.414 | 0.508 | 0.849 | 0.785 | 0.834 | 6 |
| catch%_adot => best1_yprr_graduated | 2.352 | 0.507 | 0.850 | 0.825 | 0.832 | 6 |
| catch%_adot => pg_yprr_graduated | 2.362 | 0.510 | 0.848 | 0.826 | 0.831 | 6 |
| catch%_adot => peak_yprr_graduated | 2.385 | 0.509 | 0.847 | 0.808 | 0.840 | 6 |
| both => best1_catch_pct_adot_adj | 2.206 | 0.528 | 0.832 | 0.763 | 0.827 | 5 |
| both => pg_catch_pct_adot_adj | 2.264 | 0.517 | 0.839 | 0.800 | 0.825 | 5 |
| both => best1_catch_pct_adot_adj_graduated | 2.118 | 0.504 | 0.851 | 0.772 | 0.831 | 5 |
| both => pg_catch_pct_adot_adj_graduated | 1.886 | 0.494 | 0.860 | 0.793 | 0.838 | 5 |
| both => peak_catch_pct_adot_adj_graduated | 2.363 | 0.506 | 0.842 | 0.823 | 0.828 | 5 |
| both => best1_cpaa_minus_drops | 2.509 | 0.530 | 0.843 | 0.777 | 0.823 | 5 |
| both => best1_cpaa_minus_drops_graduated | 2.301 | 0.520 | 0.856 | 0.786 | 0.832 | 5 |
| both => peak_cpaa_minus_drops_graduated | 2.426 | 0.509 | 0.843 | 0.826 | 0.828 | 5 |
| both => best1_catch_minus_drops_graduated | 2.062 | 0.510 | 0.845 | 0.782 | 0.827 | 5 |
| both => pg_catch_minus_drops_graduated | 1.915 | 0.502 | 0.850 | 0.830 | 0.828 | 5 |
| both => peak_catch_minus_drops_graduated | 2.207 | 0.505 | 0.834 | 0.816 | 0.831 | 5 |
| both => best1_clean_catch_rate | 2.356 | 0.510 | 0.847 | 0.777 | 0.830 | 5 |
| both => pg_clean_catch_rate | 2.268 | 0.505 | 0.852 | 0.806 | 0.832 | 5 |
| both => peak_clean_catch_rate | 2.412 | 0.506 | 0.836 | 0.795 | 0.835 | 5 |
| both => best1_clean_catch_rate_graduated | 2.049 | 0.488 | 0.850 | 0.783 | 0.836 | 5 |
| both => pg_clean_catch_rate_graduated | 1.943 | 0.491 | 0.854 | 0.801 | 0.837 | 5 |
| both => peak_clean_catch_rate_graduated | 2.535 | 0.503 | 0.835 | 0.793 | 0.837 | 5 |
| both => best1_yprr_graduated | 2.471 | 0.496 | 0.840 | 0.847 | 0.844 | 5 |
| both => pg_yprr_graduated | 2.491 | 0.500 | 0.837 | 0.849 | 0.843 | 5 |
| both => peak_yprr_graduated | 2.245 | 0.499 | 0.839 | 0.823 | 0.852 | 5 |

---

## Key Findings

**Best LogLoss**: QBR => pg_catch_pct_adot_adj_graduated (1.741 vs v11 2.347, delta -0.606)
**Best Brier**: both => best1_clean_catch_rate_graduated (0.488 vs v11 0.515, delta -0.027)
**Best Elite AUC**: QBR => pg_catch_pct_adot_adj_graduated (0.863 vs v11 0.842, delta +0.021)

---

## Visualizations

![Peak-Gated Selection Analysis](peak_gated_selection.png)

| File | Description |
|------|-------------|
| `wr_data/charts/peak_gated_selection.png` | 6-panel analysis: method comparison, combo results, diagnostics |