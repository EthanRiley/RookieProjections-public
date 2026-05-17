# Dynasty Rookie Draft Model — Design Document

## 1. Problem Statement

Build a machine learning system to assist dynasty fantasy football rookie drafting. Given a prospect entering the NFL, predict the probability that the player lands in each of the following outcome tiers:

- **League-Winner** — dynasty_value ≥ 350 (avg best 2 of 4 seasons above replacement, convex-transformed)
- **Stud** — dynasty_value ≥ 180
- **Elite** — dynasty_value ≥ 75
- **Starter** — dynasty_value ≥ 50
- **Flex** — dynasty_value > 0
- **Bust** — dynasty_value = 0

Tiers are computed from a **convex dynasty value metric**: for each player's first 4 NFL seasons (rookie contract), compute PPR points above WR36 replacement level per season, raise to power k=1.2 to capture the convex dynasty value curve, then average the best 2 of 4 seasons.

The output for each player is a full probability distribution across these tiers — **ordinal classification with calibrated probabilistic output**.

---

## 2. Design Principles

- **Small sample size** (~200 labeled WRs). Bayesian methods preferred. No SMOTE — class weights + proper scoring rules instead.
- **Ordinal target.** Tiers have natural ordering preserved via cumulative link models.
- **Temporal CV.** Leave-one-year-out on 2018–2021; holdout 2022–2024. No random k-fold (leaks across draft classes).
- **Draft capital as informed prior.** Log-scaled: `10 - (10 / ln(261)) * ln(pick + 1)`. The model learns the residual — what the analytical profile says *after* controlling for draft slot. That residual is where edge lives vs. the market.
- **Feature validation protocol.** 5-layer process (univariate screens, visual inspection, elastic net, permutation importance, era stability). All complete for WR. Results in `wr_data/feature_evaluation.csv` and `wr_data/reports/feature_selection_report.md`.

---

## 3. Project Structure

```
RookieProjections/
├── aggregation/              # Data pipeline (Python package)
│   ├── aggregate_college_stats.py   # WR: career/best-season college stats + graduated YPRR
│   ├── aggregate_rb_college_stats.py # RB: career/best/best2/peak/peak2
│   ├── build_dynasty_dataset.py     # Join dynasty values with college stats
│   ├── create_wr_aggregate_data.py  # Legacy WR join script
│   └── wr_dynasty_value.py          # Dynasty value metric + tier classification
├── modeling/                 # Production model scripts (Python package)
│   ├── base_model.py                # Position-agnostic shared infrastructure
│   ├── wr_model.py                  # WR-specific config + catch composite (delegates to base_model)
│   ├── rb_model.py                  # RB-specific config + composites (delegates to base_model)
│   ├── evaluate_holdout.py          # WR holdout evaluation (current, v12)
│   ├── evaluate_rb_holdout.py       # RB holdout evaluation (current, v2)
│   ├── predict_prospects.py         # WR: retrain on all data, predict 2024/2025/2026
│   ├── predict_rb_prospects.py      # RB: retrain on all data, predict 2024/2025/2026
│   ├── classify_players.py          # Interactive tier labeling tool
│   ├── research/                    # Research experiments & grid searches
│   └── archive/                     # Historical model scripts (v6-era standalone, v9, v11)
├── viz/                      # Production visualization (Python package)
│   ├── prospect_profile.py          # WR profile cards (single + --batch mode)
│   ├── rb_prospect_profile.py       # RB profile cards (with composite breakdown)
│   ├── generate_top10_pdfs.py       # Top-10 PDF tables + holdout profiles
│   ├── sophomore_profiles.py        # Sophomore lookahead profiles
│   ├── profiles/                    # Profile PNGs organized by year
│   └── research/                    # One-off viz scripts
├── scraping/                 # Data collection (nflverse, PFF)
├── wrangling/                # Data cleaning (fuzzy matching, grade joins)
├── threads/                  # Twitter threads & social media content
│   ├── cpaa/                        # CPAA feature thread + charts
│   ├── draft_capital/               # Draft capital curve thread
│   ├── ensemble/                    # Ensemble architecture thread
│   ├── rb_athleticism/              # RB athleticism thread
│   ├── rb_model_debut/              # RB model launch thread
│   ├── stribling/                   # Stribling case study thread
│   └── ...                          # Additional threads & responses
├── archive/                  # Historical exploration scripts
│   └── features/                    # Feature engineering & validation (Layers 1–5)
├── rb_data/                  # RB data files
│   ├── pff_rb_{year}.csv            # Raw PFF rushing grades (2014–2025)
│   ├── outputs/                     # Model predictions
│   ├── charts/                      # Research PNGs
│   └── reports/                     # Model reports
├── wr_data/                  # WR data files
│   ├── grades/                      # Raw PFF receiving grades (2016–2025)
│   ├── outputs/                     # Model predictions & intermediate CSVs
│   ├── charts/                      # Research PNGs
│   ├── reports/                     # Model reports
│   ├── pdfs/                        # Top-10 PDF tables
│   └── (core CSVs at top level: dynasty_value, draft_ages, etc.)
├── tests/                   # Unit tests (make test)
│   ├── test_base_model.py           # Shared infrastructure tests
│   ├── test_wr_model.py             # WR-specific tests
│   └── test_rb_model.py             # RB-specific tests
├── Makefile                  # Pipeline task runner (make help for targets)
├── retrain.py                # Legacy WR retrain pipeline
├── sync_public.sh            # Push filtered snapshot to public repo
├── pyproject.toml            # Package config (pip install -e .)
├── CLAUDE.md
└── requirements.txt
```

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # Installs as editable package — enables cross-module imports
make help                 # Show all available pipeline targets
```

### Repositories

- **Private** (`origin`): Full project including PFF data, threads, charts
- **Public** (`public`): Code + reports only (data/threads/charts excluded via `.public-exclude`)
- Run `make sync-public` (or `./sync_public.sh`) to push a filtered snapshot to public

---

## 4. Model Architecture

### Shared Base (`modeling/base_model.py`)

Position-agnostic infrastructure used by both WR and RB models:
- **Constants**: tier definitions, thresholds, cutpoints
- **Draft capital**: log-scaled `dc_log(pick)`
- **XGBoost cumulative link**: K-1 binary classifiers with Platt calibration + monotonicity enforcement
- **Bayesian ordinal regression** (PyMC): cumulative logit with ordered cutpoints, draft capital prior, MCMC posterior sampling
- **Ensemble blending**: weighted average with normalization (weights passed by position modules)
- **Evaluation**: `evaluate()`, `compute_metrics()` — log loss, Brier, per-threshold AUC
- **Output**: `build_pred_df()` — full + college-only predictions, edge, component breakdowns
- **Training pipeline**: `train_full_and_college()` — trains both DC+college and college-only variants

Position modules (`wr_model.py`, `rb_model.py`) provide position-specific constants (features, weights) and feature engineering (catch composite, RB composites), then delegate to the base. All shared names are re-exported for backward compatibility.

### WR (v12)

**Ensemble: 50% Bayesian + 50% XGBoost.** Grid-searched across 42 configurations. 50/50 optimizes equal-weighted composite of LogLoss + Brier + >=Elite AUC.

Both full (draft_capital + college) and college-only variants are produced. The "edge" column (E[college] - E[full]) shows where the analytical profile disagrees with draft capital.

---

## 5. Feature Set (WR) — v12

5 features, each measuring a distinct skill dimension:

| Feature | Dimension | Notes |
|---------|-----------|-------|
| `draft_capital` | NFL talent consensus | Log-scaled. Spearman 0.529, AUC 0.865. Full model only. |
| `pg_yprr_graduated` | Peak route efficiency (age-adjusted) | Peak-gated: max YPRR from seasons with grade >= 80, graduated age multiplier |
| `catch_composite` | Peak + career catching ability | z-avg(67% CPAA + 33% career aDOT-adj catch%). CPAA = peak-gated, age+aDOT adjusted catch%. |
| `best2_contested_catch_rate` | Contested ball skills | +0.137 residual, 97.9% bootstrap positive |
| `best2_avoided_tackles_per_rec` | Post-catch elusiveness | +0.066 residual, 84.2% bootstrap positive |

### Data Pipeline Filters (in `aggregation/aggregate_college_stats.py`)

1. **200-Route Minimum (v6).** Seasons need 200+ routes for best1/best2 selection.
2. **P5 School Filter (v7).** Non-P5 seasons excluded when P5 seasons available. P5 = SEC, Big Ten, Big 12, ACC, Notre Dame, Oregon State, Washington State.
3. **CCR Small-Sample Filter (v7).** <10 contested targets across best2 → fallback to 40%.
4. **Senior Season Discount (v8).** Age ≥ 22 on Sept 1: -10pp on CCR, caught_percent, targeted_qb_rating.
5. **Graduated Age Adjustment (v9/v12).** Per-age-class multiplicative on YPRR and catch%:
   - Freshman (< 19.5): +25% | Sophomore (19.5–20.5): +5% | Junior (20.5–21.5): -20% | Senior (21.5–22.5): -25% | Super Senior (≥ 22.5): -50%
   - Grid-searched across 500+ combinations. See `wr_data/reports/graduated_adjustment_report.md`.
7. **Birthdate Imputation.** Players missing PFR birthdates get birthdates back-calculated from `draft_age` and `draft_date` in `build_lookups()`.
6. **Peak-Gated Season Selection (v12).** For YPRR and catch%, select the season where the stat peaks among seasons with `grades_offense >= 80`. Falls back to best1 (highest grade) if no season meets the gate. See `wr_data/reports/v12_report.md`.

### Version History

| Version | Key Change | Impact |
|---------|-----------|--------|
| v6 | 5 features, 200-route minimum | Baseline |
| v7 | 6th feature (catch_pct_adot_adj), P5 filter, CCR filter | Better calibration |
| v8 | Senior season discount | >=Elite AUC 0.946→0.963, >=Stud AUC 0.856→0.888 |
| v9 | Graduated YPRR (`best1_yprr_graduated`) | LogLoss 0.799→0.771, >=Stud AUC 0.888→0.953 |
| v10 | Log-scaled draft capital (replacing sqrt) | >=Starter AUC 0.866→0.916, LogLoss 0.785→0.768 |
| v11 | 60/40 Bayesian/XGBoost ensemble (replacing 75/25) | >=Elite AUC 0.961→0.970, Brier 0.343→0.340 |
| v12 | Peak-gated features, 5F model, super senior -50%, 50/50 ensemble, catch_composite | LogLoss 0.745, Brier 0.324, >=Elite AUC 0.976 |

---

## 6. Current Results (v12)

**Holdout (88 players, 2022–2024):**
- LogLoss **0.745**, Brier **0.324**
- >=Elite AUC **0.976**, >=Stud AUC **0.953**, >=LW AUC **0.989**

**Prospect predictions** generated for 2024, 2025, and 2026 draft classes. Top 10 profile cards per class in `viz/profiles/{year}/`.

### Key Output Files

| File | Description |
|------|-------------|
| `wr_data/wr_dynasty_value_with_college.csv` | Master dataset: dynasty values + college stats + draft info |
| `wr_data/outputs/holdout_predictions_v12.csv` | Current holdout predictions (2022–2024) |
| `wr_data/outputs/prospect_predictions_{year}.csv` | Tier probability predictions per class |
| `wr_data/reports/*.md` | Model reports and research reports |
| `wr_data/pdfs/top10_{year}.pdf` | Top-10 PDF tables |
| `viz/profiles/{year}/*.png` | Prospect profile cards |

---

## 7. RB Model Architecture (v2)

**Ensemble: 45% Bayesian ordinal regression + 55% XGBoost cumulative link.**

Grid-searched at 5% intervals across 21 weights on holdout. 45/55 optimizes balanced calibration+discrimination (50/50 normalized composite of LogLoss+Brier vs Elite+Stud+Starter AUC).

### Feature Set (5 features)

| Feature | Dimension | Notes |
|---------|-----------|-------|
| `draft_capital` | NFL talent consensus | Log-scaled. Full model only. |
| `peak2_ypa` | Rushing efficiency | Weighted avg yards per attempt, best 2 eligible seasons |
| `composite_explosive` | Big-play ability | z-avg(career_explosive_per_att, best2_explosive_pg) |
| `composite_receiving` | Pass-catching skill | z-avg(career_rec_yards_pg, career_yprr, career_grades_pass_route) |
| `peak_yac_per_att` | Contact balance | Yards after contact per attempt, best single season |

### Composite Definitions

**composite_receiving** — z-avg of 3 features with low inter-correlation (rec_yards_pg vs YPRR: r=0.018):
- `career_rec_yards_pg` — receiving production volume
- `career_yprr` — yards per route run (efficiency)
- `career_grades_pass_route` — PFF route quality grade

**composite_explosive** — z-avg of 2 features (r=0.362):
- `career_explosive_per_att` — explosive play rate
- `best2_explosive_pg` — explosive plays per game (volume, best 2 seasons)

### RB Data Pipeline

- **Position module**: `modeling/rb_model.py` — RB constants, composites, fallbacks (delegates to `base_model.py`)
- **Aggregation**: `aggregation/aggregate_rb_college_stats.py` — career/best/best2/peak/peak2
- **Min attempts**: 100 per season for eligibility
- **Training data**: `rb_data/rb_dynasty_value_with_college.csv`
- **Holdout evaluation**: `modeling/evaluate_rb_holdout.py`
- **Prospect predictions**: `modeling/predict_rb_prospects.py`
- **Profile cards**: `viz/rb_prospect_profile.py` (includes composite breakdown panel)
- **Feature fallback**: `peak2_ypa` → `peak_ypa` when player has only 1 eligible season
- **Composite z-scores**: fit on training data only (holdout leakage fixed)

### RB Results (v2, leakage-fixed)

**Holdout (45 players, 2022–2024):**
- LogLoss **0.945**, Brier **0.415**
- >=Elite AUC **0.932**, >=Stud AUC **0.850**, >=Starter AUC **0.926**, >=LW AUC **1.000**

*Note: Previous metrics (LogLoss 0.790, Brier 0.348) were inflated by composite z-score leakage. Fixed 2025-05-15. Holdout expanded from 36→45 players via peak2_ypa fallback.*

### Key RB Output Files

| File | Description |
|------|-------------|
| `rb_data/rb_dynasty_value_with_college.csv` | Master dataset |
| `rb_data/outputs/holdout_predictions_rb_v1.csv` | Current holdout predictions (2022–2024) |
| `rb_data/outputs/prospect_predictions_rb_{year}.csv` | Prospect predictions per class |
| `rb_data/reports/rb_feature_grid_search_report.md` | 833-combo feature grid search report |
| `viz/profiles/rb/{year}/*.png` | RB prospect profile cards |

---

## 8. Testing

55 unit tests in `tests/` (run with `make test`):
- `test_base_model.py` — shared infrastructure: constants, draft capital, tier math, blending, metrics, output builder
- `test_wr_model.py` — WR constants, catch composite (build + apply), z-score leakage prevention, NaN handling
- `test_rb_model.py` — RB constants, feature fallbacks, composites (compute + apply), scaler leakage prevention, NaN handling

---

## 9. Open Questions

- **Opportunity modeling.** Landing spot / opportunity not included as features.
- **Other positions.** WR and RB modeled. TE, QB not started.
- **P5 filter is binary.** No gradient between SEC and AAC. (WR: full filter on season selection. RB: partial — P5 preferred for best/best2 selection only, not for career/peak/peak2 stats.)
- **Missing birthdates.** Imputation added via draft_age back-calculation, but some edge cases may remain.
- **Quality gate threshold.** `grades_offense >= 80` was chosen empirically but not grid-searched. Thresholds 70, 75, 85 could be tested.
- **RB athleticism.** Combine data (speed score, broad jump) showed real but modest signal (AUC ~0.62). Excluded from v2 due to 41% holdout coverage — revisit if coverage improves.

---

## 10. Decisions Summary

| Decision | WR (v12) | RB (v2) |
|----------|----------|---------|
| Target | Ordinal classification, 6 tiers | Same |
| Output | Full probability distribution across tiers | Same |
| Draft capital | Log-scaled: `10 - (10 / ln(261)) * ln(pick + 1)` | Same |
| Ensemble | 50% Bayesian + 50% XGBoost | 45% Bayesian + 55% XGBoost |
| CV | Leave-one-year-out 2018–2021, holdout 2022–2024 | Train 2016–2021, holdout 2022–2024 |
| Dynasty value | Best 2 of 4 seasons, above WR36, k=1.2 | Best 2 of 4 seasons, above RB24, k=1.2 |
| College features | pg_yprr_graduated, catch_composite, best2_contested_catch_rate, best2_avoided_tackles_per_rec | peak2_ypa, composite_explosive, composite_receiving, peak_yac_per_att |
| Age adjustment | Graduated (FR +25% to SSR -50%) on YPRR + catch% | Graduated (FR +25% to SR -25%) on rate stats; no super-senior bucket |
| Peak-gated | grades_offense >= 80; fallback to best1 | Computed but not used by model features; peak2/peak use min-attempts only |
| Feature fallbacks | None | peak2_ypa falls back to peak_ypa for single-season players |
| Min volume | 200 routes/season | 100 attempts/season |
| Predictions | 2024–2026; profiles in `viz/profiles/{year}/` | 2024–2026; profiles in `viz/profiles/rb/{year}/` |
