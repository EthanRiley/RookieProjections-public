PYTHON := python3
SHELL  := /bin/bash

# ──────────────────────────────────────────────────────────────
#  WR Pipeline
# ──────────────────────────────────────────────────────────────

.PHONY: aggregate-wr holdout-wr predict-wr profiles-wr retrain-wr

aggregate-wr:  ## Regenerate WR master dataset
	$(PYTHON) -m aggregation.aggregate_college_stats

holdout-wr:  ## Evaluate WR holdout (train 2018-2021, test 2022-2024)
	$(PYTHON) -m modeling.evaluate_holdout

predict-wr:  ## Predict WR prospects (retrain on all data, predict 2024-2026)
	$(PYTHON) -m modeling.predict_prospects

profiles-wr:  ## Generate WR profile cards (holdout + prospects)
	@for year in 2022 2023 2024 2025 2026; do \
		$(PYTHON) -m viz.prospect_profile --batch --year $$year --top 15; \
	done

retrain-wr: aggregate-wr holdout-wr predict-wr profiles-wr  ## Full WR pipeline

# ──────────────────────────────────────────────────────────────
#  RB Pipeline
# ──────────────────────────────────────────────────────────────

.PHONY: aggregate-rb holdout-rb predict-rb profiles-rb retrain-rb

aggregate-rb:  ## Regenerate RB master dataset
	$(PYTHON) -m aggregation.aggregate_rb_college_stats

holdout-rb:  ## Evaluate RB holdout (train 2016-2021, test 2022-2024)
	$(PYTHON) -m modeling.evaluate_rb_holdout

predict-rb:  ## Predict RB prospects (retrain on all data, predict 2024-2026)
	$(PYTHON) -m modeling.predict_rb_prospects

profiles-rb:  ## Generate RB profile cards (holdout + prospects)
	$(PYTHON) -m viz.rb_prospect_profile --batch --holdout
	@for year in 2024 2025 2026; do \
		$(PYTHON) -m viz.rb_prospect_profile --batch --year $$year; \
	done

retrain-rb: aggregate-rb holdout-rb predict-rb profiles-rb  ## Full RB pipeline

# ──────────────────────────────────────────────────────────────
#  Combined
# ──────────────────────────────────────────────────────────────

.PHONY: retrain retrain-fast aggregate holdout predict profiles

aggregate: aggregate-wr aggregate-rb  ## Regenerate all master datasets
holdout: holdout-wr holdout-rb  ## Evaluate all holdouts
predict: predict-wr predict-rb  ## Predict all prospects
profiles: profiles-wr profiles-rb  ## Generate all profile cards

retrain: retrain-wr retrain-rb  ## Full pipeline (WR + RB)
retrain-fast: aggregate holdout predict  ## Full pipeline without profiles

# ──────────────────────────────────────────────────────────────
#  Viz
# ──────────────────────────────────────────────────────────────

.PHONY: pdfs sophomore-profiles

pdfs:  ## Generate top-10 PDF tables
	$(PYTHON) -m viz.generate_top10_pdfs

sophomore-profiles:  ## Generate sophomore lookahead profiles
	$(PYTHON) -m viz.sophomore_profiles --top 15

# ──────────────────────────────────────────────────────────────
#  Git / Deployment
# ──────────────────────────────────────────────────────────────

.PHONY: sync-public

sync-public:  ## Push filtered snapshot to public repo
	./sync_public.sh

# ──────────────────────────────────────────────────────────────
#  Testing
# ──────────────────────────────────────────────────────────────

.PHONY: test

test:  ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

# ──────────────────────────────────────────────────────────────
#  Setup
# ──────────────────────────────────────────────────────────────

.PHONY: install

install:  ## Install package in editable mode
	pip install --no-deps -e .

# ──────────────────────────────────────────────────────────────
#  Help
# ──────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
