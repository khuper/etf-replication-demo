# Everything a contributor or a reviewer needs, in one place.
# `make check` is what CI runs; if it passes locally it passes there.

PYTHON ?= python3
RUN_DIR ?= outputs

.PHONY: help install install-dev lint format typecheck test test-fast check study figures verify clean

help:                     ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:                  ## Install the package
	$(PYTHON) -m pip install -e .

install-dev:              ## Install the package with development tooling
	$(PYTHON) -m pip install -e ".[dev]"

lint:                     ## Lint with ruff
	$(PYTHON) -m ruff check etflab tests
	$(PYTHON) -m ruff format --check etflab tests

format:                   ## Auto-format with ruff
	$(PYTHON) -m ruff format etflab tests
	$(PYTHON) -m ruff check --fix etflab tests

typecheck:                ## Type-check with mypy
	$(PYTHON) -m mypy

test:                     ## Run the full test suite
	$(PYTHON) -m pytest -q

test-fast:                ## Run everything except the simulation studies
	$(PYTHON) -m pytest -q -m "not slow"

check: lint typecheck test ## Everything CI runs

study:                    ## Run the shipped experiment end to end
	$(PYTHON) -m etflab study --output $(RUN_DIR)

figures:                  ## Regenerate the figures committed under docs/figures
	$(PYTHON) scripts/build_docs.py

verify:                   ## Re-run the most recent study and check the digest
	$(PYTHON) -m etflab verify $$($(PYTHON) -c "from etflab.registry import RunRegistry; print(RunRegistry('$(RUN_DIR)').latest().run_dir)")

clean:                    ## Remove build and run artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
