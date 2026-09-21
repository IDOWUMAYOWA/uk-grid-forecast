.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-all check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies and git hooks
	uv sync --all-groups
	uv run pre-commit install

lint: ## Lint with ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Static type check with mypy (strict)
	uv run mypy src tests

test: ## Run unit tests (no network)
	uv run pytest -m "not integration"

test-all: ## Run all tests including integration tests hitting real APIs
	uv run pytest

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
