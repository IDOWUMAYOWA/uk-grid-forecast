# ADR 0001: Foundational tooling

**Status:** Accepted
**Date:** 2026-09-21

## Context

The project needs to be reproducible across Codespaces, CI and production
containers, and maintainable as it grows from ingestion to a deployed service.

## Decision

- **uv** for dependency and environment management, with a committed `uv.lock`
  so every environment resolves identical versions.
- **src layout** (`src/gridcast`) so tests import the installed package, not
  the working directory, catching packaging errors early.
- **Ruff** for linting and formatting (one fast tool instead of flake8 + black + isort),
  including bandit (`S`) and timezone (`DTZ`) rules. Timezone correctness is
  critical: GB settlement periods shift at BST/GMT clock changes, producing
  46- and 50-period days.
- **mypy --strict** to keep the codebase refactorable.
- **pydantic-settings** for typed config from environment variables (12-factor).
- **structlog** for structured logs; JSON in production.
- **Typer CLI** as the single entry point for every pipeline stage, so the same
  commands run locally, in CI and in the scheduler.

## Consequences

Contributors need uv installed (automated in the dev container). Strict typing
adds some friction for pandas-heavy code; `pandas-stubs` mitigates most of it.
