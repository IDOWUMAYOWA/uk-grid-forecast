#!/usr/bin/env bash
set -euo pipefail

echo ">>> Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo ">>> Syncing dependencies"
uv sync --all-groups

echo ">>> Installing pre-commit hooks"
uv run pre-commit install

[ -f .env ] || cp .env.example .env

echo ">>> Done. Try: make check"
