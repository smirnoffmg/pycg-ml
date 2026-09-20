.PHONY: test functional-test lint

test:
	uv run pytest

# Upstream's gold standard: 119 snippets with expected call graphs.
functional-test:
	cd micro-benchmark && uv run --project .. python create_pytests.py
	cd micro-benchmark && uv run --project .. python -m pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/
