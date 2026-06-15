.PHONY: sync dev test lint fmt run-api ingest lock clean docker-build

# Install/refresh the locked environment (provisions Python 3.12 via uv).
sync:
	uv sync

# Alias: dev environment includes the `dev` dependency group by default.
dev: sync

test:
	uv run pytest

lint:
	uv run ruff check src api tests
	uv run mypy src

fmt:
	uv run ruff format src api tests
	uv run ruff check --fix src api tests

run-api:
	uv run uvicorn api.app:app --reload --port 8000

ingest:
	uv run vectorrag ingest data/raw

# Refresh the lockfile after editing dependencies in pyproject.toml.
lock:
	uv lock

# Export a pinned requirements.txt (only if a non-uv consumer needs it).
export-requirements:
	uv export --no-dev --no-emit-project --format requirements-txt -o requirements.txt

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .chroma build dist *.egg-info lambda_package

docker-build:
	docker build -t vectorrag:latest .
