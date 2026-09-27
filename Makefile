.PHONY: install test lint type format format-check check

install:
	poetry install --with dev

test:
	poetry run pytest

lint:
	poetry run ruff check src tests

type:
	poetry run mypy

format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

format-check:
	poetry run ruff format --check .

check: lint format-check type test
