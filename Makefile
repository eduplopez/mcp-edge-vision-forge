PYTHON ?= python
VENV ?= .venv
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff
MYPY ?= $(VENV)/bin/mypy
PIP ?= $(VENV)/bin/pip
IMAGE ?= mcp-edge-vision-forge:0.1.0

.DEFAULT_GOAL := help

.PHONY: help venv install dev test coverage lint format typecheck check build docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install       Create the virtual environment and install the project with dev extras"
	@echo "  test          Run the test suite"
	@echo "  coverage      Run tests with coverage reporting"
	@echo "  lint          Run ruff checks"
	@echo "  format        Format the codebase with ruff"
	@echo "  typecheck     Run mypy in strict mode"
	@echo "  check         Lint, typecheck and test"
	@echo "  build         Build the sdist and wheel"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-run    Run the MCP server in Docker"
	@echo "  clean         Remove caches and build artifacts"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

dev: install

test:
	$(PYTEST) -q

coverage:
	$(PYTEST) --cov=edge_forge --cov-report=term-missing --cov-report=xml

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

format:
	$(RUFF) format src tests
	$(RUFF) check src tests --fix

typecheck:
	$(MYPY) src

check: lint typecheck test

build: venv
	$(PIP) install build
	$(VENV)/bin/python -m build

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -it -p 8000:8000 \
		-v "$(CURDIR)/artifacts:/app/artifacts" \
		-v "$(CURDIR)/reports:/app/reports" \
		$(IMAGE)

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
