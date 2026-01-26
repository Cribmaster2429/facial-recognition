.PHONY: install install-dev test lint format clean run-api run-cli docker-build docker-up docker-down

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/facial_recognition --cov-report=html --cov-report=term

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run application
run-api:
	uvicorn facial_recognition.api.app:app --reload --host 0.0.0.0 --port 8000

run-cli:
	python -m facial_recognition --help

# Docker
docker-build:
	docker build -t facial-recognition .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
