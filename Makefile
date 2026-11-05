# Makefile for SpectraRAG-Agent

.PHONY: help install test bench run-api run-cli run-demo docker-build docker-up clean

help:
	@echo "Available commands for SpectraRAG-Agent:"
	@echo "  make install       Install dependencies in active environment"
	@echo "  make test          Run full PyTest suite"
	@echo "  make bench         Execute benchmark evaluation suite (520 queries)"
	@echo "  make run-api       Start FastAPI REST server and web dashboard"
	@echo "  make run-cli       Launch interactive terminal REPL"
	@echo "  make run-demo      Run single-query demo execution"
	@echo "  make docker-build  Build Docker production container"
	@echo "  make docker-up     Start containerized service via docker-compose"
	@echo "  make clean         Remove build artifacts and caches"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -v

bench:
	python scripts/run_benchmarks.py --queries 520

run-api:
	python scripts/run_server.py

run-cli:
	python scripts/cli.py

run-demo:
	python scripts/demo_query.py

docker-build:
	docker build -t spectrarag-agent:latest .

docker-up:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
