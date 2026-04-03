.PHONY: help install test train clean lint format run-dev run-streamlit run-gradio

help:
	@echo "Myanmar LNP Dataset - Makefile Commands"
	@echo ""
	@echo "  install        Install dependencies"
	@echo "  test          Run tests"
	@echo "  train         Train model"
	@echo "  clean        Clean cache files"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  run-dev      Run development server"
	@echo "  run-streamlit Run Streamlit app"
	@echo "  run-gradio   Run Gradio app"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=api --cov-report=html

test-quick:
	pytest tests/ -v

train:
	python -m layers.cli.main_cli train data/processed/train.jsonl data/processed/test.jsonl

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf coverage.xml

lint:
	flake8 api/ layers/ tests/ --max-line-length=100 --ignore=E501

format:
	black api/ layers/ tests/ --line-length=100
	isort api/ layers/ tests/ --line-length=100

run-dev:
	streamlit run layers/streamlit/app.py

run-streamlit:
	streamlit run layers/streamlit/app.py --server.port 8501

run-gradio:
	python -m layers.gradio.app