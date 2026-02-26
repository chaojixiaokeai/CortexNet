.PHONY: install install-dev lint test test-all build check clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e .[dev]
	python -m pip install pyflakes build twine

lint:
	python -m pyflakes cortexnet tests scripts
	python -m compileall -q cortexnet scripts tests

test:
	python -m pytest -q

test-all: test
	python scripts/dev/run_tests.py

build:
	python -m build

check: build
	python -m twine check dist/*

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -empty -delete
	find . -name ".DS_Store" -delete
