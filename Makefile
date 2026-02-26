.PHONY: install install-dev lint test test-all build check clean precommit examples-smoke report-smoke

VERSION ?= $(shell python -c "import cortexnet; print(cortexnet.__version__)")

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e .[dev]
	python -m pip install pyflakes build twine pre-commit

lint:
	python -m pyflakes cortexnet tests scripts examples
	python -m compileall -q cortexnet scripts tests examples
	@if python -c "import ruff" >/dev/null 2>&1; then \
		python -m ruff check .; \
	else \
		echo "ruff not installed; skipping ruff check"; \
	fi

test:
	python -m pytest -q

test-all: test
	python scripts/dev/run_tests.py

examples-smoke:
	python examples/minimal_infer.py
	python examples/train_tiny.py --steps 2 --batch-size 1 --seq-len 16 --device cpu

report-smoke:
	python scripts/benchmarks/benchmark_release_smoke.py --device cpu --dtype float32 --iters 30 --warmup 5 --output docs/reports/artifacts/v$(VERSION)_smoke.json

build:
	python -m build

check: build
	python -m twine check dist/*

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	rm -rf build dist *.egg-info .pytest_cache

precommit:
	pre-commit run --all-files
