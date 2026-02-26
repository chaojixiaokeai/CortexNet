# Contributing to CortexNet

Thanks for contributing to CortexNet.

## Development Setup

```bash
git clone https://github.com/chaojixiaokeai/CortexNet.git
cd CortexNet
python -m pip install --upgrade pip
pip install -e .[dev]
pip install pyflakes build twine
```

## Quality Gates

Run these before opening a PR:

```bash
python -m pyflakes cortexnet tests scripts
python -m pytest -q
python scripts/dev/run_tests.py
python -m build
python -m twine check dist/*
```

Optional local hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Branches and Commits

- Create focused branches from `main`
- Keep commits atomic and message clear
- Avoid unrelated formatting-only churn

## Coding Guidelines

- Keep public API stable (`CortexNet` is the canonical model class)
- Add tests for behavior changes
- Preserve compatibility aliases unless there is an explicit breaking release
- Prefer clarity over cleverness in core model code

## Pull Request Expectations

- Explain what changed and why
- Include verification output
- Mention known limitations or follow-up work
