# Release Guide

## 1. Prepare

```bash
make install-dev
make lint
make test-all
```

## 2. Bump Version

Update version in:

- `pyproject.toml`
- `cortexnet/__init__.py`
- `CHANGELOG.md`
- `CITATION.cff`

## 3. Build and Validate

```bash
make check
```

## 4. Publish

```bash
python -m twine upload dist/*
```

For GitHub Actions publishing, set repository secret:

- `PYPI_API_TOKEN`

Then trigger `.github/workflows/publish.yml`.

