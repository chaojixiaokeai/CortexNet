# CortexNet FAQ

## 1. Why does PyPI look unchanged after publishing?

Common reasons:

1. You re-published the same version (PyPI rejects file overwrite)
2. The project page cache has not refreshed yet

Always bump the version (for example `3.2.2 -> 3.2.3`).

## 2. What are the most common GitHub publish failures?

1. Missing or invalid `PYPI_API_TOKEN`
2. Re-uploading an existing version file
3. Build metadata issues before upload

## 3. Why does CI fail while local checks pass?

Check:

1. Python version differences (CI matrix includes 3.9/3.11/3.12)
2. lint rule differences
3. optional dependency behavior in CI runners

## 4. Why should new code use `CortexNet`?

`CortexNet` is the canonical public API. Legacy names remain compatibility aliases.

## 5. Fast install validation?

```bash
python -m cortexnet --version
python -m cortexnet --smoke-test
```

## 6. Why is first `from_pretrained` load slower?

Initial weight mapping and runtime initialization are expected to cost more. Use mapped cache/lazy warmup options when appropriate.

## 7. Minimal release steps?

1. bump version in `pyproject.toml` and `cortexnet/__init__.py`
2. update `CHANGELOG.md`
3. run `make check`
4. push and trigger `publish.yml`
