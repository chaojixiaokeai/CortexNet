# Release Guide

## 1. Prepare

```bash
make install-dev
make lint
make test-all
make check
```

## 2. Bump Version

Update version in:

- `pyproject.toml`
- `cortexnet/__init__.py`
- `CHANGELOG.md`
- `CITATION.cff`

## 3. Commit and Push

```bash
git add pyproject.toml cortexnet/__init__.py CHANGELOG.md CITATION.cff
git commit -m "release: bump version to X.Y.Z"
git push origin main
```

## 4. Publish via GitHub Actions (Recommended)

Trigger `.github/workflows/publish.yml` with one of:

- `workflow_dispatch`
- GitHub Release (`release: published`)

PyPI publish workflow now tries Trusted Publishing (OIDC) first, then falls back to API token if configured.

## 5. Verify Publish

Check:

- GitHub Actions run status (`build` + `publish` jobs)
- https://pypi.org/project/cortexnet/ latest version and upload timestamps

## 6. Update Benchmark Artifact

Generate and commit one fixed smoke benchmark artifact for each release:

```bash
python scripts/benchmarks/benchmark_release_smoke.py \
  --device cpu \
  --dtype float32 \
  --output docs/reports/artifacts/vX.Y.Z_smoke.json
```

## 7. Required Secret

Repository secret:

- `PYPI_API_TOKEN`
- `HF_TOKEN` (optional, for `.github/workflows/publish_hf.yml`)

## 8. Trusted Publishing Setup (PyPI OIDC)

Open this URL while logged in as package owner:

- https://pypi.org/manage/project/cortexnet/settings/publishing/?provider=github&owner=chaojixiaokeai&repository=CortexNet&workflow_filename=publish.yml

After setup, publishing can run without API token.

## 9. Common Failure and Fix

`HTTP 400 File already exists` means that version has already been uploaded to PyPI.

Fix:

1. bump version and release again, or
2. keep `skip-existing: true` enabled for idempotent re-runs.
