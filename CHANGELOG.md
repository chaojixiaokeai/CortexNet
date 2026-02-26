# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Changed

- Removed hardcoded local absolute model paths in chat/dev scripts:
  - `scripts/chat/chat_qwen3_cortexnet.py`
  - `scripts/dev/one_click_cortexnet.py`
  Both now support `CORTEXNET_MODEL_PATH`.
- `examples/train_tiny.py` now sets seed via `--seed` and avoids tensor-to-scalar warning by using `loss.detach().item()`.
- `cortexnet/model.py::compile_model` now logs compile skip reason and supports strict mode via `CORTEXNET_COMPILE_STRICT=1`.
- Simplified optional data import block in `cortexnet/__init__.py` by removing redundant placeholders.

### Added

- Added repository hygiene checker:
  - `scripts/dev/check_repo_hygiene.py`
  - integrated into `make lint` and `.github/workflows/ci.yml`
  - prevents tracked cache files and hardcoded local absolute paths from entering repo.

## 3.2.5 - 2026-02-26

### Added

- Added Hugging Face publishing workflow: `.github/workflows/publish_hf.yml`.
- Added Hugging Face publishing script and templates:
  - `scripts/release/publish_hf_assets.py`
  - `deploy/hf_space/*`
  - `hf/model_card/README.md`
- Added bilingual Hugging Face publishing docs:
  - `docs/en/HF_PUBLISHING.md`
  - `docs/zh-CN/HF_PUBLISHING.md`

### Changed

- PyPI publish workflow now tries Trusted Publishing (OIDC) first and falls back to token.
- Release guide now documents Trusted Publishing setup URL and HF token usage.
- Added `examples-smoke` and `report-smoke` Make targets.

## 3.2.4 - 2026-02-26

### Added

- Added release smoke benchmark script: `scripts/benchmarks/benchmark_release_smoke.py`.
- Added fixed benchmark report hub and artifacts folder under `docs/reports/`.
- Added bilingual architecture entry and release report links in docs index.

### Changed

- README upgraded with bilingual landing section (Chinese + English quick entry).
- CI/Makefile now include `examples/` in static checks.
- CI test job now runs example smoke scripts.

## 3.2.3 - 2026-02-26

### Added

- Added `SUPPORT.md` for issue routing and maintainer response expectations.
- Added bilingual FAQ docs (`docs/zh-CN/FAQ.md`, `docs/en/FAQ.md`).
- Added bilingual architecture visual docs with Mermaid flows (`docs/zh-CN/ARCHITECTURE_VISUAL.md`, `docs/en/ARCHITECTURE_VISUAL.md`).
- Added runnable examples under `examples/` for minimal inference, migration loading, and tiny training.

### Changed

- Expanded docs index and root README navigation to include support and examples.
- Upgraded release guide with GitHub Actions-first publish flow and failure recovery notes.

## 3.2.2 - 2026-02-26

### Changed

- Release workflow now uses `skip-existing: true` to make manual/duplicate publish runs idempotent.
- CI compatibility improved for Python 3.9 type annotations and current legacy code style checks.

## 3.2.1 - 2026-02-27

### Added

- Standard package layout under `cortexnet/`
- Open-source docs: `README.md`, `ARCHITECTURE.md`, `MODULE_MAP.md`
- Full bilingual documentation hub under `docs/` (architecture book, whitepaper, quickstart, training/deployment, compatibility, benchmarking, roadmap, API reference, open-source guide)
- Governance docs: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- GitHub templates and workflows for CI and publishing
- Security/dependency automation: `codeql.yml`, `dependabot.yml`
- Developer productivity files: `.pre-commit-config.yaml`, `.editorconfig`, `Makefile`
- Public CLI entrypoint: `python -m cortexnet` / `cortexnet` command
- Public API contract tests: `tests/test_public_api.py`

### Changed

- Unified public model name to `CortexNet`
- Reorganized scripts under `scripts/benchmarks`, `scripts/chat`, `scripts/eval`, `scripts/dev`
- Improved `pyproject.toml` metadata and packaging backend

### Fixed

- Cleaned unused imports/variables and stale script paths
- Aligned script-level run tests with current config defaults
- Ensured full test/build/install verification pipeline passes
