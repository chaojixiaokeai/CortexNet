# Changelog

All notable changes to this project are documented in this file.

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
