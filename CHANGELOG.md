# Changelog

All notable changes to this project are documented in this file.

## 3.2.1 - 2026-02-27

### Added

- Standard package layout under `cortexnet/`
- Open-source docs: `README.md`, `ARCHITECTURE.md`, `MODULE_MAP.md`
- Governance docs: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- GitHub templates and workflows for CI and publishing

### Changed

- Unified public model name to `CortexNet`
- Reorganized scripts under `scripts/benchmarks`, `scripts/chat`, `scripts/eval`, `scripts/dev`
- Improved `pyproject.toml` metadata and packaging backend

### Fixed

- Cleaned unused imports/variables and stale script paths
- Aligned script-level run tests with current config defaults
- Ensured full test/build/install verification pipeline passes

