# CortexNet Open Source Collaboration Guide

Version: 3.2.1

---

## 1. Goal

This guide defines practical collaboration and release workflow for maintainers and contributors.

---

## 2. Repository conventions

1. `cortexnet/`: publishable library code
2. `scripts/`: runnable tooling scripts
3. `tests/`: test and regression code
4. `docs/`: bilingual documentation

---

## 3. Contribution workflow

Recommended process:

1. create a feature/fix branch
2. run local quality gates
3. open PR with context and evidence
4. merge only after CI green and review completed

Local gates:

```bash
make lint
make test-all
make check
```

---

## 4. Coding standards

1. prioritize readability and maintainability
2. include tests for new behavior
3. document user-facing API changes
4. keep canonical naming (`CortexNet`) for new code

---

## 5. Release workflow

### 5.1 Pre-release checklist

1. bump version in `pyproject.toml`
2. update `CHANGELOG.md`
3. ensure all checks pass

### 5.2 Publish

Publishing a GitHub Release triggers the PyPI publish workflow (requires `PYPI_API_TOKEN`).

---

## 6. Issue and PR quality bar

Issue should include:

1. clear reproduction steps
2. environment details
3. expected vs actual behavior

PR should include:

1. change summary
2. impact scope
3. test evidence
4. documentation updates

---

## 7. Security and licensing

1. security reporting process is in `SECURITY.md`
2. project is Apache-2.0 licensed
3. new dependencies must pass license/security review

---

## 8. Good first contribution areas

1. docs and example improvements
2. adapter rule extensions
3. benchmark report enhancements
4. low-risk bug fixes with tests

---

## 9. Maintainer operations

1. review dependency and security alerts weekly
2. triage stale issues regularly
3. run regression benchmark checks before release windows

---

## 10. Success metrics

1. PR CI pass rate
2. test coverage for new features
3. post-release regression incidence
4. documentation freshness
