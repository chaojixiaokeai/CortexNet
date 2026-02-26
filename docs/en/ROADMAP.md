# CortexNet Roadmap and Versioning

Version: 3.2.1

---

## 1. Versioning policy

CortexNet follows semantic versioning: `MAJOR.MINOR.PATCH`.

1. `MAJOR`: breaking API changes
2. `MINOR`: backward-compatible feature additions
3. `PATCH`: bug fixes, tests, and documentation improvements

Canonical public model class remains `CortexNet`.

---

## 2. Completed milestones

1. package structure aligned with PyPI release requirements
2. canonical model naming unified as `CortexNet`
3. CI + CodeQL + Dependabot + publish workflow enabled
4. scripts reorganized by benchmark/chat/eval/dev categories
5. baseline governance docs and architecture docs established

---

## 3. Near-term plan (next 1-2 releases)

1. deepen bilingual documentation coverage
2. improve benchmark report standardization and visualization
3. expand migration examples and compatibility tests
4. improve `from_pretrained` error diagnostics

---

## 4. Mid-term plan (next 3-4 releases)

1. performance regression thresholds in CI
2. long-context cache and sparse-attention coordination improvements
3. standardized training templates (single node and distributed)
4. expanded backend validation matrix

---

## 5. Long-term direction

1. reproducible benchmark suite for community comparisons
2. plugin-style capability module interface
3. richer adapter/evaluator/deployment extension ecosystem
4. stronger governance with support lifecycle policy

---

## 6. Release quality gates

Required before release:

1. `make lint`
2. `make test-all`
3. `make check`
4. updated `CHANGELOG.md`

---

## 7. Backward compatibility commitments

1. keep `CortexNet` and `CortexNetConfig` stable by default
2. keep legacy aliases until explicit deprecation plan is announced
3. remove deprecated surface only in major releases

---

## 8. Community collaboration plan

1. maintain clear issue/PR templates
2. maintain `good first issue` and `help wanted` labels
3. encourage documentation and benchmarking contributions

---

## 9. Key risks

1. test matrix growth from multi-backend support
2. adapter rule complexity as model family coverage expands
3. benchmark interpretation drift without strict conventions

---

## 10. Maintenance principles

1. docs evolve with code
2. verification-first changes
3. release notes are source of truth for behavior changes
