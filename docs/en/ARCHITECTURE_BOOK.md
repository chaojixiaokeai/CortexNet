# CortexNet Architecture Book

Version: 3.2.1  
Document type: end-to-end engineering architecture reference  
Audience: model engineers, platform engineers, contributors, maintainers

---

## 1. Scope and Purpose

This architecture book answers the following:

1. What are CortexNet's system boundaries and layers?
2. How does data flow from input to output during training and inference?
3. How do SSM, sparse attention, memory, routing, adapters, and device abstraction cooperate?
4. How is the codebase organized for open-source maintainability?
5. How should training and deployment be executed safely?

Out of scope:

1. Full mathematical derivations (see whitepaper).
2. Business-specific prompt engineering.
3. Deep integrations with external serving engines (for example vLLM/TensorRT-LLM).

---

## 2. Design Principles

1. Layered clarity: directory structure mirrors runtime architecture.
2. Install-and-run default: package works after standard installation.
3. Stable public API: canonical class is `CortexNet`.
4. Progressive runtime modes: Lite, Compatibility, and Full.
5. Verification-first: static checks, tests, packaging checks, CI gates.

---

## 3. Layered Architecture and Code Mapping

```text
Application Layer (scripts/*)
  - benchmarks: performance, coldstart, long context, quantization
  - chat: interactive runtime scripts
  - eval: capability evaluation scripts
  - dev: one-click and regression helpers

Core Library Layer (cortexnet/*)
  - model.py: canonical model and from_pretrained pipeline
  - blocks.py: core block composition and fusion
  - attention.py / ssm.py / memory.py / routing.py
  - capability modules: causal, graph, agent, adversarial, interpretability
  - adapter/*: model detection, config conversion, weight mapping, calibration
  - ops/*: device/runtime abstraction

Quality Layer (tests/* + .github/*)
  - unit + regression tests
  - CI / CodeQL / Dependabot
  - release and package quality gates
```

---

## 4. Main Data Flows

### 4.1 Training/Forward Flow

`input_ids -> embedding -> N x CortexBlock -> final norm -> lm_head -> logits`

Each block can include:

1. SSM path
2. selective sparse attention path
3. memory path
4. optional advanced paths
5. adaptive fusion
6. FFN or MoE

### 4.2 Generation Flow

1. Incremental cache support (`past_cache`).
2. Sampling controls: `top_k`, `top_p`, `temperature`, repetition penalty.
3. Optional SSM-only decode switch in Lite mode.
4. Optional lazy runtime warmup for coldstart experience.

### 4.3 Migration Flow (`from_pretrained`)

1. Detect source model type (`adapter/model_registry.py`).
2. Convert source config to `CortexNetConfig`.
3. Build `CortexNet` instance.
4. Map and load weights (`adapter/weight_adapter.py`).
5. Apply architecture adaptation (`adapter/arch_adapter.py`).
6. Optional lightweight calibration.
7. Resolve device and dtype through runtime abstraction.

---

## 5. Core Modules

### 5.1 SSM

- File: `cortexnet/ssm.py`
- Role: linear-complexity sequence modeling with incremental state support.

### 5.2 Selective Sparse Attention

- File: `cortexnet/attention.py`
- Role: preserve key token interactions with lower average compute.
- Features: Top-K, sliding window, RoPE, GQA, incremental cache.

### 5.3 Memory and Conditional Compute

- `memory.py`: synaptic-style memory.
- `hierarchical_memory.py`: working/episodic/semantic memory tiers.
- `routing.py`: MoE routing with capacity and balancing logic.

### 5.4 Advanced Capability Modules

1. causal reasoning (`causal_reasoning.py`)
2. graph reasoning (`graph_reasoning.py`)
3. self-evolution control (`self_evolution.py`)
4. multi-agent coordination (`multi_agent.py`)
5. adversarial robustness (`adversarial.py`)
6. interpretability monitoring (`interpretability.py`)

---

## 6. Runtime Modes

1. Lite: minimal path for deployment simplicity and lower overhead.
2. Compatibility: migration-oriented path for source model fidelity.
3. Full: richer capability path for research exploration.

Practical recommendation:

1. production/limited resources: Lite or Compatibility
2. architecture research: Full

---

## 7. Platform and Device Abstraction

- File: `cortexnet/ops/device_manager.py`
- Supported device families: `cpu/cuda/mps/npu/mlu`
- Capabilities: device selection, dtype normalization, fallback behavior

Value:

1. fewer script-level device branches
2. consistent behavior across heterogeneous environments

---

## 8. Testing and Quality Gates

### 8.1 Local checks

1. `make lint`
2. `make test-all`
3. `make check`

### 8.2 CI and security

1. CI workflow: `.github/workflows/ci.yml`
2. publish workflow: `.github/workflows/publish.yml`
3. CodeQL scan: `.github/workflows/codeql.yml`
4. dependency updates: `.github/dependabot.yml`

---

## 9. Extension Guide

Typical extension points:

1. new model family adapter: extend `ModelRegistry` and `WeightAdapter`
2. new device backend: extend `ops/device_manager.py`
3. new block path: extend `blocks.py` and fusion logic
4. new evaluation task: extend `scripts/eval/`

Guardrails:

1. keep `CortexNet` public API stable
2. include minimum regression tests for new logic
3. avoid uncontrolled global runtime state

---

## 10. Known Boundaries

1. Some third-party libraries may emit deprecation warnings.
2. Real-world throughput/latency must be benchmarked on target hardware.
3. Weight mapping quality depends on source naming conventions.

---

## 11. Maintenance Policy

For each release:

1. update `CHANGELOG.md`, `pyproject.toml`, and relevant docs
2. verify `make test-all && make check`
3. keep `MODULE_MAP.md` and docs synchronized with code changes
