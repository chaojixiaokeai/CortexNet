# Release Benchmark Reports

This folder stores fixed benchmark artifacts per release.

## Policy

1. Every released version should include at least one smoke benchmark artifact.
2. Artifact format: JSON + a short markdown summary.
3. The benchmark command should be reproducible from repository scripts.

## Standard Command

```bash
python scripts/benchmarks/benchmark_release_smoke.py \
  --device cpu \
  --dtype float32 \
  --iters 30 \
  --warmup 5 \
  --output docs/reports/artifacts/vX.Y.Z_smoke.json
```

## Reports

- [v3.2.4 Smoke Report](./v3.2.4_smoke.md)
