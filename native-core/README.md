# native-core (C++ simulation core scaffold)

This folder is the incremental native migration target for hot simulation stages.

## Purpose

- Port bottleneck stages from JS to C++ bit-by-bit.
- Keep JS compiler/UI/orchestration intact.
- Validate behavior and speed stage-by-stage.

## Build (scaffold)

```bash
cd native-core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Artifacts

- `native_core` (static library): stage API implementation.
- `native_core_bench` (CLI): simple harness entrypoint.

## Current stage API (v0)

- `step_collision(...)`
- `step_body_fluid_coupling(...)`
- `step_integrate(...)`

All currently compile as no-op stubs with explicit TODO markers.

## Next

1. Wire fixture loader to consume node-harness SoA dumps.
2. Implement collision vertical slice first.
3. Add parity + perf output compatible with node-harness reports.
