# GPU Lab (Isolated Fluid Harness)

This folder is intentionally isolated from the main sim runtime.

Goal: iterate on GPU fluid solver experiments without destabilizing the live `sim-server` loop.

## Scope (Milestone 1)
- Fluid-only harness (no creatures/particles)
- CPU vs GPU side-by-side metric run:
  - SPS
  - max/avg speed
  - dye spread footprint
- Fixed seeds + fixed initial conditions for repeatable comparisons

## Contract
- Do **not** modify `sim-server` execution path while iterating here.
- Keep interfaces small:
  - `cpuFluidStep(state, cfg)`
  - `gpuFluidStep(state, cfg)`
- Export comparable snapshots for both backends with identical schema.

## Next files
- `schema.mjs` - canonical state/buffer schema
- `cpu_ref.mjs` - CPU reference wrapper (ground truth)
- `gpu_stub.mjs` - GPU backend placeholder (to be filled)
- `run_compare.mjs` - benchmark + parity checks
