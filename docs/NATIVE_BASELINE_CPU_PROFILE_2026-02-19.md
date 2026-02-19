# Baseline CPU profile (real-world path) — 2026-02-19

## Command

```bash
node --cpu-prof --cpu-prof-name=cpu-realworld-200.cpuprofile \
  node-harness/runScenario.mjs \
  --scenario micro_repro_sustain \
  --seed 23 \
  --steps 200 \
  --creatures 200 \
  --allowReproduction false \
  --maintainCreatureFloor false \
  --out /tmp/realworld-prof
```

## Top sampled functions

1. **20.83%** `lin_solve` — `js/classes/FluidField.js:443`
2. **19.17%** `_buildSparseDomainFromTiles` — `js/classes/FluidField.js:740`
3. **11.25%** `lin_solve` (second call-site sample cluster) — `js/classes/FluidField.js:443`
4. **7.08%** `pushSpanForRow` — `js/classes/FluidField.js:753`
5. **5.42%** `advect` — `js/classes/FluidField.js:564`
6. **3.33%** `project` — `js/classes/FluidField.js:524`
7. **2.92%** `_buildDeepEmptyTileMap` — `js/classes/FluidField.js:921`
8. **2.50%** `advectDensityRGB` — `js/classes/FluidField.js:620`
9. **2.08%** `step` — `js/classes/FluidField.js:938`

## Interpretation

For this baseline-real-world run, CPU time is dominated by **FluidField** operations, not rigid-soft narrowphase.

That means baseline-first native port order should prioritize:

1. `FluidField` hot kernels (`lin_solve`, `advect`, `project`)
2. Sparse-domain/tile-map build/update path
3. Then collision narrowphase/response and coupling loops

## Important scope note

This profile is from the **real-world baseline path**, which differs from the smaller runtime-solver attribution harness used earlier for GPU-only stage analysis.
