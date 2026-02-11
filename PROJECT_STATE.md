# PROJECT_STATE.md

Last updated: 2026-02-11

## Current Baseline

- **Repo:** `browser_neurogenesis`
- **Main branch:** promoted to current refactor line (real-engine path only)
- **Primary goal:** run and test Rick's actual simulation code in both browser and Node without surrogate behavior

## Working Agreements (Rick)

1. **Real code path only** for testing/simulation behavior.
2. **Video-first reporting** (JSON is internal unless explicitly requested).
3. Keep browser interaction quality high (manual zoom/pan/fluid interaction should remain usable).
4. Prioritize behavior-preserving refactors.
5. **Documentation habit:** add docstrings/comments to every piece of code touched.

## Active Refactor Sequence (ordered)

1. Extract shared world-step core. ✅
2. Introduce world state container. ✅
3. Move UI side-effects out of core step. ✅
4. Unify initialization path. ✅
5. Continue RNG/context cleanup for deterministic behavior. ✅
6. Split config/runtime concerns safely. ✅
7. Standardize snapshot/query builder across browser + Node tooling. ✅
8. Add parity regression checks. ✅
9. Add cross-runtime save/load (browser + Node). ✅

## Latest Update (2026-02-11)

- Added browser **World Save / Load** controls (`Export World State`, `Import World State`).
- Extended node harness CLI with `save <path>` / `load <path>` for full state roundtrips.
- Added save/load regression script (`node-harness/saveLoadRegression.mjs`) for both headless-real and browser-like-real runners.
- Added mobile UX updates:
  - `Next Creature` mobile button for touch devices.
  - Disabled canvas/browser touch pan+pinch gestures to prevent mobile viewport lockups.
- Fixed UI import handler wiring bug (`config.handleImportConfig` → local `handleImportConfig`).

## Current Test Track

- Long-running ecosystem probes are being executed to evaluate:
  - interaction detection between mixed node-type creatures
  - equilibrium behavior under different parameter sets
  - stress/failure modes over larger step counts

## Reporting Format

For each scenario run:

- MP4 video artifact
- concise findings: expected vs unexpected behavior
- suggested parameter adjustments for improved equilibrium

## Growth System (Phase 1 started)

- Added first-pass **heritable growth genome** plumbing to `SoftBody`.
- Added probabilistic per-tick growth pass (`_attemptGrowthStep`) with:
  - weighted anchor-node type preference
  - weighted added-node type preference
  - weighted distance-band preference
  - weighted edge-type preference (rigid vs soft)
- Growth now adds nodes and connects each to the **closest pre-existing node**.
- Growth cost now depends on number of nodes/edges and edge length, scaled by global config.
- Added global growth controls in `js/config.js` (enable/cost/chance/cooldown/limits/mutation).
- Added mutation tracking key: `growthGenomeMutations`.

- Fixed runtime growth/brain integration bug from browser console (`NEURAL_INPUT_SIZE_BASE is not defined`) by routing `SoftBody.initializeBrain()` through `Brain.initialize()`.
- Updated NN resize behavior in `js/classes/Brain.js` so topology changes preserve existing weights/biases where dimensions overlap, and random-initialize only newly added slices.
- Added topology-change RL safety: experience buffer is flushed when NN input/output dimensions change due to growth, while overlapping NN weights are still preserved.
