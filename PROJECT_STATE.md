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
- Added `node-harness/browserDefaultSoak.mjs` to reproduce browser-like failures headlessly using index-default startup conditions, with automatic crash snapshot/report output.

- Added growth guardrails + continuity telemetry pass:
  - population-based growth throttle (`soft`/`hard` limits)
  - size-based growth cost multiplier
  - per-creature telemetry counters for growth suppression reasons and RL topology resets
  - soak reports now include growth + RL continuity metrics at each checkpoint
- Added unit tests for all touched tuning/continuity code:
  - `node-harness/tests/growthControls.test.mjs`
  - `node-harness/tests/brainTopologyReset.test.mjs`
  - `node-harness/tests/softBodyGrowthGuardrails.test.mjs`
- Added reproduction stabilization controls beyond hard cap:
  - density-dependent fertility scaling (global + local crowding pressure)
  - resource-coupled reproduction gating + local nutrient/light debit per offspring
- Added reproduction-control unit tests:
  - `node-harness/tests/reproductionControls.test.mjs`
  - `node-harness/tests/softBodyReproductionControls.test.mjs`
- Extended browser-default soak harness with:
  - final-frame screenshot artifact output (PNG/PPM fallback)
  - optional progress log file output (`--logFile`) for detached runs
  - configurable runtime overrides via repeated `--set KEY=VALUE`
  - node diversity + growth cohort telemetry in checkpoints/reports
- Added unit-tested helper modules for metrics/overrides/snapshot shaping:
  - `js/engine/ecologyMetrics.mjs`
  - `js/engine/configOverride.mjs`
  - `node-harness/soakSnapshot.mjs`
- Shifted test focus from long integration soaks to deeper unit coverage + docs for complex shared-core paths.
- Added comprehensive unit tests for:
  - `js/engine/stepWorld.mjs` (population floors, reproduction gating, unstable-removal energy accounting)
  - `js/engine/worldPersistence.mjs` (config snapshot semantics, save/load round-trip with selection + spatial-grid rebuild)
- Added explanatory docstrings across complex lifecycle functions in `stepWorld.mjs` and `worldPersistence.mjs`.
- Began dedicated SoftBody + Brain hardening pass (unit tests + docstrings).
- Added Brain-focused unit coverage in `node-harness/tests/brainCore.test.mjs`:
  - brain-node selection precedence/fallback behavior
  - vector-size calculation from topology counters
  - fallback behavior when brain wiring is invalid
  - training interval trigger behavior
- Added SoftBody↔Brain bridge tests in `node-harness/tests/softBodyBrainBridge.test.mjs`:
  - initializeBrain delegation to existing brain instance
  - initializeBrain creation path when brain is missing
  - discounted reward helper behavior
  - blueprint radius monotonic update behavior
- Added docstrings/comments to complex Brain/SoftBody lifecycle helpers (`Brain._gatherBrainInputs`, action/train methods, and selected SoftBody geometry/RL helpers).
