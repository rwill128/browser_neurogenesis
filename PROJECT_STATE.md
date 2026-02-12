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
- Added direct browser UI support for new growth/reproduction tuning knobs:
  - new control panel section in `index.html` for growth and reproduction parameters
  - slider + toggle wiring in `js/ui.js` (including guardrails for min/max coupled values)
- Surfaced creature-level growth/reproduction telemetry in info panel:
  - growth event counts/suppression reasons/energy spend
  - topology/RL reset counters
  - reproduction suppression + resource debit counters
- Added mobile-specific UI scheme improvements for creature navigation:
  - replaced single mobile next button with bottom-centered Prev/Next pair
  - wired `mobilePrevCreatureButton` / `mobileNextCreatureButton` to creature cycling
  - refined mobile layout styles (control drawer width, compact top controls, reduced overlay clutter).
- Adjusted mobile info-panel behavior per Rick feedback:
  - selected-creature stats panel no longer auto-opens on mobile selection
  - added explicit mobile `Info` toggle button next to Prev/Next creature controls.
- Fixed runtime crash in `SoftBody.getAverageDamping()` (`RIGID_SPRING_DAMPING` reference): now uses `config.RIGID_SPRING_DAMPING`.
- Hardened legacy SoftBody brain helper paths by replacing bare constants with `config.*` equivalents to prevent future ReferenceErrors.
- Added regression test `SoftBody.getAverageDamping uses config rigid damping fallback for rigid-only springs`.
- Fixed save/load restore mismatch for grown creatures (`mass point count mismatch`) by hardening world persistence:
  - snapshots now serialize a phenotype-derived blueprint (point/spring topology from current runtime state)
  - restore path now has compatibility fallback that rebuilds blueprint from saved mass-point snapshots when stale blueprint counts are detected.
- Added new regression coverage:
  - `node-harness/tests/saveLoadRuntime.test.mjs` (run live world for growth-active steps, save, then load)
  - extended `worldPersistence.test.mjs` with stale-blueprint compatibility restore case.
- Clarified and enforced blueprint semantics in persistence:
  - `blueprint` now remains the heritable/reproductive body plan (birth plan + genes)
  - added separate `phenotypeBlueprint` in snapshots for reconstructing current grown physiology on load.
- Load path now restores current physiology from phenotype snapshot while preserving reproductive blueprint for offspring generation.
- Extended save/load regression tests to assert this separation explicitly (reproductive blueprint count can differ from current mass-point count after growth).
- Added growth-failure telemetry expansion:
  - `growthSuppressedByMaxPoints`
  - `growthSuppressedByNoCapacity`
  - `growthSuppressedByChanceRoll`
  - `growthSuppressedByPlacement`
- Surfaced new growth suppression counters in browser creature info panel and node soak summaries/log output.
- Added unit coverage for new growth telemetry counters in `node-harness/tests/softBodyGrowthGuardrails.test.mjs`.
- Added unstable-physics telemetry pipeline across shared runtime:
  - `SoftBody` now tags first instability cause (`unstableReason`) with explicit categories (`physics_*`, energy, age).
  - `stepWorld` now records rich per-death telemetry objects (reason class + current physiology + hereditary blueprint + heritable parameters), returns `removedBodies`, and aggregates totals in `worldState.instabilityTelemetry`.
  - persistence now round-trips `simulationStep` + `instabilityTelemetry` in world snapshots.
- Added automatic instability event dumping for micro-sim tooling:
  - `node-harness/runScenario.mjs` writes `*-instability-deaths.jsonl` + logs `[UNSTABLE_DEATH]` lines.
  - `node-harness/browserDefaultSoak.mjs` writes `*-instability-deaths.jsonl`, includes instability totals in summaries/reports, and logs each death payload.
  - browser runtime now logs `[UNSTABLE_DEATH]` entries to console and keeps `window.__instabilityDeaths` in-memory for inspection.
- Added/updated tests:
  - `node-harness/tests/stepWorld.test.mjs` instability telemetry coverage
  - `node-harness/tests/worldPersistence.test.mjs` simulationStep + instability telemetry round-trip coverage.
- Added browser startup launcher UX for scenario-driven runs:
  - initial overlay now shows two top actions: **Start Default World** and **Start Random World**
  - displays clickable card library of micro scenarios (`micro_*`) sourced from `js/engine/scenarioDefs.mjs`.
- Selecting a micro scenario launches its browser config preset directly (same named scenario family used by node micro tests).
- Added random-world launch generator with bounded slider-inspired ranges:
  - new module `js/engine/launcherConfig.mjs`
  - randomizes world size/population/fluid + growth/reproduction control subset with relational guardrails.
- Added unit coverage for random launcher config bounds/constraints:
  - `node-harness/tests/launcherConfig.test.mjs`.
- Added direct return path to launcher after simulation start:
  - new fixed **Scenario Library** button (`#openScenarioLibraryButton`) in top-left HUD
  - clicking clears startup URL params and reloads to the launcher overlay so users can switch scenarios/modes quickly.
- Camera/render architecture cleanup (mobile tiny-world scaling correctness + shared transform path):
  - introduced `js/engine/cameraMath.mjs` as canonical camera math layer (world↔display mapping, fit-world zoom, offset clamping/centering).
  - browser draw path now applies DPR-aware transform while keeping camera math in CSS-pixel space.
  - unified camera consumers (manual zoom, `View Entire Sim`, auto-follow, creature focus, resize clamp) to shared camera math helpers.
  - removed duplicate/contradictory legacy clamping blocks in UI pan flow.
  - added camera-focused tests:
    - `node-harness/tests/cameraMath.test.mjs`
    - `node-harness/tests/browserStepAdapter.test.mjs`.
- First instability-stabilization pass after browser drawing fix:
  - added world/dt-aware newborn stabilization module: `js/engine/newbornStability.mjs`.
  - spawn-time fit correction now runs for initialized populations, floor-spawned creatures, and reproduction offspring:
    - translates/scales newborn phenotype to fit current world bounds,
    - resets forced-correction implicit velocity,
    - scales spring rest lengths when downscaling is required.
  - added tiny-world newborn spring clamp logic (rigid + non-rigid stiffness/damping caps) with new config knobs in `js/config.js`.
  - step telemetry now classifies physics removals by subtype (`boundary_exit`, `numeric_or_nan`, `geometric_explosion`, `other_physics`) via `unstablePhysicsKind` and `instabilityTelemetry.removedByPhysicsKind`.
  - propagated telemetry defaults through world state/persistence and soak summary output.
  - added tests:
    - `node-harness/tests/newbornStability.test.mjs`
    - expanded `node-harness/tests/stepWorld.test.mjs`
    - expanded `node-harness/tests/worldPersistence.test.mjs`.
  - validation rerun (5 micro scenarios × 5 seeds): numeric/NaN instability dropped sharply (48 → 9); deaths are now mostly boundary exits (`boundary_exit`).
