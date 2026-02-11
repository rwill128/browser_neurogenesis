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

1. Extract shared world-step core.
2. Introduce world state container.
3. Move UI side-effects out of core step.
4. Unify initialization path.
5. Continue RNG/context cleanup for deterministic behavior.
6. Split config/runtime concerns safely.
7. Standardize snapshot/query builder across browser + Node tooling.
8. Add parity regression checks.
9. Then add cross-runtime save/load (browser + Node).

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
