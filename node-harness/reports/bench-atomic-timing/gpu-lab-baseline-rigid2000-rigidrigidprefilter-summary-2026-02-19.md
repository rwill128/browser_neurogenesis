# GPU Lab baseline 2000-rigid rigid-rigid SAT prefilter capture (2026-02-19)

## Artifacts
- Pre (after rigid-soft broadphase culling):
  - `node-harness/reports/bench-atomic-timing/gpu-lab-baseline-rigid2000-rigidsoftbroadphase-1771531722.json`
- Post (with rigid-rigid AABB + center-radius SAT prefilter):
  - `node-harness/reports/bench-atomic-timing/gpu-lab-baseline-rigid2000-rigidrigidprefilter-1771533581.json`

## Method
- `gpu-lab.html?worldScale=1`
- solver path: `baseline`
- rigid body count: `2000`
- 2 rounds per side, aggregate means from recent frame windows.

## Renderer fingerprint
- Headless Chrome / SwiftShader (`rendererFingerprint.isSwiftShader === true`)
- Relative-only signal lane.

## Deltas (post vs pre)
- `avgFrameMs`: `448.73 -> 446.02` (`-0.61%`)
- `avgStepBodiesMs`: `28.20 -> 26.85` (`-4.80%`)
- `avgCollisionMs`: `19.13 -> 17.70` (`-7.50%`)
- `avgReadbackMs`: `405.17 -> 403.85` (`-0.33%`)

Interpretation: frame-total delta is small/near-variance in this lane, but collision stage shows a clear additional reduction.

## New rigid-rigid prefilter counters (post snapshot)
From `collisionCpuRuntime`:
- `checks.rigidRigidPairs`: `66,032`
- `checks.rigidRigidProxyPairs`: `66,032`
- `checks.rigidRigidSatCalls`: `108`
- `prefilter.rigidRigidAabbRejected`: `65,916`
- `prefilter.rigidRigidRadiusRejected`: `8`
- `prefilter.rigidRigidTotalRejected`: `65,924`
- `hitRatesPct.rigidRigidSatCalls`: `0.164%` (SAT called on only ~0.16% of proxy pairs)
- `hitRatesPct.rigidRigidSatHits`: `37.04%` (SAT hit rate once called)

So the prefilter is skipping nearly all rigid-rigid proxy pairs before SAT while keeping deterministic ordering and collisions active.
