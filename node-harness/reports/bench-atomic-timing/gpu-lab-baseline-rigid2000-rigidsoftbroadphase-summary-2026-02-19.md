# GPU Lab baseline 2000-rigid rigid-soft broadphase capture (2026-02-19)

## Artifacts
- Pre (before rigid-soft spatial broadphase cull; after rigid-rigid cache pass):
  - `node-harness/reports/bench-atomic-timing/gpu-lab-baseline-rigid2000-postcache-1771530749.json`
- Post (with rigid-soft spatial broadphase cull):
  - `node-harness/reports/bench-atomic-timing/gpu-lab-baseline-rigid2000-rigidsoftbroadphase-1771531722.json`

## Method
- `gpu-lab.html?worldScale=1`
- solver path: `baseline`
- rigid body count: `2000`
- 2 rounds each side, use aggregate means from last-16-frame timing windows.

## Renderer fingerprint
- Headless Chrome / SwiftShader (`rendererFingerprint.isSwiftShader === true`)
- Treat as relative-only signal.

## Deltas (post vs pre)
- `avgFrameMs`: `441.57 -> 448.73` (`+1.62%`)
- `avgStepBodiesMs`: `45.27 -> 28.20` (`-37.71%`)
- `avgCollisionMs`: `36.12 -> 19.13` (`-47.04%`)
- `avgReadbackMs`: `380.96 -> 405.17` (`+6.36%`)

Interpretation: collision/CPU stage drops sharply, but full-frame aggregate is noisy in SwiftShader lane because readback varies and dominates.

## New rigid-soft broadphase runtime evidence (post snapshot)
- `rigidSoftBroadphaseRuntime.totalRawNodeChecks`: `184,000`
- `rigidSoftBroadphaseRuntime.totalCandidateNodeChecks`: `826`
- `rigidSoftBroadphaseRuntime.totalPrunedNodeChecks`: `183,174`
- `rigidSoftBroadphaseRuntime.nodeReductionPct`: `99.55%`
- `rigidSoftBroadphaseRuntime.totalRawEdgeChecks`: `184,000`
- `rigidSoftBroadphaseRuntime.totalCandidateEdgeChecks`: `678`
- `rigidSoftBroadphaseRuntime.totalPrunedEdgeChecks`: `183,322`
- `rigidSoftBroadphaseRuntime.edgeReductionPct`: `99.63%`

Per-collision runtime (post snapshot):
- `collisionCpuRuntime.stageMs.rigidSoft`: `2.90 ms` (down from previous ~`19.3 ms` signal)
- `collisionCpuRuntime.checks.rigidSoftNode`: `826`
- `collisionCpuRuntime.checks.rigidSoftEdge`: `678`
- `collisionCpuRuntime.hitRatesPct.rigidSoftNodeCandidate`: `0.484%`
- `collisionCpuRuntime.hitRatesPct.rigidSoftEdgeCandidate`: `0.295%`
