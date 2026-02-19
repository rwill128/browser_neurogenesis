# GPU Lab baseline solver profile (2026-02-19)

This profile targets the **default solver path in `gpu-lab.html`** (`runtimeSolverPath=baseline`).

## Capture artifacts

- CPU profile: `node-harness/reports/bench-atomic-timing/gpu-lab-baseline-cpu-1771519394279.cpuprofile`
- Status snapshot: `node-harness/reports/bench-atomic-timing/gpu-lab-baseline-status-1771519394287.json`

## Method

- Open `http://127.0.0.1:8787/gpu-lab.html`
- Set solver path to `baseline`
- Start sim
- Capture Chrome DevTools CPU profile for ~12s via CDP `Profiler.start/stop`

## Top non-idle sampled functions (baseline path)

- `stepAndRender` — `sim-server/public/gpu-lab.js`
- `stepBodiesAndInject` — `sim-server/public/gpu-lab.js`
- `stampSegmentSweepTransport` — `sim-server/public/gpu-lab.js`
- `sampleFluidForBodyCoupling` — `sim-server/public/gpu-lab.js`
- `stampBodyObstacleMask` — `sim-server/public/gpu-lab.js`
- `redistributeSweptEdgeDyeTransport` — `sim-server/public/gpu-lab.js`
- `resolveRigidVsSoftNodeCollision` — `sim-server/public/rigid-collision.js`
- `resolveRigidVsRigidPolygonCollision` — `sim-server/public/rigid-collision.js`
- `mapFullFluidReadback` — `sim-server/public/gpu-lab.js`

## Interpretation

This confirms profiling is on the `gpu-lab` default solver code path (not the legacy app path under `js/classes/*`).

Complementary timing captures show baseline frame budget dominated by:
- `fluid.readback.fullMap` (~18 ms/frame in recent captures)
- then `frame.stepBodiesAndInject.total` (~1.2 ms/frame)

So baseline bottleneck investigation in GPU Lab should prioritize fluid readback/submit behavior + core fluid pipeline costs before deeper solver micro-optimizations.
