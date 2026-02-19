# Baseline vs GPU-Validated vs GPU-Fast: Step-by-Step Math Map

_Last updated: 2026-02-19_

This is the analysis document you asked for: a side-by-side map of the actual math pipeline in the three runtime flows.

- **Baseline** = `runtimeSolverPath=baseline` / `runtimePipelineMode=standard`
- **GPU-validated** = `runtimeSolverPath=gpu-only` + `runtimePipelineMode=gpu-only-validated`
- **GPU-fast** = `runtimeSolverPath=gpu-only` + `runtimePipelineMode=gpu-only-fast`

---

## 0) Mode selection + frame-level control plane

| Step | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Mode resolution | Uses baseline path when solver path is not `gpu-only`. | Uses gpu-only path + validated profile. | Uses gpu-only path + fast profile; enables cadence/skip behavior where implemented. |
| Where | `gpu-lab.js` mode normalizers and control read (`normalizeRuntimeSolverPath`, `normalizeRuntimePipelineMode`, `readControls`) | Same control-plane code | Same control-plane code |
| Inputs | URL/query + UI controls | Same | Same + `fastReadbackInterval` |
| Outputs | `sim.controls.runtimeSolverPath`, `sim.controls.runtimePipelineMode` | Same | Same + fast cadence metadata |

**Citations:** `sim-server/public/gpu-lab.js` L232-L243, L251-L323, L360-L362, L6335-L6353.

---

## 1) Fluid core loop (runs every frame before body coupling)

> These kernels are common to all 3 flows. Main difference in fast mode is readback policy, not fluid PDE math.

| Fluid stage | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Inject source (`injectWgsl`) | **Where:** `stepAndRender` dispatch. **Math:** adds impulse/dye source in injection radius. **I/O:** `(vx,vy,r,g,b)` in-place GPU buffers. | Same kernel + dispatch path. | Same kernel + dispatch path. |
| Velocity advection (`advectVelWgsl`) | **Math:** backtrace, sample velocity, apply viscosity decay, cap speed. **I/O:** `(vx0,vy0)->(vx1,vy1)` then swap. | Same | Same |
| Divergence (`divergenceWgsl`) | **Math:** finite-difference divergence, obstacle-aware. **I/O:** `(vx,vy)->div`. | Same | Same |
| Jacobi pressure solve (`jacobiPressureWgsl`) | **Math:** iterative pressure Poisson solve (`JACOBI_ITERS`). **I/O:** `pr0/pr1` ping-pong. | Same | Same |
| Projection (`projectWgsl`) | **Math:** subtract pressure gradient from velocity. **I/O:** updates `(vx,vy)`. | Same | Same |
| Dye advection (`advectDyeWgsl`) | **Math:** backtrace/sample and apply PASS/BLOCK/EAT mask semantics. **I/O:** `(r,g,b)` ping-pong. | Same | Same |
| Fluid readback decision | Always full readback each frame (`r,g,b,vx,vy`). | Same (full readback every frame). | Conditional readback: full readback only on cadence, bootstrap, fallback anomaly, health anomaly, or missing shadow fields; otherwise keep GPU-resident + reuse shadow copy. |
| Fast health check | N/A | N/A | Runs `fastFluidHealthWgsl` to detect non-finite/velocity spike/dye spike before deciding skip/readback. |
| Post-readback CPU passes | Runs body coupling + emitters + digestive capture + swept dye transport + velocity guardrails + writes fields back to GPU. | Same | Same when readback happens; when skipped, uses shadow fields and skips re-upload that frame. |

**Citations:**
- Kernels: `sim-server/public/gpu-lab.js` L696-L1088.
- Fast health kernel: `sim-server/public/gpu-lab.js` L1090-L1179.
- Dispatch order + swaps + readback policy: `sim-server/public/gpu-lab.js` L6233-L6460.
- Readback helper funcs: `sim-server/public/gpu-lab.js` L6186-L6231.
- CPU post-fluid passes: `sim-server/public/gpu-lab.js` L6459-L6533.

---

## 2) Body/constraint/collision pipeline (`stepBodiesAndInject`) side-by-side

## 2.1 Rigid fluid coupling + rigid integration

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline CPU loop in `gpu-lab.js` | `stepRigidBodiesGpuOnly.js` with WGSL proposal + CPU fallback | Same module, but fast treats authoritative WGSL replay more aggressively |
| Inputs | Rigid body state, sampled fluid at rigid vertices, drag params, masses/inertia, swim gain | Same + WGSL proposal layout/signature | Same |
| Math | `rel = fluid - localRigidVel`; `force = rel * drag * honey * momentumScale`; `torque += r x force`; integrate `(vx,vy,omega)` with swim; damp + cap; update `(x,y,theta)`; boundary bounce | CPU computes same force model; WGSL proposal mirrors integrator and can be applied authoritatively if signature/finite checks pass | Same equations; fast mode requests authoritative route by default (`wgsl-rigid-step-authoritative-fast`) and falls back to CPU fast-tagged source if needed |
| Outputs | Updated rigid kinematics + `rigidCarryTransfer` | Same + proposal/fallback telemetry in offload state | Same with fast route labels |

**Citations:**
- Baseline CPU equations: `sim-server/public/gpu-lab.js` L4073-L4196.
- GPU-only module equations + WGSL proposal: `sim-server/public/runtime-solvers/stepRigidGpuOnly.js` L1-L115, L270-L417.
- Fast/validated authoritative routing: `sim-server/public/runtime-solvers/stepRigidGpuOnly.js` L431-L485.

## 2.2 Soft fluid coupling (node carry + cluster load + projection)

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline CPU loop in `gpu-lab.js` | `stepSoftFluidCouplingGpuOnly.js` | Same module with fast shortcuts |
| Inputs | Soft nodes, cluster IDs, sampled fluid, drag/honey, momentum scales, membrane set, cluster kinematics | Same + prepared WGSL layouts/proposals | Same |
| Math | Per node: compute cluster-local relative flow, force/torque, local carry; apply damping/caps; accumulate cluster force/torque; then apply cluster acceleration back to nodes, tug redistribution, relative damping, rigid-motion projection | Same CPU reference path retained; optional authoritative WGSL carry/cluster reductions compared against CPU parity in validated mode | Fast can skip CPU proposal arrays when authoritative WGSL proposal finite (`useFastAuthoritativeCarryShortcut`) and skip strict mismatch parity checks; still applies same resulting velocity fields |
| Outputs | Updated soft velocities + `softCarryTransfer` + cluster telemetry | Same + proposal parity telemetry | Same with fast shortcut/skip telemetry |

**Citations:**
- Baseline equations: `sim-server/public/gpu-lab.js` L4200-L4433.
- GPU-only module core flow: `sim-server/public/runtime-solvers/stepSoftFluidCouplingGpuOnly.js` L808-L1333.
- Fast shortcut + cluster-load conditions: `sim-server/public/runtime-solvers/stepSoftFluidCouplingGpuOnly.js` L925-L983, L1147-L1183, L1300-L1328.

## 2.3 Soft spring XPBD

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `applySoftSpringsXPBDVelocity` in `gpu-lab.js` | `applySoftSpringsXPBDVelocityGpuOnly` in `stepSoftSpringsXpbdGpuOnly.js` | Same module, fast replay/skips active |
| Inputs | Springs `(i,j,rest)`, node masses/velocities, `lambdaCache`, `dtPos`, compliance/stiffness | Same + colorized WGSL layout/proposal buffers | Same |
| Math | XPBD stretch solve: `C = clamp(d-rest)`, `dl = (-C - alpha*lambdaPrev)/(wSum+alpha)`, clamp lambda, apply velocity deltas to endpoints | CPU remains deterministic reference; WGSL probe/proposal/node-reduction generated and validated | Fast path allows epoch-fresh replay and may skip probe telemetry (`skipped-fast-mode`); can start CPU iterations at `softXpbdIters` when authoritative proposal accepted |
| Outputs | Updated node velocities + `lambdaCache` | Same + proposal parity/finite telemetry | Same + reduced parity overhead in fast mode |

**Citations:**
- Baseline XPBD spring math: `sim-server/public/gpu-lab.js` L3047-L3137.
- WGSL lambda math: `sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js` L60-L101.
- GPU-only apply + fast logic: `sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js` L1341-L1606, L1645-L1650.

## 2.4 Membrane boundary XPBD + membrane bend XPBD

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `applySoftMembraneBoundaryXPBDVelocity` in `gpu-lab.js` | `applySoftMembraneBoundaryXPBDVelocityGpuOnly` in `stepSoftMembraneConstraintsGpuOnly.js` | Same module |
| Inputs | Membrane loops, rest edge lengths, rest bend spans, lambdas, masses, `dtPos` | Same + WGSL proposal buffers/signatures | Same |
| Math | Edge and bend constraints each use XPBD-style `dl` solve and endpoint velocity corrections | CPU still runs deterministic reference while WGSL proposals are generated and can replace CPU deltas when authoritative checks pass | No special new math; mostly same as validated with route selection and potential replay use |
| Outputs | Updated membrane node velocities | Same + authoritative source telemetry | Same |

**Citations:**
- Baseline membrane boundary/bend math: `sim-server/public/gpu-lab.js` L3302-L3416.
- GPU-only membrane boundary module: `sim-server/public/runtime-solvers/stepSoftMembraneConstraintsGpuOnly.js` L775-L1103.

## 2.5 Membrane shape memory

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `applySoftMembraneShapeMemoryVelocity` in `gpu-lab.js` | `applySoftMembraneShapeMemoryVelocityGpuOnly` in `stepSoftMembraneConstraintsGpuOnly.js` | Same module |
| Inputs | Loop-local reference pose, centroid, rigid alignment (`theta`), gain limits, node mass/weights | Same + WGSL proposal buffers/signatures | Same |
| Math | Compute target rotated reference point per node, clamp shift, apply velocity correction proportional to gain/weight/invMass | CPU authoritative reference with optional WGSL proposal overwrite if valid | Same core math; fast mostly reduces validation burden rather than changing formula |
| Outputs | Shape-memory velocity corrections | Same + source-route telemetry | Same |

**Citations:**
- Baseline shape-memory math: `sim-server/public/gpu-lab.js` L3418-L3508.
- GPU-only shape-memory path: `sim-server/public/runtime-solvers/stepSoftMembraneConstraintsGpuOnly.js` L1136-L1315.

## 2.6 Soft area XPBD

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `applySoftAreaXPBDVelocity` in `gpu-lab.js` | `applySoftAreaXPBDVelocityGpuOnly` in `stepSoftAreaXpbdGpuOnly.js` | Same module with fast probe/replay changes |
| Inputs | Cluster loops, predicted area, rest area map, lambda map, node masses, `dtPos` | Same + deterministic CSR layout and WGSL proposal signatures | Same |
| Math | Area constraint solve with gradient accumulation per loop node, XPBD `dl`, then velocity delta distribution by gradients | CPU reference remains; WGSL probe + lambda + velocity proposals generated and parity-checked | Fast skips probe stage in many frames, allows epoch replay (`proposalEpochDelta <= 2`), and can skip endpoint parity telemetry while still enforcing finite checks |
| Outputs | Updated loop node velocities, updated area lambda | Same + parity telemetry | Same + reduced telemetry overhead |

**Citations:**
- Baseline area XPBD math: `sim-server/public/gpu-lab.js` L3139-L3217.
- GPU-only module + fast replay/skip logic: `sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js` L1193-L1399.
- WGSL area/lambda equations: `sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js` L119-L125.

## 2.7 Membrane pressure

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `applySoftMembraneCellPressure` in `gpu-lab.js` | `applySoftMembraneCellPressureGpuOnly` in `stepSoftMembranePressureGpuOnly.js` | Same module with fast validation skip behavior |
| Inputs | Membrane loops, baseline area, current area, pressure gain/radial damping, `dtPos` | Same + WGSL area probe + velocity proposal layouts | Same |
| Math | `err = clamp((areaBase-areaNow)/areaBase)`; `gain = pressureGain*(1+min(1.4,abs(err)*2.2))`; impulse along outward normal plus optional radial damping | Same reference math with optional WGSL proposal application if finite/valid | Fast mode allows WGSL proposal acceptance with skipped CPU parity when finite; fallback routes stay explicit |
| Outputs | Membrane pressure velocity corrections | Same + parity stats | Same + fast source labels |

**Citations:**
- Baseline membrane pressure math: `sim-server/public/gpu-lab.js` L3510-L3600.
- GPU-only pressure formulas and fast/validated branching: `sim-server/public/runtime-solvers/stepSoftMembranePressureGpuOnly.js` L46-L49, L110-L113, L615-L707, L728-L790.

## 2.8 Soft integrate

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline CPU in `gpu-lab.js` | `integrateSoftBodiesGpuOnly` | Same module |
| Inputs | Node `(x,y,vx,vy)`, caps, `dt`, scale, boundary callback | Same | Same |
| Math | Clamp node speed to `hybridNodeVCap`; `x += vx*dt*scale`, `y += vy*dt*scale`; boundary bounce | WGSL proposal computes same update, validated against CPU parity | Fast accepts WGSL proposal with finite checks and skips full CPU parity diff path |
| Outputs | Integrated soft positions/velocities | Same + `softIntegrateRuntime` source info | Same |

**Citations:**
- Baseline integrate: `sim-server/public/gpu-lab.js` L4569-L4596.
- GPU-only module: `sim-server/public/runtime-solvers/stepSoftIntegrateGpuOnly.js` L82-L136, L381-L437.

## 2.9 Rigid post-integrate clamp

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline CPU in `gpu-lab.js` | `stabilizeRigidPostIntegrateGpuOnly` | Same module |
| Inputs | Rigid `(vx,vy,omega)` and caps | Same | Same |
| Math | Clamp translational speed and omega | WGSL clamp then readback apply | Same math; fast currently mostly route/telemetry difference (not a separate formula) |
| Outputs | Stabilized rigid velocities | Same + runtime mode telemetry | Same |

**Citations:**
- Baseline clamp: `sim-server/public/gpu-lab.js` L4598-L4623.
- GPU-only module: `sim-server/public/runtime-solvers/stepRigidPostIntegrateGpuOnly.js` L52-L54, L203-L231.

## 2.10 Collision iteration schedule (orchestrator)

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline collision loop in `gpu-lab.js` | `runCollisionIterationsGpuOnly` orchestrator | Same orchestrator |
| Inputs | Rigid bodies, soft nodes/springs, slops, bounce coefficients | Same | Same |
| Math schedule | Per iter: rigid-rigid pre-soft → rigid-soft → soft-soft → rigid-rigid post-soft → boundary | Same order, delegated to module passes with WGSL offload states | Same order; fast modifies internals of some delegated passes (especially rigid-soft) |
| Outputs | Updated bodies after collision resolution + boundary runtime | Same | Same |

**Citations:**
- Baseline collision loop: `sim-server/public/gpu-lab.js` L4625-L4741.
- GPU-only orchestrator: `sim-server/public/runtime-solvers/stepCollisionIterationsGpuOnly.js` L8-L95.

## 2.11 Rigid-rigid collision pass

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `resolveRigidVsRigidPolygonCollision` called in baseline loop | `resolveRigidRigidCollisionPassGpuOnly` | Same module |
| Inputs | Rigid pair geometry + slop | Same | Same |
| Math | Polygon collision + impulse/correction from CPU helper | WGSL proposal can be authoritative if finite and validated; CPU fallback otherwise | Same authority rules; no major separate fast formula path |
| Outputs | Updated rigid pair velocities/positions | Same + route telemetry | Same |

**Citations:**
- Baseline call sites: `sim-server/public/gpu-lab.js` L4634-L4645, L4722-L4733.
- GPU-only module: `sim-server/public/runtime-solvers/stepRigidCollisionGpuOnly.js` L350-L380.

## 2.12 Rigid-soft collision pass (highest complexity)

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline loops in `gpu-lab.js` (`resolveRigidVsSoftNodeCollision` + edge checks) | `resolveRigidSoftCollisionPassGpuOnly` in `stepRigidSoftCollisionGpuOnly.js` | Same module with cadence/replay/skip gates |
| Inputs | Rigid states + soft nodes/springs + slops + candidate layouts | Same + WGSL broadphase/narrowphase/response buffers | Same + fast interval/cadence state |
| Math (high-level) | For each rigid-soft candidate, resolve penetration + impulse on node or edge contact | GPU path builds candidate pairs, broadphase masks, optional AABB probe filters, then response proposal (node+edge impulses), then authoritative apply if valid | Fast adds: broadphase replay between cadence frames; **node & edge narrowphase AABB probe readback skipped** on fast branch; response may use authoritative replay or CPU fallback if invalid |
| Outputs | Updated rigid+soft states after rigid-soft contacts | Same + detailed timing and source-route telemetry | Same + additional `*-skipped-fast` / replay route telemetry |

**Citations:**
- Baseline rigid-soft loops: `sim-server/public/gpu-lab.js` L4648-L4661, L2286-L2318.
- GPU-only module entry: `sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js` L2907-L2990.
- Fast cadence/replay and probe skips: `sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js` L3025-L3119, L3174-L3258.
- Authoritative response + fallback routing: `sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js` L3414-L3574.

## 2.13 Soft-soft collision pass

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Baseline inline loops for node-node + node-edge blocking edges | `resolveSoftSoftCollisionPassGpuOnly` | Same module |
| Inputs | Soft nodes/springs + slops + edge-block mask | Same | Same |
| Math | Node-node circle collision + node-edge segment collision | WGSL node-node and node-edge proposals can be authoritative when finite; CPU fallback per pass | Same formulas; fast mainly affects route labeling/pending-route telemetry, not core collision equations |
| Outputs | Updated soft positions/velocities | Same + source-route telemetry | Same |

**Citations:**
- Baseline helpers: `sim-server/public/gpu-lab.js` L2231-L2284, L2320-L2364, baseline loop L4675-L4720.
- GPU-only pass: `sim-server/public/runtime-solvers/stepSoftCollisionGpuOnly.js` L617-L740.

## 2.14 Boundary collision pass

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `applyBounceBoundary` over rigid + soft entries | `applyCollisionBoundaryPassGpuOnly` | Same module |
| Inputs | Rigid + soft entries and grid bounds | Same | Same |
| Math | Clamp positions to bounds; reflect/damp velocity component | WGSL path for finite entries, CPU path for non-finite entries; full CPU fallback on errors | Same (no special fast-only formula) |
| Outputs | Boundary-corrected states | Same + finite/non-finite counts + mode telemetry | Same |

**Citations:**
- Baseline usage: `sim-server/public/gpu-lab.js` L4736-L4737.
- GPU-only boundary pass: `sim-server/public/runtime-solvers/stepCollisionBoundaryGpuOnly.js` L300-L371.

## 2.15 Post-collision recovery (cluster projection + inside correction)

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline branch in `gpu-lab.js` | `applyPostCollisionRecoveryGpuOnly` + delegated inside-correction modules | Same modules with fast readback-telemetry relaxations in inside-correction |
| Inputs | Post-collision bodies/nodes, cluster kinematics, loops, gains | Same + WGSL mass-moment/derived kinematics proposals | Same |
| Math | Project soft nodes toward cluster rigid motion; then rigid-inside and membrane-inside correction passes; boundary pass again | Same flow, with optional WGSL authoritative cluster kinematics/mass moments and delegated GPU-assisted inside correction | Same flow; inside-correction path in fast mode skips some readback telemetry checks and uses fast source routes if proposals are finite |
| Outputs | `rigidInsideCorrections`, `membraneInsideCorrections`, corrected states | Same + parity/source-route telemetry | Same + fast validation labels |

**Citations:**
- Baseline branch: `sim-server/public/gpu-lab.js` L4744-L4800.
- Post-collision module: `sim-server/public/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js` L669-L870.
- Rigid inside correction fast behavior: `sim-server/public/runtime-solvers/stepRigidInsideCorrectionGpuOnly.js` L379-L381, L691-L774, L954-L1198.
- Baseline inside correction formulas: `sim-server/public/gpu-lab.js` L2366-L2455 and L2457-L2589.

## 2.16 Soft deformation interventions (warning/severe)

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | `buildSoftDeformationState` + severe stabilization in `gpu-lab.js` | `applySoftDeformationInterventionsGpuOnly` | Same module |
| Inputs | Soft topology/loops, deformation thresholds, intervention toggles | Same + optional WGSL spring-metric proposal | Same |
| Math | Compute stretch/area/pose error state; if severe collapse enabled, run stabilization and rebuild state | Same with optional WGSL spring-metric authoritative injection; CPU fallback remains explicit | Same core math |
| Outputs | Updated deformation state + event logs | Same + source-route telemetry | Same |

**Citations:**
- Baseline deformation state/intervention hooks: `sim-server/public/gpu-lab.js` L3680-L3875, L4802-L4836.
- GPU-only module: `sim-server/public/runtime-solvers/stepSoftDeformationGpuOnly.js` L435-L524.

## 2.17 Soft spring rest-length recovery

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline `recoverSoftSpringRests(...)` in `gpu-lab.js` | `applySoftRestRecoveryGpuOnly` | Same module |
| Inputs | Springs + baseline rest lengths + profile options from deformation state | Same + WGSL probe/proposal state | Same |
| Math | Profile-dependent recovery coefficients (baseline/warning/severe), then rest-length adaptation | Same, with optional authoritative WGSL proposal apply when signature matches | Same overall formula; fast mainly affects proposal routing/telemetry overhead |
| Outputs | Updated spring rest lengths | Same + authoritative source route | Same |

**Citations:**
- Baseline profile and call: `sim-server/public/gpu-lab.js` L4842-L4889.
- GPU-only rest recovery: `sim-server/public/runtime-solvers/stepSoftRestRecoveryGpuOnly.js` L532-L663.

## 2.18 Body→fluid injection gather/apply

| Item | Baseline | GPU-validated | GPU-fast |
|---|---|---|---|
| Where | Inline `injectPoint` + rigid/soft loops in `gpu-lab.js` | `applyBodyFluidInjectionGpuOnly` | Same module with fast gather acceptance |
| Inputs | Body/node kinematics, local fluid samples, feedback gain, momentum scales, coupling caps | Same + gather layout + WGSL gather proposal buffers | Same |
| Math | Compute relative body-vs-fluid delta, clamp by coupling limit, deposit momentum over radius-weighted neighborhood into field cells | Same gather/apply math; validated mode compares WGSL gather proposal with CPU parity before authoritative use | Fast can accept same-frame finite WGSL gather proposal (`wgsl-gather-authoritative-fast`) and skip CPU parity checks; explicit CPU-fast fallback routes remain |
| Outputs | Updated `(vx,vy)` field deltas + `injectedMomentum` + feedback maps | Same + gather parity telemetry | Same + reduced parity overhead |

**Citations:**
- Baseline inject math: `sim-server/public/gpu-lab.js` L4921-L5036.
- GPU-only gather/apply + fast branch: `sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js` L568-L760.

---

## 3) Net summary of what is mathematically different vs what is execution-policy different

### 3.1 Mostly the **same math**, different execution policy
These stages generally preserve baseline equations while changing where/when computations run:
- Rigid step
- Soft fluid coupling
- Spring XPBD
- Area XPBD
- Membrane boundary/shape/pressure
- Soft integrate
- Collision passes
- Rest recovery
- Body-fluid injection

### 3.2 Where GPU-fast currently changes behavior most
The biggest practical differences are in **readback cadence, replay acceptance, and parity/telemetry strictness**, not entirely new physics equations:
1. Fast fluid readback cadence + anomaly-triggered full readback.
2. Rigid-soft broadphase replay between cadence frames.
3. Rigid-soft node/edge narrowphase AABB probe readback skipped on fast path.
4. Multiple proposal stages allow fast authoritative replay by signature/epoch freshness.
5. Fast mode often skips CPU parity comparison when finite checks pass.

**Citations:** `sim-server/public/gpu-lab.js` L6335-L6460; `sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js` L3025-L3258; `sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js` L1400-L1479; `sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js` L1230-L1299; `sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js` L675-L730.

---

## 4) Suggested next expansion (for even deeper analysis)

If we want the next version to be fully forensic, we can add one appendix per stage with:
- exact buffer schema (field names + strides),
- full equation derivation line-by-line,
- all fallback route names and their trigger conditions,
- readback/sync points by stage (`copyBufferToBuffer`, `mapAsync`) with timing buckets.

That would make this doc significantly longer, but it would support direct bottleneck attribution stage-by-stage.
