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

## 4) Forensic expansion: CPU + GPU + sync traces for all three flows

This section is the deeper version you requested: not just buffer/readback notes, but explicit **main CPU flow** and execution order in all 3 modes.

### 4.1 Frame-level execution ledger (ordered, all three flows)

| Order | Baseline (main CPU flow) | GPU-validated (main CPU flow + GPU path) | GPU-fast (main CPU flow + GPU path) | Inputs → Outputs | Where |
|---|---|---|---|---|---|
| F1 | Read controls, enforce reinit guards (grid/seed/body-count changes). | Same. | Same. | UI/URL controls → `sim.controls` | `gpu-lab.js` L6233-L6284 |
| F2 | CPU stamps obstacle mask and dye mode masks from body edges, writes masks to GPU buffers. | Same. | Same. | body geometry + edge policy → `obstacleMaskGpu`, `dyeModeMaskGpu` | `gpu-lab.js` L6289-L6299 |
| F3 | CPU encodes fluid compute passes (`inject`, `advectVel`, `divergence`, `jacobi`, `project`, `advectDye`). | Same. | Same. | previous fluid state + uniforms → next fluid state | `gpu-lab.js` L6301-L6333 |
| F4 | Always enqueue full-fluid readback copies (5 channels). | Same (always full readback in validated). | Run health compute + health readback first; decide whether to enqueue full readback by cadence/anomaly/bootstrap rules. | GPU buffers → pending readback buffers / health stats | `gpu-lab.js` L6350-L6415 |
| F5 | Map full readback buffers every frame. | Same. | Map full readback only when `doReadback=true`; else use shadow fields from prior full readback. | read buffers/shadow → `r,g,b,vx,vy` CPU arrays | `gpu-lab.js` L6417-L6423, L6194-L6215 |
| F6 | Call `stepBodiesAndInject` on CPU arrays (baseline branch inside function). | Call `stepBodiesAndInject` gpu-only branch (runtime modules + CPU orchestration). | Same function path, but module-level fast gates/replay/skip logic active. | CPU fluid arrays + body state → updated body state + fluid deltas + telemetry | `gpu-lab.js` L6459-L6460, L3972-L5248 |
| F7 | Apply emitters, digestive capture, swept dye transport (CPU). | Same. | Same (using current readback/shadow arrays). | post-coupling fields → modified dye/velocity arrays | `gpu-lab.js` L6461-L6465 |
| F8 | Enforce velocity finite/clamp guardrails in CPU arrays. | Same. | Same. | raw `vx,vy` → bounded finite `vx,vy` | `gpu-lab.js` L6467-L6477 |
| F9 | Write modified arrays back to GPU every frame. | Same. | Write-back only on full-readback frames (`doReadback` true). | CPU arrays → GPU state buffers | `gpu-lab.js` L6479-L6486 |
| F10 | Build render image, draw overlays (CPU canvas). | Same. | Same. | dye fields + overlays → frame render | `gpu-lab.js` L6488-L6518 |
| F11 | Update HUD/status/fallback telemetry logs (CPU). | Same + richer WGSL source-route signals. | Same + fast policy decisions/reasons/health fields. | sim state → HUD/log payload | `gpu-lab.js` L6520-L6622 |
| F12 | Increment frame and schedule next RAF. | Same. | Same. | frame counter update | `gpu-lab.js` L6624-L6625 |

---

### 4.2 `stepBodiesAndInject` call graph (all three flows)

| Stage | Baseline CPU flow | GPU-validated flow | GPU-fast flow | Where |
|---|---|---|---|---|
| S0: preamble | Clamp dt, set masses/inertia, prepare feedback maps, optional scripted body motion. | Same CPU preamble. | Same CPU preamble. | `gpu-lab.js` L3972-L4072 |
| S1: rigid step | Inline rigid CPU loop. | `stepRigidBodiesGpuOnly` called; CPU computes proposal inputs and may consume WGSL authoritative proposal. | Same module; fast requests authoritative replay path more aggressively (`wgsl-rigid-step-authoritative-fast`). | `gpu-lab.js` L4073-L4196; `stepRigidGpuOnly.js` L270-L485 |
| S2: soft fluid coupling | Inline CPU loop for node carry + cluster redistribution + projection. | `applySoftFluidCouplingGpuOnly`; CPU reference maintained with WGSL proposal parity checks. | Same module; fast can use authoritative carry shortcut and reduced parity strictness. | `gpu-lab.js` L4200-L4433; `stepSoftFluidCouplingGpuOnly.js` L808-L1333 |
| S3: spring XPBD | `applySoftSpringsXPBDVelocity` CPU. | `applySoftSpringsXPBDVelocityGpuOnly` (WGSL proposal + deterministic reduction + CPU authority/replay). | Same module; fast can skip probe telemetry and use epoch-fresh proposal replay. | `gpu-lab.js` L4435-L4456; `stepSoftSpringsXpbdGpuOnly.js` L1341-L1650 |
| S4: membrane boundary/shape | CPU membrane boundary XPBD + shape memory. | GPU-only membrane modules with optional authoritative proposal apply. | Same modules; mostly same math with fast route labels. | `gpu-lab.js` L4464-L4524; `stepSoftMembraneConstraintsGpuOnly.js` L775-L1315 |
| S5: area XPBD + membrane pressure | CPU area XPBD + CPU membrane pressure. | GPU-only area/pressure modules with proposal parity checks. | Fast skips some probe/parity telemetry, permits epoch replay/fast authoritative routes when finite. | `gpu-lab.js` L4526-L4565; `stepSoftAreaXpbdGpuOnly.js` L1193-L1399; `stepSoftMembranePressureGpuOnly.js` L615-L790 |
| S6: integrate/clamp | CPU soft integrate + CPU rigid post-integrate clamp. | `integrateSoftBodiesGpuOnly` + `stabilizeRigidPostIntegrateGpuOnly`. | Same modules; fast mainly changes validation policy/route tags, not equations. | `gpu-lab.js` L4569-L4623 |
| S7: collision iterations | Inline CPU collision loops. | `runCollisionIterationsGpuOnly` delegates rigid-rigid, rigid-soft, soft-soft, boundary passes. | Same orchestrator; rigid-soft internals include fast cadence/replay/probe skips. | `gpu-lab.js` L4625-L4741; `stepCollisionIterationsGpuOnly.js` L8-L95 |
| S8: post-collision recovery | CPU cluster projection + CPU rigid/membrane inside correction + bounce. | `applyPostCollisionRecoveryGpuOnly` with optional WGSL authoritative cluster kinematics and GPU-assisted inside correction. | Same module; fast inside-correction relaxes some readback telemetry checks. | `gpu-lab.js` L4744-L4800; `stepPostCollisionRecoveryGpuOnly.js` L669-L870 |
| S9: deformation intervention | Build deformation state + optional severe stabilization CPU. | `applySoftDeformationInterventionsGpuOnly` with optional WGSL spring metric authority. | Same module; fast mostly route/policy difference. | `gpu-lab.js` L4802-L4836; `stepSoftDeformationGpuOnly.js` L435-L524 |
| S10: rest recovery | CPU `recoverSoftSpringRests` with profile tuning. | `applySoftRestRecoveryGpuOnly` with optional authoritative proposal. | Same module; fast proposal routing has lower parity burden. | `gpu-lab.js` L4842-L4889; `stepSoftRestRecoveryGpuOnly.js` L568-L663 |
| S11: body→fluid injection | CPU `injectPoint` loops for rigid and soft contributors. | `applyBodyFluidInjectionGpuOnly` gather/apply path with WGSL proposal parity checks. | Same module; fast accepts finite same-frame WGSL gather proposal and skips CPU parity diff. | `gpu-lab.js` L4921-L5036; `stepBodyFluidInjectionGpuOnly.js` L568-L760 |

---

## 5) Deep-dive stage dossiers (row-level substeps, all 3 flows)

## 5.1 Rigid step dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| R1 sample vertices | Sample fluid at rigid sample points; compute relative fluid-vs-local rigid velocity. | Same CPU sampling prepares WGSL layout too. | Same. | in: rigid verts + `(vx,vy)` field; out: rel velocities | `gpu-lab.js` L4106-L4149; `stepRigidGpuOnly.js` L321-L347 |
| R2 accumulate forces | `forceX/forceY += rel*drag*honey*momentumScale`, `torque += r x force`. | Same equations. | Same equations. | out: `(ax,ay,alpha)` | `gpu-lab.js` L4151-L4167; `stepRigidGpuOnly.js` L347-L357 |
| R3 integrate motion | Update `(vx,vy,omega)` with accel + swim; apply damping/caps; update `(x,y,theta)`; boundary bounce. | CPU performs same; WGSL proposal mirrors this transform. | Same; fast prefers authoritative WGSL application when proposal is fresh/finite. | out: new rigid state | `gpu-lab.js` L4169-L4196; `stepRigidGpuOnly.js` L1-L97, L128-L237, L431-L485 |
| R4 authority decision | CPU authoritative by definition. | If WGSL proposal signature/finite checks pass, apply proposal; else CPU fallback route. | Same decision with fast-tagged authoritative/fallback route labels. | out: route telemetry | `stepRigidGpuOnly.js` L431-L485 |

## 5.2 Soft fluid coupling dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| SF1 per-node sample | For each node: sample fluid at node, compute cluster-local velocity and deltas. | Same CPU sample layout built for WGSL proposal parity. | Same sample layout; can consume authoritative WGSL carry arrays. | out: sample deltas per node | `gpu-lab.js` L4256-L4316; `stepSoftFluidCouplingGpuOnly.js` L185-L248 |
| SF2 carry/local response | Compute force/carry/localCarry, apply local carry + swim + damping/caps. | Same CPU reference. | Fast can bypass CPU proposal accumulation and directly use finite WGSL carry proposal (`useFastAuthoritativeCarryShortcut`). | out: node `vx,vy` updates | `gpu-lab.js` L4319-L4375; `stepSoftFluidCouplingGpuOnly.js` L925-L1127 |
| SF3 cluster load | Accumulate per-cluster force/torque/count then derive cluster accel `(ax,ay,alpha)`. | Same, plus parity against WGSL cluster-load proposal in validated mode. | Fast can trust finite authoritative WGSL cluster-load proposal without mismatch parity requirement. | out: `clusterAccelMap` | `gpu-lab.js` L4378-L4411; `stepSoftFluidCouplingGpuOnly.js` L954-L1183 |
| SF4 redistribution/project | Apply cluster accel to nodes, tug redistribution, relative drift damping, rigid-motion projection. | Same math. | Same math. | out: coherent cluster motion | `gpu-lab.js` L4413-L4433; `stepSoftFluidCouplingGpuOnly.js` L1185-L1267 |

## 5.3 Spring XPBD dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| SX1 prediction | Build predicted endpoint positions from current velocity (`x+vx*dtPos`). | Same inputs prepared into WGSL layout. | Same. | node predicted positions | `gpu-lab.js` L3066-L3073; `stepSoftSpringsXpbdGpuOnly.js` L864-L971 |
| SX2 constraint solve | Compute stretch `C`, inverse masses, XPBD delta lambda `dl`, clamp lambda. | WGSL lambda proposal computes same formula; validated compares against deterministic CPU reduction. | Fast can reuse fresh proposal by epoch and skip probe telemetry. | out: lambda next | `gpu-lab.js` L3074-L3100; `stepSoftSpringsXpbdGpuOnly.js` L95-L101, L1400-L1479 |
| SX3 velocity correction | Apply endpoint velocity deltas from `dl/dtPos`. | Same corrections from proposal/reduction path or CPU fallback. | Same equations, fast lowers parity overhead. | out: endpoint `vx,vy` | `gpu-lab.js` L3102-L3107; `stepSoftSpringsXpbdGpuOnly.js` L1103-L1139, L1645-L1650 |

## 5.4 Area XPBD dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| AX1 area & gradients | Compute predicted signed area and per-node gradients for each loop. | WGSL area probe/lambda uses same geometric primitives. | Fast often skips explicit area probe telemetry stage (`skipped-fast-mode`). | out: `C`, gradients, `sumWGrad2` | `gpu-lab.js` L3156-L3206; `stepSoftAreaXpbdGpuOnly.js` L119-L125, L1273-L1278 |
| AX2 lambda solve | Compute `dl` and clamp cluster lambda. | WGSL lambda proposal + CPU parity in validated. | Fast accepts replay by signature/epoch delta (`<=2`) when finite. | out: cluster lambda update | `gpu-lab.js` L3208-L3217; `stepSoftAreaXpbdGpuOnly.js` L1230-L1260 |
| AX3 apply deltas | Apply velocity corrections by gradient weights. | Node-reduction proposal checked against CPU node reference in validated mode. | Fast may skip endpoint/node parity telemetry but still finite-checks proposal. | out: node `vx,vy` | `gpu-lab.js` L3212-L3217; `stepSoftAreaXpbdGpuOnly.js` L1293-L1369 |

## 5.5 Rigid-soft collision dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| RS1 candidate generation | Direct nested loops over rigid-soft node and rigid-soft edge candidates each iter. | Build candidate layouts and run WGSL node/edge broadphase masks + compact active pairs. | Same plus broadphase replay between cadence frames when replay criteria met. | out: compact node/edge pair sets | `gpu-lab.js` L4648-L4661; `stepRigidSoftCollisionGpuOnly.js` L3025-L3239 |
| RS2 narrowphase AABB filter | CPU contact test effectively done inline via resolvers. | Optional WGSL node/edge AABB probe + filtered pair compaction. | **Fast skips node+edge AABB probe readback paths** and marks skipped-fast sources. | out: filtered active pairs | `stepRigidSoftCollisionGpuOnly.js` L3107-L3174, L3249-L3278 |
| RS3 response apply | CPU resolves node/edge contacts directly. | WGSL response proposal (node+edge impulses), validate finite/signature, apply authoritative proposal or CPU fallback. | Same with replay-on-failure path and explicit fallback reasons/routes. | out: rigid+soft state updates, route ownership | `stepRigidSoftCollisionGpuOnly.js` L3414-L3574 |

## 5.6 Post-collision recovery dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| PR1 cluster kinematics | Compute cluster kinematics from soft nodes directly on CPU. | Prefer authoritative WGSL derived/probe kinematics when available; else CPU fallback. | Same flow. | out: post-collision kinematics map | `gpu-lab.js` L4786-L4791; `stepPostCollisionRecoveryGpuOnly.js` L760-L823 |
| PR2 projection | Project nodes toward cluster rigid motion field. | Same (possibly using WGSL-derived kinematics). | Same. | out: projected soft velocities | `gpu-lab.js` L4786-L4791; `stepPostCollisionRecoveryGpuOnly.js` L831-L838 |
| PR3 inside correction | CPU rigid-inside + membrane-inside correction passes. | Delegates to GPU-assisted inside correction module where available; CPU callbacks still supported. | Fast inside-correction path can skip some readback telemetry and still apply finite proposal routes. | out: inside correction counts | `gpu-lab.js` L4793-L4794; `stepPostCollisionRecoveryGpuOnly.js` L831-L848; `stepRigidInsideCorrectionGpuOnly.js` L691-L774, L954-L1198 |
| PR4 boundary cleanup | CPU bounce for rigid+soft. | Calls collision boundary pass (WGSL finite entries + CPU non-finite fallback). | Same. | out: boundary-consistent state | `gpu-lab.js` L4795-L4796; `stepPostCollisionRecoveryGpuOnly.js` L850-L867 |

## 5.7 Body-fluid injection dossier

| Substep | Baseline CPU | GPU-validated | GPU-fast | Inputs/Outputs | Citations |
|---|---|---|---|---|---|
| BI1 gather proposal | CPU `injectPoint` loops compute per-cell momentum deltas from rigid and soft contributors. | Build gather layout; CPU deterministic gather deltas retained for parity reference. | Build same gather layout; may avoid CPU parity work when fast authoritative gather exists. | out: `cellDeltaVx/cellDeltaVy` | `gpu-lab.js` L4947-L5036; `stepBodyFluidInjectionGpuOnly.js` L568-L639 |
| BI2 authority choice | CPU-only authoritative by default. | WGSL gather proposal accepted only when signature/finite checks match; otherwise CPU gather authoritative. | Fast accepts finite same-frame WGSL gather proposal (`wgsl-gather-authoritative-fast`) with parity skipped; explicit CPU-fast fallback otherwise. | out: chosen gather source route | `stepBodyFluidInjectionGpuOnly.js` L675-L730 |
| BI3 apply cell deltas | Apply bounded cell deltas to `(vx,vy)` + feedback maps; sum injected momentum. | Same apply path. | Same apply path. | out: field updates + `injectedMomentum` | `stepBodyFluidInjectionGpuOnly.js` L741-L760 |

---

## 6) Buffer + sync appendix (still important, but now contextualized)

You were right to call this out: readback/sync is only part of the story. Still, for perf attribution we need this list tied to CPU flow decisions.

### 6.1 Frame-level fluid sync points
- Full fluid readback copies (`copyBufferToBuffer` for `r,g,b,vx,vy`) and maps in `mapFullFluidReadback`.
- Fast health readback map for anomaly checks.
- Conditional full readback/writeback in fast mode.

**Citations:** `gpu-lab.js` L6186-L6231, L6356-L6423, L6479-L6486.

### 6.2 Stage-level major sync hotspots (gpu-only modules)
- Rigid step proposal readback arrays (vx/vy/omega/x/y/theta/carry) in `dispatchRigidStepProposal`.
- Soft-fluid coupling carry and cluster load proposals with readback-based proposal caches.
- Spring/area/membrane proposal readbacks for deterministic parity or finite validation.
- Rigid-soft broadphase/probe/response readbacks; fast mode now skips node+edge AABB probe readback paths.
- Boundary pass finite-entry WGSL with CPU non-finite fallbacks.

**Citations:**
- `stepRigidGpuOnly.js` L128-L237.
- `stepSoftFluidCouplingGpuOnly.js` L579-L633, L925-L1183.
- `stepSoftSpringsXpbdGpuOnly.js` L804-L834, L1490-L1590.
- `stepSoftAreaXpbdGpuOnly.js` L673-L692, L929-L935, L1273-L1369.
- `stepRigidSoftCollisionGpuOnly.js` L3025-L3278, L3414-L3574.
- `stepCollisionBoundaryGpuOnly.js` L316-L371.

---

## 7) Delivered expansion: exact schemas + symbol glossary + fallback matrix

You asked for the deeper pass, including **main CPU flow and all three modes**, not just GPU buffers/sync. This section delivers that with concrete schemas and trigger matrices.

---

## 8) Exact data schemas (CPU objects + GPU prep layouts)

## 8.1 Baseline in-memory runtime schema (CPU authoritative objects)

| Object | Shape / fields used in math | Where used in baseline math |
|---|---|---|
| `rigidBody` | `{ x,y,vx,vy,theta,omega,mass,inertia,r,edgeBodyMode[],edgeVelocityMode[],edgeMomentumCoupling[] }` | rigid fluid coupling/integration, rigid-soft/rigid-rigid collision, bounce, injection |
| `softNode` | `{ x,y,vx,vy,mass,r,clusterId,shapeMemoryWeight }` | soft fluid coupling, XPBD constraints, collisions, projection, membrane/deformation/recovery |
| `softSpring` tuple | `[nodeA,nodeB,restLen,edgeBodyMode,dyeModeRGB,edgeVelocityMode,momentum]` | spring XPBD, rigid-soft edge blocking, PASS/BLOCK behavior, momentum scaling |
| `softClusterLoop` | `{ clusterId, indices[] }` | area XPBD, membrane boundary/bend, membrane pressure, deformation metrics |
| `sim.softAreaRest / softAreaLambda` | `Map<clusterId, number>` | area XPBD rest-area and lambda state |
| `sim.softMembraneLoopState` | per-cluster edge/bend/shape refs and lambdas | membrane boundary, bend, shape-memory constraints |

**Citations:** `sim-server/public/gpu-lab.js` L4048-L4072, L4073-L4196, L4200-L4433, L4668-L4721, L3047-L3510.

## 8.2 Frame-fluid GPU buffer schema (shared all modes)

| Buffer group | Elements | Type | Role |
|---|---|---|---|
| Velocity ping-pong | `vx0,vx1,vy0,vy1` | `Float32Array` in storage buffers | advection/divergence/projection state |
| Pressure ping-pong + divergence | `pr0,pr1,div` | `Float32Array` | pressure solve |
| Dye ping-pong | `rr0,rr1,gg0,gg1,bb0,bb1` | `Float32Array` | RGB dye transport |
| Static masks | `viscMapGpu, obstacleMaskGpu, dyeModeMaskGpu` | float/u32 storage | viscosity and boundary/pass/block/eat rules |
| Readback buffers | `readR,readG,readB,readVx,readVy` | map-read buffers | CPU coupling/render source arrays |
| Fast health | `fastFluidHealthStats`, `fastFluidHealthReadback` | u32 stats buffer + map-read | fast anomaly gate before skip/readback decision |

**Citations:** `sim-server/public/gpu-lab.js` L6100-L6173, L6186-L6231, L6350-L6423.

## 8.3 Rigid step WGSL prep schema (`stepRigidGpuOnly.js`)

The proposal layout is a packed **14-float stride per rigid body** (`RIGID_LAYOUT_STRIDE_FLOATS = 14`).

| Offset | Field | Meaning |
|---|---|---|
| 0..5 | `vx, vy, omega, x, y, theta` | current rigid kinematic state |
| 6..8 | `ax, ay, alpha` | force/torque-derived accelerations |
| 9..11 | `swimX, swimY, swimTorque` | active swim injection terms |
| 12..13 | `damp, vmax` | viscosity response and speed cap |

Output arrays read back: `vx, vy, omega, x, y, theta, carry`.

**Citations:** `sim-server/public/runtime-solvers/stepRigidGpuOnly.js` L6-L44, L128-L237, L365-L389.

## 8.4 Soft-fluid coupling prep schema (`stepSoftFluidCouplingGpuOnly.js`)

Two aligned layouts are prepared:

### A) Topology/state layout
- `nodeX,nodeY,nodeVx,nodeVy,nodeMass,nodeMomentum`
- `nodeClusterId,nodeClusterSlot,nodeIsMembraneCluster`
- `clusterIds,clusterNodeOffsets,clusterNodeCount,clusterNodeIndices`

### B) CPU-sampled flow layout
- `fluidSampleVx,fluidSampleVy`
- `clusterLocalVx,clusterLocalVy`
- `sampleDeltaVx,sampleDeltaVy`
- `clusterCenterX,clusterCenterY`
- `localHoney`

These power carry proposals and cluster-load reductions.

**Citations:** `sim-server/public/runtime-solvers/stepSoftFluidCouplingGpuOnly.js` L68-L178, L185-L267, L844-L883.

## 8.5 Soft spring XPBD prep schema (`stepSoftSpringsXpbdGpuOnly.js`)

Plan/layout are deterministic and colorized for conflict-free reduction order.

| Group | Arrays |
|---|---|
| Active spring indexing | `activeSpringIndices`, `springNodeA`, `springNodeB`, `springRest` |
| Mass weights | `springInvMassA`, `springInvMassB` |
| Node-endpoint ownership | `nodeEndpointOffsets`, `endpointSpringIndices`, `endpointSignsI32` |
| Color scheduling | `springColors`, `springColorOffsets`, `springColorOrderedIndices` |
| Color-ordered mirrors | `springNodeAByColor`, `springNodeBByColor`, `springRestByColor`, `springInvMassAByColor`, `springInvMassBByColor` |
| Color endpoint ownership | `colorEndpointOffsets`, `endpointNodeIndicesByColor`, `endpointSpringIndicesByColor`, `endpointSignsI32ByColor` |

**Citations:** `sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js` L864-L969, L971-L1099.

## 8.6 Soft area XPBD prep schema (`stepSoftAreaXpbdGpuOnly.js`)

| Array | Meaning |
|---|---|
| `clusterOffsets` | CSR offsets into loop endpoints per cluster |
| `clusterNodeIndices` | node index list for all loop endpoints |
| `clusterNodeInvMass` | invMass for each endpoint node |
| `endpointClusterIndex` | endpoint → owning cluster map |
| `clusterRestArea` | rest area per cluster |
| `clusterLambda` | previous lambda per cluster |

**Citations:** `sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js` L1077-L1141.

## 8.7 Membrane pressure prep schema (`stepSoftMembranePressureGpuOnly.js`)

| Array | Meaning |
|---|---|
| `membraneOffsets` | CSR offsets per membrane loop |
| `loopIndices` | flat node indices for membrane loops |
| `loopMembraneIndex` | each loop index → membrane id |
| `nodeX,nodeY,nodeMass` | node state snapshot |
| `areaBase` | baseline area per membrane |
| `pressureGain` | pressure gain per membrane |
| `clusterId` | membrane cluster IDs |

**Citations:** `sim-server/public/runtime-solvers/stepSoftMembranePressureGpuOnly.js` L167-L240.

## 8.8 Body-fluid injection prep + gather schema (`stepBodyFluidInjectionGpuOnly.js`)

### A) Point prep layout
- `pointX,pointY,pointVx,pointVy`
- `localFluidX,localFluidY`
- `pointMass,pointRadius`
- `swimInjectX,swimInjectY`
- `pointMomentumScale`

### B) Gather layout
- `pointRelX,pointRelY,pointScale,pointRadius`
- `cellOffsets` (CSR cells)
- `contribPointIndex,contribWeight`
- `contributionCount`

These are reduced into per-cell `cellDeltaVx/cellDeltaVy` then applied to fluid fields.

**Citations:** `sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js` L139-L250, L256-L352, L355-L393.

## 8.9 Rigid-soft collision WGSL layout schema (`stepRigidSoftCollisionGpuOnly.js`)

### Pair layout
- `nodePairRigidIndex,nodePairNodeIndex`
- `edgePairRigidIndex,edgePairSpringIndex,edgePairNodeAIndex,edgePairNodeBIndex`

### Scene layout (narrowphase inputs)
- rigid state arrays: `rigidX,Y,Theta,Vx,Vy,Omega,InvMass,InvInertia,min/max AABB`
- rigid polygon geometry: `rigidVertexStart,rigidVertexX,rigidVertexY`
- soft state arrays: `nodeX,Y,Vx,Vy,R,InvMass`
- spring arrays: `springNodeA,springNodeB,springRestLen`

**Citations:** `sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js` L531-L583, L586-L739.

---

## 9) Symbol glossary (equations ↔ code variables)

| Symbol | Meaning | Code names (examples) | Units / domain | Main stages |
|---|---|---|---|---|
| `dt` | simulation time step | `dt`, `dtRaw` | seconds/frame | all integration/constraint passes |
| `dtNorm` | normalized dt scale for heuristic terms | `dtNorm` | unitless | swim and stabilizer terms |
| `C` | constraint residual | `C` in spring/area XPBD | geometry units | spring/area/membrane XPBD |
| `alpha` | XPBD compliance scaling | `alpha` | unit-adjusted compliance | spring/area XPBD |
| `lambdaPrev/lambdaNext` | accumulated XPBD multiplier | `lambdaPrev`, `lambdaNext` | impulse-like scalar | spring/area/membrane constraints |
| `dl` | lambda increment for current solve step | `dl`, `dlRaw` | scalar | XPBD substeps |
| `w` / `invMass` | inverse mass weighting | `wA,wB,invMass,nodeInvMass` | 1/mass | constraints/collisions |
| `sumWGrad2` | denominator term for area XPBD solve | `sumWGrad2` | scalar | area XPBD |
| `nx,ny` | contact/constraint normal direction | `nx`,`ny` | unit vector | collisions, membrane pressure |
| `ax,ay,alpha` | linear/angular acceleration from fluid coupling | `ax`,`ay`,`alpha` | vel/frame, angVel/frame | rigid and cluster coupling |
| `omega` | angular velocity | `omega`, `clusterOmega` | rad/frame (normalized) | rigid + cluster dynamics |
| `carry` | fluid-driven velocity transfer metric | `carryX/carryY`, `softCarryTransfer`, `rigidCarryTransfer` | velocity-like | fluid coupling/injection telemetry |
| `err` (pressure) | membrane area error ratio | `err` | normalized ratio [-0.65,0.65] | membrane pressure |
| `gain` | pressure correction gain | `gain`, `pressureGain` | scalar | membrane pressure |

**Citations:**
- `gpu-lab.js` L3972-L4196, L3047-L3217, L3510-L3600.
- `stepSoftSpringsXpbdGpuOnly.js` L95-L101, L1645-L1650.
- `stepSoftAreaXpbdGpuOnly.js` L119-L125, L1077-L1141.
- `stepSoftMembranePressureGpuOnly.js` L46-L49, L110-L113.

---

## 10) Fallback trigger matrix (all 3 flows, explicit route ownership)

> Baseline is CPU-authoritative by design, so “fallback” there means not applicable. Matrix below focuses on gpu-only validated/fast route transitions.

| Stage | Trigger condition | GPU-validated route | GPU-fast route | Ownership consequence | Citations |
|---|---|---|---|---|---|
| Rigid step | WGSL proposal unavailable, non-finite, or signature mismatch vs prepared frame | `cpu-rigid-step-authoritative` / `cpu-rigid-step-authoritative-fallback` | `cpu-rigid-step-authoritative-fast` or WGSL fast route if ready | CPU applies rigid state updates | `stepRigidGpuOnly.js` L431-L485 |
| Soft fluid coupling | Carry proposal non-finite or cluster-load parity mismatch (validated) | keeps CPU carry/cluster authoritative | fast can still accept finite WGSL proposal; else `cpu-carry+cluster-authoritative-fast-fallback` | CPU stays authoritative unless finite fast proposal accepted | `stepSoftFluidCouplingGpuOnly.js` L925-L933, L1162-L1183, L1326-L1328 |
| Soft spring XPBD | Proposal pipeline error or no authoritative replay match | `cpu-fallback-authoritative` path remains | same, but can use `wgsl-velocity-authoritative-fast` when fresh proposal passes | CPU iterations remain source of truth unless authoritative proposal accepted | `stepSoftSpringsXpbdGpuOnly.js` L1459-L1462, L1597-L1606 |
| Soft area XPBD | Non-finite authoritative replay/proposal | `cpu-fallback` | `cpu-fallback` (fast also) | CPU area solve applies | `stepSoftAreaXpbdGpuOnly.js` L1259-L1260, L1363, L1388 |
| Soft integrate | WGSL dispatch/readback error | `cpu-fallback-authoritative` | same | CPU integrates nodes | `stepSoftIntegrateGpuOnly.js` L423-L437 |
| Rigid post-integrate | WGSL error/unavailable | `cpu-fallback` | `cpu-fallback` | CPU clamps rigid velocity/omega | `stepRigidPostIntegrateGpuOnly.js` L222-L231 |
| Boundary pass | Non-finite subset exists but WGSL available | `wgsl-partial` + CPU only on non-finite entries | same | Mixed ownership (WGSL finite, CPU non-finite) | `stepCollisionBoundaryGpuOnly.js` L354-L360 |
| Boundary pass | WGSL execution error/unavailable | `cpu-fallback` | `cpu-fallback` | CPU owns full boundary pass | `stepCollisionBoundaryGpuOnly.js` L367-L371 |
| Rigid-soft narrowphase | Fast mode + cadence/replay path | validated runs probe path | `wgsl-rigid-soft-node-aabb-probe-skipped-fast` and edge equivalent | Fast skips probe readback telemetry stage | `stepRigidSoftCollisionGpuOnly.js` L3108-L3119, L3250-L3258 |
| Rigid-soft response | Proposal invalid and replay invalid | CPU fallback route | CPU fallback route (fast too) | CPU hard fallback response ownership | `stepRigidSoftCollisionGpuOnly.js` L3522-L3529, L3558-L3566 |
| Body-fluid injection | Gather proposal missing/non-finite | `cpu-gather-authoritative` | `cpu-gather-fallback-fast` | CPU gathers cell deltas | `stepBodyFluidInjectionGpuOnly.js` L702-L723 |
| Body-fluid injection | Gather proposal finite and matching | `wgsl-gather-authoritative` | `wgsl-gather-authoritative-fast` | WGSL proposal becomes authoritative gather source | `stepBodyFluidInjectionGpuOnly.js` L675-L707 |
| Soft rest recovery | Probe/proposal did not run or failed | `cpu-rest-recovery-authoritative` | same | CPU `recoverSoftSpringRests` remains authoritative | `stepSoftRestRecoveryGpuOnly.js` L622-L632, L655-L663 |

---

## 11) Timing-budget template (for direct attribution runs)

Use this per-stage template in benchmark notes so timing data lines up with this doc:

| Stage | CPU prep ms | GPU dispatch ms | Readback map ms | Decode/apply ms | Route | Notes |
|---|---:|---:|---:|---:|---|---|
| Fluid core |  |  |  |  | full-readback / resident-skip | include fast health stats |
| Rigid step |  |  |  |  | wgsl / cpu-fallback | include signature match flag |
| Soft fluid coupling |  |  |  |  | wgsl carry/cluster vs cpu | include mismatch counts |
| Spring XPBD |  |  |  |  | authoritative/replay/fallback | include proposal epoch delta |
| Area XPBD |  |  |  |  | probe/proposal/replay | include finite summary |
| Rigid-soft collision |  |  |  |  | replay/probe-skipped/fallback | include pair counts |
| Boundary |  |  |  |  | wgsl-partial/cpu-fallback | include non-finite entry count |
| Body-fluid injection |  |  |  |  | wgsl-gather/cpu-gather | include contribution count |

This makes before/after optimization runs auditable without ambiguity.

---

## 12) Equation cards (term-by-term mapping to code)

These cards are deliberately explicit so we can reason from algebra ↔ implementation without hand-waving.

## 12.1 Equation card — rigid fluid coupling + rigid integration

### Equation block
1. Relative fluid velocity at sample point:
   - `rel = u_fluid - u_rigid_local`
2. Drag force contribution:
   - `f = rel * dragK * honey * momentumScale`
3. Rigid acceleration:
   - `a = Σf / m`, `alpha = Σ(r × f) / I`
4. Velocity + swim update:
   - `v <- v + a*dt*60 + swim*dtNorm`
   - `ω <- ω + alpha*dt*60 + swimTorque*dtNorm`
5. Damping/cap:
   - `v <- damp(v)`, `|v| <= vmax`, `ω` clamped
6. Position integration:
   - `x <- x + vx*dt*22`, `y <- y + vy*dt*28`, `θ <- θ + ω*dt*60`

### Code mapping
| Equation term | Baseline CPU vars | GPU-validated/GPU-fast vars |
|---|---|---|
| `rel` | `relX`, `relY` | `relX`, `relY` |
| `f` | `fpx`, `fpy` | `fpx`, `fpy` |
| `a, alpha` | `ax`, `ay`, `alpha` | same + WGSL proposal buffers |
| integration | `b.vx`, `b.vy`, `b.omega`, `b.x`, `b.y`, `b.theta` | same, or WGSL authoritative arrays applied |

### Mode behavior
- **Baseline:** CPU authoritative always.
- **GPU-validated:** CPU reference + optional WGSL authoritative apply when proposal signature/finite checks pass.
- **GPU-fast:** same equations, but fast authoritative route is preferred when proposal is fresh/finite.

**Citations:**
- Baseline: `gpu-lab.js` L4150-L4153, L4164-L4175, L4192-L4194.
- GPU path: `stepRigidGpuOnly.js` L128-L237, L431-L485.

## 12.2 Equation card — soft fluid coupling (node + cluster)

### Equation block
1. Node fluid-vs-cluster relative term:
   - `Δu_cluster = u_fluid - u_cluster_local`
2. Force and carries:
   - `F = Δu_cluster * dragK * honey * flowCoupling * mass`
   - `carry = F / mass`
   - `localCarry = (u_fluid - u_node) * dragK * honey * invMass * flowCoupling * localShare`
3. Cluster load accumulation:
   - `cluster.force += F`
   - `cluster.torque += r × F`
4. Cluster acceleration projection to nodes:
   - `v_node += (a_cluster + alpha_cluster × r) * flowShare`
5. Coherence controls:
   - tug toward cluster pull, relative velocity damping, rigid-motion projection

### Mode behavior
- **Baseline:** all above computed in CPU loops.
- **GPU-validated:** same CPU reference retained; WGSL proposals compared and optionally consumed.
- **GPU-fast:** can shortcut to finite WGSL carry/cluster proposals and skip strict parity mismatch gating.

**Citations:**
- Baseline: `gpu-lab.js` L4303, L4309, L4316-L4318, L4369, L4404.
- GPU path: `stepSoftFluidCouplingGpuOnly.js` L1088-L1094, L1115-L1117, L1209-L1211, L925-L933, L1162-L1183.

## 12.3 Equation card — spring XPBD

### Equation block
1. Predicted endpoints:
   - `pA = xA + vA*dtPos`, `pB = xB + vB*dtPos`
2. Constraint residual:
   - `C = clamp(|pB-pA| - rest, ±strainCap)`
3. XPBD delta-lambda:
   - `dl = (-C - alpha*lambdaPrev) / (wSum + alpha)`
   - `lambdaNext = clamp(lambdaPrev + dl, [-20,20])`
4. Velocity correction:
   - `vA += (-wA*dl*n)/dtPos`
   - `vB += ( wB*dl*n)/dtPos`

### Mode behavior
- **Baseline:** CPU Gauss-Seidel XPBD iterations.
- **GPU-validated:** WGSL proposal + deterministic CPU parity reduction.
- **GPU-fast:** can replay fresh proposal by epoch/signature and skip probe telemetry.

**Citations:**
- Baseline: `gpu-lab.js` L3080, L3088, L3090, L3099.
- WGSL proposal formula: `stepSoftSpringsXpbdGpuOnly.js` L96-L98.
- Fast/validated route handling: `stepSoftSpringsXpbdGpuOnly.js` L1459-L1462, L1597-L1606.

## 12.4 Equation card — area XPBD

### Equation block
1. Predicted polygon area:
   - `A = signedAreaPredicted(...)`
2. Residual:
   - `C = A - A_rest`
3. Gradient and denominator:
   - `sumWGrad2 = Σ w_i * |∇_i A|²`
4. XPBD update:
   - `dl = (-C - alpha*lambdaPrev)/(sumWGrad2 + alpha)`
   - clamp `dl`, clamp `lambda`
5. Velocity correction:
   - `v_i += (w_i * ∇_i A * dl)/dtPos`

### Mode behavior
- **Baseline:** CPU reference solve.
- **GPU-validated:** WGSL probe/lambda/velocity proposals with parity checks.
- **GPU-fast:** probe may be skipped; replay may be accepted by epoch/signature when finite.

**Citations:**
- Baseline: `gpu-lab.js` L3162, L3182, L3188, L3198.
- WGSL formula: `stepSoftAreaXpbdGpuOnly.js` L120-L124.
- Fast replay/fallback: `stepSoftAreaXpbdGpuOnly.js` L1230-L1260, L1276, L1363-L1388.

## 12.5 Equation card — membrane pressure

### Equation block
1. Normalized area error:
   - `err = clamp((areaBase - areaNow)/areaBase, [-0.65, 0.65])`
2. Gain scaling:
   - `gain = pressureGain * (1 + min(1.4, abs(err)*2.2))`
3. Impulse:
   - `impulse = err * gain * dtPos * invMass`
4. Apply along outward normal (+ optional radial damping)

### Mode behavior
- **Baseline:** CPU computes area + normal impulse directly.
- **GPU-validated:** WGSL area probe + velocity proposal parity.
- **GPU-fast:** finite WGSL proposal can be accepted with reduced parity burden.

**Citations:**
- Baseline: `gpu-lab.js` L3535, L3540, L3576.
- WGSL formula: `stepSoftMembranePressureGpuOnly.js` L46, L111, L113.

## 12.6 Equation card — collision impulses (CPU reference equations)

### Circle/circle (soft-soft node-node)
- penetration correction proportional to inverse masses
- relative normal velocity `vn = (vb-va)·n`
- impulse scalar `j = -(1+e)*vn/(invA+invB)` when `vn < 0`

### Point/segment style (rigid-soft edge and soft-node-edge)
- closest point on segment
- penetration pushout along normal
- scalar impulse `j = -(1+e)*vn` applied to node/rigid endpoint velocities

### Mode behavior
- **Baseline:** these CPU equations run inline.
- **GPU-validated/fast:** WGSL proposals may be authoritative for some passes; hard CPU fallback still uses these equations/semantics.

**Citations:** `gpu-lab.js` L2246, L2264-L2266, L2312-L2314, L2350-L2352, L4668-L4721.

## 12.7 Equation card — boundary bounce

### Equation block
For each axis:
- clamp position to `[r, n-r]`
- if velocity points outward through boundary, reflect+damp:
  - `v = -v * damping`
- angular damping on impact:
  - `omega *= 0.9`

### Mode behavior
- **Baseline:** CPU `applyBounceBoundary` for all entries.
- **GPU-validated/fast:** boundary pass may be WGSL for finite entries, CPU for non-finite or fallback.

**Citations:** `gpu-lab.js` L2204-L2228; `stepCollisionBoundaryGpuOnly.js` L300-L371.

## 12.8 Equation card — body→fluid injection gather/apply

### Equation block
1. Relative injection velocity:
   - `rel = clamp(p_body - u_localFluid + swimInject, couplingLimit)`
2. Scale:
   - `scale = clamp(feedbackK * max(0.1,mass) * momentumScale, couplingLimit)`
3. Per-cell weighted impulse:
   - `j = clamp(rel * scale * weight, couplingLimit)`
4. Apply to fluid fields:
   - `vx[cell] += jx`, `vy[cell] += jy` (clamped)

### Mode behavior
- **Baseline:** CPU `injectPoint` computes and applies directly.
- **GPU-validated:** WGSL gather proposal compared with CPU parity before authority.
- **GPU-fast:** finite same-frame WGSL gather can be authoritative; else explicit fast CPU fallback route.

**Citations:**
- Baseline: `gpu-lab.js` L4956, L4960, L4971.
- GPU gather layout/apply: `stepBodyFluidInjectionGpuOnly.js` L289, L291, L375, L355-L393, L675-L723.

## 12.9 Card-level quick mode map

| Card | Baseline authority | GPU-validated authority policy | GPU-fast authority policy |
|---|---|---|---|
| Rigid step | CPU | WGSL if signature+finite, else CPU | same policy, faster acceptance path |
| Soft fluid coupling | CPU | WGSL proposals parity-checked vs CPU | finite WGSL shortcut allowed |
| Spring XPBD | CPU | WGSL proposal + deterministic parity | replay + reduced probe/parity overhead |
| Area XPBD | CPU | probe/lambda/velocity proposals + parity | probe skips/replay windows in fast |
| Membrane pressure | CPU | WGSL proposal with parity checks | finite proposal can bypass parity |
| Collisions | CPU | mixed WGSL proposal + CPU fallback | same, plus rigid-soft fast skips/replay |
| Boundary | CPU | WGSL finite + CPU non-finite | same |
| Injection | CPU | WGSL gather vs CPU parity | finite WGSL gather fast-authoritative |
