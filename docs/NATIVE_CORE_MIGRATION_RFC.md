# RFC: Native C++ Simulation Core (Strangler Migration)

Date: 2026-02-19
Owner: Rick + assistant
Status: Proposed (ready to execute)

## 1) Why now

Current JS/TS runtime quality is strong (physics feel + shape compiler), but scale target has changed.

- Current slowdown around ~200 creatures blocks evo goals.
- Target is >= two orders of magnitude higher population.
- Recent profiling/attribution shows frame cost concentrated in hot simulation stages, with GPU queue contention/sync overhead in current hybrid path.

Conclusion: keep behavior model, migrate core execution substrate.

## 2) Decision

Adopt a **strangler migration**:

- Keep in JS for now:
  - authoring/compiler/UI (Mesh Lab / GPU Lab controls)
  - scenario/config plumbing
  - visualization and experimentation surfaces
- Migrate to C++ incrementally:
  - deterministic hot simulation kernels
  - data-oriented broadphase/narrowphase/integration pipelines
  - optional native GPU path where it actually replaces CPU work and stays resident

No full rewrite upfront. Port bottleneck slices one by one behind stable contracts.

## 3) Migration principles

1. **Behavior first, then speed**: preserve sim “feel” using golden fixtures before claiming perf wins.
2. **Data-oriented SoA**: contiguous arrays, no object-per-entity hot loops.
3. **Residency honesty**: do not call work “GPU” if it still requires hot-path CPU ownership/readback.
4. **Measured gates only**: each slice must show before/after with variance.
5. **Reversible integration**: each migrated stage can be toggled back to JS.

## 4) Initial bottleneck targets (port order)

Based on current timing captures and harness attribution:

1) Collision pipeline (rigid-soft first)
- High impact stage in current timings.
- Port broadphase + narrowphase + response to C++ (single ownership).

2) Body-fluid injection / coupling hot loop
- Significant per-frame cost and memory traffic.
- Port gather/reduction + apply path with SoA buffers.

3) Integrate + post-integrate + boundary clamp
- Straightforward deterministic kernel set, good early native win.

4) Keep fluid solver strategy separate
- Treat fluid path as its own optimization track (possible GPU saturation/queue contention).

## 5) Interface architecture (bit-by-bit)

### Phase A: Native benchmark lane (no runtime integration risk)
- Build standalone C++ executable that consumes fixture snapshots and outputs metrics.
- Reuse node-harness fixture format for A/B comparison.

### Phase B: Runtime bridge lane
- Integrate via **Node-API addon** in sim-server worker path for low overhead.
- Contract: pass typed-array-backed SoA buffers into native step functions.
- Toggle per stage:
  - `nativeCollision`
  - `nativeCoupling`
  - `nativeIntegrate`

### Minimal C ABI for stage calls
- `step_collision(...)`
- `step_body_fluid_coupling(...)`
- `step_integrate(...)`

Each stage mutates shared SoA buffers in-place and returns telemetry counters.

## 6) Data contract v1 (SoA)

Core arrays (Float32/Int32/Uint32):
- Rigid: `x,y,vx,vy,theta,omega,mass,inertia,radius`
- Soft nodes: `x,y,vx,vy,mass,clusterId`
- Topology: spring endpoints, rest length, flags
- Collision candidates: compact pair buffers
- Fluid coupling fields: sampled force/reaction accumulators

Rule: hot stages consume/produce these buffers only; no per-step object graph walking.

## 7) Validation gates

Per migrated stage, required:

1. Determinism + parity
- Existing fixtures pass within explicit tolerances.

2. Performance gate
- Show >= meaningful speedup on targeted stage in fixed-step harness.
- Treat <1% as noise.

3. End-to-end guard
- No regression in baseline contracts and scenario behavior.

## 8) 2-week spike plan

### Week 1
- Day 1-2: freeze golden corpus + perf baseline dashboard.
- Day 3-4: implement SoA adapters from current runtime state.
- Day 5-7: port collision vertical slice (rigid-soft path first) + native microbench.

### Week 2
- Day 8-10: Node-API integration for collision slice behind toggle.
- Day 11-12: port integrate/post-integrate slice.
- Day 13: run A/B at scale tiers + variance report.
- Day 14: go/no-go checkpoint for full staged migration.

## 9) Go / No-Go criteria after spike

Go if all are true:
- Collision + integrate slices maintain expected behavior in golden corpus.
- Stage-level speedups are clear (> noise) and compound meaningfully.
- Integration overhead does not erase native gains.

No-Go if:
- Behavior drift is hard to control,
- Bridge overhead dominates,
- or stage wins are marginal.

## 10) Immediate next actions (start now)

1. Create `native-core/` skeleton (CMake + stage stubs + fixture loader).
2. Add `node-harness` export mode for SoA fixture dumps.
3. Implement first native collision benchmark against current fixture set.
4. Report first A/B numbers before expanding scope.

---

This plan intentionally avoids a risky all-at-once rewrite while directly attacking the hottest CPU bottlenecks with measurable milestones.
