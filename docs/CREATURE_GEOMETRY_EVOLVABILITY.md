# Creature Geometry + Evolvability Contract

This document defines the **implementation-adjacent geometry traits** used by `sim-server/public/creature-spec.js`, with emphasis on what is currently fixed and what is already positioned to become mutable/evolvable.

## Scope and goals

- Keep creature geometry deterministic and importable (`creature-spec.v2`).
- Encode morphology in a way the runtime solver can consume directly (no hidden editor-only assumptions).
- Preserve room for mutation by storing per-body/per-edge/per-vertex traits as explicit fields.

Primary API surface:
- `createCreatureSpecFromMesh(mesh, options)`
- `parseCreatureSpec(jsonText)`
- `buildBodiesFromCreatureSpec(spec, n, controls)`
- `buildMembraneRingsFromSoftField({...})`

---

## Trait levels: body vs segment(line) vs vertex

All trait levels are represented as explicit JSON fields in `creature-spec.v2`, so mutations can be applied as data transforms (without introducing hidden runtime/editor state).

### 1) Body-level traits

These define solver behavior at a coarse level.

**Rigid body fields**
- `hull`, `subHulls`, `compoundId`
- `insideCorrectionEnabled`
- `consumeDyeRGB`

**Soft body fields**
- `solverMode`: `spring` or `membrane`
- `restArea`, `pressureGain`, `radialDamping`, `shapeMemoryGain`
- `insideCorrectionEnabled`

**Evolvability note:** body-level mutation can safely toggle modes/coefficients first (lower combinatorial risk than topology mutation).

### 2) Segment/line-level traits (edges/springs)

These are the highest-leverage traits for permeability, collision semantics, and structural stiffness.

**Rigid edge arrays (indexed by hull edge):**
- `edgeBodyMode` (default compacted away when all blocking)
- `edgeDyeMode` (RGB, channel-wise pass/deflect/absorb)
- `edgePermeabilityRGB` (RGB channel permeability mask)

**Soft spring tuples:**
- `[a, b, restLength, edgeBodyMode, edgeDyeModeRGB]`
- Optional reinforcements from:
  - `softBoundaryRingSprings`
  - `softSeamWeldSprings`
  - `softCrossBeams` (+ max span guard)

**Evolvability note:** edge arrays/tuples are explicit and already normalized on import, so mutators can target per-edge properties without changing parser format.

### 3) Vertex-level traits

These traits govern localized deformability.

**Soft node fields:**
- `x`, `y`
- `shapeMemoryWeight` (0..1 clamp)

When using membrane field authoring, `shapeMemoryWeight` is sampled from `fields.membraneShapeMap` and stored per vertex.

**Evolvability note:** per-vertex mutation of `shapeMemoryWeight` is a low-risk path to heterogeneous tissues before introducing new solver modes.

---

## Runtime normalization characteristics (implementation-adjacent)

`buildBodiesFromCreatureSpec` applies predictable normalization/clamping to keep segment/vertex mutations safe:

- **Rigid edge body modes** are normalized to binary semantics (`1` = block, anything else = pass/`0`).
- **Rigid edge dye modes** are normalized per channel to enum `{0:pass, 1:deflect, 2:absorb}`; invalid values clamp to safe defaults.
- **Rigid edge permeability** is normalized per channel into binary-like behavior (`<=0` blocked, `>0` pass).
- **Rigid consume-dye masks** are normalized to binary RGB masks.
- **Soft spring tuples** preserve indices `[a,b,rest,edgeBodyMode,edgeDyeModeRGB]` and normalize `edgeBodyMode` with the same binary rule plus the same per-channel dye-mode normalization used by rigid edges. When tuple slots are omitted, importer defaults are deterministic: `edgeBodyMode -> 0 (pass)` and `edgeDyeModeRGB -> [1,1,1] (deflect)`.
- **Hybrid joints** are repaired when degenerate (`edgeA===edgeB`) and oversized `restA/restB` values are capped against rigid-body scale.
- **Soft vertex `shapeMemoryWeight`** is clamped to `0..1` and imported/exported as explicit per-node scalar state.

These characteristics are important for future mutators: exploratory edits can be aggressive, but import/build still projects the result into solver-safe ranges.

### Solver mode forward-compatibility rule

`solverMode` is intentionally normalized as:
- `"membrane"` → membrane path (plus membrane cluster metadata)
- anything else (`"spring"` or unknown future strings) → spring path

This keeps imports deterministic when experimental/future mutators emit unknown solver tags: simulation falls back to stable spring semantics instead of entering a partial membrane state.

## Deterministic guardrails already in place

Current code enforces several anti-chaos rules that should remain active while mutation expands:

- Unsupported schema versions rejected in `parseCreatureSpec`.
- Membrane imports sanitized to **perimeter-only rings** (drops interior chords).
- Concave membrane perimeter order preserved on import (not convexified).
- Degenerate hybrid joints repaired when `edgeA === edgeB`.
- Oversized hybrid rest lengths clamped.
- Unknown dye modes clamped to safe defaults.
- Long-span reinforcement springs bounded (`*MaxSpanFactor` guardrails).

These guardrails allow richer mutation while preserving solver stability.

### Deterministic verification hooks (current test coverage)

`node-harness/tests/creatureSpecV2Contract.test.mjs` contains explicit invariants for the evolvability contract above, including:

- unknown/experimental `solverMode` fallback to spring semantics,
- ring-only membrane perimeter sanitization,
- per-edge trait normalization and sparse backfilling,
- malformed soft-spring dye tuples channel-wise clamping/defaulting (`[pass|deflect|absorb]` enum),
- per-vertex `shapeMemoryWeight` clamping.

Run targeted contract checks with:

```bash
node --test node-harness/tests/creatureSpecV2Contract.test.mjs
```

---

## Mutation-ready knobs (existing fields/options)

The following are concrete, implementation-level mutation targets that already have import/export paths:

1. **Membrane edge spacing map**
   - `fields.membraneEdgeMap`
   - Controls perimeter sampling spacing (`membraneMinEdgeLength` / `membraneMaxEdgeLength`).

2. **Membrane per-vertex shape memory map**
   - `fields.membraneShapeMap`
   - Produces per-node `shapeMemoryWeight`.

3. **Soft topology reinforcement policy**
   - `softBoundaryRingSprings`, `softBoundaryRingStride`, `softBoundaryRingMaxSpanFactor`
   - `softSeamWeldSprings`
   - `softCrossBeamMaxSpanFactor`

4. **Per-edge transport/collision behavior**
   - Rigid: `edgeBodyMode`, `edgeDyeMode`, `edgePermeabilityRGB`
   - Soft springs: tuple slots for `edgeBodyMode` and `edgeDyeModeRGB`

5. **Hybrid coupling geometry**
   - `hybridJoints` (`edgeA`, `edgeB`, `restA`, `restB`)

---

## Mutation patch examples (implementation-adjacent)

These are intentionally tiny JSON-level transforms that mutate existing traits without changing schema shape.

### A) Body-level scalar mutation (soft membrane pressure)

```json
{
  "op": "replace",
  "path": "/softBodies/0/pressureGain",
  "value": 0.11
}
```

### B) Segment-level mutation (rigid edge permeability + dye behavior)

```json
[
  { "op": "replace", "path": "/rigidBodies/0/edgePermeabilityRGB/2", "value": [1, 0, 1] },
  { "op": "replace", "path": "/rigidBodies/0/edgeDyeMode/2", "value": [2, 1, 0] }
]
```

### C) Vertex-level mutation (localized deformability)

```json
{
  "op": "replace",
  "path": "/softBodies/0/nodes/7/shapeMemoryWeight",
  "value": 0.35
}
```

### Why these patches are safe to explore

On import/build, these values are projected into deterministic ranges:
- `shapeMemoryWeight` clamps to `0..1`
- unknown dye channels clamp to `{pass, deflect, absorb}` enum defaults
- permeability channels normalize to pass/block semantics

So mutation code can stay simple and expressive, while build-time normalization provides solver safety.

## Mutation surface matrix (segment/line/vertex focus)

| Trait surface | Current storage | Runtime normalization | First evolvable move |
| --- | --- | --- | --- |
| Segment body collision (`edgeBodyMode`) | Rigid per-edge arrays + soft spring tuple slot `[3]` | binary coercion (`1` block, else pass) | mutate sparse subset of edges to pass-through |
| Segment dye interaction (`edgeDyeMode`) | Rigid per-edge RGB arrays + soft spring tuple slot `[4]` | per-channel enum clamp to `{0,1,2}` | channel-specific deflect/pass/absorb patterning |
| Segment permeability (`edgePermeabilityRGB`) | Rigid per-edge RGB arrays | channel-wise pass/block normalization (`>0` pass) | morphogen gate experiments by color channel |
| Vertex deformability (`shapeMemoryWeight`) | Soft `nodes[i].shapeMemoryWeight` | hard clamp to `0..1`, missing => `1` | regional stiffness maps and local tissue zones |

This matrix is intentionally implementation-adjacent: each row points to a concrete JSON field that already round-trips through `createCreatureSpecFromMesh` → `buildBodiesFromCreatureSpec` with deterministic projection.

## Practical evolvability progression (recommended)

1. Mutate scalar coefficients first (`shapeMemoryGain`, `pressureGain`, etc.).
2. Then mutate per-vertex weights (`shapeMemoryWeight`).
3. Then mutate edge semantics/permeability arrays.
4. Finally mutate topology (spring add/remove, perimeter rewiring), while preserving existing span and membrane-ring guardrails.

This order maximizes phenotype diversity while minimizing catastrophic solver regressions.
