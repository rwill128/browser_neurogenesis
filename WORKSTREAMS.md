# Workstream Split: Legacy vs GPU Track

This repository now has two explicit tracks.

## 1) Legacy simulation track

Purpose: existing browser/main-sim runtime and compatibility behavior.

Primary paths:
- `js/`
- `js/classes/`
- `js/engine/`
- `index.html`
- legacy-oriented harness tests under `node-harness/tests/*` (mostly `softBody*`, ecology, world-step, etc.)

## 2) GPU track (new project)

Purpose: GPU-first experimentation + authoring loop.

### 2a) GPU Lab submodule
Runtime fluid/body sandbox for rigid+soft+fluid interactions.

Primary paths:
- `sim-server/public/gpu-lab.html`
- `sim-server/public/gpu-lab.js`

### 2b) Mesh Lab submodule
Field-to-structure authoring + compiler + export/import contract.

Primary paths:
- `sim-server/public/mesh-lab.html`
- `sim-server/public/mesh-lab.js`
- `sim-server/public/field-to-structure-core.js`
- `sim-server/public/creature-spec.js`

## 3) Shared + test boundaries

Shared contract:
- `sim-server/public/creature-spec.js`

GPU-track tests (authoring/import/compiler focused):
- `node-harness/tests/creatureSpec.test.mjs`
- `node-harness/tests/fieldToStructureCompiler.test.mjs`

Legacy-focused tests remain in `node-harness/tests/*` and are run separately.

## 4) Command conventions

Run only GPU-track tests:
```bash
node node-harness/test-gpu-track.mjs
```

Run only legacy-track tests:
```bash
node node-harness/test-legacy-track.mjs
```

Run all tests:
```bash
node --test node-harness/tests/*.test.mjs
```

## 5) Guardrail for implementation sessions

When task scope says “GPU-lab work”, code edits should stay in:
- `sim-server/public/gpu-lab*`
- `sim-server/public/mesh-lab*`
- `sim-server/public/field-to-structure-core.js`
- `sim-server/public/creature-spec.js`

Avoid `js/classes/*` changes unless explicitly requested.
