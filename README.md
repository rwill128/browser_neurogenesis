# Browser Neurogenesis

*A living world where structure, flow, and adaptation co-evolve in real time.*

Browser Neurogenesis is an experimental artificial-life simulation that blends **rigid-body constraints**, **soft-body deformation**, **fluid dynamics**, and **particle ecology** into one continuous system. Creatures are not pre-scripted sprites — they are physically embodied, energy-constrained, and mutation-driven organisms that survive (or fail) inside a dynamic environment.

## Core features

### 1) Hybrid body physics (rigid + soft)
- Soft bodies made from points + springs
- Rigid-constraint projection for true rigid edge behavior where required
- Stability guardrails to prevent runaway stretch/explosion

### 2) Fluid dynamics + viscosity landscapes
- Grid-based fluid field with tunable solver iterations
- Spatially varying viscosity landscapes
- Dye emitters and flow-coupled visual/ecological feedback

### 3) Particle and energy ecology
- Photosynthesis + local field effects
- Energy costs per node/function type
- Resource and density-gated reproduction controls

### 4) Evolutionary morphology
- Heritable blueprints and phenotype persistence
- Mutation pipeline focused on stable structure formation
- Triangle-primitive mutation paradigm for controlled structural exploration

### 5) Real-time observability and capture
- World status + telemetry endpoints
- Snapshot and frame timeline APIs
- Random creature portraits + short creature MP4 clips (with neighbors/fluid context)

### 6) Multi-world sim server
- Long-running worker-thread worlds
- Live browser rendering against active worlds
- Checkpoint/save/restore support via SQLite

---

## Architecture at a glance

- `js/` — browser runtime, simulation engine, UI, config
- `js/engine/` — shared world-step core and runtime helpers
- `js/classes/` — core physical/biological classes (SoftBody, Brain, etc.)
- `sim-server/` — authoritative server + world workers + API + capture endpoints
- `node-harness/` — deterministic scenario runs, soak tools, regressions, tests

The key design direction: **shared core logic across browser and Node tooling**, so experiments, regressions, and live behavior stay aligned.

---

## Quick start

### Browser mode
Open `index.html` in a browser (or run from your local static server setup).

### Sim server mode
```bash
cd sim-server
npm install
node server.mjs --port 8787 --scenario browser_default_big --seed 23
```

Then open:
- http://localhost:8787/

---

## API examples

```bash
# world status
curl -s http://localhost:8787/api/worlds/w0/status

# lightweight telemetry snapshot
curl -s "http://localhost:8787/api/worlds/w0/snapshot?mode=lite"

# capture a 5s creature clip with context
curl -s -X POST http://localhost:8787/api/worlds/w0/capture/creatureClip \
  -H 'content-type: application/json' \
  -d '{"durationSec":5,"zoomOutFactor":10,"includeFluid":true,"includeNeighbors":true}'
```

---

## Screenshots

> Current captures from active local runs.

### Creature portrait (context + fluid)
![Creature portrait](sim-server/data/captures/creature-w0-292-tick24900-1770953582142.png)

### Creature portrait (early-run morphology)
![Creature portrait 2](sim-server/data/captures/creature-w0-74-tick1994-1770949784964.png)

---

## Vision

The long-term vision is a **high-fidelity, evolvable synthetic ecology** that stays interactive and inspectable:
- stable enough for long-duration runs,
- expressive enough for surprising emergent diversity,
- instrumented enough for scientific debugging,
- and visual enough to remain emotionally compelling.

This is not just a simulation to watch.
It is a world to shape, stress, measure, and learn from.
