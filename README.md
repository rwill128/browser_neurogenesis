# Browser Neurogenesis

Browser Neurogenesis is a research project exploring the building blocks of browser-based artificial life: real-time fluid dynamics, physical embodiment, deformable structures, and morphology authoring.

**[Launch the Full Simulation](https://rwill128.github.io/browser_neurogenesis/full-sim/?mode=default)**

The full simulation runs directly in a desktop browser and starts with the default large world. The focused GPU experiments remain available separately.

## What You Can Explore

- WebGPU fluid simulation with tunable viscosity, flow, and obstacle fields
- Rigid and deformable soft bodies coupled to the fluid environment
- Adjustable body-fluid momentum and rotational coupling
- Deterministic scenarios for comparing runtime behavior
- Multiple collision, stabilization, and solver configurations
- Runtime telemetry for performance, stability, and fallback behavior
- CreatureSpec import/export for moving structures between tools

## Live Experiments

- **[Full Simulation](https://rwill128.github.io/browser_neurogenesis/full-sim/?mode=default)** - Run the full-size evolving world with creatures, environmental fields, configurable dynamics, and inspection tools.
- **[GPU Lab](https://rwill128.github.io/browser_neurogenesis/gpu-lab.html)** - Experiment with body shapes, dimensions, fluid coupling, collision behavior, and solver settings.
- **[Mesh Lab](https://rwill128.github.io/browser_neurogenesis/mesh-lab.html)** - Design and inspect creature geometry.
- **[Interaction Lab](https://rwill128.github.io/browser_neurogenesis/interaction-lab.html)** - Exercise focused body and environment interaction scenarios.

## How It Works

The runtime combines a grid-based fluid field with rigid-body and soft-body solvers. Creature structures are physical geometry rather than animated sprites: their shape, material behavior, constraints, and solver configuration determine how they move and deform.

The project includes JavaScript reference implementations, WebGPU solver paths, selected WebAssembly acceleration, deterministic scenarios, and a Node-based regression harness. The browser UI exposes solver choices and instrumentation so behavior can be compared rather than treated as a black box.

## Repository Map

- `sim-server/public/` - Static browser labs and runtime modules
- `sim-server/public/full-sim/` - Published snapshot of the full artificial-life simulation
- `sim-server/public/runtime-solvers/` - GPU-only runtime solver stages
- `sim-server/public/wasm/` - WebAssembly acceleration modules
- `node-harness/` - Deterministic simulations, fixtures, benchmarks, and regression tests
- `native-core/` - Native physics experiments and WebAssembly source
- `docs/` - Design notes, profiling results, and technical specifications
- `scripts/build-lab-pages.mjs` - GitHub Pages bundle builder

## Local Development

The hosted labs are static and do not require the simulation server.

```bash
node scripts/build-lab-pages.mjs
npx serve dist/labs
```

Run the standalone CreatureSpec contract tests with:

```bash
node --test node-harness/tests/creatureSpecV2Contract.test.mjs
```

The GitHub Pages workflow rebuilds and deploys the static bundle when relevant files are pushed to `main`.

## Current Focus

This is an active research project rather than a finished game. Current work focuses on stable fluid-body coupling, deterministic evaluation, and morphology representations that can support future evolution experiments while remaining observable and debuggable.
