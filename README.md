# Browser Neurogenesis

Browser Neurogenesis is an experimental artificial-life simulation where fluid dynamics, physical embodiment, and evolutionary morphology interact in real time.

**[Launch Browser Neurogenesis](https://rwill128.github.io/browser_neurogenesis/)**

The simulation runs directly in a WebGPU-capable desktop browser. It begins paused so you can choose a scenario and solver configuration before starting it.

## What You Can Explore

- WebGPU fluid simulation with tunable viscosity, flow, and obstacle fields
- Rigid and deformable soft bodies coupled to the fluid environment
- Energy-constrained creatures with heritable body structures
- Mutation and reproduction systems for evolving morphology
- Multiple collision, stabilization, and solver configurations
- Runtime telemetry for performance, stability, and fallback behavior
- CreatureSpec import/export for moving structures between tools

## Live Experiments

- **[GPU Lab](https://rwill128.github.io/browser_neurogenesis/gpu-lab.html)** - Run the primary fluid, physics, and artificial-life simulation.
- **[Mesh Lab](https://rwill128.github.io/browser_neurogenesis/mesh-lab.html)** - Design and inspect creature geometry.
- **[Interaction Lab](https://rwill128.github.io/browser_neurogenesis/interaction-lab.html)** - Exercise focused body and environment interaction scenarios.

## How It Works

The runtime combines a grid-based fluid field with rigid-body and soft-body solvers. Creatures are represented as physical structures rather than animated sprites: their geometry, material behavior, energy use, and control systems determine how they move and survive.

The project includes JavaScript reference implementations, WebGPU solver paths, selected WebAssembly acceleration, deterministic scenarios, and a Node-based regression harness. The browser UI exposes solver choices and instrumentation so behavior can be compared rather than treated as a black box.

## Repository Map

- `sim-server/public/` - Static browser labs and runtime modules
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

This is an active research project rather than a finished game. Current work focuses on stable fluid-body coupling, evolvable creature geometry, deterministic evaluation, and keeping increasingly complex simulated organisms observable and debuggable.
