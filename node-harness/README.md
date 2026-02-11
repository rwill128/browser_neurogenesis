# Node Harness

Node tools for deterministic scenario runs, artifact rendering, and interactive stepping.

## 1) Batch run a scenario to timeline JSON

```bash
node node-harness/runScenario.mjs \
  --scenario micro_one_creature_100 \
  --seed 23 \
  --steps 180 \
  --dt 0.001 \
  --out ./artifacts
```

Outputs JSON under `./artifacts`.

> Engine selection is removed: this harness always runs the real simulation code path.

## 2) Render timeline JSON into frames/video

```bash
node node-harness/renderTimelineFrames.mjs \
  --input ./artifacts/micro_one_creature_100-seed23-steps180.json \
  --out ./artifacts/frames-micro-one-real

ffmpeg -y -framerate 12 \
  -i ./artifacts/frames-micro-one-real/frame-%05d.ppm \
  -c:v libx264 -pix_fmt yuv420p \
  ./artifacts/micro_one_creature_100-real-seed23-steps180.mp4
```

## 3) Interactive CLI mode (step/play/rewind/query)

```bash
node node-harness/simCli.mjs --scenario micro_one_creature_100 --seed 23 --dt 0.001
```

Key commands:

- `step [n]` / `forward [n]`
- `play [hz] [batch]`
- `pause`
- `back [n]` (deterministic rewind via replay)
- `goto <tick>`
- `snapshot full [--out path]`
- `snapshot rect <x> <y> <w> <h> [--out path]`
- `snapshot creature <id> [--out path]`
- `snapshot fluid [x y w h] [--out path]`
- `save <path>`
- `load <path>`
- `set scenario <name>`
- `set dt <value>`
- `set world <w> <h>`
- `set creatures <n>`
- `set particles <n>`

## 4) Parity regression (browser-like vs headless real)

```bash
node node-harness/parityRegression.mjs \
  --scenario micro_one_creature_100 \
  --seed 23 \
  --steps 120 \
  --checkpointEvery 10
```

The script compares invariant checkpoints (population counts, structural counts, energy/fluid aggregates)
across a browser-like shared-core runner and the existing headless real runner.

## 5) Save/load regression (headless + browser-like real paths)

```bash
node node-harness/saveLoadRegression.mjs \
  --scenario micro_one_creature_100 \
  --seed 23 \
  --beforeSteps 30 \
  --afterSteps 60 \
  --runtime both
```

This validates end-to-end `save -> load -> continue` determinism across both
headless-real and browser-like-real runners.

## 6) Browser-default soak runner (repro without a live browser)

```bash
node node-harness/browserDefaultSoak.mjs \
  --seed 41 \
  --steps 20000 \
  --dt 0.01 \
  --logEvery 500 \
  --out /tmp/browser-default-soak
```

What it does:
- applies index.html-like defaults (world/population/particle emission)
- runs real shared-core stepping in node (with reproduction + floor maintenance)
- catches crashes and writes both a crash report and full snapshot for replay
- writes periodic checkpoints + final report even on successful runs
- includes growth/RL continuity telemetry in reports (`growthEvents`, `growthEnergySpent`, `rlTopologyResets`, etc.)

## 7) Unit tests (growth controls + topology-resize RL handling)

```bash
node --test node-harness/tests/*.test.mjs
```

## Current status

- ✅ Deterministic seeded runs
- ✅ Timeline JSON + frame/video artifact pipeline
- ✅ Always-real path using real simulation classes
- ✅ Interactive CLI control/query loop
- ✅ Cross-runtime save/load state snapshots (real engine path)
- 🚧 Still in-progress toward full browser/headless parity
