# Node Harness

Node tools for deterministic scenario runs, artifact rendering, and interactive stepping.

## 1) Batch run a scenario to timeline JSON

```bash
node node-harness/runScenario.mjs \
  --scenario micro_one_creature_100 \
  --engine real \
  --seed 23 \
  --steps 180 \
  --dt 0.001 \
  --out ./artifacts
```

Outputs JSON under `./artifacts`.

> Safety default: surrogate mode (`mini`) is blocked unless explicitly enabled with `--allowMini`.

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
node node-harness/simCli.mjs --engine real --scenario micro_one_creature_100 --seed 23 --dt 0.001
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
- `set engine <mini|real>`
- `set scenario <name>`
- `set dt <value>`
- `set world <w> <h>`
- `set creatures <n>`
- `set particles <n>`

## Current status

- ✅ Deterministic seeded runs
- ✅ Timeline JSON + frame/video artifact pipeline
- ✅ Real-engine path (`--engine real`) using real simulation classes
- ✅ Interactive CLI control/query loop
- 🚧 Still in-progress toward full browser/headless parity
