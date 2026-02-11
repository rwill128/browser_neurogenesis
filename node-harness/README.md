# Node Harness (Bootstrap)

This is the first Node-runnable simulation harness scaffold.

## Run

```bash
node node-harness/runScenario.mjs --scenario micro_stability --seed 42 --steps 300 --out ./artifacts
```

Outputs a JSON timeline artifact under `./artifacts`.

## Why this exists

- Gives us deterministic, filesystem-friendly scenario runs in Node now.
- Establishes the harness interface we will migrate real simulation logic into.

## Current status

- ✅ Deterministic seeded micro-world runner
- ✅ Structured timeline JSON artifact export
- 🚧 Not yet full parity with browser simulation logic

Next: progressively extract real engine logic from `js/simulation.js` + classes into shared core used by browser and Node.
