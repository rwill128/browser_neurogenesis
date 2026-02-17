# Interaction Pipeline (Capture-first, Analyze-second)

This folder scaffolds a resumable artifact workflow for Interaction Lab truth-table review.

## Stage A — scaffold a batch (one-time)

```bash
cd node-harness
node interaction-pipeline/scaffoldBatch.mjs --name truth-table-v1
```

Creates:

- `artifacts/interaction-pipeline/<timestamp>-truth-table-v1/`
  - `cases/<nn>-<id>/scenario.json`
  - `manifest.jsonl`
  - `verdicts.json` (PENDING)
  - `analysis.md` template

## Stage B — capture artifacts (scripted, end-to-end)

Run the capture runner to execute all truth-table cases plus random cases, then grab screenshots every N ticks:

```bash
cd node-harness
node interaction-pipeline/captureBatch.mjs \
  --name truth-table-plus-random \
  --randomCount 8 \
  --tickEvery 100 \
  --maxTick 1000
```

Output batch structure:

- `cases/<nn>-<id>/scenario.json` (source preset + run config)
- `cases/<nn>-<id>/scenario-config.json` (actual Interaction Lab payload/truth row)
- `cases/<nn>-<id>/screenshots/tick-XXXXXX.png`
- `cases/<nn>-<id>/metrics.json`
- `manifest.jsonl` (auto-updated statuses)
- `capture-summary.json`

Notes:

- Requires sim-server running (`http://127.0.0.1:8787` by default).
- Uses `playwright-core` from `sim-server/node_modules`.
  - Install once if missing:
    - `cd sim-server && npm install --save-dev playwright-core`
- For visible browser mode, add `--headed`.

> This keeps capture deterministic and analysis stateless while producing screenshot-first artifacts.

## Stage C — analysis pass

```bash
cd node-harness
node interaction-pipeline/analyzeBatch.mjs --batch ../artifacts/interaction-pipeline/<batch>
```

Writes:

- `analysis.md`
- `verdicts.json`

This script marks artifact completeness and produces a review-ready skeleton; human/LLM review can then fill expected-vs-observed conclusions from screenshots + scenario payloads.
