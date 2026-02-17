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

## Stage B — capture artifacts (scriptable step)

For each case dir, write:

- `screenshot.png`
- `metrics.json` (include at minimum `frame.totalRGB` and basic timing)

Then update `manifest.jsonl` entry fields:

- `screenshotPath`
- `metricsPath`
- `status` (e.g. `captured`)

> This keeps capture deterministic and analysis stateless.

## Stage C — analysis pass

```bash
cd node-harness
node interaction-pipeline/analyzeBatch.mjs --batch ../artifacts/interaction-pipeline/<batch>
```

Writes:

- `analysis.md`
- `verdicts.json`

This script marks artifact completeness and produces a review-ready skeleton; human/LLM review can then fill expected-vs-observed conclusions from screenshots + scenario payloads.
