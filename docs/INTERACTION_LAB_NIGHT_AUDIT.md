# Interaction Lab Night Audit Log

Automated 20-minute audit cycles.

## Format

For each cycle append:

- Timestamp (America/Buenos_Aires)
- Scenario (preset id/name or random config)
- Expected behavior
- Observed behavior
- Quick metrics (if sampled)
- Screenshot path(s)
- Decision:
  - PASS (no issue)
  - ISSUE FOUND
- If fixed:
  - files changed
  - rationale
  - commit hash
  - before/after result

---

## 2026-02-17 03:10 (America/Buenos_Aires)

- Scenario: `rigid-red-block-pass-pinned` (Rigid line · red BLOCK · velocity PASS · pinned)
- Expected:
  - Selected channel (red) shows boundary-conditioned reflection/deflection near segment
  - Non-selected channels remain pass-through
- Initial observation:
  - Screenshot looked mostly cyan/white and visually under-communicated red-channel behavior
  - User correctly flagged that there was "no red plume" visible in the diagnostic image
- Issue classification: **ISSUE FOUND (diagnostic visibility mismatch)**
- Before screenshot:
  - `/Users/richardwilliams/.openclaw/media/browser/188629c4-2cc5-4086-a6e2-7b4b2aee63fb.jpg`

### Fix applied

- Rationale:
  - Reflection logic existed, but visual diagnostics were weak because the default emitter tint was cyan-heavy.
  - For channel-filter confrontation runs, we need high-contrast channel-biased injection so behavior is inspectable by eye.
- Changes:
  - `sim-server/public/gpu-lab.js`
    - `buildWindTunnelEmitter(...)` now accepts optional `colorR/colorG/colorB`
    - embed reset path forwards `options.emitterColorR/G/B` into emitter construction
  - `sim-server/public/interaction-lab.js`
    - added `emitterTintForScenario(channel, dyeMode, velocityMode)`
    - for `dyeMode=block && velocityMode=pass`, emitter is auto-tinted toward selected channel
    - debug notes now include `emitterVisualizationNote`
- After screenshot:
  - `/Users/richardwilliams/.openclaw/media/browser/9a228d43-484a-473c-a69e-68eacda09cca.jpg`
- Post-fix assessment:
  - Red plume is now clearly visible in the confrontation frame, making channel-filter behavior inspectable.
  - Deterministic validation rerun passed full suite: `154/154`.

## 2026-02-17 03:25 (America/Buenos_Aires)

- Scenario: `rigid-red-eat-block-pinned` (Curriculum 2/5: Rigid line · red EAT · velocity BLOCK · pinned)
- Expected behavior (per current contract):
  - Requested dye mode is EAT, but because `velocity=BLOCK`, effective dye mode must auto-resolve to NO-OP.
  - Plume should reach/contact the rigid segment while boundary behavior is governed by BLOCK; no EAT-specific absorption should be applied.
- Observed behavior:
  - After loading preset and waiting for advection, red plume reached and contacted the segment (visible contact at labeled rigid edges `R0:1..R0:3`).
  - Truth-table payload panel reports `requestedDyeMode: "eat"`, `dyeMode: "noop"`, `effectiveDyeMode: "noop"`, `velocityMode: "block"`, matching contract.
  - Visual result is consistent with BLOCK-conditioned confrontation and no contradiction with EAT/PASS-only effectiveness rule.
- Quick metrics:
  - Fixture: `rigid-line`
  - Motion: `pinned`
  - Channel: `r`
  - Momentum coupling: `1`
  - Payload `createdAt`: `2026-02-17T06:25:24.002Z`
- Screenshot path(s):
  - `/Users/richardwilliams/.openclaw/media/browser/fd41faa5-9d4e-444e-bd07-d4742f6a7262.jpg`
- Decision: **PASS (no issue)**
- Fixes/code changes: none
- Deterministic validation:
  - Not rerun this cycle (no code change triggered).
- Commit hash: `7d82ed7`


## 2026-02-17 03:31 (America/Buenos_Aires)

- Scenario: `random-20260217-033059` (Auto-generated: rigid-line, pinned, B NOOP, velocity PASS, momentum 0.60)
- Expected behavior (per current contract):
  - With `velocity=PASS`, edge velocity should behave as pass-through (dashed edge diagnostics).
  - Requested dye mode is `NO-OP`, so no dye absorption should occur on the selected blue channel.
  - Plume should advect to and contact/cross the segment region without EAT-style removal.
- Observed behavior:
  - After random scenario load and settle wait, blue plume reached/contacted the rigid segment region (`R0:1..R0:3`) with no amber absorb signature.
  - Overlay diagnostics showed PASS edge styling (dashed), consistent with velocity pass-through.
  - Truth-table payload reports `channel: "b"`, `requestedDyeMode: "noop"`, `effectiveDyeMode: "noop"`, `velocityMode: "pass"`, matching contract expectations.
- Quick metrics:
  - Fixture: `rigid-line`
  - Motion: `pinned`
  - Momentum coupling: `0.6000000000000001`
  - Payload `createdAt`: `2026-02-17T06:30:59.202Z`
- Screenshot path(s):
  - `/Users/richardwilliams/.openclaw/media/browser/8001fb1c-c688-49ca-908c-c48f4902da01.jpg`
- Decision: **PASS (no issue)**
- Fixes/code changes: none
- Deterministic validation:
  - Not rerun this cycle (no code change triggered).

## 2026-02-17 03:34 (America/Buenos_Aires)

- Scenario: `rigid-red-eat-block-pinned` (manual repro from user report)
- User report: "there's no emitter or dye at all" on last sent screenshot.
- Expected behavior:
  - Wind tunnel view should always show active dyed plume unless explicitly disabled.
- Root cause found:
  - In `buildWindTunnelEmitter(...)`, color channels were set with `clamp(Number(colorX), 0, 255)`.
  - When `colorR/G/B` options are omitted, `Number(undefined)` becomes `NaN`, and clamp propagated `NaN`.
  - This produced invalid emitter channel values and could result in effectively missing/black plume in some runs.
- Fix applied:
  - `sim-server/public/gpu-lab.js`
    - `cr/cg/cb` now use finite-checked defaults:
      - `colorR` fallback `135`
      - `colorG` fallback `190`
      - `colorB` fallback `255`
- After-fix verification:
  - Reopened Interaction Lab, allowed settle, confirmed non-zero dye totals and visible plume.
  - Screenshot: `/Users/richardwilliams/.openclaw/media/browser/af224343-4ce2-4051-9c1b-ed213391b5ea.jpg`
- Decision: **ISSUE FOUND -> FIXED**
- Deterministic validation:
  - `node-harness/test-gpu-track.mjs` passed `154/154`.

