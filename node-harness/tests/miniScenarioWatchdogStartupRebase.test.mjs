import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const WATCHDOG_PATH = path.resolve('node-harness/miniScenarioWatchdog.mjs');

function runWatchdog(seed) {
  const out = execFileSync(process.execPath, [WATCHDOG_PATH, '--seed', String(seed)], {
    encoding: 'utf8',
  });
  return JSON.parse(out);
}

test('mini watchdog seed 1071541588 no longer reports frame-0 soft area drift regression', () => {
  const report = runWatchdog(1071541588);
  assert.equal(report.ok, true, 'expected watchdog pass for stabilized startup seed');

  const frame0 = (report.sampledTelemetry || []).find((s) => s?.step === 0);
  assert.ok(frame0, 'expected frame-0 telemetry sample');
  assert.ok((frame0.maxAreaDeviation ?? 1) <= 1e-6,
    `expected near-zero frame-0 area drift after startup rebase, got ${frame0?.maxAreaDeviation}`);
});

test('mini watchdog seed 4225125758 rebases sustained early area drift before threshold violation', () => {
  const report = runWatchdog(4225125758);
  assert.equal(report.ok, true, 'expected watchdog pass for sustained-drift adversarial seed');

  const frame10 = (report.sampledTelemetry || []).find((s) => s?.step === 10);
  assert.ok(frame10, 'expected frame-10 telemetry sample');
  assert.ok((frame10.maxAreaDeviation ?? 1) < 0.25,
    `expected sustained early-drift guardrail to lower area drift, got ${frame10?.maxAreaDeviation}`);

  assert.ok((report.final?.maxAreaDeviation ?? 1) < 0.25,
    `expected final area drift to stay below 0.25 after sustained-drift rebase, got ${report.final?.maxAreaDeviation}`);
});
