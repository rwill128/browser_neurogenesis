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
