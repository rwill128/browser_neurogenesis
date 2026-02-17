import test from 'node:test';
import assert from 'node:assert/strict';

import { buildParityEntry, evaluateParity } from '../gpu-lab/parity.mjs';

test('evaluateParity passes when candidate metrics stay within explicit tolerances', () => {
  const cpu = { meanAbsDelta: 0.5, rms: 1.2, maxAbs: 2.0, totalMass: 10.0 };
  const gpu = { meanAbsDelta: 0.57, rms: 1.31, maxAbs: 2.3, totalMass: 9.94 };

  const result = evaluateParity(cpu, gpu, {
    meanAbsDelta: 0.1,
    rms: 0.2,
    maxAbs: 0.5,
    totalMass: 0.1
  });

  assert.equal(result.pass, true);
  assert.equal(result.failing.length, 0);
});

test('evaluateParity reports per-metric failures with deltas and tolerances', () => {
  const cpu = { meanAbsDelta: 0.5, rms: 1.2, maxAbs: 2.0, totalMass: 10.0 };
  const gpu = { meanAbsDelta: 0.8, rms: 1.6, maxAbs: 2.8, totalMass: 9.6 };

  const result = evaluateParity(cpu, gpu, {
    meanAbsDelta: 0.1,
    rms: 0.2,
    maxAbs: 0.5,
    totalMass: 0.1
  });

  assert.equal(result.pass, false);
  assert.deepEqual(result.failing.map((f) => f.metric).sort(), ['maxAbs', 'meanAbsDelta', 'rms', 'totalMass']);
});

test('buildParityEntry emits non-comparable payload when candidate metrics are missing', () => {
  const entry = buildParityEntry({
    label: 'gpu-stub',
    cpuMetrics: { meanAbsDelta: 0, rms: 0, maxAbs: 0, totalMass: 1 },
    candidateMetrics: null,
    reason: 'metrics unavailable'
  });

  assert.deepEqual(entry, {
    backend: 'gpu-stub',
    comparable: false,
    reason: 'metrics unavailable'
  });
});
