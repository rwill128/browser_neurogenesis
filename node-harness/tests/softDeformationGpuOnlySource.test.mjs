import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftDeformationGpuOnly.js'), 'utf8');

test('soft deformation gpu-only stage refreshes collapse classification from WGSL spring metrics', () => {
  assert.match(
    source,
    /function refreshSoftDeformationClassificationFromMetrics\(\{[\s\S]*thresholds,[\s\S]*\}\)/,
    'expected helper to recompute warning/severe/severeCollapse summary from stage metrics',
  );

  assert.match(
    source,
    /if \(wgslApplied\) \{[\s\S]*refreshSoftDeformationClassificationFromMetrics\(\{[\s\S]*thresholds: deformationThresholds,[\s\S]*\}\);[\s\S]*\}/,
    'expected authoritative WGSL spring metrics path to refresh deformation classification before intervention gating',
  );

  assert.match(
    source,
    /if \(severeInterventionsOn && deform\?\.severeCollapseCount > 0\) \{/,
    'expected severe intervention gate to continue using severeCollapseCount after WGSL classification refresh',
  );
});
