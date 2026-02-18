import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepCollisionBoundaryGpuOnly.js'), 'utf8');

test('collision-boundary gpu-only source routes non-finite ABI entries to CPU while keeping finite entries on WGSL', () => {
  assert.match(
    source,
    /function splitFiniteBoundaryEntries\(entries\) \{[\s\S]*const finite = \[\];[\s\S]*const nonFinite = \[\];/,
    'expected finite/non-finite boundary-entry splitter for mixed-path routing',
  );

  assert.match(
    source,
    /const \{ finite: finiteRigid, nonFinite: nonFiniteRigid \} = splitFiniteBoundaryEntries\(rigidList\);[\s\S]*const \{ finite: finiteSoft, nonFinite: nonFiniteSoft \} = splitFiniteBoundaryEntries\(softNodes\);/,
    'expected rigid + soft boundary paths to partition finite vs non-finite ABI entries',
  );

  assert.match(
    source,
    /if \(nonFiniteEntryCount > 0\) \{[\s\S]*applyBoundaryCpu\(nonFiniteRigid, n, rigidBounce, applyBounceBoundary\);[\s\S]*applyBoundaryCpu\(nonFiniteSoft, n, softBounce, applyBounceBoundary\);/,
    'expected non-finite ABI entries to stay on CPU boundary clamp path',
  );

  assert.match(
    source,
    /mode: nonFiniteEntryCount > 0 \? 'wgsl-partial' : 'wgsl',[\s\S]*reason: nonFiniteEntryCount > 0 \? 'ok-with-cpu-nonfinite' : 'ok',/,
    'expected explicit source-route telemetry for mixed WGSL+CPU boundary execution',
  );
});
