import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js'), 'utf8');

test('post-collision recovery gpu-only source routes boundary pass through collision-boundary module when provided', () => {
  assert.match(
    source,
    /if \(typeof applyCollisionBoundaryPassGpuOnly === 'function'\) \{[\s\S]*applyCollisionBoundaryPassGpuOnly\(\{[\s\S]*rigidBounce: 0\.84,[\s\S]*softBounce: 0\.78,[\s\S]*wgslOffload,[\s\S]*\}\)/,
    'expected post-collision recovery to delegate boundary pass via gpu-only collision boundary module',
  );

  assert.match(
    source,
    /boundaryRuntime =\s*\{ mode: 'cpu-inline', reason: 'bounce-callback' \};[\s\S]*else if \(typeof applyBounceBoundary === 'function'\)/,
    'expected legacy CPU bounce callback path to remain available as fallback',
  );

  assert.match(
    source,
    /return \{[\s\S]*boundaryRuntime,[\s\S]*\};/,
    'expected boundary runtime source-route telemetry to be returned to gpu-only orchestrator',
  );
});
