import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepRigidInsideCorrectionGpuOnly.js'), 'utf8');

test('rigid-inside gpu-only path publishes deterministic WGSL prep layout + source-route telemetry while remaining CPU-authoritative', () => {
  assert.match(
    source,
    /function buildRigidInsideCorrectionWgslLayout\(\{[\s\S]*rigidPolyOffsets = new Uint32Array\(rigidCount \+ 1\);[\s\S]*eligibleNodeOffsets = new Uint32Array\(rigidCount \+ 1\);[\s\S]*polyPointOffsets = \[0\];/,
    'expected deterministic rigid/poly/node ownership layout builder for upcoming WGSL inside-correction dispatch',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedInsideLayout = prep\.layout;[\s\S]*lastPreparedInsideLayoutSignature = prep\.signature;[\s\S]*lastSourceRoute = 'cpu-rigid-inside-prepared-layout';/,
    'expected gpu-only inside-correction path to publish layout signature + source-route telemetry for WGSL staging',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastInsideCorrectionCount = corrected;[\s\S]*lastAuthoritativeInsideSource = 'cpu-rigid-inside-authoritative';[\s\S]*lastSourceRoute = 'cpu-rigid-inside-authoritative';/,
    'expected explicit authoritative source route to remain CPU while WGSL prep ownership becomes available',
  );
});
