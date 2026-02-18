import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js'), 'utf8');

test('soft spring gpu-only module wires WGSL prep layout alongside plan in offload branch', () => {
  assert.match(
    source,
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*buildSoftSpringXpbdWgslPlan\([\s\S]*buildSoftSpringXpbdWgslLayout\([\s\S]*wgslOffload\.state\.preparedLayout = layout;[\s\S]*lastPreparedLayoutBytes/,
    'expected WGSL prep branch to build/store deterministic layout buffers for next XPBD offload stage',
  );
});
