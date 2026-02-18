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


test('soft spring gpu-only WGSL prep branch stores deterministic spring color batches for conflict-free dispatch planning', () => {
  assert.match(
    source,
    /buildSoftSpringXpbdWgslPlan\([\s\S]*springColorOffsets[\s\S]*springColorOrderedIndices[\s\S]*wgslOffload\.state\.preparedPlan = plan/,
    'expected WGSL prep branch to retain spring color batches for no-atomic per-color dispatch staging',
  );
});

test('soft spring gpu-only path dispatches a real WGSL probe stage when offload device is available', () => {
  assert.match(
    source,
    /if \(canUseWgslOffload\(wgslOffload\)\) \{[\s\S]*dispatchSoftSpringWgslProbe\(\{ soft, offload: wgslOffload, layout \}\)[\s\S]*lastMode = 'wgsl-probe'/,
    'expected gpu-only soft spring branch to execute WGSL compute dispatch (with cpu fallback semantics preserved)',
  );
});
