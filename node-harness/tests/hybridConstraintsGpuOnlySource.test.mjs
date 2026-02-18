import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const sourcePath = resolve(process.cwd(), 'sim-server/public/runtime-solvers/stepHybridConstraintsGpuOnly.js');
const source = readFileSync(sourcePath, 'utf8');

test('hybrid constraints gpu-only module publishes deterministic WGSL prep layout ownership for upcoming offload stage', () => {
  assert.match(
    source,
    /export function buildHybridAttachmentWgslPrep\([\s\S]*const nodeIndex = new Uint32Array\(count\);[\s\S]*const rigidIndex = new Uint32Array\(count\);[\s\S]*const anchorAX = new Float32Array\(count\);[\s\S]*const restB = new Float32Array\(count\);/,
    'expected gpu-only hybrid module to build deterministic typed-array layout for attachment anchors and rests',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedPlan = plan;[\s\S]*wgslOffload\.state\.preparedLayout = layout;[\s\S]*lastPreparedValidAttachmentCount[\s\S]*lastPreparedLayoutBytes[\s\S]*lastMode = 'cpu-prepared';/,
    'expected gpu-only hybrid pass to persist WGSL prep ownership metadata while remaining CPU-authoritative',
  );
});
