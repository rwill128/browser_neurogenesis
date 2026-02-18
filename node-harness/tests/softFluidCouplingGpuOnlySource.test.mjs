import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftFluidCouplingGpuOnly.js'), 'utf8');

test('soft fluid-coupling gpu-only publishes deterministic WGSL prep layout for next-stage offload', () => {
  assert.match(
    source,
    /function buildSoftFluidCouplingWgslLayout\(\{ nodes, softNodeMomentumScale, softMembraneClusterSet \}\) \{[\s\S]*clusterNodeOffsets = new Uint32Array\(clusterCount \+ 1\);[\s\S]*clusterNodeCount = new Uint32Array\(clusterCount\);/,
    'expected deterministic node/cluster packed layout builder for upcoming WGSL stage',
  );

  assert.match(
    source,
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*preparedLayout = prep\.layout;[\s\S]*lastPreparedLayoutBytes = prep\.byteLength;[\s\S]*lastPreparedLayoutSignature = prep\.signature;/,
    'expected gpu-only path to publish WGSL prep telemetry into offload state',
  );

  assert.match(
    source,
    /lastMode = 'cpu-prepared';[\s\S]*handoff contract for the upcoming compute kernel\./,
    'expected explicit cpu-prepared source-route marker and blocker note for imminent WGSL kernel handoff',
  );
});
