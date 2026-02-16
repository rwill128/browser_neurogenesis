import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');

test('gpu-lab deformation stabilization is gated by collapse risk, not pose-only severity', () => {
  assert.match(
    source,
    /if \(deform\.severeCollapseCount > 0\) \{\s*stabilizeSeverelyDeformedSoftClusters/s,
    'expected severe stabilization to require severeCollapseCount > 0',
  );

  assert.match(
    source,
    /const severeSet = deform\.severeCollapseSet;/,
    'expected stabilizeSeverelyDeformedSoftClusters to use severeCollapseSet',
  );

  assert.match(
    source,
    /c\.severe = c\.severe \|\| c\.severeCollapse \|\| \(poseAvailable && poseSevere\);/,
    'expected pose severity to be tracked separately from collapse severity',
  );

  assert.match(
    source,
    /c\.severeCollapse = c\.severeCollapse \|\| \(!isMembraneCluster && legacySevere\);/,
    'expected membrane clusters to be excluded from legacy collapse classification',
  );
});
