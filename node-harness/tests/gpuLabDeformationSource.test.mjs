import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');

test('gpu-lab deformation stabilization is gated by collapse risk, not pose-only severity', () => {
  assert.match(
    source,
    /if \((?:severeInterventionsOn && )?deform\.severeCollapseCount > 0\) \{\s*stabilizeSeverelyDeformedSoftClusters/s,
    'expected severe stabilization to require severeCollapseCount > 0 (optionally behind severe feature flag)',
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

test('gpu-lab applies soft momentum policy in fluid→soft force coupling path', () => {
  assert.match(
    source,
    /const nodeMomentum = softNodeMomentumScale\(i\);[\s\S]*const flowCoupling = flowCouplingBase \* nodeMomentum;/,
    'expected per-node soft momentum policy scaling on fluid→soft coupling',
  );
});

test('gpu-lab membrane obstacle mask is membrane-cluster scoped (not global soft-body wall)', () => {
  assert.match(
    source,
    /if \(membraneClusterSet\.size === 0\) \{\s*return \{ membraneEdgeCount: 0, usedMembraneClusters: false \};\s*\}/,
    'expected obstacle mask to be disabled when no membrane clusters are present',
  );

  assert.match(
    source,
    /if \(!membraneClusterSet\.has\(ca\)\) continue;/,
    'expected obstacle edges to be restricted to membrane cluster membership',
  );
});
