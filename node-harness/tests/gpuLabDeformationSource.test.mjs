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

test('gpu-lab obstacle mask is edge-velocity BLOCK based for both rigid and soft (no post-pass barrier)', () => {
  assert.match(
    source,
    /function stampBodyObstacleMask\(sim\)/,
    'expected unified body obstacle-mask stamping helper',
  );

  assert.match(
    source,
    /for \(const rb of sim\?\.bodies\?\.rigid \|\| \[\]\)/,
    'expected rigid edges to participate in obstacle-mask stamping',
  );

  assert.match(
    source,
    /for \(const sp of \(soft\?\.springs \|\| \[\]\)\)/,
    'expected soft edges to participate in obstacle-mask stamping',
  );

  assert.match(
    source,
    /const edgeVelocityMode = Number\(sp\?\.\[5\]\) === EDGE_BODY_MODE\.PASS[\s\S]*if \(edgeVelocityMode === EDGE_BODY_MODE\.PASS\) continue;/,
    'expected obstacle-mask inclusion to be controlled by per-edge velocity mode',
  );

  assert.doesNotMatch(
    source,
    /applyBodyEdgeFieldBarriers\s*\(/,
    'expected runtime to stop using post-pass body barrier path',
  );
});
