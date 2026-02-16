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

test('gpu-lab samples coupling flow outside blocked edge cells to preserve body push', () => {
  assert.match(
    source,
    /function sampleFluidForBodyCoupling\(field, n, x, y, dirX, dirY, obstacleMask = null\)/,
    'expected boundary-aware fluid sampling helper for body coupling',
  );

  assert.match(
    source,
    /const fx = sampleFluidForBodyCoupling\(vxField, n, sx, sy, rx, ry, obstacleMask\);/,
    'expected rigid coupling to use boundary-aware flow sampling',
  );

  assert.match(
    source,
    /const fx = sampleFluidForBodyCoupling\(vxField, n, node\.x, node\.y, rx, ry, obstacleMask\);/,
    'expected soft coupling to use boundary-aware flow sampling',
  );
});

test('gpu-lab dye advection honors obstacle mask to prevent through-body leakage', () => {
  assert.match(
    source,
    /const advectDyeWgsl = commonWgsl \+ `[\s\S]*@group\(0\) @binding\(9\) var<storage, read> obstacleMask: array<f32>;/,
    'expected obstacle mask binding in dye advection shader',
  );

  assert.match(
    source,
    /let hereObs = obstacleMask\[i\] > 0\.5;[\s\S]*let srcObs = obstacleMask\[si\] > 0\.5;/,
    'expected obstacle sampling in dye advection branch logic',
  );

  assert.match(
    source,
    /else if \(hereObs \|\| srcObs \|\| hereR == 1u \|\| srcR == 1u\)/,
    'expected blocked-obstacle branch to retain local dye instead of crossing',
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
