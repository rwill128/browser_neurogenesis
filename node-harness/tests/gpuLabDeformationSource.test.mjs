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
    /function sampleFluidForBodyCoupling\([\s\S]*obstacleMask = null,[\s\S]*selfFeedbackSuppression = 0,[\s\S]*\)/,
    'expected boundary-aware fluid sampling helper for body coupling (with self-feedback suppression support)',
  );

  assert.match(
    source,
    /const fx = sampleFluidForBodyCoupling\([\s\S]*vxField,[\s\S]*sx,[\s\S]*sy,[\s\S]*obstacleMask,[\s\S]*bodyFeedbackPrevVx,[\s\S]*SELF_FEEDBACK_SUPPRESSION,[\s\S]*\);/,
    'expected rigid coupling to use boundary-aware flow sampling with self-feedback suppression',
  );

  assert.match(
    source,
    /const fx = sampleFluidForBodyCoupling\([\s\S]*vxField,[\s\S]*node\.x,[\s\S]*node\.y,[\s\S]*obstacleMask,[\s\S]*bodyFeedbackPrevVx,[\s\S]*SELF_FEEDBACK_SUPPRESSION,[\s\S]*\);/,
    'expected soft coupling to use boundary-aware flow sampling with self-feedback suppression',
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
    /fn deflectBacktrace\(channel:u32, x:u32, y:u32, vel:vec2<f32>\)->vec2<f32>/,
    'expected explicit deflected backtrace helper for BLOCK dye behavior',
  );

  assert.match(
    source,
    /let blockedR = hereObs \|\| srcObs \|\| hereR == 1u \|\| srcR == 1u;[\s\S]*let rp = deflectBacktrace\(0u, gid\.x, gid\.y, vel\);/,
    'expected blocked red-channel branch to use deflected backtrace',
  );
});

test('gpu-lab swept dye relocation subtracts absorbed channels before depositing displaced mass', () => {
  assert.match(
    source,
    /function redistributeSweptEdgeDyeTransport\([\s\S]*const \[modeR, modeG, modeB\] = unpackDyeMaskModes\(packedMask\);/,
    'expected swept dye relocation path to decode per-channel dye mode mask',
  );

  assert.match(
    source,
    /const rvAfterEat = modeR === EDGE_DYE_MODE\.ABSORB \? 0 : rv;[\s\S]*const gvAfterEat = modeG === EDGE_DYE_MODE\.ABSORB \? 0 : gv;[\s\S]*const bvAfterEat = modeB === EDGE_DYE_MODE\.ABSORB \? 0 : bv;/,
    'expected swept transport to subtract absorbed channels before relocation',
  );

  assert.match(
    source,
    /r\[ti\] = \(Number\(r\[ti\]\) \|\| 0\) \+ rvAfterEat;[\s\S]*g\[ti\] = \(Number\(g\[ti\]\) \|\| 0\) \+ gvAfterEat;[\s\S]*b\[ti\] = \(Number\(b\[ti\]\) \|\| 0\) \+ bvAfterEat;/,
    'expected swept transport deposits to use post-EAT channel values only',
  );
});

test('gpu-lab attempts storage-buffer stage limits 10 then 9 before default device request', () => {
  assert.match(
    source,
    /const requestedStorageLimitCandidates = \[10, 9\]/,
    'expected ordered storage-limit fallback candidates [10, 9]',
  );

  assert.match(
    source,
    /requiredLimits: \{ maxStorageBuffersPerShaderStage: limit \}/,
    'expected requestDevice requiredLimits override for storage-buffer stage limit',
  );

  assert.match(
    source,
    /if \(!device\) \{[\s\S]*device = await adapter\.requestDevice\(\);/,
    'expected fallback to default requestDevice when explicit limit requests fail',
  );
});

test('gpu-lab supports interaction-lab scripted motion fixtures (pinned/circle)', () => {
  assert.match(
    source,
    /function configureInteractionLab\(sim, options = \{\}\)/,
    'expected interaction-lab config helper in gpu-lab runtime',
  );

  assert.match(
    source,
    /function applyInteractionLabBodyMotion\(sim\)/,
    'expected scripted motion application helper for pinned/circle fixtures',
  );

  assert.match(
    source,
    /applyInteractionLabBodyMotion\(sim\);[\s\S]*let injectedMomentum = 0;/,
    'expected motion override to be re-applied before body->fluid injection',
  );
});

test('gpu-lab overlay labels rigid/soft segment IDs for confrontation debugging', () => {
  assert.match(
    source,
    /const rigidSegId = `R\$\{i\}:\$\{ei\}`;[\s\S]*drawSegmentIdLabel\(rigidSegId,/,
    'expected rigid segment id labels (R<body>:<edge>) in overlay',
  );

  assert.match(
    source,
    /const softSegId = `S\$\{sgi\}`;[\s\S]*drawSegmentIdLabel\(softSegId,/,
    'expected soft segment id labels (S<spring>) in overlay',
  );

  assert.match(
    source,
    /if \(data\.type === 'gpuLabHighlightSegment'\)/,
    'expected embed message hook to receive segment highlight commands',
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

test('gpu-lab routes rigid stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ stepRigidBodiesGpuOnly \} from '\/runtime-solvers\/stepRigidGpuOnly\.js';/,
    'expected isolated gpu-only rigid stepping module import',
  );

  assert.match(
    source,
    /if \(solverPath === 'gpu-only'\) \{[\s\S]*stepRigidBodiesGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{/,
    'expected explicit gpu-only rigid stepping dispatch with baseline fallback branch',
  );
});

test('gpu-lab routes soft integration stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ integrateSoftBodiesGpuOnly \} from '\/runtime-solvers\/stepSoftIntegrateGpuOnly\.js';/,
    'expected isolated gpu-only soft integration module import',
  );

  assert.match(
    source,
    /const hybridNodeVCap = 3\.2;[\s\S]*if \(solverPath === 'gpu-only'\) \{[\s\S]*integrateSoftBodiesGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*for \(const node of s\.nodes\)/,
    'expected explicit gpu-only soft integration dispatch with baseline fallback loop',
  );
});
