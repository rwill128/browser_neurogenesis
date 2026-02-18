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

test('gpu-lab defines rigid edge momentum helper used by fluid feedback injection', () => {
  assert.match(
    source,
    /function rigidEdgeMomentumScale\(rb\) \{[\s\S]*return c > 0 \? \(sum \/ c\) : 1;[\s\S]*\}/,
    'expected rigid edge momentum helper to exist in gpu-lab runtime source',
  );

  assert.match(
    source,
    /injectPoint\([\s\S]*rigidEdgeMomentumScale\(b\),/,
    'expected fluid feedback injection path to use rigid edge momentum helper',
  );
});

test('gpu-lab routes soft fluid-coupling stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applySoftFluidCouplingGpuOnly \} from '\/runtime-solvers\/stepSoftFluidCouplingGpuOnly\.js';/,
    'expected isolated gpu-only soft fluid-coupling module import',
  );

  assert.match(
    source,
    /const softMembraneClusterSet = ensureSoftMembraneClusterSet\(sim\);[\s\S]*if \(solverPath === 'gpu-only'\) \{[\s\S]*applySoftFluidCouplingGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*for \(let i = 0; i < s\.nodes\.length; i\+\+\)/,
    'expected explicit gpu-only soft fluid-coupling dispatch with baseline fallback loop',
  );
});

test('gpu-lab routes body-fluid momentum injection through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applyBodyFluidInjectionGpuOnly \} from '\/runtime-solvers\/stepBodyFluidInjectionGpuOnly\.js';/,
    'expected isolated gpu-only body-fluid injection module import',
  );

  assert.match(
    source,
    /let injectedMomentum = 0;[\s\S]*if \(solverPath === 'gpu-only'\) \{[\s\S]*applyBodyFluidInjectionGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*const injectPoint = \(px, py, pvx, pvy/,
    'expected explicit gpu-only body-fluid injection dispatch with baseline fallback branch',
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

test('gpu-lab routes soft spring rest-recovery stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applySoftRestRecoveryGpuOnly \} from '\/runtime-solvers\/stepSoftRestRecoveryGpuOnly\.js';/,
    'expected isolated gpu-only soft rest-recovery module import',
  );

  assert.match(
    source,
    /if \(softSpringRestRecoveryOn && sim\.softSpringRestBaseline && sim\.softSpringRestBaseline\.length === s\.springs\.length\) \{[\s\S]*if \(solverPath === 'gpu-only'\) \{[\s\S]*applySoftRestRecoveryGpuOnly\(\{[\s\S]*recoverSoftSpringRests,[\s\S]*\}\);[\s\S]*\} else \{/,
    'expected explicit gpu-only rest-recovery dispatch with baseline fallback branch',
  );
});

test('gpu-lab routes rigid-soft collision stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ resolveRigidSoftCollisionPassGpuOnly \} from '\/runtime-solvers\/stepRigidSoftCollisionGpuOnly\.js';/,
    'expected isolated gpu-only rigid-soft collision module import',
  );

  assert.match(
    source,
    /if \(solverPath === 'gpu-only'\) \{[\s\S]*resolveRigidSoftCollisionPassGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*for \(let rbi = 0; rbi < bodies\.rigid\.length; rbi\+\+\)/,
    'expected explicit gpu-only rigid-soft collision dispatch with baseline fallback loop',
  );
});

test('gpu-lab routes soft-soft collision stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ resolveSoftSoftCollisionPassGpuOnly \} from '\/runtime-solvers\/stepSoftCollisionGpuOnly\.js';/,
    'expected isolated gpu-only soft-soft collision module import',
  );

  assert.match(
    source,
    /if \(solverPath === 'gpu-only'\) \{[\s\S]*resolveSoftSoftCollisionPassGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*for \(let i = 0; i < s\.nodes\.length; i\+\+\)[\s\S]*resolveSoftNodeVsSoftEdgeCollision\(node, a, b, 0\.12\);/,
    'expected explicit gpu-only soft-soft collision dispatch with baseline fallback loops',
  );
});

test('gpu-lab routes soft spring XPBD stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applySoftSpringsXPBDVelocityGpuOnly \} from '\/runtime-solvers\/stepSoftSpringsXpbdGpuOnly\.js';/,
    'expected isolated gpu-only soft spring XPBD module import',
  );

  assert.match(
    source,
    /if \(solverPath === 'gpu-only'\) \{[\s\S]*applySoftSpringsXPBDVelocityGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*applySoftSpringsXPBDVelocity\(s, dtPos, [A-Za-z0-9_.$]+, sim\.softXPBDLambda, \{/,
    'expected explicit gpu-only soft spring XPBD dispatch with baseline fallback call',
  );
});

test('gpu-lab routes soft area XPBD stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applySoftAreaXPBDVelocityGpuOnly \} from '\/runtime-solvers\/stepSoftAreaXpbdGpuOnly\.js';/,
    'expected isolated gpu-only soft area XPBD module import',
  );

  assert.match(
    source,
    /ensureSoftAreaRestState\(sim, s, softClusterLoops, dtPos\);[\s\S]*if \(solverPath === 'gpu-only'\) \{[\s\S]*applySoftAreaXPBDVelocityGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*applySoftAreaXPBDVelocity\(sim, s, softClusterLoops, dtPos, [A-Za-z0-9_.$]+/,
    'expected explicit gpu-only soft area XPBD dispatch with baseline fallback call',
  );
});

test('gpu-lab routes hybrid rigid-soft attachment constraints through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applyHybridAttachmentConstraintsGpuOnly \} from '\/runtime-solvers\/stepHybridConstraintsGpuOnly\.js';/,
    'expected isolated gpu-only hybrid attachment module import',
  );

  assert.match(
    source,
    /if \(solverPath === 'gpu-only'\) \{[\s\S]*applyHybridAttachmentConstraintsGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*for \(let iter = 0; iter < 5; iter\+\+\) \{/,
    'expected explicit gpu-only hybrid attachment dispatch with baseline fallback loops',
  );
});

test('gpu-lab routes soft membrane cell-pressure stepping through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applySoftMembraneCellPressureGpuOnly \} from '\/runtime-solvers\/stepSoftMembranePressureGpuOnly\.js';/,
    'expected isolated gpu-only membrane pressure module import',
  );

  assert.match(
    source,
    /const membranePressureClusters = membranePressureOn[\s\S]*solverPath === 'gpu-only'[\s\S]*applySoftMembraneCellPressureGpuOnly\(\{[\s\S]*\}\)[\s\S]*: applySoftMembraneCellPressure\(sim, s, softClusterLoops, dtPos\)[\s\S]*: 0;/,
    'expected membrane pressure pass to be toggle-gated with gpu-only dispatch and baseline fallback',
  );
});

test('gpu-lab routes post-collision soft-cluster projection and inside-correction recovery through isolated gpu-only runtime solver module', () => {
  assert.match(
    source,
    /import \{ applyPostCollisionRecoveryGpuOnly \} from '\/runtime-solvers\/stepPostCollisionRecoveryGpuOnly\.js';/,
    'expected isolated gpu-only post-collision recovery module import',
  );

  assert.match(
    source,
    /if \(solverPath === 'gpu-only'\) \{[\s\S]*applyPostCollisionRecoveryGpuOnly\(\{[\s\S]*\}\);[\s\S]*\} else \{[\s\S]*computeSoftClusterKinematics\(s\.nodes\);[\s\S]*applyRigidInsideCorrectionPass\(bodies, s, hybridAttachedByRigid\);/,
    'expected explicit gpu-only post-collision recovery dispatch with baseline fallback branch',
  );
});

test('gpu-lab gates post-collision recovery pass with shared toggle across baseline and gpu-only paths', () => {
  assert.match(
    source,
    /const enablePostCollisionRecoveryEl = document\.getElementById\('enablePostCollisionRecovery'\);/,
    'expected post-collision recovery checkbox binding in gpu-lab controls',
  );

  assert.match(
    source,
    /enablePostCollisionRecovery: \(enablePostCollisionRecoveryEl\?\.checked !== false\),/,
    'expected readControls to publish post-collision recovery toggle',
  );

  assert.match(
    source,
    /const postCollisionRecoveryOn = sim\.controls\?\.enablePostCollisionRecovery !== false;[\s\S]*if \(postCollisionRecoveryOn\) \{[\s\S]*if \(solverPath === 'gpu-only'\) \{/,
    'expected post-collision recovery execution to be gated for both solver paths',
  );
});

test('gpu-lab gates always-on stabilizer layers with shared toggles across baseline and gpu-only paths', () => {
  assert.match(
    source,
    /const enableSoftClusterStabilizersEl = document\.getElementById\('enableSoftClusterStabilizers'\);[\s\S]*const enableSoftSpringRestRecoveryEl = document\.getElementById\('enableSoftSpringRestRecovery'\);[\s\S]*const enableMembraneBoundaryXpbdEl = document\.getElementById\('enableMembraneBoundaryXpbd'\);[\s\S]*const enableMembraneShapeMemoryEl = document\.getElementById\('enableMembraneShapeMemory'\);[\s\S]*const enableMembranePressureEl = document\.getElementById\('enableMembranePressure'\);/,
    'expected checkbox bindings for newly-gated always-on stabilizer layers',
  );

  assert.match(
    source,
    /enableSoftClusterStabilizers: \(enableSoftClusterStabilizersEl\?\.checked !== false\),[\s\S]*enableSoftSpringRestRecovery: \(enableSoftSpringRestRecoveryEl\?\.checked !== false\),[\s\S]*enableMembraneBoundaryXpbd: \(enableMembraneBoundaryXpbdEl\?\.checked !== false\),[\s\S]*enableMembraneShapeMemory: \(enableMembraneShapeMemoryEl\?\.checked !== false\),[\s\S]*enableMembranePressure: \(enableMembranePressureEl\?\.checked !== false\),/,
    'expected readControls to surface always-on stabilizer toggles',
  );

  assert.match(
    source,
    /const softClusterStabilizersOn = sim\.controls\?\.enableSoftClusterStabilizers !== false;[\s\S]*const softSpringRestRecoveryOn = sim\.controls\?\.enableSoftSpringRestRecovery !== false;[\s\S]*const membraneBoundaryXpbdOn = sim\.controls\?\.enableMembraneBoundaryXpbd !== false;[\s\S]*const membraneShapeMemoryOn = sim\.controls\?\.enableMembraneShapeMemory !== false;[\s\S]*const membranePressureOn = sim\.controls\?\.enableMembranePressure !== false;/,
    'expected runtime to derive stabilizer gates from controls for both solver branches',
  );
});
