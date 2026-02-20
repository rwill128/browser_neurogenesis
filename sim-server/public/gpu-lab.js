import { parseCreatureSpec, buildBodiesFromCreatureSpec } from '/creature-spec.js';
import { EDGE_DYE_MODE, normalizeEdgeDyeModeRGB, normalizePermeabilityRGB } from '/dye-barrier.js';
import {
  resolveRigidVsSoftNodeCollision,
  resolveRigidVsRigidPolygonCollision,
  getRigidCollisionPolysWorld,
  pointInPolygonInclusive,
  buildRigidSoftNodeCollisionCache,
  buildCollisionPhaseSceneCache,
} from '/rigid-collision.js';
import { loadRigidRigidCandidateBackendWasm } from '/rigid-rigid-candidate-wasm.js';
import { loadSoftClusterProjectionBackendWasm } from '/soft-cluster-projection-wasm.js';
import { loadRigidStepBaselineBackendWasm } from '/rigid-step-baseline-wasm.js';
import { sanitizeSoftSprings, ensureLambdaCacheSize, buildSoftClusterBoundaryLoops, recoverSoftSpringRests } from '/soft-xpbd.js';
import { computeVectorRms, computeRigidAlignedPoseResidual } from '/soft-deformation-metrics.js';
import { computeSoftClusterKinematics, projectNodesTowardClusterRigidMotion } from '/soft-cluster-kinematics.js';
import { stepRigidBodiesGpuOnly } from '/runtime-solvers/stepRigidGpuOnly.js';
import { integrateSoftBodiesGpuOnly } from '/runtime-solvers/stepSoftIntegrateGpuOnly.js';
import { resolveRigidRigidCollisionPassGpuOnly } from '/runtime-solvers/stepRigidCollisionGpuOnly.js';
import { resolveRigidSoftCollisionPassGpuOnly } from '/runtime-solvers/stepRigidSoftCollisionGpuOnly.js';
import { applySoftSpringsXPBDVelocityGpuOnly } from '/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js';
import { applySoftAreaXPBDVelocityGpuOnly } from '/runtime-solvers/stepSoftAreaXpbdGpuOnly.js';
import { resolveSoftSoftCollisionPassGpuOnly } from '/runtime-solvers/stepSoftCollisionGpuOnly.js';
import { applySoftMembraneCellPressureGpuOnly } from '/runtime-solvers/stepSoftMembranePressureGpuOnly.js';
import {
  applySoftMembraneBoundaryXPBDVelocityGpuOnly,
  applySoftMembraneShapeMemoryVelocityGpuOnly,
} from '/runtime-solvers/stepSoftMembraneConstraintsGpuOnly.js';
import { applySoftFluidCouplingGpuOnly } from '/runtime-solvers/stepSoftFluidCouplingGpuOnly.js';
import { applyPostCollisionRecoveryGpuOnly } from '/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js';
import { stabilizeRigidPostIntegrateGpuOnly } from '/runtime-solvers/stepRigidPostIntegrateGpuOnly.js';
import { applyCollisionBoundaryPassGpuOnly } from '/runtime-solvers/stepCollisionBoundaryGpuOnly.js';
import { runCollisionIterationsGpuOnly } from '/runtime-solvers/stepCollisionIterationsGpuOnly.js';
import { applySoftRestRecoveryGpuOnly } from '/runtime-solvers/stepSoftRestRecoveryGpuOnly.js';
import { applyBodyFluidInjectionGpuOnly } from '/runtime-solvers/stepBodyFluidInjectionGpuOnly.js';
import { applySoftDeformationInterventionsGpuOnly } from '/runtime-solvers/stepSoftDeformationGpuOnly.js';

const out = document.getElementById('out');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const clearViscBtn = document.getElementById('clearViscBtn');
const importSpecBtn = document.getElementById('importSpecBtn');
const importSpecFile = document.getElementById('importSpecFile');
const importScaleEl = document.getElementById('importScale');
const fpsHud = document.getElementById('fpsHud');
const softIntegrateHudEl = document.getElementById('softIntegrateHud');
const fallbackHudEl = document.getElementById('fallbackHud');
const fallbackHudDetailsEl = document.getElementById('fallbackHudDetails');
const copyFallbacksBtn = document.getElementById('copyFallbacksBtn');
const clearFallbacksBtn = document.getElementById('clearFallbacksBtn');
const copyFallbacksStatusEl = document.getElementById('copyFallbacksStatus');
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');

const gridEl = document.getElementById('gridSize');
const dtEl = document.getElementById('dt');
const fadeEl = document.getElementById('fade');
const viscosityEl = document.getElementById('viscosity');
const impulseEl = document.getElementById('impulse');
const radiusEl = document.getElementById('radius');
const brushSizeEl = document.getElementById('brushSize');
const paintValueEl = document.getElementById('paintValue');
const showViscEl = document.getElementById('showVisc');
const showObstacleMaskOverlayEl = document.getElementById('showObstacleMaskOverlay');
const showCollisionHullEl = document.getElementById('showCollisionHull');
const showSegmentIdsEl = document.getElementById('showSegmentIds');
const showExtraVisualsEl = document.getElementById('showExtraVisuals');
const scenarioPresetEl = document.getElementById('scenarioPreset');
const massLightEl = document.getElementById('massLight');
const massHeavyEl = document.getElementById('massHeavy');
const massSoftEl = document.getElementById('massSoft');
const rigidBodyCountEl = document.getElementById('rigidBodyCount');
const bodyDragEl = document.getElementById('bodyDrag');
const bodyFeedbackEl = document.getElementById('bodyFeedback');
const fluidVelocityCapEl = document.getElementById('fluidVelocityCap');
const fluidCouplingComponentLimitEl = document.getElementById('fluidCouplingComponentLimit');
const enableCouplingLagFrameEl = document.getElementById('enableCouplingLagFrame');
const softSpringStiffnessEl = document.getElementById('softSpringStiffness');
const softSpringBaseComplianceEl = document.getElementById('softSpringBaseCompliance');
const softAreaBaseComplianceEl = document.getElementById('softAreaBaseCompliance');
const membraneEdgeBaseComplianceEl = document.getElementById('membraneEdgeBaseCompliance');
const membraneBendBaseComplianceEl = document.getElementById('membraneBendBaseCompliance');
const softClusterFluidTorqueCouplingEl = document.getElementById('softClusterFluidTorqueCoupling');
const softClusterAngularProjectionEl = document.getElementById('softClusterAngularProjection');
const softClusterCollisionAngularProjectionEl = document.getElementById('softClusterCollisionAngularProjection');
const enableArtificialSwimEl = document.getElementById('enableArtificialSwim');
const seedRigidBodiesEl = document.getElementById('seedRigidBodies');
const seedSpringSoftBodiesEl = document.getElementById('seedSpringSoftBodies');
const spawnMembraneCellsEl = document.getElementById('spawnMembraneCells');
const enableWarningDeformInterventionsEl = document.getElementById('enableWarningDeformInterventions');
const enableSevereDeformInterventionsEl = document.getElementById('enableSevereDeformInterventions');
const enablePostCollisionRecoveryEl = document.getElementById('enablePostCollisionRecovery');
const enableSoftClusterStabilizersEl = document.getElementById('enableSoftClusterStabilizers');
const enableSoftSpringRestRecoveryEl = document.getElementById('enableSoftSpringRestRecovery');
const enableMembraneBoundaryXpbdEl = document.getElementById('enableMembraneBoundaryXpbd');
const enableMembraneShapeMemoryEl = document.getElementById('enableMembraneShapeMemory');
const enableMembranePressureEl = document.getElementById('enableMembranePressure');
const runtimeSolverPathEls = Array.from(document.querySelectorAll('input[name="runtimeSolverPath"]'));
const runtimePipelineModeEls = Array.from(document.querySelectorAll('input[name="runtimePipelineMode"]'));
const rigidRigidCandidateBackendEl = document.getElementById('rigidRigidCandidateBackend');

const GENERATED_MINI_SCENARIOS_URL = '/generated-mini-scenarios.json';
let generatedMiniScenarios = new Map();

const urlParams = new URLSearchParams(window.location.search || '');
const EMBED_MODE = (
  urlParams.get('embed') === '1'
  || urlParams.get('embedded') === '1'
  || (
    window.self !== window.top
    && urlParams.get('embed') !== '0'
    && urlParams.get('embedded') !== '0'
  )
);
setRuntimeSolverPath(urlParams.get('solverPath'), { syncUrl: false });
setRuntimePipelineMode(urlParams.get('pipelineMode'), { syncUrl: false });
setRigidRigidCandidateBackendMode(urlParams.get('rigidRigidCandidateBackend'), { syncUrl: false });
if (EMBED_MODE) {
  document.body.classList.add('embed-mode');
}

function log(v) {
  if (!out) return;
  out.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2);
}

const WORKGROUP = 8;
const JACOBI_ITERS = 20;
const SOFT_SPRING_STIFFNESS_DEFAULT = 4.2; // stronger lattice resistance for spring-mode soft bodies
const SOFT_NODE_FLOW_COUPLING = 0.42; // reduce per-vertex fluid carry so single nodes are less individually yanked
const SOFT_NODE_LOCAL_FLOW_SHARE = 0.35; // keep local deformation response while cluster inertia handles bulk flow pull
const SOFT_CLUSTER_FLOW_FORCE_SHARE = 0.58; // apply cluster-level linear/angular fluid acceleration back onto nodes
const SOFT_CLUSTER_TUG_COUPLING = 0.58; // redistribute current-induced pull across the whole soft cluster
const SOFT_CLUSTER_RELATIVE_DRAG = 0.11; // damp per-node divergence from cluster motion
const SOFT_CLUSTER_LINEAR_PROJECTION = 0.08; // mild translation projection toward cluster COM velocity
const SOFT_CLUSTER_ANGULAR_PROJECTION = 0.14; // mild rotation projection toward cluster omega
const SOFT_CLUSTER_COLLISION_LINEAR_PROJECTION = 0.1; // post-collision redistribution of linear impulse
const SOFT_CLUSTER_COLLISION_ANGULAR_PROJECTION = 0.22; // post-collision redistribution of angular impulse
const SOFT_CLUSTER_FLUID_INJECT_BLEND = 0.55; // blend node velocity with cluster rigid-like velocity for fluid feedback
const SOFT_XPBD_ITERS = 10;
const SOFT_XPBD_BASE_COMPLIANCE = 0.0012;
const RIGID_RIGID_SPATIAL_HASH_CELL_SIZE = 12;
const RIGID_SOFT_SPATIAL_HASH_CELL_SIZE = 12;
const SOFT_AREA_XPBD_ITERS = 6;
const SOFT_AREA_BASE_COMPLIANCE = 0.0009;
const SOFT_INTEGRATION_SCALE = 24;
const DEFAULT_FLUID_VELOCITY_CAP = 24.0;
const DEFAULT_FLUID_COUPLING_COMPONENT_LIMIT = 24;
// rigid-soft hybrid links removed from runtime pipelines.

// Soft deformation color-state thresholds:
// - warning (yellow/orange): first-level shape drift alert
// - severe (red): high distortion / collapse-risk escalation
const SOFT_DEFORM_WARN_STRETCH = 3.2;
const SOFT_DEFORM_SEVERE_STRETCH = 6.0;
const SOFT_DEFORM_WARN_AREA_RATIO_MIN = 0.45;
const SOFT_DEFORM_WARN_AREA_RATIO_MAX = 2.2;
const SOFT_DEFORM_SEVERE_AREA_RATIO_MIN = 0.2;
const SOFT_DEFORM_SEVERE_AREA_RATIO_MAX = 5.0;
const SOFT_DEFORM_WARN_POSE_RMS = 0.18;
const SOFT_DEFORM_SEVERE_POSE_RMS = 0.32;
const SOFT_DEFORM_WARN_POSE_MAX = 0.34;
const SOFT_DEFORM_SEVERE_POSE_MAX = 0.58;
const MEMBRANE_CELL_BASE_COUNT = 4;
const MEMBRANE_CELL_BASE_PRESSURE_GAIN = 0.08;
const MEMBRANE_CELL_BASE_RADIAL_DAMPING = 0.06;
const MEMBRANE_SHAPE_MEMORY_GAIN = 0.045;
const MEMBRANE_SHAPE_MEMORY_ITERS = 2;
const MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC = 0.08;
const MEMBRANE_EDGE_XPBD_ITERS = 8;
const MEMBRANE_EDGE_BASE_COMPLIANCE = 0.0007;
const MEMBRANE_BEND_XPBD_ITERS = 4;
const MEMBRANE_BEND_BASE_COMPLIANCE = 0.0022;
const RIGID_INSIDE_CORRECTION_ITERS = 2;
const RIGID_INSIDE_CORRECTION_SLOP = 0.04;
const WORLD_SCALE_MIN = 2.0;
const WORLD_SCALE_MAX = 8.0;
const WORLD_SCALE_START_RIGID = 200;
const WORLD_SCALE_FULL_RIGID = 2000;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function normalizeWorldScale(raw) {
  const v = Number(raw);
  if (!Number.isFinite(v)) return null;
  return clamp(v, WORLD_SCALE_MIN, WORLD_SCALE_MAX);
}

function readWorldScaleFromUrl() {
  if (typeof window === 'undefined') return null;
  const raw = new URLSearchParams(window.location.search || '').get('worldScale');
  if (raw == null || raw === '') return null;
  return normalizeWorldScale(raw);
}

function deriveWorldScale(rigidCount, override = null) {
  const forced = normalizeWorldScale(override);
  if (Number.isFinite(forced)) return forced;
  const count = Math.max(1, Number(rigidCount) || 1);
  const denom = Math.max(1, WORLD_SCALE_FULL_RIGID - WORLD_SCALE_START_RIGID);
  const t = clamp((count - WORLD_SCALE_START_RIGID) / denom, 0, 1);
  return WORLD_SCALE_MIN + t * (WORLD_SCALE_MAX - WORLD_SCALE_MIN);
}

function getWorldScale(controls) {
  const resolved = normalizeWorldScale(controls?.worldScale);
  return Number.isFinite(resolved) ? resolved : WORLD_SCALE_MIN;
}

function getWorldSize(controls) {
  return (Number(controls?.n) || 0) * getWorldScale(controls);
}

function normalizeRigidRigidCandidateBackendMode(raw) {
  const mode = String(raw || '').trim().toLowerCase();
  if (mode === 'wasm' || mode === 'wasm-rr' || mode === 'wasm-v1') return 'wasm';
  return 'js';
}

function readRigidRigidCandidateBackendModeFromUrl() {
  if (typeof window === 'undefined') return 'js';
  const raw = new URLSearchParams(window.location.search || '').get('rigidRigidCandidateBackend');
  return normalizeRigidRigidCandidateBackendMode(raw);
}

function getRigidRigidCandidateBackendMode() {
  return normalizeRigidRigidCandidateBackendMode(
    rigidRigidCandidateBackendEl?.value ?? readRigidRigidCandidateBackendModeFromUrl(),
  );
}

function normalizeSoftClusterProjectionBackendMode(raw) {
  const mode = String(raw || '').trim().toLowerCase();
  return mode === 'wasm' ? 'wasm' : 'js';
}

function readSoftClusterProjectionBackendModeFromUrl() {
  if (typeof window === 'undefined') return 'js';
  const raw = new URLSearchParams(window.location.search || '').get('postCollisionProjectionBackend');
  return normalizeSoftClusterProjectionBackendMode(raw);
}

function normalizeRigidStepBaselineBackendMode(raw) {
  const mode = String(raw || '').trim().toLowerCase();
  return mode === 'wasm' ? 'wasm' : 'js';
}

function readRigidStepBaselineBackendModeFromUrl() {
  if (typeof window === 'undefined') return 'js';
  const raw = new URLSearchParams(window.location.search || '').get('rigidStepBaselineBackend');
  return normalizeRigidStepBaselineBackendMode(raw);
}

function setRigidRigidCandidateBackendMode(mode, { syncUrl = false } = {}) {
  const normalized = normalizeRigidRigidCandidateBackendMode(mode);
  if (rigidRigidCandidateBackendEl) {
    rigidRigidCandidateBackendEl.value = normalized;
  }
  if (syncUrl && window?.history?.replaceState) {
    const params = new URLSearchParams(window.location.search || '');
    if (normalized === 'js') {
      params.delete('rigidRigidCandidateBackend');
    } else {
      params.set('rigidRigidCandidateBackend', normalized);
    }
    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash || ''}`;
    window.history.replaceState({}, '', nextUrl);
  }
  return normalized;
}

function normalizeFluidVelocityCap(raw) {
  const v = Number(raw);
  if (!Number.isFinite(v)) return DEFAULT_FLUID_VELOCITY_CAP;
  return clamp(v, 0.1, 96);
}

function normalizeFluidCouplingComponentLimit(raw) {
  const v = Number(raw);
  if (!Number.isFinite(v)) return DEFAULT_FLUID_COUPLING_COMPONENT_LIMIT;
  return clamp(v, 0.25, 128);
}

function clampFluidComponent(v, limitRaw) {
  const limit = normalizeFluidCouplingComponentLimit(limitRaw);
  if (!Number.isFinite(v)) return 0;
  return clamp(v, -limit, limit);
}

function normalizeSoftSpringStiffness(raw) {
  const v = Number(raw);
  if (!Number.isFinite(v)) return SOFT_SPRING_STIFFNESS_DEFAULT;
  return clamp(v, 0.2, 16);
}

function normalizeCompliance(raw, fallback) {
  const v = Number(raw);
  const fb = Number.isFinite(Number(fallback)) ? Number(fallback) : 0.001;
  if (!Number.isFinite(v)) return fb;
  return clamp(v, 1e-6, 0.05);
}

function resolveRigidEdgeVelocityMode(rb, edgeIndex) {
  const velocityModeRaw = Array.isArray(rb?.edgeVelocityMode)
    ? Number(rb.edgeVelocityMode[edgeIndex])
    : Number.NaN;
  if (velocityModeRaw === EDGE_BODY_MODE.PASS) return EDGE_BODY_MODE.PASS;
  const bodyModeRaw = Array.isArray(rb?.edgeBodyMode)
    ? Number(rb.edgeBodyMode[edgeIndex])
    : Number.NaN;
  return bodyModeRaw === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK;
}

function rigidEdgeMomentumScale(rb, allowPassEdgeFlowPush = false) {
  const arr = Array.isArray(rb?.edgeMomentumCoupling)
    ? rb.edgeMomentumCoupling
    : (Array.isArray(rb?.edgeMomentumTransfer) ? rb.edgeMomentumTransfer : null);
  const edgeCount = Math.max(
    Number(arr?.length) || 0,
    Array.isArray(rb?.edgeVelocityMode) ? rb.edgeVelocityMode.length : 0,
    Array.isArray(rb?.edgeBodyMode) ? rb.edgeBodyMode.length : 0,
    Array.isArray(rb?.verticesLocal) ? rb.verticesLocal.length : 0,
    Math.max(0, Number(rb?.sides) || 0),
  );
  if (edgeCount <= 0) return 1;

  let sum = 0;
  for (let ei = 0; ei < edgeCount; ei++) {
    const raw = Array.isArray(arr) ? Number(arr[ei]) : Number.NaN;
    let momentum = Number.isFinite(raw) ? clamp(raw, 0, 1) : 1;
    if (!allowPassEdgeFlowPush && resolveRigidEdgeVelocityMode(rb, ei) === EDGE_BODY_MODE.PASS) {
      momentum = 0;
    }
    sum += momentum;
  }
  return clamp(sum / Math.max(1, edgeCount), 0, 1);
}

function normalizeRuntimeSolverPath(raw) {
  const mode = String(raw || '').trim().toLowerCase();
  return mode === 'gpu-only' ? 'gpu-only' : 'baseline';
}

function normalizeRuntimePipelineMode(raw, solverPath = null) {
  const normalizedSolverPath = normalizeRuntimeSolverPath(solverPath ?? getRuntimeSolverPath());
  const mode = String(raw || '').trim().toLowerCase();
  if (mode === 'standard') return 'standard';
  if (normalizedSolverPath !== 'gpu-only') return 'standard';
  if (mode === 'gpu-only-fast') return 'gpu-only-fast';
  if (mode === 'gpu-only-validated') return 'gpu-only-validated';
  return 'gpu-only-validated';
}

function getRuntimeSolverPath() {
  const picked = runtimeSolverPathEls.find((el) => el?.checked);
  return normalizeRuntimeSolverPath(picked?.value);
}

function getRuntimePipelineMode(solverPath = null) {
  const normalizedSolverPath = normalizeRuntimeSolverPath(solverPath ?? getRuntimeSolverPath());
  if (normalizedSolverPath !== 'gpu-only') return 'standard';
  const picked = runtimePipelineModeEls.find((el) => el?.checked);
  return normalizeRuntimePipelineMode(picked?.value, normalizedSolverPath);
}

function setRuntimePipelineMode(mode, { syncUrl = false, solverPath = null } = {}) {
  const normalizedSolverPath = normalizeRuntimeSolverPath(solverPath ?? getRuntimeSolverPath());
  const normalized = normalizeRuntimePipelineMode(mode, normalizedSolverPath);
  for (const el of runtimePipelineModeEls) {
    if (!el) continue;
    el.checked = (normalizeRuntimePipelineMode(el.value, normalizedSolverPath) === normalized);
    el.disabled = normalizedSolverPath !== 'gpu-only' && normalizeRuntimePipelineMode(el.value, 'gpu-only') !== 'standard';
  }
  if (syncUrl && window?.history?.replaceState) {
    const params = new URLSearchParams(window.location.search || '');
    params.set('pipelineMode', normalized);
    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash || ''}`;
    window.history.replaceState({}, '', nextUrl);
  }
  return normalized;
}

function setRuntimeSolverPath(mode, { syncUrl = false } = {}) {
  const normalized = normalizeRuntimeSolverPath(mode);
  for (const el of runtimeSolverPathEls) {
    if (!el) continue;
    el.checked = (normalizeRuntimeSolverPath(el.value) === normalized);
  }
  const activePipelineMode = setRuntimePipelineMode(
    normalized === 'gpu-only' ? getRuntimePipelineMode('gpu-only') : 'standard',
    { syncUrl, solverPath: normalized },
  );
  if (syncUrl && window?.history?.replaceState) {
    const params = new URLSearchParams(window.location.search || '');
    params.set('solverPath', normalized);
    params.set('pipelineMode', activePipelineMode);
    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash || ''}`;
    window.history.replaceState({}, '', nextUrl);
  }
  return normalized;
}

const EDGE_BODY_MODE = {
  PASS: 0,
  BLOCK: 1,
};

function readPassEdgeFlowPushToggleFromUrl() {
  if (typeof window === 'undefined') return false;
  const raw = new URLSearchParams(window.location.search || '').get('passEdgeFlowPush');
  if (raw == null || raw === '') return false;
  const v = String(raw).trim().toLowerCase();
  return v === '1' || v === 'true' || v === 'yes' || v === 'on';
}

function normalizeFastReadbackInterval(raw) {
  const n = Math.round(Number(raw));
  if (!Number.isFinite(n)) return FAST_MODE_FULL_READBACK_INTERVAL;
  return Math.max(1, Math.min(240, n));
}

function readFastReadbackIntervalFromUrl() {
  if (typeof window === 'undefined') return FAST_MODE_FULL_READBACK_INTERVAL;
  const raw = new URLSearchParams(window.location.search || '').get('fastReadbackInterval');
  if (raw == null || raw === '') return FAST_MODE_FULL_READBACK_INTERVAL;
  return normalizeFastReadbackInterval(raw);
}

function detectRendererFingerprintSync() {
  const base = {
    userAgent: (typeof navigator !== 'undefined' && navigator?.userAgent) ? navigator.userAgent : null,
    webglVendor: null,
    webglRenderer: null,
    isSwiftShader: false,
  };

  try {
    const canvasEl = document.createElement('canvas');
    const gl = canvasEl.getContext('webgl') || canvasEl.getContext('experimental-webgl');
    if (!gl) return base;

    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const vendor = ext
      ? String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) || '')
      : String(gl.getParameter(gl.VENDOR) || '');
    const renderer = ext
      ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '')
      : String(gl.getParameter(gl.RENDERER) || '');

    base.webglVendor = vendor || null;
    base.webglRenderer = renderer || null;
    const low = `${vendor} ${renderer}`.toLowerCase();
    base.isSwiftShader = low.includes('swiftshader');
    return base;
  } catch {
    return base;
  }
}

async function readWebGpuAdapterFingerprint(adapter) {
  if (!adapter || typeof adapter.requestAdapterInfo !== 'function') return null;
  try {
    const info = await adapter.requestAdapterInfo();
    return {
      vendor: info?.vendor || null,
      architecture: info?.architecture || null,
      device: info?.device || null,
      description: info?.description || null,
    };
  } catch {
    return null;
  }
}

function readControls() {
  const rigidBodyCount = Math.max(1, Math.min(2000, Math.round(Number(rigidBodyCountEl?.value) || 10)));
  const worldScale = deriveWorldScale(rigidBodyCount, readWorldScaleFromUrl());
  return {
    n: Math.max(32, Number(gridEl.value) || 256),
    dt: Number(dtEl.value) || 0.01,
    fade: Number(fadeEl.value) || 0.9999,
    viscosity: Number(viscosityEl.value) || 0.00001,
    fluidVelocityCap: normalizeFluidVelocityCap(fluidVelocityCapEl?.value),
    impulse: Number(impulseEl.value) || 2.5,
    radius: Number(radiusEl.value) || 6,
    massLight: Math.max(0.05, Number(massLightEl.value) || 1.2),
    massHeavy: Math.max(0.05, Number(massHeavyEl.value) || 5.0),
    massSoft: Math.max(0.02, Number(massSoftEl.value) || 0.6),
    rigidBodyCount,
    worldScale,
    bodyDrag: Math.max(0, Number(bodyDragEl.value) || 0.55),
    bodyFeedback: Math.max(0, Number(bodyFeedbackEl.value) || 0.012),
    fluidCouplingComponentLimit: normalizeFluidCouplingComponentLimit(fluidCouplingComponentLimitEl?.value),
    enableCouplingLagFrame: !!enableCouplingLagFrameEl?.checked,
    showObstacleMaskOverlay: (showObstacleMaskOverlayEl?.checked !== false),
    softSpringStiffness: normalizeSoftSpringStiffness(softSpringStiffnessEl?.value),
    softSpringBaseCompliance: normalizeCompliance(softSpringBaseComplianceEl?.value, SOFT_XPBD_BASE_COMPLIANCE),
    softAreaBaseCompliance: normalizeCompliance(softAreaBaseComplianceEl?.value, SOFT_AREA_BASE_COMPLIANCE),
    membraneEdgeBaseCompliance: normalizeCompliance(membraneEdgeBaseComplianceEl?.value, MEMBRANE_EDGE_BASE_COMPLIANCE),
    membraneBendBaseCompliance: normalizeCompliance(membraneBendBaseComplianceEl?.value, MEMBRANE_BEND_BASE_COMPLIANCE),
    softClusterFluidTorqueCoupling: Math.max(0, Math.min(2, Number(softClusterFluidTorqueCouplingEl?.value) || SOFT_CLUSTER_FLOW_FORCE_SHARE)),
    softClusterAngularProjection: Math.max(0, Math.min(1, Number(softClusterAngularProjectionEl?.value) || SOFT_CLUSTER_ANGULAR_PROJECTION)),
    softClusterCollisionAngularProjection: Math.max(0, Math.min(1, Number(softClusterCollisionAngularProjectionEl?.value) || SOFT_CLUSTER_COLLISION_ANGULAR_PROJECTION)),
    enableArtificialSwim: !!enableArtificialSwimEl?.checked,
    seedRigidBodies: (seedRigidBodiesEl?.checked !== false),
    seedSpringSoftBodies: (seedSpringSoftBodiesEl?.checked === true),
    spawnMembraneCells: !!spawnMembraneCellsEl?.checked,
    enableWarningDeformInterventions: (enableWarningDeformInterventionsEl?.checked !== false),
    enableSevereDeformInterventions: (enableSevereDeformInterventionsEl?.checked !== false),
    enablePostCollisionRecovery: (enablePostCollisionRecoveryEl?.checked !== false),
    enableSoftClusterStabilizers: (enableSoftClusterStabilizersEl?.checked !== false),
    enableSoftSpringRestRecovery: (enableSoftSpringRestRecoveryEl?.checked !== false),
    enableMembraneBoundaryXpbd: (enableMembraneBoundaryXpbdEl?.checked !== false),
    enableMembraneShapeMemory: (enableMembraneShapeMemoryEl?.checked !== false),
    enableMembranePressure: (enableMembranePressureEl?.checked !== false),
    allowPassEdgeFlowPush: readPassEdgeFlowPushToggleFromUrl(),
    fastReadbackInterval: readFastReadbackIntervalFromUrl(),
    runtimeSolverPath: getRuntimeSolverPath(),
    runtimePipelineMode: getRuntimePipelineMode(),
    rigidRigidCandidateBackendMode: getRigidRigidCandidateBackendMode(),
    postCollisionProjectionBackendMode: readSoftClusterProjectionBackendModeFromUrl(),
    rigidStepBaselineBackendMode: readRigidStepBaselineBackendModeFromUrl(),
  };
}

function readImportScale() {
  const raw = Number(importScaleEl?.value);
  if (!Number.isFinite(raw)) return 0.1;
  return Math.max(0.05, Math.min(2, raw));
}

function formatSliderValue(inputEl) {
  const raw = Number(inputEl?.value);
  const decimals = Number(inputEl?.dataset?.decimals);
  if (!Number.isFinite(raw)) return String(inputEl?.value ?? '');
  if (Number.isFinite(decimals) && decimals >= 0) return raw.toFixed(decimals);
  return String(raw);
}

function refreshSliderReadoutById(inputId) {
  if (!inputId) return;
  const inputEl = document.getElementById(inputId);
  const readoutEl = document.querySelector(`[data-readout-for="${inputId}"]`);
  if (!inputEl || !readoutEl) return;
  readoutEl.textContent = formatSliderValue(inputEl);
}

function refreshAllSliderReadouts() {
  const readouts = document.querySelectorAll('[data-readout-for]');
  for (const el of readouts) {
    const inputId = el.getAttribute('data-readout-for');
    refreshSliderReadoutById(inputId);
  }
}

function bindSliderReadouts() {
  const sliders = document.querySelectorAll('input[type="range"][id]');
  for (const slider of sliders) {
    const update = () => refreshSliderReadoutById(slider.id);
    slider.addEventListener('input', update);
    slider.addEventListener('change', update);
  }
  refreshAllSliderReadouts();
}

function softIntegrateHudLabel(state) {
  const mode = String(state?.mode || '').toLowerCase();
  if (mode === 'wgsl') return { text: 'Soft integrate: WGSL', color: '#6bff9a' };
  if (mode === 'cpu-fallback') return { text: 'Soft integrate: CPU fallback', color: '#ffd36b' };
  if (mode === 'cpu-baseline') return { text: 'Soft integrate: CPU baseline', color: '#b8c7ff' };
  if (mode === 'cpu') return { text: 'Soft integrate: CPU (no WGSL)', color: '#ffd36b' };
  return { text: 'Soft integrate: --', color: '#9faec7' };
}

const GPU_FALLBACK_STAGE_LABELS = Object.freeze({
  rigidStepWgslState: 'Rigid step',
  softFluidCouplingWgslState: 'Soft fluid coupling',
  softSpringXpbdWgslState: 'Soft spring XPBD',
  softMembraneBoundaryWgslState: 'Membrane boundary XPBD',
  softMembraneShapeMemoryWgslState: 'Membrane shape memory',
  softAreaXpbdWgslState: 'Soft area XPBD',
  softMembranePressureWgslState: 'Membrane pressure',
  softIntegrateWgslState: 'Soft integrate',
  rigidPostIntegrateWgslState: 'Rigid post-integrate',
  rigidSoftCollisionWgslState: 'Rigid-soft collision',
  softCollisionWgslState: 'Soft-soft collision',
  collisionBoundaryWgslState: 'Collision boundary',
  postCollisionBoundaryWgslState: 'Post-collision recovery',
  softRestRecoveryWgslState: 'Soft rest recovery',
  bodyFluidInjectionWgslState: 'Body-fluid injection',
});

function setFallbackCopyStatus(text, color = '#9faec7') {
  if (!copyFallbacksStatusEl) return;
  copyFallbacksStatusEl.textContent = String(text || '');
  copyFallbacksStatusEl.style.color = color;
}

function collectGpuFallbackSignals(sim) {
  const entries = [];
  const seen = new Set();
  const add = (stage, route, detailKey = '') => {
    if (!stage || !route) return;
    const key = `${stage}|${route}|${detailKey}`;
    if (seen.has(key)) return;
    seen.add(key);
    entries.push({ stage, route: String(route), detailKey: String(detailKey || '') });
  };

  const addRuntimeFallback = (stage, runtime) => {
    const mode = String(runtime?.mode || '').toLowerCase();
    if (!mode.includes('cpu-fallback')) return;
    const reason = String(runtime?.reason || '').trim();
    add(stage, reason ? `mode=${mode}; reason=${reason}` : `mode=${mode}`, 'runtime-mode');
  };

  addRuntimeFallback('Soft integrate', sim?.softIntegrateRuntime);
  addRuntimeFallback('Rigid post-integrate', sim?.rigidPostIntegrateRuntime);
  addRuntimeFallback('Collision boundary', sim?.collisionBoundaryRuntime);

  for (const [stateKey, stage] of Object.entries(GPU_FALLBACK_STAGE_LABELS)) {
    const st = sim?.[stateKey];
    if (!st || typeof st !== 'object') continue;
    for (const [k, v] of Object.entries(st)) {
      if (!k.startsWith('last') || typeof v !== 'string') continue;
      const route = String(v);
      const low = route.toLowerCase();
      if (!(low.includes('fallback') || low.includes('nonfinite') || low.includes('non-finite') || low.includes('error'))) continue;
      add(stage, route, k);
    }
  }

  return entries;
}

async function copyFallbackReportToClipboard() {
  const text = String(sim?.fallbackHudState?.reportText || fallbackHudDetailsEl?.textContent || '').trim();
  if (!text) {
    setFallbackCopyStatus('nothing to copy', '#ffd36b');
    return;
  }

  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', 'readonly');
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      ta.style.top = '0';
      document.body.appendChild(ta);
      ta.select();
      ta.setSelectionRange(0, ta.value.length);
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (!ok) throw new Error('copy command rejected');
    }
    setFallbackCopyStatus('copied', '#6bff9a');
  } catch (err) {
    setFallbackCopyStatus(`copy failed: ${String(err?.message || err)}`, '#ff8f8f');
  }
}

function clearFallbackHistory() {
  if (!sim) {
    if (fallbackHudDetailsEl) fallbackHudDetailsEl.textContent = 'No fallback routes recorded yet.';
    setFallbackCopyStatus('cleared', '#9fe3ff');
    return;
  }
  sim.fallbackHudState = { byRoute: new Map(), lastFallbackFrame: null, reportText: '' };
  updateFallbackHud(sim);
  setFallbackCopyStatus('cleared', '#9fe3ff');
}

function updateFallbackHud(sim) {
  if (!fallbackHudEl) return;

  const solverPath = normalizeRuntimeSolverPath(sim?.controls?.runtimeSolverPath);
  if (solverPath !== 'gpu-only') {
    fallbackHudEl.textContent = 'Fallbacks: n/a (baseline solver)';
    fallbackHudEl.style.color = '#9fb8ff';
    const baselineText = 'Fallback tracker is active only while runtime solver is set to gpu-only.';
    fallbackHudEl.title = baselineText;
    if (fallbackHudDetailsEl) fallbackHudDetailsEl.textContent = baselineText;
    if (sim) {
      sim.fallbackHudState ||= { byRoute: new Map(), lastFallbackFrame: null, reportText: '' };
      sim.fallbackHudState.reportText = baselineText;
      sim.fallbackHudSummary = {
        active: false,
        currentCount: 0,
        hasRecent: false,
        lastFallbackFrame: null,
        currentStages: [],
        recentStages: [],
        currentEntries: [],
        recentEntries: [],
        reportText: baselineText,
      };
    }
    return;
  }

  sim.fallbackHudState ||= { byRoute: new Map(), lastFallbackFrame: null, reportText: '' };
  const hudState = sim.fallbackHudState;
  if (!(hudState.byRoute instanceof Map)) {
    hudState.byRoute = new Map();
  }

  const frame = Number(sim?.frame) || 0;
  const current = collectGpuFallbackSignals(sim);

  for (const entry of current) {
    const key = `${entry.stage}|${entry.route}|${entry.detailKey}`;
    const prev = hudState.byRoute.get(key);
    if (prev) {
      prev.lastFrame = frame;
      prev.hits = (Number(prev.hits) || 0) + 1;
      hudState.byRoute.set(key, prev);
    } else {
      hudState.byRoute.set(key, {
        ...entry,
        firstFrame: frame,
        lastFrame: frame,
        hits: 1,
      });
    }
  }
  if (current.length > 0) {
    hudState.lastFallbackFrame = frame;
  }

  const sticky = Array.from(hudState.byRoute.values())
    .sort((a, b) => (Number(b.lastFrame) || 0) - (Number(a.lastFrame) || 0));

  if (current.length > 0) {
    const stages = Array.from(new Set(current.map((e) => e.stage)));
    fallbackHudEl.textContent = `Fallback NOW (${current.length}): ${stages.join(', ')}`;
    fallbackHudEl.style.color = '#ff8f8f';
  } else if (sticky.length > 0) {
    const recentStages = Array.from(new Set(sticky.map((e) => e.stage)));
    const frameLabel = Number.isFinite(Number(hudState.lastFallbackFrame)) ? `@f${hudState.lastFallbackFrame}` : '';
    fallbackHudEl.textContent = `Last fallback ${frameLabel}: ${recentStages.join(', ')}`;
    fallbackHudEl.style.color = '#ffd36b';
  } else {
    fallbackHudEl.textContent = 'Fallbacks: none';
    fallbackHudEl.style.color = '#7df0b6';
  }

  const mode = normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath);
  const lines = [
    `Fallback tracker | frame=${frame} | solver=${solverPath} | mode=${mode}`,
    '',
    `Current frame fallback routes (${current.length}):`,
  ];
  if (current.length === 0) {
    lines.push('- none');
  } else {
    for (const entry of current) {
      lines.push(`- ${entry.stage}: ${entry.route}${entry.detailKey ? ` [${entry.detailKey}]` : ''}`);
    }
  }
  lines.push('');
  lines.push(`Persistent fallback history (unique routes=${sticky.length}):`);
  if (sticky.length === 0) {
    lines.push('- none');
  } else {
    for (const entry of sticky) {
      lines.push(`- f${entry.lastFrame} (${entry.hits}x): ${entry.stage}: ${entry.route}${entry.detailKey ? ` [${entry.detailKey}]` : ''}`);
    }
  }
  const reportText = lines.join('\n');
  hudState.reportText = reportText;

  if (fallbackHudDetailsEl) fallbackHudDetailsEl.textContent = reportText;
  fallbackHudEl.title = reportText;

  sim.fallbackHudSummary = {
    active: true,
    currentCount: current.length,
    hasRecent: sticky.length > 0,
    lastFallbackFrame: hudState.lastFallbackFrame,
    currentStages: Array.from(new Set(current.map((e) => e.stage))),
    recentStages: Array.from(new Set(sticky.map((e) => e.stage))),
    currentEntries: current.map((e) => `${e.stage}: ${e.route}${e.detailKey ? ` [${e.detailKey}]` : ''}`),
    recentEntries: sticky.map((e) => `f${e.lastFrame} (${e.hits}x): ${e.stage}: ${e.route}${e.detailKey ? ` [${e.detailKey}]` : ''}`),
    reportText,
  };
}

function createBuffer(device, bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
  return device.createBuffer({ size: bytes, usage });
}

function createUniformBuffer(device) {
  return device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

function uploadUniforms(device, uniformBuffer, s) {
  const a = new ArrayBuffer(32);
  const u32 = new Uint32Array(a);
  const f32 = new Float32Array(a);
  u32[0] = s.n;
  f32[1] = s.dt;
  f32[2] = s.fade;
  f32[3] = s.impulse;
  f32[4] = s.radius;
  f32[5] = s.viscosity;
  f32[6] = normalizeFluidVelocityCap(s.fluidVelocityCap);
  f32[7] = 0;
  device.queue.writeBuffer(uniformBuffer, 0, a);
}

function makeDefaultViscMap(n) {
  const m = new Float32Array(n * n);
  m.fill(0.5);
  return m;
}

const commonWgsl = `
struct Params {
  n: u32,
  dt: f32,
  fade: f32,
  impulse: f32,
  radius: f32,
  viscosity_scale: f32,
  fluid_velocity_cap: f32,
  _pad1: f32,
};
@group(0) @binding(0) var<uniform> p: Params;

fn idx(x:u32,y:u32)->u32 { return y * p.n + x; }
fn clampi(v:i32, lo:i32, hi:i32)->i32 { return min(max(v, lo), hi); }
fn sampleBilinear(field: ptr<storage, array<f32>, read>, fx:f32, fy:f32)->f32 {
  let n1 = f32(p.n - 1u);
  let x = clamp(fx, 0.0, n1);
  let y = clamp(fy, 0.0, n1);
  let x0 = u32(floor(x));
  let y0 = u32(floor(y));
  let x1 = min(x0 + 1u, p.n - 1u);
  let y1 = min(y0 + 1u, p.n - 1u);
  let sx = x - f32(x0);
  let sy = y - f32(y0);
  let v00 = (*field)[idx(x0,y0)];
  let v10 = (*field)[idx(x1,y0)];
  let v01 = (*field)[idx(x0,y1)];
  let v11 = (*field)[idx(x1,y1)];
  let a = mix(v00,v10,sx);
  let b = mix(v01,v11,sx);
  return mix(a,b,sy);
}
`;

const advectVelWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> vx0: array<f32>;
@group(0) @binding(2) var<storage, read> vy0: array<f32>;
@group(0) @binding(3) var<storage, read_write> vx1: array<f32>;
@group(0) @binding(4) var<storage, read_write> vy1: array<f32>;
@group(0) @binding(5) var<storage, read> viscMap: array<f32>;
@group(0) @binding(6) var<storage, read> obstacleMask: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let i = idx(gid.x, gid.y);
  if (obstacleMask[i] > 0.5) {
    vx1[i] = 0.0;
    vy1[i] = 0.0;
    return;
  }

  let x = f32(gid.x);
  let y = f32(gid.y);
  let px = x - p.dt * vx0[i];
  let py = y - p.dt * vy0[i];
  let n1 = f32(p.n - 1u);
  let sx = u32(clamp(round(px), 0.0, n1));
  let sy = u32(clamp(round(py), 0.0, n1));
  if (obstacleMask[idx(sx, sy)] > 0.5) {
    vx1[i] = 0.0;
    vy1[i] = 0.0;
    return;
  }

  let localVisc = max(0.0, viscMap[i]) * p.viscosity_scale;
  let decay = 1.0 / (1.0 + 4.0 * localVisc * p.dt);
  var nx = sampleBilinear(&vx0, px, py) * decay;
  var ny = sampleBilinear(&vy0, px, py) * decay;
  let vmax = max(0.1, p.fluid_velocity_cap);
  let mag = sqrt(nx * nx + ny * ny);
  if (mag > vmax) {
    let s = vmax / mag;
    nx = nx * s;
    ny = ny * s;
  }
  vx1[i] = nx;
  vy1[i] = ny;
}
`;

const divergenceWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> vx: array<f32>;
@group(0) @binding(2) var<storage, read> vy: array<f32>;
@group(0) @binding(3) var<storage, read_write> div: array<f32>;
@group(0) @binding(4) var<storage, read> obstacleMask: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let i = idx(gid.x, gid.y);
  if (obstacleMask[i] > 0.5) {
    div[i] = 0.0;
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let n = i32(p.n) - 1;
  let xl = u32(clampi(x - 1, 0, n));
  let xr = u32(clampi(x + 1, 0, n));
  let yt = u32(clampi(y - 1, 0, n));
  let yb = u32(clampi(y + 1, 0, n));

  let li = idx(xl, gid.y);
  let ri = idx(xr, gid.y);
  let ti = idx(gid.x, yt);
  let bi = idx(gid.x, yb);

  var vxL = vx[li];
  var vxR = vx[ri];
  var vyT = vy[ti];
  var vyB = vy[bi];

  if (obstacleMask[li] > 0.5) { vxL = 0.0; }
  if (obstacleMask[ri] > 0.5) { vxR = 0.0; }
  if (obstacleMask[ti] > 0.5) { vyT = 0.0; }
  if (obstacleMask[bi] > 0.5) { vyB = 0.0; }

  let dx = vxR - vxL;
  let dy = vyB - vyT;
  div[i] = 0.5 * (dx + dy);
}
`;

const jacobiPressureWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> pressure0: array<f32>;
@group(0) @binding(2) var<storage, read> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure1: array<f32>;
@group(0) @binding(4) var<storage, read> obstacleMask: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let n = i32(p.n) - 1;
  let xl = u32(clampi(x - 1, 0, n));
  let xr = u32(clampi(x + 1, 0, n));
  let yt = u32(clampi(y - 1, 0, n));
  let yb = u32(clampi(y + 1, 0, n));
  let i = idx(gid.x, gid.y);

  if (obstacleMask[i] > 0.5) {
    pressure1[i] = 0.0;
    return;
  }

  let pC = pressure0[i];
  let li = idx(xl, gid.y);
  let ri = idx(xr, gid.y);
  let ti = idx(gid.x, yt);
  let bi = idx(gid.x, yb);

  var pL = pressure0[li];
  var pR = pressure0[ri];
  var pT = pressure0[ti];
  var pB = pressure0[bi];
  if (obstacleMask[li] > 0.5) { pL = pC; }
  if (obstacleMask[ri] > 0.5) { pR = pC; }
  if (obstacleMask[ti] > 0.5) { pT = pC; }
  if (obstacleMask[bi] > 0.5) { pB = pC; }

  let sumN = pL + pR + pT + pB;
  pressure1[i] = (sumN - div[i]) * 0.25;
}
`;

const projectWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read_write> vx: array<f32>;
@group(0) @binding(2) var<storage, read_write> vy: array<f32>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;
@group(0) @binding(4) var<storage, read> obstacleMask: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let n = i32(p.n) - 1;
  let xl = u32(clampi(x - 1, 0, n));
  let xr = u32(clampi(x + 1, 0, n));
  let yt = u32(clampi(y - 1, 0, n));
  let yb = u32(clampi(y + 1, 0, n));
  let i = idx(gid.x, gid.y);

  if (obstacleMask[i] > 0.5) {
    vx[i] = 0.0;
    vy[i] = 0.0;
    return;
  }

  let pC = pressure[i];
  let li = idx(xl, gid.y);
  let ri = idx(xr, gid.y);
  let ti = idx(gid.x, yt);
  let bi = idx(gid.x, yb);

  var pL = pressure[li];
  var pR = pressure[ri];
  var pT = pressure[ti];
  var pB = pressure[bi];
  let obsL = obstacleMask[li] > 0.5;
  let obsR = obstacleMask[ri] > 0.5;
  let obsT = obstacleMask[ti] > 0.5;
  let obsB = obstacleMask[bi] > 0.5;

  if (obsL) { pL = pC; }
  if (obsR) { pR = pC; }
  if (obsT) { pT = pC; }
  if (obsB) { pB = pC; }

  vx[i] = vx[i] - 0.5 * (pR - pL);
  vy[i] = vy[i] - 0.5 * (pB - pT);

  // No-through boundary condition at obstacle interfaces.
  if (obsL && vx[i] < 0.0) { vx[i] = 0.0; }
  if (obsR && vx[i] > 0.0) { vx[i] = 0.0; }
  if (obsT && vy[i] < 0.0) { vy[i] = 0.0; }
  if (obsB && vy[i] > 0.0) { vy[i] = 0.0; }

  let edge = (gid.x == 0u) || (gid.x == p.n - 1u) || (gid.y == 0u) || (gid.y == p.n - 1u);
  if (edge) {
    if (gid.x == 0u || gid.x == p.n - 1u) {
      vx[i] = 0.0;
      vy[i] = vy[i] * 0.75;
    }
    if (gid.y == 0u || gid.y == p.n - 1u) {
      vy[i] = 0.0;
      vx[i] = vx[i] * 0.75;
    }
  }
}
`;

const advectDyeWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> vx: array<f32>;
@group(0) @binding(2) var<storage, read> vy: array<f32>;
@group(0) @binding(3) var<storage, read> r0: array<f32>;
@group(0) @binding(4) var<storage, read> g0: array<f32>;
@group(0) @binding(5) var<storage, read> b0: array<f32>;
@group(0) @binding(6) var<storage, read_write> r1: array<f32>;
@group(0) @binding(7) var<storage, read_write> g1: array<f32>;
@group(0) @binding(8) var<storage, read_write> b1: array<f32>;
@group(0) @binding(9) var<storage, read> obstacleMask: array<f32>;
@group(0) @binding(10) var<storage, read> dyeMask: array<u32>;

fn decodeMode(mask:u32, channel:u32)->u32 {
  if (channel == 0u) { return mask % 3u; }
  if (channel == 1u) { return (mask / 3u) % 3u; }
  return (mask / 9u) % 3u;
}

fn maskBlock(mask:u32, channel:u32)->f32 {
  return select(0.0, 1.0, decodeMode(mask, channel) == 1u);
}

fn channelBlockGradient(channel:u32, x:u32, y:u32)->vec2<f32> {
  var xm = x;
  var xp = x;
  var ym = y;
  var yp = y;
  if (x > 0u) { xm = x - 1u; }
  if (x + 1u < p.n) { xp = x + 1u; }
  if (y > 0u) { ym = y - 1u; }
  if (y + 1u < p.n) { yp = y + 1u; }

  let left = maskBlock(dyeMask[idx(xm, y)], channel);
  let right = maskBlock(dyeMask[idx(xp, y)], channel);
  let up = maskBlock(dyeMask[idx(x, ym)], channel);
  let down = maskBlock(dyeMask[idx(x, yp)], channel);
  return vec2<f32>(right - left, down - up);
}

fn obstacleGradient(x:u32, y:u32)->vec2<f32> {
  var xm = x;
  var xp = x;
  var ym = y;
  var yp = y;
  if (x > 0u) { xm = x - 1u; }
  if (x + 1u < p.n) { xp = x + 1u; }
  if (y > 0u) { ym = y - 1u; }
  if (y + 1u < p.n) { yp = y + 1u; }

  let left = obstacleMask[idx(xm, y)];
  let right = obstacleMask[idx(xp, y)];
  let up = obstacleMask[idx(x, ym)];
  let down = obstacleMask[idx(x, yp)];
  return vec2<f32>(right - left, down - up);
}

fn deflectBacktrace(channel:u32, x:u32, y:u32, vel:vec2<f32>)->vec2<f32> {
  let grad = channelBlockGradient(channel, x, y) + obstacleGradient(x, y);
  let g2 = dot(grad, grad);
  if (g2 <= 1e-6) {
    return vec2<f32>(f32(x), f32(y));
  }
  let normal = normalize(grad);
  let tangentVel = vel - normal * dot(vel, normal);
  return vec2<f32>(f32(x), f32(y)) - p.dt * tangentVel;
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let i = idx(gid.x, gid.y);
  let x = f32(gid.x);
  let y = f32(gid.y);
  let vel = vec2<f32>(vx[i], vy[i]);
  let px = x - p.dt * vel.x;
  let py = y - p.dt * vel.y;

  let n1 = f32(p.n - 1u);
  let sx = u32(clamp(round(px), 0.0, n1));
  let sy = u32(clamp(round(py), 0.0, n1));
  let si = idx(sx, sy);

  let hereObs = obstacleMask[i] > 0.5;
  let srcObs = obstacleMask[si] > 0.5;

  let hereMask = dyeMask[i];
  let srcMask = dyeMask[si];

  let hereR = decodeMode(hereMask, 0u);
  let srcR = decodeMode(srcMask, 0u);
  let blockedR = hereObs || srcObs || hereR == 1u || srcR == 1u;
  if (hereR == 2u || srcR == 2u) {
    r1[i] = 0.0;
  } else if (blockedR) {
    let rp = deflectBacktrace(0u, gid.x, gid.y, vel);
    let rpx = clamp(rp.x, 0.0, n1);
    let rpy = clamp(rp.y, 0.0, n1);
    let rsx = u32(clamp(round(rpx), 0.0, n1));
    let rsy = u32(clamp(round(rpy), 0.0, n1));
    let rsi = idx(rsx, rsy);
    let rSrcMode = decodeMode(dyeMask[rsi], 0u);
    if (rSrcMode == 2u) {
      r1[i] = 0.0;
    } else if (obstacleMask[rsi] > 0.5 || rSrcMode == 1u) {
      r1[i] = r0[i] * p.fade;
    } else {
      r1[i] = sampleBilinear(&r0, rpx, rpy) * p.fade;
    }
  } else {
    r1[i] = sampleBilinear(&r0, px, py) * p.fade;
  }

  let hereG = decodeMode(hereMask, 1u);
  let srcG = decodeMode(srcMask, 1u);
  let blockedG = hereObs || srcObs || hereG == 1u || srcG == 1u;
  if (hereG == 2u || srcG == 2u) {
    g1[i] = 0.0;
  } else if (blockedG) {
    let gp = deflectBacktrace(1u, gid.x, gid.y, vel);
    let gpx = clamp(gp.x, 0.0, n1);
    let gpy = clamp(gp.y, 0.0, n1);
    let gsx = u32(clamp(round(gpx), 0.0, n1));
    let gsy = u32(clamp(round(gpy), 0.0, n1));
    let gsi = idx(gsx, gsy);
    let gSrcMode = decodeMode(dyeMask[gsi], 1u);
    if (gSrcMode == 2u) {
      g1[i] = 0.0;
    } else if (obstacleMask[gsi] > 0.5 || gSrcMode == 1u) {
      g1[i] = g0[i] * p.fade;
    } else {
      g1[i] = sampleBilinear(&g0, gpx, gpy) * p.fade;
    }
  } else {
    g1[i] = sampleBilinear(&g0, px, py) * p.fade;
  }

  let hereB = decodeMode(hereMask, 2u);
  let srcB = decodeMode(srcMask, 2u);
  let blockedB = hereObs || srcObs || hereB == 1u || srcB == 1u;
  if (hereB == 2u || srcB == 2u) {
    b1[i] = 0.0;
  } else if (blockedB) {
    let bp = deflectBacktrace(2u, gid.x, gid.y, vel);
    let bpx = clamp(bp.x, 0.0, n1);
    let bpy = clamp(bp.y, 0.0, n1);
    let bsx = u32(clamp(round(bpx), 0.0, n1));
    let bsy = u32(clamp(round(bpy), 0.0, n1));
    let bsi = idx(bsx, bsy);
    let bSrcMode = decodeMode(dyeMask[bsi], 2u);
    if (bSrcMode == 2u) {
      b1[i] = 0.0;
    } else if (obstacleMask[bsi] > 0.5 || bSrcMode == 1u) {
      b1[i] = b0[i] * p.fade;
    } else {
      b1[i] = sampleBilinear(&b0, bpx, bpy) * p.fade;
    }
  } else {
    b1[i] = sampleBilinear(&b0, px, py) * p.fade;
  }
}
`;

const injectWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read_write> vx: array<f32>;
@group(0) @binding(2) var<storage, read_write> vy: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> g: array<f32>;
@group(0) @binding(5) var<storage, read_write> b: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let cx = f32(p.n) * 0.5;
  let cy = f32(p.n) * 0.5;
  let dx = f32(gid.x) - cx;
  let dy = f32(gid.y) - cy;
  let d2 = dx*dx + dy*dy;
  let i = idx(gid.x, gid.y);
  let r2 = p.radius * p.radius;
  if (d2 < r2) {
    vx[i] = vx[i] + p.impulse;
    vy[i] = vy[i] + 1.2 * sin(f32(i) * 0.0007);
    r[i] = 255.0;
    g[i] = 140.0;
    b[i] = 60.0;
  }
}
`;

const FAST_MODE_FULL_READBACK_INTERVAL = 6;
const FAST_MODE_DYE_ABS_LIMIT = 2048;
const FAST_MODE_HEALTH_CLEAR = new Uint32Array([0, 0, 0, 0]);

const fastFluidHealthWgsl = `
struct HealthParams {
  cellCount: u32,
  velocityLimit: f32,
  dyeLimit: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> p: HealthParams;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> vx: array<f32>;
@group(0) @binding(5) var<storage, read> vy: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;

fn finite(v: f32) -> bool {
  return (v == v) && abs(v) <= 3.402823e38;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.cellCount) { return; }

  let rv = r[i];
  let gv = g[i];
  let bv = b[i];
  let vvx = vx[i];
  let vvy = vy[i];

  if (!(finite(rv) && finite(gv) && finite(bv) && finite(vvx) && finite(vvy))) {
    atomicAdd(&stats[0], 1u);
    return;
  }

  let vLimit = max(1.0, p.velocityLimit);
  let speed2 = vvx * vvx + vvy * vvy;
  if (speed2 > vLimit * vLimit) {
    atomicAdd(&stats[1], 1u);
  }

  let dLimit = max(255.0, p.dyeLimit);
  if (abs(rv) > dLimit || abs(gv) > dLimit || abs(bv) > dLimit) {
    atomicAdd(&stats[2], 1u);
  }
}
`;

async function createPipeline(device, code) {
  const module = device.createShaderModule({ code });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  return {
    pipeline,
    bg(resources) {
      return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((buffer, i) => ({ binding: i, resource: { buffer } }))
      });
    }
  };
}

let running = false;
let sim = null;
let pendingImportedSpecs = [];
let painting = false;
let panning = false;
let panLastX = 0;
let panLastY = 0;

function uploadViscMap() {
  if (!sim) return;
  sim.device.queue.writeBuffer(sim.viscMapGpu, 0, sim.viscMapCpu);
}

function drawGridTextureToView(tmp, n, view, worldScale) {
  const scale = Math.max(1e-6, Number(worldScale) || 1);
  const srcX = Number(view?.x) / scale;
  const srcY = Number(view?.y) / scale;
  const srcW = Number(view?.w) / scale;
  const srcH = Number(view?.h) / scale;
  if (!Number.isFinite(srcX) || !Number.isFinite(srcY) || !Number.isFinite(srcW) || !Number.isFinite(srcH)) return;
  if (srcW <= 0 || srcH <= 0) return;
  ctx.drawImage(tmp, srcX, srcY, srcW, srcH, 0, 0, canvas.width, canvas.height);
}

function drawViscosityOverlay() {
  if (!sim) return;
  const n = sim.controls.n;
  const img = ctx.createImageData(n, n);
  for (let i = 0; i < sim.viscMapCpu.length; i++) {
    const v = Math.max(0, Math.min(1, sim.viscMapCpu[i]));
    const d = v - 0.5; // neutral is transparent
    const o = i * 4;
    if (Math.abs(d) < 0.03) {
      img.data[o] = 0;
      img.data[o + 1] = 0;
      img.data[o + 2] = 0;
      img.data[o + 3] = 0;
      continue;
    }
    const t = Math.min(1, Math.abs(d) / 0.5);
    if (d > 0) {
      img.data[o] = 255;
      img.data[o + 1] = 40;
      img.data[o + 2] = 30;
    } else {
      img.data[o] = 45;
      img.data[o + 1] = 140;
      img.data[o + 2] = 255;
    }
    img.data[o + 3] = Math.floor(145 * t);
  }
  const tmp = document.createElement('canvas');
  tmp.width = n; tmp.height = n;
  tmp.getContext('2d').putImageData(img, 0, 0);
  const view = getCameraView(sim);
  drawGridTextureToView(tmp, n, view, getWorldScale(sim.controls));
}

function resetViscMap() {
  if (!sim) return;
  sim.viscMapCpu.fill(0.5);
  uploadViscMap();
  log({ ok: true, msg: 'viscosity map reset' });
}

function clampCamera(s) {
  const worldSize = getWorldSize(s.controls);
  const cam = s.camera;
  const minZoom = 1;
  const maxZoom = Math.max(1, worldSize / 64);
  cam.zoom = Math.max(minZoom, Math.min(maxZoom, cam.zoom));
  const halfW = worldSize / (2 * cam.zoom);
  const halfH = worldSize / (2 * cam.zoom);
  cam.x = Math.max(halfW, Math.min(worldSize - halfW, cam.x));
  cam.y = Math.max(halfH, Math.min(worldSize - halfH, cam.y));
}

function getCameraView(s) {
  clampCamera(s);
  const worldSize = getWorldSize(s.controls);
  const cam = s.camera;
  const vw = worldSize / cam.zoom;
  const vh = worldSize / cam.zoom;
  return { x: cam.x - vw * 0.5, y: cam.y - vh * 0.5, w: vw, h: vh };
}

function screenToWorld(s, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const nx = (clientX - rect.left) / Math.max(1, rect.width);
  const ny = (clientY - rect.top) / Math.max(1, rect.height);
  const v = getCameraView(s);
  return {
    x: v.x + nx * v.w,
    y: v.y + ny * v.h,
  };
}

function worldToScreen(s, wx, wy) {
  const v = getCameraView(s);
  const sx = ((wx - v.x) / v.w) * canvas.width;
  const sy = ((wy - v.y) / v.h) * canvas.height;
  return { x: sx, y: sy };
}

function paintAt(clientX, clientY, erase = false) {
  if (!sim) return;
  const p = screenToWorld(sim, clientX, clientY);
  const x = p.x;
  const y = p.y;
  const view = getCameraView(sim);
  const r = Math.max(1, Number(brushSizeEl.value) || 12) * (view.w / canvas.width);
  const value = erase ? 0.05 : Math.max(0, Math.min(1, Number(paintValueEl.value) || 0.85));
  const worldScale = getWorldScale(sim.controls);
  const gx = x / worldScale;
  const gy = y / worldScale;
  const gr = r / worldScale;

  const minX = Math.max(0, Math.floor(gx - gr));
  const maxX = Math.min(sim.controls.n - 1, Math.ceil(gx + gr));
  const minY = Math.max(0, Math.floor(gy - gr));
  const maxY = Math.min(sim.controls.n - 1, Math.ceil(gy + gr));

  for (let yy = minY; yy <= maxY; yy++) {
    for (let xx = minX; xx <= maxX; xx++) {
      const dx = xx - gx;
      const dy = yy - gy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > gr) continue;
      const t = 1 - d / gr;
      const idx = yy * sim.controls.n + xx;
      const current = sim.viscMapCpu[idx];
      sim.viscMapCpu[idx] = current * (1 - t) + value * t;
    }
  }
  uploadViscMap();
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  if (!sim) return;
  const panIntent = e.button === 1 || e.altKey || (e.button === 2 && !e.shiftKey);
  if (panIntent) {
    panning = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
    return;
  }
  painting = true;
  paintAt(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
window.addEventListener('mouseup', () => { painting = false; panning = false; });
canvas.addEventListener('mousemove', (e) => {
  if (!sim) return;
  if (panning) {
    const dx = e.clientX - panLastX;
    const dy = e.clientY - panLastY;
    panLastX = e.clientX;
    panLastY = e.clientY;
    const view = getCameraView(sim);
    sim.camera.x -= (dx / canvas.width) * view.w;
    sim.camera.y -= (dy / canvas.height) * view.h;
    clampCamera(sim);
    return;
  }
  if (!painting) return;
  paintAt(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});
canvas.addEventListener('wheel', (e) => {
  if (!sim) return;
  e.preventDefault();
  const before = screenToWorld(sim, e.clientX, e.clientY);
  const zoomMul = e.deltaY < 0 ? 1.12 : (1 / 1.12);
  sim.camera.zoom *= zoomMul;
  clampCamera(sim);
  const after = screenToWorld(sim, e.clientX, e.clientY);
  sim.camera.x += before.x - after.x;
  sim.camera.y += before.y - after.y;
  clampCamera(sim);
}, { passive: false });

function sampleFieldBilinear(field, n, x, y, worldScale = 1) {
  const scale = Math.max(1e-6, Number(worldScale) || 1);
  const gx = Number(x) / scale;
  const gy = Number(y) / scale;
  const cx = Math.max(0, Math.min(n - 1.001, gx));
  const cy = Math.max(0, Math.min(n - 1.001, gy));
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(n - 1, x0 + 1), y1 = Math.min(n - 1, y0 + 1);
  const sx = cx - x0, sy = cy - y0;
  const i00 = y0 * n + x0, i10 = y0 * n + x1, i01 = y1 * n + x0, i11 = y1 * n + x1;

  const v00 = Number(field[i00]);
  const v10 = Number(field[i10]);
  const v01 = Number(field[i01]);
  const v11 = Number(field[i11]);

  const a = (Number.isFinite(v00) ? v00 : 0) * (1 - sx) + (Number.isFinite(v10) ? v10 : 0) * sx;
  const b = (Number.isFinite(v01) ? v01 : 0) * (1 - sx) + (Number.isFinite(v11) ? v11 : 0) * sx;
  const out = a * (1 - sy) + b * sy;
  if (!Number.isFinite(out)) return 0;
  return clampFluidComponent(out, sim?.controls?.fluidCouplingComponentLimit);
}

function sampleObstacleMaskNearest(mask, n, x, y, worldScale = 1) {
  if (!(mask instanceof Float32Array) || mask.length !== n * n) return 0;
  const scale = Math.max(1e-6, Number(worldScale) || 1);
  const gx = Number(x) / scale;
  const gy = Number(y) / scale;
  const xi = Math.max(0, Math.min(n - 1, Math.round(gx)));
  const yi = Math.max(0, Math.min(n - 1, Math.round(gy)));
  return Number(mask[yi * n + xi]) || 0;
}

function sampleFluidForBodyCoupling(
  field,
  n,
  x,
  y,
  dirX,
  dirY,
  obstacleMask = null,
  selfFeedbackField = null,
  selfFeedbackSuppression = 0,
) {
  const suppress = Math.max(0, Math.min(1, Number(selfFeedbackSuppression) || 0));
  const worldScale = getWorldScale(sim?.controls);
  const sampleCouplingField = (sx, sy) => {
    const base = sampleFieldBilinear(field, n, sx, sy, worldScale);
    if (!(selfFeedbackField instanceof Float32Array) || selfFeedbackField.length !== n * n || suppress <= 0) {
      return base;
    }
    const own = sampleFieldBilinear(selfFeedbackField, n, sx, sy, worldScale);
    const v = base - own * suppress;
    if (!Number.isFinite(v)) return 0;
    return clampFluidComponent(v, sim?.controls?.fluidCouplingComponentLimit);
  };

  const samples = [];
  const baseBlocked = sampleObstacleMaskNearest(obstacleMask, n, x, y, worldScale) > 0.5;
  if (!baseBlocked) {
    samples.push({ value: sampleCouplingField(x, y), off: 0 });
  }

  const len = Math.hypot(dirX, dirY);
  if (!Number.isFinite(len) || len < 1e-6) {
    if (samples.length > 0) return samples[0].value;
    return 0;
  }
  const nx = dirX / len;
  const ny = dirY / len;

  // Robust boundary-aware read: avoid selecting extreme recirculation outliers
  // (which can create self-propulsive bias near BLOCK edges).
  const offsets = [1.1, 1.8, 2.6, 3.4, 4.2];
  for (const off of offsets) {
    const sx = x + nx * off;
    const sy = y + ny * off;
    if (sampleObstacleMaskNearest(obstacleMask, n, sx, sy, worldScale) > 0.5) continue;
    samples.push({ value: sampleCouplingField(sx, sy), off });
  }

  if (samples.length === 0) return 0;
  if (samples.length === 1) return samples[0].value;

  const values = samples.map((s) => s.value).sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  const median = (values.length % 2 === 1)
    ? values[mid]
    : 0.5 * (values[mid - 1] + values[mid]);

  // Mild ambient-flow preference from farther-out valid samples.
  const far = samples.filter((s) => s.off >= 2.6).map((s) => s.value);
  const farMean = far.length > 0
    ? (far.reduce((sum, v) => sum + v, 0) / far.length)
    : median;

  // If near-edge samples oppose farther ambient flow, bias toward far flow to
  // avoid bodies re-grabbing their own injected wake and "swimming upstream".
  if (far.length > 0 && median * farMean < 0) {
    return median * 0.2 + farMean * 0.8;
  }

  return median * 0.7 + farMean * 0.3;
}

function rigidVerticesWorld(b) {
  if (Array.isArray(b.verticesLocal) && b.verticesLocal.length >= 3) {
    const th = b.theta || 0;
    const c = Math.cos(th), s = Math.sin(th);
    return b.verticesLocal.map((v) => ({
      x: b.x + v.x * c - v.y * s,
      y: b.y + v.x * s + v.y * c,
    }));
  }
  const sides = Math.max(3, b.sides || 3);
  const rot = (b.theta || 0) + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: b.x + Math.cos(a) * b.r, y: b.y + Math.sin(a) * b.r });
  }
  return verts;
}

function stepRigidBodiesBaselineCpu({
  sim,
  bodies,
  vxField,
  vyField,
  n,
  worldSize,
  dt,
  dtNorm,
  dragK,
  swimGain,
  localHoneyDrag,
  viscosityMotionResponse,
  obstacleMask,
  bodyFeedbackPrevVx,
  bodyFeedbackPrevVy,
  selfFeedbackSuppression,
  rigidVerticesWorld,
  sampleFluidForBodyCoupling,
  applyBounceBoundary,
  allowPassEdgeFlowPush,
}) {
  let rigidCarryTransfer = 0;
  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const edgeMomentumScale = rigidEdgeMomentumScale(b, allowPassEdgeFlowPush);

    const invMass = 1 / Math.max(0.05, b.mass);
    const invInertia = 1 / Math.max(0.05, b.inertia || 1);
    const sampleVerts = rigidVerticesWorld(b);
    const sampleCount = Math.max(1, sampleVerts.length);
    let forceX = 0;
    let forceY = 0;
    let torque = 0;

    for (let si = 0; si < sampleCount; si++) {
      const sx = sampleVerts[si].x;
      const sy = sampleVerts[si].y;
      const rx = sx - b.x;
      const ry = sy - b.y;
      const fx = sampleFluidForBodyCoupling(
        vxField,
        n,
        sx,
        sy,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVx,
        selfFeedbackSuppression,
      );
      const fy = sampleFluidForBodyCoupling(
        vyField,
        n,
        sx,
        sy,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVy,
        selfFeedbackSuppression,
      );
      const localVx = b.vx + (-(b.omega || 0) * ry);
      const localVy = b.vy + ((b.omega || 0) * rx);
      const relX = fx - localVx;
      const relY = fy - localVy;
      const honey = localHoneyDrag(sx, sy);
      const fpx = relX * dragK * honey * edgeMomentumScale;
      const fpy = relY * dragK * honey * edgeMomentumScale;
      forceX += fpx;
      forceY += fpy;
      torque += rx * fpy - ry * fpx;
    }

    forceX /= sampleCount;
    forceY /= sampleCount;
    torque /= sampleCount;

    const ax = forceX * invMass;
    const ay = forceY * invMass;
    const alpha = torque * invInertia;

    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    const swimX = swimGain * Math.cos(swimPhase) * 0.012 * invMass;
    const swimY = swimGain * Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    const swimTorque = swimGain * Math.sin(swimPhase * 1.1) * 0.0025;

    b.vx += ax * dt * 60 + swimX * dtNorm;
    b.vy += ay * dt * 60 + swimY * dtNorm;
    b.omega = (b.omega || 0) + alpha * dt * 60 + swimTorque * dtNorm;

    const centerHoney = localHoneyDrag(b.x, b.y);
    const rigidVisc = viscosityMotionResponse(centerHoney, 3.2);
    b.vx *= rigidVisc.damp;
    b.vy *= rigidVisc.damp;
    b.omega *= Math.max(0.72, 0.99 - 0.01 * centerHoney);

    const bMax = rigidVisc.vmax;
    const bMag = Math.hypot(b.vx, b.vy);
    if (bMag > bMax) {
      b.vx = (b.vx / bMag) * bMax;
      b.vy = (b.vy / bMag) * bMax;
    }
    b.omega = Math.max(-0.25, Math.min(0.25, b.omega));

    rigidCarryTransfer += Math.hypot(ax, ay);
    b.x = b.x + b.vx * dt * 22;
    b.y = b.y + b.vy * dt * 28;
    b.theta = (b.theta || 0) + b.omega * dt * 60;
    applyBounceBoundary(b, worldSize, 0.84);
  }
  return rigidCarryTransfer;
}

function rigidVertexWorld(b, vertexIndex) {
  const verts = rigidVerticesWorld(b);
  const n = verts.length || 1;
  return verts[((vertexIndex % n) + n) % n];
}

function polygonSignedAreaPoints(points) {
  let s = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    s += a.x * b.y - b.x * a.y;
  }
  return 0.5 * s;
}

function pushMembraneCellCluster({
  softNodes,
  springs,
  clusterId,
  cx,
  cy,
  radius,
  nodeCount,
  controls,
  scale,
  bodyScale,
  nodeRadiusScale = 1,
}) {
  const base = softNodes.length;
  const ring = [];

  for (let i = 0; i < nodeCount; i++) {
    const a = (i / nodeCount) * Math.PI * 2;
    const wobble = 1 + 0.08 * Math.sin(i * 1.7 + clusterId * 0.61) + (Math.random() - 0.5) * 0.04;
    const rr = Math.max(0.5, radius * wobble);
    ring.push({ x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr });
  }

  const digestRGB = (clusterId % 3 === 0)
    ? [0.95, 0.3, 0.3]
    : ((clusterId % 3 === 1) ? [0.3, 0.95, 0.3] : [0.3, 0.3, 0.95]);

  for (const p of ring) {
    softNodes.push({
      x: p.x,
      y: p.y,
      vx: 0,
      vy: 0,
      mass: controls.massSoft,
      r: 1.35 * scale * bodyScale * Math.max(0.45, Number(nodeRadiusScale) || 1),
      clusterId,
      digestEnabled: false,
      digestRGB,
      membraneCell: true,
      shapeMemoryWeight: 1,
    });
  }

  // Membrane ring (BLOCK for contact, mixed dye permeability profile for semi-permeable feel).
  for (let i = 0; i < nodeCount; i++) {
    const j = (i + 1) % nodeCount;
    const a = ring[i];
    const b = ring[j];
    springs.push([
      base + i,
      base + j,
      Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
      EDGE_BODY_MODE.BLOCK,
      [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS],
    ]);
  }

  return {
    clusterId,
    restArea: Math.max(1e-4, Math.abs(polygonSignedAreaPoints(ring))),
    pressureGain: MEMBRANE_CELL_BASE_PRESSURE_GAIN,
    radialDamping: MEMBRANE_CELL_BASE_RADIAL_DAMPING,
    shapeMemoryGain: MEMBRANE_SHAPE_MEMORY_GAIN,
    insideCorrectionEnabled: 1,
  };
}

function initBodies(n, controls, worldSizeOverride = null) {
  const scale = n / 256;
  const bigMode = n >= 1024;
  const bodyScale = bigMode ? 0.5 : 1.0;
  const worldScale = getWorldScale(controls);
  const worldSize = Number(worldSizeOverride) || (n * worldScale);
  // Keep the same primitive catalog across 128/256/512/1024+ so
  // deformation differences are easier to attribute to fluid resolution.
  const rigidCount = (controls?.seedRigidBodies === false)
    ? 0
    : Math.max(1, Math.min(2000, Math.round(Number(controls?.rigidBodyCount) || 10)));
  const softClusterCount = (controls?.seedSpringSoftBodies === false) ? 0 : 10;

  const rigidShapeCycle = [3, 4, 5, 6];
  const rigid = [];
  const membraneCount = controls.spawnMembraneCells
    ? Math.max(2, Math.round(MEMBRANE_CELL_BASE_COUNT * Math.min(2.0, Math.max(0.7, scale))))
    : 0;

  // High body counts (especially on 1024/2048 grids) need structured placement
  // and adaptive sizing so rigid/soft/membrane seeds don't begin in dense clumps.
  const spawnMinX = worldSize * 0.08;
  const spawnMaxX = worldSize * 0.92;
  const spawnMinY = worldSize * 0.12;
  const spawnMaxY = worldSize * 0.88;
  const spawnW = Math.max(1, spawnMaxX - spawnMinX);
  const spawnH = Math.max(1, spawnMaxY - spawnMinY);
  const spawnAspect = spawnW / spawnH;
  const totalSeedGroups = Math.max(1, rigidCount + softClusterCount + membraneCount);
  const cols = Math.max(1, Math.ceil(Math.sqrt(totalSeedGroups * spawnAspect)));
  const rows = Math.max(1, Math.ceil(totalSeedGroups / cols));
  const cellW = spawnW / cols;
  const cellH = spawnH / rows;
  const jitterFrac = totalSeedGroups > 64 ? 0.2 : 0.33;
  const crowdScale = rigidCount <= 24
    ? 1
    : Math.max(0.35, Math.min(1, Math.sqrt(24 / Math.max(1, rigidCount))));
  const softRadiusScale = Math.max(0.45, crowdScale);
  const softNodeRadiusScale = Math.max(0.55, Math.sqrt(softRadiusScale));

  const seedCenterAt = (seedIndex, radiusPad = 0) => {
    const idx = Math.max(0, Number(seedIndex) || 0);
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    const jitterX = (Math.random() - 0.5) * cellW * jitterFrac;
    const jitterY = (Math.random() - 0.5) * cellH * jitterFrac;
    const baseX = spawnMinX + (col + 0.5) * cellW;
    const baseY = spawnMinY + (row + 0.5) * cellH;
    const pad = Math.max(0, Number(radiusPad) || 0);
    return {
      x: clamp(baseX + jitterX, spawnMinX + pad, spawnMaxX - pad),
      y: clamp(baseY + jitterY, spawnMinY + pad, spawnMaxY - pad),
    };
  };

  for (let i = 0; i < rigidCount; i++) {
    const mass = (i % 2 === 0) ? controls.massLight : controls.massHeavy;
    const sides = rigidShapeCycle[i % rigidShapeCycle.length];
    const baseR = ((i % 2 === 0) ? 5 : 6) * scale * bodyScale * (sides >= 5 ? 0.95 : 1.0);
    const r = Math.max(0.9 * scale * bodyScale, baseR * crowdScale);
    const edgeDyeMode = Array.from({ length: sides }, (_, ei) => {
      const phase = (ei + i) % 3;
      if (phase === 0) return [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB];
      if (phase === 1) return [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT];
      return [EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS];
    });
    const edgeBodyMode = Array.from({ length: sides }, (_, ei) => ((ei + i) % 2 === 0) ? EDGE_BODY_MODE.BLOCK : EDGE_BODY_MODE.PASS);
    const edgePermeabilityRGB = Array.from({ length: sides }, () => [0, 0, 0]);
    const digestRGB = (i % 3 === 0) ? [1, 0.2, 0.2] : ((i % 3 === 1) ? [0.2, 1, 0.2] : [0.2, 0.2, 1]);
    const consumeDyeRGB = (i % 2) === 0 ? [1, 1, 1] : [0, 0, 0];

    const center = seedCenterAt(i, r);

    rigid.push({
      x: center.x,
      y: center.y,
      vx: 0,
      vy: 0,
      r,
      sides,
      edgeDyeMode,
      edgeBodyMode,
      edgePermeabilityRGB,
      digestEnabled: (i % 2) === 0,
      digestRGB,
      consumeDyeRGB,
      insideCorrectionEnabled: true,
      mass,
      theta: Math.random() * Math.PI * 2,
      omega: 0,
      inertia: 0.5 * mass * r * r,
    });
  }

  const softNodes = [];
  const springs = [];
  const softMembraneClusters = [];
  const softShapeCycle = [3, 4, 6];

  for (let c = 0; c < softClusterCount; c++) {
    const nodeCount = softShapeCycle[c % softShapeCycle.length];
    const radius = (8 + (nodeCount === 6 ? 2 : 0)) * scale * bodyScale * softRadiusScale;
    const center = seedCenterAt(rigidCount + c, radius);
    const cx = center.x;
    const cy = center.y;
    const base = softNodes.length;

    const local = [];
    for (let i = 0; i < nodeCount; i++) {
      const a = (i / nodeCount) * Math.PI * 2;
      local.push({ x: cx + Math.cos(a) * radius, y: cy + Math.sin(a) * radius });
    }

    const clusterDigestRGB = (c % 3 === 0) ? [1, 0.15, 0.15] : ((c % 3 === 1) ? [0.15, 1, 0.15] : [0.15, 0.15, 1]);
    for (const p of local) {
      softNodes.push({
        x: p.x,
        y: p.y,
        vx: 0,
        vy: 0,
        mass: controls.massSoft,
        r: 1.4 * scale * bodyScale * softNodeRadiusScale,
        clusterId: c,
        digestEnabled: (c % 2) === 0,
        digestRGB: clusterDigestRGB,
      });
    }

    // Ring springs (solid edges: rigid bodies should collide with them).
    for (let i = 0; i < nodeCount; i++) {
      const j = (i + 1) % nodeCount;
      const a = local[i], b = local[j];
      const ringDyeMode = ((i + c) % 3 === 0)
        ? [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB]
        : (((i + c) % 3 === 1)
            ? [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT]
            : [EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS]);
      springs.push([
        base + i,
        base + j,
        Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
        EDGE_BODY_MODE.BLOCK,
        ringDyeMode,
      ]);
    }
    // Cross/diagonal springs for shape retention (internal, non-solid by default).
    for (let i = 0; i < nodeCount; i++) {
      const j = (i + 2) % nodeCount;
      if (i < j || nodeCount <= 4) {
        const a = local[i], b = local[j];
        springs.push([
          base + i,
          base + j,
          Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
          EDGE_BODY_MODE.PASS,
          [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS],
        ]);
      }
    }
    // Opposite braces for even polygons (especially hex) to prevent skew collapse.
    if (nodeCount % 2 === 0) {
      for (let i = 0; i < nodeCount / 2; i++) {
        const j = (i + nodeCount / 2) % nodeCount;
        const a = local[i], b = local[j];
        springs.push([
          base + i,
          base + j,
          Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
          EDGE_BODY_MODE.PASS,
          [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS],
        ]);
      }
    }
    // Hex-specific mirror chords to remove the last skew mode and keep visual symmetry.
    if (nodeCount === 6) {
      const extraPairs = [[0, 4], [1, 5]];
      for (const [i, j] of extraPairs) {
        const a = local[i], b = local[j];
        springs.push([
          base + i,
          base + j,
          Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
          EDGE_BODY_MODE.PASS,
          [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS],
        ]);
      }
    }
  }

  if (controls.spawnMembraneCells) {
    for (let ci = 0; ci < membraneCount; ci++) {
      const nodeCount = 10 + ((ci % 3) * 2);
      const radius = (7.2 + (ci % 2) * 1.8) * scale * bodyScale * softRadiusScale;
      const center = seedCenterAt(rigidCount + softClusterCount + ci, radius);
      const cx = center.x;
      const cy = center.y;
      const clusterId = softClusterCount + ci;
      const membrane = pushMembraneCellCluster({
        softNodes,
        springs,
        clusterId,
        cx,
        cy,
        radius,
        nodeCount,
        controls,
        scale,
        bodyScale,
        nodeRadiusScale: softNodeRadiusScale,
      });
      softMembraneClusters.push(membrane);
    }
  }

  const hybrid = [];

  return { rigid, soft: { nodes: softNodes, springs }, hybrid, softMembraneClusters };
}

function initEmitters(n) {
  if (n < 1024) return [];
  const count = n >= 2048 ? 16 : 10;
  const palette = [
    [255, 70, 50],
    [50, 170, 255],
    [255, 220, 70],
    [170, 90, 255],
    [80, 255, 170],
  ];
  const emitters = [];
  for (let i = 0; i < count; i++) {
    const c = palette[i % palette.length];
    emitters.push({
      x: n * (0.1 + 0.8 * Math.random()),
      y: n * (0.1 + 0.8 * Math.random()),
      vx: (Math.random() * 2 - 1) * 0.2,
      vy: (Math.random() * 2 - 1) * 0.2,
      r: (n >= 2048 ? 14 : 10) + Math.random() * 6,
      cr: c[0],
      cg: c[1],
      cb: c[2],
      strength: n >= 2048 ? 1.6 : 1.2,
      spin: (Math.random() < 0.5 ? -1 : 1) * (0.45 + Math.random() * 0.55),
      curlGain: 1.1,
      driftGain: 0.014,
      chaosGain: 1.1,
      wobbleAmp: Math.max(0.6, n / 220),
      wobbleFreq: 0.045 + Math.random() * 0.02,
      swirlJitter: Math.random() * Math.PI * 2,
    });
  }
  return emitters;
}

function buildPresetEmitters(n, preset) {
  const emitters = [];
  const mk = (xf, yf, vxf, vyf, color, radius, strength, spin = 0.8, curlGain = 1.25, driftGain = 0.012, chaosGain = 1.2, wobbleAmp = 0, wobbleFreq = 0.05) => ({
    x: n * xf,
    y: n * yf,
    vx: vxf,
    vy: vyf,
    r: radius,
    cr: color[0],
    cg: color[1],
    cb: color[2],
    strength,
    spin,
    curlGain,
    driftGain,
    chaosGain,
    wobbleAmp,
    wobbleFreq,
    swirlJitter: Math.random() * Math.PI * 2,
  });

  if (preset === 'vortexGarden') {
    const colors = [[255,90,70],[80,170,255],[255,220,90],[190,90,255],[90,255,190]];
    const count = Math.max(8, Math.floor(n / 120));
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2;
      const rf = 0.23 + (i % 3) * 0.06;
      emitters.push(mk(
        0.5 + Math.cos(a) * rf,
        0.5 + Math.sin(a) * rf,
        -Math.sin(a) * 0.22,
        Math.cos(a) * 0.22,
        colors[i % colors.length],
        Math.max(8, n / 42),
        1.35,
        (i % 2 ? 1 : -1) * 1.15,
        1.35,
        0.01,
      ));
    }
  } else if (preset === 'shearCanals') {
    const lanes = 5;
    for (let i = 0; i < lanes; i++) {
      const y = 0.16 + (i / (lanes - 1)) * 0.68;
      const dir = i % 2 === 0 ? 1 : -1;
      const wobbleAmp = Math.max(2.5, n / 60);
      const wobbleFreq = 0.08 + i * 0.01;
      emitters.push(mk(0.08, y, 0.34 * dir, 0.05 * dir, [255,150,80], Math.max(7, n / 55), 1.12, 2.1 * dir, 1.9, 0.004, 2.3, wobbleAmp, wobbleFreq));
      emitters.push(mk(0.92, y, -0.34 * dir, -0.05 * dir, [90,180,255], Math.max(7, n / 55), 1.12, 2.1 * dir, 1.9, 0.004, 2.3, wobbleAmp, wobbleFreq * 1.07));
    }
  } else if (preset === 'islands') {
    const centers = [[0.22,0.24],[0.78,0.28],[0.30,0.75],[0.75,0.72],[0.52,0.50]];
    const colors = [[255,110,80],[70,160,255],[255,220,90],[180,90,255],[90,255,180]];
    for (let i = 0; i < centers.length; i++) {
      const c = centers[i];
      emitters.push(mk(c[0], c[1], (Math.random()*2-1)*0.12, (Math.random()*2-1)*0.12, colors[i], Math.max(10, n / 35), 1.45, (i%2?1:-1)*1.2, 1.4, 0.009));
    }
  } else if (preset === 'checkerPlumes') {
    const cols = 4, rows = 4;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const xf = 0.16 + x * 0.22;
        const yf = 0.16 + y * 0.22;
        const odd = (x + y) % 2 === 1;
        const color = odd ? [255,120,70] : [90,170,255];
        emitters.push(mk(xf, yf, odd ? 0.15 : -0.15, odd ? -0.08 : 0.08, color, Math.max(6, n / 65), 0.95, odd ? 1.25 : -1.25, 1.45, 0.008));
      }
    }
  } else {
    return initEmitters(n);
  }
  return emitters;
}

function applyPresetViscosityTerrain(sim, preset) {
  const n = sim.controls.n;
  const m = sim.viscMapCpu;
  m.fill(0.5);
  const addBlob = (xf, yf, rf, target) => {
    const cx = n * xf, cy = n * yf, rr = n * rf;
    const minX = Math.max(0, Math.floor(cx - rr));
    const maxX = Math.min(n - 1, Math.ceil(cx + rr));
    const minY = Math.max(0, Math.floor(cy - rr));
    const maxY = Math.min(n - 1, Math.ceil(cy + rr));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const d = Math.hypot(x - cx, y - cy);
        if (d > rr) continue;
        const w = 1 - d / rr;
        const i = y * n + x;
        m[i] = m[i] * (1 - w) + target * w;
      }
    }
  };

  if (preset === 'vortexGarden') {
    addBlob(0.5, 0.5, 0.26, 0.15);
    addBlob(0.5, 0.5, 0.42, 0.78);
  } else if (preset === 'shearCanals') {
    for (let i = 0; i < 6; i++) {
      addBlob(0.5, 0.12 + i * 0.15, 0.06, i % 2 === 0 ? 0.2 : 0.82);
    }
  } else if (preset === 'islands') {
    addBlob(0.24, 0.24, 0.16, 0.88);
    addBlob(0.76, 0.30, 0.14, 0.84);
    addBlob(0.30, 0.76, 0.15, 0.86);
    addBlob(0.74, 0.72, 0.13, 0.82);
    addBlob(0.52, 0.50, 0.19, 0.18);
  } else if (preset === 'checkerPlumes') {
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        addBlob(0.1 + x * 0.2, 0.1 + y * 0.2, 0.07, ((x + y) % 2 === 0) ? 0.2 : 0.8);
      }
    }
  }

  uploadViscMap();
}

function cloneMiniBodies(miniBodies, controls) {
  const rigid = Array.isArray(miniBodies?.rigid)
    ? miniBodies.rigid.map((rb) => {
        const sides = Math.max(3, Number(rb?.sides) || 3);
        const edgeDyeMode = Array.isArray(rb?.edgeDyeMode)
          ? rb.edgeDyeMode.map((m) => normalizeEdgeDyeModeRGB(m))
          : Array.from({ length: sides }, () => [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]);
        const edgeBodyMode = Array.isArray(rb?.edgeBodyMode)
          ? rb.edgeBodyMode.map((m) => Number(m) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK)
          : Array.from({ length: sides }, () => EDGE_BODY_MODE.BLOCK);
        const edgePermeabilityRGB = Array.isArray(rb?.edgePermeabilityRGB)
          ? rb.edgePermeabilityRGB.map((m) => [Number(m?.[0]) > 0 ? 1 : 0, Number(m?.[1]) > 0 ? 1 : 0, Number(m?.[2]) > 0 ? 1 : 0])
          : Array.from({ length: sides }, () => [0, 0, 0]);
        const edgeVelocityMode = Array.isArray(rb?.edgeVelocityMode)
          ? rb.edgeVelocityMode.map((m) => Number(m) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK)
          : [...edgeBodyMode];
        const edgeMomentumTransfer = Array.isArray(rb?.edgeMomentumCoupling)
          ? rb.edgeMomentumCoupling.map((m) => clamp(Number(m) || 0, 0, 1))
          : (Array.isArray(rb?.edgeMomentumTransfer)
            ? rb.edgeMomentumTransfer.map((m) => clamp(Number(m) || 0, 0, 1))
            : Array.from({ length: sides }, () => 1));
        const mass = Math.max(0.05, Number(rb?.mass) || controls.massHeavy || 3.5);
        const r = Math.max(1.0, Number(rb?.r) || 6);
        return {
          x: Number(rb?.x) || 0,
          y: Number(rb?.y) || 0,
          vx: Number(rb?.vx) || 0,
          vy: Number(rb?.vy) || 0,
          r,
          sides,
          theta: Number(rb?.theta) || 0,
          omega: Number(rb?.omega) || 0,
          edgeDyeMode,
          edgeBodyMode,
          edgePermeabilityRGB,
          edgeVelocityMode,
          edgeMomentumTransfer,
          digestEnabled: Boolean(rb?.digestEnabled),
          digestRGB: Array.isArray(rb?.digestRGB) ? [Number(rb.digestRGB[0]) || 0, Number(rb.digestRGB[1]) || 0, Number(rb.digestRGB[2]) || 0] : [0, 0, 0],
          consumeDyeRGB: Array.isArray(rb?.consumeDyeRGB) ? [Number(rb.consumeDyeRGB[0]) > 0 ? 1 : 0, Number(rb.consumeDyeRGB[1]) > 0 ? 1 : 0, Number(rb.consumeDyeRGB[2]) > 0 ? 1 : 0] : [0, 0, 0],
          insideCorrectionEnabled: rb?.insideCorrectionEnabled !== false,
          mass,
          inertia: Math.max(0.05, Number(rb?.inertia) || (0.5 * mass * r * r)),
          verticesLocal: Array.isArray(rb?.verticesLocal)
            ? rb.verticesLocal.map((v) => ({ x: Number(v?.x) || 0, y: Number(v?.y) || 0 }))
            : undefined,
        };
      })
    : [];

  const softNodes = Array.isArray(miniBodies?.soft?.nodes)
    ? miniBodies.soft.nodes.map((n) => ({
        x: Number(n?.x) || 0,
        y: Number(n?.y) || 0,
        vx: Number(n?.vx) || 0,
        vy: Number(n?.vy) || 0,
        mass: Math.max(0.02, Number(n?.mass) || controls.massSoft || 0.6),
        r: Math.max(0.2, Number(n?.r) || 1.2),
        clusterId: Number.isFinite(Number(n?.clusterId)) ? Number(n.clusterId) : 0,
        digestEnabled: Boolean(n?.digestEnabled),
        digestRGB: Array.isArray(n?.digestRGB) ? [Number(n.digestRGB[0]) || 0, Number(n.digestRGB[1]) || 0, Number(n.digestRGB[2]) || 0] : [0, 0, 0],
        shapeMemoryWeight: Number.isFinite(Number(n?.shapeMemoryWeight)) ? clamp(Number(n.shapeMemoryWeight), 0, 1) : 1,
      }))
    : [];

  const sanitized = sanitizeSoftSprings(miniBodies?.soft?.springs, softNodes.length, {
    edgeBodyPass: EDGE_BODY_MODE.PASS,
    edgeBodyBlock: EDGE_BODY_MODE.BLOCK,
    restFloor: 1e-3,
  });
  const softSprings = sanitized.springs.map((sp) => [
    Number(sp[0]) | 0,
    Number(sp[1]) | 0,
    Math.max(1e-3, Number(sp[2]) || 1),
    Number(sp[3]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK,
    normalizeEdgeDyeModeRGB(sp[4]),
    Number(sp[5]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK,
    Number.isFinite(Number(sp[6])) ? clamp(Number(sp[6]), 0, 1) : 1,
  ]);

  const hybrid = [];

  const softMembraneClusters = Array.isArray(miniBodies?.softMembraneClusters)
    ? miniBodies.softMembraneClusters
        .map((c) => ({
          clusterId: Number(c?.clusterId),
          restArea: Number(c?.restArea),
          pressureGain: Number(c?.pressureGain),
          radialDamping: Number(c?.radialDamping),
          shapeMemoryGain: Number(c?.shapeMemoryGain),
          insideCorrectionEnabled: Number(c?.insideCorrectionEnabled),
        }))
        .filter((c) => Number.isInteger(c.clusterId))
        .map((c) => ({
          clusterId: c.clusterId,
          restArea: Number.isFinite(c.restArea) && c.restArea > 0 ? c.restArea : 1,
          pressureGain: Number.isFinite(c.pressureGain) && c.pressureGain > 0 ? c.pressureGain : MEMBRANE_CELL_BASE_PRESSURE_GAIN,
          radialDamping: Number.isFinite(c.radialDamping) ? clamp(c.radialDamping, 0, 0.2) : MEMBRANE_CELL_BASE_RADIAL_DAMPING,
          shapeMemoryGain: Number.isFinite(c.shapeMemoryGain) ? clamp(c.shapeMemoryGain, 0, 0.35) : MEMBRANE_SHAPE_MEMORY_GAIN,
          insideCorrectionEnabled: Number.isFinite(c.insideCorrectionEnabled) ? (c.insideCorrectionEnabled > 0 ? 1 : 0) : 1,
        }))
    : [];

  return {
    rigid,
    soft: { nodes: softNodes, springs: softSprings },
    hybrid,
    softMembraneClusters,
    topologyGuardrails: {
      droppedSoftSprings: sanitized.dropped,
    },
  };
}

function getMiniScenarioFromPreset(preset) {
  const p = String(preset || '');
  if (!p.startsWith('mini:')) return null;
  const id = p.slice(5);
  return generatedMiniScenarios.get(id) || null;
}

function ensureGridOption(value) {
  const s = String(value);
  if (![...gridEl.options].some((o) => o.value === s)) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    gridEl.appendChild(opt);
  }
}

function ensureMiniScenarioGridSelection(preset) {
  const mini = getMiniScenarioFromPreset(preset);
  if (!mini) return null;
  const requiredGrid = Math.max(32, Number(mini.grid) || 100);
  ensureGridOption(requiredGrid);
  if (String(gridEl.value) !== String(requiredGrid)) {
    gridEl.value = String(requiredGrid);
  }
  return mini;
}

function applyMiniScenarioPreset(sim, mini) {
  sim.bodies = cloneMiniBodies(mini?.bodies, sim.controls);
  sim.emitters = Array.isArray(mini?.emitters)
    ? mini.emitters.map((e) => ({ ...e }))
    : [];

  sim.frame = 0;
  sim.couplingTelemetry = [];
  sim.lastRigidContacts = [];
  sim.softXPBDLambda = null;
  sim.softSpringRestBaseline = null;
  sim.softAreaRest = null;
  sim.softAreaLambda = null;
  sim.softMembraneAreaBaseline = null;
  sim.softMembraneLoopState = null;
  sim.softMembraneClusterSet = null;
  sim.softMembraneClusterMap = null;
  sim.softDeformReferenceState = null;
  sim.viscMapCpu.fill(0.5);
  uploadViscMap();
  focusCameraOnBodies(sim, sim.bodies);

  log({
    ok: true,
    msg: 'mini scenario loaded',
    id: mini.id,
    combo: mini.combo,
    seed: mini.seed,
    steps: mini.steps,
    grid: mini.grid,
    droppedSoftSprings: sim.bodies?.topologyGuardrails?.droppedSoftSprings || 0,
  });
}

async function loadGeneratedMiniScenarios() {
  if (!scenarioPresetEl) return;
  const previous = scenarioPresetEl.value;
  try {
    const res = await fetch(`${GENERATED_MINI_SCENARIOS_URL}?t=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) return;
    const payload = await res.json();
    const scenarios = Array.isArray(payload?.scenarios) ? payload.scenarios : [];

    for (const opt of [...scenarioPresetEl.querySelectorAll('option[data-generated-mini="1"]')]) {
      opt.remove();
    }

    generatedMiniScenarios = new Map();
    const sorted = [...scenarios]
      .filter((s) => s && s.id && s.bodies)
      .sort((a, b) => String(b.createdAt || '').localeCompare(String(a.createdAt || '')))
      .slice(0, 24);

    for (const s of sorted) {
      const id = String(s.id);
      generatedMiniScenarios.set(id, s);
      const opt = document.createElement('option');
      opt.value = `mini:${id}`;
      opt.dataset.generatedMini = '1';
      const combo = s.combo || 'mixed';
      const steps = Number(s.steps) || 0;
      opt.textContent = `${s.name || `Mini ${id}`} (${combo}, ${steps} steps)`;
      scenarioPresetEl.appendChild(opt);
    }

    if ([...scenarioPresetEl.options].some((o) => o.value === previous)) {
      scenarioPresetEl.value = previous;
    }
  } catch {
    // Optional file; ignore when absent.
  }
}

function applyScenarioPreset(sim, preset) {
  if (!sim) return;
  const p = preset || 'baseline';
  const mini = getMiniScenarioFromPreset(p);
  if (mini) {
    if (Number(mini.grid) !== sim.controls.n) {
      log({
        ok: false,
        msg: 'mini scenario requires matching grid; select preset again to reinit',
        id: mini.id,
        requiredGrid: mini.grid,
        currentGrid: sim.controls.n,
      });
      return;
    }
    applyMiniScenarioPreset(sim, mini);
    return;
  }

  const worldSize = getWorldSize(sim.controls);
  sim.emitters = buildPresetEmitters(worldSize, p);
  applyPresetViscosityTerrain(sim, p);
  sim.camera.x = worldSize * 0.5;
  sim.camera.y = worldSize * 0.5;
  sim.camera.zoom = 1.0;
}

function applyEmitters(sim, r, g, b, vx, vy) {
  const n = sim.controls.n;
  const worldScale = getWorldScale(sim.controls);
  const worldSize = getWorldSize(sim.controls);
  for (const e of sim.emitters || []) {
    const lockedPosition = !!e.lockPosition;
    if (!lockedPosition) {
      e.x += e.vx;
      e.y += e.vy;
      if (e.x < e.r || e.x > worldSize - e.r) e.vx *= -1;
      if (e.y < e.r || e.y > worldSize - e.r) e.vy *= -1;
      e.x = Math.max(e.r, Math.min(worldSize - e.r, e.x));
      e.y = Math.max(e.r, Math.min(worldSize - e.r, e.y));
    } else {
      e.x = Math.max(e.r, Math.min(worldSize - e.r, Number(e.x) || (worldSize * 0.5)));
      e.y = Math.max(e.r, Math.min(worldSize - e.r, Number(e.y) || (worldSize * 0.12)));
    }

    const wobbleAmpRaw = Number.isFinite(Number(e.wobbleAmp)) ? Number(e.wobbleAmp) : 0;
    const wobbleAmp = lockedPosition ? 0 : wobbleAmpRaw;
    const wobbleFreq = Number.isFinite(Number(e.wobbleFreq)) ? Number(e.wobbleFreq) : 0.05;
    const wobblePhase = sim.frame * wobbleFreq + (Number(e.swirlJitter) || 0);
    const ex = e.x + Math.cos(wobblePhase * 1.31) * wobbleAmp;
    const ey = e.y + Math.sin(wobblePhase * 1.73) * wobbleAmp;
    const exGrid = ex / worldScale;
    const eyGrid = ey / worldScale;
    const rGrid = Math.max(1e-6, Number(e.r) / worldScale);

    const minX = Math.max(0, Math.floor(exGrid - rGrid));
    const maxX = Math.min(n - 1, Math.ceil(exGrid + rGrid));
    const minY = Math.max(0, Math.floor(eyGrid - rGrid));
    const maxY = Math.min(n - 1, Math.ceil(eyGrid + rGrid));
    const pulse = 0.75 + 0.25 * Math.sin(sim.frame * 0.03 + e.swirlJitter);
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - exGrid;
        const dy = y - eyGrid;
        const d = Math.hypot(dx, dy);
        if (d > rGrid) continue;
        const nd = d / Math.max(1e-6, rGrid);
        const w = (1 - nd) * e.strength * pulse;
        const i = y * n + x;
        r[i] = Math.min(255, r[i] + e.cr * 0.03 * w);
        g[i] = Math.min(255, g[i] + e.cg * 0.03 * w);
        b[i] = Math.min(255, b[i] + e.cb * 0.03 * w);

        const tx = d > 1e-6 ? (-dy / d) : 0;
        const ty = d > 1e-6 ? (dx / d) : 0;
        const rx = d > 1e-6 ? (dx / d) : 0;
        const ry = d > 1e-6 ? (dy / d) : 0;
        const spin = Number.isFinite(Number(e.spin)) ? Number(e.spin) : 0.9;
        const curlGain = Number.isFinite(Number(e.curlGain)) ? Number(e.curlGain) : 1.0;
        const driftGain = Number.isFinite(Number(e.driftGain)) ? Number(e.driftGain) : 0.016;
        const chaosGain = Number.isFinite(Number(e.chaosGain)) ? Number(e.chaosGain) : 1.0;
        const swirlProfile = (1 - nd) * (0.45 + 0.55 * nd);
        const swirlOsc = 0.9 + 0.1 * Math.sin(sim.frame * 0.08 + e.swirlJitter * 1.7);
        const swirl = spin * 0.22 * swirlProfile * swirlOsc * curlGain;
        const radialPulse = Math.sin(sim.frame * 0.12 + nd * 11.0 + e.swirlJitter * 0.7);
        const ringPulse = Math.cos(sim.frame * 0.06 + nd * 15.0 + e.swirlJitter * 0.5);
        const radial = spin * 0.1 * (1 - nd) * radialPulse * chaosGain;
        const ring = spin * 0.075 * (1 - nd) * (1 - nd) * ringPulse * chaosGain;
        vx[i] += (e.vx * driftGain + tx * swirl + rx * radial + tx * ring) * w;
        vy[i] += (e.vy * driftGain + ty * swirl + ry * radial + ty * ring) * w;
      }
    }
  }
}

function computeSoftCentroid(nodes) {
  let sx = 0;
  let sy = 0;
  for (const node of nodes) {
    sx += node.x;
    sy += node.y;
  }
  const inv = nodes.length > 0 ? (1 / nodes.length) : 0;
  return { x: sx * inv, y: sy * inv };
}

function applyBounceBoundary(body, n, damping = 0.82) {
  const minX = body.r;
  const maxX = n - body.r;
  const minY = body.r;
  const maxY = n - body.r;

  if (body.x < minX) {
    body.x = minX;
    if (body.vx < 0) body.vx = -body.vx * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  } else if (body.x > maxX) {
    body.x = maxX;
    if (body.vx > 0) body.vx = -body.vx * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  }

  if (body.y < minY) {
    body.y = minY;
    if (body.vy < 0) body.vy = -body.vy * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  } else if (body.y > maxY) {
    body.y = maxY;
    if (body.vy > 0) body.vy = -body.vy * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  }
}

function resolveCircleCollision(a, b, restitution = 0.35) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const d2 = dx * dx + dy * dy;
  const minDist = (a.r || 1) + (b.r || 1);
  if (d2 <= 1e-10) {
    const jitter = 0.01;
    a.x -= jitter; b.x += jitter;
    return true;
  }
  if (d2 >= minDist * minDist) return false;

  const d = Math.sqrt(d2);
  const nx = dx / d;
  const ny = dy / d;
  const penetration = minDist - d;

  const ma = Math.max(0.02, a.mass || 1);
  const mb = Math.max(0.02, b.mass || 1);
  const invA = 1 / ma;
  const invB = 1 / mb;
  const invSum = invA + invB;

  // Positional correction
  const corr = (penetration / Math.max(1e-6, invSum)) * 0.85;
  a.x -= nx * corr * invA;
  a.y -= ny * corr * invA;
  b.x += nx * corr * invB;
  b.y += ny * corr * invB;

  // Impulse resolution
  const rvx = b.vx - a.vx;
  const rvy = b.vy - a.vy;
  const vn = rvx * nx + rvy * ny;
  if (vn > 0) return true;
  const j = (-(1 + restitution) * vn) / Math.max(1e-6, invSum);
  const ix = j * nx;
  const iy = j * ny;
  a.vx -= ix * invA;
  a.vy -= iy * invA;
  b.vx += ix * invB;
  b.vy += iy * invB;
  return true;
}

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 < 1e-8) return { t: 0, x: ax, y: ay, abx, aby, ab2 };
  const apx = px - ax;
  const apy = py - ay;
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  return { t, x: ax + abx * t, y: ay + aby * t, abx, aby, ab2 };
}

function resolveRigidVsSoftEdgeCollision(rigid, a, b, restitution = 0.28) {
  const cp = closestPointOnSegment(rigid.x, rigid.y, a.x, a.y, b.x, b.y);
  let nx = rigid.x - cp.x;
  let ny = rigid.y - cp.y;
  let dist = Math.hypot(nx, ny);
  const minDist = Math.max(0.8, rigid.r || 1);
  if (dist >= minDist) return false;

  if (dist < 1e-6) {
    const invLen = 1 / Math.max(1e-6, Math.hypot(-cp.aby, cp.abx));
    nx = -cp.aby * invLen;
    ny = cp.abx * invLen;
    dist = 1e-6;
  } else {
    nx /= dist;
    ny /= dist;
  }

  const penetration = minDist - dist;
  rigid.x += nx * penetration * 0.92;
  rigid.y += ny * penetration * 0.92;

  const edgeVx = (a.vx + b.vx) * 0.5;
  const edgeVy = (a.vy + b.vy) * 0.5;
  const rvx = rigid.vx - edgeVx;
  const rvy = rigid.vy - edgeVy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + restitution) * vn;
    rigid.vx += nx * j;
    rigid.vy += ny * j;
  }
  return true;
}

function resolveSoftNodeVsSoftEdgeCollision(node, a, b, restitution = 0.12) {
  const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);
  let nx = node.x - cp.x;
  let ny = node.y - cp.y;
  let dist = Math.hypot(nx, ny);
  const minDist = Math.max(0.4, node.r || 1.0);
  if (dist >= minDist) return false;

  if (dist < 1e-6) {
    const invLen = 1 / Math.max(1e-6, Math.hypot(-cp.aby, cp.abx));
    nx = -cp.aby * invLen;
    ny = cp.abx * invLen;
    dist = 1e-6;
  } else {
    nx /= dist;
    ny /= dist;
  }

  const penetration = minDist - dist;
  node.x += nx * penetration * 0.92;
  node.y += ny * penetration * 0.92;
  a.x -= nx * penetration * 0.04;
  a.y -= ny * penetration * 0.04;
  b.x -= nx * penetration * 0.04;
  b.y -= ny * penetration * 0.04;

  const edgeVx = (a.vx + b.vx) * 0.5;
  const edgeVy = (a.vy + b.vy) * 0.5;
  const rvx = node.vx - edgeVx;
  const rvy = node.vy - edgeVy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + restitution) * vn;
    node.vx += nx * j;
    node.vy += ny * j;
  }
  return true;
}

function buildRigidCollisionPolyMeta(polys) {
  if (!Array.isArray(polys)) return [];
  return polys.map((poly) => {
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let sx = 0;
    let sy = 0;
    let count = 0;

    for (const p of poly || []) {
      const x = Number(p?.x);
      const y = Number(p?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      sx += x;
      sy += y;
      count += 1;
    }

    if (count <= 0) {
      return { minX: 0, maxX: 0, minY: 0, maxY: 0, cx: 0, cy: 0, radius: 0 };
    }

    const cx = sx / count;
    const cy = sy / count;
    let radius = 0;
    for (const p of poly || []) {
      const x = Number(p?.x);
      const y = Number(p?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const d = Math.hypot(x - cx, y - cy);
      if (d > radius) radius = d;
    }

    return { minX, maxX, minY, maxY, cx, cy, radius };
  });
}

function createRigidWorldPolyPhaseCache(rigidBodies, stats = null, reuseCache = null) {
  const bodyCount = Array.isArray(rigidBodies) ? rigidBodies.length : 0;
  const cache = new Array(bodyCount);

  let sharedEntries = null;
  if (reuseCache && typeof reuseCache === 'object') {
    if (!Array.isArray(reuseCache.entries) || reuseCache.entries.length !== bodyCount) {
      reuseCache.entries = new Array(bodyCount);
    }
    sharedEntries = reuseCache.entries;
  }

  const isEntryMatch = (entry, rb, x, y, theta, verticesLocal, verticesLocalLength) => (
    entry
    && entry.rb === rb
    && entry.x === x
    && entry.y === y
    && entry.theta === theta
    && entry.verticesLocal === verticesLocal
    && entry.verticesLocalLength === verticesLocalLength
  );

  return (index) => {
    if (!Array.isArray(rigidBodies) || index < 0 || index >= rigidBodies.length) {
      return { polys: [], meta: [] };
    }
    const rb = rigidBodies[index];
    if (!rb) return { polys: [], meta: [] };
    if (stats) stats.polyCacheLookups = (Number(stats.polyCacheLookups) || 0) + 1;

    const x = Number(rb.x) || 0;
    const y = Number(rb.y) || 0;
    const theta = Number(rb.theta) || 0;
    const verticesLocal = Array.isArray(rb.verticesLocal) ? rb.verticesLocal : null;
    const verticesLocalLength = verticesLocal ? verticesLocal.length : 0;

    const prev = cache[index];
    if (isEntryMatch(prev, rb, x, y, theta, verticesLocal, verticesLocalLength)) {
      if (stats) {
        stats.polyCacheHits = (Number(stats.polyCacheHits) || 0) + 1;
        stats.polyCacheLocalHits = (Number(stats.polyCacheLocalHits) || 0) + 1;
        stats.polyBodiesReused = (Number(stats.polyBodiesReused) || 0) + 1;
      }
      return prev;
    }

    if (sharedEntries) {
      const shared = sharedEntries[index];
      if (isEntryMatch(shared, rb, x, y, theta, verticesLocal, verticesLocalLength)) {
        cache[index] = shared;
        if (stats) {
          stats.polyCacheHits = (Number(stats.polyCacheHits) || 0) + 1;
          stats.polyCacheSharedHits = (Number(stats.polyCacheSharedHits) || 0) + 1;
          stats.polyBodiesReused = (Number(stats.polyBodiesReused) || 0) + 1;
        }
        return shared;
      }
    }

    if (stats) {
      stats.polyCacheMisses = (Number(stats.polyCacheMisses) || 0) + 1;
      stats.polyCacheRebuilds = (Number(stats.polyCacheRebuilds) || 0) + 1;
      stats.polyBodiesBuilt = (Number(stats.polyBodiesBuilt) || 0) + 1;
    }

    const polyBuildStartMs = performance.now();
    const polys = getRigidCollisionPolysWorld(rb);
    const meta = buildRigidCollisionPolyMeta(polys);
    if (stats) stats.polyBuildMs = (Number(stats.polyBuildMs) || 0) + (performance.now() - polyBuildStartMs);

    const next = {
      rb,
      x,
      y,
      theta,
      verticesLocal,
      verticesLocalLength,
      polys,
      meta,
    };
    cache[index] = next;
    if (sharedEntries) sharedEntries[index] = next;
    return next;
  };
}

function resolveRigidInsideProjection(rb, node, poly) {
  if (!rb || !node || !Array.isArray(poly) || poly.length < 3) return false;
  if (!pointInPolygonInclusive(node.x, node.y, poly)) return false;

  let cx = 0;
  let cy = 0;
  for (const p of poly) {
    cx += p.x;
    cy += p.y;
  }
  cx /= Math.max(1, poly.length);
  cy /= Math.max(1, poly.length);

  let best = null;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);

    const ex = b.x - a.x;
    const ey = b.y - a.y;
    const eLen = Math.max(1e-6, Math.hypot(ex, ey));
    let nx = -ey / eLen;
    let ny = ex / eLen;
    const mx = (a.x + b.x) * 0.5;
    const my = (a.y + b.y) * 0.5;
    const toCenterX = cx - mx;
    const toCenterY = cy - my;
    if (toCenterX * nx + toCenterY * ny > 0) {
      nx = -nx;
      ny = -ny;
    }

    const dx = node.x - cp.x;
    const dy = node.y - cp.y;
    const d2 = dx * dx + dy * dy;
    if (!best || d2 < best.d2) best = { cp, d2, nx, ny };
  }
  if (!best) return false;

  const pad = Math.max(0.25, Number(node.r) || 1) + RIGID_INSIDE_CORRECTION_SLOP;
  const tx = best.cp.x + best.nx * pad;
  const ty = best.cp.y + best.ny * pad;

  const corrX = tx - node.x;
  const corrY = ty - node.y;
  const corrLen = Math.hypot(corrX, corrY);
  if (!Number.isFinite(corrLen) || corrLen < 1e-8) return false;

  const mNode = Math.max(0.02, Number(node.mass) || 1);
  const mRigid = Math.max(0.05, Number(rb.mass) || 1);
  const invNode = 1 / mNode;
  const invRigid = 1 / mRigid;
  const invSum = invNode + invRigid;

  const nodeShare = invNode / Math.max(1e-8, invSum);
  const rigidShare = invRigid / Math.max(1e-8, invSum);

  node.x += corrX * nodeShare;
  node.y += corrY * nodeShare;
  rb.x -= corrX * rigidShare * 0.45;
  rb.y -= corrY * rigidShare * 0.45;

  const vn = node.vx * best.nx + node.vy * best.ny;
  if (vn < 0) {
    node.vx -= best.nx * vn;
    node.vy -= best.ny * vn;
  }

  return true;
}

function applyRigidInsideCorrectionPass(bodies, soft) {
  if (!bodies?.rigid?.length || !soft?.nodes?.length) return 0;
  let corrected = 0;

  for (let iter = 0; iter < RIGID_INSIDE_CORRECTION_ITERS; iter++) {
    for (let rbi = 0; rbi < bodies.rigid.length; rbi++) {
      const rb = bodies.rigid[rbi];
      if (!rb || rb.insideCorrectionEnabled === false) continue;
      const polys = getRigidCollisionPolysWorld(rb);
      if (!Array.isArray(polys) || polys.length === 0) continue;

      for (let ni = 0; ni < soft.nodes.length; ni++) {
        const node = soft.nodes[ni];
        if (!node) continue;
        for (const poly of polys) {
          if (!Array.isArray(poly) || poly.length < 3) continue;
          if (!pointInPolygonInclusive(node.x, node.y, poly)) continue;
          if (resolveRigidInsideProjection(rb, node, poly)) corrected += 1;
          break;
        }
      }
    }
  }

  return corrected;
}

function applyMembraneInsideCorrectionPass(sim, soft, loops) {
  if (!soft?.nodes?.length || !Array.isArray(loops) || loops.length === 0) return 0;
  const clusterMap = sim?.softMembraneClusterMap;
  if (!(clusterMap instanceof Map) || clusterMap.size === 0) return 0;

  const candidates = [];
  for (const loop of loops) {
    const cid = Number(loop?.clusterId);
    if (!Number.isInteger(cid)) continue;
    const membrane = clusterMap.get(cid);
    if (!membrane) continue;
    if (Number(membrane?.insideCorrectionEnabled) <= 0) continue;

    const ids = Array.isArray(loop?.indices) ? loop.indices : [];
    if (ids.length < 3) continue;

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let area = 0;
    const poly = [];

    for (let i = 0; i < ids.length; i++) {
      const node = soft.nodes[ids[i]];
      if (!node) continue;
      poly.push({ x: node.x, y: node.y });
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x);
      maxY = Math.max(maxY, node.y);

      const next = soft.nodes[ids[(i + 1) % ids.length]];
      if (next) area += node.x * next.y - next.x * node.y;
    }

    if (poly.length < 3 || !Number.isFinite(area) || Math.abs(area) < 1e-8) continue;
    candidates.push({ cid, ids, poly, minX, minY, maxX, maxY, areaSign: area >= 0 ? 1 : -1 });
  }

  if (candidates.length === 0) return 0;

  let corrected = 0;
  for (let iter = 0; iter < RIGID_INSIDE_CORRECTION_ITERS; iter++) {
    for (let ni = 0; ni < soft.nodes.length; ni++) {
      const node = soft.nodes[ni];
      if (!node) continue;

      for (const c of candidates) {
        if (Number(node.clusterId) === c.cid) continue;
        const pad = Math.max(0.2, Number(node.r) || 1) + RIGID_INSIDE_CORRECTION_SLOP;
        if (node.x < c.minX - pad || node.x > c.maxX + pad || node.y < c.minY - pad || node.y > c.maxY + pad) continue;
        if (!pointInPolygonInclusive(node.x, node.y, c.poly)) continue;

        let best = null;
        for (let i = 0; i < c.ids.length; i++) {
          const ai = c.ids[i];
          const bi = c.ids[(i + 1) % c.ids.length];
          const a = soft.nodes[ai];
          const b = soft.nodes[bi];
          if (!a || !b) continue;

          const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);
          const dx = node.x - cp.x;
          const dy = node.y - cp.y;
          const d2 = dx * dx + dy * dy;
          if (!best || d2 < best.d2) {
            const ex = b.x - a.x;
            const ey = b.y - a.y;
            const len = Math.max(1e-6, Math.hypot(ex, ey));
            let nx = c.areaSign >= 0 ? (ey / len) : (-ey / len);
            let ny = c.areaSign >= 0 ? (-ex / len) : (ex / len);
            best = { d2, cp, nx, ny, ai, bi };
          }
        }
        if (!best) continue;

        const tx = best.cp.x + best.nx * pad;
        const ty = best.cp.y + best.ny * pad;
        const corrX = tx - node.x;
        const corrY = ty - node.y;
        const corrLen = Math.hypot(corrX, corrY);
        if (!Number.isFinite(corrLen) || corrLen < 1e-8) continue;

        node.x += corrX * 0.95;
        node.y += corrY * 0.95;

        const a = soft.nodes[best.ai];
        const b = soft.nodes[best.bi];
        if (a && b) {
          a.x -= corrX * 0.02;
          a.y -= corrY * 0.02;
          b.x -= corrX * 0.02;
          b.y -= corrY * 0.02;
        }

        const vn = node.vx * best.nx + node.vy * best.ny;
        if (vn < 0) {
          node.vx -= best.nx * vn;
          node.vy -= best.ny * vn;
        }

        corrected += 1;
        break;
      }
    }
  }

  return corrected;
}

function stampSegmentObstacleMask(mask, n, ax, ay, bx, by, thickness = 1.8) {
  const ex = bx - ax;
  const ey = by - ay;
  const segLenSq = ex * ex + ey * ey;
  if (!Number.isFinite(segLenSq) || segLenSq < 1e-9) return;

  const minX = Math.max(0, Math.floor(Math.min(ax, bx) - thickness - 1));
  const maxX = Math.min(n - 1, Math.ceil(Math.max(ax, bx) + thickness + 1));
  const minY = Math.max(0, Math.floor(Math.min(ay, by) - thickness - 1));
  const maxY = Math.min(n - 1, Math.ceil(Math.max(ay, by) + thickness + 1));

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      const apx = px - ax;
      const apy = py - ay;
      const t = Math.max(0, Math.min(1, (apx * ex + apy * ey) / Math.max(1e-9, segLenSq)));
      const cx = ax + ex * t;
      const cy = ay + ey * t;
      const dx = px - cx;
      const dy = py - cy;
      if (dx * dx + dy * dy > thickness * thickness) continue;
      mask[y * n + x] = 1;
    }
  }
}

function stampSegmentSweepTransport(sweep, n, ax, ay, bx, by, thickness, dirX, dirY) {
  if (!sweep || !Array.isArray(sweep.touched)) return;
  const ex = bx - ax;
  const ey = by - ay;
  const segLenSq = ex * ex + ey * ey;
  if (!Number.isFinite(segLenSq) || segLenSq < 1e-9) return;

  const minX = Math.max(0, Math.floor(Math.min(ax, bx) - thickness - 1));
  const maxX = Math.min(n - 1, Math.ceil(Math.max(ax, bx) + thickness + 1));
  const minY = Math.max(0, Math.floor(Math.min(ay, by) - thickness - 1));
  const maxY = Math.min(n - 1, Math.ceil(Math.max(ay, by) + thickness + 1));

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      const apx = px - ax;
      const apy = py - ay;
      const t = Math.max(0, Math.min(1, (apx * ex + apy * ey) / Math.max(1e-9, segLenSq)));
      const cx = ax + ex * t;
      const cy = ay + ey * t;
      const dx = px - cx;
      const dy = py - cy;
      if (dx * dx + dy * dy > thickness * thickness) continue;
      const i = y * n + x;
      if (sweep.mask[i] <= 0) sweep.touched.push(i);
      sweep.mask[i] = 1;
      sweep.pushX[i] += dirX;
      sweep.pushY[i] += dirY;
      sweep.weight[i] += 1;
    }
  }
}

function stampSweptSegmentObstacleMask(mask, n, prevA, prevB, currA, currB, thickness = 1.8, sweepTransport = null) {
  if (!prevA || !prevB || !currA || !currB) return;
  const ax0 = Number(prevA.x); const ay0 = Number(prevA.y);
  const bx0 = Number(prevB.x); const by0 = Number(prevB.y);
  const ax1 = Number(currA.x); const ay1 = Number(currA.y);
  const bx1 = Number(currB.x); const by1 = Number(currB.y);

  if (![ax0, ay0, bx0, by0, ax1, ay1, bx1, by1].every(Number.isFinite)) return;

  const moveA = Math.hypot(ax1 - ax0, ay1 - ay0);
  const moveB = Math.hypot(bx1 - bx0, by1 - by0);
  const maxMove = Math.max(moveA, moveB);
  const stepSpan = Math.max(0.45, Number(thickness) * 0.65);
  const steps = Math.max(1, Math.ceil(maxMove / stepSpan));

  let prevMidX = (ax0 + bx0) * 0.5;
  let prevMidY = (ay0 + by0) * 0.5;
  for (let si = 0; si <= steps; si++) {
    const t = si / steps;
    const ax = ax0 + (ax1 - ax0) * t;
    const ay = ay0 + (ay1 - ay0) * t;
    const bx = bx0 + (bx1 - bx0) * t;
    const by = by0 + (by1 - by0) * t;
    stampSegmentObstacleMask(mask, n, ax, ay, bx, by, thickness);

    if (sweepTransport) {
      const mx = (ax + bx) * 0.5;
      const my = (ay + by) * 0.5;
      const motionX = mx - prevMidX;
      const motionY = my - prevMidY;
      const ex = bx - ax;
      const ey = by - ay;
      const len = Math.hypot(ex, ey);
      if (len > 1e-6) {
        let nx = -ey / len;
        let ny = ex / len;
        if (motionX * nx + motionY * ny < 0) {
          nx = -nx;
          ny = -ny;
        }
        stampSegmentSweepTransport(sweepTransport, n, ax, ay, bx, by, thickness, nx, ny);
      }
      prevMidX = mx;
      prevMidY = my;
    }
  }
}

function resolveRigidEdgeScalar(spec, edgeIndex, fallback) {
  if (!Array.isArray(spec)) return fallback;
  const edgeValue = spec[edgeIndex];
  return edgeValue === undefined ? fallback : edgeValue;
}

function resolveRigidEdgeRGBTuple(spec, edgeIndex, fallback) {
  if (!Array.isArray(spec)) return fallback;
  const edgeValue = spec[edgeIndex];
  if (!Array.isArray(edgeValue) || edgeValue.length < 3) return fallback;
  return edgeValue;
}

function packDyeMaskModes(rMode, gMode, bMode) {
  return (Number(rMode) || 0) + 3 * (Number(gMode) || 0) + 9 * (Number(bMode) || 0);
}

function unpackDyeMaskModes(packed) {
  const p = Math.max(0, Math.min(26, Math.round(Number(packed) || 0)));
  const r = p % 3;
  const g = Math.floor(p / 3) % 3;
  const b = Math.floor(p / 9) % 3;
  return [r, g, b];
}

function stampSegmentDyeMask(mask, n, ax, ay, bx, by, thickness = 1.8, modeRGB = [0, 0, 0]) {
  const ex = bx - ax;
  const ey = by - ay;
  const segLenSq = ex * ex + ey * ey;
  if (!Number.isFinite(segLenSq) || segLenSq < 1e-9) return;

  const mR = (modeRGB?.[0] === EDGE_DYE_MODE.ABSORB) ? EDGE_DYE_MODE.ABSORB
    : ((modeRGB?.[0] === EDGE_DYE_MODE.DEFLECT) ? EDGE_DYE_MODE.DEFLECT : EDGE_DYE_MODE.PASS);
  const mG = (modeRGB?.[1] === EDGE_DYE_MODE.ABSORB) ? EDGE_DYE_MODE.ABSORB
    : ((modeRGB?.[1] === EDGE_DYE_MODE.DEFLECT) ? EDGE_DYE_MODE.DEFLECT : EDGE_DYE_MODE.PASS);
  const mB = (modeRGB?.[2] === EDGE_DYE_MODE.ABSORB) ? EDGE_DYE_MODE.ABSORB
    : ((modeRGB?.[2] === EDGE_DYE_MODE.DEFLECT) ? EDGE_DYE_MODE.DEFLECT : EDGE_DYE_MODE.PASS);
  if (mR === EDGE_DYE_MODE.PASS && mG === EDGE_DYE_MODE.PASS && mB === EDGE_DYE_MODE.PASS) return;

  const minX = Math.max(0, Math.floor(Math.min(ax, bx) - thickness - 1));
  const maxX = Math.min(n - 1, Math.ceil(Math.max(ax, bx) + thickness + 1));
  const minY = Math.max(0, Math.floor(Math.min(ay, by) - thickness - 1));
  const maxY = Math.min(n - 1, Math.ceil(Math.max(ay, by) + thickness + 1));

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      const apx = px - ax;
      const apy = py - ay;
      const t = Math.max(0, Math.min(1, (apx * ex + apy * ey) / Math.max(1e-9, segLenSq)));
      const cx = ax + ex * t;
      const cy = ay + ey * t;
      const dx = px - cx;
      const dy = py - cy;
      if (dx * dx + dy * dy > thickness * thickness) continue;

      const i = y * n + x;
      const [cR, cG, cB] = unpackDyeMaskModes(mask[i]);
      const nR = Math.max(cR, mR);
      const nG = Math.max(cG, mG);
      const nB = Math.max(cB, mB);
      mask[i] = packDyeMaskModes(nR, nG, nB);
    }
  }
}

function stampPerChannelDyeMask(sim) {
  const n = Number(sim?.controls?.n) || 0;
  const mask = sim?.dyeModeMaskCpu;
  if (!(mask instanceof Uint32Array) || mask.length !== n * n || n <= 0) {
    return { rigidEdges: 0, softEdges: 0, nonPassCells: 0 };
  }

  mask.fill(0);
  const worldScale = getWorldScale(sim?.controls);
  const softThickness = Math.max(1.2, 1.1 * (n / 256));
  const rigidThickness = Math.max(1.4, 1.2 * (n / 256));
  const softThicknessGrid = softThickness / worldScale;
  const rigidThicknessGrid = rigidThickness / worldScale;

  let rigidEdges = 0;
  for (const rb of sim?.bodies?.rigid || []) {
    const verts = rigidVerticesWorld(rb);
    const sides = verts.length;
    for (let ei = 0; ei < sides; ei++) {
      const rawModeRGB = normalizeEdgeDyeModeRGB(resolveRigidEdgeRGBTuple(rb?.edgeDyeMode, ei, [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]));
      const permeabilityRGB = normalizePermeabilityRGB(resolveRigidEdgeRGBTuple(rb?.edgePermeabilityRGB, ei, [0, 0, 0]));
      const modeRGB = [0, 0, 0].map((_, ci) => {
        if (permeabilityRGB[ci] > 0) return EDGE_DYE_MODE.PASS;
        return rawModeRGB[ci] === EDGE_DYE_MODE.ABSORB ? EDGE_DYE_MODE.ABSORB : EDGE_DYE_MODE.DEFLECT;
      });

      const a = verts[ei];
      const b = verts[(ei + 1) % sides];
      if (!a || !b) continue;
      const ax = Number(a.x), ay = Number(a.y), bx = Number(b.x), by = Number(b.y);
      if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) continue;
      const axGrid = ax / worldScale;
      const ayGrid = ay / worldScale;
      const bxGrid = bx / worldScale;
      const byGrid = by / worldScale;

      stampSegmentDyeMask(mask, n, axGrid, ayGrid, bxGrid, byGrid, rigidThicknessGrid, modeRGB);
      rigidEdges += 1;
    }
  }

  let softEdges = 0;
  const s = sim?.bodies?.soft;
  for (const [ai, bi, _rest, _edgeBodyMode, edgeDyeMode] of (s?.springs || [])) {
    const modeRGB = normalizeEdgeDyeModeRGB(edgeDyeMode);
    const a = s?.nodes?.[ai];
    const b = s?.nodes?.[bi];
    if (!a || !b) continue;
    const ax = Number(a.x), ay = Number(a.y), bx = Number(b.x), by = Number(b.y);
    if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) continue;
    const axGrid = ax / worldScale;
    const ayGrid = ay / worldScale;
    const bxGrid = bx / worldScale;
    const byGrid = by / worldScale;
    stampSegmentDyeMask(mask, n, axGrid, ayGrid, bxGrid, byGrid, softThicknessGrid, modeRGB);
    softEdges += 1;
  }

  let nonPassCells = 0;
  for (let i = 0; i < mask.length; i++) {
    if ((mask[i] | 0) !== 0) nonPassCells += 1;
  }

  return { rigidEdges, softEdges, nonPassCells };
}

function stampBodyObstacleMask(sim) {
  const n = Number(sim?.controls?.n) || 0;
  const mask = sim?.obstacleMaskCpu;
  if (!(mask instanceof Float32Array) || mask.length !== n * n || n <= 0) {
    return { rigidBlockedEdges: 0, softBlockedEdges: 0, blockedEdgeCount: 0, blockedCells: 0, sweptCells: 0 };
  }

  const cells = n * n;
  if (!(sim.sweptEdgeMaskCpu instanceof Float32Array) || sim.sweptEdgeMaskCpu.length !== cells) {
    sim.sweptEdgeMaskCpu = new Float32Array(cells);
    sim.sweptEdgePushXCpu = new Float32Array(cells);
    sim.sweptEdgePushYCpu = new Float32Array(cells);
    sim.sweptEdgePushWeightCpu = new Float32Array(cells);
    sim.sweptEdgeTouched = [];
  }

  const sweepTouched = Array.isArray(sim.sweptEdgeTouched) ? sim.sweptEdgeTouched : (sim.sweptEdgeTouched = []);
  for (const idx of sweepTouched) {
    sim.sweptEdgeMaskCpu[idx] = 0;
    sim.sweptEdgePushXCpu[idx] = 0;
    sim.sweptEdgePushYCpu[idx] = 0;
    sim.sweptEdgePushWeightCpu[idx] = 0;
  }
  sweepTouched.length = 0;

  const sweepTransport = {
    mask: sim.sweptEdgeMaskCpu,
    pushX: sim.sweptEdgePushXCpu,
    pushY: sim.sweptEdgePushYCpu,
    weight: sim.sweptEdgePushWeightCpu,
    touched: sweepTouched,
  };

  mask.fill(0);
  const worldScale = getWorldScale(sim?.controls);
  const rigidThickness = Math.max(1.4, 1.2 * (n / 256));
  const softThickness = Math.max(1.5, 1.35 + 1.15 * (n / 256));
  const rigidThicknessGrid = rigidThickness / worldScale;
  const softThicknessGrid = softThickness / worldScale;

  let rigidBlockedEdges = 0;
  for (const rb of sim?.bodies?.rigid || []) {
    const verts = rigidVerticesWorld(rb);
    const sides = verts.length;
    const prevVerts = Array.isArray(rb?._prevObstacleVerts) && rb._prevObstacleVerts.length === sides
      ? rb._prevObstacleVerts
      : null;

    for (let ei = 0; ei < sides; ei++) {
      const edgeVelocityMode = Number(resolveRigidEdgeScalar(
        rb?.edgeVelocityMode,
        ei,
        resolveRigidEdgeScalar(rb?.edgeBodyMode, ei, EDGE_BODY_MODE.BLOCK),
      )) === EDGE_BODY_MODE.PASS
        ? EDGE_BODY_MODE.PASS
        : EDGE_BODY_MODE.BLOCK;
      if (edgeVelocityMode === EDGE_BODY_MODE.PASS) continue;

      const a = verts[ei];
      const b = verts[(ei + 1) % sides];
      if (!a || !b) continue;
      const ax = Number(a.x);
      const ay = Number(a.y);
      const bx = Number(b.x);
      const by = Number(b.y);
      if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) continue;
      const axGrid = ax / worldScale;
      const ayGrid = ay / worldScale;
      const bxGrid = bx / worldScale;
      const byGrid = by / worldScale;

      stampSegmentObstacleMask(mask, n, axGrid, ayGrid, bxGrid, byGrid, rigidThicknessGrid);

      // Swept stamping bridges previous->current rigid edge pose so fast-moving
      // boundaries don't leave single-frame gaps for fluid transport.
      if (prevVerts) {
        const pa = prevVerts[ei];
        const pb = prevVerts[(ei + 1) % sides];
        const paGrid = { x: Number(pa?.x) / worldScale, y: Number(pa?.y) / worldScale };
        const pbGrid = { x: Number(pb?.x) / worldScale, y: Number(pb?.y) / worldScale };
        const aGrid = { x: axGrid, y: ayGrid };
        const bGrid = { x: bxGrid, y: byGrid };
        stampSweptSegmentObstacleMask(mask, n, paGrid, pbGrid, aGrid, bGrid, rigidThicknessGrid, sweepTransport);
      }

      rigidBlockedEdges += 1;
    }

    rb._prevObstacleVerts = verts.map((p) => ({ x: Number(p?.x) || 0, y: Number(p?.y) || 0 }));
  }

  let softBlockedEdges = 0;
  const soft = sim?.bodies?.soft;
  const softNodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const prevSoftNodes = Array.isArray(sim?._prevObstacleSoftNodes) && sim._prevObstacleSoftNodes.length === softNodes.length
    ? sim._prevObstacleSoftNodes
    : null;

  for (const sp of (soft?.springs || [])) {
    const ai = Number(sp?.[0]);
    const bi = Number(sp?.[1]);
    if (!Number.isInteger(ai) || !Number.isInteger(bi)) continue;

    const edgeVelocityMode = Number(sp?.[5]) === EDGE_BODY_MODE.PASS
      ? EDGE_BODY_MODE.PASS
      : (Number(sp?.[3]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK);
    if (edgeVelocityMode === EDGE_BODY_MODE.PASS) continue;

    const a = softNodes[ai];
    const b = softNodes[bi];
    if (!a || !b) continue;

    const ax = Number(a.x);
    const ay = Number(a.y);
    const bx = Number(b.x);
    const by = Number(b.y);
    if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) continue;
    const axGrid = ax / worldScale;
    const ayGrid = ay / worldScale;
    const bxGrid = bx / worldScale;
    const byGrid = by / worldScale;

    stampSegmentObstacleMask(mask, n, axGrid, ayGrid, bxGrid, byGrid, softThicknessGrid);

    if (prevSoftNodes) {
      const pa = prevSoftNodes[ai];
      const pb = prevSoftNodes[bi];
      if (pa && pb) {
        const paGrid = { x: Number(pa?.x) / worldScale, y: Number(pa?.y) / worldScale };
        const pbGrid = { x: Number(pb?.x) / worldScale, y: Number(pb?.y) / worldScale };
        const aGrid = { x: axGrid, y: ayGrid };
        const bGrid = { x: bxGrid, y: byGrid };
        stampSweptSegmentObstacleMask(mask, n, paGrid, pbGrid, aGrid, bGrid, softThicknessGrid, sweepTransport);
      }
    }

    softBlockedEdges += 1;
  }

  sim._prevObstacleSoftNodes = softNodes.map((node) => ({
    x: Number(node?.x) || 0,
    y: Number(node?.y) || 0,
  }));

  let blockedCells = 0;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) blockedCells += 1;
  }

  return {
    rigidBlockedEdges,
    softBlockedEdges,
    blockedEdgeCount: rigidBlockedEdges + softBlockedEdges,
    blockedCells,
    sweptCells: sweepTouched.length,
  };
}

function enforceFluidEdgeBoundariesCpu(vxField, vyField, n) {
  const last = n - 1;
  for (let x = 0; x < n; x++) {
    const top = x;
    const bottom = last * n + x;
    vyField[top] = 0;
    vyField[bottom] = 0;
    vxField[top] *= 0.7;
    vxField[bottom] *= 0.7;
  }
  for (let y = 0; y < n; y++) {
    const left = y * n;
    const right = y * n + last;
    vxField[left] = 0;
    vxField[right] = 0;
    vyField[left] *= 0.7;
    vyField[right] *= 0.7;
  }
}

function redistributeSweptEdgeDyeTransport(sim, r, g, b) {
  const n = Number(sim?.controls?.n) || 0;
  const mask = sim?.sweptEdgeMaskCpu;
  const pushX = sim?.sweptEdgePushXCpu;
  const pushY = sim?.sweptEdgePushYCpu;
  const weight = sim?.sweptEdgePushWeightCpu;
  const touched = Array.isArray(sim?.sweptEdgeTouched) ? sim.sweptEdgeTouched : [];
  const obstacle = sim?.obstacleMaskCpu;
  const dyeModeMask = sim?.dyeModeMaskCpu;

  if (
    !(mask instanceof Float32Array) ||
    !(pushX instanceof Float32Array) ||
    !(pushY instanceof Float32Array) ||
    !(weight instanceof Float32Array) ||
    !(obstacle instanceof Float32Array) ||
    !(r instanceof Float32Array) ||
    !(g instanceof Float32Array) ||
    !(b instanceof Float32Array) ||
    n <= 0 ||
    touched.length === 0
  ) {
    return { touchedCells: 0, movedMass: 0, deletedMass: 0, eatenMass: 0 };
  }

  let movedMass = 0;
  let deletedMass = 0;
  let eatenMass = 0;
  let touchedCells = 0;

  for (const idx of touched) {
    if (mask[idx] <= 0.5) continue;
    const rv = Number(r[idx]) || 0;
    const gv = Number(g[idx]) || 0;
    const bv = Number(b[idx]) || 0;
    const dyeSum = rv + gv + bv;
    if (dyeSum <= 1e-5) continue;

    touchedCells += 1;
    const packedMask = (dyeModeMask instanceof Uint32Array && idx >= 0 && idx < dyeModeMask.length)
      ? (dyeModeMask[idx] | 0)
      : 0;
    const [modeR, modeG, modeB] = unpackDyeMaskModes(packedMask);

    const rvAfterEat = modeR === EDGE_DYE_MODE.ABSORB ? 0 : rv;
    const gvAfterEat = modeG === EDGE_DYE_MODE.ABSORB ? 0 : gv;
    const bvAfterEat = modeB === EDGE_DYE_MODE.ABSORB ? 0 : bv;
    eatenMass += (rv - rvAfterEat) + (gv - gvAfterEat) + (bv - bvAfterEat);

    const dyeRemaining = rvAfterEat + gvAfterEat + bvAfterEat;
    const w = Math.max(1e-6, Number(weight[idx]) || 0);
    const dirX = (Number(pushX[idx]) || 0) / w;
    const dirY = (Number(pushY[idx]) || 0) / w;

    const x = idx % n;
    const y = Math.floor(idx / n);
    const stepX = dirX > 0.1 ? 1 : (dirX < -0.1 ? -1 : 0);
    const stepY = dirY > 0.1 ? 1 : (dirY < -0.1 ? -1 : 0);

    let deposited = false;
    if (dyeRemaining > 1e-5 && (stepX !== 0 || stepY !== 0)) {
      for (let k = 1; k <= 2; k++) {
        const tx = x + stepX * k;
        const ty = y + stepY * k;
        if (tx < 0 || tx >= n || ty < 0 || ty >= n) continue;
        const ti = ty * n + tx;
        if ((Number(obstacle[ti]) || 0) > 0.5) continue;
        r[ti] = (Number(r[ti]) || 0) + rvAfterEat;
        g[ti] = (Number(g[ti]) || 0) + gvAfterEat;
        b[ti] = (Number(b[ti]) || 0) + bvAfterEat;
        movedMass += dyeRemaining;
        deposited = true;
        break;
      }
    }

    if (!deposited && dyeRemaining > 1e-5) {
      deletedMass += dyeRemaining;
    }

    r[idx] = 0;
    g[idx] = 0;
    b[idx] = 0;
  }

  return { touchedCells, movedMass, deletedMass, eatenMass };
}

function applySoftSpringsXPBDVelocity(
  s,
  dtPos,
  stiffnessScale,
  lambdaCache,
  {
    skipClusterSet = null,
    softXpbdBaseCompliance = SOFT_XPBD_BASE_COMPLIANCE,
    softXpbdIters = SOFT_XPBD_ITERS,
  } = {},
) {
  if (!s?.nodes?.length || !s?.springs?.length) return;
  const alpha = (softXpbdBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < softXpbdIters; iter++) {
    for (let si = 0; si < s.springs.length; si++) {
      const [i, j, rest] = s.springs[si];
      const a = s.nodes[i];
      const b = s.nodes[j];
      if (!a || !b) continue;
      if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
      const C = clamp(d - rest, -strainCap, strainCap);

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}

function signedAreaPredicted(nodes, indices, dtPos) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function ensureSoftAreaRestState(sim, s, loops, dtPos) {
  sim.softAreaRest = sim.softAreaRest || new Map();
  sim.softAreaLambda = sim.softAreaLambda || new Map();

  const live = new Set(loops.map((l) => l.clusterId));
  for (const key of sim.softAreaRest.keys()) if (!live.has(key)) sim.softAreaRest.delete(key);
  for (const key of sim.softAreaLambda.keys()) if (!live.has(key)) sim.softAreaLambda.delete(key);

  for (const loop of loops) {
    if (!sim.softAreaRest.has(loop.clusterId)) {
      const a0 = signedAreaPredicted(s.nodes, loop.indices, dtPos);
      sim.softAreaRest.set(loop.clusterId, Math.abs(a0) > 1e-4 ? a0 : 1e-4);
    }
    const prev = Number(sim.softAreaLambda.get(loop.clusterId));
    sim.softAreaLambda.set(loop.clusterId, Number.isFinite(prev) ? Math.max(-20, Math.min(20, prev)) : 0);
  }
}

function applySoftAreaXPBDVelocity(
  sim,
  s,
  loops,
  dtPos,
  stiffnessScale,
  {
    softAreaBaseCompliance = SOFT_AREA_BASE_COMPLIANCE,
    softAreaXpbdIters = SOFT_AREA_XPBD_ITERS,
  } = {},
) {
  if (!loops.length) return;
  const alpha = (softAreaBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < softAreaXpbdIters; iter++) {
    for (const loop of loops) {
      const ids = loop.indices;
      const m = ids.length;
      if (m < 3) continue;
      const restArea = sim.softAreaRest.get(loop.clusterId);
      if (!Number.isFinite(restArea)) continue;

      const area = signedAreaPredicted(s.nodes, ids, dtPos);
      const C = area - restArea;

      const gradX = new Array(m);
      const gradY = new Array(m);
      let sumWGrad2 = 0;

      for (let k = 0; k < m; k++) {
        const prev = s.nodes[ids[(k - 1 + m) % m]];
        const next = s.nodes[ids[(k + 1) % m]];
        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nx = next.x + next.vx * dtPos;
        const ny = next.y + next.vy * dtPos;
        const gx = 0.5 * (ny - py);
        const gy = 0.5 * (px - nx);
        gradX[k] = gx;
        gradY[k] = gy;

        const node = s.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sumWGrad2 += w * (gx * gx + gy * gy);
      }

      if (sumWGrad2 <= 1e-10) continue;

      const lambdaPrev = Number(sim.softAreaLambda.get(loop.clusterId)) || 0;
      let dl = (-C - alpha * lambdaPrev) / (sumWGrad2 + alpha);
      if (!Number.isFinite(dl)) continue;
      dl = Math.max(-2.0, Math.min(2.0, dl));
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      sim.softAreaLambda.set(loop.clusterId, lambdaNext);

      for (let k = 0; k < m; k++) {
        const node = s.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gradX[k] * dl) / dtPos;
        node.vy += (w * gradY[k] * dl) / dtPos;
      }
    }
  }
}

function signedAreaCurrent(nodes, indices) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    if (!a || !b) continue;
    s += a.x * b.y - b.x * a.y;
  }
  return 0.5 * s;
}

function ensureSoftMembraneClusterSet(sim) {
  const set = new Set();
  const map = new Map();
  for (const c of (sim?.bodies?.softMembraneClusters || [])) {
    const cid = Number(c?.clusterId);
    if (!Number.isInteger(cid)) continue;
    set.add(cid);
    map.set(cid, c);
  }
  sim.softMembraneClusterSet = set;
  sim.softMembraneClusterMap = map;
  return set;
}

function ensureSoftMembraneLoopState(sim, s, loops) {
  sim.softMembraneLoopState = sim.softMembraneLoopState || new Map();
  const live = new Set();
  const membraneSet = ensureSoftMembraneClusterSet(sim);

  for (const loop of loops || []) {
    const cid = loop.clusterId ?? 0;
    if (!membraneSet.has(cid)) continue;
    const ids = loop.indices || [];
    if (ids.length < 3) continue;
    live.add(cid);

    const key = ids.join(',');
    let st = sim.softMembraneLoopState.get(cid);
    const needsReset = !st || st.key !== key || st.edgeRest.length !== ids.length;
    if (needsReset) {
      st = {
        key,
        edgeRest: new Float32Array(ids.length),
        edgeLambda: new Float32Array(ids.length),
        bendRest: new Float32Array(ids.length),
        bendLambda: new Float32Array(ids.length),
        shapeRef: new Float32Array(ids.length * 2),
        shapeRefRms: 1,
      };

      let cx = 0;
      let cy = 0;
      let count = 0;
      for (let i = 0; i < ids.length; i++) {
        const node = s.nodes[ids[i]];
        if (!node) continue;
        cx += node.x || 0;
        cy += node.y || 0;
        count += 1;
      }
      if (count > 0) {
        cx /= count;
        cy /= count;
      }

      let refR2 = 0;
      for (let i = 0; i < ids.length; i++) {
        const a = s.nodes[ids[i]];
        const b = s.nodes[ids[(i + 1) % ids.length]];
        if (a && b) {
          st.edgeRest[i] = Math.max(1e-4, Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0)));
        }

        const p = s.nodes[ids[(i - 1 + ids.length) % ids.length]];
        const n = s.nodes[ids[(i + 1) % ids.length]];
        if (p && n) {
          st.bendRest[i] = Math.max(1e-4, Math.hypot((n.x || 0) - (p.x || 0), (n.y || 0) - (p.y || 0)));
        }

        const node = s.nodes[ids[i]];
        const rx = (node?.x || 0) - cx;
        const ry = (node?.y || 0) - cy;
        st.shapeRef[i * 2] = rx;
        st.shapeRef[i * 2 + 1] = ry;
        refR2 += rx * rx + ry * ry;
      }
      st.shapeRefRms = Math.max(1e-3, Math.sqrt(refR2 / Math.max(1, ids.length)));
      sim.softMembraneLoopState.set(cid, st);
    }
  }

  for (const cid of [...sim.softMembraneLoopState.keys()]) {
    if (!live.has(cid)) sim.softMembraneLoopState.delete(cid);
  }
}

function applySoftMembraneBoundaryXPBDVelocity(
  sim,
  s,
  loops,
  dtPos,
  {
    membraneEdgeBaseCompliance = MEMBRANE_EDGE_BASE_COMPLIANCE,
    membraneBendBaseCompliance = MEMBRANE_BEND_BASE_COMPLIANCE,
    membraneEdgeXpbdIters = MEMBRANE_EDGE_XPBD_ITERS,
    membraneBendXpbdIters = MEMBRANE_BEND_XPBD_ITERS,
  } = {},
) {
  const membraneSet = ensureSoftMembraneClusterSet(sim);
  if (!membraneSet.size) return 0;
  ensureSoftMembraneLoopState(sim, s, loops);

  const edgeAlpha = membraneEdgeBaseCompliance / Math.max(1e-8, dtPos * dtPos);
  const bendAlpha = membraneBendBaseCompliance / Math.max(1e-8, dtPos * dtPos);
  let touched = 0;

  for (let iter = 0; iter < membraneEdgeXpbdIters; iter++) {
    for (const loop of loops || []) {
      const cid = loop.clusterId ?? 0;
      if (!membraneSet.has(cid)) continue;
      const ids = loop.indices || [];
      if (ids.length < 3) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st) continue;

      for (let i = 0; i < ids.length; i++) {
        const a = s.nodes[ids[i]];
        const b = s.nodes[ids[(i + 1) % ids.length]];
        if (!a || !b) continue;

        const ax = a.x + a.vx * dtPos;
        const ay = a.y + a.vy * dtPos;
        const bx = b.x + b.vx * dtPos;
        const by = b.y + b.vy * dtPos;
        const dx = bx - ax;
        const dy = by - ay;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const nx = dx / d;
        const ny = dy / d;

        const rest = Math.max(1e-4, Number(st.edgeRest[i]) || d);
        const C = clamp(d - rest, -Math.max(0.08, rest * 0.28), Math.max(0.08, rest * 0.28));
        const wA = 1 / Math.max(0.02, a.mass || 1);
        const wB = 1 / Math.max(0.02, b.mass || 1);
        const wSum = wA + wB;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(st.edgeLambda[i]) || 0;
        let dl = (-C - edgeAlpha * lambdaPrev) / (wSum + edgeAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = clamp(lambdaPrev + dl, -20, 20);
        dl = lambdaNext - lambdaPrev;
        st.edgeLambda[i] = lambdaNext;

        a.vx += (-wA * dl * nx) / dtPos;
        a.vy += (-wA * dl * ny) / dtPos;
        b.vx += (wB * dl * nx) / dtPos;
        b.vy += (wB * dl * ny) / dtPos;
        touched += 1;
      }
    }
  }

  for (let iter = 0; iter < membraneBendXpbdIters; iter++) {
    for (const loop of loops || []) {
      const cid = loop.clusterId ?? 0;
      if (!membraneSet.has(cid)) continue;
      const ids = loop.indices || [];
      if (ids.length < 4) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st) continue;

      for (let i = 0; i < ids.length; i++) {
        const prev = s.nodes[ids[(i - 1 + ids.length) % ids.length]];
        const next = s.nodes[ids[(i + 1) % ids.length]];
        if (!prev || !next) continue;

        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nxp = next.x + next.vx * dtPos;
        const nyp = next.y + next.vy * dtPos;
        const dx = nxp - px;
        const dy = nyp - py;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const ux = dx / d;
        const uy = dy / d;

        const rest = Math.max(1e-4, Number(st.bendRest[i]) || d);
        const C = clamp(d - rest, -Math.max(0.1, rest * 0.35), Math.max(0.1, rest * 0.35));
        const wP = 1 / Math.max(0.02, prev.mass || 1);
        const wN = 1 / Math.max(0.02, next.mass || 1);
        const wSum = wP + wN;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(st.bendLambda[i]) || 0;
        let dl = (-C - bendAlpha * lambdaPrev) / (wSum + bendAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = clamp(lambdaPrev + dl, -20, 20);
        dl = lambdaNext - lambdaPrev;
        st.bendLambda[i] = lambdaNext;

        prev.vx += (-wP * dl * ux) / dtPos;
        prev.vy += (-wP * dl * uy) / dtPos;
        next.vx += (wN * dl * ux) / dtPos;
        next.vy += (wN * dl * uy) / dtPos;
      }
    }
  }

  return touched;
}

function applySoftMembraneShapeMemoryVelocity(sim, s, loops, dtPos) {
  const clusterMap = sim?.softMembraneClusterMap;
  if (!(clusterMap instanceof Map) || clusterMap.size === 0) return 0;

  let touched = 0;
  for (let iter = 0; iter < MEMBRANE_SHAPE_MEMORY_ITERS; iter++) {
    for (const loop of loops || []) {
      const cid = Number(loop?.clusterId);
      if (!Number.isInteger(cid)) continue;
      const membrane = clusterMap.get(cid);
      if (!membrane) continue;

      const shapeMemoryGainRaw = Number(membrane?.shapeMemoryGain);
      const shapeMemoryGain = Number.isFinite(shapeMemoryGainRaw)
        ? clamp(shapeMemoryGainRaw, 0, 0.35)
        : MEMBRANE_SHAPE_MEMORY_GAIN;
      if (shapeMemoryGain <= 1e-6) continue;

      const ids = loop.indices || [];
      if (ids.length < 3) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st || !(st.shapeRef instanceof Float32Array) || st.shapeRef.length !== ids.length * 2) continue;

      let cx = 0;
      let cy = 0;
      let count = 0;
      for (const idx of ids) {
        const node = s.nodes[idx];
        if (!node) continue;
        cx += (node.x || 0) + (node.vx || 0) * dtPos;
        cy += (node.y || 0) + (node.vy || 0) * dtPos;
        count += 1;
      }
      if (count < 3) continue;
      cx /= count;
      cy /= count;

      let dot = 0;
      let cross = 0;
      for (let i = 0; i < ids.length; i++) {
        const node = s.nodes[ids[i]];
        if (!node) continue;
        const px = (node.x || 0) + (node.vx || 0) * dtPos - cx;
        const py = (node.y || 0) + (node.vy || 0) * dtPos - cy;
        const rx = st.shapeRef[i * 2] || 0;
        const ry = st.shapeRef[i * 2 + 1] || 0;
        dot += rx * px + ry * py;
        cross += rx * py - ry * px;
      }

      const theta = (Math.abs(dot) + Math.abs(cross)) > 1e-9 ? Math.atan2(cross, dot) : 0;
      const c = Math.cos(theta);
      const sn = Math.sin(theta);
      const maxShift = Math.max(0.02, (st.shapeRefRms || 1) * MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC);

      for (let i = 0; i < ids.length; i++) {
        const node = s.nodes[ids[i]];
        if (!node) continue;

        const px = (node.x || 0) + (node.vx || 0) * dtPos;
        const py = (node.y || 0) + (node.vy || 0) * dtPos;
        const rx = st.shapeRef[i * 2] || 0;
        const ry = st.shapeRef[i * 2 + 1] || 0;

        const tx = cx + rx * c - ry * sn;
        const ty = cy + rx * sn + ry * c;

        let ex = tx - px;
        let ey = ty - py;
        const eLen = Math.hypot(ex, ey);
        if (!Number.isFinite(eLen) || eLen <= 1e-7) continue;
        if (eLen > maxShift) {
          const k = maxShift / eLen;
          ex *= k;
          ey *= k;
        }

        const localWeight = clamp(Number.isFinite(Number(node.shapeMemoryWeight)) ? Number(node.shapeMemoryWeight) : 1, 0, 1);
        if (localWeight <= 1e-6) continue;
        const invMass = 1 / Math.max(0.02, Number(node.mass) || 1);
        const corrPos = shapeMemoryGain * localWeight * invMass;
        node.vx += (ex * corrPos) / dtPos;
        node.vy += (ey * corrPos) / dtPos;
      }

      touched += 1;
    }
  }

  return touched;
}

function applySoftMembraneCellPressure(sim, s, loops, dtPos) {
  const membranes = sim?.bodies?.softMembraneClusters;
  if (!Array.isArray(membranes) || membranes.length === 0) return 0;
  if (!(sim.softMembraneAreaBaseline instanceof Map)) sim.softMembraneAreaBaseline = new Map();

  const loopByCluster = new Map();
  for (const loop of loops || []) {
    loopByCluster.set(loop.clusterId ?? 0, loop);
  }

  let touched = 0;
  for (const membrane of membranes) {
    const cid = Number(membrane?.clusterId);
    if (!Number.isInteger(cid)) continue;
    const loop = loopByCluster.get(cid);
    if (!loop || !Array.isArray(loop.indices) || loop.indices.length < 3) continue;

    const areaNow = Math.abs(signedAreaCurrent(s.nodes, loop.indices));
    if (!Number.isFinite(areaNow) || areaNow < 1e-6) continue;

    const seededBase = Math.max(1e-4, Number(membrane?.restArea) || areaNow);
    if (!sim.softMembraneAreaBaseline.has(cid)) {
      sim.softMembraneAreaBaseline.set(cid, seededBase);
    }
    const areaBase = Math.max(1e-4, Number(sim.softMembraneAreaBaseline.get(cid)) || seededBase);
    const err = clamp((areaBase - areaNow) / areaBase, -0.65, 0.65);
    if (Math.abs(err) < 1e-4) continue;

    const pressureGain = Math.max(0.005, Number(membrane?.pressureGain) || MEMBRANE_CELL_BASE_PRESSURE_GAIN);
    const radialDamping = clamp(Number(membrane?.radialDamping) || MEMBRANE_CELL_BASE_RADIAL_DAMPING, 0, 0.2);
    const gain = pressureGain * (1 + Math.min(1.4, Math.abs(err) * 2.2));

    let cx = 0;
    let cy = 0;
    for (const ni of loop.indices) {
      const node = s.nodes[ni];
      if (!node) continue;
      cx += node.x;
      cy += node.y;
    }
    cx /= loop.indices.length;
    cy /= loop.indices.length;

    for (let k = 0; k < loop.indices.length; k++) {
      const iPrev = loop.indices[(k - 1 + loop.indices.length) % loop.indices.length];
      const iCurr = loop.indices[k];
      const iNext = loop.indices[(k + 1) % loop.indices.length];
      const prev = s.nodes[iPrev];
      const curr = s.nodes[iCurr];
      const next = s.nodes[iNext];
      if (!prev || !curr || !next) continue;

      // Outward normal for CCW loop via right-normal accumulation.
      let nx = (curr.y - prev.y) + (next.y - curr.y);
      let ny = -((curr.x - prev.x) + (next.x - curr.x));
      let nLen = Math.hypot(nx, ny);
      if (!Number.isFinite(nLen) || nLen < 1e-6) {
        nx = curr.x - cx;
        ny = curr.y - cy;
        nLen = Math.hypot(nx, ny);
      }
      if (!Number.isFinite(nLen) || nLen < 1e-6) continue;
      nx /= nLen;
      ny /= nLen;

      const invMass = 1 / Math.max(0.02, curr.mass || 1);
      const impulse = err * gain * dtPos * invMass;
      curr.vx += nx * impulse;
      curr.vy += ny * impulse;

      if (radialDamping > 0) {
        const rv = curr.vx * nx + curr.vy * ny;
        curr.vx -= nx * rv * radialDamping;
        curr.vy -= ny * rv * radialDamping;
      }
    }

    touched += 1;
  }

  return touched;
}

/**
 * Build/refresh per-cluster reference pose state used by deformation signals.
 *
 * For each live soft cluster, this caches:
 * - `indices`: stable node index list for the cluster,
 * - `refLocal`: centroid-local reference coordinates,
 * - `refRms`: scalar reference size used to normalize pose residuals,
 * - `curLocal`: reusable current-frame local coordinate buffer.
 *
 * The reference is reset whenever cluster membership topology changes
 * (keyed by node-index membership string), and stale clusters are removed.
 *
 * @param {object} sim - Live simulation state.
 * @param {object} s - Soft-body solver state (`nodes`, `springs`).
 * @returns {Map<number, {key:string, indices:Int32Array, refLocal:Float32Array, refRms:number, curLocal:Float32Array}>}
 */
function ensureSoftDeformationReferenceState(sim, s) {
  sim.softDeformReferenceState = sim.softDeformReferenceState || new Map();
  const refs = sim.softDeformReferenceState;

  const clusterIndices = new Map();
  for (let i = 0; i < (s.nodes || []).length; i++) {
    const node = s.nodes[i];
    const cid = node?.clusterId ?? 0;
    if (!clusterIndices.has(cid)) clusterIndices.set(cid, []);
    clusterIndices.get(cid).push(i);
  }

  const live = new Set();
  for (const [cid, indices] of clusterIndices.entries()) {
    const key = indices.join(',');
    let st = refs.get(cid);
    if (!st || st.key !== key) {
      let cx = 0;
      let cy = 0;
      let count = 0;
      for (const idx of indices) {
        const node = s.nodes[idx];
        const x = Number(node?.x);
        const y = Number(node?.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        cx += x;
        cy += y;
        count += 1;
      }
      if (count < 3) {
        refs.delete(cid);
        continue;
      }
      cx /= count;
      cy /= count;

      const validIndices = [];
      const local = [];
      for (const idx of indices) {
        const node = s.nodes[idx];
        const x = Number(node?.x);
        const y = Number(node?.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        validIndices.push(idx);
        local.push(x - cx, y - cy);
      }
      if (validIndices.length < 3) {
        refs.delete(cid);
        continue;
      }

      const refLocal = Float32Array.from(local);
      st = {
        key,
        indices: Int32Array.from(validIndices),
        refLocal,
        refRms: Math.max(1e-3, computeVectorRms(refLocal)),
        curLocal: new Float32Array(refLocal.length),
      };
      refs.set(cid, st);
    }
    live.add(cid);
  }

  for (const cid of [...refs.keys()]) {
    if (!live.has(cid)) refs.delete(cid);
  }
  return refs;
}

/**
 * Compute warning/severe deformation state for every soft cluster.
 *
 * Signals computed per cluster:
 * - `stretchMax`: largest spring stretch ratio.
 * - `stretchMin`: smallest spring stretch ratio (collapse indicator when very small).
 * - `areaRatio`: boundary-loop area / reference area.
 * - `poseErrorRms`: RMS rigid-aligned pose residual vs reference local shape.
 * - `poseErrorMax`: max rigid-aligned pose residual vs reference local shape.
 *
 * Color-state decision policy:
 * - warning (yellow/orange): legacy warning thresholds OR pose warning thresholds.
 * - severe (red): collapse-risk severe (non-membrane only) OR pose severe thresholds.
 *
 * @param {object} sim - Live simulation state (contains deformation reference cache).
 * @param {object} s - Soft-body solver state.
 * @param {Array<object>} loops - Cluster boundary loops from topology analysis.
 * @returns {{
 *   clusters:Array<object>,
 *   warningClusters:number[], severeClusters:number[], severeCollapseClusters:number[],
 *   warningSet:Set<number>, severeSet:Set<number>, severeCollapseSet:Set<number>,
 *   warningCount:number, severeCount:number, severeCollapseCount:number,
 *   worstStretch:number, worstAreaRatio:number, worstPoseError:number,
 * }}
 */
function buildSoftDeformationState(sim, s, loops) {
  const membraneSet = ensureSoftMembraneClusterSet(sim);
  const clusters = new Map();
  const ensureCluster = (cid) => {
    if (!clusters.has(cid)) {
      clusters.set(cid, {
        clusterId: cid,
        springs: 0,
        stretchMax: 1,
        stretchMin: 1,
        areaRatio: 1,
        poseErrorRms: 0,
        poseErrorMax: 0,
        severe: false,
        severeCollapse: false,
        warning: false,
      });
    }
    return clusters.get(cid);
  };

  for (const n of s.nodes || []) {
    const c = ensureCluster(n.clusterId ?? 0);
    if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) {
      c.severe = true;
      c.severeCollapse = true;
      c.warning = true;
    }
  }

  for (const sp of s.springs || []) {
    if (!Array.isArray(sp) || sp.length < 3) continue;
    const i = Number(sp[0]) | 0;
    const j = Number(sp[1]) | 0;
    const rest = Math.max(1e-4, Number(sp[2]) || 1e-4);
    const a = s.nodes[i];
    const b = s.nodes[j];
    if (!a || !b) continue;
    const cid = a.clusterId ?? b.clusterId ?? 0;
    const c = ensureCluster(cid);
    const len = Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0));
    if (!Number.isFinite(len)) {
      c.severe = true;
      c.severeCollapse = true;
      c.warning = true;
      continue;
    }
    const ratio = Math.max(1e-6, len / rest);
    c.springs += 1;
    c.stretchMax = Math.max(c.stretchMax, ratio);
    c.stretchMin = Math.min(c.stretchMin, ratio);
  }

  for (const loop of loops || []) {
    const cid = loop.clusterId ?? 0;
    const c = ensureCluster(cid);
    const areaNow = Math.abs(signedAreaCurrent(s.nodes, loop.indices));
    const restRaw = Math.abs(Number(sim.softAreaRest?.get(cid)) || 0);
    const rest = Math.max(1e-4, restRaw || areaNow || 1e-4);
    const areaRatio = Math.max(1e-6, areaNow / rest);
    c.areaRatio = areaRatio;
  }

  const refState = ensureSoftDeformationReferenceState(sim, s);
  for (const [cid, st] of refState.entries()) {
    const c = ensureCluster(cid);
    const ids = st.indices;
    if (!(ids instanceof Int32Array) || ids.length < 3) continue;

    let cx = 0;
    let cy = 0;
    let count = 0;
    for (let k = 0; k < ids.length; k++) {
      const node = s.nodes[ids[k]];
      const x = Number(node?.x);
      const y = Number(node?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      cx += x;
      cy += y;
      count += 1;
    }
    if (count < 3) {
      c.warning = true;
      c.severe = true;
      c.severeCollapse = true;
      continue;
    }
    cx /= count;
    cy /= count;

    if (!(st.curLocal instanceof Float32Array) || st.curLocal.length !== st.refLocal.length) {
      st.curLocal = new Float32Array(st.refLocal.length);
    }
    let valid = 0;
    for (let k = 0; k < ids.length; k++) {
      const node = s.nodes[ids[k]];
      const x = Number(node?.x);
      const y = Number(node?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        st.curLocal[k * 2] = 0;
        st.curLocal[k * 2 + 1] = 0;
        continue;
      }
      st.curLocal[k * 2] = x - cx;
      st.curLocal[k * 2 + 1] = y - cy;
      valid += 1;
    }
    if (valid < 3) {
      c.warning = true;
      c.severe = true;
      c.severeCollapse = true;
      continue;
    }

    const pose = computeRigidAlignedPoseResidual(st.refLocal, st.curLocal, st.refRms);
    if (!pose) {
      c.warning = true;
      c.severe = true;
      c.severeCollapse = true;
      continue;
    }
    c.poseErrorRms = Number.isFinite(pose.normalizedRms) ? pose.normalizedRms : 0;
    c.poseErrorMax = Number.isFinite(pose.normalizedMax) ? pose.normalizedMax : 0;
  }

  const clusterList = [...clusters.values()];
  let worstStretch = 1;
  let worstAreaRatio = 1;
  let worstPoseError = 0;
  const warningClusters = [];
  const severeClusters = [];
  const severeCollapseClusters = [];

  for (const c of clusterList) {
    const legacyWarn = c.stretchMax >= SOFT_DEFORM_WARN_STRETCH
      || c.areaRatio <= SOFT_DEFORM_WARN_AREA_RATIO_MIN
      || c.areaRatio >= SOFT_DEFORM_WARN_AREA_RATIO_MAX;
    const legacySevere = c.stretchMax >= SOFT_DEFORM_SEVERE_STRETCH
      || c.stretchMin <= 0.12
      || c.areaRatio <= SOFT_DEFORM_SEVERE_AREA_RATIO_MIN
      || c.areaRatio >= SOFT_DEFORM_SEVERE_AREA_RATIO_MAX;

    const poseAvailable = Number.isFinite(c.poseErrorRms) && Number.isFinite(c.poseErrorMax);
    const poseWarn = c.poseErrorRms >= SOFT_DEFORM_WARN_POSE_RMS
      || c.poseErrorMax >= SOFT_DEFORM_WARN_POSE_MAX;
    const poseSevere = c.poseErrorRms >= SOFT_DEFORM_SEVERE_POSE_RMS
      || c.poseErrorMax >= SOFT_DEFORM_SEVERE_POSE_MAX;

    const isMembraneCluster = membraneSet.has(c.clusterId);
    c.warning = c.warning || legacyWarn || (poseAvailable && poseWarn);
    c.severeCollapse = c.severeCollapse || (!isMembraneCluster && legacySevere);
    c.severe = c.severe || c.severeCollapse || (poseAvailable && poseSevere);

    if (c.warning) warningClusters.push(c.clusterId);
    if (c.severe) severeClusters.push(c.clusterId);
    if (c.severeCollapse) severeCollapseClusters.push(c.clusterId);
    worstStretch = Math.max(worstStretch, c.stretchMax);
    worstAreaRatio = Math.max(worstAreaRatio, Math.max(c.areaRatio, c.areaRatio > 0 ? 1 / c.areaRatio : 1));
    worstPoseError = Math.max(worstPoseError, c.poseErrorRms);
  }

  return {
    clusters: clusterList,
    warningClusters,
    severeClusters,
    severeCollapseClusters,
    warningSet: new Set(warningClusters),
    severeSet: new Set(severeClusters),
    severeCollapseSet: new Set(severeCollapseClusters),
    warningCount: warningClusters.length,
    severeCount: severeClusters.length,
    severeCollapseCount: severeCollapseClusters.length,
    worstStretch,
    worstAreaRatio,
    worstPoseError,
  };
}

function stabilizeSeverelyDeformedSoftClusters(sim, s, loops, deform) {
  if (!deform?.severeCollapseSet || deform.severeCollapseSet.size === 0) return;

  const severeSet = deform.severeCollapseSet;
  const centroids = new Map();
  for (const node of s.nodes || []) {
    const cid = node.clusterId ?? 0;
    if (!severeSet.has(cid)) continue;
    if (!centroids.has(cid)) centroids.set(cid, { x: 0, y: 0, n: 0 });
    const c = centroids.get(cid);
    c.x += node.x;
    c.y += node.y;
    c.n += 1;
  }
  for (const [cid, c] of centroids.entries()) {
    if (c.n > 0) {
      c.x /= c.n;
      c.y /= c.n;
    } else {
      centroids.delete(cid);
    }
  }

  const springClusterByIndex = new Array((s.springs || []).length).fill(-1);
  const clusterRestStats = new Map();
  for (let si = 0; si < (s.springs || []).length; si++) {
    const sp = s.springs[si];
    if (!Array.isArray(sp) || sp.length < 3) continue;
    const a = s.nodes[Number(sp[0]) | 0];
    const b = s.nodes[Number(sp[1]) | 0];
    if (!a || !b) continue;
    const cid = (a.clusterId ?? b.clusterId ?? 0);
    springClusterByIndex[si] = cid;
    if (!clusterRestStats.has(cid)) clusterRestStats.set(cid, { sum: 0, n: 0 });
    const st = clusterRestStats.get(cid);
    st.sum += Math.max(1e-4, Number(sp[2]) || 1e-4);
    st.n += 1;
  }

  for (const node of s.nodes || []) {
    const cid = node.clusterId ?? 0;
    if (!severeSet.has(cid)) continue;
    const c = centroids.get(cid);
    if (!c) continue;
    node.vx *= 0.5;
    node.vy *= 0.5;

    const dx = node.x - c.x;
    const dy = node.y - c.y;
    const d = Math.hypot(dx, dy);
    const meanRest = (clusterRestStats.get(cid)?.sum || 0) / Math.max(1, clusterRestStats.get(cid)?.n || 0);
    const maxRadius = Math.max(1.2, meanRest * 2.8);
    if (d > maxRadius && Number.isFinite(d)) {
      const k = maxRadius / Math.max(1e-6, d);
      node.x = c.x + dx * k;
      node.y = c.y + dy * k;
    }

    node.x = node.x * 0.86 + c.x * 0.14;
    node.y = node.y * 0.86 + c.y * 0.14;
  }

  for (let si = 0; si < (s.springs || []).length; si++) {
    const sp = s.springs[si];
    if (!Array.isArray(sp) || sp.length < 3) continue;
    const cid = springClusterByIndex[si];
    if (!severeSet.has(cid)) continue;
    const a = s.nodes[Number(sp[0]) | 0];
    const b = s.nodes[Number(sp[1]) | 0];
    if (!a || !b) continue;
    const len = Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0));
    if (!Number.isFinite(len)) continue;
    const rest = Math.max(1e-4, Number(sp[2]) || 1e-4);
    const baseRest = Math.max(1e-4, Number(sim.softSpringRestBaseline?.[si]) || rest);
    if (len > rest * 3.2 || len < rest * 0.15) {
      const boundedTarget = Math.max(baseRest * 0.65, Math.min(baseRest * 1.55, len));
      sp[2] = Math.max(1e-4, rest + (boundedTarget - rest) * 0.35);
      if (sim.softXPBDLambda && Number.isFinite(sim.softXPBDLambda[si])) sim.softXPBDLambda[si] = 0;
    }
  }

  for (const loop of loops || []) {
    const cid = loop.clusterId ?? 0;
    if (!severeSet.has(cid)) continue;
    const areaNow = signedAreaCurrent(s.nodes, loop.indices);
    sim.softAreaRest.set(cid, Math.abs(areaNow) > 1e-4 ? areaNow : 1e-4);
    sim.softAreaLambda.set(cid, 0);
  }
}

function ensureFramePipelineTiming(sim) {
  if (!sim) return null;
  const frame = Number(sim.frame) || 0;
  const solverPath = normalizeRuntimeSolverPath(sim?.controls?.runtimeSolverPath);
  const pipelineMode = normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, solverPath);
  if (!sim.pipelineTimingFrame || Number(sim.pipelineTimingFrame.frame) !== frame) {
    sim.pipelineTimingFrame = {
      frame,
      solverPath,
      pipelineMode,
      startedAtMs: performance.now(),
      spans: Object.create(null),
      counts: Object.create(null),
      events: [],
    };
  } else {
    sim.pipelineTimingFrame.solverPath = solverPath;
    sim.pipelineTimingFrame.pipelineMode = pipelineMode;
  }
  return sim.pipelineTimingFrame;
}

function recordPipelineTiming(sim, key, ms, meta = null) {
  const timing = ensureFramePipelineTiming(sim);
  if (!timing || !key) return;
  const dur = Math.max(0, Number(ms) || 0);
  timing.spans[key] = (Number(timing.spans[key]) || 0) + dur;
  timing.counts[key] = (Number(timing.counts[key]) || 0) + 1;
  timing.events.push({ key, ms: dur, ...(meta && typeof meta === 'object' ? meta : null) });
}

function measurePipelineSync(sim, key, fn, meta = null) {
  const startMs = performance.now();
  try {
    return fn();
  } finally {
    recordPipelineTiming(sim, key, performance.now() - startMs, meta);
  }
}

async function measurePipelineAsync(sim, key, fn, meta = null) {
  const startMs = performance.now();
  try {
    return await fn();
  } finally {
    recordPipelineTiming(sim, key, performance.now() - startMs, meta);
  }
}

function finalizeFramePipelineTiming(sim) {
  const timing = sim?.pipelineTimingFrame;
  if (!timing) return null;
  const finalized = {
    frame: Number(timing.frame) || 0,
    solverPath: timing.solverPath || 'baseline',
    pipelineMode: timing.pipelineMode || 'standard',
    totalMs: Math.max(0, performance.now() - (Number(timing.startedAtMs) || performance.now())),
    spans: { ...(timing.spans || {}) },
    counts: { ...(timing.counts || {}) },
    events: Array.isArray(timing.events) ? timing.events.slice(-400) : [],
  };
  sim.pipelineTimingLast = finalized;
  sim.pipelineTimingHistory = Array.isArray(sim.pipelineTimingHistory) ? sim.pipelineTimingHistory : [];
  sim.pipelineTimingHistory.push(finalized);
  if (sim.pipelineTimingHistory.length > 240) {
    sim.pipelineTimingHistory.splice(0, sim.pipelineTimingHistory.length - 240);
  }
  return finalized;
}

async function stepBodiesAndInject(sim, vxField, vyField) {
  ensureFramePipelineTiming(sim);
  const preambleStartMs = performance.now();
  const n = sim.controls.n;
  const worldScale = getWorldScale(sim.controls);
  const worldSize = n * worldScale;
  const dtRaw = Number(sim.controls.dt) || 0.01;
  const dt = Math.max(0.001, Math.min(0.02, dtRaw));
  const dtNorm = Math.max(0.2, Math.min(1.5, (dt * 60) || 1));
  if (Math.abs(dtRaw - dt) > 1e-6) {
    const prevWarnFrame = Number(sim.softDtClampWarnFrame) || -9999;
    if ((sim.frame - prevWarnFrame) > 240) {
      sim.softDtClampWarnFrame = sim.frame;
      console.warn('[gpu-lab] soft-body dt clamped for stability', { dtRaw, dt });
    }
  }
  const bodies = sim.bodies;
  const dragK = sim.controls.bodyDrag;
  const feedbackK = sim.controls.bodyFeedback;
  const softClusterFluidTorqueCoupling = Math.max(0, Number(sim.controls?.softClusterFluidTorqueCoupling) || SOFT_CLUSTER_FLOW_FORCE_SHARE);
  const softClusterAngularProjection = Math.max(0, Number(sim.controls?.softClusterAngularProjection) || SOFT_CLUSTER_ANGULAR_PROJECTION);
  const softClusterCollisionAngularProjection = Math.max(0, Number(sim.controls?.softClusterCollisionAngularProjection) || SOFT_CLUSTER_COLLISION_ANGULAR_PROJECTION);
  const softClusterStabilizersOn = sim.controls?.enableSoftClusterStabilizers !== false;
  const softClusterTugCouplingBase = softClusterStabilizersOn ? SOFT_CLUSTER_TUG_COUPLING : 0;
  const softClusterRelativeDragBase = softClusterStabilizersOn ? SOFT_CLUSTER_RELATIVE_DRAG : 0;
  const softClusterLinearProjectionBase = softClusterStabilizersOn ? SOFT_CLUSTER_LINEAR_PROJECTION : 0;
  const softClusterAngularProjectionBase = softClusterStabilizersOn ? softClusterAngularProjection : 0;
  const softSpringRestRecoveryOn = sim.controls?.enableSoftSpringRestRecovery !== false;
  const membraneBoundaryXpbdOn = sim.controls?.enableMembraneBoundaryXpbd !== false;
  const membraneShapeMemoryOn = sim.controls?.enableMembraneShapeMemory !== false;
  const membranePressureOn = sim.controls?.enableMembranePressure !== false;
  const swimGain = sim.controls?.enableArtificialSwim ? 1 : 0;
  const viscMap = sim.viscMapCpu;

  const localHoneyDrag = (x, y) => {
    const v = sampleFieldBilinear(viscMap, n, x, y, worldScale);
    // Unified viscosity response curve for both rigid and soft bodies.
    return 1.0 + Math.pow(Math.max(0, Math.min(1, v)), 2.2) * 14.0;
  };
  const viscosityMotionResponse = (honey, vmaxBase) => ({
    damp: Math.max(0.72, 1.0 - 0.018 * honey),
    vmax: Math.max(0.4, vmaxBase / (1 + 0.28 * honey)),
  });

  if (bodies.rigid[0]) bodies.rigid[0].mass = sim.controls.massLight;
  if (bodies.rigid[1]) bodies.rigid[1].mass = sim.controls.massHeavy;
  for (const rb of bodies.rigid) {
    rb.inertia = 0.5 * rb.mass * rb.r * rb.r;
  }
  for (const node of bodies.soft.nodes) node.mass = sim.controls.massSoft;

  const softCentroidBefore = computeSoftCentroid(bodies.soft.nodes);
  const computeRigidCenter = (arr) => {
    if (!arr.length) return { x: 0, y: 0 };
    let sx = 0, sy = 0;
    for (const r of arr) { sx += r.x; sy += r.y; }
    return { x: sx / arr.length, y: sy / arr.length };
  };
  const rigidCenterBefore = computeRigidCenter(bodies.rigid);
  const s = bodies.soft;
  const obstacleMask = sim?.obstacleMaskCpu;

  const cells = n * n;
  const SELF_FEEDBACK_SUPPRESSION = 0.82;
  if (!(sim._bodyFeedbackPrevVx instanceof Float32Array) || sim._bodyFeedbackPrevVx.length !== cells) {
    sim._bodyFeedbackPrevVx = new Float32Array(cells);
    sim._bodyFeedbackPrevVy = new Float32Array(cells);
    sim._bodyFeedbackCurrVx = new Float32Array(cells);
    sim._bodyFeedbackCurrVy = new Float32Array(cells);
  }
  const bodyFeedbackPrevVx = sim._bodyFeedbackPrevVx;
  const bodyFeedbackPrevVy = sim._bodyFeedbackPrevVy;
  const bodyFeedbackCurrVx = sim._bodyFeedbackCurrVx;
  const bodyFeedbackCurrVy = sim._bodyFeedbackCurrVy;
  bodyFeedbackCurrVx.fill(0);
  bodyFeedbackCurrVy.fill(0);

  // Optional interaction-lab scripted body motion (pinned / circular drag).
  applyInteractionLabBodyMotion(sim);

  const solverPath = normalizeRuntimeSolverPath(sim?.controls?.runtimeSolverPath);
  const allowPassEdgeFlowPush = sim?.controls?.allowPassEdgeFlowPush === true;
  const softNodeMomentumScale = (() => {
    const sums = new Float32Array(s.nodes.length);
    const counts = new Uint16Array(s.nodes.length);
    for (const sp of (s.springs || [])) {
      const a = Number(sp?.[0])|0;
      const b = Number(sp?.[1])|0;
      if (a < 0 || b < 0 || a >= s.nodes.length || b >= s.nodes.length) continue;
      const edgeVelocityMode = Number(sp?.[5]) === EDGE_BODY_MODE.PASS
        ? EDGE_BODY_MODE.PASS
        : (Number(sp?.[3]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK);
      const m = Number(sp?.[6]);
      let mm = Number.isFinite(m) ? clamp(m,0,1) : 1;
      if (!allowPassEdgeFlowPush && edgeVelocityMode === EDGE_BODY_MODE.PASS) {
        mm = 0;
      }
      sums[a] += mm; sums[b] += mm; counts[a] += 1; counts[b] += 1;
    }
    return (idx) => counts[idx] > 0 ? (sums[idx] / counts[idx]) : 1;
  })();

  let rigidCarryTransfer = 0;
  let softCarryTransfer = 0;

  recordPipelineTiming(sim, 'bodies.preamble', performance.now() - preambleStartMs);
  const rigidStageStartMs = performance.now();
  sim.rigidStepBaselineRuntime = {
    mode: solverPath === 'gpu-only' ? 'gpu-only-path' : 'pending',
    reason: solverPath === 'gpu-only' ? 'handled-in-gpu-only-step' : 'pending',
    processedBodies: 0,
    sampleCount: 0,
    jsMarshalMs: 0,
    backendComputeMs: 0,
    totalMs: 0,
  };
  if (solverPath === 'gpu-only') {
    rigidCarryTransfer = await stepRigidBodiesGpuOnly({
      sim,
      bodies,
      vxField,
      vyField,
      n,
      worldSize,
      dt,
      dtNorm,
      dragK,
      swimGain,
      localHoneyDrag,
      viscosityMotionResponse,
      obstacleMask,
      bodyFeedbackPrevVx,
      bodyFeedbackPrevVy,
      selfFeedbackSuppression: SELF_FEEDBACK_SUPPRESSION,
      rigidVerticesWorld,
      sampleFluidForBodyCoupling,
      applyBounceBoundary,
      allowPassEdgeFlowPush,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
        state: (() => {
          sim.rigidStepWgslState ||= {};
          const st = sim.rigidStepWgslState;
          if (st.enableAuthoritativeRigidStep !== false) {
            st.enableAuthoritativeRigidStep = true;
          }
          return st;
        })(),
      },
    });
  } else {
    const runRigidBaselineCpu = () => stepRigidBodiesBaselineCpu({
      sim,
      bodies,
      vxField,
      vyField,
      n,
      worldSize,
      dt,
      dtNorm,
      dragK,
      swimGain,
      localHoneyDrag,
      viscosityMotionResponse,
      obstacleMask,
      bodyFeedbackPrevVx,
      bodyFeedbackPrevVy,
      selfFeedbackSuppression: SELF_FEEDBACK_SUPPRESSION,
      rigidVerticesWorld,
      sampleFluidForBodyCoupling,
      applyBounceBoundary,
      allowPassEdgeFlowPush,
    });

    if (sim?.rigidStepBaselineBackend) {
      try {
        const wasmRigid = sim.rigidStepBaselineBackend.stepBodies({
          sim,
          bodies,
          vxField,
          vyField,
          n,
          worldSize,
          dt,
          dtNorm,
          dragK,
          swimGain,
          obstacleMask,
          bodyFeedbackPrevVx,
          bodyFeedbackPrevVy,
          selfFeedbackSuppression: SELF_FEEDBACK_SUPPRESSION,
          rigidVerticesWorld,
          sampleFluidForBodyCoupling,
          localHoneyDrag,
          allowPassEdgeFlowPush,
        });
        if (wasmRigid?.ok === false) {
          throw new Error(String(wasmRigid?.reason || 'rigid-step-rejected'));
        }
        rigidCarryTransfer = Number(wasmRigid?.rigidCarryTransfer) || 0;
        sim.rigidStepBaselineRuntime = {
          mode: 'wasm',
          reason: sim?.rigidStepBaselineBackend?.label || 'wasm',
          processedBodies: Number(wasmRigid?.processedBodies) || 0,
          sampleCount: Number(wasmRigid?.sampleCount) || 0,
          jsMarshalMs: Number(wasmRigid?.jsMarshalMs) || 0,
          backendComputeMs: Number(wasmRigid?.backendComputeMs) || 0,
          totalMs: Number(wasmRigid?.totalMs) || 0,
        };
      } catch (err) {
        rigidCarryTransfer = runRigidBaselineCpu();
        sim.rigidStepBaselineRuntime = {
          mode: 'cpu-fallback',
          reason: `wasm-error:${String(err?.message || err)}`,
          processedBodies: Number(bodies?.rigid?.length) || 0,
          sampleCount: 0,
          jsMarshalMs: 0,
          backendComputeMs: 0,
          totalMs: 0,
        };
      }
    } else {
      rigidCarryTransfer = runRigidBaselineCpu();
      sim.rigidStepBaselineRuntime = {
        mode: 'cpu-baseline',
        reason: 'backend-disabled',
        processedBodies: Number(bodies?.rigid?.length) || 0,
        sampleCount: 0,
        jsMarshalMs: 0,
        backendComputeMs: 0,
        totalMs: 0,
      };
    }
  }
  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.rigid.gpuOnly' : 'bodies.rigid.baseline', performance.now() - rigidStageStartMs);

  const softMembraneClusterSet = ensureSoftMembraneClusterSet(sim);
  const softFluidStageStartMs = performance.now();
  if (solverPath === 'gpu-only') {
    const softFluidResult = applySoftFluidCouplingGpuOnly({
      sim,
      soft: s,
      n,
      dt,
      dtNorm,
      vxField,
      vyField,
      dragK,
      swimGain,
      localHoneyDrag,
      viscosityMotionResponse,
      obstacleMask,
      bodyFeedbackPrevVx,
      bodyFeedbackPrevVy,
      selfFeedbackSuppression: SELF_FEEDBACK_SUPPRESSION,
      softMembraneClusterSet,
      softNodeMomentumScale,
      constants: {
        SOFT_NODE_FLOW_COUPLING,
        SOFT_NODE_LOCAL_FLOW_SHARE,
        SOFT_CLUSTER_TUG_COUPLING: softClusterTugCouplingBase,
        SOFT_CLUSTER_RELATIVE_DRAG: softClusterRelativeDragBase,
        softClusterFluidTorqueCoupling,
        SOFT_CLUSTER_LINEAR_PROJECTION: softClusterLinearProjectionBase,
        softClusterAngularProjection: softClusterAngularProjectionBase,
      },
      computeSoftCentroid,
      computeSoftClusterKinematics,
      projectNodesTowardClusterRigidMotion,
      sampleFluidForBodyCoupling,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        state: (sim.softFluidCouplingWgslState ||= {}),
      },
    });
    softCarryTransfer += softFluidResult.softCarryTransfer || 0;
  } else {
    const softCentroid = computeSoftCentroid(s.nodes);
    let softClusterKinematics = computeSoftClusterKinematics(s.nodes);
    const clusterCarryMap = new Map();
    const clusterFluidLoadMap = new Map();
    const ensureClusterLoad = (cid) => {
      if (!clusterFluidLoadMap.has(cid)) {
        clusterFluidLoadMap.set(cid, { forceX: 0, forceY: 0, torque: 0, count: 0 });
      }
      return clusterFluidLoadMap.get(cid);
    };

    for (let i = 0; i < s.nodes.length; i++) {
      const node = s.nodes[i];
      const mass = Math.max(0.02, node.mass);
      const invMass = 1 / mass;
      const cx = node.x - softCentroid.x;
      const cy = node.y - softCentroid.y;
      const activeSwimPhase = sim.frame * 0.12 + i * 1.57;
      const activeSwimAmp = 0.008 * (1 + 0.2 * Math.sin(sim.frame * 0.05 + i));
      const swimX = swimGain * (-cy * activeSwimAmp + Math.cos(activeSwimPhase) * 0.004) * invMass;
      const swimY = swimGain * (cx * activeSwimAmp + Math.sin(activeSwimPhase) * 0.004) * invMass;
      const honey = localHoneyDrag(node.x, node.y);
      const cid = node.clusterId ?? 0;
      const isMembraneCluster = softMembraneClusterSet.has(cid);
      const nodeMomentum = softNodeMomentumScale(i);
      const flowCouplingBase = isMembraneCluster ? (SOFT_NODE_FLOW_COUPLING * 0.88) : SOFT_NODE_FLOW_COUPLING;
      const flowCoupling = flowCouplingBase * nodeMomentum;

      const clusterKin = softClusterKinematics.get(cid);
      const clusterX = Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x;
      const clusterY = Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y;
      const clusterVx = Number.isFinite(Number(clusterKin?.vx)) ? Number(clusterKin.vx) : 0;
      const clusterVy = Number.isFinite(Number(clusterKin?.vy)) ? Number(clusterKin.vy) : 0;
      const clusterOmega = Number.isFinite(Number(clusterKin?.omega)) ? Number(clusterKin.omega) : 0;

      const rx = node.x - clusterX;
      const ry = node.y - clusterY;
      const fx = sampleFluidForBodyCoupling(
        vxField,
        n,
        node.x,
        node.y,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVx,
        SELF_FEEDBACK_SUPPRESSION,
      );
      const fy = sampleFluidForBodyCoupling(
        vyField,
        n,
        node.x,
        node.y,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVy,
        SELF_FEEDBACK_SUPPRESSION,
      );
      const clusterLocalVx = clusterVx - clusterOmega * ry;
      const clusterLocalVy = clusterVy + clusterOmega * rx;

      // Explicit soft-cluster force/torque accumulation from fluid relative motion.
      const forceX = (fx - clusterLocalVx) * dragK * honey * flowCoupling * mass;
      const forceY = (fy - clusterLocalVy) * dragK * honey * flowCoupling * mass;
      const carryX = forceX * invMass;
      const carryY = forceY * invMass;

      // Keep a local deformation response path so springs still flex naturally.
      const localCarryX = (fx - node.vx) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;
      const localCarryY = (fy - node.vy) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;

      node.vx += localCarryX * dt * 60 + swimX * dtNorm;
      node.vy += localCarryY * dt * 60 + swimY * dtNorm;

      const load = ensureClusterLoad(cid);
      load.forceX += forceX;
      load.forceY += forceY;
      load.torque += rx * forceY - ry * forceX;
      load.count += 1;

      const st = clusterCarryMap.get(cid) || { sumX: 0, sumY: 0, count: 0, maxX: 0, maxY: 0, maxMag: 0 };
      st.sumX += carryX;
      st.sumY += carryY;
      st.count += 1;
      const cmag = Math.hypot(carryX, carryY);
      if (cmag > st.maxMag) {
        st.maxMag = cmag;
        st.maxX = carryX;
        st.maxY = carryY;
      }
      clusterCarryMap.set(cid, st);

      const softVisc = viscosityMotionResponse(honey, 2.8);
      node.vx *= softVisc.damp;
      node.vy *= softVisc.damp;
      const nMax = softVisc.vmax;
      const nMag = Math.hypot(node.vx, node.vy);
      if (nMag > nMax) {
        node.vx = (node.vx / nMag) * nMax;
        node.vy = (node.vy / nMag) * nMax;
      }
      softCarryTransfer += Math.hypot(carryX, carryY);
    }

    // Apply cluster linear/angular acceleration from accumulated fluid loads.
    const clusterAccelMap = new Map();
    for (const [cid, load] of clusterFluidLoadMap.entries()) {
      const clusterKin = softClusterKinematics.get(cid);
      const clusterMass = Math.max(0.02, Number(clusterKin?.mass) || (Math.max(1, load.count) * sim.controls.massSoft));
      const clusterInertia = Math.max(1e-4, Number(clusterKin?.inertia) || 1e-4);
      clusterAccelMap.set(cid, {
        x: Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x,
        y: Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y,
        ax: load.forceX / clusterMass,
        ay: load.forceY / clusterMass,
        alpha: load.torque / clusterInertia,
      });
    }

    for (const node of s.nodes) {
      const cid = node.clusterId ?? 0;
      const acc = clusterAccelMap.get(cid);
      if (!acc) continue;
      const rx = node.x - acc.x;
      const ry = node.y - acc.y;
      const flowShare = softMembraneClusterSet.has(cid)
        ? (softClusterFluidTorqueCoupling * 0.7)
        : softClusterFluidTorqueCoupling;
      node.vx += (acc.ax - acc.alpha * ry) * flowShare * dt * 60;
      node.vy += (acc.ay + acc.alpha * rx) * flowShare * dt * 60;
    }

    // Redistribute flow pull across each cluster so one stretched node drags the body along.
    for (const node of s.nodes) {
      const cid = node.clusterId ?? 0;
      const st = clusterCarryMap.get(cid);
      if (!st || st.count <= 0) continue;
      const meanX = st.sumX / st.count;
      const meanY = st.sumY / st.count;
      const pullX = meanX * 0.6 + st.maxX * 0.4;
      const pullY = meanY * 0.6 + st.maxY * 0.4;
      const tugCoupling = softMembraneClusterSet.has(cid) ? (softClusterTugCouplingBase * 0.45) : (softClusterTugCouplingBase * 0.72);
      node.vx += pullX * tugCoupling * dt * 60;
      node.vy += pullY * tugCoupling * dt * 60;
    }

    // Keep clusters coherent: damp relative drift around cluster mean velocity.
    const clusterVelMap = new Map();
    for (const node of s.nodes) {
      const cid = node.clusterId ?? 0;
      const st = clusterVelMap.get(cid) || { sumVx: 0, sumVy: 0, count: 0 };
      st.sumVx += node.vx || 0;
      st.sumVy += node.vy || 0;
      st.count += 1;
      clusterVelMap.set(cid, st);
    }
    for (const node of s.nodes) {
      const cid = node.clusterId ?? 0;
      const st = clusterVelMap.get(cid);
      if (!st || st.count <= 0) continue;
      const meanVx = st.sumVx / st.count;
      const meanVy = st.sumVy / st.count;
      const relDamp = softMembraneClusterSet.has(cid) ? (softClusterRelativeDragBase * 0.65) : softClusterRelativeDragBase;
      node.vx -= (node.vx - meanVx) * relDamp * dtNorm;
      node.vy -= (node.vy - meanVy) * relDamp * dtNorm;
    }

    // Explicit soft angular inertia response: project nodes toward cluster rigid motion field.
    softClusterKinematics = computeSoftClusterKinematics(s.nodes);
    projectNodesTowardClusterRigidMotion(s.nodes, softClusterKinematics, {
      linearGain: softClusterLinearProjectionBase * dtNorm,
      angularGain: softClusterAngularProjectionBase * dtNorm,
      membraneClusterSet: softMembraneClusterSet,
      membraneGainScale: 0.72,
    });
  }

  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.softFluidCoupling.gpuOnly' : 'bodies.softFluidCoupling.baseline', performance.now() - softFluidStageStartMs);

  const dtPos = Math.max(1e-4, dt * SOFT_INTEGRATION_SCALE);
  const softSpringStiffness = normalizeSoftSpringStiffness(sim.controls?.softSpringStiffness);
  const softSpringBaseCompliance = normalizeCompliance(sim.controls?.softSpringBaseCompliance, SOFT_XPBD_BASE_COMPLIANCE);
  const softAreaBaseCompliance = normalizeCompliance(sim.controls?.softAreaBaseCompliance, SOFT_AREA_BASE_COMPLIANCE);
  const membraneEdgeBaseCompliance = normalizeCompliance(sim.controls?.membraneEdgeBaseCompliance, MEMBRANE_EDGE_BASE_COMPLIANCE);
  const membraneBendBaseCompliance = normalizeCompliance(sim.controls?.membraneBendBaseCompliance, MEMBRANE_BEND_BASE_COMPLIANCE);

  if (!sim.softXPBDLambda || sim.softXPBDLambda.length !== s.springs.length) {
    sim.softXPBDLambda = ensureLambdaCacheSize(sim.softXPBDLambda, s.springs.length, 20);
  }
  if (!sim.softSpringRestBaseline || sim.softSpringRestBaseline.length !== s.springs.length) {
    sim.softSpringRestBaseline = new Float32Array(s.springs.length);
    for (let i = 0; i < s.springs.length; i++) {
      sim.softSpringRestBaseline[i] = Math.max(1e-4, Number(s.springs[i]?.[2]) || 1e-4);
    }
  }

  const softSpringXpbdStageStartMs = performance.now();
  if (solverPath === 'gpu-only') {
    applySoftSpringsXPBDVelocityGpuOnly({
      soft: s,
      dtPos,
      stiffnessScale: softSpringStiffness,
      lambdaCache: sim.softXPBDLambda,
      softXpbdIters: SOFT_XPBD_ITERS,
      softXpbdBaseCompliance: softSpringBaseCompliance,
      clamp,
      skipClusterSet: softMembraneClusterSet,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
        state: (sim.softSpringXpbdWgslState ||= {}),
      },
    });
  } else {
    applySoftSpringsXPBDVelocity(s, dtPos, softSpringStiffness, sim.softXPBDLambda, {
      skipClusterSet: softMembraneClusterSet,
      softXpbdBaseCompliance: softSpringBaseCompliance,
      softXpbdIters: SOFT_XPBD_ITERS,
    });
  }
  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.softSpringXpbd.gpuOnly' : 'bodies.softSpringXpbd.baseline', performance.now() - softSpringXpbdStageStartMs);

  const softClusterLoops = buildSoftClusterBoundaryLoops(s.nodes, s.springs, {
    blockMode: EDGE_BODY_MODE.BLOCK,
  });
  const membraneConstraintStageStartMs = performance.now();
  const membraneBoundaryClusters = membraneBoundaryXpbdOn
    ? (solverPath === 'gpu-only'
      ? await applySoftMembraneBoundaryXPBDVelocityGpuOnly({
        sim,
        soft: s,
        loops: softClusterLoops,
        dtPos,
        membraneClusterSet: softMembraneClusterSet,
        clamp,
        membraneEdgeXpbdIters: MEMBRANE_EDGE_XPBD_ITERS,
        membraneEdgeBaseCompliance,
        membraneBendXpbdIters: MEMBRANE_BEND_XPBD_ITERS,
        membraneBendBaseCompliance,
        wgslOffload: {
          enabled: true,
          device: sim?.device,
          modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
          state: (() => {
            const st = (sim.softMembraneBoundaryWgslState ||= {});
            if (st.enableAuthoritativeMembraneBoundaryEdge !== false) {
              st.enableAuthoritativeMembraneBoundaryEdge = true;
            }
            if (st.enableAuthoritativeMembraneBend !== false) {
              st.enableAuthoritativeMembraneBend = true;
            }
            return st;
          })(),
        },
      })
      : applySoftMembraneBoundaryXPBDVelocity(sim, s, softClusterLoops, dtPos, {
        membraneEdgeBaseCompliance,
        membraneBendBaseCompliance,
        membraneEdgeXpbdIters: MEMBRANE_EDGE_XPBD_ITERS,
        membraneBendXpbdIters: MEMBRANE_BEND_XPBD_ITERS,
      }))
    : 0;
  const membraneShapeClusters = membraneShapeMemoryOn
    ? (solverPath === 'gpu-only'
      ? await applySoftMembraneShapeMemoryVelocityGpuOnly({
        sim,
        soft: s,
        loops: softClusterLoops,
        dtPos,
        membraneClusterMap: sim.softMembraneClusterMap,
        clamp,
        membraneShapeMemoryIters: MEMBRANE_SHAPE_MEMORY_ITERS,
        membraneShapeMemoryGain: MEMBRANE_SHAPE_MEMORY_GAIN,
        membraneShapeMemoryMaxShiftFrac: MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC,
        wgslOffload: {
          enabled: true,
          device: sim?.device,
          state: (() => {
            const st = (sim.softMembraneShapeMemoryWgslState ||= {});
            if (st.enableAuthoritativeShapeMemory !== false) {
              st.enableAuthoritativeShapeMemory = true;
            }
            return st;
          })(),
        },
      })
      : applySoftMembraneShapeMemoryVelocity(sim, s, softClusterLoops, dtPos))
    : 0;
  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.membraneConstraints.gpuOnly' : 'bodies.membraneConstraints.baseline', performance.now() - membraneConstraintStageStartMs);

  ensureSoftAreaRestState(sim, s, softClusterLoops, dtPos);
  const softAreaXpbdStageStartMs = performance.now();
  if (solverPath === 'gpu-only') {
    applySoftAreaXPBDVelocityGpuOnly({
      sim,
      soft: s,
      loops: softClusterLoops,
      dtPos,
      stiffnessScale: softSpringStiffness,
      softAreaXpbdIters: SOFT_AREA_XPBD_ITERS,
      softAreaBaseCompliance,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
        state: (sim.softAreaXpbdWgslState ||= {}),
      },
    });
  } else {
    applySoftAreaXPBDVelocity(sim, s, softClusterLoops, dtPos, softSpringStiffness, {
      softAreaBaseCompliance,
      softAreaXpbdIters: SOFT_AREA_XPBD_ITERS,
    });
  }
  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.softAreaXpbd.gpuOnly' : 'bodies.softAreaXpbd.baseline', performance.now() - softAreaXpbdStageStartMs);

  const membranePressureStageStartMs = performance.now();
  const membranePressureClusters = membranePressureOn
    ? (solverPath === 'gpu-only'
      ? applySoftMembraneCellPressureGpuOnly({
        sim,
        soft: s,
        loops: softClusterLoops,
        dtPos,
        signedAreaCurrent,
        clamp,
        membraneCellBasePressureGain: MEMBRANE_CELL_BASE_PRESSURE_GAIN,
        membraneCellBaseRadialDamping: MEMBRANE_CELL_BASE_RADIAL_DAMPING,
        wgslOffload: {
          enabled: true,
          device: sim?.device,
          state: (sim.softMembranePressureWgslState ||= {}),
        },
      })
      : applySoftMembraneCellPressure(sim, s, softClusterLoops, dtPos))
    : 0;

  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.membranePressure.gpuOnly' : 'bodies.membranePressure.baseline', performance.now() - membranePressureStageStartMs);

  const hybridNodeVCap = 3.2;
  const softIntegrateStageStartMs = performance.now();
  if (solverPath === 'gpu-only') {
    const softIntegrateRuntime = await integrateSoftBodiesGpuOnly({
      soft: s,
      n,
      worldSize,
      dt,
      softIntegrationScale: SOFT_INTEGRATION_SCALE,
      hybridNodeVCap,
      applyBounceBoundary,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        state: (sim.softIntegrateWgslState ||= {}),
      },
    });
    sim.softIntegrateRuntime = softIntegrateRuntime || { mode: 'cpu-fallback', reason: 'unknown' };
  } else {
    for (const node of s.nodes) {
      const vmag = Math.hypot(node.vx, node.vy);
      if (vmag > hybridNodeVCap) {
        node.vx = (node.vx / vmag) * hybridNodeVCap;
        node.vy = (node.vy / vmag) * hybridNodeVCap;
      }
      node.x = node.x + node.vx * dt * SOFT_INTEGRATION_SCALE;
      node.y = node.y + node.vy * dt * SOFT_INTEGRATION_SCALE;
      applyBounceBoundary(node, worldSize, 0.78);
    }
    sim.softIntegrateRuntime = { mode: 'cpu-baseline', reason: 'baseline-path' };
  }

  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.softIntegrate.gpuOnly' : 'bodies.softIntegrate.baseline', performance.now() - softIntegrateStageStartMs);

  const rigidPostIntegrateStageStartMs = performance.now();
  if (solverPath === 'gpu-only') {
    const rigidPostIntegrateRuntime = await stabilizeRigidPostIntegrateGpuOnly({
      rigidBodies: bodies.rigid,
      velocityCap: 4.0,
      omegaCap: 0.22,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        state: (sim.rigidPostIntegrateWgslState ||= {}),
      },
    });
    sim.rigidPostIntegrateRuntime = rigidPostIntegrateRuntime || { mode: 'cpu-fallback', reason: 'unknown' };
  } else {
    sim.rigidPostIntegrateRuntime = { mode: 'cpu-baseline', reason: 'baseline-path' };
    for (const rb of bodies.rigid) {
      const vmag = Math.hypot(rb.vx, rb.vy);
      const vcap = 4.0;
      if (vmag > vcap) {
        rb.vx = (rb.vx / vmag) * vcap;
        rb.vy = (rb.vy / vmag) * vcap;
      }
      rb.omega = Math.max(-0.22, Math.min(0.22, rb.omega || 0));
    }
  }

  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.rigidPostIntegrate.gpuOnly' : 'bodies.rigidPostIntegrate.baseline', performance.now() - rigidPostIntegrateStageStartMs);

  const rigidContactDebug = [];

  const collisionStageStartMs = performance.now();
  const rigidRigidBroadphaseRuntime = {
    enabled: solverPath !== 'gpu-only',
    cellSize: RIGID_RIGID_SPATIAL_HASH_CELL_SIZE,
    backendMode: sim?.rigidRigidCandidateBackendMode || 'js',
    backendReason: sim?.rigidRigidCandidateBackendReason || null,
    backendLabel: null,
    phases: [],
    totalBruteForcePairs: 0,
    totalCheckedPairs: 0,
    totalPrunedPairs: 0,
    reductionPct: 0,
    reuseAttempts: 0,
    reuseHits: 0,
    reuseMisses: 0,
    reuseHitRatePct: 0,
    dedupeMs: 0,
    sortMs: 0,
    totalBuildMs: 0,
    jsMarshalMs: 0,
    backendComputeMs: 0,
    emitAttempts: 0,
    duplicatesRejected: 0,
    pairsOut: 0,
    postPassSkipped: 0,
  };
  const rigidSoftBroadphaseRuntime = {
    enabled: solverPath !== 'gpu-only',
    cellSize: RIGID_SOFT_SPATIAL_HASH_CELL_SIZE,
    phases: [],
    totalRawNodeChecks: 0,
    totalCandidateNodeChecks: 0,
    totalPrunedNodeChecks: 0,
    nodeReductionPct: 0,
    totalRawEdgeChecks: 0,
    totalCandidateEdgeChecks: 0,
    totalPrunedEdgeChecks: 0,
    edgeReductionPct: 0,
    totalBlockedEdges: 0,
  };
  const collisionCpuRuntime = {
    enabled: solverPath !== 'gpu-only',
    reason: solverPath === 'gpu-only' ? 'gpu-only-collision-path' : null,
    collisionIterations: 2,
    stageMs: {
      rigidRigidPre: 0,
      rigidSoft: 0,
      softSoftNode: 0,
      softSoftEdge: 0,
      rigidRigidPost: 0,
    },
    substageMs: {
      candidateBuildPre: 0,
      candidateBuildPost: 0,
      candidateBuildTotal: 0,
      rigidRigidNarrowphasePre: 0,
      rigidSoftNarrowphase: 0,
      softSoftNodeNarrowphase: 0,
      softSoftEdgeNarrowphase: 0,
      rigidRigidNarrowphasePost: 0,
      narrowphaseTotal: 0,
      boundaryApply: 0,
      applyTotal: 0,
      total: 0,
    },
    rigidRigidPairChecks: 0,
    rigidRigidPairHits: 0,
    rigidRigidProxyPairChecks: 0,
    rigidRigidAabbRejected: 0,
    rigidRigidRadiusRejected: 0,
    rigidRigidSatCalls: 0,
    rigidRigidSatHits: 0,
    rigidRigidEarlyRejectChecks: 0,
    rigidRigidEarlyRadiusRejected: 0,
    rigidRigidPostSkipped: 0,
    rigidSoftNodeRawChecks: 0,
    rigidSoftNodeChecks: 0,
    rigidSoftNodeCandidateChecks: 0,
    rigidSoftNodeHits: 0,
    rigidSoftEdgeRawChecks: 0,
    rigidSoftEdgeChecks: 0,
    rigidSoftEdgeCandidateChecks: 0,
    rigidSoftEdgeHits: 0,
    softSoftNodeChecks: 0,
    softSoftNodeHits: 0,
    softSoftEdgeChecks: 0,
    softSoftEdgeHits: 0,
    polyCacheLookups: 0,
    polyCacheHits: 0,
    polyCacheLocalHits: 0,
    polyCacheSharedHits: 0,
    polyCacheMisses: 0,
    polyCacheRebuilds: 0,
    polyBodiesBuilt: 0,
    polyBodiesReused: 0,
    polyBuildMs: 0,
    polyCacheHitRatePct: 0,
  };
  if (solverPath === 'gpu-only') {
    const collisionResult = await runCollisionIterationsGpuOnly({
      rigidBodies: bodies.rigid,
      soft: s,
      rigidContactDebug,
      collisionIterations: 2,
      rigidRigidSlop: 0.32,
      rigidSoftNodeSlop: 0.18,
      rigidSoftEdgeSlop: 0.16,
      softNodeNodeSlop: 0.22,
      softNodeEdgeSlop: 0.12,
      edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
      n: worldSize,
      rigidBounce: 0.84,
      softBounce: 0.78,
      resolveRigidRigidCollisionPassGpuOnly,
      resolveRigidSoftCollisionPassGpuOnly,
      resolveSoftSoftCollisionPassGpuOnly,
      applyCollisionBoundaryPassGpuOnly,
      resolveRigidVsRigidPolygonCollision,
      resolveCircleCollision,
      resolveSoftNodeVsSoftEdgeCollision,
      applyBounceBoundary,
      wgslOffload: {
        enabled: Boolean(sim?.device),
        device: sim?.device,
        modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
        state: (sim.collisionBoundaryWgslState ||= {}),
        rigidSoft: {
          enabled: Boolean(sim?.device),
          device: sim?.device,
          modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
          state: (sim.rigidSoftCollisionWgslState ||= {}),
        },
        softSoft: {
          enabled: Boolean(sim?.device),
          device: sim?.device,
          modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
          state: (sim.softCollisionWgslState ||= {}),
        },
        boundary: {
          enabled: Boolean(sim?.device),
          device: sim?.device,
          modeProfile: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
          state: (sim.collisionBoundaryWgslState ||= {}),
        },
      },
    });
    sim.collisionBoundaryRuntime = collisionResult.boundaryRuntime || { mode: 'cpu-fallback', reason: 'unknown' };
    rigidRigidBroadphaseRuntime.enabled = false;
    rigidRigidBroadphaseRuntime.reason = 'gpu-only-collision-path';
    rigidSoftBroadphaseRuntime.enabled = false;
    rigidSoftBroadphaseRuntime.reason = 'gpu-only-collision-path';
    collisionCpuRuntime.enabled = false;
    collisionCpuRuntime.reason = 'gpu-only-collision-path';
  } else {
    sim.collisionBoundaryRuntime = { mode: 'cpu-baseline', reason: 'baseline-path' };
    const rigidRigidReuseCache = (sim.rigidRigidBroadphaseReuseCache ||= {
      lastBodyCount: -1,
      lastCellSize: 0,
      lastCellSpans: null,
      lastPairs: null,
      lastStats: null,
    });
    const rigidWorldPolyReuseCache = (sim.rigidWorldPolyReuseCache ||= {
      entries: null,
    });

    // Body-body collisions: rigid↔rigid, rigid↔soft, soft↔soft
    for (let iter = 0; iter < 2; iter++) {
      const rigidRigidPreStartMs = performance.now();
      const preSceneBuildStartMs = performance.now();
      const phaseScene = buildCollisionPhaseSceneCache(
        bodies.rigid,
        s.nodes,
        s.springs,
        {
          cellSize: RIGID_RIGID_SPATIAL_HASH_CELL_SIZE,
          edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
          nodePad: 0.8,
          edgePad: 0.8,
          includeRigidRigid: true,
          includeRigidSoft: true,
          rigidRigidReuseCache,
          rigidRigidCandidateBackend: sim?.rigidRigidCandidateBackend || null,
        },
      );
      collisionCpuRuntime.substageMs.candidateBuildPre += performance.now() - preSceneBuildStartMs;

      const preBroadphase = phaseScene.rigidRigid;
      if (!rigidRigidBroadphaseRuntime.backendLabel && preBroadphase?.stats?.backend) {
        rigidRigidBroadphaseRuntime.backendLabel = String(preBroadphase.stats.backend);
      }
      rigidRigidBroadphaseRuntime.phases.push({
        phase: 'pre-soft',
        iter,
        ...preBroadphase.stats,
      });
      rigidRigidBroadphaseRuntime.totalBruteForcePairs += Number(preBroadphase.stats?.bruteForcePairs) || 0;
      rigidRigidBroadphaseRuntime.totalCheckedPairs += Number(preBroadphase.stats?.checkedPairs) || 0;
      rigidRigidBroadphaseRuntime.totalPrunedPairs += Number(preBroadphase.stats?.prunedPairs) || 0;
      rigidRigidBroadphaseRuntime.reuseAttempts += 1;
      rigidRigidBroadphaseRuntime.reuseHits += preBroadphase.stats?.reuseHit === true ? 1 : 0;
      rigidRigidBroadphaseRuntime.reuseMisses += preBroadphase.stats?.reuseHit === true ? 0 : 1;
      rigidRigidBroadphaseRuntime.dedupeMs += Number(preBroadphase.stats?.dedupeMs) || 0;
      rigidRigidBroadphaseRuntime.sortMs += Number(preBroadphase.stats?.sortMs) || 0;
      rigidRigidBroadphaseRuntime.totalBuildMs += Number(preBroadphase.stats?.totalBuildMs) || 0;
      rigidRigidBroadphaseRuntime.jsMarshalMs += Number(preBroadphase.stats?.jsMarshalMs) || 0;
      rigidRigidBroadphaseRuntime.backendComputeMs += Number(preBroadphase.stats?.backendComputeMs) || 0;
      rigidRigidBroadphaseRuntime.emitAttempts += Number(preBroadphase.stats?.emitAttempts) || 0;
      rigidRigidBroadphaseRuntime.duplicatesRejected += Number(preBroadphase.stats?.duplicatesRejected) || 0;
      rigidRigidBroadphaseRuntime.pairsOut += Number(preBroadphase.stats?.pairsOut) || 0;
      const preRigidNarrowphaseStartMs = performance.now();
      const preRigidMeta = Array.isArray(phaseScene?.rigidMeta) ? phaseScene.rigidMeta : [];
      const getPreRigidWorldPolys = createRigidWorldPolyPhaseCache(bodies.rigid, collisionCpuRuntime, rigidWorldPolyReuseCache);
      for (const [i, j] of preBroadphase.pairs) {
        collisionCpuRuntime.rigidRigidPairChecks += 1;

        const broadMetaA = preRigidMeta[i];
        const broadMetaB = preRigidMeta[j];
        if (broadMetaA && broadMetaB) {
          collisionCpuRuntime.rigidRigidEarlyRejectChecks += 1;
          const dx = (Number(broadMetaB.bx) || 0) - (Number(broadMetaA.bx) || 0);
          const dy = (Number(broadMetaB.by) || 0) - (Number(broadMetaA.by) || 0);
          const radiusA = Math.max(0.5, Number(broadMetaA.radius) || 0.5);
          const radiusB = Math.max(0.5, Number(broadMetaB.radius) || 0.5);
          const reach = radiusA + radiusB + 0.32;
          if ((dx * dx + dy * dy) > (reach * reach)) {
            collisionCpuRuntime.rigidRigidEarlyRadiusRejected += 1;
            continue;
          }
        }

        const cacheA = getPreRigidWorldPolys(i);
        const cacheB = getPreRigidWorldPolys(j);
        const hit = resolveRigidVsRigidPolygonCollision(bodies.rigid[i], bodies.rigid[j], 0.32, {
          contacts: rigidContactDebug,
          aIndex: i,
          bIndex: j,
          iter,
          phase: 'pre-soft',
        }, {
          polysA: cacheA.polys,
          polysB: cacheB.polys,
          metaA: cacheA.meta,
          metaB: cacheB.meta,
          stats: collisionCpuRuntime,
        });
        if (hit) collisionCpuRuntime.rigidRigidPairHits += 1;
      }
      collisionCpuRuntime.substageMs.rigidRigidNarrowphasePre += performance.now() - preRigidNarrowphaseStartMs;
      collisionCpuRuntime.stageMs.rigidRigidPre += performance.now() - rigidRigidPreStartMs;

      const rigidSoftStartMs = performance.now();
      const rigidSoftBroadphase = phaseScene.rigidSoft;
      rigidSoftBroadphaseRuntime.phases.push({
        iter,
        ...rigidSoftBroadphase.stats,
      });
      rigidSoftBroadphaseRuntime.totalRawNodeChecks += Number(rigidSoftBroadphase.stats?.rawNodeChecks) || 0;
      rigidSoftBroadphaseRuntime.totalCandidateNodeChecks += Number(rigidSoftBroadphase.stats?.candidateNodeChecks) || 0;
      rigidSoftBroadphaseRuntime.totalPrunedNodeChecks += Number(rigidSoftBroadphase.stats?.prunedNodeChecks) || 0;
      rigidSoftBroadphaseRuntime.totalRawEdgeChecks += Number(rigidSoftBroadphase.stats?.rawEdgeChecks) || 0;
      rigidSoftBroadphaseRuntime.totalCandidateEdgeChecks += Number(rigidSoftBroadphase.stats?.candidateEdgeChecks) || 0;
      rigidSoftBroadphaseRuntime.totalPrunedEdgeChecks += Number(rigidSoftBroadphase.stats?.prunedEdgeChecks) || 0;
      rigidSoftBroadphaseRuntime.totalBlockedEdges += Number(rigidSoftBroadphase.stats?.blockedEdgeCount) || 0;
      collisionCpuRuntime.rigidSoftNodeRawChecks += Number(rigidSoftBroadphase.stats?.rawNodeChecks) || 0;
      collisionCpuRuntime.rigidSoftEdgeRawChecks += Number(rigidSoftBroadphase.stats?.rawEdgeChecks) || 0;
      const iterRigidSoftNodeHitsBefore = collisionCpuRuntime.rigidSoftNodeHits;
      const iterRigidSoftEdgeHitsBefore = collisionCpuRuntime.rigidSoftEdgeHits;

      const rigidSoftNarrowphaseStartMs = performance.now();
      for (let rbi = 0; rbi < bodies.rigid.length; rbi++) {
        const rb = bodies.rigid[rbi];
        // Build per-rigid cache once per collision iteration so node contacts
        // don't repeatedly sanitize/recompute world polygon edge normals.
        const rigidSoftNodeCache = buildRigidSoftNodeCollisionCache(rb, null);

        const nodeCandidates = rigidSoftBroadphase.nodeCandidatesByRigid?.[rbi] || [];
        collisionCpuRuntime.rigidSoftNodeCandidateChecks += nodeCandidates.length;
        for (const ni of nodeCandidates) {
          const sn = s.nodes[ni];
          if (!sn) continue;
          collisionCpuRuntime.rigidSoftNodeChecks += 1;
          if (resolveRigidVsSoftNodeCollision(rb, sn, rigidSoftNodeCache, 0.18)) {
            collisionCpuRuntime.rigidSoftNodeHits += 1;
          }
        }

        const edgeCandidates = rigidSoftBroadphase.edgeCandidatesByRigid?.[rbi] || [];
        collisionCpuRuntime.rigidSoftEdgeCandidateChecks += edgeCandidates.length;
        for (const si of edgeCandidates) {
          const sp = s.springs?.[si];
          if (!Array.isArray(sp)) continue;
          const i = Number(sp[0]) | 0;
          const j = Number(sp[1]) | 0;
          const edgeBodyMode = Number(sp[3]);
          if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
          if (i < 0 || j < 0 || i >= s.nodes.length || j >= s.nodes.length) continue;
          collisionCpuRuntime.rigidSoftEdgeChecks += 1;
          if (resolveRigidVsSoftEdgeCollision(rb, s.nodes[i], s.nodes[j], 0.16)) {
            collisionCpuRuntime.rigidSoftEdgeHits += 1;
          }
        }
      }
      collisionCpuRuntime.substageMs.rigidSoftNarrowphase += performance.now() - rigidSoftNarrowphaseStartMs;
      collisionCpuRuntime.stageMs.rigidSoft += performance.now() - rigidSoftStartMs;

      const softSoftNodeStartMs = performance.now();
      for (let i = 0; i < s.nodes.length; i++) {
        for (let j = i + 1; j < s.nodes.length; j++) {
          collisionCpuRuntime.softSoftNodeChecks += 1;
          if (resolveCircleCollision(s.nodes[i], s.nodes[j], 0.22)) {
            collisionCpuRuntime.softSoftNodeHits += 1;
          }
        }
      }
      collisionCpuRuntime.substageMs.softSoftNodeNarrowphase += performance.now() - softSoftNodeStartMs;
      collisionCpuRuntime.stageMs.softSoftNode += performance.now() - softSoftNodeStartMs;

      // Soft-node vs foreign soft-edge blocking for solid edges.
      const softSoftEdgeStartMs = performance.now();
      for (let ni = 0; ni < s.nodes.length; ni++) {
        const node = s.nodes[ni];
        for (const [i, j, _rest, edgeBodyMode] of s.springs) {
          if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
          if (i === ni || j === ni) continue;
          const a = s.nodes[i], b = s.nodes[j];
          if (a.clusterId === node.clusterId && b.clusterId === node.clusterId) continue;
          collisionCpuRuntime.softSoftEdgeChecks += 1;
          if (resolveSoftNodeVsSoftEdgeCollision(node, a, b, 0.12)) {
            collisionCpuRuntime.softSoftEdgeHits += 1;
          }
        }
      }
      collisionCpuRuntime.substageMs.softSoftEdgeNarrowphase += performance.now() - softSoftEdgeStartMs;
      collisionCpuRuntime.stageMs.softSoftEdge += performance.now() - softSoftEdgeStartMs;

      // Re-run rigid-rigid contacts after rigid-soft pushes to avoid late interpenetration.
      const shouldRunRigidRigidPost =
        (collisionCpuRuntime.rigidSoftNodeHits > iterRigidSoftNodeHitsBefore)
        || (collisionCpuRuntime.rigidSoftEdgeHits > iterRigidSoftEdgeHitsBefore);

      if (shouldRunRigidRigidPost) {
        const rigidRigidPostStartMs = performance.now();
        const postSceneBuildStartMs = performance.now();
        const postScene = buildCollisionPhaseSceneCache(
          bodies.rigid,
          s.nodes,
          s.springs,
          {
            cellSize: RIGID_RIGID_SPATIAL_HASH_CELL_SIZE,
            edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
            includeRigidRigid: true,
            includeRigidSoft: false,
            rigidRigidReuseCache,
            rigidRigidCandidateBackend: sim?.rigidRigidCandidateBackend || null,
          },
        );
        collisionCpuRuntime.substageMs.candidateBuildPost += performance.now() - postSceneBuildStartMs;
        const postBroadphase = postScene.rigidRigid;
        if (!rigidRigidBroadphaseRuntime.backendLabel && postBroadphase?.stats?.backend) {
          rigidRigidBroadphaseRuntime.backendLabel = String(postBroadphase.stats.backend);
        }
        rigidRigidBroadphaseRuntime.phases.push({
          phase: 'post-soft',
          iter,
          ...postBroadphase.stats,
        });
        rigidRigidBroadphaseRuntime.totalBruteForcePairs += Number(postBroadphase.stats?.bruteForcePairs) || 0;
        rigidRigidBroadphaseRuntime.totalCheckedPairs += Number(postBroadphase.stats?.checkedPairs) || 0;
        rigidRigidBroadphaseRuntime.totalPrunedPairs += Number(postBroadphase.stats?.prunedPairs) || 0;
        rigidRigidBroadphaseRuntime.reuseAttempts += 1;
        rigidRigidBroadphaseRuntime.reuseHits += postBroadphase.stats?.reuseHit === true ? 1 : 0;
        rigidRigidBroadphaseRuntime.reuseMisses += postBroadphase.stats?.reuseHit === true ? 0 : 1;
        rigidRigidBroadphaseRuntime.dedupeMs += Number(postBroadphase.stats?.dedupeMs) || 0;
        rigidRigidBroadphaseRuntime.sortMs += Number(postBroadphase.stats?.sortMs) || 0;
        rigidRigidBroadphaseRuntime.totalBuildMs += Number(postBroadphase.stats?.totalBuildMs) || 0;
        rigidRigidBroadphaseRuntime.jsMarshalMs += Number(postBroadphase.stats?.jsMarshalMs) || 0;
        rigidRigidBroadphaseRuntime.backendComputeMs += Number(postBroadphase.stats?.backendComputeMs) || 0;
        rigidRigidBroadphaseRuntime.emitAttempts += Number(postBroadphase.stats?.emitAttempts) || 0;
        rigidRigidBroadphaseRuntime.duplicatesRejected += Number(postBroadphase.stats?.duplicatesRejected) || 0;
        rigidRigidBroadphaseRuntime.pairsOut += Number(postBroadphase.stats?.pairsOut) || 0;

        const postRigidMeta = Array.isArray(postScene?.rigidMeta) ? postScene.rigidMeta : [];
        const postRigidNarrowphaseStartMs = performance.now();
        const getPostRigidWorldPolys = createRigidWorldPolyPhaseCache(bodies.rigid, collisionCpuRuntime, rigidWorldPolyReuseCache);
        for (const [i, j] of postBroadphase.pairs) {
          collisionCpuRuntime.rigidRigidPairChecks += 1;

          const broadMetaA = postRigidMeta[i];
          const broadMetaB = postRigidMeta[j];
          if (broadMetaA && broadMetaB) {
            collisionCpuRuntime.rigidRigidEarlyRejectChecks += 1;
            const dx = (Number(broadMetaB.bx) || 0) - (Number(broadMetaA.bx) || 0);
            const dy = (Number(broadMetaB.by) || 0) - (Number(broadMetaA.by) || 0);
            const radiusA = Math.max(0.5, Number(broadMetaA.radius) || 0.5);
            const radiusB = Math.max(0.5, Number(broadMetaB.radius) || 0.5);
            const reach = radiusA + radiusB + 0.32;
            if ((dx * dx + dy * dy) > (reach * reach)) {
              collisionCpuRuntime.rigidRigidEarlyRadiusRejected += 1;
              continue;
            }
          }

          const cacheA = getPostRigidWorldPolys(i);
          const cacheB = getPostRigidWorldPolys(j);
          const hit = resolveRigidVsRigidPolygonCollision(bodies.rigid[i], bodies.rigid[j], 0.32, {
            contacts: rigidContactDebug,
            aIndex: i,
            bIndex: j,
            iter,
            phase: 'post-soft',
          }, {
            polysA: cacheA.polys,
            polysB: cacheB.polys,
            metaA: cacheA.meta,
            metaB: cacheB.meta,
            stats: collisionCpuRuntime,
          });
          if (hit) collisionCpuRuntime.rigidRigidPairHits += 1;
        }
        collisionCpuRuntime.substageMs.rigidRigidNarrowphasePost += performance.now() - postRigidNarrowphaseStartMs;
        collisionCpuRuntime.stageMs.rigidRigidPost += performance.now() - rigidRigidPostStartMs;
      } else {
        collisionCpuRuntime.rigidRigidPostSkipped += 1;
        rigidRigidBroadphaseRuntime.postPassSkipped += 1;
      }

      const boundaryApplyStartMs = performance.now();
      for (const rb of bodies.rigid) applyBounceBoundary(rb, worldSize, 0.84);
      for (const sn of s.nodes) applyBounceBoundary(sn, worldSize, 0.78);
      collisionCpuRuntime.substageMs.boundaryApply += performance.now() - boundaryApplyStartMs;
    }
  }
  rigidRigidBroadphaseRuntime.reductionPct = rigidRigidBroadphaseRuntime.totalBruteForcePairs > 0
    ? (rigidRigidBroadphaseRuntime.totalPrunedPairs / rigidRigidBroadphaseRuntime.totalBruteForcePairs) * 100
    : 0;
  rigidRigidBroadphaseRuntime.reuseHitRatePct = rigidRigidBroadphaseRuntime.reuseAttempts > 0
    ? (rigidRigidBroadphaseRuntime.reuseHits / rigidRigidBroadphaseRuntime.reuseAttempts) * 100
    : 0;
  rigidSoftBroadphaseRuntime.nodeReductionPct = rigidSoftBroadphaseRuntime.totalRawNodeChecks > 0
    ? (rigidSoftBroadphaseRuntime.totalPrunedNodeChecks / rigidSoftBroadphaseRuntime.totalRawNodeChecks) * 100
    : 0;
  rigidSoftBroadphaseRuntime.edgeReductionPct = rigidSoftBroadphaseRuntime.totalRawEdgeChecks > 0
    ? (rigidSoftBroadphaseRuntime.totalPrunedEdgeChecks / rigidSoftBroadphaseRuntime.totalRawEdgeChecks) * 100
    : 0;

  collisionCpuRuntime.rigidRigidPairHitRatePct = collisionCpuRuntime.rigidRigidPairChecks > 0
    ? (collisionCpuRuntime.rigidRigidPairHits / collisionCpuRuntime.rigidRigidPairChecks) * 100
    : 0;
  collisionCpuRuntime.rigidRigidPrefilterRejected =
    (Number(collisionCpuRuntime.rigidRigidAabbRejected) || 0)
    + (Number(collisionCpuRuntime.rigidRigidRadiusRejected) || 0)
    + (Number(collisionCpuRuntime.rigidRigidEarlyRadiusRejected) || 0);
  collisionCpuRuntime.rigidRigidSatCallRatePct = collisionCpuRuntime.rigidRigidProxyPairChecks > 0
    ? (collisionCpuRuntime.rigidRigidSatCalls / collisionCpuRuntime.rigidRigidProxyPairChecks) * 100
    : 0;
  collisionCpuRuntime.rigidRigidSatHitRatePct = collisionCpuRuntime.rigidRigidSatCalls > 0
    ? (collisionCpuRuntime.rigidRigidSatHits / collisionCpuRuntime.rigidRigidSatCalls) * 100
    : 0;
  collisionCpuRuntime.rigidSoftNodeHitRatePct = collisionCpuRuntime.rigidSoftNodeChecks > 0
    ? (collisionCpuRuntime.rigidSoftNodeHits / collisionCpuRuntime.rigidSoftNodeChecks) * 100
    : 0;
  collisionCpuRuntime.rigidSoftEdgeHitRatePct = collisionCpuRuntime.rigidSoftEdgeChecks > 0
    ? (collisionCpuRuntime.rigidSoftEdgeHits / collisionCpuRuntime.rigidSoftEdgeChecks) * 100
    : 0;
  collisionCpuRuntime.rigidSoftNodeCandidateHitRatePct = collisionCpuRuntime.rigidSoftNodeCandidateChecks > 0
    ? (collisionCpuRuntime.rigidSoftNodeHits / collisionCpuRuntime.rigidSoftNodeCandidateChecks) * 100
    : 0;
  collisionCpuRuntime.rigidSoftEdgeCandidateHitRatePct = collisionCpuRuntime.rigidSoftEdgeCandidateChecks > 0
    ? (collisionCpuRuntime.rigidSoftEdgeHits / collisionCpuRuntime.rigidSoftEdgeCandidateChecks) * 100
    : 0;
  collisionCpuRuntime.softSoftNodeHitRatePct = collisionCpuRuntime.softSoftNodeChecks > 0
    ? (collisionCpuRuntime.softSoftNodeHits / collisionCpuRuntime.softSoftNodeChecks) * 100
    : 0;
  collisionCpuRuntime.softSoftEdgeHitRatePct = collisionCpuRuntime.softSoftEdgeChecks > 0
    ? (collisionCpuRuntime.softSoftEdgeHits / collisionCpuRuntime.softSoftEdgeChecks) * 100
    : 0;
  collisionCpuRuntime.stageMs.total =
    (collisionCpuRuntime.stageMs.rigidRigidPre || 0)
    + (collisionCpuRuntime.stageMs.rigidSoft || 0)
    + (collisionCpuRuntime.stageMs.softSoftNode || 0)
    + (collisionCpuRuntime.stageMs.softSoftEdge || 0)
    + (collisionCpuRuntime.stageMs.rigidRigidPost || 0);

  collisionCpuRuntime.substageMs.candidateBuildTotal =
    (collisionCpuRuntime.substageMs.candidateBuildPre || 0)
    + (collisionCpuRuntime.substageMs.candidateBuildPost || 0);
  collisionCpuRuntime.substageMs.narrowphaseTotal =
    (collisionCpuRuntime.substageMs.rigidRigidNarrowphasePre || 0)
    + (collisionCpuRuntime.substageMs.rigidSoftNarrowphase || 0)
    + (collisionCpuRuntime.substageMs.softSoftNodeNarrowphase || 0)
    + (collisionCpuRuntime.substageMs.softSoftEdgeNarrowphase || 0)
    + (collisionCpuRuntime.substageMs.rigidRigidNarrowphasePost || 0);
  collisionCpuRuntime.substageMs.applyTotal = collisionCpuRuntime.substageMs.boundaryApply || 0;
  collisionCpuRuntime.substageMs.total =
    (collisionCpuRuntime.substageMs.candidateBuildTotal || 0)
    + (collisionCpuRuntime.substageMs.narrowphaseTotal || 0)
    + (collisionCpuRuntime.substageMs.applyTotal || 0);

  collisionCpuRuntime.polyCacheHitRatePct = collisionCpuRuntime.polyCacheLookups > 0
    ? (collisionCpuRuntime.polyCacheHits / collisionCpuRuntime.polyCacheLookups) * 100
    : 0;

  sim.rigidRigidBroadphaseRuntime = rigidRigidBroadphaseRuntime;
  sim.rigidSoftBroadphaseRuntime = rigidSoftBroadphaseRuntime;
  sim.collisionCpuRuntime = collisionCpuRuntime;

  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.collisions.gpuOnly' : 'bodies.collisions.baseline', performance.now() - collisionStageStartMs);

  // After collision impulses, optionally redistribute linear/angular momentum
  // across each soft cluster and run inside-correction recovery passes.
  let rigidInsideCorrections = 0;
  let membraneInsideCorrections = 0;
  const postCollisionRecoveryStageStartMs = performance.now();
  const postCollisionRecoveryOn = sim.controls?.enablePostCollisionRecovery !== false;
  sim.postCollisionProjectionRuntime = {
    mode: 'inactive',
    reason: postCollisionRecoveryOn ? 'pending' : 'disabled',
    projectedNodes: 0,
    meanDelta: 0,
    jsMarshalMs: 0,
    backendComputeMs: 0,
    totalMs: 0,
  };
  if (postCollisionRecoveryOn) {
    if (solverPath === 'gpu-only') {
      const postCollisionRecovery = await applyPostCollisionRecoveryGpuOnly({
        sim,
        bodies,
        soft: s,
        dtNorm,
        softMembraneClusterSet,
        softClusterCollisionLinearProjection: SOFT_CLUSTER_COLLISION_LINEAR_PROJECTION,
        softClusterCollisionAngularProjection,
        membraneGainScale: 0.72,
        applyBounceBoundary,
        applyCollisionBoundaryPassGpuOnly,
        wgslOffload: {
          enabled: true,
          device: sim?.device,
          authoritativeSoftClusterMass: (() => {
            const st = (sim.postCollisionBoundaryWgslState ||= {});
            if (st.enableAuthoritativeSoftClusterMass !== false) {
              st.enableAuthoritativeSoftClusterMass = true;
            }
            return st.enableAuthoritativeSoftClusterMass === true;
          })(),
          state: (() => {
            const st = (sim.postCollisionBoundaryWgslState ||= {});
            if (st.enableAuthoritativeSoftClusterProjection !== false) {
              st.enableAuthoritativeSoftClusterProjection = true;
            }
            if (st.enableAuthoritativeMembraneInsideCorrection !== false) {
              st.enableAuthoritativeMembraneInsideCorrection = true;
            }
            return st;
          })(),
        },
        n: worldSize,
        softClusterLoops,
      });
      rigidInsideCorrections = postCollisionRecovery.rigidInsideCorrections;
      membraneInsideCorrections = postCollisionRecovery.membraneInsideCorrections;
      sim.postCollisionProjectionRuntime = {
        mode: 'gpu-only-path',
        reason: 'handled-in-gpu-only-recovery',
        projectedNodes: 0,
        meanDelta: 0,
        jsMarshalMs: 0,
        backendComputeMs: 0,
        totalMs: 0,
      };
    } else {
      const postCollisionClusterKinematics = computeSoftClusterKinematics(s.nodes);
      const projectionLinearGain = SOFT_CLUSTER_COLLISION_LINEAR_PROJECTION * dtNorm;
      const projectionAngularGain = softClusterCollisionAngularProjection * dtNorm;

      if (sim?.postCollisionProjectionBackend) {
        try {
          const wasmProjection = sim.postCollisionProjectionBackend.project({
            nodes: s.nodes,
            clusterKinematics: postCollisionClusterKinematics,
            linearGain: projectionLinearGain,
            angularGain: projectionAngularGain,
            membraneClusterSet: softMembraneClusterSet,
            membraneGainScale: 0.72,
          });
          if (wasmProjection?.ok === false) {
            throw new Error(String(wasmProjection?.reason || 'projection-rejected'));
          }
          sim.postCollisionProjectionRuntime = {
            mode: 'wasm',
            reason: sim?.postCollisionProjectionBackend?.label || 'wasm',
            projectedNodes: Number(wasmProjection?.projectedNodes) || 0,
            meanDelta: Number(wasmProjection?.meanDelta) || 0,
            jsMarshalMs: Number(wasmProjection?.jsMarshalMs) || 0,
            backendComputeMs: Number(wasmProjection?.backendComputeMs) || 0,
            totalMs: Number(wasmProjection?.totalMs) || 0,
          };
        } catch (err) {
          const fallbackProjection = projectNodesTowardClusterRigidMotion(s.nodes, postCollisionClusterKinematics, {
            linearGain: projectionLinearGain,
            angularGain: projectionAngularGain,
            membraneClusterSet: softMembraneClusterSet,
            membraneGainScale: 0.72,
          });
          sim.postCollisionProjectionRuntime = {
            mode: 'cpu-fallback',
            reason: `wasm-error:${String(err?.message || err)}`,
            projectedNodes: Number(fallbackProjection?.projectedNodes) || 0,
            meanDelta: Number(fallbackProjection?.meanDelta) || 0,
            jsMarshalMs: 0,
            backendComputeMs: 0,
            totalMs: 0,
          };
        }
      } else {
        const fallbackProjection = projectNodesTowardClusterRigidMotion(s.nodes, postCollisionClusterKinematics, {
          linearGain: projectionLinearGain,
          angularGain: projectionAngularGain,
          membraneClusterSet: softMembraneClusterSet,
          membraneGainScale: 0.72,
        });
        sim.postCollisionProjectionRuntime = {
          mode: 'cpu-baseline',
          reason: 'backend-disabled',
          projectedNodes: Number(fallbackProjection?.projectedNodes) || 0,
          meanDelta: Number(fallbackProjection?.meanDelta) || 0,
          jsMarshalMs: 0,
          backendComputeMs: 0,
          totalMs: 0,
        };
      }

      rigidInsideCorrections = applyRigidInsideCorrectionPass(bodies, s);
      membraneInsideCorrections = applyMembraneInsideCorrectionPass(sim, s, softClusterLoops);
      for (const rb of bodies.rigid) applyBounceBoundary(rb, worldSize, 0.84);
      for (const sn of s.nodes) applyBounceBoundary(sn, worldSize, 0.78);
    }
  }

  recordPipelineTiming(sim, postCollisionRecoveryOn
    ? (solverPath === 'gpu-only' ? 'bodies.postCollisionRecovery.gpuOnly' : 'bodies.postCollisionRecovery.baseline')
    : 'bodies.postCollisionRecovery.disabled', performance.now() - postCollisionRecoveryStageStartMs);

  sim.lastRigidContacts = rigidContactDebug.length > 64 ? rigidContactDebug.slice(0, 64) : rigidContactDebug;

  const warningInterventionsOn = sim.controls?.enableWarningDeformInterventions !== false;
  const severeInterventionsOn = sim.controls?.enableSevereDeformInterventions !== false;

  const deformationStageStartMs = performance.now();
  let deform;
  if (solverPath === 'gpu-only') {
    deform = await applySoftDeformationInterventionsGpuOnly({
      sim,
      soft: s,
      softClusterLoops,
      severeInterventionsOn,
      buildSoftDeformationState,
      stabilizeSeverelyDeformedSoftClusters,
      membraneClusterSet: softMembraneClusterSet,
      deformationThresholds: {
        warnStretch: SOFT_DEFORM_WARN_STRETCH,
        severeStretch: SOFT_DEFORM_SEVERE_STRETCH,
        warnAreaMin: SOFT_DEFORM_WARN_AREA_RATIO_MIN,
        warnAreaMax: SOFT_DEFORM_WARN_AREA_RATIO_MAX,
        severeAreaMin: SOFT_DEFORM_SEVERE_AREA_RATIO_MIN,
        severeAreaMax: SOFT_DEFORM_SEVERE_AREA_RATIO_MAX,
        warnPoseRms: SOFT_DEFORM_WARN_POSE_RMS,
        warnPoseMax: SOFT_DEFORM_WARN_POSE_MAX,
        severePoseRms: SOFT_DEFORM_SEVERE_POSE_RMS,
        severePoseMax: SOFT_DEFORM_SEVERE_POSE_MAX,
      },
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        state: (sim.softDeformationWgslState ||= {}),
      },
    });
  } else {
    deform = buildSoftDeformationState(sim, s, softClusterLoops);
    if (severeInterventionsOn && deform.severeCollapseCount > 0) {
      stabilizeSeverelyDeformedSoftClusters(sim, s, softClusterLoops, deform);
      deform = buildSoftDeformationState(sim, s, softClusterLoops);
    }
  }

  const restRecoveryStageStartMs = performance.now();
  if (softSpringRestRecoveryOn && sim.softSpringRestBaseline && sim.softSpringRestBaseline.length === s.springs.length) {
    if (solverPath === 'gpu-only') {
      applySoftRestRecoveryGpuOnly({
        springs: s.springs,
        softNodes: s.nodes,
        restBaseline: sim.softSpringRestBaseline,
        severeInterventionsOn,
        warningInterventionsOn,
        deform,
        recoverSoftSpringRests,
        wgslOffload: {
          enabled: true,
          device: sim?.device,
          state: (sim.softRestRecoveryWgslState ||= {}),
        },
      });
    } else {
      const severeProfile = severeInterventionsOn && deform.severeCollapseCount > 0;
      const warningProfile = warningInterventionsOn && !severeProfile && deform.warningCount > 0;
      const profile = severeProfile ? 'severe' : (warningProfile ? 'warning' : 'baseline');
      const pick = (baseline, warning, severe) => (profile === 'baseline' ? baseline : (profile === 'warning' ? warning : severe));

      recoverSoftSpringRests(s.springs, sim.softSpringRestBaseline, {
        recoverRate: pick(0.056, 0.036, 0.015),
        hardMinFactor: 0.7,
        hardMaxFactor: 1.45,
        jitterDeadband: 1e-5,
        adaptiveGainMax: pick(2.4, 1.85, 1.35),
        adaptiveExponent: pick(0.78, 0.9, 1.0),
        elongationBiasMax: pick(1.22, 1.14, 1.08),
        compressionBiasMax: pick(1.12, 1.08, 1.04),
        errorPivot: 0.16,
        outlierRecoveryCouplingMax: pick(1.22, 1.12, 1.05),
        outlierErrorPivot: pick(0.75, 0.85, 0.95),
        localEndpointCouplingMax: pick(1.16, 1.08, 1.03),
        localDirectionalCouplingMax: pick(1.1, 1.06, 1.02),
        localImbalanceCouplingMax: pick(1.12, 1.07, 1.02),
        localConsensusCouplingMax: pick(1.08, 1.04, 1.01),
        localOutlierCouplingMax: pick(1.12, 1.07, 1.02),
        localOutlierErrorPivot: pick(0.85, 0.95, 1.1),
        localErrorPivot: pick(0.2, 0.25, 0.3),
        counterPolarityCouplingMax: pick(1.09, 1.05, 1.02),
        smallRestRecoveryCouplingMax: pick(1.14, 1.08, 1.0),
        smallRestPivot: pick(0.9, 1.05, 1.2),
        lowErrorRecoveryCouplingMax: pick(1.12, 1.06, 1.0),
        lowErrorRecoveryGate: pick(0.08, 0.065, 0.05),
      });
    }
  }
  recordPipelineTiming(sim,
    (softSpringRestRecoveryOn && sim.softSpringRestBaseline && sim.softSpringRestBaseline.length === s.springs.length)
      ? (solverPath === 'gpu-only' ? 'bodies.softRestRecovery.gpuOnly' : 'bodies.softRestRecovery.baseline')
      : 'bodies.softRestRecovery.disabled',
    performance.now() - restRecoveryStageStartMs,
  );

  sim.softDeformationState = deform;
  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.softDeformation.gpuOnly' : 'bodies.softDeformation.baseline', performance.now() - deformationStageStartMs);

  const previousSevere = sim.softDeformationPrevSevereSet || new Set();
  const newlySevere = deform.severeClusters.filter((cid) => !previousSevere.has(cid));
  if (newlySevere.length > 0) {
    sim.softDeformationEvents = sim.softDeformationEvents || [];
    for (const cid of newlySevere) {
      const c = deform.clusters.find((x) => x.clusterId === cid);
      const evt = {
        frame: sim.frame,
        clusterId: cid,
        poseErrorRms: +((c?.poseErrorRms || 0).toFixed(3)),
        poseErrorMax: +((c?.poseErrorMax || 0).toFixed(3)),
        stretchMax: +((c?.stretchMax || 0).toFixed(3)),
        areaRatio: +((c?.areaRatio || 0).toFixed(3)),
      };
      sim.softDeformationEvents.push(evt);
      console.warn('[gpu-lab] soft cluster severe deformation', evt);
    }
    if (sim.softDeformationEvents.length > 24) {
      sim.softDeformationEvents.splice(0, sim.softDeformationEvents.length - 24);
    }
  }
  sim.softDeformationPrevSevereSet = new Set(deform.severeClusters);

  // Re-apply scripted motion after collisions/integration so pinned/circle
  // bodies remain exact confrontation fixtures frame-to-frame.
  applyInteractionLabBodyMotion(sim);

  let injectedMomentum = 0;
  let softClusterForInjection;
  const bodyFluidInjectionStageStartMs = performance.now();
  if (solverPath === 'gpu-only') {
    const fluidInjectionResult = await applyBodyFluidInjectionGpuOnly({
      sim,
      bodies,
      soft: s,
      n,
      vxField,
      vyField,
      feedbackK,
      swimGain,
      bodyFeedbackCurrVx,
      bodyFeedbackCurrVy,
      rigidEdgeMomentumScale,
      softNodeMomentumScale,
      sampleFieldBilinear,
      computeSoftClusterKinematics,
      softClusterFluidInjectBlend: SOFT_CLUSTER_FLUID_INJECT_BLEND,
      fluidCouplingComponentLimit: sim?.controls?.fluidCouplingComponentLimit,
      wgslOffload: {
        enabled: true,
        device: sim?.device,
        state: (sim.bodyFluidInjectionWgslState ||= {}),
      },
    });
    injectedMomentum = fluidInjectionResult.injectedMomentum || 0;
    softClusterForInjection = fluidInjectionResult.softClusterForInjection || computeSoftClusterKinematics(s.nodes);
  } else {
    const injectPoint = (px, py, pvx, pvy, localFluidX, localFluidY, mass, rad=3.0, swimInjectX = 0, swimInjectY = 0, momentumScale = 1) => {
      if (!Number.isFinite(px) || !Number.isFinite(py)) return;
      const radius = Math.max(0.4, Number.isFinite(rad) ? rad : 3.0);
      const pxGrid = px / worldScale;
      const pyGrid = py / worldScale;
      const radiusGrid = radius / worldScale;
      const minX = Math.max(0, Math.floor(pxGrid - radiusGrid));
      const maxX = Math.min(n - 1, Math.ceil(pxGrid + radiusGrid));
      const minY = Math.max(0, Math.floor(pyGrid - radiusGrid));
      const maxY = Math.min(n - 1, Math.ceil(pyGrid + radiusGrid));
      const couplingLimit = normalizeFluidCouplingComponentLimit(sim?.controls?.fluidCouplingComponentLimit);
      const relXRaw = pvx - localFluidX + swimInjectX;
      const relYRaw = pvy - localFluidY + swimInjectY;
      const relX = clampFluidComponent(relXRaw, couplingLimit);
      const relY = clampFluidComponent(relYRaw, couplingLimit);
      const scaleRaw = feedbackK * Math.max(0.1, Number.isFinite(mass) ? mass : 0.1) * clamp(Number(momentumScale), 0, 1);
      const scale = Number.isFinite(scaleRaw) ? clamp(scaleRaw, 0, couplingLimit) : 0;
      if (scale <= 0) return;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const dx = x - pxGrid, dy = y - pyGrid;
          const d = Math.hypot(dx, dy);
          if (d > radiusGrid) continue;
          const w = 1 - d / radiusGrid;
          const idx = y * n + x;
          const jxRaw = relX * scale * w;
          const jyRaw = relY * scale * w;
          const jx = clampFluidComponent(jxRaw, couplingLimit);
          const jy = clampFluidComponent(jyRaw, couplingLimit);
          const nextVx = Number(vxField[idx]) + jx;
          const nextVy = Number(vyField[idx]) + jy;
          vxField[idx] = clampFluidComponent(nextVx, couplingLimit);
          vyField[idx] = clampFluidComponent(nextVy, couplingLimit);
          bodyFeedbackCurrVx[idx] += jx;
          bodyFeedbackCurrVy[idx] += jy;
          injectedMomentum += Math.hypot(jx, jy);
        }
      }
    };

    for (let bi = 0; bi < bodies.rigid.length; bi++) {
      const b = bodies.rigid[bi];
      const fx = sampleFieldBilinear(vxField, n, b.x, b.y, worldScale);
      const fy = sampleFieldBilinear(vyField, n, b.x, b.y, worldScale);
      const swimPhase = sim.frame * 0.08 + bi * 2.1;
      injectPoint(
        b.x,
        b.y,
        b.vx,
        b.vy,
        fx,
        fy,
        b.mass,
        b.r * 0.8,
        swimGain * Math.cos(swimPhase) * 0.015,
        swimGain * Math.sin(swimPhase) * 0.012,
        rigidEdgeMomentumScale(b, allowPassEdgeFlowPush),
      );
    }
    softClusterForInjection = computeSoftClusterKinematics(s.nodes);
    for (let i = 0; i < s.nodes.length; i++) {
      const node = s.nodes[i];
      const fx = sampleFieldBilinear(vxField, n, node.x, node.y, worldScale);
      const fy = sampleFieldBilinear(vyField, n, node.x, node.y, worldScale);
      const cid = node.clusterId ?? 0;
      const c = softClusterForInjection.get(cid);
      const cx = Number.isFinite(Number(c?.x)) ? Number(c.x) : node.x;
      const cy = Number.isFinite(Number(c?.y)) ? Number(c.y) : node.y;
      const cvx = Number.isFinite(Number(c?.vx)) ? Number(c.vx) : node.vx;
      const cvy = Number.isFinite(Number(c?.vy)) ? Number(c.vy) : node.vy;
      const omega = Number.isFinite(Number(c?.omega)) ? Number(c.omega) : 0;
      const rx = node.x - cx;
      const ry = node.y - cy;
      const rigidLikeVx = cvx - omega * ry;
      const rigidLikeVy = cvy + omega * rx;
      const blend = SOFT_CLUSTER_FLUID_INJECT_BLEND;
      const injectVx = node.vx * (1 - blend) + rigidLikeVx * blend;
      const injectVy = node.vy * (1 - blend) + rigidLikeVy * blend;
      const swimPhase = sim.frame * 0.12 + i * 1.57;
      injectPoint(
        node.x,
        node.y,
        injectVx,
        injectVy,
        fx,
        fy,
        node.mass,
        2.2,
        swimGain * Math.cos(swimPhase) * 0.01,
        swimGain * Math.sin(swimPhase) * 0.01,
        softNodeMomentumScale(i),
      );
    }
  }

  recordPipelineTiming(sim, solverPath === 'gpu-only' ? 'bodies.bodyFluidInjection.gpuOnly' : 'bodies.bodyFluidInjection.baseline', performance.now() - bodyFluidInjectionStageStartMs);

  const softCentroidAfter = computeSoftCentroid(s.nodes);
  const rigidCenterAfter = computeRigidCenter(bodies.rigid);

  let softClusterCount = 0;
  let softClusterInertiaSum = 0;
  let softClusterOmegaAbsSum = 0;
  let softClusterAngularEnergy = 0;
  for (const c of softClusterForInjection.values()) {
    const inertia = Math.max(1e-4, Number(c?.inertia) || 0);
    const omega = Number(c?.omega) || 0;
    softClusterCount += 1;
    softClusterInertiaSum += inertia;
    softClusterOmegaAbsSum += Math.abs(omega);
    softClusterAngularEnergy += 0.5 * inertia * omega * omega;
  }

  const metricsBuildStartMs = performance.now();
  const metrics = {
    rigidCenterDelta: Math.hypot(rigidCenterAfter.x - rigidCenterBefore.x, rigidCenterAfter.y - rigidCenterBefore.y),
    softCentroidDelta: Math.hypot(softCentroidAfter.x - softCentroidBefore.x, softCentroidAfter.y - softCentroidBefore.y),
    rigidCarryTransfer,
    softCarryTransfer,
    softClusterCount,
    softClusterInertiaAvg: softClusterCount > 0 ? (softClusterInertiaSum / softClusterCount) : 0,
    softClusterOmegaAbsAvg: softClusterCount > 0 ? (softClusterOmegaAbsSum / softClusterCount) : 0,
    softClusterAngularEnergy,
    injectedMomentum,
    selfFeedbackSuppression: SELF_FEEDBACK_SUPPRESSION,
    softDeformWarningCount: deform.warningCount || 0,
    softDeformSevereCount: deform.severeCount || 0,
    softDeformSevereCollapseCount: deform.severeCollapseCount || 0,
    softDeformWorstStretch: Number.isFinite(deform.worstStretch) ? deform.worstStretch : 1,
    softDeformWorstAreaRatio: Number.isFinite(deform.worstAreaRatio) ? deform.worstAreaRatio : 1,
    softDeformWorstPoseError: Number.isFinite(deform.worstPoseError) ? deform.worstPoseError : 0,
    membraneCellClusters: Array.isArray(sim?.bodies?.softMembraneClusters) ? sim.bodies.softMembraneClusters.length : 0,
    membraneBoundaryClusters,
    membraneShapeClusters,
    membranePressureClusters,
    rigidInsideCorrections,
    membraneInsideCorrections,
    softDtUsed: dt,
    softDtRaw: dtRaw,
  };
  sim.couplingTelemetry = sim.couplingTelemetry || [];
  sim.couplingTelemetry.push(metrics);
  if (sim.couplingTelemetry.length > 120) sim.couplingTelemetry.shift();
  recordPipelineTiming(sim, 'bodies.metricsPack', performance.now() - metricsBuildStartMs);

  // Ping-pong feedback fields: next frame samples this frame's injected momentum
  // so we can suppress self-induced wake bias during flow->body coupling reads.
  sim._bodyFeedbackPrevVx = bodyFeedbackCurrVx;
  sim._bodyFeedbackPrevVy = bodyFeedbackCurrVy;
  sim._bodyFeedbackCurrVx = bodyFeedbackPrevVx;
  sim._bodyFeedbackCurrVy = bodyFeedbackPrevVy;

  return metrics;
}


function summarizeCouplingTelemetry(telemetry) {
  if (!Array.isArray(telemetry) || telemetry.length === 0) {
    return {
      rigidCenterDeltaAvg: 0,
      softCentroidDeltaAvg: 0,
      rigidCarryTransferAvg: 0,
      softCarryTransferAvg: 0,
      softClusterCountAvg: 0,
      softClusterInertiaAvg: 0,
      softClusterOmegaAbsAvg: 0,
      softClusterAngularEnergyAvg: 0,
      injectedMomentumAvg: 0,
      softDeformWarningCountAvg: 0,
      softDeformSevereCountAvg: 0,
      softDeformSevereCollapseCountAvg: 0,
      softDeformWorstStretchAvg: 1,
      softDeformWorstAreaRatioAvg: 1,
      softDeformWorstPoseErrorAvg: 0,
      rigidInsideCorrectionsAvg: 0,
      membraneInsideCorrectionsAvg: 0,
    };
  }
  const acc = {
    rigidCenterDelta: 0,
    softCentroidDelta: 0,
    rigidCarryTransfer: 0,
    softCarryTransfer: 0,
    softClusterCount: 0,
    softClusterInertiaAvg: 0,
    softClusterOmegaAbsAvg: 0,
    softClusterAngularEnergy: 0,
    injectedMomentum: 0,
    softDeformWarningCount: 0,
    softDeformSevereCount: 0,
    softDeformSevereCollapseCount: 0,
    softDeformWorstStretch: 0,
    softDeformWorstAreaRatio: 0,
    softDeformWorstPoseError: 0,
    rigidInsideCorrections: 0,
    membraneInsideCorrections: 0,
  };
  for (const t of telemetry) {
    acc.rigidCenterDelta += t.rigidCenterDelta || 0;
    acc.softCentroidDelta += t.softCentroidDelta || 0;
    acc.rigidCarryTransfer += t.rigidCarryTransfer || 0;
    acc.softCarryTransfer += t.softCarryTransfer || 0;
    acc.softClusterCount += t.softClusterCount || 0;
    acc.softClusterInertiaAvg += t.softClusterInertiaAvg || 0;
    acc.softClusterOmegaAbsAvg += t.softClusterOmegaAbsAvg || 0;
    acc.softClusterAngularEnergy += t.softClusterAngularEnergy || 0;
    acc.injectedMomentum += t.injectedMomentum || 0;
    acc.softDeformWarningCount += t.softDeformWarningCount || 0;
    acc.softDeformSevereCount += t.softDeformSevereCount || 0;
    acc.softDeformSevereCollapseCount += t.softDeformSevereCollapseCount || 0;
    acc.softDeformWorstStretch += t.softDeformWorstStretch || 1;
    acc.softDeformWorstAreaRatio += t.softDeformWorstAreaRatio || 1;
    acc.softDeformWorstPoseError += t.softDeformWorstPoseError || 0;
    acc.rigidInsideCorrections += t.rigidInsideCorrections || 0;
    acc.membraneInsideCorrections += t.membraneInsideCorrections || 0;
  }
  const k = 1 / telemetry.length;
  return {
    rigidCenterDeltaAvg: +(acc.rigidCenterDelta * k).toFixed(4),
    softCentroidDeltaAvg: +(acc.softCentroidDelta * k).toFixed(4),
    rigidCarryTransferAvg: +(acc.rigidCarryTransfer * k).toFixed(4),
    softCarryTransferAvg: +(acc.softCarryTransfer * k).toFixed(4),
    softClusterCountAvg: +(acc.softClusterCount * k).toFixed(3),
    softClusterInertiaAvg: +(acc.softClusterInertiaAvg * k).toFixed(4),
    softClusterOmegaAbsAvg: +(acc.softClusterOmegaAbsAvg * k).toFixed(4),
    softClusterAngularEnergyAvg: +(acc.softClusterAngularEnergy * k).toFixed(4),
    injectedMomentumAvg: +(acc.injectedMomentum * k).toFixed(4),
    softDeformWarningCountAvg: +(acc.softDeformWarningCount * k).toFixed(3),
    softDeformSevereCountAvg: +(acc.softDeformSevereCount * k).toFixed(3),
    softDeformSevereCollapseCountAvg: +(acc.softDeformSevereCollapseCount * k).toFixed(3),
    softDeformWorstStretchAvg: +(acc.softDeformWorstStretch * k).toFixed(3),
    softDeformWorstAreaRatioAvg: +(acc.softDeformWorstAreaRatio * k).toFixed(3),
    softDeformWorstPoseErrorAvg: +(acc.softDeformWorstPoseError * k).toFixed(3),
    rigidInsideCorrectionsAvg: +(acc.rigidInsideCorrections * k).toFixed(3),
    membraneInsideCorrectionsAvg: +(acc.membraneInsideCorrections * k).toFixed(3),
  };
}

function summarizeSoftEdgePolicies(springs) {
  const out = {
    edgeCount: 0,
    velocityPassEdges: 0,
    velocityBlockEdges: 0,
    dyePassChannels: 0,
    dyeDeflectChannels: 0,
    dyeAbsorbChannels: 0,
    momentumAvg: 1,
    momentumMin: 1,
    momentumMax: 1,
  };
  if (!Array.isArray(springs) || springs.length === 0) return out;

  let momentumSum = 0;
  let momentumCount = 0;
  let momentumMin = 1;
  let momentumMax = 0;

  for (const sp of springs) {
    if (!Array.isArray(sp) || sp.length < 2) continue;
    out.edgeCount += 1;

    const edgeBodyMode = Number(sp[3]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK;
    const edgeVelocityMode = Number(sp[5]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : edgeBodyMode;
    if (edgeVelocityMode === EDGE_BODY_MODE.PASS) out.velocityPassEdges += 1;
    else out.velocityBlockEdges += 1;

    const dye = normalizeEdgeDyeModeRGB(sp[4]);
    for (const d of dye) {
      if (d === EDGE_DYE_MODE.PASS) out.dyePassChannels += 1;
      else if (d === EDGE_DYE_MODE.ABSORB) out.dyeAbsorbChannels += 1;
      else out.dyeDeflectChannels += 1;
    }

    const m = clamp(Number.isFinite(Number(sp[6])) ? Number(sp[6]) : 1, 0, 1);
    momentumSum += m;
    momentumCount += 1;
    momentumMin = Math.min(momentumMin, m);
    momentumMax = Math.max(momentumMax, m);
  }

  if (momentumCount > 0) {
    out.momentumAvg = +(momentumSum / momentumCount).toFixed(3);
    out.momentumMin = +momentumMin.toFixed(3);
    out.momentumMax = +momentumMax.toFixed(3);
  }
  return out;
}

function pointInPolygon(x, y, verts) {
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
    const xi = verts[i].x, yi = verts[i].y;
    const xj = verts[j].x, yj = verts[j].y;
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < ((xj - xi) * (y - yi)) / Math.max(1e-9, (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function normalizeBinaryRGB(v) {
  if (!Array.isArray(v) || v.length < 3) return [0, 0, 0];
  return [Number(v[0]) > 0 ? 1 : 0, Number(v[1]) > 0 ? 1 : 0, Number(v[2]) > 0 ? 1 : 0];
}

function applyDigestiveCapture(sim, r, g, b) {
  const n = sim.controls.n;
  const rigidCapture = 0.055;
  const softCapture = 0.045;
  let captured = 0;

  for (const rb of sim.bodies.rigid) {
    const consumeMask = normalizeBinaryRGB(rb.consumeDyeRGB || (rb.digestEnabled ? [1, 1, 1] : [0, 0, 0]));
    if (consumeMask[0] === 0 && consumeMask[1] === 0 && consumeMask[2] === 0) continue;
    const dig = rb.digestRGB || [1, 1, 1];
    const verts = rigidVerticesWorld(rb);
    let minX = n - 1, minY = n - 1, maxX = 0, maxY = 0;
    for (const v of verts) {
      minX = Math.min(minX, v.x); minY = Math.min(minY, v.y);
      maxX = Math.max(maxX, v.x); maxY = Math.max(maxY, v.y);
    }
    minX = Math.max(0, Math.floor(minX)); minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(n - 1, Math.ceil(maxX)); maxY = Math.min(n - 1, Math.ceil(maxY));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (!pointInPolygon(x + 0.5, y + 0.5, verts)) continue;
        const i = y * n + x;
        const takeR = r[i] * rigidCapture * dig[0] * consumeMask[0];
        const takeG = g[i] * rigidCapture * dig[1] * consumeMask[1];
        const takeB = b[i] * rigidCapture * dig[2] * consumeMask[2];
        r[i] -= takeR; g[i] -= takeG; b[i] -= takeB;
        captured += takeR + takeG + takeB;
      }
    }
  }

  const clusters = new Map();
  for (const node of sim.bodies.soft.nodes) {
    const id = node.clusterId ?? 0;
    if (!clusters.has(id)) clusters.set(id, []);
    clusters.get(id).push(node);
  }
  for (const nodes of clusters.values()) {
    if (nodes.length < 3) continue;
    if (!nodes[0].digestEnabled) continue;
    const dig = nodes[0].digestRGB || [1, 1, 1];
    let cx = 0, cy = 0;
    for (const n0 of nodes) { cx += n0.x; cy += n0.y; }
    cx /= nodes.length; cy /= nodes.length;
    const verts = [...nodes]
      .map((p) => ({ x: p.x, y: p.y, a: Math.atan2(p.y - cy, p.x - cx) }))
      .sort((a, b2) => a.a - b2.a)
      .map(({ x, y }) => ({ x, y }));

    let minX = n - 1, minY = n - 1, maxX = 0, maxY = 0;
    for (const v of verts) {
      minX = Math.min(minX, v.x); minY = Math.min(minY, v.y);
      maxX = Math.max(maxX, v.x); maxY = Math.max(maxY, v.y);
    }
    minX = Math.max(0, Math.floor(minX)); minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(n - 1, Math.ceil(maxX)); maxY = Math.min(n - 1, Math.ceil(maxY));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (!pointInPolygon(x + 0.5, y + 0.5, verts)) continue;
        const i = y * n + x;
        const takeR = r[i] * softCapture * dig[0];
        const takeG = g[i] * softCapture * dig[1];
        const takeB = b[i] * softCapture * dig[2];
        r[i] -= takeR; g[i] -= takeG; b[i] -= takeB;
        captured += takeR + takeG + takeB;
      }
    }
  }

  sim.digestiveCapture = (sim.digestiveCapture || 0) * 0.97 + captured * 0.03;
}

function edgeModeColor(modeRGB, blocked) {
  const m = normalizeEdgeDyeModeRGB(modeRGB);
  const key = `${m[0]}-${m[1]}-${m[2]}`;
  if (!blocked) return 'rgba(90,180,190,0.35)';
  if (key === `${EDGE_DYE_MODE.PASS}-${EDGE_DYE_MODE.PASS}-${EDGE_DYE_MODE.PASS}`) return 'rgba(120,150,255,0.95)';
  if (key === `${EDGE_DYE_MODE.DEFLECT}-${EDGE_DYE_MODE.DEFLECT}-${EDGE_DYE_MODE.DEFLECT}`) return '#ffffff';
  if (key === `${EDGE_DYE_MODE.ABSORB}-${EDGE_DYE_MODE.ABSORB}-${EDGE_DYE_MODE.ABSORB}`) return 'rgba(255,180,70,0.95)';
  return 'rgba(210,120,255,0.95)';
}

function drawRegularPolygon(cx, cy, radius, sides, rotation = 0) {
  const n = Math.max(3, sides | 0);
  for (let i = 0; i < n; i++) {
    const a = rotation + (i / n) * Math.PI * 2;
    const x = cx + Math.cos(a) * radius;
    const y = cy + Math.sin(a) * radius;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function getBodiesBounds(bodies) {
  if (!bodies) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  const includePoint = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  };

  for (const rb of bodies.rigid || []) {
    const verts = rigidVerticesWorld(rb);
    for (const p of verts) includePoint(p.x, p.y);
  }
  for (const n of bodies.soft?.nodes || []) includePoint(n.x, n.y);

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) return null;
  return { minX, minY, maxX, maxY, cx: (minX + maxX) * 0.5, cy: (minY + maxY) * 0.5 };
}

function focusCameraOnBodies(sim, bodies) {
  if (!sim || !bodies) return;
  const b = getBodiesBounds(bodies);
  if (!b) return;
  sim.camera.x = b.cx;
  sim.camera.y = b.cy;
  // Keep current zoom but clamp camera into world bounds.
  clampCamera(sim);
}

function scaleImportedBodies(bodies, scale) {
  const s = Number(scale);
  if (!Number.isFinite(s) || Math.abs(s - 1) < 1e-6) return bodies;
  if (!bodies) return bodies;

  const b = getBodiesBounds(bodies);
  const cx = b?.cx ?? 0;
  const cy = b?.cy ?? 0;

  const scalePoint = (x, y) => ({
    x: cx + (x - cx) * s,
    y: cy + (y - cy) * s,
  });

  for (const rb of bodies.rigid || []) {
    const p = scalePoint(Number(rb.x) || 0, Number(rb.y) || 0);
    rb.x = p.x;
    rb.y = p.y;
    rb.r = Math.max(0.5, (Number(rb.r) || 0) * s);
    if (Number.isFinite(Number(rb.inertia))) rb.inertia = Number(rb.inertia) * s * s;
    if (Array.isArray(rb.verticesLocal)) {
      rb.verticesLocal = rb.verticesLocal.map((v) => ({ x: (Number(v?.x) || 0) * s, y: (Number(v?.y) || 0) * s }));
    }
    if (Array.isArray(rb.subPolysLocal)) {
      rb.subPolysLocal = rb.subPolysLocal.map((poly) => (Array.isArray(poly)
        ? poly.map((v) => ({ x: (Number(v?.x) || 0) * s, y: (Number(v?.y) || 0) * s }))
        : []));
    }
    delete rb._collisionPolysLocal;
    delete rb._collisionPolyCacheKey;
  }

  for (const n of bodies.soft?.nodes || []) {
    const p = scalePoint(Number(n.x) || 0, Number(n.y) || 0);
    n.x = p.x;
    n.y = p.y;
    n.r = Math.max(0.1, (Number(n.r) || 0) * s);
  }

  if (Array.isArray(bodies.soft?.springs)) {
    for (const sp of bodies.soft.springs) {
      if (!Array.isArray(sp) || sp.length < 3) continue;
      const rest = Number(sp[2]);
      sp[2] = Number.isFinite(rest) ? Math.max(1e-4, rest * s) : rest;
    }
  }

  // rigid-soft hybrid links removed from runtime pipelines.

  for (const c of bodies.softMembraneClusters || []) {
    if (Number.isFinite(Number(c?.restArea))) {
      c.restArea = Math.max(1e-4, Number(c.restArea) * s * s);
    }
  }

  return bodies;
}

function mergeBodiesIntoSim(target, incoming) {
  if (!target || !incoming) return;
  target.rigid = target.rigid || [];
  target.soft = target.soft || { nodes: [], springs: [] };
  target.softMembraneClusters = target.softMembraneClusters || [];

  const nodeOffset = target.soft.nodes.length;
  const maxCluster = target.soft.nodes.reduce((m, n) => Math.max(m, n.clusterId || 0), -1);
  const clusterOffset = maxCluster + 1;

  for (const rb of incoming.rigid || []) target.rigid.push({ ...rb });

  for (const n of incoming.soft?.nodes || []) {
    target.soft.nodes.push({ ...n, clusterId: (n.clusterId || 0) + clusterOffset });
  }
  for (const s of incoming.soft?.springs || []) {
    const [a, b, rest, edgeBodyMode, edgeDyeMode, edgeVelocityMode, edgeMomentumTransfer] = s;
    target.soft.springs.push([a + nodeOffset, b + nodeOffset, rest, edgeBodyMode, edgeDyeMode, edgeVelocityMode, edgeMomentumTransfer]);
  }

  // rigid-soft hybrid links removed from runtime pipelines.

  for (const c of incoming.softMembraneClusters || []) {
    const cid = Number(c?.clusterId);
    if (!Number.isInteger(cid)) continue;
    target.softMembraneClusters.push({
      clusterId: cid + clusterOffset,
      restArea: Number.isFinite(Number(c?.restArea)) ? Math.max(1e-4, Number(c.restArea)) : 1,
      pressureGain: Number.isFinite(Number(c?.pressureGain)) ? Math.max(0.001, Number(c.pressureGain)) : MEMBRANE_CELL_BASE_PRESSURE_GAIN,
      radialDamping: Number.isFinite(Number(c?.radialDamping)) ? clamp(Number(c.radialDamping), 0, 0.2) : MEMBRANE_CELL_BASE_RADIAL_DAMPING,
      shapeMemoryGain: Number.isFinite(Number(c?.shapeMemoryGain)) ? clamp(Number(c.shapeMemoryGain), 0, 0.35) : MEMBRANE_SHAPE_MEMORY_GAIN,
      insideCorrectionEnabled: Number.isFinite(Number(c?.insideCorrectionEnabled)) ? (Number(c.insideCorrectionEnabled) > 0 ? 1 : 0) : 1,
    });
  }

}

function translateBodies(bodies, dx, dy) {
  if (!bodies) return;
  const tx = Number(dx) || 0;
  const ty = Number(dy) || 0;

  for (const rb of bodies.rigid || []) {
    rb.x = (Number(rb.x) || 0) + tx;
    rb.y = (Number(rb.y) || 0) + ty;
  }
  for (const node of bodies.soft?.nodes || []) {
    node.x = (Number(node.x) || 0) + tx;
    node.y = (Number(node.y) || 0) + ty;
  }
}

function centerBodiesInWorld(bodies, worldSize, targetSpanFraction = 0.42) {
  if (!bodies) return;
  const bounds = getBodiesBounds(bodies);
  if (!bounds) return;

  const span = Math.max(1e-6, Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY));
  const target = Math.max(6, Math.min(worldSize * 0.8, worldSize * Math.max(0.12, Number(targetSpanFraction) || 0.42)));
  const scale = target / span;
  if (Number.isFinite(scale) && Math.abs(scale - 1) > 1e-4) {
    scaleImportedBodies(bodies, scale);
  }

  const recentered = getBodiesBounds(bodies);
  if (!recentered) return;
  translateBodies(bodies, worldSize * 0.5 - recentered.cx, worldSize * 0.5 - recentered.cy);
}

function configureInteractionLab(sim, options = {}) {
  const cfg = options?.interactionLab;
  if (!cfg || typeof cfg !== 'object') {
    sim.interactionLab = null;
    return;
  }

  const targetType = String(cfg.targetType || 'rigid').toLowerCase() === 'soft' ? 'soft' : 'rigid';
  const mode = String(cfg.mode || 'pinned').toLowerCase() === 'circle' ? 'circle' : 'pinned';
  const dt = Math.max(1e-4, Number(sim?.controls?.dt) || 0.01);
  const worldSize = getWorldSize(sim.controls);

  const defaultRigid = sim?.bodies?.rigid?.[0] || { x: worldSize * 0.5, y: worldSize * 0.5, theta: 0 };
  const anchorX = Number.isFinite(Number(cfg.anchorX)) ? Number(cfg.anchorX) : (Number(defaultRigid.x) || worldSize * 0.5);
  const anchorY = Number.isFinite(Number(cfg.anchorY)) ? Number(cfg.anchorY) : (Number(defaultRigid.y) || worldSize * 0.5);
  const anchorTheta = Number.isFinite(Number(cfg.anchorTheta)) ? Number(cfg.anchorTheta) : (Number(defaultRigid.theta) || 0);

  const circleCenterX = Number.isFinite(Number(cfg.circleCenterX)) ? Number(cfg.circleCenterX) : anchorX;
  const circleCenterY = Number.isFinite(Number(cfg.circleCenterY)) ? Number(cfg.circleCenterY) : anchorY;
  const circleRadius = Math.max(0, Number.isFinite(Number(cfg.circleRadius)) ? Number(cfg.circleRadius) : 12);
  const circleAngularSpeed = Number.isFinite(Number(cfg.circleAngularSpeed)) ? Number(cfg.circleAngularSpeed) : (Math.PI * 0.22);
  const thetaSpin = Number.isFinite(Number(cfg.thetaSpin)) ? Number(cfg.thetaSpin) : 0;

  const softNodeIndices = Array.isArray(cfg.softNodeIndices)
    ? cfg.softNodeIndices.map((v) => Number(v) | 0).filter((v) => v >= 0 && v < (sim?.bodies?.soft?.nodes?.length || 0))
    : null;

  const softBasePositions = Array.isArray(cfg.softBasePositions)
    ? cfg.softBasePositions.map((p) => ({ x: Number(p?.x) || 0, y: Number(p?.y) || 0 }))
    : null;

  sim.interactionLab = {
    enabled: cfg.enabled !== false,
    targetType,
    mode,
    rigidIndex: Number.isInteger(Number(cfg.rigidIndex)) ? (Number(cfg.rigidIndex) | 0) : 0,
    softNodeIndices,
    softBasePositions,
    anchorX,
    anchorY,
    anchorTheta,
    circleCenterX,
    circleCenterY,
    circleRadius,
    circleAngularSpeed,
    thetaSpin,
    dt,
  };
}

function applyInteractionLabBodyMotion(sim) {
  const cfg = sim?.interactionLab;
  if (!cfg || cfg.enabled === false) return;

  const dt = Math.max(1e-4, Number(sim?.controls?.dt) || Number(cfg.dt) || 0.01);
  const t = (Number(sim?.frame) || 0) * dt;
  const isCircle = cfg.mode === 'circle';

  const tx = isCircle
    ? (cfg.circleCenterX + cfg.circleRadius * Math.cos(cfg.circleAngularSpeed * t))
    : cfg.anchorX;
  const ty = isCircle
    ? (cfg.circleCenterY + cfg.circleRadius * Math.sin(cfg.circleAngularSpeed * t))
    : cfg.anchorY;
  const tvx = isCircle
    ? (-cfg.circleRadius * cfg.circleAngularSpeed * Math.sin(cfg.circleAngularSpeed * t))
    : 0;
  const tvy = isCircle
    ? (cfg.circleRadius * cfg.circleAngularSpeed * Math.cos(cfg.circleAngularSpeed * t))
    : 0;
  const ttheta = cfg.anchorTheta + (cfg.thetaSpin || 0) * t;
  const tomega = isCircle ? (cfg.thetaSpin || 0) : 0;

  if (cfg.targetType === 'soft') {
    const nodes = sim?.bodies?.soft?.nodes || [];
    const indices = (Array.isArray(cfg.softNodeIndices) && cfg.softNodeIndices.length > 0)
      ? cfg.softNodeIndices
      : nodes.map((_, i) => i);

    if (!Array.isArray(cfg.softBasePositions) || cfg.softBasePositions.length !== indices.length) {
      cfg.softBasePositions = indices.map((idx) => {
        const node = nodes[idx];
        return { x: Number(node?.x) || 0, y: Number(node?.y) || 0 };
      });
    }

    // Anchor offsets are derived from first base point against anchor.
    const ref = cfg.softBasePositions[0] || { x: tx, y: ty };
    const ox = tx - (Number(ref.x) || 0);
    const oy = ty - (Number(ref.y) || 0);

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const node = nodes[idx];
      const base = cfg.softBasePositions[i] || node;
      if (!node || !base) continue;
      node.x = (Number(base.x) || 0) + ox;
      node.y = (Number(base.y) || 0) + oy;
      node.vx = tvx;
      node.vy = tvy;
    }
    return;
  }

  const rb = sim?.bodies?.rigid?.[cfg.rigidIndex | 0];
  if (!rb) return;
  rb.x = tx;
  rb.y = ty;
  rb.theta = ttheta;
  rb.vx = tvx;
  rb.vy = tvy;
  rb.omega = tomega;
}

function buildWindTunnelEmitter(n, {
  strength = 3.8,
  radius = Math.max(7, n / 13),
  yFraction = 0.12,
  spin = 1.25,
  curlGain = 1.65,
  driftGain = 0.035,
  chaosGain = 1.35,
  wobbleAmp = Math.max(0.8, n * 0.016),
  wobbleFreq = 0.085,
  swirlJitter = 0.73,
  jetVy = 0.38,
  colorR = 135,
  colorG = 190,
  colorB = 255,
  lockPosition = false,
} = {}) {
  return {
    x: n * 0.5,
    y: n * Math.max(0.06, Math.min(0.35, Number(yFraction) || 0.12)),
    vx: 0,
    vy: Number.isFinite(Number(jetVy)) ? Number(jetVy) : 0.38,
    r: Math.max(3, Number(radius) || (n / 13)),
    cr: Number.isFinite(Number(colorR)) ? clamp(Number(colorR), 0, 255) : 135,
    cg: Number.isFinite(Number(colorG)) ? clamp(Number(colorG), 0, 255) : 190,
    cb: Number.isFinite(Number(colorB)) ? clamp(Number(colorB), 0, 255) : 255,
    strength: Math.max(0.1, Number(strength) || 3.8),
    spin: Number.isFinite(Number(spin)) ? Number(spin) : 1.25,
    curlGain: Number.isFinite(Number(curlGain)) ? Number(curlGain) : 1.65,
    driftGain: Number.isFinite(Number(driftGain)) ? Number(driftGain) : 0.035,
    chaosGain: Number.isFinite(Number(chaosGain)) ? Number(chaosGain) : 1.35,
    wobbleAmp: Number.isFinite(Number(wobbleAmp)) ? Number(wobbleAmp) : Math.max(0.8, n * 0.016),
    wobbleFreq: Number.isFinite(Number(wobbleFreq)) ? Number(wobbleFreq) : 0.085,
    swirlJitter: Number.isFinite(Number(swirlJitter)) ? Number(swirlJitter) : 0.73,
    lockPosition: !!lockPosition,
  };
}

async function resetEmbedWindTunnelFromSpec(specInput, options = {}) {
  const spec = (typeof specInput === 'string')
    ? parseCreatureSpec(specInput)
    : parseCreatureSpec(JSON.stringify(specInput || {}));
  const grid = Math.max(32, Math.min(2048, Math.round(Number(options?.grid) || readControls().n || 128)));
  const importScale = Number.isFinite(Number(options?.importScale)) ? Number(options.importScale) : 1;
  const targetSpanFraction = Number.isFinite(Number(options?.targetSpanFraction))
    ? Number(options.targetSpanFraction)
    : 0.42;

  if (gridEl) gridEl.value = String(grid);
  if (showViscEl) showViscEl.checked = false;
  const selectedSolverPath = setRuntimeSolverPath(options?.solverPath ?? getRuntimeSolverPath(), { syncUrl: true });
  const selectedPipelineMode = setRuntimePipelineMode(options?.pipelineMode ?? getRuntimePipelineMode(selectedSolverPath), {
    syncUrl: true,
    solverPath: selectedSolverPath,
  });

  running = false;
  sim = await initSim();
  // Wind-tunnel embed should use only the explicit configured emitter.
  sim.disableDefaultInject = true;
  sim.controls.runtimeSolverPath = selectedSolverPath;
  sim.controls.runtimePipelineMode = selectedPipelineMode;
  sim.controls.fastReadbackInterval = normalizeFastReadbackInterval(
    options?.fastReadbackInterval ?? sim.controls.fastReadbackInterval,
  );
  if (typeof options?.allowPassEdgeFlowPush === 'boolean') {
    sim.controls.allowPassEdgeFlowPush = options.allowPassEdgeFlowPush;
  }

  sim.bodies = {
    rigid: [],
    soft: { nodes: [], springs: [] },
    hybrid: [],
    softMembraneClusters: [],
  };

  const imported = buildBodiesFromCreatureSpec(spec, sim.controls.n, sim.controls);
  imported.hybrid = [];
  scaleImportedBodies(imported, importScale);
  centerBodiesInWorld(imported, getWorldSize(sim.controls), targetSpanFraction);
  mergeBodiesIntoSim(sim.bodies, imported);
  configureInteractionLab(sim, options);

  sim.overlayShowSegmentIds = options?.overlayShowSegmentIds === true;
  sim.overlayShowExtraVisuals = options?.overlayShowExtraVisuals !== false;

  const worldSize = getWorldSize(sim.controls);
  sim.emitters = [buildWindTunnelEmitter(worldSize, {
    strength: Number(options?.emitterStrength) || 3.8,
    radius: Number(options?.emitterRadius) || Math.max(7, worldSize / 13),
    yFraction: Number(options?.emitterYFraction) || 0.12,
    // Embed wind-tunnel: fixed top emitter with strong downward jet + moderate turbulence.
    spin: Number(options?.emitterSpin) || 1.55,
    curlGain: Number(options?.emitterCurlGain) || 1.95,
    driftGain: Number(options?.emitterDriftGain) || 0.22,
    chaosGain: Number(options?.emitterChaosGain) || 1.45,
    wobbleAmp: Number(options?.emitterWobbleAmp) || Math.max(1.4, worldSize * 0.02),
    wobbleFreq: Number(options?.emitterWobbleFreq) || 0.11,
    swirlJitter: Number(options?.emitterSwirlJitter) || 1.17,
    jetVy: Number(options?.emitterJetVy) || 1.35,
    colorR: Number(options?.emitterColorR),
    colorG: Number(options?.emitterColorG),
    colorB: Number(options?.emitterColorB),
    lockPosition: options?.emitterLockPosition !== false,
  })];

  sim.camera.x = worldSize * 0.5;
  sim.camera.y = worldSize * 0.5;
  sim.camera.zoom = 1.0;
  clampCamera(sim);

  sim.softDeformReferenceState = null;

  running = true;
  stepAndRender().catch((e) => {
    running = false;
    log({ ok: false, error: String(e) });
  });

  return {
    ok: true,
    mode: 'embed-wind-tunnel',
    runtimeSolverPath: normalizeRuntimeSolverPath(sim?.controls?.runtimeSolverPath),
    runtimePipelineMode: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
    grid: sim.controls.n,
    rigidBodies: sim.bodies.rigid.length,
    softNodes: sim.bodies.soft?.nodes?.length || 0,
    emitterCount: sim.emitters.length,
    interactionLab: sim.interactionLab ? {
      enabled: sim.interactionLab.enabled !== false,
      targetType: sim.interactionLab.targetType,
      motion: sim.interactionLab.mode,
    } : null,
  };
}

function isConcavePolygon(verts) {
  if (!Array.isArray(verts) || verts.length < 4) return false;
  let hasPos = false;
  let hasNeg = false;
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    const c = verts[(i + 2) % verts.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross > 1e-6) hasPos = true;
    if (cross < -1e-6) hasNeg = true;
    if (hasPos && hasNeg) return true;
  }
  return false;
}

function drawBodiesOverlay(sim) {
  const smooth = 0.35;
  const v = getCameraView(sim);
  const collisionDebug = !!showCollisionHullEl?.checked;
  const showSegmentIds = sim?.overlayShowSegmentIds !== false;
  const showExtraVisuals = sim?.overlayShowExtraVisuals !== false;
  let concaveCount = 0;

  const drawSegmentIdLabel = (id, pa, pb, color = 'rgba(255,255,255,0.92)') => {
    if (!id || !pa || !pb) return;
    const dx = pb.x - pa.x;
    const dy = pb.y - pa.y;
    const len = Math.hypot(dx, dy);
    if (!Number.isFinite(len) || len < 1e-3) return;
    const nx = -dy / len;
    const ny = dx / len;
    const mx = 0.5 * (pa.x + pb.x);
    const my = 0.5 * (pa.y + pb.y);
    const off = 7;

    ctx.save();
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = color;
    ctx.strokeStyle = 'rgba(0,0,0,0.7)';
    ctx.lineWidth = 3;
    ctx.strokeText(id, mx + nx * off, my + ny * off);
    ctx.fillText(id, mx + nx * off, my + ny * off);
    ctx.restore();
  };

  const highlightId = String(sim?.highlightSegmentId || '');
  const highlightOn = !!highlightId && (Number(sim?.frame) <= Number(sim?.highlightUntilFrame || 0));
  const segmentHighlighted = (id) => highlightOn && id === highlightId;

  ctx.save();
  ctx.lineWidth = 1.5;
  for (let i = 0; i < sim.bodies.rigid.length; i++) {
    const b = sim.bodies.rigid[i];
    if (b._rx == null) {
      b._rx = b.x; b._ry = b.y; b._rtheta = b.theta || 0;
    } else {
      b._rx += (b.x - b._rx) * smooth;
      b._ry += (b.y - b._ry) * smooth;
      b._rtheta += ((b.theta || 0) - b._rtheta) * smooth;
    }
    ctx.fillStyle = b.digestEnabled ? 'rgba(255, 90, 120, 0.22)' : 'rgba(255,255,255,0.06)';
    const vertsW = rigidVerticesWorld({ ...b, x: b._rx, y: b._ry, theta: b._rtheta });
    const verts = vertsW.map((v) => worldToScreen(sim, v.x, v.y));
    const solverVertsW = rigidVerticesWorld(b);
    if (isConcavePolygon(solverVertsW)) concaveCount += 1;
    const sides = verts.length;
    ctx.beginPath();
    for (let vi = 0; vi < sides; vi++) {
      const p = verts[vi];
      if (vi === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
    for (let ei = 0; ei < sides; ei++) {
      const a = verts[ei];
      const b2 = verts[(ei + 1) % sides];
      const dyeMode = !b.edgeDyeMode ? EDGE_DYE_MODE.DEFLECT : b.edgeDyeMode[ei];
      const bodyMode = Array.isArray(b.edgeVelocityMode) ? b.edgeVelocityMode[ei] : (!b.edgeBodyMode ? EDGE_BODY_MODE.BLOCK : b.edgeBodyMode[ei]);
      const momentum = clamp(
        Array.isArray(b.edgeMomentumCoupling)
          ? Number(b.edgeMomentumCoupling[ei])
          : (Array.isArray(b.edgeMomentumTransfer) ? Number(b.edgeMomentumTransfer[ei]) : 1),
        0,
        1,
      );
      ctx.strokeStyle = edgeModeColor(dyeMode, bodyMode === EDGE_BODY_MODE.BLOCK);
      const prevWidth = ctx.lineWidth;
      ctx.lineWidth = 1.0 + 1.5 * momentum;
      if (Number(bodyMode) === EDGE_BODY_MODE.PASS) ctx.setLineDash([6, 4]);
      else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b2.x, b2.y);
      ctx.stroke();
      const rigidSegId = `R${i}:${ei}`;
      if (showSegmentIds) drawSegmentIdLabel(rigidSegId, a, b2, 'rgba(255,245,200,0.96)');
      if (segmentHighlighted(rigidSegId)) {
        ctx.save();
        ctx.setLineDash([]);
        ctx.lineWidth = 4;
        ctx.strokeStyle = 'rgba(255,255,120,0.98)';
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b2.x, b2.y);
        ctx.stroke();
        ctx.restore();
      }
      ctx.setLineDash([]);
      ctx.lineWidth = prevWidth;
    }

    if (collisionDebug) {
      const solverVerts = solverVertsW.map((p) => worldToScreen(sim, p.x, p.y));

      // Solver collision hull (actual polygon used by rigid-vs-soft concave contact).
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = isConcavePolygon(solverVertsW) ? 'rgba(80,255,120,0.98)' : 'rgba(120,220,255,0.95)';
      ctx.beginPath();
      for (let vi = 0; vi < solverVerts.length; vi++) {
        const p = solverVerts[vi];
        if (vi === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.stroke();

      // Solver convex proxies used by rigid-rigid polygon collisions.
      const proxyPolys = getRigidCollisionPolysWorld(b);
      ctx.strokeStyle = 'rgba(255,170,70,0.9)';
      for (const poly of proxyPolys) {
        const ps = poly.map((pp) => worldToScreen(sim, pp.x, pp.y));
        if (ps.length < 3) continue;
        ctx.beginPath();
        for (let k = 0; k < ps.length; k++) {
          const p = ps[k];
          if (k === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.stroke();
      }
      ctx.setLineDash([]);

      for (const p of solverVerts) {
        ctx.fillStyle = 'rgba(255,255,0,0.95)';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  const s = sim.bodies.soft;
  const deformState = sim.softDeformationState || null;
  const severeClusters = deformState?.severeSet || new Set();
  const warningClusters = deformState?.warningSet || new Set();

  for (const node of s.nodes) {
    if (node._rx == null) {
      node._rx = node.x; node._ry = node.y;
    } else {
      node._rx += (node.x - node._rx) * smooth;
      node._ry += (node.y - node._ry) * smooth;
    }
  }

  if (showExtraVisuals) {
    const softClusters = new Map();
    for (const node of s.nodes) {
      const cid = node.clusterId ?? 0;
      if (!softClusters.has(cid)) softClusters.set(cid, []);
      softClusters.get(cid).push(node);
    }
    for (const [cid, nodes] of softClusters.entries()) {
      const severe = severeClusters.has(cid);
      const warning = warningClusters.has(cid);
      if (!nodes.length || (!nodes[0].digestEnabled && !severe && !warning)) continue;
      let cx = 0, cy = 0;
      for (const n0 of nodes) { cx += n0._rx; cy += n0._ry; }
      cx /= nodes.length; cy /= nodes.length;
      const ordered = [...nodes]
        .map((p) => ({ x: p._rx, y: p._ry, a: Math.atan2(p._ry - cy, p._rx - cx) }))
        .sort((a, b2) => a.a - b2.a);
      ctx.fillStyle = severe
        ? 'rgba(255, 60, 60, 0.24)'
        : (warning ? 'rgba(255, 180, 70, 0.2)' : 'rgba(255, 80, 140, 0.2)');
      ctx.beginPath();
      for (let i = 0; i < ordered.length; i++) {
        const p = worldToScreen(sim, ordered[i].x, ordered[i].y);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.fill();
    }
  }
  for (let sgi = 0; sgi < (s.springs || []).length; sgi++) {
    const spring = s.springs[sgi];
    if (!Array.isArray(spring) || spring.length < 2) continue;
    const [i, j, _rest, edgeBodyMode, edgeDyeMode, edgeVelocityMode, edgeMomentumCoupling] = spring;
    const a = s.nodes[i], b = s.nodes[j];
    if (!a || !b) continue;
    const pa = worldToScreen(sim, a._rx, a._ry);
    const pb = worldToScreen(sim, b._rx, b._ry);
    const aSevere = severeClusters.has(a.clusterId ?? 0);
    const bSevere = severeClusters.has(b.clusterId ?? 0);
    const aWarn = warningClusters.has(a.clusterId ?? 0);
    const bWarn = warningClusters.has(b.clusterId ?? 0);
    const resolvedBodyMode = Number(edgeVelocityMode) === EDGE_BODY_MODE.PASS
      ? EDGE_BODY_MODE.PASS
      : (Number(edgeBodyMode) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK);
    const momentumCoupling = clamp(Number.isFinite(Number(edgeMomentumCoupling)) ? Number(edgeMomentumCoupling) : 1, 0, 1);

    if (aSevere || bSevere) {
      ctx.strokeStyle = 'rgba(255, 70, 70, 0.98)';
    } else if (aWarn || bWarn) {
      ctx.strokeStyle = 'rgba(255, 190, 80, 0.92)';
    } else {
      const baseColor = edgeModeColor(edgeDyeMode, resolvedBodyMode === EDGE_BODY_MODE.BLOCK);
      // Keep blocked+deflect soft perimeter cyan-ish for readability.
      const m = normalizeEdgeDyeModeRGB(edgeDyeMode);
      if (resolvedBodyMode === EDGE_BODY_MODE.BLOCK && m[0] === EDGE_DYE_MODE.DEFLECT && m[1] === EDGE_DYE_MODE.DEFLECT && m[2] === EDGE_DYE_MODE.DEFLECT) {
        ctx.strokeStyle = '#00ffd0';
      } else {
        ctx.strokeStyle = baseColor;
      }
    }

    const lineWidthPrev = ctx.lineWidth;
    ctx.lineWidth = (aSevere || bSevere || aWarn || bWarn)
      ? 2
      : (0.9 + 1.8 * momentumCoupling);
    if (resolvedBodyMode === EDGE_BODY_MODE.PASS) {
      ctx.setLineDash([5, 4]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
    const softSegId = `S${sgi}`;
    if (showSegmentIds) drawSegmentIdLabel(softSegId, pa, pb, 'rgba(180,255,220,0.96)');
    if (segmentHighlighted(softSegId)) {
      ctx.save();
      ctx.setLineDash([]);
      ctx.lineWidth = 4;
      ctx.strokeStyle = 'rgba(255,255,120,0.98)';
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.restore();
    }

    ctx.setLineDash([]);
    ctx.lineWidth = lineWidthPrev;
  }
  if (showExtraVisuals) {
    for (const node of s.nodes) {
      const p = worldToScreen(sim, node._rx, node._ry);
      const cid = node.clusterId ?? 0;
      ctx.fillStyle = severeClusters.has(cid)
        ? 'rgba(255, 70, 70, 0.98)'
        : (warningClusters.has(cid) ? 'rgba(255, 190, 80, 0.95)' : '#00ffd0');
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1.6, 2.2 * Math.max(1, sim.camera.zoom * 0.6)), 0, Math.PI * 2);
      ctx.fill();
    }

    // rigid-soft hybrid link visualization removed.

    if (collisionDebug && Array.isArray(sim.lastRigidContacts)) {
      ctx.fillStyle = 'rgba(255,90,90,0.95)';
      for (const c of sim.lastRigidContacts.slice(0, 24)) {
        if (!Array.isArray(c.contact) || c.contact.length < 2) continue;
        const p = worldToScreen(sim, c.contact[0], c.contact[1]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    const collisionDebugSuffix = collisionDebug
      ? ` | solver hull debug ON (concave ${concaveCount}/${sim.bodies.rigid.length}, contacts ${(sim.lastRigidContacts || []).length})`
      : '';
    const obstacleOverlayLabel = sim?.controls?.showObstacleMaskOverlay === false
      ? 'obstacle mask overlay=off'
      : 'obstacle mask overlay=red';
    ctx.fillText(`Dye edges: PASS=blue, DEFLECT=white/cyan, REMOVE=amber, MIXED=violet | ${obstacleOverlayLabel} | zoom ${sim.camera.zoom.toFixed(2)}x${collisionDebugSuffix}`, 10, canvas.height - 28);
    ctx.fillStyle = 'rgba(0,255,208,0.95)';
    const line2 = collisionDebug
      ? 'Body edges: BLOCK solid vs PASS dashed | segment IDs: R<body>:<edge>, S<soft-spring> | soft momentum: thin→thick (0→1) | dashed green/cyan=solver hull, dashed amber=rigid-rigid convex proxies | soft deform warn=orange, severe=red'
      : 'Body edges: BLOCK solid vs PASS dashed | segment IDs: R<body>:<edge>, S<soft-spring> | soft momentum: thin→thick (0→1) | soft deform warn=orange, severe=red';
    ctx.fillText(line2, 10, canvas.height - 12);
  }
  ctx.restore();
}

async function initSim() {
  const controls = readControls();
  if (!navigator.gpu) throw new Error('WebGPU unavailable in browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');

  const rendererFingerprint = detectRendererFingerprintSync();
  rendererFingerprint.webgpuAdapter = await readWebGpuAdapterFingerprint(adapter);

  // Some browsers/hardware only expose higher storage-buffer stage limits when
  // explicitly requested at device creation time. Advect-dye now needs 10.
  // Be robust even if adapter.limits is absent/stale by optimistic fallback tries.
  let device = null;
  const maxStoragePerStage = Number(adapter.limits?.maxStorageBuffersPerShaderStage || 8);
  const requestedStorageLimitCandidates = [10, 9].filter((limit) => Number.isFinite(limit));

  for (const limit of requestedStorageLimitCandidates) {
    // If adapter explicitly reports a lower max, skip impossible requests.
    if (Number.isFinite(maxStoragePerStage) && maxStoragePerStage < limit) continue;
    try {
      device = await adapter.requestDevice({
        requiredLimits: { maxStorageBuffersPerShaderStage: limit },
      });
      break;
    } catch (err) {
      console.warn('[gpu-lab] requestDevice with storage-buffer limit failed; falling back', {
        limit,
        error: String(err?.message || err),
      });
    }
  }

  if (!device) {
    device = await adapter.requestDevice();
  }

  const cells = controls.n * controls.n;
  const bytes = cells * 4;

  const uniform = createUniformBuffer(device);
  uploadUniforms(device, uniform, controls);

  const vxA = createBuffer(device, bytes), vxB = createBuffer(device, bytes);
  const vyA = createBuffer(device, bytes), vyB = createBuffer(device, bytes);
  const div = createBuffer(device, bytes);
  const pA = createBuffer(device, bytes), pB = createBuffer(device, bytes);
  const rA = createBuffer(device, bytes), rB = createBuffer(device, bytes);
  const gA = createBuffer(device, bytes), gB = createBuffer(device, bytes);
  const bA = createBuffer(device, bytes), bB = createBuffer(device, bytes);

  const viscMapCpu = makeDefaultViscMap(controls.n);
  const viscMapGpu = createBuffer(device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(viscMapGpu, 0, viscMapCpu);

  const obstacleMaskCpu = new Float32Array(cells);
  const obstacleMaskGpu = createBuffer(device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(obstacleMaskGpu, 0, obstacleMaskCpu);

  const dyeModeMaskCpu = new Uint32Array(cells);
  const dyeModeMaskGpu = createBuffer(device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(dyeModeMaskGpu, 0, dyeModeMaskCpu);

  const readR = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readG = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readB = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readVx = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readVy = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const inject = await createPipeline(device, injectWgsl);
  const advVel = await createPipeline(device, advectVelWgsl);
  const divPipe = await createPipeline(device, divergenceWgsl);
  const jacobiP = await createPipeline(device, jacobiPressureWgsl);
  const project = await createPipeline(device, projectWgsl);
  const advDye = await createPipeline(device, advectDyeWgsl);
  const fastFluidHealthPipeline = await createPipeline(device, fastFluidHealthWgsl);
  const fastFluidHealthParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const fastFluidHealthStats = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const fastFluidHealthReadback = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  let rigidRigidCandidateBackend = null;
  let rigidRigidCandidateBackendMode = controls.rigidRigidCandidateBackendMode || 'js';
  let rigidRigidCandidateBackendReason = rigidRigidCandidateBackendMode === 'wasm'
    ? 'loading'
    : 'disabled';
  if (rigidRigidCandidateBackendMode === 'wasm') {
    try {
      rigidRigidCandidateBackend = await loadRigidRigidCandidateBackendWasm();
      rigidRigidCandidateBackendReason = 'loaded';
    } catch (err) {
      rigidRigidCandidateBackend = null;
      rigidRigidCandidateBackendMode = 'js';
      rigidRigidCandidateBackendReason = `load-failed:${String(err?.message || err)}`;
      console.warn('[gpu-lab] rigid-rigid wasm candidate backend unavailable; using JS path', {
        error: String(err?.message || err),
      });
    }
  }

  let postCollisionProjectionBackend = null;
  let postCollisionProjectionBackendMode = controls.postCollisionProjectionBackendMode || 'js';
  let postCollisionProjectionBackendReason = postCollisionProjectionBackendMode === 'wasm'
    ? 'loading'
    : 'disabled';
  if (postCollisionProjectionBackendMode === 'wasm') {
    try {
      postCollisionProjectionBackend = await loadSoftClusterProjectionBackendWasm();
      postCollisionProjectionBackendReason = 'loaded';
    } catch (err) {
      postCollisionProjectionBackend = null;
      postCollisionProjectionBackendMode = 'js';
      postCollisionProjectionBackendReason = `load-failed:${String(err?.message || err)}`;
      console.warn('[gpu-lab] post-collision projection wasm backend unavailable; using JS path', {
        error: String(err?.message || err),
      });
    }
  }

  let rigidStepBaselineBackend = null;
  let rigidStepBaselineBackendMode = controls.rigidStepBaselineBackendMode || 'js';
  let rigidStepBaselineBackendReason = rigidStepBaselineBackendMode === 'wasm'
    ? 'loading'
    : 'disabled';
  if (rigidStepBaselineBackendMode === 'wasm') {
    try {
      rigidStepBaselineBackend = await loadRigidStepBaselineBackendWasm();
      rigidStepBaselineBackendReason = 'loaded';
    } catch (err) {
      rigidStepBaselineBackend = null;
      rigidStepBaselineBackendMode = 'js';
      rigidStepBaselineBackendReason = `load-failed:${String(err?.message || err)}`;
      console.warn('[gpu-lab] rigid baseline step wasm backend unavailable; using JS path', {
        error: String(err?.message || err),
      });
    }
  }

  return {
    controls, cells, bytes,
    device, uniform,
    rendererFingerprint,
    rigidRigidCandidateBackend,
    rigidRigidCandidateBackendMode,
    rigidRigidCandidateBackendReason,
    postCollisionProjectionBackend,
    postCollisionProjectionBackendMode,
    postCollisionProjectionBackendReason,
    rigidStepBaselineBackend,
    rigidStepBaselineBackendMode,
    rigidStepBaselineBackendReason,
    inject, advVel, divPipe, jacobiP, project, advDye,
    vx0: vxA, vx1: vxB, vy0: vyA, vy1: vyB,
    pr0: pA, pr1: pB,
    rr0: rA, rr1: rB, gg0: gA, gg1: gB, bb0: bA, bb1: bB,
    viscMapCpu, viscMapGpu,
    obstacleMaskCpu, obstacleMaskGpu,
    dyeModeMaskCpu, dyeModeMaskGpu,
    div, readR, readG, readB, readVx, readVy,
    fastFluidHealthPipeline,
    fastFluidHealthParams,
    fastFluidHealthStats,
    fastFluidHealthReadback,
    fastReadbackPolicy: {
      enabled: true,
      intervalFrames: FAST_MODE_FULL_READBACK_INTERVAL,
      skippedFramesSinceFull: 0,
      lastFullReadbackFrame: -1,
      lastDecision: 'full-readback',
      lastReason: 'init',
      lastHealth: { nonFiniteCount: 0, velocitySpikeCount: 0, dyeSpikeCount: 0 },
    },
    fastShadowFields: null,
    laggedCouplingFields: null,
    couplingLagRuntime: {
      enabled: false,
      active: false,
      lagFrames: 0,
      bootstrapFrames: 0,
      reusedFrames: 0,
      lastMode: 'disabled',
      lastReadbackWaitMs: 0,
    },
    bodies: initBodies(controls.n, controls, getWorldSize(controls)),
    emitters: initEmitters(getWorldSize(controls)),
    disableDefaultInject: false,
    camera: { x: getWorldSize(controls) * 0.5, y: getWorldSize(controls) * 0.5, zoom: 1.0 },
    couplingTelemetry: [],
    pipelineTimingFrame: null,
    pipelineTimingLast: null,
    pipelineTimingHistory: [],
    lastFluidObstacleStats: { rigidBlockedEdges: 0, softBlockedEdges: 0, blockedEdgeCount: 0, blockedCells: 0, sweptCells: 0 },
    lastDyeMaskStats: { rigidEdges: 0, softEdges: 0, nonPassCells: 0 },
    lastSweptDyeTransportStats: { touchedCells: 0, movedMass: 0, deletedMass: 0, eatenMass: 0 },
    overlayShowSegmentIds: false,
    overlayShowExtraVisuals: true,
    highlightSegmentId: null,
    highlightUntilFrame: 0,
    softIntegrateRuntime: { mode: 'cpu-baseline', reason: 'init' },
    rigidStepBaselineRuntime: { mode: 'cpu-baseline', reason: 'init', processedBodies: 0, sampleCount: 0, jsMarshalMs: 0, backendComputeMs: 0, totalMs: 0 },
    postCollisionProjectionRuntime: { mode: 'cpu-baseline', reason: 'init', projectedNodes: 0, meanDelta: 0, jsMarshalMs: 0, backendComputeMs: 0, totalMs: 0 },
    frame: 0, t0: performance.now(),
  };
}

function workgroups(n) { return Math.ceil(n / WORKGROUP); }

function enqueueFullFluidReadbackCopies(sim, enc) {
  enc.copyBufferToBuffer(sim.rr0, 0, sim.readR, 0, sim.bytes);
  enc.copyBufferToBuffer(sim.gg0, 0, sim.readG, 0, sim.bytes);
  enc.copyBufferToBuffer(sim.bb0, 0, sim.readB, 0, sim.bytes);
  enc.copyBufferToBuffer(sim.vx0, 0, sim.readVx, 0, sim.bytes);
  enc.copyBufferToBuffer(sim.vy0, 0, sim.readVy, 0, sim.bytes);
}

async function mapFullFluidReadback(sim) {
  await Promise.all([
    sim.readR.mapAsync(GPUMapMode.READ),
    sim.readG.mapAsync(GPUMapMode.READ),
    sim.readB.mapAsync(GPUMapMode.READ),
    sim.readVx.mapAsync(GPUMapMode.READ),
    sim.readVy.mapAsync(GPUMapMode.READ),
  ]);

  const r = new Float32Array(sim.readR.getMappedRange().slice(0));
  const g = new Float32Array(sim.readG.getMappedRange().slice(0));
  const b = new Float32Array(sim.readB.getMappedRange().slice(0));
  const vx = new Float32Array(sim.readVx.getMappedRange().slice(0));
  const vy = new Float32Array(sim.readVy.getMappedRange().slice(0));
  sim.readR.unmap();
  sim.readG.unmap();
  sim.readB.unmap();
  sim.readVx.unmap();
  sim.readVy.unmap();

  return { r, g, b, vx, vy };
}

async function readFastFluidHealthStats(sim) {
  if (!sim?.fastFluidHealthReadback) {
    return { nonFiniteCount: 0, velocitySpikeCount: 0, dyeSpikeCount: 0 };
  }

  await sim.fastFluidHealthReadback.mapAsync(GPUMapMode.READ);
  const arr = new Uint32Array(sim.fastFluidHealthReadback.getMappedRange().slice(0));
  sim.fastFluidHealthReadback.unmap();

  return {
    nonFiniteCount: Number(arr[0]) || 0,
    velocitySpikeCount: Number(arr[1]) || 0,
    dyeSpikeCount: Number(arr[2]) || 0,
  };
}

async function stepAndRender() {
  if (!running || !sim) return;
  const s = sim;
  ensureFramePipelineTiming(s);
  const frameStageStartMs = performance.now();
  const uiControls = readControls();

  // Grid-size changes require full GPU buffer reallocation; hot-swapping n causes dimension mismatches.
  if (uiControls.n !== s.controls.n) {
    running = false;
    sim = null;
    log({ ok: true, msg: `reinitializing for grid ${uiControls.n}` });
    await start();
    return;
  }
  // Body archetype selection changes require body re-seeding.
  if (
    uiControls.seedRigidBodies !== s.controls.seedRigidBodies ||
    uiControls.seedSpringSoftBodies !== s.controls.seedSpringSoftBodies ||
    uiControls.spawnMembraneCells !== s.controls.spawnMembraneCells
  ) {
    running = false;
    sim = null;
    log({
      ok: true,
      msg: `reinitializing for seed toggles rigid=${uiControls.seedRigidBodies ? 'ON' : 'OFF'} springSoft=${uiControls.seedSpringSoftBodies ? 'ON' : 'OFF'} membraneSoft=${uiControls.spawnMembraneCells ? 'ON' : 'OFF'}`,
    });
    await start();
    return;
  }
  if (uiControls.rigidBodyCount !== s.controls.rigidBodyCount) {
    running = false;
    sim = null;
    log({ ok: true, msg: `reinitializing for rigid body count ${uiControls.rigidBodyCount}` });
    await start();
    return;
  }

  s.controls = { ...s.controls, ...uiControls, n: s.controls.n };
  if (!EMBED_MODE && showSegmentIdsEl) s.overlayShowSegmentIds = !!showSegmentIdsEl.checked;
  if (!EMBED_MODE && showExtraVisualsEl) s.overlayShowExtraVisuals = !!showExtraVisualsEl.checked;
  uploadUniforms(s.device, s.uniform, s.controls);

  // Per-frame body obstacle stamping from BLOCK velocity edges (rigid + soft).
  s.lastFluidObstacleStats = stampBodyObstacleMask(s);
  if (s.obstacleMaskGpu && s.obstacleMaskCpu) {
    s.device.queue.writeBuffer(s.obstacleMaskGpu, 0, s.obstacleMaskCpu);
  }

  // Per-frame per-channel dye mask stamping from rigid + soft/membrane edges.
  s.lastDyeMaskStats = stampPerChannelDyeMask(s);
  if (s.dyeModeMaskGpu && s.dyeModeMaskCpu) {
    s.device.queue.writeBuffer(s.dyeModeMaskGpu, 0, s.dyeModeMaskCpu);
  }
  recordPipelineTiming(s, 'frame.controlsAndMaskStamp', performance.now() - frameStageStartMs);

  const fluidEncodeStartMs = performance.now();
  const enc = s.device.createCommandEncoder();

  let pass = null;
  if (!s.disableDefaultInject) {
    pass = enc.beginComputePass();
    pass.setPipeline(s.inject.pipeline);
    pass.setBindGroup(0, s.inject.bg([s.uniform, s.vx0, s.vy0, s.rr0, s.gg0, s.bb0]));
    pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
    pass.end();
  }

  pass = enc.beginComputePass();
  pass.setPipeline(s.advVel.pipeline);
  pass.setBindGroup(0, s.advVel.bg([s.uniform, s.vx0, s.vy0, s.vx1, s.vy1, s.viscMapGpu, s.obstacleMaskGpu]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();
  [s.vx0, s.vx1] = [s.vx1, s.vx0];
  [s.vy0, s.vy1] = [s.vy1, s.vy0];

  pass = enc.beginComputePass();
  pass.setPipeline(s.divPipe.pipeline);
  pass.setBindGroup(0, s.divPipe.bg([s.uniform, s.vx0, s.vy0, s.div, s.obstacleMaskGpu]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();

  for (let i = 0; i < JACOBI_ITERS; i++) {
    pass = enc.beginComputePass();
    pass.setPipeline(s.jacobiP.pipeline);
    pass.setBindGroup(0, s.jacobiP.bg([s.uniform, s.pr0, s.div, s.pr1, s.obstacleMaskGpu]));
    pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
    pass.end();
    [s.pr0, s.pr1] = [s.pr1, s.pr0];
  }

  pass = enc.beginComputePass();
  pass.setPipeline(s.project.pipeline);
  pass.setBindGroup(0, s.project.bg([s.uniform, s.vx0, s.vy0, s.pr0, s.obstacleMaskGpu]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(s.advDye.pipeline);
  pass.setBindGroup(0, s.advDye.bg([s.uniform, s.vx0, s.vy0, s.rr0, s.gg0, s.bb0, s.rr1, s.gg1, s.bb1, s.obstacleMaskGpu, s.dyeModeMaskGpu]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();
  [s.rr0, s.rr1] = [s.rr1, s.rr0];
  [s.gg0, s.gg1] = [s.gg1, s.gg0];
  [s.bb0, s.bb1] = [s.bb1, s.bb0];

  const solverPathNow = normalizeRuntimeSolverPath(s?.controls?.runtimeSolverPath);
  const pipelineModeNow = normalizeRuntimePipelineMode(s?.controls?.runtimePipelineMode, solverPathNow);
  const fastMode = solverPathNow === 'gpu-only' && pipelineModeNow === 'gpu-only-fast';

  const fallbackSignalsNow = fastMode ? collectGpuFallbackSignals(s) : [];
  const fallbackAnomaly = fastMode && fallbackSignalsNow.length > 0;
  const hasFastShadow = Boolean(
    s?.fastShadowFields
      && s.fastShadowFields.r
      && s.fastShadowFields.g
      && s.fastShadowFields.b
      && s.fastShadowFields.vx
      && s.fastShadowFields.vy,
  );

  const fastReadbackInterval = normalizeFastReadbackInterval(
    s?.controls?.fastReadbackInterval ?? s?.fastReadbackPolicy?.intervalFrames ?? FAST_MODE_FULL_READBACK_INTERVAL,
  );
  const cadenceDue = fastMode ? (((Number(s.frame) || 0) % fastReadbackInterval) === 0) : true;

  if (!fastMode) {
    enqueueFullFluidReadbackCopies(s, enc);
  } else if (s.fastFluidHealthPipeline && s.fastFluidHealthParams && s.fastFluidHealthStats && s.fastFluidHealthReadback) {
    const healthUniform = new ArrayBuffer(16);
    const healthU32 = new Uint32Array(healthUniform);
    const healthF32 = new Float32Array(healthUniform);
    healthU32[0] = s.cells;
    healthF32[1] = normalizeFluidCouplingComponentLimit(s.controls?.fluidCouplingComponentLimit);
    healthF32[2] = FAST_MODE_DYE_ABS_LIMIT;
    healthF32[3] = 0;
    s.device.queue.writeBuffer(s.fastFluidHealthParams, 0, healthUniform);
    s.device.queue.writeBuffer(s.fastFluidHealthStats, 0, FAST_MODE_HEALTH_CLEAR);

    pass = enc.beginComputePass();
    pass.setPipeline(s.fastFluidHealthPipeline.pipeline);
    pass.setBindGroup(0, s.fastFluidHealthPipeline.bg([
      s.fastFluidHealthParams,
      s.rr0,
      s.gg0,
      s.bb0,
      s.vx0,
      s.vy0,
      s.fastFluidHealthStats,
    ]));
    pass.dispatchWorkgroups(Math.ceil(s.cells / 64));
    pass.end();

    enc.copyBufferToBuffer(s.fastFluidHealthStats, 0, s.fastFluidHealthReadback, 0, 16);
  }

  recordPipelineTiming(s, 'fluid.encodeDispatch', performance.now() - fluidEncodeStartMs);
  const submitAndHealthStartMs = performance.now();
  s.device.queue.submit([enc.finish()]);

  let healthStats = { nonFiniteCount: 0, velocitySpikeCount: 0, dyeSpikeCount: 0 };
  if (fastMode) {
    healthStats = await readFastFluidHealthStats(s);
  }

  const explosiveVelocitySpikeCount = Math.max(4096, Math.floor(s.cells * 0.5));
  const explosiveDyeSpikeCount = Math.max(4096, Math.floor(s.cells * 0.5));
  const velocityExplosive = (healthStats.velocitySpikeCount || 0) >= explosiveVelocitySpikeCount;
  const dyeExplosive = (healthStats.dyeSpikeCount || 0) >= explosiveDyeSpikeCount;
  const healthAnomaly = fastMode
    && ((healthStats.nonFiniteCount > 0) || velocityExplosive || dyeExplosive);
  recordPipelineTiming(s, 'fluid.submitAndHealth', performance.now() - submitAndHealthStartMs);

  let doReadback = !fastMode || cadenceDue || fallbackAnomaly || healthAnomaly || !hasFastShadow;
  let readbackReason = !fastMode
    ? 'non-fast-mode'
    : (!hasFastShadow
      ? 'fast-bootstrap'
      : (fallbackAnomaly
        ? 'fast-fallback-anomaly'
        : (healthAnomaly
          ? 'fast-health-anomaly'
          : (cadenceDue ? 'fast-cadence' : 'fast-resident-skip'))));

  const pipelineCouplingLagEnabled = (
    solverPathNow === 'baseline'
    && s?.controls?.enableCouplingLagFrame === true
    && doReadback
  );

  if (!s.couplingLagRuntime || typeof s.couplingLagRuntime !== 'object') {
    s.couplingLagRuntime = {
      enabled: false,
      active: false,
      lagFrames: 0,
      bootstrapFrames: 0,
      reusedFrames: 0,
      lastMode: 'disabled',
      lastReadbackWaitMs: 0,
    };
  }
  s.couplingLagRuntime.enabled = pipelineCouplingLagEnabled;
  s.couplingLagRuntime.active = false;
  s.couplingLagRuntime.lastMode = pipelineCouplingLagEnabled ? 'pending' : 'disabled';
  if (!pipelineCouplingLagEnabled) {
    s.couplingLagRuntime.lagFrames = 0;
    s.laggedCouplingFields = null;
  }

  const readbackStageStartMs = performance.now();
  if (fastMode && doReadback) {
    const copyEnc = s.device.createCommandEncoder();
    enqueueFullFluidReadbackCopies(s, copyEnc);
    s.device.queue.submit([copyEnc.finish()]);
  }

  let fields = null;
  let lagReadbackPromise = null;
  if (pipelineCouplingLagEnabled) {
    lagReadbackPromise = mapFullFluidReadback(s);
    if (s.laggedCouplingFields) {
      fields = s.laggedCouplingFields;
      readbackReason = 'baseline-coupling-lag-reuse';
      s.couplingLagRuntime.active = true;
      s.couplingLagRuntime.reusedFrames = (Number(s.couplingLagRuntime.reusedFrames) || 0) + 1;
      s.couplingLagRuntime.lastMode = 'reused-prev-frame';
      s.couplingLagRuntime.lagFrames = 1;
    } else {
      fields = await lagReadbackPromise;
      s.laggedCouplingFields = fields;
      lagReadbackPromise = null;
      readbackReason = 'baseline-coupling-lag-bootstrap';
      s.couplingLagRuntime.bootstrapFrames = (Number(s.couplingLagRuntime.bootstrapFrames) || 0) + 1;
      s.couplingLagRuntime.lastMode = 'bootstrap-await';
      s.couplingLagRuntime.lagFrames = 1;
    }
  } else if (doReadback) {
    fields = await mapFullFluidReadback(s);
    s.fastShadowFields = fields;
  } else {
    fields = s.fastShadowFields;
  }
  recordPipelineTiming(s, doReadback ? 'fluid.readback.fullMap' : 'fluid.readback.shadowReuse', performance.now() - readbackStageStartMs, {
    reason: readbackReason,
  });

  if (!fields) {
    recordPipelineTiming(s, 'frame.abort.noFields', 0, { reason: 'no-fields-after-readback-decision' });
    finalizeFramePipelineTiming(s);
    s.frame += 1;
    if (running) requestAnimationFrame(() => stepAndRender());
    return;
  }

  if (!s.fastReadbackPolicy || typeof s.fastReadbackPolicy !== 'object') {
    s.fastReadbackPolicy = {};
  }
  s.fastReadbackPolicy.enabled = true;
  s.fastReadbackPolicy.intervalFrames = fastReadbackInterval;
  s.fastReadbackPolicy.lastHealth = healthStats;
  s.fastReadbackPolicy.lastHealthAnomaly = {
    healthAnomaly,
    velocityExplosive,
    dyeExplosive,
    explosiveVelocitySpikeCount,
    explosiveDyeSpikeCount,
  };
  s.fastReadbackPolicy.lastDecision = doReadback ? 'full-readback' : 'gpu-resident-skip';
  s.fastReadbackPolicy.lastReason = readbackReason;
  if (doReadback) {
    s.fastReadbackPolicy.lastFullReadbackFrame = Number(s.frame) || 0;
    s.fastReadbackPolicy.skippedFramesSinceFull = 0;
  } else {
    s.fastReadbackPolicy.skippedFramesSinceFull = (Number(s.fastReadbackPolicy.skippedFramesSinceFull) || 0) + 1;
  }

  const r = fields.r;
  const g = fields.g;
  const b = fields.b;
  const vx = fields.vx;
  const vy = fields.vy;

  // Rigid + soft coupling: carry/drag from flow + two-way pushback/swim impulses.
  const bodiesStageStartMs = performance.now();
  const couplingInstant = await stepBodiesAndInject(s, vx, vy);
  recordPipelineTiming(s, 'frame.stepBodiesAndInject.total', performance.now() - bodiesStageStartMs);

    const postCouplingCpuStartMs = performance.now();
    applyEmitters(s, r, g, b, vx, vy);
    const obstacleEdgesNow = Number(s?.lastFluidObstacleStats?.blockedEdgeCount) || 0;
    applyDigestiveCapture(s, r, g, b);
    const sweptDyeTransportNow = redistributeSweptEdgeDyeTransport(s, r, g, b);
    s.lastSweptDyeTransportStats = sweptDyeTransportNow;
    enforceFluidEdgeBoundariesCpu(vx, vy, s.controls.n);

    // Last-resort guardrail: prevent non-finite/unsafe velocity components from
    // being re-uploaded into the next GPU fluid step.
    const couplingLimit = normalizeFluidCouplingComponentLimit(s.controls?.fluidCouplingComponentLimit);
    for (let i = 0; i < s.cells; i++) {
      const vxi = Number(vx[i]);
      const vyi = Number(vy[i]);
      vx[i] = clampFluidComponent(vxi, couplingLimit);
      vy[i] = clampFluidComponent(vyi, couplingLimit);
    }
    recordPipelineTiming(s, 'frame.postCouplingCpu', performance.now() - postCouplingCpuStartMs);

    const writebackStartMs = performance.now();
    if (doReadback) {
      s.device.queue.writeBuffer(s.vx0, 0, vx);
      s.device.queue.writeBuffer(s.vy0, 0, vy);
      s.device.queue.writeBuffer(s.rr0, 0, r);
      s.device.queue.writeBuffer(s.gg0, 0, g);
      s.device.queue.writeBuffer(s.bb0, 0, b);
    }

    recordPipelineTiming(s, doReadback ? 'frame.gpuWriteback.updatedFields' : 'frame.gpuWriteback.skipped', performance.now() - writebackStartMs);

    if (lagReadbackPromise) {
      const lagRefreshWaitStartMs = performance.now();
      const nextLagFields = await lagReadbackPromise;
      const lagRefreshWaitMs = performance.now() - lagRefreshWaitStartMs;
      s.laggedCouplingFields = nextLagFields;
      s.couplingLagRuntime.lastReadbackWaitMs = lagRefreshWaitMs;
      s.couplingLagRuntime.lagFrames = 1;
      recordPipelineTiming(s, 'fluid.readback.pipelineLag.refreshWait', lagRefreshWaitMs, {
        reason: 'baseline-coupling-lag-refresh',
      });
    }

    const renderStageStartMs = performance.now();
    const n = s.controls.n;
    const img = ctx.createImageData(n, n);
    const px = img.data;
    let sum = 0;
    const showObstacleMaskOverlay = s.controls?.showObstacleMaskOverlay !== false;
    for (let i = 0; i < s.cells; i++) {
      const ri = Math.max(0, Math.min(255, r[i]));
      const gi = Math.max(0, Math.min(255, g[i]));
      const bi = Math.max(0, Math.min(255, b[i]));
      const obstacle = Number(s?.obstacleMaskCpu?.[i]) > 0.5;
      const o = i * 4;
      if (showObstacleMaskOverlay && obstacle) {
        // Obstacle-mask visualization overlay (BLOCK velocity edges influence region).
        px[o] = Math.max(ri, 220);
        px[o + 1] = Math.min(gi, 70);
        px[o + 2] = Math.min(bi, 90);
      } else {
        px[o] = ri;
        px[o + 1] = gi;
        px[o + 2] = bi;
      }
      px[o + 3] = 255;
      sum += ri + gi + bi;
    }

    const tmp = document.createElement('canvas');
    tmp.width = n; tmp.height = n;
    tmp.getContext('2d').putImageData(img, 0, 0);
    const view = getCameraView(s);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGridTextureToView(tmp, n, view, getWorldScale(s.controls));
    if (!showViscEl || showViscEl.checked) drawViscosityOverlay();
    drawBodiesOverlay(s);
    recordPipelineTiming(s, 'frame.renderCompose', performance.now() - renderStageStartMs);

    const hudStageStartMs = performance.now();
    const elapsed = (performance.now() - s.t0) / 1000;
    const fpsNow = +(s.frame / Math.max(1e-6, elapsed)).toFixed(1);
    if (fpsHud) fpsHud.textContent = `FPS: ${fpsNow}`;
    if (softIntegrateHudEl) {
      const hud = softIntegrateHudLabel(s.softIntegrateRuntime);
      softIntegrateHudEl.textContent = hud.text;
      softIntegrateHudEl.style.color = hud.color;
    }
    updateFallbackHud(s);
    const couplingAverages = summarizeCouplingTelemetry(s.couplingTelemetry);
    const couplingSnapshot = { ...couplingAverages, ...Object.fromEntries(Object.entries(couplingInstant || {}).map(([k,v]) => [k+'Now', +((v || 0).toFixed(4))])) };
    const rigidContacts = Array.isArray(s.lastRigidContacts) ? s.lastRigidContacts : [];
    const softEdgePoliciesNow = summarizeSoftEdgePolicies(s?.bodies?.soft?.springs || []);
    window.__gpuLabCoupling = couplingSnapshot;
    window.__gpuLabRigidContacts = rigidContacts;

    if (!EMBED_MODE) {
      log({
        ok: true,
        mode: 'live-fluid',
        grid: n,
        frames: s.frame,
        fps: fpsNow,
        fastGpuResidentMode: fastMode,
        fastGpuResidentIntervalFrames: fastReadbackInterval,
        fastGpuResidentSkippedFrames: Number(s?.fastReadbackPolicy?.skippedFramesSinceFull) || 0,
        fastGpuResidentLastDecision: s?.fastReadbackPolicy?.lastDecision || null,
        fastGpuResidentLastReason: s?.fastReadbackPolicy?.lastReason || null,
        fastGpuResidentReadbackThisFrame: doReadback,
        fastGpuResidentHealth: healthStats,
        softIntegrateMode: s.softIntegrateRuntime?.mode || 'unknown',
        softIntegrateReason: s.softIntegrateRuntime?.reason || 'n/a',
        dyeEnergy: +sum.toFixed(1),
        viscosityScale: s.controls.viscosity,
        fluidVelocityCap: normalizeFluidVelocityCap(s.controls.fluidVelocityCap),
        massLight: s.controls.massLight,
        massHeavy: s.controls.massHeavy,
        massSoft: s.controls.massSoft,
        bodyDrag: s.controls.bodyDrag,
        bodyFeedback: s.controls.bodyFeedback,
        fluidCouplingComponentLimit: normalizeFluidCouplingComponentLimit(s.controls.fluidCouplingComponentLimit),
        softSpringStiffness: normalizeSoftSpringStiffness(s.controls.softSpringStiffness),
        softSpringBaseCompliance: normalizeCompliance(s.controls.softSpringBaseCompliance, SOFT_XPBD_BASE_COMPLIANCE),
        softAreaBaseCompliance: normalizeCompliance(s.controls.softAreaBaseCompliance, SOFT_AREA_BASE_COMPLIANCE),
        membraneEdgeBaseCompliance: normalizeCompliance(s.controls.membraneEdgeBaseCompliance, MEMBRANE_EDGE_BASE_COMPLIANCE),
        membraneBendBaseCompliance: normalizeCompliance(s.controls.membraneBendBaseCompliance, MEMBRANE_BEND_BASE_COMPLIANCE),
        enableArtificialSwim: !!s.controls.enableArtificialSwim,
        softClusterFluidTorqueCoupling: s.controls.softClusterFluidTorqueCoupling,
        softClusterAngularProjection: s.controls.softClusterAngularProjection,
        softClusterCollisionAngularProjection: s.controls.softClusterCollisionAngularProjection,
        seedRigidBodies: s.controls.seedRigidBodies !== false,
        seedSpringSoftBodies: s.controls.seedSpringSoftBodies !== false,
        spawnMembraneCells: !!s.controls.spawnMembraneCells,
        warningInterventions: s.controls.enableWarningDeformInterventions !== false,
        severeInterventions: s.controls.enableSevereDeformInterventions !== false,
        postCollisionRecovery: s.controls.enablePostCollisionRecovery !== false,
        softClusterStabilizers: s.controls.enableSoftClusterStabilizers !== false,
        softSpringRestRecovery: s.controls.enableSoftSpringRestRecovery !== false,
        membraneBoundaryXpbd: s.controls.enableMembraneBoundaryXpbd !== false,
        membraneShapeMemory: s.controls.enableMembraneShapeMemory !== false,
        membranePressure: s.controls.enableMembranePressure !== false,
        membraneClusters: Array.isArray(s.bodies?.softMembraneClusters) ? s.bodies.softMembraneClusters.length : 0,
        fluidObstacleEdgesNow: obstacleEdgesNow,
        fluidObstacleCellsNow: Number(s?.lastFluidObstacleStats?.blockedCells) || 0,
        fluidObstacleSweptCellsNow: Number(s?.lastFluidObstacleStats?.sweptCells) || 0,
        dyeMaskCellsNow: Number(s?.lastDyeMaskStats?.nonPassCells) || 0,
        sweptDyeTouchedCellsNow: Number(sweptDyeTransportNow?.touchedCells) || 0,
        sweptDyeMovedMassNow: +((Number(sweptDyeTransportNow?.movedMass) || 0).toFixed(2)),
        sweptDyeDeletedMassNow: +((Number(sweptDyeTransportNow?.deletedMass) || 0).toFixed(2)),
        sweptDyeEatenMassNow: +((Number(sweptDyeTransportNow?.eatenMass) || 0).toFixed(2)),
        softEdgePoliciesNow,
        paintValue: Number(paintValueEl?.value) || 0.85,
        brushSize: Number(brushSizeEl?.value) || 12,
        digestiveCapture: +((s.digestiveCapture || 0).toFixed(2)),
        coupling: couplingSnapshot,
        rigidContactCount: rigidContacts.length,
        rigidContactPairs: (showCollisionHullEl?.checked ? rigidContacts.slice(0, 8) : undefined),
        droppedSoftSprings: s.bodies?.topologyGuardrails?.droppedSoftSprings || 0,
        softDeformation: {
          warningClustersNow: sim.softDeformationState?.warningCount || 0,
          severeClustersNow: sim.softDeformationState?.severeCount || 0,
          severeCollapseClustersNow: sim.softDeformationState?.severeCollapseCount || 0,
          worstStretchNow: +((sim.softDeformationState?.worstStretch || 1).toFixed(3)),
          worstAreaRatioNow: +((sim.softDeformationState?.worstAreaRatio || 1).toFixed(3)),
          worstPoseErrorNow: +((sim.softDeformationState?.worstPoseError || 0).toFixed(3)),
          severeEventsRecent: (sim.softDeformationEvents || []).slice(-4),
        },
      });
    }
    recordPipelineTiming(s, 'frame.hudAndStatus', performance.now() - hudStageStartMs);
    finalizeFramePipelineTiming(s);

  s.frame += 1;
  if (running) requestAnimationFrame(() => stepAndRender());
}

async function start() {
  if (running) return;
  running = true;
  if (fpsHud) fpsHud.textContent = 'FPS: --';

  refreshAllSliderReadouts();
  const selectedPreset = scenarioPresetEl?.value || 'baseline';
  if (selectedPreset.startsWith('mini:') && generatedMiniScenarios.size === 0) {
    await loadGeneratedMiniScenarios();
  }
  ensureMiniScenarioGridSelection(selectedPreset);

  sim = await initSim();
  applyScenarioPreset(sim, selectedPreset);
  if (pendingImportedSpecs.length) {
    let lastImported = null;
    for (const entry of pendingImportedSpecs) {
      const spec = entry?.spec || entry;
      const importScale = entry?.importScale ?? 0.1;
      const imported = buildBodiesFromCreatureSpec(spec, sim.controls.n, sim.controls);
      imported.hybrid = [];
      scaleImportedBodies(imported, importScale);
      mergeBodiesIntoSim(sim.bodies, imported);
      lastImported = imported;
    }
    if (lastImported) focusCameraOnBodies(sim, lastImported);
    pendingImportedSpecs = [];
  }
  log('starting live GPU fluid sim...');
  stepAndRender().catch((e) => {
    running = false;
    log({ ok: false, error: String(e) });
  });
}

function stop() {
  running = false;
  if (fpsHud) fpsHud.textContent = 'FPS: --';
  log('stopped');
}

bindSliderReadouts();

copyFallbacksBtn?.addEventListener('click', () => {
  copyFallbackReportToClipboard();
});

clearFallbacksBtn?.addEventListener('click', () => {
  clearFallbackHistory();
});

setFallbackCopyStatus('ready', '#9faec7');

for (const solverEl of runtimeSolverPathEls) {
  solverEl?.addEventListener('change', async () => {
    if (!solverEl.checked) return;
    const selected = setRuntimeSolverPath(solverEl.value, { syncUrl: true });
    if (sim?.controls) {
      sim.controls.runtimeSolverPath = selected;
      sim.controls.runtimePipelineMode = getRuntimePipelineMode(selected);
    }
    if (!running || !sim) return;
    running = false;
    sim = null;
    await start();
  });
}


for (const pipelineEl of runtimePipelineModeEls) {
  pipelineEl?.addEventListener('change', async () => {
    if (!pipelineEl.checked) return;
    const selectedPath = getRuntimeSolverPath();
    const selectedMode = setRuntimePipelineMode(pipelineEl.value, { syncUrl: true, solverPath: selectedPath });
    if (sim?.controls) sim.controls.runtimePipelineMode = selectedMode;
    if (!running || !sim || selectedPath !== 'gpu-only') return;
    running = false;
    sim = null;
    await start();
  });
}

rigidRigidCandidateBackendEl?.addEventListener('change', async () => {
  const selectedBackend = setRigidRigidCandidateBackendMode(rigidRigidCandidateBackendEl.value, { syncUrl: true });
  if (sim?.controls) sim.controls.rigidRigidCandidateBackendMode = selectedBackend;
  if (!running || !sim) return;
  running = false;
  sim = null;
  await start();
});

runBtn.addEventListener('click', () => start().catch((e) => log({ ok: false, error: String(e) })));
stopBtn.addEventListener('click', stop);
clearViscBtn.addEventListener('click', resetViscMap);
if (scenarioPresetEl) {
  scenarioPresetEl.addEventListener('change', async () => {
    const preset = scenarioPresetEl.value || 'baseline';
    const mini = ensureMiniScenarioGridSelection(preset);
    if (!sim) return;
    if (mini && sim.controls.n !== Number(mini.grid || sim.controls.n)) {
      running = false;
      sim = null;
      await start();
      return;
    }
    applyScenarioPreset(sim, preset);
    refreshAllSliderReadouts();
  });
}

if (importSpecBtn && importSpecFile) {
  importSpecBtn.addEventListener('click', () => importSpecFile.click());
  importSpecFile.addEventListener('change', async () => {
    const f = importSpecFile.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      const spec = parseCreatureSpec(text);
      const targetN = readControls().n;

      // Important: if a stale sim instance exists at a different grid,
      // importing directly into it gets wiped by the next reinit.
      const simMatchesTargetGrid = !!sim && sim.controls.n === targetN;

      const importScale = readImportScale();
      if (simMatchesTargetGrid) {
        const imported = buildBodiesFromCreatureSpec(spec, sim.controls.n, sim.controls);
        imported.hybrid = [];
        scaleImportedBodies(imported, importScale);
        mergeBodiesIntoSim(sim.bodies, imported);
        sim.softDeformReferenceState = null;
        focusCameraOnBodies(sim, imported);
        log({
          ok: true,
          msg: 'CreatureSpec imported (appended)',
          name: spec.name || 'unnamed',
          grid: sim.controls.n,
          importScale,
          rigidAdded: imported.rigid?.length || 0,
          softNodesAdded: imported.soft?.nodes?.length || 0,
        });
      } else {
        pendingImportedSpecs.push({ spec, importScale });
        log({
          ok: true,
          msg: 'CreatureSpec queued for import on next start/reinit at selected grid',
          name: spec.name || 'unnamed',
          selectedGrid: targetN,
          currentGrid: sim?.controls?.n ?? null,
          importScale,
        });
      }
    } catch (e) {
      log({ ok: false, error: `Import failed: ${String(e)}` });
    } finally {
      importSpecFile.value = '';
    }
  });
}

function captureCanvasMetrics() {
  if (!(canvas instanceof HTMLCanvasElement) || !ctx) {
    return {
      ok: false,
      error: 'canvas-not-ready',
      frame: Number(sim?.frame) || 0,
      totalRGB: 0,
      nonZeroPixels: 0,
      width: 0,
      height: 0,
    };
  }
  const w = Math.max(0, Number(canvas.width) || 0);
  const h = Math.max(0, Number(canvas.height) || 0);
  if (w <= 0 || h <= 0) {
    return {
      ok: false,
      error: 'canvas-size-zero',
      frame: Number(sim?.frame) || 0,
      totalRGB: 0,
      nonZeroPixels: 0,
      width: w,
      height: h,
    };
  }

  try {
    const image = ctx.getImageData(0, 0, w, h);
    const data = image?.data || null;
    if (!data || data.length === 0) {
      return {
        ok: false,
        error: 'image-data-empty',
        frame: Number(sim?.frame) || 0,
        totalRGB: 0,
        nonZeroPixels: 0,
        width: w,
        height: h,
      };
    }

    let totalRGB = 0;
    let nonZeroPixels = 0;
    for (let i = 0; i < data.length; i += 4) {
      const sum = (data[i] || 0) + (data[i + 1] || 0) + (data[i + 2] || 0);
      totalRGB += sum;
      if (sum > 0) nonZeroPixels += 1;
    }

    return {
      ok: true,
      frame: Number(sim?.frame) || 0,
      totalRGB,
      nonZeroPixels,
      width: w,
      height: h,
      capturedAt: new Date().toISOString(),
    };
  } catch (err) {
    return {
      ok: false,
      error: String(err?.message || err),
      frame: Number(sim?.frame) || 0,
      totalRGB: 0,
      nonZeroPixels: 0,
      width: w,
      height: h,
    };
  }
}

window.__gpuLabApi = {
  start: () => start(),
  stop: () => { stop(); return { ok: true }; },
  resetWindTunnelFromSpec: (spec, options = {}) => resetEmbedWindTunnelFromSpec(spec, options),
  getStatus: () => {
    const il = sim?.interactionLab || null;
    let fixturePose = null;
    if (il?.targetType === 'soft') {
      const ids = Array.isArray(il?.softNodeIndices) ? il.softNodeIndices : [];
      const nodes = sim?.bodies?.soft?.nodes || [];
      let sx = 0; let sy = 0; let svx = 0; let svy = 0; let c = 0;
      for (const idx of ids) {
        const n = nodes[idx | 0];
        if (!n) continue;
        sx += Number(n.x) || 0;
        sy += Number(n.y) || 0;
        svx += Number(n.vx) || 0;
        svy += Number(n.vy) || 0;
        c += 1;
      }
      if (c > 0) {
        fixturePose = { x: sx / c, y: sy / c, vx: svx / c, vy: svy / c };
      }
    } else {
      const rb = sim?.bodies?.rigid?.[Number(il?.rigidIndex) | 0] || null;
      if (rb) {
        fixturePose = {
          x: Number(rb.x) || 0,
          y: Number(rb.y) || 0,
          vx: Number(rb.vx) || 0,
          vy: Number(rb.vy) || 0,
        };
      }
    }

    return {
      ok: true,
      running: !!running,
      frame: Number(sim?.frame) || 0,
      grid: Number(sim?.controls?.n) || null,
      worldScale: getWorldScale(sim?.controls),
      worldSize: getWorldSize(sim?.controls),
      runtimeSolverPath: normalizeRuntimeSolverPath(sim?.controls?.runtimeSolverPath),
      runtimePipelineMode: normalizeRuntimePipelineMode(sim?.controls?.runtimePipelineMode, sim?.controls?.runtimeSolverPath),
      rigidRigidCandidateBackend: {
        mode: sim?.rigidRigidCandidateBackendMode || 'js',
        reason: sim?.rigidRigidCandidateBackendReason || null,
      },
      postCollisionProjectionBackend: {
        mode: sim?.postCollisionProjectionBackendMode || 'js',
        reason: sim?.postCollisionProjectionBackendReason || null,
      },
      rigidStepBaselineBackend: {
        mode: sim?.rigidStepBaselineBackendMode || 'js',
        reason: sim?.rigidStepBaselineBackendReason || null,
      },
      allowPassEdgeFlowPush: sim?.controls?.allowPassEdgeFlowPush === true,
      enableCouplingLagFrame: sim?.controls?.enableCouplingLagFrame === true,
      rendererFingerprint: sim?.rendererFingerprint
        ? {
          userAgent: sim.rendererFingerprint.userAgent || null,
          webglVendor: sim.rendererFingerprint.webglVendor || null,
          webglRenderer: sim.rendererFingerprint.webglRenderer || null,
          isSwiftShader: sim.rendererFingerprint.isSwiftShader === true,
          webgpuAdapter: sim.rendererFingerprint.webgpuAdapter
            ? {
              vendor: sim.rendererFingerprint.webgpuAdapter.vendor || null,
              architecture: sim.rendererFingerprint.webgpuAdapter.architecture || null,
              device: sim.rendererFingerprint.webgpuAdapter.device || null,
              description: sim.rendererFingerprint.webgpuAdapter.description || null,
            }
            : null,
        }
        : {
          userAgent: null,
          webglVendor: null,
          webglRenderer: null,
          isSwiftShader: false,
          webgpuAdapter: null,
        },
      fastReadbackRuntime: {
        enabled: !!sim?.fastReadbackPolicy?.enabled,
        intervalFrames: Number(sim?.fastReadbackPolicy?.intervalFrames) || FAST_MODE_FULL_READBACK_INTERVAL,
        configuredIntervalFrames: normalizeFastReadbackInterval(sim?.controls?.fastReadbackInterval),
        skippedFramesSinceFull: Number(sim?.fastReadbackPolicy?.skippedFramesSinceFull) || 0,
        lastFullReadbackFrame: Number(sim?.fastReadbackPolicy?.lastFullReadbackFrame ?? -1),
        lastDecision: sim?.fastReadbackPolicy?.lastDecision || null,
        lastReason: sim?.fastReadbackPolicy?.lastReason || null,
        lastHealth: sim?.fastReadbackPolicy?.lastHealth || null,
        lastHealthAnomaly: sim?.fastReadbackPolicy?.lastHealthAnomaly || null,
      },
      couplingLagRuntime: sim?.couplingLagRuntime
        ? {
          enabled: sim.couplingLagRuntime.enabled === true,
          active: sim.couplingLagRuntime.active === true,
          lagFrames: Number(sim.couplingLagRuntime.lagFrames) || 0,
          bootstrapFrames: Number(sim.couplingLagRuntime.bootstrapFrames) || 0,
          reusedFrames: Number(sim.couplingLagRuntime.reusedFrames) || 0,
          lastMode: sim.couplingLagRuntime.lastMode || 'unknown',
          lastReadbackWaitMs: Number(sim.couplingLagRuntime.lastReadbackWaitMs) || 0,
        }
        : {
          enabled: false,
          active: false,
          lagFrames: 0,
          bootstrapFrames: 0,
          reusedFrames: 0,
          lastMode: 'uninitialized',
          lastReadbackWaitMs: 0,
        },
      pipelineTiming: {
        last: sim?.pipelineTimingLast
          ? {
            ...sim.pipelineTimingLast,
            events: Array.isArray(sim?.pipelineTimingLast?.events)
              ? sim.pipelineTimingLast.events.slice(-160)
              : [],
          }
          : null,
        recentFrames: Array.isArray(sim?.pipelineTimingHistory)
          ? sim.pipelineTimingHistory.slice(-16).map((entry) => ({
            frame: Number(entry?.frame) || 0,
            solverPath: entry?.solverPath || 'baseline',
            pipelineMode: entry?.pipelineMode || 'standard',
            totalMs: Number(entry?.totalMs) || 0,
            spans: entry?.spans ? { ...entry.spans } : {},
          }))
          : [],
      },
      fallbackHud: sim?.fallbackHudSummary || {
        active: normalizeRuntimeSolverPath(sim?.controls?.runtimeSolverPath) === 'gpu-only',
        currentCount: 0,
        hasRecent: false,
        lastFallbackFrame: null,
        currentStages: [],
        recentStages: [],
        currentEntries: [],
        recentEntries: [],
        reportText: '',
      },
      rigidBodies: Number(sim?.bodies?.rigid?.length) || 0,
      softNodes: Number(sim?.bodies?.soft?.nodes?.length) || 0,
      interactionLab: il ? {
        enabled: il.enabled !== false,
        targetType: il.targetType,
        motion: il.mode,
        circleRadius: Number(il.circleRadius) || 0,
        circleAngularSpeed: Number(il.circleAngularSpeed) || 0,
        fixturePose,
      } : null,
      fluidObstacle: {
        blockedCells: Number(sim?.lastFluidObstacleStats?.blockedCells) || 0,
        sweptCells: Number(sim?.lastFluidObstacleStats?.sweptCells) || 0,
      },
      sweptDyeTransport: {
        touchedCells: Number(sim?.lastSweptDyeTransportStats?.touchedCells) || 0,
        movedMass: Number(sim?.lastSweptDyeTransportStats?.movedMass) || 0,
        deletedMass: Number(sim?.lastSweptDyeTransportStats?.deletedMass) || 0,
        eatenMass: Number(sim?.lastSweptDyeTransportStats?.eatenMass) || 0,
      },
      rigidStepRuntime: {
        lastSourceRoute: sim?.rigidStepWgslState?.lastSourceRoute || null,
        lastMode: sim?.rigidStepWgslState?.lastMode || null,
        lastRigidStepAuthoritativeSource: sim?.rigidStepWgslState?.lastRigidStepAuthoritativeSource || null,
        lastRigidStepAuthoritativeSignature: sim?.rigidStepWgslState?.lastRigidStepAuthoritativeSignature || null,
        lastRigidStepProposalSource: sim?.rigidStepWgslState?.lastRigidStepProposalSource || null,
        lastRigidStepProposalError: sim?.rigidStepWgslState?.lastRigidStepProposalError || null,
        lastRigidStepProposalLayoutBytes: Number(sim?.rigidStepWgslState?.lastRigidStepProposalLayoutBytes) || 0,
        lastRigidStepProposalSignature: sim?.rigidStepWgslState?.lastRigidStepProposalSignature || null,
        lastRigidStepProposalSignaturePrepared: sim?.rigidStepWgslState?.lastRigidStepProposalSignaturePrepared || null,
        lastRigidStepProposalVxLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalVx?.length) || 0,
        lastRigidStepProposalVyLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalVy?.length) || 0,
        lastRigidStepProposalOmegaLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalOmega?.length) || 0,
        lastRigidStepProposalXLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalX?.length) || 0,
        lastRigidStepProposalYLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalY?.length) || 0,
        lastRigidStepProposalThetaLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalTheta?.length) || 0,
        lastRigidStepProposalCarryLen: Number(sim?.rigidStepWgslState?.lastRigidStepProposalCarry?.length) || 0,
        lastTiming: sim?.rigidStepWgslState?.lastTiming ? { ...sim.rigidStepWgslState.lastTiming } : null,
      },
      rigidStepBaselineRuntime: {
        mode: sim?.rigidStepBaselineRuntime?.mode || 'uninitialized',
        reason: sim?.rigidStepBaselineRuntime?.reason || null,
        processedBodies: Number(sim?.rigidStepBaselineRuntime?.processedBodies) || 0,
        sampleCount: Number(sim?.rigidStepBaselineRuntime?.sampleCount) || 0,
        jsMarshalMs: Number(sim?.rigidStepBaselineRuntime?.jsMarshalMs) || 0,
        backendComputeMs: Number(sim?.rigidStepBaselineRuntime?.backendComputeMs) || 0,
        totalMs: Number(sim?.rigidStepBaselineRuntime?.totalMs) || 0,
      },
      postCollisionProjectionRuntime: {
        mode: sim?.postCollisionProjectionRuntime?.mode || 'uninitialized',
        reason: sim?.postCollisionProjectionRuntime?.reason || null,
        projectedNodes: Number(sim?.postCollisionProjectionRuntime?.projectedNodes) || 0,
        meanDelta: Number(sim?.postCollisionProjectionRuntime?.meanDelta) || 0,
        jsMarshalMs: Number(sim?.postCollisionProjectionRuntime?.jsMarshalMs) || 0,
        backendComputeMs: Number(sim?.postCollisionProjectionRuntime?.backendComputeMs) || 0,
        totalMs: Number(sim?.postCollisionProjectionRuntime?.totalMs) || 0,
      },
      rigidRigidBroadphaseRuntime: sim?.rigidRigidBroadphaseRuntime
        ? {
          enabled: sim.rigidRigidBroadphaseRuntime.enabled !== false,
          reason: sim.rigidRigidBroadphaseRuntime.reason || null,
          cellSize: Number(sim.rigidRigidBroadphaseRuntime.cellSize) || RIGID_RIGID_SPATIAL_HASH_CELL_SIZE,
          backendMode: sim.rigidRigidBroadphaseRuntime.backendMode || 'js',
          backendReason: sim.rigidRigidBroadphaseRuntime.backendReason || null,
          backendLabel: sim.rigidRigidBroadphaseRuntime.backendLabel || null,
          totalBruteForcePairs: Number(sim.rigidRigidBroadphaseRuntime.totalBruteForcePairs) || 0,
          totalCheckedPairs: Number(sim.rigidRigidBroadphaseRuntime.totalCheckedPairs) || 0,
          totalPrunedPairs: Number(sim.rigidRigidBroadphaseRuntime.totalPrunedPairs) || 0,
          reductionPct: Number(sim.rigidRigidBroadphaseRuntime.reductionPct) || 0,
          reuseAttempts: Number(sim.rigidRigidBroadphaseRuntime.reuseAttempts) || 0,
          reuseHits: Number(sim.rigidRigidBroadphaseRuntime.reuseHits) || 0,
          reuseMisses: Number(sim.rigidRigidBroadphaseRuntime.reuseMisses) || 0,
          reuseHitRatePct: Number(sim.rigidRigidBroadphaseRuntime.reuseHitRatePct) || 0,
          dedupeMs: Number(sim.rigidRigidBroadphaseRuntime.dedupeMs) || 0,
          sortMs: Number(sim.rigidRigidBroadphaseRuntime.sortMs) || 0,
          totalBuildMs: Number(sim.rigidRigidBroadphaseRuntime.totalBuildMs) || 0,
          jsMarshalMs: Number(sim.rigidRigidBroadphaseRuntime.jsMarshalMs) || 0,
          backendComputeMs: Number(sim.rigidRigidBroadphaseRuntime.backendComputeMs) || 0,
          emitAttempts: Number(sim.rigidRigidBroadphaseRuntime.emitAttempts) || 0,
          duplicatesRejected: Number(sim.rigidRigidBroadphaseRuntime.duplicatesRejected) || 0,
          pairsOut: Number(sim.rigidRigidBroadphaseRuntime.pairsOut) || 0,
          postPassSkipped: Number(sim.rigidRigidBroadphaseRuntime.postPassSkipped) || 0,
          phases: Array.isArray(sim?.rigidRigidBroadphaseRuntime?.phases)
            ? sim.rigidRigidBroadphaseRuntime.phases.slice(-8).map((entry) => ({
              phase: entry?.phase || 'unknown',
              iter: Number(entry?.iter) || 0,
              bodyCount: Number(entry?.bodyCount) || 0,
              bruteForcePairs: Number(entry?.bruteForcePairs) || 0,
              checkedPairs: Number(entry?.checkedPairs) || 0,
              prunedPairs: Number(entry?.prunedPairs) || 0,
              reductionPct: Number(entry?.reductionPct) || 0,
              occupiedCells: Number(entry?.occupiedCells) || 0,
              maxBodiesPerCell: Number(entry?.maxBodiesPerCell) || 0,
              avgBodiesPerCell: Number(entry?.avgBodiesPerCell) || 0,
              emitAttempts: Number(entry?.emitAttempts) || 0,
              duplicatesRejected: Number(entry?.duplicatesRejected) || 0,
              pairsOut: Number(entry?.pairsOut) || 0,
              dedupeMs: Number(entry?.dedupeMs) || 0,
              sortMs: Number(entry?.sortMs) || 0,
              totalBuildMs: Number(entry?.totalBuildMs) || 0,
              jsMarshalMs: Number(entry?.jsMarshalMs) || 0,
              backendComputeMs: Number(entry?.backendComputeMs) || 0,
              backendMarshalPath: entry?.backendMarshalPath || null,
              backend: entry?.backend || 'js',
              reuseHit: entry?.reuseHit === true,
              reuseMiss: entry?.reuseMiss === true,
            }))
            : [],
        }
        : {
          enabled: false,
          reason: 'uninitialized',
          cellSize: RIGID_RIGID_SPATIAL_HASH_CELL_SIZE,
          backendMode: 'js',
          backendReason: null,
          backendLabel: null,
          totalBruteForcePairs: 0,
          totalCheckedPairs: 0,
          totalPrunedPairs: 0,
          reductionPct: 0,
          reuseAttempts: 0,
          reuseHits: 0,
          reuseMisses: 0,
          reuseHitRatePct: 0,
          dedupeMs: 0,
          sortMs: 0,
          totalBuildMs: 0,
          jsMarshalMs: 0,
          backendComputeMs: 0,
          emitAttempts: 0,
          duplicatesRejected: 0,
          pairsOut: 0,
          postPassSkipped: 0,
          phases: [],
        },
      rigidSoftBroadphaseRuntime: sim?.rigidSoftBroadphaseRuntime
        ? {
          enabled: sim.rigidSoftBroadphaseRuntime.enabled !== false,
          reason: sim.rigidSoftBroadphaseRuntime.reason || null,
          cellSize: Number(sim.rigidSoftBroadphaseRuntime.cellSize) || RIGID_SOFT_SPATIAL_HASH_CELL_SIZE,
          totalRawNodeChecks: Number(sim.rigidSoftBroadphaseRuntime.totalRawNodeChecks) || 0,
          totalCandidateNodeChecks: Number(sim.rigidSoftBroadphaseRuntime.totalCandidateNodeChecks) || 0,
          totalPrunedNodeChecks: Number(sim.rigidSoftBroadphaseRuntime.totalPrunedNodeChecks) || 0,
          nodeReductionPct: Number(sim.rigidSoftBroadphaseRuntime.nodeReductionPct) || 0,
          totalRawEdgeChecks: Number(sim.rigidSoftBroadphaseRuntime.totalRawEdgeChecks) || 0,
          totalCandidateEdgeChecks: Number(sim.rigidSoftBroadphaseRuntime.totalCandidateEdgeChecks) || 0,
          totalPrunedEdgeChecks: Number(sim.rigidSoftBroadphaseRuntime.totalPrunedEdgeChecks) || 0,
          edgeReductionPct: Number(sim.rigidSoftBroadphaseRuntime.edgeReductionPct) || 0,
          totalBlockedEdges: Number(sim.rigidSoftBroadphaseRuntime.totalBlockedEdges) || 0,
          phases: Array.isArray(sim?.rigidSoftBroadphaseRuntime?.phases)
            ? sim.rigidSoftBroadphaseRuntime.phases.slice(-8).map((entry) => ({
              iter: Number(entry?.iter) || 0,
              rigidCount: Number(entry?.rigidCount) || 0,
              nodeCount: Number(entry?.nodeCount) || 0,
              blockedEdgeCount: Number(entry?.blockedEdgeCount) || 0,
              rawNodeChecks: Number(entry?.rawNodeChecks) || 0,
              candidateNodeChecks: Number(entry?.candidateNodeChecks) || 0,
              prunedNodeChecks: Number(entry?.prunedNodeChecks) || 0,
              nodeReductionPct: Number(entry?.nodeReductionPct) || 0,
              rawEdgeChecks: Number(entry?.rawEdgeChecks) || 0,
              candidateEdgeChecks: Number(entry?.candidateEdgeChecks) || 0,
              prunedEdgeChecks: Number(entry?.prunedEdgeChecks) || 0,
              edgeReductionPct: Number(entry?.edgeReductionPct) || 0,
            }))
            : [],
        }
        : {
          enabled: false,
          reason: 'uninitialized',
          cellSize: RIGID_SOFT_SPATIAL_HASH_CELL_SIZE,
          totalRawNodeChecks: 0,
          totalCandidateNodeChecks: 0,
          totalPrunedNodeChecks: 0,
          nodeReductionPct: 0,
          totalRawEdgeChecks: 0,
          totalCandidateEdgeChecks: 0,
          totalPrunedEdgeChecks: 0,
          edgeReductionPct: 0,
          totalBlockedEdges: 0,
          phases: [],
        },
      collisionCpuRuntime: sim?.collisionCpuRuntime
        ? {
          enabled: sim.collisionCpuRuntime.enabled !== false,
          reason: sim.collisionCpuRuntime.reason || null,
          collisionIterations: Number(sim.collisionCpuRuntime.collisionIterations) || 0,
          stageMs: {
            rigidRigidPre: Number(sim?.collisionCpuRuntime?.stageMs?.rigidRigidPre) || 0,
            rigidSoft: Number(sim?.collisionCpuRuntime?.stageMs?.rigidSoft) || 0,
            softSoftNode: Number(sim?.collisionCpuRuntime?.stageMs?.softSoftNode) || 0,
            softSoftEdge: Number(sim?.collisionCpuRuntime?.stageMs?.softSoftEdge) || 0,
            rigidRigidPost: Number(sim?.collisionCpuRuntime?.stageMs?.rigidRigidPost) || 0,
            total: Number(sim?.collisionCpuRuntime?.stageMs?.total) || 0,
          },
          substageMs: {
            candidateBuildPre: Number(sim?.collisionCpuRuntime?.substageMs?.candidateBuildPre) || 0,
            candidateBuildPost: Number(sim?.collisionCpuRuntime?.substageMs?.candidateBuildPost) || 0,
            candidateBuildTotal: Number(sim?.collisionCpuRuntime?.substageMs?.candidateBuildTotal) || 0,
            rigidRigidNarrowphasePre: Number(sim?.collisionCpuRuntime?.substageMs?.rigidRigidNarrowphasePre) || 0,
            rigidSoftNarrowphase: Number(sim?.collisionCpuRuntime?.substageMs?.rigidSoftNarrowphase) || 0,
            softSoftNodeNarrowphase: Number(sim?.collisionCpuRuntime?.substageMs?.softSoftNodeNarrowphase) || 0,
            softSoftEdgeNarrowphase: Number(sim?.collisionCpuRuntime?.substageMs?.softSoftEdgeNarrowphase) || 0,
            rigidRigidNarrowphasePost: Number(sim?.collisionCpuRuntime?.substageMs?.rigidRigidNarrowphasePost) || 0,
            narrowphaseTotal: Number(sim?.collisionCpuRuntime?.substageMs?.narrowphaseTotal) || 0,
            boundaryApply: Number(sim?.collisionCpuRuntime?.substageMs?.boundaryApply) || 0,
            applyTotal: Number(sim?.collisionCpuRuntime?.substageMs?.applyTotal) || 0,
            total: Number(sim?.collisionCpuRuntime?.substageMs?.total) || 0,
          },
          checks: {
            rigidRigidPairs: Number(sim.collisionCpuRuntime.rigidRigidPairChecks) || 0,
            rigidRigidProxyPairs: Number(sim.collisionCpuRuntime.rigidRigidProxyPairChecks) || 0,
            rigidRigidSatCalls: Number(sim.collisionCpuRuntime.rigidRigidSatCalls) || 0,
            rigidRigidEarlyRejectChecks: Number(sim.collisionCpuRuntime.rigidRigidEarlyRejectChecks) || 0,
            rigidRigidPostSkipped: Number(sim.collisionCpuRuntime.rigidRigidPostSkipped) || 0,
            rigidSoftNodeRaw: Number(sim.collisionCpuRuntime.rigidSoftNodeRawChecks) || 0,
            rigidSoftNode: Number(sim.collisionCpuRuntime.rigidSoftNodeChecks) || 0,
            rigidSoftNodeCandidate: Number(sim.collisionCpuRuntime.rigidSoftNodeCandidateChecks) || 0,
            rigidSoftEdgeRaw: Number(sim.collisionCpuRuntime.rigidSoftEdgeRawChecks) || 0,
            rigidSoftEdge: Number(sim.collisionCpuRuntime.rigidSoftEdgeChecks) || 0,
            rigidSoftEdgeCandidate: Number(sim.collisionCpuRuntime.rigidSoftEdgeCandidateChecks) || 0,
            softSoftNode: Number(sim.collisionCpuRuntime.softSoftNodeChecks) || 0,
            softSoftEdge: Number(sim.collisionCpuRuntime.softSoftEdgeChecks) || 0,
          },
          hits: {
            rigidRigidPairs: Number(sim.collisionCpuRuntime.rigidRigidPairHits) || 0,
            rigidRigidSat: Number(sim.collisionCpuRuntime.rigidRigidSatHits) || 0,
            rigidSoftNode: Number(sim.collisionCpuRuntime.rigidSoftNodeHits) || 0,
            rigidSoftEdge: Number(sim.collisionCpuRuntime.rigidSoftEdgeHits) || 0,
            softSoftNode: Number(sim.collisionCpuRuntime.softSoftNodeHits) || 0,
            softSoftEdge: Number(sim.collisionCpuRuntime.softSoftEdgeHits) || 0,
          },
          hitRatesPct: {
            rigidRigidPairs: Number(sim.collisionCpuRuntime.rigidRigidPairHitRatePct) || 0,
            rigidRigidSatCalls: Number(sim.collisionCpuRuntime.rigidRigidSatCallRatePct) || 0,
            rigidRigidSatHits: Number(sim.collisionCpuRuntime.rigidRigidSatHitRatePct) || 0,
            rigidSoftNode: Number(sim.collisionCpuRuntime.rigidSoftNodeHitRatePct) || 0,
            rigidSoftEdge: Number(sim.collisionCpuRuntime.rigidSoftEdgeHitRatePct) || 0,
            rigidSoftNodeCandidate: Number(sim.collisionCpuRuntime.rigidSoftNodeCandidateHitRatePct) || 0,
            rigidSoftEdgeCandidate: Number(sim.collisionCpuRuntime.rigidSoftEdgeCandidateHitRatePct) || 0,
            softSoftNode: Number(sim.collisionCpuRuntime.softSoftNodeHitRatePct) || 0,
            softSoftEdge: Number(sim.collisionCpuRuntime.softSoftEdgeHitRatePct) || 0,
          },
          prefilter: {
            rigidRigidAabbRejected: Number(sim.collisionCpuRuntime.rigidRigidAabbRejected) || 0,
            rigidRigidRadiusRejected: Number(sim.collisionCpuRuntime.rigidRigidRadiusRejected) || 0,
            rigidRigidEarlyRadiusRejected: Number(sim.collisionCpuRuntime.rigidRigidEarlyRadiusRejected) || 0,
            rigidRigidTotalRejected: Number(sim.collisionCpuRuntime.rigidRigidPrefilterRejected) || 0,
          },
          polyCache: {
            lookups: Number(sim.collisionCpuRuntime.polyCacheLookups) || 0,
            hits: Number(sim.collisionCpuRuntime.polyCacheHits) || 0,
            localHits: Number(sim.collisionCpuRuntime.polyCacheLocalHits) || 0,
            sharedHits: Number(sim.collisionCpuRuntime.polyCacheSharedHits) || 0,
            misses: Number(sim.collisionCpuRuntime.polyCacheMisses) || 0,
            rebuilds: Number(sim.collisionCpuRuntime.polyCacheRebuilds) || 0,
            bodiesBuilt: Number(sim.collisionCpuRuntime.polyBodiesBuilt) || 0,
            bodiesReused: Number(sim.collisionCpuRuntime.polyBodiesReused) || 0,
            buildMs: Number(sim.collisionCpuRuntime.polyBuildMs) || 0,
            hitRatePct: Number(sim.collisionCpuRuntime.polyCacheHitRatePct) || 0,
          },
        }
        : {
          enabled: false,
          reason: 'uninitialized',
          collisionIterations: 0,
          stageMs: { rigidRigidPre: 0, rigidSoft: 0, softSoftNode: 0, softSoftEdge: 0, rigidRigidPost: 0, total: 0 },
          substageMs: {
            candidateBuildPre: 0,
            candidateBuildPost: 0,
            candidateBuildTotal: 0,
            rigidRigidNarrowphasePre: 0,
            rigidSoftNarrowphase: 0,
            softSoftNodeNarrowphase: 0,
            softSoftEdgeNarrowphase: 0,
            rigidRigidNarrowphasePost: 0,
            narrowphaseTotal: 0,
            boundaryApply: 0,
            applyTotal: 0,
            total: 0,
          },
          checks: {
            rigidRigidPairs: 0,
            rigidRigidProxyPairs: 0,
            rigidRigidSatCalls: 0,
            rigidRigidEarlyRejectChecks: 0,
            rigidRigidPostSkipped: 0,
            rigidSoftNodeRaw: 0,
            rigidSoftNode: 0,
            rigidSoftNodeCandidate: 0,
            rigidSoftEdgeRaw: 0,
            rigidSoftEdge: 0,
            rigidSoftEdgeCandidate: 0,
            softSoftNode: 0,
            softSoftEdge: 0,
          },
          hits: {
            rigidRigidPairs: 0,
            rigidRigidSat: 0,
            rigidSoftNode: 0,
            rigidSoftEdge: 0,
            softSoftNode: 0,
            softSoftEdge: 0,
          },
          hitRatesPct: {
            rigidRigidPairs: 0,
            rigidRigidSatCalls: 0,
            rigidRigidSatHits: 0,
            rigidSoftNode: 0,
            rigidSoftEdge: 0,
            rigidSoftNodeCandidate: 0,
            rigidSoftEdgeCandidate: 0,
            softSoftNode: 0,
            softSoftEdge: 0,
          },
          prefilter: {
            rigidRigidAabbRejected: 0,
            rigidRigidRadiusRejected: 0,
            rigidRigidEarlyRadiusRejected: 0,
            rigidRigidTotalRejected: 0,
          },
          polyCache: {
            lookups: 0,
            hits: 0,
            localHits: 0,
            sharedHits: 0,
            misses: 0,
            rebuilds: 0,
            bodiesBuilt: 0,
            bodiesReused: 0,
            buildMs: 0,
            hitRatePct: 0,
          },
        },
      softFluidCouplingRuntime: {
        lastSourceRoute: sim?.softFluidCouplingWgslState?.lastSourceRoute || null,
        lastMode: sim?.softFluidCouplingWgslState?.lastMode || null,
        lastAuthoritativeCarrySource: sim?.softFluidCouplingWgslState?.lastAuthoritativeCarrySource || null,
        lastAuthoritativeClusterLoadSource: sim?.softFluidCouplingWgslState?.lastAuthoritativeClusterLoadSource || null,
        lastTiming: sim?.softFluidCouplingWgslState?.lastTiming ? { ...sim.softFluidCouplingWgslState.lastTiming } : null,
      },
      softSpringRuntime: {
        lastSourceRoute: sim?.softSpringXpbdWgslState?.lastSourceRoute || null,
        lastMode: sim?.softSpringXpbdWgslState?.lastMode || null,
        lastAuthoritativeProposalSource: sim?.softSpringXpbdWgslState?.lastAuthoritativeProposalSource || null,
        lastTiming: sim?.softSpringXpbdWgslState?.lastTiming ? { ...sim.softSpringXpbdWgslState.lastTiming } : null,
      },
      softAreaRuntime: {
        lastSourceRoute: sim?.softAreaXpbdWgslState?.lastSourceRoute || null,
        lastMode: sim?.softAreaXpbdWgslState?.lastMode || null,
        lastAuthoritativeProposalSource: sim?.softAreaXpbdWgslState?.lastAuthoritativeProposalSource || null,
        lastTiming: sim?.softAreaXpbdWgslState?.lastTiming ? { ...sim.softAreaXpbdWgslState.lastTiming } : null,
      },
      membranePressureRuntime: {
        lastSourceRoute: sim?.softMembranePressureWgslState?.lastSourceRoute || null,
        lastMode: sim?.softMembranePressureWgslState?.lastMode || null,
        lastVelocityProposalSource: sim?.softMembranePressureWgslState?.lastVelocityProposalSource || null,
        lastTiming: sim?.softMembranePressureWgslState?.lastTiming ? { ...sim.softMembranePressureWgslState.lastTiming } : null,
      },
      softIntegrateRuntime: {
        lastSourceRoute: sim?.softIntegrateWgslState?.lastSourceRoute || null,
        lastMode: sim?.softIntegrateWgslState?.lastMode || null,
        lastError: sim?.softIntegrateWgslState?.lastError || null,
        lastTiming: sim?.softIntegrateWgslState?.lastTiming ? { ...sim.softIntegrateWgslState.lastTiming } : null,
      },
      rigidPostIntegrateRuntime: {
        lastMode: sim?.rigidPostIntegrateWgslState?.lastMode || null,
        lastError: sim?.rigidPostIntegrateWgslState?.lastError || null,
        lastTiming: sim?.rigidPostIntegrateWgslState?.lastTiming ? { ...sim.rigidPostIntegrateWgslState.lastTiming } : null,
      },
      membraneBoundaryRuntime: {
        lastMode: sim?.softMembraneBoundaryWgslState?.lastMode || null,
        lastMembraneBoundaryEdgeAuthoritativeSource: sim?.softMembraneBoundaryWgslState?.lastMembraneBoundaryEdgeAuthoritativeSource || null,
        lastMembraneBendAuthoritativeSource: sim?.softMembraneBoundaryWgslState?.lastMembraneBendAuthoritativeSource || null,
        lastTiming: sim?.softMembraneBoundaryWgslState?.lastTiming ? { ...sim.softMembraneBoundaryWgslState.lastTiming } : null,
      },
      membraneShapeMemoryRuntime: {
        lastMode: sim?.softMembraneShapeMemoryWgslState?.lastMode || null,
        lastShapeMemoryAuthoritativeSource: sim?.softMembraneShapeMemoryWgslState?.lastShapeMemoryAuthoritativeSource || null,
        lastTiming: sim?.softMembraneShapeMemoryWgslState?.lastTiming ? { ...sim.softMembraneShapeMemoryWgslState.lastTiming } : null,
      },
      softRestRecoveryRuntime: {
        lastMode: sim?.softRestRecoveryWgslState?.lastMode || null,
        lastProbeSource: sim?.softRestRecoveryWgslState?.lastProbeSource || null,
        lastProposalSource: sim?.softRestRecoveryWgslState?.lastProposalSource || null,
        lastTiming: sim?.softRestRecoveryWgslState?.lastTiming ? { ...sim.softRestRecoveryWgslState.lastTiming } : null,
      },
      softDeformationRuntime: {
        lastMode: sim?.softDeformationWgslState?.lastMode || null,
        lastSourceRoute: sim?.softDeformationWgslState?.lastSourceRoute || null,
        lastSoftDeformationWgslMetricsSource: sim?.softDeformationWgslState?.lastSoftDeformationWgslMetricsSource || null,
        lastTiming: sim?.softDeformationWgslState?.lastTiming ? { ...sim.softDeformationWgslState.lastTiming } : null,
      },
      bodyFluidInjectionRuntime: {
        lastSourceRoute: sim?.bodyFluidInjectionWgslState?.lastSourceRoute || null,
        lastMode: sim?.bodyFluidInjectionWgslState?.lastMode || null,
        lastAuthoritativeGatherSource: sim?.bodyFluidInjectionWgslState?.lastAuthoritativeGatherSource || null,
        lastTiming: sim?.bodyFluidInjectionWgslState?.lastTiming ? { ...sim.bodyFluidInjectionWgslState.lastTiming } : null,
      },
      rigidSoftRuntime: {
        lastSourceRoute: sim?.rigidSoftCollisionWgslState?.lastSourceRoute || null,
        lastMode: sim?.rigidSoftCollisionWgslState?.lastMode || null,
        lastError: sim?.rigidSoftCollisionWgslState?.lastError || null,
        lastWgslUnavailableReason: sim?.rigidSoftCollisionWgslState?.lastWgslUnavailableReason || null,
        lastRigidSoftResponseAuthoritativeSource: sim?.rigidSoftCollisionWgslState?.lastRigidSoftResponseAuthoritativeSource || null,
        lastRigidSoftResponseOwnership: sim?.rigidSoftCollisionWgslState?.lastRigidSoftResponseOwnership || null,
        lastRigidSoftResponseOwnershipDetailed: sim?.rigidSoftCollisionWgslState?.lastRigidSoftResponseOwnershipDetailed || null,
        lastRigidSoftResponseRoute: sim?.rigidSoftCollisionWgslState?.lastRigidSoftResponseRoute || null,
        lastRigidSoftResponseFallbackReason: sim?.rigidSoftCollisionWgslState?.lastRigidSoftResponseFallbackReason || null,
        lastPreparedNodePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastPreparedNodePairCount) || 0,
        lastPreparedEdgePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastPreparedEdgePairCount) || 0,
        lastPreparedNodeNarrowphasePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastPreparedNodeNarrowphasePairCount) || 0,
        lastPreparedEdgeNarrowphasePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastPreparedEdgeNarrowphasePairCount) || 0,
        lastPreparedNarrowphaseImpulseSeedNodePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastPreparedNarrowphaseImpulseSeedNodePairCount) || 0,
        lastPreparedNarrowphaseImpulseSeedEdgePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastPreparedNarrowphaseImpulseSeedEdgePairCount) || 0,
        lastNodeBroadphaseReadbackCount: Number(sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseReadbackCount) || 0,
        lastEdgeBroadphaseReadbackCount: Number(sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseReadbackCount) || 0,
        lastNodeBroadphaseCandidatePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseCandidatePairCount) || 0,
        lastEdgeBroadphaseCandidatePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseCandidatePairCount) || 0,
        lastNodeBroadphaseActivePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseActivePairCount) || 0,
        lastEdgeBroadphaseActivePairCount: Number(sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseActivePairCount) || 0,
        lastNodeNarrowphaseDroppedPairCount: Number(sim?.rigidSoftCollisionWgslState?.lastNodeNarrowphaseDroppedPairCount) || 0,
        lastNodeBroadphaseReason: sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseReason || null,
        lastEdgeBroadphaseReason: sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseReason || null,
        lastNodeBroadphaseAuthoritativeSource: sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseAuthoritativeSource || null,
        lastEdgeBroadphaseAuthoritativeSource: sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseAuthoritativeSource || null,
        lastNodeBroadphaseReadbackSourceRoute: sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseReadbackSourceRoute || null,
        lastEdgeBroadphaseReadbackSourceRoute: sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseReadbackSourceRoute || null,
        lastNodeBroadphaseReplayEpochDelta: Number(sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseReplayEpochDelta) || 0,
        lastEdgeBroadphaseReplayEpochDelta: Number(sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseReplayEpochDelta) || 0,
        lastNodeBroadphaseReplayHitCount: Number(sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseReplayHitCount) || 0,
        lastNodeBroadphaseDispatchHitCount: Number(sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseDispatchHitCount) || 0,
        lastEdgeBroadphaseReplayHitCount: Number(sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseReplayHitCount) || 0,
        lastEdgeBroadphaseDispatchHitCount: Number(sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseDispatchHitCount) || 0,
        lastRigidSoftBroadphaseCadenceDue: !!sim?.rigidSoftCollisionWgslState?.lastRigidSoftBroadphaseCadenceDue,
        lastRigidSoftBroadphaseInterval: Number(sim?.rigidSoftCollisionWgslState?.lastRigidSoftBroadphaseInterval) || 0,
        lastNodeBroadphaseTiming: sim?.rigidSoftCollisionWgslState?.lastNodeBroadphaseTiming
          ? { ...sim.rigidSoftCollisionWgslState.lastNodeBroadphaseTiming }
          : null,
        lastEdgeBroadphaseTiming: sim?.rigidSoftCollisionWgslState?.lastEdgeBroadphaseTiming
          ? { ...sim.rigidSoftCollisionWgslState.lastEdgeBroadphaseTiming }
          : null,
        lastNodeNarrowphaseAabbProbeTiming: sim?.rigidSoftCollisionWgslState?.lastNodeNarrowphaseAabbProbeTiming
          ? { ...sim.rigidSoftCollisionWgslState.lastNodeNarrowphaseAabbProbeTiming }
          : null,
        lastEdgeNarrowphaseAabbProbeTiming: sim?.rigidSoftCollisionWgslState?.lastEdgeNarrowphaseAabbProbeTiming
          ? { ...sim.rigidSoftCollisionWgslState.lastEdgeNarrowphaseAabbProbeTiming }
          : null,
        lastRigidSoftResponseTiming: sim?.rigidSoftCollisionWgslState?.lastRigidSoftResponseTiming
          ? { ...sim.rigidSoftCollisionWgslState.lastRigidSoftResponseTiming }
          : null,
      },
    };
  },
  captureCanvasDataUrl: (mimeType = 'image/png') => {
    try {
      if (!(canvas instanceof HTMLCanvasElement)) return null;
      return canvas.toDataURL(String(mimeType || 'image/png'));
    } catch {
      return null;
    }
  },
  captureCanvasMetrics: () => captureCanvasMetrics(),
};

if (EMBED_MODE) {
  window.addEventListener('message', async (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};

    if (data.type === 'gpuLabHighlightSegment') {
      if (sim) {
        sim.highlightSegmentId = String(data.segmentId || '');
        const pulseMs = Math.max(200, Number(data.pulseMs) || 2200);
        const dt = Math.max(1e-4, Number(sim.controls?.dt) || 0.01);
        const pulseFrames = Math.max(1, Math.round(pulseMs / (dt * 1000)));
        sim.highlightUntilFrame = (Number(sim.frame) || 0) + pulseFrames;
      }
      return;
    }

    if (data.type !== 'gpuLabEmbedReset') return;

    try {
      const result = await resetEmbedWindTunnelFromSpec(data.spec, data.options || {});
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: 'gpuLabEmbedResetAck', ok: true, result }, window.location.origin);
      }
    } catch (err) {
      const msg = String(err?.message || err);
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: 'gpuLabEmbedResetAck', ok: false, error: msg }, window.location.origin);
      }
      log({ ok: false, error: msg });
    }
  });

  if (window.parent && window.parent !== window) {
    window.parent.postMessage({ type: 'gpuLabEmbedReady' }, window.location.origin);
  }

  log('embed ready: awaiting mesh-lab compiled spec');
} else {
  loadGeneratedMiniScenarios().finally(() => {
    log('ready: choose scenario, paint, then Start');
  });
}
