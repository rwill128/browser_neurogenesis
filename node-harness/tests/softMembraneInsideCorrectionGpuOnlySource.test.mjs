import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftMembraneInsideCorrectionGpuOnly.js'), 'utf8');

test('soft membrane inside-correction gpu-only path stages deterministic WGSL layout metadata and authoritative WGSL proposal replay path', () => {
  assert.match(
    source,
    /function buildSoftMembraneInsideCorrectionWgslLayout\([\s\S]*nodeData = new Float32Array\(nodeCount \* 4\);[\s\S]*loopPointOffsets = new Uint32Array\(loopCount \+ 1\);[\s\S]*flatPoints: Float32Array\.from\(flatPoints\)/,
    'expected deterministic soft-membrane inside-correction WGSL prep layout builder for upcoming compute dispatch',
  );

  assert.match(
    source,
    /computeSoftMembraneInsideLayoutSignature\([\s\S]*WGSL_LAYOUT_SIG_SEED[\s\S]*hashArray\(layout\.flatPoints, true\);/,
    'expected deterministic WGSL layout signature hashing so future authoritative replay can gate on stable inputs',
  );

  assert.match(
    source,
    /const pipelineMode = getGpuOnlyPipelineModeProfile\(wgslOffload\);[\s\S]*const stagedWgslLayout = !standardMode[\s\S]*lastPreparedMembraneInsideLayoutSignature = stagedWgslLayoutSignature >>> 0;[\s\S]*lastMembraneInsideSourceRoute = 'cpu-membrane-inside-prepared-layout';/,
    'expected gpu-only inside-correction pass to publish explicit cpu-prepared WGSL staging route telemetry while baseline remains authoritative',
  );

  assert.match(
    source,
    /const softMembraneInsideCorrectionProposalWgsl = \/\* wgsl \*\/[\s\S]*point_xy\([\s\S]*out_corr_x\[ni\] = best_corr\.x;[\s\S]*out_hit\[ni\] = hit;/,
    'expected concrete WGSL membrane-inside correction proposal kernel with polygon containment + nearest-edge correction math',
  );

  assert.match(
    source,
    /canApplyAuthoritativeMembraneInsideProposal\([\s\S]*enableAuthoritativeMembraneInsideCorrection[\s\S]*applyAuthoritativeMembraneInsideProposal\([\s\S]*wgsl-membrane-inside-authoritative/,
    'expected authoritative WGSL replay gating + explicit source routes for membrane-inside correction stage',
  );
});
