import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const gpuLabSource = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');
const moduleSource = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftMembraneConstraintsGpuOnly.js'), 'utf8');

test('gpu-lab wires membrane shape-memory gpu-only pass with WGSL offload state bag', () => {
  assert.match(
    gpuLabSource,
    /applySoftMembraneShapeMemoryVelocityGpuOnly\(\{[\s\S]*wgslOffload:[\s\S]*enabled: true,[\s\S]*device: sim\?\.device,[\s\S]*state: \(sim\.softMembraneShapeMemoryWgslState \|\|= \{\}\),[\s\S]*\}\)/,
    'expected gpu-lab membrane shape-memory gpu-only branch to wire isolated WGSL offload context while baseline path stays untouched',
  );
});

test('membrane constraints gpu-only module defines concrete WGSL shape-memory proposal kernel with source-route telemetry', () => {
  assert.match(
    moduleSource,
    /const softMembraneShapeMemoryProposalWgsl = \/\* wgsl \*\/[\s\S]*deltaVxOut\[i\] = ex \* corrPos \* invDt;[\s\S]*deltaVyOut\[i\] = ey \* corrPos \* invDt;/,
    'expected concrete WGSL kernel math for membrane shape-memory velocity proposal deltas',
  );

  assert.match(
    moduleSource,
    /dispatchSoftMembraneShapeMemoryProposal\([\s\S]*lastShapeMemoryProposalFinite[\s\S]*lastShapeMemoryProposalSignature[\s\S]*lastShapeMemoryProposalSource = finite\.allFinite \? 'wgsl-shape-memory-proposal' : 'cpu-shape-memory-authoritative-nonfinite'/,
    'expected proposal dispatch to preserve explicit source-route and finite fallback telemetry',
  );

  assert.match(
    moduleSource,
    /pendingWgslShapeMemoryProposalPromise[\s\S]*dispatchSoftMembraneShapeMemoryProposal\([\s\S]*lastShapeMemoryProposalSource = 'cpu-shape-memory-authoritative'[\s\S]*lastMode = 'cpu-shape-memory-authoritative'/,
    'expected serialized WGSL proposal dispatch with hard CPU fallback source-route when dispatch fails',
  );
});
