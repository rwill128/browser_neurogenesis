import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const repoRoot = path.resolve(import.meta.dirname, '..', '..');

async function read(relPath) {
  return fs.readFile(path.join(repoRoot, relPath), 'utf8');
}

function expectRuntimeSolverRadios(html, relPath) {
  assert.match(html, /name="runtimeSolverPath"[^>]*value="baseline"/i, `${relPath} missing baseline runtime solver radio`);
  assert.match(html, /name="runtimeSolverPath"[^>]*value="gpu-only"/i, `${relPath} missing gpu-only runtime solver radio`);
  assert.match(html, /Baseline solver/i, `${relPath} missing explicit baseline label`);
  assert.match(html, /GPU-only solver \(experimental\)/i, `${relPath} missing explicit experimental label`);
}

test('all three labs expose runtime solver selector radios with explicit labels', async () => {
  const [gpuLabHtml, meshLabHtml, interactionLabHtml] = await Promise.all([
    read('sim-server/public/gpu-lab.html'),
    read('sim-server/public/mesh-lab.html'),
    read('sim-server/public/interaction-lab.html'),
  ]);

  expectRuntimeSolverRadios(gpuLabHtml, 'sim-server/public/gpu-lab.html');
  expectRuntimeSolverRadios(meshLabHtml, 'sim-server/public/mesh-lab.html');
  expectRuntimeSolverRadios(interactionLabHtml, 'sim-server/public/interaction-lab.html');
});

test('mesh and interaction lab forward selected solver path to embedded gpu lab resets', async () => {
  const [meshLabJs, interactionLabJs] = await Promise.all([
    read('sim-server/public/mesh-lab.js'),
    read('sim-server/public/interaction-lab.js'),
  ]);

  assert.match(meshLabJs, /solverPath:\s*getRuntimeSolverPath\(\)/, 'mesh-lab must include runtime solver path in embed reset options');
  assert.match(interactionLabJs, /solverPath:\s*getRuntimeSolverPath\(\)/, 'interaction-lab must include runtime solver path in embed reset options');
});

test('gpu lab consumes and reports runtime solver path for embed resets/status', async () => {
  const gpuLabJs = await read('sim-server/public/gpu-lab.js');

  assert.match(gpuLabJs, /setRuntimeSolverPath\(options\?\.solverPath\s*\?\?\s*getRuntimeSolverPath\(\)/, 'gpu-lab embed reset must apply requested runtime solver path');
  assert.match(gpuLabJs, /runtimeSolverPath:\s*normalizeRuntimeSolverPath\(sim\?\.controls\?\.runtimeSolverPath\)/, 'gpu-lab API status must report active runtime solver path');
});
