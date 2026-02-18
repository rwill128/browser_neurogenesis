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
  assert.match(gpuLabJs, /setRuntimePipelineMode\(options\?\.pipelineMode\s*\?\?\s*getRuntimePipelineMode\(selectedSolverPath\)/, 'gpu-lab embed reset must apply requested runtime pipeline mode');
  assert.match(gpuLabJs, /runtimePipelineMode:\s*normalizeRuntimePipelineMode\(sim\?\.controls\?\.runtimePipelineMode, sim\?\.controls\?\.runtimeSolverPath\)/, 'gpu-lab API status must report active runtime pipeline mode');
  assert.match(gpuLabJs, /if \(mode === 'standard'\) return 'standard';/, 'gpu-lab pipeline normalization must preserve explicit standard mode even under gpu-only solver path');
});

test('gpu lab html exposes 3 runtime pipeline modes with explicit labels', async () => {
  const gpuLabHtml = await read('sim-server/public/gpu-lab.html');
  assert.match(gpuLabHtml, /name="runtimePipelineMode"[^>]*value="standard"/i, 'gpu-lab missing standard runtime pipeline mode radio');
  assert.match(gpuLabHtml, /name="runtimePipelineMode"[^>]*value="gpu-only-validated"/i, 'gpu-lab missing gpu-only validated mode radio');
  assert.match(gpuLabHtml, /name="runtimePipelineMode"[^>]*value="gpu-only-fast"/i, 'gpu-lab missing gpu-only fast mode radio');
  assert.match(gpuLabHtml, /standard \(baseline reference\)/i, 'gpu-lab missing standard mode label');
  assert.match(gpuLabHtml, /gpu-only validated \(parity-heavy\)/i, 'gpu-lab missing validated mode label');
  assert.match(gpuLabHtml, /gpu-only fast \(reduced readback\/parity\)/i, 'gpu-lab missing fast mode label');
});
