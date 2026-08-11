import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const TEST_DIR = fileURLToPath(new URL('.', import.meta.url));
const source = readFileSync(resolve(TEST_DIR, '../../sim-server/public/gpu-lab.js'), 'utf8');

test('standalone GPU Lab starts automatically after scenario discovery', () => {
  const startup = source.slice(source.lastIndexOf('\nif (EMBED_MODE) {'));
  const standaloneBranchOffset = startup.lastIndexOf('\n} else {');

  assert.notEqual(standaloneBranchOffset, -1, 'expected a standalone startup branch');
  const embedBranch = startup.slice(0, standaloneBranchOffset);
  const standaloneBranch = startup.slice(standaloneBranchOffset);

  assert.doesNotMatch(
    embedBranch,
    /startWithErrorHandling\(\)/,
    'embed mode should continue waiting for a compiled spec',
  );
  assert.match(
    standaloneBranch,
    /loadGeneratedMiniScenarios\(\)[\s\S]*\.finally\(startWithErrorHandling\);/,
    'standalone mode should start after generated scenarios finish loading',
  );
});

test('manual Start uses the same guarded startup path', () => {
  assert.match(
    source,
    /function startWithErrorHandling\(\) \{[\s\S]*running = false;[\s\S]*log\(\{ ok: false, error: String\(e\) \}\);/,
    'startup failures should restore the stopped state and be reported',
  );
  assert.match(
    source,
    /runBtn\.addEventListener\('click', startWithErrorHandling\);/,
    'the Start button should retain the same startup behavior',
  );
});
