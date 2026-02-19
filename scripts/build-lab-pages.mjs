#!/usr/bin/env node
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const srcDir = resolve(root, 'sim-server', 'public');
const outDir = resolve(root, 'dist', 'labs');

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });
cpSync(srcDir, outDir, { recursive: true });

const isTextAsset = (path) => path.endsWith('.html') || path.endsWith('.js');

function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    const full = resolve(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, files);
    else files.push(full);
  }
  return files;
}

function toRelativeAssetPath(filePath, targetRaw) {
  const qIndex = targetRaw.search(/[?#]/);
  const targetPath = qIndex >= 0 ? targetRaw.slice(0, qIndex) : targetRaw;
  const suffix = qIndex >= 0 ? targetRaw.slice(qIndex) : '';
  const absTarget = resolve(outDir, targetPath);
  let rel = relative(dirname(filePath), absTarget).replace(/\\/g, '/');
  if (!rel.startsWith('.')) rel = `./${rel}`;
  return `${rel}${suffix}`;
}

function rewriteRootAssetSpecifiers(filePath, content) {
  // Rewrites quoted root-absolute asset specifiers (e.g. '/gpu-lab.js', '/runtime-solvers/x.js').
  // Only rewrites likely local file paths to keep non-file literals untouched.
  return content.replace(/(["'])\/([A-Za-z0-9._\-/]+(?:\?[^"']*)?)\1/g, (_m, quote, targetRaw) => {
    return `${quote}${toRelativeAssetPath(filePath, targetRaw)}${quote}`;
  });
}

const rewritten = [];
for (const filePath of walk(outDir)) {
  if (!isTextAsset(filePath)) continue;
  const before = readFileSync(filePath, 'utf8');
  const after = rewriteRootAssetSpecifiers(filePath, before);
  if (after !== before) {
    writeFileSync(filePath, after, 'utf8');
    rewritten.push(relative(outDir, filePath));
  }
}

const indexPath = resolve(outDir, 'index.html');
if (!statSync(indexPath, { throwIfNoEntry: false })) {
  writeFileSync(indexPath, '<!doctype html><meta http-equiv="refresh" content="0; url=./gpu-lab.html" />\n', 'utf8');
}

console.log(`[build-lab-pages] copied ${srcDir} -> ${outDir}`);
console.log(`[build-lab-pages] rewrote ${rewritten.length} files`);
for (const f of rewritten.slice(0, 40)) console.log(`  - ${f}`);
if (rewritten.length > 40) console.log(`  ... and ${rewritten.length - 40} more`);
