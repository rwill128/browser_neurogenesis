#!/usr/bin/env node
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve, basename } from 'node:path';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

const input = resolve(arg('input'));
const outDir = resolve(arg('out', './artifacts/frames'));
const w = Number(arg('width', '640'));
const h = Number(arg('height', '360'));

const data = JSON.parse(readFileSync(input, 'utf8'));
mkdirSync(outDir, { recursive: true });

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function setPixel(buf, x, y, r, g, b) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  const i = (y * w + x) * 3;
  buf[i] = r; buf[i + 1] = g; buf[i + 2] = b;
}

function drawCircle(buf, cx, cy, rad, r, g, b) {
  const r2 = rad * rad;
  for (let y = Math.floor(cy - rad); y <= Math.ceil(cy + rad); y++) {
    for (let x = Math.floor(cx - rad); x <= Math.ceil(cx + rad); x++) {
      const dx = x - cx, dy = y - cy;
      if (dx*dx + dy*dy <= r2) setPixel(buf, x, y, r, g, b);
    }
  }
}

const world = data.world || { width: 120, height: 80 };
(data.timeline || []).forEach((snap, i) => {
  const buf = Buffer.alloc(w * h * 3, 0);

  // background gradient
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 3;
      buf[idx] = 10;
      buf[idx + 1] = 12 + Math.floor((y / h) * 20);
      buf[idx + 2] = 18 + Math.floor((x / w) * 15);
    }
  }

  for (const c of snap.sampleCreatures || []) {
    const px = Math.floor((c.center.x / world.width) * (w - 1));
    const py = Math.floor((c.center.y / world.height) * (h - 1));
    const e = clamp(c.energy || 0, 0, 140);
    const red = Math.floor(255 - (e / 140) * 140);
    const green = Math.floor(80 + (e / 140) * 170);
    drawCircle(buf, px, py, 5, red, green, 220);
  }

  const header = `P6\n${w} ${h}\n255\n`;
  const framePath = resolve(outDir, `frame-${String(i).padStart(5, '0')}.ppm`);
  writeFileSync(framePath, header);
  writeFileSync(framePath, buf, { flag: 'a' });
});

console.log(`Rendered ${(data.timeline || []).length} frames to ${outDir}`);
console.log(`Example ffmpeg:`);
console.log(`ffmpeg -y -framerate 10 -i ${outDir}/frame-%05d.ppm -c:v libx264 -pix_fmt yuv420p ${outDir}/${basename(input).replace(/\.json$/, '.mp4')}`);
