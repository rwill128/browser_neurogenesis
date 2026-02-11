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

function drawLine(buf, x0, y0, x1, y1, r, g, b) {
  let dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  let dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  while (true) {
    setPixel(buf, x0, y0, r, g, b);
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

function worldToPixel(vx, vy, world) {
  return {
    x: Math.floor((vx / world.width) * (w - 1)),
    y: Math.floor((vy / world.height) * (h - 1))
  };
}

function nodeColor(vertex, fallbackEnergy = 0) {
  const byTypeName = {
    NEURON: [200, 100, 255],
    PREDATOR: [255, 50, 50],
    SWIMMER: [0, 200, 255],
    PHOTOSYNTHETIC: [60, 179, 113],
    EMITTER: [0, 255, 100],
    EATER: [255, 165, 0],
    EYE: [180, 180, 250],
    JET: [255, 255, 100],
    ATTRACTOR: [255, 105, 180],
    REPULSOR: [128, 0, 128]
  };

  const named = byTypeName[vertex?.nodeTypeName];
  if (named) return named;

  const e = clamp(fallbackEnergy || 0, 0, 140);
  return [Math.floor(255 - (e / 140) * 140), Math.floor(80 + (e / 140) * 170), 220];
}

const world = data.world || { width: 120, height: 80 };
(data.timeline || []).forEach((snap, i) => {
  const buf = Buffer.alloc(w * h * 3, 0);

  const isBlankScenario = (data.scenario || '').includes('blank');
  if (!isBlankScenario) {
    // background gradient
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 3;
        buf[idx] = 10;
        buf[idx + 1] = 12 + Math.floor((y / h) * 20);
        buf[idx + 2] = 18 + Math.floor((x / w) * 15);
      }
    }
  }

  const creatures = (snap.creatures && snap.creatures.length) ? snap.creatures : (snap.sampleCreatures || []);
  for (const c of creatures) {
    const verts = Array.isArray(c.vertices) ? c.vertices : [];

    // Preferred: draw actual spring connectivity when available (closer to browser look).
    if (Array.isArray(c.springs) && c.springs.length > 0 && verts.length > 1) {
      for (const s of c.springs) {
        const va = verts[s.a];
        const vb = verts[s.b];
        if (!va || !vb) continue;
        const a = worldToPixel(va.x, va.y, world);
        const b = worldToPixel(vb.x, vb.y, world);
        if (s.isRigid) {
          drawLine(buf, a.x, a.y, b.x, b.y, 255, 235, 90);
          drawLine(buf, a.x + 1, a.y, b.x + 1, b.y, 255, 235, 90);
        } else {
          drawLine(buf, a.x, a.y, b.x, b.y, 150, 150, 150);
        }
      }
      for (const v of verts) {
        const p = worldToPixel(v.x, v.y, world);
        const [r, g, b] = nodeColor(v, c.energy || 0);
        const rad = Math.max(1, Math.round((v.radius || 4) * 0.22));
        drawCircle(buf, p.x, p.y, rad, r, g, b);
      }
      continue;
    }

    // Fallback: connect vertices in listed order.
    if (verts.length >= 3) {
      const pts = verts.map(v => worldToPixel(v.x, v.y, world));
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i];
        const b = pts[(i + 1) % pts.length];
        drawLine(buf, a.x, a.y, b.x, b.y, 130, 170, 220);
      }
      for (let i = 0; i < verts.length; i++) {
        const [r, g, b] = nodeColor(verts[i], c.energy || 0);
        drawCircle(buf, pts[i].x, pts[i].y, 2, r, g, b);
      }
    } else if (verts.length === 2) {
      const a = worldToPixel(verts[0].x, verts[0].y, world);
      const b = worldToPixel(verts[1].x, verts[1].y, world);
      drawLine(buf, a.x, a.y, b.x, b.y, 130, 170, 220);
      const ca = nodeColor(verts[0], c.energy || 0);
      const cb = nodeColor(verts[1], c.energy || 0);
      drawCircle(buf, a.x, a.y, 2, ca[0], ca[1], ca[2]);
      drawCircle(buf, b.x, b.y, 2, cb[0], cb[1], cb[2]);
    } else {
      const p = worldToPixel(c.center.x, c.center.y, world);
      const [r, g, b] = nodeColor(null, c.energy || 0);
      drawCircle(buf, p.x, p.y, 6, r, g, b);
    }
  }

  const header = `P6\n${w} ${h}\n255\n`;
  const framePath = resolve(outDir, `frame-${String(i).padStart(5, '0')}.ppm`);
  writeFileSync(framePath, header);
  writeFileSync(framePath, buf, { flag: 'a' });
});

console.log(`Rendered ${(data.timeline || []).length} frames to ${outDir}`);
console.log(`Example ffmpeg:`);
console.log(`ffmpeg -y -framerate 10 -i ${outDir}/frame-%05d.ppm -c:v libx264 -pix_fmt yuv420p ${outDir}/${basename(input).replace(/\.json$/, '.mp4')}`);
