const EPS = 1e-8;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function signedAreaFromIndices(nodes, indices) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    s += a.x * b.y - b.x * a.y;
  }
  return s * 0.5;
}

function isFiniteNode(node) {
  return Number.isFinite(Number(node?.x)) && Number.isFinite(Number(node?.y));
}

function convexHullIndices(nodes, indices) {
  const unique = new Map();
  for (const idx of indices) {
    const n = nodes[idx];
    if (!isFiniteNode(n)) continue;
    const k = `${Math.round(n.x * 1e6)},${Math.round(n.y * 1e6)}`;
    if (!unique.has(k)) unique.set(k, idx);
  }

  const pts = [...unique.values()].map((idx) => ({ idx, x: nodes[idx].x, y: nodes[idx].y }));
  if (pts.length <= 2) return pts.map((p) => p.idx);
  pts.sort((a, b) => (a.x - b.x) || (a.y - b.y));

  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }

  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }

  const hull = lower.slice(0, -1).concat(upper.slice(0, -1)).map((p) => p.idx);
  return hull.length >= 3 ? hull : pts.map((p) => p.idx);
}

function traverseCycle(component, adjacency, nodes) {
  const set = new Set(component);
  const degreeTwo = component.every((idx) => (adjacency.get(idx)?.size || 0) === 2);
  if (!degreeTwo) return null;

  const start = [...component].sort((a, b) => (nodes[a].x - nodes[b].x) || (nodes[a].y - nodes[b].y))[0];
  const nbs = [...(adjacency.get(start) || [])].filter((nb) => set.has(nb));
  if (nbs.length !== 2) return null;

  let prev = start;
  let curr = nbs[0];
  if (nbs.length === 2) {
    const a = nodes[nbs[0]];
    const b = nodes[nbs[1]];
    const sa = Math.atan2(a.y - nodes[start].y, a.x - nodes[start].x);
    const sb = Math.atan2(b.y - nodes[start].y, b.x - nodes[start].x);
    curr = sa <= sb ? nbs[0] : nbs[1];
  }

  const loop = [start];
  const guard = component.length * 4 + 16;
  let steps = 0;

  while (steps++ < guard) {
    if (curr === start) break;
    if (!set.has(curr)) return null;
    loop.push(curr);

    const nextCandidates = [...(adjacency.get(curr) || [])].filter((nb) => set.has(nb) && nb !== prev);
    if (!nextCandidates.length) return null;
    if (nextCandidates.length > 1) {
      // Shouldn't happen in degree-2 cycle, but keep deterministic tie-break.
      nextCandidates.sort((ia, ib) => (nodes[ia].x - nodes[ib].x) || (nodes[ia].y - nodes[ib].y));
    }
    const next = nextCandidates[0];
    prev = curr;
    curr = next;
  }

  if (loop.length < 3) return null;
  if (curr !== start) return null;

  const unique = [...new Set(loop)];
  return unique.length >= 3 ? unique : null;
}

export function sanitizeSoftSprings(rawSprings, nodeCount, {
  edgeBodyPass = 0,
  edgeBodyBlock = 1,
  restFloor = 1e-3,
} = {}) {
  const out = [];
  const seenPairs = new Set();
  let dropped = 0;

  for (const sp of (Array.isArray(rawSprings) ? rawSprings : [])) {
    if (!Array.isArray(sp)) {
      dropped += 1;
      continue;
    }
    const [aRaw, bRaw, restRaw, edgeBodyRaw, edgeDyeRaw] = sp;

    const a = Number(aRaw);
    const b = Number(bRaw);
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      dropped += 1;
      continue;
    }
    if (a < 0 || b < 0 || a >= nodeCount || b >= nodeCount || a === b) {
      dropped += 1;
      continue;
    }

    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = `${lo}:${hi}`;
    if (seenPairs.has(key)) {
      dropped += 1;
      continue;
    }
    seenPairs.add(key);

    const rest = Math.max(restFloor, Number.isFinite(Number(restRaw)) ? Number(restRaw) : 1);
    const edgeBodyMode = Number(edgeBodyRaw) === edgeBodyPass ? edgeBodyPass : edgeBodyBlock;
    out.push([a, b, rest, edgeBodyMode, edgeDyeRaw]);
  }

  return { springs: out, dropped };
}

export function ensureLambdaCacheSize(previous, nextLength, clampAbs = 20) {
  const len = Math.max(0, Number(nextLength) | 0);
  const next = new Float32Array(len);
  if (!previous || typeof previous.length !== 'number') return next;

  const copy = Math.min(previous.length, len);
  for (let i = 0; i < copy; i++) {
    const v = Number(previous[i]);
    next[i] = Number.isFinite(v) ? clamp(v, -clampAbs, clampAbs) : 0;
  }
  return next;
}

export function buildSoftClusterBoundaryLoops(nodes, springs, { blockMode = 1 } = {}) {
  const byClusterNodes = new Map();
  for (let i = 0; i < (nodes?.length || 0); i++) {
    const n = nodes[i];
    if (!isFiniteNode(n)) continue;
    const cid = n.clusterId ?? 0;
    if (!byClusterNodes.has(cid)) byClusterNodes.set(cid, new Set());
    byClusterNodes.get(cid).add(i);
  }

  const byClusterAdj = new Map();
  for (const sp of (springs || [])) {
    if (!Array.isArray(sp) || sp.length < 4) continue;
    const a = Number(sp[0]);
    const b = Number(sp[1]);
    const mode = Number(sp[3]);
    if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
    if (a < 0 || b < 0 || a >= nodes.length || b >= nodes.length || a === b) continue;
    if (mode !== blockMode) continue;

    const na = nodes[a];
    const nb = nodes[b];
    if (!isFiniteNode(na) || !isFiniteNode(nb)) continue;
    const ca = na.clusterId ?? 0;
    const cb = nb.clusterId ?? 0;
    if (ca !== cb) continue;

    if (!byClusterAdj.has(ca)) byClusterAdj.set(ca, new Map());
    const adj = byClusterAdj.get(ca);
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a).add(b);
    adj.get(b).add(a);
  }

  const loops = [];

  for (const [clusterId, nodeSet] of byClusterNodes.entries()) {
    const clusterNodes = [...nodeSet];
    if (clusterNodes.length < 3) continue;

    const adj = byClusterAdj.get(clusterId) || new Map();
    const visited = new Set();
    const candidateLoops = [];

    for (const start of adj.keys()) {
      if (visited.has(start)) continue;
      const stack = [start];
      visited.add(start);
      const comp = [];
      while (stack.length) {
        const cur = stack.pop();
        comp.push(cur);
        for (const nb of (adj.get(cur) || [])) {
          if (!visited.has(nb)) {
            visited.add(nb);
            stack.push(nb);
          }
        }
      }
      if (comp.length < 3) continue;
      const cycle = traverseCycle(comp, adj, nodes);
      if (cycle && cycle.length >= 3) candidateLoops.push(cycle);
    }

    let indices = null;
    let source = 'hull';
    if (candidateLoops.length) {
      candidateLoops.sort((a, b) => Math.abs(signedAreaFromIndices(nodes, b)) - Math.abs(signedAreaFromIndices(nodes, a)));
      indices = candidateLoops[0];
      source = 'boundary';
    } else {
      indices = convexHullIndices(nodes, clusterNodes);
    }

    if (!indices || indices.length < 3) continue;
    const area = signedAreaFromIndices(nodes, indices);
    if (Math.abs(area) < EPS) continue;
    if (area < 0) indices = [...indices].reverse();

    loops.push({ clusterId, indices, source });
  }

  return loops;
}
