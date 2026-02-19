# Native-core port order: CPU hotspot evidence (2026-02-19)

Profile command:
```bash
FIXED_STEPS=512 WARMUP_STEPS=16 SWEEPS=scale SCALES=200 \
  node --cpu-prof --cpu-prof-name=cpu-scale200.cpuprofile \
  node-harness/bench-runtime-solver-scale.mjs
```

Top sampled JS functions (cpu-scale200.cpuprofile):

1. 26.42% — `sanitizeFinitePolygonVerts`  
   `sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js:104`
2. 20.13% — `resolveRigidVsSoftNodeCollisionGpuOnly`  
   `.../stepRigidSoftCollisionGpuOnly.js:152`
3. 8.18% — `applyCpuRigidSoftResponseFallback`  
   `.../stepRigidSoftCollisionGpuOnly.js:2774`
4. 5.35% — `rigidVerticesWorld`  
   `.../stepRigidSoftCollisionGpuOnly.js:31`
5. 4.72% — `pointInPolygonInclusive`  
   `.../stepRigidSoftCollisionGpuOnly.js:74`
6. 4.09% — `closestPointOnSegment`  
   `.../stepRigidSoftCollisionGpuOnly.js:123`
7. 4.09% — `edgeOutwardNormal`  
   `.../stepRigidSoftCollisionGpuOnly.js:134`
8. 3.46% — `resolveRigidVsSoftEdgeCollisionGpuOnly`  
   `.../stepRigidSoftCollisionGpuOnly.js:259`

## Port priority implied by data

### Priority 1 (first native slice)
`stepRigidSoftCollisionGpuOnly.js` narrowphase/response geometry stack:
- point-in-polygon
- closest-point/edge normal
- rigid vs soft node/edge resolve
- CPU fallback response apply

Reason: these functions dominate sampled CPU time in this profile.

### Priority 2
Precomputed polygon/hull sanitization and transform cache path:
- sanitizeFinitePolygonVerts
- rigidVerticesWorld

Reason: repeated geometry prep overhead appears very high in hot loop.

### Priority 3
After collision vertical slice:
- body-fluid injection gather/reduction/apply
- integrate/post-integrate kernels

Reason: these stages are sizeable in frame timing but lower than collision hotspot in this sampled CPU profile.
