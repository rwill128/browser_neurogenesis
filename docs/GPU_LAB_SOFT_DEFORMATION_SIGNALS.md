# GPU Lab — Soft Deformation Warning/Severe Signals

This document describes exactly how GPU Lab decides when a soft-body cluster is marked as:

- **Warning** (shown in yellow/orange), or
- **Severe** (shown in red).

Implementation source: `sim-server/public/gpu-lab.js` (see `buildSoftDeformationState(...)` and `ensureSoftDeformationReferenceState(...)`).

## Signals computed per soft cluster

Each frame, the solver computes these metrics per cluster:

1. **`stretchMax`**
   - Largest spring stretch ratio in the cluster.
   - Higher means at least one spring is much longer than its rest length.

2. **`stretchMin`**
   - Smallest spring stretch ratio in the cluster.
   - Very small values indicate compression/collapse behavior.

3. **`areaRatio`**
   - Current boundary-loop area divided by reference area.
   - Values far from `1.0` indicate inflation/collapse drift.

4. **`poseErrorRms`**
   - RMS residual from rigid-aligned shape comparison to reference local pose.
   - Captures non-rigid deformation while being translation/rotation invariant.

5. **`poseErrorMax`**
   - Max per-vertex residual from the same rigid-aligned comparison.
   - Captures worst local outlier deformation.

## Reference shape used for pose error

Reference pose is built per cluster from valid node positions:

- Build cluster-local coordinates around centroid (`refLocal`).
- Cache reference RMS scale (`refRms`).
- On each frame, rebuild current local coordinates (`curLocal`) and compare with rigid alignment.

This means pose error intentionally ignores simple translation/rotation and focuses on shape change.

## Thresholds used today

Current constants in `gpu-lab.js`:

- **Stretch**
  - Warning: `SOFT_DEFORM_WARN_STRETCH = 3.2`
  - Severe: `SOFT_DEFORM_SEVERE_STRETCH = 6.0`
  - Severe-collapse helper: `stretchMin <= 0.12`

- **Area ratio**
  - Warning: `<= 0.45` or `>= 2.2`
  - Severe: `<= 0.2` or `>= 5.0`

- **Pose error**
  - Warning: `poseErrorRms >= 0.18` OR `poseErrorMax >= 0.34`
  - Severe: `poseErrorRms >= 0.32` OR `poseErrorMax >= 0.58`

## Decision logic

For each cluster:

- `warning = legacyWarn || poseWarn`
- `severe = severeCollapse || poseSevere`

Where:

- `legacyWarn` comes from stretch/area warning thresholds.
- `poseWarn` comes from pose warning thresholds.
- `poseSevere` comes from pose severe thresholds.
- `severeCollapse` comes from severe legacy collapse-like thresholds.

### Membrane exception

`severeCollapse` is only applied to **non-membrane** clusters.
Membrane clusters can still become severe via pose-severe thresholds, but they are excluded from collapse-shove behavior tied to legacy collapse signals.

## Color mapping in UI

In the body overlay rendering:

- **Red** = cluster in severe set.
- **Yellow/Orange** = warning set (not severe).
- **Cyan** = normal.

The same state is also exposed in runtime telemetry (`softDeformation` block in the status log).
