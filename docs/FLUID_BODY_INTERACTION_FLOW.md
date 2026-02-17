# Fluid ↔ Body Interaction Flow (GPU Lab)

This document describes the frame-by-frame branching logic for fluid/body interaction in GPU Lab.

> Naming note: in code, dye **BLOCK** is represented as `DEFLECT`.

---

## High-level per-frame flow

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         PER FRAME (GPU LAB)                         │
└──────────────────────────────────────────────────────────────────────┘
                |
                v
      ┌───────────────────────┐
      │ 1) Build obstacleMask │   (velocity barrier mask)
      └───────────────────────┘
                |
                |  For EACH rigid edge and soft spring:
                |    velMode = edgeVelocityMode ?? edgeBodyMode ?? BLOCK
                |    ├─ if velMode == BLOCK -> stamp segment into obstacleMask
                |    └─ if velMode == PASS  -> do not stamp
                v
      ┌───────────────────────┐
      │ 2) Build dyeModeMask  │   (per-channel dye policy mask)
      └───────────────────────┘
                |
                |  For EACH edge/channel (R,G,B):
                |    rigid channel mode:
                |      if permeability[channel] > 0 -> PASS
                |      else if edgeDyeMode[channel] == EAT -> EAT
                |      else -> BLOCK(DEFLECT)
                |
                |    soft channel mode:
                |      edgeDyeMode[channel] (PASS / BLOCK(DEFLECT) / EAT)
                |
                |    Stamp only non-PASS modes
                v
      ┌──────────────────────────────────────────────────────────────┐
      │ 3) GPU fluid solve                                           │
      │    - advect velocity + pressure/projection use obstacleMask │
      │    - advect dye uses dyeModeMask per channel:               │
      │        if src/dst has EAT     -> channel = 0               │
      │        else if src/dst BLOCK  -> keep local * fade         │
      │        else PASS              -> advect normally * fade    │
      └──────────────────────────────────────────────────────────────┘
                |
                v
      ┌──────────────────────────────────────────────────────────────┐
      │ 4) Body coupling (fluid <-> body)                            │
      │    - readback fluid                                           │
      │    - rigid + soft sample fluid near body edges/nodes         │
      │      (sampling just outside blocked cells when needed)       │
      │    - compute forces/torques from relative flow               │
      │    - scale by momentum coupling policy                       │
      │    - update body linear/angular velocities                   │
      │    - inject body momentum back into fluid (feedback path)    │
      └──────────────────────────────────────────────────────────────┘
                |
                v
      ┌───────────────────────┐
      │ 5) Upload next fields │
      └───────────────────────┘
```

---

## Branching rules (explicit)

### 1) Velocity interaction branch (obstacle mask)

Per edge:

- `BLOCK` → edge contributes to GPU obstacle mask (velocity no-through).
- `PASS` → edge does not contribute to obstacle mask.

Applies to both **rigid** and **soft** edges via the same edge-mode decision.

### 2) Dye interaction branch (per channel)

Per edge, per dye channel (R/G/B):

- `PASS` → dye channel advects normally.
- `BLOCK` (`DEFLECT` in code) → channel does not advect through edge crossing path.
- `EAT` (`ABSORB` in code) → channel is removed/zeroed when masked contact branch is hit.

### 3) Momentum transfer branch

Per edge/spring momentum coupling scalar in `[0..1]`:

- Lower values reduce body↔fluid transfer strength.
- Higher values increase transfer strength.

Used in force accumulation and feedback injection paths.

---

## Important current contract limits

1. **Velocity BLOCK can suppress dye-contact opportunities for EAT**
   - If an edge is stamped into `obstacleMask` (`velocity=BLOCK`), plume trajectories can be diverted before they reach the dye-mask contact path.
   - In practice this makes EAT less observable/reliable in BLOCK-heavy setups.

2. **Interaction Lab contract (current mode): only EAT vs NO-OP authoring**
   - Interaction Lab exposes dye as `EAT` or `NO-OP` per selected channel.
   - `BLOCK`/`DEFLECT` is not exposed in this lab mode.

3. **Velocity-gated EAT behavior in Interaction Lab**
   - `velocity=BLOCK` is treated as the hard boundary branch.
   - In this branch, requested `EAT` is automatically disabled (effective dye mode becomes `NO-OP`).
   - `EAT` is active only when `velocity=PASS`.

---

## Why this shape exists

- Keeps policy decisions edge-local and consistent across rigid + soft.
- Uses GPU-side masks for primary flow/dye interaction.
- Avoids old runtime post-pass barrier dependence.
- Preserves force coupling by sampling outside blocked cells when needed.
