# Chemofluid Evo-Sim: Energy, Biology, and Genetics Design Spec (Draft v0)

Status: **Design-only (no implementation in this document)**

## 1) Purpose
Define a principled biological/energetic model for the new chemofluid simulation so evolution can optimize meaningful survival/reproduction strategies (not one-frame threshold hacks).

## 2) Core design goals
1. **Continuity over frame-gates**: reproduction and survival decisions use state over time (integrated history), not single-frame spikes.
2. **Conservation-aware accounting**: track energy entering, transforming, and leaving the system well enough to estimate carrying capacity.
3. **Trait→physics coupling**: genetic traits map to measurable physical consequences (drag, permeability, digestion selectivity, actuation cost, etc.).
4. **Evolvable modularity**: genome supports mutation, drift, and developmental growth plans without exploding complexity.
5. **Stable-yet-rich search space**: enough constraints to avoid trivial runaway/degenerate attractors, enough freedom for novelty.

## 3) System boundary and energy accounting

### 3.1 Energy pools
- **External resource pool**: dye emitters inject colored resource mass/energy into world.
- **Environmental dissipation**: advection/viscosity/absorption loss channels.
- **Organism internal energy**: per-organism (or per-body) metabolizable store.

### 3.2 Accounting model (first-principles but practical)
At each step, estimate:
- `E_in_emitters` (resource injected)
- `E_capture_total` (organism capture via in-polygon/channel-selective digestion)
- `E_maint_total` (basal maintenance)
- `E_actuation_total` (swim/shape change/motion work proxy)
- `E_repro_total` (allocated to offspring)
- `E_loss_env` (resource removed/absorbed/dissipated)

Track rolling balances over windows (e.g., 30s, 120s) to estimate carrying capacity and detect pathological growth.

## 4) Organism energy dynamics

### 4.1 Intake
- Intake is channel-selective (`digestRGB` / future affinity vectors).
- Intake scales by local dye availability and edge/body capture geometry.
- Optional diminishing returns when local concentration is very high.

### 4.2 Costs
- Basal maintenance (size/mass-dependent).
- Locomotion/actuation cost (velocity change + deformation + active swimmer thrust proxies).
- Structural upkeep (optional): higher for more complex morphologies.

### 4.3 Storage + buffers
- Energy store with min/max and reserve buffer.
- If below starvation thresholds for sustained interval, degrade function first (reduced actuation) before death.

### 4.4 Reproduction rule (time-integrated)
Reproduction eligibility should require all of:
1. `energy > reserve + reproduction_cost`
2. `rolling_surplus(T) > surplus_threshold` over window `T` (not a single frame)
3. cooldown satisfied
4. local ecological condition sane (optional: local resource floor / crowding cap)

## 5) Carrying capacity and population regulation
Use resource budget instead of ad hoc caps:
- Estimate `K_estimate` from recent `E_in_emitters` and median per-organism energy burn.
- Compare actual population to energy-sustainable range.
- Use this as diagnostic + tuning target, not hard kill-switch.

## 6) Genome redesign (conceptual)

### 6.1 Trait families
1. **Material traits**: edge dye mode (pass/deflect/absorb), body permeability (pass/block), drag/friction.
2. **Metabolic traits**: channel affinity/selectivity, uptake saturation curves, storage capacity, efficiency.
3. **Motor traits**: actuation amplitude/frequency/phasing, torque/carry coupling gains.
4. **Material-distribution traits (local, not global)**:
   - soft bodies: per-node drag/friction coefficients,
   - rigid bodies: per-edge (or per-hull-sample) drag coefficients,
   - enables asymmetric drag layouts as an evolvable behavior axis.
5. **Morphological traits**: topology templates, edge types, growth sequence controls.

### 6.2 Mutation/drift model
- Point mutations for scalar parameters.
- Structural mutations for topology/edge-type changes.
- Drift + occasional larger jumps.
- Keep mutation rates adaptive but bounded.

## 7) Local drag/friction granularity (new core evolvable axis)

### 7.1 Why this matters
Global drag coefficients are too coarse. Local coefficients create directional mechanics (e.g., sticky leading edge, slippery trailing edge), unlocking richer locomotion, trapping, and channeling strategies.

### 7.2 Soft bodies
- Store drag/friction per node.
- Node drag participates in local force exchange with fluid.
- Mutation can tune node-level values independently (bounded ranges + smooth mutation).

### 7.3 Rigid bodies
- Store drag/friction per edge (or equivalent hull sample points mapped to edges).
- Fluid coupling integrates per-edge drag into net force + torque.
- Asymmetric edge drag should produce directional turning bias and passive steering.

### 7.4 Design constraints
- Keep trait counts bounded (avoid huge genomes by default).
- Add regularization or mutation penalties against extreme checkerboard coefficients.
- Ensure deterministic ordering/indexing so heredity and parity checks remain stable.

## 8) Developmental growth plan (evo-devo direction)
Instead of fully static bodies:
- Define staged growth (node/edge addition, materialization over time).
- Growth consumes energy.
- Growth plan is heritable + mutable.
- Enables life-history strategies (fast repro vs robust mature forms).

## 8) Evaluation scenarios (for future implementation)
Create fixed scenario suite for meaningful comparisons:
1. Uniform field + sparse emitters
2. High-shear channels
3. Viscosity islands
4. Mixed toxic/food color regime
5. Boundary-heavy arena

For each scenario, track survival, reproduction, lineage persistence, diversity, and energy efficiency.

## 9) Metrics to report continuously
- Population + lineage diversity
- Energy intake/outgo by channel
- Rolling surplus distribution
- Reproduction attempts/success by lineage
- Resource depletion maps and recovery rates
- Morphology complexity vs fitness proxies

## 10) Open questions for refinement
1. Should energy be organism-level only, or distributed per node/organ compartment?
2. How strong should coupling be between morphology complexity and maintenance cost?
3. Do we introduce explicit toxicity channels now or later?
4. What minimal genome representation gives expressive power without combinatorial blowup?
5. Which invariants must always hold to prevent non-physical exploits?

## 11) Next design-only refinement pass
- Formalize equations for rolling surplus/reproduction eligibility.
- Define initial parameter priors and safe ranges.
- Draft genome schema v0 (JSON-like).
- Define acceptance criteria before implementation begins.
