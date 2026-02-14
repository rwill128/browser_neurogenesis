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

### 4.4 Reproduction rule (time-integrated, implementation-targeted)
Define per-organism instantaneous net flux:
- `S(t) = I(t) - M(t) - A(t) - G(t)`
  - `I(t)`: selective digestion intake (respecting existing `digestRGB` / affinity channels)
  - `M(t)`: maintenance (size/mass + optional morphology complexity term)
  - `A(t)`: actuation/motion/deformation expenditure
  - `G(t)`: growth-plan spend (0 if no growth event active)

Define EMA surplus over horizon `T_s`:
- `S_ema(t) = α_s * S(t) + (1-α_s) * S_ema(t-Δt)`
- `α_s = 1 - exp(-Δt / T_s)`

Define reserve-aware reproducible energy:
- `E_free(t) = E_store(t) - E_reserve_min`
- `E_spawn_need = E_offspring_seed + E_spawn_overhead`

Reproduction eligibility requires all conditions true for a hold interval `T_hold`:
1. **Energy sufficiency:** `E_free(t) >= E_spawn_need`
2. **Sustained surplus:** `S_ema(t) >= S_thresh` continuously for `T_hold`
3. **No debt trend:** `dE_store/dt` EMA over `T_s` is non-negative
4. **Cooldown:** `t - t_last_spawn >= T_cooldown`
5. **Local ecology gate:** local edible dye concentration above floor and local crowding below cap

Spawn accounting (must be conservative):
- Parent update at spawn: `E_store_parent -= (E_spawn_need + E_lineage_tax)`
- Offspring init: `E_store_child = E_offspring_seed`
- `E_spawn_overhead + E_lineage_tax` is removed from metabolizable pool (entropy/inefficiency sink)

Suggested starter ranges (tunable):
- `T_s`: 20-60 s
- `T_hold`: 5-15 s
- `S_thresh`: 0.5-2.0x median maintenance flux
- `T_cooldown`: 10-45 s

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

### 6.2 Mutation/drift model (bounded + testable)
Let each offspring receive mutations from three channels:
1. **Scalar point mutation** (continuous traits: drag, permeability, affinity, gains)
2. **Discrete state mutation** (edge dye/body enum modes, actuator type switches)
3. **Structural mutation** (node/edge add/remove, growth-plan step edits)

Per-birth mutation budget:
- Sample `B_mut ~ Poisson(λ_mut)` and allocate events across channels via fixed weights.
- Enforce `B_mut <= B_max` to avoid burst instability.

Scalar mutation rule:
- `x' = clamp(x + Normal(0, σ_x), x_min, x_max)`
- Optional multiplicative form for strictly positive traits: `x' = clamp(x * exp(Normal(0, σ_logx)), x_min, x_max)`

Discrete mutation rule:
- Transition only through allowed adjacency map (no invalid enum jumps).
- Example: `edge dye mode` can only mutate among explicitly supported simulation enums.

Structural mutation safeguards:
- Reject edits violating topology invariants (min connectivity, non-self-intersection constraints if used).
- Apply at most one major structural edit per birth unless lineage health score is high.

Adaptive drift (slow exploration):
- Lineage-local `σ` drifts by small log-random walk every `N_gen` generations, bounded in `[σ_min, σ_max]`.
- If lineage extinction risk rises (persistent negative surplus), bias toward conservative `σ` reduction.

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

### 8.1 Growth-plan representation
Each lineage carries a finite, ordered growth plan `P = [p1..pn]`, where each step `pi` is:
- `kind`: `{add_node, add_edge, reinforce_material, activate_actuator, remodel_local_drag}`
- `cost_energy`: metabolizable energy required before execution
- `maturity_gate`: minimum organism age or stage index
- `topology_guard`: invariant checks that must pass pre/post step
- `rollback_policy`: `{skip, retry_later, abort_plan}`

`P` is heritable and mutation-editable (insert/delete/retune step params) under Section 6.2 budget limits.

### 8.2 Execution state machine
Per organism, maintain:
- `growth_stage` (index into `P`)
- `growth_active` (bool)
- `growth_debt` (reserved but not yet spent energy)
- `growth_cooldown_until`

Step `pi` can execute only if all are true:
1. `E_store - E_reserve_min >= cost_energy + E_operational_buffer`
2. `S_ema(t) >= S_growth_min` over `T_growth_hold` (prevents growth during transient spikes)
3. `now >= growth_cooldown_until`
4. Topology/material guards succeed

On execute:
- Reserve `cost_energy` into `growth_debt`, apply structural/material change, then settle spend:
  - success: `E_store -= cost_energy`, `growth_debt -> 0`, `growth_stage++`
  - failure with rollback `skip`: `growth_stage++`, debt returned
  - failure with rollback `retry_later`: debt returned, keep stage, set cooldown
  - failure with rollback `abort_plan`: debt returned, disable growth for life

### 8.3 Coupling to reproduction and survival
- While `growth_active`, reproduction gate in Section 4.4 uses `G(t)` including planned growth amortization term:
  `G(t) = G_exec(t) + G_reserve(t)` where `G_reserve(t)` tracks active reservation pressure.
- If starvation mode is entered, pending growth is automatically paused and reservations released.
- Optional life-history switch: once final stage completes, maintenance multiplier may increase (larger mature form) while actuator efficiency may improve.

### 8.4 Mutation constraints for growth plans
- Max steps: `|P| <= P_max`.
- Structural edits to `P` must preserve at least one viable minimal morphology path.
- `activate_actuator` cannot reference unsupported actuator enum/state.
- `remodel_local_drag` must respect per-node/per-edge bounds and smoothness regularizer.

### 8.5 Starter constants (design defaults)
- `T_growth_hold`: 8-20 s
- `S_growth_min`: 0.25-1.0x median maintenance flux
- `E_operational_buffer`: 0.5-1.5x instantaneous maintenance
- `growth_cooldown`: 3-10 s between steps

## 9) Actuator taxonomy and expenditure model (refinement)

### 8.1 Supported actuator classes (design-level)
Actuators should be explicit genome-addressable enums to avoid hidden behavior channels:
- `none`: passive body, no active thrust.
- `edge_undulation`: periodic edge deformation producing propulsion via fluid coupling.
- `volume_pulse`: body expansion/contraction cycle, thrust from asymmetric drag/permeability.
- `jet_emitter`: directional dye/mass ejection for impulse-like motion (high burst cost).
- `ciliary_band` (future-capable): distributed low-amplitude surface actuation with strong maintenance overhead.

Each actuator instance must declare:
- `actuator_type` (enum)
- `gain` (amplitude/intensity)
- `duty_cycle` (active fraction over period)
- `phase` (for multi-actuator coordination)
- `orientation_ref` (edge index / local frame)

### 8.2 Actuation energy model (accounting-compatible)
For organism `o`, instantaneous actuation spend:
`A_o(t) = Σ_k A_k(t)` over active actuators `k`.

Per-actuator spend is decomposed:
`A_k(t) = C_idle(type_k) * duty_k(t) + C_work(type_k) * W_k(t)`
where:
- `C_idle`: baseline energetic overhead when actuator is enabled.
- `W_k(t)`: physics work proxy (e.g., absolute impulse, deformation rate integral, or ejected mass * speed proxy), implementation-chosen but fixed per actuator class.
- `C_work`: converts work proxy units into metabolizable energy spend.

Hard constraints:
1. `A_k(t) >= 0` always.
2. If actuator is inactive and no deformation/ejection occurs, `A_k(t)=0`.
3. `A_o(t)` must be included in Section 4.4 surplus exactly once (no double-count with maintenance).

### 8.3 Performance/efficiency coupling
Define actuator efficiency trait `η_k ∈ [η_min, η_max]` that scales effective propulsion but not free energy:
- Effective thrust proxy: `T_eff_k = η_k * T_cmd_k`
- Cost side remains non-decreasing with command magnitude.

Recommended anti-exploit rule:
- For any actuator class, net displacement gain from increasing command should show diminishing returns beyond class-specific knee `u_knee`, while energy spend continues rising.

### 8.4 Compatibility with current chemofluid mechanics
- `edge_undulation` and `ciliary_band` interact with local drag traits (node-level soft / edge-level rigid) exactly as in Section 7.
- `jet_emitter` cannot bypass selective digestion: expelled material is not counted as intake and cannot be re-counted as immediate self-digestion without transport delay.
- Body/edge dye mode enums remain authoritative for pass/deflect/absorb behavior during actuation.

## 10) Evaluation scenarios (for future implementation)
Create fixed scenario suite for meaningful comparisons:
1. Uniform field + sparse emitters
2. High-shear channels
3. Viscosity islands
4. Mixed toxic/food color regime
5. Boundary-heavy arena

For each scenario, track survival, reproduction, lineage persistence, diversity, and energy efficiency.

## 11) Metrics to report continuously
- Population + lineage diversity
- Energy intake/outgo by channel
- Rolling surplus distribution
- Reproduction attempts/success by lineage
- Resource depletion maps and recovery rates
- Morphology complexity vs fitness proxies

## 12) Acceptance criteria (design validation checklist)

### 12.1 Energy accounting invariants
- [ ] For every step, computed flux terms satisfy:
  `ΔE_internal_total ≈ E_capture_total - E_maint_total - E_actuation_total - E_repro_total - E_growth_total`
  within numerical tolerance `ε_balance`.
- [ ] No organism can increase `E_store` when `I(t)=0` and all costs are non-negative.
- [ ] Spawn event conserves accounting exactly per Section 4.4 parent/child updates.

### 12.2 Sustained-surplus reproduction gating
- [ ] Synthetic pulse test: one short intake spike that raises instantaneous energy but not `S_ema` for `T_hold` must **not** permit spawn.
- [ ] Synthetic sustained test: constant positive surplus over `T_hold` with cooldown satisfied must permit spawn.
- [ ] Cooldown test: second spawn attempt before `T_cooldown` must fail even with surplus.

### 12.3 Mutation/drift safety and expressivity
- [ ] 1000-birth fuzz run yields zero invalid enum states for edge/body material modes.
- [ ] Scalar traits remain within declared bounds after mutation.
- [ ] Structural mutation never breaks required topology invariants.
- [ ] Drift adaptation never sets `σ` outside `[σ_min, σ_max]`.

### 12.4 Growth-plan logic and gating checks
- [ ] Growth step cannot execute unless energy + sustained-surplus gates in Section 8.2 are simultaneously satisfied.
- [ ] Reservation accounting test: when a step reserves `cost_energy`, that amount is unavailable to reproduction gating until released or spent.
- [ ] Rollback test matrix (`skip/retry_later/abort_plan`) preserves energy (no hidden gain/loss) and advances/halts stage as specified.
- [ ] Starvation interrupt test: entering starvation mode pauses growth and releases pending reservation within one simulation step.
- [ ] Plan mutation validity test: mutated plans never exceed `P_max`, never reference invalid actuator/material enums, and always retain at least one viable minimal morphology path.

### 12.5 Chemofluid consistency checks
- [ ] Selective digestion uses existing channel semantics (`digestRGB` / affinity) with no bypass path.
- [ ] Local drag traits are read at node-level (soft) and edge-level (rigid) in force integration.
- [ ] Emitter-driven resource intake remains dependent on local dye concentration and body/edge interaction modes.

### 12.6 Actuator taxonomy/accounting checks
- [ ] All actuator instances in genomes validate against the supported enum set (`none`, `edge_undulation`, `volume_pulse`, `jet_emitter`, optional `ciliary_band` if enabled).
- [ ] For each actuator class, zero-command test yields `A_k(t)=0` and no propulsion impulse.
- [ ] Monotonic-cost test: increasing command magnitude over a fixture trajectory does not reduce total actuation spend.
- [ ] No-double-count test: per-step `E_actuation_total` equals the sum of actuator spends and is not also included in maintenance.
- [ ] Jet recapture delay test: ejected material cannot be digested by the same organism in the same simulation step.

## 13) Open questions for refinement
1. Should energy be organism-level only, or distributed per node/organ compartment?
2. How strong should coupling be between morphology complexity and maintenance cost?
3. Do we introduce explicit toxicity channels now or later?
4. What minimal genome representation gives expressive power without combinatorial blowup?
5. Should `E_lineage_tax` be constant or ecology-adaptive to stabilize booms?

## 14) Next design-only refinement pass
- Draft genome schema v0 (JSON-like) including enum adjacency maps and mutation budget fields.
- Define benchmark fixtures for the acceptance tests in Section 12.
- Choose initial default constants (`T_s`, `T_hold`, `S_thresh`, `λ_mut`, `σ` bounds) for first calibration sweep.
