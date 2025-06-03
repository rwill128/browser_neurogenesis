const NodeType = {
    PREDATOR: 0,
    EATER: 1,
    PHOTOSYNTHETIC: 2,
    NEURON: 3,
    EMITTER: 4, // For dye
    SWIMMER: 5, // For propulsion
    EYE: 6      // New: For particle detection
    // Old types like NEUTRAL, FLOATING, FIXED_ROOT, EMITTER_SWIMMER are removed
};

const RLRewardStrategy = {
    ENERGY_CHANGE: 0,
    REPRODUCTION_EVENT: 1,
    PARTICLE_PROXIMITY: 2
};

const RLAlgorithmType = {
    REINFORCE: 0, // Your current policy gradient
    SAC: 1        // Soft Actor-Critic
};

const MovementType = {
    FIXED: 0,    // Fixed in place, does not interact with fluid velocity but can affect it (if Swimmer)
    FLOATING: 1, // Pushed by fluid, cannot be a Swimmer
    NEUTRAL: 2   // Standard soft body physics, only interacts with fluid if Swimmer (by pushing it)
};

// --- MassPoint Class (Soft Body with Verlet Integration) ---
class MassPoint {
    constructor(x, y, mass = 0.5, radius = 5, color = 'rgba(0,150,255,0.8)') {
        this.pos = new Vec2(x, y);
        this.prevPos = new Vec2(x, y);
        this.force = new Vec2();
        this.mass = mass;
        this.invMass = this.mass !== 0 ? 1 / this.mass : 0;
        this.radius = radius;
        this.color = color;
        this.nodeType = NodeType.EATER; // Default to a base type, will be set in createShape
        this.movementType = MovementType.NEUTRAL; // Default movement type
        this.dyeColor = [0,0,0]; // Still needed for Emitter type
        this.neuronData = null;
        this.currentExertionLevel = 0; // New: For dynamic energy costs
        this.isGrabbing = false; // New: For NN-controlled grabbing state
        this.isDesignatedEye = false; // New: To identify the creature's primary eye point
        this.canBeGrabber = false; // New: Gene, false by default
        this.seesParticle = false;       // New: Eye state
        this.nearestParticleMagnitude = 0; // New: Eye state (distance)
        this.nearestParticleDirection = 0; // New: Eye state (angle)
    }
    applyForce(f) { this.force = this.force.add(f); }

    get isFixed() { // Getter for convenience, based on movementType OR if grabbing
        return this.movementType === MovementType.FIXED || this.isGrabbing;
    }

    update(dt) {
        if (this.isFixed || this.invMass === 0) { // isFixed getter now includes isGrabbing
            this.force = new Vec2();
            return;
        }

        const acceleration = this.force.mul(this.invMass);

        const tempX = this.pos.x;
        const tempY = this.pos.y;

        this.pos.x = 2 * this.pos.x - this.prevPos.x + acceleration.x * dt * dt;
        this.pos.y = 2 * this.pos.y - this.prevPos.y + acceleration.y * dt * dt;

        this.prevPos.x = tempX;
        this.prevPos.y = tempY;

        this.force = new Vec2();
    }

    draw(ctx) {
        // Draw interaction radii first (underneath the point)
        const exertion = this.currentExertionLevel || 0; // Default to 0 if undefined

        if (this.nodeType === NodeType.EATER) {
            const effectiveEatingRadiusMultiplier = EATING_RADIUS_MULTIPLIER_BASE + (EATING_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectiveEatingRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 165, 0, ${0.15 + exertion * 0.2})`; // Increased base opacity
            ctx.fill();
        }
        if (this.nodeType === NodeType.PREDATOR) {
            const effectivePredationRadiusMultiplier = PREDATION_RADIUS_MULTIPLIER_BASE + (PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectivePredationRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 50, 50, ${0.15 + exertion * 0.2})`; // Increased base opacity
            ctx.fill();
        }


        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
        let mainColor = this.isFixed ? 'rgba(255,0,0,0.9)' : this.color;
        if (this.nodeType === NodeType.NEURON) {
            mainColor = 'rgba(200, 100, 255, 0.9)';
        } else if (this.nodeType === NodeType.PREDATOR) {
            mainColor = 'rgba(255, 50, 50, 0.9)';
        } else if (this.nodeType === NodeType.SWIMMER) {
             mainColor = 'rgba(0,200,255,0.9)'; // Bright blue for Swimmer
        } else if (this.nodeType === NodeType.PHOTOSYNTHETIC) { 
            mainColor = 'rgba(60, 179, 113, 0.9)'; // MediumSeaGreen for photosynthesis
        } else if (this.nodeType === NodeType.EMITTER) {
            mainColor = 'rgba(0,255,100,0.9)';
        } else if (this.nodeType === NodeType.EATER) {
            mainColor = 'rgba(255,165,0,0.9)';
        } else if (this.nodeType === NodeType.EYE) { // New Eye color
            mainColor = 'rgba(180, 180, 250, 0.9)'; // Light purple/blue for Eye
        }
        ctx.fillStyle = mainColor;
        ctx.fill();

        // Glow effect using shadow
        ctx.save(); // Save context state before applying shadow
        ctx.shadowColor = mainColor;
        ctx.shadowBlur = 7;
        ctx.fill(); // Re-fill to apply shadow as a glow
        ctx.restore(); // Restore context state to remove shadow for subsequent drawings


        if (this.nodeType === NodeType.NEURON) {
            ctx.beginPath(); // Inner dot for neuron
            ctx.arc(this.pos.x, this.pos.y, this.radius * 0.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(250, 200, 255, 0.9)';
            ctx.fill();
        }
        // ctx.closePath(); // Already closed by fill()
    }
}

// --- Spring Class (Soft Body) ---
class Spring {
    constructor(p1, p2, stiffness, dampingFactor, restLength = null, isRigid = false) {
        this.p1 = p1;
        this.p2 = p2;
        this.isRigid = isRigid;
        if (this.isRigid) {
            this.stiffness = RIGID_SPRING_STIFFNESS;
            this.dampingFactor = RIGID_SPRING_DAMPING;
        } else {
            this.stiffness = stiffness;
            this.dampingFactor = dampingFactor;
        }
        this.restLength = restLength === null ? p1.pos.sub(p2.pos).mag() : restLength;
    }
    applyForce() {
        const diffPos = this.p1.pos.sub(this.p2.pos);
        const currentLength = diffPos.mag();
        if (currentLength === 0) return;
        const displacement = currentLength - this.restLength;
        const direction = diffPos.normalize();

        const springForceMagnitude = -this.stiffness * displacement;
        const springForce = direction.mul(springForceMagnitude);

        const p1_vel_implicit = this.p1.pos.sub(this.p1.prevPos);
        const p2_vel_implicit = this.p2.pos.sub(this.p2.prevPos);
        const relVel_implicit = p1_vel_implicit.sub(p2_vel_implicit);

        const velAlongSpring = Vec2.dot(relVel_implicit, direction);
        const dampingForceMagnitude = -this.dampingFactor * velAlongSpring;
        const dampingForce = direction.mul(dampingForceMagnitude);

        const totalForce = springForce.add(dampingForce);
        this.p1.applyForce(totalForce);
        this.p2.applyForce(totalForce.mul(-1));
    }
    draw(ctx) {
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(this.p1.pos.x, this.p1.pos.y);
        ctx.lineTo(this.p2.pos.x, this.p2.pos.y);
        ctx.strokeStyle = this.isRigid ? 'rgba(255, 255, 0, 0.9)' : 'rgba(150,150,150,0.6)'; // Bright yellow for rigid springs
        ctx.lineWidth = this.isRigid ? 3 : 2; // Thicker line for rigid springs
        ctx.stroke();
    }
}

// --- SoftBody Class ---
class SoftBody {
    constructor(id, initialX, initialY, parentBody = null) {
        this.id = id;
        this.massPoints = [];
        this.springs = [];

        // Genetic blueprint
        this.blueprintPoints = []; // Array of { relX, relY, radius, mass, nodeType, movementType, dyeColor, canBeGrabber, neuronDataBlueprint }
        this.blueprintSprings = []; // Array of { p1Index, p2Index, restLength, isRigid } (indices refer to blueprintPoints)

        this.isUnstable = false;
        this.ticksSinceBirth = 0;
        this.canReproduce = false;
        this.shapeType = parentBody ? parentBody.shapeType : Math.floor(Math.random() * 3);
        this.justReproduced = false; // New: Flag for reproduction reward

        this.energyGainedFromPhotosynthesis = 0;
        this.energyGainedFromEating = 0;
        this.energyGainedFromPredation = 0;

        // New: Energy cost accumulators
        this.energyCostFromBaseNodes = 0;
        this.energyCostFromEmitterNodes = 0;
        this.energyCostFromEaterNodes = 0;
        this.energyCostFromPredatorNodes = 0;
        this.energyCostFromNeuronNodes = 0;
        this.energyCostFromSwimmerNodes = 0;
        this.energyCostFromPhotosyntheticNodes = 0;
        this.energyCostFromGrabbingNodes = 0;
        this.energyCostFromEyeNodes = 0;

        this.currentMaxEnergy = BASE_MAX_CREATURE_ENERGY; // Initial placeholder

        // Initialize heritable/mutable properties
        if (parentBody) {
            this.stiffness = parentBody.stiffness * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            if (this.stiffness !== parentBody.stiffness) mutationStats.stiffness++;
            this.springDamping = parentBody.springDamping * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            if (this.springDamping !== parentBody.springDamping) mutationStats.damping++;
            
            let oldMotorInterval = parentBody.motorImpulseInterval;
            this.motorImpulseInterval = parentBody.motorImpulseInterval * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            if (Math.floor(this.motorImpulseInterval) !== Math.floor(oldMotorInterval)) mutationStats.motorInterval++;

            let oldMotorCap = parentBody.motorImpulseMagnitudeCap;
            this.motorImpulseMagnitudeCap = parentBody.motorImpulseMagnitudeCap * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            if (this.motorImpulseMagnitudeCap !== oldMotorCap) mutationStats.motorCap++;

            let oldEmitterStrength = parentBody.emitterStrength;
            this.emitterStrength = parentBody.emitterStrength * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            if (this.emitterStrength !== oldEmitterStrength) mutationStats.emitterStrength++;

            let offspringNumChange = (Math.random() < Math.max(0, Math.min(1, MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER))) ? (Math.random() < 0.5 ? -1 : 1) : 0;
            this.numOffspring = parentBody.numOffspring + offspringNumChange;
            if (offspringNumChange !== 0) mutationStats.numOffspring++;

            let oldOffspringSpawnRadius = parentBody.offspringSpawnRadius;
            this.offspringSpawnRadius = parentBody.offspringSpawnRadius * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 0.5));
            if (this.offspringSpawnRadius !== oldOffspringSpawnRadius) mutationStats.offspringSpawnRadius++;
            
            let oldPointAddChance = parentBody.pointAddChance;
            this.pointAddChance = parentBody.pointAddChance * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 2));
            if (this.pointAddChance !== oldPointAddChance) mutationStats.pointAddChanceGene++;

            let oldSpringConnectionRadius = parentBody.springConnectionRadius;
            this.springConnectionRadius = parentBody.springConnectionRadius * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            if (this.springConnectionRadius !== oldSpringConnectionRadius) mutationStats.springConnectionRadiusGene++;
            
            if (parentBody.emitterDirection) {
                const oldEmitterDirX = parentBody.emitterDirection.x;
                const angleMutation = (Math.random() - 0.5) * Math.PI * 0.2 * GLOBAL_MUTATION_RATE_MODIFIER;
                const cosA = Math.cos(angleMutation);
                const sinA = Math.sin(angleMutation);
                this.emitterDirection = new Vec2(parentBody.emitterDirection.x * cosA - parentBody.emitterDirection.y * sinA, parentBody.emitterDirection.x * sinA + parentBody.emitterDirection.y * cosA).normalize();
                if (this.emitterDirection.x !== oldEmitterDirX) mutationStats.emitterDirection++; // Simplified check
            } else {
                this.emitterDirection = new Vec2(Math.random()*2-1, Math.random()*2-1).normalize();
                console.warn(`Parent body ${parentBody.id} was missing emitterDirection. Offspring ${this.id} gets random emitterDirection.`);
                mutationStats.emitterDirection++; // Count as a change if parent was missing it
            }
            
            let oldReproThreshold = parentBody.reproductionEnergyThreshold;
            this.reproductionEnergyThreshold = parentBody.reproductionEnergyThreshold; 
            // Mutation of reproductionEnergyThreshold happens later, after currentMaxEnergy is set for the offspring

            // Inherit and mutate activation pattern properties
            if (Math.random() < ACTIVATION_PATTERN_MUTATION_CHANCE) {
                const patterns = Object.values(ActivationPatternType);
                this.defaultActivationPattern = patterns[Math.floor(Math.random() * patterns.length)];
            } else {
                this.defaultActivationPattern = parentBody.defaultActivationPattern;
            }
            this.defaultActivationLevel = parentBody.defaultActivationLevel * (1 + (Math.random() - 0.5) * 2 * ACTIVATION_PARAM_MUTATION_MAGNITUDE);
            this.defaultActivationPeriod = parentBody.defaultActivationPeriod * (1 + (Math.random() - 0.5) * 2 * ACTIVATION_PARAM_MUTATION_MAGNITUDE);
            this.defaultActivationPhaseOffset = parentBody.defaultActivationPhaseOffset + (Math.random() - 0.5) * (parentBody.defaultActivationPeriod * 0.2); 
            
            // Inherit/Mutate Reward Strategy
            if (Math.random() < RLRewardStrategy_MUTATION_CHANCE) {
                const strategies = Object.values(RLRewardStrategy);
                let newStrategy = strategies[Math.floor(Math.random() * strategies.length)];
                if (newStrategy !== parentBody.rewardStrategy) {
                    this.rewardStrategy = newStrategy;
                    mutationStats.rewardStrategyChange++;
                } else {
                    if (strategies.length > 1) {
                        let tempStrategies = strategies.filter(s => s !== parentBody.rewardStrategy);
                        if (tempStrategies.length > 0) {
                            this.rewardStrategy = tempStrategies[Math.floor(Math.random() * tempStrategies.length)];
                            mutationStats.rewardStrategyChange++; 
                        } else {
                            this.rewardStrategy = parentBody.rewardStrategy;
                        }
                    } else {
                        this.rewardStrategy = parentBody.rewardStrategy;
                    }
                }
            } else {
                this.rewardStrategy = parentBody.rewardStrategy;
            }
            // RL Algorithm type inheritance (can add mutation if needed)
            this.rlAlgorithmType = parentBody.rlAlgorithmType || RLAlgorithmType.REINFORCE;

            // Inherit and mutate reproductionCooldownGene
            this.reproductionCooldownGene = parentBody.reproductionCooldownGene * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 0.2));
            this.reproductionCooldownGene = Math.max(50, Math.min(Math.floor(this.reproductionCooldownGene), 20000)); // Clamp
            if (this.reproductionCooldownGene !== parentBody.reproductionCooldownGene) {
                mutationStats.reproductionCooldownGene = (mutationStats.reproductionCooldownGene || 0) + 1;
            }

        } else {
            // Initial defaults for brand new creatures
            this.stiffness = 500 + Math.random() * 2500;
            this.springDamping = 5 + Math.random() * 20;
            this.motorImpulseInterval = 30 + Math.floor(Math.random() * 90);
            this.motorImpulseMagnitudeCap = 0.5 + Math.random() * 2.0;
            this.emitterStrength = 0.2 + Math.random() * 1.0;
            this.emitterDirection = new Vec2(Math.random()*2-1, Math.random()*2-1).normalize();
            this.numOffspring = 1 + Math.floor(Math.random() * 3);
            this.offspringSpawnRadius = 50 + Math.random() * 50;
            this.pointAddChance = 0.02 + Math.random() * 0.06;
            this.springConnectionRadius = 40 + Math.random() * 40;
            this.reproductionEnergyThreshold = BASE_MAX_CREATURE_ENERGY; // Will be refined based on actual max energy

            // Default Activation Pattern Properties for new creature
            const patterns = Object.values(ActivationPatternType);
            this.defaultActivationPattern = patterns[Math.floor(Math.random() * patterns.length)];
            this.defaultActivationLevel = DEFAULT_ACTIVATION_LEVEL_MIN + Math.random() * (DEFAULT_ACTIVATION_LEVEL_MAX - DEFAULT_ACTIVATION_LEVEL_MIN);
            this.defaultActivationPeriod = DEFAULT_ACTIVATION_PERIOD_MIN_TICKS + Math.floor(Math.random() * (DEFAULT_ACTIVATION_PERIOD_MAX_TICKS - DEFAULT_ACTIVATION_PERIOD_MIN_TICKS + 1));
            this.defaultActivationPhaseOffset = Math.random() * this.defaultActivationPeriod;
            
            // Default RL Algorithm for brand new creatures
            this.rlAlgorithmType = RLAlgorithmType.REINFORCE; 
            
            // New: Default Reward Strategy for brand new creatures
            const strategies = Object.values(RLRewardStrategy);
            this.rewardStrategy = strategies[Math.floor(Math.random() * strategies.length)];

            // Initialize reproductionCooldownGene for new creatures
            this.reproductionCooldownGene = 100 + Math.floor(Math.random() * 4901); // Random between 100 and 5000
        }

        // Clamp activation properties after they've been set/inherited/mutated
        this.defaultActivationLevel = Math.max(DEFAULT_ACTIVATION_LEVEL_MIN * 0.1, Math.min(this.defaultActivationLevel, DEFAULT_ACTIVATION_LEVEL_MAX * 2.0)); // Wider clamping for more variance
        this.defaultActivationPeriod = Math.max(DEFAULT_ACTIVATION_PERIOD_MIN_TICKS * 0.25, Math.min(this.defaultActivationPeriod, DEFAULT_ACTIVATION_PERIOD_MAX_TICKS * 2.0));
        this.defaultActivationPeriod = Math.floor(this.defaultActivationPeriod);
        if (this.defaultActivationPeriod === 0) this.defaultActivationPeriod = 1; // Avoid division by zero
        if (this.defaultActivationPhaseOffset < 0) this.defaultActivationPhaseOffset += this.defaultActivationPeriod;
        this.defaultActivationPhaseOffset %= this.defaultActivationPeriod;

        // Clamp other properties (already existing logic)
        this.stiffness = Math.max(100, Math.min(this.stiffness, 10000));
        this.springDamping = Math.max(0.1, Math.min(this.springDamping, 50));
        this.motorImpulseInterval = Math.max(10, Math.floor(this.motorImpulseInterval));
        this.motorImpulseMagnitudeCap = Math.max(0, Math.min(this.motorImpulseMagnitudeCap, 5.0));
        this.emitterStrength = Math.max(0, Math.min(this.emitterStrength, 3.0));
        this.numOffspring = Math.max(1, Math.min(this.numOffspring, 5));
        this.offspringSpawnRadius = Math.max(20, Math.min(this.offspringSpawnRadius, 150));
        this.pointAddChance = Math.max(0, Math.min(0.5, this.pointAddChance));
        this.springConnectionRadius = Math.max(10, Math.min(this.springConnectionRadius, 100));

        this.fluidEntrainment = BODY_FLUID_ENTRAINMENT_FACTOR;
        this.fluidCurrentStrength = FLUID_CURRENT_STRENGTH_ON_BODY;
        this.bodyPushStrength = SOFT_BODY_PUSH_STRENGTH;

        // 1. Create shape (points and initial springs)
        this.createShape(initialX, initialY, parentBody);

        // 2. Body Scale Mutation (if offspring)
        if (parentBody && Math.random() < BODY_SCALE_MUTATION_CHANCE) {
            const scaleFactor = 1.0 + (Math.random() - 0.5) * 2 * BODY_SCALE_MUTATION_MAGNITUDE;
            if (scaleFactor > 0.1 && Math.abs(scaleFactor - 1.0) > 0.001) { // Check if it actually scaled
                this.springs.forEach(spring => {
                    spring.restLength *= scaleFactor;
                    spring.restLength = Math.max(1, spring.restLength); 
                });
                this.massPoints.forEach(point => {
                    point.radius *= scaleFactor;
                    point.radius = Math.max(0.5, point.radius); 
                });
                this.offspringSpawnRadius *= scaleFactor;
                this.offspringSpawnRadius = Math.max(10, this.offspringSpawnRadius); 
                mutationStats.bodyScale++;
            }
        }

        // 3. Calculate dynamic max energy based on final point count
        this.calculateCurrentMaxEnergy(); 

        // Calculate effective reproduction cooldown based on gene and point count
        this.effectiveReproductionCooldown = Math.floor(this.reproductionCooldownGene * (1 + (0.2 * Math.max(0, this.massPoints.length - 1))));
        if (this.massPoints.length === 0) { // Safety for empty body after blueprint instantiation (should not happen ideally)
             this.effectiveReproductionCooldown = Math.floor(this.reproductionCooldownGene);
        }

        // 4. Set initial creature energy
        this.creatureEnergy = this.currentMaxEnergy * OFFSPRING_INITIAL_ENERGY_SHARE; 

        let oldReproThresholdForStat = this.reproductionEnergyThreshold; // Capture before mutation
        if (parentBody) {
            this.reproductionEnergyThreshold = this.reproductionEnergyThreshold * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 0.2));
        } else {
            this.reproductionEnergyThreshold = this.currentMaxEnergy * (0.75 + Math.random() * 0.2);
        }
        this.reproductionEnergyThreshold = Math.max(this.currentMaxEnergy * 0.05, Math.min(this.reproductionEnergyThreshold, this.currentMaxEnergy));
        this.reproductionEnergyThreshold = Math.round(this.reproductionEnergyThreshold);
        if (this.reproductionEnergyThreshold !== oldReproThresholdForStat && parentBody) { // Only count if from parent and changed
            mutationStats.reproductionEnergyThreshold++;
        }
        
        this.primaryEyePoint = null; // New: For the creature's main eye
        if (this.massPoints.length > 0) {
            // Attempt to find an existing EYE node to designate as primary
            for (const point of this.massPoints) {
                if (point.nodeType === NodeType.EYE) {
                    this.primaryEyePoint = point;
                    point.isDesignatedEye = true;
                    break; 
                }
            }
            // If no EYE node found, and we want to ensure one, we could designate/change one here.
            // For now, it's only set if an EYE node is already present (e.g. from mutation or initial gen).
            // If still null, the NN input for eye will be default/neutral.
        }

        this.initializeBrain(); 
    }

    calculateCurrentMaxEnergy() {
        if (this.massPoints.length === 0) {
            this.currentMaxEnergy = 0; 
        } else {
            this.currentMaxEnergy = this.massPoints.length * ENERGY_PER_MASS_POINT_BONUS;
            // Ensure currentMaxEnergy is at least 1 if points exist and ENERGY_PER_MASS_POINT_BONUS might be zero or very small.
            if (this.currentMaxEnergy < 1) { 
                this.currentMaxEnergy = 1;
            }
        }
    }

    createShape(startX, startY, parentBody = null) {
        this.massPoints = []; // Clear actual points
        this.springs = [];  // Clear actual springs
        this.blueprintPoints = []; // Clear any previous blueprint
        this.blueprintSprings = [];// Clear any previous blueprint

        const baseRadius = 1 + Math.random() * 1; 
        const availableFunctionalNodeTypes = [NodeType.PREDATOR, NodeType.EATER, NodeType.PHOTOSYNTHETIC, NodeType.NEURON, NodeType.EMITTER, NodeType.SWIMMER, NodeType.EYE];
        const dyeColorChoices = [DYE_COLORS.RED, DYE_COLORS.GREEN, DYE_COLORS.BLUE];

        if (parentBody) {
            // --- Reproduction: Inherit and Mutate Blueprint ---

            // 1. Deep copy blueprint from parent
            this.blueprintPoints = JSON.parse(JSON.stringify(parentBody.blueprintPoints));
            this.blueprintSprings = JSON.parse(JSON.stringify(parentBody.blueprintSprings));

            // 2. Mutate blueprint points (coordinates, types, properties)
            this.blueprintPoints.forEach(bp => {
                // Mutate relative coordinates
                if (Math.random() < (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 0.5)) {
                    bp.relX += (Math.random() - 0.5) * 2; // Smaller jitter for blueprint stability
                    mutationStats.blueprintCoordinateChange = (mutationStats.blueprintCoordinateChange || 0) + 1;
                }
                if (Math.random() < (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 0.5)) {
                    bp.relY += (Math.random() - 0.5) * 2;
                    mutationStats.blueprintCoordinateChange = (mutationStats.blueprintCoordinateChange || 0) + 1;
                }

                // Mutate mass & radius
                if (Math.random() < (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.mass = Math.max(0.1, Math.min(bp.mass * (1 + (Math.random() - 0.5) * 0.4), 1.0));
                    mutationStats.blueprintMassRadiusChange = (mutationStats.blueprintMassRadiusChange || 0) + 1;
                }
                if (Math.random() < (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.radius = Math.max(0.5, Math.min(bp.radius * (1 + (Math.random() - 0.5) * 0.4), baseRadius * 2.5)); // Max based on baseRadius
                     mutationStats.blueprintMassRadiusChange = (mutationStats.blueprintMassRadiusChange || 0) + 1;
                }

                // Mutate nodeType
                if (Math.random() < (MUTATION_CHANCE_NODE_TYPE * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    const oldNodeType = bp.nodeType;
                    bp.nodeType = availableFunctionalNodeTypes[Math.floor(Math.random() * availableFunctionalNodeTypes.length)];
                    if (bp.nodeType !== oldNodeType) mutationStats.nodeTypeChange++;
                }

                // Mutate movementType
                if (Math.random() < (MUTATION_CHANCE_NODE_TYPE * GLOBAL_MUTATION_RATE_MODIFIER)) { 
                    const oldMovementType = bp.movementType;
                    const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                    bp.movementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                    if (bp.movementType !== oldMovementType) mutationStats.movementTypeChange++;
                }
                // Ensure Swimmer nodes are not Floating after potential mutation
                if (bp.nodeType === NodeType.SWIMMER && bp.movementType === MovementType.FLOATING) {
                    bp.movementType = (Math.random() < 0.5) ? MovementType.NEUTRAL : MovementType.FIXED;
                }

                // Mutate canBeGrabber gene
                if (Math.random() < GRABBER_GENE_MUTATION_CHANCE) {
                    bp.canBeGrabber = !bp.canBeGrabber;
                    mutationStats.grabberGeneChange++;
                }

                // Mutate dyeColor
                if (Math.random() < (MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    mutationStats.blueprintDyeColorChange = (mutationStats.blueprintDyeColorChange || 0) + 1;
                }

                // Mutate neuronDataBlueprint (specifically hiddenLayerSize if neuron)
                if (bp.nodeType === NodeType.NEURON) {
                    if (!bp.neuronDataBlueprint) { // Ensure it exists
                        bp.neuronDataBlueprint = { hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) };
                    }
                    if (Math.random() < (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER)) {
                        let newSize = bp.neuronDataBlueprint.hiddenLayerSize + Math.floor((Math.random() * 6) - 3); // Mutate by +/- up to 3
                        bp.neuronDataBlueprint.hiddenLayerSize = Math.max(DEFAULT_HIDDEN_LAYER_SIZE_MIN, Math.min(newSize, DEFAULT_HIDDEN_LAYER_SIZE_MAX));
                        mutationStats.blueprintNeuronHiddenSizeChange = (mutationStats.blueprintNeuronHiddenSizeChange || 0) + 1;
                    }
                } else {
                    bp.neuronDataBlueprint = null; // Crucial: ensure non-neurons have null neuronDataBlueprint
                }
            });

            // 3. Mutate blueprint springs (restLength, isRigid)
            this.blueprintSprings.forEach(bs => {
                if (Math.random() < (SPRING_PROP_MUTATION_MAGNITUDE * GLOBAL_MUTATION_RATE_MODIFIER)) { // Use magnitude as chance here
                    const oldRestLength = bs.restLength;
                    bs.restLength = Math.max(1, bs.restLength * (1 + (Math.random() - 0.5) * 2 * SPRING_PROP_MUTATION_MAGNITUDE));
                    if (Math.abs(bs.restLength - oldRestLength) > 0.01) mutationStats.springRestLength++;
                }
                if (Math.random() < (MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    const oldRigid = bs.isRigid;
                    bs.isRigid = !bs.isRigid; // Simple flip for now
                    if (bs.isRigid !== oldRigid) mutationStats.springRigidityFlip++;
                }
            });

            // Step 4: Structural Blueprint Mutations (Point Add, Spring Add/Delete, Subdivision, Scale, etc.)
            // --- Point Addition Mutation (Blueprint) ---
            if (Math.random() < this.pointAddChance * GLOBAL_MUTATION_RATE_MODIFIER && this.blueprintPoints.length > 0) {
                const lastBp = this.blueprintPoints[this.blueprintPoints.length - 1];
                const newRelX = lastBp.relX + (Math.random() - 0.5) * NEW_POINT_OFFSET_RADIUS * 0.5; // Smaller offset for blueprint
                const newRelY = lastBp.relY + (Math.random() - 0.5) * NEW_POINT_OFFSET_RADIUS * 0.5;
                const newMass = 0.1 + Math.random() * 0.9;
                const newRadius = baseRadius * (0.8 + Math.random() * 0.4);
                let newNodeType = availableFunctionalNodeTypes[Math.floor(Math.random() * availableFunctionalNodeTypes.length)];
                const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                let newMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                if (newNodeType === NodeType.SWIMMER && newMovementType === MovementType.FLOATING) {
                    newMovementType = (Math.random() < 0.5) ? MovementType.NEUTRAL : MovementType.FIXED;
                }
                const newBp = {
                    relX: newRelX, relY: newRelY, radius: newRadius, mass: newMass,
                    nodeType: newNodeType, movementType: newMovementType,
                    dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                    canBeGrabber: Math.random() < GRABBER_GENE_MUTATION_CHANCE,
                    neuronDataBlueprint: newNodeType === NodeType.NEURON ? { hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) } : null
                };
                this.blueprintPoints.push(newBp);
                const newPointIndex = this.blueprintPoints.length - 1;
                mutationStats.pointAddActual++;

                // Connect new blueprint point with springs
                const numSpringsToAddNewPoint = MIN_SPRINGS_PER_NEW_NODE + Math.floor(Math.random() * (MAX_SPRINGS_PER_NEW_NODE - MIN_SPRINGS_PER_NEW_NODE + 1));
                const existingBpIndices = this.blueprintPoints.map((_, i) => i).filter(i => i !== newPointIndex);
                const shuffledExistingBpIndices = existingBpIndices.sort(() => 0.5 - Math.random());

                for (let k = 0; k < Math.min(numSpringsToAddNewPoint, shuffledExistingBpIndices.length); k++) {
                    const connectToBpIndex = shuffledExistingBpIndices[k];
                    const connectToBp = this.blueprintPoints[connectToBpIndex];
                    const dist = Math.sqrt((newBp.relX - connectToBp.relX)**2 + (newBp.relY - connectToBp.relY)**2);
                    let newRestLength = dist * (1 + (Math.random() - 0.5) * 2 * NEW_SPRING_REST_LENGTH_VARIATION);
                    newRestLength = Math.max(1, newRestLength);
                    const becomeRigid = Math.random() < CHANCE_FOR_RIGID_SPRING;
                    this.blueprintSprings.push({ p1Index: newPointIndex, p2Index: connectToBpIndex, restLength: newRestLength, isRigid: becomeRigid });
                }
            }

            // --- Spring Deletion Mutation (Blueprint) ---
            if (this.blueprintSprings.length > this.blueprintPoints.length -1 && Math.random() < SPRING_DELETION_CHANCE) { // Ensure min connectivity
                // Complex check to avoid orphaning needed here for blueprint springs
                // For now, simple random deletion if enough springs exist
                const springToDeleteIndex = Math.floor(Math.random() * this.blueprintSprings.length);
                this.blueprintSprings.splice(springToDeleteIndex, 1);
                mutationStats.springDeletion++;
            }

            // --- Spring Addition Mutation (Blueprint) ---
            if (this.blueprintPoints.length >= 2 && Math.random() < SPRING_ADDITION_CHANCE) {
                let attempts = 0;
                while (attempts < 10) {
                    const idx1 = Math.floor(Math.random() * this.blueprintPoints.length);
                    let idx2 = Math.floor(Math.random() * this.blueprintPoints.length);
                    if (idx1 === idx2 && this.blueprintPoints.length > 1) {
                        idx2 = (idx1 + 1) % this.blueprintPoints.length;
                    }
                    if (idx1 === idx2) break;

                    const pA_bp = this.blueprintPoints[idx1];
                    const pB_bp = this.blueprintPoints[idx2];
                    let alreadyConnected = this.blueprintSprings.some(bs => 
                        (bs.p1Index === idx1 && bs.p2Index === idx2) || (bs.p1Index === idx2 && bs.p2Index === idx1)
                    );
                    if (!alreadyConnected) {
                        const dist = Math.sqrt((pA_bp.relX - pB_bp.relX)**2 + (pA_bp.relY - pB_bp.relY)**2);
                        let newRestLength = dist * (1 + (Math.random() - 0.5) * 2 * NEW_SPRING_REST_LENGTH_VARIATION);
                        newRestLength = Math.max(1, newRestLength);
                        const becomeRigid = Math.random() < CHANCE_FOR_RIGID_SPRING;
                        this.blueprintSprings.push({ p1Index: idx1, p2Index: idx2, restLength: newRestLength, isRigid: becomeRigid });
                        mutationStats.springAddition++; 
                        break;
                    }
                    attempts++;
                }
            }
            
            // --- Spring Subdivision Mutation (Blueprint) ---
            if (this.blueprintSprings.length > 0 && Math.random() < SPRING_SUBDIVISION_MUTATION_CHANCE) {
                const springToSubdivideIndex = Math.floor(Math.random() * this.blueprintSprings.length);
                const originalBs = this.blueprintSprings[springToSubdivideIndex];
                    const bp1 = this.blueprintPoints[originalBs.p1Index];
                    const bp2 = this.blueprintPoints[originalBs.p2Index];

                    const midRelX = (bp1.relX + bp2.relX) / 2;
                    const midRelY = (bp1.relY + bp2.relY) / 2;
                    const newRadius = ((bp1.radius || baseRadius) + (bp2.radius || baseRadius)) / 2 * (0.8 + Math.random() * 0.4);
                    const newMass = ((bp1.mass || 0.5) + (bp2.mass || 0.5)) / 2 * (0.8 + Math.random() * 0.4);
                    let newNodeType = availableFunctionalNodeTypes[Math.floor(Math.random() * availableFunctionalNodeTypes.length)];
                    const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                    let newMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                     if (newNodeType === NodeType.SWIMMER && newMovementType === MovementType.FLOATING) {
                        newMovementType = (Math.random() < 0.5) ? MovementType.NEUTRAL : MovementType.FIXED;
                    }

                    const newMidBp = {
                        relX: midRelX, relY: midRelY, radius: Math.max(0.5, newRadius), mass: Math.max(0.1, newMass),
                        nodeType: newNodeType, movementType: newMovementType,
                        dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                        canBeGrabber: Math.random() < GRABBER_GENE_MUTATION_CHANCE,
                        neuronDataBlueprint: newNodeType === NodeType.NEURON ? { hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) } : null
                    };
                    this.blueprintPoints.push(newMidBp);
                    const newMidPointIndex = this.blueprintPoints.length - 1;

                    this.blueprintSprings.splice(springToSubdivideIndex, 1); // Remove original spring

                    let restLength1 = Math.sqrt((bp1.relX - midRelX)**2 + (bp1.relY - midRelY)**2) * (1 + (Math.random() - 0.5) * 0.1);
                    this.blueprintSprings.push({ p1Index: originalBs.p1Index, p2Index: newMidPointIndex, restLength: Math.max(1, restLength1), isRigid: originalBs.isRigid });

                    let restLength2 = Math.sqrt((midRelX - bp2.relX)**2 + (midRelY - bp2.relY)**2) * (1 + (Math.random() - 0.5) * 0.1);
                    this.blueprintSprings.push({ p1Index: newMidPointIndex, p2Index: originalBs.p2Index, restLength: Math.max(1, restLength2), isRigid: originalBs.isRigid });
                    
                    mutationStats.springSubdivision++; 
                    mutationStats.pointAddActual++; 
            }

            // --- Body Scale Mutation (Blueprint) ---
            if (Math.random() < BODY_SCALE_MUTATION_CHANCE) {
                const scaleFactor = 1.0 + (Math.random() - 0.5) * 2 * BODY_SCALE_MUTATION_MAGNITUDE;
                if (scaleFactor > 0.1 && Math.abs(scaleFactor - 1.0) > 0.001) {
                    this.blueprintPoints.forEach(bp => {
                        bp.relX *= scaleFactor;
                        bp.relY *= scaleFactor;
                        bp.radius = Math.max(0.5, bp.radius * scaleFactor); 
                    });
                    this.blueprintSprings.forEach(bs => {
                        bs.restLength = Math.max(1, bs.restLength * scaleFactor); 
                    });
                    this.offspringSpawnRadius *= scaleFactor; // This is a creature property, not blueprint, but scale it too
                    this.offspringSpawnRadius = Math.max(10, this.offspringSpawnRadius); 
                    mutationStats.bodyScale++;
                }
            }
            // TODO: Add blueprint versions of Segment Duplication, Symmetrical Duplication etc.

            // Step 5: Instantiate Phenotype from the mutated blueprint
            this._instantiatePhenotypeFromBlueprint(startX, startY);

        } else { 
            // --- Initial Generation: Create Blueprint from Geometric Primitives ---
            let initialTempMassPoints = []; // Temporary MassPoint objects to get initial geometry
            let initialTempSprings = [];  // Temporary Spring objects

            // Create initial geometric shape (grid, line, or star) using startX, startY as a reference origin for now
            const basePointDist = 5 + Math.random() * 3; 
            if (this.shapeType === 0) { // Grid
                const numPointsX = 3; const numPointsY = 3; let gridPoints = [];
                for (let i = 0; i < numPointsY; i++) { gridPoints[i] = []; for (let j = 0; j < numPointsX; j++) {
                    // Position points relative to an arbitrary origin (0,0) for now, will adjust with centroid
                    const point = new MassPoint(j * basePointDist, i * basePointDist, 0.3 + Math.random() * 0.4, baseRadius);
                    initialTempMassPoints.push(point); gridPoints[i][j] = point;
                }}
                for (let i=0; i<numPointsY; i++) for (let j=0; j<numPointsX-1; j++) initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i][j+1], this.stiffness, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                for (let j=0; j<numPointsX; j++) for (let i=0; i<numPointsY-1; i++) initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i+1][j], this.stiffness, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                for (let i=0; i<numPointsY-1; i++) for (let j=0; j<numPointsX-1; j++) {
                    initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i+1][j+1], this.stiffness*0.7, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                    initialTempSprings.push(new Spring(gridPoints[i+1][j], gridPoints[i][j+1], this.stiffness*0.7, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                }
            } else if (this.shapeType === 1) { // Line
                const numLinePoints = Math.floor(3 + Math.random() * 3); const isHorizontal = Math.random() < 0.5; let linePoints = [];
                for (let i=0; i<numLinePoints; i++) {
                    const x = (isHorizontal ? i * basePointDist : 0);
                    const y = (isHorizontal ? 0 : i * basePointDist);
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    initialTempMassPoints.push(point); linePoints.push(point);
                }
                for (let i=0; i<numLinePoints-1; i++) initialTempSprings.push(new Spring(linePoints[i], linePoints[i+1], this.stiffness, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                if (numLinePoints > 2) initialTempSprings.push(new Spring(linePoints[0], linePoints[numLinePoints-1], this.stiffness*0.5, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
            } else { // Star
                const numOuterPoints = Math.floor(4 + Math.random()*3);
                const centralPoint = new MassPoint(0, 0, (0.3+Math.random()*0.4)*1.5, baseRadius*1.2); // Center at (0,0) for now
                initialTempMassPoints.push(centralPoint);
                const circleRadius = basePointDist * 1.5;
                for (let i=0; i<numOuterPoints; i++) {
                    const angle = (i / numOuterPoints) * Math.PI * 2;
                    const x = Math.cos(angle)*circleRadius;
                    const y = Math.sin(angle)*circleRadius;
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    initialTempMassPoints.push(point);
                    initialTempSprings.push(new Spring(centralPoint, point, this.stiffness, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                    if (i>0) initialTempSprings.push(new Spring(initialTempMassPoints[initialTempMassPoints.length-2], point, this.stiffness*0.8, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
                }
                if (numOuterPoints > 1) initialTempSprings.push(new Spring(initialTempMassPoints[1], initialTempMassPoints[initialTempMassPoints.length-1], this.stiffness*0.8, this.springDamping, null, Math.random() < CHANCE_FOR_RIGID_SPRING));
            }

            // Calculate centroid of these initial temporary points (which were created around a local 0,0)
            let centroidX = 0, centroidY = 0;
            if (initialTempMassPoints.length > 0) {
                initialTempMassPoints.forEach(p => { centroidX += p.pos.x; centroidY += p.pos.y; });
                centroidX /= initialTempMassPoints.length;
                centroidY /= initialTempMassPoints.length;
            } 
            
            // Populate blueprintPoints from initialTempMassPoints, making coordinates relative to their own centroid
            initialTempMassPoints.forEach(p_temp => {
                let chosenNodeType = availableFunctionalNodeTypes[Math.floor(Math.random() * availableFunctionalNodeTypes.length)];
                const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                let chosenMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                if (chosenNodeType === NodeType.SWIMMER && chosenMovementType === MovementType.FLOATING) {
                    chosenMovementType = (Math.random() < 0.5) ? MovementType.NEUTRAL : MovementType.FIXED;
                }
                const canBeGrabberInitial = Math.random() < GRABBER_GENE_MUTATION_CHANCE;
                const neuronDataBp = chosenNodeType === NodeType.NEURON ? {
                    hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                } : null;

                this.blueprintPoints.push({
                    relX: p_temp.pos.x - centroidX, // Store relative to calculated centroid of the initial shape
                    relY: p_temp.pos.y - centroidY,
                    radius: p_temp.radius,
                    mass: p_temp.mass,
                    nodeType: chosenNodeType,
                    movementType: chosenMovementType,
                    dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                    canBeGrabber: canBeGrabberInitial,
                    neuronDataBlueprint: neuronDataBp
                });
            });

            // Populate blueprintSprings from initialTempSprings
            initialTempSprings.forEach(s_temp => {
                const p1Index = initialTempMassPoints.indexOf(s_temp.p1);
                const p2Index = initialTempMassPoints.indexOf(s_temp.p2);
                if (p1Index !== -1 && p2Index !== -1) {
                    this.blueprintSprings.push({
                        p1Index: p1Index,
                        p2Index: p2Index,
                        restLength: s_temp.restLength, // This was calculated by Spring constructor from initial geometry
                        isRigid: s_temp.isRigid
                    });
                }
            });
            
            // Instantiate actual phenotype using the new blueprint.
            // The `startX, startY` provided to createShape becomes the target world position for the *centroid* of this new creature.
            this._instantiatePhenotypeFromBlueprint(startX, startY); 
        }

        // Common post-creation neuron linking (for both initial and reproduced)
        this.massPoints.forEach((p, idx) => {
            if (p.nodeType === NodeType.NEURON && p.neuronData) {
                if (this.massPoints.length > 1) {
                    let newSensorIndex;
                    do { newSensorIndex = Math.floor(Math.random() * this.massPoints.length); } while (newSensorIndex === idx);
                    p.neuronData.sensorPointIndex = newSensorIndex;
                    } else {
                    p.neuronData.sensorPointIndex = -1;
                }
                const effectorCandidates = this.massPoints.map((ep, epIdx) => ep.nodeType === NodeType.EMITTER ? epIdx : -1).filter(epIdx => epIdx !== -1 && epIdx !== idx);
                if (effectorCandidates.length > 0) {
                     p.neuronData.effectorPointIndex = effectorCandidates[Math.floor(Math.random() * effectorCandidates.length)];
                } else {
                    p.neuronData.effectorPointIndex = -1;
                }
            }
        });
    }

    _instantiatePhenotypeFromBlueprint(spawnX, spawnY) {
        this.massPoints = [];
        this.springs = [];

        // 1. Instantiate MassPoints from blueprintPoints
        for (const bp of this.blueprintPoints) {
            // Calculate absolute world position
            const worldX = spawnX + bp.relX;
            const worldY = spawnY + bp.relY;

            const newPoint = new MassPoint(worldX, worldY, bp.mass, bp.radius);
            newPoint.nodeType = bp.nodeType;
            newPoint.movementType = bp.movementType;
            newPoint.dyeColor = [...bp.dyeColor]; // Ensure deep copy for array
            newPoint.canBeGrabber = bp.canBeGrabber;

            if (bp.neuronDataBlueprint) {
                newPoint.neuronData = {
                    // Copy properties from neuronDataBlueprint
                    // isBrain will be determined later in initializeBrain
                    isBrain: false, // Default, will be set by initializeBrain if this becomes the brain
                    hiddenLayerSize: bp.neuronDataBlueprint.hiddenLayerSize,
                    // sensorPointIndex and effectorPointIndex will be linked later
                    sensorPointIndex: -1, 
                    effectorPointIndex: -1
                };
            } else {
                newPoint.neuronData = null;
            }
            this.massPoints.push(newPoint);
        }

        // 2. Instantiate Springs from blueprintSprings
        for (const bs of this.blueprintSprings) {
            if (bs.p1Index >= 0 && bs.p1Index < this.massPoints.length &&
                bs.p2Index >= 0 && bs.p2Index < this.massPoints.length) {
                
                const p1 = this.massPoints[bs.p1Index];
                const p2 = this.massPoints[bs.p2Index];

                // Use the creature's overall stiffness and damping
                // The blueprint spring carries restLength and isRigid
                this.springs.push(new Spring(p1, p2, this.stiffness, this.springDamping, bs.restLength, bs.isRigid));
            } else {
                console.warn(`Body ${this.id}: Invalid spring blueprint indices ${bs.p1Index}, ${bs.p2Index} for ${this.massPoints.length} points.`);
            }
        }

        // Recalculate things that depend on the final massPoints
        this.calculateCurrentMaxEnergy(); 
        // Note: initializeBrain() and primaryEyePoint assignment will happen after createShape() finishes
        // in the main constructor flow.
    }

    // --- Main Update Method ---
    updateSelf(dt, fluidFieldRef) {
        if (this.isUnstable) return;

        this._updateSensoryInputsAndDefaultActivations(fluidFieldRef, nutrientField, lightField); // Removed particles argument
        
        let brainNode = this.massPoints.find(p => p.neuronData && p.neuronData.isBrain);

        if (brainNode) {
            this._processBrain(brainNode, dt, fluidFieldRef, nutrientField, lightField); // Removed particles argument
        } else {
            this._applyFallbackBehaviors(dt, fluidFieldRef);
        }

        this._updateEnergyBudget(dt, fluidFieldRef, nutrientField, lightField); // nutrientField and lightField are global
        if (this.isUnstable) return; // Death from energy budget check

        this._performPhysicalUpdates(dt, fluidFieldRef);
        if (this.isUnstable) return; // Instability from physical updates

        this._finalizeUpdateAndCheckStability(dt); // dt might be needed for some interaction logic if it moves here

    }

    // --- Refactored Helper Methods (Shells) ---
    _updateSensoryInputsAndDefaultActivations(fluidFieldRef, nutrientField, lightField) { // Removed particles argument
        this._applyDefaultActivationPatterns();
        this._updateEyeNodes(); // Removed particles argument
    }

    _applyDefaultActivationPatterns() {
        this.massPoints.forEach(point => {
            let baseActivation = 0;
            const timeFactor = (this.ticksSinceBirth + this.defaultActivationPhaseOffset) / Math.max(1, this.defaultActivationPeriod);

            switch (this.defaultActivationPattern) {
                case ActivationPatternType.FLAT:
                    baseActivation = this.defaultActivationLevel;
                    break;
                case ActivationPatternType.SINE:
                    baseActivation = this.defaultActivationLevel * (Math.sin(2 * Math.PI * timeFactor) * 0.5 + 0.5); // Ranges 0 to level
                    break;
                case ActivationPatternType.PULSE:
                    // Simple pulse: on for 10% of period, uses defaultActivationLevel as max
                    baseActivation = (timeFactor % 1.0 < 0.1) ? this.defaultActivationLevel : 0;
                    break;
            }
            point.currentExertionLevel = Math.max(0, Math.min(1, baseActivation)); // Clamp to 0-1
        });
    }

    _updateEyeNodes() { // Removed particles argument
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.EYE) {
                point.seesParticle = false; // Reset first
                point.nearestParticleMagnitude = 0;
                point.nearestParticleDirection = 0;
                let closestDistSq = EYE_DETECTION_RADIUS * EYE_DETECTION_RADIUS;
                let nearestParticleFound = null;

                // Determine the grid cell range to check based on EYE_DETECTION_RADIUS
                const eyeGxMin = Math.max(0, Math.floor((point.pos.x - EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));
                const eyeGxMax = Math.min(GRID_COLS - 1, Math.floor((point.pos.x + EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));
                const eyeGyMin = Math.max(0, Math.floor((point.pos.y - EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));
                const eyeGyMax = Math.min(GRID_ROWS - 1, Math.floor((point.pos.y + EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));

                for (let gy = eyeGyMin; gy <= eyeGyMax; gy++) {
                    for (let gx = eyeGxMin; gx <= eyeGxMax; gx++) {
                        const cellIndex = gx + gy * GRID_COLS;
                        if (spatialGrid[cellIndex] && spatialGrid[cellIndex].length > 0) {
                            const cellBucket = spatialGrid[cellIndex];
                            for (const item of cellBucket) {
                                if (item.type === 'particle') {
                                    const particle = item.particleRef;
                                    if (particle.life <= 0) continue;

                                    const distSq = point.pos.sub(particle.pos).magSq();
                                    if (distSq < closestDistSq) {
                                        closestDistSq = distSq;
                                        nearestParticleFound = particle;
                                    }
                                }
                            }
                        }
                    }
                }

                if (nearestParticleFound) {
                    point.seesParticle = true;
                    const vecToParticle = nearestParticleFound.pos.sub(point.pos);
                    point.nearestParticleMagnitude = vecToParticle.mag() / EYE_DETECTION_RADIUS; 
                    point.nearestParticleDirection = Math.atan2(vecToParticle.y, vecToParticle.x);
                }
            }
        });
    }

    _processBrain(brainNode, dt, fluidFieldRef, nutrientField, lightField) {
        const nd = brainNode.neuronData;
        if (!nd || !nd.weightsIH || !nd.biasesH || !nd.weightsHO || !nd.biasesO ||
            typeof nd.inputVectorSize !== 'number' ||
            typeof nd.hiddenLayerSize !== 'number' ||
            typeof nd.outputVectorSize !== 'number') {
            // console.warn(`Body ${this.id} brain is missing essential data for processing.`);
            this._applyFallbackBehaviors(dt, fluidFieldRef); // Fallback if brain data is incomplete
            return;
        }

        const inputVector = this._gatherBrainInputs(brainNode, fluidFieldRef, nutrientField, lightField, particles);
        this._propagateBrainOutputs(brainNode, inputVector);
        this._applyBrainActionsToPoints(brainNode, dt);
        this._updateBrainTrainingBuffer(brainNode, inputVector); // Reward logic now inside this method or called from it
        this._triggerBrainPolicyUpdateIfNeeded(brainNode);
    }

    _gatherBrainInputs(brainNode, fluidFieldRef, nutrientField, lightField, particles) {
        const nd = brainNode.neuronData;
        const inputVector = [];

        // Start with base inputs (dye, energy, CoM pos/vel, nutrient)
        if (fluidFieldRef) {
            const brainGx = Math.floor(brainNode.pos.x / fluidFieldRef.scaleX);
            const brainGy = Math.floor(brainNode.pos.y / fluidFieldRef.scaleY);
            const brainIdx = fluidFieldRef.IX(brainGx, brainGy);
            inputVector.push((fluidFieldRef.densityR[brainIdx] || 0) / 255);
            inputVector.push((fluidFieldRef.densityG[brainIdx] || 0) / 255);
            inputVector.push((fluidFieldRef.densityB[brainIdx] || 0) / 255);
        } else {
            inputVector.push(0, 0, 0);
        }
        inputVector.push(this.creatureEnergy / this.currentMaxEnergy); 
        const comPos = this.getAveragePosition();
        const relComPosX = (comPos.x - brainNode.pos.x) / WORLD_WIDTH;
        const relComPosY = (comPos.y - brainNode.pos.y) / WORLD_HEIGHT;
        inputVector.push(Math.tanh(relComPosX));
        inputVector.push(Math.tanh(relComPosY));
        const comVel = this.getAverageVelocity();
        const brainVelX = brainNode.pos.x - brainNode.prevPos.x;
        const brainVelY = brainNode.pos.y - brainNode.prevPos.y;
        const relComVelX = comVel.x - brainVelX;
        const relComVelY = comVel.y - brainVelY;
        inputVector.push(Math.tanh(relComVelX / MAX_PIXELS_PER_FRAME_DISPLACEMENT));
        inputVector.push(Math.tanh(relComVelY / MAX_PIXELS_PER_FRAME_DISPLACEMENT));
        if (nutrientField && fluidFieldRef) {
            const brainGx = Math.floor(brainNode.pos.x / fluidFieldRef.scaleX);
            const brainGy = Math.floor(brainNode.pos.y / fluidFieldRef.scaleY);
            const nutrientIdx = fluidFieldRef.IX(brainGx, brainGy);
            const currentNutrient = nutrientField[nutrientIdx] !== undefined ? nutrientField[nutrientIdx] : 1.0;
            const normalizedNutrient = (currentNutrient - MIN_NUTRIENT_VALUE) / (MAX_NUTRIENT_VALUE - MIN_NUTRIENT_VALUE);
            inputVector.push(Math.max(0, Math.min(1, normalizedNutrient)));
        } else {
            inputVector.push(0.5);
        }

        // Add Eye Inputs (iterate through all points, add if it's an Eye)
        let eyeNodesFoundForInput = 0;
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.EYE) {
                inputVector.push(point.seesParticle ? 1 : 0);
                inputVector.push(point.nearestParticleMagnitude); // Already normalized 0-1 or 0
                inputVector.push((point.nearestParticleDirection / (Math.PI * 2)) + 0.5); // Normalize angle to ~0-1 (0 if no particle)
                eyeNodesFoundForInput++;
            }
        });
        
        // Ensure inputVector is the correct size (final check after all inputs added)
        while(inputVector.length < nd.inputVectorSize) { inputVector.push(0); }
        if(inputVector.length > nd.inputVectorSize) { inputVector.splice(nd.inputVectorSize); }
        
        return inputVector;
    }

    _propagateBrainOutputs(brainNode, inputVector) {
        const nd = brainNode.neuronData;
        const hiddenLayerInputs = multiplyMatrixVector(nd.weightsIH, inputVector);
        const hiddenLayerBiasedInputs = addVectors(hiddenLayerInputs, nd.biasesH);
        const hiddenLayerActivations = hiddenLayerBiasedInputs.map(val => Math.tanh(val));
        const outputLayerInputs = multiplyMatrixVector(nd.weightsHO, hiddenLayerActivations);
        const rawOutputs = addVectors(outputLayerInputs, nd.biasesO);
        nd.rawOutputs = rawOutputs;

        // DEBUG LOG ADDED HERE
        if (nd.rawOutputs.length !== nd.outputVectorSize) {
            console.warn(`Body ${this.id} _propagateBrainOutputs: nd.rawOutputs.length (${nd.rawOutputs.length}) !== nd.outputVectorSize (${nd.outputVectorSize})`);
        }
    }

    _applyBrainActionsToPoints(brainNode, dt) {
        const nd = brainNode.neuronData;

        // Recalculate expected output vector size based on current points
        let currentNumEmitterPoints = 0;
        let currentNumSwimmerPoints = 0;
        let currentNumEaterPoints = 0;
        let currentNumPredatorPoints = 0;
        let currentNumPotentialGrabberPoints = 0;
        this.massPoints.forEach(p => {
            if (p.nodeType === NodeType.EMITTER) currentNumEmitterPoints++;
            else if (p.nodeType === NodeType.SWIMMER) currentNumSwimmerPoints++;
            else if (p.nodeType === NodeType.EATER) currentNumEaterPoints++;
            else if (p.nodeType === NodeType.PREDATOR) currentNumPredatorPoints++;
            if (p.canBeGrabber) currentNumPotentialGrabberPoints++;
        });

        const recalculatedOutputVectorSize = (currentNumEmitterPoints * NEURAL_OUTPUTS_PER_EMITTER) +
                                           (currentNumSwimmerPoints * NEURAL_OUTPUTS_PER_SWIMMER) +
                                           (currentNumEaterPoints * NEURAL_OUTPUTS_PER_EATER) +
                                           (currentNumPredatorPoints * NEURAL_OUTPUTS_PER_PREDATOR) +
                                           (currentNumPotentialGrabberPoints * NEURAL_OUTPUTS_PER_GRABBER_TOGGLE);

        if (nd.outputVectorSize !== recalculatedOutputVectorSize) {
            console.warn(`Body ${this.id} _applyBrainActionsToPoints: MISMATCH between stored nd.outputVectorSize (${nd.outputVectorSize}) and recalculatedOutputVectorSize (${recalculatedOutputVectorSize}) based on current points.`);
            const pointTypes = this.massPoints.map((p, idx) => `Idx ${idx}: Type ${p.nodeType} Grabber: ${p.canBeGrabber}`).join(', ');
            console.warn(`Body ${this.id} Current Points (${this.massPoints.length}): [${pointTypes}]`);
            // Log counts that led to recalculatedOutputVectorSize
            console.warn(`Body ${this.id} Recalculated Counts: Emitters: ${currentNumEmitterPoints}, Swimmers: ${currentNumSwimmerPoints}, Eaters: ${currentNumEaterPoints}, Predators: ${currentNumPredatorPoints}, Grabbers: ${currentNumPotentialGrabberPoints}`);
            // To see what it was at birth, you'd need to check the initializeBrain logs for this body ID.
        }

        // The old log for all-zero controllable points is less useful now, so we can comment it out or remove.
        // if (nd.outputVectorSize > 0 && currentNumEmitterPoints === 0 && currentNumSwimmerPoints === 0 && currentNumEaterPoints === 0 && currentNumPredatorPoints === 0 && currentNumPotentialGrabberPoints === 0) {
        //     const pointTypes = this.massPoints.map((p, idx) => `Idx ${idx}: Type ${p.nodeType} (IsEmitter: ${p.nodeType === NodeType.EMITTER})`).join(', ');
        //     console.warn(`Body ${this.id} _applyBrainActionsToPoints: outputVectorSize is ${nd.outputVectorSize}, but NO controllable points found. Current Points (${this.massPoints.length}): [${pointTypes}]`);
        // }

        let currentRawOutputIndex = 0;
        nd.currentFrameActionDetails = [];

        function sampleAndLogAction(rawMean, rawStdDev) {
            const mean = rawMean;
            const stdDev = Math.exp(rawStdDev) + 1e-6;
            const sampledActionValue = sampleGaussian(mean, stdDev);
            const logProb = logPdfGaussian(sampledActionValue, mean, stdDev);
            return { detail: { mean, stdDev, sampledAction: sampledActionValue, logProb }, value: sampledActionValue };
        }

        // Process Emitters
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.EMITTER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_EMITTER) {
                    const detailsForThisEmitter = [];
                    let localPairIdx = 0;
                    for (let i = 0; i < 3; i++) { // Dye R, G, B
                        const res = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                        detailsForThisEmitter.push(res.detail);
                        point.dyeColor[i] = sigmoid(res.value) * 255;
                    }
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    detailsForThisEmitter.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value); // NN sets exertion for Emitter
                    nd.currentFrameActionDetails.push(...detailsForThisEmitter);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EMITTER;
                } else {
                    // DEBUG LOG ADDED HERE
                    console.warn(`Body ${this.id} _applyBrainActionsToPoints (Emitter): Skipped logging. rawOutputs.length: ${nd.rawOutputs ? nd.rawOutputs.length : 'undefined'}, outputStartRawIdx: ${outputStartRawIdx}, NEURAL_OUTPUTS_PER_EMITTER: ${NEURAL_OUTPUTS_PER_EMITTER}, current nd.outputVectorSize: ${nd.outputVectorSize}`);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EMITTER;
                }
            }
        });

        // Process Swimmers
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.SWIMMER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_SWIMMER) {
                    const detailsForThisSwimmer = [];
                    let localPairIdx = 0;

                    const magnitudeResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    detailsForThisSwimmer.push(magnitudeResult.detail);
                    const rawMagnitude = magnitudeResult.value;

                    const directionResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    detailsForThisSwimmer.push(directionResult.detail);
                    const angle = directionResult.value; 

                    const exertionResultSwimmer = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    detailsForThisSwimmer.push(exertionResultSwimmer.detail);
                    point.currentExertionLevel = sigmoid(exertionResultSwimmer.value); // NN sets exertion for Swimmer

                    const finalMagnitude = sigmoid(rawMagnitude) * MAX_SWIMMER_OUTPUT_MAGNITUDE * this.emitterStrength * point.currentExertionLevel;
                    const appliedForceX = finalMagnitude * Math.cos(angle);
                    const appliedForceY = finalMagnitude * Math.sin(angle);
                    
                    point.applyForce(new Vec2(appliedForceX / dt, appliedForceY / dt)); 
                    nd.currentFrameActionDetails.push(...detailsForThisSwimmer);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_SWIMMER;
                } else {
                    // DEBUG LOG ADDED HERE
                    console.warn(`Body ${this.id} _applyBrainActionsToPoints (Swimmer): Skipped logging. rawOutputs.length: ${nd.rawOutputs ? nd.rawOutputs.length : 'undefined'}, outputStartRawIdx: ${outputStartRawIdx}, NEURAL_OUTPUTS_PER_SWIMMER: ${NEURAL_OUTPUTS_PER_SWIMMER}, current nd.outputVectorSize: ${nd.outputVectorSize}`);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_SWIMMER; 
                }
            }
        });

        // Process Eaters
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.EATER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_EATER) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value); // NN sets exertion for Eater
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EATER;
                } else {
                    // DEBUG LOG ADDED HERE
                    console.warn(`Body ${this.id} _applyBrainActionsToPoints (Eater): Skipped logging. rawOutputs.length: ${nd.rawOutputs ? nd.rawOutputs.length : 'undefined'}, outputStartRawIdx: ${outputStartRawIdx}, NEURAL_OUTPUTS_PER_EATER: ${NEURAL_OUTPUTS_PER_EATER}, current nd.outputVectorSize: ${nd.outputVectorSize}`);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EATER; }
            }
        });

        // Process Predators
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.PREDATOR) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_PREDATOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value); // NN sets exertion for Predator
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_PREDATOR;
                } else {
                    // DEBUG LOG ADDED HERE
                    console.warn(`Body ${this.id} _applyBrainActionsToPoints (Predator): Skipped logging. rawOutputs.length: ${nd.rawOutputs ? nd.rawOutputs.length : 'undefined'}, outputStartRawIdx: ${outputStartRawIdx}, NEURAL_OUTPUTS_PER_PREDATOR: ${NEURAL_OUTPUTS_PER_PREDATOR}, current nd.outputVectorSize: ${nd.outputVectorSize}`);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_PREDATOR; }
            }
        });

        // Process Grabber Toggles for each point
        this.massPoints.forEach(point => {
            if (point.canBeGrabber) { 
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) {
                    const detailsForThisGrab = [];
                    const grabToggleResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    detailsForThisGrab.push(grabToggleResult.detail);
                    point.isGrabbing = sigmoid(grabToggleResult.value) > 0.5; 
                    
                    nd.currentFrameActionDetails.push(...detailsForThisGrab);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_GRABBER_TOGGLE;
                } else {
                    // DEBUG LOG ADDED HERE
                    console.warn(`Body ${this.id} _applyBrainActionsToPoints (Grabber): Skipped logging. rawOutputs.length: ${nd.rawOutputs ? nd.rawOutputs.length : 'undefined'}, outputStartRawIdx: ${outputStartRawIdx}, NEURAL_OUTPUTS_PER_GRABBER_TOGGLE: ${NEURAL_OUTPUTS_PER_GRABBER_TOGGLE}, current nd.outputVectorSize: ${nd.outputVectorSize}`);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_GRABBER_TOGGLE; 
                }
            }
        });

        // DEBUG LOG: Log final state for this function call before buffer push
        if (nd.currentFrameActionDetails.length !== nd.outputVectorSize / 2 && nd.outputVectorSize > 0) { // only log if mismatch and outputsize > 0
            console.warn(`Body ${this.id} _applyBrainActionsToPoints END MISMATCH: currentFrameActionDetails.length (${nd.currentFrameActionDetails.length}) !== nd.outputVectorSize/2 (${nd.outputVectorSize / 2}). Final currentRawOutputIndex: ${currentRawOutputIndex}, Expected final CRI (outputVecSize): ${nd.outputVectorSize}`);
        }
    }

    _updateBrainTrainingBuffer(brainNode, inputVector) { 
        const nd = brainNode.neuronData;
        if (nd.currentFrameActionDetails && nd.currentFrameActionDetails.length > 0) {
            let reward = 0;
            switch (this.rewardStrategy) {
                case RLRewardStrategy.ENERGY_CHANGE:
                    reward = this.creatureEnergy - nd.previousEnergyForReward;
                    break;
                case RLRewardStrategy.REPRODUCTION_EVENT:
                    if (this.justReproduced) {
                        reward = REPRODUCTION_REWARD_VALUE;
                        this.justReproduced = false; // Reset flag after giving reward
                    } else {
                        reward = 0;
                    }
                    break;
                case RLRewardStrategy.PARTICLE_PROXIMITY:
                    let minParticleMagnitude = 1.0; 
                    let particleSeenByAnyEye = false;
                    this.massPoints.forEach(point => {
                        if (point.nodeType === NodeType.EYE && point.seesParticle) {
                            particleSeenByAnyEye = true;
                            if (point.nearestParticleMagnitude < minParticleMagnitude) {
                                minParticleMagnitude = point.nearestParticleMagnitude;
                            }
                        }
                    });
                    if (particleSeenByAnyEye) {
                        reward = (1.0 - minParticleMagnitude) * PARTICLE_PROXIMITY_REWARD_SCALE;
                    } else {
                        reward = 0; 
                    }
                    break;
                default:
                    reward = this.creatureEnergy - nd.previousEnergyForReward; // Fallback
            }

            nd.experienceBuffer.push({
                state: [...inputVector],
                actionDetails: JSON.parse(JSON.stringify(nd.currentFrameActionDetails)),
                reward: reward
            });
            if (nd.experienceBuffer.length > nd.maxExperienceBufferSize) {
                nd.experienceBuffer.shift();
            }
        }
        nd.previousEnergyForReward = this.creatureEnergy; // Always update this after calculating reward for the frame
    }

    _triggerBrainPolicyUpdateIfNeeded(brainNode) {
        const nd = brainNode.neuronData;
        nd.framesSinceLastTrain++;
        if (nd.framesSinceLastTrain >= TRAINING_INTERVAL_FRAMES) {
            this.updateBrainPolicy(); // updateBrainPolicy itself will check buffer size
            // nd.framesSinceLastTrain = 0; // updateBrainPolicy should reset this if it trains
        }
    }

    _applyFallbackBehaviors(dt, fluidFieldRef) {
        // This is called if brainNode is null or has incomplete data.
        // Apply default activation patterns (already done in _updateSensoryInputsAndDefaultActivations)
        // Apply random motor impulses for non-NN controlled movement
        if (this.motorImpulseMagnitudeCap > 0.0001 && (this.ticksSinceBirth % this.motorImpulseInterval === 0)) {
            for (let point of this.massPoints) {
                if (!point.isFixed && point.movementType !== MovementType.FLOATING) { // Swimmers (not floating) can still have motor impulses
                    const randomAngle = Math.random() * Math.PI * 2;
                    const impulseDir = new Vec2(Math.cos(randomAngle), Math.sin(randomAngle));
                    const impulseMag = Math.random() * this.motorImpulseMagnitudeCap;
                    point.applyForce(impulseDir.mul(impulseMag / dt));
                }
            }
        }
        // Note: Emitter dye emission and Swimmer fluid push in fallback mode are currently handled
        // by the fluid interaction logic in _performPhysicalUpdates, based on point.currentExertionLevel
        // which would have been set by _applyDefaultActivationPatterns.
    }

    _updateEnergyBudget(dt, fluidFieldRef, nutrientField, lightField) {
        // --- Apply Red Dye Poison Effect ---
        if (fluidFieldRef && RED_DYE_POISON_STRENGTH > 0) {
            let poisonDamageThisFrame = 0;
            // Cache scale factors for this specific loop if fluidFieldRef is valid
            const localScaleX = fluidFieldRef.scaleX;
            const localScaleY = fluidFieldRef.scaleY;

            for (const point of this.massPoints) {
                const fluidGridX = Math.floor(point.pos.x / localScaleX);
                const fluidGridY = Math.floor(point.pos.y / localScaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);
                const redDensity = (fluidFieldRef.densityR[idx] || 0) / 255;

                if (redDensity > 0.01) {
                    poisonDamageThisFrame += redDensity * RED_DYE_POISON_STRENGTH * (point.radius / 5);
                }
            }
            if (poisonDamageThisFrame > 0) {
                this.creatureEnergy -= poisonDamageThisFrame * dt * 60; // dt is in seconds, scale strength to be per-second
            }
        }
        // --- End of Red Dye Poison Effect ---

        let currentFrameEnergyCost = 0;
        let currentFrameEnergyGain = 0;

        const hasFluidField = fluidFieldRef !== null && typeof fluidFieldRef !== 'undefined';
        const hasNutrientField = nutrientField !== null && typeof nutrientField !== 'undefined';
        const hasLightField = lightField !== null && typeof lightField !== 'undefined';

        // Cache scale factors if fluidFieldRef exists, to avoid repeated access in loop
        const scaleX = hasFluidField ? fluidFieldRef.scaleX : 0;
        const scaleY = hasFluidField ? fluidFieldRef.scaleY : 0;

        for (const point of this.massPoints) {
            let costMultiplier = 1.0;
            let mapIdx = -1; // Initialize map index

            if (hasFluidField) { // Calculate grid coordinates and index once if fluid field is present
                const gx = Math.floor(point.pos.x / scaleX);
                const gy = Math.floor(point.pos.y / scaleY);
                mapIdx = fluidFieldRef.IX(gx, gy);

                if (hasNutrientField) {
                    const baseNutrientValue = nutrientField[mapIdx] !== undefined ? nutrientField[mapIdx] : 1.0;
                const effectiveNutrientValue = baseNutrientValue * globalNutrientMultiplier;
                costMultiplier = 1.0 / Math.max(MIN_NUTRIENT_VALUE, effectiveNutrientValue);
                }
            }


            const baseNodeCostThisFrame = BASE_NODE_EXISTENCE_COST * costMultiplier;
            currentFrameEnergyCost += baseNodeCostThisFrame;
            this.energyCostFromBaseNodes += baseNodeCostThisFrame * dt;

            const exertion = point.currentExertionLevel || 0; 
            const exertionSq = exertion * exertion; // Calculate once if used multiple times

            // NodeType specific costs
            switch (point.nodeType) {
                case NodeType.EMITTER:
                    const emitterCostThisFrame = EMITTER_NODE_ENERGY_COST * exertionSq * costMultiplier;
                currentFrameEnergyCost += emitterCostThisFrame;
                this.energyCostFromEmitterNodes += emitterCostThisFrame * dt;
                    break;
                case NodeType.SWIMMER:
                    const swimmerCostThisFrame = SWIMMER_NODE_ENERGY_COST * exertionSq * costMultiplier;
                currentFrameEnergyCost += swimmerCostThisFrame;
                this.energyCostFromSwimmerNodes += swimmerCostThisFrame * dt;
                    break;
                case NodeType.EATER:
                    const eaterCostThisFrame = EATER_NODE_ENERGY_COST * exertionSq * costMultiplier;
                currentFrameEnergyCost += eaterCostThisFrame;
                this.energyCostFromEaterNodes += eaterCostThisFrame * dt;
                    break;
                case NodeType.PREDATOR:
                    const predatorCostThisFrame = PREDATOR_NODE_ENERGY_COST * exertionSq * costMultiplier;
                currentFrameEnergyCost += predatorCostThisFrame;
                this.energyCostFromPredatorNodes += predatorCostThisFrame * dt;
                    break;
                case NodeType.PHOTOSYNTHETIC:
                const photosyntheticCostThisFrame = PHOTOSYNTHETIC_NODE_ENERGY_COST * costMultiplier;
                currentFrameEnergyCost += photosyntheticCostThisFrame;
                this.energyCostFromPhotosyntheticNodes += photosyntheticCostThisFrame * dt;

                    if (hasLightField && hasFluidField && mapIdx !== -1) { // mapIdx would have been calculated if hasFluidField
                        const baseLightValue = lightField[mapIdx] !== undefined ? lightField[mapIdx] : 0.0;
                    const effectiveLightValue = baseLightValue * globalLightMultiplier; 
                    const energyGainThisPoint = effectiveLightValue * PHOTOSYNTHESIS_EFFICIENCY * (point.radius / 5) * dt;
                        currentFrameEnergyGain += energyGainThisPoint;
                    this.energyGainedFromPhotosynthesis += energyGainThisPoint;
                }
                    break;
                // Note: Neuron, Grabbing, Eye costs are handled by separate 'if' statements below
                // as they can co-exist or have different conditions than the primary functional type.
            }

            // Neuron cost (can be any type of point, but has neuronData)
            if (point.nodeType === NodeType.NEURON) { // This check is okay, as NodeType is exclusive.
                let neuronCostThisFrame = 0;
                if (point.neuronData && point.neuronData.isBrain) {
                    neuronCostThisFrame = NEURON_NODE_ENERGY_COST * 5 * costMultiplier;
                    neuronCostThisFrame += (point.neuronData.hiddenLayerSize || 0) * NEURON_NODE_ENERGY_COST * 0.1 * costMultiplier;
                } else {
                    neuronCostThisFrame = NEURON_NODE_ENERGY_COST * costMultiplier;
                }
                currentFrameEnergyCost += neuronCostThisFrame;
                this.energyCostFromNeuronNodes += neuronCostThisFrame * dt;
            }

            // Grabbing cost (independent of NodeType, depends on isGrabbing state)
            if (point.isGrabbing) { 
                const grabbingCostThisFrame = GRABBING_NODE_ENERGY_COST * costMultiplier;
                currentFrameEnergyCost += grabbingCostThisFrame; 
                this.energyCostFromGrabbingNodes += grabbingCostThisFrame * dt;
            }

            // Eye cost (independent of NodeType, depends on isDesignatedEye state)
            if (point.isDesignatedEye) {
                 const eyeCostThisFrame = EYE_NODE_ENERGY_COST * costMultiplier;
                 currentFrameEnergyCost += eyeCostThisFrame;
                 this.energyCostFromEyeNodes += eyeCostThisFrame * dt;
            }
        }
        
        this.creatureEnergy += currentFrameEnergyGain; // Gains are already dt-scaled
        this.creatureEnergy -= currentFrameEnergyCost * dt; // Costs are per-frame, so scale by dt here
        this.creatureEnergy = Math.min(this.currentMaxEnergy, Math.max(0, this.creatureEnergy));

        this.ticksSinceBirth++;
        if (this.ticksSinceBirth > this.effectiveReproductionCooldown) { // Use effective cooldown
            this.canReproduce = true;
        }

        if (this.creatureEnergy <= 0) {
            this.isUnstable = true;
            // return; // The original function had other logic after this, ensure this return is appropriate if truly unstable
        }

        if (this.motorImpulseMagnitudeCap > 0.0001 && (this.ticksSinceBirth % this.motorImpulseInterval === 0)) {
            for (let point of this.massPoints) {
                if (!point.isFixed && point.movementType !== MovementType.FLOATING) {
                    const randomAngle = Math.random() * Math.PI * 2;
                    const impulseDir = new Vec2(Math.cos(randomAngle), Math.sin(randomAngle));
                    const impulseMag = Math.random() * this.motorImpulseMagnitudeCap;
                    const impulseForce = impulseDir.mul(impulseMag / dt);
                    point.applyForce(impulseForce);
                }
            }
        }


        if (fluidFieldRef) {
            for (let point of this.massPoints) {
                if (point.isFixed) continue; // Fixed points don't get pushed by fluid

                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);

                // Fluid Pushes Point (if Floating)
                if (point.movementType === MovementType.FLOATING) {
                    // Swimmers cannot be Floating, so no need to check point.nodeType === NodeType.SWIMMER here
                    const rawFluidVx = fluidFieldRef.Vx[idx];
                    const rawFluidVy = fluidFieldRef.Vy[idx];
                    let fluidDisplacementPx = new Vec2(rawFluidVx * fluidFieldRef.scaleX * dt, rawFluidVy * fluidFieldRef.scaleY * dt);
                    let effectiveFluidDisplacementPx = fluidDisplacementPx.mul(this.fluidCurrentStrength);
                    let currentPointDisplacementPx = point.pos.sub(point.prevPos);
                    let blendedDisplacementPx = currentPointDisplacementPx.mul(1.0 - this.fluidEntrainment)
                                                     .add(effectiveFluidDisplacementPx.mul(this.fluidEntrainment));
                    point.prevPos = point.pos.clone().sub(blendedDisplacementPx);
                }
                
                // Point Affects Fluid (if Swimmer or Emitter)
                if (point.nodeType === NodeType.SWIMMER && this.emitterStrength > 0.0001) { // Using this.emitterStrength as a general "action strength"
                    let currentActionStrength = this.emitterStrength; // TODO: Later, Swimmers might have their own strength attribute
                    // if (point.neuralEffectiveStrength !== null && point.neuralEffectiveStrength !== undefined) { // For NN control
                    //     currentActionStrength *= point.neuralEffectiveStrength;
                    // }
                    const swimForceX = this.emitterDirection.x * currentActionStrength; // TODO: Later, Swimmers will have their own direction or NN output for this
                    const swimForceY = this.emitterDirection.y * currentActionStrength;
                    fluidFieldRef.addVelocity(fluidGridX, fluidGridY, swimForceX, swimForceY);
                }

                if (point.nodeType === NodeType.EMITTER) { // Dye Emission
                    // Emission strength for dye could be from point.currentExertionLevel if NN controlled, or a fixed value
                    let dyeEmissionStrength = 50; // Default fixed strength
                    if (point.neuronData && point.neuronData.isBrain && point.currentExertionLevel > 0) {
                        // Example: Link dye emission strength to brain-controlled exertion for emitters
                        // This part would be refined when NN outputs for Emitters are fully defined
                        // dyeEmissionStrength *= point.currentExertionLevel;
                    }
                    fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], dyeEmissionStrength);
                }
            }
        }

        for (let spring of this.springs) spring.applyForce();

        // Inter-body repulsion & Predation (combined loop) - USING SPATIAL GRID
        for (let i_p1 = 0; i_p1 < this.massPoints.length; i_p1++) {
            const p1 = this.massPoints[i_p1];
            if (p1.isFixed) continue;

            const p1Gx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(p1.pos.x / GRID_CELL_SIZE)));
            const p1Gy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(p1.pos.y / GRID_CELL_SIZE)));

            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const checkGx = p1Gx + dx;
                    const checkGy = p1Gy + dy;

                    if (checkGx >= 0 && checkGx < GRID_COLS && checkGy >= 0 && checkGy < GRID_ROWS) {
                        const cellIndex = checkGx + checkGy * GRID_COLS;
                        if (Array.isArray(spatialGrid[cellIndex])) {
                            const cellBucket = spatialGrid[cellIndex];
                            for (const otherItem of cellBucket) {
                                if (otherItem.type === 'softbody_point') {
                                    if (otherItem.bodyRef === this) continue;

                                    const p2 = otherItem.pointRef;
                                    if (p2.isFixed) continue;

                                    const diff = p1.pos.sub(p2.pos);
                                    const distSq = diff.magSq();
                                    const interactionRadius = (p1.radius + p2.radius) * BODY_REPULSION_RADIUS_FACTOR;

                                    if (distSq < interactionRadius * interactionRadius && distSq > 0.0001) {
                                        const dist = Math.sqrt(distSq);
                                        const overlap = interactionRadius - dist;
                                        const forceDir = diff.normalize();
                                        const repulsionForceMag = BODY_REPULSION_STRENGTH * overlap * 0.5;
                                        const repulsionForce = forceDir.mul(repulsionForceMag);
                                        p1.applyForce(repulsionForce);
                                    }

                                    // CORRECTED PREDATION LOGIC
                                    if (p1.nodeType === NodeType.PREDATOR) {
                                        const p1Exertion = p1.currentExertionLevel || 0;
                                        const effectivePredationRadiusMultiplier = PREDATION_RADIUS_MULTIPLIER_BASE + (PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * p1Exertion);
                                        const predationRadius = p1.radius * effectivePredationRadiusMultiplier;
                                        
                                        if (distSq < predationRadius * predationRadius) {
                                            const effectiveEnergySapped = ENERGY_SAPPED_PER_PREDATION_BASE + (ENERGY_SAPPED_PER_PREDATION_MAX_BONUS * p1Exertion);
                                            const energyToSap = Math.min(otherItem.bodyRef.creatureEnergy, effectiveEnergySapped);
                                            if (energyToSap > 0) {
                                                otherItem.bodyRef.creatureEnergy -= energyToSap;
                                                this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyToSap); // Use currentMaxEnergy
                                                this.energyGainedFromPredation += energyToSap;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        for (let point of this.massPoints) point.update(dt);

        // Instability Checks
        const MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ = (MAX_PIXELS_PER_FRAME_DISPLACEMENT)**2;
        const localMaxSpringStretchFactor = MAX_SPRING_STRETCH_FACTOR;
        const localMaxSpanPerPointFactor = MAX_SPAN_PER_POINT_FACTOR;

        for (let point of this.massPoints) {
            const displacementSq = (point.pos.x - point.prevPos.x)**2 + (point.pos.y - point.prevPos.y)**2;
            let instabilityReason = null;

            if (displacementSq > MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ) {
                instabilityReason = "excessive displacement";
            } else if (isNaN(point.pos.x) || isNaN(point.pos.y)) {
                instabilityReason = "NaN position";
            } else if (!isFinite(point.pos.x) || !isFinite(point.pos.y)) {
                instabilityReason = "Infinite position";
            }

            if (instabilityReason) {
                this.isUnstable = true;
                console.warn(`Soft body ID ${this.id}, Point Index ${this.massPoints.indexOf(point)}: Instability due to ${instabilityReason}. Point:`, JSON.parse(JSON.stringify(point.pos)));
                break;
            }
        }
        if (this.isUnstable) return;

        for (const spring of this.springs) {
            const currentLength = spring.p1.pos.sub(spring.p2.pos).mag();
            if (currentLength > spring.restLength * localMaxSpringStretchFactor) {
                this.isUnstable = true;
                console.warn(`Soft body ID ${this.id} spring instability (over-stretched)! Spring rest: ${spring.restLength.toFixed(1)}, current: ${currentLength.toFixed(1)}`);
                break;
            }
        }
        if (this.isUnstable) return;

        if (this.massPoints.length > 2) { 
            const bbox = this.getBoundingBox();
            if (bbox.width > this.massPoints.length * localMaxSpanPerPointFactor ||
                bbox.height > this.massPoints.length * localMaxSpanPerPointFactor) {
                this.isUnstable = true;
                console.warn(`Soft body ID ${this.id} span instability! Width: ${bbox.width.toFixed(1)}, Height: ${bbox.height.toFixed(1)}, Points: ${this.massPoints.length}`);
            }
        }
        if (this.isUnstable) return;


        if (!this.isUnstable) { 
            for (let point of this.massPoints) {
                if (point.isFixed) continue; 

                // CORRECTED EATING LOGIC
                if (point.nodeType === NodeType.EATER) { 
                    const pointExertion = point.currentExertionLevel || 0;
                    const effectiveEatingRadiusMultiplier = EATING_RADIUS_MULTIPLIER_BASE + (EATING_RADIUS_MULTIPLIER_MAX_BONUS * pointExertion);
                    const eatingRadius = point.radius * effectiveEatingRadiusMultiplier;
                    const eatingRadiusSq = eatingRadius * eatingRadius;

                    const eaterGx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(point.pos.x / GRID_CELL_SIZE)));
                    const eaterGy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(point.pos.y / GRID_CELL_SIZE)));

                    for (let dy = -1; dy <= 1; dy++) { 
                        for (let dx = -1; dx <= 1; dx++) {
                            const checkGx = eaterGx + dx;
                            const checkGy = eaterGy + dy;
                            if (checkGx >= 0 && checkGx < GRID_COLS && checkGy >= 0 && checkGy < GRID_ROWS) {
                                const cellIndex = checkGx + checkGy * GRID_COLS;
                                if (Array.isArray(spatialGrid[cellIndex])) {
                                    const cellBucket = spatialGrid[cellIndex];
                                    for (let k = cellBucket.length - 1; k >= 0; k--) { 
                                        const item = cellBucket[k];
                                        if (item.type === 'particle') {
                                            const particle = item.particleRef;
                                            if (particle.life > 0 && !particle.isEaten) {
                                                const distSq = point.pos.sub(particle.pos).magSq();
                                                if (distSq < eatingRadiusSq) {
                                                    particle.isEaten = true;
                                                    particle.life = 0; 
                                                    
                                                    particle.life = 0; // Mark for removal from main particles array
                                                    
                                                    let energyGain = ENERGY_PER_PARTICLE;
                                                    if (nutrientField && fluidField) { // fluidFieldRef is fluidField in this context
                                                        const particleGx = Math.floor(particle.pos.x / fluidField.scaleX);
                                                        const particleGy = Math.floor(particle.pos.y / fluidField.scaleY);
                                                        const nutrientIdxAtParticle = fluidField.IX(particleGx, particleGy);
                                                        const baseNutrientValueAtParticle = nutrientField[nutrientIdxAtParticle] !== undefined ? nutrientField[nutrientIdxAtParticle] : 1.0;
                                                        const effectiveNutrientAtParticle = baseNutrientValueAtParticle * globalNutrientMultiplier;
                                                        energyGain *= Math.max(MIN_NUTRIENT_VALUE, effectiveNutrientAtParticle);
                                                    }
                                                    this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyGain); // Use currentMaxEnergy
                                                    this.energyGainedFromEating += energyGain;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        if (!this.isUnstable) {
            for (let point of this.massPoints) {
                if (point.isFixed || point.nodeType === NodeType.FIXED_ROOT) continue;
                const implicitVelX = point.pos.x - point.prevPos.x;
                const implicitVelY = point.pos.y - point.prevPos.y;

                if (IS_WORLD_WRAPPING) {
                    if (point.pos.x < 0) { point.pos.x += WORLD_WIDTH; point.prevPos.x += WORLD_WIDTH; }
                    else if (point.pos.x > WORLD_WIDTH) { point.pos.x -= WORLD_WIDTH; point.prevPos.x -= WORLD_WIDTH; }
                    if (point.pos.y < 0) { point.pos.y += WORLD_HEIGHT; point.prevPos.y += WORLD_HEIGHT; }
                    else if (point.pos.y > WORLD_HEIGHT) { point.pos.y -= WORLD_HEIGHT; point.prevPos.y -= WORLD_HEIGHT; }
                } else {
                    if (point.pos.x - point.radius < 0) {
                        point.pos.x = point.radius;
                        point.prevPos.x = point.pos.x - implicitVelX * restitution;
                    } else if (point.pos.x + point.radius > WORLD_WIDTH) {
                        point.pos.x = WORLD_WIDTH - point.radius;
                        point.prevPos.x = point.pos.x - implicitVelX * restitution;
                    }
                    if (point.pos.y - point.radius < 0) {
                        point.pos.y = point.radius;
                        point.prevPos.y = point.pos.y - implicitVelY * restitution;
                    } else if (point.pos.y + point.radius > WORLD_HEIGHT) {
                        point.pos.y = WORLD_HEIGHT - point.radius;
                        point.prevPos.y = point.pos.y - implicitVelY * restitution;
                    }
                }
            }
        }
    }

    translateBody(dx, dy) {
        this.massPoints.forEach(p => {
            p.pos.x += dx;
            p.pos.y += dy;
            p.prevPos.x += dx;
            p.prevPos.y += dy;
        });
    }
    getMinX() { return Math.min(...this.massPoints.map(p => p.pos.x - p.radius)); }
    getMaxX() { return Math.max(...this.massPoints.map(p => p.pos.x + p.radius)); }
    getMinY() { return Math.min(...this.massPoints.map(p => p.pos.y - p.radius)); }
    getMaxY() { return Math.max(...this.massPoints.map(p => p.pos.y + p.radius)); }


    reproduce() {
        if (this.isUnstable || !this.canReproduce || !canCreaturesReproduceGlobally) return []; // Check global flag

        const energyForOneOffspring = this.currentMaxEnergy * OFFSPRING_INITIAL_ENERGY_SHARE; // Use currentMaxEnergy for cost basis
        let successfullyPlacedOffspring = 0;
        let offspring = [];

        // Cache bounding boxes of existing population
        const existingBodyBBoxes = softBodyPopulation.map(body => {
            if (body.isUnstable) return null; // or some indicator for unstable bodies
            return body.getBoundingBox();
        }).filter(bbox => bbox !== null);


        for (let i = 0; i < this.numOffspring; i++) {
            if (this.creatureEnergy < energyForOneOffspring) break; // Not enough energy for this one

            let placedThisOffspring = false;
            for (let attempt = 0; attempt < OFFSPRING_PLACEMENT_ATTEMPTS; attempt++) {
                const angle = Math.random() * Math.PI * 2;
                const radiusOffset = this.offspringSpawnRadius * (0.5 + Math.random() * 0.5);
                const offsetX = Math.cos(angle) * radiusOffset;
                const offsetY = Math.sin(angle) * radiusOffset;

                const parentAvgPos = this.getAveragePosition();
                let spawnX = parentAvgPos.x + offsetX;
                let spawnY = parentAvgPos.y + offsetY;

                // Tentatively create child to get its bounding box
                const tempChild = new SoftBody( -1, spawnX, spawnY, this); // -1 ID for temp
                if (tempChild.massPoints.length === 0) continue; // Should not happen with new logic
                const childBBox = tempChild.getBoundingBox();

                // Adjust spawnX, spawnY to be the center of the child's tentative bbox
                spawnX = spawnX - (childBBox.minX - spawnX) + childBBox.width / 2;
                spawnY = spawnY - (childBBox.minY - spawnY) + childBBox.height / 2;

                const childWorldMinX = spawnX - childBBox.width / 2;
                const childWorldMaxX = spawnX + childBBox.width / 2;
                const childWorldMinY = spawnY - childBBox.height / 2;
                const childWorldMaxY = spawnY + childBBox.height / 2;


                let isSpotClear = true;
                // Check against existing population
                for (const otherBBox of existingBodyBBoxes) {
                    // No need to check otherBody.isUnstable here as it's filtered during caching
                    if (!(childWorldMaxX < otherBBox.minX || childWorldMinX > otherBBox.maxX || childWorldMaxY < otherBBox.minY || childWorldMinY > otherBBox.maxY)) {
                        isSpotClear = false; break;
                    }
                }
                // Check against already spawned new offspring in this cycle
                if (isSpotClear) {
                    for (const newBorn of offspring) {
                        const newBornBBox = newBorn.getBoundingBox();
                         if (!(childWorldMaxX < newBornBBox.minX || childWorldMinX > newBornBBox.maxX || childWorldMaxY < newBornBBox.minY || childWorldMinY > newBornBBox.maxY)) {
                            isSpotClear = false; break;
                        }
                    }
                }

                if (isSpotClear) {
                    this.creatureEnergy -= energyForOneOffspring;
                    // Use the already constructed tempChild's points, but create a new SoftBody instance with proper ID and translated points.
                    const finalChild = new SoftBody(nextSoftBodyId++, spawnX, spawnY, this);

                    finalChild.creatureEnergy = energyForOneOffspring; // Set energy for the actual child
                    offspring.push(finalChild);
                    successfullyPlacedOffspring++;
                    placedThisOffspring = true;
                    break; // Break from placement attempts for this offspring
                }
            }
            if (!placedThisOffspring && this.creatureEnergy < energyForOneOffspring) {
                break; // Not enough energy for further attempts for other offspring either
            }
        }

        if (successfullyPlacedOffspring > 0) {
            this.creatureEnergy *= (1 - REPRODUCTION_ADDITIONAL_COST_FACTOR);
            if(this.creatureEnergy < 0) this.creatureEnergy = 0;
            this.ticksSinceBirth = 0;
            this.canReproduce = false;
            this.justReproduced = true; // Set the flag here
        }
        return offspring;
    }

    getBoundingBox() {
        if (this.massPoints.length === 0) {
            return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 };
        }
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.massPoints.forEach(p => {
            minX = Math.min(minX, p.pos.x - p.radius);
            minY = Math.min(minY, p.pos.y - p.radius);
            maxX = Math.max(maxX, p.pos.x + p.radius);
            maxY = Math.max(maxY, p.pos.y + p.radius);
        });
        return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
    }

    getAveragePosition() {
        if (this.massPoints.length === 0) return new Vec2(WORLD_WIDTH/2, WORLD_HEIGHT/2);
        let sumX = 0, sumY = 0;
        this.massPoints.forEach(p => { sumX += p.pos.x; sumY += p.pos.y; });
        return new Vec2(sumX / this.massPoints.length, sumY / this.massPoints.length);
    }

    getAverageVelocity() {
        if (this.massPoints.length === 0) return new Vec2(0,0);
        let sumVelX = 0, sumVelY = 0;
        this.massPoints.forEach(p => {
            const velX = p.pos.x - p.prevPos.x;
            const velY = p.pos.y - p.prevPos.y;
            sumVelX += velX;
            sumVelY += velY;
        });
        return new Vec2(sumVelX / this.massPoints.length, sumVelY / this.massPoints.length);
    }

    initializeBrain() {
        let brainNode = null;
        // First, try to find an already designated brain (e.g., from a parent or loaded config)
        for (const point of this.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        // If no brain was pre-designated, find the first NEURON type node and make it the brain.
        if (!brainNode) {
            for (const point of this.massPoints) {
                if (point.nodeType === NodeType.NEURON) {
                    if (!point.neuronData) { // Ensure neuronData exists
                        point.neuronData = { 
                            hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                        };
                    }
                    point.neuronData.isBrain = true;
                    brainNode = point; // Assign it as the brain for subsequent logic
                    // Ensure other neurons are not brains (important if multiple neurons exist)
                    this.massPoints.forEach(otherP => {
                        if (otherP !== point && otherP.nodeType === NodeType.NEURON && otherP.neuronData) {
                            otherP.neuronData.isBrain = false;
                        }
                    });
                    break; // Found and assigned a brain
                }
            }
        }

        if (brainNode && brainNode.neuronData && brainNode.neuronData.isBrain) {
            const nd = brainNode.neuronData;

            let numEmitterPoints = 0;
            let numSwimmerPoints = 0;
            let numEaterPoints = 0;
            let numPredatorPoints = 0;
            let numEyeNodes = 0;
            let numPotentialGrabberPoints = 0;

            this.massPoints.forEach(p => {
                if (p.nodeType === NodeType.EMITTER) numEmitterPoints++;
                else if (p.nodeType === NodeType.SWIMMER) numSwimmerPoints++;
                else if (p.nodeType === NodeType.EATER) numEaterPoints++;
                else if (p.nodeType === NodeType.PREDATOR) numPredatorPoints++;
                else if (p.nodeType === NodeType.EYE) numEyeNodes++;

                if (p.canBeGrabber) numPotentialGrabberPoints++; // Count only points that can be grabbers
            });

            // DEBUG LOG ADDED HERE (in initializeBrain)
            // console.log(`Body ${this.id} initializeBrain Counts: Emitters: ${numEmitterPoints}, Swimmers: ${numSwimmerPoints}, Eaters: ${numEaterPoints}, Predators: ${numPredatorPoints}, Grabbers: ${numPotentialGrabberPoints}, Eyes: ${numEyeNodes}`);
            // Corrected log name for clarity
            console.log(`Body ${this.id} initializeBrain Counts from .massPoints loop: E:${numEmitterPoints}, S:${numSwimmerPoints}, Ea:${numEaterPoints}, P:${numPredatorPoints}, G:${numPotentialGrabberPoints}, Ey:${numEyeNodes}`);

            nd.inputVectorSize = NEURAL_INPUT_SIZE + (numEyeNodes * NEURAL_INPUTS_PER_EYE);
            
            // Log individual counts IMMEDIATELY BEFORE SUM
            console.log(`Body ${this.id} initializeBrain Counts directly before sum: E:${numEmitterPoints}, S:${numSwimmerPoints}, Ea:${numEaterPoints}, P:${numPredatorPoints}, G:${numPotentialGrabberPoints}`);

            nd.outputVectorSize = (numEmitterPoints * NEURAL_OUTPUTS_PER_EMITTER) +
                                  (numSwimmerPoints * NEURAL_OUTPUTS_PER_SWIMMER) +
                                  (numEaterPoints * NEURAL_OUTPUTS_PER_EATER) +
                                  (numPredatorPoints * NEURAL_OUTPUTS_PER_PREDATOR) +
                                  (numPotentialGrabberPoints * NEURAL_OUTPUTS_PER_GRABBER_TOGGLE);

            console.log(`Body ${this.id} initializeBrain: Calculated nd.outputVectorSize = ${nd.outputVectorSize} (using E:${numEmitterPoints},S:${numSwimmerPoints},Ea:${numEaterPoints},P:${numPredatorPoints},G:${numPotentialGrabberPoints})`);

            // Ensure hiddenLayerSize is valid or initialize it if it's from an older creature version
            if (typeof nd.hiddenLayerSize !== 'number' || nd.hiddenLayerSize < DEFAULT_HIDDEN_LAYER_SIZE_MIN || nd.hiddenLayerSize > DEFAULT_HIDDEN_LAYER_SIZE_MAX) {
                nd.hiddenLayerSize = DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1));
            }

            // Initialize weights and biases only once for this creature instance
            // (or if hiddenLayerSize was just reset)
            if (!nd.weightsIH || nd.weightsIH.length !== nd.hiddenLayerSize || (nd.weightsIH.length > 0 && nd.weightsIH[0].length !== nd.inputVectorSize) ) {
                nd.weightsIH = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize);
                nd.biasesH = initializeVector(nd.hiddenLayerSize);
                console.log(`Body ${this.id} brain: Initialized weightsIH/biasesH. Inputs: ${nd.inputVectorSize}, Hidden: ${nd.hiddenLayerSize}`);
            }
            
            if (!nd.weightsHO || nd.weightsHO.length !== nd.outputVectorSize || (nd.weightsHO.length > 0 && nd.weightsHO[0].length !== nd.hiddenLayerSize) ) {
                nd.weightsHO = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize);
                nd.biasesO = initializeVector(nd.outputVectorSize);
                console.log(`Body ${this.id} brain: Initialized weightsHO/biasesO. Outputs: ${nd.outputVectorSize}, Hidden: ${nd.hiddenLayerSize}`);
            }
            
            // nd.lastInitializedNumEyeNodes = numEyeNodes; // No longer needed to track changes within initializeBrain

            // Initialize RL components if they don't exist
            if (!nd.experienceBuffer) nd.experienceBuffer = [];
            if (typeof nd.framesSinceLastTrain !== 'number') nd.framesSinceLastTrain = 0; 
            if (typeof nd.previousEnergyForReward !== 'number') nd.previousEnergyForReward = this.creatureEnergy;
            if (typeof nd.lastAvgNormalizedReward !== 'number') nd.lastAvgNormalizedReward = 0;
            if (typeof nd.maxExperienceBufferSize !== 'number') nd.maxExperienceBufferSize = 10;

        } else {
            // No brain node found or brainNode.neuronData is missing somehow
        }
    }

    calculateDiscountedRewards(rewards, gamma) {
        const discountedRewards = new Array(rewards.length);
        let runningAdd = 0;
        for (let i = rewards.length - 1; i >= 0; i--) {
            runningAdd = rewards[i] + gamma * runningAdd;
            discountedRewards[i] = runningAdd;
        }
        return discountedRewards;
    }

    updateBrainPolicy() {
        let brainNode = null;
        for (const point of this.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        if (!brainNode || !brainNode.neuronData) return;
        const nd = brainNode.neuronData;

        if (!nd.experienceBuffer || nd.experienceBuffer.length < nd.maxExperienceBufferSize) {
            return; // Not enough experiences to train yet
        }

        const states = nd.experienceBuffer.map(exp => exp.state);
        const actionDetailsBatch = nd.experienceBuffer.map(exp => exp.actionDetails); // Array of arrays of actionDetail objects
        const rewards = nd.experienceBuffer.map(exp => exp.reward);

        const discountedRewards = this.calculateDiscountedRewards(rewards, DISCOUNT_FACTOR_GAMMA);

        // Normalize discounted rewards (optional but good practice)
        let meanDiscountedReward = 0;
        for (const r of discountedRewards) meanDiscountedReward += r;
        meanDiscountedReward /= discountedRewards.length;

        let stdDevDiscountedReward = 0;
        for (const r of discountedRewards) stdDevDiscountedReward += (r - meanDiscountedReward) ** 2;
        stdDevDiscountedReward = Math.sqrt(stdDevDiscountedReward / discountedRewards.length);
        
        const normalizedDiscountedRewards = discountedRewards.map(
            r => (r - meanDiscountedReward) / (stdDevDiscountedReward + 1e-6) // Add epsilon for stability
        );

        // Store the average normalized reward for diagnostics
        nd.lastAvgNormalizedReward = meanDiscountedReward; // Using mean before normalization for a more direct sense of reward scale
        // If you prefer the normalized mean: nd.lastAvgNormalizedReward = normalizedDiscountedRewards.reduce((a, b) => a + b, 0) / normalizedDiscountedRewards.length;

        // Initialize gradients for weights and biases (accumulate over the batch)
        const gradWeightsHO_acc = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize, 0); // Initialize with zeros
        const gradBiasesO_acc = initializeVector(nd.outputVectorSize, 0);
        const gradWeightsIH_acc = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize, 0);
        const gradBiasesH_acc = initializeVector(nd.hiddenLayerSize, 0);
        

        // Iterate through each trajectory/experience in the batch
        for (let t = 0; t < nd.experienceBuffer.length; t++) {
            const state_t = states[t];
            const actionDetails_t = actionDetailsBatch[t]; // This is an array of actionDetail objects for this timestep
            const G_t_normalized = normalizedDiscountedRewards[t];

            // Re-run forward pass to get hidden activations for this state_t
            // (assuming weights haven't changed *during* this batch processing)
            const hiddenLayerInputs_t = multiplyMatrixVector(nd.weightsIH, state_t);
            const hiddenLayerBiasedInputs_t = addVectors(hiddenLayerInputs_t, nd.biasesH);
            const hiddenActivations_t = hiddenLayerBiasedInputs_t.map(val => Math.tanh(val));
            
            // Gradients for Output Layer (weightsHO, biasesO)
            let currentActionDetailIdx = 0;
            for (let i = 0; i < nd.outputVectorSize / 2; i++) { // Iterate through 6 logical actions
                const meanOutputIdx = i * 2;
                const stdDevOutputIdx = i * 2 + 1;

                const ad = actionDetails_t[currentActionDetailIdx]; // Access detail for the current logical action
                
                if (!ad) { 
                    // This implies that actionDetails_t (from experience buffer) is shorter 
                    // than what outputVectorSize/2 expects. This could happen if not all 
                    // controllable points could take actions in that specific past frame 
                    // (e.g., rawOutputs array was too short for them). 
                    // We can't calculate a gradient for a missing action detail.
                    // console.warn(`Body ${this.id} updateBrainPolicy: Missing action detail at index ${currentActionDetailIdx} for timestep ${t}. Skipping this action component's gradient.`);
                    currentActionDetailIdx++; // Still need to advance to the next logical action slot
                    continue; 
                }
                currentActionDetailIdx++; // Increment after successful access and use.

                const { sampledAction, mean, stdDev } = ad;
                const grad_logProb_d_mean = (sampledAction - mean) / (stdDev * stdDev + 1e-9); 
                const grad_logProb_d_stdDev_output = (((sampledAction - mean) ** 2) - (stdDev * stdDev)) / (stdDev * stdDev * stdDev + 1e-9);
                
                // Error for the mean output neuron
                const error_mean = G_t_normalized * grad_logProb_d_mean;
                // Error for the stdDev output neuron (raw output from NN, before exp())
                // actualStdDev = exp(rawStdDev) + eps --> d(actualStdDev)/d(rawStdDev) approx actualStdDev
                const error_stdDev = G_t_normalized * grad_logProb_d_stdDev_output * (stdDev - 1e-6); // (stdDev-eps) is approx exp(rawStdDev)

                for (let k = 0; k < nd.hiddenLayerSize; k++) {
                    gradWeightsHO_acc[meanOutputIdx][k] += error_mean * hiddenActivations_t[k];
                    gradWeightsHO_acc[stdDevOutputIdx][k] += error_stdDev * hiddenActivations_t[k];
                }
                gradBiasesO_acc[meanOutputIdx] += error_mean;
                gradBiasesO_acc[stdDevOutputIdx] += error_stdDev;
            }

            // Gradients for Hidden Layer (weightsIH, biasesH)
            const d_tanh = (x) => 1 - x * x; // Derivative of tanh(y) is 1 - tanh(y)^2 where x = tanh(y)
            for (let h = 0; h < nd.hiddenLayerSize; h++) {
                let sum_error_weighted_HO = 0;
                for (let j = 0; j < nd.outputVectorSize; j++) {
                    // Determine if j is a mean or stdDev output to get the correct error term
                    const actionDetailIndex = Math.floor(j/2);
                    if (actionDetailIndex >= actionDetails_t.length) {
                        // This experience was recorded with a different number of actions
                        // than the current network configuration. Skip this gradient component.
                        console.warn(`Skipping gradient for output ${j} in experience ${t}: actionDetailIndex ${actionDetailIndex} out of bounds for actionDetails_t length ${actionDetails_t.length}`);
                        continue; 
                    }

                    const currentActionDetail = actionDetails_t[actionDetailIndex];
                    if (!currentActionDetail) { // Should ideally not happen if the length check passes
                        console.warn(`Skipping gradient for output ${j} in experience ${t}: currentActionDetail is undefined at index ${actionDetailIndex}`);
                        continue;
                    }

                    const error_from_output_j = (j % 2 === 0) ? 
                        G_t_normalized * (currentActionDetail.sampledAction - currentActionDetail.mean) / (currentActionDetail.stdDev**2 + 1e-9) :
                        G_t_normalized * (((currentActionDetail.sampledAction - currentActionDetail.mean)**2 - currentActionDetail.stdDev**2) / (currentActionDetail.stdDev**3 + 1e-9)) * (currentActionDetail.stdDev - 1e-6);
                    sum_error_weighted_HO += error_from_output_j * nd.weightsHO[j][h];
                }
                const error_h = sum_error_weighted_HO * d_tanh(hiddenActivations_t[h]);

                for (let i_input = 0; i_input < nd.inputVectorSize; i_input++) {
                    gradWeightsIH_acc[h][i_input] += error_h * state_t[i_input];
                }
                gradBiasesH_acc[h] += error_h;
            }
        }

        // Apply accumulated gradients (averaged over batch size implicitly by not dividing here, or explicitly divide)
        const batchSize = nd.experienceBuffer.length;
        for (let j = 0; j < nd.outputVectorSize; j++) {
            for (let k = 0; k < nd.hiddenLayerSize; k++) {
                nd.weightsHO[j][k] += LEARNING_RATE * gradWeightsHO_acc[j][k] / batchSize;
            }
            nd.biasesO[j] += LEARNING_RATE * gradBiasesO_acc[j] / batchSize;
        }
        for (let h = 0; h < nd.hiddenLayerSize; h++) {
            for (let i = 0; i < nd.inputVectorSize; i++) {
                nd.weightsIH[h][i] += LEARNING_RATE * gradWeightsIH_acc[h][i] / batchSize;
            }
            nd.biasesH[h] += LEARNING_RATE * gradBiasesH_acc[h] / batchSize;
        }

        nd.experienceBuffer = []; // Clear buffer after training
        nd.framesSinceLastTrain = 0;
        // console.log(`Body ${this.id} brain updated. Normalized G_t example: ${normalizedDiscountedRewards[0]}`);
    }

    drawSelf(ctx) {
        if (this.isUnstable) return;
        for (let spring of this.springs) spring.draw(ctx);
        for (let point of this.massPoints) point.draw(ctx);
    }

    // Helper function to find linear segments of points
    findLinearSegments(minLength, maxLength) {
        const segments = [];
        if (this.massPoints.length < minLength) return segments;

        const adjacency = new Map(); // point_index -> [neighbor_indices]
        this.massPoints.forEach((_, i) => adjacency.set(i, []));

        this.springs.forEach(spring => {
            const p1Index = this.massPoints.indexOf(spring.p1);
            const p2Index = this.massPoints.indexOf(spring.p2);
            if (p1Index !== -1 && p2Index !== -1) {
                adjacency.get(p1Index).push(p2Index);
                adjacency.get(p2Index).push(p1Index);
            }
        });

        for (let i = 0; i < this.massPoints.length; i++) {
            // Try to start a segment from each point
            // Explore paths using DFS/BFS to find linear chains
            const stack = [[i, [i]]]; // [current_point_index, path_so_far_indices]
            const visitedInPath = new Set(); // To avoid cycles in the current path being built

            while (stack.length > 0) {
                const [currentIndex, currentPathIndices] = stack.pop();
                visitedInPath.add(currentIndex);

                if (currentPathIndices.length >= minLength) {
                    segments.push(currentPathIndices.map(idx => this.massPoints[idx]));
                }

                if (currentPathIndices.length < maxLength) {
                    const neighbors = adjacency.get(currentIndex) || [];
                    for (const neighborIndex of neighbors) {
                        if (!currentPathIndices.includes(neighborIndex)) { // Avoid immediate backtracking and simple cycles
                             // Check if adding this neighbor would still constitute a "linear-enough" extension.
                             // For this version, we are more permissive: any sequence of connected points.
                            stack.push([neighborIndex, [...currentPathIndices, neighborIndex]]);
                        }
                    }
                }
            }
        }
        
        // Post-process to remove duplicate or fully contained sub-segments if desired, but for now, allow overlaps.
        // Ensure segments are unique point sequences if multiple paths lead to same point sequence.
        const uniqueSegments = [];
        const segmentSignatures = new Set();
        for (const seg of segments) {
            const signature = seg.map(p => this.massPoints.indexOf(p)).sort((a,b)=>a-b).join('-'); // Create a unique signature
            if(!segmentSignatures.has(signature)){
                segmentSignatures.add(signature);
                uniqueSegments.push(seg);
            }
        }
        return uniqueSegments;
    }

    _performPhysicalUpdates(dt, fluidFieldRef) {
        // Fluid interactions
        if (fluidFieldRef) {
            for (let point of this.massPoints) {
                if (point.isFixed) continue; 

                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);

                if (point.movementType === MovementType.FLOATING) {
                    const rawFluidVx = fluidFieldRef.Vx[idx];
                    const rawFluidVy = fluidFieldRef.Vy[idx];
                    let fluidDisplacementPx = new Vec2(rawFluidVx * fluidFieldRef.scaleX * dt, rawFluidVy * fluidFieldRef.scaleY * dt);
                    let effectiveFluidDisplacementPx = fluidDisplacementPx.mul(this.fluidCurrentStrength);
                    let currentPointDisplacementPx = point.pos.sub(point.prevPos);
                    let blendedDisplacementPx = currentPointDisplacementPx.mul(1.0 - this.fluidEntrainment)
                                                     .add(effectiveFluidDisplacementPx.mul(this.fluidEntrainment));
                    point.prevPos = point.pos.clone().sub(blendedDisplacementPx);
                }
                
                if (point.nodeType === NodeType.EMITTER) { 
                    let dyeEmissionStrength = 50 * point.currentExertionLevel; 
                    fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], dyeEmissionStrength);
                }
            }
        }

        for (let spring of this.springs) spring.applyForce();
        for (let point of this.massPoints) point.update(dt);

        const MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ = (MAX_PIXELS_PER_FRAME_DISPLACEMENT)**2;
        for (let point of this.massPoints) {
            if (point.isFixed) continue; 

            const displacementSq = (point.pos.x - point.prevPos.x)**2 + (point.pos.y - point.prevPos.y)**2;
            if (displacementSq > MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ || isNaN(point.pos.x) || isNaN(point.pos.y) || !isFinite(point.pos.x) || !isFinite(point.pos.y)) {
                this.isUnstable = true;
                // console.warn(...)
                return; 
            }

            const implicitVelX = point.pos.x - point.prevPos.x;
            const implicitVelY = point.pos.y - point.prevPos.y;

            if (IS_WORLD_WRAPPING) {
                if (point.pos.x < 0) { point.pos.x += WORLD_WIDTH; point.prevPos.x += WORLD_WIDTH; }
                else if (point.pos.x > WORLD_WIDTH) { point.pos.x -= WORLD_WIDTH; point.prevPos.x -= WORLD_WIDTH; }
                if (point.pos.y < 0) { point.pos.y += WORLD_HEIGHT; point.prevPos.y += WORLD_HEIGHT; }
                else if (point.pos.y > WORLD_HEIGHT) { point.pos.y -= WORLD_HEIGHT; point.prevPos.y -= WORLD_HEIGHT; }
            } else {
                if (point.pos.x - point.radius < 0) {
                    point.pos.x = point.radius;
                    point.prevPos.x = point.pos.x - implicitVelX * restitution;
                } else if (point.pos.x + point.radius > WORLD_WIDTH) {
                    point.pos.x = WORLD_WIDTH - point.radius;
                    point.prevPos.x = point.pos.x - implicitVelX * restitution;
                }
                if (point.pos.y - point.radius < 0) {
                    point.pos.y = point.radius;
                    point.prevPos.y = point.pos.y - implicitVelY * restitution;
                } else if (point.pos.y + point.radius > WORLD_HEIGHT) {
                    point.pos.y = WORLD_HEIGHT - point.radius;
                    point.prevPos.y = point.pos.y - implicitVelY * restitution;
                }
            }
        }
    }

    _finalizeUpdateAndCheckStability(dt) { 
        if (this.isUnstable) return; 

        // Inter-body repulsion & Predation
        for (let i_p1 = 0; i_p1 < this.massPoints.length; i_p1++) {
            const p1 = this.massPoints[i_p1];
            if (p1.isFixed) continue;

            const p1Gx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(p1.pos.x / GRID_CELL_SIZE)));
            const p1Gy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(p1.pos.y / GRID_CELL_SIZE)));

            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const checkGx = p1Gx + dx;
                    const checkGy = p1Gy + dy;

                    if (checkGx >= 0 && checkGx < GRID_COLS && checkGy >= 0 && checkGy < GRID_ROWS) {
                        const cellIndex = checkGx + checkGy * GRID_COLS;
                        if (Array.isArray(spatialGrid[cellIndex])) {
                            const cellBucket = spatialGrid[cellIndex];
                            for (const otherItem of cellBucket) {
                                if (otherItem.type === 'softbody_point') {
                                    if (otherItem.bodyRef === this) continue;
                                    const p2 = otherItem.pointRef;
                                    if (p2.isFixed) continue;

                                    const diff = p1.pos.sub(p2.pos);
                                    const distSq = diff.magSq();
                                    const interactionRadius = (p1.radius + p2.radius) * BODY_REPULSION_RADIUS_FACTOR;

                                    if (distSq < interactionRadius * interactionRadius && distSq > 0.0001) {
                                        const dist = Math.sqrt(distSq);
                                        const overlap = interactionRadius - dist;
                                        const forceDir = diff.normalize();
                                        const repulsionForceMag = BODY_REPULSION_STRENGTH * overlap * 0.5;
                                        const repulsionForce = forceDir.mul(repulsionForceMag);
                                        p1.applyForce(repulsionForce);
                                    }

                                    if (p1.nodeType === NodeType.PREDATOR) {
                                        const p1Exertion = p1.currentExertionLevel || 0;
                                        const effectivePredationRadiusMultiplier = PREDATION_RADIUS_MULTIPLIER_BASE + (PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * p1Exertion);
                                        const predationRadius = p1.radius * effectivePredationRadiusMultiplier;
                                        
                                        if (distSq < predationRadius * predationRadius) {
                                            const effectiveEnergySapped = ENERGY_SAPPED_PER_PREDATION_BASE + (ENERGY_SAPPED_PER_PREDATION_MAX_BONUS * p1Exertion);
                                            const energyToSap = Math.min(otherItem.bodyRef.creatureEnergy, effectiveEnergySapped);
                                            if (energyToSap > 0) {
                                                otherItem.bodyRef.creatureEnergy -= energyToSap;
                                                this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyToSap); 
                                                this.energyGainedFromPredation += energyToSap;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Eating Logic
        for (let point of this.massPoints) {
            if (point.isFixed) continue; 
            if (point.nodeType === NodeType.EATER) { 
                const pointExertion = point.currentExertionLevel || 0;
                const effectiveEatingRadiusMultiplier = EATING_RADIUS_MULTIPLIER_BASE + (EATING_RADIUS_MULTIPLIER_MAX_BONUS * pointExertion);
                const eatingRadius = point.radius * effectiveEatingRadiusMultiplier;
                const eatingRadiusSq = eatingRadius * eatingRadius;
                const eaterGx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(point.pos.x / GRID_CELL_SIZE)));
                const eaterGy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(point.pos.y / GRID_CELL_SIZE)));

                for (let dy = -1; dy <= 1; dy++) { 
                    for (let dx = -1; dx <= 1; dx++) {
                        const checkGx = eaterGx + dx;
                        const checkGy = eaterGy + dy;
                        if (checkGx >= 0 && checkGx < GRID_COLS && checkGy >= 0 && checkGy < GRID_ROWS) {
                            const cellIndex = checkGx + checkGy * GRID_COLS;
                            if (Array.isArray(spatialGrid[cellIndex])) {
                                const cellBucket = spatialGrid[cellIndex];
                                for (let k = cellBucket.length - 1; k >= 0; k--) { 
                                    const item = cellBucket[k];
                                    if (item.type === 'particle') {
                                        const particle = item.particleRef;
                                        if (particle.life > 0 && !particle.isEaten) {
                                            const distSq = point.pos.sub(particle.pos).magSq();
                                            if (distSq < eatingRadiusSq) {
                                                particle.isEaten = true;
                                                particle.life = 0; 
                                                let energyGain = ENERGY_PER_PARTICLE;
                                                if (nutrientField && fluidField) { 
                                                    const particleGx = Math.floor(particle.pos.x / fluidField.scaleX);
                                                    const particleGy = Math.floor(particle.pos.y / fluidField.scaleY);
                                                    const nutrientIdxAtParticle = fluidField.IX(particleGx, particleGy);
                                                    const baseNutrientValueAtParticle = nutrientField[nutrientIdxAtParticle] !== undefined ? nutrientField[nutrientIdxAtParticle] : 1.0;
                                                    const effectiveNutrientAtParticle = baseNutrientValueAtParticle * globalNutrientMultiplier;
                                                    energyGain *= Math.max(MIN_NUTRIENT_VALUE, effectiveNutrientAtParticle);
                                                }
                                                this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyGain); 
                                                this.energyGainedFromEating += energyGain;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (this.isUnstable) return; 

        // Final Instability Checks: Springs and Span
        const localMaxSpringStretchFactor = MAX_SPRING_STRETCH_FACTOR;
        const localMaxSpanPerPointFactor = MAX_SPAN_PER_POINT_FACTOR;

        for (const spring of this.springs) {
            const currentLength = spring.p1.pos.sub(spring.p2.pos).mag();
            if (currentLength > spring.restLength * localMaxSpringStretchFactor) {
                this.isUnstable = true;
                // console.warn(...)
                return;
            }
        }

        if (this.massPoints.length > 2) { 
            const bbox = this.getBoundingBox();
            if (bbox.width > this.massPoints.length * localMaxSpanPerPointFactor ||
                bbox.height > this.massPoints.length * localMaxSpanPerPointFactor) {
                this.isUnstable = true;
                // console.warn(...)
                return;
            }
        }

        this.ticksSinceBirth++;
        if (this.ticksSinceBirth > this.effectiveReproductionCooldown) { // Use effective cooldown
            this.canReproduce = true;
        }
    }
}


// --- FluidField Class (Simplified) ---
class FluidField {
    constructor(size, diffusion, viscosity, dt, scaleX, scaleY) {
        this.size = Math.round(size);
        this.dt = dt;
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.scaleX = scaleX; 
        this.scaleY = scaleY; 
        this.useWrapping = false;
        this.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT;

        this.densityR = new Float32Array(this.size * this.size).fill(0);
        this.densityG = new Float32Array(this.size * this.size).fill(0);
        this.densityB = new Float32Array(this.size * this.size).fill(0);
        this.densityR0 = new Float32Array(this.size * this.size).fill(0);
        this.densityG0 = new Float32Array(this.size * this.size).fill(0);
        this.densityB0 = new Float32Array(this.size * this.size).fill(0);

        this.Vx = new Float32Array(this.size * this.size).fill(0);
        this.Vy = new Float32Array(this.size * this.size).fill(0);
        this.Vx0 = new Float32Array(this.size * this.size).fill(0);
        this.Vy0 = new Float32Array(this.size * this.size).fill(0);

        this.iterations = 4; 
    }

    IX(x, y) {
        if (this.useWrapping) {
            x = (Math.floor(x) % this.size + this.size) % this.size;
            y = (Math.floor(y) % this.size + this.size) % this.size;
        } else {
            x = Math.max(0, Math.min(x, this.size - 1));
            y = Math.max(0, Math.min(y, this.size - 1));
        }
        return Math.floor(x) + Math.floor(y) * this.size;
    }

    addDensity(x, y, emitterR, emitterG, emitterB, emissionStrength) {
        const idx = this.IX(x, y);
        const normalizedEmissionEffect = (emissionStrength / 50.0) * DYE_PULL_RATE;
        this.densityR[idx] = Math.max(0, Math.min(255, this.densityR[idx] + (emitterR - this.densityR[idx]) * normalizedEmissionEffect));
        this.densityG[idx] = Math.max(0, Math.min(255, this.densityG[idx] + (emitterG - this.densityG[idx]) * normalizedEmissionEffect));
        this.densityB[idx] = Math.max(0, Math.min(255, this.densityB[idx] + (emitterB - this.densityB[idx]) * normalizedEmissionEffect));
    }

    addVelocity(x, y, amountX, amountY) {
        const idx = this.IX(x, y);
        this.Vx[idx] = Math.max(-this.maxVelComponent, Math.min(this.Vx[idx] + amountX, this.maxVelComponent));
        this.Vy[idx] = Math.max(-this.maxVelComponent, Math.min(this.Vy[idx] + amountY, this.maxVelComponent));
    }

    clampVelocityComponents(arr) {
        for(let i=0; i < arr.length; i++) {
            arr[i] = Math.max(-this.maxVelComponent, Math.min(arr[i], this.maxVelComponent));
        }
    }

    lin_solve(b, x, x0, a_global_param, c_global_param, field_type, base_diff_rate, dt_param) {
        const cRecipGlobal = 1.0 / c_global_param;
        for (let k_iter = 0; k_iter < this.iterations; k_iter++) {
            for (let j = 1; j < this.size - 1; j++) {
                for (let i = 1; i < this.size - 1; i++) {
                    const idx = this.IX(i,j);
                    let effective_a = a_global_param;
                    let effective_cRecip = cRecipGlobal;

                    if ((field_type === 'velX' || field_type === 'velY')) {
                        if (viscosityField && viscosityField[idx] !== undefined) { 
                            const localViscosityMultiplier = Math.max(MIN_VISCOSITY_MULTIPLIER, Math.min(viscosityField[idx], MAX_VISCOSITY_MULTIPLIER));
                            const cell_specific_diff_rate = base_diff_rate * localViscosityMultiplier; 
                            const temp_effective_a = dt_param * cell_specific_diff_rate * (this.size - 2) * (this.size - 2);
                            const temp_denominator_c = 1 + 4 * temp_effective_a;
                            if (temp_denominator_c !== 0 && !isNaN(temp_effective_a) && isFinite(temp_effective_a)) {
                                effective_a = temp_effective_a;
                                effective_cRecip = 1.0 / temp_denominator_c;
                            } 
                        }
                    }
                    x[idx] = (x0[idx] + effective_a * (x[this.IX(i+1,j)] + x[this.IX(i-1,j)] + x[this.IX(i,j+1)] + x[this.IX(i,j-1)])) * effective_cRecip;
                }
            }
            this.set_bnd(b, x);
        }
    }

    diffuse(b, x_out, x_in, base_diff_rate, dt, field_type = 'density') {
        const a_global = dt * base_diff_rate * (this.size - 2) * (this.size - 2);
        this.lin_solve(b, x_out, x_in, a_global, 1 + 4 * a_global, field_type, base_diff_rate, dt);
    }

    project(velocX_in_out, velocY_in_out, p_temp, div_temp) {
        for (let j = 1; j < this.size - 1; j++) {
            for (let i = 1; i < this.size - 1; i++) {
                const idx = this.IX(i,j);
                div_temp[idx] = -0.5 * (velocX_in_out[this.IX(i+1,j)] - velocX_in_out[this.IX(i-1,j)] + velocY_in_out[this.IX(i,j+1)] - velocY_in_out[this.IX(i,j-1)]) / this.size;
                p_temp[idx] = 0;
            }
        }
        this.set_bnd(0, div_temp);
        this.set_bnd(0, p_temp);
        this.lin_solve(0, p_temp, div_temp, 1, 4, 'pressure', 0, 0);

        for (let j = 1; j < this.size - 1; j++) {
            for (let i = 1; i < this.size - 1; i++) {
                const idx = this.IX(i,j);
                velocX_in_out[idx] -= 0.5 * (p_temp[this.IX(i+1,j)] - p_temp[this.IX(i-1,j)]) * this.size;
                velocY_in_out[idx] -= 0.5 * (p_temp[this.IX(i,j+1)] - p_temp[this.IX(i,j-1)]) * this.size;
            }
        }
        this.set_bnd(1, velocX_in_out);
        this.set_bnd(2, velocY_in_out);
    }

    advect(b, d_out, d_in, velocX_source, velocY_source, dt) {
        let i0, i1, j0, j1;
        const N = this.size;
        const dtx_scaled = dt * N; 
        const dty_scaled = dt * N;

        let s0, s1, t0, t1;
        let x, y;

        for (let j_cell = 1; j_cell < N - 1; j_cell++) {
            for (let i_cell = 1; i_cell < N - 1; i_cell++) {
                const current_idx = this.IX(i_cell, j_cell);
                x = i_cell - (dtx_scaled * velocX_source[current_idx]); 
                y = j_cell - (dty_scaled * velocY_source[current_idx]);

                if (this.useWrapping) {
                    x = (x % N + N) % N;
                    y = (y % N + N) % N;
                    i0 = Math.floor(x);
                    j0 = Math.floor(y);
                    i1 = (i0 + 1) % N;
                    j1 = (j0 + 1) % N;
                } else {
                    if (x < 0.5) x = 0.5;
                    if (x > N - 1.5) x = N - 1.5; 
                    i0 = Math.floor(x);
                    i1 = i0 + 1;
                    if (y < 0.5) y = 0.5;
                    if (y > N - 1.5) y = N - 1.5;
                    j0 = Math.floor(y);
                    j1 = j0 + 1;
                }

                s1 = x - i0;
                s0 = 1.0 - s1;
                t1 = y - j0;
                t0 = 1.0 - t1;
                
                d_out[current_idx] = s0 * (t0 * d_in[this.IX(i0,j0)] + t1 * d_in[this.IX(i0,j1)]) +
                                     s1 * (t0 * d_in[this.IX(i1,j0)] + t1 * d_in[this.IX(i1,j1)]);
            }
        }
        this.set_bnd(b, d_out);
    }

    set_bnd(b, x_arr) {
        if (this.useWrapping) {
            for (let i = 1; i < this.size - 1; i++) {
                x_arr[this.IX(i, 0)] = x_arr[this.IX(i, this.size - 2)];
                x_arr[this.IX(i, this.size - 1)] = x_arr[this.IX(i, 1)];
            }
            for (let j = 1; j < this.size - 1; j++) {
                x_arr[this.IX(0, j)] = x_arr[this.IX(this.size - 2, j)];
                x_arr[this.IX(this.size - 1, j)] = x_arr[this.IX(1, j)];
            }
            x_arr[this.IX(0, 0)] = 0.5 * (x_arr[this.IX(1, 0)] + x_arr[this.IX(0, 1)]);
            x_arr[this.IX(0, this.size - 1)] = 0.5 * (x_arr[this.IX(1, this.size - 1)] + x_arr[this.IX(0, this.size - 2)]);
            x_arr[this.IX(this.size - 1, 0)] = 0.5 * (x_arr[this.IX(this.size - 2, 0)] + x_arr[this.IX(this.size - 1, 1)]);
            x_arr[this.IX(this.size - 1, this.size - 1)] = 0.5 * (x_arr[this.IX(this.size - 2, this.size - 1)] + x_arr[this.IX(this.size - 1, this.size - 2)]);
        } else {
            for (let i = 1; i < this.size - 1; i++) {
                x_arr[this.IX(i, 0)] = b === 2 ? -x_arr[this.IX(i, 1)] : x_arr[this.IX(i, 1)];
                x_arr[this.IX(i, this.size - 1)] = b === 2 ? -x_arr[this.IX(i, this.size - 2)] : x_arr[this.IX(i, this.size - 2)];
            }
            for (let j = 1; j < this.size - 1; j++) {
                x_arr[this.IX(0, j)] = b === 1 ? -x_arr[this.IX(1, j)] : x_arr[this.IX(1, j)];
                x_arr[this.IX(this.size - 1, j)] = b === 1 ? -x_arr[this.IX(this.size - 2, j)] : x_arr[this.IX(this.size - 2, j)];
            }
            x_arr[this.IX(0, 0)] = 0.5 * (x_arr[this.IX(1, 0)] + x_arr[this.IX(0, 1)]);
            x_arr[this.IX(0, this.size - 1)] = 0.5 * (x_arr[this.IX(1, this.size - 1)] + x_arr[this.IX(0, this.size - 2)]);
            x_arr[this.IX(this.size - 1, 0)] = 0.5 * (x_arr[this.IX(this.size - 2, 0)] + x_arr[this.IX(this.size - 1, 1)]);
            x_arr[this.IX(this.size - 1, this.size - 1)] = 0.5 * (x_arr[this.IX(this.size - 2, this.size - 1)] + x_arr[this.IX(this.size - 1, this.size - 2)]);
        }
    }

    step() {
        this.diffuse(1, this.Vx0, this.Vx, this.viscosity, this.dt, 'velX');
        this.diffuse(2, this.Vy0, this.Vy, this.viscosity, this.dt, 'velY');
        this.clampVelocityComponents(this.Vx0);
        this.clampVelocityComponents(this.Vy0);
        this.project(this.Vx0, this.Vy0, this.Vx, this.Vy);
        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, this.dt);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, this.dt);
        this.clampVelocityComponents(this.Vx);
        this.clampVelocityComponents(this.Vy);
        this.project(this.Vx, this.Vy, this.Vx0, this.Vy0);
        this.diffuse(0, this.densityR0, this.densityR, this.diffusion, this.dt, 'density');
        this.diffuse(0, this.densityG0, this.densityG, this.diffusion, this.dt, 'density');
        this.diffuse(0, this.densityB0, this.densityB, this.diffusion, this.dt, 'density');
        this.advect(0, this.densityR, this.densityR0, this.Vx, this.Vy, this.dt);
        this.advect(0, this.densityG, this.densityG0, this.Vx, this.Vy, this.dt);
        this.advect(0, this.densityB, this.densityB0, this.Vx, this.Vy, this.dt);
        for (let i = 0; i < this.densityR.length; i++) {
            this.densityR[i] = Math.max(0, this.densityR[i] - FLUID_FADE_RATE * 255 * this.dt);
            this.densityG[i] = Math.max(0, this.densityG[i] - FLUID_FADE_RATE * 255 * this.dt);
            this.densityB[i] = Math.max(0, this.densityB[i] - FLUID_FADE_RATE * 255 * this.dt);
        }
    }

    draw(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
        const N = Math.round(this.size);
        if (N <= 0 || !Number.isFinite(N)) {
            console.error("FluidField.draw: Invalid N size:", N);
            return;
        }

        const worldCellWidth = WORLD_WIDTH / N;
        const worldCellHeight = WORLD_HEIGHT / N;

        const viewportWorldWidth = viewportCanvasWidth / currentZoom;
        const viewportWorldHeight = viewportCanvasHeight / currentZoom;

        const viewLeftWorld = viewOffsetXWorld;
        const viewTopWorld = viewOffsetYWorld;
        const viewRightWorld = viewOffsetXWorld + viewportWorldWidth;
        const viewBottomWorld = viewOffsetYWorld + viewportWorldHeight;

        const startCol = Math.max(0, Math.floor(viewLeftWorld / worldCellWidth));
        const endCol = Math.min(N - 1, Math.floor(viewRightWorld / worldCellWidth));
        const startRow = Math.max(0, Math.floor(viewTopWorld / worldCellHeight));
        const endRow = Math.min(N - 1, Math.floor(viewBottomWorld / worldCellHeight));

        if (startCol > endCol || startRow > endRow) return; // No visible cells

        for (let j = startRow; j <= endRow; j++) {
            for (let i = startCol; i <= endCol; i++) {
                const idx = this.IX(i, j);
                const rVal = Math.min(255, Math.max(0, Math.floor(this.densityR[idx])));
                const gVal = Math.min(255, Math.max(0, Math.floor(this.densityG[idx])));
                const bVal = Math.min(255, Math.max(0, Math.floor(this.densityB[idx])));
                const alphaVal = (rVal > 1 || gVal > 1 || bVal > 1) ? 0.4 : 0; // Original alpha for blocky rendering

                if (alphaVal > 0) {
                    const cellWorldX = i * worldCellWidth;
                    const cellWorldY = j * worldCellHeight;
                    ctxToDrawOn.fillStyle = `rgba(${rVal},${gVal},${bVal},${alphaVal.toFixed(2)})`;
                    ctxToDrawOn.fillRect(cellWorldX, cellWorldY, worldCellWidth, worldCellHeight);
                }
            }
        }
    }

    clear() {
        this.densityR.fill(0); this.densityG.fill(0); this.densityB.fill(0);
        this.densityR0.fill(0); this.densityG0.fill(0); this.densityB0.fill(0);
        this.Vx.fill(0); this.Vy.fill(0);
        this.Vx0.fill(0); this.Vy0.fill(0);
    }
}

// --- Particle Class (for fluid visualization) ---
class Particle {
    constructor(x, y, fluidFieldRef) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(Math.random() * 0.2 - 0.1, Math.random() * 0.2 - 0.1);
        this.fluidField = fluidFieldRef;
        this.life = 1.0;
        this.lifeDecay = PARTICLE_BASE_LIFE_DECAY + Math.random() * PARTICLE_LIFE_DECAY_RANDOM_FACTOR;
        this.size = Math.random() * 1.5 + 0.5;
        this.isEaten = false;
    }

    update(dt) {
        if (this.isEaten) {
             // Particle is fading out after being eaten
             this.life -= 0.2 * dt * 60; // Faster fade for eaten particles
             if (this.life <= 0) {
                 // To be removed in updatePhysics
             }
            return;
        }

        if (!this.fluidField) return;
        const fluidGridX = Math.floor(this.pos.x / this.fluidField.scaleX); // Use scaleX
        const fluidGridY = Math.floor(this.pos.y / this.fluidField.scaleY); // Use scaleY
        const idx = this.fluidField.IX(fluidGridX, fluidGridY);

        const fluidVelX = this.fluidField.Vx[idx];
        const fluidVelY = this.fluidField.Vy[idx];

        this.vel.x = this.vel.x * (1.0 - PARTICLE_FLUID_INFLUENCE) + fluidVelX * PARTICLE_FLUID_INFLUENCE;
        this.vel.y = this.vel.y * (1.0 - PARTICLE_FLUID_INFLUENCE) + fluidVelY * PARTICLE_FLUID_INFLUENCE;
        this.vel.x += (Math.random() - 0.5) * 0.05;
        this.vel.y += (Math.random() - 0.5) * 0.05;

        this.pos = this.pos.add(this.vel.mul(dt * 100));

        if (IS_WORLD_WRAPPING) {
            if (this.pos.x < 0) this.pos.x += WORLD_WIDTH;
            if (this.pos.x > WORLD_WIDTH) this.pos.x -= WORLD_WIDTH;
            if (this.pos.y < 0) this.pos.y += WORLD_HEIGHT;
            if (this.pos.y > WORLD_HEIGHT) this.pos.y -= WORLD_HEIGHT;
        } else {
            if (this.pos.x - this.radius < 0) {
                this.pos.x = this.radius;
                this.vel.x = 0;
            } else if (this.pos.x + this.radius > WORLD_WIDTH) {
                this.pos.x = WORLD_WIDTH - this.radius;
                this.vel.x = 0;
            }
            if (this.pos.y - this.radius < 0) {
                this.pos.y = this.radius;
                this.vel.y = 0;
            } else if (this.pos.y + this.radius > WORLD_HEIGHT) {
                this.pos.y = WORLD_HEIGHT - this.radius;
                this.vel.y = 0;
            }
        }

        if (!IS_PARTICLE_LIFE_INFINITE) {
            this.life -= (PARTICLE_BASE_LIFE_DECAY + Math.random() * PARTICLE_LIFE_DECAY_RANDOM_FACTOR) * dt * 60;
            if (this.life <=0) {
               // To be removed in updatePhysics
            }
        } else {
            this.life = 1.0;
        }
    }

    // respawn() removed as particles are now emitted and removed

    draw(ctx) {
        const alpha = IS_PARTICLE_LIFE_INFINITE ? 0.7 : Math.max(0, this.life * 0.7);
        if (alpha <= 0.01 && !IS_PARTICLE_LIFE_INFINITE) return;
        ctx.fillStyle = `rgba(220, 220, 250, ${alpha})`;
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
} 