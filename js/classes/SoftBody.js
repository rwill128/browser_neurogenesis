import config from '../config.js';
import { Vec2 } from '../utils.js';
import { NodeType, RLRewardStrategy, RLAlgorithmType, EyeTargetType, MovementType } from './constants.js';
import {MassPoint} from "./MassPoint.js";
import {Spring} from "./Spring.js";
import {Brain} from "./Brain.js";

// --- SoftBody Class ---
export class SoftBody {
    constructor(id, initialX, initialY, creationData = null, isBlueprint = false) {
        this.id = id;
        this.massPoints = [];
        this.springs = [];

        // Genetic blueprint
        this.blueprintPoints = []; // Array of { relX, relY, radius, mass, nodeType, movementType, dyeColor, canBeGrabber, neuronDataBlueprint }
        this.blueprintSprings = []; // Array of { p1Index, p2Index, restLength, isRigid } (indices refer to blueprintPoints)

        this.isUnstable = false;
        this.ticksSinceBirth = 0;
        this.canReproduce = false;
        this.shapeType = creationData ? creationData.shapeType : Math.floor(Math.random() * 3);
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
        this.energyCostFromJetNodes = 0;
        this.energyCostFromAttractorNodes = 0;
        this.energyCostFromRepulsorNodes = 0;

        this.currentMaxEnergy = config.BASE_MAX_CREATURE_ENERGY; // Initial placeholder
        this.blueprintRadius = 0; // New: Approximate radius based on blueprint points

        // Node counts for brain initialization optimization
        this.numEmitterNodes = 0;
        this.numSwimmerNodes = 0;
        this.numEaterNodes = 0;
        this.numPredatorNodes = 0;
        this.numEyeNodes = 0;
        this.numJetNodes = 0;
        this.numPotentialGrabberNodes = 0;
        this.numAttractorNodes = 0;
        this.numRepulsorNodes = 0;

        this.failedReproductionCooldown = 0; // New: Cooldown after a failed reproduction attempt
        this.energyGainedFromPhotosynthesisThisTick = 0; // New: Photosynthesis gain in the current tick

        // Initialize heritable/mutable properties
        if (isBlueprint && creationData) {
            // --- CREATION FROM IMPORTED BLUEPRINT ---
            const blueprint = creationData;
            this.stiffness = blueprint.stiffness;
            this.springDamping = blueprint.springDamping;
            this.motorImpulseInterval = blueprint.motorImpulseInterval;
            this.motorImpulseMagnitudeCap = blueprint.motorImpulseMagnitudeCap;
            this.emitterStrength = blueprint.emitterStrength;
            this.emitterDirection = new Vec2(blueprint.emitterDirection.x, blueprint.emitterDirection.y);
            this.numOffspring = blueprint.numOffspring;
            this.offspringSpawnRadius = blueprint.offspringSpawnRadius;
            this.pointAddChance = blueprint.pointAddChance;
            this.springConnectionRadius = blueprint.springConnectionRadius;
            this.jetMaxVelocityGene = blueprint.jetMaxVelocityGene;
            this.reproductionEnergyThreshold = blueprint.reproductionEnergyThreshold;
            this.reproductionCooldownGene = blueprint.reproductionCooldownGene;
            this.defaultActivationPattern = blueprint.defaultActivationPattern;
            this.defaultActivationLevel = blueprint.defaultActivationLevel;
            this.defaultActivationPeriod = blueprint.defaultActivationPeriod;
            this.defaultActivationPhaseOffset = blueprint.defaultActivationPhaseOffset;
            this.rlAlgorithmType = blueprint.rlAlgorithmType;
            this.rewardStrategy = blueprint.rewardStrategy;
            
            // Directly use the blueprint's structure
            this.blueprintPoints = JSON.parse(JSON.stringify(blueprint.blueprintPoints));
            this.blueprintSprings = JSON.parse(JSON.stringify(blueprint.blueprintSprings));
            this._instantiatePhenotypeFromBlueprint(initialX, initialY);

        } else {
            // --- CREATION FROM PARENT (REPRODUCTION) OR FROM SCRATCH ---
            const parentBody = creationData; // In this case, creationData is the parentBody
            if (parentBody) {
                this.stiffness = parentBody.stiffness * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.stiffness !== parentBody.stiffness) config.mutationStats.springStiffness++;
                this.springDamping = parentBody.springDamping * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.springDamping !== parentBody.springDamping) config.mutationStats.springDamping++;
                
                let oldMotorInterval = parentBody.motorImpulseInterval;
                this.motorImpulseInterval = parentBody.motorImpulseInterval * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (Math.floor(this.motorImpulseInterval) !== Math.floor(oldMotorInterval)) config.mutationStats.motorInterval++;

                let oldMotorCap = parentBody.motorImpulseMagnitudeCap;
                this.motorImpulseMagnitudeCap = parentBody.motorImpulseMagnitudeCap * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.motorImpulseMagnitudeCap !== oldMotorCap) config.mutationStats.motorCap++;

                let oldEmitterStrength = parentBody.emitterStrength;
                this.emitterStrength = parentBody.emitterStrength * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.emitterStrength !== oldEmitterStrength) config.mutationStats.emitterStrength++;

                let oldJetMaxVel = parentBody.jetMaxVelocityGene;
                this.jetMaxVelocityGene = parentBody.jetMaxVelocityGene * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.jetMaxVelocityGene !== oldJetMaxVel) config.mutationStats.jetMaxVelocityGene++;

                let offspringNumChange = (Math.random() < Math.max(0, Math.min(1, config.MUTATION_CHANCE_BOOL * config.GLOBAL_MUTATION_RATE_MODIFIER))) ? (Math.random() < 0.5 ? -1 : 1) : 0;
                this.numOffspring = parentBody.numOffspring + offspringNumChange;
                if (offspringNumChange !== 0) config.mutationStats.numOffspring++;

                let oldOffspringSpawnRadius = parentBody.offspringSpawnRadius;
                this.offspringSpawnRadius = parentBody.offspringSpawnRadius * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.5));
                if (this.offspringSpawnRadius !== oldOffspringSpawnRadius) config.mutationStats.offspringSpawnRadius++;
                
                let oldPointAddChance = parentBody.pointAddChance;
                this.pointAddChance = parentBody.pointAddChance * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 2));
                if (this.pointAddChance !== oldPointAddChance) config.mutationStats.pointAddChanceGene++;

                let oldSpringConnectionRadius = parentBody.springConnectionRadius;
                this.springConnectionRadius = parentBody.springConnectionRadius * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.springConnectionRadius !== oldSpringConnectionRadius) config.mutationStats.springConnectionRadiusGene++;
                
                if (parentBody.emitterDirection) {
                    const oldEmitterDirX = parentBody.emitterDirection.x;
                    const angleMutation = (Math.random() - 0.5) * Math.PI * 0.2 * config.GLOBAL_MUTATION_RATE_MODIFIER;
                    const cosA = Math.cos(angleMutation);
                    const sinA = Math.sin(angleMutation);
                    this.emitterDirection = new Vec2(parentBody.emitterDirection.x * cosA - parentBody.emitterDirection.y * sinA, parentBody.emitterDirection.x * sinA + parentBody.emitterDirection.y * cosA).normalize();
                    if (this.emitterDirection.x !== oldEmitterDirX) config.mutationStats.emitterDirection++; // Simplified check
                } else {
                    this.emitterDirection = new Vec2(Math.random()*2-1, Math.random()*2-1).normalize();
                    console.warn(`Parent body ${parentBody.id} was missing emitterDirection. Offspring ${this.id} gets random emitterDirection.`);
                    config.mutationStats.emitterDirection++; // Count as a change if parent was missing it
                }
                
                let oldReproThreshold = parentBody.reproductionEnergyThreshold;
                this.reproductionEnergyThreshold = parentBody.reproductionEnergyThreshold; 
                // Mutation of reproductionEnergyThreshold happens later, after currentMaxEnergy is set for the offspring

                // Inherit and mutate activation pattern properties
                if (Math.random() < config.ACTIVATION_PATTERN_MUTATION_CHANCE) {
                    const patterns = Object.values(config.ActivationPatternType);
                    this.defaultActivationPattern = patterns[Math.floor(Math.random() * patterns.length)];
                } else {
                    this.defaultActivationPattern = parentBody.defaultActivationPattern;
                }
                this.defaultActivationLevel = parentBody.defaultActivationLevel * (1 + (Math.random() - 0.5) * 2 * config.ACTIVATION_PARAM_MUTATION_MAGNITUDE);
                this.defaultActivationPeriod = parentBody.defaultActivationPeriod * (1 + (Math.random() - 0.5) * 2 * config.ACTIVATION_PARAM_MUTATION_MAGNITUDE);
                this.defaultActivationPhaseOffset = parentBody.defaultActivationPhaseOffset + (Math.random() - 0.5) * (parentBody.defaultActivationPeriod * 0.2); 
                
                // Inherit/Mutate Reward Strategy
                if (Math.random() < config.RLRewardStrategy_MUTATION_CHANCE) {
                    const strategies = Object.values(RLRewardStrategy);
                    let newStrategy = strategies[Math.floor(Math.random() * strategies.length)];
                    if (newStrategy !== parentBody.rewardStrategy) {
                        this.rewardStrategy = newStrategy;
                        config.mutationStats.rewardStrategyChange++;
                    } else {
                        if (strategies.length > 1) {
                            let tempStrategies = strategies.filter(s => s !== parentBody.rewardStrategy);
                            if (tempStrategies.length > 0) {
                                this.rewardStrategy = tempStrategies[Math.floor(Math.random() * tempStrategies.length)];
                                config.mutationStats.rewardStrategyChange++; 
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
                this.reproductionCooldownGene = parentBody.reproductionCooldownGene * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.2));
                this.reproductionCooldownGene = Math.max(50, Math.min(Math.floor(this.reproductionCooldownGene), 20000)); // Clamp
                if (this.reproductionCooldownGene !== parentBody.reproductionCooldownGene) {
                    config.mutationStats.reproductionCooldownGene = (config.mutationStats.reproductionCooldownGene || 0) + 1;
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
                this.jetMaxVelocityGene = config.JET_MAX_VELOCITY_GENE_DEFAULT * (0.8 + Math.random() * 0.4);
                this.reproductionEnergyThreshold = config.BASE_MAX_CREATURE_ENERGY; // Will be refined based on actual max energy

                // Default Activation Pattern Properties for new creature
                const patterns = Object.values(config.ActivationPatternType);
                this.defaultActivationPattern = patterns[Math.floor(Math.random() * patterns.length)];
                this.defaultActivationLevel = config.DEFAULT_ACTIVATION_LEVEL_MIN + Math.random() * (config.DEFAULT_ACTIVATION_LEVEL_MAX - config.DEFAULT_ACTIVATION_LEVEL_MIN);
                this.defaultActivationPeriod = config.DEFAULT_ACTIVATION_PERIOD_MIN_TICKS + Math.floor(Math.random() * (config.DEFAULT_ACTIVATION_PERIOD_MAX_TICKS - config.DEFAULT_ACTIVATION_PERIOD_MIN_TICKS + 1));
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
            this.defaultActivationLevel = Math.max(config.DEFAULT_ACTIVATION_LEVEL_MIN * 0.1, Math.min(this.defaultActivationLevel, config.DEFAULT_ACTIVATION_LEVEL_MAX * 2.0)); // Wider clamping for more variance
            this.defaultActivationPeriod = Math.max(config.DEFAULT_ACTIVATION_PERIOD_MIN_TICKS * 0.25, Math.min(this.defaultActivationPeriod, config.DEFAULT_ACTIVATION_PERIOD_MAX_TICKS * 2.0));
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
            this.jetMaxVelocityGene = Math.max(0.1, Math.min(this.jetMaxVelocityGene, 50.0));

            this.fluidEntrainment = config.BODY_FLUID_ENTRAINMENT_FACTOR;
            this.fluidCurrentStrength = config.FLUID_CURRENT_STRENGTH_ON_BODY;
            this.bodyPushStrength = config.SOFT_BODY_PUSH_STRENGTH;

            // 1. Create shape (points and initial springs)
            // If created from blueprint, this is already done.
            if (!isBlueprint) {
                this.createShape(initialX, initialY, parentBody);
            }

            // 2. Body Scale Mutation (if offspring)
            const parentBodyForMutation = isBlueprint ? null : creationData;
            if (parentBodyForMutation && Math.random() < config.BODY_SCALE_MUTATION_CHANCE) {
                const scaleFactor = 1.0 + (Math.random() - 0.5) * 2 * config.BODY_SCALE_MUTATION_MAGNITUDE;
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
                    config.mutationStats.bodyScale++;
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
            this.creatureEnergy = this.currentMaxEnergy * config.OFFSPRING_INITIAL_ENERGY_SHARE;

            let oldReproThresholdForStat = this.reproductionEnergyThreshold; // Capture before mutation
            if (!isBlueprint && parentBodyForMutation) {
                this.reproductionEnergyThreshold = this.reproductionEnergyThreshold * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.2));
            } else if (!parentBodyForMutation) { // This is for brand new from scratch or imported
                this.reproductionEnergyThreshold = this.currentMaxEnergy * (0.75 + Math.random() * 0.2);
            }
            // For blueprints, the threshold is already set, but we still clamp it.
            this.reproductionEnergyThreshold = Math.max(this.currentMaxEnergy * 0.05, Math.min(this.reproductionEnergyThreshold, this.currentMaxEnergy));
            this.reproductionEnergyThreshold = Math.round(this.reproductionEnergyThreshold);
            if (parentBodyForMutation && this.reproductionEnergyThreshold !== oldReproThresholdForStat) { // Only count if from parent and changed
                config.mutationStats.reproductionEnergyThreshold++;
            }
            
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
            // For now, it's only set if an EYE node is already present (e.g., from mutation or initial gen).
            // If still null, the NN input for eye will be default/neutral.
        }

        this.brain = new Brain(this); 

        // Temporary vectors for calculations to reduce allocations
        this._tempVec1 = new Vec2();
        this._tempVec2 = new Vec2();
    }

    calculateCurrentMaxEnergy() {
        if (this.massPoints.length === 0) {
            this.currentMaxEnergy = 0; 
        } else {
            this.currentMaxEnergy = this.massPoints.length * config.ENERGY_PER_MASS_POINT_BONUS;
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
        const availableFunctionalNodeTypes = [NodeType.PREDATOR, NodeType.EATER, NodeType.PHOTOSYNTHETIC, NodeType.NEURON, NodeType.EMITTER, NodeType.SWIMMER, NodeType.EYE, NodeType.JET, NodeType.ATTRACTOR, NodeType.REPULSOR];
        const dyeColorChoices = [config.DYE_COLORS.RED, config.DYE_COLORS.GREEN, config.DYE_COLORS.BLUE];

        if (parentBody) {
            // --- Reproduction: Inherit and Mutate Blueprint ---

            // 1. Deep copy blueprint from parent
            this.blueprintPoints = JSON.parse(JSON.stringify(parentBody.blueprintPoints));
            this.blueprintSprings = JSON.parse(JSON.stringify(parentBody.blueprintSprings));

            // 2. Mutate blueprint points (coordinates, types, properties)
            this.blueprintPoints.forEach(bp => {
                // Mutate relative coordinates
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.5)) {
                    bp.relX += (Math.random() - 0.5) * 2; // Smaller jitter for blueprint stability
                    config.mutationStats.blueprintCoordinateChange = (config.mutationStats.blueprintCoordinateChange || 0) + 1;
                }
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.5)) {
                    bp.relY += (Math.random() - 0.5) * 2;
                    config.mutationStats.blueprintCoordinateChange = (config.mutationStats.blueprintCoordinateChange || 0) + 1;
                }

                // Mutate mass & radius
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.mass = Math.max(0.1, Math.min(bp.mass * (1 + (Math.random() - 0.5) * 0.4), 1.0));
                    config.mutationStats.blueprintMassRadiusChange = (config.mutationStats.blueprintMassRadiusChange || 0) + 1;
                }
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.radius = Math.max(0.5, Math.min(bp.radius * (1 + (Math.random() - 0.5) * 0.4), baseRadius * 2.5)); // Max based on baseRadius
                     config.mutationStats.blueprintMassRadiusChange = (config.mutationStats.blueprintMassRadiusChange || 0) + 1;
                }

                // Mutate nodeType
                if (Math.random() < (config.MUTATION_CHANCE_NODE_TYPE * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    const oldNodeType = bp.nodeType;
                    if (Math.random() < config.NEURON_CHANCE) {
                        bp.nodeType = NodeType.NEURON;
                    } else {
                        const otherNodeTypes = availableFunctionalNodeTypes.filter(t => t !== NodeType.NEURON);
                        bp.nodeType = otherNodeTypes[Math.floor(Math.random() * otherNodeTypes.length)];
                    }
                    if (bp.nodeType !== oldNodeType) config.mutationStats.nodeTypeChange++;

                    // If it becomes an EYE, initialize eyeTargetType randomly
                    if (bp.nodeType === NodeType.EYE && bp.eyeTargetType === undefined) {
                        bp.eyeTargetType = Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT;
                    }
                }

                // Mutate movementType
                if (Math.random() < (config.MUTATION_CHANCE_NODE_TYPE * config.GLOBAL_MUTATION_RATE_MODIFIER)) { 
                    const oldMovementType = bp.movementType;
                    const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                    bp.movementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                    if (bp.movementType !== oldMovementType) config.mutationStats.movementTypeChange++;
                }
                // Ensure Swimmer nodes are always Neutral type
                if (bp.nodeType === NodeType.SWIMMER) {
                    bp.movementType = MovementType.NEUTRAL;
                }

                // Mutate canBeGrabber gene
                if (Math.random() < config.GRABBER_GENE_MUTATION_CHANCE) {
                    bp.canBeGrabber = !bp.canBeGrabber;
                    config.mutationStats.grabberGeneChange++;
                }

                // Mutate dyeColor
                if (Math.random() < (config.MUTATION_CHANCE_BOOL * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    config.mutationStats.blueprintDyeColorChange = (config.mutationStats.blueprintDyeColorChange || 0) + 1;
                }

                // Mutate neuronDataBlueprint (specifically hiddenLayerSize if neuron)
                if (bp.nodeType === NodeType.NEURON) {
                    if (!bp.neuronDataBlueprint) { // Ensure it exists
                        bp.neuronDataBlueprint = { hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) };
                    }
                    if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                        let newSize = bp.neuronDataBlueprint.hiddenLayerSize + Math.floor((Math.random() * 6) - 3); // Mutate by +/- up to 3
                        bp.neuronDataBlueprint.hiddenLayerSize = Math.max(config.DEFAULT_HIDDEN_LAYER_SIZE_MIN, Math.min(newSize, config.DEFAULT_HIDDEN_LAYER_SIZE_MAX));
                        config.mutationStats.blueprintNeuronHiddenSizeChange = (config.mutationStats.blueprintNeuronHiddenSizeChange || 0) + 1;
                    }
                } else {
                    bp.neuronDataBlueprint = null; // Crucial: ensure non-neurons have null neuronDataBlueprint
                }

                // Mutate eyeTargetType if it's an EYE node
                if (bp.nodeType === NodeType.EYE && bp.eyeTargetType !== undefined && Math.random() < config.EYE_TARGET_TYPE_MUTATION_CHANCE) {
                    const oldEyeTargetType = bp.eyeTargetType;
                    bp.eyeTargetType = (bp.eyeTargetType === EyeTargetType.PARTICLE) ? EyeTargetType.FOREIGN_BODY_POINT : EyeTargetType.PARTICLE;
                    if (bp.eyeTargetType !== oldEyeTargetType) {
                        config.mutationStats.eyeTargetTypeChange = (config.mutationStats.eyeTargetTypeChange || 0) + 1;
                    }
                }
            });

            // 3. Mutate blueprint springs (restLength, isRigid)
            this.blueprintSprings.forEach(bs => {
                if (Math.random() < (config.SPRING_PROP_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER)) { // Use magnitude as chance here
                    const oldRestLength = bs.restLength;
                    bs.restLength = Math.max(1, bs.restLength * (1 + (Math.random() - 0.5) * 2 * config.SPRING_PROP_MUTATION_MAGNITUDE));
                    if (Math.abs(bs.restLength - oldRestLength) > 0.01) config.mutationStats.springRestLength++;
                }
                if (Math.random() < (config.MUTATION_CHANCE_BOOL * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    const oldRigid = bs.isRigid;
                    bs.isRigid = !bs.isRigid; // Simple flip for now
                    if (bs.isRigid !== oldRigid) config.mutationStats.springRigidityFlip++;
                }
                 if (Math.random() < (config.SPRING_PROP_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    if (typeof bs.stiffness === 'undefined') bs.stiffness = this.stiffness;
                    const oldStiffness = bs.stiffness;
                    bs.stiffness = Math.max(100, Math.min(bs.stiffness * (1 + (Math.random() - 0.5) * 2 * config.SPRING_PROP_MUTATION_MAGNITUDE), 10000));
                    if (Math.abs(bs.stiffness - oldStiffness) > 0.01) config.mutationStats.springStiffness++;
                }
                if (Math.random() < (config.SPRING_PROP_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    if (typeof bs.damping === 'undefined') bs.damping = this.springDamping;
                    const oldDamping = bs.damping;
                    bs.damping = Math.max(0.1, Math.min(bs.damping * (1 + (Math.random() - 0.5) * 2 * config.SPRING_PROP_MUTATION_MAGNITUDE), 50));
                    if (Math.abs(bs.damping - oldDamping) > 0.01) config.mutationStats.springDamping++;
                }
            });

            // Step 4: Structural Blueprint Mutations (Point Add, Spring Add/Delete, Subdivision, Scale, etc.)
            // --- Point Addition Mutation (Blueprint) ---
            if (Math.random() < this.pointAddChance * config.GLOBAL_MUTATION_RATE_MODIFIER && this.blueprintPoints.length > 0) {
                const lastBp = this.blueprintPoints[this.blueprintPoints.length - 1];
                const newRelX = lastBp.relX + (Math.random() - 0.5) * config.NEW_POINT_OFFSET_RADIUS * 0.5; // Smaller offset for blueprint
                const newRelY = lastBp.relY + (Math.random() - 0.5) * config.NEW_POINT_OFFSET_RADIUS * 0.5;
                const newMass = 0.1 + Math.random() * 0.9;
                const newRadius = baseRadius * (0.8 + Math.random() * 0.4);
                
                let newNodeType;
                if (Math.random() < config.NEURON_CHANCE) {
                    newNodeType = NodeType.NEURON;
                } else {
                    const otherNodeTypes = availableFunctionalNodeTypes.filter(t => t !== NodeType.NEURON);
                    newNodeType = otherNodeTypes[Math.floor(Math.random() * otherNodeTypes.length)];
                }

                const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                let newMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                if (newNodeType === NodeType.SWIMMER) {
                    newMovementType = MovementType.NEUTRAL;
                }
                const newBp = {
                    relX: newRelX, relY: newRelY, radius: newRadius, mass: newMass,
                    nodeType: newNodeType, movementType: newMovementType,
                    dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                    canBeGrabber: Math.random() < config.GRABBER_GENE_MUTATION_CHANCE,
                    neuronDataBlueprint: newNodeType === NodeType.NEURON ? { hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) } : null,
                    eyeTargetType: newNodeType === NodeType.EYE ? (Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT) : undefined,
                    maxEffectiveJetVelocity: this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4)
                };
                this.blueprintPoints.push(newBp);
                const newPointIndex = this.blueprintPoints.length - 1;
                config.mutationStats.pointAddActual++;

                // Connect new blueprint point with springs
                const numSpringsToAddNewPoint = config.MIN_SPRINGS_PER_NEW_NODE + Math.floor(Math.random() * (config.MAX_SPRINGS_PER_NEW_NODE - config.MIN_SPRINGS_PER_NEW_NODE + 1));
                const existingBpIndices = this.blueprintPoints.map((_, i) => i).filter(i => i !== newPointIndex);
                const shuffledExistingBpIndices = existingBpIndices.sort(() => 0.5 - Math.random());

                for (let k = 0; k < Math.min(numSpringsToAddNewPoint, shuffledExistingBpIndices.length); k++) {
                    const connectToBpIndex = shuffledExistingBpIndices[k];
                    const connectToBp = this.blueprintPoints[connectToBpIndex];
                    const dist = Math.sqrt((newBp.relX - connectToBp.relX)**2 + (newBp.relY - connectToBp.relY)**2);
                    let newRestLength = dist * (1 + (Math.random() - 0.5) * 2 * config.NEW_SPRING_REST_LENGTH_VARIATION);
                    newRestLength = Math.max(1, newRestLength);
                    const becomeRigid = Math.random() < config.CHANCE_FOR_RIGID_SPRING;
                    const newStiffness = 500 + Math.random() * 2500;
                    const newDamping = 5 + Math.random() * 20;
                    this.blueprintSprings.push({ p1Index: newPointIndex, p2Index: connectToBpIndex, restLength: newRestLength, isRigid: becomeRigid, stiffness: newStiffness, damping: newDamping });
                }
            }

            // --- Spring Deletion Mutation (Blueprint) ---
            if (this.blueprintSprings.length > this.blueprintPoints.length -1 && Math.random() < config.SPRING_DELETION_CHANCE) { // Ensure min connectivity
            // Ensure min connectivity
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
                        const becomeRigid = Math.random() < config.CHANCE_FOR_RIGID_SPRING;
                        const newStiffness = 500 + Math.random() * 2500;
                        const newDamping = 5 + Math.random() * 20;
                        this.blueprintSprings.push({ p1Index: idx1, p2Index: idx2, restLength: newRestLength, isRigid: becomeRigid, stiffness: newStiffness, damping: newDamping });
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
                    if (newNodeType === NodeType.SWIMMER) {
                        newMovementType = MovementType.NEUTRAL;
                    }

                    const newMidBp = {
                        relX: midRelX, relY: midRelY, radius: Math.max(0.5, newRadius), mass: Math.max(0.1, newMass),
                        nodeType: newNodeType, movementType: newMovementType,
                        dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                        canBeGrabber: Math.random() < GRABBER_GENE_MUTATION_CHANCE,
                        neuronDataBlueprint: newNodeType === NodeType.NEURON ? { hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) } : null,
                        eyeTargetType: newNodeType === NodeType.EYE ? (Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT) : undefined,
                        maxEffectiveJetVelocity: this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4)
                    };
                    this.blueprintPoints.push(newMidBp);
                    const newMidPointIndex = this.blueprintPoints.length - 1;

                    this.blueprintSprings.splice(springToSubdivideIndex, 1); // Remove original spring

                    let restLength1 = Math.sqrt((bp1.relX - midRelX)**2 + (bp1.relY - midRelY)**2) * (1 + (Math.random() - 0.5) * 0.1);
                    this.blueprintSprings.push({ p1Index: originalBs.p1Index, p2Index: newMidPointIndex, restLength: Math.max(1, restLength1), isRigid: originalBs.isRigid, stiffness: originalBs.stiffness, damping: originalBs.damping });

                    let restLength2 = Math.sqrt((midRelX - bp2.relX)**2 + (midRelY - bp2.relY)**2) * (1 + (Math.random() - 0.5) * 0.1);
                    this.blueprintSprings.push({ p1Index: newMidPointIndex, p2Index: originalBs.p2Index, restLength: Math.max(1, restLength2), isRigid: originalBs.isRigid, stiffness: originalBs.stiffness, damping: originalBs.damping });
                    
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

            // --- Shape Addition Mutation (Blueprint) ---
            if (this.blueprintPoints.length >= 2 && Math.random() < 0.02) { // 2% chance
                const blueprintPointsForHull = this.blueprintPoints.map(bp => ({ x: bp.relX, y: bp.relY, originalBlueprintPoint: bp }));
                if (blueprintPointsForHull.length >= 3) {
                    const hullPointsWithOriginal = convexHull(blueprintPointsForHull);
                    if (hullPointsWithOriginal && hullPointsWithOriginal.length >= 2) {
                        const hullIndices = hullPointsWithOriginal.map(p => this.blueprintPoints.indexOf(p.originalBlueprintPoint));

                        let externalEdge = null;
                        let p1_hull_idx, p2_hull_idx;
                        let attempts = 0;
                        let foundValidEdge = false;

                        while (attempts < hullIndices.length * 2 && !foundValidEdge) {
                            let randHullStartIdx = Math.floor(Math.random() * hullIndices.length);
                            p1_hull_idx = hullIndices[randHullStartIdx];
                            p2_hull_idx = hullIndices[(randHullStartIdx + 1) % hullIndices.length];

                            if (this.blueprintSprings.some(bs => (bs.p1Index === p1_hull_idx && bs.p2Index === p2_hull_idx) || (bs.p1Index === p2_hull_idx && bs.p2Index === p1_hull_idx))) {
                                foundValidEdge = true;
                                externalEdge = {
                                    p1: this.blueprintPoints[p1_hull_idx],
                                    p2: this.blueprintPoints[p2_hull_idx],
                                };
                            }
                            attempts++;
                        }

                        if (externalEdge) {
                            const P1_bp = externalEdge.p1;
                            const P2_bp = externalEdge.p2;
                            const p1_idx = this.blueprintPoints.indexOf(P1_bp);
                            const p2_idx = this.blueprintPoints.indexOf(P2_bp);
                            const P1 = new Vec2(P1_bp.relX, P1_bp.relY);
                            const P2 = new Vec2(P2_bp.relX, P2_bp.relY);
                            const sideVector = P2.sub(P1);
                            const sideLength = sideVector.mag();

                            if (sideLength > 1) {
                                let normal = new Vec2(-sideVector.y, sideVector.x).normalize();
                                const midPoint = P1.add(sideVector.mul(0.5));
                                if (Vec2.dot(normal, midPoint) > 0) {
                                    normal.mulInPlace(-1);
                                }

                                const newPoints = [];
                                const shapeType = Math.random() < 0.5 ? 'square' : 'pentagon';
                                let newSpringsInfo = [];

                                if (shapeType === 'square') {
                                    const P4 = P1.add(normal.mul(sideLength));
                                    const P3 = P2.add(normal.mul(sideLength));
                                    newPoints.push(P4, P3);
                                    newSpringsInfo = [
                                        { from: p2_idx, toNew: 1, len: sideLength },
                                        { new1: 1, toNew: 0, len: sideLength },
                                        { new1: 0, to: p1_idx, len: sideLength },
                                        { from: p1_idx, toNew: 1, len: sideLength * Math.SQRT2 }
                                    ];
                                } else { // Pentagon
                                    const angle = -72 * Math.PI / 180;
                                    const P1_minus_P2 = P1.sub(P2);
                                    let P3_vec = new Vec2(P1_minus_P2.x * Math.cos(angle) - P1_minus_P2.y * Math.sin(angle), P1_minus_P2.x * Math.sin(angle) + P1_minus_P2.y * Math.cos(angle));
                                    const P3 = P2.add(P3_vec);

                                    const P2_minus_P3 = P2.sub(P3);
                                    let P4_vec = new Vec2(P2_minus_P3.x * Math.cos(angle) - P2_minus_P3.y * Math.sin(angle), P2_minus_P3.x * Math.sin(angle) + P2_minus_P3.y * Math.cos(angle));
                                    const P4 = P3.add(P4_vec);

                                    const P2_minus_P1 = P2.sub(P1);
                                    let P5_vec = new Vec2(P2_minus_P1.x * Math.cos(-angle) - P2_minus_P1.y * Math.sin(-angle), P2_minus_P1.x * Math.sin(-angle) + P2_minus_P1.y * Math.sin(-angle));
                                    const P5 = P1.add(P5_vec);
                                    
                                    newPoints.push(P3, P4, P5);
                                    newSpringsInfo = [
                                        { from: p2_idx, toNew: 0, len: sideLength },
                                        { new1: 0, toNew: 1, len: sideLength },
                                        { new1: 1, toNew: 2, len: sideLength },
                                        { new1: 2, to: p1_idx, len: sideLength },
                                        { from: p1_idx, toNew: 0, len: P3.sub(P1).mag() },
                                        { from: p2_idx, toNew: 2, len: P5.sub(P2).mag() }
                                    ];
                                }
                                
                                const firstNewPointIndex = this.blueprintPoints.length;
                                newPoints.forEach(p_vec => {
                                    this.blueprintPoints.push({
                                        relX: p_vec.x, relY: p_vec.y,
                                        radius: (P1_bp.radius + P2_bp.radius) / 2,
                                        mass: (P1_bp.mass + P2_bp.mass) / 2,
                                        nodeType: P1_bp.nodeType, movementType: P1_bp.movementType,
                                        dyeColor: P1_bp.dyeColor, canBeGrabber: P1_bp.canBeGrabber,
                                        neuronDataBlueprint: null
                                    });
                                });

                                newSpringsInfo.forEach(info => {
                                    const p1 = info.from !== undefined ? info.from : firstNewPointIndex + info.new1;
                                    const p2 = info.to !== undefined ? info.to : firstNewPointIndex + info.toNew;
                                    this.blueprintSprings.push({ p1Index: p1, p2Index: p2, restLength: info.len, isRigid: false, stiffness: this.stiffness, damping: this.springDamping });
                                });
                                
                                mutationStats.shapeAddition++;
                            }
                        }
                    }
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
                for (let i=0; i<numPointsY; i++) for (let j=0; j<numPointsX-1; j++) initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i][j+1], 500 + Math.random() * 2500, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                for (let j=0; j<numPointsX; j++) for (let i=0; i<numPointsY-1; i++) initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i+1][j], 500 + Math.random() * 2500, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                for (let i=0; i<numPointsY-1; i++) for (let j=0; j<numPointsX-1; j++) {
                    initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i+1][j+1], (500 + Math.random() * 2500)*0.7, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                    initialTempSprings.push(new Spring(gridPoints[i+1][j], gridPoints[i][j+1], (500 + Math.random() * 2500)*0.7, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                }
            } else if (this.shapeType === 1) { // Line
                const numLinePoints = Math.floor(3 + Math.random() * 3); const isHorizontal = Math.random() < 0.5; let linePoints = [];
                for (let i=0; i<numLinePoints; i++) {
                    const x = (isHorizontal ? i * basePointDist : 0);
                    const y = (isHorizontal ? 0 : i * basePointDist);
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    initialTempMassPoints.push(point); linePoints.push(point);
                }
                for (let i=0; i<numLinePoints-1; i++) initialTempSprings.push(new Spring(linePoints[i], linePoints[i+1], 500 + Math.random() * 2500, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                if (numLinePoints > 2) initialTempSprings.push(new Spring(linePoints[0], linePoints[numLinePoints-1], (500 + Math.random() * 2500)*0.5, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
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
                    initialTempSprings.push(new Spring(centralPoint, point, 500 + Math.random() * 2500, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                    if (i>0) initialTempSprings.push(new Spring(initialTempMassPoints[initialTempMassPoints.length-2], point, (500 + Math.random() * 2500)*0.8, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                }
                if (numOuterPoints > 1) initialTempSprings.push(new Spring(initialTempMassPoints[1], initialTempMassPoints[initialTempMassPoints.length-1], (500 + Math.random() * 2500)*0.8, 5 + Math.random() * 20, null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
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
                let chosenNodeType;
                if (Math.random() < config.NEURON_CHANCE) {
                    chosenNodeType = NodeType.NEURON;
                } else {
                    const otherNodeTypes = availableFunctionalNodeTypes.filter(t => t !== NodeType.NEURON);
                    chosenNodeType = otherNodeTypes[Math.floor(Math.random() * otherNodeTypes.length)];
                }

                const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                let chosenMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                if (chosenNodeType === NodeType.SWIMMER) {
                    chosenMovementType = MovementType.NEUTRAL;
                }
                const canBeGrabberInitial = Math.random() < config.GRABBER_GENE_MUTATION_CHANCE;
                const neuronDataBp = chosenNodeType === NodeType.NEURON ? {
                    hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
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
                    neuronDataBlueprint: neuronDataBp,
                    eyeTargetType: chosenNodeType === NodeType.EYE ? (Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT) : undefined,
                    maxEffectiveJetVelocity: this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4)
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
                        isRigid: s_temp.isRigid,
                        stiffness: s_temp.stiffness,
                        damping: s_temp.dampingFactor
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
        this._calculateBlueprintRadius(); // Calculate after all blueprint points are finalized
    }

    _calculateBlueprintRadius() {
        if (!this.blueprintPoints || this.blueprintPoints.length === 0) {
            this.blueprintRadius = 5; // Default small radius if no blueprint points
            return;
        }
        let maxDistSq = 0;
        this.blueprintPoints.forEach(bp => {
            // relX, relY are distances from the blueprint's own centroid (0,0)
            const distToPointCenterSq = bp.relX * bp.relX + bp.relY * bp.relY;
            // Add the point's own radius to get distance to its edge
            const effectiveDistToEdge = Math.sqrt(distToPointCenterSq) + (bp.radius || 0);
            if (effectiveDistToEdge * effectiveDistToEdge > maxDistSq) {
                maxDistSq = effectiveDistToEdge * effectiveDistToEdge;
            }
        });
        this.blueprintRadius = Math.sqrt(maxDistSq);
        if (isNaN(this.blueprintRadius) || this.blueprintRadius === 0) {
            // Fallback if calculation results in NaN or 0 (e.g. single point at origin with 0 radius)
            this.blueprintRadius = 5; 
        }
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
            newPoint.eyeTargetType = bp.eyeTargetType === undefined ? EyeTargetType.PARTICLE : bp.eyeTargetType;
            newPoint.maxEffectiveJetVelocity = this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4);

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

                const springStiffness = bs.stiffness === undefined ? this.stiffness : bs.stiffness;
                const springDamping = bs.damping === undefined ? this.springDamping : bs.damping;

                // Use the creature's overall stiffness and damping
                // The blueprint spring carries restLength and isRigid
                this.springs.push(new Spring(p1, p2, springStiffness, springDamping, bs.restLength, bs.isRigid));
            } else {
                console.warn(`Body ${this.id}: Invalid spring blueprint indices ${bs.p1Index}, ${bs.p2Index} for ${this.massPoints.length} points.`);
            }
        }

        // Recalculate things that depend on the final massPoints
        this.calculateCurrentMaxEnergy(); 

        // Calculate and store node counts for brain initialization optimization
        this.numEmitterNodes = 0; // Resetting here before counting
        this.numSwimmerNodes = 0;
        this.numEaterNodes = 0;
        this.numPredatorNodes = 0;
        this.numEyeNodes = 0;
        this.numJetNodes = 0;
        this.numPotentialGrabberNodes = 0;
        this.numAttractorNodes = 0;
        this.numRepulsorNodes = 0;

        for (const p of this.massPoints) { // Changed from forEach to for...of for clarity
            if (p.nodeType === NodeType.EMITTER) this.numEmitterNodes++;
            else if (p.nodeType === NodeType.SWIMMER) this.numSwimmerNodes++;
            else if (p.nodeType === NodeType.EATER) this.numEaterNodes++;
            else if (p.nodeType === NodeType.PREDATOR) this.numPredatorNodes++;
            else if (p.nodeType === NodeType.EYE) this.numEyeNodes++;
            else if (p.nodeType === NodeType.JET) this.numJetNodes++;
            else if (p.nodeType === NodeType.ATTRACTOR) this.numAttractorNodes++;
            else if (p.nodeType === NodeType.REPULSOR) this.numRepulsorNodes++;

            if (p.canBeGrabber) this.numPotentialGrabberNodes++;
        }

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
        this._updateJetAndSwimmerFluidSensor(fluidFieldRef);
    }

    _applyDefaultActivationPatterns() {
        this.massPoints.forEach(point => {
            // If the point is an EATER or PREDATOR, its exertion is controlled by the brain or defaults to 0.
            // No default pattern-based exertion for these types.
            if (point.nodeType === NodeType.EATER || point.nodeType === NodeType.PREDATOR) {
                // If not controlled by a brain, their exertion will naturally be 0 from initialization or previous brain step.
                // If a brain *is* controlling it, the brain's output will override this later in _applyBrainActionsToPoints.
                // For clarity, we can ensure it's 0 here if no brain is influencing it yet for this frame.
                // However, the main logic for brain control is separate. This ensures no *default pattern* applies.
                point.currentExertionLevel = 0; 
                return; // Skip pattern-based activation for these types
            }

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
                point.seesTarget = false; // Reset first
                point.nearestTargetMagnitude = 0;
                point.nearestTargetDirection = 0;
                let closestDistSq = EYE_DETECTION_RADIUS * EYE_DETECTION_RADIUS;

                // Determine the grid cell range to check based on EYE_DETECTION_RADIUS
                const eyeGxMin = Math.max(0, Math.floor((point.pos.x - EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));
                const eyeGxMax = Math.min(GRID_COLS - 1, Math.floor((point.pos.x + EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));
                const eyeGyMin = Math.max(0, Math.floor((point.pos.y - EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));
                const eyeGyMax = Math.min(GRID_ROWS - 1, Math.floor((point.pos.y + EYE_DETECTION_RADIUS) / GRID_CELL_SIZE));

                if (point.eyeTargetType === EyeTargetType.PARTICLE) {
                    let nearestParticleFound = null;
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
                        point.seesTarget = true;
                        const vecToTarget = nearestParticleFound.pos.sub(point.pos);
                        point.nearestTargetMagnitude = vecToTarget.mag() / EYE_DETECTION_RADIUS; 
                        point.nearestTargetDirection = Math.atan2(vecToTarget.y, vecToTarget.x);
                    }
                } else if (point.eyeTargetType === EyeTargetType.FOREIGN_BODY_POINT) {
                    let nearestForeignPointFound = null;
                    for (let gy = eyeGyMin; gy <= eyeGyMax; gy++) {
                        for (let gx = eyeGxMin; gx <= eyeGxMax; gx++) {
                            const cellIndex = gx + gy * GRID_COLS;
                            if (spatialGrid[cellIndex] && spatialGrid[cellIndex].length > 0) {
                                const cellBucket = spatialGrid[cellIndex];
                                for (const item of cellBucket) {
                                    // Check for softbody_point, ensure it's not from the current body, and body is not unstable
                                    if (item.type === 'softbody_point' && item.bodyRef !== this && !item.bodyRef.isUnstable) {
                                        const foreignPoint = item.pointRef;
                                        const distSq = point.pos.sub(foreignPoint.pos).magSq();
                                        if (distSq < closestDistSq) {
                                            closestDistSq = distSq;
                                            nearestForeignPointFound = foreignPoint;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nearestForeignPointFound) {
                        point.seesTarget = true;
                        const vecToTarget = nearestForeignPointFound.pos.sub(point.pos);
                        point.nearestTargetMagnitude = vecToTarget.mag() / EYE_DETECTION_RADIUS;
                        point.nearestTargetDirection = Math.atan2(vecToTarget.y, vecToTarget.x);
                    }
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
        // console.log("Body", this.id, "_updateEnergyBudget called. fluidFieldRef type:", fluidFieldRef ? fluidFieldRef.constructor.name : 'undefined');
        let currentFrameEnergyCost = 0;
        let currentFrameEnergyGain = 0;
        let poisonDamageThisFrame = 0; // Initialize here
        this.energyGainedFromPhotosynthesisThisTick = 0; // Reset for the current tick

        const hasFluidField = fluidFieldRef !== null && typeof fluidFieldRef !== 'undefined';
        const hasNutrientField = nutrientField !== null && typeof nutrientField !== 'undefined';
        const hasLightField = lightField !== null && typeof lightField !== 'undefined';

        const scaleX = hasFluidField ? fluidFieldRef.scaleX : 0;
        const scaleY = hasFluidField ? fluidFieldRef.scaleY : 0;

        for (const point of this.massPoints) {
            // --- Red Dye Poison Effect (Moved inside main loop) ---
            if (hasFluidField && RED_DYE_POISON_STRENGTH > 0) {
                // const fluidGridX = Math.floor(point.pos.x / scaleX); // gx already calculated below
                // const fluidGridY = Math.floor(point.pos.y / scaleY); // gy already calculated below
                // let tempMapIdxForPoison = fluidFieldRef.IX(fluidGridX, fluidGridY); // mapIdx used below is the same
                // const redDensity = (fluidFieldRef.densityR[tempMapIdxForPoison] || 0) / 255;
                // Calculation will use mapIdx determined below if hasFluidField
            }
            // --- End of Moved Red Dye Poison Effect ---

            let costMultiplier = 1.0;
            let mapIdx = -1;

            if (hasFluidField) {
                const gx = Math.floor(point.pos.x / scaleX);
                const gy = Math.floor(point.pos.y / scaleY);
                mapIdx = fluidFieldRef.IX(gx, gy);

                // --- Red Dye Poison Calculation (using mapIdx) ---
                if (RED_DYE_POISON_STRENGTH > 0) {
                    const redDensity = (fluidFieldRef.densityR[mapIdx] || 0) / 255;
                    if (redDensity > 0.01) {
                        poisonDamageThisFrame += redDensity * RED_DYE_POISON_STRENGTH * (point.radius / 5);
                    }
                }
                // --- End of Red Dye Poison Calculation ---

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
                case NodeType.JET:
                    const jetCostThisFrame = JET_NODE_ENERGY_COST * exertionSq * costMultiplier;
                    currentFrameEnergyCost += jetCostThisFrame;
                    this.energyCostFromJetNodes += jetCostThisFrame * dt;
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
                    this.energyGainedFromPhotosynthesis += energyGainThisPoint; // Lifetime total
                    this.energyGainedFromPhotosynthesisThisTick += energyGainThisPoint; // Current tick total
                }
                    break;
                case NodeType.ATTRACTOR:
                    const attractorCostThisFrame = ATTRACTOR_NODE_ENERGY_COST * costMultiplier;
                    currentFrameEnergyCost += attractorCostThisFrame;
                    this.energyCostFromAttractorNodes += attractorCostThisFrame * dt;
                    break;
                case NodeType.REPULSOR:
                    const repulsorCostThisFrame = REPULSOR_NODE_ENERGY_COST * costMultiplier;
                    currentFrameEnergyCost += repulsorCostThisFrame;
                    this.energyCostFromRepulsorNodes += repulsorCostThisFrame * dt;
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
        
        // Apply poison damage after the loop, before other adjustments
        if (poisonDamageThisFrame > 0) {
            this.creatureEnergy -= poisonDamageThisFrame * dt * 60; // dt is in seconds, scale strength to be per-second
        }

        this.creatureEnergy += currentFrameEnergyGain; // Gains are already dt-scaled
        this.creatureEnergy -= currentFrameEnergyCost * dt; // Costs are per-frame, so scale by dt here
        this.creatureEnergy = Math.min(this.currentMaxEnergy, Math.max(0, this.creatureEnergy));

        if (this.creatureEnergy <= 0) {
            this.isUnstable = true;
        }
    }

    _performPhysicalUpdates(dt, fluidFieldRef) {
        if (fluidFieldRef) {
            for (let point of this.massPoints) {
                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);

                if (!isFinite(fluidGridX) || !isFinite(fluidGridY)) continue;
                
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);
                // Check index validity once, for all interactions.
                // Assuming Vx, Vy, density arrays all have the same length.
                if (idx < 0 || idx >= (fluidFieldRef.Vx ? fluidFieldRef.Vx.length : (fluidFieldRef.textures.velocityPing ? fluidFieldRef.size * fluidFieldRef.size : 0))) continue;


                // --- Fluid Interactions (should occur even if fixed) ---
                if (point.nodeType === NodeType.EMITTER) {
                    let dyeEmissionStrength = 50 * point.currentExertionLevel;
                    fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], dyeEmissionStrength);
                }

                if (point.nodeType === NodeType.JET) {
                    const exertion = point.currentExertionLevel || 0;
                    if (exertion > 0.01) {
                        const currentFluidVelX = fluidFieldRef.Vx ? fluidFieldRef.Vx[idx] : 0; // Placeholder for GPU
                        const currentFluidVelY = fluidFieldRef.Vy ? fluidFieldRef.Vy[idx] : 0; // Placeholder for GPU
                        const currentFluidSpeedSq = currentFluidVelX ** 2 + currentFluidVelY ** 2;

                        if (currentFluidSpeedSq < point.maxEffectiveJetVelocity ** 2) {
                            const finalMagnitude = point.jetData.currentMagnitude;
                            const angle = point.jetData.currentAngle;
                            const appliedForceX = finalMagnitude * Math.cos(angle);
                            const appliedForceY = finalMagnitude * Math.sin(angle);
                            fluidFieldRef.addVelocity(fluidGridX, fluidGridY, appliedForceX, appliedForceY);
                        }
                    }
                }
                
                // --- Physics Updates for Mobile Points Only ---
                if (point.isFixed) continue;

                if (point.movementType === MovementType.FLOATING) {
                    const rawFluidVx = fluidFieldRef.Vx ? fluidFieldRef.Vx[idx] : 0;
                    const rawFluidVy = fluidFieldRef.Vy ? fluidFieldRef.Vy[idx] : 0;
                    this._tempVec1.copyFrom(point.pos).subInPlace(point.prevPos).mulInPlace(1.0 - this.fluidEntrainment);
                    this._tempVec2.x = rawFluidVx * fluidFieldRef.scaleX * dt;
                    this._tempVec2.y = rawFluidVy * fluidFieldRef.scaleY * dt;
                    this._tempVec2.mulInPlace(this.fluidCurrentStrength).mulInPlace(this.fluidEntrainment);
                    this._tempVec1.addInPlace(this._tempVec2);
                    
                    point.prevPos.copyFrom(point.pos).subInPlace(this._tempVec1);
                }
            }
        }

        for (let spring of this.springs) {
            spring.applyForce();
        }

        for (let point of this.massPoints) {
            point.update(dt);
        }

        const MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ = (MAX_PIXELS_PER_FRAME_DISPLACEMENT) ** 2;
        for (let point of this.massPoints) {
            if (point.isFixed) continue;

            const displacementSq = (point.pos.x - point.prevPos.x) ** 2 + (point.pos.y - point.prevPos.y) ** 2;
            if (displacementSq > MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ || isNaN(point.pos.x) || isNaN(point.pos.y) || !isFinite(point.pos.x) || !isFinite(point.pos.y)) {
                this.isUnstable = true;
                return;
            }

            if (point.pos.x < 0 || point.pos.x > WORLD_WIDTH || point.pos.y < 0 || point.pos.y > WORLD_HEIGHT) {
                this.isUnstable = true;
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
        this.preyPredatedThisTick = new Set(); // Initialize/clear for this body for this tick
        if (this.isUnstable) return; 

        // Inter-body repulsion & Predation
        // Use this._tempVec1 for diff, and this._tempVec2 for forceDir/repulsionForce
        const tempDiffVec = this._tempVec1; 
        const tempForceVec = this._tempVec2;

        for (let i_p1 = 0; i_p1 < this.massPoints.length; i_p1++) {
            const p1 = this.massPoints[i_p1];

            // Attractor/Repulsor logic: find nearest foreign point and apply force
            if ((p1.nodeType === NodeType.ATTRACTOR || p1.nodeType === NodeType.REPULSOR) && (p1.currentExertionLevel || 0) > 0.01) {
                let nearestForeignPoint = null;
                const isAttractor = p1.nodeType === NodeType.ATTRACTOR;
                const p1Exertion = p1.currentExertionLevel;

                const radiusMultiplierConfig = isAttractor ?
                    { base: ATTRACTION_RADIUS_MULTIPLIER_BASE, bonus: ATTRACTION_RADIUS_MULTIPLIER_MAX_BONUS } :
                    { base: REPULSION_RADIUS_MULTIPLIER_BASE, bonus: REPULSION_RADIUS_MULTIPLIER_MAX_BONUS };
                const radiusMultiplier = radiusMultiplierConfig.base + (radiusMultiplierConfig.bonus * p1Exertion);
                const interactionRadius = p1.radius * radiusMultiplier;
                let min_dist_sq = interactionRadius * interactionRadius; // Start search radius at max interaction radius

                const p1Gx_force = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(p1.pos.x / GRID_CELL_SIZE)));
                const p1Gy_force = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(p1.pos.y / GRID_CELL_SIZE)));
                const searchRadiusInCells = Math.ceil(interactionRadius / GRID_CELL_SIZE);

                for (let dy = -searchRadiusInCells; dy <= searchRadiusInCells; dy++) {
                    for (let dx = -searchRadiusInCells; dx <= searchRadiusInCells; dx++) {
                        const checkGx = p1Gx_force + dx;
                        const checkGy = p1Gy_force + dy;
                        if (checkGx >= 0 && checkGx < GRID_COLS && checkGy >= 0 && checkGy < GRID_ROWS) {
                            const cellIndex = checkGx + checkGy * GRID_COLS;
                            if (Array.isArray(spatialGrid[cellIndex])) {
                                const cellBucket = spatialGrid[cellIndex];
                                for (const otherItem of cellBucket) {
                                    if (otherItem.type === 'softbody_point' && otherItem.bodyRef !== this && !otherItem.bodyRef.isUnstable) {
                                        const p2_candidate = otherItem.pointRef;
                                        tempDiffVec.copyFrom(p1.pos).subInPlace(p2_candidate.pos);
                                        const distSq = tempDiffVec.magSq();
                                        if (distSq < min_dist_sq) {
                                            min_dist_sq = distSq;
                                            nearestForeignPoint = p2_candidate;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (nearestForeignPoint) {
                    const maxForce = isAttractor ? ATTRACTOR_MAX_FORCE : REPULSOR_MAX_FORCE;
                    const forceMagnitude = p1Exertion * maxForce * (1.0 - (Math.sqrt(min_dist_sq) / interactionRadius));
                    const finalForceMagnitude = isAttractor ? -forceMagnitude : forceMagnitude;

                    const forceDir = this._tempVec2.copyFrom(p1.pos).subInPlace(nearestForeignPoint.pos).normalizeInPlace();
                    const forceOnP1 = forceDir.mulInPlace(finalForceMagnitude);

                    if (!p1.isFixed) {
                        p1.applyForce(forceOnP1);
                    }
                    if (!nearestForeignPoint.isFixed) {
                        nearestForeignPoint.applyForce(forceOnP1.clone().mulInPlace(-1));
                    }
                }
            }


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

                                    // const diff = p1.pos.sub(p2.pos);
                                    tempDiffVec.copyFrom(p1.pos).subInPlace(p2.pos);
                                    const distSq = tempDiffVec.magSq();
                                    const interactionRadius = (p1.radius + p2.radius) * BODY_REPULSION_RADIUS_FACTOR;

                                    if (distSq < interactionRadius * interactionRadius && distSq > 0.0001) {
                                        const dist = Math.sqrt(distSq);
                                        const overlap = interactionRadius - dist;
                                        // const forceDir = diff.normalize();
                                        // const repulsionForce = forceDir.mul(repulsionForceMag);
                                        // p1.applyForce(repulsionForce);
                                        
                                        tempForceVec.copyFrom(tempDiffVec).normalizeInPlace();
                                        const repulsionForceMag = BODY_REPULSION_STRENGTH * overlap * 0.5;
                                        tempForceVec.mulInPlace(repulsionForceMag);
                                        p1.applyForce(tempForceVec);
                                    }

                                    if (p1.nodeType === NodeType.PREDATOR) {
                                        const p1Exertion = p1.currentExertionLevel || 0;
                                        const effectivePredationRadiusMultiplier = PREDATION_RADIUS_MULTIPLIER_BASE + (PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * p1Exertion);
                                        const predationRadius = p1.radius * effectivePredationRadiusMultiplier;
                                        
                                        if (distSq < predationRadius * predationRadius) {
                                            // NEW CHECK: Has this prey body (otherItem.bodyRef) already been predated by THIS predator (this) this tick?
                                            if (!this.preyPredatedThisTick.has(otherItem.bodyRef.id)) {
                                                const effectiveEnergySapped = ENERGY_SAPPED_PER_PREDATION_BASE + (ENERGY_SAPPED_PER_PREDATION_MAX_BONUS * p1Exertion);
                                                const energyToSap = Math.min(otherItem.bodyRef.creatureEnergy, effectiveEnergySapped); 
                                                if (energyToSap > 0) {
                                                    otherItem.bodyRef.creatureEnergy -= energyToSap;
                                                    this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyToSap); 
                                                    this.energyGainedFromPredation += energyToSap;
                                                    this.preyPredatedThisTick.add(otherItem.bodyRef.id); // Mark this prey as predated for this tick
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

        // Eating Logic
        for (let point of this.massPoints) {
            if (point.isFixed) continue; 
            if (point.nodeType === NodeType.EATER) { 
                const pointExertion = point.currentExertionLevel || 0;
                const effectiveEatingRadiusMultiplier = EATING_RADIUS_MULTIPLIER_BASE + (EATING_RADIUS_MULTIPLIER_MAX_BONUS * pointExertion);
                const eatingRadius = point.radius * effectiveEatingRadiusMultiplier;
                const eatingRadiusSq = eatingRadius * eatingRadius;

                // Determine the grid cell range to check based on eatingRadius
                const eaterGxMin = Math.max(0, Math.floor((point.pos.x - eatingRadius) / GRID_CELL_SIZE));
                const eaterGxMax = Math.min(GRID_COLS - 1, Math.floor((point.pos.x + eatingRadius) / GRID_CELL_SIZE));
                const eaterGyMin = Math.max(0, Math.floor((point.pos.y - eatingRadius) / GRID_CELL_SIZE));
                const eaterGyMax = Math.min(GRID_ROWS - 1, Math.floor((point.pos.y + eatingRadius) / GRID_CELL_SIZE));

                for (let gy = eaterGyMin; gy <= eaterGyMax; gy++) {
                    for (let gx = eaterGxMin; gx <= eaterGxMax; gx++) {
                        const cellIndex = gx + gy * GRID_COLS;
                        if (Array.isArray(spatialGrid[cellIndex])) {
                            const cellBucket = spatialGrid[cellIndex];
                            for (let k = cellBucket.length - 1; k >= 0; k--) { 
                                const item = cellBucket[k];
                                if (item.type === 'particle') {
                                    const particle = item.particleRef;
                                    if (particle.life > 0 && !particle.isEaten) {
                                        // const distSq = point.pos.sub(particle.pos).magSq();
                                        this._tempVec1.copyFrom(point.pos).subInPlace(particle.pos);
                                        const distSq = this._tempVec1.magSq();
                                        if (distSq < eatingRadiusSq) {
                                            particle.isEaten = true;
                                            particle.life = 0; 
                                            
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
        if (this.isUnstable) return; 

        // Final Instability Checks: Springs and Span
        const localMaxSpringStretchFactor = MAX_SPRING_STRETCH_FACTOR;
        const localMaxSpanPerPointFactor = MAX_SPAN_PER_POINT_FACTOR;

        for (const spring of this.springs) {
            // const currentLength = spring.p1.pos.sub(spring.p2.pos).mag();
            this._tempVec1.copyFrom(spring.p1.pos).subInPlace(spring.p2.pos);
            const currentLength = this._tempVec1.mag();
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

        // Check for max age
        if (this.ticksSinceBirth > MAX_CREATURE_AGE_TICKS) {
            this.isUnstable = true;
            return; // Creature dies of old age
        }
        
        if (this.ticksSinceBirth > this.effectiveReproductionCooldown) { // Use effective cooldown
            this.canReproduce = true;
        }
    }

    getAverageStiffness() {
        if (this.springs.length === 0) return 0;
        const nonRigidSprings = this.springs.filter(s => !s.isRigid);
        if (nonRigidSprings.length === 0) return RIGID_SPRING_STIFFNESS;
        const totalStiffness = nonRigidSprings.reduce((sum, spring) => sum + spring.stiffness, 0);
        return totalStiffness / nonRigidSprings.length;
    }

    getAverageDamping() {
        if (this.springs.length === 0) return 0;
        const nonRigidSprings = this.springs.filter(s => !s.isRigid);
        if (nonRigidSprings.length === 0) return RIGID_SPRING_DAMPING;
        const totalDamping = nonRigidSprings.reduce((sum, spring) => sum + spring.dampingFactor, 0);
        return totalDamping / nonRigidSprings.length;
    }

    _updateJetAndSwimmerFluidSensor(fluidFieldRef) {
        if (!fluidFieldRef) return;
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.JET || point.nodeType === NodeType.SWIMMER) {
                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);
                const fluidVelX = fluidFieldRef.Vx[idx];
                const fluidVelY = fluidFieldRef.Vy[idx];
                point.sensedFluidVelocity.x = fluidVelX;
                point.sensedFluidVelocity.y = fluidVelY;
            }
        });
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
        if (this.failedReproductionCooldown > 0) {
            this.failedReproductionCooldown--;
            return []; // On cooldown from a previous failed attempt
        }

        if (this.isUnstable || !this.canReproduce || !canCreaturesReproduceGlobally) return []; // Check global flag

        const energyForOneOffspring = this.currentMaxEnergy * OFFSPRING_INITIAL_ENERGY_SHARE; // Use currentMaxEnergy for cost basis
        let hadEnoughEnergyForAttempt = this.creatureEnergy >= energyForOneOffspring;

        let successfullyPlacedOffspring = 0;
        let offspring = [];

        // Pre-calculate spatial info for existing bodies to optimize collision checks
        const existingBodiesSpatialInfo = [];
        for (const body of softBodyPopulation) {
            if (body !== this && !body.isUnstable) { // Don't include self or unstable bodies
                existingBodiesSpatialInfo.push({
                    center: body.getAveragePosition(),
                    radius: body.blueprintRadius
                });
            }
        }

        for (let i = 0; i < this.numOffspring; i++) {
            if (this.creatureEnergy < energyForOneOffspring) break; // Not enough energy for this one

            let placedThisOffspring = false;
            for (let attempt = 0; attempt < OFFSPRING_PLACEMENT_ATTEMPTS; attempt++) {
                const angle = Math.random() * Math.PI * 2;
                const radiusOffset = this.offspringSpawnRadius * (0.5 + Math.random() * 0.5); // offspringSpawnRadius is a gene
                const offsetX = Math.cos(angle) * radiusOffset;
                const offsetY = Math.sin(angle) * radiusOffset;

                const parentAvgPos = this.getAveragePosition();
                let spawnX = parentAvgPos.x + offsetX;
                let spawnY = parentAvgPos.y + offsetY;

                // Create the potential child. Its blueprintRadius will be calculated in its constructor.
                // We will assign a proper ID only if placement is successful.
                let potentialChild = new SoftBody(-1, spawnX, spawnY, this); // Use -1 or a temporary ID marker
                
                if (potentialChild.massPoints.length === 0 || potentialChild.blueprintRadius === 0) continue; 

                let isSpotClear = true;
                // Check against existing population using blueprintRadius and cached positions
                for (const otherBodyInfo of existingBodiesSpatialInfo) {
                    // if (otherBody.isUnstable || otherBody === this) continue; // Already filtered
                    // const otherBodyCenter = otherBody.getAveragePosition(); // Use cached center
                    const distSq = (spawnX - otherBodyInfo.center.x)**2 + (spawnY - otherBodyInfo.center.y)**2;
                    // Use the sum of blueprint radii plus a clearance value for the check
                    const combinedRadii = potentialChild.blueprintRadius + otherBodyInfo.radius + OFFSPRING_PLACEMENT_CLEARANCE_RADIUS;
                    if (distSq < combinedRadii * combinedRadii) {
                        isSpotClear = false; 
                        break;
                    }
                }
                // Check against already spawned new offspring in this cycle
                if (isSpotClear) {
                    for (const newBorn of offspring) { // offspring contains fully created children
                        const newBornCenter = newBorn.getAveragePosition();
                        const distSq = (spawnX - newBornCenter.x)**2 + (spawnY - newBornCenter.y)**2;
                        const combinedRadii = potentialChild.blueprintRadius + newBorn.blueprintRadius + OFFSPRING_PLACEMENT_CLEARANCE_RADIUS;
                        if (distSq < combinedRadii * combinedRadii) {
                            isSpotClear = false; 
                            break;
                        }
                    }
                }

                if (isSpotClear) {
                    this.creatureEnergy -= energyForOneOffspring;
                    // Use the already constructed tempChild's points, but create a new SoftBody instance with proper ID and translated points.
                    // const finalChild = new SoftBody(nextSoftBodyId++, spawnX, spawnY, this);

                    // finalChild.creatureEnergy = energyForOneOffspring; // Set energy for the actual child
                    // offspring.push(finalChild);

                    // Optimization: tempChild becomes the finalChild
                    potentialChild.id = nextSoftBodyId++; // Assign final ID and increment global counter
                    potentialChild.creatureEnergy = energyForOneOffspring;
                    offspring.push(potentialChild);

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
        } else if (hadEnoughEnergyForAttempt && successfullyPlacedOffspring === 0) {
            // If had enough energy but couldn't place any offspring (e.g., due to space)
            this.failedReproductionCooldown = FAILED_REPRODUCTION_COOLDOWN_TICKS;
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
        const brainNode = this._findOrCreateBrainNode();

        if (brainNode && brainNode.neuronData && brainNode.neuronData.isBrain) {
            this._calculateBrainVectorSizes(brainNode);
            this._initializeBrainWeightsAndBiases(brainNode);
            this._initializeBrainRLComponents(brainNode);
        } else {
            // No brain node found or brainNode.neuronData is missing somehow
            // console.warn(`Body ${this.id} initializeBrain: No suitable brain node found or neuronData missing.`);
        }
    }

    _findOrCreateBrainNode() {
        let brainNode = null;
        // First, try to find an already designated brain
        for (const point of this.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        // If no brain was pre-designated, find the first NEURON type node
        if (!brainNode) {
            for (const point of this.massPoints) {
                if (point.nodeType === NodeType.NEURON) {
                    if (!point.neuronData) { // Ensure neuronData exists
                        point.neuronData = { 
                            hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                        };
                    }
                    point.neuronData.isBrain = true;
                    brainNode = point; 
                    // Ensure other neurons are not brains
                    this.massPoints.forEach(otherP => {
                        if (otherP !== point && otherP.nodeType === NodeType.NEURON && otherP.neuronData) {
                            otherP.neuronData.isBrain = false;
                        }
                    });
                    break; 
                }
            }
        }
        return brainNode;
    }

    _calculateBrainVectorSizes(brainNode) {
        const nd = brainNode.neuronData;
        const numEmitterPoints = this.numEmitterNodes;
        const numSwimmerPoints = this.numSwimmerNodes;
        const numEaterPoints = this.numEaterNodes;
        const numPredatorPoints = this.numPredatorNodes;
        const numEyeNodes = this.numEyeNodes;
        const numJetNodes = this.numJetNodes;
        const numPotentialGrabberPoints = this.numPotentialGrabberNodes;
        const numAttractorPoints = this.numAttractorNodes;
        const numRepulsorPoints = this.numRepulsorNodes;

        // console.log(`Body ${this.id} _calculateBrainVectorSizes: Using Stored Counts: E:${numEmitterPoints}, S:${numSwimmerPoints}, Ea:${numEaterPoints}, P:${numPredatorPoints}, G:${numPotentialGrabberPoints}, Ey:${numEyeNodes}`);
        nd.inputVectorSize = NEURAL_INPUT_SIZE_BASE +
                             (numEyeNodes * NEURAL_INPUTS_PER_EYE) +
                             (numSwimmerPoints * NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (numJetNodes * NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (this.springs.length * NEURAL_INPUTS_PER_SPRING_SENSOR);
        // console.log(`Body ${this.id} _calculateBrainVectorSizes: Counts (from stored) directly before sum: E:${numEmitterPoints}, S:${numSwimmerPoints}, Ea:${numEaterPoints}, P:${predatorPoints}, G:${numPotentialGrabberPoints}`);
        nd.outputVectorSize = (numEmitterPoints * NEURAL_OUTPUTS_PER_EMITTER) +
                              (numSwimmerPoints * NEURAL_OUTPUTS_PER_SWIMMER) +
                              (numEaterPoints * NEURAL_OUTPUTS_PER_EATER) +
                              (numPredatorPoints * NEURAL_OUTPUTS_PER_PREDATOR) +
                              (numJetNodes * NEURAL_OUTPUTS_PER_JET) +
                              (numPotentialGrabberPoints * NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) +
                              (numAttractorPoints * NEURAL_OUTPUTS_PER_ATTRACTOR) +
                              (numRepulsorPoints * NEURAL_OUTPUTS_PER_REPULSOR);
        // console.log(`Body ${this.id} _calculateBrainVectorSizes: Calculated nd.outputVectorSize = ${nd.outputVectorSize}`);
    }

    _initializeBrainWeightsAndBiases(brainNode) {
        const nd = brainNode.neuronData;
        if (typeof nd.hiddenLayerSize !== 'number' || nd.hiddenLayerSize < config.DEFAULT_HIDDEN_LAYER_SIZE_MIN || nd.hiddenLayerSize > config.DEFAULT_HIDDEN_LAYER_SIZE_MIN) {
            nd.hiddenLayerSize = config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1));
        }

        if (!nd.weightsIH || nd.weightsIH.length !== nd.hiddenLayerSize || (nd.weightsIH.length > 0 && nd.weightsIH[0].length !== nd.inputVectorSize) ) {
            nd.weightsIH = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize);
            nd.biasesH = initializeVector(nd.hiddenLayerSize);
            // console.log(`Body ${this.id} brain: Initialized weightsIH/biasesH. Inputs: ${nd.inputVectorSize}, Hidden: ${nd.hiddenLayerSize}`);
        }
        
        if (!nd.weightsHO || nd.weightsHO.length !== nd.outputVectorSize || (nd.weightsHO.length > 0 && nd.weightsHO[0].length !== nd.hiddenLayerSize) ) {
            nd.weightsHO = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize);
            nd.biasesO = initializeVector(nd.outputVectorSize);
            // console.log(`Body ${this.id} brain: Initialized weightsHO/biasesO. Outputs: ${nd.outputVectorSize}, Hidden: ${nd.hiddenLayerSize}`);
        }
    }

    _initializeBrainRLComponents(brainNode) {
        const nd = brainNode.neuronData;
        if (!nd.experienceBuffer) nd.experienceBuffer = [];
        if (typeof nd.framesSinceLastTrain !== 'number') nd.framesSinceLastTrain = 0; 
        if (typeof nd.previousEnergyForReward !== 'number') nd.previousEnergyForReward = this.creatureEnergy;
        if (typeof nd.previousEnergyChangeForNN !== 'number') nd.previousEnergyChangeForNN = 0; // New: For 2nd derivative input
        if (typeof nd.lastAvgNormalizedReward !== 'number') nd.lastAvgNormalizedReward = 0;
        if (typeof nd.maxExperienceBufferSize !== 'number') nd.maxExperienceBufferSize = 10;
        nd.currentFrameInputVectorWithLabels = []; // New: For UI display
        nd.currentFrameActionDetails = []; // Already exists, just ensure it's initialized
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
                adjacency.get(p2Index).push(p2Index);
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

    exportBlueprint() {
        const blueprint = {
            version: 1,
            stiffness: this.stiffness,
            springDamping: this.springDamping,
            motorImpulseInterval: this.motorImpulseInterval,
            motorImpulseMagnitudeCap: this.motorImpulseMagnitudeCap,
            emitterStrength: this.emitterStrength,
            emitterDirection: { x: this.emitterDirection.x, y: this.emitterDirection.y },
            numOffspring: this.numOffspring,
            offspringSpawnRadius: this.offspringSpawnRadius,
            pointAddChance: this.pointAddChance,
            springConnectionRadius: this.springConnectionRadius,
            reproductionEnergyThreshold: this.reproductionEnergyThreshold,
            jetMaxVelocityGene: this.jetMaxVelocityGene,
            reproductionCooldownGene: this.reproductionCooldownGene,
            defaultActivationPattern: this.defaultActivationPattern,
            defaultActivationLevel: this.defaultActivationLevel,
            defaultActivationPeriod: this.defaultActivationPeriod,
            defaultActivationPhaseOffset: this.defaultActivationPhaseOffset,
            rlAlgorithmType: this.rlAlgorithmType,
            rewardStrategy: this.rewardStrategy,
            blueprintPoints: this.blueprintPoints,
            blueprintSprings: this.blueprintSprings
        };
        return blueprint;
    }
} 