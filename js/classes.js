const NodeType = {
    NEUTRAL: 0,
    FLOATING: 1,
    FIXED_ROOT: 2,
    EMITTER_SWIMMER: 3,
    NEURON: 4,
    PHOTOSYNTHETIC: 5
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
        this.isFixed = false;
        this.nodeType = NodeType.NEUTRAL;
        this.isEater = false;
        this.isPredator = false;
        this.emitsDye = false;
        this.dyeColor = [0,0,0];
        this.neuronData = null;
        this.currentExertionLevel = 0; // New: For dynamic energy costs
    }
    applyForce(f) { this.force = this.force.add(f); }

    update(dt) {
        if (this.isFixed || this.invMass === 0 || this.nodeType === NodeType.FIXED_ROOT) {
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

        if (this.isEater) {
            const effectiveEatingRadiusMultiplier = EATING_RADIUS_MULTIPLIER_BASE + (EATING_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectiveEatingRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 165, 0, ${0.05 + exertion * 0.1})`; // Opacity increases with exertion
            ctx.fill();
        }
        if (this.isPredator) {
            const effectivePredationRadiusMultiplier = PREDATION_RADIUS_MULTIPLIER_BASE + (PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectivePredationRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 50, 50, ${0.05 + exertion * 0.1})`; // Opacity increases with exertion
            ctx.fill();
        }


        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
        let mainColor = this.isFixed ? 'rgba(255,0,0,0.9)' : this.color;
        if (this.nodeType === NodeType.NEURON) {
            mainColor = 'rgba(200, 100, 255, 0.9)';
        } else if (this.isPredator) {
            mainColor = 'rgba(255, 50, 50, 0.9)';
        } else if (this.nodeType === NodeType.EMITTER_SWIMMER && this.isEater) {
             mainColor = 'rgba(255, 105, 180, 0.9)';
        } else if (this.nodeType === NodeType.PHOTOSYNTHETIC) { 
            mainColor = 'rgba(60, 179, 113, 0.9)'; // MediumSeaGreen for photosynthesis
        } else if (this.nodeType === NodeType.EMITTER_SWIMMER) {
            mainColor = 'rgba(0,255,100,0.9)';
        } else if (this.isEater) {
            mainColor = 'rgba(255,165,0,0.9)';
        } else if (this.nodeType === NodeType.FIXED_ROOT) {
            mainColor = 'rgba(100, 70, 30, 0.9)';
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
    constructor(p1, p2, stiffness, dampingFactor, restLength = null) {
        this.p1 = p1; this.p2 = p2;
        this.stiffness = stiffness;
        this.dampingFactor = dampingFactor;
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
        ctx.strokeStyle = 'rgba(150,150,150,0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();
        // ctx.closePath(); // Not needed for lineTo
    }
}

// --- SoftBody Class ---
class SoftBody {
    constructor(id, initialX, initialY, parentBody = null) {
        this.id = id;
        this.massPoints = [];
        this.springs = [];
        this.isUnstable = false;
        this.creatureEnergy = MAX_CREATURE_ENERGY * OFFSPRING_INITIAL_ENERGY_SHARE;
        this.ticksSinceBirth = 0;
        this.canReproduce = false;
        this.shapeType = parentBody ? parentBody.shapeType : Math.floor(Math.random() * 3); // Used for initial generation

        this.motorImpulseInterval = 30 + Math.floor(Math.random() * 90);
        this.motorImpulseMagnitudeCap = 0.5 + Math.random() * 2.0;
        this.emitterStrength = 0.2 + Math.random() * 1.0;
        this.emitterDirection = new Vec2(Math.random()*2-1, Math.random()*2-1).normalize();
        this.numOffspring = parentBody ? parentBody.numOffspring : (1 + Math.floor(Math.random() * 3));
        this.offspringSpawnRadius = parentBody ? parentBody.offspringSpawnRadius : (50 + Math.random() * 50);
        this.pointAddChance = parentBody ? parentBody.pointAddChance : (0.02 + Math.random() * 0.06);
        this.springConnectionRadius = parentBody ? parentBody.springConnectionRadius : (40 + Math.random() * 40);


        if (parentBody) {
            this.stiffness = parentBody.stiffness * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            this.springDamping = parentBody.springDamping * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            this.motorImpulseInterval = Math.max(10, Math.floor(parentBody.motorImpulseInterval * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER))));
            this.motorImpulseMagnitudeCap = parentBody.motorImpulseMagnitudeCap * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
            this.emitterStrength = parentBody.emitterStrength * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));

            let offspringNumChange = 0;
            if (Math.random() < Math.max(0, Math.min(1, MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER))) {
                offspringNumChange = (Math.random() < 0.5 ? -1 : 1);
            }
            this.numOffspring = Math.max(1, Math.min(5, Math.floor(parentBody.numOffspring + offspringNumChange)));

            this.offspringSpawnRadius = Math.max(20, parentBody.offspringSpawnRadius * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 0.5)));
            this.pointAddChance = Math.max(0, Math.min(0.5, parentBody.pointAddChance * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER * 2))));
            this.springConnectionRadius = Math.max(10, parentBody.springConnectionRadius * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER)));


            const angleMutation = (Math.random() - 0.5) * Math.PI * 0.2 * GLOBAL_MUTATION_RATE_MODIFIER;
            const cosA = Math.cos(angleMutation);
            const sinA = Math.sin(angleMutation);
            this.emitterDirection = new Vec2(
                parentBody.emitterDirection.x * cosA - parentBody.emitterDirection.y * sinA,
                parentBody.emitterDirection.x * sinA + parentBody.emitterDirection.y * cosA
            ).normalize();

        } else {
            this.stiffness = 500 + Math.random() * 2500;
            this.springDamping = 5 + Math.random() * 20;
        }
        this.stiffness = Math.max(100, Math.min(this.stiffness, 10000));
        this.springDamping = Math.max(0.1, Math.min(this.springDamping, 50));
        this.motorImpulseMagnitudeCap = Math.max(0, Math.min(this.motorImpulseMagnitudeCap, 5.0));
        this.motorImpulseInterval = Math.max(10, Math.min(this.motorImpulseInterval, 300));
        this.emitterStrength = Math.max(0, Math.min(this.emitterStrength, 3.0));
        this.numOffspring = Math.max(1, Math.min(this.numOffspring, 5));
        this.offspringSpawnRadius = Math.max(20, Math.min(this.offspringSpawnRadius, 150));
        this.pointAddChance = Math.max(0, Math.min(0.5, this.pointAddChance));
        this.springConnectionRadius = Math.max(10, Math.min(this.springConnectionRadius, 100));


        this.fluidEntrainment = BODY_FLUID_ENTRAINMENT_FACTOR;
        this.fluidCurrentStrength = FLUID_CURRENT_STRENGTH_ON_BODY;
        this.bodyPushStrength = SOFT_BODY_PUSH_STRENGTH;

        this.createShape(initialX, initialY, parentBody);
        this.initializeBrain(); // Call initializeBrain after shape creation
    }

    createShape(startX, startY, parentBody = null) {
        this.massPoints = [];
        this.springs = [];

        const baseRadius = 3 + Math.random() * 3;
        const eaterChance = 0.25;
        const predatorChance = 0.15;
        const dyeEmitterChance = 0.2;
        // const neuronChance = NEURON_CHANCE; // Use global NEURON_CHANCE directly
        const nodeTypeChoices = [NodeType.NEUTRAL, NodeType.FLOATING, NodeType.FIXED_ROOT, NodeType.EMITTER_SWIMMER, NodeType.NEURON, NodeType.PHOTOSYNTHETIC];
        const dyeColorChoices = [DYE_COLORS.RED, DYE_COLORS.GREEN, DYE_COLORS.BLUE];

        if (parentBody) {
            // Morphological evolution from parent
            let lastPointPos = parentBody.massPoints.length > 0 ? parentBody.massPoints[parentBody.massPoints.length-1].pos.clone() : new Vec2(startX, startY);

            parentBody.massPoints.forEach(parentPoint => {
                let mass = parentPoint.mass * (1 + (Math.random() - 0.5) * 2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER));
                mass = Math.max(0.1, Math.min(mass, 1.0));

                let nodeType = parentPoint.nodeType;
                if (Math.random() < (MUTATION_CHANCE_NODE_TYPE * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    nodeType = nodeTypeChoices[Math.floor(Math.random() * nodeTypeChoices.length)];
                }

                const offspringPoint = new MassPoint(
                    parentPoint.pos.x + (Math.random() - 0.5) * 5,
                    parentPoint.pos.y + (Math.random() - 0.5) * 5,
                    mass,
                    parentPoint.radius * (1 + (Math.random() - 0.5) * 0.2 * (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER))
                );
                offspringPoint.nodeType = nodeType;
                    offspringPoint.isEater = Math.random() < (MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER) ? !parentPoint.isEater : parentPoint.isEater;
                    offspringPoint.isPredator = Math.random() < (MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER) ? !parentPoint.isPredator : parentPoint.isPredator;
                    offspringPoint.emitsDye = Math.random() < (MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER) ? !parentPoint.emitsDye : parentPoint.emitsDye;
                offspringPoint.dyeColor = Math.random() < (MUTATION_CHANCE_BOOL * GLOBAL_MUTATION_RATE_MODIFIER) ? dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)] : [...parentPoint.dyeColor];

                // Determine hiddenLayerSize for potential neuron
                let newHiddenLayerSizeForNeuron = DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1));
                if (parentPoint.neuronData && typeof parentPoint.neuronData.hiddenLayerSize === 'number') {
                    newHiddenLayerSizeForNeuron = parentPoint.neuronData.hiddenLayerSize;
                }
                if (Math.random() < (MUTATION_RATE_PERCENT * GLOBAL_MUTATION_RATE_MODIFIER)) {
                    newHiddenLayerSizeForNeuron += Math.floor((Math.random() * 6) - 3);
                    newHiddenLayerSizeForNeuron = Math.max(DEFAULT_HIDDEN_LAYER_SIZE_MIN, Math.min(newHiddenLayerSizeForNeuron, DEFAULT_HIDDEN_LAYER_SIZE_MAX));
                }

                if (nodeType === NodeType.NEURON) {
                    offspringPoint.neuronData = {
                        isBrain: false, 
                        hiddenLayerSize: newHiddenLayerSizeForNeuron
                    };
                } else {
                    offspringPoint.neuronData = null; // Crucial: ensure non-neurons have null neuronData
                }

                if(offspringPoint.nodeType === NodeType.FIXED_ROOT) offspringPoint.isFixed = true;
                this.massPoints.push(offspringPoint);
                lastPointPos = offspringPoint.pos.clone();

                if (Math.random() < this.pointAddChance * GLOBAL_MUTATION_RATE_MODIFIER) {
                    const newMass = 0.1 + Math.random() * 0.9;
                    const newNodeType = nodeTypeChoices[Math.floor(Math.random() * nodeTypeChoices.length)];
                    const newPoint = new MassPoint(
                        lastPointPos.x + (Math.random() - 0.5) * NEW_POINT_OFFSET_RADIUS * 2,
                        lastPointPos.y + (Math.random() - 0.5) * NEW_POINT_OFFSET_RADIUS * 2,
                        newMass,
                        baseRadius * (0.8 + Math.random() * 0.4)
                    );
                    newPoint.nodeType = newNodeType;
                        newPoint.isEater = Math.random() < eaterChance;
                        newPoint.isPredator = Math.random() < predatorChance;
                        newPoint.emitsDye = Math.random() < dyeEmitterChance;
                    newPoint.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    if (newNodeType === NodeType.NEURON) {
                        newPoint.neuronData = {
                            isBrain: false,
                            hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                        };
                    } else {
                        newPoint.neuronData = null; // Crucial: ensure non-neurons have null neuronData
                    }
                    if(newPoint.nodeType === NodeType.FIXED_ROOT) newPoint.isFixed = true;
                    this.massPoints.push(newPoint);
                    lastPointPos = newPoint.pos.clone();
                }
            });

            if (this.massPoints.length === 0) {
                this.massPoints.push(new MassPoint(startX, startY, 0.5, baseRadius));
            }

            for (let i = 0; i < this.massPoints.length; i++) {
                for (let j = i + 1; j < this.massPoints.length; j++) {
                    const p1 = this.massPoints[i];
                    const p2 = this.massPoints[j];
                    const dist = p1.pos.sub(p2.pos).mag();
                    if (dist < this.springConnectionRadius && dist > 0.1) {
                        this.springs.push(new Spring(p1, p2, this.stiffness, this.springDamping, dist));
                    }
                }
            }
            if (this.massPoints.length > 1 && this.springs.length === 0) {
                for(let i = 0; i < this.massPoints.length -1; i++){
                     this.springs.push(new Spring(this.massPoints[i], this.massPoints[i+1], this.stiffness, this.springDamping));
                }
            }


        } else { // Initial generation - use old shape types
            const basePointDist = 20 + Math.random() * 10;
            if (this.shapeType === 0) {
                const numPointsX = 3; const numPointsY = 3; let gridPoints = [];
                for (let i = 0; i < numPointsY; i++) { gridPoints[i] = []; for (let j = 0; j < numPointsX; j++) {
                    const point = new MassPoint(startX + j * basePointDist, startY + i * basePointDist, 0.3 + Math.random() * 0.4, baseRadius);
                    point.isEater = Math.random() < eaterChance;
                    point.isPredator = Math.random() < predatorChance;
                    point.emitsDye = Math.random() < dyeEmitterChance;
                    point.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    this.massPoints.push(point); gridPoints[i][j] = point;
                }}
                for (let i=0; i<numPointsY; i++) for (let j=0; j<numPointsX-1; j++) this.springs.push(new Spring(gridPoints[i][j], gridPoints[i][j+1], this.stiffness, this.springDamping));
                for (let j=0; j<numPointsX; j++) for (let i=0; i<numPointsY-1; i++) this.springs.push(new Spring(gridPoints[i][j], gridPoints[i+1][j], this.stiffness, this.springDamping));
                for (let i=0; i<numPointsY-1; i++) for (let j=0; j<numPointsX-1; j++) {
                    this.springs.push(new Spring(gridPoints[i][j], gridPoints[i+1][j+1], this.stiffness*0.7, this.springDamping));
                    this.springs.push(new Spring(gridPoints[i+1][j], gridPoints[i][j+1], this.stiffness*0.7, this.springDamping));
                }
            } else if (this.shapeType === 1) {
                const numLinePoints = Math.floor(3 + Math.random() * 3); const isHorizontal = Math.random() < 0.5; let linePoints = [];
                for (let i=0; i<numLinePoints; i++) {
                    const x = startX + (isHorizontal ? i * basePointDist : 0); const y = startY + (isHorizontal ? 0 : i * basePointDist);
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    point.isEater = Math.random() < eaterChance; point.isPredator = Math.random() < predatorChance; point.emitsDye = Math.random() < dyeEmitterChance; point.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    this.massPoints.push(point); linePoints.push(point);
                }
                for (let i=0; i<numLinePoints-1; i++) this.springs.push(new Spring(linePoints[i], linePoints[i+1], this.stiffness, this.springDamping));
                if (numLinePoints > 2) this.springs.push(new Spring(linePoints[0], linePoints[numLinePoints-1], this.stiffness*0.5, this.springDamping));
            } else {
                const numOuterPoints = Math.floor(4 + Math.random()*3); const centralPoint = new MassPoint(startX, startY, (0.3+Math.random()*0.4)*1.5, baseRadius*1.2);
                centralPoint.isEater = Math.random() < eaterChance; centralPoint.isPredator = Math.random() < predatorChance; centralPoint.emitsDye = Math.random() < dyeEmitterChance; centralPoint.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                this.massPoints.push(centralPoint); const circleRadius = basePointDist * 1.5;
                for (let i=0; i<numOuterPoints; i++) {
                    const angle = (i / numOuterPoints) * Math.PI * 2; const x = startX + Math.cos(angle)*circleRadius; const y = startY + Math.sin(angle)*circleRadius;
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    point.isEater = Math.random() < eaterChance; point.isPredator = Math.random() < predatorChance; point.emitsDye = Math.random() < dyeEmitterChance; point.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    this.massPoints.push(point);
                    this.springs.push(new Spring(centralPoint, point, this.stiffness, this.springDamping));
                    if (i>0) this.springs.push(new Spring(this.massPoints[this.massPoints.length-2], point, this.stiffness*0.8, this.springDamping));
                }
                if (numOuterPoints > 1) this.springs.push(new Spring(this.massPoints[1], this.massPoints[this.massPoints.length-1], this.stiffness*0.8, this.springDamping));
            }

             this.massPoints.forEach((p, idx) => {
                // Initial random choice from the list (can be overridden by specific chances below)
                let currentNodeType = nodeTypeChoices[Math.floor(Math.random() * nodeTypeChoices.length)]; // Corrected: use .length
                const ACTUAL_PHOTOSYNTHETIC_CHANCE = 0.15; // Define the chance here
                // NEURON_CHANCE is assumed to be globally defined (e.g., const NEURON_CHANCE = 0.1;)

                const rand = Math.random(); // Single random draw for mutually exclusive chances
                if (rand < NEURON_CHANCE && this.massPoints.length > 1) {
                    currentNodeType = NodeType.NEURON;
                } else if (rand < NEURON_CHANCE + ACTUAL_PHOTOSYNTHETIC_CHANCE) { // Check for photosynthetic if not neuron
                    currentNodeType = NodeType.PHOTOSYNTHETIC;
                }
                // If neither of the above, currentNodeType remains as initially picked from nodeTypeChoices

                p.nodeType = currentNodeType; // Assign type

                // Now set neuronData based on the final type
                if (p.nodeType === NodeType.NEURON) {
                     p.neuronData = {
                        isBrain: false,
                        hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                    };
                } else {
                    // This correctly ensures that if it's not a NEURON (e.g., it became PHOTOSYNTHETIC),
                    // neuronData is null.
                    p.neuronData = null;
                }
                if (p.nodeType === NodeType.FIXED_ROOT) p.isFixed = true;
            });
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
                const effectorCandidates = this.massPoints.map((ep, epIdx) => ep.nodeType === NodeType.EMITTER_SWIMMER ? epIdx : -1).filter(epIdx => epIdx !== -1 && epIdx !== idx);
                if (effectorCandidates.length > 0) {
                     p.neuronData.effectorPointIndex = effectorCandidates[Math.floor(Math.random() * effectorCandidates.length)];
                } else {
                    p.neuronData.effectorPointIndex = -1;
                }
            }
        });
    }


    updateSelf(dt, fluidFieldRef) {
        if (this.isUnstable) return;

        // --- New Neural Network Processing (Step 4) ---
        let brainNode = null;
        for (const point of this.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        if (brainNode && brainNode.neuronData && 
            brainNode.neuronData.weightsIH && brainNode.neuronData.biasesH && 
            brainNode.neuronData.weightsHO && brainNode.neuronData.biasesO &&
            typeof brainNode.neuronData.inputVectorSize === 'number' &&
            typeof brainNode.neuronData.hiddenLayerSize === 'number' &&
            typeof brainNode.neuronData.outputVectorSize === 'number') {

            const nd = brainNode.neuronData;
            const inputVector = [];

            // 1. Gather Inputs (8 inputs total)
            // a. Local Dye (RGB) at brain node's position
            if (fluidFieldRef) {
                const brainGx = Math.floor(brainNode.pos.x / fluidFieldRef.scaleX);
                const brainGy = Math.floor(brainNode.pos.y / fluidFieldRef.scaleY);
                const brainIdx = fluidFieldRef.IX(brainGx, brainGy);
                inputVector.push((fluidFieldRef.densityR[brainIdx] || 0) / 255); // Normalize to 0-1
                inputVector.push((fluidFieldRef.densityG[brainIdx] || 0) / 255);
                inputVector.push((fluidFieldRef.densityB[brainIdx] || 0) / 255);
            } else {
                inputVector.push(0, 0, 0); // No fluid field, no dye input
            }

            // b. Creature Energy (normalized)
            inputVector.push(this.creatureEnergy / MAX_CREATURE_ENERGY); // Normalize to 0-1

            // c. Relative Center of Mass Position (XY)
            const comPos = this.getAveragePosition();
            const relComPosX = (comPos.x - brainNode.pos.x) / WORLD_WIDTH; // Normalize by world dim
            const relComPosY = (comPos.y - brainNode.pos.y) / WORLD_HEIGHT;
            inputVector.push(Math.tanh(relComPosX)); // Use tanh to keep values bounded nicely
            inputVector.push(Math.tanh(relComPosY));

            // d. Relative Center of Mass Velocity (XY)
            const comVel = this.getAverageVelocity();
            const brainVelX = brainNode.pos.x - brainNode.prevPos.x;
            const brainVelY = brainNode.pos.y - brainNode.prevPos.y;
            const relComVelX = comVel.x - brainVelX;
            const relComVelY = comVel.y - brainVelY;
            inputVector.push(Math.tanh(relComVelX / MAX_PIXELS_PER_FRAME_DISPLACEMENT)); // Normalize and tanh
            inputVector.push(Math.tanh(relComVelY / MAX_PIXELS_PER_FRAME_DISPLACEMENT));
            
            // e. Nutrient Level (1 input)
            if (nutrientField && fluidFieldRef) {
                const brainGx = Math.floor(brainNode.pos.x / fluidFieldRef.scaleX);
                const brainGy = Math.floor(brainNode.pos.y / fluidFieldRef.scaleY);
                const nutrientIdx = fluidFieldRef.IX(brainGx, brainGy);
                const currentNutrient = nutrientField[nutrientIdx] !== undefined ? nutrientField[nutrientIdx] : 1.0;
                const normalizedNutrient = (currentNutrient - MIN_NUTRIENT_VALUE) / (MAX_NUTRIENT_VALUE - MIN_NUTRIENT_VALUE);
                inputVector.push(Math.max(0, Math.min(1, normalizedNutrient))); // Clamp to 0-1
                // REMOVED: inputVector.push(Math.max(0, Math.min(1, normalizedNutrient))); // Duplicate for the second nutrient input for now
            } else {
                inputVector.push(0.5); // Default neutral nutrient if no field
                // REMOVED: inputVector.push(0.5);
            }

            // Ensure inputVector is the correct size, pad with 0 if necessary (e.g. if fluidFieldRef was null)
            while(inputVector.length < nd.inputVectorSize) {
                inputVector.push(0);
            }
            if(inputVector.length > nd.inputVectorSize) {
                inputVector.splice(nd.inputVectorSize); // Truncate if too long
            }


            // 2. Forward Propagation
            // Hidden Layer
            const hiddenLayerInputs = multiplyMatrixVector(nd.weightsIH, inputVector);
            const hiddenLayerBiasedInputs = addVectors(hiddenLayerInputs, nd.biasesH);
            const hiddenLayerActivations = hiddenLayerBiasedInputs.map(val => Math.tanh(val));

            // Output Layer
            const outputLayerInputs = multiplyMatrixVector(nd.weightsHO, hiddenLayerActivations);
            const rawOutputs = addVectors(outputLayerInputs, nd.biasesO);

            nd.rawOutputs = rawOutputs; // Store for Step 5
            // console.log(`Body ${this.id} brain processed. Inputs: [${inputVector.map(v=>v.toFixed(2))}], Outputs: [${rawOutputs.map(v=>v.toFixed(2))}]`);

            // --- Apply Neural Outputs (Modified for Policy Gradient - Step 1 of RL Phase) ---
            nd.currentFrameActionDetails = []; // Initialize/clear array for the current frame
            let currentRawOutputIndex = 0; // Tracks the current starting index in nd.rawOutputs

            // Helper to process one action component (mean + stdDev pair)
            // Moved this helper function outside the point iteration loop for broader access
            function sampleAndLogAction(rawMean, rawStdDev) {
                const mean = rawMean; // Assuming rawMean is suitable as direct mean
                const stdDev = Math.exp(rawStdDev) + 1e-6; // Ensure stdDev is positive and non-zero
                const sampledActionValue = sampleGaussian(mean, stdDev);
                const logProb = logPdfGaussian(sampledActionValue, mean, stdDev);
                
                return { 
                    detail: { mean, stdDev, sampledAction: sampledActionValue, logProb }, 
                    value: sampledActionValue 
                };
            }

            this.massPoints.forEach(point => {
                point.currentExertionLevel = 0; // Reset exertion for all points before NN potentially sets it
            });

            // Process Emitter/Swimmers first (matches order in initializeBrain for outputVectorSize)
            this.massPoints.forEach(point => {
                if (point.nodeType === NodeType.EMITTER_SWIMMER) {
                    const outputStartRawIdx = currentRawOutputIndex;
                    if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_EMITTER_SWIMMER) {
                        const detailsForThisEmitter = [];
                        let localRawOutputPairIdx = 0;

                        // --- Action 1: Force X ---
                        const forceXResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(forceXResult.detail);
                        // Applied force will be scaled by exertion later, after exertion is determined for this point

                        // --- Action 2: Force Y ---
                        const forceYResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(forceYResult.detail);
                        // Applied force will be scaled by exertion later

                        // --- Action 3: Target Dye Color R ---
                        const dyeRResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(dyeRResult.detail);
                        point.dyeColor[0] = sigmoid(dyeRResult.value) * 255;

                        // --- Action 4: Target Dye Color G ---
                        const dyeGResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(dyeGResult.detail);
                        point.dyeColor[1] = sigmoid(dyeGResult.value) * 255;

                        // --- Action 5: Target Dye Color B ---
                        const dyeBResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(dyeBResult.detail);
                        point.dyeColor[2] = sigmoid(dyeBResult.value) * 255;

                        // --- Action 6: Emission Pull Strength ---
                        const pullResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(pullResult.detail);
                        // Emission pull strength will be scaled by exertion later

                        // --- Action 7: Emitter/Swimmer Exertion ---
                        const exertionResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++],
                            nd.rawOutputs[outputStartRawIdx + localRawOutputPairIdx++]
                        );
                        detailsForThisEmitter.push(exertionResult.detail);
                        point.currentExertionLevel = sigmoid(exertionResult.value); 

                        // Now apply exertion-scaled effects
                        const appliedForceX = Math.tanh(forceXResult.value) * MAX_NEURAL_FORCE_COMPONENT * this.emitterStrength * point.currentExertionLevel;
                        const appliedForceY = Math.tanh(forceYResult.value) * MAX_NEURAL_FORCE_COMPONENT * this.emitterStrength * point.currentExertionLevel;
                        point.applyForce(new Vec2(appliedForceX / dt, appliedForceY / dt));

                        const neuralPullStrength = sigmoid(pullResult.value) * MAX_NEURAL_EMISSION_PULL_STRENGTH * point.currentExertionLevel;
                        if (fluidFieldRef && point.emitsDye) {
                            const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                            const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                            fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], neuralPullStrength * 50);
                        }
                        
                        nd.currentFrameActionDetails.push(...detailsForThisEmitter);
                        currentRawOutputIndex += NEURAL_OUTPUTS_PER_EMITTER_SWIMMER;
                    } else {
                        // Not enough rawOutputs for this ES point.
                        // Still advance the raw output index to keep subsequent points aligned if possible,
                        // or handle error more explicitly if this state is unexpected.
                        currentRawOutputIndex += NEURAL_OUTPUTS_PER_EMITTER_SWIMMER;
                    }
                }
            });

            // Process Eaters next
            this.massPoints.forEach(point => {
                // Process only if it's an Eater AND NOT ALSO an Emitter/Swimmer (to avoid double processing outputs)
                if (point.isEater && point.nodeType !== NodeType.EMITTER_SWIMMER) {
                    const outputStartRawIdx = currentRawOutputIndex;
                    if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_EATER) {
                        const detailsForThisEater = [];
                        const exertionResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx],
                            nd.rawOutputs[outputStartRawIdx + 1]
                        );
                        detailsForThisEater.push(exertionResult.detail);
                        point.currentExertionLevel = sigmoid(exertionResult.value); 
                        
                        nd.currentFrameActionDetails.push(...detailsForThisEater);
                        currentRawOutputIndex += NEURAL_OUTPUTS_PER_EATER;
                    } else {
                        currentRawOutputIndex += NEURAL_OUTPUTS_PER_EATER;
                    }
                }
            });

            // Process Predators last
            this.massPoints.forEach(point => {
                // Process only if Predator AND NOT ES AND NOT Eater
                if (point.isPredator && point.nodeType !== NodeType.EMITTER_SWIMMER && !point.isEater) {
                   const outputStartRawIdx = currentRawOutputIndex;
                    if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_PREDATOR) {
                        const detailsForThisPredator = [];
                        const exertionResult = sampleAndLogAction(
                            nd.rawOutputs[outputStartRawIdx],
                            nd.rawOutputs[outputStartRawIdx + 1]
                        );
                        detailsForThisPredator.push(exertionResult.detail);
                        point.currentExertionLevel = sigmoid(exertionResult.value);

                        nd.currentFrameActionDetails.push(...detailsForThisPredator);
                        currentRawOutputIndex += NEURAL_OUTPUTS_PER_PREDATOR;
                    } else {
                        currentRawOutputIndex += NEURAL_OUTPUTS_PER_PREDATOR;
                    }
                }
            });
            // --- End of Apply Neural Outputs ---

            // --- Store Experience for RL (Step 2 of RL Phase) ---
            // This happens after actions are taken and energy costs for the frame are implicitly about to be (or have been) accounted for by the simulation loop.
            if (nd.currentFrameActionDetails && nd.currentFrameActionDetails.length > 0) {
                const reward = this.creatureEnergy - nd.previousEnergyForReward;
                const experience = {
                    state: [...inputVector], // Store a copy of the inputVector used for this frame
                    actionDetails: JSON.parse(JSON.stringify(nd.currentFrameActionDetails)), // Deep copy of action details
                    reward: reward
                };
                nd.experienceBuffer.push(experience);

                // Manage buffer size
                if (nd.experienceBuffer.length > nd.maxExperienceBufferSize) {
                    nd.experienceBuffer.shift(); // Remove oldest experience
                }
            }
            nd.previousEnergyForReward = this.creatureEnergy; // Update for next frame's reward calculation
            // --- End of Store Experience ---

            // --- RL Training Trigger ---
            nd.framesSinceLastTrain++;
            if (nd.framesSinceLastTrain >= TRAINING_INTERVAL_FRAMES) {
                this.updateBrainPolicy(); 
            }
            // --- End of RL Training Trigger ---

        } else {
            // No operable brain, or brain not fully initialized
            if (brainNode && brainNode.neuronData) brainNode.neuronData.rawOutputs = []; // Clear if partially set

            // Fallback 1: Original random motor impulses for all points
            if (this.motorImpulseMagnitudeCap > 0.0001 && (this.ticksSinceBirth % this.motorImpulseInterval === 0)) {
                for (let point of this.massPoints) {
                    if (!point.isFixed && point.nodeType !== NodeType.FIXED_ROOT) {
                        const randomAngle = Math.random() * Math.PI * 2;
                        const impulseDir = new Vec2(Math.cos(randomAngle), Math.sin(randomAngle));
                        const impulseMag = Math.random() * this.motorImpulseMagnitudeCap;
                        const impulseForce = impulseDir.mul(impulseMag / dt);
                        point.applyForce(impulseForce);
                    }
                }
            }

            // Fallback 2: Original EMITTER_SWIMMER behavior (fluid push and dye emission)
            if (fluidFieldRef) {
                this.massPoints.forEach(point => {
                    if (point.nodeType === NodeType.EMITTER_SWIMMER && this.emitterStrength > 0.0001 && !point.isFixed) {
                        const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                        const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                        
                        // Fluid push based on fixed emitterDirection
                        const emitterPushVx = this.emitterDirection.x * this.emitterStrength;
                        const emitterPushVy = this.emitterDirection.y * this.emitterStrength;
                        fluidFieldRef.addVelocity(fluidGridX, fluidGridY, emitterPushVx, emitterPushVy);
                        
                        // Dye emission based on point's own fixed dyeColor
                        if (point.emitsDye) { 
                            fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], 50); // 50 is default emission strength
                        }
                    }
                });
            }
        } // --- End of New Neural Network Processing (Brain & Fallback) ---

        // --- Apply Red Dye Poison Effect ---
        if (fluidFieldRef && RED_DYE_POISON_STRENGTH > 0) {
            let poisonDamageThisFrame = 0;
            for (const point of this.massPoints) {
                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);
                const redDensity = (fluidFieldRef.densityR[idx] || 0) / 255; // Normalize to 0-1

                if (redDensity > 0.01) { // Only apply if there's a meaningful amount of red dye
                    poisonDamageThisFrame += redDensity * RED_DYE_POISON_STRENGTH * (point.radius / 5); // Scale by point radius (avg radius 5)
                }
            }
            if (poisonDamageThisFrame > 0) {
                this.creatureEnergy -= poisonDamageThisFrame * dt * 60; // dt is in seconds, scale strength to be per-second like other costs
            }
        }
        // --- End of Red Dye Poison Effect ---


        // Calculate and apply energy cost
        let currentFrameEnergyCost = 0;
        for (const point of this.massPoints) {
            let costMultiplier = 1.0;
            if (nutrientField && fluidFieldRef) {
                const gx = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const gy = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const nutrientIdx = fluidFieldRef.IX(gx, gy);
                const baseNutrientValue = nutrientField[nutrientIdx] !== undefined ? nutrientField[nutrientIdx] : 1.0;
                const effectiveNutrientValue = baseNutrientValue * globalNutrientMultiplier;
                costMultiplier = 1.0 / Math.max(MIN_NUTRIENT_VALUE, effectiveNutrientValue);
            }

            currentFrameEnergyCost += BASE_NODE_EXISTENCE_COST * costMultiplier;
            const exertion = point.currentExertionLevel || 0; // Default to 0 if undefined

            if (point.nodeType === NodeType.EMITTER_SWIMMER) {
                currentFrameEnergyCost += EMITTER_NODE_ENERGY_COST * exertion * exertion * costMultiplier;
            }
             if (point.nodeType === NodeType.NEURON) {
                if (point.neuronData && point.neuronData.isBrain) {
                    currentFrameEnergyCost += NEURON_NODE_ENERGY_COST * 5 * costMultiplier; 
                    currentFrameEnergyCost += (point.neuronData.hiddenLayerSize || 0) * NEURON_NODE_ENERGY_COST * 0.1 * costMultiplier; 
                } else {
                    currentFrameEnergyCost += NEURON_NODE_ENERGY_COST * costMultiplier; 
                }
            }
            if (point.isEater) {
                currentFrameEnergyCost += EATER_NODE_ENERGY_COST * exertion * exertion * costMultiplier;
            }
            if (point.isPredator) {
                currentFrameEnergyCost += PREDATOR_NODE_ENERGY_COST * exertion * exertion * costMultiplier;
            }
            if (point.nodeType === NodeType.PHOTOSYNTHETIC) {
                currentFrameEnergyCost += PHOTOSYNTHETIC_NODE_ENERGY_COST * costMultiplier;
                if (lightField && fluidFieldRef) {
                    const gx_photo = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                    const gy_photo = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                    const lightIdx = fluidFieldRef.IX(gx_photo, gy_photo);
                    const baseLightValue = lightField[lightIdx] !== undefined ? lightField[lightIdx] : 0.0;
                    const effectiveLightValue = baseLightValue * globalLightMultiplier; // Use globalLightMultiplier

                    const energyGain = effectiveLightValue * PHOTOSYNTHESIS_EFFICIENCY * (point.radius / 5) * dt;
                    this.creatureEnergy = Math.min(MAX_CREATURE_ENERGY, this.creatureEnergy + energyGain);
                }
            }
        }
        this.creatureEnergy -= currentFrameEnergyCost * dt;


        this.ticksSinceBirth++;
        if (this.ticksSinceBirth > REPRODUCTION_COOLDOWN_TICKS) {
            this.canReproduce = true;
        }

        if (this.creatureEnergy <= 0) {
            this.isUnstable = true;
            return;
        }

        if (this.motorImpulseMagnitudeCap > 0.0001 && (this.ticksSinceBirth % this.motorImpulseInterval === 0)) {
            for (let point of this.massPoints) {
                if (!point.isFixed && point.nodeType !== NodeType.FIXED_ROOT) {
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
                if (point.isFixed || point.nodeType === NodeType.FIXED_ROOT) continue;

                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);

                if (point.nodeType === NodeType.FLOATING) {
                    const rawFluidVx = fluidFieldRef.Vx[idx];
                    const rawFluidVy = fluidFieldRef.Vy[idx];
                    let fluidDisplacementPx = new Vec2(rawFluidVx * fluidFieldRef.scaleX * dt, rawFluidVy * fluidFieldRef.scaleY * dt);
                    let effectiveFluidDisplacementPx = fluidDisplacementPx.mul(this.fluidCurrentStrength);
                    let currentPointDisplacementPx = point.pos.sub(point.prevPos);
                    let blendedDisplacementPx = currentPointDisplacementPx.mul(1.0 - this.fluidEntrainment)
                                                     .add(effectiveFluidDisplacementPx.mul(this.fluidEntrainment));
                    point.prevPos = point.pos.clone().sub(blendedDisplacementPx);
                } else if (point.nodeType === NodeType.EMITTER_SWIMMER && this.emitterStrength > 0.0001) {
                    let currentEmitterStrength = this.emitterStrength;
                    if (point.neuralEffectiveStrength !== null && point.neuralEffectiveStrength !== undefined) {
                        currentEmitterStrength *= point.neuralEffectiveStrength;
                    }

                    const emitterPushVx = this.emitterDirection.x * currentEmitterStrength;
                    const emitterPushVy = this.emitterDirection.y * currentEmitterStrength;
                    fluidFieldRef.addVelocity(fluidGridX, fluidGridY, emitterPushVx, emitterPushVy);
                }

                if (point.emitsDye) {
                    fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], 50);
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

                                    if (p1.isPredator) {
                                        const p1Exertion = p1.currentExertionLevel || 0;
                                        const effectivePredationRadiusMultiplier = PREDATION_RADIUS_MULTIPLIER_BASE + (PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * p1Exertion);
                                        const predationRadius = p1.radius * effectivePredationRadiusMultiplier;
                                        
                                        if (distSq < predationRadius * predationRadius) {
                                            const effectiveEnergySapped = ENERGY_SAPPED_PER_PREDATION_BASE + (ENERGY_SAPPED_PER_PREDATION_MAX_BONUS * p1Exertion);
                                            const energyToSap = Math.min(otherItem.bodyRef.creatureEnergy, effectiveEnergySapped);
                                            if (energyToSap > 0) {
                                                otherItem.bodyRef.creatureEnergy -= energyToSap;
                                                this.creatureEnergy = Math.min(MAX_CREATURE_ENERGY, this.creatureEnergy + energyToSap);
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
            if (displacementSq > MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ ||
                isNaN(point.pos.x) || isNaN(point.pos.y) ||
                !isFinite(point.pos.x) || !isFinite(point.pos.y)) {
                this.isUnstable = true;
                console.warn(`Soft body ID ${this.id} point instability (displacement/NaN/Infinite)!`, point);
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

        if (this.massPoints.length > 2) { // Only check span for bodies with more than 2 points
            const bbox = this.getBoundingBox();
            if (bbox.width > this.massPoints.length * localMaxSpanPerPointFactor ||
                bbox.height > this.massPoints.length * localMaxSpanPerPointFactor) {
                this.isUnstable = true;
                console.warn(`Soft body ID ${this.id} span instability! Width: ${bbox.width.toFixed(1)}, Height: ${bbox.height.toFixed(1)}, Points: ${this.massPoints.length}`);
            }
        }
        if (this.isUnstable) return;


        if (!this.isUnstable) { // Re-check after all instability checks
            for (let point of this.massPoints) {
                if (point.isFixed || point.nodeType === NodeType.FIXED_ROOT) continue;

                if (point.isEater) {
                    const pointExertion = point.currentExertionLevel || 0;
                    const effectiveEatingRadiusMultiplier = EATING_RADIUS_MULTIPLIER_BASE + (EATING_RADIUS_MULTIPLIER_MAX_BONUS * pointExertion);
                    const eatingRadius = point.radius * effectiveEatingRadiusMultiplier;
                    const eatingRadiusSq = eatingRadius * eatingRadius;

                    // Re-add eaterGx and eaterGy calculation here
                    const eaterGx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(point.pos.x / GRID_CELL_SIZE)));
                    const eaterGy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(point.pos.y / GRID_CELL_SIZE)));

                    for (let dy = -1; dy <= 1; dy++) { // Check local grid cells for particles
                        for (let dx = -1; dx <= 1; dx++) {
                            const checkGx = eaterGx + dx;
                            const checkGy = eaterGy + dy;
                            if (checkGx >= 0 && checkGx < GRID_COLS && checkGy >= 0 && checkGy < GRID_ROWS) {
                                const cellIndex = checkGx + checkGy * GRID_COLS;
                                if (Array.isArray(spatialGrid[cellIndex])) {
                                    const cellBucket = spatialGrid[cellIndex];
                                    for (let k = cellBucket.length - 1; k >= 0; k--) { // Iterate backwards for safe removal
                                        const item = cellBucket[k];
                                        if (item.type === 'particle') {
                                            const particle = item.particleRef;
                                            if (particle.life > 0 && !particle.isEaten) {
                                                const distSq = point.pos.sub(particle.pos).magSq();
                                                if (distSq < eatingRadiusSq) {
                                                    particle.isEaten = true;
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
                                                    this.creatureEnergy = Math.min(MAX_CREATURE_ENERGY, this.creatureEnergy + energyGain);
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

        const energyForOneOffspring = MAX_CREATURE_ENERGY * OFFSPRING_INITIAL_ENERGY_SHARE;
        let successfullyPlacedOffspring = 0;
        let offspring = [];

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
                for (const otherBody of softBodyPopulation) {
                    if (otherBody.isUnstable) continue;
                    const otherBBox = otherBody.getBoundingBox();
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
                    const finalChild = new SoftBody(nextSoftBodyId++, 0, 0, this); // Create with dummy coords first
                    finalChild.massPoints = []; // Clear default points
                    tempChild.massPoints.forEach(tp => { // Copy points and translate them
                        const newP = new MassPoint(tp.pos.x + spawnX - tempChild.getAveragePosition().x, tp.pos.y + spawnY - tempChild.getAveragePosition().y, tp.mass, tp.radius, tp.color);
                        newP.nodeType = tp.nodeType; newP.isEater = tp.isEater; newP.isPredator = tp.isPredator; newP.emitsDye = tp.emitsDye; newP.dyeColor = [...tp.dyeColor];
                        if(tp.neuronData) newP.neuronData = JSON.parse(JSON.stringify(tp.neuronData)); // Deep copy neuron data
                        if(newP.nodeType === NodeType.FIXED_ROOT) newP.isFixed = true;
                        finalChild.massPoints.push(newP);
                    });
                    // Recreate springs for the final child based on its actual points
                    finalChild.springs = [];
                    for (let k = 0; k < finalChild.massPoints.length; k++) {
                        for (let l = k + 1; l < finalChild.massPoints.length; l++) {
                            const p1 = finalChild.massPoints[k];
                            const p2 = finalChild.massPoints[l];
                            const dist = p1.pos.sub(p2.pos).mag();
                            if (dist < finalChild.springConnectionRadius && dist > 0.1) {
                                finalChild.springs.push(new Spring(p1, p2, finalChild.stiffness, finalChild.springDamping, dist));
                            }
                        }
                    }
                     if (finalChild.massPoints.length > 1 && finalChild.springs.length === 0) {
                        for(let k = 0; k < finalChild.massPoints.length -1; k++){
                            finalChild.springs.push(new Spring(finalChild.massPoints[k], finalChild.massPoints[k+1], finalChild.stiffness, finalChild.springDamping));
                        }
                    }
                    // Re-link neurons for the final child
                    finalChild.massPoints.forEach((p, pIdx) => {
                        if (p.nodeType === NodeType.NEURON && p.neuronData) {
                            if (finalChild.massPoints.length > 1) {
                                let newSensorIndex;
                                do { newSensorIndex = Math.floor(Math.random() * finalChild.massPoints.length); } while (newSensorIndex === pIdx);
                                p.neuronData.sensorPointIndex = newSensorIndex;
                            } else { p.neuronData.sensorPointIndex = -1; }
                            const effectorCandidates = finalChild.massPoints.map((ep, epIdx) => ep.nodeType === NodeType.EMITTER_SWIMMER ? epIdx : -1).filter(epIdx => epIdx !== -1 && epIdx !== pIdx);
                            if (effectorCandidates.length > 0) { p.neuronData.effectorPointIndex = effectorCandidates[Math.floor(Math.random() * effectorCandidates.length)]; } else { p.neuronData.effectorPointIndex = -1;}
                        }
                    });


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
        for (const point of this.massPoints) {
            if (point.nodeType === NodeType.NEURON) {
                if (!brainNode) { // Designate first neuron as brain
                    brainNode = point;
                    if (!brainNode.neuronData) { // Should have been initialized in createShape
                        brainNode.neuronData = { isBrain: false, hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN }; 
                    }
                    brainNode.neuronData.isBrain = true;
                } else {
                    // If other neuron points exist, ensure they are not marked as brain
                    if (point.neuronData) point.neuronData.isBrain = false;
                }
            }
        }

        if (brainNode && brainNode.neuronData && brainNode.neuronData.isBrain) {
            const nd = brainNode.neuronData;
            nd.inputVectorSize = NEURAL_INPUT_SIZE;

            let numEmitterSwimmerPoints = 0;
            let numEaterPoints = 0;
            let numPredatorPoints = 0;

            this.massPoints.forEach(p => {
                if (p.nodeType === NodeType.EMITTER_SWIMMER) {
                    numEmitterSwimmerPoints++;
                } else if (p.isEater) { // Assuming Eater nodes are distinct or NodeType will be expanded
                    numEaterPoints++;
                } else if (p.isPredator) { // Assuming Predator nodes are distinct
                    numPredatorPoints++;
                }
            });
            nd.outputVectorSize = (numEmitterSwimmerPoints * NEURAL_OUTPUTS_PER_EMITTER_SWIMMER) +
                                  (numEaterPoints * NEURAL_OUTPUTS_PER_EATER) +
                                  (numPredatorPoints * NEURAL_OUTPUTS_PER_PREDATOR);

            // Initialize weights and biases
            if (typeof nd.hiddenLayerSize !== 'number' || nd.hiddenLayerSize < DEFAULT_HIDDEN_LAYER_SIZE_MIN) {
                console.warn(`Body ${this.id} brain node had invalid hiddenLayerSize: ${nd.hiddenLayerSize}. Resetting to default: ${DEFAULT_HIDDEN_LAYER_SIZE_MIN}. NeuronData was:`, JSON.parse(JSON.stringify(nd)));
                nd.hiddenLayerSize = DEFAULT_HIDDEN_LAYER_SIZE_MIN;
            }

            nd.weightsIH = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize);
            nd.biasesH = initializeVector(nd.hiddenLayerSize);
            nd.weightsHO = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize);
            nd.biasesO = initializeVector(nd.outputVectorSize);
            
            // For Reinforcement Learning - Experience Buffer & Reward Tracking
            nd.experienceBuffer = [];
            nd.maxExperienceBufferSize = 10; // e.g., last 10 frames/experiences
            nd.previousEnergyForReward = this.creatureEnergy; // Initialize for first reward calculation
            nd.framesSinceLastTrain = 0; 
            nd.lastAvgNormalizedReward = 0; // Initialize diagnostic value for average reward

            // console.log(`Body ${this.id} brain initialized. Inputs: ${nd.inputVectorSize}, Hidden: ${nd.hiddenLayerSize}, Outputs: ${nd.outputVectorSize}`);
        } else {
            // No brain node found or brainNode.neuronData is missing somehow
            // console.log(`Body ${this.id} has no operable brain node.`);
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
}


// --- FluidField Class (Simplified) ---
class FluidField {
    constructor(size, diffusion, viscosity, dt, scaleX, scaleY) { // Added scaleX, scaleY
        this.size = Math.round(size); // Ensure integer
        this.dt = dt;
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.scaleX = scaleX; // Store separate scales
        this.scaleY = scaleY;
        this.useWrapping = false;
        this.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT; // New

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
        const normalizedEmissionEffect = (emissionStrength / 50.0) * DYE_PULL_RATE; // Use 50 as a reference emission strength

        let currentR = this.densityR[idx];
        let targetDiffR = emitterR - currentR;
        this.densityR[idx] += targetDiffR * normalizedEmissionEffect;
        this.densityR[idx] = Math.max(0, Math.min(255, this.densityR[idx]));

        let currentG = this.densityG[idx];
        let targetDiffG = emitterG - currentG;
        this.densityG[idx] += targetDiffG * normalizedEmissionEffect;
        this.densityG[idx] = Math.max(0, Math.min(255, this.densityG[idx]));

        let currentB = this.densityB[idx];
        let targetDiffB = emitterB - currentB;
        this.densityB[idx] += targetDiffB * normalizedEmissionEffect;
        this.densityB[idx] = Math.max(0, Math.min(255, this.densityB[idx]));
    }


    addVelocity(x, y, amountX, amountY) {
        const idx = this.IX(x, y);
        let newVx = this.Vx[idx] + amountX;
        let newVy = this.Vy[idx] + amountY;

        // Cap individual components
        this.Vx[idx] = Math.max(-this.maxVelComponent, Math.min(newVx, this.maxVelComponent));
        this.Vy[idx] = Math.max(-this.maxVelComponent, Math.min(newVy, this.maxVelComponent));
    }

    clampVelocityComponents(arr) {
        for(let i=0; i < arr.length; i++) {
            arr[i] = Math.max(-this.maxVelComponent, Math.min(arr[i], this.maxVelComponent));
        }
    }


    lin_solve(b, x, x0, a_global_param, c_global_param, field_type, base_diff_rate, dt_param) { // Checkpoint 2 Signature & params used
        // const cRecip = 1.0 / c; // Not needed directly like this anymore
        for (let k_iter = 0; k_iter < this.iterations; k_iter++) { 
            for (let j = 1; j < this.size - 1; j++) {
                for (let i = 1; i < this.size - 1; i++) {
                    const idx = this.IX(i,j);
                    
                    let effective_a = a_global_param; // Use the passed global 'a' by default
                    let effective_cRecip = 1.0 / c_global_param; // Use the passed global 'c' by default

                    if ((field_type === 'velX' || field_type === 'velY')) {
                        if (viscosityField && viscosityField[idx] !== undefined) { // Check viscosityField exists and has a value for this cell
                            const localViscosityMultiplier = Math.max(MIN_VISCOSITY_MULTIPLIER, Math.min(viscosityField[idx], MAX_VISCOSITY_MULTIPLIER));
                            const cell_specific_diff_rate = base_diff_rate * localViscosityMultiplier; // base_diff_rate is global fluid viscosity
                            
                            const temp_effective_a = dt_param * cell_specific_diff_rate * (this.size - 2) * (this.size - 2);
                            const temp_denominator_c = 1 + 4 * temp_effective_a;

                            if (temp_denominator_c !== 0 && !isNaN(temp_effective_a) && isFinite(temp_effective_a)) { // Check for sane values
                                effective_a = temp_effective_a;
                                effective_cRecip = 1.0 / temp_denominator_c;
                            } else {
                                // Fallback to global if local calculation is problematic
                                // effective_a and effective_cRecip remain as their _global_param defaults already set above
                                console.warn(`Viscosity solver: problematic local calculation at [${i},${j}]. Falling back to global viscosity for this cell. Multiplier: ${viscosityField[idx]}, Effective_a_attempt: ${temp_effective_a}`);
                            }
                        }
                        // If viscosityField doesn't exist or no value for idx, effective_a and effective_cRecip 
                        // remain as their global_param defaults, which is correct.
                    }
                    
                    x[idx] =
                        (x0[idx] +
                        effective_a * ( x[this.IX(i+1,j)] + x[this.IX(i-1,j)] +
                                      x[this.IX(i,j+1)] + x[this.IX(i,j-1)]
                                    )) * effective_cRecip; 
                }
            }
            this.set_bnd(b, x);
        }
    }

    diffuse(b, x_out, x_in, base_diff_rate, dt, field_type = 'density') { // Checkpoint 1 Signature
        const a_global = dt * base_diff_rate * (this.size - 2) * (this.size - 2); 
        // Pass new params; lin_solve will ignore field_type, base_diff_rate, dt for Checkpoint 1
        this.lin_solve(b, x_out, x_in, a_global, 1 + 4 * a_global, field_type, base_diff_rate, dt); 
    }

    project(velocX_in_out, velocY_in_out, p_temp, div_temp) {
        for (let j = 1; j < this.size - 1; j++) {
            for (let i = 1; i < this.size - 1; i++) {
                const idx = this.IX(i,j);
                div_temp[idx] = -0.5 * (
                    velocX_in_out[this.IX(i+1,j)] - velocX_in_out[this.IX(i-1,j)] +
                    velocY_in_out[this.IX(i,j+1)] - velocY_in_out[this.IX(i,j-1)]
                ) / this.size; // Using this.size here assumes square cells for divergence calculation
                p_temp[idx] = 0;
            }
        }
        this.set_bnd(0, div_temp);
        this.set_bnd(0, p_temp);
        this.lin_solve(0, p_temp, div_temp, 1, 4);

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
        const dtx = dt * (this.size - 2);
        const dty = dt * (this.size - 2);
        let s0, s1, t0, t1;
        let tmp1, tmp2, x, y;

        const N = this.size;

        for (let j_cell = 1; j_cell < N - 1; j_cell++) {
            for (let i_cell = 1; i_cell < N - 1; i_cell++) {
                const current_idx = this.IX(i_cell, j_cell);

                tmp1 = dtx * velocX_source[current_idx];
                tmp2 = dty * velocY_source[current_idx];
                x = i_cell - tmp1;
                y = j_cell - tmp2;

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
                    i1 = i0 + 1.0;

                    if (y < 0.5) y = 0.5;
                    if (y > N - 1.5) y = N - 1.5;
                    j0 = Math.floor(y);
                    j1 = j0 + 1.0;
                }

                s1 = x - i0;
                if (this.useWrapping && i1 === 0 && i0 === N - 1) s0 = 1.0 - s1 + 1; else s0 = 1.0 - s1;

                t1 = y - j0;
                if (this.useWrapping && j1 === 0 && j0 === N - 1) t0 = 1.0 - t1 + 1; else t0 = 1.0 - t1;

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

        this.project(this.Vx0, this.Vy0, this.Vx, this.Vy); // Vx, Vy are temp here

        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, this.dt);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, this.dt);
        this.clampVelocityComponents(this.Vx); // Clamp after advection
        this.clampVelocityComponents(this.Vy);

        this.project(this.Vx0, this.Vy0, this.Vx, this.Vy); // Vx0, Vy0 are temp here again, using them as p_temp and div_temp

        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, this.dt);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, this.dt);
        this.clampVelocityComponents(this.Vx); // Clamp after advection
        this.clampVelocityComponents(this.Vy);

        this.project(this.Vx, this.Vy, this.Vx0, this.Vy0); // Vx0, Vy0 are temp here again

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

    draw(ctxToDrawOn, targetWidth, targetHeight) {
        const roundedSize = Math.round(this.size);
         if (roundedSize <= 0 || !Number.isFinite(roundedSize)) {
            console.error("Invalid fluid field size for drawing (this.size):", this.size);
            return;
        }

        if (!offscreenFluidCanvas || offscreenFluidCanvas.width !== roundedSize) {
            offscreenFluidCanvas = document.createElement('canvas');
            offscreenFluidCanvas.width = roundedSize;
            offscreenFluidCanvas.height = roundedSize;
            offscreenFluidCtx = offscreenFluidCanvas.getContext('2d');
        }

        const imgData = offscreenFluidCtx.createImageData(roundedSize, roundedSize);
        const data = imgData.data;
        for (let i = 0; i < roundedSize * roundedSize; i++) {
            data[i * 4 + 0] = Math.min(255, Math.max(0, Math.floor(this.densityR[i])));
            data[i * 4 + 1] = Math.min(255, Math.max(0, Math.floor(this.densityG[i])));
            data[i * 4 + 2] = Math.min(255, Math.max(0, Math.floor(this.densityB[i])));
            data[i * 4 + 3] = (this.densityR[i] > 1 || this.densityG[i] > 1 || this.densityB[i] > 1) ? 255 : 0;
        }
        offscreenFluidCtx.putImageData(imgData, 0, 0);

        ctxToDrawOn.imageSmoothingEnabled = true;
        ctxToDrawOn.drawImage(offscreenFluidCanvas, 0, 0, roundedSize, roundedSize, 0, 0, targetWidth, targetHeight);
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