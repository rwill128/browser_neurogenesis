class Brain {
    constructor(softBody) {
        this.softBody = softBody;
        this.brainNode = null; // The MassPoint that acts as the brain
        this._findOrCreateBrainNode();
        this.initialize();
    }

    initialize() {
        if (this.brainNode && this.brainNode.neuronData && this.brainNode.neuronData.isBrain) {
            this._calculateBrainVectorSizes();
            this._initializeBrainWeightsAndBiases();
            this._initializeBrainRLComponents();
        }
    }

    _findOrCreateBrainNode() {
        // First, try to find an already designated brain
        for (const point of this.softBody.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                this.brainNode = point;
                return;
            }
        }

        // If no brain was pre-designated, find the first NEURON type node
        for (const point of this.softBody.massPoints) {
            if (point.nodeType === NodeType.NEURON) {
                if (!point.neuronData) { // Ensure neuronData exists
                    point.neuronData = {
                        hiddenLayerSize: DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                    };
                }
                point.neuronData.isBrain = true;
                this.brainNode = point;
                // Ensure other neurons are not brains
                this.softBody.massPoints.forEach(otherP => {
                    if (otherP !== point && otherP.nodeType === NodeType.NEURON && otherP.neuronData) {
                        otherP.neuronData.isBrain = false;
                    }
                });
                return;
            }
        }
    }

    _calculateBrainVectorSizes() {
        const nd = this.brainNode.neuronData;
        const body = this.softBody;
        nd.inputVectorSize = NEURAL_INPUT_SIZE_BASE +
                             (body.numEyeNodes * NEURAL_INPUTS_PER_EYE) +
                             (body.numSwimmerNodes * NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (body.numJetNodes * NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (body.springs.length * NEURAL_INPUTS_PER_SPRING_SENSOR);
        nd.outputVectorSize = (body.numEmitterNodes * NEURAL_OUTPUTS_PER_EMITTER) +
                              (body.numSwimmerNodes * NEURAL_OUTPUTS_PER_SWIMMER) +
                              (body.numEaterNodes * NEURAL_OUTPUTS_PER_EATER) +
                              (body.numPredatorNodes * NEURAL_OUTPUTS_PER_PREDATOR) +
                              (body.numJetNodes * NEURAL_OUTPUTS_PER_JET) +
                              (body.numPotentialGrabberNodes * NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) +
                              (body.numAttractorNodes * NEURAL_OUTPUTS_PER_ATTRACTOR) +
                              (body.numRepulsorNodes * NEURAL_OUTPUTS_PER_REPULSOR);
    }

    _initializeBrainWeightsAndBiases() {
        const nd = this.brainNode.neuronData;
        if (typeof nd.hiddenLayerSize !== 'number' || nd.hiddenLayerSize < DEFAULT_HIDDEN_LAYER_SIZE_MIN || nd.hiddenLayerSize > DEFAULT_HIDDEN_LAYER_SIZE_MAX) {
            nd.hiddenLayerSize = DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (DEFAULT_HIDDEN_LAYER_SIZE_MAX - DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1));
        }

        if (!nd.weightsIH || nd.weightsIH.length !== nd.hiddenLayerSize || (nd.weightsIH.length > 0 && nd.weightsIH[0].length !== nd.inputVectorSize) ) {
            nd.weightsIH = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize);
            nd.biasesH = initializeVector(nd.hiddenLayerSize);
        }

        if (!nd.weightsHO || nd.weightsHO.length !== nd.outputVectorSize || (nd.weightsHO.length > 0 && nd.weightsHO[0].length !== nd.hiddenLayerSize) ) {
            nd.weightsHO = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize);
            nd.biasesO = initializeVector(nd.outputVectorSize);
        }
    }

    _initializeBrainRLComponents() {
        const nd = this.brainNode.neuronData;
        if (!nd.experienceBuffer) nd.experienceBuffer = [];
        if (typeof nd.framesSinceLastTrain !== 'number') nd.framesSinceLastTrain = 0;
        if (typeof nd.previousEnergyForReward !== 'number') nd.previousEnergyForReward = this.softBody.creatureEnergy;
        if (typeof nd.previousEnergyChangeForNN !== 'number') nd.previousEnergyChangeForNN = 0;
        if (typeof nd.lastAvgNormalizedReward !== 'number') nd.lastAvgNormalizedReward = 0;
        if (typeof nd.maxExperienceBufferSize !== 'number') nd.maxExperienceBufferSize = 10;
        nd.currentFrameInputVectorWithLabels = [];
        nd.currentFrameActionDetails = [];
    }

    process(dt, fluidFieldRef, nutrientField, lightField) {
        if (!this.brainNode || !this.brainNode.neuronData || !this.brainNode.neuronData.isBrain) return;

        const nd = this.brainNode.neuronData;
        if (!nd.weightsIH || !nd.biasesH || !nd.weightsHO || !nd.biasesO) {
            this.softBody._applyFallbackBehaviors(dt, fluidFieldRef);
            return;
        }

        const inputVector = this._gatherBrainInputs(fluidFieldRef, nutrientField, lightField);
        this._propagateBrainOutputs(inputVector);
        this._applyBrainActionsToPoints(dt);
        this._updateBrainTrainingBuffer(inputVector);
        this._triggerBrainPolicyUpdateIfNeeded();
    }
    
    _gatherBrainInputs(fluidFieldRef, nutrientField, lightField) {
        const nd = this.brainNode.neuronData;
        const body = this.softBody;
        const inputVector = [];
        nd.currentFrameInputVectorWithLabels = [];

        const currentEnergyChange = body.creatureEnergy - (nd.previousEnergyForReward || body.creatureEnergy);
        const energySecondDerivative = currentEnergyChange - (nd.previousEnergyChangeForNN || 0);
        const normalizedEnergySecondDerivative = Math.tanh(energySecondDerivative / (body.currentMaxEnergy * 0.05 || 1));

        if (fluidFieldRef) {
            const brainGx = Math.floor(this.brainNode.pos.x / fluidFieldRef.scaleX);
            const brainGy = Math.floor(this.brainNode.pos.y / fluidFieldRef.scaleY);
            const brainIdx = fluidFieldRef.IX(brainGx, brainGy);
            const valR = (fluidFieldRef.densityR[brainIdx] || 0) / 255;
            const valG = (fluidFieldRef.densityG[brainIdx] || 0) / 255;
            const valB = (fluidFieldRef.densityB[brainIdx] || 0) / 255;
            inputVector.push(valR, valG, valB);
            nd.currentFrameInputVectorWithLabels.push({ label: 'Sensed Dye (R) @Brain', value: valR });
            nd.currentFrameInputVectorWithLabels.push({ label: 'Sensed Dye (G) @Brain', value: valG });
            nd.currentFrameInputVectorWithLabels.push({ label: 'Sensed Dye (B) @Brain', value: valB });
        } else {
            inputVector.push(0, 0, 0);
            nd.currentFrameInputVectorWithLabels.push({ label: 'Sensed Dye (R) @Brain', value: 0 });
            nd.currentFrameInputVectorWithLabels.push({ label: 'Sensed Dye (G) @Brain', value: 0 });
            nd.currentFrameInputVectorWithLabels.push({ label: 'Sensed Dye (B) @Brain', value: 0 });
        }
        const energyRatio = body.creatureEnergy / body.currentMaxEnergy;
        inputVector.push(energyRatio);
        nd.currentFrameInputVectorWithLabels.push({ label: 'Energy Ratio', value: energyRatio });

        const comPos = body.getAveragePosition();
        const relComPosX = (comPos.x - this.brainNode.pos.x) / WORLD_WIDTH;
        const relComPosY = (comPos.y - this.brainNode.pos.y) / WORLD_HEIGHT;
        inputVector.push(Math.tanh(relComPosX));
        inputVector.push(Math.tanh(relComPosY));
        nd.currentFrameInputVectorWithLabels.push({ label: 'Relative CoM Pos X', value: Math.tanh(relComPosX) });
        nd.currentFrameInputVectorWithLabels.push({ label: 'Relative CoM Pos Y', value: Math.tanh(relComPosY) });

        const comVel = body.getAverageVelocity();
        const normComVelX = Math.tanh(comVel.x / MAX_PIXELS_PER_FRAME_DISPLACEMENT);
        const normComVelY = Math.tanh(comVel.y / MAX_PIXELS_PER_FRAME_DISPLACEMENT);
        inputVector.push(normComVelX);
        inputVector.push(normComVelY);
        nd.currentFrameInputVectorWithLabels.push({ label: 'CoM Vel X', value: normComVelX });
        nd.currentFrameInputVectorWithLabels.push({ label: 'CoM Vel Y', value: normComVelY });

        if (nutrientField && fluidFieldRef) {
            const brainGx = Math.floor(this.brainNode.pos.x / fluidFieldRef.scaleX);
            const brainGy = Math.floor(this.brainNode.pos.y / fluidFieldRef.scaleY);
            const nutrientIdx = fluidFieldRef.IX(brainGx, brainGy);
            const currentNutrient = nutrientField[nutrientIdx] !== undefined ? nutrientField[nutrientIdx] : 1.0;
            const normalizedNutrient = (currentNutrient - MIN_NUTRIENT_VALUE) / (MAX_NUTRIENT_VALUE - MIN_NUTRIENT_VALUE);
            const finalNutrientVal = Math.max(0, Math.min(1, normalizedNutrient));
            inputVector.push(finalNutrientVal);
            nd.currentFrameInputVectorWithLabels.push({ label: 'Nutrient @Brain', value: finalNutrientVal });
        } else {
            inputVector.push(0.5);
            nd.currentFrameInputVectorWithLabels.push({ label: 'Nutrient @Brain', value: 0.5 });
        }

        inputVector.push(normalizedEnergySecondDerivative);
        nd.currentFrameInputVectorWithLabels.push({ label: 'Energy Δ-Rate', value: normalizedEnergySecondDerivative });

        body.springs.forEach((spring, i) => {
            const dx = spring.p1.pos.x - spring.p2.pos.x;
            const dy = spring.p1.pos.y - spring.p2.pos.y;
            const currentLength = Math.sqrt(dx*dx + dy*dy);
            const normalizedLength = Math.tanh((currentLength / spring.restLength) - 1.0);
            inputVector.push(normalizedLength);
            nd.currentFrameInputVectorWithLabels.push({ label: `Spring ${i} Length`, value: normalizedLength });
        });

        body.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.SWIMMER || point.nodeType === NodeType.JET) {
                const typeStr = point.nodeType === NodeType.SWIMMER ? 'Swimmer' : 'Jet';
                const sensedVx = Math.tanh(point.sensedFluidVelocity.x);
                const sensedVy = Math.tanh(point.sensedFluidVelocity.y);
                inputVector.push(sensedVx, sensedVy);
                nd.currentFrameInputVectorWithLabels.push({ label: `Fluid Vel X @${typeStr} P${pointIndex}`, value: sensedVx });
                nd.currentFrameInputVectorWithLabels.push({ label: `Fluid Vel Y @${typeStr} P${pointIndex}`, value: sensedVy });
            }
        });

        body.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.EYE) {
                const seesTargetVal = point.seesTarget ? 1 : 0;
                const targetMagVal = point.nearestTargetMagnitude;
                const targetDirVal = (point.nearestTargetDirection / (Math.PI * 2)) + 0.5;
                inputVector.push(seesTargetVal, targetMagVal, targetDirVal);
                nd.currentFrameInputVectorWithLabels.push({ label: `Eye @P${pointIndex} Sees Target`, value: seesTargetVal });
                nd.currentFrameInputVectorWithLabels.push({ label: `Eye @P${pointIndex} Target Dist`, value: targetMagVal });
                nd.currentFrameInputVectorWithLabels.push({ label: `Eye @P${pointIndex} Target Dir`, value: targetDirVal });
            }
        });
        
        while(inputVector.length < nd.inputVectorSize) { inputVector.push(0); }
        if(inputVector.length > nd.inputVectorSize) { inputVector.splice(nd.inputVectorSize); }
        
        nd.previousEnergyChangeForNN = currentEnergyChange;

        return inputVector;
    }
    
    _propagateBrainOutputs(inputVector) {
        const nd = this.brainNode.neuronData;
        const hiddenLayerInputs = multiplyMatrixVector(nd.weightsIH, inputVector);
        const hiddenLayerBiasedInputs = addVectors(hiddenLayerInputs, nd.biasesH);
        const hiddenLayerActivations = hiddenLayerBiasedInputs.map(val => Math.tanh(val));
        const outputLayerInputs = multiplyMatrixVector(nd.weightsHO, hiddenLayerActivations);
        nd.rawOutputs = addVectors(outputLayerInputs, nd.biasesO);
    }
    
    _applyBrainActionsToPoints(dt) {
        const nd = this.brainNode.neuronData;
        let currentRawOutputIndex = 0;
        nd.currentFrameActionDetails = [];

        function sampleAndLogAction(rawMean, rawStdDev) {
            const mean = rawMean;
            const stdDev = Math.exp(rawStdDev) + 1e-6;
            const sampledActionValue = sampleGaussian(mean, stdDev);
            const logProb = logPdfGaussian(sampledActionValue, mean, stdDev);
            return { detail: { mean, stdDev, sampledAction: sampledActionValue, logProb }, value: sampledActionValue };
        }

        this.softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.EMITTER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_EMITTER) {
                    const detailsForThisEmitter = [];
                    let localPairIdx = 0;
                    for (let i = 0; i < 3; i++) {
                        const res = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                        res.detail.label = `Emitter @P${pointIndex} ${['Red', 'Green', 'Blue'][i]}`;
                        detailsForThisEmitter.push(res.detail);
                        point.dyeColor[i] = sigmoid(res.value) * 255;
                    }
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    exertionRes.detail.label = `Emitter @P${pointIndex} Exertion`;
                    detailsForThisEmitter.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...detailsForThisEmitter);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EMITTER;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EMITTER;
                }
            } else if (point.nodeType === NodeType.SWIMMER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_SWIMMER) {
                    const details = [];
                    const magRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    magRes.detail.label = `Swimmer @P${pointIndex} Magnitude`;
                    details.push(magRes.detail);
                    const dirRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + 2], nd.rawOutputs[outputStartRawIdx + 3]);
                    dirRes.detail.label = `Swimmer @P${pointIndex} Direction`;
                    details.push(dirRes.detail);
                    
                    point.currentExertionLevel = sigmoid(magRes.value);
                    const finalMagnitude = point.currentExertionLevel * MAX_SWIMMER_OUTPUT_MAGNITUDE;
                    const angle = dirRes.value;
                    const force = new Vec2(Math.cos(angle) * finalMagnitude, Math.sin(angle) * finalMagnitude);
                    point.applyForce(force.div(dt));
                    
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_SWIMMER;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_SWIMMER;
                }
            } else if (point.nodeType === NodeType.EATER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_EATER) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Eater @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EATER;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_EATER;
                }
            } else if (point.nodeType === NodeType.PREDATOR) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_PREDATOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Predator @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_PREDATOR;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_PREDATOR;
                }
            } else if (point.nodeType === NodeType.JET) {
                 const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_JET) {
                    const details = [];
                    const magRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    magRes.detail.label = `Jet @P${pointIndex} Magnitude`;
                    details.push(magRes.detail);
                    const dirRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + 2], nd.rawOutputs[outputStartRawIdx + 3]);
                    dirRes.detail.label = `Jet @P${pointIndex} Direction`;
                    details.push(dirRes.detail);
                    
                    point.currentExertionLevel = sigmoid(magRes.value);
                    point.jetData.currentMagnitude = point.currentExertionLevel * MAX_JET_OUTPUT_MAGNITUDE;
                    point.jetData.currentAngle = dirRes.value;

                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_JET;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_JET;
                }
            } else if (point.nodeType === NodeType.ATTRACTOR) {
                 const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_ATTRACTOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Attractor @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_ATTRACTOR;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_ATTRACTOR;
                }
            } else if (point.nodeType === NodeType.REPULSOR) {
                 const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_REPULSOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Repulsor @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_REPULSOR;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_REPULSOR;
                }
            }
        });

        this.softBody.massPoints.forEach((point, pointIndex) => {
            if (point.canBeGrabber) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) {
                    const details = [];
                    const grabRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    grabRes.detail.label = `Grabber @P${pointIndex} Toggle`;
                    details.push(grabRes.detail);
                    point.isGrabbing = sigmoid(grabRes.value) > 0.5;
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_GRABBER_TOGGLE;
                } else {
                    currentRawOutputIndex += NEURAL_OUTPUTS_PER_GRABBER_TOGGLE;
                }
            }
        });
    }

    _updateBrainTrainingBuffer(inputVector) {
        const nd = this.brainNode.neuronData;
        if (nd.currentFrameActionDetails && nd.currentFrameActionDetails.length > 0) {
            let reward = this._calculateReward();
            nd.experienceBuffer.push({
                state: [...inputVector],
                actionDetails: JSON.parse(JSON.stringify(nd.currentFrameActionDetails)),
                reward: reward
            });
            if (nd.experienceBuffer.length > nd.maxExperienceBufferSize) {
                nd.experienceBuffer.shift();
            }
        }
        nd.previousEnergyForReward = this.softBody.creatureEnergy;
    }

    _calculateReward() {
        const nd = this.brainNode.neuronData;
        const body = this.softBody;
        let reward = 0;

        const findInputValue = (label) => {
            const entry = nd.currentFrameInputVectorWithLabels.find(item => item.label === label);
            return entry ? entry.value : 0;
        };
        const findAndAverageInputValues = (labelPrefix) => {
            const entries = nd.currentFrameInputVectorWithLabels.filter(item => item.label.startsWith(labelPrefix));
            if (entries.length === 0) return 0;
            return entries.reduce((acc, item) => acc + item.value, 0) / entries.length;
        };

        switch (body.rewardStrategy) {
            case RLRewardStrategy.ENERGY_CHANGE:
                reward = (body.creatureEnergy - nd.previousEnergyForReward) - body.energyGainedFromPhotosynthesisThisTick;
                break;
            case RLRewardStrategy.REPRODUCTION_EVENT:
                reward = body.justReproduced ? REPRODUCTION_REWARD_VALUE : 0;
                if(body.justReproduced) body.justReproduced = false;
                break;
            case RLRewardStrategy.PARTICLE_PROXIMITY:
                let minParticleMagnitude = 1.0;
                let particleSeen = body.massPoints.some(p => {
                    if (p.nodeType === NodeType.EYE && p.seesTarget) {
                        minParticleMagnitude = Math.min(minParticleMagnitude, p.nearestTargetMagnitude);
                        return true;
                    }
                    return false;
                });
                reward = particleSeen ? (1.0 - minParticleMagnitude) * PARTICLE_PROXIMITY_REWARD_SCALE : 0;
                break;
            case RLRewardStrategy.ENERGY_SECOND_DERIVATIVE:
                const currentEnergyChange = body.creatureEnergy - (nd.previousEnergyForReward || body.creatureEnergy);
                const energySecondDerivative = currentEnergyChange - (nd.previousEnergyChangeForNN || 0);
                reward = energySecondDerivative * ENERGY_SECOND_DERIVATIVE_REWARD_SCALE;
                break;
            case RLRewardStrategy.SENSED_DYE_R: reward = findInputValue('Sensed Dye (R) @Brain'); break;
            case RLRewardStrategy.SENSED_DYE_R_INV: reward = 1.0 - findInputValue('Sensed Dye (R) @Brain'); break;
            case RLRewardStrategy.SENSED_DYE_G: reward = findInputValue('Sensed Dye (G) @Brain'); break;
            case RLRewardStrategy.SENSED_DYE_G_INV: reward = 1.0 - findInputValue('Sensed Dye (G) @Brain'); break;
            case RLRewardStrategy.SENSED_DYE_B: reward = findInputValue('Sensed Dye (B) @Brain'); break;
            case RLRewardStrategy.SENSED_DYE_B_INV: reward = 1.0 - findInputValue('Sensed Dye (B) @Brain'); break;
            case RLRewardStrategy.ENERGY_RATIO: reward = findInputValue('Energy Ratio'); break;
            case RLRewardStrategy.ENERGY_RATIO_INV: reward = 1.0 - findInputValue('Energy Ratio'); break;
            case RLRewardStrategy.REL_COM_POS_X_POS: reward = Math.max(0, findInputValue('Relative CoM Pos X')); break;
            case RLRewardStrategy.REL_COM_POS_X_NEG: reward = Math.max(0, -findInputValue('Relative CoM Pos X')); break;
            case RLRewardStrategy.REL_COM_POS_Y_POS: reward = Math.max(0, findInputValue('Relative CoM Pos Y')); break;
            case RLRewardStrategy.REL_COM_POS_Y_NEG: reward = Math.max(0, -findInputValue('Relative CoM Pos Y')); break;
            case RLRewardStrategy.REL_COM_VEL_X_POS: reward = Math.max(0, findInputValue('CoM Vel X')); break;
            case RLRewardStrategy.REL_COM_VEL_X_NEG: reward = Math.max(0, -findInputValue('CoM Vel X')); break;
            case RLRewardStrategy.REL_COM_VEL_Y_POS: reward = Math.max(0, findInputValue('CoM Vel Y')); break;
            case RLRewardStrategy.REL_COM_VEL_Y_NEG: reward = Math.max(0, -findInputValue('CoM Vel Y')); break;
            case RLRewardStrategy.SENSED_NUTRIENT: reward = findInputValue('Nutrient @Brain'); break;
            case RLRewardStrategy.SENSED_NUTRIENT_INV: reward = 1.0 - findInputValue('Nutrient @Brain'); break;
            case RLRewardStrategy.AVG_SPRING_COMPRESSION: reward = Math.max(0, -findAndAverageInputValues('Spring')); break;
            case RLRewardStrategy.AVG_SPRING_EXTENSION: reward = Math.max(0, findAndAverageInputValues('Spring')); break;
            case RLRewardStrategy.AVG_FLUID_VEL_X_POS: reward = Math.max(0, findAndAverageInputValues('Fluid Vel X')); break;
            case RLRewardStrategy.AVG_FLUID_VEL_X_NEG: reward = Math.max(0, -findAndAverageInputValues('Fluid Vel X')); break;
            case RLRewardStrategy.AVG_FLUID_VEL_Y_POS: reward = Math.max(0, findAndAverageInputValues('Fluid Vel Y')); break;
            case RLRewardStrategy.AVG_FLUID_VEL_Y_NEG: reward = Math.max(0, -findAndAverageInputValues('Fluid Vel Y')); break;
            case RLRewardStrategy.EYE_SEES_TARGET:
                const seesTargetValues = nd.currentFrameInputVectorWithLabels.filter(item => item.label.endsWith('Sees Target')).map(i => i.value);
                reward = seesTargetValues.some(v => v > 0) ? 1.0 : 0.0;
                break;
            case RLRewardStrategy.EYE_TARGET_PROXIMITY:
                const distValues = nd.currentFrameInputVectorWithLabels.filter(item => item.label.endsWith('Target Dist') && item.value > 0).map(i => i.value);
                if (distValues.length > 0) reward = 1.0 - Math.min(...distValues);
                break;
            case RLRewardStrategy.EYE_TARGET_DISTANCE:
                const distValuesFar = nd.currentFrameInputVectorWithLabels.filter(item => item.label.endsWith('Target Dist') && item.value > 0).map(i => i.value);
                if (distValuesFar.length > 0) reward = Math.min(...distValuesFar);
                break;
            default:
                reward = (body.creatureEnergy - nd.previousEnergyForReward) - body.energyGainedFromPhotosynthesisThisTick;
        }
        return reward;
    }

    _triggerBrainPolicyUpdateIfNeeded() {
        const nd = this.brainNode.neuronData;
        nd.framesSinceLastTrain++;
        if (nd.framesSinceLastTrain >= TRAINING_INTERVAL_FRAMES) {
            this.updateBrainPolicy();
        }
    }

    updateBrainPolicy() {
        const nd = this.brainNode.neuronData;
        if (!nd.experienceBuffer || nd.experienceBuffer.length < nd.maxExperienceBufferSize) return;

        const batch = nd.experienceBuffer;
        const rewards = batch.map(exp => exp.reward);
        const discountedRewards = this.calculateDiscountedRewards(rewards, DISCOUNT_FACTOR_GAMMA);

        let mean = discountedRewards.reduce((a, b) => a + b, 0) / discountedRewards.length;
        let stdDev = Math.sqrt(discountedRewards.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / discountedRewards.length);
        const normalizedRewards = discountedRewards.map(r => (r - mean) / (stdDev + 1e-6));
        
        nd.lastAvgNormalizedReward = mean;

        const gradWeightsIH_acc = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize, 0);
        const gradBiasesH_acc = initializeVector(nd.hiddenLayerSize, 0);
        const gradWeightsHO_acc = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize, 0);
        const gradBiasesO_acc = initializeVector(nd.outputVectorSize, 0);

        for (let t = 0; t < batch.length; t++) {
            const { state, actionDetails } = batch[t];
            const G_t = normalizedRewards[t];

            const hiddenActivations = multiplyMatrixVector(nd.weightsIH, state).map((v, i) => Math.tanh(v + nd.biasesH[i]));

            let d_log_p_d_outputs = [];
            actionDetails.forEach(ad => {
                const { sampledAction, mean, stdDev } = ad;
                const d_log_p_d_mean = (sampledAction - mean) / (stdDev * stdDev + 1e-9);
                const d_log_p_d_raw_std = (((sampledAction - mean) ** 2) / (stdDev ** 3 + 1e-9) - 1 / (stdDev + 1e-9)) * (stdDev - 1e-6);
                d_log_p_d_outputs.push(d_log_p_d_mean, d_log_p_d_raw_std);
            });
            
            for (let j = 0; j < nd.outputVectorSize; j++) {
                const grad_output_j = G_t * (d_log_p_d_outputs[j] || 0);
                for (let k = 0; k < nd.hiddenLayerSize; k++) {
                    gradWeightsHO_acc[j][k] += grad_output_j * hiddenActivations[k];
                }
                gradBiasesO_acc[j] += grad_output_j;
            }

            const d_tanh = (x) => 1 - x * x;
            for (let h = 0; h < nd.hiddenLayerSize; h++) {
                let error_h = 0;
                for (let j = 0; j < nd.outputVectorSize; j++) {
                    error_h += G_t * (d_log_p_d_outputs[j] || 0) * nd.weightsHO[j][h];
                }
                error_h *= d_tanh(hiddenActivations[h]);
                for (let i = 0; i < nd.inputVectorSize; i++) {
                    gradWeightsIH_acc[h][i] += error_h * state[i];
                }
                gradBiasesH_acc[h] += error_h;
            }
        }

        const batchSize = batch.length;
        for(let j=0; j<nd.outputVectorSize; ++j) {
            for(let k=0; k<nd.hiddenLayerSize; ++k) nd.weightsHO[j][k] += LEARNING_RATE * gradWeightsHO_acc[j][k] / batchSize;
            nd.biasesO[j] += LEARNING_RATE * gradBiasesO_acc[j] / batchSize;
        }
        for(let h=0; h<nd.hiddenLayerSize; ++h) {
            for(let i=0; i<nd.inputVectorSize; ++i) nd.weightsIH[h][i] += LEARNING_RATE * gradWeightsIH_acc[h][i] / batchSize;
            nd.biasesH[h] += LEARNING_RATE * gradBiasesH_acc[h] / batchSize;
        }

        nd.experienceBuffer = [];
        nd.framesSinceLastTrain = 0;
    }

    calculateDiscountedRewards(rewards, gamma) {
        const discounted = new Array(rewards.length);
        let running_add = 0;
        for (let t = rewards.length - 1; t >= 0; t--) {
            running_add = rewards[t] + gamma * running_add;
            discounted[t] = running_add;
        }
        return discounted;
    }
} 