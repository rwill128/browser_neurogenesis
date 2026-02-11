import {NodeType, RLRewardStrategy} from "./constants.js";
import config from '../config.js';
import {
    addVectors,
    initializeMatrix,
    initializeVector,
    logPdfGaussian,
    multiplyMatrixVector,
    sampleGaussian, sigmoid, Vec2
} from "../utils.js";

export class Brain {
    constructor(softBody) {
        this.softBody = softBody;
        this.brainNode = null;
        this.initialize();
    }

    initialize() {
        this.brainNode = this._findOrCreateBrainNode();

        if (this.brainNode && this.brainNode.neuronData && this.brainNode.neuronData.isBrain) {
            this._calculateBrainVectorSizes();
            this._initializeBrainWeightsAndBiases();
            this._initializeBrainRLComponents();
        }
    }

    _findOrCreateBrainNode() {
        let brainNode = null;
        // First, try to find an already designated brain
        for (const point of this.softBody.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        // If no brain was pre-designated, find the first NEURON type node
        if (!brainNode) {
            for (const point of this.softBody.massPoints) {
                if (point.nodeType === NodeType.NEURON) {
                    if (!point.neuronData) { // Ensure neuronData exists
                        point.neuronData = {
                            hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                        };
                    }
                    point.neuronData.isBrain = true;
                    brainNode = point;
                    // Ensure other neurons are not brains
                    this.softBody.massPoints.forEach(otherP => {
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

    _calculateBrainVectorSizes() {
        const nd = this.brainNode.neuronData;
        const softBody = this.softBody;

        const numEmitterPoints = softBody.numEmitterNodes;
        const numSwimmerPoints = softBody.numSwimmerNodes;
        const numEaterPoints = softBody.numEaterNodes;
        const numPredatorPoints = softBody.numPredatorNodes;
        const numEyeNodes = softBody.numEyeNodes;
        const numJetNodes = softBody.numJetNodes;
        const numPotentialGrabberPoints = softBody.numPotentialGrabberNodes;
        const numAttractorPoints = softBody.numAttractorNodes;
        const numRepulsorPoints = softBody.numRepulsorNodes;

        nd.inputVectorSize = config.NEURAL_INPUT_SIZE_BASE +
                             (numEyeNodes * config.NEURAL_INPUTS_PER_EYE) +
                             (numSwimmerPoints * config.NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (numJetNodes * config.NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (softBody.springs.length * config.NEURAL_INPUTS_PER_SPRING_SENSOR);

        nd.outputVectorSize = (numEmitterPoints * config.NEURAL_OUTPUTS_PER_EMITTER) +
                              (numSwimmerPoints * config.NEURAL_OUTPUTS_PER_SWIMMER) +
                              (numEaterPoints * config.NEURAL_OUTPUTS_PER_EATER) +
                              (numPredatorPoints * config.NEURAL_OUTPUTS_PER_PREDATOR) +
                              (numJetNodes * config.NEURAL_OUTPUTS_PER_JET) +
                              (numPotentialGrabberPoints * config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) +
                              (numAttractorPoints * config.NEURAL_OUTPUTS_PER_ATTRACTOR) +
                              (numRepulsorPoints * config.NEURAL_OUTPUTS_PER_REPULSOR);
    }

    /**
     * Resize NN tensors while preserving overlapping learned parameters.
     *
     * Existing rows/cols keep their values; only newly introduced slices are
     * random-initialized. This supports topology growth without wiping policy.
     */
    _initializeBrainWeightsAndBiases() {
        const nd = this.brainNode.neuronData;
        if (typeof nd.hiddenLayerSize !== 'number' || nd.hiddenLayerSize < config.DEFAULT_HIDDEN_LAYER_SIZE_MIN || nd.hiddenLayerSize > config.DEFAULT_HIDDEN_LAYER_SIZE_MAX) {
            nd.hiddenLayerSize = config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1));
        }

        nd.weightsIH = this._resizeMatrixPreserve(nd.weightsIH, nd.hiddenLayerSize, nd.inputVectorSize);
        nd.biasesH = this._resizeVectorPreserve(nd.biasesH, nd.hiddenLayerSize);

        nd.weightsHO = this._resizeMatrixPreserve(nd.weightsHO, nd.outputVectorSize, nd.hiddenLayerSize);
        nd.biasesO = this._resizeVectorPreserve(nd.biasesO, nd.outputVectorSize);
    }

    /**
     * Helper: resize matrix with value preservation on overlapping region.
     */
    _resizeMatrixPreserve(existingMatrix, targetRows, targetCols) {
        const resized = initializeMatrix(targetRows, targetCols);
        if (!Array.isArray(existingMatrix) || existingMatrix.length === 0) {
            return resized;
        }

        const rowsToCopy = Math.min(targetRows, existingMatrix.length);
        for (let r = 0; r < rowsToCopy; r++) {
            const oldRow = Array.isArray(existingMatrix[r]) ? existingMatrix[r] : [];
            const colsToCopy = Math.min(targetCols, oldRow.length);
            for (let c = 0; c < colsToCopy; c++) {
                const oldVal = Number(oldRow[c]);
                if (Number.isFinite(oldVal)) {
                    resized[r][c] = oldVal;
                }
            }
        }
        return resized;
    }

    /**
     * Helper: resize vector with value preservation on overlapping prefix.
     */
    _resizeVectorPreserve(existingVector, targetSize) {
        const resized = initializeVector(targetSize);
        if (!Array.isArray(existingVector) || existingVector.length === 0) {
            return resized;
        }

        const copy = Math.min(targetSize, existingVector.length);
        for (let i = 0; i < copy; i++) {
            const oldVal = Number(existingVector[i]);
            if (Number.isFinite(oldVal)) {
                resized[i] = oldVal;
            }
        }
        return resized;
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
        if (!this.brainNode) {
            this.softBody._applyFallbackBehaviors(dt, fluidFieldRef);
            return;
        }

        const nd = this.brainNode.neuronData;
        if (!nd || !nd.weightsIH || !nd.biasesH || !nd.weightsHO || !nd.biasesO ||
            typeof nd.inputVectorSize !== 'number' ||
            typeof nd.hiddenLayerSize !== 'number' ||
            typeof nd.outputVectorSize !== 'number') {
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
        const softBody = this.softBody;
        const inputVector = [];
        nd.currentFrameInputVectorWithLabels = []; // Initialize/clear for this frame

        const currentEnergyChange = softBody.creatureEnergy - (nd.previousEnergyForReward || softBody.creatureEnergy);
        const energySecondDerivative = currentEnergyChange - (nd.previousEnergyChangeForNN || 0);
        const normalizedEnergySecondDerivative = Math.tanh(energySecondDerivative / (softBody.currentMaxEnergy * 0.05 || 1));

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
        const energyRatio = softBody.creatureEnergy / softBody.currentMaxEnergy;
        inputVector.push(energyRatio);
        nd.currentFrameInputVectorWithLabels.push({ label: 'Energy Ratio', value: energyRatio });

        const comPos = softBody.getAveragePosition();
        const relComPosX = (comPos.x - this.brainNode.pos.x) / config.WORLD_WIDTH;
        const relComPosY = (comPos.y - this.brainNode.pos.y) / config.WORLD_HEIGHT;
        inputVector.push(Math.tanh(relComPosX));
        inputVector.push(Math.tanh(relComPosY));
        nd.currentFrameInputVectorWithLabels.push({ label: 'Relative CoM Pos X', value: Math.tanh(relComPosX) });
        nd.currentFrameInputVectorWithLabels.push({ label: 'Relative CoM Pos Y', value: Math.tanh(relComPosY) });

        const comVel = softBody.getAverageVelocity();
        const normComVelX = Math.tanh(comVel.x / config.MAX_PIXELS_PER_FRAME_DISPLACEMENT);
        const normComVelY = Math.tanh(comVel.y / config.MAX_PIXELS_PER_FRAME_DISPLACEMENT);
        inputVector.push(normComVelX);
        inputVector.push(normComVelY);
        nd.currentFrameInputVectorWithLabels.push({ label: 'CoM Vel X', value: normComVelX });
        nd.currentFrameInputVectorWithLabels.push({ label: 'CoM Vel Y', value: normComVelY });

        if (nutrientField && fluidFieldRef) {
            const brainGx = Math.floor(this.brainNode.pos.x / fluidFieldRef.scaleX);
            const brainGy = Math.floor(this.brainNode.pos.y / fluidFieldRef.scaleY);
            const nutrientIdx = fluidFieldRef.IX(brainGx, brainGy);
            const currentNutrient = nutrientField[nutrientIdx] !== undefined ? nutrientField[nutrientIdx] : 1.0;
            const normalizedNutrient = (currentNutrient - config.MIN_NUTRIENT_VALUE) / (config.MAX_NUTRIENT_VALUE - config.MIN_NUTRIENT_VALUE);
            const finalNutrientVal = Math.max(0, Math.min(1, normalizedNutrient));
            inputVector.push(finalNutrientVal);
            nd.currentFrameInputVectorWithLabels.push({ label: 'Nutrient @Brain', value: finalNutrientVal });
        } else {
            inputVector.push(0.5);
            nd.currentFrameInputVectorWithLabels.push({ label: 'Nutrient @Brain', value: 0.5 });
        }

        inputVector.push(normalizedEnergySecondDerivative);
        nd.currentFrameInputVectorWithLabels.push({ label: 'Energy Δ-Rate', value: normalizedEnergySecondDerivative });

        softBody.springs.forEach((spring, i) => {
            const dx = spring.p1.pos.x - spring.p2.pos.x;
            const dy = spring.p1.pos.y - spring.p2.pos.y;
            const currentLength = Math.sqrt(dx*dx + dy*dy);
            const normalizedLength = Math.tanh((currentLength / spring.restLength) - 1.0);
            inputVector.push(normalizedLength);
            nd.currentFrameInputVectorWithLabels.push({ label: `Spring ${i} Length`, value: normalizedLength });
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.SWIMMER || point.nodeType === NodeType.JET) {
                const typeStr = point.nodeType === NodeType.SWIMMER ? 'Swimmer' : 'Jet';
                const sensedVx = Math.tanh(point.sensedFluidVelocity.x);
                const sensedVy = Math.tanh(point.sensedFluidVelocity.y);
                inputVector.push(sensedVx, sensedVy);
                nd.currentFrameInputVectorWithLabels.push({ label: `Fluid Vel X @${typeStr} P${pointIndex}`, value: sensedVx });
                nd.currentFrameInputVectorWithLabels.push({ label: `Fluid Vel Y @${typeStr} P${pointIndex}`, value: sensedVy });
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
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
        const rawOutputs = addVectors(outputLayerInputs, nd.biasesO);
        nd.rawOutputs = rawOutputs;
    }

    _applyBrainActionsToPoints(dt) {
        const nd = this.brainNode.neuronData;
        const softBody = this.softBody;

        let currentNumEmitterPoints = 0;
        let currentNumSwimmerPoints = 0;
        let currentNumEaterPoints = 0;
        let currentNumPredatorPoints = 0;
        let currentNumPotentialGrabberPoints = 0;
        let currentNumJetPoints = 0;
        let currentNumAttractorPoints = 0;
        let currentNumRepulsorPoints = 0;
        softBody.massPoints.forEach(p => {
            if (p.nodeType === NodeType.EMITTER) currentNumEmitterPoints++;
            else if (p.nodeType === NodeType.SWIMMER) currentNumSwimmerPoints++;
            else if (p.nodeType === NodeType.EATER) currentNumEaterPoints++;
            else if (p.nodeType === NodeType.PREDATOR) currentNumPredatorPoints++;
            else if (p.nodeType === NodeType.JET) currentNumJetPoints++;
            else if (p.nodeType === NodeType.ATTRACTOR) currentNumAttractorPoints++;
            else if (p.nodeType === NodeType.REPULSOR) currentNumRepulsorPoints++;
            if (p.canBeGrabber) currentNumPotentialGrabberPoints++;
        });

        const recalculatedOutputVectorSize = (currentNumEmitterPoints * config.NEURAL_OUTPUTS_PER_EMITTER) +
                                           (currentNumSwimmerPoints * config.NEURAL_OUTPUTS_PER_SWIMMER) +
                                           (currentNumEaterPoints * config.NEURAL_OUTPUTS_PER_EATER) +
                                           (currentNumPredatorPoints * config.NEURAL_OUTPUTS_PER_PREDATOR) +
                                           (currentNumJetPoints * config.NEURAL_OUTPUTS_PER_JET) +
                                           (currentNumPotentialGrabberPoints * config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) +
                                           (currentNumAttractorPoints * config.NEURAL_OUTPUTS_PER_ATTRACTOR) +
                                           (currentNumRepulsorPoints * config.NEURAL_OUTPUTS_PER_REPULSOR);

        if (nd.outputVectorSize !== recalculatedOutputVectorSize) {
            // console.warn(`Body ${softBody.id} _applyBrainActionsToPoints: MISMATCH between stored nd.outputVectorSize (${nd.outputVectorSize}) and recalculatedOutputVectorSize (${recalculatedOutputVectorSize}) based on current points.`);
        }

        let currentRawOutputIndex = 0;
        nd.currentFrameActionDetails = [];

        function sampleAndLogAction(rawMean, rawStdDev) {
            const mean = rawMean;
            const stdDev = Math.exp(rawStdDev) + 1e-6;
            const sampledActionValue = sampleGaussian(mean, stdDev);
            const logProb = logPdfGaussian(sampledActionValue, mean, stdDev);
            return { detail: { mean, stdDev, sampledAction: sampledActionValue, logProb }, value: sampledActionValue };
        }

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.EMITTER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_EMITTER) {
                    const detailsForThisEmitter = [];
                    let localPairIdx = 0;
                    for (let i = 0; i < 3; i++) {
                        const res = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                        const channel = ['Red', 'Green', 'Blue'][i];
                        res.detail.label = `Emitter @P${pointIndex} ${channel}`;
                        detailsForThisEmitter.push(res.detail);
                        point.dyeColor[i] = sigmoid(res.value) * 255;
                    }
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    exertionRes.detail.label = `Emitter @P${pointIndex} Exertion`;
                    detailsForThisEmitter.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...detailsForThisEmitter);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_EMITTER;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_EMITTER;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.SWIMMER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_SWIMMER) {
                    const detailsForThisSwimmer = [];
                    let localPairIdx = 0;

                    const magnitudeResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    magnitudeResult.detail.label = `Swimmer @P${pointIndex} Magnitude`;
                    detailsForThisSwimmer.push(magnitudeResult.detail);
                    const rawMagnitude = magnitudeResult.value;

                    const directionResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    directionResult.detail.label = `Swimmer @P${pointIndex} Direction`;
                    detailsForThisSwimmer.push(directionResult.detail);
                    const angle = directionResult.value;

                    point.currentExertionLevel = sigmoid(rawMagnitude);

                    const finalMagnitude = point.currentExertionLevel * config.MAX_SWIMMER_OUTPUT_MAGNITUDE;
                    const appliedForceX = finalMagnitude * Math.cos(angle);
                    const appliedForceY = finalMagnitude * Math.sin(angle);

                    point.applyForce(new Vec2(appliedForceX / dt, appliedForceY / dt));
                    nd.currentFrameActionDetails.push(...detailsForThisSwimmer);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_SWIMMER;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_SWIMMER;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.EATER) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_EATER) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Eater @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_EATER;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_EATER;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.PREDATOR) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_PREDATOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Predator @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_PREDATOR;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_PREDATOR;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.JET) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_JET) {
                    const detailsForThisJet = [];
                    let localPairIdx = 0;

                    const magnitudeResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    magnitudeResult.detail.label = `Jet @P${pointIndex} Magnitude`;
                    detailsForThisJet.push(magnitudeResult.detail);
                    const rawMagnitude = magnitudeResult.value;

                    const directionResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx + localPairIdx++], nd.rawOutputs[outputStartRawIdx + localPairIdx++]);
                    directionResult.detail.label = `Jet @P${pointIndex} Direction`;
                    detailsForThisJet.push(directionResult.detail);
                    const angle = directionResult.value;

                    point.currentExertionLevel = sigmoid(rawMagnitude);
                    point.jetData.currentMagnitude = point.currentExertionLevel * config.MAX_JET_OUTPUT_MAGNITUDE;
                    point.jetData.currentAngle = angle;

                    nd.currentFrameActionDetails.push(...detailsForThisJet);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_JET;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_JET;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.ATTRACTOR) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_ATTRACTOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Attractor @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_ATTRACTOR;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_ATTRACTOR;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.nodeType === NodeType.REPULSOR) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_REPULSOR) {
                    const details = [];
                    const exertionRes = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    exertionRes.detail.label = `Repulsor @P${pointIndex} Exertion`;
                    details.push(exertionRes.detail);
                    point.currentExertionLevel = sigmoid(exertionRes.value);
                    nd.currentFrameActionDetails.push(...details);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_REPULSOR;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_REPULSOR;
                }
            }
        });

        softBody.massPoints.forEach((point, pointIndex) => {
            if (point.canBeGrabber) {
                const outputStartRawIdx = currentRawOutputIndex;
                if (nd.rawOutputs && nd.rawOutputs.length >= outputStartRawIdx + config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) {
                    const detailsForThisGrab = [];
                    const grabToggleResult = sampleAndLogAction(nd.rawOutputs[outputStartRawIdx], nd.rawOutputs[outputStartRawIdx + 1]);
                    grabToggleResult.detail.label = `Grabber @P${pointIndex} Toggle`;
                    detailsForThisGrab.push(grabToggleResult.detail);
                    point.isGrabbing = sigmoid(grabToggleResult.value) > 0.5;

                    nd.currentFrameActionDetails.push(...detailsForThisGrab);
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE;
                } else {
                    currentRawOutputIndex += config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE;
                }
            }
        });
    }

    _updateBrainTrainingBuffer(inputVector) {
        const nd = this.brainNode.neuronData;
        const softBody = this.softBody;

        if (nd.currentFrameActionDetails && nd.currentFrameActionDetails.length > 0) {
            let reward = 0;

            const findInputValue = (label) => {
                const entry = nd.currentFrameInputVectorWithLabels.find(item => item.label === label);
                return entry ? entry.value : 0;
            };
            const findAndAverageInputValues = (labelPrefix) => {
                const entries = nd.currentFrameInputVectorWithLabels.filter(item => item.label.startsWith(labelPrefix));
                if (entries.length === 0) return 0;
                const sum = entries.reduce((acc, item) => acc + item.value, 0);
                return sum / entries.length;
            };

            switch (softBody.rewardStrategy) {
                case RLRewardStrategy.ENERGY_CHANGE:
                    reward = (softBody.creatureEnergy - nd.previousEnergyForReward) - softBody.energyGainedFromPhotosynthesisThisTick;
                    break;
                case RLRewardStrategy.REPRODUCTION_EVENT:
                    if (softBody.justReproduced) {
                        reward = config.REPRODUCTION_REWARD_VALUE;
                        softBody.justReproduced = false;
                    } else {
                        reward = 0;
                    }
                    break;
                case RLRewardStrategy.PARTICLE_PROXIMITY:
                    let minParticleMagnitude = 1.0;
                    let particleSeenByAnyEye = false;
                    softBody.massPoints.forEach(point => {
                        if (point.nodeType === NodeType.EYE && point.seesTarget) {
                            particleSeenByAnyEye = true;
                            if (point.nearestTargetMagnitude < minParticleMagnitude) {
                                minParticleMagnitude = point.nearestTargetMagnitude;
                            }
                        }
                    });
                    if (particleSeenByAnyEye) {
                        reward = (1.0 - minParticleMagnitude) * config.PARTICLE_PROXIMITY_REWARD_SCALE;
                    } else {
                        reward = 0;
                    }
                    break;
                case RLRewardStrategy.ENERGY_SECOND_DERIVATIVE:
                    const currentEnergyChangeForReward = softBody.creatureEnergy - (nd.previousEnergyForReward || softBody.creatureEnergy);
                    const energySecondDerivativeForReward = currentEnergyChangeForReward - (nd.previousEnergyChangeForNN || 0);
                    reward = energySecondDerivativeForReward * config.ENERGY_SECOND_DERIVATIVE_REWARD_SCALE;
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

                case RLRewardStrategy.AVG_SPRING_EXTENSION: reward = Math.max(0, findAndAverageInputValues('Spring')); break;
                case RLRewardStrategy.AVG_SPRING_COMPRESSION: reward = Math.max(0, -findAndAverageInputValues('Spring')); break;

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
                    reward = (softBody.creatureEnergy - nd.previousEnergyForReward) - softBody.energyGainedFromPhotosynthesisThisTick; // Fallback
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
        nd.previousEnergyForReward = softBody.creatureEnergy;
    }

    _triggerBrainPolicyUpdateIfNeeded() {
        const nd = this.brainNode.neuronData;
        nd.framesSinceLastTrain++;
        if (nd.framesSinceLastTrain >= config.TRAINING_INTERVAL_FRAMES) {
            this.updateBrainPolicy();
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
        if (!this.brainNode) return;
        const nd = this.brainNode.neuronData;

        if (!nd.experienceBuffer || nd.experienceBuffer.length < nd.maxExperienceBufferSize) {
            return;
        }

        const states = nd.experienceBuffer.map(exp => exp.state);
        const actionDetailsBatch = nd.experienceBuffer.map(exp => exp.actionDetails);
        const rewards = nd.experienceBuffer.map(exp => exp.reward);

        const discountedRewards = this.calculateDiscountedRewards(rewards, config.DISCOUNT_FACTOR_GAMMA);

        let meanDiscountedReward = 0;
        for (const r of discountedRewards) meanDiscountedReward += r;
        meanDiscountedReward /= discountedRewards.length;

        let stdDevDiscountedReward = 0;
        for (const r of discountedRewards) stdDevDiscountedReward += (r - meanDiscountedReward) ** 2;
        stdDevDiscountedReward = Math.sqrt(stdDevDiscountedReward / discountedRewards.length);

        const normalizedDiscountedRewards = discountedRewards.map(
            r => (r - meanDiscountedReward) / (stdDevDiscountedReward + 1e-6)
        );

        nd.lastAvgNormalizedReward = meanDiscountedReward;

        const gradWeightsHO_acc = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize, 0);
        const gradBiasesO_acc = initializeVector(nd.outputVectorSize, 0);
        const gradWeightsIH_acc = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize, 0);
        const gradBiasesH_acc = initializeVector(nd.hiddenLayerSize, 0);

        for (let t = 0; t < nd.experienceBuffer.length; t++) {
            const state_t = states[t];
            const actionDetails_t = actionDetailsBatch[t];
            const G_t_normalized = normalizedDiscountedRewards[t];

            const hiddenLayerInputs_t = multiplyMatrixVector(nd.weightsIH, state_t);
            const hiddenLayerBiasedInputs_t = addVectors(hiddenLayerInputs_t, nd.biasesH);
            const hiddenActivations_t = hiddenLayerBiasedInputs_t.map(val => Math.tanh(val));

            let currentActionDetailIdx = 0;
            for (let i = 0; i < nd.outputVectorSize / 2; i++) {
                const meanOutputIdx = i * 2;
                const stdDevOutputIdx = i * 2 + 1;

                const ad = actionDetails_t[currentActionDetailIdx];

                if (!ad) { 
                    currentActionDetailIdx++;
                    continue; 
                }
                currentActionDetailIdx++;

                const { sampledAction, mean, stdDev } = ad;
                const grad_logProb_d_mean = (sampledAction - mean) / (stdDev * stdDev + 1e-9); 
                const grad_logProb_d_stdDev_output = (((sampledAction - mean) ** 2) - (stdDev * stdDev)) / (stdDev * stdDev * stdDev + 1e-9);

                const error_mean = G_t_normalized * grad_logProb_d_mean;
                const error_stdDev = G_t_normalized * grad_logProb_d_stdDev_output * (stdDev - 1e-6);

                for (let k = 0; k < nd.hiddenLayerSize; k++) {
                    gradWeightsHO_acc[meanOutputIdx][k] += error_mean * hiddenActivations_t[k];
                    gradWeightsHO_acc[stdDevOutputIdx][k] += error_stdDev * hiddenActivations_t[k];
                }
                gradBiasesO_acc[meanOutputIdx] += error_mean;
                gradBiasesO_acc[stdDevOutputIdx] += error_stdDev;
            }

            const d_tanh = (x) => 1 - x * x;
            for (let h = 0; h < nd.hiddenLayerSize; h++) {
                let sum_error_weighted_HO = 0;
                for (let j = 0; j < nd.outputVectorSize; j++) {
                    const actionDetailIndex = Math.floor(j/2);
                    if (actionDetailIndex >= actionDetails_t.length) {
                        continue; 
                    }

                    const currentActionDetail = actionDetails_t[actionDetailIndex];
                    if (!currentActionDetail) {
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

        const batchSize = nd.experienceBuffer.length;
        for (let j = 0; j < nd.outputVectorSize; j++) {
            for (let k = 0; k < nd.hiddenLayerSize; k++) {
                nd.weightsHO[j][k] += config.LEARNING_RATE * gradWeightsHO_acc[j][k] / batchSize;
            }
            nd.biasesO[j] += config.LEARNING_RATE * gradBiasesO_acc[j] / batchSize;
        }
        for (let h = 0; h < nd.hiddenLayerSize; h++) {
            for (let i = 0; i < nd.inputVectorSize; i++) {
                nd.weightsIH[h][i] += config.LEARNING_RATE * gradWeightsIH_acc[h][i] / batchSize;
            }
            nd.biasesH[h] += config.LEARNING_RATE * gradBiasesH_acc[h] / batchSize;
        }

        nd.experienceBuffer = [];
        nd.framesSinceLastTrain = 0;
    }
} 
