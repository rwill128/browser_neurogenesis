import config from './config.js';
import {
    initializeSpatialGrid,
    initializePopulation,
    initFluidSimulation,
    initNutrientMap,
    initLightMap,
    initViscosityMap,
    initParticles,
    softBodyPopulation,
    particles,
    fluidField,
    spatialGrid,
    mutationStats, globalEnergyGains, globalEnergyCosts,
    saveCurrentWorldSnapshot,
    loadWorldFromSnapshot
} from './simulation.js';
import { perlin, getNodeTypeString, getRewardStrategyString, getEyeTargetTypeString, getMovementTypeString, sigmoid } from './utils.js';
import {NodeType} from "./classes/constants.js";
import viewport from './viewport.js';
import {
    displayToWorld,
    solveViewOffsetForAnchor,
    clampCameraOffsets,
    buildFitWorldCamera,
    centerCameraOnPoint
} from './engine/cameraMath.mjs';

// --- DOM Element Selections ---
const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d', { alpha: false });
const webgpuCanvas = document.getElementById('webgpuFluidCanvas');
let offscreenFluidCanvas, offscreenFluidCtx;

const worldWidthInput = document.getElementById('worldWidthInput');
const worldHeightInput = document.getElementById('worldHeightInput');
const resizeWorldButton = document.getElementById('resizeWorldButton');

const worldWrapToggle = document.getElementById('worldWrapToggle');
const maxTimestepSlider = document.getElementById('maxTimestep');
const maxTimestepValueSpan = document.getElementById('maxTimestepValue');
const zoomSensitivitySlider = document.getElementById('zoomSensitivitySlider');
const zoomSensitivityValueSpan = document.getElementById('zoomSensitivityValueSpan');
const pauseResumeButton = document.getElementById('pauseResumeButton');
const toggleControlsButton = document.getElementById('toggleControlsButton');
const screensaverButton = document.getElementById('screensaverButton');
const controlsPanel = document.getElementById('controls');
const viewEntireSimButton = document.getElementById('viewEntireSimButton');
const toggleStatsPanelButton = document.getElementById('toggleStatsPanelButton');
const statsPanel = document.getElementById('statsPanel');
const closeStatsPanelButton = document.getElementById('closeStatsPanelButton');
const nodeTypeStatsDiv = document.getElementById('nodeTypeStats');
const copyStatsPanelButton = document.getElementById('copyStatsPanelButton');

const creaturePopulationFloorSlider = document.getElementById('creaturePopulationFloorSlider');
const creaturePopulationFloorValueSpan = document.getElementById('creaturePopulationFloorValueSpan');
const creaturePopulationCeilingSlider = document.getElementById('creaturePopulationCeilingSlider');
const creaturePopulationCeilingValueSpan = document.getElementById('creaturePopulationCeilingValueSpan');

const bodyFluidEntrainmentSlider = document.getElementById('bodyFluidEntrainment');
const fluidCurrentStrengthSlider = document.getElementById('fluidCurrentStrength');
const bodyPushStrengthSlider = document.getElementById('bodyPushStrength');
const bodyRepulsionStrengthSlider = document.getElementById('bodyRepulsionStrength');
const bodyRepulsionRadiusFactorSlider = document.getElementById('bodyRepulsionRadiusFactor');
const globalMutationRateSlider = document.getElementById('globalMutationRate');
const baseNodeCostSlider = document.getElementById('baseNodeCost');
const emitterNodeCostSlider = document.getElementById('emitterNodeCost');
const eaterNodeCostSlider = document.getElementById('eaterNodeCost');
const predatorNodeCostSlider = document.getElementById('predatorNodeCost');
const neuronNodeCostSlider = document.getElementById('neuronNodeCost');
const photosyntheticNodeCostSlider = document.getElementById('photosyntheticNodeCost');
const photosynthesisEfficiencySlider = document.getElementById('photosynthesisEfficiency');
const swimmerNodeCostSlider = document.getElementById('swimmerNodeCost');
const jetNodeCostSlider = document.getElementById('jetNodeCostSlider');
const attractorNodeCostSlider = document.getElementById('attractorNodeCostSlider');
const repulsorNodeCostSlider = document.getElementById('repulsorNodeCostSlider');
const eyeNodeCostSlider = document.getElementById('eyeNodeCostSlider');
const eyeDetectionRadiusSlider = document.getElementById('eyeDetectionRadiusSlider');
const neuronChanceSlider = document.getElementById('neuronChanceSlider');
const jetMaxVelocityGeneSlider = document.getElementById('jetMaxVelocityGeneSlider');

const growthEnabledToggle = document.getElementById('growthEnabledToggle');
const growthEnergyCostScalarSlider = document.getElementById('growthEnergyCostScalarSlider');
const growthBaseChanceMinSlider = document.getElementById('growthBaseChanceMinSlider');
const growthBaseChanceMaxSlider = document.getElementById('growthBaseChanceMaxSlider');
const growthPopSoftLimitMultiplierSlider = document.getElementById('growthPopSoftLimitMultiplierSlider');
const growthPopHardLimitMultiplierSlider = document.getElementById('growthPopHardLimitMultiplierSlider');
const growthSizeCostMaxMultiplierSlider = document.getElementById('growthSizeCostMaxMultiplierSlider');
const reproFertilityGlobalSoftMultiplierSlider = document.getElementById('reproFertilityGlobalSoftMultiplierSlider');
const reproFertilityGlobalHardMultiplierSlider = document.getElementById('reproFertilityGlobalHardMultiplierSlider');
const reproResourceMinNutrientSlider = document.getElementById('reproResourceMinNutrientSlider');
const reproResourceMinLightSlider = document.getElementById('reproResourceMinLightSlider');
const reproResourceNutrientDebitSlider = document.getElementById('reproResourceNutrientDebitSlider');
const reproResourceLightDebitSlider = document.getElementById('reproResourceLightDebitSlider');

const instabilityLight = document.getElementById('instabilityLight');
const populationCountDisplay = document.getElementById('populationCount');
const resetButton = document.getElementById('resetButton');
const bodyFluidEntrainmentValueSpan = document.getElementById('bodyFluidEntrainmentValue');
const fluidCurrentStrengthValueSpan = document.getElementById('fluidCurrentStrengthValue');
const bodyPushStrengthValueSpan = document.getElementById('bodyPushStrengthValue');
const bodyRepulsionStrengthValueSpan = document.getElementById('bodyRepulsionStrengthValue');
const bodyRepulsionRadiusFactorValueSpan = document.getElementById('bodyRepulsionRadiusFactorValue');
const globalMutationRateValueSpan = document.getElementById('globalMutationRateValue');
const baseNodeCostValueSpan = document.getElementById('baseNodeCostValue');
const emitterNodeCostValueSpan = document.getElementById('emitterNodeCostValue');
const eaterNodeCostValueSpan = document.getElementById('eaterNodeCostValue');
const predatorNodeCostValueSpan = document.getElementById('predatorNodeCostValue');
const neuronNodeCostValueSpan = document.getElementById('neuronNodeCostValue');
const photosyntheticNodeCostValueSpan = document.getElementById('photosyntheticNodeCostValue');
const photosynthesisEfficiencyValueSpan = document.getElementById('photosynthesisEfficiencyValue');
const swimmerNodeCostValueSpan = document.getElementById('swimmerNodeCostValue');
const jetNodeCostValueSpan = document.getElementById('jetNodeCostValueSpan');
const attractorNodeCostValueSpan = document.getElementById('attractorNodeCostValueSpan');
const repulsorNodeCostValueSpan = document.getElementById('repulsorNodeCostValueSpan');
const eyeNodeCostValueSpan = document.getElementById('eyeNodeCostValueSpan');
const eyeDetectionRadiusValueSpan = document.getElementById('eyeDetectionRadiusValueSpan');
const neuronChanceValueSpan = document.getElementById('neuronChanceValueSpan');
const jetMaxVelocityGeneValueSpan = document.getElementById('jetMaxVelocityGeneValueSpan');

const growthEnergyCostScalarValueSpan = document.getElementById('growthEnergyCostScalarValueSpan');
const growthBaseChanceMinValueSpan = document.getElementById('growthBaseChanceMinValueSpan');
const growthBaseChanceMaxValueSpan = document.getElementById('growthBaseChanceMaxValueSpan');
const growthPopSoftLimitMultiplierValueSpan = document.getElementById('growthPopSoftLimitMultiplierValueSpan');
const growthPopHardLimitMultiplierValueSpan = document.getElementById('growthPopHardLimitMultiplierValueSpan');
const growthSizeCostMaxMultiplierValueSpan = document.getElementById('growthSizeCostMaxMultiplierValueSpan');
const reproFertilityGlobalSoftMultiplierValueSpan = document.getElementById('reproFertilityGlobalSoftMultiplierValueSpan');
const reproFertilityGlobalHardMultiplierValueSpan = document.getElementById('reproFertilityGlobalHardMultiplierValueSpan');
const reproResourceMinNutrientValueSpan = document.getElementById('reproResourceMinNutrientValueSpan');
const reproResourceMinLightValueSpan = document.getElementById('reproResourceMinLightValueSpan');
const reproResourceNutrientDebitValueSpan = document.getElementById('reproResourceNutrientDebitValueSpan');
const reproResourceLightDebitValueSpan = document.getElementById('reproResourceLightDebitValueSpan');

const fluidGridSizeSlider = document.getElementById('fluidGridSize');
const fluidGridSizeValueSpan = document.getElementById('fluidGridSizeValue');
const fluidDiffusionSlider = document.getElementById('fluidDiffusion');
const fluidViscositySlider = document.getElementById('fluidViscosity');
const fluidFadeSlider = document.getElementById('fluidFade');
const clearFluidButton = document.getElementById('clearFluidButton');
const fluidDiffusionValueSpan = document.getElementById('fluidDiffusionValue');
const fluidViscosityValueSpan = document.getElementById('fluidViscosityValue');
// Keep this ID aligned with index.html (#fluidFadeValueSpan) so the live value updates correctly.
const fluidFadeValueSpan = document.getElementById('fluidFadeValueSpan');
const maxFluidVelocityComponentSlider = document.getElementById('maxFluidVelocityComponentSlider');
const maxFluidVelocityComponentValueSpan = document.getElementById('maxFluidVelocityComponentValueSpan');

const particlePopulationFloorSlider = document.getElementById('particlePopulationFloorSlider');
const particlePopulationFloorValueSpan = document.getElementById('particlePopulationFloorValueSpan');
const particlePopulationCeilingSlider = document.getElementById('particlePopulationCeilingSlider');
const particlePopulationCeilingValueSpan = document.getElementById('particlePopulationCeilingValueSpan');
const particlesPerSecondSlider = document.getElementById('particlesPerSecondSlider');
const particlesPerSecondValueSpan = document.getElementById('particlesPerSecondValueSpan');
const particleFluidInfluenceSlider = document.getElementById('particleFluidInfluence');
const particleFluidInfluenceValueSpan = document.getElementById('particleFluidInfluenceValue');
const particleLifeDecaySlider = document.getElementById('particleLifeDecay');
const particleLifeDecayValueSpan = document.getElementById('particleLifeDecayValue');
const infiniteParticleLifeToggle = document.getElementById('infiniteParticleLifeToggle');
const particleLifeDecayLabel = document.getElementById('particleLifeDecayLabel');
const resetParticlesButton = document.getElementById('resetParticlesButton');
const particleCountDisplay = document.getElementById('particleCount');

const exportConfigButton = document.getElementById('exportConfigButton');
const importConfigFile = document.getElementById('importConfigFile');
const importConfigButton = document.getElementById('importConfigButton');
const exportStateButton = document.getElementById('exportStateButton');
const importStateButton = document.getElementById('importStateButton');
const importStateFile = document.getElementById('importStateFile');
const mobilePrevCreatureButton = document.getElementById('mobilePrevCreatureButton');
const mobileNextCreatureButton = document.getElementById('mobileNextCreatureButton');
const mobileInfoToggleButton = document.getElementById('mobileInfoToggleButton');
const importCreatureButton = document.getElementById('importCreatureButton');
const importCreatureFile = document.getElementById('importCreatureFile');
const creatureImportStatus = document.getElementById('creatureImportStatus');
const exportCreatureButton = document.getElementById('exportCreatureButton');

const emitterEditModeToggle = document.getElementById('emitterEditModeToggle');
const emitterStrengthSlider = document.getElementById('emitterStrength');
const emitterStrengthValueSpan = document.getElementById('emitterStrengthValue');
const clearEmittersButton = document.getElementById('clearEmittersButton');

const infoPanel = document.getElementById('infoPanel');
const closeInfoPanelButton = document.getElementById('closeInfoPanel');
const allPointsInfoContainer = document.getElementById('allPointsInfoContainer');
const copyInfoPanelButton = document.getElementById('copyInfoPanelButton');

const showNutrientMapToggle = document.getElementById('showNutrientMapToggle');
const nutrientEditModeToggle = document.getElementById('nutrientEditModeToggle');

const nutrientBrushValueSlider = document.getElementById('nutrientBrushValueSlider');
const nutrientBrushValueSpan = document.getElementById('nutrientBrushValueSpan');
const nutrientBrushSizeSlider = document.getElementById('nutrientBrushSizeSlider');
const nutrientBrushSizeSpan = document.getElementById('nutrientBrushSizeSpan');
const nutrientBrushStrengthSlider = document.getElementById('nutrientBrushStrengthSlider');
const nutrientBrushStrengthSpan = document.getElementById('nutrientBrushStrengthSpan');
const clearNutrientMapButton = document.getElementById('clearNutrientMapButton');

const showLightMapToggle = document.getElementById('showLightMapToggle');
const lightEditModeToggle = document.getElementById('lightEditModeToggle');
const lightBrushValueSlider = document.getElementById('lightBrushValueSlider');
const lightBrushValueSpan = document.getElementById('lightBrushValueSpan');
const lightBrushSizeSlider = document.getElementById('lightBrushSizeSlider');
const lightBrushSizeSpan = document.getElementById('lightBrushSizeSpan');
const lightBrushStrengthSlider = document.getElementById('lightBrushStrengthSlider');
const lightBrushStrengthSpan = document.getElementById('lightBrushStrengthSpan');
const clearLightMapButton = document.getElementById('clearLightMapButton');

const showViscosityMapToggle = document.getElementById('showViscosityMapToggle');
const viscosityEditModeToggle = document.getElementById('viscosityEditModeToggle');
const viscosityBrushValueSlider = document.getElementById('viscosityBrushValueSlider');
const viscosityBrushValueSpan = document.getElementById('viscosityBrushValueSpan');
const viscosityBrushSizeSlider = document.getElementById('viscosityBrushSizeSlider');
const viscosityBrushSizeSpan = document.getElementById('viscosityBrushSizeSpan');
const viscosityBrushStrengthSlider = document.getElementById('viscosityBrushStrengthSlider');
const viscosityBrushStrengthSpan = document.getElementById('viscosityBrushStrengthSpan');
const clearViscosityMapButton = document.getElementById('clearViscosityMapButton');

const nutrientCyclePeriodSlider = document.getElementById('nutrientCyclePeriodSlider');
const nutrientCyclePeriodSpan = document.getElementById('nutrientCyclePeriodSpan');
const nutrientCycleBaseAmplitudeSlider = document.getElementById('nutrientCycleBaseAmplitudeSlider');
const nutrientCycleBaseAmplitudeSpan = document.getElementById('nutrientCycleBaseAmplitudeSpan');
const nutrientCycleWaveAmplitudeSlider = document.getElementById('nutrientCycleWaveAmplitudeSlider');
const nutrientCycleWaveAmplitudeSpan = document.getElementById('nutrientCycleWaveAmplitudeSpan');
const lightCyclePeriodSlider = document.getElementById('lightCyclePeriodSlider');
const lightCyclePeriodSpan = document.getElementById('lightCyclePeriodSpan');
const currentNutrientMultiplierDisplay = document.getElementById('currentNutrientMultiplierDisplay');
const currentLightMultiplierDisplay = document.getElementById('currentLightMultiplierDisplay');
const showFluidVelocityToggle = document.getElementById('showFluidVelocityToggle');
const headlessModeToggle = document.getElementById('headlessModeToggle');
const useGpuFluidToggle = document.getElementById('useGpuFluidToggle');

// --- Cycling through creatures with specific node types ---
let cyclingNodeType = null;
let cyclingCreatureList = [];
let cyclingCreatureIndex = -1;

// --- Mouse Interaction State Variables ---
let mouse = {x: 0, y: 0, prevX: 0, prevY: 0, isDown: false, dx: 0, dy: 0};

// --- UI Update Functions ---
function updateSliderDisplay(slider, span) {
    let value = parseFloat(slider.value);
    if (!slider || !span) return;

    // Determine display format based on slider properties (e.g., step or id)
    const step = parseFloat(slider.step);
    const id = slider.id;

    if (step === 0.0005 || id === 'zoomSensitivitySlider' || id === 'particleLifeDecay') {
        span.textContent = value.toFixed(4);
    } else if (
        id === 'bodyFluidEntrainment' ||
        id === 'fluidFade' ||
        id === 'growthBaseChanceMinSlider' ||
        id === 'growthBaseChanceMaxSlider' ||
        id === 'reproResourceMinNutrientSlider' ||
        id === 'reproResourceMinLightSlider' ||
        id === 'reproResourceNutrientDebitSlider' ||
        id === 'reproResourceLightDebitSlider'
    ) {
        span.textContent = value.toFixed(3);
    } else if (
        id === 'globalMutationRate' ||
        id === 'bodyPushStrength' ||
        id === 'photosyntheticNodeCost' ||
        id === 'maxFluidVelocityComponentSlider' ||
        id === 'particleFluidInfluence' ||
        id === 'neuronChanceSlider' ||
        id === 'growthEnergyCostScalarSlider' ||
        id === 'growthPopSoftLimitMultiplierSlider' ||
        id === 'growthPopHardLimitMultiplierSlider' ||
        id === 'growthSizeCostMaxMultiplierSlider' ||
        id === 'reproFertilityGlobalSoftMultiplierSlider' ||
        id === 'reproFertilityGlobalHardMultiplierSlider'
    ) {
        span.textContent = value.toFixed(2);
    } else if (
        id === 'fluidCurrentStrength' ||
        id === 'bodyRepulsionStrength' ||
        id === 'bodyRepulsionRadiusFactor' ||
        id === 'baseNodeCost' ||
        id === 'emitterNodeCost' ||
        id === 'eaterNodeCost' ||
        id === 'predatorNodeCost' ||
        id === 'photosynthesisEfficiency' ||
        id === 'emitterStrength'
    ) {
        span.textContent = value.toFixed(1);
    } else if (id === 'fluidDiffusion' || id === 'fluidViscosity') {
        span.textContent = value.toExponential(1);
    } else { // Integer display for others (like population counts, cooldowns)
        span.textContent = Math.floor(value);
    }
}

function getAllSliderBindings() {
    return [
        [zoomSensitivitySlider, "ZOOM_SENSITIVITY", true, zoomSensitivityValueSpan],
        [creaturePopulationFloorSlider, "CREATURE_POPULATION_FLOOR", false, creaturePopulationFloorValueSpan],
        [creaturePopulationCeilingSlider, "CREATURE_POPULATION_CEILING", false, creaturePopulationCeilingValueSpan],
        [globalMutationRateSlider, "GLOBAL_MUTATION_RATE_MODIFIER", true, globalMutationRateValueSpan],
        [bodyFluidEntrainmentSlider, "BODY_FLUID_ENTRAINMENT_FACTOR", true, bodyFluidEntrainmentValueSpan],
        [fluidCurrentStrengthSlider, "FLUID_CURRENT_STRENGTH_ON_BODY", true, fluidCurrentStrengthValueSpan],
        [bodyPushStrengthSlider, "SOFT_BODY_PUSH_STRENGTH", true, bodyPushStrengthValueSpan],
        [bodyRepulsionStrengthSlider, "BODY_REPULSION_STRENGTH", true, bodyRepulsionStrengthValueSpan],
        [bodyRepulsionRadiusFactorSlider, "BODY_REPULSION_RADIUS_FACTOR", true, bodyRepulsionRadiusFactorValueSpan],
        [baseNodeCostSlider, "BASE_NODE_EXISTENCE_COST", true, baseNodeCostValueSpan],
        [emitterNodeCostSlider, "EMITTER_NODE_ENERGY_COST", true, emitterNodeCostValueSpan],
        [eaterNodeCostSlider, "EATER_NODE_ENERGY_COST", true, eaterNodeCostValueSpan],
        [predatorNodeCostSlider, "PREDATOR_NODE_ENERGY_COST", true, predatorNodeCostValueSpan],
        [neuronNodeCostSlider, "NEURON_NODE_ENERGY_COST", true, neuronNodeCostValueSpan],
        [swimmerNodeCostSlider, "SWIMMER_NODE_ENERGY_COST", true, swimmerNodeCostValueSpan],
        [jetNodeCostSlider, "JET_NODE_ENERGY_COST", true, jetNodeCostValueSpan],
        [attractorNodeCostSlider, "ATTRACTOR_NODE_ENERGY_COST", true, attractorNodeCostValueSpan],
        [repulsorNodeCostSlider, "REPULSOR_NODE_ENERGY_COST", true, repulsorNodeCostValueSpan],
        [photosyntheticNodeCostSlider, "PHOTOSYNTHETIC_NODE_ENERGY_COST", true, photosyntheticNodeCostValueSpan],
        [photosynthesisEfficiencySlider, "PHOTOSYNTHESIS_EFFICIENCY", true, photosynthesisEfficiencyValueSpan],
        [eyeNodeCostSlider, "EYE_NODE_ENERGY_COST", true, eyeNodeCostValueSpan],
        [eyeDetectionRadiusSlider, "EYE_DETECTION_RADIUS", false, eyeDetectionRadiusValueSpan],
        [neuronChanceSlider, "NEURON_CHANCE", true, neuronChanceValueSpan],
        [jetMaxVelocityGeneSlider, "JET_MAX_VELOCITY_GENE_DEFAULT", true, jetMaxVelocityGeneValueSpan],

        [growthEnergyCostScalarSlider, "GROWTH_ENERGY_COST_SCALAR", true, growthEnergyCostScalarValueSpan],
        [growthBaseChanceMinSlider, "GROWTH_BASE_CHANCE_MIN", true, growthBaseChanceMinValueSpan],
        [growthBaseChanceMaxSlider, "GROWTH_BASE_CHANCE_MAX", true, growthBaseChanceMaxValueSpan],
        [growthPopSoftLimitMultiplierSlider, "GROWTH_POP_SOFT_LIMIT_MULTIPLIER", true, growthPopSoftLimitMultiplierValueSpan],
        [growthPopHardLimitMultiplierSlider, "GROWTH_POP_HARD_LIMIT_MULTIPLIER", true, growthPopHardLimitMultiplierValueSpan],
        [growthSizeCostMaxMultiplierSlider, "GROWTH_SIZE_COST_MAX_MULTIPLIER", true, growthSizeCostMaxMultiplierValueSpan],
        [reproFertilityGlobalSoftMultiplierSlider, "REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER", true, reproFertilityGlobalSoftMultiplierValueSpan],
        [reproFertilityGlobalHardMultiplierSlider, "REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER", true, reproFertilityGlobalHardMultiplierValueSpan],
        [reproResourceMinNutrientSlider, "REPRO_RESOURCE_MIN_NUTRIENT", true, reproResourceMinNutrientValueSpan],
        [reproResourceMinLightSlider, "REPRO_RESOURCE_MIN_LIGHT", true, reproResourceMinLightValueSpan],
        [reproResourceNutrientDebitSlider, "REPRO_RESOURCE_NUTRIENT_DEBIT_PER_OFFSPRING", true, reproResourceNutrientDebitValueSpan],
        [reproResourceLightDebitSlider, "REPRO_RESOURCE_LIGHT_DEBIT_PER_OFFSPRING", true, reproResourceLightDebitValueSpan],

        [fluidGridSizeSlider, "FLUID_GRID_SIZE_CONTROL", false, fluidGridSizeValueSpan],
        [fluidDiffusionSlider, "FLUID_DIFFUSION", true, fluidDiffusionValueSpan],
        [fluidViscositySlider, "FLUID_VISCOSITY", true, fluidViscosityValueSpan],
        [fluidFadeSlider, "FLUID_FADE_RATE", true, fluidFadeValueSpan],
        [maxTimestepSlider, "MAX_DELTA_TIME_MS", false, maxTimestepValueSpan],
        [maxFluidVelocityComponentSlider, "MAX_FLUID_VELOCITY_COMPONENT", true, maxFluidVelocityComponentValueSpan],
        [particlePopulationFloorSlider, "PARTICLE_POPULATION_FLOOR", false, particlePopulationFloorValueSpan],
        [particlePopulationCeilingSlider, "PARTICLE_POPULATION_CEILING", false, particlePopulationCeilingValueSpan],
        [particlesPerSecondSlider, "PARTICLES_PER_SECOND", false, particlesPerSecondValueSpan],
        [particleFluidInfluenceSlider, "PARTICLE_FLUID_INFLUENCE", true, particleFluidInfluenceValueSpan],
        [particleLifeDecaySlider, "PARTICLE_BASE_LIFE_DECAY", true, particleLifeDecayValueSpan],
        [emitterStrengthSlider, "EMITTER_STRENGTH", true, emitterStrengthValueSpan],

        [nutrientBrushValueSlider, "NUTRIENT_BRUSH_VALUE", true, nutrientBrushValueSpan],
        [nutrientBrushSizeSlider, "NUTRIENT_BRUSH_SIZE", false, nutrientBrushSizeSpan],
        [nutrientBrushStrengthSlider, "NUTRIENT_BRUSH_STRENGTH", true, nutrientBrushStrengthSpan],
        [nutrientCyclePeriodSlider, "nutrientCyclePeriodSeconds", false, nutrientCyclePeriodSpan],
        [nutrientCycleBaseAmplitudeSlider, "nutrientCycleBaseAmplitude", true, nutrientCycleBaseAmplitudeSpan],
        [nutrientCycleWaveAmplitudeSlider, "nutrientCycleWaveAmplitude", true, nutrientCycleWaveAmplitudeSpan],

        [lightBrushValueSlider, "LIGHT_BRUSH_VALUE", true, lightBrushValueSpan],
        [lightBrushSizeSlider, "LIGHT_BRUSH_SIZE", false, lightBrushSizeSpan],
        [lightBrushStrengthSlider, "LIGHT_BRUSH_STRENGTH", true, lightBrushStrengthSpan],
        [lightCyclePeriodSlider, "lightCyclePeriodSeconds", false, lightCyclePeriodSpan],

        [viscosityBrushValueSlider, "VISCOSITY_BRUSH_VALUE", true, viscosityBrushValueSpan],
        [viscosityBrushSizeSlider, "VISCOSITY_BRUSH_SIZE", false, viscosityBrushSizeSpan],
        [viscosityBrushStrengthSlider, "VISCOSITY_BRUSH_STRENGTH", true, viscosityBrushStrengthSpan]
    ];
}

function randomizeSliderValue(sliderElement, isFloat, rng = Math.random) {
    if (!sliderElement) return null;

    const min = Number(sliderElement.min);
    const max = Number(sliderElement.max);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;

    const lo = Math.min(min, max);
    const hi = Math.max(min, max);
    const step = Number(sliderElement.step);

    let value = lo + rng() * (hi - lo);

    if (Number.isFinite(step) && step > 0) {
        const snappedSteps = Math.round((value - lo) / step);
        value = lo + snappedSteps * step;
    }

    if (!isFloat) value = Math.round(value);

    return Math.max(lo, Math.min(hi, value));
}

/**
 * Randomize all slider-driven runtime controls using each slider's min/max range.
 */
function randomizeControlsPanelConfig(rng = Math.random) {
    const allSliders = getAllSliderBindings();

    allSliders.forEach(([sliderElement, jsVarName, isFloat]) => {
        if (!sliderElement || typeof config[jsVarName] === 'undefined') return;
        const randomized = randomizeSliderValue(sliderElement, isFloat, rng);
        if (randomized !== null) {
            config[jsVarName] = randomized;
        }
    });

    // Keep coupled constraints valid.
    config.CREATURE_POPULATION_CEILING = Math.max(config.CREATURE_POPULATION_FLOOR, config.CREATURE_POPULATION_CEILING);
    config.PARTICLE_POPULATION_CEILING = Math.max(config.PARTICLE_POPULATION_FLOOR, config.PARTICLE_POPULATION_CEILING);
    config.GROWTH_BASE_CHANCE_MAX = Math.max(config.GROWTH_BASE_CHANCE_MIN, config.GROWTH_BASE_CHANCE_MAX);
    config.GROWTH_POP_HARD_LIMIT_MULTIPLIER = Math.max(config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER, config.GROWTH_POP_HARD_LIMIT_MULTIPLIER);
    config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER = Math.max(config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER, config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER);

    // Randomize key non-slider toggles as part of launch randomization.
    config.IS_WORLD_WRAPPING = rng() < 0.5;
    config.GROWTH_ENABLED = rng() < 0.85;

    // Keep particle life decay slider usable by default in random worlds.
    config.IS_PARTICLE_LIFE_INFINITE = false;
}

function initializeAllSliderDisplays() {
    const allSliders = getAllSliderBindings();

    worldWidthInput.value = config.WORLD_WIDTH;
    worldHeightInput.value = config.WORLD_HEIGHT;

    allSliders.forEach(([sliderElement, jsVarName, isFloat, spanElement]) => {
        if (sliderElement && typeof config[jsVarName] !== 'undefined') {
            // Keep UI controls aligned with runtime config values (do not overwrite config from HTML defaults).
            sliderElement.value = String(config[jsVarName]);
            config[jsVarName] = isFloat ? parseFloat(sliderElement.value) : parseInt(sliderElement.value);

            if (spanElement) {
                updateSliderDisplay(sliderElement, spanElement);
            }
        } else {
            if (!sliderElement) console.warn(`Slider element for ${jsVarName} not found.`);
            if (typeof config[jsVarName] === 'undefined') console.warn(`Global JS variable ${jsVarName} not found.`);
        }
    });

    // Update checkbox states based on runtime config.
    particleLifeDecaySlider.disabled = config.IS_PARTICLE_LIFE_INFINITE;
    particleLifeDecayLabel.style.color = config.IS_PARTICLE_LIFE_INFINITE ? '#777' : '#ddd';
    particleLifeDecayValueSpan.style.color = config.IS_PARTICLE_LIFE_INFINITE ? '#777' : '#00aeff';
    worldWrapToggle.checked = config.IS_WORLD_WRAPPING;
    emitterEditModeToggle.checked = config.IS_EMITTER_EDIT_MODE;
    growthEnabledToggle.checked = config.GROWTH_ENABLED;
    infiniteParticleLifeToggle.checked = config.IS_PARTICLE_LIFE_INFINITE;
    canvas.classList.toggle('emitter-edit-mode', config.IS_EMITTER_EDIT_MODE);

    showNutrientMapToggle.checked = config.SHOW_NUTRIENT_MAP;
    nutrientEditModeToggle.checked = config.IS_NUTRIENT_EDIT_MODE;
    showLightMapToggle.checked = config.SHOW_LIGHT_MAP;
    lightEditModeToggle.checked = config.IS_LIGHT_EDIT_MODE;
    showViscosityMapToggle.checked = config.SHOW_VISCOSITY_MAP;
    viscosityEditModeToggle.checked = config.IS_VISCOSITY_EDIT_MODE;
    showFluidVelocityToggle.checked = config.SHOW_FLUID_VELOCITY;
    headlessModeToggle.checked = config.IS_HEADLESS_MODE;
}

function updateInstabilityIndicator() {
    if (config.isAnySoftBodyUnstable) {
        instabilityLight.classList.add('unstable');
    } else {
        instabilityLight.classList.remove('unstable');
    }
}

function updatePopulationCount() {
    populationCountDisplay.textContent = `Population: ${softBodyPopulation.length}`;
    particleCountDisplay.textContent = `Particles: ${particles.length}`;
}

/**
 * Mobile layout heuristic used for touch-first panel behavior.
 */
function isMobileLayout() {
    return window.matchMedia('(max-width: 900px)').matches;
}

/**
 * Toggle info panel visibility without altering selected creature state.
 */
function toggleInfoPanelVisibility() {
    infoPanel.classList.toggle('open');
}

function updateInfoPanel() {
    if (config.selectedInspectBody && config.selectedInspectPoint) {
        document.getElementById('infoBodyId').textContent = config.selectedInspectBody.id;
        document.getElementById('infoBodyStiffness').textContent = config.selectedInspectBody.getAverageStiffness().toFixed(2);
        document.getElementById('infoBodyDamping').textContent = config.selectedInspectBody.getAverageDamping().toFixed(2);
        document.getElementById('infoBodyMotorInterval').textContent = config.selectedInspectBody.motorImpulseInterval;
        document.getElementById('infoBodyMotorCap').textContent = config.selectedInspectBody.motorImpulseMagnitudeCap.toFixed(2);
        document.getElementById('infoBodyEmitterStrength').textContent = config.selectedInspectBody.emitterStrength.toFixed(2);
        document.getElementById('infoBodyEmitterDirX').textContent = config.selectedInspectBody.emitterDirection.x.toFixed(2);
        document.getElementById('infoBodyEmitterDirY').textContent = config.selectedInspectBody.emitterDirection.y.toFixed(2);
        document.getElementById('infoBodyNumOffspring').textContent = config.selectedInspectBody.numOffspring;
        document.getElementById('infoBodyOffspringRadius').textContent = config.selectedInspectBody.offspringSpawnRadius.toFixed(1);
        document.getElementById('infoBodyPointAddChance').textContent = config.selectedInspectBody.pointAddChance.toFixed(3);
        document.getElementById('infoBodySpringConnectionRadius').textContent = config.selectedInspectBody.springConnectionRadius.toFixed(1);
        document.getElementById('infoBodyEnergy').textContent = config.selectedInspectBody.creatureEnergy.toFixed(2);
        document.getElementById('infoBodyReproEnergyThreshold').textContent = config.selectedInspectBody.reproductionEnergyThreshold;
        document.getElementById('infoBodyCurrentMaxEnergy').textContent = config.selectedInspectBody.currentMaxEnergy.toFixed(2);
        document.getElementById('infoBodyTicksBirth').textContent = config.selectedInspectBody.ticksSinceBirth;
        document.getElementById('infoBodyCanReproduce').textContent = config.selectedInspectBody.canReproduce;
        document.getElementById('infoBodyRewardStrategy').textContent = getRewardStrategyString(config.selectedInspectBody.rewardStrategy);
        document.getElementById('infoBodyEnergyPhoto').textContent = config.selectedInspectBody.energyGainedFromPhotosynthesis.toFixed(2);
        document.getElementById('infoBodyEnergyEat').textContent = config.selectedInspectBody.energyGainedFromEating.toFixed(2);
        document.getElementById('infoBodyEnergyPred').textContent = config.selectedInspectBody.energyGainedFromPredation.toFixed(2);
        document.getElementById('infoBodyGrowthEvents').textContent = config.selectedInspectBody.growthEventsCompleted || 0;
        document.getElementById('infoBodyGrowthEnergySpent').textContent = (config.selectedInspectBody.totalGrowthEnergySpent || 0).toFixed(2);
        document.getElementById('infoBodyGrowthSuppEnergy').textContent = config.selectedInspectBody.growthSuppressedByEnergy || 0;
        document.getElementById('infoBodyGrowthSuppCooldown').textContent = config.selectedInspectBody.growthSuppressedByCooldown || 0;
        document.getElementById('infoBodyGrowthSuppPopulation').textContent = config.selectedInspectBody.growthSuppressedByPopulation || 0;
        document.getElementById('infoBodyGrowthSuppMaxPoints').textContent = config.selectedInspectBody.growthSuppressedByMaxPoints || 0;
        document.getElementById('infoBodyGrowthSuppNoCapacity').textContent = config.selectedInspectBody.growthSuppressedByNoCapacity || 0;
        document.getElementById('infoBodyGrowthSuppChanceRoll').textContent = config.selectedInspectBody.growthSuppressedByChanceRoll || 0;
        document.getElementById('infoBodyGrowthSuppPlacement').textContent = config.selectedInspectBody.growthSuppressedByPlacement || 0;
        document.getElementById('infoBodyNnTopologyVersion').textContent = config.selectedInspectBody.nnTopologyVersion || 0;
        document.getElementById('infoBodyRlTopologyResets').textContent = config.selectedInspectBody.rlBufferResetsDueToTopology || 0;
        document.getElementById('infoBodyReproSuppDensity').textContent = config.selectedInspectBody.reproductionSuppressedByDensity || 0;
        document.getElementById('infoBodyReproSuppResources').textContent = config.selectedInspectBody.reproductionSuppressedByResources || 0;
        document.getElementById('infoBodyReproSuppFertility').textContent = config.selectedInspectBody.reproductionSuppressedByFertilityRoll || 0;
        document.getElementById('infoBodyReproResourceDebits').textContent = config.selectedInspectBody.reproductionResourceDebitApplied || 0;

        // Populate new energy cost fields
        document.getElementById('infoBodyCostBase').textContent = config.selectedInspectBody.energyCostFromBaseNodes.toFixed(2);
        document.getElementById('infoBodyCostEmitter').textContent = config.selectedInspectBody.energyCostFromEmitterNodes.toFixed(2);
        document.getElementById('infoBodyCostEater').textContent = config.selectedInspectBody.energyCostFromEaterNodes.toFixed(2);
        document.getElementById('infoBodyCostPredator').textContent = config.selectedInspectBody.energyCostFromPredatorNodes.toFixed(2);
        document.getElementById('infoBodyCostNeuron').textContent = config.selectedInspectBody.energyCostFromNeuronNodes.toFixed(2);
        document.getElementById('infoBodyCostSwimmer').textContent = config.selectedInspectBody.energyCostFromSwimmerNodes.toFixed(2);
        document.getElementById('infoBodyCostJet').textContent = config.selectedInspectBody.energyCostFromJetNodes.toFixed(2);
        document.getElementById('infoBodyCostAttractor').textContent = config.selectedInspectBody.energyCostFromAttractorNodes.toFixed(2);
        document.getElementById('infoBodyCostRepulsor').textContent = config.selectedInspectBody.energyCostFromRepulsorNodes.toFixed(2);
        document.getElementById('infoBodyCostPhoto').textContent = config.selectedInspectBody.energyCostFromPhotosyntheticNodes.toFixed(2);
        document.getElementById('infoBodyCostGrabbing').textContent = config.selectedInspectBody.energyCostFromGrabbingNodes.toFixed(2);
        document.getElementById('infoBodyCostEye').textContent = config.selectedInspectBody.energyCostFromEyeNodes.toFixed(2);

        // Add display for new reproduction cooldown properties
        let reproGeneEl = document.getElementById('infoBodyReproCooldownGeneVal'); // Target the span directly
        let reproGenePEL = document.getElementById('infoBodyReproCooldownGeneP'); // Target the p element
        if (!reproGenePEL) { // If the paragraph doesn't exist, create it
            reproGenePEL = createInfoPanelParagraph(infoPanel.querySelector('.info-section'), 'infoBodyReproCooldownGene', 'Repro. Cooldown Gene:');
            reproGeneEl = reproGenePEL.querySelector('span'); // Get the span from the new P
        } else {
            reproGeneEl = reproGenePEL.querySelector('span'); // Ensure we have the span if P exists
        }
        if (reproGeneEl) reproGeneEl.textContent = config.selectedInspectBody.reproductionCooldownGene;


        let effectiveReproEl = document.getElementById('infoBodyEffectiveReproCooldownVal'); // Target the span
        let effectiveReproPEL = document.getElementById('infoBodyEffectiveReproCooldownP'); // Target the P
        if (!effectiveReproPEL) {
            effectiveReproPEL = createInfoPanelParagraph(infoPanel.querySelector('.info-section'), 'infoBodyEffectiveReproCooldown', 'Effective Repro. Cooldown:');
            effectiveReproEl = effectiveReproPEL.querySelector('span');
        } else {
            effectiveReproEl = effectiveReproPEL.querySelector('span');
        }
        if (effectiveReproEl) effectiveReproEl.textContent = config.selectedInspectBody.effectiveReproductionCooldown;

        allPointsInfoContainer.innerHTML = '<h5>All Mass Points</h5>';
        config.selectedInspectBody.massPoints.forEach((point, index) => {
            const pointEntryDiv = document.createElement('div');
            pointEntryDiv.className = 'point-info-entry';

            let content = `<p><strong>Point Index:</strong> ${index}</p>`;
            content += `<p><strong>Node Type:</strong> ${getNodeTypeString(point.nodeType)}</p>`;
            content += `<p><strong>Movement Type:</strong> ${getMovementTypeString(point.movementType)}</p>`;
            content += `<p><strong>Mass:</strong> ${point.mass.toFixed(2)}</p>`;
            content += `<p><strong>Radius:</strong> ${point.radius.toFixed(2)}</p>`;
            content += `<p><strong>World Pos:</strong> X: ${point.pos.x.toFixed(2)}, Y: ${point.pos.y.toFixed(2)}</p>`;
            content += `<p><strong>Can Be Grabber:</strong> ${point.canBeGrabber}</p>`;
            if (point.nodeType === NodeType.PREDATOR) {
                const predatorRadiusGene = Number.isFinite(Number(point.predatorRadiusGene)) ? Number(point.predatorRadiusGene) : Number(config.PREDATOR_RADIUS_GENE_MIN || 0.2);
                content += `<p><strong>Predator Radius Gene:</strong> ${predatorRadiusGene.toFixed(3)}</p>`;
            }
            if (point.nodeType === NodeType.EMITTER) {
                content += `<p><strong>Dye Color:</strong> R:${point.dyeColor[0].toFixed(0)} G:${point.dyeColor[1].toFixed(0)} B:${point.dyeColor[2].toFixed(0)}</p>`;
            }
            if (point.nodeType === NodeType.JET) {
                content += `<p><strong>Max Effective Velocity:</strong> ${point.maxEffectiveJetVelocity.toFixed(2)}</p>`;
            }
            if (point.isGrabbing) {
                content += `<p><strong>State:</strong> Grabbing</p>`;
            } else {
                content += `<p><strong>State:</strong> Normal</p>`;
            }

            if (point.nodeType === NodeType.EYE) {
                content += `<h6>Eye Sensor Data:</h6>`;
                content += `<p><strong>Target Type:</strong> ${getEyeTargetTypeString(point.eyeTargetType)}</p>`;
                content += `<p><strong>Sees Target:</strong> ${point.seesTarget}</p>`;
                if (point.seesTarget) {
                    content += `<p><strong>Target Distance:</strong> ${(point.nearestTargetMagnitude * config.EYE_DETECTION_RADIUS).toFixed(1)} (norm: ${point.nearestTargetMagnitude.toFixed(3)})</p>`;
                    content += `<p><strong>Target Angle:</strong> ${(point.nearestTargetDirection * 180 / Math.PI).toFixed(1)}&deg;</p>`;
                }
            }

            if (point.nodeType === NodeType.NEURON && point.neuronData) {
                if (typeof point.neuronData.hiddenLayerSize === 'undefined') {
                    const bodyIdForLog = config.selectedInspectBody ? config.selectedInspectBody.id : 'UnknownBody';
                    console.warn(`Neuron in Body ${bodyIdForLog}, Point Index ${index}, has neuronData, but hiddenLayerSize is UNDEFINED. neuronData:`, JSON.parse(JSON.stringify(point.neuronData)));
                }

                if (point.neuronData.isBrain) {
                    content += `<h6>Brain Details:</h6>`;
                    content += `<p><strong>Role:</strong> Active Brain</p>`;
                    content += `<p><strong>Hidden Layer Size:</strong> ${point.neuronData.hiddenLayerSize || 'N/A'}</p>`;
                    content += `<p><strong>Input Vector Size:</strong> ${point.neuronData.inputVectorSize || 'N/A'}</p>`;
                    content += `<p><strong>Output Vector Size:</strong> ${point.neuronData.outputVectorSize || 'N/A'}</p>`;
                    if (typeof point.neuronData.lastAvgNormalizedReward === 'number') {
                        content += `<p><strong>Avg Batch Reward:</strong> ${point.neuronData.lastAvgNormalizedReward.toFixed(3)}</p>`;
                    }

                    // NEW: Display Labeled Inputs
                    if (point.neuronData.currentFrameInputVectorWithLabels && point.neuronData.currentFrameInputVectorWithLabels.length > 0) {
                        content += `<h6>Brain Inputs (Real-time):</h6>`;
                        point.neuronData.currentFrameInputVectorWithLabels.forEach(input => {
                             content += `<p><strong style="color:#aadeff;">${input.label}:</strong> <span class="stat-value">${input.value.toFixed(3)}</span></p>`;
                        });
                    }

                    // NEW: Display Labeled Outputs/Actions
                    if (point.neuronData.currentFrameActionDetails && point.neuronData.currentFrameActionDetails.length > 0) {
                        content += `<h6>Brain Actions (Real-time Outputs):</h6>`;
                        point.neuronData.currentFrameActionDetails.forEach(action => {
                            // Determine the final action value based on the label, as some are sigmoided, some are not.
                            let finalValueDisplay = "";
                            if (action.label.includes("Direction")) {
                                // Angle, not sigmoided
                                finalValueDisplay = `${(action.sampledAction).toFixed(2)} rad`;
                            } else if (action.label.includes("Toggle")) {
                                // Boolean based on sigmoid
                                finalValueDisplay = sigmoid(action.sampledAction) > 0.5 ? "ON" : "OFF";
                            }
                            else {
                                // Default to sigmoid for magnitude, exertion, color channels
                                finalValueDisplay = `${sigmoid(action.sampledAction).toFixed(3)}`;
                            }

                            content += `<p><strong style="color:#aadeff;">${action.label}:</strong> <span class="stat-value">${finalValueDisplay}</span> <em style="font-size:0.9em; color:#999;">(&mu;:${action.mean.toFixed(2)}, &sigma;:${action.stdDev.toFixed(2)})</em></p>`;
                        });
                    }
                } else {
                    content += `<h6>Neuron (Non-Brain)</h6>`;
                    content += `<p><strong>Hidden Layer Size (if applicable):</strong> ${point.neuronData.hiddenLayerSize || 'N/A'}</p>`;
                }
            }
            pointEntryDiv.innerHTML = content;
            allPointsInfoContainer.appendChild(pointEntryDiv);
        });

        // Desktop keeps auto-open behavior; mobile requires explicit user toggle.
        if (!infoPanel.classList.contains('open') && !isMobileLayout()) {
            infoPanel.classList.add('open');
        }
    } else {
        allPointsInfoContainer.innerHTML = '';
        document.getElementById('infoBodyId').textContent = '-';
        document.getElementById('infoBodyStiffness').textContent = '-';
        document.getElementById('infoBodyDamping').textContent = '-';
        document.getElementById('infoBodyMotorInterval').textContent = '-';
        document.getElementById('infoBodyMotorCap').textContent = '-';
        document.getElementById('infoBodyEmitterStrength').textContent = '-';
        document.getElementById('infoBodyEmitterDirX').textContent = '-';
        document.getElementById('infoBodyEmitterDirY').textContent = '-';
        document.getElementById('infoBodyNumOffspring').textContent = '-';
        document.getElementById('infoBodyOffspringRadius').textContent = '-';
        document.getElementById('infoBodyPointAddChance').textContent = '-';
        document.getElementById('infoBodySpringConnectionRadius').textContent = '-';
        document.getElementById('infoBodyEnergy').textContent = '-';
        document.getElementById('infoBodyReproEnergyThreshold').textContent = '-';
        document.getElementById('infoBodyCurrentMaxEnergy').textContent = '-';
        document.getElementById('infoBodyTicksBirth').textContent = '-';
        document.getElementById('infoBodyCanReproduce').textContent = '-';
        document.getElementById('infoBodyRewardStrategy').textContent = '-';
        document.getElementById('infoBodyEnergyPhoto').textContent = '-';
        document.getElementById('infoBodyEnergyEat').textContent = '-';
        document.getElementById('infoBodyEnergyPred').textContent = '-';
        document.getElementById('infoBodyGrowthEvents').textContent = '-';
        document.getElementById('infoBodyGrowthEnergySpent').textContent = '-';
        document.getElementById('infoBodyGrowthSuppEnergy').textContent = '-';
        document.getElementById('infoBodyGrowthSuppCooldown').textContent = '-';
        document.getElementById('infoBodyGrowthSuppPopulation').textContent = '-';
        document.getElementById('infoBodyNnTopologyVersion').textContent = '-';
        document.getElementById('infoBodyRlTopologyResets').textContent = '-';
        document.getElementById('infoBodyReproSuppDensity').textContent = '-';
        document.getElementById('infoBodyReproSuppResources').textContent = '-';
        document.getElementById('infoBodyReproSuppFertility').textContent = '-';
        document.getElementById('infoBodyReproResourceDebits').textContent = '-';

        document.getElementById('infoBodyCostBase').textContent = '-';
        document.getElementById('infoBodyCostEmitter').textContent = '-';
        document.getElementById('infoBodyCostEater').textContent = '-';
        document.getElementById('infoBodyCostPredator').textContent = '-';
        document.getElementById('infoBodyCostNeuron').textContent = '-';
        document.getElementById('infoBodyCostSwimmer').textContent = '-';
        document.getElementById('infoBodyCostJet').textContent = '-';
        document.getElementById('infoBodyCostAttractor').textContent = '-';
        document.getElementById('infoBodyCostRepulsor').textContent = '-';
        document.getElementById('infoBodyCostPhoto').textContent = '-';
        document.getElementById('infoBodyCostGrabbing').textContent = '-';
        document.getElementById('infoBodyCostEye').textContent = '-';

        infoPanel.classList.remove('open');
    }
}

function showMessageModal(message) {
    let modal = document.getElementById('messageModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'messageModal';
        modal.style.position = 'fixed';
        modal.style.left = '50%';
        modal.style.top = '50%';
        modal.style.transform = 'translate(-50%, -50%)';
        modal.style.backgroundColor = 'rgba(50,50,50,0.9)';
        modal.style.color = 'white';
        modal.style.padding = '20px';
        modal.style.border = '1px solid #777';
        modal.style.borderRadius = '8px';
        modal.style.boxShadow = '0 0 15px rgba(0,0,0,0.5)';
        modal.style.zIndex = '2000';

        const messageP = document.createElement('p');
        messageP.id = 'modalMessageText';
        messageP.style.margin = '0 0 15px 0';

        const closeButton = document.createElement('button');
        closeButton.textContent = 'OK';
        closeButton.style.padding = '8px 15px';
        closeButton.style.backgroundColor = '#007bff';
        closeButton.style.color = 'white';
        closeButton.style.border = 'none';
        closeButton.style.borderRadius = '4px';
        closeButton.style.cursor = 'pointer';
        closeButton.onclick = () => modal.style.display = 'none';

        modal.appendChild(messageP);
        modal.appendChild(closeButton);
        document.body.appendChild(modal);
    }
    document.getElementById('modalMessageText').textContent = message;
    modal.style.display = 'block';
}

// --- Mouse Interaction Logic ---
function updateMouse(e) {
    const rect = canvas.getBoundingClientRect();
    mouse.prevX = mouse.x;
    mouse.prevY = mouse.y;
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.dx = mouse.x - mouse.prevX;
    mouse.dy = mouse.y - mouse.prevY;
}

function getMouseWorldCoordinates(displayMouseX, displayMouseY) {
    // Camera mapping is always screen=(world-offset)*zoom, even when offsets are negative
    // to center a small world in a large viewport. Keep input math aligned with renderer math.
    return displayToWorld({
        displayX: displayMouseX,
        displayY: displayMouseY,
        viewZoom: config.viewZoom,
        viewOffsetX: config.viewOffsetX,
        viewOffsetY: config.viewOffsetY
    });
}

/**
 * Resolve a safe max zoom even if imported config is missing/invalid for MAX_ZOOM.
 */
function getSafeMaxZoom() {
    const maxZoom = Number(config.MAX_ZOOM);
    if (Number.isFinite(maxZoom) && maxZoom > 0) return maxZoom;
    return 8.0;
}

// --- Event Listeners ---
function cycleSelectedCreature(direction = 1, { relockFollow = true } = {}) {
    const liveBodies = softBodyPopulation.filter(body => !body.isUnstable && body.massPoints && body.massPoints.length > 0);
    if (liveBodies.length === 0) return;

    let idx = liveBodies.findIndex(body => body === config.selectedInspectBody);
    if (idx === -1) idx = 0;
    idx = (idx + direction + liveBodies.length) % liveBodies.length;

    const nextBody = liveBodies[idx];

    // Arrow/mobile creature cycling is intended as a follow-navigation control.
    // Re-lock follow after manual pan/zoom and jump camera immediately to the new target.
    if (relockFollow) {
        config.AUTO_FOLLOW_CREATURE = true;
        focusOnCreature(nextBody);
        return;
    }

    config.selectedInspectBody = nextBody;
    config.selectedInspectPointIndex = 0;
    config.selectedInspectPoint = nextBody.massPoints[0] || null;
    updateInfoPanel();
}

function disableAutoFollowForManualControl() {
    if (config.AUTO_FOLLOW_CREATURE) {
        config.AUTO_FOLLOW_CREATURE = false;
        console.log('[CAMERA] Auto-follow disabled for manual control.');
    }
}

function applyManualZoom(displayX, displayY, direction) {
    // Sync local camera cache before computing deltas.
    viewZoom = config.viewZoom;
    viewOffsetX = config.viewOffsetX;
    viewOffsetY = config.viewOffsetY;

    // Use the same world-coordinate mapping used by input handling.
    const worldBefore = getMouseWorldCoordinates(displayX, displayY);
    const worldXBefore = worldBefore.x;
    const worldYBefore = worldBefore.y;

    let newZoom = viewZoom * Math.pow(1 + config.ZOOM_SENSITIVITY * 10, direction);
    const minZoomToFitX = canvas.clientWidth / config.WORLD_WIDTH;
    const minZoomToFitY = canvas.clientHeight / config.WORLD_HEIGHT;
    const minAllowedZoom = Math.max(0.01, Math.min(minZoomToFitX, minZoomToFitY));
    newZoom = Math.max(minAllowedZoom, Math.min(newZoom, getSafeMaxZoom()));

    viewZoom = newZoom;

    // Preserve the same world anchor under the pointer after zoom changes.
    const anchoredOffset = solveViewOffsetForAnchor({
        displayX,
        displayY,
        worldX: worldXBefore,
        worldY: worldYBefore,
        viewZoom
    });
    viewOffsetX = anchoredOffset.offsetX;
    viewOffsetY = anchoredOffset.offsetY;

    clampViewOffsets();

    config.viewZoom = viewZoom;
    config.viewOffsetX = viewOffsetX;
    config.viewOffsetY = viewOffsetY;

    viewport.zoom = viewZoom;
    viewport.offsetX = viewOffsetX;
    viewport.offsetY = viewOffsetY;
}

document.addEventListener('keydown', (e) => {
    // Allow key controls even while paused.
    if (
        e.key.toLowerCase() !== 'p' &&
        e.key.toLowerCase() !== 'f' &&
        e.key.toLowerCase() !== 'w' &&
        e.key.toLowerCase() !== 'a' &&
        e.key.toLowerCase() !== 's' &&
        e.key.toLowerCase() !== 'd' &&
        e.key !== 'ArrowLeft' &&
        e.key !== 'ArrowRight' &&
        e.key !== 'ArrowUp' &&
        e.key !== 'ArrowDown' &&
        config.IS_SIMULATION_PAUSED
    ) {
        return; // Only block unrelated keys if paused
    }

    // Sync local camera cache in case auto-follow updated viewport this frame.
    viewZoom = config.viewZoom;
    viewOffsetX = config.viewOffsetX;
    viewOffsetY = config.viewOffsetY;

    const panSpeed = config.VIEW_PAN_SPEED / viewZoom;


    switch (e.key.toLowerCase()) {
        case 'w':
            disableAutoFollowForManualControl();
            viewOffsetY -= panSpeed;
            clampViewOffsets();
            break;
        case 's':
            disableAutoFollowForManualControl();
            viewOffsetY += panSpeed;
            clampViewOffsets();
            break;
        case 'a':
            disableAutoFollowForManualControl();
            viewOffsetX -= panSpeed;
            clampViewOffsets();
            break;
        case 'd':
            disableAutoFollowForManualControl();
            viewOffsetX += panSpeed;
            clampViewOffsets();
            break;
        case 'f':
            config.AUTO_FOLLOW_CREATURE = !config.AUTO_FOLLOW_CREATURE;
            console.log(`[CAMERA] Auto-follow ${config.AUTO_FOLLOW_CREATURE ? 'enabled' : 'disabled'}.`);
            break;
        case 'p':
            config.IS_SIMULATION_PAUSED = !config.IS_SIMULATION_PAUSED;
            pauseResumeButton.textContent = config.IS_SIMULATION_PAUSED ? "Resume" : "Pause";
            if (!config.IS_SIMULATION_PAUSED) {
                lastTime = performance.now(); // lastTime is in main.js
                requestAnimationFrame(gameLoop); // gameLoop is in main.js
            }
            break;
    }

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        cycleSelectedCreature(-1);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        cycleSelectedCreature(1);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        disableAutoFollowForManualControl();
        applyManualZoom(canvas.clientWidth / 2, canvas.clientHeight / 2, 1);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        disableAutoFollowForManualControl();
        applyManualZoom(canvas.clientWidth / 2, canvas.clientHeight / 2, -1);
    }

    // After handling panning keys, synchronize config so helpers using config.* stay accurate
    config.viewOffsetX = viewOffsetX;
    config.viewOffsetY = viewOffsetY;
    config.viewZoom = viewZoom;

    // After handling panning keys and updating config, sync viewport object
    viewport.zoom = viewZoom;
    viewport.offsetX = viewOffsetX;
    viewport.offsetY = viewOffsetY;
});


worldWrapToggle.onchange = function () {
    config.IS_WORLD_WRAPPING = this.checked;
    if (fluidField) fluidField.useWrapping = config.IS_WORLD_WRAPPING;
}
maxTimestepSlider.oninput = function () {
    config.MAX_DELTA_TIME_MS = parseInt(this.value);
    updateSliderDisplay(this, maxTimestepValueSpan);
}
zoomSensitivitySlider.oninput = function () {
    config.ZOOM_SENSITIVITY = parseFloat(this.value);
    updateSliderDisplay(this, zoomSensitivityValueSpan);
}

pauseResumeButton.onclick = function () {
    config.IS_SIMULATION_PAUSED = !config.IS_SIMULATION_PAUSED;
    this.textContent = config.IS_SIMULATION_PAUSED ? "Resume" : "Pause";
    if (!config.IS_SIMULATION_PAUSED) {
        lastTime = performance.now(); // lastTime is in main.js
        requestAnimationFrame(gameLoop); // gameLoop is in main.js
    }
}
toggleControlsButton.onclick = function () {
    controlsPanel.classList.toggle('open');
}

function toggleStatsPanel() {
    statsPanel.classList.toggle('open');
    if (statsPanel.classList.contains('open')) {
        updateStatsPanel();
    }
}

toggleStatsPanelButton.onclick = toggleStatsPanel;

closeStatsPanelButton.onclick = function () {
    statsPanel.classList.remove('open');
}

copyStatsPanelButton.onclick = function () {
    const nodeStatsDiv = document.getElementById('nodeTypeStats');
    const mutationStatsDiv = document.getElementById('mutationTypeStats');
    let textToCopy = "Simulation Statistics\n----------------------\n\n";

    if (nodeStatsDiv) {
        const children = nodeStatsDiv.querySelectorAll('p');
        children.forEach(child => {
            textToCopy += child.textContent.trim() + "\n";
        });
    }
    textToCopy += "\n"; // Add a separator

    if (mutationStatsDiv) {
        const children = mutationStatsDiv.querySelectorAll('p');
        children.forEach(child => {
            textToCopy += child.textContent.trim() + "\n";
        });
    }

    navigator.clipboard.writeText(textToCopy.trim()).then(() => {
        const originalText = copyStatsPanelButton.textContent;
        copyStatsPanelButton.textContent = "Copied!";
        setTimeout(() => {
            copyStatsPanelButton.textContent = originalText;
        }, 1500);
    }).catch(err => {
        console.error('Failed to copy stats text: ', err);
        showMessageModal("Failed to copy stats. See console for details.");
    });
}

function toggleScreensaverMode(forceOff = false) {
    const isCurrentlyFullscreen = !!document.fullscreenElement;
    const isBodyCssFullscreen = document.body.classList.contains('css-screensaver-active');

    if (forceOff) { // Force exit
        if (isCurrentlyFullscreen) {
            document.exitFullscreen().catch(err => console.error("Error exiting fullscreen:", err));
        }
        document.body.classList.remove('css-screensaver-active');
        screensaverButton.textContent = "Enter Screensaver";
        screensaverButton.classList.remove('in-screensaver');
        return;
    }

    if (isCurrentlyFullscreen || isBodyCssFullscreen) {
        // Exit fullscreen/screensaver
        if (isCurrentlyFullscreen) {
            document.exitFullscreen().catch(err => console.error("Error exiting fullscreen:", err));
        }
        document.body.classList.remove('css-screensaver-active');
        screensaverButton.textContent = "Enter Screensaver";
        screensaverButton.classList.remove('in-screensaver');
    } else {
        // Enter fullscreen/screensaver
        if (canvasContainer.requestFullscreen) {
            canvasContainer.requestFullscreen().then(() => {
                screensaverButton.textContent = "Exit Screensaver";
                screensaverButton.classList.add('in-screensaver');
                document.body.classList.add('css-screensaver-active');
            }).catch(err => {
                console.warn("Fullscreen API failed, falling back to CSS mode:", err);
                document.body.classList.add('css-screensaver-active');
                screensaverButton.textContent = "Exit Screensaver (CSS)";
                screensaverButton.classList.add('in-screensaver');
            });
        } else {
            document.body.classList.add('css-screensaver-active');
            screensaverButton.textContent = "Exit Screensaver (CSS)";
            screensaverButton.classList.add('in-screensaver');
        }
    }
}

screensaverButton.onclick = () => toggleScreensaverMode();

document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
        document.body.classList.remove('css-screensaver-active');
        screensaverButton.textContent = "Enter Screensaver";
        screensaverButton.classList.remove('in-screensaver');
    } else {
        document.body.classList.add('css-screensaver-active');
        screensaverButton.textContent = "Exit Screensaver";
        screensaverButton.classList.add('in-screensaver');
    }
});


creaturePopulationFloorSlider.oninput = function () {
    config.CREATURE_POPULATION_FLOOR = parseInt(this.value);
    updateSliderDisplay(this, creaturePopulationFloorValueSpan);
}
creaturePopulationCeilingSlider.oninput = function () {
    config.CREATURE_POPULATION_CEILING = parseInt(this.value);
    updateSliderDisplay(this, creaturePopulationCeilingValueSpan);
}
particlePopulationFloorSlider.oninput = function () {
    config.PARTICLE_POPULATION_FLOOR = parseInt(this.value);
    updateSliderDisplay(this, particlePopulationFloorValueSpan);
}
particlePopulationCeilingSlider.oninput = function () {
    config.PARTICLE_POPULATION_CEILING = parseInt(this.value);
    updateSliderDisplay(this, particlePopulationCeilingValueSpan);
}


emitterEditModeToggle.onchange = function () {
    config.IS_EMITTER_EDIT_MODE = this.checked;
    canvas.classList.toggle('emitter-edit-mode', config.IS_EMITTER_EDIT_MODE);
    if (!config.IS_EMITTER_EDIT_MODE) {
        currentEmitterPreview = null;
        emitterDragStartCell = null;
    }
}
emitterStrengthSlider.oninput = function () {
    config.EMITTER_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, emitterStrengthValueSpan);
}
clearEmittersButton.onclick = function () {
    velocityEmitters = [];
}


bodyFluidEntrainmentSlider.oninput = function () {
    config.BODY_FLUID_ENTRAINMENT_FACTOR = parseFloat(this.value);
    updateSliderDisplay(this, bodyFluidEntrainmentValueSpan);
}
fluidCurrentStrengthSlider.oninput = function () {
    config.FLUID_CURRENT_STRENGTH_ON_BODY = parseFloat(this.value);
    updateSliderDisplay(this, fluidCurrentStrengthValueSpan);
}
bodyPushStrengthSlider.oninput = function () {
    config.SOFT_BODY_PUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, bodyPushStrengthValueSpan);
}
bodyRepulsionStrengthSlider.oninput = function () {
    config.BODY_REPULSION_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, bodyRepulsionStrengthValueSpan);
}
bodyRepulsionRadiusFactorSlider.oninput = function () {
    config.BODY_REPULSION_RADIUS_FACTOR = parseFloat(this.value);
    updateSliderDisplay(this, bodyRepulsionRadiusFactorValueSpan);
}
globalMutationRateSlider.oninput = function () {
    config.GLOBAL_MUTATION_RATE_MODIFIER = parseFloat(this.value);
    updateSliderDisplay(this, globalMutationRateValueSpan);
}

baseNodeCostSlider.oninput = function () {
    config.BASE_NODE_EXISTENCE_COST = parseFloat(this.value);
    updateSliderDisplay(this, baseNodeCostValueSpan);
}
emitterNodeCostSlider.oninput = function () {
    config.EMITTER_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, emitterNodeCostValueSpan);
}
eaterNodeCostSlider.oninput = function () {
    config.EATER_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, eaterNodeCostValueSpan);
}
predatorNodeCostSlider.oninput = function () {
    config.PREDATOR_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, predatorNodeCostValueSpan);
}
neuronNodeCostSlider.oninput = function () {
    config.NEURON_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, neuronNodeCostValueSpan);
}
photosyntheticNodeCostSlider.oninput = function () {
    config.PHOTOSYNTHETIC_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, photosyntheticNodeCostValueSpan);
}
photosynthesisEfficiencySlider.oninput = function () {
    config.PHOTOSYNTHESIS_EFFICIENCY = parseFloat(this.value);
    updateSliderDisplay(this, photosynthesisEfficiencyValueSpan);
}
swimmerNodeCostSlider.oninput = function () {
    config.SWIMMER_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, swimmerNodeCostValueSpan);
}
jetNodeCostSlider.oninput = function () {
    config.JET_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, jetNodeCostValueSpan);
}
attractorNodeCostSlider.oninput = function () {
    config.ATTRACTOR_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, attractorNodeCostValueSpan);
}
repulsorNodeCostSlider.oninput = function () {
    config.REPULSOR_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, repulsorNodeCostValueSpan);
}
eyeNodeCostSlider.oninput = function () {
    config.EYE_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, eyeNodeCostValueSpan);
}
neuronChanceSlider.oninput = function() {
    config.NEURON_CHANCE = parseFloat(this.value);
    updateSliderDisplay(this, neuronChanceValueSpan);
}
jetMaxVelocityGeneSlider.oninput = function() {
    config.JET_MAX_VELOCITY_GENE_DEFAULT = parseFloat(this.value);
    updateSliderDisplay(this, jetMaxVelocityGeneValueSpan);
}

growthEnabledToggle.onchange = function () {
    config.GROWTH_ENABLED = this.checked;
}

growthEnergyCostScalarSlider.oninput = function () {
    config.GROWTH_ENERGY_COST_SCALAR = parseFloat(this.value);
    updateSliderDisplay(this, growthEnergyCostScalarValueSpan);
}

growthBaseChanceMinSlider.oninput = function () {
    const next = parseFloat(this.value);
    config.GROWTH_BASE_CHANCE_MIN = next;
    if (config.GROWTH_BASE_CHANCE_MAX < next) {
        config.GROWTH_BASE_CHANCE_MAX = next;
        growthBaseChanceMaxSlider.value = String(next);
        updateSliderDisplay(growthBaseChanceMaxSlider, growthBaseChanceMaxValueSpan);
    }
    updateSliderDisplay(this, growthBaseChanceMinValueSpan);
}

growthBaseChanceMaxSlider.oninput = function () {
    const next = parseFloat(this.value);
    config.GROWTH_BASE_CHANCE_MAX = next;
    if (config.GROWTH_BASE_CHANCE_MIN > next) {
        config.GROWTH_BASE_CHANCE_MIN = next;
        growthBaseChanceMinSlider.value = String(next);
        updateSliderDisplay(growthBaseChanceMinSlider, growthBaseChanceMinValueSpan);
    }
    updateSliderDisplay(this, growthBaseChanceMaxValueSpan);
}

growthPopSoftLimitMultiplierSlider.oninput = function () {
    const next = parseFloat(this.value);
    config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER = next;
    if (config.GROWTH_POP_HARD_LIMIT_MULTIPLIER < next + 0.1) {
        config.GROWTH_POP_HARD_LIMIT_MULTIPLIER = next + 0.1;
        growthPopHardLimitMultiplierSlider.value = String(config.GROWTH_POP_HARD_LIMIT_MULTIPLIER);
        updateSliderDisplay(growthPopHardLimitMultiplierSlider, growthPopHardLimitMultiplierValueSpan);
    }
    updateSliderDisplay(this, growthPopSoftLimitMultiplierValueSpan);
}

growthPopHardLimitMultiplierSlider.oninput = function () {
    const next = parseFloat(this.value);
    config.GROWTH_POP_HARD_LIMIT_MULTIPLIER = next;
    if (config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER > next - 0.1) {
        config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER = Math.max(1, next - 0.1);
        growthPopSoftLimitMultiplierSlider.value = String(config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER);
        updateSliderDisplay(growthPopSoftLimitMultiplierSlider, growthPopSoftLimitMultiplierValueSpan);
    }
    updateSliderDisplay(this, growthPopHardLimitMultiplierValueSpan);
}

growthSizeCostMaxMultiplierSlider.oninput = function () {
    config.GROWTH_SIZE_COST_MAX_MULTIPLIER = parseFloat(this.value);
    updateSliderDisplay(this, growthSizeCostMaxMultiplierValueSpan);
}

reproFertilityGlobalSoftMultiplierSlider.oninput = function () {
    const next = parseFloat(this.value);
    config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER = next;
    if (config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER < next + 0.1) {
        config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER = next + 0.1;
        reproFertilityGlobalHardMultiplierSlider.value = String(config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER);
        updateSliderDisplay(reproFertilityGlobalHardMultiplierSlider, reproFertilityGlobalHardMultiplierValueSpan);
    }
    updateSliderDisplay(this, reproFertilityGlobalSoftMultiplierValueSpan);
}

reproFertilityGlobalHardMultiplierSlider.oninput = function () {
    const next = parseFloat(this.value);
    config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER = next;
    if (config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER > next - 0.1) {
        config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER = Math.max(1, next - 0.1);
        reproFertilityGlobalSoftMultiplierSlider.value = String(config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER);
        updateSliderDisplay(reproFertilityGlobalSoftMultiplierSlider, reproFertilityGlobalSoftMultiplierValueSpan);
    }
    updateSliderDisplay(this, reproFertilityGlobalHardMultiplierValueSpan);
}

reproResourceMinNutrientSlider.oninput = function () {
    config.REPRO_RESOURCE_MIN_NUTRIENT = parseFloat(this.value);
    updateSliderDisplay(this, reproResourceMinNutrientValueSpan);
}

reproResourceMinLightSlider.oninput = function () {
    config.REPRO_RESOURCE_MIN_LIGHT = parseFloat(this.value);
    updateSliderDisplay(this, reproResourceMinLightValueSpan);
}

reproResourceNutrientDebitSlider.oninput = function () {
    config.REPRO_RESOURCE_NUTRIENT_DEBIT_PER_OFFSPRING = parseFloat(this.value);
    updateSliderDisplay(this, reproResourceNutrientDebitValueSpan);
}

reproResourceLightDebitSlider.oninput = function () {
    config.REPRO_RESOURCE_LIGHT_DEBIT_PER_OFFSPRING = parseFloat(this.value);
    updateSliderDisplay(this, reproResourceLightDebitValueSpan);
}


resetButton.onclick = function () {
    initializePopulation();
    config.isAnySoftBodyUnstable = false;
    updateInstabilityIndicator();
    // Reset mutation stats
    for (const key in mutationStats) {
        mutationStats[key] = 0;
    }
    // Reset global energy gains
    globalEnergyGains.photosynthesis = 0;
    globalEnergyGains.eating = 0;
    globalEnergyGains.predation = 0;

    // Reset global energy costs
    globalEnergyCosts.baseNodes = 0;
    globalEnergyCosts.emitterNodes = 0;
    globalEnergyCosts.eaterNodes = 0;
    globalEnergyCosts.predatorNodes = 0;
    globalEnergyCosts.neuronNodes = 0;
    globalEnergyCosts.swimmerNodes = 0;
    globalEnergyCosts.photosyntheticNodes = 0;
    globalEnergyCosts.grabbingNodes = 0;
    globalEnergyCosts.eyeNodes = 0;
    globalEnergyCosts.jetNodes = 0;
    globalEnergyCosts.attractorNodes = 0;
    globalEnergyCosts.repulsorNodes = 0;

    if (statsPanel.classList.contains('open')) {
        updateStatsPanel(); // Update if open
    }
}

resizeWorldButton.onclick = function () {
    const newWidth = parseInt(worldWidthInput.value);
    const newHeight = parseInt(worldHeightInput.value);

    if (isNaN(newWidth) || isNaN(newHeight) || newWidth < 500 || newHeight < 500 || newWidth > 40000 || newHeight > 40000) {
        showMessageModal("Invalid world dimensions. Min 500x500, Max 40000x40000.");
        worldWidthInput.value = config.WORLD_WIDTH;
        worldHeightInput.value = config.WORLD_HEIGHT;
        return;
    }
    config.WORLD_WIDTH = newWidth;
    config.WORLD_HEIGHT = newHeight;
    // canvas.width = config.WORLD_WIDTH; // Remove - canvas size is fixed
    // canvas.height = config.WORLD_HEIGHT; // Remove - canvas size is fixed
    // MAX_DISPLACEMENT_SQ_THRESHOLD = (config.WORLD_WIDTH / 5) * (config.WORLD_WIDTH / 5); // This constant is not used here

    viewOffsetX = 0;
    viewOffsetY = 0;

    initializeSpatialGrid();
    initFluidSimulation(USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initParticles();
    initializePopulation();
    initNutrientMap();
    initLightMap();
    initViscosityMap();
    isAnySoftBodyUnstable = false;
    updateInstabilityIndicator();
    console.log(`World resized to ${config.WORLD_WIDTH}x${config.WORLD_HEIGHT} and simulation reset.`);
}

fluidGridSizeSlider.oninput = function () {
    config.FLUID_GRID_SIZE_CONTROL = parseInt(this.value);
    updateSliderDisplay(this, fluidGridSizeValueSpan);
    velocityEmitters = [];
initFluidSimulation(USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initParticles();
    initNutrientMap();
    initLightMap();
    initViscosityMap();
}
fluidDiffusionSlider.oninput = function () {
    config.FLUID_DIFFUSION = parseFloat(this.value);
    updateSliderDisplay(this, fluidDiffusionValueSpan);
    if (fluidField) fluidField.diffusion = config.FLUID_DIFFUSION;
}
fluidViscositySlider.oninput = function () {
    config.FLUID_VISCOSITY = parseFloat(this.value);
    // console.log("Fluid Viscosity slider raw value:", this.value, "Parsed FLUID_VISCOSITY:", config.FLUID_VISCOSITY); // DEBUG
    updateSliderDisplay(this, fluidViscosityValueSpan);
    if (fluidField) fluidField.viscosity = config.FLUID_VISCOSITY;
}
fluidFadeSlider.oninput = function () {
    // Mobile sliders can be finicky at the minimum edge; snap near-zero to exactly zero.
    const raw = parseFloat(this.value);
    const step = parseFloat(this.step || '0.001');
    const snapped = raw <= (step * 0.5) ? 0 : raw;

    if (snapped === 0) {
        this.value = '0';
    }

    config.FLUID_FADE_RATE = snapped;
    updateSliderDisplay(this, fluidFadeValueSpan);
}
maxFluidVelocityComponentSlider.oninput = function () {
    config.MAX_FLUID_VELOCITY_COMPONENT = parseFloat(this.value);
    updateSliderDisplay(this, maxFluidVelocityComponentValueSpan);
    if (fluidField) fluidField.maxVelComponent = config.MAX_FLUID_VELOCITY_COMPONENT;
}
clearFluidButton.onclick = function () {
    if (fluidField) fluidField.clear();
}

particlesPerSecondSlider.oninput = function () {
    config.PARTICLES_PER_SECOND = parseInt(this.value);
    updateSliderDisplay(this, particlesPerSecondValueSpan);
}
particleFluidInfluenceSlider.oninput = function () {
    config.PARTICLE_FLUID_INFLUENCE = parseFloat(this.value);
    updateSliderDisplay(this, particleFluidInfluenceValueSpan);
}
particleLifeDecaySlider.oninput = function () {
    config.PARTICLE_BASE_LIFE_DECAY = parseFloat(this.value);
    updateSliderDisplay(this, particleLifeDecayValueSpan);
}
infiniteParticleLifeToggle.onchange = function () {
    config.IS_PARTICLE_LIFE_INFINITE = this.checked;
    particleLifeDecaySlider.disabled = config.IS_PARTICLE_LIFE_INFINITE;
    particleLifeDecayLabel.style.color = config.IS_PARTICLE_LIFE_INFINITE ? '#777' : '#ddd';
    particleLifeDecayValueSpan.style.color = config.IS_PARTICLE_LIFE_INFINITE ? '#777' : '#00aeff';
}
resetParticlesButton.onclick = function () {
    initParticles();
}

exportConfigButton.onclick = handleExportConfig;
importConfigButton.onclick = () => importConfigFile.click();
/**
 * Route config-file imports through the local UI handler.
 *
 * Note: this intentionally avoids dispatching through config.js,
 * which does not own the file-reader/import orchestration.
 */
importConfigFile.onchange = handleImportConfig;

if (exportStateButton) {
    exportStateButton.onclick = handleExportWorldState;
}
if (importStateButton) {
    importStateButton.onclick = () => importStateFile?.click();
}
if (importStateFile) {
    importStateFile.onchange = handleImportWorldState;
}
if (mobilePrevCreatureButton) {
    mobilePrevCreatureButton.onclick = () => cycleSelectedCreature(-1);
}
if (mobileNextCreatureButton) {
    mobileNextCreatureButton.onclick = () => cycleSelectedCreature(1);
}
if (mobileInfoToggleButton) {
    mobileInfoToggleButton.onclick = () => toggleInfoPanelVisibility();
}

closeInfoPanelButton.onclick = () => {
    infoPanel.classList.remove('open');
    config.selectedInspectBody = null;
    config.selectedInspectPoint = null;
    config.selectedInspectPointIndex = null;
}

showNutrientMapToggle.onchange = function () {
    config.SHOW_NUTRIENT_MAP = this.checked;
    console.log("[Debug] showNutrientMapToggle changed. SHOW_NUTRIENT_MAP is now:", config.SHOW_NUTRIENT_MAP);
};
nutrientEditModeToggle.onchange = function () {
    config.IS_NUTRIENT_EDIT_MODE = this.checked;
    if (config.IS_NUTRIENT_EDIT_MODE) {
        emitterEditModeToggle.checked = false;
        config.IS_EMITTER_EDIT_MODE = false;
        canvas.classList.remove('emitter-edit-mode');
        lightEditModeToggle.checked = false;
        config.IS_LIGHT_EDIT_MODE = false;
        viscosityEditModeToggle.checked = false;
        config.IS_VISCOSITY_EDIT_MODE = false;
    }
};

nutrientBrushValueSlider.oninput = function () {
    config.NUTRIENT_BRUSH_VALUE = parseFloat(this.value);
    updateSliderDisplay(this, nutrientBrushValueSpan);
}
nutrientBrushSizeSlider.oninput = function () {
    config.NUTRIENT_BRUSH_SIZE = parseInt(this.value);
    updateSliderDisplay(this, nutrientBrushSizeSpan);
}
nutrientBrushStrengthSlider.oninput = function () {
    config.NUTRIENT_BRUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, nutrientBrushStrengthSpan);
}
clearNutrientMapButton.onclick = function () {
    initNutrientMap();
    console.log("Nutrient map cleared.");
};

showLightMapToggle.onchange = function () {
    config.SHOW_LIGHT_MAP = this.checked;
    console.log("[Debug] showLightMapToggle changed. SHOW_LIGHT_MAP is now:", config.SHOW_LIGHT_MAP);
};
lightEditModeToggle.onchange = function () {
    config.IS_LIGHT_EDIT_MODE = this.checked;
    if (config.IS_LIGHT_EDIT_MODE) {
        emitterEditModeToggle.checked = false;
        config.IS_EMITTER_EDIT_MODE = false;
        nutrientEditModeToggle.checked = false;
        config.IS_NUTRIENT_EDIT_MODE = false;
        viscosityEditModeToggle.checked = false;
        config.IS_VISCOSITY_EDIT_MODE = false;
        canvas.classList.remove('emitter-edit-mode');
    }
};
lightBrushValueSlider.oninput = function () {
    config.LIGHT_BRUSH_VALUE = parseFloat(this.value);
    updateSliderDisplay(this, lightBrushValueSpan);
}
lightBrushSizeSlider.oninput = function () {
    config.LIGHT_BRUSH_SIZE = parseInt(this.value);
    updateSliderDisplay(this, lightBrushSizeSpan);
}
lightBrushStrengthSlider.oninput = function () {
    config.LIGHT_BRUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, lightBrushStrengthSpan);
}
clearLightMapButton.onclick = function () {
    initLightMap();
    console.log("Light map reset to surface pattern.");
};

showViscosityMapToggle.onchange = function () {
    config.SHOW_VISCOSITY_MAP = this.checked;
    console.log("[Debug] showViscosityMapToggle changed. SHOW_VISCOSITY_MAP is now:", config.SHOW_VISCOSITY_MAP);
};
viscosityEditModeToggle.onchange = function () {
    config.IS_VISCOSITY_EDIT_MODE = this.checked;
    if (config.IS_VISCOSITY_EDIT_MODE) {
        emitterEditModeToggle.checked = false;
        config.IS_EMITTER_EDIT_MODE = false;
        nutrientEditModeToggle.checked = false;
        config.IS_NUTRIENT_EDIT_MODE = false;
        lightEditModeToggle.checked = false;
        config.IS_LIGHT_EDIT_MODE = false;
        canvas.classList.remove('emitter-edit-mode');
    }
};
viscosityBrushValueSlider.oninput = function () {
    config.VISCOSITY_BRUSH_VALUE = parseFloat(this.value);
    updateSliderDisplay(this, viscosityBrushValueSpan);
}
viscosityBrushSizeSlider.oninput = function () {
    config.VISCOSITY_BRUSH_SIZE = parseInt(this.value);
    updateSliderDisplay(this, viscosityBrushSizeSpan);
}
viscosityBrushStrengthSlider.oninput = function () {
    config.VISCOSITY_BRUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, viscosityBrushStrengthSpan);
}
clearViscosityMapButton.onclick = function () {
    initViscosityMap();
    console.log("Viscosity map reset to normal.");
};

nutrientCyclePeriodSlider.oninput = function () {
    config.nutrientCyclePeriodSeconds = parseInt(this.value);
    updateSliderDisplay(this, nutrientCyclePeriodSpan);
};
nutrientCycleBaseAmplitudeSlider.oninput = function () {
    config.nutrientCycleBaseAmplitude = parseFloat(this.value);
    updateSliderDisplay(this, nutrientCycleBaseAmplitudeSpan);
};
nutrientCycleWaveAmplitudeSlider.oninput = function () {
    config.nutrientCycleWaveAmplitude = parseFloat(this.value);
    updateSliderDisplay(this, nutrientCycleWaveAmplitudeSpan);
};
lightCyclePeriodSlider.oninput = function () {
    config.lightCyclePeriodSeconds = parseInt(this.value);
    updateSliderDisplay(this, lightCyclePeriodSpan);
};

viewEntireSimButton.onclick = function () {
    const fitted = buildFitWorldCamera({
        worldWidth: config.WORLD_WIDTH,
        worldHeight: config.WORLD_HEIGHT,
        viewportWidth: canvas.clientWidth,
        viewportHeight: canvas.clientHeight,
        minZoom: 0.01,
        maxZoom: getSafeMaxZoom()
    });

    viewZoom = fitted.zoom;
    viewOffsetX = fitted.offsetX;
    viewOffsetY = fitted.offsetY;

    // Keep config and viewport camera state synchronized.
    config.viewZoom = viewZoom;
    config.viewOffsetX = viewOffsetX;
    config.viewOffsetY = viewOffsetY;

    viewport.zoom = viewZoom;
    viewport.offsetX = viewOffsetX;
    viewport.offsetY = viewOffsetY;
}

function copyInfoToClipboard() {
    if (!config.selectedInspectBody) {
        showMessageModal("No creature selected to copy info from.");
        return;
    }
    let infoText = "";
    const panelElements = infoPanel.querySelectorAll('.info-section p, .info-section h5, .point-info-entry p, .point-info-entry h6');

    panelElements.forEach(el => {
        let label = "";
        let value = "";
        if (el.tagName === 'H5' || el.tagName === 'H6') {
            infoText += el.textContent.trim() + "\n";
        } else if (el.tagName === 'P') {
            const strongTag = el.querySelector('strong');
            if (strongTag) {
                label = strongTag.textContent.trim();
                let allText = el.textContent.trim();
                value = allText.substring(label.length).trim();
                if (value.startsWith(":")) value = value.substring(1).trim();
                infoText += label + ": " + value + "\n";
            } else {
                infoText += el.textContent.trim() + "\n";
            }
        }
    });

    navigator.clipboard.writeText(infoText.trim()).then(() => {
        const originalText = copyInfoPanelButton.textContent;
        copyInfoPanelButton.textContent = "Copied!";
        setTimeout(() => {
            copyInfoPanelButton.textContent = originalText;
        }, 1500);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        showMessageModal("Failed to copy info. See console for details.");
    });
}

copyInfoPanelButton.onclick = copyInfoToClipboard;

function toggleInfoPanel() {
    infoPanel.classList.toggle('open');
    if (!infoPanel.classList.contains('open')) {
        config.selectedInspectBody = null;
        config.selectedInspectPoint = null;
        config.selectedInspectPointIndex = null;
    }
}

closeInfoPanelButton.onclick = toggleInfoPanel;

showFluidVelocityToggle.onchange = function () {
    config.SHOW_FLUID_VELOCITY = this.checked;
};

headlessModeToggle.onchange = function () {
    config.IS_HEADLESS_MODE = this.checked;
    if (config.IS_HEADLESS_MODE) {
        console.log("Headless mode enabled: Drawing will be skipped.");
    } else {
        console.log("Headless mode disabled: Drawing will resume.");
    }
};

if (useGpuFluidToggle) { // Check if the element exists before assigning onchange
    useGpuFluidToggle.onchange = function () {
        console.log("useGpuFluidToggle changed!"); // Add this unconditional log
        config.USE_GPU_FLUID_SIMULATION = this.checked;
        console.log(`GPU Fluid Simulation toggled: ${config.USE_GPU_FLUID_SIMULATION}. Re-initializing fluid simulation.`);
        initFluidSimulation(config.USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    };
} else {
    console.error("useGpuFluidToggle element not found!");
}

/**
 * Disable mobile touch gestures on the simulation canvas.
 *
 * Rationale: on phones/tablets, browser pan/zoom gestures can lock rendering/input.
 * We force explicit button-based mobile controls instead.
 */
canvas.style.touchAction = 'none';
const blockCanvasTouchGestures = (e) => {
    if (e.cancelable) {
        e.preventDefault();
    }
};
canvas.addEventListener('touchstart', blockCanvasTouchGestures, { passive: false });
canvas.addEventListener('touchmove', blockCanvasTouchGestures, { passive: false });
canvas.addEventListener('touchend', blockCanvasTouchGestures, { passive: false });
canvas.addEventListener('touchcancel', blockCanvasTouchGestures, { passive: false });
canvas.addEventListener('gesturestart', blockCanvasTouchGestures, { passive: false });
canvas.addEventListener('gesturechange', blockCanvasTouchGestures, { passive: false });
canvas.addEventListener('gestureend', blockCanvasTouchGestures, { passive: false });

canvas.addEventListener('mousedown', (e) => {
    updateMouse(e);

    if (e.button === 2) { // Right mouse button
        config.isRightDragging = true;
        mouse.isDown = false;
        config.panStartMouseDisplayX = mouse.x;
        config.panStartMouseDisplayY = mouse.y;
        config.panInitialViewOffsetX = config.viewOffsetX;
        config.panInitialViewOffsetY = config.viewOffsetY;
        e.preventDefault();
        return;
    } else if (e.button === 0) { // Left mouse button
        mouse.isDown = true;
        config.isRightDragging = false;

        const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
        const simMouseX = worldCoords.x;
        const simMouseY = worldCoords.y;

        if (config.IS_CREATURE_IMPORT_MODE && config.IMPORTED_CREATURE_DATA) {
            placeImportedCreature(simMouseX, simMouseY);
            return; // Exit after placing
        }

        if (config.IS_EMITTER_EDIT_MODE && fluidField) {
            if (config.IS_SIMULATION_PAUSED) return;
            const gridX = Math.floor(simMouseX / fluidField.scaleX);
            const gridY = Math.floor(simMouseY / fluidField.scaleY);
            config.emitterDragStartCell = {gridX, gridY, mouseStartX: simMouseX, mouseStartY: simMouseY};
            config.currentEmitterPreview = {
                startX: (gridX + 0.5) * fluidField.scaleX,
                startY: (gridY + 0.5) * fluidField.scaleY,
                endX: simMouseX,
                endY: simMouseY
            };
            config.selectedInspectBody = null;
            config.selectedInspectPoint = null;
            updateInfoPanel();
            return;
        }

        if (config.IS_NUTRIENT_EDIT_MODE) {
            if (config.IS_SIMULATION_PAUSED) return;
            config.isPaintingNutrients = true;
            paintNutrientBrush(simMouseX, simMouseY);
            config.selectedInspectBody = null;
            updateInfoPanel();
            config.selectedSoftBodyPoint = null;
            return;
        } else if (config.IS_LIGHT_EDIT_MODE) {
            if (config.IS_SIMULATION_PAUSED) return;
            config.isPaintingLight = true;
            paintLightBrush(simMouseX, simMouseY);
            config.selectedInspectBody = null;
            updateInfoPanel();
            config.selectedSoftBodyPoint = null;
            return;
        } else if (config.IS_VISCOSITY_EDIT_MODE) {
            if (config.IS_SIMULATION_PAUSED) return;
            config.isPaintingViscosity = true;
            paintViscosityBrush(simMouseX, simMouseY);
            config.selectedInspectBody = null;
            updateInfoPanel();
            config.selectedSoftBodyPoint = null;
            return;
        }

        let clickedOnPoint = false;
        // Try to find a new point that was clicked.
        for (let body of softBodyPopulation) {
            if (body.isUnstable) continue;
            for (let i = 0; i < body.massPoints.length; i++) {
                const point = body.massPoints[i];
                const dist = Math.sqrt((point.pos.x - simMouseX) ** 2 + (point.pos.y - simMouseY) ** 2);
                if (dist < point.radius * 2.5) {
                    // A point was clicked. Update the selection for both inspection and dragging.
                    config.selectedSoftBodyPoint = {body: body, point: point};
                    config.selectedInspectBody = body;
                    config.selectedInspectPoint = point;
                    config.selectedInspectPointIndex = i;

                    if (!config.IS_SIMULATION_PAUSED) {
                        point.isFixed = true;
                        point.prevPos.x = point.pos.x;
                        point.prevPos.y = point.pos.y;
                    }
                    clickedOnPoint = true;
                    break;
                }
            }
            if (clickedOnPoint) break;
        }

        if (!clickedOnPoint) {
            // No point was clicked, indicating an interaction with the fluid.
            // We keep the info panel open by not clearing selectedInspectBody.
            // We only clear the point for dragging to prevent moving the old selection.
            config.selectedSoftBodyPoint = null;
            disableAutoFollowForManualControl();
        }
        updateInfoPanel();
    }
});

canvas.addEventListener('mousemove', (e) => {
    updateMouse(e);

    if (config.isRightDragging) {
        disableAutoFollowForManualControl();
        const currentDisplayMouseX = mouse.x;
        const currentDisplayMouseY = mouse.y;

        const displayDx = currentDisplayMouseX - config.panStartMouseDisplayX;
        const displayDy = currentDisplayMouseY - config.panStartMouseDisplayY;

        // World shift per display pixel is simply 1/viewZoom (letterbox cancels because both points move within same bar)
        const panDeltaX_world = displayDx / config.viewZoom;
        const panDeltaY_world = displayDy / config.viewZoom;

        config.viewOffsetX = config.panInitialViewOffsetX - panDeltaX_world;
        config.viewOffsetY = config.panInitialViewOffsetY - panDeltaY_world;

        // Keep offsets valid / centred.
        clampViewOffsets();

        // Synchronize exported globals so the renderer uses the newest camera values.
        viewOffsetX = config.viewOffsetX;
        viewOffsetY = config.viewOffsetY;
        viewport.offsetX = viewOffsetX;
        viewport.offsetY = viewOffsetY;
        return;
    }


    const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
    const simMouseX = worldCoords.x;
    const simMouseY = worldCoords.y;

    const worldPrevCoords = getMouseWorldCoordinates(mouse.prevX, mouse.prevY);
    const worldMouseDx = simMouseX - worldPrevCoords.x;
    const worldMouseDy = simMouseY - worldPrevCoords.y;


    if (mouse.isDown && !config.IS_SIMULATION_PAUSED) { // Only do these if NOT paused
        if (config.IS_EMITTER_EDIT_MODE && config.emitterDragStartCell) {
            config.currentEmitterPreview.endX = simMouseX;
            config.currentEmitterPreview.endY = simMouseY;
        } else if (config.selectedSoftBodyPoint) {
            const point = config.selectedSoftBodyPoint.point;
            point.prevPos.x = point.pos.x;
            point.prevPos.y = point.pos.y;
            point.pos.x = simMouseX;
            point.pos.y = simMouseY;
        } else if (fluidField) {
            disableAutoFollowForManualControl();
            const fluidGridX = Math.floor(simMouseX / fluidField.scaleX);
            const fluidGridY = Math.floor(simMouseY / fluidField.scaleY);
            const r1 = Math.random() * 100 + 155;
            const g1 = Math.random() * 50 + 25;
            const b1 = Math.random() * 100 + 100;
            fluidField.addDensity(fluidGridX, fluidGridY, r1, g1, b1, 150 + Math.random() * 50);

            const r2 = Math.random() * 50 + 25;
            const g2 = Math.random() * 100 + 155;
            const b2 = Math.random() * 100 + 155;
            fluidField.addDensity(fluidGridX, fluidGridY, r2, g2, b2, 150 + Math.random() * 50);

            const fluidVelX = worldMouseDx * config.FLUID_MOUSE_DRAG_VELOCITY_SCALE;
            const fluidVelY = worldMouseDy * config.FLUID_MOUSE_DRAG_VELOCITY_SCALE;
            fluidField.addVelocity(fluidGridX, fluidGridY, fluidVelX, fluidVelY);
        }
    } else if (mouse.isDown && config.IS_EMITTER_EDIT_MODE && config.emitterDragStartCell) {
        // Allow emitter preview to update even if paused, but don't affect sim state
        config.currentEmitterPreview.endX = simMouseX;
        config.currentEmitterPreview.endY = simMouseY;
    }

    if (config.IS_NUTRIENT_EDIT_MODE && config.isPaintingNutrients && mouse.isDown && !config.IS_SIMULATION_PAUSED) {
        paintNutrientBrush(simMouseX, simMouseY);
        return;
    } else if (config.IS_LIGHT_EDIT_MODE && config.isPaintingLight && mouse.isDown && !config.IS_SIMULATION_PAUSED) {
        paintLightBrush(simMouseX, simMouseY);
        return;
    } else if (config.IS_VISCOSITY_EDIT_MODE && config.isPaintingViscosity && mouse.isDown && !config.IS_SIMULATION_PAUSED) {
        paintViscosityBrush(simMouseX, simMouseY);
        return;
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (e.button === 2) { // Right mouse button up
        config.isRightDragging = false;
        e.preventDefault();
    } else if (e.button === 0) { // Left mouse button up
        mouse.isDown = false;
        if (config.isPaintingNutrients) {
            config.isPaintingNutrients = false;
        }
        if (config.isPaintingLight) {
            config.isPaintingLight = false;
        }
        if (config.isPaintingViscosity) {
            config.isPaintingViscosity = false;
        }

        if (config.IS_EMITTER_EDIT_MODE && config.emitterDragStartCell && fluidField && !config.IS_SIMULATION_PAUSED) {
            const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
            const simMouseX = worldCoords.x;
            const simMouseY = worldCoords.y;

            const worldForceX = (simMouseX - config.emitterDragStartCell.mouseStartX) * config.EMITTER_MOUSE_DRAG_SCALE;
            const worldForceY = (simMouseY - config.emitterDragStartCell.mouseStartY) * config.EMITTER_MOUSE_DRAG_SCALE;

            const gridForceX = worldForceX / fluidField.scaleX;
            const gridForceY = worldForceY / fluidField.scaleY;


            const existingEmitter = config.velocityEmitters.find(em => em.gridX === config.emitterDragStartCell.gridX && em.gridY === config.emitterDragStartCell.gridY);
            if (existingEmitter) {
                existingEmitter.forceX = gridForceX;
                existingEmitter.forceY = gridForceY;
            } else {
                config.velocityEmitters.push({
                    gridX: config.emitterDragStartCell.gridX,
                    gridY: config.emitterDragStartCell.gridY,
                    forceX: gridForceX,
                    forceY: gridForceY
                });
            }
            config.emitterDragStartCell = null;
            config.currentEmitterPreview = null;
        }
        if (config.selectedSoftBodyPoint) {
            const point = config.selectedSoftBodyPoint.point;
            if (!config.IS_SIMULATION_PAUSED) {
                point.isFixed = false;
                const worldDx = (mouse.dx / Math.min(canvas.clientWidth / config.WORLD_WIDTH, canvas.clientHeight / config.WORLD_HEIGHT) / config.viewZoom);
                const worldDy = (mouse.dy / Math.min(canvas.clientWidth / config.WORLD_WIDTH, canvas.clientHeight / config.WORLD_HEIGHT) / config.viewZoom);
                point.prevPos.x = point.pos.x - worldDx * 1.0;
                point.prevPos.y = point.pos.y - worldDy * 1.0;
            }
            config.selectedSoftBodyPoint = null;
        }
    }
});
canvas.addEventListener('mouseleave', () => {
    mouse.isDown = false;
    config.isRightDragging = false;
    if (config.IS_EMITTER_EDIT_MODE && config.emitterDragStartCell) {
        const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
        const simMouseX = worldCoords.x;
        const simMouseY = worldCoords.y;
        const worldForceX = (simMouseX - config.emitterDragStartCell.mouseStartX) * config.EMITTER_MOUSE_DRAG_SCALE;
        const worldForceY = (simMouseY - config.emitterDragStartCell.mouseStartY) * config.EMITTER_MOUSE_DRAG_SCALE;
        const gridForceX = worldForceX / fluidField.scaleX;
        const gridForceY = worldForceY / fluidField.scaleY;
        const existingEmitter = config.velocityEmitters.find(em => em.gridX === config.emitterDragStartCell.gridX && em.gridY === config.emitterDragStartCell.gridY);
        if (existingEmitter) {
            existingEmitter.forceX = gridForceX;
            existingEmitter.forceY = gridForceY;
        } else {
            config.velocityEmitters.push({
                gridX: config.emitterDragStartCell.gridX,
                gridY: config.emitterDragStartCell.gridY,
                forceX: gridForceX,
                forceY: gridForceY
            });
        }
    }
    config.emitterDragStartCell = null;
    config.currentEmitterPreview = null;

    if (config.selectedSoftBodyPoint) {
        const point = config.selectedSoftBodyPoint.point;
        point.isFixed = false;
        const worldDx = (mouse.dx / Math.min(canvas.clientWidth / config.WORLD_WIDTH, canvas.clientHeight / config.WORLD_HEIGHT) / config.viewZoom);
        const worldDy = (mouse.dy / Math.min(canvas.clientWidth / config.WORLD_WIDTH, canvas.clientHeight / config.WORLD_HEIGHT) / config.viewZoom);
        point.prevPos.x = point.pos.x - worldDx * 1.0;
        point.prevPos.y = point.pos.y - worldDy * 1.0;
        config.selectedSoftBodyPoint = null;
    }

    if (config.isPaintingNutrients) {
        config.isPaintingNutrients = false;
    }
    if (config.isPaintingLight) {
        config.isPaintingLight = false;
    }
    if (config.isPaintingViscosity) {
        config.isPaintingViscosity = false;
    }
});

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (config.IS_CREATURE_IMPORT_MODE) {
        config.IS_CREATURE_IMPORT_MODE = false;
        config.IMPORTED_CREATURE_DATA = null;
        creatureImportStatus.textContent = "";
        canvas.style.cursor = 'default';
        console.log("Creature import cancelled via right-click.");
    }
});


canvas.addEventListener('wheel', (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    const scrollDir = e.deltaY < 0 ? 1 : -1;

    disableAutoFollowForManualControl();
    applyManualZoom(displayX, displayY, scrollDir);
});

eyeDetectionRadiusSlider.oninput = function () {
    config.EYE_DETECTION_RADIUS = parseInt(this.value);
    updateSliderDisplay(this, eyeDetectionRadiusValueSpan);
}

function updateStatsPanel() {
    if (!nodeTypeStatsDiv) return;
    const mutationTypeStatsDiv = document.getElementById('mutationTypeStats'); // Get the new div
    const globalEnergyGainsStatsDiv = document.getElementById('globalEnergyGainsStats');
    const globalEnergyCostsStatsDiv = document.getElementById('globalEnergyCostsStats'); // New: Get the costs div

    // Node Type Proportions
    nodeTypeStatsDiv.innerHTML = ''; // Clear existing content

    const nodeCounts = {};
    let totalNodes = 0;
    for (const typeName in NodeType) {
        nodeCounts[NodeType[typeName]] = 0;
    }
    softBodyPopulation.forEach(body => {
        body.massPoints.forEach(point => {
            if (nodeCounts[point.nodeType] !== undefined) {
                nodeCounts[point.nodeType]++;
            }
            totalNodes++;
        });
    });

    const title = document.createElement('p');
    title.innerHTML = "<strong>Node Type Proportions:</strong>";
    nodeTypeStatsDiv.appendChild(title);

    if (totalNodes === 0) {
        const noCreatures = document.createElement('p');
        noCreatures.textContent = "No creatures to analyze.";
        nodeTypeStatsDiv.appendChild(noCreatures);
    } else {
        for (const typeName in NodeType) {
            const typeEnum = NodeType[typeName];
            const count = nodeCounts[typeEnum];
            const percentage = ((count / totalNodes) * 100).toFixed(2);
            const typeString = getNodeTypeString(typeEnum);

            const statLineContainer = document.createElement('div');
            statLineContainer.className = 'stat-line-container';

            const p = document.createElement('p');
            p.style.margin = '0';
            p.innerHTML = `<strong>${typeString}:</strong> <span class="stat-value">${count} (${percentage}%)</span>`;
            statLineContainer.appendChild(p);

            if (count > 0) {
                const button = document.createElement('button');
                button.textContent = '🔍';
                button.classList.add('stats-panel-button');
                button.title = `Find next creature with a ${typeString} node`;
                button.onclick = () => { handleNodeTypeLabelClick(typeName); };
                statLineContainer.appendChild(button);
            }
            nodeTypeStatsDiv.appendChild(statLineContainer);
        }
        const totalP = document.createElement('p');
        totalP.innerHTML = `<strong>Total Nodes:</strong> <span class="stat-value">${totalNodes}</span>`;
        totalP.style.marginTop = '8px';
        nodeTypeStatsDiv.appendChild(totalP);
    }

    // Mutation Type Counts
    if (mutationTypeStatsDiv) {
        let mutationStatsHTML = "<p><strong>Mutation Occurrences:</strong></p>";
        let totalMutations = 0;
        for (const key in mutationStats) {
            mutationStatsHTML += `<p><strong>${key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:</strong> <span class=\"stat-value\">${mutationStats[key]}</span></p>`;
            totalMutations += mutationStats[key];
        }
        mutationStatsHTML += `<p><strong>Total Mutations Tracked:</strong> <span class=\"stat-value\">${totalMutations}</span></p>`;
        mutationTypeStatsDiv.innerHTML = mutationStatsHTML;
    }

    // Global Energy Gains
    if (globalEnergyGainsStatsDiv) {
        let energyGainsHTML = "<p><strong>Global Energy Gains (All Time):</strong></p>";
        energyGainsHTML += `<p><strong>Photosynthesis:</strong> <span class="stat-value">${globalEnergyGains.photosynthesis.toFixed(2)}</span></p>`;
        energyGainsHTML += `<p><strong>Eating:</strong> <span class="stat-value">${globalEnergyGains.eating.toFixed(2)}</span></p>`;
        energyGainsHTML += `<p><strong>Predation:</strong> <span class="stat-value">${globalEnergyGains.predation.toFixed(2)}</span></p>`;
        globalEnergyGainsStatsDiv.innerHTML = energyGainsHTML;
    }

    // Global Energy Costs
    if (globalEnergyCostsStatsDiv) {
        let energyCostsHTML = "<p><strong>Global Energy Costs (All Time):</strong></p>";
        energyCostsHTML += `<p><strong>Base Nodes:</strong> <span class="stat-value">${globalEnergyCosts.baseNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Emitter Nodes:</strong> <span class="stat-value">${globalEnergyCosts.emitterNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Eater Nodes:</strong> <span class="stat-value">${globalEnergyCosts.eaterNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Predator Nodes:</strong> <span class="stat-value">${globalEnergyCosts.predatorNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Neuron Nodes:</strong> <span class="stat-value">${globalEnergyCosts.neuronNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Swimmer Nodes:</strong> <span class="stat-value">${globalEnergyCosts.swimmerNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Jet Nodes:</strong> <span class="stat-value">${globalEnergyCosts.jetNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Attractor Nodes:</strong> <span class="stat-value">${globalEnergyCosts.attractorNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Repulsor Nodes:</strong> <span class="stat-value">${globalEnergyCosts.repulsorNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Photosynthetic Nodes:</strong> <span class="stat-value">${globalEnergyCosts.photosyntheticNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Grabbing Nodes:</strong> <span class="stat-value">${globalEnergyCosts.grabbingNodes.toFixed(2)}</span></p>`;
        energyCostsHTML += `<p><strong>Eye Nodes:</strong> <span class="stat-value">${globalEnergyCosts.eyeNodes.toFixed(2)}</span></p>`;
        globalEnergyCostsStatsDiv.innerHTML = energyCostsHTML;
    }
}

// Helper to create and insert info panel paragraphs if they don't exist
function createInfoPanelParagraph(parentElement, baseId, labelText) {
    let pElement = document.getElementById(baseId + 'P');
    if (!pElement) {
        pElement = document.createElement('p');
        pElement.id = baseId + 'P'; // e.g. infoBodyReproCooldownGeneP

        const strong = document.createElement('strong');
        strong.textContent = labelText;

        const span = document.createElement('span');
        span.id = baseId + 'Val'; // e.g. infoBodyReproCooldownGeneVal

        pElement.appendChild(strong);
        pElement.appendChild(document.createTextNode(' ')); // Space
        pElement.appendChild(span);

        // Insert before the 'Ticks Since Birth' paragraph, or at the end of the first .info-section
        const ticksBirthElement = document.getElementById('infoBodyTicksBirthP'); // Assuming Ticks Since Birth P has an ID like this
        const targetSection = parentElement.querySelector('.info-section h5 + p, .info-section h5') ?
            parentElement.querySelector('.info-section h5 + p, .info-section h5').closest('.info-section')
            : parentElement.querySelector('.info-section'); // Fallback to parentElement if it's the section

        if (targetSection) {
            const referenceNode = targetSection.querySelector('#infoBodyTicksBirthP') || targetSection.querySelector('#infoBodyRewardStrategyP'); // Try to insert before Ticks or Reward Strategy
            if (referenceNode) {
                targetSection.insertBefore(pElement, referenceNode);
            } else {
                targetSection.appendChild(pElement); // Fallback: append to section
            }
        } else {
            console.warn('Could not find target section to append info paragraph for', baseId);
            parentElement.appendChild(pElement); // Absolute fallback
        }
    }
    return pElement;
}

function focusOnCreature(creature) {
    if (!creature || creature.isUnstable || creature.massPoints.length === 0) return;

    const creatureCenter = creature.getAveragePosition();
    const creatureRadius = creature.blueprintRadius || creature.getBoundingBox().width / 2;

    const smallerViewportDim = Math.min(canvas.clientWidth, canvas.clientHeight);
    const targetZoom = smallerViewportDim / (creatureRadius * 2 * 3);

    viewZoom = Math.min(getSafeMaxZoom(), Math.max(0.2, targetZoom));

    const centered = centerCameraOnPoint({
        worldX: creatureCenter.x,
        worldY: creatureCenter.y,
        viewZoom,
        viewportWidth: canvas.clientWidth,
        viewportHeight: canvas.clientHeight
    });
    const clamped = clampCameraOffsets({
        offsetX: centered.offsetX,
        offsetY: centered.offsetY,
        viewZoom,
        worldWidth: config.WORLD_WIDTH,
        worldHeight: config.WORLD_HEIGHT,
        viewportWidth: canvas.clientWidth,
        viewportHeight: canvas.clientHeight
    });

    viewOffsetX = clamped.offsetX;
    viewOffsetY = clamped.offsetY;

    // Synchronize to config for helper consistency
    config.viewZoom = viewZoom;
    config.viewOffsetX = viewOffsetX;
    config.viewOffsetY = viewOffsetY;

    config.selectedInspectBody = creature;
    config.selectedInspectPoint = creature.massPoints[0];
    config.selectedInspectPointIndex = 0;
    updateInfoPanel();

    // Sync viewport camera
    viewport.zoom = viewZoom;
    viewport.offsetX = viewOffsetX;
    viewport.offsetY = viewOffsetY;

    clampViewOffsets();

    config.viewZoom = viewZoom;
}

function handleNodeTypeLabelClick(nodeTypeName) {
    const nodeType = NodeType[nodeTypeName];

    if (cyclingNodeType !== nodeType) {
        cyclingNodeType = nodeType;
        cyclingCreatureList = softBodyPopulation.filter(body =>
            !body.isUnstable && body.massPoints.some(point => point.nodeType === nodeType)
        );
        cyclingCreatureIndex = 0;
    } else if (cyclingCreatureList.length > 0) {
        cyclingCreatureIndex = (cyclingCreatureIndex + 1) % cyclingCreatureList.length;
    } else { // This case can happen if the list became empty since the last click
        cyclingCreatureList = softBodyPopulation.filter(body =>
            !body.isUnstable && body.massPoints.some(point => point.nodeType === nodeType)
        );
        cyclingCreatureIndex = 0;
    }

    if (cyclingCreatureList.length > 0) {
        const creatureToFocus = cyclingCreatureList[cyclingCreatureIndex];
        focusOnCreature(creatureToFocus);
    } else {
        console.log(`No creatures found with node type ${getNodeTypeString(nodeType)}`);
        cyclingNodeType = null;
    }
}

/**
 * Download a serialized world snapshot JSON file.
 */
function handleExportWorldState() {
    try {
        const snapshot = saveCurrentWorldSnapshot({
            trigger: 'browser-ui'
        });
        const jsonString = JSON.stringify(snapshot, null, 2);
        const blob = new Blob([jsonString], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const now = new Date();
        const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
        a.download = `sim_state_${stamp}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log('World state exported.');
    } catch (error) {
        console.error('Failed to export world state snapshot:', error);
        showMessageModal(`Failed to export world state: ${error.message}`);
    }
}

/**
 * Load a full world snapshot from disk and rehydrate runtime state.
 */
function handleImportWorldState(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const snapshot = JSON.parse(e.target.result);
            const loadInfo = loadWorldFromSnapshot(snapshot);
            console.log('World state imported successfully.', loadInfo || {});
            showMessageModal('World state loaded successfully.');
        } catch (error) {
            console.error('Error parsing/importing world state snapshot:', error);
            showMessageModal(`Failed to import world state: ${error.message}`);
        }
    };

    reader.readAsText(file);
    if (importStateFile) importStateFile.value = '';
}

function handleExportConfig() {
    const exportedConfig = {};
    for (const key in config) {
        // Exclude non-serializable or state-related properties
        if (typeof config[key] !== 'function' && key !== 'velocityEmitters' && key !== 'currentEmitterPreview' && key !== 'emitterDragStartCell' && key !== 'selectedInspectBody' && key !== 'selectedInspectPoint' && key !== 'selectedInspectPointIndex') {
            exportedConfig[key] = config[key];
        }
    }
    // Manually add velocityEmitters if needed, assuming it's an array of simple objects
    exportedConfig.velocityEmitters = config.velocityEmitters;

    const jsonString = JSON.stringify(exportedConfig, null, 2);
    const blob = new Blob([jsonString], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sim_config.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log("Config exported.");
}

function handleImportConfig(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const importedConfig = JSON.parse(e.target.result);
            applyImportedConfig(importedConfig);
            console.log("Config imported successfully.");
        } catch (error) {
            console.error("Error parsing imported config:", error);
            showMessageModal("Failed to import config. Make sure it's a valid JSON file.");
        }
    };
    reader.readAsText(file);
    if (importConfigFile) importConfigFile.value = ''; // Clear file input
}

function applyImportedConfig(importedConfig) {
    // Overwrite properties of the existing config object
    for (const key in importedConfig) {
        if (config.hasOwnProperty(key)) {
            config[key] = importedConfig[key];
        }
    }

    if (canvas) {
        // These might not be in the config file, so we handle them separately
        // Or better, ensure they are part of the export/import
        config.WORLD_WIDTH = importedConfig.WORLD_WIDTH || config.WORLD_WIDTH;
        config.WORLD_HEIGHT = importedConfig.WORLD_HEIGHT || config.WORLD_HEIGHT;
    }
    initializeSpatialGrid(); // This function now uses the config object

    // Re-initialize things that depend on the new config
    initializeAllSliderDisplays(); // This function should now be in ui.js and imported
    initFluidSimulation(config.USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initNutrientMap();
    initLightMap();
    initViscosityMap();
    initParticles();
    initializePopulation();
    console.log("Applied imported config. Reset population if needed for full effect on creatures.");
}

exportCreatureButton.onclick = handleExportCreature;
importCreatureButton.onclick = () => importCreatureFile.click();
importCreatureFile.onchange = handleImportCreature;

function handleExportCreature() {
    if (!config.selectedInspectBody) {
        showMessageModal("No creature selected to export.");
        return;
    }

    const creatureBlueprint = config.selectedInspectBody.exportBlueprint();
    const jsonString = JSON.stringify(creatureBlueprint, null, 2);
    const blob = new Blob([jsonString], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `creature_${config.selectedInspectBody.id}_blueprint.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`Exported blueprint for creature ${config.selectedInspectBody.id}.`);
}

function handleImportCreature(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const importedData = JSON.parse(e.target.result);
            // Basic validation
            if (importedData.version && importedData.blueprintPoints && importedData.blueprintSprings) {
                config.IMPORTED_CREATURE_DATA = importedData;
                config.IS_CREATURE_IMPORT_MODE = true;
                creatureImportStatus.textContent = "Click to place creature. Right-click to cancel.";
                canvas.style.cursor = 'copy';
                console.log("Creature blueprint loaded. Awaiting placement.");
            } else {
                throw new Error("Invalid creature blueprint file format.");
            }
        } catch (error) {
            console.error("Error parsing imported creature blueprint:", error);
            showMessageModal("Failed to import creature. Make sure it's a valid blueprint JSON file.");
            creatureImportStatus.textContent = "";
            config.IS_CREATURE_IMPORT_MODE = false;
            config.IMPORTED_CREATURE_DATA = null;
        }
    };
    reader.readAsText(file);
    importCreatureFile.value = ''; // Clear file input
}

function placeImportedCreature(worldX, worldY) {
    if (!config.IS_CREATURE_IMPORT_MODE || !config.IMPORTED_CREATURE_DATA) return;

    try {
        const newCreature = new SoftBody(nextSoftBodyId++, worldX, worldY, config.IMPORTED_CREATURE_DATA, true);
        newCreature.setNutrientField(nutrientField);
        newCreature.setLightField(lightField);
        newCreature.setParticles(particles);
        newCreature.setSpatialGrid(spatialGrid);
        softBodyPopulation.push(newCreature);
        console.log(`Placed imported creature with new ID ${newCreature.id} at (${worldX.toFixed(0)}, ${worldY.toFixed(0)}).`);

        // Keep import mode active to allow placing multiple creatures

    } catch(error) {
        console.error("Error creating creature from imported data:", error);
        showMessageModal("An error occurred while creating the creature from the blueprint.");
        config.IS_CREATURE_IMPORT_MODE = false;
        config.IMPORTED_CREATURE_DATA = null;
        creatureImportStatus.textContent = "";
        canvas.style.cursor = 'default';
    }
}

// Replace old exported globals with references to the viewport.
let viewZoom = viewport.zoom;
let viewOffsetX = viewport.offsetX;
let viewOffsetY = viewport.offsetY;

// Ensure config has corresponding initial values
config.viewZoom = viewZoom;
config.viewOffsetX = viewOffsetX;
config.viewOffsetY = viewOffsetY;

// Add just after the helper getMouseWorldCoordinates
// Ensure viewOffsetX/Y remain in a valid range. If the viewport is larger than the world in either
// dimension, we allow negative offsets so the world can be centred with letter-box margins.
function clampViewOffsets({ syncFromRuntime = false } = {}) {
    if (syncFromRuntime) {
        viewZoom = config.viewZoom;
        viewOffsetX = config.viewOffsetX;
        viewOffsetY = config.viewOffsetY;
    }

    // Self-heal invalid camera state (e.g., bad imported config or stale NaN state).
    if (!Number.isFinite(viewZoom) || viewZoom <= 0) {
        viewZoom = Math.max(0.01, Math.min(1.0, getSafeMaxZoom()));
    }
    if (!Number.isFinite(viewOffsetX)) viewOffsetX = 0;
    if (!Number.isFinite(viewOffsetY)) viewOffsetY = 0;

    const clamped = clampCameraOffsets({
        offsetX: viewOffsetX,
        offsetY: viewOffsetY,
        viewZoom,
        worldWidth: config.WORLD_WIDTH,
        worldHeight: config.WORLD_HEIGHT,
        viewportWidth: canvas.clientWidth,
        viewportHeight: canvas.clientHeight
    });

    viewOffsetX = clamped.offsetX;
    viewOffsetY = clamped.offsetY;

    // keep config & camera objects aligned
    config.viewZoom = viewZoom;
    config.viewOffsetX = viewOffsetX;
    config.viewOffsetY = viewOffsetY;
    viewport.zoom = viewZoom;
    viewport.offsetX = viewOffsetX;
    viewport.offsetY = viewOffsetY;
}

export { 
    canvas, webgpuCanvas, ctx, viewport as camera,
    worldWidthInput, worldHeightInput, 
    updateInstabilityIndicator, updatePopulationCount, updateStatsPanel, 
    updateInfoPanel, copyInfoToClipboard, toggleInfoPanel, toggleStatsPanel, 
    initializeAllSliderDisplays, randomizeControlsPanelConfig, handleExportConfig, handleImportConfig, applyImportedConfig,
    handleExportCreature, handleImportCreature, placeImportedCreature,
    updateMouse, getMouseWorldCoordinates,
    mouse, clampViewOffsets
};
