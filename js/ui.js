// --- DOM Element Selections ---
const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d');
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
const eyeNodeCostSlider = document.getElementById('eyeNodeCostSlider');
const eyeDetectionRadiusSlider = document.getElementById('eyeDetectionRadiusSlider');
const neuronChanceSlider = document.getElementById('neuronChanceSlider');
const jetMaxVelocityGeneSlider = document.getElementById('jetMaxVelocityGeneSlider');

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
const eyeNodeCostValueSpan = document.getElementById('eyeNodeCostValueSpan');
const eyeDetectionRadiusValueSpan = document.getElementById('eyeDetectionRadiusValueSpan');
const neuronChanceValueSpan = document.getElementById('neuronChanceValueSpan');
const jetMaxVelocityGeneValueSpan = document.getElementById('jetMaxVelocityGeneValueSpan');

const fluidGridSizeSlider = document.getElementById('fluidGridSize');
const fluidGridSizeValueSpan = document.getElementById('fluidGridSizeValue');
const fluidDiffusionSlider = document.getElementById('fluidDiffusion');
const fluidViscositySlider = document.getElementById('fluidViscosity');
const fluidFadeSlider = document.getElementById('fluidFade');
const clearFluidButton = document.getElementById('clearFluidButton');
const fluidDiffusionValueSpan = document.getElementById('fluidDiffusionValue');
const fluidViscosityValueSpan = document.getElementById('fluidViscosityValue');
const fluidFadeValueSpan = document.getElementById('fluidFadeValue');
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
const headlessModeToggle = document.getElementById('headlessModeToggle'); // New headless mode toggle
const useGpuFluidToggle = document.getElementById('useGpuFluidToggle');
console.log("useGpuFluidToggle element:", useGpuFluidToggle); // Log to see if element is found

// --- Cycling through creatures with specific node types ---
let cyclingNodeType = null;
let cyclingCreatureList = [];
let cyclingCreatureIndex = -1;

// --- Mouse Interaction State Variables ---
let selectedSoftBodyPoint = null;
let mouse = {x: 0, y: 0, prevX: 0, prevY: 0, isDown: false, dx: 0, dy: 0};
// isRightDragging already declared in config.js
let panStartMouseDisplayX = 0;
let panStartMouseDisplayY = 0;
let panInitialViewOffsetX = 0;
let panInitialViewOffsetY = 0;

// --- UI Update Functions ---
function updateSliderDisplay(slider, span) {
    let value = parseFloat(slider.value);
    if (!slider || !span) return;

    // Determine display format based on slider properties (e.g., step or id)
    const step = parseFloat(slider.step);
    if (step === 0.0005 || slider.id === 'zoomSensitivitySlider' || slider.id === 'particleLifeDecay') {
        span.textContent = value.toFixed(4);
    } else if (slider.id === 'bodyFluidEntrainment' || slider.id === 'fluidFade') { // fluidFadeSlider was fluidFadeValueSpan
        span.textContent = value.toFixed(3);
    } else if (slider.id === 'globalMutationRate' || slider.id === 'bodyPushStrength' ||
        slider.id === 'photosyntheticNodeCost' || slider.id === 'maxFluidVelocityComponentSlider' ||
        slider.id === 'particleFluidInfluence' || slider.id === 'neuronChanceSlider') {
        span.textContent = value.toFixed(2);
    } else if (slider.id === 'fluidCurrentStrength' || slider.id === 'bodyRepulsionStrength' ||
        slider.id === 'bodyRepulsionRadiusFactor' || slider.id === 'baseNodeCost' ||
        slider.id === 'emitterNodeCost' || slider.id === 'eaterNodeCost' ||
        slider.id === 'predatorNodeCost' || slider.id === 'photosynthesisEfficiency' ||
        slider.id === 'emitterStrength') {
        span.textContent = value.toFixed(1);
    } else if (slider.id === 'fluidDiffusion' || slider.id === 'fluidViscosity') {
        span.textContent = value.toExponential(1);
    } else { // Integer display for others (like population counts, cooldowns)
        span.textContent = Math.floor(value);
    }
}

function initializeAllSliderDisplays() {
    // Array of [sliderElement, jsVariableString, needsFloatConversion]
    const allSliders = [
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
        [photosyntheticNodeCostSlider, "PHOTOSYNTHETIC_NODE_ENERGY_COST", true, photosyntheticNodeCostValueSpan],
        [photosynthesisEfficiencySlider, "PHOTOSYNTHESIS_EFFICIENCY", true, photosynthesisEfficiencyValueSpan],
        [eyeNodeCostSlider, "EYE_NODE_ENERGY_COST", true, eyeNodeCostValueSpan],
        [eyeDetectionRadiusSlider, "EYE_DETECTION_RADIUS", false, eyeDetectionRadiusValueSpan],
        [neuronChanceSlider, "NEURON_CHANCE", true, neuronChanceValueSpan],
        [jetMaxVelocityGeneSlider, "JET_MAX_VELOCITY_GENE_DEFAULT", true, jetMaxVelocityGeneValueSpan],
        [fluidGridSizeSlider, "FLUID_GRID_SIZE_CONTROL", false, fluidGridSizeValueSpan],
        [fluidDiffusionSlider, "FLUID_DIFFUSION", true, fluidDiffusionValueSpan],
        [fluidViscositySlider, "FLUID_VISCOSITY", true, fluidViscosityValueSpan],
        [fluidFadeSlider, "FLUID_FADE_RATE", true, fluidFadeValueSpan], // Corrected from fluidFadeValueSpan
        [maxTimestepSlider, "MAX_DELTA_TIME_MS", false, maxTimestepValueSpan],
        [maxFluidVelocityComponentSlider, "MAX_FLUID_VELOCITY_COMPONENT", true, maxFluidVelocityComponentValueSpan],
        [particlePopulationFloorSlider, "PARTICLE_POPULATION_FLOOR", false, particlePopulationFloorValueSpan],
        [particlePopulationCeilingSlider, "PARTICLE_POPULATION_CEILING", false, particlePopulationCeilingValueSpan],
        [particlesPerSecondSlider, "PARTICLES_PER_SECOND", false, particlesPerSecondValueSpan],
        [particleFluidInfluenceSlider, "PARTICLE_FLUID_INFLUENCE", true, particleFluidInfluenceValueSpan],
        [particleLifeDecaySlider, "PARTICLE_BASE_LIFE_DECAY", true, particleLifeDecayValueSpan],
        [emitterStrengthSlider, "EMITTER_STRENGTH", true, emitterStrengthValueSpan],
        // Nutrient map sliders
        [nutrientBrushValueSlider, "NUTRIENT_BRUSH_VALUE", true, nutrientBrushValueSpan],
        [nutrientBrushSizeSlider, "NUTRIENT_BRUSH_SIZE", false, nutrientBrushSizeSpan],
        [nutrientBrushStrengthSlider, "NUTRIENT_BRUSH_STRENGTH", true, nutrientBrushStrengthSpan],
        [nutrientCyclePeriodSlider, "nutrientCyclePeriodSeconds", false, nutrientCyclePeriodSpan],
        [nutrientCycleBaseAmplitudeSlider, "nutrientCycleBaseAmplitude", true, nutrientCycleBaseAmplitudeSpan],
        [nutrientCycleWaveAmplitudeSlider, "nutrientCycleWaveAmplitude", true, nutrientCycleWaveAmplitudeSpan],
        // Light map sliders
        [lightBrushValueSlider, "LIGHT_BRUSH_VALUE", true, lightBrushValueSpan],
        [lightBrushSizeSlider, "LIGHT_BRUSH_SIZE", false, lightBrushSizeSpan],
        [lightBrushStrengthSlider, "LIGHT_BRUSH_STRENGTH", true, lightBrushStrengthSpan],
        [lightCyclePeriodSlider, "lightCyclePeriodSeconds", false, lightCyclePeriodSpan],
        // Viscosity map sliders
        [viscosityBrushValueSlider, "VISCOSITY_BRUSH_VALUE", true, viscosityBrushValueSpan],
        [viscosityBrushSizeSlider, "VISCOSITY_BRUSH_SIZE", false, viscosityBrushSizeSpan],
        [viscosityBrushStrengthSlider, "VISCOSITY_BRUSH_STRENGTH", true, viscosityBrushStrengthSpan]
    ];

    worldWidthInput.value = WORLD_WIDTH;
    worldHeightInput.value = WORLD_HEIGHT;

    allSliders.forEach(([sliderElement, jsVarName, isFloat, spanElement]) => {
        if (sliderElement && typeof window[jsVarName] !== 'undefined') {
            // Set the JS global variable from the HTML slider's current value attribute
            window[jsVarName] = isFloat ? parseFloat(sliderElement.value) : parseInt(sliderElement.value);
            // Then update the display span to match this initial value
            if (spanElement) {
                updateSliderDisplay(sliderElement, spanElement);
            }
        } else {
            if (!sliderElement) console.warn(`Slider element for ${jsVarName} not found.`);
            if (typeof window[jsVarName] === 'undefined') console.warn(`Global JS variable ${jsVarName} not found.`);
        }
    });

    // Update checkbox states based on global JS variables (which have their defaults from config.js)
    particleLifeDecaySlider.disabled = IS_PARTICLE_LIFE_INFINITE;
    particleLifeDecayLabel.style.color = IS_PARTICLE_LIFE_INFINITE ? '#777' : '#ddd';
    particleLifeDecayValueSpan.style.color = IS_PARTICLE_LIFE_INFINITE ? '#777' : '#00aeff';
    worldWrapToggle.checked = IS_WORLD_WRAPPING;
    emitterEditModeToggle.checked = IS_EMITTER_EDIT_MODE;
    infiniteParticleLifeToggle.checked = IS_PARTICLE_LIFE_INFINITE;
    canvas.classList.toggle('emitter-edit-mode', IS_EMITTER_EDIT_MODE);

    showNutrientMapToggle.checked = SHOW_NUTRIENT_MAP;
    nutrientEditModeToggle.checked = IS_NUTRIENT_EDIT_MODE;
    showLightMapToggle.checked = SHOW_LIGHT_MAP;
    lightEditModeToggle.checked = IS_LIGHT_EDIT_MODE;
    showViscosityMapToggle.checked = SHOW_VISCOSITY_MAP;
    viscosityEditModeToggle.checked = IS_VISCOSITY_EDIT_MODE;
    showFluidVelocityToggle.checked = SHOW_FLUID_VELOCITY;
    headlessModeToggle.checked = IS_HEADLESS_MODE; // Ensure headless toggle reflects JS var on init
}

function updateInstabilityIndicator() {
    if (isAnySoftBodyUnstable) {
        instabilityLight.classList.add('unstable');
    } else {
        instabilityLight.classList.remove('unstable');
    }
}

function updatePopulationCount() {
    populationCountDisplay.textContent = `Population: ${softBodyPopulation.length}`;
    particleCountDisplay.textContent = `Particles: ${particles.length}`;
}

function updateInfoPanel() {
    if (selectedInspectBody && selectedInspectPoint) {
        document.getElementById('infoBodyId').textContent = selectedInspectBody.id;
        document.getElementById('infoBodyStiffness').textContent = selectedInspectBody.getAverageStiffness().toFixed(2);
        document.getElementById('infoBodyDamping').textContent = selectedInspectBody.getAverageDamping().toFixed(2);
        document.getElementById('infoBodyMotorInterval').textContent = selectedInspectBody.motorImpulseInterval;
        document.getElementById('infoBodyMotorCap').textContent = selectedInspectBody.motorImpulseMagnitudeCap.toFixed(2);
        document.getElementById('infoBodyEmitterStrength').textContent = selectedInspectBody.emitterStrength.toFixed(2);
        document.getElementById('infoBodyEmitterDirX').textContent = selectedInspectBody.emitterDirection.x.toFixed(2);
        document.getElementById('infoBodyEmitterDirY').textContent = selectedInspectBody.emitterDirection.y.toFixed(2);
        document.getElementById('infoBodyNumOffspring').textContent = selectedInspectBody.numOffspring;
        document.getElementById('infoBodyOffspringRadius').textContent = selectedInspectBody.offspringSpawnRadius.toFixed(1);
        document.getElementById('infoBodyPointAddChance').textContent = selectedInspectBody.pointAddChance.toFixed(3);
        document.getElementById('infoBodySpringConnectionRadius').textContent = selectedInspectBody.springConnectionRadius.toFixed(1);
        document.getElementById('infoBodyEnergy').textContent = selectedInspectBody.creatureEnergy.toFixed(2);
        document.getElementById('infoBodyReproEnergyThreshold').textContent = selectedInspectBody.reproductionEnergyThreshold;
        document.getElementById('infoBodyCurrentMaxEnergy').textContent = selectedInspectBody.currentMaxEnergy.toFixed(2);
        document.getElementById('infoBodyTicksBirth').textContent = selectedInspectBody.ticksSinceBirth;
        document.getElementById('infoBodyCanReproduce').textContent = selectedInspectBody.canReproduce;
        document.getElementById('infoBodyRewardStrategy').textContent = getRewardStrategyString(selectedInspectBody.rewardStrategy);
        document.getElementById('infoBodyEnergyPhoto').textContent = selectedInspectBody.energyGainedFromPhotosynthesis.toFixed(2);
        document.getElementById('infoBodyEnergyEat').textContent = selectedInspectBody.energyGainedFromEating.toFixed(2);
        document.getElementById('infoBodyEnergyPred').textContent = selectedInspectBody.energyGainedFromPredation.toFixed(2);

        // Populate new energy cost fields
        document.getElementById('infoBodyCostBase').textContent = selectedInspectBody.energyCostFromBaseNodes.toFixed(2);
        document.getElementById('infoBodyCostEmitter').textContent = selectedInspectBody.energyCostFromEmitterNodes.toFixed(2);
        document.getElementById('infoBodyCostEater').textContent = selectedInspectBody.energyCostFromEaterNodes.toFixed(2);
        document.getElementById('infoBodyCostPredator').textContent = selectedInspectBody.energyCostFromPredatorNodes.toFixed(2);
        document.getElementById('infoBodyCostNeuron').textContent = selectedInspectBody.energyCostFromNeuronNodes.toFixed(2);
        document.getElementById('infoBodyCostSwimmer').textContent = selectedInspectBody.energyCostFromSwimmerNodes.toFixed(2);
        document.getElementById('infoBodyCostJet').textContent = selectedInspectBody.energyCostFromJetNodes.toFixed(2);
        document.getElementById('infoBodyCostPhoto').textContent = selectedInspectBody.energyCostFromPhotosyntheticNodes.toFixed(2);
        document.getElementById('infoBodyCostGrabbing').textContent = selectedInspectBody.energyCostFromGrabbingNodes.toFixed(2);
        document.getElementById('infoBodyCostEye').textContent = selectedInspectBody.energyCostFromEyeNodes.toFixed(2);

        // Add display for new reproduction cooldown properties
        let reproGeneEl = document.getElementById('infoBodyReproCooldownGeneVal'); // Target the span directly
        let reproGenePEL = document.getElementById('infoBodyReproCooldownGeneP'); // Target the p element
        if (!reproGenePEL) { // If the paragraph doesn't exist, create it
            reproGenePEL = createInfoPanelParagraph(infoPanel.querySelector('.info-section'), 'infoBodyReproCooldownGene', 'Repro. Cooldown Gene:');
            reproGeneEl = reproGenePEL.querySelector('span'); // Get the span from the new P
        } else {
            reproGeneEl = reproGenePEL.querySelector('span'); // Ensure we have the span if P exists
        }
        if (reproGeneEl) reproGeneEl.textContent = selectedInspectBody.reproductionCooldownGene;


        let effectiveReproEl = document.getElementById('infoBodyEffectiveReproCooldownVal'); // Target the span
        let effectiveReproPEL = document.getElementById('infoBodyEffectiveReproCooldownP'); // Target the P
        if (!effectiveReproPEL) {
            effectiveReproPEL = createInfoPanelParagraph(infoPanel.querySelector('.info-section'), 'infoBodyEffectiveReproCooldown', 'Effective Repro. Cooldown:');
            effectiveReproEl = effectiveReproPEL.querySelector('span');
        } else {
            effectiveReproEl = effectiveReproPEL.querySelector('span');
        }
        if (effectiveReproEl) effectiveReproEl.textContent = selectedInspectBody.effectiveReproductionCooldown;

        allPointsInfoContainer.innerHTML = '<h5>All Mass Points</h5>';
        selectedInspectBody.massPoints.forEach((point, index) => {
            const pointEntryDiv = document.createElement('div');
            pointEntryDiv.className = 'point-info-entry';

            let content = `<p><strong>Point Index:</strong> ${index}</p>`;
            content += `<p><strong>Node Type:</strong> ${getNodeTypeString(point.nodeType)}</p>`;
            content += `<p><strong>Movement Type:</strong> ${getMovementTypeString(point.movementType)}</p>`;
            content += `<p><strong>Mass:</strong> ${point.mass.toFixed(2)}</p>`;
            content += `<p><strong>Radius:</strong> ${point.radius.toFixed(2)}</p>`;
            content += `<p><strong>World Pos:</strong> X: ${point.pos.x.toFixed(2)}, Y: ${point.pos.y.toFixed(2)}</p>`;
            content += `<p><strong>Can Be Grabber:</strong> ${point.canBeGrabber}</p>`;
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
                    content += `<p><strong>Target Distance:</strong> ${(point.nearestTargetMagnitude * EYE_DETECTION_RADIUS).toFixed(1)} (norm: ${point.nearestTargetMagnitude.toFixed(3)})</p>`;
                    content += `<p><strong>Target Angle:</strong> ${(point.nearestTargetDirection * 180 / Math.PI).toFixed(1)}&deg;</p>`;
                }
            }

            if (point.nodeType === NodeType.NEURON && point.neuronData) {
                if (typeof point.neuronData.hiddenLayerSize === 'undefined') {
                    const bodyIdForLog = selectedInspectBody ? selectedInspectBody.id : 'UnknownBody';
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

        if (!infoPanel.classList.contains('open')) {
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

        document.getElementById('infoBodyCostBase').textContent = '-';
        document.getElementById('infoBodyCostEmitter').textContent = '-';
        document.getElementById('infoBodyCostEater').textContent = '-';
        document.getElementById('infoBodyCostPredator').textContent = '-';
        document.getElementById('infoBodyCostNeuron').textContent = '-';
        document.getElementById('infoBodyCostSwimmer').textContent = '-';
        document.getElementById('infoBodyCostJet').textContent = '-';
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
    // displayMouseX, displayMouseY are mouse coordinates relative to the canvas CSS dimensions (e.g., from event.clientX - rect.left)

    // Canvas internal bitmap dimensions (now fixed, e.g., 1920x1080)
    const bitmapInternalWidth = canvas.width;
    const bitmapInternalHeight = canvas.height;

    // Canvas CSS dimensions (how it's displayed on the page)
    const cssClientWidth = canvas.clientWidth;
    const cssClientHeight = canvas.clientHeight;

    // How the internal bitmap is scaled to fit the CSS dimensions, maintaining aspect ratio
    const bitmapDisplayScale = Math.min(cssClientWidth / bitmapInternalWidth, cssClientHeight / bitmapInternalHeight);

    // The size of the (scaled) internal bitmap as it appears within the CSS box
    const displayedBitmapWidthInCss = bitmapInternalWidth * bitmapDisplayScale;
    const displayedBitmapHeightInCss = bitmapInternalHeight * bitmapDisplayScale;

    // Offset if the scaled bitmap is letterboxed/pillarboxed within the CSS box
    const letterboxOffsetXcss = (cssClientWidth - displayedBitmapWidthInCss) / 2;
    const letterboxOffsetYcss = (cssClientHeight - displayedBitmapHeightInCss) / 2;

    // Coordinates of the mouse click *on the scaled bitmap image*
    const mouseOnScaledBitmapX = displayMouseX - letterboxOffsetXcss;
    const mouseOnScaledBitmapY = displayMouseY - letterboxOffsetYcss;

    // Convert these coordinates to what they would be on the *unscaled internal bitmap*
    const mouseOnUnscaledBitmapX = mouseOnScaledBitmapX / bitmapDisplayScale;
    const mouseOnUnscaledBitmapY = mouseOnScaledBitmapY / bitmapDisplayScale;

    // Now, transform from unscaled internal bitmap coordinates to world coordinates
    // This transformation is the inverse of what's applied in the main draw call:
    // canvas_bitmap_x = (world_x - viewOffsetX) * viewZoom
    // canvas_bitmap_y = (world_y - viewOffsetY) * viewZoom
    const worldX = (mouseOnUnscaledBitmapX / viewZoom) + viewOffsetX;
    const worldY = (mouseOnUnscaledBitmapY / viewZoom) + viewOffsetY;

    return {x: worldX, y: worldY};
}

// --- Event Listeners ---
document.addEventListener('keydown', (e) => {
    // Allow 'P' to toggle pause, and WASD for panning regardless of pause state.
    if (e.key.toLowerCase() !== 'p' && e.key.toLowerCase() !== 'w' && e.key.toLowerCase() !== 'a' && e.key.toLowerCase() !== 's' && e.key.toLowerCase() !== 'd' && IS_SIMULATION_PAUSED) {
        return; // Only block other keys if paused
    }

    const effectiveViewportWidth = canvas.clientWidth / viewZoom;
    const effectiveViewportHeight = canvas.clientHeight / viewZoom;
    const maxPanX = Math.max(0, WORLD_WIDTH - effectiveViewportWidth);
    const maxPanY = Math.max(0, WORLD_HEIGHT - effectiveViewportHeight);
    const panSpeed = VIEW_PAN_SPEED / viewZoom;


    switch (e.key.toLowerCase()) {
        case 'w':
            viewOffsetY = Math.max(0, viewOffsetY - panSpeed);
            break;
        case 's':
            viewOffsetY = Math.min(maxPanY, viewOffsetY + panSpeed);
            break;
        case 'a':
            viewOffsetX = Math.max(0, viewOffsetX - panSpeed);
            break;
        case 'd':
            viewOffsetX = Math.min(maxPanX, viewOffsetX + panSpeed);
            break;
        case 'p':
            IS_SIMULATION_PAUSED = !IS_SIMULATION_PAUSED;
            pauseResumeButton.textContent = IS_SIMULATION_PAUSED ? "Resume" : "Pause";
            if (!IS_SIMULATION_PAUSED) {
                lastTime = performance.now(); // lastTime is in main.js
                requestAnimationFrame(gameLoop); // gameLoop is in main.js
            }
            break;
    }
});


worldWrapToggle.onchange = function () {
    IS_WORLD_WRAPPING = this.checked;
    if (fluidField) fluidField.useWrapping = IS_WORLD_WRAPPING;
}
maxTimestepSlider.oninput = function () {
    MAX_DELTA_TIME_MS = parseInt(this.value);
    updateSliderDisplay(this, maxTimestepValueSpan);
}
zoomSensitivitySlider.oninput = function () {
    ZOOM_SENSITIVITY = parseFloat(this.value);
    updateSliderDisplay(this, zoomSensitivityValueSpan);
}

pauseResumeButton.onclick = function () {
    IS_SIMULATION_PAUSED = !IS_SIMULATION_PAUSED;
    this.textContent = IS_SIMULATION_PAUSED ? "Resume" : "Pause";
    if (!IS_SIMULATION_PAUSED) {
        lastTime = performance.now(); // lastTime is in main.js
        requestAnimationFrame(gameLoop); // gameLoop is in main.js
    }
}
toggleControlsButton.onclick = function () {
    controlsPanel.classList.toggle('open');
}

toggleStatsPanelButton.onclick = function () {
    statsPanel.classList.toggle('open');
    if (statsPanel.classList.contains('open')) {
        updateStatsPanel();
    }
}

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
    CREATURE_POPULATION_FLOOR = parseInt(this.value);
    updateSliderDisplay(this, creaturePopulationFloorValueSpan);
}
creaturePopulationCeilingSlider.oninput = function () {
    CREATURE_POPULATION_CEILING = parseInt(this.value);
    updateSliderDisplay(this, creaturePopulationCeilingValueSpan);
}
particlePopulationFloorSlider.oninput = function () {
    PARTICLE_POPULATION_FLOOR = parseInt(this.value);
    updateSliderDisplay(this, particlePopulationFloorValueSpan);
}
particlePopulationCeilingSlider.oninput = function () {
    PARTICLE_POPULATION_CEILING = parseInt(this.value);
    updateSliderDisplay(this, particlePopulationCeilingValueSpan);
}


emitterEditModeToggle.onchange = function () {
    IS_EMITTER_EDIT_MODE = this.checked;
    canvas.classList.toggle('emitter-edit-mode', IS_EMITTER_EDIT_MODE);
    if (!IS_EMITTER_EDIT_MODE) {
        currentEmitterPreview = null;
        emitterDragStartCell = null;
    }
}
emitterStrengthSlider.oninput = function () {
    EMITTER_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, emitterStrengthValueSpan);
}
clearEmittersButton.onclick = function () {
    velocityEmitters = [];
}


bodyFluidEntrainmentSlider.oninput = function () {
    BODY_FLUID_ENTRAINMENT_FACTOR = parseFloat(this.value);
    updateSliderDisplay(this, bodyFluidEntrainmentValueSpan);
}
fluidCurrentStrengthSlider.oninput = function () {
    FLUID_CURRENT_STRENGTH_ON_BODY = parseFloat(this.value);
    updateSliderDisplay(this, fluidCurrentStrengthValueSpan);
}
bodyPushStrengthSlider.oninput = function () {
    SOFT_BODY_PUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, bodyPushStrengthValueSpan);
}
bodyRepulsionStrengthSlider.oninput = function () {
    BODY_REPULSION_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, bodyRepulsionStrengthValueSpan);
}
bodyRepulsionRadiusFactorSlider.oninput = function () {
    BODY_REPULSION_RADIUS_FACTOR = parseFloat(this.value);
    updateSliderDisplay(this, bodyRepulsionRadiusFactorValueSpan);
}
globalMutationRateSlider.oninput = function () {
    GLOBAL_MUTATION_RATE_MODIFIER = parseFloat(this.value);
    updateSliderDisplay(this, globalMutationRateValueSpan);
}

baseNodeCostSlider.oninput = function () {
    BASE_NODE_EXISTENCE_COST = parseFloat(this.value);
    updateSliderDisplay(this, baseNodeCostValueSpan);
}
emitterNodeCostSlider.oninput = function () {
    EMITTER_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, emitterNodeCostValueSpan);
}
eaterNodeCostSlider.oninput = function () {
    EATER_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, eaterNodeCostValueSpan);
}
predatorNodeCostSlider.oninput = function () {
    PREDATOR_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, predatorNodeCostValueSpan);
}
neuronNodeCostSlider.oninput = function () {
    NEURON_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, neuronNodeCostValueSpan);
}
photosyntheticNodeCostSlider.oninput = function () {
    PHOTOSYNTHETIC_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, photosyntheticNodeCostValueSpan);
}
photosynthesisEfficiencySlider.oninput = function () {
    PHOTOSYNTHESIS_EFFICIENCY = parseFloat(this.value);
    updateSliderDisplay(this, photosynthesisEfficiencyValueSpan);
}
swimmerNodeCostSlider.oninput = function () {
    SWIMMER_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, swimmerNodeCostValueSpan);
}
jetNodeCostSlider.oninput = function () {
    JET_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, jetNodeCostValueSpan);
}
eyeNodeCostSlider.oninput = function () {
    EYE_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, eyeNodeCostValueSpan);
}
neuronChanceSlider.oninput = function() {
    NEURON_CHANCE = parseFloat(this.value);
    updateSliderDisplay(this, neuronChanceValueSpan);
}
jetMaxVelocityGeneSlider.oninput = function() {
    JET_MAX_VELOCITY_GENE_DEFAULT = parseFloat(this.value);
    updateSliderDisplay(this, jetMaxVelocityGeneValueSpan);
}


resetButton.onclick = function () {
    initializePopulation();
    isAnySoftBodyUnstable = false;
    updateInstabilityIndicator();
    // Reset mutation stats
    for (const key in mutationStats) {
        mutationStats[key] = 0;
    }
    // Reset global energy gains
    globalEnergyGains.photosynthesis = 0;
    globalEnergyGains.eating = 0;
    globalEnergyGains.predation = 0;
    if (statsPanel.classList.contains('open')) {
        updateStatsPanel(); // Update if open
    }
}

resizeWorldButton.onclick = function () {
    const newWidth = parseInt(worldWidthInput.value);
    const newHeight = parseInt(worldHeightInput.value);

    if (isNaN(newWidth) || isNaN(newHeight) || newWidth < 500 || newHeight < 500 || newWidth > 40000 || newHeight > 40000) {
        showMessageModal("Invalid world dimensions. Min 500x500, Max 40000x40000.");
        worldWidthInput.value = WORLD_WIDTH;
        worldHeightInput.value = WORLD_HEIGHT;
        return;
    }
    WORLD_WIDTH = newWidth;
    WORLD_HEIGHT = newHeight;
    // canvas.width = WORLD_WIDTH; // Remove - canvas size is fixed
    // canvas.height = WORLD_HEIGHT; // Remove - canvas size is fixed
    // MAX_DISPLACEMENT_SQ_THRESHOLD = (WORLD_WIDTH / 5) * (WORLD_WIDTH / 5); // This constant is not used here

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
    console.log(`World resized to ${WORLD_WIDTH}x${WORLD_HEIGHT} and simulation reset.`);
}

fluidGridSizeSlider.oninput = function () {
    FLUID_GRID_SIZE_CONTROL = parseInt(this.value);
    updateSliderDisplay(this, fluidGridSizeValueSpan);
    velocityEmitters = [];
initFluidSimulation(USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initParticles();
    initNutrientMap();
    initLightMap();
    initViscosityMap();
}
fluidDiffusionSlider.oninput = function () {
    FLUID_DIFFUSION = parseFloat(this.value);
    updateSliderDisplay(this, fluidDiffusionValueSpan);
    if (fluidField) fluidField.diffusion = FLUID_DIFFUSION;
}
fluidViscositySlider.oninput = function () {
    FLUID_VISCOSITY = parseFloat(this.value);
    // console.log("Fluid Viscosity slider raw value:", this.value, "Parsed FLUID_VISCOSITY:", FLUID_VISCOSITY); // DEBUG
    updateSliderDisplay(this, fluidViscosityValueSpan);
    if (fluidField) fluidField.viscosity = FLUID_VISCOSITY;
}
fluidFadeSlider.oninput = function () {
    FLUID_FADE_RATE = parseFloat(this.value);
    updateSliderDisplay(this, fluidFadeValueSpan);
}
maxFluidVelocityComponentSlider.oninput = function () {
    MAX_FLUID_VELOCITY_COMPONENT = parseFloat(this.value);
    updateSliderDisplay(this, maxFluidVelocityComponentValueSpan);
    if (fluidField) fluidField.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT;
}
clearFluidButton.onclick = function () {
    if (fluidField) fluidField.clear();
}

particlesPerSecondSlider.oninput = function () {
    PARTICLES_PER_SECOND = parseInt(this.value);
    updateSliderDisplay(this, particlesPerSecondValueSpan);
}
particleFluidInfluenceSlider.oninput = function () {
    PARTICLE_FLUID_INFLUENCE = parseFloat(this.value);
    updateSliderDisplay(this, particleFluidInfluenceValueSpan);
}
particleLifeDecaySlider.oninput = function () {
    PARTICLE_BASE_LIFE_DECAY = parseFloat(this.value);
    updateSliderDisplay(this, particleLifeDecayValueSpan);
}
infiniteParticleLifeToggle.onchange = function () {
    IS_PARTICLE_LIFE_INFINITE = this.checked;
    particleLifeDecaySlider.disabled = IS_PARTICLE_LIFE_INFINITE;
    particleLifeDecayLabel.style.color = IS_PARTICLE_LIFE_INFINITE ? '#777' : '#ddd';
    particleLifeDecayValueSpan.style.color = IS_PARTICLE_LIFE_INFINITE ? '#777' : '#00aeff';
}
resetParticlesButton.onclick = function () {
    initParticles();
}

exportConfigButton.onclick = handleExportConfig;
importConfigButton.onclick = () => importConfigFile.click();
importConfigFile.onchange = (event) => handleImportConfig(event, canvas, webgpuCanvas);
closeInfoPanelButton.onclick = () => {
    infoPanel.classList.remove('open');
    selectedInspectBody = null;
    selectedInspectPoint = null;
}

showNutrientMapToggle.onchange = function () {
    SHOW_NUTRIENT_MAP = this.checked;
    console.log("[Debug] showNutrientMapToggle changed. SHOW_NUTRIENT_MAP is now:", SHOW_NUTRIENT_MAP);
};
nutrientEditModeToggle.onchange = function () {
    IS_NUTRIENT_EDIT_MODE = this.checked;
    if (IS_NUTRIENT_EDIT_MODE) {
        emitterEditModeToggle.checked = false;
        IS_EMITTER_EDIT_MODE = false;
        canvas.classList.remove('emitter-edit-mode');
        lightEditModeToggle.checked = false;
        IS_LIGHT_EDIT_MODE = false;
        viscosityEditModeToggle.checked = false;
        IS_VISCOSITY_EDIT_MODE = false;
    }
};

nutrientBrushValueSlider.oninput = function () {
    NUTRIENT_BRUSH_VALUE = parseFloat(this.value);
    updateSliderDisplay(this, nutrientBrushValueSpan);
}
nutrientBrushSizeSlider.oninput = function () {
    NUTRIENT_BRUSH_SIZE = parseInt(this.value);
    updateSliderDisplay(this, nutrientBrushSizeSpan);
}
nutrientBrushStrengthSlider.oninput = function () {
    NUTRIENT_BRUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, nutrientBrushStrengthSpan);
}
clearNutrientMapButton.onclick = function () {
    initNutrientMap();
    console.log("Nutrient map cleared.");
};

showLightMapToggle.onchange = function () {
    SHOW_LIGHT_MAP = this.checked;
    console.log("[Debug] showLightMapToggle changed. SHOW_LIGHT_MAP is now:", SHOW_LIGHT_MAP);
};
lightEditModeToggle.onchange = function () {
    IS_LIGHT_EDIT_MODE = this.checked;
    if (IS_LIGHT_EDIT_MODE) {
        emitterEditModeToggle.checked = false;
        IS_EMITTER_EDIT_MODE = false;
        nutrientEditModeToggle.checked = false;
        IS_NUTRIENT_EDIT_MODE = false;
        viscosityEditModeToggle.checked = false;
        IS_VISCOSITY_EDIT_MODE = false;
        canvas.classList.remove('emitter-edit-mode');
    }
};
lightBrushValueSlider.oninput = function () {
    LIGHT_BRUSH_VALUE = parseFloat(this.value);
    updateSliderDisplay(this, lightBrushValueSpan);
}
lightBrushSizeSlider.oninput = function () {
    LIGHT_BRUSH_SIZE = parseInt(this.value);
    updateSliderDisplay(this, lightBrushSizeSpan);
}
lightBrushStrengthSlider.oninput = function () {
    LIGHT_BRUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, lightBrushStrengthSpan);
}
clearLightMapButton.onclick = function () {
    initLightMap();
    console.log("Light map reset to surface pattern.");
};

showViscosityMapToggle.onchange = function () {
    SHOW_VISCOSITY_MAP = this.checked;
    console.log("[Debug] showViscosityMapToggle changed. SHOW_VISCOSITY_MAP is now:", SHOW_VISCOSITY_MAP);
};
viscosityEditModeToggle.onchange = function () {
    IS_VISCOSITY_EDIT_MODE = this.checked;
    if (IS_VISCOSITY_EDIT_MODE) {
        emitterEditModeToggle.checked = false;
        IS_EMITTER_EDIT_MODE = false;
        nutrientEditModeToggle.checked = false;
        IS_NUTRIENT_EDIT_MODE = false;
        lightEditModeToggle.checked = false;
        IS_LIGHT_EDIT_MODE = false;
        canvas.classList.remove('emitter-edit-mode');
    }
};
viscosityBrushValueSlider.oninput = function () {
    VISCOSITY_BRUSH_VALUE = parseFloat(this.value);
    updateSliderDisplay(this, viscosityBrushValueSpan);
}
viscosityBrushSizeSlider.oninput = function () {
    VISCOSITY_BRUSH_SIZE = parseInt(this.value);
    updateSliderDisplay(this, viscosityBrushSizeSpan);
}
viscosityBrushStrengthSlider.oninput = function () {
    VISCOSITY_BRUSH_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, viscosityBrushStrengthSpan);
}
clearViscosityMapButton.onclick = function () {
    initViscosityMap();
    console.log("Viscosity map reset to normal.");
};

nutrientCyclePeriodSlider.oninput = function () {
    nutrientCyclePeriodSeconds = parseInt(this.value);
    updateSliderDisplay(this, nutrientCyclePeriodSpan);
};
nutrientCycleBaseAmplitudeSlider.oninput = function () {
    nutrientCycleBaseAmplitude = parseFloat(this.value);
    updateSliderDisplay(this, nutrientCycleBaseAmplitudeSpan);
};
nutrientCycleWaveAmplitudeSlider.oninput = function () {
    nutrientCycleWaveAmplitude = parseFloat(this.value);
    updateSliderDisplay(this, nutrientCycleWaveAmplitudeSpan);
};
lightCyclePeriodSlider.oninput = function () {
    lightCyclePeriodSeconds = parseInt(this.value);
    updateSliderDisplay(this, lightCyclePeriodSpan);
};

viewEntireSimButton.onclick = function () {
    const targetZoomX = canvas.clientWidth / WORLD_WIDTH;
    const targetZoomY = canvas.clientHeight / WORLD_HEIGHT;
    viewZoom = Math.min(targetZoomX, targetZoomY); // Zoom to fit entire world
    viewZoom = Math.min(viewZoom, MAX_ZOOM); // Respect MAX_ZOOM
    // Ensure MIN_ZOOM calculation from wheel event is considered if it was more restrictive
    const minZoomToSeeAllX = canvas.clientWidth / WORLD_WIDTH;
    const minZoomToSeeAllY = canvas.clientHeight / WORLD_HEIGHT;
    let currentMinZoom = Math.min(minZoomToSeeAllX, minZoomToSeeAllY);
    currentMinZoom = Math.min(1.0, currentMinZoom);
    if (WORLD_WIDTH <= canvas.clientWidth && WORLD_HEIGHT <= canvas.clientHeight) {
        currentMinZoom = Math.min(minZoomToSeeAllX, minZoomToSeeAllY);
    } else {
        currentMinZoom = Math.min(minZoomToSeeAllX, minZoomToSeeAllY);
    }
    currentMinZoom = Math.max(0.01, currentMinZoom);
    viewZoom = Math.max(viewZoom, currentMinZoom); // Ensure we don't zoom out beyond what MIN_ZOOM allows

    // Center the view
    // Calculate the dimensions of the world as they would appear on screen at the new zoom level
    const worldDisplayWidth = WORLD_WIDTH * viewZoom;
    const worldDisplayHeight = WORLD_HEIGHT * viewZoom;

    // Calculate required offset to center this displayed world within the canvas
    // This calculation needs to be in world coordinates for viewOffsetX/Y
    viewOffsetX = (WORLD_WIDTH / 2) - (canvas.clientWidth / viewZoom / 2);
    viewOffsetY = (WORLD_HEIGHT / 2) - (canvas.clientHeight / viewZoom / 2);

    // Clamp offsets to prevent viewing outside world boundaries
    const effectiveViewportWidth = canvas.clientWidth / viewZoom;
    const effectiveViewportHeight = canvas.clientHeight / viewZoom;
    const maxPanX = Math.max(0, WORLD_WIDTH - effectiveViewportWidth);
    const maxPanY = Math.max(0, WORLD_HEIGHT - effectiveViewportHeight);
    viewOffsetX = Math.max(0, Math.min(viewOffsetX, maxPanX));
    viewOffsetY = Math.max(0, Math.min(viewOffsetY, maxPanY));
}

copyInfoPanelButton.onclick = function () {
    if (!selectedInspectBody) {
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

showFluidVelocityToggle.onchange = function () {
    SHOW_FLUID_VELOCITY = this.checked;
};

headlessModeToggle.onchange = function () {
    IS_HEADLESS_MODE = this.checked;
    if (IS_HEADLESS_MODE) {
        console.log("Headless mode enabled: Drawing will be skipped.");
    } else {
        console.log("Headless mode disabled: Drawing will resume.");
    }
};

if (useGpuFluidToggle) { // Check if the element exists before assigning onchange
    useGpuFluidToggle.onchange = function () {
        console.log("useGpuFluidToggle changed!"); // Add this unconditional log
        USE_GPU_FLUID_SIMULATION = this.checked;
        console.log(`GPU Fluid Simulation toggled: ${USE_GPU_FLUID_SIMULATION}. Re-initializing fluid simulation.`);
        initFluidSimulation(USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    };
} else {
    console.error("useGpuFluidToggle element not found!");
}

canvas.addEventListener('mousedown', (e) => {
    updateMouse(e);

    if (e.button === 2) { // Right mouse button
        isRightDragging = true;
        mouse.isDown = false;
        panStartMouseDisplayX = mouse.x;
        panStartMouseDisplayY = mouse.y;
        panInitialViewOffsetX = viewOffsetX;
        panInitialViewOffsetY = viewOffsetY;
        e.preventDefault();
        return;
    } else if (e.button === 0) { // Left mouse button
        mouse.isDown = true;
        isRightDragging = false;

        const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
        const simMouseX = worldCoords.x;
        const simMouseY = worldCoords.y;

        if (IS_CREATURE_IMPORT_MODE && IMPORTED_CREATURE_DATA) {
            placeImportedCreature(simMouseX, simMouseY);
            return; // Exit after placing
        }

        if (IS_EMITTER_EDIT_MODE && fluidField) {
            if (IS_SIMULATION_PAUSED) return;
            const gridX = Math.floor(simMouseX / fluidField.scaleX);
            const gridY = Math.floor(simMouseY / fluidField.scaleY);
            emitterDragStartCell = {gridX, gridY, mouseStartX: simMouseX, mouseStartY: simMouseY};
            currentEmitterPreview = {
                startX: (gridX + 0.5) * fluidField.scaleX,
                startY: (gridY + 0.5) * fluidField.scaleY,
                endX: simMouseX,
                endY: simMouseY
            };
            selectedInspectBody = null;
            selectedInspectPoint = null;
            updateInfoPanel();
            return;
        }

        if (IS_NUTRIENT_EDIT_MODE) {
            if (IS_SIMULATION_PAUSED) return;
            isPaintingNutrients = true;
            paintNutrientBrush(simMouseX, simMouseY);
            selectedInspectBody = null;
            updateInfoPanel();
            selectedSoftBodyPoint = null;
            return;
        } else if (IS_LIGHT_EDIT_MODE) {
            if (IS_SIMULATION_PAUSED) return;
            isPaintingLight = true;
            paintLightBrush(simMouseX, simMouseY);
            selectedInspectBody = null;
            updateInfoPanel();
            selectedSoftBodyPoint = null;
            return;
        } else if (IS_VISCOSITY_EDIT_MODE) {
            if (IS_SIMULATION_PAUSED) return;
            isPaintingViscosity = true;
            paintViscosityBrush(simMouseX, simMouseY);
            selectedInspectBody = null;
            updateInfoPanel();
            selectedSoftBodyPoint = null;
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
                    selectedSoftBodyPoint = {body: body, point: point};
                    selectedInspectBody = body;
                    selectedInspectPoint = point;
                    selectedInspectPointIndex = i;

                    if (!IS_SIMULATION_PAUSED) {
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
            selectedSoftBodyPoint = null;
        }
        updateInfoPanel();
    }
});

canvas.addEventListener('mousemove', (e) => {
    updateMouse(e);

    if (isRightDragging) {
        const currentDisplayMouseX = mouse.x;
        const currentDisplayMouseY = mouse.y;

        const displayDx = currentDisplayMouseX - panStartMouseDisplayX;
        const displayDy = currentDisplayMouseY - panStartMouseDisplayY;

        // Scale factor of the internal bitmap to its displayed size on the CSS canvas
        const bitmapDisplayScale = Math.min(canvas.clientWidth / canvas.width, canvas.clientHeight / canvas.height);

        // How much the world should shift, based on mouse movement on the displayed bitmap, scaled by current zoom
        const panDeltaX_world = displayDx / (bitmapDisplayScale * viewZoom);
        const panDeltaY_world = displayDy / (bitmapDisplayScale * viewZoom);

        viewOffsetX = panInitialViewOffsetX - panDeltaX_world;
        viewOffsetY = panInitialViewOffsetY - panDeltaY_world;

        const effectiveViewportWidth = canvas.clientWidth / viewZoom; // This is viewport width in world units
        const effectiveViewportHeight = canvas.clientHeight / viewZoom; // This is viewport height in world units
        // Correct maxPan calculations for clamping viewOffset
        const maxPanX = Math.max(0, WORLD_WIDTH - (canvas.clientWidth / bitmapDisplayScale / viewZoom));
        const maxPanY = Math.max(0, WORLD_HEIGHT - (canvas.clientHeight / bitmapDisplayScale / viewZoom));

        viewOffsetX = Math.max(0, Math.min(viewOffsetX, maxPanX));
        viewOffsetY = Math.max(0, Math.min(viewOffsetY, maxPanY));
        return;
    }


    const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
    const simMouseX = worldCoords.x;
    const simMouseY = worldCoords.y;

    const worldPrevCoords = getMouseWorldCoordinates(mouse.prevX, mouse.prevY);
    const worldMouseDx = simMouseX - worldPrevCoords.x;
    const worldMouseDy = simMouseY - worldPrevCoords.y;


    if (mouse.isDown && !IS_SIMULATION_PAUSED) { // Only do these if NOT paused
        if (IS_EMITTER_EDIT_MODE && emitterDragStartCell) {
            currentEmitterPreview.endX = simMouseX;
            currentEmitterPreview.endY = simMouseY;
        } else if (selectedSoftBodyPoint) {
            const point = selectedSoftBodyPoint.point;
            point.prevPos.x = point.pos.x;
            point.prevPos.y = point.pos.y;
            point.pos.x = simMouseX;
            point.pos.y = simMouseY;
        } else if (fluidField) {
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

            const fluidVelX = worldMouseDx * FLUID_MOUSE_DRAG_VELOCITY_SCALE;
            const fluidVelY = worldMouseDy * FLUID_MOUSE_DRAG_VELOCITY_SCALE;
            fluidField.addVelocity(fluidGridX, fluidGridY, fluidVelX, fluidVelY);
        }
    } else if (mouse.isDown && IS_EMITTER_EDIT_MODE && emitterDragStartCell) {
        // Allow emitter preview to update even if paused, but don't affect sim state
        currentEmitterPreview.endX = simMouseX;
        currentEmitterPreview.endY = simMouseY;
    }

    if (IS_NUTRIENT_EDIT_MODE && isPaintingNutrients && mouse.isDown && !IS_SIMULATION_PAUSED) {
        paintNutrientBrush(simMouseX, simMouseY);
        return;
    } else if (IS_LIGHT_EDIT_MODE && isPaintingLight && mouse.isDown && !IS_SIMULATION_PAUSED) {
        paintLightBrush(simMouseX, simMouseY);
        return;
    } else if (IS_VISCOSITY_EDIT_MODE && isPaintingViscosity && mouse.isDown && !IS_SIMULATION_PAUSED) {
        paintViscosityBrush(simMouseX, simMouseY);
        return;
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (e.button === 2) { // Right mouse button up
        isRightDragging = false;
        e.preventDefault();
    } else if (e.button === 0) { // Left mouse button up
        mouse.isDown = false;
        if (isPaintingNutrients) {
            isPaintingNutrients = false;
        }
        if (isPaintingLight) {
            isPaintingLight = false;
        }
        if (isPaintingViscosity) {
            isPaintingViscosity = false;
        }

        if (IS_EMITTER_EDIT_MODE && emitterDragStartCell && fluidField && !IS_SIMULATION_PAUSED) {
            const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
            const simMouseX = worldCoords.x;
            const simMouseY = worldCoords.y;

            const worldForceX = (simMouseX - emitterDragStartCell.mouseStartX) * EMITTER_MOUSE_DRAG_SCALE;
            const worldForceY = (simMouseY - emitterDragStartCell.mouseStartY) * EMITTER_MOUSE_DRAG_SCALE;

            const gridForceX = worldForceX / fluidField.scaleX;
            const gridForceY = worldForceY / fluidField.scaleY;


            const existingEmitter = velocityEmitters.find(em => em.gridX === emitterDragStartCell.gridX && em.gridY === emitterDragStartCell.gridY);
            if (existingEmitter) {
                existingEmitter.forceX = gridForceX;
                existingEmitter.forceY = gridForceY;
            } else {
                velocityEmitters.push({
                    gridX: emitterDragStartCell.gridX,
                    gridY: emitterDragStartCell.gridY,
                    forceX: gridForceX,
                    forceY: gridForceY
                });
            }
            emitterDragStartCell = null;
            currentEmitterPreview = null;
        }
        if (selectedSoftBodyPoint) {
            const point = selectedSoftBodyPoint.point;
            if (!IS_SIMULATION_PAUSED) {
                point.isFixed = false;
                const worldDx = (mouse.dx / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
                const worldDy = (mouse.dy / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
                point.prevPos.x = point.pos.x - worldDx * 1.0;
                point.prevPos.y = point.pos.y - worldDy * 1.0;
            }
            selectedSoftBodyPoint = null;
        }
    }
});
canvas.addEventListener('mouseleave', () => {
    mouse.isDown = false;
    isRightDragging = false;
    if (IS_EMITTER_EDIT_MODE && emitterDragStartCell) {
        const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
        const simMouseX = worldCoords.x;
        const simMouseY = worldCoords.y;
        const worldForceX = (simMouseX - emitterDragStartCell.mouseStartX) * EMITTER_MOUSE_DRAG_SCALE;
        const worldForceY = (simMouseY - emitterDragStartCell.mouseStartY) * EMITTER_MOUSE_DRAG_SCALE;
        const gridForceX = worldForceX / fluidField.scaleX;
        const gridForceY = worldForceY / fluidField.scaleY;
        const existingEmitter = velocityEmitters.find(em => em.gridX === emitterDragStartCell.gridX && em.gridY === emitterDragStartCell.gridY);
        if (existingEmitter) {
            existingEmitter.forceX = gridForceX;
            existingEmitter.forceY = gridForceY;
        } else {
            velocityEmitters.push({
                gridX: emitterDragStartCell.gridX,
                gridY: emitterDragStartCell.gridY,
                forceX: gridForceX,
                forceY: gridForceY
            });
        }
    }
    emitterDragStartCell = null;
    currentEmitterPreview = null;

    if (selectedSoftBodyPoint) {
        const point = selectedSoftBodyPoint.point;
        point.isFixed = false;
        const worldDx = (mouse.dx / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
        const worldDy = (mouse.dy / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
        point.prevPos.x = point.pos.x - worldDx * 1.0;
        point.prevPos.y = point.pos.y - worldDy * 1.0;
        selectedSoftBodyPoint = null;
    }

    if (isPaintingNutrients) {
        isPaintingNutrients = false;
    }
    if (isPaintingLight) {
        isPaintingLight = false;
    }
    if (isPaintingViscosity) {
        isPaintingViscosity = false;
    }
});

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (IS_CREATURE_IMPORT_MODE) {
        IS_CREATURE_IMPORT_MODE = false;
        IMPORTED_CREATURE_DATA = null;
        creatureImportStatus.textContent = "";
        canvas.style.cursor = 'default';
        console.log("Creature import cancelled via right-click.");
    }
});


canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const displayMouseX = e.clientX - rect.left;
    const displayMouseY = e.clientY - rect.top;

    // Get world coordinates of the mouse pointer BEFORE the zoom operation
    const worldMouseBeforeZoom = getMouseWorldCoordinates(displayMouseX, displayMouseY);

    const scroll = e.deltaY < 0 ? 1 : -1;
    const oldZoom = viewZoom;
    let newZoom = viewZoom * Math.pow(1 + ZOOM_SENSITIVITY * 10, scroll);

    const minZoomToSeeAllX = canvas.clientWidth / WORLD_WIDTH;
    const minZoomToSeeAllY = canvas.clientHeight / WORLD_HEIGHT;
    let dynamicMinZoom = Math.min(minZoomToSeeAllX, minZoomToSeeAllY);
    if (WORLD_WIDTH <= canvas.clientWidth && WORLD_HEIGHT <= canvas.clientHeight) {
        dynamicMinZoom = Math.min(minZoomToSeeAllX, minZoomToSeeAllY);
    } else {
        dynamicMinZoom = Math.min(minZoomToSeeAllX, minZoomToSeeAllY);
    }
    dynamicMinZoom = Math.max(0.01, dynamicMinZoom);

    viewZoom = Math.max(dynamicMinZoom, Math.min(newZoom, MAX_ZOOM));

    // After zoom, the same mouse display position will point to a different world coordinate.
    // We want the world point that was under the mouse before zoom to still be under the mouse.
    // world_mouse = (display_mouse_on_unscaled_bitmap / new_view_zoom) + new_view_offset_x
    // new_view_offset_x = world_mouse - (display_mouse_on_unscaled_bitmap / new_view_zoom)

    // To get display_mouse_on_unscaled_bitmap:
    const bitmapInternalWidth = canvas.width;
    const bitmapInternalHeight = canvas.height;
    const cssClientWidth = canvas.clientWidth;
    const cssClientHeight = canvas.clientHeight;
    const bitmapDisplayScale = Math.min(cssClientWidth / bitmapInternalWidth, cssClientHeight / bitmapInternalHeight);
    const displayedBitmapWidthInCss = bitmapInternalWidth * bitmapDisplayScale;
    const displayedBitmapHeightInCss = bitmapInternalHeight * bitmapDisplayScale;
    const letterboxOffsetXcss = (cssClientWidth - displayedBitmapWidthInCss) / 2;
    const letterboxOffsetYcss = (cssClientHeight - displayedBitmapHeightInCss) / 2;
    const mouseOnScaledBitmapX = displayMouseX - letterboxOffsetXcss;
    const mouseOnScaledBitmapY = displayMouseY - letterboxOffsetYcss;
    const mouseOnUnscaledBitmapX = mouseOnScaledBitmapX / bitmapDisplayScale;
    const mouseOnUnscaledBitmapY = mouseOnScaledBitmapY / bitmapDisplayScale;

    viewOffsetX = worldMouseBeforeZoom.x - (mouseOnUnscaledBitmapX / viewZoom);
    viewOffsetY = worldMouseBeforeZoom.y - (mouseOnUnscaledBitmapY / viewZoom);

    // Clamp offsets
    const maxPanX = Math.max(0, WORLD_WIDTH - (cssClientWidth / bitmapDisplayScale / viewZoom));
    const maxPanY = Math.max(0, WORLD_HEIGHT - (cssClientHeight / bitmapDisplayScale / viewZoom));
    viewOffsetX = Math.max(0, Math.min(viewOffsetX, maxPanX));
    viewOffsetY = Math.max(0, Math.min(viewOffsetY, maxPanY));
});

eyeDetectionRadiusSlider.oninput = function () {
    EYE_DETECTION_RADIUS = parseInt(this.value);
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

            const p = document.createElement('p');
            p.innerHTML = `<strong>${typeString}:</strong> <span class="stat-value">${count} (${percentage}%)</span>`;

            if (count > 0) {
                p.classList.add('clickable-stat');
                p.title = `Click to cycle through creatures with ${typeString} nodes`;
                p.onclick = () => { handleNodeTypeLabelClick(typeName); };
            }
            nodeTypeStatsDiv.appendChild(p);
        }
        const totalP = document.createElement('p');
        totalP.innerHTML = `<strong>Total Nodes:</strong> <span class="stat-value">${totalNodes}</span>`;
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
    
    viewZoom = Math.min(MAX_ZOOM, Math.max(0.2, targetZoom)); 

    viewOffsetX = creatureCenter.x - (canvas.clientWidth / viewZoom / 2);
    viewOffsetY = creatureCenter.y - (canvas.clientHeight / viewZoom / 2);

    const effectiveViewportWidth = canvas.clientWidth / viewZoom;
    const effectiveViewportHeight = canvas.clientHeight / viewZoom;
    const maxPanX = Math.max(0, WORLD_WIDTH - effectiveViewportWidth);
    const maxPanY = Math.max(0, WORLD_HEIGHT - effectiveViewportHeight);
    viewOffsetX = Math.max(0, Math.min(viewOffsetX, maxPanX));
    viewOffsetY = Math.max(0, Math.min(viewOffsetY, maxPanY));

    selectedInspectBody = creature;
    selectedInspectPoint = creature.massPoints[0];
    selectedInspectPointIndex = 0;
    updateInfoPanel();
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

exportCreatureButton.onclick = handleExportCreature;
importCreatureButton.onclick = () => importCreatureFile.click();
importCreatureFile.onchange = handleImportCreature;

function handleExportCreature() {
    if (!selectedInspectBody) {
        showMessageModal("No creature selected to export.");
        return;
    }

    const creatureBlueprint = selectedInspectBody.exportBlueprint();
    const jsonString = JSON.stringify(creatureBlueprint, null, 2);
    const blob = new Blob([jsonString], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `creature_${selectedInspectBody.id}_blueprint.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`Exported blueprint for creature ${selectedInspectBody.id}.`);
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
                IMPORTED_CREATURE_DATA = importedData;
                IS_CREATURE_IMPORT_MODE = true;
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
            IS_CREATURE_IMPORT_MODE = false;
            IMPORTED_CREATURE_DATA = null;
        }
    };
    reader.readAsText(file);
    importCreatureFile.value = ''; // Clear file input
}

function placeImportedCreature(worldX, worldY) {
    if (!IS_CREATURE_IMPORT_MODE || !IMPORTED_CREATURE_DATA) return;

    try {
        const newCreature = new SoftBody(nextSoftBodyId++, worldX, worldY, IMPORTED_CREATURE_DATA, true);
        softBodyPopulation.push(newCreature);
        console.log(`Placed imported creature with new ID ${newCreature.id} at (${worldX.toFixed(0)}, ${worldY.toFixed(0)}).`);

        // Keep import mode active to allow placing multiple creatures

    } catch(error) {
        console.error("Error creating creature from imported data:", error);
        showMessageModal("An error occurred while creating the creature from the blueprint.");
        IS_CREATURE_IMPORT_MODE = false;
        IMPORTED_CREATURE_DATA = null;
        creatureImportStatus.textContent = "";
        canvas.style.cursor = 'default';
    }
}