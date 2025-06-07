// --- DOM Element Selections for Sliders ---
const maxTimestepSlider = document.getElementById('maxTimestep');
const maxTimestepValueSpan = document.getElementById('maxTimestepValue');
const zoomSensitivitySlider = document.getElementById('zoomSensitivitySlider');
const zoomSensitivityValueSpan = document.getElementById('zoomSensitivityValueSpan');
const creaturePopulationFloorSlider = document.getElementById('creaturePopulationFloorSlider');
const creaturePopulationFloorValueSpan = document.getElementById('creaturePopulationFloorValueSpan');
const creaturePopulationCeilingSlider = document.getElementById('creaturePopulationCeilingSlider');
const creaturePopulationCeilingValueSpan = document.getElementById('creaturePopulationCeilingValueSpan');
const bodyFluidEntrainmentSlider = document.getElementById('bodyFluidEntrainment');
const bodyFluidEntrainmentValueSpan = document.getElementById('bodyFluidEntrainmentValue');
const fluidCurrentStrengthSlider = document.getElementById('fluidCurrentStrength');
const fluidCurrentStrengthValueSpan = document.getElementById('fluidCurrentStrengthValue');
const bodyPushStrengthSlider = document.getElementById('bodyPushStrength');
const bodyPushStrengthValueSpan = document.getElementById('bodyPushStrengthValue');
const bodyRepulsionStrengthSlider = document.getElementById('bodyRepulsionStrength');
const bodyRepulsionStrengthValueSpan = document.getElementById('bodyRepulsionStrengthValue');
const bodyRepulsionRadiusFactorSlider = document.getElementById('bodyRepulsionRadiusFactor');
const bodyRepulsionRadiusFactorValueSpan = document.getElementById('bodyRepulsionRadiusFactorValue');
const globalMutationRateSlider = document.getElementById('globalMutationRate');
const globalMutationRateValueSpan = document.getElementById('globalMutationRateValue');
const baseNodeCostSlider = document.getElementById('baseNodeCost');
const baseNodeCostValueSpan = document.getElementById('baseNodeCostValue');
const emitterNodeCostSlider = document.getElementById('emitterNodeCost');
const emitterNodeCostValueSpan = document.getElementById('emitterNodeCostValue');
const eaterNodeCostSlider = document.getElementById('eaterNodeCost');
const eaterNodeCostValueSpan = document.getElementById('eaterNodeCostValue');
const predatorNodeCostSlider = document.getElementById('predatorNodeCost');
const predatorNodeCostValueSpan = document.getElementById('predatorNodeCostValue');
const neuronNodeCostSlider = document.getElementById('neuronNodeCost');
const neuronNodeCostValueSpan = document.getElementById('neuronNodeCostValue');
const photosyntheticNodeCostSlider = document.getElementById('photosyntheticNodeCost');
const photosyntheticNodeCostValueSpan = document.getElementById('photosyntheticNodeCostValue');
const photosynthesisEfficiencySlider = document.getElementById('photosynthesisEfficiency');
const photosynthesisEfficiencyValueSpan = document.getElementById('photosynthesisEfficiencyValue');
const swimmerNodeCostSlider = document.getElementById('swimmerNodeCost');
const swimmerNodeCostValueSpan = document.getElementById('swimmerNodeCostValue');
const jetNodeCostSlider = document.getElementById('jetNodeCostSlider');
const jetNodeCostValueSpan = document.getElementById('jetNodeCostValueSpan');
const attractorNodeCostSlider = document.getElementById('attractorNodeCostSlider');
const attractorNodeCostValueSpan = document.getElementById('attractorNodeCostValueSpan');
const repulsorNodeCostSlider = document.getElementById('repulsorNodeCostSlider');
const repulsorNodeCostValueSpan = document.getElementById('repulsorNodeCostValueSpan');
const eyeNodeCostSlider = document.getElementById('eyeNodeCostSlider');
const eyeNodeCostValueSpan = document.getElementById('eyeNodeCostValueSpan');
const eyeDetectionRadiusSlider = document.getElementById('eyeDetectionRadiusSlider');
const eyeDetectionRadiusValueSpan = document.getElementById('eyeDetectionRadiusValueSpan');
const neuronChanceSlider = document.getElementById('neuronChanceSlider');
const neuronChanceValueSpan = document.getElementById('neuronChanceValueSpan');
const jetMaxVelocityGeneSlider = document.getElementById('jetMaxVelocityGeneSlider');
const jetMaxVelocityGeneValueSpan = document.getElementById('jetMaxVelocityGeneValueSpan');
const fluidGridSizeSlider = document.getElementById('fluidGridSize');
const fluidGridSizeValueSpan = document.getElementById('fluidGridSizeValue');
const fluidDiffusionSlider = document.getElementById('fluidDiffusion');
const fluidDiffusionValueSpan = document.getElementById('fluidDiffusionValue');
const fluidViscositySlider = document.getElementById('fluidViscosity');
const fluidViscosityValueSpan = document.getElementById('fluidViscosityValue');
const fluidFadeSlider = document.getElementById('fluidFade');
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
const emitterStrengthSlider = document.getElementById('emitterStrength');
const emitterStrengthValueSpan = document.getElementById('emitterStrengthValue');
const nutrientBrushValueSlider = document.getElementById('nutrientBrushValueSlider');
const nutrientBrushValueSpan = document.getElementById('nutrientBrushValueSpan');
const nutrientBrushSizeSlider = document.getElementById('nutrientBrushSizeSlider');
const nutrientBrushSizeSpan = document.getElementById('nutrientBrushSizeSpan');
const nutrientBrushStrengthSlider = document.getElementById('nutrientBrushStrengthSlider');
const nutrientBrushStrengthSpan = document.getElementById('nutrientBrushStrengthSpan');
const lightBrushValueSlider = document.getElementById('lightBrushValueSlider');
const lightBrushValueSpan = document.getElementById('lightBrushValueSpan');
const lightBrushSizeSlider = document.getElementById('lightBrushSizeSlider');
const lightBrushSizeSpan = document.getElementById('lightBrushSizeSpan');
const lightBrushStrengthSlider = document.getElementById('lightBrushStrengthSlider');
const lightBrushStrengthSpan = document.getElementById('lightBrushStrengthSpan');
const viscosityBrushValueSlider = document.getElementById('viscosityBrushValueSlider');
const viscosityBrushValueSpan = document.getElementById('viscosityBrushValueSpan');
const viscosityBrushSizeSlider = document.getElementById('viscosityBrushSizeSlider');
const viscosityBrushSizeSpan = document.getElementById('viscosityBrushSizeSpan');
const viscosityBrushStrengthSlider = document.getElementById('viscosityBrushStrengthSlider');
const viscosityBrushStrengthSpan = document.getElementById('viscosityBrushStrengthSpan');
const nutrientCyclePeriodSlider = document.getElementById('nutrientCyclePeriodSlider');
const nutrientCyclePeriodSpan = document.getElementById('nutrientCyclePeriodSpan');
const nutrientCycleBaseAmplitudeSlider = document.getElementById('nutrientCycleBaseAmplitudeSlider');
const nutrientCycleBaseAmplitudeSpan = document.getElementById('nutrientCycleBaseAmplitudeSpan');
const nutrientCycleWaveAmplitudeSlider = document.getElementById('nutrientCycleWaveAmplitudeSlider');
const nutrientCycleWaveAmplitudeSpan = document.getElementById('nutrientCycleWaveAmplitudeSpan');
const lightCyclePeriodSlider = document.getElementById('lightCyclePeriodSlider');
const lightCyclePeriodSpan = document.getElementById('lightCyclePeriodSpan');


// --- UI Update Functions ---
function updateSliderDisplay(slider, span) {
    let value = parseFloat(slider.value);
    if (!slider || !span) return;

    // Determine display format based on slider properties (e.g., step or id)
    const step = parseFloat(slider.step);
    if (step === 0.0005 || slider.id === 'zoomSensitivitySlider' || slider.id === 'particleLifeDecay') {
        span.textContent = value.toFixed(4);
    } else if (slider.id === 'bodyFluidEntrainment' || slider.id === 'fluidFade') { 
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
    } else { // Integer display for others
        span.textContent = Math.floor(value);
    }
}

function initializeAllSliderDisplays() {
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
        [attractorNodeCostSlider, "ATTRACTOR_NODE_ENERGY_COST", true, attractorNodeCostValueSpan],
        [repulsorNodeCostSlider, "REPULSOR_NODE_ENERGY_COST", true, repulsorNodeCostValueSpan],
        [photosyntheticNodeCostSlider, "PHOTOSYNTHETIC_NODE_ENERGY_COST", true, photosyntheticNodeCostValueSpan],
        [photosynthesisEfficiencySlider, "PHOTOSYNTHESIS_EFFICIENCY", true, photosynthesisEfficiencyValueSpan],
        [eyeNodeCostSlider, "EYE_NODE_ENERGY_COST", true, eyeNodeCostValueSpan],
        [eyeDetectionRadiusSlider, "EYE_DETECTION_RADIUS", false, eyeDetectionRadiusValueSpan],
        [neuronChanceSlider, "NEURON_CHANCE", true, neuronChanceValueSpan],
        [jetMaxVelocityGeneSlider, "JET_MAX_VELOCITY_GENE_DEFAULT", true, jetMaxVelocityGeneValueSpan],
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

    allSliders.forEach(([sliderElement, jsVarName, isFloat, spanElement]) => {
        if (sliderElement && typeof window[jsVarName] !== 'undefined') {
            window[jsVarName] = isFloat ? parseFloat(sliderElement.value) : parseInt(sliderElement.value);
            if (spanElement) {
                updateSliderDisplay(sliderElement, spanElement);
            }
        }
    });
}

// --- Event Listeners ---
maxTimestepSlider.oninput = function () {
    MAX_DELTA_TIME_MS = parseInt(this.value);
    updateSliderDisplay(this, maxTimestepValueSpan);
}
zoomSensitivitySlider.oninput = function () {
    ZOOM_SENSITIVITY = parseFloat(this.value);
    updateSliderDisplay(this, zoomSensitivityValueSpan);
}
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
emitterStrengthSlider.oninput = function () {
    EMITTER_STRENGTH = parseFloat(this.value);
    updateSliderDisplay(this, emitterStrengthValueSpan);
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
attractorNodeCostSlider.oninput = function () {
    ATTRACTOR_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, attractorNodeCostValueSpan);
}
repulsorNodeCostSlider.oninput = function () {
    REPULSOR_NODE_ENERGY_COST = parseFloat(this.value);
    updateSliderDisplay(this, repulsorNodeCostValueSpan);
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
eyeDetectionRadiusSlider.oninput = function () {
    EYE_DETECTION_RADIUS = parseInt(this.value);
    updateSliderDisplay(this, eyeDetectionRadiusValueSpan);
} 