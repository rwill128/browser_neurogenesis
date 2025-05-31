// --- Viewport and World Dimensions ---
let WORLD_WIDTH = 8000;
let WORLD_HEIGHT = 6000;
const VIEW_PAN_SPEED = 80;

let viewOffsetX = 0;
let viewOffsetY = 0;
let viewZoom = 1.0;
let ZOOM_SENSITIVITY = 0.02;
const MAX_ZOOM = 8.0;
let MIN_ZOOM = 0.1;

// Spatial Grid for optimization
const GRID_CELL_SIZE = 100;
let GRID_COLS = Math.ceil(WORLD_WIDTH / GRID_CELL_SIZE);
let GRID_ROWS = Math.ceil(WORLD_HEIGHT / GRID_CELL_SIZE);

// Moved these critical constants higher up
const NEURON_CHANCE = 0.1;
const MAX_PIXELS_PER_FRAME_DISPLACEMENT = 100;
const MAX_SPRING_STRETCH_FACTOR = 4.0;
const MAX_SPAN_PER_POINT_FACTOR = GRID_CELL_SIZE * 2;
const DYE_PULL_RATE = 0.05;

// Original Radius Multipliers (will become base values)
const EATING_RADIUS_MULTIPLIER_BASE = 2.0; // Original was 3.5, reducing base
const PREDATION_RADIUS_MULTIPLIER_BASE = 1.5; // Original was 2.5, reducing base

// Exertion Bonuses for Radius Multipliers
const EATING_RADIUS_MULTIPLIER_MAX_BONUS = 3.0; // Max bonus at full exertion
const PREDATION_RADIUS_MULTIPLIER_MAX_BONUS = 2.5; // Max bonus at full exertion

const ENERGY_PER_PARTICLE = 25;
// Original Energy Sapped (will become base value)
const ENERGY_SAPPED_PER_PREDATION_BASE = 3; // Original was 5
// Exertion Bonus for Energy Sapped
const ENERGY_SAPPED_PER_PREDATION_MAX_BONUS = 7; // Max bonus at full exertion (total 10)

const MAX_CREATURE_ENERGY = 100;
const OFFSPRING_INITIAL_ENERGY_SHARE = 0.25;
const REPRODUCTION_ADDITIONAL_COST_FACTOR = 0.1;
const OFFSPRING_PLACEMENT_ATTEMPTS = 10;
const OFFSPRING_PLACEMENT_CLEARANCE_RADIUS = 50;
const MUTATION_RATE_PERCENT = 0.1;
const MUTATION_CHANCE_BOOL = 0.05;
const MUTATION_CHANCE_NODE_TYPE = 0.1;
const MUTATION_CHANCE_REASSIGN_NEURON_LINK = 0.02;
const ADD_POINT_MUTATION_CHANCE = 0.03;
const NEW_POINT_OFFSET_RADIUS = 15;
let isAnySoftBodyUnstable = false;
const RED_DYE_POISON_STRENGTH = 0.5; // Energy lost per unit of normalized red dye (0-1) per point, per dt=1 frame


// --- Global Variables & Constants (with initial hardcoded defaults) ---
let CREATURE_POPULATION_FLOOR = 10;
let CREATURE_POPULATION_CEILING = 1000;
let PARTICLE_POPULATION_FLOOR = 5000;
let PARTICLE_POPULATION_CEILING = 20000;
let canCreaturesReproduceGlobally = true;

let BODY_FLUID_ENTRAINMENT_FACTOR = 0.465;
let FLUID_CURRENT_STRENGTH_ON_BODY = 19.7;
let SOFT_BODY_PUSH_STRENGTH = 0.10;
let REPRODUCTION_COOLDOWN_TICKS = 1000;
let BODY_REPULSION_STRENGTH = 100.0;
let BODY_REPULSION_RADIUS_FACTOR = 5.0;
let GLOBAL_MUTATION_RATE_MODIFIER = 0.25;
let MAX_DELTA_TIME_MS = 10;
let IS_SIMULATION_PAUSED = false;
let IS_EMITTER_EDIT_MODE = false; // Default to false, will be set by checkbox
let EMITTER_STRENGTH = 3.0;
const EMITTER_MOUSE_DRAG_SCALE = 0.1;
const FLUID_MOUSE_DRAG_VELOCITY_SCALE = 0.1;

let BASE_NODE_EXISTENCE_COST = 0.3;
let EMITTER_NODE_ENERGY_COST = 1.0;
let EATER_NODE_ENERGY_COST = 0.3;
let PREDATOR_NODE_ENERGY_COST = 0.8;
let NEURON_NODE_ENERGY_COST = 0.01;
let PHOTOSYNTHETIC_NODE_ENERGY_COST = 0.01; // Added
let PHOTOSYNTHESIS_EFFICIENCY = 5; // Added


let FLUID_GRID_SIZE_CONTROL = 128;
let FLUID_DIFFUSION = 0.00047;
let FLUID_VISCOSITY = 0.000068;
let FLUID_FADE_RATE = 0.002;
let MAX_FLUID_VELOCITY_COMPONENT = 10.0; // Updated default
let IS_WORLD_WRAPPING = false; // Default to false, will be set by checkbox
let PARTICLES_PER_SECOND = 500;
let PARTICLE_FLUID_INFLUENCE = 2.0;
let PARTICLE_BASE_LIFE_DECAY = 0.001;
let IS_PARTICLE_LIFE_INFINITE = false; // Default to false, will be set by checkbox
const PARTICLE_LIFE_DECAY_RANDOM_FACTOR = 0.002;
let particleEmissionDebt = 0;

let IS_NUTRIENT_EDIT_MODE = false;
let SHOW_NUTRIENT_MAP = false;

let NUTRIENT_BRUSH_VALUE = 1.0;
let NUTRIENT_BRUSH_SIZE = 5; // In grid cells
let NUTRIENT_BRUSH_STRENGTH = 0.1; // 0 to 1, how quickly it applies
const MIN_NUTRIENT_VALUE = 0.1;
const MAX_NUTRIENT_VALUE = 2.0;
let isPaintingNutrients = false;

let IS_LIGHT_EDIT_MODE = false;
let SHOW_LIGHT_MAP = false; // Default to false
let LIGHT_BRUSH_VALUE = 0.5;
let LIGHT_BRUSH_SIZE = 5;
let LIGHT_BRUSH_STRENGTH = 0.1;
const MIN_LIGHT_VALUE = 0.0;
const MAX_LIGHT_VALUE = 1.0;
let isPaintingLight = false;

let IS_VISCOSITY_EDIT_MODE = false;
let SHOW_VISCOSITY_MAP = false;
let VISCOSITY_BRUSH_VALUE = 1.0;
let VISCOSITY_BRUSH_SIZE = 5;
let VISCOSITY_BRUSH_STRENGTH = 0.1;
const MIN_VISCOSITY_MULTIPLIER = 0.2;
const MAX_VISCOSITY_MULTIPLIER = 10.0; // Was 5.0
let isPaintingViscosity = false;

let totalSimulationTime = 0.0;
let nutrientCyclePeriodSeconds = 300; 
let nutrientCycleBaseAmplitude = 0.65;
let nutrientCycleWaveAmplitude = 0.35;
let lightCyclePeriodSeconds = 480; 
let globalNutrientMultiplier = 1.0;
let globalLightMultiplier = 1.0;

let INITIAL_POPULATION_SIZE; // Will be set after CREATURE_POPULATION_FLOOR is loaded/defaulted

let velocityEmitters = [];
let currentEmitterPreview = null;
let emitterDragStartCell = null;
let selectedInspectBody = null;
let selectedInspectPoint = null;
let selectedInspectPointIndex = -1;
let isRightDragging = false;
let lastPanMouseX = 0;
let lastPanMouseY = 0;

// --- Neural Network Constants ---
const NEURAL_INPUT_SIZE = 9; // 3 (dye RGB) + 1 (energy) + 2 (CoM rel pos) + 2 (CoM rel vel) + 1 (nutrient level)
const NEURAL_OUTPUTS_PER_EMITTER_SWIMMER = 14; // For Policy Gradient: 6 actions (12) + 1 exertion (2) = 14
const NEURAL_OUTPUTS_PER_EATER = 2; // For Policy Gradient: 1 exertion (mean + stdDev)
const NEURAL_OUTPUTS_PER_PREDATOR = 2; // For Policy Gradient: 1 exertion (mean + stdDev)

const DEFAULT_HIDDEN_LAYER_SIZE_MIN = 5;
const DEFAULT_HIDDEN_LAYER_SIZE_MAX = 30; // Reduced from 50 for initial testing
const MAX_NEURAL_FORCE_COMPONENT = 1.0; // Max output for a single force component from NN
const MAX_NEURAL_EMISSION_PULL_STRENGTH = 1.0; // Max output for emission pull strength

// --- RL Training Constants ---
const LEARNING_RATE = 0.001;
const DISCOUNT_FACTOR_GAMMA = 0.99;
const TRAINING_INTERVAL_FRAMES = 10; 

const DyeChannel = {
    RED: 0,
    GREEN: 1,
    BLUE: 2,
    AVERAGE: 3
};
const DYE_COLORS = {
    RED: [200, 50, 50],
    GREEN: [50, 200, 50],
    BLUE: [50, 50, 200]
};

// --- Slider Config Loading ---
async function loadAndApplySliderConfig() {
    let configUrl;
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "" || window.location.protocol === "file:") {
        // Added window.location.hostname === "" for cases when opened as a local file directly not through a server
        configUrl = 'slider_config.json'; // Assumes slider_config.json is in the same directory or accessible via relative path
    } else {
        // Likely running on GitHub Pages or another deployed server
        configUrl = 'https://rwill128.github.io/browser_neurogenesis/slider_config.json';
    }

    try {
        console.log(`Fetching slider config from: ${configUrl}`); // Log which URL is being used
        const response = await fetch(configUrl);
        if (!response.ok) {
            // If fetching from the primary URL fails, try to fall back to a local version if not already tried.
            // This is a basic fallback, could be more sophisticated.
            if (configUrl !== 'slider_config.json' && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "" || window.location.protocol === "file:")) {
                console.warn(`Failed to fetch from ${configUrl}, attempting fallback to local 'slider_config.json'.`);
                configUrl = 'slider_config.json';
                const fallbackResponse = await fetch(configUrl);
                if (!fallbackResponse.ok) {
                    throw new Error(`HTTP error! status: ${fallbackResponse.status} on fallback URL. Using hardcoded defaults.`);
                }
                const sliderConfigs = await fallbackResponse.json();
                window.loadedSliderConfigs = sliderConfigs;
                applySliderConfigData(sliderConfigs); // Use a helper to apply
                console.log("Slider configuration loaded and applied from LOCAL FALLBACK URL.");
                return; // Exit after successful fallback
            }
            throw new Error(`HTTP error! status: ${response.status}. Using hardcoded defaults.`);
        }
        const sliderConfigs = await response.json();
        window.loadedSliderConfigs = sliderConfigs; // Store for updateSliderDisplay
        applySliderConfigData(sliderConfigs); // Use a helper to apply

        console.log("Slider configuration loaded and applied from URL.");
    } catch (error) {
        console.error("Could not load or apply slider_config.json. Using hardcoded defaults (already set).", error);
        // Fallbacks are already set by initial 'let' declarations for JS variables.
        // We might want to ensure sliders in HTML still get their min/max/step from a hardcoded JS object if fetch totally fails.
        // For now, this relies on HTML defaults if JSON load fails and no JS defaults were set by slider_config.json for HTML attributes.
    }
}

function applySliderConfigData(configs) { // Helper function to apply slider data
    if (!configs) return;
    configs.forEach(config => {
        const slider = document.getElementById(config.id);
        if (slider) {
            slider.min = config.min;
            slider.max = config.max;
            slider.step = config.step;
            slider.value = config.defaultValue;

            // Update global JS variable directly
            if (config.jsVariable) {
                if (typeof window[config.jsVariable] !== 'undefined') {
                    window[config.jsVariable] = parseFloat(config.defaultValue);
                } else {
                    console.warn(`Global variable ${config.jsVariable} was not pre-initialized. Setting it now.`);
                    window[config.jsVariable] = parseFloat(config.defaultValue);
                }
            }
        }
    });
}

// --- Config Import/Export ---
function handleExportConfig() {
    const config = {
        worldWidth: WORLD_WIDTH,
        worldHeight: WORLD_HEIGHT,
        zoomSensitivity: ZOOM_SENSITIVITY,
        creaturePopulationFloor: CREATURE_POPULATION_FLOOR,
        creaturePopulationCeiling: CREATURE_POPULATION_CEILING,
        particlePopulationFloor: PARTICLE_POPULATION_FLOOR,
        particlePopulationCeiling: PARTICLE_POPULATION_CEILING,
        maxFluidVelocityComponent: MAX_FLUID_VELOCITY_COMPONENT, // New
        bodyFluidEntrainment: BODY_FLUID_ENTRAINMENT_FACTOR,
        fluidCurrentStrength: FLUID_CURRENT_STRENGTH_ON_BODY,
        softBodyPushStrength: SOFT_BODY_PUSH_STRENGTH,
        baseNodeCost: BASE_NODE_EXISTENCE_COST,
        emitterNodeCost: EMITTER_NODE_ENERGY_COST,
        neuronNodeCost: NEURON_NODE_ENERGY_COST,
        eaterNodeCost: EATER_NODE_ENERGY_COST,
        predatorNodeCost: PREDATOR_NODE_ENERGY_COST,
        reproductionCooldown: REPRODUCTION_COOLDOWN_TICKS,
        bodyRepulsionStrength: BODY_REPULSION_STRENGTH,
        bodyRepulsionRadiusFactor: BODY_REPULSION_RADIUS_FACTOR,
        maxTimestepMs: MAX_DELTA_TIME_MS,
        globalMutationRate: GLOBAL_MUTATION_RATE_MODIFIER,
        fluidGridSize: FLUID_GRID_SIZE_CONTROL,
        fluidDiffusion: FLUID_DIFFUSION,
        fluidViscosity: FLUID_VISCOSITY,
        fluidFadeRate: FLUID_FADE_RATE,
        isWorldWrapping: IS_WORLD_WRAPPING,
        particlesPerSecond: PARTICLES_PER_SECOND,
        particleFluidInfluence: PARTICLE_FLUID_INFLUENCE,
        particleBaseLifeDecay: PARTICLE_BASE_LIFE_DECAY,
        isParticleLifeInfinite: IS_PARTICLE_LIFE_INFINITE,
        emitterStrength: EMITTER_STRENGTH,
        velocityEmitters: velocityEmitters,
        viewZoom: viewZoom,
        viewOffsetX: viewOffsetX,
        viewOffsetY: viewOffsetY,
        nutrientFieldData: nutrientField ? Array.from(nutrientField) : null,
        nutrientMapSize: nutrientField ? Math.round(FLUID_GRID_SIZE_CONTROL) : 0,
        lightFieldData: lightField ? Array.from(lightField) : null,
        lightMapSize: lightField ? Math.round(FLUID_GRID_SIZE_CONTROL) : 0,
        viscosityFieldData: viscosityField ? Array.from(viscosityField) : null,
        viscosityMapSize: viscosityField ? Math.round(FLUID_GRID_SIZE_CONTROL) : 0,
        photosyntheticNodeCost: PHOTOSYNTHETIC_NODE_ENERGY_COST,
        photosynthesisEfficiency: PHOTOSYNTHESIS_EFFICIENCY
    };
    const jsonString = JSON.stringify(config, null, 2);
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
            alert("Failed to import config. Make sure it's a valid JSON file.");
        }
    };
    reader.readAsText(file);
    importConfigFile.value = '';
}

function applyImportedConfig(config) {
    if (config.worldWidth !== undefined) WORLD_WIDTH = config.worldWidth;
    if (config.worldHeight !== undefined) WORLD_HEIGHT = config.worldHeight;
    canvas.width = WORLD_WIDTH;
    canvas.height = WORLD_HEIGHT;
    MAX_DISPLACEMENT_SQ_THRESHOLD = (WORLD_WIDTH / 5) * (WORLD_WIDTH / 5); // Update threshold
    initializeSpatialGrid(); // Re-initialize grid with new world dimensions

    if (config.zoomSensitivity !== undefined) ZOOM_SENSITIVITY = config.zoomSensitivity;
    if (config.creaturePopulationFloor !== undefined) CREATURE_POPULATION_FLOOR = config.creaturePopulationFloor;
    if (config.creaturePopulationCeiling !== undefined) CREATURE_POPULATION_CEILING = config.creaturePopulationCeiling;
    if (config.particlePopulationFloor !== undefined) PARTICLE_POPULATION_FLOOR = config.particlePopulationFloor;
    if (config.particlePopulationCeiling !== undefined) PARTICLE_POPULATION_CEILING = config.particlePopulationCeiling;
    if (config.maxFluidVelocityComponent !== undefined) MAX_FLUID_VELOCITY_COMPONENT = config.maxFluidVelocityComponent; // New

    if (config.bodyFluidEntrainment !== undefined) BODY_FLUID_ENTRAINMENT_FACTOR = config.bodyFluidEntrainment;
    if (config.fluidCurrentStrength !== undefined) FLUID_CURRENT_STRENGTH_ON_BODY = config.fluidCurrentStrength;
    if (config.softBodyPushStrength !== undefined) SOFT_BODY_PUSH_STRENGTH = config.softBodyPushStrength;
    if (config.reproductionCooldown !== undefined) REPRODUCTION_COOLDOWN_TICKS = config.reproductionCooldown;
    if (config.bodyRepulsionStrength !== undefined) BODY_REPULSION_STRENGTH = config.bodyRepulsionStrength;
    if (config.bodyRepulsionRadiusFactor !== undefined) BODY_REPULSION_RADIUS_FACTOR = config.bodyRepulsionRadiusFactor;
    if (config.maxTimestepMs !== undefined) MAX_DELTA_TIME_MS = config.maxTimestepMs;
    if (config.globalMutationRate !== undefined) GLOBAL_MUTATION_RATE_MODIFIER = config.globalMutationRate;

    if (config.baseNodeCost !== undefined) BASE_NODE_EXISTENCE_COST = config.baseNodeCost;
    if (config.emitterNodeCost !== undefined) EMITTER_NODE_ENERGY_COST = config.emitterNodeCost;
    if (config.neuronNodeCost !== undefined) NEURON_NODE_ENERGY_COST = config.neuronNodeCost;
    if (config.eaterNodeCost !== undefined) EATER_NODE_ENERGY_COST = config.eaterNodeCost;
    if (config.predatorNodeCost !== undefined) PREDATOR_NODE_ENERGY_COST = config.predatorNodeCost;
    if (config.emitterStrength !== undefined) EMITTER_STRENGTH = config.emitterStrength;
    if (config.velocityEmitters !== undefined) velocityEmitters = config.velocityEmitters;
    if (config.fluidGridSize !== undefined) FLUID_GRID_SIZE_CONTROL = config.fluidGridSize;
    if (config.viewZoom !== undefined) viewZoom = config.viewZoom;
    if (config.viewOffsetX !== undefined) viewOffsetX = config.viewOffsetX;
    if (config.viewOffsetY !== undefined) viewOffsetY = config.viewOffsetY;


    if (config.fluidDiffusion !== undefined) FLUID_DIFFUSION = config.fluidDiffusion;
    if (config.fluidViscosity !== undefined) FLUID_VISCOSITY = config.fluidViscosity;
    if (config.fluidFadeRate !== undefined) FLUID_FADE_RATE = config.fluidFadeRate;
    if (config.isWorldWrapping !== undefined) IS_WORLD_WRAPPING = config.isWorldWrapping;

    if (config.particlesPerSecond !== undefined) PARTICLES_PER_SECOND = config.particlesPerSecond;
    if (config.particleFluidInfluence !== undefined) PARTICLE_FLUID_INFLUENCE = config.particleFluidInfluence;
    if (config.particleBaseLifeDecay !== undefined) PARTICLE_BASE_LIFE_DECAY = config.particleBaseLifeDecay;
    if (config.isParticleLifeInfinite !== undefined) IS_PARTICLE_LIFE_INFINITE = config.isParticleLifeInfinite;

    if (config.photosyntheticNodeCost !== undefined) PHOTOSYNTHETIC_NODE_ENERGY_COST = config.photosyntheticNodeCost;
    if (config.photosynthesisEfficiency !== undefined) PHOTOSYNTHESIS_EFFICIENCY = config.photosynthesisEfficiency;
    
    // Update UI elements to reflect imported values
    // This should be done *before* initializeAllSliderDisplays if that function reads from DOM for initial values
    // Or, ensure initializeAllSliderDisplays *sets* DOM values from these JS vars.
    worldWidthInput.value = WORLD_WIDTH;
    worldHeightInput.value = WORLD_HEIGHT;
    zoomSensitivitySlider.value = ZOOM_SENSITIVITY;
    creaturePopulationFloorSlider.value = CREATURE_POPULATION_FLOOR;
    creaturePopulationCeilingSlider.value = CREATURE_POPULATION_CEILING;
    particlePopulationFloorSlider.value = PARTICLE_POPULATION_FLOOR;
    particlePopulationCeilingSlider.value = PARTICLE_POPULATION_CEILING;
    maxFluidVelocityComponentSlider.value = MAX_FLUID_VELOCITY_COMPONENT;
    bodyFluidEntrainmentSlider.value = BODY_FLUID_ENTRAINMENT_FACTOR;
    fluidCurrentStrengthSlider.value = FLUID_CURRENT_STRENGTH_ON_BODY;
    bodyPushStrengthSlider.value = SOFT_BODY_PUSH_STRENGTH;
    baseNodeCostSlider.value = BASE_NODE_EXISTENCE_COST;
    emitterNodeCostSlider.value = EMITTER_NODE_ENERGY_COST;
    neuronNodeCostSlider.value = NEURON_NODE_ENERGY_COST;
    eaterNodeCostSlider.value = EATER_NODE_ENERGY_COST;
    predatorNodeCostSlider.value = PREDATOR_NODE_ENERGY_COST;
    photosyntheticNodeCostSlider.value = PHOTOSYNTHETIC_NODE_ENERGY_COST;
    photosynthesisEfficiencySlider.value = PHOTOSYNTHESIS_EFFICIENCY;
    reproductionCooldownSlider.value = REPRODUCTION_COOLDOWN_TICKS;
    bodyRepulsionStrengthSlider.value = BODY_REPULSION_STRENGTH;
    bodyRepulsionRadiusFactorSlider.value = BODY_REPULSION_RADIUS_FACTOR;
    maxTimestepSlider.value = MAX_DELTA_TIME_MS;
    globalMutationRateSlider.value = GLOBAL_MUTATION_RATE_MODIFIER;
    fluidGridSizeSlider.value = FLUID_GRID_SIZE_CONTROL;
    fluidDiffusionSlider.value = FLUID_DIFFUSION;
    fluidViscositySlider.value = FLUID_VISCOSITY;
    fluidFadeSlider.value = FLUID_FADE_RATE;
    worldWrapToggle.checked = IS_WORLD_WRAPPING;
    particlesPerSecondSlider.value = PARTICLES_PER_SECOND;
    particleFluidInfluenceSlider.value = PARTICLE_FLUID_INFLUENCE;
    particleLifeDecaySlider.value = PARTICLE_BASE_LIFE_DECAY;
    infiniteParticleLifeToggle.checked = IS_PARTICLE_LIFE_INFINITE;
    emitterStrengthSlider.value = EMITTER_STRENGTH;
    emitterEditModeToggle.checked = IS_EMITTER_EDIT_MODE; // Assuming IS_EMITTER_EDIT_MODE is part of config or handled

    // Initialize all slider displays to match loaded JS variables
    initializeAllSliderDisplays(); 

    initFluidSimulation(); // This needs to be called so fluidField.scaleX/Y are set
    initNutrientMap(); // Initialize nutrient map after fluid sim is ready
    initLightMap(); // Initialize light map
    initViscosityMap(); // Initialize viscosity map
    initParticles();
    // Initialize nutrient map after fluid simulation setup to ensure FLUID_GRID_SIZE_CONTROL is correctly applied
    if (config.nutrientFieldData && config.nutrientMapSize) {
        if (config.nutrientMapSize === Math.round(FLUID_GRID_SIZE_CONTROL)) {
            nutrientField = new Float32Array(config.nutrientFieldData);
            console.log("Nutrient map loaded from config.");
        } else {
            console.warn("Nutrient map size in config (" + config.nutrientMapSize + ") does not match current grid size (" + Math.round(FLUID_GRID_SIZE_CONTROL) + "). Re-initializing nutrient map.");
            initNutrientMap(); // Re-initialize if sizes don't match
        }
    } else {
        initNutrientMap(); // Initialize if not in config
    }
    // Initialize light map after fluid simulation setup and nutrient map handling
    if (config.lightFieldData && config.lightMapSize) {
        if (config.lightMapSize === Math.round(FLUID_GRID_SIZE_CONTROL)) {
            lightField = new Float32Array(config.lightFieldData);
            console.log("Light map loaded from config.");
        } else {
            console.warn("Light map size in config (" + config.lightMapSize + ") does not match current grid size (" + Math.round(FLUID_GRID_SIZE_CONTROL) + "). Re-initializing light map.");
            initLightMap();
        }
    } else {
        initLightMap(); // Initialize if not in config
    }
    // Initialize viscosity map after other maps
    if (config.viscosityFieldData && config.viscosityMapSize) {
        if (config.viscosityMapSize === Math.round(FLUID_GRID_SIZE_CONTROL)) {
            viscosityField = new Float32Array(config.viscosityFieldData);
            console.log("Viscosity map loaded from config.");
        } else {
            console.warn("Viscosity map size in config (" + config.viscosityMapSize + ") does not match current grid size (" + Math.round(FLUID_GRID_SIZE_CONTROL) + "). Re-initializing viscosity map.");
            initViscosityMap();
        }
    } else {
        initViscosityMap(); // Initialize if not in config
    }
    initializePopulation(); // Population depends on nutrient map for potential future cost calculations.

    console.log("Applied imported config. Reset population if needed for full effect on creatures.");
} 