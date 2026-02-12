import config from './config.js';
import './classes/constants.js';
import './classes/MassPoint.js';
import './classes/Spring.js';
import './classes/Brain.js';
import './classes/SoftBody.js';
import './classes/FluidField.js';
import './classes/Particle.js';
import {
    initializeSpatialGrid,
    initializePopulation,
    updatePhysics,
    initFluidSimulation,
    initParticles,
    initNutrientMap,
    initLightMap,
    initViscosityMap,
    draw
} from './simulation.js';
import {
    canvas, webgpuCanvas, worldWidthInput, worldHeightInput,
    updateInstabilityIndicator, initializeAllSliderDisplays, updatePopulationCount, updateStatsPanel, updateInfoPanel,
    clampViewOffsets
} from './ui.js';
import { applyScenarioFromUrl } from './debug/scenarios.js';
import { initDebugRuntime, shouldForceStep, consumeForcedStep, onSimulationTick } from './debug/telemetry.js';
import { getScenarioDef, scenarioDefs } from './engine/scenarioDefs.mjs';
import { buildRandomWorldLaunchConfig } from './engine/launcherConfig.mjs';

let lastTime = 0;
let deltaTime = 0;
let animationFrameId = null;
let statsUpdateCounter = 0;
const STATS_UPDATE_INTERVAL = 60;
let frameTimeDisplayElement = null;
let frameTimeAccumulator = 0;
let frameCountForAvg = 0;
let perfLogLastTs = 0;
const PERF_LOG_INTERVAL_MS = 5000;

/**
 * Copy browser-config overrides onto runtime config.
 */
function applyBrowserOverrides(overrides = {}) {
    for (const [key, value] of Object.entries(overrides)) {
        config[key] = value;
    }
}

/**
 * Apply one named scenario definition directly to browser runtime config.
 */
function applyNamedScenario(name) {
    const scenario = getScenarioDef(name);
    applyBrowserOverrides(scenario.browserConfig || {});
    config.DEBUG_SCENARIO = scenario.name;
    return {
        name: scenario.name,
        description: scenario.description || '',
        seed: null
    };
}

/**
 * Keep launcher choices in URL so refresh preserves startup mode.
 */
function writeLaunchSelectionToUrl(selection) {
    const params = new URLSearchParams(window.location.search);
    params.delete('scenario');
    params.delete('mini');
    params.delete('seed');
    params.delete('mode');

    if (selection.mode === 'scenario') {
        params.set('scenario', selection.name);
    } else if (selection.mode === 'default') {
        params.set('mode', 'default');
    } else if (selection.mode === 'random') {
        params.set('mode', 'random');
    }

    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}`;
    window.history.replaceState({}, '', nextUrl);
}

function hideLauncherOverlay() {
    const overlay = document.getElementById('scenarioLauncherOverlay');
    if (overlay) overlay.classList.add('hidden');
}

/**
 * Return to the launcher overlay by clearing startup URL mode and reloading.
 */
function returnToScenarioLibrary() {
    const cleanUrl = `${window.location.pathname}${window.location.hash || ''}`;
    window.location.assign(cleanUrl);
}

/**
 * Render clickable cards for all micro scenarios (same family used in node harness).
 */
function renderMicroScenarioCards(container, onSelect) {
    if (!container) return;
    container.innerHTML = '';

    const micros = Object.values(scenarioDefs)
        .filter((s) => String(s?.name || '').startsWith('micro_'))
        .sort((a, b) => String(a.name).localeCompare(String(b.name)));

    for (const scenario of micros) {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'scenario-launcher-card';
        const nodeCfg = scenario.nodeConfig || {};
        const world = nodeCfg.world || {};
        const summary = `node: ${world.width ?? '?'}x${world.height ?? '?'} • creatures ${nodeCfg.creatures ?? '?'} • particles ${nodeCfg.particles ?? '?'}`;

        card.innerHTML = `
            <div class="name">${scenario.name}</div>
            <div class="desc">${scenario.description || 'Micro scenario'}</div>
            <div class="meta">${summary}</div>
        `;

        card.addEventListener('click', () => onSelect({ mode: 'scenario', name: scenario.name }));
        container.appendChild(card);
    }
}

/**
 * Show launch picker and resolve once the user chooses an entry.
 */
function presentScenarioLauncher() {
    return new Promise((resolve) => {
        const overlay = document.getElementById('scenarioLauncherOverlay');
        const cards = document.getElementById('scenarioLauncherCards');
        const defaultBtn = document.getElementById('launcherDefaultWorldButton');
        const randomBtn = document.getElementById('launcherRandomWorldButton');

        if (!overlay || !cards || !defaultBtn || !randomBtn) {
            resolve({ mode: 'default' });
            return;
        }

        overlay.classList.remove('hidden');

        const choose = (selection) => {
            hideLauncherOverlay();
            writeLaunchSelectionToUrl(selection);
            resolve(selection);
        };

        renderMicroScenarioCards(cards, choose);

        defaultBtn.onclick = () => choose({ mode: 'default' });
        randomBtn.onclick = () => choose({ mode: 'random' });
    });
}

/**
 * Resolve startup mode from URL or interactive launcher.
 */
async function resolveLaunchSelection() {
    const params = new URLSearchParams(window.location.search);
    const explicitScenario = params.get('scenario') || params.get('mini');
    if (explicitScenario) {
        hideLauncherOverlay();
        return { mode: 'url-scenario' };
    }

    const mode = params.get('mode');
    if (mode === 'default') {
        hideLauncherOverlay();
        return { mode: 'default' };
    }
    if (mode === 'random') {
        hideLauncherOverlay();
        return { mode: 'random' };
    }

    return presentScenarioLauncher();
}

/**
 * Apply selected launch mode and return display metadata.
 */
function applyLaunchSelection(selection) {
    if (selection.mode === 'url-scenario') {
        return applyScenarioFromUrl();
    }

    if (selection.mode === 'random') {
        const preset = buildRandomWorldLaunchConfig();
        applyBrowserOverrides(preset.browserConfig);
        config.DEBUG_SCENARIO = preset.name;
        return {
            name: preset.name,
            description: preset.description,
            seed: null,
            randomPreset: preset.browserConfig
        };
    }

    if (selection.mode === 'scenario') {
        return applyNamedScenario(selection.name);
    }

    return applyNamedScenario('baseline');
}

async function main() {
    const launchSelection = await resolveLaunchSelection();
    const scenarioInfo = applyLaunchSelection(launchSelection);

    initDebugRuntime();
    console.log(`[SCENARIO] Loaded: ${scenarioInfo.name} - ${scenarioInfo.description}${scenarioInfo.seed !== null && scenarioInfo.seed !== undefined ? ` | seed=${scenarioInfo.seed}` : ''}`);
    if (scenarioInfo.randomPreset) {
        console.log(`[SCENARIO] Random preset: ${JSON.stringify(scenarioInfo.randomPreset)}`);
    }

    const libraryButton = document.getElementById('openScenarioLibraryButton');
    if (libraryButton) {
        libraryButton.onclick = () => returnToScenarioLibrary();
    }

    worldWidthInput.value = String(config.WORLD_WIDTH || parseInt(worldWidthInput.value) || 8000);
    worldHeightInput.value = String(config.WORLD_HEIGHT || parseInt(worldHeightInput.value) || 6000);
    config.WORLD_WIDTH = parseInt(worldWidthInput.value) || 8000;
    config.WORLD_HEIGHT = parseInt(worldHeightInput.value) || 6000;

    // Initial resize
    resizeCanvas();

    config.GRID_COLS = Math.ceil(config.WORLD_WIDTH / config.GRID_CELL_SIZE);
    config.GRID_ROWS = Math.ceil(config.WORLD_HEIGHT / config.GRID_CELL_SIZE);

    config.INITIAL_POPULATION_SIZE = config.CREATURE_POPULATION_FLOOR;

    initializeSpatialGrid();
    initializeAllSliderDisplays();
    await initFluidSimulation(config.USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initNutrientMap();
    initLightMap();
    initViscosityMap();
    initParticles();
    initializePopulation();
    updateInstabilityIndicator();
    lastTime = performance.now();
    frameTimeDisplayElement = document.getElementById('frameTimeDisplay');
    animationFrameId = requestAnimationFrame(gameLoop);
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';

    webgpuCanvas.width = window.innerWidth * dpr;
    webgpuCanvas.height = window.innerHeight * dpr;
    webgpuCanvas.style.width = window.innerWidth + 'px';
    webgpuCanvas.style.height = window.innerHeight + 'px';

    clampViewOffsets();

    // If using WebGPU, reconfigure the context with new size.
    if (config.USE_GPU_FLUID_SIMULATION && fluidField && fluidField.context) {
        fluidField.context.configure({
            device: fluidField.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            size: { width: webgpuCanvas.width, height: webgpuCanvas.height }
        });
    }
}

window.addEventListener('resize', resizeCanvas);

function maybeLogPerformance(nowTs, currentFrameDuration) {
    if ((nowTs - perfLogLastTs) < PERF_LOG_INTERVAL_MS) return;

    const approxFps = currentFrameDuration > 0 ? (1000 / currentFrameDuration) : 0;
    const isGpu = (typeof fluidField !== 'undefined' && fluidField && fluidField.gpuEnabled);
    const mode = isGpu ? 'GPU' : 'CPU';

    let gpuPerfText = '';
    if (isGpu && fluidField.perfStats) {
        gpuPerfText = ` | gpuStepLast=${fluidField.perfStats.lastStepMs.toFixed(2)}ms gpuDrawLast=${fluidField.perfStats.lastDrawMs.toFixed(2)}ms`;
    }

    const pop = (typeof softBodyPopulation !== 'undefined' && softBodyPopulation) ? softBodyPopulation.length : -1;
    const particleCount = (typeof particles !== 'undefined' && particles) ? particles.length : -1;
    console.log(`[PERF] mode=${mode} frame=${currentFrameDuration.toFixed(2)}ms fps~${approxFps.toFixed(1)} pop=${pop} particles=${particleCount}${gpuPerfText}`);
    perfLogLastTs = nowTs;
}

function gameLoop(timestamp) {
    const loopStartTime = performance.now();

    deltaTime = (timestamp - lastTime) / 1000;
    if (isNaN(deltaTime) || deltaTime <= 0) deltaTime = 1 / 60;
    lastTime = timestamp;

    const currentMaxDeltaTime = config.MAX_DELTA_TIME_MS / 1000.0;
    const effectiveDeltaTime = Math.min(deltaTime, currentMaxDeltaTime);

    const forcedStep = shouldForceStep();
    if (!config.IS_SIMULATION_PAUSED || forcedStep) {
        if (forcedStep) consumeForcedStep();
        config.totalSimulationTime += effectiveDeltaTime;

        if (config.nutrientCyclePeriodSeconds > 0) {
            config.globalNutrientMultiplier = config.nutrientCycleBaseAmplitude + config.nutrientCycleWaveAmplitude * Math.sin((config.totalSimulationTime * 2 * Math.PI) / config.nutrientCyclePeriodSeconds);
            config.globalNutrientMultiplier = Math.max(0.01, config.globalNutrientMultiplier);
        } else {
            config.globalNutrientMultiplier = config.nutrientCycleBaseAmplitude + config.nutrientCycleWaveAmplitude;
        }
        currentNutrientMultiplierDisplay.textContent = config.globalNutrientMultiplier.toFixed(2);

        if (config.lightCyclePeriodSeconds > 0) {
            config.globalLightMultiplier = (Math.sin((config.totalSimulationTime * 2 * Math.PI) / config.lightCyclePeriodSeconds) + 1) / 2;
        } else {
            config.globalLightMultiplier = 0.5;
        }
        currentLightMultiplierDisplay.textContent = config.globalLightMultiplier.toFixed(2);

        updatePhysics(effectiveDeltaTime);
        onSimulationTick();
    }

    if (!config.IS_HEADLESS_MODE) {
        draw();
    }

    statsUpdateCounter++;
    if (statsUpdateCounter >= STATS_UPDATE_INTERVAL) {
        if (statsPanel && statsPanel.classList.contains('open')) {
            updateStatsPanel();
        }
        if (frameTimeDisplayElement && frameCountForAvg > 0) {
            const avgFrameTime = frameTimeAccumulator / frameCountForAvg;
            frameTimeDisplayElement.textContent = `Frame Time: ${avgFrameTime.toFixed(2)} ms`;
            frameTimeAccumulator = 0;
            frameCountForAvg = 0;
        }
        statsUpdateCounter = 0;
    }

    const loopEndTime = performance.now();
    const currentFrameDuration = loopEndTime - loopStartTime;
    frameTimeAccumulator += currentFrameDuration;
    frameCountForAvg++;

    maybeLogPerformance(loopEndTime, currentFrameDuration);

    animationFrameId = requestAnimationFrame(gameLoop);
}

document.addEventListener('DOMContentLoaded', main);
