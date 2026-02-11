import config from '../config.js';
import { softBodyPopulation, particles, fluidField } from '../simulation.js';

let tickCounter = 0;
let pendingForcedSteps = 0;
let lastCaptureTick = -1;
const SNAPSHOT_INTERVAL_TICKS = 30;

function creatureSummary(body) {
  const center = body.getAveragePosition();
  const bbox = body.getBoundingBox();
  return {
    id: body.id,
    unstable: !!body.isUnstable,
    points: body.massPoints.length,
    energy: Number(body.creatureEnergy?.toFixed?.(2) || 0),
    center: { x: Number(center.x.toFixed(1)), y: Number(center.y.toFixed(1)) },
    size: { w: Number((bbox.maxX - bbox.minX).toFixed(1)), h: Number((bbox.maxY - bbox.minY).toFixed(1)) }
  };
}

function buildSnapshot() {
  const live = softBodyPopulation.filter(b => !b.isUnstable);
  const selected = config.selectedInspectBody ? creatureSummary(config.selectedInspectBody) : null;
  return {
    tick: tickCounter,
    scenario: config.DEBUG_SCENARIO || 'baseline',
    seed: config.DEBUG_SEED ?? null,
    mode: (fluidField && fluidField.gpuEnabled) ? 'GPU' : 'CPU',
    populations: {
      creatures: softBodyPopulation.length,
      liveCreatures: live.length,
      particles: particles.length
    },
    selected,
    sampleCreatures: live.slice(0, 5).map(creatureSummary)
  };
}

function ensureDebugPanel() {
  if (document.getElementById('simDebugPanel')) return;
  const panel = document.createElement('div');
  panel.id = 'simDebugPanel';
  panel.style.cssText = 'position:fixed;right:8px;bottom:8px;z-index:99999;background:rgba(0,0,0,.75);color:#9ef;padding:8px 10px;border-radius:8px;font:12px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace;max-width:420px;max-height:42vh;overflow:auto;white-space:pre-wrap;';
  panel.textContent = 'SimDebug booting...';
  document.body.appendChild(panel);
}

function updatePanel(snapshot) {
  const panel = document.getElementById('simDebugPanel');
  if (!panel) return;
  panel.textContent = JSON.stringify(snapshot, null, 2);
}

export function initDebugRuntime() {
  ensureDebugPanel();
  if (!window.SimDebug) {
    window.SimDebug = {
      timeline: [],
      requestSteps(n = 1) {
        pendingForcedSteps += Math.max(1, Math.floor(n));
      },
      captureNow() {
        const snap = buildSnapshot();
        this.timeline.push(snap);
        updatePanel(snap);
        return snap;
      },
      clearTimeline() {
        this.timeline.length = 0;
      },
      exportTimeline() {
        return {
          scenario: config.DEBUG_SCENARIO || 'baseline',
          seed: config.DEBUG_SEED ?? null,
          capturedAt: new Date().toISOString(),
          samples: this.timeline
        };
      },
      exportTimelineJson() {
        return JSON.stringify(this.exportTimeline(), null, 2);
      },
      downloadTimeline(filename) {
        const defaultName = `timeline-${config.DEBUG_SCENARIO || 'baseline'}-${Date.now()}.json`;
        const blob = new Blob([this.exportTimelineJson()], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || defaultName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    };
  }
}

export function shouldForceStep() {
  return pendingForcedSteps > 0;
}

export function consumeForcedStep() {
  if (pendingForcedSteps > 0) pendingForcedSteps--;
}

export function onSimulationTick() {
  tickCounter++;
  if (tickCounter - lastCaptureTick < SNAPSHOT_INTERVAL_TICKS) return;
  lastCaptureTick = tickCounter;

  const snap = buildSnapshot();
  if (window.SimDebug) window.SimDebug.timeline.push(snap);
  updatePanel(snap);
}
