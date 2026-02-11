import config from '../config.js';
import { softBodyPopulation, particles, fluidField } from '../simulation.js';
import { buildTelemetrySnapshot } from '../engine/snapshot.mjs';

let tickCounter = 0;
let pendingForcedSteps = 0;
let lastCaptureTick = -1;
const SNAPSHOT_INTERVAL_TICKS = 30;

function buildSnapshot() {
  return buildTelemetrySnapshot({
    tick: tickCounter,
    scenario: config.DEBUG_SCENARIO || 'baseline',
    seed: config.DEBUG_SEED ?? null,
    fluidField,
    softBodyPopulation,
    particles,
    selectedBody: config.selectedInspectBody
  });
}

export function initDebugRuntime() {
  if (!window.SimDebug) {
    window.SimDebug = {
      timeline: [],
      requestSteps(n = 1) {
        pendingForcedSteps += Math.max(1, Math.floor(n));
      },
      captureNow() {
        const snap = buildSnapshot();
        this.timeline.push(snap);
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
}
