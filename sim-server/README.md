# sim-server

Authoritative simulation server + browser client.

## Install

```bash
cd sim-server
npm install
```

## Run

```bash
node server.mjs --port 8787 \
  --scenario micro_repro_sustain \
  --seed 23 \
  --maxWorlds 8 \
  --checkpointEverySec 60 \
  --checkpointKeep 50 \
  --restoreLatest true
```

`--restoreLatest` is optional (default `false`):
- `false` (default) → fresh world start on boot
- `true` → after creating `w0`, load latest checkpoint for `w0` if one exists

Open:
- http://localhost:8787/

## API

### Scenarios

- `GET /api/scenarios`

### Worlds (multi-world)

- `GET /api/worlds` → list worlds + status
- `POST /api/worlds` body `{ "scenario": "micro_repro_sustain", "seed": 23, "id": "optional" }` → create world
- `DELETE /api/worlds/:id` → delete world (default `w0` protected)
- `GET /api/worlds/:id/status`
- `GET /api/worlds/:id/snapshot?mode=lite|render|renderFull|full`
- `GET /api/worlds/:id/frameTimeline` → short rolling frame buffer metadata + archive bounds
- `GET /api/worlds/:id/frame/:tick` → exact archived frame by tick
- `GET /api/worlds/:id/frames?startTick=0&endTick=1000&stride=1` → archived frame range
- `GET /api/worlds/:id/config`
- `POST /api/worlds/:id/capture/randomCreature` body `{ "creatureId": 123?, "size": 800?, "zoomOutFactor": 1?, "includeFluid": true?, "includeNeighbors": true? }` → SVG portrait
- `POST /api/worlds/:id/capture/creatureClip` body `{ "creatureId": 123?, "durationSec": 5?, "fps": 12?, "size": 800?, "zoomOutFactor": 1?, "includeFluid": true?, "includeNeighbors": true? }` → MP4 clip of next seconds of life
- `GET /api/worlds/:id/captures/:file` → fetch saved `.svg`/`.png`/`.mp4` capture artifacts
- `POST /api/worlds/:id/control/pause`
- `POST /api/worlds/:id/control/resume`
- `POST /api/worlds/:id/control/setScenario` body `{ "name": "micro_predation", "seed": 23 }` (resets that world)
- `POST /api/worlds/:id/control/configOverrides` body `{ "overrides": { "PHOTOSYNTHESIS_EFFICIENCY": 160 } }`

### Persistence (SQLite)

- `POST /api/worlds/:id/checkpoints` body `{ "label": "optional" }` → create checkpoint
- `GET /api/worlds/:id/checkpoints?limit=20` → list checkpoints
- `POST /api/worlds/:id/restore` body `{ "checkpointId": 123 }` → restore checkpoint into the world

### Legacy aliases (operate on default world `w0`)

- `GET /api/status`
- `GET /api/snapshot?mode=lite|render|full`
- `POST /api/capture/randomCreature`
- `POST /api/capture/creatureClip`
- `GET /api/captures/:file`
- `POST /api/control/pause`
- `POST /api/control/resume`
- `POST /api/control/setScenario`

## WebSocket

- `WS /ws?world=w0&mode=renderFull&hz=10`
  - sends messages:
    - `{ kind: "status", data: <worldStatus> }`
    - `{ kind: "snapshot", data: <worldSnapshot> }`

## Notes

- Server runs the real engine via `node-harness/realWorld.mjs` + shared `stepWorld`.
- Worlds run in **worker threads** to isolate the module-level config singleton (`js/config.js`).
- Persistence uses SQLite with runtime fallback:
  - prefers built-in `node:sqlite` when available,
  - falls back to `better-sqlite3` on Node versions without `node:sqlite` (e.g. Node 22).
- Next steps for remote hosting: auth token + rate limiting + metrics endpoints.
