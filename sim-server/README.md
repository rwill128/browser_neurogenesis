# sim-server

Authoritative simulation server + browser client.

## Install

```bash
cd sim-server
npm install
```

## Run

```bash
node server.mjs --port 8787 --scenario micro_repro_sustain --seed 23
```

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
- `GET /api/worlds/:id/snapshot?mode=lite|render|full`
- `POST /api/worlds/:id/control/pause`
- `POST /api/worlds/:id/control/resume`
- `POST /api/worlds/:id/control/setScenario` body `{ "name": "micro_predation", "seed": 23 }` (resets that world)

### Legacy aliases (operate on default world `w0`)

- `GET /api/status`
- `GET /api/snapshot?mode=lite|render|full`
- `POST /api/control/pause`
- `POST /api/control/resume`
- `POST /api/control/setScenario`

## WebSocket

- `WS /ws?world=w0&mode=render&hz=10`
  - sends messages:
    - `{ kind: "status", data: <worldStatus> }`
    - `{ kind: "snapshot", data: <worldSnapshot> }`

## Notes

- Server runs the real engine via `node-harness/realWorld.mjs` + shared `stepWorld`.
- Worlds run in **worker threads** to isolate the module-level config singleton (`js/config.js`).
- Next steps for remote hosting: auth token + rate limiting + persistence (periodic checkpoints to disk).
