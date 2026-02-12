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

- `GET /api/status`
- `GET /api/scenarios`
- `GET /api/snapshot?mode=lite|render|full`
- `POST /api/control/pause`
- `POST /api/control/resume`
- `POST /api/control/setScenario` body: `{ "name": "micro_repro_sustain", "seed": 23 }`

## WebSocket

- `WS /ws?mode=render&hz=10`
  - sends messages:
    - `{ kind: "status", data: <status> }`
    - `{ kind: "snapshot", data: <snapshot> }`

## Notes

- Server runs the real engine via `node-harness/realWorld.mjs` + shared `stepWorld`.
- Current scope is **single-world**; next is multi-world/shards + auth/rate limiting.
