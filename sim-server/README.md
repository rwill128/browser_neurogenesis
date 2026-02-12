# sim-server

Minimal authoritative simulation server (no external deps).

## Run

```bash
node sim-server/server.mjs --port 8787 --scenario micro_repro_sustain --seed 23
```

Open:
- http://localhost:8787/ (client)

API:
- `GET /api/status`
- `GET /api/scenarios`
- `GET /api/snapshot?mode=lite|render|full`
- `POST /api/control/pause`
- `POST /api/control/resume`
- `POST /api/control/setScenario` body: `{ "name": "micro_repro_sustain", "seed": 23 }`

## Notes

- Server runs the real engine via `node-harness/realWorld.mjs` + shared `stepWorld`.
- Client is intentionally dumb: polling + canvas render.
- This is the first step toward multi-world sharding + websocket delta streaming.
