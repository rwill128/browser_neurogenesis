# gpu-sidecar (Rust + wgpu)

Headless GPU compute sidecar for fluid simulation experiments on Apple Silicon (Metal backend via `wgpu`).

## Current status
- ✅ Rust toolchain installed
- ✅ Sidecar project builds/runs
- ✅ GPU smoke command works (`cmd=smoke`) and executes compute on Metal

## Run
```bash
cd node-harness/gpu-sidecar
cargo run --release <<'JSON'
{"cmd":"smoke","n":1024}
JSON
```

## Protocol (stdin JSON -> stdout JSON)
Request:
```json
{"cmd":"smoke","n":1024}
```

Response:
```json
{
  "ok": true,
  "backend": "metal/wgpu",
  "n": 1024,
  "elapsed_ms": 113.1,
  "sample": [1.0,2.0,11.0,1024.0]
}
```

## Next
1. Add persistent process mode (avoid process startup overhead)
2. Add `fluid_init` and `fluid_step` commands
3. Mirror CPU reference kernels (advect/divergence/jacobi/project/fade)
4. Add parity + SPS benchmarks vs CPU full-domain baseline
