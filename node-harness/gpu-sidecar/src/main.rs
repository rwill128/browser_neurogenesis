use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Request {
    Smoke { n: u32 },
}

#[derive(Debug, Serialize)]
struct SmokeResponse {
    ok: bool,
    backend: &'static str,
    n: u32,
    elapsed_ms: f64,
    sample: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn main() {
    if let Err(err) = run() {
        let out = serde_json::json!({"ok": false, "error": format!("{err:#}")});
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{\"ok\":false}".into()));
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let req: Request = if input.trim().is_empty() {
        Request::Smoke { n: 1024 }
    } else {
        serde_json::from_str(&input).context("invalid JSON request")?
    };

    match req {
        Request::Smoke { n } => {
            let resp = pollster::block_on(run_smoke(n.max(64)))?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }
    }

    Ok(())
}

async fn run_smoke(n: u32) -> Result<SmokeResponse> {
    let t0 = std::time::Instant::now();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .context("no GPU adapter")?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .context("request_device failed")?;

    let len = n as usize;
    let bytes = (len * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let src: Vec<f32> = (0..len).map(|i| i as f32).collect();

    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&storage, 0, bytemuck::cast_slice(&src));
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&Params { n, _pad0: 0, _pad1: 0, _pad2: 0 }));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("smoke"),
        source: wgpu::ShaderSource::Wgsl(
            "
struct Params {
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < p.n) {
    data[i] = data[i] + 1.0;
  }
}
"
            .into(),
        ),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("smoke-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("smoke-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: storage.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = n.div_ceil(64);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().context("map_async channel closed")??;

    let mapped = slice.get_mapped_range();
    let out: &[f32] = bytemuck::cast_slice(&mapped);
    let sample = [out[0], out[1], out[10.min(len - 1)], out[len - 1]];
    drop(mapped);
    readback.unmap();

    Ok(SmokeResponse {
        ok: true,
        backend: "metal/wgpu",
        n,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        sample,
    })
}
