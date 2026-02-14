use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Request {
    Smoke { n: u32 },
    SmokeSweep { sizes: Vec<u32> },
    FluidInit {
        width: u32,
        height: u32,
        #[serde(default = "default_dye_radius")]
        dye_radius: f32,
        #[serde(default = "default_impulse")]
        impulse: f32,
    },
    FluidStep {
        width: u32,
        height: u32,
        #[serde(default = "default_steps")]
        steps: u32,
        #[serde(default = "default_dt")]
        dt: f32,
        #[serde(default = "default_fade")]
        fade: f32,
        #[serde(default = "default_jacobi")]
        jacobi_iters: u32,
        #[serde(default = "default_dye_radius")]
        dye_radius: f32,
        #[serde(default = "default_impulse")]
        impulse: f32,
    },
}

fn default_steps() -> u32 { 1 }
fn default_dt() -> f32 { 0.1 }
fn default_fade() -> f32 { 0.995 }
fn default_jacobi() -> u32 { 30 }
fn default_dye_radius() -> f32 { 0.15 }
fn default_impulse() -> f32 { 25.0 }

#[derive(Debug, Serialize)]
struct SmokeResponse {
    ok: bool,
    backend: &'static str,
    n: u32,
    elapsed_ms: f64,
    sample: [f32; 4],
    mismatch_count: u32,
    max_abs_error: f32,
}

#[derive(Debug, Serialize)]
struct SmokeSweepResponse {
    ok: bool,
    backend: &'static str,
    runs: Vec<SmokeResponse>,
}

#[derive(Debug, Serialize)]
struct FluidInitResponse {
    ok: bool,
    backend: &'static str,
    width: u32,
    height: u32,
    initialized_cells: u32,
    elapsed_ms: f64,
}

#[derive(Debug, Serialize)]
struct FluidStepResponse {
    ok: bool,
    backend: &'static str,
    width: u32,
    height: u32,
    steps: u32,
    elapsed_ms: f64,
    sps: f64,
    avg_speed: f32,
    max_speed: f32,
    avg_divergence: f32,
    max_divergence: f32,
    dye_footprint: f32,
    dye_total: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    jacobi_iters: u32,
    _pad0: u32,
    dt: f32,
    fade: f32,
    dye_radius: f32,
    impulse: f32,
}

fn main() {
    if let Err(err) = run() {
        let out = serde_json::json!({"ok": false, "error": format!("{err:#}")});
        println!(
            "{}",
            serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{\"ok\":false}".into())
        );
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
        Request::SmokeSweep { sizes } => {
            let fallback = vec![1024, 4096, 16384, 65536];
            let mut runs = Vec::new();
            for n in if sizes.is_empty() { &fallback } else { &sizes } {
                runs.push(pollster::block_on(run_smoke((*n).max(64)))?);
            }
            let ok = runs.iter().all(|r| r.ok);
            let resp = SmokeSweepResponse {
                ok,
                backend: "metal/wgpu",
                runs,
            };
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }
        Request::FluidInit {
            width,
            height,
            dye_radius,
            impulse,
        } => {
            let resp = pollster::block_on(run_fluid_init(width.max(16), height.max(16), dye_radius, impulse))?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }
        Request::FluidStep {
            width,
            height,
            steps,
            dt,
            fade,
            jacobi_iters,
            dye_radius,
            impulse,
        } => {
            let resp = pollster::block_on(run_fluid_step(
                width.max(16),
                height.max(16),
                steps.max(1),
                dt.max(1e-4),
                fade.clamp(0.8, 1.0),
                jacobi_iters.clamp(5, 120),
                dye_radius,
                impulse,
            ))?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }
    }

    Ok(())
}

async fn create_device() -> Result<(wgpu::Device, wgpu::Queue)> {
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
    Ok((device, queue))
}

async fn run_fluid_init(width: u32, height: u32, dye_radius: f32, impulse: f32) -> Result<FluidInitResponse> {
    let t0 = std::time::Instant::now();
    let (device, queue) = create_device().await?;
    let cells = (width as usize) * (height as usize);

    let params = Params {
        width,
        height,
        jacobi_iters: 0,
        _pad0: 0,
        dt: default_dt(),
        fade: default_fade(),
        dye_radius,
        impulse,
    };

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fluid-params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let vel_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vel"),
        size: (cells * std::mem::size_of::<[f32; 2]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let dye_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dye"),
        size: (cells * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fluid-init"),
        source: wgpu::ShaderSource::Wgsl(FLUID_INIT_WGSL.into()),
    });
    let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fluid-init-pipeline"),
        layout: None,
        module: &init_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let init_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fluid-init-bg"),
        layout: &init_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dye_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&init_pipeline);
        pass.set_bind_group(0, &init_bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(FluidInitResponse {
        ok: true,
        backend: "metal/wgpu",
        width,
        height,
        initialized_cells: cells as u32,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

async fn run_fluid_step(
    width: u32,
    height: u32,
    steps: u32,
    dt: f32,
    fade: f32,
    jacobi_iters: u32,
    dye_radius: f32,
    impulse: f32,
) -> Result<FluidStepResponse> {
    let t0 = std::time::Instant::now();
    let (device, queue) = create_device().await?;
    let cells = (width as usize) * (height as usize);

    let params = Params {
        width,
        height,
        jacobi_iters,
        _pad0: 0,
        dt,
        fade,
        dye_radius,
        impulse,
    };

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fluid-params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let vel_a = mk_storage_vec2(&device, "vel-a", cells);
    let vel_b = mk_storage_vec2(&device, "vel-b", cells);
    let dye_a = mk_storage_f32(&device, "dye-a", cells);
    let dye_b = mk_storage_f32(&device, "dye-b", cells);
    let div = mk_storage_f32(&device, "div", cells);
    let pressure_a = mk_storage_f32(&device, "pressure-a", cells);
    let pressure_b = mk_storage_f32(&device, "pressure-b", cells);

    let vel_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vel-read"),
        size: (cells * std::mem::size_of::<[f32; 2]>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let dye_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dye-read"),
        size: (cells * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fluid-init"),
        source: wgpu::ShaderSource::Wgsl(FLUID_INIT_WGSL.into()),
    });
    let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fluid-init-pipeline"),
        layout: None,
        module: &init_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let advect_vel_pipeline = mk_pipeline(&device, "advect-vel", FLUID_ADVECT_VEL_WGSL);
    let divergence_pipeline = mk_pipeline(&device, "divergence", FLUID_DIVERGENCE_WGSL);
    let jacobi_pipeline = mk_pipeline(&device, "jacobi", FLUID_JACOBI_WGSL);
    let project_pipeline = mk_pipeline(&device, "project", FLUID_PROJECT_WGSL);
    let advect_dye_pipeline = mk_pipeline(&device, "advect-dye", FLUID_ADVECT_DYE_WGSL);
    let fade_pipeline = mk_pipeline(&device, "fade", FLUID_FADE_WGSL);

    // pre-build bind groups so per-step work stays focused on GPU kernels (less CPU descriptor churn)
    let bg_init = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-init"),
        layout: &init_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dye_a.as_entire_binding() },
        ],
    });
    let bg_advect_vel = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-advect-vel"),
        layout: &advect_vel_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: vel_b.as_entire_binding() },
        ],
    });
    let bg_div = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-div"),
        layout: &divergence_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: div.as_entire_binding() },
        ],
    });
    let bg_jacobi_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-jacobi-ab"),
        layout: &jacobi_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pressure_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: div.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pressure_b.as_entire_binding() },
        ],
    });
    let bg_jacobi_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-jacobi-ba"),
        layout: &jacobi_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pressure_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: div.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pressure_a.as_entire_binding() },
        ],
    });
    let bg_project_from_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-project-from-a"),
        layout: &project_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: vel_a.as_entire_binding() },
        ],
    });
    let bg_project_from_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-project-from-b"),
        layout: &project_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pressure_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: vel_a.as_entire_binding() },
        ],
    });
    let bg_advect_dye = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-advect-dye"),
        layout: &advect_dye_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dye_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dye_b.as_entire_binding() },
        ],
    });
    let bg_fade = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg-fade"),
        layout: &fade_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dye_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dye_a.as_entire_binding() },
        ],
    });

    // seed initial velocity + dye
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&init_pipeline);
        pass.set_bind_group(0, &bg_init, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        drop(pass);
        queue.submit(Some(encoder.finish()));
    }

    for _ in 0..steps {
        let mut encoder = device.create_command_encoder(&Default::default());

        // reset pressure buffers before solve so each projection starts from a clean slate
        encoder.clear_buffer(&pressure_a, 0, None);
        encoder.clear_buffer(&pressure_b, 0, None);

        // velocity advection
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&advect_vel_pipeline);
            pass.set_bind_group(0, &bg_advect_vel, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        // divergence
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&divergence_pipeline);
            pass.set_bind_group(0, &bg_div, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        for i in 0..jacobi_iters {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&jacobi_pipeline);
            pass.set_bind_group(0, if i % 2 == 0 { &bg_jacobi_ab } else { &bg_jacobi_ba }, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        // projection
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&project_pipeline);
            pass.set_bind_group(0, if jacobi_iters % 2 == 0 { &bg_project_from_a } else { &bg_project_from_b }, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        // dye advection
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&advect_dye_pipeline);
            pass.set_bind_group(0, &bg_advect_dye, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        // dye fade and re-seed source slightly
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&fade_pipeline);
            pass.set_bind_group(0, &bg_fade, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&vel_a, 0, &vel_read, 0, (cells * std::mem::size_of::<[f32; 2]>()) as u64);
        encoder.copy_buffer_to_buffer(&dye_a, 0, &dye_read, 0, (cells * std::mem::size_of::<f32>()) as u64);
        queue.submit(Some(encoder.finish()));
    }

    let vel_slice = vel_read.slice(..);
    let dye_slice = dye_read.slice(..);
    map_wait(&device, &vel_slice)?;
    map_wait(&device, &dye_slice)?;

    let vel_mapped = vel_slice.get_mapped_range();
    let dye_mapped = dye_slice.get_mapped_range();
    let vel: &[[f32; 2]] = bytemuck::cast_slice(&vel_mapped);
    let dye: &[f32] = bytemuck::cast_slice(&dye_mapped);

    let mut sum_speed = 0.0f32;
    let mut max_speed = 0.0f32;
    for v in vel {
        let s = (v[0] * v[0] + v[1] * v[1]).sqrt();
        sum_speed += s;
        max_speed = max_speed.max(s);
    }

    let w = width as usize;
    let h = height as usize;
    let mut sum_div = 0.0f32;
    let mut max_div = 0.0f32;
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let vl = vel[y * w + xm][0];
            let vr = vel[y * w + xp][0];
            let vb = vel[ym * w + x][1];
            let vt = vel[yp * w + x][1];
            let d = 0.5 * ((vr - vl) + (vt - vb));
            let ad = d.abs();
            sum_div += ad;
            max_div = max_div.max(ad);
        }
    }

    let mut dye_total = 0.0f32;
    let mut nonzero = 0usize;
    for &d in dye {
        dye_total += d;
        if d > 0.01 {
            nonzero += 1;
        }
    }

    drop(vel_mapped);
    drop(dye_mapped);
    vel_read.unmap();
    dye_read.unmap();

    let elapsed = t0.elapsed().as_secs_f64();
    Ok(FluidStepResponse {
        ok: true,
        backend: "metal/wgpu",
        width,
        height,
        steps,
        elapsed_ms: elapsed * 1000.0,
        sps: (steps as f64) / elapsed.max(1e-6),
        avg_speed: sum_speed / (cells as f32),
        max_speed,
        avg_divergence: sum_div / (cells as f32),
        max_divergence: max_div,
        dye_footprint: (nonzero as f32) / (cells as f32),
        dye_total,
    })
}

fn mk_pipeline(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn mk_storage_vec2(device: &wgpu::Device, label: &str, cells: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (cells * std::mem::size_of::<[f32; 2]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn mk_storage_f32(device: &wgpu::Device, label: &str, cells: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (cells * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn map_wait(device: &wgpu::Device, slice: &wgpu::BufferSlice<'_>) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().context("map_async channel closed")??;
    Ok(())
}

async fn run_smoke(n: u32) -> Result<SmokeResponse> {
    let t0 = std::time::Instant::now();

    let (device, queue) = create_device().await?;

    let len = n as usize;
    let bytes = (len * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let src: Vec<f32> = (0..len).map(|i| i as f32).collect();

    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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
    queue.write_buffer(
        &params_buf,
        0,
        bytemuck::bytes_of(&Params {
            width: n,
            height: 1,
            jacobi_iters: 0,
            _pad0: 0,
            dt: 0.0,
            fade: 0.0,
            dye_radius: 0.0,
            impulse: 0.0,
        }),
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("smoke"),
        source: wgpu::ShaderSource::Wgsl(
            "
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < p.width) {
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
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
    map_wait(&device, &slice)?;

    let mapped = slice.get_mapped_range();
    let out: &[f32] = bytemuck::cast_slice(&mapped);
    let sample = [out[0], out[1], out[10.min(len - 1)], out[len - 1]];

    let mut mismatch_count = 0u32;
    let mut max_abs_error = 0.0f32;
    for (i, &v) in out.iter().enumerate() {
        let expected = (i as f32) + 1.0;
        let err = (v - expected).abs();
        if err > 1e-5 {
            mismatch_count += 1;
        }
        max_abs_error = max_abs_error.max(err);
    }

    drop(mapped);
    readback.unmap();

    let ok = mismatch_count == 0 && max_abs_error <= 1e-5;

    Ok(SmokeResponse {
        ok,
        backend: "metal/wgpu",
        n,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        sample,
        mismatch_count,
        max_abs_error,
    })
}

const FLUID_INIT_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dye: array<f32>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / vec2<f32>(f32(p.width), f32(p.height));
  let c = uv - vec2<f32>(0.5, 0.5);
  let r = length(c);
  let id = idx(gid.x, gid.y);
  let swirl = vec2<f32>(-c.y, c.x) * p.impulse * exp(-30.0 * r * r);
  vel[id] = swirl;
  dye[id] = select(0.0, 1.0 - r / max(p.dye_radius, 0.01), r <= p.dye_radius);
}
"#;

const FLUID_ADVECT_VEL_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }
fn clamp_xy(x: i32, y: i32) -> vec2<u32> {
  let cx = u32(clamp(x, 0, i32(p.width) - 1));
  let cy = u32(clamp(y, 0, i32(p.height) - 1));
  return vec2<u32>(cx, cy);
}
fn sample_vel(pos: vec2<f32>) -> vec2<f32> {
  let x = clamp(pos.x, 0.0, f32(p.width) - 1.001);
  let y = clamp(pos.y, 0.0, f32(p.height) - 1.001);
  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  let fx = fract(x);
  let fy = fract(y);
  let a = src[idx(clamp_xy(x0, y0).x, clamp_xy(x0, y0).y)];
  let b = src[idx(clamp_xy(x1, y0).x, clamp_xy(x1, y0).y)];
  let c = src[idx(clamp_xy(x0, y1).x, clamp_xy(x0, y1).y)];
  let d = src[idx(clamp_xy(x1, y1).x, clamp_xy(x1, y1).y)];
  return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let id = idx(gid.x, gid.y);
  let edge = gid.x == 0u || gid.y == 0u || gid.x == (p.width - 1u) || gid.y == (p.height - 1u);
  if (edge) {
    dst[id] = vec2<f32>(0.0, 0.0);
    return;
  }

  let pos = vec2<f32>(f32(gid.x), f32(gid.y));
  let v = src[id];
  let back = pos - p.dt * v;

  // Semi-Lagrangian advection + sustained center forcing so flow does real work over many steps.
  var v_next = sample_vel(back) * 0.999;
  let center = vec2<f32>(f32(p.width) * 0.5, f32(p.height) * 0.5);
  let rel = pos - center;
  let r = length(rel) / max(f32(min(p.width, p.height)), 1.0);
  if (r <= p.dye_radius) {
    let tangential = normalize(vec2<f32>(-rel.y, rel.x) + vec2<f32>(1e-4, 0.0));
    let falloff = 1.0 - r / max(p.dye_radius, 1e-3);
    v_next = v_next + tangential * (p.impulse * p.dt * falloff);
  }

  dst[id] = v_next;
}
"#;

const FLUID_DIVERGENCE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> div: array<f32>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }
fn c(x: i32, maxv: u32) -> u32 { return u32(clamp(x, 0, i32(maxv) - 1)); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let vl = vel[idx(c(x - 1, p.width), c(y, p.height))].x;
  let vr = vel[idx(c(x + 1, p.width), c(y, p.height))].x;
  let vb = vel[idx(c(x, p.width), c(y - 1, p.height))].y;
  let vt = vel[idx(c(x, p.width), c(y + 1, p.height))].y;
  div[idx(gid.x, gid.y)] = 0.5 * ((vr - vl) + (vt - vb));
}
"#;

const FLUID_JACOBI_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> p_in: array<f32>;
@group(0) @binding(2) var<storage, read> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> p_out: array<f32>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }
fn c(x: i32, maxv: u32) -> u32 { return u32(clamp(x, 0, i32(maxv) - 1)); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let pl = p_in[idx(c(x - 1, p.width), c(y, p.height))];
  let pr = p_in[idx(c(x + 1, p.width), c(y, p.height))];
  let pb = p_in[idx(c(x, p.width), c(y - 1, p.height))];
  let pt = p_in[idx(c(x, p.width), c(y + 1, p.height))];
  let d = div[idx(gid.x, gid.y)];
  p_out[idx(gid.x, gid.y)] = (pl + pr + pb + pt - d) * 0.25;
}
"#;

const FLUID_PROJECT_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_vel: array<vec2<f32>>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }
fn c(x: i32, maxv: u32) -> u32 { return u32(clamp(x, 0, i32(maxv) - 1)); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let pl = pressure[idx(c(x - 1, p.width), c(y, p.height))];
  let pr = pressure[idx(c(x + 1, p.width), c(y, p.height))];
  let pb = pressure[idx(c(x, p.width), c(y - 1, p.height))];
  let pt = pressure[idx(c(x, p.width), c(y + 1, p.height))];
  let grad = vec2<f32>(pr - pl, pt - pb) * 0.5;

  let edge = gid.x == 0u || gid.y == 0u || gid.x == (p.width - 1u) || gid.y == (p.height - 1u);
  out_vel[idx(gid.x, gid.y)] = select(vel[idx(gid.x, gid.y)] - grad, vec2<f32>(0.0, 0.0), edge);
}
"#;

const FLUID_ADVECT_DYE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> dye_src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dye_dst: array<f32>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }
fn clamp_xy(x: i32, y: i32) -> vec2<u32> {
  let cx = u32(clamp(x, 0, i32(p.width) - 1));
  let cy = u32(clamp(y, 0, i32(p.height) - 1));
  return vec2<u32>(cx, cy);
}

fn sample_dye(pos: vec2<f32>) -> f32 {
  let x = clamp(pos.x, 0.0, f32(p.width) - 1.001);
  let y = clamp(pos.y, 0.0, f32(p.height) - 1.001);
  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  let fx = fract(x);
  let fy = fract(y);
  let a = dye_src[idx(clamp_xy(x0, y0).x, clamp_xy(x0, y0).y)];
  let b = dye_src[idx(clamp_xy(x1, y0).x, clamp_xy(x1, y0).y)];
  let c = dye_src[idx(clamp_xy(x0, y1).x, clamp_xy(x0, y1).y)];
  let d = dye_src[idx(clamp_xy(x1, y1).x, clamp_xy(x1, y1).y)];
  return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let id = idx(gid.x, gid.y);
  let pos = vec2<f32>(f32(gid.x), f32(gid.y));
  let back = pos - p.dt * vel[id];
  dye_dst[id] = sample_dye(back);
}
"#;

const FLUID_FADE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  jacobi_iters: u32,
  _pad0: u32,
  dt: f32,
  fade: f32,
  dye_radius: f32,
  impulse: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn idx(x: u32, y: u32) -> u32 { return y * p.width + x; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / vec2<f32>(f32(p.width), f32(p.height));
  let c = uv - vec2<f32>(0.5, 0.5);
  let r = length(c);
  let id = idx(gid.x, gid.y);
  let source = select(0.0, 0.02, r <= p.dye_radius * 0.4);
  dst[id] = src[id] * p.fade + source;
}
"#;
