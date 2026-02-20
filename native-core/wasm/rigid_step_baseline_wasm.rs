#![no_std]

use core::panic::PanicInfo;

type I32 = i32;
type U32 = u32;

const RSB_HEAP_BASE: U32 = 65536;

static mut HEAP_PTR: U32 = RSB_HEAP_BASE;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[inline]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

#[inline]
fn sqrt_approx(v: f32) -> f32 {
    if !(v > 0.0) {
        return 0.0;
    }
    let mut x = if v >= 1.0 { v } else { 1.0 };
    let mut i = 0;
    while i < 6 {
        x = 0.5 * (x + v / x);
        i += 1;
    }
    x
}

#[no_mangle]
pub extern "C" fn rsb_heap_base() -> U32 {
    RSB_HEAP_BASE
}

#[no_mangle]
pub extern "C" fn rsb_reset_heap() {
    unsafe {
        HEAP_PTR = RSB_HEAP_BASE;
    }
}

#[no_mangle]
pub extern "C" fn rsb_alloc(bytes: U32) -> U32 {
    let aligned = (bytes + 7) & !7;
    unsafe {
        let out = HEAP_PTR;
        HEAP_PTR = HEAP_PTR.saturating_add(aligned);
        out
    }
}

#[no_mangle]
pub extern "C" fn rsb_step_bodies(
    body_count: U32,
    body_vx_ptr: U32,
    body_vy_ptr: U32,
    body_omega_ptr: U32,
    body_x_ptr: U32,
    body_y_ptr: U32,
    body_theta_ptr: U32,
    body_mass_ptr: U32,
    body_inertia_ptr: U32,
    body_edge_momentum_ptr: U32,
    body_center_honey_ptr: U32,
    body_swim_x_ptr: U32,
    body_swim_y_ptr: U32,
    body_swim_torque_ptr: U32,
    body_sample_offsets_ptr: U32,
    sample_rx_ptr: U32,
    sample_ry_ptr: U32,
    sample_fx_ptr: U32,
    sample_fy_ptr: U32,
    sample_honey_ptr: U32,
    dt: f32,
    dt_norm: f32,
    drag_k: f32,
    world_size: f32,
    stats_ptr: U32,
) -> U32 {
    unsafe {
        let body_vx = body_vx_ptr as *mut f32;
        let body_vy = body_vy_ptr as *mut f32;
        let body_omega = body_omega_ptr as *mut f32;
        let body_x = body_x_ptr as *mut f32;
        let body_y = body_y_ptr as *mut f32;
        let body_theta = body_theta_ptr as *mut f32;

        let body_mass = body_mass_ptr as *const f32;
        let body_inertia = body_inertia_ptr as *const f32;
        let body_edge_momentum = body_edge_momentum_ptr as *const f32;
        let body_center_honey = body_center_honey_ptr as *const f32;
        let body_swim_x = body_swim_x_ptr as *const f32;
        let body_swim_y = body_swim_y_ptr as *const f32;
        let body_swim_torque = body_swim_torque_ptr as *const f32;

        let body_sample_offsets = body_sample_offsets_ptr as *const I32;
        let sample_rx = sample_rx_ptr as *const f32;
        let sample_ry = sample_ry_ptr as *const f32;
        let sample_fx = sample_fx_ptr as *const f32;
        let sample_fy = sample_fy_ptr as *const f32;
        let sample_honey = sample_honey_ptr as *const f32;

        let stats = stats_ptr as *mut f32;
        let boundary_edge = clamp(world_size - 1.0, 0.0, 1.0e9);

        let mut carry_sum: f32 = 0.0;
        let mut total_samples: U32 = 0;

        let mut bi: U32 = 0;
        while bi < body_count {
            let idx = bi as usize;

            let mut vx = finite_or(*body_vx.add(idx), 0.0);
            let mut vy = finite_or(*body_vy.add(idx), 0.0);
            let mut omega = finite_or(*body_omega.add(idx), 0.0);
            let mut x = finite_or(*body_x.add(idx), 0.0);
            let mut y = finite_or(*body_y.add(idx), 0.0);
            let mut theta = finite_or(*body_theta.add(idx), 0.0);

            let mass = finite_or(*body_mass.add(idx), 1.0);
            let inertia = finite_or(*body_inertia.add(idx), 1.0);
            let inv_mass = 1.0 / clamp(mass, 0.05, 1.0e9);
            let inv_inertia = 1.0 / clamp(inertia, 0.05, 1.0e9);
            let edge_momentum = clamp(finite_or(*body_edge_momentum.add(idx), 1.0), 0.0, 1.0);

            let start = (*body_sample_offsets.add(idx)).max(0) as U32;
            let end = (*body_sample_offsets.add(idx + 1)).max(start as I32) as U32;
            let sample_count = if end > start { end - start } else { 0 };

            let mut force_x: f32 = 0.0;
            let mut force_y: f32 = 0.0;
            let mut torque: f32 = 0.0;

            let mut si = start;
            while si < end {
                let sidx = si as usize;
                let rx = finite_or(*sample_rx.add(sidx), 0.0);
                let ry = finite_or(*sample_ry.add(sidx), 0.0);
                let fx = finite_or(*sample_fx.add(sidx), 0.0);
                let fy = finite_or(*sample_fy.add(sidx), 0.0);
                let honey = clamp(finite_or(*sample_honey.add(sidx), 1.0), 0.0, 1000.0);

                let local_vx = vx + (-omega * ry);
                let local_vy = vy + (omega * rx);
                let rel_x = fx - local_vx;
                let rel_y = fy - local_vy;

                let fpx = rel_x * drag_k * honey * edge_momentum;
                let fpy = rel_y * drag_k * honey * edge_momentum;

                force_x += fpx;
                force_y += fpy;
                torque += rx * fpy - ry * fpx;

                si += 1;
            }

            if sample_count > 0 {
                let inv_count = 1.0 / (sample_count as f32);
                force_x *= inv_count;
                force_y *= inv_count;
                torque *= inv_count;
                total_samples += sample_count;
            }

            let ax = force_x * inv_mass;
            let ay = force_y * inv_mass;
            let alpha = torque * inv_inertia;

            let swim_x = finite_or(*body_swim_x.add(idx), 0.0);
            let swim_y = finite_or(*body_swim_y.add(idx), 0.0);
            let swim_torque = finite_or(*body_swim_torque.add(idx), 0.0);

            vx += ax * dt * 60.0 + swim_x * dt_norm;
            vy += ay * dt * 60.0 + swim_y * dt_norm;
            omega += alpha * dt * 60.0 + swim_torque * dt_norm;

            let center_honey = clamp(finite_or(*body_center_honey.add(idx), 1.0), 0.0, 1000.0);
            let damp = if 1.0 - 0.018 * center_honey > 0.72 {
                1.0 - 0.018 * center_honey
            } else {
                0.72
            };
            let vmax = {
                let base = 3.2 / (1.0 + 0.28 * center_honey);
                if base > 0.4 { base } else { 0.4 }
            };

            vx *= damp;
            vy *= damp;
            omega *= if 0.99 - 0.01 * center_honey > 0.72 {
                0.99 - 0.01 * center_honey
            } else {
                0.72
            };

            let vmag = sqrt_approx(vx * vx + vy * vy);
            if vmag > vmax && vmax > 1.0e-8 {
                let inv_mag = 1.0 / if vmag > 1.0e-9 { vmag } else { 1.0e-9 };
                vx = vx * inv_mag * vmax;
                vy = vy * inv_mag * vmax;
            }
            omega = clamp(omega, -0.25, 0.25);

            carry_sum += sqrt_approx(ax * ax + ay * ay);

            x += vx * dt * 22.0;
            y += vy * dt * 28.0;
            theta += omega * dt * 60.0;

            let e = 0.84;
            if x < 0.0 {
                x = 0.0;
                vx = vx.abs() * e;
            }
            if y < 0.0 {
                y = 0.0;
                vy = vy.abs() * e;
            }
            if x > boundary_edge {
                x = boundary_edge;
                vx = -vx.abs() * e;
            }
            if y > boundary_edge {
                y = boundary_edge;
                vy = -vy.abs() * e;
            }

            *body_vx.add(idx) = finite_or(vx, 0.0);
            *body_vy.add(idx) = finite_or(vy, 0.0);
            *body_omega.add(idx) = finite_or(omega, 0.0);
            *body_x.add(idx) = finite_or(x, 0.0);
            *body_y.add(idx) = finite_or(y, 0.0);
            *body_theta.add(idx) = finite_or(theta, 0.0);

            bi += 1;
        }

        *stats.add(0) = carry_sum;
        *stats.add(1) = total_samples as f32;
        *stats.add(2) = body_count as f32;
        body_count
    }
}
