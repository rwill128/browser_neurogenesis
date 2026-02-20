#![no_std]

use core::panic::PanicInfo;

type I32 = i32;
type U32 = u32;

const SCP_HEAP_BASE: U32 = 65536;

static mut HEAP_PTR: U32 = SCP_HEAP_BASE;

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
pub extern "C" fn scp_heap_base() -> U32 {
    SCP_HEAP_BASE
}

#[no_mangle]
pub extern "C" fn scp_reset_heap() {
    unsafe {
        HEAP_PTR = SCP_HEAP_BASE;
    }
}

#[no_mangle]
pub extern "C" fn scp_alloc(bytes: U32) -> U32 {
    let aligned = (bytes + 7) & !7;
    unsafe {
        let out = HEAP_PTR;
        HEAP_PTR = HEAP_PTR.saturating_add(aligned);
        out
    }
}

#[no_mangle]
pub extern "C" fn scp_project_nodes(
    node_count: U32,
    node_x_ptr: U32,
    node_y_ptr: U32,
    node_vx_ptr: U32,
    node_vy_ptr: U32,
    node_cluster_index_ptr: U32,
    cluster_count: U32,
    cluster_x_ptr: U32,
    cluster_y_ptr: U32,
    cluster_vx_ptr: U32,
    cluster_vy_ptr: U32,
    cluster_omega_ptr: U32,
    cluster_membrane_flag_ptr: U32,
    linear_gain: f32,
    angular_gain: f32,
    membrane_gain_scale: f32,
    stats_ptr: U32,
) -> U32 {
    unsafe {
        let node_x = node_x_ptr as *const f32;
        let node_y = node_y_ptr as *const f32;
        let node_vx = node_vx_ptr as *mut f32;
        let node_vy = node_vy_ptr as *mut f32;
        let node_cluster_index = node_cluster_index_ptr as *const I32;

        let cluster_x = cluster_x_ptr as *const f32;
        let cluster_y = cluster_y_ptr as *const f32;
        let cluster_vx = cluster_vx_ptr as *const f32;
        let cluster_vy = cluster_vy_ptr as *const f32;
        let cluster_omega = cluster_omega_ptr as *const f32;
        let cluster_membrane_flag = cluster_membrane_flag_ptr as *const I32;

        let stats = stats_ptr as *mut f32;

        let mut projected_nodes: U32 = 0;
        let mut sum_delta: f32 = 0.0;

        let mut i: U32 = 0;
        while i < node_count {
            let ci = *node_cluster_index.add(i as usize);
            if ci >= 0 && (ci as U32) < cluster_count {
                let cidx = ci as usize;
                let x = *node_x.add(i as usize);
                let y = *node_y.add(i as usize);
                if x.is_finite() && y.is_finite() {
                    let membrane_scale = if *cluster_membrane_flag.add(cidx) != 0 {
                        membrane_gain_scale
                    } else {
                        1.0
                    };

                    let lg = clamp(linear_gain * membrane_scale, 0.0, 1.0);
                    let ag = clamp(angular_gain * membrane_scale, 0.0, 1.0);
                    if lg > 0.0 || ag > 0.0 {
                        let st_x = finite_or(*cluster_x.add(cidx), 0.0);
                        let st_y = finite_or(*cluster_y.add(cidx), 0.0);
                        let st_vx = finite_or(*cluster_vx.add(cidx), 0.0);
                        let st_vy = finite_or(*cluster_vy.add(cidx), 0.0);
                        let omega = finite_or(*cluster_omega.add(cidx), 0.0);

                        let vx = finite_or(*node_vx.add(i as usize), 0.0);
                        let vy = finite_or(*node_vy.add(i as usize), 0.0);

                        let rx = x - st_x;
                        let ry = y - st_y;
                        let target_rel_vx = -omega * ry;
                        let target_rel_vy = omega * rx;
                        let cur_rel_vx = vx - st_vx;
                        let cur_rel_vy = vy - st_vy;

                        let delta_vx = (st_vx - vx) * lg + (target_rel_vx - cur_rel_vx) * ag;
                        let delta_vy = (st_vy - vy) * lg + (target_rel_vy - cur_rel_vy) * ag;

                        *node_vx.add(i as usize) = vx + delta_vx;
                        *node_vy.add(i as usize) = vy + delta_vy;

                        projected_nodes += 1;
                        sum_delta += sqrt_approx(delta_vx * delta_vx + delta_vy * delta_vy);
                    }
                }
            }
            i += 1;
        }

        *stats.add(0) = sum_delta;
        projected_nodes
    }
}
