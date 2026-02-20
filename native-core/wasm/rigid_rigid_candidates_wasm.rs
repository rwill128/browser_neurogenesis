#![no_std]

use core::panic::PanicInfo;

type I32 = i32;
type U32 = u32;

const RR_HEAP_BASE: U32 = 65536;

static mut HEAP_PTR: U32 = RR_HEAP_BASE;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub extern "C" fn rr_heap_base() -> U32 {
    RR_HEAP_BASE
}

#[no_mangle]
pub extern "C" fn rr_reset_heap() {
    unsafe {
        HEAP_PTR = RR_HEAP_BASE;
    }
}

#[no_mangle]
pub extern "C" fn rr_alloc(bytes: U32) -> U32 {
    let aligned = (bytes + 7) & !7;
    unsafe {
        let out = HEAP_PTR;
        HEAP_PTR = HEAP_PTR.saturating_add(aligned);
        out
    }
}

unsafe fn clear_seen(seen: *mut I32, body_count: U32) {
    let mut i: U32 = 0;
    while i < body_count {
        *seen.add(i as usize) = 0;
        i += 1;
    }
}

#[no_mangle]
pub extern "C" fn rr_build_pairs_from_cells(
    body_count: U32,
    body_cell_offsets_ptr: U32,
    body_cell_indices_ptr: U32,
    cell_offsets_ptr: U32,
    cell_body_ids_ptr: U32,
    seen_ptr: U32,
    pairs_ptr: U32,
    stats_ptr: U32,
) -> U32 {
    unsafe {
        let body_cell_offsets = body_cell_offsets_ptr as *const I32;
        let body_cell_indices = body_cell_indices_ptr as *const I32;
        let cell_offsets = cell_offsets_ptr as *const I32;
        let cell_body_ids = cell_body_ids_ptr as *const I32;
        let seen = seen_ptr as *mut I32;
        let pairs = pairs_ptr as *mut I32;
        let stats = stats_ptr as *mut I32;

        clear_seen(seen, body_count);

        let mut emit_attempts: I32 = 0;
        let mut duplicates_rejected: I32 = 0;
        let mut pair_count: I32 = 0;
        let mut stamp_token: I32 = 1;

        let mut bi: U32 = 0;
        while bi < body_count {
            let mut stamp = stamp_token;
            stamp_token += 1;
            if stamp <= 0 {
                clear_seen(seen, body_count);
                stamp_token = 2;
                stamp = 1;
            }

            let body_cells_start = *body_cell_offsets.add(bi as usize);
            let body_cells_end = *body_cell_offsets.add((bi + 1) as usize);

            let mut bci = body_cells_start;
            while bci < body_cells_end {
                let cell_index = *body_cell_indices.add(bci as usize);
                if cell_index >= 0 {
                    let cell_start = *cell_offsets.add(cell_index as usize);
                    let cell_end = *cell_offsets.add((cell_index as usize) + 1);

                    let mut ci = cell_start;
                    while ci < cell_end {
                        emit_attempts += 1;
                        let j = *cell_body_ids.add(ci as usize);
                        if j <= bi as I32 || j < 0 || j >= body_count as I32 {
                            duplicates_rejected += 1;
                        } else {
                            let seen_j = seen.add(j as usize);
                            if *seen_j == stamp {
                                duplicates_rejected += 1;
                            } else {
                                *seen_j = stamp;
                            }
                        }
                        ci += 1;
                    }
                }
                bci += 1;
            }

            let mut bj = bi + 1;
            while bj < body_count {
                if *seen.add(bj as usize) == stamp {
                    let base = (pair_count as usize) * 2;
                    *pairs.add(base) = bi as I32;
                    *pairs.add(base + 1) = bj as I32;
                    pair_count += 1;
                }
                bj += 1;
            }

            bi += 1;
        }

        *stats.add(0) = emit_attempts;
        *stats.add(1) = duplicates_rejected;
        *stats.add(2) = pair_count;
        pair_count as U32
    }
}
