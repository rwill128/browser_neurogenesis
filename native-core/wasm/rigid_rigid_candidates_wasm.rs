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

#[no_mangle]
pub extern "C" fn rr_build_pairs(
    body_count: U32,
    offsets_ptr: U32,
    neighbors_ptr: U32,
    seen_ptr: U32,
    pairs_ptr: U32,
    stats_ptr: U32,
) -> U32 {
    unsafe {
        let offsets = offsets_ptr as *const I32;
        let neighbors = neighbors_ptr as *const I32;
        let seen = seen_ptr as *mut I32;
        let pairs = pairs_ptr as *mut I32;
        let stats = stats_ptr as *mut I32;

        let mut i: U32 = 0;
        while i < body_count {
            *seen.add(i as usize) = 0;
            i += 1;
        }

        let mut emit_attempts: I32 = 0;
        let mut duplicates_rejected: I32 = 0;
        let mut pair_count: I32 = 0;
        let mut stamp_token: I32 = 1;

        let mut bi: U32 = 0;
        while bi < body_count {
            let mut stamp = stamp_token;
            stamp_token += 1;
            if stamp <= 0 {
                let mut k: U32 = 0;
                while k < body_count {
                    *seen.add(k as usize) = 0;
                    k += 1;
                }
                stamp_token = 2;
                stamp = 1;
            }

            let start = *offsets.add(bi as usize);
            let end = *offsets.add((bi + 1) as usize);
            let mut idx = start;
            while idx < end {
                emit_attempts += 1;
                let j = *neighbors.add(idx as usize);
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
                idx += 1;
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
