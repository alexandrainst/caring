use core::slice;
use std::{ffi::{CStr, c_char}, panic, sync::Mutex};

static ENGINE : Mutex<Option<AdderEngine>> = Mutex::new(None);

use wecare::*;
/// # Safety
/// Strings must null terminated and the array have length `len`
///
/// # Error Codes
/// 1. my_addr malformed
/// 2. others malformed
/// 3. something went wrong in setup_engine, bad address format?
///
#[no_mangle]
pub unsafe extern "C" fn care_setup(my_addr: *const c_char, others: *const *const c_char, len: usize) -> i32 {
    let my_addr = unsafe { CStr::from_ptr(my_addr) };
    let Ok(my_addr) = my_addr.to_str() else {
        return 1;
    };

    let Ok(others) = (unsafe {
        let others = slice::from_raw_parts(others, len);
        let others: Result<Vec<_>, _> = others
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr))
            .map(|cstr| cstr.to_str())
            .collect();
        others
    }) else {
        return 2;
    };
    let Ok(engine) = setup_engine(my_addr, &others) else {
        return 3;
    };
    *ENGINE.lock().unwrap() = Some(engine);
    0
}

/// Run a sum procedure in which each party supplies a double floating point
///
/// Returns `NaN` on unknown panics.
///
#[no_mangle]
pub extern "C" fn care_sum(a: f64) -> f64 {

    let result = panic::catch_unwind(|| {
        let engine = &mut ENGINE.lock().unwrap();
        let engine = engine.as_mut().unwrap();
        mpc_sum(engine, &[a])
    });
    let Ok(result) = result else {
        return f64::NAN;
    };
    let Some(result) = result else {
        return -1.0;
    };
    result[0]
}

/// Compute many sums (vectorized)
///
/// # Safety
/// The `buf` pointer must have length `len`.
///
/// Returns `-1` on unknown panics.
///
#[no_mangle]
pub unsafe extern "C" fn care_sum_many(buf: *const f64, des: *mut f64, len: usize) -> i32 {
    let input: &[_] = unsafe { slice::from_raw_parts(buf, len) };
    let result = panic::catch_unwind(|| {
        let engine = &mut ENGINE.lock().unwrap();
        let engine = engine.as_mut().unwrap();
        mpc_sum(engine, input)
    });
    let Ok(res) = result else {
        return -1;
    };
    let Some(res) = res else {
        return 1;
    };
    let res = res.as_slice().as_ptr();
    unsafe { std::ptr::copy_nonoverlapping(res, des, len) };
    0
}

/// Takedown the MPC engine, freeing the memory and dropping the connections
#[no_mangle]
pub extern "C" fn care_takedown() {
    ENGINE.lock().unwrap().as_mut().take();
}
