use core::slice;
use std::{ffi::{c_char, CStr}, fs::File, panic, sync::Mutex};

use wecare::SpdzEngine;

static ENGINE : Mutex<Option<SpdzEngine>> = Mutex::new(None);


/// Setup a MPC engine with address `my_addr` connected to the parties with addresses in the
/// array `others`.
///
/// This returns an integer returning zero on no errors otherwise.
///
/// # Error Codes
/// 1. preproc malformed
/// 2. could not open preproc file
/// 3. my_addr malformed
/// 4. others malformed
/// 5. something went wrong in setup_engine, bad address format?
///
/// # Safety
/// Strings must null terminated and the array have length `len`
#[no_mangle]
pub unsafe extern "C" fn care_setup(preproc: *const c_char, my_addr: *const c_char, others: *const *const c_char, len: usize) -> i32 {
    let file = unsafe { CStr::from_ptr(preproc) };
    let Ok(file) = file.to_str() else {
        return 1;
    };
    let Ok(mut file) = File::open(file) else {
        return 2;
    };

    let my_addr = unsafe { CStr::from_ptr(my_addr) };
    let Ok(my_addr) = my_addr.to_str() else {
        return 3;
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
        return 4;
    };
    let Ok(engine) = SpdzEngine::spdz(my_addr, &others, &mut file) else {
        return 5;
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
        engine.mpc_sum(&[a])
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
        engine.mpc_sum(input)
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
