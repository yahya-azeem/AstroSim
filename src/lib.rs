pub mod app;
pub mod physics;
pub mod render;
#[cfg(not(target_arch = "wasm32"))]
pub mod ephemeris;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn __assert_fail(
    _assertion: *const std::os::raw::c_char,
    _file: *const std::os::raw::c_char,
    _line: std::os::raw::c_uint,
    _function: *const std::os::raw::c_char,
) -> ! {
    panic!("C++ assertion failed");
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn malloc(_size: usize) -> *mut u8 {
    std::ptr::null_mut()
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn free(_ptr: *mut u8) {}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn realloc(_ptr: *mut u8, _size: usize) -> *mut u8 {
    std::ptr::null_mut()
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn __syscall_fcntl64(_fd: i32, _cmd: i32, _arg: i32) -> i32 {
    -1
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn __syscall_ioctl(_fd: i32, _req: i32, _arg: i32) -> i32 {
    -1
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn __syscall_openat(_dirfd: i32, _pathname: *const u8, _flags: i32, _mode: i32) -> i32 {
    -1
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn fd_write(_fd: i32, _iovs: i32, _iovs_len: i32, _nwritten: i32) -> i32 {
    0
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn fd_read(_fd: i32, _iovs: i32, _iovs_len: i32, _nread: i32) -> i32 {
    0
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn fd_seek(_fd: i32, _offset: i64, _whence: i32, _newoffset: i32) -> i32 {
    0
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn fd_close(_fd: i32) -> i32 {
    0
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start_simulation() -> Result<(), String> {
    // Redirect panic reports and logging outputs to browser console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init_with_level(log::Level::Info);

    log::info!("WebAssembly WebGPU/WebGL2 simulation initializing...");

    use winit::event_loop::EventLoop;
    use winit::platform::web::EventLoopExtWebSys;

    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    let app = app::AstroSimApp::new();
    event_loop.spawn_app(app);

    Ok(())
}
