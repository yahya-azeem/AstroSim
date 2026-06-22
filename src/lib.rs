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
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
use std::alloc::{alloc, dealloc, realloc as rust_realloc, Layout};

const ALIGNMENT: usize = 16;
const HEADER_SIZE: usize = 16;

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = match Layout::from_size_align(size + HEADER_SIZE, ALIGNMENT) {
        Ok(l) => l,
        Err(_) => {
            log(&format!("malloc({}) -> Layout Error", size));
            return std::ptr::null_mut();
        }
    };
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        log(&format!("malloc({}) -> Allocation Failed (null)", size));
        return std::ptr::null_mut();
    }
    unsafe {
        *(ptr as *mut usize) = size;
        ptr.add(HEADER_SIZE)
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let real_ptr = ptr.sub(HEADER_SIZE);
        let size = *(real_ptr as *const usize);
        let layout = Layout::from_size_align_unchecked(size + HEADER_SIZE, ALIGNMENT);
        dealloc(real_ptr, layout);
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return unsafe { malloc(new_size) };
    }
    if new_size == 0 {
        unsafe { free(ptr) };
        return std::ptr::null_mut();
    }
    unsafe {
        let real_ptr = ptr.sub(HEADER_SIZE);
        let old_size = *(real_ptr as *const usize);
        let old_layout = Layout::from_size_align_unchecked(old_size + HEADER_SIZE, ALIGNMENT);
        let new_real_ptr = rust_realloc(real_ptr, old_layout, new_size + HEADER_SIZE);
        if new_real_ptr.is_null() {
            log(&format!("realloc({:?}, {}) -> Reallocation Failed (null)", ptr, new_size));
            return std::ptr::null_mut();
        }
        *(new_real_ptr as *mut usize) = new_size;
        new_real_ptr.add(HEADER_SIZE)
    }
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
