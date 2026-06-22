pub mod app;
pub mod physics;
pub mod render;
#[cfg(not(target_arch = "wasm32"))]
pub mod ephemeris;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

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
