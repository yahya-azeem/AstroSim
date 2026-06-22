// AstroSim Library Port Boilerplate for Browser WASM & WebGPU
// Guarded to compile only for the wasm32 target architecture

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    // Redirect panic reports and logging outputs to browser console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("Failed to initialize console log");

    log::info!("WebAssembly WebGPU simulation initializing...");

    // Spawn the async WebGPU and Winit application loop
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run_web_app().await {
            log::error!("Error running WebGPU WASM application: {:?}", e);
        }
    });
}

#[cfg(target_arch = "wasm32")]
async fn run_web_app() -> Result<(), Box<dyn std::error::Error>> {
    // Create the event loop and initialize winit window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("AstroSim Browser Port")
        .build(&event_loop)?;

    // Query Document and Window from web-sys
    let web_window = web_sys::window().ok_or("No global window found")?;
    let document = web_window.document().ok_or("No global document found")?;
    let body = document.body().ok_or("No body element found")?;

    // Create and attach a canvas element for rendering
    let canvas = document.create_element("canvas")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    canvas.set_id("astrosim-canvas");
    canvas.set_width(1280);
    canvas.set_height(720);
    
    // Add canvas style for full window or responsive display
    canvas.style().set_property("background-color", "black")?;
    canvas.style().set_property("display", "block")?;

    // Try to mount canvas to app-container element if present, else fall back to body
    if let Some(container) = document.get_element_by_id("app-container") {
        container.append_child(&canvas)?;
    } else {
        body.append_child(&canvas)?;
    }

    // Embed the winit window directly into the canvas using raw-window-handle features
    #[allow(deprecated)]
    use winit::platform::web::WindowExtWebSys;
    let _canvas_assoc = window.canvas(); // Assoc canvas with winit window

    log::info!("Web canvas mounted. Initializing WebGPU...");

    // WebGPU Initialization
    let instance = wgpu::Instance::default();
    
    // Create rendering surface from window canvas
    let surface = unsafe { instance.create_surface(&window) }?;

    // Request graphics adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to locate compatible WebGPU adapter")?;

    // Request logical GPU device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WebGPU Logical Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None,
        )
        .await?;

    let surface_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = surface_capabilities.formats[0];

    // Configure swapchain surface configuration
    let size = window.inner_size();
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_capabilities.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    log::info!("WebGPU initialization successful. Starting render loop...");

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let frame = match surface.get_current_texture() {
                            Ok(texture) => texture,
                            Err(e) => {
                                log::error!("Dropped frame: {:?}", e);
                                return;
                            }
                        };
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Frame Encoder"),
                        });

                        // Standard render pass clearing the canvas to a deep space color
                        {
                            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Main Render Pass"),
                                color_attachments: &[Some(wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.03,
                                    g: 0.03,
                                    b: 0.06,
                                    a: 1.0,
                                }).ops(wgpu::StoreOp::Store))],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                        }

                        queue.submit(std::iter::once(encoder.finish()));
                        frame.present();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
