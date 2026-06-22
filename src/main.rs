mod app;
mod physics;
mod render;
mod ephemeris;

use anyhow::Result;
use winit::event_loop::EventLoop;

fn main() -> Result<()> {
    env_logger::init();
    println!("Initializing High-Performance Astrophysics Simulator...");

    let event_loop = EventLoop::new()?;
    let mut app = app::AstroSimApp::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
