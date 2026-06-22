mod app;
mod physics;
mod render;
mod ephemeris;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();
    println!("Initializing High-Performance Astrophysics Simulator...");

    let mut app = app::App::new()?;
    app.run()?;

    Ok(())
}
