# AstroSim: High-Performance Astrophysics Simulator

AstroSim is an advanced, high-performance astrophysics simulator that renders the solar system, main asteroid belt, and Earth satellite constellations in real-time. Built on Rust, WebGPU (`wgpu`), and Dear ImGui, it targets both native desktop platforms and web browsers via WebAssembly (WebGL2 fallback).

---

## 🌟 Key Features

### 1. Accurate Time & Ephemerides
- **J2000 Epoch Time Synchronization**: Automatically determines the J2000 Julian date based on the user's local system time at startup.
- **Planetary State Calculations**: Uses Standish planetary rate tables to compute exact heliocentric starting positions and velocity vectors for the 8 major planets.
- **N-Body Gravity Integration**: Integrates the gravitational interactions of the major bodies using a Runge-Kutta 4th-order (RK4) integrator, allowing orbits to evolve dynamically.

### 2. Keplerian Orbit Solver & Small Bodies
- **Keplerian Analytical Solver**: Runs a numerically stable, analytical Keplerian solver for smaller objects to prevent orbit decay under fast simulation speeds.
- **Main Asteroid Belt**: Simulates 150 main-belt asteroids procedurally orbiting the Sun between Mars and Jupiter.
- **Earth Satellite Constellations**:
  - **ISS**: Low Earth Orbit (LEO) simulation at ~420 km altitude.
  - **Starlink**: 20 satellites in two separate orbital planes in LEO.
  - **GPS**: 6 satellites distributed across 6 Medium Earth Orbit (MEO) planes.
- **Visual Scaling**: Satellite orbits are dynamically scaled up relative to Earth's visual exaggeration factor to prevent visual clipping.

### 3. Advanced WebGPU Rendering Pipeline
- **Dynamic Spacetime Gravity Grid**: Visualizes gravitational potential wells using a high-resolution vertical grid warping mesh.
- **Procedural Volumetric Skybox**: Renders gaseous nebulae, starfields, and the Milky Way band procedurally using fractional Brownian motion (fBm) noise.
- **Atmospheric Rim Glow**: Features a custom shader atmospheric rim glow effect when hovering over planets.
- **Distance-Adaptive Scaling**: Dynamically scales celestial bodies at high zoom levels ($r_{\text{visual}} = \max(r_{\text{physical}}, \text{dist} \times \text{min\_size\_factor})$) so that tiny planets remain visible.

### 4. Interactive UI & Controls
- **Entity Inspector**: A comprehensive list of planets and satellites. Displays distance from the parent, orbital period, altitude, eccentricity, inclination, and instantaneous speed.
- **Simulation Control Panel**: Features time speed control (unlocked up to 100 days/second), time pausing, and real-time (1.0x) speed synchronization.
- **3D Raycast Selection**: Hover over and click on planets in 3D space to select them, aided by a neon rim glow and floating name labels.
- **Follow Camera (ImGui/Click)**: Centers on any selected planet or satellite. The yaw, pitch, and zoom are fully unlocked so you can rotate or zoom around the moving target.

---

## 🎮 Controls

- **Select Entity**: Select a planet by clicking on it in the viewport or using the **Select Entity** combo box in the *Entity Inspector* panel.
- **Follow Camera**: Check the **Follow Camera** box next to the selection dropdown to center the camera on the target.
- **Orbit Target**: Right-click and drag or use the **Arrow Keys** to rotate the camera around the planet/Sun.
- **Pan View**: Use **WASD** to translate the camera target in space (this automatically unlocks the follow camera).
- **Zoom In/Out**: Use the **Scroll Wheel** or **Q / E** keys to zoom.

---

## 🛠️ Build and Compilation

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (edition 2024)
- WebAssembly target if building for the web: `rustup target add wasm32-unknown-unknown`
- Emscripten SDK (required for WebAssembly C/C++ linking)

### Native Desktop Target
To run the simulator locally on native desktop platforms:
```bash
cargo run --release
```

### WebAssembly Target (Web)
To compile the WebAssembly library module manually:
1. Ensure the Emscripten SDK is sourced:
   ```bash
   source /path/to/emsdk/emsdk_env.sh
   ```
2. Build the target:
   ```bash
   mkdir -p lib
   echo "" > empty.c
   clang --target=wasm32-unknown-unknown -c empty.c -o empty.o
   ar rcs lib/libstdc++.a empty.o
   export CFLAGS="-I$EMSDK/upstream/emscripten/cache/sysroot/include"
   export CXXFLAGS="-I$EMSDK/upstream/emscripten/cache/sysroot/include -I$EMSDK/upstream/emscripten/cache/sysroot/include/c++/v1"
   export RUSTFLAGS="-L $EMSDK/upstream/emscripten/cache/sysroot/lib/wasm32-emscripten/ -C link-arg=-lc -L ./lib --cfg getrandom_backend=\"wasm_js\""
   
   cargo build --lib --target wasm32-unknown-unknown --release
   ```
3. Generate JS bindings:
   ```bash
   wasm-bindgen --target web --out-dir pkg target/wasm32-unknown-unknown/release/AstroSim.wasm
   ```
4. Serve the directory containing `index.html`, `wasi.js`, and `pkg`.
