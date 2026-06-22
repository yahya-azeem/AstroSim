use crate::physics::PhysicsEngine;
use crate::physics::cuda::CudaPhysicsEngine;
use crate::physics::vulkan_compute::VulkanComputePhysicsEngine;
use crate::render::{VulkanContext, Renderer};
use crate::render::renderer::BodyUbo;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::Instant;
use log::{info, warn};
use nalgebra::Vector3;

pub struct App {
    sdl_context: sdl2::Sdl,
    window: sdl2::video::Window,
    #[allow(dead_code)]
    vulkan_context: Arc<VulkanContext>,
    renderer: Renderer,
    physics_engine: Box<dyn PhysicsEngine>,
    imgui: imgui::Context,
    platform: imgui_sdl2_support::SdlPlatform,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vector3<f32>,
    selected_body_idx: usize,
    visual_warp_factor: f32,
    sim_speed: f64,
    paused: bool,
    body_names: Vec<String>,
    body_radii: Vec<f32>,
    body_types: Vec<u32>,
    history_trails: Vec<std::collections::VecDeque<Vector3<f32>>>,
    search_query: String,
    fetch_status: Arc<std::sync::Mutex<String>>,
    active_system_name: String,
    pending_system_data: Arc<std::sync::Mutex<Option<(String, Vec<(String, Vector3<f64>, Vector3<f64>, f64, f32, u32)>)>>>,
    follow_camera: bool,
    hovered_body_idx: Option<usize>,
}

impl App {
    pub fn new() -> Result<Self> {
        info!("Initializing SDL2 context...");
        let sdl_context = sdl2::init().map_err(|e| anyhow!(e))?;
        let video_subsystem = sdl_context.video().map_err(|e| anyhow!(e))?;

        info!("Creating SDL2 window requesting Vulkan support...");
        let window = video_subsystem
            .window("High-Performance Astrophysics Simulator", 1280, 720)
            .position_centered()
            .vulkan()
            .resizable()
            .build()?;

        // Query Vulkan instance extensions required by SDL2
        let sdl_instance_extensions = window.vulkan_instance_extensions()
            .map_err(|e| anyhow!("Failed to query SDL2 Vulkan extensions: {:?}", e))?;

        // Initialize Vulkan Context (creates surface inside)
        let vulkan_context = Arc::new(VulkanContext::new(&window, &sdl_instance_extensions)?);

        // Initialize ImGui Context
        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        
        // Initialize SdlPlatform backend for ImGui
        let platform = imgui_sdl2_support::SdlPlatform::new(&mut imgui);

        // Initialize Renderer
        let renderer = Renderer::new(Arc::clone(&vulkan_context), &mut imgui)?;

        // Initialize Physics Engine (attempt CUDA, fallback to Vulkan Compute CPU solver)
        let mut physics_engine: Box<dyn PhysicsEngine> = match CudaPhysicsEngine::try_new() {
            Ok(cuda_engine) => {
                info!("Using CUDA physics backend.");
                Box::new(cuda_engine)
            }
            Err(e) => {
                warn!("CUDA backend not available (using Vulkan Compute CPU fallback): {:?}", e);
                Box::new(VulkanComputePhysicsEngine::new())
            }
        };

        // Initialize the 9 main Solar System bodies
        let g = 6.67430e-11_f64;
        let m_sun = 1.989e30_f64;
        let au = 1.496e11_f64;

        let planet_params: [(&str, f64, f64); 9] = [
            ("Sun", 0.0, 1.989e30),
            ("Mercury", 0.387, 3.285e23),
            ("Venus", 0.723, 4.867e24),
            ("Earth", 1.000, 5.972e24),
            ("Mars", 1.524, 6.390e23),
            ("Jupiter", 5.203, 1.898e27),
            ("Saturn", 9.537, 5.683e26),
            ("Uranus", 19.191, 8.681e25),
            ("Neptune", 30.070, 1.024e26),
        ];

        for &(_name, r_au, mass) in &planet_params {
            if r_au == 0.0 {
                physics_engine.add_body(Vector3::zeros(), Vector3::zeros(), mass);
            } else {
                let r_m = r_au * au;
                let v_m = (g * m_sun / r_m).sqrt();
                physics_engine.add_body(
                    Vector3::new(r_m, 0.0, 0.0),
                    Vector3::new(0.0, 0.0, v_m),
                    mass,
                );
            }
        }

        let body_names = planet_params.iter().map(|(name, _, _)| name.to_string()).collect();
        let body_radii = vec![0.163, 0.00057, 0.00142, 0.0015, 0.0008, 0.0168, 0.0142, 0.006, 0.0058];
        let body_types = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let history_trails = vec![std::collections::VecDeque::with_capacity(1000); planet_params.len()];

        Ok(Self {
            sdl_context,
            window,
            vulkan_context,
            renderer,
            physics_engine,
            imgui,
            platform,
            camera_yaw: 0.0,
            camera_pitch: 35.0,
            camera_distance: 18.0,
            camera_target: Vector3::new(0.0, 0.0, 0.0),
            selected_body_idx: 3, // Default to Earth
            visual_warp_factor: 1.2,
            sim_speed: 15.0, // 15 days per second
            paused: false,
            body_names,
            body_radii,
            body_types,
            history_trails,
            search_query: String::new(),
            fetch_status: Arc::new(std::sync::Mutex::new("Idle".to_string())),
            active_system_name: "Solar System".to_string(),
            pending_system_data: Arc::new(std::sync::Mutex::new(None)),
            follow_camera: false,
            hovered_body_idx: None,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        let mut event_pump = self.sdl_context.event_pump().map_err(|e| anyhow!(e))?;
        unsafe {
            for t in 0x200..=0x20B {
                sdl2::sys::SDL_EventState(t, 0); // Disable window/syswm events
            }
            sdl2::sys::SDL_EventState(0x150, 0); // Disable display events
            
            // Disable joystick events (0x600 range)
            for t in 0x600..=0x604 {
                sdl2::sys::SDL_EventState(t, 0);
            }
            
            // Disable game controller events (0x700 range)
            for t in 0x700..=0x715 {
                sdl2::sys::SDL_EventState(t, 0);
            }
            
            // Disable audio device events (0xB00 range)
            for t in 0xB00..=0xB02 {
                sdl2::sys::SDL_EventState(t, 0);
            }
            
            // Disable sensor events (0xC00 range)
            sdl2::sys::SDL_EventState(0xC00, 0);
        }
        let mut last_time = Instant::now();
        let mut running = true;

        info!("Starting main simulation and render loop...");

        while running {
            // Check for loaded exoplanet system data from background fetch thread
            if let Ok(mut pending) = self.pending_system_data.try_lock() {
                if let Some((star_name, bodies)) = pending.take() {
                    info!("Loading exoplanetary system: {}", star_name);
                    self.active_system_name = star_name;
                    
                    self.physics_engine.clear();
                    self.body_names.clear();
                    self.body_radii.clear();
                    self.body_types.clear();
                    self.history_trails.clear();
                    
                    for (name, pos, vel, mass, radius, body_type) in bodies {
                        self.physics_engine.add_body(pos, vel, mass);
                        self.body_names.push(name);
                        self.body_radii.push(radius);
                        self.body_types.push(body_type);
                        self.history_trails.push(std::collections::VecDeque::with_capacity(1000));
                    }
                    
                    self.selected_body_idx = 0;
                    
                    let positions = self.physics_engine.get_positions();
                    let mut max_dist = 0.0;
                    for p in positions {
                        let dist_au = (p.norm() / 1.496e11) as f32;
                        if dist_au > max_dist {
                            max_dist = dist_au;
                        }
                    }
                    
                    if max_dist > 0.01 {
                        self.camera_distance = (max_dist * 1.5).clamp(2.0, 300.0);
                    } else {
                        self.camera_distance = 15.0;
                    }
                    self.camera_target = Vector3::zeros();
                    self.follow_camera = false;
                    self.hovered_body_idx = None;
                }
            }

            // Compute dimensions
            let window_size = self.window.size();
            let width = window_size.0;
            let height = window_size.1;

            // Compute delta time
            let now = Instant::now();
            let dt = now.duration_since(last_time).as_secs_f64();
            last_time = now;

            // Step the EIH physics integration
            let capped_dt = dt.min(0.1);
            if !self.paused {
                let sim_dt = capped_dt * 86400.0 * self.sim_speed; 
                self.physics_engine.step(sim_dt);
            }

            let au = 1.496e11_f64;
            let m_earth = 5.972e24_f64;
            let m_sun = 1.989e30_f64;

            let positions = self.physics_engine.get_positions().to_vec();
            let velocities = self.physics_engine.get_velocities().to_vec();
            let masses = self.physics_engine.get_masses().to_vec();

            // Chase camera lock logic: update camera target/yaw/pitch to track planet
            if self.follow_camera && !positions.is_empty() {
                let current = self.selected_body_idx.min(positions.len() - 1);
                let p_si = positions[current];
                
                let target_pos = Vector3::new(
                    (p_si.x / au) as f32,
                    (p_si.y / au) as f32,
                    (p_si.z / au) as f32,
                );
                self.camera_target = target_pos;
                
                let star_pos = if !positions.is_empty() {
                    let p_star = positions[0];
                    Vector3::new(
                        (p_star.x / au) as f32,
                        (p_star.y / au) as f32,
                        (p_star.z / au) as f32,
                    )
                } else {
                    Vector3::zeros()
                };

                let rel_pos = target_pos - star_pos;
                let dir_cam = if current > 0 && rel_pos.norm_squared() > 1e-6 {
                    let r_norm = rel_pos.normalize();
                    let tangent = Vector3::new(rel_pos.z, 0.0, -rel_pos.x).normalize();
                    
                    // Blend radial (outwards) and tangent (behind) for a cinematic over-the-shoulder view
                    let offset_horiz = (r_norm * 0.95 + tangent * 0.15).normalize();
                    (offset_horiz + Vector3::y() * 0.26).normalize()
                } else {
                    let yaw_rad = self.camera_yaw.to_radians();
                    let dir_behind = Vector3::new(yaw_rad.sin(), 0.0, yaw_rad.cos());
                    (dir_behind + Vector3::y() * 0.26).normalize()
                };
                
                let pitch_rad = dir_cam.y.asin();
                let yaw_rad = dir_cam.x.atan2(dir_cam.z);
                
                self.camera_pitch = pitch_rad.to_degrees().clamp(-85.0, 85.0);
                self.camera_yaw = yaw_rad.to_degrees();
            }

            // Compute MVP matrices on host
            let pitch_rad = self.camera_pitch.to_radians();
            let yaw_rad = self.camera_yaw.to_radians();

            let camera_pos = self.camera_target + Vector3::new(
                self.camera_distance * pitch_rad.cos() * yaw_rad.sin(),
                self.camera_distance * pitch_rad.sin(),
                self.camera_distance * pitch_rad.cos() * yaw_rad.cos(),
            );

            let view = nalgebra::Matrix4::look_at_rh(
                &nalgebra::Point3::from(camera_pos),
                &nalgebra::Point3::from(self.camera_target),
                &Vector3::y(),
            );

            let proj = nalgebra::Matrix4::new_perspective(
                (width as f32) / (height as f32),
                45.0f32.to_radians(),
                0.005,
                1000.0,
            );

            let correction = nalgebra::Matrix4::new(
                1.0,  0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0,  0.0, 0.5, 0.5,
                0.0,  0.0, 0.0, 1.0,
            );
            let proj_vk = correction * proj;
            
            // Calculate inverse View-Projection matrix for procedural skybox (no translation)
            let mut view_rot = view;
            view_rot.m14 = 0.0;
            view_rot.m24 = 0.0;
            view_rot.m34 = 0.0;
            view_rot.m41 = 0.0;
            view_rot.m42 = 0.0;
            view_rot.m43 = 0.0;
            view_rot.m44 = 1.0;
            let inv_view_proj = (proj_vk * view_rot).try_inverse().unwrap_or_else(|| nalgebra::Matrix4::identity());

            // 3D Raycasting Hover Check
            let mut hovered_idx = None;
            let mut min_t = f32::MAX;
            
            if !self.imgui.io().want_capture_mouse {
                let mouse_state = event_pump.mouse_state();
                let mx = mouse_state.x();
                let my = mouse_state.y();
                
                let x_ndc = (2.0 * mx as f32 / width as f32) - 1.0;
                let y_ndc = (2.0 * my as f32 / height as f32) - 1.0;
                
                let vp = proj_vk * view;
                if let Some(inv_vp) = vp.try_inverse() {
                    let p_near_h = inv_vp * nalgebra::Vector4::new(x_ndc, y_ndc, 0.0, 1.0);
                    let p_far_h = inv_vp * nalgebra::Vector4::new(x_ndc, y_ndc, 1.0, 1.0);
                    
                    let p_near = p_near_h.xyz() / p_near_h.w;
                    let p_far = p_far_h.xyz() / p_far_h.w;
                    let ray_dir = (p_far - p_near).normalize();
                    
                    for i in 0..positions.len() {
                        let p_si = positions[i];
                        let pos_render = Vector3::new(
                            (p_si.x / au) as f32,
                            (p_si.y / au) as f32,
                            (p_si.z / au) as f32,
                        );
                        
                        let dx = pos_render.x - camera_pos.x;
                        let dy = pos_render.y - camera_pos.y;
                        let dz = pos_render.z - camera_pos.z;
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                        
                        let b_type = self.body_types.get(i).copied().unwrap_or(101);
                        let min_size_factor = if b_type == 0 || b_type == 100 { 0.006 } else { 0.0025 };
                        let visual_radius = self.body_radii[i].max(dist * min_size_factor);
                        
                        let v = pos_render - p_near;
                        let t_proj = v.dot(&ray_dir);
                        if t_proj > 0.0 {
                            let d2 = v.norm_squared() - t_proj * t_proj;
                            let r2 = visual_radius * visual_radius;
                            let select_margin = 1.35;
                            if d2 <= r2 * select_margin {
                                let t_hit = t_proj - (r2 * select_margin - d2).sqrt();
                                if t_hit < min_t {
                                    min_t = t_hit;
                                    hovered_idx = Some(i);
                                }
                            }
                        }
                    }
                }
            }
            self.hovered_body_idx = hovered_idx;

            // Prepare ImGui Frame (updates mouse state before polling new events)
            self.platform.prepare_frame(&mut self.imgui, &self.window, &event_pump);

            let mut left_click_occurred = false;

            // Handle events
            for event in event_pump.poll_iter() {
                // Pass event to ImGui SdlPlatform
                self.platform.handle_event(&mut self.imgui, &event);

                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => {
                        running = false;
                    }
                    Event::Window {
                        win_event: sdl2::event::WindowEvent::Resized(w, h),
                        ..
                    } => {
                        self.renderer.resize(w as u32, h as u32)?;
                    }
                    // Handle camera zoom with scroll (proportional zoom for smooth multi-scale zoom)
                    Event::MouseWheel { y, .. } => {
                        if !self.imgui.io().want_capture_mouse {
                            if y > 0 {
                                self.camera_distance = (self.camera_distance * 0.85_f32.powi(y)).clamp(0.001, 300.0);
                            } else if y < 0 {
                                self.camera_distance = (self.camera_distance * 1.15_f32.powi(-y)).clamp(0.001, 300.0);
                            }
                            self.follow_camera = false; // Zooming breaks follow!
                        }
                    }
                    // Orbit camera by dragging right click
                    Event::MouseMotion { xrel, yrel, mousestate, .. } => {
                        if !self.imgui.io().want_capture_mouse && mousestate.right() {
                            self.camera_yaw += xrel as f32 * 0.25;
                            self.camera_pitch = (self.camera_pitch + yrel as f32 * 0.25).clamp(-85.0, 85.0);
                            self.follow_camera = false; // Disable chase camera lock on drag
                        }
                    }
                    // Left click selection event detection
                    Event::MouseButtonDown { mouse_btn: sdl2::mouse::MouseButton::Left, .. } => {
                        left_click_occurred = true;
                    }
                    _ => {}
                }
            }

            // Keyboard updates for camera controls (ignored if ImGui is capturing input)
            let keyboard_state = event_pump.keyboard_state();
            if !self.imgui.io().want_capture_keyboard {
                // Check if any manual movement key is pressed to disable chase camera follow
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Left)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Right)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Up)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Down)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::W)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::S)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::A)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::D)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Q)
                    || keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::E)
                {
                    self.follow_camera = false;
                }

                // Rotate camera with arrow keys
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Left) {
                    self.camera_yaw -= 1.5;
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Right) {
                    self.camera_yaw += 1.5;
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Up) {
                    self.camera_pitch = (self.camera_pitch - 1.5).clamp(-85.0, 85.0);
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Down) {
                    self.camera_pitch = (self.camera_pitch + 1.5).clamp(-85.0, 85.0);
                }

                // Pan target with WASD
                let yaw_rad = self.camera_yaw.to_radians();
                let forward = Vector3::new(yaw_rad.sin(), 0.0, yaw_rad.cos());
                let right = Vector3::new(yaw_rad.cos(), 0.0, -yaw_rad.sin());
                
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::W) {
                    self.camera_target -= forward * 0.2;
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::S) {
                    self.camera_target += forward * 0.2;
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::A) {
                    self.camera_target -= right * 0.2;
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::D) {
                    self.camera_target += right * 0.2;
                }
                
                // Zoom with Q/E keys (proportional zoom for smooth scaling)
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::Q) {
                    self.camera_distance = (self.camera_distance * 1.02).clamp(0.001, 300.0);
                }
                if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::E) {
                    self.camera_distance = (self.camera_distance * 0.98).clamp(0.001, 300.0);
                }
            }



            // Update trails history
            for i in 0..positions.len() {
                let p_si = positions[i];
                let p_render = Vector3::new(
                    (p_si.x / au) as f32,
                    (p_si.y / au) as f32,
                    (p_si.z / au) as f32,
                );
                
                if i >= self.history_trails.len() {
                    self.history_trails.push(std::collections::VecDeque::with_capacity(1000));
                }
                let trail = &mut self.history_trails[i];
                trail.push_back(p_render);
                if trail.len() > 1000 {
                    trail.pop_front();
                }
            }

            let mut edit_body: Option<(usize, Vector3<f64>, Vector3<f64>, f64)> = None;
            let mut restore_circular = false;
            let mut action_load_solar = false;
            let mut action_fetch_star: Option<String> = None;

            let active_system_name = self.active_system_name.clone();
            let mut selected_body_idx = self.selected_body_idx;
            let body_names = self.body_names.clone();

            // Run GUI windows inside their own local scope so that the Ui reference is dropped before rendering
            {
                let ui = self.imgui.new_frame();

                if left_click_occurred && !ui.io().want_capture_mouse {
                    if let Some(hovered) = self.hovered_body_idx {
                        selected_body_idx = hovered;
                        self.follow_camera = true;
                        
                        // Set camera distance to follow the body (10x further out than before)
                        let radius = self.body_radii[hovered];
                        self.camera_distance = (radius * 100.0).clamp(0.005, 300.0);
                    } else {
                        // Clicked empty space: stop following!
                        self.follow_camera = false;
                    }
                }

                // 1. Simulation Control Panel
                let mut paused = self.paused;
                let mut sim_speed = self.sim_speed as f32;
                let mut visual_warp = self.visual_warp_factor;
                
                ui.window("Simulation Control Panel")
                    .size([320.0, 200.0], imgui::Condition::FirstUseEver)
                    .build(|| {
                        ui.text("AstroSim Solar System Controller");
                        ui.separator();
                        
                        ui.checkbox("Pause Simulation", &mut paused);
                        ui.slider("Sim Speed (Days/Sec)", 0.0, 100.0, &mut sim_speed);
                        ui.slider("Gravity Well Warp", 0.01, 10.0, &mut visual_warp);
                        
                        ui.separator();
                        ui.text("Camera Controls:");
                        ui.text("- Arrow Keys: Rotate / Tilt");
                        ui.text("- WASD: Pan center position");
                        ui.text("- Scroll or Q/E: Zoom in/out");
                    });
                
                self.paused = paused;
                self.sim_speed = sim_speed as f64;
                self.visual_warp_factor = visual_warp;

                // 2. Entity Inspector Panel
                ui.window("Entity Inspector")
                    .size([380.0, 310.0], imgui::Condition::FirstUseEver)
                    .build(|| {
                        let mut current = selected_body_idx.min(body_names.len() - 1);
                        
                        // Combo box selector
                        if let Some(_token) = ui.begin_combo("Select Entity", &body_names[current]) {
                            for i in 0..body_names.len() {
                                if ui.selectable(&body_names[i]) {
                                    current = i;
                                }
                            }
                        }
                        selected_body_idx = current;
                        
                        ui.separator();
                        
                        let p = positions[current];
                        let v = velocities[current];
                        let m = masses[current];
                        
                        ui.text(format!("Inspect: {}", body_names[current]));
                        ui.text(format!("Distance from Star: {:.4} AU", p.norm() / au));
                        ui.text(format!("Velocity: {:.2} km/s", v.norm() / 1000.0));
                        
                        ui.separator();
                        
                        // Display and edit properties
                        let mut pos_au = [ (p.x / au) as f32, (p.y / au) as f32, (p.z / au) as f32 ];
                        let mut vel_kms = [ (v.x / 1000.0) as f32, (v.y / 1000.0) as f32, (v.z / 1000.0) as f32 ];
                        
                        // Edit mass in terms of Earth mass (or Sun mass for Sun)
                        let mut mass_scale = if current == 0 {
                            (m / m_sun) as f32
                        } else {
                            (m / m_earth) as f32
                        };
                        
                        let unit_label = if current == 0 { "M_solar" } else { "M_earth" };
                        
                        let mut pos_changed = false;
                        let mut vel_changed = false;
                        let mut mass_changed = false;
                        
                        if ui.input_float3("Position (AU)", &mut pos_au).build() {
                            pos_changed = true;
                        }
                        if ui.input_float3("Velocity (km/s)", &mut vel_kms).build() {
                            vel_changed = true;
                        }
                        
                        let mass_label = format!("Mass ({})", unit_label);
                        if ui.input_float(mass_label, &mut mass_scale).step(0.1).build() {
                            mass_changed = true;
                        }
                        
                        if pos_changed || vel_changed || mass_changed {
                            let new_pos = Vector3::new(
                                pos_au[0] as f64 * au,
                                pos_au[1] as f64 * au,
                                pos_au[2] as f64 * au,
                            );
                            let new_vel = Vector3::new(
                                vel_kms[0] as f64 * 1000.0,
                                vel_kms[1] as f64 * 1000.0,
                                vel_kms[2] as f64 * 1000.0,
                            );
                            let new_mass = if current == 0 {
                                mass_scale as f64 * m_sun
                            } else {
                                mass_scale as f64 * m_earth
                            };
                            edit_body = Some((current, new_pos, new_vel, new_mass));
                        }
                        
                        ui.separator();
                        if ui.button("Restore Circular Orbit") {
                            restore_circular = true;
                        }
                    });

                // 3. Exoplanet System Loader Panel
                let mut search_query = self.search_query.clone();
                let fetch_status = {
                    if let Ok(st) = self.fetch_status.lock() {
                        st.clone()
                    } else {
                        "Locked".to_string()
                    }
                };

                ui.window("Exoplanet System Loader")
                    .size([380.0, 220.0], imgui::Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!("Active System: {}", active_system_name));
                        ui.separator();

                        ui.text("Presets:");
                        if ui.button("Load Solar System") {
                            action_load_solar = true;
                        }
                        ui.same_line();
                        if ui.button("TRAPPIST-1") {
                            action_fetch_star = Some("TRAPPIST-1".to_string());
                        }
                        ui.same_line();
                        if ui.button("55 Cancri") {
                            action_fetch_star = Some("55 Cnc".to_string());
                        }

                        if ui.button("Kepler-90") {
                            action_fetch_star = Some("KOI-351".to_string());
                        }
                        ui.same_line();
                        if ui.button("Kepler-11") {
                            action_fetch_star = Some("Kepler-11".to_string());
                        }
                        ui.same_line();
                        if ui.button("Kepler-186") {
                            action_fetch_star = Some("Kepler-186".to_string());
                        }

                        ui.separator();
                        ui.text("Search NASA Exoplanet Archive:");
                        if ui.input_text("Star Name", &mut search_query).enter_returns_true(true).build() {
                            action_fetch_star = Some(search_query.clone());
                        }
                        ui.same_line();
                        if ui.button("Fetch") {
                            action_fetch_star = Some(search_query.clone());
                        }

                        ui.text(format!("Status: {}", fetch_status));
                    });
                
                self.search_query = search_query;

                // 4. Hover floating text label above mouse cursor
                if let Some(hovered) = self.hovered_body_idx {
                    if !body_names.is_empty() {
                        let mouse_state = event_pump.mouse_state();
                        let mx = mouse_state.x() as f32;
                        let my = mouse_state.y() as f32;

                        let name = &body_names[hovered.min(body_names.len() - 1)];
                        let est_width = name.len() as f32 * 7.5 + 16.0;

                        ui.window("HoverLabel")
                            .position([mx - est_width / 2.0, my - 35.0], imgui::Condition::Always)
                            .size([est_width, 25.0], imgui::Condition::Always)
                            .title_bar(false)
                            .resizable(false)
                            .movable(false)
                            .scroll_bar(false)
                            .no_inputs()
                            .draw_background(true)
                            .build(|| {
                                ui.set_cursor_pos([8.0, 4.0]);
                                ui.text(name);
                            });
                    }
                }

                // Obsolete ImGui overlay trails and labels removed.
                // Orbits are now rendered directly as 3D line strips in Vulkan.
            }

            self.selected_body_idx = selected_body_idx;

            if action_load_solar {
                self.load_preset_solar_system();
            } else if let Some(query) = action_fetch_star {
                self.trigger_fetch(query);
            }

            if let Some((idx, new_pos, new_vel, new_mass)) = edit_body {
                self.physics_engine.set_body(idx, new_pos, new_vel, new_mass);
            }

            if restore_circular {
                let current = self.selected_body_idx.min(self.body_names.len() - 1);
                if current > 0 {
                    let g = 6.67430e-11_f64;
                    let positions = self.physics_engine.get_positions().to_vec();
                    let masses = self.physics_engine.get_masses().to_vec();
                    let star_mass = masses[0];
                    let p = positions[current];
                    let r_m = p.norm();
                    if r_m > 0.0 {
                        let v_mag = (g * (star_mass + masses[current]) / r_m).sqrt();
                        let dir = if p.x.abs() < 1e-5 && p.z.abs() < 1e-5 {
                            Vector3::new(1.0, 0.0, 0.0)
                        } else {
                            Vector3::new(-p.z, 0.0, p.x).normalize()
                        };
                        let new_vel = dir * v_mag;
                        self.physics_engine.set_body(current, p, new_vel, masses[current]);
                    }
                }
            }

            // Once the UI block ends and the Ui reference is dropped, retrieve the draw data
            let imgui_draw_data = self.imgui.render();

            // Build BodyUbo data to pass to the Vulkan renderer for grid warping
            let mut body_ubos = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                let p_si = positions[i];
                let p_render = Vector3::new(
                    (p_si.x / au) as f32,
                    (p_si.y / au) as f32,
                    (p_si.z / au) as f32,
                );

                // Warping factor based log-scale of mass relative to Earth
                let m = masses[i];
                let relative_mass = m / m_earth;
                let log_mass = (relative_mass + 1.0).log10();
                
                let strength = self.visual_warp_factor * (log_mass as f32) * 0.15;
                
                body_ubos.push(BodyUbo {
                    pos_mass: [p_render.x, p_render.y, p_render.z, strength],
                });
            }

            // Construct body colors dynamically
            let mut body_colors = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                let b_type = self.body_types.get(i).copied().unwrap_or(101);
                let col = match b_type {
                    0 => [1.0, 0.9, 0.2, 1.0],   // Sun
                    1 => [0.6, 0.6, 0.6, 1.0],   // Mercury
                    2 => [0.9, 0.7, 0.5, 1.0],   // Venus
                    3 => [0.2, 0.6, 1.0, 1.0],   // Earth
                    4 => [0.9, 0.3, 0.2, 1.0],   // Mars
                    5 => [0.8, 0.6, 0.5, 1.0],   // Jupiter
                    6 => [0.9, 0.8, 0.6, 1.0],   // Saturn
                    7 => [0.5, 0.8, 0.9, 1.0],   // Uranus
                    8 => [0.2, 0.4, 0.9, 1.0],   // Neptune
                    100 => [0.9, 0.4, 0.2, 1.0], // Exoplanet Star
                    _ => [0.4, 0.6, 0.8, 1.0],   // Exoplanet Planet
                };
                body_colors.push(col);
            }

            // Prepare 2D coordinates (X, Z) of trails to pass to Vulkan orbit renderer
            let trails: Vec<Vec<[f32; 2]>> = self.history_trails.iter().map(|trail| {
                trail.iter().map(|p| [p.x, p.z]).collect()
            }).collect();

            // 4. Trigger Vulkan rendering pass
            self.renderer.draw_frame(
                view.into(),
                proj_vk.into(),
                inv_view_proj.into(),
                &body_ubos,
                &self.body_radii,
                &self.body_types,
                &body_colors,
                self.selected_body_idx,
                self.hovered_body_idx,
                [camera_pos.x, camera_pos.y, camera_pos.z],
                &trails,
                imgui_draw_data,
            )?;

            // Frame throttling
            std::thread::sleep(std::time::Duration::from_millis(16));
        }

        info!("Main simulation and render loop terminated.");
        Ok(())
    }

    fn load_preset_solar_system(&mut self) {
        info!("Loading Solar System preset...");
        self.active_system_name = "Solar System".to_string();
        self.physics_engine.clear();
        self.body_names.clear();
        self.body_radii.clear();
        self.body_types.clear();
        self.history_trails.clear();
        
        let planet_params: [(&str, f64, f64, f32, u32); 9] = [
            ("Sun", 0.0, 1.989e30, 0.163, 0),
            ("Mercury", 0.387, 3.285e23, 0.00057, 1),
            ("Venus", 0.723, 4.867e24, 0.00142, 2),
            ("Earth", 1.000, 5.972e24, 0.0015, 3),
            ("Mars", 1.524, 6.390e23, 0.0008, 4),
            ("Jupiter", 5.203, 1.898e27, 0.0168, 5),
            ("Saturn", 9.537, 5.683e26, 0.0142, 6),
            ("Uranus", 19.191, 8.681e25, 0.006, 7),
            ("Neptune", 30.070, 1.024e26, 0.0058, 8),
        ];

        let g = 6.67430e-11_f64;
        let m_sun = 1.989e30_f64;
        let au = 1.496e11_f64;

        for &(name, r_au, mass, radius, body_type) in &planet_params {
            if r_au == 0.0 {
                self.physics_engine.add_body(Vector3::zeros(), Vector3::zeros(), mass);
            } else {
                let r_m = r_au * au;
                let v_m = (g * m_sun / r_m).sqrt();
                self.physics_engine.add_body(
                    Vector3::new(r_m, 0.0, 0.0),
                    Vector3::new(0.0, 0.0, v_m),
                    mass,
                );
            }
            self.body_names.push(name.to_string());
            self.body_radii.push(radius);
            self.body_types.push(body_type);
            self.history_trails.push(std::collections::VecDeque::with_capacity(1000));
        }

        self.selected_body_idx = 3; // Earth
        self.camera_distance = 18.0;
        self.camera_target = Vector3::zeros();
        self.follow_camera = false;
        self.hovered_body_idx = None;
    }

    fn trigger_fetch(&mut self, query: String) {
        if query.trim().is_empty() {
            return;
        }
        {
            if let Ok(mut status) = self.fetch_status.lock() {
                *status = format!("Searching {}...", query);
            }
        }
        let pending = Arc::clone(&self.pending_system_data);
        let status = Arc::clone(&self.fetch_status);
        fetch_star_system(query.trim().to_string(), pending, status);
    }
}

fn fetch_star_system(
    query: String,
    pending_data: Arc<std::sync::Mutex<Option<(String, Vec<(String, Vector3<f64>, Vector3<f64>, f64, f32, u32)>)>>>,
    status: Arc<std::sync::Mutex<String>>,
) {
    std::thread::spawn(move || {
        let url1 = format!(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+hostname+from+pscomppars+where+hostname+like+'{}%25'+or+pl_name+like+'{}%25'&format=json",
            query.replace(" ", "+"),
            query.replace(" ", "+")
        );
        
        info!("Fetching hostname mapping from Caltech: {}", url1);
        let output1 = std::process::Command::new("curl")
            .arg("-s")
            .arg(&url1)
            .output();
            
        let hostname = match output1 {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(arr) = json.as_array() {
                        if !arr.is_empty() {
                            arr[0]["hostname"].as_str().map(|s| s.to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Err(_) => None,
        };
        
        let found_host = match hostname {
            Some(h) => h,
            None => {
                // Try query directly if mapping was empty
                query.clone()
            }
        };
        
        {
            if let Ok(mut st) = status.lock() {
                *st = format!("Fetching data for {}...", found_host);
            }
        }
        
        let url2 = format!(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbsmax,pl_orbeccen,pl_masse,pl_rade,st_mass,st_rad+from+pscomppars+where+hostname='{}'&format=json",
            found_host.replace(" ", "+")
        );
        
        info!("Fetching planet data from Caltech: {}", url2);
        let output2 = std::process::Command::new("curl")
            .arg("-s")
            .arg(&url2)
            .output();
            
        match output2 {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(arr) = json.as_array() {
                        if arr.is_empty() {
                            if let Ok(mut st) = status.lock() {
                                *st = format!("No planets found for {}", found_host);
                            }
                            return;
                        }
                        
                        let mut bodies = Vec::new();
                        let mut st_mass_val = arr[0]["st_mass"].as_f64().unwrap_or(1.0);
                        if st_mass_val <= 0.0 {
                            st_mass_val = 1.0;
                        }
                        let mut st_rad_val = arr[0]["st_rad"].as_f64().unwrap_or(1.0);
                        if st_rad_val <= 0.0 {
                            st_rad_val = 1.0;
                        }
                        let m_sun = 1.989e30_f64;
                        let star_mass = st_mass_val * m_sun;
                        
                        // Find the innermost planet's orbit to prevent collisions in compact systems
                        let mut min_orbit = 999.0_f64;
                        for pl in arr {
                            let a = pl["pl_orbsmax"].as_f64().unwrap_or(1.0);
                            if a > 0.0 && a < min_orbit {
                                min_orbit = a;
                            }
                        }
                        if min_orbit > 500.0 {
                            min_orbit = 1.0;
                        }
                        
                        let ideal_star_radius = st_rad_val as f32 * 0.163;
                        let star_radius = ideal_star_radius.clamp(0.005, (min_orbit as f32 * 0.35).max(0.005));
                        
                        // Add host star at [0,0,0]
                        bodies.push((
                            found_host.clone(),
                            Vector3::zeros(),
                            Vector3::zeros(),
                            star_mass,
                            star_radius,
                            100u32, // Exoplanet star type
                        ));
                        
                        let g = 6.67430e-11_f64;
                        let au = 1.496e11_f64;
                        let m_earth = 5.972e24_f64;
                        
                        for pl in arr {
                            let name = pl["pl_name"].as_str().unwrap_or("Unknown Planet").to_string();
                            let a = pl["pl_orbsmax"].as_f64().unwrap_or(1.0);
                            let e = pl["pl_orbeccen"].as_f64().unwrap_or(0.0);
                            let pl_mass_val = pl["pl_masse"].as_f64().unwrap_or(1.0);
                            let planet_mass = pl_mass_val * m_earth;
                            
                            let mut pl_rade_val = pl["pl_rade"].as_f64().unwrap_or(1.0);
                            if pl_rade_val <= 0.0 {
                                pl_rade_val = 1.0;
                            }
                            let planet_radius = (star_radius * pl_rade_val as f32 / (st_rad_val as f32 * 109.0)).max(0.0001);
                            
                            let a_m = a * au;
                            let r_m = a_m * (1.0 - e);
                            let v_m = ((g * (star_mass + planet_mass) / a_m) * (1.0 + e) / (1.0 - e)).sqrt();
                            
                            bodies.push((
                                name,
                                Vector3::new(r_m, 0.0, 0.0),
                                Vector3::new(0.0, 0.0, v_m),
                                planet_mass,
                                planet_radius,
                                101u32, // Generic Exoplanet planet type
                            ));
                        }
                        
                        if let Ok(mut pending) = pending_data.lock() {
                            *pending = Some((found_host.clone(), bodies));
                        }
                        if let Ok(mut st) = status.lock() {
                            *st = format!("Loaded {} system!", found_host);
                        }
                    }
                } else {
                    if let Ok(mut st) = status.lock() {
                        *st = format!("Failed to parse JSON for {}", found_host);
                    }
                }
            }
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = format!("Network query failed for {}", found_host);
                }
            }
        }
    });
}
