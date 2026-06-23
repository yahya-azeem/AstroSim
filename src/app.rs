use crate::physics::PhysicsEngine;
use crate::physics::vulkan_compute::VulkanComputePhysicsEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::physics::cuda::CudaPhysicsEngine;

use crate::render::renderer::BodyUbo;
use crate::render::Renderer;

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::collections::HashSet;
use log::{info, warn};
use nalgebra::Vector3;
use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, MouseButton, MouseScrollDelta};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};
use winit::keyboard::{KeyCode, PhysicalKey};

#[cfg(target_arch = "wasm32")]
fn get_time_seconds() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0)
}

#[cfg(not(target_arch = "wasm32"))]
fn get_time_seconds() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

pub struct AppState {
    window: Arc<Window>,
    #[allow(dead_code)]
    device: Arc<wgpu::Device>,
    #[allow(dead_code)]
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    physics_engine: Box<dyn PhysicsEngine>,
    imgui: imgui::Context,
    #[cfg(not(target_arch = "wasm32"))]
    platform: imgui_winit_support::WinitPlatform,
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
    last_time: f64,
    pressed_keys: HashSet<KeyCode>,
    mouse_position: (f32, f32),
    #[cfg(target_arch = "wasm32")]
    left_mouse_down: bool,
    right_mouse_down: bool,
    left_click_occurred: bool,
}

impl AppState {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        info!("Initializing cross-platform wgpu context...");
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())
            .map_err(|e| anyhow!("Failed to create surface: {:?}", e))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow!("Failed to request adapter: {:?}", e))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Logical Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    experimental_features: Default::default(),
                    memory_hints: Default::default(),
                    trace: wgpu::Trace::default(),
                },
            )
            .await
            .map_err(|e| anyhow!("Failed to request device: {:?}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = surface_capabilities.formats[0];

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        info!("Initializing ImGui Context...");
        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        
        #[cfg(not(target_arch = "wasm32"))]
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        #[cfg(not(target_arch = "wasm32"))]
        platform.attach_window(imgui.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);

        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            surface,
            surface_config,
            &mut imgui,
        )?;

        let mut physics_engine: Box<dyn PhysicsEngine> = Box::new(VulkanComputePhysicsEngine::new());
        #[cfg(not(target_arch = "wasm32"))]
        {
            match CudaPhysicsEngine::try_new() {
                Ok(cuda_engine) => {
                    info!("Using CUDA physics backend.");
                    physics_engine = Box::new(cuda_engine);
                }
                Err(e) => {
                    warn!("CUDA backend not available (using CPU solver fallback): {:?}", e);
                }
            }
        }

        let mut state = Self {
            window,
            device,
            queue,
            renderer,
            physics_engine,
            imgui,
            #[cfg(not(target_arch = "wasm32"))]
            platform,
            camera_yaw: 0.0,
            camera_pitch: 35.0,
            camera_distance: 18.0,
            camera_target: Vector3::new(0.0, 0.0, 0.0),
            selected_body_idx: 3, // Default to Earth
            visual_warp_factor: 1.2,
            sim_speed: 15.0, // 15 days per second
            paused: false,
            body_names: Vec::new(),
            body_radii: Vec::new(),
            body_types: Vec::new(),
            history_trails: Vec::new(),
            search_query: String::new(),
            fetch_status: Arc::new(std::sync::Mutex::new("Idle".to_string())),
            active_system_name: "Solar System".to_string(),
            pending_system_data: Arc::new(std::sync::Mutex::new(None)),
            follow_camera: false,
            hovered_body_idx: None,
            last_time: get_time_seconds(),
            pressed_keys: HashSet::new(),
            mouse_position: (0.0, 0.0),
            #[cfg(target_arch = "wasm32")]
            left_mouse_down: false,
            right_mouse_down: false,
            left_click_occurred: false,
        };

        state.load_preset_solar_system();
        Ok(state)
    }

    pub fn load_preset_solar_system(&mut self) {
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

    pub fn update_and_render(&mut self) -> Result<()> {
        // Check for loaded exoplanet system data
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

        // Get window dimensions
        let window_size = self.window.inner_size();
        let width = window_size.width;
        let height = window_size.height;
        if width == 0 || height == 0 {
            return Ok(());
        }

        // Compute delta time
        let now = get_time_seconds();
        let dt = now - self.last_time;
        self.last_time = now;

        // Step physics engine
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

        // Keyboard camera updates
        if !self.imgui.io().want_capture_keyboard {
            if self.pressed_keys.contains(&KeyCode::ArrowLeft)
                || self.pressed_keys.contains(&KeyCode::ArrowRight)
                || self.pressed_keys.contains(&KeyCode::ArrowUp)
                || self.pressed_keys.contains(&KeyCode::ArrowDown)
                || self.pressed_keys.contains(&KeyCode::KeyW)
                || self.pressed_keys.contains(&KeyCode::KeyS)
                || self.pressed_keys.contains(&KeyCode::KeyA)
                || self.pressed_keys.contains(&KeyCode::KeyD)
                || self.pressed_keys.contains(&KeyCode::KeyQ)
                || self.pressed_keys.contains(&KeyCode::KeyE)
            {
                self.follow_camera = false;
            }

            if self.pressed_keys.contains(&KeyCode::ArrowLeft) {
                self.camera_yaw -= 1.5;
            }
            if self.pressed_keys.contains(&KeyCode::ArrowRight) {
                self.camera_yaw += 1.5;
            }
            if self.pressed_keys.contains(&KeyCode::ArrowUp) {
                self.camera_pitch = (self.camera_pitch - 1.5).clamp(-85.0, 85.0);
            }
            if self.pressed_keys.contains(&KeyCode::ArrowDown) {
                self.camera_pitch = (self.camera_pitch + 1.5).clamp(-85.0, 85.0);
            }

            let yaw_rad = self.camera_yaw.to_radians();
            let forward = Vector3::new(yaw_rad.sin(), 0.0, yaw_rad.cos());
            let right = Vector3::new(yaw_rad.cos(), 0.0, -yaw_rad.sin());
            
            if self.pressed_keys.contains(&KeyCode::KeyW) {
                self.camera_target -= forward * 0.2;
            }
            if self.pressed_keys.contains(&KeyCode::KeyS) {
                self.camera_target += forward * 0.2;
            }
            if self.pressed_keys.contains(&KeyCode::KeyA) {
                self.camera_target -= right * 0.2;
            }
            if self.pressed_keys.contains(&KeyCode::KeyD) {
                self.camera_target += right * 0.2;
            }
            
            if self.pressed_keys.contains(&KeyCode::KeyQ) {
                self.camera_distance = (self.camera_distance * 1.02).clamp(0.001, 300.0);
            }
            if self.pressed_keys.contains(&KeyCode::KeyE) {
                self.camera_distance = (self.camera_distance * 0.98).clamp(0.001, 300.0);
            }
        }

        // Chase camera follow lock logic
        if self.follow_camera && !positions.is_empty() {
            let current = self.selected_body_idx.min(positions.len() - 1);
            let p_si = positions[current];
            
            let target_pos = Vector3::new(
                (p_si.x / au) as f32,
                (p_si.y / au) as f32,
                (p_si.z / au) as f32,
            );
            self.camera_target = target_pos;
            
            let p_star = if !positions.is_empty() {
                positions[0]
            } else {
                Vector3::zeros()
            };

            let rel_pos_f64 = p_si - p_star;
            let rel_pos = Vector3::new(
                (rel_pos_f64.x / au) as f32,
                (rel_pos_f64.y / au) as f32,
                (rel_pos_f64.z / au) as f32,
            );
            let dir_cam = if current > 0 && rel_pos.norm_squared() > 1e-6 {
                let r_norm = rel_pos.normalize();
                let tangent = Vector3::new(rel_pos.z, 0.0, -rel_pos.x).normalize();
                
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

        // Compute View, Projection, and VP matrices
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
            0.0,  1.0, 0.0, 0.0,
            0.0,  0.0, 0.5, 0.5,
            0.0,  0.0, 0.0, 1.0,
        );
        let proj_vk = correction * proj;
        
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
            let mx = self.mouse_position.0;
            let my = self.mouse_position.1;
            
            let x_ndc = (2.0 * mx / width as f32) - 1.0;
            let y_ndc = 1.0 - (2.0 * my / height as f32);
            
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

        // ImGui frame update
        let current_left_click = self.left_click_occurred;
        self.left_click_occurred = false;

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

        #[cfg(not(target_arch = "wasm32"))]
        self.platform.prepare_frame(self.imgui.io_mut(), &self.window).unwrap();
        #[cfg(target_arch = "wasm32")]
        {
            self.imgui.io_mut().display_size = [width as f32, height as f32];
            self.imgui.io_mut().delta_time = dt as f32;
            self.imgui.io_mut().mouse_pos = [self.mouse_position.0, self.mouse_position.1];
            self.imgui.io_mut().mouse_down[0] = self.left_mouse_down;
            self.imgui.io_mut().mouse_down[1] = self.right_mouse_down;
        }

        {
            let ui = self.imgui.new_frame();

            if current_left_click && !ui.io().want_capture_mouse {
                if let Some(hovered) = self.hovered_body_idx {
                    selected_body_idx = hovered;
                    self.follow_camera = true;
                    let radius = self.body_radii[hovered];
                    self.camera_distance = (radius * 100.0).clamp(0.005, 300.0);
                } else {
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
                    if body_names.is_empty() { return; }
                    let mut current = selected_body_idx.min(body_names.len() - 1);
                    
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
                    
                    let mut pos_au = [ (p.x / au) as f32, (p.y / au) as f32, (p.z / au) as f32 ];
                    let mut vel_kms = [ (v.x / 1000.0) as f32, (v.y / 1000.0) as f32, (v.z / 1000.0) as f32 ];
                    
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
                    let mx = self.mouse_position.0;
                    let my = self.mouse_position.1;

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

        let imgui_draw_data = self.imgui.render();

        let mut body_ubos = Vec::with_capacity(positions.len());
        for i in 0..positions.len() {
            let p_si = positions[i];
            let p_render = Vector3::new(
                (p_si.x / au) as f32,
                (p_si.y / au) as f32,
                (p_si.z / au) as f32,
            );

            let m = masses[i];
            let relative_mass = m / m_earth;
            let log_mass = (relative_mass + 1.0).log10();
            
            let strength = self.visual_warp_factor * (log_mass as f32) * 0.15;
            
            body_ubos.push(BodyUbo {
                pos_mass: [p_render.x, p_render.y, p_render.z, strength],
            });
        }

        let mut body_colors = Vec::with_capacity(positions.len());
        for i in 0..positions.len() {
            let b_type = self.body_types.get(i).copied().unwrap_or(101);
            let col = match b_type {
                0 => [1.0, 0.9, 0.2, 1.0],
                1 => [0.6, 0.6, 0.6, 1.0],
                2 => [0.9, 0.7, 0.5, 1.0],
                3 => [0.2, 0.6, 1.0, 1.0],
                4 => [0.9, 0.3, 0.2, 1.0],
                5 => [0.8, 0.6, 0.5, 1.0],
                6 => [0.9, 0.8, 0.6, 1.0],
                7 => [0.5, 0.8, 0.9, 1.0],
                8 => [0.2, 0.4, 0.9, 1.0],
                100 => [0.9, 0.4, 0.2, 1.0],
                _ => [0.4, 0.6, 0.8, 1.0],
            };
            body_colors.push(col);
        }

        let trails: Vec<Vec<[f32; 2]>> = self.history_trails.iter().map(|trail| {
            trail.iter().map(|p| [p.x, p.z]).collect()
        }).collect();

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

        Ok(())
    }
}

pub struct AstroSimApp {
    state: Option<AppState>,
    #[cfg(target_arch = "wasm32")]
    async_state: Arc<std::sync::Mutex<Option<AppState>>>,
    #[cfg(target_arch = "wasm32")]
    initializing: bool,
}

impl AstroSimApp {
    pub fn new() -> Self {
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            async_state: Arc::new(std::sync::Mutex::new(None)),
            #[cfg(target_arch = "wasm32")]
            initializing: false,
        }
    }
}

impl ApplicationHandler for AstroSimApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.state.is_none() {
                let window = Arc::new(event_loop.create_window(
                    Window::default_attributes()
                        .with_title("High-Performance Astrophysics Simulator")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
                ).unwrap());
                
                let state = pollster::block_on(AppState::new(window)).expect("Failed to initialize AppState");
                self.state = Some(state);
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            if self.state.is_none() && !self.initializing {
                self.initializing = true;
                
                let window = Arc::new(event_loop.create_window(
                    Window::default_attributes()
                        .with_title("AstroSim Browser Port")
                ).unwrap());
                
                // Mount canvas to DOM
                let web_window = web_sys::window().expect("No global window found");
                let document = web_window.document().expect("No global document found");
                let body = document.body().expect("No body element found");
                
                #[allow(deprecated)]
                use winit::platform::web::WindowExtWebSys;
                let canvas = window.canvas().expect("Failed to retrieve winit canvas");
                
                canvas.set_id("astrosim-canvas");
                canvas.set_width(1280);
                canvas.set_height(720);
                
                canvas.style().set_property("background-color", "black").unwrap();
                canvas.style().set_property("display", "block").unwrap();
                
                if let Some(container) = document.get_element_by_id("app-container") {
                    container.append_child(&canvas).unwrap();
                } else {
                    body.append_child(&canvas).unwrap();
                }
                
                let async_state = self.async_state.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    log::info!("Starting async AppState initialization...");
                    match AppState::new(window).await {
                        Ok(state) => {
                            log::info!("AppState initialized successfully!");
                            *async_state.lock().unwrap() = Some(state);
                        }
                        Err(e) => {
                            log::error!("Failed to initialize AppState: {:?}", e);
                        }
                    }
                });
            }
        }
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.state.is_none() {
                if let Ok(mut lock) = self.async_state.try_lock() {
                    if let Some(state) = lock.take() {
                        self.state = Some(state);
                    }
                }
            }
        }
        
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };
        
        if window_id != state.window.id() {
            return;
        }
        
        // Pass event to ImGui
        #[cfg(not(target_arch = "wasm32"))]
        {
            let winit_event: winit::event::Event<()> = winit::event::Event::WindowEvent {
                window_id,
                event: event.clone(),
            };
            state.platform.handle_event(state.imgui.io_mut(), &state.window, &winit_event);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let io = state.imgui.io_mut();
            match &event {
                WindowEvent::CursorMoved { position, .. } => {
                    io.add_mouse_pos_event([position.x as f32, position.y as f32]);
                }
                WindowEvent::MouseInput { state: button_state, button, .. } => {
                    let pressed = button_state.is_pressed();
                    let imgui_button = match button {
                        MouseButton::Left => {
                            state.left_mouse_down = pressed;
                            Some(imgui::MouseButton::Left)
                        }
                        MouseButton::Right => {
                            state.right_mouse_down = pressed;
                            Some(imgui::MouseButton::Right)
                        }
                        MouseButton::Middle => Some(imgui::MouseButton::Middle),
                        _ => None,
                    };
                    if let Some(b) = imgui_button {
                        io.add_mouse_button_event(b, pressed);
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let (x, y) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (*x, *y),
                        MouseScrollDelta::PixelDelta(pos) => (pos.x as f32 / 120.0, pos.y as f32 / 120.0),
                    };
                    io.add_mouse_wheel_event([x, y]);
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    let pressed = key_event.state.is_pressed();
                    if let PhysicalKey::Code(keycode) = key_event.physical_key {
                        if let Some(imgui_key) = map_winit_key_to_imgui(keycode) {
                            io.add_key_event(imgui_key, pressed);
                        }
                    }
                    if pressed {
                        if let Some(text) = &key_event.text {
                            for c in text.chars() {
                                if !c.is_control() {
                                    io.add_input_character(c);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size.width, new_size.height);
                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if let PhysicalKey::Code(keycode) = key_event.physical_key {
                    if key_event.state.is_pressed() {
                        state.pressed_keys.insert(keycode);
                    } else {
                        state.pressed_keys.remove(&keycode);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x as f32 - state.mouse_position.0;
                let dy = position.y as f32 - state.mouse_position.1;
                state.mouse_position = (position.x as f32, position.y as f32);
                
                if state.right_mouse_down && !state.imgui.io().want_capture_mouse {
                    state.camera_yaw += dx * 0.25;
                    state.camera_pitch = (state.camera_pitch + dy * 0.25).clamp(-85.0, 85.0);
                    state.follow_camera = false;
                }
            }
            WindowEvent::MouseInput { state: button_state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        if button_state.is_pressed() {
                            if !state.imgui.io().want_capture_mouse {
                                state.left_click_occurred = true;
                            }
                        }
                    }
                    MouseButton::Right => {
                        state.right_mouse_down = button_state.is_pressed();
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !state.imgui.io().want_capture_mouse {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y_scroll) => y_scroll,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y / 30.0) as f32,
                    };
                    if y > 0.0 {
                        state.camera_distance = (state.camera_distance * 0.85_f32.powf(y)).clamp(0.001, 300.0);
                    } else if y < 0.0 {
                        state.camera_distance = (state.camera_distance * 1.15_f32.powf(-y)).clamp(0.001, 300.0);
                    }
                    state.follow_camera = false;
                }
            }
            WindowEvent::RedrawRequested => {
                let _ = state.update_and_render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
    
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.state.is_none() {
                if let Ok(mut lock) = self.async_state.try_lock() {
                    if let Some(state) = lock.take() {
                        self.state = Some(state);
                    }
                }
            }
        }
        
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

// Target-gated exoplanet table loader function
#[cfg(not(target_arch = "wasm32"))]
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
        
        let found_host = hostname.unwrap_or(query);
        
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
                        
                        bodies.push((
                            found_host.clone(),
                            Vector3::zeros(),
                            Vector3::zeros(),
                            star_mass,
                            star_radius,
                            100u32,
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
                                101u32,
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

#[cfg(target_arch = "wasm32")]
fn fetch_star_system(
    query: String,
    pending_data: Arc<std::sync::Mutex<Option<(String, Vec<(String, Vector3<f64>, Vector3<f64>, f64, f32, u32)>)>>>,
    status: Arc<std::sync::Mutex<String>>,
) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    wasm_bindgen_futures::spawn_local(async move {
        let url1 = format!(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+hostname+from+pscomppars+where+hostname+like+'{}%25'+or+pl_name+like+'{}%25'&format=json",
            query.replace(" ", "+"),
            query.replace(" ", "+")
        );

        log::info!("Web fetching hostname mapping from Caltech: {}", url1);

        let window = web_sys::window().unwrap();
        let mut opts = web_sys::RequestInit::new();
        opts.method("GET");
        opts.mode(web_sys::RequestMode::Cors);

        let request = match web_sys::Request::new_with_str_and_init(&url1, &opts) {
            Ok(r) => r,
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Failed to construct request".to_string();
                }
                return;
            }
        };

        let resp_value = match JsFuture::from(window.fetch_with_request(&request)).await {
            Ok(v) => v,
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Network query failed".to_string();
                }
                return;
            }
        };

        let resp: web_sys::Response = resp_value.dyn_into().unwrap();
        if !resp.ok() {
            if let Ok(mut st) = status.lock() {
                *st = "HTTP error response".to_string();
            }
            return;
        }

        let json_value = match JsFuture::from(resp.json().unwrap()).await {
            Ok(v) => v,
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Failed to read JSON".to_string();
                }
                return;
            }
        };

        let text: String = match js_sys::JSON::stringify(&json_value) {
            Ok(s) => s.into(),
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Failed to stringify JSON".to_string();
                }
                return;
            }
        };

        let hostname = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
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
        };

        let found_host = hostname.unwrap_or(query);

        {
            if let Ok(mut st) = status.lock() {
                *st = format!("Fetching data for {}...", found_host);
            }
        }

        let url2 = format!(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbsmax,pl_orbeccen,pl_masse,pl_rade,st_mass,st_rad+from+pscomppars+where+hostname='{}'&format=json",
            found_host.replace(" ", "+")
        );

        log::info!("Web fetching planet data from Caltech: {}", url2);

        let request2 = match web_sys::Request::new_with_str_and_init(&url2, &opts) {
            Ok(r) => r,
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Failed to construct request 2".to_string();
                }
                return;
            }
        };

        let resp_value2 = match JsFuture::from(window.fetch_with_request(&request2)).await {
            Ok(v) => v,
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Network query 2 failed".to_string();
                }
                return;
            }
        };

        let resp2: web_sys::Response = resp_value2.dyn_into().unwrap();
        if !resp2.ok() {
            if let Ok(mut st) = status.lock() {
                *st = "HTTP error response 2".to_string();
            }
            return;
        }

        let json_value2 = match JsFuture::from(resp2.json().unwrap()).await {
            Ok(v) => v,
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Failed to read JSON 2".to_string();
                }
                return;
            }
        };

        let text2: String = match js_sys::JSON::stringify(&json_value2) {
            Ok(s) => s.into(),
            Err(_) => {
                if let Ok(mut st) = status.lock() {
                    *st = "Failed to stringify JSON 2".to_string();
                }
                return;
            }
        };

        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text2) {
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

                bodies.push((
                    found_host.clone(),
                    Vector3::zeros(),
                    Vector3::zeros(),
                    star_mass,
                    star_radius,
                    100u32,
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
                        101u32,
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
    });
}

#[cfg(target_arch = "wasm32")]
fn map_winit_key_to_imgui(keycode: KeyCode) -> Option<imgui::Key> {
    match keycode {
        KeyCode::ArrowUp => Some(imgui::Key::UpArrow),
        KeyCode::ArrowDown => Some(imgui::Key::DownArrow),
        KeyCode::ArrowLeft => Some(imgui::Key::LeftArrow),
        KeyCode::ArrowRight => Some(imgui::Key::RightArrow),
        KeyCode::Enter => Some(imgui::Key::Enter),
        KeyCode::Space => Some(imgui::Key::Space),
        KeyCode::Backspace => Some(imgui::Key::Backspace),
        KeyCode::Delete => Some(imgui::Key::Delete),
        KeyCode::Escape => Some(imgui::Key::Escape),
        KeyCode::Tab => Some(imgui::Key::Tab),
        
        KeyCode::KeyA => Some(imgui::Key::A),
        KeyCode::KeyB => Some(imgui::Key::B),
        KeyCode::KeyC => Some(imgui::Key::C),
        KeyCode::KeyD => Some(imgui::Key::D),
        KeyCode::KeyE => Some(imgui::Key::E),
        KeyCode::KeyF => Some(imgui::Key::F),
        KeyCode::KeyG => Some(imgui::Key::G),
        KeyCode::KeyH => Some(imgui::Key::H),
        KeyCode::KeyI => Some(imgui::Key::I),
        KeyCode::KeyJ => Some(imgui::Key::J),
        KeyCode::KeyK => Some(imgui::Key::K),
        KeyCode::KeyL => Some(imgui::Key::L),
        KeyCode::KeyM => Some(imgui::Key::M),
        KeyCode::KeyN => Some(imgui::Key::N),
        KeyCode::KeyO => Some(imgui::Key::O),
        KeyCode::KeyP => Some(imgui::Key::P),
        KeyCode::KeyQ => Some(imgui::Key::Q),
        KeyCode::KeyR => Some(imgui::Key::R),
        KeyCode::KeyS => Some(imgui::Key::S),
        KeyCode::KeyT => Some(imgui::Key::T),
        KeyCode::KeyU => Some(imgui::Key::U),
        KeyCode::KeyV => Some(imgui::Key::V),
        KeyCode::KeyW => Some(imgui::Key::W),
        KeyCode::KeyX => Some(imgui::Key::X),
        KeyCode::KeyY => Some(imgui::Key::Y),
        KeyCode::KeyZ => Some(imgui::Key::Z),
        
        KeyCode::Digit0 => Some(imgui::Key::Alpha0),
        KeyCode::Digit1 => Some(imgui::Key::Alpha1),
        KeyCode::Digit2 => Some(imgui::Key::Alpha2),
        KeyCode::Digit3 => Some(imgui::Key::Alpha3),
        KeyCode::Digit4 => Some(imgui::Key::Alpha4),
        KeyCode::Digit5 => Some(imgui::Key::Alpha5),
        KeyCode::Digit6 => Some(imgui::Key::Alpha6),
        KeyCode::Digit7 => Some(imgui::Key::Alpha7),
        KeyCode::Digit8 => Some(imgui::Key::Alpha8),
        KeyCode::Digit9 => Some(imgui::Key::Alpha9),

        KeyCode::ShiftLeft => Some(imgui::Key::LeftShift),
        KeyCode::ShiftRight => Some(imgui::Key::RightShift),
        KeyCode::ControlLeft => Some(imgui::Key::LeftCtrl),
        KeyCode::ControlRight => Some(imgui::Key::RightCtrl),
        KeyCode::AltLeft => Some(imgui::Key::LeftAlt),
        KeyCode::AltRight => Some(imgui::Key::RightAlt),
        KeyCode::SuperLeft => Some(imgui::Key::LeftSuper),
        KeyCode::SuperRight => Some(imgui::Key::RightSuper),
        KeyCode::Minus => Some(imgui::Key::Minus),
        KeyCode::Equal => Some(imgui::Key::Equal),
        KeyCode::BracketLeft => Some(imgui::Key::LeftBracket),
        KeyCode::BracketRight => Some(imgui::Key::RightBracket),
        KeyCode::Semicolon => Some(imgui::Key::Semicolon),
        KeyCode::Quote => Some(imgui::Key::Apostrophe),
        KeyCode::Comma => Some(imgui::Key::Comma),
        KeyCode::Period => Some(imgui::Key::Period),
        KeyCode::Slash => Some(imgui::Key::Slash),
        KeyCode::Backslash => Some(imgui::Key::Backslash),
        KeyCode::Backquote => Some(imgui::Key::GraveAccent),

        _ => None,
    }
}




