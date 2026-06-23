use anyhow::{Result, anyhow};
use std::sync::Arc;
use log::info;
use nalgebra::Vector3;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C, align(256))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstants {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub body_type: u32,
    pub is_selected: u32,
    pub _padding: [u32; 42], // Pad to 256 bytes for hardware alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BodyUbo {
    pub pos_mass: [f32; 4], // xyz = position, w = visual warping strength
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub bodies: [BodyUbo; 10],
    pub num_bodies: i32,
    pub time: f32,
    pub _padding: [f32; 2],
}

pub struct OrbitParams {
    pub a: f32,
    pub e: f32,
}

// WGSL Shaders compiled into a single clean string
const SHADER_SRC: &str = r#"
struct Uniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    bodies_pos_mass: array<vec4<f32>, 10>,
    num_bodies: i32,
    time: f32,
}

struct PushConstants {
    model: mat4x4<f32>,
    color: vec4<f32>,
    body_type: u32,
    is_selected: u32,
}

@group(0) @binding(0) var<uniform> ubo: Uniforms;
@group(0) @binding(1) var<uniform> push: PushConstants;

// ==================== SKYBOX ====================
struct SkyboxOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
}

@vertex
fn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxOutput {
    var ndc = vec2<f32>(-1.0, -1.0);
    if (vertex_index == 1u) {
        ndc = vec2<f32>(3.0, -1.0);
    } else if (vertex_index == 2u) {
        ndc = vec2<f32>(-1.0, 3.0);
    }
    var out: SkyboxOutput;
    out.position = vec4<f32>(ndc, 0.9999, 1.0);
    let target_pos = ubo.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    out.view_dir = target_pos.xyz / target_pos.w;
    return out;
}

fn hash(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.3183099 + vec3<f32>(0.1));
    p = p * 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

fn noise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    var f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    let h000 = hash(i + vec3<f32>(0.0, 0.0, 0.0));
    let h100 = hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let h010 = hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let h110 = hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let h001 = hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let h101 = hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let h011 = hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let h111 = hash(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(
        mix(mix(h000, h100, f.x), mix(h010, h110, f.x), f.y),
        mix(mix(h001, h101, f.x), mix(h011, h111, f.x), f.y),
        f.z
    );
}

fn fbm(p_in: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var p = p_in;
    let shift = vec3<f32>(100.0);
    for (var i = 0; i < 4; i = i + 1) {
        v = v + a * noise(p);
        p = p * 2.0 + shift;
        a = a * 0.5;
    }
    return v;
}

@fragment
fn fs_skybox(in: SkyboxOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.view_dir);
    let star_grid = dir * 250.0;
    let star_strength = hash(floor(star_grid));
    var star_rgb = vec3<f32>(0.0);
    if (star_strength > 0.994) {
        let f = hash(floor(star_grid) + vec3<f32>(0.5));
        let star_intensity = pow(f, 15.0) * 1.5;
        let col_hash = hash(floor(star_grid) + vec3<f32>(0.2));
        var star_col = vec3<f32>(1.0);
        if (col_hash < 0.3) {
            star_col = vec3<f32>(0.8, 0.9, 1.0);
        } else if (col_hash > 0.75) {
            star_col = vec3<f32>(1.0, 0.9, 0.7);
        }
        star_rgb = star_col * star_intensity;
    }
    let n1 = fbm(dir * 2.5 + vec3<f32>(1.2));
    let n2 = fbm(dir * 3.5 - vec3<f32>(5.7));
    let nebula1 = vec3<f32>(0.04, 0.015, 0.08) * n1;
    let nebula2 = vec3<f32>(0.01, 0.03, 0.06) * n2;
    let milky_way_band = smoothstep(0.45, 0.0, abs(dir.y + 0.4 * dir.x - 0.2 * dir.z));
    let mw_noise = fbm(dir * 6.0);
    let milky_way = vec3<f32>(0.12, 0.08, 0.15) * milky_way_band * (mw_noise + 0.3);
    let core_glow = smoothstep(0.8, 0.0, distance(dir, vec3<f32>(0.6, -0.2, -0.7)));
    let core = vec3<f32>(0.25, 0.15, 0.1) * core_glow * (fbm(dir * 8.0) * 0.7 + 0.3);
    var final_color = star_rgb + nebula1 + nebula2 + milky_way + core;
    final_color = 1.0 - exp(-final_color * 1.5);
    return vec4<f32>(final_color, 1.0);
}

// ==================== GRID ====================
struct GridOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) height: f32,
}

@vertex
fn vs_grid(@location(0) in_pos: vec2<f32>) -> GridOutput {
    var depth = 0.0;
    for (var i = 0; i < ubo.num_bodies; i = i + 1) {
        let body_pos = ubo.bodies_pos_mass[i].xyz;
        let strength = ubo.bodies_pos_mass[i].w;
        let d = distance(vec3<f32>(in_pos.x, 0.0, in_pos.y), vec3<f32>(body_pos.x, 0.0, body_pos.z));
        depth = depth - (strength * 2.5) / (d + 0.45);
    }
    let pos = vec3<f32>(in_pos.x, depth, in_pos.y);
    var out: GridOutput;
    out.position = ubo.proj * ubo.view * vec4<f32>(pos, 1.0);
    out.uv = in_pos * 0.1;
    out.height = depth;
    return out;
}

@fragment
fn fs_grid(in: GridOutput) -> @location(0) vec4<f32> {
    let depth = clamp(-in.height * 0.4, 0.0, 1.0);
    let base_color = vec3<f32>(0.0, 0.9, 0.7);
    let deep_color = vec3<f32>(0.1, 0.3, 1.0);
    let grid_color = mix(base_color, deep_color, depth);
    let alpha = mix(0.5, 0.95, depth);
    return vec4<f32>(grid_color, alpha);
}

// ==================== ORBITS ====================
struct OrbitOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) height: f32,
}

@vertex
fn vs_orbit(@location(0) in_pos: vec2<f32>) -> OrbitOutput {
    var depth = 0.0;
    for (var i = 0; i < ubo.num_bodies; i = i + 1) {
        let body_pos = ubo.bodies_pos_mass[i].xyz;
        let strength = ubo.bodies_pos_mass[i].w;
        let d = distance(vec3<f32>(in_pos.x, 0.0, in_pos.y), vec3<f32>(body_pos.x, 0.0, body_pos.z));
        depth = depth - (strength * 2.5) / (d + 0.45);
    }
    let pos = vec3<f32>(in_pos.x, depth, in_pos.y);
    var out: OrbitOutput;
    out.position = ubo.proj * ubo.view * vec4<f32>(pos, 1.0);
    out.height = depth;
    return out;
}

@fragment
fn fs_orbit(in: OrbitOutput) -> @location(0) vec4<f32> {
    return push.color;
}

// ==================== SPHERES ====================
struct SphereOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) body_type: u32,
    @location(4) @interpolate(flat) is_selected: u32,
    @location(5) view_dir: vec3<f32>,
}

@vertex
fn vs_sphere(
    @location(0) in_pos: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>
) -> SphereOutput {
    let world_pos = push.model * vec4<f32>(in_pos, 1.0);
    var out: SphereOutput;
    out.position = ubo.proj * ubo.view * world_pos;
    out.world_pos = world_pos.xyz;
    out.normal = normalize((push.model * vec4<f32>(in_normal, 0.0)).xyz);
    out.uv = in_uv;
    out.body_type = push.body_type;
    out.is_selected = push.is_selected;
    let cam_pos4 = ubo.inv_view_proj * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let cam_pos = cam_pos4.xyz / cam_pos4.w;
    out.view_dir = cam_pos - world_pos.xyz;
    return out;
}

@fragment
fn fs_sphere(in: SphereOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let L = normalize(vec3<f32>(1.0, 1.5, 1.0));
    let diff = max(dot(N, L), 0.0);
    let ambient = 0.12;
    var albedo = vec3<f32>(0.8);
    var glow = 0.0;
    var alpha = 1.0;
    let b_type = in.body_type;
    if (b_type == 0u) {
        let n = fbm(N * 8.0);
        albedo = mix(vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(1.0, 0.9, 0.1), n);
        glow = 1.0;
    } else if (b_type == 1u) {
        let n = fbm(N * 16.0);
        albedo = vec3<f32>(0.5 + 0.2 * n);
    } else if (b_type == 2u) {
        let n = fbm(N * 6.0);
        albedo = mix(vec3<f32>(0.85, 0.7, 0.45), vec3<f32>(0.95, 0.85, 0.6), n);
    } else if (b_type == 3u) {
        let n = fbm(N * 10.0);
        let clouds = fbm(N * 14.0 + vec3<f32>(1.2, 0.0, 0.5));
        if (n > 0.46) {
            albedo = mix(vec3<f32>(0.2, 0.5, 0.25), vec3<f32>(0.4, 0.35, 0.25), (n - 0.46) * 4.0);
        } else {
            albedo = vec3<f32>(0.08, 0.25, 0.65);
        }
        if (clouds > 0.55) {
            albedo = mix(albedo, vec3<f32>(0.95), (clouds - 0.55) * 2.0);
        }
    } else if (b_type == 4u) {
        let n = fbm(N * 12.0);
        albedo = mix(vec3<f32>(0.68, 0.28, 0.15), vec3<f32>(0.8, 0.45, 0.25), n);
        if (abs(N.y) > 0.88) {
            let cap_noise = fbm(N * 20.0);
            if (abs(N.y) + 0.05 * cap_noise > 0.90) {
                albedo = vec3<f32>(0.95);
            }
        }
    } else if (b_type == 5u) {
        let lat = N.y;
        let band = sin(lat * 35.0 + fbm(N * 3.0) * 2.5);
        let n = fbm(N * 12.0);
        let col1 = vec3<f32>(0.75, 0.6, 0.48);
        let col2 = vec3<f32>(0.48, 0.32, 0.22);
        albedo = mix(col1, col2, (band + 1.0) * 0.5);
        albedo = albedo + vec3<f32>(0.05 * n);
        let spot_pos = normalize(vec3<f32>(0.8, -0.34, 0.48));
        let dist_spot = distance(N, spot_pos);
        if (dist_spot < 0.14) {
            albedo = mix(vec3<f32>(0.55, 0.15, 0.08), albedo, smoothstep(0.08, 0.14, dist_spot));
        }
    } else if (b_type == 6u) {
        let lat = N.y;
        let band = sin(lat * 15.0 + fbm(N * 2.0) * 0.5);
        let col1 = vec3<f32>(0.88, 0.82, 0.65);
        let col2 = vec3<f32>(0.72, 0.64, 0.48);
        albedo = mix(col1, col2, (band + 1.0) * 0.5);
    } else if (b_type == 7u) {
        let lat = N.y;
        let band = sin(lat * 5.0);
        let col1 = vec3<f32>(0.65, 0.88, 0.88);
        let col2 = vec3<f32>(0.55, 0.8, 0.85);
        albedo = mix(col1, col2, (band + 1.0) * 0.5);
    } else if (b_type == 8u) {
        let lat = N.y;
        let band = sin(lat * 8.0 + fbm(N * 2.0) * 0.5);
        let col1 = vec3<f32>(0.2, 0.4, 0.9);
        let col2 = vec3<f32>(0.1, 0.25, 0.7);
        albedo = mix(col1, col2, (band + 1.0) * 0.5);
        let spot_pos = normalize(vec3<f32>(-0.6, -0.3, -0.6));
        let dist_spot = distance(N, spot_pos);
        if (dist_spot < 0.18) {
            albedo = mix(vec3<f32>(0.05, 0.1, 0.4), albedo, smoothstep(0.1, 0.18, dist_spot));
        }
    } else if (b_type == 9u) {
        let r = in.uv.x;
        let band = sin(r * 120.0) * 0.5 + 0.5;
        let gap1 = smoothstep(0.48, 0.52, r) * (1.0 - smoothstep(0.53, 0.57, r));
        let opacity = (0.3 + 0.55 * band) * (1.0 - gap1);
        albedo = mix(vec3<f32>(0.65, 0.58, 0.5), vec3<f32>(0.85, 0.8, 0.72), band);
        alpha = opacity;
    } else if (b_type == 100u) {
        let n = fbm(N * 12.0);
        albedo = mix(vec3<f32>(0.2, 0.55, 1.0), vec3<f32>(0.4, 0.75, 1.0), n);
        glow = 1.0;
    } else {
        let lat = N.y;
        let band = sin(lat * 25.0 + fbm(N * 4.0));
        let col1 = vec3<f32>(0.3, 0.45, 0.6);
        let col2 = vec3<f32>(0.15, 0.2, 0.35);
        albedo = mix(col1, col2, (band + 1.0) * 0.5);
    }
    var color = vec3<f32>(0.0);
    if (glow > 0.5) {
        color = albedo;
    } else {
        color = albedo * (diff + ambient);
    }
    if (in.is_selected == 1u) {
        let V = normalize(in.view_dir);
        let rim = 1.0 - max(dot(N, V), 0.0);
        let rim_glow = pow(rim, 4.0) * 1.5;
        color = color + vec3<f32>(0.0, 0.8, 1.0) * rim_glow;
    }
    return vec4<f32>(color, alpha);
}
"#;

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    // Rendering pipelines
    skybox_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    orbit_pipeline: wgpu::RenderPipeline,
    sphere_pipeline: wgpu::RenderPipeline,
    
    // Static meshes
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
    
    sphere_vertex_buffer: wgpu::Buffer,
    sphere_index_buffer: wgpu::Buffer,
    sphere_index_count: u32,
    
    ring_vertex_buffer: wgpu::Buffer,
    ring_index_buffer: wgpu::Buffer,
    ring_index_count: u32,
    
    // Dynamic trail/orbit buffer
    orbit_vertex_buffer: wgpu::Buffer,
    orbit_vertex_capacity: usize,
    
    // Uniform buffers
    ubo_buffer: wgpu::Buffer,
    push_buffer: wgpu::Buffer,
    
    // Bind group & layout
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    
    // Depth stencil view
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    
    // ImGui renderer
    pub imgui_renderer: super::imgui_renderer::CustomImguiRenderer,
}

impl Renderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface: wgpu::Surface<'static>,
        surface_config: wgpu::SurfaceConfiguration,
        imgui_context: &mut imgui::Context,
    ) -> Result<Self> {
        info!("Initializing cross-platform wgpu Renderer...");
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simulation Shaders"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        // 1. Create Bind Group Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Renderer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true, // Use dynamic offset for instanced push constants!
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // 2. Create Render Pipelines
        // Skybox Pipeline (triangles, no vertex buffers, no depth writes)
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_skybox"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_skybox"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Grid Pipeline (line list, 2D float vertices)
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_grid"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_grid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Orbit Pipeline (line list, 2D float vertices)
        let orbit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Orbit Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_orbit"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_orbit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Sphere & Ring Pipeline (triangle list)
        let sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sphere Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sphere"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SphereVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sphere"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // 3. Generate Static Meshes
        let grid_verts = generate_grid_vertices(40.0, 100, 140);
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let grid_vertex_count = grid_verts.len() as u32;

        let (sphere_verts, sphere_indices) = generate_sphere(24, 24);
        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(&sphere_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Index Buffer"),
            contents: bytemuck::cast_slice(&sphere_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let sphere_index_count = sphere_indices.len() as u32;

        let (ring_verts, ring_indices) = generate_ring(1.4, 2.3, 32);
        let ring_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Saturn Ring Vertex Buffer"),
            contents: bytemuck::cast_slice(&ring_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ring_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Saturn Ring Index Buffer"),
            contents: bytemuck::cast_slice(&ring_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let ring_index_count = ring_indices.len() as u32;

        // 4. Setup Orbit Dynamic Buffer
        let orbit_vertex_capacity = 32768; // capacity in vertices
        let orbit_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Orbits Vertex Buffer"),
            size: (orbit_vertex_capacity * 8) as wgpu::BufferAddress, // vec2 (8 bytes)
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 5. Setup Uniform buffers
        let ubo_size = std::mem::size_of::<UniformBufferObject>() as wgpu::BufferAddress;
        let ubo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global UBO Buffer"),
            size: ubo_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Buffer allocated for dynamic offset push constants:
        // We support max 30 draw calls per frame, each gets 256 bytes slot
        let push_buffer_size = (30 * 256) as wgpu::BufferAddress;
        let push_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Push Constants Uniform Buffer (Dynamic Offsets)"),
            size: push_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 6. Create Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &push_buffer,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(256).unwrap()),
                    }),
                },
            ],
        });

        // Depth Stencil texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width.max(1),
                height: surface_config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 7. Setup ImGui Renderer Config
        let imgui_renderer = super::imgui_renderer::CustomImguiRenderer::new(
            imgui_context,
            &device,
            &queue,
            surface_config.format,
            Some(wgpu::TextureFormat::Depth32Float),
        );

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            skybox_pipeline,
            grid_pipeline,
            orbit_pipeline,
            sphere_pipeline,
            grid_vertex_buffer,
            grid_vertex_count,
            sphere_vertex_buffer,
            sphere_index_buffer,
            sphere_index_count,
            ring_vertex_buffer,
            ring_index_buffer,
            ring_index_count,
            orbit_vertex_buffer,
            orbit_vertex_capacity,
            ubo_buffer,
            push_buffer,
            bind_group_layout,
            bind_group,
            depth_texture,
            depth_view,
            imgui_renderer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
            
            // Recreate Depth Stencil texture
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    pub fn draw_frame(
        &mut self,
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        inv_view_proj: [[f32; 4]; 4],
        bodies_pos_mass: &[BodyUbo],
        body_radii: &[f32],
        body_types: &[u32],
        body_colors: &[[f32; 4]],
        selected_body_idx: usize,
        hovered_body_idx: Option<usize>,
        camera_pos: [f32; 3],
        trails: &[Vec<[f32; 2]>],
        imgui_draw_data: &imgui::DrawData,
    ) -> Result<()> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return Err(anyhow!("Failed to acquire next surface texture")),
        };
        let view_target = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // 1. Update Global Uniform Buffer Object
        let mut bodies = [BodyUbo { pos_mass: [0.0; 4] }; 10];
        let num_bodies = bodies_pos_mass.len().min(10);
        for i in 0..num_bodies {
            bodies[i] = bodies_pos_mass[i];
        }

        let ubo = UniformBufferObject {
            model: nalgebra::Matrix4::identity().into(),
            view,
            proj,
            inv_view_proj,
            bodies,
            num_bodies: num_bodies as i32,
            time: 0.0,
            _padding: [0.0; 2],
        };
        self.queue.write_buffer(&self.ubo_buffer, 0, bytemuck::bytes_of(&ubo));

        // 2. Prepare dynamic push constants buffer
        // We write slots for grid, orbits, and planets
        let mut push_slots = Vec::new();
        
        // Slot 0: Grid (Identity matrix)
        let identity = nalgebra::Matrix4::identity().into();
        push_slots.push(PushConstants {
            model: identity,
            color: [1.0; 4],
            body_type: 0,
            is_selected: 0,
            _padding: [0; 42],
        });

        // Slots for Orbits
        let orbit_slot_start = push_slots.len();
        for i in 0..num_bodies {
            let color = body_colors.get(i).copied().unwrap_or([0.8, 0.8, 0.8, 0.8]);
            push_slots.push(PushConstants {
                model: identity,
                color,
                body_type: 0,
                is_selected: 0,
                _padding: [0; 42],
            });
        }

        // Slots for Spheres & Rings
        let sphere_slot_start = push_slots.len();
        for i in 0..num_bodies {
            let p = bodies[i].pos_mass;
            let radius = body_radii[i];
            
            let dx = p[0] - camera_pos[0];
            let dy = p[1] - camera_pos[1];
            let dz = p[2] - camera_pos[2];
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            
            let b_type = body_types[i];
            let min_size_factor = if b_type == 0 || b_type == 100 { 0.006 } else { 0.0025 };
            let visual_radius = radius.max(dist * min_size_factor);
            
            let scale = nalgebra::Matrix4::new_scaling(visual_radius);
            let translation = nalgebra::Matrix4::new_translation(&Vector3::new(p[0], p[1], p[2]));
            let model = translation * scale;
            
            let is_selected = if i == selected_body_idx || Some(i) == hovered_body_idx { 1u32 } else { 0u32 };

            push_slots.push(PushConstants {
                model: model.into(),
                color: [1.0; 4],
                body_type: b_type,
                is_selected,
                _padding: [0; 42],
            });
        }

        // Slot for Saturn rings (if present, Saturn is index 6/type 6 usually)
        let mut has_saturn_rings = false;
        let mut saturn_model = identity;
        for i in 0..num_bodies {
            if body_types[i] == 6 {
                has_saturn_rings = true;
                let p = bodies[i].pos_mass;
                let radius = body_radii[i];
                
                let dx = p[0] - camera_pos[0];
                let dy = p[1] - camera_pos[1];
                let dz = p[2] - camera_pos[2];
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                
                let b_type = body_types[i];
                let min_size_factor = if b_type == 0 || b_type == 100 { 0.006 } else { 0.0025 };
                let visual_radius = radius.max(dist * min_size_factor);
                
                let scale = nalgebra::Matrix4::new_scaling(visual_radius);
                let translation = nalgebra::Matrix4::new_translation(&Vector3::new(p[0], p[1], p[2]));
                saturn_model = (translation * scale).into();
                break;
            }
        }
        
        let ring_slot_index = push_slots.len();
        if has_saturn_rings {
            push_slots.push(PushConstants {
                model: saturn_model,
                color: [1.0; 4],
                body_type: 9, // Ring type
                is_selected: 0,
                _padding: [0; 42],
            });
        }

        // Upload the entire push buffer at once!
        self.queue.write_buffer(&self.push_buffer, 0, bytemuck::cast_slice(&push_slots));

        // 3. Assemble dynamic orbit trails into orbit vertex buffer
        let mut flat_orbit_vertices = Vec::new();
        let mut orbit_draw_calls = Vec::new(); // Vec of (start_vertex, count) for each planet orbit
        
        for i in 0..num_bodies {
            let trail = &trails[i];
            let start = flat_orbit_vertices.len() as u32;
            let mut count = 0;
            if trail.len() >= 2 {
                for j in 0..(trail.len() - 1) {
                    flat_orbit_vertices.push(trail[j]);
                    flat_orbit_vertices.push(trail[j + 1]);
                    count += 2;
                }
            }
            orbit_draw_calls.push((start, count));
        }

        if !flat_orbit_vertices.is_empty() {
            let upload_len = flat_orbit_vertices.len().min(self.orbit_vertex_capacity);
            self.queue.write_buffer(
                &self.orbit_vertex_buffer,
                0,
                bytemuck::cast_slice(&flat_orbit_vertices[0..upload_len]),
            );
        }

        // 4. Render Pass Execution
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Simulation Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.03,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // A. Draw Skybox
            rpass.set_pipeline(&self.skybox_pipeline);
            // Dynamic offset: 0 (Grid slot padding used as dummy)
            rpass.set_bind_group(0, &self.bind_group, &[0]);
            rpass.draw(0..3, 0..1);

            // B. Draw Grid
            rpass.set_pipeline(&self.grid_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[0]); // Grid slot: 0
            rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            rpass.draw(0..self.grid_vertex_count, 0..1);

            // C. Draw Orbits/Trails
            rpass.set_pipeline(&self.orbit_pipeline);
            rpass.set_vertex_buffer(0, self.orbit_vertex_buffer.slice(..));
            for i in 0..num_bodies {
                let (start, count) = orbit_draw_calls[i];
                if count > 0 {
                    let offset = ((orbit_slot_start + i) * 256) as wgpu::DynamicOffset;
                    rpass.set_bind_group(0, &self.bind_group, &[offset]);
                    rpass.draw(start..(start + count), 0..1);
                }
            }

            // D. Draw Spheres
            rpass.set_pipeline(&self.sphere_pipeline);
            rpass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
            rpass.set_index_buffer(self.sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..num_bodies {
                let offset = ((sphere_slot_start + i) * 256) as wgpu::DynamicOffset;
                rpass.set_bind_group(0, &self.bind_group, &[offset]);
                rpass.draw_indexed(0..self.sphere_index_count, 0, 0..1);
            }

            // E. Draw Saturn Rings
            if has_saturn_rings {
                let offset = (ring_slot_index * 256) as wgpu::DynamicOffset;
                rpass.set_bind_group(0, &self.bind_group, &[offset]);
                rpass.set_vertex_buffer(0, self.ring_vertex_buffer.slice(..));
                rpass.set_index_buffer(self.ring_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..self.ring_index_count, 0, 0..1);
            }

            // F. Draw ImGui
            self.imgui_renderer
                .render(imgui_draw_data, &self.queue, &self.device, &mut rpass)
                .map_err(|e| anyhow!("Failed to draw ImGui command list: {:?}", e))?;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }
}

// ==================== MESH GENERATORS ====================

use wgpu::util::DeviceExt;

fn generate_grid_vertices(grid_size: f32, grid_lines: u32, segments: u32) -> Vec<[f32; 2]> {
    let mut vertices = Vec::new();
    let step = (grid_size * 2.0) / (grid_lines as f32);
    for i in 0..=grid_lines {
        let y = -grid_size + (i as f32) * step;
        let segment_step = (grid_size * 2.0) / (segments as f32);
        for j in 0..segments {
            let x1 = -grid_size + (j as f32) * segment_step;
            let x2 = x1 + segment_step;
            vertices.push([x1, y]);
            vertices.push([x2, y]);
        }
    }
    for i in 0..=grid_lines {
        let x = -grid_size + (i as f32) * step;
        let segment_step = (grid_size * 2.0) / (segments as f32);
        for j in 0..segments {
            let y1 = -grid_size + (j as f32) * segment_step;
            let y2 = y1 + segment_step;
            vertices.push([x, y1]);
            vertices.push([x, y2]);
        }
    }
    vertices
}

fn generate_sphere(lat_segments: u32, lon_segments: u32) -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for y in 0..=lat_segments {
        let y_f = y as f32 / lat_segments as f32;
        let theta = y_f * std::f32::consts::PI;
        for x in 0..=lon_segments {
            let x_f = x as f32 / lon_segments as f32;
            let phi = x_f * 2.0 * std::f32::consts::PI;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            let px = sin_theta * cos_phi;
            let py = cos_theta;
            let pz = sin_theta * sin_phi;
            vertices.push(SphereVertex {
                pos: [px, py, pz],
                normal: [px, py, pz],
                uv: [x_f, y_f],
            });
        }
    }
    for y in 0..lat_segments {
        for x in 0..lon_segments {
            let i0 = y * (lon_segments + 1) + x;
            let i1 = i0 + 1;
            let i2 = (y + 1) * (lon_segments + 1) + x;
            let i3 = i2 + 1;
            indices.push(i0);
            indices.push(i2);
            indices.push(i1);
            indices.push(i1);
            indices.push(i2);
            indices.push(i3);
        }
    }
    (vertices, indices)
}

fn generate_ring(inner_radius: f32, outer_radius: f32, segments: u32) -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=segments {
        let angle = i as f32 / segments as f32 * 2.0 * std::f32::consts::PI;
        let c = angle.cos();
        let s = angle.sin();
        vertices.push(SphereVertex {
            pos: [c * inner_radius, 0.0, s * inner_radius],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, i as f32 / segments as f32],
        });
        vertices.push(SphereVertex {
            pos: [c * outer_radius, 0.0, s * outer_radius],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, i as f32 / segments as f32],
        });
    }
    for i in 0..segments {
        let i0 = i * 2;
        let i1 = i0 + 1;
        let i2 = i0 + 2;
        let i3 = i0 + 3;
        indices.push(i0);
        indices.push(i1);
        indices.push(i2);
        indices.push(i2);
        indices.push(i1);
        indices.push(i3);
    }
    (vertices, indices)
}
