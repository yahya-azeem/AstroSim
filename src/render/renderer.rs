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
    pub star_radius: f32,
    pub _padding: f32,
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
    star_radius: f32,
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
    
    // Deep space: faint, atmospheric background nebulae and stars
    let n1 = fbm(dir * 2.5 + vec3<f32>(1.2));
    let nebula1 = vec3<f32>(0.005, 0.002, 0.01) * n1; // Faint dark cosmic dust
    
    // Calculate God Rays from the Sun (at origin/bodies_pos_mass[0])
    let R = mat3x3<f32>(ubo.view[0].xyz, ubo.view[1].xyz, ubo.view[2].xyz);
    let cam_pos = transpose(R) * -ubo.view[3].xyz;
    let sun_dir = normalize(ubo.bodies_pos_mass[0].xyz - cam_pos);
    let cos_theta = dot(dir, sun_dir);
    let d = length(ubo.bodies_pos_mass[0].xyz - cam_pos);
    
    var god_rays = vec3<f32>(0.0);
    if (cos_theta > 0.0) {
        let proj_dir = dir - sun_dir * cos_theta;
        let proj_len = length(proj_dir);
        let radial_dir = proj_dir / (proj_len + 1e-5);
        
        let time_scale = ubo.time * 0.18;
        let ray_noise1 = fbm(radial_dir * 7.0 + vec3<f32>(time_scale, 0.0, -time_scale * 0.5));
        let ray_noise2 = fbm(radial_dir * 15.0 - vec3<f32>(time_scale * 0.7, time_scale * 0.3, 0.0));
        var ray_noise = mix(ray_noise1, ray_noise2, 0.35);
        ray_noise = pow(ray_noise, 2.2) * 2.5; // High contrast sharp rays!
        
        // Compute angular radius of the star safely
        let d_safe = max(d, 1e-6);
        let theta_star = asin(clamp(ubo.star_radius / d_safe, 0.0, 1.0));
        let angle_from_center = acos(clamp(cos_theta, -1.0, 1.0));
        
        // Angular distance from the edge of the star's visual disk
        let delta_theta = max(0.0, angle_from_center - theta_star);
        
        // Halo and rays decay based on angular distance from the star's edge
        let inner_halo = exp(-delta_theta * 35.0) * 1.5; 
        let outer_halo = exp(-delta_theta * 8.0) * 0.35; 
        
        // Scale intensity smoothly with the visual size of the star relative to its Earth-baseline size (0.163 rad)
        let size_ratio = theta_star / 0.163;
        let intensity_factor = clamp(sqrt(size_ratio), 0.01, 1.2);
        
        let ray_col = vec3<f32>(1.0, 0.72, 0.4); // Golden solar light
        god_rays = ray_col * (inner_halo * (ray_noise * 1.2 + 0.3) + outer_halo * (ray_noise * 1.0 + 0.15)) * intensity_factor;
    }
    
    var final_color = star_rgb + nebula1 + god_rays;
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
    let base_color = vec3<f32>(0.0, 0.4, 0.3);
    let deep_color = vec3<f32>(0.05, 0.12, 0.45);
    let grid_color = mix(base_color, deep_color, depth);
    let alpha = mix(0.12, 0.35, depth);
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
    return vec4<f32>(push.color.rgb, push.color.a * 0.35);
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

const EARTH_MAP: array<u32, 256> = array<u32, 256>(
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00ff01f8u, 0x00000000u, 0x00000000u,
    0xc0000000u, 0x01ffff7fu, 0x000003c0u, 0x00000006u,
    0x00600000u, 0x01ffff0eu, 0x00000020u, 0x00000030u,
    0x00000000u, 0x01fff003u, 0x80100000u, 0x0000007fu,
    0x77d80000u, 0x00fff03fu, 0xff0c0000u, 0x001c3fffu,
    0xc93fffe0u, 0x007fc0e5u, 0xfb401fc0u, 0xefffffffu,
    0xffffffefu, 0x0607e3c1u, 0xffffaff0u, 0xffffffffu,
    0xffffffc8u, 0x0001c1c0u, 0xffffff78u, 0xffffffffu,
    0x3ffff7e0u, 0x00010070u, 0xffffff3cu, 0x1c7fffffu,
    0x7fff0300u, 0x400003e0u, 0xfffffe30u, 0x0303ffffu,
    0xfffc0000u, 0xe00007f7u, 0xffffff88u, 0x0380ffffu,
    0xfff80000u, 0xc0000ff7u, 0xffffffffu, 0x0007ffffu,
    0xfff80000u, 0x800018ffu, 0xffffffffu, 0x0007ffffu,
    0xfff00000u, 0x800007ffu, 0xfff9dfffu, 0x0005ffffu,
    0xfff00000u, 0xa000007fu, 0xfffdc3d7u, 0x000cffffu,
    0xfff00000u, 0xe000003fu, 0xfffbffa0u, 0x00003fffu,
    0xfff00000u, 0xe000001fu, 0xfff9ff10u, 0x000233ffu,
    0xffe00000u, 0xc000001fu, 0xffffe00fu, 0x0003a3ffu,
    0xffc00000u, 0xe0000007u, 0xfffff09fu, 0x000047ffu,
    0xbf000000u, 0xf0000004u, 0xfffdffffu, 0x000007ffu,
    0x1e000000u, 0xf8000018u, 0xfffbefffu, 0x000007ffu,
    0x1c000000u, 0xfc000000u, 0xff0fdfffu, 0x000003ffu,
    0x18000000u, 0xfc000011u, 0x7e1fdfffu, 0x0000003eu,
    0xb8000000u, 0xfc000151u, 0x3c0fbfffu, 0x0000003cu,
    0x80000000u, 0xfc000003u, 0x0c03bfffu, 0x00000078u,
    0x00000000u, 0xfc000002u, 0x0800ffffu, 0x00000078u,
    0x00000000u, 0xf80003a4u, 0x0803ffffu, 0x00000020u,
    0x00000000u, 0xf00007f0u, 0x1003ffffu, 0x00001008u,
    0x00000000u, 0x00003ff0u, 0x0001fff8u, 0x00000214u,
    0x00000000u, 0x00003ff0u, 0x0000fff8u, 0x00003388u,
    0x00000000u, 0x0000fff8u, 0x00007ff8u, 0x00008d90u,
    0x00000000u, 0x0007fff8u, 0x00003ff0u, 0x000f0830u,
    0x00000000u, 0x000ffff0u, 0x00003fe0u, 0x000e00c0u,
    0x00000000u, 0x0007fff0u, 0x00003fe0u, 0x00100400u,
    0x00000000u, 0x0003ffe0u, 0x00023fe0u, 0x0005c000u,
    0x00000000u, 0x0003ffe0u, 0x00023ff0u, 0x000cf000u,
    0x00000000u, 0x0003ff80u, 0x00031ff0u, 0x000ff800u,
    0x00000000u, 0x0001ff80u, 0x00010fe0u, 0x001ffe00u,
    0x00000000u, 0x0000ff80u, 0x00011fe0u, 0x003fff00u,
    0x00000000u, 0x00007f80u, 0x00000fe0u, 0x003fff00u,
    0x00000000u, 0x00003f80u, 0x000007c0u, 0x007ffe00u,
    0x00000000u, 0x00001f80u, 0x000003c0u, 0x003f1e00u,
    0x00000000u, 0x00000fc0u, 0x00000000u, 0x003e0000u,
    0x00000000u, 0x00000fc0u, 0x00000000u, 0x40140000u,
    0x00000000u, 0x000003c0u, 0x00000000u, 0x40080000u,
    0x00000000u, 0x000001c0u, 0x00000000u, 0x10000000u,
    0x00000000u, 0x000000e0u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x000000e0u, 0x01000000u, 0x00000000u,
    0x00000000u, 0x000004e0u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000180u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000200u, 0x00080000u, 0x00000110u,
    0x00000000u, 0x00000100u, 0xf1ff9000u, 0x007fffffu,
    0x00000000u, 0xf00003e0u, 0xfeffffffu, 0x1fffffffu,
    0xf1ff8000u, 0xfc0003ffu, 0xffffffffu, 0x07ffffffu,
    0xffffff00u, 0xffe0001fu, 0xffffffffu, 0x03ffffffu,
    0xfffff000u, 0xffc0e41fu, 0xffffffffu, 0x01ffffffu,
    0xfffffc00u, 0xfffff7ffu, 0xffffffffu, 0x07ffffffu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
);

fn get_earth_land(uv: vec2<f32>) -> bool {
    let r = clamp(i32(uv.y * 64.0), 0, 63);
    let u = (uv.x + 0.5) % 1.0;
    let c = clamp(i32((1.0 - u) * 128.0), 0, 127);
    let word_idx = r * 4 + (c / 32);
    let bit_idx = u32(c % 32);
    let word = EARTH_MAP[word_idx];
    return ((word >> bit_idx) & 1u) == 1u;
}

@fragment
fn fs_sphere(in: SphereOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let light_pos = ubo.bodies_pos_mass[0].xyz;
    let L = normalize(light_pos - in.world_pos);
    let diff = max(dot(N, L), 0.0);
    let d_au = distance(light_pos, in.world_pos);
    let intensity = clamp(1.5 / (d_au + 0.5), 0.08, 3.0);
    let ambient = 0.015;
    var albedo = vec3<f32>(0.8);
    var glow = 0.0;
    var alpha = 1.0;
    let b_type = in.body_type;
    if (b_type == 0u) {
        let time_scale = ubo.time * 0.22;
        let n_coord1 = N * 7.5 + vec3<f32>(sin(time_scale * 0.3), time_scale * 0.8, cos(time_scale * 0.2));
        let n_coord2 = N * 18.0 - vec3<f32>(time_scale * 0.5, sin(time_scale * 0.4), time_scale * 0.9);
        let n1 = fbm(n_coord1);
        let n2 = fbm(n_coord2);
        let granulation = mix(n1, n2, 0.35);
        
        // Dynamic hot fiery solar gradient
        albedo = mix(vec3<f32>(0.98, 0.22, 0.01), vec3<f32>(1.0, 0.96, 0.45), granulation);
        glow = 1.0;
        
        let V = normalize(in.view_dir);
        let rim = 1.0 - max(dot(N, V), 0.0);
        
        // Solar flare/corona effect peeking from the rim
        let flare_coord = N * 22.0 + vec3<f32>(time_scale * 1.8, -time_scale * 1.1, time_scale * 1.4);
        let flare_noise = fbm(flare_coord);
        let flare_intensity = pow(rim, 4.0) * flare_noise * 6.5;
        
        albedo = albedo + vec3<f32>(1.0, 0.4, 0.05) * flare_intensity;
    } else if (b_type == 1u) {
        let n = fbm(N * 16.0);
        albedo = vec3<f32>(0.5 + 0.2 * n);
    } else if (b_type == 2u) {
        let n = fbm(N * 6.0);
        albedo = mix(vec3<f32>(0.85, 0.7, 0.45), vec3<f32>(0.95, 0.85, 0.6), n);
    } else if (b_type == 3u) {
        let is_land = get_earth_land(in.uv);
        let clouds = fbm(N * 13.0 + vec3<f32>(ubo.time * 0.02, 0.0, -ubo.time * 0.01));
        
        if (is_land) {
            let lat_factor = abs(N.y);
            let height_noise = fbm(N * 18.0);
            
            var land_color = mix(vec3<f32>(0.15, 0.38, 0.18), vec3<f32>(0.42, 0.34, 0.22), height_noise);
            if (lat_factor > 0.82) {
                land_color = mix(land_color, vec3<f32>(0.95, 0.95, 0.98), smoothstep(0.82, 0.92, lat_factor));
            } else if (lat_factor < 0.35) {
                let desert_noise = fbm(N * 6.0);
                if (desert_noise > 0.42) {
                    land_color = mix(land_color, vec3<f32>(0.58, 0.52, 0.38), (desert_noise - 0.42) * 1.8);
                }
            }
            albedo = land_color;
        } else {
            let shelf = fbm(N * 16.0);
            albedo = mix(vec3<f32>(0.03, 0.12, 0.38), vec3<f32>(0.06, 0.22, 0.52), shelf * 0.45);
        }
        
        if (clouds > 0.52) {
            let cloud_opacity = clamp((clouds - 0.52) * 2.5, 0.0, 0.85);
            albedo = mix(albedo, vec3<f32>(0.92, 0.92, 0.95), cloud_opacity);
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
    } else if (b_type == 11u) {
        let n = fbm(N * 20.0);
        albedo = vec3<f32>(0.4 + 0.15 * n, 0.38 + 0.12 * n, 0.36 + 0.1 * n);
    } else if (b_type == 12u || b_type == 13u || b_type == 14u) {
        if (in.uv.x > 0.05) {
            // Solar panel wings: deep blue with solar grid cells
            let grid_line = sin(in.uv.x * 24.0) * sin(in.uv.y * 8.0);
            let panel_col = mix(vec3<f32>(0.03, 0.08, 0.22), vec3<f32>(0.1, 0.22, 0.55), smoothstep(-0.25, 0.25, grid_line));
            albedo = panel_col;
            glow = 0.25;
        } else {
            // Body: metallic color based on satellite type
            if (b_type == 12u) {
                albedo = vec3<f32>(0.85, 0.85, 0.88); // Silver ISS
                glow = 0.15;
            } else if (b_type == 13u) {
                albedo = vec3<f32>(0.3, 0.75, 1.0); // Cyan Starlink
                glow = 0.5;
            } else {
                albedo = vec3<f32>(1.0, 0.8, 0.2); // Gold GPS
                glow = 0.4;
            }
        }
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
    var spec = 0.0;
    if (b_type == 3u) {
        let n = fbm(N * 10.0);
        if (n <= 0.46) {
            let V = normalize(in.view_dir);
            let H = normalize(L + V);
            spec = pow(max(dot(N, H), 0.0), 64.0) * 0.6;
        }
    } else if (b_type == 12u) {
        let V = normalize(in.view_dir);
        let H = normalize(L + V);
        spec = pow(max(dot(N, H), 0.0), 16.0) * 0.8;
    }

    var color = vec3<f32>(0.0);
    if (glow > 0.5) {
        color = albedo;
    } else {
        color = albedo * (diff * intensity + ambient) + vec3<f32>(spec * intensity);
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
    
    iss_vertex_buffer: wgpu::Buffer,
    iss_index_buffer: wgpu::Buffer,
    iss_index_count: u32,
    
    starlink_vertex_buffer: wgpu::Buffer,
    starlink_index_buffer: wgpu::Buffer,
    starlink_index_count: u32,
    
    gps_vertex_buffer: wgpu::Buffer,
    gps_index_buffer: wgpu::Buffer,
    gps_index_count: u32,
    
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

        let (iss_verts, iss_indices) = generate_iss();
        let iss_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISS Vertex Buffer"),
            contents: bytemuck::cast_slice(&iss_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let iss_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISS Index Buffer"),
            contents: bytemuck::cast_slice(&iss_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let iss_index_count = iss_indices.len() as u32;

        let (starlink_verts, starlink_indices) = generate_starlink();
        let starlink_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Starlink Vertex Buffer"),
            contents: bytemuck::cast_slice(&starlink_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let starlink_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Starlink Index Buffer"),
            contents: bytemuck::cast_slice(&starlink_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let starlink_index_count = starlink_indices.len() as u32;

        let (gps_verts, gps_indices) = generate_gps();
        let gps_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPS Vertex Buffer"),
            contents: bytemuck::cast_slice(&gps_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let gps_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPS Index Buffer"),
            contents: bytemuck::cast_slice(&gps_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let gps_index_count = gps_indices.len() as u32;

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

        // We support max 500 draw calls per frame, each gets 256 bytes slot
        let push_buffer_size = (500 * 256) as wgpu::BufferAddress;
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
            iss_vertex_buffer,
            iss_index_buffer,
            iss_index_count,
            starlink_vertex_buffer,
            starlink_index_buffer,
            starlink_index_count,
            gps_vertex_buffer,
            gps_index_buffer,
            gps_index_count,
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
        time: f32,
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
            time,
            star_radius: body_radii.first().copied().unwrap_or(0.163),
            _padding: 0.0,
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
        for i in 0..trails.len() {
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
        let num_spheres = bodies_pos_mass.len();
        for i in 0..num_spheres {
            let p = bodies_pos_mass[i].pos_mass;
            let radius = body_radii.get(i).copied().unwrap_or(0.001);
            
            let dx = p[0] - camera_pos[0];
            let dy = p[1] - camera_pos[1];
            let dz = p[2] - camera_pos[2];
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            
            let b_type = body_types.get(i).copied().unwrap_or(101);
            let min_size_factor = if b_type == 0 || b_type == 100 { 
                0.006 
            } else if b_type >= 12 && b_type <= 14 { // ISS, Starlink, GPS
                0.0005 
            } else if b_type == 11 { // Asteroids
                0.0008
            } else { 
                0.0025 
            };
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
        for i in 0..num_spheres {
            if body_types.get(i).copied().unwrap_or(101) == 6 {
                has_saturn_rings = true;
                let p = bodies_pos_mass[i].pos_mass;
                let radius = body_radii.get(i).copied().unwrap_or(0.001);
                
                let dx = p[0] - camera_pos[0];
                let dy = p[1] - camera_pos[1];
                let dz = p[2] - camera_pos[2];
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                
                let min_size_factor = 0.0025;
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
        
        for i in 0..trails.len() {
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
            for i in 0..trails.len() {
                let (start, count) = orbit_draw_calls[i];
                if count > 0 {
                    let offset = ((orbit_slot_start + i) * 256) as wgpu::DynamicOffset;
                    rpass.set_bind_group(0, &self.bind_group, &[offset]);
                    rpass.draw(start..(start + count), 0..1);
                }
            }

            // D. Draw Spheres & Satellites
            rpass.set_pipeline(&self.sphere_pipeline);
            let num_spheres = bodies_pos_mass.len();
            for i in 0..num_spheres {
                let offset = ((sphere_slot_start + i) * 256) as wgpu::DynamicOffset;
                rpass.set_bind_group(0, &self.bind_group, &[offset]);
                
                let b_type = body_types.get(i).copied().unwrap_or(101);
                if b_type == 12 {
                    rpass.set_vertex_buffer(0, self.iss_vertex_buffer.slice(..));
                    rpass.set_index_buffer(self.iss_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..self.iss_index_count, 0, 0..1);
                } else if b_type == 13 {
                    rpass.set_vertex_buffer(0, self.starlink_vertex_buffer.slice(..));
                    rpass.set_index_buffer(self.starlink_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..self.starlink_index_count, 0, 0..1);
                } else if b_type == 14 {
                    rpass.set_vertex_buffer(0, self.gps_vertex_buffer.slice(..));
                    rpass.set_index_buffer(self.gps_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..self.gps_index_count, 0, 0..1);
                } else {
                    rpass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
                    rpass.set_index_buffer(self.sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..self.sphere_index_count, 0, 0..1);
                }
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
            uv: [0.0, 0.0],
        });
        vertices.push(SphereVertex {
            pos: [c * outer_radius, 0.0, s * outer_radius],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 1.0],
        });
        if i < segments {
            let i0 = i * 2;
            let i1 = i0 + 1;
            let i2 = ((i + 1) * 2) % (segments * 2);
            let i3 = (i2 + 1) % (segments * 2);
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

fn add_quad_helper(
    p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3],
    normal: [f32; 3], uv_rect: [f32; 4],
    vertices: &mut Vec<SphereVertex>, indices: &mut Vec<u32>
) {
    let start = vertices.len() as u32;
    vertices.push(SphereVertex { pos: p0, normal, uv: [uv_rect[0], uv_rect[1]] });
    vertices.push(SphereVertex { pos: p1, normal, uv: [uv_rect[2], uv_rect[1]] });
    vertices.push(SphereVertex { pos: p2, normal, uv: [uv_rect[2], uv_rect[3]] });
    vertices.push(SphereVertex { pos: p3, normal, uv: [uv_rect[0], uv_rect[3]] });
    
    indices.push(start);
    indices.push(start + 1);
    indices.push(start + 2);
    
    indices.push(start);
    indices.push(start + 2);
    indices.push(start + 3);
}

fn add_box_helper(
    x_min: f32, x_max: f32, y_min: f32, y_max: f32, z_min: f32, z_max: f32,
    uv_rect: [f32; 4],
    vertices: &mut Vec<SphereVertex>, indices: &mut Vec<u32>
) {
    add_quad_helper([x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max], [0.0, 0.0, 1.0], uv_rect, vertices, indices);
    add_quad_helper([x_max, y_min, z_min], [x_min, y_min, z_min], [x_min, y_max, z_min], [x_max, y_max, z_min], [0.0, 0.0, -1.0], uv_rect, vertices, indices);
    add_quad_helper([x_min, y_max, z_max], [x_max, y_max, z_max], [x_max, y_max, z_min], [x_min, y_max, z_min], [0.0, 1.0, 0.0], uv_rect, vertices, indices);
    add_quad_helper([x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max], [0.0, -1.0, 0.0], uv_rect, vertices, indices);
    add_quad_helper([x_max, y_min, z_max], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [1.0, 0.0, 0.0], uv_rect, vertices, indices);
    add_quad_helper([x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_max], [x_min, y_max, z_min], [-1.0, 0.0, 0.0], uv_rect, vertices, indices);
}

fn generate_iss() -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // ISS components:
    // 1. Central backbone truss (along X)
    add_box_helper(-0.85, 0.85, -0.02, 0.02, -0.02, 0.02, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 2. Pressurized Modules (cluster at the center)
    add_box_helper(-0.06, 0.06, -0.06, 0.06, -0.25, 0.25, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);
    add_box_helper(-0.16, 0.16, -0.05, 0.05, -0.05, 0.05, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);
    add_box_helper(-0.05, 0.05, -0.12, 0.12, -0.05, 0.05, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 3. Left transverse truss and solar wings at x = -0.75
    add_box_helper(-0.77, -0.73, -0.02, 0.02, -0.4, 0.4, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);
    // Left Wing 1 (Z: 0.08 to 0.42) - Top and bottom double-sided
    add_quad_helper([-0.95, 0.0, 0.42], [-0.55, 0.0, 0.42], [-0.55, 0.0, 0.08], [-0.95, 0.0, 0.08], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([-0.55, 0.0, 0.42], [-0.95, 0.0, 0.42], [-0.95, 0.0, 0.08], [-0.55, 0.0, 0.08], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    // Left Wing 2 (Z: -0.42 to -0.08)
    add_quad_helper([-0.95, 0.0, -0.08], [-0.55, 0.0, -0.08], [-0.55, 0.0, -0.42], [-0.95, 0.0, -0.42], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([-0.55, 0.0, -0.08], [-0.95, 0.0, -0.08], [-0.95, 0.0, -0.42], [-0.55, 0.0, -0.42], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);

    // 4. Right transverse truss and solar wings at x = 0.75
    add_box_helper(0.73, 0.77, -0.02, 0.02, -0.4, 0.4, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);
    // Right Wing 1 (Z: 0.08 to 0.42)
    add_quad_helper([0.55, 0.0, 0.42], [0.95, 0.0, 0.42], [0.95, 0.0, 0.08], [0.55, 0.0, 0.08], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([0.95, 0.0, 0.42], [0.55, 0.0, 0.42], [0.55, 0.0, 0.08], [0.95, 0.0, 0.08], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    // Right Wing 2 (Z: -0.42 to -0.08)
    add_quad_helper([0.55, 0.0, -0.08], [0.95, 0.0, -0.08], [0.95, 0.0, -0.42], [0.55, 0.0, -0.42], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([0.95, 0.0, -0.08], [0.55, 0.0, -0.08], [0.55, 0.0, -0.42], [0.95, 0.0, -0.42], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);

    (vertices, indices)
}

fn generate_starlink() -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // Starlink components:
    // 1. Flat central chassis (box)
    add_box_helper(-0.15, 0.15, -0.04, 0.04, -0.15, 0.15, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 2. Connector boom to solar panel (only on left side, -x)
    add_box_helper(-0.25, -0.15, -0.015, 0.015, -0.02, 0.02, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 3. Single large solar array panel (extending on -x side)
    add_quad_helper([-0.9, 0.0, 0.18], [-0.25, 0.0, 0.18], [-0.25, 0.0, -0.18], [-0.9, 0.0, -0.18], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([-0.25, 0.0, 0.18], [-0.9, 0.0, 0.18], [-0.9, 0.0, -0.18], [-0.25, 0.0, -0.18], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);

    (vertices, indices)
}

fn generate_gps() -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // GPS components:
    // 1. Central boxy chassis (cube)
    add_box_helper(-0.12, 0.12, -0.12, 0.12, -0.12, 0.12, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 2. Top-facing dish/aperture antenna (pointing along +y)
    add_box_helper(-0.04, 0.04, 0.12, 0.22, -0.04, 0.04, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);
    add_box_helper(-0.06, 0.06, 0.22, 0.24, -0.06, 0.06, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 3. Connector booms for left/right panels
    add_box_helper(-0.25, -0.12, -0.015, 0.015, -0.015, 0.015, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);
    add_box_helper(0.12, 0.25, -0.015, 0.015, -0.015, 0.015, [0.0, 0.0, 0.0, 0.0], &mut vertices, &mut indices);

    // 4. Two symmetrical solar panel wings
    // Left wing
    add_quad_helper([-0.85, 0.0, 0.16], [-0.25, 0.0, 0.16], [-0.25, 0.0, -0.16], [-0.85, 0.0, -0.16], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([-0.25, 0.0, 0.16], [-0.85, 0.0, 0.16], [-0.85, 0.0, -0.16], [-0.25, 0.0, -0.16], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    // Right wing
    add_quad_helper([0.25, 0.0, 0.16], [0.85, 0.0, 0.16], [0.85, 0.0, -0.16], [0.25, 0.0, -0.16], [0.0, 1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);
    add_quad_helper([0.85, 0.0, 0.16], [0.25, 0.0, 0.16], [0.25, 0.0, -0.16], [0.85, 0.0, -0.16], [0.0, -1.0, 0.0], [0.1, 0.1, 1.0, 1.0], &mut vertices, &mut indices);

    (vertices, indices)
}
