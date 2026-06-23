use imgui::{Context, DrawData, DrawVert, TextureId};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use std::collections::HashMap;

#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
struct DrawVertPod(DrawVert);

unsafe impl bytemuck::Zeroable for DrawVertPod {}
unsafe impl bytemuck::Pod for DrawVertPod {}

const SHADER_SRC: &str = r#"
struct Uniforms {
    u_Matrix: mat4x4<f32>,
};

struct VertexInput {
    @location(0) a_Pos: vec2<f32>,
    @location(1) a_UV: vec2<f32>,
    @location(2) a_Color: vec4<f32>,
};

struct VertexOutput {
    @location(0) v_UV: vec2<f32>,
    @location(1) v_Color: vec4<f32>,
    @builtin(position) v_Position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.v_UV = in.a_UV;
    out.v_Color = in.a_Color;
    out.v_Position = uniforms.u_Matrix * vec4<f32>(in.a_Pos.xy, 0.0, 1.0);
    return out;
}

struct FragmentOutput {
    @location(0) o_Target: vec4<f32>,
};

@group(1) @binding(0)
var u_Texture: texture_2d<f32>;
@group(1) @binding(1)
var u_Sampler: sampler;

fn srgb_to_linear(srgb: vec4<f32>) -> vec4<f32> {
    let color_srgb = srgb.rgb;
    let selector = ceil(color_srgb - 0.04045);
    let under = color_srgb / 12.92;
    let over = pow((color_srgb + 0.055) / 1.055, vec3<f32>(2.4));
    let result = mix(under, over, selector);
    return vec4<f32>(result, srgb.a);
}

@fragment
fn fs_main_linear(in: VertexOutput) -> FragmentOutput {
    let color = srgb_to_linear(in.v_Color);
    return FragmentOutput(color * textureSample(u_Texture, u_Sampler, in.v_UV));
}
"#;

pub struct CustomImguiRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub textures: HashMap<TextureId, wgpu::BindGroup>,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_buffer_size: usize,
    index_buffer: Option<wgpu::Buffer>,
    index_buffer_size: usize,
    fb_width: f32,
    fb_height: f32,
}

impl CustomImguiRenderer {
    pub fn new(
        imgui: &mut Context,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Custom ImGui Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imgui uniform buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imgui uniform bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("imgui uniform bind group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imgui texture bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("imgui pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&texture_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("imgui render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DrawVert>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Unorm8x4
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main_linear"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("imgui sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let mut renderer = Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            sampler,
            textures: HashMap::new(),
            vertex_buffer: None,
            vertex_buffer_size: 0,
            index_buffer: None,
            index_buffer_size: 0,
            fb_width: 0.0,
            fb_height: 0.0,
        };

        renderer.reload_font_texture(imgui, device, queue);

        renderer
    }

    pub fn reload_font_texture(
        &mut self,
        imgui: &mut Context,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let fonts = imgui.fonts();
        let handle = fonts.build_rgba32_texture();

        let font_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imgui font atlas"),
            size: wgpu::Extent3d {
                width: handle.width,
                height: handle.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &font_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            handle.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * handle.width),
                rows_per_image: Some(handle.height),
            },
            wgpu::Extent3d {
                width: handle.width,
                height: handle.height,
                depth_or_array_layers: 1,
            },
        );

        let view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("imgui font bind group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.textures.insert(fonts.tex_id, bind_group);
        fonts.clear_tex_data();
    }

    pub fn render(
        &mut self,
        draw_data: &DrawData,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> anyhow::Result<()> {
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        if fb_width <= 0.0 || fb_height <= 0.0 || draw_data.draw_lists_count() == 0 {
            return Ok(());
        }

        if (self.fb_width - fb_width).abs() > f32::EPSILON
            || (self.fb_height - fb_height).abs() > f32::EPSILON
        {
            self.fb_width = fb_width;
            self.fb_height = fb_height;

            let width = draw_data.display_size[0];
            let height = draw_data.display_size[1];
            let offset_x = draw_data.display_pos[0] / width;
            let offset_y = draw_data.display_pos[1] / height;

            let matrix = [
                [2.0 / width, 0.0, 0.0, 0.0],
                [0.0, 2.0 / -height, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0 - offset_x * 2.0, 1.0 + offset_y * 2.0, 0.0, 1.0],
            ];
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&matrix));
        }

        let mut vertices: Vec<DrawVertPod> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        struct DrawBatch {
            index_start: u32,
            index_end: u32,
            texture_id: TextureId,
            clip_rect: [f32; 4],
        }
        let mut batches = Vec::new();

        let fb_size = [fb_width, fb_height];
        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;

        for draw_list in draw_data.draw_lists() {
            let vtx_offset = vertices.len() as u32;
            let idx_offset = indices.len() as u32;

            let draw_list_vertices = draw_list.vtx_buffer();
            for v in draw_list_vertices {
                vertices.push(DrawVertPod(*v));
            }

            let draw_list_indices = draw_list.idx_buffer();
            for &idx in draw_list_indices {
                indices.push(vtx_offset + idx as u32);
            }

            for cmd in draw_list.commands() {
                if let imgui::DrawCmd::Elements { count, cmd_params } = cmd {
                    let clip_rect = [
                        (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0],
                        (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1],
                        (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0],
                        (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1],
                    ];

                    let start = idx_offset + cmd_params.idx_offset as u32;
                    let end = start + count as u32;

                    batches.push(DrawBatch {
                        index_start: start,
                        index_end: end,
                        texture_id: cmd_params.texture_id,
                        clip_rect,
                    });
                }
            }
        }

        if vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        let vertices_bytes = bytemuck::cast_slice(&vertices);
        if self.vertex_buffer.is_none() || self.vertex_buffer_size < vertices_bytes.len() {
            self.vertex_buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("imgui vertex buffer"),
                contents: vertices_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }));
            self.vertex_buffer_size = vertices_bytes.len();
        } else if let Some(ref buffer) = self.vertex_buffer {
            queue.write_buffer(buffer, 0, vertices_bytes);
        }

        let indices_bytes = bytemuck::cast_slice(&indices);
        if self.index_buffer.is_none() || self.index_buffer_size < indices_bytes.len() {
            self.index_buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("imgui index buffer"),
                contents: indices_bytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }));
            self.index_buffer_size = indices_bytes.len();
        } else if let Some(ref buffer) = self.index_buffer {
            queue.write_buffer(buffer, 0, indices_bytes);
        }

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
        rpass.set_index_buffer(self.index_buffer.as_ref().unwrap().slice(..), wgpu::IndexFormat::Uint32);

        for batch in batches {
            if batch.clip_rect[0] < fb_size[0]
                && batch.clip_rect[1] < fb_size[1]
                && batch.clip_rect[2] >= 0.0
                && batch.clip_rect[3] >= 0.0
            {
                let scissors = (
                    batch.clip_rect[0].max(0.0).floor() as u32,
                    batch.clip_rect[1].max(0.0).floor() as u32,
                    (batch.clip_rect[2].min(fb_size[0]) - batch.clip_rect[0].max(0.0))
                        .abs()
                        .ceil() as u32,
                    (batch.clip_rect[3].min(fb_size[1]) - batch.clip_rect[1].max(0.0))
                        .abs()
                        .ceil() as u32,
                );

                if scissors.2 > 0 && scissors.3 > 0 {
                    rpass.set_scissor_rect(scissors.0, scissors.1, scissors.2, scissors.3);

                    if let Some(tex_bind_group) = self.textures.get(&batch.texture_id) {
                        rpass.set_bind_group(1, tex_bind_group, &[]);
                        rpass.draw_indexed(batch.index_start..batch.index_end, 0, 0..1);
                    } else {
                        anyhow::bail!("imgui render error: bad texture id '{}'", batch.texture_id.id());
                    }
                }
            }
        }

        Ok(())
    }
}
