use crate::render::vulkan_context::VulkanContext;
use crate::render::swapchain::SwapchainManager;
use anyhow::{Result, anyhow};
use ash::{vk, Device};
use ash::vk::Handle;
use std::sync::Arc;
use log::{info, warn};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SphereVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SpherePushConstants {
    pub model: [[f32; 4]; 4],
    pub body_type: u32,
    pub is_selected: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct OrbitPushConstants {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BodyUbo {
    pub pos_mass: [f32; 4], // xyz = position in render space, w = visual warping strength
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
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
    pub a: f32, // semi-major axis
    pub e: f32, // eccentricity
}

pub struct Renderer {
    pub vulkan_context: Arc<VulkanContext>,
    pub render_pass: vk::RenderPass,
    pub swapchain_manager: SwapchainManager,
    
    // Graphics pipeline resources
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub skybox_pipeline: vk::Pipeline,
    pub sphere_pipeline: vk::Pipeline,
    pub orbit_pipeline: vk::Pipeline,
    pub orbit_vertex_buffer_size: vk::DeviceSize,
    
    // Grid & Orbits meshes
    pub grid_vertex_buffer: vk::Buffer,
    pub grid_vertex_memory: vk::DeviceMemory,
    pub grid_vertex_count: u32,
    
    pub orbit_vertex_buffer: vk::Buffer,
    pub orbit_vertex_memory: vk::DeviceMemory,
    pub orbit_vertex_count: u32,

    // Sphere & Rings meshes
    pub sphere_vertex_buffer: vk::Buffer,
    pub sphere_vertex_memory: vk::DeviceMemory,
    pub sphere_index_buffer: vk::Buffer,
    pub sphere_index_memory: vk::DeviceMemory,
    pub sphere_index_count: u32,

    pub ring_vertex_buffer: vk::Buffer,
    pub ring_vertex_memory: vk::DeviceMemory,
    pub ring_index_buffer: vk::Buffer,
    pub ring_index_memory: vk::DeviceMemory,
    pub ring_index_count: u32,
    
    // Uniform buffers (one per swapchain image)
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    
    // Sync objects
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub current_frame: usize,
    
    // Command buffers
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    
    // ImGui renderer
    pub imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
}

impl Renderer {
    pub fn new(vulkan_context: Arc<VulkanContext>, imgui_context: &mut imgui::Context) -> Result<Self> {
        info!("Creating Vulkan Renderer...");
        let device = &vulkan_context.device;
        let instance = &vulkan_context.instance;
        let physical_device = vulkan_context.physical_device;

        // 1. Create Render Pass
        let color_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::B8G8R8A8_UNORM) // Will match swapchain format
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };

        // 2. Create Swapchain
        let swapchain_manager = SwapchainManager::new(Arc::clone(&vulkan_context), 1280, 720, render_pass)?;
        info!("Render pass format: {:?}, Swapchain format: {:?}", vk::Format::B8G8R8A8_UNORM, swapchain_manager.format);

        // 3. Create Descriptor Set Layout for MVP + Planets UBO
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&ubo_layout_binding));

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // 4. Create Pipeline Layout
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(80); // max of SpherePushConstants (72) and OrbitPushConstants (80)

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // 5. Create Graphics Pipeline
        let vert_spv = include_bytes!("../../shaders/grid_vert.spv");
        let frag_spv = include_bytes!("../../shaders/grid_frag.spv");

        let vert_module = load_shader_module(device, vert_spv)?;
        let frag_module = load_shader_module(device, frag_spv)?;

        let main_entry = std::ffi::CString::new("main")?;
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main_entry),
        ];

        // Vertex input descriptions (vec2 inPos)
        let binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(8) // 2 * sizeof(f32)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attribute_description = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0);

        let binding_descriptions = [binding_description];
        let attribute_descriptions = [attribute_description];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::LINE_LIST) // Draw as lines
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(swapchain_manager.extent.width as f32)
            .height(swapchain_manager.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(swapchain_manager.extent);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true) // Enable transparency for grid glow
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .map_err(|e| anyhow!("Failed to create graphics pipeline: {:?}", e))?
        };
        let pipeline = pipelines[0];

        // 5b. Create Skybox Graphics Pipeline
        let skybox_vert_spv = include_bytes!("../../shaders/skybox_vert.spv");
        let skybox_frag_spv = include_bytes!("../../shaders/skybox_frag.spv");

        let skybox_vert_module = load_shader_module(device, skybox_vert_spv)?;
        let skybox_frag_module = load_shader_module(device, skybox_frag_spv)?;

        let skybox_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(skybox_vert_module)
                .name(&main_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(skybox_frag_module)
                .name(&main_entry),
        ];

        let skybox_vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let skybox_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let skybox_rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let skybox_color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let skybox_color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&skybox_color_blend_attachment));

        let skybox_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&skybox_stages)
            .vertex_input_state(&skybox_vertex_input_info)
            .input_assembly_state(&skybox_input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&skybox_rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&skybox_color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let skybox_pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&skybox_pipeline_info), None)
                .map_err(|e| anyhow!("Failed to create skybox graphics pipeline: {:?}", e))?
        };
        let skybox_pipeline = skybox_pipelines[0];

        // 5c. Create Sphere Graphics Pipeline
        let sphere_vert_spv = include_bytes!("../../shaders/sphere_vert.spv");
        let sphere_frag_spv = include_bytes!("../../shaders/sphere_frag.spv");

        let sphere_vert_module = load_shader_module(device, sphere_vert_spv)?;
        let sphere_frag_module = load_shader_module(device, sphere_frag_spv)?;

        let sphere_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(sphere_vert_module)
                .name(&main_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(sphere_frag_module)
                .name(&main_entry),
        ];

        let sphere_binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<SphereVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        let sphere_attribute_descriptions = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(24),
        ];

        let sphere_vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&sphere_binding_description))
            .vertex_attribute_descriptions(&sphere_attribute_descriptions);

        let sphere_rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let sphere_color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let sphere_color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&sphere_color_blend_attachment));

        let sphere_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&sphere_stages)
            .vertex_input_state(&sphere_vertex_input_info)
            .input_assembly_state(&skybox_input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&sphere_rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&sphere_color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let sphere_pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&sphere_pipeline_info), None)
                .map_err(|e| anyhow!("Failed to create sphere graphics pipeline: {:?}", e))?
        };
        let sphere_pipeline = sphere_pipelines[0];

        // 5d. Create Orbit Graphics Pipeline
        let orbit_vert_spv = include_bytes!("../../shaders/orbit_vert.spv");
        let orbit_frag_spv = include_bytes!("../../shaders/orbit_frag.spv");

        let orbit_vert_module = load_shader_module(device, orbit_vert_spv)?;
        let orbit_frag_module = load_shader_module(device, orbit_frag_spv)?;

        let orbit_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(orbit_vert_module)
                .name(&main_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(orbit_frag_module)
                .name(&main_entry),
        ];

        let orbit_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::LINE_STRIP)
            .primitive_restart_enable(false);

        let orbit_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&orbit_stages)
            .vertex_input_state(&vertex_input_info) // Reuses vec2 inPos vertex binding
            .input_assembly_state(&orbit_input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let orbit_pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&orbit_pipeline_info), None)
                .map_err(|e| anyhow!("Failed to create orbit graphics pipeline: {:?}", e))?
        };
        let orbit_pipeline = orbit_pipelines[0];

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
            device.destroy_shader_module(skybox_vert_module, None);
            device.destroy_shader_module(skybox_frag_module, None);
            device.destroy_shader_module(sphere_vert_module, None);
            device.destroy_shader_module(sphere_frag_module, None);
            device.destroy_shader_module(orbit_vert_module, None);
            device.destroy_shader_module(orbit_frag_module, None);
        }

        // 6b. Generate Sphere Mesh & Upload to GPU
        let (sphere_verts, sphere_inds) = generate_sphere(32, 32);
        let sphere_index_count = sphere_inds.len() as u32;
        let sphere_vertex_buffer_size = (sphere_verts.len() * std::mem::size_of::<SphereVertex>()) as vk::DeviceSize;
        let sphere_index_buffer_size = (sphere_inds.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;

        let (sphere_vertex_buffer, sphere_vertex_memory) = create_buffer(
            device,
            instance,
            physical_device,
            sphere_vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(sphere_vertex_memory, 0, sphere_vertex_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(sphere_verts.as_ptr() as *const u8, data_ptr as *mut u8, sphere_vertex_buffer_size as usize);
            device.unmap_memory(sphere_vertex_memory);
        }

        let (sphere_index_buffer, sphere_index_memory) = create_buffer(
            device,
            instance,
            physical_device,
            sphere_index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(sphere_index_memory, 0, sphere_index_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(sphere_inds.as_ptr() as *const u8, data_ptr as *mut u8, sphere_index_buffer_size as usize);
            device.unmap_memory(sphere_index_memory);
        }

        // 6c. Generate Saturn Ring Mesh & Upload to GPU
        let (ring_verts, ring_inds) = generate_ring(1.35, 2.5, 64);
        let ring_index_count = ring_inds.len() as u32;
        let ring_vertex_buffer_size = (ring_verts.len() * std::mem::size_of::<SphereVertex>()) as vk::DeviceSize;
        let ring_index_buffer_size = (ring_inds.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;

        let (ring_vertex_buffer, ring_vertex_memory) = create_buffer(
            device,
            instance,
            physical_device,
            ring_vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(ring_vertex_memory, 0, ring_vertex_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ring_verts.as_ptr() as *const u8, data_ptr as *mut u8, ring_vertex_buffer_size as usize);
            device.unmap_memory(ring_vertex_memory);
        }

        let (ring_index_buffer, ring_index_memory) = create_buffer(
            device,
            instance,
            physical_device,
            ring_index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(ring_index_memory, 0, ring_index_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ring_inds.as_ptr() as *const u8, data_ptr as *mut u8, ring_index_buffer_size as usize);
            device.unmap_memory(ring_index_memory);
        }

        // 6. Generate Grid Mesh & Upload to GPU
        let grid_size = 40.0;
        let grid_lines = 100;
        let grid_segments = 140;
        let grid_verts = generate_grid_vertices(grid_size, grid_lines, grid_segments);
        let grid_vertex_count = grid_verts.len() as u32;
        let grid_buffer_size = (grid_verts.len() * std::mem::size_of::<[f32; 2]>()) as vk::DeviceSize;

        let (grid_vertex_buffer, grid_vertex_memory) = create_buffer(
            device,
            instance,
            physical_device,
            grid_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(grid_vertex_memory, 0, grid_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(grid_verts.as_ptr() as *const u8, data_ptr as *mut u8, grid_buffer_size as usize);
            device.unmap_memory(grid_vertex_memory);
        }

        // 7. Generate Orbit Paths Buffer & Upload to GPU (dynamic size for 12,000 vertices)
        let orbit_vertex_count = 12000_u32;
        let orbit_vertex_buffer_size = (orbit_vertex_count as usize * std::mem::size_of::<[f32; 2]>()) as vk::DeviceSize;

        let (orbit_vertex_buffer, orbit_vertex_memory) = create_buffer(
            device,
            instance,
            physical_device,
            orbit_vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 8. Create Uniform Buffers (one per swapchain frame)
        let swapchain_image_count = swapchain_manager.images.len();
        let ubo_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let mut uniform_buffers = Vec::with_capacity(swapchain_image_count);
        let mut uniform_buffers_memory = Vec::with_capacity(swapchain_image_count);

        for _ in 0..swapchain_image_count {
            let (buf, mem) = create_buffer(
                device,
                instance,
                physical_device,
                ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            uniform_buffers.push(buf);
            uniform_buffers_memory.push(mem);
        }

        // 9. Create Descriptor Pool
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain_image_count as u32,
        };

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(swapchain_image_count as u32);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // 10. Allocate Descriptor Sets
        let layouts = vec![descriptor_set_layout; swapchain_image_count];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        // Update descriptor sets
        for i in 0..swapchain_image_count {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(ubo_size);

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]) };
        }

        // 11. Command Buffers allocation
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(vulkan_context.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain_image_count as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };

        // 12. Create Synchronization Objects
        let max_frames_in_flight = 2;
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::with_capacity(max_frames_in_flight);
        let mut render_finished_semaphores = Vec::with_capacity(max_frames_in_flight);
        let mut in_flight_fences = Vec::with_capacity(max_frames_in_flight);

        for _ in 0..max_frames_in_flight {
            image_available_semaphores.push(unsafe { device.create_semaphore(&semaphore_info, None)? });
            render_finished_semaphores.push(unsafe { device.create_semaphore(&semaphore_info, None)? });
            in_flight_fences.push(unsafe { device.create_fence(&fence_info, None)? });
        }

        // 13. Initialize Dear ImGui Vulkan Renderer
        // ImGui vulkan renderer requires physical_device_properties for memory types
        info!("Initializing Dear ImGui Vulkan Renderer...");
        let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
            instance,
            physical_device,
            device.clone(),
            vulkan_context.queue,
            command_pool,
            render_pass,
            imgui_context,
            Some(imgui_rs_vulkan_renderer::Options {
                in_flight_frames: max_frames_in_flight,
                ..Default::default()
            }),
        ).map_err(|e| anyhow!("Failed to create ImGui Vulkan renderer: {:?}", e))?;

        Ok(Self {
            vulkan_context,
            render_pass,
            swapchain_manager,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            skybox_pipeline,
            sphere_pipeline,
            orbit_pipeline,
            orbit_vertex_buffer_size,
            grid_vertex_buffer,
            grid_vertex_memory,
            grid_vertex_count,
            orbit_vertex_buffer,
            orbit_vertex_memory,
            orbit_vertex_count,
            sphere_vertex_buffer,
            sphere_vertex_memory,
            sphere_index_buffer,
            sphere_index_memory,
            sphere_index_count,
            ring_vertex_buffer,
            ring_vertex_memory,
            ring_index_buffer,
            ring_index_memory,
            ring_index_count,
            uniform_buffers,
            uniform_buffers_memory,
            descriptor_pool,
            descriptor_sets,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            command_pool,
            command_buffers,
            imgui_renderer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        let device = &self.vulkan_context.device;
        unsafe { device.device_wait_idle()? };

        self.swapchain_manager.destroy();
        self.swapchain_manager = SwapchainManager::new(
            Arc::clone(&self.vulkan_context),
            width,
            height,
            self.render_pass,
        )?;
        Ok(())
    }

    pub fn draw_frame(
        &mut self,
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        inv_view_proj: [[f32; 4]; 4],
        bodies: &[BodyUbo],
        body_radii: &[f32],
        body_types: &[u32],
        body_colors: &[[f32; 4]],
        selected_idx: usize,
        hovered_idx: Option<usize>,
        camera_pos: [f32; 3],
        trails: &[Vec<[f32; 2]>],
        imgui_draw_data: &imgui::DrawData,
    ) -> Result<()> {
        let device = &self.vulkan_context.device;
        let queue = self.vulkan_context.queue;
        let swapchain_loader = &self.swapchain_manager.swapchain_loader;
        let swapchain = self.swapchain_manager.swapchain;

        // 1. Wait for fence
        let fence = self.in_flight_fences[self.current_frame];
        unsafe {
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // 2. Acquire Image
        let image_available = self.image_available_semaphores[self.current_frame];
        let image_index = match unsafe {
            swapchain_loader.acquire_next_image(swapchain, u64::MAX, image_available, vk::Fence::null())
        } {
            Ok((idx, _)) => idx as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // Trigger resize in case of layout changes
                return Ok(());
            }
            Err(e) => return Err(anyhow!("Failed to acquire next image: {:?}", e)),
        };

        // Reset fence only if we are presenting
        unsafe {
            device.reset_fences(&[fence])?;
        }

        // Map and copy dynamic orbit trails to orbit_vertex_buffer
        let mut offset = 0;
        let mut draw_commands = Vec::new();
        unsafe {
            let data_ptr = device.map_memory(
                self.orbit_vertex_memory,
                0,
                self.orbit_vertex_buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;
            let writer = data_ptr as *mut [f32; 2];
            for (i, trail) in trails.iter().enumerate() {
                let len = trail.len();
                if len < 2 {
                    continue;
                }
                if offset + len > 12000 {
                    break;
                }
                std::ptr::copy_nonoverlapping(trail.as_ptr(), writer.add(offset), len);
                draw_commands.push((offset as u32, len as u32, i));
                offset += len;
            }
            device.unmap_memory(self.orbit_vertex_memory);
        }

        // 3. Update Uniform Buffer
        let mut active_bodies = [BodyUbo { pos_mass: [0.0; 4] }; 10];
        let num_bodies = bodies.len().min(10);
        for i in 0..num_bodies {
            active_bodies[i] = bodies[i];
        }

        let ubo = UniformBufferObject {
            model: nalgebra::Matrix4::identity().into(),
            view,
            proj,
            inv_view_proj,
            bodies: active_bodies,
            num_bodies: num_bodies as i32,
            time: 0.0,
            _padding: [0.0; 2],
        };

        let ubo_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        unsafe {
            let data_ptr = device.map_memory(self.uniform_buffers_memory[image_index], 0, ubo_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(&ubo as *const UniformBufferObject as *const u8, data_ptr as *mut u8, ubo_size as usize);
            device.unmap_memory(self.uniform_buffers_memory[image_index]);
        }

        // 4. Record Command Buffer
        let cmd_buf = self.command_buffers[image_index];
        unsafe {
            device.reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::empty())?;
        }

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.03, 0.03, 0.06, 1.0] }, // Deep space background
            }];

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.swapchain_manager.framebuffers[image_index])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain_manager.extent,
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cmd_buf, &render_pass_begin, vk::SubpassContents::INLINE);

            // Set dynamic states
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain_manager.extent.width as f32,
                height: self.swapchain_manager.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd_buf, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_manager.extent,
            };
            device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            // Draw Skybox
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.skybox_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
            device.cmd_draw(cmd_buf, 3, 1, 0, 0);

            // Draw Gravity Grid
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );

            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.grid_vertex_buffer], &[0]);
            device.cmd_draw(cmd_buf, self.grid_vertex_count, 1, 0, 0);

            // Draw Orbit Trails
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.orbit_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.orbit_vertex_buffer], &[0]);

            for &(start, count, body_idx) in &draw_commands {
                let color = if body_idx < body_colors.len() {
                    body_colors[body_idx]
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                };

                let pcs = OrbitPushConstants {
                    model: nalgebra::Matrix4::identity().into(),
                    color,
                };

                let pcs_bytes = std::slice::from_raw_parts(
                    &pcs as *const OrbitPushConstants as *const u8,
                    std::mem::size_of::<OrbitPushConstants>(),
                );

                device.cmd_push_constants(
                    cmd_buf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pcs_bytes,
                );

                device.cmd_draw(cmd_buf, count, 1, start, 0);
            }

            // Painters' algorithm: sort spheres by distance to camera
            let mut render_order: Vec<usize> = (0..num_bodies).collect();
            let cam_vec = nalgebra::Vector3::new(camera_pos[0], camera_pos[1], camera_pos[2]);

            render_order.sort_by(|&a, &b| {
                let pos_a = nalgebra::Vector3::new(bodies[a].pos_mass[0], bodies[a].pos_mass[1], bodies[a].pos_mass[2]);
                let pos_b = nalgebra::Vector3::new(bodies[b].pos_mass[0], bodies[b].pos_mass[1], bodies[b].pos_mass[2]);
                let dist_a = (pos_a - cam_vec).norm_squared();
                let dist_b = (pos_b - cam_vec).norm_squared();
                dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal) // descending order: back to front
            });

            // Draw Shaded 3D Spheres
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.sphere_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );

            for &idx in &render_order {
                if idx >= body_radii.len() || idx >= body_types.len() {
                    continue;
                }
                let p = bodies[idx].pos_mass;
                let radius = body_radii[idx];
                let b_type = body_types[idx];
                let is_selected = if idx == selected_idx {
                    1u32
                } else if Some(idx) == hovered_idx {
                    2u32
                } else {
                    0u32
                };

                let dx = p[0] - camera_pos[0];
                let dy = p[1] - camera_pos[1];
                let dz = p[2] - camera_pos[2];
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                let min_size_factor = if b_type == 0 || b_type == 100 { 0.006 } else { 0.0025 };
                let visual_radius = radius.max(dist * min_size_factor);

                let scale = nalgebra::Matrix4::new_scaling(visual_radius);
                let translation = nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(p[0], p[1], p[2]));
                let model = translation * scale;

                let pcs = SpherePushConstants {
                    model: model.into(),
                    body_type: b_type,
                    is_selected,
                };

                let pcs_bytes = std::slice::from_raw_parts(
                    &pcs as *const SpherePushConstants as *const u8,
                    std::mem::size_of::<SpherePushConstants>(),
                );

                device.cmd_push_constants(
                    cmd_buf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pcs_bytes,
                );

                device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.sphere_vertex_buffer], &[0]);
                device.cmd_bind_index_buffer(cmd_buf, self.sphere_index_buffer, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd_buf, self.sphere_index_count, 1, 0, 0, 0);

                // Draw Saturn rings (type 6 Saturn)
                if b_type == 6 {
                    let ring_pcs = SpherePushConstants {
                        model: model.into(), // scale rings proportionally with saturn
                        body_type: 9, // Ring type
                        is_selected: 0,
                    };
                    let ring_pcs_bytes = std::slice::from_raw_parts(
                        &ring_pcs as *const SpherePushConstants as *const u8,
                        std::mem::size_of::<SpherePushConstants>(),
                    );
                    device.cmd_push_constants(
                        cmd_buf,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        ring_pcs_bytes,
                    );
                    device.cmd_bind_vertex_buffers(cmd_buf, 0, &[self.ring_vertex_buffer], &[0]);
                    device.cmd_bind_index_buffer(cmd_buf, self.ring_index_buffer, 0, vk::IndexType::UINT32);
                    device.cmd_draw_indexed(cmd_buf, self.ring_index_count, 1, 0, 0, 0);
                }
            }

            // Draw ImGui
            self.imgui_renderer
                .cmd_draw(cmd_buf, imgui_draw_data)
                .map_err(|e| anyhow!("Failed to draw ImGui command list: {:?}", e))?;

            device.cmd_end_render_pass(cmd_buf);
            device.end_command_buffer(cmd_buf)?;
        }

        // 5. Submit Command Buffer
        let wait_semaphores = [image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let render_finished = self.render_finished_semaphores[self.current_frame];
        let signal_semaphores = [render_finished];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd_buf))
            .signal_semaphores(&signal_semaphores);

        unsafe {
            device.queue_submit(queue, std::slice::from_ref(&submit_info), fence)?;
        }

        // 6. Present
        let image_indices = [image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(std::slice::from_ref(&swapchain))
            .image_indices(&image_indices);

        match unsafe { swapchain_loader.queue_present(queue, &present_info) } {
            Ok(_) | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // Resize will be handled in event loop
            }
            Err(e) => return Err(anyhow!("Failed to present queue: {:?}", e)),
        }

        self.current_frame = (self.current_frame + 1) % 2;

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.vulkan_context.device;
        unsafe {
            info!("Waiting for Vulkan device idle before destroy...");
            let _ = device.device_wait_idle();

            info!("Destroying synchronization objects...");
            for &sem in &self.image_available_semaphores {
                device.destroy_semaphore(sem, None);
            }
            for &sem in &self.render_finished_semaphores {
                device.destroy_semaphore(sem, None);
            }
            for &fence in &self.in_flight_fences {
                device.destroy_fence(fence, None);
            }

            info!("Destroying buffers...");
            for &buf in &self.uniform_buffers {
                device.destroy_buffer(buf, None);
            }
            for &mem in &self.uniform_buffers_memory {
                device.free_memory(mem, None);
            }

            device.destroy_buffer(self.grid_vertex_buffer, None);
            device.free_memory(self.grid_vertex_memory, None);
            
            device.destroy_buffer(self.orbit_vertex_buffer, None);
            device.free_memory(self.orbit_vertex_memory, None);

            device.destroy_buffer(self.sphere_vertex_buffer, None);
            device.free_memory(self.sphere_vertex_memory, None);
            device.destroy_buffer(self.sphere_index_buffer, None);
            device.free_memory(self.sphere_index_memory, None);

            device.destroy_buffer(self.ring_vertex_buffer, None);
            device.free_memory(self.ring_vertex_memory, None);
            device.destroy_buffer(self.ring_index_buffer, None);
            device.free_memory(self.ring_index_memory, None);

            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.skybox_pipeline, None);
            device.destroy_pipeline(self.sphere_pipeline, None);
            device.destroy_pipeline(self.orbit_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);

            self.swapchain_manager.destroy();

            device.destroy_command_pool(self.command_pool, None);
            device.destroy_render_pass(self.render_pass, None);
            info!("Vulkan Renderer destroyed successfully.");
        }
    }
}

// Helper: Buffer Creation
fn create_buffer(
    device: &Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let mut memory_type_index = None;
    for i in 0..mem_properties.memory_type_count {
        if (mem_requirements.memory_type_bits & (1 << i)) != 0
            && mem_properties.memory_types[i as usize].property_flags.contains(properties)
        {
            memory_type_index = Some(i);
            break;
        }
    }
    let memory_type_index = memory_type_index
        .ok_or_else(|| anyhow!("Failed to find suitable memory type for buffer"))?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    Ok((buffer, memory))
}

// Helper: Shader module creation
fn load_shader_module(device: &Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let u32_code = unsafe {
        let (prefix, code, suffix) = bytes.align_to::<u32>();
        if !prefix.is_empty() || !suffix.is_empty() {
            let mut code_copy = vec![0u32; bytes.len() / 4];
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), code_copy.as_mut_ptr() as *mut u8, bytes.len());
            code_copy
        } else {
            code.to_vec()
        }
    };
    let create_info = vk::ShaderModuleCreateInfo::default().code(&u32_code);
    let shader_module = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(shader_module)
}

// Grid generation logic
fn generate_grid_vertices(grid_size: f32, grid_lines: u32, segments: u32) -> Vec<[f32; 2]> {
    let mut vertices = Vec::new();
    let step = (grid_size * 2.0) / (grid_lines as f32);

    // Horizontal lines
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

    // Vertical lines
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

// Keplerian Orbit paths generation
fn generate_ellipse_orbit_vertices(orbits: &[OrbitParams]) -> Vec<[f32; 2]> {
    let mut vertices = Vec::new();
    let segments = 128;
    for orbit in orbits {
        let a = orbit.a;
        let e = orbit.e;
        let b = a * (1.0 - e * e).sqrt();
        let c = a * e; // Focus distance
        for i in 0..segments {
            let theta1 = (i as f32) * 2.0 * std::f32::consts::PI / (segments as f32);
            let theta2 = ((i + 1) as f32) * 2.0 * std::f32::consts::PI / (segments as f32);
            
            // Sun is at focus, which sits at (0, 0). Ellipse center is at (-c, 0).
            let x1 = a * theta1.cos() - c;
            let y1 = b * theta1.sin();
            let x2 = a * theta2.cos() - c;
            let y2 = b * theta2.sin();
            vertices.push([x1, y1]);
            vertices.push([x2, y2]);
        }
    }
    vertices
}

fn generate_sphere(lat_segments: u32, lon_segments: u32) -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for y in 0..=lat_segments {
        let y_f = y as f32 / lat_segments as f32;
        let theta = y_f * std::f32::consts::PI; // 0 to PI

        for x in 0..=lon_segments {
            let x_f = x as f32 / lon_segments as f32;
            let phi = x_f * 2.0 * std::f32::consts::PI; // 0 to 2PI

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

            // First triangle
            indices.push(i0);
            indices.push(i2);
            indices.push(i1);

            // Second triangle
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

        // Inner vertex
        vertices.push(SphereVertex {
            pos: [c * inner_radius, 0.0, s * inner_radius],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, i as f32 / segments as f32],
        });

        // Outer vertex
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

        // First triangle
        indices.push(i0);
        indices.push(i1);
        indices.push(i2);

        // Second triangle
        indices.push(i2);
        indices.push(i1);
        indices.push(i3);
    }

    (vertices, indices)
}
