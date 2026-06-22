use ash::vk;
use std::sync::Arc;
use crate::render::VulkanContext;
use anyhow::{Result, anyhow};
use log::info;

pub struct SwapchainManager {
    vulkan_context: Arc<VulkanContext>,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl SwapchainManager {
    pub fn new(vulkan_context: Arc<VulkanContext>, width: u32, height: u32, render_pass: vk::RenderPass) -> Result<Self> {
        let instance = &vulkan_context.instance;
        let device = &vulkan_context.device;
        let physical_device = vulkan_context.physical_device;
        let surface = vulkan_context.surface;
        let surface_loader = &vulkan_context.surface_loader;

        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

        // 1. Query capabilities
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };

        // 2. Select format (prefer B8G8R8A8_UNORM)
        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .cloned()
            .unwrap_or_else(|| formats[0].clone());

        // 3. Select present mode (prefer MAILBOX for low latency, fallback to FIFO which is always supported)
        let present_mode = present_modes.iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .cloned()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // 4. Select extent
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        // Determine image count (minimum + 1, capped by max if max > 0)
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        // 5. Create Swapchain
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        info!("Creating Swapchain ({}x{})...", extent.width, extent.height);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        // 6. Retrieve images & create views
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let mut image_views = Vec::with_capacity(images.len());
        for &img in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = unsafe { device.create_image_view(&view_info, None)? };
            image_views.push(view);
        }

        // 7. Create Framebuffers
        let mut framebuffers = Vec::with_capacity(image_views.len());
        for &view in &image_views {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            let fb = unsafe { device.create_framebuffer(&fb_info, None)? };
            framebuffers.push(fb);
        }

        Ok(Self {
            vulkan_context,
            swapchain_loader,
            swapchain,
            images,
            image_views,
            framebuffers,
            format: format.format,
            extent,
        })
    }

    pub fn destroy(&mut self) {
        let device = &self.vulkan_context.device;
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
