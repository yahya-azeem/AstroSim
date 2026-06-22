use anyhow::{Result, anyhow};
use ash::{vk, Entry, Instance, Device};
use ash::vk::Handle;
use std::ffi::{CString, CStr};
use log::{info, warn};

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub has_barycentric: bool,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,
}

impl VulkanContext {
    /// Initialize a Vulkan Context.
    /// Takes sdl_instance_extensions from the SDL2 window to enable proper presentation.
    pub fn new(window: &sdl2::video::Window, sdl_instance_extensions: &[&str]) -> Result<Self> {
        info!("Initializing Vulkan entry...");
        let entry = unsafe { Entry::load().map_err(|e| anyhow!("Failed to load Vulkan library: {:?}", e))? };

        // 1. Instance Creation
        let app_name = CString::new("AstroSim")?;
        let engine_name = CString::new("NoEngine")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2); // Require Vulkan 1.2 for timeline semaphores

        // Convert extensions to CStrings
        let mut extension_names_raw: Vec<*const i8> = sdl_instance_extensions
            .iter()
            .map(|ext| CString::new(*ext).unwrap().into_raw() as *const i8)
            .collect();

        // Add VK_KHR_surface if not present in sdl_instance_extensions
        let mut has_surface_ext = false;
        for ext in sdl_instance_extensions {
            if *ext == "VK_KHR_surface" {
                has_surface_ext = true;
            }
        }
        let raw_surface_khr = CString::new("VK_KHR_surface")?;
        if !has_surface_ext {
            extension_names_raw.push(raw_surface_khr.as_ptr());
        }

        // Query available layers to enable validation if present
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|e| anyhow!("Failed to enumerate instance layers: {:?}", e))?;
        let has_validation = available_layers.iter().any(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            name.to_str().unwrap_or("").contains("VK_LAYER_KHRONOS_validation")
        });

        let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation")?;
        let mut enabled_layers = Vec::new();
        if has_validation {
            info!("Vulkan validation layers are supported and enabled.");
            enabled_layers.push(validation_layer_name.as_ptr());
        } else {
            warn!("Vulkan validation layers are NOT supported on this system.");
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names_raw)
            .enabled_layer_names(&enabled_layers);

        info!("Creating Vulkan Instance...");
        let instance = unsafe {
            entry.create_instance(&create_info, None)
                .map_err(|e| anyhow!("Failed to create Vulkan Instance: {:?}", e))?
        };

        // Free the allocated raw CStrings
        for ext in extension_names_raw {
            if ext != raw_surface_khr.as_ptr() {
                unsafe { let _ = CString::from_raw(ext as *mut i8); }
            }
        }

        // Create surface from SDL2 window
        let surface_handle = unsafe {
            window.vulkan_create_surface(instance.handle().as_raw() as usize)
                .map_err(|e| anyhow!("Failed to create Vulkan surface: {}", e))? as u64
        };
        let surface = vk::SurfaceKHR::from_raw(surface_handle);
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // 2. Physical Device Selection
        info!("Selecting Physical Device...");
        let pdevices = unsafe {
            instance.enumerate_physical_devices()
                .map_err(|e| anyhow!("Failed to enumerate physical devices: {:?}", e))?
        };

        let physical_device = pdevices.into_iter().next()
            .ok_or_else(|| anyhow!("No Vulkan-compatible physical devices found"))?;

        // 3. Queue Family Selection
        info!("Selecting Queue Family...");
        let queue_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut queue_family_index = None;
        for (index, prop) in queue_props.iter().enumerate() {
            let idx = index as u32;
            let supports_present = unsafe {
                surface_loader.get_physical_device_surface_support(physical_device, idx, surface)?
            };
            if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) && supports_present {
                queue_family_index = Some(idx);
                break;
            }
        }
        let queue_family_index = queue_family_index
            .ok_or_else(|| anyhow!("No graphics/compute queue family with presentation support found"))?;

        // 4. Logical Device Creation
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        // Required base device extensions via raw CStr to avoid ash naming mismatches
        let mut enabled_extensions = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            CStr::from_bytes_with_nul(b"VK_KHR_external_memory\0").unwrap().as_ptr(),
            CStr::from_bytes_with_nul(b"VK_KHR_external_memory_fd\0").unwrap().as_ptr(),
            CStr::from_bytes_with_nul(b"VK_KHR_external_semaphore\0").unwrap().as_ptr(),
            CStr::from_bytes_with_nul(b"VK_KHR_external_semaphore_fd\0").unwrap().as_ptr(),
        ];

        let supported_exts = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let has_barycentric = supported_exts.iter().any(|ext| {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name.to_str().unwrap_or("").contains("fragment_shader_barycentric")
        });

        // Enable timeline semaphores feature
        let mut timeline_semaphore_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
            .timeline_semaphore(true);

        // Enable fragment shader barycentric feature if supported
        let mut barycentric_features = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default()
            .fragment_shader_barycentric(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .push_next(&mut timeline_semaphore_features);

        if has_barycentric {
            info!("Barycentric coordinates extension is supported and enabled.");
            enabled_extensions.push(CStr::from_bytes_with_nul(b"VK_KHR_fragment_shader_barycentric\0").unwrap().as_ptr());
            
            // Chain features: device_create_info -> barycentric -> timeline
            barycentric_features.p_next = device_create_info.p_next as *mut _;
            device_create_info.p_next = &barycentric_features as *const _ as *mut _;
        } else {
            warn!("Barycentric coordinates extension is NOT supported. Barycentric shaders will be bypassed.");
        }

        device_create_info = device_create_info.enabled_extension_names(&enabled_extensions);

        info!("Creating Vulkan Logical Device...");
        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
                .map_err(|e| anyhow!("Failed to create Vulkan Device: {:?}", e))?
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        info!("Vulkan context successfully initialized.");
        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            queue_family_index,
            queue,
            has_barycentric,
            surface,
            surface_loader,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            info!("Destroying Vulkan surface...");
            self.surface_loader.destroy_surface(self.surface, None);
            info!("Destroying Vulkan logical device...");
            self.device.destroy_device(None);
            info!("Destroying Vulkan instance...");
            self.instance.destroy_instance(None);
        }
    }
}
