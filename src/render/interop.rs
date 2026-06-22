use anyhow::{Result, anyhow};
use ash::{vk, Instance, Device};
use log::{info, warn};

#[allow(non_camel_case_types)]
pub type CUexternalMemory = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type CUdeviceptr = std::os::raw::c_ulonglong;

#[repr(C)]
#[derive(Copy, Clone)]
pub union CUexternalMemoryHandleDesc_st_handle {
    pub fd: std::os::raw::c_int,
    pub win32: CUexternalMemoryHandleDesc_st_handle_win32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUexternalMemoryHandleDesc_st_handle_win32 {
    pub handle: *mut std::ffi::c_void,
    pub name: *const std::ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUexternalMemoryHandleDesc_st {
    pub type_: u32,
    pub handle: CUexternalMemoryHandleDesc_st_handle,
    pub size: u64,
    pub flags: u32,
    pub reserved: [u32; 16],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUexternalMemoryBufferDesc_st {
    pub offset: u64,
    pub size: u64,
    pub flags: u32,
    pub reserved: [u32; 16],
}

type FnCuImportExternalMemory = unsafe extern "C" fn(
    extMem_out: *mut CUexternalMemory,
    memHandleDesc: *const CUexternalMemoryHandleDesc_st,
) -> i32;

type FnCuExternalMemoryGetMappedBuffer = unsafe extern "C" fn(
    devPtr_out: *mut CUdeviceptr,
    extMem: CUexternalMemory,
    bufferDesc: *const CUexternalMemoryBufferDesc_st,
) -> i32;

type FnCuDestroyExternalMemory = unsafe extern "C" fn(
    extMem: CUexternalMemory,
) -> i32;

struct CudaInteropLoader {
    _lib: libloading::Library,
    cu_import_external_memory: FnCuImportExternalMemory,
    cu_external_memory_get_mapped_buffer: FnCuExternalMemoryGetMappedBuffer,
    cu_destroy_external_memory: FnCuDestroyExternalMemory,
}

impl CudaInteropLoader {
    fn try_new() -> Result<Self> {
        let lib = unsafe {
            libloading::Library::new("libcuda.so")
                .or_else(|_| libloading::Library::new("libcuda.so.1"))
                .map_err(|e| anyhow!("Failed to load libcuda: {:?}", e))?
        };

        let cu_import_external_memory = unsafe {
            *lib.get(b"cuImportExternalMemory\0")
                .map_err(|e| anyhow!("Symbol cuImportExternalMemory not found: {:?}", e))?
        };

        let cu_external_memory_get_mapped_buffer = unsafe {
            *lib.get(b"cuExternalMemoryGetMappedBuffer\0")
                .map_err(|e| anyhow!("Symbol cuExternalMemoryGetMappedBuffer not found: {:?}", e))?
        };

        let cu_destroy_external_memory = unsafe {
            *lib.get(b"cuDestroyExternalMemory\0")
                .map_err(|e| anyhow!("Symbol cuDestroyExternalMemory not found: {:?}", e))?
        };

        Ok(Self {
            _lib: lib,
            cu_import_external_memory,
            cu_external_memory_get_mapped_buffer,
            cu_destroy_external_memory,
        })
    }
}

pub struct CudaVulkanBridge {
    pub device_memory: vk::DeviceMemory,
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
    #[allow(dead_code)]
    pub fd: std::os::unix::io::RawFd,
    cuda_ext_memory: Option<CUexternalMemory>,
    pub cuda_device_ptr: Option<CUdeviceptr>,
    cuda_destroy_fn: Option<FnCuDestroyExternalMemory>,
}

impl CudaVulkanBridge {
    /// Creates a Vulkan buffer backed by memory that is exported for CUDA interop.
    /// If CUDA is available, imports the memory into CUDA and maps it to a CUdeviceptr.
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        cuda_device_available: bool,
    ) -> Result<Self> {
        // 1. Create Vulkan Buffer with external memory support
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // Get memory requirements
        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Find memory type index
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mut memory_type_index = None;
        for i in 0..mem_props.memory_type_count {
            if (mem_reqs.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                memory_type_index = Some(i);
                break;
            }
        }
        let memory_type_index = memory_type_index
            .ok_or_else(|| anyhow!("Failed to find suitable device local memory type"))?;

        // 2. Allocate exportable Vulkan Device Memory
        let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_alloc_info);

        let device_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, device_memory, 0)? };

        // 3. Export POSIX File Descriptor (Linux)
        let fd_loader = ash::khr::external_memory_fd::Device::new(instance, device);
        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(device_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let fd = unsafe { fd_loader.get_memory_fd(&get_fd_info)? };

        // 4. Import memory into CUDA if available
        let mut cuda_ext_memory = None;
        let mut cuda_device_ptr = None;
        let mut cuda_destroy_fn = None;

        if cuda_device_available {
            if let Ok(loader) = CudaInteropLoader::try_new() {
                unsafe {
                    let mut ext_mem = std::ptr::null_mut();
                    let mut handle = CUexternalMemoryHandleDesc_st_handle { fd };
                    let handle_desc = CUexternalMemoryHandleDesc_st {
                        type_: 1, // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
                        handle,
                        size,
                        flags: 0,
                        reserved: [0; 16],
                    };

                    let res = (loader.cu_import_external_memory)(&mut ext_mem, &handle_desc);
                    if res == 0 {
                        cuda_ext_memory = Some(ext_mem);
                        cuda_destroy_fn = Some(loader.cu_destroy_external_memory);

                        let mut dev_ptr = 0;
                        let buffer_desc = CUexternalMemoryBufferDesc_st {
                            offset: 0,
                            size,
                            flags: 0,
                            reserved: [0; 16],
                        };

                        let res_map = (loader.cu_external_memory_get_mapped_buffer)(&mut dev_ptr, ext_mem, &buffer_desc);
                        if res_map == 0 {
                            cuda_device_ptr = Some(dev_ptr);
                            info!("Successfully imported Vulkan buffer memory to CUDA.");
                        } else {
                            warn!("CUDA external memory mapping failed with code {}", res_map);
                        }
                    } else {
                        warn!("CUDA external memory import failed with code {}", res);
                    }
                }
            } else {
                warn!("CUDA interop loader could not locate libcuda.");
            }
        }

        Ok(Self {
            device_memory,
            buffer,
            size,
            fd,
            cuda_ext_memory,
            cuda_device_ptr,
            cuda_destroy_fn,
        })
    }
}

impl Drop for CudaVulkanBridge {
    fn drop(&mut self) {
        if let (Some(ext_mem), Some(destroy_fn)) = (self.cuda_ext_memory, self.cuda_destroy_fn) {
            unsafe {
                let _ = destroy_fn(ext_mem);
            }
        }
    }
}
