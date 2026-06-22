use crate::physics::PhysicsEngine;
use nalgebra::Vector3;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use log::info;

pub struct CudaPhysicsEngine {
    device: Arc<CudaDevice>,
    positions: Vec<Vector3<f64>>,
    velocities: Vec<Vector3<f64>>,
    masses: Vec<f64>,
}

impl CudaPhysicsEngine {
    /// Attempts to initialize the CUDA device and compile the EIH kernel.
    pub fn try_new() -> Result<Self> {
        info!("Pre-flight check: attempting to locate CUDA shared library...");
        let _lib = unsafe {
            libloading::Library::new("libcuda.so")
                .or_else(|_| libloading::Library::new("libcuda.so.1"))
                .map_err(|e| anyhow!("CUDA shared library (libcuda.so) not found: {:?}", e))?
        };

        info!("Attempting to initialize CUDA device 0...");
        let device = CudaDevice::new(0)
            .map_err(|e| anyhow!("Failed to initialize CUDA device 0: {:?}", e))?;
            
        info!("Compiling EIH CUDA kernel at runtime via NVRTC...");
        let cuda_src = include_str!("../../kernels/eih.cu");
        let ptx = cudarc::nvrtc::compile_ptx(cuda_src)
            .map_err(|e| anyhow!("Failed to compile EIH kernel: {:?}", e))?;
            
        device.load_ptx(ptx, "eih_module", &["integrate_eih"])
            .map_err(|e| anyhow!("Failed to load EIH module into device: {:?}", e))?;
            
        info!("CUDA physics solver successfully initialized.");
        Ok(Self {
            device,
            positions: Vec::new(),
            velocities: Vec::new(),
            masses: Vec::new(),
        })
    }
}

impl PhysicsEngine for CudaPhysicsEngine {
    fn step(&mut self, dt: f64) {
        let n = self.positions.len();
        if n == 0 {
            return;
        }
        
        // Flatten vectors for GPU transfer
        let mut flat_positions = Vec::with_capacity(3 * n);
        let mut flat_velocities = Vec::with_capacity(3 * n);
        for i in 0..n {
            flat_positions.push(self.positions[i].x);
            flat_positions.push(self.positions[i].y);
            flat_positions.push(self.positions[i].z);
            
            flat_velocities.push(self.velocities[i].x);
            flat_velocities.push(self.velocities[i].y);
            flat_velocities.push(self.velocities[i].z);
        }
        
        // Copy data to GPU
        let mut pos_dev = self.device.htod_copy(flat_positions).expect("CUDA htod copy pos failed");
        let mut vel_dev = self.device.htod_copy(flat_velocities).expect("CUDA htod copy vel failed");
        let mass_dev = self.device.htod_copy(self.masses.clone()).expect("CUDA htod copy mass failed");
        
        let func = self.device.get_func("eih_module", "integrate_eih").expect("CUDA EIH function not found");
        
        // Configure block and grid dimensions
        let threads_per_block = 256;
        let blocks = ((n + threads_per_block - 1) / threads_per_block) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch kernel
        unsafe {
            func.launch(cfg, (&mut pos_dev, &mut vel_dev, &mass_dev, dt, n as i32))
                .expect("Failed to launch CUDA EIH kernel");
        }
        
        // Sync copy data back from GPU
        let updated_positions = self.device.dtoh_sync_copy(&pos_dev).expect("CUDA dtoh copy pos failed");
        let updated_velocities = self.device.dtoh_sync_copy(&vel_dev).expect("CUDA dtoh copy vel failed");
        
        // Unflatten vectors back into host storage
        for i in 0..n {
            self.positions[i] = Vector3::new(updated_positions[3*i], updated_positions[3*i+1], updated_positions[3*i+2]);
            self.velocities[i] = Vector3::new(updated_velocities[3*i], updated_velocities[3*i+1], updated_velocities[3*i+2]);
        }
    }

    fn get_positions(&self) -> &[Vector3<f64>] {
        &self.positions
    }

    fn get_velocities(&self) -> &[Vector3<f64>] {
        &self.velocities
    }

    fn get_masses(&self) -> &[f64] {
        &self.masses
    }

    fn add_body(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64) {
        self.positions.push(position);
        self.velocities.push(velocity);
        self.masses.push(mass);
    }

    fn set_body(&mut self, index: usize, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64) {
        if index < self.positions.len() {
            self.positions[index] = position;
            self.velocities[index] = velocity;
            self.masses[index] = mass;
        }
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.masses.clear();
    }
}
