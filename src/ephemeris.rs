use anise::prelude::*;
use hifitime::Epoch;
use nalgebra::Vector3;
use anyhow::{Result, anyhow};
use log::{info, warn};

pub struct EphemerisEngine {
    almanac: Almanac,
}

impl EphemerisEngine {
    /// Create a new empty EphemerisEngine.
    pub fn new() -> Result<Self> {
        let almanac = Almanac::default();
        Ok(Self { almanac })
    }

    /// Load a SPICE kernel file (BSP or PCK) into the Almanac context.
    pub fn load_kernel(&mut self, path: &str) -> Result<()> {
        info!("Loading SPICE kernel: {}", path);
        let new_almanac = self.almanac.clone().load(path).map_err(|e| {
            anyhow!("Failed to load kernel '{}': {:?}", path, e)
        })?;
        self.almanac = new_almanac;
        Ok(())
    }

    /// Load standard Solar System datasets from a given folder if present.
    /// Files expected: de440.bsp, de440s.bsp, pck08.pca, pck11.pca.
    pub fn load_solar_system_kernels(&mut self, data_dir: &str) -> Result<()> {
        let kernels = ["de440.bsp", "de440s.bsp", "pck08.pca", "pck11.pca"];
        let mut loaded_any = false;

        for kernel in &kernels {
            let path = format!("{}/{}", data_dir, kernel);
            if std::path::Path::new(&path).exists() {
                if let Err(e) = self.load_kernel(&path) {
                    warn!("Failed to load kernel {}: {:?}", path, e);
                } else {
                    loaded_any = true;
                }
            } else {
                info!("Kernel file not found at: {} (will use default/computed states if not loaded)", path);
            }
        }

        if !loaded_any {
            warn!("No standard planetary kernels loaded from '{}'. Coordinate translations may fail if frames are not defined.", data_dir);
        }
        Ok(())
    }

    /// Validate if a transformation path exists between two NAIF IDs.
    /// Returns Ok(()) if the DAG has a valid path, or an error detailing the separation.
    pub fn validate_transformation(&self, target_id: i32, observer_id: i32) -> Result<()> {
        let target_uid = FrameUid { ephemeris_id: target_id, orientation_id: 1 };
        let observer_uid = FrameUid { ephemeris_id: observer_id, orientation_id: 1 };

        let target_frame = self.almanac.frame_info(target_uid)
            .map_err(|e| anyhow!("Target frame (NAIF ID {}) not loaded: {:?}", target_id, e))?;
        let observer_frame = self.almanac.frame_info(observer_uid)
            .map_err(|e| anyhow!("Observer frame (NAIF ID {}) not loaded: {:?}", observer_id, e))?;

        // Perform a test translation at default epoch to check path connectivity
        let test_epoch = Epoch::default();
        self.almanac.translate(target_frame, observer_frame, test_epoch, None)
            .map_err(|e| anyhow!("Disconnected frame graph between NAIF ID {} and {}: {:?}", target_id, observer_id, e))?;

        Ok(())
    }

    /// Query the position (in km) and velocity (in km/s) of a target body relative to an observer body.
    pub fn get_state(&self, target_id: i32, observer_id: i32, epoch: Epoch) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let target_uid = FrameUid { ephemeris_id: target_id, orientation_id: 1 };
        let observer_uid = FrameUid { ephemeris_id: observer_id, orientation_id: 1 };

        let target_frame = self.almanac.frame_info(target_uid)
            .map_err(|e| anyhow!("Target frame (NAIF ID {}) not loaded: {:?}", target_id, e))?;
        let observer_frame = self.almanac.frame_info(observer_uid)
            .map_err(|e| anyhow!("Observer frame (NAIF ID {}) not loaded: {:?}", observer_id, e))?;

        let state = self.almanac.translate(target_frame, observer_frame, epoch, None)
            .map_err(|e| anyhow!("Failed translation from {} to {}: {:?}", observer_id, target_id, e))?;

        Ok((state.radius_km, state.velocity_km_s))
    }
}
