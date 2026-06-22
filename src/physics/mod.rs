use nalgebra::Vector3;

pub trait PhysicsEngine {
    fn step(&mut self, dt: f64);
    fn get_positions(&self) -> &[Vector3<f64>];
    fn get_velocities(&self) -> &[Vector3<f64>];
    fn get_masses(&self) -> &[f64];
    fn add_body(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64);
    fn set_body(&mut self, index: usize, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64);
    fn clear(&mut self);
}

pub mod cuda;
pub mod vulkan_compute;
