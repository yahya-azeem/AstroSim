use crate::physics::PhysicsEngine;
use nalgebra::Vector3;

pub struct VulkanComputePhysicsEngine {
    positions: Vec<Vector3<f64>>,
    velocities: Vec<Vector3<f64>>,
    masses: Vec<f64>,
}

impl VulkanComputePhysicsEngine {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            masses: Vec::new(),
        }
    }

    /// Compute the EIH acceleration for each body given their positions and velocities.
    fn compute_eih_accelerations(&self, pos: &[Vector3<f64>], vel: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = pos.len();
        let mut acc = vec![Vector3::zeros(); n];
        let mut newtonian_acc = vec![Vector3::zeros(); n];

        let g = 6.67430e-11; // SI units: m^3 kg^-1 s^-2
        let c = 299792458.0; // m/s
        let c2 = c * c;

        // First pass: compute Newtonian accelerations
        for i in 0..n {
            let mut a_newt = Vector3::zeros();
            for j in 0..n {
                if i == j {
                    continue;
                }
                let r_vec = pos[j] - pos[i]; // points from i to j (n_BA)
                let r = r_vec.norm();
                if r > 1e-5 {
                    a_newt += (g * self.masses[j] / (r * r * r)) * r_vec;
                }
            }
            newtonian_acc[i] = a_newt;
        }

        // Second pass: EIH corrections
        for i in 0..n {
            let mut a_eih = newtonian_acc[i];

            let v_i = vel[i];
            let v_i_sq = v_i.norm_squared();

            let mut pn_term = Vector3::zeros();

            for j in 0..n {
                if i == j {
                    continue;
                }
                let r_vec = pos[j] - pos[i]; // points from i to j (n_BA * r)
                let r = r_vec.norm();
                if r <= 1e-5 {
                    continue;
                }

                let n_ba = r_vec / r; // unit vector pointing from i to j
                let n_ab = -n_ba;     // unit vector pointing from j to i

                let v_j = vel[j];
                let v_j_sq = v_j.norm_squared();

                // sum_{C != i} G m_C / r_iC
                let mut sum_pot_i = 0.0;
                for c in 0..n {
                    if c == i {
                        continue;
                    }
                    let r_ic = (pos[c] - pos[i]).norm();
                    if r_ic > 1e-5 {
                        sum_pot_i += g * self.masses[c] / r_ic;
                    }
                }

                // sum_{C != j} G m_C / r_jC
                let mut sum_pot_j = 0.0;
                for c in 0..n {
                    if c == j {
                        continue;
                    }
                    let r_jc = (pos[c] - pos[j]).norm();
                    if r_jc > 1e-5 {
                        sum_pot_j += g * self.masses[c] / r_jc;
                    }
                }

                let a_j_approx = newtonian_acc[j];

                let dot_vi_vj = v_i.dot(&v_j);
                let dot_nab_vj = n_ab.dot(&v_j);
                let dot_r_aj = r_vec.dot(&a_j_approx);

                let bracket = v_i_sq
                    + 2.0 * v_j_sq
                    - 4.0 * dot_vi_vj
                    - 1.5 * dot_nab_vj * dot_nab_vj
                    - 4.0 * sum_pot_i
                    - sum_pot_j
                    + 0.5 * dot_r_aj;

                let factor1 = (g * self.masses[j] / (r * r)) * n_ba * bracket;

                let dot_nab_v_diff = n_ab.dot(&(4.0 * v_i - 3.0 * v_j));
                let factor2 = (g * self.masses[j] / (r * r)) * dot_nab_v_diff * (v_i - v_j);

                pn_term += factor1 + factor2;
            }

            a_eih += (1.0 / c2) * pn_term;
            acc[i] = a_eih;
        }

        acc
    }
}

impl PhysicsEngine for VulkanComputePhysicsEngine {
    fn step(&mut self, dt: f64) {
        let n = self.positions.len();
        if n == 0 {
            return;
        }

        // RK4 integration for N-body dynamics
        // Y = [pos, vel]
        // k1 = dt * f(Y)
        // k2 = dt * f(Y + k1/2)
        // k3 = dt * f(Y + k2/2)
        // k4 = dt * f(Y + k3)
        // Y_new = Y + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

        // k1
        let pos_k1 = self.velocities.clone();
        let acc_k1 = self.compute_eih_accelerations(&self.positions, &self.velocities);

        // Y + k1/2
        let mut pos_temp = vec![Vector3::zeros(); n];
        let mut vel_temp = vec![Vector3::zeros(); n];
        for i in 0..n {
            pos_temp[i] = self.positions[i] + 0.5 * dt * pos_k1[i];
            vel_temp[i] = self.velocities[i] + 0.5 * dt * acc_k1[i];
        }

        // k2
        let pos_k2 = vel_temp.clone();
        let acc_k2 = self.compute_eih_accelerations(&pos_temp, &vel_temp);

        // Y + k2/2
        for i in 0..n {
            pos_temp[i] = self.positions[i] + 0.5 * dt * pos_k2[i];
            vel_temp[i] = self.velocities[i] + 0.5 * dt * acc_k2[i];
        }

        // k3
        let pos_k3 = vel_temp.clone();
        let acc_k3 = self.compute_eih_accelerations(&pos_temp, &vel_temp);

        // Y + k3
        for i in 0..n {
            pos_temp[i] = self.positions[i] + dt * pos_k3[i];
            vel_temp[i] = self.velocities[i] + dt * acc_k3[i];
        }

        // k4
        let pos_k4 = vel_temp.clone();
        let acc_k4 = self.compute_eih_accelerations(&pos_temp, &vel_temp);

        // Update positions and velocities
        for i in 0..n {
            self.positions[i] += (dt / 6.0) * (pos_k1[i] + 2.0 * pos_k2[i] + 2.0 * pos_k3[i] + pos_k4[i]);
            self.velocities[i] += (dt / 6.0) * (acc_k1[i] + 2.0 * acc_k2[i] + 2.0 * acc_k3[i] + acc_k4[i]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newtonian_orbit() {
        let mut engine = VulkanComputePhysicsEngine::new();
        let m1 = 1.989e30_f64; // Sun mass in kg
        let m2 = 5.972e24_f64; // Earth mass in kg

        let r = 1.496e11_f64; // 1 AU in meters
        let g = 6.67430e-11_f64;
        let v = (g * m1 / r).sqrt();

        engine.add_body(Vector3::zeros(), Vector3::zeros(), m1);
        engine.add_body(Vector3::new(r, 0.0, 0.0), Vector3::new(0.0, v, 0.0), m2);

        // Take 10 steps of 1 day each
        let dt = 86400.0;
        for _ in 0..10 {
            engine.step(dt);
        }

        let pos = engine.get_positions();
        assert!(pos[0].norm() < 1e7, "Central body drifted too much");
        let dist = (pos[1] - pos[0]).norm();
        assert!((dist - r).abs() / r < 1e-3, "Orbital radius not conserved: got {} vs expected {}", dist, r);
    }
}
