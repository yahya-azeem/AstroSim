extern "C" __global__ void integrate_eih(
    double* positions,   // 3 * n
    double* velocities,  // 3 * n
    const double* masses,
    double dt,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double g = 6.67430e-11;
    double c = 299792458.0;
    double c2 = c * c;

    // Load body i state
    double pos_i[3] = { positions[3*i], positions[3*i+1], positions[3*i+2] };
    double vel_i[3] = { velocities[3*i], velocities[3*i+1], velocities[3*i+2] };
    double v_i_sq = vel_i[0]*vel_i[0] + vel_i[1]*vel_i[1] + vel_i[2]*vel_i[2];

    double a_newt_i[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double r_vec[3] = {
            positions[3*j] - pos_i[0],
            positions[3*j+1] - pos_i[1],
            positions[3*j+2] - pos_i[2]
        };
        double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
        double r = sqrt(r2);
        if (r > 1e-5) {
            double factor = g * masses[j] / (r2 * r);
            a_newt_i[0] += factor * r_vec[0];
            a_newt_i[1] += factor * r_vec[1];
            a_newt_i[2] += factor * r_vec[2];
        }
    }

    double pn_term[3] = {0.0, 0.0, 0.0};

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        double r_vec[3] = {
            positions[3*j] - pos_i[0],
            positions[3*j+1] - pos_i[1],
            positions[3*j+2] - pos_i[2]
        };
        double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
        double r = sqrt(r2);
        if (r <= 1e-5) continue;

        double n_ba[3] = { r_vec[0] / r, r_vec[1] / r, r_vec[2] / r };
        double n_ab[3] = { -n_ba[0], -n_ba[1], -n_ba[2] };

        double vel_j[3] = { velocities[3*j], velocities[3*j+1], velocities[3*j+2] };
        double v_j_sq = vel_j[0]*vel_j[0] + vel_j[1]*vel_j[1] + vel_j[2]*vel_j[2];

        // Potential sums
        double sum_pot_i = 0.0;
        for (int c = 0; c < n; ++c) {
            if (c == i) continue;
            double dx = positions[3*c] - pos_i[0];
            double dy = positions[3*c+1] - pos_i[1];
            double dz = positions[3*c+2] - pos_i[2];
            double r_ic = sqrt(dx*dx + dy*dy + dz*dz);
            if (r_ic > 1e-5) sum_pot_i += g * masses[c] / r_ic;
        }

        double sum_pot_j = 0.0;
        for (int c = 0; c < n; ++c) {
            if (c == j) continue;
            double dx = positions[3*c] - positions[3*j];
            double dy = positions[3*c+1] - positions[3*j+1];
            double dz = positions[3*c+2] - positions[3*j+2];
            double r_jc = sqrt(dx*dx + dy*dy + dz*dz);
            if (r_jc > 1e-5) sum_pot_j += g * masses[c] / r_jc;
        }

        // Newtonian acceleration of j (approximates a_B)
        double a_j_approx[3] = {0.0, 0.0, 0.0};
        for (int c = 0; c < n; ++c) {
            if (c == j) continue;
            double dx = positions[3*c] - positions[3*j];
            double dy = positions[3*c+1] - positions[3*j+1];
            double dz = positions[3*c+2] - positions[3*j+2];
            double r_jc2 = dx*dx + dy*dy + dz*dz;
            double r_jc = sqrt(r_jc2);
            if (r_jc > 1e-5) {
                double factor = g * masses[c] / (r_jc2 * r_jc);
                a_j_approx[0] += factor * dx;
                a_j_approx[1] += factor * dy;
                a_j_approx[2] += factor * dz;
            }
        }

        double dot_vi_vj = vel_i[0]*vel_j[0] + vel_i[1]*vel_j[1] + vel_i[2]*vel_j[2];
        double dot_nab_vj = n_ab[0]*vel_j[0] + n_ab[1]*vel_j[1] + n_ab[2]*vel_j[2];
        double dot_r_aj = r_vec[0]*a_j_approx[0] + r_vec[1]*a_j_approx[1] + r_vec[2]*a_j_approx[2];

        double bracket = v_i_sq
            + 2.0 * v_j_sq
            - 4.0 * dot_vi_vj
            - 1.5 * dot_nab_vj * dot_nab_vj
            - 4.0 * sum_pot_i
            - sum_pot_j
            + 0.5 * dot_r_aj;

        double factor1 = (g * masses[j] / r2) * bracket;
        double dot_nab_v_diff = n_ab[0]*(4.0*vel_i[0] - 3.0*vel_j[0]) 
                              + n_ab[1]*(4.0*vel_i[1] - 3.0*vel_j[1]) 
                              + n_ab[2]*(4.0*vel_i[2] - 3.0*vel_j[2]);
        double factor2 = (g * masses[j] / r2) * dot_nab_v_diff;

        pn_term[0] += factor1 * n_ba[0] + factor2 * (vel_i[0] - vel_j[0]);
        pn_term[1] += factor1 * n_ba[1] + factor2 * (vel_i[1] - vel_j[1]);
        pn_term[2] += factor1 * n_ba[2] + factor2 * (vel_i[2] - vel_j[2]);
    }

    double a_eih[3] = {
        a_newt_i[0] + (1.0 / c2) * pn_term[0],
        a_newt_i[1] + (1.0 / c2) * pn_term[1],
        a_newt_i[2] + (1.0 / c2) * pn_term[2]
    };

    // Update positions and velocities
    positions[3*i] += velocities[3*i] * dt + 0.5 * a_eih[0] * dt * dt;
    positions[3*i+1] += velocities[3*i+1] * dt + 0.5 * a_eih[1] * dt * dt;
    positions[3*i+2] += velocities[3*i+2] * dt + 0.5 * a_eih[2] * dt * dt;

    velocities[3*i] += a_eih[0] * dt;
    velocities[3*i+1] += a_eih[1] * dt;
    velocities[3*i+2] += a_eih[2] * dt;
}
