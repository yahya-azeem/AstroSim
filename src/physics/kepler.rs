use nalgebra::Vector3;

#[derive(Clone, Copy)]
pub struct KeplerElements {
    pub a0: f64, pub a_dot: f64,
    pub e0: f64, pub e_dot: f64,
    pub i0: f64, pub i_dot: f64,
    pub L0: f64, pub L_dot: f64,
    pub varpi0: f64, pub varpi_dot: f64,
    pub omega0: f64, pub omega_dot: f64,
}

pub const PLANET_ELEMENTS: [KeplerElements; 8] = [
    // Mercury
    KeplerElements {
        a0: 0.38709927, a_dot: 0.00000037,
        e0: 0.20563593, e_dot: 0.00001906,
        i0: 7.00497902, i_dot: -0.00594749,
        L0: 252.25032350, L_dot: 149472.67411175,
        varpi0: 77.45779628, varpi_dot: 0.16047689,
        omega0: 48.33076593, omega_dot: -0.12534081,
    },
    // Venus
    KeplerElements {
        a0: 0.72333566, a_dot: 0.00000390,
        e0: 0.00677672, e_dot: -0.00004107,
        i0: 3.39467605, i_dot: -0.00078890,
        L0: 181.97909950, L_dot: 58517.81538729,
        varpi0: 131.60246718, varpi_dot: 0.00268329,
        omega0: 76.67984255, omega_dot: -0.27769418,
    },
    // Earth-Moon Barycenter (representing Earth)
    KeplerElements {
        a0: 1.00000261, a_dot: 0.00000562,
        e0: 0.01671123, e_dot: -0.00004392,
        i0: -0.00001531, i_dot: -0.01294668,
        L0: 100.46457166, L_dot: 35999.37244981,
        varpi0: 102.93768193, varpi_dot: 0.32327364,
        omega0: 0.0, omega_dot: 0.0,
    },
    // Mars
    KeplerElements {
        a0: 1.52371034, a_dot: 0.00001847,
        e0: 0.09339410, e_dot: 0.00007882,
        i0: 1.84969142, i_dot: -0.00813131,
        L0: -4.55343205, L_dot: 19140.30268499,
        varpi0: -23.94362959, varpi_dot: 0.44441088,
        omega0: 49.55953891, omega_dot: -0.29257343,
    },
    // Jupiter
    KeplerElements {
        a0: 5.20288700, a_dot: -0.00011607,
        e0: 0.04838624, e_dot: -0.00013253,
        i0: 1.30439695, i_dot: -0.00183714,
        L0: 34.39644051, L_dot: 3034.74612775,
        varpi0: 14.72847983, varpi_dot: 0.21252668,
        omega0: 100.47390909, omega_dot: 0.20469106,
    },
    // Saturn
    KeplerElements {
        a0: 9.53667594, a_dot: -0.00125060,
        e0: 0.05386179, e_dot: -0.00050991,
        i0: 2.48599187, i_dot: 0.00193609,
        L0: 49.95424423, L_dot: 1222.49362201,
        varpi0: 92.59887831, varpi_dot: -0.41897216,
        omega0: 113.66242448, omega_dot: -0.28867794,
    },
    // Uranus
    KeplerElements {
        a0: 19.18916464, a_dot: -0.00196176,
        e0: 0.04725744, e_dot: -0.00004397,
        i0: 0.77263783, i_dot: -0.00242939,
        L0: 313.23810451, L_dot: 428.48202785,
        varpi0: 170.95427630, varpi_dot: 0.40805281,
        omega0: 74.01692503, omega_dot: 0.04240589,
    },
    // Neptune
    KeplerElements {
        a0: 30.06992276, a_dot: 0.00026291,
        e0: 0.00859048, e_dot: 0.00005105,
        i0: 1.77004347, i_dot: 0.00035372,
        L0: -55.12002969, L_dot: 218.45945325,
        varpi0: 44.96476227, varpi_dot: -0.32241464,
        omega0: 131.78422574, omega_dot: -0.00508664,
    },
];

pub fn get_planet_state(idx: usize, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let elements = PLANET_ELEMENTS[idx];
    let t = (jd - 2451545.0) / 36525.0; // Julian centuries since J2000.0
    
    let a = elements.a0 + elements.a_dot * t;
    let e = elements.e0 + elements.e_dot * t;
    let i = (elements.i0 + elements.i_dot * t).to_radians();
    let l = (elements.L0 + elements.L_dot * t).to_radians();
    let varpi = (elements.varpi0 + elements.varpi_dot * t).to_radians();
    let omega = (elements.omega0 + elements.omega_dot * t).to_radians();
    
    let arg_peri = varpi - omega;
    let m = l - varpi;
    
    // Normalize mean anomaly to [-PI, PI]
    let mut m = m % (2.0 * std::f64::consts::PI);
    if m < -std::f64::consts::PI {
        m += 2.0 * std::f64::consts::PI;
    } else if m > std::f64::consts::PI {
        m -= 2.0 * std::f64::consts::PI;
    }
    
    // Solve Kepler's equation
    let mut eccentric_anomaly = m;
    for _ in 0..5 {
        eccentric_anomaly = eccentric_anomaly - (eccentric_anomaly - e * eccentric_anomaly.sin() - m) / (1.0 - e * eccentric_anomaly.cos());
    }
    
    // Coordinates in the orbital plane
    let au = 1.495978707e11; // meters
    let a_m = a * au;
    
    let x_plane = a_m * (eccentric_anomaly.cos() - e);
    let y_plane = a_m * (1.0 - e * e).sqrt() * eccentric_anomaly.sin();
    
    // Rotation matrix elements
    let cos_w = arg_peri.cos();
    let sin_w = arg_peri.sin();
    let cos_lan = omega.cos();
    let sin_lan = omega.sin();
    let cos_i = i.cos();
    let sin_i = i.sin();
    
    let r11 = cos_w * cos_lan - sin_w * sin_lan * cos_i;
    let r12 = -sin_w * cos_lan - cos_w * sin_lan * cos_i;
    let r21 = cos_w * sin_lan + sin_w * cos_lan * cos_i;
    let r22 = -sin_w * sin_lan + cos_w * cos_lan * cos_i;
    let r31 = sin_w * sin_i;
    let r32 = cos_w * sin_i;
    
    let pos = Vector3::new(
        x_plane * r11 + y_plane * r12,
        x_plane * r31 + y_plane * r32,
        x_plane * r21 + y_plane * r22,
    );
    
    // Velocities
    let g = 6.67430e-11;
    let m_sun = 1.989e30;
    let n = (g * m_sun / (a_m * a_m * a_m)).sqrt();
    let r = a_m * (1.0 - e * eccentric_anomaly.cos());
    
    let vx_plane = (-a_m * a_m * n * eccentric_anomaly.sin()) / r;
    let vy_plane = (a_m * a_m * n * (1.0 - e * e).sqrt() * eccentric_anomaly.cos()) / r;
    
    let vel = Vector3::new(
        vx_plane * r11 + vy_plane * r12,
        vx_plane * r31 + vy_plane * r32,
        vx_plane * r21 + vy_plane * r22,
    );
    
    (pos, vel)
}

pub fn jd_to_calendar(jd: f64) -> (i32, i32, i32, i32, i32) {
    let jd = jd + 0.5;
    let z = jd.floor() as i64;
    let f = jd - jd.floor();
    
    let a = if z < 2299161 {
        z
    } else {
        let alpha = (((z as f64 - 1867216.25) / 36524.25).floor()) as i64;
        z + 1 + alpha - (alpha / 4)
    };
    
    let b = a + 1524;
    let c = (((b as f64 - 122.1) / 365.25).floor()) as i64;
    let d = (365.25 * c as f64).floor() as i64;
    let e = (((b - d) as f64 / 30.6001).floor()) as i64;
    
    let day = (b - d - (30.6001 * e as f64).floor() as i64) as i32;
    let month = if e < 14 { (e - 1) as i32 } else { (e - 13) as i32 };
    let year = if month > 2 { (c - 4716) as i32 } else { (c - 4715) as i32 };
    
    let time_day = f * 24.0;
    let hour = time_day.floor() as i32;
    let time_hour = (time_day - hour as f64) * 60.0;
    let minute = time_hour.floor() as i32;
    
    (year, month, day, hour, minute)
}

#[derive(Clone)]
pub struct KeplerianBody {
    pub name: String,
    pub parent_idx: usize,
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub inclination: f64,
    pub longitude_ascending_node: f64,
    pub argument_periapsis: f64,
    pub mean_anomaly_epoch: f64,
    pub epoch_jd: f64,
    pub period: f64,
    pub radius_render: f32,
    pub body_type: u32,
}

pub fn get_keplerian_body_state(
    body: &KeplerianBody,
    jd: f64,
    parent_pos: Vector3<f64>,
    parent_vel: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let dt_sec = (jd - body.epoch_jd) * 86400.0;
    let mut m = body.mean_anomaly_epoch + (2.0 * std::f64::consts::PI / body.period) * dt_sec;
    
    m = m % (2.0 * std::f64::consts::PI);
    if m < -std::f64::consts::PI {
        m += 2.0 * std::f64::consts::PI;
    } else if m > std::f64::consts::PI {
        m -= 2.0 * std::f64::consts::PI;
    }
    
    let mut eccentric_anomaly = m;
    for _ in 0..5 {
        eccentric_anomaly = eccentric_anomaly - (eccentric_anomaly - body.eccentricity * eccentric_anomaly.sin() - m) / (1.0 - body.eccentricity * eccentric_anomaly.cos());
    }
    
    let x_plane = body.semi_major_axis * (eccentric_anomaly.cos() - body.eccentricity);
    let y_plane = body.semi_major_axis * (1.0 - body.eccentricity * body.eccentricity).sqrt() * eccentric_anomaly.sin();
    
    let cos_w = body.argument_periapsis.cos();
    let sin_w = body.argument_periapsis.sin();
    let cos_lan = body.longitude_ascending_node.cos();
    let sin_lan = body.longitude_ascending_node.sin();
    let cos_i = body.inclination.cos();
    let sin_i = body.inclination.sin();
    
    let r11 = cos_w * cos_lan - sin_w * sin_lan * cos_i;
    let r12 = -sin_w * cos_lan - cos_w * sin_lan * cos_i;
    let r21 = cos_w * sin_lan + sin_w * cos_lan * cos_i;
    let r22 = -sin_w * sin_lan + cos_w * cos_lan * cos_i;
    let r31 = sin_w * sin_i;
    let r32 = cos_w * sin_i;
    
    let rel_pos = Vector3::new(
        x_plane * r11 + y_plane * r12,
        x_plane * r31 + y_plane * r32,
        x_plane * r21 + y_plane * r22,
    );
    
    let n = 2.0 * std::f64::consts::PI / body.period;
    let r = body.semi_major_axis * (1.0 - body.eccentricity * eccentric_anomaly.cos());
    
    let vx_plane = (-body.semi_major_axis * body.semi_major_axis * n * eccentric_anomaly.sin()) / r;
    let vy_plane = (body.semi_major_axis * body.semi_major_axis * n * (1.0 - body.eccentricity * body.eccentricity).sqrt() * eccentric_anomaly.cos()) / r;
    
    let rel_vel = Vector3::new(
        vx_plane * r11 + vy_plane * r12,
        vx_plane * r31 + vy_plane * r32,
        vx_plane * r21 + vy_plane * r22,
    );
    
    (parent_pos + rel_pos, parent_vel + rel_vel)
}

