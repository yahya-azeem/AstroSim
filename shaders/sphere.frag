#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in flat uint inBodyType;
layout(location = 4) in flat uint inIsSelected;
layout(location = 5) in vec3 inViewDir;

layout(location = 0) out vec4 outColor;

// Simple hash/noise functions for procedural textures
float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.1, 0.1));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    return mix(mix(mix(hash(i+vec3(0,0,0)), hash(i+vec3(1,0,0)), f.x),
                   mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
                   mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    vec3 shift = vec3(100.0);
    for (int i = 0; i < 4; ++i) {
        v += a * noise(p);
        p = p * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

void main() {
    // Normal lighting
    vec3 N = normalize(inNormal);
    // Simple directional light (from top-left-front)
    vec3 L = normalize(vec3(1.0, 1.5, 1.0));
    float diff = max(dot(N, L), 0.0);
    
    // Ambient lighting
    float ambient = 0.12;

    vec3 albedo = vec3(0.8);
    float glow = 0.0;
    float ring_alpha = 1.0;

    // Procedural color based on inBodyType
    if (inBodyType == 0) { // Sun
        // Glowing sun texture
        float n = fbm(N * 8.0);
        albedo = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.9, 0.1), n);
        glow = 1.0; // Self-luminous
    }
    else if (inBodyType == 1) { // Mercury
        // Cratered gray look
        float n = fbm(N * 16.0);
        albedo = vec3(0.5 + 0.2 * n);
    }
    else if (inBodyType == 2) { // Venus
        // Thick yellow-orange clouds
        float n = fbm(N * 6.0);
        albedo = mix(vec3(0.85, 0.7, 0.45), vec3(0.95, 0.85, 0.6), n);
    }
    else if (inBodyType == 3) { // Earth
        // Continents, oceans, and clouds
        float n = fbm(N * 10.0);
        float clouds = fbm(N * 14.0 + vec3(1.2, 0.0, 0.5));
        
        // Ocean vs Land
        if (n > 0.46) {
            // Land (green/brown)
            albedo = mix(vec3(0.2, 0.5, 0.25), vec3(0.4, 0.35, 0.25), (n - 0.46) * 4.0);
        } else {
            // Ocean (blue)
            albedo = vec3(0.08, 0.25, 0.65);
        }
        
        // Add clouds (white)
        if (clouds > 0.55) {
            albedo = mix(albedo, vec3(0.95), (clouds - 0.55) * 2.0);
        }
    }
    else if (inBodyType == 4) { // Mars
        // Red planet with polar caps
        float n = fbm(N * 12.0);
        albedo = mix(vec3(0.68, 0.28, 0.15), vec3(0.8, 0.45, 0.25), n);
        
        // Polar caps (Y close to 1.0 or -1.0)
        if (abs(N.y) > 0.88) {
            float cap_noise = fbm(N * 20.0);
            if (abs(N.y) + 0.05 * cap_noise > 0.90) {
                albedo = mix(albedo, vec3(0.95, 0.95, 1.0), 0.9);
            }
        }
    }
    else if (inBodyType == 5) { // Jupiter
        // Orange & white bands
        float band = sin(N.y * 30.0 + fbm(N * 5.0) * 3.0) * 0.5 + 0.5;
        albedo = mix(vec3(0.8, 0.55, 0.35), vec3(0.9, 0.85, 0.75), band);
        
        // Add a red spot
        float dist_to_spot = distance(N, normalize(vec3(0.8, -0.2, 0.6)));
        if (dist_to_spot < 0.15) {
            float spot_factor = smoothstep(0.15, 0.08, dist_to_spot);
            albedo = mix(albedo, vec3(0.55, 0.12, 0.08), spot_factor);
        }
    }
    else if (inBodyType == 6) { // Saturn
        // Gold/beige bands
        float band = sin(N.y * 22.0 + fbm(N * 4.0) * 2.0) * 0.5 + 0.5;
        albedo = mix(vec3(0.85, 0.75, 0.55), vec3(0.92, 0.88, 0.78), band);
    }
    else if (inBodyType == 7) { // Uranus
        // Pale cyan
        float n = fbm(N * 4.0);
        albedo = mix(vec3(0.55, 0.8, 0.85), vec3(0.65, 0.85, 0.9), n);
    }
    else if (inBodyType == 8) { // Neptune
        // Deep blue
        float n = fbm(N * 6.0);
        albedo = mix(vec3(0.15, 0.32, 0.75), vec3(0.2, 0.45, 0.85), n);
    }
    else if (inBodyType == 9) { // Saturn Rings (flat disc)
        // Rings with concentric gap lines
        float r_uv = inUV.x;
        float ring_pattern = sin(r_uv * 120.0) * 0.5 + 0.5;
        
        ring_alpha = 0.8;
        if (r_uv > 0.35 && r_uv < 0.42) ring_alpha = 0.1; // Cassini division
        if (r_uv > 0.75 && r_uv < 0.78) ring_alpha = 0.2; // Encke gap
        
        albedo = mix(vec3(0.65, 0.55, 0.45), vec3(0.85, 0.78, 0.68), ring_pattern);
        outColor = vec4(albedo * (diff + 0.2), ring_alpha);
        return;
    }
    else if (inBodyType == 100) { // Generic Exoplanet Star
        float n = fbm(N * 6.0);
        albedo = mix(vec3(0.9, 0.3, 0.1), vec3(1.0, 0.85, 0.45), n);
        glow = 1.0;
    }
    else { // Generic Exoplanet Planet (type 101)
        float n = fbm(N * 8.0);
        albedo = mix(vec3(0.4, 0.5, 0.65), vec3(0.55, 0.65, 0.8), n);
    }

    vec3 finalColor = albedo * (diff + ambient);
    if (glow > 0.0) {
        finalColor = albedo;
    }

    // Calculate view-dependent rim factor
    vec3 V = normalize(inViewDir);
    float rim = 1.0 - max(dot(N, V), 0.0);

    // Wider and brighter atmospheric/rim glow so it's easily visible from far away
    float rimGlow = pow(rim, 1.8) * 0.7;
    finalColor += albedo * rimGlow;

    // Add selection outline trace/glow
    if (inIsSelected == 1) {
        // Outline trace: thicker and brighter edge-aligned highlight
        float edge = smoothstep(0.1, 1.0, rim);
        finalColor += vec3(0.0, 0.8, 1.0) * edge * 1.0; // Cyber-blue highlight
    }
    else if (inIsSelected == 2) {
        // Hovered: very powerful atmospheric rim glow combining planet color and a bright white-blue tint
        float edge = pow(rim, 1.2);
        finalColor += albedo * edge * 3.5 + vec3(0.8, 0.9, 1.0) * pow(rim, 3.0) * 1.5;
    }

    outColor = vec4(finalColor, 1.0);
}
