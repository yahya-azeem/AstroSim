#version 450

layout(location = 0) in vec3 outViewDir;
layout(location = 0) out vec4 outColor;

// Simple hash function for stars
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// 3D Value Noise
float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

// Fractal Brownian Motion for Nebula details
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
    vec3 dir = normalize(outViewDir);

    // 1. Generate procedural stars
    // We scale the input coordinates to create grid cells
    vec3 starGrid = dir * 250.0;
    float starStrength = hash(floor(starGrid));
    vec3 starRGB = vec3(0.0);
    
    // Twinkling effect: stars are drawn when their hash is very high
    if (starStrength > 0.994) {
        float f = hash(floor(starGrid) + vec3(0.5));
        // Star size and intensity: sharp falloff
        float starIntensity = pow(f, 15.0) * 1.5;
        
        // Slightly color the stars (blueish, yellowish, white)
        float colHash = hash(floor(starGrid) + vec3(0.2));
        vec3 starCol = vec3(1.0);
        if (colHash < 0.3) {
            starCol = vec3(0.8, 0.9, 1.0); // blue
        } else if (colHash > 0.75) {
            starCol = vec3(1.0, 0.9, 0.7); // yellow-orange
        }
        starRGB = starCol * starIntensity;
    }
    
    // 2. Cosmic Nebulas
    float n1 = fbm(dir * 2.5 + vec3(1.2));
    float n2 = fbm(dir * 3.5 - vec3(5.7));
    
    vec3 nebula1 = vec3(0.04, 0.015, 0.08) * n1;
    vec3 nebula2 = vec3(0.01, 0.03, 0.06) * n2;
    
    // 3. Milky Way Band (diagonal glow)
    // Core band equation: close to a diagonal plane
    float milkyWayBand = smoothstep(0.45, 0.0, abs(dir.y + 0.4 * dir.x - 0.2 * dir.z));
    float mwNoise = fbm(dir * 6.0);
    vec3 milkyWay = vec3(0.12, 0.08, 0.15) * milkyWayBand * (mwNoise + 0.3);
    
    // Glowing center core region
    float coreGlow = smoothstep(0.8, 0.0, distance(dir, vec3(0.6, -0.2, -0.7)));
    vec3 core = vec3(0.25, 0.15, 0.1) * coreGlow * (fbm(dir * 8.0) * 0.7 + 0.3);
    
    vec3 finalColor = starRGB + nebula1 + nebula2 + milkyWay + core;
    
    // Tone mapping and clamp
    finalColor = 1.0 - exp(-finalColor * 1.5);
    
    outColor = vec4(finalColor, 1.0);
}
