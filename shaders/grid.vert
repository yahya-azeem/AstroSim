#version 450

layout(location = 0) in vec2 inPos; // Flat grid input coordinates (x, y) mapped to (x, z) in 3D

// Max 10 bodies (Sun + 8 planets + extra/black hole)
struct Body {
    vec4 pos_mass; // xyz = position in render units, w = mass/warping strength
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
    Body bodies[10];
    int numBodies;
    float time;
} ubo;

layout(location = 0) out vec2 outUV;
layout(location = 1) out float outHeight;

void main() {
    float depth = 0.0;

    // Sum the gravitational potential wells from all active bodies
    for (int i = 0; i < ubo.numBodies; i++) {
        vec3 bodyPos = ubo.bodies[i].pos_mass.xyz; // Position in XZ plane (y=0)
        float strength = ubo.bodies[i].pos_mass.w;
        
        // Distance in the XZ plane from grid vertex to body
        float d = distance(vec3(inPos.x, 0.0, inPos.y), vec3(bodyPos.x, 0.0, bodyPos.z));
        
        // Accumulate depth (negative Y) using a smoothed potential well (wider smoothing factor)
        depth -= (strength * 2.5) / (d + 0.45);
    }

    // Grid is horizontal in XZ plane, warped downwards along the Y axis
    vec3 pos = vec3(inPos.x, depth, inPos.y);
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);

    outUV = inPos * 0.1;
    outHeight = depth;
}
