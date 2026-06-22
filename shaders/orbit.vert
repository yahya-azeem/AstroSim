#version 450

layout(location = 0) in vec2 inPos; // Flat coordinates (x, z) of the orbit path

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

layout(push_constant) uniform PushConstants {
    mat4 model; // Identity matrix
    vec4 color;
} pcs;

layout(location = 0) out float outHeight;

void main() {
    float depth = 0.0;

    // Sum the gravitational potential wells from all active bodies
    for (int i = 0; i < ubo.numBodies; i++) {
        vec3 bodyPos = ubo.bodies[i].pos_mass.xyz;
        float strength = ubo.bodies[i].pos_mass.w;
        
        float d = distance(vec3(inPos.x, 0.0, inPos.y), vec3(bodyPos.x, 0.0, bodyPos.z));
        
        // Warping formula matching grid.vert (wider smoothing factor)
        depth -= (strength * 2.5) / (d + 0.45);
    }

    vec3 pos = vec3(inPos.x, depth, inPos.y);
    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(pos, 1.0);
    outHeight = depth;
}
