#version 450

layout(location = 0) out vec3 outViewDir;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
} ubo;

void main() {
    // Generate a full-screen triangle in NDC
    // Vertex ID: 0 -> (-1, -1), 1 -> (3, -1), 2 -> (-1, 3)
    vec2 ndc = vec2(
        (gl_VertexIndex == 1) ? 3.0 : -1.0,
        (gl_VertexIndex == 2) ? 3.0 : -1.0
    );
    gl_Position = vec4(ndc, 0.9999, 1.0); // Render at the far plane

    // Unproject to find world-space view direction
    vec4 target = ubo.invViewProj * vec4(ndc, 1.0, 1.0);
    outViewDir = target.xyz / target.w;
}
