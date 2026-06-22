#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint body_type;
    uint is_selected;
} pcs;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out flat uint outBodyType;
layout(location = 4) out flat uint outIsSelected;
layout(location = 5) out vec3 outViewDir;

void main() {
    vec4 worldPos = pcs.model * vec4(inPos, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    outWorldPos = worldPos.xyz;
    outNormal = normalize(mat3(pcs.model) * inNormal);
    outUV = inUV;
    outBodyType = pcs.body_type;
    outIsSelected = pcs.is_selected;

    vec4 camPos4 = ubo.invViewProj * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 camPos = camPos4.xyz / camPos4.w;
    outViewDir = camPos - worldPos.xyz;
}
