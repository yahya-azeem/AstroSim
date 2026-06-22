#version 450

layout(location = 0) in float outHeight;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} pcs;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = pcs.color;
}
