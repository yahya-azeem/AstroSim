#version 450

layout(location = 0) in vec2 outUV;
layout(location = 1) in float outHeight;

layout(location = 0) out vec4 outColor;

void main() {
    // Height-responsive color transition: cyan-green in flat space, deep blue in wells
    float depth = clamp(-outHeight * 0.4, 0.0, 1.0);
    
    vec3 base_color = vec3(0.0, 0.9, 0.7); // Cyan-green
    vec3 deep_color = vec3(0.1, 0.3, 1.0); // Blue
    vec3 grid_color = mix(base_color, deep_color, depth);

    // Glowing brightness increases in deep wells
    float alpha = mix(0.5, 0.95, depth);
    outColor = vec4(grid_color, alpha);
}
