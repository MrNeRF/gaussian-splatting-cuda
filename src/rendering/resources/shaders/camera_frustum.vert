#version 430 core

// Vertex attributes
layout(location = 0) in vec3 aPos;

// Instance attributes
layout(location = 1) in mat4 aInstanceMatrix;
layout(location = 5) in vec3 aInstanceColor;

// Uniforms
uniform mat4 viewProj;
uniform float frustumScale;
uniform vec3 viewPos;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec4 vertexColor;

void main() {
    // Apply instance transformation
    vec4 worldPos = aInstanceMatrix * vec4(aPos, 1.0);
    
    // Simple normal calculation (assuming frustum faces outward)
    Normal = normalize(mat3(aInstanceMatrix) * aPos);
    
    // Pass color
    vertexColor = vec4(aInstanceColor, 0.7);
    
    // World position for lighting
    FragPos = vec3(worldPos);
    
    // Final position
    gl_Position = viewProj * worldPos;
}
