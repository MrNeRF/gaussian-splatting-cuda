#version 430 core

// Vertex attributes
layout(location = 0) in vec3 aPos;

// Instance attributes
layout(location = 1) in mat4 aInstanceMatrix;
layout(location = 5) in vec4 aInstanceColorAlpha;  // Changed to vec4 to include alpha

// Uniforms
uniform mat4 viewProj;
uniform vec3 viewPos;
uniform bool pickingMode = false;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec4 vertexColor;
flat out int instanceID;

void main() {
    // Pass instance ID (gl_InstanceID is the actual instance index)
    instanceID = gl_InstanceID;
    
    // Apply instance transformation
    vec4 worldPos = aInstanceMatrix * vec4(aPos, 1.0);
    
    // Simple normal calculation (assuming frustum faces outward)
    Normal = normalize(mat3(aInstanceMatrix) * aPos);
    
    // Pass color with alpha
    vertexColor = aInstanceColorAlpha;
    
    // World position for lighting
    FragPos = vec3(worldPos);
    
    // Final position
    gl_Position = viewProj * worldPos;
    
    // In picking mode, encode instance ID as color
    if (pickingMode) {
        // Encode instance ID + 1 (to avoid 0) as RGB
        int id = gl_InstanceID + 1;
        float r = float((id >> 16) & 0xFF) / 255.0;
        float g = float((id >> 8) & 0xFF) / 255.0;
        float b = float(id & 0xFF) / 255.0;
        vertexColor = vec4(r, g, b, 1.0);
    }
}
