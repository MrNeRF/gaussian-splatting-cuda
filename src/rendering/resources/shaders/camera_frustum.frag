#version 430 core

// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec4 vertexColor;
flat in int instanceID;

// Output
out vec4 FragColor;

// Uniforms
uniform vec3 viewPos;
uniform bool enableShading;
uniform int highlightIndex = -1;
uniform vec3 highlightColor = vec3(1.0, 0.85, 0.0);
uniform bool pickingMode = false;
uniform float minimumPickDistance = 0.5;

void main() {
    // Calculate distance from camera to fragment
    float distance = length(viewPos - FragPos);
    
    // In picking mode, discard fragments that are too close
    if (pickingMode) {
        if (distance < minimumPickDistance) {
            discard;  // Don't render this fragment for picking
        }
        // In picking mode, just output the encoded color
        FragColor = vertexColor;
        return;
    }
    
    // Normal rendering mode
    vec4 finalColor = vertexColor;

    // Apply highlight if this instance is selected
    if (instanceID == highlightIndex) {
        finalColor.rgb = highlightColor;
        finalColor.a = min(1.0, finalColor.a + 0.3);  // Make highlighted frustum more opaque
    }

    if (enableShading) {
        // Simple lighting
        vec3 lightDir = normalize(viewPos - FragPos);
        vec3 norm = normalize(Normal);

        // Ambient + diffuse
        float ambient = 0.3;
        float diff = max(dot(norm, lightDir), 0.0);
        float lighting = ambient + diff * 0.7;

        finalColor.rgb *= lighting;
    } else {
        // Wireframe - use the instance color but darker
        finalColor.rgb *= 0.3;
        finalColor.a = 1.0;
    }

    FragColor = finalColor;
}
