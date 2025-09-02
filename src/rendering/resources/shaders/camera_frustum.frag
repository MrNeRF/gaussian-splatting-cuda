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
uniform int highlightIndex = -1;  // Default value for when not highlighting
uniform vec3 highlightColor = vec3(1.0, 1.0, 0.0);  // Default yellow highlight

void main() {
    vec4 finalColor = vertexColor;

    // Apply highlight if this instance is selected
    if (instanceID == highlightIndex) {
        finalColor.rgb = highlightColor;
        finalColor.a = 1.0;  // Make highlighted frustum more opaque
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
        // Wireframe - darker for contrast
        finalColor = vec4(0.0, 0.0, 0.0, 1.0);
    }

    FragColor = finalColor;
}