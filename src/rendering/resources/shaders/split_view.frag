#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D leftTexture;
uniform sampler2D rightTexture;
uniform float splitPosition;
uniform float showDivider;
uniform vec4 dividerColor;
uniform float dividerWidth;

void main() {
    // Check if we're on the divider
    if (showDivider > 0.5 && abs(TexCoord.x - splitPosition) < dividerWidth) {
        FragColor = dividerColor;
    } else if (TexCoord.x < splitPosition) {
        // Sample left texture
        FragColor = texture(leftTexture, TexCoord);
    } else {
        // Sample right texture
        FragColor = texture(rightTexture, TexCoord);
    }

    // Ensure alpha is 1.0 for proper display
    FragColor.a = 1.0;
}