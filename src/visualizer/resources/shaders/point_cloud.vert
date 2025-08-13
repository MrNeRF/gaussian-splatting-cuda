#version 430 core

layout(location = 0) in vec3 a_vertex_position;
layout(location = 1) in vec3 a_instance_position;
layout(location = 2) in vec3 a_instance_color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_voxel_size;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    // Scale and translate the cube vertex
    vec3 scaled_vertex = a_vertex_position * u_voxel_size;
    vec3 world_position = scaled_vertex + a_instance_position;
    
    // Transform to clip space
    vec4 view_pos = u_view * vec4(world_position, 1.0);
    gl_Position = u_projection * view_pos;
    
    // Pass color to fragment shader
    v_color = a_instance_color;
    
    // Calculate normal (for simple shading)
    // Since we're using cubes, we can use the vertex position as normal
    v_normal = normalize(a_vertex_position);
    v_frag_pos = view_pos.xyz;
}
