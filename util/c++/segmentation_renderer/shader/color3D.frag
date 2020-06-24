#version 330 core

// FRAGMENT SHADER
// Takes in the RGB color and uses it fully opaque

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

flat in vec3 colorV;
out vec4 color;

float near = 0.1; 
float far  = 10.0; 
  
float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main( )
{
    // USE THIS TO DRAW THE REAL COLORS TO IMAGE
    // color = vec4(colorV, 1.0);

    // USE THIS TO DRAW THE LINEARIZED DEPTH TO IMAGE
    // float depth = LinearizeDepth(gl_FragCoord.z) / far; // divide by far for demonstration
    // color = vec4(vec3(depth), 1.0);

    vec4 ndc = vec4(gl_FragCoord.xyz * 2.0 - 1.0, 1.0);
    vec4 view_space = inverse(projection) * ndc;

    // Perspective division
    view_space /= view_space.w;

    vec4 world_space = inverse(view) * view_space;

    color = vec4(vec3(world_space.z), 1.0);
}