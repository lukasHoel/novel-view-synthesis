#version 330 core

// FRAGMENT SHADER
// Takes in the RGB color and uses it fully opaque

flat in vec3 colorV;
out vec4 color;

void main( )
{
    color = vec4(colorV, 1.0);
}