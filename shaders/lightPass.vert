#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aNormal;

uniform mat4 model;
uniform mat4 lightView;
uniform mat4 lightProj;

void main() {
	gl_Position = lightProj * lightView * model * vec4(aPos, 1.0);
}
