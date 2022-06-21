#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 3) in vec3 aNormal;
layout (location = 4) in vec3 aSplitNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float displacement;

void main() {
	vec3 pos = aPos + (displacement * aNormal);
	gl_Position = projection * view * model * vec4(pos, 1.0);
}