#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aNormal;
layout (location = 4) in vec3 aSplitNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform int useSplit;

uniform float displacement;

out vec2 texCoords;
out vec4 worldPos;
out vec3 normal;

void main() {
	vec3 pos = aPos + (displacement * aNormal);
	gl_Position = projection * view * model * vec4(pos, 1.0);
	worldPos = model * vec4(pos, 1.0);

	normal = (aSplitNormal * useSplit) + ((1-useSplit) * aNormal);
	normal = mat3(transpose(inverse(model))) * normal; 
	texCoords = aTexCoords;
}