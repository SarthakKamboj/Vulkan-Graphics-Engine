#version 330 core

uniform sampler2D depthTexUnit;
uniform int extraVisible;

in vec2 tex;

out vec4 FragColor;

void main() {
	float val = (extraVisible * pow(texture(depthTexUnit, tex).x, 32)) + ((1-extraVisible) * texture(depthTexUnit, tex).x);
	FragColor = vec4(val, val, val, 1.0);
}