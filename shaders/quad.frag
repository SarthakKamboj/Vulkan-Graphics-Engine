#version 330 core

in vec2 tex;

uniform sampler2D texUnit;

out vec4 fragColor;

void main() {
	fragColor = texture(texUnit, tex);
}