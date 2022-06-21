#version 330 core

out vec4 FragColor;

uniform vec3 color;
uniform int renderTexture;
uniform vec3 viewPos;

struct Light {
	vec3 pos;
	vec3 color;
	float ambientFactor;
};

struct Material {
	sampler2D diffuse;
	float specularStrength;
	float shininess;
};

uniform Material material;
uniform Light light;

in vec2 texCoords;
in vec4 worldPos;
in vec3 normal;

uniform sampler2D depthTexUnit;

uniform mat4 lightProj;
uniform mat4 lightView;

float ndcToZeroToOne(float ndc) {
	return (ndc * 0.5) + 0.5;
}

void main() {
	vec4 objectColor = (renderTexture * texture(material.diffuse, texCoords)) + ((1 - renderTexture) * vec4(color, 1.0));

	vec3 normNormal = normalize(normal);
	vec3 normLightDir = normalize(light.pos - worldPos.xyz);
	float diffuseFactor = max(dot(normNormal, normLightDir), 0.0);
	vec4 diffuse = vec4(light.color * diffuseFactor, 1);

	vec3 reflectLightDir = reflect(-normLightDir, normNormal);
	vec3 normViewDir = normalize(viewPos - worldPos.xyz);
	float specularFactor = pow(max(dot(normViewDir, reflectLightDir), 0.0), material.shininess);
	vec4 specular = vec4(light.color * material.specularStrength * specularFactor, 1);

	vec4 ambient = vec4(light.color * light.ambientFactor, 1);

	vec4 curPosRelToLight = lightProj * lightView * worldPos;
	curPosRelToLight = curPosRelToLight / curPosRelToLight.w;
	float curLightDepth = ndcToZeroToOne(curPosRelToLight.z);
	
	vec2 tex = vec2(ndcToZeroToOne(curPosRelToLight.x), ndcToZeroToOne(curPosRelToLight.y));
	float closestToLightDist = texture(depthTexUnit, tex).x;
	if (curLightDepth < (closestToLightDist + 0.000001)) {
		FragColor = objectColor * (ambient + diffuse + specular);
	} else {
		FragColor = objectColor * ambient;
	}

}