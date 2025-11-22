#version 460
#extension GL_EXT_ray_tracing : require

hitAttributeEXT vec3 attribs;   // intersection shader에서 넘긴 hit attribute
layout(location = 0) rayPayloadInEXT vec3 rayPayload;

void main()
{
    vec3 normalColor = attribs * 0.5 + 0.5;
    rayPayload = normalColor;
}