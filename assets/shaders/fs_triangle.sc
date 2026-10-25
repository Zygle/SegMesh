$input v_worldPos, v_worldNormal, v_confidence

#include <bgfx_shader.sh>

uniform vec4 u_baseColor;
uniform vec4 u_lightDir;
uniform vec4 u_lightColor;
uniform vec4 u_cameraPos;
uniform vec4 u_material;
uniform vec4 u_drawParams;

void main()
{
    vec3 N = normalize(v_worldNormal);
    vec3 L = normalize(-u_lightDir.xyz);
    vec3 V = normalize(u_cameraPos.xyz - v_worldPos);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), u_material.z) * u_material.y;

    vec3 ambient = u_baseColor.rgb * u_material.x;
    vec3 diffuse = u_baseColor.rgb * ndotl * u_lightColor.rgb * u_lightDir.w * u_material.w;
    vec3 specular = spec * u_lightColor.rgb;

    vec3 linearColor = ambient + diffuse + specular;
    vec3 toneMapped = linearColor / (linearColor + vec3(1.0));
    vec3 srgb = pow(toneMapped, vec3(1.0 / 2.2));
    // Spread lower probabilities out: dense seeding often produces values well
    // below one even inside a region. Keep the ramp monotonic and hue-based.
    float certainty = sqrt(clamp(v_confidence.x, 0.0, 1.0));
    vec3 paleColor = mix(srgb, vec3(1.0), 0.65);
    vec3 strongColor = srgb * 0.45;
    vec3 confidenceColor = mix(paleColor, strongColor, certainty);
    vec3 contrastedColor = clamp(srgb + (confidenceColor - srgb) * u_drawParams.z, 0.0, 1.0);

    gl_FragColor = vec4(mix(srgb, contrastedColor, u_drawParams.y), u_baseColor.a);
}
