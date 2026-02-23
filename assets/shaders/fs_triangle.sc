$input v_worldPos, v_worldNormal

#include <bgfx_shader.sh>

uniform vec4 u_baseColor;
uniform vec4 u_lightDir;
uniform vec4 u_lightColor;
uniform vec4 u_cameraPos;
uniform vec4 u_material;

void main()
{
    vec3 N = normalize(v_worldNormal);
    vec3 L = normalize(-u_lightDir.xyz);
    vec3 V = normalize(u_cameraPos.xyz - v_worldPos);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), u_material.z) * u_material.y;

    vec3 ambient = u_baseColor.rgb * u_material.x;
    vec3 diffuse = u_baseColor.rgb * ndotl * u_lightColor.rgb * u_lightDir.w;
    vec3 specular = spec * u_lightColor.rgb;

    vec3 linearColor = ambient + diffuse + specular;
    vec3 toneMapped = linearColor / (linearColor + vec3(1.0));
    vec3 srgb = pow(toneMapped, vec3(1.0 / 2.2));

    gl_FragColor = vec4(srgb, u_baseColor.a);
}
