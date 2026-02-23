$input a_position, a_normal
$output v_worldPos, v_worldNormal

#include <bgfx_shader.sh>

void main()
{
    vec4 worldPos = mul(u_model[0], vec4(a_position, 1.0));
    gl_Position = mul(u_modelViewProj, vec4(a_position, 1.0));

    vec3 normal = a_normal.xyz * 2.0 - 1.0;
    v_worldPos = worldPos.xyz;
    v_worldNormal = mul(u_model[0], vec4(normal, 0.0)).xyz;
}
