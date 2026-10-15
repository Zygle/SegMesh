vec3 v_worldPos    : TEXCOORD0 = vec3(0.0, 0.0, 0.0);
vec3 v_worldNormal : TEXCOORD1 = vec3(0.0, 1.0, 0.0);
vec4 v_confidence  : TEXCOORD2 = vec4(1.0, 1.0, 1.0, 1.0);

vec3 a_position : POSITION;
vec4 a_normal   : NORMAL;
vec4 a_texcoord0 : TEXCOORD0;
