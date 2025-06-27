precision mediump float; 

uniform vec2 u_resolution;

float plot(vec2 st) {
    // 含义是<0时候为1，>0.02的时候为0，0到0.02之间平滑插值
    // 这样只有在st.x和st.y非常接近时候权重才为1，否则为0
    return smoothstep(0.02, 0.0, abs(st.x - st.y));
}

void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    float y = st.x;
    vec3 color = vec3(y);
    vec3 line_color = vec3(1.0, 0.0, 0.0);

    float pct = plot(st);
    color = (1.0 - pct) * color + pct * line_color;
    gl_FragColor = vec4(color, 1.0);
}