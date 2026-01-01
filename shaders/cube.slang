struct VSInput {
    float3 position : LOCATION0;
    float4 color : LOCATION1;
};

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

struct Push {
    column_major float4x4 mvp;
};

[[vk::push_constant]]
cbuffer PushConstants {
    Push pc;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput o;
    o.position = mul(pc.mvp, float4(input.position, 1.0));
    o.color    = input.color;
    return o;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    return input.color;
}


