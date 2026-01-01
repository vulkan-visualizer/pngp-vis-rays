// ============================================================================
// Ray line rendering shader (v2 RayBaseRecord).
// ============================================================================
struct RayBaseRecordV2 {
    float3 origin;
    float pad0;
    float3 direction;
    float pad1;
    uint pixel_x;
    uint pixel_y;
    uint ray_flags;
    uint pad2;
    uint sample_offset;
    uint sample_count;
    uint result_index;
    uint pad3;
};

struct RayPush {
    float4x4 mvp;
    float line_length;
    float opacity;
    float2 pad;
};

[[vk::push_constant]]
cbuffer PushConstants {
    RayPush push;
};

[[vk::binding(0, 0)]]
StructuredBuffer<RayBaseRecordV2> rays;

struct VsOut {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VsOut vertMain(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    const uint ray_index = instance_id;
    const bool is_end = (vertex_id & 1u) == 1u;

    const RayBaseRecordV2 r = rays[ray_index];
    const float t = is_end ? push.line_length : 0.0f;
    const float3 pos = r.origin + r.direction * t;

    float3 dir = r.direction;
    const float len = length(dir);
    if (len > 1e-6f) {
        dir /= len;
    } else {
        dir = float3(0.0f, 0.0f, 0.0f);
    }

    VsOut o;
    o.position = mul(push.mvp, float4(pos, 1.0f));
    o.color = float4(0.5f * (dir + 1.0f), push.opacity);
    return o;
}

[shader("fragment")]
float4 fragMain(VsOut input) : SV_Target0 {
    return input.color;
}
