// ============================================================================
// Ray line rendering shader (line list, vertex pulling).
// ============================================================================
struct RayRecord {
    float3 origin;
    float pad0;
    float3 direction;
    float pad1;
    uint pixel_x;
    uint pixel_y;
    uint flags;
    uint pad2;
};

struct RayPush {
    float4x4 mvp;
    float4 params;
    uint mode;
    uint3 pad;
};

[[vk::push_constant]]
cbuffer PushConstants {
    RayPush push;
};

[[vk::binding(0, 0)]]
StructuredBuffer<RayRecord> rays;
[[vk::binding(1, 0)]]
StructuredBuffer<uint4> results;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> mask_ids;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> batch_ids;

struct VsOut {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

float3 hash_color(uint v) {
    uint x = v * 1664525u + 1013904223u;
    float3 rgb = float3(x & 0xFFu, (x >> 8) & 0xFFu, (x >> 16) & 0xFFu) / 255.0f;
    return rgb;
}

float3 flags_color(uint flags) {
    float3 color = float3(0.1f, 0.1f, 0.1f);
    if ((flags & 1u) != 0u) color += float3(0.2f, 0.6f, 0.2f);
    if ((flags & 2u) != 0u) color += float3(0.2f, 0.2f, 0.6f);
    if ((flags & 4u) != 0u) color += float3(0.6f, 0.2f, 0.2f);
    if ((flags & 8u) != 0u) color += float3(0.6f, 0.6f, 0.2f);
    return saturate(color);
}

// ============================================================================
// Vertex shader: build a line segment per ray using SV_VertexID.
// ============================================================================
[shader("vertex")]
VsOut vertMain(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    const uint ray_index = instance_id;
    const bool is_end = (vertex_id & 1u) == 1u;

    const RayRecord r = rays[ray_index];
    const float t = is_end ? push.params.x : 0.0f;
    const float3 pos = r.origin + r.direction * t;

    float3 dir = r.direction;
    const float len = length(dir);
    if (len > 1e-6f) {
        dir /= len;
    } else {
        dir = float3(0.0f, 0.0f, 0.0f);
    }

    float3 color = 0.5f * (dir + 1.0f);
    if (push.mode == 1u) {
        color = flags_color(r.flags);
    } else if (push.mode == 2u) {
        color = hash_color(mask_ids[ray_index]);
    } else if (push.mode == 3u) {
        color = hash_color(batch_ids[ray_index]);
    }

    VsOut o;
    o.position = mul(push.mvp, float4(pos, 1.0f));
    o.color = float4(color, push.params.y);
    return o;
}

// ============================================================================
// Fragment shader: flat color from vertex stage.
// ============================================================================
[shader("fragment")]
float4 fragMain(VsOut input) : SV_Target0 {
    return input.color;
}
