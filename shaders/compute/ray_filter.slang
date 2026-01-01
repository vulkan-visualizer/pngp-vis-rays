// ============================================================================
// Ray filtering compute shader (Milestone 2).
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

struct FilterParams {
    uint in_count;
    uint stride;
    uint max_out;
    uint pad;
};

[[vk::push_constant]]
cbuffer PushConstants {
    FilterParams params;
};

[[vk::binding(0, 0)]]
StructuredBuffer<RayRecord> in_rays;
[[vk::binding(1, 0)]]
RWStructuredBuffer<RayRecord> out_rays;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> out_count;

[shader("compute")]
[numthreads(256, 1, 1)]
void compMain(uint3 tid : SV_DispatchThreadID) {
    const uint idx = tid.x;
    if (idx >= params.in_count) return;

    if (params.stride > 1 && (idx % params.stride) != 0) return;

    uint dst = 0;
    while (true) {
        uint expected = out_count[0];
        if (expected >= params.max_out) return;

        uint original = 0;
        InterlockedCompareExchange(out_count[0], expected, expected + 1, original);
        if (original == expected) {
            dst = expected;
            break;
        }
    }

    out_rays[dst] = in_rays[idx];
}
