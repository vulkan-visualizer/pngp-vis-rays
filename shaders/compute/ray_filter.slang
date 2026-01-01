// ============================================================================
// Ray filtering compute shader (prefix-sum compaction).
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
    uint flags;
    uint roi_min_x;
    uint roi_min_y;
    uint roi_max_x;
    uint roi_max_y;
    uint sample_state_mask;
    uint omit_reason_mask;
    uint mask_id;
    uint batch_id;
};

static const uint k_filter_roi = 1u << 0;
static const uint k_filter_sample = 1u << 1;
static const uint k_filter_mask_id = 1u << 2;
static const uint k_filter_batch_id = 1u << 3;

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
[[vk::binding(3, 0)]]
StructuredBuffer<uint> sample_records;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> ray_mask_id;
[[vk::binding(5, 0)]]
StructuredBuffer<uint> ray_batch_id;

groupshared uint s_flags[256];
groupshared uint s_scan[256];
groupshared uint s_group_base;
groupshared uint s_group_allow;

[shader("compute")]
[numthreads(256, 1, 1)]
void compMain(uint3 tid : SV_DispatchThreadID,
              uint3 group_id : SV_GroupID,
              uint3 local_id : SV_GroupThreadID) {
    const uint idx = tid.x;
    bool keep = idx < params.in_count;
    if (keep && params.stride > 1 && (idx % params.stride) != 0) keep = false;

    if (keep && (params.flags & k_filter_roi) != 0u) {
        const RayRecord r = in_rays[idx];
        if (r.pixel_x < params.roi_min_x || r.pixel_x > params.roi_max_x ||
            r.pixel_y < params.roi_min_y || r.pixel_y > params.roi_max_y) {
            keep = false;
        }
    }

    if (keep && (params.flags & k_filter_mask_id) != 0u) {
        if (ray_mask_id[idx] != params.mask_id) keep = false;
    }

    if (keep && (params.flags & k_filter_batch_id) != 0u) {
        if (ray_batch_id[idx] != params.batch_id) keep = false;
    }

    const uint local = local_id.x;
    s_flags[local] = keep ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    s_scan[local] = s_flags[local];
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        const uint add = (local >= offset) ? s_scan[local - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_scan[local] += add;
        GroupMemoryBarrierWithGroupSync();
    }

    const uint inclusive = s_scan[local];
    const uint exclusive = inclusive - s_flags[local];
    const uint group_total = s_scan[255];

    if (local == 0u) {
        uint expected = 0u;
        uint desired = 0u;
        uint original = 0u;
        do {
            expected = out_count[0];
            desired = min(expected + group_total, params.max_out);
            InterlockedCompareExchange(out_count[0], expected, desired, original);
        } while (original != expected);

        s_group_base = expected;
        s_group_allow = desired - expected;
    }

    GroupMemoryBarrierWithGroupSync();
    if (keep && exclusive < s_group_allow) {
        const uint out_idx = s_group_base + exclusive;
        out_rays[out_idx] = in_rays[idx];
    }
}
