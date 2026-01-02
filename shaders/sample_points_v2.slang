// ============================================================================
// Sample point visualization shader (v2).
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

struct SampleRecordV2 {
    float t;
    float dt;
    uint level_mip;
    uint state_omit;
    uint ray_index;
    uint sample_flags;
    uint rng_seed;
    uint pad1;
};

struct SampleEvalV2 {
    float4 density_rgb;
    float4 weight_trans_contrib;
    float4 contrib_b_pad;
};

struct SamplePush {
    float4x4 mvp;
    float4 params;
    uint mode;
    uint stride;
    uint2 pad;
};

[[vk::push_constant]]
cbuffer PushConstants {
    SamplePush push;
};

[[vk::binding(0, 0)]]
StructuredBuffer<RayBaseRecordV2> rays;
[[vk::binding(1, 0)]]
StructuredBuffer<SampleRecordV2> samples;
[[vk::binding(2, 0)]]
StructuredBuffer<SampleEvalV2> evals;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> sample_indices;

struct VsOut {
    float4 position : SV_Position;
    float4 color : COLOR0;
    float point_size : PSIZE;
};

float3 state_color(uint state) {
    if (state == 0u) return float3(0.7f, 0.7f, 0.7f);
    if (state == 1u) return float3(0.2f, 0.9f, 0.2f);
    if (state == 2u) return float3(0.9f, 0.2f, 0.2f);
    return float3(0.9f, 0.8f, 0.2f);
}

float3 omit_color(uint reason) {
    if (reason == 0u) return float3(0.7f, 0.7f, 0.7f);
    if (reason == 1u) return float3(0.3f, 0.6f, 0.9f);
    if (reason == 2u) return float3(0.9f, 0.6f, 0.2f);
    if (reason == 3u) return float3(0.6f, 0.3f, 0.9f);
    if (reason == 4u) return float3(0.9f, 0.3f, 0.6f);
    if (reason == 5u) return float3(0.3f, 0.9f, 0.6f);
    if (reason == 6u) return float3(0.9f, 0.2f, 0.8f);
    return float3(0.8f, 0.8f, 0.8f);
}

[shader("vertex")]
VsOut vertMain(uint vertex_id : SV_VertexID) {
    const uint idx = vertex_id;
    const bool keep = (push.stride <= 1u) || ((idx % push.stride) == 0u);

    VsOut o;
    if (!keep) {
        o.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o.color = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o.point_size = 0.0f;
        return o;
    }

    const uint sample_index = sample_indices[idx];
    const SampleRecordV2 s = samples[sample_index];
    const RayBaseRecordV2 r = rays[s.ray_index];
    const float3 pos = r.origin + r.direction * s.t;

    const uint state = s.state_omit & 0xFFu;
    const uint omit = (s.state_omit >> 8u) & 0xFFu;
    const SampleEvalV2 e = evals[sample_index];

    float3 color = state_color(state);
    if (push.mode == 1u) {
        color = omit_color(omit);
    } else if (push.mode == 2u) {
        const float density = e.density_rgb.x * push.params.y;
        color = saturate(float3(density, density, density));
    } else if (push.mode == 3u) {
        const float weight = e.weight_trans_contrib.x * push.params.z;
        color = saturate(float3(weight, weight, weight));
    } else if (push.mode == 4u) {
        const float3 contrib = float3(e.weight_trans_contrib.z, e.weight_trans_contrib.w, e.contrib_b_pad.x);
        color = saturate(contrib * push.params.w);
    }

    o.position = mul(push.mvp, float4(pos, 1.0f));
    o.color = float4(color, 1.0f);
    o.point_size = push.params.x;
    return o;
}

[shader("fragment")]
float4 fragMain(VsOut input) : SV_Target0 {
    return input.color;
}
