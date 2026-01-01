// ============================================================================
// Build a VkDrawIndirectCommand from the filtered ray count.
// ============================================================================
struct DrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> in_count;
[[vk::binding(0, 1)]]
RWStructuredBuffer<DrawIndirectCommand> out_draw;

[shader("compute")]
[numthreads(1, 1, 1)]
void compMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x > 0) return;
    const uint rays = in_count[0];

    DrawIndirectCommand cmd;
    cmd.vertexCount   = rays * 2u;
    cmd.instanceCount = 1u;
    cmd.firstVertex   = 0u;
    cmd.firstInstance = 0u;
    out_draw[0] = cmd;
}
