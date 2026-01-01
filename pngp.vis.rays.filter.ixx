module;
#include <vulkan/vulkan_raii.hpp>
export module pngp.vis.rays.filter;
// ============================================================================
// GPU filter/compaction pipeline for ray records (Milestone 2).
// ============================================================================
import std;
import vk.context;
import vk.pipeline;
import vk.memory;
import pngp.vis.rays.record;

namespace {
    // ========================================================================
    // Load SPIR-V from the first available path.
    // ========================================================================
    std::vector<std::byte> read_spv_bytes(std::span<const char* const> paths) {
        std::exception_ptr last_error;
        for (const char* path : paths) {
            try {
                return vk::pipeline::read_file_bytes(path);
            } catch (...) {
                last_error = std::current_exception();
            }
        }
        if (last_error) std::rethrow_exception(last_error);
        throw std::runtime_error("filter: shader not found");
    }
} // namespace

namespace pngp::vis::rays::filter {
    // ========================================================================
    // Push constants for the compute shader.
    // ========================================================================
    export struct FilterParams {
        std::uint32_t in_count = 0;
        std::uint32_t stride   = 1;
        std::uint32_t max_out  = 0;
        std::uint32_t pad      = 0;
    };

    static_assert(std::is_standard_layout_v<FilterParams>);
    static_assert(sizeof(FilterParams) == 16);

    // ========================================================================
    // Pipeline bundle for compute filtering.
    // ========================================================================
    export struct FilterPipeline {
        vk::raii::DescriptorSetLayout set_layout{nullptr};
        vk::raii::PipelineLayout layout{nullptr};
        vk::raii::Pipeline pipeline{nullptr};
    };

    export struct FilterBindings {
        vk::raii::DescriptorPool pool{nullptr};
        vk::raii::DescriptorSet set{nullptr};
    };

    // ========================================================================
    // Pipeline bundle for indirect draw command emission.
    // ========================================================================
    export struct IndirectPipeline {
        vk::raii::DescriptorSetLayout set_layout{nullptr};
        vk::raii::PipelineLayout layout{nullptr};
        vk::raii::Pipeline pipeline{nullptr};
    };

    export struct IndirectBindings {
        vk::raii::DescriptorPool pool{nullptr};
        vk::raii::DescriptorSet set{nullptr};
    };

    // ========================================================================
    // Pipeline + descriptor setup.
    // ========================================================================
    export [[nodiscard]] FilterPipeline create_filter_pipeline_from_paths(
        const vk::raii::Device& device,
        std::span<const char* const> paths) {
        const vk::DescriptorSetLayoutBinding bindings[] = {
            {
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            },
            {
                .binding         = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            },
            {
                .binding         = 2,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            },
        };

        const vk::DescriptorSetLayoutCreateInfo set_ci{
            .bindingCount = static_cast<std::uint32_t>(std::size(bindings)),
            .pBindings    = bindings,
        };

        FilterPipeline out{};
        out.set_layout = vk::raii::DescriptorSetLayout{device, set_ci};

        const vk::PushConstantRange push{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(FilterParams),
        };

        const vk::PipelineLayoutCreateInfo layout_ci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*out.set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &push,
        };

        out.layout = vk::raii::PipelineLayout{device, layout_ci};

        const auto spv = read_spv_bytes(paths);
        const auto shader = vk::pipeline::load_shader_module(device, spv);

        const vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = *shader,
            .pName  = "compMain",
        };

        const vk::ComputePipelineCreateInfo ci{
            .stage  = stage,
            .layout = *out.layout,
        };

        out.pipeline = vk::raii::Pipeline{device, nullptr, ci};
        return out;
    }

    export [[nodiscard]] FilterPipeline create_filter_pipeline(const vk::raii::Device& device,
                                                              const vk::Format) {
        constexpr std::array paths{
            "../shaders/ray_filter.spv",
            "../shaders/ray_filter.spv",
        };
        return create_filter_pipeline_from_paths(device, paths);
    }

    export [[nodiscard]] FilterBindings create_filter_bindings(const vk::raii::Device& device) {
        const vk::DescriptorPoolSize pool_sizes[] = {
            {vk::DescriptorType::eStorageBuffer, 3},
        };

        const vk::DescriptorPoolCreateInfo pool_ci{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 1,
            .poolSizeCount = static_cast<std::uint32_t>(std::size(pool_sizes)),
            .pPoolSizes    = pool_sizes,
        };

        FilterBindings out{};
        out.pool = vk::raii::DescriptorPool{device, pool_ci};
        return out;
    }

    export void allocate_filter_set(const vk::raii::Device& device,
                                    const FilterPipeline& pipe,
                                    FilterBindings& bindings) {
        const vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *bindings.pool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &*pipe.set_layout,
        };

        auto sets = vk::raii::DescriptorSets{device, ai};
        bindings.set = std::move(sets.front());
    }

    export void update_filter_set(const vk::raii::Device& device,
                                  const FilterBindings& bindings,
                                  const vk::memory::Buffer& in_buf,
                                  const vk::memory::Buffer& out_buf,
                                  const vk::memory::Buffer& count_buf) {
        const vk::DescriptorBufferInfo in_info{*in_buf.buffer, 0, in_buf.size};
        const vk::DescriptorBufferInfo out_info{*out_buf.buffer, 0, out_buf.size};
        const vk::DescriptorBufferInfo count_info{*count_buf.buffer, 0, count_buf.size};

        const vk::WriteDescriptorSet writes[] = {
            {
                .dstSet          = *bindings.set,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &in_info,
            },
            {
                .dstSet          = *bindings.set,
                .dstBinding      = 1,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &out_info,
            },
            {
                .dstSet          = *bindings.set,
                .dstBinding      = 2,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &count_info,
            },
        };

        device.updateDescriptorSets(writes, {});
    }

    export void dispatch_filter(const vk::raii::CommandBuffer& cmd,
                                const FilterPipeline& pipe,
                                const FilterBindings& bindings,
                                const FilterParams& params,
                                const std::uint32_t group_count) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipe.layout, 0, {*bindings.set}, {});
        cmd.pushConstants(*pipe.layout, vk::ShaderStageFlagBits::eCompute, 0,
                          vk::ArrayProxy<const FilterParams>{params});
        cmd.dispatch(group_count, 1, 1);
    }
    // ========================================================================
    // Indirect draw emission (count -> VkDrawIndirectCommand).
    // ========================================================================
    export [[nodiscard]] IndirectPipeline create_indirect_pipeline(const vk::raii::Device& device) {
        const vk::DescriptorSetLayoutBinding bindings[] = {
            {
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            },
            {
                .binding         = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            },
        };

        const vk::DescriptorSetLayoutCreateInfo set_ci{
            .bindingCount = static_cast<std::uint32_t>(std::size(bindings)),
            .pBindings    = bindings,
        };

        IndirectPipeline out{};
        out.set_layout = vk::raii::DescriptorSetLayout{device, set_ci};

        const vk::PipelineLayoutCreateInfo layout_ci{
            .setLayoutCount = 1,
            .pSetLayouts    = &*out.set_layout,
        };

        out.layout = vk::raii::PipelineLayout{device, layout_ci};

        const auto spv = vk::pipeline::read_file_bytes("../shaders/ray_indirect.spv");
        const auto shader = vk::pipeline::load_shader_module(device, spv);

        const vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = *shader,
            .pName  = "compMain",
        };

        const vk::ComputePipelineCreateInfo ci{
            .stage  = stage,
            .layout = *out.layout,
        };

        out.pipeline = vk::raii::Pipeline{device, nullptr, ci};
        return out;
    }

    export [[nodiscard]] IndirectBindings create_indirect_bindings(const vk::raii::Device& device) {
        const vk::DescriptorPoolSize pool_sizes[] = {
            {vk::DescriptorType::eStorageBuffer, 2},
        };

        const vk::DescriptorPoolCreateInfo pool_ci{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 1,
            .poolSizeCount = static_cast<std::uint32_t>(std::size(pool_sizes)),
            .pPoolSizes    = pool_sizes,
        };

        IndirectBindings out{};
        out.pool = vk::raii::DescriptorPool{device, pool_ci};
        return out;
    }

    export void allocate_indirect_set(const vk::raii::Device& device,
                                      const IndirectPipeline& pipe,
                                      IndirectBindings& bindings) {
        const vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *bindings.pool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &*pipe.set_layout,
        };

        auto sets = vk::raii::DescriptorSets{device, ai};
        bindings.set = std::move(sets.front());
    }

    export void update_indirect_set(const vk::raii::Device& device,
                                    const IndirectBindings& bindings,
                                    const vk::memory::Buffer& count_buf,
                                    const vk::memory::Buffer& indirect_buf) {
        const vk::DescriptorBufferInfo count_info{*count_buf.buffer, 0, count_buf.size};
        const vk::DescriptorBufferInfo indirect_info{*indirect_buf.buffer, 0, indirect_buf.size};

        const vk::WriteDescriptorSet writes[] = {
            {
                .dstSet          = *bindings.set,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &count_info,
            },
            {
                .dstSet          = *bindings.set,
                .dstBinding      = 1,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &indirect_info,
            },
        };

        device.updateDescriptorSets(writes, {});
    }

    export void dispatch_indirect(const vk::raii::CommandBuffer& cmd,
                                  const IndirectPipeline& pipe,
                                  const IndirectBindings& bindings) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipe.layout, 0,
                               {*bindings.set}, {});
        cmd.dispatch(1, 1, 1);
    }
} // namespace pngp::vis::rays::filter
