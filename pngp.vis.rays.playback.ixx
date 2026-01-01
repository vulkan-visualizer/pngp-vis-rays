export module pngp.vis.rays.playback;
// ============================================================================
// GPU-first playback helpers (Milestone 1).
// ============================================================================
import std;
import vk.context;
import vk.memory;
import pngp.vis.rays.record;

namespace pngp::vis::rays::playback {
    // ========================================================================
    // CPU frame payload: header + rays + samples.
    // ========================================================================
    struct RayFrameCPU {
        record::FrameHeader header{};
        std::vector<record::RayRecord> rays{};
        std::vector<record::SampleRecord> samples{};
    };

    // ========================================================================
    // GPU buffers for a frame.
    // ========================================================================
    struct RayBufferGPU {
        vk::memory::Buffer buffer{};
        std::uint64_t count{};
    };

    struct SampleBufferGPU {
        vk::memory::Buffer buffer{};
        std::uint64_t count{};
    };

    // ========================================================================
    // Playback controller: load frames and upload to GPU.
    // ========================================================================
    export class RayPlayback {
    public:
        void open(std::string_view path) { reader_.open(path); }
        void close() noexcept { reader_.close(); }
        [[nodiscard]] bool is_open() const noexcept { return reader_.is_open(); }

        [[nodiscard]] std::size_t frame_count() const noexcept { return reader_.frame_count(); }

        [[nodiscard]] RayFrameCPU load_frame(std::size_t index) const {
            RayFrameCPU out{};
            out.header  = reader_.frame_header(index);
            out.rays    = reader_.read_rays(index);
            out.samples = reader_.read_samples(index);
            return out;
        }

        [[nodiscard]] RayBufferGPU upload_rays(const vk::context::VulkanContext& ctx,
                                               std::span<const record::RayRecord> rays) const {
            if (rays.empty()) return {};
            const auto bytes = std::as_bytes(rays);
            auto buffer = vk::memory::upload_to_device_local_buffer(
                ctx.physical_device, ctx.device, ctx.command_pool, ctx.graphics_queue, bytes,
                vk::BufferUsageFlagBits::eStorageBuffer);
            return RayBufferGPU{std::move(buffer), static_cast<std::uint64_t>(rays.size())};
        }

        [[nodiscard]] SampleBufferGPU upload_samples(const vk::context::VulkanContext& ctx,
                                                     std::span<const record::SampleRecord> samples) const {
            if (samples.empty()) return {};
            const auto bytes = std::as_bytes(samples);
            auto buffer = vk::memory::upload_to_device_local_buffer(
                ctx.physical_device, ctx.device, ctx.command_pool, ctx.graphics_queue, bytes,
                vk::BufferUsageFlagBits::eStorageBuffer);
            return SampleBufferGPU{std::move(buffer), static_cast<std::uint64_t>(samples.size())};
        }

    private:
        record::RecordReader reader_{};
    };
} // namespace pngp::vis::rays::playback
