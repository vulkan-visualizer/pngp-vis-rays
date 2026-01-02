export module pngp.vis.rays;
// ============================================================================
// Rays Inspector public interface.
// ============================================================================
import vk.context;
import vk.swapchain;
import vk.frame;
import vk.imgui;
import vk.camera;
import vk.pipeline;
import vk.memory;
import vk.geometry;
import vk.math;
import pngp.vis.rays.record_v2;
import pngp.vis.rays.filter;
import std;

namespace pngp::vis::rays {
    // ========================================================================
    // Ground grid UI + render settings.
    // These values are mirrored into shader push constants each frame, so only
    // geometry-dependent knobs should trigger mesh rebuilds.
    // ========================================================================
    struct GridSettings {
        bool show_grid   = true;
        bool show_axes   = true;
        bool show_origin = true;
        bool fly_mode    = false;

        float grid_extent = 30.0f;
        float grid_step   = 1.0f;
        int major_every   = 5;

        float axis_length  = 4.0f;
        float origin_scale = 0.25f;
    };

    // ========================================================================
    // Lightweight input cache (GLFW callbacks fill, camera consumes).
    // ========================================================================
    struct InputState {
        bool lmb = false;
        bool mmb = false;
        bool rmb = false;

        std::array<bool, 512> keys{};

        double last_x  = 0.0;
        double last_y  = 0.0;
        bool have_last = false;

        float dx     = 0.0f;
        float dy     = 0.0f;
        float scroll = 0.0f;
    };

    // ========================================================================
    // Viewer render defaults (camera + ImGui behavior).
    // ========================================================================
    struct ViewerRenderConfig {
        float fov_y_rad       = std::numbers::pi_v<float> / 3.0f;
        float near_plane      = 0.05f;
        float far_plane       = 2000.0f;
        bool srgb_textures    = true;
        bool enable_docking   = true;
        bool enable_viewports = true;
    };

    // ========================================================================
    // Ray visualization color modes.
    // ========================================================================
    enum class RayColorMode : std::uint32_t {
        Direction = 0,
        Flags     = 1,
        MaskId    = 2,
        BatchId   = 3,
        ResultRgb = 4,
        Depth     = 5,
    };

    // ========================================================================
    // Ray rendering + filter settings.
    // ========================================================================
    struct RaySettings {
        bool show_rays = true;
        float line_length = 5.0f;
        float opacity     = 0.7f;
        std::uint32_t stride   = 1;
        std::uint32_t max_rays = 250000;
        RayColorMode color_mode = RayColorMode::Direction;
        float depth_min = 0.0f;
        float depth_max = 10.0f;

        bool roi_enabled = false;
        bool roi_initialized = false;
        int roi_min_x = 0;
        int roi_min_y = 0;
        int roi_max_x = 0;
        int roi_max_y = 0;

        bool filter_sample_state = false;
        bool filter_omit_reason  = false;
        std::uint32_t sample_state_mask = 0x0Fu;
        std::uint32_t omit_reason_mask  = 0xFFu;

        bool filter_mask_id  = false;
        bool filter_batch_id = false;
        std::uint32_t mask_id  = 0;
        std::uint32_t batch_id = 0;
    };

    // ========================================================================
    // Sample visualization settings (v2 only).
    // ========================================================================
    enum class SampleColorMode : std::uint32_t {
        State     = 0,
        Omit      = 1,
        Density   = 2,
        Weight    = 3,
        Contrib   = 4,
    };

    struct SampleSettings {
        bool show_samples = false;
        float point_size = 2.0f;
        std::uint32_t stride = 1;
        std::uint32_t max_samples = 300000;
        SampleColorMode color_mode = SampleColorMode::State;
        float density_min = 0.0f;
        float density_max = 1.0f;
        float weight_min  = 0.0f;
        float weight_max  = 1.0f;
        float contrib_min = 0.0f;
        float contrib_max = 1.0f;
        float alpha = 0.9f;
        bool isolate_ray = false;
        bool depth_fade = false;
        float depth_fade_near  = 0.0f;
        float depth_fade_far   = 50.0f;
        float depth_fade_power = 1.0f;
    };

    // ========================================================================
    // Ray picking + inspection settings.
    // ========================================================================
    struct PickSettings {
        bool enable = false;
        bool visible_only = true;
        bool auto_pin = true;
        float radius = 0.15f;
        std::uint32_t stride = 4;
    };

    // ========================================================================
    // GPU buffer wrapper for ray payloads.
    // ========================================================================
    struct RayBufferGPU {
        vk::memory::Buffer buffer{};
        std::uint64_t count{};
    };

    export struct RaysInspectorInfo {
        ViewerRenderConfig render{};
    };

    // ========================================================================
    // Main app: owns Vulkan context, swapchain, grid resources, and UI.
    // ========================================================================
    export class RaysInspector {
    public:
        // ====================================================================
        // Main loop: update input/camera, record commands, present.
        // ====================================================================
        void run();

        explicit RaysInspector(const RaysInspectorInfo& info);
        ~RaysInspector()                               = default;
        RaysInspector(const RaysInspector&)            = delete;
        RaysInspector& operator=(const RaysInspector&) = delete;
        RaysInspector(RaysInspector&&)                 = delete;
        RaysInspector& operator=(RaysInspector&&)      = delete;

    protected:
        // ====================================================================
        // Build the per-frame command buffer contents.
        // ====================================================================
        void record_commands(std::uint32_t frame_index, std::uint32_t image_index);
        // ====================================================================
        // Draw ImGui widgets; returns true when geometry needs rebuild.
        // ====================================================================
        bool imgui_panel();

    private:
        // ====================================================================
        // Core Vulkan systems.
        // ====================================================================
        vk::context::VulkanContext ctx;
        vk::context::SurfaceContext surface;
        vk::swapchain::Swapchain swapchain;
        vk::frame::FrameSystem frames;
        vk::imgui::ImGuiSystem imgui;
        // ====================================================================
        // Camera controller.
        // ====================================================================
        vk::camera::Camera cam;
        // ====================================================================
        // Grid GPU resources.
        // ====================================================================
        vk::pipeline::GraphicsPipeline grid_pipeline;
        vk::memory::MeshGPU grid_mesh;
        vk::math::mat4 grid_mvp{};
        // ====================================================================
        // Ray record + GPU resources.
        // ====================================================================
        std::size_t active_frame_index = 0;
        record_v2::RecordReaderV2 record_v2_reader{};
        record_v2::FrameViewV2 v2_view{};
        record_v2::FrameIndexEntryV2 active_frame_v2{};
        int v2_ray_index = 0;
        int v2_sample_index = 0;
        int v2_mask_attr_index = -1;
        int v2_batch_attr_index = -1;

        RayBufferGPU ray_input{};
        vk::memory::Buffer ray_filtered{};
        vk::memory::Buffer ray_count{};
        vk::memory::Buffer ray_indirect{};
        std::uint32_t ray_capacity = 0;

        vk::memory::Buffer v2_samples{};
        vk::memory::Buffer v2_evals{};
        vk::memory::Buffer v2_results{};
        vk::memory::Buffer sample_indices{};
        vk::memory::Buffer sample_count{};
        vk::memory::Buffer sample_indirect{};
        std::uint32_t sample_capacity = 0;
        vk::memory::Buffer dummy_storage{};
        std::vector<vk::memory::Buffer> v2_attribute_buffers{};
        std::uint32_t v2_attribute_capacity = 0;

        filter::FilterPipeline filter_pipeline{};
        filter::FilterBindings filter_bindings{};
        filter::IndirectPipeline indirect_pipeline{};
        filter::IndirectBindings indirect_bindings{};
        filter::FilterPipeline sample_filter_pipeline{};
        filter::FilterBindings sample_filter_bindings{};
        filter::IndirectPipeline sample_indirect_pipeline{};
        filter::IndirectBindings sample_indirect_bindings{};

        vk::raii::DescriptorSetLayout ray_set_layout{nullptr};
        vk::raii::DescriptorPool ray_pool{nullptr};
        vk::raii::DescriptorSet ray_set{nullptr};
        vk::pipeline::GraphicsPipeline ray_pipeline;

        vk::raii::DescriptorSetLayout sample_set_layout{nullptr};
        vk::raii::DescriptorPool sample_pool{nullptr};
        vk::raii::DescriptorSet sample_set{nullptr};
        vk::pipeline::GraphicsPipeline sample_pipeline;

        vk::raii::DescriptorSetLayout v2_attribute_set_layout{nullptr};
        vk::raii::DescriptorPool v2_attribute_pool{nullptr};
        std::vector<vk::raii::DescriptorSet> v2_attribute_sets{};
        // ====================================================================
        // UI + record state.
        // ====================================================================
        RaySettings rays{};
        SampleSettings samples{};
        std::array<char, 512> record_path_buf{};
        std::string record_error{};
        bool request_open_record  = false;
        bool request_close_record = false;
        bool request_frame_upload = false;
        bool request_ray_resize   = false;
        bool request_sample_resize = false;
        bool filter_dirty         = true;
        bool sample_filter_dirty  = true;
        // ====================================================================
        // UI + input state.
        // ====================================================================
        GridSettings grid{};
        PickSettings pick{};
        int pinned_ray_index = -1;
        int pinned_sample_index = 0;
        int pinned_table_limit = 128;
        int pinned_plot_limit = 512;
        int last_picked_ray_index = -1;
        float last_pick_distance = 0.0f;
        InputState input{};
    };
} // namespace pngp::vis::rays
