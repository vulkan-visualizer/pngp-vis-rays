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
import pngp.vis.rays.record;
import pngp.vis.rays.playback;
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
    // Ray rendering + filter settings.
    // ========================================================================
    struct RaySettings {
        bool show_rays = true;
        float line_length = 5.0f;
        float opacity     = 0.7f;
        std::uint32_t stride   = 1;
        std::uint32_t max_rays = 250000;
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
        // Ray playback + GPU resources.
        // ====================================================================
        playback::RayPlayback playback{};
        record::FrameHeader active_frame{};
        std::size_t active_frame_index = 0;

        playback::RayBufferGPU ray_input{};
        vk::memory::Buffer ray_filtered{};
        vk::memory::Buffer ray_count{};
        vk::memory::Buffer ray_indirect{};
        std::uint32_t ray_capacity = 0;

        filter::FilterPipeline filter_pipeline{};
        filter::FilterBindings filter_bindings{};
        filter::IndirectPipeline indirect_pipeline{};
        filter::IndirectBindings indirect_bindings{};

        vk::raii::DescriptorSetLayout ray_set_layout{nullptr};
        vk::raii::DescriptorPool ray_pool{nullptr};
        vk::raii::DescriptorSet ray_set{nullptr};
        vk::pipeline::GraphicsPipeline ray_pipeline;
        // ====================================================================
        // UI + playback state.
        // ====================================================================
        RaySettings rays{};
        std::array<char, 512> record_path_buf{};
        std::string record_error{};
        bool request_open_record  = false;
        bool request_close_record = false;
        bool request_frame_upload = false;
        bool request_ray_resize   = false;
        bool filter_dirty         = true;
        // ====================================================================
        // UI + input state.
        // ====================================================================
        GridSettings grid{};
        InputState input{};
    };
} // namespace pngp::vis::rays
