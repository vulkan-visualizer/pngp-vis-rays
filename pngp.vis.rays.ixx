export module pngp.vis.rays;
import vk.context;
import vk.swapchain;
import vk.frame;
import vk.imgui;
import vk.camera;
import vk.pipeline;
import vk.memory;
import vk.geometry;
import vk.math;
import std;

namespace pngp::vis::rays {

    struct GridSettings {
        bool show_grid   = true;
        bool show_axes   = true;
        bool show_origin = true;
        bool fly_mode    = false;

        float grid_extent = 12.0f;
        float grid_step   = 1.0f;
        int major_every   = 5;

        float axis_length  = 4.0f;
        float origin_scale = 0.25f;
    };

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

    struct ViewerRenderConfig {
        float fov_y_rad       = std::numbers::pi_v<float> / 3.0f;
        float near_plane      = 0.05f;
        float far_plane       = 2000.0f;
        bool srgb_textures    = true;
        bool enable_docking   = true;
        bool enable_viewports = true;
    };
    export struct RaysInspectorInfo {
        ViewerRenderConfig render{};
    };
    export class RaysInspector {
    public:
        void run();

        explicit RaysInspector(const RaysInspectorInfo& info);
        ~RaysInspector()                               = default;
        RaysInspector(const RaysInspector&)            = delete;
        RaysInspector& operator=(const RaysInspector&) = delete;
        RaysInspector(RaysInspector&&)                 = delete;
        RaysInspector& operator=(RaysInspector&&)      = delete;

    protected:
        void record_commands(std::uint32_t frame_index, std::uint32_t image_index);
        bool imgui_panel();

    private:
        vk::context::VulkanContext ctx;
        vk::context::SurfaceContext surface;
        vk::swapchain::Swapchain swapchain;
        vk::frame::FrameSystem frames;
        vk::imgui::ImGuiSystem imgui;
        vk::camera::Camera cam;
        vk::pipeline::GraphicsPipeline grid_pipeline;
        vk::memory::MeshGPU grid_mesh;
        vk::math::mat4 grid_mvp{};
        GridSettings grid{};
        InputState input{};
    };
} // namespace pngp::vis::rays
