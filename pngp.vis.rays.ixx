export module pngp.vis.rays;
import vk.context;
import vk.swapchain;
import vk.frame;
import vk.imgui;
import vk.camera;
import std;

namespace pngp::vis::rays {

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
        void imgui_panel();

    private:
        vk::context::VulkanContext ctx;
        vk::context::SurfaceContext surface;
        vk::swapchain::Swapchain swapchain;
        vk::frame::FrameSystem frames;
        vk::imgui::ImGuiSystem imgui;
        vk::camera::Camera cam;
    };
} // namespace pngp::vis::rays
