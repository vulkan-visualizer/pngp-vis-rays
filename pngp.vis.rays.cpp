module;
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <vulkan/vulkan_raii.hpp>
module pngp.vis.rays;
import std;
import vk.pipeline;
import vk.memory;
import vk.geometry;
import vk.math;

namespace {
    using pngp::vis::rays::GridSettings;
    using pngp::vis::rays::InputState;

    void glfw_key_cb(GLFWwindow* w, int key, int, int action, int) {
        auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(w));
        if (!s) return;
        if (key < 0 || key >= static_cast<int>(s->keys.size())) return;
        if (action == GLFW_PRESS) s->keys[static_cast<size_t>(key)] = true;
        if (action == GLFW_RELEASE) s->keys[static_cast<size_t>(key)] = false;
    }

    void glfw_mouse_button_cb(GLFWwindow* w, int button, int action, int) {
        auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(w));
        if (!s) return;

        const bool down = (action == GLFW_PRESS);
        if (button == GLFW_MOUSE_BUTTON_LEFT) s->lmb = down;
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) s->mmb = down;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) s->rmb = down;

        if (!down) s->have_last = false;
    }

    void glfw_cursor_pos_cb(GLFWwindow* w, double x, double y) {
        auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(w));
        if (!s) return;

        if (!s->have_last) {
            s->last_x    = x;
            s->last_y    = y;
            s->have_last = true;
            return;
        }

        s->dx += float(x - s->last_x);
        s->dy += float(y - s->last_y);

        s->last_x = x;
        s->last_y = y;
    }

    void glfw_scroll_cb(GLFWwindow* w, double, double yoff) {
        auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(w));
        if (!s) return;
        s->scroll += float(yoff);
    }

    struct GridPushConstants {
        vk::math::mat4 mvp{};
        vk::math::vec4 params0{};
        vk::math::vec4 params1{};
    };

    vk::memory::MeshCPU<vk::geometry::VertexP3C4> build_ground_plane(const GridSettings& grid) {
        vk::memory::MeshCPU<vk::geometry::VertexP3C4> mesh{};
        const float extent = std::max(0.1f, grid.grid_extent);
        const vk::math::vec4 color{1.0f, 1.0f, 1.0f, 1.0f};

        mesh.vertices = {
            {{-extent, 0.0f, -extent, 0.0f}, color},
            {{extent, 0.0f, -extent, 0.0f}, color},
            {{extent, 0.0f, extent, 0.0f}, color},
            {{-extent, 0.0f, extent, 0.0f}, color},
        };

        mesh.indices = {0, 1, 2, 0, 2, 3};
        return mesh;
    }

    vk::memory::MeshGPU upload_mesh_safe(
        const vk::context::VulkanContext& vkctx,
        const vk::memory::MeshCPU<vk::geometry::VertexP3C4>& mesh) {
        if (mesh.vertices.empty() || mesh.indices.empty()) return {};
        return vk::memory::upload_mesh(vkctx.physical_device, vkctx.device, vkctx.command_pool,
                                       vkctx.graphics_queue, mesh);
    }

    std::vector<std::byte> read_shader_bytes(const char* primary, const char* fallback) {
        try {
            return vk::pipeline::read_file_bytes(primary);
        } catch (const std::exception&) {
            return vk::pipeline::read_file_bytes(fallback);
        }
    }

    vk::pipeline::GraphicsPipeline create_grid_pipeline(const vk::context::VulkanContext& vkctx,
                                                        const vk::swapchain::Swapchain& sc) {
        const auto vin = vk::pipeline::make_vertex_input<vk::geometry::VertexP3C4>();
        const auto spv = read_shader_bytes("shaders/ground_grid.spv", "../shaders/ground_grid.spv");
        auto shader    = vk::pipeline::load_shader_module(vkctx.device, spv);

        vk::pipeline::GraphicsPipelineDesc desc{};
        desc.color_format         = sc.format;
        desc.depth_format         = sc.depth_format;
        desc.use_depth            = true;
        desc.cull                 = vk::CullModeFlagBits::eNone;
        desc.front_face           = vk::FrontFace::eCounterClockwise;
        desc.polygon_mode         = vk::PolygonMode::eFill;
        desc.topology             = vk::PrimitiveTopology::eTriangleList;
        desc.enable_blend         = true;
        desc.push_constant_bytes  = sizeof(GridPushConstants);
        desc.push_constant_stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        return vk::pipeline::create_graphics_pipeline(vkctx.device, vin, desc, shader, "vertMain",
                                                      "fragMain");
    }
} // namespace

int main() {
    pngp::vis::rays::RaysInspector app{{}};
    app.run();
    return 0;
}
void pngp::vis::rays::RaysInspector::run() {
    std::uint32_t frame_index = 0;
    using clock               = std::chrono::steady_clock;
    auto t_prev               = clock::now();
    while (!glfwWindowShouldClose(surface.window.get())) {
        glfwPollEvents();

        auto t_now = clock::now();
        float dt   = std::chrono::duration<float>(t_now - t_prev).count();
        t_prev     = t_now;
        if (!(dt > 0.0f)) dt = 1.0f / 60.0f;
        dt = std::min(dt, 0.05f);

        const auto ar = vk::frame::begin_frame(ctx, swapchain, frames, frame_index);
        if (!ar.ok || ar.need_recreate) {
            vk::swapchain::recreate_swapchain(ctx, surface, swapchain);
            vk::frame::on_swapchain_recreated(ctx, swapchain, frames);
            vk::imgui::set_min_image_count(imgui, 2);
            ctx.device.waitIdle();
            grid_pipeline = create_grid_pipeline(ctx, swapchain);
            continue;
        }
        vk::frame::begin_commands(frames, frame_index);
        vk::imgui::begin_frame();

        const bool rebuild_mesh = imgui_panel();
        if (rebuild_mesh) {
            ctx.device.waitIdle();
            const auto mesh_cpu = build_ground_plane(grid);
            grid_mesh = upload_mesh_safe(ctx, mesh_cpu);
        }

        cam.set_mode(grid.fly_mode ? vk::camera::Mode::Fly : vk::camera::Mode::Orbit);

        const ImGuiIO& io      = ImGui::GetIO();
        const bool block_mouse = io.WantCaptureMouse;
        const bool block_kbd   = io.WantCaptureKeyboard;

        vk::camera::CameraInput ci{};
        ci.lmb = (!block_mouse) && input.lmb;
        ci.mmb = (!block_mouse) && input.mmb;
        ci.rmb = (!block_mouse) && input.rmb;

        ci.mouse_dx = (!block_mouse) ? input.dx : 0.0f;
        ci.mouse_dy = (!block_mouse) ? input.dy : 0.0f;
        ci.scroll   = (!block_mouse) ? input.scroll : 0.0f;

        ci.shift = (!block_kbd) && (input.keys[GLFW_KEY_LEFT_SHIFT] || input.keys[GLFW_KEY_RIGHT_SHIFT]);
        ci.ctrl  = (!block_kbd) && (input.keys[GLFW_KEY_LEFT_CONTROL] || input.keys[GLFW_KEY_RIGHT_CONTROL]);
        ci.alt   = (!block_kbd) && (input.keys[GLFW_KEY_LEFT_ALT] || input.keys[GLFW_KEY_RIGHT_ALT]);
        ci.space = (!block_kbd) && (input.keys[GLFW_KEY_SPACE]);

        ci.forward  = (!block_kbd) && input.keys[GLFW_KEY_W];
        ci.backward = (!block_kbd) && input.keys[GLFW_KEY_S];
        ci.left     = (!block_kbd) && input.keys[GLFW_KEY_A];
        ci.right    = (!block_kbd) && input.keys[GLFW_KEY_D];
        ci.down     = (!block_kbd) && input.keys[GLFW_KEY_Q];
        ci.up       = (!block_kbd) && input.keys[GLFW_KEY_E];

        cam.update(dt, swapchain.extent.width, swapchain.extent.height, ci);
        vk::imgui::draw_mini_axis_gizmo(cam.matrices().c2w);

        input.dx     = 0.0f;
        input.dy     = 0.0f;
        input.scroll = 0.0f;

        grid_mvp = cam.matrices().view_proj;

        record_commands(frame_index, ar.image_index);
        if (vk::frame::end_frame(ctx, swapchain, frames, frame_index, ar.image_index)) {
            vk::swapchain::recreate_swapchain(ctx, surface, swapchain);
            vk::frame::on_swapchain_recreated(ctx, swapchain, frames);
            vk::imgui::set_min_image_count(imgui, 2);
            ctx.device.waitIdle();
            grid_pipeline = create_grid_pipeline(ctx, swapchain);
        }

        frame_index = (frame_index + 1) % frames.frames_in_flight;
    }
    this->ctx.device.waitIdle();
    vk::imgui::shutdown(imgui);
}
pngp::vis::rays::RaysInspector::RaysInspector(const RaysInspectorInfo& info) {
    auto [vkctx, surface] = vk::context::setup_vk_context_glfw("Dataset Viewer", "Engine");

    this->ctx       = std::move(vkctx);
    this->surface   = std::move(surface);
    glfwSetWindowUserPointer(this->surface.window.get(), &this->input);
    glfwSetKeyCallback(this->surface.window.get(), &glfw_key_cb);
    glfwSetMouseButtonCallback(this->surface.window.get(), &glfw_mouse_button_cb);
    glfwSetCursorPosCallback(this->surface.window.get(), &glfw_cursor_pos_cb);
    glfwSetScrollCallback(this->surface.window.get(), &glfw_scroll_cb);

    this->swapchain = vk::swapchain::setup_swapchain(this->ctx, this->surface);
    this->frames    = vk::frame::create_frame_system(this->ctx, this->swapchain, 2);
    this->imgui     = vk::imgui::create(this->ctx, this->surface.window.get(), this->swapchain.format, 2, static_cast<std::uint32_t>(this->swapchain.images.size()), info.render.enable_docking, info.render.enable_viewports);

    vk::camera::CameraConfig cam_cfg{};
    cam_cfg.fov_y_rad = info.render.fov_y_rad;
    cam_cfg.znear     = info.render.near_plane;
    cam_cfg.zfar      = info.render.far_plane;
    cam.set_config(cam_cfg);
    cam.home();
    cam.set_mode(vk::camera::Mode::Orbit);
    {
        auto st = cam.state();
        st.orbit.distance = std::max(1.0f, grid.grid_extent * 1.15f);
        cam.set_state(st);
    }

    const auto mesh_cpu = build_ground_plane(grid);
    grid_mesh = upload_mesh_safe(ctx, mesh_cpu);
    grid_pipeline = create_grid_pipeline(ctx, swapchain);
}
void pngp::vis::rays::RaysInspector::record_commands(std::uint32_t frame_index, std::uint32_t image_index) {
    auto& cmd = vk::frame::cmd(this->frames, frame_index);


    {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask     = vk::PipelineStageFlagBits2::eTopOfPipe,
            .dstStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout        = frames.swapchain_image_layout[image_index],
            .newLayout        = vk::ImageLayout::eColorAttachmentOptimal,
            .image            = swapchain.images[image_index],
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        const vk::DependencyInfo dep{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        cmd.pipelineBarrier2(dep);
        frames.swapchain_image_layout[image_index] = vk::ImageLayout::eColorAttachmentOptimal;
    }


    {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask     = vk::PipelineStageFlagBits2::eTopOfPipe,
            .dstStageMask     = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            .dstAccessMask    = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout        = swapchain.depth_layout,
            .newLayout        = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .image            = *swapchain.depth_image,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        };

        const vk::DependencyInfo dep{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        cmd.pipelineBarrier2(dep);
        swapchain.depth_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    }

    vk::ClearValue clear_color{};
    clear_color.color = vk::ClearColorValue{std::array{0.f, 0.f, 0.f, 1.0f}};

    vk::ClearValue clear_depth{};
    clear_depth.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

    const vk::RenderingAttachmentInfo color{
        .imageView   = *swapchain.image_views[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp      = vk::AttachmentLoadOp::eClear,
        .storeOp     = vk::AttachmentStoreOp::eStore,
        .clearValue  = clear_color,
    };

    const vk::RenderingAttachmentInfo depth{
        .imageView   = *swapchain.depth_view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp      = vk::AttachmentLoadOp::eClear,
        .storeOp     = vk::AttachmentStoreOp::eStore,
        .clearValue  = clear_depth,
    };

    const vk::RenderingInfo rendering{
        .renderArea           = {{0, 0}, swapchain.extent},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &color,
        .pDepthAttachment     = &depth,
    };

    cmd.beginRendering(rendering);
    const vk::Viewport vp{
        .x        = 0.f,
        .y        = static_cast<float>(swapchain.extent.height),
        .width    = static_cast<float>(swapchain.extent.width),
        .height   = -static_cast<float>(swapchain.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };

    const vk::Rect2D scissor{{0, 0}, swapchain.extent};

    cmd.setViewport(0, {vp});
    cmd.setScissor(0, {scissor});
    if (grid_mesh.index_count > 0) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *grid_pipeline.pipeline);
        const float step        = std::max(0.001f, grid.grid_step);
        const float extent      = std::max(0.1f, grid.grid_extent);
        const float major_every = static_cast<float>(std::max(1, grid.major_every));

        GridPushConstants push{};
        push.mvp     = grid_mvp;
        push.params0 = {step, step * major_every, extent, std::max(0.001f, grid.axis_length)};
        push.params1 = {std::max(0.001f, grid.origin_scale), grid.show_grid ? 1.0f : 0.0f,
                        grid.show_axes ? 1.0f : 0.0f, grid.show_origin ? 1.0f : 0.0f};

        cmd.pushConstants(*grid_pipeline.layout,
                          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                          vk::ArrayProxy<const GridPushConstants>{push});

        vk::DeviceSize offset = 0;
        cmd.bindVertexBuffers(0, {*grid_mesh.vertex_buffer.buffer}, {offset});
        cmd.bindIndexBuffer(*grid_mesh.index_buffer.buffer, 0, vk::IndexType::eUint32);
        cmd.drawIndexed(grid_mesh.index_count, 1, 0, 0, 0);
    }

    cmd.endRendering();

    vk::imgui::render(imgui, cmd, swapchain.extent, *swapchain.image_views[image_index], vk::ImageLayout::eColorAttachmentOptimal);
    vk::imgui::end_frame();


    {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask     = vk::PipelineStageFlagBits2::eBottomOfPipe,
            .oldLayout        = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout        = vk::ImageLayout::ePresentSrcKHR,
            .image            = swapchain.images[image_index],
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        const vk::DependencyInfo dep{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        cmd.pipelineBarrier2(dep);
        frames.swapchain_image_layout[image_index] = vk::ImageLayout::ePresentSrcKHR;
    }
}
bool pngp::vis::rays::RaysInspector::imgui_panel() {
    bool rebuild = false;
    ImGui::Begin("Rays Inspector");
    ImGui::TextUnformatted("Ground Plane");
    rebuild |= ImGui::Checkbox("Show grid", &grid.show_grid);
    rebuild |= ImGui::Checkbox("Show axes", &grid.show_axes);
    rebuild |= ImGui::Checkbox("Show origin", &grid.show_origin);
    ImGui::Separator();
    rebuild |= ImGui::SliderFloat("Grid extent", &grid.grid_extent, 2.0f, 100.0f);
    rebuild |= ImGui::SliderFloat("Grid step", &grid.grid_step, 0.1f, 5.0f);
    rebuild |= ImGui::SliderInt("Major every", &grid.major_every, 1, 20);
    rebuild |= ImGui::SliderFloat("Axis length", &grid.axis_length, 0.5f, 20.0f);
    rebuild |= ImGui::SliderFloat("Origin scale", &grid.origin_scale, 0.05f, 2.0f);
    ImGui::Separator();
    ImGui::Checkbox("Fly mode", &grid.fly_mode);
    ImGui::TextUnformatted("Orbit: Alt/Space + LMB rotate, MMB pan, wheel zoom");
    ImGui::TextUnformatted("Fly: RMB look + WASD move, Q/E down/up");
    ImGui::End();
    return rebuild;
}
