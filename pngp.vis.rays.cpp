module;
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <vulkan/vulkan_raii.hpp>
module pngp.vis.rays;
// ============================================================================
// RaysInspector implementation.
// ============================================================================
import std;
import vk.pipeline;
import vk.memory;
import vk.geometry;
import vk.math;
import pngp.vis.rays.record_v2;
import pngp.vis.rays.filter;

// ============================================================================
// Translation-unit helpers (input callbacks, grid helpers, pipeline setup).
// ============================================================================
namespace {
    using pngp::vis::rays::GridSettings;
    using pngp::vis::rays::InputState;
    using pngp::vis::rays::RaySettings;

    constexpr std::uint32_t k_filter_roi       = 1u << 0;
    constexpr std::uint32_t k_filter_sample    = 1u << 1;
    constexpr std::uint32_t k_filter_mask_id   = 1u << 2;
    constexpr std::uint32_t k_filter_batch_id  = 1u << 3;
    constexpr std::uint32_t k_filter_ray_index = 1u << 4;

    // =========================================================================
    // GLFW input callbacks: collect raw input into InputState.
    // =========================================================================
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

        const bool down = action == GLFW_PRESS;
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

        s->dx += static_cast<float>(x - s->last_x);
        s->dy += static_cast<float>(y - s->last_y);

        s->last_x = x;
        s->last_y = y;
    }

    void glfw_scroll_cb(GLFWwindow* w, double, double yoff) {
        auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(w));
        if (!s) return;
        s->scroll += static_cast<float>(yoff);
    }

    // =========================================================================
    // Push constants consumed by ground_grid.slang.
    // =========================================================================
    struct GridPush {
        vk::math::mat4 mvp{};
        vk::math::vec4 grid{};
        vk::math::vec4 toggles{};
    };

    // =========================================================================
    // Push constants consumed by ray_lines_v2.slang.
    // =========================================================================
    struct RayPush {
        vk::math::mat4 mvp{};
        vk::math::vec4 params{};
        std::uint32_t mode{};
        std::uint32_t pad0{};
        std::uint32_t pad1{};
        std::uint32_t pad2{};
    };

    // =========================================================================
    // Push constants consumed by sample_points_v2.slang.
    // =========================================================================
    struct SamplePush {
        vk::math::mat4 mvp{};
        vk::math::vec4 params{};
        vk::math::vec4 depth{};
        vk::math::vec4 camera{};
        std::uint32_t mode{};
        std::uint32_t stride{};
        std::uint32_t pad0{};
        std::uint32_t pad1{};
    };

    // =========================================================================
    // Single quad for the grid surface; shader draws the lines procedurally.
    // =========================================================================
    vk::memory::MeshCPU<vk::geometry::VertexP3C4> build_ground_plane(const float extent) {
        vk::memory::MeshCPU<vk::geometry::VertexP3C4> mesh{};
        const float clamped_extent = std::max(0.1f, extent);
        constexpr vk::math::vec4 color{1.0f, 1.0f, 1.0f, 1.0f};

        mesh.vertices = {
            {{-clamped_extent, 0.0f, -clamped_extent, 0.0f}, color},
            {{clamped_extent, 0.0f, -clamped_extent, 0.0f}, color},
            {{clamped_extent, 0.0f, clamped_extent, 0.0f}, color},
            {{-clamped_extent, 0.0f, clamped_extent, 0.0f}, color},
        };

        mesh.indices = {0, 1, 2, 0, 2, 3};
        return mesh;
    }

    // =========================================================================
    // Helper that returns an empty GPU mesh if CPU data is empty.
    // =========================================================================
    vk::memory::MeshGPU upload_mesh_safe(const vk::context::VulkanContext& vkctx, const vk::memory::MeshCPU<vk::geometry::VertexP3C4>& mesh) {
        if (mesh.vertices.empty() || mesh.indices.empty()) return {};
        return vk::memory::upload_mesh(vkctx.physical_device, vkctx.device, vkctx.command_pool, vkctx.graphics_queue, mesh);
    }

    // =========================================================================
    // Load SPIR-V from first available path (build dir or repo root).
    // =========================================================================
    std::vector<std::byte> read_shader_bytes(std::span<const char* const> paths) {
        std::exception_ptr last_error;
        for (const char* path : paths) {
            try {
                return vk::pipeline::read_file_bytes(path);
            } catch (...) {
                last_error = std::current_exception();
            }
        }
        if (last_error) std::rethrow_exception(last_error);
        throw std::runtime_error("ground_grid.spv not found");
    }

    // =========================================================================
    // Attribute stream lookup helpers.
    // =========================================================================
    int find_attribute_index(
        std::span<const pngp::vis::rays::record_v2::AttributeStreamViewV2> attrs,
        std::string_view name) {
        for (std::size_t i = 0; i < attrs.size(); ++i) {
            const auto& attr = attrs[i];
            if (attr.name != name) continue;
            if (attr.desc.target !=
                static_cast<std::uint32_t>(pngp::vis::rays::record_v2::AttributeTarget::Ray)) {
                continue;
            }
            if (attr.desc.format !=
                static_cast<std::uint32_t>(pngp::vis::rays::record_v2::AttributeFormat::U32)) {
                continue;
            }
            if (attr.desc.components != 1) continue;
            return static_cast<int>(i);
        }
        return -1;
    }

    // =========================================================================
    // Convert UI settings to shader-friendly constants.
    // =========================================================================
    GridPush make_grid_push(const GridSettings& grid, const vk::math::mat4& mvp) {
        const float step   = std::max(0.001f, grid.grid_step);
        const float extent = std::max(0.1f, grid.grid_extent);
        const float major  = static_cast<float>(std::max(1, grid.major_every));

        GridPush push{};
        push.mvp     = mvp;
        push.grid    = {step, step * major, extent, std::max(0.001f, grid.axis_length)};
        push.toggles = {std::max(0.001f, grid.origin_scale), grid.show_grid ? 1.0f : 0.0f, grid.show_axes ? 1.0f : 0.0f, grid.show_origin ? 1.0f : 0.0f};
        return push;
    }

    // =========================================================================
    // Convert ray draw settings to shader-friendly constants.
    // =========================================================================
    RayPush make_ray_push(const RaySettings& rays, const vk::math::mat4& mvp) {
        const float depth_range = std::max(0.0001f, rays.depth_max - rays.depth_min);
        const float depth_scale = 1.0f / depth_range;
        const float depth_bias  = -rays.depth_min * depth_scale;
        RayPush push{};
        push.mvp    = mvp;
        push.params = {std::max(0.001f, rays.line_length),
                       std::clamp(rays.opacity, 0.0f, 1.0f),
                       depth_scale,
                       depth_bias};
        push.mode   = static_cast<std::uint32_t>(rays.color_mode);
        return push;
    }

    SamplePush make_sample_push(const pngp::vis::rays::SampleSettings& samples,
                                const vk::math::mat4& mvp,
                                const vk::math::vec3& cam_pos) {
        float range_min = 0.0f;
        float range_max = 1.0f;
        switch (samples.color_mode) {
            case pngp::vis::rays::SampleColorMode::Density:
                range_min = samples.density_min;
                range_max = samples.density_max;
                break;
            case pngp::vis::rays::SampleColorMode::Weight:
                range_min = samples.weight_min;
                range_max = samples.weight_max;
                break;
            case pngp::vis::rays::SampleColorMode::Contrib:
                range_min = samples.contrib_min;
                range_max = samples.contrib_max;
                break;
            default:
                break;
        }

        const float alpha = std::clamp(samples.alpha, 0.0f, 1.0f);
        float fade_near = std::max(0.0f, samples.depth_fade_near);
        float fade_far  = std::max(fade_near + 0.001f, samples.depth_fade_far);
        const float fade_power = std::max(0.01f, samples.depth_fade_power);
        const float fade_enable = samples.depth_fade ? 1.0f : 0.0f;

        SamplePush push{};
        push.mvp = mvp;
        push.params = {std::max(0.5f, samples.point_size),
                       range_min,
                       range_max,
                       alpha};
        push.depth  = {fade_near, fade_far, fade_power, fade_enable};
        push.camera = {cam_pos.x, cam_pos.y, cam_pos.z, 0.0f};
        push.mode   = static_cast<std::uint32_t>(samples.color_mode);
        push.stride = std::max(1u, samples.stride);
        return push;
    }

    vk::math::vec3 heatmap_color(const float t) {
        const float u = std::clamp(t, 0.0f, 1.0f);
        const vk::math::vec3 c0{0.10f, 0.20f, 0.90f, 0.0f};
        const vk::math::vec3 c1{0.00f, 0.80f, 0.90f, 0.0f};
        const vk::math::vec3 c2{0.20f, 0.90f, 0.40f, 0.0f};
        const vk::math::vec3 c3{0.95f, 0.90f, 0.20f, 0.0f};
        const vk::math::vec3 c4{0.90f, 0.20f, 0.20f, 0.0f};
        const float scaled = u * 4.0f;
        const int idx = std::clamp(static_cast<int>(scaled), 0, 3);
        const float f = scaled - static_cast<float>(idx);
        const auto lerp = [f](vk::math::vec3 a, vk::math::vec3 b) {
            return vk::math::vec3{
                a.x + (b.x - a.x) * f,
                a.y + (b.y - a.y) * f,
                a.z + (b.z - a.z) * f,
                0.0f,
            };
        };
        if (idx == 0) return lerp(c0, c1);
        if (idx == 1) return lerp(c1, c2);
        if (idx == 2) return lerp(c2, c3);
        return lerp(c3, c4);
    }

    void draw_heatmap_legend(const char* label, float min_v, float max_v) {
        ImGui::TextUnformatted(label);
        const float width = ImGui::CalcItemWidth();
        const float height = 10.0f;
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        auto* draw = ImGui::GetWindowDrawList();
        constexpr int steps = 32;
        for (int i = 0; i < steps; ++i) {
            const float t0 = static_cast<float>(i) / static_cast<float>(steps);
            const float t1 = static_cast<float>(i + 1) / static_cast<float>(steps);
            const auto c = heatmap_color((t0 + t1) * 0.5f);
            const ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(c.x, c.y, c.z, 1.0f));
            draw->AddRectFilled({pos.x + width * t0, pos.y},
                                {pos.x + width * t1, pos.y + height},
                                col);
        }
        ImGui::Dummy(ImVec2(width, height));
        ImGui::Text("min %.3f   max %.3f", static_cast<double>(min_v),
                    static_cast<double>(max_v));
    }

    struct AttrU32View {
        const std::byte* data = nullptr;
        std::size_t bytes = 0;
        std::uint32_t stride = 0;
        std::uint32_t count = 0;
        bool valid = false;
    };

    AttrU32View make_attr_u32_view(
        std::span<const pngp::vis::rays::record_v2::AttributeStreamViewV2> attrs,
        const int index) {
        if (index < 0 || static_cast<std::size_t>(index) >= attrs.size()) return {};
        const auto& attr = attrs[static_cast<std::size_t>(index)];
        if (attr.desc.components != 1) return {};
        if (attr.desc.format !=
            static_cast<std::uint32_t>(pngp::vis::rays::record_v2::AttributeFormat::U32)) {
            return {};
        }
        if (attr.desc.stride_bytes < sizeof(std::uint32_t)) return {};
        if (attr.desc.count == 0 || attr.data.empty()) return {};
        return {attr.data.data(), attr.data.size(), attr.desc.stride_bytes, attr.desc.count, true};
    }

    std::uint32_t read_attr_u32(const AttrU32View& attr, const std::uint32_t index, const std::uint32_t fallback) {
        if (!attr.valid || index >= attr.count) return fallback;
        const std::size_t offset = static_cast<std::size_t>(index) * attr.stride;
        if (offset + sizeof(std::uint32_t) > attr.bytes) return fallback;
        std::uint32_t value = fallback;
        std::memcpy(&value, attr.data + offset, sizeof(std::uint32_t));
        return value;
    }

    vk::math::vec3 normalize_or(vk::math::vec3 v, vk::math::vec3 fallback) {
        const float len = vk::math::length(v);
        if (len > 1e-6f) return vk::math::mul(v, 1.0f / len);
        return fallback;
    }

    vk::math::vec3 make_vec3(const float x, const float y, const float z) {
        return {x, y, z, 0.0f};
    }

    float distance_ray_segment(const vk::math::vec3& ro,
                               const vk::math::vec3& rd,
                               const vk::math::vec3& so,
                               const vk::math::vec3& sd,
                               const float seg_len,
                               float& out_seg_t) {
        const vk::math::vec3 r = vk::math::sub(ro, so);
        const float a = vk::math::dot(rd, rd);
        const float e = vk::math::dot(sd, sd);
        const float b = vk::math::dot(rd, sd);
        const float c = vk::math::dot(rd, r);
        const float f = vk::math::dot(sd, r);
        const float denom = a * e - b * b;

        float s = 0.0f;
        float t = 0.0f;
        if (denom > 1e-6f) {
            s = (b * f - c * e) / denom;
            t = (a * f - b * c) / denom;
        } else {
            t = (e > 1e-6f) ? (f / e) : 0.0f;
        }

        t = std::clamp(t, 0.0f, std::max(0.0f, seg_len));
        if (s < 0.0f) s = 0.0f;
        out_seg_t = t;

        const vk::math::vec3 p1 = vk::math::add(ro, vk::math::mul(rd, s));
        const vk::math::vec3 p2 = vk::math::add(so, vk::math::mul(sd, t));
        const vk::math::vec3 d = vk::math::sub(p1, p2);
        return vk::math::length2(d);
    }

    bool ray_passes_filters_cpu(
        const std::uint32_t idx,
        const pngp::vis::rays::record_v2::FrameViewV2& view,
        const pngp::vis::rays::record_v2::FrameIndexEntryV2& frame,
        const pngp::vis::rays::RaySettings& rays,
        const AttrU32View& mask_attr,
        const AttrU32View& batch_attr) {
        const auto& ray = view.rays[idx];
        if (rays.roi_enabled && frame.width > 0 && frame.height > 0) {
            const int max_x = static_cast<int>(frame.width - 1);
            const int max_y = static_cast<int>(frame.height - 1);
            int roi_min_x = std::clamp(rays.roi_min_x, 0, max_x);
            int roi_min_y = std::clamp(rays.roi_min_y, 0, max_y);
            int roi_max_x = std::clamp(rays.roi_max_x, 0, max_x);
            int roi_max_y = std::clamp(rays.roi_max_y, 0, max_y);
            if (roi_min_x > roi_max_x) std::swap(roi_min_x, roi_max_x);
            if (roi_min_y > roi_max_y) std::swap(roi_min_y, roi_max_y);
            if (ray.pixel_x < static_cast<std::uint32_t>(roi_min_x) ||
                ray.pixel_x > static_cast<std::uint32_t>(roi_max_x) ||
                ray.pixel_y < static_cast<std::uint32_t>(roi_min_y) ||
                ray.pixel_y > static_cast<std::uint32_t>(roi_max_y)) {
                return false;
            }
        }

        if ((rays.filter_sample_state || rays.filter_omit_reason) && !view.samples.empty()) {
            const std::size_t base = static_cast<std::size_t>(ray.sample_offset);
            const std::size_t count = static_cast<std::size_t>(ray.sample_count);
            if (count == 0 || base + count > view.samples.size()) return false;
            bool any = false;
            for (std::size_t i = 0; i < count; ++i) {
                const auto& s = view.samples[base + i];
                const bool state_ok = !rays.filter_sample_state ||
                                      ((rays.sample_state_mask & (1u << s.state)) != 0u);
                const bool omit_ok = !rays.filter_omit_reason ||
                                     ((rays.omit_reason_mask & (1u << s.omit_reason)) != 0u);
                if (state_ok && omit_ok) {
                    any = true;
                    break;
                }
            }
            if (!any) return false;
        }

        if (rays.filter_mask_id && mask_attr.valid) {
            if (read_attr_u32(mask_attr, idx, std::numeric_limits<std::uint32_t>::max()) != rays.mask_id) {
                return false;
            }
        }

        if (rays.filter_batch_id && batch_attr.valid) {
            if (read_attr_u32(batch_attr, idx, std::numeric_limits<std::uint32_t>::max()) != rays.batch_id) {
                return false;
            }
        }

        return true;
    }

    vk::math::vec3 compute_pick_dir(const vk::camera::Camera& cam,
                                    const ImVec2& mouse,
                                    const vk::Extent2D extent) {
        if (extent.width == 0 || extent.height == 0) {
            return {0.0f, 0.0f, -1.0f, 0.0f};
        }
        const float w = static_cast<float>(extent.width);
        const float h = static_cast<float>(extent.height);
        const float ndc_x = (2.0f * mouse.x / w) - 1.0f;
        const float ndc_y = 1.0f - (2.0f * mouse.y / h);

        const auto& cfg = cam.config();
        const auto& m   = cam.matrices();
        if (cfg.projection != vk::camera::Projection::Perspective) {
            const float sign = (cfg.convention.view_forward.sign == vk::camera::Sign::Positive) ? 1.0f : -1.0f;
            return normalize_or(vk::math::mul(m.forward, sign), {0.0f, 0.0f, -1.0f, 0.0f});
        }

        const float aspect = w / h;
        const float tan_half = std::tan(cfg.fov_y_rad * 0.5f);
        const float vx = ndc_x * aspect * tan_half;
        const float vy = ndc_y * tan_half;
        const float vz = (cfg.convention.view_forward.sign == vk::camera::Sign::Positive) ? 1.0f : -1.0f;

        const vk::math::vec3 dir = vk::math::add(
            vk::math::add(vk::math::mul(m.right, vx), vk::math::mul(m.up, vy)),
            vk::math::mul(m.forward, vz));
        return normalize_or(dir, {0.0f, 0.0f, -1.0f, 0.0f});
    }

    struct PickResult {
        int index = -1;
        float dist2 = 0.0f;
        float ray_t = 0.0f;
    };

    std::optional<PickResult> pick_ray_index(
        const pngp::vis::rays::record_v2::FrameViewV2& view,
        const pngp::vis::rays::record_v2::FrameIndexEntryV2& frame,
        const pngp::vis::rays::RaySettings& rays,
        const pngp::vis::rays::PickSettings& pick,
        const vk::camera::Camera& cam,
        const vk::Extent2D extent,
        const ImVec2& mouse,
        const int mask_attr_index,
        const int batch_attr_index) {
        if (view.rays.empty()) return std::nullopt;
        if (pick.visible_only && !rays.show_rays) return std::nullopt;

        const auto mask_attr = make_attr_u32_view(view.attributes, mask_attr_index);
        const auto batch_attr = make_attr_u32_view(view.attributes, batch_attr_index);

        const vk::math::vec3 pick_origin = cam.matrices().eye;
        const vk::math::vec3 pick_dir = compute_pick_dir(cam, mouse, extent);

        const float max_dist2 = pick.radius * pick.radius;
        const float seg_len = std::max(0.001f, rays.line_length);
        const std::uint32_t stride = std::max(1u, pick.stride);

        PickResult best{};
        best.dist2 = max_dist2;

        for (std::uint32_t i = 0; i < view.rays.size(); i += stride) {
            if (pick.visible_only &&
                !ray_passes_filters_cpu(i, view, frame, rays, mask_attr, batch_attr)) {
                continue;
            }
            const auto& r = view.rays[i];
            const vk::math::vec3 ray_origin = make_vec3(r.ox, r.oy, r.oz);
            const vk::math::vec3 ray_dir = normalize_or(make_vec3(r.dx, r.dy, r.dz),
                                                        {0.0f, 0.0f, -1.0f, 0.0f});
            float seg_t = 0.0f;
            const float dist2 = distance_ray_segment(pick_origin, pick_dir, ray_origin, ray_dir,
                                                     seg_len, seg_t);
            if (dist2 < best.dist2) {
                best.index = static_cast<int>(i);
                best.dist2 = dist2;
                best.ray_t = seg_t;
            }
        }

        if (best.index < 0) return std::nullopt;
        return best;
    }

    // =========================================================================
    // Minimal pipeline for a transparent grid surface with depth testing.
    // =========================================================================
    vk::pipeline::GraphicsPipeline create_grid_pipeline(const vk::context::VulkanContext& vkctx, const vk::swapchain::Swapchain& sc) {
        const auto vin = vk::pipeline::make_vertex_input<vk::geometry::VertexP3C4>();
        constexpr std::array paths{"../shaders/ground_grid.spv", "../shaders/ground_grid.spv"};
        const auto spv = read_shader_bytes(paths);
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
        desc.push_constant_bytes  = sizeof(GridPush);
        desc.push_constant_stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        return vk::pipeline::create_graphics_pipeline(vkctx.device, vin, desc, shader, "vertMain", "fragMain");
    }

    // =========================================================================
    // Descriptor set + pipeline helpers for ray line rendering.
    // =========================================================================
    vk::raii::DescriptorSetLayout create_ray_set_layout(const vk::raii::Device& device) {
        const vk::DescriptorSetLayoutBinding bindings[] = {
            {
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
            {
                .binding         = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
            {
                .binding         = 2,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
            {
                .binding         = 3,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
        };

        const vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<std::uint32_t>(std::size(bindings)),
            .pBindings    = bindings,
        };
        return vk::raii::DescriptorSetLayout{device, ci};
    }

    vk::raii::DescriptorPool create_ray_pool(const vk::raii::Device& device) {
        const vk::DescriptorPoolSize pool_size{
            .type            = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 4,
        };
        const vk::DescriptorPoolCreateInfo ci{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 1,
            .poolSizeCount = 1,
            .pPoolSizes    = &pool_size,
        };
        return vk::raii::DescriptorPool{device, ci};
    }

    vk::raii::DescriptorSet allocate_ray_set(const vk::raii::Device& device,
                                             const vk::raii::DescriptorPool& pool,
                                             const vk::raii::DescriptorSetLayout& layout) {
        const vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *pool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &*layout,
        };
        auto sets = vk::raii::DescriptorSets{device, ai};
        return std::move(sets.front());
    }

    void update_ray_set(const vk::raii::Device& device,
                        const vk::raii::DescriptorSet& set,
                        const vk::memory::Buffer& rays,
                        const vk::memory::Buffer& results,
                        const vk::memory::Buffer& mask_ids,
                        const vk::memory::Buffer& batch_ids) {
        const vk::DescriptorBufferInfo ray_info{*rays.buffer, 0, rays.size};
        const vk::DescriptorBufferInfo result_info{*results.buffer, 0, results.size};
        const vk::DescriptorBufferInfo mask_info{*mask_ids.buffer, 0, mask_ids.size};
        const vk::DescriptorBufferInfo batch_info{*batch_ids.buffer, 0, batch_ids.size};
        const vk::WriteDescriptorSet writes[] = {
            {
                .dstSet          = *set,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &ray_info,
            },
            {
                .dstSet          = *set,
                .dstBinding      = 1,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &result_info,
            },
            {
                .dstSet          = *set,
                .dstBinding      = 2,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &mask_info,
            },
            {
                .dstSet          = *set,
                .dstBinding      = 3,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &batch_info,
            },
        };
        device.updateDescriptorSets(writes, {});
    }

    // =========================================================================
    // Upload helpers for v2 payloads.
    // =========================================================================
    vk::memory::Buffer upload_storage_buffer(const vk::context::VulkanContext& vkctx,
                                             std::span<const std::byte> bytes) {
        if (bytes.empty()) return {};
        return vk::memory::upload_to_device_local_buffer(
            vkctx.physical_device, vkctx.device, vkctx.command_pool, vkctx.graphics_queue, bytes,
            vk::BufferUsageFlagBits::eStorageBuffer);
    }

    vk::memory::Buffer upload_storage_buffer_non_empty(const vk::context::VulkanContext& vkctx,
                                                       std::span<const std::byte> bytes) {
        if (!bytes.empty()) return upload_storage_buffer(vkctx, bytes);

        std::array<std::byte, 16> zeros{};
        return vk::memory::upload_to_device_local_buffer(
            vkctx.physical_device, vkctx.device, vkctx.command_pool, vkctx.graphics_queue, zeros,
            vk::BufferUsageFlagBits::eStorageBuffer);
    }

    // =========================================================================
    // Attribute stream descriptor helpers (v2 GPU uploads).
    // =========================================================================
    vk::raii::DescriptorSetLayout create_attribute_set_layout(const vk::raii::Device& device) {
        const vk::DescriptorSetLayoutBinding binding{
            .binding         = 0,
            .descriptorType  = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eAll,
        };

        const vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = 1,
            .pBindings    = &binding,
        };
        return vk::raii::DescriptorSetLayout{device, ci};
    }

    vk::raii::DescriptorPool create_attribute_pool(const vk::raii::Device& device,
                                                   const std::uint32_t count) {
        if (count == 0) return vk::raii::DescriptorPool{nullptr};

        const vk::DescriptorPoolSize pool_size{
            .type            = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = count,
        };
        const vk::DescriptorPoolCreateInfo ci{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = count,
            .poolSizeCount = 1,
            .pPoolSizes    = &pool_size,
        };
        return vk::raii::DescriptorPool{device, ci};
    }

    std::vector<vk::raii::DescriptorSet> allocate_attribute_sets(
        const vk::raii::Device& device,
        const vk::raii::DescriptorPool& pool,
        const vk::raii::DescriptorSetLayout& layout,
        const std::uint32_t count) {
        if (count == 0) return {};

        std::vector<vk::DescriptorSetLayout> layouts(count, *layout);
        const vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *pool,
            .descriptorSetCount = count,
            .pSetLayouts        = layouts.data(),
        };

        auto sets = vk::raii::DescriptorSets{device, ai};
        std::vector<vk::raii::DescriptorSet> out{};
        out.reserve(count);
        for (auto& set : sets) {
            out.emplace_back(std::move(set));
        }
        return out;
    }

    void update_attribute_sets(const vk::raii::Device& device,
                               std::span<const vk::raii::DescriptorSet> sets,
                               std::span<const vk::memory::Buffer> buffers) {
        if (sets.empty() || buffers.empty()) return;

        const std::size_t count = std::min(sets.size(), buffers.size());
        std::vector<vk::DescriptorBufferInfo> infos{};
        std::vector<vk::WriteDescriptorSet> writes{};
        infos.reserve(count);
        writes.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            infos.push_back(vk::DescriptorBufferInfo{*buffers[i].buffer, 0, buffers[i].size});
            writes.push_back(vk::WriteDescriptorSet{
                .dstSet          = *sets[i],
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &infos.back(),
            });
        }

        device.updateDescriptorSets(writes, {});
    }

    void ensure_attribute_sets(const vk::raii::Device& device,
                               const vk::raii::DescriptorSetLayout& layout,
                               const std::uint32_t desired,
                               vk::raii::DescriptorPool& pool,
                               std::vector<vk::raii::DescriptorSet>& sets,
                               std::uint32_t& capacity) {
        if (desired == 0) {
            sets.clear();
            pool = vk::raii::DescriptorPool{nullptr};
            capacity = 0;
            return;
        }

        if (desired > capacity) {
            sets.clear();
            pool = create_attribute_pool(device, desired);
            sets = allocate_attribute_sets(device, pool, layout, desired);
            capacity = desired;
        } else if (sets.size() < desired) {
            sets.clear();
            pool = create_attribute_pool(device, desired);
            sets = allocate_attribute_sets(device, pool, layout, desired);
            capacity = desired;
        }
    }

    vk::pipeline::GraphicsPipeline create_ray_pipeline_with_paths(const vk::context::VulkanContext& vkctx,
                                                                  const vk::swapchain::Swapchain& sc,
                                                                  const vk::raii::DescriptorSetLayout& set_layout,
                                                                  std::span<const char* const> paths) {
        vk::pipeline::VertexInput vin{};
        const auto spv = read_shader_bytes(paths);
        auto shader    = vk::pipeline::load_shader_module(vkctx.device, spv);

        vk::pipeline::GraphicsPipelineDesc desc{};
        desc.color_format         = sc.format;
        desc.depth_format         = sc.depth_format;
        desc.use_depth            = true;
        desc.cull                 = vk::CullModeFlagBits::eNone;
        desc.front_face           = vk::FrontFace::eCounterClockwise;
        desc.polygon_mode         = vk::PolygonMode::eFill;
        desc.topology             = vk::PrimitiveTopology::eLineList;
        desc.enable_blend         = true;
        desc.push_constant_bytes  = sizeof(RayPush);
        desc.push_constant_stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        const vk::DescriptorSetLayout layouts[] = {*set_layout};
        desc.set_layouts = layouts;
        return vk::pipeline::create_graphics_pipeline(vkctx.device, vin, desc, shader, "vertMain", "fragMain");
    }

    vk::pipeline::GraphicsPipeline create_ray_pipeline(const vk::context::VulkanContext& vkctx,
                                                       const vk::swapchain::Swapchain& sc,
                                                       const vk::raii::DescriptorSetLayout& set_layout) {
        constexpr std::array paths{"../shaders/ray_lines_v2.spv", "../shaders/ray_lines_v2.spv"};
        return create_ray_pipeline_with_paths(vkctx, sc, set_layout, paths);
    }

    // =========================================================================
    // Descriptor set + pipeline helpers for sample point rendering (v2 only).
    // =========================================================================
    vk::raii::DescriptorSetLayout create_sample_set_layout(const vk::raii::Device& device) {
        const vk::DescriptorSetLayoutBinding bindings[] = {
            {
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
            {
                .binding         = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
            {
                .binding         = 2,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
            {
                .binding         = 3,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eVertex,
            },
        };

        const vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<std::uint32_t>(std::size(bindings)),    
            .pBindings    = bindings,
        };
        return vk::raii::DescriptorSetLayout{device, ci};
    }

    vk::raii::DescriptorPool create_sample_pool(const vk::raii::Device& device) {
        const vk::DescriptorPoolSize pool_size{
            .type            = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 4,
        };
        const vk::DescriptorPoolCreateInfo ci{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 1,
            .poolSizeCount = 1,
            .pPoolSizes    = &pool_size,
        };
        return vk::raii::DescriptorPool{device, ci};
    }

    vk::raii::DescriptorSet allocate_sample_set(const vk::raii::Device& device,
                                                const vk::raii::DescriptorPool& pool,
                                                const vk::raii::DescriptorSetLayout& layout) {
        const vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *pool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &*layout,
        };
        auto sets = vk::raii::DescriptorSets{device, ai};
        return std::move(sets.front());
    }

    void update_sample_set(const vk::raii::Device& device,
                           const vk::raii::DescriptorSet& set,
                           const vk::memory::Buffer& rays,
                           const vk::memory::Buffer& samples,
                           const vk::memory::Buffer& evals,
                           const vk::memory::Buffer& indices) {
        const vk::DescriptorBufferInfo ray_info{*rays.buffer, 0, rays.size};    
        const vk::DescriptorBufferInfo sample_info{*samples.buffer, 0, samples.size};
        const vk::DescriptorBufferInfo eval_info{*evals.buffer, 0, evals.size};
        const vk::DescriptorBufferInfo index_info{*indices.buffer, 0, indices.size};
        const vk::WriteDescriptorSet writes[] = {
            {
                .dstSet          = *set,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &ray_info,
            },
            {
                .dstSet          = *set,
                .dstBinding      = 1,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &sample_info,
            },
            {
                .dstSet          = *set,
                .dstBinding      = 2,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &eval_info,
            },
            {
                .dstSet          = *set,
                .dstBinding      = 3,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &index_info,
            },
        };
        device.updateDescriptorSets(writes, {});
    }

    vk::pipeline::GraphicsPipeline create_sample_pipeline(const vk::context::VulkanContext& vkctx,
                                                          const vk::swapchain::Swapchain& sc,
                                                          const vk::raii::DescriptorSetLayout& set_layout) {
        vk::pipeline::VertexInput vin{};
        constexpr std::array paths{"../shaders/sample_points_v2.spv", "../shaders/sample_points_v2.spv"};
        const auto spv = read_shader_bytes(paths);
        auto shader    = vk::pipeline::load_shader_module(vkctx.device, spv);

        vk::pipeline::GraphicsPipelineDesc desc{};
        desc.color_format         = sc.format;
        desc.depth_format         = sc.depth_format;
        desc.use_depth            = true;
        desc.cull                 = vk::CullModeFlagBits::eNone;
        desc.front_face           = vk::FrontFace::eCounterClockwise;
        desc.polygon_mode         = vk::PolygonMode::eFill;
        desc.topology             = vk::PrimitiveTopology::ePointList;
        desc.enable_blend         = true;
        desc.push_constant_bytes  = sizeof(SamplePush);
        desc.push_constant_stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        const vk::DescriptorSetLayout layouts[] = {*set_layout};
        desc.set_layouts = layouts;
        return vk::pipeline::create_graphics_pipeline(vkctx.device, vin, desc, shader, "vertMain", "fragMain");
    }

    // =========================================================================
    // Buffer allocation helpers for filtered rays and indirect draw command.
    // =========================================================================
    vk::memory::Buffer create_device_buffer(const vk::context::VulkanContext& vkctx,
                                            const vk::DeviceSize size,
                                            const vk::BufferUsageFlags usage) {
        if (size == 0) return {};
        return vk::memory::create_buffer(vkctx.physical_device, vkctx.device, size, usage,
                                         vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    void ensure_ray_buffers(const vk::context::VulkanContext& vkctx,
                            const std::uint32_t desired_capacity,
                            const std::size_t element_bytes,
                            vk::memory::Buffer& filtered,
                            vk::memory::Buffer& count,
                            vk::memory::Buffer& indirect,
                            std::uint32_t& capacity) {
        const std::uint32_t safe_capacity = std::max(1u, desired_capacity);
        if (safe_capacity > capacity || filtered.size == 0) {
            const vk::DeviceSize size_bytes = static_cast<vk::DeviceSize>(safe_capacity) * element_bytes;
            filtered = create_device_buffer(vkctx, size_bytes, vk::BufferUsageFlagBits::eStorageBuffer);
            capacity = safe_capacity;
        }

        if (count.size == 0) {
            count = create_device_buffer(vkctx, sizeof(std::uint32_t),
                                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        }

        if (indirect.size == 0) {
            indirect = create_device_buffer(vkctx, sizeof(vk::DrawIndirectCommand),
                                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
        }
    }

    void ensure_sample_buffers(const vk::context::VulkanContext& vkctx,
                               const std::uint32_t desired_capacity,
                               vk::memory::Buffer& indices,
                               vk::memory::Buffer& count,
                               vk::memory::Buffer& indirect,
                               std::uint32_t& capacity) {
        const std::uint32_t safe_capacity = std::max(1u, desired_capacity);
        if (safe_capacity > capacity || indices.size == 0) {
            const vk::DeviceSize size_bytes =
                static_cast<vk::DeviceSize>(safe_capacity) * sizeof(std::uint32_t);
            indices = create_device_buffer(vkctx, size_bytes, vk::BufferUsageFlagBits::eStorageBuffer);
            capacity = safe_capacity;
        }

        if (count.size == 0) {
            count = create_device_buffer(vkctx, sizeof(std::uint32_t),
                                         vk::BufferUsageFlagBits::eStorageBuffer |
                                             vk::BufferUsageFlagBits::eTransferDst);
        }

        if (indirect.size == 0) {
            indirect = create_device_buffer(vkctx, sizeof(vk::DrawIndirectCommand),
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eIndirectBuffer);
        }
    }
} // namespace

// ============================================================================
// App entry point.
// ============================================================================
int main() {
    pngp::vis::rays::RaysInspector app{{}};
    app.run();
    return 0;
}

// ============================================================================
// Main loop: poll input, update camera, draw, and present.
// ============================================================================
void pngp::vis::rays::RaysInspector::run() {
    std::uint32_t frame_index = 0;
    using clock               = std::chrono::steady_clock;
    auto t_prev               = clock::now();
    while (!glfwWindowShouldClose(surface.window.get())) {
        glfwPollEvents();

        // ====================================================================
        // Frame timing with a small clamp to keep camera stable.
        // ====================================================================
        auto t_now = clock::now();
        float dt   = std::chrono::duration<float>(t_now - t_prev).count();
        t_prev     = t_now;
        if (!(dt > 0.0f)) dt = 1.0f / 60.0f;
        dt = std::min(dt, 0.05f);

        // ====================================================================
        // Acquire swapchain image and sync to start a new frame.
        // ====================================================================
        const auto [ok, need_recreate, image_index] = vk::frame::begin_frame(ctx, swapchain, frames, frame_index);
        if (!ok || need_recreate) {
            // =================================================================
            // Swapchain is invalid (resize/minimize). Recreate dependent resources.
            // =================================================================
            vk::swapchain::recreate_swapchain(ctx, surface, swapchain);
            vk::frame::on_swapchain_recreated(ctx, swapchain, frames);
            vk::imgui::set_min_image_count(imgui, 2);
            ctx.device.waitIdle();
            grid_pipeline = create_grid_pipeline(ctx, swapchain);
            ray_pipeline  = create_ray_pipeline(ctx, swapchain, ray_set_layout);
            sample_pipeline = create_sample_pipeline(ctx, swapchain, sample_set_layout);
            continue;
        }
        vk::frame::begin_commands(frames, frame_index);

        // ====================================================================
        // Start a new ImGui frame so UI can collect input state.
        // ====================================================================
        vk::imgui::begin_frame();

        // ====================================================================
        // Build the UI and decide whether the grid geometry needs rebuild.
        // ====================================================================
        const bool rebuild_mesh = imgui_panel();
        if (rebuild_mesh) {
            ctx.device.waitIdle();
            const auto mesh_cpu = build_ground_plane(grid.grid_extent);
            grid_mesh           = upload_mesh_safe(ctx, mesh_cpu);
        }

        // ====================================================================
        // Handle record requests and GPU buffer updates.
        // ====================================================================
        if (request_close_record) {
            request_close_record = false;
            ctx.device.waitIdle();
            record_v2_reader.close();
            v2_view = {};
            active_frame_v2 = {};
            v2_mask_attr_index = -1;
            v2_batch_attr_index = -1;
            ray_input     = {};
            ray_filtered  = {};
            ray_count     = {};
            ray_indirect  = {};
            ray_capacity  = 0;
            v2_samples    = {};
            v2_evals      = {};
            v2_results    = {};
            sample_indices = {};
            sample_count   = {};
            sample_indirect = {};
            sample_capacity = 0;
            v2_attribute_buffers.clear();
            v2_attribute_sets.clear();
            v2_attribute_pool = vk::raii::DescriptorPool{nullptr};
            v2_attribute_capacity = 0;
            rays.roi_initialized = false;
            record_error.clear();
            sample_filter_dirty = true;
        }

        if (request_open_record) {
            request_open_record = false;
            record_error.clear();
            if (record_path_buf[0] == '\0') {
                record_error = "Record path is empty.";
            } else {
                try {
                    record_v2_reader.close();

                    if (record_v2::is_v2_file(record_path_buf.data())) {
                        record_v2_reader.open(record_path_buf.data());
                        v2_mask_attr_index = -1;
                        v2_batch_attr_index = -1;
                        ray_input     = {};
                        ray_filtered  = {};
                        ray_count     = {};
                        ray_indirect  = {};
                        ray_capacity  = 0;
                        v2_samples    = {};
                        v2_evals      = {};
                        v2_results    = {};
                        sample_indices = {};
                        sample_count   = {};
                        sample_indirect = {};
                        sample_capacity = 0;
                        v2_attribute_buffers.clear();
                        v2_attribute_sets.clear();
                        v2_attribute_pool = vk::raii::DescriptorPool{nullptr};
                        v2_attribute_capacity = 0;
                        rays.roi_initialized = false;
                    } else {
                        record_error = "Unsupported record format (expected v2).";
                    }

                    active_frame_index = 0;
                    request_frame_upload = true;
                    filter_dirty = true;
                    sample_filter_dirty = true;
                } catch (const std::exception& e) {
                    record_error = e.what();
                    record_v2_reader.close();
                }
            }
        }

        if (request_frame_upload) {
            request_frame_upload = false;
            if (record_v2_reader.is_open()) {
                v2_view = record_v2_reader.frame_view(active_frame_index);
                active_frame_v2 = v2_view.header;
                v2_ray_index = 0;
                v2_sample_index = 0;
                v2_mask_attr_index = find_attribute_index(v2_view.attributes, "mask_id");
                v2_batch_attr_index = find_attribute_index(v2_view.attributes, "batch_id");

                ctx.device.waitIdle();
                ray_input = {};
                v2_samples = {};
                v2_evals = {};
                v2_results = {};
                v2_attribute_buffers.clear();
                if (!v2_view.rays.empty()) {
                    const auto bytes = std::as_bytes(v2_view.rays);
                    auto buffer = vk::memory::upload_to_device_local_buffer(
                        ctx.physical_device, ctx.device, ctx.command_pool, ctx.graphics_queue, bytes,
                        vk::BufferUsageFlagBits::eStorageBuffer);
                    ray_input = RayBufferGPU{std::move(buffer), static_cast<std::uint64_t>(v2_view.rays.size())};
                }

                v2_samples = upload_storage_buffer(ctx, std::as_bytes(v2_view.samples));
                v2_evals = upload_storage_buffer(ctx, std::as_bytes(v2_view.evals));
                v2_results = upload_storage_buffer(ctx, std::as_bytes(v2_view.results));

                const auto attr_count = static_cast<std::uint32_t>(v2_view.attributes.size());
                ensure_attribute_sets(ctx.device, v2_attribute_set_layout, attr_count,
                                      v2_attribute_pool, v2_attribute_sets, v2_attribute_capacity);
                if (attr_count > 0) {
                    v2_attribute_buffers.reserve(attr_count);
                    for (const auto& attr : v2_view.attributes) {
                        v2_attribute_buffers.push_back(upload_storage_buffer_non_empty(ctx, attr.data));
                    }
                    update_attribute_sets(ctx.device, v2_attribute_sets, v2_attribute_buffers);
                }

                const auto in_count_u64 = std::min<std::uint64_t>(ray_input.count, std::numeric_limits<std::uint32_t>::max());
                const auto in_count = static_cast<std::uint32_t>(in_count_u64);
                if (in_count > 0) {
                    const std::uint32_t desired = std::min(rays.max_rays, in_count);
                    ensure_ray_buffers(ctx, desired, sizeof(record_v2::RayBaseRecordV2),
                                       ray_filtered, ray_count, ray_indirect, ray_capacity);
                    const auto& sample_buf = v2_samples.size > 0 ? v2_samples : dummy_storage;
                    const auto& result_buf = v2_results.size > 0 ? v2_results : dummy_storage;
                    const auto& mask_buf =
                        (v2_mask_attr_index >= 0 && static_cast<std::size_t>(v2_mask_attr_index) < v2_attribute_buffers.size())
                            ? v2_attribute_buffers[static_cast<std::size_t>(v2_mask_attr_index)]
                            : dummy_storage;
                    const auto& batch_buf =
                        (v2_batch_attr_index >= 0 && static_cast<std::size_t>(v2_batch_attr_index) < v2_attribute_buffers.size())
                            ? v2_attribute_buffers[static_cast<std::size_t>(v2_batch_attr_index)]
                            : dummy_storage;
                    filter::update_filter_set(ctx.device, filter_bindings, ray_input.buffer, ray_filtered, ray_count,
                                              sample_buf, mask_buf, batch_buf);
                    update_ray_set(ctx.device, ray_set, ray_filtered, result_buf, mask_buf, batch_buf);
                    const auto& eval_buf = v2_evals.size > 0 ? v2_evals : dummy_storage;
                    const auto& ray_buf  = ray_input.buffer.size > 0 ? ray_input.buffer : dummy_storage;
                    const auto sample_in_u64 = std::min<std::uint64_t>(active_frame_v2.sample_count,
                                                                      std::numeric_limits<std::uint32_t>::max());
                    const auto sample_in_count = static_cast<std::uint32_t>(sample_in_u64);
                    if (sample_in_count > 0) {
                        const std::uint32_t desired_samples = std::min(samples.max_samples, sample_in_count);
                        ensure_sample_buffers(ctx, desired_samples, sample_indices, sample_count,
                                              sample_indirect, sample_capacity);
                    }
                    const auto& index_buf = sample_indices.size > 0 ? sample_indices : dummy_storage;
                    update_sample_set(ctx.device, sample_set, ray_buf, sample_buf, eval_buf, index_buf);
                    if (sample_in_count > 0 && sample_indices.size > 0 && sample_count.size > 0 &&
                        sample_indirect.size > 0) {
                        filter::update_filter_set(ctx.device, sample_filter_bindings, ray_input.buffer,
                                                  sample_indices, sample_count, sample_buf, mask_buf, batch_buf);
                        filter::update_indirect_set(ctx.device, sample_indirect_bindings, sample_count,
                                                    sample_indirect);
                    }
                    filter::update_indirect_set(ctx.device, indirect_bindings, ray_count, ray_indirect);
                    filter_dirty = true;
                    sample_filter_dirty = true;
                }
            }
        }

        if (request_ray_resize) {
            request_ray_resize = false;
            const auto in_count_u64 = std::min<std::uint64_t>(ray_input.count, std::numeric_limits<std::uint32_t>::max());
            const auto in_count = static_cast<std::uint32_t>(in_count_u64);
            if (in_count > 0) {
                const std::uint32_t desired = std::min(rays.max_rays, in_count);
                if (desired > ray_capacity) {
                    ctx.device.waitIdle();
                    ensure_ray_buffers(ctx, desired, sizeof(record_v2::RayBaseRecordV2),
                                       ray_filtered, ray_count, ray_indirect, ray_capacity);
                    const auto& sample_buf = v2_samples.size > 0 ? v2_samples : dummy_storage;
                    const auto& result_buf = v2_results.size > 0 ? v2_results : dummy_storage;
                    const auto& mask_buf =
                        (v2_mask_attr_index >= 0 &&
                         static_cast<std::size_t>(v2_mask_attr_index) < v2_attribute_buffers.size())
                            ? v2_attribute_buffers[static_cast<std::size_t>(v2_mask_attr_index)]
                            : dummy_storage;
                    const auto& batch_buf =
                        (v2_batch_attr_index >= 0 &&
                         static_cast<std::size_t>(v2_batch_attr_index) < v2_attribute_buffers.size())
                            ? v2_attribute_buffers[static_cast<std::size_t>(v2_batch_attr_index)]
                            : dummy_storage;
                    filter::update_filter_set(ctx.device, filter_bindings, ray_input.buffer, ray_filtered, ray_count,
                                              sample_buf, mask_buf, batch_buf);
                    update_ray_set(ctx.device, ray_set, ray_filtered, result_buf, mask_buf, batch_buf);
                    filter::update_indirect_set(ctx.device, indirect_bindings, ray_count, ray_indirect);
                    filter_dirty = true;
                }
            }
        }

        if (request_sample_resize) {
            request_sample_resize = false;
            const auto sample_in_u64 = std::min<std::uint64_t>(active_frame_v2.sample_count,
                                                              std::numeric_limits<std::uint32_t>::max());
            const auto sample_in_count = static_cast<std::uint32_t>(sample_in_u64);
            if (sample_in_count > 0) {
                const std::uint32_t desired_samples = std::min(samples.max_samples, sample_in_count);
                if (desired_samples > sample_capacity) {
                    ctx.device.waitIdle();
                    ensure_sample_buffers(ctx, desired_samples, sample_indices, sample_count,
                                          sample_indirect, sample_capacity);
                    const auto& sample_buf = v2_samples.size > 0 ? v2_samples : dummy_storage;
                    const auto& eval_buf = v2_evals.size > 0 ? v2_evals : dummy_storage;
                    const auto& ray_buf  = ray_input.buffer.size > 0 ? ray_input.buffer : dummy_storage;
                    const auto& mask_buf =
                        (v2_mask_attr_index >= 0 &&
                         static_cast<std::size_t>(v2_mask_attr_index) < v2_attribute_buffers.size())
                            ? v2_attribute_buffers[static_cast<std::size_t>(v2_mask_attr_index)]
                            : dummy_storage;
                    const auto& batch_buf =
                        (v2_batch_attr_index >= 0 &&
                         static_cast<std::size_t>(v2_batch_attr_index) < v2_attribute_buffers.size())
                            ? v2_attribute_buffers[static_cast<std::size_t>(v2_batch_attr_index)]
                            : dummy_storage;
                    filter::update_filter_set(ctx.device, sample_filter_bindings, ray_input.buffer,
                                              sample_indices, sample_count, sample_buf, mask_buf, batch_buf);
                    filter::update_indirect_set(ctx.device, sample_indirect_bindings, sample_count,
                                                sample_indirect);
                    update_sample_set(ctx.device, sample_set, ray_buf, sample_buf, eval_buf, sample_indices);
                }
                sample_filter_dirty = true;
            }
        }

        // ====================================================================
        // Apply camera mode and prepare input for the controller.
        // ====================================================================
        cam.set_mode(grid.fly_mode ? vk::camera::Mode::Fly : vk::camera::Mode::Orbit);

        const ImGuiIO& io      = ImGui::GetIO();
        const bool block_mouse = io.WantCaptureMouse;
        const bool block_kbd   = io.WantCaptureKeyboard;

        // ====================================================================
        // Respect ImGui capture flags so the UI can own the mouse/keyboard.
        // ====================================================================
        vk::camera::CameraInput ci{};
        ci.lmb = !block_mouse && input.lmb;
        ci.mmb = !block_mouse && input.mmb;
        ci.rmb = !block_mouse && input.rmb;

        ci.mouse_dx = !block_mouse ? input.dx : 0.0f;
        ci.mouse_dy = !block_mouse ? input.dy : 0.0f;
        ci.scroll   = !block_mouse ? input.scroll : 0.0f;

        ci.shift = !block_kbd && (input.keys[GLFW_KEY_LEFT_SHIFT] || input.keys[GLFW_KEY_RIGHT_SHIFT]);
        ci.ctrl  = !block_kbd && (input.keys[GLFW_KEY_LEFT_CONTROL] || input.keys[GLFW_KEY_RIGHT_CONTROL]);
        ci.alt   = !block_kbd && (input.keys[GLFW_KEY_LEFT_ALT] || input.keys[GLFW_KEY_RIGHT_ALT]);
        ci.space = !block_kbd && input.keys[GLFW_KEY_SPACE];

        ci.forward  = !block_kbd && input.keys[GLFW_KEY_W];
        ci.backward = !block_kbd && input.keys[GLFW_KEY_S];
        ci.left     = !block_kbd && input.keys[GLFW_KEY_A];
        ci.right    = !block_kbd && input.keys[GLFW_KEY_D];
        ci.down     = !block_kbd && input.keys[GLFW_KEY_Q];
        ci.up       = !block_kbd && input.keys[GLFW_KEY_E];

        // ====================================================================
        // Update camera matrices (view/projection) for this frame.
        // ====================================================================
        cam.update(dt, swapchain.extent.width, swapchain.extent.height, ci);
        vk::imgui::draw_mini_axis_gizmo(cam.matrices().c2w);

        // ====================================================================
        // Consume per-frame deltas so callbacks accumulate fresh movement.
        // ====================================================================
        input.dx     = 0.0f;
        input.dy     = 0.0f;
        input.scroll = 0.0f;

        // ====================================================================
        // Cache per-frame MVP for the grid draw call.
        // ====================================================================
        grid_mvp = cam.matrices().view_proj;

        // ====================================================================
        // Ray picking (screen -> world) on click.
        // ====================================================================
        if (pick.enable && !block_mouse && record_v2_reader.is_open() &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const ImVec2 mouse = ImGui::GetIO().MousePos;
            if (const auto picked = pick_ray_index(v2_view, active_frame_v2, rays, pick, cam,
                                                   swapchain.extent, mouse,
                                                   v2_mask_attr_index, v2_batch_attr_index)) {
                last_picked_ray_index = picked->index;
                last_pick_distance = std::sqrt(picked->dist2);
                v2_ray_index = picked->index;
                v2_sample_index = 0;
                if (samples.isolate_ray) {
                    sample_filter_dirty = true;
                }
                if (pick.auto_pin) {
                    pinned_ray_index = picked->index;
                    pinned_sample_index = 0;
                }
            } else {
                last_picked_ray_index = -1;
            }
        }

        // ====================================================================
        // Record GPU work for this frame (grid + ImGui).
        // ====================================================================
        record_commands(frame_index, image_index);

        // ====================================================================
        // Present the frame; recreate swapchain if presentation fails.
        // ====================================================================
        if (vk::frame::end_frame(ctx, swapchain, frames, frame_index, image_index)) {
            vk::swapchain::recreate_swapchain(ctx, surface, swapchain);
            vk::frame::on_swapchain_recreated(ctx, swapchain, frames);
            vk::imgui::set_min_image_count(imgui, 2);
            ctx.device.waitIdle();
            grid_pipeline = create_grid_pipeline(ctx, swapchain);
            ray_pipeline  = create_ray_pipeline(ctx, swapchain, ray_set_layout);
            sample_pipeline = create_sample_pipeline(ctx, swapchain, sample_set_layout);
        }

        frame_index = (frame_index + 1) % frames.frames_in_flight;
    }
    ctx.device.waitIdle();
    vk::imgui::shutdown(imgui);
}

// ============================================================================
// Constructor: create Vulkan systems and the initial grid resources.
// ============================================================================
pngp::vis::rays::RaysInspector::RaysInspector(const RaysInspectorInfo& info) {
    auto [vkctx, surface] = vk::context::setup_vk_context_glfw("Dataset Viewer", "Engine");

    ctx           = std::move(vkctx);
    this->surface = std::move(surface);

    // ========================================================================
    // Connect GLFW callbacks before ImGui init so it can chain them.
    // ========================================================================
    glfwSetWindowUserPointer(this->surface.window.get(), &input);
    glfwSetKeyCallback(this->surface.window.get(), &glfw_key_cb);
    glfwSetMouseButtonCallback(this->surface.window.get(), &glfw_mouse_button_cb);
    glfwSetCursorPosCallback(this->surface.window.get(), &glfw_cursor_pos_cb);
    glfwSetScrollCallback(this->surface.window.get(), &glfw_scroll_cb);

    swapchain = vk::swapchain::setup_swapchain(ctx, this->surface);
    frames    = vk::frame::create_frame_system(ctx, swapchain, 2);
    imgui     = vk::imgui::create(ctx, this->surface.window.get(), swapchain.format, 2, static_cast<std::uint32_t>(swapchain.images.size()), info.render.enable_docking, info.render.enable_viewports);

    // ========================================================================
    // Camera defaults tuned for a comfortable workspace view.
    // ========================================================================
    vk::camera::CameraConfig cam_cfg{};
    cam_cfg.fov_y_rad = info.render.fov_y_rad;
    cam_cfg.znear     = info.render.near_plane;
    cam_cfg.zfar      = info.render.far_plane;
    cam.set_config(cam_cfg);
    cam.home();
    cam.set_mode(vk::camera::Mode::Orbit);
    {
        auto st           = cam.state();
        st.orbit.distance = std::max(1.0f, grid.grid_extent * 1.15f);
        cam.set_state(st);
    }

    // ========================================================================
    // Create grid resources once at startup.
    // ========================================================================
    const auto mesh_cpu = build_ground_plane(grid.grid_extent);
    grid_mesh           = upload_mesh_safe(ctx, mesh_cpu);
    grid_pipeline       = create_grid_pipeline(ctx, swapchain);

    // ========================================================================
    // Ray rendering + compute pipelines.
    // ========================================================================
    ray_set_layout   = create_ray_set_layout(ctx.device);
    ray_pool         = create_ray_pool(ctx.device);
    ray_set          = allocate_ray_set(ctx.device, ray_pool, ray_set_layout);
    ray_pipeline     = create_ray_pipeline(ctx, swapchain, ray_set_layout);
    v2_attribute_set_layout = create_attribute_set_layout(ctx.device);
    sample_set_layout = create_sample_set_layout(ctx.device);
    sample_pool       = create_sample_pool(ctx.device);
    sample_set        = allocate_sample_set(ctx.device, sample_pool, sample_set_layout);
    sample_pipeline   = create_sample_pipeline(ctx, swapchain, sample_set_layout);
    {
        std::array<std::byte, 64> zeros{};
        dummy_storage = upload_storage_buffer_non_empty(ctx, zeros);
    }

    filter_pipeline  = filter::create_filter_pipeline(ctx.device);
    filter_bindings  = filter::create_filter_bindings(ctx.device);
    filter::allocate_filter_set(ctx.device, filter_pipeline, filter_bindings);  

    indirect_pipeline = filter::create_indirect_pipeline(ctx.device);
    indirect_bindings = filter::create_indirect_bindings(ctx.device);
    filter::allocate_indirect_set(ctx.device, indirect_pipeline, indirect_bindings);

    sample_filter_pipeline = filter::create_sample_filter_pipeline(ctx.device);
    sample_filter_bindings = filter::create_filter_bindings(ctx.device);
    filter::allocate_filter_set(ctx.device, sample_filter_pipeline, sample_filter_bindings);

    sample_indirect_pipeline = filter::create_sample_indirect_pipeline(ctx.device);
    sample_indirect_bindings = filter::create_indirect_bindings(ctx.device);
    filter::allocate_indirect_set(ctx.device, sample_indirect_pipeline, sample_indirect_bindings);
}

// ============================================================================
// Record a frame: render grid, then ImGui.
// ============================================================================
void pngp::vis::rays::RaysInspector::record_commands(std::uint32_t frame_index, std::uint32_t image_index) {
    auto& cmd = vk::frame::cmd(frames, frame_index);

    // ========================================================================
    // Transition swapchain color image for rendering.
    // ========================================================================
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

    // ========================================================================
    // Transition depth image for depth testing.
    // ========================================================================
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

    // ========================================================================
    // GPU filter + indirect build for rays (compute before rendering).
    // ========================================================================
    const auto in_count_u64 = std::min<std::uint64_t>(ray_input.count, std::numeric_limits<std::uint32_t>::max());
    const auto in_count     = static_cast<std::uint32_t>(in_count_u64);
    const bool can_dispatch = rays.show_rays && in_count > 0 && ray_capacity > 0 &&
                              ray_filtered.size > 0 && ray_count.size > 0 && ray_indirect.size > 0;
    const std::uint32_t ray_stride  = std::max(1u, rays.stride);
    const std::uint32_t ray_max_out = std::min(std::min(rays.max_rays, ray_capacity), in_count);

    const auto sample_in_u64 = std::min<std::uint64_t>(active_frame_v2.sample_count,
                                                       std::numeric_limits<std::uint32_t>::max());
    const auto sample_in_count = static_cast<std::uint32_t>(sample_in_u64);
    const bool sample_can_dispatch =
        samples.show_samples && sample_in_count > 0 && sample_capacity > 0 && sample_indices.size > 0 &&
        sample_count.size > 0 && sample_indirect.size > 0;
    const std::uint32_t sample_stride  = std::max(1u, samples.stride);
    const std::uint32_t sample_max_out =
        std::min(std::min(samples.max_samples, sample_capacity), sample_in_count);

    if (can_dispatch && ray_max_out > 0 && filter_dirty) {
        cmd.fillBuffer(*ray_count.buffer, 0, ray_count.size, 0);

        const vk::BufferMemoryBarrier2 clear_barrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .buffer        = *ray_count.buffer,
            .offset        = 0,
            .size          = ray_count.size,
        };

        const vk::DependencyInfo clear_dep{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &clear_barrier,
        };
        cmd.pipelineBarrier2(clear_dep);

        const auto& filter_pipe = filter_pipeline;
        filter::FilterParams params{};
        params.in_count = in_count;
        params.stride   = ray_stride;
        params.max_out  = ray_max_out;
        params.flags    = 0;
        params.sample_state_mask = 0;
        params.omit_reason_mask  = 0;
        params.mask_id  = rays.mask_id;
        params.batch_id = rays.batch_id;
        params.ray_index = static_cast<std::uint32_t>(std::max(0, v2_ray_index));
        params.ray_index = 0;

        const std::uint32_t frame_width = active_frame_v2.width;
        const std::uint32_t frame_height = active_frame_v2.height;
        const int max_x = frame_width > 0 ? static_cast<int>(frame_width - 1) : 0;
        const int max_y = frame_height > 0 ? static_cast<int>(frame_height - 1) : 0;
        int roi_min_x = std::clamp(rays.roi_min_x, 0, max_x);
        int roi_min_y = std::clamp(rays.roi_min_y, 0, max_y);
        int roi_max_x = std::clamp(rays.roi_max_x, 0, max_x);
        int roi_max_y = std::clamp(rays.roi_max_y, 0, max_y);
        if (roi_min_x > roi_max_x) std::swap(roi_min_x, roi_max_x);
        if (roi_min_y > roi_max_y) std::swap(roi_min_y, roi_max_y);
        params.roi_min_x = static_cast<std::uint32_t>(roi_min_x);
        params.roi_min_y = static_cast<std::uint32_t>(roi_min_y);
        params.roi_max_x = static_cast<std::uint32_t>(roi_max_x);
        params.roi_max_y = static_cast<std::uint32_t>(roi_max_y);

        if (rays.roi_enabled && frame_width > 0 && frame_height > 0) {
            params.flags |= k_filter_roi;
        }
        if ((rays.filter_sample_state || rays.filter_omit_reason) && v2_samples.size > 0) {
            params.flags |= k_filter_sample;
            params.sample_state_mask = rays.filter_sample_state ? rays.sample_state_mask : 0u;
            params.omit_reason_mask  = rays.filter_omit_reason ? rays.omit_reason_mask : 0u;
        }
        if (rays.filter_mask_id && v2_mask_attr_index >= 0) {
            params.flags |= k_filter_mask_id;
        }
        if (rays.filter_batch_id && v2_batch_attr_index >= 0) {
            params.flags |= k_filter_batch_id;
        }
        if (samples.isolate_ray) {
            params.flags |= k_filter_ray_index;
        }
        const std::uint32_t group_count = (in_count + 255u) / 256u;
        filter::dispatch_filter(cmd, filter_pipe, filter_bindings, params, group_count);

        const vk::BufferMemoryBarrier2 count_barrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .buffer        = *ray_count.buffer,
            .offset        = 0,
            .size          = ray_count.size,
        };

        const vk::DependencyInfo count_dep{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &count_barrier,
        };
        cmd.pipelineBarrier2(count_dep);

        filter::dispatch_indirect(cmd, indirect_pipeline, indirect_bindings);

        const vk::BufferMemoryBarrier2 barriers[] = {
            {
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eVertexShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .buffer        = *ray_filtered.buffer,
                .offset        = 0,
                .size          = ray_filtered.size,
            },
            {
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect,
                .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
                .buffer        = *ray_indirect.buffer,
                .offset        = 0,
                .size          = ray_indirect.size,
            },
        };

        const vk::DependencyInfo ray_dep{
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(std::size(barriers)),
            .pBufferMemoryBarriers    = barriers,
        };
        cmd.pipelineBarrier2(ray_dep);
        filter_dirty = false;
    }

    if (sample_can_dispatch && sample_max_out > 0 && sample_filter_dirty) {
        cmd.fillBuffer(*sample_count.buffer, 0, sample_count.size, 0);

        const vk::BufferMemoryBarrier2 clear_barrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .buffer        = *sample_count.buffer,
            .offset        = 0,
            .size          = sample_count.size,
        };

        const vk::DependencyInfo clear_dep{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &clear_barrier,
        };
        cmd.pipelineBarrier2(clear_dep);

        filter::FilterParams params{};
        params.in_count = sample_in_count;
        params.stride   = sample_stride;
        params.max_out  = sample_max_out;
        params.flags    = 0;
        params.sample_state_mask = 0;
        params.omit_reason_mask  = 0;
        params.mask_id  = rays.mask_id;
        params.batch_id = rays.batch_id;

        const std::uint32_t frame_width = active_frame_v2.width;
        const std::uint32_t frame_height = active_frame_v2.height;
        const int max_x = frame_width > 0 ? static_cast<int>(frame_width - 1) : 0;
        const int max_y = frame_height > 0 ? static_cast<int>(frame_height - 1) : 0;
        int roi_min_x = std::clamp(rays.roi_min_x, 0, max_x);
        int roi_min_y = std::clamp(rays.roi_min_y, 0, max_y);
        int roi_max_x = std::clamp(rays.roi_max_x, 0, max_x);
        int roi_max_y = std::clamp(rays.roi_max_y, 0, max_y);
        if (roi_min_x > roi_max_x) std::swap(roi_min_x, roi_max_x);
        if (roi_min_y > roi_max_y) std::swap(roi_min_y, roi_max_y);
        params.roi_min_x = static_cast<std::uint32_t>(roi_min_x);
        params.roi_min_y = static_cast<std::uint32_t>(roi_min_y);
        params.roi_max_x = static_cast<std::uint32_t>(roi_max_x);
        params.roi_max_y = static_cast<std::uint32_t>(roi_max_y);

        if (rays.roi_enabled && frame_width > 0 && frame_height > 0) {
            params.flags |= k_filter_roi;
        }
        if ((rays.filter_sample_state || rays.filter_omit_reason) && v2_samples.size > 0) {
            params.flags |= k_filter_sample;
            params.sample_state_mask = rays.filter_sample_state ? rays.sample_state_mask : 0u;
            params.omit_reason_mask  = rays.filter_omit_reason ? rays.omit_reason_mask : 0u;
        }
        if (rays.filter_mask_id && v2_mask_attr_index >= 0) {
            params.flags |= k_filter_mask_id;
        }
        if (rays.filter_batch_id && v2_batch_attr_index >= 0) {
            params.flags |= k_filter_batch_id;
        }

        const std::uint32_t group_count = (sample_in_count + 255u) / 256u;
        filter::dispatch_filter(cmd, sample_filter_pipeline, sample_filter_bindings, params, group_count);

        const vk::BufferMemoryBarrier2 count_barrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .buffer        = *sample_count.buffer,
            .offset        = 0,
            .size          = sample_count.size,
        };

        const vk::DependencyInfo count_dep{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &count_barrier,
        };
        cmd.pipelineBarrier2(count_dep);

        filter::dispatch_indirect(cmd, sample_indirect_pipeline, sample_indirect_bindings);

        const vk::BufferMemoryBarrier2 barriers[] = {
            {
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eVertexShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .buffer        = *sample_indices.buffer,
                .offset        = 0,
                .size          = sample_indices.size,
            },
            {
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect,
                .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
                .buffer        = *sample_indirect.buffer,
                .offset        = 0,
                .size          = sample_indirect.size,
            },
        };

        const vk::DependencyInfo sample_dep{
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(std::size(barriers)),
            .pBufferMemoryBarriers    = barriers,
        };
        cmd.pipelineBarrier2(sample_dep);
        sample_filter_dirty = false;
    }

    // ========================================================================
    // Clear targets: black background + default depth.
    // ========================================================================
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

    // ========================================================================
    // Grid draw: one quad + procedural shader.
    // ========================================================================
    const bool grid_visible = grid.show_grid || grid.show_axes || grid.show_origin;
    if (grid_mesh.index_count > 0 && grid_visible) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *grid_pipeline.pipeline);
        const GridPush push = make_grid_push(grid, grid_mvp);
        cmd.pushConstants(*grid_pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, vk::ArrayProxy<const GridPush>{push});

        vk::DeviceSize offset = 0;
        cmd.bindVertexBuffers(0, {*grid_mesh.vertex_buffer.buffer}, {offset});
        cmd.bindIndexBuffer(*grid_mesh.index_buffer.buffer, 0, vk::IndexType::eUint32);
        cmd.drawIndexed(grid_mesh.index_count, 1, 0, 0, 0);
    }

    // ========================================================================
    // Ray draw: compacted line list + indirect draw.
    // ========================================================================
    if (can_dispatch && ray_max_out > 0) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *ray_pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *ray_pipeline.layout, 0, {*ray_set}, {});

        const RayPush push = make_ray_push(rays, grid_mvp);
        cmd.pushConstants(*ray_pipeline.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                          vk::ArrayProxy<const RayPush>{push});
        cmd.drawIndirect(*ray_indirect.buffer, 0, 1, sizeof(vk::DrawIndirectCommand));
    }

    // ========================================================================
    // Sample draw: v2 sample points (GPU compaction + indirect draw).
    // ========================================================================
    const bool sample_draw_ready = sample_can_dispatch && v2_samples.size > 0;
    if (record_v2_reader.is_open() && sample_draw_ready) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *sample_pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *sample_pipeline.layout, 0, {*sample_set}, {});
        SamplePush push = make_sample_push(samples, grid_mvp, cam.matrices().eye);
        push.stride = 1u;
        cmd.pushConstants(*sample_pipeline.layout,
                          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                          vk::ArrayProxy<const SamplePush>{push});
        cmd.drawIndirect(*sample_indirect.buffer, 0, 1, sizeof(vk::DrawIndirectCommand));
    }

    cmd.endRendering();

    // ========================================================================
    // ImGui pass (draw UI on top of scene).
    // ========================================================================
    vk::imgui::render(imgui, cmd, swapchain.extent, *swapchain.image_views[image_index], vk::ImageLayout::eColorAttachmentOptimal);
    vk::imgui::end_frame();

    // ========================================================================
    // Transition swapchain image for presentation.
    // ========================================================================
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

// ============================================================================
// ImGui panel: return true when geometry should be rebuilt.
// ============================================================================
bool pngp::vis::rays::RaysInspector::imgui_panel() {
    bool rebuild = false;
    ImGui::Begin("Rays Inspector");
    ImGui::TextUnformatted("Ground Plane");
    ImGui::Checkbox("Show grid", &grid.show_grid);
    ImGui::Checkbox("Show axes", &grid.show_axes);
    ImGui::Checkbox("Show origin", &grid.show_origin);
    ImGui::Separator();
    rebuild |= ImGui::SliderFloat("Grid extent", &grid.grid_extent, 2.0f, 100.0f);
    ImGui::SliderFloat("Grid step", &grid.grid_step, 0.1f, 5.0f);
    ImGui::SliderInt("Major every", &grid.major_every, 1, 20);
    ImGui::SliderFloat("Axis length", &grid.axis_length, 0.5f, 20.0f);
    ImGui::SliderFloat("Origin scale", &grid.origin_scale, 0.05f, 2.0f);
    ImGui::Separator();
    ImGui::Checkbox("Fly mode", &grid.fly_mode);
    ImGui::TextUnformatted("Orbit: Alt/Space + LMB rotate, MMB pan, wheel zoom");
    ImGui::TextUnformatted("Fly: RMB look + WASD move, Q/E down/up");

    ImGui::Separator();
    ImGui::TextUnformatted("Ray Record");
    ImGui::InputText("Record path", record_path_buf.data(), record_path_buf.size());
    if (ImGui::Button("Open")) request_open_record = true;
    ImGui::SameLine();
    if (ImGui::Button("Close")) request_close_record = true;
    if (!record_error.empty()) ImGui::TextWrapped("%s", record_error.c_str());

    if (record_v2_reader.is_open()) {
        const auto total_frames = record_v2_reader.frame_count();
        ImGui::Text("Frames: %zu", total_frames);
        if (total_frames > 0) {
            int frame = static_cast<int>(active_frame_index);
            const int max_frame = static_cast<int>(total_frames - 1);
            if (ImGui::SliderInt("Frame", &frame, 0, max_frame)) {
                active_frame_index = static_cast<std::size_t>(frame);
                request_frame_upload = true;
            }
            ImGui::Text("Rays: %llu", static_cast<unsigned long long>(active_frame_v2.ray_count));
            ImGui::Text("Samples: %llu", static_cast<unsigned long long>(active_frame_v2.sample_count));
        }
    }

    if (record_v2_reader.is_open() && !v2_view.rays.empty()) {
        ImGui::Separator();
        ImGui::TextUnformatted("v2 CPU Inspection");

        const int max_ray = static_cast<int>(v2_view.rays.size() - 1);
        v2_ray_index = std::clamp(v2_ray_index, 0, max_ray);
        if (ImGui::SliderInt("Ray index", &v2_ray_index, 0, max_ray)) {
            v2_sample_index = 0;
            if (samples.isolate_ray) {
                sample_filter_dirty = true;
            }
        }

        const auto& ray = v2_view.rays[static_cast<std::size_t>(v2_ray_index)];
        ImGui::Text("Origin: %.3f %.3f %.3f", ray.ox, ray.oy, ray.oz);
        ImGui::Text("Dir: %.3f %.3f %.3f", ray.dx, ray.dy, ray.dz);
        ImGui::Text("Pixel: %u %u", ray.pixel_x, ray.pixel_y);
        ImGui::Text("Samples: %u", ray.sample_count);

        const std::size_t sample_base = static_cast<std::size_t>(ray.sample_offset);
        const std::size_t sample_count = static_cast<std::size_t>(ray.sample_count);
        if (sample_count > 0 && sample_base + sample_count <= v2_view.samples.size()) {
            const int max_sample = static_cast<int>(sample_count - 1);
            v2_sample_index = std::clamp(v2_sample_index, 0, max_sample);
            ImGui::SliderInt("Sample index", &v2_sample_index, 0, max_sample);

            const auto& sample = v2_view.samples[sample_base + static_cast<std::size_t>(v2_sample_index)];
            ImGui::Text("Sample t: %.3f, dt: %.3f", sample.t, sample.dt);
            ImGui::Text("State: %u, Omit: %u", sample.state, sample.omit_reason);

            if (sample_base + static_cast<std::size_t>(v2_sample_index) < v2_view.evals.size()) {
                const auto& eval = v2_view.evals[sample_base + static_cast<std::size_t>(v2_sample_index)];
                ImGui::Text("Eval density: %.3f weight: %.3f", eval.density, eval.weight);
                ImGui::Text("Contrib: %.3f %.3f %.3f", eval.contrib_r, eval.contrib_g, eval.contrib_b);
            }
        }

        if (ray.result_index < v2_view.results.size()) {
            const auto& res = v2_view.results[ray.result_index];
            ImGui::Text("Result rgb: %.3f %.3f %.3f", res.r, res.g, res.b);
            ImGui::Text("Alpha: %.3f, Depth: %.3f", res.alpha, res.depth);
            ImGui::Text("Term: %u, Steps: %u", res.termination_reason, res.step_count);
        }
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Ray Visualization");
    ImGui::Checkbox("Show rays", &rays.show_rays);
    {
        const char* modes[] = {
            "Direction",
            "Flags",
            "Mask id",
            "Batch id",
            "Result RGB",
            "Depth",
        };
        int mode = static_cast<int>(rays.color_mode);
        if (ImGui::Combo("Ray color", &mode, modes, static_cast<int>(std::size(modes)))) {
            rays.color_mode = static_cast<pngp::vis::rays::RayColorMode>(mode);
        }
        if (rays.color_mode == pngp::vis::rays::RayColorMode::Depth) {
            ImGui::SliderFloat("Depth min", &rays.depth_min, 0.0f, rays.depth_max);
            ImGui::SliderFloat("Depth max", &rays.depth_max, rays.depth_min, rays.depth_min + 1000.0f);
        }
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Ray Filter");
    ImGui::SliderFloat("Line length", &rays.line_length, 0.1f, 100.0f);
    ImGui::SliderFloat("Opacity", &rays.opacity, 0.05f, 1.0f);
    {
        int stride = static_cast<int>(rays.stride);
        if (ImGui::SliderInt("Stride", &stride, 1, 64)) {
            rays.stride = static_cast<std::uint32_t>(stride);
            filter_dirty = true;
        }
    }
    {
        int max_rays = static_cast<int>(rays.max_rays);
        if (ImGui::SliderInt("Max rays", &max_rays, 1, 2000000)) {
            rays.max_rays = static_cast<std::uint32_t>(max_rays);
            request_ray_resize = true;
            filter_dirty = true;
        }
    }

    const bool have_frame =
        record_v2_reader.is_open() && active_frame_v2.width > 0 && active_frame_v2.height > 0;
    const int max_x = have_frame ? static_cast<int>(active_frame_v2.width - 1) : 0;
    const int max_y = have_frame ? static_cast<int>(active_frame_v2.height - 1) : 0;

    if (!rays.roi_initialized && have_frame) {
        rays.roi_min_x = 0;
        rays.roi_min_y = 0;
        rays.roi_max_x = max_x;
        rays.roi_max_y = max_y;
        rays.roi_initialized = true;
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Advanced Filters");
    if (ImGui::Checkbox("Enable ROI", &rays.roi_enabled)) {
        filter_dirty = true;
        sample_filter_dirty = true;
    }
    if (rays.roi_enabled) {
        int roi_min[2] = {rays.roi_min_x, rays.roi_min_y};
        int roi_max[2] = {rays.roi_max_x, rays.roi_max_y};
        if (ImGui::InputInt2("ROI min", roi_min)) {
            rays.roi_min_x = std::clamp(roi_min[0], 0, max_x);
            rays.roi_min_y = std::clamp(roi_min[1], 0, max_y);
            filter_dirty = true;
            sample_filter_dirty = true;
        }
        if (ImGui::InputInt2("ROI max", roi_max)) {
            rays.roi_max_x = std::clamp(roi_max[0], 0, max_x);
            rays.roi_max_y = std::clamp(roi_max[1], 0, max_y);
            filter_dirty = true;
            sample_filter_dirty = true;
        }
    }

    const bool v2_sample_available =
        record_v2_reader.is_open() && !v2_view.samples.empty();
    ImGui::BeginDisabled(!v2_sample_available);
    if (ImGui::Checkbox("Filter sample state", &rays.filter_sample_state)) {
        filter_dirty = true;
        sample_filter_dirty = true;
    }
    if (rays.filter_sample_state) {
        std::uint32_t mask = rays.sample_state_mask;
        bool candidate = (mask & (1u << 0)) != 0;
        bool kept      = (mask & (1u << 1)) != 0;
        bool omitted   = (mask & (1u << 2)) != 0;
        bool term      = (mask & (1u << 3)) != 0;
        if (ImGui::Checkbox("State: candidate", &candidate)) {
            mask = candidate ? (mask | (1u << 0)) : (mask & ~(1u << 0));
            filter_dirty = true;
            sample_filter_dirty = true;
        }
        if (ImGui::Checkbox("State: kept", &kept)) {
            mask = kept ? (mask | (1u << 1)) : (mask & ~(1u << 1));
            filter_dirty = true;
            sample_filter_dirty = true;
        }
        if (ImGui::Checkbox("State: omitted", &omitted)) {
            mask = omitted ? (mask | (1u << 2)) : (mask & ~(1u << 2));
            filter_dirty = true;
            sample_filter_dirty = true;
        }
        if (ImGui::Checkbox("State: terminated", &term)) {
            mask = term ? (mask | (1u << 3)) : (mask & ~(1u << 3));
            filter_dirty = true;
            sample_filter_dirty = true;
        }
        rays.sample_state_mask = mask;
    }

    if (ImGui::Checkbox("Filter omit reason", &rays.filter_omit_reason)) {
        filter_dirty = true;
        sample_filter_dirty = true;
    }
    if (rays.filter_omit_reason) {
        std::uint32_t mask = rays.omit_reason_mask;
        const char* labels[] = {
            "Omit: none",
            "Omit: occupancy",
            "Omit: alpha",
            "Omit: bounds",
            "Omit: step limit",
            "Omit: density thresh",
            "Omit: user mask",
            "Omit: other",
        };
        for (std::uint32_t i = 0; i < 8; ++i) {
            bool on = (mask & (1u << i)) != 0;
            if (ImGui::Checkbox(labels[i], &on)) {
                mask = on ? (mask | (1u << i)) : (mask & ~(1u << i));
                filter_dirty = true;
                sample_filter_dirty = true;
            }
        }
        rays.omit_reason_mask = mask;
    }
    ImGui::EndDisabled();

    const bool has_mask_attr = record_v2_reader.is_open() && v2_mask_attr_index >= 0;
    const bool has_batch_attr = record_v2_reader.is_open() && v2_batch_attr_index >= 0;
    ImGui::BeginDisabled(!has_mask_attr);
    if (ImGui::Checkbox("Filter mask id", &rays.filter_mask_id)) {
        filter_dirty = true;
        sample_filter_dirty = true;
    }
    if (rays.filter_mask_id) {
        int value = static_cast<int>(rays.mask_id);
        if (ImGui::InputInt("Mask id", &value)) {
            rays.mask_id = static_cast<std::uint32_t>(std::max(0, value));
            filter_dirty = true;
            sample_filter_dirty = true;
        }
    }
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!has_batch_attr);
    if (ImGui::Checkbox("Filter batch id", &rays.filter_batch_id)) {
        filter_dirty = true;
        sample_filter_dirty = true;
    }
    if (rays.filter_batch_id) {
        int value = static_cast<int>(rays.batch_id);
        if (ImGui::InputInt("Batch id", &value)) {
            rays.batch_id = static_cast<std::uint32_t>(std::max(0, value));
            filter_dirty = true;
            sample_filter_dirty = true;
        }
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    ImGui::TextUnformatted("Picking");
    ImGui::Checkbox("Enable picking", &pick.enable);
    ImGui::Checkbox("Visible only", &pick.visible_only);
    ImGui::Checkbox("Auto pin", &pick.auto_pin);
    ImGui::SliderFloat("Pick radius", &pick.radius, 0.01f, 5.0f);
    {
        int stride = static_cast<int>(pick.stride);
        if (ImGui::SliderInt("Pick stride", &stride, 1, 256)) {
            pick.stride = static_cast<std::uint32_t>(stride);
        }
    }
    if (last_picked_ray_index >= 0) {
        ImGui::Text("Last pick: %d (dist %.3f)", last_picked_ray_index, static_cast<double>(last_pick_distance));
    } else {
        ImGui::TextUnformatted("Last pick: none");
    }
    if (ImGui::Button("Pin selected ray")) {
        pinned_ray_index = v2_ray_index;
        pinned_sample_index = 0;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear pin")) {
        pinned_ray_index = -1;
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Sample Visualization (v2)");
    ImGui::BeginDisabled(!record_v2_reader.is_open());
    if (ImGui::Checkbox("Show samples", &samples.show_samples)) {
        sample_filter_dirty = true;
    }
    ImGui::SliderFloat("Point size", &samples.point_size, 0.5f, 10.0f);
    ImGui::SliderFloat("Alpha", &samples.alpha, 0.0f, 1.0f);
    if (ImGui::Checkbox("Isolate selected ray", &samples.isolate_ray)) {
        sample_filter_dirty = true;
    }
    if (samples.isolate_ray) {
        ImGui::SameLine();
        ImGui::Text("Ray %d", v2_ray_index);
    }
    {
        int stride = static_cast<int>(samples.stride);
        if (ImGui::SliderInt("Sample stride", &stride, 1, 64)) {
            samples.stride = static_cast<std::uint32_t>(stride);
            sample_filter_dirty = true;
        }
    }
    {
        int max_samples = static_cast<int>(samples.max_samples);
        if (ImGui::SliderInt("Max samples", &max_samples, 1, 5000000)) {
            samples.max_samples = static_cast<std::uint32_t>(max_samples);
            request_sample_resize = true;
            sample_filter_dirty = true;
        }
    }
    {
        const char* modes[] = {
            "State",
            "Omit reason",
            "Density",
            "Weight",
            "Contribution",
        };
        int mode = static_cast<int>(samples.color_mode);
        if (ImGui::Combo("Sample color", &mode, modes, static_cast<int>(std::size(modes)))) {
            samples.color_mode = static_cast<pngp::vis::rays::SampleColorMode>(mode);
        }
        if (samples.color_mode == pngp::vis::rays::SampleColorMode::Density) {
            float min_v = samples.density_min;
            float max_v = samples.density_max;
            if (ImGui::DragFloatRange2("Density range", &min_v, &max_v, 0.01f, 0.0f, 1000.0f,
                                       "Min %.3f", "Max %.3f")) {
                if (min_v > max_v) std::swap(min_v, max_v);
                samples.density_min = min_v;
                samples.density_max = max_v;
            }
            draw_heatmap_legend("Density heatmap", samples.density_min, samples.density_max);
        }
        if (samples.color_mode == pngp::vis::rays::SampleColorMode::Weight) {
            float min_v = samples.weight_min;
            float max_v = samples.weight_max;
            if (ImGui::DragFloatRange2("Weight range", &min_v, &max_v, 0.001f, 0.0f, 100.0f,
                                       "Min %.3f", "Max %.3f")) {
                if (min_v > max_v) std::swap(min_v, max_v);
                samples.weight_min = min_v;
                samples.weight_max = max_v;
            }
            draw_heatmap_legend("Weight heatmap", samples.weight_min, samples.weight_max);
        }
        if (samples.color_mode == pngp::vis::rays::SampleColorMode::Contrib) {
            float min_v = samples.contrib_min;
            float max_v = samples.contrib_max;
            if (ImGui::DragFloatRange2("Contrib range", &min_v, &max_v, 0.001f, 0.0f, 100.0f,
                                       "Min %.3f", "Max %.3f")) {
                if (min_v > max_v) std::swap(min_v, max_v);
                samples.contrib_min = min_v;
                samples.contrib_max = max_v;
            }
            draw_heatmap_legend("Contrib heatmap", samples.contrib_min, samples.contrib_max);
        }
    }
    if (ImGui::Checkbox("Depth fade", &samples.depth_fade)) {
        // No compaction; shader reads the new values directly.
    }
    if (samples.depth_fade) {
        ImGui::SliderFloat("Fade near", &samples.depth_fade_near, 0.0f, 500.0f);
        ImGui::SliderFloat("Fade far", &samples.depth_fade_far, 0.1f, 2000.0f);
        ImGui::SliderFloat("Fade power", &samples.depth_fade_power, 0.1f, 4.0f);
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Pinned Ray");
    const bool pinned_valid = record_v2_reader.is_open() && pinned_ray_index >= 0 &&
                              pinned_ray_index < static_cast<int>(v2_view.rays.size());
    if (!pinned_valid) {
        ImGui::TextUnformatted("None");
    } else {
        const auto& ray = v2_view.rays[static_cast<std::size_t>(pinned_ray_index)];
        ImGui::Text("Index: %d", pinned_ray_index);
        ImGui::Text("Origin: %.3f %.3f %.3f", ray.ox, ray.oy, ray.oz);
        ImGui::Text("Dir: %.3f %.3f %.3f", ray.dx, ray.dy, ray.dz);
        ImGui::Text("Pixel: %u %u", ray.pixel_x, ray.pixel_y);
        ImGui::Text("Samples: %u", ray.sample_count);
        if (ray.result_index < v2_view.results.size()) {
            const auto& res = v2_view.results[ray.result_index];
            ImGui::Text("Result rgb: %.3f %.3f %.3f", res.r, res.g, res.b);
            ImGui::Text("Alpha: %.3f, Depth: %.3f", res.alpha, res.depth);
            ImGui::Text("Term: %u, Steps: %u", res.termination_reason, res.step_count);
        }

        const std::size_t sample_base = static_cast<std::size_t>(ray.sample_offset);
        const std::size_t sample_count = static_cast<std::size_t>(ray.sample_count);
        const bool samples_in_range =
            sample_count > 0 && sample_base + sample_count <= v2_view.samples.size();
        if (samples_in_range) {
            const int max_sample = static_cast<int>(sample_count - 1);
            pinned_sample_index = std::clamp(pinned_sample_index, 0, max_sample);
            ImGui::SliderInt("Pinned sample", &pinned_sample_index, 0, max_sample);
            const auto& sample = v2_view.samples[sample_base + static_cast<std::size_t>(pinned_sample_index)];
            ImGui::Text("Sample t: %.3f, dt: %.3f", sample.t, sample.dt);
            ImGui::Text("State: %u, Omit: %u", sample.state, sample.omit_reason);

            std::array<float, 4> state_counts{};
            std::array<float, 8> omit_counts{};
            const std::size_t plot_limit =
                std::min<std::size_t>(sample_count, static_cast<std::size_t>(std::max(1, pinned_plot_limit)));
            std::vector<float> density_values;
            std::vector<float> weight_values;
            std::vector<float> contrib_values;
            density_values.reserve(plot_limit);
            weight_values.reserve(plot_limit);
            contrib_values.reserve(plot_limit);

            const bool evals_ok = sample_base + sample_count <= v2_view.evals.size();
            for (std::size_t i = 0; i < sample_count; ++i) {
                const auto& s = v2_view.samples[sample_base + i];
                const std::size_t state_idx = static_cast<std::size_t>(std::min<std::uint8_t>(s.state, 3));
                const std::size_t omit_idx = static_cast<std::size_t>(std::min<std::uint8_t>(s.omit_reason, 7));
                state_counts[state_idx] += 1.0f;
                omit_counts[omit_idx] += 1.0f;
                if (evals_ok && i < plot_limit) {
                    const auto& e = v2_view.evals[sample_base + i];
                    density_values.push_back(e.density);
                    weight_values.push_back(e.weight);
                    const float contrib = std::sqrt(e.contrib_r * e.contrib_r +
                                                    e.contrib_g * e.contrib_g +
                                                    e.contrib_b * e.contrib_b);
                    contrib_values.push_back(contrib);
                }
            }

            const float state_max = *std::max_element(state_counts.begin(), state_counts.end());
            const float omit_max = *std::max_element(omit_counts.begin(), omit_counts.end());
            ImGui::TextUnformatted("State histogram");
            ImGui::PlotHistogram("##state_hist", state_counts.data(), static_cast<int>(state_counts.size()), 0,
                                 nullptr, 0.0f, state_max + 1.0f, ImVec2(0, 60));
            ImGui::TextUnformatted("Omit histogram");
            ImGui::PlotHistogram("##omit_hist", omit_counts.data(), static_cast<int>(omit_counts.size()), 0,
                                 nullptr, 0.0f, omit_max + 1.0f, ImVec2(0, 60));

            ImGui::SliderInt("Table rows", &pinned_table_limit, 1, 2048);
            ImGui::SliderInt("Plot samples", &pinned_plot_limit, 1, 4096);
            if (!density_values.empty()) {
                const float max_density = *std::max_element(density_values.begin(), density_values.end());
                ImGui::PlotHistogram("Density", density_values.data(), static_cast<int>(density_values.size()), 0,
                                     nullptr, 0.0f, max_density * 1.05f, ImVec2(0, 60));
            }
            if (!weight_values.empty()) {
                const float max_weight = *std::max_element(weight_values.begin(), weight_values.end());
                ImGui::PlotHistogram("Weight", weight_values.data(), static_cast<int>(weight_values.size()), 0,
                                     nullptr, 0.0f, max_weight * 1.05f, ImVec2(0, 60));
            }
            if (!contrib_values.empty()) {
                const float max_contrib = *std::max_element(contrib_values.begin(), contrib_values.end());
                ImGui::PlotHistogram("Contrib", contrib_values.data(), static_cast<int>(contrib_values.size()), 0,
                                     nullptr, 0.0f, max_contrib * 1.05f, ImVec2(0, 60));
            }

            const int row_count = std::min<int>(max_sample + 1, std::max(1, pinned_table_limit));
            const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY;
            if (ImGui::BeginTable("PinnedSampleTable", 8, flags, ImVec2(0.0f, 240.0f))) {
                ImGui::TableSetupColumn("i");
                ImGui::TableSetupColumn("t");
                ImGui::TableSetupColumn("dt");
                ImGui::TableSetupColumn("state");
                ImGui::TableSetupColumn("omit");
                ImGui::TableSetupColumn("density");
                ImGui::TableSetupColumn("weight");
                ImGui::TableSetupColumn("contrib");
                ImGui::TableHeadersRow();
                for (int i = 0; i < row_count; ++i) {
                    const auto& s = v2_view.samples[sample_base + static_cast<std::size_t>(i)];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", i);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", s.t);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", s.dt);
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", s.state);
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", s.omit_reason);
                    if (evals_ok) {
                        const auto& e = v2_view.evals[sample_base + static_cast<std::size_t>(i)];
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", e.density);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", e.weight);
                        ImGui::TableNextColumn();
                        const float contrib = std::sqrt(e.contrib_r * e.contrib_r +
                                                        e.contrib_g * e.contrib_g +
                                                        e.contrib_b * e.contrib_b);
                        ImGui::Text("%.3f", contrib);
                    } else {
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted("-");
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted("-");
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted("-");
                    }
                }
                ImGui::EndTable();
            }
        } else {
            ImGui::TextUnformatted("No samples available for this ray.");
        }
    }

    ImGui::EndDisabled();

    ImGui::End();
    return rebuild;
}
