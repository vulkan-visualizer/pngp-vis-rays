#include "ray_generator.h"
#include "record_schema_v2.h"

// ============================================================================
// v2 record generator (ray-centric debug schema).
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace raygen {
    inline constexpr float k_pi = 3.14159265358979323846f;

    namespace detail {
        struct Vec3 {
            float x{};
            float y{};
            float z{};
        };

        Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
        Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
        Vec3 operator*(const Vec3& v, const float s) { return {v.x * s, v.y * s, v.z * s}; }

        float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

        Vec3 cross(const Vec3& a, const Vec3& b) {
            return {a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x};
        }

        Vec3 normalize(const Vec3& v) {
            const float len = std::sqrt(dot(v, v));
            if (len <= 1e-8f) return {};
            return v * (1.0f / len);
        }

        CameraTransform make_look_at(const Vec3& eye, const Vec3& target, const Vec3& up) {
            const Vec3 f = normalize(target - eye);
            const Vec3 r = normalize(cross(f, up));
            const Vec3 u = cross(r, f);

            CameraTransform out{};
            out.c2w[0]  = r.x;
            out.c2w[1]  = u.x;
            out.c2w[2]  = f.x;
            out.c2w[3]  = eye.x;
            out.c2w[4]  = r.y;
            out.c2w[5]  = u.y;
            out.c2w[6]  = f.y;
            out.c2w[7]  = eye.y;
            out.c2w[8]  = r.z;
            out.c2w[9]  = u.z;
            out.c2w[10] = f.z;
            out.c2w[11] = eye.z;
            return out;
        }

        struct Options {
            std::string output_path = "ray_record_v2.bin";
            std::uint32_t width  = 800;
            std::uint32_t height = 600;
            std::uint32_t frames = 1;
            float fov_y_deg = 60.0f;
            float fx = 0.0f;
            float fy = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
            float radius = 4.0f;
            float elevation = 1.0f;
            float fps = 30.0f;
            std::uint32_t samples_per_ray = 8;
            float omit_rate = 0.2f;
            float max_depth = 6.0f;
        };

        void print_usage() {
            std::cout
                << "ray_generator_v2 options:\n"
                << "  --output <path>          Output record path (default: ray_record_v2.bin)\n"
                << "  --width <int>            Image width (default: 800)\n"
                << "  --height <int>           Image height (default: 600)\n"
                << "  --frames <int>           Frame count (default: 1)\n"
                << "  --fov-y <deg>            Vertical FOV in degrees (default: 60)\n"
                << "  --fx <float>             Override fx (default: derived from fov)\n"
                << "  --fy <float>             Override fy (default: derived from fov)\n"
                << "  --cx <float>             Override cx (default: width * 0.5)\n"
                << "  --cy <float>             Override cy (default: height * 0.5)\n"
                << "  --radius <float>         Orbit radius (default: 4)\n"
                << "  --elevation <float>      Orbit height (default: 1)\n"
                << "  --fps <float>            Timestamp FPS (default: 30)\n"
                << "  --samples-per-ray <int>  Samples per ray (default: 8)\n"
                << "  --omit-rate <float>      Omit rate in [0,1] (default: 0.2)\n"
                << "  --max-depth <float>      Max ray depth (default: 6)\n"
                << "  --help                   Show this help\n";
        }

        bool parse_args(const int argc, char** argv, Options& out) {
            for (int i = 1; i < argc; ++i) {
                const std::string_view arg = argv[i];
                auto next = [&](std::string_view flag) -> std::string_view {
                    if (i + 1 >= argc) {
                        std::cerr << "Missing value for " << flag << "\n";
                        std::exit(1);
                    }
                    return argv[++i];
                };

                if (arg == "--help" || arg == "-h") {
                    print_usage();
                    return false;
                }
                if (arg == "--output") {
                    out.output_path = std::string(next(arg));
                } else if (arg == "--width") {
                    out.width = static_cast<std::uint32_t>(std::stoul(std::string(next(arg))));
                } else if (arg == "--height") {
                    out.height = static_cast<std::uint32_t>(std::stoul(std::string(next(arg))));
                } else if (arg == "--frames") {
                    out.frames = static_cast<std::uint32_t>(std::stoul(std::string(next(arg))));
                } else if (arg == "--fov-y") {
                    out.fov_y_deg = std::stof(std::string(next(arg)));
                } else if (arg == "--fx") {
                    out.fx = std::stof(std::string(next(arg)));
                } else if (arg == "--fy") {
                    out.fy = std::stof(std::string(next(arg)));
                } else if (arg == "--cx") {
                    out.cx = std::stof(std::string(next(arg)));
                } else if (arg == "--cy") {
                    out.cy = std::stof(std::string(next(arg)));
                } else if (arg == "--radius") {
                    out.radius = std::stof(std::string(next(arg)));
                } else if (arg == "--elevation") {
                    out.elevation = std::stof(std::string(next(arg)));
                } else if (arg == "--fps") {
                    out.fps = std::stof(std::string(next(arg)));
                } else if (arg == "--samples-per-ray") {
                    out.samples_per_ray = static_cast<std::uint32_t>(std::stoul(std::string(next(arg))));
                } else if (arg == "--omit-rate") {
                    out.omit_rate = std::stof(std::string(next(arg)));
                } else if (arg == "--max-depth") {
                    out.max_depth = std::stof(std::string(next(arg)));
                } else {
                    std::cerr << "Unknown argument: " << arg << "\n";
                    print_usage();
                    return false;
                }
            }
            return true;
        }

        std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
            const std::uint64_t mask = alignment - 1u;
            return (value + mask) & ~mask;
        }

        void write_padding(std::ofstream& out, const std::uint64_t bytes) {
            if (bytes == 0) return;
            std::array<char, 256> zeros{};
            std::uint64_t remaining = bytes;
            while (remaining > 0) {
                const std::uint64_t chunk = std::min<std::uint64_t>(remaining, zeros.size());
                out.write(zeros.data(), static_cast<std::streamsize>(chunk));
                remaining -= chunk;
            }
        }

        float rand01(std::uint32_t& rng) {
            rng = 1664525u * rng + 1013904223u;
            return static_cast<float>((rng >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
        }

        struct FrameSections {
            std::array<record_v2::SectionTableEntryV2, 4> entries{};
        };

        void fill_section(record_v2::SectionTableEntryV2& entry,
                          const record_v2::SectionType type,
                          const std::uint64_t offset,
                          const std::uint64_t count,
                          const std::uint32_t stride) {
            entry.type         = static_cast<std::uint32_t>(type);
            entry.flags        = 0;
            entry.alignment    = 16;
            entry.reserved0    = 0;
            entry.offset       = offset;
            entry.size_bytes   = count * stride;
            entry.count        = count;
            entry.stride_bytes = stride;
            entry.name_offset  = 0;
            entry.reserved1    = {0, 0};
        }
    } // namespace detail
} // namespace raygen

int main(int argc, char** argv) {
    raygen::detail::Options opt{};
    if (!raygen::detail::parse_args(argc, argv, opt)) return 0;

    if (opt.width == 0 || opt.height == 0 || opt.frames == 0 || opt.samples_per_ray == 0) {
        std::cerr << "Width/height/frames/samples-per-ray must be > 0.\n";
        return 1;
    }

    opt.omit_rate = std::clamp(opt.omit_rate, 0.0f, 1.0f);
    if (opt.fx <= 0.0f && opt.fy <= 0.0f) {
        const float fov_rad = opt.fov_y_deg * raygen::k_pi / 180.0f;
        opt.fy = 0.5f * static_cast<float>(opt.height) / std::tan(fov_rad * 0.5f);
        opt.fx = opt.fy;
    } else if (opt.fx <= 0.0f) {
        opt.fx = opt.fy;
    } else if (opt.fy <= 0.0f) {
        opt.fy = opt.fx;
    }

    if (opt.cx <= 0.0f) opt.cx = static_cast<float>(opt.width) * 0.5f;
    if (opt.cy <= 0.0f) opt.cy = static_cast<float>(opt.height) * 0.5f;

    const std::uint64_t rays_per_frame =
        static_cast<std::uint64_t>(opt.width) * opt.height;
    const std::uint64_t samples_per_frame =
        rays_per_frame * opt.samples_per_ray;

    // =========================================================================
    // Build frame index + section table.
    // =========================================================================
    std::vector<raygen::record_v2::FrameIndexEntryV2> frames(opt.frames);
    const std::uint64_t header_bytes = sizeof(raygen::record_v2::RecordHeaderV2);
    const std::uint64_t frame_index_offset = raygen::detail::align_up(header_bytes, 16);
    const std::uint64_t frame_index_bytes =
        static_cast<std::uint64_t>(frames.size()) * sizeof(raygen::record_v2::FrameIndexEntryV2);
    const std::uint64_t section_table_offset =
        raygen::detail::align_up(frame_index_offset + frame_index_bytes, 16);
    constexpr std::uint32_t sections_per_frame = 4;
    const std::uint64_t section_table_bytes =
        static_cast<std::uint64_t>(frames.size()) * sections_per_frame * sizeof(raygen::record_v2::SectionTableEntryV2);
    const std::uint64_t string_table_offset =
        raygen::detail::align_up(section_table_offset + section_table_bytes, 16);
    const std::uint64_t string_table_bytes = 0;
    std::uint64_t payload_offset =
        raygen::detail::align_up(string_table_offset + string_table_bytes, 16);

    std::vector<raygen::record_v2::SectionTableEntryV2> sections;
    sections.resize(frames.size() * sections_per_frame);
    std::vector<raygen::CameraTransform> c2w_frames(frames.size());

    std::uint64_t current_payload = payload_offset;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        auto& frame = frames[i];
        const float t = (opt.frames > 1) ? (static_cast<float>(i) / opt.frames) : 0.0f;
        const float angle = 2.0f * raygen::k_pi * t;
        const raygen::detail::Vec3 eye{std::cos(angle) * opt.radius, opt.elevation, std::sin(angle) * opt.radius};
        const raygen::detail::Vec3 target{0.0f, 0.0f, 0.0f};
        const raygen::detail::Vec3 up{0.0f, 1.0f, 0.0f};
        c2w_frames[i] = raygen::detail::make_look_at(eye, target, up);

        frame.frame_index   = static_cast<std::uint64_t>(i);
        frame.timestamp_sec = opt.fps > 0.0f ? static_cast<double>(i) / opt.fps : 0.0;
        frame.width         = opt.width;
        frame.height        = opt.height;
        frame.fx            = opt.fx;
        frame.fy            = opt.fy;
        frame.cx            = opt.cx;
        frame.cy            = opt.cy;
        std::copy(std::begin(c2w_frames[i].c2w), std::end(c2w_frames[i].c2w), frame.c2w_3x4.begin());
        frame.ray_count     = rays_per_frame;
        frame.sample_count  = samples_per_frame;
        frame.section_offset =
            section_table_offset + static_cast<std::uint64_t>(i * sections_per_frame * sizeof(raygen::record_v2::SectionTableEntryV2));
        frame.section_count = sections_per_frame;

        auto* entry = sections.data() + i * sections_per_frame;
        raygen::detail::fill_section(entry[0], raygen::record_v2::SectionType::RayBase,
                                     current_payload, rays_per_frame,
                                     sizeof(raygen::record_v2::RayBaseRecordV2));
        current_payload = raygen::detail::align_up(current_payload + entry[0].size_bytes, 16);

        raygen::detail::fill_section(entry[1], raygen::record_v2::SectionType::SampleRecord,
                                     current_payload, samples_per_frame,
                                     sizeof(raygen::record_v2::SampleRecordV2));
        current_payload = raygen::detail::align_up(current_payload + entry[1].size_bytes, 16);

        raygen::detail::fill_section(entry[2], raygen::record_v2::SectionType::SampleEval,
                                     current_payload, samples_per_frame,
                                     sizeof(raygen::record_v2::SampleEvalV2));
        current_payload = raygen::detail::align_up(current_payload + entry[2].size_bytes, 16);

        raygen::detail::fill_section(entry[3], raygen::record_v2::SectionType::RayResult,
                                     current_payload, rays_per_frame,
                                     sizeof(raygen::record_v2::RayResultV2));
        current_payload = raygen::detail::align_up(current_payload + entry[3].size_bytes, 16);
    }

    // =========================================================================
    // Write header + tables.
    // =========================================================================
    std::ofstream out(opt.output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    raygen::record_v2::RecordHeaderV2 header{};
    header.magic               = raygen::record_v2::k_magic;
    header.version_major       = raygen::record_v2::k_version_major;
    header.version_minor       = raygen::record_v2::k_version_minor;
    header.endian              = raygen::record_v2::Endian::Little;
    header.compression         = raygen::record_v2::Compression::None;
    header.header_bytes        = static_cast<std::uint16_t>(header_bytes);
    header.flags               = 0;
    header.schema_hash         = {0, 0};
    header.frame_count         = static_cast<std::uint64_t>(frames.size());
    header.frame_index_offset  = frame_index_offset;
    header.section_table_offset = section_table_offset;
    header.section_table_bytes = section_table_bytes;
    header.string_table_offset = string_table_offset;
    header.string_table_bytes  = string_table_bytes;

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    raygen::detail::write_padding(out, frame_index_offset - header_bytes);
    out.write(reinterpret_cast<const char*>(frames.data()),
              static_cast<std::streamsize>(frame_index_bytes));
    raygen::detail::write_padding(out, section_table_offset - (frame_index_offset + frame_index_bytes));
    out.write(reinterpret_cast<const char*>(sections.data()),
              static_cast<std::streamsize>(section_table_bytes));
    raygen::detail::write_padding(out, string_table_offset - (section_table_offset + section_table_bytes));
    raygen::detail::write_padding(out, payload_offset - (string_table_offset + string_table_bytes));

    // =========================================================================
    // Generate per-frame payloads and write sequentially.
    // =========================================================================
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const raygen::CameraTransform c2w = c2w_frames[i];

        raygen::RayGenConfig cfg{};
        cfg.width  = opt.width;
        cfg.height = opt.height;
        cfg.fx = opt.fx;
        cfg.fy = opt.fy;
        cfg.cx = opt.cx;
        cfg.cy = opt.cy;
        cfg.c2w = c2w;

        std::vector<raygen::record::RayRecord> rays_frame;
        raygen::generate_rays_cuda(rays_frame, cfg);

        std::vector<raygen::record_v2::RayBaseRecordV2> rays;
        std::vector<raygen::record_v2::SampleRecordV2> samples;
        std::vector<raygen::record_v2::SampleEvalV2> evals;
        std::vector<raygen::record_v2::RayResultV2> results;

        const std::size_t ray_count = rays_frame.size();
        const std::size_t sample_count = ray_count * opt.samples_per_ray;
        rays.resize(ray_count);
        samples.resize(sample_count);
        evals.resize(sample_count);
        results.resize(ray_count);

        std::uint32_t rng = 2166136261u ^ static_cast<std::uint32_t>(i * 16777619u);
        const float dt = opt.max_depth / static_cast<float>(opt.samples_per_ray);

        for (std::size_t r = 0; r < ray_count; ++r) {
            const auto& src = rays_frame[r];
            auto& dst = rays[r];
            dst.ox = src.ox;
            dst.oy = src.oy;
            dst.oz = src.oz;
            dst.dx = src.dx;
            dst.dy = src.dy;
            dst.dz = src.dz;
            dst.pixel_x = src.pixel_x;
            dst.pixel_y = src.pixel_y;
            dst.ray_flags = 1;
            dst.sample_offset = static_cast<std::uint32_t>(r * opt.samples_per_ray);
            dst.sample_count  = opt.samples_per_ray;
            dst.result_index  = static_cast<std::uint32_t>(r);

            float trans = 1.0f;
            float accum_r = 0.0f;
            float accum_g = 0.0f;
            float accum_b = 0.0f;
            float depth_accum = 0.0f;
            float weight_sum = 0.0f;

            for (std::uint32_t s = 0; s < opt.samples_per_ray; ++s) {
                const std::size_t idx = r * opt.samples_per_ray + s;
                const float t_sample = (static_cast<float>(s) + 0.5f) * dt;
                const float dice = raygen::detail::rand01(rng);

                const bool omit = dice < opt.omit_rate;
                const auto state = omit ? raygen::record_v2::SampleState::Omitted
                                        : raygen::record_v2::SampleState::Kept;

                auto& sample = samples[idx];
                sample.t = t_sample;
                sample.dt = dt;
                sample.level = 0;
                sample.mip = 0;
                sample.state = static_cast<std::uint8_t>(state);
                sample.omit_reason = omit ? static_cast<std::uint8_t>(raygen::record_v2::OmitReason::Occupancy)
                                          : static_cast<std::uint8_t>(raygen::record_v2::OmitReason::None);
                sample.ray_index = static_cast<std::uint32_t>(r);
                sample.sample_flags = 0;
                sample.rng_seed = rng;

                auto& eval = evals[idx];
                const float trans_before = trans;
                float weight = 0.0f;
                float alpha = 0.0f;

                if (!omit) {
                    alpha = 0.05f + 0.35f * raygen::detail::rand01(rng);
                    weight = alpha * trans;
                    trans *= (1.0f - alpha);
                }

                const float cr = raygen::detail::rand01(rng);
                const float cg = raygen::detail::rand01(rng);
                const float cb = raygen::detail::rand01(rng);

                eval.density = alpha * 10.0f;
                eval.r = cr;
                eval.g = cg;
                eval.b = cb;
                eval.weight = weight;
                eval.transmittance = trans_before;
                eval.contrib_r = cr * weight;
                eval.contrib_g = cg * weight;
                eval.contrib_b = cb * weight;

                accum_r += eval.contrib_r;
                accum_g += eval.contrib_g;
                accum_b += eval.contrib_b;
                depth_accum += t_sample * weight;
                weight_sum += weight;
            }

            auto& result = results[r];
            result.r = accum_r;
            result.g = accum_g;
            result.b = accum_b;
            result.alpha = 1.0f - trans;
            result.depth = weight_sum > 0.0f ? depth_accum / weight_sum : 0.0f;
            result.termination_reason =
                trans < 0.01f ? static_cast<std::uint32_t>(raygen::record_v2::TerminationReason::AlphaConverged)
                             : static_cast<std::uint32_t>(raygen::record_v2::TerminationReason::MaxSteps);
            result.step_count = opt.samples_per_ray;
        }

        const auto* entry = sections.data() + i * sections_per_frame;
        out.seekp(static_cast<std::streamoff>(entry[0].offset), std::ios::beg);
        out.write(reinterpret_cast<const char*>(rays.data()),
                  static_cast<std::streamsize>(entry[0].size_bytes));
        raygen::detail::write_padding(out, raygen::detail::align_up(entry[0].offset + entry[0].size_bytes, 16) -
                                               (entry[0].offset + entry[0].size_bytes));

        out.seekp(static_cast<std::streamoff>(entry[1].offset), std::ios::beg);
        out.write(reinterpret_cast<const char*>(samples.data()),
                  static_cast<std::streamsize>(entry[1].size_bytes));
        raygen::detail::write_padding(out, raygen::detail::align_up(entry[1].offset + entry[1].size_bytes, 16) -
                                               (entry[1].offset + entry[1].size_bytes));

        out.seekp(static_cast<std::streamoff>(entry[2].offset), std::ios::beg);
        out.write(reinterpret_cast<const char*>(evals.data()),
                  static_cast<std::streamsize>(entry[2].size_bytes));
        raygen::detail::write_padding(out, raygen::detail::align_up(entry[2].offset + entry[2].size_bytes, 16) -
                                               (entry[2].offset + entry[2].size_bytes));

        out.seekp(static_cast<std::streamoff>(entry[3].offset), std::ios::beg);
        out.write(reinterpret_cast<const char*>(results.data()),
                  static_cast<std::streamsize>(entry[3].size_bytes));
        raygen::detail::write_padding(out, raygen::detail::align_up(entry[3].offset + entry[3].size_bytes, 16) -
                                               (entry[3].offset + entry[3].size_bytes));

    }

    std::cout << "Wrote v2 record: " << opt.output_path << "\n";
    std::cout << "Frames: " << frames.size() << ", Rays/frame: " << rays_per_frame
              << ", Samples/ray: " << opt.samples_per_ray << "\n";
    return 0;
}
