#include "ray_generator.h"

// ============================================================================
// Standalone CLI for generating NeRF-style ray records.
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
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

        struct Options {
            std::string output_path = "ray_record.bin";
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
        };

        void print_usage() {
            std::cout
                << "ray_generator options:\n"
                << "  --output <path>       Output record path (default: ray_record.bin)\n"
                << "  --width <int>         Image width (default: 800)\n"
                << "  --height <int>        Image height (default: 600)\n"
                << "  --frames <int>        Frame count (default: 1)\n"
                << "  --fov-y <deg>         Vertical FOV in degrees (default: 60)\n"
                << "  --fx <float>          Override fx (default: derived from fov)\n"
                << "  --fy <float>          Override fy (default: derived from fov)\n"
                << "  --cx <float>          Override cx (default: width * 0.5)\n"
                << "  --cy <float>          Override cy (default: height * 0.5)\n"
                << "  --radius <float>      Orbit radius (default: 4)\n"
                << "  --elevation <float>   Orbit height (default: 1)\n"
                << "  --fps <float>         Timestamp FPS (default: 30)\n"
                << "  --help                Show this help\n";
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
                } else {
                    std::cerr << "Unknown argument: " << arg << "\n";
                    print_usage();
                    return false;
                }
            }
            return true;
        }

        void write_record(const std::string& path,
                          const std::vector<record::FrameHeader>& frames,
                          const std::vector<record::RayRecord>& rays) {
            const std::uint64_t header_bytes = sizeof(record::RecordHeader);
            const std::uint64_t frame_table_offset = align_up(header_bytes, 16);
            const std::uint64_t frame_table_bytes =
                static_cast<std::uint64_t>(frames.size()) * sizeof(record::FrameHeader);
            const std::uint64_t ray_data_offset = align_up(frame_table_offset + frame_table_bytes, 16);

            record::RecordHeader header{};
            header.magic = record::k_magic;
            header.version_major = record::k_version_major;
            header.version_minor = record::k_version_minor;
            header.endian = record::Endian::Little;
            header.compression = record::Compression::None;
            header.header_bytes = static_cast<std::uint16_t>(header_bytes);
            header.frame_count = static_cast<std::uint64_t>(frames.size());
            header.frame_table_offset = frame_table_offset;
            header.ray_data_offset = ray_data_offset;
            header.sample_data_offset = 0;

            std::ofstream out(path, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open output file");

            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            write_padding(out, frame_table_offset - header_bytes);

            if (!frames.empty()) {
                out.write(reinterpret_cast<const char*>(frames.data()),
                          static_cast<std::streamsize>(frame_table_bytes));
            }
            write_padding(out, ray_data_offset - (frame_table_offset + frame_table_bytes));

            if (!rays.empty()) {
                out.write(reinterpret_cast<const char*>(rays.data()),
                          static_cast<std::streamsize>(rays.size() * sizeof(record::RayRecord)));
            }
        }
    } // namespace detail
} // namespace raygen

int main(int argc, char** argv) {
    raygen::detail::Options opt{};
    if (!raygen::detail::parse_args(argc, argv, opt)) return 0;

    if (opt.width == 0 || opt.height == 0 || opt.frames == 0) {
        std::cerr << "Width/height/frames must be > 0.\n";
        return 1;
    }

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
    const std::uint64_t total_rays =
        rays_per_frame * static_cast<std::uint64_t>(opt.frames);

    std::vector<raygen::record::FrameHeader> frames;
    frames.resize(opt.frames);

    std::vector<raygen::record::RayRecord> all_rays;
    if (total_rays > 0) {
        const std::size_t reserve_count =
            static_cast<std::size_t>(std::min<std::uint64_t>(total_rays, std::numeric_limits<std::size_t>::max()));
        all_rays.reserve(reserve_count);
    }

    for (std::uint32_t i = 0; i < opt.frames; ++i) {
        const float t = (opt.frames > 1) ? (static_cast<float>(i) / opt.frames) : 0.0f;
        const float angle = 2.0f * raygen::k_pi * t;

        const raygen::detail::Vec3 eye{std::cos(angle) * opt.radius, opt.elevation, std::sin(angle) * opt.radius};
        const raygen::detail::Vec3 target{0.0f, 0.0f, 0.0f};
        const raygen::detail::Vec3 up{0.0f, 1.0f, 0.0f};

        const raygen::CameraTransform c2w = raygen::detail::make_look_at(eye, target, up);

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

        raygen::record::FrameHeader header{};
        header.frame_index = i;
        header.timestamp_sec = opt.fps > 0.0f ? static_cast<double>(i) / opt.fps : 0.0;
        header.width = opt.width;
        header.height = opt.height;
        header.fx = opt.fx;
        header.fy = opt.fy;
        header.cx = opt.cx;
        header.cy = opt.cy;
        std::copy(std::begin(c2w.c2w), std::end(c2w.c2w), header.c2w_3x4.begin());
        header.ray_count = static_cast<std::uint64_t>(rays_frame.size());
        header.ray_offset = static_cast<std::uint64_t>(all_rays.size() * sizeof(raygen::record::RayRecord));
        header.sample_count = 0;
        header.sample_offset = 0;

        frames[i] = header;
        all_rays.insert(all_rays.end(), rays_frame.begin(), rays_frame.end());
    }

    try {
        raygen::detail::write_record(opt.output_path, frames, all_rays);
        std::cout << "Wrote record: " << opt.output_path << "\n";
        std::cout << "Frames: " << frames.size() << ", Rays: " << all_rays.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to write record: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
