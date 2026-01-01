#pragma once

// ============================================================================
// Record schema for NeRF ray generation (matches pngp.vis.rays.record.ixx).
// ============================================================================
#include <array>
#include <cstdint>
#include <type_traits>

namespace raygen::record {
    inline constexpr std::array<char, 4> k_magic{{'R', 'F', 'R', 'Y'}};
    inline constexpr std::uint16_t k_version_major = 1;
    inline constexpr std::uint16_t k_version_minor = 0;

    enum class Endian : std::uint8_t {
        Little = 1,
        Big    = 2,
    };

    enum class Compression : std::uint8_t {
        None = 0,
        Zstd = 1,
    };

    struct alignas(16) RecordHeader {
        std::array<char, 4> magic{};
        std::uint16_t version_major{};
        std::uint16_t version_minor{};
        Endian endian{};
        Compression compression{};
        std::uint16_t header_bytes{};
        std::uint32_t reserved0{};
        std::uint64_t frame_count{};
        std::uint64_t frame_table_offset{};
        std::uint64_t ray_data_offset{};
        std::uint64_t sample_data_offset{};
        std::array<std::uint64_t, 2> reserved1{};
    };

    struct alignas(16) FrameHeader {
        std::uint64_t frame_index{};
        double timestamp_sec{};
        std::uint32_t width{};
        std::uint32_t height{};
        float fx{};
        float fy{};
        float cx{};
        float cy{};
        std::array<float, 12> c2w_3x4{};
        std::uint64_t ray_count{};
        std::uint64_t ray_offset{};
        std::uint64_t sample_count{};
        std::uint64_t sample_offset{};
        std::array<std::uint64_t, 2> reserved{};
    };

    struct alignas(16) RayRecord {
        float ox{};
        float oy{};
        float oz{};
        float pad0{};
        float dx{};
        float dy{};
        float dz{};
        float pad1{};
        std::uint32_t pixel_x{};
        std::uint32_t pixel_y{};
        std::uint32_t flags{};
        std::uint32_t pad2{};
    };

    struct alignas(16) SampleRecord {
        float t{};
        float density{};
        float r{};
        float g{};
        float b{};
        float weight{};
        std::uint32_t ray_index{};
        std::array<std::uint32_t, 3> pad{};
    };

    static_assert(std::is_standard_layout<RecordHeader>::value);
    static_assert(std::is_standard_layout<FrameHeader>::value);
    static_assert(std::is_standard_layout<RayRecord>::value);
    static_assert(std::is_standard_layout<SampleRecord>::value);

    static_assert(sizeof(RecordHeader) == 64);
    static_assert(sizeof(FrameHeader) == 144);
    static_assert(sizeof(RayRecord) == 48);
    static_assert(sizeof(SampleRecord) == 48);
} // namespace raygen::record
