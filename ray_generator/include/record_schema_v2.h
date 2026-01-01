#pragma once

// ============================================================================
// Record schema v2 (ray-centric debug format).
// ============================================================================
#include <array>
#include <cstdint>
#include <type_traits>

namespace raygen::record_v2 {
    inline constexpr std::array<char, 4> k_magic{{'R', 'F', 'R', 'Y'}};
    inline constexpr std::uint16_t k_version_major = 2;
    inline constexpr std::uint16_t k_version_minor = 0;

    enum class Endian : std::uint8_t {
        Little = 1,
        Big    = 2,
    };

    enum class Compression : std::uint8_t {
        None = 0,
        Zstd = 1,
    };

    enum class SectionType : std::uint32_t {
        RayBase         = 0,
        RayResult       = 1,
        SampleRecord    = 2,
        SampleEval      = 3,
        AttributeStream = 4,
    };

    enum class SampleState : std::uint8_t {
        Candidate  = 0,
        Kept       = 1,
        Omitted    = 2,
        Terminated = 3,
    };

    enum class OmitReason : std::uint8_t {
        None          = 0,
        Occupancy     = 1,
        Alpha         = 2,
        Bounds        = 3,
        StepLimit     = 4,
        DensityThresh = 5,
        UserMask      = 6,
        Other         = 7,
    };

    enum class TerminationReason : std::uint32_t {
        None           = 0,
        AlphaConverged = 1,
        MaxSteps       = 2,
        DepthClamp     = 3,
        EmptySpace     = 4,
        UserStop       = 5,
    };

    enum class AttributeTarget : std::uint32_t {
        Ray    = 0,
        Sample = 1,
        Result = 2,
    };

    enum class AttributeFormat : std::uint32_t {
        U8  = 0,
        U16 = 1,
        U32 = 2,
        F16 = 3,
        F32 = 4,
    };

    struct alignas(16) RecordHeaderV2 {
        std::array<char, 4> magic{};
        std::uint16_t version_major{};
        std::uint16_t version_minor{};
        Endian endian{};
        Compression compression{};
        std::uint16_t header_bytes{};
        std::uint32_t flags{};
        std::array<std::uint64_t, 2> schema_hash{};
        std::uint64_t frame_count{};
        std::uint64_t frame_index_offset{};
        std::uint64_t section_table_offset{};
        std::uint64_t section_table_bytes{};
        std::uint64_t string_table_offset{};
        std::uint64_t string_table_bytes{};
        std::array<std::uint64_t, 6> reserved{};
    };

    struct alignas(16) FrameIndexEntryV2 {
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
        std::uint64_t sample_count{};
        std::uint64_t section_offset{};
        std::uint32_t section_count{};
        std::uint32_t reserved0{};
        std::array<std::uint64_t, 3> reserved1{};
    };

    struct alignas(16) SectionTableEntryV2 {
        std::uint32_t type{};
        std::uint32_t flags{};
        std::uint32_t alignment{};
        std::uint32_t reserved0{};
        std::uint64_t offset{};
        std::uint64_t size_bytes{};
        std::uint64_t count{};
        std::uint32_t stride_bytes{};
        std::uint32_t name_offset{};
        std::array<std::uint64_t, 2> reserved1{};
    };

    struct alignas(16) RayBaseRecordV2 {
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
        std::uint32_t ray_flags{};
        std::uint32_t pad2{};
        std::uint32_t sample_offset{};
        std::uint32_t sample_count{};
        std::uint32_t result_index{};
        std::uint32_t pad3{};
    };

    struct alignas(16) SampleRecordV2 {
        float t{};
        float dt{};
        std::uint16_t level{};
        std::uint16_t mip{};
        std::uint8_t state{};
        std::uint8_t omit_reason{};
        std::uint16_t pad0{};
        std::uint32_t ray_index{};
        std::uint32_t sample_flags{};
        std::uint32_t rng_seed{};
        std::uint32_t pad1{};
    };

    struct alignas(16) SampleEvalV2 {
        float density{};
        float r{};
        float g{};
        float b{};
        float weight{};
        float transmittance{};
        float contrib_r{};
        float contrib_g{};
        float contrib_b{};
        float pad0{};
        std::array<float, 2> pad1{};
    };

    struct alignas(16) RayResultV2 {
        float r{};
        float g{};
        float b{};
        float alpha{};
        float depth{};
        std::uint32_t termination_reason{};
        std::uint32_t step_count{};
        std::uint32_t pad0{};
    };

    struct alignas(16) AttributeStreamDesc {
        std::uint32_t target{};
        std::uint32_t format{};
        std::uint32_t components{};
        std::uint32_t flags{};
        std::uint32_t name_offset{};
        std::uint32_t count{};
        std::uint32_t stride_bytes{};
        std::uint32_t reserved0{};
        std::uint64_t data_offset{};
        std::uint64_t data_bytes{};
        std::array<std::uint64_t, 2> reserved1{};
    };

    static_assert(std::is_standard_layout<RecordHeaderV2>::value);
    static_assert(std::is_standard_layout<FrameIndexEntryV2>::value);
    static_assert(std::is_standard_layout<SectionTableEntryV2>::value);
    static_assert(std::is_standard_layout<RayBaseRecordV2>::value);
    static_assert(std::is_standard_layout<SampleRecordV2>::value);
    static_assert(std::is_standard_layout<SampleEvalV2>::value);
    static_assert(std::is_standard_layout<RayResultV2>::value);
    static_assert(std::is_standard_layout<AttributeStreamDesc>::value);

    static_assert(sizeof(RecordHeaderV2) == 128);
    static_assert(sizeof(FrameIndexEntryV2) == 144);
    static_assert(sizeof(SectionTableEntryV2) == 64);
    static_assert(sizeof(RayBaseRecordV2) == 64);
    static_assert(sizeof(SampleRecordV2) == 32);
    static_assert(sizeof(SampleEvalV2) == 48);
    static_assert(sizeof(RayResultV2) == 32);
    static_assert(sizeof(AttributeStreamDesc) == 64);
} // namespace raygen::record_v2
