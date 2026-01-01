#pragma once

// ============================================================================
// CUDA ray generator interface.
// ============================================================================
#include <cstdint>
#include <vector>

#include "record_schema.h"

namespace raygen {
    struct CameraTransform {
        float c2w[12]{};
    };

    struct RayGenConfig {
        std::uint32_t width{};
        std::uint32_t height{};
        float fx{};
        float fy{};
        float cx{};
        float cy{};
        CameraTransform c2w{};
    };

    void generate_rays_cuda(std::vector<record::RayRecord>& out, const RayGenConfig& cfg);
} // namespace raygen
