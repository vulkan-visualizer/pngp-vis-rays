#include "ray_generator.h"

// ============================================================================
// CUDA ray generation kernel + host wrapper.
// ============================================================================
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace raygen {
    namespace {
        struct DeviceBuffer {
            record::RayRecord* ptr{};
            ~DeviceBuffer() {
                if (ptr) cudaFree(ptr);
            }
        };

        void cuda_check(const cudaError_t err, const char* msg) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
            }
        }

        __device__ float3 normalize3(const float3 v) {
            const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
            if (len <= 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);
            return make_float3(v.x / len, v.y / len, v.z / len);
        }

        __global__ void generate_kernel(record::RayRecord* out, const RayGenConfig cfg) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = cfg.width * cfg.height;
            if (idx >= total) return;

            const std::uint32_t px = idx % cfg.width;
            const std::uint32_t py = idx / cfg.width;

            const float x = (static_cast<float>(px) + 0.5f - cfg.cx) / cfg.fx;
            const float y = (static_cast<float>(py) + 0.5f - cfg.cy) / cfg.fy;
            const float3 dir_cam = normalize3(make_float3(x, y, 1.0f));

            const float3 dir_world = make_float3(
                cfg.c2w.c2w[0] * dir_cam.x + cfg.c2w.c2w[1] * dir_cam.y + cfg.c2w.c2w[2] * dir_cam.z,
                cfg.c2w.c2w[4] * dir_cam.x + cfg.c2w.c2w[5] * dir_cam.y + cfg.c2w.c2w[6] * dir_cam.z,
                cfg.c2w.c2w[8] * dir_cam.x + cfg.c2w.c2w[9] * dir_cam.y + cfg.c2w.c2w[10] * dir_cam.z);

            const float3 origin = make_float3(cfg.c2w.c2w[3], cfg.c2w.c2w[7], cfg.c2w.c2w[11]);

            record::RayRecord r{};
            r.ox = origin.x;
            r.oy = origin.y;
            r.oz = origin.z;
            r.dx = dir_world.x;
            r.dy = dir_world.y;
            r.dz = dir_world.z;
            r.pixel_x = px;
            r.pixel_y = py;
            r.flags   = 0;

            out[idx] = r;
        }
    } // namespace

    void generate_rays_cuda(std::vector<record::RayRecord>& out, const RayGenConfig& cfg) {
        const std::size_t count = static_cast<std::size_t>(cfg.width) * cfg.height;
        out.clear();
        if (count == 0) return;
        out.resize(count);

        DeviceBuffer device{};
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device.ptr), count * sizeof(record::RayRecord)),
                   "cudaMalloc rays");

        const dim3 block(256);
        const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
        generate_kernel<<<grid, block>>>(device.ptr, cfg);
        cuda_check(cudaGetLastError(), "generate_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "generate_kernel sync");

        cuda_check(cudaMemcpy(out.data(), device.ptr, count * sizeof(record::RayRecord),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy rays");
    }
} // namespace raygen
