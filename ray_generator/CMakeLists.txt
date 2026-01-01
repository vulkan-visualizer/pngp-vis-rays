cmake_minimum_required(VERSION 3.24)
project(ray_generator LANGUAGES CXX CUDA)

# ============================================================================
# C++17 + CUDA configuration.
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA archs")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(ray_generator
        src/main.cpp
        src/ray_generator.cu
)

target_include_directories(ray_generator PRIVATE include)
set_target_properties(ray_generator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
