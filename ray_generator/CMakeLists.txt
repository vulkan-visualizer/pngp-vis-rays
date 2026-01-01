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

add_executable(ray_generator_v2
        src/main_v2.cpp
        src/ray_generator.cu
)
target_include_directories(ray_generator_v2 PRIVATE include)
set_target_properties(ray_generator_v2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(record_schema_v2_layout
        tests/record_schema_v2_layout.cpp
)
target_include_directories(record_schema_v2_layout PRIVATE include)

add_executable(record_validate_v2
        src/record_validate_v2.cpp
)
target_include_directories(record_validate_v2 PRIVATE include)
