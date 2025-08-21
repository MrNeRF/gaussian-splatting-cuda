cmake_minimum_required(VERSION 3.24...3.30)

# TODO: This file is a bit of a mess and needs to be cleaned up.

# Get torch from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

# All fastgs sources together
set(FASTGS_SOURCES
        # rasterizer
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/backward.cu
        # optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam.cu
)

# One unified library
add_library(fastgs_backend STATIC ${FASTGS_SOURCES})

set_target_properties(fastgs_backend PROPERTIES
        CUDA_ARCHITECTURES ${gaussian_splatting_cuda_CUDA_ARCH}
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(fastgs_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        PRIVATE
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(fastgs_backend
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${TORCH_LIBRARIES}
)

# Compile options for both CUDA and C++
target_compile_options(fastgs_backend PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=20012,186>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(fastgs_backend PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(fastgs_backend PRIVATE RELEASE_BUILD)
endif()
