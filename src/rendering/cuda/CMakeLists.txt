# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# CUDA rasterization backend for rendering module
set(CUDA_RASTERIZER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
)

# Create internal CUDA rasterization library
add_library(rendering_cuda_backend STATIC ${CUDA_RASTERIZER_SOURCES})

set_target_properties(rendering_cuda_backend PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(rendering_cuda_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(rendering_cuda_backend
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${TORCH_LIBRARIES}
)

target_compile_options(rendering_cuda_backend PRIVATE
        # CUDA
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

        # C++ MSVC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

        # C++ non-MSVC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(rendering_cuda_backend PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(rendering_cuda_backend PRIVATE RELEASE_BUILD)
endif()
