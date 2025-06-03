cmake_minimum_required(VERSION 3.24...3.30)

# Ensure we have access to parent project settings
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Get torch includes from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

# ==============================================================================
# GSPLAT CUDA KERNELS - Compiled with nvcc (only actual CUDA kernels)
# ==============================================================================

set(GSPLAT_CUDA_SOURCES
        # CUDA kernel implementations only
        AdamCUDA.cu
        IntersectTile.cu
        NullCUDA.cu
        ProjectionEWA3DGSFused.cu
        ProjectionEWA3DGSPacked.cu
        ProjectionEWASimple.cu
        ProjectionUT3DGSFused.cu
        QuatScaleToCovarCUDA.cu
        RasterizeToIndices3DGS.cu
        RasterizeToPixels3DGSBwd.cu
        RasterizeToPixels3DGSFwd.cu
        RasterizeToPixelsFromWorld3DGSBwd.cu
        RasterizeToPixelsFromWorld3DGSFwd.cu
        RelocationCUDA.cu
        SphericalHarmonicsCUDA.cu
        # 2DGS CUDA files
        Projection2DGSFused.cu
        Projection2DGSPacked.cu
        RasterizeToIndices2DGS.cu
        RasterizeToPixels2DGSBwd.cu
        RasterizeToPixels2DGSFwd.cu
)

# Create CUDA kernels library
add_library(gsplat_cuda STATIC ${GSPLAT_CUDA_SOURCES})

set_target_properties(gsplat_cuda PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gsplat_cuda
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        PRIVATE
        ${TORCH_INCLUDE_DIRS}  # Add torch includes for ATen headers
)

# CUDA kernels need CUDA runtime and torch (for ATen types)
target_link_libraries(gsplat_cuda
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        PRIVATE
        ${TORCH_LIBRARIES}  # Link torch privately for CUDA kernels
)

# CUDA optimization flags
target_compile_options(gsplat_cuda PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>
)

# Debug definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_cuda PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_cuda PRIVATE RELEASE_BUILD)
endif()

# ==============================================================================
# GSPLAT C++ WRAPPERS - Compiled with g++ (fast!)
# ==============================================================================

set(GSPLAT_CPP_SOURCES
        # C++ wrapper implementations with torch dependencies
        Adam.cpp
        Intersect.cpp
        Null.cpp
        Projection.cpp
        QuatScaleToCovar.cpp
        Rasterization.cpp
        Relocation.cpp
        SphericalHarmonics.cpp
)

# Create C++ wrappers library
add_library(gsplat_cpp STATIC ${GSPLAT_CPP_SOURCES})

target_include_directories(gsplat_cpp
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        PRIVATE
        ${Python3_INCLUDE_DIRS}
)

# C++ wrappers need torch
target_link_libraries(gsplat_cpp
        PUBLIC
        ${TORCH_LIBRARIES}
        gsplat_cuda  # Link to CUDA kernels
)

# Fast C++ compilation
target_compile_options(gsplat_cpp PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3 -march=native>
)

# Debug definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_cpp PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_cpp PRIVATE RELEASE_BUILD)
endif()

# ==============================================================================
# GSPLAT BACKEND - Interface library that combines both
# ==============================================================================

add_library(gsplat_backend INTERFACE)

target_link_libraries(gsplat_backend
        INTERFACE
        gsplat_cpp
        gsplat_cuda
)

# Export the interface include directories
target_include_directories(gsplat_backend
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

# For INTERFACE libraries, we can only use INTERFACE properties
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend INTERFACE DEBUG_BUILD _DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_backend INTERFACE RELEASE_BUILD)
endif()