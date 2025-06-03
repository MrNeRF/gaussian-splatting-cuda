# gsplat backend library
cmake_minimum_required(VERSION 3.24)

# Ensure we have access to parent project settings
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Collect all source files
set(GSPLAT_SOURCES
    Adam.cpp
    AdamCUDA.cu
    Intersect.cpp
    IntersectTile.cu
    Null.cpp
    NullCUDA.cu
    Projection.cpp
    ProjectionEWA3DGSFused.cu
    ProjectionEWA3DGSPacked.cu
    ProjectionEWASimple.cu
    ProjectionUT3DGSFused.cu
    QuatScaleToCovar.cpp
    QuatScaleToCovarCUDA.cu
    Rasterization.cpp
    RasterizeToIndices3DGS.cu
    RasterizeToPixels3DGSBwd.cu
    RasterizeToPixels3DGSFwd.cu
    RasterizeToPixelsFromWorld3DGSBwd.cu
    RasterizeToPixelsFromWorld3DGSFwd.cu
    Relocation.cpp
    RelocationCUDA.cu
    SphericalHarmonics.cpp
    SphericalHarmonicsCUDA.cu
    # 2DGS files if needed
    Projection2DGSFused.cu
    Projection2DGSPacked.cu
    RasterizeToIndices2DGS.cu
    RasterizeToPixels2DGSBwd.cu
    RasterizeToPixels2DGSFwd.cu
)

# Create the gsplat backend library
add_library(gsplat_backend STATIC ${GSPLAT_SOURCES})

# Set CUDA properties
set_target_properties(gsplat_backend PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(gsplat_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(gsplat_backend
    PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${TORCH_LIBRARIES}
)

# Compile options - only apply to CUDA files
target_compile_options(gsplat_backend PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

# Debug definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend PRIVATE _DEBUG)
endif()

# Export build configuration if needed by parent project
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend PRIVATE DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_backend PRIVATE RELEASE_BUILD)
endif()
